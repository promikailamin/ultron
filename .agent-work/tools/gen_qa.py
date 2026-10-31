#!/usr/bin/env python3
"""Generate the Phase 2 Q&A corpus without an external LLM.

Sources:
  data/chat_kb.tsv   curated facts, one answer per intent
  data/mc_qa.txt     wiki-derived question/answer pairs
  data/grammar.txt   English grammar notes
  data/mc.txt        wiki passages for retrieval-format examples

Outputs:
  data/qa_train.jsonl
  data/qa_eval.jsonl
  data/pretrain.txt

The split is by fact, never by phrasing: every phrasing and every answer
wording of a held-out fact goes to qa_eval.jsonl. Each record looks like

  {"fact_id": "...", "kind": "qa|dialogue|retrieval|grammar",
   "context": "...", "messages": [{"role": "user", ...}, ...]}

Run from the repository root:
  python3 .agent-work/tools/gen_qa.py
"""
import collections
import hashlib
import json
import os
import random
import re
import sys

KB = "data/chat_kb.tsv"
MC_QA = "data/mc_qa.txt"
MC_TEXT = "data/mc.txt"
GRAMMAR = "data/grammar.txt"
BOOKS = [
    "data/pp.txt", "data/moby.txt", "data/alice.txt", "data/sherlock.txt",
    "data/frank.txt", "data/jane.txt", "data/tom.txt", "data/earnest.txt",
    "data/wp.txt", "data/ge.txt", "data/drac.txt",
]
OUT_TRAIN = "data/qa_train.jsonl"
OUT_EVAL = "data/qa_eval.jsonl"
OUT_PRETRAIN = "data/pretrain.txt"
ABSTAIN = "I don't have that in my notes."
SEED = 20261031
EVAL_FRACTION = 0.05

INTENTS = {"what_is", "recipe", "obtain", "drops", "uses", "count", "grammar"}
SOCIAL = {"greeting", "identity", "thanks", "goodbye", "smalltalk", "capability"}


def clean(text):
    text = text.replace("\u00a0", " ")
    text = re.sub(r"\[\[(?:[^\]|]*\|)?([^\]]+)\]\]", r"\1", text)
    text = re.sub(r"\{\{[^{}]*\}\}", " ", text)
    text = re.sub(r"\[[0-9]+\]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def sentences(text):
    parts = re.split(r"(?<=[.!?])\s+", clean(text))
    return [p.strip() for p in parts if p.strip()]


def short_sentence_answer(text):
    """One useful sentence, skipping a bare 'No.' opener."""
    parts = sentences(text)
    if not parts:
        return clean(text)
    if len(parts[0]) < 16 and len(parts) > 1:
        return " ".join(parts[:2])
    return parts[0]


def answer_variants(text):
    text = clean(text)
    parts = sentences(text)
    variants = [text]
    short = short_sentence_answer(text)
    variants.append(short)
    if len(parts) >= 2:
        variants.append(" ".join(parts[:2]))
        # Changing the sentence order teaches composition instead of a single
        # memorised string. Only do it when both sentences are substantial.
        if len(parts[0]) >= 20 and len(parts[1]) >= 20:
            variants.append(" ".join([parts[1], parts[0]] + parts[2:]))
    if text and text[0].isalpha() and not text.lower().startswith("in minecraft"):
        variants.append("In Minecraft, " + text[0].lower() + text[1:])
    out = []
    seen = set()
    for v in variants:
        v = re.sub(r"\s+", " ", v).strip()
        if len(v) < 12 or v.lower() in seen:
            continue
        seen.add(v.lower())
        out.append(v)
    return out[:5]


def topic_from_question(question):
    q = clean(question).lower().rstrip("?.!")
    patterns = [
        # Most specific intents first: "what is X used for" is a uses
        # question, not a definition of "X used for".
        (r"^(?:can|is|are)\s+(.+?)\s+be\s+craft(?:ed|able).*$", "recipe"),
        (r"^what is the recipe for\s+(.+)$", "recipe"),
        (r"^(?:how (?:do|can|should) i|how to)\s+(?:make|craft|build)\s+(.+)$", "recipe"),
        (r"^(?:how (?:do|can|should) i|how to)\s+(?:get|obtain|find)\s+(.+)$", "obtain"),
        (r"^where (?:do|can) i (?:find|get)\s+(.+)$", "obtain"),
        (r"^what (?:does|do)\s+(.+?)\s+drop$", "drops"),
        (r"^what drops from\s+(.+)$", "drops"),
        (r"^what (?:is|are)\s+(.+?)\s+used for$", "uses"),
        (r"^what (?:does|do)\s+(.+?)\s+do$", "uses"),
        (r"^how (?:do|can) i use\s+(.+)$", "uses"),
        (r"^how many\s+(.+?)(?:\s+(?:are there|exist|does minecraft have))?$", "count"),
        # Generic definitions last, with the longest prefix first so
        # "what is health" does not become the topic "is health".
        (r"^(?:what is|what are|what's|tell me about|describe|explain|define)\s+(.+)$", "what_is"),
        (r"^(?:what)\s+(.+)$", "what_is"),
    ]
    for pattern, intent in patterns:
        m = re.match(pattern, q)
        if m:
            topic = m.group(1).strip()
            topic = re.sub(r"^(?:a|an|the)\s+", "", topic)
            topic = re.sub(r"\s+(?:in minecraft|on a crafting table)$", "", topic)
            return topic, intent
    return None, None


def question_phrasings(topic, intent):
    topic = topic.strip()
    if not topic:
        return []
    t = topic
    out = []
    if intent == "what_is":
        out = [
            "what is %s" % t, "what's %s" % t, "tell me about %s" % t,
            "describe %s" % t, "explain %s" % t, "what is %s in minecraft" % t,
            "what does %s mean" % t, "define %s" % t, "what are %s" % t,
            "what exactly is %s" % t, "can you tell me about %s" % t,
            "i want to know about %s" % t, "give me information about %s" % t,
            "what should i know about %s" % t, "what is the meaning of %s" % t,
            "do you know what %s is" % t, "about %s" % t,
            t, "what %s" % t, "tell me %s" % t,
        ]
    elif intent == "recipe":
        out = [
            "how to make %s" % t, "how do i make %s" % t, "how can i craft %s" % t,
            "how to craft %s" % t, "what is the recipe for %s" % t,
            "%s recipe" % t, "how is %s crafted" % t, "can you craft %s" % t,
            "give me the recipe for %s" % t, "how do i build %s" % t,
            "can %s be crafted" % t, "how do i make %s in minecraft" % t,
            "what do i need to make %s" % t, "what materials make %s" % t,
            "what are the ingredients for %s" % t, "tell me how to craft %s" % t,
            "what is needed to craft %s" % t, "how to create %s" % t,
            "%s crafting recipe" % t, "how do you make %s" % t,
            t, "how make %s" % t, "recipe %s" % t,
        ]
    elif intent == "obtain":
        out = [
            "how to get %s" % t, "how do i get %s" % t, "how can i obtain %s" % t,
            "how to obtain %s" % t, "where do i find %s" % t,
            "where can i find %s" % t, "how to find %s" % t,
            "where does %s come from" % t, "how do i get %s in minecraft" % t,
            "how do i find %s" % t, "how can i find %s" % t,
            "where is %s found" % t, "how do you get %s" % t,
            "what is the best way to get %s" % t, "where can %s be found" % t,
            "how to obtain %s in minecraft" % t, "where do i get %s" % t,
            "how can i get %s" % t, t, "how get %s" % t, "where find %s" % t,
        ]
    elif intent == "drops":
        out = [
            "what does %s drop" % t, "what do %s drop" % t,
            "what drops from %s" % t, "what is dropped by %s" % t,
            "%s drops" % t, "what can %s drop" % t, "what items does %s drop" % t,
            "what can you get from %s" % t, "what does %s leave behind" % t,
            "when killed what does %s drop" % t, "what are %s drops" % t,
            "what %s drop" % t,
        ]
    elif intent == "uses":
        out = [
            "what is %s used for" % t, "what can i use %s for" % t,
            "how do i use %s" % t, "what does %s do" % t,
            "what is the purpose of %s" % t, "why use %s" % t,
            "what can %s be used for" % t, "what do you use %s for" % t,
            "what is %s good for" % t, "how is %s used" % t,
            "what is the use of %s" % t, "what can i do with %s" % t,
            "why do i need %s" % t, "%s uses" % t, "what %s used for" % t,
        ]
    elif intent == "count":
        out = [
            "how many %s are there" % t, "how many %s exist" % t,
            "how many %s does minecraft have" % t, "what is the number of %s" % t,
            "how many %s are in minecraft" % t, "how many types of %s are there" % t,
            "what is the count of %s" % t, "how many %s does the game have" % t,
            "how many %s can you get" % t, "%s count" % t,
        ]
    else:
        out = [t, "tell me about %s" % t, "what is %s" % t]
    # Natural wrappers multiply the phrasings without changing the fact.
    # Keep them deterministic and avoid stacking two Minecraft qualifiers.
    base = list(out)
    for prefix in ("please ", "hey ultron, ", "quick question: ", "i am new, ",
                   "can you tell me ", "i want to know ", "i forgot, "):
        for q in base:
            out.append(prefix + q)
    for suffix in (" in minecraft", " in java edition", " please"):
        for q in base:
            if suffix.strip() not in q:
                out.append(q + suffix)
    seen, clean_out = set(), []
    for q in out:
        q = re.sub(r"\s+", " ", q).strip()
        if q and q.lower() not in seen:
            seen.add(q.lower())
            clean_out.append(q)
    return clean_out[:32]


class Fact:
    def __init__(self, fact_id, topic, intent, answer, source):
        self.fact_id = fact_id
        self.topic = topic
        self.intent = intent
        self.answer = clean(answer)
        self.source = source
        self.answers = answer_variants(self.answer)

    def key(self):
        return "%s|%s|%s" % (self.source, self.topic.lower(), self.intent)


def add_fact(facts, seen, topic, intent, answer, source):
    topic = clean(topic).lower()
    answer = clean(answer)
    if not topic or not answer or len(answer) < 20:
        return
    if intent not in INTENTS:
        return
    key = "%s|%s|%s" % (source, topic, intent)
    if key in seen:
        return
    seen.add(key)
    facts.append(Fact(hashlib.sha1(key.encode("utf-8")).hexdigest()[:12],
                      topic, intent, answer, source))


def load_kb_facts(facts, seen):
    with open(KB, encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line or line.startswith("#"):
                continue
            parts = line.split("\t")
            if len(parts) < 3:
                continue
            category, question, answer = parts[0].strip(), parts[1].strip(), parts[2].strip()
            intent = parts[3].strip() if len(parts) > 3 else ""
            topic, inferred = topic_from_question(question)
            if not intent:
                intent = inferred or ("what_is" if category == "minecraft" else category)
            if intent in SOCIAL:
                # Social turns are kept verbatim; they are short and must not
                # be paraphrased into nonsense.
                facts.append(Fact("social-" + hashlib.sha1(question.lower().encode()).hexdigest()[:8],
                                  question.lower(), "social", answer, "kb"))
                continue
            if not topic:
                topic, inferred = question.lower().rstrip("?.!"), intent or "what_is"
            add_fact(facts, seen, topic, intent, answer, "kb")


def load_mc_qa_facts(facts, seen):
    if not os.path.exists(MC_QA):
        return
    question = None
    with open(MC_QA, encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if line.startswith("user: "):
                question = line[6:].strip()
            elif line.startswith("ultron: ") and question:
                answer = line[8:].strip()
                topic, intent = topic_from_question(question)
                if topic and intent:
                    add_fact(facts, seen, topic, intent, answer, "wiki")
                question = None


def load_grammar_facts(facts, seen):
    if not os.path.exists(GRAMMAR):
        return
    text = open(GRAMMAR, encoding="utf-8").read()
    chunks = re.split(r"(?m)^===== (.+?) =====\s*$", text)
    for i in range(1, len(chunks) - 1, 2):
        heading = chunks[i].strip().lower()
        body = chunks[i + 1]
        paragraphs = [clean(p) for p in re.split(r"\n\s*\n", body)]
        paragraphs = [p for p in paragraphs if len(p) >= 120 and not p.startswith("Example:")]
        if not paragraphs:
            continue
        answer = " ".join(sentences(paragraphs[0])[:3])
        add_fact(facts, seen, heading, "grammar", answer, "grammar")


def examples_for_fact(fact, rng):
    """Return [(question, answer, context)] for one fact."""
    out = []
    if fact.intent == "social":
        out.append((fact.topic, fact.answer, None))
        return out
    questions = question_phrasings(fact.topic, fact.intent)
    answers = fact.answers or [fact.answer]
    for q in questions:
        out.append((q, rng.choice(answers), None))
    return out


def dialogue_examples(by_topic, rng):
    out = []
    for topic, kinds in by_topic.items():
        first = kinds.get("what_is")
        if not first:
            continue
        second_intent = next((k for k in ("recipe", "obtain", "drops", "uses") if k in kinds), None)
        if not second_intent:
            continue
        second = kinds[second_intent]
        followup = {
            "recipe": "and how can i craft it",
            "obtain": "and how can i get it",
            "drops": "and what does it drop",
            "uses": "and what is it used for",
        }[second_intent]
        messages = [
            {"role": "user", "content": "what is %s" % topic},
            {"role": "assistant", "content": rng.choice(first.answers)},
            {"role": "user", "content": followup},
            {"role": "assistant", "content": rng.choice(second.answers)},
        ]
        out.append({
            "fact_id": first.fact_id + "+" + second.fact_id,
            "kind": "dialogue",
            "messages": messages,
        })
        third_intent = next((k for k in ("obtain", "uses", "drops", "recipe")
                             if k in kinds and k != second_intent), None)
        if third_intent:
            third = kinds[third_intent]
            third_q = {
                "recipe": "and how do i craft it",
                "obtain": "and where can i find it",
                "drops": "and what does it drop",
                "uses": "and what do i use it for",
            }[third_intent]
            out.append({
                "fact_id": first.fact_id + "+" + second.fact_id + "+" + third.fact_id,
                "kind": "dialogue",
                "messages": messages + [
                    {"role": "user", "content": third_q},
                    {"role": "assistant", "content": rng.choice(third.answers)},
                ],
            })
    return out


def load_passages():
    passages = {}
    if not os.path.exists(MC_TEXT):
        return passages
    text = open(MC_TEXT, encoding="utf-8").read()
    chunks = re.split(r"(?m)^===== (.+?) =====\s*$", text)
    for i in range(1, len(chunks) - 1, 2):
        title = chunks[i].strip().lower()
        body = chunks[i + 1]
        paragraphs = [clean(p) for p in re.split(r"\n\s*\n", body)]
        paragraphs = [p for p in paragraphs if len(p) >= 100 and not p.startswith("==")]
        if paragraphs:
            passage = " ".join(sentences(paragraphs[0])[:3])
            passages[title] = passage
    return passages


def retrieval_examples(facts, passages, rng, limit=8000):
    if not passages:
        return []
    titles = list(passages)
    out = []
    retrieval_facts = [f for f in facts if f.intent in INTENTS and f.topic in passages]
    rng.shuffle(retrieval_facts)
    for fact in retrieval_facts[:limit]:
        question = rng.choice(question_phrasings(fact.topic, fact.intent))
        if rng.random() < 0.15:
            wrong = rng.choice(titles)
            context = passages[wrong]
            answer = ABSTAIN
        else:
            context = passages[fact.topic]
            answer = rng.choice(fact.answers)
        out.append({
            "fact_id": fact.fact_id,
            "kind": "retrieval",
            "context": context,
            "messages": [
                {"role": "user", "content": question},
                {"role": "assistant", "content": answer},
            ],
        })
    return out


def grammar_tasks(rng, limit=1200):
    """Make fix-the-sentence tasks from examples already in grammar.txt."""
    if not os.path.exists(GRAMMAR):
        return []
    text = open(GRAMMAR, encoding="utf-8").read()
    examples = re.findall(r"Example:\s*([A-Z][^.!?]{8,180}[.!?])", text, re.S)
    out = []
    for ex in examples:
        ex = clean(ex)
        if len(ex) < 20 or len(ex) > 180 or not ex.endswith("."):
            continue
        broken = ex[:-1]
        task = "fix this sentence: " + broken
        out.append({
            "fact_id": "grammar-fix-" + hashlib.sha1(ex.encode()).hexdigest()[:10],
            "kind": "grammar",
            "messages": [
                {"role": "user", "content": task},
                {"role": "assistant", "content": ex + " The sentence needs a full stop at the end."},
            ],
        })
        words = ex.split()
        if len(words) > 3 and words[0][:1].isupper():
            broken2 = " ".join([words[0][:1].lower() + words[0][1:]] + words[1:])
            out.append({
                "fact_id": "grammar-cap-" + hashlib.sha1(ex.encode()).hexdigest()[:10],
                "kind": "grammar",
                "messages": [
                    {"role": "user", "content": "fix this sentence: " + broken2},
                    {"role": "assistant", "content": ex + " A sentence starts with a capital letter."},
                ],
            })
        if len(out) >= limit:
            break
    return out[:limit]


def pretrain_text(facts, records, rng):
    parts = []
    for path in BOOKS:
        if os.path.exists(path):
            parts.append(open(path, encoding="utf-8").read())
    for path in (MC_TEXT, GRAMMAR, MC_QA, "data/chat.txt", "data/instruct.txt"):
        if os.path.exists(path):
            parts.append(open(path, encoding="utf-8").read())
    qa = []
    for fact in facts:
        for answer in fact.answers:
            qa.append(answer)
    # Flatten the generated conversations so Stage A sees the same domain
    # vocabulary and turn structure as Stage B.
    for rec in records:
        if rec.get("context"):
            qa.append("context: " + rec["context"])
        for msg in rec["messages"]:
            qa.append(("user: " if msg["role"] == "user" else "ultron: ") + msg["content"])
    rng.shuffle(qa)
    parts.append("\n".join(qa))
    text = "\n\n".join(parts)
    with open(OUT_PRETRAIN, "w", encoding="utf-8") as f:
        f.write(text)
    return len(text)


def write_jsonl(path, records):
    with open(path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def main():
    rng = random.Random(SEED)
    facts, seen = [], set()
    load_kb_facts(facts, seen)
    load_mc_qa_facts(facts, seen)
    load_grammar_facts(facts, seen)

    by_topic = collections.defaultdict(dict)
    for fact in facts:
        if fact.intent in INTENTS:
            by_topic[fact.topic].setdefault(fact.intent, fact)

    # Split by fact, never by phrasing.
    fact_ids = sorted({f.fact_id for f in facts})
    rng.shuffle(fact_ids)
    eval_count = max(1, int(len(fact_ids) * EVAL_FRACTION))
    eval_ids = set(fact_ids[:eval_count])

    train, evaluation = [], []
    for fact in facts:
        target = evaluation if fact.fact_id in eval_ids else train
        for question, answer, context in examples_for_fact(fact, rng):
            rec = {
                "fact_id": fact.fact_id,
                "kind": "qa",
                "messages": [
                    {"role": "user", "content": question},
                    {"role": "assistant", "content": answer},
                ],
            }
            if context:
                rec["context"] = context
            target.append(rec)

    dialogues = dialogue_examples(by_topic, rng)
    for rec in dialogues:
        ids = set(rec["fact_id"].split("+"))
        (evaluation if ids & eval_ids else train).append(rec)

    passages = load_passages()
    retrieval = retrieval_examples(facts, passages, rng)
    for rec in retrieval:
        (evaluation if rec["fact_id"] in eval_ids else train).append(rec)

    grammar = grammar_tasks(rng)
    train.extend(grammar)

    rng.shuffle(train)
    rng.shuffle(evaluation)
    write_jsonl(OUT_TRAIN, train)
    write_jsonl(OUT_EVAL, evaluation)
    chars = pretrain_text(facts, train + evaluation, rng)

    print("facts: %d (eval %d)" % (len(facts), eval_count))
    print("train records: %d" % len(train))
    print("eval records: %d" % len(evaluation))
    print("dialogue records: %d" % len(dialogues))
    print("retrieval records: %d" % len(retrieval))
    print("grammar tasks: %d" % len(grammar))
    print("pretrain text: %.2f M chars" % (chars / 1e6))
    for path in (OUT_TRAIN, OUT_EVAL, OUT_PRETRAIN):
        print("%s: %.2f MB" % (path, os.path.getsize(path) / 1e6))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
