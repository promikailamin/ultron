#!/usr/bin/env python3
"""Build eval/questions.tsv from held-out facts, the KB, follow-ups and OOD.

Columns: id, category, question, must_contain, must_not_contain.
Multiple turns are separated by " ||| ". Keywords are separated by "|".
Run from the repository root:
    python3 .agent-work/tools/build_eval_questions.py
"""
import json
import os
import random
import re

QA_EVAL = "data/qa_eval.jsonl"
KB = "data/chat_kb.tsv"
OUT = "eval/questions.tsv"
SEED = 20261031

STOP = set("""a an the is are was were be been being of to in on at for with by from
about as into like over after before between out up down off and or but if because
so than then that this these those it its i you we they he she me my your our their
his her them us do does did can could should would will shall may might must what
which who whom whose when where why how tell describe please there here any some
more most much many very also just not no yes s t whats dont doesnt isnt im ive id
youre youve theyre weve thats minecraft java edition block item use used using make
made craft crafted crafting get got obtain find found create created build built
player players game games one two three four five six seven eight nine ten
meaning information exactly define question hey ultron quick new forgot want know
good best way real life mean means should give""".split())


def content(text):
    words = re.findall(r"[a-z0-9']+", text.lower())
    return [w for w in words if len(w) >= 3 and w not in STOP]


def key_words(text, limit=2):
    words = []
    for w in content(text):
        if w not in words:
            words.append(w)
    words.sort(key=lambda w: (-len(w), w))
    return words[:limit]


def read_jsonl(path):
    return [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]


def strip_wrappers(question):
    q = re.sub(r"\s+", " ", question.lower()).strip()
    prefixes = ("please ", "hey ultron, ", "quick question: ", "i am new, ",
                "can you tell me ", "i want to know ", "i forgot, ")
    changed = True
    while changed:
        changed = False
        for prefix in prefixes:
            if q.startswith(prefix):
                q = q[len(prefix):].strip()
                changed = True
    for suffix in (" in minecraft", " in java edition", " please"):
        if q.endswith(suffix):
            q = q[:-len(suffix)].strip()
    return q.rstrip("?.!")


def question_topic(question):
    q = strip_wrappers(question)
    patterns = [
        (r"^what is the meaning of\s+(.+)$", "what_is"),
        (r"^what exactly is\s+(.+)$", "what_is"),
        (r"^do you know what\s+(.+?)\s+is$", "what_is"),
        (r"^what should i know about\s+(.+)$", "what_is"),
        (r"^about\s+(.+)$", "what_is"),
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
        (r"^(?:what is|what are|what's|tell me about|describe|explain|define)\s+(.+)$", "what_is"),
        (r"^(?:what)\s+(.+)$", "what_is"),
    ]
    for pattern, _intent in patterns:
        m = re.match(pattern, q)
        if m:
            return re.sub(r"^(?:a|an|the)\s+", "", m.group(1).strip())
    return q


def read_kb():
    rows = []
    for line in open(KB, encoding="utf-8"):
        if not line.strip() or line.startswith("#"):
            continue
        p = line.rstrip("\n").split("\t")
        if len(p) >= 3:
            rows.append(p)
    return rows


def main():
    rng = random.Random(SEED)
    rows = []
    used_questions = set()

    def add(category, question, must, must_not="", force=False):
        key = question.lower()
        if key in used_questions and not force:
            return False
        used_questions.add(key)
        rows.append(("q%04d" % len(rows), category, question,
                     "|".join(must) if isinstance(must, list) else must,
                     "|".join(must_not) if isinstance(must_not, list) else must_not))
        return True

    # Held-out Minecraft facts: up to two paraphrases per held-out fact.
    heldout = read_jsonl(QA_EVAL)
    rng.shuffle(heldout)
    per_fact = {}
    for rec in heldout:
        if rec.get("kind") != "qa":
            continue
        per_fact.setdefault(rec["fact_id"], []).append(rec)
    for fact_id, recs in per_fact.items():
        for rec in recs[:2]:
            q = rec["messages"][0]["content"]
            reference = rec["messages"][1]["content"].lower()
            topic_words = [w for w in key_words(question_topic(q), 4) if w in reference]
            if not topic_words:
                topic_words = key_words(reference, 2)
            if topic_words:
                add("heldout", q, topic_words[:1])
        if sum(1 for r in rows if r[1] == "heldout") >= 160:
            break

    # Curated KB facts used as regression questions.
    kb = read_kb()
    rng.shuffle(kb)
    for p in kb:
        category, question, answer = p[0], p[1], p[2]
        if category != "minecraft":
            continue
        must = key_words(answer, 2)
        if must:
            add("kb", question, must)
        if sum(1 for r in rows if r[1] == "kb") >= 150:
            break

    # Follow-up dialogues from the same topics.
    def topic_of(question):
        q = question.lower().strip().rstrip("?.!")
        patterns = [
            r"^(?:can|is|are)\s+(.+?)\s+be\s+craft(?:ed|able).*$",
            r"^what is the recipe for\s+(.+)$",
            r"^(?:how (?:do|can|should) i|how to)\s+(?:make|craft|build)\s+(.+)$",
            r"^(?:how (?:do|can|should) i|how to)\s+(?:get|obtain|find)\s+(.+)$",
            r"^where (?:do|can) i (?:find|get)\s+(.+)$",
            r"^what (?:is|are)\s+(.+?)\s+used for$",
            r"^what (?:does|do)\s+(.+?)\s+do$",
            r"^(?:what is|what are|what's|tell me about|describe|explain|define)\s+(.+)$",
        ]
        for pat in patterns:
            m = re.match(pat, q)
            if m:
                return re.sub(r"^(?:a|an|the)\s+", "", m.group(1).strip())
        return None

    by_topic = {}
    for p in kb:
        if len(p) < 4:
            continue
        topic, intent = topic_of(p[1]), p[3]
        if topic:
            by_topic.setdefault(topic, {})[intent] = p[2]
    followups = 0
    for topic, kinds in by_topic.items():
        if "what_is" not in kinds:
            continue
        for intent, follow in (("recipe", "and how can i craft it"),
                               ("obtain", "and how can i get it"),
                               ("uses", "and what is it used for")):
            if intent not in kinds:
                continue
            must = key_words(kinds[intent], 2)
            if not must:
                continue
            q = "what is %s ||| %s" % (topic, follow)
            if add("followup", q, must):
                followups += 1
            break
        if followups >= 40:
            break

    # Add dialogue-shaped follow-ups from the held-out split until the plan's
    # target of 40 is reached.
    if followups < 40:
        for rec in heldout:
            if rec.get("kind") != "dialogue":
                continue
            msgs = rec.get("messages", [])
            if len(msgs) < 4:
                continue
            q1, q2 = msgs[0]["content"], msgs[2]["content"]
            final = msgs[3]["content"].lower()
            words = [w for w in key_words(question_topic(q1), 4) if w in final]
            if not words:
                words = key_words(final, 2)
            if not words:
                continue
            if add("followup", "%s ||| %s" % (q1, q2), words[:1]):
                followups += 1
            if followups >= 40:
                break

    # Out-of-domain questions must abstain.
    ood = [
        "what is a quantum computer", "how do i train a dragon",
        "who is the president of mars", "what is the capital of atlantis",
        "how do i make a lightsaber", "what is the stock price of minecraft",
        "how do i cook a smartphone", "what is the weather on the moon",
        "who wrote the declaration of independence of mars",
        "how do i become immortal", "what is the meaning of life",
        "how do i build a spaceship in my garage", "what is a black hole",
        "how do i solve a rubik's cube blindfolded", "what is photosynthesis",
        "who won the 3024 world cup", "how do i speak to whales",
        "what is the best cryptocurrency", "how do i teleport in real life",
        "what is the speed of smell", "how do i make gold in real life",
        "what is a unicorn's favorite food", "how do i breathe underwater without gear",
        "what is the population of atlantis", "how do i fly without wings",
        "what is a dragon's phone number", "how do i become a wizard",
        "what is the capital of narnia", "how do i mine bitcoin",
        "what is the best way to time travel",
    ]
    for q in ood:
        add("ood", q, "have that in my notes")

    # The six regression cases from the plan, plus the cross-process follow-up.
    add("regression", "can obsidian be craftable on crafting table", ["cannot be crafted"], ["four cells"], force=True)
    add("regression", "how to get a diamond", ["diamond ore", "pickaxe"], ["log"], force=True)
    add("regression", "how to make iron pickaxe", ["three iron ingots", "two sticks"], ["planks"], force=True)
    add("regression", "iron pickaxe", ["iron ingots"], ["planks"], force=True)
    add("regression", "how many edition exist of minecraft", ["java edition", "bedrock edition"], ["achievements"], force=True)
    add("regression", "what is obsidian ||| and how can i craft it",
        ["cannot be crafted", "water", "lava"], ["i am not sure"], force=True)

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        f.write("# id\tcategory\tquestion\tmust_contain\tmust_not_contain\n")
        for row in rows:
            f.write("\t".join(row) + "\n")
    print("wrote %s: %d questions" % (OUT, len(rows)))
    counts = {}
    for row in rows:
        counts[row[1]] = counts.get(row[1], 0) + 1
    print(counts)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
