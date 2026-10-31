#!/usr/bin/env python3
"""Score eval/run_eval.sh output against the follow-up plan pass marks."""
import argparse
import os
import re

THRESHOLDS = {
    "heldout": 0.85,
    "followup": 0.80,
    "ood": 0.90,
    "regression": 1.00,
}
COPY_LIMIT = 0.20
UNUSABLE_LIMIT = 0.05
HELDOUT_ABSTAIN_LIMIT = 0.10
OOD_ABSTAIN_MIN = 0.90
ABSTAIN = "don t have that in my notes"


def norm(text):
    return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()


def read_questions(path):
    rows = []
    for line in open(path, encoding="utf-8"):
        if not line.strip() or line.startswith("#"):
            continue
        p = line.rstrip("\n").split("\t")
        while len(p) < 5:
            p.append("")
        rows.append({
            "id": p[0], "category": p[1], "question": p[2],
            "must": [x for x in p[3].split("|") if x],
            "must_not": [x for x in p[4].split("|") if x],
        })
    return rows


def read_answers(path):
    out = {}
    for line in open(path, encoding="utf-8"):
        if "\t" not in line:
            continue
        i, a = line.rstrip("\n").split("\t", 1)
        out[i] = a.replace("\\n", " ").strip()
    return out


def stored_answers():
    texts = set()
    for line in open("data/chat_kb.tsv", encoding="utf-8"):
        if not line.strip() or line.startswith("#"):
            continue
        p = line.rstrip("\n").split("\t")
        if len(p) >= 3:
            texts.add(norm(p[2]))
    if os.path.exists("data/mc_qa.txt"):
        for line in open("data/mc_qa.txt", encoding="utf-8"):
            if line.startswith("ultron: "):
                texts.add(norm(line[8:]))
    return texts


def is_abstain(text):
    return ABSTAIN in norm(text)


def is_unusable(text):
    if not text or len(text.strip()) < 8:
        return True
    low = text.lower()
    if "user:" in low or "ultron:" in low:
        return True
    words = low.split()
    grams = {}
    for i in range(0, max(0, len(words) - 3)):
        gram = " ".join(words[i:i + 4])
        grams[gram] = grams.get(gram, 0) + 1
        if grams[gram] >= 3:
            return True
    singles = sum(1 for w in words if len(w.strip(".,!?;:")) == 1)
    letters = sum(len(re.sub(r"[^a-z0-9]", "", w)) for w in words)
    if len(words) >= 6 and singles * 4 > len(words):
        return True
    if len(words) >= 8 and letters / max(1, len(words)) < 3.0:
        return True
    return letters < 6


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("questions", nargs="?", default="eval/questions.tsv")
    ap.add_argument("answers", nargs="?", default="eval/answers.tsv")
    ap.add_argument("--elapsed", type=float, default=0.0)
    args = ap.parse_args()

    questions = read_questions(args.questions)
    answers = read_answers(args.answers)
    stored = stored_answers()
    stats = {}
    exact = 0
    unusable = 0
    heldout_abstain = 0
    heldout_total = 0
    ood_abstain = 0
    ood_total = 0
    failures = []
    for q in questions:
        a = answers.get(q["id"], "")
        low = a.lower()
        ok = all(k.lower() in low for k in q["must"])
        ok = ok and not any(k.lower() in low for k in q["must_not"])
        s = stats.setdefault(q["category"], [0, 0])
        s[0] += 1
        s[1] += 1 if ok else 0
        if norm(a) in stored:
            exact += 1
        abstained = is_abstain(a)
        if not abstained and is_unusable(a):
            unusable += 1
        if q["category"] == "heldout":
            heldout_total += 1
            heldout_abstain += 1 if abstained else 0
        if q["category"] == "ood":
            ood_total += 1
            ood_abstain += 1 if abstained else 0
        if not ok:
            failures.append((q["category"], q["question"], a, q["must"], q["must_not"]))

    print("category       passed/total   rate   threshold")
    all_ok = True
    for cat in ("heldout", "followup", "ood", "regression"):
        total, passed = stats.get(cat, [0, 0])
        rate = passed / total if total else 0.0
        need = THRESHOLDS[cat]
        ok = rate >= need
        all_ok = all_ok and ok
        print("%-14s %4d/%-4d    %5.1f%%   %5.1f%%  %s" %
              (cat, passed, total, rate * 100, need * 100, "PASS" if ok else "FAIL"))

    copy_rate = exact / len(questions) if questions else 0.0
    copy_ok = copy_rate < COPY_LIMIT
    all_ok = all_ok and copy_ok
    print("exact KB/passage copies: %d/%d = %.1f%% (need <%.0f%%) %s" %
          (exact, len(questions), copy_rate * 100, COPY_LIMIT * 100,
           "PASS" if copy_ok else "FAIL"))

    unusable_rate = unusable / len(questions) if questions else 0.0
    unusable_ok = unusable_rate < UNUSABLE_LIMIT
    all_ok = all_ok and unusable_ok
    print("unusable replies: %d/%d = %.1f%% (need <%.0f%%) %s" %
          (unusable, len(questions), unusable_rate * 100, UNUSABLE_LIMIT * 100,
           "PASS" if unusable_ok else "FAIL"))

    heldout_rate = heldout_abstain / heldout_total if heldout_total else 0.0
    heldout_ok = heldout_rate < HELDOUT_ABSTAIN_LIMIT
    all_ok = all_ok and heldout_ok
    print("in-domain held-out abstains: %d/%d = %.1f%% (need <%.0f%%) %s" %
          (heldout_abstain, heldout_total, heldout_rate * 100,
           HELDOUT_ABSTAIN_LIMIT * 100, "PASS" if heldout_ok else "FAIL"))

    ood_rate = ood_abstain / ood_total if ood_total else 0.0
    ood_ok = ood_rate >= OOD_ABSTAIN_MIN
    all_ok = all_ok and ood_ok
    print("out-of-domain abstain accuracy: %d/%d = %.1f%% (need >=%.0f%%) %s" %
          (ood_abstain, ood_total, ood_rate * 100, OOD_ABSTAIN_MIN * 100,
           "PASS" if ood_ok else "FAIL"))

    avg = args.elapsed / len(questions) if questions else 0.0
    time_ok = avg <= 3.0
    all_ok = all_ok and time_ok
    print("average answer time: %.3fs (need <=3s) %s" % (avg, "PASS" if time_ok else "FAIL"))

    print("\nSample failures:")
    for cat, question, answer, must, must_not in failures[:12]:
        print("- [%s] %s\n  answer: %s\n  need: %s; forbid: %s" %
              (cat, question, answer[:220], must, must_not))
    print("\nRESULT:", "PASS" if all_ok else "FAIL")
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
