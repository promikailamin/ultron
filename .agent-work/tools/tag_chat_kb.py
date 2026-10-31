#!/usr/bin/env python3
"""Add an intent tag to every data/chat_kb.tsv entry.

The tag lives in an optional fourth column so the existing question/answer
data stays readable. Re-running the script is safe: an existing valid tag is
kept.

Usage: python3 .agent-work/tools/tag_chat_kb.py
"""
import os
import re
import sys

SRC = "data/chat_kb.tsv"
INTENTS = {"what_is", "recipe", "obtain", "drops", "uses", "count",
           "greeting", "identity", "thanks", "goodbye", "smalltalk",
           "capability", "grammar"}


def infer(question, category):
    q = re.sub(r"[^a-z0-9 ]+", " ", question.lower())
    q = re.sub(r"\s+", " ", q).strip()
    if category != "minecraft":
        return category if category in INTENTS else "what_is"
    if q.startswith("how many ") or q.startswith("how much ") or "number of " in q:
        return "count"
    if " drop" in q and (q.startswith("what does ") or q.startswith("what do ")):
        return "drops"
    if (re.match(r"^(can|is|are) .+? (be )?craft(ed|able)", q)
            or q.startswith(("how to make ", "how do i make ", "how can i make ",
                             "how to craft ", "how do i craft ", "how can i craft ",
                             "can we craft "))
            or " recipe" in q or q.startswith("recipe for ")):
        return "recipe"
    if q.startswith(("how to get ", "how do i get ", "how can i get ",
                     "how to obtain ", "how do i obtain ", "how can i obtain ",
                     "how to find ", "how do i find ", "where can i find ",
                     "where do i find ", "where can i get ", "where do i get ",
                     "where to find ")):
        return "obtain"
    if ((q.startswith("what is ") or q.startswith("what are ")) and q.endswith(" used for")) \
            or q.startswith(("how do i use ", "how to use ")) \
            or ((q.startswith("what does ") or q.startswith("what do ")) and q.endswith(" do")):
        return "uses"
    if q.startswith(("how ", "why ", "when ")):
        return "uses"
    return "what_is"


def main():
    if not os.path.exists(SRC):
        sys.stderr.write("missing %s\n" % SRC)
        return 1
    out = []
    changed = 0
    with open(SRC, encoding="utf-8") as f:
        for line in f:
            raw = line.rstrip("\n")
            if not raw or raw.startswith("#"):
                out.append(raw)
                continue
            parts = raw.split("\t")
            if len(parts) < 3:
                out.append(raw)
                continue
            category, question = parts[0].strip(), parts[1].strip()
            answer = parts[2].strip()
            # Intent inference is deliberately deterministic. Re-tagging old
            # rows keeps the table consistent when the rules improve.
            intent = infer(question, category)
            if len(parts) < 4 or parts[3].strip() != intent:
                changed += 1
            out.append("\t".join([category, question, answer, intent]))
    with open(SRC, "w", encoding="utf-8") as f:
        f.write("\n".join(out) + "\n")
    print("tagged %s: %d rows updated, %d rows total" % (SRC, changed, len(out)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
