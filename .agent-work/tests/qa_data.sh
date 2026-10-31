#!/usr/bin/env bash
# Phase 2 data validation: regenerate the corpus and check its invariants.
set -euo pipefail

cd "$(dirname "$0")/../.."
python3 .agent-work/tools/gen_qa.py

python3 - <<'PY'
import json
import sys

train = [json.loads(l) for l in open("data/qa_train.jsonl", encoding="utf-8")]
evals = [json.loads(l) for l in open("data/qa_eval.jsonl", encoding="utf-8")]
train_ids = {r["fact_id"] for r in train}
eval_ids = {r["fact_id"] for r in evals}
assert len(train) >= 50000, len(train)
assert len(evals) >= 2500, len(evals)
assert not (train_ids & eval_ids), "fact leaked across the split"
assert any(r["kind"] == "dialogue" for r in train)
assert any(r["kind"] == "retrieval" for r in train)
assert any(r["kind"] == "grammar" for r in train)
assert any(r.get("context") for r in train)
assert any(m["content"] == "I don't have that in my notes."
           for r in train for m in r["messages"] if m["role"] == "assistant")
seen = set()
for r in train:
    key = json.dumps(r, sort_keys=True)
    assert key not in seen, "duplicate record"
    seen.add(key)
    assert r["messages"] and r["messages"][0]["role"] == "user"
    assert r["messages"][-1]["role"] == "assistant"
    for m in r["messages"]:
        assert m["content"].strip(), "empty message"
        assert "In Minecraft, in Minecraft" not in m["content"]

words = sum(len(l.split()) for l in open("data/pretrain.txt", encoding="utf-8"))
assert words >= 5000000, words
print("train=%d eval=%d facts=%d pretrain_words=%d" %
      (len(train), len(evals), len(train_ids | eval_ids), words))
PY

echo "PHASE 2 DATA TESTS PASSED"
