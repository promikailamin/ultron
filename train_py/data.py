#!/usr/bin/env python3
"""Dataset construction shared by train.py and export/parity tests."""
import json

EOS = 1


def conversation_prompt(record):
    """Turn a QA/dialogue/retrieval record into the text prompt the jar uses."""
    context = record.get("context")
    parts = []
    if context:
        parts.append("context: " + context)
    messages = record["messages"]
    for msg in messages[:-1]:
        role = "user" if msg["role"] == "user" else "ultron"
        parts.append("%s: %s" % (role, msg["content"]))
    parts.append("user: " + messages[-1]["content"])
    parts.append("ultron:")
    return "\n".join(parts)


def make_example(tok, record, ctx):
    """Return (ids, mask) with answer-only loss, or None if it does not fit."""
    answer = record["messages"][-1]["content"].strip()
    if not answer:
        return None
    prompt_ids = tok.encode(conversation_prompt(record))
    answer_ids = tok.encode(answer) + [EOS]
    max_len = ctx + 1
    if len(answer_ids) >= max_len:
        answer_ids = answer_ids[:max_len - 1] + [EOS]
    keep_prompt = max_len - len(answer_ids)
    if len(prompt_ids) > keep_prompt:
        prompt_ids = prompt_ids[-keep_prompt:] if keep_prompt > 0 else []
    ids = prompt_ids + answer_ids
    if len(ids) < 3:
        return None
    s = len(ids) - 1
    mask = [0] * s
    for t in range(s):
        mask[t] = 1 if (t + 1) >= len(prompt_ids) else 0
    return ids, mask


def load_records(path):
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records
