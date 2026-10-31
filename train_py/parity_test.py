#!/usr/bin/env python3
"""Python/Java parity test for token ids and logits.

Usage:
    python3 train_py/parity_test.py --checkpoint out/ultron-py/best.pt

The script exports the checkpoint as fp32 into a temporary directory, then
runs the Java LogitDump/TokenizerDump classes from that directory so the jar
and Python see exactly the same vocab, merges and weights.
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile

import torch

from bpe import BPETokenizer
from export import write_model
from model import build_model

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def run_java(tmp, classpath, main, args=None, stdin=None):
    cmd = ["java", "-Xmx2g", "-cp", classpath, main] + (args or [])
    return subprocess.run(cmd, cwd=tmp, input=stdin, text=True,
                          capture_output=True, check=True).stdout


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="out/ultron-py/best.pt")
    ap.add_argument("--tokenizer-dir", default="out/ultron-py")
    ap.add_argument("--samples", default="data/qa_eval.jsonl")
    ap.add_argument("--max-tokenizer-samples", type=int, default=1000)
    ap.add_argument("--max-prompts", type=int, default=20)
    ap.add_argument("--positions", type=int, default=10)
    args = ap.parse_args()

    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model = build_model(ckpt["config"])
    model.load_state_dict(ckpt["model"])
    model.eval()
    tok = BPETokenizer.load(os.path.join(args.tokenizer_dir, "ultron-vocab.txt"),
                            os.path.join(args.tokenizer_dir, "ultron-merges.txt"))

    tmp = tempfile.mkdtemp(prefix="ultron-parity-")
    try:
        # Compile into an isolated classpath so Resources falls back to the
        # temporary data/ directory instead of an old vocab in build/.
        classes = os.path.join(tmp, "classes")
        os.makedirs(classes, exist_ok=True)
        subprocess.run(["javac", "-d", classes, *[
            os.path.join(ROOT, "src", f) for f in os.listdir(os.path.join(ROOT, "src"))
            if f.endswith(".java")]], cwd=ROOT, check=True)
        classpath = classes
        os.makedirs(os.path.join(tmp, "out"), exist_ok=True)
        os.makedirs(os.path.join(tmp, "data"), exist_ok=True)
        write_model(model, os.path.join(tmp, "out", "ultron.bin"), fp16=False)
        shutil.copyfile(os.path.join(args.tokenizer_dir, "ultron-vocab.txt"),
                        os.path.join(tmp, "data", "ultron-vocab.txt"))
        shutil.copyfile(os.path.join(args.tokenizer_dir, "ultron-merges.txt"),
                        os.path.join(tmp, "data", "ultron-merges.txt"))

        records = [json.loads(l) for l in open(args.samples, encoding="utf-8")]
        questions = [r["messages"][0]["content"] for r in records if r.get("messages")]
        samples = questions[:args.max_tokenizer_samples]
        stdin = "\n".join(samples) + "\n"
        java_ids = run_java(tmp, classpath, "ultron.TokenizerDump", stdin=stdin).splitlines()
        assert len(java_ids) == len(samples), (len(java_ids), len(samples))
        tokenizer_mismatches = 0
        for text, line in zip(samples, java_ids):
            py = tok.encode(text)
            ja = [int(x) for x in line.split()] if line.strip() else []
            if py != ja:
                tokenizer_mismatches += 1
                if tokenizer_mismatches <= 3:
                    print("token mismatch for %r\n  py=%s\n  ja=%s" % (text[:80], py[:20], ja[:20]))
        print("tokenizer: %d/%d strings identical" %
              (len(samples) - tokenizer_mismatches, len(samples)))
        if tokenizer_mismatches:
            return 1

        max_diff = 0.0
        prompts = questions[:args.max_prompts]
        for text in prompts:
            ids = tok.encode(text)
            if not ids:
                continue
            ids = ids[-model.ctx:]
            with torch.no_grad():
                x = torch.tensor([ids], dtype=torch.long)
                logits = model(x)[0]
            n = min(args.positions, len(ids))
            out = run_java(tmp, classpath, "ultron.LogitDump", [text, str(n)])
            got = {}
            for line in out.splitlines():
                if not line or line.startswith("#"):
                    continue
                t, v, val = line.split()
                got[(int(t), int(v))] = float(val)
            for t in range(n):
                for v in range(model.vocab):
                    diff = abs(float(logits[t, v]) - got[(t, v)])
                    if diff > max_diff:
                        max_diff = diff
            print("prompt %-40r max diff so far %.3g" % (text[:40], max_diff))
            if max_diff >= 1e-3:
                print("PARITY FAILED: max abs diff %.6g" % max_diff)
                return 1
        print("PARITY PASSED: max abs logit difference %.6g" % max_diff)
        return 0
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())
