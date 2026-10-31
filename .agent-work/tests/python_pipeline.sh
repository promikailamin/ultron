#!/usr/bin/env bash
# Phase 3/4 smoke test: train a tiny PyTorch model, export it, then verify
# Python and Java token ids and logits agree.
set -euo pipefail

cd "$(dirname "$0")/../.."
work=".agent-work/tmp/py-smoke"
rm -rf "$work"
python3 train_py/train.py --smoke --out "$work"
python3 train_py/export.py --checkpoint "$work/best.pt" \
  --out "$work/out/ultron.bin" --vocab-out "$work/data" --tokenizer-dir "$work"
python3 train_py/parity_test.py --checkpoint "$work/best.pt" \
  --tokenizer-dir "$work" --max-tokenizer-samples 100 \
  --max-prompts 2 --positions 3
echo "PYTHON PIPELINE TESTS PASSED"
