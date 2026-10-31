# Ultron changelog

## 2026-10-31 - Issue 7, Phase 1

- Added `.agent-work/tests/phase1_regression.sh` covering all six regression
  cases from `ULTRON_FIX_PLAN.md`, the two-turn pronoun follow-up, and the
  follow-up after a process restart.
- Confirmed the intent-aware `KnowledgeBase` already keeps intent words in the
  query, requires every distinctive token, tags entries with an intent, and
  persists `lastTopic` through the session.
- Result: `PHASE 1 REGRESSION TESTS PASSED`.

## 2026-10-31 - Issue 7, Phase 2

- Added `.agent-work/tools/gen_qa.py`. It reads `chat_kb.tsv`, `mc_qa.txt`,
  `mc.txt` and `grammar.txt`, then generates multiple question phrasings and
  3-5 answer wordings per fact, 2-4 turn pronoun dialogues, retrieval-format
  examples (including deliberate abstains), and fix-the-sentence grammar tasks.
- The train/eval split is by fact (5% of facts), so no phrasing of an eval fact
  leaks into training.
- Generated `data/qa_train.jsonl` (52,130 records), `data/qa_eval.jsonl`
  (2,794 records) and `data/pretrain.txt` (29.7 MB, 5.13M words).
- Added `.agent-work/tests/qa_data.sh`; result:
  `PHASE 2 DATA TESTS PASSED`.

## 2026-10-31 - Issue 7, Phases 3 and 4

- Added `train_py/` with a from-scratch PyTorch mirror of `Ultron.java`
  (pre-LN, learned positions, separate Q/K/V/O, tanh GELU, tied output head).
  Defaults match the plan: V=8192 BPE, D=384, L=8, H=6, F=1536, ctx=512.
- Added a small whitespace-pre-split BPE trainer (`train_py/bpe.py`) with a
  plain merges file, plus a Stage A/Stage B training script with fp16 autocast,
  AdamW, warmup/cosine, gradient clipping, answer-only loss and early stopping.
- Added `train_py/export.py` (fp16 by default) in the exact Java parameter
  order, transposing every PyTorch linear weight.
- Extended `Tokenizer.java` to replay the Python merges file and decode the
  same text, and added `Tokenizer.loadDefault()`.
- Extended `Ultron.load` to accept fp16 checkpoints (new `ULT2` magic) while
  keeping legacy fp32 checkpoints working.
- Added `LogitDump`, `TokenizerDump`, `train_py/parity_test.py` and
  `.agent-work/tests/python_pipeline.sh`. Smoke result:
  100/100 token strings identical, max logit difference 2.39e-07 (fp32);
  fp16 load max difference 2.63e-04.
- Updated `build.yml`: removed Java CPU training from the normal build, runs a
  tiny PyTorch train/export/parity smoke test instead, and packages optional
  `ultron-merges.txt` into the jar.
