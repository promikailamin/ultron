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

## 2026-10-31 - Issue 7, Phase 5

- Added `src/ChatBatch.java` and `Chat.resetTopic()` so the real `Chat.reply`
  path can be evaluated in one JVM over a TSV of single- and multi-turn cases.
- Added `.agent-work/tools/build_eval_questions.py` and generated
  `eval/questions.tsv` (386 cases: 160 held-out, 150 KB, 40 follow-up,
  30 out-of-domain, 6 regression).
- Added `eval/run_eval.sh` and `eval/score.py`; the scorer checks every
  `must_contain` term, forbids every `must_not_contain` term, reports
  per-category accuracy, exact-copy rate and average latency.
- Hardened the Phase 1 retrieval stopgap: it now loads `mc_qa.txt` as a strict
  secondary corpus, strips conversational wrappers, treats intent words as
  generic, requires the extracted topic words, and refuses model output that
  does not mention the topic early in the answer. Unknown topics now abstain
  instead of inventing an answer.
- Local result with the committed legacy checkpoint:
  held-out 140/160 (87.5%), follow-up 39/40 (97.5%), OOD abstention 30/30
  (100%), six regression cases 6/6, average answer time 0.020 s.
- Exact-copy is still 335/386 (86.8%) because the retrieval stopgap returns
  stored KB/wiki answers. The plan's <20% target requires the GPU-trained 17M
  model and the Phase 4 removal of the stored-answer path; no GPU was
  available in this environment.
