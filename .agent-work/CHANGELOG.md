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
