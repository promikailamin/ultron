# Ultron changelog

## 2026-10-31 - Issue 7, Phase 1

- Added `.agent-work/tests/phase1_regression.sh` covering all six regression
  cases from `ULTRON_FIX_PLAN.md`, the two-turn pronoun follow-up, and the
  follow-up after a process restart.
- Confirmed the intent-aware `KnowledgeBase` already keeps intent words in the
  query, requires every distinctive token, tags entries with an intent, and
  persists `lastTopic` through the session.
- Result: `PHASE 1 REGRESSION TESTS PASSED`.
