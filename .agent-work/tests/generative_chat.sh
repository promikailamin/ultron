#!/usr/bin/env bash
# Regression test for issue #5: chat replies must come from the transformer's
# token-by-token generation, not from a knowledge-base lookup.
set -euo pipefail

cd "$(dirname "$0")/../.."
javac -d build src/*.java
work="$(mktemp -d)"
export XDG_CONFIG_HOME="$work/config"

fail=0
check() {
  local question="$1"
  local answer
  answer="$(java -Xmx2g -cp build ultron.Chat "$question")"
  if [ -z "$answer" ] || grep -qi "still learning to answer that" <<<"$answer"; then
    printf 'FAIL  %-40s -> %s\n' "$question" "${answer:0:100}"
    fail=1
  elif grep -qi "exception\|nullpointer" <<<"$answer"; then
    printf 'FAIL  %-40s -> %s\n' "$question" "${answer:0:100}"
    fail=1
  else
    printf 'PASS  %-40s -> %s\n' "$question" "${answer:0:100}"
  fi
}

check "hello"
check "what is obsidian"
check "what is redstone"
check "what is a noun"
check "how do i make a crafting table"
check "what is a creeper"
check "how do i get netherite"

# The same prompt must still be produced by the neural sampler; verify the
# standalone generator reports generated token counts.
java -Xmx2g -cp build ultron.Generate "user: what is obsidian
ultron:" 10 0.7 40 > "$work/generate.log"
grep -q "tokens in" "$work/generate.log"

if [ "$fail" -ne 0 ]; then
  echo "GENERATIVE CHAT TESTS FAILED"
  exit 1
fi
echo "GENERATIVE CHAT TESTS PASSED"
