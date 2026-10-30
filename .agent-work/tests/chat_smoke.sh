#!/usr/bin/env bash
# Smoke tests for the Ultron chat CLI.
# Usage: bash .agent-work/tests/chat_smoke.sh
set -euo pipefail

cd "$(dirname "$0")/../.."
javac -d build src/*.java
export XDG_CONFIG_HOME="${XDG_CONFIG_HOME:-$(mktemp -d)}"

fail=0
check() {
  local question="$1" expected="$2"
  local answer
  answer="$(java -Xmx2g -cp build ultron.Chat "$question")"
  if grep -qi -- "$expected" <<<"$answer"; then
    printf 'PASS  %-42s -> %s\n' "$question" "${answer:0:90}"
  else
    printf 'FAIL  %-42s -> %s\n' "$question" "${answer:0:90}"
    fail=1
  fi
}

check "hello" "Ultron"
check "who are you" "language model"
check "what is a noun" "noun"
check "what is a comma splice" "comma"
check "what is a creeper" "creeper"
check "what is redstone" "redstone"
check "how do I make a crafting table" "crafting"
check "how do I get netherite" "netherite"
check "what is a beacon" "beacon"
check "what is a boomstick" "not sure"
check "what obsidian" "obsidian"
check "can we craft obsidian" "cannot be crafted"
check "how many edition exist of minecraft" "Java Edition"
check "how to make an iron pickaxe" "iron"
check "describe amethyst shard" "amethyst"

if [ "$fail" -ne 0 ]; then
  echo "CHAT SMOKE TESTS FAILED"
  exit 1
fi
echo "CHAT SMOKE TESTS PASSED"
