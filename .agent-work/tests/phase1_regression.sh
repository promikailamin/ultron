#!/usr/bin/env bash
# Phase 1 regression cases from ULTRON_FIX_PLAN.md.
# Every distinctive word must match, the intent must agree, and follow-ups
# must inherit the previous topic (also across a process restart).
set -euo pipefail

cd "$(dirname "$0")/../.."
javac -d build src/*.java

work="$(mktemp -d)"
export XDG_CONFIG_HOME="$work/config"
fail=0

ask() {
  java -Xmx2g -cp build ultron.Chat "$1"
}

check() {
  local question="$1"; shift
  local answer
  answer="$(ask "$question")"
  local ok=1
  for needle in "$@"; do
    if [ "${needle#!}" != "$needle" ]; then
      grep -qi -- "${needle#!}" <<<"$answer" && ok=0
    else
      grep -qi -- "$needle" <<<"$answer" || ok=0
    fi
  done
  if [ "$ok" -eq 1 ]; then
    printf 'PASS  %-52s -> %s\n' "$question" "${answer:0:90}"
  else
    printf 'FAIL  %-52s -> %s\n' "$question" "${answer:0:160}"
    fail=1
  fi
}

check "what is obsidian" "obsidian"
check "can obsidian be craftable on crafting table" "cannot be crafted" "!four cells"
check "how to get a diamond" "diamond ore" "pickaxe" "!log"
check "how to make iron pickaxe" "three iron ingots" "two sticks" "!planks"
check "iron pickaxe" "iron ingots" "!planks"
check "how many edition exist of minecraft" "Java Edition" "Bedrock Edition" "!achievements"

# Two-turn pronoun follow-up in one process.
answer="$(printf 'what is obsidian\nand how can i craft it\n:quit\n' \
  | java -Xmx2g -cp build ultron.Chat)"
if grep -qi "cannot be crafted" <<<"$answer" \
   && grep -qi "water" <<<"$answer" \
   && grep -qi "lava" <<<"$answer" \
   && ! grep -qi "I am not sure" <<<"$answer"; then
  printf 'PASS  %-52s -> %s\n' "what is obsidian + and how can i craft it" \
    "$(grep -i 'cannot be crafted' <<<"$answer" | head -c 90)"
else
  printf 'FAIL  %-52s -> %s\n' "what is obsidian + and how can i craft it" "${answer:0:240}"
  fail=1
fi

# The topic must survive a process restart via the saved session.
work2="$(mktemp -d)"
XDG_CONFIG_HOME="$work2/config" java -Xmx2g -cp build ultron.Chat "what is obsidian" >/dev/null
answer="$(printf 'and how can i craft it\n:quit\n' \
  | XDG_CONFIG_HOME="$work2/config" java -Xmx2g -cp build ultron.Chat)"
if grep -qi "cannot be crafted" <<<"$answer"; then
  printf 'PASS  %-52s -> %s\n' "follow-up after session restart" \
    "$(grep -i 'cannot be crafted' <<<"$answer" | head -c 90)"
else
  printf 'FAIL  %-52s -> %s\n' "follow-up after session restart" "${answer:0:240}"
  fail=1
fi

if [ "$fail" -ne 0 ]; then
  echo "PHASE 1 REGRESSION TESTS FAILED"
  exit 1
fi
echo "PHASE 1 REGRESSION TESTS PASSED"
