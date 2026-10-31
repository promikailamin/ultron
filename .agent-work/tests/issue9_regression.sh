#!/usr/bin/env bash
# Issue #9: typo-tolerant retrieval, follow-up context, numeric memory and
# non-repeating composed replies.
set -euo pipefail

cd "$(dirname "$0")/../.."
javac -d build src/*.java

work="$(mktemp -d)"
export XDG_CONFIG_HOME="$work/config"
fail=0

check() {
  local question="$1"; shift
  local answer
  answer="$(java -Xmx2g -cp build ultron.Chat "$question")"
  local ok=1
  for needle in "$@"; do
    if [ "${needle#!}" != "$needle" ]; then
      grep -qi -- "${needle#!}" <<<"$answer" && ok=0
    else
      grep -qi -- "$needle" <<<"$answer" || ok=0
    fi
  done
  if [ "$ok" -eq 1 ]; then
    printf 'PASS  %-44s -> %s\n' "$question" "${answer:0:90}"
  else
    printf 'FAIL  %-44s -> %s\n' "$question" "${answer:0:160}"
    fail=1
  fi
}

check "how can i get sugger cane" "sugar cane" "!not sure"
check "how can i get suger cane" "sugar cane" "!not sure"
check "how to get ieon pickaxe" "iron" "pickaxe" "!not sure"

# Follow-up must inherit the diamond pickaxe topic and discuss durability.
answer="$(printf 'how can i get diamond pickaxe\nso how much durability difference they have\n:quit\n' \
  | java -Xmx2g -cp build ultron.Chat)"
if grep -qi "durability" <<<"$answer" && ! grep -qi "I am not sure" <<<"$answer"; then
  printf 'PASS  %-44s -> %s\n' "durability follow-up" "$(grep -i durability <<<"$answer" | head -c 90)"
else
  printf 'FAIL  %-44s -> %s\n' "durability follow-up" "${answer:0:200}"
  fail=1
fi

# Numeric possessions must survive the turn and the misspelling.
answer="$(printf 'i have 400 doller\nhow many doller i have\n:quit\n' \
  | java -Xmx2g -cp build ultron.Chat)"
if grep -qi "400" <<<"$answer" && grep -qi "dollar" <<<"$answer"; then
  printf 'PASS  %-44s -> %s\n' "numeric memory" "$(grep -i '400' <<<"$answer" | tail -1 | head -c 90)"
else
  printf 'FAIL  %-44s -> %s\n' "numeric memory" "${answer:0:200}"
  fail=1
fi

# The same question should not replay one fixed sentence.
answer="$(printf 'how to get iron pickaxe\nhow to get iron pickaxe\nhow to get iron pickaxe\n:quit\n' \
  | java -Xmx2g -cp build ultron.Chat)"
distinct="$(grep 'ultron>' <<<"$answer" | sort -u | wc -l | tr -d ' ')"
if [ "$distinct" -ge 2 ]; then
  printf 'PASS  %-44s -> %s distinct replies\n' "repeat variation" "$distinct"
else
  printf 'FAIL  %-44s -> only %s distinct reply\n' "repeat variation" "$distinct"
  fail=1
fi

if [ "$fail" -ne 0 ]; then
  echo "ISSUE 9 REGRESSION TESTS FAILED"
  exit 1
fi
echo "ISSUE 9 REGRESSION TESTS PASSED"
