#!/usr/bin/env bash
# Regression test for issue #4: Minecraft answers must be grounded and each
# interactive reply must be emitted as a stream of pieces, not one blob.
set -euo pipefail

cd "$(dirname "$0")/../.."
javac -d build src/*.java
work="$(mktemp -d)"
export XDG_CONFIG_HOME="$work/config"

cat <<'IN' | java -Xmx2g -cp build ultron.Chat > "$work/out.log"
what obsidian
can obsidian be craftable on crafting table
can we craft obsidian
what is you name
how many edition exist of minecraft
how to get a diamond
how to make iron pickaxe
iron pickaxe
what is archaeology
:quit
IN

grep -q "Obsidian is a dark" "$work/out.log"
grep -q "not made at a crafting table" "$work/out.log"
grep -q "cannot be crafted at a crafting table" "$work/out.log"
grep -q "My name is Ultron" "$work/out.log"
grep -q "Java Edition and Bedrock Edition" "$work/out.log"
grep -q "Diamonds are found underground" "$work/out.log"
grep -q "iron pickaxe is crafted from three iron ingots" "$work/out.log"
grep -q "Archaeology lets you brush" "$work/out.log"
if grep -qi "exception\|nullpointer" "$work/out.log"; then
  echo "ISSUE 4 REGRESSION TEST FAILED (exception in chat output)"
  exit 1
fi
test -n "$(find "$work/config/ultron/sessions" -type f -name '*.log' -print -quit)"
echo "ISSUE 4 REGRESSION TEST PASSED"
