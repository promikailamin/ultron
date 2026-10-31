#!/usr/bin/env bash
# Verifies that ultron.jar embeds model+vocab and keeps persistent chat memory.
# Usage: bash .agent-work/tests/jar_memory_smoke.sh
set -euo pipefail

repo="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$repo"
rm -rf build
javac -d build src/*.java

cp out/ultron.bin build/ultron.bin
cp data/ultron-vocab.txt build/ultron-vocab.txt
jar cfe ultron.jar ultron.Chat -C build .

work="$(mktemp -d)"
config="$work/config"
cd "$work"

printf 'My name is MIKAILAMIN\nwhat is my name?\nI like chess\nwhat do I like?\n:quit\n' \
  | XDG_CONFIG_HOME="$config" java -jar "$repo/ultron.jar" > first.log

test -n "$(find "$config/ultron/sessions" -type f -name '*.log' -print -quit)"
grep -q "MIKAILAMIN" "$config/ultron/memory.tsv"
grep -q "chess" "$config/ultron/memory.tsv"

printf 'what is my name?\n:quit\n' \
  | XDG_CONFIG_HOME="$config" java -jar "$repo/ultron.jar" > second.log
grep -q "MIKAILAMIN" "$config/ultron/memory.tsv"

java -cp "$repo/ultron.jar" ultron.Generate "it was the best of times" 5 0.7 40 > generate.log
grep -q "tokens in" generate.log

echo "JAR + MEMORY SMOKE TESTS PASSED"
