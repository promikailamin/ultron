#!/usr/bin/env bash
# Run the Phase 5 evaluation through the real Chat.reply path.
set -euo pipefail

cd "$(dirname "$0")/.."
javac -d build src/*.java

work="$(mktemp -d)"
export XDG_CONFIG_HOME="$work/config"
start="$(date +%s%N)"
java -Xmx2g -cp build ultron.ChatBatch < eval/questions.tsv > "$work/answers.tsv"
end="$(date +%s%N)"
elapsed="$(python3 -c "print(($end-$start)/1e9)")"

mkdir -p .agent-work/outputs
cp "$work/answers.tsv" .agent-work/outputs/eval-answers.tsv
python3 eval/score.py eval/questions.tsv "$work/answers.tsv" --elapsed "$elapsed" \
  | tee .agent-work/outputs/eval-report.txt
