#!/bin/bash
# Claude command runner.
# Start once in your own shell (with the thesis conda env active):
#     cd ~/Desktop/afcp-credit-thesis && bash tools/claude_runner.sh
# It watches .claude_queue/*.sh, runs each script in this repo, writes the
# output to .claude_queue/done/<name>.log and moves the script to done/.
# Stop with Ctrl+C.
cd "$(dirname "$0")/.." || exit 1
mkdir -p .claude_queue/done
echo "runner ready in $(pwd) — python: $(which python)"
while true; do
  for f in .claude_queue/*.sh; do
    [ -e "$f" ] || continue
    name=$(basename "$f" .sh)
    echo "$(date '+%H:%M:%S') ▶ $name"
    bash "$f" > ".claude_queue/done/$name.log" 2>&1
    echo "exit=$?" >> ".claude_queue/done/$name.log"
    mv "$f" ".claude_queue/done/$name.sh"
    echo "$(date '+%H:%M:%S') ✔ $name"
  done
  sleep 3
done
