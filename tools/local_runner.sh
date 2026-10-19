#!/bin/bash
# Local command runner.
# Start once in your own shell (with the thesis conda env active):
#     cd ~/Desktop/afcp-credit-thesis && bash tools/local_runner.sh
# It watches .runner_queue/*.sh, runs each script in this repo, writes the
# output to .runner_queue/done/<name>.log and moves the script to done/.
# Stop with Ctrl+C.
cd "$(dirname "$0")/.." || exit 1
mkdir -p .runner_queue/done
echo "runner ready in $(pwd) — python: $(which python)"
while true; do
  for f in .runner_queue/*.sh; do
    [ -e "$f" ] || continue
    name=$(basename "$f" .sh)
    echo "$(date '+%H:%M:%S') ▶ $name"
    bash "$f" > ".runner_queue/done/$name.log" 2>&1
    echo "exit=$?" >> ".runner_queue/done/$name.log"
    mv "$f" ".runner_queue/done/$name.sh"
    echo "$(date '+%H:%M:%S') ✔ $name"
  done
  sleep 3
done
