#!/bin/bash
# One RQ4 (group-definition) run. Usage: rq4_job.sh <ds[@cand]> <model> <seed> <cfg> <cond> <n_train> <n_calib> <n_test> <adaptive1>
# Skips the run if its result file already exists (default ttest_delta = 0.0, sig_level = 0.1).
# Output: thesis_experiments/results/v2/afcp/rq4_groupdef/; logs: logs/rq4/
ds=$1; model=$2; seed=$3; cfg=$4; cond=$5; ntr=$6; nc=$7; nte=$8; a1=$9
tag=$(echo "$ds" | tr '@+' '-_')
f=thesis_experiments/results/v2/afcp/rq4_groupdef/${tag}_${model}_ntr${ntr}_ncal${nc}_nte${nte}_cond${cond}_td0.0_sl0.1_seed${seed}.txt
if [ -f "$f" ]; then echo "skip $f"; exit 0; fi
mkdir -p logs/rq4
python thesis_experiments/v2/run_afcp.py "$ds" "$model" "$seed" "$cfg" "$cond" n_train=$ntr n_calib=$nc n_test=$nte adaptive1=$a1 \
  > "logs/rq4/${tag}_${model}_${cond}_${seed}.log" 2>&1 || echo "FAILED: $*"
