#!/bin/bash
# Run F_strict + F_strict-baseline serially for Q2 hypothesis under strict-novice prompt.

set -u
PYTHON=/Users/hj/miniforge3/envs/patientsim/bin/python
SRC=/Users/hj/ai_study/healthcare/PatientSim/src
LOGDIR=/tmp/patientsim_logs
mkdir -p "$LOGDIR"
cd "$SRC"

STRICT="--doctor-prompt-file-base novice_doctor --doctor-prompt-file-mem novice_doctor_with_memory"

echo "[$(date)] starting F_strict (low_recall + memory)"
$PYTHON run_memory_simulation.py \
    --exp-name expF_strict_low_recall \
    --mode cross_type --filter-recall low \
    --num-scenarios 5 --total-inferences 8 --memory-window 3 --seed 42 \
    $STRICT > "$LOGDIR/S-F.log" 2>&1
echo "[$(date)] F_strict done"

echo "[$(date)] starting F_strict-baseline (low_recall, no memory)"
$PYTHON run_memory_simulation.py \
    --exp-name expF_strict_low_recall_nomem \
    --mode no_memory --filter-recall low \
    --num-scenarios 5 --total-inferences 8 --memory-window 3 --seed 42 \
    $STRICT > "$LOGDIR/S-F-baseline.log" 2>&1
echo "[$(date)] F_strict-baseline done"

echo "[$(date)] All Q2 strict experiments complete"
