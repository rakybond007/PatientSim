#!/bin/bash
# Run novice-doctor experiments serially.

set -u
PYTHON=/Users/hj/miniforge3/envs/patientsim/bin/python
SRC=/Users/hj/ai_study/healthcare/PatientSim/src
LOGDIR=/tmp/patientsim_logs
mkdir -p "$LOGDIR"
cd "$SRC"

NOVICE_BASE="--doctor-prompt-file-base novice_doctor --doctor-prompt-file-mem novice_doctor_with_memory"

# B_novice: same-type IO + memory (novice doctor)
echo "[$(date)] starting B_novice"
$PYTHON run_memory_simulation.py \
    --exp-name expB_novice_same_type \
    --mode same_type \
    --diagnosis "Intestinal obstruction" \
    --num-scenarios 5 --total-inferences 8 --memory-window 3 --seed 42 \
    $NOVICE_BASE \
    > "$LOGDIR/N-B.log" 2>&1
echo "[$(date)] B_novice done"

# C_novice: cross-type + memory
echo "[$(date)] starting C_novice"
$PYTHON run_memory_simulation.py \
    --exp-name expC_novice_cross_type \
    --mode cross_type \
    --num-scenarios 5 --total-inferences 8 --memory-window 3 --seed 42 \
    $NOVICE_BASE \
    > "$LOGDIR/N-C.log" 2>&1
echo "[$(date)] C_novice done"

# F_novice: low_recall + memory
echo "[$(date)] starting F_novice"
$PYTHON run_memory_simulation.py \
    --exp-name expF_novice_low_recall \
    --mode cross_type --filter-recall low \
    --num-scenarios 5 --total-inferences 8 --memory-window 3 --seed 42 \
    $NOVICE_BASE \
    > "$LOGDIR/N-F.log" 2>&1
echo "[$(date)] F_novice done"

# F_novice_baseline: low_recall + no_memory
echo "[$(date)] starting F_novice_baseline"
$PYTHON run_memory_simulation.py \
    --exp-name expF_novice_low_recall_nomem \
    --mode no_memory --filter-recall low \
    --num-scenarios 5 --total-inferences 8 --memory-window 3 --seed 42 \
    $NOVICE_BASE \
    > "$LOGDIR/N-F-baseline.log" 2>&1
echo "[$(date)] F_novice_baseline done"

echo "[$(date)] All novice experiments complete"
