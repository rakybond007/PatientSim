#!/bin/bash
# Run Patient Persona Evolution experiment chain (P1-P5).

set -u
PYTHON=/Users/hj/miniforge3/envs/patientsim/bin/python
SRC=/Users/hj/ai_study/healthcare/PatientSim/src
LOGDIR=/tmp/patientsim_logs
mkdir -p "$LOGDIR"
cd "$SRC"

NUM_VISITS=5
TURNS=6
SEED=42

# P1: distrust patient + dismissive doctor (memory ON)
echo "[$(date)] starting P1 distrust+dismissive (memory on)"
$PYTHON run_persona_evolution.py \
    --exp-name P1_distrust_dismissive \
    --mode constant_doctor --doctor-style dismissive \
    --persona-filter distrust --num-visits $NUM_VISITS --total-inferences $TURNS --seed $SEED \
    > "$LOGDIR/P1.log" 2>&1
echo "[$(date)] P1 done"

# P2: distrust patient + empathetic doctor (memory ON)
echo "[$(date)] starting P2 distrust+empathetic"
$PYTHON run_persona_evolution.py \
    --exp-name P2_distrust_empathetic \
    --mode constant_doctor --doctor-style empathetic \
    --persona-filter distrust --num-visits $NUM_VISITS --total-inferences $TURNS --seed $SEED \
    > "$LOGDIR/P2.log" 2>&1
echo "[$(date)] P2 done"

# P3: distrust patient + dismissive(2) then empathetic(3) — repair trajectory
echo "[$(date)] starting P3 prefix_then_repair"
$PYTHON run_persona_evolution.py \
    --exp-name P3_prefix_then_repair \
    --mode prefix_then_repair --bad-prefix 2 \
    --persona-filter distrust --num-visits $NUM_VISITS --total-inferences $TURNS --seed $SEED \
    > "$LOGDIR/P3.log" 2>&1
echo "[$(date)] P3 done"

# P5: memory OFF control — distrust + dismissive, no persona evolution accumulation
echo "[$(date)] starting P5 memory_off control"
$PYTHON run_persona_evolution.py \
    --exp-name P5_memory_off_distrust_dismissive \
    --mode memory_off --doctor-style dismissive \
    --persona-filter distrust --num-visits $NUM_VISITS --total-inferences $TURNS --seed $SEED \
    > "$LOGDIR/P5.log" 2>&1
echo "[$(date)] P5 done"

echo "[$(date)] All persona-evolution experiments complete"
