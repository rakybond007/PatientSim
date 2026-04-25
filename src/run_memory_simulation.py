"""Iterative doctor-patient-reviewer loop with Medical Memory.

Implements the Phase 1 proposal architecture: Doctor Agent consults Patient
Simulator, Reviewer Agent scores the consultation on a 5-category rubric, and
the result is stored into a MedicalMemoryStore (sliding window + rolling
summary). Subsequent consultations retrieve the memory via system-prompt
injection.

Usage:
    python run_memory_simulation.py \\
        --exp-name smoke \\
        --mode same_type \\
        --num-scenarios 3 \\
        --total-inferences 5 \\
        --memory-window 3

Modes:
    no_memory   Baseline: memory is never injected. Reviewer still scores.
    same_type   Scenarios filtered by a single diagnosis. Memory accumulates.
    cross_type  Random scenarios across diagnoses. Memory accumulates.
    zero_shot   First N scenarios are "training" (memory accumulates), final
                scenario is a different diagnosis with that memory injected.
"""
import os
import sys
import json
import time
import random
import logging
import argparse
from datetime import datetime

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from agent.patient_agent import PatientAgent
from agent.doctor_agent import DoctorAgent
from agent.reviewer_agent import ReviewerAgent
from memory import MedicalMemoryStore
from utils import set_seed, detect_termination, save_to_dialogue


# ------------------------------ scenario selection ------------------------------

def _apply_filters(profiles, filter_diagnosis=None, filter_recall=None, filter_personality=None, filter_cefr=None):
    pool = list(profiles)
    if filter_diagnosis:
        pool = [p for p in pool if p.get("diagnosis") == filter_diagnosis]
    if filter_recall:
        pool = [p for p in pool if p.get("recall_level") == filter_recall]
    if filter_personality:
        pool = [p for p in pool if p.get("personality") == filter_personality]
    if filter_cefr:
        pool = [p for p in pool if p.get("cefr") == filter_cefr]
    return pool


def pick_same_type(profiles, diagnosis, k, rng, **filters):
    pool = _apply_filters(profiles, filter_diagnosis=diagnosis, **filters)
    rng.shuffle(pool)
    return pool[:k]


def pick_cross_type(profiles, k, rng, **filters):
    pool = _apply_filters(profiles, **filters)
    rng.shuffle(pool)
    return pool[:k]


def pick_zero_shot(profiles, train_dx, target_dx, k_train, rng, **filters):
    train_pool = _apply_filters(profiles, filter_diagnosis=train_dx, **filters)
    rng.shuffle(train_pool)
    train = train_pool[:k_train]
    target_pool = _apply_filters(profiles, filter_diagnosis=target_dx, **filters)
    rng.shuffle(target_pool)
    target = target_pool[:1]
    return train + target


# ------------------------------ consultation loop ------------------------------

def run_single_consultation(scenario, memory_text, args, prompt_dirs, agent_cfg):
    """Run one doctor-patient dialogue. Returns (dialog_history, agents)."""
    patient_agent = PatientAgent(
        patient_profile=scenario.copy(),
        backend_str=agent_cfg["patient_backend"],
        backend_api_type=agent_cfg["patient_api_type"],
        prompt_dir=prompt_dirs["simulation"],
        prompt_file="initial_system_patient_w_persona",
        num_word_sample=10,
        cefr_type=scenario.get("cefr"),
        personality_type=scenario.get("personality"),
        recall_level_type=scenario.get("recall_level"),
        dazed_level_type=scenario.get("dazed_level"),
        client_params={"temperature": agent_cfg["patient_temperature"], "seed": args.seed},
        verbose=args.verbose,
    )

    doctor_prompt_file = (
        "initial_system_doctor_with_memory" if memory_text else "initial_system_doctor"
    )
    doctor_agent = DoctorAgent(
        max_infs=args.total_inferences,
        top_k_diagnosis=5,
        backend_str=agent_cfg["doctor_backend"],
        backend_api_type=agent_cfg["doctor_api_type"],
        prompt_dir=prompt_dirs["simulation"],
        prompt_file=doctor_prompt_file,
        patient_info=scenario.copy(),
        client_params={"temperature": agent_cfg["doctor_temperature"], "seed": args.seed},
        medical_memory=memory_text or "",
        verbose=args.verbose,
    )

    dialog_history = [{"role": "Doctor", "content": doctor_agent.doctor_greet}]
    logging.info(f"Doctor: {doctor_agent.doctor_greet}")

    for inf_idx in range(args.total_inferences):
        patient_response = patient_agent.inference(dialog_history[-1]["content"])
        dialog_history.append({"role": "Patient", "content": patient_response})
        logging.info(f"Patient [{inf_idx+1}/{args.total_inferences}]: {patient_response}")

        if inf_idx == args.total_inferences - 1:
            doctor_input = dialog_history[-1]["content"] + "\nThis is the final turn. Now, you must provide your top5 differential diagnosis."
        else:
            doctor_input = dialog_history[-1]["content"]

        doctor_response = doctor_agent.inference(doctor_input)
        dialog_history.append({"role": "Doctor", "content": doctor_response})
        logging.info(f"Doctor [{inf_idx+1}/{args.total_inferences}]: {doctor_response}")

        if detect_termination(doctor_response):
            break

        time.sleep(0.5)  # gentle pacing

    return dialog_history, patient_agent, doctor_agent


# ------------------------------ main orchestration ------------------------------

def main():
    parser = argparse.ArgumentParser(description="PatientSim Medical Memory simulator")
    parser.add_argument("--exp-name", default=f"memory_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    parser.add_argument("--mode", choices=["no_memory", "same_type", "cross_type", "zero_shot"], default="same_type")
    parser.add_argument("--diagnosis", default="Intestinal obstruction",
                        help="Used by same_type and as the 'training' diagnosis for zero_shot.")
    parser.add_argument("--target-diagnosis", default="Pneumonia",
                        help="Final zero-shot target diagnosis (zero_shot mode only).")
    parser.add_argument("--num-scenarios", type=int, default=3,
                        help="Total scenarios (training + 1 target for zero_shot).")
    parser.add_argument("--total-inferences", type=int, default=10)
    parser.add_argument("--memory-window", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data-dir", default="./data/final_data")
    parser.add_argument("--data-file", default="patient_profile")
    parser.add_argument("--prompt-dir-sim", default="./prompts/simulation")
    parser.add_argument("--prompt-dir-review", default="./prompts/review")
    parser.add_argument("--results-dir", default="./results/memory")
    parser.add_argument("--doctor-backend", default="gemini-2.5-flash")
    parser.add_argument("--doctor-api-type", default="genai")
    parser.add_argument("--patient-backend", default="gemini-2.5-flash")
    parser.add_argument("--patient-api-type", default="genai")
    parser.add_argument("--reviewer-backend", default="gemini-2.5-flash")
    parser.add_argument("--reviewer-api-type", default="genai")
    parser.add_argument("--doctor-temperature", type=float, default=0.7)
    parser.add_argument("--patient-temperature", type=float, default=0.7)
    parser.add_argument("--verbose", action="store_true")
    # Persona / recall filters (apply within the chosen mode)
    parser.add_argument("--filter-recall", choices=["low", "high"], default=None,
                        help="Restrict scenarios to a recall level (low/high). Filters Q2 hypothesis cohorts.")
    parser.add_argument("--filter-personality", default=None,
                        help="Restrict scenarios to a single personality (plain/verbose/distrust/pleasing/impatient/overanxious).")
    parser.add_argument("--filter-cefr", choices=["A", "B", "C"], default=None,
                        help="Restrict scenarios to a CEFR language proficiency tier.")
    parser.add_argument("--memory-init", default=None,
                        help="Path to a memory.json snapshot to pre-load. Useful for transfer experiments.")
    parser.add_argument("--memory-read-only", action="store_true",
                        help="Inject memory but do not append/distill new records.")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    set_seed(args.seed)
    rng = random.Random(args.seed)

    # Paths
    exp_dir = os.path.join(args.results_dir, args.exp_name)
    os.makedirs(exp_dir, exist_ok=True)
    memory_path = os.path.join(exp_dir, "memory.json")
    dialog_path = os.path.join(exp_dir, "dialogues.jsonl")
    review_path = os.path.join(exp_dir, "reviews.jsonl")
    config_path = os.path.join(exp_dir, "config.json")
    with open(config_path, "w") as f:
        json.dump(vars(args), f, indent=2)
    logging.info(f"Experiment dir: {exp_dir}")

    # Data
    with open(os.path.join(args.data_dir, f"{args.data_file}.json")) as f:
        profiles = json.load(f)

    # Scenario selection
    extra_filters = {
        "filter_recall": args.filter_recall,
        "filter_personality": args.filter_personality,
        "filter_cefr": args.filter_cefr,
    }
    if args.mode == "same_type":
        scenarios = pick_same_type(profiles, args.diagnosis, args.num_scenarios, rng, **extra_filters)
    elif args.mode == "cross_type":
        scenarios = pick_cross_type(profiles, args.num_scenarios, rng, **extra_filters)
    elif args.mode == "zero_shot":
        k_train = max(args.num_scenarios - 1, 1)
        scenarios = pick_zero_shot(profiles, args.diagnosis, args.target_diagnosis, k_train, rng, **extra_filters)
    elif args.mode == "no_memory":
        scenarios = pick_cross_type(profiles, args.num_scenarios, rng, **extra_filters)
    else:
        raise ValueError(f"unknown mode: {args.mode}")

    if not scenarios:
        raise RuntimeError("No scenarios selected - check --diagnosis and --data-dir.")
    logging.info(f"Selected {len(scenarios)} scenarios (mode={args.mode})")

    # Agents shared across scenarios
    reviewer = ReviewerAgent(
        backend_str=args.reviewer_backend,
        backend_api_type=args.reviewer_api_type,
        prompt_dir=args.prompt_dir_review,
        verbose=args.verbose,
    )
    if args.memory_init and os.path.isfile(args.memory_init):
        import shutil
        shutil.copy(args.memory_init, memory_path)
        logging.info(f"Pre-loaded memory from {args.memory_init}")
    memory = MedicalMemoryStore(path=memory_path, window_size=args.memory_window)
    agent_cfg = {
        "doctor_backend": args.doctor_backend,
        "doctor_api_type": args.doctor_api_type,
        "patient_backend": args.patient_backend,
        "patient_api_type": args.patient_api_type,
        "doctor_temperature": args.doctor_temperature,
        "patient_temperature": args.patient_temperature,
    }
    prompt_dirs = {"simulation": args.prompt_dir_sim, "review": args.prompt_dir_review}

    # Loop
    for idx, scenario in enumerate(scenarios, start=1):
        logging.info(
            f"\n=== Scenario {idx}/{len(scenarios)} | hadm_id={scenario['hadm_id']} | "
            f"diagnosis={scenario['diagnosis']} | memory_entries={len(memory.data['recent'])} ==="
        )

        memory_text = memory.render_for_prompt() if args.mode != "no_memory" else ""
        if memory_text:
            logging.info(f"Injecting memory ({len(memory_text)} chars)")

        start = time.time()
        dialog_history, patient_agent, doctor_agent = run_single_consultation(
            scenario, memory_text, args, prompt_dirs, agent_cfg
        )
        elapsed_consult = time.time() - start

        # Review
        start = time.time()
        try:
            review_record = reviewer.review(dialog_history, scenario)
        except Exception as e:
            logging.error(f"Review failed: {e}. Skipping memory update for this scenario.")
            review_record = None
        elapsed_review = time.time() - start

        # Persist dialogue + review
        save_to_dialogue({
            "scenario_idx": idx,
            "hadm_id": scenario["hadm_id"],
            "diagnosis": scenario["diagnosis"],
            "mode": args.mode,
            "memory_injected": bool(memory_text),
            "memory_entries_before": len(memory.data["recent"]),
            "dialog_history": dialog_history,
            "patient_token_log": patient_agent.token_log,
            "doctor_token_log": doctor_agent.token_log,
            "elapsed_consult_sec": elapsed_consult,
        }, dialog_path)

        if review_record is not None:
            save_to_dialogue({
                "scenario_idx": idx,
                **review_record,
                "elapsed_review_sec": elapsed_review,
            }, review_path)

            # Add to memory (no_memory mode still logs but does not inject)
            if args.mode != "no_memory":
                memory.add(review_record, distill_fn=reviewer.distill)

        logging.info(
            f"Scenario {idx} done | consult={elapsed_consult:.1f}s review={elapsed_review:.1f}s | "
            f"score={review_record['total_over_25'] if review_record else 'N/A'}"
        )

    logging.info(f"\nAll scenarios done. Artifacts in {exp_dir}")
    if memory.has_memory():
        logging.info(f"Final memory: {len(memory.data['recent'])} recent record(s), distilled length={len(memory.data['distilled'])}")


if __name__ == "__main__":
    main()
