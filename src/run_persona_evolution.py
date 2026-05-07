"""Multi-visit Patient Persona Evolution simulator.

Same patient (single hadm_id) is brought in for N sequential ED visits. Between
visits, a Persona Evolution Updater reads the just-finished consultation +
the patient-experience review, and writes a 2-4 sentence update that the
next visit's PatientAgent reads in its system prompt.

Each visit produces:
  - the dialog
  - clinical-quality review (existing 5-rubric ReviewerAgent)
  - patient-experience review (new PatientExperienceReviewer)
  - persona-expression scores (new PersonaExpressionReviewer)
  - updated persona_evolution string for the NEXT visit

Modes:
  - constant_doctor    : doctor style is the same every visit (dismissive | empathetic | default)
  - alternating_doctor : doctor style flips per visit (1=dismissive, 2=empathetic, 3=dismissive...)
  - prefix_then_repair : first --bad-prefix visits use dismissive, rest empathetic
  - memory_off         : same flow but persona_evolution is never injected (control)
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
from agent.patient_experience_reviewer import PatientExperienceReviewer
from agent.persona_evolution_updater import PersonaEvolutionUpdater
from agent.persona_expression_reviewer import PersonaExpressionReviewer
from utils import set_seed, detect_termination, save_to_dialogue, load_json


DOCTOR_STYLE_PROMPTS = {
    "dismissive": "doctor_dismissive",
    "empathetic": "doctor_empathetic",
    "default": "initial_system_doctor",
}


def pick_patient(profiles, hadm_id=None, persona_filter=None, seed=42):
    rng = random.Random(seed)
    if hadm_id:
        for p in profiles:
            if str(p.get("hadm_id")) == str(hadm_id):
                return p
        raise ValueError(f"No patient with hadm_id={hadm_id}")
    pool = list(profiles)
    if persona_filter:
        pool = [p for p in pool if p.get("personality") == persona_filter]
    if not pool:
        raise ValueError("No patients matched filter.")
    rng.shuffle(pool)
    return pool[0]


def doctor_style_for_visit(args, visit_idx_1based):
    if args.mode == "constant_doctor":
        return args.doctor_style
    if args.mode == "alternating_doctor":
        return "dismissive" if visit_idx_1based % 2 == 1 else "empathetic"
    if args.mode == "prefix_then_repair":
        return "dismissive" if visit_idx_1based <= args.bad_prefix else "empathetic"
    if args.mode == "memory_off":
        return args.doctor_style
    raise ValueError(f"Unknown mode {args.mode}")


def run_one_visit(scenario, persona_evolution, doctor_style, args, prompt_dirs):
    patient_agent = PatientAgent(
        patient_profile=scenario.copy(),
        backend_str=args.patient_backend,
        backend_api_type=args.patient_api_type,
        prompt_dir=prompt_dirs["simulation"],
        prompt_file="initial_system_patient_w_persona_evolved",
        num_word_sample=10,
        cefr_type=scenario.get("cefr"),
        personality_type=scenario.get("personality"),
        recall_level_type=scenario.get("recall_level"),
        dazed_level_type=scenario.get("dazed_level"),
        client_params={"temperature": args.patient_temperature, "seed": args.seed},
        verbose=args.verbose,
        persona_evolution=persona_evolution,
    )
    doctor_prompt_file = DOCTOR_STYLE_PROMPTS.get(doctor_style, "initial_system_doctor")
    doctor_agent = DoctorAgent(
        max_infs=args.total_inferences,
        top_k_diagnosis=5,
        backend_str=args.doctor_backend,
        backend_api_type=args.doctor_api_type,
        prompt_dir=prompt_dirs["simulation"],
        prompt_file=doctor_prompt_file,
        patient_info=scenario.copy(),
        client_params={"temperature": args.doctor_temperature, "seed": args.seed},
        verbose=args.verbose,
    )

    dialog_history = [{"role": "Doctor", "content": doctor_agent.doctor_greet}]
    logging.info(f"  Doctor greet: {doctor_agent.doctor_greet}")

    for inf_idx in range(args.total_inferences):
        patient_response = patient_agent.inference(dialog_history[-1]["content"])
        dialog_history.append({"role": "Patient", "content": patient_response})
        logging.info(f"  Patient[{inf_idx+1}/{args.total_inferences}]: {patient_response}")

        if inf_idx == args.total_inferences - 1:
            doctor_input = dialog_history[-1]["content"] + "\nThis is the final turn. Provide your top5 differential diagnosis."
        else:
            doctor_input = dialog_history[-1]["content"]
        doctor_response = doctor_agent.inference(doctor_input)
        dialog_history.append({"role": "Doctor", "content": doctor_response})
        logging.info(f"  Doctor[{inf_idx+1}/{args.total_inferences}]: {doctor_response}")

        if detect_termination(doctor_response):
            break
        time.sleep(0.4)

    return dialog_history, patient_agent, doctor_agent


def build_base_persona_summary(scenario):
    return (
        f"personality={scenario.get('personality')}, cefr={scenario.get('cefr')}, "
        f"recall={scenario.get('recall_level')}, dazed={scenario.get('dazed_level')}"
    )


def main():
    p = argparse.ArgumentParser(description="Patient Persona Evolution simulator")
    p.add_argument("--exp-name", default=f"persona_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    p.add_argument("--mode", choices=["constant_doctor", "alternating_doctor", "prefix_then_repair", "memory_off"],
                   default="constant_doctor")
    p.add_argument("--doctor-style", choices=["dismissive", "empathetic", "default"], default="dismissive")
    p.add_argument("--bad-prefix", type=int, default=2, help="for prefix_then_repair mode")

    p.add_argument("--num-visits", type=int, default=5)
    p.add_argument("--total-inferences", type=int, default=8)
    p.add_argument("--seed", type=int, default=42)

    p.add_argument("--hadm-id", default=None, help="specific patient by hadm_id; otherwise filter+sample")
    p.add_argument("--persona-filter", default=None, help="restrict to a personality (distrust|pleasing|...)")

    p.add_argument("--data-dir", default="./data/final_data")
    p.add_argument("--data-file", default="patient_profile")
    p.add_argument("--prompt-dir-sim", default="./prompts/simulation")
    p.add_argument("--prompt-dir-review", default="./prompts/review")
    p.add_argument("--prompt-dir-persona", default="./prompts/persona_evolution")
    p.add_argument("--results-dir", default="./results/persona")

    p.add_argument("--patient-backend", default="gemini-2.5-flash")
    p.add_argument("--patient-api-type", default="genai")
    p.add_argument("--doctor-backend", default="gemini-2.5-flash")
    p.add_argument("--doctor-api-type", default="genai")
    p.add_argument("--reviewer-backend", default="gemini-2.5-flash")
    p.add_argument("--reviewer-api-type", default="genai")

    p.add_argument("--patient-temperature", type=float, default=0.7)
    p.add_argument("--doctor-temperature", type=float, default=0.7)
    p.add_argument("--verbose", action="store_true")

    args = p.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    set_seed(args.seed)

    exp_dir = os.path.join(args.results_dir, args.exp_name)
    os.makedirs(exp_dir, exist_ok=True)
    config_path = os.path.join(exp_dir, "config.json")
    with open(config_path, "w") as f:
        json.dump(vars(args), f, indent=2)
    logging.info(f"Experiment dir: {exp_dir}")

    profiles = load_json(os.path.join(args.data_dir, f"{args.data_file}.json"))
    scenario = pick_patient(profiles, hadm_id=args.hadm_id, persona_filter=args.persona_filter, seed=args.seed)
    logging.info(
        f"Patient: hadm_id={scenario['hadm_id']} ({scenario.get('age')}{scenario.get('gender')}) "
        f"persona={scenario.get('personality')}/{scenario.get('cefr')}/{scenario.get('recall_level')}"
    )

    base_persona_summary = build_base_persona_summary(scenario)

    # Reviewers / updater (shared)
    clinical_reviewer = ReviewerAgent(
        backend_str=args.reviewer_backend, backend_api_type=args.reviewer_api_type,
        prompt_dir=args.prompt_dir_review,
    )
    experience_reviewer = PatientExperienceReviewer(
        backend_str=args.reviewer_backend, backend_api_type=args.reviewer_api_type,
        prompt_dir=args.prompt_dir_persona,
    )
    expression_reviewer = PersonaExpressionReviewer(
        backend_str=args.reviewer_backend, backend_api_type=args.reviewer_api_type,
        prompt_dir=args.prompt_dir_persona,
    )
    updater = PersonaEvolutionUpdater(
        backend_str=args.reviewer_backend, backend_api_type=args.reviewer_api_type,
        prompt_dir=args.prompt_dir_persona,
    )

    persona_evolution_text = ""
    visits_log = []

    visits_path = os.path.join(exp_dir, "visits.jsonl")
    persona_path = os.path.join(exp_dir, "persona_state.jsonl")

    for visit in range(1, args.num_visits + 1):
        doctor_style = doctor_style_for_visit(args, visit)
        logging.info(f"\n=== VISIT {visit}/{args.num_visits} | doctor={doctor_style} | "
                     f"persona_evo_len={len(persona_evolution_text)} ===")

        injected_evolution = "" if args.mode == "memory_off" else persona_evolution_text

        start = time.time()
        dialog_history, patient_agent, doctor_agent = run_one_visit(
            scenario, injected_evolution, doctor_style, args,
            prompt_dirs={"simulation": args.prompt_dir_sim, "review": args.prompt_dir_review},
        )
        elapsed_consult = time.time() - start

        # Reviews
        try:
            clinical_record = clinical_reviewer.review(dialog_history, patient_agent.patient_profile)
        except Exception as e:
            logging.error(f"clinical reviewer failed: {e}")
            clinical_record = None
        try:
            experience_record = experience_reviewer.review(dialog_history, base_persona_summary=base_persona_summary)
        except Exception as e:
            logging.error(f"experience reviewer failed: {e}")
            experience_record = None
        try:
            expression_record = expression_reviewer.review(dialog_history)
        except Exception as e:
            logging.error(f"expression reviewer failed: {e}")
            expression_record = None

        # Update persona_evolution_text via the updater (skip in memory_off so the
        # control truly stays at base persona).
        new_evolution_text = persona_evolution_text
        if args.mode != "memory_off" and experience_record is not None:
            try:
                visit_outcome = (
                    f"clinical_total={clinical_record.get('total_score','?') if clinical_record else 'NA'}, "
                    f"doctor_label={experience_record.get('doctor_behavior_label','?')}"
                )
                new_evolution_text = updater.update(
                    base_persona_summary=base_persona_summary,
                    prior_evolution=persona_evolution_text,
                    dialog_history=dialog_history,
                    review=experience_record,
                    visit_outcome=visit_outcome,
                )
                logging.info(f"  Persona evolution update ({len(new_evolution_text)} chars):")
                for line in new_evolution_text.splitlines():
                    logging.info(f"    {line}")
            except Exception as e:
                logging.error(f"persona updater failed: {e}")

        # Persist
        visit_record = {
            "visit": visit,
            "hadm_id": scenario["hadm_id"],
            "doctor_style": doctor_style,
            "persona_evolution_before": persona_evolution_text,
            "persona_evolution_after": new_evolution_text,
            "dialog_history": dialog_history,
            "clinical_review": clinical_record,
            "experience_review": experience_record,
            "expression_review": expression_record,
            "elapsed_consult_sec": elapsed_consult,
        }
        save_to_dialogue(visit_record, visits_path)
        save_to_dialogue({
            "visit": visit,
            "doctor_style": doctor_style,
            "persona_evolution_before": persona_evolution_text,
            "persona_evolution_after": new_evolution_text,
            "expression_scores": expression_record["scores"] if expression_record else None,
            "experience_scores": experience_record["scores"] if experience_record else None,
            "clinical_total": clinical_record["total_score"] if clinical_record else None,
        }, persona_path)

        visits_log.append(visit_record)
        persona_evolution_text = new_evolution_text

    summary = {
        "args": vars(args),
        "patient": {
            "hadm_id": scenario["hadm_id"],
            "age": scenario.get("age"),
            "gender": scenario.get("gender"),
            "personality": scenario.get("personality"),
            "cefr": scenario.get("cefr"),
            "recall_level": scenario.get("recall_level"),
            "dazed_level": scenario.get("dazed_level"),
            "diagnosis": scenario.get("diagnosis"),
        },
        "visits": [
            {
                "visit": v["visit"],
                "doctor_style": v["doctor_style"],
                "expression_scores": v["expression_review"]["scores"] if v["expression_review"] else None,
                "experience_scores": v["experience_review"]["scores"] if v["experience_review"] else None,
                "clinical_total": v["clinical_review"]["total_score"] if v["clinical_review"] else None,
            }
            for v in visits_log
        ],
        "final_persona_evolution": persona_evolution_text,
    }
    with open(os.path.join(exp_dir, "summary.json"), "w") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    logging.info(f"\nAll {args.num_visits} visits done. Artifacts in {exp_dir}")


if __name__ == "__main__":
    main()
