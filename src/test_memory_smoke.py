"""Smoke test for memory module (no LLM calls)."""
import sys, os, tempfile
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from memory import MedicalMemoryStore
from agent.reviewer_agent import ReviewerAgent
from agent.doctor_agent import DoctorAgent
from agent.patient_agent import PatientAgent
print('All imports OK')

with tempfile.TemporaryDirectory() as tmp:
    mem = MedicalMemoryStore(os.path.join(tmp, 'memory.json'), window_size=2)
    assert not mem.has_memory()

    rec_template = lambda hid: {
        'hadm_id': hid, 'patient_context': f'ctx{hid}', 'gt_diagnosis': f'dx{hid}',
        'scores': {'history_taking': 3, 'ddx_reasoning': 3, 'clinical_communication': 4,
                   'safety_risk_management': 3, 'diagnosis_management_plan': 3},
        'summary': f's{hid}', 'questioning_strategy': f'q{hid}',
        'identified_errors': [], 'key_lessons': [f'lesson{hid}']
    }
    mem.add(rec_template('1'))
    mem.add(rec_template('2'))
    assert len(mem.data['recent']) == 2

    distill_calls = []
    def fake_distill(cur, arch):
        distill_calls.append(arch['hadm_id'])
        return 'DISTILLED_' + str(arch['hadm_id'])

    mem.add(rec_template('3'), distill_fn=fake_distill)
    assert len(mem.data['recent']) == 2, f"expected 2 got {len(mem.data['recent'])}"
    assert distill_calls == ['1'], f"expected ['1'] got {distill_calls}"
    assert 'DISTILLED_1' in mem.data['distilled']
    assert mem.data['recent'][0]['hadm_id'] == '2'
    assert mem.data['recent'][1]['hadm_id'] == '3'

    print('Sliding window + distillation test: PASS')
    print('Distilled:', mem.data['distilled'])
    print('--- render_for_prompt (first 600 chars) ---')
    print(mem.render_for_prompt()[:600])
    print('--- end preview ---')

    # Reload test
    mem2 = MedicalMemoryStore(os.path.join(tmp, 'memory.json'), window_size=2)
    assert mem2.data['meta']['num_consultations'] == 3
    assert len(mem2.data['recent']) == 2
    assert 'DISTILLED_1' in mem2.data['distilled']
    print('Persistence reload: PASS')
