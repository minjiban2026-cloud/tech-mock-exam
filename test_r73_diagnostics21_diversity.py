from pathlib import Path
import ast

ROOT=Path(__file__).resolve().parent

def test_r73_builder_has_bounded_two_judges_and_reject_memory():
    s=(ROOT/'exam_builder.py').read_text(encoding='utf-8')
    assert 'R73_BOUNDED_DIVERSITY_CHECKPOINT_GATE' in s
    assert 'max_judge_per_domain=2' in s
    assert 'forbidden_attempts=None' in s
    assert "rejected_pairs_out.setdefault(d,[]).append(list(ps))" in s
    assert "candidate_priority_policy':'R73_DIFFICULTY_FIRST'" in s
    assert "forbidden_pairs=persisted_pairs | prior_rejected" in s

def test_r73_writer_targets_diagnostics21_failure():
    s=(ROOT/'capability_contracts.py').read_text(encoding='utf-8')
    assert 'inferential_distance와 difficulty_fit이 각각 4 이상' in s
    assert '최소 두 개의 source-supported 관찰/조건' in s
    assert 'R73-DIVERSITY-ATOMIC-CERTIFICATION-1' in s

def test_r73_app_carries_rejected_pairs_between_clicks():
    s=(ROOT/'app.py').read_text(encoding='utf-8')
    assert "R73_REJECTED_PAIRS" in s
    assert 'max_judge_per_domain=2' in s
    assert "forbidden_attempts=st.session_state.get('R73_REJECTED_PAIRS',{})" in s

def test_python_files_parse():
    for p in ROOT.glob('*.py'):
        ast.parse(p.read_text(encoding='utf-8'), filename=str(p))
