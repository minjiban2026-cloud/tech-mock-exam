import json
from pathlib import Path
import capability_contracts as cc
import exam_builder as eb

D20=Path('/mnt/data/generation_diagnostics (20).json')

def test_diagnostics20_confirms_selector_bottleneck_moved_downstream():
    if not D20.exists():
        import pytest; pytest.skip('diagnostics20 fixture not present')
    d=json.loads(D20.read_text())
    assert d['mode']=='R70_SOURCE_FACT_INSTANCE_SEPARATED_GATE'
    assert d['before_inventory']['verified_slots']==11
    assert d['after_inventory']['verified_slots']==11
    logs={x['domain']:x for x in d['domain_logs']}
    assert logs['제조기술']['generation']['selector_returned']>=1
    assert logs['제조기술']['generation']['writer_returned']>=1
    assert logs['제조기술']['judge_tested']>=1
    assert logs['건설기술']['generation']['selector_returned']>=1

def test_r71_atomic_candidates_are_exact_and_reject_truncated_anchor():
    a={'id':1,'answer':'복합 재료의 성질에 영향을 주는 인자',
       'evidence':'- 복합 재료의 성질에 영향을 주는 인자 : 분산 입자의 크기, 형상 및 양'}
    vals=cc._r71_atomic_result_candidates(a)
    assert '분산 입자의 크기, 형상 및 양' in vals
    assert all(cc._norm(x) in cc._norm(a['evidence']) for x in vals)
    broken={'id':2,'answer':'기울기가 1','evidence':'- 기울기가 1:4보다 급한 경사지에 흙쌓기를 할 때에는 땅바닥에 층따기를 함 (최소높이'}
    assert cc._r71_atomic_result_candidates(broken)==[]

def test_r71_rebuild_accepts_only_python_atomic_result_and_forbids_pair():
    a1={'id':1,'answer':'시준축 오차','topic':'기준 A','evidence':'조건 X에서는 시준축 오차를 제거한다.'}
    a2={'id':2,'answer':'관측자의 읽기 오차','topic':'기준 B','evidence':'후속 절차에서는 관측자의 읽기 오차도 소거한다.'}
    bundle={'anchors':[a1,a2],'contract_type':'contrastive_error_transfer','selector_relation':{'relation_type':'conditional_choice'}}
    raw={'task1_anchor_id':1,'task1_result':'시준축 오차','task2_anchor_id':2,'task2_result':'관측자의 읽기 오차',
         'dependency_reason':'①에서 확인한 소거 기준을 바탕으로 ②에서 추가되는 오차를 비교해야 한다.'}
    built=cc._r70_rebuild_plan_from_writer(raw,bundle)
    assert built is not None and built['relation_type']=='contrast'
    assert cc._r70_rebuild_plan_from_writer(dict(raw,task2_result='읽기 오차'),bundle) is None
    assert cc._r70_rebuild_plan_from_writer(raw,bundle,forbidden_anchor_pairs=[(1,2)]) is None

def test_r71_writer_prompt_contains_atomic_whitelist_and_anti_predecision_rules():
    a={'id':1,'answer':'시준축 오차','topic':'오차','evidence':'시준축 오차를 제거한다.'}
    b={'anchors':[a],'selector_relation':{},'source_plan':{}}
    w=cc._r59_prompt('건설기술',[b],[])
    assert 'allowed_results' in w
    assert '판정을 자료가 미리 확정' in w
    assert 'variant_id 0,1' in w

def test_r71_version():
    assert eb.BUILDER_API_VERSION=='ACTUAL-EXAM-TRANSFER-R73-20260908'
