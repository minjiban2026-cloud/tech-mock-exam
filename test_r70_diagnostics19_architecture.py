import json
from pathlib import Path
import capability_contracts as cc
import exam_builder as eb

D19=Path('/mnt/data/generation_diagnostics (19).json')

def test_diagnostics19_confirms_r69_stall_and_incremental_skip():
    if not D19.exists():
        import pytest; pytest.skip('diagnostics19 fixture not present')
    d=json.loads(D19.read_text())
    assert d['mode']=='R69_EXTRACTIVE_RESULT_DIVERSITY_GATE'
    assert d['before_inventory']['verified_slots']==11
    assert d['after_inventory']['verified_slots']==11
    assert d['summary']['judge_pass']==0
    logs={x['domain']:x for x in d['domain_logs']}
    assert logs['기술교육론']['need_before']==0 and logs['기술교육론']['judge_tested']==0
    assert logs['발명']['need_before']==0 and logs['발명']['judge_tested']==0
    assert logs['제조기술']['generation']['selector_returned']==0

def test_r70_source_packet_assessment_is_not_old_final_item_gate():
    row={'selection_stage':'SOURCE_PACKET','source_support':5,'construction_potential':4,
         'evidence_diversity':4,'source_hygiene':5,'rote_risk':3,'verdict':'SELECT'}
    assert cc._r70_assessment_pass(row)
    row['construction_potential']=3
    assert not cc._r70_assessment_pass(row)

def test_r70_writer_result_must_be_exact_source_and_pair_not_reused():
    a1={'id':1,'answer':'시준축 오차','topic':'기준 A','evidence':'조건 X에서는 시준축 오차를 제거한다.'}
    a2={'id':2,'answer':'관측자의 읽기 오차','topic':'기준 B','evidence':'후속 절차에서는 관측자의 읽기 오차도 소거한다.'}
    bundle={'anchors':[a1,a2],'contract_type':'contrastive_error_transfer','selector_relation':{'relation_type':'conditional_choice'}}
    raw={'task1_anchor_id':1,'task1_result':'시준축 오차','task2_anchor_id':2,'task2_result':'관측자의 읽기 오차',
         'dependency_reason':'①에서 확인한 소거 기준을 바탕으로 ②에서 추가되는 오차를 비교해야 한다.'}
    built=cc._r70_rebuild_plan_from_writer(raw,bundle)
    assert built is not None and built['relation_type']=='contrast'
    bad=dict(raw,task2_result='새로 만든 기술 사실')
    assert cc._r70_rebuild_plan_from_writer(bad,bundle) is None
    assert cc._r70_rebuild_plan_from_writer(raw,bundle,forbidden_anchor_pairs=[(1,2)]) is None

def test_r70_prompt_separates_source_packet_from_final_quality():
    text=cc._r67_selector_prompt('제조기술',[],2)
    assert 'SOURCE PACKET' in text
    assert '완성된 ①→② dependency가 존재할 필요는 없다' in text
    w=cc._r59_prompt('제조기술',[],[])
    assert 'task1_result' in w and 'allowed_results' in w

def test_r70_version():
    assert eb.BUILDER_API_VERSION=='ACTUAL-EXAM-TRANSFER-R72-20260908'
