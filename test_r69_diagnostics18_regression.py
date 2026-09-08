import json
from pathlib import Path
import capability_contracts as cc
import exam_builder as eb

DIAG=Path('/mnt/data/generation_diagnostics (18).json')

def test_diag18_reproduces_stall_and_quarantine():
    d=json.loads(DIAG.read_text(encoding='utf-8'))
    assert d['mode']=='R68_MULTI_ANCHOR_COVERAGE_AWARE_GATE'
    assert d['before_inventory']['verified_slots']==11
    assert d['after_inventory']['verified_slots']==11
    assert len(d['before_inventory']['quarantined_verified'])==3
    assert d['summary']['judge_tested']==1
    assert d['summary']['judge_pass']==0

def test_r69_exact_result_must_be_source_extract():
    a={'answer':'큰 제목','topic':'큰 제목','evidence':'조건 A에서는 관측자의 읽기 오차가 소거된다.'}
    assert cc._r69_exact_result('관측자의 읽기 오차',a)=='관측자의 읽기 오차'
    assert cc._r69_exact_result('임의로 만든 정답',a) is None

def test_r69_composed_can_score_exact_evidence_phrase():
    cluster={'score':10,'anchors':[
        {'id':1,'answer':'방법 A','topic':'방법 A','evidence':'조건 A에서 시준축 오차를 제거한다.','page_no':1},
        {'id':2,'answer':'방법 B','topic':'방법 B','evidence':'조건 B에서 관측자의 읽기 오차를 제거한다.','page_no':2},
    ]}
    plan={'verdict':'SELECT','source_support':5,'dependency':5,'inferential_distance':4,'transferability':5,'rote_risk':1,
          'task1_anchor_id':1,'task2_anchor_id':2,'task1_result':'시준축 오차','task2_result':'관측자의 읽기 오차',
          'support_anchor_ids':[],'contract_type':'contrastive_error_transfer','relation_type':'contrast',
          'dependency_reason':'①에서 판별한 오차 소거 기준을 ②의 변경된 관측 조건에 다시 적용해야 결과를 정할 수 있다.'}
    row=cc._r68_build_composed_relation(plan,cluster)
    assert row is not None
    assert row['source_plan']['task1']['result']['value']=='시준축 오차'
    assert row['source_plan']['task2']['result']['value']=='관측자의 읽기 오차'

def test_r69_prompt_requests_extractive_results():
    prompt=cc._r67_selector_prompt('건설기술',[],1,clusters=[])
    assert 'task1_result' in prompt and 'task2_result' in prompt
    assert '정확한 부분문자열' in prompt

def test_r69_versions():
    assert eb.BUILDER_API_VERSION=='ACTUAL-EXAM-TRANSFER-R73-20260908'
