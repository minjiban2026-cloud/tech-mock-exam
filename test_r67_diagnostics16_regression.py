import sys
import types
from unittest.mock import patch
import capability_contracts as cc


def _candidate(aid1=1, aid2=2):
    a1={'id':aid1,'domain':'제조기술','topic':'A','answer':'정답A','evidence':'조건과 절차를 비교하여 첫 번째 판단 기준을 적용한다.','source_name':'x.pdf','page_no':1}
    a2={'id':aid2,'domain':'제조기술','topic':'B','answer':'정답B','evidence':'첫 판단 기준을 바탕으로 변경 조건에서 후속 결과를 결정한다.','source_name':'x.pdf','page_no':2}
    plan={'schema':'SOURCE_BOUND_TASK_PLAN_V1',
          'criterion':{'text':a1['evidence'],'binding':{'anchor_id':aid1,'quote':a1['evidence']}},
          'transfer_condition':{'text':a2['evidence'],'binding':{'anchor_id':aid2,'quote':a2['evidence']}},
          'task1':{'result':{'value':'정답A','binding':{'anchor_id':aid1,'quote':a1['evidence']}},'reason':{'text':a1['evidence'],'binding':{'anchor_id':aid1,'quote':a1['evidence']}}},
          'task2':{'result':{'value':'정답B','binding':{'anchor_id':aid2,'quote':a2['evidence']}},'reason':{'text':a2['evidence'],'binding':{'anchor_id':aid2,'quote':a2['evidence']}}},
          'dependency':{'input':'task1.result','output':'task2.result','required_result':{'anchor_id':aid1,'quote':a1['evidence']},'why_required':'①에서 판별한 결과를 후속 상황의 판단 대상으로 사용해야 ②의 적용 결과를 결정할 수 있다.'}}
    return [a1,a2], {'score':20,'anchors':[a1,a2],'anchor_ids':[aid1,aid2],'page_span':1,'bridge_terms':[],
                    'direct_crossref_score':0,'operation_score':5,'operation_profiles':[{'richness':4,'fragment_penalty':0},{'richness':4,'fragment_penalty':0}],
                    'relation_type':'dependent_sequence','contract_type':'criterion_conflict_resolution','source_plan':plan,
                    'fixed_answers':['정답A\n근거: '+a1['evidence'],'정답B\n근거: '+a2['evidence']],
                    'master_relation':'Python source-grounded relation candidate','reasoning_viability':10}


def _fake_openai(output_text):
    class Responses:
        def create(self, **kwargs):
            return types.SimpleNamespace(output_text=output_text)
    class OpenAI:
        def __init__(self, **kwargs): self.responses=Responses()
    return types.SimpleNamespace(OpenAI=OpenAI)


def test_semantic_empty_is_not_repopulated_by_python_fallback():
    anchors,cand=_candidate()
    payload='{"assessments":[{"candidate_id":0,"source_support":3,"dependency":2,"inferential_distance":1,"transferability":2,"rote_risk":5,"verdict":"REJECT","reason":"rote"}],"selected_ids":[],"reserve_ids":[],"rejected":{"0":"rote"}}'
    with patch.object(cc,'_r60_python_relation_candidates',return_value=(anchors,[cand])), patch.dict(sys.modules,{'openai':_fake_openai(payload)}):
        pool=cc._r59_select_bundles('key','gpt-5.6-luna','unused.db','제조기술',wanted=2)
    assert len(pool)==0
    assert pool.diagnostics.get('selector_empty_reason')=='SEMANTICALLY_NO_4PT_WORTHY_PAIR'
    assert pool.diagnostics.get('selector_fallback') is None
    assert pool.diagnostics.get('selector_returned')==0


def test_only_assessment_threshold_pass_candidates_reach_writer_pool():
    anchors,cand=_candidate()
    payload='{"assessments":[{"candidate_id":0,"source_support":5,"dependency":4,"inferential_distance":4,"transferability":4,"rote_risk":1,"verdict":"SELECT","reason":"good"}],"selected_ids":[0],"reserve_ids":[],"rejected":{}}'
    with patch.object(cc,'_r60_python_relation_candidates',return_value=(anchors,[cand])), patch.dict(sys.modules,{'openai':_fake_openai(payload)}):
        pool=cc._r59_select_bundles('key','gpt-5.6-luna','unused.db','제조기술',wanted=2)
    assert len(pool)==1
    sem=pool[0]['selector_relation']['semantic_assessment']
    assert sem['inferential_distance']==4 and sem['rote_risk']==1


def test_prejudge_rejects_unpresented_numbered_answer_structure():
    c={'selector_relation':{'semantic_assessment':{'source_support':5,'dependency':4,'inferential_distance':4,'transferability':4,'rote_risk':1,'verdict':'SELECT'}},
       'source_plan':{},'student_claim':'학생은 잘못 판단하였다.','public_clues':False,'validation':{'anchors':[]}}
    q={'passage':'시료의 변화와 조건을 관찰하였다.','tasks':['① 자료의 미생물 목록에서 번호와 명칭을 쓰시오.','② ①을 적용하시오.'],
       'answer':['4) 박테리아','시아노박테리아']}
    errs=cc.r59_prejudge_errors(c,q)
    assert 'R67_UNSUPPORTED_PUBLIC_NUMBERING' in errs


def test_selector_prompt_requires_all_4point_axes():
    p=cc._r67_selector_prompt('생명기술',[],4)
    for token in ('source_support>=4','dependency>=4','inferential_distance>=4','transferability>=4','rote_risk<=1'):
        assert token in p
    assert '정상 결과' in p

def test_truncated_answer_and_predicate_fragments_are_source_vetoed():
    assert cc._obviously_incomplete_answer('수직 중복 검사(VRC')
    assert not cc._obviously_incomplete_answer('오류 검출 코드')
    assert cc._obviously_incomplete_evidence('- 광포화점 : 어느 한계에 다다르면 더 이')


def test_local_recall_expands_biology_but_keeps_source_grounding():
    _,rows=cc._r60_python_relation_candidates('knowledge.db','생명기술',limit=180,max_candidates=48)
    assert len(rows)>=10
    assert any(r.get('master_relation')=='R67 operational local recall candidate' for r in rows)
    assert all(len(r.get('anchors') or [])==2 for r in rows)
