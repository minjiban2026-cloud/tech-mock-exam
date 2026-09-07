"""Offline regression of the reported zero pools and definition-recognition failure.
Plan fixtures below check provenance/wiring only; they are not exam questions.
"""
import copy,json,unittest
from unittest.mock import patch
from types import SimpleNamespace
import capability_contracts as cc
import exam_builder as eb
from question_plans import validate_plan,PLAN_SCHEMA
from test_stabilization import fixture,DB,review


def plan_for(anchors):
    a,b=anchors[:2]
    def spans(anchor):
        text=cc._clean(anchor['evidence'])
        # Two source excerpts for exercising bindings, not an assertion of reasoning validity.
        size=max(12,len(text)*3//4)
        return ({'anchor_id':anchor['id'],'quote':text[:size]},
                {'anchor_id':anchor['id'],'quote':text[-size:]})
    ar,ag=spans(a);br,bg=spans(b)
    return {'schema':PLAN_SCHEMA,'criterion':ag,'transfer_condition':bg,
            'task1':{'result':ar,'reason':ag},'task2':{'result':br,'reason':bg},
            'dependency':{'input':'task1.result','output':'task2.result','required_result':ar,
                          'why_required':'테스트 전용으로 첫 소문항의 결과를 다음 소문항의 입력으로 연결한다.'}}


def fake_client(responses):
    iterator=iter(responses)
    def create(**kwargs):
        row=next(iterator)
        if isinstance(row,Exception):raise row
        return SimpleNamespace(output_text=json.dumps(row,ensure_ascii=False))
    return SimpleNamespace(responses=SimpleNamespace(create=create))


class GenerationDiagnosticsTests(unittest.TestCase):
    def setUp(self):
        self.c=fixture(eb.DOMAINS[0]);self.anchors=self.c['validation']['anchors']
        self.plan=plan_for(self.anchors)

    def test_source_plan_and_invalid_quote(self):
        ok,d=validate_plan(self.plan,self.anchors);self.assertTrue(ok,d)
        bad=copy.deepcopy(self.plan);bad['task2']['result']['quote']='원자료에 없는 새 기술적 효과를 임의로 주장한다.'
        self.assertFalse(validate_plan(bad,self.anchors)[0])

    def test_independent_dependency_rejected(self):
        bad=copy.deepcopy(self.plan);bad['dependency']['input']='independent_fact'
        self.assertFalse(validate_plan(bad,self.anchors)[0])
        bad=copy.deepcopy(self.plan);bad['dependency']['required_result']=bad['task2']['result']
        self.assertFalse(validate_plan(bad,self.anchors)[0])

    def test_python_miner_empty_is_visible(self):
        with patch.object(cc,'_r60_python_relation_candidates',return_value=(self.anchors,[])):
            pool=cc.synthesize_r59_pool('mock','mock',DB,self.c['domain'],1)
        self.assertFalse(pool);self.assertEqual(pool.diagnostics['writer_calls'],0)
        self.assertIn('python_relation_miner:NO_RELATION_CANDIDATE',pool.diagnostics['failure_counts'])
        self.assertEqual(pool.diagnostics['selector_calls'],0)

    def test_python_miner_requires_no_selector_api(self):
        with patch.object(cc,'_r60_python_relation_candidates',return_value=(self.anchors,[])),patch('openai.OpenAI') as client:
            pool=cc.synthesize_r59_pool('mock','mock',DB,self.c['domain'],1)
        client.assert_not_called();self.assertFalse(pool)

    def bundle(self):
        return {'anchors':self.anchors,'source_plan':self.plan,'fixed_answers':validate_plan(self.plan,self.anchors)[1]['answers'],
                'contract_type':cc.R59_ALLOWED_TYPES[0],'selector_relation':{'relation_type':'contrast'}}

    def test_writer_omission_and_timeout_are_different(self):
        for response,expected in [({'contracts':[],'omissions':[{'bundle_id':0,'reason':'no coherent task'}]},'writer:NO_QUESTION_WRITTEN'),
                                  (TimeoutError(),'writer_call:TimeoutError')]:
            with self.subTest(expected=expected),patch.object(cc,'_r59_select_bundles',return_value=[self.bundle()]),patch('openai.OpenAI',return_value=fake_client([response])):
                pool=cc.synthesize_r59_pool('mock','mock',DB,self.c['domain'],1)
                self.assertIn(expected,pool.diagnostics['failure_counts'])

    def test_writer_cannot_replace_fixed_answer(self):
        candidate=dict(self.c,bundle_id=0,exact_answers=['wrong','answer'])
        with patch.object(cc,'_r59_select_bundles',return_value=[self.bundle()]),patch('openai.OpenAI',return_value=fake_client([{'contracts':[candidate]}])):
            pool=cc.synthesize_r59_pool('mock','mock',DB,self.c['domain'],1)
        self.assertFalse(pool)
        self.assertIn('writer_schema:WRITER_CHANGED_FIXED_ANSWERS',pool.diagnostics['failure_counts'])

    def test_raw_count_separate_from_validated_count(self):
        candidate=dict(self.c,bundle_id=0);candidate.pop('exact_answers')
        with patch.object(cc,'_r59_select_bundles',return_value=[self.bundle()]),patch('openai.OpenAI',return_value=fake_client([{'contracts':[candidate]}])),patch.object(cc,'r59_prejudge_errors',return_value=[]):
            pool=cc.synthesize_r59_pool('mock','mock',DB,self.c['domain'],1)
        self.assertEqual(pool.diagnostics['writer_returned'],1)
        self.assertEqual(pool.diagnostics['python_validated'],1)

    def test_reported_definition_failure_blocked_before_judge(self):
        # Reconstruct the failure class from DB source; the original question was not saved by R59.
        import sqlite3
        con=sqlite3.connect(DB.as_uri()+'?mode=ro',uri=True);con.row_factory=sqlite3.Row
        anchors=[dict(r) for r in con.execute("select * from anchors where domain=? and (replace(answer,' ','')=? or replace(answer,' ','')=?)",('제조기술','냉각곡선','개량처리'))]
        con.close();self.assertGreaterEqual(len(anchors),2)
        clues=[{'side':side,'anchor_id':a['id'],'text':a['evidence'].replace(a['answer'],'[항목]')} for side,a in zip(('A','B'),anchors)]
        c={'clues':clues,'validation':{'anchors':anchors}}
        q={'points':4,'passage':' '.join(x['text'] for x in clues),'tasks':['판단 오류를 수정하고 근거를 설명하시오.','앞의 판단을 적용하고 근거를 설명하시오.'],
           'answer':[a['answer'] for a in anchors]}
        errors=cc.r59_prejudge_errors(c,q)
        self.assertTrue(any(x.startswith('DEFINITION_RECOGNITION') for x in errors),errors)
        self.assertTrue(any(x.startswith('LABEL_ONLY_SCORING') for x in errors),errors)
        with patch.object(cc,'synthesize_r59_pool',return_value=[self.c]),patch.object(eb,'judge_question') as judge:
            run=eb.certify_r59_missing_slots(DB,[],domains=[self.c['domain']],api_key='mock',seed=0)
        judge.assert_not_called();self.assertTrue(run['domain_logs'][0]['pre_judge_rejections'])

    def test_judge_reject_keeps_question_and_raw_review(self):
        # Only state/reporting wiring is mocked here; no quality approval is claimed.
        rv=dict(review(),**{'pass':False,'fatal_flags':['ROTE_ONLY'],'weakest_point':'mock weakness'})
        with patch.object(cc,'synthesize_r59_pool',return_value=[self.c]),patch.object(cc,'r59_prejudge_errors',return_value=[]),patch.object(eb,'judge_question',return_value=rv):
            run=eb.certify_r59_missing_slots(DB,[],domains=[self.c['domain']],api_key='mock',seed=1)
        row=run['reviews'][0]
        self.assertEqual(row['question']['answer'],self.c['exact_answers'])
        self.assertEqual(row['judge_review']['weakest_point'],'mock weakness')
        json.dumps(run,ensure_ascii=False)

    def test_selector_writer_answer_contract_roundtrip(self):
        # R60 has no AI selector; fixed source-plan ownership still survives Writer.
        candidate=dict(self.c,bundle_id=0);candidate.pop('exact_answers')
        client=fake_client([{'contracts':[candidate]}])
        with patch.object(cc,'_r59_select_bundles',return_value=[self.bundle()]),patch('openai.OpenAI',return_value=client),patch.object(cc,'r59_prejudge_errors',return_value=[]):
            pool=cc.synthesize_r59_pool('mock','mock',DB,self.c['domain'],1)
        self.assertEqual(len(pool),1,pool.diagnostics)
        self.assertEqual(pool[0]['exact_answers'],validate_plan(self.plan,self.anchors)[1]['answers'])
        q=cc.r59_contract_to_question(pool[0]);self.assertEqual(len(q['rubric']),2)
        self.assertEqual(pool.diagnostics['python_validated'],1)

    def test_non_object_contract_rejected(self):
        for bad in (None, [], {'clues':[None]}):
            self.assertFalse(cc.validate_r59_contract(DB,eb.DOMAINS[0],bad)[0])

if __name__=='__main__':unittest.main(verbosity=2)
