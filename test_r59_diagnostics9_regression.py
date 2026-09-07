import unittest

import capability_contracts as cc
from question_plans import validate_plan


class Diagnostics9RegressionTests(unittest.TestCase):
    def test_selector_recovers_repeated_missing_relation_closers(self):
        raw='''{"relations":[{"anchor_ids":[1,2],"relation_type":"dependent_sequence","contract_type":"criterion_conflict_resolution","master_relation":"x","source_plan":{"schema":"SOURCE_BOUND_TASK_PLAN_V1","criterion":{"text":"aaaa","binding":{"anchor_id":1,"quote":"aaaa"}},"transfer_condition":{"text":"bbbb","binding":{"anchor_id":2,"quote":"bbbb"}},"task1":{"result":{"value":"A","binding":{"anchor_id":1,"quote":"A"}},"reason":{"text":"aaaa","binding":{"anchor_id":1,"quote":"aaaa"}}},"task2":{"result":{"value":"B","binding":{"anchor_id":2,"quote":"B"}},"reason":{"text":"bbbb","binding":{"anchor_id":2,"quote":"bbbb"}}},"dependency":{"input":"task1.result","output":"task2.result","required_result":{"anchor_id":1,"quote":"A"},"why_required":"task1 result is needed for the second decision"}}, {"anchor_ids":[3,4],"relation_type":"dependent_sequence","contract_type":"criterion_conflict_resolution","master_relation":"y","source_plan":{"schema":"SOURCE_BOUND_TASK_PLAN_V1","criterion":{"text":"cccc","binding":{"anchor_id":3,"quote":"cccc"}},"transfer_condition":{"text":"dddd","binding":{"anchor_id":4,"quote":"dddd"}},"task1":{"result":{"value":"C","binding":{"anchor_id":3,"quote":"C"}},"reason":{"text":"cccc","binding":{"anchor_id":3,"quote":"cccc"}}},"task2":{"result":{"value":"D","binding":{"anchor_id":4,"quote":"D"}},"reason":{"text":"dddd","binding":{"anchor_id":4,"quote":"dddd"}}},"dependency":{"input":"task1.result","output":"task2.result","required_result":{"anchor_id":3,"quote":"C"},"why_required":"task1 result is needed for the second decision"}}],"omissions":[]}'''
        obj=cc._load_r59_selector_json(raw)
        self.assertEqual([x['anchor_ids'] for x in obj['relations']], [[1,2],[3,4]])

    def test_paraphrased_result_normalizes_to_canonical_anchor_label(self):
        anchors=[
            {'id':610,'answer':'찬성','topic':'찬성','evidence':'- 찬성 : 병충해 피해가 적어서 수확량이 많다'},
            {'id':611,'answer':'반대','topic':'반대','evidence':'- 반대 : 아직 유해성이 검증되지 않았다'},
        ]
        plan={
            'schema':'SOURCE_BOUND_TASK_PLAN_V1',
            'criterion':{'text':'생산성 향상을 판단한다','binding':{'anchor_id':610,'quote':'병충해 피해가 적어서 수확량이 많다'}},
            'transfer_condition':{'text':'안전성 우려를 판단한다','binding':{'anchor_id':611,'quote':'아직 유해성이 검증되지 않았다'}},
            'task1':{'result':{'value':'수확량 증가를 근거로 한 찬성 판단','binding':{'anchor_id':610,'quote':'병충해 피해가 적어서 수확량이 많다'}},'reason':{'text':'수확량이 증가한다','binding':{'anchor_id':610,'quote':'병충해 피해가 적어서 수확량이 많다'}}},
            'task2':{'result':{'value':'유해성 미검증을 근거로 한 반대 판단','binding':{'anchor_id':611,'quote':'아직 유해성이 검증되지 않았다'}},'reason':{'text':'유해성이 검증되지 않았다','binding':{'anchor_id':611,'quote':'아직 유해성이 검증되지 않았다'}}},
            'dependency':{'input':'task1.result','output':'task2.result','required_result':{'anchor_id':610,'quote':'병충해 피해가 적어서 수확량이 많다'},'why_required':'첫 판단 결과를 바탕으로 반대 근거와 대조하여 후속 판단을 수행한다.'}
        }
        ok,out=validate_plan(plan,anchors,relation_type='dependent_sequence')
        self.assertTrue(ok,out)
        self.assertEqual(out['rubric'][0]['result'],'찬성')
        self.assertEqual(out['rubric'][1]['result'],'반대')

    def test_hidden_transfer_can_name_task2_but_public_gate_must_handle_leak(self):
        anchors=[
            {'id':1,'answer':'트리밍','topic':'트리밍','evidence':'트리밍: 금속 시트를 자르는 작업'},
            {'id':2,'answer':'셰이빙','topic':'셰이빙','evidence':'셰이빙: 이미 가공된 부품 가장자리를 다듬는 작업'},
        ]
        plan={
            'schema':'SOURCE_BOUND_TASK_PLAN_V1',
            'criterion':{'text':'금속 시트인지 판단','binding':{'anchor_id':1,'quote':'금속 시트를 자르는 작업'}},
            'transfer_condition':{'text':'셰이빙 조건을 후속 판정','binding':{'anchor_id':2,'quote':'이미 가공된 부품 가장자리를 다듬는 작업'}},
            'task1':{'result':{'value':'트리밍','binding':{'anchor_id':1,'quote':'트리밍'}},'reason':{'text':'금속 시트 절단','binding':{'anchor_id':1,'quote':'금속 시트를 자르는 작업'}}},
            'task2':{'result':{'value':'셰이빙','binding':{'anchor_id':2,'quote':'셰이빙'}},'reason':{'text':'가공 부품 가장자리 정리','binding':{'anchor_id':2,'quote':'이미 가공된 부품 가장자리를 다듬는 작업'}}},
            'dependency':{'input':'task1.result','output':'task2.result','required_result':{'anchor_id':1,'quote':'트리밍'},'why_required':'트리밍을 먼저 판별해야 대비되는 후속 공정을 구별하여 적용할 수 있다.'}
        }
        ok,out=validate_plan(plan,anchors,relation_type='contrast')
        self.assertTrue(ok,out)

    def test_positive_misclassification_is_not_rejected_for_missing_negation_word(self):
        c={
            'student_claim':'학생은 금속 시트 절단 작업을 이미 가공된 부품의 가장자리를 다듬는 공정으로 판단하였다.',
            'source_plan':None,
        }
        q={'tasks':[],'answer':[],'intro':'','passage':''}
        errs=cc.r59_prejudge_errors(c,q)
        self.assertNotIn('NO_ACTUAL_ERROR_CLAIM',errs)
        self.assertNotIn('CLAIM_LACKS_JUDGMENT',errs)


if __name__=='__main__':
    unittest.main()
