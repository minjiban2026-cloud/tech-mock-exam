import unittest
from question_plans import validate_plan
from capability_contracts import _load_r59_selector_json

class R59SelectorSchemaRegression(unittest.TestCase):
    def test_wrapped_source_bindings_are_validated_against_anchor(self):
        anchors=[
            {'id':1,'answer':'A','evidence':'조건 판단 결과 A를 선택한다. 조건 판단을 위해 사용하는 충분히 긴 원문 근거 구절이다. 후속 상황은 별도 조건에서 검토한다.'},
            {'id':2,'answer':'B','evidence':'후속 결과 B를 적용한다. 후속 결과의 이유를 설명하는 충분히 긴 다른 원문 근거 구절이다.'},
        ]
        plan={
            'schema':'SOURCE_BOUND_TASK_PLAN_V1',
            'criterion':{'text':'설명','binding':{'anchor_id':1,'quote':'조건 판단을 위해 사용하는 충분히 긴 원문 근거 구절이다.'}},
            'transfer_condition':{'text':'설명','binding':{'anchor_id':1,'quote':'후속 상황은 별도 조건에서 검토한다.'}},
            'task1':{'result':{'value':'A','binding':{'anchor_id':1,'quote':'A'}},'reason':{'text':'근거','binding':{'anchor_id':1,'quote':'조건 판단을 위해 사용하는 충분히 긴 원문 근거 구절이다.'}}},
            'task2':{'result':{'value':'B','binding':{'anchor_id':2,'quote':'B'}},'reason':{'text':'근거','binding':{'anchor_id':2,'quote':'후속 결과의 이유를 설명하는 충분히 긴 다른 원문 근거 구절이다.'}}},
            'dependency':{'input':'task1.result','output':'task2.result','required_result':{'anchor_id':1,'quote':'A'},'why_required':'첫 판단 결과가 있어야 후속 결과의 적용 대상을 확정할 수 있기 때문이다.'}
        }
        ok,detail=validate_plan(plan,anchors,relation_type='dependent_sequence')
        self.assertTrue(ok,detail)

    def test_multiple_criterion_bindings_are_provenance_checked(self):
        anchors=[
            {'id':1,'answer':'A','evidence':'A를 선택한다. 첫 번째 조건을 판단하는 데 필요한 충분히 긴 원문 근거 문장이다. 후속 적용 조건은 여기에서 별도로 확인한다.'},
            {'id':2,'answer':'B','evidence':'B를 선택한다. 두 번째 대안을 비교하는 데 필요한 충분히 긴 원문 근거 문장이다.'},
        ]
        plan={
            'schema':'SOURCE_BOUND_TASK_PLAN_V1',
            'criterion':{'bindings':[{'anchor_id':1,'quote':'첫 번째 조건을 판단하는 데 필요한 충분히 긴 원문 근거 문장이다.'},{'anchor_id':2,'quote':'두 번째 대안을 비교하는 데 필요한 충분히 긴 원문 근거 문장이다.'}]},
            'transfer_condition':{'binding':{'anchor_id':1,'quote':'후속 적용 조건은 여기에서 별도로 확인한다.'}},
            'task1':{'result':{'binding':{'anchor_id':1,'quote':'A'}},'reason':{'binding':{'anchor_id':1,'quote':'첫 번째 조건을 판단하는 데 필요한 충분히 긴 원문 근거 문장이다.'}}},
            'task2':{'result':{'binding':{'anchor_id':2,'quote':'B'}},'reason':{'binding':{'anchor_id':2,'quote':'두 번째 대안을 비교하는 데 필요한 충분히 긴 원문 근거 문장이다.'}}},
            'dependency':{'input':'task1.result','output':'task2.result','required_result':{'anchor_id':1,'quote':'A'},'why_required':'첫 판단에서 선택한 조건이 있어야 두 번째 대안의 적용 여부를 판단할 수 있기 때문이다.'}
        }
        ok,detail=validate_plan(plan,anchors,relation_type='contrast')
        self.assertTrue(ok,detail)

    def test_known_missing_relation_brace_is_repaired_without_semantic_change(self):
        malformed='{"relations":[{"anchor_ids":[1,2],"source_plan":{"schema":"SOURCE_BOUND_TASK_PLAN_V1"}}],"omissions":[]}'
        obj=_load_r59_selector_json(malformed)
        self.assertEqual(obj['relations'][0]['anchor_ids'],[1,2])
        self.assertEqual(obj['omissions'],[])

if __name__=='__main__': unittest.main()
