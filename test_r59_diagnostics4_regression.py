import unittest

from question_plans import validate_plan
from capability_contracts import _r59_span_bridge, r59_contract_to_question, r59_prejudge_errors


class Diagnostics4RegressionTests(unittest.TestCase):
    def test_hidden_binding_quote_does_not_false_leak_transfer(self):
        anchors=[
            {'id':485,'answer':'영국','topic':'영국','evidence':'- 영국: 문제해결과설계활동을중심으로한과정중심(process oriented) 교육과정이바탕'},
            {'id':483,'answer':'기술 vs 설계·기술','topic':'기술 vs 설계·기술','evidence':'■ 기술 vs 설계·기술 기술(Technology) ☞ 미국 설계·기술(Design&Technology) ☞ 영국 교과대상'},
        ]
        plan={
            'schema':'SOURCE_BOUND_TASK_PLAN_V1',
            'criterion':{'text':'문제해결과 설계활동을 중심으로 한 과정중심 교육과정이면 영국을 선택한다.','binding':{'anchor_id':485,'quote':'영국: 문제해결과설계활동을중심으로한과정중심(process oriented) 교육과정이바탕'}},
            'transfer_condition':{'text':'선택한 국가를 국가별 교과 대상 명칭의 대응 판단에 적용한다.','binding':{'anchor_id':483,'quote':'기술(Technology) ☞ 미국 설계·기술(Design&Technology) ☞ 영국 교과대상'}},
            'task1':{'result':{'value':'영국','binding':{'anchor_id':485,'quote':'영국'}},'reason':{'text':'과정중심 교육과정 특성','binding':{'anchor_id':485,'quote':'문제해결과설계활동을중심으로한과정중심(process oriented) 교육과정이바탕'}}},
            'task2':{'result':{'value':'설계·기술(Design&Technology)','binding':{'anchor_id':483,'quote':'설계·기술(Design&Technology)'}},'reason':{'text':'국가별 교과 대상 대응','binding':{'anchor_id':483,'quote':'기술(Technology) ☞ 미국 설계·기술(Design&Technology) ☞ 영국 교과대상'}}},
            'dependency':{'input':'task1.result','output':'task2.result','required_result':{'anchor_id':485,'quote':'영국'},'why_required':'①에서 국가를 영국으로 확정해야 국가별 대응표에서 해당 교과 대상을 선택할 수 있다.'}
        }
        ok, detail=validate_plan(plan,anchors,relation_type='conditional_choice')
        self.assertTrue(ok,detail)

    def test_semantic_bridge_allows_moderate_page_span(self):
        anchors=[
            {'id':625,'page_no':20,'answer':'펄라이트경','topic':'펄라이트경','evidence':'펄라이트를 배지로 사용하는 방식, 통기 배수가 좋음'},
            {'id':641,'page_no':30,'answer':'꺾꽂이(삽목)','topic':'꺾꽂이','evidence':'영양기관을 분리한 후 모래나 펄라이트 등의 모판에 심음'},
        ]
        ok,span,bridge=_r59_span_bridge(anchors)
        self.assertTrue(ok)
        self.assertEqual(span,10)
        self.assertTrue(any('펄라이트' in x for x in bridge),bridge)

    def test_far_unrelated_pair_still_rejected(self):
        anchors=[
            {'id':1163,'page_no':107,'answer':'냉각 곡선','topic':'냉각 곡선','evidence':'금속을 용융 상태에서 냉각할 때 온도와 시간의 관계를 나타낸 곡선'},
            {'id':1252,'page_no':160,'answer':'해결방안','topic':'압탕','evidence':'압탕을 설치하면 압탕부에 수축공이 발생하기 때문에 주조 후 제거'},
        ]
        ok,span,bridge=_r59_span_bridge(anchors)
        self.assertFalse(ok)
        self.assertGreater(span,18)

    def test_transport_visible_answer_leak_remains_blocked(self):
        c={
            'status':'R59_PYTHON_VALIDATED','domain':'수송기술','topic':'가역 과정과 카르노 사이클의 판별',
            'clues':[{'side':'A','anchor_id':267,'text':'손실과 마찰을 고려하지 않는 이상적 과정이다.'},
                     {'side':'A','anchor_id':267,'text':'원래 상태로 완전히 복원될 수 있다.'},
                     {'side':'B','anchor_id':271,'text':'모든 과정은 가역 과정이며 이론적으로 최대 효율이다.'},
                     {'side':'B','anchor_id':271,'text':'두 개의 등온 과정과 두 개의 단열 과정으로 구성된다.'}],
            'student_claim':'A 상태변화의 성격을 판단하였다.','transfer_case':'A에서 판단한 과정이 모두 적용되는 후속 사이클이다.',
            'tasks':['① A가 어떤 과정인지 판단하고 근거를 쓰시오.','② ①의 결과를 사용해 사이클 명칭과 근거를 쓰시오.'],
            'exact_answers':['가역 과정\n근거: 원래 상태로 완벽히 복원될 수 있는 이상적인 과정','카르노 사이클\n근거: 모든 과정이 가역 과정인 이론적으로 최대 효율의 사이클'],
            'source_plan':{'schema':'SOURCE_BOUND_TASK_PLAN_V1','criterion':{'text':'손실과 마찰이 없는 이상적 과정을 선택한다.','binding':{'anchor_id':267,'quote':'상태변화에서 발생하는 손실(마찰)이 없다고 가정하는 과정.'}},'transfer_condition':{'text':'후속 사이클의 구성을 판정한다.','binding':{'anchor_id':271,'quote':'이상 기체의 사이클(2개의 등온, 2개의 단열 과정)'}},'task1':{'result':{'value':'가역 과정','binding':{'anchor_id':267,'quote':'가역 과정'}},'reason':{'text':'복원 가능','binding':{'anchor_id':267,'quote':'과정이 완료된 후에 시스템이 원래 상태 로 완벽히 복원될 수 있는 이상적인 과정'}}},'task2':{'result':{'value':'카르노 사이클','binding':{'anchor_id':271,'quote':'카르노 사이클'}},'reason':{'text':'최대 효율','binding':{'anchor_id':271,'quote':'모든 과정이 가역 과정인 이론적으로 최대 효율의 사이클'}}},'dependency':{'input':'task1.result','output':'task2.result','required_result':{'anchor_id':267,'quote':'가역 과정'},'why_required':'①에서 과정의 성격을 판정해야 ②에서 모든 과정이 같은 성격인지 확인하여 사이클을 판별할 수 있다.'}},
            'selector_relation':{'relation_type':'dependent_sequence'},
            'validation':{'anchors':[{'id':267,'answer':'열역학적 과정','evidence':'가역 과정 : 상태변화에서 발생하는 손실(마찰)이 없다고 가정하는 과정. 과정이 완료된 후에 시스템이 원래 상태 로 완벽히 복원될 수 있는 이상적인 과정'},{'id':271,'answer':'카르노 사이클','evidence':'카르노 사이클 이상 기체의 사이클(2개의 등온, 2개의 단열 과정) 모든 과정이 가역 과정인 이론적으로 최대 효율의 사이클'}]}
        }
        q=r59_contract_to_question(c)
        self.assertNotIn('FIXED_RESULT_LEAK_1',r59_prejudge_errors(c,q))


if __name__=='__main__':
    unittest.main()
