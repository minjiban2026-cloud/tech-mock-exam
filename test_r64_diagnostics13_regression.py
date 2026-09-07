import unittest

from capability_contracts import (
    _r62_operation_profile,
    _r64_public_quality_errors,
)


class R64Diagnostics13Regression(unittest.TestCase):
    def test_truncated_numeric_answer_is_not_ranked_as_rich(self):
        profile = _r62_operation_profile({
            "answer": "기울기가 1",
            "evidence": "일반적으로 비탈면의 기울기가 1:1보다 급한 경우에는 돌쌓기를 하",
        })
        self.assertGreaterEqual(profile["fragment_penalty"], 12)

    def test_truncated_fixed_answer_is_rejected_before_judge(self):
        q = {
            "passage": "경사지 A와 C의 조건을 비교하였다.",
            "tasks": ["① 적용 여부를 판단하시오.", "② ①의 기준으로 후속 작업을 판단하시오."],
            "answer": ["기울기가 1\n근거: 원문", "돌쌓기\n근거: 원문"],
        }
        self.assertIn("R64_TRUNCATED_FIXED_ANSWER_1", _r64_public_quality_errors({}, q))

    def test_compound_answer_keywords_are_treated_as_exposure(self):
        q = {
            "passage": "설계 절차와 문제 해결의 구조화 정도를 그대로 대조하였다.",
            "tasks": ["① 명칭을 쓰시오.", "② 해당하는 명칭을 쓰시오."],
            "answer": ["설계와 문제해결의 개념 절차 비교\n근거: 원문", "문제해결 수업모형\n근거: 원문"],
        }
        errors = _r64_public_quality_errors({}, q)
        self.assertIn("R64_COMPOUND_ANSWER_EXPOSED_1", errors)
        self.assertIn("R64_TWO_LABEL_LOOKUPS", errors)

    def test_non_exposed_transfer_item_is_not_false_positive(self):
        q = {
            "passage": "송신 측은 검사 비트를 덧붙이고 수신 측은 일치 여부만 확인하였다. 수신 자료만으로 잘못된 위치를 특정하는 절차는 없다.",
            "tasks": ["① 학생의 판단을 수정하고 근거를 쓰시오.", "② ①의 결과를 적용하여 복원 주장이 옳은지 판단하시오."],
            "answer": ["수직 중복 검사(VRC)\n근거: 원문", "오류 검출 코드\n근거: 원문"],
        }
        self.assertEqual([], _r64_public_quality_errors({}, q))


if __name__ == "__main__":
    unittest.main()
