import json
import unittest
from pathlib import Path
from unittest.mock import patch

import archive_store
import capability_contracts as cc
from certification_state import verified_contract_receipt

ROOT=Path(__file__).parent
DB=ROOT/"knowledge.db"


class Diagnostics11Regression(unittest.TestCase):
    def test_bootstrap_keeps_prior_live_passes(self):
        obj=json.loads((ROOT/"certified_contracts_bootstrap.json").read_text(encoding="utf-8"))
        rows=obj["contracts"]
        self.assertEqual({r["domain"] for r in rows},{"통신기술","재료역학"})
        self.assertTrue(all(verified_contract_receipt(r) for r in rows))
        import exam_builder
        inv=cc.combined_coverage_inventory(
            DB,rows,
            ["기술교육론","발명","제조기술","건설기술","생명기술","전기·전자","통신기술","재료역학","수송기술"],
            exam_builder.FORMULA_DOMAINS,
        )
        self.assertEqual(inv["verified_slots"],5)
        self.assertEqual(len(inv.get("quarantined_verified",[])),2)

    def test_judgment_marker_accepts_panjeong(self):
        c={
            "contract_type":"criterion_conflict_resolution",
            "cited_anchor_ids":[162,166],
            "exact_answers":[
                "수직 중복 검사(VRC\n근거: - 수직 중복 검사(VRC: Vertical Redundancy Check) = 패리티 검사",
                "오류 검출 코드\n근거: ● 오류 검출 코드 수직 중복 검사(VRC) : 패리티 검사",
            ],
            "clues":[],
            "tasks":[
                "① 학생의 판정을 바로잡고 근거를 쓰시오.",
                "② ①에서 판별한 결과를 적용하여 후속 분류와 근거를 쓰시오.",
            ],
            "reasoning_chain":["자료 분석","중간 판단","후속 적용"],
            "student_claim":"학생은 이를 다른 방식이라고 판정하였다.",
            "transfer_case":"①에서 판별한 결과를 후속 오류 확인 상황에 적용한다.",
            "task2_uses_task1":True,
            "domain":"통신기술",
            "public_clues":False,
            "source_plan":{
                "schema":"SOURCE_BOUND_TASK_PLAN_V1",
                "criterion":{"text":"- 수직 중복 검사(VRC: Vertical Redundancy Check) = 패리티 검사",
                             "binding":{"anchor_id":162,"quote":"- 수직 중복 검사(VRC: Vertical Redundancy Check) = 패리티 검사"}},
                "transfer_condition":{"text":"● 오류 검출 코드 수직 중복 검사(VRC) : 패리티 검사",
                                      "binding":{"anchor_id":166,"quote":"● 오류 검출 코드 수직 중복 검사(VRC) : 패리티 검사"}},
                "task1":{"result":{"value":"수직 중복 검사(VRC","binding":{"anchor_id":162,"quote":"- 수직 중복 검사(VRC: Vertical Redundancy Check) = 패리티 검사"}},
                         "reason":{"text":"- 수직 중복 검사(VRC: Vertical Redundancy Check) = 패리티 검사","binding":{"anchor_id":162,"quote":"- 수직 중복 검사(VRC: Vertical Redundancy Check) = 패리티 검사"}}},
                "task2":{"result":{"value":"오류 검출 코드","binding":{"anchor_id":166,"quote":"● 오류 검출 코드 수직 중복 검사(VRC) : 패리티 검사"}},
                         "reason":{"text":"● 오류 검출 코드 수직 중복 검사(VRC) : 패리티 검사","binding":{"anchor_id":166,"quote":"● 오류 검출 코드 수직 중복 검사(VRC) : 패리티 검사"}}},
                "dependency":{"input":"task1.result","output":"task2.result",
                              "required_result":{"anchor_id":162,"quote":"- 수직 중복 검사(VRC: Vertical Redundancy Check) = 패리티 검사"},
                              "why_required":"①에서 판별한 결과를 후속 상황의 판단 대상으로 사용해야 ②의 적용 결과를 결정할 수 있다."}
            },
            "selector_relation":{"relation_type":"contrast"}
        }
        ok,detail=cc.validate_r59_contract(DB,"통신기술",c)
        self.assertTrue(ok,detail)
        c["validation"]=detail
        q=cc.r59_contract_to_question(dict(c,status="R59_PYTHON_VALIDATED"))
        self.assertNotIn("CLAIM_LACKS_JUDGMENT",cc.r59_prejudge_errors(c,q))

    def test_operation_ranking_features_exist_all_domains(self):
        domains=["기술교육론","발명","제조기술","건설기술","생명기술","전기·전자","통신기술","재료역학","수송기술"]
        for d in domains:
            _,rows=cc._r60_python_relation_candidates(DB,d,limit=160,max_candidates=24)
            self.assertGreaterEqual(len(rows),2,d)
            self.assertIn("operation_score",rows[0])

    def test_truncated_answer_is_removed_before_support_ranking(self):
        bundles=cc._r59_select_bundles("","",DB,"통신기술",wanted=6)
        pairs=[b["selector_relation"]["anchor_ids"] for b in bundles]
        self.assertFalse(any(162 in pair for pair in pairs))  # "수직 중복 검사(VRC" is truncated
        self.assertTrue(all(8 not in b["selector_relation"]["support_anchor_ids"] for b in bundles))

    def test_writer_budget_is_not_wasted_on_mirrored_pairs(self):
        bundles=cc._r59_select_bundles("","",DB,"기술교육론",wanted=3)
        pairs=[tuple(b["selector_relation"]["anchor_ids"]) for b in bundles]
        unordered=[frozenset(x) for x in pairs]
        self.assertEqual(len(unordered),len(set(unordered)))
        self.assertEqual(bundles.diagnostics.get("selection_strategy"),"R69_EXTRACTIVE_RESULT_DIVERSITY_NO_REJECTED_FALLBACK")

    def test_archive_state_uses_hidden_row_without_schema_change(self):
        calls=[]
        def fake_request(url,key,method="GET",payload=None,prefer=None):
            calls.append((url,method,payload))
            if method=="GET":
                return []
            if method=="POST":
                return [dict(payload,id="state-id")]
            return []
        with patch.object(archive_store,"_request",side_effect=fake_request):
            out=archive_store.save_certification_state("https://x.supabase.co","k",[{"contract_id":"c"}],{"verified_slots":7})
        self.assertEqual(out["id"],"state-id")
        self.assertEqual(out["title"],archive_store.CERT_STATE_TITLE)
        self.assertEqual(out["exam_a"]["contracts"][0]["contract_id"],"c")

    def test_app_has_no_manual_contract_json_import(self):
        text=(ROOT/"app.py").read_text(encoding="utf-8")
        self.assertNotIn("R59 contract JSON 불러오기",text)
        self.assertNotIn("R59 contract JSON 저장",text)
        self.assertIn("실제기출 기반 생성 + Judge 인증",text)


if __name__=="__main__":
    unittest.main()
