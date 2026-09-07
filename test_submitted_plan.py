"""Regression for the user's actual rejected selector plan. No live API."""
import copy,json,sqlite3,unittest
from pathlib import Path
from unittest.mock import patch
import capability_contracts as cc
import exam_builder as eb
from question_plans import validate_plan,bind
from test_stabilization import DB
from test_generation_diagnostics import fake_client,plan_for

class SubmittedPlanTests(unittest.TestCase):
    def setUp(self):
        self.row=json.loads(Path(__file__).with_name('test_data').joinpath('rejected_heat_source_plan.json').read_text(encoding='utf-8'))
        con=sqlite3.connect(DB.as_uri()+'?mode=ro',uri=True);con.row_factory=sqlite3.Row
        ids=self.row['anchor_ids'];self.anchors=[dict(x) for x in con.execute('select * from anchors where id in (?,?)',ids)];con.close()

    def test_actual_plan_reports_all_independent_faults(self):
        ok,d=validate_plan(self.row['source_plan'],self.anchors,relation_type=self.row['relation_type'])
        self.assertFalse(ok)
        expected={'RESULT_AND_REASON_IDENTICAL','DEPENDENCY_RESULT_MISMATCH',
                  'CHOICE_CRITERION_IS_CATALOG','CHOICE_RESULT_HAS_MULTIPLE_OPTIONS'}
        self.assertTrue(expected.issubset(d['errors']),d)
        self.assertIn({'code':'DEPENDENCY_RESULT_MISMATCH','path':'dependency.required_result'},d['error_details'])

    def test_fixing_only_duplicate_reason_cannot_promote_bad_plan(self):
        plan=copy.deepcopy(self.row['source_plan']);plan['task2']['reason']=plan['criterion']
        ok,d=validate_plan(plan,self.anchors,relation_type='conditional_choice')
        self.assertFalse(ok);self.assertIn('DEPENDENCY_RESULT_MISMATCH',d['errors'])
        self.assertIn('CHOICE_CRITERION_IS_CATALOG',d['errors'])

    def test_bad_plan_never_calls_writer_or_judge(self):
        # R60 never asks an AI selector to propose this known-bad plan. Validate it
        # directly and ensure the deterministic miner does not need selector calls.
        ok,d=validate_plan(self.row['source_plan'],self.anchors,relation_type=self.row['relation_type'])
        self.assertFalse(ok);self.assertGreaterEqual(len(d['error_details']),4)
        rows=cc._r59_select_bundles('unused','unused',DB,'제조기술',wanted=2)
        self.assertEqual(rows.diagnostics['selector_calls'],0)

    def test_short_result_requires_exact_provenance_not_padding(self):
        anchor={'id':1,'answer':'선택값','evidence':'조건이 충족된 경우 선택값을 적용하며 기준을 충족하지 못하면 다른 방법을 검토한다.'}
        ref={'anchor_id':1,'quote':'선택값'}
        self.assertEqual(bind(ref,{1:anchor},role='result'),'선택값')
        with self.assertRaises(ValueError):bind(ref,{1:anchor},role='evidence')
        with self.assertRaises(ValueError):bind({'anchor_id':1,'quote':'없는값'},{1:anchor},role='result')
        with self.assertRaises(ValueError):bind({'anchor_id':2,'quote':'12'},{2:{'evidence':'입력 1234를 적용한다.'}},role='result')

    def test_multiple_bad_fields_reported_without_exception(self):
        p=copy.deepcopy(self.row['source_plan']);p['task1']=None;p['dependency']={'input':'wrong','output':'wrong'};p['criterion']=None
        ok,d=validate_plan(p,self.anchors)
        self.assertFalse(ok);self.assertGreaterEqual(len(d['error_details']),3)

if __name__=='__main__':unittest.main(verbosity=2)
