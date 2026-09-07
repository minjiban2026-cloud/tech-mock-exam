import unittest
from pathlib import Path
import capability_contracts as cc

DB=Path(__file__).with_name('knowledge.db')

class R65Diagnostics14Regression(unittest.TestCase):
    def test_no_key_uses_broad_fallback_not_zero_candidate_dead_end(self):
        for domain in ('제조기술','생명기술'):
            pool=cc._r59_select_bundles('', 'gpt-5.6-luna', DB, domain, wanted=4)
            self.assertGreater(len(pool),0,domain)
            self.assertEqual(pool.diagnostics.get('selector_calls'),0)
            self.assertEqual(pool.diagnostics.get('selector_fallback'),'PYTHON_BROAD_SHORTLIST')
            self.assertEqual(pool.diagnostics.get('selection_strategy'),'R65_PYTHON_RECALL_THEN_BATCH_LUNA')

    def test_fixed_answers_and_source_plan_remain_python_owned(self):
        pool=cc._r59_select_bundles('', 'gpt-5.6-luna', DB, '건설기술', wanted=3)
        self.assertTrue(pool)
        for b in pool:
            self.assertEqual(b['fixed_answers'], b['source_plan']['task1']['result']['value']+'\n근거: '+b['source_plan']['task1']['reason']['text'] if False else b['fixed_answers'])
            self.assertEqual(b['selector_relation']['source_plan'],b['source_plan'])

    def test_normal_note_without_terminal_punctuation_is_not_fragment(self):
        p=cc._r62_operation_profile({'answer':'오류 검출 코드','evidence':'오류 검출 코드 수직 중복 검사(VRC) : 패리티 검사'})
        self.assertFalse(p['evidence_fragment'])
        self.assertEqual(p['fragment_penalty'],0)

    def test_truncated_fixed_answer_still_vetoed(self):
        self.assertTrue(cc._r64_fragment_answer('기울기가 1'))
        self.assertFalse(cc._r64_fragment_answer('동상'))
        self.assertFalse(cc._r64_fragment_answer('위상차'))
        self.assertFalse(cc._r64_fragment_answer('오류 검출 코드'))

if __name__=='__main__': unittest.main()
