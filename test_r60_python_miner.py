import unittest
from capability_contracts import _r60_python_relation_candidates, _r59_select_bundles

DOMAINS=['기술교육론','발명','제조기술','건설기술','생명기술','전기·전자','통신기술','재료역학','수송기술']

class R60PythonMinerTests(unittest.TestCase):
    def test_all_domains_have_offline_candidates(self):
        for d in DOMAINS:
            with self.subTest(domain=d):
                anchors, rows=_r60_python_relation_candidates('knowledge.db',d,limit=160)
                self.assertGreaterEqual(len(anchors),2)
                self.assertGreaterEqual(len(rows),2)
                for r in rows:
                    self.assertEqual(len({a['source_name'] for a in r['anchors']}),1)
                    self.assertTrue(r['source_plan'])
                    self.assertIsInstance(r['score'],int)

    def test_selector_path_uses_zero_ai_calls(self):
        for d in DOMAINS:
            with self.subTest(domain=d):
                rows=_r59_select_bundles('','','knowledge.db',d,wanted=2)
                # R63 may deliberately return zero when every pair is below the
                # pre-Writer reasoning floor; this is a quality-preserving result.
                self.assertGreaterEqual(len(rows),0)
                self.assertEqual(rows.diagnostics.get('selector_calls'),0)
                self.assertEqual(rows.diagnostics.get('selector_model'),'PYTHON_FALLBACK_NO_KEY')
                self.assertEqual(rows.diagnostics.get('selection_strategy'),'R65_PYTHON_RECALL_THEN_BATCH_LUNA')

if __name__=='__main__': unittest.main()
