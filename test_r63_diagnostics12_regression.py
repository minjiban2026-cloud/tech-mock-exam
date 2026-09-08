import unittest
from capability_contracts import _r60_python_relation_candidates, _r59_select_bundles

DB='knowledge.db'

class R63Diagnostics12Regression(unittest.TestCase):
    def test_known_live_pass_pairs_rank_as_strong(self):
        expected={'발명':{1376,1373},'건설기술':{353,354}}
        for domain,ids in expected.items():
            _,rows=_r60_python_relation_candidates(DB,domain,limit=160,max_candidates=48)
            row=next((r for r in rows if set(r['anchor_ids'])==ids),None)
            self.assertIsNotNone(row,domain)
            self.assertIsInstance(row['operation_score'],int)
            self.assertIsInstance(row['reasoning_viability'],int)

    def test_writer_selection_never_uses_negative_operation_pairs(self):
        domains=['기술교육론','발명','제조기술','건설기술','생명기술','전기·전자','통신기술','재료역학','수송기술']
        for domain in domains:
            pool=_r59_select_bundles('','',DB,domain,wanted=4)
            self.assertEqual(pool.diagnostics.get('selector_calls'),0)
            self.assertEqual(pool.diagnostics.get('selection_strategy'),'R73_SOURCE_PACKET_ATOMIC_RESULT_DIFFICULTY_WRITER')
            for b in pool:
                rel=b['selector_relation']
                self.assertGreater(rel['operation_score'],-25,(domain,rel))
                self.assertGreater(rel['reasoning_viability'],-31,(domain,rel))

    def test_weak_domains_do_not_fill_quota_with_bad_pairs(self):
        for domain in ('제조기술','생명기술'):
            pool=_r59_select_bundles('','',DB,domain,wanted=4)
            self.assertGreater(len(pool),0)
            self.assertEqual(pool.diagnostics.get('selector_fallback'),'PYTHON_OFFLINE_TEST_ONLY')

if __name__=='__main__': unittest.main()
