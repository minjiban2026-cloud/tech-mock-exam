import json, sqlite3, sys, types, unittest
from pathlib import Path
from unittest.mock import patch
import capability_contracts as cc

DB=Path(__file__).with_name('knowledge.db')

class R59SourceHygieneTests(unittest.TestCase):
    def test_every_domain_keeps_at_least_one_local_pair(self):
        domains=['기술교육론','발명','제조기술','건설기술','생명기술','전기·전자','통신기술','재료역학','수송기술']
        for domain in domains:
            rows=cc._anchor_rows(DB,domain,32)
            self.assertGreaterEqual(len(rows),2,domain)
            self.assertTrue(any(a['source_name']==b['source_name'] and
                                (not a.get('page_no') or not b.get('page_no') or abs(int(a['page_no'])-int(b['page_no']))<=cc.R59_MAX_PAGE_SPAN)
                                for i,a in enumerate(rows) for b in rows[i+1:]),domain)

    def test_truncated_connective_evidence_is_not_retrieved(self):
        rows=cc._anchor_rows(DB,'제조기술',200)
        ids={r['id'] for r in rows}
        self.assertNotIn(1205,ids)  # evidence ends with "...발달하여"
        self.assertTrue(cc._obviously_incomplete_evidence('규소의 결정이 크게 발달하여'))
        self.assertFalse(cc._obviously_incomplete_evidence('냉각 속도가 느리면 결정이 크게 발달한다.'))

    def test_far_apart_same_pdf_relation_is_rejected_before_writer(self):
        con=sqlite3.connect(DB);con.row_factory=sqlite3.Row
        anchors=[dict(r) for r in con.execute('select * from anchors where id in (1163,1205) order by id')]
        con.close()
        relation={'anchor_ids':[1163,1205],'relation_type':'contrast','contract_type':cc.R59_ALLOWED_TYPES[0],
                  'source_plan':{'schema':'SOURCE_BOUND_TASK_PLAN_V1'}}
        response=types.SimpleNamespace(output_text=json.dumps({'relations':[relation]},ensure_ascii=False))
        client=types.SimpleNamespace(responses=types.SimpleNamespace(create=lambda **kw:response))
        fake_openai=types.SimpleNamespace(OpenAI=lambda **kw:client)
        with patch.object(cc,'_anchor_rows',return_value=anchors),patch.dict(sys.modules,{'openai':fake_openai}):
            pool=cc._r59_select_bundles('mock','mock',DB,'제조기술',wanted=1)
        self.assertEqual(len(pool),0)
        self.assertIn('python_relation_miner:NO_RELATION_CANDIDATE',pool.diagnostics['failure_counts'])
        self.assertEqual(pool.diagnostics.get('selector_calls'),0)

if __name__=='__main__': unittest.main(verbosity=2)
