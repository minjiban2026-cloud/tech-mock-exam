import json
import unittest
from pathlib import Path
from unittest.mock import patch

import capability_contracts as cc
import exam_builder as eb

ROOT=Path(__file__).resolve().parent
DB=ROOT/'knowledge.db'
DIAG=Path('/mnt/data/generation_diagnostics (17).json')


class R68Diagnostics17Regression(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.diag=json.loads(DIAG.read_text(encoding='utf-8')) if DIAG.exists() else None

    def test_diagnostics17_stalled_at_13(self):
        if self.diag is None: self.skipTest('diagnostics17 not available')
        self.assertEqual(self.diag['summary']['before_verified'],13)
        self.assertEqual(self.diag['summary']['after_verified'],13)
        self.assertEqual(self.diag['summary']['judge_pass'],0)

    def test_persisted_broken_passes_are_quarantined_not_counted(self):
        if self.diag is None: self.skipTest('diagnostics17 not available')
        inv=cc.combined_coverage_inventory(str(DB),self.diag['contracts'],eb.DOMAINS,eb.FORMULA_DOMAINS)
        q={x['contract_id']:x for x in inv['quarantined_verified']}
        self.assertIn('R59-36f8501cbfe333f721bfc242',q)  # VRC(
        self.assertIn('R59-3cd0bc00e28f016ead67380f',q)  # 기울기가 1 / cut evidence
        self.assertIn('R59-57e6c47e306411fe4dc632e1',q)  # cut scored evidence
        # History remains present in the input; quarantine is selection behavior, not deletion.
        ids={x.get('contract_id') for x in self.diag['contracts']}
        self.assertTrue(set(q).issubset(ids))

    def test_biology_gets_multi_anchor_clusters(self):
        anchors,_=cc._r60_python_relation_candidates(str(DB),'생명기술',limit=180,max_candidates=80)
        clusters=cc._r68_cluster_packets(anchors)
        self.assertGreaterEqual(len(clusters),3)
        self.assertTrue(any(len(c['anchors'])>=4 for c in clusters))

    def test_composed_relation_is_python_grounded(self):
        anchors,_=cc._r60_python_relation_candidates(str(DB),'생명기술',limit=180,max_candidates=80)
        cluster=cc._r68_cluster_packets(anchors)[0]
        safe=[a for a in cluster['anchors'] if not __import__('re').match(r'^\s*\d{1,2}\s*[.)]',str(a.get('answer','')))]
        a1,a2=safe[0]['id'],safe[1]['id']
        plan={'verdict':'SELECT','source_support':5,'dependency':5,'inferential_distance':4,
              'transferability':4,'rote_risk':1,'task1_anchor_id':a1,'task2_anchor_id':a2,
              'support_anchor_ids':[a['id'] for a in cluster['anchors'][2:5]],
              'contract_type':'contrastive_error_transfer','relation_type':'contrast',
              'dependency_reason':'①에서 판별한 기준이 ②의 달라진 조건에서 후속 판단을 결정하는 데 반드시 필요하다.','reason':'test'}
        row=cc._r68_build_composed_relation(plan,cluster)
        self.assertIsNotNone(row)
        self.assertEqual(row['anchor_ids'],[a1,a2])
        self.assertEqual(row['contract_type'],'contrastive_error_transfer')
        self.assertEqual(len(row['fixed_answers']),2)
        self.assertTrue(set(row['selector_relation']['support_anchor_ids']) if 'selector_relation' in row else True)


    def test_composed_plan_flows_through_selector_parser(self):
        import re, sys, types
        anchors,_=cc._r60_python_relation_candidates(str(DB),'생명기술',limit=180,max_candidates=80)
        cluster=cc._r68_cluster_packets(anchors)[0]
        safe=[a for a in cluster['anchors'] if not re.match(r'^\s*\d{1,2}\s*[.)]',str(a.get('answer','')))]
        a1,a2=safe[0]['id'],safe[1]['id']
        cp={'cluster_id':0,'task1_anchor_id':a1,'task2_anchor_id':a2,'support_anchor_ids':[a['id'] for a in safe[2:4]],
            'contract_type':'contrastive_error_transfer','relation_type':'contrast','source_support':5,'dependency':5,
            'inferential_distance':4,'transferability':4,'rote_risk':1,'verdict':'SELECT',
            'dependency_reason':'①에서 판별한 기준이 ②의 달라진 조건에서 후속 판단을 결정하는 데 반드시 필요하다.','reason':'test'}
        raw={'assessments':[],'selected_ids':[],'reserve_ids':[],'composed_plans':[cp],'rejected':{}}
        response=types.SimpleNamespace(output_text=json.dumps(raw,ensure_ascii=False))
        client=types.SimpleNamespace(responses=types.SimpleNamespace(create=lambda **kw:response))
        fake=types.SimpleNamespace(OpenAI=lambda **kw:client)
        with patch.dict(sys.modules,{'openai':fake}):
            pool=cc._r59_select_bundles('mock','mock',str(DB),'생명기술',wanted=2,preferred_types=['contrastive_error_transfer'])
        self.assertEqual(len(pool),1)
        self.assertEqual(pool.diagnostics.get('semantic_composed_selected'),1)
        self.assertEqual(pool[0]['contract_type'],'contrastive_error_transfer')
        self.assertGreaterEqual(len(pool[0]['selector_relation'].get('support_anchor_ids',[])),1)

    def test_selector_prompt_is_coverage_aware_and_multi_anchor(self):
        txt=cc._r67_selector_prompt('생명기술',[],2,preferred_types=['contrastive_error_transfer'],clusters=[])
        self.assertIn('현재 coverage에서 우선 필요한 contract_type',txt)
        self.assertIn('composed_plans',txt)
        self.assertIn('cluster_packet',txt)
        self.assertIn('contrastive_error_transfer',txt)

    def test_certifier_passes_missing_contract_types_to_synthesis(self):
        # Use a domain with no historical formula slot so the preferred type is observable.
        d='제조기술'
        if self.diag is None: self.skipTest('diagnostics17 not available')
        existing=self.diag['contracts']
        calls=[]
        def fake_pool(key,model,db,domain,need,pool_size=None,preferred_types=None,forbidden_anchor_pairs=None):
            calls.append((domain,need,tuple(preferred_types or [])))
            return cc.GenerationPool()
        with patch.object(cc,'synthesize_r59_pool',side_effect=fake_pool):
            # exam_builder imports the function inside the certifier, so patching module attr works.
            eb.certify_r59_missing_slots(str(DB),existing,domains=[d],api_key='mock',model='mock',judge_model='mock')
        self.assertTrue(calls)
        self.assertEqual(calls[0][0],d)
        self.assertIn('contrastive_error_transfer',calls[0][2])
        self.assertNotIn('criterion_conflict_resolution',calls[0][2])

    def test_versions_are_r68(self):
        self.assertEqual(eb.BUILDER_API_VERSION,'ACTUAL-EXAM-TRANSFER-R69-20260907')


if __name__=='__main__':
    unittest.main()
