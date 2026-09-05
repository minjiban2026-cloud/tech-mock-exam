"""Offline stabilization regressions. Mock PASS is never a quality claim."""
import copy
import hashlib
import inspect
import itertools
import json
from pathlib import Path
import random
import sqlite3
import unittest
from unittest.mock import patch
from types import SimpleNamespace

import capability_contracts as cc
import certification_state as state
import exam_builder as eb
import quality_judge as qj

DB=Path(__file__).with_name('knowledge.db')
BASE_HASH='d7720e21bea5456c33fa80151960b030834f3e604e7414669f40a1321411d8e3'


def review():
    return {'pass':True,'blind_verdict':'PASS','scores':{k:4.5 for k in qj.SCORE_KEYS},
            'fatal_flags':[],'reason':'OFFLINE MOCK ONLY'}


def fixture(domain, typ=None):
    """Source-backed shape fixture, NOT an exam-quality example or a fallback generator."""
    rows=cc._anchor_rows(DB,domain,72)
    for a,b in itertools.combinations(rows,2):
        if a['source_name']!=b['source_name'] or a['answer']==b['answer']: continue
        clues=[]
        for side,anchor in [('A',a),('B',b)]:
            text=anchor['evidence']
            for x in (a,b): text=text.replace(x['answer'],'[항목]')
            words=text.split(); mid=max(1,len(words)//4)
            clues.extend({'side':side,'anchor_id':anchor['id'],'text':chunk}
                         for chunk in (' '.join(words[:-mid]),' '.join(words[mid:])))
        c={'domain':domain,'topic':a['topic'],'contract_type':typ or cc.R59_ALLOWED_TYPES[0],
           'schema_version':cc.R59_SCHEMA_VERSION,'status':'R59_PYTHON_VALIDATED',
           'cited_anchor_ids':[a['id'],b['id']],'exact_answers':[a['answer'],b['answer']],
           'clues':clues,'student_claim':'학생은 두 사례에 같은 기준을 적용했다고 판단하였다.',
           'transfer_case':'추가 상황은 사례 A의 첫 조건과 사례 B의 둘째 조건을 함께 가진다.',
           'tasks':['① 두 사례의 학생 판단 오류를 수정하고 판단 근거를 설명하시오.',
                    '② ①에서 수정한 판단 기준을 추가 상황에 적용하고 근거를 설명하시오.'],
           'reasoning_chain':['두 근거 비교','학생 오류 수정','수정 기준의 추가 적용'],
           'task2_uses_task1':True,'contract_id':domain+'-'+(typ or cc.R59_ALLOWED_TYPES[0])}
        ok,diag=cc.validate_r59_contract(DB,domain,c)
        if ok: c['validation']=diag; return c
    raise AssertionError('No shape fixture for '+domain)


class StabilizationTests(unittest.TestCase):
    def test_compile_all(self):
        for f in DB.parent.glob('*.py'):
            compile(f.read_text(encoding='utf-8-sig'),str(f),'exec')

    def test_interface(self):
        self.assertEqual(str(inspect.signature(eb.make_ab)), "(db_path, a_count=12, a_points=40, b_count=11, b_points=40, domains=None, api_key='', model='gpt-5.6-luna', ai_enabled=True, ai_quality_enabled=True, judge_model=None, seed=None)")

    def test_baseline_legacy_and_forged_status(self):
        rows=[{'domain':d,'status':s,'contract_type':t} for d in eb.DOMAINS
              for s in ['PYTHON_VALIDATED','AI_VERIFIED','R57_AI_VERIFIED','R58_AI_VERIFIED','R59_AI_VERIFIED']
              for t in cc.R59_ALLOWED_TYPES]
        inv=cc.combined_coverage_inventory(DB,rows,eb.DOMAINS,eb.FORMULA_DOMAINS)
        self.assertEqual(inv['verified_slots'],5)

    def test_rerun_preserves_new_contract(self):
        raw=json.dumps({'contracts':[{'contract_id':'old'}]}).encode()
        s={}; state.import_contract_upload(s,raw)
        s['R59_CONTRACTS'].append({'contract_id':'new'})
        self.assertIsNone(state.import_contract_upload(s,raw))
        self.assertEqual(len(s['R59_CONTRACTS']),2)
        before=copy.deepcopy(s)
        with self.assertRaises(ValueError): state.import_contract_upload(s,b'{"contracts":[1]}')
        self.assertEqual(s,before)

    def test_judge_context_precedence_and_fallback(self):
        payload={'verdict':'PASS','scores':review()['scores'],'fatal_flags':[]}
        captured=[]
        def ask(*args): captured.append(args[2]); return payload
        q={'points':4,'evidence':['DB evidence'],'source_context_override':'OVERRIDE evidence'}
        with patch.object(qj,'_ask_json',side_effect=ask):
            qj.judge_question('mock','mock',q,'EXPLICIT evidence')
            qj.judge_question('mock','mock',q,'')
            q.pop('source_context_override'); qj.judge_question('mock','mock',q,' ')
        for expected,prompt in zip(['EXPLICIT evidence','OVERRIDE evidence','DB evidence'],captured):
            self.assertIn(expected,prompt)

    def test_bad_shape_rejections(self):
        c=fixture(eb.DOMAINS[0]); self.assertTrue(cc.validate_r59_contract(DB,c['domain'],c)[0])
        changes=[{'student_claim':c['exact_answers'][0]}, {'task2_uses_task1':False},
                 {'tasks':['① 명칭을 쓰시오.','② 다른 명칭을 쓰시오.']},
                 {'clues':[]}, {'reasoning_chain':['반복']*3},
                 {'exact_answers':[c['exact_answers'][0],'없는 근거 정답']},
                 {'clues':[dict(x,side='C') for x in c['clues']]}]
        for change in changes:
            with self.subTest(change=change):
                bad=copy.deepcopy(c); bad.update(change)
                self.assertFalse(cc.validate_r59_contract(DB,c['domain'],bad)[0])

    def test_receipt_rejects_missing_failed_and_edited_reviews(self):
        c=fixture(eb.DOMAINS[0]); c.update(status='R59_AI_VERIFIED',ai_quality=review(),judge_model='offline-mock')
        state.attach_receipt(c); self.assertTrue(state.verified_contract_receipt(c))
        bad=copy.deepcopy(c); bad['tasks'][0]+=' 변경'; self.assertFalse(state.verified_contract_receipt(bad))
        for value in [0,3.9,float('nan'),float('inf')]:
            bad=copy.deepcopy(c); bad['ai_quality']['scores']['grounding']=value
            self.assertFalse(state.verified_contract_receipt(bad))
        bad=copy.deepcopy(c); bad['ai_quality']['fatal_flags']=['TOO_EASY']; self.assertFalse(state.verified_contract_receipt(bad))

    def test_mock_certification_5_to_18_and_json_roundtrip(self):
        pools={d:[fixture(d,t) for t in cc.R59_ALLOWED_TYPES] for d in eb.DOMAINS}
        with patch.object(cc,'synthesize_r59_pool',side_effect=lambda key,model,db,d,*a,**kw:copy.deepcopy(pools[d])), patch.object(eb,'judge_question',return_value=review()) as judge:
            run=eb.certify_r59_missing_slots(DB,[],domains=eb.DOMAINS,api_key='offline-mock',model='mock',judge_model='mock',seed=59)
        self.assertEqual(run['summary']['before_verified'],5)
        self.assertEqual(run['summary']['after_verified'],18)
        self.assertEqual(judge.call_count,13)
        for call in judge.call_args_list: self.assertIn('근거:',call.args[3])
        restored=json.loads(json.dumps(run['contracts']))
        self.assertEqual(cc.combined_coverage_inventory(DB,restored,eb.DOMAINS,eb.FORMULA_DOMAINS)['verified_slots'],18)

    def test_retry_upserts_and_refreshes_evidence(self):
        d=eb.DOMAINS[0]; c=fixture(d); c['validation']={'anchors':[{'evidence':'STALE EVIDENCE'}]}
        with patch.object(cc,'synthesize_r59_pool',return_value=[c]),patch.object(eb,'judge_question',return_value=review()) as judge:
            run=eb.certify_r59_missing_slots(DB,[c],domains=[d],api_key='mock',seed=4)
        self.assertEqual(len(run['contracts']),1)
        self.assertEqual(run['summary']['after_verified'],2)
        self.assertNotIn('STALE EVIDENCE',judge.call_args.args[3])

    def test_domain_failure_preserves_previous_pass(self):
        ds=eb.DOMAINS[:2]
        def pool(key,model,db,d,*args,**kwargs):
            if d==ds[1]: raise TimeoutError('mock timeout')
            return [fixture(d)]
        with patch.object(cc,'synthesize_r59_pool',side_effect=pool),patch.object(eb,'judge_question',return_value=review()):
            run=eb.certify_r59_missing_slots(DB,[],domains=ds,api_key='mock',seed=2)
        self.assertEqual(len(run['accepted_contracts']),1)
        self.assertIn('generation_error',run['domain_logs'][1])

    def test_writer_executes_after_successful_selector(self):
        c=fixture(eb.DOMAINS[0]); ids=c['cited_anchor_ids']; anchors={a['id']:a for a in cc._anchor_rows(DB,c['domain'],72)}
        bundle={'anchors':[anchors[i] for i in ids],'selector_relation':{'master_relation':'MOCK RELATION'}}
        payload=dict(c,bundle_id=0); response=SimpleNamespace(output_text=json.dumps({'contracts':[payload]}))
        client=SimpleNamespace(responses=SimpleNamespace(create=lambda **kw:response))
        with patch.object(cc,'_r59_select_bundles',return_value=[bundle]),patch('openai.OpenAI',return_value=client):
            pool=cc.synthesize_r59_pool('mock','mock',DB,c['domain'],1)
        self.assertEqual(len(pool),1)
        self.assertIn('MOCK RELATION',cc._r59_prompt(c['domain'],[bundle],[]))

    def test_database_unchanged_and_readonly(self):
        self.assertEqual(hashlib.sha256(DB.read_bytes()).hexdigest(),BASE_HASH)
        from retrieval import connect
        con=connect(DB)
        try:
            self.assertEqual(con.execute('pragma quick_check').fetchone()[0],'ok')
            self.assertEqual(con.execute('pragma query_only').fetchone()[0],0)
            # EXPLAIN executes no write, so also verify the URI connection refuses a write transaction.
            with self.assertRaises(sqlite3.OperationalError): con.execute('CREATE TABLE forbidden_test (x)')
        finally: con.close()
        missing=DB.with_name('must_not_be_created.db')
        with self.assertRaises(sqlite3.OperationalError): connect(missing)
        self.assertFalse(missing.exists())

    @unittest.expectedFailure
    def test_semantic_dependency_not_proven_by_reference_word(self):
        # Known blocker: merely mentioning task 1 still passes the lexical validator.
        c=fixture(eb.DOMAINS[0]); c['tasks']=['① 사례 A를 판단하시오.','② ①과 무관하게 사례 B만 보고 판단하시오.']
        self.assertFalse(cc.validate_r59_contract(DB,c['domain'],c)[0])

    @unittest.expectedFailure
    def test_decorative_transfer_not_proven_by_source_overlap(self):
        c=fixture(eb.DOMAINS[0]); c['transfer_case']='자료를 사용하지 않고 일반적인 지식을 그대로 쓰면 된다.'
        self.assertFalse(cc.validate_r59_contract(DB,c['domain'],c)[0])


class AppExecutionTests(unittest.TestCase):
    def test_streamlit_button_and_rerun(self):
        from streamlit.testing.v1 import AppTest
        app=AppTest.from_file(str(DB.with_name('app.py')),default_timeout=60)
        app.secrets['OPENAI_API_KEY']='offline-mock'
        app.secrets['SUPABASE_URL']=''
        app.secrets['SUPABASE_SERVICE_ROLE_KEY']=''
        app.secrets['SUPABASE_KEY']=''
        with patch('openai.OpenAI',side_effect=AssertionError('Live API forbidden')), patch.object(cc,'synthesize_r59_pool',return_value=[]) as pool:
            app.run(); self.assertFalse(list(app.exception))
            next(b for b in app.button if b.label=='R59 실제기출 전이형 생성 + Judge 인증').click().run()
            self.assertFalse(list(app.exception))
            self.assertEqual(pool.call_count,9)
            self.assertEqual(app.session_state['R59_CERT_RUN']['summary']['after_verified'],5)
            app.run(); self.assertFalse(list(app.exception))
            self.assertEqual(pool.call_count,9)

    def test_supabase_failure_is_nonfatal(self):
        import ast
        tree=ast.parse(DB.with_name('app.py').read_text(encoding='utf-8'))
        fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='save_generated_to_archive')
        def fail(*args): raise RuntimeError('mock archive failure')
        context={'archive_credentials':lambda:('mock-url','mock-key',True,False),
                 'archive_is_configured':lambda *a:True,'default_archive_title':lambda s:'mock',
                 'create_exam':fail}
        exec(compile(ast.Module(body=[fn],type_ignores=[]),'app.py','exec'),context)
        saved,warning=context['save_generated_to_archive']({}, {}, 'mock', 1, [])
        self.assertIsNone(saved); self.assertIn('보존',warning)

if __name__=='__main__': unittest.main(verbosity=2)
