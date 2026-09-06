import json, sqlite3
from pathlib import Path
import unittest
import capability_contracts as cc
from question_plans import validate_plan

DB=Path(__file__).with_name('knowledge.db')
DIAG=Path('/mnt/data/generation_diagnostics (8).json')

class Diagnostics8Regression(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not DIAG.exists():
            raise unittest.SkipTest('local diagnostics 8 fixture unavailable')
        cls.data=json.loads(DIAG.read_text(encoding='utf-8'))

    def test_observed_selector_json_repairs(self):
        for domain in ('기술교육론','발명'):
            row=next(x for x in self.data['domain_logs'] if x['domain']==domain)
            obj=cc._load_r59_selector_json(row['generation']['selector_response_text'])
            self.assertGreater(len(obj.get('relations',[])),0)

    def test_manufacturing_valid_source_plans_survive_whitespace_and_short_quotes(self):
        row=next(x for x in self.data['domain_logs'] if x['domain']=='제조기술')
        con=sqlite3.connect(DB); con.row_factory=sqlite3.Row
        accepted=0
        try:
            for rej in row['generation']['rejections']:
                c=rej.get('candidate') or {}; ids=c.get('anchor_ids') or []
                if not ids: continue
                aa=[dict(r) for r in con.execute('select id,domain,topic,answer,evidence,source_name,page_no from anchors where id in (%s)'%(','.join('?'*len(ids))),ids)]
                ok,_=validate_plan(c.get('source_plan'),aa,relation_type=c.get('relation_type'))
                accepted += int(ok)
        finally: con.close()
        self.assertGreaterEqual(accepted,2)

    def test_new_questions_do_not_render_definition_bullets(self):
        c={'status':'R59_PYTHON_VALIDATED','domain':'생명기술','topic':'x','contract_type':'criterion_conflict_resolution',
           'clues':[{'side':'A','anchor_id':1,'text':'정의문장A'},{'side':'B','anchor_id':2,'text':'정의문장B'}],
           'student_claim':'학생은 두 기준을 바꾸어 적용해도 된다고 잘못 판단하였다.',
           'transfer_case':'①에서 수정한 기준을 다른 조건에 적용한다.','tasks':['① 오류를 수정하시오.','② ①의 수정 결과를 적용하시오.'],
           'exact_answers':['A','B'],'reasoning_chain':['a','b','c'],'public_clues':False,'validation':{'anchors':[]}}
        # source_plan omitted intentionally: conversion still demonstrates public rendering rule
        q=cc.r59_contract_to_question(c)
        self.assertNotIn('[사례 A]',q['passage'])
        self.assertNotIn('정의문장A',q['passage'])

if __name__=='__main__': unittest.main()
