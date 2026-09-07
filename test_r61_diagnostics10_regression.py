import unittest, tempfile, sqlite3, os
from capability_contracts import _r59_select_bundles, r59_prejudge_errors

class Diagnostics10Regression(unittest.TestCase):
    def test_hidden_rubric_similarity_alone_does_not_mean_duplicate_tasks(self):
        c={'source_plan':None,'public_clues':False,'student_claim':'학생은 A로 판단하였다.'}
        q={'tasks':['① 학생의 분류가 타당한지 판단하고 수정 근거를 쓰시오.','② ①에서 판별한 대상을 새 조건에 적용하여 효과를 판단하시오.'],'answer':[]}
        errs=r59_prejudge_errors(c,q)
        self.assertNotIn('DUPLICATE_SEMANTIC_TASKS',errs)

    def test_contract_validator_accepts_hidden_support_anchor_window(self):
        # R61 contracts may cite 2 scored anchors plus up to 4 same-source support anchors.
        from capability_contracts import validate_r59_contract
        fd,path=tempfile.mkstemp(suffix='.db'); os.close(fd)
        try:
            con=sqlite3.connect(path)
            con.execute('create table anchors(id integer,domain text,topic text,answer text,evidence text,source_name text,page_no integer)')
            for i in range(1,7):
                con.execute('insert into anchors values(?,?,?,?,?,?,?)',(i,'D',f'T{i}',f'A{i}',f'근거 문장 {i} 조건과 절차를 설명한다','S',1))
            con.commit(); con.close()
            c={'contract_type':'contrastive_error_transfer','cited_anchor_ids':list(range(1,7)),
               'exact_answers':['A1','A2'],'clues':[],
               'tasks':['① 학생의 판단 오류를 수정하고 근거를 쓰시오.','② ①에서 수정한 판단을 앞의 기준으로 적용하고 결과를 쓰시오.'],
               'reasoning_chain':['자료 분석','중간 판단','후속 적용'],'task2_uses_task1':True,
               'student_claim':'학생은 A2로 판단하였다.','transfer_case':'①에서 판별한 대상을 다른 조건에 적용한다.',
               'public_clues':False}
            ok,detail=validate_r59_contract(path,'D',c)
            self.assertFalse(any(e=='R59_NEED_2_TO_6_ANCHORS' for e in detail['errors']))
        finally:
            os.unlink(path)

if __name__=='__main__': unittest.main()
