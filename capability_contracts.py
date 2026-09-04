import json, os, re, sqlite3, hashlib
from pathlib import Path

CONTRACT_FILE='capability_contracts.json'
ALLOWED_TYPES={
    'scenario_constraint_application',
    'error_repair_transfer',
    'threshold_decision',
    'ordered_sequence_application',
}


def _clean(s):
    return re.sub(r'\s+',' ',str(s or '').replace('\x01',' ')).strip()


def _norm(s):
    return re.sub(r'[^0-9A-Za-z가-힣]+','',_clean(s)).lower()


def _fp(x):
    return hashlib.sha256(json.dumps(x,ensure_ascii=False,sort_keys=True).encode()).hexdigest()[:24]


def _anchor_rows(db_path, domain, limit=36):
    con=sqlite3.connect(db_path)
    rows=con.execute('''select id,answer,evidence,source_name,page_no,confidence
                        from anchors where domain=? order by confidence desc,id asc''',(domain,)).fetchall()
    con.close()
    out=[]
    for rid,a,e,s,p,c in rows:
        a=_clean(a); e=_clean(e)
        if not (2<=len(a)<=80 and 20<=len(e)<=700):
            continue
        if len(re.findall(r'[가-힣A-Za-z]',e))<12:
            continue
        # reject obvious fragments/page debris
        if e.endswith((':','/','·','▶','-')):
            continue
        out.append({'id':int(rid),'answer':a,'evidence':e,'source_name':str(s or ''),'page_no':int(p or 0),'confidence':float(c or 0)})
        if len(out)>=limit:
            break
    return out


def mining_packet(db_path, domain, limit=36):
    return {'domain':domain,'anchors':_anchor_rows(db_path,domain,limit)}


def _strip_json(text):
    t=str(text or '').strip()
    t=re.sub(r'^```(?:json)?\s*','',t)
    t=re.sub(r'\s*```$','',t)
    return t


def mine_domain_contracts(api_key, model, db_path, domain, wanted=2):
    """One-time semantic mining. AI may SELECT/STRUCTURE source facts but cannot invent facts.
    Output is not trusted until validate_contract() and the final Judge both pass.
    """
    from openai import OpenAI
    packet=mining_packet(db_path,domain,limit=42)
    if not packet['anchors']:
        return []
    client=OpenAI(api_key=api_key,timeout=90,max_retries=1)
    prompt=f'''
너는 대한민국 중등 기술 임용시험의 "출제 가능 관계 계약"을 채굴하는 분석기다.
문항을 직접 예쁘게 쓰는 것이 아니라, 제공된 서브노트 원문만으로 4점 추론문항을 만들 수 있는 구조를 찾아 JSON으로 반환한다.

영역: {domain}
필요 계약 수: {wanted}
원문 anchors:
{json.dumps(packet['anchors'],ensure_ascii=False)}

절대 규칙:
1. anchor 밖의 사실, 사례, 수치, 인과를 새로 만들지 않는다.
2. 학생에게 source evidence 원문 또는 정답을 그대로 보여주는 계약은 금지한다. source는 출제자용 hidden ground truth다.
3. 단순 "개념명 쓰기 + 특징 쓰기", 원문 정의 복사, 증가/감소 문장 재진술, 원자료에 나온 대책 그대로 쓰기는 금지한다.
4. 두 채점 요구는 하나의 사고사슬이어야 한다. task2가 task1의 판단을 실제로 사용해야 한다.
5. 가능한 계약 유형은 다음 네 가지뿐이다.
   - scenario_constraint_application: 원문의 복수 조건/특징을 조합해 사례를 판별하고 그 판단을 이용해 후속 선택/설명을 한다.
   - error_repair_transfer: 원문 규칙을 기준으로 학생 설명의 오류를 특정하고, 그 수정 원리를 다른 진술에 적용한다.
   - threshold_decision: 명시된 수치/범위/기준을 비교하여 적합 여부를 판단하고, 그 판단을 이용해 후속 결정을 한다.
   - ordered_sequence_application: 명시된 자연적 절차 순서를 복원/적용한다.
6. public_material에는 정답 문자열이나 source evidence의 완성 문장을 그대로 쓰지 않는다. 원문 사실을 학생에게 그대로 베껴주는 구조는 금지한다.
7. exact_answers는 채점 가능한 짧은 정답이며 반드시 cited_anchor_ids의 evidence가 직접 뒷받침해야 한다.
8. public_material_plan은 실제 지문이 아니라 "어떤 정보는 보이고 어떤 정보는 숨길지"를 서술한다. 새 기술 사실을 쓰지 않는다.
9. 같은 계약을 표현만 바꾼 중복 두 개로 내지 않는다.

JSON 하나만 출력:
{{"contracts":[
 {{
  "contract_type":"...",
  "topic":"...",
  "cited_anchor_ids":[1,2],
  "exact_answers":["...","..."],
  "reasoning_chain":["1단계 판단","2단계 적용"],
  "public_material_plan":"...",
  "public_material":"수험생에게 실제로 제시할 자료. 원문/정답 직접복사 금지",
  "task_plan":["...","앞 판단을 이용하여 ..."],
  "why_not_rote":"..."
 }}
]}}
'''
    r=client.responses.create(model=model,input=prompt,reasoning={'effort':'medium'})
    obj=json.loads(_strip_json(r.output_text))
    rows=obj.get('contracts',[]) if isinstance(obj,dict) else []
    valid=[]
    for x in rows:
        ok,detail=validate_contract(db_path,domain,x)
        if ok:
            x=dict(x); x['domain']=domain; x['status']='PYTHON_VALIDATED'; x['validation']=detail
            x['contract_id']=_fp({'domain':domain,'type':x.get('contract_type'),'anchors':x.get('cited_anchor_ids'),'answers':x.get('exact_answers')})
            valid.append(x)
        if len(valid)>=wanted:
            break
    return valid


def validate_contract(db_path, domain, contract):
    errs=[]
    typ=str(contract.get('contract_type',''))
    if typ not in ALLOWED_TYPES: errs.append('unsupported_contract_type')
    ids=contract.get('cited_anchor_ids') or []
    if not (1<=len(ids)<=4 and all(str(x).isdigit() for x in ids)): errs.append('bad_anchor_ids')
    answers=[_clean(x) for x in (contract.get('exact_answers') or []) if _clean(x)]
    if len(answers)<2: errs.append('need_two_scoring_answers')
    chain=[_clean(x) for x in (contract.get('reasoning_chain') or []) if _clean(x)]
    if len(chain)<2: errs.append('reasoning_chain_lt2')
    tasks=[_clean(x) for x in (contract.get('task_plan') or []) if _clean(x)]
    if len(tasks)<2: errs.append('task_plan_lt2')
    if len(tasks)>=2 and not any(k in tasks[1] for k in ('앞','첫','이를','위 판단','그 판단','그 결과')):
        errs.append('task2_not_dependent')
    plan=_clean(contract.get('public_material_plan'))
    public=_clean(contract.get('public_material'))
    if len(plan)<20: errs.append('material_plan_too_short')
    if len(public)<30: errs.append('public_material_too_short')

    anchors=[]
    if ids and all(str(x).isdigit() for x in ids):
        con=sqlite3.connect(db_path)
        q='select id,domain,answer,evidence,source_name,page_no from anchors where id in (%s)'%(','.join('?'*len(ids)))
        anchors=[{'id':int(r[0]),'domain':str(r[1]),'answer':_clean(r[2]),'evidence':_clean(r[3]),'source_name':str(r[4] or ''),'page_no':int(r[5] or 0)} for r in con.execute(q,tuple(int(x) for x in ids)).fetchall()]
        con.close()
    if len(anchors)!=len(set(int(x) for x in ids if str(x).isdigit())): errs.append('anchor_not_found')
    if any(a['domain']!=domain for a in anchors): errs.append('cross_domain_anchor')
    source=' '.join(a['answer']+' '+a['evidence'] for a in anchors)
    ns=_norm(source)
    for a in answers:
        na=_norm(a)
        if len(na)>=2 and na not in ns:
            errs.append('answer_not_grounded:'+a[:24])
    # public plan must not simply quote exact answer tokens
    np=_norm(plan+' '+public)
    for a in answers:
        na=_norm(a)
        if len(na)>=3 and na in np:
            errs.append('answer_leaked_in_public_material')
            break
    # Reject near-verbatim source dumping: public material may use source-backed clues, not reproduce a long evidence sentence.
    npublic=_norm(public)
    for anc in anchors:
        nev=_norm(anc['evidence'])
        if len(nev)>=30 and (nev in npublic or npublic in nev):
            errs.append('near_verbatim_source_material')
            break
    if not _clean(contract.get('why_not_rote')):
        errs.append('missing_why_not_rote')
    return (not errs, {'errors':errs,'anchors':anchors})


def load_contracts(path=CONTRACT_FILE):
    p=Path(path)
    if not p.exists(): return []
    try:
        obj=json.loads(p.read_text(encoding='utf-8'))
        return obj.get('contracts',[]) if isinstance(obj,dict) else []
    except Exception:
        return []


def save_contracts(contracts,path=CONTRACT_FILE):
    p=Path(path)
    obj={'schema_version':'R51-CONTRACT-V1','contracts':contracts}
    p.write_text(json.dumps(obj,ensure_ascii=False,indent=2),encoding='utf-8')
    return str(p)


def merge_contracts(existing,new_rows):
    by={str(x.get('contract_id')):x for x in existing if x.get('contract_id')}
    for x in new_rows:
        by[str(x.get('contract_id'))]=x
    return list(by.values())


def contract_inventory(contracts,domains):
    rows={}; total=0
    for d in domains:
        ds=[x for x in contracts if x.get('domain')==d and x.get('status') in ('PYTHON_VALIDATED','AI_VERIFIED')]
        rows[d]={'count':len(ds),'types':sorted(set(str(x.get('contract_type')) for x in ds)),'target_met':len(ds)>=2}
        total+=min(2,len(ds))
    return {'domains':rows,'validated_slots':total,'target':len(domains)*2,'all_domains_two':all(v['target_met'] for v in rows.values())}


def contract_to_question(contract):
    answers=[_clean(x) for x in contract.get('exact_answers',[])][:2]
    tasks=[_clean(x) for x in contract.get('task_plan',[])][:2]
    if len(answers)<2 or len(tasks)<2: return None
    q={
      'domain':contract.get('domain',''),'topic':contract.get('topic','출제 가능 관계 계약 적용'),'points':4,
      'verifier':'mined_source_contract','pattern_id':'T4_CONTRACT22','capability_id':'contract:'+str(contract.get('contract_type','')),
      'question_type':'자료해석/적용','material_form':'계약자료',
      'intro':'다음 자료를 분석하여 <작성 방법>에 따라 쓰시오.',
      'passage':_clean(contract.get('public_material')),
      'conditions':['제시된 자료와 서브노트의 동일 기술 관계만을 근거로 판단한다.'],
      'tasks':tasks,'answer':answers,
      'solution':[str(x) for x in contract.get('reasoning_chain',[])[:2]],
      'subpoints':[2,2],
      'sources':[{'source_name':a.get('source_name',''),'page_no':a.get('page_no',0)} for a in (contract.get('validation',{}).get('anchors') or [])],
      'source_context_override':'\n'.join(a.get('evidence','') for a in (contract.get('validation',{}).get('anchors') or [])),
      'source_basis':'R51 one-time mined + Python-validated source reasoning contract',
      'derived_answer_flags':[True,True],
      'contract_id':contract.get('contract_id'),'contract_type':contract.get('contract_type'),
    }
    q['fingerprint']=_fp({k:q.get(k) for k in ('domain','contract_id','passage','answer')})
    return q
