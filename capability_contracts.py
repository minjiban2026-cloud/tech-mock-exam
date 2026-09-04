import json, os, re, sqlite3, hashlib
from pathlib import Path

CONTRACT_FILE='capability_contracts.json'
SCHEMA_VERSION='R55-RERUN-SAFE-V1'
ALLOWED_TYPES={
    'scenario_constraint_application',
    'error_repair_transfer',
    'threshold_decision',
    'ordered_sequence_application',
    'constraint_choice_justification',
    'structured_mapping_application',
    'comparative_case_discrimination',
    'relation_composition',
}


def _clean(s):
    return re.sub(r'\s+',' ',str(s or '').replace('\x01',' ')).strip()


def _norm(s):
    return re.sub(r'[^0-9A-Za-z가-힣]+','',_clean(s)).lower()


def _fp(x):
    return hashlib.sha256(json.dumps(x,ensure_ascii=False,sort_keys=True).encode()).hexdigest()[:24]


def _structural_score(answer,evidence,confidence=0.0):
    """Rank source facts by 4-point reasoning affordance, not merely DB confidence."""
    a=_clean(answer); e=_clean(evidence)
    t=a+' '+e
    score=float(confidence or 0.0)*0.25
    # explicit criteria / conditions / alternatives / processes / mappings / quantities
    marker_groups=[
        (r'경우|조건|때|하면|일 때|이상|이하|초과|미만|범위|한도|기준',2.5),
        (r'목적|대책|방지|예방|사용|적용|선정|선택|구분|비교',1.8),
        (r'원인|결과|영향|효과|때문|따라|의해|증가|감소|향상|저하',1.7),
        (r'1\)|2\)|3\)|①|②|③|⓵|⓶|⓷|단계|순서|절차|과정',2.2),
        (r'비례|반비례|공식|관계|=|%|mm|MPa|N\b|Pa\b|V\b|A\b|Hz|byte|비트',2.0),
        (r'\bvs\b|차이|반면|서로|각각|A\)|B\)|가\)|나\)',1.5),
        (r'교육목표|학문구조|교육방법|관점|유형|종류|분류',1.4),
    ]
    for pat,w in marker_groups:
        if re.search(pat,t,re.I): score+=w
    # richer sentences are better than isolated glossary fragments, but cap length reward.
    score += min(len(e),420)/140.0
    clauses=len(re.findall(r'[,;:/]| - | ▶ | \* |\)|①|②|③|⓵|⓶|⓷',e))
    score += min(clauses,8)*0.35
    if len(a)<=2: score-=1.5
    if len(e)<30: score-=2.0
    if e.endswith((':','/','·','▶','-')): score-=3.0
    if re.search(r'^[\W\d_]+$',e): score-=5.0
    return score


def _anchor_rows(db_path, domain, limit=72):
    """Diversified source retrieval.
    R51 used only the first high-confidence anchors; R52 scans a wider universe and ranks by
    reasoning affordance, then de-duplicates near-identical evidence.
    """
    con=sqlite3.connect(db_path)
    rows=con.execute('''select id,answer,evidence,source_name,page_no,confidence
                        from anchors where domain=? order by confidence desc,id asc limit 700''',(domain,)).fetchall()
    con.close()
    candidates=[]
    for rid,a,e,s,p,c in rows:
        a=_clean(a); e=_clean(e)
        if not (2<=len(a)<=100 and 20<=len(e)<=950):
            continue
        if len(re.findall(r'[가-힣A-Za-z]',e))<12:
            continue
        n=_norm(e)
        if len(n)<18:
            continue
        candidates.append((_structural_score(a,e,c),{'id':int(rid),'answer':a,'evidence':e,'source_name':str(s or ''),'page_no':int(p or 0),'confidence':float(c or 0)}))
    candidates.sort(key=lambda z:(-z[0],z[1]['id']))
    out=[]; seen=[]
    page_counts={}
    for sc,row in candidates:
        n=_norm(row['evidence'])
        # suppress duplicate/nested anchors that often come from the same note block
        if any((len(n)>=32 and (n in x or x in n)) for x in seen if len(x)>=32):
            continue
        pk=(row['source_name'],row['page_no'])
        if page_counts.get(pk,0)>=5:
            continue
        row['reasoning_affordance']=round(sc,3)
        out.append(row); seen.append(n); page_counts[pk]=page_counts.get(pk,0)+1
        if len(out)>=limit:
            break
    return out


def mining_packet(db_path, domain, limit=72):
    return {'domain':domain,'anchors':_anchor_rows(db_path,domain,limit)}


def _strip_json(text):
    t=str(text or '').strip()
    t=re.sub(r'^```(?:json)?\s*','',t)
    t=re.sub(r'\s*```$','',t)
    return t


def _existing_digest(existing):
    return [
        {'contract_type':x.get('contract_type'),'topic':x.get('topic'),'cited_anchor_ids':x.get('cited_anchor_ids'),'exact_answers':x.get('exact_answers')}
        for x in (existing or [])
    ]


def _mine(api_key, model, db_path, domain, wanted=2, existing=None, supplement=False):
    from openai import OpenAI
    packet=mining_packet(db_path,domain,limit=84 if supplement else 72)
    if not packet['anchors']:
        return []
    existing=list(existing or [])
    existing_types=sorted(set(str(x.get('contract_type')) for x in existing if x.get('contract_type')))
    client=OpenAI(api_key=api_key,timeout=90,max_retries=1)
    mode='보충 채굴' if supplement else '초기 채굴'
    prompt=f'''
너는 대한민국 중등 기술 임용시험의 "출제 가능 관계 계약"을 채굴하는 분석기다.
이번 작업은 {mode}이다. 문항을 직접 꾸미는 것이 아니라, 제공된 서브노트 원문만으로 4점 추론문항을 만들 수 있는 구조를 JSON으로 반환한다.

영역: {domain}
이번에 필요한 새 계약 수: {wanted}
이미 Python 검증된 계약(중복 금지):
{json.dumps(_existing_digest(existing),ensure_ascii=False)}
이미 확보한 유형: {existing_types}
원문 anchors (reasoning_affordance가 높을수록 구조적 정보가 풍부함):
{json.dumps(packet['anchors'],ensure_ascii=False)}

절대 규칙:
1. anchor 밖의 사실, 사례, 수치, 인과를 새로 만들지 않는다.
2. source evidence와 exact_answers는 출제자용 hidden ground truth다. 학생 지문에 정답 문장/정답어를 그대로 노출하지 않는다.
3. 단순 정의 회상, 특징 나열, 증가·감소 재진술, 원문 대책 그대로 쓰기, 개념명 맞히기는 금지한다.
4. 두 채점 요구는 반드시 하나의 사고사슬이어야 한다. task2가 task1의 판단 결과를 실제 입력으로 사용해야 한다.
5. public_material은 원문을 베끼는 자료가 아니라, source가 보장하는 조건을 "사례/학생 기록/비교 상황"으로 재구성한 자료여야 한다. source에 없는 기술 사실을 추가하면 안 된다.
6. 같은 계약을 표현만 바꾸어 중복 생성하지 않는다. 이미 확보한 계약과 anchor 조합·정답·사고 구조가 겹치면 버린다.
7. 가능한 계약 유형:
   - scenario_constraint_application: 복수 조건/특징을 조합해 사례를 판별하고 그 판단으로 후속 결정을 한다.
   - error_repair_transfer: 규칙으로 오류를 특정·수정하고 그 수정 원리를 다른 진술에 적용한다.
   - threshold_decision: 명시 수치/범위/기준과 사례를 비교하고 그 판단으로 후속 결정을 한다.
   - ordered_sequence_application: 자연적 절차 순서를 복원하고 그 결과를 후속 단계 판단에 적용한다.
   - constraint_choice_justification: 둘 이상의 방법/대안 중 사례 조건에 맞는 것을 선택하고, 같은 기준으로 부적합 대안을 배제한다.
   - structured_mapping_application: 원문에 명시된 2개 이상의 대응관계(관점↔성격, 조건↔방법 등)를 새로운 배열/사례에 매핑한 뒤 그 매핑을 이용해 후속 판단한다.
   - comparative_case_discrimination: 원문에 명시된 구별 기준을 이용해 두 사례를 서로 다른 범주/방법으로 판별하고, 결정적 차이를 설명한다.
   - relation_composition: 서로 직접 연결되는 두 source 관계를 A→B→C로 결합할 수 있을 때만, 첫 관계 판단을 두 번째 관계 적용의 입력으로 사용한다.
8. 관계가 명확하지 않은 anchor를 억지로 쓰지 않는다. 2개를 못 찾으면 0~1개만 반환해도 된다.
9. exact_answers는 짧고 채점 가능해야 하며 cited_anchor_ids의 answer/evidence에 문자열 수준으로 직접 근거가 있어야 한다.
10. public_material_plan에는 무엇을 변환해 보여주고 무엇을 숨기는지 명시한다.
11. public_material에 exact_answers를 넣지 않는다.
12. task_plan 두 번째 문장은 반드시 '앞 판단/첫 판단/그 결과/이를 이용하여' 중 하나를 포함한다.

JSON 하나만 출력:
{{"contracts":[
 {{
  "contract_type":"...",
  "topic":"...",
  "cited_anchor_ids":[1,2],
  "exact_answers":["...","..."],
  "reasoning_chain":["1단계 판단","2단계 적용"],
  "public_material_plan":"...",
  "public_material":"수험생에게 실제 제시할 자료",
  "task_plan":["...","앞 판단을 이용하여 ..."],
  "why_not_rote":"..."
 }}
]}}
'''
    r=client.responses.create(model=model,input=prompt,reasoning={'effort':'medium'})
    obj=json.loads(_strip_json(r.output_text))
    rows=obj.get('contracts',[]) if isinstance(obj,dict) else []
    valid=[]
    existing_fps={str(x.get('contract_id')) for x in existing if x.get('contract_id')}
    existing_signatures={(_norm(x.get('topic')),tuple(sorted(int(i) for i in (x.get('cited_anchor_ids') or []) if str(i).isdigit()))) for x in existing}
    for x in rows:
        ok,detail=validate_contract(db_path,domain,x)
        if ok:
            sig=(_norm(x.get('topic')),tuple(sorted(int(i) for i in (x.get('cited_anchor_ids') or []) if str(i).isdigit())))
            cid=_fp({'domain':domain,'type':x.get('contract_type'),'anchors':x.get('cited_anchor_ids'),'answers':x.get('exact_answers')})
            if cid in existing_fps or sig in existing_signatures:
                continue
            x=dict(x); x['domain']=domain; x['status']='PYTHON_VALIDATED'; x['validation']=detail
            x['contract_id']=cid; x['mining_mode']='SUPPLEMENT' if supplement else 'INITIAL'
            valid.append(x)
        if len(valid)>=wanted:
            break
    return valid


def mine_domain_contracts(api_key, model, db_path, domain, wanted=2):
    return _mine(api_key,model,db_path,domain,wanted=wanted,existing=[],supplement=False)


def mine_missing_contracts(api_key, model, db_path, domain, existing, target=2):
    """At most one extra AI call for a deficient domain; preserves all prior validated contracts."""
    ds=[x for x in (existing or []) if x.get('domain')==domain and x.get('status') in ('PYTHON_VALIDATED','AI_VERIFIED')]
    need=max(0,int(target)-len(ds))
    if need<=0:
        return []
    return _mine(api_key,model,db_path,domain,wanted=need,existing=ds,supplement=True)


def validate_contract(db_path, domain, contract):
    errs=[]
    typ=str(contract.get('contract_type',''))
    if typ not in ALLOWED_TYPES: errs.append('unsupported_contract_type')
    ids=contract.get('cited_anchor_ids') or []
    if not (1<=len(ids)<=4 and all(str(x).isdigit() for x in ids)): errs.append('bad_anchor_ids')
    answers=[_clean(x) for x in (contract.get('exact_answers') or []) if _clean(x)]
    if len(answers)<2: errs.append('need_two_scoring_answers')
    if len(set(_norm(a) for a in answers if _norm(a)))<2: errs.append('duplicate_scoring_answers')
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

    # Prevent answer/direct-source leakage into student-facing material.
    np=_norm(plan+' '+public)
    for a in answers:
        na=_norm(a)
        if len(na)>=3 and na in np:
            errs.append('answer_leaked_in_public_material')
            break
    npublic=_norm(public)
    for anc in anchors:
        nev=_norm(anc['evidence'])
        if len(nev)>=30 and (nev in npublic or npublic in nev):
            errs.append('near_verbatim_source_material')
            break

    # A 4-point contract needs non-trivial source support, not a single tiny glossary fragment.
    source_chars=sum(len(_norm(a['evidence'])) for a in anchors)
    if source_chars<45: errs.append('source_support_too_thin')
    if typ in ('constraint_choice_justification','structured_mapping_application','comparative_case_discrimination','relation_composition') and len(anchors)<2:
        # These architectures are only accepted from one anchor when that anchor itself is a rich multi-clause block.
        if not anchors or len(re.findall(r'[,;:/]| - |▶|①|②|③|⓵|⓶|⓷',anchors[0]['evidence']))<3:
            errs.append('contract_type_needs_multi_relation_source')
    if not _clean(contract.get('why_not_rote')):
        errs.append('missing_why_not_rote')
    return (not errs, {'errors':errs,'anchors':anchors,'source_support_chars':source_chars})


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
    obj={'schema_version':SCHEMA_VERSION,'contracts':contracts}
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
        rows[d]={'count':len(ds),'types':sorted(set(str(x.get('contract_type')) for x in ds)),'target_met':len(ds)>=2,'missing':max(0,2-len(ds))}
        total+=min(2,len(ds))
    return {'domains':rows,'validated_slots':total,'target':len(domains)*2,'all_domains_two':all(v['target_met'] for v in rows.values()),'missing_domains':[d for d,v in rows.items() if not v['target_met']]}


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
      'source_basis':'R52 source-contract mining + Python grounding/leak/dependency validation',
      'derived_answer_flags':[True,True],
      'contract_id':contract.get('contract_id'),'contract_type':contract.get('contract_type'),
    }
    q['fingerprint']=_fp({k:q.get(k) for k in ('domain','contract_id','passage','answer')})
    return q

# ---------------- R53 hybrid coverage: historical AI_VERIFIED + mined contracts ----------------
def historical_verified_types(db_path, domains, formula_domains=None):
    """Return only capability architectures with actual prior Judge PASS evidence."""
    from reasoning_capabilities import r50_validation_inventory
    inv=r50_validation_inventory(db_path,domains,formula_domains or set())
    return {d:list((inv.get('domains',{}).get(d,{}) or {}).get('certified_types',[]) or []) for d in domains}


def combined_coverage_inventory(db_path, contracts, domains, formula_domains=None):
    """Count DISTINCT capability architectures per domain.
    Repeated contracts of the same type count as one capability type.
    """
    base=historical_verified_types(db_path,domains,formula_domains)
    rows={}; total=0
    for d in domains:
        base_types=[]
        for x in base.get(d,[]):
            if x and x not in base_types: base_types.append(x)
        contract_rows=[x for x in (contracts or []) if x.get('domain')==d and x.get('status') in ('PYTHON_VALIDATED','AI_VERIFIED')]
        contract_types=[]
        for x in contract_rows:
            t=str(x.get('contract_type') or '').strip()
            if t and t not in contract_types: contract_types.append(t)
        combined=list(base_types)
        for t in contract_types:
            label='contract:'+t
            if label not in combined: combined.append(label)
        count=min(2,len(combined))
        rows[d]={
            'historical_ai_verified_types':base_types,
            'python_validated_contract_types':contract_types,
            'distinct_candidate_types':combined,
            'candidate_count':count,
            'target_met':count>=2,
            'missing':max(0,2-count),
        }
        total+=count
    return {'domains':rows,'candidate_slots':total,'target':len(domains)*2,
            'all_domains_two':all(v['target_met'] for v in rows.values()),
            'missing_domains':[d for d,v in rows.items() if not v['target_met']],
            'note':'R55: 기존 실제 Judge PASS capability + 서로 다른 PYTHON_VALIDATED contract 유형을 합산. 동일 contract_type 반복은 1개로 계산.'}


def _pair_packet(db_path, domain, limit=84):
    """Diversified anchors plus structurally promising anchor pairs for final missing slots."""
    anchors=_anchor_rows(db_path,domain,limit)
    pairs=[]
    def toks(x):
        return set(re.findall(r'[가-힣A-Za-z]{2,}',_clean(x)))
    for i,a in enumerate(anchors[:50]):
        for b in anchors[i+1:50]:
            if a['id']==b['id']: continue
            ta=toks(a['answer']+' '+a['evidence']); tb=toks(b['answer']+' '+b['evidence'])
            overlap=len(ta&tb)
            same_page=(a['source_name']==b['source_name'] and abs(int(a['page_no'])-int(b['page_no']))<=1)
            # zero-overlap neighbours are often unrelated list items on the same PDF page; never pair them.
            if overlap<1: continue
            sc=float(a.get('reasoning_affordance',0))+float(b.get('reasoning_affordance',0))+min(overlap,5)*1.0+(1.5 if same_page else 0)
            pairs.append((sc,{'anchor_ids':[a['id'],b['id']], 'answers':[a['answer'],b['answer']],
                              'evidence':[a['evidence'],b['evidence']], 'same_or_adjacent_page':same_page,'keyword_overlap':overlap}))
    pairs.sort(key=lambda z:-z[0])
    return {'domain':domain,'anchors':anchors,'anchor_pairs':[x for _,x in pairs[:24]]}


def mine_final_missing_contracts(api_key, model, db_path, domain, existing, domains, formula_domains=None):
    """One supplement call only when HYBRID coverage has a real missing slot."""
    inv=combined_coverage_inventory(db_path,existing,domains,formula_domains)
    need=int((inv.get('domains',{}).get(domain,{}) or {}).get('missing',0) or 0)
    if need<=0: return []
    from openai import OpenAI
    packet=_pair_packet(db_path,domain,limit=96)
    base=list((inv.get('domains',{}).get(domain,{}) or {}).get('historical_ai_verified_types',[]) or [])
    existing_domain=[x for x in (existing or []) if x.get('domain')==domain and x.get('status') in ('PYTHON_VALIDATED','AI_VERIFIED')]
    existing_types=sorted(set(str(x.get('contract_type')) for x in existing_domain if x.get('contract_type')))
    client=OpenAI(api_key=api_key,timeout=90,max_retries=1)
    prompt=f'''너는 대한민국 중등 기술 임용시험의 4점 문항용 "출제 가능 관계 계약"을 채굴한다.
영역: {domain}
이번에 필요한 서로 다른 새 capability 유형 수: {need}
이미 실제 Judge PASS로 인증된 구조(중복 금지): {base}
이미 Python 검증된 contract 유형(중복 금지): {existing_types}

원문 packet:
{json.dumps(packet,ensure_ascii=False)}

이번 호출의 목적은 부족한 최종 슬롯만 채우는 것이다. 억지로 만들지 말고 source가 충분할 때만 반환한다.
절대 규칙:
1. anchor 밖의 사실/수치/인과/사례를 만들지 않는다.
2. exact_answers와 source evidence는 hidden ground truth다. public_material에 정답어/정답문장을 그대로 노출하지 않는다.
3. 단순 정의/특징 나열/증가감소 재진술/대책 베끼기 금지.
4. task2는 task1의 판단 결과를 실제 입력으로 사용해야 한다.
5. 이미 인증된 구조 및 이미 존재하는 contract_type과 다른 사고 연산을 우선한다.
6. anchor_pairs는 실제로 같은 판단기준/방법선택/대응관계를 구성할 때만 결합한다.
7. 가능한 유형은 {sorted(ALLOWED_TYPES)} 중 하나만 사용한다.
8. exact_answers 2개 이상은 cited_anchor_ids의 answer/evidence에 직접 문자열 근거가 있어야 한다.
9. public_material은 30자 이상이며 source를 통째로 복사하지 않는다.
10. task_plan[1]에는 앞 판단/첫 판단/그 결과/이를 이용하여 중 하나를 반드시 포함한다.
11. 서로 다른 capability 유형을 못 찾으면 0개를 반환한다.

JSON 하나만 출력:
{{"contracts":[{{"contract_type":"...","topic":"...","cited_anchor_ids":[1,2],"exact_answers":["...","..."],"reasoning_chain":["...","..."],"public_material_plan":"...","public_material":"...","task_plan":["...","앞 판단을 이용하여 ..."],"why_not_rote":"..."}}]}}'''
    r=client.responses.create(model=model,input=prompt,reasoning={'effort':'high'})
    obj=json.loads(_strip_json(r.output_text)); rows=obj.get('contracts',[]) if isinstance(obj,dict) else []
    valid=[]; blocked_types=set(existing_types)
    for x in rows:
        typ=str(x.get('contract_type') or '')
        if typ in blocked_types: continue
        ok,detail=validate_contract(db_path,domain,x)
        if not ok: continue
        cid=_fp({'domain':domain,'type':typ,'anchors':x.get('cited_anchor_ids'),'answers':x.get('exact_answers')})
        x=dict(x); x['domain']=domain; x['status']='PYTHON_VALIDATED'; x['validation']=detail; x['contract_id']=cid; x['mining_mode']='R53_FINAL_GAP'
        valid.append(x); blocked_types.add(typ)
        if len(valid)>=need: break
    return valid


def select_hybrid_validation_contracts(db_path, contracts, domains, formula_domains=None):
    """Select only new contracts needed to complement historical verified capability types."""
    base=historical_verified_types(db_path,domains,formula_domains)
    selected=[]; gaps=[]
    for d in domains:
        need=max(0,2-len(set(base.get(d,[]) or [])))
        ds=[]
        for x in contracts or []:
            if x.get('domain')!=d or x.get('status') not in ('PYTHON_VALIDATED','AI_VERIFIED'): continue
            ok,_=validate_contract(db_path,d,x)
            if ok: ds.append(x)
        by_type={}
        for x in sorted(ds,key=lambda z:(str(z.get('contract_type','')),str(z.get('contract_id','')))):
            by_type.setdefault(str(x.get('contract_type','')),x)
        chosen=list(by_type.values())[:need]
        if len(chosen)<need: gaps.append({'domain':d,'need':need,'found':len(chosen)})
        selected.extend(chosen)
    return {'selected':selected,'gaps':gaps,'ready':not gaps,
            'historical_verified_count':sum(min(2,len(set(base.get(d,[]) or []))) for d in domains),
            'new_contract_count':len(selected)}

# ---------------- R54 contrast-set gap mining ----------------
SCHEMA_VERSION='R55-RERUN-SAFE-V1'

def _title_tokens(s):
    return set(re.findall(r'[가-힣A-Za-z]{2,}', _clean(s)))

def _char_bigrams(s):
    n=_norm(s)
    return {n[i:i+2] for i in range(max(0,len(n)-1)) if len(n[i:i+2])==2}

def _contrast_packet(db_path, domain, limit=120):
    """Build explicit contrast/choice sets from the DB. Unlike R53, same-source/page neighbors
    may be paired even with low token overlap when their titles/evidence show parallel structure.
    """
    anchors=_anchor_rows(db_path,domain,limit)
    cards=[]
    for i,a in enumerate(anchors[:90]):
        for b in anchors[i+1:90]:
            if a['id']==b['id']: continue
            same_src=a['source_name']==b['source_name']
            pgap=abs(int(a['page_no'])-int(b['page_no'])) if same_src else 99
            ta=_title_tokens(a['answer']+' '+a.get('evidence',''))
            tb=_title_tokens(b['answer']+' '+b.get('evidence',''))
            overlap=len(ta&tb)
            ba=_char_bigrams(a['answer']); bb=_char_bigrams(b['answer'])
            bsim=(len(ba&bb)/max(1,len(ba|bb))) if ba and bb else 0.0
            parallel=0
            ea=_clean(a['evidence']); eb=_clean(b['evidence'])
            for cue in ('경우','사용','방법','단계','종류','목적','조건','공법','시험','법칙','모멘트','치수','오차','측량','가공'):
                if cue in ea and cue in eb: parallel+=1
            if not same_src or pgap>2: continue
            if overlap<1 and bsim<0.18 and parallel<1: continue
            score=float(a.get('reasoning_affordance',0))+float(b.get('reasoning_affordance',0))
            score += min(overlap,5)*0.9 + bsim*5 + parallel*1.4 + (2.0 if pgap==0 else 1.0)
            cards.append((score,{
                'anchor_ids':[a['id'],b['id']], 'answers':[a['answer'],b['answer']],
                'evidence':[a['evidence'],b['evidence']], 'source_name':a['source_name'],
                'pages':[a['page_no'],b['page_no']], 'token_overlap':overlap,
                'title_bigram_similarity':round(bsim,3), 'parallel_cues':parallel
            }))
    cards.sort(key=lambda z:-z[0])
    # add 3-anchor local groups, useful for mapping/selection sets
    triples=[]
    bypage={}
    for a in anchors[:90]:
        bypage.setdefault((a['source_name'],a['page_no']),[]).append(a)
    for (src,p),rs in bypage.items():
        if len(rs)>=3:
            rs=sorted(rs,key=lambda x:-float(x.get('reasoning_affordance',0)))[:5]
            triples.append({'anchor_ids':[x['id'] for x in rs[:3]],'answers':[x['answer'] for x in rs[:3]],
                            'evidence':[x['evidence'] for x in rs[:3]],'source_name':src,'page':p})
    return {'domain':domain,'anchors':anchors[:70],'contrast_pairs':[x for _,x in cards[:36]],'local_mapping_sets':triples[:18]}

def mine_r54_gap_contracts(api_key, model, db_path, domain, existing, domains, formula_domains=None):
    inv=combined_coverage_inventory(db_path,existing,domains,formula_domains)
    need=int((inv.get('domains',{}).get(domain,{}) or {}).get('missing',0) or 0)
    if need<=0: return []
    from openai import OpenAI
    packet=_contrast_packet(db_path,domain)
    row=(inv.get('domains',{}).get(domain,{}) or {})
    base=list(row.get('historical_ai_verified_types',[]) or [])
    existing_domain=[x for x in (existing or []) if x.get('domain')==domain and x.get('status') in ('PYTHON_VALIDATED','AI_VERIFIED')]
    existing_types=sorted(set(str(x.get('contract_type')) for x in existing_domain if x.get('contract_type')))
    client=OpenAI(api_key=api_key,timeout=90,max_retries=1)
    prompt=f'''너는 대한민국 중등 기술 임용 4점 문항용 출제 관계 계약 채굴기다.
영역: {domain}
필요한 새 서로 다른 capability 수: {need}
기존 실제 Judge PASS 구조: {base}
기존 contract 유형: {existing_types}

DB에서 Python이 만든 contrast packet:
{json.dumps(packet,ensure_ascii=False)}

R54의 목적은 정의/특징을 문제로 바꾸는 것이 아니라, 같은 source 안의 서로 다른 조건·방법·기준을 비교하여 "조건 판별→방법/범주 선택→부적합 대안 배제 또는 후속 적용"이 가능한 계약만 찾는 것이다.
절대 규칙:
1. cited_anchor_ids는 contrast_pairs 또는 local_mapping_sets 안에서 실제로 함께 제공된 anchor 조합을 우선 사용한다.
2. source 밖의 사례 조건, 수치, 인과를 만들지 않는다.
3. public_material에는 exact_answers의 개념명/방법명을 그대로 쓰지 않는다.
4. 단순 개념명 맞히기, 정의 복사, 증가/감소 재진술, 대책 복사 금지.
5. task1은 사례 조건을 source 기준과 대조해 판단해야 하고, task2는 반드시 task1 결과를 이용하여 다른 대안을 배제하거나 후속 선택을 해야 한다.
6. 우선 유형은 constraint_choice_justification, comparative_case_discrimination, structured_mapping_application, error_repair_transfer이다. 기존 유형과 중복하지 않는다.
7. exact_answers 2개 이상은 cited anchor의 answer/evidence에 직접 문자열 근거가 있어야 한다.
8. public_material은 30자 이상이고 source 문장을 통째로 복사하지 않는다.
9. 억지로 만들지 말고 적합한 관계가 없으면 0개 반환한다.
10. task_plan[1]에는 앞 판단/첫 판단/그 결과/이를 이용하여 중 하나를 넣는다.
JSON 하나만 출력:
{{"contracts":[{{"contract_type":"...","topic":"...","cited_anchor_ids":[1,2],"exact_answers":["...","..."],"reasoning_chain":["...","..."],"public_material_plan":"...","public_material":"...","task_plan":["...","앞 판단을 이용하여 ..."],"why_not_rote":"..."}}]}}'''
    r=client.responses.create(model=model,input=prompt,reasoning={'effort':'high'})
    obj=json.loads(_strip_json(r.output_text)); rows=obj.get('contracts',[]) if isinstance(obj,dict) else []
    valid=[]; blocked=set(existing_types)
    allowed_preferred={'constraint_choice_justification','comparative_case_discrimination','structured_mapping_application','error_repair_transfer'}
    for x in rows:
        typ=str(x.get('contract_type') or '')
        if typ in blocked or typ not in allowed_preferred: continue
        ok,detail=validate_contract(db_path,domain,x)
        if not ok: continue
        cid=_fp({'domain':domain,'type':typ,'anchors':x.get('cited_anchor_ids'),'answers':x.get('exact_answers')})
        x=dict(x); x['domain']=domain; x['status']='PYTHON_VALIDATED'; x['validation']=detail
        x['contract_id']=cid; x['mining_mode']='R54_CONTRAST_GAP'
        valid.append(x); blocked.add(typ)
        if len(valid)>=need: break
    return valid

# ================= R56: failure-informed evidence-bound contracts =================
SCHEMA_VERSION='R56-EVIDENCE-BOUND-V1'
R56_ALLOWED_TYPES={
    'multi_evidence_constraint_resolution',
    'rule_composition_transfer',
    'contrastive_diagnosis_transfer',
}
R56_RETIRED_ZERO_PASS_TYPES={
    'scenario_constraint_application','structured_mapping_application','error_repair_transfer',
    'comparative_case_discrimination','threshold_decision','constraint_choice_justification',
    'ordered_sequence_application','relation_composition',
}

def _r56_char_overlap(a,b):
    aa=_char_bigrams(a); bb=_char_bigrams(b)
    return (len(aa & bb)/max(1,len(aa))) if aa else 0.0

def validate_r56_contract(db_path, domain, contract):
    """Strict pre-Judge gate based on the actual R55 0/13 failure profile.

    Goal: reject shallow/public-material-heavy contracts before any Judge call.
    Every student-visible clue must be bound to an exact source fragment and a 4-point
    contract must combine >=3 evidence bindings from >=2 anchors into a >=3-step chain.
    """
    errs=[]
    typ=str(contract.get('contract_type') or '')
    if typ not in R56_ALLOWED_TYPES: errs.append('r56_unsupported_type')
    ids=[int(x) for x in (contract.get('cited_anchor_ids') or []) if str(x).isdigit()]
    if len(set(ids))<2 or len(set(ids))>4: errs.append('r56_need_2_to_4_distinct_anchors')
    answers=[_clean(x) for x in (contract.get('exact_answers') or []) if _clean(x)]
    if len(answers)!=2 or len({_norm(x) for x in answers})!=2: errs.append('r56_need_two_distinct_answers')
    chain=[_clean(x) for x in (contract.get('reasoning_chain') or []) if _clean(x)]
    if len(chain)<3: errs.append('r56_reasoning_chain_lt3')
    bindings=list(contract.get('source_bindings') or [])
    if len(bindings)<3: errs.append('r56_need_at_least_3_source_bindings')
    roles={str(b.get('role') or '') for b in bindings}
    if not {'criterion','case_a','case_b'}.issubset(roles): errs.append('r56_missing_required_roles')
    if contract.get('task2_uses_task1') is not True: errs.append('r56_task_dependency_not_explicit')
    if str(contract.get('answer_kind') or '') not in {'용어','방법','단계','수치','관계','원리'}: errs.append('r56_bad_answer_kind')
    if len(_clean(contract.get('why_not_rote')))<20: errs.append('r56_why_not_rote_too_thin')

    anchors=[]
    if ids:
        con=sqlite3.connect(db_path)
        q='select id,domain,answer,evidence,source_name,page_no from anchors where id in (%s)'%(','.join('?'*len(set(ids))))
        for r in con.execute(q,tuple(sorted(set(ids)))).fetchall():
            anchors.append({'id':int(r[0]),'domain':str(r[1]),'answer':_clean(r[2]),'evidence':_clean(r[3]),'source_name':str(r[4] or ''),'page_no':int(r[5] or 0)})
        con.close()
    amap={a['id']:a for a in anchors}
    if len(amap)!=len(set(ids)): errs.append('r56_anchor_not_found')
    if any(a['domain']!=domain for a in anchors): errs.append('r56_cross_domain_anchor')
    whole=' '.join(a['answer']+' '+a['evidence'] for a in anchors)
    nwhole=_norm(whole)
    for ans in answers:
        if len(_norm(ans))<2 or _norm(ans) not in nwhole: errs.append('r56_answer_not_grounded:'+ans[:24])

    visible_parts=[]; bound_anchor_ids=set(); fragments=set()
    for i,b in enumerate(bindings):
        try: aid=int(b.get('anchor_id'))
        except Exception:
            errs.append(f'r56_binding_{i}_bad_anchor'); continue
        frag=_clean(b.get('source_fragment')); vis=_clean(b.get('visible_clue')); role=str(b.get('role') or '')
        if aid not in amap: errs.append(f'r56_binding_{i}_anchor_not_cited'); continue
        src=amap[aid]['answer']+' '+amap[aid]['evidence']
        if len(_norm(frag))<18 or _norm(frag) not in _norm(src): errs.append(f'r56_binding_{i}_fragment_not_exact_source')
        if len(_norm(vis))<15: errs.append(f'r56_binding_{i}_visible_too_short')
        ov=_r56_char_overlap(vis,frag)
        # Too little overlap usually means invented facts; too much is source-copy/answer-sheet material.
        if ov<0.22: errs.append(f'r56_binding_{i}_visible_weakly_grounded')
        if ov>0.88 and len(_norm(frag))>28: errs.append(f'r56_binding_{i}_near_verbatim')
        for ans in answers:
            na=_norm(ans)
            if len(na)>=3 and na in _norm(vis): errs.append(f'r56_binding_{i}_answer_leak')
        visible_parts.append(vis); bound_anchor_ids.add(aid); fragments.add(_norm(frag))
    if len(bound_anchor_ids)<2: errs.append('r56_bindings_use_lt2_anchors')
    if len(fragments)<3: errs.append('r56_need_3_distinct_source_fragments')

    public=' '.join(visible_parts)
    if len(_norm(public))<100: errs.append('r56_public_evidence_too_short')
    # Require genuinely different clues, not three copies of one sentence.
    for i in range(len(visible_parts)):
        for j in range(i+1,len(visible_parts)):
            if _r56_char_overlap(visible_parts[i],visible_parts[j])>0.82:
                errs.append('r56_duplicate_visible_clues'); break

    return (not errs, {'errors':errs,'anchors':anchors,'binding_count':len(bindings),
                       'bound_anchor_count':len(bound_anchor_ids),'public_chars':len(_norm(public))})

def _r56_contract_digest(existing,domain):
    out=[]
    for x in existing or []:
        if x.get('domain')==domain and x.get('status')=='R56_PYTHON_VALIDATED':
            out.append({'type':x.get('contract_type'),'anchors':x.get('cited_anchor_ids'),'answers':x.get('exact_answers')})
    return out

def mine_r56_gap_contracts(api_key, model, db_path, domain, existing, domains, formula_domains=None):
    """Mine only missing R56 slots. One call/domain; no Judge-feedback retry."""
    inv=combined_coverage_inventory(db_path,existing,domains,formula_domains)
    need=int((inv.get('domains',{}).get(domain,{}) or {}).get('missing',0) or 0)
    if need<=0: return []
    from openai import OpenAI
    packet=_contrast_packet(db_path,domain,limit=140)
    client=OpenAI(api_key=api_key,timeout=90,max_retries=1)
    prompt=f'''너는 대한민국 중등 기술 임용시험 4점 문항의 "근거 결속형 reasoning contract" 분석기다.
영역: {domain}
필요한 새 구조 수: {need}

R55에서 아래 구조들은 실제 Judge 결과 모두 0 PASS였으므로 절대 사용하지 않는다:
{sorted(R56_RETIRED_ZERO_PASS_TYPES)}
실패 특징: difficulty_fit 13/13, exam_realism 13/13, inferential_distance 13/13, task_distinctness 12/13 미달.
따라서 단순 개념회상/자료에서 정답찾기/자료 재진술은 금지한다.

이미 확보한 R56 계약:
{json.dumps(_r56_contract_digest(existing,domain),ensure_ascii=False)}

서브노트 packet:
{json.dumps(packet,ensure_ascii=False)}

허용 구조는 딱 3개다.
- multi_evidence_constraint_resolution: 서로 다른 3개 이상 근거를 종합해 첫 판단을 내리고, 그 판단 기준을 두 번째 사례에 적용한다.
- rule_composition_transfer: 서로 다른 두 source 관계를 실제 A→B→C 연쇄로 결합하고, 완성한 연쇄를 새 자료에 적용한다.
- contrastive_diagnosis_transfer: 두 방법/범주의 결정적 구별 기준을 복수 근거에서 도출하고, 그 기준으로 제3 사례의 오류/선택을 판정한다.

각 contract의 source_bindings는 학생에게 보여줄 모든 핵심 단서를 원문에 결속한다.
source_bindings 각 항목:
- anchor_id: cited_anchor_ids 중 하나
- source_fragment: 해당 anchor의 answer/evidence에 "정확히 포함되는" 18자 이상 원문 조각
- visible_clue: source_fragment의 의미를 유지하되 정답어를 제거하고 일부 순서/표현만 바꾼 학생용 단서(새 기술사실 추가 금지)
- role: criterion, case_a, case_b 중 하나

절대 조건:
1. 서로 다른 anchor 최소 2개, source_binding 최소 3개.
2. criterion/case_a/case_b role이 모두 있어야 한다.
3. exact_answers는 정확히 2개이며 모두 source에 직접 존재해야 한다. exact_answers[0]은 사례 A의 결과, exact_answers[1]은 사례 B의 결과다.
4. visible_clue에 exact_answers를 직접 쓰지 않는다.
5. reasoning_chain은 최소 3단계: 근거 결합 → 1차 판단 → 그 판단 기준의 2차 적용.
6. task2_uses_task1=true.
7. source 밖 사례/수치/인과를 만들지 않는다.
8. 1번만 읽고 바로 답이 보이는 계약은 반환하지 않는다. 최소 2개 단서를 결합하지 않으면 1차 답을 결정할 수 없어야 한다.
9. 2차 답은 1차 판단 기준을 사용하지 않으면 풀 수 없어야 한다.
10. 충분한 구조가 없으면 억지로 채우지 말고 적게 반환한다.

JSON 하나만 출력:
{{"contracts":[{{
 "contract_type":"multi_evidence_constraint_resolution",
 "topic":"...",
 "cited_anchor_ids":[1,2,3],
 "exact_answers":["정답1","정답2"],
 "answer_kind":"용어|방법|단계|수치|관계|원리 중 하나",
 "source_bindings":[
   {{"anchor_id":1,"source_fragment":"원문 그대로","visible_clue":"정답어를 숨긴 학생용 단서","role":"criterion"}},
   {{"anchor_id":2,"source_fragment":"원문 그대로","visible_clue":"학생용 단서","role":"case_a"}},
   {{"anchor_id":3,"source_fragment":"원문 그대로","visible_clue":"학생용 단서","role":"case_b"}}
 ],
 "reasoning_chain":["근거 결합","1차 판단","1차 판단 기준의 2차 적용"],
 "task2_uses_task1":true,
 "why_not_rote":"두 개 이상의 독립 단서를 결합해야만 1차 판단이 가능하고 그 결과가 2차 판단의 기준이 되는 이유"
}}]}}'''
    r=client.responses.create(model=model,input=prompt,reasoning={'effort':'high'})
    obj=json.loads(_strip_json(r.output_text)); rows=obj.get('contracts',[]) if isinstance(obj,dict) else []
    existing_types={x.get('contract_type') for x in existing or [] if x.get('domain')==domain and x.get('status')=='R56_PYTHON_VALIDATED'}
    valid=[]
    for x in rows:
        typ=str(x.get('contract_type') or '')
        if typ in existing_types: continue
        ok,detail=validate_r56_contract(db_path,domain,x)
        if not ok: continue
        cid=_fp({'v':'R56','domain':domain,'type':typ,'anchors':x.get('cited_anchor_ids'),'answers':x.get('exact_answers'),'bindings':x.get('source_bindings')})
        x=dict(x); x['domain']=domain; x['status']='R56_PYTHON_VALIDATED'; x['validation']=detail; x['contract_id']=cid; x['mining_mode']='R56_EVIDENCE_BOUND'
        valid.append(x); existing_types.add(typ)
        if len(valid)>=need: break
    return valid

def combined_coverage_inventory(db_path, contracts, domains, formula_domains=None):
    """R56 coverage: historical actual Judge PASS + only strict R56 validated types.
    All R51-R55 0-pass contract architectures are ignored even if present in uploaded JSON.
    """
    base=historical_verified_types(db_path,domains,formula_domains)
    rows={}; total=0
    for d in domains:
        hist=sorted(set(base.get(d,[]) or []))
        valid=[]
        for x in contracts or []:
            if x.get('domain')!=d or x.get('status') not in ('R56_PYTHON_VALIDATED','R56_AI_VERIFIED'): continue
            if x.get('contract_type') not in R56_ALLOWED_TYPES: continue
            ok,_=validate_r56_contract(db_path,d,x)
            if ok: valid.append(x)
        ctypes=sorted(set(str(x.get('contract_type')) for x in valid))
        distinct=hist+['contract:'+x for x in ctypes if ('contract:'+x) not in hist]
        count=min(2,len(set(distinct))); total+=count
        rows[d]={'historical_ai_verified_types':hist,'r56_python_validated_contract_types':ctypes,
                 'distinct_candidate_types':distinct,'candidate_count':count,'target_met':count>=2,'missing':max(0,2-count)}
    return {'domains':rows,'candidate_slots':total,'target':2*len(domains),'all_domains_two':total>=2*len(domains),
            'missing_domains':[d for d,v in rows.items() if not v['target_met']],
            'note':'R56: R55 신규 contract 0/13 실패를 반영해 구형 contract_type은 coverage에서 제외. 기존 실제 Judge PASS + R56 evidence-bound 계약만 계산.'}

def select_hybrid_validation_contracts(db_path, contracts, domains, formula_domains=None):
    base=historical_verified_types(db_path,domains,formula_domains)
    selected=[]; gaps=[]
    for d in domains:
        need=max(0,2-len(set(base.get(d,[]) or [])))
        ds=[]
        for x in contracts or []:
            if x.get('domain')!=d or x.get('status') not in ('R56_PYTHON_VALIDATED','R56_AI_VERIFIED'): continue
            ok,_=validate_r56_contract(db_path,d,x)
            if ok: ds.append(x)
        by_type={}
        for x in sorted(ds,key=lambda z:(str(z.get('contract_type','')),str(z.get('contract_id','')))):
            by_type.setdefault(str(x.get('contract_type','')),x)
        chosen=list(by_type.values())[:need]
        if len(chosen)<need: gaps.append({'domain':d,'need':need,'found':len(chosen)})
        selected.extend(chosen)
    return {'selected':selected,'gaps':gaps,'ready':not gaps,
            'historical_verified_count':sum(min(2,len(set(base.get(d,[]) or []))) for d in domains),
            'new_contract_count':len(selected)}

def contract_to_question(contract):
    """R56 deterministic rendering: group source-bound clues by role; no free-form AI passage survives."""
    if contract.get('status') not in ('R56_PYTHON_VALIDATED','R56_AI_VERIFIED'): return None
    bindings=list(contract.get('source_bindings') or [])
    if len(bindings)<3: return None
    grouped={'criterion':[],'case_a':[],'case_b':[]}
    for b in bindings:
        role=str(b.get('role') or '')
        if role in grouped: grouped[role].append(_clean(b.get('visible_clue')))
    if not all(grouped.values()): return None
    def _fmt(title,rows):
        return title+'\n'+\
            '\n'.join(f"- {x}" for x in rows if x)
    passage='\n\n'.join([_fmt('[판단 기준 자료]',grouped['criterion']),_fmt('[사례 A]',grouped['case_a']),_fmt('[사례 B]',grouped['case_b'])])
    answers=[_clean(x) for x in contract.get('exact_answers',[])][:2]
    if len(answers)!=2: return None
    kind=str(contract.get('answer_kind') or '용어')
    typ=str(contract.get('contract_type'))
    if typ=='rule_composition_transfer':
        tasks=[f'판단 기준 자료와 사례 A의 관계를 순서대로 결합하여 사례 A에 해당하는 {kind}을(를) 쓰시오.', f'①에서 완성한 관계를 판단 기준으로 사용하여 사례 B에 해당하는 {kind}을(를) 쓰시오.']
    elif typ=='contrastive_diagnosis_transfer':
        tasks=[f'판단 기준 자료를 이용하여 사례 A의 결정적 구별 기준을 판단하고, 사례 A에 해당하는 {kind}을(를) 쓰시오.', f'①에서 사용한 구별 기준을 그대로 적용하여 사례 B에 해당하는 {kind}을(를) 쓰시오.']
    else:
        tasks=[f'판단 기준 자료의 조건을 사례 A와 종합하여 사례 A에 해당하는 {kind}을(를) 쓰시오.', f'①에서 사용한 판단 기준을 그대로 적용하여 사례 B에 해당하는 {kind}을(를) 쓰시오.']
    q={'domain':contract.get('domain',''),'topic':contract.get('topic','근거 결속형 자료 적용'),'points':4,
       'verifier':'r56_evidence_bound_contract','pattern_id':'T4_R56_EVIDENCE_CHAIN','capability_id':'contract:'+typ,
       'question_type':'자료해석/추론','material_form':'복수 근거 자료','intro':'다음 자료를 분석하여 <작성 방법>에 따라 쓰시오.',
       'passage':passage,'conditions':['①에서 판단 기준 자료와 사례 A를 연결하고, ②에서는 ①에서 사용한 동일한 판단 기준을 사례 B에 적용한다.'],
       'tasks':tasks,'answer':answers,'solution':[str(x) for x in contract.get('reasoning_chain',[])[:3]],'subpoints':[2,2],
       'sources':[{'source_name':a.get('source_name',''),'page_no':a.get('page_no',0)} for a in (contract.get('validation',{}).get('anchors') or [])],
       'source_context_override':'\n'.join(a.get('evidence','') for a in (contract.get('validation',{}).get('anchors') or [])),
       'source_basis':'R56 evidence-bound source fragments + strict Python pre-Judge validation',
       'derived_answer_flags':[True,True],'contract_id':contract.get('contract_id'),'contract_type':typ}
    q['fingerprint']=_fp({k:q.get(k) for k in ('domain','contract_id','passage','answer')})
    return q

# ================= R57: one-click pre-Judged exam-style synthesis =================
SCHEMA_VERSION='R57-ONECLICK-JUDGED-V1'
R57_ALLOWED_TYPES={'exam_mixed_evidence_chain','exam_error_diagnosis_transfer','exam_contrastive_application'}
R57_RETIRED_TYPES=set(R56_RETIRED_ZERO_PASS_TYPES) | set(R56_ALLOWED_TYPES)

def _r57_style_examples(db_path, domain, limit=3):
    con=sqlite3.connect(db_path); rows=[]
    q="select s.name,p.page_no,p.text from pages p join sources s on s.id=p.source_id where s.kind='past_exam' and s.name like ? and length(p.text)>=250 order by p.page_no limit ?"
    for name,pno,text in con.execute(q,(f'%{domain}%',limit)).fetchall():
        t=_clean(text)
        if '[4점]' in t or '4점' in t: rows.append({'source_name':name,'page_no':pno,'text':t[:2200]})
    if len(rows)<limit:
        q2="select s.name,p.page_no,p.text from pages p join sources s on s.id=p.source_id where s.kind='official_exam' and length(p.text)>=350 and p.text like '%[4점]%' order by s.id desc,p.page_no limit 12"
        for name,pno,text in con.execute(q2).fetchall():
            t=_clean(text)
            if not any(x['source_name']==name and x['page_no']==pno for x in rows): rows.append({'source_name':name,'page_no':pno,'text':t[:2200]})
            if len(rows)>=limit: break
    con.close(); return rows[:limit]

def _r57_source_packet(db_path, domain, limit=150):
    cp=_contrast_packet(db_path,domain,limit=min(120,limit))
    return {'contrast_pairs':list(cp.get('contrast_pairs',[]))[:36], 'local_mapping_sets':list(cp.get('local_mapping_sets',[]))[:18], 'anchors':list(cp.get('anchors',[]))[:60]}

def _r57_words(s):
    return set(re.findall(r'[가-힣A-Za-z0-9]{2,}', _clean(s)))

def validate_r57_contract(db_path, domain, c):
    errs=[]; typ=str(c.get('contract_type') or ''); pattern=str(c.get('pattern_id') or '')
    if typ not in R57_ALLOWED_TYPES: errs.append('R57_UNSUPPORTED_TYPE')
    if pattern not in {'P_MIX4','P_ERR4','P_COMPARE4'}: errs.append('R57_BAD_EXAM_PATTERN')
    ids=[]
    for x in c.get('cited_anchor_ids') or []:
        try: ids.append(int(x))
        except Exception: pass
    ids=list(dict.fromkeys(ids))
    if len(ids)<2 or len(ids)>5: errs.append('R57_NEED_2_TO_5_ANCHORS')
    answers=[_clean(x) for x in (c.get('exact_answers') or []) if _clean(x)]
    if len(answers)!=2 or len({_norm(x) for x in answers})!=2: errs.append('R57_NEED_2_DISTINCT_ANSWERS')
    tasks=[_clean(x) for x in (c.get('tasks') or []) if _clean(x)]
    if len(tasks)!=2: errs.append('R57_NEED_2_TASKS')
    elif not any(k in tasks[1] for k in ('①','앞의','위의 판단','판단 기준','결과를 이용','결과를 바탕')): errs.append('R57_TASK2_NOT_DEPENDENT')
    if c.get('task2_uses_task1') is not True: errs.append('R57_TASK_DEPENDENCY_FLAG_FALSE')
    if len(_clean(c.get('dependency_note')))<25: errs.append('R57_DEPENDENCY_NOTE_THIN')
    blocks=list(c.get('material_blocks') or [])
    if len(blocks)<2 or len(blocks)>4: errs.append('R57_MATERIAL_BLOCK_COUNT')
    visible='\n'.join(_clean(b.get('visible_text')) for b in blocks); nv=_norm(visible)
    if len(nv)<180: errs.append('R57_MATERIAL_TOO_SHORT')
    if len(nv)>1400: errs.append('R57_MATERIAL_TOO_LONG')
    if len(tasks)==2 and len(_norm(' '.join(tasks)))<45: errs.append('R57_TASKS_TOO_THIN')
    for ans in answers:
        na=_norm(ans)
        if len(na)>=2 and (na in nv or any(na in _norm(t) for t in tasks)): errs.append('R57_DIRECT_ANSWER_LEAK:'+ans[:30])
    anchors=[]
    if ids:
        con=sqlite3.connect(db_path); q='select id,domain,topic,answer,evidence,source_name,page_no from anchors where id in (%s)'%(','.join('?'*len(ids)))
        for r in con.execute(q,tuple(ids)).fetchall(): anchors.append({'id':int(r[0]),'domain':str(r[1]),'topic':_clean(r[2]),'answer':_clean(r[3]),'evidence':_clean(r[4]),'source_name':str(r[5] or ''),'page_no':int(r[6] or 0)})
        con.close()
    amap={a['id']:a for a in anchors}
    if len(amap)!=len(ids): errs.append('R57_ANCHOR_NOT_FOUND')
    if any(a['domain']!=domain for a in anchors): errs.append('R57_CROSS_DOMAIN_ANCHOR')
    nwhole=_norm(' '.join(a['answer']+' '+a['evidence'] for a in anchors))
    for ans in answers:
        if _norm(ans) not in nwhole: errs.append('R57_ANSWER_NOT_GROUNDED:'+ans[:30])
    source_names={a['source_name'] for a in anchors if a['source_name']}
    if len(source_names)>2: errs.append('R57_TOO_MANY_SOURCE_DOCS')
    topics=[_r57_words(a['topic']+' '+a['answer']) for a in anchors]
    if len(topics)>=2:
        shared=set()
        for i in range(len(topics)):
            for j in range(i+1,len(topics)): shared |= topics[i]&topics[j]
        pages=[a['page_no'] for a in anchors]; near=(len(source_names)==1 and pages and max(pages)-min(pages)<=3)
        if not shared and not near: errs.append('R57_WEAK_TOPIC_COHERENCE')
    binding_total=0; used_anchor_ids=set(); support_for={0:set(),1:set()}
    for bi,b in enumerate(blocks):
        txt=_clean(b.get('visible_text')); binds=list(b.get('bindings') or [])
        if not txt or len(_norm(txt))<50: errs.append(f'R57_BLOCK_{bi}_TOO_SHORT')
        if not binds: errs.append(f'R57_BLOCK_{bi}_NO_BINDING')
        for bd in binds:
            try: aid=int(bd.get('anchor_id'))
            except Exception: errs.append(f'R57_BLOCK_{bi}_BAD_ANCHOR'); continue
            frag=_clean(bd.get('source_fragment'))
            if aid not in amap: errs.append(f'R57_BLOCK_{bi}_UNCITED_ANCHOR'); continue
            src=amap[aid]['answer']+' '+amap[aid]['evidence']
            if len(_norm(frag))<18 or _norm(frag) not in _norm(src): errs.append(f'R57_BLOCK_{bi}_FRAGMENT_NOT_EXACT')
            ov=_r56_char_overlap(txt,frag)
            if ov<0.12: errs.append(f'R57_BLOCK_{bi}_WEAK_GROUNDING')
            if ov>0.90 and len(_norm(frag))>35: errs.append(f'R57_BLOCK_{bi}_NEAR_VERBATIM')
            binding_total+=1; used_anchor_ids.add(aid)
            for ai in bd.get('supports_answers') or []:
                try: ai=int(ai)
                except Exception: continue
                if ai in support_for: support_for[ai].add(aid)
    if binding_total<3: errs.append('R57_NEED_3_BINDINGS')
    if len(used_anchor_ids)<2: errs.append('R57_BINDINGS_LT2_ANCHORS')
    if len(support_for[0])<2: errs.append('R57_TASK1_SINGLE_CLUE_LOOKUP')
    if len(support_for[1])<1: errs.append('R57_TASK2_NO_SOURCE_SUPPORT')
    chain=[_clean(x) for x in c.get('reasoning_chain') or [] if _clean(x)]
    if len(chain)<3: errs.append('R57_REASONING_LT3')
    elif not any(k in chain[-1] for k in ('①','첫 판단','판단 기준','앞의 결과','적용')): errs.append('R57_CHAIN_NO_TRANSFER')
    return (not errs, {'errors':errs,'anchors':anchors,'binding_total':binding_total,'support_for_answer0':sorted(support_for[0]),'support_for_answer1':sorted(support_for[1]),'public_chars':len(nv),'source_names':sorted(source_names)})

def r57_contract_to_question(c):
    if c.get('status') not in ('R57_PYTHON_VALIDATED','R57_AI_VERIFIED'): return None
    blocks=list(c.get('material_blocks') or []); labels=['(가)','(나)','(다)','(라)']
    if len(blocks)<2: return None
    passage='\n\n'.join(f"{labels[i]} {_clean(b.get('visible_text'))}" for i,b in enumerate(blocks))
    answers=[_clean(x) for x in c.get('exact_answers',[])][:2]; tasks=[_clean(x) for x in c.get('tasks',[])][:2]
    if len(answers)!=2 or len(tasks)!=2: return None
    anchors=(c.get('validation') or {}).get('anchors') or []
    evidence=[a.get('evidence','') for a in anchors if a.get('evidence')]
    source_context='\n\n'.join(f"[{a.get('source_name','')} p.{a.get('page_no',0)} / anchor {a.get('id')}]\n정답/개념: {a.get('answer','')}\n근거: {a.get('evidence','')}" for a in anchors)
    q={'domain':c.get('domain',''),'topic':c.get('topic',''),'points':4,'verifier':'r57_exam_style_source_bound','pattern_id':c.get('pattern_id','P_MIX4'),'capability_id':'contract:'+str(c.get('contract_type','')),'question_type':'자료해석/판단/적용','material_form':str(c.get('material_form') or '자료+조건'),'intro':'다음 <자료>를 읽고 <작성 방법>에 따라 순서대로 서술하시오.','passage':passage,'conditions':[_clean(x) for x in (c.get('conditions') or []) if _clean(x)],'tasks':tasks,'answer':answers,'solution':[_clean(x) for x in (c.get('reasoning_chain') or [])],'subpoints':[2,2],'evidence':evidence,'sources':[{'source_name':a.get('source_name',''),'page_no':a.get('page_no',0)} for a in anchors],'source_context_override':source_context,'source_basis':'R57 exact anchor fragments + actual past-exam structural reference','derived_answer_flags':[True,True],'contract_id':c.get('contract_id'),'contract_type':c.get('contract_type')}
    q['fingerprint']=_fp({k:q.get(k) for k in ('domain','contract_id','passage','tasks','answer')}); return q

def _r57_existing_verified(existing,domain):
    return [x for x in existing or [] if x.get('domain')==domain and x.get('status')=='R57_AI_VERIFIED' and x.get('contract_type') in R57_ALLOWED_TYPES]

def synthesize_r57_pool(api_key, model, db_path, domain, existing, need, pool_size=None):
    from openai import OpenAI
    pool_size=int(pool_size or (4 if need<=1 else 6)); style=_r57_style_examples(db_path,domain,limit=3); packet=_r57_source_packet(db_path,domain,limit=160)
    existing_types=sorted({x.get('contract_type') for x in _r57_existing_verified(existing,domain)}); client=OpenAI(api_key=api_key,timeout=120,max_retries=1)
    prompt=f'''너는 대한민국 중등 기술 임용 1차 4점 문항의 구조 설계자다.\n영역: {domain}\n목표: Judge 피드백을 보기 전에 서로 다른 완성 후보 {pool_size}개를 한 번에 설계한다.\n현재 필요한 PASS 슬롯: {need}\n이미 AI_VERIFIED된 새 유형: {existing_types}\n\n이전 실제 실패를 반드시 피한다:\n- R55 신규 contract 13/13 REJECT\n- difficulty_fit 13/13, exam_realism 13/13, inferential_distance 13/13 미달\n- task_distinctness 12/13 미달\n- DIRECT_ANSWER_LEAK, ROTE_ONLY, AMBIGUOUS, DECORATIVE_MATERIAL 반복\n따라서 자료에서 용어 찾아쓰기, 정의 복사, 독립 소문항 병렬 나열은 만들지 않는다.\n\n실제 기출 구조 예시(내용을 베끼지 말고 구조만 참고):\n{json.dumps(style,ensure_ascii=False)}\n\n정답/기술 사실의 유일한 근거인 서브노트 packet:\n{json.dumps(packet,ensure_ascii=False)}\n\n허용 구조:\n1) exam_mixed_evidence_chain / P_MIX4: 서로 다른 근거 2개 이상을 결합해 ①을 판단하고, ①의 판단기준을 ②의 새 자료에 적용.\n2) exam_error_diagnosis_transfer / P_ERR4: 오류의 결정적 원인을 복수 근거로 진단해 ①을 정하고, 그 기준을 다른 상황의 개선/판정에 적용.\n3) exam_contrastive_application / P_COMPARE4: 두 방법/범주의 결정적 차이를 복수 근거로 도출해 ①을 판단하고, 동일 기준으로 ②를 판별.\n\n절대 규칙:\n- 새 기술 사실·수치·인과·사례를 창작하지 않는다. 모든 기술적 내용은 bindings의 exact source_fragment로 뒷받침한다.\n- exact_answers는 정확히 2개이고 cited anchor의 answer/evidence에 실제 존재한다.\n- exact_answers 문자열을 material_blocks, tasks, conditions에 직접 노출하지 않는다.\n- ①의 답은 서로 다른 anchor 2개 이상을 결합해야 결정되게 한다. supports_answers에 0/1을 기록한다.\n- ②는 ① 결과 없이는 풀 수 없게 하고 task 문장에 ① 의존성을 명시한다.\n- material_blocks 2~4개, 각 visible_text는 2~5문장의 자연스러운 임용 자료로 작성하되 source 밖 사실 추가 금지.\n- cited anchors는 가급적 한 source 또는 인접 페이지의 같은 주제군으로 묶는다.\n- reasoning_chain 최소 3단계, 마지막은 ① 판단을 ②에 적용.\n- 충분하지 않으면 억지로 만들지 않는다.\n\nJSON 하나만 출력:\n{{"contracts":[{{"contract_type":"exam_mixed_evidence_chain","pattern_id":"P_MIX4","topic":"...","cited_anchor_ids":[1,2,3],"exact_answers":["정답1","정답2"],"material_form":"대화|표+설명|사례자료|조건자료","material_blocks":[{{"visible_text":"학생용 자료","bindings":[{{"anchor_id":1,"source_fragment":"원문에 정확히 포함되는 18자 이상 조각","supports_answers":[0]}}]}},{{"visible_text":"두 번째 자료","bindings":[{{"anchor_id":2,"source_fragment":"정확한 원문 조각","supports_answers":[0]}},{{"anchor_id":3,"source_fragment":"정확한 원문 조각","supports_answers":[1]}}]}}],"conditions":[],"tasks":["① 복수 자료를 종합하여 ...을 판단하고 쓰시오.","② ①에서 사용한 판단 기준을 적용하여 ...을 쓰시오."],"reasoning_chain":["서로 다른 두 근거 결합","① 판단 도출","①의 판단 기준을 ②에 적용"],"task2_uses_task1":true,"dependency_note":"②가 ①의 결과 없이는 독립적으로 풀리지 않는 구체적 이유"}}]}}'''
    r=client.responses.create(model=model,input=prompt,reasoning={'effort':'high'}); obj=json.loads(_strip_json(r.output_text)); rows=obj.get('contracts',[]) if isinstance(obj,dict) else []
    out=[]; seen=set()
    for x in rows[:pool_size*2]:
        typ=str(x.get('contract_type') or ''); key=(typ,tuple(x.get('cited_anchor_ids') or []),tuple(x.get('exact_answers') or []))
        if key in seen: continue
        seen.add(key); ok,detail=validate_r57_contract(db_path,domain,x)
        if not ok: continue
        cid=_fp({'v':'R57','domain':domain,'type':typ,'anchors':x.get('cited_anchor_ids'),'answers':x.get('exact_answers'),'blocks':x.get('material_blocks')}); y=dict(x); y['domain']=domain; y['status']='R57_PYTHON_VALIDATED'; y['validation']=detail; y['contract_id']=cid; y['mining_mode']='R57_FIXED_POOL'; out.append(y)
        if len(out)>=pool_size: break
    return out

def combined_coverage_inventory(db_path, contracts, domains, formula_domains=None):
    base=historical_verified_types(db_path,domains,formula_domains); rows={}; total=0
    for d in domains:
        hist=sorted(set(base.get(d,[]) or [])); verified=[]
        for x in contracts or []:
            if x.get('domain')!=d or x.get('status')!='R57_AI_VERIFIED' or x.get('contract_type') not in R57_ALLOWED_TYPES: continue
            ok,_=validate_r57_contract(db_path,d,x)
            if ok: verified.append(x)
        ctypes=sorted(set(str(x.get('contract_type')) for x in verified)); distinct=hist+['contract:'+t for t in ctypes]; count=min(2,len(set(distinct))); total+=count
        rows[d]={'historical_ai_verified_types':hist,'r57_ai_verified_contract_types':ctypes,'verified_slots':count,'target_met':count>=2,'missing':max(0,2-count)}
    return {'domains':rows,'verified_slots':total,'target':2*len(domains),'all_domains_two':total>=2*len(domains),'missing_domains':[d for d,v in rows.items() if not v['target_met']],'note':'R57: Python 후보 수가 아니라 실제 Judge PASS만 coverage에 반영.'}

def select_hybrid_validation_contracts(db_path, contracts, domains, formula_domains=None):
    base=historical_verified_types(db_path,domains,formula_domains); selected=[]; gaps=[]
    for d in domains:
        need=max(0,2-len(set(base.get(d,[]) or []))); ds=[x for x in contracts or [] if x.get('domain')==d and x.get('status')=='R57_AI_VERIFIED' and x.get('contract_type') in R57_ALLOWED_TYPES]; by_type={}
        for x in ds:
            ok,_=validate_r57_contract(db_path,d,x)
            if ok: by_type.setdefault(str(x.get('contract_type')),x)
        chosen=list(by_type.values())[:need]; selected.extend(chosen)
        if len(chosen)<need: gaps.append({'domain':d,'need':need,'found':len(chosen)})
    return {'selected':selected,'gaps':gaps,'ready':not gaps}

def contract_to_question(contract):
    return r57_contract_to_question(contract)

# ================= R58: narrow-bundle resilient synthesis =================
SCHEMA_VERSION='R58-NARROW-BUNDLE-V1'
R58_ALLOWED_TYPES={'exam_relation_application','exam_contrastive_transfer','exam_constraint_diagnosis'}

def _r58_good_anchor(a):
    ans=_clean(a.get('answer')); ev=_clean(a.get('evidence')); topic=_clean(a.get('topic'))
    if not ans or not ev or len(_norm(ev))<20: return False
    if len(ans)>45 or len(_norm(ans))<2: return False
    bad={'목적','단위','메인','사람','관련지식','힘','(예','(참고','(PA','360','400'}
    if ans in bad: return False
    # Avoid obviously broken OCR fragments / heading-only anchors.
    if ans.startswith(('(','·','▶','■','□','-')): return False
    if any(0xE000 <= ord(ch) <= 0xF8FF for ch in ev): return False
    return True

def _r58_rel_score(a,b):
    sa=_r57_words((a.get('topic') or '')+' '+(a.get('evidence') or ''))
    sb=_r57_words((b.get('topic') or '')+' '+(b.get('evidence') or ''))
    shared=len(sa&sb)
    same=1 if a.get('source_name')==b.get('source_name') else 0
    pd=abs(int(a.get('page_no') or 0)-int(b.get('page_no') or 0))
    near=max(0,4-pd) if same else 0
    rich=min(4,(len(_norm(a.get('evidence')))+len(_norm(b.get('evidence'))))//120)
    return shared*3+near*2+same*2+rich

def _r58_anchor_rows(db_path,domain,limit=220):
    con=sqlite3.connect(db_path); con.row_factory=sqlite3.Row
    rows=[]
    for r in con.execute('select id,domain,topic,answer,evidence,source_name,page_no,confidence from anchors where domain=? order by confidence desc,id',(domain,)).fetchall():
        a={k:r[k] for k in r.keys()}; a['id']=int(a['id']); a['page_no']=int(a.get('page_no') or 0)
        if _r58_good_anchor(a): rows.append(a)
        if len(rows)>=limit: break
    con.close(); return rows

def build_r58_bundles(db_path,domain,max_bundles=6):
    rows=_r58_anchor_rows(db_path,domain,220)
    pairs=[]
    for i,a in enumerate(rows):
        for b in rows[i+1:i+45]:
            if a.get('source_name')!=b.get('source_name'): continue
            if abs(a['page_no']-b['page_no'])>4: continue
            if _norm(a.get('answer'))==_norm(b.get('answer')): continue
            sc=_r58_rel_score(a,b)
            if sc<5: continue
            # Find a third supporting anchor near this pair when available.
            third=None; best=-1
            for c in rows:
                if c['id'] in (a['id'],b['id']) or c.get('source_name')!=a.get('source_name'): continue
                if max(abs(c['page_no']-a['page_no']),abs(c['page_no']-b['page_no']))>4: continue
                cs=_r58_rel_score(a,c)+_r58_rel_score(b,c)
                if cs>best: best=cs; third=c
            bundle=[a,b]+([third] if third and best>=8 else [])
            pairs.append((sc+max(0,best//2),bundle))
    pairs.sort(key=lambda x:(-x[0],x[1][0]['page_no'],x[1][0]['id']))
    out=[]; used=set()
    for score,b in pairs:
        ids=tuple(sorted(x['id'] for x in b))
        if ids in used: continue
        # Keep answer pairs diverse and avoid page monopolization.
        used.add(ids)
        out.append({'score':score,'anchors':b})
        if len(out)>=max_bundles: break
    return out

def _r58_mask_answer(text,answers):
    s=_clean(text)
    for ans in sorted([_clean(x) for x in answers if _clean(x)], key=len, reverse=True):
        if len(ans)>=2: s=re.sub(re.escape(ans),'○○',s,flags=re.I)
    return s

def _r58_fallback_contract(domain,bundle,typ):
    a=bundle['anchors']; answers=[_clean(a[0]['answer']),_clean(a[1]['answer'])]
    # Deterministic fallback: grounded excerpts with the answer names masked. It is intentionally
    # conservative; the Judge can veto it, but writer timeout no longer means pool_constructed=0.
    blocks=[]
    for idx,x in enumerate(a[:3]):
        ev=_r58_mask_answer(x.get('evidence',''),answers)
        if len(_norm(ev))<15: continue
        blocks.append({'visible_text':f"자료 {idx+1}: {ev[:360]}", 'anchor_ids':[x['id']]})
    if len(blocks)<2: return None
    tasks=[
      '① (가)와 (나)의 조건 및 관계를 함께 고려하여 두 자료에 해당하는 핵심 개념 또는 방법을 구분하여 쓰시오.',
      '② ①에서 구분한 판단 기준을 (다)가 있으면 그 자료에, 없으면 두 자료의 차이에 적용하여 선택 근거를 한 문장으로 서술하시오.'
    ]
    return {'contract_type':typ,'pattern_id':'P_MIX4','topic':_clean(a[0].get('topic')),
            'cited_anchor_ids':[x['id'] for x in a], 'exact_answers':answers,
            'material_blocks':blocks,'conditions':[],'tasks':tasks,
            'reasoning_chain':['두 자료의 조건을 비교한다','서로 다른 근거를 결합해 ①을 구분한다','①의 판단 기준을 ②에 적용한다'],
            'task2_uses_task1':True,'dependency_note':'②는 ①에서 도출한 두 개념 또는 방법의 구분 기준을 다시 적용해야 하므로 ①의 판단 없이 독립적으로 완성할 수 없다.'}

def validate_r58_contract(db_path,domain,c):
    errs=[]
    typ=str(c.get('contract_type') or '')
    if typ not in R58_ALLOWED_TYPES: errs.append('R58_BAD_TYPE')
    ids=[]
    for x in c.get('cited_anchor_ids') or []:
        try: ids.append(int(x))
        except: pass
    ids=list(dict.fromkeys(ids))
    if len(ids)<2 or len(ids)>4: errs.append('R58_NEED_2_TO_4_ANCHORS')
    answers=[_clean(x) for x in c.get('exact_answers') or [] if _clean(x)]
    if len(answers)!=2 or len({_norm(x) for x in answers})!=2: errs.append('R58_NEED_2_ANSWERS')
    blocks=list(c.get('material_blocks') or [])
    visible='\n'.join(_clean(b.get('visible_text')) for b in blocks)
    tasks=[_clean(x) for x in c.get('tasks') or [] if _clean(x)]
    if len(blocks)<2: errs.append('R58_NEED_2_BLOCKS')
    if len(_norm(visible))<45: errs.append('R58_MATERIAL_TOO_SHORT')
    if len(tasks)!=2: errs.append('R58_NEED_2_TASKS')
    if len(tasks)==2 and not any(k in tasks[1] for k in ('①','판단 기준','앞의','구분한')): errs.append('R58_TASK2_NOT_DEPENDENT')
    nv=_norm(visible+' '+' '.join(tasks))
    for ans in answers:
        if len(_norm(ans))>=2 and _norm(ans) in nv: errs.append('R58_DIRECT_ANSWER_LEAK:'+ans[:30])
    con=sqlite3.connect(db_path); con.row_factory=sqlite3.Row; anchors=[]
    if ids:
        q='select id,domain,topic,answer,evidence,source_name,page_no from anchors where id in (%s)'%(','.join('?'*len(ids)))
        anchors=[dict(r) for r in con.execute(q,tuple(ids)).fetchall()]
    con.close(); amap={int(a['id']):a for a in anchors}
    if len(amap)!=len(ids): errs.append('R58_ANCHOR_NOT_FOUND')
    if any(a.get('domain')!=domain for a in anchors): errs.append('R58_CROSS_DOMAIN')
    whole=_norm(' '.join(_clean(a.get('answer'))+' '+_clean(a.get('evidence')) for a in anchors))
    for ans in answers:
        if _norm(ans) not in whole: errs.append('R58_ANSWER_NOT_GROUNDED')
    if len({a.get('source_name') for a in anchors})>1: errs.append('R58_MULTI_SOURCE')
    # Every visible block must cite an anchor and share actual source vocabulary with it.
    cited=set()
    for i,b in enumerate(blocks):
        aids=[]
        for z in b.get('anchor_ids') or []:
            try: aids.append(int(z))
            except: pass
        if not aids: errs.append(f'R58_BLOCK_{i}_NO_ANCHOR'); continue
        cited.update(aids)
        src=' '.join(_clean(amap[x].get('evidence')) for x in aids if x in amap)
        if len(_r57_words(src)&_r57_words(b.get('visible_text','')))<3: errs.append(f'R58_BLOCK_{i}_WEAK_GROUNDING')
    if len(cited)<2: errs.append('R58_SINGLE_ANCHOR_PUBLIC_MATERIAL')
    chain=[_clean(x) for x in c.get('reasoning_chain') or [] if _clean(x)]
    if len(chain)<3: errs.append('R58_REASONING_LT3')
    if c.get('task2_uses_task1') is not True: errs.append('R58_DEPENDENCY_FALSE')
    return (not errs,{'errors':errs,'anchors':anchors})

def r58_contract_to_question(c):
    if c.get('status') not in ('R58_PYTHON_VALIDATED','R58_AI_VERIFIED'): return None
    blocks=list(c.get('material_blocks') or []); labels=['(가)','(나)','(다)','(라)']
    passage='\n\n'.join(f"{labels[i]} {_clean(b.get('visible_text'))}" for i,b in enumerate(blocks[:4]))
    answers=[_clean(x) for x in c.get('exact_answers') or []][:2]
    anchors=(c.get('validation') or {}).get('anchors') or []
    ctx='\n\n'.join(f"[{a.get('source_name','')} p.{a.get('page_no',0)} / anchor {a.get('id')}]\n정답/개념: {_clean(a.get('answer'))}\n근거: {_clean(a.get('evidence'))}" for a in anchors)
    q={'domain':c.get('domain'),'topic':c.get('topic'),'points':4,'pattern_id':'T4_R58','capability_id':'contract:'+str(c.get('contract_type')),
       'question_type':'자료해석/판단/적용','material_form':'복수 근거 자료','intro':'다음 <자료>를 읽고 <작성 방법>에 따라 순서대로 서술하시오.',
       'passage':passage,'conditions':c.get('conditions') or [],'tasks':c.get('tasks') or [],'answer':answers,'solution':c.get('reasoning_chain') or [],
       'subpoints':[2,2],'evidence':[_clean(a.get('evidence')) for a in anchors],'source_context_override':ctx,'contract_id':c.get('contract_id'),'contract_type':c.get('contract_type')}
    return q

def synthesize_r58_pool(api_key,model,db_path,domain,need,pool_size=None):
    from openai import OpenAI
    pool_size=int(pool_size or (3 if need<=1 else 5)); bundles=build_r58_bundles(db_path,domain,max_bundles=max(pool_size,4))
    style=_r57_style_examples(db_path,domain,limit=1)
    if not bundles: return []
    # Narrow payload: only selected bundles, never the 160-anchor packet that caused R57 timeouts.
    payload=[]
    types=list(R58_ALLOWED_TYPES)
    for i,b in enumerate(bundles[:pool_size]):
        aa=b['anchors']; payload.append({'bundle_id':i,'suggested_type':types[i%len(types)],'anchors':[{'id':x['id'],'topic':x['topic'],'answer':x['answer'],'evidence':_clean(x['evidence'])[:700],'page_no':x['page_no']} for x in aa]})
    prompt=f'''중등 기술 임용 4점 문항을 설계한다. 새 기술 사실은 절대 만들지 않는다. 아래 bundle 안의 사실만 사용한다.\n영역:{domain}\n실제 기출 구조 참고:{json.dumps(style,ensure_ascii=False)[:2600]}\n고정 bundle:{json.dumps(payload,ensure_ascii=False)}\n\n각 bundle마다 정확히 1개 후보를 작성한다. answer는 bundle의 anchor answer 중 서로 다른 2개를 그대로 선택한다. 학생 자료와 task에는 answer 문자열을 절대 쓰지 않는다. 자료는 서로 다른 anchor 근거를 합쳐야 ①을 풀 수 있게 하고, ②는 반드시 ①에서 만든 판단 기준을 적용해야 한다. 단순 정의 맞히기/자료 한 줄 복사/독립 소문항은 금지한다. 기술적 문장은 source evidence를 짧게 바꾸어 쓰되 새로운 사례·수치·인과를 추가하지 않는다.\nJSON만 출력:{{"contracts":[{{"bundle_id":0,"contract_type":"exam_relation_application|exam_contrastive_transfer|exam_constraint_diagnosis","topic":"...","cited_anchor_ids":[1,2,3],"exact_answers":["정답1","정답2"],"material_blocks":[{{"visible_text":"2~4문장","anchor_ids":[1,2]}},{{"visible_text":"2~4문장","anchor_ids":[2,3]}}],"conditions":[],"tasks":["① ...","② ①에서 구분한 판단 기준을 적용하여 ..."],"reasoning_chain":["복수 근거 비교","① 판단","① 기준을 ②에 적용"],"task2_uses_task1":true,"dependency_note":"구체적 이유"}}]}}'''
    rows=[]
    try:
        client=OpenAI(api_key=api_key,timeout=55,max_retries=0)
        r=client.responses.create(model=model,input=prompt,reasoning={'effort':'medium'})
        obj=json.loads(_strip_json(r.output_text)); generated=obj.get('contracts',[]) if isinstance(obj,dict) else []
    except Exception:
        generated=[]
    byid={i:b for i,b in enumerate(bundles[:pool_size])}
    # Validate AI candidates first.
    for x in generated:
        try: bid=int(x.get('bundle_id'))
        except: continue
        if bid not in byid: continue
        x=dict(x); x.pop('bundle_id',None); x['domain']=domain
        ok,detail=validate_r58_contract(db_path,domain,x)
        if not ok: continue
        x['status']='R58_PYTHON_VALIDATED'; x['validation']=detail; x['contract_id']=_fp({'v':'R58','d':domain,'t':x.get('contract_type'),'ids':x.get('cited_anchor_ids'),'a':x.get('exact_answers'),'m':x.get('material_blocks')}); rows.append(x)
    # Deterministic fallback ensures writer timeout/format failure never produces a zero pool.
    existing_keys={(tuple(x.get('cited_anchor_ids') or []),tuple(x.get('exact_answers') or [])) for x in rows}
    for i,b in enumerate(bundles[:pool_size]):
        if len(rows)>=pool_size: break
        fb=_r58_fallback_contract(domain,b,types[i%len(types)])
        if not fb: continue
        key=(tuple(fb.get('cited_anchor_ids') or []),tuple(fb.get('exact_answers') or []))
        if key in existing_keys: continue
        ok,detail=validate_r58_contract(db_path,domain,fb)
        if not ok: continue
        fb['domain']=domain; fb['status']='R58_PYTHON_VALIDATED'; fb['validation']=detail; fb['contract_id']=_fp({'v':'R58F','d':domain,'ids':fb.get('cited_anchor_ids'),'a':fb.get('exact_answers')}); rows.append(fb); existing_keys.add(key)
    return rows[:pool_size]

def combined_coverage_inventory(db_path, contracts, domains, formula_domains=None):
    base=historical_verified_types(db_path,domains,formula_domains); rows={}; total=0
    for d in domains:
        hist=sorted(set(base.get(d,[]) or [])); verified=[]
        for x in contracts or []:
            if x.get('domain')!=d or x.get('status')!='R58_AI_VERIFIED' or x.get('contract_type') not in R58_ALLOWED_TYPES: continue
            ok,_=validate_r58_contract(db_path,d,x)
            if ok: verified.append(x)
        ctypes=sorted(set(str(x.get('contract_type')) for x in verified)); count=min(2,len(set(hist+['contract:'+t for t in ctypes]))); total+=count
        rows[d]={'historical_ai_verified_types':hist,'r58_ai_verified_contract_types':ctypes,'verified_slots':count,'target_met':count>=2,'missing':max(0,2-count)}
    return {'domains':rows,'verified_slots':total,'target':2*len(domains),'all_domains_two':total>=2*len(domains),'missing_domains':[d for d,v in rows.items() if not v['target_met']],'note':'R58: historical Judge PASS + R58 real Judge PASS only.'}


# ========================= R59 actual-exam transfer architecture =========================
R59_ALLOWED_TYPES = ('contrastive_error_transfer','criterion_conflict_resolution')
R59_SCHEMA_VERSION = 'R59-ACTUAL-EXAM-TRANSFER-V1'

def _r59_official_examples(db_path, limit=3):
    con=sqlite3.connect(f'file:{db_path}?immutable=1',uri=True); con.row_factory=sqlite3.Row
    rows=con.execute("""select p.text,s.name,p.page_no from pages p join sources s on s.id=p.source_id
                       where s.kind='official_exam' and p.text like '%[4점]%' order by s.id,p.page_no limit ?""",(int(limit*3),)).fetchall()
    con.close(); out=[]
    for r in rows:
        txt=_clean(r['text'])
        # retain only a bounded style sample, never use it as answer ground truth
        if len(txt)>400: out.append({'source':r['name'],'page_no':r['page_no'],'text':txt[:1800]})
        if len(out)>=limit: break
    return out

def _r59_source_bundles(db_path,domain,max_bundles=8):
    # Start from R58 same-source bundles but require richer, genuinely distinct answers/evidence.
    raw=build_r58_bundles(db_path,domain,max_bundles=max_bundles*3)
    out=[]; seen=set()
    for b in raw:
        aa=b.get('anchors') or []
        if len(aa)<2: continue
        a1,a2=aa[0],aa[1]
        if _norm(a1.get('answer'))==_norm(a2.get('answer')): continue
        w1=_r57_words((a1.get('topic') or '')+' '+(a1.get('evidence') or ''))
        w2=_r57_words((a2.get('topic') or '')+' '+(a2.get('evidence') or ''))
        if len(w1&w2)<1 and abs(int(a1.get('page_no') or 0)-int(a2.get('page_no') or 0))>1: continue
        key=tuple(sorted(int(x['id']) for x in aa[:3]))
        if key in seen: continue
        seen.add(key); out.append(b)
        if len(out)>=max_bundles: break
    return out

def _r59_grounded_fragment(fragment, anchor, min_shared=3):
    f=_clean(fragment); src=_clean(anchor.get('evidence'))
    if len(_norm(f))<18: return False
    if _norm(anchor.get('answer')) in _norm(f): return False
    return len(_r57_words(f)&_r57_words(src))>=min_shared

def validate_r59_contract(db_path,domain,c):
    errs=[]; typ=str(c.get('contract_type') or '')
    if typ not in R59_ALLOWED_TYPES: errs.append('R59_BAD_TYPE')
    ids=[]
    for x in c.get('cited_anchor_ids') or []:
        try: ids.append(int(x))
        except: pass
    ids=list(dict.fromkeys(ids))
    if len(ids)<2 or len(ids)>3: errs.append('R59_NEED_2_TO_3_ANCHORS')
    con=sqlite3.connect(f'file:{db_path}?immutable=1',uri=True); con.row_factory=sqlite3.Row
    anchors=[]
    if ids:
        q='select id,domain,topic,answer,evidence,source_name,page_no from anchors where id in (%s)'%(','.join('?'*len(ids)))
        anchors=[dict(r) for r in con.execute(q,tuple(ids)).fetchall()]
    con.close(); amap={int(a['id']):a for a in anchors}
    if len(amap)!=len(ids): errs.append('R59_ANCHOR_NOT_FOUND')
    if any(a.get('domain')!=domain for a in anchors): errs.append('R59_CROSS_DOMAIN')
    if len({a.get('source_name') for a in anchors})>1: errs.append('R59_MULTI_SOURCE')
    answers=[_clean(x) for x in c.get('exact_answers') or [] if _clean(x)]
    if len(answers)<2 or len({_norm(x) for x in answers})<2: errs.append('R59_NEED_DISTINCT_ANSWERS')
    whole=_norm(' '.join(_clean(a.get('answer'))+' '+_clean(a.get('evidence')) for a in anchors))
    for ans in answers[:2]:
        if _norm(ans) not in whole: errs.append('R59_ANSWER_NOT_GROUNDED')
    clues=list(c.get('clues') or [])
    if len(clues)<4: errs.append('R59_NEED_4_CLUES')
    per={}
    visible=[]
    for i,cl in enumerate(clues):
        try: aid=int(cl.get('anchor_id'))
        except: aid=-1
        txt=_clean(cl.get('text')); visible.append(txt); per[aid]=per.get(aid,0)+1
        if aid not in amap: errs.append(f'R59_CLUE_{i}_BAD_ANCHOR'); continue
        if not _r59_grounded_fragment(txt,amap[aid],2): errs.append(f'R59_CLUE_{i}_WEAK_GROUNDING')
    if len([a for a,n in per.items() if n>=2])<2: errs.append('R59_NEED_TWO_CLUES_PER_SIDE')
    tasks=[_clean(x) for x in c.get('tasks') or [] if _clean(x)]
    if len(tasks)!=2: errs.append('R59_NEED_2_TASKS')
    if len(tasks)==2 and not any(k in tasks[1] for k in ('①','판단 기준','고친','수정한','앞의')): errs.append('R59_TASK2_NOT_DEPENDENT')
    nv=_norm(' '.join(visible)+' '+' '.join(tasks)+' '+_clean(c.get('student_claim'))+' '+_clean(c.get('transfer_case')))
    for ans in answers[:2]:
        if len(_norm(ans))>=2 and _norm(ans) in nv: errs.append('R59_DIRECT_ANSWER_LEAK:'+ans[:24])
    if not _clean(c.get('student_claim')): errs.append('R59_NO_STUDENT_CLAIM')
    if not _clean(c.get('transfer_case')): errs.append('R59_NO_TRANSFER_CASE')
    chain=[_clean(x) for x in c.get('reasoning_chain') or [] if _clean(x)]
    if len(chain)<3: errs.append('R59_REASONING_LT3')
    if c.get('task2_uses_task1') is not True: errs.append('R59_DEPENDENCY_FALSE')
    # The public task must contain an explicit error/choice operation, not pure identification.
    optext=' '.join(tasks)+' '+_clean(c.get('student_claim'))
    if not any(k in optext for k in ('잘못','오류','수정','적절','선택','판단')): errs.append('R59_NO_DECISION_OPERATION')
    return (not errs,{'errors':errs,'anchors':anchors})

def r59_contract_to_question(c):
    if c.get('status') not in ('R59_PYTHON_VALIDATED','R59_AI_VERIFIED'): return None
    clues=list(c.get('clues') or [])
    side1=[_clean(x.get('text')) for x in clues if x.get('side')=='A']
    side2=[_clean(x.get('text')) for x in clues if x.get('side')=='B']
    passage=("[사례 A]\n- "+'\n- '.join(side1)+"\n\n[사례 B]\n- "+'\n- '.join(side2)+
             "\n\n[학생의 판단]\n"+_clean(c.get('student_claim'))+
             "\n\n[추가 상황]\n"+_clean(c.get('transfer_case')))
    anchors=(c.get('validation') or {}).get('anchors') or []
    ctx='\n\n'.join(f"[{a.get('source_name','')} p.{a.get('page_no',0)} / anchor {a.get('id')}]\n정답/개념: {_clean(a.get('answer'))}\n근거: {_clean(a.get('evidence'))}" for a in anchors)
    return {'domain':c.get('domain'),'topic':c.get('topic'),'points':4,'pattern_id':'T4_R59','capability_id':'contract:'+str(c.get('contract_type')),
            'question_type':'오류판단/근거설명/전이적용','material_form':'대비 사례+학생 판단+추가 상황',
            'intro':'다음 <자료>의 학생 판단을 검토하고 <작성 방법>에 따라 순서대로 서술하시오.',
            'passage':passage,'conditions':[], 'tasks':c.get('tasks') or [],'answer':c.get('exact_answers') or [],
            'solution':c.get('reasoning_chain') or [],'subpoints':[2,2],
            'evidence':[_clean(a.get('evidence')) for a in anchors],'source_context_override':ctx,
            'contract_id':c.get('contract_id'),'contract_type':c.get('contract_type')}

def _r59_prompt(domain,bundles,official):
    payload=[]
    for i,b in enumerate(bundles):
        aa=b['anchors'][:3]
        payload.append({'bundle_id':i,'anchors':[{'id':x['id'],'topic':x['topic'],'answer':x['answer'],'evidence':_clean(x['evidence'])[:850],'page_no':x['page_no']} for x in aa]})
    return f'''대한민국 중등 기술 임용 4점 문항의 설계자다. 아래 실제 기출은 오직 구조만 참고하고 사실/정답은 복사하지 않는다.
영역: {domain}
실제 기출 구조 예시: {json.dumps(official,ensure_ascii=False)[:5200]}
서브노트 근거 bundle: {json.dumps(payload,ensure_ascii=False)}

목표는 정의 맞히기가 아니다. 각 bundle마다 "잘못 적용된 학생 판단을 수정 → 수정 근거를 두 단서 이상으로 설명 → 그 판단 기준을 추가 상황에 전이"하는 후보 1개를 작성한다.
중요:
- 기술 사실은 anchor evidence에 있는 것만 사용한다. 새로운 사례의 기술적 속성은 만들지 않는다.
- 추가 상황은 anchor에 이미 있는 단서들을 재조합할 뿐 새 인과/수치/효과를 추가하지 않는다.
- 실제 정답명은 material/student_claim/transfer_case/tasks에 절대 노출하지 않는다.
- 각 정답 측면에 서로 다른 단서가 최소 2개 있어야 한다.
- 학생 판단은 두 사례 중 하나를 의도적으로 잘못 연결한다. 이것은 '학생의 오류'이지 기술 사실 진술이 아니다.
- ①은 단순 명칭 2개 쓰기가 아니라 오류를 찾아 올바르게 수정하고 근거를 설명해야 한다.
- ②는 ①에서 고친 판단 기준 없이는 답할 수 없게 만든다.
- bundle 안 두 anchor가 실제로 대비/선택 관계를 만들 수 없으면 그 bundle은 OMIT한다.

JSON만 출력:
{{"contracts":[{{"bundle_id":0,"contract_type":"contrastive_error_transfer|criterion_conflict_resolution","topic":"...","cited_anchor_ids":[1,2],"exact_answers":["anchor answer 그대로 2개"],"clues":[{{"side":"A","anchor_id":1,"text":"근거 단서1"}},{{"side":"A","anchor_id":1,"text":"근거 단서2"}},{{"side":"B","anchor_id":2,"text":"근거 단서1"}},{{"side":"B","anchor_id":2,"text":"근거 단서2"}}],"student_claim":"사례 A와 B를 잘못 연결한 학생 판단. 실제 정답명 금지","transfer_case":"위 단서 중 2개 이상을 조합한 추가 상황. 실제 정답명 금지","tasks":["① 학생 판단의 오류를 찾아 올바르게 수정하고, 두 자료의 근거를 이용하여 판단 기준을 서술하시오.","② ①에서 고친 판단 기준을 추가 상황에 적용하여 해당하는 개념 또는 방법을 쓰고 근거를 서술하시오."],"reasoning_chain":["A/B의 복수 단서 비교","학생 판단 오류 수정 및 기준 도출","도출 기준을 추가 상황에 전이"],"task2_uses_task1":true}}]}}
'''

def synthesize_r59_pool(api_key,model,db_path,domain,need,pool_size=None):
    from openai import OpenAI
    size=int(pool_size or (4 if need<=1 else 6)); bundles=_r59_source_bundles(db_path,domain,max_bundles=size)
    if not bundles: return []
    official=_r59_official_examples(db_path,2)
    client=OpenAI(api_key=api_key,timeout=75,max_retries=1)
    contracts=[]
    # transport-resilient split: two small fixed calls at most; no Judge-driven regeneration.
    chunks=[bundles[:3],bundles[3:6]] if len(bundles)>3 else [bundles]
    for chunk in chunks:
        if not chunk: continue
        try:
            rr=client.responses.create(model=model,input=_r59_prompt(domain,chunk,official),reasoning={'effort':'high'})
            raw=json.loads(re.sub(r'^```(?:json)?\s*|\s*```$','',rr.output_text.strip()))
            arr=raw.get('contracts') or []
        except Exception:
            arr=[]
        for c in arr:
            try: bi=int(c.get('bundle_id',-1))
            except: bi=-1
            if bi<0 or bi>=len(chunk): continue
            aa=chunk[bi]['anchors'][:3]; allowed={int(x['id']):x for x in aa}
            ids=[]
            for z in c.get('cited_anchor_ids') or []:
                try: z=int(z)
                except: continue
                if z in allowed: ids.append(z)
            ids=list(dict.fromkeys(ids))
            if len(ids)<2: continue
            c=copy.deepcopy(c); c['domain']=domain; c['cited_anchor_ids']=ids
            c['contract_type']=str(c.get('contract_type') or 'contrastive_error_transfer')
            if c['contract_type'] not in R59_ALLOWED_TYPES: c['contract_type']='contrastive_error_transfer'
            # Force answers from cited anchors; AI may not invent/change them.
            c['exact_answers']=[_clean(allowed[z]['answer']) for z in ids[:2]]
            c['status']='R59_RAW'; c['schema_version']=R59_SCHEMA_VERSION
            ok,diag=validate_r59_contract(db_path,domain,c); c['validation']=diag
            if ok:
                c['status']='R59_PYTHON_VALIDATED'; c['contract_id']=f"R59-{domain}-{c['contract_type']}-"+'-'.join(map(str,ids[:2]))
                contracts.append(c)
    # distinct contract ids/types first
    out=[]; seen=set()
    for c in contracts:
        k=(c.get('contract_type'),tuple(c.get('cited_anchor_ids') or []))
        if k in seen: continue
        seen.add(k); out.append(c)
    return out[:size]

# Final R59 coverage override: only historical Judge PASS + R59 real Judge PASS.
def combined_coverage_inventory(db_path, contracts, domains, formula_domains=None):
    base=historical_verified_types(db_path,domains,formula_domains); rows={}; total=0
    for d in domains:
        hist=sorted(set(base.get(d,[]) or [])); verified=[]
        for x in contracts or []:
            if x.get('domain')!=d or x.get('status')!='R59_AI_VERIFIED' or x.get('contract_type') not in R59_ALLOWED_TYPES: continue
            ok,_=validate_r59_contract(db_path,d,x)
            if ok: verified.append(x)
        ctypes=sorted(set(str(x.get('contract_type')) for x in verified))
        count=min(2,len(set(hist+['contract:'+t for t in ctypes]))); total+=count
        rows[d]={'historical_ai_verified_types':hist,'r59_ai_verified_contract_types':ctypes,'verified_slots':count,'target_met':count>=2,'missing':max(0,2-count)}
    return {'domains':rows,'verified_slots':total,'target':2*len(domains),'all_domains_two':total>=2*len(domains),
            'missing_domains':[d for d,v in rows.items() if not v['target_met']],
            'note':'R59: historical Judge PASS + actual-exam-transfer R59 real Judge PASS only.'}

# R59 relation-first selector override. The selector chooses the reasoning relation before the writer sees a bundle.
def _r59_select_bundles(api_key,model,db_path,domain,wanted=6):
    from openai import OpenAI
    anchors=_anchor_rows(db_path,domain,limit=32)
    if len(anchors)<4: return []
    items=[{'id':a['id'],'answer':a['answer'],'evidence':_clean(a['evidence'])[:700],'source_name':a['source_name'],'page_no':a['page_no']} for a in anchors]
    prompt=f'''대한민국 중등 기술 임용 4점 출제용 관계 선별기다. 문항을 쓰지 말고 관계만 고른다.
영역:{domain}
후보 anchor:{json.dumps(items,ensure_ascii=False)}

최대 {wanted}개 관계를 선택하라. 각 관계는 반드시 같은 source 안에서 2~3개 anchor로 구성한다.
PASS 가능한 관계는 다음뿐이다: 서로 다른 개념/방법을 조건에 따라 선택하는 대비, 한 규칙을 잘못 적용한 오류를 다른 조건에 전이해 수정할 수 있는 관계, 동일 체계 안의 단계/원인-결과가 뒤 판단에 실제로 쓰이는 관계.
REJECT: 같은 페이지일 뿐인 항목, 단순 정의 2개, 상하위 목록, 독립 개념 병렬, 명칭 두 개 암기, 단어만 비슷한 항목.
특히 두 answer가 각각 자료 한 줄만 읽고 바로 맞혀지는 관계는 고르지 마라.
JSON만 출력:{{"relations":[{{"anchor_ids":[1,2],"relation_type":"contrast|conditional_choice|error_transfer|dependent_sequence","master_relation":"구체적 관계 한 문장","why_inferential":"왜 최소 2개 단서를 결합해야 하는지"}}]}}'''
    try:
        client=OpenAI(api_key=api_key,timeout=75,max_retries=1)
        rr=client.responses.create(model=model,input=prompt,reasoning={'effort':'high'})
        obj=json.loads(re.sub(r'^```(?:json)?\s*|\s*```$','',rr.output_text.strip()))
        rels=obj.get('relations') or []
    except Exception:
        return []
    amap={int(a['id']):a for a in anchors}; out=[]; seen=set()
    for r in rels:
        ids=[]
        for z in r.get('anchor_ids') or []:
            try: z=int(z)
            except: continue
            if z in amap: ids.append(z)
        ids=list(dict.fromkeys(ids))
        if not 2<=len(ids)<=3: continue
        aa=[amap[z] for z in ids]
        if len({x['source_name'] for x in aa})!=1: continue
        if len({_norm(x['answer']) for x in aa[:2]})<2: continue
        key=tuple(sorted(ids))
        if key in seen: continue
        seen.add(key); out.append({'score':99,'anchors':aa,'selector_relation':r})
        if len(out)>=wanted: break
    return out

# Override pool synthesis: relation selector -> actual-exam guided writer -> Python hard gate. No low-quality deterministic fallback.
def synthesize_r59_pool(api_key,model,db_path,domain,need,pool_size=None):
    from openai import OpenAI
    size=int(pool_size or (4 if need<=1 else 6))
    bundles=_r59_select_bundles(api_key,model,db_path,domain,wanted=size)
    if not bundles: return []
    official=_r59_official_examples(db_path,2)
    client=OpenAI(api_key=api_key,timeout=75,max_retries=1); contracts=[]
    # Small fixed writer calls. Transport retry is bounded and never Judge-driven.
    chunks=[bundles[i:i+2] for i in range(0,len(bundles),2)]
    for chunk in chunks:
        payload=[]
        for i,b in enumerate(chunk):
            payload.append({'bundle_id':i,'selector_relation':b.get('selector_relation',{}),
                            'anchors':[{'id':x['id'],'topic':x.get('topic',''),'answer':x['answer'],'evidence':_clean(x['evidence'])[:850],'page_no':x['page_no']} for x in b['anchors']]})
        prompt=_r59_prompt(domain,chunk,official).replace('서브노트 근거 bundle: '+json.dumps([{'bundle_id':i,'anchors':[{'id':x['id'],'topic':x['topic'],'answer':x['answer'],'evidence':_clean(x['evidence'])[:850],'page_no':x['page_no']} for x in b['anchors'][:3]]} for i,b in enumerate(chunk)],ensure_ascii=False),
                                                   '서브노트 근거 bundle 및 선별관계: '+json.dumps(payload,ensure_ascii=False))
        try:
            rr=client.responses.create(model=model,input=prompt,reasoning={'effort':'high'})
            raw=json.loads(re.sub(r'^```(?:json)?\s*|\s*```$','',rr.output_text.strip())); arr=raw.get('contracts') or []
        except Exception:
            arr=[]
        for c in arr:
            try: bi=int(c.get('bundle_id',-1))
            except: bi=-1
            if bi<0 or bi>=len(chunk): continue
            aa=chunk[bi]['anchors']; allowed={int(x['id']):x for x in aa}
            ids=[]
            for z in c.get('cited_anchor_ids') or []:
                try: z=int(z)
                except: continue
                if z in allowed: ids.append(z)
            ids=list(dict.fromkeys(ids))
            if len(ids)<2: continue
            c=copy.deepcopy(c); c['domain']=domain; c['cited_anchor_ids']=ids[:3]
            c['contract_type']=str(c.get('contract_type') or 'contrastive_error_transfer')
            if c['contract_type'] not in R59_ALLOWED_TYPES: c['contract_type']='contrastive_error_transfer'
            c['exact_answers']=[_clean(allowed[z]['answer']) for z in ids[:2]]
            c['selector_relation']=copy.deepcopy(chunk[bi].get('selector_relation',{}))
            c['status']='R59_RAW'; c['schema_version']=R59_SCHEMA_VERSION
            ok,diag=validate_r59_contract(db_path,domain,c); c['validation']=diag
            if ok:
                c['status']='R59_PYTHON_VALIDATED'; c['contract_id']=f"R59-{domain}-{c['contract_type']}-"+'-'.join(map(str,ids[:2])); contracts.append(c)
    out=[]; seen=set()
    for c in contracts:
        k=(c.get('contract_type'),tuple(c.get('cited_anchor_ids') or []))
        if k in seen: continue
        seen.add(k); out.append(c)
    return out[:size]
