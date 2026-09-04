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
