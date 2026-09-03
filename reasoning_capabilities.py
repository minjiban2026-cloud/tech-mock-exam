import re, sqlite3, random, hashlib, json

CAP_ORDERED='ordered_sequence_repair'
CAP_CONDITION='condition_outcome_swap'
_NUM_PAT=re.compile(r'(①|②|③|④|⑤|⑥|⑦|⑧|⓵|⓶|⓷|⓸|⓹|⓺|⓻|⓼|(?<!\d)([1-8])\))')
_COND_WORDS=('하면','경우','때에는','때는','때 ','조건에서','조건을','따라')

def _clean(s):
    s=str(s or '').replace('\x01',' ').replace('\u200b',' ')
    return re.sub(r'\s+',' ',s).strip(' -·▶□■\t\n')

def _fp(obj):
    return hashlib.sha256(json.dumps(obj,ensure_ascii=False,sort_keys=True).encode()).hexdigest()[:20]

def _domain_page_refs(con, domain):
    rows=con.execute('select distinct source_name,page_no from anchors where domain=?',(domain,)).fetchall()
    return [(str(r[0]),int(r[1] or 0)) for r in rows if r[0]]

def _page_text(con, source_name, page_no):
    r=con.execute('select text from pages where source_name=? and page_no=? limit 1',(source_name,page_no)).fetchone()
    return str(r[0] or '') if r else ''

def _ordered_items(text):
    text=_clean(text); ms=list(_NUM_PAT.finditer(text));
    if len(ms)<4: return []
    process_words=('절차','단계','과정','순서','공정')
    for start in range(0,len(ms)-3):
        pre=text[max(0,ms[start].start()-140):ms[start].start()]
        if not any(k in pre for k in process_words):
            continue
        out=[]
        for i in range(start,min(len(ms),start+6)):
            a=ms[i].end(); b=ms[i+1].start() if i+1<len(ms) else min(len(text),a+260)
            frag=_clean(text[a:b])
            if 12<=len(frag)<=220 and len(re.findall(r'[가-힣A-Za-z]',frag))>=8:
                out.append(frag)
            else:
                break
            if len(out)>=4: return out[:4]
    return []

def _sentences(text):
    parts=re.split(r'(?<=[.!?])\s+|\s[-–—]\s|\s[▶■□]\s',_clean(text))
    return [_clean(x) for x in parts if 18<=len(_clean(x))<=220]

def _condition_pair(sentence):
    s=_clean(sentence)
    for marker in _COND_WORDS:
        idx=s.find(marker)
        if idx<5: continue
        left=_clean(s[:idx+len(marker)]); right=_clean(s[idx+len(marker):])
        if 8<=len(left)<=120 and 8<=len(right)<=140 and len(re.findall(r'[가-힣A-Za-z]',right))>=5:
            return left,right
    return None

def discover_capabilities(db_path, domain, max_each=20):
    con=sqlite3.connect(db_path); ordered=[]; cond=[]
    for src,pno in _domain_page_refs(con,domain):
        text=_page_text(con,src,pno)
        if not text: continue
        items=_ordered_items(text)
        if items and len(ordered)<max_each:
            ordered.append({'capability_id':CAP_ORDERED,'domain':domain,'source_name':src,'page_no':pno,'items':items[:4]})
        cps=[]
        for sent in _sentences(text):
            cp=_condition_pair(sent)
            if cp and cp not in cps: cps.append(cp)
        if len(cps)>=2 and len(cond)<max_each and _clean(cps[0][1])!=_clean(cps[1][1]):
            cond.append({'capability_id':CAP_CONDITION,'domain':domain,'source_name':src,'page_no':pno,'pairs':cps[:2]})
    con.close(); return {CAP_ORDERED:ordered,CAP_CONDITION:cond}

def capability_inventory(db_path, domains):
    rows={}; total=0
    for d in domains:
        caps=discover_capabilities(db_path,d); present=[k for k,v in caps.items() if v]
        rows[d]={'capability_types':present,'type_count':len(present),'candidate_counts':{k:len(v) for k,v in caps.items()},'target_met':len(present)>=2}
        total+=len(present)
    return {'domains':rows,'covered_domain_capabilities':total,'target':len(domains)*2,'all_domains_two_types':all(v['target_met'] for v in rows.values())}

def _source_meta(spec): return [{'source_name':spec['source_name'],'page_no':spec['page_no']}]

def generate_ordered_question(spec,rng=None):
    rng=rng or random.Random(); items=list(spec['items'][:4]); j=rng.choice([0,1,2])
    wrong=list(range(4)); wrong[j],wrong[j+1]=wrong[j+1],wrong[j]
    labels=['ㄱ','ㄴ','ㄷ','ㄹ']; displayed=[items[i] for i in wrong]; pos={orig:k for k,orig in enumerate(wrong)}
    bad_pair=f"{labels[pos[j]]}↔{labels[pos[j+1]]}"; correct_order=' → '.join(labels[pos[i]] for i in range(4))
    q={'domain':spec['domain'],'topic':'원자료 절차 순서 적용','points':4,'verifier':'python_source_capability','pattern_id':'T4_CAP22','capability_id':CAP_ORDERED,'question_type':'과정/순서','material_form':'절차자료','intro':'다음 절차 자료를 분석하시오.','passage':'다음은 하나의 절차를 정리한 학생의 기록이다.\n'+'\n'.join(f'{labels[k]}. {displayed[k]}' for k in range(4)),'conditions':['원자료의 절차적 선후관계를 기준으로 판단한다.','학생 기록에서는 서로 이웃한 두 단계의 위치만 바뀌었다.'],'tasks':['순서가 서로 바뀐 두 항목의 기호를 쓰시오.','첫 판단을 반영하여 ㄱ~ㄹ의 전체 순서를 올바르게 배열하시오.'],'answer':[bad_pair,correct_order],'solution':[f'원자료에서 해당 두 단계는 {j+1}번째→{j+2}번째 순서이다.',f'원자료 순서에 대응하면 {correct_order}이다.'],'subpoints':[2,2],'sources':_source_meta(spec),'source_context_override':'\n'.join(items),'source_basis':'동일 원자료에 명시된 4단계 이상의 절차 순서','derived_answer_flags':[True,True]}
    q['fingerprint']=_fp({k:q[k] for k in ['domain','capability_id','passage','answer']}); return q

def generate_condition_question(spec,rng=None):
    rng=rng or random.Random(); (c1,o1),(c2,o2)=spec['pairs'][:2]
    q={'domain':spec['domain'],'topic':'조건-결과 관계 진단','points':4,'verifier':'python_source_capability','pattern_id':'T4_CAP22','capability_id':CAP_CONDITION,'question_type':'오류진단/적용','material_form':'조건자료','intro':'다음 조건과 결과의 연결을 검토하시오.','passage':f'사례 A | 조건: {c1} / 학생의 결과 판단: {o2}\n사례 B | 조건: {c2} / 학생의 결과 판단: {o1}','conditions':['두 사례의 결과 판단은 서로 바뀌어 기록되었다.','원자료에 명시된 조건-결과 관계만을 사용한다.'],'tasks':['사례 A의 결과 판단을 원자료와 일치하도록 바르게 수정하시오.','사례 B의 결과 판단을 원자료와 일치하도록 바르게 수정하시오.'],'answer':[o1,o2],'solution':[f'{c1}에 대응하는 결과는 {o1}',f'{c2}에 대응하는 결과는 {o2}'],'subpoints':[2,2],'sources':_source_meta(spec),'source_context_override':f'{c1} {o1}\n{c2} {o2}','source_basis':'동일 원자료에 명시된 두 조건-결과 관계','derived_answer_flags':[True,True]}
    q['fingerprint']=_fp({k:q[k] for k in ['domain','capability_id','passage','answer']}); return q

def generate_reasoning_question(db_path,domain,capability_id,rng=None,avoid_fingerprints=None):
    rng=rng or random.Random(); avoid=set(avoid_fingerprints or []); specs=list(discover_capabilities(db_path,domain).get(capability_id,[])); rng.shuffle(specs)
    for spec in specs:
        q=generate_ordered_question(spec,rng) if capability_id==CAP_ORDERED else generate_condition_question(spec,rng)
        if q['fingerprint'] not in avoid: return q
    return None

CAP_CONTRAST='paired_concept_discrimination'
_GENERIC_ANS={'목적','장점','단점','특징','방법','정의','원인','효과','종류','구분','개념','과정','절차'}

def _lex_tokens(s):
    s=re.sub(r'[^가-힣A-Za-z0-9]+',' ',_clean(s))
    ws=[w for w in s.split() if len(w)>=2]
    out=set(ws)
    for w in ws:
        if len(w)>=4:
            out.update(w[i:i+2] for i in range(len(w)-1))
    return out

def _contrast_specs(db_path,domain,max_each=20):
    con=sqlite3.connect(db_path); rows=con.execute('select answer,evidence,source_name,page_no from anchors where domain=? order by source_name,page_no,id',(domain,)).fetchall(); con.close()
    candidates=[]
    for i in range(len(rows)):
        a1,e1,s1,p1=_clean(rows[i][0]),_clean(rows[i][1]),str(rows[i][2]),int(rows[i][3] or 0)
        if not (2<=len(a1)<=35) or a1 in _GENERIC_ANS: continue
        c1=_clean(e1.replace(a1,' ')); t1=_lex_tokens(c1)
        if not (20<=len(c1)<=240 and t1): continue
        for j in range(i+1,len(rows)):
            a2,e2,s2,p2=_clean(rows[j][0]),_clean(rows[j][1]),str(rows[j][2]),int(rows[j][3] or 0)
            if s1!=s2 or abs(p1-p2)>1 or not (2<=len(a2)<=35) or a2 in _GENERIC_ANS or a1==a2: continue
            c2=_clean(e2.replace(a2,' ')); t2=_lex_tokens(c2)
            if not (20<=len(c2)<=240 and t2) or c1==c2: continue
            inter=len(t1&t2); union=len(t1|t2) or 1; jac=inter/union
            if inter<3 or jac<0.06: continue
            candidates.append((jac,inter,{'capability_id':CAP_CONTRAST,'domain':domain,'source_name':s1,'page_no':p1,'pairs':[(a1,c1),(a2,c2)]}))
    candidates.sort(key=lambda x:(-x[0],-x[1]))
    out=[]; seen=set()
    for _,__,spec in candidates:
        key=tuple(x[0] for x in spec['pairs'])
        if key in seen: continue
        seen.add(key); out.append(spec)
        if len(out)>=max_each: break
    return out

_old_discover=discover_capabilities
def discover_capabilities(db_path, domain, max_each=20):
    base=_old_discover(db_path,domain,max_each=max_each)
    base[CAP_CONTRAST]=_contrast_specs(db_path,domain,max_each=max_each)
    return base

def generate_contrast_question(spec,rng=None):
    (a1,c1),(a2,c2)=spec['pairs'][:2]
    q={'domain':spec['domain'],'topic':'대조 사례 판별','points':4,'verifier':'python_source_capability','pattern_id':'T4_CAP22','capability_id':CAP_CONTRAST,'question_type':'비교/구분','material_form':'대조자료','intro':'다음 두 설명을 비교하여 판단하시오.','passage':f'사례 A | {c1}\n사례 B | {c2}','conditions':['두 사례는 같은 원자료에서 서로 다른 개념을 설명한 것이다.','사례에 개념명은 직접 제시하지 않았다.'],'tasks':['사례 A에 해당하는 개념명을 쓰고, 판단 근거가 되는 핵심 특징을 함께 쓰시오.','사례 B에 해당하는 개념명을 쓰고, 사례 A와 구분되는 핵심 특징을 함께 쓰시오.'],'answer':[a1,a2],'solution':[f'사례 A는 {a1}: {c1}',f'사례 B는 {a2}: {c2}'],'subpoints':[2,2],'sources':_source_meta(spec),'source_context_override':f'{a1} {c1}\n{a2} {c2}','source_basis':'동일 원자료·동일 페이지의 서로 다른 두 개념 설명','derived_answer_flags':[False,False]}
    q['fingerprint']=_fp({k:q[k] for k in ['domain','capability_id','passage','answer']}); return q

_old_generate=generate_reasoning_question
def generate_reasoning_question(db_path,domain,capability_id,rng=None,avoid_fingerprints=None):
    if capability_id!=CAP_CONTRAST:
        return _old_generate(db_path,domain,capability_id,rng,avoid_fingerprints)
    rng=rng or random.Random(); avoid=set(avoid_fingerprints or []); specs=list(discover_capabilities(db_path,domain).get(CAP_CONTRAST,[])); rng.shuffle(specs)
    for spec in specs:
        q=generate_contrast_question(spec,rng)
        if q['fingerprint'] not in avoid: return q
    return None

def coverage_inventory(db_path,domains,formula_domains=None):
    formula_domains=set(formula_domains or [])
    priority=['deterministic_formula_operation',CAP_ORDERED,CAP_CONTRAST,CAP_CONDITION]
    rows={}; targets=[]
    for d in domains:
        caps=discover_capabilities(db_path,d)
        avail=[]
        if d in formula_domains: avail.append('deterministic_formula_operation')
        avail += [k for k in (CAP_ORDERED,CAP_CONTRAST,CAP_CONDITION) if caps.get(k)]
        chosen=avail[:2]
        rows[d]={'available_types':avail,'selected_target_types':chosen,'target_count':len(chosen),'target_met':len(chosen)>=2,'candidate_counts':{k:len(v) for k,v in caps.items()}}
        for typ in chosen: targets.append({'domain':d,'capability_id':typ,'coverage_key':d+'::'+typ})
    return {'domains':rows,'targets':targets,'target_total':len(domains)*2,'constructed_target_total':len(targets),'all_domains_two_targets':all(v['target_met'] for v in rows.values())}
