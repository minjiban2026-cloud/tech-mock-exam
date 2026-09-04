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
        # R47: the full 18-suite proved paired-concept (0/7) and condition-swap (0/1)
        # are not valid 4-point capabilities.  They remain discoverable for diagnostics,
        # but can never satisfy a coverage target.  Do not fabricate 9x2 readiness.
        if caps.get(CAP_ORDERED):
            avail.append(CAP_ORDERED)
        chosen=avail[:2]
        rows[d]={'available_types':avail,'selected_target_types':chosen,'target_count':len(chosen),'target_met':len(chosen)>=2,'candidate_counts':{k:len(v) for k,v in caps.items()}}
        for typ in chosen: targets.append({'domain':d,'capability_id':typ,'coverage_key':d+'::'+typ})
    return {'domains':rows,'targets':targets,'target_total':len(domains)*2,'constructed_target_total':len(targets),'all_domains_two_targets':all(v['target_met'] for v in rows.values())}

# R48: semantic-operation discovery. These are DISCOVERED only; they do not count
# toward final coverage until a deterministic constructor + full Judge suite certifies them.
CAP_DIRECTIONAL='directional_rule_application'
CAP_CAUSAL='cause_intervention_prediction'
_DIR_WORDS=('증가','감소','향상','저하','커지','작아지','높아지','낮아지','비례','반비례')
_CAUSE_WORDS=('때문','원인','방지','예방','대책','위해','필요')

def _semantic_operation_specs(db_path,domain,max_each=20):
    con=sqlite3.connect(db_path)
    rows=con.execute('select answer,evidence,source_name,page_no from anchors where domain=? order by source_name,page_no,id',(domain,)).fetchall()
    con.close()
    directional=[]; causal=[]; seen_d=set(); seen_c=set()
    for ans,ev,src,pno in rows:
        a=_clean(ans); e=_clean(ev)
        if not (18<=len(e)<=320):
            continue
        # reject obvious page/list debris and incomplete fragments
        if e.count('□')+e.count('■')>3 or e.endswith(('및','또는','으로부터','따라','경우')):
            continue
        dmarks=[w for w in _DIR_WORDS if w in e]
        if dmarks and len(re.findall(r'[가-힣A-Za-z]',e))>=16:
            key=(str(src),int(pno or 0),e)
            if key not in seen_d:
                seen_d.add(key); directional.append({'capability_id':CAP_DIRECTIONAL,'domain':domain,'answer':a,'evidence':e,'source_name':str(src),'page_no':int(pno or 0),'markers':dmarks})
        cmarks=[w for w in _CAUSE_WORDS if w in e]
        # causal candidate must contain an explicit action/effect cue as well, not just a heading.
        if cmarks and any(x in e for x in ('방지','예방','대책','필요','향상','저하','증가','감소','사용','이용')) and len(re.findall(r'[가-힣A-Za-z]',e))>=18:
            key=(str(src),int(pno or 0),e)
            if key not in seen_c:
                seen_c.add(key); causal.append({'capability_id':CAP_CAUSAL,'domain':domain,'answer':a,'evidence':e,'source_name':str(src),'page_no':int(pno or 0),'markers':cmarks})
    return {CAP_DIRECTIONAL:directional[:max_each],CAP_CAUSAL:causal[:max_each]}

def semantic_operation_inventory(db_path,domains):
    out={}
    for d in domains:
        specs=_semantic_operation_specs(db_path,d,max_each=50)
        out[d]={
            'directional_rule_application':len(specs[CAP_DIRECTIONAL]),
            'cause_intervention_prediction':len(specs[CAP_CAUSAL]),
            'status':'DISCOVERED_ONLY',
            'examples':{
                CAP_DIRECTIONAL:[x['evidence'][:180] for x in specs[CAP_DIRECTIONAL][:2]],
                CAP_CAUSAL:[x['evidence'][:180] for x in specs[CAP_CAUSAL][:2]],
            }
        }
    return out

# R48 authoritative coverage: only capabilities with actual full-suite PASS evidence
# are CERTIFIED. R46 proved deterministic_formula_operation 3/3. Ordered remains
# EXPERIMENTAL and cannot inflate readiness merely because a source sequence exists.
def coverage_inventory(db_path,domains,formula_domains=None):
    formula_domains=set(formula_domains or [])
    rows={}; targets=[]
    semantic=semantic_operation_inventory(db_path,domains)
    for d in domains:
        caps=discover_capabilities(db_path,d)
        certified=[]; experimental=[]
        if d in formula_domains:
            certified.append('deterministic_formula_operation')
        if caps.get(CAP_ORDERED):
            experimental.append(CAP_ORDERED)
        if semantic[d][CAP_DIRECTIONAL]: experimental.append(CAP_DIRECTIONAL)
        if semantic[d][CAP_CAUSAL]: experimental.append(CAP_CAUSAL)
        rows[d]={
            'certified_types':certified,
            'experimental_types':experimental,
            'certified_count':len(certified),
            'target_met':len(certified)>=2,
            'semantic_discovery':semantic[d],
            'retired_candidate_counts':{
                CAP_CONDITION:len(caps.get(CAP_CONDITION,[])),
                CAP_CONTRAST:len(caps.get(CAP_CONTRAST,[])),
            }
        }
        for typ in certified:
            targets.append({'domain':d,'capability_id':typ,'coverage_key':d+'::'+typ,'status':'AI_VERIFIED'})
    return {
        'domains':rows,'targets':targets,'target_total':len(domains)*2,
        'certified_target_total':len(targets),'constructed_target_total':len(targets),
        'all_domains_two_targets':all(v['target_met'] for v in rows.values()),
        'status_legend':{'AI_VERIFIED':'실제 전수 Judge PASS','EXPERIMENTAL':'후보/구조 발견, coverage 미산입','RETIRED':'전수검증 0-pass로 폐기'},
        'note':'R48: 최종 coverage에는 실제 전수 Judge로 인증된 capability만 산입. ordered 및 새 semantic operation은 인증 전까지 EXPERIMENTAL.'
    }


# R49: executable semantic-operation candidates.  These are Python-grounded
# constructors for full-suite validation; they remain EXPERIMENTAL until Judge PASS.

def _direction_from_evidence(e):
    e=_clean(e)
    if '반비례' in e: return '반대 방향으로 변한다'
    if '비례' in e: return '같은 방향으로 변한다'
    # retain the strongest explicit directional predicate without inventing a variable name
    for w,ans in [('증가','증가한다'),('향상','향상된다'),('높아지','높아진다'),('커지','커진다'),('감소','감소한다'),('저하','저하된다'),('낮아지','낮아진다'),('작아지','작아진다')]:
        if w in e: return ans
    return None

def generate_directional_question(spec,rng=None):
    e=_clean(spec.get('evidence'))
    direction=_direction_from_evidence(e)
    if not direction: return None
    # The source rule is provided as evidence, but the tasks force two-step transfer:
    # identify the operative relation, then apply it to a changed condition.
    passage=(
        '다음은 한 기술적 관계에 관한 원자료의 설명이다.\n'
        f'자료 | {e}\n\n'
        '학생 A는 이 관계를 이용할 때 원자료의 변화 방향을 그대로 적용해야 한다고 보았고, '
        '학생 B는 조건이 달라지면 변화 방향도 반대로 보아야 한다고 주장하였다.'
    )
    q={'domain':spec['domain'],'topic':'관계 규칙의 조건 적용','points':4,'verifier':'python_semantic_operation',
       'pattern_id':'T4_SEM22','capability_id':CAP_DIRECTIONAL,'question_type':'관계판단/적용','material_form':'관계자료',
       'intro':'다음 자료를 바탕으로 기술적 관계를 판단하시오.','passage':passage,
       'conditions':['원자료에 명시된 변화 방향만을 근거로 판단한다.','새로운 사실이나 수치를 가정하지 않는다.'],
       'tasks':['원자료에서 판단에 사용되는 변화 관계를 한 문장으로 정리하시오.',
                '그 관계가 유지되는 조건에서 원인 쪽 변수가 더 커지는 경우 결과 쪽 변수의 변화 방향을 쓰고, 학생 A와 B 중 타당한 판단을 한 학생을 함께 쓰시오.'],
       'answer':[direction,f'{direction} / 학생 A'],
       'solution':[f'원자료의 명시적 변화 관계는 {direction}.',f'같은 관계가 유지되므로 결과 변화는 {direction}이며 학생 A가 타당하다.'],
       'subpoints':[2,2],'sources':_source_meta(spec),'source_context_override':e,
       'source_basis':'원자료에 명시된 방향성 관계를 동일 조건의 새로운 상황에 적용','derived_answer_flags':[True,True]}
    q['fingerprint']=_fp({k:q[k] for k in ['domain','capability_id','passage','answer']}); return q

def _causal_action(e):
    e=_clean(e)
    # Only explicit intervention clauses are allowed.  Generic explanatory text,
    # headings such as "목적", and definition-like sentences are not actions.
    strong=('대책','예방','방지','설치','첨가','제염','재배','차단','보강','유지')
    if not any(k in e for k in strong):
        return None
    # Prefer text following explicit heading/cue.
    for cue in ('대책 :','대책:','예방 :','예방:','방지 :','방지:'):
        if cue in e:
            frag=_clean(e.split(cue,1)[1])
            if 8<=len(frag)<=180:
                return frag
    # Otherwise take the sentence/clause containing the intervention cue.
    chunks=[_clean(x) for x in re.split(r'[.;]|\s/\s|▶|■|□',e) if _clean(x)]
    for ch in chunks:
        if any(k in ch for k in ('설치','첨가','제염','재배','차단','보강','유지','방지','예방')) and 8<=len(ch)<=180:
            return ch
    return None

def _causal_problem(e):
    e=_clean(e)
    for marker in ('때문','원인','악취','벌레','누전','장해','저하','파손','편마멸'):
        if marker in e:
            return marker
    return '문제 상황'

def generate_causal_question(spec,rng=None):
    e=_clean(spec.get('evidence')); action=_causal_action(e)
    if not action: return None
    problem=_causal_problem(e)
    passage=(
        '다음은 한 기술적 문제와 그에 대한 원자료의 설명이다.\n'
        f'자료 | {e}\n\n'
        f'현장에서는 자료와 같은 계열의 {problem}이 발생하였으며, 학생은 원자료의 조치가 이 상황에도 적용 가능한지 검토하고 있다.'
    )
    q={'domain':spec['domain'],'topic':'원인-조치 관계의 적용','points':4,'verifier':'python_semantic_operation',
       'pattern_id':'T4_SEM22','capability_id':CAP_CAUSAL,'question_type':'원인진단/대책적용','material_form':'문제상황자료',
       'intro':'다음 자료를 바탕으로 문제 상황에 대한 조치를 판단하시오.','passage':passage,
       'conditions':['원자료에 직접 명시된 조치만 사용한다.','자료에 없는 효과나 원리를 새로 추가하지 않는다.'],
       'tasks':['자료에서 문제 상황에 대응하기 위해 사용되는 조치를 쓰시오.',
                '그 조치를 적용하는 것이 타당한 이유를 원자료의 문제-조치 관계에 근거하여 설명하시오.'],
       'answer':[action,f'{problem}에 대응하기 위한 조치이기 때문이다.'],
       'solution':[action,f'원자료에서 {problem}과 해당 조치가 직접 연결되어 있으므로 같은 계열의 문제 상황에 적용한다.'],
       'subpoints':[2,2],'sources':_source_meta(spec),'source_context_override':e,
       'source_basis':'원자료에 명시된 문제/원인과 조치의 연결을 새로운 상황에 적용','derived_answer_flags':[True,True]}
    q['fingerprint']=_fp({k:q[k] for k in ['domain','capability_id','passage','answer']}); return q

def generate_semantic_question(db_path,domain,capability_id,rng=None,avoid_fingerprints=None):
    rng=rng or random.Random(); avoid=set(avoid_fingerprints or [])
    specs=list(_semantic_operation_specs(db_path,domain,max_each=50).get(capability_id,[])); rng.shuffle(specs)
    for spec in specs:
        q=generate_directional_question(spec,rng) if capability_id==CAP_DIRECTIONAL else generate_causal_question(spec,rng)
        if q and q.get('fingerprint') not in avoid: return q
    return None

def validation_inventory(db_path,domains,formula_domains=None):
    """Build exactly two executable candidates per domain for the next full Judge suite.
    Selection priority favors already-certified formula operations, then semantic operations,
    and only then ordered sequence as an experimental fallback. Retired 0-pass structures are excluded.
    """
    formula_domains=set(formula_domains or [])
    rows={}; targets=[]
    for d in domains:
        sem=_semantic_operation_specs(db_path,d,max_each=50); old=discover_capabilities(db_path,d)
        avail=[]
        if d in formula_domains: avail.append('deterministic_formula_operation')
        if any(generate_directional_question(x) for x in sem.get(CAP_DIRECTIONAL,[])[:8]): avail.append(CAP_DIRECTIONAL)
        if any(generate_causal_question(x) for x in sem.get(CAP_CAUSAL,[])[:8]): avail.append(CAP_CAUSAL)
        if old.get(CAP_ORDERED): avail.append(CAP_ORDERED)
        chosen=avail[:2]
        rows[d]={'executable_types':avail,'selected_target_types':chosen,'target_count':len(chosen),'target_met':len(chosen)>=2,
                 'candidate_counts':{CAP_DIRECTIONAL:len(sem.get(CAP_DIRECTIONAL,[])),CAP_CAUSAL:len(sem.get(CAP_CAUSAL,[])),CAP_ORDERED:len(old.get(CAP_ORDERED,[]))}}
        for typ in chosen: targets.append({'domain':d,'capability_id':typ,'coverage_key':d+'::'+typ,'status':'VALIDATION_CANDIDATE'})
    return {'domains':rows,'targets':targets,'target_total':len(domains)*2,'constructed_target_total':len(targets),'all_domains_two_targets':all(v['target_met'] for v in rows.values()),
            'note':'R49: full-suite validation pool. Retired paired/swap are excluded; semantic operations are executable but not certified until Judge PASS.'}

# R50: failure-class correction from the R49 full-suite evidence.
# directional_rule_application = 0/9 and cause_intervention_prediction = 0/4.
# They remain discoverable for diagnostics only and can never enter a validation target.
CAP_ORDERED_INSERT='ordered_missing_step_insertion'


def generate_ordered_insertion_question(spec,rng=None):
    rng=rng or random.Random(); items=list(spec.get('items',[])[:4])
    if len(items)<4: return None
    missing=rng.choice([1,2])
    kept=[(i,x) for i,x in enumerate(items) if i!=missing]
    labels=['ㄱ','ㄴ','ㄷ']
    passage='다음은 하나의 절차에서 한 단계가 누락된 학생의 기록이다.\n'+'\n'.join(f'{labels[k]}. {x}' for k,(_,x) in enumerate(kept))
    before='처음' if missing==0 else f'{missing}번째 단계 다음'
    after='마지막' if missing==3 else f'{missing+2}번째 단계 이전'
    q={'domain':spec['domain'],'topic':'원자료 절차 누락 단계 복원','points':4,'verifier':'python_source_capability',
       'pattern_id':'T4_CAP22','capability_id':CAP_ORDERED_INSERT,'question_type':'과정/복원','material_form':'절차자료',
       'intro':'다음 절차 기록을 분석하시오.','passage':passage,
       'conditions':['원자료의 절차적 선후관계를 기준으로 판단한다.','학생 기록에서는 한 단계만 누락되었고 나머지 단계의 상대적 순서는 유지되었다.'],
       'tasks':['누락된 단계의 내용을 쓰시오.','그 단계가 들어갈 위치를 앞뒤 단계와의 관계가 드러나도록 쓰시오.'],
       'answer':[items[missing],f'{before}, {after}'],
       'solution':[f'원자료의 {missing+1}번째 단계는 {items[missing]}이다.',f'원자료 전체 순서에서 {before}이며 {after}이다.'],
       'subpoints':[2,2],'sources':_source_meta(spec),'source_context_override':'\n'.join(items),
       'source_basis':'동일 원자료에 명시된 4단계 이상의 절차 순서에서 한 단계 누락을 복원',
       'derived_answer_flags':[True,True]}
    q['fingerprint']=_fp({k:q[k] for k in ['domain','capability_id','passage','answer']}); return q


def r50_validation_inventory(db_path,domains,formula_domains=None):
    """Only architectures with positive full-suite evidence are promoted.
    R49 zero-pass semantic architectures are retired. Ordered insertion is a NEW
    experimental architecture and is exposed only where the same strict natural
    sequence detector can construct it; it is not certified until Judge evidence.
    """
    formula_domains=set(formula_domains or [])
    rows={}; targets=[]
    # R49 actual full-suite positive evidence: ordered swap passed in these two domains.
    observed_ordered_pass={'기술교육론','발명'}
    for d in domains:
        old=discover_capabilities(db_path,d); seq=old.get(CAP_ORDERED,[])
        certified=[]; experimental=[]
        if d in formula_domains:
            certified.append('deterministic_formula_operation')
        if d in observed_ordered_pass and seq:
            certified.append(CAP_ORDERED)
        if seq:
            # new architecture; never silently counted as certified
            experimental.append(CAP_ORDERED_INSERT)
        rows[d]={'certified_types':certified,'experimental_types':experimental,
                 'retired_types':[CAP_DIRECTIONAL,CAP_CAUSAL,CAP_CONTRAST,CAP_CONDITION],
                 'certified_count':len(certified),'target_met':len(certified)>=2,
                 'candidate_counts':{CAP_ORDERED:len(seq),CAP_ORDERED_INSERT:len(seq)}}
        for typ in certified:
            targets.append({'domain':d,'capability_id':typ,'coverage_key':d+'::'+typ,'status':'AI_VERIFIED'})
    return {'domains':rows,'targets':targets,'target_total':len(domains)*2,
            'certified_target_total':len(targets),'all_domains_two_targets':all(v['target_met'] for v in rows.values()),
            'retired_failure_evidence':{CAP_DIRECTIONAL:'R49 0/9 PASS',CAP_CAUSAL:'R49 0/4 PASS',CAP_CONTRAST:'R46 0/7 PASS',CAP_CONDITION:'R46 0/1 PASS'},
            'note':'R50: 0-pass 구조는 전수검증 후보에서 제거. 실제 PASS한 구조만 certified. 새 ordered insertion은 별도 실험 대상으로만 유지.'}


def generate_r50_experimental_question(db_path,domain,capability_id,rng=None,avoid_fingerprints=None):
    if capability_id!=CAP_ORDERED_INSERT: return None
    rng=rng or random.Random(); avoid=set(avoid_fingerprints or [])
    specs=list(discover_capabilities(db_path,domain).get(CAP_ORDERED,[])); rng.shuffle(specs)
    for spec in specs:
        q=generate_ordered_insertion_question(spec,rng)
        if q and q.get('fingerprint') not in avoid: return q
    return None
