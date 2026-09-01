BUILDER_API_VERSION = "SAMPLE6-RELATION-AUDIT-R6-20260901"

import random, math, re, sqlite3
from formula_templates import generate_formula_question
from retrieval import related_bundle,bundle_context,official_style_profile,candidate_cluster
from ai_wrapper import rewrite_bundle,safe_bundle_question
from validators import validate_formula_question,validate_grounded_question,too_similar,validate_exam,fingerprint
from patterns import blueprint,weighted_pick
from concept_families import families_for
from quality_judge import select_coherent_bundle,judge_question,judge_exam,judge_ab_pair

DOMAINS=["기술교육론","발명","제조기술","건설기술","생명기술","전기·전자","통신기술","재료역학","수송기술"]
FORMULA_DOMAINS={"재료역학","수송기술","통신기술"}



def _norm_anchor_text(s):
    s=re.sub(r"[\x00-\x1f]+"," ",str(s or ""))
    s=re.sub(r"\s+"," ",s).strip()
    return s

_REL_STOP={
    "기술","종류","특징","방법","과정","내용","관련","사용","이용","경우",
    "한다","있다","대한","통해","위한","의한","것을","에서","으로","되는",
    "설명","자료","다음","해당","구분","분류","단계","요소","기능","개념",
    "목적","장점","단점","효과","원리","구조","형태","상태","의미","활용",
    "적용","정의","특성","방식","구성","관계","조건","문제","예시","기타",
    "따른","따라","위해","때문","또는","그리고","등을","등의","정도"
}

def _anchor_tokens(*parts):
    text=" ".join(_norm_anchor_text(x).lower() for x in parts)
    toks=re.findall(r"[가-힣A-Za-z0-9]{2,}",text)
    return {t for t in toks if t not in _REL_STOP and not t.isdigit()}

def _topic_core(s):
    s=_norm_anchor_text(s).lower()
    s=re.sub(r"^[\(\[]?\d+[\)\].:\-]?\s*","",s)
    s=re.sub(r"^(신재생에너지의|신재생에너지|재생에너지의|재생에너지)\s+","",s)
    s=re.sub(r"[^가-힣a-z0-9]+","",s)
    return s

def _heading_like(text):
    x=_norm_anchor_text(text)
    if not x:
        return True
    bad_exact={
        "종류","특징","개념","목적","장점","단점","활용","분류","구조","구성",
        "형태","기능","원리","방법","과정","효과","특성","의미","내용"
    }
    if x in bad_exact:
        return True
    if re.search(r"(에\s*따라|에\s*따른|에\s*의한)\s*$",x):
        return True
    if re.search(r"(종류|분류|특징|장점|단점|구분|형태에 따른 분류)\s*$",x) and len(x)<=24:
        return True
    return False

def _anchor_quality_reason(a):
    topic=_norm_anchor_text(a.get("topic",""))
    answer=_norm_anchor_text(a.get("answer",""))
    evidence=_norm_anchor_text(a.get("evidence",""))
    if len(topic)<2:
        return "topic_too_short"
    if len(answer)<1:
        return "answer_empty"
    if len(evidence)<10:
        return "evidence_too_short"
    if len(topic)>110 or len(answer)>90:
        return "text_too_long"
    if topic.count("·")>=5:
        return "topic_fragmented"
    junk=[
        r"^[\(\[]?\d+[\)\].]?$",
        r"^[가-힣A-Za-z]\)$",
        r"^(참고|기타|정리|예시|종류|특징|개요|내용)$",
    ]
    if any(re.match(q,topic,re.I) for q in junk):
        return "heading_or_junk"
    if re.match(r"^[가-힣]\s+[가-힣A-Za-z0-9]{4,}(?:\s|$)",topic):
        return "ocr_broken_prefix"
    if _heading_like(answer):
        return "answer_is_heading"

    acore=_topic_core(answer)
    ecore=_topic_core(evidence)
    if len(acore)>=3 and acore not in ecore:
        atoks=_anchor_tokens(answer)
        etoks=_anchor_tokens(evidence)
        if atoks and not (atoks & etoks):
            return "answer_not_supported_in_evidence"
    return ""

def _anchor_ok(a):
    return _anchor_quality_reason(a)==""

def _near_duplicate_anchor(a,b):
    at=_topic_core(a.get("topic",""))
    bt=_topic_core(b.get("topic",""))
    aa=_topic_core(a.get("answer",""))
    ba=_topic_core(b.get("answer",""))
    for x,y in ((at,bt),(aa,ba),(at,ba),(aa,bt)):
        if not x or not y:
            continue
        if x==y:
            return True
        short,long=(x,y) if len(x)<=len(y) else (y,x)
        if len(short)>=4 and short in long and len(short)/max(1,len(long))>=0.60:
            return True
    ta=_anchor_tokens(a.get("topic"),a.get("answer"))
    tb=_anchor_tokens(b.get("topic"),b.get("answer"))
    if ta and tb:
        j=len(ta & tb)/max(1,len(ta | tb))
        if j>=0.82:
            return True
    return False

def _cross_reference_strength(a,b):
    ae=_norm_anchor_text(a.get("evidence","")).lower()
    be=_norm_anchor_text(b.get("evidence","")).lower()
    akeys=_anchor_tokens(a.get("topic"),a.get("answer"))
    bkeys=_anchor_tokens(b.get("topic"),b.get("answer"))
    cross=0
    for t in list(akeys)[:10]:
        if len(t)>=3 and t in be:
            cross+=1
    for t in list(bkeys)[:10]:
        if len(t)>=3 and t in ae:
            cross+=1
    ea=_anchor_tokens(a.get("evidence",""))
    eb=_anchor_tokens(b.get("evidence",""))
    shared={t for t in (ea & eb) if len(t)>=3}
    return cross,shared

def _pair_relation_score(a,b):
    """
    강한 relation 신호가 없으면 같은 페이지여도 REJECT.
    """
    if not _anchor_ok(a) or not _anchor_ok(b):
        return -999.0
    if _near_duplicate_anchor(a,b):
        return -999.0
    if str(a.get("source_name","")) != str(b.get("source_name","")):
        return -999.0
    try:
        gap=abs(int(a.get("page_no",0))-int(b.get("page_no",0)))
    except Exception:
        return -999.0
    if gap>2:
        return -999.0

    cross,shared=_cross_reference_strength(a,b)
    if cross==0 and len(shared)<2:
        return -999.0

    score=cross*3.0 + min(8.0,len(shared)*2.0)
    if gap==0:
        score+=1.0
    elif gap==1:
        score+=0.5

    ta=_anchor_tokens(a.get("topic"),a.get("answer"))
    tb=_anchor_tokens(b.get("topic"),b.get("answer"))
    score += min(3.0,len(ta & tb)*1.5)
    return score

def _bundle_connected(bundle,min_edge=4.0):
    n=len(bundle)
    if n<=1:
        return True,0.0
    adj=[set() for _ in range(n)]
    scores=[]
    for i in range(n):
        for j in range(i+1,n):
            s=_pair_relation_score(bundle[i],bundle[j])
            if s>=min_edge:
                adj[i].add(j); adj[j].add(i); scores.append(s)
    seen={0}; stack=[0]
    while stack:
        x=stack.pop()
        for y in adj[x]:
            if y not in seen:
                seen.add(y); stack.append(y)
    if len(seen)!=n:
        return False,0.0
    return True,(sum(scores)/len(scores) if scores else 0.0)

def _pattern_thinking_types(pattern_id):
    pid=str(pattern_id or "").upper()
    return {
        "T2_REL":["관계판단","근거서술"],
        "T2_ERR":["오류판단","수정"],
        "T2_CMP":["비교판단","구분근거"],
        "T2_DATA":["자료해석","적용판단"],
        "T4_DATA112":["자료해석","개념판단","적용"],
        "T4_ERR22":["오류판단","수정","근거서술"],
        "T4_112":["개념판단","관계설명","적용"],
    }.get(pid,["자료해석","관계판단"])

def _smart_relation_bundle(db_path, domain, need, used_answers, excluded_topics, rng, pattern_id=""):
    con=sqlite3.connect(db_path)
    con.row_factory=sqlite3.Row

    cols={r[1] for r in con.execute("PRAGMA table_info(anchors)").fetchall()}
    required={"domain","topic","answer","evidence","source_name","page_no"}
    missing=sorted(required-cols)
    if missing:
        con.close()
        raise RuntimeError("knowledge.db anchors 스키마 누락: "+", ".join(missing))

    has_conf="confidence" in cols
    conf_expr="COALESCE(confidence,0)" if has_conf else "0"
    rows=con.execute(
        f"""SELECT domain,topic,answer,evidence,source_name,page_no,
                   {conf_expr} AS confidence
            FROM anchors
            WHERE domain=?
            ORDER BY {conf_expr} DESC, source_name, page_no""",
        (domain,)
    ).fetchall()
    con.close()

    used_answers={_topic_core(x) for x in (used_answers or set())}
    excluded_topics={_topic_core(x) for x in (excluded_topics or set())}

    anchors=[]
    seen=set()
    for r in rows:
        a=dict(r)
        a["topic"]=_norm_anchor_text(a.get("topic"))
        a["answer"]=_norm_anchor_text(a.get("answer"))
        a["evidence"]=_norm_anchor_text(a.get("evidence"))
        if not _anchor_ok(a):
            continue
        if _topic_core(a["answer"]) in used_answers:
            continue
        if _topic_core(a["topic"]) in excluded_topics:
            continue
        key=(_topic_core(a["topic"]),_topic_core(a["answer"]),a.get("source_name"),a.get("page_no"))
        if key in seen:
            continue
        seen.add(key)
        anchors.append(a)

    if len(anchors)<need:
        return [],{}

    candidates=[]
    for i,a in enumerate(anchors):
        edges=[]
        for j,c in enumerate(anchors):
            if i==j:
                continue
            s=_pair_relation_score(a,c)
            if s>=4.0:
                if str(pattern_id).upper()=="T2_CMP":
                    cross,shared=_cross_reference_strength(a,c)
                    if _near_duplicate_anchor(a,c) or (cross<1 and len(shared)<2):
                        continue
                edges.append((s,c))
        edges.sort(key=lambda x:x[0],reverse=True)
        if len(edges)<need-1:
            continue

        pool=[x[1] for x in edges[:min(6,len(edges))]]
        trial_sets=[[a]+pool[:need-1]]
        for shift in range(1,min(3,len(pool))):
            tail=pool[shift:shift+need-1]
            if len(tail)==need-1:
                trial_sets.append([a]+tail)

        for chosen in trial_sets:
            if any(
                _near_duplicate_anchor(chosen[x],chosen[y])
                for x in range(len(chosen))
                for y in range(x+1,len(chosen))
            ):
                continue

            connected,graph_score=_bundle_connected(chosen,min_edge=4.0)
            if not connected:
                continue

            if len({str(x.get("source_name","")) for x in chosen})!=1:
                continue
            try:
                pages=[int(x.get("page_no",0)) for x in chosen]
            except Exception:
                continue
            if max(pages)-min(pages)>2:
                continue

            try:
                conf=sum(float(x.get("confidence") or 0) for x in chosen)/len(chosen)
            except Exception:
                conf=0.0
            candidates.append((graph_score+min(1.0,max(0.0,conf)),chosen))

    if not candidates:
        return [],{}

    uniq=[]; seen_sets=set()
    for score,chosen in sorted(candidates,key=lambda x:x[0],reverse=True):
        k=tuple(sorted(_topic_core(x.get("answer","")) for x in chosen))
        if k in seen_sets:
            continue
        seen_sets.add(k)
        uniq.append((score,chosen))
        if len(uniq)>=8:
            break

    top=uniq[:min(4,len(uniq))]
    score,chosen=top[rng.randrange(len(top))]
    relation_meta={
        "master_concept":"동일 원문 문맥에서 직접 연결되는 개념군",
        "relation":"교차참조 또는 공통 핵심 문맥으로 연결된 관계",
        "thinking_types":_pattern_thinking_types(pattern_id),
        "selector_reason":f"Python strong-relation score={score:.2f}",
        "relation_score":round(score,2),
        "selection_mode":"python_strong_relation_graph",
    }
    return chosen,relation_meta

def _compact_candidate_cluster(rows, need, limit=6):
    rows=[dict(r) for r in (rows or []) if _anchor_ok(r)]
    if len(rows)<need:
        return []
    ranked=[]
    for i,a in enumerate(rows):
        neighbors=[]
        for j,c in enumerate(rows):
            if i==j:
                continue
            s=_pair_relation_score(a,c)
            if s>=4.0:
                neighbors.append((s,c))
        neighbors.sort(key=lambda x:x[0],reverse=True)
        if len(neighbors)>=need-1:
            chosen=[a]+[x[1] for x in neighbors[:need-1]]
            connected,score=_bundle_connected(chosen,4.0)
            if connected:
                ranked.append((score,chosen))
    if not ranked:
        return []
    ranked.sort(key=lambda x:x[0],reverse=True)
    return ranked[0][1][:limit]

def _candidate_shape_errors(cand, pat, pts):
    if not isinstance(cand,dict):
        return ["문항 객체가 dict가 아님"]
    errs=[]
    expected=list(pat.get("subpoints",[]))
    if int(cand.get("points",pts) or 0)!=int(pts):
        errs.append("배점 불일치")
    sub=list(cand.get("subpoints",[]))
    if sub and sub!=expected:
        errs.append(f"부분점수 불일치: {sub} != {expected}")
    tasks=list(cand.get("tasks",[]) or [])
    answers=list(cand.get("answer",[]) or [])
    solutions=list(cand.get("solution",[]) or [])
    evidence=list(cand.get("evidence",[]) or [])
    if len(tasks)!=len(expected):
        errs.append(f"작성방법 수 불일치({len(tasks)}/{len(expected)})")
    if len(answers)!=len(expected):
        errs.append(f"정답 수 불일치({len(answers)}/{len(expected)})")
    if solutions and len(solutions)!=len(expected):
        errs.append(f"해설 수 불일치({len(solutions)}/{len(expected)})")
    if evidence and len(evidence)!=len(expected):
        errs.append(f"근거 수 불일치({len(evidence)}/{len(expected)})")
    cores=[_topic_core(x) for x in answers if _topic_core(x)]
    if len(cores)!=len(set(cores)):
        errs.append("정답 중복")
    if any(_heading_like(ans) for ans in answers):
        errs.append("목차/분류형 답안 포함")
    pid=str(cand.get("pattern_id","") or "")
    if pid and pid!=str(pat.get("id","")):
        errs.append(f"패턴 ID 불일치({pid}/{pat.get('id')})")
    if sum(expected)!=pts:
        errs.append("패턴 부분점수 합계 오류")
    return errs

def _validate_sample_exam(qs):
    errs=[]
    if len(qs)!=6:
        errs.append(f"문항수 {len(qs)} != 6")
    got=[int(q.get("points",0) or 0) for q in qs]
    if got!=[2,2,4,4,4,4]:
        errs.append(f"배점 구조 {got} != [2,2,4,4,4,4]")
    if sum(got)!=20:
        errs.append(f"총점 {sum(got)} != 20")
    for i,q in enumerate(qs,1):
        subs=list(q.get("subpoints",[]) or [])
        if sum(subs)!=int(q.get("points",0) or 0):
            errs.append(f"{i}번 부분점수 합 불일치")
        if len(q.get("tasks",[]) or [])!=len(subs):
            errs.append(f"{i}번 작성방법/부분점수 수 불일치")
        if len(q.get("answer",[]) or [])!=len(subs):
            errs.append(f"{i}번 정답/부분점수 수 불일치")
    return errs

def _concept_patterns(rng,pts,first=None):
    from patterns import PATTERNS
    valid=[p for p in PATTERNS if p["points"]==pts and not p["calc"] and p.get("weight",1)>0]
    if pts==4:
        priority={"T4_DATA112":0,"T4_ERR22":1,"T4_112":2}
        valid=sorted(valid,key=lambda p:(priority.get(p["id"],9),-p.get("weight",1)))
    else:
        valid=sorted(valid,key=lambda p:-p.get("weight",1))
    out=[]
    if first is not None: out.append(first)
    for p in valid:
        if not any(x["id"]==p["id"] for x in out): out.append(p)
    return out

def score_pattern(section,count,points):
    defaults={
        "A":[2,2,2,2]+[4]*8,
        "B":[2,2]+[4]*9,
        "SAMPLE":[2,2,4,4,4,4],
    }
    p=defaults.get(section,[])
    if len(p)==count and sum(p)==points:
        return p[:]
    if section=="SAMPLE":
        raise ValueError("SAMPLE은 6문항 20점(2,2,4,4,4,4)만 지원합니다.")
    raise ValueError("실제 A/B 기본 배점 구조만 지원합니다.")

def _prepend_formula_element(q, task, answer, solution):
    q["tasks"]=[task]+list(q.get("tasks",[]))
    q["answer"]=[answer]+list(q.get("answer",[]))
    q["solution"]=[solution]+list(q.get("solution",[]))
    return q

def _enrich_formula(q,pts):
    """계산형 배점을 실제 채점요소로 환산한다. 4점은 서로 다른 3개 요구(1+1+2)만 허용한다."""
    topic=q.get("topic","")
    q["material_form"]="수치자료"
    q["question_type"]="계산/판단" if pts==4 else "간단계산"
    q["concept_family"]=next(iter(families_for(q)), "formula_"+re.sub(r"\W+","_",topic))

    if pts==2:
        if len(q.get("answer",[]))==1:
            relation=None
            if "수직응력" in topic: relation=("적용되는 수직응력의 관계식을 쓸 것.","σ=P/A","σ=P/A")
            elif "정수압" in topic: relation=("정수압의 관계식을 쓸 것.","p=ρgh","p=ρgh")
            elif "열효율" in topic: relation=("열기관 열효율의 관계식을 쓸 것.","η=(QH-QL)/QH×100","η=(QH-QL)/QH×100")
            if relation: q=_prepend_formula_element(q,*relation)
        if len(q.get("answer",[]))<2:return None
        q["answer"]=q["answer"][:2];q["tasks"]=q["tasks"][:2];q["solution"]=q["solution"][:2]
        q["subpoints"]=[1,1]
    else:
        if len(q.get("answer",[]))<3 or len(q.get("tasks",[]))<3:return None
        q["answer"]=q["answer"][:3];q["tasks"]=q["tasks"][:3];q["solution"]=q["solution"][:3]
        canon=[re.sub(r"[^0-9A-Za-z가-힣]+","",str(x)).lower() for x in q["answer"][:3]]
        if len(set(canon))<3:return None
        q["subpoints"]=[1,1,2]
    q["points"]=pts
    q["pattern_id"]="T4_C112" if pts==4 else "T2_C11"
    q["fingerprint"]=fingerprint(q)
    return q

def make_section(db_path,section,count,points,domains=None,api_key="",model="gpt-5.6-luna",
                 ai_enabled=True,ai_quality_enabled=True,judge_model=None,seed=None,
                 previous_questions=None,shared_answers=None,tuning_mode=False):
    rng=random.Random(seed)
    domains=list(domains or DOMAINS)
    scores=score_pattern(section,count,points)
    plan=blueprint(section,scores,domains,rng)
    style=official_style_profile(db_path)
    judge_model=judge_model or model
    writer_active=bool(ai_enabled and api_key)
    quality_active=bool(writer_active and ai_quality_enabled and not tuning_mode)
    selector_active=bool(writer_active)

    prior=list(previous_questions or [])
    qs=[]
    used_topics=set()
    used_answers=set(shared_answers or [])
    used_patterns=[]
    ai_calls=0; fallbacks=0; formula_used=0
    judge_calls=0; judge_rejects=0; selector_calls=0
    formula_cap=2

    diagnostics=[]
    diagnostic_limit=100

    def diag(slot,stage,reason="",**extra):
        if len(diagnostics)>=diagnostic_limit:
            return
        row={
            "section":section,
            "number":slot.get("number"),
            "domain":slot.get("domain"),
            "points":slot.get("points"),
            "stage":stage,
            "reason":str(reason or ""),
        }
        for k,v in extra.items():
            if v not in (None,"",[],{}):
                row[k]=v
        diagnostics.append(row)

    for slot in plan:
        pts=slot["points"]; dom=slot["domain"]; q=None
        wants_calc=slot["question_type"] in {"간단계산","계산/판단"}

        if wants_calc and dom not in FORMULA_DOMAINS:
            slot["question_type"]="자료해석" if pts==4 else "자료식별"
            wants_calc=False

        if wants_calc and dom in FORMULA_DOMAINS and formula_used<formula_cap:
            for _ in range(16 if quality_active else 40):
                cand=generate_formula_question(dom,rng)
                if not cand:
                    diag(slot,"formula_generator","계산형 템플릿 후보 없음")
                    break
                cand=_enrich_formula(cand,pts)
                if not cand:
                    diag(slot,"formula_enrich","계산형 채점요소 조건 불충족")
                    continue
                if too_similar(cand,prior+qs):
                    diag(slot,"formula_similarity","기존 문항과 유사")
                    continue
                formula_errs=validate_formula_question(cand)
                if formula_errs:
                    diag(slot,"formula_python_validator",
                         " / ".join(map(str,formula_errs)) if isinstance(formula_errs,(list,tuple)) else str(formula_errs))
                    continue

                if quality_active:
                    try:
                        judge_calls+=1
                        review=judge_question(api_key,judge_model,cand,"",style)
                    except Exception as ex:
                        review={"pass":False,"reason":"AI 품질심사 호출 실패: "+str(ex)}
                    cand["ai_quality"]=review
                    if not review.get("pass"):
                        judge_rejects+=1
                        diag(slot,"formula_ai_judge",review.get("reason",""),
                             fatal_flags=review.get("fatal_flags",[]),
                             scores=review.get("scores",{}),
                             weakest_point=review.get("weakest_point",""),
                             blind_verdict=review.get("blind_verdict"),
                             grounded_verdict=review.get("grounded_verdict"))
                        continue
                elif tuning_mode:
                    cand["ai_quality"]={"pass":None,"mode":"tuning_fast_python_checked"}
                else:
                    cand["ai_quality"]={"pass":None,"mode":"not_run"}

                q=cand;formula_used+=1;break

        if q is None:
            first_pat=weighted_pick(rng,pts,calc=False,used=used_patterns)
            local_rejected_topics=set()

            # 한 문제 슬롯이 수십 번 반복되지 않도록 AI 후보 예산을 제한한다.
            # 품질 기준은 그대로이며, 실패 원인에 따라 다른 패턴으로 즉시 전환한다.
            slot_candidate_budget = 1 if quality_active else (3 if tuning_mode else 32)
            slot_candidates_used = 0
            # selector 자체 호출도 슬롯당 제한한다. REJECT/timeout도 호출 1회로 계산한다.
            selector_attempt_limit = 1 if quality_active else (0 if tuning_mode else 32)
            selector_attempts = 0

            for pat in _concept_patterns(rng,pts,first_pat):
                if (quality_active or tuning_mode) and (
                    slot_candidates_used >= slot_candidate_budget or
                    (not tuning_mode and selector_attempts >= selector_attempt_limit)
                ):
                    _budget_reason=(
                        f"샘플 후보 예산 소진(candidate={slot_candidates_used}/{slot_candidate_budget})"
                        if tuning_mode else
                        f"AI 호출 예산 소진(candidate={slot_candidates_used}/{slot_candidate_budget}, selector={selector_attempts}/{selector_attempt_limit})"
                    )
                    diag(slot,"slot_budget",_budget_reason,pattern=pat.get("id"))
                    break
                need=len(pat["subpoints"])

                for _ in range(1 if quality_active else (2 if tuning_mode else 32)):
                    if (quality_active or tuning_mode) and (
                        slot_candidates_used >= slot_candidate_budget or
                        (not tuning_mode and selector_attempts >= selector_attempt_limit)
                    ):
                        break
                    relation_meta={}
                    bundle=[]

                    if selector_active:
                        if tuning_mode:
                            bundle,relation_meta=_smart_relation_bundle(
                                db_path,dom,need,used_answers,
                                set(used_topics)|local_rejected_topics,rng,
                                pattern_id=pat.get("id","")
                            )
                            if len(bundle)<need:
                                diag(slot,"python_relation_cluster",
                                     f"강한 관계 anchor 묶음 부족: 필요 {need}, 확보 {len(bundle)}",
                                     pattern=pat.get("id"))
                                break
                            slot_candidates_used += 1
                            # 성공한 관계 묶음은 실패 진단에 기록하지 않는다.
                        else:
                            compact_limit=max(5, min(6, need+2))
                            raw_cluster=candidate_cluster(
                                db_path,dom,used_answers,
                                set(used_topics)|local_rejected_topics,
                                limit=12
                            )
                            cluster=_compact_candidate_cluster(raw_cluster,need,compact_limit)
                            if len(cluster)<need:
                                diag(slot,"candidate_cluster",
                                     f"관계성 필터 후 후보 부족: 필요 {need}, 확보 {len(cluster)}",
                                     pattern=pat.get("id"))
                                break
                            try:
                                selector_calls+=1
                                selector_attempts+=1
                                selected=select_coherent_bundle(
                                    api_key,judge_model,cluster,pts,style,need=need
                                )
                            except Exception as ex:
                                selected=None
                                diag(slot,"coherent_selector_call",
                                     "AI 관계성 선별 호출 실패: "+str(ex),
                                     pattern=pat.get("id"))
                            if not selected:
                                diag(slot,"coherent_selector",
                                     "관계성 선별 REJECT 또는 유효 묶음 없음",
                                     pattern=pat.get("id"),
                                     candidate_topics=[str(x.get("topic","")) for x in cluster[:5]])
                                for x in cluster[:2]:
                                    local_rejected_topics.add(str(x.get("topic","")))
                                continue
                            bundle,relation_meta=selected
                            slot_candidates_used += 1

                        if pts==2:
                            _tt=set(str(x).strip() for x in relation_meta.get("thinking_types",[]) if str(x).strip())
                            _rel=str(relation_meta.get("relation","")).strip()
                            if len(_tt)<2 or len(_rel)<4:
                                diag(slot,"two_point_relation_gate",
                                     "2점 관계형 문항 조건 불충족: 최소 2종 사고행동 + 명시적 관계 필요",
                                     pattern=pat.get("id"),
                                     candidate_topics=[str(x.get("topic","")) for x in bundle])
                                for a in bundle:
                                    local_rejected_topics.add(a["topic"])
                                continue
                            relation_meta["quality_directive"]=(
                                "2점 문항도 두 정답을 각 문장에서 독립적으로 찾아 쓰게 만들지 말 것. "
                                "첫 요소의 판단 또는 자료 해석이 두 번째 요소 판단에 반드시 사용되게 구성하고, "
                                "정답 정의·고유특징을 지문에 거의 그대로 제시하지 말 것. "
                                "현재 패턴의 사고구조를 반드시 지킬 것: "
                                + str(pat.get("quality_rule",pat.get("name","")))
                            )
                    else:
                        bundle=related_bundle(
                            db_path,dom,need,used_answers,
                            set(used_topics)|local_rejected_topics
                        )
                        if len(bundle)<need:
                            diag(slot,"related_bundle",
                                 f"원문 잠금 후보 부족: 필요 {need}, 확보 {len(bundle)}",
                                 pattern=pat.get("id"))
                            break

                    ctx=bundle_context(db_path,bundle)
                    cand=None

                    if selector_active:
                        try:
                            ai_calls+=1
                            cand=rewrite_bundle(
                                api_key,model,bundle,pts,section,pat,
                                slot["question_type"],slot["material_form"],style,
                                source_context=ctx,relation_meta=relation_meta
                            )
                        except Exception as ex:
                            cand=None
                            diag(slot,"question_writer_call",
                                 "AI 문항 작성 호출 실패: "+str(ex),
                                 pattern=pat.get("id"))

                        if cand is not None:
                            shape_errs=_candidate_shape_errors(cand,pat,pts)
                            if shape_errs:
                                diag(slot,"candidate_shape_validator",
                                     " / ".join(shape_errs),
                                     pattern=pat.get("id"))
                                cand=None

                        if cand is not None and quality_active:
                            try:
                                judge_calls+=1
                                review=judge_question(
                                    api_key,judge_model,cand,ctx,style
                                )
                            except Exception as ex:
                                review={"pass":False,"reason":"AI 품질심사 호출 실패: "+str(ex)}
                            cand["ai_quality"]=review

                            if not review.get("pass"):
                                judge_rejects+=1
                                _fatal=set(str(x) for x in review.get("fatal_flags",[]))
                                diag(slot,"question_ai_judge",review.get("reason",""),
                                     pattern=pat.get("id"),
                                     fatal_flags=review.get("fatal_flags",[]),
                                     scores=review.get("scores",{}),
                                     weakest_point=review.get("weakest_point",""),
                                     blind_verdict=review.get("blind_verdict"),
                                     grounded_verdict=review.get("grounded_verdict"))

                                _hard={"ROTE_ONLY","TOO_EASY","DIRECT_ANSWER_LEAK","AMBIGUOUS","DECORATIVE_MATERIAL"}
                                if _fatal & _hard:
                                    for a in bundle:
                                        local_rejected_topics.add(a["topic"])
                                    relation_meta["force_pattern_switch"]=True
                                elif bundle:
                                    local_rejected_topics.add(bundle[0]["topic"])
                                cand=None

                        # 최종 모드와 튜닝 모드 모두 DB/Python grounding은 확인한다.
                        if cand is not None:
                            grounded_errs=validate_grounded_question(
                                cand,ctx,allow_ai_grounded=True
                            )
                            if grounded_errs:
                                diag(slot,"grounded_python_validator",
                                     " / ".join(map(str,grounded_errs)) if isinstance(grounded_errs,(list,tuple)) else str(grounded_errs),
                                     pattern=pat.get("id"))
                                cand=None
                            elif too_similar(cand,prior+qs):
                                diag(slot,"question_similarity",
                                     "기존 문항과 유사",
                                     pattern=pat.get("id"))
                                cand=None
                            elif tuning_mode:
                                cand["ai_quality"]={
                                    "pass":None,
                                    "mode":"tuning_fast_manual_review",
                                    "note":"Blind/Grounded/섹션 AI 심사는 튜닝 속도를 위해 생략"
                                }

                        if cand is None:
                            if bundle and not any(a["topic"] in local_rejected_topics for a in bundle):
                                local_rejected_topics.add(bundle[0]["topic"])
                            if relation_meta.get("force_pattern_switch"):
                                break
                            continue

                    else:
                        cand=safe_bundle_question(
                            bundle,pts,pat,slot["question_type"],slot["material_form"]
                        )
                        errs=validate_grounded_question(cand,ctx)
                        if errs or too_similar(cand,prior+qs):
                            diag(slot,"safe_fallback_validator",
                                 " / ".join(map(str,errs)) if errs else "기존 문항과 유사",
                                 pattern=pat.get("id"))
                            if bundle:
                                local_rejected_topics.add(bundle[0]["topic"])
                            continue
                        cand["ai_quality"]={"pass":None,"mode":"source_locked_fallback"}
                        fallbacks+=1

                    q=cand
                    used_answers.update(q["answer"])
                    for a in bundle:
                        used_topics.add(a["topic"])
                    break

                if q is not None:
                    break

        if q is None:
            err=RuntimeError(
                f"{section} {slot['number']}번({dom})을 품질 기준으로 만들 수 없습니다. "
                "품질 기준을 낮추지 않고 다른 청사진을 재시도합니다."
            )
            err.generation_diagnostics=diagnostics[-15:]
            raise err

        q["number"]=slot["number"]
        q["blueprint_domain"]=dom
        q["concept_families"]=sorted(families_for(q))
        q["fingerprint"]=q.get("fingerprint") or fingerprint(q)
        qs.append(q)
        used_patterns.append(q.get("pattern_id"))

    errs=_validate_sample_exam(qs) if tuning_mode else validate_exam(qs,count,points)
    if errs:
        err=RuntimeError("시험 자동검증 실패: "+" / ".join(errs))
        err.generation_diagnostics=diagnostics[-15:]+[{
            "section":section,"stage":"exam_python_validator",
            "reason":" / ".join(errs)
        }]
        raise err

    exam={
      "exam_title":f"기술 임용 모의고사 전공 {section}",
      "section":section,"total_points":points,"questions":qs,"verified":False if tuning_mode else True,
      "blueprint":plan,
      "generation_stats":{
          "ai_calls":ai_calls,
          "safe_fallbacks":fallbacks,
          "formula_questions":formula_used,
          "ai_judge_calls":judge_calls,
          "ai_judge_rejects":judge_rejects,
          "ai_selector_calls":selector_calls,
          "tuning_relation_selector":"python" if tuning_mode else "ai",
      },
      "used_answers":list(used_answers),
      "verification_note":("품질 튜닝용 6문항: 강한 관계 그래프 선별 + DB/Python 정답·구조 검증 + AI 작성. 최종 AI 품질심사 생략" if tuning_mode else "DB/Python 정답 고정 + 구조검증 + AI 독립 품질심사 통과")
    }

    if quality_active and not tuning_mode:
        try:
            judge_calls+=1
            section_review=judge_exam(api_key,judge_model,exam,style)
        except Exception as ex:
            section_review={"pass":False,"reason":"섹션 AI 심사 호출 실패: "+str(ex)}
        exam["section_ai_quality"]=section_review
        exam["generation_stats"]["ai_judge_calls"]=judge_calls
        if not section_review.get("pass"):
            err=RuntimeError(
                f"{section} 섹션 전체 AI 품질심사 탈락: "
                +str(section_review.get("reason",""))
            )
            err.generation_diagnostics=diagnostics[-12:]+[{
                "section":section,
                "stage":"section_ai_judge",
                "reason":str(section_review.get("reason","")),
                "scores":{
                    "exam_realism":section_review.get("exam_realism"),
                    "variety":section_review.get("variety"),
                    "difficulty_balance":section_review.get("difficulty_balance"),
                }
            }]
            raise err
    else:
        exam["section_ai_quality"]={
            "pass":None,
            "mode":"tuning_fast_manual_review" if tuning_mode else "not_run"
        }

    return exam


def make_quality_sample(db_path,domains=None,api_key="",model="gpt-5.6-luna",
                        ai_enabled=True,judge_model=None,seed=None):
    """
    품질 튜닝 전용 6문항.
    정확히 2점 2개 + 4점 4개.
    AI 관계성 selector/Blind/Grounded/섹션 심사는 호출하지 않는다.
    Python 관계묶음 + Luna writer + deterministic grounding 검증만 수행한다.
    """
    exam=make_section(
        db_path,"SAMPLE",6,20,domains,api_key,model,
        ai_enabled,False,judge_model,seed=seed,
        tuning_mode=True
    )

    stats=exam.get("generation_stats",{})
    if exam.get("section")!="SAMPLE":
        raise RuntimeError("SAMPLE6 내부 검증 실패: section이 SAMPLE이 아닙니다.")

    scores=[int(q.get("points",0)) for q in exam.get("questions",[])]
    if scores != [2,2,4,4,4,4]:
        raise RuntimeError(
            "SAMPLE6 내부 검증 실패: 배점이 [2,2,4,4,4,4]가 아닙니다."
        )

    if int(stats.get("ai_selector_calls",0) or 0) != 0:
        raise RuntimeError(
            "SAMPLE6 내부 검증 실패: 튜닝 모드에서 AI selector가 호출되었습니다. "
            "현재 배포 파일 버전이 섞여 있습니다."
        )

    judge_count=int(
        stats.get("ai_quality_judge_calls",
        stats.get("ai_judge_calls",
        stats.get("judge_calls",0))) or 0
    )
    if judge_count != 0:
        raise RuntimeError(
            "SAMPLE6 내부 검증 실패: 튜닝 모드에서 AI 품질심사가 호출되었습니다. "
            "현재 배포 파일 버전이 섞여 있습니다."
        )

    exam["builder_api_version"]=BUILDER_API_VERSION
    exam["sample_mode"]="PYTHON_RELATION_WRITER_ONLY"
    return exam


def make_ab(db_path,a_count=12,a_points=40,b_count=11,b_points=40,domains=None,
            api_key="",model="gpt-5.6-luna",ai_enabled=True,ai_quality_enabled=True,
            judge_model=None,seed=None):
    base=0 if seed is None else int(seed)
    last_error=None
    ab_diagnostics=[]

    def collect_error(ex,attempt):
        rows=getattr(ex,"generation_diagnostics",None)
        if rows:
            for row in rows:
                x=dict(row)
                x["attempt"]=attempt
                ab_diagnostics.append(x)
        else:
            ab_diagnostics.append({
                "attempt":attempt,
                "stage":"section_or_pair",
                "reason":str(ex)
            })

    for a_try in range(1):
        a_seed=None if seed is None else base + a_try*1000
        try:
            A=make_section(
                db_path,"A",a_count,a_points,domains,api_key,model,
                ai_enabled,ai_quality_enabled,judge_model,
                seed=a_seed
            )
        except Exception as ex:
            last_error=ex
            collect_error(ex,f"A-{a_try+1}")
            continue

        af=set().union(*(families_for(q) for q in A["questions"]))
        for b_try in range(1):
            b_seed=None if seed is None else base + 1 + a_try*1000 + b_try*37
            try:
                B=make_section(
                    db_path,"B",b_count,b_points,domains,api_key,model,
                    ai_enabled,ai_quality_enabled,judge_model,
                    seed=b_seed,
                    previous_questions=A["questions"],
                    shared_answers=set(A.get("used_answers",[]))
                )
                bf=set().union(*(families_for(q) for q in B["questions"]))
                if af & bf:
                    last_error=RuntimeError(
                        "A/B concept-family 중복 발견: "+", ".join(sorted(af&bf))
                    )
                    collect_error(last_error,f"A-{a_try+1}/B-{b_try+1}")
                    continue
                quality_active=bool(ai_enabled and ai_quality_enabled and api_key)
                if quality_active:
                    try:
                        pair_review=judge_ab_pair(
                            api_key,judge_model or model,A,B,
                            official_style_profile(db_path)
                        )
                    except Exception as ex:
                        pair_review={"pass":False,"reason":"A/B 종합 AI 심사 호출 실패: "+str(ex)}
                    if not pair_review.get("pass"):
                        last_error=RuntimeError(
                            "A/B 종합 AI 품질심사 탈락: "
                            +str(pair_review.get("reason",""))
                        )
                        collect_error(last_error,f"A-{a_try+1}/B-{b_try+1}")
                        continue
                    A["ab_pair_ai_quality"]=pair_review
                    B["ab_pair_ai_quality"]=pair_review
                else:
                    A["ab_pair_ai_quality"]={"pass":None,"mode":"not_run"}
                    B["ab_pair_ai_quality"]={"pass":None,"mode":"not_run"}

                A["effective_seed"]=a_seed
                B["effective_seed"]=b_seed
                A["generation_retry"]={"A_attempt":a_try+1}
                B["generation_retry"]={"B_attempt":b_try+1}
                return A,B
            except Exception as ex:
                last_error=ex
                collect_error(ex,f"A-{a_try+1}/B-{b_try+1}")
                continue

    err=RuntimeError(
        "정답/품질 기준을 낮추지 않은 상태에서 A/B 편성에 실패했습니다: "
        +str(last_error)
    )
    err.generation_diagnostics=ab_diagnostics[-30:]
    raise err
