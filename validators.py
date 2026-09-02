import re,json,hashlib,difflib,math
from collections import Counter
from concept_families import concept_family,families_for

def norm(s): return re.sub(r"[^0-9A-Za-z가-힣]+","",str(s)).lower()
def contains_loose(h,n): return bool(norm(n)) and norm(n) in norm(h)

def fingerprint(q):
    core={k:q.get(k) for k in ["domain","topic","pattern_id","passage","tasks","answer"]}
    return hashlib.sha256(json.dumps(core,ensure_ascii=False,sort_keys=True).encode()).hexdigest()[:20]

def complexity_score(q):
    tasks=q.get("tasks",[])
    score=len(tasks)
    score += sum(max(0,int(p)-1) for p in q.get("subpoints",[]))
    if q.get("verifier")=="python": score+=1
    if any(k in " ".join(tasks) for k in ["이유","근거","비교","설명","서술","판단"]): score+=1
    if len(q.get("conditions",[]))>=2: score+=1
    if q.get("material_form") in {"과정자료","표형자료"}: score+=0.5
    return score

def _basic(q):
    e=[]; pts=int(q.get("points",0)); sp=q.get("subpoints",[]); tasks=q.get("tasks",[])
    if pts not in (2,4):e.append("배점 오류")
    if not sp or sum(sp)!=pts:e.append("부분점수 합계 오류")
    if len(sp)!=len(q.get("tasks",[])):e.append("부분점수-작성요구 수 불일치")
    if len(sp)!=len(q.get("answer",[])):e.append("부분점수-정답 수 불일치")
    if not q.get("passage"):e.append("지문 없음")
    qtext=" ".join([q.get("intro",""),q.get("passage","")," ".join(q.get("conditions",[]))])
    if "그림" in qtext and not q.get("visual_data"):e.append("실제 그림 없이 그림 언급")
    for a in q.get("answer",[]):
        if 2<=len(norm(a))<=45 and contains_loose(qtext,a):
            e.append(f"정답 노출:{a}")
    if pts==2 and len(q.get("passage",""))>900:e.append("2점 지문 과다")
    if pts==4:
        # 중요: 2+2는 작성요구 2개가 정상이다.
        # 기존 "최소 3개 요구"는 T4_ERR22 같은 2+2 패턴과 논리적으로 충돌했다.
        if len(tasks)<2:e.append("4점 채점요소 부족(최소 2개 요구)")
        if complexity_score(q)<4:e.append("4점 복잡도 미달")
        if sp==[1,1,2] and len(tasks)!=3:e.append("1+1+2 요구 수 오류")
        if sp==[2,2] and len(tasks)!=2:e.append("2+2 요구 수 오류")
    return e

def _max_copy_similarity(passage,evidence,answer):
    """정답만 지우고 원문 정의를 거의 그대로 복사했는지 보수적으로 검사."""
    p=norm(passage)
    ev=str(evidence)
    if answer:
        ev=re.sub(re.escape(str(answer)),"",ev,flags=re.I)
    e=norm(ev)
    if len(e)<18 or len(p)<18:return 0.0
    if e in p:return 1.0
    L=len(e)
    if len(p)<=L*2:
        return difflib.SequenceMatcher(None,p,e).ratio()
    step=max(1,L//5)
    best=0.0
    for i in range(0,max(1,len(p)-L+1),step):
        w=p[i:i+L]
        best=max(best,difflib.SequenceMatcher(None,w,e).ratio())
    return best


_REASON_TERMS=("이유","근거","관계","비교","판단","설명","계산","수정","적용","영향","과정","순서","해석","도출","예측")
_DEPENDENCY_TERMS=("앞의","앞서","위 판단","위 결과","이를 이용","이를 근거","그 결과","이 결과","두 결과","공통 원리","같은 원리","앞 문항","앞선")

def _recall_only_task(task):
    t=str(task or "")
    recall=any(k in t for k in ("명칭","용어","이름","쓰시오","적으시오"))
    reason=any(k in t for k in _REASON_TERMS)
    return bool(recall and not reason)

def _dependency_quality_errors(q):
    e=[]
    pts=int(q.get("points",0))
    tasks=[str(x) for x in q.get("tasks",[])]
    pid=str(q.get("pattern_id","")).upper()

    if pts==2 and len(tasks)>=2 and all(_recall_only_task(t) for t in tasks[:2]):
        e.append("2점 독립 단순회상형")

    if pts==4:
        recall_count=sum(1 for t in tasks if _recall_only_task(t))
        if recall_count>=2:
            e.append("4점 독립 명칭회상 과다")

        if pid in {"T4_DATA112","T4_112"} and tasks:
            last=tasks[-1]
            if not any(k in last for k in _DEPENDENCY_TERMS):
                e.append("4점 사고사슬 연결표지 부족")

        if pid=="T4_ERR22" and len(tasks)>=2:
            joined=" ".join(tasks[1:])
            if not any(k in joined for k in ("공통","같은 원리","위 판단","앞의","이를 근거","관계")):
                e.append("4점 오류수정 공통근거 연결 부족")
    return e


def _two_point_recall_errors(q):
    e=[]
    if int(q.get("points",0))!=2:
        return e
    tasks=[str(x) for x in q.get("tasks",[])]
    if len(tasks)<2:
        return e

    recall_flags=[_recall_only_task(t) for t in tasks[:2]]
    if all(recall_flags):
        e.append("2점 양쪽 모두 단순 명칭회상")
        return e

    # 첫 요구가 명칭회상이라면 두 번째는 실제 판단/근거/수정/비교를 요구해야 한다.
    if recall_flags[0]:
        second=tasks[1]
        if not any(k in second for k in ("이유","근거","판단","수정","비교","차이","관계","적용","해석")):
            e.append("2점 명칭회상 후속 사고요구 부족")
    return e


def _material_length_errors(q):
    e=[]
    pts=int(q.get("points",0))
    passage=str(q.get("passage",""))
    conditions=[str(x) for x in q.get("conditions",[])]
    tasks=[str(x) for x in q.get("tasks",[])]

    # 너무 긴 지문은 실제 임용 난도보다 '읽기량'으로 어려워지는 문제를 만든다.
    if pts==2:
        if len(passage)>650:
            e.append("2점 자료 과다장문")
        if len(conditions)>2:
            e.append("2점 조건 과다")
        if any(len(t)>170 for t in tasks):
            e.append("2점 작성요구 과다장문")
    elif pts==4:
        if len(passage)>1150:
            e.append("4점 자료 과다장문")
        if len(conditions)>3:
            e.append("4점 조건 과다")
        if any(len(t)>210 for t in tasks):
            e.append("4점 작성요구 과다장문")
    return e

def _independent_fact_listing_errors(q):
    """
    독립 사례·종류의 과도한 병렬 나열을 차단한다.
    R20: ONE-ANCHOR 2점은 한 개념의 근거 설명이므로 쉼표 개수만으로 오탐하지 않는다.
    """
    e=[]
    passage=str(q.get("passage",""))
    pts=int(q.get("points",0))
    mode=str(q.get("selection_mode",""))
    list_marks = passage.count(",") + passage.count("·") + passage.count(";")
    enumerators = sum(passage.count(x) for x in ("A","B","C","D","①","②","③","④"))

    if pts==2 and mode=="python_exam_value_one_anchor_t2":
        # 실제 열거 구조가 있을 때만 차단. 단일 개념 설명의 쉼표는 허용.
        explicit_list = (
            enumerators>=4
            or len(re.findall(r"(?:^|\s)\d+[.)]\s*",passage))>=4
            or sum(passage.count(k) for k in ("첫째","둘째","셋째","넷째"))>=4
        )
        if explicit_list or list_marks>=14:
            e.append("2점 독립사실 나열 과다")
        return e

    if pts==2 and (list_marks>=8 or enumerators>=5):
        e.append("2점 독립사실 나열 과다")
    if pts==4 and (list_marks>=16 or enumerators>=8):
        e.append("4점 독립사실 나열 과다")
    return e

def _four_point_direct_support_errors(q):
    """
    4점에서 자료가 사실상 정답표처럼 되어 있는 경우를 보수적으로 차단한다.
    정답 문자열 노출은 기존 grounding에서 이미 잡으므로,
    여기서는 '근거문장 거의 그대로 + task는 명칭/용어 회상' 조합만 추가 차단한다.
    """
    e=[]
    if int(q.get("points",0))!=4:
        return e

    passage=str(q.get("passage",""))
    tasks=[str(x) for x in q.get("tasks",[])]
    evidences=[str(x) for x in q.get("evidence",[])]
    answers=[str(x) for x in q.get("answer",[])]

    direct_hits=0
    for i,ev in enumerate(evidences):
        task=tasks[i] if i<len(tasks) else ""
        ans=answers[i] if i<len(answers) else ""
        if not _recall_only_task(task):
            continue
        sim=_max_copy_similarity(passage,ev,ans)
        if sim>=0.68:
            direct_hits += 1

    if direct_hits>=2:
        e.append("4점 자료 직접지원 과다(정답표형)")
    return e

def _answer_distance_errors(q):
    """
    정답 단어 자체 노출뿐 아니라 '근거를 거의 보여주고 명칭만 쓰게 하는' 저변환 문항을 줄인다.
    단순회상 task일 때는 기존 0.86보다 엄격한 0.74 유사도 기준을 적용한다.
    """
    e=[]
    passage=str(q.get("passage",""))
    tasks=[str(x) for x in q.get("tasks",[])]
    for i,(ans,ev) in enumerate(zip(q.get("answer",[]),q.get("evidence",[]))):
        task=tasks[i] if i<len(tasks) else ""
        if _recall_only_task(task):
            sim=_max_copy_similarity(passage,ev,ans)
            if sim>=0.74:
                e.append("정답 변환거리 부족(근거→명칭 회상)")
                break
    return e


def _subpoint_scope_errors(q):
    """
    부분점수와 task가 요구하는 채점행동 수가 맞는지 확인한다.
    특히 1점 task에서 '정답 + 이유/근거/추가 결과'를 동시에 요구하는 것을 차단한다.
    """
    e=[]
    sp=list(q.get("subpoints",[]) or [])
    tasks=[str(x or "") for x in (q.get("tasks",[]) or [])]

    reasoning_terms=(
        "이유","근거","설명","과정","관계","비교","차이","수정",
        "적용","영향","까닭","왜","도출","해석"
    )
    first_action_terms=(
        "쓰고","적고","제시하고","구하고","계산하고","판단하고",
        "분류하고","선택하고","수정하고","비교하고"
    )

    for i,(pt,task) in enumerate(zip(sp,tasks),1):
        if int(pt)!=1:
            continue
        compact=re.sub(r"\s+"," ",task).strip()

        # 가장 문제가 많았던 형태:
        # "A를 쓰고, B를 설명하시오" / "값을 구하고 원리의 명칭을 쓰시오"
        chained=any(k in compact for k in first_action_terms)
        has_reason=any(k in compact for k in reasoning_terms)
        # 연결 뒤 또 하나의 명시적 답 요구가 있는 경우도 2개 채점행동으로 본다.
        repeated_answer_action=(
            compact.count("쓰시오") + compact.count("적으시오")
            + compact.count("구하시오") + compact.count("설명하시오")
            + compact.count("제시하시오")
        ) >= 2

        if (chained and (has_reason or re.search(r"(명칭|용어|값|개념).{0,30}(쓰시오|적으시오|구하시오)",compact))) or repeated_answer_action:
            e.append(f"1점 소문항 과다요구({i}번 task)")
    return e

def static_quality_errors(q,require_ai_quality=True):
    e=[]
    if q.get("premise_mode")!="ai_grounded":
        return e
    pts=int(q.get("points",0))
    passage=q.get("passage","")
    for ans,ev in zip(q.get("answer",[]),q.get("evidence",[])):
        if _max_copy_similarity(passage,ev,ans)>=0.86:
            e.append("원문 정의/근거 과다복사")
            break
    e.extend(_dependency_quality_errors(q))
    e.extend(_two_point_recall_errors(q))
    e.extend(_answer_distance_errors(q))
    e.extend(_four_point_direct_support_errors(q))
    e.extend(_material_length_errors(q))
    e.extend(_independent_fact_listing_errors(q))
    e.extend(_subpoint_scope_errors(q))
    if pts==4:
        aq=q.get("ai_quality",{}) or {}
        # 최종 A/B: judge 결과가 있으면 judge의 thinking_types 사용.
        # 6문항 튜닝: judge를 의도적으로 생략하므로 writer의 intended_thinking_types 사용.
        think_src=aq.get("thinking_types",[]) or q.get("intended_thinking_types",[])
        think={str(x).strip() for x in think_src if str(x).strip()}
        if len(think)<2:e.append("4점 사고행동 다양성 부족")
        tasks=" ".join(q.get("tasks",[]))
        reasoning_terms=("이유","근거","관계","비교","판단","설명","계산","수정","적용","영향","과정","순서","해석","도출")
        if sum(1 for k in reasoning_terms if k in tasks)<1:
            e.append("4점 단순회상형")
    return e

def validate_grounded_question(q,source_context,allow_ai_grounded=False,require_ai_quality=True):
    """
    최종 A/B: require_ai_quality=True -> ai_grounded는 AI judge PASS 필수.
    6문항 튜닝: require_ai_quality=False -> AI judge 없이 deterministic 검증만 수행.
    """
    e=_basic(q)
    if q.get("verifier")!="source":e.append("개념형 verifier 오류")
    mode=q.get("premise_mode")
    if mode=="ai_grounded":
        if not allow_ai_grounded:e.append("AI 생성 지문은 품질심사 전 통과 불가")
        if require_ai_quality:
            aq=q.get("ai_quality",{}) or {}
            if allow_ai_grounded and not aq.get("pass"):e.append("AI 품질심사 미통과")
    elif mode!="source_locked":
        e.append("지문 전제 모드 오류")
    ev=q.get("evidence",[]); ans=q.get("answer",[])
    derived=list(q.get("derived_answer_flags",[]) or [])
    if len(ev)!=len(ans):e.append("근거-정답 수 불일치")
    for i,(a,v) in enumerate(zip(ans,ev)):
        if not v or not contains_loose(source_context,v):e.append("근거가 원자료에 없음")
        is_derived=(i<len(derived) and bool(derived[i]))
        # ㉠/㉡ 같은 Python 결정형 답은 원자료의 사실을 근거로 계산된 채점값이므로
        # answer 문자열 자체가 evidence에 있을 필요는 없다.
        if (not is_derived) and len(norm(a))<=55 and not contains_loose(v,a):
            e.append(f"정답이 대응 근거에 없음:{a}")
    if not q.get("sources"):e.append("출처 없음")
    e.extend(static_quality_errors(q,require_ai_quality=require_ai_quality))
    if q.get("premise_mode")=="ai_grounded":
        if not str(q.get("master_concept","")).strip():e.append("AI 지문 master concept 누락")
        if not str(q.get("relation","")).strip():e.append("AI 지문 개념 관계 누락")
    if q.get("points")==4 and q.get("sources"):
        srcs=q.get("sources") or []
        names={str(s.get("source_name","")) for s in srcs}
        pages=[int(s.get("page_no",0)) for s in srcs if str(s.get("page_no","")).isdigit()]
        if len(names)>1:e.append("4점 자료 응집도 미달(서로 다른 출처 혼합)")
        if pages and max(pages)-min(pages)>2:e.append("4점 자료 응집도 미달(페이지 간격 과다)")
    return list(dict.fromkeys(e))

def validate_formula_question(q):
    e=_basic(q)
    if q.get("verifier")!="python":e.append("계산형 verifier 오류")
    if not q.get("solution"):e.append("풀이 없음")
    if len(q.get("tasks",[]))!=len(q.get("answer",[])):e.append("계산형 요구-정답 불일치")
    if q.get("points")==4 and len(q.get("answer",[]))<3:e.append("4점 계산형은 최소 3개 채점요소")
    return list(dict.fromkeys(e))

def too_similar(q,previous,threshold=.72):
    fams=families_for(q)
    a_topic=norm(q.get("topic",""))
    a=norm(q.get("topic","")+" "+q.get("passage",""))
    qdom=norm(q.get("domain",""))
    for p in previous:
        p_topic=norm(p.get("topic",""))
        pdom=norm(p.get("domain",""))
        shared_family=bool(fams & families_for(p))

        # 같은 broad family(예: invention_thinking)라는 이유만으로는 탈락시키지 않는다.
        # 같은 영역에서 topic 자체가 매우 유사할 때만 family 중복을 강한 중복으로 본다.
        if shared_family and qdom and qdom==pdom and a_topic and p_topic:
            topic_ratio=difflib.SequenceMatcher(None,a_topic,p_topic).ratio()
            if a_topic==p_topic or topic_ratio>=0.78:
                return True

        b=norm(p.get("topic","")+" "+p.get("passage",""))
        if a and b and difflib.SequenceMatcher(None,a,b).ratio()>=threshold:
            return True
    return False

def validate_exam(qs,target_count,target_points):
    e=[]
    if len(qs)!=target_count:e.append("문항 수 불일치")
    if sum(int(q.get("points",0)) for q in qs)!=target_points:e.append("총점 불일치")
    fps=[q.get("fingerprint") or fingerprint(q) for q in qs]
    if len(fps)!=len(set(fps)):e.append("중복 문항")
    seen=set()
    for q in qs:
        fs=families_for(q)
        if seen & fs: e.append("동일 concept family 중복")
        seen |= fs
    c=Counter(q.get("domain") for q in qs)
    if qs:
        unique=max(1,len(c))
        allowed=max(2,math.ceil(len(qs)/unique))
        if max(c.values())>allowed:e.append("영역 편중")
    types=[q.get("question_type") for q in qs]
    for i in range(len(types)-2):
        if types[i] and types[i]==types[i+1]==types[i+2]:e.append("동일 유형 3연속")
    forms=Counter(q.get("material_form","서술자료") for q in qs if q.get("verifier")=="source")
    if len(qs)>=10 and len(forms)<3:e.append("자료형 다양성 부족")
    return list(dict.fromkeys(e))
