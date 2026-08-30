
import re,json,hashlib,difflib,math
from collections import Counter
from concept_families import concept_family,families_for

def norm(s): return re.sub(r"[^0-9A-Za-z가-힣]+","",str(s)).lower()
def contains_loose(h,n): return bool(norm(n)) and norm(n) in norm(h)

def fingerprint(q):
    core={k:q.get(k) for k in ["domain","topic","pattern_id","passage","tasks","answer"]}
    return hashlib.sha256(json.dumps(core,ensure_ascii=False,sort_keys=True).encode()).hexdigest()[:20]

def complexity_score(q):
    # 배점과 별개로 실제 사고 부담을 측정
    tasks=q.get("tasks",[])
    score=len(tasks)                         # 답안 요소
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
            # source_locked 지문은 [정답란]으로 마스킹되어야 함
            e.append(f"정답 노출:{a}")
    if pts==2 and len(q.get("passage",""))>900:e.append("2점 지문 과다")
    if pts==4:
        if len(tasks)<2:e.append("4점 채점요소 부족")
        if complexity_score(q)<3:e.append("4점 복잡도 미달")
        # 1+1+2이면 실제 3개 요구 강제
        if sp==[1,1,2] and len(tasks)!=3:e.append("1+1+2 요구 수 오류")
    return e

def validate_grounded_question(q,source_context):
    e=_basic(q)
    if q.get("verifier")!="source":e.append("개념형 verifier 오류")
    if q.get("premise_mode")!="source_locked":e.append("지문 전제 잠금 누락")
    ev=q.get("evidence",[]); ans=q.get("answer",[])
    if len(ev)!=len(ans):e.append("근거-정답 수 불일치")
    for a,v in zip(ans,ev):
        if not v or not contains_loose(source_context,v):e.append("근거가 원자료에 없음")
        if len(norm(a))<=55 and not contains_loose(v,a):e.append(f"정답이 대응 근거에 없음:{a}")
    if not q.get("sources"):e.append("출처 없음")
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
    a=norm(q.get("topic","")+" "+q.get("passage",""))
    for p in previous:
        if fams & families_for(p): return True
        b=norm(p.get("topic","")+" "+p.get("passage",""))
        if a and b and difflib.SequenceMatcher(None,a,b).ratio()>=threshold:return True
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
