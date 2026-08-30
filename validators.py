
import re,json,hashlib,difflib,math
from collections import Counter

def norm(s): return re.sub(r"[^0-9A-Za-z가-힣]+","",str(s)).lower()

def contains_loose(haystack,needle):
    h,n=norm(haystack),norm(needle)
    return bool(n) and n in h

def fingerprint(q):
    core={
      "domain":q.get("domain"),"topic":q.get("topic"),"pattern_id":q.get("pattern_id"),
      "passage":q.get("passage"),"tasks":q.get("tasks"),"answer":q.get("answer")
    }
    return hashlib.sha256(json.dumps(core,ensure_ascii=False,sort_keys=True).encode()).hexdigest()[:20]

def _basic(q):
    e=[]
    pts=int(q.get("points",0))
    if pts not in (2,4): e.append("배점 오류")
    sp=q.get("subpoints",[])
    if not sp or sum(sp)!=pts: e.append("부분점수 합계 오류")
    if len(sp)!=len(q.get("answer",[])): e.append("부분점수-정답 요소 수 불일치")
    if len(q.get("tasks",[]))!=len(q.get("answer",[])): e.append("작성요구-정답 요소 수 불일치")
    if not q.get("passage"): e.append("지문 없음")
    if len(q.get("tasks",[]))>4: e.append("하위 요구 과다")

    qtext=" ".join([q.get("intro",""),q.get("passage","")," ".join(q.get("conditions",[]))])
    if "그림" in qtext and not q.get("visual_data"): e.append("실제 그림 없이 그림 언급")
    for a in q.get("answer",[]):
        if 2<=len(norm(a))<=45 and contains_loose(qtext,a):
            e.append(f"정답 노출:{a}")

    # 2점이 지나치게 길거나 4점이 지나치게 빈약한 경우
    n=len(q.get("passage",""))
    if pts==2 and n>1000: e.append("2점 지문 과다")
    if pts==4 and len(q.get("tasks",[]))<2: e.append("4점 채점요소 부족")
    return e

def validate_grounded_question(q,source_context):
    e=_basic(q)
    if q.get("verifier")!="source": e.append("개념형 verifier 오류")
    evid=q.get("evidence",[])
    if isinstance(evid,str): evid=[evid]
    ans=q.get("answer",[])
    if len(evid)!=len(ans): e.append("근거-정답 수 불일치")
    for a,ev in zip(ans,evid):
        if not ev or not contains_loose(source_context,ev):
            e.append("근거가 원자료에 없음")
        if len(norm(a))<=55 and not contains_loose(ev,a):
            e.append(f"정답이 대응 근거에 없음:{a}")
    if not q.get("sources"): e.append("출처 없음")
    return list(dict.fromkeys(e))

def validate_formula_question(q):
    e=_basic(q)
    if q.get("verifier")!="python": e.append("계산형 verifier 오류")
    if not q.get("solution"): e.append("풀이 없음")
    if len(q.get("tasks",[]))!=len(q.get("answer",[])): e.append("계산형 요구-정답 불일치")
    for a in q.get("answer",[]):
        if isinstance(a,(int,float)) and not math.isfinite(a): e.append("비정상 수치")
    return list(dict.fromkeys(e))

def too_similar(q,previous,threshold=.72):
    a=norm(q.get("topic","")+" "+q.get("passage",""))
    for p in previous:
        if q.get("topic")==p.get("topic"): return True
        b=norm(p.get("topic","")+" "+p.get("passage",""))
        if a and b and difflib.SequenceMatcher(None,a,b).ratio()>=threshold: return True
    return False

def validate_exam(qs,target_count,target_points):
    e=[]
    if len(qs)!=target_count:e.append("문항 수 불일치")
    if sum(int(q.get("points",0)) for q in qs)!=target_points:e.append("총점 불일치")
    fps=[q.get("fingerprint") or fingerprint(q) for q in qs]
    if len(fps)!=len(set(fps)):e.append("중복 문항")
    # 부분점수 없는 문항 금지
    for q in qs:
        if sum(q.get("subpoints",[]))!=q.get("points"):e.append(f"{q.get('number')}번 부분점수 오류")
    # 동일 유형 3연속 금지
    types=[q.get("question_type") for q in qs]
    for i in range(len(types)-2):
        if types[i] and types[i]==types[i+1]==types[i+2]: e.append("동일 문항유형 3연속")
    # 영역 쏠림
    c=Counter(q.get("domain") for q in qs)
    if qs and max(c.values())>max(3,math.ceil(len(qs)*0.35)):e.append("특정 영역 과다")
    return list(dict.fromkeys(e))
