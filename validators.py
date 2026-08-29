
import re, json, hashlib, difflib

def norm(s):
    return re.sub(r"[^0-9A-Za-z가-힣]+","",str(s)).lower()

def contains_loose(haystack, needle):
    h,n=norm(haystack),norm(needle)
    return bool(n) and n in h

def fingerprint(q):
    core={
      "domain":q.get("domain"),"topic":q.get("topic"),
      "passage":q.get("passage"),"tasks":q.get("tasks"),"answer":q.get("answer")
    }
    return hashlib.sha256(json.dumps(core,ensure_ascii=False,sort_keys=True).encode()).hexdigest()[:20]

def validate_grounded_question(q, source_context):
    errors=[]
    ans=q.get("answer",[])
    evidence=q.get("evidence","")
    question_text=" ".join([
        q.get("intro",""),q.get("passage",""),
        " ".join(q.get("conditions",[]))," ".join(q.get("tasks",[]))
    ])

    if not evidence:
        errors.append("근거 없음")
    else:
        # 복합 근거는 줄 단위로 원문에 각각 존재하는지 확인
        evidence_parts=[x.strip() for x in evidence.split("\n") if x.strip()]
        for ev in evidence_parts:
            if not contains_loose(source_context,ev):
                errors.append("근거 문장이 원자료 문맥에 존재하지 않음")
                break

    for a in ans:
        # 짧은 정답은 근거 안에 실제로 있어야 한다.
        if len(norm(a))<=40 and not contains_loose(evidence,a):
            errors.append(f"정답이 근거 문장에 없음: {a}")
        # 정답 용어가 지문에 그대로 노출되면 탈락
        if 2<=len(norm(a))<=40 and contains_loose(question_text,a):
            errors.append(f"정답 용어가 문제 지문에 노출됨: {a}")

    if not q.get("tasks"):
        errors.append("작성 방법 없음")
    if not q.get("source_name") or not q.get("page_no"):
        errors.append("출처 포인터 없음")
    return errors

def validate_formula_question(q):
    errors=[]
    if q.get("verifier")!="python":
        errors.append("계산형 verifier가 python이 아님")
    if not q.get("answer") or not q.get("solution"):
        errors.append("정답/풀이 없음")
    if not q.get("tasks"):
        errors.append("작성 방법 없음")
    return errors

def too_similar(q, previous, threshold=0.78):
    a=norm(q.get("topic","")+" "+q.get("passage",""))
    for p in previous:
        b=norm(p.get("topic","")+" "+p.get("passage",""))
        if a and b and difflib.SequenceMatcher(None,a,b).ratio()>=threshold:
            return True
    return False

def validate_exam(qs, target_count, target_points):
    errors=[]
    if len(qs)!=target_count:
        errors.append(f"문항 수 {len(qs)} != {target_count}")
    pts=sum(int(q.get("points",0)) for q in qs)
    if pts!=target_points:
        errors.append(f"총점 {pts} != {target_points}")

    fps=[q.get("fingerprint") or fingerprint(q) for q in qs]
    if len(fps)!=len(set(fps)):
        errors.append("중복 fingerprint 존재")

    # 동일 주제 과다 반복 방지
    topics=[norm(q.get("topic","")) for q in qs]
    if len(topics)!=len(set(topics)):
        errors.append("동일 주제 반복")

    return errors
