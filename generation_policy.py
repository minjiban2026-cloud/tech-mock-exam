import re

CIRCUIT_KEYWORDS=["회로","키르히호프","테브난","노튼","밀만","rlc","rc회로","rl회로","정류","다이오드","트랜지스터","mosfet","op amp","연산증폭기","논리회로","논리 게이트","플립플롭","브리지 정류"]
BAD_ANSWER_TOKENS=["요약","정리","단원","목차","개요","서브노트","교과서","part","내용체계","학습목표"]
BAD_ANCHOR_TOKENS=["아이디어의구상","발명기법과실제","정보통신기술"]+BAD_ANSWER_TOKENS

def _text(q):
    return " ".join([str(q.get("domain","")),str(q.get("topic","")),str(q.get("intro","")),str(q.get("passage",""))," ".join(q.get("conditions",[]))," ".join(q.get("tasks",[]))]).lower()

def is_circuit_question(q):
    t=_text(q); return any(k.lower() in t for k in CIRCUIT_KEYWORDS)

def circuit_allowed(q,policy):
    return not is_circuit_question(q) if policy in ("완전 제외","최대한 제외") else True

def looks_like_bad_anchor(a):
    if not a:return True
    topic=str(a.get("topic","")).strip(); ans=str(a.get("answer","")).strip(); ev=str(a.get("evidence","")).strip(); whole=f"{topic} {ans} {ev}".lower()
    if len(ans)<2 or len(ans)>36 or len(ev)<18:return True
    if any(x.lower() in (topic+" "+ans).lower() for x in BAD_ANSWER_TOKENS):return True
    if any(x.lower() in whole for x in BAD_ANCHOR_TOKENS):return True
    if topic in {"단원","영역","내용","목차","개요","정리","특징","종류","분류"}:return True
    if re.fullmatch(r"\s*[ⅠⅡⅢⅣⅤⅥⅦⅧⅨⅩ0-9]+[.)]?\s*",topic):return True
    sym=sum(1 for ch in ans if not(ch.isalnum() or ch.isspace() or ('가'<=ch<='힣')))
    return sym/max(1,len(ans))>.30

def formula_quality(q,points):
    if points==2:return len(q.get("tasks",[])) in (1,2)
    return len(q.get("tasks",[]))>=2 and "=" in " ".join(q.get("solution",[]))

def concept_quality(q,points,difficulty="적당히 어려움",pattern_type=None):
    passage=str(q.get("passage","")); tasks=q.get("tasks",[]); conds=q.get("conditions",[])
    if not passage or not tasks:return False
    if points==2:
        # 단순 원문 가리기형은 허용하지 않음. 상황/자료가 있어야 함.
        if len(passage)<80:return False
        if len(tasks)>2:return False
    if points==4:
        if len(tasks)<2:return False
        if len(passage)<110 and not conds:return False
    if difficulty=="적당히 어려움" and (len(passage)>1300 or len(tasks)>4 or len(conds)>6):return False
    return True

def target_instruction(difficulty,points,pattern_label="자료 해석"):
    base=("실제 중등 기술 임용 대비의 중간~중상 난이도로 한다. 배운 범위를 벗어난 지엽적 지식으로 어렵게 하지 말고, "
          "제시 자료를 읽고 핵심 개념을 연결하는 사고로 변별한다. ")
    if difficulty=="기본": base="핵심 개념을 정확히 확인하는 수준으로 한다. "
    elif difficulty=="어려움": base="원자료 범위 안에서 복합 적용을 허용하되 근거 없는 심화지식은 금지한다. "
    if points==2:
        base += f"2점이며 문제 유형은 '{pattern_label}'이다. 단순 정의 한 줄 가리기 대신 한 단계의 자료 해석/적용을 요구한다."
    else:
        base += f"4점이며 문제 유형은 '{pattern_label}'이다. 같은 자료를 바탕으로 서로 연결된 두 가지 이상 작성 요구를 두고 최소 두 단계 사고를 요구한다."
    return base
