
import re

CIRCUIT_KEYWORDS=[
    "회로","키르히호프","테브난","노튼","밀만","rlc","rc회로","rl회로",
    "정류","다이오드","트랜지스터","mosfet","op amp","연산증폭기",
    "논리회로","논리 게이트","플립플롭","브리지 정류"
]
BAD_ANCHOR_TOKENS=[
    "아이디어의구상","발명기법과실제","정보통신기술",
    "서브노트","교과서 정리","part1","part2","part3"
]

def _all_text(q):
    return " ".join([
        str(q.get("domain","")), str(q.get("topic","")), str(q.get("intro","")),
        str(q.get("passage","")), " ".join(q.get("conditions",[])),
        " ".join(q.get("tasks",[])), str(q.get("source_basis",""))
    ]).lower()

def is_circuit_question(q):
    t=_all_text(q)
    return any(k.lower() in t for k in CIRCUIT_KEYWORDS)

def circuit_allowed(q, policy):
    if not is_circuit_question(q):
        return True
    if policy in ("완전 제외","최대한 제외"):
        return False
    return True

def looks_like_bad_anchor(anchor):
    if not anchor:
        return True
    topic=str(anchor.get("topic","")).strip()
    ans=str(anchor.get("answer","")).strip()
    ev=str(anchor.get("evidence","")).strip()
    whole=f"{topic} {ans} {ev}".lower()

    if len(ans)<2 or len(ans)>45:
        return True
    if any(tok.lower() in whole for tok in BAD_ANCHOR_TOKENS):
        return True
    if topic in {"단원","영역","내용","목차","개요","정리","특징","종류","분류"}:
        return True
    if re.match(r"^\s*[ⅠⅡⅢⅣⅤⅥⅦⅧⅨⅩ0-9]+[.)]?\s*$", topic):
        return True

    # 목차 조각/기호 덩어리 제거
    sym=sum(1 for ch in ans if not(ch.isalnum() or ch.isspace() or ('가'<=ch<='힣')))
    if sym/max(1,len(ans))>0.35:
        return True
    return False

def formula_quality(q, points):
    """4점 계산형을 단순 1회 대입 문제로 쓰지 않는다."""
    if points!=4:
        return True
    tasks=q.get("tasks",[])
    solution=" ".join(q.get("solution",[]))
    # 두 하위 요구 + 풀이식이 적어도 2개
    eq_count=solution.count("=")
    return len(tasks)>=2 and eq_count>=2

def concept_quality(q, points, difficulty):
    tasks=q.get("tasks",[])
    passage=str(q.get("passage",""))
    conds=q.get("conditions",[])

    # 너무 짧은 빈칸/용어맞히기형 억제
    if points==4:
        if len(tasks)<2:
            return False
        if len(passage)<100 and len(conds)==0:
            return False
        joined=" ".join(tasks)
        if ("용어" in joined or "명칭" in joined) and len(tasks)==1:
            return False
    if difficulty=="적당히 어려움":
        # 지나치게 긴/복합적인 문제는 제외
        if len(passage)>1100 or len(tasks)>3 or len(conds)>5:
            return False
    return True

def target_instruction(difficulty, points):
    if difficulty=="기본":
        base="핵심 개념을 직접 확인하는 수준으로 구성한다."
    elif difficulty=="적당히 어려움":
        base=("실제 중등 기술 임용 대비의 중간~중상 난이도로 구성한다. "
              "단순 정의 암기보다 자료·상황을 해석해 핵심 개념 1~2개를 연결하게 한다. "
              "지엽적인 심화지식, 함정 위주 조건, 과도하게 긴 계산은 피한다.")
    else:
        base="원자료 범위 안에서 복합 적용과 높은 변별력을 허용한다."

    if points==4:
        base += (
            " 4점은 반드시 두 단계 이상의 사고를 요구한다. "
            "예: 자료 해석→개념 판별→근거 제시, 또는 원리 선택→중간값 계산→최종값 계산. "
            "단순 공식 1회 대입, 목차/단원 제목 맞히기, 원문 한 줄을 가린 빈칸 문제는 금지한다."
        )
    else:
        base += " 2점은 핵심 개념 확인 또는 한 단계 적용 정도로 한다."
    return base
