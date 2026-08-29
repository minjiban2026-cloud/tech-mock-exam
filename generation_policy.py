
CIRCUIT_KEYWORDS=[
 "회로","키르히호프","테브난","노튼","밀만","rlc","rc회로","rl회로",
 "정류","다이오드","트랜지스터","mosfet","op amp","연산증폭기",
 "논리회로","논리 게이트","플립플롭","브리지 정류"
]
def _text(q):
    return " ".join([str(q.get("domain","")),str(q.get("topic","")),str(q.get("intro","")),
      str(q.get("passage",""))," ".join(q.get("conditions",[]))," ".join(q.get("tasks",[])),
      str(q.get("source_basis",""))]).lower()
def is_circuit_question(q):
    t=_text(q)
    return any(k.lower() in t for k in CIRCUIT_KEYWORDS)
def circuit_allowed(q,policy,circuit_count=0,total_count=0):
    if not is_circuit_question(q): return True
    if policy=="완전 제외": return False
    if policy=="최대한 제외":
        # 기본적으로 0개를 지향. 시험 구성이 막힐 때 후반부에 1개까지만 허용.
        return circuit_count<1 and total_count>=8
    return True
def difficulty_score(q):
    tasks=len(q.get("tasks",[])); conds=len(q.get("conditions",[]))
    passage=str(q.get("passage","")); sol=" ".join(q.get("solution",[]))
    s=1
    if tasks>=2 or conds>=1 or len(passage)>=90: s=2
    if tasks>=3 or conds>=3 or len(passage)>=450 or len(sol)>=450: s=3
    if q.get("verifier")=="python" and tasks>=2: s=max(s,2)
    return s
def difficulty_allowed(q,target):
    s=difficulty_score(q)
    if target=="기본": return s<=1
    if target=="적당히 어려움": return s<=2
    return True
def target_instruction(target):
    if target=="기본":
        return "핵심개념 하나를 직접 확인하는 수준으로 만들고 복잡한 추론과 다단계 계산을 피한다."
    if target=="적당히 어려움":
        return ("단순 정의 암기만 묻지 말고 자료·상황에서 핵심개념 1~2개를 연결해 판단하게 한다. "
                "낯선 심화지식, 과도한 다단계 계산, 함정 위주 조건, 세 개 이상의 독립 개념 결합은 피한다. "
                "중등 기술 임용 대비 중간~중상 난이도를 목표로 한다.")
    return "원자료 범위 안에서 복합 적용을 허용하되 근거 없는 심화지식은 요구하지 않는다."
