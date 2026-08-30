
"""실제 중등 기술 임용 원본(2020~2026)에서 일반화한 문항 구조.
문장/정답은 복제하지 않고 배점·하위요구·자료형태의 뼈대만 사용한다.
"""
from dataclasses import dataclass

PATTERNS = [
    # 2점: 실제 기출은 두 개의 1점 요소를 묻는 형태가 매우 흔함
    {"id":"T2_11","points":2,"name":"두 요소 식별/적용","subpoints":[1,1],
     "verbs":["식별","식별"],"calc":False,"visual":False,"weight":7},
    {"id":"T2_CALC11","points":2,"name":"간단 계산+개념","subpoints":[1,1],
     "verbs":["계산","식별"],"calc":True,"visual":False,"weight":3},

    # 4점: 실제 기출에서 자주 보이는 부분점수 구조
    {"id":"T4_112","points":4,"name":"개념+개념+적용/계산","subpoints":[1,1,2],
     "verbs":["식별","식별","적용"],"calc":False,"visual":False,"weight":7},
    {"id":"T4_CALC112","points":4,"name":"개념+중간값+계산","subpoints":[1,1,2],
     "verbs":["식별","계산","계산"],"calc":True,"visual":False,"weight":6},
    {"id":"T4_22","points":4,"name":"두 개의 연결된 고난도 요구","subpoints":[2,2],
     "verbs":["판단","서술"],"calc":False,"visual":False,"weight":4},
    {"id":"T4_CALC22","points":4,"name":"연쇄 계산 2요소","subpoints":[2,2],
     "verbs":["계산","계산"],"calc":True,"visual":False,"weight":3},
    {"id":"T4_1111","points":4,"name":"네 개의 짧은 채점요소","subpoints":[1,1,1,1],
     "verbs":["식별","식별","식별","식별"],"calc":False,"visual":False,"weight":2},
    {"id":"T4_13","points":4,"name":"개념+근거/비교 서술","subpoints":[1,3],
     "verbs":["식별","서술"],"calc":False,"visual":False,"weight":3},
]

# 실제 기출에서 반복적으로 확인되는 문항 유형. 한 시험에서 같은 유형이 몰리지 않게 사용.
TYPE_CYCLE_2 = ["자료식별","순서/과정","간단계산","오류판단"]
TYPE_CYCLE_4 = [
    "개념+적용","비교/구분","과정/순서","오류수정",
    "계산/판단","자료해석","원인/이유","사례/설계"
]

def candidates(points, calc=False):
    rows=[p for p in PATTERNS if p["points"]==points and p["calc"]==calc]
    return rows

def weighted_pick(rng, points, calc=False, used_pattern_ids=None):
    used=set(used_pattern_ids or [])
    rows=candidates(points,calc)
    # 다양성을 위해 아직 안 쓴 구조를 우선
    fresh=[p for p in rows if p["id"] not in used]
    pool=fresh or rows
    weighted=[]
    for p in pool:
        weighted.extend([p]*max(1,int(p.get("weight",1))))
    return rng.choice(weighted)

def blueprint(section, scores, domains, rng):
    """A/B 전체를 먼저 설계. 영역·유형을 가능한 한 균등하게 배치한다."""
    domains=list(domains)
    if not domains:
        raise ValueError("출제 영역이 없습니다.")
    # 영역 순환
    domain_seq=[]
    while len(domain_seq)<len(scores):
        cycle=domains[:]
        rng.shuffle(cycle)
        domain_seq.extend(cycle)
    domain_seq=domain_seq[:len(scores)]

    types2=TYPE_CYCLE_2[:]
    types4=TYPE_CYCLE_4[:]
    rng.shuffle(types2); rng.shuffle(types4)
    i2=i4=0
    rows=[]
    for no,(pts,dom) in enumerate(zip(scores,domain_seq),1):
        if pts==2:
            typ=types2[i2%len(types2)]; i2+=1
        else:
            typ=types4[i4%len(types4)]; i4+=1
        rows.append({"number":no,"points":pts,"domain":dom,"question_type":typ})
    return rows
