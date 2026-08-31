PATTERNS = [
 {"id":"T2_11","points":2,"name":"관계형 두 요소","subpoints":[1,1],
  "verbs":["식별/판단","관계/근거"],"calc":False,"weight":8,
  "quality_rule":"두 정답을 독립적으로 찾아 쓰게 하지 말고, 첫 판단이 두 번째 판단의 근거가 되도록 연결한다."},
 {"id":"T2_C11","points":2,"name":"식+계산","subpoints":[1,1],"verbs":["관계식","계산"],"calc":True,"weight":4},
 {"id":"T4_112","points":4,"name":"개념+개념+적용","subpoints":[1,1,2],"verbs":["식별","식별","적용"],"calc":False,"weight":8},
 {"id":"T4_22","points":4,"name":"판단+근거","subpoints":[2,2],"verbs":["판단","서술"],"calc":False,"weight":0},
 {"id":"T4_1111","points":4,"name":"네 요소","subpoints":[1,1,1,1],"verbs":["식별"]*4,"calc":False,"weight":3},
 {"id":"T4_13","points":4,"name":"개념+심화서술","subpoints":[1,3],"verbs":["식별","서술"],"calc":False,"weight":0},
 {"id":"T4_C112","points":4,"name":"원리+중간값+계산","subpoints":[1,1,2],"verbs":["원리/식","중간값","계산"],"calc":True,"weight":10},
]

TYPE2=["관계판단","오류판단","비교/구분","자료해석"]
TYPE4=["개념+적용","비교/구분","과정/순서","오류수정","계산/판단","자료해석","원인/이유","사례/설계"]
MATERIAL_FORMS=["서술자료","대화자료","과정자료","표형자료"]

def weighted_pick(rng,points,calc=False,used=None):
    rows=[p for p in PATTERNS if p["points"]==points and p["calc"]==calc and p.get("weight",1)>0]
    used=set(used or [])
    fresh=[p for p in rows if p["id"] not in used] or rows
    pool=[]
    for p in fresh: pool += [p]*p.get("weight",1)
    return rng.choice(pool)

def blueprint(section,scores,domains,rng):
    domains=list(domains)
    if not domains: raise ValueError("출제 영역이 없습니다.")
    seq=[]
    first_cycle=domains[:]
    rng.shuffle(first_cycle)
    seq.extend(first_cycle)
    while len(seq)<len(scores):
        cycle=domains[:]
        if len(domains)>1 and "발명" in cycle and "발명" in seq:
            cycle=[d for d in cycle if d!="발명"]
        if not cycle: cycle=domains[:]
        rng.shuffle(cycle)
        seq.extend(cycle)
    seq=seq[:len(scores)]

    t2=TYPE2[:]; t4=TYPE4[:]; rng.shuffle(t2); rng.shuffle(t4)
    f=MATERIAL_FORMS[:]; rng.shuffle(f)
    i2=i4=0
    out=[]
    for no,(pts,dom) in enumerate(zip(scores,seq),1):
        typ=t2[i2%len(t2)] if pts==2 else t4[i4%len(t4)]
        if pts==2:i2+=1
        else:i4+=1
        out.append({
            "number":no,"points":pts,"domain":dom,"question_type":typ,
            "material_form":f[(no-1)%len(f)]
        })

    formula_domains={"재료역학","수송기술","통신기술"}
    calc4=[r for r in out if r["points"]==4 and r["domain"] in formula_domains]
    if calc4:
        chosen=rng.choice(calc4)
        chosen["question_type"]="계산/판단"
        chosen["material_form"]="수치자료"
    return out
