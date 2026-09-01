PATTERNS = [
 {"id":"T2_REL","points":2,"name":"원인·조건→결과 관계형","subpoints":[1,1],
  "verbs":["판단","근거/관계"],"calc":False,"weight":5,
  "quality_rule":"첫 판단이 두 번째 답의 근거가 되게 한다. 명칭 두 개 독립 회상 금지."},
 {"id":"T2_ERR","points":2,"name":"오류판단→수정형","subpoints":[1,1],
  "verbs":["오류판단","수정"],"calc":False,"weight":4,
  "quality_rule":"자료의 잘못된 적용/설명을 판단한 뒤 고정정답에 근거해 수정하게 한다."},
 {"id":"T2_CMP","points":2,"name":"비교→구분근거형","subpoints":[1,1],
  "verbs":["비교판단","구분근거"],"calc":False,"weight":4,
  "quality_rule":"두 요소의 차이를 자료에서 판단하고, 그 판단 근거를 연결해 쓰게 한다."},
 {"id":"T2_DATA","points":2,"name":"자료해석→적용판단형","subpoints":[1,1],
  "verbs":["자료해석","적용판단"],"calc":False,"weight":4,
  "quality_rule":"자료에서 먼저 특성/조건을 해석한 뒤 그 결과를 다음 판단에 사용한다."},
 {"id":"T2_11","points":2,"name":"구형 식별형(비사용)","subpoints":[1,1],
  "verbs":["식별","식별"],"calc":False,"weight":0},
 {"id":"T2_C11","points":2,"name":"식+계산","subpoints":[1,1],"verbs":["관계식","계산"],"calc":True,"weight":4},

 {"id":"T4_DATA112","points":4,"name":"자료해석→개념판단→적용","subpoints":[1,1,2],
  "verbs":["자료해석","개념판단","적용"],"calc":False,"weight":9,
  "quality_rule":"앞선 자료 해석과 개념 판단을 마지막 적용 판단의 근거로 사용하게 한다."},
 {"id":"T4_ERR22","points":4,"name":"오류판단→수정·근거","subpoints":[2,2],
  "verbs":["오류판단","수정/근거"],"calc":False,"weight":7,
  "quality_rule":"자료의 오류를 판단하고, 같은 원리의 근거를 사용해 수정하게 한다. 두 요구는 독립 회상이 아니어야 한다."},
 {"id":"T4_112","points":4,"name":"개념+개념+적용","subpoints":[1,1,2],"verbs":["식별","식별","적용"],"calc":False,"weight":6,
  "quality_rule":"두 개념을 독립적으로 나열하지 말고 마지막 적용 판단에서 앞의 두 개념을 함께 사용하게 한다."},
 {"id":"T4_22","points":4,"name":"판단+근거","subpoints":[2,2],"verbs":["판단","서술"],"calc":False,"weight":0},
 {"id":"T4_1111","points":4,"name":"네 요소","subpoints":[1,1,1,1],"verbs":["식별"]*4,"calc":False,"weight":0},
 {"id":"T4_13","points":4,"name":"개념+심화서술","subpoints":[1,3],"verbs":["식별","서술"],"calc":False,"weight":0},
 {"id":"T4_C112","points":4,"name":"원리+중간값+계산","subpoints":[1,1,2],"verbs":["원리/식","중간값","계산"],"calc":True,"weight":10},
]

TYPE2=["관계판단","오류수정","비교/구분","자료해석"]
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
    first_cycle=domains[:]; rng.shuffle(first_cycle); seq.extend(first_cycle)
    while len(seq)<len(scores):
        cycle=domains[:]
        if len(domains)>1 and "발명" in cycle and "발명" in seq:
            cycle=[d for d in cycle if d!="발명"]
        if not cycle: cycle=domains[:]
        rng.shuffle(cycle); seq.extend(cycle)
    seq=seq[:len(scores)]
    t2=TYPE2[:]; t4=TYPE4[:]; rng.shuffle(t2); rng.shuffle(t4)
    f=MATERIAL_FORMS[:]; rng.shuffle(f)
    i2=i4=0; out=[]
    for no,(pts,dom) in enumerate(zip(scores,seq),1):
        typ=t2[i2%len(t2)] if pts==2 else t4[i4%len(t4)]
        if pts==2:i2+=1
        else:i4+=1
        out.append({"number":no,"points":pts,"domain":dom,"question_type":typ,
                    "material_form":f[(no-1)%len(f)]})
    formula_domains={"재료역학","수송기술","통신기술"}
    calc4=[r for r in out if r["points"]==4 and r["domain"] in formula_domains]
    if calc4:
        chosen=rng.choice(calc4); chosen["question_type"]="계산/판단"; chosen["material_form"]="수치자료"
    return out
