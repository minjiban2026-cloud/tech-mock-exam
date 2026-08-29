
import math, random, hashlib, json

def _fp(obj):
    raw=json.dumps(obj,ensure_ascii=False,sort_keys=True)
    return hashlib.sha256(raw.encode()).hexdigest()[:20]

def material_stress(rng):
    A=rng.choice([100,125,200,250,400])
    sigma=rng.choice([40,60,80,100,120])
    P=A*sigma/1000  # kN if MPa=N/mm2
    q={
      "domain":"재료역학","topic":"수직응력","points":2,"verifier":"python",
      "intro":"다음은 축방향 하중을 받는 부재에 관한 자료이다.",
      "passage":f"단면적이 {A} mm²인 균일한 봉에 축방향 인장하중 {P:g} kN이 작용한다.",
      "conditions":["하중은 단면의 도심을 지나며 재료는 탄성범위에 있다."],
      "tasks":["봉에 발생하는 수직응력[MPa]을 풀이 과정과 함께 구할 것."],
      "answer":[f"{sigma:g} MPa"],
      "solution":[f"σ=P/A=({P:g}×1000)/{A}={sigma:g} MPa"],
      "source_basis":"재료역학 서브노트의 수직응력 σ=P/A"
    }
    q["fingerprint"]=_fp({k:q[k] for k in ["domain","topic","passage","answer"]})
    return q

def material_hooke(rng):
    A=rng.choice([100,200,250])
    L=rng.choice([100,200,300])
    E=rng.choice([100000,200000]) # MPa
    P=rng.choice([10,20,25]) * 1000
    delta=P*L/(A*E)
    q={
      "domain":"재료역학","topic":"훅의 법칙과 변형량","points":4,"verifier":"python",
      "intro":"다음은 탄성영역에서 축방향 하중을 받는 봉에 관한 자료이다.",
      "passage":f"길이 {L} mm, 단면적 {A} mm², 탄성계수 {E/1000:g} GPa인 봉에 {P/1000:g} kN의 인장하중이 작용한다.",
      "conditions":["σ=Eε의 관계가 성립하는 범위로 가정한다."],
      "tasks":["봉의 수직응력[MPa]을 구할 것.","봉의 길이 방향 변형량[mm]을 풀이 과정과 함께 구할 것."],
      "answer":[f"{P/A:g} MPa",f"{delta:.4g} mm"],
      "solution":[f"σ=P/A={P}/{A}={P/A:g} MPa",f"δ=PL/(AE)={P}×{L}/({A}×{E})={delta:.4g} mm"],
      "source_basis":"재료역학 서브노트의 훅의 법칙과 δ=PL/AE"
    }
    q["fingerprint"]=_fp({k:q[k] for k in ["domain","topic","passage","answer"]})
    return q

def material_cylinder(rng):
    D=rng.choice([200,300,400,600])
    p=rng.choice([1,2,4])
    allow=rng.choice([40,50,80,100])
    t=p*D/(2*allow)
    q={
      "domain":"재료역학","topic":"얇은 원통 내압","points":4,"verifier":"python",
      "intro":"다음은 내압을 받는 얇은 원통 용기에 관한 자료이다.",
      "passage":f"안지름이 {D} mm인 얇은 원통에 {p} N/mm²의 내압이 작용한다. 재료의 허용 원주응력은 {allow} N/mm²이다.",
      "conditions":["원주방향 응력을 기준으로 설계한다.","얇은 원통으로 가정한다."],
      "tasks":["원주방향 응력식을 쓸 것.","필요한 최소 두께[mm]를 구할 것."],
      "answer":["σₕ=pD/(2t)",f"{t:g} mm"],
      "solution":[f"σₕ=pD/(2t)",f"t=pD/(2σₕ)={p}×{D}/(2×{allow})={t:g} mm"],
      "source_basis":"재료역학 서브노트의 원통 내압 원주방향 응력"
    }
    q["fingerprint"]=_fp({k:q[k] for k in ["domain","topic","passage","answer"]})
    return q

def material_spring(rng):
    k1=rng.choice([10,20,30])
    k2=rng.choice([20,30,40])
    mode=rng.choice(["직렬","병렬"])
    if mode=="직렬":
        k=k1*k2/(k1+k2)
    else:
        k=k1+k2
    P=rng.choice([100,120,180,200])
    delta=P/k
    q={
      "domain":"재료역학","topic":f"스프링 {mode} 연결","points":4,"verifier":"python",
      "intro":"다음은 두 스프링으로 구성된 탄성계에 관한 자료이다.",
      "passage":f"스프링 상수 k₁={k1} N/mm, k₂={k2} N/mm인 두 스프링을 {mode}로 연결하고 전체에 {P} N의 하중을 가하였다.",
      "conditions":["스프링은 선형 탄성범위에서 작동한다."],
      "tasks":["등가 스프링 상수[N/mm]를 구할 것.","전체 변형량[mm]을 구할 것."],
      "answer":[f"{k:.4g} N/mm",f"{delta:.4g} mm"],
      "solution":[("1/k=1/k₁+1/k₂" if mode=="직렬" else "k=k₁+k₂")+f" → k={k:.4g} N/mm",f"δ=P/k={P}/{k:.4g}={delta:.4g} mm"],
      "source_basis":"재료역학 서브노트의 스프링 직렬·병렬 합성"
    }
    q["fingerprint"]=_fp({k:q[k] for k in ["domain","topic","passage","answer"]})
    return q

def euler_buckling(rng):
    E=200000 # N/mm2
    I=rng.choice([1e6,2e6,4e6])
    L=rng.choice([2000,3000,4000])
    K=rng.choice([0.5,0.7,1.0,2.0])
    Pcr=(math.pi**2*E*I)/(K*L)**2/1000 # kN
    q={
      "domain":"재료역학","topic":"오일러 좌굴","points":4,"verifier":"python",
      "intro":"다음은 세장한 기둥의 좌굴에 관한 자료이다.",
      "passage":f"탄성계수 E=200 GPa, 단면 2차 모멘트 I={I/1e6:g}×10⁶ mm⁴, 길이 L={L/1000:g} m인 기둥의 유효길이계수 K는 {K:g}이다.",
      "conditions":["오일러 좌굴식을 적용한다.","π는 계산기의 값을 사용한다."],
      "tasks":["유효길이 KL[mm]을 구할 것.","임계 좌굴하중[kN]을 구할 것."],
      "answer":[f"{K*L:g} mm",f"{Pcr:.3g} kN"],
      "solution":[f"KL={K:g}×{L}={K*L:g} mm",f"Pcr=π²EI/(KL)²={Pcr:.3g} kN"],
      "source_basis":"재료역학 서브노트의 오일러 방정식"
    }
    q["fingerprint"]=_fp({k:q[k] for k in ["domain","topic","passage","answer"]})
    return q

def fluid_hydrostatic(rng):
    rho=1000
    g=9.8
    h=rng.choice([2,3,5,10])
    p=rho*g*h
    q={
      "domain":"수송기술","topic":"정수압","points":2,"verifier":"python",
      "intro":"다음은 정지한 물속 한 지점의 압력에 관한 자료이다.",
      "passage":f"물의 밀도는 {rho} kg/m³, 중력가속도는 {g} m/s²이다. 수면 아래 {h} m 지점의 계기압력을 구하려 한다.",
      "conditions":["대기압은 제외한다."],
      "tasks":["정수압[Pa]을 풀이 과정과 함께 구할 것."],
      "answer":[f"{p:g} Pa"],
      "solution":[f"p=ρgh={rho}×{g}×{h}={p:g} Pa"],
      "source_basis":"유체역학 서브노트의 정수압 p=ρgh"
    }
    q["fingerprint"]=_fp({k:q[k] for k in ["domain","topic","passage","answer"]})
    return q

def fluid_pascal(rng):
    d1=rng.choice([10,20])
    d2=rng.choice([30,40,60])
    F1=rng.choice([10,20,50])
    F2=F1*(d2/d1)**2
    q={
      "domain":"수송기술","topic":"파스칼의 원리","points":4,"verifier":"python",
      "intro":"다음은 서로 다른 크기의 피스톤을 가진 유압장치에 관한 자료이다.",
      "passage":f"작은 피스톤의 지름은 {d1} cm, 큰 피스톤의 지름은 {d2} cm이다. 작은 피스톤에 {F1} N의 힘을 가한다.",
      "conditions":["유체 손실과 피스톤 마찰은 무시한다."],
      "tasks":["두 피스톤에서 동일하게 전달되는 물리량의 명칭을 쓸 것.","큰 피스톤에서 얻는 힘[N]을 구할 것."],
      "answer":["압력",f"{F2:g} N"],
      "solution":["밀폐된 유체에 가한 압력은 모든 방향으로 동일하게 전달된다.",f"F₂=F₁(A₂/A₁)=F₁(d₂/d₁)²={F2:g} N"],
      "source_basis":"유체역학 서브노트의 파스칼 원리"
    }
    q["fingerprint"]=_fp({k:q[k] for k in ["domain","topic","passage","answer"]})
    return q

def fluid_continuity(rng):
    d1=rng.choice([40,60,80])
    d2=rng.choice([20,30,40])
    v1=rng.choice([1,2,3])
    v2=v1*(d1/d2)**2
    q={
      "domain":"수송기술","topic":"연속방정식","points":4,"verifier":"python",
      "intro":"다음은 정상상태로 흐르는 비압축성 유체의 관로에 관한 자료이다.",
      "passage":f"원형 관의 지름이 {d1} mm에서 {d2} mm로 감소한다. 지름 {d1} mm 구간의 평균 유속은 {v1} m/s이다.",
      "conditions":["밀도는 일정하고 누설은 없다."],
      "tasks":["적용되는 보존 법칙을 쓸 것.","지름 {d2} mm 구간의 평균 유속[m/s]을 구할 것."],
      "answer":["질량 보존 법칙",f"{v2:g} m/s"],
      "solution":["비압축성 유체에서 A₁V₁=A₂V₂이다.",f"V₂=V₁(d₁/d₂)²={v1}×({d1}/{d2})²={v2:g} m/s"],
      "source_basis":"유체역학 서브노트의 연속방정식"
    }
    q["fingerprint"]=_fp({k:q[k] for k in ["domain","topic","passage","answer"]})
    return q

def thermo_efficiency(rng):
    QH=rng.choice([500,600,800,1000])
    QL=rng.choice([int(QH*0.5),int(QH*0.6),int(QH*0.8)])
    eta=(QH-QL)/QH*100
    q={
      "domain":"수송기술","topic":"열기관 열효율","points":2,"verifier":"python",
      "intro":"다음은 한 열기관의 에너지 출입에 관한 자료이다.",
      "passage":f"고열원에서 {QH} kJ의 열을 받아 저열원으로 {QL} kJ를 방출한다.",
      "conditions":[],
      "tasks":["열기관의 열효율[%]을 구할 것."],
      "answer":[f"{eta:g}%"],
      "solution":[f"η=(QH-QL)/QH×100=({QH}-{QL})/{QH}×100={eta:g}%"],
      "source_basis":"열역학 서브노트의 열기관 열효율"
    }
    q["fingerprint"]=_fp({k:q[k] for k in ["domain","topic","passage","answer"]})
    return q

def refrigerator_cop(rng):
    TL=rng.choice([200,240,250])
    TH=rng.choice([300,320,350])
    cop=TL/(TH-TL)
    q={
      "domain":"수송기술","topic":"역카르노 냉동기 COP","points":4,"verifier":"python",
      "intro":"다음은 역카르노 사이클로 작동하는 냉동기에 관한 자료이다.",
      "passage":f"저온 열원의 절대온도는 {TL} K, 고온 열원의 절대온도는 {TH} K이다.",
      "conditions":["이상적인 역카르노 냉동기로 가정한다."],
      "tasks":["냉동기의 성능계수 식을 온도로 나타낼 것.","성능계수를 구할 것."],
      "answer":["COP=TL/(TH-TL)",f"{cop:.4g}"],
      "solution":["역카르노 냉동기의 COP=TL/(TH-TL)",f"{TL}/({TH}-{TL})={cop:.4g}"],
      "source_basis":"열역학 서브노트의 냉동기 성능계수"
    }
    q["fingerprint"]=_fp({k:q[k] for k in ["domain","topic","passage","answer"]})
    return q

def ipv4_fragment(rng):
    total=rng.choice([4000,5000,6000,7000])
    mtu=rng.choice([1500,1000])
    header=20
    payload=total-header
    max_payload=((mtu-header)//8)*8
    n=math.ceil(payload/max_payload)
    last_payload=payload-max_payload*(n-1)
    last_total=last_payload+header
    second_offset=max_payload//8 if n>1 else 0
    q={
      "domain":"통신기술","topic":"IPv4 단편화","points":4,"verifier":"python",
      "intro":"다음은 IPv4 데이터그램이 MTU가 더 작은 네트워크를 통과하는 상황이다.",
      "passage":f"IPv4 데이터그램의 전체 길이는 {total} byte이고 헤더는 {header} byte이다. 통과해야 하는 네트워크의 MTU는 {mtu} byte이다.",
      "conditions":["각 단편에 IPv4 헤더 20 byte가 붙는다.","마지막 단편을 제외한 데이터 부분은 8 byte의 배수로 한다."],
      "tasks":["생성되는 단편의 개수를 구할 것.","두 번째 단편의 Fragment Offset 값을 구할 것.","마지막 단편의 전체 길이[byte]를 구할 것."],
      "answer":[f"{n}개",f"{second_offset}",f"{last_total} byte"],
      "solution":[f"단편당 최대 데이터={max_payload} byte, 원 데이터={payload} byte → {n}개",f"두 번째 offset={max_payload}/8={second_offset}",f"마지막 데이터={last_payload} byte, 전체={last_payload}+20={last_total} byte"],
      "source_basis":"통신기술 기출/서브노트의 IPv4 단편화"
    }
    q["fingerprint"]=_fp({k:q[k] for k in ["domain","topic","passage","answer"]})
    return q

def subnet_hosts(rng):
    prefix=rng.choice([24,25,26,27,28])
    hostbits=32-prefix
    hosts=2**hostbits-2
    q={
      "domain":"통신기술","topic":"IPv4 서브넷 호스트 수","points":2,"verifier":"python",
      "intro":"다음은 IPv4 CIDR 표기와 관련된 자료이다.",
      "passage":f"한 서브넷의 프리픽스 길이가 /{prefix}이다.",
      "conditions":["네트워크 주소와 브로드캐스트 주소는 일반 호스트에 할당하지 않는다."],
      "tasks":["호스트 부분의 비트 수를 쓸 것.","할당 가능한 호스트 주소의 수를 구할 것."],
      "answer":[f"{hostbits}비트",f"{hosts}개"],
      "solution":[f"호스트 비트=32-{prefix}={hostbits}",f"호스트 수=2^{hostbits}-2={hosts}"],
      "source_basis":"정보통신 서브노트의 IPv4 주소/서브넷"
    }
    q["fingerprint"]=_fp({k:q[k] for k in ["domain","topic","passage","answer"]})
    return q

TEMPLATES=[
    material_stress, material_hooke, material_cylinder, material_spring, euler_buckling,
    fluid_hydrostatic, fluid_pascal, fluid_continuity, thermo_efficiency, refrigerator_cop,
    ipv4_fragment, subnet_hosts
]

def generate_formula_question(domain, rng=None):
    rng=rng or random.Random()
    candidates=[]
    for f in TEMPLATES:
        for _ in range(2):
            q=f(random.Random(rng.random()))
            if q["domain"]==domain:
                candidates.append(f)
                break
    if not candidates:
        return None
    return rng.choice(candidates)(rng)
