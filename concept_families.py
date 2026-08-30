
import re

# topic/answer의 표현이 달라도 같은 원리·공식이면 같은 family로 묶는다.
RULES = [
 ("spring_system", r"스프링|탄성계"),
 ("euler_buckling", r"오일러|좌굴"),
 ("axial_stress_deformation", r"수직응력|축방향|훅의 법칙|변형량"),
 ("thin_cylinder", r"얇은 원통|내압|원주응력"),
 ("fluid_pascal", r"파스칼|유압"),
 ("fluid_continuity", r"연속방정식|질량 보존"),
 ("hydrostatic_pressure", r"정수압|ρgh"),
 ("thermal_efficiency", r"열효율|열기관"),
 ("refrigeration_cop", r"냉동기|성능계수|COP|역카르노"),
 ("ipv4_fragment", r"단편화|fragment offset|MTU"),
 ("ipv4_subnet", r"서브넷|CIDR|호스트 주소"),
 ("vpn_network_separation", r"VPN|망 분리|가상사설망"),
 ("design_thinking_problem_solving", r"디자인 씽킹|기술적 문제해결|문제 해결"),
 ("cooperative_learning", r"Jigsaw|STAD|TGT|협동학습|과제분담"),
 ("invention_thinking", r"브레인스토밍|브레인라이팅|SCAMPER|기회의 원|ALU|하이라이팅|강제결합|사고 기법"),
 ("patent_ip", r"특허|지식 재산|지식재산"),
 ("drawing_scale_projection", r"척도|투상|정면도|평면도|도면"),
 ("machining_forming", r"압출|압연|단조|드로잉|소성|가공"),
 ("construction_foundation", r"기초|지반|보링|표준관입"),
 ("construction_finish", r"외벽|미장|마감|커튼월"),
 ("bio_culture_fermentation", r"배양|발효|미생물"),
 ("network_model", r"TCP/IP|OSI|계층"),
]

def norm(s):
    return re.sub(r"\s+"," ",str(s)).strip().lower()

def concept_family(q_or_topic):
    if isinstance(q_or_topic,dict):
        text=" ".join([
            str(q_or_topic.get("topic","")),
            " ".join(map(str,q_or_topic.get("answer",[]))),
            str(q_or_topic.get("source_basis",""))
        ])
    else:
        text=str(q_or_topic)
    t=norm(text)
    for fam,pat in RULES:
        if re.search(pat,t,re.I):
            return fam
    # 모르는 경우 너무 넓게 막지 않도록 주제 자체를 정규화
    topic=t.split("·")[0].strip()
    topic=re.sub(r"[^0-9a-z가-힣]+","_",topic)
    return "topic_"+topic[:50]


def families_for(q):
    """중복 판정용 family.
    - 알려진 핵심 원리는 topic/answer/source_basis 전체에서 탐지
    - fallback family는 topic 조각에만 부여
    - 같은 PDF 파일명만 같다는 이유로 중복 처리하지 않음
    """
    if isinstance(q,dict):
        topic_parts=[x.strip() for x in str(q.get("topic","")).split("·") if x.strip()]
        combined=" ".join(topic_parts + [str(x) for x in q.get("answer",[])] + [str(q.get("source_basis",""))])
    else:
        topic_parts=[str(q)]
        combined=str(q)

    out=set()
    # known conceptual families
    for fam,pat in RULES:
        if re.search(pat,norm(combined),re.I):
            out.add(fam)

    # fallback only from actual topic labels
    for text in topic_parts:
        tt=norm(text)
        if any(re.search(pat,tt,re.I) for _,pat in RULES):
            continue
        k=re.sub(r"[^0-9a-z가-힣]+","_",tt)[:50]
        if k:
            out.add("topic_"+k)
    return out

