
import random, math, re
from formula_templates import generate_formula_question
from retrieval import related_bundle,bundle_context,official_style_profile
from ai_wrapper import rewrite_bundle,safe_bundle_question
from validators import validate_formula_question,validate_grounded_question,too_similar,validate_exam,fingerprint
from patterns import blueprint,weighted_pick
from concept_families import families_for

DOMAINS=["기술교육론","발명","제조기술","건설기술","생명기술","전기·전자","통신기술","재료역학","수송기술"]
FORMULA_DOMAINS={"재료역학","수송기술","통신기술"}

def score_pattern(section,count,points):
    defaults={"A":[2,2,2,2]+[4]*8,"B":[2,2]+[4]*9}
    p=defaults.get(section,[])
    if len(p)==count and sum(p)==points:return p[:]
    raise ValueError("실제 A/B 기본 배점 구조만 지원합니다.")

def _prepend_formula_element(q, task, answer, solution):
    q["tasks"]=[task]+list(q.get("tasks",[]))
    q["answer"]=[answer]+list(q.get("answer",[]))
    q["solution"]=[solution]+list(q.get("solution",[]))
    return q

def _enrich_formula(q,pts):
    """단순 공식 대입 4점 금지. 계산형은 개념/관계→중간값→최종 계산의 3요소로 만든다."""
    topic=q.get("topic","")
    q["material_form"]="수치자료"
    q["question_type"]="계산/판단" if pts==4 else "간단계산"
    q["concept_family"]=next(iter(families_for(q)), "formula_"+re.sub(r"\W+","_",topic))

    if pts==2:
        # 2점도 가능한 한 관계식/개념 + 결과의 1+1
        if len(q.get("answer",[]))==1:
            if "수직응력" in topic:
                q=_prepend_formula_element(q,"적용되는 수직응력의 관계식을 쓸 것.","σ=P/A","σ=P/A")
            elif "정수압" in topic:
                q=_prepend_formula_element(q,"정수압의 관계식을 쓸 것.","p=ρgh","p=ρgh")
            elif "열효율" in topic:
                q=_prepend_formula_element(q,"열기관 열효율의 관계식을 쓸 것.","η=(QH-QL)/QH×100","η=(QH-QL)/QH×100")
        if len(q["answer"])>=2:
            q["answer"]=q["answer"][:2];q["tasks"]=q["tasks"][:2];q["solution"]=q["solution"][:2]
            q["subpoints"]=[1,1]
        else:
            q["subpoints"]=[2]
    else:
        # 4점 계산형은 최소 3개 채점요소로 강제
        if len(q.get("answer",[]))==2:
            if "스프링" in topic:
                mode="직렬" if "직렬" in topic else "병렬"
                rel="1/k=1/k₁+1/k₂" if mode=="직렬" else "k=k₁+k₂"
                q=_prepend_formula_element(q,f"{mode} 연결의 등가 스프링 상수 관계식을 쓸 것.",rel,rel)
            elif "오일러" in topic or "좌굴" in topic:
                q=_prepend_formula_element(q,"오일러 좌굴식에 사용하는 유효길이의 관계식을 쓸 것.","Le=KL","유효길이 Le=KL")
            elif "얇은 원통" in topic:
                q=_prepend_formula_element(q,"얇은 원통에서 설계를 지배하는 원주방향 응력식을 쓸 것.","σh=pD/(2t)","σh=pD/(2t)")
            elif "파스칼" in topic:
                q=_prepend_formula_element(q,"파스칼의 원리에 따른 힘과 면적의 관계를 쓸 것.","F1/A1=F2/A2","F1/A1=F2/A2")
            elif "연속방정식" in topic:
                q=_prepend_formula_element(q,"비압축성 정상유동의 연속방정식을 쓸 것.","A1V1=A2V2","A1V1=A2V2")
            elif "냉동기" in topic or "COP" in topic:
                q=_prepend_formula_element(q,"역카르노 냉동기의 성능계수 관계식을 쓸 것.","COP=TL/(TH-TL)","COP=TL/(TH-TL)")
            else:
                return None
        if len(q.get("answer",[]))<3:
            return None
        q["answer"]=q["answer"][:3];q["tasks"]=q["tasks"][:3];q["solution"]=q["solution"][:3]
        q["subpoints"]=[1,1,2]
    q["points"]=pts
    q["pattern_id"]="T4_C112" if pts==4 else "T2_C11"
    q["fingerprint"]=fingerprint(q)
    return q

def make_section(db_path,section,count,points,domains=None,api_key="",model="gpt-5.6-terra",
                 ai_enabled=True,seed=None,previous_questions=None,shared_answers=None):
    rng=random.Random(seed)
    domains=list(domains or DOMAINS)
    scores=score_pattern(section,count,points)
    plan=blueprint(section,scores,domains,rng)
    style=official_style_profile(db_path)

    prior=list(previous_questions or [])
    qs=[]
    used_topics=set()
    used_answers=set(shared_answers or [])
    used_patterns=[]
    ai_calls=0; fallbacks=0; formula_used=0
    # 실제 기출의 혼합성을 위해 계산형은 섹션당 최대 2개.
    formula_cap=2

    for slot in plan:
        pts=slot["points"]; dom=slot["domain"]; q=None
        wants_calc=slot["question_type"] in {"간단계산","계산/판단"}

        # 계산 지원 영역이 아니면 자료형 문제로 자동 전환
        if wants_calc and dom not in FORMULA_DOMAINS:
            slot["question_type"]="자료해석" if pts==4 else "자료식별"
            wants_calc=False

        # ---------- deterministic calculation ----------
        if wants_calc and dom in FORMULA_DOMAINS and formula_used<formula_cap:
            for _ in range(30):
                cand=generate_formula_question(dom,rng)
                if not cand:break
                cand=_enrich_formula(cand,pts)
                if not cand:continue
                if too_similar(cand,prior+qs):continue
                if not validate_formula_question(cand):
                    q=cand;formula_used+=1;break

        # ---------- source-locked concept question ----------
        if q is None:
            pat=weighted_pick(rng,pts,calc=False,used=used_patterns)
            need=len(pat["subpoints"])
            # 영역 균형을 지키기 위해 다른 영역으로 무작정 fallback하지 않는다.
            for _ in range(18):
                bundle=related_bundle(db_path,dom,need,used_answers,used_topics)
                if len(bundle)<need:break
                ctx=bundle_context(db_path,bundle)
                cand=None
                if ai_enabled and api_key:
                    try:
                        ai_calls+=1
                        cand=rewrite_bundle(
                            api_key,model,bundle,pts,section,pat,
                            slot["question_type"],slot["material_form"],style
                        )
                    except Exception:
                        cand=None

                if cand is not None:
                    errs=validate_grounded_question(cand,ctx)
                    if errs or too_similar(cand,prior+qs):
                        cand=None

                if cand is None:
                    cand=safe_bundle_question(
                        bundle,pts,pat,slot["question_type"],slot["material_form"]
                    )
                    errs=validate_grounded_question(cand,ctx)
                    if errs or too_similar(cand,prior+qs):
                        # 이 bundle은 다음 시도에서 제외
                        for a in bundle:used_topics.add(a["topic"])
                        continue
                    fallbacks+=1

                q=cand
                used_answers.update(q["answer"])
                for a in bundle:used_topics.add(a["topic"])
                break

        if q is None:
            raise RuntimeError(
                f"{section} {slot['number']}번({dom})을 엄격 기준으로 만들 수 없습니다. "
                "다른 랜덤 시드를 사용하거나 선택 영역을 넓혀 주세요."
            )

        q["number"]=slot["number"]
        q["blueprint_domain"]=dom
        q["concept_families"]=sorted(families_for(q))
        q["fingerprint"]=q.get("fingerprint") or fingerprint(q)
        qs.append(q);used_patterns.append(q.get("pattern_id"))

    errs=validate_exam(qs,count,points)
    if errs:raise RuntimeError("시험 자동검증 실패: "+" / ".join(errs))

    return {
      "exam_title":f"기술 임용 모의고사 전공 {section}",
      "section":section,"total_points":points,"questions":qs,"verified":True,
      "blueprint":plan,
      "generation_stats":{"ai_calls":ai_calls,"safe_fallbacks":fallbacks,"formula_questions":formula_used},
      "used_answers":list(used_answers),
      "verification_note":"부분점수·복잡도·원문전제잠금·concept-family·영역균형·A/B중복 자동검증 통과"
    }

def make_ab(db_path,a_count=12,a_points=40,b_count=11,b_points=40,domains=None,
            api_key="",model="gpt-5.6-terra",ai_enabled=True,seed=None):
    A=make_section(db_path,"A",a_count,a_points,domains,api_key,model,ai_enabled,seed=seed)
    B=make_section(
        db_path,"B",b_count,b_points,domains,api_key,model,ai_enabled,
        seed=None if seed is None else seed+1,
        previous_questions=A["questions"],
        shared_answers=set(A.get("used_answers",[]))
    )
    # 최종 교차 family 검사
    af=set().union(*(families_for(q) for q in A["questions"]))
    bf=set().union(*(families_for(q) for q in B["questions"]))
    if af & bf:
        raise RuntimeError("A/B concept-family 중복 발견: "+", ".join(sorted(af&bf)))
    return A,B
