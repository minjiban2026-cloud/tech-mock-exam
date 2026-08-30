
import random, math, re
from formula_templates import generate_formula_question
from retrieval import related_bundle,bundle_context,official_style_profile
from ai_wrapper import rewrite_bundle,safe_bundle_question
from validators import validate_formula_question,validate_grounded_question,too_similar,validate_exam,fingerprint
from patterns import blueprint,weighted_pick
from concept_families import families_for

DOMAINS=["기술교육론","발명","제조기술","건설기술","생명기술","전기·전자","통신기술","재료역학","수송기술"]
FORMULA_DOMAINS={"재료역학","수송기술","통신기술"}

def _concept_patterns(rng,pts,first=None):
    from patterns import PATTERNS
    valid=[p for p in PATTERNS if p["points"]==pts and not p["calc"] and p.get("weight",1)>0]
    if pts==4:
        valid=sorted(valid,key=lambda p:(0 if p["id"]=="T4_112" else 1,-p.get("weight",1)))
    else:
        valid=sorted(valid,key=lambda p:-p.get("weight",1))
    out=[]
    if first is not None: out.append(first)
    for p in valid:
        if not any(x["id"]==p["id"] for x in out): out.append(p)
    return out

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
    """계산형 배점을 실제 채점요소로 환산한다. 4점은 서로 다른 3개 요구(1+1+2)만 허용한다."""
    topic=q.get("topic","")
    q["material_form"]="수치자료"
    q["question_type"]="계산/판단" if pts==4 else "간단계산"
    q["concept_family"]=next(iter(families_for(q)), "formula_"+re.sub(r"\W+","_",topic))

    if pts==2:
        if len(q.get("answer",[]))==1:
            relation=None
            if "수직응력" in topic: relation=("적용되는 수직응력의 관계식을 쓸 것.","σ=P/A","σ=P/A")
            elif "정수압" in topic: relation=("정수압의 관계식을 쓸 것.","p=ρgh","p=ρgh")
            elif "열효율" in topic: relation=("열기관 열효율의 관계식을 쓸 것.","η=(QH-QL)/QH×100","η=(QH-QL)/QH×100")
            if relation: q=_prepend_formula_element(q,*relation)
        if len(q.get("answer",[]))<2:return None
        q["answer"]=q["answer"][:2];q["tasks"]=q["tasks"][:2];q["solution"]=q["solution"][:2]
        q["subpoints"]=[1,1]
    else:
        # 템플릿 자체가 서로 다른 3개 채점요소를 제공해야 한다. 같은 공식을 복제해 점수를 부풀리지 않는다.
        if len(q.get("answer",[]))<3 or len(q.get("tasks",[]))<3:return None
        q["answer"]=q["answer"][:3];q["tasks"]=q["tasks"][:3];q["solution"]=q["solution"][:3]
        # 정답 요소가 사실상 중복이면 폐기
        canon=[re.sub(r"[^0-9A-Za-z가-힣]+","",str(x)).lower() for x in q["answer"][:3]]
        if len(set(canon))<3:return None
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
            first_pat=weighted_pick(rng,pts,calc=False,used=used_patterns)
            local_rejected_topics=set()
            for pat in _concept_patterns(rng,pts,first_pat):
                need=len(pat["subpoints"])
                for _ in range(24):
                    bundle=related_bundle(
                        db_path,dom,need,used_answers,
                        set(used_topics)|local_rejected_topics
                    )
                    if len(bundle)<need: break
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
                        if errs or too_similar(cand,prior+qs): cand=None
                    if cand is None:
                        cand=safe_bundle_question(
                            bundle,pts,pat,slot["question_type"],slot["material_form"]
                        )
                        errs=validate_grounded_question(cand,ctx)
                        if errs or too_similar(cand,prior+qs):
                            if bundle: local_rejected_topics.add(bundle[0]["topic"])
                            continue
                        fallbacks+=1
                    q=cand
                    used_answers.update(q["answer"])
                    for a in bundle: used_topics.add(a["topic"])
                    break
                if q is not None: break

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
    # 품질기준을 낮추지 않고 청사진만 결정론적으로 재시도한다.
    # 같은 입력 seed는 항상 같은 재시도 순서를 사용한다.
    base=0 if seed is None else int(seed)
    last_error=None
    for a_try in range(4):
        a_seed=None if seed is None else base + a_try*1000
        try:
            A=make_section(db_path,"A",a_count,a_points,domains,api_key,model,ai_enabled,seed=a_seed)
        except Exception as ex:
            last_error=ex
            continue

        af=set().union(*(families_for(q) for q in A["questions"]))
        for b_try in range(12):
            b_seed=None if seed is None else base + 1 + a_try*1000 + b_try*37
            try:
                B=make_section(
                    db_path,"B",b_count,b_points,domains,api_key,model,ai_enabled,
                    seed=b_seed,
                    previous_questions=A["questions"],
                    shared_answers=set(A.get("used_answers",[]))
                )
                bf=set().union(*(families_for(q) for q in B["questions"]))
                if af & bf:
                    last_error=RuntimeError("A/B concept-family 중복 발견: "+", ".join(sorted(af&bf)))
                    continue
                A["effective_seed"]=a_seed
                B["effective_seed"]=b_seed
                A["generation_retry"]={"A_attempt":a_try+1}
                B["generation_retry"]={"B_attempt":b_try+1}
                return A,B
            except Exception as ex:
                last_error=ex
                continue
    raise RuntimeError("엄격 품질 기준을 유지한 채 A/B 편성에 실패했습니다: "+str(last_error))
