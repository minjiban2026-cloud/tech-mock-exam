
import random, math, re
from formula_templates import generate_formula_question
from retrieval import related_bundle,bundle_context,official_style_profile,candidate_cluster
from ai_wrapper import rewrite_bundle,safe_bundle_question
from validators import validate_formula_question,validate_grounded_question,too_similar,validate_exam,fingerprint
from patterns import blueprint,weighted_pick
from concept_families import families_for
from quality_judge import select_coherent_bundle,judge_question,judge_exam,judge_ab_pair

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
                 ai_enabled=True,ai_quality_enabled=True,judge_model=None,seed=None,
                 previous_questions=None,shared_answers=None):
    rng=random.Random(seed)
    domains=list(domains or DOMAINS)
    scores=score_pattern(section,count,points)
    plan=blueprint(section,scores,domains,rng)
    style=official_style_profile(db_path)
    judge_model=judge_model or model
    quality_active=bool(ai_enabled and ai_quality_enabled and api_key)

    prior=list(previous_questions or [])
    qs=[]
    used_topics=set()
    used_answers=set(shared_answers or [])
    used_patterns=[]
    ai_calls=0; fallbacks=0; formula_used=0
    judge_calls=0; judge_rejects=0; selector_calls=0
    formula_cap=2

    for slot in plan:
        pts=slot["points"]; dom=slot["domain"]; q=None
        wants_calc=slot["question_type"] in {"간단계산","계산/판단"}

        if wants_calc and dom not in FORMULA_DOMAINS:
            slot["question_type"]="자료해석" if pts==4 else "자료식별"
            wants_calc=False

        # ---------- deterministic calculation + AI quality veto ----------
        if wants_calc and dom in FORMULA_DOMAINS and formula_used<formula_cap:
            for _ in range(16 if quality_active else 40):
                cand=generate_formula_question(dom,rng)
                if not cand:break
                cand=_enrich_formula(cand,pts)
                if not cand:continue
                if too_similar(cand,prior+qs):continue
                if validate_formula_question(cand):continue

                if quality_active:
                    try:
                        judge_calls+=1
                        review=judge_question(api_key,judge_model,cand,"",style)
                    except Exception:
                        review={"pass":False,"reason":"AI 품질심사 호출 실패"}
                    cand["ai_quality"]=review
                    if not review.get("pass"):
                        judge_rejects+=1
                        continue
                else:
                    cand["ai_quality"]={"pass":None,"mode":"not_run"}

                q=cand;formula_used+=1;break

        # ---------- concept question ----------
        if q is None:
            first_pat=weighted_pick(rng,pts,calc=False,used=used_patterns)
            local_rejected_topics=set()

            for pat in _concept_patterns(rng,pts,first_pat):
                need=len(pat["subpoints"])

                for _ in range(10 if quality_active else 32):
                    relation_meta={}
                    bundle=[]

                    if quality_active:
                        cluster=candidate_cluster(
                            db_path,dom,used_answers,
                            set(used_topics)|local_rejected_topics,
                            limit=9
                        )
                        if len(cluster)<need:
                            break
                        try:
                            selector_calls+=1
                            selected=select_coherent_bundle(
                                api_key,judge_model,cluster,pts,style,need=need
                            )
                        except Exception:
                            selected=None
                        if not selected:
                            if cluster:
                                local_rejected_topics.add(cluster[0]["topic"])
                            continue
                        bundle,relation_meta=selected
                    else:
                        bundle=related_bundle(
                            db_path,dom,need,used_answers,
                            set(used_topics)|local_rejected_topics
                        )
                        if len(bundle)<need:
                            break

                    ctx=bundle_context(db_path,bundle)
                    cand=None

                    if quality_active:
                        # AI 작성자와 AI 검토위원을 분리한다.
                        try:
                            ai_calls+=1
                            cand=rewrite_bundle(
                                api_key,model,bundle,pts,section,pat,
                                slot["question_type"],slot["material_form"],style,
                                source_context=ctx,relation_meta=relation_meta
                            )
                        except Exception:
                            cand=None

                        if cand is not None:
                            # 정답/evidence/source의 deterministic 검사는 judge 뒤에도 다시 수행한다.
                            try:
                                judge_calls+=1
                                review=judge_question(
                                    api_key,judge_model,cand,ctx,style
                                )
                            except Exception:
                                review={"pass":False,"reason":"AI 품질심사 호출 실패"}
                            cand["ai_quality"]=review

                            if not review.get("pass"):
                                judge_rejects+=1
                                cand=None
                            elif validate_grounded_question(
                                cand,ctx,allow_ai_grounded=True
                            ) or too_similar(cand,prior+qs):
                                cand=None

                        if cand is None:
                            if bundle:
                                local_rejected_topics.add(bundle[0]["topic"])
                            continue

                    else:
                        # API/품질심사를 사용하지 않는 경우는 사실 지문을 원문으로 잠근
                        # 보수적 폴백만 허용한다. AI 생성 지문을 무심사로 내보내지 않는다.
                        cand=safe_bundle_question(
                            bundle,pts,pat,slot["question_type"],slot["material_form"]
                        )
                        errs=validate_grounded_question(cand,ctx)
                        if errs or too_similar(cand,prior+qs):
                            if bundle:
                                local_rejected_topics.add(bundle[0]["topic"])
                            continue
                        cand["ai_quality"]={"pass":None,"mode":"source_locked_fallback"}
                        fallbacks+=1

                    q=cand
                    used_answers.update(q["answer"])
                    for a in bundle:
                        used_topics.add(a["topic"])
                    break

                if q is not None:
                    break

        if q is None:
            raise RuntimeError(
                f"{section} {slot['number']}번({dom})을 품질 기준으로 만들 수 없습니다. "
                "품질 기준을 낮추지 않고 다른 청사진을 재시도합니다."
            )

        q["number"]=slot["number"]
        q["blueprint_domain"]=dom
        q["concept_families"]=sorted(families_for(q))
        q["fingerprint"]=q.get("fingerprint") or fingerprint(q)
        qs.append(q)
        used_patterns.append(q.get("pattern_id"))

    errs=validate_exam(qs,count,points)
    if errs:
        raise RuntimeError("시험 자동검증 실패: "+" / ".join(errs))

    exam={
      "exam_title":f"기술 임용 모의고사 전공 {section}",
      "section":section,"total_points":points,"questions":qs,"verified":True,
      "blueprint":plan,
      "generation_stats":{
          "ai_calls":ai_calls,
          "safe_fallbacks":fallbacks,
          "formula_questions":formula_used,
          "ai_judge_calls":judge_calls,
          "ai_judge_rejects":judge_rejects,
          "ai_selector_calls":selector_calls,
      },
      "used_answers":list(used_answers),
      "verification_note":"DB/Python 정답 고정 + 구조검증 + AI 독립 품질심사 통과"
    }

    if quality_active:
        try:
            judge_calls+=1
            section_review=judge_exam(api_key,judge_model,exam,style)
        except Exception:
            section_review={"pass":False,"reason":"섹션 AI 심사 호출 실패"}
        exam["section_ai_quality"]=section_review
        exam["generation_stats"]["ai_judge_calls"]=judge_calls
        if not section_review.get("pass"):
            raise RuntimeError(
                f"{section} 섹션 전체 AI 품질심사 탈락: "
                +str(section_review.get("reason",""))
            )
    else:
        exam["section_ai_quality"]={"pass":None,"mode":"not_run"}

    return exam


def make_ab(db_path,a_count=12,a_points=40,b_count=11,b_points=40,domains=None,
            api_key="",model="gpt-5.6-terra",ai_enabled=True,ai_quality_enabled=True,
            judge_model=None,seed=None):
    base=0 if seed is None else int(seed)
    last_error=None

    # AI 품질심사로 문항/섹션이 탈락할 수 있으므로 청사진을 재시도한다.
    # 검증 기준은 낮추지 않는다.
    for a_try in range(4):
        a_seed=None if seed is None else base + a_try*1000
        try:
            A=make_section(
                db_path,"A",a_count,a_points,domains,api_key,model,
                ai_enabled,ai_quality_enabled,judge_model,
                seed=a_seed
            )
        except Exception as ex:
            last_error=ex
            continue

        af=set().union(*(families_for(q) for q in A["questions"]))
        for b_try in range(8):
            b_seed=None if seed is None else base + 1 + a_try*1000 + b_try*37
            try:
                B=make_section(
                    db_path,"B",b_count,b_points,domains,api_key,model,
                    ai_enabled,ai_quality_enabled,judge_model,
                    seed=b_seed,
                    previous_questions=A["questions"],
                    shared_answers=set(A.get("used_answers",[]))
                )
                bf=set().union(*(families_for(q) for q in B["questions"]))
                if af & bf:
                    last_error=RuntimeError(
                        "A/B concept-family 중복 발견: "+", ".join(sorted(af&bf))
                    )
                    continue
                quality_active=bool(ai_enabled and ai_quality_enabled and api_key)
                if quality_active:
                    try:
                        pair_review=judge_ab_pair(
                            api_key,judge_model or model,A,B,
                            official_style_profile(db_path)
                        )
                    except Exception:
                        pair_review={"pass":False,"reason":"A/B 종합 AI 심사 호출 실패"}
                    if not pair_review.get("pass"):
                        last_error=RuntimeError(
                            "A/B 종합 AI 품질심사 탈락: "
                            +str(pair_review.get("reason",""))
                        )
                        continue
                    A["ab_pair_ai_quality"]=pair_review
                    B["ab_pair_ai_quality"]=pair_review
                else:
                    A["ab_pair_ai_quality"]={"pass":None,"mode":"not_run"}
                    B["ab_pair_ai_quality"]={"pass":None,"mode":"not_run"}

                A["effective_seed"]=a_seed
                B["effective_seed"]=b_seed
                A["generation_retry"]={"A_attempt":a_try+1}
                B["generation_retry"]={"B_attempt":b_try+1}
                return A,B
            except Exception as ex:
                last_error=ex
                continue
    raise RuntimeError(
        "정답/품질 기준을 낮추지 않은 상태에서 A/B 편성에 실패했습니다: "
        +str(last_error)
    )

