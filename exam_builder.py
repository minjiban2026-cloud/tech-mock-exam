
import random
from formula_templates import generate_formula_question
from retrieval import random_anchor, source_context
from ai_wrapper import rewrite_anchor, conservative_anchor_question
from validators import validate_formula_question, validate_grounded_question, too_similar, validate_exam, fingerprint
from generation_policy import (
    is_circuit_question, circuit_allowed, looks_like_bad_anchor,
    formula_quality, concept_quality
)

DOMAINS=["기술교육론","발명","제조기술","건설기술","생명기술","전기·전자","통신기술","재료역학","수송기술"]

def score_pattern(section,count,points):
    defaults={
      "A":[2,2,2,2,4,4,4,4,4,4,4,4],
      "B":[2,2,4,4,4,4,4,4,4,4,4]
    }
    p=defaults.get(section,[])
    if len(p)==count and sum(p)==points:
        return p[:]
    n2=2*count-points//2
    if points%2==0 and 0<=n2<=count:
        return [2]*n2+[4]*(count-n2)
    raise ValueError("문항 수/총점으로 2점·4점 조합을 만들 수 없습니다.")

def _domain_plan(domains,count,rng):
    """선택 영역을 가능한 한 고르게 순환한다."""
    base=list(domains)
    rng.shuffle(base)
    out=[]
    while len(out)<count:
        cycle=base[:]
        rng.shuffle(cycle)
        out.extend(cycle)
    return out[:count]

def _candidate_domains(primary, domains, rng):
    rest=[d for d in domains if d!=primary]
    rng.shuffle(rest)
    return [primary]+rest

def make_section(db_path,section,count,points,domains=None,api_key="",model="gpt-5.6-luna",
                 ai_enabled=True,strict=True,seed=None,avoid_topics=None,
                 difficulty="적당히 어려움",circuit_policy="최대한 제외"):
    rng=random.Random(seed)
    domains=domains or DOMAINS
    scores=score_pattern(section,count,points)

    # 2점/4점 배점 순서는 실제 시험처럼 앞쪽에 저배점이 오도록 유지
    domain_plan=_domain_plan(domains,count,rng)

    questions=[]
    used_answers=set()
    used_topics=set(avoid_topics or [])
    formula_domains={"재료역학","수송기술","통신기술"}
    formula_cap=max(2, round(count*0.25)) if any(d in formula_domains for d in domains) else 0
    formula_used=0

    # AI 속도/비용 안전장치
    max_ai_calls_per_question=2
    total_ai_calls=0

    for i,pts in enumerate(scores):
        q=None
        primary_domain=domain_plan[i]

        # 계산형은 전체 약 25%까지만. 4점은 2단계 이상 구조인 템플릿만 통과.
        if formula_used<formula_cap and primary_domain in formula_domains:
            for _ in range(10):
                cand=generate_formula_question(primary_domain,rng)
                if not cand:
                    continue
                cand["points"]=pts
                if cand["topic"] in used_topics or too_similar(cand,questions):
                    continue
                if validate_formula_question(cand):
                    continue
                if not formula_quality(cand,pts):
                    continue
                if not circuit_allowed(cand,circuit_policy):
                    continue
                q=cand
                formula_used+=1
                break

        # 개념/상황형
        if q is None:
            ai_calls_for_this_question=0
            for d in _candidate_domains(primary_domain,domains,rng):
                if q is not None:
                    break
                for _ in range(10):
                    anchor=random_anchor(db_path,d,used_answers)
                    if looks_like_bad_anchor(anchor):
                        continue
                    if anchor["topic"] in used_topics:
                        continue

                    ctx=source_context(db_path,anchor["source_name"],anchor["page_no"],radius=0)

                    if ai_enabled and api_key and ai_calls_for_this_question<max_ai_calls_per_question:
                        ai_calls_for_this_question+=1
                        total_ai_calls+=1
                        try:
                            cand,errs=rewrite_anchor(
                                api_key,model,anchor,ctx,
                                points=pts,section=section,difficulty=difficulty
                            )
                        except Exception:
                            cand,errs=None,["AI 호출 실패"]
                        if cand is None or errs:
                            continue
                    else:
                        # 4점에는 단순 빈칸 폴백을 넣지 않는다.
                        if pts==4:
                            continue
                        cand=conservative_anchor_question(anchor,points=pts)

                    cand["points"]=pts
                    errs=validate_grounded_question(cand,ctx)
                    if errs:
                        continue
                    if not concept_quality(cand,pts,difficulty):
                        continue
                    if too_similar(cand,questions):
                        continue
                    if not circuit_allowed(cand,circuit_policy):
                        continue

                    q=cand
                    used_answers.add(anchor["answer"])
                    break

        if q is None:
            raise RuntimeError(
                f"{section} {i+1}번 문항을 품질 기준에 맞게 생성하지 못했습니다. "
                "같은 설정으로 한 번 더 생성하거나 일부 영역을 추가해 주세요."
            )

        q["number"]=i+1
        q["fingerprint"]=q.get("fingerprint") or fingerprint(q)
        questions.append(q)
        used_topics.add(q["topic"])

    errs=validate_exam(questions,count,points)
    if errs:
        raise RuntimeError("시험 자동검증 실패: "+" / ".join(errs))

    return {
      "exam_title":f"기술 임용 모의고사 전공 {section}",
      "section":section,
      "total_points":points,
      "questions":questions,
      "verified":True,
      "generation_stats":{
          "formula_questions":formula_used,
          "ai_calls":total_ai_calls
      },
      "verification_note":"4점 난이도 필터 + 계산형 Python 검산 + 개념형 원문 근거검증 + 영역 균형 편성"
    }

def make_ab(db_path,a_count=12,a_points=40,b_count=11,b_points=40,domains=None,
            api_key="",model="gpt-5.6-luna",ai_enabled=True,seed=None,
            difficulty="적당히 어려움",circuit_policy="최대한 제외"):
    a=make_section(
        db_path,"A",a_count,a_points,domains,api_key,model,ai_enabled,True,seed,
        difficulty=difficulty,circuit_policy=circuit_policy
    )
    avoid={q["topic"] for q in a["questions"]}
    b=make_section(
        db_path,"B",b_count,b_points,domains,api_key,model,ai_enabled,True,
        None if seed is None else seed+1,avoid_topics=avoid,
        difficulty=difficulty,circuit_policy=circuit_policy
    )
    return a,b
