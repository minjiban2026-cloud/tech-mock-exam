
import random, re
from formula_templates import generate_formula_question
from retrieval import anchor_pool, source_context, style_snippets
from ai_wrapper import rewrite_anchor, conservative_anchor_question, paired_anchor_question
from validators import (
    validate_formula_question, validate_grounded_question,
    too_similar, validate_exam, fingerprint
)
from generation_policy import (
    circuit_allowed, looks_like_bad_anchor,
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
    """선택한 영역이 가능한 한 고르게 출제되게 한다."""
    domains=list(domains)
    out=[]
    while len(out)<count:
        cycle=domains[:]
        rng.shuffle(cycle)
        out.extend(cycle)
    return out[:count]

def _valid_pool(db_path,domain,used_answers,used_topics):
    out=[]
    for a in anchor_pool(db_path,domain,used_answers,180):
        if looks_like_bad_anchor(a):
            continue
        if a["topic"] in used_topics:
            continue
        out.append(a)
    return out

def _combined_context(db_path,anchors):
    parts=[]
    for a in anchors:
        parts.append(source_context(db_path,a["source_name"],a["page_no"],radius=0))
    return "\n\n".join(parts)

def _pick_pair(pool,used_topics):
    for i,a1 in enumerate(pool):
        for a2 in pool[i+1:]:
            if a1["answer"]==a2["answer"]:
                continue
            if a1["topic"]==a2["topic"]:
                continue
            if a1["topic"] in used_topics or a2["topic"] in used_topics:
                continue
            return a1,a2
    return None,None

def make_section(db_path,section,count,points,domains=None,api_key="",model="gpt-5.6-luna",
                 ai_enabled=True,strict=True,seed=None,avoid_topics=None,
                 difficulty="적당히 어려움",circuit_policy="최대한 제외",
                 shared_used_answers=None):
    rng=random.Random(seed)
    domains=list(domains or DOMAINS)
    if not domains:
        raise ValueError("출제 영역을 한 개 이상 선택해 주세요.")

    scores=score_pattern(section,count,points)
    domain_plan=_domain_plan(domains,count,rng)

    questions=[]
    used_answers=set(shared_used_answers or [])
    used_topics=set(avoid_topics or [])
    styles=style_snippets(db_path,2)

    # 계산형 과다 출제를 막기 위해 섹션당 약 20%만 허용
    formula_domains={"재료역학","수송기술","통신기술"}
    formula_cap=max(1,round(count*0.20)) if any(d in formula_domains for d in domains) else 0
    formula_used=0
    ai_calls=0
    fallback_count=0

    for i,pts in enumerate(scores):
        q=None
        primary=domain_plan[i]

        # 1) 계산형: 해당 순번의 주영역이 계산 지원영역일 때만 시도
        if formula_used<formula_cap and primary in formula_domains:
            for _ in range(12):
                cand=generate_formula_question(primary,rng)
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

        # 2) 원자료 기반 개념형
        if q is None:
            ordered=[primary]+[d for d in domains if d!=primary]
            # 먼저 주영역에서 충분히 찾고, 정말 없을 때만 다른 영역으로 이동
            for d in ordered:
                pool=_valid_pool(db_path,d,used_answers,used_topics)
                if not pool:
                    continue

                # 2점: AI를 쓰지 않아도 안전하게 즉시 생성
                if pts==2:
                    for anchor in pool[:12]:
                        cand=conservative_anchor_question(anchor,2)
                        ctx=source_context(db_path,anchor["source_name"],anchor["page_no"],radius=0)
                        if validate_grounded_question(cand,ctx):
                            continue
                        if too_similar(cand,questions):
                            continue
                        if not circuit_allowed(cand,circuit_policy):
                            continue
                        q=cand
                        used_answers.add(anchor["answer"])
                        break
                    if q:
                        break
                    continue

                # 4점: 좋은 앵커 하나를 AI가 실제 임용형으로 재구성.
                # 문항당 AI는 딱 1회만 사용해 시간/비용 폭주를 방지.
                if ai_enabled and api_key:
                    for anchor in pool[:8]:
                        ctx=source_context(db_path,anchor["source_name"],anchor["page_no"],radius=0)
                        try:
                            ai_calls+=1
                            cand,errs=rewrite_anchor(
                                api_key,model,anchor,ctx,points=4,section=section,
                                difficulty=difficulty,style_examples=styles
                            )
                        except Exception:
                            cand,errs=None,["AI 호출 실패"]

                        if cand is not None and not errs:
                            if (concept_quality(cand,4,difficulty)
                                and not too_similar(cand,questions)
                                and circuit_allowed(cand,circuit_policy)):
                                q=cand
                                used_answers.add(anchor["answer"])
                                break
                        # 한 문항 자리에서 API 재시도는 하지 않는다.
                        break
                    if q:
                        break

                # 3) AI 실패/미사용 시: 동일 영역의 원문 앵커 두 개로 검증 가능한 4점 폴백
                a1,a2=_pick_pair(pool,used_topics)
                if a1 and a2:
                    cand=paired_anchor_question(a1,a2)
                    ctx=_combined_context(db_path,[a1,a2])
                    if (not validate_grounded_question(cand,ctx)
                        and not too_similar(cand,questions)
                        and circuit_allowed(cand,circuit_policy)):
                        q=cand
                        fallback_count+=1
                        used_answers.update([a1["answer"],a2["answer"]])
                        # 두 주제를 모두 사용 처리
                        used_topics.add(a2["topic"])
                        break

        # 4) 마지막 안전망: 모든 선택 영역을 다시 훑어 반드시 채운다.
        if q is None:
            for d in domains:
                pool=_valid_pool(db_path,d,used_answers,used_topics)
                if pts==2 and pool:
                    anchor=pool[0]
                    cand=conservative_anchor_question(anchor,2)
                    ctx=source_context(db_path,anchor["source_name"],anchor["page_no"],radius=0)
                    if not validate_grounded_question(cand,ctx) and circuit_allowed(cand,circuit_policy):
                        q=cand
                        used_answers.add(anchor["answer"])
                        fallback_count+=1
                        break
                if pts==4:
                    a1,a2=_pick_pair(pool,used_topics)
                    if a1 and a2:
                        cand=paired_anchor_question(a1,a2)
                        ctx=_combined_context(db_path,[a1,a2])
                        if not validate_grounded_question(cand,ctx) and circuit_allowed(cand,circuit_policy):
                            q=cand
                            used_answers.update([a1["answer"],a2["answer"]])
                            used_topics.add(a2["topic"])
                            fallback_count+=1
                            break

        if q is None:
            # 여기까지 오는 경우는 DB 자체에 선택영역의 안전 앵커가 부족한 경우뿐이다.
            raise RuntimeError(
                f"{section} {i+1}번: 선택한 영역의 검증 가능한 원자료가 부족합니다. "
                "다른 영역을 1개 이상 추가해 주세요."
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
          "ai_calls":ai_calls,
          "safe_fallbacks":fallback_count
      },
      "used_answers":list(used_answers),
      "verification_note":"기출형 형식 참고 + Python 계산검산 + 원문근거 검증 + A/B 주제중복 억제 + 안전 폴백"
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
        difficulty=difficulty,circuit_policy=circuit_policy,
        shared_used_answers=set(a.get("used_answers",[]))
    )
    return a,b
