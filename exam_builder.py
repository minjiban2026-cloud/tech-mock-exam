
import random, os, sqlite3, json
from formula_templates import generate_formula_question
from retrieval import random_anchor, source_context
from ai_wrapper import rewrite_anchor, conservative_anchor_question
from validators import validate_formula_question, validate_grounded_question, too_similar, validate_exam, fingerprint
from generation_policy import is_circuit_question, circuit_allowed, difficulty_allowed

DOMAINS=["기술교육론","발명","제조기술","건설기술","생명기술","전기·전자","통신기술","재료역학","수송기술"]

def score_pattern(section,count,points):
    defaults={
      "A":[2,2,2,2,4,4,4,4,4,4,4,4],
      "B":[2,2,4,4,4,4,4,4,4,4,4]
    }
    p=defaults.get(section,[])
    if len(p)==count and sum(p)==points:
        return p[:]
    # 2/4점 조합 해
    n2=2*count-points//2
    if points%2==0 and 0<=n2<=count:
        return [2]*n2+[4]*(count-n2)
    raise ValueError("문항 수/총점으로 2점·4점 조합을 만들 수 없습니다.")

def make_section(db_path,section,count,points,domains=None,api_key="",model="gpt-5.6-terra",
                 ai_enabled=True,strict=True,seed=None,avoid_topics=None,difficulty="적당히 어려움",circuit_policy="최대한 제외"):
    rng=random.Random(seed)
    domains=domains or DOMAINS
    scores=score_pattern(section,count,points)
    rng.shuffle(scores)
    questions=[]
    used_answers=set()
    used_topics=set(avoid_topics or [])
    # 계산형을 전체의 약 35% 우선 배치
    formula_domains=[d for d in domains if d in {"재료역학","수송기술","통신기술"}]
    target_formula=max(2,round(count*0.35)) if formula_domains else 0

    for i,pts in enumerate(scores):
        q=None
        # formula
        if i < target_formula and formula_domains:
            for _ in range(20):
                d=rng.choice(formula_domains)
                cand=generate_formula_question(d,rng)
                if not cand or cand["topic"] in used_topics or too_similar(cand,questions):
                    continue
                cand["points"]=pts
                if (not validate_formula_question(cand)
                    and difficulty_allowed(cand,difficulty)
                    and circuit_allowed(cand,circuit_policy,
                        sum(1 for x in questions if is_circuit_question(x)),len(questions))):
                    q=cand; break
        # source-grounded concept
        if q is None:
            shuffled=domains[:]
            rng.shuffle(shuffled)
            for d in shuffled:
                for _ in range(8):
                    anchor=random_anchor(db_path,d,used_answers)
                    if not anchor or anchor["topic"] in used_topics:
                        continue
                    ctx=source_context(db_path,anchor["source_name"],anchor["page_no"],radius=0)
                    if ai_enabled and api_key:
                        try:
                            cand,errs=rewrite_anchor(api_key,model,anchor,ctx,points=pts,section=section,difficulty=difficulty)
                            if errs:
                                cand=conservative_anchor_question(anchor,points=pts)
                        except Exception:
                            cand=conservative_anchor_question(anchor,points=pts)
                    else:
                        cand=conservative_anchor_question(anchor,points=pts)
                    cand["points"]=pts
                    errs=validate_grounded_question(cand,ctx)
                    if (not errs and not too_similar(cand,questions)
                        and difficulty_allowed(cand,difficulty)
                        and circuit_allowed(cand,circuit_policy,
                            sum(1 for x in questions if is_circuit_question(x)),len(questions))):
                        q=cand
                        used_answers.add(anchor["answer"])
                        break
                if q: break
        if q is None:
            raise RuntimeError(f"{section} {i+1}번 문항을 안전하게 생성하지 못했습니다.")
        q["number"]=i+1
        q["fingerprint"]=q.get("fingerprint") or fingerprint(q)
        questions.append(q)
        used_topics.add(q["topic"])

    errs=validate_exam(questions,count,points)
    if errs:
        raise RuntimeError("시험 자동검증 실패: "+" / ".join(errs))
    return {
      "exam_title":f"기술 임용 모의고사 전공 {section}",
      "section":section,"total_points":points,"questions":questions,
      "verified":True,
      "verification_note":"계산형은 Python 정답, 개념형은 원문 evidence/source/page 자동검증 통과"
    }

def make_ab(db_path,a_count=12,a_points=40,b_count=11,b_points=40,domains=None,
            api_key="",model="gpt-5.6-terra",ai_enabled=True,seed=None,difficulty="적당히 어려움",circuit_policy="최대한 제외"):
    a=make_section(db_path,"A",a_count,a_points,domains,api_key,model,ai_enabled,True,seed,difficulty=difficulty,circuit_policy=circuit_policy)
    avoid={q["topic"] for q in a["questions"]}
    b=make_section(db_path,"B",b_count,b_points,domains,api_key,model,ai_enabled,True,
                   None if seed is None else seed+1,avoid_topics=avoid,difficulty=difficulty,circuit_policy=circuit_policy)
    return a,b
