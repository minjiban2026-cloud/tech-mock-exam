
import random
from formula_templates import generate_formula_question
from retrieval import related_bundle,bundle_context,official_style_profile
from ai_wrapper import rewrite_bundle,safe_bundle_question
from validators import validate_formula_question,validate_grounded_question,too_similar,validate_exam,fingerprint
from patterns import blueprint,weighted_pick

DOMAINS=["기술교육론","발명","제조기술","건설기술","생명기술","전기·전자","통신기술","재료역학","수송기술"]
FORMULA_DOMAINS={"재료역학","수송기술","통신기술"}

def score_pattern(section,count,points):
    defaults={"A":[2,2,2,2,4,4,4,4,4,4,4,4],"B":[2,2,4,4,4,4,4,4,4,4,4]}
    p=defaults.get(section,[])
    if len(p)==count and sum(p)==points:return p[:]
    raise ValueError("현재 생성기는 실제 A/B 기본 배점 구조만 지원합니다.")

def _adapt_formula(q,pts,pattern):
    q["points"]=pts
    n=len(q.get("answer",[]))
    # 계산 템플릿의 실제 채점요소 수에 맞춰 부분점수 결정
    if pts==2:
        if n==1:
            q["subpoints"]=[2]
        elif n>=2:
            q["answer"]=q["answer"][:2]; q["tasks"]=q["tasks"][:2]; q["solution"]=q["solution"][:2]
            q["subpoints"]=[1,1]
    else:
        if n>=3:
            q["answer"]=q["answer"][:3]; q["tasks"]=q["tasks"][:3]; q["solution"]=q["solution"][:3]
            q["subpoints"]=[1,1,2]
        elif n==2:
            q["subpoints"]=[2,2]
        else:
            # 4점인데 채점요소 하나인 계산형은 사용하지 않음
            return None
    q["pattern_id"]=pattern["id"]
    q["question_type"]="계산/판단"
    q["fingerprint"]=fingerprint(q)
    return q

def make_section(db_path,section,count,points,domains=None,api_key="",model="gpt-5.6-luna",
                 ai_enabled=True,strict=True,seed=None,avoid_topics=None,shared_answers=None):
    rng=random.Random(seed)
    domains=list(domains or DOMAINS)
    scores=score_pattern(section,count,points)
    plan=blueprint(section,scores,domains,rng)
    style=official_style_profile(db_path)

    qs=[]; used_topics=set(avoid_topics or []); used_answers=set(shared_answers or [])
    used_pattern_ids=[]; ai_calls=0; fallbacks=0; formula_used=0
    # 실제 기출처럼 계산은 섹션당 소수만; 회로 자유생성은 없음
    formula_cap=2

    for slot in plan:
        pts=slot["points"]; primary=slot["domain"]; q=None

        # 계산형: 청사진 유형이 계산일 때 + 지원영역일 때만
        wants_calc=slot["question_type"] in {"간단계산","계산/판단"}
        # 계산 지원 영역이 아니면 유형명을 비계산형으로 보정한다.
        if wants_calc and primary not in FORMULA_DOMAINS:
            slot["question_type"]="자료해석" if pts==4 else "자료식별"
            wants_calc=False

        if wants_calc and primary in FORMULA_DOMAINS and formula_used<formula_cap:
            pat=weighted_pick(rng,pts,calc=True,used_pattern_ids=used_pattern_ids)
            for _ in range(18):
                cand=generate_formula_question(primary,rng)
                if not cand or cand.get("topic") in used_topics or too_similar(cand,qs): continue
                cand=_adapt_formula(cand,pts,pat)
                if not cand: continue
                if not validate_formula_question(cand):
                    q=cand; formula_used+=1; break

        # 개념형: 해당 영역의 관련 앵커 묶음
        if q is None:
            pat=weighted_pick(rng,pts,calc=False,used_pattern_ids=used_pattern_ids)
            need=len(pat["subpoints"])
            domain_order=[primary]+[d for d in domains if d!=primary]
            for d in domain_order:
                bundle=related_bundle(db_path,d,need,used_answers,used_topics)
                if len(bundle)<need: continue
                ctx=bundle_context(db_path,bundle)

                cand=None
                if ai_enabled and api_key:
                    try:
                        ai_calls+=1
                        cand=rewrite_bundle(api_key,model,bundle,ctx,pts,section,pat,slot["question_type"],style)
                    except Exception:
                        cand=None
                if cand is not None:
                    errs=validate_grounded_question(cand,ctx)
                    if errs or too_similar(cand,qs):
                        cand=None
                if cand is None:
                    # 품질 기준을 낮추지 않고 같은 부분점수 구조를 유지하는 원문형 안전 폴백
                    cand=safe_bundle_question(bundle,pts,pat,slot["question_type"])
                    errs=validate_grounded_question(cand,ctx)
                    if errs or too_similar(cand,qs):
                        continue
                    fallbacks+=1
                q=cand
                used_answers.update(q["answer"])
                # bundle의 개별 topic까지 재사용 금지
                for a in bundle: used_topics.add(a["topic"])
                break

        if q is None:
            raise RuntimeError(
              f"{section} {slot['number']}번: 현재 선택 영역에서 검증 가능한 문항 조합을 만들 수 없습니다. "
              "전체 영역을 선택하거나 다른 랜덤 시드로 다시 생성해 주세요."
            )

        q["number"]=slot["number"]
        q["blueprint_domain"]=primary
        q["question_type"]=q.get("question_type") or slot["question_type"]
        q["fingerprint"]=q.get("fingerprint") or fingerprint(q)
        qs.append(q); used_topics.add(q["topic"]); used_pattern_ids.append(q.get("pattern_id"))

    errs=validate_exam(qs,count,points)
    if errs: raise RuntimeError("시험 자동검증 실패: "+" / ".join(errs))
    return {
      "exam_title":f"기술 임용 모의고사 전공 {section}",
      "section":section,"total_points":points,"questions":qs,"verified":True,
      "blueprint":plan,
      "generation_stats":{"ai_calls":ai_calls,"safe_fallbacks":fallbacks,"formula_questions":formula_used},
      "used_answers":list(used_answers),
      "verification_note":"실제기출 청사진·부분점수·원문근거·Python계산·중복·영역분산 자동검증 통과"
    }

def make_ab(db_path,a_count=12,a_points=40,b_count=11,b_points=40,domains=None,
            api_key="",model="gpt-5.6-luna",ai_enabled=True,seed=None):
    a=make_section(db_path,"A",a_count,a_points,domains,api_key,model,ai_enabled,True,seed)
    avoid=set()
    for q in a["questions"]:
        avoid.update(x.strip() for x in q["topic"].split("·"))
    b=make_section(db_path,"B",b_count,b_points,domains,api_key,model,ai_enabled,True,
                   None if seed is None else seed+1,avoid_topics=avoid,
                   shared_answers=set(a.get("used_answers",[])))
    af={q["fingerprint"] for q in a["questions"]}; bf={q["fingerprint"] for q in b["questions"]}
    if af & bf: raise RuntimeError("A/B 교차 중복 발견")
    return a,b
