import random
from formula_templates import generate_formula_question
from retrieval import anchor_pool,source_context,style_snippets
from ai_wrapper import rewrite_anchor
from validators import validate_formula_question,validate_grounded_question,too_similar,validate_exam,fingerprint
from generation_policy import circuit_allowed,looks_like_bad_anchor,formula_quality,concept_quality
from pattern_engine import build_blueprint,PATTERNS

DOMAINS=["기술교육론","발명","제조기술","건설기술","생명기술","전기·전자","통신기술","재료역학","수송기술"]

def score_pattern(section,count,points):
    defaults={"A":[2,2,2,2,4,4,4,4,4,4,4,4],"B":[2,2,4,4,4,4,4,4,4,4,4]}
    p=defaults.get(section,[])
    if len(p)==count and sum(p)==points:return p[:]
    n2=2*count-points//2
    if points%2==0 and 0<=n2<=count:return [2]*n2+[4]*(count-n2)
    raise ValueError("배점 조합 오류")

def _pool(db_path,d,used_answers,used_topics):
    return [a for a in anchor_pool(db_path,d,used_answers,220) if not looks_like_bad_anchor(a) and a['topic'] not in used_topics]

def _formula_allowed(domain,typ):
    return typ=="calc" and domain in {"재료역학","수송기술","통신기술"}

def make_section(db_path,section,count,points,domains=None,api_key="",model="gpt-5.6-luna",ai_enabled=True,strict=True,seed=None,avoid_topics=None,difficulty="적당히 어려움",circuit_policy="최대한 제외",shared_used_answers=None):
    rng=random.Random(seed); domains=list(domains or DOMAINS); scores=score_pattern(section,count,points)
    blueprint=build_blueprint(section,scores,domains,rng)
    questions=[]; used_answers=set(shared_used_answers or []); used_topics=set(avoid_topics or []); styles=style_snippets(db_path,3)
    ai_calls=0; formula_used=0; type_counts={}

    for slot in blueprint:
        pts=slot['points']; primary=slot['domain']; typ=slot['type']; q=None
        # 전기전자는 회로/계산을 기본 배제. 계산 청사진이어도 다른 적합 영역으로 넘긴다.
        if primary=="전기·전자" and typ=="calc":typ="data_interpret"; slot['type']=typ; slot['pattern']=PATTERNS[typ]

        if _formula_allowed(primary,typ):
            for _ in range(16):
                cand=generate_formula_question(primary,rng)
                if not cand:continue
                cand['points']=pts; cand['question_type']='calc'
                if cand['topic'] in used_topics or too_similar(cand,questions):continue
                if validate_formula_question(cand) or not formula_quality(cand,pts) or not circuit_allowed(cand,circuit_policy):continue
                q=cand; formula_used+=1; break

        # 개념형은 2점/4점 모두 AI가 '형식'을 재구성. 각 자리 최대 3개의 다른 앵커를 시도.
        if q is None and ai_enabled and api_key:
            ordered=[primary]+[d for d in domains if d!=primary]
            attempts=0
            for d in ordered:
                for a in _pool(db_path,d,used_answers,used_topics)[:8]:
                    if attempts>=3 or q is not None:break
                    attempts+=1; ai_calls+=1
                    ctx=source_context(db_path,a['source_name'],a['page_no'],radius=0)
                    try:cand,errs=rewrite_anchor(api_key,model,a,ctx,pts,section,difficulty,styles,slot)
                    except Exception:cand,errs=None,["AI 실패"]
                    if cand is None or errs:continue
                    if too_similar(cand,questions) or not circuit_allowed(cand,circuit_policy):continue
                    if not concept_quality(cand,pts,difficulty,typ):continue
                    q=cand; used_answers.add(a['answer']); break
                if q is not None or attempts>=3:break

        # AI가 꺼졌거나 3회 모두 실패: 단순 클로즈를 넣지 않고 검증된 계산형 중 적합한 것을 찾아 채움.
        # 이 경우 유형 유사도는 낮아지므로 stats에 degraded로 기록한다.
        degraded=False
        if q is None:
            fallback_domains=[d for d in [primary,"재료역학","수송기술","통신기술"] if d in domains]
            for d in fallback_domains:
                for _ in range(20):
                    cand=generate_formula_question(d,rng)
                    if not cand:continue
                    cand['points']=pts; cand['question_type']='calc_fallback'
                    if cand['topic'] in used_topics or too_similar(cand,questions):continue
                    if validate_formula_question(cand) or not formula_quality(cand,pts) or not circuit_allowed(cand,circuit_policy):continue
                    q=cand; formula_used+=1; degraded=True; break
                if q:break
        if q is None:
            raise RuntimeError(f"{section} {slot['slot']}번: 기출형 품질 기준을 만족하는 문항을 만들지 못했습니다. AI 사용 여부/API 상태를 확인해 주세요.")

        q['number']=slot['slot']; q['blueprint_type']=slot['type']; q['degraded']=degraded; q['fingerprint']=q.get('fingerprint') or fingerprint(q)
        questions.append(q); used_topics.add(q['topic']); type_counts[q.get('question_type',typ)]=type_counts.get(q.get('question_type',typ),0)+1

    errs=validate_exam(questions,count,points)
    if errs:raise RuntimeError("시험 자동검증 실패: "+" / ".join(errs))
    return {"exam_title":f"기술 임용 모의고사 전공 {section}","section":section,"total_points":points,"questions":questions,"verified":True,
            "generation_stats":{"formula_questions":formula_used,"ai_calls":ai_calls,"type_counts":type_counts,"degraded":sum(q.get('degraded',False) for q in questions)},
            "used_answers":list(used_answers),"blueprint":blueprint,
            "verification_note":"기출 구조 청사진 + 원문 근거 고정 + 계산 Python 검산 + 유형 다양성 검사"}

def make_ab(db_path,a_count=12,a_points=40,b_count=11,b_points=40,domains=None,api_key="",model="gpt-5.6-luna",ai_enabled=True,seed=None,difficulty="적당히 어려움",circuit_policy="최대한 제외"):
    a=make_section(db_path,"A",a_count,a_points,domains,api_key,model,ai_enabled,True,seed,difficulty=difficulty,circuit_policy=circuit_policy)
    avoid={q['topic'] for q in a['questions']}
    b=make_section(db_path,"B",b_count,b_points,domains,api_key,model,ai_enabled,True,None if seed is None else seed+1,avoid_topics=avoid,difficulty=difficulty,circuit_policy=circuit_policy,shared_used_answers=set(a.get('used_answers',[])))
    return a,b
