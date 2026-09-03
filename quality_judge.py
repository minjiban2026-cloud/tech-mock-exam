import json, re, statistics

SCORE_KEYS = [
    "grounding","answer_leakage","coherence","inferential_distance",
    "task_distinctness","exam_realism","difficulty_fit","ambiguity_control",
]

def _strip_json(text):
    t=(text or "").strip()
    t=re.sub(r"^```(?:json)?\s*","",t)
    t=re.sub(r"\s*```$","",t)
    return t

def _client(api_key):
    from openai import OpenAI
    return OpenAI(api_key=api_key, timeout=20, max_retries=0)

def _ask_json(api_key, model, prompt, effort="medium"):
    r=_client(api_key).responses.create(model=model,input=prompt,reasoning={"effort":effort})
    return json.loads(_strip_json(r.output_text))

def select_coherent_bundle(api_key, model, candidates, points, style_profile, need=None):
    items=[]
    for i,a in enumerate(candidates):
        evidence=str(a.get("evidence","") or "")
        items.append({
            "index":i,
            "topic":str(a.get("topic","") or "")[:120],
            "answer":str(a.get("answer","") or "")[:160],
            "evidence":evidence[:420],
            "source_name":str(a.get("source_name","") or "")[:160],
            "page_no":a.get("page_no"),
        })
    need=int(need or (3 if points==4 else 2))
    prompt=f"""
너는 대한민국 중등 기술 임용시험의 관계성 선별 심사자다.
새 사실이나 정답을 만들지 말고 원문 anchor 중 한 문제에서 실제로 연결해 물을 수 있는 것만 고른다.

공통 규칙:
- 하나의 master concept / 현상 / 절차 / 시스템 안에서 연결되어야 한다.
- 같은 페이지라는 이유만으로 묶지 않는다.
- 서로 다른 출처를 섞지 않는다.
- 정의나 명칭을 병렬로 찾아 쓰는 조합은 REJECT한다.
- 지문의 정의를 그대로 옮겨야만 답할 수 있는 조합도 REJECT한다.

2점 문항:
- '명칭 A 쓰기 + 명칭 B 쓰기'처럼 독립 회상 2개면 반드시 REJECT한다.
- 두 정답요소 사이에 원인-결과, 조건-결과, 개념-사례, 오류-수정,
  상위-하위, 비교-구분, 과정상 선후관계 중 하나가 원문으로 성립해야 한다.
- 최소 2종의 사고행동이 가능해야 한다.
- 자료를 종합하지 않고 각 줄에서 답을 하나씩 찾는 구조면 REJECT한다.

4점 문항:
- 최소 2종 이상의 사고행동이 가능해야 하며 단순 병렬 암기는 REJECT한다.

실제 기출의 구조적 특징:
{str(style_profile)[:1400]}

배점: {points}
필요 anchor 수: {need}
후보:
{json.dumps(items,ensure_ascii=False)}

JSON만 출력:
{{
 "verdict":"PASS 또는 REJECT",
 "selected_indices":[정수...],
 "master_concept":"한 문장",
 "relation":"선택 항목들이 어떻게 연결되는지 한 문장",
 "thinking_types":["식별","판단","관계설명","비교","적용" 등],
 "reason":"짧은 근거"
}}
"""
    x=_ask_json(api_key,model,prompt,"low")
    idx=x.get("selected_indices",[])
    if x.get("verdict")!="PASS": return None
    if not isinstance(idx,list) or len(idx)!=need: return None
    if any(not isinstance(i,int) or i<0 or i>=len(candidates) for i in idx): return None
    if len(set(idx))!=len(idx): return None
    chosen=[candidates[i] for i in idx]
    if len({a.get("source_name") for a in chosen})!=1: return None
    pages=[int(a.get("page_no",0)) for a in chosen]
    if pages and max(pages)-min(pages)>2: return None
    thinking=[str(v).strip() for v in x.get("thinking_types",[]) if str(v).strip()][:4]
    relation=str(x.get("relation","")).strip()
    if points==2 and (len(set(thinking))<2 or len(relation)<4):
        return None
    return chosen,{
        "master_concept":str(x.get("master_concept","")).strip(),
        "relation":relation,
        "thinking_types":thinking,
        "selector_reason":str(x.get("reason","")).strip(),
    }

def judge_question(api_key, model, question, source_context="", style_profile=""):
    """
    R10: 문항당 AI 품질심사는 1회만 호출한다.
    DB/Python grounding validator가 사실·정답을 먼저 검증하고,
    이 reviewer는 임용 유사성/추론거리/자료필요성/모호성에 veto만 행사한다.
    """
    review_q={
        "domain":question.get("domain"),"points":question.get("points"),
        "pattern_id":question.get("pattern_id"),
        "question_type":question.get("question_type"),"material_form":question.get("material_form"),
        "intro":question.get("intro"),"passage":question.get("passage"),
        "conditions":question.get("conditions",[]),"tasks":question.get("tasks",[]),
        "fixed_answer":question.get("answer",[]),"evidence":question.get("evidence",[]),
        "master_concept":question.get("master_concept",""),"relation":question.get("relation",""),
    }
    prompt=f"""
너는 대한민국 중등 기술 임용 1차 전공시험의 최종 외부 검토위원이다.
정답은 DB/Python에서 이미 고정되었다. 절대로 정답을 고치거나 새 답을 제안하지 말고 PASS/REJECT veto만 한다.

실제 기출 구조:
{style_profile}

고정 원자료 문맥:
{source_context[:9000]}

문항과 고정정답:
{json.dumps(review_q,ensure_ascii=False)}

강한 REJECT:
- 서브노트의 주변적·사소한 명칭 하나를 맞히는 것이 사실상 핵심인 문제
- 자료 한 문장/표 한 행에서 답을 바로 옮기는 문제
- 2점의 두 채점요소가 모두 명칭/용어 회상인 문제
- 2점에서 첫 명칭을 쓴 뒤 두 번째도 사실상 같은 정보를 반복 확인하는 문제
- 2점인데 서로 다른 독립 개념·사례·활용처를 여러 개 억지로 묶어 지문만 길어진 문제
- 난도를 사고가 아니라 정보량/장문으로 만든 문제
- 기술 임용 핵심범위보다 지나치게 지엽적인 고유명·특수사례·세부종류 자체를 정답으로 요구하는 문제
- 완성 공식/정의를 보여주고 공식명·개념명만 묻는 문제
- 4점 자료가 여러 정답의 정의·조건을 거의 그대로 나열하여 수험생이 복사만 하면 되는 문제
- 4점인데 소문항들이 서로 독립적으로 풀리는 병렬 암기 묶음
- 4점의 앞 판단/계산이 뒤 적용·설명에 실제로 쓰이지 않음
- 서로 다른 독립 개념 3개를 '같은 분야'라는 이유만으로 한 문제에 합친 문제
- 자료가 없어도 거의 같은 답을 할 수 있어 자료가 장식적임
- 배점 대비 너무 쉽거나 모호함
- 원자료가 뒷받침하지 않는 사실·수치·인과관계가 추가됨
- 고정정답과 작성 요구의 대응이 모호함

0~5로 평가:
grounding, answer_leakage(5=노출 없음), coherence, inferential_distance,
task_distinctness, exam_realism, difficulty_fit, ambiguity_control

JSON만 출력:
{{
 "verdict":"PASS 또는 REJECT",
 "scores":{{
  "grounding":0,"answer_leakage":0,"coherence":0,"inferential_distance":0,
  "task_distinctness":0,"exam_realism":0,"difficulty_fit":0,"ambiguity_control":0
 }},
 "thinking_types":["자료해석","판단","관계설명","적용" 등],
 "fatal_flags":["ROTE_ONLY","DECORATIVE_MATERIAL","TOO_EASY","AMBIGUOUS","UNRELATED_SUBPARTS","DIRECT_ANSWER_LEAK","UNSUPPORTED_FACT" 중 해당되는 것만],
 "reason":"2~4문장",
 "weakest_point":"가장 큰 약점 한 문장"
}}
"""
    x=_ask_json(api_key,model,prompt,"low")
    try:
        ss=x.get("scores",{})
        vals={k:float(ss.get(k,-1)) for k in SCORE_KEYS}
    except Exception:
        return {"pass":False,"review_stage":"integrated","reason":"통합 AI 심사 점수 파싱 실패","raw":x}

    if any(v<0 or v>5 for v in vals.values()):
        return {"pass":False,"review_stage":"integrated","reason":"통합 AI 심사 점수 범위 오류","scores":vals}

    fatal=[str(f) for f in x.get("fatal_flags",[]) if str(f).strip()]
    avg=sum(vals.values())/len(vals)
    pts=int(question.get("points",0))

    if pts==4:
        # R44: deterministic calculation/application questions are allowed to have
        # inferential_distance=3 when the external reviewer itself says PASS and
        # all other quality dimensions clear the 4-point floor.  R43 exposed an
        # inconsistency where the reviewer described a question as appropriate
        # but our local 3.5 threshold silently flipped it to REJECT.
        pattern_id=str(question.get("pattern_id","") or "")
        is_deterministic_operation=(
            pattern_id=="T4_C112"
            or str(question.get("selection_mode","") or "")=="deterministic_formula_operation"
            or str(question.get("reasoning_mode","") or "")=="deterministic_formula_operation"
        )
        inferential_floor=3.0 if is_deterministic_operation else 3.5
        passed=(
            x.get("verdict")=="PASS"
            and vals["grounding"]>=4
            and vals["answer_leakage"]>=4
            and vals["coherence"]>=4
            and vals["inferential_distance"]>=inferential_floor
            and vals["task_distinctness"]>=4
            and vals["exam_realism"]>=4
            and vals["difficulty_fit"]>=4
            and vals["ambiguity_control"]>=4
            and avg>=4.0
            and not fatal
        )
    else:
        passed=(
            x.get("verdict")=="PASS"
            and vals["grounding"]>=4
            and vals["answer_leakage"]>=4
            and vals["exam_realism"]>=3.5
            and vals["ambiguity_control"]>=4
            and vals["inferential_distance"]>=3
            and avg>=3.7
            and not fatal
        )

    return {
        "pass":bool(passed),
        "review_stage":"integrated",
        "scores":vals,
        "average":round(avg,3),
        "thinking_types":[str(v) for v in x.get("thinking_types",[])][:6],
        "fatal_flags":fatal,
        "reason":str(x.get("reason","")),
        "weakest_point":str(x.get("weakest_point","")),
        # 기존 diagnostics/UI 호환
        "blind_verdict":x.get("verdict"),
        "grounded_verdict":"PYTHON_GROUNDING",
    }

def judge_exam(api_key, model, exam, style_profile=""):
    rows=[]
    for q in exam.get("questions",[]):
        rows.append({
            "number":q.get("number"),"domain":q.get("domain"),"points":q.get("points"),
            "question_type":q.get("question_type"),"material_form":q.get("material_form"),
            "passage":str(q.get("passage",""))[:500],"tasks":q.get("tasks",[]),
            "quality":q.get("ai_quality",{}),"topic":q.get("topic",""),
        })
    prompt=f"""
너는 중등 기술 임용시험의 최종 편집위원이다.
한 섹션 전체의 시험다운 구성만 평가한다.

실제 기출 구조:
{style_profile}

섹션:
{json.dumps(rows,ensure_ascii=False)}

REJECT 조건:
- 단순 회상/분류형 과도 반복
- 4점 대부분이 같은 문법
- 자료형만 다르고 실제 사고가 동일
- 특정 영역/사고행동 편중
- 실제 임용보다 현저히 단조로움

JSON만:
{{"verdict":"PASS 또는 REJECT","exam_realism":0,"variety":0,"difficulty_balance":0,"reason":"2~4문장"}}
"""
    x=_ask_json(api_key,model,prompt,"low")
    try:
        er=float(x.get("exam_realism",-1)); va=float(x.get("variety",-1)); db=float(x.get("difficulty_balance",-1))
    except Exception:
        return {"pass":False,"reason":"섹션 심사 파싱 실패","raw":x}
    return {"pass":x.get("verdict")=="PASS" and min(er,va,db)>=4,
            "exam_realism":er,"variety":va,"difficulty_balance":db,"reason":str(x.get("reason",""))}

def judge_ab_pair(api_key, model, A, B, style_profile=""):
    def compact(exam):
        return [{
            "number":q.get("number"),"domain":q.get("domain"),"points":q.get("points"),
            "topic":q.get("topic",""),"question_type":q.get("question_type"),
            "tasks":q.get("tasks",[]),"thinking_types":q.get("ai_quality",{}).get("thinking_types",[]),
        } for q in exam.get("questions",[])]
    prompt=f"""
너는 중등 기술 임용 전공 A/B 최종 편집위원이다.
정답을 바꾸지 말고 두 책 전체 구성 품질만 평가한다.

실제 기출 구조:
{style_profile}

A:
{json.dumps(compact(A),ensure_ascii=False)}

B:
{json.dumps(compact(B),ensure_ascii=False)}

REJECT:
- A/B에서 사실상 같은 원리/사고 반복
- 같은 작성방법 문법 과도 반복
- 4점 사고행동 대부분 동일
- 문제 경험이 단조로움
- 실제 임용 A/B보다 현저히 단순함

JSON만:
{{"verdict":"PASS 또는 REJECT","cross_section_variety":0,"semantic_duplication_control":0,
"overall_exam_realism":0,"reason":"2~4문장"}}
"""
    x=_ask_json(api_key,model,prompt,"low")
    try:
        a=float(x.get("cross_section_variety",-1)); b=float(x.get("semantic_duplication_control",-1)); c=float(x.get("overall_exam_realism",-1))
    except Exception:
        return {"pass":False,"reason":"A/B 종합심사 파싱 실패","raw":x}
    return {"pass":x.get("verdict")=="PASS" and min(a,b,c)>=4,
            "cross_section_variety":a,"semantic_duplication_control":b,
            "overall_exam_realism":c,"reason":str(x.get("reason",""))}
