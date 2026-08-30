
import json, re, statistics

SCORE_KEYS = [
    "grounding",
    "answer_leakage",
    "coherence",
    "inferential_distance",
    "task_distinctness",
    "exam_realism",
    "difficulty_fit",
    "ambiguity_control",
]

def _strip_json(text):
    t=(text or "").strip()
    t=re.sub(r"^```(?:json)?\s*","",t)
    t=re.sub(r"\s*```$","",t)
    return t

def _client(api_key):
    from openai import OpenAI
    return OpenAI(api_key=api_key, timeout=90, max_retries=2)

def _ask_json(api_key, model, prompt, effort="medium"):
    r=_client(api_key).responses.create(
        model=model,
        input=prompt,
        reasoning={"effort":effort},
    )
    x=json.loads(_strip_json(r.output_text))
    return x

def select_coherent_bundle(api_key, model, candidates, points, style_profile, need=None):
    """
    AI는 정답을 만들지 않고 후보 anchor 중 '한 개념 관계망'으로 묶일 항목만 선택한다.
    반환 인덱스 외의 사실은 생성에 사용되지 않는다.
    """
    items=[]
    for i,a in enumerate(candidates):
        items.append({
            "index":i,
            "topic":a.get("topic",""),
            "answer":a.get("answer",""),
            "evidence":a.get("evidence",""),
            "source_name":a.get("source_name",""),
            "page_no":a.get("page_no"),
        })
    need = int(need or (3 if points==4 else 2))
    prompt=f"""
너는 대한민국 중등 기술 임용시험의 '문항 설계자'가 아니라 '관계성 선별 심사자'다.
새로운 사실이나 정답을 만들면 안 된다.
아래 후보는 서브노트에서 추출된 정답 anchor다.

목표:
- {points}점 문항 하나에서 함께 물을 수 있는 후보 {need}개를 고른다.
- 반드시 하나의 master concept / 하나의 현상 / 하나의 절차 / 하나의 시스템 안에서
  '식별→관계/판단→적용/설명'처럼 연결되는 항목이어야 한다.
- 같은 페이지라는 이유만으로 무관한 용어를 묶으면 안 된다.
- 단순 병렬 나열(종류 3개 암기, 정의 3개 복사)은 reject한다.
- 서로 다른 출처를 섞지 않는다.
- 4점이면 최소 두 종류 이상의 사고행동이 가능한 관계여야 한다.

실제 기출의 구조적 특징:
{style_profile}

후보:
{json.dumps(items,ensure_ascii=False)}

JSON만 출력:
{{
 "verdict":"PASS 또는 REJECT",
 "selected_indices":[정수...],
 "master_concept":"한 문장",
 "relation":"항목들이 어떻게 연결되는지 한 문장",
 "thinking_types":["식별","관계설명","적용" 등],
 "reason":"짧은 근거"
}}
"""
    x=_ask_json(api_key,model,prompt,"medium")
    idx=x.get("selected_indices",[])
    if x.get("verdict")!="PASS": return None
    if not isinstance(idx,list) or len(idx)!=need: return None
    if any(not isinstance(i,int) or i<0 or i>=len(candidates) for i in idx): return None
    if len(set(idx))!=len(idx): return None
    chosen=[candidates[i] for i in idx]
    if len({a.get("source_name") for a in chosen})!=1: return None
    pages=[int(a.get("page_no",0)) for a in chosen]
    if pages and max(pages)-min(pages)>2: return None
    meta={
        "master_concept":str(x.get("master_concept","")).strip(),
        "relation":str(x.get("relation","")).strip(),
        "thinking_types":[str(v) for v in x.get("thinking_types",[])][:4],
        "selector_reason":str(x.get("reason","")).strip(),
    }
    return chosen, meta


def judge_question(api_key, model, question, source_context="", style_profile=""):
    """
    2중 독립 심사:
    1) Blind reviewer: 정답/근거를 보지 않고 실제 시험으로서의 난도·사고·모호성·베껴쓰기 느낌을 평가.
    2) Grounding reviewer: 고정 정답/근거를 보고 사실성·정답 노출·개념 응집도를 평가.
    정답의 진위 결정권은 DB/Python에 있고 AI는 veto만 행사한다.
    """
    public_q={
        "domain":question.get("domain"),
        "points":question.get("points"),
        "question_type":question.get("question_type"),
        "material_form":question.get("material_form"),
        "intro":question.get("intro"),
        "passage":question.get("passage"),
        "conditions":question.get("conditions",[]),
        "tasks":question.get("tasks",[]),
    }
    blind_prompt=f"""
너는 대한민국 중등 기술 임용 1차 전공시험의 외부 출제 검토위원이다.
중요: 정답과 출처는 제공하지 않는다. 실제 수험생처럼 '문제만' 보고 평가한다.
문항을 수정하지 말고 PASS/REJECT만 판정한다.

실제 기출 구조:
{style_profile}

문항:
{json.dumps(public_q,ensure_ascii=False)}

강한 REJECT 사유:
- 4점인데 사실상 정의/용어를 찾아 옮겨 쓰는 수준
- 하위 요구가 모두 같은 회상 행동
- 자료를 읽지 않아도 작성 방법만 보고 답할 수 있음
- 자료가 정답을 거의 그대로 말해 추론거리가 없음
- 서로 다른 요구 사이에 논리적 연결이 없음
- 배점 대비 너무 쉽거나, 무엇을 써야 할지 모호함
- 실제 임용보다 단순한 교과서 확인문제처럼 보임

0~5:
inferential_distance, task_distinctness, exam_realism, difficulty_fit, ambiguity_control

JSON만 출력:
{{
 "verdict":"PASS 또는 REJECT",
 "scores":{{
  "inferential_distance":0,
  "task_distinctness":0,
  "exam_realism":0,
  "difficulty_fit":0,
  "ambiguity_control":0
 }},
 "thinking_types":["식별","관계설명","적용" 등],
 "fatal_flags":["ROTE_ONLY","DECORATIVE_MATERIAL","TOO_EASY","AMBIGUOUS","UNRELATED_SUBPARTS" 중 해당되는 것만],
 "reason":"2~4문장"
}}
"""
    blind=_ask_json(api_key,model,blind_prompt,"medium")

    grounded_q=dict(public_q)
    grounded_q.update({
        "fixed_answer":question.get("answer",[]),
        "evidence":question.get("evidence",[]),
        "solution":question.get("solution",[]),
        "master_concept":question.get("master_concept",""),
        "relation":question.get("relation",""),
        "verifier":question.get("verifier"),
    })
    grounding_prompt=f"""
너는 중등 기술 임용시험의 사실검증·채점 검토위원이다.
정답 자체는 DB 또는 Python이 이미 고정했다. 새 정답을 만들지 말고 문항의 품질에 veto만 행사한다.

실제 기출 구조:
{style_profile}

고정 원자료 문맥:
{source_context[:9000]}

문항+고정정답:
{json.dumps(grounded_q,ensure_ascii=False)}

강한 REJECT 사유:
- 지문/조건에 원자료가 뒷받침하지 않는 기술 사실·수치·인과관계가 추가됨
- 정답 용어는 가렸지만 원문 정의를 거의 그대로 제시하여 사실상 답이 노출됨
- anchor들이 하나의 master concept가 아니라 단순히 가까운 페이지라서 묶임
- 작성 방법이 고정정답과 일대일 대응하지 않음
- 복수 정답/채점 모호성이 큼

0~5:
grounding, answer_leakage(5=노출 없음), coherence, ambiguity_control

JSON만 출력:
{{
 "verdict":"PASS 또는 REJECT",
 "scores":{{
   "grounding":0,
   "answer_leakage":0,
   "coherence":0,
   "ambiguity_control":0
 }},
 "fatal_flags":["DIRECT_ANSWER_LEAK","UNSUPPORTED_FACT","UNRELATED_SUBPARTS","AMBIGUOUS" 중 해당되는 것만],
 "reason":"2~4문장",
 "weakest_point":"가장 큰 약점 한 문장"
}}
"""
    grounded=_ask_json(api_key,model,grounding_prompt,"medium")

    try:
        bs=blind.get("scores",{})
        gs=grounded.get("scores",{})
        vals={
            "grounding":float(gs.get("grounding",-1)),
            "answer_leakage":float(gs.get("answer_leakage",-1)),
            "coherence":float(gs.get("coherence",-1)),
            "inferential_distance":float(bs.get("inferential_distance",-1)),
            "task_distinctness":float(bs.get("task_distinctness",-1)),
            "exam_realism":float(bs.get("exam_realism",-1)),
            "difficulty_fit":float(bs.get("difficulty_fit",-1)),
            "ambiguity_control":min(
                float(bs.get("ambiguity_control",-1)),
                float(gs.get("ambiguity_control",-1))
            ),
        }
    except Exception:
        return {"pass":False,"reason":"2중 AI 심사 점수 파싱 실패","blind_raw":blind,"grounded_raw":grounded}

    if any(v<0 or v>5 for v in vals.values()):
        return {"pass":False,"reason":"2중 AI 심사 점수 범위 오류","scores":vals}

    fatal=list(dict.fromkeys(
        [str(f) for f in blind.get("fatal_flags",[]) if str(f).strip()] +
        [str(f) for f in grounded.get("fatal_flags",[]) if str(f).strip()]
    ))
    avg=sum(vals.values())/len(vals)
    pts=int(question.get("points",0))

    blind_pass=blind.get("verdict")=="PASS"
    grounded_pass=grounded.get("verdict")=="PASS"

    if pts==4:
        passed=(
            blind_pass and grounded_pass and
            vals["grounding"]>=4 and
            vals["answer_leakage"]>=4 and
            vals["coherence"]>=4 and
            vals["inferential_distance"]>=3.5 and
            vals["task_distinctness"]>=4 and
            vals["exam_realism"]>=4 and
            vals["difficulty_fit"]>=4 and
            vals["ambiguity_control"]>=4 and
            avg>=4.05 and not fatal
        )
    else:
        passed=(
            blind_pass and grounded_pass and
            vals["grounding"]>=4 and
            vals["answer_leakage"]>=4 and
            vals["ambiguity_control"]>=4 and
            vals["exam_realism"]>=3.5 and
            avg>=3.7 and not fatal
        )

    return {
        "pass":bool(passed),
        "scores":vals,
        "average":round(avg,3),
        "thinking_types":[str(v) for v in blind.get("thinking_types",[])][:6],
        "fatal_flags":fatal,
        "reason":"[blind] "+str(blind.get("reason",""))+" [grounded] "+str(grounded.get("reason","")),
        "weakest_point":str(grounded.get("weakest_point","")),
        "blind_verdict":blind.get("verdict"),
        "grounded_verdict":grounded.get("verdict"),
    }

def judge_exam(api_key, model, exam, style_profile=""):
    rows=[]
    for q in exam.get("questions",[]):
        rows.append({
            "number":q.get("number"),"domain":q.get("domain"),"points":q.get("points"),
            "question_type":q.get("question_type"),"material_form":q.get("material_form"),
            "passage":str(q.get("passage",""))[:500],
            "tasks":q.get("tasks",[]),
            "quality":q.get("ai_quality",{}),
            "topic":q.get("topic",""),
        })
    prompt=f"""
너는 중등 기술 임용시험의 최종 편집위원이다.
개별 정답의 진위를 재판정하지 말고, 한 섹션 전체의 '시험다운 구성'만 평가한다.

실제 기출 구조:
{style_profile}

섹션:
{json.dumps(rows,ensure_ascii=False)}

REJECT 조건:
- 단순 회상/분류형이 과도하게 반복
- 4점 대부분이 같은 1+1+2 문법으로만 느껴짐
- 자료형이 겉모양만 다르고 실제 사고가 동일
- 특정 영역/사고행동이 과도하게 편중
- A/B 한 섹션으로서 실제 시험보다 현저히 단조로움

JSON만 출력:
{{
 "verdict":"PASS 또는 REJECT",
 "exam_realism":0,
 "variety":0,
 "difficulty_balance":0,
 "reason":"2~4문장"
}}
"""
    x=_ask_json(api_key,model,prompt,"medium")
    try:
        er=float(x.get("exam_realism",-1)); va=float(x.get("variety",-1)); db=float(x.get("difficulty_balance",-1))
    except Exception:
        return {"pass":False,"reason":"섹션 심사 파싱 실패","raw":x}
    passed=x.get("verdict")=="PASS" and min(er,va,db)>=4
    return {"pass":passed,"exam_realism":er,"variety":va,"difficulty_balance":db,"reason":str(x.get("reason",""))}


def judge_ab_pair(api_key, model, A, B, style_profile=""):
    def compact(exam):
        return [{
            "number":q.get("number"),"domain":q.get("domain"),"points":q.get("points"),
            "topic":q.get("topic",""),"question_type":q.get("question_type"),
            "tasks":q.get("tasks",[]),
            "thinking_types":q.get("ai_quality",{}).get("thinking_types",[]),
        } for q in exam.get("questions",[])]
    prompt=f"""
너는 중등 기술 임용 전공 A/B 최종 편집위원이다.
정답을 바꾸지 말고 A/B 두 책 전체의 구성 품질만 PASS/REJECT한다.

실제 기출 구조:
{style_profile}

A:
{json.dumps(compact(A),ensure_ascii=False)}

B:
{json.dumps(compact(B),ensure_ascii=False)}

REJECT:
- A/B 사이에서 concept 이름은 달라도 사실상 같은 원리/사고를 반복
- 같은 작성방법 문법이 지나치게 반복
- 4점 문항의 사고행동이 대부분 동일
- 영역은 분산됐지만 문제 경험이 단조로움
- 실제 임용 A/B 세트보다 현저히 단순함

JSON만:
{{
 "verdict":"PASS 또는 REJECT",
 "cross_section_variety":0,
 "semantic_duplication_control":0,
 "overall_exam_realism":0,
 "reason":"2~4문장"
}}
"""
    x=_ask_json(api_key,model,prompt,"medium")
    try:
        a=float(x.get("cross_section_variety",-1))
        b=float(x.get("semantic_duplication_control",-1))
        c=float(x.get("overall_exam_realism",-1))
    except Exception:
        return {"pass":False,"reason":"A/B 종합심사 파싱 실패","raw":x}
    passed=x.get("verdict")=="PASS" and min(a,b,c)>=4
    return {
        "pass":passed,
        "cross_section_variety":a,
        "semantic_duplication_control":b,
        "overall_exam_realism":c,
        "reason":str(x.get("reason",""))
    }
