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
    return OpenAI(api_key=api_key, timeout=30, max_retries=0)

def _ask_json(api_key, model, prompt, effort="medium"):
    r=_client(api_key).responses.create(model=model,input=prompt,reasoning={"effort":effort})
    return json.loads(_strip_json(r.output_text))

def select_coherent_bundle(api_key, model, candidates, points, style_profile, need=None):
    items=[]
    for i,a in enumerate(candidates):
        items.append({
            "index":i,"topic":a.get("topic",""),"answer":a.get("answer",""),
            "evidence":a.get("evidence",""),"source_name":a.get("source_name",""),
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
{style_profile}

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
    public_q={
        "domain":question.get("domain"),"points":question.get("points"),
        "question_type":question.get("question_type"),"material_form":question.get("material_form"),
        "intro":question.get("intro"),"passage":question.get("passage"),
        "conditions":question.get("conditions",[]),"tasks":question.get("tasks",[]),
    }
    blind_prompt=f"""
너는 대한민국 중등 기술 임용 1차 전공시험의 외부 출제 검토위원이다.
정답과 출처를 보지 않고 실제 수험생처럼 문제만 평가한다. 수정하지 말고 PASS/REJECT만 판정한다.

실제 기출 구조:
{style_profile}

문항:
{json.dumps(public_q,ensure_ascii=False)}

강한 REJECT:
- 2점이라도 명칭 두 개를 각 줄에서 찾아 쓰는 독립 회상 문제
- 하위 요구가 모두 같은 회상 행동
- 자료를 종합할 필요가 없음
- 자료가 정답 정의/고유특징을 거의 그대로 말해 추론거리가 없음
- 문제유형 이름과 실제 요구행동이 불일치
- 서로 다른 요구 사이에 논리적 연결이 없음
- 배점 대비 너무 쉽거나 모호함
- 4점인데 사실상 정의/용어를 옮겨 쓰는 수준

0~5:
inferential_distance, task_distinctness, exam_realism, difficulty_fit, ambiguity_control

JSON만 출력:
{{
 "verdict":"PASS 또는 REJECT",
 "scores":{{
  "inferential_distance":0,"task_distinctness":0,"exam_realism":0,
  "difficulty_fit":0,"ambiguity_control":0
 }},
 "thinking_types":["식별","관계설명","적용" 등],
 "fatal_flags":["ROTE_ONLY","DECORATIVE_MATERIAL","TOO_EASY","AMBIGUOUS","UNRELATED_SUBPARTS" 중 해당되는 것만],
 "reason":"2~4문장"
}}
"""
    blind=_ask_json(api_key,model,blind_prompt,"low")

    # Blind 단계에서 이미 탈락이 확정되면 Grounded 호출을 생략한다.
    # 품질 기준은 그대로이고, 불필요한 두 번째 AI 호출만 제거한다.
    try:
        _bs=blind.get("scores",{})
        _blind_vals={
            "inferential_distance":float(_bs.get("inferential_distance",-1)),
            "task_distinctness":float(_bs.get("task_distinctness",-1)),
            "exam_realism":float(_bs.get("exam_realism",-1)),
            "difficulty_fit":float(_bs.get("difficulty_fit",-1)),
            "ambiguity_control":float(_bs.get("ambiguity_control",-1)),
        }
    except Exception:
        return {"pass":False,"review_stage":"blind","reason":"Blind AI 심사 점수 파싱 실패",
                "blind_raw":blind,"blind_verdict":blind.get("verdict")}

    _blind_fatal=[str(f) for f in blind.get("fatal_flags",[]) if str(f).strip()]
    _pts=int(question.get("points",0))
    if _pts==4:
        _blind_threshold_ok=(
            _blind_vals["inferential_distance"]>=3.5 and
            _blind_vals["task_distinctness"]>=4 and
            _blind_vals["exam_realism"]>=4 and
            _blind_vals["difficulty_fit"]>=4 and
            _blind_vals["ambiguity_control"]>=4
        )
    else:
        _blind_threshold_ok=(
            _blind_vals["exam_realism"]>=3.5 and
            _blind_vals["ambiguity_control"]>=4
        )

    if blind.get("verdict")!="PASS" or _blind_fatal or not _blind_threshold_ok:
        return {
            "pass":False,"review_stage":"blind",
            "scores":_blind_vals,"fatal_flags":_blind_fatal,
            "thinking_types":[str(v) for v in blind.get("thinking_types",[])][:6],
            "reason":"[blind] "+str(blind.get("reason","")),
            "weakest_point":"Blind 심사 단계에서 이미 탈락 확정",
            "blind_verdict":blind.get("verdict"),"grounded_verdict":"SKIPPED"
        }

    grounded_q=dict(public_q)
    grounded_q.update({
        "fixed_answer":question.get("answer",[]),"evidence":question.get("evidence",[]),
        "solution":question.get("solution",[]),"master_concept":question.get("master_concept",""),
        "relation":question.get("relation",""),"verifier":question.get("verifier"),
    })
    grounding_prompt=f"""
너는 중등 기술 임용시험의 사실검증·채점 검토위원이다.
정답은 DB/Python이 이미 고정했다. 새 정답을 만들지 말고 품질에 veto만 행사한다.

실제 기출 구조:
{style_profile}

고정 원자료 문맥:
{source_context[:9000]}

문항+고정정답:
{json.dumps(grounded_q,ensure_ascii=False)}

강한 REJECT:
- 원자료가 뒷받침하지 않는 사실·수치·인과관계 추가
- 정답 용어만 가리고 원문 정의/고유특징을 거의 그대로 제시
- anchor들이 하나의 master concept가 아님
- 작성 방법과 고정정답이 일대일 대응하지 않음
- 복수 정답/채점 모호성이 큼

0~5:
grounding, answer_leakage(5=노출 없음), coherence, ambiguity_control

JSON만 출력:
{{
 "verdict":"PASS 또는 REJECT",
 "scores":{{"grounding":0,"answer_leakage":0,"coherence":0,"ambiguity_control":0}},
 "fatal_flags":["DIRECT_ANSWER_LEAK","UNSUPPORTED_FACT","UNRELATED_SUBPARTS","AMBIGUOUS" 중 해당되는 것만],
 "reason":"2~4문장",
 "weakest_point":"가장 큰 약점 한 문장"
}}
"""
    grounded=_ask_json(api_key,model,grounding_prompt,"low")

    try:
        bs=blind.get("scores",{}); gs=grounded.get("scores",{})
        vals={
            "grounding":float(gs.get("grounding",-1)),
            "answer_leakage":float(gs.get("answer_leakage",-1)),
            "coherence":float(gs.get("coherence",-1)),
            "inferential_distance":float(bs.get("inferential_distance",-1)),
            "task_distinctness":float(bs.get("task_distinctness",-1)),
            "exam_realism":float(bs.get("exam_realism",-1)),
            "difficulty_fit":float(bs.get("difficulty_fit",-1)),
            "ambiguity_control":min(float(bs.get("ambiguity_control",-1)),float(gs.get("ambiguity_control",-1))),
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
        passed=(blind_pass and grounded_pass and vals["grounding"]>=4 and vals["answer_leakage"]>=4
                and vals["coherence"]>=4 and vals["inferential_distance"]>=3.5
                and vals["task_distinctness"]>=4 and vals["exam_realism"]>=4
                and vals["difficulty_fit"]>=4 and vals["ambiguity_control"]>=4
                and avg>=4.05 and not fatal)
    else:
        passed=(blind_pass and grounded_pass and vals["grounding"]>=4 and vals["answer_leakage"]>=4
                and vals["ambiguity_control"]>=4 and vals["exam_realism"]>=3.5
                and avg>=3.7 and not fatal)

    return {
        "pass":bool(passed),"review_stage":"grounded","scores":vals,"average":round(avg,3),
        "thinking_types":[str(v) for v in blind.get("thinking_types",[])][:6],
        "fatal_flags":fatal,
        "reason":"[blind] "+str(blind.get("reason",""))+" [grounded] "+str(grounded.get("reason","")),
        "weakest_point":str(grounded.get("weakest_point","")),
        "blind_verdict":blind.get("verdict"),"grounded_verdict":grounded.get("verdict"),
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
