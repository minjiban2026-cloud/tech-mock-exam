import json,re
from validators import fingerprint

LABELS=["㉠","㉡","㉢","㉣"]
def _masked_evidence(bundle):
    # 한 근거문장 안에 다른 채점요소의 정답이 함께 등장하는 경우까지 모두 마스킹한다.
    answers=[str(a["answer"]).strip() for a in bundle]
    rows=[]
    for i,a in enumerate(bundle):
        ev=str(a["evidence"]).strip()
        masked=ev
        for j,ans in enumerate(answers):
            masked=re.sub(re.escape(ans),LABELS[j],masked,flags=re.I)
        if masked==ev and answers[i]:
            masked=f"{LABELS[i]}에 관한 원문 근거: {ev.replace(answers[i],'[정답란]')}"
        rows.append(masked)
    return rows
def build_source_passage(bundle,material_form):
    rows=_masked_evidence(bundle)
    if material_form=="대화자료":
        speakers=["교사","학생 A","학생 B","교사"]
        return "\n".join(f"{speakers[i%len(speakers)]}: {r}" for i,r in enumerate(rows))
    if material_form=="과정자료":
        return "\n".join(f"{i+1}단계. {r}" for i,r in enumerate(rows))
    if material_form=="표형자료":
        return "\n".join(f"자료 {i+1} | {r}" for i,r in enumerate(rows))
    return "\n".join(f"◦ {r}" for r in rows)
def _strip_json(text):
    t=text.strip()
    t=re.sub(r"^```(?:json)?\s*","",t)
    t=re.sub(r"\s*```$","",t)
    return t

def rewrite_bundle(api_key,model,bundle,points,section,pattern,question_type,material_form,style_profile,
                   source_context="",relation_meta=None):
    from openai import OpenAI
    client=OpenAI(api_key=api_key,timeout=60,max_retries=1)
    answers=[a["answer"] for a in bundle]
    relation_meta=relation_meta or {}
    evidence=[a["evidence"] for a in bundle]
    source_context=source_context or "\n".join(evidence)
    selection_mode=str(relation_meta.get("selection_mode",""))
    if int(points)==2 and selection_mode=="python_exam_value_one_anchor_t2":
        # ONE-ANCHOR T2는 페이지 전체의 다른 항목을 섞지 않는다.
        # 중심 anchor + 같은 anchor의 채점근거만 Writer에게 제공한다.
        source_context="\n".join(str(x) for x in evidence if str(x).strip())
    prompt=f"""
너는 대한민국 중등 기술 임용시험 문항 초안 작성자다.
정답과 채점근거는 DB 원문을 바탕으로 Python에서 고정되어 있으며 절대로 바꾸거나 새 정답을 만들면 안 된다.

전공 {section}, {points}점, 유형={question_type}, 자료형={material_form}
부분점수={pattern['subpoints']}
master concept={relation_meta.get('master_concept','')}
관계={relation_meta.get('relation','')}
권장 사고행동={relation_meta.get('thinking_types',[])}
출제 핵심지시={relation_meta.get('quality_directive','')}
실제 임용형 문항 골격={relation_meta.get('exam_skeleton','')}
Python 선결정 채점 논리={json.dumps(relation_meta.get('scoring_plan',[]),ensure_ascii=False)}
자료 길이 제한={json.dumps(relation_meta.get('material_limits',{}),ensure_ascii=False)}
자연적 문제단위 점수={relation_meta.get('natural_unit_score','')}
2점 후보 정책={relation_meta.get('two_point_label_policy','')}\n선택 모드={selection_mode}
임용 핵심도 프로필={json.dumps(relation_meta.get('core_exam_profile',[]),ensure_ascii=False)}

고정정답:
{json.dumps(answers,ensure_ascii=False)}

각 정답의 원문 근거:
{json.dumps(evidence,ensure_ascii=False)}

해당 출처 문맥:
{source_context[:9000]}

실제 기출 구조:
{style_profile}
반드시 지킬 규칙:
- passage/conditions/tasks만 작성한다. 정답은 수정하지 않는다.
- 기술적 사실, 수치, 인과관계는 위 출처 문맥이 직접 뒷받침하는 것만 사용한다.
- 출처에 없는 실제 사례, 장치 조건, 수치, 효과를 꾸며내지 않는다.
- 중립적 수업 상황 프레임("교사가 자료를 제시했다", "학생이 검토했다")은 허용하되 기술 사실을 추가하지 않는다.
- 정답 단어를 passage/conditions/tasks에 직접 쓰지 않는다.
- 원문 정의를 정답 단어만 빈칸 처리한 채 거의 그대로 제시하지 않는다.
- 한 문장만 읽고 정답을 그대로 복원할 수 있는 '정의 베껴쓰기' 문제를 만들지 않는다.
- 자료 전체를 해석해야 하도록 단서를 분산하되, 고정정답을 도출할 정보는 충분히 남긴다.
- 고정정답이 방정식/원리/법칙의 명칭일 때 완성된 식이나 정의를 그대로 제시하고 그 이름만 묻는 방식은 금지한다.
- 4점은 하나의 coherent scenario 안에서 앞의 해석·판단이 뒤의 관계설명·적용에 쓰이는 사고 사슬이어야 한다.
- 4점의 소문항들이 서로 없어도 풀리는 독립된 1점/2점 암기문항 묶음이면 안 된다.
- T4_DATA112/T4_112의 마지막 task는 반드시 '앞의 결과', '이를 이용하여', '위 판단을 근거로'처럼 앞선 판단을 실제로 사용하도록 명시한다.
- T4_ERR22는 두 오류를 병렬로 고치는 데 그치지 말고 같은 원리/공통 근거로 연결한다.
- 자료에서 한 문장 또는 한 표의 한 행만 읽으면 답을 바로 옮길 수 있는 요구는 만들지 않는다.
- 1점 채점요소도 가능하면 단순 명칭 회상이 아니라 조건 판별·중간값·오류 판단 등 뒤 추론에 필요한 중간 판단으로 만든다.
- 4점은 최소 두 종류 이상의 사고행동(식별/관계설명/오류판단/비교/적용/계산/수정)을 포함해야 한다.
- 4점의 세 하위 요구가 모두 '용어 쓰기' 또는 '분류하기'이면 안 된다.
- 같은 요구를 표현만 바꾸어 반복하지 않는다.
- 각 task는 같은 순번의 고정정답과 원문 근거에서 채점 가능한 내용만 요구한다. 고정정답/근거 밖의 새 결과·효과·원인·힘·압력·작용을 추가로 요구하지 않는다.
- 부분점수 1점짜리 task는 채점행동을 정확히 1개만 요구한다. "명칭을 쓰고 이유를 설명", "값을 구하고 원리를 쓰기", "분류하고 근거 설명"처럼 두 답을 동시에 요구하면 안 된다.
- 특히 1점 task에서 '쓰고/제시하고/구하고/판단하고/분류하고 ... 설명하시오/쓰시오' 구조를 만들지 않는다.
- 부분점수 2점짜리 task만 하나의 고정정답을 중심으로 필요한 근거 설명·비교·적용을 함께 요구할 수 있다.
- 자료에서 직접 뒷받침되지 않는 후속 물리현상으로 확장하지 않는다. 예: 고정정답이 '유체의 운동량'이면 원문에 없는 관벽의 힘·압력·충격까지 새로 묻지 않는다.
- 2점은 짧고 명확하게 쓰되 두 채점요소가 가능하면 하나의 판단관계 안에 있어야 한다.
- 2점은 ONE-ANCHOR-DISTINCT 구조를 사용한다. 첫 번째 고정답은 중심개념이고, 두 번째 고정답은 같은 DB 원문에서 Python이 선별한 별도의 조건·비교·절차·효과·적용 채점근거다.
- 2점에서 두 번째 고정답을 별도의 개념명처럼 묻지 않는다. 또한 첫 명칭을 맞히게 한 동일 특징을 반대로 고치거나 그대로 반복 설명하게 하지 않는다. 두 번째 1점은 반드시 별도 조건·비교·절차·효과·적용 판단으로 채점되게 한다.
- 2점은 '서로 다른 개념 2개를 억지로 연결'하지 않는다.
- 첫 요구가 명칭/용어 판단이라면 두 번째 요구는 반드시 같은 핵심개념을 전제로 한 판단·근거·오류수정·비교·적용 중 하나가 되게 한다.
- 2점의 두 요구가 모두 명칭/용어 회상이면 안 된다.
- ONE-ANCHOR 2점에서는 passage에 독립 사실을 나열하지 말고, 중심개념을 판단하는 상황 1개와 두 번째 채점근거에 필요한 정보 1개만 쓴다.
- ONE-ANCHOR 2점의 passage에는 첫 고정정답을 다른 유사개념과 구별할 수 있는 핵심 특징/조건을 최소 1개 포함한다. 단, 정답명이나 원문 정의 전체를 그대로 노출하지 않는다.
- 두 번째 task는 첫 판단을 전제로 같은 anchor의 근거/특징/오류수정/적용을 묻게 한다. 별도 개념을 새로 묻지 않는다.
- ONE-ANCHOR 2점에서는 A/B/C 사례 나열, ①②③ 열거, 여러 종류·장점·활용처의 병렬 나열을 금지한다.
- 2점 자료는 필요한 정보만 짧게 제시한다. 난도를 올리기 위해 독립 사실·사례·활용처를 여러 개 나열하지 않는다.
- 4점은 원자료에 원래 존재하는 하나의 과정/장치/현상/계산관계를 중심으로 만든다. 독립 개념 3개를 억지로 한 지문에 합치지 않는다.
- 4점 자료에는 고정정답의 정의·조건·수치관계를 완성된 문장으로 그대로 나열하지 않는다. 수험생이 최소 한 번 계산·비교·판단해야 답이 나오게 한다.
- 난도는 지문 길이와 정보량이 아니라 사고과정으로 만든다.
- 정답은 CORE/NORMAL을 우선한다. SUPPORT는 자료·상황·오답·설명용으로 활용할 수 있지만, 세부 고유명·특수사례 자체를 주된 정답으로 요구하지 않는다.
- 자료형은 실제 답 풀이에 필요한 형태여야 하고 장식용이면 안 된다.
- JSON 하나만 출력한다.
{{
 "intro":"...",
 "passage":"...",
 "conditions":["..."],
 "tasks":["..."],
 "thinking_types":["..."]
}}
"""
    r=client.responses.create(model=model,input=prompt,reasoning={"effort":("medium" if points==4 else "low")})
    x=json.loads(_strip_json(r.output_text))
    q={
      "domain":bundle[0]["domain"],
      "topic":" · ".join(a["topic"] for a in bundle),
      "points":points,"subpoints":pattern["subpoints"],"pattern_id":pattern["id"],
      "question_type":question_type,"material_form":material_form,
      "verifier":"source",
      "intro":x.get("intro","다음 자료를 읽고 <작성 방법>에 따라 쓰시오."),
      "passage":str(x.get("passage","")).strip(),
      "conditions":[str(v).strip() for v in x.get("conditions",[]) if str(v).strip()],
      "tasks":[str(v).strip() for v in x.get("tasks",[]) if str(v).strip()],
      "answer":answers,
      "solution":[f"{LABELS[i]}: {a['answer']}" for i,a in enumerate(bundle)],
      "evidence":evidence,
      "sources":[{"source_name":a["source_name"],"page_no":a["page_no"]} for a in bundle],
      "source_basis":"; ".join(f"{a['source_name']} p.{a['page_no']}" for a in bundle),
      "premise_mode":"ai_grounded",
      "master_concept":relation_meta.get("master_concept",""),
      "relation":relation_meta.get("relation",""),
      "selection_mode":selection_mode,
      "intended_thinking_types":[str(v) for v in x.get("thinking_types",[])],
    }
    q["fingerprint"]=fingerprint(q)
    return q
def safe_bundle_question(bundle,points,pattern,question_type,material_form):
    passage=build_source_passage(bundle,material_form)
    q={
      "domain":bundle[0]["domain"],
      "topic":" · ".join(a["topic"] for a in bundle),
      "points":points,"subpoints":pattern["subpoints"],"pattern_id":pattern["id"],
      "question_type":question_type,"material_form":material_form,
      "verifier":"source",
      "intro":"다음 자료를 읽고 <작성 방법>에 따라 쓰시오.",
      "passage":passage,"conditions":[],
      "tasks":[
          (f"{LABELS[i]}에 해당하는 용어 또는 내용을 쓸 것."
           if pattern["subpoints"][i]==1 else
           f"{LABELS[i]}에 해당하는 용어 또는 내용을 쓰고, 자료에 제시된 특징·이유·관계 중 해당하는 내용을 근거로 함께 설명할 것.")
          for i in range(len(bundle))
      ],
      "answer":[a["answer"] for a in bundle],
      "solution":[
          (f"{LABELS[i]}: {a['answer']}"
           if pattern["subpoints"][i]==1 else
           f"{LABELS[i]}: {a['answer']} / 근거: {a['evidence']}")
          for i,a in enumerate(bundle)
      ],
      "evidence":[a["evidence"] for a in bundle],
      "sources":[{"source_name":a["source_name"],"page_no":a["page_no"]} for a in bundle],
      "source_basis":"; ".join(f"{a['source_name']} p.{a['page_no']}" for a in bundle),
      "premise_mode":"source_locked"
    }
    q["fingerprint"]=fingerprint(q)
    return q
