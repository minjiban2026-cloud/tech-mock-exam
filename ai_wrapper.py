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
    prompt=f"""
너는 대한민국 중등 기술 임용시험 문항 초안 작성자다.
정답은 이미 DB에서 고정되어 있으며 절대로 바꾸거나 새 정답을 만들면 안 된다.

전공 {section}, {points}점, 유형={question_type}, 자료형={material_form}
부분점수={pattern['subpoints']}
master concept={relation_meta.get('master_concept','')}
관계={relation_meta.get('relation','')}
권장 사고행동={relation_meta.get('thinking_types',[])}
출제 핵심지시={relation_meta.get('quality_directive','')}
실제 임용형 문항 골격={relation_meta.get('exam_skeleton','')}
Python 선결정 채점 논리={json.dumps(relation_meta.get('scoring_plan',[]),ensure_ascii=False)}

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
- 각 task는 해당 고정정답 요소와 명확하게 대응되어야 한다.
- 2점은 짧고 명확하게 쓰되 두 채점요소가 가능하면 하나의 판단관계 안에 있어야 한다.
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
