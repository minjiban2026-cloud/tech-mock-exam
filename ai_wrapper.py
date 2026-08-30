
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

def rewrite_bundle(api_key,model,bundle,points,section,pattern,question_type,material_form,style_profile):
    from openai import OpenAI
    client=OpenAI(api_key=api_key,timeout=45,max_retries=0)
    passage=build_source_passage(bundle,material_form)
    answers=[a["answer"] for a in bundle]
    prompt=f"""
대한민국 중등 기술 임용시험의 '작성 방법'만 편집한다.
사실 지문은 프로그램이 원문 근거로 이미 고정했으며 절대로 수정하지 않는다.

전공 {section}, {points}점, 유형={question_type}, 자료형={material_form}
부분점수={pattern['subpoints']}
고정정답={answers}
고정지문:
{passage}

실제 기출 구조:
{style_profile}

규칙:
- tasks 개수는 정확히 {len(pattern['subpoints'])}개.
- task i는 고정정답 i 하나를 평가한다.
- 4점은 최소 3개의 서로 다른 채점 요구를 가져야 한다.
- 2점짜리 하위요소는 단순 용어 쓰기만 시키지 말고, 고정지문에서 확인 가능한 특징·이유·관계·적용 중 하나를 함께 요구한다.
- 같은 동사와 같은 요구를 표현만 바꾸어 반복하지 않는다.
- 가능한 경우 식별 → 관계/비교 → 적용/설명의 흐름으로 구성한다.
- 원자료에 없는 숫자, 사실, 사례, 조건, 인과관계를 추가하지 않는다.
- 정답을 task 안에 직접 쓰지 않는다.
- 2점은 짧게, 4점은 부분점수별 요구가 분명하게 보이게 한다.
- JSON 하나만 출력한다.

{{"intro":"...", "tasks":["..."], "solution_labels":["..."]}}
"""
    r=client.responses.create(model=model,input=prompt,reasoning={"effort":"low"})
    x=json.loads(_strip_json(r.output_text))
    q={
      "domain":bundle[0]["domain"],
      "topic":" · ".join(a["topic"] for a in bundle),
      "points":points,"subpoints":pattern["subpoints"],"pattern_id":pattern["id"],
      "question_type":question_type,"material_form":material_form,
      "verifier":"source",
      "intro":x.get("intro","다음 자료를 읽고 <작성 방법>에 따라 쓰시오."),
      "passage":passage,
      "conditions":[],
      "tasks":x.get("tasks",[]),
      "answer":answers,
      "solution":[f"{x.get('solution_labels',['정답']*len(bundle))[i] if i < len(x.get('solution_labels',[])) else '정답'}: {a['answer']}" for i,a in enumerate(bundle)],
      "evidence":[a["evidence"] for a in bundle],
      "sources":[{"source_name":a["source_name"],"page_no":a["page_no"]} for a in bundle],
      "source_basis":"; ".join(f"{a['source_name']} p.{a['page_no']}" for a in bundle),
      "premise_mode":"source_locked"
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
