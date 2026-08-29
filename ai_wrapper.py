
import json, re
from validators import validate_grounded_question, fingerprint
from generation_policy import target_instruction, looks_like_bad_anchor, concept_quality

def strip_json(text):
    t=text.strip()
    t=re.sub(r"^```(?:json)?\s*","",t)
    t=re.sub(r"\s*```$","",t)
    return t

def rewrite_anchor(api_key, model, anchor, source_context, points=4, section="A", difficulty="적당히 어려움"):
    if looks_like_bad_anchor(anchor):
        return None, ["bad_anchor"]

    from openai import OpenAI
    client=OpenAI(api_key=api_key)

    prompt=f"""
너는 대한민국 중등 기술 임용시험 문항 편집자다.
너는 정답을 새로 결정하지 않는다. 아래 정답·근거·출처가 이미 확정되어 있다.

[전공] {section}
[영역] {anchor['domain']}
[고정 정답] {anchor['answer']}
[출처] {anchor['source_name']} p.{anchor['page_no']}
[고정 근거문장] {anchor['evidence']}

[원자료 문맥]
{source_context[:3500]}

목표:
- 위 원자료만 이용하여 새로운 임용형 상황/자료 문항으로 재구성한다.
- {target_instruction(difficulty, points)}
- 문제 지문에 고정 정답 자체를 직접 노출하지 않는다.
- 원문 목차나 단원명을 그대로 맞히게 하지 않는다.
- 정답 근거는 반드시 고정 근거문장과 원자료 문맥에서 확인 가능해야 한다.
- <조건>, <작성 방법>을 필요한 만큼만 사용한다.
- 정답은 "{anchor['answer']}"로 유지한다.
- evidence/source/page는 절대 변경하지 않는다.
- solution에는 왜 그 정답인지 원자료 근거를 짧게 설명한다.
- JSON 하나만 출력한다.

스키마:
{{
 "domain":"{anchor['domain']}",
 "topic":"{anchor['topic']}",
 "points":{points},
 "verifier":"source",
 "intro":"...",
 "passage":"...",
 "conditions":["..."],
 "tasks":["..."],
 "answer":["{anchor['answer']}"],
 "solution":["..."],
 "evidence":"{anchor['evidence']}",
 "source_name":"{anchor['source_name']}",
 "page_no":{anchor['page_no']},
 "source_basis":"..."
}}
"""
    r=client.responses.create(
        model=model,
        input=prompt,
        reasoning={"effort":"low"}
    )
    q=json.loads(strip_json(r.output_text))
    q["points"]=points
    q["fingerprint"]=fingerprint(q)

    errs=validate_grounded_question(q,source_context)
    if not concept_quality(q,points,difficulty):
        errs.append("문항 난이도/구조 기준 미달")
    return q, errs

def conservative_anchor_question(anchor, points=2):
    # 2점 문항에만 쓰는 안전 폴백.
    ev=anchor["evidence"]
    ans=anchor["answer"]
    masked=re.sub(re.escape(ans),"㉠",ev,flags=re.IGNORECASE)
    if masked==ev:
        masked=f"다음 설명에 해당하는 개념을 쓰시오. {ev}"
    q={
      "domain":anchor["domain"],"topic":anchor["topic"],"points":points,
      "verifier":"source",
      "intro":"다음은 기술 관련 개념에 관한 자료이다.",
      "passage":masked,
      "conditions":[],
      "tasks":["㉠에 해당하는 개념을 쓸 것."],
      "answer":[ans],
      "solution":[f"정답은 {ans}이다."],
      "evidence":ev,
      "source_name":anchor["source_name"],
      "page_no":anchor["page_no"],
      "source_basis":f"{anchor['source_name']} p.{anchor['page_no']}"
    }
    q["fingerprint"]=fingerprint(q)
    return q
