
import json, re
from validators import validate_grounded_question, fingerprint
from generation_policy import target_instruction

def strip_json(text):
    t=text.strip()
    t=re.sub(r"^```(?:json)?\s*","",t)
    t=re.sub(r"\s*```$","",t)
    return t

def rewrite_anchor(api_key, model, anchor, source_context, points=4, section="A", difficulty="적당히 어려움"):
    from openai import OpenAI
    client=OpenAI(api_key=api_key)
    prompt=f"""
너는 대한민국 중등 기술 임용시험의 문항 문장 편집자다.
중요: 너는 정답을 결정하는 출제자가 아니다. 정답과 근거는 프로그램이 이미 고정했다.
다음 고정정보를 절대 변경하지 마라.

[전공] {section}
[영역] {anchor['domain']}
[고정 정답] {anchor['answer']}
[출처] {anchor['source_name']} p.{anchor['page_no']}
[근거 문장] {anchor['evidence']}

[원자료 문맥]
{source_context[:8000]}

이 정답을 수험자가 자료를 해석해서 찾아내도록, 실제 임용형 상황문항으로 바꾸어라.
난이도 지침: {target_instruction(difficulty)}
규칙:
- 문제 지문이나 제목에 고정 정답을 직접 쓰지 마라.
- 원자료 문맥 밖의 새로운 사실을 정답 근거로 사용하지 마라.
- 정답은 정확히 "{anchor['answer']}" 하나로 유지한다.
- evidence는 위 [근거 문장]을 한 글자도 바꾸지 않고 그대로 넣는다.
- 출처 파일명과 페이지를 그대로 유지한다.
- {points}점 문항으로 작성한다.
- 4점이면 가능할 때 정답 용어 식별 + 근거가 되는 특징 1개를 쓰게 하되, 특징의 정답도 반드시 evidence 안에서 직접 확인 가능해야 한다.
- <조건>과 <작성 방법>을 명확히 한다.
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
        reasoning={"effort":"high"}
    )
    q=json.loads(strip_json(r.output_text))
    q["fingerprint"]=fingerprint(q)
    errs=validate_grounded_question(q,source_context)
    return q, errs

def conservative_anchor_question(anchor, points=2):
    # AI 실패 시: 원자료에 근거한 보수적 식별 문항
    ev=anchor["evidence"]
    ans=anchor["answer"]
    masked=re.sub(re.escape(ans),"㉠",ev,flags=re.IGNORECASE)
    if masked==ev:
        # term이 정확히 evidence에 없으면 첫 출현 부분을 문제 상황으로 사용
        masked=f"다음 설명에 해당하는 용어를 쓰시오. {ev}"
    q={
      "domain":anchor["domain"],"topic":anchor["topic"],"points":points,
      "verifier":"source",
      "intro":"다음은 기술 관련 개념에 관한 자료이다.",
      "passage":masked,
      "conditions":[],
      "tasks":["㉠에 해당하는 용어를 쓸 것."],
      "answer":[ans],
      "solution":[f"정답은 {ans}이다."],
      "evidence":ev,
      "source_name":anchor["source_name"],
      "page_no":anchor["page_no"],
      "source_basis":f"{anchor['source_name']} p.{anchor['page_no']}"
    }
    q["fingerprint"]=fingerprint(q)
    return q
