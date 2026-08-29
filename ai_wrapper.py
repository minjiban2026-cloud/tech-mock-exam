
import json, re
from validators import validate_grounded_question, fingerprint
from generation_policy import target_instruction, looks_like_bad_anchor, concept_quality

def strip_json(text):
    t=text.strip()
    t=re.sub(r"^```(?:json)?\s*","",t)
    t=re.sub(r"\s*```$","",t)
    return t

def _style_text(style_examples):
    if not style_examples:
        return ""
    bits=[]
    for s in style_examples[:2]:
        txt=re.sub(r"\s+"," ",str(s.get("text","")))[:700]
        bits.append(f"- {txt}")
    return "\n".join(bits)

def rewrite_anchor(api_key, model, anchor, source_context, points=4, section="A",
                   difficulty="적당히 어려움", style_examples=None):
    if looks_like_bad_anchor(anchor):
        return None, ["bad_anchor"]

    from openai import OpenAI
    client=OpenAI(api_key=api_key, timeout=35.0, max_retries=0)

    prompt=f"""
너는 대한민국 중등 기술 임용시험의 문항 편집자다.
정답을 새로 추론하거나 바꾸지 말고, 아래에서 고정된 정답과 원자료만 사용하라.

[전공] {section}
[영역] {anchor['domain']}
[고정 정답] {anchor['answer']}
[출처] {anchor['source_name']} p.{anchor['page_no']}
[고정 근거] {anchor['evidence']}

[원자료 문맥]
{source_context[:3200]}

[실제 모의고사 형식 참고]
{_style_text(style_examples)}

작성 원칙:
1. 실제 임용형의 '자료 → <조건> → <작성 방법>' 흐름을 참고하되 참고문항의 문장이나 내용을 복사하지 않는다.
2. {target_instruction(difficulty, points)}
3. 문제 지문에 고정 정답 용어를 그대로 쓰지 않는다.
4. 원자료에서 확인할 수 없는 사실·수치·조건·인과관계를 새로 만들지 않는다.
5. 정답과 evidence/source/page는 절대 변경하지 않는다.
6. 4점이면 같은 자료를 이용하는 2개의 작성 요구를 만든다.
   두 번째 요구는 첫 번째와 무관한 새 지식을 요구하면 안 된다.
7. 해설은 원자료 근거에 한정해 짧고 명확하게 쓴다.
8. JSON 하나만 출력한다.

JSON:
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
 "source_basis":"{anchor['source_name']} p.{anchor['page_no']}"
}}
"""
    r=client.responses.create(
        model=model,
        input=prompt,
        reasoning={"effort":"low"}
    )
    q=json.loads(strip_json(r.output_text))
    # correctness-critical fields are overwritten, not trusted from AI
    q["domain"]=anchor["domain"]
    q["topic"]=anchor["topic"]
    q["points"]=points
    q["verifier"]="source"
    q["answer"]=[anchor["answer"]]
    q["evidence"]=anchor["evidence"]
    q["source_name"]=anchor["source_name"]
    q["page_no"]=anchor["page_no"]
    q["source_basis"]=f"{anchor['source_name']} p.{anchor['page_no']}"
    q["fingerprint"]=fingerprint(q)

    errs=validate_grounded_question(q,source_context)
    if not concept_quality(q,points,difficulty):
        errs.append("문항 난이도/구조 기준 미달")
    return q,errs

def conservative_anchor_question(anchor, points=2):
    """2점용 안전 폴백. 원자료의 한 개념만 확인한다."""
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
      "solution":[f"㉠은 {ans}이다."],
      "evidence":ev,
      "source_name":anchor["source_name"],
      "page_no":anchor["page_no"],
      "source_basis":f"{anchor['source_name']} p.{anchor['page_no']}"
    }
    q["fingerprint"]=fingerprint(q)
    return q

def paired_anchor_question(a1, a2):
    """
    AI 실패 시 4점 문항 생성이 중단되지 않도록 하는 완전 원문근거형 폴백.
    서로 다른 두 개념을 (가),(나) 자료로 제시해 각각 판별한다.
    """
    def mask(ev,ans,label):
        m=re.sub(re.escape(ans),label,ev,flags=re.IGNORECASE)
        if m==ev:
            return f"{label}에 관한 설명: {ev}"
        return m

    p1=mask(a1["evidence"],a1["answer"],"㉠")
    p2=mask(a2["evidence"],a2["answer"],"㉡")
    evidence=f"{a1['evidence']}\n{a2['evidence']}"
    q={
      "domain":a1["domain"],
      "topic":f"{a1['topic']}·{a2['topic']}",
      "points":4,
      "verifier":"source",
      "intro":"다음 (가), (나)는 같은 기술 영역에서 다루는 두 개념에 관한 자료이다.",
      "passage":f"(가) {p1}\n\n(나) {p2}",
      "conditions":["각 자료는 제시된 원자료의 의미 범위 안에서 판단한다."],
      "tasks":["㉠에 해당하는 개념을 쓸 것.","㉡에 해당하는 개념을 쓸 것."],
      "answer":[a1["answer"],a2["answer"]],
      "solution":[f"㉠: {a1['answer']}",f"㉡: {a2['answer']}"],
      "evidence":evidence,
      "source_name":f"{a1['source_name']} / {a2['source_name']}",
      "page_no":f"{a1['page_no']} / {a2['page_no']}",
      "source_basis":f"{a1['source_name']} p.{a1['page_no']}; {a2['source_name']} p.{a2['page_no']}"
    }
    q["fingerprint"]=fingerprint(q)
    return q
