import json,re
from validators import validate_grounded_question,fingerprint
from generation_policy import target_instruction,looks_like_bad_anchor,concept_quality

def strip_json(text):
    t=text.strip(); t=re.sub(r"^```(?:json)?\s*","",t); t=re.sub(r"\s*```$","",t); return t

def _style_text(examples):
    if not examples:return ""
    return "\n".join("- "+re.sub(r"\s+"," ",str(x.get("text","")))[:800] for x in examples[:3])

def rewrite_anchor(api_key,model,anchor,source_context,points=4,section="A",difficulty="적당히 어려움",style_examples=None,pattern=None):
    if looks_like_bad_anchor(anchor):return None,["bad_anchor"]
    from openai import OpenAI
    client=OpenAI(api_key=api_key,timeout=40,max_retries=0)
    pattern=pattern or {"type":"data_interpret","pattern":{"label":"자료 해석·근거","material":"설명 자료"}}
    typ=pattern.get("type","data_interpret"); label=pattern.get("pattern",{}).get("label",typ)
    prompt=f'''너는 대한민국 중등 기술 임용시험 출제 형식 편집자다. 정답은 이미 확정되어 있으므로 새 정답을 만들지 마라.
[전공] {section} [영역] {anchor['domain']} [배점] {points} [문항유형] {label}
[고정 정답] {anchor['answer']}
[고정 근거] {anchor['evidence']}
[출처] {anchor['source_name']} p.{anchor['page_no']}
[원자료]\n{source_context[:3800]}
[업로드 모의고사 형식 참고 - 내용 복사 금지]\n{_style_text(style_examples)}

규칙:
- {target_instruction(difficulty,points,label)}
- 실제 모의고사처럼 자료/상황, 필요한 경우 <조건>, 그리고 <작성 방법>으로 구성한다.
- '{label}' 유형의 사고를 실제로 요구해야 한다. 모든 문제를 '개념명+특징' 형식으로 만들지 마라.
- 정답 용어를 지문에 그대로 노출하지 마라.
- 원자료에 없는 사실/수치/원인/효과를 창작하지 마라.
- 2점도 원문 한 줄을 그대로 가린 빈칸 문제로 만들지 마라.
- 4점은 서로 연결된 두 작성 요구를 기본으로 한다.
- 정답, evidence, source, page는 바꾸지 않는다.
- JSON 하나만 출력한다.
{{"domain":"{anchor['domain']}","topic":"{anchor['topic']}","points":{points},"verifier":"source","question_type":"{typ}","intro":"...","passage":"...","conditions":[],"tasks":["..."],"answer":["{anchor['answer']}"],"solution":["..."],"evidence":"{anchor['evidence']}","source_name":"{anchor['source_name']}","page_no":{anchor['page_no']},"source_basis":"{anchor['source_name']} p.{anchor['page_no']}"}}'''
    r=client.responses.create(model=model,input=prompt,reasoning={"effort":"low"})
    q=json.loads(strip_json(r.output_text))
    # AI가 correctness-critical 필드를 변경하지 못하게 덮어쓴다.
    q.update({"domain":anchor["domain"],"topic":anchor["topic"],"points":points,"verifier":"source","question_type":typ,
              "answer":[anchor["answer"]],"evidence":anchor["evidence"],"source_name":anchor["source_name"],"page_no":anchor["page_no"],
              "source_basis":f"{anchor['source_name']} p.{anchor['page_no']}"})
    q["fingerprint"]=fingerprint(q)
    errs=validate_grounded_question(q,source_context)
    if not concept_quality(q,points,difficulty,typ):errs.append("난이도/유형 기준 미달")
    return q,errs
