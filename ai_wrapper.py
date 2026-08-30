
import json,re
from validators import fingerprint

def strip_json(text):
    t=text.strip()
    t=re.sub(r"^```(?:json)?\s*","",t)
    t=re.sub(r"\s*```$","",t)
    return t

def rewrite_bundle(api_key,model,bundle,source_context,points,section,pattern,question_type,style_profile):
    from openai import OpenAI
    client=OpenAI(api_key=api_key, timeout=45.0, max_retries=0)

    items=[]
    for i,a in enumerate(bundle,1):
        items.append(
          f"[채점요소 {i}] 고정정답={a['answer']} / 근거={a['evidence']} / "
          f"출처={a['source_name']} p.{a['page_no']}"
        )
    items_text="\n".join(items)
    subpoints=pattern["subpoints"]
    verbs=pattern["verbs"]

    prompt=f"""
너는 대한민국 중등 기술 임용시험 문항 '편집자'다. 정답을 새로 만들 권한은 없다.

[전공] {section}
[영역] {bundle[0]['domain']}
[문항유형] {question_type}
[배점] {points}점
[부분점수] {subpoints}
[요구행동] {verbs}

{items_text}

[원자료]
{source_context[:8000]}

[실제 임용 구조 프로필]
{style_profile}

절대 규칙
1. 고정정답과 각 근거를 변경하지 않는다.
2. 각 하위 작성요구는 반드시 대응하는 고정정답 하나를 평가해야 한다.
3. 원자료에 없는 수치, 조건, 교육과정 문구, 인과관계, 사례를 새로 만들어 넣지 않는다.
4. 서로 무관한 채점요소처럼 보이면 억지 상황을 만들지 말고, 같은 자료의 서로 관련된 항목을 확인하는 형태로 작성한다.
5. 정답 단어 자체를 문제 지문에 노출하지 않는다.
6. 2점은 짧고 명료하게, 4점은 실제 임용처럼 2~4개의 부분점수 요소를 가진다.
7. 그림이 실제로 제공되지 않으므로 '그림을 보고'라는 표현을 쓰지 않는다.
8. solution에는 원자료 근거만 사용한다.
9. JSON 하나만 출력한다.

JSON 스키마
{{
 "intro":"...",
 "passage":"...",
 "conditions":["..."],
 "tasks":["..."],
 "solution":["..."]
}}
"""
    r=client.responses.create(model=model,input=prompt,reasoning={"effort":"low"})
    x=json.loads(strip_json(r.output_text))

    q={
      "domain":bundle[0]["domain"],
      "topic":" · ".join(a["topic"] for a in bundle),
      "points":points,
      "subpoints":subpoints,
      "pattern_id":pattern["id"],
      "question_type":question_type,
      "verifier":"source",
      "intro":x.get("intro",""),
      "passage":x.get("passage",""),
      "conditions":x.get("conditions",[]),
      "tasks":x.get("tasks",[]),
      "answer":[a["answer"] for a in bundle],
      "solution":x.get("solution",[]),
      "evidence":[a["evidence"] for a in bundle],
      "sources":[{"source_name":a["source_name"],"page_no":a["page_no"]} for a in bundle],
      "source_basis":"; ".join(f"{a['source_name']} p.{a['page_no']}" for a in bundle)
    }
    q["fingerprint"]=fingerprint(q)
    return q

def safe_bundle_question(bundle,points,pattern,question_type):
    """AI가 실패해도 시험 전체가 깨지지 않는 원문기반 폴백. 부분점수 구조는 유지한다."""
    labels=["㉠","㉡","㉢","㉣"]
    passages=[]
    tasks=[]
    for i,a in enumerate(bundle):
        label=labels[i]
        ev=a["evidence"]
        ans=a["answer"]
        masked=re.sub(re.escape(ans),label,ev,flags=re.IGNORECASE)
        if masked==ev:
            masked=f"{label}에 해당하는 개념에 관한 설명: {ev}"
        passages.append(f"({chr(44032+i)}) {masked}")  # 가, 각... readability not critical
        tasks.append(f"{label}에 해당하는 용어 또는 내용을 쓸 것.")
    q={
      "domain":bundle[0]["domain"],
      "topic":" · ".join(a["topic"] for a in bundle),
      "points":points,
      "subpoints":pattern["subpoints"],
      "pattern_id":pattern["id"],
      "question_type":question_type,
      "verifier":"source",
      "intro":"다음은 같은 기술 영역의 관련 내용을 정리한 자료이다.",
      "passage":"\n\n".join(passages),
      "conditions":[],
      "tasks":tasks,
      "answer":[a["answer"] for a in bundle],
      "solution":[f"{labels[i]}: {a['answer']}" for i,a in enumerate(bundle)],
      "evidence":[a["evidence"] for a in bundle],
      "sources":[{"source_name":a["source_name"],"page_no":a["page_no"]} for a in bundle],
      "source_basis":"; ".join(f"{a['source_name']} p.{a['page_no']}" for a in bundle)
    }
    q["fingerprint"]=fingerprint(q)
    return q
