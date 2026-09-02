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
    selection_mode=str(relation_meta.get("selection_mode",""))
    derived_answer_flags=[False]*len(answers)
    if int(points)==2 and selection_mode in {
        "python_exam_value_decision_first_t2",
        "python_exam_value_t2_reasoning_matrix",
    }:
        answers=[str(relation_meta.get("correct_option","㉠")), answers[1]]
        derived_answer_flags=[True,False]
    source_context=source_context or "\n".join(evidence)
    if int(points)==2 and selection_mode=="python_exam_value_t2_reasoning_matrix":
        # R28: Python이 이미 고정한 세 사실만 Writer에게 제공한다.
        _spec=relation_meta.get("reasoning_spec") or {}
        _parts=[
            "[중심 사실] "+str(_spec.get("base_fact","")),
            "[연결 사실] "+str(_spec.get("linked_fact","")),
            "[혼합 오답용 sibling 사실] "+str(_spec.get("distractor_fact","")),
        ]
        source_context="\n".join(x for x in _parts if x.strip())
    elif int(points)==2 and selection_mode=="python_exam_value_decision_first_t2":
        _parts=[str(x) for x in evidence if str(x).strip()]
        _cc=str(relation_meta.get("contrast_context","") or "").strip()
        if _cc:
            _parts.append("[비교용 비정답 원문] "+_cc)
        source_context="\n".join(_parts)
    elif int(points)==2 and selection_mode in {
        "python_exam_value_one_anchor_t2",
        "python_exam_value_one_anchor_distinct_t2",
        "python_exam_value_one_anchor_inference_t2",
    }:
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
2점 후보 정책={relation_meta.get('two_point_label_policy','')}
2점 Python 사고논리={json.dumps(relation_meta.get('reasoning_spec',{}),ensure_ascii=False)}
2점 비교용 비정답 원문={relation_meta.get('contrast_context','')}
비교용 비정답 개념={relation_meta.get('contrast_answer','')}
숨은 중심개념(정답으로 직접 묻지 않음)={relation_meta.get('hidden_core_answer','')}
정답 옵션={relation_meta.get('correct_option','')}
선택 모드={selection_mode}
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
- 2점 reasoning-matrix에서는 case_correct와 case_wrong도 반드시 각각 작성한다. case_correct는 base_fact+linked_fact의 정답 사례, case_wrong은 base_fact+distractor_fact의 오답 사례다. 두 필드의 역할을 서로 바꾸지 않는다.
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
- 자료에서 직접 뒷받침되지 않는 후속 현상·효과·힘·조건으로 확장하지 않는다. 고정정답의 원문 근거 밖의 내용을 새로 묻지 않는다.
- 2점은 짧고 명확하게 쓰되 두 채점요소가 하나의 사고사슬이어야 한다.
- selection_mode가 python_exam_value_t2_reasoning_matrix이면 Python 사고논리를 그대로 따른다.
- 이 모드에서는 passage에 반드시 ㉠과 ㉡ 두 익명 사례를 제시한다. 두 사례 모두 기술 개념명을 직접 쓰지 않는다.
- 정답 사례는 base_fact와 linked_fact가 같은 대상/원리에서 함께 성립하도록 의미를 보존해 재표현한다.
- 오답 사례는 base_fact와 distractor_fact를 의도적으로 섞어 내부적으로 한 군데만 어긋나게 만든다.
- 원문 문장, 정의, 숫자열을 그대로 복사하지 말고 문장 구조와 표현 순서를 바꾸어 재구성한다. 단, 기술적 의미·인과·수치는 바꾸지 않는다.
- 첫 task와 둘째 task의 최종 문구는 Python이 고정하므로 AI가 만든 tasks는 사용되지 않는다. passage/conditions가 이 고정 과업과 정확히 맞도록 작성한다.
- 고정 첫 task는 ㉠/㉡ 중 내부적으로 일관된 사례를 하나 고르는 판단이다.
- 고정 둘째 task는 첫 판단을 이용하여 다른 사례에서 잘못 섞인 한 부분을 linked_fact에 맞게 수정하는 과업이다.
- 둘째 정답 문구를 passage나 conditions에 그대로 제시하지 않는다.
- 첫 task와 둘째 task를 서로 독립적으로 풀 수 있게 만들지 않는다. 둘째 task에는 "첫 판단을 근거로", "선택하지 않은 사례에서" 같은 연결 표현을 사용한다.
- hidden_core_answer와 hidden_contrast_answer는 passage/conditions/tasks에 직접 쓰지 않는다.
- 별도의 제3개념, 새로운 기술 사실, 새로운 수치, 새로운 효과를 추가하지 않는다.
- 2점의 난도는 장문·장식자료가 아니라 '두 사실의 조합 일관성 판단 → 혼합 오류 수정'이라는 최소 2단계 사고에서 만든다.
- 4점은 원자료에 원래 존재하는 하나의 과정/장치/현상/계산관계를 중심으로 만든다. 독립 개념 3개를 억지로 한 지문에 합치지 않는다.
- selection_mode가 python_global_t4_single_concept_chain이면 Python이 고정한 동일 local source block의 사실들만 사용한다. 각 사실을 따로 찾아 옮기는 문항이 아니라 앞의 해석/판단이 다음 요구의 전제가 되는 하나의 사고사슬로 재구성한다.
- 이 4점 fallback에서도 원문 정의·목록을 그대로 나열하거나 정답 문구를 passage에 직접 노출하지 않는다. 새로운 사례·수치·효과를 추가하지 않는다.
- 4점 자료에는 고정정답의 정의·조건·수치관계를 완성된 문장으로 그대로 나열하지 않는다. 수험생이 최소 한 번 계산·비교·판단해야 답이 나오게 한다.
- 난도는 지문 길이와 정보량이 아니라 사고과정으로 만든다.
- 정답은 CORE/NORMAL을 우선한다. SUPPORT는 자료·상황·오답·설명용으로 활용할 수 있지만, 세부 고유명·특수사례 자체를 주된 정답으로 요구하지 않는다.
- 자료형은 실제 답 풀이에 필요한 형태여야 하고 장식용이면 안 된다.
- JSON 하나만 출력한다.
{{
 "intro":"...",
 "passage":"...",
 "case_correct":"...",
 "case_wrong":"...",
 "conditions":["..."],
 "tasks":["..."],
 "thinking_types":["..."]
}}
"""
    r=client.responses.create(model=model,input=prompt,reasoning={"effort":"medium"})
    x=json.loads(_strip_json(r.output_text))

    # R30: 2점 reasoning-matrix의 채점행동(tasks)은 AI 표현에 맡기지 않는다.
    # Python이 이미 reasoning_spec과 정답 구조를 확정했으므로,
    # 첫 1점=자료판단, 둘째 1점=첫 판단을 사용한 오류수정이라는 계약도 고정한다.
    # AI는 passage/conditions의 자연스러운 임용형 표현만 담당한다.
    if int(points)==2 and selection_mode=="python_exam_value_t2_reasoning_matrix":
        _writer_tasks=[
            "㉠과 ㉡ 중 두 사실의 관계가 서로 일관된 것을 고르시오.",
            "첫 판단을 근거로, 선택하지 않은 사례에서 잘못 연결된 내용을 바르게 수정하시오.",
        ]
        # R31: AI가 ㉠/㉡의 역할을 뒤집거나 둘을 섞지 못하도록 사례 배치도 Python이 소유한다.
        _case_correct=str(x.get("case_correct","") or "").strip()
        _case_wrong=str(x.get("case_wrong","") or "").strip()
        _co=str(relation_meta.get("correct_option","㉠") or "㉠")
        if _case_correct and _case_wrong:
            if _co=="㉡":
                _writer_passage=f"㉠ {_case_wrong}\n㉡ {_case_correct}"
            else:
                _writer_passage=f"㉠ {_case_correct}\n㉡ {_case_wrong}"
        else:
            # 구형/불완전 Writer 응답은 그대로 살리지 않고 downstream prejudge에서 실패시키기 위해
            # passage를 유지하되 구조 검사가 ㉠/㉡ 및 semantic contract를 확인한다.
            _writer_passage=str(x.get("passage","")).strip()
    else:
        _writer_tasks=[str(v).strip() for v in x.get("tasks",[]) if str(v).strip()]
        _writer_passage=str(x.get("passage","")).strip()

    _sources=[{"source_name":a["source_name"],"page_no":a["page_no"]} for a in bundle]
    if int(points)==2 and selection_mode in {"python_exam_value_decision_first_t2","python_exam_value_t2_reasoning_matrix"}:
        _cs=str(relation_meta.get("contrast_source_name","") or "")
        _cp=relation_meta.get("contrast_page_no")
        if _cs and not any(str(z.get("source_name",""))==_cs and z.get("page_no")==_cp for z in _sources):
            _sources.append({"source_name":_cs,"page_no":_cp})
    q={
      "domain":bundle[0]["domain"],
      "topic":" · ".join(a["topic"] for a in bundle),
      "points":points,"subpoints":pattern["subpoints"],"pattern_id":pattern["id"],
      "question_type":question_type,"material_form":material_form,
      "verifier":"source",
      "intro":x.get("intro","다음 자료를 읽고 <작성 방법>에 따라 쓰시오."),
      "passage":_writer_passage,
      "conditions":[str(v).strip() for v in x.get("conditions",[]) if str(v).strip()],
      "tasks":_writer_tasks,
      "answer":answers,
      "solution":[f"{LABELS[i]}: {answers[i]}" for i in range(len(answers))],
      "evidence":evidence,
      "derived_answer_flags":derived_answer_flags,
      "sources":_sources,
      "source_basis":"; ".join(f"{z['source_name']} p.{z['page_no']}" for z in _sources),
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
