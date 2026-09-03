import random, copy
from validators import static_quality_errors, validate_grounded_question, source_evidence_grounded

T4_PATTERNS={"T4_DATA112":3,"T4_ERR22":2,"T4_112":3}


def historical_regression_cases():
    """Known failure classes from R30-R38.  No concept-specific production exceptions:
    examples exist only to prove the generic validators stay active."""
    base={
        "points":4,"premise_mode":"ai_grounded","verifier":"source","conditions":[],
        "sources":[{"source_name":"regression","page_no":1}],"ai_quality":{"pass":True},
        "intended_thinking_types":["자료해석","적용"],"master_concept":"x","relation":"x-y"
    }
    cases=[]
    q=copy.deepcopy(base)
    q.update({
      "pattern_id":"T4_DATA112","subpoints":[1,1,2],
      "passage":"자료에는 첫 성질과 둘째 성질의 정의가 원문과 거의 같은 형태로 제시되어 있다.",
      "tasks":["X의 명칭을 쓰시오.","Y의 명칭을 쓰시오.","앞의 결과를 이용하여 Z의 명칭을 쓰시오."],
      "answer":["X","Y","Z"],
      "evidence":["첫 성질의 정의가 원문과 거의 같은 형태로 제시되어 있다","둘째 성질의 정의가 원문과 거의 같은 형태로 제시되어 있다","Z의 정의가 원문과 거의 같은 형태로 제시되어 있다"]
    })
    cases.append(("t4_direct_definition_lookup",q,{"4점 추론거리 부족(자료→정답 직접대응 과다)"}))

    q=copy.deepcopy(base)
    q.update({
      "pattern_id":"T4_ERR22","subpoints":[2,2],
      "passage":"학생이 두 설명을 검토하였다.",
      "tasks":["첫 설명의 명칭을 쓰고 잘못된 부분을 수정하시오.","둘째 설명의 명칭을 쓰고 같은 원리를 근거로 수정하시오."],
      "answer":["자기력선","자기력"],
      "evidence":["자기력선에 관한 올바른 설명","자기력에 관한 올바른 설명"]
    })
    cases.append(("t4_err22_answer_contract",q,{"4점 채점요구-정답 계약 불일치(1번 task)","4점 ERR22 수정정답 누락 위험(1번 task)"}))

    # R41: generated 4-point tasks must never regress to three independent recall prompts.
    q=copy.deepcopy(base)
    q.update({
      "pattern_id":"T4_112","subpoints":[1,1,2],
      "passage":"자료 A, 자료 B, 자료 C를 제시하였다.",
      "tasks":["A의 명칭을 쓰시오.","B의 명칭을 쓰시오.","C의 명칭을 쓰시오."],
      "answer":["A","B","C"],"evidence":["A 근거","B 근거","C 근거"]
    })
    cases.append(("t4_recall_only_tasks",q,{"4점 독립 명칭회상 과다","4점 단순회상형"}))

    q=copy.deepcopy(base)
    q.update({
      "pattern_id":"T4_DATA112","subpoints":[1,1,2],
      "passage":"① A, ② B, ③ C, ④ D, ⑤ E, ⑥ F, ⑦ G, ⑧ H의 사실을 나열하였다.",
      "tasks":["자료를 해석하여 첫 판단을 쓰시오.","첫 판단과의 관계를 판단하여 둘째 내용을 쓰시오.","앞의 두 판단을 근거로 결과를 설명하시오."],
      "answer":["판단A","판단B","결과C"],"evidence":["근거A","근거B","근거C"]
    })
    cases.append(("t4_independent_fact_listing",q,{"4점 독립사실 나열 과다"}))
    return cases


def run_historical_regressions():
    rows=[]; ok=True
    for name,q,required in historical_regression_cases():
        errs=set(static_quality_errors(q,require_ai_quality=False))
        missing=sorted(required-errs)
        passed=not missing
        ok &= passed
        rows.append({"name":name,"pass":passed,"missing":missing,"errors":sorted(errs)})
    return {"pass":ok,"cases":rows}



def run_grounding_regressions():
    """R40: regressions for PDF source noise and invented visual-media references."""
    rows=[]; ok=True
    # PDF extraction may insert a page number/header in the middle of an otherwise
    # verbatim source sentence.  This must still ground, while new factual text must not.
    src="기교론서브노트기가교강지현108문제확인기법문제를정확하게확인하기위한기법이있다"
    good="기교론서브노트기가교강지현문제확인기법문제를정확하게확인하기위한기법이있다"
    bad="기교론서브노트기가교강지현문제확인기법출처에없는새로운효과가발생한다"
    g1=source_evidence_grounded(src,good)
    g2=not source_evidence_grounded(src,bad)
    rows.append({"name":"pdf_header_insertion_tolerance","pass":bool(g1)})
    rows.append({"name":"new_fact_still_rejected","pass":bool(g2)})
    ok &= bool(g1 and g2)

    q={
      "points":4,"pattern_id":"T4_DATA112","subpoints":[1,1,2],
      "premise_mode":"ai_grounded","verifier":"source",
      "intro":"","passage":"아래 그림을 보고 자료를 해석하시오.","conditions":[],
      "tasks":["첫 판단을 쓰시오.","둘째 판단을 쓰시오.","앞 판단을 이용해 설명하시오."],
      "answer":["A","B","C"],"evidence":["첫 사실","둘째 사실","셋째 사실"],
      "sources":[{"source_name":"regression","page_no":1}],
      "ai_quality":{"pass":True},"intended_thinking_types":["자료해석","적용"],
      "master_concept":"x","relation":"x-y"
    }
    errs=validate_grounded_question(q,"첫 사실 둘째 사실 셋째 사실",allow_ai_grounded=True,require_ai_quality=False)
    media_ok="실제 그림 없이 그림 언급" in errs
    rows.append({"name":"invented_visual_reference_rejected","pass":media_ok,"errors":errs})
    ok &= media_ok
    return {"pass":ok,"cases":rows}


def audit_t4_operation_capabilities(db_path,domains,seeds=40):
    """R43 API-free audit of executable 4-point capabilities.

    A domain is 4-point-capable only when Python can repeatedly generate a three-element
    1+1+2 question, compute all fixed answers, and pass the formula validator.  Concept
    relation/direct-chain candidates are deliberately not counted.
    """
    import exam_builder as eb
    from formula_templates import generate_formula_question
    from validators import validate_formula_question
    cells={}
    capable=[]
    for domain in domains:
        attempts=0; accepted=0; topics=set(); failures=[]
        if domain in eb.FORMULA_DOMAINS:
            for seed in range(int(seeds)):
                attempts+=1
                q=generate_formula_question(domain,random.Random(43000+seed))
                if not q:
                    failures.append('no_template'); continue
                q=eb._enrich_formula(q,4)
                if not q:
                    failures.append('not_three_step'); continue
                errs=validate_formula_question(q)
                if errs:
                    failures.extend([str(x) for x in errs[:2]]); continue
                if list(q.get('subpoints',[])) != [1,1,2] or len(q.get('answer',[])) != 3:
                    failures.append('bad_scoring_shape'); continue
                accepted+=1; topics.add(str(q.get('topic','')))
        passed=accepted>0
        if passed: capable.append(domain)
        cells[domain]={
            'pass':passed,'attempts':attempts,'accepted':accepted,
            'topics':sorted(topics),'failure_examples':failures[:8],
            'capability_mode':'deterministic_formula_operation' if passed else 'none'
        }
    unsupported=[d for d in domains if d not in capable]
    # Four SAMPLE slots can safely reuse a certified operation domain; final A/B cannot
    # claim domain-complete 4-point readiness until every required domain has a capability.
    return {
        'pass':bool(capable),'sample_ready':bool(capable),'final_ab_ready':not unsupported,
        'capable_domains':capable,'unsupported_domains':unsupported,'domains':cells,
        'relation_chain_counted_as_capability':False,
        'note':'R43: 4점 capability는 Python 실행 가능한 풀이연산만 인정; direct-chain/core score는 자격 근거가 아님'
    }

def audit_sample_plans(db_path,domains,seeds=50):
    import exam_builder as eb
    from patterns import blueprint
    failures=[]; formula=0; rule=0
    for seed in range(int(seeds)):
        rng=random.Random(seed)
        plan=blueprint("SAMPLE",[2,2,4,4,4,4],domains,rng)
        plan,cap,diag=eb._rebalance_t2_domains_by_capability(db_path,plan,domains,rng,previous_questions=[])
        for slot in plan:
            if int(slot.get("points",0))==2:
                mode=str(slot.get("t2_capability_mode",""))
                if mode=="planning_error_no_capability" or not mode:
                    failures.append({"seed":seed,"slot":copy.deepcopy(slot)})
                if mode.startswith("formula_capability"): formula+=1
                if mode=="numeric_series_rule_application": rule+=1
    return {"pass":not failures,"seeds":int(seeds),"failures":failures,"formula_slots":formula,"rule_slots":rule}


def run_release_regression(db_path,domains,seeds=50):
    hist=run_historical_regressions()
    grounding=run_grounding_regressions()
    t4=audit_t4_operation_capabilities(db_path,domains)
    plans=audit_sample_plans(db_path,domains,seeds=seeds)
    from reasoning_capabilities import coverage_inventory, validation_inventory
    try:
        import exam_builder as eb
        coverage=coverage_inventory(db_path,domains,getattr(eb,"FORMULA_DOMAINS",set()))
        validation=validation_inventory(db_path,domains,getattr(eb,"FORMULA_DOMAINS",set()))
        # Actually construct all 18 without Judge. This is the API-free gate.
        suite=eb.make_capability_validation_suite(db_path,domains=domains,api_key="",ai_quality_enabled=False,seed=4901)
        constructed=len(suite.get("questions",[]) or [])
        suite_construct={"pass":constructed==len(domains)*2,"constructed":constructed,"target":len(domains)*2,
                         "selected_targets":validation.get("targets",[])}
    except Exception as ex:
        coverage={"all_domains_two_targets":False,"error":str(ex)}
        validation={"all_domains_two_targets":False,"error":str(ex)}
        suite_construct={"pass":False,"error":str(ex)}
    coverage_ready=bool(coverage.get("all_domains_two_targets"))
    return {
      "pass":bool(hist.get("pass") and grounding.get("pass") and t4.get("pass") and plans.get("pass") and suite_construct.get("pass")),
      "coverage_ready":coverage_ready,
      "historical":hist,"grounding":grounding,"t4_operation_capabilities":t4,"sample_plan_coverage":plans,
      "certified_coverage":coverage,"reasoning_validation_candidates":validation,"capability18_construction":suite_construct,
      "note":"R49: API-free gate는 새 semantic operation을 포함한 18개 문항의 실제 구성 가능성을 검사한다. 최종 coverage에는 Judge PASS만 산입한다."
    }


def audit_r44_judge_alignment():
    """R44 regression: deterministic T4_C112 with reviewer PASS and
    inferential_distance=3 must not be silently flipped to REJECT, while
    non-operation T4 keeps the stricter 3.5 floor and fatal flags still veto.
    """
    import quality_judge as qj
    original=qj._ask_json
    payload={
        "verdict":"PASS",
        "scores":{
            "grounding":5,"answer_leakage":5,"coherence":5,
            "inferential_distance":3,"task_distinctness":4,
            "exam_realism":5,"difficulty_fit":4,"ambiguity_control":5
        },
        "thinking_types":["계산","적용"],"fatal_flags":[],
        "reason":"4점 문항으로 수용 가능","weakest_point":"표준적 계산"
    }
    try:
        qj._ask_json=lambda *a,**k: dict(payload)
        calc=qj.judge_question("k","m",{"points":4,"pattern_id":"T4_C112"})
        noncalc=qj.judge_question("k","m",{"points":4,"pattern_id":"T4_DATA112"})
        fatal_payload=dict(payload)
        fatal_payload["fatal_flags"]=["TOO_EASY"]
        qj._ask_json=lambda *a,**k: dict(fatal_payload)
        fatal=qj.judge_question("k","m",{"points":4,"pattern_id":"T4_C112"})
    finally:
        qj._ask_json=original
    return {
        "pass": bool(calc.get("pass") and not noncalc.get("pass") and not fatal.get("pass")),
        "operation_id3_pass": bool(calc.get("pass")),
        "nonoperation_id3_reject": not bool(noncalc.get("pass")),
        "fatal_still_rejects": not bool(fatal.get("pass")),
    }

# R47 full-suite failure-class regression: 0-pass architectures must never count
# toward 9x2 coverage merely because a lexical candidate exists.
def audit_r47_failure_class_correction(db_path, domains):
    from reasoning_capabilities import coverage_inventory, CAP_CONTRAST, CAP_CONDITION
    import exam_builder as eb
    inv=coverage_inventory(db_path,domains,getattr(eb,'FORMULA_DOMAINS',set()))
    selected=[t.get('capability_id') for t in inv.get('targets',[])]
    bad=[x for x in selected if x in (CAP_CONTRAST,CAP_CONDITION)]
    return {
        'pass': not bad,
        'retired_zero_pass_capabilities_not_selected': not bad,
        'bad_selected': bad,
        'constructed_target_total': inv.get('constructed_target_total'),
        'target_total': inv.get('target_total'),
        'all_domains_two_targets': inv.get('all_domains_two_targets'),
    }
