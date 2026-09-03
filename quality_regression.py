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


def audit_t4_universe(db_path,domains):
    """API-free T4 universe audit.
    Each selector call already evaluates the whole domain/pattern universe; use its
    candidate_accept/leaderboard diagnostics instead of repeatedly re-reading the DB.
    """
    import exam_builder as eb
    result={}; total=0
    for domain in domains:
        result[domain]={}
        for pid,need in T4_PATTERNS.items():
            bundle,meta=eb._smart_relation_bundle(
                db_path,domain,need,set(),set(),random.Random(39000),
                pattern_id=pid,used_source_pages=set()
            )
            pd=copy.deepcopy((meta or {}).get('score_pipeline_diagnostic',{}) or {})
            sd=copy.deepcopy((meta or {}).get('score_diagnostic',{}) or {})
            accepted=int(pd.get('candidate_accept',0) or 0)
            if bundle and accepted<=0:
                accepted=int(pd.get('four_point_single_anchor_candidates',1) or 1)
            ctx=str((meta or {}).get('source_context_override','') or '')
            if not ctx and bundle:
                ctx=eb.bundle_context(db_path,bundle)
            grounding=[source_evidence_grounded(ctx,x.get('evidence','')) for x in bundle] if bundle else []
            pre=eb._t4_candidate_prejudge(bundle,pid) if bundle else (False,'no_candidate')
            top={
                'topics':[str(x.get('topic','')) for x in bundle],
                'answers':[str(x.get('answer','')) for x in bundle],
                'selection_mode':str((meta or {}).get('selection_mode','')),
                'selector_reason':str((meta or {}).get('selector_reason','')),
                'prejudge':pre,
                'source_grounding':grounding
            }
            cell_pass=bool(bundle and accepted>0 and pre[0] and grounding and all(grounding))
            result[domain][pid]={
                'pass':cell_pass,
                'accepted_count':accepted,
                'top_candidate':top,
                'prejudge_reject_count':int(pd.get('four_point_prejudge_reject',0) or 0),
                'source_ground_reject_count':int(pd.get('four_point_source_ground_reject',0) or 0),
                'prejudge_examples':copy.deepcopy(pd.get('four_point_prejudge_examples',[]) or []),
                'final_reason':pd.get('final_reason',''),
                'leaderboard':copy.deepcopy(sd.get('leaderboard',[]) or [])
            }
            total += accepted
    all_pass=all(cell.get('pass') for dm in result.values() for cell in dm.values())
    return {'pass':all_pass,'total_accepted_candidates':total,'domains':result}


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
    t4=audit_t4_universe(db_path,domains)
    plans=audit_sample_plans(db_path,domains,seeds=seeds)
    return {
      "pass":bool(hist.get("pass") and grounding.get("pass") and t4.get("pass") and plans.get("pass")),
      "historical":hist,"grounding":grounding,"t4_universe":t4,"sample_plan_coverage":plans,
      "note":"API-free release gate: historical failures + grounding/media regressions + T4 grounded universe + SAMPLE6 capability plans"
    }
