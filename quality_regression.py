import random, copy
from validators import static_quality_errors

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
            top={
                'topics':[str(x.get('topic','')) for x in bundle],
                'answers':[str(x.get('answer','')) for x in bundle],
                'selection_mode':str((meta or {}).get('selection_mode','')),
                'selector_reason':str((meta or {}).get('selector_reason','')),
                'prejudge':eb._t4_candidate_prejudge(bundle,pid) if bundle else (False,'no_candidate')
            }
            result[domain][pid]={
                'accepted_count':accepted,
                'top_candidate':top,
                'prejudge_reject_count':int(pd.get('four_point_prejudge_reject',0) or 0),
                'prejudge_examples':copy.deepcopy(pd.get('four_point_prejudge_examples',[]) or []),
                'final_reason':pd.get('final_reason',''),
                'leaderboard':copy.deepcopy(sd.get('leaderboard',[]) or [])
            }
            total += accepted
    return {'total_accepted_candidates':total,'domains':result}


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
    t4=audit_t4_universe(db_path,domains)
    plans=audit_sample_plans(db_path,domains,seeds=seeds)
    return {
      "pass":bool(hist.get("pass") and plans.get("pass")),
      "historical":hist,"t4_universe":t4,"sample_plan_coverage":plans,
      "note":"API-free release gate: historical failure classes + T4 DB universe + SAMPLE6 capability plans"
    }
