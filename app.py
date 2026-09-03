import os, json, sqlite3, copy
from pathlib import Path
from datetime import datetime

import streamlit as st

import exam_builder as exam_builder_module
make_ab = exam_builder_module.make_ab
make_quality_sample = getattr(exam_builder_module, "make_quality_sample", None)
try:
    from quality_regression import run_release_regression
except Exception:
    run_release_regression = None
DOMAINS = exam_builder_module.DOMAINS
BUILDER_API_VERSION = getattr(exam_builder_module, "BUILDER_API_VERSION", "UNKNOWN")
from pdf_export import export_pdf
from retrieval import search_pages
from archive_store import (
    is_configured as archive_is_configured,
    ping as archive_ping,
    list_exams, get_exam, create_exam, update_exam, delete_exam
)

ROOT=Path(__file__).parent
DB=ROOT/"knowledge.db"

st.set_page_config(page_title="기술 임용 자동검증 모의고사",layout="wide")
st.title("기술 임용 A/B 자동검증 모의고사 생성기")
st.caption("서브노트=정답 근거 · 실제 기출=문항 구조 · Python=계산/검증 · AI=표현만 담당 · Supabase=모의고사 영구 보관")
st.caption("배포 버전: FINAL-STABLE-20260831 · SAMPLE6-R40-20260903")

def secret(name, default=""):
    try:
        return st.secrets[name]
    except Exception:
        return os.getenv(name,default)

def api_key():
    return secret("OPENAI_API_KEY","")

def archive_credentials():
    url=secret("SUPABASE_URL","")
    service_key=secret("SUPABASE_SERVICE_ROLE_KEY","")
    fallback=secret("SUPABASE_KEY","")
    return url, (service_key or fallback), bool(service_key), bool(fallback and not service_key)

def db_stats():
    con=sqlite3.connect(DB)
    a=con.execute("select count(*) from sources").fetchone()[0]
    p=con.execute("select count(*) from pages").fetchone()[0]
    an=con.execute("select count(*) from anchors").fetchone()[0]
    kinds=dict(con.execute("select kind,count(*) from sources group by kind").fetchall())
    con.close()
    return a,p,an,kinds

def default_archive_title(seed):
    return f"{datetime.now().strftime('%Y-%m-%d %H:%M')} · seed {seed}"

def save_generated_to_archive(A,B,model,seed,domains):
    url,key,has_service,_=archive_credentials()
    if not archive_is_configured(url,key):
        return None, "Supabase 보관소가 설정되지 않아 이번 결과는 현재 세션에만 남습니다."
    if not has_service:
        return None, "SUPABASE_SERVICE_ROLE_KEY가 없어 자동 저장하지 않았습니다. 보관소에는 Service Role Key 사용을 권장합니다."
    rec={
        "title":default_archive_title(seed),
        "note":"",
        "model":model,
        "seed":int(seed),
        "domains":list(domains),
        "exam_a":A,
        "exam_b":B,
        "manually_edited":False,
    }
    saved=create_exam(url,key,rec)
    return saved, None

def exam_editor(exam,prefix):
    edited=copy.deepcopy(exam)
    for q in edited.get("questions",[]):
        qn=q.get("number","?")
        with st.expander(f"{qn}번 · {q.get('domain','')} · {q.get('points','')}점"):
            q["intro"]=st.text_input(
                "문항 안내",value=str(q.get("intro","")),
                key=f"{prefix}_{qn}_intro"
            )
            q["passage"]=st.text_area(
                "지문",value=str(q.get("passage","")),height=150,
                key=f"{prefix}_{qn}_passage"
            )
            cond_text=st.text_area(
                "조건 (한 줄에 하나)",value="\n".join(map(str,q.get("conditions",[]))),
                height=80,key=f"{prefix}_{qn}_conditions"
            )
            q["conditions"]=[x.strip() for x in cond_text.splitlines() if x.strip()]
            task_text=st.text_area(
                "작성 방법 (한 줄에 하나)",value="\n".join(map(str,q.get("tasks",[]))),
                height=100,key=f"{prefix}_{qn}_tasks"
            )
            q["tasks"]=[x.strip() for x in task_text.splitlines() if x.strip()]
            ans_text=st.text_area(
                "정답 요소 (한 줄에 하나)",value="\n".join(map(str,q.get("answer",[]))),
                height=90,key=f"{prefix}_{qn}_answer"
            )
            q["answer"]=[x.strip() for x in ans_text.splitlines() if x.strip()]
            sol_text=st.text_area(
                "해설 요소 (한 줄에 하나)",value="\n".join(map(str,q.get("solution",[]))),
                height=110,key=f"{prefix}_{qn}_solution"
            )
            q["solution"]=[x.strip() for x in sol_text.splitlines() if x.strip()]
            sp_text=st.text_input(
                "부분점수 (예: 1,1,2)",
                value=",".join(map(str,q.get("subpoints",[]))),
                key=f"{prefix}_{qn}_subpoints"
            )
            try:
                q["subpoints"]=[int(x.strip()) for x in sp_text.split(",") if x.strip()]
            except Exception:
                st.warning(f"{qn}번 부분점수는 숫자를 쉼표로 구분해 입력하세요.")
    edited["verified"]=False
    edited["manually_edited"]=True
    edited["verification_note"]="보관소에서 수동 수정됨: 원래의 자동검증 상태를 그대로 보장하지 않음"
    return edited


def _show_score_pipeline_diagnostic(pd):
    if not pd:
        return
    st.markdown("##### 🧪 후보 탈락 단계")
    st.write({
        "DB 원본 rows":pd.get("raw_rows",0),
        "서브노트 출처 통과":pd.get("primary_source_pass",0),
        "anchor_ok 통과":pd.get("anchor_ok_pass",0),
        "사용/제외/중복 제거 후":pd.get("usable_anchors",0),
        "관계점수≥4 pair":pd.get("pair_score_pass",0),
        "충분한 이웃 보유 anchor":pd.get("edge_neighbor_pass",0),
        "근접중복 탈락":pd.get("near_duplicate_reject",0),
        "연결성 탈락":pd.get("connected_reject",0),
        "natural-unit 탈락":pd.get("natural_unit_reject",0),
        "direct-chain 탈락":pd.get("direct_chain_reject",0),
        "다른 출처 탈락":pd.get("same_source_reject",0),
        "페이지거리>2 탈락":pd.get("page_distance_reject",0),
        "실제 지엽성 강함 탈락":pd.get("support_only_reject",0),
        "SUPPORT-only fallback":pd.get("support_only_fallback",0),
        "2점 명칭+명칭 탈락":pd.get("two_point_label_reject",0),
        "2점 관계성 부족 탈락":pd.get("two_point_relation_reject",0),
        "2점 이중개념 억지결합 탈락":pd.get("two_point_dual_target_reject",0),
        "2점 단일-anchor 후보":pd.get("two_point_one_anchor_candidates",0),
        "anchor 내부모순 탈락":pd.get("anchor_contradiction_reject",0),
        "불완전 anchor 탈락":pd.get("anchor_fragment_reject",0),
        "불완전 bundle 탈락":pd.get("bundle_fragment_reject",0),
        "최종 candidate":pd.get("candidate_accept",0),
    })
    st.caption(
        f"{pd.get('domain','')} / {pd.get('pattern_id','')} / "
        f"필요 {pd.get('required_count','?')} · 최종원인={pd.get('final_reason','')}"
    )
    rej=pd.get("reject_examples",{})
    labels=[
        ("support_only","SUPPORT/지엽성 판정 예시"),
        ("two_point_label","명칭+명칭 탈락 예시"),
        ("two_point_relation","2점 관계성 부족 탈락 예시"),
        ("two_point_dual_target","2점 이중개념 억지결합 탈락 예시"),
        ("anchor_contradiction","anchor 내부모순 탈락 예시"),
        ("anchor_fragment","불완전 anchor 탈락 예시"),
        ("bundle_fragment","불완전 bundle 탈락 예시"),
        ("same_source","다른 출처 탈락 예시"),
        ("page_distance","페이지 거리 탈락 예시"),
        ("natural_unit","natural-unit 탈락 예시"),
    ]
    for key,title in labels:
        rows=rej.get(key,[])
        if rows:
            with st.expander(title):
                for row in rows:
                    st.json(row)

def show_generation_diagnostics(exc):
    diagnostics=getattr(exc,"generation_diagnostics",None)
    if not diagnostics:
        return
    st.markdown("#### 생성 실패 진단")
    st.caption("실제로 탈락한 단계만 표시합니다. 성공한 Python 관계선별은 이 목록에 표시하지 않습니다.")
    for i,row in enumerate(diagnostics,1):
        sec=row.get("section","")
        no=row.get("number","")
        dom=row.get("domain","")
        attempt=row.get("attempt","")
        stage=row.get("stage","")
        title=f"{i}. {attempt}"
        if sec or no:
            title+=f" · {sec} {no}번"
        if dom:
            title+=f" · {dom}"
        if stage:
            title+=f" · {stage}"
        with st.expander(title):
            if row.get("pattern"):
                st.write("패턴:",row["pattern"])
            if row.get("reason"):
                st.write("사유:",row["reason"])
            if row.get("fatal_flags"):
                st.write("Fatal flags:",row["fatal_flags"])
            if row.get("scores"):
                st.write("심사 점수:")
                st.json(row["scores"])
            if row.get("weakest_point"):
                st.write("가장 큰 약점:",row["weakest_point"])
            if row.get("candidate_topics"):
                st.write("후보 주제:",row["candidate_topics"])
            if row.get("blind_verdict") or row.get("grounded_verdict"):
                st.write(
                    "Blind / Grounded:",
                    row.get("blind_verdict","-"),
                    "/",
                    row.get("grounded_verdict","-")
                )
            if row.get("score_pipeline_diagnostic"):
                _show_score_pipeline_diagnostic(row.get("score_pipeline_diagnostic",{}))

tabs=st.tabs(["① DB 상태","② 출제범위 검색","③ A/B 생성","④ 모의고사 보관소","⑤ 검증 원리","⑥ 기출 구조"])

with tabs[0]:
    s,p,a,k=db_stats()
    c1,c2,c3=st.columns(3)
    c1.metric("등록 자료",s)
    c2.metric("색인 페이지",p)
    c3.metric("원문 정답 앵커",a)
    st.write("자료 유형:",k)
    con=sqlite3.connect(DB)
    rows=con.execute("select name,kind,domain,page_count from sources order by kind,domain,name").fetchall()
    con.close()
    st.dataframe([{"파일":r[0],"유형":r[1],"영역":r[2],"쪽수":r[3]} for r in rows],use_container_width=True)
    st.info("배포 버전에는 PDF 원본을 넣지 않고 SQLite 지식 DB만 포함합니다.")

with tabs[1]:
    c1,c2=st.columns([1,2])
    with c1:
        d=st.selectbox("영역",DOMAINS)
        query=st.text_input("검색어",placeholder="예: 좌굴, FCM, 압연, TCP, 핵치환")
        n=st.slider("결과 수",1,20,6)
    res=search_pages(DB,d,query,kind="subnote",limit=n)
    with c2:
        for r in res:
            with st.expander(f"{r['source_name']} p.{r['page_no']}"):
                st.text(r["text"][:6000])

with tabs[2]:
    st.subheader("생성 설정")
    c1,c2,c3=st.columns(3)
    with c1:
        domains=st.multiselect("출제 영역",DOMAINS,default=DOMAINS)
    with c2:
        use_ai=st.toggle("AI로 임용형 문장 재구성",value=True)
        model=st.text_input("문항 작성 AI",value="gpt-5.6-luna")
        use_ai_judge=st.toggle("AI 출제검토위원 품질심사",value=True)
        judge_model=st.text_input("품질심사 AI",value="gpt-5.6-luna")
    with c3:
        seed=st.number_input("랜덤 시드(재현용)",min_value=0,value=20260830,step=1)

    st.caption("품질우선모드: DB/Python 정답 고정 → AI 관계성 선별 → AI 문항 작성 → 정답을 보지 않는 Blind 검토 + 근거를 보는 Grounded 검토 → Python 재검증 → 섹션 심사 → A/B 종합심사 순으로 생성합니다.")
    if use_ai and use_ai_judge:
        st.info("품질심사를 여러 단계로 수행하므로 이전 버전보다 생성 시간이 길고 API 호출도 많습니다. 대신 탈락 문항은 저장하지 않고 다른 후보로 교체합니다.")
    key=api_key()
    if use_ai and not key:
        st.warning("OPENAI_API_KEY가 없어 이번 생성은 AI 없이 안전모드로 동작합니다.")

    url,skey,has_service,fallback=archive_credentials()
    if archive_is_configured(url,skey) and has_service:
        st.success("모의고사 보관소 연결됨: 생성 완료 시 Supabase에 자동 저장됩니다.")
    elif fallback:
        st.warning("SUPABASE_KEY는 발견했지만 보관소 자동 저장에는 SUPABASE_SERVICE_ROLE_KEY 사용을 권장합니다.")
    else:
        st.info("Supabase 보관소를 연결하면 새로고침 후에도 생성 결과가 유지됩니다.")

    st.markdown("#### 🧪 품질 튜닝용 6문항")
    loaded_builder_path=str(getattr(exam_builder_module,"__file__","UNKNOWN"))
    st.caption(f"현재 로드된 exam_builder: {BUILDER_API_VERSION}")
    st.caption(f"로드 경로: {loaded_builder_path}")

    # 문자열 버전이 조금 달라도 샘플 기능 자체가 있으면 실행 가능하게 한다.
    builder_ok=(
        callable(getattr(exam_builder_module,"make_quality_sample",None))
        and callable(getattr(exam_builder_module,"make_ab",None))
        and hasattr(exam_builder_module,"_smart_relation_bundle")
    )
    if not builder_ok:
        st.error(
            "⚠️ 현재 실행 중인 exam_builder.py에 샘플6 기능이 없습니다. "
            "위의 '로드 경로'가 GitHub에서 덮어쓴 exam_builder.py 위치와 같은지 확인하세요."
        )
    st.caption(
        "먼저 API 0회의 전수 회귀검사를 권장합니다. 그 뒤 2점 2문항 + 4점 4문항을 생성하고, "
        "완성된 6문항을 각각 딱 한 번 AI Judge에 넣어 실제 품질을 확인합니다. "
        "Judge REJECT가 나와도 자동 재시도하지 않아 비용과 원인 추적을 통제합니다."
    )

    if st.button("API 0원 · DB 전체 회귀검사", use_container_width=True, disabled=(run_release_regression is None)):
        try:
            with st.spinner("과거 실패 회귀 + T4 전체 후보 + 6문항 capability 청사진 검사 중..."):
                rr=run_release_regression(DB,domains,seeds=50)
            st.session_state["QUALITY_REGRESSION"]=rr
            if rr.get("pass"):
                st.success("DB 전체 회귀검사 통과")
            else:
                st.error("DB 전체 회귀검사 실패: 6문항 AI 검사를 실행하기 전에 아래 항목을 먼저 수정하세요.")
        except Exception as e:
            st.error("회귀검사 실행 실패: "+str(e))

    if "QUALITY_REGRESSION" in st.session_state:
        rr=st.session_state["QUALITY_REGRESSION"]
        with st.expander("🧪 DB 전체 회귀검사 결과", expanded=not bool(rr.get("pass"))):
            st.json(rr)

    if st.button(
        "6문항 빠른 품질 샘플 생성",
        type="primary",
        use_container_width=True,
        disabled=(not builder_ok)
    ):
        if make_quality_sample is None:
            st.error(f"exam_builder.py에 make_quality_sample이 없습니다. 현재 로드 경로: {loaded_builder_path}")
        else:
            with st.spinner("2점 2개 + 4점 4개 생성 중..."):
                try:
                    sample=make_quality_sample(
                        DB,domains=domains,api_key=key,model=model,
                        ai_enabled=bool(use_ai and key),
                        judge_model=judge_model,
                        seed=int(seed),
                        ai_quality_enabled=bool(use_ai and key)
                    )
                    st.session_state["QUALITY_SAMPLE"]=sample
                    st.success("6문항 품질 튜닝 샘플 생성 완료.")
                    st.caption(
                        f"실행 모드: {sample.get('sample_mode','-')} · "
                        f"builder: {sample.get('builder_api_version','-')}"
                    )
                    ss=sample.get("generation_stats",{})
                    st.caption(
                        f"문항작성 AI {ss.get('ai_calls',0)}회 · "
                        f"AI 관계성 선별 {ss.get('ai_selector_calls',0)}회(정상=0) · "
                        f"계산형 {ss.get('formula_questions',0)}문항"
                    )
                    if sample.get("sample_ai_reviews"):
                        st.caption(
                            f"단발 AI Judge: PASS {sample.get('sample_ai_pass_count',0)} · "
                            f"REJECT {sample.get('sample_ai_reject_count',0)} · 자동 재시도 0회"
                        )
                except Exception as e:
                    st.error(str(e))
                    show_generation_diagnostics(e)

    if "QUALITY_SAMPLE" in st.session_state:
        sample=st.session_state["QUALITY_SAMPLE"]
        st.markdown("### 🧪 6문항 품질 튜닝 결과")
        st.warning("이 결과는 튜닝용입니다. 최종 A/B 자동검증 완료본으로 저장하지 않습니다.")
        for q in sample.get("questions",[]):
            badge="🧮 Python 검산" if q.get("verifier")=="python" else "📚 원문 근거검증"
            with st.expander(
                f"{q['number']}번 · {q['domain']} · {q.get('question_type','')} · "
                f"{q['points']}점({'+'.join(map(str,q.get('subpoints',[])))}) · {badge}",
                expanded=True
            ):
                st.caption(f"패턴: {q.get('pattern_id','-')} · 주제: {q.get('topic','')}")
                st.write(q.get("passage",""))
                if q.get("conditions"):
                    st.markdown("**<조건>**")
                    for x in q["conditions"]:
                        st.write("○",x)
                st.markdown("**<작성 방법>**")
                for x in q.get("tasks",[]):
                    st.write("○",x)
                with st.popover("정답/검증근거"):
                    st.write("정답:",q.get("answer",[]))
                    st.write("해설:",q.get("solution",[]))
                    if q.get("source_basis"):
                        st.write("출처:",q.get("source_basis"))
                    if q.get("evidence"):
                        st.write("근거:",q.get("evidence"))

                _sq=q.get("sample_ai_quality") or {}
                if _sq:
                    if _sq.get("pass"):
                        st.success("AI Judge: PASS")
                    else:
                        st.error("AI Judge: REJECT · "+str(_sq.get("reason","")))
                    if _sq.get("fatal_flags"):
                        st.write("Fatal flags:",_sq.get("fatal_flags"))
                    if _sq.get("scores"):
                        st.write("심사 점수:",_sq.get("scores"))

                # R14D1: 실제 후보 점수 진단. PDF에는 넣지 않고 화면에서만 확인한다.
                _sd=q.get("_score_diagnostic",{})
                if _sd:
                    with st.expander("🔎 핵심도·관계성 점수 진단",expanded=True):
                        _sel=_sd.get("selected",{})
                        st.caption(
                            f"선택 순위: TOP {_sd.get('selected_rank','-')} · "
                            f"최종 selector score: {_sel.get('final_selector_score','-')} · "
                            f"평균 core_exam_score: {_sel.get('avg_core_exam_score','-')}"
                        )
                        st.write(
                            "점수 기여:",
                            {
                                "기존 importance":_sel.get("importance_contribution"),
                                "기존 exam_value":_sel.get("exam_value_contribution"),
                                "core_exam":_sel.get("core_exam_contribution"),
                                "SUPPORT 패널티":_sel.get("support_penalty"),
                                "natural_unit":_sel.get("natural_unit_score"),
                                "실제 풀이 연결":(_sel.get("reasoning_chain_profile") or {}).get("effective_score",(_sel.get("reasoning_chain_profile") or {}).get("score")),
                                "정의합성 shortcut 패널티":_sel.get("shortcut_penalty"),
                            }
                        )

                        st.markdown("**선택된 정답 anchor별 core_exam_score**")
                        for _a in _sel.get("anchors",[]):
                            _b=_a.get("breakdown",{})
                            st.write(
                                f"• [{_a.get('core_exam_tier','SUPPORT')}] "
                                f"{_a.get('topic','')} → {_a.get('answer','')} "
                                f"(총 {_a.get('core_exam_score',0):.1f})"
                            )
                            st.caption(
                                "  기출 {past_exam}×4 / 서브노트 {subnote_importance}×3 / "
                                "대표 {representative}×3 / 반복 {repeatability}×4 / "
                                "연결 {centrality}×3 / 지엽 {peripherality}×(-2)".format(
                                    past_exam=_b.get("past_exam",0),
                                    subnote_importance=_b.get("subnote_importance",0),
                                    representative=_b.get("representative",0),
                                    repeatability=_b.get("repeatability",0),
                                    centrality=_b.get("centrality",0),
                                    peripherality=_b.get("peripherality",0),
                                )
                            )

                        st.markdown("**이 슬롯의 상위 후보**")
                        for _row in _sd.get("leaderboard",[])[:5]:
                            st.write(
                                f"TOP {_row.get('rank')} · selector {_row.get('final_selector_score')} · "
                                f"core 평균 {_row.get('avg_core_exam_score')} · "
                                f"tier {','.join(_row.get('tiers',[]))}"
                            )
                            st.caption(" → ".join(_row.get("topics",[])))

                        st.caption(
                            "이 화면은 진단 전용입니다. 현재 실행 중인 후보 점수·관계성 진단을 표시합니다."
                        )

        spdf=export_pdf(sample,False)
        sapdf=export_pdf(sample,True)
        c1,c2=st.columns(2)
        c1.download_button(
            "6문항 샘플 문제지 PDF",spdf,
            file_name="품질튜닝_6문항_문제지.pdf",
            mime="application/pdf",use_container_width=True
        )
        c2.download_button(
            "6문항 샘플 정답·해설 PDF",sapdf,
            file_name="품질튜닝_6문항_정답해설.pdf",
            mime="application/pdf",use_container_width=True
        )

    st.divider()
    st.markdown("#### 최종 A/B 생성")
    st.caption("6문항 샘플 품질이 안정된 뒤 최종 A/B를 생성하세요.")

    if st.button("전공 A + B 생성",type="primary",use_container_width=True):
        with st.spinner("정답 고정 → 문항 생성 → 근거 대조 → 중복 검사 → A/B 편성 → 보관 중..."):
            try:
                A,B=make_ab(
                    DB,domains=domains,api_key=key,model=model,
                    ai_enabled=bool(use_ai and key),
                    ai_quality_enabled=bool(use_ai_judge and key),
                    judge_model=judge_model,
                    seed=int(seed)
                )
                st.session_state["A"]=A
                st.session_state["B"]=B
                saved,warn=save_generated_to_archive(A,B,model,int(seed),domains)
                if saved:
                    st.session_state["current_archive_id"]=saved.get("id")
                    st.success("A/B 자동검증 통과 + 보관소 자동 저장 완료.")
                else:
                    st.success("A/B 모두 자동검증을 통과했습니다.")
                    if warn: st.warning(warn)
                astat=A.get("generation_stats",{}); bstat=B.get("generation_stats",{})
                st.caption(
                    f"문항작성 AI {astat.get('ai_calls',0)+bstat.get('ai_calls',0)}회 · "
                    f"AI 품질심사 대상 {astat.get('ai_judge_calls',0)+bstat.get('ai_judge_calls',0)}개 · "
                    f"관계성 선별 {astat.get('ai_selector_calls',0)+bstat.get('ai_selector_calls',0)}회 · "
                    f"AI 탈락 {astat.get('ai_judge_rejects',0)+bstat.get('ai_judge_rejects',0)}문항 · "
                    f"Python 계산형 {astat.get('formula_questions',0)+bstat.get('formula_questions',0)}문항"
                )
            except Exception as e:
                st.error(str(e))
                show_generation_diagnostics(e)

    if "A" in st.session_state and "B" in st.session_state:
        for sec in ["A","B"]:
            exam=st.session_state[sec]
            st.markdown(f"### 전공 {sec}")
            if exam.get("manually_edited"):
                st.warning("이 시험은 보관소에서 수동 수정되었습니다. 수정 후에는 기존 자동검증을 그대로 보장하지 않습니다.")
            for q in exam["questions"]:
                badge="🧮 Python 검산" if q.get("verifier")=="python" else "📚 원문 근거검증"
                with st.expander(f"{q['number']}번 · {q['domain']} · {q.get('question_type','')} · {q['points']}점({'+'.join(map(str,q.get('subpoints',[])))}) · {badge}"):
                    st.caption(f"주제: {q['topic']}")
                    st.write(q["passage"])
                    if q.get("conditions"):
                        st.markdown("**<조건>**")
                        for x in q["conditions"]: st.write("○",x)
                    st.markdown("**<작성 방법>**")
                    for x in q["tasks"]: st.write("○",x)
                    aq=q.get("ai_quality",{})
                    if aq.get("pass") is True:
                        st.caption(
                            f"AI 출제검토 통과 · 평균 {aq.get('average','-')}/5 · "
                            f"사고행동: {', '.join(aq.get('thinking_types',[])) or '-'}"
                        )
                    with st.popover("정답/검증근거"):
                        st.write("정답:",q["answer"])
                        st.write("해설:",q["solution"])
                        if aq:
                            st.write("AI 품질심사:",aq)
                        if q.get("verifier")=="source":
                            st.write("출처:", q.get("source_basis"))
                            st.write("근거:", q.get("evidence"))
            ppdf=export_pdf(exam,False); apdf=export_pdf(exam,True)
            c1,c2=st.columns(2)
            c1.download_button(f"전공 {sec} 문제지 PDF",ppdf,file_name=f"전공_{sec}_문제지.pdf",mime="application/pdf",use_container_width=True)
            c2.download_button(f"전공 {sec} 정답·해설 PDF",apdf,file_name=f"전공_{sec}_정답해설.pdf",mime="application/pdf",use_container_width=True)

with tabs[3]:
    st.subheader("📚 모의고사 보관소")
    url,skey,has_service,fallback=archive_credentials()

    if not archive_is_configured(url,skey):
        st.error("보관소가 아직 연결되지 않았습니다.")
        st.code(
            'SUPABASE_URL = "https://xxxx.supabase.co"\n'
            'SUPABASE_SERVICE_ROLE_KEY = "서비스 역할 키"\n',
            language="toml"
        )
        st.write("그리고 프로젝트에 포함된 `supabase_archive_schema.sql`을 Supabase SQL Editor에서 한 번 실행하세요.")
    else:
        try:
            archive_ping(url,skey)
            if not has_service:
                st.warning("현재 Service Role Key가 아닌 키를 사용 중입니다. RLS 설정에 따라 저장/수정/삭제가 막힐 수 있습니다.")
            records=list_exams(url,skey,100)
            st.caption(f"보관된 모의고사: {len(records)}개")

            if not records:
                st.info("아직 저장된 모의고사가 없습니다. A/B를 새로 생성하면 자동 저장됩니다.")
            else:
                labels={}
                for r in records:
                    edited=" · ✏️수정됨" if r.get("manually_edited") else ""
                    label=f"{r.get('title','제목 없음')}{edited} · {str(r.get('created_at',''))[:16]}"
                    labels[label]=r["id"]

                selected_label=st.selectbox("불러올 모의고사",list(labels.keys()))
                exam_id=labels[selected_label]
                rec=get_exam(url,skey,exam_id)

                if rec:
                    c1,c2,c3=st.columns(3)
                    with c1:
                        if st.button("현재 화면으로 불러오기",use_container_width=True):
                            st.session_state["A"]=rec["exam_a"]
                            st.session_state["B"]=rec["exam_b"]
                            st.session_state["current_archive_id"]=rec["id"]
                            st.success("불러왔습니다. ③ A/B 생성 탭에서도 확인할 수 있습니다.")
                    with c2:
                        edit_mode=st.toggle("수정 모드",value=False,key=f"edit_{exam_id}")
                    with c3:
                        delete_mode=st.toggle("삭제 모드",value=False,key=f"delete_{exam_id}")

                    st.markdown("#### 보관 정보")
                    if not edit_mode:
                        st.write("**제목:**",rec.get("title",""))
                        st.write("**메모:**",rec.get("note","") or "-")
                        st.write("**모델:**",rec.get("model",""))
                        st.write("**시드:**",rec.get("seed",""))
                        st.write("**영역:**",", ".join(rec.get("domains") or []))
                        if rec.get("manually_edited"):
                            st.warning("수동 수정된 시험입니다. 자동검증 통과 당시의 원본과 내용이 달라질 수 있습니다.")

                    if edit_mode:
                        st.warning("문제·정답을 직접 수정하면 자동검증 보장은 해제됩니다. 저장 후 '수정됨' 표시가 붙습니다.")
                        title=st.text_input("제목",value=rec.get("title",""),key=f"title_{exam_id}")
                        note=st.text_area("메모",value=rec.get("note",""),key=f"note_{exam_id}",height=80)
                        st.markdown("##### 전공 A 수정")
                        edited_A=exam_editor(rec["exam_a"],f"{exam_id}_A")
                        st.markdown("##### 전공 B 수정")
                        edited_B=exam_editor(rec["exam_b"],f"{exam_id}_B")
                        if st.button("수정 내용 저장",type="primary",use_container_width=True):
                            errors=[]
                            for sec,ex in [("A",edited_A),("B",edited_B)]:
                                for q in ex.get("questions",[]):
                                    if len(q.get("tasks",[])) != len(q.get("answer",[])):
                                        errors.append(f"{sec} {q.get('number')}번: 작성 방법 수와 정답 요소 수가 다릅니다.")
                                    if sum(q.get("subpoints",[])) != q.get("points",0):
                                        errors.append(f"{sec} {q.get('number')}번: 부분점수 합이 배점과 다릅니다.")
                            if errors:
                                for e in errors: st.error(e)
                            else:
                                update_exam(url,skey,exam_id,{
                                    "title":title.strip() or rec.get("title","제목 없음"),
                                    "note":note,
                                    "exam_a":edited_A,
                                    "exam_b":edited_B,
                                    "manually_edited":True,
                                })
                                st.session_state["A"]=edited_A
                                st.session_state["B"]=edited_B
                                st.success("보관소 내용을 수정했습니다.")
                                st.rerun()

                    if delete_mode:
                        st.error("삭제하면 보관소에서 완전히 제거됩니다.")
                        confirm=st.checkbox("이 모의고사를 삭제하는 것에 동의합니다.",key=f"confirm_{exam_id}")
                        if st.button("선택한 모의고사 영구 삭제",disabled=not confirm,use_container_width=True):
                            delete_exam(url,skey,exam_id)
                            if st.session_state.get("current_archive_id")==exam_id:
                                st.session_state.pop("current_archive_id",None)
                            st.success("삭제했습니다.")
                            st.rerun()

                    st.markdown("#### 저장본 바로 다운로드")
                    for sec,keyname in [("A","exam_a"),("B","exam_b")]:
                        ex=rec[keyname]
                        c1,c2=st.columns(2)
                        c1.download_button(
                            f"보관본 전공 {sec} 문제지 PDF",
                            export_pdf(ex,False),
                            file_name=f"보관본_전공_{sec}_문제지.pdf",
                            mime="application/pdf",
                            use_container_width=True,
                            key=f"arch_q_{exam_id}_{sec}"
                        )
                        c2.download_button(
                            f"보관본 전공 {sec} 정답·해설 PDF",
                            export_pdf(ex,True),
                            file_name=f"보관본_전공_{sec}_정답해설.pdf",
                            mime="application/pdf",
                            use_container_width=True,
                            key=f"arch_a_{exam_id}_{sec}"
                        )
        except Exception as e:
            st.error(str(e))
            st.info("`supabase_archive_schema.sql` 실행 여부와 Streamlit Secrets의 SUPABASE_URL / SUPABASE_SERVICE_ROLE_KEY를 확인하세요.")

with tabs[4]:
    st.markdown("""
### 자동검증이 하는 일

**계산형**
1. Python이 조건과 정답을 동시에 생성합니다.
2. 문제에 사용되는 수치는 Python이 만든 값을 그대로 사용합니다.
3. AI가 계산한 값은 정답으로 채택하지 않습니다.

**개념형**
1. DB에서 정답과 원문 근거를 먼저 고정합니다.
2. 4점 문항은 AI 관계성 선별기가 같은 master concept로 묶일 anchor만 고릅니다.
3. 문항 작성 AI는 고정된 정답·원문 범위 안에서 자료와 작성 방법을 구성합니다.
4. 별도의 AI 출제검토위원이 정답 노출, 단순 베껴쓰기, 무관한 하위문항, 난도 부족, 모호성, 기출 유사성을 평가하여 탈락시킬 수 있습니다.
5. AI 심사를 통과한 뒤에도 정답/evidence/source/page 구조를 Python이 다시 검사합니다.
6. AI 품질심사가 꺼져 있으면 AI 생성 지문은 사용하지 않고 원문 잠금 폴백만 사용합니다.

**A/B**
- 동일 원리·공식은 concept-family로 묶어 A/B 중복을 막습니다.
- 4점 계산형은 단순 공식 대입만으로 끝나지 않도록 최소 3개 채점요소를 갖습니다.

**보관소 수동 수정**
- 사람이 저장본을 직접 수정하면 해당 저장본에는 '수정됨' 표시가 붙습니다.
- 수동 수정 이후에는 생성 당시의 자동검증을 그대로 보장하지 않습니다.
""")

with tabs[5]:
    st.subheader("실제 기출 기반 문항 구조")
    con=sqlite3.connect(DB)
    pats=con.execute("select id,points,name,verbs,visual,calc,provenance from exam_patterns order by points,id").fetchall()
    off=con.execute("select name,page_count from sources where kind='official_exam' order by name").fetchall()
    inv=con.execute("select count(*) from anchors where domain='발명'").fetchone()[0]
    con.close()
    st.write(f"실제 기출 원본 {len(off)}개가 구조 참고 자료로 별도 색인되어 있습니다. 발명 직접 앵커: {inv}개")
    st.dataframe([{"ID":r[0],"배점":r[1],"구조":r[2],"요구행동":r[3],"그림":bool(r[4]),"계산":bool(r[5])} for r in pats],use_container_width=True)
    st.warning("검증된 도식 템플릿이 없는 기술 그림과 회로는 AI가 자유 생성하지 않습니다.")
