
import os, json, sqlite3, tempfile, shutil, hashlib
from pathlib import Path
import streamlit as st
import fitz

from exam_builder import make_ab, make_section, DOMAINS
from pdf_export import export_pdf
from retrieval import search_pages

ROOT=Path(__file__).parent
DB=ROOT/"knowledge.db"

st.set_page_config(page_title="기술 임용 자동검증 모의고사",layout="wide")
st.title("기술 임용 A/B 자동검증 모의고사 생성기")
st.caption("서브노트는 출제범위·정답 근거, 기출은 출제 방식, Python은 계산 검산, AI는 문항 표현만 담당합니다.")

def api_key():
    try: return st.secrets["OPENAI_API_KEY"]
    except Exception: return os.getenv("OPENAI_API_KEY","")

def db_stats():
    con=sqlite3.connect(DB)
    a=con.execute("select count(*) from sources").fetchone()[0]
    p=con.execute("select count(*) from pages").fetchone()[0]
    an=con.execute("select count(*) from anchors").fetchone()[0]
    kinds=dict(con.execute("select kind,count(*) from sources group by kind").fetchall())
    con.close()
    return a,p,an,kinds

tabs=st.tabs(["① DB 상태","② 출제범위 검색","③ A/B 생성","④ 검증 원리"])

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
    st.info("배포 버전에는 PDF 원본을 넣지 않고, 이 SQLite 지식 DB만 포함합니다. 따라서 매번 PDF를 다시 올릴 필요가 없습니다.")

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
        model=st.text_input("OpenAI 모델",value="gpt-5.6-luna")
    with c3:
        difficulty=st.selectbox("난이도",["기본","적당히 어려움","어려움"],index=1)
        circuit_policy=st.selectbox("회로 문제",["완전 제외","최대한 제외","허용"],index=1)
        seed=st.number_input("랜덤 시드(재현용)",min_value=0,value=20260829,step=1)
    st.caption("기본값은 '적당히 어려움 + 회로 최대한 제외'입니다. 계산형은 섹션당 약 20%로 제한합니다. 4점은 실제 임용형 자료·작성방법 구조를 우선하고, AI가 실패하면 원문 검증형 4점 문항으로 자동 대체하여 전체 생성이 중단되지 않게 했습니다.")
    key=api_key()
    if use_ai and not key:
        st.warning("OPENAI_API_KEY가 없어 이번 생성은 AI 없이 안전모드로 동작합니다.")
    if st.button("전공 A + B 생성",type="primary",use_container_width=True):
        with st.spinner("정답 고정 → 문항 생성 → 근거 대조 → 중복 검사 → A/B 편성 중..."):
            try:
                A,B=make_ab(DB,domains=domains,api_key=key,model=model,
                            ai_enabled=bool(use_ai and key),seed=int(seed),difficulty=difficulty,circuit_policy=circuit_policy)
                st.session_state["A"]=A; st.session_state["B"]=B
                st.success("A/B 모두 자동검증을 통과했습니다.")
                a_stats=A.get("generation_stats",{})
                b_stats=B.get("generation_stats",{})
                st.caption(
                    f"생성 통계 · 계산형 {a_stats.get('formula_questions',0)+b_stats.get('formula_questions',0)}문항 · "
                    f"AI 호출 {a_stats.get('ai_calls',0)+b_stats.get('ai_calls',0)}회 · "
                    f"안전 폴백 {a_stats.get('safe_fallbacks',0)+b_stats.get('safe_fallbacks',0)}문항"
                )
            except Exception as e:
                st.error(str(e))
    if "A" in st.session_state and "B" in st.session_state:
        for sec in ["A","B"]:
            exam=st.session_state[sec]
            st.markdown(f"### 전공 {sec}")
            for q in exam["questions"]:
                badge="🧮 Python 검산" if q.get("verifier")=="python" else "📚 원문 근거검증"
                with st.expander(f"{q['number']}번 · {q['domain']} · {q['topic']} · {q['points']}점 · {badge}"):
                    st.write(q["passage"])
                    if q.get("conditions"):
                        st.markdown("**<조건>**")
                        for x in q["conditions"]: st.write("○",x)
                    st.markdown("**<작성 방법>**")
                    for x in q["tasks"]: st.write("○",x)
                    with st.popover("정답/검증근거"):
                        st.write("정답:",q["answer"])
                        st.write("해설:",q["solution"])
                        if q.get("verifier")=="source":
                            st.write("출처:",q.get("source_name"),"p.",q.get("page_no"))
                            st.write("근거:",q.get("evidence"))
            ppdf=export_pdf(exam,False); apdf=export_pdf(exam,True)
            c1,c2=st.columns(2)
            c1.download_button(f"전공 {sec} 문제지 PDF",ppdf,file_name=f"전공_{sec}_문제지.pdf",mime="application/pdf",use_container_width=True)
            c2.download_button(f"전공 {sec} 정답·해설 PDF",apdf,file_name=f"전공_{sec}_정답해설.pdf",mime="application/pdf",use_container_width=True)

with tabs[3]:
    st.markdown("""
### 자동검증이 하는 일

**계산형**
1. Python이 조건과 정답을 동시에 생성합니다.
2. 문제에 사용되는 수치는 Python이 만든 값을 그대로 사용합니다.
3. AI가 계산한 값은 정답으로 채택하지 않습니다.

**개념형**
1. DB에서 원문 정답(anchor)과 근거문장(evidence)을 먼저 뽑습니다.
2. AI에는 정답을 바꾸지 못하도록 고정값으로 전달합니다.
3. 생성 후 정답이 evidence 안에 있는지, evidence가 실제 해당 PDF 페이지에 있는지 다시 검사합니다.
4. 정답 용어가 문제 지문에 노출되면 폐기합니다.
5. AI 생성이 실패하면 원문 기반 보수적 문항으로 자동 대체합니다.

**A/B**
- 같은 topic/fingerprint가 A와 B에 중복되지 않도록 검사합니다.
- 검증을 통과하지 못한 문항은 시험지에 들어가지 않습니다.

> 이 구조는 오류 가능성을 크게 줄이지만, 자연어 문항의 '완전한 무오류'를 수학적으로 보장할 수는 없습니다.
> 대신 AI가 정답을 새로 만들어내는 권한을 최소화했습니다.
""")
