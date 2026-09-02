import copy
BUILDER_API_VERSION = "SAMPLE6-ONE-ANCHOR-T2-R19-20260902"

import random, math, re, sqlite3, itertools
from formula_templates import generate_formula_question
from retrieval import related_bundle,bundle_context,official_style_profile,candidate_cluster
from ai_wrapper import rewrite_bundle,safe_bundle_question
from validators import validate_formula_question,validate_grounded_question,too_similar,validate_exam,fingerprint
from patterns import blueprint,weighted_pick
from concept_families import families_for
from quality_judge import select_coherent_bundle,judge_question,judge_exam,judge_ab_pair

DOMAINS=["기술교육론","발명","제조기술","건설기술","생명기술","전기·전자","통신기술","재료역학","수송기술"]
FORMULA_DOMAINS={"재료역학","수송기술","통신기술"}



def _norm_anchor_text(s):
    s=re.sub(r"[\x00-\x1f]+"," ",str(s or ""))
    s=re.sub(r"\s+"," ",s).strip()
    return s

_REL_STOP={
    "기술","종류","특징","방법","과정","내용","관련","사용","이용","경우",
    "한다","있다","대한","통해","위한","의한","것을","에서","으로","되는",
    "설명","자료","다음","해당","구분","분류","단계","요소","기능","개념",
    "목적","장점","단점","효과","원리","구조","형태","상태","의미","활용",
    "적용","정의","특성","방식","구성","관계","조건","문제","예시","기타",
    "따른","따라","위해","때문","또는","그리고","등을","등의","정도"
}

def _anchor_tokens(*parts):
    text=" ".join(_norm_anchor_text(x).lower() for x in parts)
    toks=re.findall(r"[가-힣A-Za-z0-9]{2,}",text)
    return {t for t in toks if t not in _REL_STOP and not t.isdigit()}

def _topic_core(s):
    s=_norm_anchor_text(s).lower()
    s=re.sub(r"^[\(\[]?\d+[\)\].:\-]?\s*","",s)
    s=re.sub(r"^(신재생에너지의|신재생에너지|재생에너지의|재생에너지)\s+","",s)
    s=re.sub(r"[^가-힣a-z0-9]+","",s)
    return s

def _heading_like(text):
    x=_norm_anchor_text(text)
    if not x:
        return True
    bad_exact={
        "종류","특징","개념","목적","장점","단점","활용","분류","구조","구성",
        "형태","기능","원리","방법","과정","효과","특성","의미","내용"
    }
    if x in bad_exact:
        return True
    if re.search(r"(에\s*따라|에\s*따른|에\s*의한)\s*$",x):
        return True
    if re.search(r"(종류|분류|특징|장점|단점|구분|형태에 따른 분류)\s*$",x) and len(x)<=24:
        return True
    return False

def _anchor_quality_reason(a):
    topic=_norm_anchor_text(a.get("topic",""))
    answer=_norm_anchor_text(a.get("answer",""))
    evidence=_norm_anchor_text(a.get("evidence",""))
    if len(topic)<2:
        return "topic_too_short"
    if len(answer)<1:
        return "answer_empty"
    if len(evidence)<10:
        return "evidence_too_short"
    if len(topic)>110 or len(answer)>90:
        return "text_too_long"
    if topic.count("·")>=5:
        return "topic_fragmented"
    junk=[
        r"^[\(\[]?\d+[\)\].]?$",
        r"^[가-힣A-Za-z]\)$",
        r"^(참고|기타|정리|예시|종류|특징|개요|내용)$",
    ]
    if any(re.match(q,topic,re.I) for q in junk):
        return "heading_or_junk"
    if re.match(r"^[가-힣]\s+[가-힣A-Za-z0-9]{4,}(?:\s|$)",topic):
        return "ocr_broken_prefix"
    if re.match(r"^(?:cf\s*[\)\.\:]?|참고|부록|보충|심화|기타|예시)\b", topic, re.I):
        return "supplementary_topic"
    if _heading_like(answer):
        return "answer_is_heading"

    acore=_topic_core(answer)
    ecore=_topic_core(evidence)
    if len(acore)>=3 and acore not in ecore:
        atoks=_anchor_tokens(answer)
        etoks=_anchor_tokens(evidence)
        if atoks and not (atoks & etoks):
            return "answer_not_supported_in_evidence"
    return ""


def _strip_anchor_noise(text):
    t=str(text or "").strip()
    t=re.sub(r"^[\s·•○●◦▪■□◆◇▶▷→\-–—:;|/\\]+","",t)
    t=re.sub(r"[\s·•○●◦▪■□◆◇▶▷→\-–—:;|/\\]+$","",t)
    t=re.sub(r"\s+"," ",t).strip()
    return t


def _anchor_internal_contradiction_reason(a):
    topic=_strip_anchor_noise(a.get("topic",""))
    answer=_strip_anchor_noise(a.get("answer",""))
    text=f"{topic} {answer}"

    if re.search(r"면심입방격자\s*\(\s*BCC\s*\)", text, re.I):
        return "fcc_bcc_name_mismatch"
    if re.search(r"체심입방격자\s*\(\s*FCC\s*\)", text, re.I):
        return "bcc_fcc_name_mismatch"
    return ""

def _anchor_fragment_reason(a):
    """
    R16.1 conservative fragment detector.
    Leading bullets are normalization noise, not a rejection reason.
    Reject only clearly incomplete parser fragments / grammatical location fragments.
    """
    topic=_strip_anchor_noise(a.get("topic",""))
    answer=_strip_anchor_noise(a.get("answer",""))

    if not topic or not answer:
        return "empty_topic_or_answer"

    # Obvious parser remnants such as "(If", "(V", lone bracketed tokens.
    if re.fullmatch(r"[\(\[]\s*[A-Za-z]{1,4}", topic) or re.fullmatch(r"[\(\[]\s*[A-Za-z]{1,4}", answer):
        return "parser_fragment"
    if re.fullmatch(r"[\(\[][^)\]]{0,5}", topic) or re.fullmatch(r"[\(\[][^)\]]{0,5}", answer):
        return "unclosed_parser_fragment"

    # Grammatical/location fragment: require a boundary/space so lexical words
    # like '광전효과' are never mistaken for ending particle '과'.
    if len(answer) <= 18:
        spaced_endings = (
            " 내"," 외"," 에"," 에서"," 으로"," 로"," 와"," 과"," 의",
            " 및"," 때"," 경우"," 위해"," 통해"," 따라"," 대한"," 관한"
        )
        if any(answer.endswith(x) for x in spaced_endings):
            return "grammatical_fragment"

    if re.fullmatch(r"(실린더|기관|회로|재료|구조물|시스템|장치|부품|공간)\s*(내|안|밖|외)", answer):
        return "generic_place_fragment"

    if topic == answer and answer in {
        "내부","외부","위치","부분","구간","영역","장소","공간",
        "실린더 내","기관 내","회로 내","재료 내","장치 내",
    }:
        return "generic_identity"

    if answer in {"내","외","안","밖","위","아래","앞","뒤","중","부분","위치","영역","구간"}:
        return "relational_only"

    return ""
def _bundle_anchor_integrity(chosen):
    reasons=[]
    for x in chosen:
        r=_anchor_fragment_reason(x)
        if r:
            reasons.append({
                "topic":str(x.get("topic","")),
                "answer":str(x.get("answer","")),
                "reason":r,
            })
    return (len(reasons)==0), reasons

def _independent_scoring_targets(chosen):
    answers=[_strip_anchor_noise(x.get("answer","")) for x in chosen]
    bad={"내부","외부","위치","부분","구간","영역","장소","공간","내","외","안","밖"}
    if any(a in bad for a in answers):
        return False
    if len(set(answers)) != len(answers):
        return False
    return True

def _anchor_ok(a):
    if _anchor_fragment_reason(a):
        return False
    if _anchor_internal_contradiction_reason(a):
        return False
    return _anchor_quality_reason(a)==""
def _near_duplicate_anchor(a,b):
    at=_topic_core(a.get("topic",""))
    bt=_topic_core(b.get("topic",""))
    aa=_topic_core(a.get("answer",""))
    ba=_topic_core(b.get("answer",""))
    for x,y in ((at,bt),(aa,ba),(at,ba),(aa,bt)):
        if not x or not y:
            continue
        if x==y:
            return True
        short,long=(x,y) if len(x)<=len(y) else (y,x)
        if len(short)>=4 and short in long and len(short)/max(1,len(long))>=0.60:
            return True
    ta=_anchor_tokens(a.get("topic"),a.get("answer"))
    tb=_anchor_tokens(b.get("topic"),b.get("answer"))
    if ta and tb:
        j=len(ta & tb)/max(1,len(ta | tb))
        if j>=0.82:
            return True
    return False

def _cross_reference_strength(a,b):
    ae=_norm_anchor_text(a.get("evidence","")).lower()
    be=_norm_anchor_text(b.get("evidence","")).lower()
    akeys=_anchor_tokens(a.get("topic"),a.get("answer"))
    bkeys=_anchor_tokens(b.get("topic"),b.get("answer"))
    cross=0
    for t in list(akeys)[:10]:
        if len(t)>=3 and t in be:
            cross+=1
    for t in list(bkeys)[:10]:
        if len(t)>=3 and t in ae:
            cross+=1
    ea=_anchor_tokens(a.get("evidence",""))
    eb=_anchor_tokens(b.get("evidence",""))
    shared={t for t in (ea & eb) if len(t)>=3}
    return cross,shared

def _pair_relation_score(a,b):
    """
    강한 relation 신호가 없으면 같은 페이지여도 REJECT.
    """
    if not _anchor_ok(a) or not _anchor_ok(b):
        return -999.0
    if _near_duplicate_anchor(a,b):
        return -999.0
    if str(a.get("source_name","")) != str(b.get("source_name","")):
        return -999.0
    try:
        gap=abs(int(a.get("page_no",0))-int(b.get("page_no",0)))
    except Exception:
        return -999.0
    if gap>2:
        return -999.0

    cross,shared=_cross_reference_strength(a,b)
    if cross==0 and len(shared)<2:
        return -999.0

    score=cross*3.0 + min(8.0,len(shared)*2.0)
    if gap==0:
        score+=1.0
    elif gap==1:
        score+=0.5

    ta=_anchor_tokens(a.get("topic"),a.get("answer"))
    tb=_anchor_tokens(b.get("topic"),b.get("answer"))
    score += min(3.0,len(ta & tb)*1.5)
    return score

def _bundle_connected(bundle,min_edge=4.0):
    n=len(bundle)
    if n<=1:
        return True,0.0
    adj=[set() for _ in range(n)]
    scores=[]
    for i in range(n):
        for j in range(i+1,n):
            s=_pair_relation_score(bundle[i],bundle[j])
            if s>=min_edge:
                adj[i].add(j); adj[j].add(i); scores.append(s)
    seen={0}; stack=[0]
    while stack:
        x=stack.pop()
        for y in adj[x]:
            if y not in seen:
                seen.add(y); stack.append(y)
    if len(seen)!=n:
        return False,0.0
    return True,(sum(scores)/len(scores) if scores else 0.0)

def _pattern_thinking_types(pattern_id):
    pid=str(pattern_id or "").upper()
    return {
        "T2_REL":["관계판단","근거서술"],
        "T2_ERR":["오류판단","수정"],
        "T2_CMP":["비교판단","구분근거"],
        "T2_DATA":["자료해석","적용판단"],
        "T4_DATA112":["자료해석","개념판단","적용"],
        "T4_ERR22":["오류판단","수정","근거서술"],
        "T4_112":["개념판단","관계설명","적용"],
    }.get(pid,["자료해석","관계판단"])


def _source_kind_map(con):
    try:
        cols={r[1] for r in con.execute("PRAGMA table_info(sources)").fetchall()}
        if {"name","kind"} <= cols:
            return {str(r[0]):str(r[1] or "") for r in con.execute("SELECT name,kind FROM sources").fetchall()}
    except Exception:
        pass
    return {}

def _primary_source(source_name, kind=""):
    """
    정답/출제내용은 사용자가 넣은 서브노트 계열만 사용한다.
    공식기출/기출풀이/모의고사는 문제 구조·스타일 참고용이지 정답 후보가 아니다.
    """
    name=_norm_anchor_text(source_name).lower()
    k=_norm_anchor_text(kind).lower()
    if k=="subnote":
        return True
    allow=("서브노트","교과서 정리","[역학] part","자동차-에너지")
    deny=("기출 문제 풀이","모의고사","전공 a","전공 b","official")
    if any(x in name for x in deny):
        return False
    return any(x in name for x in allow)

def _page_text_map(con, source_names):
    names=[str(x) for x in source_names if str(x)]
    if not names:
        return {}
    try:
        cols={r[1] for r in con.execute("PRAGMA table_info(pages)").fetchall()}
        if not {"source_name","page_no","text"} <= cols:
            return {}
        qmarks=",".join("?" for _ in names)
        rows=con.execute(
            f"SELECT source_name,page_no,text FROM pages WHERE source_name IN ({qmarks})",
            names
        ).fetchall()
        return {(str(r[0]),int(r[1])):_norm_anchor_text(r[2]) for r in rows}
    except Exception:
        return {}


def _page_raw_text_map(con, source_names):
    names=[str(x) for x in source_names if str(x)]
    if not names:
        return {}
    try:
        cols={r[1] for r in con.execute("PRAGMA table_info(pages)").fetchall()}
        if not {"source_name","page_no","text"} <= cols:
            return {}
        qmarks=",".join("?" for _ in names)
        rows=con.execute(
            f"SELECT source_name,page_no,text FROM pages WHERE source_name IN ({qmarks})",
            names
        ).fetchall()
        return {(str(r[0]),int(r[1])):str(r[2] or "") for r in rows}
    except Exception:
        return {}

def _subnote_importance_score(a, page_text=""):
    """
    서브노트 안에서도 주변 참고사항보다 핵심/기출표시가 있는 내용을 우선한다.
    점수는 '정답의 진위'를 바꾸지 않고 후보 순위에만 사용한다.
    """
    topic=_norm_anchor_text(a.get("topic",""))
    answer=_norm_anchor_text(a.get("answer",""))
    evidence=_norm_anchor_text(a.get("evidence",""))
    page=_norm_anchor_text(page_text)
    blob=" ".join((topic,evidence,page))

    score=0.0
    # 서브노트에 실제 기출 연도/전공 표기가 있으면 강한 우선 신호
    if (
        re.search(r"★?\s*(?:20)?\d{2}\s*[AB]", blob, re.I)
        or re.search(r"\b\d{2}[AB]\s*\(", blob, re.I)
        or re.search(r"기금\s*\d{2}", blob)
    ):
        score += 3.0
    # 공식/원리/과정/비교처럼 연결형 문항으로 만들기 좋은 핵심 서술
    if re.search(r"(공식|관계식|원리|과정|순서|구조|작동|작용|원인|결과|조건|비교|차이|장단점|특징)", blob):
        score += 0.8
    # 너무 일반적인 단답은 후순위
    if answer.strip().lower() in {"찬성","반대","예","아니오","가능","불가능","장점","단점"}:
        score -= 3.0
    # 보충/참고 표시는 강한 감점(대부분 _anchor_quality_reason에서 이미 제외됨)
    if re.match(r"^(?:cf\\s*[\\)\\.\\:]?|참고|부록|보충|심화|기타|예시)\\b", topic, re.I):
        score -= 6.0
    try:
        score += min(0.8,max(0.0,float(a.get("confidence") or 0))*0.8)
    except Exception:
        pass
    return score



def _past_exam_strength(a, page_text=""):
    local=" ".join([
        _norm_anchor_text(a.get("topic","")),
        _norm_anchor_text(a.get("evidence","")),
    ])
    if (
        re.search(r"★?\s*(?:20)?\d{2}\s*[AB]",local,re.I)
        or re.search(r"\b\d{2}[AB]\s*\(",local,re.I)
        or re.search(r"기금\s*\d{2}",local)
        or re.search(r"(임용|기출)",local)
    ):
        return 5
    if re.search(r"\b(?:19|20)\d{2}\b",local):
        return 4
    if re.search(r"(출제|문제풀이|기출유사)",local):
        return 3
    page=_norm_anchor_text(page_text)
    if re.search(r"(임용|기출|★?\s*(?:20)?\d{2}\s*[AB]|\b(?:19|20)\d{2}\b)",page,re.I):
        return 2
    return 0

def _core_subnote_importance(a, page_text=""):
    topic=_norm_anchor_text(a.get("topic",""))
    ev=_norm_anchor_text(a.get("evidence",""))
    local=" ".join([topic,ev])
    if re.search(r"(cf\)|참고|보충|부록|심화|기타|예시)",local,re.I):
        return 0
    score=2
    if re.search(r"(공식|정리|핵심|원리|법칙|과정|공정|구조|작동|특징|비교)",local):
        score+=1
    if re.search(r"(표|단계|조건|관계|계산|식\b|=)",local):
        score+=1
    if _past_exam_strength(a,page_text)>=4:
        score+=1
    return max(0,min(5,score))

def _representative_concept_score(a, page_text=""):
    topic=_norm_anchor_text(a.get("topic",""))
    ev=_norm_anchor_text(a.get("evidence",""))
    ans=_norm_anchor_text(a.get("answer",""))
    blob=" ".join([topic,ev,_norm_anchor_text(page_text)])
    score=2
    if re.search(r"(원리|법칙|공식|구조|작동|공정|과정|방법|장치|회로|극성|응력|변형|압력|전압|전류|저항|기어|정류|단편|MTU|용접)",blob,re.I):
        score+=2
    if re.search(r"(대표|기본|핵심|주요|일반)",blob):
        score+=1
    if re.search(r"(특정|고유|예외|희귀|특수|세부|사례명|종명|균명)",blob):
        score-=2
    if len(ans)<=18 and re.search(r"(종류|명칭|이름|균|미생물|고유명)",topic+ans):
        score-=1
    return max(0,min(5,score))

def _repeatability_score(a, page_text=""):
    topic=_norm_anchor_text(a.get("topic",""))
    ev=_norm_anchor_text(a.get("evidence",""))
    ans=_norm_anchor_text(a.get("answer",""))
    blob=" ".join([topic,ev,_norm_anchor_text(page_text)])
    score=1
    if re.search(r"(계산|공식|조건|비교|오류|원인|결과|관계|과정|단계|작동|설계|판단|적용|변화)",blob):
        score+=2
    if re.search(r"\d",blob) and re.search(r"(Pa|MPa|N\b|kN|V\b|A\b|W\b|Hz|byte|bit|mm|cm|m\b|kg|℃|%|MTU)",blob,re.I):
        score+=1
    if re.search(r"(극성|용입|응력|변형|압력|전압|전류|저항|기어비|정류|단편화|MTU|효율|열처리|공정)",blob,re.I):
        score+=1
    if len(ans)<=18 and not re.search(r"(계산|조건|관계|오류|적용|비교|과정|원인|결과)",blob):
        score-=1
    return max(0,min(5,score))

def _concept_centrality_score(a, page_text=""):
    topic=_norm_anchor_text(a.get("topic",""))
    ev=_norm_anchor_text(a.get("evidence",""))
    blob=" ".join([topic,ev,_norm_anchor_text(page_text)])
    score=2
    if re.search(r"(원리|법칙|공식|구조|관계|작동|과정|공정|기초|기본)",blob):
        score+=2
    if re.search(r"(응력|변형|전압|전류|저항|압력|에너지|효율|극성|용입|기어|단편화|열처리)",blob,re.I):
        score+=1
    if re.search(r"(특정 사례|예시|고유|예외|특수)",blob):
        score-=1
    return max(0,min(5,score))

def _peripherality_score(a, page_text=""):
    topic=_norm_anchor_text(a.get("topic",""))
    ev=_norm_anchor_text(a.get("evidence",""))
    ans=_norm_anchor_text(a.get("answer",""))
    local=" ".join([topic,ev])
    score=0
    if re.search(r"(cf\)|참고|보충|부록|심화|기타|예시)",local,re.I):
        score+=3
    if re.search(r"(특정|고유|예외|희귀|특수|세부|사례명|종명|균명)",local):
        score+=2
    if len(ans)<=18 and re.search(r"(미생물|균|종류|명칭|이름|고유명)",topic+ans):
        score+=1
    if re.search(r"(원리|법칙|공식|구조|작동|공정|과정|주요|핵심)",local):
        score-=1
    page=_norm_anchor_text(page_text)
    if score<=0 and re.search(r"(cf\)|참고|보충|부록|심화|예시)",page,re.I):
        score+=1
    return max(0,min(5,score))

def _core_exam_score(a, page_text=""):
    # 사용자 확정 가중치:
    # 기출성 4 / 서브노트 중요도 3 / 기본·대표 개념성 3 /
    # 반복 출제 가능성 4 / 연결 중심성 3 / 지엽성 -2
    past=_past_exam_strength(a,page_text)
    note=_core_subnote_importance(a,page_text)
    rep=_representative_concept_score(a,page_text)
    repeat=_repeatability_score(a,page_text)
    central=_concept_centrality_score(a,page_text)
    peripheral=_peripherality_score(a,page_text)

    raw=(past*4)+(note*3)+(rep*3)+(repeat*4)+(central*3)-(peripheral*2)
    qualified=(past>=3 or rep>=4 or repeat>=4)

    if qualified and raw>=58:
        tier="CORE"
    elif qualified and raw>=40:
        tier="NORMAL"
    else:
        tier="SUPPORT"

    return {
        "score":float(raw),"tier":tier,"qualified":bool(qualified),
        "past_exam":past,"subnote_importance":note,"representative":rep,
        "repeatability":repeat,"centrality":central,"peripherality":peripheral,
    }

def _exam_value_score(a, page_text=""):
    """
    API 호출 전에 서브노트 후보의 '임용 출제 가치'를 점수화한다.
    정답의 진위를 바꾸지 않고 후보 순위만 정한다.
    """
    topic=_norm_anchor_text(a.get("topic",""))
    answer=_norm_anchor_text(a.get("answer",""))
    evidence=_norm_anchor_text(a.get("evidence",""))
    page=_norm_anchor_text(page_text)
    blob=" ".join((topic,evidence,page))

    score=0.0

    # 실제 기출 표시: 중요도 강한 신호
    if (
        re.search(r"★?\s*(?:20)?\d{2}\s*[AB]", blob, re.I)
        or re.search(r"\b\d{2}[AB]\s*\(", blob, re.I)
        or re.search(r"기금\s*\d{2}", blob)
    ):
        score += 2.5

    # 임용에서 연결형/적용형으로 만들기 좋은 지식
    high_groups = [
        r"(원인|결과|영향|때문|따라서|증가|감소|변화)",
        r"(조건|경우|이상|이하|초과|미만|온도|압력|속도|하중|응력|전압|전류)",
        r"(원리|법칙|관계식|공식|방정식|계산|비|율|효율)",
        r"(과정|순서|단계|작동|작용|흐름|전달|변환)",
        r"(비교|차이|반면|보다|구분|오류|수정)",
        r"(구조|기능|재료|성질|결정|조직|결함)",
    ]
    score += 0.75 * sum(1 for p in high_groups if re.search(p, blob))

    # 숫자/단위/수식이 있으면 계산·조건판단 문항으로 만들 가능성이 큼
    if re.search(r"\d", blob) and re.search(r"(mm|cm|m\b|Pa|MPa|N\b|kN|V\b|A\b|W\b|Hz|byte|℃|%|ρ|σ|η|Q|MTU)", blob, re.I):
        score += 1.25

    # 너무 일반적인 정답, 찬반·단순 레이블은 강한 감점
    if answer.strip().lower() in {
        "찬성","반대","예","아니오","가능","불가능","장점","단점",
        "높다","낮다","크다","작다","증가","감소"
    }:
        score -= 3.5

    # 단순 '명칭/종류'만 있는 heading성 topic은 후순위
    if re.search(r"(명칭|종류|분류|이름)\s*$", topic) and not re.search(r"(원리|관계|조건|과정|오류|비교)", blob):
        score -= 1.5

    # R11: 2점에서 특히 문제가 되었던 '정의/특징 한 줄 → 용어명 회상' 후보를 더 강하게 낮춘다.
    # 출제 자체를 막지는 않고, 같은 분야에 판단·관계·오류수정 가능한 후보가 있으면 그쪽을 우선한다.
    short_ev = len(evidence) <= 120
    naming_topic = bool(re.search(r"(정의|명칭|종류|분류|의미|무엇)", topic))
    relational = bool(re.search(r"(원인|결과|관계|조건|과정|오류|비교|차이|영향|계산|적용)", blob))
    if short_ev and naming_topic and not relational:
        score -= 2.0

    return score


def _natural_unit_score(chosen):
    """
    같은 주제라는 이유만으로 억지로 묶지 않고,
    서브노트 근거 자체에 원래 존재하는 하나의 문제해결 단위인지 점수화한다.
    """
    chosen=list(chosen or [])
    if not chosen:
        return -99.0

    topics=[_norm_anchor_text(x.get("topic","")) for x in chosen]
    evidences=[_norm_anchor_text(x.get("evidence","")) for x in chosen]
    blob=" ".join(evidences)

    score=0.0

    # 같은 출처/근접 페이지는 약한 보조 신호
    if len({str(x.get("source_name","")) for x in chosen})==1:
        score += 0.8
    pages=[int(x.get("page_no",0) or 0) for x in chosen]
    if pages and max(pages)-min(pages)<=1:
        score += 0.8

    # 실제 인과·과정·조건·계산 연결어가 여러 근거에 반복 등장하면 자연적 단위 가능성이 높다.
    relation_terms=("때문","따라","따라서","증가","감소","변화","조건","과정","단계",
                    "이용","계산","결과","원인","연결","작용","전달","변환","비교","오류")
    score += 0.35 * sum(1 for t in relation_terms if t in blob)

    # 같은 기술명/계통명을 공유하는 병렬 개념은 하나의 자연스러운 출제 단위다.
    # 예: 테브난 등가전압 / 테브난 등가저항, 전기자 철심 / 전기자 권선.
    sibling_bonus=0.0
    clean_topics=[_strip_anchor_noise(t) for t in topics]
    first_tokens=[]
    for t in clean_topics:
        m=re.match(r"([가-힣A-Za-z0-9]+)",t)
        first_tokens.append(m.group(1) if m else "")
    meaningful=[x for x in first_tokens if len(x)>=2]
    if len(meaningful)>=2:
        from collections import Counter
        common=Counter(meaningful).most_common(1)
        if common and common[0][1]>=2:
            sibling_bonus += 1.7
    score += sibling_bonus

    # 한 anchor의 topic/answer가 다른 anchor evidence에 직접 등장하면 강한 자연 연결
    cross=0
    for i,a in enumerate(chosen):
        ta=_topic_core(a.get("topic",""))
        aa=_topic_core(a.get("answer",""))
        for j,b in enumerate(chosen):
            if i==j:
                continue
            ev=_topic_core(b.get("evidence",""))
            if (ta and ta in ev) or (aa and aa in ev):
                cross += 1
    score += min(4.0,cross*1.25)

    # '서로 다른 독립 사례/종류/활용 예시'를 나열하는 성격은 감점
    if sum(1 for t in topics if re.search(r"(활용|사례|종류|예시|용도|분류)",t)) >= 2:
        score -= 2.0

    # 찬반/단순명칭 anchor가 섞이면 강한 감점
    simple_answers={"찬성","반대","예","아니오","장점","단점"}
    if any(_norm_anchor_text(x.get("answer","")).strip() in simple_answers for x in chosen):
        score -= 2.5

    return score


def _answer_is_simple_label(a):
    ans=_norm_anchor_text(a.get("answer","")).strip()
    topic=_norm_anchor_text(a.get("topic","")).strip()
    ev=_norm_anchor_text(a.get("evidence","")).strip()
    if not ans:
        return True

    # 실제 계산값/식/단위가 핵심인 답은 명칭회상으로 보지 않는다.
    # '단상 2선식'처럼 명칭 안에 숫자가 들어가는 경우는 여기서 제외하지 않는다.
    numeric_result=bool(
        re.search(r"[=/%℃°]",ans)
        or re.search(r"\d+(?:\.\d+)?\s*(?:Pa|MPa|kPa|N|kN|V|A|W|Hz|byte|bit|mm|cm|m|kg|s)\b",ans,re.I)
    )
    if numeric_result:
        return False

    # 설명형/관계형 정답은 단순 레이블이 아니다.
    if len(ans)>=28 or re.search(r"(때문|따라|증가|감소|변화|관계|오류|영향|과정|작용|이용|조건)",ans):
        return False

    short_answer=len(ans)<=18
    naming_hint=bool(re.search(r"(명칭|종류|분류|정의|의미|기법|방법|장치|재료|구조|방식|현상)",topic))
    definition_like=bool(
        len(ev)<=90
        and not re.search(r"(때문|따라서|조건|경우|과정|단계|원인|결과|영향|작용|계산|비교|차이|오류|적용)",ev)
    )
    return bool(short_answer and (naming_hint or definition_like))

def _two_point_bundle_penalty(chosen):
    chosen=list(chosen or [])
    if len(chosen)<2:
        return 0.0,False
    flags=[_answer_is_simple_label(x) for x in chosen[:2]]
    if all(flags):
        return -99.0,True
    if any(flags):
        return -1.75,False
    return 1.0,False

def _compact_material_limits(points):
    # 실제 임용형 자료는 '정보량'이 아니라 '판단'으로 난도를 만든다.
    if int(points)==2:
        return {
            "passage_max_chars":520,
            "conditions_max":2,
            "tasks_max_chars_each":140,
            "policy":"핵심 상황 1개 + 필요한 근거만. 장황한 설명 금지."
        }
    return {
        "passage_max_chars":900,
        "conditions_max":3,
        "tasks_max_chars_each":170,
        "policy":"하나의 상황/과정/장치 중심. 독립 사실 나열 금지."
    }

def _skeleton_for_pattern(pattern_id):
    """
    실제 임용에서 자주 보이는 '문항 골격'만 고정한다.
    내용/정답은 서브노트에서만 가져온다.
    """
    pid=str(pattern_id or "").upper()
    return {
        "T2_REL":"현상/조건 자료 → 첫 판단 → 그 판단을 근거로 결과·관계 판단",
        "T2_ERR":"학생 설명 제시 → 오류 판단 → 같은 근거로 올바르게 수정",
        "T2_CMP":"두 사례/조건 제시 → 차이 판단 → 구분 근거 설명",
        "T2_DATA":"자료의 조건 해석 → 해석 결과를 실제 판단에 적용",
        "T4_DATA112":"자료 해석 → 중간 개념/값 판단 → 앞의 결과를 이용한 적용·결과예측",
        "T4_ERR22":"학생의 오류 판단 → 수정 → 두 판단을 관통하는 공통 원리·근거 설명",
        "T4_112":"조건/현상 해석 → 두 개념의 관계 판단 → 앞의 판단을 함께 이용한 적용",
    }.get(pid,"자료 해석 → 관계 판단 → 적용")

def _scoring_plan(pattern, chosen):
    """
    AI가 먼저 긴 지문을 만들지 않도록 채점 논리를 Python에서 선결정한다.
    정답 자체는 DB anchor 그대로 고정한다.
    """
    sp=list(pattern.get("subpoints",[]))
    topics=[_norm_anchor_text(x.get("topic","")) for x in chosen]
    answers=[_norm_anchor_text(x.get("answer","")) for x in chosen]
    rows=[]
    for i,(pt,topic,ans) in enumerate(zip(sp,topics,answers),1):
        if len(sp)==2:
            if i==1:
                role="하나의 핵심 개념/방법/원리를 짧은 자료에서 판단"
            else:
                role="첫 판단과 같은 핵심개념에 대해 근거·오류수정·비교·적용 중 하나를 수행"
        else:
            if i==1:
                role="자료에서 핵심 조건/오류/차이를 해석·판단"
            elif i==len(sp):
                role="앞선 판단을 실제로 이용해 관계·적용·결과를 도출"
            else:
                role="첫 판단과 마지막 적용을 이어 주는 중간 판단"
        rows.append({
            "order":i,"points":int(pt),"topic":topic,
            "fixed_answer":ans,"role":role
        })
    return rows

def _direct_chain_order(bundle):
    """
    4점 문항은 '같은 분야'가 아니라 실제 교차참조가 이어지는 사슬이어야 한다.
    인접 두 개념 사이에 최소 1개의 직접 교차참조가 있는 순서를 찾는다.
    """
    b=list(bundle or [])
    if len(b)<=1:
        return b,0.0
    best=None
    for perm in itertools.permutations(b):
        total=0.0
        ok=True
        for i in range(len(perm)-1):
            cross,_shared=_cross_reference_strength(perm[i],perm[i+1])
            rel=_pair_relation_score(perm[i],perm[i+1])
            if cross < 1 or rel < 4.0:
                ok=False
                break
            total += rel + cross
        if ok and (best is None or total>best[0]):
            best=(total,list(perm))
    return (best[1],best[0]) if best else ([],0.0)

def _relation_directive(pattern_id, chosen):
    topics=[_norm_anchor_text(x.get("topic","")) for x in chosen]
    chain=" → ".join(x for x in topics if x)
    if str(pattern_id).upper().startswith("T4"):
        return (
            "4점 문항은 독립된 용어 맞히기 여러 개를 한 문제에 붙이지 말 것. "
            "하나의 상황/자료에서 앞 소문항의 해석·판단이 뒤 소문항의 관계설명·적용에 실제로 필요하게 구성할 것. "
            "완성된 공식·정의를 자료에 그대로 제시한 뒤 공식명/개념명만 묻지 말 것. "
            "자료는 답 풀이에 반드시 사용되게 할 것. 관계 사슬: " + chain
        )
    return (
        "2점 문항은 서로 다른 두 개념을 억지로 연결하지 말 것. "
        "가능하면 하나의 중심개념을 정답 대상으로 두고, 다른 1점은 같은 개념의 근거·오류수정·비교·적용으로 구성할 것. "
        "자료 해석 또는 첫 판단이 두 번째 판단의 근거가 되게 하고, "
        "정답 정의를 거의 그대로 제시한 뒤 명칭만 묻지 말 것. 관계: " + chain
    )




def _two_point_single_core_profile(chosen):
    """
    2점 문항은 '서로 다른 두 개념'보다
    '하나의 중심개념 + 같은 개념의 근거/오류수정/적용'을 우선한다.
    """
    if len(chosen) != 2:
        return {"single_core": False, "score": 0.0, "reason": "not_two"}

    a, b = chosen
    ta = _topic_core(a.get("topic", ""))
    tb = _topic_core(b.get("topic", ""))
    aa = _topic_core(a.get("answer", ""))
    ab = _topic_core(b.get("answer", ""))
    ea = _norm_anchor_text(a.get("evidence", ""))
    eb = _norm_anchor_text(b.get("evidence", ""))

    signals = 0
    if ta and (ta in tb or ta in ab or ta in eb):
        signals += 1
    if tb and (tb in ta or tb in aa or tb in ea):
        signals += 1
    if aa and aa != ta and (aa in tb or aa in eb):
        signals += 1
    if ab and ab != tb and (ab in ta or ab in ea):
        signals += 1

    _, shared = _cross_reference_strength(a, b)
    shared_n = len(shared)

    simple_a = _answer_is_simple_label(a)
    simple_b = _answer_is_simple_label(b)
    mixed_role = (simple_a != simple_b)

    score = min(
        3.0,
        signals * 1.1
        + min(1.2, shared_n * 0.4)
        + (0.5 if mixed_role else 0.0),
    )
    return {
        "single_core": bool(signals >= 1 or shared_n >= 2),
        "score": round(score, 3),
        "signals": int(signals),
        "shared_terms": list(shared)[:8],
        "mixed_role": bool(mixed_role),
    }


def _two_point_unrelated_dual_target(chosen):
    """
    박테리아 + 조림 CO2 고정 한계처럼
    서로 다른 syllabus target을 한 문제에 억지로 붙인 경우만 차단한다.
    """
    if len(chosen) != 2:
        return False, {}

    prof = _two_point_single_core_profile(chosen)
    natural = float(_natural_unit_score(chosen))
    pair = float(_pair_relation_score(chosen[0], chosen[1]))
    cross, shared = _cross_reference_strength(chosen[0], chosen[1])

    a_ans = _topic_core(chosen[0].get("answer", ""))
    b_ans = _topic_core(chosen[1].get("answer", ""))

    reject = (
        not prof["single_core"]
        and natural < 2.2
        and int(cross) < 1
        and len(shared) < 2
        and a_ans != b_ans
    )

    return reject, {
        "single_core_score": prof["score"],
        "natural_unit": round(natural, 3),
        "pair_relation": round(pair, 3),
        "cross_reference": int(cross),
        "shared_terms": list(shared)[:8],
    }

def _two_point_relation_integrity(chosen):
    if len(chosen)!=2:
        return True, {}
    natural=float(_natural_unit_score(chosen))
    a,b=chosen[0],chosen[1]
    pair=float(_pair_relation_score(a,b))
    cross,shared=_cross_reference_strength(a,b)
    ok = not (
        natural < 2.0
        and pair < 5.0
        and int(cross) < 1
        and len(shared) < 2
    )
    return ok, {
        "natural_unit":round(natural,3),
        "pair_relation":round(pair,3),
        "cross_reference":int(cross),
        "shared_terms":list(shared)[:8],
    }


def _two_point_support_text(anchor, page_text):
    """
    하나의 DB anchor에서 2번째 1점 채점근거를 만든다.
    새 지식을 생성하지 않고 같은 서브노트 페이지의 실제 설명 구절만 사용한다.
    page_text는 T2에서 raw page text를 우선 전달한다.
    """
    ans=_norm_anchor_text(anchor.get("answer","")).strip()
    topic=_norm_anchor_text(anchor.get("topic","")).strip()
    ev=str(anchor.get("evidence","") or "").replace("\x01"," ").replace("\u200b"," ")

    def clean(x):
        x=str(x or "").replace("\x01"," ").replace("\u200b"," ")
        x=re.sub(r"[ \t]+"," ",x).strip()
        x=re.sub(r"^\s*[-•·○]+\s*","",x)
        return x.strip()

    def compact(x):
        return re.sub(r"\s+","",clean(x))

    ac=compact(ans)
    tc=compact(topic)

    raw=str(page_text or "")
    raw_lines=[x.rstrip() for x in raw.splitlines() if x.strip()]

    # anchor가 있는 줄을 찾는다. 공백/OCR 흔들림은 compact 비교로 흡수.
    anchor_idx=None
    for i,line in enumerate(raw_lines):
        cl=compact(line)
        if (ac and ac in cl) or (tc and tc in cl):
            anchor_idx=i
            break

    def collect_continuation(start_i, first_text):
        first=clean(first_text)
        parts=[first] if first else []
        terminal_end=re.compile(
            r"(?:함|됨|한다|된다|있음|없음|방법|과정|원리|관계|증가|감소|발생|가능|"
            r"유용함|단자|대사과정|분류|활용|사용|평가|해석|산출|제시|구별|보호|"
            r"확보|방지|적용|전송률|기체|재료|공법|기법|법칙|현상|효과|측정)$"
        )
        if first and terminal_end.search(first):
            return first

        for j in range(start_i+1,min(len(raw_lines),start_i+7)):
            rawj=raw_lines[j].strip()
            cj=clean(rawj)
            if not cj:
                continue
            if rawj.lstrip().startswith(("■","□","▶","-","※","○")):
                break
            if cj in {"정의","특징","절차","상황","규칙","장점","단점","종류","활용","원리","목적","개념"}:
                break
            if re.match(r"^\(?\d+\)?[.)]\s*",cj):
                break
            if ":" in cj and len(cj.split(":",1)[0].strip())<=30:
                break
            parts.append(cj)
            joined=" ".join(parts).strip()
            if len(joined)>=170 or terminal_end.search(cj):
                break
        return " ".join(parts).strip()


    if anchor_idx is not None:
        # 1) "개념 : 설명" 줄은 설명부 + 줄바꿈 continuation을 사용.
        line=clean(raw_lines[anchor_idx])
        if ":" in line:
            left,right=line.split(":",1)
            if (ac in compact(left) or tc in compact(left)) and right.strip():
                support=collect_continuation(anchor_idx,right)
                if 8<=len(support)<=180:
                    return support

        # 2) 같은 항목의 특징/상황/원리/절차를 정의보다 우선.
        section_end=len(raw_lines)
        for j in range(anchor_idx+1,len(raw_lines)):
            if raw_lines[j].lstrip().startswith("■"):
                section_end=j
                break

        for header in ("특징","상황","원리","절차","규칙","정의","개념"):
            hidx=None
            for j in range(anchor_idx+1,section_end):
                if clean(raw_lines[j])==header:
                    hidx=j
                    break
            if hidx is None:
                continue
            for j in range(hidx+1,min(section_end,hidx+8)):
                rawj=raw_lines[j].strip()
                if rawj.lstrip().startswith("-"):
                    first=clean(rawj)
                    support=collect_continuation(j,first)
                    if 10<=len(support)<=180:
                        return support
                cj=clean(rawj)
                if cj in {"정의","특징","절차","상황","규칙","장점","단점","종류","활용","원리","목적","개념"}:
                    break

        # 3) heading 바로 아래의 첫 설명 bullet.
        for j in range(anchor_idx+1,min(section_end,anchor_idx+8)):
            rawj=raw_lines[j].strip()
            if rawj.lstrip().startswith("-"):
                support=collect_continuation(j,clean(rawj))
                if 10<=len(support)<=180:
                    return support

    # 4) evidence 자체의 "개념 : 설명".
    cev=clean(ev)
    if ":" in cev:
        left,right=cev.split(":",1)
        if (ac in compact(left) or tc in compact(left)) and right.strip():
            right=clean(right)
            if 8<=len(right)<=180:
                return right

    # 5) evidence 정의/설명 fallback.
    m=re.search(r"\b정의\b\s*[-:]\s*(.*)$",cev)
    if m:
        v=clean(m.group(1))
        if 10<=len(v)<=180:
            return v

    fallback=cev
    fallback=re.sub(r"^■\s*","",fallback)
    fallback=re.sub(r"\s+_(?:최|서브).*?(?=\b정의\b|\b특징\b|-|$)"," ",fallback)
    fallback=re.sub(r"★\S+"," ",fallback)
    fallback=re.sub(r"\b정의\b\s*[-:]?\s*","",fallback)
    return clean(fallback)[:180]

def _make_two_point_support_anchor(anchor, page_text):
    """
    downstream 형식(정답 2개/근거 2개)을 유지하기 위한 source-grounded support rubric.
    두 번째 요소는 별도 개념 anchor가 아니라 첫 anchor의 같은-page 근거이다.
    """
    support=_two_point_support_text(anchor,page_text)
    if not support:
        return None
    b=copy.deepcopy(anchor)
    b["topic"]=f"{_norm_anchor_text(anchor.get('topic','')).strip()} · 근거/적용"
    b["answer"]=support
    b["evidence"]=support
    b["derived_support"]=True
    b["core_exam_tier"]=anchor.get("core_exam_tier","SUPPORT")
    b["core_exam_score"]=anchor.get("core_exam_score",0)
    b["core_exam_breakdown"]=copy.deepcopy(anchor.get("core_exam_breakdown",{}))
    return b



def _two_point_core_anchor_clean(anchor):
    a=copy.deepcopy(anchor)
    for key in ("topic","answer"):
        v=_norm_anchor_text(a.get(key,"")).strip()
        v=re.sub(r"^\s*\d+\)\s*","",v)
        v=re.sub(r"^\s*[□▶■]+\s*","",v)
        a[key]=v.strip()
    return a


def _two_point_core_target_ok(anchor):
    ans=_norm_anchor_text(anchor.get("answer","")).strip()
    topic=_norm_anchor_text(anchor.get("topic","")).strip()
    if not ans:
        return False
    # parser/문서 표지 및 문장 중간 조각
    if re.search(r"(기출|여기서\s*[A-Za-z]|^\(|^[가나다라마바사아자차카타파하]\)$)",ans,re.I):
        return False
    if re.fullmatch(r"표준\s*\d+",ans):
        return False
    if ans in {"영국","미국","한국","과거","현재","방지"}:
        return False
    # topic의 수식어를 떼면 너무 일반적인 1~4자 답만 남는 anchor는 독립 정답으로 쓰지 않는다.
    if len(ans.replace(" ",""))<=4 and _topic_core(topic)!=_topic_core(ans):
        return False
    if ans.count("(")!=ans.count(")"):
        return False
    # '무엇을 묻는가'가 아니라 표/목차의 열 이름에 가까운 메타 레이블
    generic={
        "원료","용도","사용목적","주요 구성","과거","현재","가로축","세로축",
        "장점","단점","종류","특징","절차","정의","분류","기준"
    }
    if ans.replace(" ","") in {x.replace(" ","") for x in generic}:
        return False
    # 지나치게 긴 목차형 topic은 중심 정답으로 쓰지 않는다.
    if _heading_like(ans) or _heading_like(topic):
        return False
    return True


def _two_point_support_ok(text):
    t=_norm_anchor_text(text).strip()
    if len(t)<10 or len(t)>180:
        return False
    if re.search(r"[□▶■◎♥]|_최|_서브|★|\b기출\b",t):
        return False
    if re.fullmatch(r"[\W_]*[A-Za-z]+[\W_]*",t):
        return False
    if t.lower() in {"procedures","procedure","definition","features"}:
        return False
    if re.fullmatch(r"\(?\s*(?:procedures?|definition|features?)\s*\)?",t,re.I):
        return False
    # 끝이 명백한 조사/접속어로 잘린 문장은 채점근거로 쓰지 않는다.
    if re.search(r"(?:\s|^)(?:의|을|를|이|가|은|는|와|과|및|또는|으로|로|에|에서)$",t):
        return False
    # 설명형 근거: 동작/관계/조건 표현이 있거나 충분히 구체적인 수치·기호 설명이어야 한다.
    explanatory=bool(re.search(
        r"(함|됨|한다|된다|있|없|경우|따라|때문|이용|적용|발생|변화|증가|감소|"
        r"측정|전송|분해|전환|생성|산출|선정|발견|비교|평형|비례|작용|사용|흐르|"
        r"먹고|뱉|저항|전류|전압|속도|하중|응력|변형|아이디어|제시|구별|기초|목표|중심|활용|구성|유지|분류|평가|해석)",
        t
    ))
    if not explanatory:
        return False
    return True

def _select_two_point_one_anchor(anchors, page_map, raw_page_map, pd, rng, pattern_id):
    """
    T2 전용 selector.
    DB 정답 anchor는 1개만 선택하고, 두 번째 1점은 같은 anchor/page의 근거·오류수정·비교·적용으로 만든다.
    """
    ranked=[]
    for a0 in anchors:
        a=_two_point_core_anchor_clean(a0)
        if not _two_point_core_target_ok(a):
            continue
        peripheral=int((a.get("core_exam_breakdown") or {}).get("peripherality",0) or 0)
        if peripheral>=4:
            pd["support_only_reject"]+=1
            continue

        _page_key=(str(a.get("source_name","")),int(a.get("page_no",0) or 0))
        page_text=raw_page_map.get(_page_key) or page_map.get(_page_key,"")
        support=_make_two_point_support_anchor(a,page_text)
        if not support or not _two_point_support_ok(support.get("answer","")):
            continue

        # support answer가 core answer와 같으면 1+1 구조가 성립하지 않는다.
        if _topic_core(support.get("answer",""))==_topic_core(a.get("answer","")):
            continue

        try:
            conf=float(a.get("confidence") or 0)
        except Exception:
            conf=0.0
        importance=float(a.get("importance_score") or 0)
        exam_value=float(a.get("exam_value_score") or 0)
        core=float(a.get("core_exam_score") or 0)
        tier=str(a.get("core_exam_tier") or "SUPPORT")

        score=(
            min(1.0,max(0.0,conf))
            + importance*1.05
            + exam_value*1.50
            + core*0.18
            - (3.5 if tier=="SUPPORT" else 0.0)
        )
        # 설명형 evidence가 충분할수록 2번째 채점요소를 안정적으로 만들 수 있다.
        ev=_norm_anchor_text(support.get("answer",""))
        if len(ev)>=35:
            score+=1.0
        if re.search(r"(때문|따라|과정|원인|결과|특징|달리|변화|작용|이용|판단|분해|전환|선정|발견)",ev):
            score+=1.0

        ranked.append((score,[a,support]))

    if not ranked:
        pd["final_reason"]="no_single_anchor_candidates"
        return [],{"score_pipeline_diagnostic":pd}

    ranked.sort(key=lambda x:x[0],reverse=True)

    uniq=[]
    seen=set()
    for score,bundle in ranked:
        k=_topic_core(bundle[0].get("answer",""))
        if k in seen:
            continue
        seen.add(k)
        uniq.append((score,bundle))
        if len(uniq)>=8:
            break

    pd["candidate_accept"]=len(uniq)
    pd["two_point_one_anchor_candidates"]=len(uniq)
    pd["final_reason"]="single_anchor_candidates_ready"

    def cand_diag(score0,bundle0,rank0):
        a=bundle0[0]
        return {
            "rank":rank0,
            "final_selector_score":round(float(score0),3),
            "avg_importance_score":round(float(a.get("importance_score") or 0),3),
            "importance_contribution":round(float(a.get("importance_score") or 0)*1.05,3),
            "avg_exam_value_score":round(float(a.get("exam_value_score") or 0),3),
            "exam_value_contribution":round(float(a.get("exam_value_score") or 0)*1.50,3),
            "avg_core_exam_score":round(float(a.get("core_exam_score") or 0),3),
            "core_exam_contribution":round(float(a.get("core_exam_score") or 0)*0.18,3),
            "support_count":1 if str(a.get("core_exam_tier") or "")=="SUPPORT" else 0,
            "support_penalty":-3.5 if str(a.get("core_exam_tier") or "")=="SUPPORT" else 0.0,
            "natural_unit_score":None,
            "tiers":[str(a.get("core_exam_tier") or "SUPPORT"),"SAME_ANCHOR_SUPPORT"],
            "topics":[str(a.get("topic","")),str(bundle0[1].get("topic",""))],
            "anchors":[
                {
                    "topic":str(a.get("topic","")),
                    "answer":str(a.get("answer","")),
                    "source_name":str(a.get("source_name","")),
                    "page_no":a.get("page_no"),
                    "core_exam_score":a.get("core_exam_score",0),
                    "core_exam_tier":a.get("core_exam_tier","SUPPORT"),
                    "core_exam_qualified":a.get("core_exam_qualified",False),
                    "breakdown":copy.deepcopy(a.get("core_exam_breakdown",{})),
                    "importance_score":a.get("importance_score",0),
                    "exam_value_score":a.get("exam_value_score",0),
                },
                {
                    "topic":str(bundle0[1].get("topic","")),
                    "answer":str(bundle0[1].get("answer","")),
                    "source_name":str(bundle0[1].get("source_name","")),
                    "page_no":bundle0[1].get("page_no"),
                    "derived_support":True,
                },
            ],
        }

    leaderboard=[cand_diag(sc,b,idx+1) for idx,(sc,b) in enumerate(uniq)]
    top=uniq[:min(2,len(uniq))]
    chosen_idx=rng.randrange(len(top))
    score,bundle=top[chosen_idx]
    selected_rank=chosen_idx+1
    core_anchor=bundle[0]

    relation_meta={
        "master_concept":_norm_anchor_text(core_anchor.get("topic","")),
        "relation":"하나의 중심개념을 판단하고 같은 원문 근거로 이유·오류수정·비교·적용을 수행",
        "thinking_types":_pattern_thinking_types(pattern_id),
        "exam_skeleton":_skeleton_for_pattern(pattern_id),
        "scoring_plan":_scoring_plan({"id":pattern_id,"subpoints":[1,1]},bundle),
        "material_limits":_compact_material_limits(2),
        "natural_unit_score":None,
        "two_point_label_policy":"ONE_ANCHOR: 첫 1점은 중심개념 판단, 둘째 1점은 같은 개념의 source-grounded 근거/오류수정/비교/적용. 둘째를 별도 개념명으로 묻지 말 것.",
        "core_exam_profile":[{
            "topic":core_anchor.get("topic",""),
            "score":core_anchor.get("core_exam_score",0),
            "tier":core_anchor.get("core_exam_tier","SUPPORT"),
            "breakdown":core_anchor.get("core_exam_breakdown",{}),
        }],
        "quality_directive":(
            "2점은 ONE-ANCHOR 구조다. 서로 다른 두 개념을 결합하지 말 것. "
            "첫 요구는 중심개념을 자료에서 판단하게 하고, 두 번째 요구는 반드시 같은 중심개념에 대한 "
            "근거·오류수정·비교·적용 중 하나를 요구한다. 두 번째 고정답은 별도 개념명이 아니라 원문 채점근거다."
        ),
        "selector_reason":f"Python one-anchor exam-value score={score:.2f}",
        "relation_score":round(score,2),
        "selection_mode":"python_exam_value_one_anchor_t2",
        "source_policy":"subnote_only_for_answer_content",
        "score_pipeline_diagnostic":copy.deepcopy(pd),
        "score_diagnostic":{
            "selected_rank":selected_rank,
            "selected":cand_diag(score,bundle,selected_rank),
            "leaderboard":leaderboard,
            "core_exam_weights":{
                "past_exam":4,"subnote_importance":3,"representative":3,
                "repeatability":4,"centrality":3,"peripherality":-2,
            },
            "tier_thresholds":{
                "CORE":"qualified and raw >= 58",
                "NORMAL":"qualified and raw >= 40",
                "SUPPORT":"otherwise",
                "qualification":"past_exam >= 3 OR representative >= 4 OR repeatability >= 4",
            },
            "note":"T2는 DB answer anchor 1개 + 동일 페이지의 source-grounded 채점근거 1개로 구성.",
        },
    }
    return bundle,relation_meta

def _smart_relation_bundle(db_path, domain, need, used_answers, excluded_topics, rng, pattern_id=""):
    con=sqlite3.connect(db_path)
    con.row_factory=sqlite3.Row

    cols={r[1] for r in con.execute("PRAGMA table_info(anchors)").fetchall()}
    required={"domain","topic","answer","evidence","source_name","page_no"}
    missing=sorted(required-cols)
    if missing:
        con.close()
        raise RuntimeError("knowledge.db anchors 스키마 누락: "+", ".join(missing))

    has_conf="confidence" in cols
    conf_expr="COALESCE(confidence,0)" if has_conf else "0"
    rows=con.execute(
        f"""SELECT domain,topic,answer,evidence,source_name,page_no,
                   {conf_expr} AS confidence
            FROM anchors
            WHERE domain=?
            ORDER BY {conf_expr} DESC, source_name, page_no""",
        (domain,)
    ).fetchall()
    source_kinds=_source_kind_map(con)
    _source_names={str(r["source_name"]) for r in rows}
    page_map=_page_text_map(con,_source_names)
    raw_page_map=_page_raw_text_map(con,_source_names)
    con.close()

    _pd={
        "domain":domain,
        "pattern_id":pattern_id,
        "required_count":need,
        "raw_rows":len(rows),
        "primary_source_pass":0,
        "anchor_ok_pass":0,
        "used_answer_reject":0,
        "excluded_topic_reject":0,
        "dedupe_reject":0,
        "usable_anchors":0,
        "pair_score_pass":0,
        "cmp_relation_reject":0,
        "edge_neighbor_pass":0,
        "near_duplicate_reject":0,
        "connected_reject":0,
        "natural_unit_reject":0,
        "direct_chain_reject":0,
        "same_source_reject":0,
        "page_distance_reject":0,
        "support_only_reject":0,
        "support_only_fallback":0,
        "two_point_label_reject":0,
        "two_point_relation_reject":0,
        "two_point_dual_target_reject":0,
        "two_point_one_anchor_candidates":0,
        "anchor_fragment_reject":0,
        "anchor_contradiction_reject":0,
        "anchor_normalization_policy":"strip_bullets_then_validate",
        "bundle_fragment_reject":0,
        "candidate_accept":0,
        "reject_examples":{
            "support_only":[],
            "two_point_label":[],
            "two_point_relation":[],
            "two_point_dual_target":[],
            "anchor_fragment":[],
            "anchor_contradiction":[],
            "bundle_fragment":[],
            "same_source":[],
            "page_distance":[],
            "natural_unit":[],
        },
    }

    used_answers={_topic_core(x) for x in (used_answers or set())}
    excluded_topics={_topic_core(x) for x in (excluded_topics or set())}

    anchors=[]
    seen=set()
    for r in rows:
        a=dict(r)
        a["topic"]=_strip_anchor_noise(_norm_anchor_text(a.get("topic")))
        a["answer"]=_strip_anchor_noise(_norm_anchor_text(a.get("answer")))
        a["evidence"]=_norm_anchor_text(a.get("evidence"))
        a["source_kind"]=source_kinds.get(str(a.get("source_name","")),"")
        if not _primary_source(a.get("source_name",""),a.get("source_kind","")):
            continue
        _pd["primary_source_pass"]+=1
        if not _anchor_ok(a):
            _cr=_anchor_internal_contradiction_reason(a)
            if _cr:
                _pd["anchor_contradiction_reject"]+=1
                if len(_pd["reject_examples"]["anchor_contradiction"])<5:
                    _pd["reject_examples"]["anchor_contradiction"].append({
                        "topic":str(a.get("topic","")),
                        "answer":str(a.get("answer","")),
                        "reason":_cr,
                    })
            _fr=_anchor_fragment_reason(a)
            if _fr:
                _pd["anchor_fragment_reject"]+=1
                if len(_pd["reject_examples"]["anchor_fragment"])<5:
                    _pd["reject_examples"]["anchor_fragment"].append({
                        "topic":str(a.get("topic","")),
                        "answer":str(a.get("answer","")),
                        "reason":_fr,
                    })
            continue
        _pd["anchor_ok_pass"]+=1
        _page_text=page_map.get(
            (str(a.get("source_name","")),int(a.get("page_no",0) or 0)),""
        )
        a["importance_score"]=_subnote_importance_score(a,_page_text)
        a["exam_value_score"]=_exam_value_score(a,_page_text)
        _core=_core_exam_score(a,_page_text)
        a["core_exam_score"]=_core["score"]
        a["core_exam_tier"]=_core["tier"]
        a["core_exam_qualified"]=_core["qualified"]
        a["core_exam_breakdown"]=_core
        if _topic_core(a["answer"]) in used_answers:
            _pd["used_answer_reject"]+=1
            continue
        if _topic_core(a["topic"]) in excluded_topics:
            _pd["excluded_topic_reject"]+=1
            continue
        key=(_topic_core(a["topic"]),_topic_core(a["answer"]),a.get("source_name"),a.get("page_no"))
        if key in seen:
            _pd["dedupe_reject"]+=1
            continue
        seen.add(key)
        anchors.append(a)

    _pd["usable_anchors"]=len(anchors)

    # R19: 모든 2점 패턴(T2_DATA/T2_REL/T2_ERR/T2_CMP)은 DB answer anchor 1개만 고른다.
    # downstream 형식은 [1,1]을 유지하기 위해 같은 페이지의 source-grounded support rubric을 두 번째 요소로 만든다.
    if not str(pattern_id).upper().startswith("T4"):
        if len(anchors)<1:
            _pd["final_reason"]="usable_anchors_below_one"
            return [],{"score_pipeline_diagnostic":_pd}
        return _select_two_point_one_anchor(anchors,page_map,raw_page_map,_pd,rng,pattern_id)

    if len(anchors)<need:
        _pd["final_reason"]="usable_anchors_below_need"
        return [],{"score_pipeline_diagnostic":_pd}

    candidates=[]
    for i,a in enumerate(anchors):
        edges=[]
        for j,c in enumerate(anchors):
            if i==j:
                continue
            s=_pair_relation_score(a,c)
            if s>=4.0:
                _pd["pair_score_pass"]+=1
                if str(pattern_id).upper()=="T2_CMP":
                    cross,shared=_cross_reference_strength(a,c)
                    if _near_duplicate_anchor(a,c) or (cross<1 and len(shared)<2):
                        _pd["cmp_relation_reject"]+=1
                        continue
                edges.append((s,c))
        edges.sort(key=lambda x:x[0],reverse=True)
        if len(edges)<need-1:
            continue
        _pd["edge_neighbor_pass"]+=1

        pool=[x[1] for x in edges[:min(6,len(edges))]]
        trial_sets=[[a]+pool[:need-1]]
        for shift in range(1,min(3,len(pool))):
            tail=pool[shift:shift+need-1]
            if len(tail)==need-1:
                trial_sets.append([a]+tail)

        for chosen in trial_sets:
            if any(
                _near_duplicate_anchor(chosen[x],chosen[y])
                for x in range(len(chosen))
                for y in range(x+1,len(chosen))
            ):
                _pd["near_duplicate_reject"]+=1
                continue

            integrity_ok,integrity_reasons=_bundle_anchor_integrity(chosen)
            if not integrity_ok:
                _pd["bundle_fragment_reject"]+=1
                if len(_pd["reject_examples"]["bundle_fragment"])<5:
                    _pd["reject_examples"]["bundle_fragment"].append({
                        "topics":[str(x.get("topic","")) for x in chosen],
                        "answers":[str(x.get("answer","")) for x in chosen],
                        "reasons":integrity_reasons,
                    })
                continue

            connected,graph_score=_bundle_connected(chosen,min_edge=4.0)
            if not connected:
                _pd["connected_reject"]+=1
                continue

            # R12:
            # 2점은 억지 관계사슬을 만들지 않고 하나의 핵심 개념 + 근거 판단을 허용한다.
            # 4점만 '원래 존재하는 자연적 문제해결 단위'와 직접 사슬을 요구한다.
            natural_score=_natural_unit_score(chosen)

            if str(pattern_id).upper().startswith("T4"):
                if not _independent_scoring_targets(chosen):
                    _pd["bundle_fragment_reject"]+=1
                    if len(_pd["reject_examples"]["bundle_fragment"])<5:
                        _pd["reject_examples"]["bundle_fragment"].append({
                            "topics":[str(x.get("topic","")) for x in chosen],
                            "answers":[str(x.get("answer","")) for x in chosen],
                            "reasons":[{"reason":"not_independent_scoring_targets"}],
                        })
                    continue
                if natural_score < 3.0:
                    _pd["natural_unit_reject"]+=1
                    if len(_pd["reject_examples"]["natural_unit"])<5:
                        _pd["reject_examples"]["natural_unit"].append({
                            "topics":[str(x.get("topic","")) for x in chosen],
                            "score":round(float(natural_score),2),
                        })
                    continue
                ordered,chain_score=_direct_chain_order(chosen)
                if not ordered:
                    _pd["direct_chain_reject"]+=1
                    continue
                chosen=ordered
                graph_score += min(5.0,chain_score*0.22) + min(4.0,natural_score)
            else:
                # 2점은 4점과 같은 natural-unit 임계값을 쓰지 않는다.
                # 여러 관계 신호가 모두 약한 조합만 제거하여 억지 연결만 차단한다.
                if natural_score < 0.0:
                    _pd["natural_unit_reject"]+=1
                    if len(_pd["reject_examples"]["natural_unit"])<5:
                        _pd["reject_examples"]["natural_unit"].append({
                            "topics":[str(x.get("topic","")) for x in chosen],
                            "score":round(float(natural_score),2),
                        })
                    continue
                _dual_reject,_dual_diag=_two_point_unrelated_dual_target(chosen)
                if _dual_reject:
                    _pd["two_point_dual_target_reject"]+=1
                    if len(_pd["reject_examples"]["two_point_dual_target"])<5:
                        _pd["reject_examples"]["two_point_dual_target"].append({
                            "topics":[str(x.get("topic","")) for x in chosen],
                            "answers":[str(x.get("answer","")) for x in chosen],
                            **_dual_diag,
                        })
                    continue

                _rel_ok,_rel_diag=_two_point_relation_integrity(chosen)
                if not _rel_ok:
                    _pd["two_point_relation_reject"]+=1
                    if len(_pd["reject_examples"]["two_point_relation"])<5:
                        _pd["reject_examples"]["two_point_relation"].append({
                            "topics":[str(x.get("topic","")) for x in chosen],
                            "answers":[str(x.get("answer","")) for x in chosen],
                            **_rel_diag,
                        })
                    continue
                graph_score += min(2.0,max(0.0,natural_score*0.4))

            if len({str(x.get("source_name","")) for x in chosen})!=1:
                _pd["same_source_reject"]+=1
                if len(_pd["reject_examples"]["same_source"])<5:
                    _pd["reject_examples"]["same_source"].append({
                        "topics":[str(x.get("topic","")) for x in chosen],
                        "sources":[str(x.get("source_name","")) for x in chosen],
                    })
                continue
            try:
                pages=[int(x.get("page_no",0)) for x in chosen]
            except Exception:
                continue
            if max(pages)-min(pages)>2:
                _pd["page_distance_reject"]+=1
                if len(_pd["reject_examples"]["page_distance"])<5:
                    _pd["reject_examples"]["page_distance"].append({
                        "topics":[str(x.get("topic","")) for x in chosen],
                        "pages":pages,
                    })
                continue

            try:
                conf=sum(float(x.get("confidence") or 0) for x in chosen)/len(chosen)
            except Exception:
                conf=0.0
            importance=sum(float(x.get("importance_score") or 0) for x in chosen)/len(chosen)
            exam_value=sum(float(x.get("exam_value_score") or 0) for x in chosen)/len(chosen)
            core_exam=sum(float(x.get("core_exam_score") or 0) for x in chosen)/len(chosen)
            core_tiers=[str(x.get("core_exam_tier") or "SUPPORT") for x in chosen]
            natural=_natural_unit_score(chosen)

            support_count=sum(1 for t in core_tiers if t=="SUPPORT")

            # R15: SUPPORT라는 등급 하나만으로 하드 탈락시키지 않는다.
            # 대신 모든 anchor가 실제 지엽성 4점 이상일 때만 직접 차단한다.
            peripheral_scores=[
                int((x.get("core_exam_breakdown") or {}).get("peripherality",0) or 0)
                for x in chosen
            ]
            if peripheral_scores and all(v>=4 for v in peripheral_scores):
                _pd["support_only_reject"]+=1
                if len(_pd["reject_examples"]["support_only"])<5:
                    _pd["reject_examples"]["support_only"].append({
                        "topics":[str(x.get("topic","")) for x in chosen],
                        "answers":[str(x.get("answer","")) for x in chosen],
                        "tiers":core_tiers,
                        "core_scores":[float(x.get("core_exam_score") or 0) for x in chosen],
                        "peripherality":peripheral_scores,
                        "r15_action":"rejected_as_truly_peripheral",
                    })
                continue

            support_only=(support_count==len(chosen))
            if support_only:
                _pd["support_only_fallback"]+=1
                if len(_pd["reject_examples"]["support_only"])<5:
                    _pd["reject_examples"]["support_only"].append({
                        "topics":[str(x.get("topic","")) for x in chosen],
                        "answers":[str(x.get("answer","")) for x in chosen],
                        "tiers":core_tiers,
                        "core_scores":[float(x.get("core_exam_score") or 0) for x in chosen],
                        "breakdowns":[copy.deepcopy(x.get("core_exam_breakdown",{})) for x in chosen],
                        "r15_action":"kept_with_strong_penalty",
                    })

            score_total=(
                graph_score
                + min(1.0,max(0.0,conf))
                + importance*1.05
                + exam_value*1.50
                + core_exam*0.18
                - support_count*1.75
                - (4.0 if support_only else 0.0)
            )
            if str(pattern_id).upper().startswith("T4"):
                # R17: 임계값은 유지하고, 자연스러운 개념 묶음이 순위에서 조금 더 우선되게 한다.
                score_total += natural*1.65
            else:
                score_total += max(-1.5,min(2.0,natural*0.55))
                _single_prof=_two_point_single_core_profile(chosen)
                score_total += min(2.5,float(_single_prof.get("score",0.0))*0.85)
                label_adjust,label_reject=_two_point_bundle_penalty(chosen)
                if label_reject:
                    _pd["two_point_label_reject"]+=1
                    if len(_pd["reject_examples"]["two_point_label"])<5:
                        _pd["reject_examples"]["two_point_label"].append({
                            "topics":[str(x.get("topic","")) for x in chosen],
                            "answers":[str(x.get("answer","")) for x in chosen],
                            "tiers":[str(x.get("core_exam_tier") or "SUPPORT") for x in chosen],
                            "core_scores":[float(x.get("core_exam_score") or 0) for x in chosen],
                        })
                    # 두 채점요소가 모두 단순 명칭회상형이면 API 호출 전에 제거한다.
                    continue
                score_total += label_adjust
            candidates.append((score_total,chosen))
            _pd["candidate_accept"]+=1

    if not candidates:
        _pd["final_reason"]="no_candidates_after_all_filters"
        return [],{"score_pipeline_diagnostic":_pd}

    uniq=[]; seen_sets=set()
    for score,chosen in sorted(candidates,key=lambda x:x[0],reverse=True):
        k=tuple(sorted(_topic_core(x.get("answer","")) for x in chosen))
        if k in seen_sets:
            continue
        seen_sets.add(k)
        uniq.append((score,chosen))
        if len(uniq)>=8:
            break

    # DIAGNOSTIC ONLY:
    # 선택 로직은 R14와 완전히 동일하다. 아래 정보는 화면 표시용이며
    # 후보 순위나 랜덤 선택에는 영향을 주지 않는다.
    def _candidate_diag(score0, chosen0, rank0):
        importance0=sum(float(x.get("importance_score") or 0) for x in chosen0)/len(chosen0)
        exam_value0=sum(float(x.get("exam_value_score") or 0) for x in chosen0)/len(chosen0)
        core_exam0=sum(float(x.get("core_exam_score") or 0) for x in chosen0)/len(chosen0)
        tiers0=[str(x.get("core_exam_tier") or "SUPPORT") for x in chosen0]
        support0=sum(1 for t in tiers0 if t=="SUPPORT")
        natural0=_natural_unit_score(chosen0)
        try:
            conf0=sum(float(x.get("confidence") or 0) for x in chosen0)/len(chosen0)
        except Exception:
            conf0=0.0

        # graph_score는 candidate 생성 시 이미 계산되어 score0에 포함되므로
        # 여기서는 역산하지 않고, 우리가 조절한 핵심 가중치 항목을 그대로 표시한다.
        return {
            "rank":rank0,
            "final_selector_score":round(float(score0),3),
            "avg_importance_score":round(importance0,3),
            "importance_contribution":round(importance0*1.05,3),
            "avg_exam_value_score":round(exam_value0,3),
            "exam_value_contribution":round(exam_value0*1.50,3),
            "avg_core_exam_score":round(core_exam0,3),
            "core_exam_contribution":round(core_exam0*0.18,3),
            "support_count":support0,
            "support_penalty":round(-support0*1.75,3),
            "natural_unit_score":round(float(natural0),3),
            "tiers":tiers0,
            "topics":[str(x.get("topic","")) for x in chosen0],
            "anchors":[
                {
                    "topic":str(x.get("topic","")),
                    "answer":str(x.get("answer","")),
                    "source_name":str(x.get("source_name","")),
                    "page_no":x.get("page_no"),
                    "core_exam_score":x.get("core_exam_score",0),
                    "core_exam_tier":x.get("core_exam_tier","SUPPORT"),
                    "core_exam_qualified":x.get("core_exam_qualified",False),
                    "breakdown":copy.deepcopy(x.get("core_exam_breakdown",{})),
                    "importance_score":x.get("importance_score",0),
                    "exam_value_score":x.get("exam_value_score",0),
                }
                for x in chosen0
            ],
        }

    leaderboard=[
        _candidate_diag(sc,ch,idx+1)
        for idx,(sc,ch) in enumerate(uniq[:8])
    ]

    top=uniq[:min(2,len(uniq))]
    chosen_idx=rng.randrange(len(top))
    score,chosen=top[chosen_idx]
    selected_rank=chosen_idx+1
    topic_chain=" → ".join(_norm_anchor_text(x.get("topic","")) for x in chosen)
    relation_meta={
        "master_concept":_norm_anchor_text(chosen[0].get("topic","")) if chosen else "",
        "relation":"직접 교차참조를 따라 이어지는 핵심 개념 사슬: "+topic_chain,
        "thinking_types":_pattern_thinking_types(pattern_id),
        "exam_skeleton":_skeleton_for_pattern(pattern_id),
        "scoring_plan":_scoring_plan(
            {"id":pattern_id,"subpoints":[1]*len(chosen)}, chosen
        ),
        "material_limits":_compact_material_limits(
            4 if str(pattern_id).upper().startswith("T4") else 2
        ),
        "natural_unit_score":round(_natural_unit_score(chosen),2),
        "two_point_label_policy":(
            "2점은 하나의 중심개념을 우선 선택하고 1점+1점은 개념 판단 + 같은 개념의 근거/오류수정/비교/적용으로 구성; 서로 다른 두 개념 억지 연결 금지"
            if not str(pattern_id).upper().startswith("T4") else ""
        ),
        "core_exam_profile":[
            {"topic":x.get("topic",""),"score":x.get("core_exam_score",0),
             "tier":x.get("core_exam_tier","SUPPORT"),
             "breakdown":x.get("core_exam_breakdown",{})}
            for x in chosen
        ],
        "quality_directive":_relation_directive(pattern_id,chosen),
        "selector_reason":f"Python exam-value relation score={score:.2f}",
        "relation_score":round(score,2),
        "selection_mode":"python_exam_value_direct_chain",
        "source_policy":"subnote_only_for_answer_content",
        "score_pipeline_diagnostic":copy.deepcopy(_pd),
        "score_diagnostic":{
            "selected_rank":selected_rank,
            "selected":_candidate_diag(score,chosen,selected_rank),
            "leaderboard":leaderboard,
            "core_exam_weights":{
                "past_exam":4,
                "subnote_importance":3,
                "representative":3,
                "repeatability":4,
                "centrality":3,
                "peripherality":-2,
            },
            "tier_thresholds":{
                "CORE":"qualified and raw >= 58",
                "NORMAL":"qualified and raw >= 40",
                "SUPPORT":"otherwise",
                "qualification":"past_exam >= 3 OR representative >= 4 OR repeatability >= 4",
            },
            "note":"진단 표시 전용. R14 후보 순위/가중치/선택 로직은 변경하지 않음.",
        },
    }
    return chosen,relation_meta

def _compact_candidate_cluster(rows, need, limit=6):
    rows=[dict(r) for r in (rows or []) if _anchor_ok(r)]
    if len(rows)<need:
        return []
    ranked=[]
    for i,a in enumerate(rows):
        neighbors=[]
        for j,c in enumerate(rows):
            if i==j:
                continue
            s=_pair_relation_score(a,c)
            if s>=4.0:
                neighbors.append((s,c))
        neighbors.sort(key=lambda x:x[0],reverse=True)
        if len(neighbors)>=need-1:
            chosen=[a]+[x[1] for x in neighbors[:need-1]]
            connected,score=_bundle_connected(chosen,4.0)
            if connected:
                ranked.append((score,chosen))
    if not ranked:
        return []
    ranked.sort(key=lambda x:x[0],reverse=True)
    return ranked[0][1][:limit]

def _candidate_shape_errors(cand, pat, pts):
    if not isinstance(cand,dict):
        return ["문항 객체가 dict가 아님"]
    errs=[]
    expected=list(pat.get("subpoints",[]))
    if int(cand.get("points",pts) or 0)!=int(pts):
        errs.append("배점 불일치")
    sub=list(cand.get("subpoints",[]))
    if sub and sub!=expected:
        errs.append(f"부분점수 불일치: {sub} != {expected}")
    tasks=list(cand.get("tasks",[]) or [])
    answers=list(cand.get("answer",[]) or [])
    solutions=list(cand.get("solution",[]) or [])
    evidence=list(cand.get("evidence",[]) or [])
    if len(tasks)!=len(expected):
        errs.append(f"작성방법 수 불일치({len(tasks)}/{len(expected)})")
    if len(answers)!=len(expected):
        errs.append(f"정답 수 불일치({len(answers)}/{len(expected)})")
    if solutions and len(solutions)!=len(expected):
        errs.append(f"해설 수 불일치({len(solutions)}/{len(expected)})")
    if evidence and len(evidence)!=len(expected):
        errs.append(f"근거 수 불일치({len(evidence)}/{len(expected)})")
    cores=[_topic_core(x) for x in answers if _topic_core(x)]
    if len(cores)!=len(set(cores)):
        errs.append("정답 중복")
    if any(_heading_like(ans) for ans in answers):
        errs.append("목차/분류형 답안 포함")
    pid=str(cand.get("pattern_id","") or "")
    if pid and pid!=str(pat.get("id","")):
        errs.append(f"패턴 ID 불일치({pid}/{pat.get('id')})")
    if sum(expected)!=pts:
        errs.append("패턴 부분점수 합계 오류")
    return errs

def _validate_sample_exam(qs):
    errs=[]
    if len(qs)!=6:
        errs.append(f"문항수 {len(qs)} != 6")
    got=[int(q.get("points",0) or 0) for q in qs]
    if got!=[2,2,4,4,4,4]:
        errs.append(f"배점 구조 {got} != [2,2,4,4,4,4]")
    if sum(got)!=20:
        errs.append(f"총점 {sum(got)} != 20")
    for i,q in enumerate(qs,1):
        subs=list(q.get("subpoints",[]) or [])
        if sum(subs)!=int(q.get("points",0) or 0):
            errs.append(f"{i}번 부분점수 합 불일치")
        if len(q.get("tasks",[]) or [])!=len(subs):
            errs.append(f"{i}번 작성방법/부분점수 수 불일치")
        if len(q.get("answer",[]) or [])!=len(subs):
            errs.append(f"{i}번 정답/부분점수 수 불일치")
    return errs

def _concept_patterns(rng,pts,first=None):
    from patterns import PATTERNS
    valid=[p for p in PATTERNS if p["points"]==pts and not p["calc"] and p.get("weight",1)>0]
    if pts==4:
        priority={"T4_DATA112":0,"T4_ERR22":1,"T4_112":2}
        valid=sorted(valid,key=lambda p:(priority.get(p["id"],9),-p.get("weight",1)))
    else:
        valid=sorted(valid,key=lambda p:-p.get("weight",1))
    out=[]
    if first is not None: out.append(first)
    for p in valid:
        if not any(x["id"]==p["id"] for x in out): out.append(p)
    return out

def score_pattern(section,count,points):
    defaults={
        "A":[2,2,2,2]+[4]*8,
        "B":[2,2]+[4]*9,
        "SAMPLE":[2,2,4,4,4,4],
    }
    p=defaults.get(section,[])
    if len(p)==count and sum(p)==points:
        return p[:]
    if section=="SAMPLE":
        raise ValueError("SAMPLE은 6문항 20점(2,2,4,4,4,4)만 지원합니다.")
    raise ValueError("실제 A/B 기본 배점 구조만 지원합니다.")

def _prepend_formula_element(q, task, answer, solution):
    q["tasks"]=[task]+list(q.get("tasks",[]))
    q["answer"]=[answer]+list(q.get("answer",[]))
    q["solution"]=[solution]+list(q.get("solution",[]))
    return q

def _enrich_formula(q,pts):
    """계산형 배점을 실제 채점요소로 환산한다. 4점은 서로 다른 3개 요구(1+1+2)만 허용한다."""
    topic=q.get("topic","")
    q["material_form"]="수치자료"
    q["question_type"]="계산/판단" if pts==4 else "간단계산"
    q["concept_family"]=next(iter(families_for(q)), "formula_"+re.sub(r"\W+","_",topic))

    if pts==2:
        if len(q.get("answer",[]))==1:
            relation=None
            if "수직응력" in topic: relation=("적용되는 수직응력의 관계식을 쓸 것.","σ=P/A","σ=P/A")
            elif "정수압" in topic: relation=("정수압의 관계식을 쓸 것.","p=ρgh","p=ρgh")
            elif "열효율" in topic: relation=("열기관 열효율의 관계식을 쓸 것.","η=(QH-QL)/QH×100","η=(QH-QL)/QH×100")
            if relation: q=_prepend_formula_element(q,*relation)
        if len(q.get("answer",[]))<2:return None
        q["answer"]=q["answer"][:2];q["tasks"]=q["tasks"][:2];q["solution"]=q["solution"][:2]
        q["subpoints"]=[1,1]
    else:
        if len(q.get("answer",[]))<3 or len(q.get("tasks",[]))<3:return None
        q["answer"]=q["answer"][:3];q["tasks"]=q["tasks"][:3];q["solution"]=q["solution"][:3]
        canon=[re.sub(r"[^0-9A-Za-z가-힣]+","",str(x)).lower() for x in q["answer"][:3]]
        if len(set(canon))<3:return None
        q["subpoints"]=[1,1,2]
    q["points"]=pts
    q["pattern_id"]="T4_C112" if pts==4 else "T2_C11"
    q["fingerprint"]=fingerprint(q)
    return q

def make_section(db_path,section,count,points,domains=None,api_key="",model="gpt-5.6-luna",
                 ai_enabled=True,ai_quality_enabled=True,judge_model=None,seed=None,
                 previous_questions=None,shared_answers=None,tuning_mode=False):
    rng=random.Random(seed)
    domains=list(domains or DOMAINS)
    scores=score_pattern(section,count,points)
    plan=blueprint(section,scores,domains,rng)
    style=official_style_profile(db_path)
    judge_model=judge_model or model
    writer_active=bool(ai_enabled and api_key)
    quality_active=bool(writer_active and ai_quality_enabled and not tuning_mode)
    selector_active=bool(writer_active)

    prior=list(previous_questions or [])
    qs=[]
    used_topics=set()
    used_answers=set(shared_answers or [])
    used_patterns=[]
    ai_calls=0; fallbacks=0; formula_used=0
    judge_calls=0; judge_rejects=0; selector_calls=0
    formula_cap=2

    diagnostics=[]
    diagnostic_limit=100

    def diag(slot,stage,reason="",**extra):
        if len(diagnostics)>=diagnostic_limit:
            return
        row={
            "section":section,
            "number":slot.get("number"),
            "domain":slot.get("domain"),
            "points":slot.get("points"),
            "stage":stage,
            "reason":str(reason or ""),
        }
        for k,v in extra.items():
            if v not in (None,"",[],{}):
                row[k]=v
        diagnostics.append(row)

    for slot in plan:
        pts=slot["points"]; dom=slot["domain"]; q=None
        wants_calc=slot["question_type"] in {"간단계산","계산/판단"}

        if wants_calc and dom not in FORMULA_DOMAINS:
            slot["question_type"]="자료해석" if pts==4 else "자료식별"
            wants_calc=False

        if wants_calc and dom in FORMULA_DOMAINS and formula_used<formula_cap:
            _formula_enrich_failures=0
            for _ in range(16 if quality_active else 40):
                cand=generate_formula_question(dom,rng)
                if not cand:
                    diag(slot,"formula_generator","계산형 템플릿 후보 없음")
                    break
                cand=_enrich_formula(cand,pts)
                if not cand:
                    _formula_enrich_failures+=1
                    diag(slot,"formula_enrich","계산형 채점요소 조건 불충족")
                    # Same template family repeatedly failing cannot improve by brute-force retries.
                    # Fall back to the normal selector early to save time/API and avoid noisy loops.
                    if _formula_enrich_failures>=4:
                        diag(slot,"formula_fallback","계산 템플릿 채점요소 부족으로 일반 출제 경로 전환")
                        break
                    continue
                if too_similar(cand,prior+qs):
                    diag(slot,"formula_similarity","기존 문항과 유사")
                    continue
                formula_errs=validate_formula_question(cand)
                if formula_errs:
                    diag(slot,"formula_python_validator",
                         " / ".join(map(str,formula_errs)) if isinstance(formula_errs,(list,tuple)) else str(formula_errs))
                    continue

                if quality_active:
                    try:
                        judge_calls+=1
                        review=judge_question(api_key,judge_model,cand,"",style)
                    except Exception as ex:
                        review={"pass":False,"reason":"AI 품질심사 호출 실패: "+str(ex)}
                    cand["ai_quality"]=review
                    if not review.get("pass"):
                        judge_rejects+=1
                        diag(slot,"formula_ai_judge",review.get("reason",""),
                             fatal_flags=review.get("fatal_flags",[]),
                             scores=review.get("scores",{}),
                             weakest_point=review.get("weakest_point",""),
                             blind_verdict=review.get("blind_verdict"),
                             grounded_verdict=review.get("grounded_verdict"))
                        continue
                elif tuning_mode:
                    cand["ai_quality"]={"pass":None,"mode":"tuning_fast_python_checked"}
                else:
                    cand["ai_quality"]={"pass":None,"mode":"not_run"}

                q=cand;formula_used+=1;break

        if q is None:
            first_pat=weighted_pick(rng,pts,calc=False,used=used_patterns)
            local_rejected_topics=set()

            # 한 문제 슬롯이 수십 번 반복되지 않도록 AI 후보 예산을 제한한다.
            # 품질 기준은 그대로이며, 실패 원인에 따라 다른 패턴으로 즉시 전환한다.
            slot_candidate_budget = 2 if quality_active else (4 if tuning_mode else 8)
            slot_candidates_used = 0
            # selector 자체 호출도 슬롯당 제한한다. REJECT/timeout도 호출 1회로 계산한다.
            selector_attempt_limit = 0
            selector_attempts = 0

            for pat in _concept_patterns(rng,pts,first_pat):
                if (quality_active or tuning_mode) and (
                    slot_candidates_used >= slot_candidate_budget or
                    False
                ):
                    _budget_reason=(
                        f"샘플 후보 예산 소진(candidate={slot_candidates_used}/{slot_candidate_budget})"
                        if tuning_mode else
                        f"문항 후보 예산 소진(candidate={slot_candidates_used}/{slot_candidate_budget})"
                    )
                    diag(slot,"slot_budget",_budget_reason,pattern=pat.get("id"))
                    break
                need=len(pat["subpoints"])

                for _ in range(1 if quality_active else (2 if tuning_mode else 32)):
                    if (quality_active or tuning_mode) and (
                        slot_candidates_used >= slot_candidate_budget or
                        False
                    ):
                        break
                    relation_meta={}
                    bundle=[]

                    if selector_active:
                        # R10: 관계 selector는 Python에서 수행한다.
                        # 최종 A/B에서도 selector AI 호출을 없애 생성시간/비용을 줄인다.
                        bundle,relation_meta=_smart_relation_bundle(
                            db_path,dom,need,used_answers,
                            set(used_topics)|local_rejected_topics,rng,
                            pattern_id=pat.get("id","")
                        )
                        if len(bundle)<need:
                            diag(slot,"python_exam_value_selector",
                                 f"출제 가치·관계성 후보 부족: 필요 {need}, 확보 {len(bundle)}",
                                 pattern=pat.get("id"),
                                 score_pipeline_diagnostic=copy.deepcopy(
                                     (relation_meta or {}).get("score_pipeline_diagnostic",{})
                                 ))
                            break
                        slot_candidates_used += 1
                        if pts==2:
                            _tt=set(str(x).strip() for x in relation_meta.get("thinking_types",[]) if str(x).strip())
                            _rel=str(relation_meta.get("relation","")).strip()
                            if len(_tt)<2 or len(_rel)<4:
                                diag(slot,"two_point_relation_gate",
                                     "2점 관계형 문항 조건 불충족: 최소 2종 사고행동 + 명시적 관계 필요",
                                     pattern=pat.get("id"),
                                     candidate_topics=[str(x.get("topic","")) for x in bundle])
                                for a in bundle:
                                    local_rejected_topics.add(a["topic"])
                                continue
                            relation_meta["quality_directive"]=(
                                "2점 문항도 두 정답을 각 문장에서 독립적으로 찾아 쓰게 만들지 말 것. "
                                "첫 요소의 판단 또는 자료 해석이 두 번째 요소 판단에 반드시 사용되게 구성하고, "
                                "정답 정의·고유특징을 지문에 거의 그대로 제시하지 말 것. "
                                "현재 패턴의 사고구조를 반드시 지킬 것: "
                                + str(pat.get("quality_rule",pat.get("name","")))
                            )
                    else:
                        bundle=related_bundle(
                            db_path,dom,need,used_answers,
                            set(used_topics)|local_rejected_topics
                        )
                        if len(bundle)<need:
                            diag(slot,"related_bundle",
                                 f"원문 잠금 후보 부족: 필요 {need}, 확보 {len(bundle)}",
                                 pattern=pat.get("id"))
                            break

                    # AI 호출 전에 Python이 채점 논리와 기출형 골격을 확정한다.
                    if relation_meta:
                        relation_meta["exam_skeleton"]=_skeleton_for_pattern(pat.get("id",""))
                        relation_meta["scoring_plan"]=_scoring_plan(pat,bundle)
                        relation_meta["quality_directive"]=(
                            str(relation_meta.get("quality_directive","")) + " "
                            + str(pat.get("quality_rule",""))
                        ).strip()

                    ctx=bundle_context(db_path,bundle)
                    cand=None

                    if selector_active:
                        try:
                            ai_calls+=1
                            cand=rewrite_bundle(
                                api_key,model,bundle,pts,section,pat,
                                slot["question_type"],slot["material_form"],style,
                                source_context=ctx,relation_meta=relation_meta
                            )
                        except Exception as ex:
                            cand=None
                            diag(slot,"question_writer_call",
                                 "AI 문항 작성 호출 실패: "+str(ex),
                                 pattern=pat.get("id"))

                        if cand is not None:
                            shape_errs=_candidate_shape_errors(cand,pat,pts)
                            if shape_errs:
                                diag(slot,"candidate_shape_validator",
                                     " / ".join(shape_errs),
                                     pattern=pat.get("id"))
                                if tuning_mode:
                                    for _a in bundle:
                                        local_rejected_topics.add(str(_a.get("topic","")))
                                cand=None

                        if tuning_mode and quality_active:
                            raise RuntimeError("내부 모드 오류: tuning_mode에서 AI quality judge가 활성화되었습니다.")

                        if cand is not None and quality_active:
                            try:
                                judge_calls+=1
                                review=judge_question(
                                    api_key,judge_model,cand,ctx,style
                                )
                            except Exception as ex:
                                review={"pass":False,"reason":"AI 품질심사 호출 실패: "+str(ex)}
                            cand["ai_quality"]=review

                            if not review.get("pass"):
                                judge_rejects+=1
                                _fatal=set(str(x) for x in review.get("fatal_flags",[]))
                                diag(slot,"question_ai_judge",review.get("reason",""),
                                     pattern=pat.get("id"),
                                     fatal_flags=review.get("fatal_flags",[]),
                                     scores=review.get("scores",{}),
                                     weakest_point=review.get("weakest_point",""),
                                     blind_verdict=review.get("blind_verdict"),
                                     grounded_verdict=review.get("grounded_verdict"))

                                _hard={"ROTE_ONLY","TOO_EASY","DIRECT_ANSWER_LEAK","AMBIGUOUS","DECORATIVE_MATERIAL"}
                                if _fatal & _hard:
                                    for a in bundle:
                                        local_rejected_topics.add(a["topic"])
                                    relation_meta["force_pattern_switch"]=True
                                elif bundle:
                                    local_rejected_topics.add(bundle[0]["topic"])
                                cand=None

                        # 최종 모드와 튜닝 모드 모두 DB/Python grounding은 확인한다.
                        if cand is not None:
                            grounded_errs=validate_grounded_question(
                                cand,ctx,
                                allow_ai_grounded=True,
                                require_ai_quality=(not tuning_mode)
                            )
                            if grounded_errs:
                                diag(slot,"grounded_python_validator",
                                     "DB/Python 근거검증 미통과: " +
                                     (" / ".join(map(str,grounded_errs)) if isinstance(grounded_errs,(list,tuple)) else str(grounded_errs)),
                                     pattern=pat.get("id"))
                                if tuning_mode:
                                    for _a in bundle:
                                        local_rejected_topics.add(str(_a.get("topic","")))
                                cand=None
                            elif too_similar(cand,prior+qs):
                                diag(slot,"question_similarity",
                                     "기존 문항과 유사",
                                     pattern=pat.get("id"))
                                if tuning_mode:
                                    for _a in bundle:
                                        local_rejected_topics.add(str(_a.get("topic","")))
                                cand=None
                            elif tuning_mode:
                                cand["ai_quality"]={
                                    "pass":None,
                                    "mode":"tuning_fast_manual_review",
                                    "note":"Blind/Grounded/섹션 AI 심사는 튜닝 속도를 위해 생략"
                                }

                        if cand is None:
                            if bundle and not any(a["topic"] in local_rejected_topics for a in bundle):
                                local_rejected_topics.add(bundle[0]["topic"])
                            if relation_meta.get("force_pattern_switch"):
                                break
                            continue

                    else:
                        cand=safe_bundle_question(
                            bundle,pts,pat,slot["question_type"],slot["material_form"]
                        )
                        errs=validate_grounded_question(cand,ctx)
                        if errs or too_similar(cand,prior+qs):
                            diag(slot,"safe_fallback_validator",
                                 " / ".join(map(str,errs)) if errs else "기존 문항과 유사",
                                 pattern=pat.get("id"))
                            if bundle:
                                local_rejected_topics.add(bundle[0]["topic"])
                            continue
                        cand["ai_quality"]={"pass":None,"mode":"source_locked_fallback"}
                        fallbacks+=1

                    q=cand
                    if tuning_mode and relation_meta.get("score_diagnostic"):
                        q["_score_diagnostic"]=copy.deepcopy(relation_meta["score_diagnostic"])
                    used_answers.update(q["answer"])
                    for a in bundle:
                        used_topics.add(a["topic"])
                    break

                if q is not None:
                    break

        if q is None:
            err=RuntimeError(
                f"{section} {slot['number']}번({dom})을 품질 기준으로 만들 수 없습니다. "
                "품질 기준을 낮추지 않고 다른 청사진을 재시도합니다."
            )
            err.generation_diagnostics=diagnostics[-15:]
            raise err

        q["number"]=slot["number"]
        q["blueprint_domain"]=dom
        q["concept_families"]=sorted(families_for(q))
        q["fingerprint"]=q.get("fingerprint") or fingerprint(q)
        qs.append(q)
        used_patterns.append(q.get("pattern_id"))

    errs=_validate_sample_exam(qs) if tuning_mode else validate_exam(qs,count,points)
    if errs:
        err=RuntimeError("시험 자동검증 실패: "+" / ".join(errs))
        err.generation_diagnostics=diagnostics[-15:]+[{
            "section":section,"stage":"exam_python_validator",
            "reason":" / ".join(errs)
        }]
        raise err

    exam={
      "exam_title":f"기술 임용 모의고사 전공 {section}",
      "section":section,"total_points":points,"questions":qs,"verified":False if tuning_mode else True,
      "blueprint":plan,
      "generation_stats":{
          "ai_calls":ai_calls,
          "safe_fallbacks":fallbacks,
          "formula_questions":formula_used,
          "ai_judge_calls":judge_calls,
          "ai_judge_rejects":judge_rejects,
          "ai_selector_calls":selector_calls,
          "tuning_relation_selector":"python" if tuning_mode else "ai",
      },
      "used_answers":list(used_answers),
      "verification_note":("품질 튜닝용 6문항: 강한 관계 그래프 선별 + DB/Python 정답·구조 검증 + AI 작성. 최종 AI 품질심사 생략" if tuning_mode else "DB/Python 정답 고정 + 구조검증 + AI 독립 품질심사 통과")
    }

    if quality_active and not tuning_mode:
        try:
            judge_calls+=1
            section_review=judge_exam(api_key,judge_model,exam,style)
        except Exception as ex:
            section_review={"pass":False,"reason":"섹션 AI 심사 호출 실패: "+str(ex)}
        exam["section_ai_quality"]=section_review
        exam["generation_stats"]["ai_judge_calls"]=judge_calls
        if not section_review.get("pass"):
            err=RuntimeError(
                f"{section} 섹션 전체 AI 품질심사 탈락: "
                +str(section_review.get("reason",""))
            )
            err.generation_diagnostics=diagnostics[-12:]+[{
                "section":section,
                "stage":"section_ai_judge",
                "reason":str(section_review.get("reason","")),
                "scores":{
                    "exam_realism":section_review.get("exam_realism"),
                    "variety":section_review.get("variety"),
                    "difficulty_balance":section_review.get("difficulty_balance"),
                }
            }]
            raise err
    else:
        exam["section_ai_quality"]={
            "pass":None,
            "mode":"tuning_fast_manual_review" if tuning_mode else "not_run"
        }

    return exam


def make_quality_sample(db_path,domains=None,api_key="",model="gpt-5.6-luna",
                        ai_enabled=True,judge_model=None,seed=None):
    """
    품질 튜닝 전용 6문항.
    정확히 2점 2개 + 4점 4개.
    AI 관계성 selector/Blind/Grounded/섹션 심사는 호출하지 않는다.
    Python 관계묶음 + Luna writer + deterministic grounding 검증만 수행한다.
    """
    exam=make_section(
        db_path,"SAMPLE",6,20,domains,api_key,model,
        ai_enabled,False,judge_model,seed=seed,
        tuning_mode=True
    )

    stats=exam.get("generation_stats",{})
    if exam.get("section")!="SAMPLE":
        raise RuntimeError("SAMPLE6 내부 검증 실패: section이 SAMPLE이 아닙니다.")

    scores=[int(q.get("points",0)) for q in exam.get("questions",[])]
    if scores != [2,2,4,4,4,4]:
        raise RuntimeError(
            "SAMPLE6 내부 검증 실패: 배점이 [2,2,4,4,4,4]가 아닙니다."
        )

    if int(stats.get("ai_selector_calls",0) or 0) != 0:
        raise RuntimeError(
            "SAMPLE6 내부 검증 실패: 튜닝 모드에서 AI selector가 호출되었습니다. "
            "현재 배포 파일 버전이 섞여 있습니다."
        )

    judge_count=int(
        stats.get("ai_quality_judge_calls",
        stats.get("ai_judge_calls",
        stats.get("judge_calls",0))) or 0
    )
    if judge_count != 0:
        raise RuntimeError(
            "SAMPLE6 내부 검증 실패: 튜닝 모드에서 AI 품질심사가 호출되었습니다. "
            "현재 배포 파일 버전이 섞여 있습니다."
        )

    exam["builder_api_version"]=BUILDER_API_VERSION
    exam["sample_mode"]="PYTHON_RELATION_WRITER_ONLY"
    return exam


def make_ab(db_path,a_count=12,a_points=40,b_count=11,b_points=40,domains=None,
            api_key="",model="gpt-5.6-luna",ai_enabled=True,ai_quality_enabled=True,
            judge_model=None,seed=None):
    base=0 if seed is None else int(seed)
    last_error=None
    ab_diagnostics=[]

    def collect_error(ex,attempt):
        rows=getattr(ex,"generation_diagnostics",None)
        if rows:
            for row in rows:
                x=dict(row)
                x["attempt"]=attempt
                ab_diagnostics.append(x)
        else:
            ab_diagnostics.append({
                "attempt":attempt,
                "stage":"section_or_pair",
                "reason":str(ex)
            })

    for a_try in range(1):
        a_seed=None if seed is None else base + a_try*1000
        try:
            A=make_section(
                db_path,"A",a_count,a_points,domains,api_key,model,
                ai_enabled,ai_quality_enabled,judge_model,
                seed=a_seed
            )
        except Exception as ex:
            last_error=ex
            collect_error(ex,f"A-{a_try+1}")
            continue

        af=set().union(*(families_for(q) for q in A["questions"]))
        for b_try in range(1):
            b_seed=None if seed is None else base + 1 + a_try*1000 + b_try*37
            try:
                B=make_section(
                    db_path,"B",b_count,b_points,domains,api_key,model,
                    ai_enabled,ai_quality_enabled,judge_model,
                    seed=b_seed,
                    previous_questions=A["questions"],
                    shared_answers=set(A.get("used_answers",[]))
                )
                bf=set().union(*(families_for(q) for q in B["questions"]))
                if af & bf:
                    last_error=RuntimeError(
                        "A/B concept-family 중복 발견: "+", ".join(sorted(af&bf))
                    )
                    collect_error(last_error,f"A-{a_try+1}/B-{b_try+1}")
                    continue
                quality_active=bool(ai_enabled and ai_quality_enabled and api_key)
                if quality_active:
                    try:
                        pair_review=judge_ab_pair(
                            api_key,judge_model or model,A,B,
                            official_style_profile(db_path)
                        )
                    except Exception as ex:
                        pair_review={"pass":False,"reason":"A/B 종합 AI 심사 호출 실패: "+str(ex)}
                    if not pair_review.get("pass"):
                        last_error=RuntimeError(
                            "A/B 종합 AI 품질심사 탈락: "
                            +str(pair_review.get("reason",""))
                        )
                        collect_error(last_error,f"A-{a_try+1}/B-{b_try+1}")
                        continue
                    A["ab_pair_ai_quality"]=pair_review
                    B["ab_pair_ai_quality"]=pair_review
                else:
                    A["ab_pair_ai_quality"]={"pass":None,"mode":"not_run"}
                    B["ab_pair_ai_quality"]={"pass":None,"mode":"not_run"}

                A["effective_seed"]=a_seed
                B["effective_seed"]=b_seed
                A["generation_retry"]={"A_attempt":a_try+1}
                B["generation_retry"]={"B_attempt":b_try+1}
                return A,B
            except Exception as ex:
                last_error=ex
                collect_error(ex,f"A-{a_try+1}/B-{b_try+1}")
                continue

    err=RuntimeError(
        "정답/품질 기준을 낮추지 않은 상태에서 A/B 편성에 실패했습니다: "
        +str(last_error)
    )
    err.generation_diagnostics=ab_diagnostics[-30:]
    raise err
