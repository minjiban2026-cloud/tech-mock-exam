import copy
BUILDER_API_VERSION = "SEMANTIC-OP-DISCOVERY-R48-20260903"

import random, math, re, sqlite3, itertools
from difflib import SequenceMatcher
from formula_templates import generate_formula_question
from reasoning_capabilities import generate_reasoning_question, coverage_inventory
from retrieval import related_bundle,bundle_context,official_style_profile,candidate_cluster
from ai_wrapper import rewrite_bundle,safe_bundle_question
from validators import validate_formula_question,validate_grounded_question,too_similar,validate_exam,fingerprint,t2_clause_quality,source_evidence_grounded
from patterns import blueprint,weighted_pick
from concept_families import families_for
from quality_judge import select_coherent_bundle,judge_question,judge_exam,judge_ab_pair


# R34: DB source/candidate analysis cache.  The cache is invalidated by DB mtime.
_T2_SOURCE_CACHE={}
_T2_CAPACITY_CACHE={}

def _t2_identity_key(value):
    """Preserve short numeric/alphanumeric identities such as 4G, 2.5G, p1, p16."""
    x=_norm_anchor_text(value).lower()
    x=re.sub(r"^[\s·•○▶□■♥ü\-–—]+","",x).strip()
    x=re.split(r"[:：]",x,1)[0].strip()
    x=re.sub(r"[()（）\[\]]","",x)
    return re.sub(r"[^가-힣a-z0-9.]+","",x)

def _t2_series_signature(value):
    k=_t2_identity_key(value)
    if not k: return ""
    # Keep letters/shape, normalize varying numeric members.  p1/p2 and 4g/5g become stable series.
    return re.sub(r"\d+(?:\.\d+)?","#",k)

def _t2_structural_series(a,b):
    ka,kb=_t2_identity_key(a),_t2_identity_key(b)
    if not ka or not kb or ka==kb: return False
    sa,sb=_t2_series_signature(a),_t2_series_signature(b)
    return bool(sa and sa==sb and "#" in sa)

def _t2_axis_signature(text):
    x=_norm_anchor_text(text)
    cats=set()
    rules={
      "number":r"\d", "speed":r"속도|bps|hz|대역", "start":r"시작|부터",
      "count":r"개|회|번", "increase":r"증가|향상|높", "decrease":r"감소|저하|낮",
      "condition":r"경우|조건|때|하면|따라", "effect":r"효과|가능|향상|개선|방지|촉진|억제",
      "procedure":r"단계|순서|검사|반복|처리|수정|전송|분석|선정|생성|결합",
      "property":r"특징|특성|장점|단점|범위|간격|크기|길이|온도|압력|전류|전압|응력|변형",
    }
    for name,pat in rules.items():
        if re.search(pat,x,re.I): cats.add(name)
    return cats

def _t2_axis_similarity(a,b):
    aa=re.sub(r"\d+(?:\.\d+)?","#",re.sub(r"[^가-힣A-Za-z0-9.%]+","",_norm_anchor_text(a)).lower())
    bb=re.sub(r"\d+(?:\.\d+)?","#",re.sub(r"[^가-힣A-Za-z0-9.%]+","",_norm_anchor_text(b)).lower())
    seq=SequenceMatcher(None,aa,bb).ratio() if aa and bb else 0.0
    ca,cb=_t2_axis_signature(a),_t2_axis_signature(b)
    cat=(len(ca&cb)/max(1,len(ca|cb))) if (ca or cb) else 0.0
    return max(seq,0.55*seq+0.45*cat)

def _t2_joint_pair_quality(core_anchor,core_pair,contrast_anchor,contrast_pair):
    """Validate the comparison axis jointly, not as two independent good pairs."""
    if not core_pair or not contrast_pair: return {"ok":False,"reason":"joint pair missing"}
    a1,a2=map(_norm_anchor_text,core_pair); b1,b2=map(_norm_anchor_text,contrast_pair)
    if _t2_identity_key(core_anchor.get("answer"))==_t2_identity_key(contrast_anchor.get("answer")):
        return {"ok":False,"reason":"same identity"}
    s1=_t2_axis_similarity(a1,b1); s2=_t2_axis_similarity(a2,b2)
    series=_t2_structural_series(core_anchor.get("answer",""),contrast_anchor.get("answer",""))
    # The swapped second fact must actually differ on the same axis.
    cross=_t2_axis_similarity(a2,b2)
    raw2a=re.sub(r"[^가-힣A-Za-z0-9]","",a2).lower(); raw2b=re.sub(r"[^가-힣A-Za-z0-9]","",b2).lower()
    exact_same=bool(raw2a and raw2a==raw2b)
    if exact_same: return {"ok":False,"reason":"slot2 identical"}
    # Strong structural series (4G/5G, p1/p2...) may differ mainly by values; otherwise both axes need clear alignment.
    if series:
        ok=(s1>=0.30 and s2>=0.30)
    else:
        # Near-identical slot1 across unrelated labels is a strong sign of local-block ownership contamination.
        n1a=re.findall(r"\d+(?:\.\d+)?",a1); n1b=re.findall(r"\d+(?:\.\d+)?",b1)
        if s1>=0.985 and not (n1a and n1b and n1a!=n1b):
            return {"ok":False,"reason":"slot1 ownership collision","slot1_axis":round(s1,3),"slot2_axis":round(s2,3),"series":False}
        ok=(s1>=0.38 and s2>=0.38 and (s1+s2)>=1.28)
    if not ok:
        return {"ok":False,"reason":"comparison axis mismatch","slot1_axis":round(s1,3),"slot2_axis":round(s2,3),"series":series}
    return {"ok":True,"reason":"joint_axis_pass","slot1_axis":round(s1,3),"slot2_axis":round(s2,3),"series":series}

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
    # 동일 상위개념 아래의 병렬 하위항목은 자연적 문제단위 점수만으로 과대평가하지 않는다.
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


def _exact_anchor_reference(a,b):
    """
    generic shared words가 아니라 anchor 개념 자체가 상대 근거에 실제 등장하는지 본다.
    반환: 0/1/2 (a->b, b->a 각각 1점)
    """
    ae=_topic_core(a.get("evidence",""))
    be=_topic_core(b.get("evidence",""))
    akeys={
        _topic_core(a.get("topic","")),
        _topic_core(a.get("answer","")),
    }
    bkeys={
        _topic_core(b.get("topic","")),
        _topic_core(b.get("answer","")),
    }
    akeys={x for x in akeys if len(x)>=2}
    bkeys={x for x in bkeys if len(x)>=2}
    score=0
    if any(x and x in be for x in akeys):
        score+=1
    if any(x and x in ae for x in bkeys):
        score+=1
    return score


def _four_point_reasoning_chain_profile(chosen):
    """
    4점 후보의 '실제 풀이 연결성' 진단.
    단순히 같은 페이지/같은 분야의 공통어만 공유하는 경우를 낮춘다.
    """
    b=list(chosen or [])
    if len(b)<2:
        return {"exact_links":0,"generic_only_pairs":0,"score":0.0}

    exact_links=0
    generic_only=0
    for i in range(len(b)-1):
        ex=_exact_anchor_reference(b[i],b[i+1])
        exact_links += ex
        if ex==0:
            generic_only += 1

    # exact anchor 참조는 실제 개념 연결의 강한 신호.
    # generic-only 인접쌍은 '유체/방정식/흐름' 같은 공통어 때문에 묶인 경우가 많아 감점.
    score=exact_links*1.5 - generic_only*4.0
    return {
        "exact_links":int(exact_links),
        "generic_only_pairs":int(generic_only),
        "score":float(score),
    }


def _four_point_shortcut_penalty(chosen):
    """
    4점 문항에서 '실제 풀이'가 아니라 정의의 포함/합성 관계만으로
    여러 점수를 채우는 후보를 감점한다.

    - 2-anchor: 상위범주/하위개념 또는 A의 정의에 B가 그대로 들어가는 경우
      예) 같은 상위범주의 병렬 하위항목
    - 3-anchor: 하나의 정의가 나머지 둘을 '+' 등으로 그대로 합성하는 경우
      예) 한 항목이 다른 두 항목의 단순 합성 정의인 경우

    단순히 서로의 용어가 근거에 등장한다는 이유만으로는 감점하지 않는다.
    정의·포함·구성 신호가 함께 있어야 한다.
    """
    b=list(chosen or [])
    if len(b)<2:
        return 0.0

    def _definition_link(a, other):
        raw=_norm_anchor_text(a.get("evidence",""))
        ev=_topic_core(raw)
        ot=_topic_core(other.get("answer","")) or _topic_core(other.get("topic",""))
        if not ot or ot not in ev:
            return False

        # 정의/범주/합성 신호. '+'는 강한 정의합성 신호로 취급한다.
        relation_signal=(
            bool(re.search(r"\s\+\s",raw))
            or bool(re.search(
                r"(포함|구성|범주|종류|일종|나뉘|구분|묶|전체|하위|상위|"
                r"정의상\s*포함|단순\s*합성)",
                raw
            ))
        )
        # 단순한 'A : B' 정의 표기만으로는 감점하지 않는다.
        # 단순한 콜론 표기 자체는 실제 과정/관계 설명일 수도 있으므로 감점 근거로 쓰지 않는다.
        return relation_signal

    penalty=0.0

    if len(b)==2:
        # 양쪽 중 한쪽이라도 다른 anchor를 정의상 포함하는 구조면 4점 2+2의
        # 독립 채점요소로 보기 어렵다. 강하게 감점한다.
        if _definition_link(b[0],b[1]) or _definition_link(b[1],b[0]):
            penalty -= 7.0
        return penalty

    # 3개 이상: 하나의 근거가 나머지 두 정답을 그대로 합성 정의하는 경우.
    for i,a in enumerate(b):
        ev=_topic_core(a.get("evidence",""))
        others=[
            _topic_core(x.get("answer","")) or _topic_core(x.get("topic",""))
            for j,x in enumerate(b) if j!=i
        ]
        if len(others)>=2 and all(x and x in ev for x in others[:2]):
            raw=_norm_anchor_text(a.get("evidence",""))
            if "+" in raw or re.search(r"(합|묶|구성|포함|범주|전체)",raw):
                penalty -= 5.0
    return penalty

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
        v=re.sub(r"^\s*(?:\(\d+\)|\d+[.)])\s*","",v)
        v=re.sub(r"^\s*[□▶■·•○●♥ü\-–—]+\s*","",v)
        a[key]=v.strip()
    return a


def _two_point_core_target_ok(anchor):
    ans=_norm_anchor_text(anchor.get("answer","")).strip()
    topic=_norm_anchor_text(anchor.get("topic","")).strip()
    if not ans:
        return False
    # parser/문서 표지 및 문장 중간 조각
    # Parenthesized numbering such as (1) SFD is a valid concept label; strip numbering before judging.
    if re.search(r"(기출|여기서\s*[A-Za-z]|^[가나다라마바사아자차카타파하]\)$)",ans,re.I):
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
        "장점","단점","종류","특징","절차","정의","분류","기준","역할","사람",
        "유지","성장","증식","수확","과정","방법","효과","목적","원인","이유","결과",
        "조치","대책","겨울철","여름철","봄철","가을철","전제","조건","현상"
    }
    if ans.replace(" ","") in {x.replace(" ","") for x in generic}:
        return False
    # 지나치게 긴 목차형 topic은 중심 정답으로 쓰지 않는다.
    if _heading_like(ans) or _heading_like(topic):
        return False

    ev=_norm_anchor_text(anchor.get("evidence","")).strip()
    # Example markers may coexist with a complete definition/relationship; do not reject the whole anchor solely for that.
    if ev:
        # Remove leading bullets and the answer/topic label once.
        body=re.sub(r"^[■▶●♥□·•○\-\s]+","",ev).strip()
        for label in (ans,topic):
            if label and body.startswith(label):
                body=body[len(label):].strip()
                break
        body=re.sub(r"^[\s:_\-·]+","",body).strip()

        # Genuine concept evidence normally proceeds with a definition/description.
        # A container heading often immediately opens ① child-item / another heading instead.
        if re.match(r"^(?:①|1\)|1\.|[가-힣A-Za-z0-9 ]{2,25}\s*(?:엔진|기술|용어|종류|구성)\s*(?:[-:]|$))",body):
            if not re.search(r"(정의|의미|란|:|：)",body[:45]):
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

def _two_point_support_quality(core_anchor, support_text, page_text=""):
    """
    Writer/Judge 호출 전에 T2 support가 '첫 정답의 정의 재진술'인지 검사한다.
    핵심은 문장 안에 단순 키워드가 있는지가 아니라, 첫 개념을 식별하는 정의부 외에
    별도의 조건/효과/절차/비교 정보가 실제로 존재하는지다.
    """
    t=_norm_anchor_text(support_text).strip()
    ev=_norm_anchor_text(core_anchor.get("evidence","")).strip()
    if not _two_point_support_ok(t):
        return {"ok":False,"score":-99.0,"reason":"support_not_explanatory","action":"reject"}

    def toks(x):
        stop={"정의","특징","개념","방법","기법","경우","대한","위한","하는","한다","된다","있다","사용","이용"}
        return {z for z in re.findall(r"[가-힣A-Za-z0-9]{2,}",_norm_anchor_text(x)) if z not in stop}

    st=toks(t); et=toks(ev)
    overlap=(len(st & et)/max(1,len(st))) if st else 1.0

    # 실제 두 번째 사실을 나타내는 구조: 원인→결과, 조건→결과, 추가 절차/효과, 명시 비교.
    causal=bool(re.search(r"(때문|따라|하므로|하므로|하여.*(?:증가|감소|향상|발생|가능|파손|변화)|"
                          r"(?:증가|감소|향상|발생|변화).*따라)",t))
    condition=bool(re.search(r"(경우|조건|일\s*때|할\s*때|에서.*(?:가능|유용|사용|적용)|없이)",t))
    compare=bool(re.search(r"(반면|달리|비교|차이|구별|보다\s+(?:크|작|높|낮)|대신)",t))
    procedure=bool(re.search(r"(먼저|다음|이후|절차|단계|기록.*제출|제출.*반복|순서)",t))
    effect=bool(re.search(r"(유용|향상|감소|증가|줄여|개선|방지|촉진|발휘|확장|파손|급\s*상승)",t))
    application=bool(re.search(r"(적용.*(?:경우|대상|상황)|활용.*(?:경우|목적|분야)|선정.*기준)",t))
    secondary=sum([causal,condition,compare,procedure,effect,application])

    # 짧은 정의/단일 작용 및 evidence와 거의 동일한 support는 제거.
    simple_action=bool(re.search(r"(먹고.*뱉|이용.*(?:발생|생성)|받아들.*내보|흡수.*배출)",t))
    if simple_action and secondary==0:
        return {"ok":False,"score":-8.0,"reason":"same_fact_action_restatement","action":"reject",
                "overlap":round(overlap,3)}
    if overlap>=0.72 and secondary==0:
        return {"ok":False,"score":-7.0,"reason":"same_fact_definition_restatement","action":"reject",
                "overlap":round(overlap,3)}
    if overlap>=0.82 and secondary<=1 and not (causal or condition or compare or procedure):
        return {"ok":False,"score":-6.5,"reason":"definition_dominates_second_task","action":"reject",
                "overlap":round(overlap,3)}
    if len(t)<34 and secondary==0:
        return {"ok":False,"score":-6.0,"reason":"short_single_fact_support","action":"reject",
                "overlap":round(overlap,3)}
    if secondary==0:
        return {"ok":False,"score":-4.0,"reason":"no_distinct_second_task_signal","action":"reject",
                "overlap":round(overlap,3)}

    score=secondary*1.6 + min(1.6,max(0.0,(len(t)-28)/60.0)) + max(0.0,0.7-overlap)
    return {"ok":True,"score":round(float(score),3),"reason":"distinct_support_ready","action":"keep",
            "overlap":round(overlap,3),
            "signals":{"causal":causal,"condition":condition,"compare":compare,
                       "procedure":procedure,"effect":effect,"application":application}}

def _two_point_candidate_key(bundle):
    if not bundle:
        return ""
    a=bundle[0]
    return "|".join([
        str(a.get("source_name","")),
        str(a.get("page_no","")),
        _topic_core(a.get("answer","")),
    ])

def _two_point_contrast_text(anchor, page_text, support_text=""):
    """
    같은 페이지의 '다른 개념/다른 절차' 한 조각을 오답·비교 자료로만 제공한다.
    정답 anchor로 쓰지 않으며, Writer가 첫 명칭을 정의 한 문장으로 노출하지 않게 하는
    판별 거리(discrimination distance)를 확보하기 위한 context다.
    """
    core=_topic_core(anchor.get("answer",""))
    support_comp=re.sub(r"\s+","",_norm_anchor_text(support_text))
    lines=[re.sub(r"[ \t]+"," ",str(x)).strip() for x in str(page_text or "").splitlines() if str(x).strip()]
    candidates=[]
    for line in lines:
        clean=re.sub(r"^\s*[-•·○▶□■]+\s*","",line).strip()
        comp=re.sub(r"\s+","",clean)
        if not clean or len(clean)<12 or len(clean)>180:
            continue
        if core and core in _topic_core(clean):
            continue
        if support_comp and (support_comp[:24] in comp or comp[:24] in support_comp):
            continue
        # sibling 정의/절차/상황처럼 비교에 실제 쓸 수 있는 행을 우선
        structured=bool(re.search(r"[:：]|^\d+\)|^\d+\.|특징|상황|절차|원리|조건|효과|원인|결과|비교",clean,re.I))
        if not structured:
            continue
        # 문서 표지/목차 제외
        if re.search(r"(서브노트|교과서|^\d+$|^B\d)",clean):
            continue
        score=0.0
        if ":" in clean or "：" in clean: score+=2.0
        if re.search(r"(방법|기법|특징|상황|절차|원리|경우|이용|적용|발생|변화)",clean): score+=1.2
        if 24<=len(clean)<=120: score+=0.8
        candidates.append((score,clean))
    candidates.sort(key=lambda x:x[0],reverse=True)
    return candidates[0][1] if candidates else ""


def _t2_near_copy_errors(cand, bundle):
    """
    AI Judge 전에 2점 지문이 고정 원문/채점근거를 거의 복사했는지 검사한다.
    정확한 정답명 노출 + 긴 원문 구절 복사를 deterministic하게 차단한다.
    """
    if int(cand.get("points",0) or 0)!=2:
        return []
    material=" ".join([
        str(cand.get("intro","") or ""), str(cand.get("passage","") or ""),
        " ".join(map(str,cand.get("conditions",[]) or [])),
        " ".join(map(str,cand.get("tasks",[]) or [])),
    ])
    mc=re.sub(r"[^가-힣A-Za-z0-9]","",material).lower()
    errs=[]
    if not mc:
        return ["2점 자료 없음"]
    # 정답명 직접 노출
    for i,a in enumerate(bundle or []):
        if i>0 and a.get("derived_support"):
            continue
        ans=re.sub(r"[^가-힣A-Za-z0-9]","",_norm_anchor_text(a.get("answer",""))).lower()
        if len(ans)>=3 and ans in mc:
            errs.append("중심 정답명 직접 노출")
            break
    # 원문/채점근거의 14자 이상 연속 구절 복사
    for a in bundle or []:
        for raw in (a.get("evidence",""), a.get("answer","")):
            rc=re.sub(r"[^가-힣A-Za-z0-9]","",_norm_anchor_text(raw)).lower()
            if len(rc)<14:
                continue
            n=min(22,max(14,len(rc)//3))
            hit=False
            for j in range(0,max(1,len(rc)-n+1),max(1,n//3)):
                frag=rc[j:j+n]
                if len(frag)>=14 and frag in mc:
                    hit=True; break
            if hit:
                errs.append("원문 채점근거 장구절 직접 복사")
                return list(dict.fromkeys(errs))
    return list(dict.fromkeys(errs))

def _two_point_refine_support(core_anchor, support_text):
    """정의 전체가 아니라 둘째 1점에 쓸 추가 조건/효과/절차 한 조각을 고른다."""
    t=_norm_anchor_text(support_text).strip()
    if not t:
        return t
    parts=[x.strip(" -") for x in re.split(r"(?<=[.!?])\s+|\s+---\s+|\s+⇛\s+",t) if x.strip(" -")]
    if len(parts)<=1:
        parts=[t]
    signal=re.compile(r"(경우|조건|때|따라|때문|촉진|증가|감소|향상|개선|방지|유용|활용|적용|"
                      r"반복|제출|기록|평형|파손|급\s*상승|오차|제거|필요)")
    ev=_norm_anchor_text(core_anchor.get("evidence",""))
    et=set(_anchor_tokens(ev))
    ranked=[]
    for part in parts:
        if len(part)<10:
            continue
        pt=set(_anchor_tokens(part))
        overlap=len(pt&et)/max(1,len(pt)) if pt else 1.0
        score=(2.5 if signal.search(part) else 0.0)+min(1.5,len(part)/90.0)-overlap*1.2
        ranked.append((score,part))
    return max(ranked,key=lambda x:x[0])[1] if ranked else t


def _two_point_contrast_anchors(core_anchor, anchors):
    """R34 discovery pool only. Final acceptance is decided by joint slot-axis validation.
    Same-page proximity may broaden discovery, but can never by itself make a final T2 candidate valid.
    """
    src=str(core_anchor.get("source_name","")); page=int(core_anchor.get("page_no",0) or 0)
    ca=_anchor_tokens(core_anchor.get("topic"),core_anchor.get("answer"),core_anchor.get("evidence"))
    rows=[]
    for b0 in anchors:
        b=_two_point_core_anchor_clean(b0)
        if str(b.get("source_name",""))!=src: continue
        try: gap=abs(int(b.get("page_no",0) or 0)-page)
        except Exception: continue
        if gap>1: continue
        if _t2_identity_key(b.get("answer",""))==_t2_identity_key(core_anchor.get("answer","")): continue
        if not _two_point_core_target_ok(b): continue
        if _anchor_fragment_reason(b) or _anchor_internal_contradiction_reason(b): continue
        series=_t2_structural_series(core_anchor.get("answer",""),b.get("answer",""))
        rel=float(_pair_relation_score(core_anchor,b))
        cb=_anchor_tokens(b.get("topic"),b.get("answer"),b.get("evidence"))
        shared={x for x in ca&cb if len(x)>=2}
        # Discovery can be broad because _t2_joint_pair_quality is the mandatory final gate.
        # Adjacent-page candidates still require a real relation; same-page candidates need at least
        # one weak relation signal unless they are an explicit structural series.
        if gap==1 and not series and rel<4.5: continue
        # same-page is discovery-only; final acceptance still requires _t2_joint_pair_quality.
        score=rel+(6.0 if series else 0.0)+min(2.0,len(shared)*0.4)-(1.5*gap)
        rows.append((score,b))
    rows.sort(key=lambda x:x[0],reverse=True)
    return [copy.deepcopy(b) for _,b in rows[:24]]


def _two_point_contrast_anchor(core_anchor, anchors):
    """Backward-compatible single contrast accessor."""
    rows=_two_point_contrast_anchors(core_anchor,anchors)
    return copy.deepcopy(rows[0]) if rows else None

def _t2_reasoning_spec(core_anchor, support_anchor, contrast_anchor, contrast_facts, rng):
    """
    R32 paired-slot contract.

    서로 다른 두 source item에서 각각 두 사실을 확보한다.
    - core_base/core_link: 중심 item의 두 사실
    - contrast_base/contrast_link: sibling item의 두 사실

    표시 사례는 Python이 다음 구조로 고정한다.
      정답 사례: core_base + core_link
      오답 사례: contrast_base + core_link
    오답 사례의 ②는 contrast_link로 수정해야 한다.

    따라서 둘째 정답(contrast_link)은 다른 사례 어디에도 미리 제시되지 않으며,
    수정 대상 슬롯과 채점답이 구조적으로 1:1 대응한다.
    """
    core_base=_norm_anchor_text(core_anchor.get("reasoning_base_fact") or core_anchor.get("evidence","")).strip()
    core_link=_norm_anchor_text(support_anchor.get("evidence") or support_anchor.get("answer","")).strip()
    cf=[_norm_anchor_text(x).strip() for x in (contrast_facts or []) if _norm_anchor_text(x).strip()]
    if len(cf)<2:
        return None
    contrast_base,contrast_link=cf[0],cf[1]
    if not (core_base and core_link and contrast_base and contrast_link):
        return None

    def meaningful_tokens(x):
        stop={"정의","특징","방법","기법","경우","대한","위한","하는","한다","된다","있다",
              "사용","이용","것","수","및","또는","때","원리","개념"}
        return {z for z in re.findall(r"[가-힣A-Za-z0-9]{2,}",x) if z not in stop}
    def sim(a,b):
        aa=re.sub(r"[^가-힣A-Za-z0-9]","",a).lower(); bb=re.sub(r"[^가-힣A-Za-z0-9]","",b).lower()
        return SequenceMatcher(None,aa,bb).ratio() if aa and bb else 1.0
    facts=[core_base,core_link,contrast_base,contrast_link]
    def fact_has_content(x):
        c=re.sub(r"[^가-힣A-Za-z0-9]","",x)
        return len(meaningful_tokens(x))>=2 or len(c)>=14
    if any(not fact_has_content(x) for x in facts):
        return None

    # 같은 item 내부 두 사실은 단순 반복이면 안 된다.
    if sim(core_base,core_link)>=0.80 or sim(contrast_base,contrast_link)>=0.80:
        return None
    # 오답에 들어갈 core_link와 실제 수정답 contrast_link가 사실상 같으면 안 된다.
    if sim(core_link,contrast_link)>=0.78:
        return None

    # 숨은 개념명이 source fact 안에 그대로 박혀 있으면 Writer가 정의-명칭 매칭으로 무너지기 쉽다.
    def contains_label(text,label):
        t=re.sub(r"[^가-힣A-Za-z0-9]","",text).lower()
        l=re.sub(r"[^가-힣A-Za-z0-9]","",label).lower()
        return len(l)>=3 and l in t
    core_label=_norm_anchor_text(core_anchor.get("answer","")).strip()
    contrast_label=_norm_anchor_text(contrast_anchor.get("answer","")).strip()
    if contains_label(core_base,core_label) or contains_label(core_link,core_label):
        return None
    if contains_label(contrast_base,contrast_label) or contains_label(contrast_link,contrast_label):
        return None

    correct="㉠" if rng.random()<0.5 else "㉡"
    wrong="㉡" if correct=="㉠" else "㉠"
    return {
        "mode":"paired_fact_slot_consistency",
        "core_base_fact":core_base,
        "core_link_fact":core_link,
        "contrast_base_fact":contrast_base,
        "contrast_link_fact":contrast_link,
        "hidden_core_answer":core_label,
        "hidden_contrast_answer":contrast_label,
        "correct_option":correct,
        "wrong_option":wrong,
        "first_scoring_action":"두 익명 사례 중 ①과 ②가 같은 source item에 속하는 사례 선택",
        "second_scoring_action":"선택하지 않은 사례의 ②를 그 사례의 ①과 같은 item에 속하는 사실로 수정",
        "correction_contract":{
            "wrong_option":wrong,
            "wrong_slot":"②",
            "displayed_wrong_fact":core_link,
            "expected_replacement_fact":contrast_link,
            "replacement_source":"contrast_item_second_local_source_fact",
        },
        "minimum_inference_steps":2,
    }


def _t2_reasoning_shape_errors(cand, relation_meta, bundle):
    """AI Judge 전에 2점 reasoning-matrix 구조가 실제로 지켜졌는지 확인한다."""
    if int(cand.get("points",0) or 0)!=2:
        return []
    spec=(relation_meta or {}).get("reasoning_spec") or {}
    if spec.get("mode")=="numeric_series_rule_application":
        errs=[]
        passage=str(cand.get("passage","") or "")
        tasks=[str(x) for x in (cand.get("tasks",[]) or [])]
        ans=[str(x) for x in (cand.get("answer",[]) or [])]
        if len(tasks)!=2 or len(ans)!=2:
            errs.append("2점 규칙도출형 채점구조 오류")
        if not ("자료 A" in passage and "자료 B" in passage and "자료 C" in passage and "?" in passage):
            errs.append("2점 규칙도출형 자료구조 누락")
        if len(tasks)>=2 and not any(k in tasks[1] for k in ("앞에서","도출한","이용")):
            errs.append("2점 적용요구 선행규칙 의존성 부족")
        if len(ans)>=2 and re.sub(r"[^가-힣A-Za-z0-9]","",ans[1]).lower() in re.sub(r"[^가-힣A-Za-z0-9]","",passage).lower():
            errs.append("2점 적용 정답 직접 노출")
        return errs
    if spec.get("mode")!="paired_fact_slot_consistency":
        return ["2점 reasoning_spec 누락"]

    passage=str(cand.get("passage","") or "")
    conditions=" ".join(map(str,cand.get("conditions",[]) or []))
    tasks=[str(x) for x in (cand.get("tasks",[]) or [])]
    material=" ".join([passage,conditions]+tasks)
    compact=re.sub(r"\s+"," ",material).strip()
    errs=[]

    # 두 익명 사례가 실제로 있어야 한다.
    if "㉠" not in material or "㉡" not in material:
        errs.append("2점 비교 사례 ㉠/㉡ 누락")
    if len(tasks)!=2:
        errs.append("2점 작성요구 2개 아님")
        return errs

    # R32 paired-slot layout contract: 두 사례 모두 ①/②가 있고, ②는 동일한 표시 사실이어야 한다.
    # 오답 사례의 수정 위치를 Python이 ②로 고정했으므로 이후 Writer가 사례를 바꿔도 여기서 차단한다.
    def _case_slots(label):
        case=_case_text(label) if "_case_text" in locals() else ""
        if not case:
            m=re.search(re.escape(label)+r"(.*?)(?=㉠|㉡|$)", passage, re.S)
            case=(m.group(1) if m else "").strip()
        m1=re.search(r"①\s*(.*?)(?=②|$)",case,re.S)
        m2=re.search(r"②\s*(.*)$",case,re.S)
        return ((m1.group(1).strip() if m1 else ""),(m2.group(1).strip() if m2 else ""))
    _a1,_a2=_case_slots("㉠"); _b1,_b2=_case_slots("㉡")
    if not all((_a1,_a2,_b1,_b2)):
        errs.append("2점 ①/② 슬롯 구조 누락")
    else:
        def _slot_comp(x): return re.sub(r"[^가-힣A-Za-z0-9]","",x).lower()
        if _slot_comp(_a2)!=_slot_comp(_b2):
            errs.append("2점 두 사례의 ② 표시사실 불일치")
        if _slot_comp(_a1)==_slot_comp(_b1):
            errs.append("2점 두 사례의 ① 구분단서 동일")
    if "②" not in tasks[1]:
        errs.append("2점 둘째 요구의 수정 슬롯 ② 미지정")

    # R33: 화면에 실제로 표시된 네 clause 자체도 완결되어야 하고, source를 장구절 복사하면 안 된다.
    _co=str(spec.get("correct_option","") or "")
    if all((_a1,_a2,_b1,_b2)):
        _source_map = ({
            "㉠①":spec.get("core_base_fact",""), "㉠②":spec.get("core_link_fact",""),
            "㉡①":spec.get("contrast_base_fact",""), "㉡②":spec.get("core_link_fact",""),
        } if _co=="㉠" else {
            "㉠①":spec.get("contrast_base_fact",""), "㉠②":spec.get("core_link_fact",""),
            "㉡①":spec.get("core_base_fact",""), "㉡②":spec.get("core_link_fact",""),
        })
        _forbidden=[spec.get("hidden_core_answer",""),spec.get("hidden_contrast_answer","")]
        for _name,_clause in (("㉠①",_a1),("㉠②",_a2),("㉡①",_b1),("㉡②",_b2)):
            _src=_source_map.get(_name,"")
            _qd=t2_clause_quality(_clause,_src,_forbidden)
            if not _qd.get("ok"):
                errs.append(f"2점 {_name} 표시문구 불합격: {_qd.get('reason','')}")

    # T2는 실제 시각자료를 사용하지 않으므로 Writer가 임의로 그림/표/그래프를 만들거나 언급할 수 없다.
    _t2_visible=" ".join([str(cand.get("intro","") or ""),passage,conditions])
    if re.search(r"(그림|도식|그래프|사진|아래 표|다음 표|표를 보고|그림을 보고)",_t2_visible):
        errs.append("2점 실제 자료 없이 시각자료 언급")

    # R32 semantic/scoring contract: 수정 대상 슬롯과 채점답이 1:1로 대응해야 한다.
    cc=spec.get("correction_contract") or {}
    expected=_norm_anchor_text(cc.get("expected_replacement_fact","")).strip()
    wrong_fact=_norm_anchor_text(cc.get("displayed_wrong_fact","")).strip()
    wrong_option=str(cc.get("wrong_option","")).strip()
    wrong_slot=str(cc.get("wrong_slot","")).strip()
    if not expected or not wrong_fact or wrong_option not in {"㉠","㉡"} or wrong_slot!="②":
        errs.append("2점 오류수정 semantic contract 누락")
    else:
        _expected_reason=_t2_complete_fact_reason(expected)
        if _expected_reason:
            errs.append("2점 수정정답 불완전: "+_expected_reason)
        answers=[_norm_anchor_text(x).strip() for x in (cand.get("answer",[]) or [])]
        if len(answers)<2 or _topic_core(answers[1])!=_topic_core(expected):
            errs.append("2점 수정대상-채점답 불일치")

    # 첫 요구는 선택/판단, 둘째는 첫 판단을 참조한 수정/적용이어야 한다.
    if not re.search(r"(고르|선택|판단|적절|타당)",tasks[0]):
        errs.append("2점 첫 요구가 자료판단이 아님")
    if not re.search(r"(선택|판단|오류|잘못|수정|바르게|고쳐|근거|적용)",tasks[1]):
        errs.append("2점 둘째 요구가 첫 판단의 후속 사고가 아님")

    # R31: 다른 사례에 수정 정답이 거의 그대로 제시되면 둘째 1점이 복사 문제가 된다.
    # 의미 보존 재표현은 AI Judge가 보되, Python은 우선 높은 어휘 중복을 deterministic하게 차단한다.
    def _case_text(label):
        m=re.search(re.escape(label)+r"(.*?)(?=㉠|㉡|$)", passage+" "+conditions, re.S)
        return (m.group(1) if m else "").strip()
    def _lex_sim(a,b):
        aa=re.sub(r"[^가-힣A-Za-z0-9]","",_norm_anchor_text(a)).lower()
        bb=re.sub(r"[^가-힣A-Za-z0-9]","",_norm_anchor_text(b)).lower()
        if not aa or not bb: return 0.0
        return SequenceMatcher(None,aa,bb).ratio()
    if expected:
        _visible=" ".join([passage,conditions])
        if _lex_sim(_visible,expected)>=0.52:
            errs.append("2점 수정정답이 자료에 직접 제시")
        else:
            # 긴 핵심구절이 어느 사례/조건에라도 그대로 있으면 복사형으로 본다.
            ec=re.sub(r"[^가-힣A-Za-z0-9]","",expected).lower()
            vc=re.sub(r"[^가-힣A-Za-z0-9]","",_visible).lower()
            if len(ec)>=12:
                n=min(18,max(12,len(ec)//3))
                if any(ec[j:j+n] in vc for j in range(0,max(1,len(ec)-n+1),max(1,n//3))):
                    errs.append("2점 수정정답 핵심구절 자료 노출")

    # 숨은 중심개념/비교개념 명칭을 자료에 직접 쓰지 않는다.
    for key in ("hidden_core_answer","hidden_contrast_answer"):
        ans=re.sub(r"[^가-힣A-Za-z0-9]","",str(spec.get(key,""))).lower()
        mc=re.sub(r"[^가-힣A-Za-z0-9]","",compact).lower()
        if len(ans)>=3 and ans in mc:
            errs.append("2점 숨은 개념명 직접 노출")
            break

    # 세 원문 사실의 긴 구절 복사를 모두 차단한다.
    mc=re.sub(r"[^가-힣A-Za-z0-9]","",compact).lower()
    for key in ("core_base_fact","core_link_fact","contrast_base_fact","contrast_link_fact"):
        raw=re.sub(r"[^가-힣A-Za-z0-9]","",str(spec.get(key,""))).lower()
        if len(raw)<14:
            continue
        n=min(22,max(14,len(raw)//3))
        for j in range(0,max(1,len(raw)-n+1),max(1,n//3)):
            frag=raw[j:j+n]
            if len(frag)>=14 and frag in mc:
                errs.append("2점 원문 장구절 직접 복사")
                return list(dict.fromkeys(errs))

    return list(dict.fromkeys(errs))

def _t2_local_item_segment(anchor, page_text, anchors):
    """
    같은 페이지 전체가 아니라 현재 anchor 항목의 로컬 블록만 support 추출에 사용한다.
    다음 sibling anchor가 시작되면 즉시 끊어 8PR→스탠딩웨이브 같은 페이지 횡단 오염을 막는다.
    """
    raw=str(page_text or "")
    if not raw:
        return raw
    lines=[x.rstrip() for x in raw.splitlines() if x.strip()]
    ac=_topic_core(anchor.get("answer",""))
    tc=_topic_core(anchor.get("topic",""))
    aid=_t2_identity_key(anchor.get("answer","")); tid=_t2_identity_key(anchor.get("topic",""))
    matches=[]
    for i,line in enumerate(lines):
        clean=re.sub(r"^[\s·•○▶□■♥ü\-–—]+","",line).strip()
        left=re.split(r"[:：]",clean,1)[0].strip()
        lid=_t2_identity_key(left)
        lc=_topic_core(line)
        # R34: short labels use exact left-label identity; do not let 5G match 2.5G or p1 match p16.
        short_identity=bool(aid and len(aid)<=5)
        if short_identity:
            if lid!=aid:
                continue
        elif not ((ac and ac in lc) or (tc and tc in lc) or (aid and lid==aid) or (tid and lid==tid)):
            continue
        cc=_topic_core(clean)
        score=0.0
        # explicit "용어 :" or bullet label is the strongest item start
        if ":" in clean or "：" in clean:
            left=re.split(r"[:：]",clean,1)[0]
            if (ac and ac in _topic_core(left)) or (tc and tc in _topic_core(left)):
                score+=6.0
        if clean.startswith(str(anchor.get("answer","")).strip()) or clean.startswith(str(anchor.get("topic","")).strip()):
            score+=4.0
        if re.match(r"^[■▶♥]",line.strip()):
            score+=2.0
        # incidental mention in a long process sentence is weaker
        if len(clean)>80:
            score-=1.0
        matches.append((score,i))
    if not matches:
        return raw
    matches.sort(key=lambda x:(x[0],x[1]),reverse=True)
    start=matches[0][1]

    # same-page sibling answer/topic tokens
    siblings=[]
    for b in anchors:
        if b is anchor:
            continue
        if str(b.get("source_name",""))!=str(anchor.get("source_name","")):
            continue
        try:
            if int(b.get("page_no",0) or 0)!=int(anchor.get("page_no",0) or 0):
                continue
        except Exception:
            continue
        for v in (b.get("answer",""),b.get("topic","")):
            c=_topic_core(v); kid=_t2_identity_key(v)
            if kid and kid not in {aid,tid}:
                siblings.append((c,kid))
    siblings=list(dict.fromkeys(siblings))

    end=min(len(lines),start+28)
    for j in range(start+1,min(len(lines),start+28)):
        lj=_topic_core(lines[j])
        # strong document section boundary
        if re.match(r"^\s*[■▶♥]",lines[j]) or re.match(r"^\s*\d+[\.\)]\s+[가-힣A-Za-z]",lines[j]):
            end=j; break
        # PDF에서 새 개념 표제가 bullet 한 줄, 설명의 ':'가 다음 줄로 분리되는 경우.
        # 예: '• 하이빔 공법(HI-BEAM공법)' 다음 줄 ': 하이빔은 ...'
        _bullet_head=re.sub(r"^\s*[·•●○□-]\s*","",lines[j]).strip()
        _next_line=lines[j+1].strip() if j+1 < len(lines) else ""
        if re.match(r"^\s*[·•●○□-]",lines[j]) and len(_bullet_head)<=55 and _next_line.startswith((":","：")):
            end=j; break
        # 짧은 독립 표제/예제 번호 뒤에 새 설명이 시작되면 현재 항목의 경계로 본다.
        _cur=lines[j].strip()
        _nxt=lines[j+1].strip() if j+1 < len(lines) else ""
        _nxt2=lines[j+2].strip() if j+2 < len(lines) else ""
        # PDF may split a new short heading across two lines, followed by its bullet/explanation.
        # Treat it as a boundary only when the two-line title is short and the following line clearly starts content.
        if (2<=len(_cur)<=16 and 2<=len(_nxt)<=20 and len((_cur+_nxt).replace(" ",""))<=28
                and re.match(r"^[·•●○□\-]",_nxt2)):
            end=j; break
        # "용어 : 설명"도 실제 sibling anchor가 그 줄의 왼쪽 표제로 시작할 때만 새 항목으로 본다.
        # ※ 노치 효과 : ... 같은 현재 개념의 보조설명을 잘못 끊지 않는다.
        if re.match(r"^[^:：]{2,40}\s*[:：]",_cur):
            _left=_topic_core(re.split(r"[:：]",re.sub(r"^[\s·•○▶□■♥ü\-–—]+","",_cur),1)[0])
            _left_id=_t2_identity_key(re.split(r"[:：]",re.sub(r"^[\s·•○▶□■♥ü\-–—]+","",_cur),1)[0])
            if _left_id and any(_left_id==kid for _c,kid in siblings):
                end=j; break
        if (len(_cur)<=24 and (
                re.match(r"^\(?예제\s*\d+\)?$",_cur)
                or re.match(r"^[①-⑳]\s*[^:：]{0,18}$",_cur)
                or (re.match(r"^[가-힣A-Za-z0-9 ()·/+-]{2,20}$",_cur) and _nxt.startswith("-")
                    and _cur not in {"정의","특징","절차","상황","규칙","장점","단점","종류","활용","원리","목적","개념","효과","유형"})
            )):
            end=j; break
        # known sibling anchor가 문장 안에 단순 포함된 경우가 아니라 새 항목의 표제로 시작할 때만 경계로 본다.
        _clean_boundary=re.sub(r"^[\s·•○▶□■♥ü\-–—]+","",lines[j]).strip()
        _boundary_left=re.split(r"[:：]",_clean_boundary,1)[0].strip()
        _boundary_id=_t2_identity_key(_boundary_left)
        _next_boundary=(lines[j+1].strip() if j+1 < len(lines) else "")
        _two_line_id=_t2_identity_key(_boundary_left+" "+_next_boundary)
        if (_boundary_id and any(_boundary_id==kid for _c,kid in siblings)) or (_two_line_id and any(_two_line_id==kid for _c,kid in siblings)):
            end=j; break
    return "\n".join(lines[start:end]).strip()

def _t2_atomic_facts(anchor, local_segment):
    """
    R33 source reflow parser.
    PDF/서브노트의 시각적 줄바꿈을 사실 경계로 오인하지 않는다.
    bullet/번호가 새 사실의 시작이고, 그 다음 줄은 새 bullet/heading이 나오기 전까지 이어 붙인다.
    """
    raw=str(local_segment or "").replace("\x01"," ").replace("\u200b"," ")
    ans=_norm_anchor_text(anchor.get("answer","")).strip()
    topic=_norm_anchor_text(anchor.get("topic","")).strip()

    physical=[re.sub(r"\s+"," ",x).strip() for x in raw.splitlines() if x.strip()]
    logical=[]
    current=""

    def is_bullet(line):
        return bool(re.match(r"^(?:[·•●○■▶□♥]|[-–—]\s+|[①-⑳]|\(?\d{1,2}\)?[.)]\s+|[가-하][.)]\s+)",line))
    def is_meta(line):
        return bool(re.search(r"(?:Made By|기술임용시험|^-\s*\d+\s*-$|^\d+\s*$)",line,re.I))
    def headingish(line):
        z=re.sub(r"^[\s■▶●♥□·•○ü\-–—]+","",line).strip()
        if z in {"정의","특징","절차","상황","규칙","장점","단점","종류","활용","원리","목적","개념","시험목적","실험순서","관련지식","개요"}:
            return True
        if len(z)<=24 and not re.search(r"[.!?:：;/]",z) and not re.search(r"(한다|된다|있다|없다|함|됨|사용|이용|발생|변화|증가|감소|생성|산출|작용|가능|필요)",z):
            # answer/topic 자체 또는 'OO의 특성' 같은 짧은 소제목
            return bool(z in {ans,topic} or re.search(r"(의\s*)?(특성|특징|종류|구성|원리|절차|장단점|장점|단점|정의)$",z))
        return False

    for line in physical:
        if is_meta(line):
            continue
        if is_bullet(line) or headingish(line):
            if current:
                logical.append(current.strip())
            current=line
        else:
            if current:
                # PDF soft-wrap continuation: 공백 하나로 이어 붙인다.
                _prev=current.rstrip(); _next=line.lstrip()
                _last=(re.findall(r"([가-힣]+)$",_prev) or [""])[-1]
                _sep="" if len(_last)==1 and _next[:1] and re.match(r"[가-힣]",_next[:1]) else " "
                current=(_prev+_sep+_next).strip()
            else:
                current=line
    if current:
        logical.append(current.strip())

    chunks=[]
    for line in logical:
        line=re.sub(r"^[\s■▶●♥□·•○ü\-–—]+","",line).strip()
        if not line or headingish(line):
            continue
        parts=re.split(r"(?<=[.!?])\s+|;\s*|(?=※)|(?=①|②|③|④|⑤|⑥|⑦)",line)
        for part in parts:
            part=part.strip(" -※")
            if not part:
                continue
            for label in (ans,topic):
                if label and part.startswith(label):
                    part=part[len(label):].lstrip(" :：-_").strip()
            part=re.sub(r"^(?:정의|특징|목적|원리)\s*[-:：]?\s*","",part).strip()
            _pc=re.sub(r"[^가-힣A-Za-z0-9]","",part)
            if (12<=len(part)<=220) or (len(_pc)>=7 and re.search(r"\d",part) and re.search(r"(시작|검사|반복|속도|비율|범위|이상|이하|증가|감소)",part)):
                chunks.append(part)

    # R34: split compound facts only inside the current anchor's own local block.
    # This recovers safe series such as p1/p2 and 4G/5G without borrowing sibling sentences.
    expanded=[]
    relsig=re.compile(r"(시작|검사|반복|증가|감소|가능|사용|이용|전송|속도|범위|비율|과정|단계|기능|장점|단점|효과|조건|경우|때|따라|발생|변화|작용|제어|처리|저장|연결|통합)")
    for c0 in chunks:
        parts=[z.strip() for z in re.split(r"\s*[,，]\s*",c0) if z.strip()]
        if len(parts)>=2:
            good=[]
            for z in parts:
                zc=re.sub(r"[^가-힣A-Za-z0-9]","",z)
                if len(zc)>=7 and (relsig.search(z) or re.search(r"\d",z)):
                    good.append(z)
            if len(good)>=2:
                expanded.extend(good)
                # keep the full clause too; pair selection may prefer it when split pieces are weak.
                expanded.append(c0)
                continue
        expanded.append(c0)
    chunks=expanded

    facts=[]; seen=[]
    for c in chunks:
        # Normalize common source fragments into complete factual clauses without adding new facts.
        c=re.sub(r"^(\d+번\s*비트)\s*를\s*시작으로$",r"검사는 \1에서 시작한다",c).strip()
        c=re.sub(r"\s+즉$","",c).strip()
        if re.search(r"(_최|★\d|기출|서브노트|^\d+[AB]\()",c):
            c=re.sub(r"_최.*?(?=정의|특징|-|$)","",c)
            c=re.sub(r"★\S+","",c).strip()
        _shortc=re.sub(r"[^가-힣A-Za-z0-9]","",c)
        if len(c)<12 and not (len(_shortc)>=7 and re.search(r"\d",c) and re.search(r"(시작|검사|반복|속도|비율|범위|이상|이하|증가|감소)",c)):
            continue
        cc=_shortc.lower()
        if not cc:
            continue
        if any(cc==x or (len(cc)>=18 and (cc in x or x in cc)) for x in seen):
            continue
        seen.append(cc); facts.append(c)
    return facts[:10]


def _t2_choose_fact_pair(anchor, facts):
    """R32: local block의 첫 두 줄이 아니라 출제 가능한 두 source fact를 전역 규칙으로 고른다."""
    label=_norm_anchor_text(anchor.get("answer","")).strip()
    def compact(x): return re.sub(r"[^가-힣A-Za-z0-9]","",_norm_anchor_text(x)).lower()
    def contains_label(x):
        l=compact(label); t=compact(x)
        return len(l)>=3 and l in t
    def meaningful(x):
        stop={"정의","특징","방법","기법","경우","대한","위한","하는","한다","된다","있다","사용","이용","것","수","및","또는","때","원리","개념"}
        return {z for z in re.findall(r"[가-힣A-Za-z0-9]{2,}",x) if z not in stop}
    signal=re.compile(r"(때|경우|조건|따라|때문|원인|결과|효과|장점|단점|목적|과정|단계|활용|적용|증가|감소|향상|개선|방지|필요|가능|도움|유용|선정|분석|발전|발견|비교|판단|변화|작용|촉진|억제|반복|집단|개인|시간)")
    usable=[]
    for idx,f0 in enumerate(facts or []):
        f=_norm_anchor_text(f0).strip()
        if not (12<=len(f)<=180): continue
        _fc=compact(f)
        if len(meaningful(f))<2 and len(_fc)<14: continue
        if contains_label(f): continue
        # 괄호 속 영문명/약어, 목록 표지만으로 된 조각은 자료판단 사실이 아니다.
        if re.fullmatch(r"[\[(（]?[A-Za-z][A-Za-z0-9 ,./&+-]{1,50}[\])）]?",f): continue
        if re.match(r"^[①-⑳ⓐ-ⓩ]\s*[^가-힣]{0,30}$",f): continue
        score=0.0
        if 22<=len(f)<=120: score+=1.0
        if signal.search(f): score+=1.5
        if re.search(r"(정의|의미|라고도|일컫|말한다)$",f): score-=1.0
        usable.append((idx,score,f))
    if len(usable)<2: return None
    best=None
    for i in range(len(usable)):
        for j in range(i+1,len(usable)):
            i0,si,a=usable[i]; j0,sj,b=usable[j]
            ac=compact(a); bc=compact(b)
            sim=SequenceMatcher(None,ac,bc).ratio() if ac and bc else 1.0
            if sim>=0.80: continue
            ta=meaningful(a); tb=meaningful(b)
            overlap=len(ta&tb)/max(1,min(len(ta),len(tb)))
            if overlap>=0.88: continue
            # 둘 다 단순 정의 문장인 조합보다 조건/효과/과정 등이 섞인 조합을 우선한다.
            relational=(1 if signal.search(a) else 0)+(1 if signal.search(b) else 0)
            pair_score=si+sj+relational*0.8-min(1.2,overlap)
            # 너무 멀리 떨어진 subsection 조합보다 로컬 인접 사실을 선호한다.
            pair_score-=min(1.0,abs(j0-i0)*0.08)
            row=(pair_score,-j0,i0,a,b)
            if best is None or row>best: best=row
    return (best[3],best[4]) if best else None



def _t2_candidate_fact_pairs(anchor, facts, anchors):
    """Return multiple internally complete fact pairs.
    Discrimination is evaluated jointly after the comparison item is known; structural series often
    contain intentionally parallel/generic wording (p1/p2, 4G/5G) that is not discriminative alone.
    """
    vals=[]
    fs=[_norm_anchor_text(x).strip() for x in (facts or []) if _norm_anchor_text(x).strip()]
    for i in range(len(fs)):
        for j in range(i+1,len(fs)):
            pair=(fs[i],fs[j])
            bad=False
            for x in pair:
                if _t2_complete_fact_reason(x): bad=True; break
                ok,_why=_t2_fact_scope_compatible(anchor,x)
                if not ok: bad=True; break
            if bad: continue
            sim=_t2_axis_similarity(pair[0],pair[1])
            if sim>=0.90: continue
            score=2.0
            if re.search(r"\d",pair[0]): score+=0.25
            if re.search(r"\d",pair[1]): score+=0.25
            vals.append((score,pair))
    vals.sort(key=lambda x:x[0],reverse=True)
    return [p for _,p in vals[:16]]


def _t2_complete_fact_reason(text):
    """R33: 모든 T2 source fact에 공통 적용하는 완결성 gate."""
    x=_norm_anchor_text(text).strip()
    c=re.sub(r"[^가-힣A-Za-z0-9]","",x)
    # R36: 짧더라도 '대상/속성 + 기술 술어(변화·기능·수치·상태)'가 있으면 독립 사실이다.
    # 예: '데이터 전송 속도 증가', '전송 속도 1Gbps 이상', '압력이 감소'.
    # 반대로 '다양한 부가 서비스' 같은 명사구만 있는 조각은 계속 차단한다.
    _short_pred=bool(re.search(r"(증가|감소|향상|저하|가능|불가능|사용|이용|전송|통합|결합|저장|발생|유지|제어|검사|시작|반복|변화|작용|생성|산출|차지|생산|구성|연결|분류|평가|처리|된다|한다|이다|있다|없다|이상|이하|초과|미만)",x))
    _short_value=bool(re.search(r"\d+(?:[.,]\d+)?\s*(?:%|[A-Za-z가-힣/²³]+)?",x))
    _short_terms=[t for t in re.findall(r"[가-힣A-Za-z0-9.%/]+",x) if len(re.sub(r"[^가-힣A-Za-z0-9]","",t))>=2]
    _short_complete=(len(c)>=7 and len(_short_terms)>=2 and (_short_pred or _short_value))
    if len(c)<14 and not _short_complete:
        return "사실 문구가 너무 짧음"
    if len(x)>180:
        return "사실 문구가 지나치게 김"
    if re.search(r"[\ue000-\uf8ff]",x):
        return "수식/PDF 깨짐 문자가 포함됨"
    # 앞 문맥이 없으면 뜻이 성립하지 않는 접속/대용 표현, 잘린 서술어를 차단한다.
    if re.match(r"^(?:및|또는|그리고|그러나|따라서|그러므로|이때|이 경우|이를|이것을|그것을|내거나|하거나|하여|해서|하며|하면서|되며|있으며|없으며|때문에|따라|반면에|한편)",x):
        return "앞 문맥에 의존하는 부분 문구"
    if re.search(r"(?:,|;|:|·|/|그리고|또는|및)\s*$",x):
        return "끝이 열린 부분 문구"
    # 조사·관형형·연결형으로 끝나는 조각은 독립 채점답/자료문구로 쓰지 않는다.
    # Do not use single Korean particles as suffix tests: words such as '증가' cause false positives.
    # Reject only clearly connective/attributive endings that require following context.
    if re.search(r"(?:위한|통한|하는|되는|있는|없는|같은|따른|관한|비롯한|의한|때문인|경우의)\s*[.!?]?$",x):
        return "완결되지 않은 관형형 종결"
    # 충분한 길이와 동작/관계 표현이 있으면 완결 명사구도 허용한다(예: '... 아이디어를 생성').
    relational=bool(re.search(r"(한다|된다|이다|있다|없다|함|됨|생성|산출|발휘|도움|분석|판단|선정|발견|비교|사용|이용|적용|촉진|억제|향상|증가|감소|변화|작용|해결|제시|기록|연결|분류|평가|표현|시각화|반복|예상|조사|생산|구성|유지|구별|설명|목적|원리|특징|장점|단점|방법|기법|과정|현상|방식|구조|상태|관계|조건|기준|단계|활동|기능)",x))
    technical_value=bool(re.search(r"\d|증가|감소|가능|불가능|이상|이하|초과|미만|시작|반복|전송|통합|결합|저장|발생|유지|제어|검사|차지|생산",x))
    # R36: 짧은 기술 사실은 위의 subject/value/predicate gate를 통과했으면 정보량 부족으로 이중 탈락시키지 않는다.
    if not relational and not technical_value and not _short_complete and len(c)<24:
        return "독립 사실로서 정보량 부족"
    return ""

def _t2_char_features(text,n=4):
    c=re.sub(r"[^가-힣A-Za-z0-9]","",_norm_anchor_text(text)).lower()
    if len(c)<n:
        return {c} if c else set()
    return {c[i:i+n] for i in range(len(c)-n+1)}

def _t2_fact_discrimination(fact, own_anchor, anchors):
    """R33: ①이 여러 개념에 두루 적용되는 일반론인지 전역 anchor universe에서 추정한다."""
    ff=_t2_char_features(fact,4)
    if len(ff)<5:
        return {"ok":False,"reason":"식별 특징 부족","max_other":1.0,"unique_ratio":0.0}
    own_key=(_topic_core(own_anchor.get("answer","")),str(own_anchor.get("source_name","")),int(own_anchor.get("page_no",0) or 0))
    other_sets=[]
    feature_df={g:0 for g in ff}
    for b in anchors:
        key=(_topic_core(b.get("answer","")),str(b.get("source_name","")),int(b.get("page_no",0) or 0))
        if key==own_key:
            continue
        bt=_t2_char_features(" ".join(map(str,[b.get("topic",""),b.get("answer",""),b.get("evidence","")])),4)
        if not bt:
            continue
        inter=len(ff & bt)
        union=len(ff | bt) or 1
        sim=inter/union
        other_sets.append(sim)
        for g in ff & bt:
            feature_df[g]+=1
    max_other=max(other_sets) if other_sets else 0.0
    unique=sum(1 for g,v in feature_df.items() if v==0)/max(1,len(ff))
    # 다른 anchor와 문구 특징이 지나치게 겹치거나, 고유 특징이 거의 없으면 ①로 쓰지 않는다.
    ok=(max_other<0.48 and unique>=0.28)
    reason="" if ok else ("다른 anchor에도 적용 가능한 일반적 사실" if max_other>=0.48 else "개념 식별 고유단서 부족")
    return {"ok":ok,"reason":reason,"max_other":round(max_other,3),"unique_ratio":round(unique,3)}

def _t2_fact_scope_compatible(anchor, fact):
    """R33: 상위 기법명과 특정 하위 절차/부분만 억지로 연결되는 범위 혼합을 보수적으로 차단."""
    ans=_norm_anchor_text(anchor.get("answer","")).strip()
    f=_norm_anchor_text(fact).strip()
    # source 자체가 'A·B', 'A/B', 'A 및 B' 같은 복합 상위 명칭인데 fact가 그중 한 하위명만 명시하면 범위가 흔들릴 수 있다.
    parts=[z.strip() for z in re.split(r"[·/&]|\s+및\s+",ans) if z.strip()]
    if len(parts)>=2:
        hits=sum(1 for z in parts if len(re.sub(r"\W","",z))>=2 and re.sub(r"\W","",z).lower() in re.sub(r"\W","",f).lower())
        if hits==1:
            return False,"복합 상위개념과 단일 하위범위 혼합"
    return True,""

def _t2_pair_global_quality(anchor, pair, anchors):
    """R33 DB-universe 공통 gate: 완결성·식별력·범위 일관성을 한 번에 평가."""
    if not pair or len(pair)!=2:
        return {"ok":False,"reason":"사실쌍 없음"}
    a,b=map(lambda x:_norm_anchor_text(x).strip(),pair)
    for idx,x in enumerate((a,b),1):
        why=_t2_complete_fact_reason(x)
        if why:
            return {"ok":False,"reason":f"사실{idx} {why}"}
        ok,why2=_t2_fact_scope_compatible(anchor,x)
        if not ok:
            return {"ok":False,"reason":f"사실{idx} {why2}"}
    disc=_t2_fact_discrimination(a,anchor,anchors)
    if not disc.get("ok"):
        return {"ok":False,"reason":"사실1 "+disc.get("reason","식별력 부족"),"discrimination":disc}
    return {"ok":True,"reason":"global_t2_universe_gate_pass","discrimination":disc}

def _t2_clause_copy_score(clause, source_fact):
    a=re.sub(r"[^가-힣A-Za-z0-9]","",_norm_anchor_text(clause)).lower()
    b=re.sub(r"[^가-힣A-Za-z0-9]","",_norm_anchor_text(source_fact)).lower()
    if not a or not b:
        return 1.0
    seq=SequenceMatcher(None,a,b).ratio()
    fa=_t2_char_features(a,4); fb=_t2_char_features(b,4)
    sh=len(fa&fb)/max(1,min(len(fa),len(fb)))
    return max(seq,sh)



def _t2_equal_parameter_rule(anchor):
    """Detect a source-owned numeric procedure rule without concept hardcoding.
    Requires one parameter to control start position, inspected count and skipped count.
    Example shape: n번째에서 시작, n개 검사, n개 건너뛰기.
    """
    text=_norm_anchor_text(anchor.get("evidence", ""))
    label=_t2_identity_key(anchor.get("answer", ""))
    sig=_t2_series_signature(anchor.get("answer", ""))
    if not sig or "#" not in sig:
        return None
    lm=re.search(r"(\d+(?:\.\d+)?)",label)
    sm=re.search(r"(\d+)\s*번[^,;]{0,40}시작",text)
    cm=re.search(r"(\d+)\s*개\s*검사",text)
    km=re.search(r"(\d+)\s*개\s*건너",text)
    if not (lm and sm and cm and km):
        return None
    try:
        lv=float(lm.group(1)); sv=int(sm.group(1)); cv=int(cm.group(1)); kv=int(km.group(1))
    except Exception:
        return None
    if abs(lv-round(lv))>1e-9 or int(round(lv))!=sv or not (sv==cv==kv):
        return None
    return {"signature":sig,"n":sv,"source":text}

def _t2_numeric_rule_application_candidate(anchors, pd):
    """R38 capability-first T2.
    A relationship question is allowed only when >=3 source items in the same structural
    series independently instantiate the same numeric rule. Two items establish the rule;
    a third item is reserved for application. This prevents definition comparison/copy tasks.
    """
    groups={}
    for a0 in anchors:
        a=_two_point_core_anchor_clean(a0)
        if not _two_point_core_target_ok(a):
            continue
        r=_t2_equal_parameter_rule(a)
        if not r:
            continue
        groups.setdefault(r["signature"],[]).append((r["n"],copy.deepcopy(a),r))
    candidates=[]
    for sig,rows in groups.items():
        uniq={}
        for n,a,r in rows:
            uniq.setdefault(n,(n,a,r))
        rows=sorted(uniq.values(),key=lambda z:z[0])
        if len(rows)<3:
            continue
        # Use separated examples and a later target when possible; this makes rule induction
        # and application distinct scoring actions rather than a one-line lookup.
        ex1=rows[0]; ex2=rows[1]; target=rows[-1]
        if target[0] in {ex1[0],ex2[0]}:
            continue
        n1,a1,r1=ex1; n2,a2,r2=ex2; nt,at,rt=target
        rule_answer="한 주기의 비트 수는 시작 비트 번호의 2배이다."
        target_answer=f"{2*nt}개"
        spec={
            "mode":"numeric_series_rule_application",
            "series_signature":sig,
            "example_values":[n1,n2],
            "target_value":nt,
            "example_sources":[r1["source"],r2["source"]],
            "target_source":rt["source"],
            "rule_answer":rule_answer,
            "target_answer":target_answer,
            "target_rule_detail":f"{nt}개 검사 후 {nt}개 건너뛰므로 한 주기는 {2*nt}개 비트이다.",
            "source_items":[
                {"answer":a1.get("answer",""),"source_name":a1.get("source_name",""),"page_no":a1.get("page_no",0),"evidence":r1["source"]},
                {"answer":a2.get("answer",""),"source_name":a2.get("source_name",""),"page_no":a2.get("page_no",0),"evidence":r2["source"]},
                {"answer":at.get("answer",""),"source_name":at.get("source_name",""),"page_no":at.get("page_no",0),"evidence":rt["source"]},
            ],
            "minimum_inference_steps":2,
        }
        support=copy.deepcopy(a1)
        support["topic"]=f"{_norm_anchor_text(a1.get('topic','')).strip()} · 규칙적용"
        support["answer"]=target_answer
        support["evidence"]=rt["source"]
        support["derived_support"]=True
        support["reasoning_spec"]=copy.deepcopy(spec)
        support["contrast_anchor"]=copy.deepcopy(at)
        score=float(a1.get("exam_value_score") or 0)+float(a1.get("core_exam_score") or 0)*0.15+len(rows)
        candidates.append((score,[a1,support],spec,a2,at))
    if not candidates:
        pd["final_reason"]="no_numeric_rule_application_candidates"
        pd["candidate_accept"]=0
        return None
    candidates.sort(key=lambda x:x[0],reverse=True)
    score,bundle,spec,a2,at=candidates[0]
    pd["candidate_accept"]=len(candidates)
    pd["two_point_rule_application_candidates"]=len(candidates)
    pd["final_reason"]="numeric_rule_application_ready"
    core=bundle[0]
    meta={
        "master_concept":_norm_anchor_text(core.get("topic","")),
        "relation":"여러 source item에서 공통 수치 규칙을 도출한 뒤 새 source item에 적용",
        "thinking_types":["자료비교","규칙도출","적용계산"],
        "exam_skeleton":"두 사례의 공통 수치관계 도출 → 관계식 일반화 → 세 번째 사례에 적용",
        "scoring_plan":[
            {"points":1,"action":"공통 규칙 도출","answer":spec["rule_answer"]},
            {"points":1,"action":"도출한 규칙을 목표 사례에 적용","answer":spec["target_answer"]},
        ],
        "material_limits":_compact_material_limits(2),
        "natural_unit_score":None,
        "two_point_label_policy":"RULE-APPLICATION: source에 반복 확인되는 규칙을 먼저 도출하고 다음 source item에 적용한다.",
        "reasoning_spec":copy.deepcopy(spec),
        "contrast_context":spec["target_source"],
        "contrast_topic":_norm_anchor_text(at.get("topic","")),
        "contrast_answer":_norm_anchor_text(at.get("answer","")),
        "contrast_source_name":str(at.get("source_name","")),
        "contrast_page_no":at.get("page_no"),
        "correct_option":"",
        "hidden_core_answer":_norm_anchor_text(core.get("answer","")),
        "core_exam_profile":[{"topic":core.get("topic",""),"score":core.get("core_exam_score",0),"tier":core.get("core_exam_tier","SUPPORT"),"breakdown":core.get("core_exam_breakdown",{})}],
        "quality_directive":"자료 문구 찾기/복사형 금지. 두 예에서 공통 규칙을 도출해야 첫 1점을 얻고, 그 규칙을 세 번째 source item에 적용해야 둘째 1점을 얻도록 Python이 고정한다.",
        "selector_reason":f"R38 numeric series rule-application score={score:.2f}",
        "relation_score":round(score,2),
        "selection_mode":"python_exam_value_t2_reasoning_matrix",
        "source_policy":"subnote_only_for_answer_content",
        "score_pipeline_diagnostic":copy.deepcopy(pd),
        "score_diagnostic":{"selected_mode":"numeric_series_rule_application","series_signature":spec["series_signature"],"example_values":spec["example_values"],"target_value":spec["target_value"]},
    }
    return bundle,meta

def _select_two_point_one_anchor(anchors, page_map, raw_page_map, pd, rng, pattern_id):
    """
    T2 전용 selector.
    DB 정답 anchor는 1개만 선택하고, 두 번째 1점은 같은 anchor/page의 근거·오류수정·비교·적용으로 만든다.
    """
    # R38: old paired-slot lookup/correction is disabled.  Only a rule that is
    # independently instantiated by at least three source items may become a relational T2.
    _rule=_t2_numeric_rule_application_candidate(anchors,pd)
    if _rule is not None:
        return _rule
    return [],{"score_pipeline_diagnostic":pd}

    ranked=[]  # legacy paired code retained below for compatibility, intentionally unreachable
    for a0 in anchors:
        a=_two_point_core_anchor_clean(a0)
        if not _two_point_core_target_ok(a):
            continue
        peripheral=int((a.get("core_exam_breakdown") or {}).get("peripherality",0) or 0)
        if peripheral>=4:
            pd["support_only_reject"]+=1
            continue

        _page_key=(str(a.get("source_name","")),int(a.get("page_no",0) or 0))
        _whole_page=raw_page_map.get(_page_key) or page_map.get(_page_key,"")
        page_text=_t2_local_item_segment(a,_whole_page,anchors)
        atomic_facts=_t2_atomic_facts(a,page_text)

        support=None
        core_pairs=_t2_candidate_fact_pairs(a,atomic_facts,anchors)
        if not core_pairs:
            pd.setdefault("two_point_global_pair_reject",0)
            pd["two_point_global_pair_reject"]+=1
            continue

        chosen_core_pair=None
        contrast_anchor=None
        contrast_facts=None
        reasoning_spec=None
        joint_quality=None

        # R34 JOINT SELECTION: do not pick A-pair and B-pair independently.
        # Search all valid pair combinations and require both slots to share a comparison axis.
        for _ca in _two_point_contrast_anchors(a,anchors):
            _ck=(str(_ca.get("source_name","")),int(_ca.get("page_no",0) or 0))
            _cpage=raw_page_map.get(_ck) or page_map.get(_ck,"")
            _clocal=_t2_local_item_segment(_ca,_cpage,anchors)
            _cfacts=_t2_atomic_facts(_ca,_clocal)
            _contrast_pairs=_t2_candidate_fact_pairs(_ca,_cfacts,anchors)
            if not _contrast_pairs:
                continue
            best_local=None
            for _ap in core_pairs:
                for _bp in _contrast_pairs:
                    # R35: paired-slot is a capability, not a generic similarity fallback.
                    # Only explicit structural series (4G/5G, p1/p2, numbered generations/stages)
                    # may enter this structure.  Non-series pairs are never rescued by page proximity
                    # or lexical similarity.
                    if not _t2_structural_series(a.get("answer",""), _ca.get("answer","")):
                        continue
                    jq=_t2_joint_pair_quality(a,_ap,_ca,_bp)
                    if not jq.get("ok"):
                        continue
                    jq["capability"]="numeric_series_rule_application"
                    # Build support only after the joint axis has been proven.
                    _support=copy.deepcopy(a)
                    _support["topic"]=f"{_norm_anchor_text(a.get('topic','')).strip()} · 연결사실"
                    _support["answer"]=_ap[1]
                    _support["evidence"]=_ap[1]
                    _support["derived_support"]=True
                    _trial_a=copy.deepcopy(a); _trial_a["reasoning_base_fact"]=_ap[0]
                    _trial=_t2_reasoning_spec(_trial_a,_support,_ca,list(_bp),rng)
                    if not _trial:
                        continue
                    axis=float(jq.get("slot1_axis",0))+float(jq.get("slot2_axis",0))+(0.8 if jq.get("series") else 0.0)
                    cand=(axis,_ap,_bp,_trial,jq,_support,_trial_a)
                    if best_local is None or cand[0]>best_local[0]:
                        best_local=cand
            if best_local is not None:
                _,chosen_core_pair,contrast_facts,reasoning_spec,joint_quality,support,a=best_local
                contrast_anchor=copy.deepcopy(_ca)
                break

        if not contrast_anchor or not chosen_core_pair:
            pd.setdefault("two_point_no_contrast_pair_reject",0)
            pd["two_point_no_contrast_pair_reject"]+=1
            continue

        a["reasoning_base_fact"]=chosen_core_pair[0]
        a["t2_joint_quality"]=copy.deepcopy(joint_quality or {})
        support["support_quality"]={"ok":True,"score":4.6,"reason":"r34_joint_axis_ready","action":"keep",
                                    "atomic_fact_count":len(atomic_facts),"joint_quality":copy.deepcopy(joint_quality or {})}
        support["contrast_anchor"]=copy.deepcopy(contrast_anchor)
        support["contrast_facts"]=copy.deepcopy(contrast_facts)
        support["reasoning_spec"]=copy.deepcopy(reasoning_spec)
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
        # R25: 둘째 1점이 첫 명칭 식별과 실제로 구별되는 정도를 selector 점수에 직접 반영.
        score += float((support.get("support_quality") or {}).get("score",0.0))*1.8
        if support.get("contrast_anchor"):
            score += 2.0
        if support.get("reasoning_spec"):
            score += 3.0

        ranked.append((score,[a,support]))

    if not ranked:
        pd["final_reason"]="no_single_anchor_candidates"
        return [],{"score_pipeline_diagnostic":pd}

    ranked.sort(key=lambda x:x[0],reverse=True)

    uniq=[]
    seen=set()
    for score,bundle in ranked:
        k=_t2_identity_key(bundle[0].get("answer",""))
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
                    "support_quality":copy.deepcopy(bundle0[1].get("support_quality",{})),
                },
            ],
        }

    leaderboard=[cand_diag(sc,b,idx+1) for idx,(sc,b) in enumerate(uniq)]
    # R25: T2는 Python prefilter 이후 가장 강한 후보를 우선한다.
    # 임의 top2 선택은 쉬운 후보를 다시 끌어올릴 수 있으므로 제거한다.
    score,bundle=uniq[0]
    selected_rank=1
    core_anchor=bundle[0]
    _contrast_anchor=copy.deepcopy(bundle[1].get("contrast_anchor") or {})
    _reasoning_spec=copy.deepcopy(bundle[1].get("reasoning_spec") or {})
    _correct_option=str(_reasoning_spec.get("correct_option","㉠"))

    relation_meta={
        "master_concept":_norm_anchor_text(core_anchor.get("topic","")),
        "relation":"서로 다른 두 source item의 사실쌍을 비교해 일관된 사례를 판단한 뒤, 다른 사례의 ②를 그 사례의 ①과 일관되도록 수정",
        "thinking_types":["자료비교","관계판단","오류수정"],
        "exam_skeleton":"익명 사례 ㉠/㉡의 ①·② 관계 비교 → 일관된 사례 선택 → 다른 사례의 ② 수정",
        "scoring_plan":[
            {"points":1,"action":"㉠/㉡ 중 내부적으로 일관된 사례 판단","answer":_correct_option},
            {"points":1,"action":"오답 사례의 ② 수정","answer":_norm_anchor_text((_reasoning_spec.get("correction_contract") or {}).get("expected_replacement_fact", bundle[1].get("answer",""))).strip()},
        ],
        "material_limits":_compact_material_limits(2),
        "natural_unit_score":None,
        "two_point_label_policy":"PAIRED-SLOT: 두 source item의 사실쌍 관계를 판단하고, 선택하지 않은 사례의 ②를 그 사례의 ①에 맞게 수정한다.",
        "reasoning_spec":_reasoning_spec,
        "contrast_context":_norm_anchor_text(_contrast_anchor.get("evidence","")).strip(),
        "contrast_topic":_norm_anchor_text(_contrast_anchor.get("topic","")).strip(),
        "contrast_answer":_norm_anchor_text(_contrast_anchor.get("answer","")).strip(),
        "contrast_source_name":str(_contrast_anchor.get("source_name","")),
        "contrast_page_no":_contrast_anchor.get("page_no"),
        "correct_option":_correct_option,
        "hidden_core_answer":_norm_anchor_text(core_anchor.get("answer","")).strip(),
        "core_exam_profile":[{
            "topic":core_anchor.get("topic",""),
            "score":core_anchor.get("core_exam_score",0),
            "tier":core_anchor.get("core_exam_tier","SUPPORT"),
            "breakdown":core_anchor.get("core_exam_breakdown",{}),
        }],
        "quality_directive":(
            "2점은 PAIRED-SLOT 구조다. 첫 1점은 개념명 회상이 아니라 ㉠/㉡ 두 익명 사례에서 ①과 ②가 "
            "같은 source item에 속하는지를 판단하게 한다. 정답 사례는 core_base_fact+core_link_fact, "
            "오답 사례는 contrast_base_fact+core_link_fact로 Python이 조립한다. 둘째 1점은 첫 판단을 사용해 "
            "선택하지 않은 사례의 ②를 contrast_link_fact에 맞게 수정한다. contrast_link_fact는 화면 자료에 "
            "미리 제시하지 않는다. 네 source fact는 원문을 그대로 복사하지 말고 의미를 보존해 재표현한다. "
            "hidden core/contrast 개념명은 passage, conditions, tasks에 직접 쓰지 않는다."
        ),
        "selector_reason":f"Python T2 reasoning-matrix score={score:.2f}",
        "relation_score":round(score,2),
        "selection_mode":"python_exam_value_t2_reasoning_matrix",
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
            "note":"T2는 Python이 base+linked / base+distractor의 2단계 판단논리를 먼저 고정하고 AI는 표현만 담당.",
        },
    }
    return bundle,relation_meta

def _four_point_core_target_ok(anchor):
    """4점 single-concept fallback용 일반 중심개념 gate.
    T2의 '짧은 명칭 금지'는 적용하지 않는다. 짧은 약어도 실제 핵심개념일 수 있다.
    """
    ans=_norm_anchor_text(anchor.get("answer","")).strip()
    topic=_norm_anchor_text(anchor.get("topic","")).strip()
    if not ans or _heading_like(ans) or _heading_like(topic):
        return False
    generic={"원료","용도","장점","단점","종류","특징","절차","정의","분류","기준","역할","사람","과정","방법","효과","목적","원인","이유","결과","조건","현상","재료","정보"}
    if ans.replace(" ","") in {x.replace(" ","") for x in generic}:
        return False
    if re.search(r"(기출|서브노트|^\(?예제|^[가나다라마바사아자차카타파하]\)$)",ans,re.I):
        return False
    if ans.count("(")!=ans.count(")"):
        return False
    return True

def _t4_candidate_prejudge(chosen, pattern_id):
    """R39: Writer/API 전에 4점 후보의 구조적 저난도 위험을 제거한다.
    특정 개념이 아니라 answer/evidence의 형태와 패턴 요구만 본다.
    """
    pid=str(pattern_id or '').upper()
    rows=list(chosen or [])
    if not rows:
        return False,'empty'
    def _raw_label(x):
        a=_norm_anchor_text(x.get("answer","")).strip()
        if not a: return True
        if re.search(r"[=/%℃°]",a) or re.search(r"\d+(?:\.\d+)?\s*(?:Pa|MPa|kPa|N|kN|V|A|W|Hz|byte|bit|mm|cm|m|kg|s|Gbps|Mbps)",a,re.I):
            return False
        if len(a)>22 or re.search(r"(때문|따라|증가|감소|변화|관계|오류|영향|과정|작용|이용|조건|수정|검사|반복|전송|한다$|된다$|있다$)",a):
            return False
        return True
    simple=sum(1 for x in rows if _raw_label(x))
    evs=[_norm_anchor_text(x.get('evidence','')).strip() for x in rows]
    relation_hits=sum(1 for ev in evs if re.search(r'(때문|따라|경우|하면|결과|영향|변화|증가|감소|과정|순서|오류|수정|계산|비교|관계|비례|반비례)',ev))
    # DATA112/112가 단순 label 2~3개 + 정의문 대응이라면 Writer가 사고사슬을 꾸며도 본질은 회상형이다.
    if pid in {'T4_DATA112','T4_112'} and simple>=2 and relation_hits<=2:
        return False,'t4_definition_lookup_risk'
    # ERR22는 두 정답이 모두 단순 명칭이면 '명칭+오류수정' 계약이 깨지기 쉽고 실제 수정값이 고정되지 않는다.
    if pid=='T4_ERR22' and simple>=2:
        return False,'t4_err22_label_only_risk'
    return True,''

def _select_four_point_single_anchor(anchors, page_map, raw_page_map, pd, rng, pattern_id, need):
    """Multi-anchor chain이 없는 영역에서도 통용되는 4점 fallback.
    하나의 검증된 핵심 anchor의 local source block 안에서 서로 다른 2~3개 사실을
    Python이 고정한다. 특정 개념명/영역 하드코딩은 사용하지 않는다.
    """
    ranked=[]
    for a0 in anchors:
        a=_two_point_core_anchor_clean(a0)
        if not _four_point_core_target_ok(a):
            continue
        # 4점 중심개념은 문장조각/복합 나열이 아니라 독립적으로 지칭 가능한 개념이어야 한다.
        _core_ans=_norm_anchor_text(a.get("answer","")).strip()
        if (len(_core_ans)>34 or _core_ans.count(" - ")>=2
                or (re.search(r"\d\s*$",_core_ans) and len(_core_ans)>20)
                or re.search(r"(위함|하기\s*위|때문|경우|한다$|된다$|있다$|없다$)",_core_ans)):
            continue
        if str(a.get("core_exam_tier") or "SUPPORT")=="SUPPORT" and int((a.get("core_exam_breakdown") or {}).get("peripherality",0) or 0)>=4:
            continue
        key=(str(a.get("source_name","")),int(a.get("page_no",0) or 0))
        whole=raw_page_map.get(key) or page_map.get(key,"")
        seg=_t2_local_item_segment(a,whole,anchors)
        facts=_t2_atomic_facts(a,seg)
        min_facts=2
        if len(facts)<min_facts:
            continue
        chosen_facts=facts[:2] if need>=2 else facts[:need]
        # 같은 사실의 표현 반복은 4점 채점요소로 인정하지 않는다.
        bad=False
        for i in range(len(chosen_facts)):
            for j in range(i+1,len(chosen_facts)):
                x=re.sub(r"[^가-힣A-Za-z0-9]","",chosen_facts[i]).lower()
                y=re.sub(r"[^가-힣A-Za-z0-9]","",chosen_facts[j]).lower()
                if not x or not y or SequenceMatcher(None,x,y).ratio()>=0.78:
                    bad=True; break
            if bad: break
        if bad:
            continue
        derived=[]
        if need==3:
            # 1+1+2: 중심개념 1개 + 같은 local block의 서로 다른 source fact 2개.
            # 세 개념을 억지로 연결하지 않고 하나의 개념 안에서 사고를 확장한다.
            derived.append(copy.deepcopy(a))
        for idx0,fact in enumerate(chosen_facts):
            z=copy.deepcopy(a)
            z["topic"]=f"{_norm_anchor_text(a.get('topic','')).strip()} · 근거{idx0+1}"
            z["answer"]=fact
            z["evidence"]=fact
            z["derived_support"]=True
            derived.append(z)
        derived=derived[:need]
        # R40: selector acceptance itself must guarantee that every fixed scoring fact
        # is traceable to the exact local source block used to build this candidate.
        # This makes the API-free universe gate exercise the same grounding contract
        # as final question validation.
        if any(not source_evidence_grounded(seg, d.get("evidence","")) for d in derived):
            pd.setdefault("four_point_source_ground_reject",0)
            pd["four_point_source_ground_reject"]+=1
            continue
        _pre_ok,_pre_reason=_t4_candidate_prejudge(derived,pattern_id)
        if not _pre_ok:
            pd.setdefault("four_point_prejudge_reject",0)
            pd["four_point_prejudge_reject"]+=1
            pd.setdefault("four_point_prejudge_examples",[])
            if len(pd["four_point_prejudge_examples"])<5:
                pd["four_point_prejudge_examples"].append({"topic":a.get("topic",""),"reason":_pre_reason})
            continue
        _fact_text=" ".join(chosen_facts)
        _reasoning_signals=len(re.findall(r"(때문|따라|경우|하면|위해|결과|발생|증가|감소|분리|결합|촉진|금지|이용|작용|과정|처리|조건|변화|대체|해결)",_fact_text))
        score=(float(a.get("importance_score") or 0)*1.05
               +float(a.get("exam_value_score") or 0)*1.50
               +float(a.get("core_exam_score") or 0)*0.35
               +min(4.0,len(seg)/180.0)
               + (4.0 if a.get("core_exam_qualified") else -4.0)
               + (2.0 if str(a.get("core_exam_tier") or "")=="CORE" else 0.0)
               - (4.0 if str(a.get("core_exam_tier") or "")=="SUPPORT" else 0.0)
               - min(3.0,len(_core_ans)*0.08)
               + min(8.0,_reasoning_signals*2.5)
               - (4.0 if _reasoning_signals==0 else 0.0))
        ranked.append((score,a,derived,seg))
    if not ranked:
        return [],{"score_pipeline_diagnostic":pd}
    ranked.sort(key=lambda x:x[0],reverse=True)
    score,core,derived,seg=ranked[0]
    pd["candidate_accept"]=max(int(pd.get("candidate_accept",0)),len(ranked))
    pd["four_point_single_anchor_candidates"]=len(ranked)
    pd["final_reason"]="single_anchor_reasoning_candidates_ready"
    sub=[2,2] if need==2 else [1,1,2]
    relation_meta={
        "master_concept":_norm_anchor_text(core.get("topic","")),
        "relation":"하나의 핵심 개념/과정 안에 원래 함께 존재하는 source fact들을 순차 판단하는 근거 사슬",
        "thinking_types":["자료해석","관계판단","적용/수정"],
        "exam_skeleton":"동일 source block의 사실 해석 → 관계 판단 → 앞 판단을 이용한 적용/수정",
        "scoring_plan":[{"points":p,"action":"source fact를 이용한 순차 판단","answer":d.get("answer","")} for p,d in zip(sub,derived)],
        "material_limits":_compact_material_limits(4),
        "natural_unit_score":4.0,
        "core_exam_profile":[{"topic":core.get("topic",""),"score":core.get("core_exam_score",0),"tier":core.get("core_exam_tier","SUPPORT"),"breakdown":core.get("core_exam_breakdown",{})}],
        "quality_directive":(
            "4점 SINGLE-CONCEPT-CHAIN 구조다. 모든 채점요소는 하나의 핵심개념과 그 local source block에서 Python이 고정한 서로 다른 사실들이다. "
            "각 사실을 독립적으로 복사하게 하지 말고, 첫 자료해석이 다음 관계판단의 전제가 되고 마지막 2점 요구가 앞 판단을 실제로 사용하게 구성한다. "
            "원문 문장을 그대로 나열하거나 정의 3개를 빈칸처럼 묻지 않는다. 새로운 기술 사실·수치·효과는 추가하지 않는다."
        ),
        "selector_reason":f"Python global single-concept 4pt score={score:.2f}",
        "relation_score":round(score,2),
        "selection_mode":"python_global_t4_single_concept_chain",
        "source_policy":"subnote_only_for_answer_content",
        "source_context_override":seg,
        "score_pipeline_diagnostic":copy.deepcopy(pd),
        "score_diagnostic":{"selected_rank":1,"topics":[core.get("topic","")],"candidate_count":len(ranked),"note":"R29 global fallback: no concept-specific hardcoding"},
    }
    return derived,relation_meta

def _smart_relation_bundle(db_path, domain, need, used_answers, excluded_topics, rng, pattern_id="", used_source_pages=None):
    import os
    try: _mtime=os.path.getmtime(db_path)
    except Exception: _mtime=0
    _cache_key=(os.path.abspath(db_path),_mtime,domain)
    _cached=_T2_SOURCE_CACHE.get(_cache_key) if not str(pattern_id).upper().startswith("T4") else None
    if _cached is not None:
        rows=copy.deepcopy(_cached["rows"]); source_kinds=copy.deepcopy(_cached["source_kinds"])
        page_map=copy.deepcopy(_cached["page_map"]); raw_page_map=copy.deepcopy(_cached["raw_page_map"])
    else:
        con=sqlite3.connect(db_path); con.row_factory=sqlite3.Row
        cols={r[1] for r in con.execute("PRAGMA table_info(anchors)").fetchall()}
        required={"domain","topic","answer","evidence","source_name","page_no"}
        missing=sorted(required-cols)
        if missing:
            con.close(); raise RuntimeError("knowledge.db anchors 스키마 누락: "+", ".join(missing))
        has_conf="confidence" in cols; conf_expr="COALESCE(confidence,0)" if has_conf else "0"
        rows=[dict(r) for r in con.execute(
            f"""SELECT domain,topic,answer,evidence,source_name,page_no, {conf_expr} AS confidence
                FROM anchors WHERE domain=? ORDER BY {conf_expr} DESC, source_name, page_no""",(domain,)).fetchall()]
        source_kinds=_source_kind_map(con); _source_names={str(r["source_name"]) for r in rows}
        page_map=_page_text_map(con,_source_names); raw_page_map=_page_raw_text_map(con,_source_names); con.close()
        if not str(pattern_id).upper().startswith("T4"):
            _T2_SOURCE_CACHE[_cache_key]={"rows":copy.deepcopy(rows),"source_kinds":copy.deepcopy(source_kinds),"page_map":copy.deepcopy(page_map),"raw_page_map":copy.deepcopy(raw_page_map)}

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
        "generic_only_chain_pairs":0,
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

    _is_t2=not str(pattern_id).upper().startswith("T4")
    _idnorm=(_t2_identity_key if _is_t2 else _topic_core)
    used_answers={_idnorm(x) for x in (used_answers or set()) if _idnorm(x)}
    excluded_topics={_idnorm(x) for x in (excluded_topics or set()) if _idnorm(x)}

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
        if used_source_pages and (str(a.get("source_name","")),int(a.get("page_no",0) or 0)) in set(used_source_pages):
            continue
        if _idnorm(a["answer"]) in used_answers:
            _pd["used_answer_reject"]+=1
            continue
        if _idnorm(a["topic"]) in excluded_topics:
            _pd["excluded_topic_reject"]+=1
            continue
        key=(_idnorm(a["topic"]),_idnorm(a["answer"]),a.get("source_name"),a.get("page_no"))
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
                _reason_prof=_four_point_reasoning_chain_profile(chosen)
                _pd["generic_only_chain_pairs"] += int(_reason_prof.get("generic_only_pairs",0))
                _shortcut_penalty=_four_point_shortcut_penalty(chosen)
                _reason_score=float(_reason_prof.get("score",0.0))
                # R23: 정의합성 shortcut이 감지되면 그 정의문 때문에 생긴 exact-link 보너스를
                # 실제 풀이 연결성으로 다시 보상하지 않는다.
                if float(_shortcut_penalty) < 0:
                    _reason_score=min(0.0,_reason_score)
                # R29 global gate: 특정 개념 예외 없이 정의합성 shortcut/약한 사슬은 API 전에 제거한다.
                if float(_shortcut_penalty) <= -5.0:
                    _pd.setdefault("four_point_shortcut_hard_reject",0)
                    _pd["four_point_shortcut_hard_reject"]+=1
                    continue
                if _reason_score < -2.0 and natural_score < 4.5:
                    _pd.setdefault("four_point_weak_chain_reject",0)
                    _pd["four_point_weak_chain_reject"]+=1
                    continue
                graph_score += (
                    min(5.0,chain_score*0.22)
                    + min(4.0,natural_score)
                    + _reason_score
                    + float(_shortcut_penalty)
                )
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
                + core_exam*0.35
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
            if str(pattern_id).upper().startswith("T4"):
                _pre_ok,_pre_reason=_t4_candidate_prejudge(chosen,pattern_id)
                if not _pre_ok:
                    _pd.setdefault("four_point_prejudge_reject",0)
                    _pd["four_point_prejudge_reject"]+=1
                    _pd.setdefault("four_point_prejudge_examples",[])
                    if len(_pd["four_point_prejudge_examples"])<5:
                        _pd["four_point_prejudge_examples"].append({
                            "topics":[str(x.get("topic","")) for x in chosen],"reason":_pre_reason
                        })
                    continue
            candidates.append((score_total,chosen))
            _pd["candidate_accept"]+=1

    if not candidates:
        # R42: 4점에서 single-concept raw-fact fallback을 금지한다.
        # 이 fallback은 하나의 source block에 있는 정의/특징 두 개를 1+1+2 또는 2+2로
        # 승격하면서 Writer가 사고사슬처럼 꾸며도 실제 채점은 원문 복사형이 되는 문제가 반복되었다.
        # 4점은 Python이 직접 관계/자연적 문제단위를 증명한 capability 후보만 사용한다.
        _pd["single_concept_t4_fallback_disabled"]=True
        _pd["final_reason"]="no_proven_t4_capability_after_all_filters"
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
            "core_exam_contribution":round(core_exam0*0.35,3),
            "support_count":support0,
            "support_penalty":round(-support0*1.75,3),
            "natural_unit_score":round(float(natural0),3),
            "reasoning_chain_profile":(
                (lambda _rp,_sp: dict(_rp, effective_score=round(
                    min(0.0,float(_rp.get("score",0.0))) if float(_sp)<0
                    else float(_rp.get("score",0.0)),3
                )))(
                    _four_point_reasoning_chain_profile(chosen0),
                    _four_point_shortcut_penalty(chosen0)
                )
                if len(chosen0)>=2 else {}
            ),
            "shortcut_penalty":round(float(_four_point_shortcut_penalty(chosen0)),3),
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

    if str(pattern_id).upper().startswith("T4"):
        top=uniq[:1]
        chosen_idx=0
    else:
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
            "note":"R22: 4점은 exact anchor 연결을 우선하고 generic-only 인접쌍을 감점. core_exam 메타가중치 0.35.",
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


def _question_content_tokens(q):
    vals=[]
    vals.extend(str(x) for x in (q.get("answer",[]) or []))
    vals.append(str(q.get("topic","")))
    toks=set()
    for v in vals:
        for t in _anchor_tokens(v):
            if len(t)>=3:
                toks.add(t)
    return toks


def _content_family_key(q):
    """
    broad families_for보다 세밀한 시험지 내용군.
    정답/주제의 핵심 토큰을 이용하여 같은 세부 내용의 반복을 잡는다.
    """
    domain=str(q.get("domain") or q.get("blueprint_domain") or "")
    toks=sorted(_question_content_tokens(q))
    return domain+"|"+"|".join(toks[:8])


def _content_overlap_too_high(q, previous, same_section=True):
    """
    exact answer 중복 + 세부 concept token 중복을 차단한다.
    대영역 자체의 재출제는 허용하되, 같은 세부 개념군 재탕은 막는다.
    """
    qa={_topic_core(x) for x in (q.get("answer",[]) or []) if _topic_core(x)}
    qt=_question_content_tokens(q)
    qdom=str(q.get("domain") or q.get("blueprint_domain") or "")
    for p in previous or []:
        pa={_topic_core(x) for x in (p.get("answer",[]) or []) if _topic_core(x)}
        if qa & pa:
            return True,"exact_answer"
        pdom=str(p.get("domain") or p.get("blueprint_domain") or "")
        if qdom and pdom and qdom!=pdom:
            continue
        pt=_question_content_tokens(p)
        common=qt & pt
        # 같은 영역에서 핵심 토큰 2개 이상 겹치면 같은 세부 내용군으로 본다.
        if len(common)>=2:
            return True,"concept_neighborhood"
    return False,""


def _source_neighborhood_key_from_bundle(bundle):
    if not bundle:
        return set()
    out=set()
    for a in bundle:
        src=str(a.get("source_name",""))
        try:
            page=int(a.get("page_no",0) or 0)
        except Exception:
            page=0
        if src:
            out.add((src,page))
    return out


def _bundle_content_overlap_too_high(bundle, previous):
    """AI 호출 전에 anchor 수준에서 최종 A/B 세부내용 중복을 차단한다."""
    qa={_topic_core(x.get("answer","")) for x in (bundle or []) if _topic_core(x.get("answer",""))}
    qdom=str((bundle or [{}])[0].get("domain","")) if bundle else ""
    qt=set()
    for x in bundle or []:
        for v in (x.get("topic",""),x.get("answer","")):
            qt.update(t for t in _anchor_tokens(v) if len(t)>=3)
    for p in previous or []:
        pa={_topic_core(x) for x in (p.get("answer",[]) or []) if _topic_core(x)}
        if qa & pa:
            return True,"exact_answer"
        pdom=str(p.get("domain") or p.get("blueprint_domain") or "")
        if qdom and pdom and qdom!=pdom:
            continue
        pt=_question_content_tokens(p)
        if len(qt & pt)>=2:
            return True,"concept_neighborhood"
    return False,""

def _source_neighborhood_conflict(bundle, used_source_pages):
    """
    최종 A/B에서 같은 서브노트의 '같은 페이지'를 정답원으로 재사용하지 않는다.
    인접 페이지까지 무조건 차단하면 23문항 편성에서 후보 고갈이 생길 수 있으므로,
    인접 페이지의 내용 중복은 exact answer / concept-neighborhood 검사에 맡긴다.
    """
    for src,page in _source_neighborhood_key_from_bundle(bundle):
        for usrc,upage in used_source_pages:
            if src==usrc and page==upage:
                return True
    return False


def _t2_remaining_capacity(db_path,domain,previous_questions=None):
    """R35 source-capability scan using distinct source units.
    Only candidates already proven as multi-item numeric rule-application capability are counted.
    Deterministic, API-free, cached for the exact prior-source state.
    """
    import os
    used_pages=set(); used_answers=set()
    for q in previous_questions or []:
        for src in q.get("sources",[]) or []:
            sn=str(src.get("source_name",""));
            try: pn=int(src.get("page_no",0) or 0)
            except Exception: pn=0
            if sn: used_pages.add((sn,pn))
    try: mt=os.path.getmtime(db_path)
    except Exception: mt=0
    key=(os.path.abspath(db_path),mt,domain,tuple(sorted(used_pages)))
    if key in _T2_CAPACITY_CACHE:
        return copy.deepcopy(_T2_CAPACITY_CACHE[key])
    universe=[]
    local_pages=set(used_pages); local_answers=set(used_answers)
    for k in range(16):
        b,m=_smart_relation_bundle(db_path,domain,2,local_answers,set(),random.Random(91731+k),"T2_REL",used_source_pages=local_pages)
        if not b: break
        ca=_norm_anchor_text(b[0].get("answer","")); cb=_norm_anchor_text((b[1].get("contrast_anchor") or {}).get("answer",""))
        rs=copy.deepcopy(b[1].get("reasoning_spec") or {})
        if rs.get("mode")!="numeric_series_rule_application":
            break
        pages=set()
        for _si in rs.get("source_items",[]) or []:
            _sn=str(_si.get("source_name", ""));
            try: _pn=int(_si.get("page_no",0) or 0)
            except Exception: _pn=0
            if _sn: pages.add((_sn,_pn))
        for a in [b[0],b[1].get("contrast_anchor") or {}]:
            sn=str(a.get("source_name",""));
            try: pn=int(a.get("page_no",0) or 0)
            except Exception: pn=0
            if sn: pages.add((sn,pn))
        universe.append({"core":ca,"contrast":cb,"pages":sorted(pages),"spec":rs})
        local_answers.update(x for x in (ca,cb) if x)
        local_pages.update(pages)
    out={"capacity":len(universe),"universe":universe}
    _T2_CAPACITY_CACHE[key]=copy.deepcopy(out)
    return out

def _rebalance_t2_domains_by_capability(db_path, plan, domains, rng, previous_questions=None):
    """R35 source-capability-first assignment.

    1) Count only deterministic structural-series paired candidates.
    2) Assign those candidates before any question pattern is chosen.
    3) Every remaining 2-point slot is routed to an existing deterministic formula capability
       by swapping domains with a 4-point slot when needed.
    4) Never lower the paired gate and never leave a normal 2-point slot in
       ``no_safe_structure`` state.

    The domain multiset is preserved whenever a formula-capable 4-point slot is available.
    """
    diagnostics={}; capacity={}
    for d in domains:
        try:
            scan=_t2_remaining_capacity(db_path,d,previous_questions)
            capacity[d]=int(scan.get("capacity",0) or 0)
            diagnostics[d]={
                "capable":capacity[d]>0,
                "candidate_count":capacity[d],
                "reason":"r38_numeric_rule_application_capability",
                "universe":scan.get("universe",[])
            }
        except Exception as ex:
            capacity[d]=0
            diagnostics[d]={"capable":False,"candidate_count":0,"reason":"capacity_scan_error:"+str(ex),"universe":[]}

    t2idx=[i for i,s in enumerate(plan) if int(s.get("points",0) or 0)==2]
    used4=set()
    for i in t2idx:
        cur=plan[i].get("domain")
        if capacity.get(cur,0)>0:
            capacity[cur]-=1
            plan[i]["t2_capability_mode"]="numeric_series_rule_application"
            continue

        # Prefer swapping in another proven paired capability.
        choices=[j for j,s in enumerate(plan)
                 if j not in used4 and int(s.get("points",0) or 0)==4
                 and capacity.get(s.get("domain"),0)>0]
        if choices:
            choices.sort(key=lambda j:(-capacity.get(plan[j].get("domain"),0),j))
            j=choices[0]; old2=cur; old4=plan[j].get("domain")
            plan[i]["domain"],plan[j]["domain"]=old4,old2
            plan[i]["capability_swap_from"]=old2; plan[j]["capability_swap_from"]=old4
            plan[i]["t2_capability_mode"]="numeric_series_rule_application"
            capacity[old4]-=1; used4.add(j)
            continue

        # No proven relational structure remains.  Route before generation to deterministic
        # formula capability; do not let REL/ERR/DATA/CMP selectors discover a bad pair.
        if cur in FORMULA_DOMAINS:
            plan[i]["question_type"]="간단계산"
            plan[i]["t2_capability_mode"]="formula_capability"
            continue

        fchoices=[j for j,s in enumerate(plan)
                  if j not in used4 and int(s.get("points",0) or 0)==4
                  and s.get("domain") in FORMULA_DOMAINS]
        if fchoices:
            # Prefer a domain-preserving swap when possible.
            j=fchoices[0]; old2=cur; old4=plan[j].get("domain")
            plan[i]["domain"],plan[j]["domain"]=old4,old2
            plan[i]["capability_swap_from"]=old2; plan[j]["capability_swap_from"]=old4
            plan[i]["question_type"]="간단계산"
            plan[i]["t2_capability_mode"]="formula_capability"
            used4.add(j)
            continue

        # If preserving the total domain multiset is impossible, never lower relation quality.
        # Keep all 4-point coverage intact and let only the 2-point slot repeat a proven formula domain.
        # Choose the least-used formula domain among already planned 2-point capability slots.
        counts={d:0 for d in FORMULA_DOMAINS}
        for _k,_s in enumerate(plan):
            if _k==i: continue
            if int(_s.get("points",0) or 0)==2 and _s.get("domain") in counts:
                counts[_s.get("domain")]+=1
        fd=sorted(FORMULA_DOMAINS,key=lambda d:(counts[d],d))[0]
        plan[i]["capability_domain_override_from"]=cur
        plan[i]["domain"]=fd
        plan[i]["question_type"]="간단계산"
        plan[i]["t2_capability_mode"]="formula_capability_repeat_domain"

    return plan,{d:(diagnostics[d]["candidate_count"]>0) for d in domains},diagnostics



_SAMPLE_T4_CAP_CACHE={}

def _sample_t4_operation_domains(domains):
    """R43 SAMPLE6: rank domains by deterministic 4-point operation diversity.

    The ranking is derived from Python-generated 1+1+2 templates, not concept names.
    Richer domains are repeated first when SAMPLE6 needs four 4-point slots.
    """
    ranked=[]
    for d in domains:
        if d not in FORMULA_DOMAINS:
            continue
        topics=set(); accepted=0
        for seed in range(18):
            q=generate_formula_question(d,random.Random(43100+seed))
            if not q:
                continue
            q=_enrich_formula(q,4)
            if not q or validate_formula_question(q):
                continue
            accepted+=1; topics.add(str(q.get("topic","")))
        if accepted:
            ranked.append((len(topics),accepted,d))
    ranked.sort(key=lambda x:(-x[0],-x[1],str(x[2])))
    return [x[2] for x in ranked]

def make_section(db_path,section,count,points,domains=None,api_key="",model="gpt-5.6-luna",
                 ai_enabled=True,ai_quality_enabled=True,judge_model=None,seed=None,
                 previous_questions=None,shared_answers=None,tuning_mode=False,coverage_targets=None):
    rng=random.Random(seed)
    domains=list(domains or DOMAINS)
    scores=score_pattern(section,count,points)
    plan=blueprint(section,scores,domains,rng)
    plan,_t2_capability,_t2_capability_diagnostics=_rebalance_t2_domains_by_capability(db_path,plan,domains,rng,previous_questions=previous_questions)
    # R45: SAMPLE6 may run in coverage mode.  Four 4-point slots are bound to
    # explicit domain x reasoning-capability targets.  A target never falls back to
    # an unrelated capability; failed candidates stay failed so coverage is honest.
    _coverage_targets=list(coverage_targets or [])
    if tuning_mode and section=="SAMPLE":
        if _coverage_targets:
            if len(_coverage_targets)!=4:
                raise RuntimeError("R45 coverage_targets는 4점 슬롯 4개와 정확히 대응해야 합니다.")
            _k=0
            for _slot in plan:
                if int(_slot.get("points",0) or 0)==4:
                    _t=dict(_coverage_targets[_k])
                    _slot["domain_override_from"]=_slot.get("domain")
                    _slot["domain"]=_t.get("domain")
                    _slot["t4_capability_mode"]=_t.get("capability_id")
                    _slot["coverage_key"]=_t.get("coverage_key") or (str(_slot["domain"])+"::"+str(_slot["t4_capability_mode"]))
                    _slot["question_type"]="계산/판단" if _slot["t4_capability_mode"]=="deterministic_formula_operation" else "추론/적용"
                    _k+=1
        else:
            _t4_domains=_sample_t4_operation_domains(domains)
            if not _t4_domains:
                raise RuntimeError("SAMPLE6: Python으로 풀이 연산을 증명한 4점 capability 영역이 없습니다.")
            _k=0
            for _slot in plan:
                if int(_slot.get("points",0) or 0)==4:
                    _slot["domain_override_from"]=_slot.get("domain")
                    _slot["domain"]=_t4_domains[_k % len(_t4_domains)]
                    _slot["question_type"]="계산/판단"
                    _slot["t4_capability_mode"]="deterministic_formula_operation"
                    _slot["coverage_key"]=str(_slot["domain"])+"::deterministic_formula_operation"
                    _k+=1
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
    used_source_pages=set()
    for _pq in prior:
        for _src in (_pq.get("sources",[]) or []):
            _sn=str(_src.get("source_name",""))
            try:
                _pn=int(_src.get("page_no",0) or 0)
            except Exception:
                _pn=0
            if _sn:
                used_source_pages.add((_sn,_pn))
    ai_calls=0; fallbacks=0; formula_used=0
    judge_calls=0; judge_rejects=0; selector_calls=0
    # R35: source-capability planning decides how many 2-point formula fallbacks are required.
    # Do not silently push a pre-routed formula slot back into the relation selector.
    formula_cap=max(2, sum(1 for _s in plan if (
        (int(_s.get("points",0) or 0)==2 and str(_s.get("t2_capability_mode","")).startswith("formula_capability"))
        or (int(_s.get("points",0) or 0)==4 and _s.get("t4_capability_mode")=="deterministic_formula_operation")
    )))

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
        if int(slot.get("points",0) or 0)==2 and slot.get("t2_capability_mode")=="planning_error_no_capability":
            raise RuntimeError("R35 T2 capability planning failed before Writer/API: no proven relational or formula structure")
        pts=slot["points"]; dom=slot["domain"]; q=None
        wants_calc=slot["question_type"] in {"간단계산","계산/판단"}

        # R45 source-owned reasoning capability. Python fixes the source spans,
        # task shape and answers; AI Judge is used only after the complete SAMPLE.
        _cap_mode=str(slot.get("t4_capability_mode","") or "")
        if q is None and pts==4 and _cap_mode and _cap_mode!="deterministic_formula_operation":
            cand=generate_reasoning_question(
                db_path,dom,_cap_mode,rng,
                avoid_fingerprints={str(x.get("fingerprint","")) for x in (prior+qs)}
            )
            if not cand:
                raise RuntimeError(f"R45 source capability 후보 소진: {dom} · {_cap_mode}")
            cand["coverage_key"]=slot.get("coverage_key") or (str(dom)+"::"+_cap_mode)
            cand["ai_quality"]={"pass":None,"mode":"coverage_candidate_python_grounded"}
            q=cand

        if wants_calc and dom not in FORMULA_DOMAINS:
            slot["question_type"]="자료해석" if pts==4 else "자료식별"
            wants_calc=False

        if wants_calc and dom in FORMULA_DOMAINS and formula_used<formula_cap:
            _formula_enrich_failures=0
            _formula_attempt_cap = (120 if pts==4 and slot.get("t4_capability_mode")=="deterministic_formula_operation" else (16 if quality_active else 40))
            for _ in range(_formula_attempt_cap):
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
                    _enrich_fail_limit = (80 if pts==4 and slot.get("t4_capability_mode")=="deterministic_formula_operation" else 4)
                    if _formula_enrich_failures>=_enrich_fail_limit:
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

        # R35: capability routing is binding.  A formula-capability slot may never
        # silently fall back to a relational selector after template/judge failure.
        if q is None and pts==2 and str(slot.get("t2_capability_mode","")).startswith("formula_capability"):
            raise RuntimeError(
                f"R35 formula capability exhausted before Writer fallback: {section}-{slot.get('number')} · {dom}"
            )
        if q is None and pts==4 and slot.get("t4_capability_mode")=="deterministic_formula_operation":
            raise RuntimeError(
                f"R43 4점 풀이연산 capability 소진: {section}-{slot.get('number')} · {dom}. "
                "relation/direct-chain fallback은 허용하지 않습니다."
            )

        if q is None:
            first_pat=weighted_pick(rng,pts,calc=False,used=used_patterns)
            local_rejected_topics=set()

            # 한 문제 슬롯이 수십 번 반복되지 않도록 AI 후보 예산을 제한한다.
            # 품질 기준은 그대로이며, 실패 원인에 따라 다른 패턴으로 즉시 전환한다.
            slot_candidate_budget = ((4 if pts==2 else 2) if quality_active else ((4 if pts==2 else 3) if tuning_mode else 8))
            slot_candidates_used = 0
            # selector 자체 호출도 슬롯당 제한한다. REJECT/timeout도 호출 1회로 계산한다.
            selector_attempt_limit = 0
            selector_attempts = 0

            for pat in _concept_patterns(rng,pts,first_pat):
                # R35 capability owns pattern eligibility.  Rule-application questions use the
                # error-correction frame only; DATA/CMP/REL are not retried against the same source
                # merely to fill a slot.
                if pts==2 and slot.get("t2_capability_mode")=="numeric_series_rule_application" and pat.get("id")!="T2_ERR":
                    continue
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

                for _ in range(1 if (quality_active or tuning_mode) else 32):
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
                            pattern_id=pat.get("id",""), used_source_pages=used_source_pages
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
                            # R28: selector가 결정한 사고논리를 절대 덮어쓰지 않는다.
                            # 이전 R27까지는 여기서 quality_directive를 통째로 교체해
                            # DECISION-FIRST/ONE-ANCHOR 지시가 Writer에 전달되지 않는 회귀가 있었다.
                            _base_qd=str(relation_meta.get("quality_directive","")).strip()
                            _pattern_qd=(
                                " 현재 패턴의 표현 골격도 지킬 것: "
                                + str(pat.get("quality_rule",pat.get("name","")))
                            )
                            relation_meta["quality_directive"]=(_base_qd+_pattern_qd).strip()
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

                    # R25: 최종 A/B 세부내용 중복은 Writer/API 호출 전에 먼저 차단한다.
                    if not tuning_mode:
                        _predup,_prewhy=_bundle_content_overlap_too_high(bundle,prior+qs)
                        if _predup:
                            diag(slot,"content_diversity_pre_writer",
                                 "AI 호출 전 세부 내용 반복 방지: "+str(_prewhy),
                                 pattern=pat.get("id"),
                                 candidate_topics=[str(x.get("topic","")) for x in bundle])
                            if bundle:
                                local_rejected_topics.add(str(bundle[0].get("topic","")))
                            continue

                    # 최종 A/B에서는 같은 서브노트의 같은 페이지가 정답원으로 반복되지 않게 한다.
                    # 6문항 튜닝은 버전 비교 가능성을 위해 이 제한을 적용하지 않는다.
                    if (not tuning_mode) and _source_neighborhood_conflict(bundle,used_source_pages):
                        diag(slot,"content_source_diversity",
                             "같은 출처의 동일 페이지 내용 반복 방지",
                             pattern=pat.get("id"),
                             candidate_topics=[str(x.get("topic","")) for x in bundle])
                        if bundle:
                            local_rejected_topics.add(str(bundle[0].get("topic","")))
                        continue

                    # AI 호출 전에 Python이 채점 논리와 기출형 골격을 확정한다.
                    if relation_meta:
                        relation_meta["exam_skeleton"]=_skeleton_for_pattern(pat.get("id",""))
                        relation_meta["scoring_plan"]=_scoring_plan(pat,bundle)
                        relation_meta["quality_directive"]=(
                            str(relation_meta.get("quality_directive","")) + " "
                            + str(pat.get("quality_rule",""))
                        ).strip()

                    ctx=bundle_context(db_path,bundle)
                    # R40: single-concept T4 candidates are selected from an exact local
                    # source block.  Use that same block for Writer/Judge/grounded validation
                    # instead of silently switching back to the whole PDF page.
                    _ctx_override=str(relation_meta.get("source_context_override","") or "").strip()
                    if _ctx_override:
                        ctx=_ctx_override
                    # R27: DECISION-FIRST의 비교용 sibling도 Writer/Judge가 검증할 수 있는
                    # source context에 포함한다. 같은 출처/인접 페이지에서 Python이 고른 원문만 사용한다.
                    if pts==2 and relation_meta.get("selection_mode") in {
                        "python_exam_value_decision_first_t2",
                        "python_exam_value_t2_reasoning_matrix",
                    }:
                        _spec=relation_meta.get("reasoning_spec") or {}
                        _cc=str(relation_meta.get("contrast_context","") or "").strip()
                        if _spec.get("mode")=="paired_fact_slot_consistency":
                            ctx=(ctx
                                 +"\n[중심 item 사실1]\n"+str(_spec.get("core_base_fact",""))
                                 +"\n[중심 item 사실2]\n"+str(_spec.get("core_link_fact",""))
                                 +"\n[비교 item 사실1]\n"+str(_spec.get("contrast_base_fact",""))
                                 +"\n[비교 item 사실2]\n"+str(_spec.get("contrast_link_fact",""))).strip()
                        elif _cc:
                            ctx=(ctx+"\n[비교용 비정답 원문]\n"+_cc).strip()
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

                        if cand is not None and pts==2:
                            _copy_errs=_t2_near_copy_errors(cand,bundle)
                            _reason_errs=_t2_reasoning_shape_errors(cand,relation_meta,bundle)
                            _t2_prejudge_errs=list(dict.fromkeys(list(_copy_errs)+list(_reason_errs)))
                            if _t2_prejudge_errs:
                                diag(slot,"two_point_prejudge_reasoning",
                                     " / ".join(_t2_prejudge_errs),
                                     pattern=pat.get("id"),
                                     candidate_topics=[str(x.get("topic","")) for x in bundle])
                                if bundle:
                                    local_rejected_topics.add(str(bundle[0].get("topic","")))
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
                                    # R25: 2점은 Python distinct-support prefilter를 통과한 다른 anchor를
                                    # 최대 4개까지만 시도한다. 같은 쉬운 anchor를 반복 호출하지 않는다.
                                    relation_meta["force_pattern_switch"]=(pts!=2)
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
                            elif (not tuning_mode) and _content_overlap_too_high(cand,prior+qs)[0]:
                                _dup,_why=_content_overlap_too_high(cand,prior+qs)
                                diag(slot,"content_diversity",
                                     "같은 세부 내용 반복 방지: "+str(_why),
                                     pattern=pat.get("id"),
                                     candidate_topics=[str(x.get("topic","")) for x in bundle])
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
                    used_source_pages.update(_source_neighborhood_key_from_bundle(bundle))
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

        if slot.get("coverage_key"):
            q["coverage_key"]=slot.get("coverage_key")
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
                        ai_enabled=True,judge_model=None,seed=None,ai_quality_enabled=True,coverage_targets=None):
    """
    품질 튜닝 전용 6문항.
    정확히 2점 2개 + 4점 4개.
    AI 관계성 selector/Blind/Grounded/섹션 심사는 호출하지 않는다.
    Python 관계묶음 + Luna writer + deterministic grounding 검증만 수행한다.
    """
    exam=make_section(
        db_path,"SAMPLE",6,20,domains,api_key,model,
        ai_enabled,False,judge_model,seed=seed,
        tuning_mode=True,coverage_targets=coverage_targets
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
            "SAMPLE6 내부 검증 실패: 생성 단계에서 AI 품질심사가 호출되었습니다. "
            "SAMPLE6은 생성 후 문항별 1회 심사만 허용합니다."
        )

    # R39: SAMPLE6은 재시도 없는 실전 품질검사다.
    # 생성 단계에서는 Judge를 호출하지 않고, 완성된 6문항을 각각 정확히 한 번만 심사한다.
    sample_reviews=[]
    if bool(ai_quality_enabled and ai_enabled and api_key):
        _style=official_style_profile(db_path)
        _jm=judge_model or model
        for q in exam.get("questions",[]):
            try:
                rv=judge_question(api_key,_jm,q,"",_style)
            except Exception as ex:
                rv={"pass":False,"reason":"SAMPLE6 AI 품질심사 호출 실패: "+str(ex),"fatal_flags":["JUDGE_CALL_ERROR"]}
            q["sample_ai_quality"]=rv
            sample_reviews.append({
                "number":q.get("number"),"domain":q.get("domain"),"pattern_id":q.get("pattern_id"),
                "coverage_key":q.get("coverage_key"),"capability_id":q.get("capability_id") or q.get("t4_capability_mode"),
                "pass":bool(rv.get("pass")),"reason":str(rv.get("reason","")),
                "fatal_flags":list(rv.get("fatal_flags",[]) or []),"scores":copy.deepcopy(rv.get("scores",{}))
            })
    exam["sample_ai_reviews"]=sample_reviews
    exam["sample_ai_pass_count"]=sum(1 for r in sample_reviews if r.get("pass"))
    exam["sample_ai_reject_count"]=sum(1 for r in sample_reviews if not r.get("pass"))
    exam["builder_api_version"]=BUILDER_API_VERSION
    exam["sample_mode"]="PYTHON_PREFLIGHT_THEN_SINGLE_JUDGE"
    return exam



def _coverage_failure_signals(review, question):
    """Normalize Judge output into regression-level failure classes.

    This intentionally preserves the complete failure distribution.  It is not
    used to retry or replace a question during the 18-capability suite.
    """
    signals=[]
    for flag in (review.get("fatal_flags",[]) or []):
        flag=str(flag or '').strip()
        if flag:
            signals.append("FATAL:"+flag)
    scores=review.get("scores",{}) or {}
    cap=str(question.get("capability_id") or question.get("t4_capability_mode") or '')
    infer_floor=3.0 if (cap in ('deterministic_formula_operation','ordered_sequence_repair') or question.get('pattern_id')=='T4_C112') else 3.5
    floors={
        'grounding':4.0,'answer_leakage':4.0,'coherence':4.0,
        'inferential_distance':infer_floor,'task_distinctness':4.0,
        'exam_realism':4.0,'difficulty_fit':4.0,'ambiguity_control':4.0,
    }
    for key,floor in floors.items():
        try: val=float(scores.get(key,0) or 0)
        except Exception: val=0.0
        if val < floor:
            signals.append(f"SCORE:{key}<{floor:g}")
    if not bool(review.get('pass')) and not signals:
        signals.append('JUDGE:REJECT_OTHER')
    return sorted(set(signals))


def make_capability_validation_suite(db_path,domains=None,api_key="",model="gpt-5.6-luna",
                                     judge_model=None,seed=None,ai_quality_enabled=True):
    """R46: exhaustively test all 9×2 selected 4-point capabilities in one run.

    Exactly one Python-grounded question is constructed per coverage key.  Judge
    rejection NEVER triggers replacement/retry.  The point is to observe the
    whole failure surface first and fix generators by failure class afterwards.
    """
    if domains is None:
        domains=list(DEFAULT_DOMAINS)
    inv=coverage_inventory(db_path,domains,FORMULA_DOMAINS)
    targets=list(inv.get('targets',[]) or [])
    expected=len(domains)*2
    if not inv.get('all_domains_two_targets') or len(targets)!=expected:
        gaps=[d for d,v in (inv.get('domains',{}) or {}).items() if not v.get('target_met')]
        raise RuntimeError(
            f"R48 전수검증 시작 차단: 검증된 구조 후보가 {len(targets)}/{expected}입니다. "
            f"0-pass capability를 폐기한 뒤 부족 영역: {', '.join(gaps)}. "
            "18개를 억지로 채워 API를 다시 소모하지 않습니다."
        )

    rng=random.Random(460000 + (0 if seed is None else int(seed)))
    questions=[]
    construction=[]
    for idx,target in enumerate(targets,1):
        dom=str(target.get('domain',''))
        cap=str(target.get('capability_id',''))
        ck=str(target.get('coverage_key') or f'{dom}::{cap}')
        q=None
        if cap=='deterministic_formula_operation':
            # Python-only candidate search is allowed before Judge.  No Judge
            # feedback is used here and there is no post-Judge replacement.
            for _ in range(160):
                cand=generate_formula_question(dom,rng)
                if not cand:
                    continue
                cand=_enrich_formula(cand,4)
                if not cand or validate_formula_question(cand):
                    continue
                q=cand
                break
        else:
            q=generate_reasoning_question(db_path,dom,cap,rng)
        if not q:
            construction.append({'coverage_key':ck,'domain':dom,'capability_id':cap,'constructed':False})
            raise RuntimeError(f"R46 전수검증 문항 구성 실패: {ck}")
        q['number']=idx
        q['section']='CAPABILITY18'
        q['coverage_key']=ck
        q['capability_id']=cap
        q['points']=4
        construction.append({'coverage_key':ck,'domain':dom,'capability_id':cap,'constructed':True,'topic':q.get('topic','')})
        questions.append(q)

    style=official_style_profile(db_path)
    jm=judge_model or model
    reviews=[]
    for q in questions:
        if ai_quality_enabled and api_key:
            try:
                rv=judge_question(api_key,jm,q,"",style)
            except Exception as ex:
                rv={'pass':False,'reason':'CAPABILITY18 AI 품질심사 호출 실패: '+str(ex),'fatal_flags':['JUDGE_CALL_ERROR'],'scores':{}}
        else:
            rv={'pass':None,'reason':'AI Judge 미실행','fatal_flags':[],'scores':{}}
        signals=_coverage_failure_signals(rv,q) if rv.get('pass') is not None else []
        q['capability_suite_ai_quality']=copy.deepcopy(rv)
        reviews.append({
            'number':q.get('number'),'coverage_key':q.get('coverage_key'),'domain':q.get('domain'),
            'capability_id':q.get('capability_id'),'pattern_id':q.get('pattern_id'),'topic':q.get('topic',''),
            'pass':rv.get('pass'),'reason':str(rv.get('reason','')),
            'fatal_flags':list(rv.get('fatal_flags',[]) or []),'scores':copy.deepcopy(rv.get('scores',{})),
            'failure_signals':signals,
        })

    by_signal={}; by_capability={}; by_domain={}
    for r in reviews:
        d=r['domain']; c=r['capability_id']
        by_domain.setdefault(d,{'tested':0,'pass':0,'reject':0})
        by_domain[d]['tested']+=1
        if r.get('pass') is True: by_domain[d]['pass']+=1
        elif r.get('pass') is False: by_domain[d]['reject']+=1
        by_capability.setdefault(c,{'tested':0,'pass':0,'reject':0,'signals':{}})
        by_capability[c]['tested']+=1
        if r.get('pass') is True: by_capability[c]['pass']+=1
        elif r.get('pass') is False: by_capability[c]['reject']+=1
        for sig in r.get('failure_signals',[]):
            by_signal[sig]=by_signal.get(sig,0)+1
            m=by_capability[c]['signals']; m[sig]=m.get(sig,0)+1

    tested=sum(1 for r in reviews if r.get('pass') is not None)
    passed=sum(1 for r in reviews if r.get('pass') is True)
    rejected=sum(1 for r in reviews if r.get('pass') is False)
    return {
        'mode':'FULL_18_CAPABILITY_REGRESSION_NO_JUDGE_RETRY',
        'builder_api_version':BUILDER_API_VERSION,
        'candidate_inventory':inv,
        'construction':construction,
        'questions':questions,
        'reviews':reviews,
        'summary':{'target_total':expected,'constructed':len(questions),'judge_tested':tested,'pass':passed,'reject':rejected,'all_pass':bool(tested==expected and passed==expected)},
        'failure_class_counts':dict(sorted(by_signal.items(),key=lambda kv:(-kv[1],kv[0]))),
        'capability_summary':by_capability,
        'domain_summary':by_domain,
        'final_ab_coverage_ready':bool(tested==expected and passed==expected),
    }

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
                _ab_dup=[]
                for _bq in B["questions"]:
                    _dup,_why=_content_overlap_too_high(_bq,A["questions"])
                    if _dup:
                        _ab_dup.append((_bq.get("number"),_why,_bq.get("topic","")))
                if _ab_dup:
                    last_error=RuntimeError(
                        "A/B 세부 내용 중복 발견: "+str(_ab_dup[:5])
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
