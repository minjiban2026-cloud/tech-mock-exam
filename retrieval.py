
import sqlite3, random, re

BAD_WORDS=("요약","목차","단원","개요","정리","내용체계","서브노트","교과서 정리")

def connect(db_path):
    con=sqlite3.connect(db_path)
    con.row_factory=sqlite3.Row
    return con

def search_pages(db_path, domain, query="", kind="subnote", limit=8):
    con=connect(db_path)
    if query.strip():
        rows=con.execute("""
          SELECT source_name,kind,domain,page_no,text
          FROM pages_fts WHERE pages_fts MATCH ? AND kind=? AND domain=? LIMIT ?
        """,(query,kind,domain,limit)).fetchall()
    else:
        rows=con.execute("""
          SELECT source_name,kind,domain,page_no,text FROM pages
          WHERE kind=? AND domain=? AND length(text)>150 ORDER BY RANDOM() LIMIT ?
        """,(kind,domain,limit)).fetchall()
    con.close()
    return [dict(r) for r in rows]

def _safe_anchor(a):
    if not a: return False
    topic=str(a["topic"]).strip(); ans=str(a["answer"]).strip(); ev=str(a["evidence"]).strip()
    if len(ans)<2 or len(ans)>55 or len(ev)<15: return False
    if topic in {"방법","내용","특징","종류","분류","영역","단원","목차","개요","정리"}: return False
    if any(w in topic or w in ans for w in BAD_WORDS): return False
    if re.fullmatch(r"표준\d+",ans): return False
    return True

def anchor_pool(db_path, domain, used_answers=None, limit=250):
    used=set(used_answers or [])
    con=connect(db_path)
    rows=con.execute("""
      SELECT * FROM anchors WHERE domain=? ORDER BY confidence DESC, RANDOM() LIMIT ?
    """,(domain,limit)).fetchall()
    con.close()
    out=[]
    for r in rows:
        d=dict(r)
        if d["answer"] in used or not _safe_anchor(d): continue
        out.append(d)
    return out

def random_anchor(db_path, domain, used_answers=None):
    p=anchor_pool(db_path,domain,used_answers,100)
    return p[0] if p else None

def related_bundle(db_path, domain, size, used_answers=None, used_topics=None):
    """같은 페이지 또는 인접 페이지의 앵커를 우선 묶어 실제 기출처럼 한 맥락에서 여러 요소를 묻는다."""
    used_topics=set(used_topics or [])
    pool=[a for a in anchor_pool(db_path,domain,used_answers,300) if a["topic"] not in used_topics]
    if not pool: return []
    # 같은 source/page 중심의 후보군
    for seed in pool:
        near=[a for a in pool
              if a["source_name"]==seed["source_name"]
              and abs(int(a["page_no"])-int(seed["page_no"]))<=1
              and a["answer"]!=seed["answer"]]
        group=[seed]
        seen={seed["answer"]}
        for a in near:
            if a["answer"] not in seen and a["topic"] not in {x["topic"] for x in group}:
                group.append(a); seen.add(a["answer"])
            if len(group)>=size: return group[:size]
    # 같은 영역 안에서라도 최대한 근접한 페이지끼리
    pool=sorted(pool,key=lambda a:(a["source_name"],int(a["page_no"])))
    for i in range(len(pool)):
        group=[pool[i]]
        for j in range(i+1,len(pool)):
            a=pool[j]
            if a["source_name"]==group[0]["source_name"] and int(a["page_no"])-int(group[0]["page_no"])<=3:
                if a["answer"] not in {x["answer"] for x in group}:
                    group.append(a)
            if len(group)>=size: return group[:size]
    return pool[:size] if len(pool)>=size else []

def source_context(db_path, source_name, page_no, radius=0):
    con=connect(db_path)
    rows=con.execute("""
      SELECT page_no,text FROM pages WHERE source_name=? AND page_no BETWEEN ? AND ? ORDER BY page_no
    """,(source_name,max(1,int(page_no)-radius),int(page_no)+radius)).fetchall()
    con.close()
    return "\n\n".join(f"[p.{r['page_no']}]\n{r['text']}" for r in rows)

def bundle_context(db_path,bundle):
    parts=[]
    done=set()
    for a in bundle:
        key=(a["source_name"],a["page_no"])
        if key in done: continue
        done.add(key)
        parts.append(source_context(db_path,a["source_name"],a["page_no"],0))
    return "\n\n".join(parts)

def official_style_profile(db_path, limit=4):
    """원문을 복사하지 않고 실제 기출의 구조적 특징만 전달하기 위한 고정 프로필."""
    return (
      "실제 기술 임용 원본의 전형적 형식: 2점은 대체로 1점짜리 채점요소 2개. "
      "4점은 1+1+2, 2+2, 1+1+1+1, 1+3 등으로 부분점수가 구성된다. "
      "한 자료 안에서 용어→원리/특징→계산·적용·비교가 자연스럽게 이어진다. "
      "표·대화·수업계획·조건·과정자료를 활용하며 서로 무관한 개념을 억지로 묶지 않는다."
    )
