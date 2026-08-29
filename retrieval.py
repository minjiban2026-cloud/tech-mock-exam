
import sqlite3, random

def connect(db_path):
    con=sqlite3.connect(db_path)
    con.row_factory=sqlite3.Row
    return con

def domain_clause(domain):
    if domain=="발명":
        return "기술교육론"
    return domain

def search_pages(db_path, domain, query="", kind="subnote", limit=8):
    con=connect(db_path)
    d=domain_clause(domain)
    if query.strip():
        rows=con.execute("""
          SELECT source_name,kind,domain,page_no,text
          FROM pages_fts
          WHERE pages_fts MATCH ? AND kind=? AND domain=?
          LIMIT ?
        """,(query,kind,d,limit)).fetchall()
    else:
        rows=con.execute("""
          SELECT source_name,kind,domain,page_no,text
          FROM pages
          WHERE kind=? AND domain=? AND length(text)>150
          ORDER BY RANDOM() LIMIT ?
        """,(kind,d,limit)).fetchall()
    con.close()
    return [dict(r) for r in rows]

def anchor_pool(db_path, domain, used_answers=None, limit=160):
    """한 번에 충분한 후보를 읽어 반복적인 DB 랜덤 호출과 후보 고갈을 줄인다."""
    used_answers=set(used_answers or [])
    con=connect(db_path)
    rows=con.execute("""
      SELECT * FROM anchors
      WHERE domain=? AND length(answer)>=2 AND length(evidence)>=12
      ORDER BY confidence DESC, RANDOM()
      LIMIT ?
    """,(domain,limit)).fetchall()
    con.close()
    out=[]
    for r in rows:
        d=dict(r)
        if d["answer"] not in used_answers:
            out.append(d)
    return out

def random_anchor(db_path, domain, used_answers=None):
    pool=anchor_pool(db_path,domain,used_answers,80)
    return pool[0] if pool else None

def source_context(db_path, source_name, page_no, radius=0):
    con=connect(db_path)
    rows=con.execute("""
      SELECT page_no,text FROM pages
      WHERE source_name=? AND page_no BETWEEN ? AND ?
      ORDER BY page_no
    """,(source_name,max(1,int(page_no)-radius),int(page_no)+radius)).fetchall()
    con.close()
    return "\n\n".join(f"[p.{r['page_no']}]\n{r['text']}" for r in rows)

def style_snippets(db_path, limit=3):
    """기존 모의고사 문제 PDF에서 형식 참고용 짧은 조각만 가져온다."""
    con=connect(db_path)
    rows=con.execute("""
      SELECT source_name,page_no,text FROM pages
      WHERE kind='style' AND length(text)>80
      ORDER BY RANDOM() LIMIT ?
    """,(limit,)).fetchall()
    con.close()
    return [dict(r) for r in rows]
