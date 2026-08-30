import sqlite3, json
from pathlib import Path
from patterns import PATTERNS
DB=Path(__file__).with_name('knowledge.db')
con=sqlite3.connect(DB)
con.executescript('''
CREATE TABLE IF NOT EXISTS exam_patterns(id TEXT PRIMARY KEY, points INTEGER, name TEXT, verbs TEXT, visual INTEGER, calc INTEGER, provenance TEXT);
CREATE TABLE IF NOT EXISTS source_issues(id INTEGER PRIMARY KEY AUTOINCREMENT, source_name TEXT, page_no INTEGER, issue_type TEXT, detail TEXT, severity TEXT, status TEXT DEFAULT 'open');
CREATE TABLE IF NOT EXISTS concept_links(id INTEGER PRIMARY KEY AUTOINCREMENT, domain TEXT, concept_a TEXT, concept_b TEXT, relation TEXT, provenance TEXT);
CREATE TABLE IF NOT EXISTS generation_audit(id INTEGER PRIMARY KEY AUTOINCREMENT, created_at TEXT DEFAULT CURRENT_TIMESTAMP, section TEXT, qno INTEGER, verifier TEXT, pattern_id TEXT, source_name TEXT, page_no INTEGER, checks TEXT);
''')
for p in PATTERNS:
    con.execute('INSERT OR REPLACE INTO exam_patterns VALUES(?,?,?,?,?,?,?)',(p['id'],p['points'],p['name'],json.dumps(p['verbs'],ensure_ascii=False),int(p['visual']),int(p['calc']),'2021~2026 실제 임용 기출 구조 일반화'))
# 발명 앵커 보강: 기술교육론 앵커 중 명확한 발명/지재 키워드만 복제
keys=('발명','특허','지식재산','아이디어','창의','TRIZ','트리즈','브레인스토밍')
rows=con.execute("SELECT topic,answer,evidence,source_name,page_no,confidence FROM anchors WHERE domain='기술교육론'").fetchall()
for r in rows:
    blob=' '.join(str(x or '') for x in r[:3])
    if any(k.lower() in blob.lower() for k in keys):
        con.execute('INSERT INTO anchors(domain,topic,answer,evidence,source_name,page_no,confidence) SELECT ?,?,?,?,?,?,? WHERE NOT EXISTS(SELECT 1 FROM anchors WHERE domain=? AND topic=? AND answer=? AND source_name=? AND page_no=?)',('발명',*r,'발명',r[0],r[1],r[3],r[4]))
con.commit();con.close()
print('migration complete')
