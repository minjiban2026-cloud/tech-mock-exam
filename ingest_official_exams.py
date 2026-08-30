import sqlite3, hashlib, re
from pathlib import Path
import fitz
ROOT=Path('/mnt/data'); DB=Path(__file__).with_name('knowledge.db')
files=sorted(ROOT.glob('20??_중등1차_기술_전공[AB].pdf'))
con=sqlite3.connect(DB)
for f in files:
    if con.execute('select 1 from sources where name=?',(f.name,)).fetchone(): continue
    doc=fitz.open(f); sha=hashlib.sha256(f.read_bytes()).hexdigest()
    con.execute('insert into sources(name,kind,domain,page_count,sha256) values(?,?,?,?,?)',(f.name,'official_exam','복합',len(doc),sha))
    sid=con.execute('select id from sources where name=?',(f.name,)).fetchone()[0]
    for i,p in enumerate(doc):
        text=p.get_text('text')
        con.execute('insert into pages(source_id,source_name,kind,domain,page_no,text) values(?,?,?,?,?,?)',(sid,f.name,'official_exam','복합',i+1,text))
        pid=con.execute('select last_insert_rowid()').fetchone()[0]
        con.execute('insert into pages_fts(rowid,source_name,kind,domain,page_no,text) values(?,?,?,?,?,?)',(pid,f.name,'official_exam','복합',i+1,text))
con.commit(); print('official exams',len(files)); con.close()
