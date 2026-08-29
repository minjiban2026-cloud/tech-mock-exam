
import io, os, textwrap
from PIL import Image, ImageDraw, ImageFont

def font_path():
    cands=[
      "/usr/share/fonts/truetype/nanum/NanumGothic.ttf",
      "/usr/share/fonts/truetype/nanum/NanumGothicBold.ttf",
      "C:/Windows/Fonts/malgun.ttf",
      "/System/Library/Fonts/AppleSDGothicNeo.ttc"
    ]
    for p in cands:
        if os.path.exists(p): return p
    raise RuntimeError("한글 폰트가 없습니다. Linux에서는 fonts-nanum 설치가 필요합니다.")

def wrap(draw,text,font,width):
    out=[]
    for para in str(text).splitlines() or [""]:
        cur=""
        for ch in para:
            if draw.textlength(cur+ch,font=font)<=width:
                cur+=ch
            else:
                if cur: out.append(cur)
                cur=ch
        if cur: out.append(cur)
    return out

def export_pdf(exam,answers=False):
    dpi=150; W=int(8.27*dpi); H=int(11.69*dpi); m=70; gap=24
    fp=font_path()
    body=ImageFont.truetype(fp,18); small=ImageFont.truetype(fp,15)
    head=ImageFont.truetype(fp,21); title=ImageFont.truetype(fp,30)
    pages=[]
    # cover
    im=Image.new("RGB",(W,H),"white"); d=ImageDraw.Draw(im)
    y=170; t=exam["exam_title"]+(" 정답·해설" if answers else "")
    d.text(((W-d.textlength(t,font=title))/2,y),t,font=title,fill="black")
    y+=70
    s=f"{len(exam['questions'])}문항 · {exam['total_points']}점 · 자동검증 통과"
    d.text(((W-d.textlength(s,font=head))/2,y),s,font=head,fill="black")
    pages.append(im)

    qs=exam["questions"]
    for start in range(0,len(qs),2):
        im=Image.new("RGB",(W,H),"white"); d=ImageDraw.Draw(im)
        mid=W//2; d.line((mid,m,mid,H-m),fill="black",width=1)
        for side,q in enumerate(qs[start:start+2]):
            x0=m if side==0 else mid+gap//2
            x1=mid-gap//2 if side==0 else W-m
            y=m
            titleline=f"{q['number']}. {q.get('intro','')} [{q['points']}점]"
            for ln in wrap(d,titleline,head,x1-x0):
                d.text((x0,y),ln,font=head,fill="black"); y+=29
            y+=8
            for ln in wrap(d,q.get("passage",""),body,x1-x0):
                d.text((x0,y),ln,font=body,fill="black"); y+=25
            if q.get("conditions"):
                y+=10; d.text((x0,y),"＜조건＞",font=body,fill="black"); y+=26
                for c in q["conditions"]:
                    for ln in wrap(d,"○ "+c,small,x1-x0):
                        d.text((x0,y),ln,font=small,fill="black"); y+=21
            y+=10; d.text((x0,y),"＜작성 방법＞",font=body,fill="black"); y+=26
            for t in q.get("tasks",[]):
                for ln in wrap(d,"○ "+t,small,x1-x0):
                    d.text((x0,y),ln,font=small,fill="black"); y+=21
            if answers:
                y+=12; d.text((x0,y),"＜정답·해설＞",font=body,fill="black"); y+=26
                for a in q.get("answer",[]):
                    for ln in wrap(d,"• "+a,small,x1-x0):
                        d.text((x0,y),ln,font=small,fill="black"); y+=21
                for s in q.get("solution",[]):
                    for ln in wrap(d,"- "+s,small,x1-x0):
                        d.text((x0,y),ln,font=small,fill="black"); y+=21
                if q.get("verifier")=="source":
                    src=f"근거: {q.get('source_name')} p.{q.get('page_no')}"
                    for ln in wrap(d,src,small,x1-x0):
                        d.text((x0,y),ln,font=small,fill="black"); y+=21
        pages.append(im)
    bio=io.BytesIO()
    pages[0].save(bio,"PDF",save_all=True,append_images=pages[1:],resolution=dpi)
    return bio.getvalue()
