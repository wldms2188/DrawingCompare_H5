from pathlib import Path
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm

from core.image_loader import ImageLoader
from core.change_detector_v2 import ChangeDetector

ROOT=Path(__file__).resolve().parents[1]
TMP=ROOT/"tests"/"_synthetic_runtime"


def make_pdf(path, after=False):
    c=canvas.Canvas(str(path),pagesize=A4)
    x,y,w,h=35*mm,105*mm,120*mm,85*mm
    c.rect(x,y,w,h); c.rect(x+25*mm,y+25*mm,55*mm,30*mm)
    c.line(x+80*mm,y+40*mm,x+105*mm,y+40*mm)
    c.line(x+25*mm,y+18*mm,x+80*mm,y+18*mm)
    c.setFont("Helvetica",9); c.drawCentredString(x+52.5*mm,y+12*mm,"30" if after else "25")
    gx,gy=x+18*mm,y+68*mm; c.rect(gx,gy,38*mm,9*mm); c.line(gx+12*mm,gy,gx+12*mm,gy+9*mm); c.line(gx+25*mm,gy,gx+25*mm,gy+9*mm)
    c.setFont("Helvetica",7); c.drawCentredString(gx+6*mm,gy+2.8*mm,"POS"); c.drawCentredString(gx+18.5*mm,gy+2.8*mm,"0.10" if after else "0.05"); c.drawCentredString(gx+31.5*mm,gy+2.8*mm,"A")
    nx,ny,nw,nh=170*mm,125*mm,28*mm,55*mm; c.rect(nx,ny,nw,nh); ty=ny+nh-8*mm
    for line in ["NOTE","1. DEBURR","2. MATERIAL:","   AL7075" if after else "   AL6061","3. FINISH: NONE"]:
        c.drawString(nx+2*mm,ty,line); ty-=7*mm
    c.drawString(x+88*mm,y+67*mm,"Ø10" if after else "Ø8")
    c.rect(35*mm,45*mm,55*mm,35*mm); c.drawString(40*mm,67*mm,"FIXED"); c.drawString(40*mm,57*mm,"AREA")
    c.save()


def test_synthetic_native_changes():
    TMP.mkdir(parents=True,exist_ok=True)
    before=TMP/"synthetic_before.pdf"; after=TMP/"synthetic_after.pdf"
    make_pdf(before,False); make_pdf(after,True)
    loader=ImageLoader(); bd=loader.load_pdf(before); ad=loader.load_pdf(after)
    assert bd.pages and ad.pages
    result=ChangeDetector().detect(bd.pages[0],ad.pages[0],aligned_after=ad.pages[0].image,alignment_matrix=None)
    assert result.success, result.reason
    values={(r.old_text.strip(),r.new_text.strip()) for r in result.regions if r.old_text and r.new_text}
    assert ("25","30") in values, values
    assert ("0.05","0.10") in values, values
    assert ("AL6061","AL7075") in values, values
    assert ("Ø8","Ø10") in values or ("Ø8","Ø10") in {(a.replace('⌀','Ø'),b.replace('⌀','Ø')) for a,b in values}, values
    assert not any("FIXED" in r.old_text.upper() or "FIXED" in r.new_text.upper() for r in result.regions)
