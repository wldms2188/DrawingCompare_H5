from pathlib import Path
import json
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A3, landscape
from reportlab.lib.units import mm

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "synthetic_mechanical_stress"
BEFORE = OUT / "Before"
AFTER = OUT / "After"
GT = OUT / "ground_truth.json"
PAGE = landscape(A3)

CHANGES = [
    ("DIMENSION", "Ø20", "Ø22", (92, 210, 28, 10)),
    ("DIMENSION", "125.5", "126.0", (145, 265, 35, 10)),
    ("DIMENSION", "R10", "R12", (252, 205, 25, 10)),
    ("TOLERANCE", "±0.05", "±0.10", (152, 195, 35, 10)),
    ("GDT", "POS | 0.05 | A", "POS | 0.10 | A", (235, 255, 62, 12)),
    ("NOTE", "MATERIAL: AL6061-T6", "MATERIAL: AL7075-T6", (315, 174, 76, 10)),
    ("NOTE", "", "6. INSPECT BORE 100%", (315, 139, 76, 10)),
]

def t(c, x, y, s, size=7):
    c.setFont("Helvetica", size); c.drawString(x*mm, y*mm, s)

def line(c, x1,y1,x2,y2): c.line(x1*mm,y1*mm,x2*mm,y2*mm)
def rect(c,x,y,w,h): c.rect(x*mm,y*mm,w*mm,h*mm)

def dimension(c, x1,y1,x2,y2,label, tx,ty):
    line(c,x1,y1,x2,y2)
    # extension/arrow-like ticks
    line(c,x1,y1-3,x1,y1+3); line(c,x2,y2-3,x2,y2+3)
    line(c,x1,y1,x1+3,y1+1.2); line(c,x1,y1,x1+3,y1-1.2)
    line(c,x2,y2,x2-3,y2+1.2); line(c,x2,y2,x2-3,y2-1.2)
    t(c,tx,ty,label,7)

def draw_part(c, changed=False):
    # border
    rect(c,8,8,404,281)
    # FRONT VIEW: flange/bracket-like machined part
    rect(c,55,120,165,105)
    rect(c,72,138,130,69)
    c.circle(105*mm,172*mm,20*mm)
    c.circle(105*mm,172*mm,10*mm)
    c.circle(175*mm,172*mm,8*mm)
    # centerlines
    c.setDash(5,3); line(c,105,130,105,215); line(c,65,172,205,172); c.setDash()
    # hidden lines
    c.setDash(2,2); line(c,170,145,170,199); line(c,180,145,180,199); c.setDash()
    t(c,120,112,"FRONT VIEW",8)

    # SIDE VIEW
    rect(c,245,125,48,95); rect(c,258,145,22,55)
    c.setDash(5,3); line(c,238,172,300,172); c.setDash()
    t(c,250,112,"SIDE VIEW",8)

    # SECTION A-A
    rect(c,55,35,165,55)
    for x in range(58,215,8): line(c,x,36,x+18,89)
    rect(c,90,48,60,28)
    t(c,112,27,"SECTION A-A",8)

    # dimensions / changed values
    dimension(c,75,235,200,235,"126.0" if changed else "125.5",145,265)
    dimension(c,85,218,125,218,"Ø22" if changed else "Ø20",92,210)
    dimension(c,245,225,293,225,"R12" if changed else "R10",252,205)
    t(c,152,195,"±0.10" if changed else "±0.05",8)

    # distractor/repeated dimensions to expose greedy matching errors
    for x,y,s in [(60,245,"20"),(205,245,"20"),(65,101,"10"),(185,101,"10"),(230,95,"0.05"),(275,95,"0.05"),(300,230,"R10")]:
        t(c,x,y,s,7)

    # GD&T frame
    rect(c,232,248,68,12); line(c,252,248,252,260); line(c,278,248,278,260)
    t(c,236,252,"POS",6); t(c,257,252,"0.10" if changed else "0.05",6); t(c,284,252,"A",6)

    # notes block
    rect(c,308,125,94,92); t(c,313,207,"GENERAL NOTES",8)
    notes = [
        "1. REMOVE ALL BURRS",
        "2. UNLESS OTHERWISE SPECIFIED",
        "3. FINISH: NONE",
        "4. MATERIAL: AL7075-T6" if changed else "4. MATERIAL: AL6061-T6",
        "5. INSPECT CRITICAL FEATURES",
    ]
    if changed: notes.append("6. INSPECT BORE 100%")
    yy=196
    for s in notes: t(c,313,yy,s,6); yy-=11

    # title block with dense repeated numbers/text
    rect(c,250,15,152,72)
    for yy in [31,47,63]: line(c,250,yy,402,yy)
    for xx in [300,345,375]: line(c,xx,15,xx,87)
    t(c,254,72,"PART: MOTOR BRACKET",7); t(c,304,72,"DWG NO: H5-TEST-001",6)
    t(c,254,55,"MATERIAL",6); t(c,304,55,"AL7075-T6" if changed else "AL6061-T6",6)
    t(c,254,39,"SCALE",6); t(c,304,39,"1:2",6); t(c,348,39,"REV",6); t(c,378,39,"B" if changed else "A",6)
    t(c,254,23,"DRAWN",6); t(c,304,23,"SYNTHETIC",6); t(c,348,23,"SHEET",6); t(c,378,23,"1/1",6)

def make(path, changed=False, transformed=False):
    c=canvas.Canvas(str(path), pagesize=PAGE)
    if transformed:
        # Intentional small scale/rotation/translation to stress After->Before coordinate mapping.
        c.translate(2.0*mm, -1.5*mm); c.rotate(0.35); c.scale(1.006,1.006)
    draw_part(c, changed)
    c.showPage(); c.save()

def main():
    BEFORE.mkdir(parents=True,exist_ok=True); AFTER.mkdir(parents=True,exist_ok=True)
    make(BEFORE/"mechanical_before.pdf",False,False)
    make(AFTER/"mechanical_after.pdf",True,True)
    gt={"description":"Synthetic A3 mechanical drawing stress test. Coordinates are nominal PDF drawing coordinates in mm before After transform.","after_transform":{"scale":1.006,"rotation_deg":0.35,"translate_mm":[2.0,-1.5]},"changes":[{"id":i+1,"type":typ,"before":old,"after":new,"bbox_mm":{"x":b[0],"y":b[1],"w":b[2],"h":b[3]}} for i,(typ,old,new,b) in enumerate(CHANGES)]}
    GT.write_text(json.dumps(gt,ensure_ascii=False,indent=2),encoding="utf-8")
    print("Created:",BEFORE/"mechanical_before.pdf")
    print("Created:",AFTER/"mechanical_after.pdf")
    print("Ground truth:",GT)

if __name__=="__main__": main()
