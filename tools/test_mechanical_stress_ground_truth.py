"""Validate the synthetic mechanical stress dataset itself before H5 comparison tests."""
from pathlib import Path
import json
import sys
import fitz

ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/"synthetic_mechanical_stress"
EXPECTED={
    "Ø20":"Ø22",
    "125.5":"126.0",
    "R10":"R12",
    "±0.05":"±0.10",
    "AL6061-T6":"AL7075-T6",
}

def text(path):
    doc=fitz.open(path)
    try: return "\n".join(p.get_text("text") for p in doc)
    finally: doc.close()

def main():
    b=DATA/"Before"/"mechanical_before.pdf"; a=DATA/"After"/"mechanical_after.pdf"; g=DATA/"ground_truth.json"
    missing=[str(p) for p in (b,a,g) if not p.exists()]
    if missing:
        print("FAIL: dataset missing. Run: py tools\\generate_mechanical_stress_drawings.py")
        print(*missing,sep="\n"); return 2
    bt,at=text(b),text(a); gt=json.loads(g.read_text(encoding="utf-8"))
    errors=[]
    for old,new in EXPECTED.items():
        if old not in bt: errors.append(f"Before missing expected token: {old}")
        if new not in at: errors.append(f"After missing expected token: {new}")
    if len(gt.get("changes",[])) != 7: errors.append("Ground truth must contain exactly 7 intended changes")
    print("MECHANICAL STRESS DATASET VALIDATION")
    print("Before chars:",len(bt),"After chars:",len(at),"Ground-truth changes:",len(gt.get("changes",[])))
    if errors:
        print("FAIL"); print(*errors,sep="\n- "); return 1
    print("PASS: PDFs and seven known changes are present.")
    print("Next OCR check:")
    print(r'  py tools\run_ocr_diagnostic.py --pdf "synthetic_mechanical_stress\Before\mechanical_before.pdf" --dpi 400')
    print(r'  py tools\run_ocr_diagnostic.py --pdf "synthetic_mechanical_stress\After\mechanical_after.pdf" --dpi 400')
    return 0

if __name__=="__main__": sys.exit(main())
