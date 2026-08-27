from pathlib import Path
import fitz

OUT=Path(__file__).resolve().parent/'fixtures'
OUT.mkdir(parents=True,exist_ok=True)

def make(path, after=False):
    doc=fitz.open(); p=doc.new_page(width=842,height=595)
    # Border / title block / simple mechanical geometry.
    p.draw_rect((30,30,812,565),width=1)
    p.draw_circle((250,280),80,color=(0,0,0),width=2)
    p.draw_rect((430,180,620,380),width=2)
    p.draw_line((170,280,330,280),width=1)
    p.draw_line((250,200,250,360),width=1)
    dims=[('Ø22' if after else 'Ø20',210,145),('40' if after else '35',500,145),('R6' if after else 'R5',325,410),('12±0.2' if after else '10±0.2',110,470)]
    for text,x,y in dims: p.insert_text((x,y),text,fontsize=14)
    note='MATERIAL: AL7075' if after else 'MATERIAL: AL6061'
    p.insert_text((430,420),note,fontsize=13)
    p.insert_text((430,445),'REMOVE BURR',fontsize=13)
    p.insert_text((430,470),'ALL DIMENSIONS IN mm',fontsize=13)
    p.insert_text((80,70),'SYNTHETIC MECHANICAL DRAWING',fontsize=12)
    p.insert_text((500,530),'POSITION 0.08 | A' if after else 'POSITION 0.05 | A',fontsize=12)
    doc.save(path); doc.close()

make(OUT/'synthetic_before.pdf',False)
make(OUT/'synthetic_after.pdf',True)
print(OUT)
