from pathlib import Path
import threading
import tkinter as tk
from tkinter import filedialog,messagebox,ttk
from .engine import H6Engine
from .report import write_report

class App:
    def __init__(self,root):
        self.root=root; root.title('DrawingCompare H6'); root.geometry('760x500')
        self.vars={k:tk.StringVar() for k in ('before','after','output')}
        for i,k in enumerate(self.vars):
            ttk.Label(root,text={'before':'Before PDF 폴더','after':'After PDF 폴더','output':'결과 폴더'}[k]).grid(row=i,column=0,padx=12,pady=10,sticky='w')
            ttk.Entry(root,textvariable=self.vars[k],width=70).grid(row=i,column=1,padx=5,pady=10)
            ttk.Button(root,text='찾기',command=lambda x=k:self.pick(x)).grid(row=i,column=2,padx=8)
        self.runbtn=ttk.Button(root,text='비교 시작',command=self.start); self.runbtn.grid(row=3,column=1,pady=15)
        self.bar=ttk.Progressbar(root,mode='indeterminate'); self.bar.grid(row=4,column=1,sticky='ew',padx=10)
        self.log=tk.Text(root,height=16); self.log.grid(row=5,column=0,columnspan=3,padx=10,pady=10,sticky='nsew'); root.grid_columnconfigure(1,weight=1); root.grid_rowconfigure(5,weight=1)
    def pick(self,k):
        p=filedialog.askdirectory();
        if p:self.vars[k].set(p)
    def write(self,s): self.root.after(0,lambda:(self.log.insert('end',s+'\n'),self.log.see('end')))
    def start(self):
        b,a,o=[Path(self.vars[k].get()) for k in ('before','after','output')]
        if not b.is_dir() or not a.is_dir():messagebox.showerror('확인','Before와 After 폴더를 선택하세요.'); return
        o.mkdir(parents=True,exist_ok=True); self.runbtn.config(state='disabled'); self.bar.start(10); threading.Thread(target=self.worker,args=(b,a,o),daemon=True).start()
    def worker(self,b,a,o):
        try:
            e=H6Engine(); rows=[]
            bfs=sorted(b.glob('*.pdf')); afs=sorted(a.glob('*.pdf')); self.write(f'PDF: Before {len(bfs)} / After {len(afs)}')
            for bp in bfs:
                # filename match first; if unavailable, page-count/visual matching across all files
                same=next((x for x in afs if x.stem.lower()==bp.stem.lower()),None)
                if same is not None: pairs=e.match_pages(e.load_pdf(bp),e.load_pdf(same))
                else:
                    best=None
                    for ap in afs:
                        ps=e.match_pages(e.load_pdf(bp),e.load_pdf(ap)); s=sum(x.score for x in ps)/max(1,len(ps))
                        if best is None or s>best[0]:best=(s,ps,ap)
                    pairs=[] if best is None else best[1]
                for m in pairs:
                    cs=e.compare_match(m)
                    for c in cs: rows.append((bp.name,m.before.index+1,m.after.pdf.name,m.after.index+1,c))
                    self.write(f'{bp.name} p{m.before.index+1} -> {m.after.pdf.name} p{m.after.index+1}: {len(cs)}개 변경')
            out=write_report(rows,o/'DrawingCompare_H6_Result.xlsx'); self.write(f'완료: {out}')
            self.root.after(0,lambda:messagebox.showinfo('완료',f'비교가 끝났습니다.\n{out}'))
        except Exception as ex:self.write(f'오류: {type(ex).__name__}: {ex}'); self.root.after(0,lambda:messagebox.showerror('오류',str(ex)))
        finally:self.root.after(0,lambda:(self.bar.stop(),self.runbtn.config(state='normal')))

def main():
    root=tk.Tk(); App(root); root.mainloop()
if __name__=='__main__':main()
