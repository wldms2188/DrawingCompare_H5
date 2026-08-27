from __future__ import annotations
import threading
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import cv2
import numpy as np
from version import APP_NAME, VERSION
from config import BEFORE_DIR, AFTER_DIR, OUTPUT_DIR
from core.image_loader import ImageLoader
from core.auto_align import AutoAlign, AlignmentResult
from core.change_detector import ChangeDetector
from core.page_matcher import PageMatcher

class DrawingCompareApp(tk.Tk):
    def __init__(self):
        super().__init__(); self.title(f"{APP_NAME} {VERSION}"); self.geometry("900x650"); self.minsize(760,560)
        self.before_dir=tk.StringVar(value=BEFORE_DIR); self.after_dir=tk.StringVar(value=AFTER_DIR); self.output_dir=tk.StringVar(value=OUTPUT_DIR); self.status=tk.StringVar(value="Before / After 폴더를 확인한 뒤 비교를 시작하세요."); self._build()
    def _build(self):
        root=ttk.Frame(self,padding=18); root.pack(fill="both",expand=True); ttk.Label(root,text=f"{APP_NAME} {VERSION}",font=("Segoe UI",20,"bold")).pack(anchor="w"); ttk.Label(root,text="Engineering Drawing PDF Compare",font=("Segoe UI",10)).pack(anchor="w",pady=(0,18)); box=ttk.LabelFrame(root,text="입력 / 출력",padding=12); box.pack(fill="x"); self._path_row(box,"Before",self.before_dir); self._path_row(box,"After",self.after_dir); self._path_row(box,"결과 폴더",self.output_dir); self.run_btn=ttk.Button(root,text="도면 비교 시작",command=self.start); self.run_btn.pack(anchor="e",pady=14); self.progress=ttk.Progressbar(root,mode="indeterminate"); self.progress.pack(fill="x",pady=(0,10)); ttk.Label(root,textvariable=self.status).pack(anchor="w"); logbox=ttk.LabelFrame(root,text="처리 로그",padding=8); logbox.pack(fill="both",expand=True,pady=(12,0)); self.log=tk.Text(logbox,wrap="word",state="disabled",font=("Consolas",9)); self.log.pack(side="left",fill="both",expand=True); scroll=ttk.Scrollbar(logbox,command=self.log.yview); scroll.pack(side="right",fill="y"); self.log.configure(yscrollcommand=scroll.set)
    def _path_row(self,parent,label,variable):
        row=ttk.Frame(parent); row.pack(fill="x",pady=4); ttk.Label(row,text=label,width=10).pack(side="left"); ttk.Entry(row,textvariable=variable).pack(side="left",fill="x",expand=True,padx=6); ttk.Button(row,text="찾아보기",command=lambda:self.choose(variable)).pack(side="right")
    def choose(self,variable):
        path=filedialog.askdirectory();
        if path: variable.set(path)
    def write_log(self,text): self.after(0,self._write_log,text)
    def _write_log(self,text):
        self.log.configure(state="normal"); self.log.insert("end",text+"\n"); self.log.see("end"); self.log.configure(state="disabled")
    def start(self):
        before,after=Path(self.before_dir.get()),Path(self.after_dir.get()); output=Path(self.output_dir.get())
        if not before.is_dir() or not after.is_dir(): messagebox.showwarning("입력 확인","Before와 After 폴더를 모두 선택하세요."); return
        self.run_btn.configure(state="disabled"); self.progress.start(12); self.status.set("PDF를 읽고 있습니다..."); threading.Thread(target=self._worker,args=(before,after,output),daemon=True).start()
    def _worker(self,before_dir,after_dir,output_dir):
        try:
            output_dir.mkdir(parents=True,exist_ok=True); loader=ImageLoader(); before_docs=loader.load_folder(before_dir); after_docs=loader.load_folder(after_dir)
            if not before_docs or not after_docs: raise RuntimeError("Before 또는 After 폴더에 읽을 수 있는 PDF가 없습니다.")
            self.write_log(f"Before PDF {len(before_docs)}개 / After PDF {len(after_docs)}개")
            pairs=self._match_documents(before_docs,after_docs); aligner,detector,page_matcher=AutoAlign(),ChangeDetector(),PageMatcher(); rows=[]; capture_dir=output_dir/"captures"; capture_dir.mkdir(parents=True,exist_ok=True)
            for pair_no,(bd,ad) in enumerate(pairs,1):
                page_matches=page_matcher.match_pages(bd,ad); self.write_log(f"문서 {pair_no}: {bd.filename} ↔ {ad.filename}, 페이지 후보 {len(page_matches)}개")
                for pm in page_matches:
                    if pm.status == "NO_MATCH": self.write_log(f"페이지 보류: B{pm.before_page.page_index+1} ↔ A{pm.after_page.page_index+1}, score={pm.score:.3f}"); continue
                    bp,ap=pm.before_page,pm.after_page; self.status.set(f"비교 중... {bd.filename} p{bp.page_index+1}")
                    try:
                        alignment=aligner.align(bp.image,ap.image)
                        if isinstance(alignment, AlignmentResult):
                            aligned=alignment.image
                            self.write_log(f"정렬: {alignment.method}, success={alignment.success}, scale={alignment.scale:.3f}, rotation={alignment.rotation:.2f}°, valid={alignment.valid_ratio:.2f}")
                        else:
                            aligned=alignment
                    except Exception as exc: self.write_log(f"정렬 경고: {exc}"); aligned=ap.image
                    result=detector.detect(bp,ap,aligned_after=aligned); self.write_log(f"검출 진단 p{bp.page_index+1}: {result.reason}")
                    for region_no,region in enumerate(result.regions,1):
                        stem=f"D{pair_no:02d}_P{bp.page_index+1:03d}_R{region_no:03d}"; old_path,new_path=capture_dir/f"{stem}_before.png",capture_dir/f"{stem}_after.png"; cv2.imwrite(str(old_path),self._to_bgr(region.old_crop)); cv2.imwrite(str(new_path),self._to_bgr(region.new_crop))
                        rows.append({"No":len(rows)+1,"Before PDF":bd.filename,"After PDF":ad.filename,"Before Page":bp.page_index+1,"After Page":ap.page_index+1,"Type":region.region_type,"Confidence":round(region.confidence,3),"X":region.x,"Y":region.y,"Width":region.width,"Height":region.height,"Change Ratio":round(region.change_ratio,4),"Before Image":str(old_path),"After Image":str(new_path),"Before Text":region.old_text,"After Text":region.new_text})
            report=output_dir/"DrawingCompare_H5_Result.xlsx"; self._write_excel(rows,report); self.after(0,lambda:self._finished(report,len(rows)))
        except Exception as exc: self.write_log(f"ERROR: {exc}"); self.after(0,lambda e=str(exc):self._failed(e))
    @staticmethod
    def _to_bgr(img):
        if img is None:return np.full((80,150,3),255,dtype=np.uint8)
        arr=np.asarray(img)
        if arr.ndim==2:return cv2.cvtColor(arr,cv2.COLOR_GRAY2BGR)
        if arr.shape[2]==4:return cv2.cvtColor(arr,cv2.COLOR_RGBA2BGR)
        return arr
    @staticmethod
    def _match_documents(before_docs,after_docs):
        remaining=list(after_docs); pairs=[]
        for bd in before_docs:
            if not remaining:break
            best=min(remaining,key=lambda ad:(0 if Path(bd.filename).stem.lower()==Path(ad.filename).stem.lower() else 10)+abs(bd.page_count-ad.page_count)); pairs.append((bd,best)); remaining.remove(best)
        return pairs
    @staticmethod
    def _write_excel(rows,path):
        from openpyxl import Workbook
        from openpyxl.styles import Font,Alignment
        from openpyxl.drawing.image import Image as XLImage
        wb=Workbook(); ws=wb.active; ws.title="Drawing Compare"; headers=["No","Before PDF","After PDF","Before Page","After Page","Type","Confidence","X","Y","Width","Height","Change Ratio","Before Text","After Text","Before Image","After Image"]
        for c,h in enumerate(headers,1):ws.cell(1,c,h).font=Font(bold=True); ws.cell(1,c).alignment=Alignment(horizontal="center")
        for r,item in enumerate(rows,2):
            for c,h in enumerate(headers,1):ws.cell(r,c,item.get(h,""))
            ws.row_dimensions[r].height=90
            for col,key in ((15,"Before Image"),(16,"After Image")):
                p=item.get(key)
                if p and Path(p).exists():img=XLImage(p); img.width=150; img.height=80; ws.add_image(img,f"{chr(64+col)}{r}")
        wb.save(path)
    def _finished(self,report,count): self.progress.stop(); self.run_btn.configure(state="normal"); self.status.set(f"완료 — 변경영역 {count}개"); messagebox.showinfo("비교 완료",f"비교가 완료되었습니다.\n\n변경영역: {count}개\n결과: {report}")
    def _failed(self,error): self.progress.stop(); self.run_btn.configure(state="normal"); self.status.set("오류가 발생했습니다."); messagebox.showerror("비교 오류",error)

if __name__=="__main__":DrawingCompareApp().mainloop()
