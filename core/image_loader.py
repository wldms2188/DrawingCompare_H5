"""DrawingCompare H5 - high-resolution PDF page/region loader."""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple
import hashlib
import cv2
import fitz
import numpy as np
from config import CONFIG

@dataclass
class PageImage:
    pdf_path: Path
    page_index: int
    image: np.ndarray
    width: int
    height: int
    original_width: int
    original_height: int
    dpi: int
    page_hash: str
    aspect_ratio: float
    rotation: int = 0

@dataclass
class PDFDocument:
    path: Path
    filename: str
    page_count: int
    pages: List[PageImage]

class ImageLoader:
    def __init__(self, config=None):
        self.config = config or CONFIG
        self.dpi = int(getattr(self.config.pdf, "dpi", 400))
        self.max_image_size = int(getattr(self.config.image, "max_image_size", 6000))
        self.min_image_size = int(getattr(self.config.image, "min_image_size", 1000))

    def load_pdf(self, pdf_path: str | Path) -> PDFDocument:
        path = Path(pdf_path); self._validate_pdf(path)
        document = fitz.open(path)
        try:
            pages = [self._render_page(document.load_page(i), path, i) for i in range(document.page_count)]
            return PDFDocument(path, path.name, document.page_count, pages)
        finally: document.close()

    def load_folder(self, folder_path: str | Path) -> List[PDFDocument]:
        folder=Path(folder_path)
        if not folder.exists() or not folder.is_dir(): return []
        return [self.load_pdf(p) for p in sorted(folder.iterdir()) if p.is_file() and p.suffix.lower()==".pdf"]

    def get_page_count(self,pdf_path:str|Path)->int:
        path=Path(pdf_path); self._validate_pdf(path); doc=fitz.open(path)
        try:return doc.page_count
        finally:doc.close()

    def render_region(self,page:PageImage,box:Tuple[int,int,int,int],dpi:int=1200,margin:int=220)->np.ndarray:
        """Render a local region directly from the original PDF vector data.
        The input box is in page.image pixels; no screenshot/upscaling is used.
        """
        x0,y0,x1,y1=[int(v) for v in box]
        x0=max(0,min(page.width-1,x0)); y0=max(0,min(page.height-1,y0))
        x1=max(x0+1,min(page.width,x1)); y1=max(y0+1,min(page.height,y1))
        doc=fitz.open(Path(page.pdf_path))
        try:
            p=doc.load_page(int(page.page_index)); pr=p.rect
            px0=pr.x0+(x0/page.width)*pr.width; py0=pr.y0+(y0/page.height)*pr.height
            px1=pr.x0+(x1/page.width)*pr.width; py1=pr.y0+(y1/page.height)*pr.height
            mx=(margin/page.width)*pr.width; my=(margin/page.height)*pr.height
            clip=fitz.Rect(max(pr.x0,px0-mx),max(pr.y0,py0-my),min(pr.x1,px1+mx),min(pr.y1,py1+my))
            pix=p.get_pixmap(matrix=fitz.Matrix(dpi/72.0,dpi/72.0),clip=clip,alpha=False,colorspace=fitz.csRGB)
            arr=np.frombuffer(pix.samples,dtype=np.uint8).reshape(pix.height,pix.width,3)
            return cv2.cvtColor(arr,cv2.COLOR_RGB2BGR)
        finally:doc.close()

    def _validate_pdf(self,pdf_path:Path)->None:
        if not pdf_path.exists():raise FileNotFoundError(f"PDF 파일을 찾을 수 없습니다: {pdf_path}")
        if not pdf_path.is_file():raise ValueError(f"파일이 아닙니다: {pdf_path}")
        if pdf_path.suffix.lower()!=".pdf":raise ValueError(f"PDF 파일이 아닙니다: {pdf_path}")

    def _render_page(self,page,pdf_path:Path,page_index:int)->PageImage:
        pix=page.get_pixmap(matrix=fitz.Matrix(self.dpi/72.0,self.dpi/72.0),alpha=False,colorspace=fitz.csRGB)
        ow,oh=pix.width,pix.height
        image=np.frombuffer(pix.samples,dtype=np.uint8).reshape(pix.height,pix.width,3)
        image=cv2.cvtColor(image,cv2.COLOR_RGB2BGR); image=self._resize_if_needed(image)
        h,w=image.shape[:2]; thumb=cv2.resize(cv2.cvtColor(image,cv2.COLOR_BGR2GRAY),(128,128),interpolation=cv2.INTER_AREA)
        return PageImage(pdf_path,page_index,image,w,h,ow,oh,self.dpi,hashlib.sha256(thumb.tobytes()).hexdigest(),w/h if h else 0.0,0)

    def _resize_if_needed(self,image):
        h,w=image.shape[:2]; largest=max(h,w)
        if largest<=self.max_image_size:return image
        scale=self.max_image_size/largest
        return cv2.resize(image,(max(int(w*scale),self.min_image_size),max(int(h*scale),self.min_image_size)),interpolation=cv2.INTER_AREA)

    def get_page_metadata(self,page):
        gray=cv2.cvtColor(page.image,cv2.COLOR_BGR2GRAY); binary=cv2.threshold(gray,200,255,cv2.THRESH_BINARY_INV)[1]
        return {"page_index":page.page_index,"width":page.width,"height":page.height,"aspect_ratio":page.aspect_ratio,"page_hash":page.page_hash,"ink_ratio":float(np.count_nonzero(binary)/binary.size),"orientation":0}

    def create_feature_image(self,page,max_size=1600):
        image=page.image; largest=max(image.shape[:2])
        if largest<=max_size:return image.copy()
        scale=max_size/largest
        return cv2.resize(image,(int(image.shape[1]*scale),int(image.shape[0]*scale)),interpolation=cv2.INTER_AREA)
