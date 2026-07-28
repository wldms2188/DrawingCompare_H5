from pathlib import Path
 
from pdf2image import convert_from_path
 
 
class PDFImageLoader:
 
    def __init__(self, dpi=300):
        self.dpi = dpi
 
    def load(self, pdf_path):
 
        pdf_path = Path(pdf_path)
 
        if not pdf_path.exists():
            raise FileNotFoundError(
                f"PDF 파일을 찾을 수 없습니다.\n{pdf_path}"
            )
 
        pages = convert_from_path(
            pdf_path,
            dpi=self.dpi
        )
 
        print(f"페이지 수 : {len(pages)}")
 
        return pages