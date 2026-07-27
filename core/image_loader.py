from pathlib import Path

from PIL import Image
Image.MAX_IMAGE_PIXELS = None
 
from pdf2image import convert_from_path
 
 
class PDFImageLoader:
    def __init__(self, dpi=300):
        self.dpi = dpi
 
    def load(self, pdf_path):
        pdf_path = Path(pdf_path)
 
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF 파일을 찾을 수 없습니다.\n{pdf_path}")
 
        images = convert_from_path(
            pdf_path,
            dpi=self.dpi,
            poppler_path=r"C:\Users\LGRnD\Downloads\poppler-26.02.0\Library\bin"
        )
 
        return images
 
    def save(self, image, output_path):
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
 
        image.save(output_path)