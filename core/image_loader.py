"""DrawingCompare H5 - high-resolution PDF page loader."""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import List
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
        path = Path(pdf_path)
        self._validate_pdf(path)
        document = fitz.open(path)
        try:
            page_count = document.page_count
            pages = [self._render_page(document.load_page(i), path, i) for i in range(page_count)]
        finally:
            document.close()
        return PDFDocument(path, path.name, page_count, pages)

    def load_folder(self, folder_path: str | Path) -> List[PDFDocument]:
        """Load every PDF in a folder; retained for the GUI/pipeline API."""
        folder = Path(folder_path)
        if not folder.exists() or not folder.is_dir():
            return []
        files = sorted(
            p for p in folder.iterdir()
            if p.is_file() and p.suffix.lower() == ".pdf"
        )
        return [self.load_pdf(p) for p in files]

    def get_page_count(self, pdf_path: str | Path) -> int:
        path = Path(pdf_path)
        self._validate_pdf(path)
        document = fitz.open(path)
        try:
            return document.page_count
        finally:
            document.close()

    def _validate_pdf(self, pdf_path: Path) -> None:
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF 파일을 찾을 수 없습니다: {pdf_path}")
        if not pdf_path.is_file():
            raise ValueError(f"파일이 아닙니다: {pdf_path}")
        if pdf_path.suffix.lower() != ".pdf":
            raise ValueError(f"PDF 파일이 아닙니다: {pdf_path}")

    def _render_page(self, page, pdf_path: Path, page_index: int) -> PageImage:
        # Render directly from PDF vector data. Never screenshot and upscale.
        matrix = fitz.Matrix(self.dpi / 72.0, self.dpi / 72.0)
        pix = page.get_pixmap(matrix=matrix, alpha=False, colorspace=fitz.csRGB)
        original_width, original_height = pix.width, pix.height
        image = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, 3)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        image = self._resize_if_needed(image)
        h, w = image.shape[:2]
        thumb = cv2.resize(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), (128, 128), interpolation=cv2.INTER_AREA)
        return PageImage(
            pdf_path, page_index, image, w, h, original_width, original_height,
            self.dpi, hashlib.sha256(thumb.tobytes()).hexdigest(), w / h if h else 0.0, 0
        )

    def _resize_if_needed(self, image: np.ndarray) -> np.ndarray:
        h, w = image.shape[:2]
        largest = max(h, w)
        if largest <= self.max_image_size:
            return image
        scale = self.max_image_size / largest
        nw, nh = int(w * scale), int(h * scale)
        return cv2.resize(
            image,
            (max(nw, self.min_image_size), max(nh, self.min_image_size)),
            interpolation=cv2.INTER_AREA,
        )

    def get_page_metadata(self, page: PageImage) -> dict:
        gray = cv2.cvtColor(page.image, cv2.COLOR_BGR2GRAY)
        binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)[1]
        return {
            "page_index": page.page_index,
            "width": page.width,
            "height": page.height,
            "aspect_ratio": page.aspect_ratio,
            "page_hash": page.page_hash,
            "ink_ratio": float(np.count_nonzero(binary) / binary.size),
            "orientation": 0,
        }

    def create_feature_image(self, page: PageImage, max_size: int = 1600) -> np.ndarray:
        image = page.image
        largest = max(image.shape[:2])
        if largest <= max_size:
            return image.copy()
        scale = max_size / largest
        return cv2.resize(
            image,
            (int(image.shape[1] * scale), int(image.shape[0] * scale)),
            interpolation=cv2.INTER_AREA,
        )
