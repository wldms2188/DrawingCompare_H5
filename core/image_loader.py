"""
DrawingCompare H5
core/image_loader.py
 
역할
------------------------------------------------------------
1. PDF 파일 존재 여부 확인
2. PDF 페이지 수 자동 확인
3. 모든 페이지를 자동으로 읽기
4. PDF 페이지를 OpenCV 이미지로 변환
5. 페이지별 기본 정보 생성
6. 페이지별 fingerprint 생성
7. 이후 자동 페이지 매칭 / Auto Align에서 사용할 데이터 제공
 
주의
------------------------------------------------------------
- 여러 PDF를 처리할 수 있도록 설계
- PDF 페이지 수를 사용자가 입력할 필요 없음
- 페이지 순서를 기준으로 매칭하지 않음
- 실제 페이지 매칭은 page_matcher.py에서 수행
"""
 
from __future__ import annotations
 
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional
 
import hashlib
 
import cv2
import fitz
import numpy as np
 
from config import CONFIG
 
 
# ============================================================
# DATA CLASS
# ============================================================
 
@dataclass
class PageImage:
    """
    PDF 한 페이지를 표현하는 데이터 구조
    """
 
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
    """
    하나의 PDF 전체를 표현하는 데이터 구조
    """
 
    path: Path
 
    filename: str
 
    page_count: int
 
    pages: List[PageImage]
 
 
# ============================================================
# IMAGE LOADER
# ============================================================
 
class ImageLoader:
    """
    PDF → PageImage 변환 담당
 
    파일 매칭과 페이지 매칭은 담당하지 않는다.
    """
 
    def __init__(self, config=None):
 
        self.config = config or CONFIG
 
        self.dpi = self.config.pdf.dpi
 
        self.max_image_size = (
            self.config.image.max_image_size
        )
 
        self.min_image_size = (
            self.config.image.min_image_size
        )
 
 
    # ========================================================
    # PUBLIC
    # ========================================================
 
    def load_pdf(
        self,
        pdf_path: str | Path
    ) -> PDFDocument:
        """
        PDF 전체를 읽는다.
 
        페이지 수는 PDF에서 자동으로 읽는다.
        """
 
        path = Path(pdf_path)
 
        self._validate_pdf(path)
 
        document = fitz.open(path)
 
        try:
 
            page_count = document.page_count
 
            pages: List[PageImage] = []
 
            for page_index in range(page_count):
 
                page = document.load_page(
                    page_index
                )
 
                page_image = self._render_page(
                    document=document,
                    page=page,
                    pdf_path=path,
                    page_index=page_index
                )
 
                pages.append(page_image)
 
        finally:
 
            document.close()
 
        return PDFDocument(
            path=path,
            filename=path.name,
            page_count=page_count,
            pages=pages
        )
 
 
    # ========================================================
    # PAGE COUNT
    # ========================================================
 
    def get_page_count(
        self,
        pdf_path: str | Path
    ) -> int:
        """
        PDF 페이지 수를 자동으로 반환한다.
        """
 
        path = Path(pdf_path)
 
        self._validate_pdf(path)
 
        document = fitz.open(path)
 
        try:
 
            return document.page_count
 
        finally:
 
            document.close()
 
 
    # ========================================================
    # PDF VALIDATION
    # ========================================================
 
    def _validate_pdf(
        self,
        pdf_path: Path
    ) -> None:
 
        if not pdf_path.exists():
 
            raise FileNotFoundError(
                f"PDF 파일을 찾을 수 없습니다: "
                f"{pdf_path}"
            )
 
        if not pdf_path.is_file():
 
            raise ValueError(
                f"파일이 아닙니다: {pdf_path}"
            )
 
        if pdf_path.suffix.lower() != ".pdf":
 
            raise ValueError(
                f"PDF 파일이 아닙니다: {pdf_path}"
            )
 
 
    # ========================================================
    # RENDER PAGE
    # ========================================================
 
    def _render_page(
        self,
        document,
        page,
        pdf_path: Path,
        page_index: int
    ) -> PageImage:
        """
        PDF 페이지 하나를 OpenCV 이미지로 변환한다.
        """
 
        matrix = fitz.Matrix(
            self.dpi / 72.0,
            self.dpi / 72.0
        )
 
        pixmap = page.get_pixmap(
            matrix=matrix,
            alpha=False
        )
 
        original_width = pixmap.width
 
        original_height = pixmap.height
 
        image = np.frombuffer(
            pixmap.samples,
            dtype=np.uint8
        )
 
        image = image.reshape(
            pixmap.height,
            pixmap.width,
            pixmap.n
        )
 
        if pixmap.n == 4:
 
            image = cv2.cvtColor(
                image,
                cv2.COLOR_RGBA2BGR
            )
 
        elif pixmap.n == 3:
 
            image = cv2.cvtColor(
                image,
                cv2.COLOR_RGB2BGR
            )
 
        else:
 
            image = cv2.cvtColor(
                image,
                cv2.COLOR_GRAY2BGR
            )
 
        image = self._preprocess(image)
 
        height, width = image.shape[:2]
 
        page_hash = self._make_hash(image)
 
        aspect_ratio = (
            width / height
            if height > 0
            else 0.0
        )
 
        return PageImage(
            pdf_path=pdf_path,
            page_index=page_index,
            image=image,
            width=width,
            height=height,
            original_width=original_width,
            original_height=original_height,
            dpi=self.dpi,
            page_hash=page_hash,
            aspect_ratio=aspect_ratio,
            rotation=0
        )
 
 
    # ========================================================
    # PREPROCESS
    # ========================================================
 
    def _preprocess(
        self,
        image: np.ndarray
    ) -> np.ndarray:
        """
        페이지 이미지의 기본 전처리.
 
        여기서는 비교에 필요한 원본 구조를 최대한 유지한다.
        강한 이진화 등은 이후 비교 단계에서 수행한다.
        """
 
        if image is None:
 
            raise ValueError(
                "이미지를 생성하지 못했습니다."
            )
 
        if image.size == 0:
 
            raise ValueError(
                "빈 이미지가 생성되었습니다."
            )
 
        image = self._resize_if_needed(
            image
        )
 
        if self.config.image.denoise:
 
            image = cv2.GaussianBlur(
                image,
                (3, 3),
                0
            )
 
        return image
 
 
    # ========================================================
    # RESIZE
    # ========================================================
 
    def _resize_if_needed(
        self,
        image: np.ndarray
    ) -> np.ndarray:
        """
        지나치게 큰 PDF 페이지를 안전한 크기로 줄인다.
 
        원본 PDF 자체는 변경하지 않는다.
        """
 
        height, width = image.shape[:2]
 
        max_size = self.max_image_size
 
        current_max = max(
            width,
            height
        )
 
        if current_max <= max_size:
 
            return image
 
        scale = (
            max_size / current_max
        )
 
        new_width = max(
            int(width * scale),
            self.min_image_size
        )
 
        new_height = max(
            int(height * scale),
            self.min_image_size
        )
 
        return cv2.resize(
            image,
            (
                new_width,
                new_height
            ),
            interpolation=cv2.INTER_AREA
        )
 
 
    # ========================================================
    # HASH
    # ========================================================
 
    def _make_hash(
        self,
        image: np.ndarray
    ) -> str:
        """
        페이지 fingerprint용 기본 hash.
 
        단순 변경 검출용이 아니라
        동일 이미지 재처리 방지용으로 사용한다.
        """
 
        gray = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2GRAY
        )
 
        small = cv2.resize(
            gray,
            (128, 128),
            interpolation=cv2.INTER_AREA
        )
 
        return hashlib.sha256(
            small.tobytes()
        ).hexdigest()
    # ========================================================
    # PAGE METADATA
    # ========================================================
 
    def get_page_metadata(
        self,
        page: PageImage
    ) -> dict:
        """
        페이지 자동 매칭에 사용할 기본 특징값을 계산한다.
 
        여기서 계산하는 값은
        파일/페이지 매칭의 후보를 줄이는 용도다.
        """
 
        image = page.image
 
        gray = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2GRAY
        )
 
        metadata = {
            "page_index": page.page_index,
            "width": page.width,
            "height": page.height,
            "aspect_ratio": page.aspect_ratio,
            "page_hash": page.page_hash,
            "ink_ratio": self._calculate_ink_ratio(gray),
            "border": self._detect_border(gray),
            "orientation": self._detect_orientation(gray),
        }
 
        return metadata
 
 
    # ========================================================
    # INK RATIO
    # ========================================================
 
    def _calculate_ink_ratio(
        self,
        gray: np.ndarray
    ) -> float:
        """
        도면에서 실제 선/문자 등이 차지하는 비율을 계산한다.
 
        페이지 크기가 달라도 비율을 이용하면
        페이지 매칭에 활용할 수 있다.
        """
 
        if gray.size == 0:
 
            return 0.0
 
        _, binary = cv2.threshold(
            gray,
            200,
            255,
            cv2.THRESH_BINARY_INV
        )
 
        ink_pixels = np.count_nonzero(
            binary
        )
 
        total_pixels = binary.size
 
        if total_pixels == 0:
 
            return 0.0
 
        return (
            ink_pixels /
            total_pixels
        )
 
 
    # ========================================================
    # BORDER DETECTION
    # ========================================================
 
    def _detect_border(
        self,
        gray: np.ndarray
    ) -> dict:
        """
        도면 테두리 위치를 추정한다.
 
        실제 Auto Align에서는
        feature matching 결과와 함께 사용한다.
        """
 
        height, width = gray.shape[:2]
 
        threshold = 200
 
        binary = cv2.threshold(
            gray,
            threshold,
            255,
            cv2.THRESH_BINARY_INV
        )[1]
 
        horizontal_projection = np.sum(
            binary > 0,
            axis=1
        )
 
        vertical_projection = np.sum(
            binary > 0,
            axis=0
        )
 
        top = self._find_border_position(
            horizontal_projection,
            height,
            from_start=True
        )
 
        bottom = self._find_border_position(
            horizontal_projection,
            height,
            from_start=False
        )
 
        left = self._find_border_position(
            vertical_projection,
            width,
            from_start=True
        )
 
        right = self._find_border_position(
            vertical_projection,
            width,
            from_start=False
        )
 
        return {
            "top": top,
            "bottom": bottom,
            "left": left,
            "right": right
        }
 
 
    # ========================================================
    # BORDER POSITION
    # ========================================================
 
    def _find_border_position(
        self,
        projection: np.ndarray,
        size: int,
        from_start: bool
    ) -> int:
        """
        Projection을 이용하여 테두리 후보 위치를 찾는다.
        """
 
        if size <= 0:
 
            return 0
 
        minimum_density = (
            size * 0.15
        )
 
        indices = np.where(
            projection >= minimum_density
        )[0]
 
        if len(indices) == 0:
 
            return 0 if from_start else size - 1
 
        if from_start:
 
            return int(indices[0])
 
        return int(indices[-1])
 
 
    # ========================================================
    # ORIENTATION
    # ========================================================
 
    def _detect_orientation(
        self,
        gray: np.ndarray
    ) -> int:
        """
        페이지의 기본 방향을 추정한다.
 
        반환값:
            0   = 기본
            90  = 90도
            180 = 180도
            270 = 270도
 
        초기 단계에서는 과도한 자동 회전을 방지하기 위해
        확실한 경우에만 회전값을 반환한다.
        """
 
        height, width = gray.shape[:2]
 
        if height == 0 or width == 0:
 
            return 0
 
        # 도면의 가로/세로 방향 자체만으로
        # 회전을 확정하면 잘못된 판단이 생길 수 있다.
        #
        # 따라서 여기서는 강제 회전을 하지 않고
        # 기본 방향을 반환한다.
        #
        # 실제 회전 판단은 AutoAlign 단계에서
        # feature matching을 통해 결정한다.
 
        return 0
 
 
    # ========================================================
    # FEATURE IMAGE
    # ========================================================
 
    def create_feature_image(
        self,
        page: PageImage,
        max_size: int = 1600
    ) -> np.ndarray:
        """
        페이지 매칭용 축소 이미지를 생성한다.
 
        원본 image는 절대 변경하지 않는다.
        """
 
        image = page.image
 
        height, width = image.shape[:2]
 
        current_max = max(
            width,
            height
        )
 
        if current_max <= max_size:
 
            return image.copy()
 
        scale = (
            max_size /
            current_max
        )
 
        new_width = max(
            int(width * scale),
            1
        )
 
        new_height = max(
            int(height * scale),
            1
        )
 
        return cv2.resize(
            image,
            (
                new_width,
                new_height
            ),
            interpolation=cv2.INTER_AREA
        )
 
 
    # ========================================================
    # FEATURE POINTS
    # ========================================================
 
    def extract_feature_points(
        self,
        page: PageImage
    ) -> dict:
        """
        페이지의 특징점을 추출한다.
 
        ORB와 AKAZE를 모두 준비한다.
 
        실제 어느 결과를 채택할지는
        page_matcher / aligner에서 결정한다.
        """
 
        feature_image = self.create_feature_image(
            page
        )
 
        gray = cv2.cvtColor(
            feature_image,
            cv2.COLOR_BGR2GRAY
        )
 
        result = {
            "orb_keypoints": [],
            "orb_descriptors": None,
            "akaze_keypoints": [],
            "akaze_descriptors": None,
        }
 
        # ----------------------------------------------------
        # ORB
        # ----------------------------------------------------
 
        if self.config.align.use_orb:
 
            orb = cv2.ORB_create(
                nfeatures=self.config.align.orb_features
            )
 
            keypoints, descriptors = (
                orb.detectAndCompute(
                    gray,
                    None
                )
            )
 
            if keypoints:
 
                result[
                    "orb_keypoints"
                ] = keypoints
 
                result[
                    "orb_descriptors"
                ] = descriptors
 
 
        # ----------------------------------------------------
        # AKAZE
        # ----------------------------------------------------
 
        if self.config.align.use_akaze:
 
            akaze = cv2.AKAZE_create()
 
            keypoints, descriptors = (
                akaze.detectAndCompute(
                    gray,
                    None
                )
            )
 
            if keypoints:
 
                result[
                    "akaze_keypoints"
                ] = keypoints
 
                result[
                    "akaze_descriptors"
                ] = descriptors
 
 
        return result
 
 
    # ========================================================
    # PAGE SUMMARY
    # ========================================================
 
    def create_page_summary(
        self,
        page: PageImage
    ) -> dict:
        """
        페이지 하나를 자동 매칭하기 위한
        요약 정보를 생성한다.
        """
 
        metadata = self.get_page_metadata(
            page
        )
 
        features = self.extract_feature_points(
            page
        )
 
        metadata["orb_count"] = len(
            features["orb_keypoints"]
        )
 
        metadata["akaze_count"] = len(
            features["akaze_keypoints"]
        )
 
        return metadata
 
    # ========================================================
    # MULTI PDF LOADING
    # ========================================================
 
    def load_folder(
        self,
        folder_path: str | Path
    ) -> List[PDFDocument]:
        """
        폴더 안의 모든 PDF를 자동으로 읽는다.
 
        하위 폴더까지 검색할 수 있다.
 
        반환:
            PDFDocument 리스트
        """
 
        folder = Path(folder_path)
 
        if not folder.exists():
 
            raise FileNotFoundError(
                f"폴더를 찾을 수 없습니다: "
                f"{folder}"
            )
 
        if not folder.is_dir():
 
            raise ValueError(
                f"폴더가 아닙니다: {folder}"
            )
 
        if self.config.project.recursive_search:
 
            pdf_files = sorted(
                folder.rglob("*.pdf")
            )
 
        else:
 
            pdf_files = sorted(
                folder.glob("*.pdf")
            )
 
        documents: List[PDFDocument] = []
 
        for pdf_path in pdf_files:
 
            try:
 
                document = self.load_pdf(
                    pdf_path
                )
 
                documents.append(
                    document
                )
 
            except Exception as exc:
 
                # 한 개의 PDF가 문제가 있어도
                # 전체 Batch 작업이 중단되지 않도록 한다.
                print(
                    f"[WARNING] PDF 로드 실패: "
                    f"{pdf_path}"
                )
 
                print(
                    f"           원인: {exc}"
                )
 
        return documents
 
 
    # ========================================================
    # MULTI PDF SUMMARY
    # ========================================================
 
    def create_document_summary(
        self,
        document: PDFDocument
    ) -> dict:
        """
        하나의 PDF에 대한 전체 요약 정보를 생성한다.
 
        파일 매칭 단계에서 사용한다.
        """
 
        page_summaries = []
 
        for page in document.pages:
 
            summary = self.create_page_summary(
                page
            )
 
            page_summaries.append(
                summary
            )
 
        return {
            "filename": document.filename,
            "path": str(document.path),
            "page_count": document.page_count,
            "pages": page_summaries
        }
 
 
    # ========================================================
    # CACHE KEY
    # ========================================================
 
    def create_cache_key(
        self,
        pdf_path: str | Path
    ) -> str:
        """
        PDF 파일의 변경 여부를 판단하기 위한
        cache key를 생성한다.
 
        같은 이름이라도 파일 내용이 변경되면
        다른 key가 생성된다.
        """
 
        path = Path(pdf_path)
 
        if not path.exists():
 
            raise FileNotFoundError(
                path
            )
 
        stat = path.stat()
 
        raw = (
            f"{path.resolve()}|"
            f"{stat.st_size}|"
            f"{stat.st_mtime_ns}|"
            f"{self.dpi}"
        )
 
        return hashlib.sha256(
            raw.encode("utf-8")
        ).hexdigest()
 
 
    # ========================================================
    # CACHE PATH
    # ========================================================
 
    def get_cache_path(
        self,
        pdf_path: str | Path
    ) -> Path:
        """
        PDF별 캐시 파일 경로를 반환한다.
        """
 
        cache_key = self.create_cache_key(
            pdf_path
        )
 
        cache_dir = (
            Path(self.config.project.output_folder)
            / "cache"
        )
 
        cache_dir.mkdir(
            parents=True,
            exist_ok=True
        )
 
        return (
            cache_dir /
            f"{cache_key}.json"
        )
 
 
    # ========================================================
    # DOCUMENT IDENTIFIER
    # ========================================================
 
    def create_document_identifier(
        self,
        document: PDFDocument
    ) -> str:
        """
        PDF 전체를 대표하는 fingerprint를 만든다.
 
        파일명이 바뀌어도 페이지 내용이 같으면
        유사한 문서임을 판단할 수 있도록
        페이지 fingerprint를 기반으로 한다.
        """
 
        page_hashes = [
            page.page_hash
            for page in document.pages
        ]
 
        raw = "|".join(
            page_hashes
        )
 
        return hashlib.sha256(
            raw.encode("utf-8")
        ).hexdigest()
 
 
    # ========================================================
    # PAGE IDENTIFIER
    # ========================================================
 
    def create_page_identifier(
        self,
        page: PageImage
    ) -> str:
        """
        페이지 fingerprint를 반환한다.
        """
 
        raw = (
            f"{page.page_hash}|"
            f"{page.width}|"
            f"{page.height}|"
            f"{page.aspect_ratio:.6f}"
        )
 
        return hashlib.sha256(
            raw.encode("utf-8")
        ).hexdigest()
 
 
    # ========================================================
    # IMAGE COPY
    # ========================================================
 
    @staticmethod
    def copy_image(
        image: np.ndarray
    ) -> np.ndarray:
        """
        원본 이미지가 수정되는 것을 방지하기 위한
        안전한 복사.
        """
 
        if image is None:
 
            raise ValueError(
                "복사할 이미지가 없습니다."
            )
 
        return image.copy()
 
 
    # ========================================================
    # IMAGE VALIDATION
    # ========================================================
 
    @staticmethod
    def validate_image(
        image: Optional[np.ndarray]
    ) -> bool:
        """
        이미지가 정상적인 OpenCV 이미지인지 확인한다.
        """
 
        if image is None:
 
            return False
 
        if not isinstance(
            image,
            np.ndarray
        ):
 
            return False
 
        if image.size == 0:
 
            return False
 
        if image.ndim not in (2, 3):
 
            return False
 
        return True
 
 
    # ========================================================
    # SAFE PAGE ACCESS
    # ========================================================
 
    @staticmethod
    def get_page(
        document: PDFDocument,
        page_index: int
    ) -> Optional[PageImage]:
        """
        페이지 번호가 범위를 벗어나도
        프로그램이 죽지 않도록 안전하게 접근한다.
        """
 
        if page_index < 0:
 
            return None
 
        if page_index >= document.page_count:
 
            return None
 
        return document.pages[
            page_index
        ]
 
 
    # ========================================================
    # ITERATE PAGES
    # ========================================================
 
    @staticmethod
    def iter_pages(
        document: PDFDocument
    ):
        """
        PDF 페이지를 순서대로 반환한다.
 
        페이지 매칭 자체는 수행하지 않는다.
        """
 
        for page in document.pages:
 
            yield page
 
    # ========================================================
    # BATCH SUMMARY
    # ========================================================
 
    def create_folder_summary(
        self,
        folder_path: str | Path
    ) -> dict:
        """
        폴더 내 모든 PDF를 읽고
        전체 구조를 요약한다.
 
        파일 매칭 단계에서 사용한다.
        """
 
        documents = self.load_folder(
            folder_path
        )
 
        summary = {
            "folder": str(
                Path(folder_path).resolve()
            ),
            "document_count": len(
                documents
            ),
            "total_page_count": 0,
            "documents": []
        }
 
        for document in documents:
 
            document_summary = (
                self.create_document_summary(
                    document
                )
            )
 
            document_summary[
                "document_identifier"
            ] = self.create_document_identifier(
                document
            )
 
            summary[
                "documents"
            ].append(
                document_summary
            )
 
            summary[
                "total_page_count"
            ] += document.page_count
 
        return summary
 
 
    # ========================================================
    # CHECK PDF
    # ========================================================
 
    def check_pdf(
        self,
        pdf_path: str | Path
    ) -> dict:
        """
        PDF를 실제로 열어볼 수 있는지 검사한다.
 
        프로그램 시작 시 입력 파일 검증에 사용한다.
        """
 
        path = Path(pdf_path)
 
        result = {
            "path": str(path),
            "exists": False,
            "valid": False,
            "page_count": 0,
            "error": None
        }
 
        if not path.exists():
 
            result["error"] = (
                "파일이 존재하지 않습니다."
            )
 
            return result
 
        result["exists"] = True
 
        if path.suffix.lower() != ".pdf":
 
            result["error"] = (
                "PDF 파일이 아닙니다."
            )
 
            return result
 
        try:
 
            document = fitz.open(path)
 
            result["page_count"] = (
                document.page_count
            )
 
            document.close()
 
            result["valid"] = True
 
        except Exception as exc:
 
            result["error"] = str(exc)
 
        return result
 
 
    # ========================================================
    # CHECK FOLDER
    # ========================================================
 
    def check_folder(
        self,
        folder_path: str | Path
    ) -> dict:
        """
        Before / After 폴더를 검사한다.
        """
 
        folder = Path(folder_path)
 
        result = {
            "path": str(folder),
            "exists": folder.exists(),
            "is_directory": folder.is_dir(),
            "pdf_count": 0,
            "files": [],
            "errors": []
        }
 
        if not folder.exists():
 
            return result
 
        if not folder.is_dir():
 
            return result
 
        if self.config.project.recursive_search:
 
            pdf_files = sorted(
                folder.rglob("*.pdf")
            )
 
        else:
 
            pdf_files = sorted(
                folder.glob("*.pdf")
            )
 
        result["pdf_count"] = len(
            pdf_files
        )
 
        for pdf_file in pdf_files:
 
            check = self.check_pdf(
                pdf_file
            )
 
            result[
                "files"
            ].append(
                check
            )
 
            if not check["valid"]:
 
                result[
                    "errors"
                ].append(
                    check
                )
 
        return result
 
 
    # ========================================================
    # LOAD BOTH SIDES
    # ========================================================
 
    def load_before_after(
        self,
        before_folder: str | Path,
        after_folder: str | Path
    ) -> tuple[
        List[PDFDocument],
        List[PDFDocument]
    ]:
        """
        Before / After 폴더를 모두 읽는다.
 
        파일 개수와 페이지 개수가 달라도 허용한다.
 
        실제 매칭은 이후 단계에서 수행한다.
        """
 
        before_documents = (
            self.load_folder(
                before_folder
            )
        )
 
        after_documents = (
            self.load_folder(
                after_folder
            )
        )
 
        return (
            before_documents,
            after_documents
        )
 
 
    # ========================================================
    # TOTAL PAGE COUNT
    # ========================================================
 
    @staticmethod
    def get_total_page_count(
        documents: List[PDFDocument]
    ) -> int:
        """
        여러 PDF의 전체 페이지 수를 반환한다.
        """
 
        return sum(
            document.page_count
            for document in documents
        )
 
 
    # ========================================================
    # DOCUMENT INFO
    # ========================================================
 
    @staticmethod
    def get_document_info(
        document: PDFDocument
    ) -> dict:
        """
        PDF 기본 정보를 반환한다.
        """
 
        return {
            "filename": document.filename,
            "path": str(document.path),
            "page_count": document.page_count,
            "widths": [
                page.width
                for page in document.pages
            ],
            "heights": [
                page.height
                for page in document.pages
            ],
            "hashes": [
                page.page_hash
                for page in document.pages
            ]
        }
 
 
# ============================================================
# DEFAULT LOADER
# ============================================================
 
_default_loader = ImageLoader()
 
 
def load_pdf(
    pdf_path: str | Path
) -> PDFDocument:
    """
    외부 모듈에서 간단하게 PDF를 읽을 수 있도록
    제공하는 함수.
    """
 
    return _default_loader.load_pdf(
        pdf_path
    )
 
 
def load_folder(
    folder_path: str | Path
) -> List[PDFDocument]:
    """
    외부 모듈에서 폴더 전체 PDF를 읽는다.
    """
 
    return _default_loader.load_folder(
        folder_path
    )
 
 
def get_page_count(
    pdf_path: str | Path
) -> int:
    """
    외부 모듈에서 페이지 수만 확인한다.
    """
 
    return _default_loader.get_page_count(
        pdf_path
    )
 
 
# ============================================================
# TEST
# ============================================================
 
if __name__ == "__main__":
 
    print("=" * 60)
    print("DrawingCompare H5 - Image Loader Test")
    print("=" * 60)
 
    print(
        "image_loader.py 로드 성공"
    )
 
    print(
        f"PDF DPI : "
        f"{CONFIG.pdf.dpi}"
    )
 
    print(
        f"Auto Align : "
        f"{CONFIG.project.auto_align}"
    )
 
    print(
        f"Auto Page Match : "
        f"{CONFIG.project.auto_page_match}"
    )
 
    print(
        f"Auto File Match : "
        f"{CONFIG.project.auto_file_match}"
    )
 
    print("=" * 60)
 
 