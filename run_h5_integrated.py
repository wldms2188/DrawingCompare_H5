from pathlib import Path
 
import numpy as np
 
from core.image_loader import ImageLoader
from core.auto_align import AutoAlign
from core.change_detector import ChangeDetector
 
 
# ============================================================
# CONFIG
# ============================================================
 
BASE_DIR = Path(__file__).resolve().parent
 
BEFORE_DIR = BASE_DIR / "input" / "before"
AFTER_DIR = BASE_DIR / "input" / "after"
 
OUTPUT_DIR = BASE_DIR / "output"
 
 
# ============================================================
# IMAGE EXTRACTION
# ============================================================
 
def get_page_image(page):
 
    # PageImage.image
    if hasattr(page, "image"):
 
        image = page.image
 
    # 혹시 numpy array 자체인 경우
    elif isinstance(page, np.ndarray):
 
        image = page
 
    # fallback
    else:
 
        raise TypeError(
            "페이지 이미지 객체에서 "
            "image 데이터를 찾을 수 없습니다."
        )
 
    if image is None:
 
        raise ValueError(
            "페이지 이미지가 None입니다."
        )
 
    image = np.asarray(
        image
    )
 
    if image.size == 0:
 
        raise ValueError(
            "페이지 이미지가 비어 있습니다."
        )
 
    return image
 
 
# ============================================================
# PAGE LIST
# ============================================================
 
def get_pages(document):
 
    if hasattr(
        document,
        "pages"
    ):
 
        return document.pages
 
    raise TypeError(
        "PDFDocument에서 pages를 "
        "찾을 수 없습니다."
    )
 
 
# ============================================================
# MAIN
# ============================================================
 
def main():
 
    print("=" * 70)
    print("DrawingCompare H5 - Integrated Pipeline")
    print("=" * 70)
 
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )
 
    # --------------------------------------------------------
    # 1. Load
    # --------------------------------------------------------
 
    print()
    print("[1/6] PDF 로딩")
 
    loader = ImageLoader()
 
    before_documents, after_documents = (
        loader.load_before_after(
            BEFORE_DIR,
            AFTER_DIR
        )
    )
 
    print(
        f"Before PDF : "
        f"{len(before_documents)}"
    )
 
    print(
        f"After PDF  : "
        f"{len(after_documents)}"
    )
 
    if not before_documents:
 
        raise RuntimeError(
            "Before PDF가 없습니다."
        )
 
    if not after_documents:
 
        raise RuntimeError(
            "After PDF가 없습니다."
        )
 
    # --------------------------------------------------------
    # 2. Document matching
    # --------------------------------------------------------
 
    print()
    print("[2/6] PDF 대응")
 
    document_count = min(
        len(before_documents),
        len(after_documents)
    )
 
    print(
        f"비교할 문서쌍: "
        f"{document_count}"
    )
 
    aligner = AutoAlign()
 
    detector = ChangeDetector()
 
    total_regions = 0
 
    # --------------------------------------------------------
    # 3. Document loop
    # --------------------------------------------------------
 
    for document_index in range(
        document_count
    ):
 
        before_document = (
            before_documents[
                document_index
            ]
        )
 
        after_document = (
            after_documents[
                document_index
            ]
        )
 
        before_pages = get_pages(
            before_document
        )
 
        after_pages = get_pages(
            after_document
        )
 
        page_count = min(
            len(before_pages),
            len(after_pages)
        )
 
        print()
        print(
            f"Document "
            f"{document_index + 1}: "
            f"{page_count} pages"
        )
 
        # ----------------------------------------------------
        # 4. Page loop
        # ----------------------------------------------------
 
        for page_index in range(
            page_count
        ):
 
            print()
            print(
                f"PAGE "
                f"{page_index + 1}"
            )
 
            before_page = (
                before_pages[
                    page_index
                ]
            )
 
            after_page = (
                after_pages[
                    page_index
                ]
            )
 
            before_image = get_page_image(
                before_page
            )
 
            after_image = get_page_image(
                after_page
            )
 
            print(
                "Before image:",
                before_image.shape
            )
 
            print(
                "After image :",
                after_image.shape
            )
 
            # ------------------------------------------------
            # Alignment
            # ------------------------------------------------
 
            print(
                "  정렬..."
            )
 
            try:
 
                aligned = aligner.align(
                    before_image,
                    after_image
                )
 
            except Exception as exc:
 
                print(
                    "  정렬 실패:"
                )
 
                print(
                    f"  {exc}"
                )
 
                aligned = after_image
 
            if aligned is None:
 
                aligned = after_image
 
            # ------------------------------------------------
            # Change detection
            # ------------------------------------------------
 
            print(
                "  변경영역 검출..."
            )
 
            try:
 
                detection = detector.detect(
                    before_image,
                    aligned
                )
 
            except Exception as exc:
 
                print(
                    "  변경 검출 실패:"
                )
 
                print(
                    f"  {exc}"
                )
 
                continue
 
            # ------------------------------------------------
            # Normalize
            # ------------------------------------------------
 
            if hasattr(
                detection,
                "regions"
            ):
 
                regions = (
                    detection.regions
                )
 
            elif isinstance(
                detection,
                dict
            ):
 
                regions = (
                    detection.get(
                        "regions",
                        []
                    )
                )
 
            elif isinstance(
                detection,
                list
            ):
 
                regions = detection
 
            else:
 
                regions = []
 
            region_count = len(
                regions
            )
 
            total_regions += (
                region_count
            )
 
            print(
                f"  변경영역: "
                f"{region_count}"
            )
 
    # --------------------------------------------------------
    # 5. Summary
    # --------------------------------------------------------
 
    print()
    print("=" * 70)
 
    print(
        "전체 변경영역:",
        total_regions
    )
 
    print(
        "실제 Before / After "
        "페이지 이미지까지 연결 완료"
    )
 
    print("=" * 70)
 
 
# ============================================================
# ENTRY
# ============================================================
 
if __name__ == "__main__":
 
    main()