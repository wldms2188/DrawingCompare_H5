import cv2
import numpy as np
 
from core.image_loader import PDFImageLoader
from core.auto_align import AutoAlign
from core.change_detector import ChangeDetector
from core.cropper import Cropper
from core.ocr_detector import OCRDetector
from core.excel_report import ExcelReport
 
 
def main():
 
    print("===== DrawingCompare H5 =====")
 
    loader = PDFImageLoader(dpi=300)
    aligner = AutoAlign()
    detector = ChangeDetector()
    cropper = Cropper()
    ocr = OCRDetector()
    report = ExcelReport()
 
    print("PDF 로딩...")
 
    before_pages = loader.load("input/before/sample_before.pdf")
    after_pages = loader.load("input/after/sample_after.pdf")
 
    before = cv2.cvtColor(
        np.array(before_pages[0]),
        cv2.COLOR_RGB2BGR
    )
 
    after = cv2.cvtColor(
        np.array(after_pages[0]),
        cv2.COLOR_RGB2BGR
    )
 
    print("자동 정렬...")
 
    aligned = aligner.align(before, after)
 
    print("변경 검출...")
 
    _, changes = detector.detect(
        before,
        aligned
    )
 
    print(f"변경영역 {len(changes)}개 발견")
 
    print("변경영역 저장...")
 
    changes = cropper.crop_changes(
        before,
        aligned,
        changes
    )
 
    print("OCR 수행...")
 
    changes = ocr.run(changes)
 
    print("Excel 생성...")
 
    report.create(
        changes,
        "output/report.xlsx"
    )
 
    print("완료!")
 
 
if __name__ == "__main__":
    main()