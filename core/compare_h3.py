import fitz
import cv2
import numpy as np
import pandas as pd
import xlsxwriter
import pytesseract
import re
import os
from datetime import datetime
 
 
# -----------------------------
# PDF → 이미지
# -----------------------------
def pdf_to_image(pdf_path):
    doc = fitz.open(pdf_path)
    page = doc[0]
 
    pix = page.get_pixmap(matrix=fitz.Matrix(4, 4))
 
    img = np.frombuffer(pix.tobytes(), dtype=np.uint8)
    img = cv2.imdecode(img, cv2.IMREAD_COLOR)
 
    doc.close()
    return img
 
 
# -----------------------------
# 도면 영역 추출 + 노이즈 제거
# -----------------------------
def preprocess(img):
    h, w = img.shape[:2]
 
    img = img.copy()
 
    # revision block 제거
    img[0:int(h*0.2), int(w*0.75):w] = 255
 
    # title block 제거
    img[int(h*0.88):h, :] = 255
 
    return img
 
 
def extract_main(img):
    h, w = img.shape[:2]
    return img[int(h*0.12):int(h*0.88), int(w*0.05):int(w*0.75)]
 
 
# -----------------------------
# 정렬 (RANSAC)
# -----------------------------
def align(img1, img2):
    orb = cv2.ORB_create(15000)
 
    kp1, des1 = orb.detectAndCompute(img1, None)
    kp2, des2 = orb.detectAndCompute(img2, None)
 
    if des1 is None or des2 is None:
        return img2
 
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)
 
    if len(matches) < 20:
        return img2
 
    matches = sorted(matches, key=lambda x: x.distance)
 
    pts1 = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1,1,2)
    pts2 = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1,1,2)
 
    matrix, _ = cv2.estimateAffinePartial2D(pts2, pts1, method=cv2.RANSAC)
 
    if matrix is None:
        return img2
 
    return cv2.warpAffine(img2, matrix, (img1.shape[1], img1.shape[0]))
 
 
# -----------------------------
# OCR (핵심 H3)
# -----------------------------
def ocr(img):
    try:
        config = "--psm 6"
        return pytesseract.image_to_string(img, config=config)
    except:
        return ""
 
 
# -----------------------------
# 치수 값 추출 (H3 핵심)
# -----------------------------
def extract_dimensions(text):
    # 10, 12, 10.5 같은 숫자 + mm 기반 추출
    pattern = r"(\d+(\.\d+)?)\s*(mm|ø|dia|radius)?"
    return re.findall(pattern, text.lower())
 
 
# -----------------------------
# 변경 유형 분석 (고급)
# -----------------------------
def classify(area, old_text, new_text):
 
    old_dims = extract_dimensions(old_text)
    new_dims = extract_dimensions(new_text)
 
    if old_dims != new_dims and len(old_dims) > 0 and len(new_dims) > 0:
        return "DIMENSION_VALUE_CHANGE"
 
    if "rev" in old_text.lower() or "rev" in new_text.lower():
        return "REVISION_CHANGE"
 
    if area < 120:
        return "TEXT_CHANGE"
 
    if area < 2000:
        return "GEOMETRY_CHANGE"
 
    return "STRUCTURE_CHANGE"
 
 
# -----------------------------
# Excel 생성
# -----------------------------
def create_report():
    wb = xlsxwriter.Workbook("H3_report.xlsx")
    ws = wb.add_worksheet()
 
    ws.write(0,0,"ID")
    ws.write(0,1,"TYPE")
    ws.write(0,2,"OLD_TEXT")
    ws.write(0,3,"NEW_TEXT")
    ws.write(0,4,"TIMESTAMP")
 
    return wb, ws
 
# -----------------------------
# 비교 실행
# -----------------------------
def run(old_path, new_path):
 
    old = pdf_to_image(old_path)
    new = pdf_to_image(new_path)
 
    old = extract_main(preprocess(old))
    new = extract_main(preprocess(new))
 
    new = align(old, new)
 
    diff_img = cv2.absdiff(
        cv2.cvtColor(old, cv2.COLOR_BGR2GRAY),
        cv2.cvtColor(new, cv2.COLOR_BGR2GRAY)
    )
 
    _, thresh = cv2.threshold(diff_img, 30, 255, cv2.THRESH_BINARY)
 
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
 
    result = new.copy()
 
    wb, ws = create_report()
 
    os.makedirs("h3_output", exist_ok=True)
 
    row = 1
    idx = 1
 
    for c in contours:
 
        area = cv2.contourArea(c)
        if area < 120:
            continue
 
        x,y,w,h = cv2.boundingRect(c)
 
        old_crop = old[y:y+h, x:x+w]
        new_crop = new[y:y+h, x:x+w]
 
        old_text = ocr(old_crop)
        new_text = ocr(new_crop)
 
        ctype = classify(area, old_text, new_text)
 
        cv2.rectangle(result, (x,y), (x+w,y+h), (0,0,255), 2)
 
        cv2.imwrite(f"h3_output/change_{idx}.png", new_crop)
 
        ws.write(row,0,idx)
        ws.write(row,1,ctype)
        ws.write(row,2,old_text[:120])
        ws.write(row,3,new_text[:120])
        ws.write(row,4,str(datetime.now()))
 
        row += 1
        idx += 1
 
    cv2.imwrite("H3_result.png", result)
    wb.close()
 
    print("H3 완료")
    print("H3_result.png / H3_report.xlsx 생성")
 
 
# -----------------------------
# 실행
# -----------------------------
if __name__ == "__main__":
    run(r"C:\Users\LGRnD\Desktop\check\Ford_V710_3P12S_BUSBAR_FRAME_FRONT_ASSY_240220_V1.0.pdf", r"C:\Users\LGRnD\Desktop\check\Ford_V710_3P12S_BUSBAR_FRAME_FRONT_ASSY_251201_C_V7.pdf")
 