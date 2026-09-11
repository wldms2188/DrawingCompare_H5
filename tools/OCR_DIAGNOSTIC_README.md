# H5 독립 OCR 진단 테스트

이 테스트는 기존 `ChangeDetector`와 완전히 분리되어 있습니다.

## 목적

회사 도면 PDF에서 **문자 자체를 실제로 읽을 수 있는지** 먼저 확인합니다.

이 단계에서는 다음을 하지 않습니다.

- Before / After 비교
- Auto Align
- 변경영역 검출
- Hough/Geometry 검출
- 변경값 판정

따라서 결과가 나쁘다면 원인을 `OCR/렌더링/전처리` 쪽으로 좁힐 수 있습니다.

## 실행

VS Code 터미널에서 프로젝트 폴더로 이동한 뒤:

```powershell
cd C:\Users\LGRnD\Desktop\DrawingCompare_H5
py tools\run_ocr_diagnostic.py
```

`--pdf`를 생략하면 `input\before` 안의 첫 번째 PDF를 사용합니다.

특정 PDF:

```powershell
py tools\run_ocr_diagnostic.py --pdf "C:\경로\도면.pdf"
```

2페이지 검사:

```powershell
py tools\run_ocr_diagnostic.py --pdf "C:\경로\도면.pdf" --page 2
```

1200 DPI가 기본입니다. 필요하면 1600 DPI로 별도 진단할 수 있습니다.

```powershell
py tools\run_ocr_diagnostic.py --pdf "C:\경로\도면.pdf" --page 1 --dpi 1600
```

## 결과 위치

기본적으로:

`output\ocr_diagnostic\`

생성 파일:

- `*_ocr_summary.txt` : 터미널과 같은 핵심 진단 결과
- `*_ocr_result.xlsx` : OCR 토큰/좌표/신뢰도 전체 결과
- `*_gray.png` : 원본 회색조 기반 OCR 박스
- `*_otsu.png` : Otsu 이진화 OCR 박스
- `*_adaptive.png` : Adaptive Threshold OCR 박스

## 꼭 확인할 값

터미널 결과에서 아래만 복사해서 알려주면 됩니다.

1. `Engine available`
2. `Image size`
3. `gray / otsu / adaptive`의 `tokens`
4. `high_conf`
5. `NOTE_candidates`
6. `[HIGH-CONFIDENCE SAMPLE]`의 문자 몇 개
7. 오류가 있으면 `reason`

회사 도면 PDF 자체나 캡처 이미지를 공유할 필요가 없습니다.

## 해석 기준

- `Engine available: NO` → OCR 실행환경부터 해결해야 합니다.
- `tokens=0` → 문자를 읽기 전에 렌더링/엔진 문제를 먼저 확인해야 합니다.
- `tokens`는 많지만 실제 도면 문자와 다른 값 → 선/치수선/기호 간섭 또는 OCR 전처리 문제 가능성이 큽니다.
- `high_conf`가 충분하고 실제 문자도 보임 → OCR은 가능성이 있으므로 다음 단계에서 **문자 좌표 기반 Before/After 매칭**으로 넘어갑니다.
- `NOTE_candidates=0`이어도 NOTE 값 자체가 없는 것이 아니라, 현재 진단의 후보 키워드가 검출되지 않았다는 뜻일 수 있습니다.
