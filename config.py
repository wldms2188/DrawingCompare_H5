from pathlib import Path
 
BASE_DIR = Path(__file__).parent
 
INPUT_DIR = BASE_DIR / "input"
OUTPUT_DIR = BASE_DIR / "output"
CAPTURE_DIR = BASE_DIR / "capture"
TEMP_DIR = BASE_DIR / "temp"
LOG_DIR = BASE_DIR / "logs"
TEMPLATE_DIR = BASE_DIR / "template"
 
SUPPORTED_FILES = [".pdf"]
 
DPI = 300
 
CAPTURE_MARGIN = 40
 
IMAGE_DIFF_THRESHOLD = 20
 
OCR_LANGUAGE = "kor+eng"