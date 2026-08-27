from pathlib import Path
from types import SimpleNamespace

BASE_DIR = Path(__file__).resolve().parent
INPUT_DIR = str(BASE_DIR / "input")
BEFORE_DIR = str(BASE_DIR / "input" / "before")
AFTER_DIR = str(BASE_DIR / "input" / "after")
OUTPUT_DIR = str(BASE_DIR / "output")
CAPTURE_DIR = str(BASE_DIR / "output" / "captures")
TEMP_DIR = str(BASE_DIR / "temp")
LOG_DIR = str(BASE_DIR / "logs")
TEMPLATE_DIR = str(BASE_DIR / "templates")

CONFIG = SimpleNamespace(
    pdf=SimpleNamespace(dpi=200),
    image=SimpleNamespace(max_image_size=3000, min_image_size=500, denoise=True),
    align=SimpleNamespace(use_orb=True, orb_features=5000, use_akaze=True),
    project=SimpleNamespace(
        recursive_search=True,
        output_folder=OUTPUT_DIR,
        auto_align=True,
        auto_file_match=True,
        auto_page_match=True,
    ),
    change=SimpleNamespace(
        pixel_threshold=30,
        minimum_area=100,
        merge_distance=15,
        morph_kernel_size=3,
        max_region_ratio=0.60,
    ),
)
