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
    pdf=SimpleNamespace(dpi=400),
    image=SimpleNamespace(max_image_size=6000, min_image_size=1000, denoise=False),
    align=SimpleNamespace(
        max_features=8000,
        ratio_test=0.75,
        minimum_matches=12,
        minimum_inliers=8,
        ransac_threshold=4.0,
        use_orb=True,
        orb_features=8000,
        use_akaze=True,
    ),
    project=SimpleNamespace(
        recursive_search=True,
        output_folder=OUTPUT_DIR,
        auto_align=True,
        auto_file_match=True,
        auto_page_match=True,
    ),
    page_match=SimpleNamespace(
        minimum_score=0.55,
        review_score=0.40,
        minimum_feature_matches=8,
    ),
    change=SimpleNamespace(
        pixel_threshold=30,
        minimum_area=100,
        merge_distance=15,
        morph_kernel_size=3,
        max_region_ratio=0.60,
    ),
)
