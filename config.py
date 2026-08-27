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
    # PDF is vector data, so render at high resolution. The original PDF is
    # never rasterized down to a small preview for actual text comparison.
    pdf=SimpleNamespace(dpi=400),

    # Keep analysis images large enough for small dimension/GD&T characters.
    # Feature matching may create its own smaller copy when needed.
    image=SimpleNamespace(
        max_image_size=6000,
        min_image_size=1000,
        denoise=False,
    ),

    # Alignment settings used after the corresponding pages have been found.
    align=SimpleNamespace(
        use_orb=True,
        orb_features=5000,
        use_akaze=True,
    ),

    # Project-level switches.
    project=SimpleNamespace(
        recursive_search=True,
        output_folder=OUTPUT_DIR,
        auto_align=True,
        auto_file_match=True,
        auto_page_match=True,
    ),

    # Page matching settings.
    # page_matcher.py expects these values directly through CONFIG.page_match.
    # A conservative threshold is used so unrelated drawing sheets are not
    # automatically accepted as the same page.
    page_match=SimpleNamespace(
        minimum_score=0.55,
        review_score=0.40,
        minimum_feature_matches=8,
    ),

    # Change detection settings.
    change=SimpleNamespace(
        pixel_threshold=30,
        minimum_area=100,
        merge_distance=15,
        morph_kernel_size=3,
        max_region_ratio=0.60,
    ),
)
