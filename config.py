from pathlib import Path
from types import SimpleNamespace
 
 
BASE_DIR = Path(__file__).resolve().parent
 
 
CONFIG = SimpleNamespace(
 
    pdf=SimpleNamespace(
        dpi=200,
    ),
 
    image=SimpleNamespace(
        max_image_size=3000,
        min_image_size=500,
        denoise=True,
    ),
 
    align=SimpleNamespace(
        use_orb=True,
        orb_features=5000,
        use_akaze=True,
    ),
 
    project=SimpleNamespace(
        recursive_search=True,
        output_folder=str(BASE_DIR / "output"),
 
        auto_align=True,
        auto_file_match=True,
        auto_page_match=True,
    ),
 
    change=SimpleNamespace(
        pixel_threshold=30,
        minimum_area=20,
        merge_distance=10,
    ),
)