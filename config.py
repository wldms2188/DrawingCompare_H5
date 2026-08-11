from pathlib import Path
from types import SimpleNamespace
 
 
BASE_DIR = Path(__file__).resolve().parent
 
 
config = SimpleNamespace(
 
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
    ),
)
 
 
CONFIG = config
 