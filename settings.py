from pathlib import Path
import sys

# -------------------------------
# PATH SETUP
# -------------------------------
FILE = Path(__file__).resolve()
ROOT = FILE.parent

if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

# -------------------------------
# DIRECTORIES
# -------------------------------
WEIGHTS_DIR = ROOT / "weights"
IMAGES_DIR = ROOT / "images"

# -------------------------------
# DEFAULT IMAGES
# -------------------------------
DEFAULT_IMAGE = IMAGES_DIR / "office_4.jpg"
DEFAULT_DETECT_IMAGE = IMAGES_DIR / "office_4_detected.jpg"

# -------------------------------
# MODEL PATHS
# -------------------------------
DETECTION_MODEL = WEIGHTS_DIR / "yolo.pt"
