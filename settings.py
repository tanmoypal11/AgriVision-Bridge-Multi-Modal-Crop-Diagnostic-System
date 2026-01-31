from pathlib import Path
import torch

# =========================================
# PROJECT ROOT
# =========================================
BASE_DIR = Path(__file__).resolve().parent

# =========================================
# APP METADATA
# =========================================
APP_NAME = "AgriVision-Bridge"
APP_TAGLINE = "Multi-Modal Crop Diagnostic System"
APP_VERSION = "1.0.0"

# =========================================
# PATHS
# =========================================
WEIGHTS_DIR = BASE_DIR / "weights"
RESULTS_DIR = BASE_DIR / "results"

YOLO_MODEL_PATH = WEIGHTS_DIR / "yolo.pt"

RESULTS_CSV = BASE_DIR / "results.csv"
CONFUSION_MATRIX_PATH = BASE_DIR / "confusion_matrix.png"
RESULTS_PLOT_PATH = BASE_DIR / "results.png"

# =========================================
# MODEL CONFIGURATION
# =========================================
IMAGE_SIZE = 640
CONFIDENCE_THRESHOLD = 0.25

# =========================================
# DEVICE CONFIGURATION
# =========================================
USE_GPU = torch.cuda.is_available()
DEVICE = "cuda" if USE_GPU else "cpu"

# =========================================
# LLM CONFIGURATION (Local / Optional)
# =========================================
LLM_MODEL_NAME = "google/gemma-2b-it"
MAX_NEW_TOKENS = 120
LLM_TEMPERATURE = 0.7

# =========================================
# STREAMLIT SETTINGS
# =========================================
PAGE_CONFIG = {
    "page_title": APP_NAME,
    "page_icon": "🌾",
    "layout": "wide"
}

# =========================================
# BENCHMARKING
# =========================================
BENCHMARK_RUNS = 10
