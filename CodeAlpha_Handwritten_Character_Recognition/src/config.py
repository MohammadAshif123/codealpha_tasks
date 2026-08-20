from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT_DIR / "models"
REPORT_DIR = ROOT_DIR / "reports"
FIGURE_DIR = REPORT_DIR / "figures"

MODEL_PATH = MODEL_DIR / "mnist_cnn.keras"
LABELS_PATH = MODEL_DIR / "labels.json"

IMAGE_SIZE = 28
NUM_CLASSES = 10
