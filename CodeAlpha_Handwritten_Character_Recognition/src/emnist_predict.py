import numpy as np
from PIL import Image, ImageOps
import tensorflow as tf
from pathlib import Path


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = (
    BASE_DIR
    / "models"
    / "emnist.keras"
)


# ============================================================
# LOAD MODEL
# ============================================================

model = tf.keras.models.load_model(
    MODEL_PATH
)


# ============================================================
# EMNIST LETTERS
# ============================================================

LETTERS = list(
    "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
)


# ============================================================
# PREPROCESS EMNIST IMAGE
# ============================================================

def preprocess_emnist(image):

    # --------------------------------------------------------
    # Convert Streamlit UploadedFile to PIL Image
    # --------------------------------------------------------

    if not isinstance(image, Image.Image):
        image = Image.open(image)

    image = image.convert("L")

    # --------------------------------------------------------
    # Convert to NumPy
    # --------------------------------------------------------

    img = np.array(image)

    # --------------------------------------------------------
    # Detect dark handwriting
    # --------------------------------------------------------

    mask = img < 200

    if not np.any(mask):
        raise ValueError(
            "No handwritten character detected."
        )

    # --------------------------------------------------------
    # Find bounding box
    # --------------------------------------------------------

    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)

    y1, y2 = np.where(rows)[0][[0, -1]]
    x1, x2 = np.where(cols)[0][[0, -1]]

    # --------------------------------------------------------
    # Crop character
    # --------------------------------------------------------

    cropped = image.crop(
        (
            x1,
            y1,
            x2 + 1,
            y2 + 1,
        )
    )

    # --------------------------------------------------------
    # Convert black handwriting to white
    #
    # Uploaded image:
    #     white background
    #     black character
    #
    # Model input:
    #     black background
    #     white character
    # --------------------------------------------------------

    cropped = ImageOps.invert(
        cropped
    )

    # --------------------------------------------------------
    # Resize while preserving aspect ratio
    # --------------------------------------------------------

    width, height = cropped.size

    scale = 20 / max(
        width,
        height
    )

    new_width = max(
        1,
        int(width * scale)
    )

    new_height = max(
        1,
        int(height * scale)
    )

    cropped = cropped.resize(
        (
            new_width,
            new_height,
        ),
        Image.Resampling.LANCZOS
    )

    # --------------------------------------------------------
    # Create 28x28 black canvas
    # --------------------------------------------------------

    canvas = Image.new(
        "L",
        (28, 28),
        0
    )

    # --------------------------------------------------------
    # Center character
    # --------------------------------------------------------

    x_offset = (
        28 - new_width
    ) // 2

    y_offset = (
        28 - new_height
    ) // 2

    canvas.paste(
        cropped,
        (
            x_offset,
            y_offset,
        )
    )

    # --------------------------------------------------------
    # Convert to NumPy
    # --------------------------------------------------------

    processed = (
        np.array(canvas)
        .astype("float32")
        / 255.0
    )

    # --------------------------------------------------------
    # Add channel dimension
    # Shape:
    #     (28, 28, 1)
    # --------------------------------------------------------

    processed = np.expand_dims(
        processed,
        axis=-1
    )

    # --------------------------------------------------------
    # Add batch dimension
    # Shape:
    #     (1, 28, 28, 1)
    # --------------------------------------------------------

    processed = np.expand_dims(
        processed,
        axis=0
    )

    return processed


# ============================================================
# PREDICT CHARACTER
# ============================================================

def predict_character(image):

    processed = preprocess_emnist(
        image
    )

    # --------------------------------------------------------
    # Prediction
    # --------------------------------------------------------

    probabilities = model.predict(
        processed,
        verbose=0
    )[0]

    # --------------------------------------------------------
    # Find highest probability
    # --------------------------------------------------------

    predicted_index = int(
        np.argmax(probabilities)
    )

    # --------------------------------------------------------
    # Convert index to letter
    # --------------------------------------------------------

    character = LETTERS[
        predicted_index
    ]

    # --------------------------------------------------------
    # Confidence
    #
    # Keep this as 0-1.
    # app.py displays it using :.2%
    # --------------------------------------------------------

    confidence = float(
        probabilities[
            predicted_index
        ]
    )

    return (
        character,
        confidence,
        probabilities
    )