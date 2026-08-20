import json
import sys
from pathlib import Path

import numpy as np
import tensorflow as tf

sys.path.append(str(Path(__file__).resolve().parent))

from config import LABELS_PATH, MODEL_PATH
from preprocess import preprocess_uploaded_image


def predict_digit(image_path):
    model = tf.keras.models.load_model(MODEL_PATH)

    with open(LABELS_PATH, "r", encoding="utf-8") as f:
        labels = json.load(f)

    image = preprocess_uploaded_image(image_path)
    probabilities = model.predict(image[np.newaxis, ...], verbose=0)[0]
    index = int(np.argmax(probabilities))

    return labels[index], float(probabilities[index]), probabilities


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python src/predict.py path\\to\\image.png")
        raise SystemExit(1)

    digit, confidence, _ = predict_digit(sys.argv[1])
    print(f"Predicted digit: {digit}")
    print(f"Confidence: {confidence:.2%}")
