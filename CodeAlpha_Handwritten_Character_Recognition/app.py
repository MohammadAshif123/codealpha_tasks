import json
import sys
from pathlib import Path

import numpy as np
import streamlit as st
import tensorflow as tf


# ============================================================
# PROJECT PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
SRC_DIR = BASE_DIR / "src"
MODELS_DIR = BASE_DIR / "models"

sys.path.append(str(SRC_DIR))

from config import LABELS_PATH, MODEL_PATH
from preprocess import preprocess_uploaded_image
from emnist_predict import predict_character


# ============================================================
# STREAMLIT CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Handwritten Character Recognition",
    page_icon="✍️",
    layout="centered",
)

st.title("✍️ Handwritten Character Recognition")

st.write(
    "Recognize handwritten digits using MNIST "
    "or handwritten characters using EMNIST."
)


# ============================================================
# RECOGNITION TYPE
# ============================================================

recognition_type = st.radio(
    "Choose Recognition Type",
    [
        "Digits (MNIST)",
        "Characters (EMNIST)",
    ],
    horizontal=True,
)


# ============================================================
# IMAGE UPLOAD
# ============================================================

uploaded_file = st.file_uploader(
    "Upload a handwritten image",
    type=["jpg", "jpeg", "png"],
)


# ============================================================
# DIGIT RECOGNITION - MNIST
# ============================================================

if recognition_type == "Digits (MNIST)":

    st.subheader("🔢 Digit Recognition")

    if not MODEL_PATH.exists():
        st.error(
            "MNIST model is missing.\n\n"
            f"Expected model:\n{MODEL_PATH}"
        )
        st.stop()

    if not LABELS_PATH.exists():
        st.error(
            "MNIST labels file is missing.\n\n"
            f"Expected labels file:\n{LABELS_PATH}"
        )
        st.stop()

    if uploaded_file is not None:

        st.image(
            uploaded_file,
            caption="Uploaded handwritten digit",
            width=250,
        )

        if st.button(
            "🔍 Predict Digit",
            type="primary",
        ):

            try:

                uploaded_file.seek(0)

                processed = preprocess_uploaded_image(
                    uploaded_file
                )

                model = tf.keras.models.load_model(
                    MODEL_PATH
                )

                probabilities = model.predict(
                    processed[np.newaxis, ...],
                    verbose=0,
                )[0]

                index = int(
                    np.argmax(probabilities)
                )

                with open(
                    LABELS_PATH,
                    "r",
                    encoding="utf-8",
                ) as f:
                    labels = json.load(f)

                # Support both dictionary and list labels
                if isinstance(labels, dict):
                    digit = labels.get(
                        str(index),
                        labels.get(index, str(index))
                    )
                else:
                    digit = labels[index]

                confidence = float(
                    probabilities[index]
                )

                st.success(
                    f"Predicted Digit: **{digit}**"
                )

                st.info(
                    f"Confidence: **{confidence:.2%}**"
                )

                st.subheader(
                    "📊 Prediction Probabilities"
                )

                if isinstance(labels, dict):
                    label_list = [
                        labels.get(
                            str(i),
                            labels.get(i, str(i))
                        )
                        for i in range(len(probabilities))
                    ]
                else:
                    label_list = labels

                results = sorted(
                    zip(
                        label_list,
                        probabilities,
                    ),
                    key=lambda x: x[1],
                    reverse=True,
                )

                for label, probability in results:
                    st.write(
                        f"**{label}** — "
                        f"{probability:.2%}"
                    )

            except Exception as exc:

                st.error(
                    "Could not process the image."
                )

                st.exception(exc)

    else:

        st.info(
            "Upload a clear handwritten digit "
            "such as 0, 1, 2, ..., 9."
        )


# ============================================================
# CHARACTER RECOGNITION - EMNIST
# ============================================================

else:

    st.subheader("🔤 Character Recognition")

    EMNIST_MODEL_PATH = (
        MODELS_DIR / "emnist.keras"
    )

    if not EMNIST_MODEL_PATH.exists():

        st.error(
            "EMNIST model is not available yet.\n\n"
            f"Expected file:\n{EMNIST_MODEL_PATH}\n\n"
            "Train the EMNIST model first."
        )

        st.stop()

    if uploaded_file is not None:

        st.image(
            uploaded_file,
            caption="Uploaded handwritten character",
            width=250,
        )

        if st.button(
            "🔍 Predict Character",
            type="primary",
        ):

            try:

                uploaded_file.seek(0)

                character, confidence, probabilities = (
                    predict_character(
                        uploaded_file
                    )
                )

                st.success(
                    f"Predicted Character: "
                    f"**{character}**"
                )

                st.info(
                    f"Confidence: **{confidence:.2%}**"
                )

                st.subheader(
                    "📊 Prediction Probabilities"
                )

                letters = list(
                    "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
                )

                results = sorted(
                    zip(
                        letters,
                        probabilities,
                    ),
                    key=lambda x: x[1],
                    reverse=True,
                )

                for label, probability in results[:10]:

                    st.write(
                        f"**{label}** — "
                        f"{probability:.2%}"
                    )

            except Exception as exc:

                st.error(
                    "Could not process the character."
                )

                st.exception(exc)

    else:

        st.info(
            "Upload a clear handwritten character "
            "such as A, B, C, etc."
        )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "Educational ML Project | "
    "MNIST for handwritten digits | "
    "EMNIST for handwritten characters"
)