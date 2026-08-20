import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix

sys.path.append(str(Path(__file__).resolve().parent))

from config import FIGURE_DIR, LABELS_PATH, MODEL_DIR, MODEL_PATH, REPORT_DIR
from model import build_model
from preprocess import preprocess_mnist


SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)


def main():
    MODEL_DIR.mkdir(exist_ok=True)
    REPORT_DIR.mkdir(exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 55)
    print("CodeAlpha - Handwritten Character Recognition")
    print("=" * 55)
    print("Loading MNIST dataset...")

    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

    x_train = preprocess_mnist(x_train)
    x_test = preprocess_mnist(x_test)

    print(f"Training samples: {len(x_train)}")
    print(f"Testing samples : {len(x_test)}")
    print("Image shape     :", x_train.shape[1:])

    model = build_model()
    model.summary()

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=3,
            restore_best_weights=True,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=1,
            min_lr=1e-5,
        ),
    ]

    print("\nTraining CNN...")
    history = model.fit(
        x_train,
        y_train,
        validation_split=0.10,
        epochs=12,
        batch_size=128,
        callbacks=callbacks,
        verbose=1,
    )

    print("\nEvaluating on the unseen test set...")
    test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=0)
    probabilities = model.predict(x_test, batch_size=256, verbose=0)
    predictions = np.argmax(probabilities, axis=1)

    labels = [str(i) for i in range(10)]
    report_text = classification_report(
        y_test,
        predictions,
        target_names=labels,
        digits=4,
        zero_division=0,
    )

    print("\n" + report_text)
    print(f"Test accuracy: {test_accuracy:.4f}")

    with open(REPORT_DIR / "classification_report.txt", "w", encoding="utf-8") as f:
        f.write(f"Test loss: {test_loss:.6f}\n")
        f.write(f"Test accuracy: {test_accuracy:.6f}\n\n")
        f.write(report_text)

    with open(LABELS_PATH, "w", encoding="utf-8") as f:
        json.dump(labels, f)

    cm = confusion_matrix(y_test, predictions)
    plt.figure(figsize=(8, 7))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels,
    )
    plt.xlabel("Predicted Digit")
    plt.ylabel("Actual Digit")
    plt.title("MNIST CNN Confusion Matrix")
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "confusion_matrix.png", dpi=160)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(history.history["accuracy"], label="Training Accuracy")
    plt.plot(history.history["val_accuracy"], label="Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("CNN Training vs Validation Accuracy")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "accuracy_curve.png", dpi=160)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(history.history["loss"], label="Training Loss")
    plt.plot(history.history["val_loss"], label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("CNN Training vs Validation Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "loss_curve.png", dpi=160)
    plt.close()

    model.save(MODEL_PATH)

    print("\nTraining complete.")
    print(f"Best trained model saved to: {MODEL_PATH}")
    print(f"Reports saved to: {REPORT_DIR}")


if __name__ == "__main__":
    main()
