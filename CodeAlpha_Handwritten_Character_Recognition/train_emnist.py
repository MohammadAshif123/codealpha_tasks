import os
import json
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds


# ============================================================
# EMNIST LETTERS - CNN TRAINING
# CodeAlpha Task 3: Handwritten Character Recognition
# ============================================================

MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "emnist.keras")
LABELS_PATH = os.path.join(MODEL_DIR, "emnist_labels.json")

os.makedirs(MODEL_DIR, exist_ok=True)

print("=" * 60)
print("EMNIST LETTER RECOGNITION - TRAINING")
print("=" * 60)

print("\nTensorFlow version:", tf.__version__)


# ============================================================
# 1. LOAD EMNIST LETTERS
# ============================================================

print("\nDownloading/loading EMNIST Letters dataset...")

(train_ds, test_ds), ds_info = tfds.load(
    "emnist/letters",
    split=["train", "test"],
    as_supervised=True,
    with_info=True
)

print("Dataset loaded successfully.")

print(
    "Training examples:",
    ds_info.splits["train"].num_examples
)

print(
    "Testing examples :",
    ds_info.splits["test"].num_examples
)


# ============================================================
# 2. PREPROCESSING
# ============================================================

def preprocess(image, label):

    # --------------------------------------------------------
    # EMNIST TFDS orientation correction
    # --------------------------------------------------------
    # TFDS EMNIST images are stored rotated/inverted.
    # Transpose converts them to human-friendly orientation.
    image = tf.transpose(
        image,
        perm=[1, 0, 2]
    )

    # Convert to float
    image = tf.cast(
        image,
        tf.float32
    ) / 255.0

    # EMNIST Letters labels are 1-26.
    # Convert to 0-25.
    label = label - 1

    return image, label


train_ds = train_ds.map(
    preprocess,
    num_parallel_calls=tf.data.AUTOTUNE
)

test_ds = test_ds.map(
    preprocess,
    num_parallel_calls=tf.data.AUTOTUNE
)


# ============================================================
# 3. BATCHING
# ============================================================

BATCH_SIZE = 128

train_ds = train_ds.shuffle(
    10000
)

train_ds = train_ds.batch(
    BATCH_SIZE
)

train_ds = train_ds.prefetch(
    tf.data.AUTOTUNE
)

test_ds = test_ds.batch(
    BATCH_SIZE
)

test_ds = test_ds.prefetch(
    tf.data.AUTOTUNE
)


# ============================================================
# 4. CNN MODEL
# ============================================================

print("\nCreating CNN model...")

model = tf.keras.Sequential([

    tf.keras.layers.Input(
        shape=(28, 28, 1)
    ),

    # --------------------------------------------------------
    # Convolution Block 1
    # --------------------------------------------------------

    tf.keras.layers.Conv2D(
        32,
        (3, 3),
        activation="relu",
        padding="same"
    ),

    tf.keras.layers.MaxPooling2D(
        (2, 2)
    ),

    # --------------------------------------------------------
    # Convolution Block 2
    # --------------------------------------------------------

    tf.keras.layers.Conv2D(
        64,
        (3, 3),
        activation="relu",
        padding="same"
    ),

    tf.keras.layers.MaxPooling2D(
        (2, 2)
    ),

    # --------------------------------------------------------
    # Convolution Block 3
    # --------------------------------------------------------

    tf.keras.layers.Conv2D(
        128,
        (3, 3),
        activation="relu",
        padding="same"
    ),

    # --------------------------------------------------------
    # Fully connected
    # --------------------------------------------------------

    tf.keras.layers.Flatten(),

    tf.keras.layers.Dense(
        128,
        activation="relu"
    ),

    tf.keras.layers.Dropout(
        0.3
    ),

    # 26 letters
    tf.keras.layers.Dense(
        26,
        activation="softmax"
    )
])


# ============================================================
# 5. COMPILE
# ============================================================

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()


# ============================================================
# 6. TRAIN
# ============================================================

print("\nStarting EMNIST training...")
print("This may take some time.\n")

EPOCHS = 5

history = model.fit(
    train_ds,
    epochs=EPOCHS,
    validation_data=test_ds
)


# ============================================================
# 7. EVALUATE
# ============================================================

print("\n" + "=" * 60)
print("EVALUATING MODEL")
print("=" * 60)

test_loss, test_accuracy = model.evaluate(
    test_ds
)

print(
    "\nEMNIST Test Accuracy:",
    f"{test_accuracy * 100:.2f}%"
)


# ============================================================
# 8. SAVE MODEL
# ============================================================

model.save(
    MODEL_PATH
)

print("\nModel saved successfully:")
print(MODEL_PATH)


# ============================================================
# 9. SAVE LABELS
# ============================================================

labels = {
    str(i): chr(ord("A") + i)
    for i in range(26)
}

with open(
    LABELS_PATH,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        labels,
        f,
        indent=4
    )

print("\nLabels saved successfully:")
print(LABELS_PATH)


# ============================================================
# 10. SAMPLE PREDICTION
# ============================================================

print("\nRunning sample prediction...")

for images, labels_batch in test_ds.take(1):

    predictions = model.predict(
        images,
        verbose=0
    )

    predicted_class = int(
        np.argmax(predictions[0])
    )

    predicted_letter = chr(
        ord("A") + predicted_class
    )

    actual_class = int(
        labels_batch[0].numpy()
    )

    actual_letter = chr(
        ord("A") + actual_class
    )

    confidence = float(
        np.max(predictions[0])
    ) * 100

    print(
        "\nActual character    :",
        actual_letter
    )

    print(
        "Predicted character :",
        predicted_letter
    )

    print(
        "Confidence          :",
        f"{confidence:.2f}%"
    )

    break


print("\n" + "=" * 60)
print("EMNIST TRAINING COMPLETE!")
print("=" * 60)