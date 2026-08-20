from pathlib import Path

import numpy as np
from PIL import Image, ImageOps


def preprocess_uploaded_image(file_or_path):
    """
    Convert a handwritten digit image into the same 28x28 format used by MNIST.
    Works with common white-background/black-ink images and also MNIST-style
    black-background/white-ink images.
    """
    if isinstance(file_or_path, (str, Path)):
        image = Image.open(file_or_path)
    else:
        image = Image.open(file_or_path)

    image = image.convert("L")

    # Resize large blank margins out of the image.
    arr = np.array(image)
    if arr.max() == arr.min():
        raise ValueError("The uploaded image appears to be blank.")

    # Make foreground white on black, which matches MNIST.
    if arr.mean() > 127:
        arr = 255 - arr

    # Threshold weak background noise.
    arr = np.where(arr > 35, arr, 0).astype(np.uint8)

    coords = np.argwhere(arr > 20)
    if coords.size == 0:
        raise ValueError("No handwritten character was detected.")

    y0, x0 = coords.min(axis=0)
    y1, x1 = coords.max(axis=0) + 1
    cropped = Image.fromarray(arr[y0:y1, x0:x1])

    # Preserve aspect ratio and center the character in a 20x20 canvas.
    cropped.thumbnail((20, 20), Image.Resampling.LANCZOS)
    canvas = Image.new("L", (28, 28), 0)
    left = (28 - cropped.width) // 2
    top = (28 - cropped.height) // 2
    canvas.paste(cropped, (left, top))

    result = np.array(canvas).astype("float32") / 255.0
    return result[..., np.newaxis]


def preprocess_mnist(x):
    x = x.astype("float32") / 255.0
    return x[..., np.newaxis]
