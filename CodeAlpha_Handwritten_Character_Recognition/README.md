# CodeAlpha — Handwritten Character Recognition

## Task 3: Handwritten Character Recognition

### Objective

Identify handwritten characters/digits using image processing and deep learning.

### CodeAlpha requirements covered

- **Dataset:** MNIST
- **Model:** Convolutional Neural Network (CNN)
- Image preprocessing
- Deep learning classification
- Model evaluation
- Handwritten digit prediction

CodeAlpha also notes that the project can be extended to full word/sentence recognition using sequence models. This submission focuses on the required MNIST handwritten-digit recognition task.

## Project pipeline

```text
MNIST handwritten digits
        ↓
Image preprocessing
        ↓
Normalization to 0–1
        ↓
CNN
        ↓
10 digit classes (0–9)
        ↓
Evaluation
        ↓
Upload handwritten image
        ↓
Predicted digit + confidence
```

## Dataset

MNIST contains 60,000 training images and 10,000 test images. Each image is a 28×28 grayscale image representing one of the digits 0–9. The dataset is downloaded automatically through TensorFlow/Keras, so it does not need to be manually placed in this repository.

## Folder structure

```text
CodeAlpha_Handwritten_Character_Recognition/
│
├── app.py
├── requirements.txt
├── README.md
├── PROJECT_REPORT.md
├── run_project.bat
│
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── model.py
│   ├── preprocess.py
│   ├── train.py
│   └── predict.py
│
├── models/
│   ├── mnist_cnn.keras        # generated after training
│   └── labels.json            # generated after training
│
├── reports/
│   ├── classification_report.txt
│   └── figures/
│       ├── accuracy_curve.png
│       ├── loss_curve.png
│       └── confusion_matrix.png
│
├── data/
└── notebooks/
```

## Installation

Open the project folder in VS Code and open a terminal.

Create the virtual environment:

```powershell
python -m venv .venv
```

Activate it:

```powershell
.\.venv\Scripts\activate
```

Install packages:

```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Train the CNN

Run:

```powershell
python src/train.py
```

The first run automatically downloads MNIST through Keras/TensorFlow.

The script trains the CNN and creates:

- test accuracy
- precision
- recall
- F1-score
- classification report
- confusion matrix
- accuracy curve
- loss curve
- trained `.keras` model

## Run a prediction from the terminal

After training:

```powershell
python src/predict.py path\to\your\digit.png
```

Example:

```powershell
python src/predict.py C:\Users\YourName\Pictures\digit7.png
```

## Run the web application

Use the project's Python environment:

```powershell
.\.venv\Scripts\python.exe -m streamlit run app.py
```

Then open the displayed local URL, normally:

```text
http://localhost:8501
```

Upload a clear image containing one handwritten digit and click **Predict Digit**.

## Example output

```text
Predicted Digit: 7
Confidence: 98.6%
```

The exact result depends on the uploaded handwriting.

## Important limitation

This implementation recognizes **single handwritten digits (0–9)** using MNIST. It does not claim to recognize arbitrary handwritten words or sentences.

## Why CNN?

A CNN is appropriate for image classification because convolutional filters learn visual patterns such as edges, curves and digit shapes. Pooling reduces spatial dimensions while retaining useful features.

## Dataset/source

MNIST is the standard handwritten-digit dataset used here. TensorFlow/Keras provides it directly through `tf.keras.datasets.mnist.load_data()`.
