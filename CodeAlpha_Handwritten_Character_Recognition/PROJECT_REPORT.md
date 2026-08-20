# CodeAlpha Task 3 — Handwritten Character Recognition

## 1. Objective

Build a machine-learning system capable of identifying handwritten characters/digits.

## 2. Dataset

The project uses MNIST, containing 60,000 training images and 10,000 test images. Each image is a 28×28 grayscale handwritten digit from 0 to 9.

## 3. Image preprocessing

The training images are normalized from pixel values 0–255 to 0–1 and reshaped to include a single grayscale channel.

For uploaded images, the application converts the image to grayscale, detects the foreground, crops excess background, scales the digit into a centered 28×28 image, and normalizes it.

## 4. CNN architecture

The model contains:

- Conv2D layers
- Batch Normalization
- Max Pooling
- Dropout
- Dense layer
- Softmax output layer with 10 classes

## 5. Training

The MNIST training set is split into training and validation portions. The separate 10,000-image MNIST test set is used for final evaluation.

## 6. Evaluation

The training script generates:

- Accuracy
- Precision
- Recall
- F1-score
- Confusion matrix
- Training/validation accuracy curve
- Training/validation loss curve

## 7. Application

A Streamlit interface allows a user to upload a handwritten digit image and receive:

- predicted digit
- confidence
- probability for all 10 classes

## 8. Scope

The required task is completed as single handwritten-digit recognition using MNIST. The CodeAlpha description mentions possible extension to words/sentences using sequence models, but that extension is not necessary for this submission.
