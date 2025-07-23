# MNIST
A repo for testing out pytorch capabilites for MNIST data set with 98% accuracy
# MNIST Classifier with PyTorch

This repository contains a simple yet sexy implementation of a digit classifier using the **MNIST dataset** and **PyTorch**. You feed it images of handwritten digits, and it confidently tells you what digit it is — like Tinder for numbers, but way less disappointing.

---

## Features

- Clean and minimal PyTorch code
- Model training & testing scripts
- GPU support (if CUDA is available)
- Plots for loss/accuracy (because data without plots is just sad)
- Easily tweakable for experiments

---

## Dataset

MNIST is a dataset of **28x28 grayscale images** of handwritten digits from 0 to 9. It contains:

- 60,000 training images  
- 10,000 test images  
- Preprocessed and normalized (because raw digits are dirty)

PyTorch makes it a breeze with `torchvision.datasets.MNIST`.

---

## Model Architecture

A simple fully connected neural net:

