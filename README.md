# CNN From Scratch (NumPy)

This project implements a **Convolutional Neural Network (CNN) from scratch**
using **only NumPy**, without relying on any deep learning frameworks
such as PyTorch or TensorFlow.

The main goal is to understand and demonstrate the **low-level mechanics**
of CNNs, including forward propagation, loss computation, and
backpropagation.

---

## Features

- Convolution layers (stride + padding)
- ReLU activation
- Max Pooling
- Fully Connected layers
- Softmax + Cross Entropy Loss
- Manual backpropagation implementation
- Mini-batch training
- Confusion Matrix evaluation
- Model save / load using `pickle`

---

## Architecture
Input (64x64 RGB)
- Conv (3x3, 8 filters) + ReLU
- MaxPool (2x2)
- Conv (3x3, 16 filters) + ReLU
- MaxPool (2x2)
- Fully Connected (64)
- Output (Softmax)

The dataset directory is **not included** in the repository.

Expected dataset structure:

```
MP3_Dataset/
├── train/
│   ├── bicycle/
│   │   └── *.jpg
│   └── cat/
│       └── *.jpg
└── val/
    ├── bicycle/
    │   └── *.jpg
    └── cat/
        └── *.jpg
```


Images are expected to be in `.jpg` format.

## To train the model on a different set of classes, update the `CLASS_NAMES` variable and dataset folder names with your image labels.
        
