# Siamese Network for ISIC 2020 Skin Lesion Classification

<img src="./images/Siamese Network.webp" width="400">

## Description

This repository implements a **Siamese Network** for **binary classification** of dermoscopic images from the **ISIC 2020 Challenge** dataset (melanoma vs. benign).  
The approach first trains a **Siamese encoder** using **Triplet Margin Loss** to learn a discriminative embedding space, and then trains a **binary classifier** (4-layer MLP) on top of frozen embeddings for final predictions.  
The implementation follows a modular design, with configuration centralized in `params.py`, dataset management in `dataset.py`, and the main training logic in `train.py`.



## How It Works

### Siamese Encoder
- Backbone: **ResNet-50** pretrained on ImageNet.  
- The final fully connected layer is replaced by a **512-dimensional projection head**.  
- Embeddings are **L2-normalized** to enforce metric consistency.  
- Optimized with **Triplet Margin Loss**, which minimizes the distance between anchor-positive pairs and maximizes distance to negatives.

### Binary Classifier
- Takes embeddings extracted from the Siamese encoder as input.  
- Composed of two hidden layers: 256 → 64 units.  
- Uses **LeakyReLU activation** and **Dropout (p=0.4)** for regularization.  
- Trained with **CrossEntropyLoss** to distinguish between benign and malignant samples.

### Evaluation
- After training, the encoder and classifier are evaluated on the test set.  
- The model reports overall accuracy, confusion matrix, and per-class precision, recall, and F1-score.  
- All plots (training curves, confusion matrix) are saved under `./images/`.



## File Explanations

- **params.py** – Stores all global variables and hyperparameters, including dataset paths, image preprocessing, model dimensions, and training settings.  
- **dataset.py** – Defines dataset classes, data augmentation, and loaders for both triplet and classification tasks.  
- **modules.py** – Contains the model definitions: the Siamese encoder (ResNet-50) and binary classifier (4-layer MLP).  
- **utils.py** – Includes helper functions for plotting, saving figures, feature extraction, and directory creation.  
- **train.py** – Main training script that trains the Siamese encoder, extracts embeddings, and trains the classifier.  
- **predict.py** – Evaluation script that loads trained models, computes predictions, and saves the confusion matrix.


