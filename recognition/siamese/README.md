# Siamese Network for ISIC 2020 Skin Lesion Classification

<p align="center">
  <img src="./images/Siamese Network.webp" width="400">
</p>

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



## Project Structure

siamese/
├── dataset.py          # Data loading and preprocessing pipeline
├── modules.py          # Model definitions (SiameseEncoder, BinaryClassifier)
├── train.py            # Training pipeline for Siamese and classifier networks
├── predict.py          # Evaluation and testing (confusion matrix, metrics)
├── utils.py            # Utility functions for plotting, saving samples, feature extraction, etc.
├── params.py           # Global configuration (hyperparameters, paths, augmentation, etc.)
└── models/             # Folder for saved models (.pth)
    ├── siamese.pth
    ├── classifier.pth
└── images/             # Folder for saved output figures
    ├── siamese_loss.png
    ├── classifier_loss.png
    ├── confusion_matrix.png
    └── input_sample.png
└── dataset/
    ├── train-image/
    ├── train-metadata.csv 



## File Explanations

- **params.py** – Stores all global variables and hyperparameters, including dataset paths, image preprocessing, model dimensions, and training settings.  
- **dataset.py** – Defines dataset classes, data augmentation, and loaders for both triplet and classification tasks.  
- **modules.py** – Contains the model definitions: the Siamese encoder (ResNet-50) and binary classifier (4-layer MLP).  
- **utils.py** – Includes helper functions for plotting, saving figures, feature extraction, and directory creation.  
- **train.py** – Main training script that trains the Siamese encoder, extracts embeddings, and trains the classifier.  
- **predict.py** – Evaluation script that loads trained models, computes predictions, and saves the confusion matrix.



## Data Preprocessing

- Input: **256×256 RGB** dermoscopic images (`train-image/`)  
- Metadata: `train-metadata.csv` (containing `isic_id`, `patient_id`, `target`)  
- Split: **70% train / 10% validation / 20% test**, grouped by patient ID to prevent data leakage.  
- Normalization: `mean = [0.5, 0.5, 0.5]`, `std = [0.5, 0.5, 0.5]`.  
- Augmentation: random rotations (±15°), color jitter, horizontal/vertical flips (p=0.5).  

All preprocessing configurations and split ratios are defined in `params.py` for reproducibility.



## Training and Testing

All experiments were conducted in **Google Colab**.  
Before running, ensure that the working directory is correctly set to the project folder.


### Train Both Networks
%cd /content/siamese
!python predict.py

#### This command will:
- Train the Siamese encoder using **Triplet Margin Loss**  
- Extract embeddings from the encoder  
- Train the binary classifier using **CrossEntropyLoss**  
- Save model weights and training plots under `./models/` and `./images/`


### Evaluate on Test Set
%cd /content/siamese
!python predict.py

#### This command loads the trained models and:
- Evaluates performance on the test dataset
- Computes accuracy, precision, recall, and F1-score
- Generates and saves the confusion matrix as `./images/confusion_matrix.png`



## Example Outputs

**1. Siamese Network Training Loss**  
<p align="center">
  <img src="./images/siamese_loss.png" width="450">
</p>
The triplet loss of the Siamese encoder steadily decreases during training, showing that the network effectively learns to minimize distances between similar image pairs while separating dissimilar ones.

---

**2. Binary Classifier Loss**  
<p align="center">
  <img src="./images/classifier_loss.png" width="450">
</p>
The CrossEntropy loss for both training and validation sets consistently declines, indicating stable convergence.  
Validation loss flattens near the end, suggesting moderate generalization with minimal overfitting.

---

**3. Confusion Matrix**  
<p align="center">
  <img src="./images/confusion_matrix.png" width="350">
</p>
The confusion matrix demonstrates that the classifier correctly identifies most benign and malignant lesions.  
Diagonal dominance confirms strong predictive performance and well-learned decision boundaries.

---

**Sample Input Example**  
<p align="center">
  <img src="./images/input_sample.png" width="220">
</p>
This sample dermoscopic image was randomly **rotated** and **color-adjusted** as part of data augmentation.  
Such transformations increase dataset diversity and improve model robustness to variations in image orientation and illumination.


