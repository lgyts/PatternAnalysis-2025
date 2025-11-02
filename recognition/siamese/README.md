# Siamese Network for Skin Lesion Classification (ISIC 2020)

## 1. Description of the Algorithm and Problem Solved
This project implements a Siamese Neural Network for binary classification of dermoscopic skin lesion images from the ISIC 2020 dataset into benign and malignant (melanoma) classes.  
The model learns an embedding space in which visually similar lesions lie close together and dissimilar lesions lie far apart, enabling both metric learning and downstream classification.  
It corresponds to COMP3710 Pattern Analysis 2025 – Project 9 (Hard).

---

## 2. How It Works
1. **Siamese Encoder Training**  
   - The encoder is a ResNet-50 backbone with its final fully-connected layer replaced by a 1000-D projection head.  
   - It is trained using TripletMarginLoss (margin = 1.0) on batches of triplets (anchor, positive, negative).  
   - Training automatically saves:
     - `images/siamese_loss_curve.png`: Siamese loss curve
     - `images/triplet_examples.png`: Triplet visualisation (anchor, positive, negative)
   - Implemented in `train_siamese()` within `train.py`.

2. **Feature Extraction and Classifier Training**  
   - The trained encoder is frozen to extract embeddings for all images.  
   - A 4-layer fully connected MLP classifier (LeakyReLU activations, CrossEntropy loss) is trained on these embeddings.  
   - Automatically saves:
     - `images/classifier_loss_curve.png`: Train vs validation loss curves.  
   - Implemented in `train_classifier()` within `train.py`.

3. **Evaluation and Visualisation**  
   - `predict.py` loads checkpoints and evaluates on the test split.  
   - It prints accuracy, confusion matrix, and classification report.  
   - Automatically saves:
     - `images/test_examples_pred_vs_gt.png`: Input test images with true and predicted labels.

---

## 3. Repository Structure