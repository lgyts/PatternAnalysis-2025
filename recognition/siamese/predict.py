# predct.py
# Evaluate trained Siamese encoder + classifier on test set.   
# Author: s4778251


import os
import torch
from sklearn.metrics import confusion_matrix, classification_report
from dataset import get_loaders
from modules import SiameseEncoder, BinaryClassifier
from utils import plot_confusion_matrix, extract_features
from params import MODELPATH, IMAGEPATH, OUT_DIM, CM_NAME


def main():
    """Evaluate trained Siamese encoder + classifier on the test set.

    This script:
        1. Loads trained model weights from disk.
        2. Extracts test embeddings using the Siamese encoder.
        3. Applies the trained binary classifier to predict classes.
        4. Computes accuracy, confusion matrix, and classification report.
        5. Saves the confusion matrix as an image.

    The results are printed to the console and saved in IMAGEPATH.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)

    # Load dataloaders for evaluation
    loaders = get_loaders()
    cls_te = loaders["classif_test"]

    # Initialize models
    encoder = SiameseEncoder(out_dim=OUT_DIM).to(device)
    clf = BinaryClassifier(in_dim=OUT_DIM).to(device)

    # Load pretrained weights
    encoder.load_state_dict(torch.load(os.path.join(MODELPATH, "siamese.pth"), map_location=device))
    clf.load_state_dict(torch.load(os.path.join(MODELPATH, "classifier.pth"), map_location=device))

    print("[INFO] Extracting test features...")

    # Extract embeddings and labels from test set
    Xte, yte = extract_features(encoder, cls_te, device)

    # Predict class logits using classifier
    clf.eval()
    with torch.no_grad():
        preds = clf(Xte.to(device)).argmax(1).cpu()

    # Compute evaluation metrics
    acc = (preds == yte).float().mean().item()
    cm = confusion_matrix(yte.numpy(), preds.numpy())

    print(f"[TEST] Accuracy: {acc*100:.2f}%")
    print("[TEST] Confusion Matrix:\n", cm)
    print("\n[TEST] Classification Report:\n",
          classification_report(yte.numpy(), preds.numpy(),
                                target_names=["benign(0)", "malignant(1)"]))

    # Save confusion matrix plot
    plot_confusion_matrix(cm, classes=["Benign", "Malignant"],
                          save_path=os.path.join(IMAGEPATH, CM_NAME))
    print(f"[INFO] Saved {CM_NAME} to: {IMAGEPATH}")


if __name__ == "__main__":
    main()
