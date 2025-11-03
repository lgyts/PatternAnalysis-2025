# predict.py
# Evaluate trained Siamese encoder + binary classifier on the test split.

from params import (MODELPATH, IMAGEPATH)
from dataset import get_loaders
from modules import SiameseEncoder, BinaryClassifier

import os
import torch
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import numpy as np

from utils import ensure_dir, plot_confusion_matrix


@torch.no_grad()
def extract_embeddings(encoder, loader, device):
    encoder.eval()
    xs, ys = [], []
    for imgs, labels, _ in loader:
        feats = encoder(imgs.to(device)).cpu()
        xs.append(feats); ys.append(labels)
    return torch.cat(xs), torch.cat(ys)


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)

    loaders = get_loaders()
    cls_te  = loaders["classif_test"]

    # Load models
    encoder = SiameseEncoder(out_dim=512).to(device)
    clf = BinaryClassifier(in_dim=512).to(device)

    encoder.load_state_dict(torch.load(os.path.join(MODELPATH, "siamese.pth"), map_location=device))
    clf.load_state_dict(torch.load(os.path.join(MODELPATH, "classifier.pth"), map_location=device))

    encoder.eval()
    clf.eval()

    print("[INFO] Extracting test features...")
    Xte, yte = [], []
    with torch.no_grad():
        for xb, yb, _ in cls_te:
            feats = encoder(xb.to(device)).cpu()
            Xte.append(feats); yte.append(yb)
    Xte = torch.cat(Xte)
    yte = torch.cat(yte)

    with torch.no_grad():
        preds = clf(Xte.to(device)).argmax(1).cpu()

    acc = (preds == yte).float().mean().item()
    cm = confusion_matrix(yte.numpy(), preds.numpy())
    print(f"[TEST] Accuracy: {acc*100:.2f}%")
    print("[TEST] Confusion Matrix:\n", cm)
    print("\n[TEST] Classification Report:\n",
          classification_report(yte.numpy(), preds.numpy(),
                                target_names=["benign(0)", "malignant(1)"]))

    plot_confusion_matrix(cm, classes=["Benign", "Malignant"],
                          save_path=os.path.join(IMAGEPATH, "confusion_matrix.png"))
    print(f"[INFO] Saved confusion_matrix.png to: {IMAGEPATH}")


if __name__ == "__main__":
    main()
