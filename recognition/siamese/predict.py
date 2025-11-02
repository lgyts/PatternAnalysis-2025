# predict.py
# Evaluation script for Siamese encoder and binary classifier on ISIC dataset.
# Computes accuracy, confusion matrix, and classification report on test set.



import os
import argparse
import torch
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report

from params import MODELPATH
from dataset import get_loaders
from modules import SiameseEncoder, BinaryClassifier
from utils import (
    ensure_dirs, get_device, save_classif_examples
)


@torch.no_grad()
def extract_features(encoder: torch.nn.Module, loader, device: str):
    encoder.eval()
    feats, labels = [], []
    for batch in loader:
        if len(batch) == 4:
            imgs, y = batch[0], batch[1]
        else:
            imgs, y = batch[0], batch[1]
        z = encoder(imgs.to(device))
        feats.append(z.cpu())
        labels.append(y.cpu())
    X = torch.cat(feats, dim=0)
    y = torch.cat(labels, dim=0).long()
    return X, y


def main():
    parser = argparse.ArgumentParser(description="Evaluate Siamese + Classifier on ISIC test set")
    parser.add_argument("--no-cuda", action="store_true", default=False)
    args = parser.parse_args()

    ensure_dirs()
    device = get_device(no_cuda=args.no_cuda)
    print("Device:", device)

    loaders = get_loaders()
    te_loader = loaders["classif_test"]

    # load models
    encoder = SiameseEncoder(out_dim=1000).to(device)
    clf = BinaryClassifier(in_dim=1000).to(device)

    enc_path = os.path.join(MODELPATH, "siamese.pth")
    clf_path = os.path.join(MODELPATH, "classifier.pth")
    if not os.path.exists(enc_path) or not os.path.exists(clf_path):
        raise FileNotFoundError("Missing checkpoints. Train first to create 'siamese.pth' and 'classifier.pth'.")

    encoder.load_state_dict(torch.load(enc_path, map_location=device))
    clf.load_state_dict(torch.load(clf_path, map_location=device))
    encoder.eval(); clf.eval()

    # ---- Visualization on a single test batch ----
    imgs, labels, *_ = next(iter(te_loader))
    with torch.no_grad():
        feats = encoder(imgs.to(device))
        logits = clf(feats)
        preds = torch.softmax(logits, dim=1).argmax(1).cpu()
    save_classif_examples(
        imgs, labels, preds,
        class_names=("benign(0)", "malignant(1)"),
        max_items=16,
        save_name="test_examples_pred_vs_gt.png"
    )

    # ---- Full test metrics ----
    Xte, yte = extract_features(encoder, te_loader, device)
    with torch.no_grad():
        logits = clf(Xte.to(device))
        probs = torch.softmax(logits, dim=1).cpu().numpy()
        preds = probs.argmax(1)

    acc = (preds == yte.numpy()).mean()
    cm = confusion_matrix(yte.numpy(), preds)
    print(f"[TEST] Accuracy: {acc*100:.2f}%")
    print("[TEST] Confusion Matrix:\n", cm)
    print("\n[TEST] Classification Report:")
    print(classification_report(yte.numpy(), preds, target_names=["benign(0)", "malignant(1)"]))


if __name__ == "__main__":
    main()
