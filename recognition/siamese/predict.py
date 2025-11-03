# predict.py
# Evaluate trained Siamese encoder + binary classifier on the test split.

import os
import argparse
import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report

from params import MODELPATH
from dataset import get_loaders
from modules import SiameseEncoder, BinaryClassifier


def extract_features(encoder, loader, device):
    """Encode images to embeddings using the (frozen) encoder."""
    encoder.eval()
    feats, labels = [], []
    total = len(loader)
    with torch.no_grad():
        for i, (xb, yb, _) in enumerate(loader):
            z = encoder(xb.to(device)).cpu()
            feats.append(z)
            labels.append(yb)
            # progress display
            if (i + 1) % 10 == 0 or (i + 1) == total:
                pct = 100.0 * (i + 1) / total
                print(f"\r[Extract] {pct:5.1f}% complete", end="")
    print()
    return torch.cat(feats), torch.cat(labels)


def main():
    parser = argparse.ArgumentParser(description="Evaluate Siamese + Classifier on ISIC test set")
    parser.add_argument("--siam", type=str, default=os.path.join(MODELPATH, "siamese.pth"),
                        help="path to siamese encoder weights (.pth)")
    parser.add_argument("--clf", type=str, default=os.path.join(MODELPATH, "classifier.pth"),
                        help="path to classifier weights (.pth)")
    parser.add_argument("--sample", type=int, default=0,
                        help="evaluate on a random subset size (0 = full test set)")
    parser.add_argument("--no_cuda", action="store_true", help="force CPU")
    args = parser.parse_args()

    device = "cuda" if (torch.cuda.is_available() and not args.no_cuda) else "cpu"
    print("Device:", device)

    # data loaders
    loaders = get_loaders()
    te_loader = loaders["classif_test"]

    # optional subsampling
    if args.sample and args.sample > 0:
        # only evaluate on a random subset
        base_ds = te_loader.dataset
        n = min(args.sample, len(base_ds))
        idx = torch.randperm(len(base_ds))[:n].tolist()
        from torch.utils.data import Subset, DataLoader
        te_loader = DataLoader(
            Subset(base_ds, idx),
            batch_size=te_loader.batch_size,
            shuffle=False,
            num_workers=te_loader.num_workers,
            pin_memory=True,
            drop_last=False
        )
        print(f"[INFO] Evaluate on a random subset of {n} samples")

    # build and load models
    encoder = SiameseEncoder(out_dim=1000).to(device)
    clf = BinaryClassifier(in_dim=1000).to(device)

    if not os.path.exists(args.siam):
        raise FileNotFoundError(f"Encoder weights not found: {args.siam}")
    if not os.path.exists(args.clf):
        raise FileNotFoundError(f"Classifier weights not found: {args.clf}")

    encoder.load_state_dict(torch.load(args.siam, map_location=device))
    clf.load_state_dict(torch.load(args.clf, map_location=device))
    encoder.eval(); clf.eval()
    print(f"[INFO] Loaded encoder:   {args.siam}")
    print(f"[INFO] Loaded classifier:{args.clf}")

    # extract test features and evaluate
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