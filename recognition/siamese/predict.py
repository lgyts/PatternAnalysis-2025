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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ensure_dir(IMAGEPATH)

    # data loaders
    loaders = get_loaders()
    cls_te  = loaders["classif_test"]

    # load models
    encoder = SiameseEncoder(out_dim=1000).to(device)
    classifier = BinaryClassifier(in_dim=1000).to(device)

    encoder.load_state_dict(torch.load(os.path.join(MODELPATH, "siamese.pth"), map_location=device))
    classifier.load_state_dict(torch.load(os.path.join(MODELPATH, "classifier.pth"), map_location=device))

    # embeddings of test set
    Xte, yte = extract_embeddings(encoder, cls_te, device)

    # predict
    classifier.eval()
    with torch.no_grad():
        logits = classifier(Xte.to(device))
        prob = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
        pred = np.argmax(logits.cpu().numpy(), axis=1)

    acc = accuracy_score(yte.numpy(), pred)
    cm = confusion_matrix(yte.numpy(), pred)
    print(f"[TEST] Accuracy: {acc*100:.2f}%")
    print("[TEST] Confusion Matrix:\n", cm)
    print("\n[TEST] Classification Report:\n",
          classification_report(yte.numpy(), pred, target_names=['benign(0)','malignant(1)']))

    # Plot confusion matrix
    plot_confusion_matrix(cm, classes=["Benign", "Malignant"],
                          save_path=os.path.join(IMAGEPATH, "confusion_matrix.png"))
    print("[INFO] Saved confusion_matrix.png to:", IMAGEPATH)


if __name__ == "__main__":
    main()

if __name__ == "__main__":
    main()