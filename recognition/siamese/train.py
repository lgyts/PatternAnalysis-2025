from params import (MODELPATH, IMAGEPATH, EPOCHS_SIAMESE, EPOCHS_CLS,
                    TRIPLET_MARGIN, LR_SIAMESE, LR_CLS)
from dataset import get_loaders
from modules import SiameseEncoder, BinaryClassifier

import os
import torch
import torch.nn as nn
from sklearn.metrics import confusion_matrix

from utils import ensure_dir, plot_lines, plot_confusion_matrix


#  Siamese training (Triplet loss) 
def train_siamese(encoder, train_loader, val_loader, device):
    encoder.train()
    opt = torch.optim.Adam(encoder.parameters(), lr=LR_SIAMESE, betas=(0.9, 0.999))
    criterion = nn.TripletMarginLoss(margin=TRIPLET_MARGIN, p=2)

    tr_hist, va_hist = [], []

    for epoch in range(EPOCHS_SIAMESE):
        #  train 
        encoder.train()
        train_sum = 0.0
        for anc, pos, neg, _ in train_loader:
            anc, pos, neg = anc.to(device), pos.to(device), neg.to(device)
            za, zp, zn = encoder(anc), encoder(pos), encoder(neg)
            loss = criterion(za, zp, zn)
            opt.zero_grad(); loss.backward(); opt.step()
            train_sum += loss.item()
        avg_tr = train_sum / max(1, len(train_loader))

        # val 
        encoder.eval()
        val_sum = 0.0
        with torch.no_grad():
            for anc, pos, neg, _ in val_loader:
                anc, pos, neg = anc.to(device), pos.to(device), neg.to(device)
                za, zp, zn = encoder(anc), encoder(pos), encoder(neg)
                vloss = criterion(za, zp, zn)
                val_sum += vloss.item()
        avg_va = val_sum / max(1, len(val_loader))

        tr_hist.append(avg_tr)
        va_hist.append(avg_va)
        print(f"[Siamese] Epoch {epoch+1}/{EPOCHS_SIAMESE} "
              f"train_loss={avg_tr:.4f} val_loss={avg_va:.4f}")

    return tr_hist, va_hist


#  feature extraction 
@torch.no_grad()
def extract_features(encoder, loader, device):
    encoder.eval()
    xs, ys = [], []
    total = len(loader)
    for i, (xb, yb, _) in enumerate(loader):
        feats = encoder(xb.to(device)).cpu()
        xs.append(feats); ys.append(yb)
        if (i + 1) % 10 == 0 or (i + 1) == total:
            pct = 100.0 * (i + 1) / total
            print(f"\r[Extract] {pct:5.1f}% complete", end="")
    print()
    return torch.cat(xs), torch.cat(ys)


#  classifier training on embeddings 
def train_classifier(clf, train_data, val_data, device):
    opt = torch.optim.Adam(
      clf.parameters(),
      lr=LR_CLS,
      betas=(0.9, 0.999),
      weight_decay=5e-4   # add L2
    )
    criterion = nn.CrossEntropyLoss()
    Xtr, ytr = train_data
    Xva, yva = val_data

    tr_hist, va_hist, va_acc_hist = [], [], []

    for epoch in range(EPOCHS_CLS):
        # train
        clf.train()
        idx = torch.randperm(len(Xtr))
        Xb, yb = Xtr[idx].to(device), ytr[idx].to(device)
        logits = clf(Xb)
        loss = criterion(logits, yb)
        opt.zero_grad(); loss.backward(); opt.step()
        train_loss = loss.item()

        # val
        clf.eval()
        with torch.no_grad():
            v_logits = clf(Xva.to(device))
            val_loss = criterion(v_logits, yva.to(device)).item()
            val_acc = (v_logits.argmax(1).cpu() == yva).float().mean().item()

        tr_hist.append(train_loss)
        va_hist.append(val_loss)
        va_acc_hist.append(val_acc)

        print(f"[CLS] Epoch {epoch+1}/{EPOCHS_CLS} "
              f"train_loss={train_loss:.4f} "
              f"val_loss={val_loss:.4f} val_acc={val_acc*100:.2f}%")

    return tr_hist, va_hist, va_acc_hist


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)

    loaders   = get_loaders()  
    tri_train = loaders["triplet_train"]
    tri_val   = loaders["triplet_val"]
    cls_tr    = loaders["classif_train"]
    cls_va    = loaders["classif_val"]
    cls_te    = loaders["classif_test"]

    ensure_dir(MODELPATH)
    ensure_dir(IMAGEPATH)

    # 1) Siamese encoder
    encoder = SiameseEncoder(out_dim=1000).to(device)
    enc_path = os.path.join(MODELPATH, "siamese.pth")
    if os.path.exists(enc_path):
        encoder.load_state_dict(torch.load(enc_path, map_location=device))
        print(f"[INFO] Loaded existing encoder: {enc_path}")
        siam_tr_hist, siam_va_hist = [], []  
    else:
        siam_tr_hist, siam_va_hist = train_siamese(encoder, tri_train, tri_val, device)
        torch.save(encoder.state_dict(), enc_path)
        print(f"[INFO] Saved encoder to {enc_path}")

        # Siamese loss (train/val)
        xs = list(range(1, len(siam_tr_hist) + 1))
        plot_lines(xs, [siam_tr_hist, siam_va_hist], ["Training", "Validation"],
                   title="Loss of the Siamese Network",
                   xlabel="Epochs", ylabel="Loss",
                   save_path=os.path.join(IMAGEPATH, "siamese_loss.png"))

    # Extract embeddings for classifier
    Xtr, ytr = extract_features(encoder, cls_tr, device)
    Xva, yva = extract_features(encoder, cls_va, device)
    Xte, yte = extract_features(encoder, cls_te, device)

    # Classifier
    clf = BinaryClassifier(in_dim=1000).to(device)
    clf_path = os.path.join(MODELPATH, "classifier.pth")
    if os.path.exists(clf_path):
        clf.load_state_dict(torch.load(clf_path, map_location=device))
        print(f"[INFO] Loaded existing classifier: {clf_path}")
        cls_tr_hist, cls_va_hist = [], []
    else:
        cls_tr_hist, cls_va_hist, _ = train_classifier(clf, (Xtr, ytr), (Xva, yva), device)
        torch.save(clf.state_dict(), clf_path)
        print(f"[INFO] Saved classifier to {clf_path}")

        # Classifier loss (train/val)
        xs = list(range(1, len(cls_tr_hist) + 1))
        plot_lines(xs, [cls_tr_hist, cls_va_hist], ["Training", "Validation"],
                   title="Loss of the Binary Classifier",
                   xlabel="Epochs", ylabel="Loss",
                   save_path=os.path.join(IMAGEPATH, "classifier_loss.png"))

    # 4) Test once + Confusion Matrix
    clf.eval()
    with torch.no_grad():
        pred = clf(Xte.to(device)).argmax(1).cpu()
    test_acc = (pred == yte).float().mean().item()
    cm = confusion_matrix(yte.numpy(), pred.numpy())
    print(f"[TEST] Accuracy: {test_acc*100:.2f}%")
    print("[TEST] Confusion Matrix:\n", cm)

    plot_confusion_matrix(cm, classes=["Benign", "Malignant"],
                          save_path=os.path.join(IMAGEPATH, "confusion_matrix.png"))

    print("[INFO] Saved plots to:", IMAGEPATH)


if __name__ == "__main__":
    main()
