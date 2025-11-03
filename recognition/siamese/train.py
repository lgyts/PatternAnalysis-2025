from params import (MODELPATH, IMAGEPATH, EPOCHS_SIAMESE, EPOCHS_CLS,
                    TRIPLET_MARGIN, LR_SIAMESE, LR_CLS)
from dataset import get_loaders
from modules import SiameseEncoder, BinaryClassifier

from tqdm import tqdm
import torch
import torch.nn as nn
import os


# ---------- train siamese encoder with triplet loss ----------
def train_siamese(encoder, train_loader, val_loader, device):
    encoder.train()
    opt = torch.optim.Adam(encoder.parameters(), lr=LR_SIAMESE, betas=(0.9, 0.999))
    criterion = nn.TripletMarginLoss(margin=TRIPLET_MARGIN, p=2)

    for epoch in range(EPOCHS_SIAMESE):
        # ---- train ----
        encoder.train()
        total_train_loss = 0.0
        for anc, pos, neg, _ in train_loader:
            anc, pos, neg = anc.to(device), pos.to(device), neg.to(device)
            za, zp, zn = encoder(anc), encoder(pos), encoder(neg)
            loss = criterion(za, zp, zn)
            opt.zero_grad(); loss.backward(); opt.step()
            total_train_loss += loss.item()
        avg_train = total_train_loss / max(1, len(train_loader))

        # ---- val ----
        encoder.eval()
        total_val_loss = 0.0
        with torch.no_grad():
            for anc, pos, neg, _ in val_loader:
                anc, pos, neg = anc.to(device), pos.to(device), neg.to(device)
                za, zp, zn = encoder(anc), encoder(pos), encoder(neg)
                vloss = criterion(za, zp, zn)
                total_val_loss += vloss.item()
        avg_val = total_val_loss / max(1, len(val_loader))

        print(f"[Siamese] Epoch {epoch+1}/{EPOCHS_SIAMESE} "
              f"train_loss={avg_train:.4f} val_loss={avg_val:.4f}")


# ---------- extract features using trained encoder ----------
@torch.no_grad()
def extract_features(encoder, loader, device):
    encoder.eval()
    feats, labels = [], []
    total = len(loader)
    for i, (xb, yb, _) in enumerate(loader):
        z = encoder(xb.to(device)).cpu()
        feats.append(z)
        labels.append(yb)
        if (i + 1) % 10 == 0 or (i + 1) == total:
            pct = 100.0 * (i + 1) / total
            print(f"\r[Extract] {pct:5.1f}% complete", end="")
    print()
    return torch.cat(feats), torch.cat(labels)


# ---------- train classifier on extracted features ----------
def train_classifier(clf, train_data, val_data, device):
    opt = torch.optim.Adam(clf.parameters(), lr=LR_CLS, betas=(0.9, 0.999))
    criterion = nn.CrossEntropyLoss()
    Xtr, ytr = train_data
    Xva, yva = val_data

    for epoch in range(EPOCHS_CLS):
        clf.train()
        # === Train ===
        idx = torch.randperm(len(Xtr))
        Xb, yb = Xtr[idx].to(device), ytr[idx].to(device)
        logits = clf(Xb)
        loss = criterion(logits, yb)
        opt.zero_grad(); loss.backward(); opt.step()
        train_loss = loss.item()

        # === Val ===
        clf.eval()
        with torch.no_grad():
            val_logits = clf(Xva.to(device))
            val_loss = criterion(val_logits, yva.to(device)).item()
            val_acc  = (val_logits.argmax(1).cpu() == yva).float().mean().item()

        print(f"[CLS] Epoch {epoch+1}/{EPOCHS_CLS} "
              f"train_loss={train_loss:.4f} "
              f"val_loss={val_loss:.4f} val_acc={val_acc*100:.2f}%")


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)

    loaders   = get_loaders()
    tri_train = loaders["triplet_train"]
    tri_val   = loaders["triplet_val"]    # <<-- 新增：siamese 验证集
    cls_tr    = loaders["classif_train"]
    cls_va    = loaders["classif_val"]
    cls_te    = loaders["classif_test"]

    os.makedirs(MODELPATH, exist_ok=True)

    # ----- train or load siamese encoder -----
    encoder = SiameseEncoder(out_dim=1000).to(device)
    enc_path = os.path.join(MODELPATH, "siamese.pth")
    if os.path.exists(enc_path):
        encoder.load_state_dict(torch.load(enc_path, map_location=device))
        print(f"[INFO] Loaded existing encoder: {enc_path}")
    else:
        train_siamese(encoder, tri_train, tri_val, device)
        torch.save(encoder.state_dict(), enc_path)
        print(f"[INFO] Saved encoder to {enc_path}")

    # ----- extract features -----
    Xtr, ytr = extract_features(encoder, cls_tr, device)
    Xva, yva = extract_features(encoder, cls_va, device)
    Xte, yte = extract_features(encoder, cls_te, device)

    # ----- classifier -----
    clf = BinaryClassifier(in_dim=1000).to(device)
    clf_path = os.path.join(MODELPATH, "classifier.pth")
    if os.path.exists(clf_path):
        clf.load_state_dict(torch.load(clf_path, map_location=device))
        print(f"[INFO] Loaded existing classifier: {clf_path}")
    else:
        train_classifier(clf, (Xtr, ytr), (Xva, yva), device)
        torch.save(clf.state_dict(), clf_path)
        print(f"[INFO] Saved classifier to {clf_path}")

    # ----- test once (after training) -----
    clf.eval()
    with torch.no_grad():
        preds = clf(Xte.to(device)).argmax(1).cpu()
    acc = (preds == yte).float().mean().item()
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(yte, preds)
    print(f"[TEST] Accuracy: {acc*100:.2f}%")
    print("[TEST] Confusion Matrix:\n", cm)


if __name__ == "__main__":
    main()
