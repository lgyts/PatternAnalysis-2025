from params import (MODELPATH, IMAGEPATH, EPOCHS_SIAMESE, EPOCHS_CLS,
                    TRIPLET_MARGIN, LR_SIAMESE, LR_CLS)
from dataset import get_loaders
from modules import SiameseEncoder, BinaryClassifier

from tqdm import tqdm
import torch
import torch.nn as nn
import os


# train siamese encoder with triplet loss
def train_siamese(encoder, loader, device):
    encoder.train()
    opt = torch.optim.Adam(encoder.parameters(), lr=LR_SIAMESE, betas=(0.9, 0.999))
    criterion = nn.TripletMarginLoss(margin=TRIPLET_MARGIN, p=2)

    for epoch in range(EPOCHS_SIAMESE):
        total_loss = 0.0
        total = len(loader)
        print(f"\n[Siamese] Epoch {epoch+1}/{EPOCHS_SIAMESE}")

        for i, (anc, pos, neg, _) in enumerate(loader):
            anc, pos, neg = anc.to(device), pos.to(device), neg.to(device)
            za, zp, zn = encoder(anc), encoder(pos), encoder(neg)
            loss = criterion(za, zp, zn)

            opt.zero_grad()
            loss.backward()
            opt.step()

            total_loss += loss.item()

            # progress display per 10 batches
            if (i + 1) % 10 == 0 or (i + 1) == total:
                pct = 100.0 * (i + 1) / total
                print(f"\rProgress: {pct:5.1f}% complete", end="")

        avg_loss = total_loss / max(1, total)
        print(f"\rProgress: 100.0% complete")
        print(f"[Siamese] epoch {epoch+1}: loss={avg_loss:.4f}")

# extract features using trained encoder
def extract_features(encoder, loader, device):
    encoder.eval()
    feats, labels = [], []
    total = len(loader)
    with torch.no_grad():
        for i, (xb, yb, _) in enumerate(loader):
            z = encoder(xb.to(device)).cpu()
            feats.append(z); labels.append(yb)
            if (i + 1) % 10 == 0 or (i + 1) == total:
                pct = 100.0 * (i + 1) / total
                print(f"\r[Extract] {pct:5.1f}% complete", end="")
    print()
    return torch.cat(feats), torch.cat(labels)

# train classifier on extracted features
def train_classifier(clf, train_data, val_data, device):
    opt = torch.optim.Adam(clf.parameters(), lr=LR_CLS, betas=(0.9, 0.999))
    criterion = nn.CrossEntropyLoss()
    Xtr, ytr = train_data
    Xva, yva = val_data

    for epoch in range(EPOCHS_CLS):
        clf.train()
        idx = torch.randperm(len(Xtr))
        Xb, yb = Xtr[idx].to(device), ytr[idx].to(device)
        logits = clf(Xb)
        loss = criterion(logits, yb)
        opt.zero_grad(); loss.backward(); opt.step()

        with torch.no_grad():
            clf.eval()
            val_logits = clf(Xva.to(device))
            val_loss = criterion(val_logits, yva.to(device))
            acc = (val_logits.argmax(1).cpu() == yva).float().mean().item()
        if (epoch+1) % 10 == 0 or epoch == 0:
            print(f"[CLS] epoch {epoch+1}: loss={loss.item():.4f} val_loss={val_loss.item():.4f} val_acc={acc*100:.2f}%")



def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)
    loaders = get_loaders()
    tri_loader = loaders["triplet_train"]
    cls_tr = loaders["classif_train"]
    cls_va = loaders["classif_val"]
    cls_te = loaders["classif_test"]

    os.makedirs(MODELPATH, exist_ok=True)  

    # load or train siamese encoder
    encoder = SiameseEncoder(out_dim=1000).to(device)
    enc_path = os.path.join(MODELPATH, "siamese.pth")
    if os.path.exists(enc_path):
        encoder.load_state_dict(torch.load(enc_path, map_location=device))
        print(f"[INFO] Loaded existing encoder: {enc_path}")
    else:
        train_siamese(encoder, tri_loader, device)
        torch.save(encoder.state_dict(), enc_path)
        print(f"[INFO] Saved encoder to {enc_path}")

    # extract features for classification
    Xtr, ytr = extract_features(encoder, cls_tr, device)
    Xva, yva = extract_features(encoder, cls_va, device)
    Xte, yte = extract_features(encoder, cls_te, device)

    # train classifier
    clf = BinaryClassifier(in_dim=1000).to(device)
    clf_path = os.path.join(MODELPATH, "classifier.pth")
    if os.path.exists(clf_path):
        clf.load_state_dict(torch.load(clf_path, map_location=device))
        print(f"[INFO] Loaded existing classifier: {clf_path}")
    else:
        train_classifier(clf, (Xtr, ytr), (Xva, yva), device)
        torch.save(clf.state_dict(), clf_path)
        print(f"[INFO] Saved classifier to {clf_path}")

    # evaluate on test set
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