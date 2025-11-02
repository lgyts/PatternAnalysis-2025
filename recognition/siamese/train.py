# train.py
# Training script for Siamese encoder and binary classifier on ISIC dataset.
# Uses triplet loss for the Siamese network.




from params import (MODELPATH, IMAGEPATH, EPOCHS_SIAMESE, EPOCHS_CLS,
                    TRIPLET_MARGIN, LR_SIAMESE, LR_CLS)
from dataset import get_loaders
from modules import SiameseEncoder, BinaryClassifier

from utils import (
    ensure_dirs, set_seed, get_device,
    save_loss_plot, save_triplet_examples, save_ckpt
)

from tqdm import tqdm
import torch
import torch.nn as nn
import os


# -------------------- Siamese training --------------------
def train_siamese(encoder: torch.nn.Module, loader, device: str):
    encoder.train()
    opt = torch.optim.Adam(encoder.parameters(), lr=LR_SIAMESE, betas=(0.9, 0.999))
    criterion = nn.TripletMarginLoss(margin=TRIPLET_MARGIN, p=2)

    epoch_losses = []
    visualized = False

    for epoch in range(EPOCHS_SIAMESE):
        total_loss = 0.0
        total = len(loader)
        print(f"\n[Siamese] Epoch {epoch+1}/{EPOCHS_SIAMESE}")

        for i, (anc, pos, neg, _) in enumerate(loader):
            anc, pos, neg = anc.to(device), pos.to(device), neg.to(device)

            # Save one visualization batch at the beginning
            if not visualized and epoch == 0 and i == 0:
                save_triplet_examples(anc, pos, neg, max_items=16,
                                      save_name="triplet_examples.png")
                visualized = True

            za, zp, zn = encoder(anc), encoder(pos), encoder(neg)
            loss = criterion(za, zp, zn)

            opt.zero_grad()
            loss.backward()
            opt.step()

            total_loss += loss.item()
            if (i + 1) % 10 == 0 or (i + 1) == total:
                pct = 100.0 * (i + 1) / total
                print(f"\rProgress: {pct:5.1f}% complete", end="")

        avg = total_loss / max(1, total)
        epoch_losses.append(avg)
        print(f"\rProgress: 100.0% complete")
        print(f"[Siamese] epoch {epoch+1}: loss={avg:.4f}")

    save_loss_plot(epoch_losses, None,
                   title="Siamese Triplet Loss",
                   save_name="siamese_loss_curve.png")


# -------------------- Feature extraction for classifier --------------------
@torch.no_grad()
def extract_features(encoder: torch.nn.Module, loader, device: str):
    encoder.eval()
    feats, labels = [], []
    for batch in loader:
        if len(batch) == 4:
            # If your classif loader returns (img, label, path, idx) etc.
            imgs, y = batch[0], batch[1]
        else:
            imgs, y = batch[0], batch[1]
        z = encoder(imgs.to(device))
        feats.append(z.cpu())
        labels.append(y.cpu())
    X = torch.cat(feats, dim=0)
    y = torch.cat(labels, dim=0).long()
    return X, y


# -------------------- Classifier training --------------------
def train_classifier(clf: torch.nn.Module, train_data, val_data, device: str):
    opt = torch.optim.Adam(clf.parameters(), lr=LR_CLS, betas=(0.9, 0.999))
    criterion = nn.CrossEntropyLoss()
    Xtr, ytr = train_data
    Xva, yva = val_data

    tr_losses, va_losses = [], []

    for epoch in range(EPOCHS_CLS):
        clf.train()
        # Simple full-batch optimization (as in your prior pipeline)
        idx = torch.randperm(len(Xtr))
        Xb, yb = Xtr[idx].to(device), ytr[idx].to(device)

        logits = clf(Xb)
        loss = criterion(logits, yb)

        opt.zero_grad()
        loss.backward()
        opt.step()

        # validation
        with torch.no_grad():
            clf.eval()
            v_logits = clf(Xva.to(device))
            v_loss = criterion(v_logits, yva.to(device))
            v_acc = (v_logits.argmax(1).cpu() == yva).float().mean().item()

        tr_losses.append(loss.item())
        va_losses.append(v_loss.item())

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"[CLS] epoch {epoch+1}: "
                  f"loss={loss.item():.4f}  "
                  f"val_loss={v_loss.item():.4f}  "
                  f"val_acc={v_acc*100:.2f}%")

    save_loss_plot(tr_losses, va_losses,
                   title="Classifier CE Loss (train/val)",
                   save_name="classifier_loss_curve.png")


# -------------------- Main --------------------
def main():
    set_seed(42)
    ensure_dirs()
    device = get_device()
    print("Device:", device)

    loaders = get_loaders()
    tri_loader = loaders["triplet_train"]
    cls_tr = loaders["classif_train"]
    cls_va = loaders["classif_val"]
    cls_te = loaders["classif_test"]

    os.makedirs(MODELPATH, exist_ok=True)

    # 1) Siamese encoder
    encoder = SiameseEncoder(out_dim=1000).to(device)
    enc_path = os.path.join(MODELPATH, "siamese.pth")
    if os.path.exists(enc_path):
        encoder.load_state_dict(torch.load(enc_path, map_location=device))
        print(f"[INFO] Loaded existing encoder: {enc_path}")
    else:
        train_siamese(encoder, tri_loader, device)
        save_ckpt(encoder, enc_path)

    # 2) Extract features
    Xtr, ytr = extract_features(encoder, cls_tr, device)
    Xva, yva = extract_features(encoder, cls_va, device)
    Xte, yte = extract_features(encoder, cls_te, device)

    # 3) Classifier
    clf = BinaryClassifier(in_dim=Xtr.shape[1]).to(device)
    clf_path = os.path.join(MODELPATH, "classifier.pth")
    if os.path.exists(clf_path):
        clf.load_state_dict(torch.load(clf_path, map_location=device))
        print(f"[INFO] Loaded existing classifier: {clf_path}")
    else:
        train_classifier(clf, (Xtr, ytr), (Xva, yva), device)
        save_ckpt(clf, clf_path)

    # 4) Test
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
