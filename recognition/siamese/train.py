from params import (MODELPATH, IMAGEPATH, EPOCHS_SIAMESE, EPOCHS_CLS,
                    TRIPLET_MARGIN, LR_SIAMESE, LR_CLS)
from dataset import get_loaders
from modules import SiameseEncoder, BinaryClassifier

import os
import torch
import torch.nn as nn

from utils import ensure_dir, plot_lines, save_sample_input, extract_features


#  Siamese training (Triplet loss) 
def train_siamese(encoder, train_loader, val_loader, device):
    encoder.train()
    opt = torch.optim.Adam(encoder.parameters(), lr=LR_SIAMESE, betas=(0.9, 0.999))
    criterion = nn.TripletMarginLoss(margin=TRIPLET_MARGIN, p=2)

    tr_hist, va_hist = [], []

    best_val = float('inf')
    patience, waited = 5, 0  # early stop after 5 epochs without improvement

    for epoch in range(EPOCHS_SIAMESE):
        # train 
        encoder.train()
        train_sum = 0.0
        for anc, pos, neg, _ in train_loader:
            anc, pos, neg = anc.to(device), pos.to(device), neg.to(device)
            za, zp, zn = encoder(anc), encoder(pos), encoder(neg)
            loss = criterion(za, zp, zn)
            opt.zero_grad(); loss.backward(); opt.step()
            train_sum += loss.item()
        avg_tr = train_sum / max(1, len(train_loader))

        # validation
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

        #  Early Stopping 
        if avg_va < best_val - 0.001:  # min_delta=0.001
            best_val = avg_va
            waited = 0
        else:
            waited += 1
            if waited >= patience:
                print(f"[Siamese] Early stopping at epoch {epoch+1}")
                break

    torch.save(encoder.state_dict(), os.path.join(MODELPATH, "siamese.pth"))
    print("[INFO] Saved final Siamese encoder (stopped model).")
    return tr_hist, va_hist


#  classifier training on embeddings 
def train_classifier(clf, train_data, val_data, device):
    opt = torch.optim.Adam(clf.parameters(), lr=LR_CLS, betas=(0.9, 0.999), weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode='min', factor=0.5, patience=3)
    criterion = nn.CrossEntropyLoss()

    Xtr, ytr = train_data
    Xva, yva = val_data

    tr_hist, va_hist, va_acc_hist = [], [], []

    best_val = float('inf')
    patience, waited = 5, 0  # early stop after 5 epochs without improvement

    for epoch in range(EPOCHS_CLS):
        #  train 
        clf.train()
        idx = torch.randperm(len(Xtr))
        Xb, yb = Xtr[idx].to(device), ytr[idx].to(device)
        logits = clf(Xb)
        loss = criterion(logits, yb)
        opt.zero_grad(); loss.backward(); opt.step()
        train_loss = loss.item()

        # validation
        clf.eval()
        with torch.no_grad():
            v_logits = clf(Xva.to(device))
            val_loss = criterion(v_logits, yva.to(device)).item()
            val_acc = (v_logits.argmax(1).cpu() == yva).float().mean().item()
        scheduler.step(val_loss)

        tr_hist.append(train_loss)
        va_hist.append(val_loss)
        va_acc_hist.append(val_acc)

        print(f"[CLS] Epoch {epoch+1}/{EPOCHS_CLS} "
              f"train_loss={train_loss:.4f} val_loss={val_loss:.4f} val_acc={val_acc*100:.2f}%")

        if val_loss < best_val - 0.001:  # min_delta=0.001
            best_val = val_loss
            waited = 0
        else:
            waited += 1
            if waited >= patience:
                print(f"[CLS] Early stopping at epoch {epoch+1}")
                break

    torch.save(clf.state_dict(), os.path.join(MODELPATH, "classifier.pth"))
    print("[INFO] Saved final classifier (stopped model).")
    return tr_hist, va_hist, va_acc_hist


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)

    loaders   = get_loaders()
    tri_train = loaders["triplet_train"]
    tri_val   = loaders["triplet_val"]
    cls_tr    = loaders["classif_train"]
    cls_va    = loaders["classif_val"]

    ensure_dir(MODELPATH)
    ensure_dir(IMAGEPATH)

    encoder = SiameseEncoder(out_dim=512).to(device)
    siam_tr_hist, siam_va_hist = train_siamese(encoder, tri_train, tri_val, device)
    xs = list(range(1, len(siam_tr_hist) + 1))
    plot_lines(xs, [siam_tr_hist, siam_va_hist], ["Training", "Validation"],
               title="Loss of the Siamese Network",
               xlabel="Epochs", ylabel="Triplet Loss",
               save_path=os.path.join(IMAGEPATH, "siamese_loss.png"))

    print("[INFO] Extracting embeddings...")
    encoder.eval()
    Xtr, ytr = extract_features(encoder, cls_tr, device)
    Xva, yva = extract_features(encoder, cls_va, device)

    clf = BinaryClassifier(in_dim=512).to(device)
    cls_tr_hist, cls_va_hist, _ = train_classifier(clf, (Xtr, ytr), (Xva, yva), device)
    xs = list(range(1, len(cls_tr_hist) + 1))
    plot_lines(xs, [cls_tr_hist, cls_va_hist], ["Training", "Validation"],
               title="Loss of the Binary Classifier",
               xlabel="Epochs", ylabel="CrossEntropy Loss",
               save_path=os.path.join(IMAGEPATH, "classifier_loss.png"))

    save_sample_input(loaders["classif_train"], IMAGEPATH)
    print(f"[INFO] Training finished. All results saved to {IMAGEPATH}")


if __name__ == "__main__":
    main()