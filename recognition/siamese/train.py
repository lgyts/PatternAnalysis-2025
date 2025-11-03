# train.py
# Train Siamese encoder + binary classifier on ISIC dataset.
# Author: s4778251


from params import (
    MODELPATH, IMAGEPATH, TRIPLET_MARGIN,
    LR_SIAMESE, LR_CLS, EPOCHS_SIAMESE, EPOCHS_CLS,
    PATIENCE, MIN_DELTA, SCHED_FACTOR, SCHED_PATIENCE,
    SIAMESE_LOSS_NAME, CLS_LOSS_NAME, SAVE_SAMPLE_NAME, OUT_DIM
)
from dataset import get_loaders
from modules import SiameseEncoder, BinaryClassifier
import os
import torch
import torch.nn as nn
from utils import ensure_dir, plot_lines, save_sample_input, extract_features


def train_siamese(encoder, train_loader, val_loader, device):
    """Train the Siamese encoder using triplet loss.

    Args:
        encoder (torch.nn.Module): Siamese feature encoder model.
        train_loader (DataLoader): Training dataloader providing triplets.
        val_loader (DataLoader): Validation dataloader providing triplets.
        device (str): Device to perform computation ("cuda" or "cpu").

    Returns:
        tuple[list[float], list[float]]:
            Two lists of per-epoch training and validation losses.
    """
    encoder.train()
    opt = torch.optim.Adam(encoder.parameters(), lr=LR_SIAMESE, betas=(0.9, 0.999))
    criterion = nn.TripletMarginLoss(margin=TRIPLET_MARGIN, p=2)

    tr_hist, va_hist = [], []
    best_val = float('inf')
    waited = 0  # patience counter for early stopping

    for epoch in range(EPOCHS_SIAMESE):
        encoder.train()
        train_sum = 0.0
        # Iterate through triplets (anchor, positive, negative, label)
        for anc, pos, neg, _ in train_loader:
            anc, pos, neg = anc.to(device), pos.to(device), neg.to(device)
            za, zp, zn = encoder(anc), encoder(pos), encoder(neg)
            loss = criterion(za, zp, zn)
            opt.zero_grad()
            loss.backward()
            opt.step()
            train_sum += loss.item()

        avg_tr = train_sum / max(1, len(train_loader))

        # Validation loop (no gradient computation)
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

        print(f"[Siamese] Epoch {epoch+1}/{EPOCHS_SIAMESE} train_loss={avg_tr:.4f} val_loss={avg_va:.4f}")

        # Early stopping logic
        if avg_va < best_val - MIN_DELTA:
            best_val = avg_va
            waited = 0
        else:
            waited += 1
            if waited >= PATIENCE:
                print(f"[Siamese] Early stopping at epoch {epoch+1}")
                break

    # Save trained encoder weights
    torch.save(encoder.state_dict(), os.path.join(MODELPATH, "siamese.pth"))
    print("[INFO] Saved final Siamese encoder (stopped model).")
    return tr_hist, va_hist


def train_classifier(clf, train_data, val_data, device):
    """Train the binary classifier on precomputed embeddings.

    Args:
        clf (torch.nn.Module): Binary classification MLP model.
        train_data (tuple[Tensor, Tensor]): Training embeddings and labels.
        val_data (tuple[Tensor, Tensor]): Validation embeddings and labels.
        device (str): Device ("cuda" or "cpu").

    Returns:
        tuple[list[float], list[float], list[float]]:
            Training losses, validation losses, and validation accuracies.
    """
    opt = torch.optim.Adam(clf.parameters(), lr=LR_CLS, betas=(0.9, 0.999), weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode='min', factor=SCHED_FACTOR, patience=SCHED_PATIENCE)
    criterion = nn.CrossEntropyLoss()

    Xtr, ytr = train_data
    Xva, yva = val_data

    tr_hist, va_hist, va_acc_hist = [], [], []
    best_val = float('inf')
    waited = 0

    for epoch in range(EPOCHS_CLS):
        clf.train()
        # Shuffle embeddings before each epoch
        idx = torch.randperm(len(Xtr))
        Xb, yb = Xtr[idx].to(device), ytr[idx].to(device)
        logits = clf(Xb)
        loss = criterion(logits, yb)
        opt.zero_grad()
        loss.backward()
        opt.step()
        train_loss = loss.item()

        # Validation phase
        clf.eval()
        with torch.no_grad():
            v_logits = clf(Xva.to(device))
            val_loss = criterion(v_logits, yva.to(device)).item()
            val_acc = (v_logits.argmax(1).cpu() == yva).float().mean().item()

        scheduler.step(val_loss)
        tr_hist.append(train_loss)
        va_hist.append(val_loss)
        va_acc_hist.append(val_acc)

        print(f"[CLS] Epoch {epoch+1}/{EPOCHS_CLS} train_loss={train_loss:.4f} val_loss={val_loss:.4f} val_acc={val_acc*100:.2f}%")

        # Early stopping
        if val_loss < best_val - MIN_DELTA:
            best_val = val_loss
            waited = 0
        else:
            waited += 1
            if waited >= PATIENCE:
                print(f"[CLS] Early stopping at epoch {epoch+1}")
                break

    # Save classifier weights
    torch.save(clf.state_dict(), os.path.join(MODELPATH, "classifier.pth"))
    print("[INFO] Saved final classifier (stopped model).")
    return tr_hist, va_hist, va_acc_hist


def main():
    """Main training routine for Siamese + classification stages.

    This pipeline:
        1. Loads dataset loaders for triplet and classification tasks.
        2. Trains the Siamese encoder using triplet loss.
        3. Extracts embeddings from the encoder for classification training.
        4. Trains the binary classifier using cross-entropy loss.
        5. Plots and saves both loss curves and one sample input image.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)

    # Prepare data loaders
    loaders = get_loaders()
    tri_train = loaders["triplet_train"]
    tri_val = loaders["triplet_val"]
    cls_tr = loaders["classif_train"]
    cls_va = loaders["classif_val"]

    # Ensure output directories exist
    ensure_dir(MODELPATH)
    ensure_dir(IMAGEPATH)

    # Train Siamese Encoder 
    encoder = SiameseEncoder(out_dim=OUT_DIM).to(device)
    siam_tr_hist, siam_va_hist = train_siamese(encoder, tri_train, tri_val, device)

    # Plot Siamese loss curve
    xs = list(range(1, len(siam_tr_hist) + 1))
    plot_lines(xs, [siam_tr_hist, siam_va_hist], ["Training", "Validation"],
               title="Loss of the Siamese Network",
               xlabel="Epochs", ylabel="Triplet Loss",
               save_path=os.path.join(IMAGEPATH, SIAMESE_LOSS_NAME))

    # Extract Embeddings
    print("[INFO] Extracting embeddings...")
    encoder.eval()
    Xtr, ytr = extract_features(encoder, cls_tr, device)
    Xva, yva = extract_features(encoder, cls_va, device)

    # Train Classifier 
    clf = BinaryClassifier(in_dim=OUT_DIM).to(device)
    cls_tr_hist, cls_va_hist, _ = train_classifier(clf, (Xtr, ytr), (Xva, yva), device)

    # Plot classifier loss curve
    xs = list(range(1, len(cls_tr_hist) + 1))
    plot_lines(xs, [cls_tr_hist, cls_va_hist], ["Training", "Validation"],
               title="Loss of the Binary Classifier",
               xlabel="Epochs", ylabel="CrossEntropy Loss",
               save_path=os.path.join(IMAGEPATH, CLS_LOSS_NAME))

    # Save a sample input image for reference
    save_sample_input(loaders["classif_train"], IMAGEPATH, filename=SAVE_SAMPLE_NAME)
    print(f"[INFO] Training finished. All results saved to {IMAGEPATH}")


if __name__ == "__main__":
    main()
