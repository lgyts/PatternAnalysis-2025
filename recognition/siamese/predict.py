import os
import torch
from sklearn.metrics import confusion_matrix, classification_report
from dataset import get_loaders
from modules import SiameseEncoder, BinaryClassifier
from utils import plot_confusion_matrix, extract_features
from params import MODELPATH, IMAGEPATH

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)

    loaders = get_loaders()
    cls_te  = loaders["classif_test"]

    encoder = SiameseEncoder(out_dim=512).to(device)
    clf     = BinaryClassifier(in_dim=512).to(device)

    encoder.load_state_dict(torch.load(os.path.join(MODELPATH, "siamese.pth"), map_location=device))
    clf.load_state_dict(torch.load(os.path.join(MODELPATH, "classifier.pth"), map_location=device))

    print("[INFO] Extracting test features...")
    Xte, yte = extract_features(encoder, cls_te, device)

    clf.eval()
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
