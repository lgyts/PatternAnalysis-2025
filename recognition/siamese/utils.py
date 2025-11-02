# utils.py
import os
import random
import math
import numpy as np
import torch
import matplotlib.pyplot as plt
from torchvision.utils import make_grid, save_image

# Optional import: fall back to defaults if not found
try:
    from params import IMAGEPATH, MODELPATH, MEAN, STD
except Exception:
    IMAGEPATH, MODELPATH = "./images", "./checkpoints"
    # Default imagenet-like mean/std; adjust if your params.py defines them differently
    MEAN, STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)


#  FS / seed / device 
def ensure_dirs() -> None:
    os.makedirs(IMAGEPATH, exist_ok=True)
    os.makedirs(MODELPATH, exist_ok=True)

def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_device(no_cuda: bool = False) -> str:
    return "cuda" if (torch.cuda.is_available() and not no_cuda) else "cpu"


#  Denormalization
def _to_tensor_on_device(vals, device, fourd: bool):
    t = torch.tensor(vals, dtype=torch.float32, device=device)
    return t.view(1, -1, 1, 1) if fourd else t.view(-1, 1, 1)

def denorm(img_tensor: torch.Tensor, mean=MEAN, std=STD) -> torch.Tensor:
    """
    Undo normalization back to [0,1] range for visualization.
    Works for (B,C,H,W) or (C,H,W).
    """
    fourd = (img_tensor.ndim == 4)
    device = img_tensor.device
    mean_t = _to_tensor_on_device(mean, device, fourd)
    std_t  = _to_tensor_on_device(std, device, fourd)
    out = img_tensor * std_t + mean_t
    return torch.clamp(out, 0.0, 1.0)


#  Visualization / logging 
def save_loss_plot(train_losses, val_losses=None,
                   title="Loss Curve", save_name="loss_curve.png") -> str:
    ensure_dirs()
    plt.figure()
    xs = range(1, len(train_losses) + 1)
    plt.plot(xs, train_losses, label="train")
    if val_losses is not None and len(val_losses) == len(train_losses):
        plt.plot(xs, val_losses, label="val")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(title)
    plt.legend()
    out_path = os.path.join(IMAGEPATH, save_name)
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"[IMG] Saved: {out_path}")
    return out_path

def save_triplet_examples(anchor, positive, negative,
                          max_items=16, save_name="triplet_examples.png") -> str:
    """
    Save a 3-row grid: anchor (row 1), positive (row 2), negative (row 3).
    """
    ensure_dirs()
    k = min(max_items, anchor.size(0))
    a = denorm(anchor[:k]).cpu()
    p = denorm(positive[:k]).cpu()
    n = denorm(negative[:k]).cpu()

    nrow = max(1, int(math.sqrt(k)))
    grid_a = make_grid(a, nrow=nrow, padding=2)
    grid_p = make_grid(p, nrow=nrow, padding=2)
    grid_n = make_grid(n, nrow=nrow, padding=2)
    stacked = torch.cat([grid_a, grid_p, grid_n], dim=1)  # vertical concat
    out_path = os.path.join(IMAGEPATH, save_name)
    save_image(stacked, out_path)
    print(f"[IMG] Saved: {out_path}")
    return out_path

def save_classif_examples(images, labels, preds=None,
                          class_names=("benign(0)", "malignant(1)"),
                          max_items=16, save_name="classif_examples.png") -> str:
    """
    Save a grid of input images (denormalized). Prints a small preview of T/P.
    """
    ensure_dirs()
    k = min(max_items, images.size(0))
    imgs = denorm(images[:k]).cpu()
    if preds is not None:
        preview = [f"T:{class_names[int(t)]}  P:{class_names[int(p)]}"
                   for t, p in zip(labels[:k], preds[:k])]
    else:
        preview = [f"T:{class_names[int(t)]}" for t in labels[:k]]

    print("[Preview] First samples:")
    for line in preview[:min(8, len(preview))]:
        print("  ", line)

    nrow = max(1, int(math.sqrt(k)))
    grid = make_grid(imgs, nrow=nrow, padding=2)
    out_path = os.path.join(IMAGEPATH, save_name)
    save_image(grid, out_path)
    print(f"[IMG] Saved: {out_path}")
    return out_path


#  Checkpoint 
def save_ckpt(model: torch.nn.Module, path: str) -> None:
    ensure_dirs()
    torch.save(model.state_dict(), path)
    print(f"[CKPT] Saved: {path}")

def load_ckpt(model: torch.nn.Module, path: str, map_location=None) -> bool:
    if not os.path.exists(path):
        return False
    state = torch.load(path, map_location=map_location)
    model.load_state_dict(state)
    print(f"[CKPT] Loaded: {path}")
    return True
