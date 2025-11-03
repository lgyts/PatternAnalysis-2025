import os
import torch
import matplotlib.pyplot as plt
import numpy as np

def ensure_dir(path):
    if path and not os.path.exists(path):
        os.makedirs(path, exist_ok=True)

def plot_lines(xs, ys_list, labels, title, xlabel, ylabel, save_path):
    plt.figure()
    for ys, lb in zip(ys_list, labels):
        plt.plot(xs, ys, label=lb)
    if labels:
        plt.legend()
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.tight_layout()
    ensure_dir(os.path.dirname(save_path))
    plt.savefig(save_path, dpi=200)
    plt.close()

def plot_confusion_matrix(cm, classes, save_path):
    """Heatmap confusion matrix with numbers and colorbar."""
    plt.figure()
    im = plt.imshow(cm, interpolation='nearest', cmap='viridis')
    plt.colorbar(im)
    ticks = np.arange(len(classes))
    plt.xticks(ticks, classes)
    plt.yticks(ticks, classes)
    # numbers
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, f"{cm[i, j]}", ha="center", va="center", color="orange")
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    ensure_dir(os.path.dirname(save_path))
    plt.savefig(save_path, dpi=200)
    plt.close()

def save_sample_input(dataloader, save_dir, filename="input_sample.png"):
    import torchvision
    ensure_dir(save_dir)
    sample_img, sample_label, _ = next(iter(dataloader))
    img = sample_img[0]
    inv_norm = torchvision.transforms.Normalize(
        mean=[-m/s for m, s in zip([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])],
        std=[1/s for s in [0.5, 0.5, 0.5]]
    )
    img_show = inv_norm(img).permute(1, 2, 0).clamp(0, 1)
    plt.imshow(img_show)
    plt.title(f"Sample Input (Label: {sample_label[0].item()})")
    plt.axis("off")
    path = os.path.join(save_dir, filename)
    plt.savefig(path, bbox_inches="tight", dpi=200)
    plt.close()

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
