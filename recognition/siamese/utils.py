# utils.py — plotting & small io helpers
import os
import numpy as np
import matplotlib.pyplot as plt
import torchvision

#  I/O 
def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

#  Plots 
def plot_lines(xs, ys_list, labels, title, xlabel, ylabel, save_path):
    """Generic multi-line plot (e.g., train vs val loss)."""
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
    """
    Save one example input image.
    Used for README or visualization.
    """
    os.makedirs(save_dir, exist_ok=True)
    sample_img, sample_label, _ = next(iter(dataloader))
    img = sample_img[0]  

    # [-1,1] -> [0,1]
    inv_norm = torchvision.transforms.Normalize(
        mean=[-m/s for m, s in zip([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])],
        std=[1/s for s in [0.5, 0.5, 0.5]]
    )
    img_show = inv_norm(img).permute(1, 2, 0).clamp(0, 1)

    plt.imshow(img_show)
    plt.title(f"Sample Input (Label: {sample_label[0].item()})")
    plt.axis("off")
    save_path = os.path.join(save_dir, filename)
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()
    print(f"[INFO] Saved sample input image → {save_path}")
