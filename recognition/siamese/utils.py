# utils.py
# Utility functions for Siamese network training and evaluation.
# Includes directory management, plotting, sample saving, and feature extraction.
# Author: s4778251

import os
import torch
import matplotlib.pyplot as plt
import numpy as np
import torchvision
from params import MEAN, STD, SAVE_SAMPLE_NAME


def ensure_dir(path):
    """Create a directory if it does not already exist.

    Args:
        path (str | None): Directory path to create. If None or empty, nothing is created.

    Notes:
        This is a safe helper that mirrors `mkdir -p` behavior. It never raises if the
        directory already exists and does nothing for falsy paths.
    """
    if path and not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


def plot_lines(xs, ys_list, labels, title, xlabel, ylabel, save_path):
    """Plot one or more lines and save the figure to disk.

    Args:
        xs (Sequence[float | int]): Common x-axis values for all lines.
        ys_list (Sequence[Sequence[float]]): A list of y-value sequences, one per line.
        labels (Sequence[str]): Legend labels corresponding to each sequence in ys_list.
        title (str): Figure title.
        xlabel (str): X-axis label.
        ylabel (str): Y-axis label.
        save_path (str): File path where the plot image will be saved.

    Behavior:
        Creates a new figure, draws each line in order, adds a legend if labels are given,
        tightens the layout, ensures the parent directory exists, saves the image, and
        finally closes the figure to free memory.
    """
    plt.figure()
    for ys, lb in zip(ys_list, labels):
        plt.plot(xs, ys, label=lb)  # one line per series
    if labels:
        plt.legend()
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.tight_layout()
    ensure_dir(os.path.dirname(save_path))
    plt.savefig(save_path, dpi=200)
    plt.close()  # avoid accumulating open figures


def plot_confusion_matrix(cm, classes, save_path):
    """Visualize and save a confusion matrix as an image.

    Args:
        cm (np.ndarray): Square confusion matrix of integer counts with shape (C, C).
        classes (Sequence[str]): Class names for tick labels, length must be C.
        save_path (str): File path where the plot image will be saved.

    """
    plt.figure()
    plt.imshow(cm, interpolation='nearest', aspect='auto')
    plt.title('Confusion Matrix')
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    thresh = cm.max() / 2.0 if cm.size else 0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(
                j,
                i,
                format(cm[i, j], 'd'),
                ha="center",
                va="center",
                color="white" if cm[i, j] > thresh else "black",
            )
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    ensure_dir(os.path.dirname(save_path))
    plt.savefig(save_path, dpi=200)
    plt.close()


def save_sample_input(dataloader, save_dir, filename=SAVE_SAMPLE_NAME):
    """Save a single example image from a dataloader to disk for quick inspection.

    Args:
        dataloader (torch.utils.data.DataLoader): A dataloader that yields (image, label, index).
        save_dir (str): Directory where the image should be stored.
        filename (str): File name for the saved image.

    Behavior:
        Takes the first batch, inverts the normalization using the configured mean and std,
        converts the first image to HWC layout, clips to the valid range, and saves it as a PNG.
    """
    ensure_dir(save_dir)
    sample_img, sample_label, _ = next(iter(dataloader))
    img = sample_img[0]
    # Build an "inverse" normalization to undo the standardization for visualization.
    inv_norm = torchvision.transforms.Normalize(
        mean=[-m / s for m, s in zip(MEAN, STD)],
        std=[1 / s for s in STD],
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
    """Run a feature encoder over a dataset and collect embeddings and labels.

    Args:
        encoder (torch.nn.Module): Model that maps images to embedding vectors.
        loader (torch.utils.data.DataLoader): Dataloader yielding (image, label, index).
        device (str | torch.device): Device spec to run the encoder on, e.g. "cuda" or "cpu".

    Returns:
        Tuple[torch.Tensor, torch.Tensor]:
            A pair (X, y) where X has shape (N, D) of embeddings and y has shape (N,) of labels.

    Notes:
        The function prints progress every 10 batches and at completion. Gradients are disabled
        via the torch.no_grad decorator to reduce memory use and increase throughput.
    """
    encoder.eval()  # ensure batchnorm/dropout layers are in eval mode
    xs, ys = [], []
    total = len(loader)
    for i, (xb, yb, _) in enumerate(loader):
        feats = encoder(xb.to(device)).cpu()  # move inputs to device, bring features back to CPU
        xs.append(feats)
        ys.append(yb)
        if (i + 1) % 10 == 0 or (i + 1) == total:
            pct = 100.0 * (i + 1) / total
            print(f"\r[Extract] {pct:5.1f}% complete", end="")
    print()
    return torch.cat(xs), torch.cat(ys)
