# utils.py — plotting & small io helpers
import os
import numpy as np
import matplotlib.pyplot as plt

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
