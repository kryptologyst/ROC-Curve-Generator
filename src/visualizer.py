import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Optional
from loguru import logger


class ROCVisualizer:
    @staticmethod
    def plot(fpr, tpr, auc_scores, save_path=None):
        plt.figure(figsize=(8, 6))
        for label in fpr:
            plt.plot(fpr[label], tpr[label], linewidth=2, label=f"{label} (AUC={auc_scores[label]:.3f})")
        plt.plot([0, 1], [0, 1], "k--", linewidth=1, alpha=0.5)
        plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
        plt.title("ROC Curve"); plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
