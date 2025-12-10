# src/utils/plot_samples.py
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams

# Global styling
rcParams['font.family'] = 'serif'
rcParams['figure.dpi'] = 200
rcParams['axes.grid'] = True
rcParams['grid.alpha'] = 0.4
rcParams['grid.linestyle'] = '--'


def _plot_line(ax, x):
    """
    x : (T, D) numpy array
    Plots D feature curves on the same subplot.
    """
    T, D = x.shape
    t = np.arange(T)
    for d in range(D):
        ax.plot(t, x[:, d], linewidth=1.3)
    ax.set_xlabel("Time")
    ax.set_ylabel("Value")


def _plot_heatmap(ax, x):
    """
    x : (T, D) or (D, T) → convert to (D,T)
    """
    if x.shape[0] > x.shape[1]:  # if (T,D), convert to (D,T)
        x = x.T

    im = ax.imshow(x, cmap='gray', aspect='auto')
    ax.grid(False)
    ax.set_xlabel("Time")
    ax.set_ylabel("Features")


def save_sample_plots(X, split_name, outdir, num_samples=5, gt=None):
    """
    Generate num_samples PNGs.
    Each PNG contains vertically stacked subplots:

        1. Heatmap of X[i]
        2. Line plot of X[i]
        3. (Optional) Heatmap of ground-truth mask gt[i]

    Args:
        X : numpy array (N, T, D)
        split_name : str ('train'/'test'/etc)
        outdir : directory to save PNGs
        num_samples : number of random samples to visualize
        gt : numpy array (N, T, D) or (N, D, T) or None
    """
    os.makedirs(outdir, exist_ok=True)

    N = X.shape[0]
    idx = np.random.choice(N, size=min(num_samples, N), replace=False)

    for i in idx:
        # Determine number of rows dynamically
        n_rows = 2 + (1 if gt is not None else 0)

        fig, axes = plt.subplots(
            n_rows, 1,
            figsize=(10, 2.5 * n_rows),
            sharex=False
        )
        if n_rows == 1:
            axes = [axes]

        row = 0

        # --- (1) HEATMAP OF X[i] ---
        ax = axes[row]
        _plot_heatmap(ax, X[i])
        ax.set_title(f"{split_name} sample {i} — Data Heatmap", fontsize=11)
        row += 1

        # --- (2) LINE PLOT OF X[i] ---
        ax = axes[row]
        _plot_line(ax, X[i])
        ax.set_title(f"{split_name} sample {i} — Line Plot", fontsize=11)
        row += 1

        # --- (3) OPTIONAL GT HEATMAP ---
        if gt is not None:
            g = gt[i]
            ax = axes[row]
            _plot_heatmap(ax, g)
            ax.set_title(f"{split_name} sample {i} — GT Heatmap", fontsize=11)

        fig.tight_layout()

        # Save as <split>_sample_<i>.png
        out_path = os.path.join(outdir, f"{split_name}_sample_{i}.png")
        fig.savefig(out_path, bbox_inches='tight')
        plt.close(fig)


def plot_explainer_samples(
    X: np.ndarray,
    y: np.ndarray,
    attributions: np.ndarray,
    expl_name: str,
    expl_dir: str,
    num_samples: int = 5,
):
    """
    Save visualizations for a few samples under expl_dir/plots/.

    Parameters
    ----------
    X : np.ndarray
        Input data (N, T, D)
    y : np.ndarray
        Labels (N,)
    attributions : np.ndarray
        Attribution maps (N, T, D)
    expl_name : str
        Name of the explainer
    expl_dir : str
        Directory where explainer outputs are stored
    num_samples : int
        Number of sample plots to create
    """

    # ensure plot directory
    plots_dir = os.path.join(expl_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    N, T, D = X.shape
    num_samples = min(num_samples, N)

    # choose evenly spaced sample indices
    sample_ids = np.linspace(0, N - 1, num_samples, dtype=int)

    for idx, sid in enumerate(sample_ids):

        fig_h = 2 + 1 + 4   # approx size for 3 subplots
        fig, axes = plt.subplots(
            3,                  # signal, label, attribution
            1,
            figsize=(14, fig_h),
            dpi=180,
        )

        ax_signal, ax_label, ax_attr = axes

        # ------------------ 1) Raw Signal (D × T) ------------------
        ax_signal.imshow(
            X[sid].T,            # transpose to D×T for visualization
            aspect='auto',
            cmap='Greys',
            interpolation='nearest'
        )
        ax_signal.set_title(f"Signal (sample {sid})", fontsize=12)
        ax_signal.set_ylabel("Features")
        ax_signal.grid(alpha=0.25)

        # ------------------ 2) Label row ------------------
        lbl = y[sid]
        label_row = np.ones((1, T)) * lbl

        ax_label.imshow(
            label_row,
            aspect='auto',
            cmap='Greens',
            vmin=0,
            vmax=1
        )
        ax_label.set_title(f"Label = {lbl}", fontsize=12)
        ax_label.set_yticks([])
        ax_label.grid(alpha=0.25)

        # ------------------ 3) Attribution heatmap ------------------
        ax_attr.imshow(
            attributions[sid].T,         # D×T
            aspect='auto',
            cmap='Greens',
            interpolation='nearest'
        )
        ax_attr.set_title(f"{expl_name} Attribution", fontsize=12)
        ax_attr.set_ylabel("Features")
        ax_attr.set_xlabel("Time")
        ax_attr.grid(alpha=0.25)

        plt.tight_layout()

        out_path = os.path.join(plots_dir, f"sample_{idx}.png")
        plt.savefig(out_path, dpi=200, bbox_inches='tight')
        plt.close()

