import os
from typing import Optional, List
from matplotlib import pyplot as plt
import numpy as np


def plot_time_series(
        X: np.ndarray, 
        method: str,
        title: str,
        xlabs: str, 
        ylabs: str,
        legends: Optional[List[str]],
        save_path: str,
        recreate: bool = False,
    ) -> str:
    """
    Simple time series plotter.
    - X can be (T,), (T,V), or (V,T)
    - xlabs, ylabs are direct axis-label strings
    - legends: list of names for each variable, or None for no legend
    """
    assert method in ["line", "spectrogram"], f"Unsupported method {method}"
    # Skip if already exists
    if os.path.exists(save_path) and not recreate:
        return save_path
    # Ensure parent folder exists
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # Normalize shape
    X = np.asarray(X).squeeze()

    if X.ndim == 1:
        X = X[:, None]              # (T,) → (T,1)
    elif X.ndim == 2:
        T, V = X.shape
        if T < V:                   # if transposed (V,T), fix it
            X = X.T
    else:
        raise ValueError(f"Unsupported shape {X.shape}")

    T, V = X.shape
    x = np.arange(T)

    plt.figure(figsize=(6, 4), dpi=100)

    if method == "spectrogram":
        for i in range(V):
            label_i = legends[i] if (legends is not None and i < len(legends)) else f"Var {i}"
            plt.subplot(V, 1, i + 1)
            plt.specgram(X[:, i], NFFT=64, Fs=1, noverlap=32)
            plt.ylabel(label_i)
            if i == 0:
                plt.title(title)
        plt.xlabel(xlabs)
    else:  # line plot
        for i in range(V):
            label_i = legends[i] if (legends is not None and i < len(legends)) else None
            plt.plot(x, X[:, i], linewidth=1, label=label_i)

        plt.xlabel(xlabs)
        plt.ylabel(ylabs)
        plt.title(title)

    # Only show legend if user gave one
    if legends is not None:
        plt.legend(fontsize=8, loc="upper right")

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

    return save_path


if __name__ == "__main__":
    # TEST / DEMONSTRATION / DEBUGGING
    T = 100
    V = 3
    X = np.random.randn(T, V)

    plot_time_series(
        X,
        title="Sample Time Series",
        xlabs="Time (seconds)",
        ylabs="Amplitude",
        legends=["X-axis", "Y-axis", "Z-axis"],
        save_path="./test_plot.png",
        recreate=True,
    )