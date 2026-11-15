from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np


def plot_roc_curves(curves: dict[str, tuple[np.ndarray, np.ndarray]], ax=None):
    """curves: label -> (alpha, beta) arrays."""
    ax = ax or plt.gca()
    for label, (alpha, beta) in curves.items():
        ax.plot(alpha, beta, label=label)
    ax.plot([0, 1], [0, 1], linestyle="--", color="grey", linewidth=1)
    ax.set_xlabel("false-positive rate (alpha)")
    ax.set_ylabel("true-positive rate (beta)")
    ax.legend()
    return ax


def plot_sigma_vs_beta(beta, sigma, ax=None, label=None):
    ax = ax or plt.gca()
    ax.plot(beta, sigma, label=label)
    ax.set_xlabel("true-positive rate (beta)")
    ax.set_ylabel("equilibrium probability of mimicry (sigma_S*)")
    if label:
        ax.legend()
    return ax


def plot_objective_vs_beta(beta, objective, ax=None, label=None):
    ax = ax or plt.gca()
    ax.plot(beta, objective, label=label)
    ax.set_xlabel("true-positive rate (beta)")
    ax.set_ylabel("designer's objective J(beta)")
    if label:
        ax.legend()
    return ax
