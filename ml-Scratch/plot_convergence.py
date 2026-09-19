"""
Convergence plots for GD-for-MSE from scratch (GATE M1, criterion 2).

Produces convergence_plot.png (300 DPI) with two panels:
  (a) Batch GD vs Mini-Batch GD vs SGD          (same lr, no decay)
  (b) Mini-Batch GD: constant lr vs lr_decay     (shows the noise floor vanishing)

y-axis = excess loss  J(w_t) - J(w*),  where w* is the exact OLS solution
(sklearn LinearRegression). Log scale, so a straight line = geometric convergence.

Run:  python plot_convergence.py
Needs final_ml_scratch_project_v3.py in the same folder.
"""
import matplotlib

matplotlib.use("Agg")  # no display needed
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_regression
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler

from final_ml_scratch_project_v3 import LinearRegressionScratch

EPS = 1e-12  # keeps log-scale finite if a curve hits the optimum exactly


def excess(model, j_star):
    return np.maximum(np.array(model.loss_history) - j_star, EPS)


def main(out_path="convergence_plot.png"):
    X, y = make_regression(n_samples=1000, n_features=5, noise=10.0, random_state=42)
    X = StandardScaler().fit_transform(X)
    sk = LinearRegression().fit(X, y)
    j_star = mean_squared_error(y, sk.predict(X))  # optimal training MSE

    epochs = 100

    # ---- (a) GD variants: same lr, momentum off so the comparison is clean ----
    variants = {
        "Batch GD": dict(batch_size=None, lr=0.1),
        "Mini-Batch GD (32)": dict(batch_size=32, lr=0.01),
        "SGD (batch=1)": dict(batch_size=1, lr=0.001),
    }
    curves_a = {
        name: excess(LinearRegressionScratch(epochs=epochs, momentum=0.0, **kw).fit(X, y), j_star)
        for name, kw in variants.items()
    }

    # ---- (b) mini-batch: constant lr vs lr_decay (momentum 0.9) ----
    common = dict(lr=0.02, epochs=epochs, batch_size=32, momentum=0.9, random_state=0)
    const = LinearRegressionScratch(**common).fit(X, y)
    decay = LinearRegressionScratch(lr_decay=1.0, **common).fit(X, y)

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), sharey=False)

    ax = axes[0]
    for name, c in curves_a.items():
        ax.plot(np.arange(1, len(c) + 1), c, label=name, linewidth=2)
    ax.set_title("(a) GD variants: convergence to the OLS optimum")

    ax = axes[1]
    ax.plot(np.arange(1, epochs + 1), excess(const, j_star), label="constant lr = 0.02", linewidth=2)
    ax.plot(np.arange(1, epochs + 1), excess(decay, j_star), label="lr_decay = 1.0", linewidth=2)
    ax.set_title("(b) Mini-Batch + momentum: noise floor vs decay")

    for ax in axes:
        ax.set_yscale("log")
        ax.set_xlabel("Epoch")
        ax.set_ylabel(r"Excess training MSE  $J(w_t) - J(w^*)$")
        ax.grid(True, which="both", alpha=0.3)
        ax.legend()

    fig.suptitle("Linear Regression from scratch (NumPy): convergence of the training loss", fontsize=13)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    print(f"saved {out_path}")
    for name, c in curves_a.items():
        print(f"{name:22s} final excess loss = {c[-1]:.3e}")
    print(f"{'Mini-Batch const lr':22s} final excess loss = {excess(const, j_star)[-1]:.3e}")
    print(f"{'Mini-Batch lr_decay':22s} final excess loss = {excess(decay, j_star)[-1]:.3e}")


if __name__ == "__main__":
    main()
