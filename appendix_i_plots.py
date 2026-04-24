"""
Appendix I: S_{(2,2)} approximation error vs delta
  g=8, N_cut=1, delta in {0.0, 0.1, 0.3, 0.5, 0.7, 0.9}
  10 random period matrices per delta (seed 100-109)
  Total: 60 cases

Outputs:
  error_s22_boxplot.png
  error_s22_violin.png
  appendix_i_results.json
"""

import numpy as np
import json
import time
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from itertools import product as iproduct

# ============================================================
# Config
# ============================================================
G           = 8
N_CUT       = 1
DELTAS      = [0.0, 0.1, 0.3, 0.5, 0.7, 0.9]
N_SEED      = 10
SEED_OFFSET = 100
COLOR       = "#1a5fa8"

# ============================================================
# Period matrix: 2-block diagonal + off-diagonal perturbation
# ============================================================
def make_omega(g, delta, seed):
    rng = np.random.default_rng(seed)
    g1, g2 = g // 2, g - g // 2

    A  = rng.standard_normal((g1, g1))
    Im1 = A @ A.T + g1 * np.eye(g1)
    Re1 = rng.standard_normal((g1, g1)); Re1 = (Re1 + Re1.T) / 2
    Omega1 = Re1 + 1j * Im1

    B  = rng.standard_normal((g2, g2))
    Im2 = B @ B.T + g2 * np.eye(g2)
    Re2 = rng.standard_normal((g2, g2)); Re2 = (Re2 + Re2.T) / 2
    Omega2 = Re2 + 1j * Im2

    Omega = np.zeros((g, g), dtype=complex)
    Omega[:g1, :g1] = Omega1
    Omega[g1:, g1:] = Omega2

    if delta > 0:
        C = delta * (rng.standard_normal((g1, g2)) + 1j * rng.standard_normal((g1, g2)))
        Omega[:g1, g1:] = C
        Omega[g1:, :g1] = C.conj().T

    z = rng.standard_normal(g)
    return Omega, z

# ============================================================
# Core: naive theta sum  theta(z|Omega)  with truncation N_cut
# theta(z|Omega) = sum_{n in Z^g, |n_i|<=N_cut}
#                     exp(pi*i * n^T Omega n + 2*pi*i * n^T z)
# ============================================================
def theta_naive(z, Omega, N_cut):
    result = 0.0 + 0.0j
    for n in iproduct(range(-N_cut, N_cut + 1), repeat=len(z)):
        n_arr = np.array(n, dtype=float)
        phase = np.pi * 1j * (n_arr @ Omega @ n_arr) + 2 * np.pi * 1j * (n_arr @ z)
        result += np.exp(phase)
    return result

# ============================================================
# S_{(2,2)} decomposition:
#   theta(z|Omega) = theta(z1|Omega1) * theta(z2|Omega2)
# ============================================================
def theta_s22(z, Omega, N_cut):
    g1 = len(z) // 2
    return (theta_naive(z[:g1],  Omega[:g1, :g1],  N_cut) *
            theta_naive(z[g1:],  Omega[g1:, g1:],  N_cut))

# ============================================================
# Fay theoretical bound: |log(delta)| / (2*pi*Im_Omega_mean)
# (used as reference curve; only valid for delta > 0)
# ============================================================
def fay_bound(delta, im_omega_mean):
    if delta <= 0:
        return None
    return abs(np.log(delta)) / (2 * np.pi * im_omega_mean)

# ============================================================
# Main computation
# ============================================================
def compute():
    t_start = time.perf_counter()
    total   = len(DELTAS) * N_SEED
    count   = 0

    log_data = {}   # delta_str -> list of log10(rel_err)
    lin_data = {}   # delta_str -> list of rel_err
    im_means = []   # for Fay curve

    for delta in DELTAS:
        log_errs, lin_errs = [], []
        for i in range(N_SEED):
            seed  = SEED_OFFSET + i
            Omega, z = make_omega(G, delta, seed)

            v_naive = theta_naive(z, Omega, N_CUT)
            v_s22   = theta_s22(z, Omega, N_CUT)

            rel_err = abs(v_naive - v_s22) / abs(v_naive)
            lin_errs.append(float(rel_err))
            log_errs.append(float(np.log10(rel_err)) if rel_err > 0 else -16.0)

            # collect Im(Omega) diagonal mean for Fay reference
            im_means.append(float(np.mean(np.diag(Omega.imag))))

            count += 1
            print(f"[{count:2d}/{total}] delta={delta}  seed={seed}"
                  f"  rel_err={rel_err:.3e}")

        log_data[str(delta)] = log_errs
        lin_data[str(delta)] = lin_errs

    elapsed = time.perf_counter() - t_start

    # Save JSON
    out = {
        "g": G, "N_cut": N_CUT, "deltas": DELTAS,
        "n_seed": N_SEED,
        "seed_range": [SEED_OFFSET, SEED_OFFSET + N_SEED - 1],
        "elapsed_s": round(elapsed, 4),
        "log10_rel_error": log_data,
        "rel_error":       lin_data,
    }
    with open("appendix_i_results.json", "w") as f:
        json.dump(out, f, indent=2)

    im_mean_global = float(np.mean(im_means))
    return log_data, elapsed, im_mean_global

# ============================================================
# Boxplot
# ============================================================
def plot_boxplot(log_data, im_mean, elapsed):
    fig, ax = plt.subplots(figsize=(8, 5))

    x          = np.arange(len(DELTAS))
    plot_data  = [log_data[str(d)] for d in DELTAS]

    bp = ax.boxplot(
        plot_data,
        positions=x,
        widths=0.5,
        patch_artist=True,
        notch=False,
        medianprops=dict(color="white", linewidth=2.0),
        whiskerprops=dict(color=COLOR, linewidth=1.2),
        capprops=dict(color=COLOR, linewidth=1.5),
        flierprops=dict(marker="o", color=COLOR, alpha=0.6,
                        markersize=5, linestyle="none"),
        boxprops=dict(facecolor=COLOR, alpha=0.7,
                      edgecolor=COLOR, linewidth=1.2),
        zorder=3,
    )

    # Fay theoretical bound curve
    fay_x, fay_y = [], []
    for xi, d in zip(x, DELTAS):
        fb = fay_bound(d, im_mean)
        if fb is not None:
            fay_x.append(xi); fay_y.append(np.log10(fb))
    if fay_x:
        ax.plot(fay_x, fay_y, color="#E53935", linestyle="--",
                linewidth=1.6, zorder=5, label="Fay bound (theory)")

    # Machine precision floor
    ax.axhline(-15, color="gray", linestyle=":", linewidth=1.0, alpha=0.6, zorder=1)
    ax.text(len(DELTAS) - 0.55, -15.35, "machine precision",
            ha="right", va="top", fontsize=9, color="gray")

    ax.set_xticks(x)
    ax.set_xticklabels([f"δ={d}" for d in DELTAS], fontsize=11)
    ax.set_xlabel(r"$\delta$  (distance from $S_{(2,2)}$)", fontsize=12)
    ax.set_ylabel(r"$\log_{10}$(relative error)", fontsize=12)
    ax.set_title(
        f"$S_{{(2,2)}}$ Approximation Error vs $\\delta$\n"
        f"(g={G}, N_cut={N_CUT}, n={N_SEED} random $\\Omega$ per point, "
        f"elapsed {elapsed:.2f}s)",
        fontsize=12, pad=10,
    )
    if fay_x:
        ax.legend(fontsize=10, framealpha=0.85)

    fig.tight_layout()
    fig.savefig("error_s22_boxplot.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("Saved: error_s22_boxplot.png")

# ============================================================
# Violin plot
# ============================================================
def plot_violin(log_data, im_mean, elapsed):
    fig, ax = plt.subplots(figsize=(8, 5))

    x         = np.arange(len(DELTAS))
    plot_data = [log_data[str(d)] for d in DELTAS]

    parts = ax.violinplot(
        plot_data,
        positions=x,
        widths=0.5,
        showmeans=False,
        showmedians=True,
        showextrema=True,
    )
    for pc in parts["bodies"]:
        pc.set_facecolor(COLOR); pc.set_alpha(0.55)
        pc.set_edgecolor(COLOR); pc.set_linewidth(0.9)
    for partname in ("cbars", "cmins", "cmaxes", "cmedians"):
        if partname in parts:
            parts[partname].set_edgecolor(COLOR)
            parts[partname].set_linewidth(1.6)

    # Individual points (N=10 so plot all)
    rng = np.random.default_rng(0)
    for xi, d in zip(x, DELTAS):
        vals   = log_data[str(d)]
        jitter = rng.uniform(-0.08, 0.08, len(vals))
        ax.scatter(xi + jitter, vals, color=COLOR, alpha=0.75, s=22,
                   zorder=4, edgecolors="white", linewidths=0.5)

    # Fay theoretical bound curve
    fay_x, fay_y = [], []
    for xi, d in zip(x, DELTAS):
        fb = fay_bound(d, im_mean)
        if fb is not None:
            fay_x.append(xi); fay_y.append(np.log10(fb))
    if fay_x:
        ax.plot(fay_x, fay_y, color="#E53935", linestyle="--",
                linewidth=1.6, zorder=5, label="Fay bound (theory)")

    # Machine precision floor
    ax.axhline(-15, color="gray", linestyle=":", linewidth=1.0, alpha=0.6, zorder=1)
    ax.text(len(DELTAS) - 0.55, -15.35, "machine precision",
            ha="right", va="top", fontsize=9, color="gray")

    ax.set_xticks(x)
    ax.set_xticklabels([f"δ={d}" for d in DELTAS], fontsize=11)
    ax.set_xlabel(r"$\delta$  (distance from $S_{(2,2)}$)", fontsize=12)
    ax.set_ylabel(r"$\log_{10}$(relative error)", fontsize=12)
    ax.set_title(
        f"$S_{{(2,2)}}$ Approximation Error vs $\\delta$ — Violin\n"
        f"(g={G}, N_cut={N_CUT}, n={N_SEED} random $\\Omega$ per point, "
        f"elapsed {elapsed:.2f}s)",
        fontsize=12, pad=10,
    )
    if fay_x:
        ax.legend(fontsize=10, framealpha=0.85)

    fig.tight_layout()
    fig.savefig("error_s22_violin.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("Saved: error_s22_violin.png")

# ============================================================
# Entry point
# ============================================================
if __name__ == "__main__":
    t0 = time.perf_counter()

    print("=" * 55)
    print(f"Appendix I: S_{{(2,2)}} error experiment")
    print(f"  g={G}  N_cut={N_CUT}  deltas={DELTAS}")
    print(f"  seeds {SEED_OFFSET}–{SEED_OFFSET+N_SEED-1}  ({len(DELTAS)*N_SEED} cases)")
    print("=" * 55)

    log_data, elapsed_compute, im_mean = compute()

    print("-" * 55)
    print("Generating plots...")
    plot_boxplot(log_data, im_mean, elapsed_compute)
    plot_violin(log_data, im_mean, elapsed_compute)

    elapsed_total = time.perf_counter() - t0
    print("=" * 55)
    print(f"Computation time : {elapsed_compute:.3f} s  ({len(DELTAS)*N_SEED} cases)")
    print(f"Total time       : {elapsed_total:.3f} s  (incl. plotting)")
    print("=" * 55)
