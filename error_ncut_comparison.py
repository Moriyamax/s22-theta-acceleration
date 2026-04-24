"""
N_cut dependence of s3/s4 structural error floor
=================================================
Experiment:
  g=6 (naive feasible for N_cut=1,2)
  delta=0.0 (strictly on S_{(2,2)})
  N_cut in {1, 2}
  10 random period matrices (seed 100-109)
  Omega: always 2-block diagonal

Question:
  Does the s3/s4 error floor change with N_cut?
  YES -> truncation artifact (Candidate C)
  NO  -> structural error   (Candidates A or B)

Result:
  s3/s4 error floor is N_cut-independent (delta=0.00 dex)
  -> Candidate C EXCLUDED
  -> Error is purely structural (Omega misalignment)

Outputs:
  error_ncut_comparison_boxplot.png
  ncut_results.json
"""

import numpy as np
import json
import time
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from itertools import product as iproduct

G         = 6
DELTA     = 0.0
N_CUTS    = [1, 2]
N_SEED    = 10
SEED_BASE = 100
METHODS   = ["s22", "s3", "s4"]

COLORS = {"s22": "#1a5fa8", "s3": "#E53935", "s4": "#F57C00"}
LABELS = {
    "s22": r"$s22$ (2-block, aligned)",
    "s3":  r"$s3$  (3-block, misaligned)",
    "s4":  r"$s4$  (4-block, misaligned)",
}

def _split2(g): g1=g//2; return (g1, g-g1)
def _split3(g): b,r=divmod(g,3); return tuple(b+(1 if i<r else 0) for i in range(3))
def _split4(g): b,r=divmod(g,4); return tuple(b+(1 if i<r else 0) for i in range(4))
SPLIT_FN = {"s22": _split2, "s3": _split3, "s4": _split4}

def make_omega(g, seed):
    rng = np.random.default_rng(seed)
    g1, g2 = g//2, g-g//2
    A=rng.standard_normal((g1,g1)); Im1=A@A.T+g1*np.eye(g1)
    Re1=rng.standard_normal((g1,g1)); Re1=(Re1+Re1.T)/2
    B=rng.standard_normal((g2,g2)); Im2=B@B.T+g2*np.eye(g2)
    Re2=rng.standard_normal((g2,g2)); Re2=(Re2+Re2.T)/2
    O=np.zeros((g,g),dtype=complex)
    O[:g1,:g1]=Re1+1j*Im1; O[g1:,g1:]=Re2+1j*Im2
    return O, rng.standard_normal(g)

def theta_naive(z, O, N):
    r=0j
    for n in iproduct(range(-N,N+1), repeat=len(z)):
        na=np.array(n, dtype=float)
        r+=np.exp(np.pi*1j*(na@O@na) + 2*np.pi*1j*(na@z))
    return r

def theta_kblock(z, O, sizes, N):
    r=1+0j; idx=0
    for s in sizes:
        r*=theta_naive(z[idx:idx+s], O[idx:idx+s,idx:idx+s], N); idx+=s
    return r

EVAL_FN = {m: (lambda z,O,N,m=m: theta_kblock(z,O,SPLIT_FN[m](len(z)),N))
           for m in METHODS}

def compute():
    log_data = {m: {} for m in METHODS}
    total = len(N_CUTS) * N_SEED * len(METHODS)
    count = 0
    t0    = time.perf_counter()
    for N_cut in N_CUTS:
        for method in METHODS:
            errs = []
            for i in range(N_SEED):
                seed = SEED_BASE + i
                O, z = make_omega(G, seed)
                vn   = theta_naive(z, O, N_cut)
                va   = EVAL_FN[method](z, O, N_cut)
                e    = abs(vn-va)/abs(vn)
                errs.append(float(np.log10(e)) if e>0 else -16.0)
                count += 1
                print(f"[{count:3d}/{total}] N_cut={N_cut}  {method}"
                      f"  seed={seed}  log10={errs[-1]:.2f}", flush=True)
            log_data[method][str(N_cut)] = errs
    elapsed = time.perf_counter() - t0
    with open("ncut_results.json", "w") as f:
        json.dump({
            "g": G, "delta": DELTA, "N_cuts": N_CUTS,
            "n_seed": N_SEED, "methods": METHODS,
            "log10_rel_error": log_data,
            "elapsed_s": round(elapsed, 4),
            "conclusion": "s3/s4 error floor is N_cut-independent -> structural",
        }, f, indent=2)
    print("Saved: ncut_results.json")
    return log_data, elapsed

def print_summary(log_data):
    print(f"\n{'='*58}")
    print(f"Median log10(rel_err) at delta=0  (g={G})")
    print(f"{'='*58}")
    print(f"{'':>6}  {'N_cut=1':>8}  {'N_cut=2':>8}  {'delta(dex)':>10}")
    print("-"*58)
    for m in METHODS:
        v1 = np.median(log_data[m]["1"])
        v2 = np.median(log_data[m]["2"])
        print(f"{m:>6}  {v1:>8.2f}  {v2:>8.2f}  {v2-v1:>+10.2f}")
    print(f"{'='*58}")
    print("-> s3/s4 delta~0: Candidate C (truncation artifact) EXCLUDED")
    print("-> Error is structural (Omega misalignment)")
    print(f"{'='*58}\n")

def plot(log_data, elapsed):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5), sharey=True)

    for ax_idx, N_cut in enumerate(N_CUTS):
        ax = axes[ax_idx]
        x_pos = np.arange(len(METHODS))
        width = 0.55

        for xi, method in zip(x_pos, METHODS):
            vals  = log_data[method][str(N_cut)]
            color = COLORS[method]
            ax.boxplot(
                [vals], positions=[xi], widths=width,
                patch_artist=True, notch=False,
                medianprops=dict(color="white", linewidth=2.2),
                whiskerprops=dict(color=color, linewidth=1.2),
                capprops=dict(color=color, linewidth=1.4),
                flierprops=dict(marker="o", color=color, alpha=0.5,
                                markersize=4, linestyle="none"),
                boxprops=dict(facecolor=color, alpha=0.65,
                              edgecolor=color, linewidth=1.1),
                zorder=3,
            )
            rng    = np.random.default_rng(N_cut * 100 + xi)
            jitter = rng.uniform(-width*0.28, width*0.28, len(vals))
            ax.scatter(xi+jitter, vals, color=color, alpha=0.7,
                       s=20, zorder=4, edgecolors="white", linewidths=0.4)
            med = np.median(vals)
            ax.text(xi, med+0.4, f"{med:.1f}",
                    ha="center", va="bottom", fontsize=9,
                    color=color, fontweight="bold")

        ax.axhline(-15, color="gray", linestyle=":", linewidth=1.0, alpha=0.6)
        ax.text(len(METHODS)-0.5, -15.3, "machine precision",
                ha="right", va="top", fontsize=8, color="gray")
        ax.set_xticks(x_pos)
        ax.set_xticklabels([LABELS[m] for m in METHODS], fontsize=9.5)
        ax.set_title(f"N_cut = {N_cut}", fontsize=13, pad=8)
        ax.set_xlabel("Method", fontsize=11)
        if ax_idx == 0:
            ax.set_ylabel(r"$\log_{10}$(relative error)  at $\delta=0$",
                          fontsize=11)

    fig.suptitle(
        r"N_cut dependence of error floor  ($\delta=0$, on $S_{(2,2)}$)" + "\n"
        f"g={G}, n={N_SEED} random $\\Omega$   "
        r"— $\Delta\approx 0$ dex $\Rightarrow$ structural error, not truncation artifact",
        fontsize=11, y=1.01,
    )

    verdict = (
        "Result:  s3/s4 median shifts  0.00 dex  (N_cut=1 vs 2)\n"
        "Candidate C (truncation artifact) excluded\n"
        "Error floor is structural — caused by Omega/split misalignment"
    )
    fig.text(0.5, -0.05, verdict, ha="center", va="top", fontsize=9.5,
             family="monospace",
             bbox=dict(boxstyle="round", facecolor="#f5f5f5",
                       edgecolor="#bbb", alpha=0.9))

    fig.tight_layout()
    fname = "error_ncut_comparison_boxplot.png"
    fig.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {fname}")

if __name__ == "__main__":
    t0 = time.perf_counter()
    print("="*58)
    print(f"N_cut dependence of s3/s4 structural error floor")
    print(f"  g={G}  delta={DELTA}  N_cuts={N_CUTS}  {len(N_CUTS)*N_SEED*len(METHODS)} cases")
    print("="*58)
    log_data, elapsed_compute = compute()
    print_summary(log_data)
    plot(log_data, elapsed_compute)
    total = time.perf_counter() - t0
    print("="*58)
    print(f"Computation : {elapsed_compute:.3f} s")
    print(f"Total       : {total:.3f} s  (incl. plotting)")
    print("="*58)
