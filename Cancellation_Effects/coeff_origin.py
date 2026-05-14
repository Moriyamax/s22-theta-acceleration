"""
C(|z|)係数 1/1.41 の代数的起源特定
仮説: sin展開に |n_j| 因子が抜けている

正しい式:
  |grad_j| ∝ Σ_{n≠0} |n_j| · |sin(2πnᵀz)| · exp(−πnᵀIm(Ω)n)
  |μ_A| = |grad|/(2π) = √(Σ_j grad_j²) / (2π)

sin展開(現行): C_theory = Σ_{n≠0} |sin| · exp(...)  ← |n_j|なし
修正版:        C_fixed  = Σ_{n≠0} |n| · |sin| · exp(...) / (2π)
                       または
               C_fixed2 = √(Σ_j [Σ_{n≠0} n_j · sin · exp(...)]²) / (2π)

実行: python coeff_origin.py
"""

import numpy as np
from itertools import product
import matplotlib.pyplot as plt

# ---- 基本関数 ----

def make_omega_pure_imag(g, lmin, seed=42):
    rng = np.random.default_rng(seed)
    A = rng.standard_normal((g, g))
    Q, _ = np.linalg.qr(A)
    eigs = lmin + rng.exponential(0.3, g)
    return 1j * (Q @ np.diag(eigs) @ Q.T)


def compute_all(Om, z, N_cut=2):
    """theta、grad、各展開式を一括計算"""
    g = Om.shape[0]
    lmin = np.linalg.eigvalsh(Om.imag).min()
    n0 = tuple([0]*g)

    theta = 0j
    grad = np.zeros(g, dtype=complex)

    # sin展開の各項を蓄積
    sin_sum_plain  = 0.0   # Σ|sin|·exp (現行)
    sin_sum_nj     = 0.0   # Σ|n|·|sin|·exp (|n|補正)
    sin_vec        = np.zeros(g)  # Σ_n n_j·sin·exp の各成分

    for n in product(range(-N_cut, N_cut+1), repeat=g):
        nv = np.array(n, dtype=float)
        exp_im = np.exp(-np.pi * nv @ Om.imag @ nv)
        phase  = np.pi * 1j * (nv @ Om @ nv + 2 * nv @ z)
        term   = np.exp(phase)

        theta += term

        if n != n0:
            sin_val = np.sin(2 * np.pi * nv @ z)
            n_norm  = np.linalg.norm(nv)

            # grad の各成分
            for j in range(g):
                grad[j] += 2 * np.pi * 1j * nv[j] * term

            # sin展開（現行）: |sin|·exp
            sin_sum_plain += abs(sin_val) * exp_im

            # sin展開（|n|補正）: |n|·|sin|·exp
            sin_sum_nj += n_norm * abs(sin_val) * exp_im

            # ベクトル展開: n_j·sin·exp（符号付き）
            sin_vec += nv * sin_val * exp_im

    mu_A = grad / (2 * np.pi * 1j)
    exp_lmin = np.exp(-np.pi * lmin)

    # C_measured
    C_me = np.linalg.norm(mu_A) / (abs(theta) * exp_lmin + 1e-30)

    # C_theory (現行)
    C_th_plain = sin_sum_plain / (exp_lmin * abs(theta) + 1e-30)

    # C_theory_nj (|n|補正)
    # |grad| ≈ 2 * sin_sum_nj * 2π → |μ_A| ≈ 2 * sin_sum_nj
    # （n/−nペアで factor 2、grad_j = 2πi·n_j·term）
    # |μ_A| = |grad|/(2π) ≈ 2 * sin_sum_nj（n/−nペアで exp(-πnᵀImΩn)が2倍されている）
    C_th_nj = 2 * sin_sum_nj / (exp_lmin * abs(theta) + 1e-30)

    # C_theory_vec (ベクトル展開、最も正確)
    # grad_j = −4π·Σ_{n>0} n_j·sin(2πnᵀz)·exp(−πnᵀIm(Ω)n)
    # |grad| = 4π·|sin_vec|
    # |μ_A| = |grad|/(2π) = 2·|sin_vec|
    C_th_vec = 2 * np.linalg.norm(sin_vec) / (exp_lmin * abs(theta) + 1e-30)

    return {
        'C_measured': C_me,
        'C_plain':    C_th_plain,
        'C_nj':       C_th_nj,
        'C_vec':      C_th_vec,
        'ratio_plain': C_th_plain / (C_me + 1e-30),
        'ratio_nj':    C_th_nj   / (C_me + 1e-30),
        'ratio_vec':   C_th_vec  / (C_me + 1e-30),
        'lmin': lmin,
    }


def main():
    print("=" * 60)
    print("係数 1/1.41 の代数的起源特定")
    print("=" * 60)

    lmin_vals = [2.0, 3.0, 5.0]
    z_scales  = [0.05, 0.1, 0.2, 0.5, 1.0]
    gb = 3
    N_cut = 2
    n_seeds = 5

    print(f"\n{'lmin':>6} {'|z|':>6} {'ratio_plain':>12} "
          f"{'ratio_nj':>10} {'ratio_vec':>10}")
    print("-" * 50)

    all_res = []
    for lmin in lmin_vals:
        for z_scale in z_scales:
            ratios_plain, ratios_nj, ratios_vec = [], [], []
            for seed in range(n_seeds):
                Om = make_omega_pure_imag(gb, lmin, seed)
                rng = np.random.default_rng(seed + 1000)
                z = rng.standard_normal(gb) * z_scale
                r = compute_all(Om, z, N_cut)
                ratios_plain.append(r['ratio_plain'])
                ratios_nj.append(r['ratio_nj'])
                ratios_vec.append(r['ratio_vec'])

            rp = np.mean(ratios_plain)
            rn = np.mean(ratios_nj)
            rv = np.mean(ratios_vec)
            print(f"{lmin:6.1f} {z_scale:6.3f} {rp:12.3f} {rn:10.3f} {rv:10.3f}")
            all_res.append({'lmin': lmin, 'z_scale': z_scale,
                            'rp': rp, 'rn': rn, 'rv': rv})

    print()
    print("ratio=1.000に最も近いものが正しい展開式")
    print("  ratio_plain: 現行（|n|因子なし）")
    print("  ratio_nj:    |n|補正あり")
    print("  ratio_vec:   ベクトル展開（最も厳密）")

    # ---- プロット ----
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Panel A: z_scale vs ratio（lmin=3.0）
    ax = axes[0]
    subset = [r for r in all_res if r['lmin'] == 3.0]
    zs = [r['z_scale'] for r in subset]
    ax.plot(zs, [r['rp'] for r in subset], 'o-', label='ratio_plain (current)')
    ax.plot(zs, [r['rn'] for r in subset], 's-', label='ratio_|n| corrected')
    ax.plot(zs, [r['rv'] for r in subset], '^-', label='ratio_vec (exact)', color='green')
    ax.axhline(1.0, color='red', linestyle='--', label='target=1.0')
    ax.axhline(1/np.sqrt(2), color='gray', linestyle=':', label='1/√2=0.707')
    ax.set_xlabel('|z| scale')
    ax.set_ylabel('C_theory / C_measured')
    ax.set_title('Panel A: ratio vs |z|\n(lmin=3.0, gb=3)')
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    # Panel B: lmin vs ratio（z_scale=0.2）
    ax2 = axes[1]
    for ls, color in [(2.0,'steelblue'),(3.0,'tomato'),(5.0,'green')]:
        subset2 = [r for r in all_res if r['lmin'] == ls and r['z_scale'] == 0.2]
        if subset2:
            r = subset2[0]
            ax2.scatter([ls]*3,
                        [r['rp'], r['rn'], r['rv']],
                        marker=['o','s','^'], s=80, color=color)
    # 凡例用
    ax2.scatter([], [], marker='o', color='gray', label='plain')
    ax2.scatter([], [], marker='s', color='gray', label='|n| corrected')
    ax2.scatter([], [], marker='^', color='gray', label='vec (exact)')
    ax2.axhline(1.0, color='red', linestyle='--', label='target=1.0')
    ax2.set_xlabel('lmin')
    ax2.set_ylabel('C_theory / C_measured')
    ax2.set_title('Panel B: ratio vs lmin\n(|z|=0.2, gb=3)')
    ax2.legend(fontsize=8); ax2.grid(True, alpha=0.3)

    plt.suptitle('Origin of the 1/1.41 coefficient in C(|z|)', fontsize=12)
    plt.tight_layout()
    plt.savefig("coeff_origin.png", dpi=130, bbox_inches='tight')
    print("\nプロット: coeff_origin.png")
    print("完了")


if __name__ == "__main__":
    main()
