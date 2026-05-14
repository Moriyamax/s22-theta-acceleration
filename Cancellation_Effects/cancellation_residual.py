"""
残差スケーリング特定解析
目的: n/−n対称性が不完全になるRe(Ω)≠0、z≠0の場合の残差の代数的起源を特定する

前回確認:
  - Re(Ω)=0, z=0: grad=0 厳密に成立（証明済み）
  - Re(Ω)≠0 or z≠0: 不完全キャンセル → 残差が残る

今回の問い:
  Q1: Re(Ω)の寄与とzの寄与はどちらが支配的か
  Q2: 残差はどのスケーリングを持つか（Re(Ω)の大きさ、|z|の大きさで）
  Q3: M_2以降の高次項は残差を説明するか、それとも別の機構か

実行: python cancellation_residual.py
"""

import numpy as np
from itertools import product
import matplotlib.pyplot as plt

# ---- 基本関数 ----

def make_omega_controlled(gb, lmin, re_scale=0.0, seed=42):
    """
    Re(Ω)のスケールを制御できるブロック行列生成
    re_scale=0: 純虚数行列
    re_scale>0: Re(Ω)の大きさ
    """
    rng = np.random.default_rng(seed)
    A = rng.standard_normal((gb, gb))
    Q, _ = np.linalg.qr(A)
    eigs = lmin + rng.exponential(0.3, gb)
    ImOm = Q @ np.diag(eigs) @ Q.T

    if re_scale > 0:
        ReOm = rng.standard_normal((gb, gb)) * re_scale
        ReOm = (ReOm + ReOm.T) / 2
    else:
        ReOm = np.zeros((gb, gb))

    return ReOm + 1j * ImOm


def theta_and_grad(Om, z, N_cut=2):
    """theta値と勾配を同時計算（格子点ループ1回）"""
    g = Om.shape[0]
    ns = range(-N_cut, N_cut + 1)
    theta = 0.0 + 0j
    grad = np.zeros(g, dtype=complex)
    for n in product(ns, repeat=g):
        n_arr = np.array(n, dtype=float)
        phase = np.pi * 1j * (n_arr @ Om @ n_arr + 2 * n_arr @ z)
        term = np.exp(phase)
        theta += term
        grad += 2 * np.pi * 1j * n_arr * term
    return theta, grad


# ============================================================
# 解析1: Re(Ω)スケールvs残差
# ============================================================

def analysis_reom_scaling(lmin=3.0, gb=4, N_cut=2, n_seeds=10):
    """
    Re(Ω)の大きさ（re_scale）に対する|mu_A|/|Theta_A|のスケーリング
    z=0固定でRe(Ω)の効果のみを見る
    """
    re_scales = [0.0, 0.01, 0.05, 0.1, 0.2, 0.5, 1.0]
    results = []

    print(f"\n[解析1] Re(Ω)スケール vs 残差 (lmin={lmin}, gb={gb}, z=0)")
    print(f"{'re_scale':>10} {'|mu_A|/|ThA|':>14} {'std':>10} {'ratio/re²':>12}")
    print("-" * 52)

    for re_scale in re_scales:
        ratios = []
        for seed in range(n_seeds):
            Om = make_omega_controlled(gb, lmin, re_scale, seed)
            z = np.zeros(gb)
            theta, grad = theta_and_grad(Om, z, N_cut)
            mu_A = grad / (2 * np.pi * 1j)
            ratio = np.linalg.norm(mu_A) / (abs(theta) + 1e-30)
            ratios.append(ratio)

        mean_r = np.mean(ratios)
        std_r = np.std(ratios)
        # re_scale=0の比との比 → re²スケーリングか確認
        results.append({'re_scale': re_scale, 'mean': mean_r, 'std': std_r})
        ratio_re2 = mean_r / (re_scale**2 + 1e-30) if re_scale > 0 else float('nan')
        print(f"{re_scale:10.3f} {mean_r:14.3e} {std_r:10.2e} {ratio_re2:12.3e}")

    return results


# ============================================================
# 解析2: |z|スケールvs残差
# ============================================================

def analysis_z_scaling(lmin=3.0, gb=4, N_cut=2, n_seeds=10):
    """
    |z|の大きさに対する|mu_A|/|Theta_A|のスケーリング
    Re(Ω)=0固定でzの効果のみを見る
    """
    z_scales = [0.0, 0.01, 0.05, 0.1, 0.2, 0.5, 1.0]
    results = []

    print(f"\n[解析2] |z|スケール vs 残差 (lmin={lmin}, gb={gb}, Re(Ω)=0)")
    print(f"{'|z|':>8} {'|mu_A|/|ThA|':>14} {'std':>10} {'ratio/|z|':>12}")
    print("-" * 50)

    for z_scale in z_scales:
        ratios = []
        for seed in range(n_seeds):
            Om = make_omega_controlled(gb, lmin, re_scale=0.0, seed=seed)
            rng = np.random.default_rng(seed + 500)
            z = rng.standard_normal(gb) * z_scale
            theta, grad = theta_and_grad(Om, z, N_cut)
            mu_A = grad / (2 * np.pi * 1j)
            ratio = np.linalg.norm(mu_A) / (abs(theta) + 1e-30)
            ratios.append(ratio)

        mean_r = np.mean(ratios)
        std_r = np.std(ratios)
        results.append({'z_scale': z_scale, 'mean': mean_r, 'std': std_r})
        ratio_z = mean_r / (z_scale + 1e-30) if z_scale > 0 else float('nan')
        print(f"{z_scale:8.3f} {mean_r:14.3e} {std_r:10.2e} {ratio_z:12.3e}")

    return results


# ============================================================
# 解析3: M_1残差の代数的構造
# ============================================================

def analysis_m1_residual_structure(lmin=3.0, gb=4, N_cut=2, n_seeds=5):
    """
    Re(Ω)≠0のとき、M_1 = 2*mu_A^T C mu_B の残差が
    どの代数的量で決まるかを特定する

    n/−n対称性の破れ: Re(Ω)があると
    exp(πi n^T Ω n) = exp(-π n^T Im(Ω) n) * exp(πi n^T Re(Ω) n)
    n→−n: exp(-π n^T Im(Ω) n) は不変, exp(πi n^T Re(Ω) n) も不変
    → Re(Ω)は対称性を壊さない！

    z≠0のとき:
    exp(2πi n^T z) → exp(-2πi n^T z) でn→−n
    → z≠0が対称性を壊す主因

    これを数値確認する
    """
    print(f"\n[解析3] M_1残差の代数的構造確認 (lmin={lmin}, gb={gb})")
    print("  Re(Ω)とzの対称性破れへの寄与を分離")
    print()

    configs = [
        ('Re=0, z=0',    0.0, 0.0),
        ('Re=0.5, z=0',  0.5, 0.0),
        ('Re=1.0, z=0',  1.0, 0.0),
        ('Re=0, z=0.1',  0.0, 0.1),
        ('Re=0, z=0.5',  0.0, 0.5),
        ('Re=0.5, z=0.5',0.5, 0.5),
    ]

    print(f"  {'設定':20} {'|mu_A|/|ThA|':>14} {'|sin(phi_n)|_max':>18}")
    print("  " + "-" * 56)

    results = []
    for label, re_scale, z_scale in configs:
        ratios = []
        sin_maxes = []
        for seed in range(n_seeds):
            Om = make_omega_controlled(gb, lmin, re_scale, seed)
            rng = np.random.default_rng(seed + 600)
            z = rng.standard_normal(gb) * z_scale

            theta, grad = theta_and_grad(Om, z, N_cut)
            mu_A = grad / (2 * np.pi * 1j)
            ratio = np.linalg.norm(mu_A) / (abs(theta) + 1e-30)
            ratios.append(ratio)

            # 位相の非対称性を測定
            # n→−nでの位相差 = 4π n^T z（Re(Ω)の寄与は相殺）
            sin_vals = []
            ns = range(-N_cut, N_cut + 1)
            for n in product(ns, repeat=gb):
                n_arr = np.array(n, dtype=float)
                if np.any(n_arr != 0):
                    phase_diff = 2 * np.pi * n_arr @ z  # n→−nでの位相差の半分
                    sin_vals.append(abs(np.sin(phase_diff)))
            sin_maxes.append(max(sin_vals) if sin_vals else 0)

        mean_r = np.mean(ratios)
        mean_sin = np.mean(sin_maxes)
        print(f"  {label:20} {mean_r:14.3e} {mean_sin:18.3e}")
        results.append({'label': label, 're_scale': re_scale, 'z_scale': z_scale,
                        'mean_ratio': mean_r, 'mean_sin_max': mean_sin})

    print()
    print("  【予測】Re(Ω)は対称性を壊さない（n→−nで位相が±で相殺）")
    print("  　　　　z≠0が主な対称性破れの原因")
    print("  　　　　残差 ∝ |z| × exp(-π λ_min)")
    return results


# ============================================================
# 解析4: 残差の精密スケーリング法則
# ============================================================

def analysis_residual_scaling_law(lmin_vals, gb=4, N_cut=2, n_seeds=10):
    """
    残差 |mu_A|/|Theta_A| の精密なスケーリング法則を決定
    Re(Ω)=0固定、z_scaleを変化させて

    仮説: |mu_A|/|Theta_A| ≈ C(lmin) × |z| × exp(-π λ_min)
    """
    z_scales = [0.0, 0.05, 0.1, 0.2, 0.5, 1.0]

    print(f"\n[解析4] 残差スケーリング法則の精密測定 (Re(Ω)=0, gb={gb})")
    print("  仮説: residual ≈ C × |z| × exp(-π λ_min)")
    print()
    print(f"  {'lmin':>6} {'|z|':>6} {'residual':>12} {'C = res/(|z|*exp(-πλ))':>24}")
    print("  " + "-" * 54)

    results = []
    for lmin in lmin_vals:
        for z_scale in z_scales[1:]:  # z=0はスキップ
            ratios = []
            for seed in range(n_seeds):
                Om = make_omega_controlled(gb, lmin, re_scale=0.0, seed=seed)
                rng = np.random.default_rng(seed + 700)
                z = rng.standard_normal(gb) * z_scale
                theta, grad = theta_and_grad(Om, z, N_cut)
                mu_A = grad / (2 * np.pi * 1j)
                ratio = np.linalg.norm(mu_A) / (abs(theta) + 1e-30)
                ratios.append(ratio)

            mean_r = np.mean(ratios)
            exp_lmin = np.exp(-np.pi * lmin)
            C = mean_r / (z_scale * exp_lmin)
            print(f"  {lmin:6.1f} {z_scale:6.3f} {mean_r:12.3e} {C:24.3e}")
            results.append({'lmin': lmin, 'z_scale': z_scale,
                            'residual': mean_r, 'C': C})
    return results


# ============================================================
# メイン
# ============================================================

def main():
    print("=" * 60)
    print("残差スケーリング特定解析")
    print("Re(Ω)≠0、z≠0の場合の追加抑制の起源")
    print("=" * 60)

    lmin = 3.0
    gb = 4
    N_cut = 2

    # 解析1: Re(Ω)の効果
    res1 = analysis_reom_scaling(lmin=lmin, gb=gb, N_cut=N_cut, n_seeds=8)

    # 解析2: zの効果
    res2 = analysis_z_scaling(lmin=lmin, gb=gb, N_cut=N_cut, n_seeds=8)

    # 解析3: 代数的構造（Re(Ω) vs z のどちらが対称性を壊すか）
    res3 = analysis_m1_residual_structure(lmin=lmin, gb=gb, N_cut=N_cut, n_seeds=5)

    # 解析4: 精密スケーリング法則
    res4 = analysis_residual_scaling_law(
        lmin_vals=[1.0, 2.0, 3.0, 5.0], gb=gb, N_cut=N_cut, n_seeds=8)

    # ---- プロット ----
    fig, axes = plt.subplots(2, 2, figsize=(13, 10))

    # Panel A: Re(Ω)スケール vs 残差
    ax = axes[0, 0]
    re_sc = [r['re_scale'] for r in res1]
    means = [r['mean'] for r in res1]
    ax.plot(re_sc, means, 'o-', color='steelblue')
    ax.set_xlabel('Re(Ω) scale')
    ax.set_ylabel(r'$|\mu_A|/|\Theta_A|$')
    ax.set_yscale('log')
    ax.set_title('Panel A: Re(Ω) scale vs residual\n(z=0, lmin=3.0)')
    ax.grid(True, alpha=0.3)

    # Panel B: |z|スケール vs 残差
    ax = axes[0, 1]
    z_sc = [r['z_scale'] for r in res2]
    means2 = [r['mean'] for r in res2]
    ax.plot(z_sc, means2, 's-', color='tomato')
    # 線形フィット（log-linearで）
    z_nonzero = [z for z in z_sc if z > 0]
    m_nonzero = [m for z, m in zip(z_sc, means2) if z > 0]
    if len(z_nonzero) > 1:
        coeffs = np.polyfit(np.log(z_nonzero), np.log(m_nonzero), 1)
        z_fit = np.linspace(min(z_nonzero), max(z_nonzero), 50)
        ax.plot(z_fit, np.exp(coeffs[1]) * z_fit**coeffs[0],
                '--', color='orange', label=f'slope={coeffs[0]:.2f}')
        ax.legend()
    ax.set_xlabel('|z| scale')
    ax.set_ylabel(r'$|\mu_A|/|\Theta_A|$')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_title('Panel B: |z| scale vs residual\n(Re(Ω)=0, lmin=3.0)')
    ax.grid(True, alpha=0.3)

    # Panel C: Re vs z の寄与比較
    ax = axes[1, 0]
    labels = [r['label'] for r in res3]
    ratios = [r['mean_ratio'] for r in res3]
    colors = ['steelblue', 'steelblue', 'steelblue', 'tomato', 'tomato', 'purple']
    bars = ax.bar(range(len(labels)), ratios, color=colors)
    ax.set_yscale('log')
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=30, ha='right', fontsize=8)
    ax.set_ylabel(r'$|\mu_A|/|\Theta_A|$')
    ax.set_title('Panel C: Re(Ω) vs z contribution\n(blue=Re only, red=z only)')
    ax.grid(True, alpha=0.3, axis='y')

    # Panel D: 精密スケーリング C = residual / (|z| * exp(-π λ))
    ax = axes[1, 1]
    for lmin_val in [1.0, 2.0, 3.0, 5.0]:
        subset = [r for r in res4 if r['lmin'] == lmin_val]
        zs = [r['z_scale'] for r in subset]
        Cs = [r['C'] for r in subset]
        ax.plot(zs, Cs, 'o-', label=f'λ={lmin_val}')
    ax.set_xlabel('|z| scale')
    ax.set_ylabel(r'$C = \text{residual} / (|z| \cdot e^{-\pi\lambda})$')
    ax.set_title('Panel D: Scaling constant C\n(const → residual ∝ |z|·exp(-πλ))')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.suptitle('Residual suppression mechanism\n(Re(Ω)≠0, z≠0 case)', fontsize=12)
    plt.tight_layout()
    plt.savefig("cancellation_residual.png", dpi=130, bbox_inches='tight')
    print("\nプロット: cancellation_residual.png")
    print("完了")


if __name__ == "__main__":
    main()
