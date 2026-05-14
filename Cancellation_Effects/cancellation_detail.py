"""
追加抑制の正体特定
目的: |mu_A|/|Theta_A| が exp(-pi*lmin) より速く減衰する理由を特定する

仮説:
  H1: mu_A の各成分が個別にキャンセル（符号が逆方向）
  H2: 格子和の稀薄性 (有効格子点数N_eff ∝ Vol(ellipsoid)) が効いている
  H3: Theta関数自体が exp(-pi*lmin) * (補正因子) のスケールを持つ

実行: python cancellation_detail.py
"""

import numpy as np
from itertools import product
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# ---- 基本関数（前回と同じ）----

def make_omega(g, lmin, offnorm=0.3, seed=42):
    rng = np.random.default_rng(seed)
    gb = g // 2
    def make_block(n):
        A = rng.standard_normal((n, n))
        Q, _ = np.linalg.qr(A)
        eigs = lmin + rng.exponential(0.5, n)
        return Q @ np.diag(eigs) @ Q.T
    B1 = make_block(gb)
    B2 = make_block(gb)
    C = rng.standard_normal((gb, gb)) * offnorm * 0.3
    C = (C + C.T) / 2
    ImOm = np.block([[B1, C], [C.T, B2]])
    ReOm = rng.standard_normal((g, g)) * 0.1
    ReOm = (ReOm + ReOm.T) / 2
    eigs = np.linalg.eigvalsh(ImOm)
    if eigs.min() < 0.01:
        ImOm += (0.01 - eigs.min()) * np.eye(g)
    return ReOm + 1j * ImOm


def theta_block_verbose(Om, z, N_cut=1):
    """格子点ごとの寄与を返す（キャンセル確認用）"""
    g = Om.shape[0]
    ns = range(-N_cut, N_cut + 1)
    terms = {}
    for n in product(ns, repeat=g):
        n_arr = np.array(n, dtype=float)
        phase = np.pi * 1j * (n_arr @ Om @ n_arr + 2 * n_arr @ z)
        terms[n] = np.exp(phase)
    total = sum(terms.values())
    return total, terms


def theta_block(Om, z, N_cut=1):
    g = Om.shape[0]
    ns = range(-N_cut, N_cut + 1)
    result = 0.0 + 0j
    for n in product(ns, repeat=g):
        n_arr = np.array(n, dtype=float)
        phase = np.pi * 1j * (n_arr @ Om @ n_arr + 2 * n_arr @ z)
        result += np.exp(phase)
    return result


def grad_theta_verbose(Om, z, N_cut=1):
    """勾配の各成分を格子点ごとに計算"""
    g = Om.shape[0]
    ns = range(-N_cut, N_cut + 1)
    grad_terms = {j: {} for j in range(g)}
    for n in product(ns, repeat=g):
        n_arr = np.array(n, dtype=float)
        phase = np.pi * 1j * (n_arr @ Om @ n_arr + 2 * n_arr @ z)
        base = np.exp(phase)
        for j in range(g):
            # d/dz_j: coefficient = 2*pi*i * n_j
            grad_terms[j][n] = 2 * np.pi * 1j * n_arr[j] * base
    grad = np.array([sum(grad_terms[j].values()) for j in range(g)])
    return grad, grad_terms


# ============================================================
# 解析A: H1の検証 - mu_A成分のキャンセル
# ============================================================

def analyze_cancellation_in_mu(lmin_vals, g=4, N_cut=2, n_seeds=3):
    """
    mu_A = grad(Theta_A) / (2*pi*i) の各成分が
    格子点ごとにキャンセルしているかを確認

    キャンセル率 = sum|terms| / |sum(terms)|  (大きいほどキャンセルが強い)
    """
    print("\n[H1検証] mu_Aの各成分でのキャンセル率")
    print(f"{'lmin':>6} {'cancel_grad':>14} {'cancel_theta':>14} {'ratio':>10}")
    print("-" * 50)

    results = []
    for lmin in lmin_vals:
        cancel_grads = []
        cancel_thetas = []
        for seed in range(n_seeds):
            Omega = make_omega(g, lmin, offnorm=0.3, seed=seed)
            gb = g // 2
            Om1 = Omega[:gb, :gb]
            rng = np.random.default_rng(seed + 100)
            z1 = rng.standard_normal(gb) * 0.3

            theta_val, theta_terms = theta_block_verbose(Om1, z1, N_cut)
            grad_val, grad_terms = grad_theta_verbose(Om1, z1, N_cut)

            # theta のキャンセル率
            sum_abs_theta = sum(abs(v) for v in theta_terms.values())
            cancel_theta = sum_abs_theta / (abs(theta_val) + 1e-30)
            cancel_thetas.append(cancel_theta)

            # 勾配の各成分のキャンセル率（平均）
            cancel_grad_j = []
            for j in range(gb):
                sum_abs_g = sum(abs(v) for v in grad_terms[j].values())
                cancel_g = sum_abs_g / (abs(grad_val[j]) + 1e-30)
                cancel_grad_j.append(cancel_g)
            cancel_grads.append(np.mean(cancel_grad_j))

        mean_cg = np.mean(cancel_grads)
        mean_ct = np.mean(cancel_thetas)
        line = f"{lmin:6.1f} {mean_cg:14.3e} {mean_ct:14.3e} {mean_cg/mean_ct:10.3f}"
        print(line)
        results.append({'lmin': lmin, 'cancel_grad': mean_cg,
                        'cancel_theta': mean_ct, 'ratio': mean_cg/mean_ct})
    return results


# ============================================================
# 解析B: H2の検証 - 有効格子点数 N_eff のスケーリング
# ============================================================

def analyze_neff_scaling(lmin_vals, g=8, N_cut=2, n_seeds=3, threshold=1e-6):
    """
    有効格子点数 N_eff = #{n : |term(n)| > threshold * |theta|}
    が lmin にどう依存するかを確認
    """
    print("\n[H2検証] 有効格子点数 N_eff の λ_min 依存性")
    print(f"{'lmin':>6} {'N_total':>10} {'N_eff':>10} {'N_eff/N_total':>14} {'vol_pred':>12}")
    print("-" * 58)

    results = []
    for lmin in lmin_vals:
        n_effs = []
        n_totals = []
        for seed in range(n_seeds):
            Omega = make_omega(g, lmin, offnorm=0.1, seed=seed)
            gb = g // 2
            Om1 = Omega[:gb, :gb]
            rng = np.random.default_rng(seed + 200)
            z1 = rng.standard_normal(gb) * 0.3

            theta_val, theta_terms = theta_block_verbose(Om1, z1, N_cut)
            abs_theta = abs(theta_val)
            n_total = len(theta_terms)
            n_eff = sum(1 for v in theta_terms.values() if abs(v) > threshold * abs_theta)
            n_effs.append(n_eff)
            n_totals.append(n_total)

        mean_neff = np.mean(n_effs)
        mean_ntot = np.mean(n_totals)
        # 楕円体体積の理論予測: (pi/lmin)^(gb/2) / Gamma(gb/2+1) * (ln(1/eps)/pi)^(gb/2)
        gb = g // 2
        from math import gamma
        eps = threshold
        vol_pred = (np.pi / lmin)**(gb/2) / gamma(gb/2 + 1) * (np.log(1/eps)/np.pi)**(gb/2)
        line = f"{lmin:6.1f} {mean_ntot:10.0f} {mean_neff:10.2f} {mean_neff/mean_ntot:14.4f} {vol_pred:12.2f}"
        print(line)
        results.append({'lmin': lmin, 'N_eff': mean_neff, 'N_total': mean_ntot,
                        'vol_pred': vol_pred})
    return results


# ============================================================
# 解析C: H3の検証 - Theta自体のスケーリングの精密測定
# ============================================================

def analyze_theta_scaling(lmin_vals, g=8, N_cut=2, n_seeds=5):
    """
    |Theta_A| 自体が exp(-pi*lmin) でスケールするかを確認
    理論: dominant term は n=0 の寄与 = 1（z=0の場合）
          次の項は exp(-pi*lmin) オーダー
    Theta = 1 + 2*sum_{n!=0} exp(-pi*n^T Im(Om) n) * ...
    """
    print("\n[H3検証] |Theta_A| と dominant term のスケーリング")
    print(f"{'lmin':>6} {'|Theta|':>12} {'|Theta-1|':>12} "
          f"{'exp(-pi*l)':>12} {'ratio':>10}")
    print("-" * 58)

    results = []
    for lmin in lmin_vals:
        thetas = []
        theta_m1s = []  # |Theta - n=0 term|
        for seed in range(n_seeds):
            Omega = make_omega(g, lmin, offnorm=0.1, seed=seed)
            gb = g // 2
            Om1 = Omega[:gb, :gb]
            # z=0 で評価してn=0項を1に規格化
            z1 = np.zeros(gb)
            theta_val, theta_terms = theta_block_verbose(Om1, z1, N_cut)
            n0_term = theta_terms[tuple([0]*gb)]
            non_n0 = abs(theta_val - n0_term) / abs(n0_term)
            thetas.append(abs(theta_val))
            theta_m1s.append(non_n0)

        mean_t = np.mean(thetas)
        mean_tm1 = np.mean(theta_m1s)
        exp_pred = np.exp(-np.pi * lmin)
        line = (f"{lmin:6.1f} {mean_t:12.4f} {mean_tm1:12.3e} "
                f"{exp_pred:12.3e} {mean_tm1/exp_pred:10.3f}")
        print(line)
        results.append({'lmin': lmin, '|Theta|': mean_t,
                        '|Theta-n0|/|n0|': mean_tm1, 'exp(-pi*l)': exp_pred})
    return results


# ============================================================
# 解析D: mu_A の各成分 vs n=0 付近の格子点への寄与
# ============================================================

def analyze_mu_dominant_terms(lmin=3.0, g=4, N_cut=2, seed=0):
    """
    mu_A = grad(Theta) / (2pi*i) の中で、
    どの格子点が主要寄与しているかを可視化
    """
    print(f"\n[mu_A詳細] lmin={lmin}, g={g}, N_cut={N_cut}")

    Omega = make_omega(g, lmin, offnorm=0.1, seed=seed)
    gb = g // 2
    Om1 = Omega[:gb, :gb]
    z1 = np.zeros(gb)  # z=0で解析的にクリーン

    theta_val, theta_terms = theta_block_verbose(Om1, z1, N_cut)
    grad_val, grad_terms = grad_theta_verbose(Om1, z1, N_cut)

    print(f"  |Theta| = {abs(theta_val):.6e}")
    print(f"  n=0 term = {theta_terms[tuple([0]*gb)]:.6e}")
    print()

    # j=0 成分の主要格子点
    j = 0
    print(f"  grad[j={j}] の上位格子点（絶対値順）:")
    sorted_terms = sorted(grad_terms[j].items(),
                          key=lambda x: abs(x[1]), reverse=True)
    running_sum = 0j
    for n, val in sorted_terms[:10]:
        running_sum += val
        print(f"    n={n}: |contrib|={abs(val):.3e}, "
              f"running_sum={abs(running_sum):.3e}, "
              f"n_j={n[j]}")

    # n=0 は n_j=0 なので勾配への寄与はゼロ
    print(f"\n  Note: n=0 term contributes 0 to gradient (n_j=0)")
    print(f"  → grad は n≠0 の格子点のみから来る")
    print(f"  → これらは全て exp(-pi*lmin) 以下のスケール")

    mu_A = grad_val / (2 * np.pi * 1j)
    print(f"\n  |mu_A| = {np.linalg.norm(mu_A):.6e}")
    print(f"  |Theta| = {abs(theta_val):.6e}")
    print(f"  |mu_A|/|Theta| = {np.linalg.norm(mu_A)/abs(theta_val):.6e}")
    print(f"  exp(-pi*lmin) = {np.exp(-np.pi*lmin):.6e}")
    print(f"  |mu_A|/|Theta| / exp(-pi*lmin) = "
          f"{np.linalg.norm(mu_A)/abs(theta_val)/np.exp(-np.pi*lmin):.6e}")
    print()
    print("  【理論的説明】")
    print("  Theta = n0_term + (n≠0 terms)")
    print("  grad  = 0       + (n≠0 terms) × 2pi*i*n_j")
    print("  → |grad| ≤ N_eff × max_n|n_j| × exp(-pi*lmin)")
    print("  → |mu_A| = |grad|/(2pi) ≤ N_eff × max_n|n_j| × exp(-pi*lmin) / (2pi)")
    print("  → |mu_A|/|Theta| ≤ N_eff × max_n|n_j| × exp(-pi*lmin) / (2pi * |Theta|)")


# ============================================================
# メイン
# ============================================================

def main():
    print("=" * 60)
    print("Anomalous Cancellation — 追加抑制の正体特定")
    print("=" * 60)

    lmin_vals = [0.5, 1.0, 2.0, 3.0, 5.0]

    # H1: キャンセル
    res_h1 = analyze_cancellation_in_mu(lmin_vals, g=4, N_cut=2)

    # H2: 有効格子点数
    res_h2 = analyze_neff_scaling(lmin_vals, g=8, N_cut=2)

    # H3: Theta自体のスケーリング
    res_h3 = analyze_theta_scaling(lmin_vals, g=8, N_cut=2)

    # 詳細分析
    analyze_mu_dominant_terms(lmin=3.0, g=4, N_cut=2, seed=0)

    # ---- プロット ----
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # H1
    ax = axes[0]
    lms = [r['lmin'] for r in res_h1]
    ax.plot(lms, [r['cancel_grad'] for r in res_h1], 'o-', label='cancel(grad)')
    ax.plot(lms, [r['cancel_theta'] for r in res_h1], 's-', label='cancel(theta)')
    ax.set_yscale('log')
    ax.set_xlabel(r'$\lambda_\min$')
    ax.set_title('H1: Cancellation ratio\n(larger = more cancellation)')
    ax.legend(); ax.grid(True, alpha=0.3)

    # H2
    ax = axes[1]
    lms2 = [r['lmin'] for r in res_h2]
    ax.plot(lms2, [r['N_eff'] for r in res_h2], 'o-', label='N_eff measured')
    ax.plot(lms2, [r['vol_pred'] for r in res_h2], '--', label='ellipsoid vol pred')
    ax.set_yscale('log')
    ax.set_xlabel(r'$\lambda_\min$')
    ax.set_title('H2: Effective lattice points N_eff')
    ax.legend(); ax.grid(True, alpha=0.3)

    # H3
    ax = axes[2]
    lms3 = [r['lmin'] for r in res_h3]
    ax.plot(lms3, [r['|Theta-n0|/|n0|'] for r in res_h3], 'o-', label='|Theta-1|/1 (z=0)')
    ax.plot(lms3, [r['exp(-pi*l)'] for r in res_h3], '--', label=r'$e^{-\pi\lambda}$')
    ax.set_yscale('log')
    ax.set_xlabel(r'$\lambda_\min$')
    ax.set_title('H3: Theta non-zero correction\n(z=0, normalized)')
    ax.legend(); ax.grid(True, alpha=0.3)

    plt.suptitle('Additional suppression mechanism — H1/H2/H3', y=1.02)
    plt.tight_layout()
    plt.savefig("cancellation_detail.png", dpi=130, bbox_inches='tight')
    print("\nプロット: cancellation_detail.png")
    print("完了")


if __name__ == "__main__":
    main()
