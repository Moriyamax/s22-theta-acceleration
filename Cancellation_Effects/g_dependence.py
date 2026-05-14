"""
g依存性の機構解析
目的: キャンセル強度がgとともに単調増加する理由を特定

n/−n対称性はg非依存 → 別の機構がgとともに強化される

候補:
  G1: 有効格子点数N_effのgスケーリング
      高次元になるほど格子点が指数的に稀になり、
      残存するn/−nペアが少なくなる → キャンセルがより完全になる

  G2: ランダム位相の次元依存的平均化
      nᵀz の分布がgが大きいほど広がり、sin(2πnᵀz)の
      平均がゼロに近づく（CLT的）

  G3: 主要格子点の固有値依存性
      gが大きいほどIm(Ω)の最小固有値に対して
      格子点の寄与が集中し、相殺がより完全になる

実行: python g_dependence.py
"""

import numpy as np
from itertools import product
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# ============================================================
# 基本関数
# ============================================================

def make_omega_pure_imag(g, lmin, seed=42):
    """Re(Ω)=0の純虚数行列（g×g、機構をクリーンに見るため）"""
    rng = np.random.default_rng(seed)
    A = rng.standard_normal((g, g))
    Q, _ = np.linalg.qr(A)
    eigs = lmin + rng.exponential(0.3, g)
    return 1j * (Q @ np.diag(eigs) @ Q.T)


def lattice_terms(Om_block, z, N_cut=1):
    """格子点ごとの寄与を返す"""
    g = Om_block.shape[0]
    terms = {}
    for n in product(range(-N_cut, N_cut+1), repeat=g):
        nv = np.array(n, dtype=float)
        phase = np.pi * 1j * (nv @ Om_block @ nv + 2 * nv @ z)
        terms[n] = np.exp(phase)
    return terms


def grad_terms(Om_block, z, N_cut=1):
    """勾配の格子点ごとの寄与"""
    g = Om_block.shape[0]
    grad = {j: {} for j in range(g)}
    for n in product(range(-N_cut, N_cut+1), repeat=g):
        nv = np.array(n, dtype=float)
        phase = np.pi * 1j * (nv @ Om_block @ nv + 2 * nv @ z)
        base = np.exp(phase)
        for j in range(g):
            grad[j][n] = 2 * np.pi * 1j * nv[j] * base
    return grad


# ============================================================
# 解析G1: N_effのgスケーリング
# ============================================================

def analysis_g1_neff(lmin=3.0, N_cut=2, n_seeds=5, threshold=1e-6):
    """
    有効格子点数N_effとgの関係
    N_eff = #{n≠0: |exp(-π nᵀIm(Ω)n)| > threshold}
    gが大きいほどN_effが指数的に減少し、
    残存する相殺ペアが少なくなる仮説を検証
    """
    gs = [2, 3, 4, 5, 6]  # ブロックサイズ

    print("\n[G1] N_effのgスケーリング")
    print(f"  {'g_block':>8} {'N_total':>10} {'N_eff(n≠0)':>12} "
          f"{'N_eff/N_total':>14} {'pairs_remain':>14}")
    print("  " + "-" * 65)

    results = []
    for gb in gs:
        n_effs, n_pairs = [], []
        for seed in range(n_seeds):
            Om = make_omega_pure_imag(gb, lmin, seed)
            terms = lattice_terms(Om, np.zeros(gb), N_cut)
            n0 = tuple([0]*gb)
            theta_val = sum(terms.values())

            # n≠0の有効格子点
            neff = sum(1 for n, v in terms.items()
                      if n != n0 and abs(v) > threshold * abs(terms[n0]))

            # 実際にn/−nペアとして機能する数
            # （両方がthreshold以上のペア数）
            pairs = 0
            counted = set()
            for n, v in terms.items():
                if n == n0 or n in counted:
                    continue
                neg_n = tuple(-x for x in n)
                if neg_n in terms and abs(v) > threshold * abs(terms[n0]):
                    pairs += 1
                    counted.add(n)
                    counted.add(neg_n)

            n_effs.append(neff)
            n_pairs.append(pairs)

        n_total = (2*N_cut+1)**gb - 1  # n≠0の総数
        mean_neff = np.mean(n_effs)
        mean_pairs = np.mean(n_pairs)
        print(f"  {gb:8d} {n_total:10d} {mean_neff:12.1f} "
              f"{mean_neff/n_total:14.4f} {mean_pairs:14.1f}")
        results.append({'gb': gb, 'N_total': n_total,
                        'N_eff': mean_neff, 'pairs': mean_pairs})
    return results


# ============================================================
# 解析G2: ランダム位相の次元依存的平均化
# ============================================================

def analysis_g2_phase_averaging(lmin=3.0, N_cut=2, n_seeds=20):
    """
    nᵀz の分布のg依存性
    z ~ N(0, 0.3²·I_g) のとき、
    各格子点nに対してnᵀz の分散は 0.09 * Σn_i²
    |n|が固定なら分散はgに依存しない

    → g依存性はnᵀzの「有効な広がり」ではなく
      Σ_{n} sin(2πnᵀz) の次元依存的な消去に起因する可能性

    実測: E[|sin(2πnᵀz)|] のg依存性を確認
    """
    gs = [2, 3, 4, 5, 6]
    z_scale = 0.3

    print("\n[G2] ランダム位相の次元依存的平均化")
    print(f"  {'g_block':>8} {'E[|sin(2πnᵀz)|]':>18} "
          f"{'std':>8} {'E[sin²(2πnᵀz)]^0.5':>20}")
    print("  " + "-" * 60)

    results = []
    for gb in gs:
        sin_means, sin_rms = [], []
        rng = np.random.default_rng(42)
        Om = make_omega_pure_imag(gb, lmin, 0)

        for seed in range(n_seeds):
            z = rng.standard_normal(gb) * z_scale
            # |n|=1の格子点のみ（支配的な寄与）
            sin_vals = []
            for n in product([-1, 0, 1], repeat=gb):
                nv = np.array(n, dtype=float)
                if np.linalg.norm(nv) < 0.5:
                    continue
                if np.linalg.norm(nv) > 1.5:
                    continue  # |n|=1のみ
                sin_vals.append(abs(np.sin(2 * np.pi * nv @ z)))
            sin_means.append(np.mean(sin_vals))
            sin_rms.append(np.sqrt(np.mean(np.array(sin_vals)**2)))

        mean_sin = np.mean(sin_means)
        std_sin = np.std(sin_means)
        mean_rms = np.mean(sin_rms)
        print(f"  {gb:8d} {mean_sin:18.4f} {std_sin:8.4f} {mean_rms:20.4f}")
        results.append({'gb': gb, 'mean_sin': mean_sin, 'rms_sin': mean_rms})
    return results


# ============================================================
# 解析G3: μ_A/Θ_A のgスケーリングの精密測定
# ============================================================

def analysis_g3_mu_scaling(lmin=3.0, N_cut=2, n_seeds=10):
    """
    |μ_A|/|Θ_A| のgスケーリングを精密測定し、
    N_effとの相関を確認する

    理論予測（G1仮説）:
      |μ_A|/|Θ_A| ≈ N_pairs × <|sin(2πnᵀz)|> × exp(-π nᵀIm(Ω)n) / |Θ_A|
                  ∝ N_pairs(g) × exp(-π λ_min)

    N_pairs(g)がgとともにどう変化するかで機構が決まる
    """
    gs = [2, 3, 4, 5, 6]
    z_scale = 0.3

    print("\n[G3] μ_A/Θ_A のgスケーリング精密測定")
    print(f"  {'g_block':>8} {'|μ_A|/|Θ_A|':>14} {'std':>8} "
          f"{'N_eff':>8} {'ratio/N_eff':>12} {'exp(-πλ)':>12}")
    print("  " + "-" * 70)

    results = []
    for gb in gs:
        ratios, n_effs = [], []
        rng = np.random.default_rng(42)

        for seed in range(n_seeds):
            Om = make_omega_pure_imag(gb, lmin, seed)
            z = rng.standard_normal(gb) * z_scale
            lmin_act = np.linalg.eigvalsh(Om.imag).min()

            # θ値と勾配
            terms = lattice_terms(Om, z, N_cut)
            theta_val = sum(terms.values())
            gterms = grad_terms(Om, z, N_cut)

            grad = np.array([sum(gterms[j].values()) for j in range(gb)])
            mu_A = grad / (2 * np.pi * 1j)
            ratio = np.linalg.norm(mu_A) / (abs(theta_val) + 1e-30)

            # N_eff
            n0 = tuple([0]*gb)
            neff = sum(1 for n, v in terms.items()
                      if n != n0 and abs(v) > 1e-6 * abs(terms[n0]))

            ratios.append(ratio)
            n_effs.append(neff)

        mean_r = np.mean(ratios)
        std_r = np.std(ratios)
        mean_neff = np.mean(n_effs)
        exp_lmin = np.exp(-np.pi * lmin)
        # ratio / (N_eff × exp(-πλ)) が定数なら G1仮説支持
        normalized = mean_r / (mean_neff * exp_lmin + 1e-30)

        print(f"  {gb:8d} {mean_r:14.3e} {std_r:8.2e} "
              f"{mean_neff:8.1f} {normalized:12.3e} {exp_lmin:12.3e}")
        results.append({'gb': gb, 'ratio': mean_r, 'N_eff': mean_neff,
                        'normalized': normalized, 'exp_lmin': exp_lmin})
    return results


# ============================================================
# 解析G4: N_effのg依存性の指数スケーリング確認
# ============================================================

def analysis_g4_neff_exponential(lmin_vals=[2.0, 3.0, 5.0], N_cut=2, n_seeds=5):
    """
    N_eff ∝ exp(-α(lmin) × g) のスケーリングを確認
    これがg依存性の主因かを検証
    """
    gs = [2, 3, 4, 5, 6]

    print("\n[G4] N_effの指数スケーリング確認")
    print("  仮説: N_eff ∝ exp(-α × g_block)")
    print()

    results = {}
    for lmin in lmin_vals:
        neffs = []
        for gb in gs:
            neff_seeds = []
            for seed in range(n_seeds):
                Om = make_omega_pure_imag(gb, lmin, seed)
                terms = lattice_terms(Om, np.zeros(gb), N_cut)
                n0 = tuple([0]*gb)
                neff = sum(1 for n, v in terms.items()
                          if n != n0 and abs(v) > 1e-6 * abs(terms[n0]))
                neff_seeds.append(neff)
            neffs.append(np.mean(neff_seeds))

        # 指数フィット
        log_neff = np.log(np.array(neffs) + 1e-10)
        coeffs = np.polyfit(gs, log_neff, 1)
        alpha = -coeffs[0]

        print(f"  lmin={lmin:.1f}: α={alpha:.3f}, "
              f"N_eff = {np.exp(coeffs[1]):.2f} × exp(-{alpha:.3f} × g)")
        print(f"  g:     " + "  ".join(f"{g:6d}" for g in gs))
        print(f"  N_eff: " + "  ".join(f"{n:6.1f}" for n in neffs))
        print()
        results[lmin] = {'gs': gs, 'neffs': neffs, 'alpha': alpha}

    return results


# ============================================================
# メイン
# ============================================================

def main():
    print("=" * 60)
    print("g依存性の機構解析")
    print("キャンセル強度がgとともに単調増加する理由")
    print("=" * 60)

    lmin = 3.0

    res_g1 = analysis_g1_neff(lmin=lmin, N_cut=2, n_seeds=5)
    res_g2 = analysis_g2_phase_averaging(lmin=lmin, N_cut=2, n_seeds=20)
    res_g3 = analysis_g3_mu_scaling(lmin=lmin, N_cut=2, n_seeds=10)
    res_g4 = analysis_g4_neff_exponential(lmin_vals=[2.0, 3.0, 5.0], N_cut=2, n_seeds=5)

    # ---- プロット ----
    fig = plt.figure(figsize=(14, 10))
    gs_layout = gridspec.GridSpec(2, 2, fig, hspace=0.4, wspace=0.35)

    # Panel A: N_eff vs g_block
    ax = fig.add_subplot(gs_layout[0, 0])
    gbs = [r['gb'] for r in res_g1]
    neffs = [r['N_eff'] for r in res_g1]
    ntots = [r['N_total'] for r in res_g1]
    ax.plot(gbs, neffs, 'o-', color='steelblue', label='N_eff (measured)')
    ax.plot(gbs, ntots, 's--', color='gray', alpha=0.5, label='N_total (n≠0)')
    ax.set_yscale('log')
    ax.set_xlabel('g_block')
    ax.set_ylabel('count')
    ax.set_title('Panel A: N_eff vs g_block\n(N_cut=2, lmin=3.0)')
    ax.legend(); ax.grid(True, alpha=0.3)

    # Panel B: E[|sin|] vs g_block（ほぼ平坦なはず）
    ax2 = fig.add_subplot(gs_layout[0, 1])
    gbs2 = [r['gb'] for r in res_g2]
    sins = [r['mean_sin'] for r in res_g2]
    rmss = [r['rms_sin'] for r in res_g2]
    ax2.plot(gbs2, sins, 'o-', color='tomato', label='E[|sin(2πnᵀz)|]')
    ax2.plot(gbs2, rmss, 's-', color='orange', label='RMS[sin(2πnᵀz)]')
    ax2.axhline(2/np.pi, color='gray', linestyle='--', label='2/π (uniform avg)')
    ax2.set_xlabel('g_block')
    ax2.set_ylabel('value')
    ax2.set_title('Panel B: Phase averaging vs g\n(|n|=1 lattice points)')
    ax2.legend(); ax2.grid(True, alpha=0.3)

    # Panel C: |μ_A|/|Θ_A| vs g_block と N_eff×exp(-πλ)
    ax3 = fig.add_subplot(gs_layout[1, 0])
    gbs3 = [r['gb'] for r in res_g3]
    ratios = [r['ratio'] for r in res_g3]
    neffs3 = [r['N_eff'] for r in res_g3]
    exp_lmin = res_g3[0]['exp_lmin']
    predicted = [n * exp_lmin for n in neffs3]
    ax3.plot(gbs3, ratios, 'o-', color='steelblue', label='|μ_A|/|Θ_A| measured')
    ax3.plot(gbs3, predicted, 's--', color='orange', label='N_eff × exp(-πλ)')
    ax3.set_yscale('log')
    ax3.set_xlabel('g_block')
    ax3.set_ylabel('value')
    ax3.set_title('Panel C: μ_A/Θ_A vs N_eff×exp(-πλ)\n(G1仮説の検証)')
    ax3.legend(); ax3.grid(True, alpha=0.3)

    # Panel D: N_effの指数スケーリング
    ax4 = fig.add_subplot(gs_layout[1, 1])
    colors = ['steelblue', 'tomato', 'green']
    for (lmin_val, res), color in zip(res_g4.items(), colors):
        gs_v = res['gs']
        neffs_v = res['neffs']
        alpha = res['alpha']
        ax4.plot(gs_v, neffs_v, 'o-', color=color,
                label=f'λ={lmin_val} (α={alpha:.2f})')
        # フィット線
        fit = np.exp(-alpha * np.array(gs_v) + np.log(neffs_v[0] + 1e-10) + alpha * gs_v[0])
        ax4.plot(gs_v, fit, '--', color=color, alpha=0.5)
    ax4.set_yscale('log')
    ax4.set_xlabel('g_block')
    ax4.set_ylabel('N_eff')
    ax4.set_title('Panel D: N_eff exponential scaling\nN_eff ∝ exp(-α×g)')
    ax4.legend(); ax4.grid(True, alpha=0.3)

    plt.suptitle('g-Dependence Mechanism — N_eff Scaling', fontsize=13, y=1.01)
    plt.savefig("g_dependence.png", dpi=130, bbox_inches='tight')
    print("\nプロット: g_dependence.png")
    print("完了")


if __name__ == "__main__":
    main()
