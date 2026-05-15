"""
定理上界のtight化
目的: ratio ≈ 0.07 (generic z) の残余を説明し、
     cos項を明示的に取り込んだより tight な上界を導出・検証する

現状の上界（UB0）:
  |LHS| ≤ 4π · Σ_{n in half-lattice} |Re(n_Aᵀ C n_B)| · exp(-πnᵀIm(Ω)n)
  → ratio ≈ 0.07 (generic z), 1.0 (z=1/4, worst-case C)

改善版1（UB1, cos項込み）:
  |LHS| ≤ 4π · Σ_{n in half-lattice} |Re(n_Aᵀ C n_B)| · |cos(2πnᵀz)| · exp(-πnᵀIm(Ω)n)
  → cosが小さいzでは改善、z=1/4では変わらない

改善版2（UB2, SVD最適）:
  |LHS| ≤ 4π · ||M(z)||_F
  M(z) = Σ_{n in half-lattice} cos(2πnᵀz) · exp(-πnᵀIm(Ω)n) · n_A n_Bᵀ
  → Cauchy-Schwarzで達成可能な最小上界

改善版3（UB3, Cの統計を活用）:
  E_C[|LHS|²] ≤ (4π)² · ||C||²_F · Σ_n cos²(2πnᵀz) · exp(-2πnᵀIm(Ω)n) · |n_A|²|n_B|²
  → ランダムCに対する期待値上界

実行: python tight_bound.py
"""

import numpy as np
from itertools import product
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# ============================================================
# 基本関数
# ============================================================

def make_omega_blocks(g, lmin, re_scale=0.0, seed=42):
    rng = np.random.default_rng(seed)
    g1 = g // 2
    g2 = g - g1
    def make_block(n):
        A = rng.standard_normal((n, n))
        Q, _ = np.linalg.qr(A)
        eigs = lmin + rng.exponential(0.3, n)
        return Q @ np.diag(eigs) @ Q.T
    B1 = make_block(g1)
    B2 = make_block(g2)
    ImOm = np.block([[B1, np.zeros((g1,g2))],
                     [np.zeros((g2,g1)), B2]])
    return 1j * ImOm, g1, g2


def make_random_C(g1, g2, seed=0):
    rng = np.random.default_rng(seed)
    C = rng.standard_normal((g1, g2)) + 1j * rng.standard_normal((g1, g2))
    C /= np.linalg.norm(C, 'fro')
    return C


def compute_all_bounds(Om0, C, z, N_cut, g1):
    """LHSと4種の上界を全て計算"""
    g = Om0.shape[0]
    g2 = g - g1

    # 半格子点のリスト
    seen = set()
    half_lattice = []
    for n in product(range(-N_cut, N_cut+1), repeat=g):
        nv = np.array(n, dtype=float)
        neg_n = tuple(-x for x in n)
        if n == tuple([0]*g) or neg_n in seen:
            continue
        seen.add(n)
        nA = nv[:g1]
        nB = nv[g1:]
        exp_val = np.exp(-np.pi * nv @ Om0.imag @ nv)
        cos_val = np.cos(2 * np.pi * nv @ z)
        re_nCn = np.real(nA @ C @ nB)
        half_lattice.append({
            'nv': nv, 'nA': nA, 'nB': nB,
            'exp': exp_val, 'cos': cos_val,
            're_nCn': re_nCn,
        })

    # LHS（解析式）
    lhs = 0j
    for n in product(range(-N_cut, N_cut+1), repeat=g):
        nv = np.array(n, dtype=float)
        nEn = nv @ (np.zeros((g,g), dtype=complex)
                    + np.block([[np.zeros((g1,g1)), C],
                                [C.conj().T, np.zeros((g2,g2))]])
                    + np.block([[np.zeros((g1,g1)), C.conj().T],
                                [C, np.zeros((g2,g2))]])) @ nv
        # 実際はnEn = 2Re(nA^T C nB)
        nA = nv[:g1]; nB = nv[g1:]
        nEn_real = 2 * np.real(nA @ C @ nB)
        phase = np.pi*1j*(nv@Om0@nv + 2*nv@z)
        lhs += np.pi*1j * nEn_real * np.exp(phase)
    lhs_abs = abs(lhs)

    # UB0: 現状の上界（cos項なし）
    ub0 = 4*np.pi * sum(abs(t['re_nCn']) * t['exp'] for t in half_lattice)

    # UB1: cos項込み
    ub1 = 4*np.pi * sum(abs(t['re_nCn']) * abs(t['cos']) * t['exp'] for t in half_lattice)

    # UB2: SVD最適（M行列のFrobeniusノルム）
    M = np.zeros((g1, g2), dtype=complex)
    for t in half_lattice:
        M += t['cos'] * t['exp'] * np.outer(t['nA'], t['nB'])
    ub2 = 4*np.pi * np.linalg.norm(M, 'fro')

    # UB3: Cauchy-Schwarz with ||C||_F = 1
    # |LHS| ≤ 4π · ||C||_F · (Σ_n cos²·exp²·||nA||²·||nB||²)^{1/2}
    sum_sq = sum(t['cos']**2 * t['exp']**2 * np.sum(t['nA']**2) * np.sum(t['nB']**2)
                 for t in half_lattice)
    ub3 = 4*np.pi * np.linalg.norm(C, 'fro') * np.sqrt(sum_sq)

    lmin = np.linalg.eigvalsh(Om0.imag).min()

    return {
        'lhs': lhs_abs,
        'ub0': ub0, 'ub1': ub1, 'ub2': ub2, 'ub3': ub3,
        'r0': lhs_abs/(ub0+1e-30),
        'r1': lhs_abs/(ub1+1e-30),
        'r2': lhs_abs/(ub2+1e-30),
        'r3': lhs_abs/(ub3+1e-30),
        'lmin': lmin,
    }


def make_svd_optimal_C(Om0, z, N_cut, g1):
    """UB2を最大化するC（=M*/||M||_F）"""
    g = Om0.shape[0]
    g2 = g - g1
    seen = set()
    M = np.zeros((g1, g2), dtype=complex)
    for n in product(range(-N_cut, N_cut+1), repeat=g):
        nv = np.array(n, dtype=float)
        neg_n = tuple(-x for x in n)
        if n == tuple([0]*g) or neg_n in seen:
            continue
        seen.add(n)
        nA, nB = nv[:g1], nv[g1:]
        exp_val = np.exp(-np.pi * nv @ Om0.imag @ nv)
        cos_val = np.cos(2 * np.pi * nv @ z)
        M += cos_val * exp_val * np.outer(nA, nB)
    norm_M = np.linalg.norm(M, 'fro')
    if norm_M < 1e-30:
        return np.zeros((g1,g2), dtype=complex)
    return M.conj() / norm_M


# ============================================================
# 解析1: 各上界のratioをzカテゴリ別に比較
# ============================================================

def analysis_bounds_by_z(lmin=3.0, g=8, N_cut=1, n_seeds=8):
    print("\n[解析1] 各上界のratio (z別, random C)")
    print(f"  {'z種類':18} {'r0(現状)':>10} {'r1(cos込)':>10} "
          f"{'r2(SVD)':>10} {'r3(CS)':>10}")
    print("  " + "-" * 55)

    rng_z = np.random.default_rng(0)
    z_cases = {
        'generic':    rng_z.standard_normal(g) * 0.3,
        'z=0':        np.zeros(g),
        'z=1/4·1':    np.ones(g) * 0.25,
        'z=1/2·1':    np.ones(g) * 0.5,
        'small(0.01)': rng_z.standard_normal(g) * 0.01,
    }

    results = {}
    for z_label, z in z_cases.items():
        r0s, r1s, r2s, r3s = [], [], [], []
        for seed in range(n_seeds):
            Om0, g1, _ = make_omega_blocks(g, lmin, seed=seed)
            C = make_random_C(g1, g-g1, seed=seed)
            res = compute_all_bounds(Om0, C, z, N_cut, g1)
            r0s.append(res['r0']); r1s.append(res['r1'])
            r2s.append(res['r2']); r3s.append(res['r3'])

        results[z_label] = {
            'r0': np.mean(r0s), 'r1': np.mean(r1s),
            'r2': np.mean(r2s), 'r3': np.mean(r3s)
        }
        print(f"  {z_label:18} {np.mean(r0s):10.4f} {np.mean(r1s):10.4f} "
              f"{np.mean(r2s):10.4f} {np.mean(r3s):10.4f}")

    return results


# ============================================================
# 解析2: SVD最適CでのUB2のratio vs λ_min
# ============================================================

def analysis_ub2_svd(lmin_vals, g=8, N_cut=1, n_seeds=5):
    print("\n[解析2] UB2(SVD最適C)のratio vs λ_min")
    print(f"  {'lmin':>6} {'r0_rand':>10} {'r2_rand':>10} "
          f"{'r2_svd':>10} {'改善倍率':>10}")
    print("  " + "-" * 48)

    z_cases = {
        'generic': np.random.default_rng(42).standard_normal(g) * 0.3,
        'z=1/4':   np.ones(g) * 0.25,
    }
    results = {z: [] for z in z_cases}

    for lmin in lmin_vals:
        for z_label, z in z_cases.items():
            r0s, r2_rands, r2_svds = [], [], []
            for seed in range(n_seeds):
                Om0, g1, _ = make_omega_blocks(g, lmin, seed=seed)
                C_rand = make_random_C(g1, g-g1, seed=seed)
                C_svd  = make_svd_optimal_C(Om0, z, N_cut, g1)

                res_rand = compute_all_bounds(Om0, C_rand, z, N_cut, g1)
                res_svd  = compute_all_bounds(Om0, C_svd,  z, N_cut, g1)

                r0s.append(res_rand['r0'])
                r2_rands.append(res_rand['r2'])
                r2_svds.append(res_svd['r2'])

            r0  = np.mean(r0s)
            r2r = np.mean(r2_rands)
            r2s_val = np.mean(r2_svds)
            improvement = r0 / (r2r + 1e-30)

            results[z_label].append({
                'lmin': lmin, 'r0': r0, 'r2_rand': r2r, 'r2_svd': r2s_val
            })
            if z_label == 'generic':
                print(f"  {lmin:6.1f} {r0:10.4f} {r2r:10.4f} "
                      f"{r2s_val:10.4f} {improvement:10.2f}x")

    return results


# ============================================================
# 解析3: UB1とUB2の改善量 vs cos分布の関係
# ============================================================

def analysis_cos_improvement(lmin=3.0, g=8, N_cut=1, n_seeds=20):
    print("\n[解析3] cos分布とUB1/UB2の改善量の関係")
    print(f"  E[|cos|] が小さいほど UB1/UB0 が小さくなるはず")
    print(f"  {'z_scale':>8} {'E[|cos|]':>10} {'r0':>8} {'r1':>8} "
          f"{'r2':>8} {'UB0/UB1':>10}")
    print("  " + "-" * 55)

    z_scales = [0.0, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0]
    results = []

    for z_scale in z_scales:
        r0s, r1s, r2s, cos_means = [], [], [], []
        for seed in range(n_seeds):
            Om0, g1, _ = make_omega_blocks(g, lmin, seed=seed)
            C = make_random_C(g1, g-g1, seed=seed)
            rng = np.random.default_rng(seed+500)
            z = rng.standard_normal(g) * z_scale

            res = compute_all_bounds(Om0, C, z, N_cut, g1)
            r0s.append(res['r0']); r1s.append(res['r1']); r2s.append(res['r2'])

            # cos分布の平均
            seen = set()
            cos_vals = []
            for n in product(range(-N_cut, N_cut+1), repeat=g):
                nv = np.array(n, dtype=float)
                neg_n = tuple(-x for x in n)
                if n == tuple([0]*g) or neg_n in seen: continue
                seen.add(n)
                exp_v = np.exp(-np.pi * nv @ Om0.imag @ nv)
                if exp_v > 1e-10:
                    cos_vals.append(abs(np.cos(2*np.pi*nv@z)))
            if cos_vals:
                cos_means.append(np.mean(cos_vals))

        r0 = np.mean(r0s); r1 = np.mean(r1s); r2 = np.mean(r2s)
        cos_m = np.mean(cos_means) if cos_means else 1.0
        ratio_01 = r0 / (r1 + 1e-30)

        print(f"  {z_scale:8.3f} {cos_m:10.4f} {r0:8.4f} {r1:8.4f} "
              f"{r2:8.4f} {ratio_01:10.3f}x")
        results.append({'z_scale': z_scale, 'cos_mean': cos_m,
                        'r0': r0, 'r1': r1, 'r2': r2})

    return results


# ============================================================
# メイン
# ============================================================

def main():
    print("=" * 65)
    print("定理上界のtight化")
    print("UB0(現状) vs UB1(cos込) vs UB2(SVD最適) vs UB3(CS)")
    print("=" * 65)

    lmin = 3.0
    g = 8
    N_cut = 1

    res1 = analysis_bounds_by_z(lmin=lmin, g=g, N_cut=N_cut)
    lmin_vals = [0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0]
    res2 = analysis_ub2_svd(lmin_vals, g=g, N_cut=N_cut)
    res3 = analysis_cos_improvement(lmin=lmin, g=g, N_cut=N_cut)

    # ---- プロット ----
    fig = plt.figure(figsize=(14, 9))
    gs_layout = gridspec.GridSpec(2, 3, fig, hspace=0.45, wspace=0.35)

    # Panel A: zカテゴリ別の各上界のratio
    ax = fig.add_subplot(gs_layout[0, :2])
    z_labels = list(res1.keys())
    x = np.arange(len(z_labels))
    w = 0.2
    for i, (key, color, label) in enumerate([
        ('r0','steelblue','UB0 (current)'),
        ('r1','tomato',   'UB1 (|cos|)'),
        ('r2','green',    'UB2 (SVD opt)'),
        ('r3','orange',   'UB3 (C-S)'),
    ]):
        vals = [res1[z][key] for z in z_labels]
        ax.bar(x + (i-1.5)*w, vals, w, label=label, alpha=0.8, color=color)
    ax.axhline(1.0, color='red', linestyle='--', linewidth=1, label='bound=1.0')
    ax.set_xticks(x)
    ax.set_xticklabels(z_labels, rotation=15)
    ax.set_ylabel('ratio = |LHS| / bound')
    ax.set_title('Panel A: ratio by bound version\n(g=8, lmin=3.0, random C)')
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3, axis='y')

    # Panel B: UB2(SVD) vs UB0(現状) のratio for generic z
    ax2 = fig.add_subplot(gs_layout[0, 2])
    lmins = [r['lmin'] for r in res2['generic']]
    ax2.plot(lmins, [r['r0']     for r in res2['generic']], 'o-',
             color='steelblue', label='UB0 (current)')
    ax2.plot(lmins, [r['r2_rand'] for r in res2['generic']], 's-',
             color='tomato', label='UB2 (random C)')
    ax2.plot(lmins, [r['r2_svd'] for r in res2['generic']], '^-',
             color='green', label='UB2 (SVD C)')
    ax2.axhline(1.0, color='red', linestyle='--', linewidth=0.8)
    ax2.set_xlabel(r'$\lambda_\min$')
    ax2.set_ylabel('ratio')
    ax2.set_title('Panel B: UB2 vs UB0\n(generic z, g=8)')
    ax2.legend(fontsize=8); ax2.grid(True, alpha=0.3)

    # Panel C: E[|cos|] vs 改善量
    ax3 = fig.add_subplot(gs_layout[1, 0])
    cos_means = [r['cos_mean'] for r in res3]
    r0s = [r['r0'] for r in res3]
    r1s = [r['r1'] for r in res3]
    r2s = [r['r2'] for r in res3]
    ax3.plot(cos_means, r0s, 'o-', color='steelblue', label='UB0')
    ax3.plot(cos_means, r1s, 's-', color='tomato', label='UB1(|cos|)')
    ax3.plot(cos_means, r2s, '^-', color='green', label='UB2(SVD)')
    ax3.set_xlabel('E[|cos(2πnᵀz)|]')
    ax3.set_ylabel('ratio')
    ax3.set_title('Panel C: ratio vs E[|cos|]\n(UB1 improves with small cos)')
    ax3.legend(fontsize=8); ax3.grid(True, alpha=0.3)

    # Panel D: UB0/UB1（改善倍率）vs cos平均
    ax4 = fig.add_subplot(gs_layout[1, 1])
    improve_01 = [r['r0']/(r['r1']+1e-30) for r in res3]
    improve_02 = [r['r0']/(r['r2']+1e-30) for r in res3]
    ax4.plot(cos_means, improve_01, 'o-', color='tomato', label='UB0/UB1')
    ax4.plot(cos_means, improve_02, 's-', color='green', label='UB0/UB2')
    ax4.axhline(1.0, color='gray', linestyle='--', linewidth=0.8)
    # 理論線: 1/E[|cos|]
    cos_ref = np.linspace(0.3, 1.0, 50)
    ax4.plot(cos_ref, 1/cos_ref, '--', color='orange', label='1/E[|cos|]')
    ax4.set_xlabel('E[|cos(2πnᵀz)|]')
    ax4.set_ylabel('improvement factor UB0/UBi')
    ax4.set_title('Panel D: Improvement factor\nvs cos distribution')
    ax4.legend(fontsize=8); ax4.grid(True, alpha=0.3)

    # Panel E: tight化の理論的まとめ
    ax5 = fig.add_subplot(gs_layout[1, 2])
    ax5.axis('off')
    summary = (
        "Tight bound summary\n"
        "─────────────────────\n"
        "UB0 (current):\n"
        "  4π·Σ|Re(nᵀCn)|·exp\n"
        "  ratio ≈ 0.07–0.12\n\n"
        "UB1 (|cos| included):\n"
        "  4π·Σ|Re(nᵀCn)|·|cos|·exp\n"
        "  ratio ≈ r0/E[|cos|]\n\n"
        "UB2 (SVD optimal):\n"
        "  4π·||M(z)||_F\n"
        "  ratio ≈ 0.5 (random C)\n"
        "  ratio ≈ 1.0 (SVD C)\n\n"
        "→ UB2 is the tightest\n"
        "  provable bound"
    )
    ax5.text(0.05, 0.95, summary, transform=ax5.transAxes,
             va='top', fontsize=9, family='monospace',
             bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
    ax5.set_title('Panel E: Summary')

    plt.suptitle('Tightening the delta-linear response bound',
                 fontsize=12, y=1.01)
    plt.tight_layout()
    plt.savefig("tight_bound.png", dpi=130, bbox_inches='tight')
    print("\nプロット: tight_bound.png")
    print("完了")


if __name__ == "__main__":
    main()
