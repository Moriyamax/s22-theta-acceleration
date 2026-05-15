"""
Resonant z での δ線形応答の挙動確認
目的: cos(2πnᵀz) が特定値に固定されるresonant zで
     ratio = |d/dδ Err| / precise_bound がどう変わるか

zのカテゴリ:
  Generic z:    z ~ N(0, 0.09·I)   → cos項が±1に均等分布（前回の設定）
  Resonant z:   z = p/q 形式の有理点 → cos(2πnᵀz) が有理数に固定
  Worst-case z: cos(2πnᵀz) = 1 を最大化するz → 上界に最も近い

実行: python resonant_z.py
"""

import numpy as np
from itertools import product
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# ============================================================
# 基本関数
# ============================================================

def make_omega_blocks(g, lmin, re_scale=0.1, seed=42):
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
    ReOm = rng.standard_normal((g, g)) * re_scale
    ReOm = (ReOm + ReOm.T) / 2
    ImOm = np.block([[B1, np.zeros((g1, g2))],
                     [np.zeros((g2, g1)), B2]])
    eigs = np.linalg.eigvalsh(ImOm)
    if eigs.min() < 0.01:
        ImOm += (0.01 - eigs.min()) * np.eye(g)
    return ReOm + 1j * ImOm, g1, g2


def make_Eoff(g1, g2, seed=0):
    rng = np.random.default_rng(seed + 999)
    C = rng.standard_normal((g1, g2)) + 1j * rng.standard_normal((g1, g2))
    C /= np.linalg.norm(C, 'fro')
    g = g1 + g2
    E = np.zeros((g, g), dtype=complex)
    E[:g1, g1:] = C
    E[g1:, :g1] = C.conj().T
    return E, C


def numerical_derivative(Om0, E, z, N_cut, g1, h=1e-5):
    g = Om0.shape[0]
    g2 = g - g1

    def theta_naive(Om):
        return sum(
            np.exp(np.pi*1j*(np.array(n,float)@Om@np.array(n,float)
                             + 2*np.array(n,float)@z))
            for n in product(range(-N_cut, N_cut+1), repeat=g)
        )

    def theta_skk(Om):
        t1 = sum(
            np.exp(np.pi*1j*(np.array(n,float)@Om[:g1,:g1]@np.array(n,float)
                             + 2*np.array(n,float)@z[:g1]))
            for n in product(range(-N_cut, N_cut+1), repeat=g1)
        )
        t2 = sum(
            np.exp(np.pi*1j*(np.array(n,float)@Om[g1:,g1:]@np.array(n,float)
                             + 2*np.array(n,float)@z[g1:]))
            for n in product(range(-N_cut, N_cut+1), repeat=g2)
        )
        return t1 * t2

    err_p = theta_naive(Om0 + h*E) - theta_skk(Om0 + h*E)
    err_m = theta_naive(Om0 - h*E) - theta_skk(Om0 - h*E)
    return abs((err_p - err_m) / (2*h))


def precise_upper_bound(Om0, C_block, N_cut, g1):
    g = Om0.shape[0]
    total = 0.0
    seen = set()
    for n in product(range(-N_cut, N_cut+1), repeat=g):
        nv = np.array(n, dtype=float)
        neg_n = tuple(-x for x in n)
        if n == tuple([0]*g) or neg_n in seen:
            continue
        seen.add(n)
        nA = nv[:g1]
        nB = nv[g1:]
        exp_val = np.exp(-np.pi * nv @ Om0.imag @ nv)
        nCn = abs(np.real(nA @ C_block @ nB))
        total += nCn * exp_val
    return 4 * np.pi * total


def analytic_lhs(Om0, E, z, N_cut, g1):
    """解析式で|d/dδ Err|を計算（数値微分より高精度）"""
    g = Om0.shape[0]
    result = 0j
    for n in product(range(-N_cut, N_cut+1), repeat=g):
        nv = np.array(n, dtype=float)
        nEn = nv @ E @ nv
        result += np.pi*1j * nEn * np.exp(np.pi*1j*(nv@Om0@nv + 2*nv@z))
    return abs(result)


def cos_stats(Om0, z, N_cut, g1, threshold=1e-10):
    """有効格子点でのcos(2πnᵀz)の統計"""
    g = Om0.shape[0]
    cos_vals = []
    seen = set()
    for n in product(range(-N_cut, N_cut+1), repeat=g):
        nv = np.array(n, dtype=float)
        neg_n = tuple(-x for x in n)
        if n == tuple([0]*g) or neg_n in seen:
            continue
        seen.add(n)
        exp_val = np.exp(-np.pi * nv @ Om0.imag @ nv)
        if exp_val > threshold:
            cos_vals.append(np.cos(2 * np.pi * nv @ z))
    if not cos_vals:
        return {'mean': 0, 'max': 0, 'min': 0, 'std': 0, 'n': 0}
    return {
        'mean': np.mean(cos_vals),
        'max': np.max(cos_vals),
        'min': np.min(cos_vals),
        'std': np.std(cos_vals),
        'n': len(cos_vals),
        'vals': cos_vals
    }


# ============================================================
# z のカテゴリ定義
# ============================================================

def generate_z_cases(g, seed=42):
    """様々なカテゴリのzを生成"""
    rng = np.random.default_rng(seed)
    cases = {}

    # 1. Generic z（前回の設定）
    cases['generic_0.3'] = rng.standard_normal(g) * 0.3

    # 2. Resonant z: z = 1/2 * ones（cos(2πnᵀz) = cos(πΣn_i)）
    cases['resonant_half'] = np.ones(g) * 0.5

    # 3. Resonant z: z = 1/4 * ones
    cases['resonant_quarter'] = np.ones(g) * 0.25

    # 4. Resonant z: z = 1/3 * ones
    cases['resonant_third'] = np.ones(g) * (1/3)

    # 5. Worst-case z候補: cos(2πnᵀz)=1を最大化
    # n=(1,0,...,0)に対してcos(2πz_1)=1 → z_1=0
    # → z=0が最悪ケース候補（全cosが1になる）
    cases['z_zero'] = np.zeros(g)

    # 6. Anti-resonant: cos項が最小化されるz
    # n=(1,0,...,0)に対してcos(2πz_1)=-1 → z_1=0.5
    # → z=(0.5, 0.5, ...) が部分的に最悪
    cases['anti_resonant'] = np.ones(g) * 0.5  # resonant_halfと同じ

    # 7. Small z（Taylor展開が有効な領域）
    cases['small_0.01'] = rng.standard_normal(g) * 0.01

    # 8. Large z
    cases['large_1.0'] = rng.standard_normal(g) * 1.0

    # 9. Mixed: 一部成分のみ非ゼロ
    z_mixed = np.zeros(g)
    z_mixed[0] = 0.5
    cases['mixed_first'] = z_mixed

    # 10. 複数のgeneric z（統計用）
    for i in range(5):
        cases[f'generic_{i}'] = rng.standard_normal(g) * 0.3

    return cases


# ============================================================
# 解析1: z カテゴリ別の ratio 比較
# ============================================================

def analysis_z_categories(g=8, lmin=3.0, N_cut=1, n_seeds=5):
    print(f"\n[解析1] z カテゴリ別の ratio (g={g}, lmin={lmin}, N_cut={N_cut})")
    print(f"  {'z種類':20} {'LHS':>12} {'精密上界':>12} {'ratio':>8} "
          f"{'cos_mean':>10} {'cos_max':>10}")
    print("  " + "-" * 80)

    results = []
    for seed in range(n_seeds):
        Om0, g1, g2 = make_omega_blocks(g, lmin, re_scale=0.1, seed=seed)
        E, C = make_Eoff(g1, g2, seed=seed)
        z_cases = generate_z_cases(g, seed=seed+100)
        pb = precise_upper_bound(Om0, C, N_cut, g1)

        for z_label, z in z_cases.items():
            if z_label.startswith('generic_') and z_label != 'generic_0.3':
                continue  # 個別のgenericは後でまとめて処理

            lhs = analytic_lhs(Om0, E, z, N_cut, g1)
            ratio = lhs / (pb + 1e-30)
            cs = cos_stats(Om0, z, N_cut, g1)

            results.append({
                'z_label': z_label, 'seed': seed,
                'lhs': lhs, 'pb': pb, 'ratio': ratio,
                'cos_mean': cs['mean'], 'cos_max': cs['max'],
                'cos_std': cs['std']
            })

    # カテゴリ別に集計
    categories = ['z_zero', 'resonant_half', 'resonant_quarter',
                  'resonant_third', 'small_0.01', 'large_1.0',
                  'mixed_first', 'generic_0.3']
    cat_results = {}
    for cat in categories:
        subset = [r for r in results if r['z_label'] == cat]
        if subset:
            cat_results[cat] = {
                'lhs_mean': np.mean([r['lhs'] for r in subset]),
                'ratio_mean': np.mean([r['ratio'] for r in subset]),
                'ratio_max': np.max([r['ratio'] for r in subset]),
                'cos_mean': np.mean([r['cos_mean'] for r in subset]),
                'cos_max': np.mean([r['cos_max'] for r in subset]),
                'pb_mean': np.mean([r['pb'] for r in subset]),
            }
            r = cat_results[cat]
            print(f"  {cat:20} {r['lhs_mean']:12.3e} {r['pb_mean']:12.3e} "
                  f"{r['ratio_mean']:8.4f} {r['cos_mean']:10.4f} {r['cos_max']:10.4f}")

    return cat_results


# ============================================================
# 解析2: z=0（worst-case候補）での ratio vs λ_min
# ============================================================

def analysis_worst_case_z(g=8, N_cut=1, n_seeds=5):
    print(f"\n[解析2] z=0 vs generic z での ratio 比較 (g={g}, N_cut={N_cut})")
    print("  z=0はcos(2πnᵀz)=1なのでLHSが最大になるはず")
    print(f"  {'lmin':>6} {'ratio(z=0)':>12} {'ratio(generic)':>14} "
          f"{'比 z=0/generic':>16} {'上界到達度':>12}")
    print("  " + "-" * 66)

    lmin_vals = [0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0]
    results = []

    for lmin in lmin_vals:
        ratios_z0, ratios_gen = [], []
        for seed in range(n_seeds):
            Om0, g1, g2 = make_omega_blocks(g, lmin, re_scale=0.0, seed=seed)
            # Re(Ω)=0にしてcosの効果を純粋に見る
            E, C = make_Eoff(g1, g2, seed=seed)
            pb = precise_upper_bound(Om0, C, N_cut, g1)

            # z=0
            lhs_z0 = analytic_lhs(Om0, E, np.zeros(g), N_cut, g1)
            ratios_z0.append(lhs_z0 / (pb + 1e-30))

            # generic z
            rng = np.random.default_rng(seed + 500)
            z_gen = rng.standard_normal(g) * 0.3
            lhs_gen = analytic_lhs(Om0, E, z_gen, N_cut, g1)
            ratios_gen.append(lhs_gen / (pb + 1e-30))

        r0 = np.mean(ratios_z0)
        rg = np.mean(ratios_gen)
        print(f"  {lmin:6.1f} {r0:12.4f} {rg:14.4f} "
              f"{r0/(rg+1e-30):16.3f} {'最大に近い' if r0 > 0.5 else '余裕あり':>12}")
        results.append({'lmin': lmin, 'ratio_z0': r0, 'ratio_gen': rg})

    return results


# ============================================================
# 解析3: cos(2πnᵀz)の分布のzカテゴリ依存性
# ============================================================

def analysis_cos_distribution(g=8, lmin=3.0, N_cut=1, seed=0):
    print(f"\n[解析3] cos(2πnᵀz)の分布 (g={g}, lmin={lmin})")
    Om0, g1, g2 = make_omega_blocks(g, lmin, re_scale=0.0, seed=seed)
    E, C = make_Eoff(g1, g2, seed=seed)
    pb = precise_upper_bound(Om0, C, N_cut, g1)

    z_cases = {
        'z=0 (worst-case)': np.zeros(g),
        'z=1/2·1 (resonant)': np.ones(g) * 0.5,
        'z=1/4·1 (resonant)': np.ones(g) * 0.25,
        'z~N(0,0.09) (generic)': np.random.default_rng(42).standard_normal(g) * 0.3,
    }

    results = {}
    print(f"  {'z種類':25} {'N_eff':>6} {'cos_mean':>10} {'cos_std':>10} "
          f"{'ratio':>8} {'上界の何%か':>12}")
    print("  " + "-" * 76)

    for label, z in z_cases.items():
        cs = cos_stats(Om0, z, N_cut, g1)
        lhs = analytic_lhs(Om0, E, z, N_cut, g1)
        ratio = lhs / (pb + 1e-30)
        print(f"  {label:25} {cs['n']:6d} {cs['mean']:10.4f} {cs['std']:10.4f} "
              f"{ratio:8.4f} {100*ratio:12.1f}%")
        results[label] = {'cs': cs, 'ratio': ratio, 'lhs': lhs}

    return results, Om0, E, C, pb


# ============================================================
# メイン
# ============================================================

def main():
    print("=" * 65)
    print("Resonant z での δ線形応答の挙動確認")
    print("cos(2πnᵀz) の z カテゴリ依存性")
    print("=" * 65)

    g = 8
    lmin = 3.0
    N_cut = 1

    res1 = analysis_z_categories(g=g, lmin=lmin, N_cut=N_cut, n_seeds=5)
    res2 = analysis_worst_case_z(g=g, N_cut=N_cut, n_seeds=5)
    res3, Om0, E, C, pb = analysis_cos_distribution(g=g, lmin=lmin, N_cut=N_cut, seed=0)

    # ---- プロット ----
    fig = plt.figure(figsize=(14, 10))
    gs_layout = gridspec.GridSpec(2, 3, fig, hspace=0.45, wspace=0.35)

    # Panel A: z カテゴリ別の ratio
    ax = fig.add_subplot(gs_layout[0, :2])
    cats = list(res1.keys())
    ratios = [res1[c]['ratio_mean'] for c in cats]
    colors = ['red' if 'zero' in c or 'resonant' in c else
              'orange' if 'small' in c or 'large' in c else
              'steelblue' for c in cats]
    bars = ax.bar(range(len(cats)), ratios, color=colors, alpha=0.7)
    ax.axhline(1.0, color='red', linestyle='--', linewidth=1, label='bound=1.0')
    ax.set_xticks(range(len(cats)))
    ax.set_xticklabels(cats, rotation=35, ha='right', fontsize=8)
    ax.set_ylabel('ratio = |d/dδ Err| / precise_bound')
    ax.set_title(f'Panel A: ratio by z category\n(g={g}, lmin={lmin}, red=resonant/worst-case)')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    from matplotlib.patches import Patch
    ax.legend(handles=[
        Patch(color='red', alpha=0.7, label='resonant/worst-case z'),
        Patch(color='steelblue', alpha=0.7, label='generic z'),
        plt.Line2D([0],[0], color='red', linestyle='--', label='bound=1.0')
    ])

    # Panel B: z=0 vs generic の ratio vs λ_min
    ax2 = fig.add_subplot(gs_layout[0, 2])
    lmins = [r['lmin'] for r in res2]
    ax2.plot(lmins, [r['ratio_z0'] for r in res2], 'o-', color='red', label='z=0 (worst-case)')
    ax2.plot(lmins, [r['ratio_gen'] for r in res2], 's-', color='steelblue', label='generic z')
    ax2.axhline(1.0, color='red', linestyle='--', linewidth=0.8)
    ax2.set_xlabel(r'$\lambda_\min$')
    ax2.set_ylabel('ratio')
    ax2.set_title('Panel B: z=0 vs generic\nratio vs λ_min')
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)

    # Panel C: cos分布のヒストグラム（各zカテゴリ）
    ax3 = fig.add_subplot(gs_layout[1, :2])
    colors_cos = ['red', 'orange', 'green', 'steelblue']
    for (label, data), color in zip(res3.items(), colors_cos):
        cs = data['cs']
        if 'vals' in cs and len(cs['vals']) > 0:
            ax3.hist(cs['vals'], bins=20, alpha=0.5, color=color,
                     label=f"{label[:20]} (r={data['ratio']:.3f})", density=True)
    ax3.axvline(0, color='black', linestyle='--', linewidth=0.8)
    ax3.set_xlabel('cos(2πnᵀz)')
    ax3.set_ylabel('density')
    ax3.set_title(f'Panel C: cos distribution by z category\n(g={g}, lmin={lmin})')
    ax3.legend(fontsize=7)
    ax3.grid(True, alpha=0.3)

    # Panel D: ratio_max の z カテゴリ依存性
    ax4 = fig.add_subplot(gs_layout[1, 2])
    cats2 = list(res1.keys())
    ratio_maxes = [res1[c]['ratio_max'] for c in cats2]
    colors2 = ['red' if 'zero' in c or 'resonant' in c else 'steelblue' for c in cats2]
    ax4.bar(range(len(cats2)), ratio_maxes, color=colors2, alpha=0.7)
    ax4.axhline(1.0, color='red', linestyle='--', linewidth=1)
    ax4.set_xticks(range(len(cats2)))
    ax4.set_xticklabels(cats2, rotation=35, ha='right', fontsize=7)
    ax4.set_ylabel('ratio_max')
    ax4.set_title('Panel D: ratio_max by z category\n(worst case over seeds)')
    ax4.grid(True, alpha=0.3, axis='y')

    plt.suptitle('Resonant z analysis: cos(2πnᵀz) dependence of δ-linear bound',
                 fontsize=12, y=1.01)
    plt.tight_layout()
    plt.savefig("resonant_z.png", dpi=130, bbox_inches='tight')
    print("\nプロット: resonant_z.png")
    print("完了")


if __name__ == "__main__":
    main()
