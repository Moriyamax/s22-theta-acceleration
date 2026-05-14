"""
δ線形項上界の拡張検証
- 複数g: g=6,8,10,12
- N_cut=1 と N_cut=2 の比較
- λ_min範囲: 0.5–5.0

目的: 定理の数値的根拠を強化する
  |d/dδ Err(Ω+δE_off,z)|_{δ=0}| ≤ 4π·||C||_F·N_eff·exp(−πλ_min)

実行: python delta_linear_extended.py
出力: delta_linear_extended.png + delta_linear_extended.txt
"""

import numpy as np
from itertools import product
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# ============================================================
# 基本関数（前回と同じ）
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
    def err(Om):
        g = Om.shape[0]
        g2 = g - g1
        naive = sum(
            np.exp(np.pi*1j*(np.array(n,float)@Om@np.array(n,float)
                             + 2*np.array(n,float)@z))
            for n in product(range(-N_cut, N_cut+1), repeat=g)
        )
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
        return naive - t1 * t2
    return (err(Om0 + h*E) - err(Om0 - h*E)) / (2*h)


def analytic_derivative(Om0, E, z, N_cut, g1):
    g = Om0.shape[0]
    result = 0j
    for n in product(range(-N_cut, N_cut+1), repeat=g):
        nv = np.array(n, dtype=float)
        nEn = nv @ E @ nv
        result += np.pi*1j * nEn * np.exp(np.pi*1j*(nv@Om0@nv + 2*nv@z))
    return result


def precise_upper_bound(Om0, C_block, N_cut, g1):
    """4π · Σ_{n in half-lattice} |Re(n_A^T C n_B)| · exp(-π n^T Im(Ω) n)"""
    g = Om0.shape[0]
    g2 = g - g1
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


def neff_count(Om0, N_cut, threshold=1e-10):
    g = Om0.shape[0]
    n0_val = 1.0  # n=0の寄与は常に~1
    count = 0
    for n in product(range(-N_cut, N_cut+1), repeat=g):
        nv = np.array(n, dtype=float)
        if np.all(nv == 0):
            continue
        if np.exp(-np.pi * nv @ Om0.imag @ nv) > threshold:
            count += 1
    return count


# ============================================================
# メイン解析
# ============================================================

def run_analysis(g_vals, lmin_vals, N_cut, n_seeds, z_scale=0.3):
    """全条件で解析を実行"""
    results = []
    total = len(g_vals) * len(lmin_vals) * n_seeds
    done = 0

    for g in g_vals:
        for lmin in lmin_vals:
            nds, ads, pbs, ratios, neffs = [], [], [], [], []
            for seed in range(n_seeds):
                Om0, g1, g2 = make_omega_blocks(g, lmin, re_scale=0.1, seed=seed)
                E, C = make_Eoff(g1, g2, seed=seed)
                rng = np.random.default_rng(seed + 500)
                z = rng.standard_normal(g) * z_scale

                nd = abs(numerical_derivative(Om0, E, z, N_cut, g1))
                ad = abs(analytic_derivative(Om0, E, z, N_cut, g1))
                pb = precise_upper_bound(Om0, C, N_cut, g1)
                neff = neff_count(Om0, N_cut)

                nds.append(nd)
                ads.append(ad)
                pbs.append(pb)
                ratios.append(nd / (pb + 1e-30))
                neffs.append(neff)

                done += 1
                if done % 20 == 0:
                    print(f"  進捗: {done}/{total} ({100*done/total:.0f}%)", end='\r')

            lmin_act = np.linalg.eigvalsh(Om0.imag).min()
            results.append({
                'g': g, 'lmin': lmin, 'lmin_act': lmin_act,
                'N_cut': N_cut,
                'nd_mean': np.mean(nds), 'nd_std': np.std(nds),
                'ad_mean': np.mean(ads),
                'pb_mean': np.mean(pbs),
                'ratio_mean': np.mean(ratios), 'ratio_max': np.max(ratios),
                'neff_mean': np.mean(neffs),
                'exp_lmin': np.exp(-np.pi * lmin_act),
                'bound_holds': all(r < 1.0 for r in ratios),
            })
    print()
    return results


def main():
    print("=" * 65)
    print("δ線形項上界の拡張検証")
    print("複数g + N_cut=1,2での確認")
    print("=" * 65)

    lmin_vals = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0]
    z_scale = 0.3
    n_seeds = 6
    out_lines = []

    # ---- N_cut=1: g=6,8,10,12 ----
    print("\n[N_cut=1] g=6,8,10,12")
    g_vals_ncut1 = [6, 8, 10, 12]
    res1 = run_analysis(g_vals_ncut1, lmin_vals, N_cut=1, n_seeds=n_seeds, z_scale=z_scale)

    # ---- N_cut=2: g=4,6,8 （N_cut=2はg=10以上が重いので小さめ） ----
    print("\n[N_cut=2] g=4,6,8")
    g_vals_ncut2 = [4, 6, 8]
    res2 = run_analysis(g_vals_ncut2, lmin_vals, N_cut=2, n_seeds=n_seeds, z_scale=z_scale)

    all_results = res1 + res2

    # ---- 結果表示 ----
    print("\n結果サマリー（全条件）")
    print(f"{'N_cut':>6} {'g':>4} {'lmin':>6} {'|d/dδ Err|':>12} "
          f"{'精密上界':>12} {'ratio':>8} {'上界成立':>8}")
    print("-" * 65)
    out_lines.append("N_cut,g,lmin,nd_mean,pb_mean,ratio_mean,ratio_max,bound_holds")

    for r in all_results:
        mark = "✓" if r['bound_holds'] else "✗"
        print(f"{r['N_cut']:6d} {r['g']:4d} {r['lmin_act']:6.2f} "
              f"{r['nd_mean']:12.3e} {r['pb_mean']:12.3e} "
              f"{r['ratio_mean']:8.4f} {mark:>8}")
        out_lines.append(
            f"{r['N_cut']},{r['g']},{r['lmin_act']:.2f},"
            f"{r['nd_mean']:.4e},{r['pb_mean']:.4e},"
            f"{r['ratio_mean']:.4f},{r['ratio_max']:.4f},{r['bound_holds']}"
        )

    # 上界成立の総括
    n_total = len(all_results)
    n_hold  = sum(1 for r in all_results if r['bound_holds'])
    print(f"\n上界成立: {n_hold}/{n_total} 条件 ({100*n_hold/n_total:.1f}%)")
    print(f"ratio の最大値: {max(r['ratio_max'] for r in all_results):.4f}")
    print(f"ratio の平均値: {np.mean([r['ratio_mean'] for r in all_results]):.4f}")
    out_lines.append(f"\n上界成立: {n_hold}/{n_total}")
    out_lines.append(f"ratio最大: {max(r['ratio_max'] for r in all_results):.4f}")

    with open("delta_linear_extended.txt", "w") as f:
        f.write("\n".join(out_lines))
    print("結果: delta_linear_extended.txt")

    # ---- プロット ----
    fig = plt.figure(figsize=(15, 12))
    gs_layout = gridspec.GridSpec(2, 3, fig, hspace=0.45, wspace=0.35)

    colors_g = {4:'purple', 6:'steelblue', 8:'tomato', 10:'green', 12:'orange'}
    lmins_unique = sorted(set(r['lmin_act'] for r in all_results))

    # Panel A: N_cut=1, g別の|d/dδ Err| vs λ_min
    ax = fig.add_subplot(gs_layout[0, 0])
    for g in g_vals_ncut1:
        sub = [r for r in res1 if r['g'] == g]
        lms = [r['lmin_act'] for r in sub]
        nds = [r['nd_mean'] for r in sub]
        ax.plot(lms, nds, 'o-', color=colors_g[g], label=f'g={g}')
    exp_ref = [np.exp(-np.pi*l) for l in lmins_unique]
    ax.plot(lmins_unique, exp_ref, 'k--', linewidth=1, label='exp(−πλ)')
    ax.set_yscale('log')
    ax.set_xlabel(r'$\lambda_\min$')
    ax.set_ylabel('|d/dδ Err|')
    ax.set_title('Panel A: N_cut=1, multiple g\n|d/dδ Err| vs λ_min')
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    # Panel B: N_cut=2, g別の|d/dδ Err| vs λ_min
    ax2 = fig.add_subplot(gs_layout[0, 1])
    for g in g_vals_ncut2:
        sub = [r for r in res2 if r['g'] == g]
        lms = [r['lmin_act'] for r in sub]
        nds = [r['nd_mean'] for r in sub]
        ax2.plot(lms, nds, 's-', color=colors_g[g], label=f'g={g}')
    ax2.plot(lmins_unique, exp_ref, 'k--', linewidth=1, label='exp(−πλ)')
    ax2.set_yscale('log')
    ax2.set_xlabel(r'$\lambda_\min$')
    ax2.set_ylabel('|d/dδ Err|')
    ax2.set_title('Panel B: N_cut=2, multiple g\n|d/dδ Err| vs λ_min')
    ax2.legend(fontsize=8); ax2.grid(True, alpha=0.3)

    # Panel C: ratio(LHS/上界) - 全条件で1以下を確認
    ax3 = fig.add_subplot(gs_layout[0, 2])
    for r in all_results:
        marker = 'o' if r['N_cut'] == 1 else 's'
        color = colors_g.get(r['g'], 'gray')
        ax3.scatter(r['lmin_act'], r['ratio_mean'],
                    marker=marker, color=color, s=40, alpha=0.7)
    ax3.axhline(1.0, color='red', linestyle='--', label='bound=1.0')
    ax3.set_xlabel(r'$\lambda_\min$')
    ax3.set_ylabel('|d/dδ Err| / precise_bound')
    ax3.set_title('Panel C: Bound validity\nall conditions (o=N_cut=1, s=N_cut=2)')
    # 凡例
    from matplotlib.lines import Line2D
    handles = [Line2D([0],[0],marker='o',color='w',markerfacecolor=colors_g[g],
                      markersize=8,label=f'g={g}') for g in sorted(colors_g)]
    handles.append(Line2D([0],[0],color='red',linestyle='--',label='bound=1.0'))
    ax3.legend(handles=handles, fontsize=7)
    ax3.grid(True, alpha=0.3)

    # Panel D: N_cut=1 vs N_cut=2 の|d/dδ Err|比較（g=6,8共通）
    ax4 = fig.add_subplot(gs_layout[1, 0])
    for g in [6, 8]:
        sub1 = [r for r in res1 if r['g'] == g]
        sub2 = [r for r in res2 if r['g'] == g]
        lms1 = [r['lmin_act'] for r in sub1]
        lms2 = [r['lmin_act'] for r in sub2]
        ax4.plot(lms1, [r['nd_mean'] for r in sub1], 'o-',
                 color=colors_g[g], label=f'g={g},N_cut=1')
        ax4.plot(lms2, [r['nd_mean'] for r in sub2], 's--',
                 color=colors_g[g], alpha=0.6, label=f'g={g},N_cut=2')
    ax4.set_yscale('log')
    ax4.set_xlabel(r'$\lambda_\min$')
    ax4.set_ylabel('|d/dδ Err|')
    ax4.set_title('Panel D: N_cut=1 vs N_cut=2\n(same g, consistent?)')
    ax4.legend(fontsize=7); ax4.grid(True, alpha=0.3)

    # Panel E: スケーリング則 LHS/exp(−πλ) - gとN_cutによらず定数か
    ax5 = fig.add_subplot(gs_layout[1, 1])
    for g in g_vals_ncut1:
        sub = [r for r in res1 if r['g'] == g]
        lms = [r['lmin_act'] for r in sub]
        scaled = [r['nd_mean']/r['exp_lmin'] for r in sub]
        ax5.plot(lms, scaled, 'o-', color=colors_g[g], label=f'g={g},Nc=1')
    for g in g_vals_ncut2:
        sub = [r for r in res2 if r['g'] == g]
        lms = [r['lmin_act'] for r in sub]
        scaled = [r['nd_mean']/r['exp_lmin'] for r in sub]
        ax5.plot(lms, scaled, 's--', color=colors_g[g], alpha=0.6, label=f'g={g},Nc=2')
    ax5.set_xlabel(r'$\lambda_\min$')
    ax5.set_ylabel('|d/dδ Err| / exp(−πλ)')
    ax5.set_title('Panel E: Scaling law\n|d/dδ Err|/exp(−πλ) vs λ_min')
    ax5.legend(fontsize=6, ncol=2); ax5.grid(True, alpha=0.3)

    # Panel F: 数値微分 vs 解析式（全条件）
    ax6 = fig.add_subplot(gs_layout[1, 2])
    for r in all_results:
        color = colors_g.get(r['g'], 'gray')
        ax6.scatter(r['ad_mean'], r['nd_mean'], color=color, s=20, alpha=0.7)
    mn = min(r['ad_mean'] for r in all_results)
    mx = max(r['ad_mean'] for r in all_results)
    ax6.plot([mn, mx], [mn, mx], 'r--', label='y=x')
    ax6.set_xscale('log'); ax6.set_yscale('log')
    ax6.set_xlabel('analytic formula')
    ax6.set_ylabel('numerical derivative')
    ax6.set_title('Panel F: Numerical vs analytic\n(all g, all N_cut)')
    ax6.legend(fontsize=8); ax6.grid(True, alpha=0.3)

    plt.suptitle(
        r'$\delta$-linear bound: extended verification (multiple $g$, $N_{\rm cut}$=1,2)',
        fontsize=12, y=1.01
    )
    plt.tight_layout()
    plt.savefig("delta_linear_extended.png", dpi=130, bbox_inches='tight')
    print("プロット: delta_linear_extended.png")
    print("完了")


if __name__ == "__main__":
    main()
