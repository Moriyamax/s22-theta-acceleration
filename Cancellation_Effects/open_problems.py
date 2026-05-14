"""
未解決問題の数値探索
目的1: collapse現象（特定(g,δ)でΩが全て機械精度にロック）の原因特定
  候補B: 素数g_blockの特殊性
  候補D: λ_min分布の集中（Marchenko-Pastur則）

目的2: C(|z|)の解析的表現の検証
  sin展開: C(|z|) ≈ Σ_{n≠0} |sin(2π nᵀz)| · exp(−π nᵀIm(Ω)n) / exp(−π λ_min)

実行: python open_problems.py
出力: open_problems_results.txt + open_problems.png
"""

import numpy as np
from itertools import product
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from math import factorial

# ============================================================
# 基本関数
# ============================================================

def make_omega(g, lmin_target, delta=0.3, seed=42):
    rng = np.random.default_rng(seed)
    # 奇数g対応: g1=g//2, g2=g-g1（g=11なら g1=5, g2=6）
    g1 = g // 2
    g2 = g - g1

    def make_block(n):
        A = rng.standard_normal((n, n))
        Q, _ = np.linalg.qr(A)
        eigs = lmin_target + rng.exponential(0.5, n)
        return Q @ np.diag(eigs) @ Q.T

    B1 = make_block(g1)
    B2 = make_block(g2)
    C_re = rng.standard_normal((g1, g2)) * delta * 0.3

    ImOm = np.block([[B1, C_re], [C_re.T, B2]])
    ReOm = rng.standard_normal((g, g)) * 0.1
    ReOm = (ReOm + ReOm.T) / 2

    eigs = np.linalg.eigvalsh(ImOm)
    if eigs.min() < 0.01:
        ImOm += (0.01 - eigs.min()) * np.eye(g)

    return ReOm + 1j * ImOm


def theta_s22(Omega, z, N_cut=1):
    g = Omega.shape[0]
    g1 = g // 2
    g2 = g - g1
    Om1, Om2 = Omega[:g1, :g1], Omega[g1:, g1:]
    z1, z2 = z[:g1], z[g1:]

    def theta_block(Om, zb):
        result = 0j
        for n in product(range(-N_cut, N_cut+1), repeat=Om.shape[0]):
            nv = np.array(n, dtype=float)
            result += np.exp(np.pi*1j*(nv@Om@nv + 2*nv@zb))
        return result

    return theta_block(Om1, z1) * theta_block(Om2, z2)


def theta_naive(Omega, z, N_cut=1):
    g = Omega.shape[0]
    result = 0j
    for n in product(range(-N_cut, N_cut+1), repeat=g):
        nv = np.array(n, dtype=float)
        result += np.exp(np.pi*1j*(nv@Omega@nv + 2*nv@z))
    return result


def rel_error(Omega, z, N_cut=1):
    s22 = theta_s22(Omega, z, N_cut)
    naive = theta_naive(Omega, z, N_cut)
    return abs(s22 - naive) / (abs(naive) + 1e-30)

# ============================================================
# 解析1: collapse現象 — λ_min分布の比較
# ============================================================

def analysis_collapse_lmin(N_Omega=50, N_cut=1):
    """
    collapse発生条件 vs 非collapse条件でλ_min分布を比較
    collapse既知: g=10/δ=0.5, g=11/δ=0.1,0.5
    非collapse: g=10/δ=0.1, g=10/δ=1.0, g=12/δ=0.5
    """
    print("\n[解析1] collapse現象 — λ_min分布とerror分布の比較")
    print("  collapse判定: 全Ωのlog10(error)中央値が-20以下")
    print()

    # (g, delta, lmin_target, label)
    configs = [
        (10, 0.1, 3.5, "g=10,d=0.1,lmin~3.5"),
        (10, 0.5, 3.5, "g=10,d=0.5,lmin~3.5"),
        (10, 1.0, 3.5, "g=10,d=1.0,lmin~3.5"),
        (11, 0.1, 3.5, "g=11,d=0.1,lmin~3.5"),
        (11, 0.5, 3.5, "g=11,d=0.5,lmin~3.5"),
        (11, 1.0, 3.5, "g=11,d=1.0,lmin~3.5"),
        (10, 0.5, 2.0, "g=10,d=0.5,lmin~2.0"),
        (11, 0.5, 2.0, "g=11,d=0.5,lmin~2.0"),
        (12, 0.5, 3.5, "g=12,d=0.5,lmin~3.5"),
    ]

    results = []
    print(f"  {'条件':28} {'lmin mean':>12} {'lmin std':>10} {'err median':>12} {'collapse':>10}")
    print("  " + "-" * 78)

    for g, delta, lmin_t, label in configs:
        lmins, errs = [], []
        rng = np.random.default_rng(999)
        for seed in range(N_Omega):
            Om = make_omega(g, lmin_target=lmin_t, delta=delta, seed=seed)
            lmin = np.linalg.eigvalsh(Om.imag).min()
            z = rng.standard_normal(g) * 0.3
            err = rel_error(Om, z, N_cut)
            lmins.append(lmin)
            errs.append(np.log10(err + 1e-35))

        lmin_mean = np.mean(lmins)
        lmin_std  = np.std(lmins)
        err_med   = np.median(errs)
        collapsed = err_med < -20

        print(f"  {label:28} {lmin_mean:12.3f} {lmin_std:10.3f} "
              f"{err_med:12.2f} {str(collapsed):>10}")

        results.append({
            'g': g, 'delta': delta, 'lmin_target': lmin_t, 'label': label,
            'lmin_mean': lmin_mean, 'lmin_std': lmin_std,
            'lmins': lmins, 'errs': errs,
            'err_median': err_med, 'collapsed': collapsed,
        })

    return results


# ============================================================
# 解析2: collapse現象 — 素数g_blockの検証
# ============================================================

def analysis_prime_gblock(N_Omega=30, N_z=20, N_cut=1):
    """
    g_block（= g/2）が素数かどうかでerror分布が変わるか検証
    g=10: g_block=5（素数）
    g=12: g_block=6（合成数）
    g=11: g_block=5.5（非整数、S(2,2)でblock=6と5の非対称）
    g=14: g_block=7（素数）
    g=16: g_block=8（合成数）
    """
    print("\n[解析2] 素数g_blockの効果")
    print(f"  {'g':>4} {'g_block':>8} {'prime?':>8} "
          f"{'err_mean':>12} {'err_std':>10} {'<-20 (%)':>10}")
    print("  " + "-" * 58)

    gs = [8, 9, 10, 11, 12]  # g>=13はN_cut=1でも格子点数過多
    results = []

    def is_prime(n):
        if n < 2: return False
        for i in range(2, int(n**0.5)+1):
            if n % i == 0: return False
        return True

    rng = np.random.default_rng(777)
    for g in gs:
        gb = g // 2
        prime = is_prime(gb)
        errs = []
        for seed in range(N_Omega):
            Om = make_omega(g, lmin_target=2.0, delta=0.5, seed=seed)
            for _ in range(N_z):
                z = rng.standard_normal(g) * 0.3
                e = rel_error(Om, z, N_cut)
                errs.append(np.log10(e + 1e-35))

        mean_e = np.mean(errs)
        std_e  = np.std(errs)
        frac20 = np.mean(np.array(errs) < -20) * 100

        print(f"  {g:4d} {gb:8d} {'yes' if prime else 'no':>8} "
              f"{mean_e:12.2f} {std_e:10.2f} {frac20:10.1f}")
        results.append({'g': g, 'gb': gb, 'prime': prime,
                        'mean': mean_e, 'std': std_e,
                        'frac20': frac20, 'errs': errs})

    return results


# ============================================================
# 解析3: C(|z|)の sin展開による解析的近似
# ============================================================

def C_sin_expansion(Om_block, z, N_cut=2):
    """
    C_theory(|z|) = Σ_{n≠0} |sin(2π nᵀz)| · exp(−π nᵀIm(Ω)n)
                    / (exp(−π λ_min) · |Θ_A|)

    これがC_measured(|z|)と一致するか検証
    """
    gb = Om_block.shape[0]
    lmin = np.linalg.eigvalsh(Om_block.imag).min()

    # Theta値（分母用）
    theta_val = 0j
    for n in product(range(-N_cut, N_cut+1), repeat=gb):
        nv = np.array(n, dtype=float)
        theta_val += np.exp(np.pi*1j*(nv@Om_block@nv + 2*nv@z))

    # sin展開
    sin_sum = 0.0
    for n in product(range(-N_cut, N_cut+1), repeat=gb):
        nv = np.array(n, dtype=float)
        if np.all(nv == 0):
            continue
        exp_term = np.exp(-np.pi * nv @ Om_block.imag @ nv)
        sin_term = abs(np.sin(2 * np.pi * nv @ z))
        sin_sum += sin_term * exp_term

    # C_theory
    C_theory = sin_sum / (np.exp(-np.pi * lmin) * abs(theta_val) + 1e-30)

    # C_measured: |μ_A|/|Θ_A| / exp(−π λ_min)
    grad = np.zeros(gb, dtype=complex)
    h = 1e-6
    for j in range(gb):
        zp, zm = z.copy(), z.copy()
        zp[j] += h; zm[j] -= h
        theta_p = sum(np.exp(np.pi*1j*(np.array(n,float)@Om_block@np.array(n,float)
                                        + 2*np.array(n,float)@zp))
                      for n in product(range(-N_cut,N_cut+1), repeat=gb))
        theta_m = sum(np.exp(np.pi*1j*(np.array(n,float)@Om_block@np.array(n,float)
                                        + 2*np.array(n,float)@zm))
                      for n in product(range(-N_cut,N_cut+1), repeat=gb))
        grad[j] = (theta_p - theta_m) / (2 * h)

    mu_A = grad / (2 * np.pi * 1j)
    C_measured = np.linalg.norm(mu_A) / (abs(theta_val) * np.exp(-np.pi * lmin) + 1e-30)

    return C_theory, C_measured, lmin


def analysis_c_sin_expansion(lmin_vals=[2.0, 3.0, 5.0], gb=3, N_cut=2, n_seeds=5):
    """sin展開によるC(|z|)の理論予測と実測値の比較"""
    print("\n[解析3] C(|z|)のsin展開検証")
    print("  仮説: C_theory = Σ|sin(2πnᵀz)|·exp(−πnᵀIm(Ω)n) / (exp(−πλ)·|Θ|)")
    print()
    print(f"  {'lmin':>6} {'|z|':>6} {'C_theory':>12} {'C_measured':>12} {'ratio':>8}")
    print("  " + "-" * 50)

    z_scales = [0.05, 0.1, 0.2, 0.5, 1.0]
    results = []

    for lmin in lmin_vals:
        for seed in range(n_seeds):
            # Re(Ω)=0で純粋にzの効果を見る
            rng = np.random.default_rng(seed)
            A = rng.standard_normal((gb, gb))
            Q, _ = np.linalg.qr(A)
            eigs = lmin + rng.exponential(0.3, gb)
            Om = 1j * (Q @ np.diag(eigs) @ Q.T)  # 純虚数

            for z_scale in z_scales:
                rng2 = np.random.default_rng(seed + 1000)
                z = rng2.standard_normal(gb) * z_scale
                C_th, C_me, lmin_act = C_sin_expansion(Om, z, N_cut)
                ratio = C_th / (C_me + 1e-30)
                results.append({
                    'lmin': lmin_act, 'z_scale': z_scale,
                    'C_theory': C_th, 'C_measured': C_me, 'ratio': ratio
                })

        # 代表値を表示
        for z_scale in z_scales:
            subset = [r for r in results
                      if abs(r['lmin'] - lmin) < 0.5 and r['z_scale'] == z_scale]
            if subset:
                ct = np.mean([r['C_theory'] for r in subset])
                cm = np.mean([r['C_measured'] for r in subset])
                print(f"  {lmin:6.1f} {z_scale:6.3f} {ct:12.3e} {cm:12.3e} {ct/cm:8.3f}")

    return results


# ============================================================
# メイン
# ============================================================

def main():
    print("=" * 60)
    print("未解決問題の数値探索")
    print("=" * 60)

    out = []

    # 解析1: collapse — λ_min分布
    res1 = analysis_collapse_lmin(N_Omega=40, N_cut=1)

    # 解析2: 素数g_block
    res2 = analysis_prime_gblock(N_Omega=15, N_z=5, N_cut=1)

    # 解析3: sin展開
    res3 = analysis_c_sin_expansion(lmin_vals=[2.0, 3.0, 5.0], gb=3, N_cut=2, n_seeds=3)

    # ---- プロット ----
    fig = plt.figure(figsize=(15, 12))
    gs_layout = gridspec.GridSpec(2, 3, fig, hspace=0.45, wspace=0.35)

    # Panel A: collapse条件のλ_min分布（boxplot）
    ax = fig.add_subplot(gs_layout[0, :2])
    labels_short = [r['label'].split('[')[0].strip() for r in res1]
    data_lmin = [r['lmins'] for r in res1]
    colors = ['tomato' if r['collapsed'] else 'steelblue' for r in res1]
    bp = ax.boxplot(data_lmin, labels=labels_short, patch_artist=True)
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
    ax.set_ylabel(r'$\lambda_\min$')
    ax.set_title('Panel A: λ_min distribution\nred=collapse, blue=non-collapse')
    ax.tick_params(axis='x', rotation=25)
    ax.grid(True, alpha=0.3, axis='y')
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color='tomato', alpha=0.6, label='collapse'),
                       Patch(color='steelblue', alpha=0.6, label='non-collapse')])

    # Panel B: collapse条件のerror分布
    ax2 = fig.add_subplot(gs_layout[0, 2])
    data_err = [r['errs'] for r in res1]
    bp2 = ax2.boxplot(data_err, labels=[str(r['g']) for r in res1], patch_artist=True)
    for patch, color in zip(bp2['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
    ax2.axhline(-20, color='red', linestyle='--', linewidth=0.8, label='−20 threshold')
    ax2.set_ylabel('log₁₀(rel. error)')
    ax2.set_xlabel('g (grouped by delta)')
    ax2.set_title('Panel B: Error distribution\nred=collapse expected')
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3, axis='y')

    # Panel C: 素数g_blockとerror mean
    ax3 = fig.add_subplot(gs_layout[1, 0])
    gs_vals = [r['g'] for r in res2]
    means = [r['mean'] for r in res2]
    primes = [r['prime'] for r in res2]
    bar_colors = ['tomato' if p else 'steelblue' for p in primes]
    ax3.bar(gs_vals, means, color=bar_colors, alpha=0.7)
    ax3.set_xlabel('g')
    ax3.set_ylabel('mean log₁₀(error)')
    ax3.set_title('Panel C: Prime g_block effect\nred=prime g_block, blue=composite')
    ax3.grid(True, alpha=0.3, axis='y')
    ax3.legend(handles=[Patch(color='tomato', alpha=0.7, label='prime g_block'),
                        Patch(color='steelblue', alpha=0.7, label='composite g_block')])

    # Panel D: 素数g_blockとfrac<-20
    ax4 = fig.add_subplot(gs_layout[1, 1])
    fracs = [r['frac20'] for r in res2]
    ax4.bar(gs_vals, fracs, color=bar_colors, alpha=0.7)
    ax4.set_xlabel('g')
    ax4.set_ylabel('% cases with error < 10⁻²⁰')
    ax4.set_title('Panel D: Extreme precision fraction\nvs prime g_block')
    ax4.grid(True, alpha=0.3, axis='y')

    # Panel E: sin展開 C_theory vs C_measured
    ax5 = fig.add_subplot(gs_layout[1, 2])
    if res3:
        ct_vals = [r['C_theory'] for r in res3]
        cm_vals = [r['C_measured'] for r in res3]
        ax5.scatter(cm_vals, ct_vals, alpha=0.5, s=20)
        mn, mx = min(cm_vals+ct_vals), max(cm_vals+ct_vals)
        ax5.plot([mn, mx], [mn, mx], 'r--', linewidth=1, label='y=x (perfect)')
        ax5.set_xscale('log'); ax5.set_yscale('log')
        ax5.set_xlabel('C measured')
        ax5.set_ylabel('C theory (sin expansion)')
        ax5.set_title('Panel E: sin-expansion prediction\nvs measured C(|z|)')
        ax5.legend()
        ax5.grid(True, alpha=0.3)

    plt.suptitle('Open Problems — Numerical Exploration', fontsize=13, y=1.01)
    plt.savefig("open_problems.png", dpi=130, bbox_inches='tight')
    print("\nプロット: open_problems.png")
    print("完了")


if __name__ == "__main__":
    main()
