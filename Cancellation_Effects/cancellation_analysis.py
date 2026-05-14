"""
Anomalous Cancellation 深堀り解析
目的: S(2,2)誤差がCLT上界より15桁以上小さい残差の起源を特定する

実行: python cancellation_analysis.py
出力: cancellation_results.txt + cancellation_plots.png
"""

import numpy as np
from itertools import product
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# ============================================================
# 基本関数
# ============================================================

def make_omega(g, lmin, offnorm=0.3, seed=42):
    """
    ランダムな正定値周期行列を生成
    lmin: Im(Omega)の最小固有値の目標値
    """
    rng = np.random.default_rng(seed)
    gb = g // 2

    # 対角ブロック: 固有値をlmin以上に制御
    def make_block(n):
        A = rng.standard_normal((n, n))
        Q, _ = np.linalg.qr(A)
        eigs = lmin + rng.exponential(0.5, n)
        return Q @ np.diag(eigs) @ Q.T

    B1 = make_block(gb)
    B2 = make_block(gb)
    # off-diagonal実部
    C = rng.standard_normal((gb, gb)) * offnorm * 0.3
    C = (C + C.T) / 2  # 対称化

    ImOm = np.block([[B1, C], [C.T, B2]])
    ReOm = rng.standard_normal((g, g)) * 0.1
    ReOm = (ReOm + ReOm.T) / 2

    # 正定値確認
    eigs = np.linalg.eigvalsh(ImOm)
    if eigs.min() < 0.01:
        ImOm += (0.01 - eigs.min()) * np.eye(g)

    return ReOm + 1j * ImOm


def theta_naive(Omega, z, N_cut=1):
    """単純なtheta関数(格子和)"""
    g = Omega.shape[0]
    ns = range(-N_cut, N_cut + 1)
    result = 0.0 + 0j
    for n in product(ns, repeat=g):
        n = np.array(n, dtype=float)
        phase = np.pi * 1j * (n @ Omega @ n + 2 * n @ z)
        result += np.exp(phase)
    return result


def theta_block(Om_block, z_block, N_cut=1):
    """ブロックのtheta関数"""
    return theta_naive(Om_block, z_block, N_cut)


def theta_s22(Omega, z, N_cut=1):
    """S(2,2)分解によるtheta関数"""
    g = Omega.shape[0]
    gb = g // 2
    Om1 = Omega[:gb, :gb]
    Om2 = Omega[gb:, gb:]
    z1 = z[:gb]
    z2 = z[gb:]
    return theta_block(Om1, z1, N_cut) * theta_block(Om2, z2, N_cut)


def grad_theta(Om_block, z_block, N_cut=1, h=1e-6):
    """theta関数の数値勾配"""
    n = len(z_block)
    grad = np.zeros(n, dtype=complex)
    for j in range(n):
        zp = z_block.copy(); zp[j] += h
        zm = z_block.copy(); zm[j] -= h
        grad[j] = (theta_block(Om_block, zp, N_cut) -
                   theta_block(Om_block, zm, N_cut)) / (2 * h)
    return grad


def compute_moments(Omega, z, E, N_cut=2, n_eps=20):
    """
    epsilon展開の各モーメントを数値的に計算
    theta_naive(Omega + eps*E, z) をepsで展開
    M_m = d^m/d(eps)^m [theta(Omega + eps*E, z)] at eps=0
    """
    eps_vals = np.linspace(-0.05, 0.05, n_eps)
    theta_vals = []
    for eps in eps_vals:
        theta_vals.append(theta_naive(Omega + eps * E, z, N_cut))
    theta_vals = np.array(theta_vals)

    # 多項式フィット
    coeffs = np.polyfit(eps_vals, theta_vals.real, 4)
    coeffs_i = np.polyfit(eps_vals, theta_vals.imag, 4)

    # M_m = m! * coeff[degree m]
    # poly degree 4: coeffs[0]*eps^4 + ... + coeffs[4]
    moments = []
    from math import factorial
    for m in range(5):
        idx = 4 - m  # 最高次から
        Mm = (coeffs[idx] + 1j * coeffs_i[idx]) * factorial(m)
        moments.append(Mm)
    return moments


# ============================================================
# 解析1: mu_A/Theta_A の λ_min 依存性
# ============================================================

def analysis_mu_ratio(lmin_vals, g=8, offnorm=0.3, N_cut=1, n_seeds=5):
    """
    mu_A/|Theta_A| の大きさを λ_min の関数として測定
    これがなぜ小さいかの起源を探る
    """
    results = {lmin: [] for lmin in lmin_vals}

    for lmin in lmin_vals:
        for seed in range(n_seeds):
            Omega = make_omega(g, lmin, offnorm, seed)
            gb = g // 2
            Om1 = Omega[:gb, :gb]
            rng = np.random.default_rng(seed + 100)
            z = rng.standard_normal(g) * 0.3

            ThetaA = theta_block(Om1, z[:gb], N_cut)
            if abs(ThetaA) < 1e-30:
                continue

            gradA = grad_theta(Om1, z[:gb], N_cut)
            mu_A = gradA / (2 * np.pi * 1j)
            ratio = np.linalg.norm(mu_A) / abs(ThetaA)
            results[lmin].append(ratio)

    return results


# ============================================================
# 解析2: M_1因数分解の精度確認とM_2以降の寄与
# ============================================================

def analysis_moment_structure(lmin_vals, g=8, offnorm=0.3, N_cut=1, n_seeds=3):
    """
    各モーメントM_mの大きさと因数分解精度を測定
    """
    results = []

    for lmin in lmin_vals:
        for seed in range(n_seeds):
            Omega = make_omega(g, lmin, offnorm, seed)
            gb = g // 2
            Om1 = Omega[:gb, :gb]
            Om2 = Omega[gb:, gb:]
            off = Omega[:gb, gb:]

            rng = np.random.default_rng(seed + 200)
            z = rng.standard_normal(g) * 0.3

            # off-diagonal成分のみのE
            E = np.zeros((g, g), dtype=complex)
            E[:gb, gb:] = off
            E[gb:, :gb] = off.T

            # モーメント計算
            moments = compute_moments(Omega - E, z, E, N_cut=N_cut)

            # 因数分解確認: M_1 = 2 mu_A^T C mu_B
            ThetaA = theta_block(Om1, z[:gb], N_cut)
            ThetaB = theta_block(Om2, z[gb:], N_cut)
            gradA = grad_theta(Om1, z[:gb], N_cut)
            gradB = grad_theta(Om2, z[gb:], N_cut)
            mu_A = gradA / (2 * np.pi * 1j)
            mu_B = gradB / (2 * np.pi * 1j)

            M1_factored = 2 * mu_A @ off @ mu_B
            M1_direct = moments[1]
            fact_error = abs(M1_direct - M1_factored) / (abs(M1_direct) + 1e-30)

            # λ_min実測値
            lmin_actual = np.linalg.eigvalsh(Omega.imag).min()

            # CLTベース上界: |theta| * sqrt(格子点数) * max|term|
            # 簡易版: exp(-pi * lmin) スケール
            clt_bound = abs(ThetaA * ThetaB) * np.exp(-np.pi * lmin_actual)

            results.append({
                'lmin_target': lmin,
                'lmin_actual': lmin_actual,
                'seed': seed,
                '|M0|': abs(moments[0]),
                '|M1|': abs(moments[1]),
                '|M2|': abs(moments[2]),
                '|M3|': abs(moments[3]),
                '|M1_factored|': abs(M1_factored),
                'factorization_error': fact_error,
                '|ThetaA*ThetaB|': abs(ThetaA * ThetaB),
                '|mu_A|/|ThetaA|': np.linalg.norm(mu_A) / (abs(ThetaA) + 1e-30),
                '|mu_B|/|ThetaB|': np.linalg.norm(mu_B) / (abs(ThetaB) + 1e-30),
                'M1/M0': abs(moments[1]) / (abs(moments[0]) + 1e-30),
                'M2/M0': abs(moments[2]) / (abs(moments[0]) + 1e-30),
                'exp(-pi*lmin)': np.exp(-np.pi * lmin_actual),
                'exp(-2pi*lmin)': np.exp(-2 * np.pi * lmin_actual),
            })

    return results


# ============================================================
# 解析3: k=2 vs k=3 の精度差の機構
# ============================================================

def analysis_k_comparison(lmin_vals, g=12, offnorm=0.1, N_cut=1, n_seeds=3):
    """
    k=2 と k=3 の誤差比較。k=2のZ_2対称性が効いているか確認。
    g=12 (= 4*3 = 6*2 なので両方可能)
    """
    results = []

    for lmin in lmin_vals:
        for seed in range(n_seeds):
            Omega = make_omega(g, lmin, offnorm, seed)
            rng = np.random.default_rng(seed + 300)
            z = rng.standard_normal(g) * 0.3

            # 真値 (N_cut=2でnaive)
            theta_true = theta_naive(Omega, z, N_cut=2)

            # k=2: S(6,6)
            gb2 = g // 2  # 6
            theta_k2 = (theta_block(Omega[:gb2, :gb2], z[:gb2], N_cut) *
                        theta_block(Omega[gb2:, gb2:], z[gb2:], N_cut))
            err_k2 = abs(theta_true - theta_k2) / (abs(theta_true) + 1e-30)

            # k=3: S(4,4,4)
            gb3 = g // 3  # 4
            theta_k3 = (theta_block(Omega[:gb3, :gb3], z[:gb3], N_cut) *
                        theta_block(Omega[gb3:2*gb3, gb3:2*gb3], z[gb3:2*gb3], N_cut) *
                        theta_block(Omega[2*gb3:, 2*gb3:], z[2*gb3:], N_cut))
            err_k3 = abs(theta_true - theta_k3) / (abs(theta_true) + 1e-30)

            lmin_actual = np.linalg.eigvalsh(Omega.imag).min()

            # k=2のZ_2対称性確認:
            # off-diag摂動E_{12}のみの場合, theta_s22はεに不変か?
            off12 = Omega[:gb2, gb2:]
            E = np.zeros((g, g), dtype=complex)
            E[:gb2, gb2:] = off12
            E[gb2:, :gb2] = off12.T
            Om0 = Omega - E
            theta_s22_0 = (theta_block(Om0[:gb2, :gb2], z[:gb2], N_cut) *
                           theta_block(Om0[gb2:, gb2:], z[gb2:], N_cut))
            theta_s22_eps = (theta_block((Om0 + 0.01*E)[:gb2, :gb2], z[:gb2], N_cut) *
                             theta_block((Om0 + 0.01*E)[gb2:, gb2:], z[gb2:], N_cut))
            invariance_k2 = abs(theta_s22_eps - theta_s22_0) / (abs(theta_s22_0) + 1e-30)

            results.append({
                'lmin_target': lmin,
                'lmin_actual': lmin_actual,
                'seed': seed,
                'err_k2': err_k2,
                'err_k3': err_k3,
                'ratio_k3_k2': err_k3 / (err_k2 + 1e-30),
                'Z2_invariance_k2': invariance_k2,
            })

    return results


# ============================================================
# メイン実行
# ============================================================

def main():
    print("=" * 60)
    print("Anomalous Cancellation 深堀り解析")
    print("=" * 60)

    lmin_vals = [0.5, 1.0, 2.0, 3.0, 4.0, 5.0]
    out_lines = []

    # --- 解析1: mu_A/Theta_A の λ_min 依存性 ---
    print("\n[解析1] mu_A/|Theta_A| の λ_min 依存性 (g=8)")
    out_lines.append("=== 解析1: mu_A/|Theta_A| vs lambda_min ===")
    mu_results = analysis_mu_ratio(lmin_vals, g=8, n_seeds=5)
    for lmin, ratios in mu_results.items():
        if ratios:
            mean_r = np.mean(ratios)
            std_r = np.std(ratios)
            exp_pred = np.exp(-np.pi * lmin) ** 0.5  # 予想スケール
            line = (f"  lmin={lmin:.1f}: |mu_A|/|ThetaA| = {mean_r:.3e} ± {std_r:.1e}"
                    f"  (exp(-pi*lmin/2)={exp_pred:.3e})")
            print(line)
            out_lines.append(line)

    # --- 解析2: モーメント構造 ---
    print("\n[解析2] モーメント構造 M_m の λ_min 依存性 (g=8)")
    out_lines.append("\n=== 解析2: モーメント構造 ===")
    moment_results = analysis_moment_structure([1.0, 2.0, 3.5, 5.0], g=8, n_seeds=3)

    out_lines.append(f"{'lmin':>6} {'|M1/M0|':>12} {'|M2/M0|':>12} "
                     f"{'exp(-2pi*l)':>12} {'fact_err':>10} {'|mu_A|/|ThA|':>13}")
    out_lines.append("-" * 70)
    for r in moment_results:
        line = (f"{r['lmin_actual']:6.2f} "
                f"{r['M1/M0']:12.3e} "
                f"{r['M2/M0']:12.3e} "
                f"{r['exp(-2pi*lmin)']:12.3e} "
                f"{r['factorization_error']:10.3e} "
                f"{r['|mu_A|/|ThetaA|']:13.3e}")
        print("  " + line)
        out_lines.append(line)

    # 重要比較: |M1/M0| vs exp(-2pi*lmin)
    out_lines.append("\n--- キー比較: |M1/M0| / exp(-2pi*lmin) ---")
    print("\n  キー比較: |M1/M0| / exp(-2pi*lmin) (= 1なら理論通り)")
    for r in moment_results:
        ratio = r['M1/M0'] / (r['exp(-2pi*lmin)'] + 1e-100)
        line = f"  lmin={r['lmin_actual']:.2f}: ratio = {ratio:.3e}"
        print(line)
        out_lines.append(line)

    # --- 解析3: k=2 vs k=3 ---
    print("\n[解析3] k=2 vs k=3 誤差比 (g=12, offnorm=0.1)")
    out_lines.append("\n=== 解析3: k=2 vs k=3 比較 ===")
    k_results = analysis_k_comparison([2.0, 3.5, 5.0], g=12, n_seeds=3)

    out_lines.append(f"{'lmin':>6} {'err_k2':>12} {'err_k3':>12} "
                     f"{'ratio k3/k2':>12} {'Z2_invar':>10}")
    out_lines.append("-" * 55)
    for r in k_results:
        line = (f"{r['lmin_actual']:6.2f} "
                f"{r['err_k2']:12.3e} "
                f"{r['err_k3']:12.3e} "
                f"{r['ratio_k3_k2']:12.1f}x "
                f"{r['Z2_invariance_k2']:10.3e}")
        print("  " + line)
        out_lines.append(line)

    # --- 結果保存 ---
    with open("cancellation_results.txt", "w") as f:
        f.write("\n".join(out_lines))
    print("\n結果を cancellation_results.txt に保存")

    # --- プロット ---
    fig = plt.figure(figsize=(14, 10))
    gs = gridspec.GridSpec(2, 2, fig, hspace=0.4, wspace=0.35)

    # Panel A: mu_A/ThetaA vs lmin
    ax = fig.add_subplot(gs[0, 0])
    lmins_plot = list(mu_results.keys())
    means = [np.mean(v) if v else np.nan for v in mu_results.values()]
    stds  = [np.std(v)  if v else np.nan for v in mu_results.values()]
    ax.errorbar(lmins_plot, means, yerr=stds, fmt='o-', color='steelblue', label='measured')
    ax.plot(lmins_plot, [np.exp(-np.pi * l / 2) for l in lmins_plot],
            '--', color='orange', label=r'$e^{-\pi\lambda/2}$')
    ax.set_yscale('log')
    ax.set_xlabel(r'$\lambda_\min$')
    ax.set_ylabel(r'$|\mu_A| / |\Theta_A|$')
    ax.set_title('Panel A: Gradient/Value ratio vs λ_min\n(g=8, offnorm=0.3)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Panel B: M1/M0 と exp(-2pi*lmin) の比較
    ax2 = fig.add_subplot(gs[0, 1])
    lmins_m = [r['lmin_actual'] for r in moment_results]
    M1M0    = [r['M1/M0']       for r in moment_results]
    M2M0    = [r['M2/M0']       for r in moment_results]
    exp2    = [r['exp(-2pi*lmin)'] for r in moment_results]
    ax2.scatter(lmins_m, M1M0, marker='o', color='steelblue', label=r'$|M_1/M_0|$')
    ax2.scatter(lmins_m, M2M0, marker='s', color='green',     label=r'$|M_2/M_0|$')
    ax2.plot(sorted(set(lmins_m)),
             [np.exp(-2*np.pi*l) for l in sorted(set(lmins_m))],
             '--', color='red', label=r'$e^{-2\pi\lambda}$')
    ax2.set_yscale('log')
    ax2.set_xlabel(r'$\lambda_\min$')
    ax2.set_ylabel('relative magnitude')
    ax2.set_title('Panel B: Moment magnitudes vs λ_min\n(g=8)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Panel C: k=2 vs k=3 誤差
    ax3 = fig.add_subplot(gs[1, 0])
    lmins_k = [r['lmin_actual'] for r in k_results]
    err_k2  = [r['err_k2'] for r in k_results]
    err_k3  = [r['err_k3'] for r in k_results]
    ax3.scatter(lmins_k, err_k2, marker='o', color='steelblue', label='k=2')
    ax3.scatter(lmins_k, err_k3, marker='^', color='tomato',    label='k=3')
    ax3.set_yscale('log')
    ax3.set_xlabel(r'$\lambda_\min$')
    ax3.set_ylabel('relative error')
    ax3.set_title('Panel C: k=2 vs k=3 error\n(g=12, offnorm=0.1)')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # Panel D: k=3/k=2 誤差比
    ax4 = fig.add_subplot(gs[1, 1])
    ratios = [r['ratio_k3_k2'] for r in k_results]
    ax4.scatter(lmins_k, ratios, marker='D', color='purple')
    ax4.axhline(22000, color='red', linestyle='--', label='observed ~22000x')
    ax4.set_yscale('log')
    ax4.set_xlabel(r'$\lambda_\min$')
    ax4.set_ylabel('err(k=3) / err(k=2)')
    ax4.set_title('Panel D: k=3/k=2 error ratio\n(expected ~22000x)')
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    plt.suptitle('Anomalous Cancellation — Structure Analysis', fontsize=13, y=1.01)
    plt.savefig("cancellation_plots.png", dpi=130, bbox_inches='tight')
    print("プロットを cancellation_plots.png に保存")
    print("\n完了。次のステップは cancellation_results.txt を確認して議論。")


if __name__ == "__main__":
    main()
