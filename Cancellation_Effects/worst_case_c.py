"""
Worst-case C の数値的構成
目的: ratio = |LHS| / precise_bound が 1.0 に近づく C を構成する

理論:
  LHS = 4π · Σ_{n in half-lattice} Re(n_Aᵀ C n_B) · cos(2πnᵀz) · exp(-πnᵀIm(Ω)n)
  UB  = 4π · Σ_{n in half-lattice} |Re(n_Aᵀ C n_B)| · exp(-πnᵀIm(Ω)n)

  ratio = LHS / UB を最大化するには:
    Re(n_Aᵀ C n_B) と cos(2πnᵀz) が全格子点で同符号になるべき

  最適 C の構成:
    C_opt = Σ_{n in half-lattice} sign(cos(2πnᵀz)) · w_n · n_A n_Bᵀ
    ただし w_n = exp(-πnᵀIm(Ω)n) / ||C_opt||_F（正規化）

  これは「cosの符号に合わせてCを調整する」という構成

検証方法:
  (1) random C での ratio（ベースライン）
  (2) cos-aligned C での ratio（理論最適）
  (3) 数値最適化による C（勾配上昇法）
  (4) 三者を比較

実行: python worst_case_c.py
"""

import numpy as np
from itertools import product
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.optimize import minimize

# ============================================================
# 基本関数
# ============================================================

def make_omega_blocks(g, lmin, re_scale=0.0, seed=42):
    """Re(Ω)=0で純粋にC/zの効果を見る"""
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
    ImOm = np.block([[B1, np.zeros((g1, g2))],
                     [np.zeros((g2, g1)), B2]])
    return 1j * ImOm, g1, g2


def get_half_lattice_terms(Om0, z, N_cut, g1):
    """
    半格子点ごとの (n_A, n_B, exp_val, cos_val) を返す
    これがLHSと上界の計算の基本単位
    """
    g = Om0.shape[0]
    terms = []
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
        cos_val = np.cos(2 * np.pi * nv @ z)
        terms.append((nA, nB, exp_val, cos_val))
    return terms


def compute_ratio(C, terms):
    """
    C（g1×g2行列）に対してratio = |LHS|/UBを計算
    """
    lhs_sum = 0.0
    ub_sum = 0.0
    for nA, nB, exp_val, cos_val in terms:
        re_nCn = np.real(nA @ C @ nB)
        lhs_sum += re_nCn * cos_val * exp_val
        ub_sum += abs(re_nCn) * exp_val

    lhs = abs(4 * np.pi * lhs_sum)
    ub = 4 * np.pi * ub_sum
    return lhs / (ub + 1e-30), lhs, ub


def make_random_C(g1, g2, seed=0):
    rng = np.random.default_rng(seed)
    C = rng.standard_normal((g1, g2)) + 1j * rng.standard_normal((g1, g2))
    C /= np.linalg.norm(C, 'fro')
    return C


# ============================================================
# 方法1: cos-aligned C（解析的構成）
# ============================================================

def make_cos_aligned_C(terms, g1, g2, z):
    """
    Re(n_Aᵀ C n_B) が cos(2πnᵀz) と同符号になるよう C を構成
    
    理論: LHS = Σ Re(n_Aᵀ C n_B) · cos · exp
    これを最大化するには Re(n_Aᵀ C n_B) = |something| · sign(cos) が必要
    
    構成: C = Σ_{n} sign(cos_n) · w_n · n_A ⊗ n_B^T
    ただし w_n = exp_val（重み）
    """
    C = np.zeros((g1, g2), dtype=complex)
    for nA, nB, exp_val, cos_val in terms:
        if abs(cos_val) < 1e-10:
            continue
        sign_cos = np.sign(cos_val)
        # n_A n_B^T の外積
        C += sign_cos * exp_val * np.outer(nA, nB)

    # 正規化
    norm = np.linalg.norm(C, 'fro')
    if norm > 1e-10:
        C /= norm
    return C


# ============================================================
# 方法2: 数値最適化（勾配上昇法）
# ============================================================

def optimize_C_gradient(terms, g1, g2, n_restarts=5, n_iter=200, lr=0.05):
    """
    ratio を最大化する C を勾配上昇法で探索
    Cは実数行列に制限（虚部はLHSに寄与しない）
    """
    best_ratio = 0.0
    best_C = None

    for restart in range(n_restarts):
        rng = np.random.default_rng(restart * 100)
        # 初期化: cos-alignedをベースにノイズ
        C_aligned = make_cos_aligned_C(terms, g1, g2, None)
        C = C_aligned.real + rng.standard_normal((g1, g2)) * 0.1
        C /= np.linalg.norm(C, 'fro')

        ratio_history = []
        for it in range(n_iter):
            # 数値勾配
            eps = 1e-5
            grad = np.zeros_like(C)
            ratio_cur, _, _ = compute_ratio(C, terms)
            for i in range(g1):
                for j in range(g2):
                    C_p = C.copy(); C_p[i, j] += eps
                    C_p /= np.linalg.norm(C_p, 'fro')
                    r_p, _, _ = compute_ratio(C_p, terms)
                    grad[i, j] = (r_p - ratio_cur) / eps

            # 勾配上昇
            C = C + lr * grad
            C /= np.linalg.norm(C, 'fro')

            ratio_cur, _, _ = compute_ratio(C, terms)
            ratio_history.append(ratio_cur)

            if it > 20 and abs(ratio_history[-1] - ratio_history[-10]) < 1e-6:
                break

        final_ratio, _, _ = compute_ratio(C, terms)
        if final_ratio > best_ratio:
            best_ratio = final_ratio
            best_C = C.copy()

    return best_C, best_ratio


# ============================================================
# 方法3: 解析的最適解（SVD）
# ============================================================

def make_svd_optimal_C(terms, g1, g2):
    """
    |LHS| を最大化する C の解析的最適解
    
    LHS = Re(Σ_n n_Aᵀ C n_B · cos_n · exp_n)
        = Re(Tr(Cᵀ · M))    ただし M = Σ_n n_A cos_n exp_n n_Bᵀ
    
    これを最大化する ||C||_F=1 の C は:
    C_opt = M / ||M||_F  （実部のみ取る場合）
    
    正確には:
    max Re(Tr(Cᵀ M)) = ||M||_F  （||C||_F=1の制約下）
    達成するのは C = M^* / ||M||_F
    """
    M = np.zeros((g1, g2), dtype=complex)
    for nA, nB, exp_val, cos_val in terms:
        M += cos_val * exp_val * np.outer(nA, nB)

    norm_M = np.linalg.norm(M, 'fro')
    if norm_M < 1e-10:
        return np.zeros((g1, g2)), 0.0, 0.0, 0.0

    # C_opt = M.conj() / ||M||_F
    C_opt = M.conj() / norm_M

    # このCでのratioを計算
    ratio, lhs, ub = compute_ratio(C_opt, terms)

    # 理論的最大LHS = 4π · ||M||_F
    theoretical_max_lhs = 4 * np.pi * norm_M

    return C_opt, ratio, theoretical_max_lhs, ub


# ============================================================
# メイン解析
# ============================================================

def analyze_single_condition(Om0, g1, g2, z, N_cut, n_random=20, label=""):
    terms = get_half_lattice_terms(Om0, z, N_cut, g1)
    g2_actual = Om0.shape[0] - g1

    # Random C のratio分布
    random_ratios = []
    for seed in range(n_random):
        C_rand = make_random_C(g1, g2_actual, seed)
        r, _, _ = compute_ratio(C_rand, terms)
        random_ratios.append(r)

    # cos-aligned C
    C_aligned = make_cos_aligned_C(terms, g1, g2_actual, z)
    ratio_aligned, lhs_aligned, ub_aligned = compute_ratio(C_aligned, terms)

    # SVD最適C
    C_svd, ratio_svd, theoretical_max, ub_svd = make_svd_optimal_C(terms, g1, g2_actual)

    # 数値最適化C（小規模のみ）
    if g1 * g2_actual <= 16:
        C_opt, ratio_opt = optimize_C_gradient(terms, g1, g2_actual, n_restarts=3)
    else:
        ratio_opt = None

    result = {
        'label': label,
        'random_mean': np.mean(random_ratios),
        'random_max': np.max(random_ratios),
        'random_ratios': random_ratios,
        'ratio_aligned': ratio_aligned,
        'ratio_svd': ratio_svd,
        'ratio_opt': ratio_opt,
        'theoretical_max_lhs': theoretical_max,
        'ub_svd': ub_svd,
        'ratio_theoretical': theoretical_max / (ub_svd + 1e-30),
        'n_terms': len(terms),
    }
    return result


def main():
    print("=" * 65)
    print("Worst-case C の数値的構成")
    print("ratio = |LHS| / precise_bound の最大化")
    print("=" * 65)

    N_cut = 1
    g = 8
    lmin = 3.0

    # --- 複数のz設定で検証 ---
    rng_z = np.random.default_rng(42)
    z_settings = [
        ('z=0',      np.zeros(g, dtype=float)),
        ('z=1/4·1',  np.ones(g, dtype=float) * 0.25),
        ('z=generic', rng_z.standard_normal(g) * 0.3),
        ('z=1/2·1',  np.ones(g, dtype=float) * 0.5),
    ]

    print(f"\n{'z種類':15} {'rand_mean':>10} {'rand_max':>10} "
          f"{'aligned':>10} {'SVD':>10} {'opt':>10} {'理論比':>10}")
    print("-" * 75)

    all_results = []
    Om0, g1, g2 = make_omega_blocks(g, lmin, re_scale=0.0, seed=0)

    for z_label, z in z_settings:
        res = analyze_single_condition(Om0, g1, g2, z, N_cut, n_random=30, label=z_label)
        opt_str = f"{res['ratio_opt']:.4f}" if res['ratio_opt'] is not None else "N/A"
        print(f"{z_label:15} {res['random_mean']:10.4f} {res['random_max']:10.4f} "
              f"{res['ratio_aligned']:10.4f} {res['ratio_svd']:10.4f} "
              f"{opt_str:>10} {res['ratio_theoretical']:10.4f}")
        all_results.append(res)

    # --- λ_min 依存性 ---
    print("\n[SVD最適C のratio vs λ_min] (g=8, z=0, z=generic)")
    print(f"{'lmin':>6} {'SVD(z=0)':>12} {'SVD(generic)':>14} {'rand_mean(z=0)':>16}")
    print("-" * 52)

    lmin_vals = [0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0]
    lmin_results = []
    for lmin_v in lmin_vals:
        Om_v, g1_v, g2_v = make_omega_blocks(g, lmin_v, re_scale=0.0, seed=0)
        z0 = np.zeros(g)
        zg = np.random.default_rng(42).standard_normal(g) * 0.3

        terms0 = get_half_lattice_terms(Om_v, z0, N_cut, g1_v)
        termsg = get_half_lattice_terms(Om_v, zg, N_cut, g1_v)

        _, r0, _, ub0 = make_svd_optimal_C(terms0, g1_v, g2_v)
        _, rg, _, ubg = make_svd_optimal_C(termsg, g1_v, g2_v)

        rand_ratios = [compute_ratio(make_random_C(g1_v, g2_v, s), terms0)[0]
                       for s in range(10)]
        rand_mean = np.mean(rand_ratios)

        print(f"{lmin_v:6.1f} {r0:12.4f} {rg:14.4f} {rand_mean:16.4f}")
        lmin_results.append({'lmin': lmin_v, 'svd_z0': r0, 'svd_gen': rg, 'rand_z0': rand_mean})

    # ---- プロット ----
    fig = plt.figure(figsize=(14, 10))
    gs_layout = gridspec.GridSpec(2, 3, fig, hspace=0.45, wspace=0.35)

    # Panel A: zカテゴリ別の各C構成法のratio
    ax = fig.add_subplot(gs_layout[0, :2])
    x = np.arange(len(all_results))
    w = 0.2
    labels = [r['label'] for r in all_results]
    ax.bar(x - 1.5*w, [r['random_mean'] for r in all_results], w,
           label='random C (mean)', color='steelblue', alpha=0.7)
    ax.bar(x - 0.5*w, [r['random_max'] for r in all_results], w,
           label='random C (max)', color='steelblue', alpha=0.4)
    ax.bar(x + 0.5*w, [r['ratio_aligned'] for r in all_results], w,
           label='cos-aligned C', color='tomato', alpha=0.7)
    ax.bar(x + 1.5*w, [r['ratio_svd'] for r in all_results], w,
           label='SVD optimal C', color='green', alpha=0.7)
    ax.axhline(1.0, color='red', linestyle='--', linewidth=1, label='bound=1.0')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel('ratio = |LHS| / precise_bound')
    ax.set_title('Panel A: ratio by C construction method\n(g=8, lmin=3.0)')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, axis='y')

    # Panel B: random C のratio分布（z=0 vs generic）
    ax2 = fig.add_subplot(gs_layout[0, 2])
    res_z0  = all_results[0]  # z=0
    res_gen = all_results[2]  # generic
    ax2.hist(res_z0['random_ratios'], bins=15, alpha=0.6,
             color='tomato', label=f"z=0 (mean={res_z0['random_mean']:.3f})", density=True)
    ax2.hist(res_gen['random_ratios'], bins=15, alpha=0.6,
             color='steelblue', label=f"generic (mean={res_gen['random_mean']:.3f})", density=True)
    ax2.axvline(res_z0['ratio_svd'], color='tomato', linestyle='--',
                label=f'SVD(z=0)={res_z0["ratio_svd"]:.3f}')
    ax2.axvline(res_gen['ratio_svd'], color='steelblue', linestyle='--',
                label=f'SVD(gen)={res_gen["ratio_svd"]:.3f}')
    ax2.axvline(1.0, color='red', linewidth=2, label='bound=1.0')
    ax2.set_xlabel('ratio')
    ax2.set_ylabel('density')
    ax2.set_title('Panel B: random C ratio distribution\n(z=0 vs generic)')
    ax2.legend(fontsize=7)
    ax2.grid(True, alpha=0.3)

    # Panel C: SVD最適C のratio vs λ_min
    ax3 = fig.add_subplot(gs_layout[1, 0])
    lmins = [r['lmin'] for r in lmin_results]
    ax3.plot(lmins, [r['svd_z0'] for r in lmin_results], 'o-',
             color='tomato', label='SVD opt (z=0)')
    ax3.plot(lmins, [r['svd_gen'] for r in lmin_results], 's-',
             color='steelblue', label='SVD opt (generic z)')
    ax3.plot(lmins, [r['rand_z0'] for r in lmin_results], '^--',
             color='gray', alpha=0.6, label='random C mean (z=0)')
    ax3.axhline(1.0, color='red', linestyle='--', linewidth=0.8)
    ax3.set_xlabel(r'$\lambda_\min$')
    ax3.set_ylabel('ratio')
    ax3.set_title('Panel C: SVD optimal C ratio vs λ_min')
    ax3.legend(fontsize=8)
    ax3.grid(True, alpha=0.3)

    # Panel D: 理論比 = theoretical_max_lhs / UB
    ax4 = fig.add_subplot(gs_layout[1, 1])
    theory_ratios = [r['ratio_theoretical'] for r in all_results]
    ax4.bar(range(len(all_results)), theory_ratios,
            color=['tomato','orange','steelblue','green'], alpha=0.7)
    ax4.axhline(1.0, color='red', linestyle='--', linewidth=1, label='bound=1.0')
    ax4.set_xticks(range(len(all_results)))
    ax4.set_xticklabels([r['label'] for r in all_results])
    ax4.set_ylabel('4π||M||_F / precise_bound')
    ax4.set_title('Panel D: Theoretical maximum ratio\n= 4π||M||_F / UB')
    ax4.legend()
    ax4.grid(True, alpha=0.3, axis='y')

    # Panel E: SVD ratio の意味（理論的tight性）
    ax5 = fig.add_subplot(gs_layout[1, 2])
    # SVD ratio が1.0に達するかどうかをg依存性で確認
    for z_label, z_vec, color in [
        ('z=0', np.zeros(g), 'tomato'),
        ('z=1/4', np.ones(g)*0.25, 'orange'),
        ('z=gen', np.random.default_rng(42).standard_normal(int(g))*0.3, 'steelblue')
    ]:
        svd_ratios = []
        for lmin_v in lmin_vals:
            Om_v, g1_v, g2_v = make_omega_blocks(g, lmin_v, re_scale=0.0, seed=0)
            terms_v = get_half_lattice_terms(Om_v, z_vec, N_cut, g1_v)
            _, r_v, _, _ = make_svd_optimal_C(terms_v, g1_v, g2_v)
            svd_ratios.append(r_v)
        ax5.plot(lmin_vals, svd_ratios, 'o-', color=color, label=z_label)
    ax5.axhline(1.0, color='red', linestyle='--', linewidth=0.8, label='bound=1.0')
    ax5.set_xlabel(r'$\lambda_\min$')
    ax5.set_ylabel('SVD optimal ratio')
    ax5.set_title('Panel E: SVD optimal ratio vs λ_min\n(upper bound tightness)')
    ax5.legend(fontsize=8)
    ax5.grid(True, alpha=0.3)

    plt.suptitle('Worst-case C analysis: how tight is the δ-linear bound?',
                 fontsize=12, y=1.01)
    plt.tight_layout()
    plt.savefig("worst_case_c.png", dpi=130, bbox_inches='tight')
    print("\nプロット: worst_case_c.png")
    print("完了")


if __name__ == "__main__":
    main()
