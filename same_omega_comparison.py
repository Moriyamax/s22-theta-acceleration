"""
same_omega_comparison.py

「同一Ω比較」: naive vs s22 を同一の Ω, z で評価し、
値の一致（相対誤差）と計算時間を比較する。

対象: g=8, 9  (N_cut=1, 2)
S_{(2,2)} 上の点 (ε_{12}=0 が厳密に成立する領域) と
S_{(2,2)} から外れた点 (ε_{12}≠0) の両方をテスト。

実行方法:
    python same_omega_comparison.py

依存: numpy のみ (scipy は任意)
"""

import time
import numpy as np

# ─────────────────────────────────────────────
# 1. Theta 関数の実装
# ─────────────────────────────────────────────

def theta_naive(z: np.ndarray, Omega: np.ndarray, N_cut: int) -> complex:
    """
    g 次元 Siegel theta 関数の打ち切り級数 (naive).

    θ(z|Ω) = Σ_{n ∈ Z^g, |n_i|≤N_cut} exp(iπ n^T Ω n + 2πi n^T z)

    Parameters
    ----------
    z     : shape (g,) complex vector
    Omega : shape (g,g) complex symmetric matrix, Im(Ω) > 0
    N_cut : 各次元の打ち切り幅

    Returns
    -------
    complex scalar
    """
    g = len(z)
    ns = np.arange(-N_cut, N_cut + 1)  # length 2*N_cut+1

    # 全格子点を meshgrid で生成
    grids = np.meshgrid(*[ns] * g, indexing='ij')
    # shape: (2N+1)^g × g
    lattice = np.stack([gr.ravel() for gr in grids], axis=1).astype(float)

    # 指数部: iπ n^T Ω n + 2πi n^T z
    # n^T Ω n = einsum
    quad = np.einsum('ki,ij,kj->k', lattice, Omega, lattice)  # shape: M
    lin  = lattice @ z                                          # shape: M

    log_terms = 1j * np.pi * quad + 2j * np.pi * lin
    # オーバーフロー対策: 実部の最大値で引き算してから和
    log_terms_re = log_terms.real
    shift = log_terms_re.max()
    terms = np.exp(log_terms - shift)
    return np.exp(shift) * terms.sum()


def theta_s22(z: np.ndarray, Omega: np.ndarray, N_cut: int) -> complex:
    """
    S_{(2,2)} 分解による近似 theta 関数.

    Ω を (g1×g1), (g2×g2), off-diag ブロックに分解し、
    θ(z|Ω) ≈ θ(z1|Ω1) × θ(z2|Ω2)  (ε_{12}=off-diag ブロック を無視)

    g を半分に割る: g1 = g//2, g2 = g - g1
    """
    g = len(z)
    g1 = g // 2
    g2 = g - g1

    z1, z2      = z[:g1], z[g1:]
    Omega1      = Omega[:g1, :g1]
    Omega2      = Omega[g1:, g1:]

    val1 = theta_naive(z1, Omega1, N_cut)
    val2 = theta_naive(z2, Omega2, N_cut)
    return val1 * val2


# ─────────────────────────────────────────────
# 2. テスト用 Ω, z の生成
# ─────────────────────────────────────────────

def make_omega_on_S22(g: int, rng: np.random.Generator) -> np.ndarray:
    """
    S_{(2,2)} 上の周期行列: off-diag ブロック = 0 (ε_{12}=0 厳密)
    g1 = g//2, g2 = g - g1 の block-diagonal Ω.
    """
    g1 = g // 2
    g2 = g - g1
    # 各ブロックを正定値になるよう構成
    def make_block(size):
        A = rng.standard_normal((size, size)) * 0.3
        return 1j * (A @ A.T + np.eye(size) * 1.5)  # Im > 0

    Omega = np.zeros((g, g), dtype=complex)
    Omega[:g1, :g1] = make_block(g1)
    Omega[g1:, g1:] = make_block(g2)
    # 対称化
    Omega = (Omega + Omega.T) / 2
    return Omega


def make_omega_off_S22(g: int, delta: float, rng: np.random.Generator) -> np.ndarray:
    """
    S_{(2,2)} から delta だけ外れた周期行列.
    off-diag ブロックに delta * small_random を加える.
    delta=0: S_{(2,2)} 上, delta=1: 一般点.
    """
    Omega = make_omega_on_S22(g, rng)
    g1 = g // 2
    g2 = g - g1
    # off-diag 摂動 (純虚数にして Im(Ω)>0 を維持)
    perturb = 1j * rng.standard_normal((g1, g2)) * 0.3 * delta
    Omega[:g1, g1:] = perturb
    Omega[g1:, :g1] = perturb.T.conj()
    # 対称化 (虚部が対称になるよう)
    Omega = (Omega + Omega.T) / 2
    return Omega


def make_z(g: int, rng: np.random.Generator) -> np.ndarray:
    return rng.standard_normal(g) * 0.5 + 1j * rng.standard_normal(g) * 0.1


# ─────────────────────────────────────────────
# 3. 比較実験
# ─────────────────────────────────────────────

def relative_error(v_naive: complex, v_s22: complex) -> float:
    """|v_naive - v_s22| / |v_naive|"""
    denom = abs(v_naive)
    if denom < 1e-300:
        return float('nan')
    return abs(v_naive - v_s22) / denom


def run_comparison(g: int, N_cut: int, Omega: np.ndarray, z: np.ndarray,
                   label: str) -> dict:
    # naive
    t0 = time.perf_counter()
    val_naive = theta_naive(z, Omega, N_cut)
    t_naive = time.perf_counter() - t0

    # s22
    t0 = time.perf_counter()
    val_s22 = theta_s22(z, Omega, N_cut)
    t_s22 = time.perf_counter() - t0

    rel_err = relative_error(val_naive, val_s22)
    speedup = t_naive / t_s22 if t_s22 > 0 else float('inf')

    return {
        'g': g, 'N_cut': N_cut, 'label': label,
        'val_naive': val_naive, 'val_s22': val_s22,
        'rel_err': rel_err,
        't_naive': t_naive, 't_s22': t_s22,
        'speedup': speedup,
    }


def print_result(r: dict):
    print(f"  g={r['g']}, N_cut={r['N_cut']}, [{r['label']}]")
    print(f"    naive : {r['val_naive']:.6f}   ({r['t_naive']:.4f}s)")
    print(f"    s22   : {r['val_s22']:.6f}   ({r['t_s22']:.4f}s)")
    print(f"    相対誤差: {r['rel_err']:.2e}")
    print(f"    高速化倍率: {r['speedup']:.1f}x")


# ─────────────────────────────────────────────
# 4. メイン
# ─────────────────────────────────────────────

def main():
    rng = np.random.default_rng(seed=42)

    g_list     = [8, 9]
    N_cut_list = [1, 2]

    results = []

    print("=" * 65)
    print("同一 Ω 比較: naive vs s22")
    print("=" * 65)

    for g in g_list:
        z = make_z(g, rng)

        # ケース A: S_{(2,2)} 上 (ε_{12}=0 厳密)
        Omega_on = make_omega_on_S22(g, rng)

        # ケース B: S_{(2,2)} から外れた点 (delta=0.5)
        Omega_off = make_omega_off_S22(g, delta=0.5, rng=rng)

        print(f"\n─── g={g} ───")

        for N_cut in N_cut_list:
            print(f"\n  [N_cut={N_cut}]")

            r_on = run_comparison(g, N_cut, Omega_on, z,
                                  label="S22上 (ε=0厳密)")
            print_result(r_on)
            results.append(r_on)

            r_off = run_comparison(g, N_cut, Omega_off, z,
                                   label="S22外 (delta=0.5)")
            print_result(r_off)
            results.append(r_off)

    # ─── サマリー表 ───
    print("\n" + "=" * 65)
    print("サマリー表 (同一Ω・z での比較)")
    print("=" * 65)
    header = f"{'g':>3} {'N':>2} {'ラベル':<18} {'相対誤差':>12} {'naive(s)':>10} {'s22(s)':>10} {'倍率':>8}"
    print(header)
    print("-" * 65)
    for r in results:
        label_short = "S22上" if "S22上" in r['label'] else "S22外"
        print(f"{r['g']:>3} {r['N_cut']:>2} {label_short:<18} "
              f"{r['rel_err']:>12.2e} "
              f"{r['t_naive']:>10.4f} "
              f"{r['t_s22']:>10.4f} "
              f"{r['speedup']:>8.1f}x")

    print()
    print("解釈:")
    print("  S22上 → 相対誤差≈0 が期待値 (Fay公式により ε_{12}=0 厳密)")
    print("  S22外 → 相対誤差 > 0 が期待値 (off-diag 無視による近似誤差)")
    print("  高速化倍率 = naive時間 / s22時間")


if __name__ == "__main__":
    main()