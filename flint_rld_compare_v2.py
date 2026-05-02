"""
flint_rld_compare_v2.py
FLINT (acb_theta) vs RLD vs Naive の三つ巴比較 - 修正版
修正点：acb文字列パースをやめて .real.mid() / .imag.mid() で直接取得

環境：AMD Ryzen 7、64GB RAM
"""

import cmath, math, time, json
import numpy as np
from datetime import datetime
from flint import acb, acb_mat

# ───────── theta 計算関数 ─────────

def log_theta_naive(z, Omega, N_cut):
    g = len(z)
    b = 2 * N_cut + 1
    total = b ** g
    nv = np.empty(g, dtype=np.float64)
    s = 0.0 + 0.0j
    for idx in range(total):
        tmp = idx
        for k in range(g):
            nv[k] = float(tmp % b - N_cut)
            tmp //= b
        qc = nv @ (Omega @ nv)
        lc = nv @ z
        s += cmath.exp(1j * math.pi * qc + 2j * math.pi * lc)
    return s

def theta_rld_blockdiag(z_np, Omega_np, N_cut):
    g = len(z_np)
    if g <= 2:
        return log_theta_naive(z_np, Omega_np, N_cut)
    g_half = g // 2
    v1 = theta_rld_blockdiag(z_np[:g_half], Omega_np[:g_half, :g_half], N_cut)
    v2 = theta_rld_blockdiag(z_np[g_half:], Omega_np[g_half:, g_half:], N_cut)
    return v1 * v2

def flint_theta00(Omega_np, z_np):
    """
    FLINT で theta_{0,0} を計算。
    修正：文字列パースをやめて .real.mid() / .imag.mid() で直接取得。
    """
    g = len(z_np)
    tau = acb_mat([[acb(Omega_np[i,j].real, Omega_np[i,j].imag)
                    for j in range(g)] for i in range(g)])
    z   = acb_mat([[acb(z_np[i].real, z_np[i].imag)] for i in range(g)])
    entries = tau.theta(z).entries()
    v = entries[0]

    # 方法1: .real.mid() / .imag.mid() で取得（推奨）
    try:
        re_part = float(v.real.mid())
        im_part = float(v.imag.mid())
        return complex(re_part, im_part), "mid()"
    except AttributeError:
        pass

    # 方法2: .real / .imag が直接 float に変換できる場合
    try:
        re_part = float(v.real)
        im_part = float(v.imag)
        return complex(re_part, im_part), "float()"
    except (TypeError, AttributeError):
        pass

    # 方法3: フォールバック（文字列パース改良版）
    import re
    s_re = str(v.real)
    s_im = str(v.imag)
    def parse_acb_str(s):
        # "[中心値 +/- 誤差]" 形式から中心値を取得
        m = re.search(r'\[?\s*([-+]?\d+\.?\d*(?:e[-+]?\d+)?)', s)
        return float(m.group(1)) if m else 0.0
    return complex(parse_acb_str(s_re), parse_acb_str(s_im)), "str_parse"

# ───────── 設定 ─────────
G_LIST  = [2, 4, 6, 8, 10, 12]
N_CUT   = 1
SEED    = 42
TAU_IM  = 2.0

results = []

print("=" * 65)
print("FLINT vs RLD vs Naive 比較 v2（修正版）")
print(f"N_cut={N_CUT}, seed={SEED}, tau_im={TAU_IM}i")
print(f"環境：AMD Ryzen 7 / 64GB RAM")
print(f"実行日時：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("=" * 65)

rng = np.random.default_rng(SEED)
max_g = max(G_LIST)

Z_master = rng.standard_normal(max_g) + 1j * rng.standard_normal(max_g) * 0.1
Omega_master = np.zeros((max_g, max_g), dtype=complex)
for i in range(max_g):
    Omega_master[i, i] = TAU_IM * 1j

# FLINTのacb取得方法を最初に診断
print("\n[診断] FLINT acb オブジェクトのメソッド確認...")
try:
    test_v = acb(1.23, 4.56)
    print(f"  test acb(1.23, 4.56) = {test_v}")
    try:
        print(f"  .real.mid() = {float(test_v.real.mid())}")
        print(f"  .imag.mid() = {float(test_v.imag.mid())}")
        print("  → mid() メソッド: 利用可能")
    except Exception as e:
        print(f"  → mid() メソッド: 利用不可 ({e})")
    try:
        print(f"  float(.real) = {float(test_v.real)}")
        print("  → float() 変換: 利用可能")
    except Exception as e:
        print(f"  → float() 変換: 利用不可 ({e})")
except Exception as e:
    print(f"  診断エラー: {e}")
print()

for g in G_LIST:
    z     = Z_master[:g].copy()
    Omega = Omega_master[:g, :g].copy()

    row = {"g": g, "N_cut": N_CUT}
    print(f"\n--- g={g} ---")

    # 1. Naive（g<=8 のみ）
    if g <= 8:
        t0 = time.perf_counter()
        val_naive = log_theta_naive(z, Omega, N_CUT)
        row["naive_time"] = time.perf_counter() - t0
        row["naive_re"]   = val_naive.real
        row["naive_im"]   = val_naive.imag
        print(f"  Naive : {val_naive.real:.8f} + {val_naive.imag:.8f}i  ({row['naive_time']:.3f}s)")
    else:
        val_naive = None
        row["naive_time"] = None
        print(f"  Naive : スキップ（g={g}は時間超過）")

    # 2. RLD
    t0 = time.perf_counter()
    val_rld = theta_rld_blockdiag(z, Omega, N_CUT)
    row["rld_time"] = time.perf_counter() - t0
    row["rld_re"]   = val_rld.real
    row["rld_im"]   = val_rld.imag
    print(f"  RLD   : {val_rld.real:.8f} + {val_rld.imag:.8f}i  ({row['rld_time']:.4f}s)")

    # 3. FLINT
    try:
        t0 = time.perf_counter()
        val_flint, method = flint_theta00(Omega, z)
        row["flint_time"]   = time.perf_counter() - t0
        row["flint_re"]     = val_flint.real
        row["flint_im"]     = val_flint.imag
        row["flint_method"] = method
        print(f"  FLINT : {val_flint.real:.8f} + {val_flint.imag:.8f}i  ({row['flint_time']:.3f}s)  [{method}]")
    except Exception as e:
        row["flint_time"] = None
        row["flint_re"]   = None
        row["flint_im"]   = None
        print(f"  FLINT : エラー {e}")
        val_flint = None

    # 4. 誤差比較
    if val_naive is not None and val_flint is not None:
        err_nf = abs(val_naive - val_flint)
        err_nr = abs(val_naive - val_rld)
        err_rf = abs(val_rld  - val_flint)
        row["err_naive_flint"] = err_nf
        row["err_naive_rld"]   = err_nr
        row["err_rld_flint"]   = err_rf
        print(f"  |Naive-FLINT| = {err_nf:.2e}")
        print(f"  |Naive-RLD|   = {err_nr:.2e}")
        print(f"  |RLD-FLINT|   = {err_rf:.2e}")
    elif val_flint is not None:
        err_rf = abs(val_rld - val_flint)
        row["err_rld_flint"] = err_rf
        print(f"  |RLD-FLINT|   = {err_rf:.2e}")

    row["timestamp"] = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
    results.append(row)

# JSON保存
out_path = "flint_rld_compare_v2_results.json"
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(results, f, indent=4, default=str)

print("\n" + "=" * 65)
print(f"完了。結果を {out_path} に保存しました。")
print("=" * 65)
