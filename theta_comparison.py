"""
theta_comparison.py  ― S22上（厳密ブロック対角）専用ベンチマーク

naive vs s22 を同一Ω・z で評価し、相対誤差と速度を比較する。
S22外（近似精度）の測定は本スクリプトの対象外。

【使い方】
  python theta_comparison.py
"""

import numpy as np
import json, time, os, sys
from datetime import datetime
from itertools import product as iproduct

# ───────── パス ─────────
SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.join(SCRIPT_DIR, "config_theta.txt")
LOG_PATH    = os.path.join(SCRIPT_DIR, "theta_log.txt")
RESULT_PATH = os.path.join(SCRIPT_DIR, "theta_results_py.json")

# ───────── ログ ─────────
def log_msg(msg):
    line = f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(line + "\n")

# ───────── 設定 ─────────
DEFAULTS = dict(
    g_list=[2,3,4,5,6,7,8,9,10,11,12],
    N_cut_list=[1,2],
    seed=42,
    resume=True,
    report_every=1_000_000,
)

def load_cfg():
    if not os.path.isfile(CONFIG_PATH):
        with open(CONFIG_PATH, "w", encoding="utf-8") as f:
            f.write("# theta_comparison.py config\n")
            f.write("g_list = 2,4,6,8\n")
            f.write("N_cut_list = 1\n")
            f.write("seed = 42\n")
            f.write("resume = true\n")
            f.write("report_every = 10000000\n")
        log_msg(f"設定ファイル生成: {CONFIG_PATH}")
        return dict(DEFAULTS)
    kv = {}
    with open(CONFIG_PATH, encoding="utf-8") as f:
        for raw in f:
            s = raw.strip()
            if not s or s.startswith("#"):
                continue
            if "=" not in s:
                continue
            k, v = s.split("=", 1)
            kv[k.strip()] = v.strip()
    d = dict(DEFAULTS)
    if "g_list"       in kv: d["g_list"]       = [int(x) for x in kv["g_list"].split(",")]
    if "N_cut_list"   in kv: d["N_cut_list"]   = [int(x) for x in kv["N_cut_list"].split(",")]
    if "seed"         in kv: d["seed"]         = int(kv["seed"])
    if "resume"       in kv: d["resume"]       = kv["resume"].lower() == "true"
    if "report_every" in kv: d["report_every"] = int(kv["report_every"])
    return d

# ───────── JSON保存（atomic write） ─────────
def save_results(rs):
    tmp = RESULT_PATH + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(rs, f, indent=2, ensure_ascii=False)
    os.replace(tmp, RESULT_PATH)

def load_results():
    if not os.path.isfile(RESULT_PATH):
        return []
    with open(RESULT_PATH, encoding="utf-8") as f:
        return json.load(f)

def already_done(rs, g, N):
    return any(r["g"] == g and r["N_cut"] == N for r in rs)

# ───────── Theta 関数 ─────────
def theta_naive(z, Omega, N_cut, report_every=0, label=""):
    g = len(z)
    b = 2 * N_cut + 1
    total = b ** g
    result = 0.0 + 0.0j
    count = 0
    for n in iproduct(range(-N_cut, N_cut + 1), repeat=g):
        n_arr = np.array(n, dtype=float)
        phase = np.pi * 1j * (n_arr @ Omega @ n_arr) + 2 * np.pi * 1j * (n_arr @ z)
        result += np.exp(phase)
        if report_every > 0:
            count += 1
            if count % report_every == 0:
                pct = 100.0 * count / total
                lbl2 = f" [{label}]" if label else ""
                log_msg(f"    [theta_naive{lbl2}] 格子点 {count} / {total} ({pct:.2f}%) 完了")
    return result

def theta_s22(z, Omega, N_cut, report_every=0, label=""):
    g1 = len(z) // 2
    l1 = f"{label}/B1" if label else "B1"
    l2 = f"{label}/B2" if label else "B2"
    v1 = theta_naive(z[:g1],  Omega[:g1, :g1],  N_cut, report_every, l1)
    v2 = theta_naive(z[g1:],  Omega[g1:, g1:],  N_cut, report_every, l2)
    return v1 * v2

# ───────── Ω生成（厳密S(2,2)ブロック対角のみ） ─────────
def pos_block(n, rng):
    A = rng.standard_normal((n, n)) * 0.3
    return 1j * (A @ A.T + 1.5 * np.eye(n))

def omega_on(g, rng):
    g1 = g // 2
    Omega = np.zeros((g, g), dtype=complex)
    Omega[:g1,  :g1]  = pos_block(g1,   rng)
    Omega[g1:,  g1:]  = pos_block(g-g1, rng)
    return (Omega + Omega.T) / 2   # transpose（非共役）で対称化

def make_z(g, rng):
    return rng.standard_normal(g) * 0.5 + 1j * rng.standard_normal(g) * 0.1

def make_data(g, seed):
    rng_z = np.random.default_rng(seed + g)
    rng_o = np.random.default_rng(seed + g + 10000)
    z = make_z(g, rng_z)
    Omega = omega_on(g, rng_o)
    return z, Omega

# ───────── 比較実行 ─────────
def run_one(g, N, Omega, z, report_every=0):
    b = 2*N+1; pts = b**g
    log_msg(f"  → naive 開始: g={g} N={N} 格子点={pts}")
    t0 = time.perf_counter()
    vn = theta_naive(z, Omega, N, report_every, f"naive/g{g}/N{N}")
    tn = time.perf_counter() - t0
    log_msg(f"  ← naive 完了: {tn:.4f}s  val={vn.real:.6f}{vn.imag:+.6f}i")

    if abs(vn) < 10.0:
        log_msg(f"  ⚠ 警告: |θ|={abs(vn):.4f} < 10（ゼロ点近傍）rel_errが大きくなる可能性あり。別seedでの再測定を推奨。")

    log_msg(f"  → s22  開始: g={g} N={N}")
    t0 = time.perf_counter()
    vs = theta_s22(z, Omega, N, report_every, f"s22/g{g}/N{N}")
    ts = time.perf_counter() - t0
    log_msg(f"  ← s22  完了: {ts:.4f}s  val={vs.real:.6f}{vs.imag:+.6f}i")

    re = abs(vn - vs) / abs(vn) if abs(vn) >= 1e-300 else float("nan")
    sp = tn / ts if ts > 1e-12 else float("inf")
    return {
        "g": g, "N_cut": N,
        "val_naive_re": vn.real, "val_naive_im": vn.imag,
        "val_s22_re":   vs.real, "val_s22_im":   vs.imag,
        "rel_err": re, "t_naive": tn, "t_s22": ts, "speedup": sp,
        "timestamp": datetime.now().strftime("%Y-%m-%dT%H:%M:%S"),
    }

def show_result(r):
    print(f"    g={r['g']} N={r['N_cut']}")
    print(f"      naive: {r['val_naive_re']:.6f}{r['val_naive_im']:+.6f}i ({r['t_naive']:.4f}s)")
    print(f"      s22  : {r['val_s22_re']:.6f}{r['val_s22_im']:+.6f}i ({r['t_s22']:.4f}s)")
    sp = r.get("speedup", float("nan"))
    print(f"      rel_err={r['rel_err']:.2e}  speedup={sp:.1f}x")
    sys.stdout.flush()

# ───────── サマリー ─────────
def summary(rs):
    log_msg("=" * 70)
    log_msg("サマリー")
    log_msg(f"{'g':>3} {'N':>2} {'rel_err':>12} {'naive(s)':>10} {'s22(s)':>10} {'倍率':>8}")
    log_msg("-" * 70)
    for r in sorted(rs, key=lambda r: (r["g"], r["N_cut"])):
        sp = r.get("speedup", float("nan"))
        log_msg(f"{r['g']:>3} {r['N_cut']:>2} {r['rel_err']:>12.2e} "
                f"{r['t_naive']:>10.4f} {r['t_s22']:>10.4f} {sp:>7.1f}x")
    log_msg(f"結果: {RESULT_PATH}  ログ: {LOG_PATH}")

# ───────── main ─────────
def main():
    cfg = load_cfg()
    rs = load_results() if cfg["resume"] else []

    log_msg("=" * 70)
    log_msg("theta_comparison.py  naive vs s22  [S22上専用]")
    log_msg(f"Python {sys.version.split()[0]}")
    log_msg(f"g={cfg['g_list']}  N_cut={cfg['N_cut_list']}  seed={cfg['seed']}")
    log_msg(f"report_every={cfg['report_every']}  resume={cfg['resume']}  既完了={len(rs)}件")
    log_msg("=" * 70)

    cases = [(g, N) for g in cfg["g_list"] for N in cfg["N_cut_list"]]
    total = len(cases)

    for n, (g, N) in enumerate(cases, 1):
        if cfg["resume"] and already_done(rs, g, N):
            log_msg(f"  スキップ [{n}/{total}] g={g} N={N}")
            rec = next((r for r in rs if r["g"]==g and r["N_cut"]==N), None)
            if rec:
                show_result(rec)
            continue

        log_msg(f"\n─── [{n}/{total}] ({100.0*n/total:.1f}%) g={g} N={N} ───")
        z, Omega = make_data(g, cfg["seed"])
        r = run_one(g, N, Omega, z, cfg["report_every"])
        show_result(r)
        rs.append(r)
        save_results(rs)
        sp = r.get("speedup", float("nan"))
        log_msg(f"  保存: g={g} N={N} rel_err={r['rel_err']:.2e} speedup={sp:.1f}x")

    summary(rs)
    log_msg("完了。")

if __name__ == "__main__":
    main()
