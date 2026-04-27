"""
RLD_theta_engine.py
Recursive Log-Decomposition Theta Engine: Handles ultra-high genus without overflow.

Juliaのexpは巨大な値でもInfを返しますが、Pythonの cmath.expは同じ状況でOverflowError。
cmath.exp(log_val)を try/except OverflowErrorで囲み、
例外発生時もisinf検出時と同じメッセージ（対数表示）に流れるようにした

"""

import os
import json
import math
import cmath
import time
import threading
import numpy as np
from datetime import datetime
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional

# ───────── 定数・パス ─────────
PREFIX      = "PY_RLD_"
SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
LOG_PATH    = os.path.join(SCRIPT_DIR, f"{PREFIX}log.txt")
RESULT_PATH = os.path.join(SCRIPT_DIR, f"{PREFIX}results.json")

LOG_LOCK     = threading.Lock()
RESULTS_LOCK = threading.Lock()

# ───────── ユーティリティ ─────────
def log_msg(msg: str) -> None:
    line = f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
    print(line)
    with LOG_LOCK:
        with open(LOG_PATH, "a", encoding="utf-8") as f:
            f.write(line + "\n")


@dataclass
class E8Cfg:
    g_list:         list[int]
    N_cut:          int
    g_limit_naive:  int
    tau_dummy_im:   complex
    seed_base:      int
    outer_loops:    int
    use_log_output: bool   # trueなら結果を対数で表示、falseなら10進に戻す


# ───────── Theta Core: Log-Naive ─────────
# Naive計算も対数で返すように変更
def log_theta_naive(z: np.ndarray, Omega: np.ndarray, N_cut: int) -> complex:
    g = len(z)
    b = 2 * N_cut + 1
    total = b ** g

    nt = os.cpu_count() or 1
    chunk = total // nt
    ranges = []
    for ci in range(nt):
        i_start = ci * chunk + min(ci, total % nt)
        i_end   = i_start + chunk - 1 + (1 if ci < total % nt else 0)
        ranges.append((i_start, i_end))

    def worker(i_start: int, i_end: int) -> complex:
        local_sum = 0.0 + 0.0j
        nv = np.empty(g, dtype=np.float64)
        for idx in range(i_start, i_end + 1):
            tmp = idx
            for k in range(g):
                nv[k] = float(tmp % b - N_cut)
                tmp //= b
            # Naiveは項の和なので、まず通常の和を計算してから最後にlogをとる
            # (内部の個別の項がオーバーフローする場合はさらに工夫が必要だが、
            #  g_limit_naiveは通常小さいのでこの実装で十分)
            qc = nv @ (Omega @ nv)
            lc = nv @ z
            local_sum += cmath.exp(1j * math.pi * qc + 2j * math.pi * lc)
        return local_sum

    with ThreadPoolExecutor(max_workers=nt) as executor:
        futures = [executor.submit(worker, s, e) for s, e in ranges]
        val = sum(f.result() for f in as_completed(futures))

    return cmath.log(val)  # 複素対数を返す


# ───────── Theta Core: Recursive Log-Decomposition ─────────
def log_theta_recursive(
    z: np.ndarray,
    Omega: np.ndarray,
    N_cut: int,
    g_limit: int,
    tau_dummy: complex,
) -> complex:
    g = len(z)

    if g <= g_limit:
        return log_theta_naive(z, Omega, N_cut)

    if g & (g - 1) != 0:  # not a power of 2
        new_g = 1 << math.ceil(math.log2(g))
        z_pad = np.zeros(new_g, dtype=complex)
        z_pad[:g] = z
        Omega_pad = np.zeros((new_g, new_g), dtype=complex)
        Omega_pad[:g, :g] = Omega
        for i in range(g, new_g):
            Omega_pad[i, i] = tau_dummy
        return log_theta_recursive(z_pad, Omega_pad, N_cut, g_limit, tau_dummy)

    g_half = g // 2
    with ThreadPoolExecutor(max_workers=2) as executor:
        t1 = executor.submit(
            log_theta_recursive,
            z[:g_half], Omega[:g_half, :g_half], N_cut, g_limit, tau_dummy
        )
        t2 = executor.submit(
            log_theta_recursive,
            z[g_half:], Omega[g_half:, g_half:], N_cut, g_limit, tau_dummy
        )
        # log(A * B) = log(A) + log(B)
        return t1.result() + t2.result()


# ───────── 出力制御 ─────────
def format_result(log_val: complex, as_log: bool) -> tuple[str, complex]:
    if as_log:
        return (
            f"Log(Theta) = {log_val.real:.6f} + {log_val.imag:.6f}i",
            log_val,
        )

    # 10進数に戻す試行
    try:
        res = cmath.exp(log_val)
        if math.isinf(res.real) or math.isinf(res.imag):
            raise OverflowError
    except OverflowError:
        msg = (
            f"数字が大きすぎて10進に戻せません。対数結果を表示します: "
            f"Log = {log_val.real:.6f} + {log_val.imag:.6f}i"
        )
        return msg, log_val
    return f"{res.real:.6e} + {res.imag:.6e}i", res


# ───────── メインロジック ─────────
def main() -> None:
    cfg = E8Cfg(
        g_list        = [16, 17, 18, 19, 20, 25, 30, 50, 100, 999, 1500],  # g=20000などの超高属数もテスト
        N_cut         = 2,               # N_cut
        g_limit_naive = 1,               # Naive限界
        tau_dummy_im  = 10.0j,           # パディング虚部
        seed_base     = 42,              # seed
        outer_loops   = 5,               # 外側ループ数
        use_log_output= False,           # true:対数、デフォルトは10進表示（false） LOGならg=20000も可能
    )

    log_msg("=" * 70)
    log_msg("RLD-Engine Start: Recursive Log-Decomposition")
    log_msg(f"G-List: {cfg.g_list}, Log-Mode Output: {cfg.use_log_output}")
    log_msg("=" * 70)

    all_results: list[dict] = []
    max_g = max(cfg.g_list)

    for loop_idx in range(1, cfg.outer_loops + 2):
        log_msg(f">>> Starting Global Loop [{loop_idx} / {cfg.outer_loops}]")

        rng = np.random.default_rng(cfg.seed_base + loop_idx)
        Z_master    = rng.standard_normal(max_g) + 1j * (rng.standard_normal(max_g) * 0.1)
        Omega_master = np.zeros((max_g, max_g), dtype=complex)
        for i in range(max_g):
            Omega_master[i, i] = 2.0j + complex(0.0, 0.2 * rng.standard_normal())
        Omega_master = (Omega_master + Omega_master.conj().T) / 2

        for g in cfg.g_list:
            z     = Z_master[:g].copy()
            Omega = Omega_master[:g, :g].copy()

            # 1. Recursive (常に内部は対数)
            t_start     = time.perf_counter()
            log_val_rec = log_theta_recursive(z, Omega, cfg.N_cut, cfg.g_limit_naive, cfg.tau_dummy_im)
            t_rec       = time.perf_counter() - t_start

            # 2. Naive比較 (gが小さい場合のみ。これも対数で受ける)
            log_val_naive: Optional[complex] = None
            abs_err_log:   Optional[float]  = None
            if g <= cfg.g_limit_naive:
                log_val_naive = log_theta_naive(z, Omega, cfg.N_cut)
                abs_err_log   = abs(log_val_rec - log_val_naive)

            # 出力フォーマットの適用
            display_str, final_val = format_result(log_val_rec, cfg.use_log_output)

            err_str = "N/A" if abs_err_log is None else f"{abs_err_log:.2e}"
            msg = (
                f"    (Loop {loop_idx}, g={g}) Result: {display_str}, "
                f"Time: {t_rec:.4f}s, LogErr: {err_str}"
            )
            log_msg(msg)

            all_results.append({
                "loop":          loop_idx,
                "g":             g,
                "is_log_format": cfg.use_log_output,
                "log_val_re":    log_val_rec.real,
                "log_val_im":    log_val_rec.imag,
                "time_rec":      t_rec,
                "abs_err_log":   abs_err_log,
                "timestamp":     datetime.now().strftime("%Y-%m-%dT%H:%M:%S"),
            })

        with RESULTS_LOCK:
            with open(RESULT_PATH, "w", encoding="utf-8") as f:
                json.dump(all_results, f, indent=4)

    log_msg("All log-reconstruction experiments completed.")


if __name__ == "__main__":
    main()
