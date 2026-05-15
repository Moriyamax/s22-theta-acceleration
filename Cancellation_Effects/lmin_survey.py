"""
超楕円曲線のλ_min探索
目的: 様々な超楕円曲線からΩを計算し、λ_minがCW2023基準(≥2.4)を
     満たすかを確認する

実行環境: WSL2 + PARI/GP + hcperiods
実行: python lmin_survey.py

hcperiodsの使い方:
  gp.read('/home/user/hcperiods/gp/periods.gp')
  M = gp.hcInit(f, p)  # f: 多項式, p: 素数(精度用)
  Omega = gp.hcSmallperiods(M)  # 周期行列
"""

import numpy as np
import subprocess
import sys

# ============================================================
# PARI/GP経由でhcperiodsを呼び出す
# ============================================================

def run_gp(script, timeout=60):
    """PARI/GPスクリプトを実行して結果を返す"""
    try:
        result = subprocess.run(
            ['gp', '-q', '--'],
            input=script.encode(),
            capture_output=True,
            timeout=timeout
        )
        return result.stdout.decode(), result.stderr.decode()
    except subprocess.TimeoutExpired:
        return None, 'timeout'
    except FileNotFoundError:
        return None, 'gp not found'


def parse_matrix(gp_output):
    """PARI/GPの行列出力をnumpy配列に変換"""
    lines = gp_output.strip().split('\n')
    rows = []
    for line in lines:
        line = line.strip()
        if not line or line.startswith('%'):
            continue
        # [a, b, c; d, e, f] 形式を解析
        line = line.replace('[', '').replace(']', '').replace(';', '\n')
        for row_str in line.split('\n'):
            row_str = row_str.strip()
            if not row_str:
                continue
            # 複素数の解析: a + b*I 形式
            entries = []
            for entry in row_str.split(','):
                entry = entry.strip()
                try:
                    # 実数
                    entries.append(complex(float(entry)))
                except ValueError:
                    # 複素数: 'a + b*I' or 'a*I' など
                    entry = entry.replace('*I', 'j').replace(' ', '')
                    try:
                        entries.append(complex(entry))
                    except ValueError:
                        entries.append(complex(0))
            if entries:
                rows.append(entries)
    if not rows:
        return None
    try:
        return np.array(rows, dtype=complex)
    except Exception:
        return None


def compute_omega_hcperiods(f_str, genus, prec=100):
    """
    hcperiodsでΩを計算する
    f_str: PARI/GP形式の多項式文字列（例: "x^5 - 1"）
    genus: 種数
    """
    script = f"""
\\\\read "/home/user/hcperiods/gp/periods.gp"
f = {f_str};
M = hcInit(f, {prec});
Om = hcSmallperiods(M);
print(Om);
quit
"""
    stdout, stderr = run_gp(script)
    if stdout is None:
        return None, stderr

    Om = parse_matrix(stdout)
    return Om, stderr


def compute_lmin(Om):
    """周期行列からλ_minを計算"""
    if Om is None:
        return None
    ImOm = Om.imag
    eigs = np.linalg.eigvalsh(ImOm)
    return eigs.min()


def compute_error_bound(Om, C_norm, prec_target=1e-2):
    """
    定理のForm 2で誤差上界を計算
    |Err| ≤ 4π·||C||_F·[ϑ₃(0|iλ_min)^g - 1]
    """
    lmin = compute_lmin(Om)
    if lmin is None:
        return None, None
    g = Om.shape[0]

    def theta3(tau, N=20):
        return sum(np.exp(-np.pi * tau * n**2) for n in range(-N, N+1))

    ub = 4 * np.pi * C_norm * (theta3(lmin)**g - 1)
    beats_cw = ub < prec_target
    return ub, beats_cw


# ============================================================
# 超楕円曲線の系統的調査
# ============================================================

def survey_hyperelliptic_curves():
    """
    様々な超楕円曲線でλ_minを計算
    y² = f(x) の形式

    対象:
    1. 分枝点が2クラスターに分かれる曲線（λ_min大が期待）
    2. 一様分布の分枝点（ベースライン）
    3. 分枝点の間隔を変えた系統的スキャン
    """

    print("=" * 65)
    print("超楕円曲線のλ_min探索")
    print("目標: λ_min ≥ 2.4 → 定理でCW2023（誤差1%）を超えることを保証")
    print("=" * 65)

    # hcperiods動作確認
    test_out, test_err = run_gp(
        '\\\\read "/home/user/hcperiods/gp/periods.gp"\nprint("OK")\nquit\n'
    )
    if test_out is None or 'OK' not in test_out:
        print(f"\nhcperiods が利用できません: {test_err}")
        print("WSL2環境で実行してください")
        print("\nWSL2での実行コマンド:")
        print("  cd /mnt/c/Riron/S22理論 && python3 lmin_survey.py")
        return

    print("\nhcperiods: OK\n")

    # 調査する曲線のリスト
    # 形式: (説明, genus, PARI/GP多項式, 期待λ_min)
    curves = [
        # genus 2: y² = f(x), deg(f)=5 or 6
        ("g=2, 一様分布",
         2, "x^5 - 1", "unknown"),

        # genus 2, 2クラスター分岐点（-L付近と+L付近）
        ("g=2, 2クラスター L=5",
         2, "(x^2-25)*(x^3+x^2-x+1)", "large?"),

        ("g=2, 2クラスター L=10",
         2, "(x^2-100)*(x^3+x^2-x+1)", "large?"),

        # genus 3: y² = f(x), deg(f)=7
        ("g=3, 一様分布",
         3, "x^7 - 1", "unknown"),

        ("g=3, 2クラスター L=5",
         3, "(x^3-125)*(x^4+x^2+1)", "large?"),

        ("g=3, 2クラスター L=10",
         3, "(x^3-1000)*(x^4+x^2+1)", "large?"),

        # genus 4: y² = f(x), deg(f)=9
        ("g=4, 2クラスター L=5",
         4, "(x^4-625)*(x^5+x^3+x+1)", "large?"),

        ("g=4, 2クラスター L=10",
         4, "(x^4-10000)*(x^5+x^3+x+1)", "large?"),

        # genus 5: y² = f(x), deg(f)=11
        ("g=5, 2クラスター L=10",
         5, "(x^5-100000)*(x^6+x^4+x^2+1)", "large?"),

        # genus 8（前回の実験に対応）
        ("g=8, 2クラスター L=20",
         8, "(x^8-1.6e11)*(x^9+x^7+x^5+x^3+x+1)", "~2?"),
    ]

    results = []
    print(f"{'曲線':35} {'g':>4} {'λ_min':>8} {'UB(C=0.1)':>12} "
          f"{'CW超え?':>8} {'状態':>10}")
    print("-" * 85)

    for desc, genus, f_str, expected in curves:
        Om, err = compute_omega_hcperiods(f_str, genus, prec=100)

        if Om is None or Om.shape[0] != genus:
            print(f"  {desc:35} {genus:4d} {'ERROR':>8} "
                  f"{'---':>12} {'---':>8} {err[:20]:>10}")
            continue

        lmin = compute_lmin(Om)
        if lmin is None or lmin < 0:
            print(f"  {desc:35} {genus:4d} {'INVALID':>8} "
                  f"{'---':>12} {'---':>8} {'---':>10}")
            continue

        ub, beats = compute_error_bound(Om, C_norm=0.1, prec_target=1e-2)

        status = "✓ CW超え" if beats else "× NG"
        print(f"  {desc:35} {genus:4d} {lmin:8.3f} {ub:12.3e} "
              f"{'YES' if beats else 'NO':>8} {status:>10}")

        results.append({
            'desc': desc, 'genus': genus,
            'lmin': lmin, 'ub': ub, 'beats_cw': beats,
            'Om': Om
        })

    # ---- λ_min vs クラスター間隔のスキャン ----
    print("\n" + "=" * 65)
    print("クラスター間隔 L vs λ_min のスキャン（genus 3）")
    print("=" * 65)
    print(f"  {'L':>6} {'λ_min':>10} {'UB':>12} {'CW超え':>8}")
    print("  " + "-" * 42)

    for L in [1, 2, 5, 10, 20, 50, 100]:
        f_str = f"(x^3-{L**3})*(x^4+x^2+1)"
        Om, err = compute_omega_hcperiods(f_str, 3, prec=100)
        if Om is None or Om.shape[0] != 3:
            print(f"  {L:6d} {'ERROR':>10}")
            continue
        lmin = compute_lmin(Om)
        ub, beats = compute_error_bound(Om, C_norm=0.1)
        print(f"  {L:6d} {lmin:10.4f} {ub:12.3e} {'YES' if beats else 'NO':>8}")

    print("\n完了")
    return results


if __name__ == "__main__":
    survey_hyperelliptic_curves()
