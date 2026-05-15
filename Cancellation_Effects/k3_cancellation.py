"""
k=3,4 での z=0 相殺機構の精査
目的: k=2と比べて2桁大きい残差の原因を特定

k=2の相殺機構（確定済み）:
  (n_A, n_B) と (n_A, -n_B) のペアで完全相殺
  → 全格子点が2点ペアに分解される

k=3の場合:
  格子点は (n_A, n_B, n_C) の3成分
  off-diagonal摂動E_offは (A,B), (A,C), (B,C) の3ブロックを持つ
  nEn = 2Re(n_A^T C_AB n_B + n_A^T C_AC n_C + n_B^T C_BC n_C)
  → 相殺ペアの構造が複雑

仮説X: k=3では (n_A, n_B, n_C) と (n_A, -n_B, n_C) のペアが
        n_A^T C_AB n_B の項を相殺するが
        n_B^T C_BC n_C の項が残る
        → 部分的な相殺のみ → 残差あり

仮説Y: k=3でも完全相殺が起きているが
        別の対称性から来ており、数値精度の問題で2桁の差が出る

実行: python k3_cancellation.py
"""

import numpy as np
from itertools import product

def make_omega_kblock(g, k, lmin, seed=42):
    """Re(Ω)=0、k等分ブロック対角"""
    rng = np.random.default_rng(seed)
    assert g % k == 0
    gb = g // k

    def make_block(n):
        A = rng.standard_normal((n, n))
        Q, _ = np.linalg.qr(A)
        eigs = lmin + rng.exponential(0.3, n)
        return Q @ np.diag(eigs) @ Q.T

    blocks = [make_block(gb) for _ in range(k)]
    ImOm = np.zeros((g, g))
    for i, B in enumerate(blocks):
        ImOm[i*gb:(i+1)*gb, i*gb:(i+1)*gb] = B
    return 1j * ImOm, gb


def make_Eoff_k(g, k, seed=0):
    """k分割off-diagonal摂動（対角ブロックはゼロ）"""
    rng = np.random.default_rng(seed + 999)
    gb = g // k
    E = np.zeros((g, g), dtype=complex)
    for i in range(k):
        for j in range(k):
            if i == j:
                continue
            C_ij = rng.standard_normal((gb, gb)) + 1j*rng.standard_normal((gb, gb))
            E[i*gb:(i+1)*gb, j*gb:(j+1)*gb] = C_ij
    E = (E + E.conj().T) / 2
    for i in range(k):
        E[i*gb:(i+1)*gb, i*gb:(i+1)*gb] = 0
    E /= np.linalg.norm(E, 'fro')
    return E


def get_nEn_decomposition(nv, E, k, gb):
    """
    nEn = nᵀE_off n を (i,j)ブロックペアに分解
    各ブロックペア (i<j) の寄与を分離して返す
    """
    contributions = {}
    for i in range(k):
        for j in range(i+1, k):
            n_i = nv[i*gb:(i+1)*gb]
            n_j = nv[j*gb:(j+1)*gb]
            E_ij = E[i*gb:(i+1)*gb, j*gb:(j+1)*gb]
            E_ji = E[j*gb:(j+1)*gb, i*gb:(i+1)*gb]
            # n^T E n の (i,j)ペア寄与: 2Re(n_i^T E_ij n_j)
            contrib = 2 * np.real(n_i @ E_ij @ n_j)
            contributions[(i,j)] = contrib
    return contributions


def analyze_cancellation_by_block_pair(Om0, E, z, N_cut, k, gb):
    """
    各ブロックペア(i,j)ごとのLHSへの寄与を計算
    どのペアで相殺が起きているかを特定
    """
    g = Om0.shape[0]
    n0 = tuple([0]*g)

    # ブロックペアごとのLHS寄与
    lhs_by_pair = {}
    for i in range(k):
        for j in range(i+1, k):
            lhs_by_pair[(i,j)] = 0j

    total_lhs = 0j

    seen = set()
    terms_list = []
    for n in product(range(-N_cut, N_cut+1), repeat=g):
        nv = np.array(n, dtype=float)
        neg_n = tuple(-x for x in n)
        if n == n0:
            continue
        seen.add(n)

        exp_val = np.exp(-np.pi * nv @ Om0.imag @ nv)
        cos_val = np.cos(2 * np.pi * nv @ z)
        nEn = nv @ E @ nv

        contrib = np.pi * 1j * nEn * np.exp(np.pi*1j*(nv@Om0@nv + 2*nv@z))
        total_lhs += contrib

        # ブロックペア別に分解
        pair_contribs = get_nEn_decomposition(nv, E, k, gb)
        for pair, pc in pair_contribs.items():
            phase = np.pi*1j*(nv@Om0@nv + 2*nv@z)
            lhs_by_pair[pair] += np.pi * 1j * pc * np.exp(phase)

        terms_list.append({
            'n': n, 'nv': nv, 'exp': exp_val, 'cos': cos_val,
            'nEn': nEn, 'contrib': contrib
        })

    return total_lhs, lhs_by_pair, terms_list


def analyze_pairing_structure(Om0, E, z, N_cut, k, gb, terms_list):
    """
    k=2の相殺: (n_A, n_B) と (n_A, -n_B) のペア
    k=3の相殺: どのペアが相殺するかを特定

    全格子点を「相殺ペア」に分類し、相殺しない格子点を特定
    """
    g = Om0.shape[0]
    n0 = tuple([0]*g)

    # 全格子点のdict
    terms_dict = {t['n']: t for t in terms_list}

    # k=2の場合: (n_A, n_B) と (n_A, -n_B) ペアの相殺確認
    if k == 2:
        paired = set()
        cancellation_pairs = []
        residual_terms = []

        for t in terms_list:
            n = t['n']
            if n in paired:
                continue
            nv = t['nv']
            # パートナー: (n_A, -n_B)
            nA = nv[:gb]
            nB = nv[gb:]
            n_partner = tuple(list(nA.astype(int)) + list((-nB).astype(int)))

            if n_partner in terms_dict and n_partner not in paired and n_partner != n:
                paired.add(n)
                paired.add(n_partner)
                t_p = terms_dict[n_partner]
                pair_sum = t['contrib'] + t_p['contrib']
                cancellation_pairs.append({
                    'n': n, 'n_p': n_partner,
                    'contrib_n': t['contrib'],
                    'contrib_np': t_p['contrib'],
                    'pair_sum': pair_sum,
                    'cancelled': abs(pair_sum) < 1e-20
                })
            else:
                residual_terms.append(t)

        return cancellation_pairs, residual_terms

    # k=3の場合: 複数の相殺パターンを試す
    elif k == 3:
        g1, g2, g3 = gb, gb, gb

        paired = set()
        cancellation_pairs = []
        residual_terms = []

        for t in terms_list:
            n = t['n']
            if n in paired:
                continue
            nv = t['nv']
            nA = nv[:g1]
            nB = nv[g1:g1+g2]
            nC = nv[g1+g2:]

            # パターン1: (n_A, n_B, n_C) と (n_A, -n_B, n_C)
            n_p1 = tuple(list(nA.astype(int)) + list((-nB).astype(int)) + list(nC.astype(int)))
            # パターン2: (n_A, n_B, n_C) と (n_A, n_B, -n_C)
            n_p2 = tuple(list(nA.astype(int)) + list(nB.astype(int)) + list((-nC).astype(int)))
            # パターン3: (n_A, n_B, n_C) と (-n_A, n_B, n_C)
            n_p3 = tuple(list((-nA).astype(int)) + list(nB.astype(int)) + list(nC.astype(int)))

            found_partner = False
            for n_p in [n_p1, n_p2, n_p3]:
                if (n_p in terms_dict and n_p not in paired and
                        n_p != n and tuple(-x for x in n_p) != n):
                    paired.add(n)
                    paired.add(n_p)
                    t_p = terms_dict[n_p]
                    pair_sum = t['contrib'] + t_p['contrib']
                    cancellation_pairs.append({
                        'n': n, 'n_p': n_p,
                        'pattern': [n_p1, n_p2, n_p3].index(n_p) + 1,
                        'contrib_n': t['contrib'],
                        'contrib_np': t_p['contrib'],
                        'pair_sum': pair_sum,
                        'cancelled': abs(pair_sum) < 1e-15
                    })
                    found_partner = True
                    break

            if not found_partner:
                residual_terms.append(t)

        return cancellation_pairs, residual_terms


def main():
    print("=" * 65)
    print("k=3,4 での z=0 相殺機構の精査")
    print("=" * 65)

    N_cut = 2
    lmin = 3.0

    for k, g in [(2, 4), (3, 6), (4, 4)]:
        print(f"\n{'='*50}")
        print(f"k={k}, g={g}, gb={g//k}, N_cut={N_cut}")
        print(f"{'='*50}")

        Om0, gb = make_omega_kblock(g, k, lmin, seed=0)
        E = make_Eoff_k(g, k, seed=0)
        z = np.zeros(g)

        # ---- 全体のLHS ----
        total_lhs, lhs_by_pair, terms_list = analyze_cancellation_by_block_pair(
            Om0, E, z, N_cut, k, gb)

        print(f"\n  総LHS (z=0): {total_lhs:.6e}")
        print(f"  |LHS|: {abs(total_lhs):.6e}")
        print(f"  格子点数（全体）: {len(terms_list)}")

        # ---- ブロックペア別の寄与 ----
        print(f"\n  ブロックペア別LHS寄与:")
        for pair, val in lhs_by_pair.items():
            print(f"    ブロック({pair[0]+1},{pair[1]+1}): {val:.6e}  |val|={abs(val):.3e}")
        print(f"    合計: {sum(lhs_by_pair.values()):.6e}")

        # ---- 相殺ペアの解析 ----
        if k <= 3:
            cancellation_pairs, residual_terms = analyze_pairing_structure(
                Om0, E, z, N_cut, k, gb, terms_list)

            n_cancelled = sum(1 for p in cancellation_pairs if p['cancelled'])
            n_pairs = len(cancellation_pairs)
            n_residual = len(residual_terms)

            print(f"\n  相殺ペア解析:")
            print(f"    ペア数: {n_pairs}")
            print(f"    完全相殺したペア: {n_cancelled}/{n_pairs}")
            print(f"    未ペア（残差）の格子点数: {n_residual}")

            if residual_terms:
                print(f"\n  残差格子点の例（最大5件）:")
                residual_sorted = sorted(residual_terms,
                                        key=lambda t: abs(t['contrib']), reverse=True)
                for t in residual_sorted[:5]:
                    print(f"    n={t['n']}, |contrib|={abs(t['contrib']):.3e}, "
                          f"nEn={t['nEn']:.4f}")

            if k == 3 and n_pairs > 0:
                print(f"\n  k=3のパターン別相殺数:")
                for pat in [1, 2, 3]:
                    count = sum(1 for p in cancellation_pairs if p['pattern'] == pat)
                    cancelled = sum(1 for p in cancellation_pairs
                                   if p['pattern'] == pat and p['cancelled'])
                    print(f"    パターン{pat}: {count}ペア, うち{cancelled}個が完全相殺")

        # ---- n/−n対称性の直接確認 ----
        print(f"\n  n/−n対称性の確認（nEnの反転）:")
        n0 = tuple([0]*g)
        nEn_pairs = []
        seen_check = set()
        for t in terms_list:
            n = t['n']
            if n in seen_check:
                continue
            neg_n = tuple(-x for x in n)
            if neg_n in {t2['n'] for t2 in terms_list} and neg_n != n:
                nEn_n = t['nv'] @ E @ t['nv']
                # n→-nでnEnは不変（証明済み）
                nv_neg = -t['nv']
                nEn_neg = nv_neg @ E @ nv_neg
                seen_check.add(n)
                seen_check.add(neg_n)
                nEn_pairs.append((nEn_n, nEn_neg))

        if nEn_pairs:
            all_same = all(abs(a - b) < 1e-12 for a, b in nEn_pairs[:20])
            print(f"    nEn(n) = nEn(-n) が全ペアで成立: {all_same}")
            print(f"    確認ペア数: {min(len(nEn_pairs), 20)}")

        # ---- k=3での新しい相殺パターンの探索 ----
        if k == 3:
            print(f"\n  k=3: 4点セット {{(nA,nB,nC), (nA,-nB,nC), (-nA,nB,nC), (-nA,-nB,nC)}} の和")
            g1 = gb

            group_sums = []
            seen_4pt = set()
            for t in terms_list:
                n = t['n']
                if n in seen_4pt:
                    continue
                nv = t['nv']
                nA, nB, nC = nv[:g1], nv[g1:2*g1], nv[2*g1:]

                # 4点セット
                patterns = [
                    tuple(list(nA.astype(int))    + list(nB.astype(int))    + list(nC.astype(int))),
                    tuple(list(nA.astype(int))    + list((-nB).astype(int)) + list(nC.astype(int))),
                    tuple(list((-nA).astype(int)) + list(nB.astype(int))    + list(nC.astype(int))),
                    tuple(list((-nA).astype(int)) + list((-nB).astype(int)) + list(nC.astype(int))),
                ]
                terms_dict_local = {t2['n']: t2 for t2 in terms_list}

                group_sum = 0j
                found_all = True
                for p in patterns:
                    neg_p = tuple(-x for x in p)
                    # 半格子の代表元（最初の非ゼロが正）を考慮
                    rep = p
                    if p in terms_dict_local:
                        group_sum += terms_dict_local[p]['contrib']
                    elif neg_p in terms_dict_local:
                        group_sum += terms_dict_local[neg_p]['contrib']
                    else:
                        found_all = False

                if found_all:
                    for p in patterns:
                        seen_4pt.add(p)
                        seen_4pt.add(tuple(-x for x in p))
                    group_sums.append(abs(group_sum))

            if group_sums:
                print(f"    4点セット数: {len(group_sums)}")
                print(f"    |4点和|の平均: {np.mean(group_sums):.3e}")
                print(f"    |4点和|の最大: {np.max(group_sums):.3e}")
                cancelled_4pt = sum(1 for s in group_sums if s < 1e-15)
                print(f"    4点和≈0のセット: {cancelled_4pt}/{len(group_sums)}")

    # ---- 定量的比較: k=2 vs k=3 でのLHS残差の起源 ----
    print(f"\n{'='*65}")
    print("定量的比較: 残差の起源")
    print(f"{'='*65}")

    for k, g in [(2, 4), (3, 6)]:
        Om0, gb = make_omega_kblock(g, k, lmin, seed=0)
        E = make_Eoff_k(g, k, seed=0)

        # z=0でLHS
        total, _, _ = analyze_cancellation_by_block_pair(
            Om0, E, np.zeros(g), N_cut, k, gb)
        print(f"\n  k={k}, g={g}: |LHS(z=0)| = {abs(total):.6e}")

        # z=0でのtheta_naiveの一次微分を解析式で確認
        # d/dδ theta_naive|_{δ=0} = Σ_n πi·nEn·exp(πinᵀΩn)
        # Re(Ω)=0のとき exp(πinᵀΩn) = exp(-πnᵀIm(Ω)n) は実数
        # → d/dδ theta_naive = πi · Σ_n nEn·exp(-πnᵀIm(Ω)n) は純虚数
        lhs_imag_sum = 0.0
        for n in product(range(-N_cut, N_cut+1), repeat=g):
            nv = np.array(n, dtype=float)
            nEn = nv @ E @ nv
            exp_val = np.exp(-np.pi * nv @ Om0.imag @ nv)
            lhs_imag_sum += nEn * exp_val

        print(f"  Σ_n nEn·exp(-πnᵀIm(Ω)n) = {lhs_imag_sum:.6e}")
        print(f"  → d/dδ theta_naive(z=0) = πi·{lhs_imag_sum:.3e} = {np.pi*1j*lhs_imag_sum:.6e}")
        print(f"  → d/dδ theta_skk(z=0)  = 0 (off-diagonal不変性)")
        print(f"  → LHS = πi·{lhs_imag_sum:.3e}")
        print(f"  → |LHS| = π·|{lhs_imag_sum:.3e}| = {np.pi*abs(lhs_imag_sum):.6e}")
        print(f"  一致: {abs(abs(total) - np.pi*abs(lhs_imag_sum)) < 1e-12}")


if __name__ == "__main__":
    main()
