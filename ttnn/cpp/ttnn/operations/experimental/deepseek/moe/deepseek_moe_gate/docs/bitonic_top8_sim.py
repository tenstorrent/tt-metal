# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""
Python equivalent of part 1 of _deepseek_moe_gate_sum_top2(): one group of 32 experts -> sorted top-8.

Reference C++:
  ttnn/cpp/ttnn/operations/experimental/deepseek/moe/deepseek_moe_gate/device/kernel_includes/
      tt_llk/tt_llk_blackhole/common/inc/sfpu/ckernel_sfpu_deepseek_moe_gate_topk_single_face.h
Write-up: docs/bitonic_top8.md

Usage:
  python3 bitonic_top8_sim.py                  # key = 1..32 in order
  python3 bitonic_top8_sim.py shuffle          # key = a fixed permutation of 1..32 (non-degenerate merge)
  python3 bitonic_top8_sim.py random [SEED]    # key = a random permutation of 1..32; the seed is
                                               # printed so a run can be reproduced

Scope of the model
------------------
On hardware the SFPU's 32 lanes are 4 rows x 8 columns, and the 8 lane columns are 8 expert groups
processed in parallel. Only 1 lane (= 1 group) is modelled here, since lanes are fully independent
in part 1. So one LREG holds only 4 values for this lane (one per row), and 4 LREGs = a 16-element
sequence.

  key[L][r]  <-> LREG0-3   sort key = score + bias (integers 1..32 here, so no ties)
  pay[L][r]  <-> LREG4-7   payload (on HW this is idx|score packed in one 32-bit reg; just the
                           expert id here)

Coordinate systems (they decide the compare distance a SFPSWAP expresses):
  canonical  (transposed=False): sequence position p = 4*L + r  -> adjacent LREGs = distance 4
  transposed (transposed=True) : sequence position p = 4*r + L  -> adjacent LREGs = distance 1
SFPTRANSP is exactly what switches between the two (a 4x4 transpose of LREG0-3 and of LREG4-7).
"""

import random
import sys

NREG = 4  # LREG0-3
NROW = 4  # 4 SFPU rows

# ---------------------------------------------------------------- SFPSWAP modes
# A list of 4 flags: True means "the first operand takes the max" on that row.
ALL_ROWS_MAX = [True, True, True, True]
ROWS_01_MAX = [True, True, False, False]  # rows 0/1 descending, rows 2/3 ascending
ROWS_02_MAX = [True, False, True, False]  # rows 0/2 descending, rows 1/3 ascending
UNCONDITIONALLY = None  # unconditional exchange, no compare


class SFPU:
    """4 key registers + 4 payload registers, index tracking always on."""

    def __init__(self):
        self.key = [[0] * NROW for _ in range(NREG)]
        self.pay = [[-1] * NROW for _ in range(NREG)]
        self.transposed = False
        self.step = 0

    # ------------------------------------------------------------ instructions
    def sfptransp(self):
        """TTI_SFPTRANSP: 4x4 transpose of LREG0-3 and LREG4-7; flips the coordinate system."""
        self.key = [[self.key[L][r] for L in range(NREG)] for r in range(NROW)]
        self.pay = [[self.pay[L][r] for L in range(NREG)] for r in range(NROW)]
        self.transposed = not self.transposed

    def sfpswap(self, a, b, mode):
        """TTI_SFPSWAP(0, LREGa, LREGb, mode): per-row compare-exchange, payload follows."""
        for r in range(NROW):
            ka, kb = self.key[a][r], self.key[b][r]
            if mode is UNCONDITIONALLY:
                swap = True
            else:
                swap = (ka < kb) if mode[r] else (ka > kb)
            if swap:
                self.key[a][r], self.key[b][r] = kb, ka
                self.pay[a][r], self.pay[b][r] = self.pay[b][r], self.pay[a][r]

    # ------------------------------------------------------------ printing
    def seq(self):
        """Decode the 4x4 into the 16-element logical sequence [(expert_id, key), ...]."""
        out = [None] * 16
        for L in range(NREG):
            for r in range(NROW):
                p = (4 * r + L) if self.transposed else (4 * L + r)
                out[p] = (self.pay[L][r], self.key[L][r])
        return out

    def dump(self, title):
        self.step += 1
        coord = "transposed (p=4r+L)" if self.transposed else "canonical  (p=4L+r)"
        print(f"\n[{self.step:02d}] {title}    <{coord}>")
        for L in range(NREG):
            cells = "  ".join(f"e{self.pay[L][r]:02d}:{self.key[L][r]:2d}" for r in range(NROW))
            print(f"      LREG{L} | {cells}")
        print("      seq   = " + " ".join(f"{v:2d}" for _, v in self.seq()))


# ---------------------------------------------------------------- sort primitives
def ph0_st1_to_1(u: SFPU, start_transpose, end_transpose, verbose=True):
    """Phase 0: build sorted pairs; the two pairs run opposite ways -> bitonic sequence of 4."""
    if start_transpose:
        u.sfptransp()
    u.sfpswap(0, 1, ALL_ROWS_MAX)  # pair (4r+0, 4r+1) descending
    u.sfpswap(3, 2, ALL_ROWS_MAX)  # operands written backwards -> pair (4r+2, 4r+3) ascending
    if end_transpose:
        u.sfptransp()
    if verbose:
        u.dump("ph0  step1(dist 1)   per 4-block: [desc, desc, asc, asc]")


def ph1_st2_to_1(u: SFPU, start_transpose, end_transpose, verbose=True):
    """Phase 1: bitonic merge of size 4; adjacent 4-blocks alternate direction (ROWS_02_MAX)."""
    if start_transpose:
        u.sfptransp()
    u.sfpswap(0, 2, ROWS_02_MAX)  # step2: dist 2
    u.sfpswap(1, 3, ROWS_02_MAX)
    u.sfpswap(0, 1, ROWS_02_MAX)  # step1: dist 1
    u.sfpswap(2, 3, ROWS_02_MAX)
    if end_transpose:
        u.sfptransp()
    if verbose:
        u.dump("ph1  step2,1(dist 2,1)   4-blocks sorted, directions desc/asc/desc/asc")


def ph2_st3_to_1(u: SFPU, end_transpose, bitonic=True, verbose=True):
    """Phase 2: merge of size 8. bitonic=True -> the two 8-blocks run opposite ways (feeds ph3);
    bitonic=False -> both descending (two independently sorted 8-blocks)."""
    u.sfpswap(0, 1, ALL_ROWS_MAX)  # step3: dist 4 (adjacent LREGs in canonical)
    if bitonic:
        u.sfpswap(3, 2, ALL_ROWS_MAX)  # second 8-block ascending
    else:
        u.sfpswap(2, 3, ALL_ROWS_MAX)  # second 8-block descending as well
    u.sfptransp()
    mode = ROWS_01_MAX if bitonic else ALL_ROWS_MAX
    u.sfpswap(0, 2, mode)  # step2: dist 2
    u.sfpswap(1, 3, mode)
    u.sfpswap(0, 1, mode)  # step1: dist 1
    u.sfpswap(2, 3, mode)
    if end_transpose:
        u.sfptransp()
    if verbose:
        tail = "first 8 desc + last 8 asc = bitonic sequence of 16" if bitonic else "both 8-blocks descending"
        u.dump(f"ph2  step3,2,1(dist 4,2,1)   {tail}")


def top8_ph3_st4_to_1(u: SFPU, descending, end_transpose, verbose=True, label=None):
    """Phase 3: bitonic merge of 16, trimmed for top-8 (the discarded half is not fully sorted)."""
    if descending:  # SortDir::ArgMax
        u.sfpswap(0, 2, ALL_ROWS_MAX)  # step4: dist 8 -> top-8 lands in L0/L1
        u.sfpswap(1, 3, ALL_ROWS_MAX)
        u.sfpswap(0, 1, ALL_ROWS_MAX)  # step3: upper half only, L2/L3 gets thrown away anyway
        u.sfptransp()
        u.sfpswap(0, 2, ALL_ROWS_MAX)  # step2
        u.sfpswap(1, 3, ALL_ROWS_MAX)
        u.sfpswap(0, 1, ALL_ROWS_MAX)  # step1
        u.sfpswap(2, 3, ALL_ROWS_MAX)
    else:  # SortDir::ArgMin: every operand pair written backwards -> ascending
        u.sfpswap(2, 0, ALL_ROWS_MAX)
        u.sfpswap(3, 1, ALL_ROWS_MAX)
        u.sfpswap(3, 2, ALL_ROWS_MAX)
        u.sfptransp()
        u.sfpswap(2, 0, ALL_ROWS_MAX)
        u.sfpswap(3, 1, ALL_ROWS_MAX)
        u.sfpswap(1, 0, ALL_ROWS_MAX)
        u.sfpswap(3, 2, ALL_ROWS_MAX)
    if end_transpose:
        u.sfptransp()
    if verbose:
        half = "L0/L1 (desc)" if descending else "L2/L3 (asc)"
        u.dump(label or f"ph3  step4..1(dist 8,4,2,1)   sorted top-8 in {half}; the other half is dropped")


def reverse_sort_order(u: SFPU, verbose=True):
    """Reverse the row order inside each LREG (only a full 8-element reversal when combined with
    loading the two halves at swapped addresses)."""
    u.sfptransp()
    u.sfpswap(0, 3, UNCONDITIONALLY)
    u.sfpswap(1, 2, UNCONDITIONALLY)
    u.sfptransp()
    if verbose:
        u.dump("reverse_sort_order   whole run reversed")


# ---------------------------------------------------------------- DEST model
class DEST:
    """Only what a single lane sees: DEST[tile][col] = 4 values (4 rows)."""

    def __init__(self):
        self.bias = {}  # tile2: key = score + bias (the sort key)
        self.interm = {}  # tile3: payload scratch

    def load16(self, u: SFPU, base):
        """bitonic_topk_load16_concat_indices_single_face<offset=base>: addresses base+{0,4,8,12}"""
        for L, col in enumerate([base + 0, base + 4, base + 8, base + 12]):
            u.key[L] = list(self.bias[col])
            u.pay[L] = list(self.interm[col])
        u.transposed = False

    def store8(self, u: SFPU):
        """bitonic_topk_store8_...: store the top-8 only (L0 -> col 0, L1 -> col 4)"""
        self.bias[0], self.interm[0] = list(u.key[0]), list(u.pay[0])
        self.bias[4], self.interm[4] = list(u.key[1]), list(u.pay[1])

    def load8(self, u: SFPU):
        """bitonic_topk_load8_even_cols_...: read that run back into L0/L1 (L2/L3 left untouched)"""
        u.key[0], u.pay[0] = list(self.bias[0]), list(self.interm[0])
        u.key[1], u.pay[1] = list(self.bias[4]), list(self.interm[4])


# ---------------------------------------------------------------- main flow
def sum_top2_one_group(keys):
    """One group of 32 experts -> sorted top-8 + top-2 sum. keys[i] = expert i's score+bias."""
    assert len(keys) == 32
    u, d = SFPU(), DEST()

    # Initial DEST content: even columns {0,4,8,12} = the first 16 experts, odd columns
    # {2,6,10,14} = the last 16. (The physical mapping on HW comes from the unpack transpose; it
    # does not affect the algorithm, it only decides which expert goes into which pass.)
    for i, col in enumerate([0, 4, 8, 12, 2, 6, 10, 14]):
        d.bias[col] = list(keys[4 * i : 4 * i + 4])
        d.interm[col] = list(range(4 * i, 4 * i + 4))

    print("=" * 92)
    print("input: key = score + bias for 32 experts   (format eXX:KK = expert id : key)")
    print("  " + " ".join(f"e{i:02d}:{v:2d}" for i, v in enumerate(keys)))
    print("=" * 92)

    # ---- (a) the even 16 -> descending, keep the top-8 only ------------------
    print("\n########## (a) even columns: 16 values -> full bitonic sort (descending), store top-8 ##########")
    d.load16(u, base=0)
    u.dump("load16<offset=0>   L0-3 <- bias{0,4,8,12}, L4-7 <- idx")
    ph0_st1_to_1(u, start_transpose=True, end_transpose=False)
    ph1_st2_to_1(u, start_transpose=False, end_transpose=True)
    ph2_st3_to_1(u, end_transpose=True, bitonic=True)
    top8_ph3_st4_to_1(u, descending=True, end_transpose=True)
    d.store8(u)
    print("      -> store8: L0 -> col 0, L1 -> col 4  (top-8 only, the other 8 are dropped)")

    # ---- (b) the odd 16 -> ascending ----------------------------------------
    print("\n########## (b) odd columns: 16 values -> full bitonic sort (ascending, !idir) ##########")
    d.load16(u, base=2)
    u.dump("load16<offset=2>   L0-3 <- bias{2,6,10,14}")
    ph0_st1_to_1(u, start_transpose=True, end_transpose=False)
    ph1_st2_to_1(u, start_transpose=False, end_transpose=True)
    ph2_st3_to_1(u, end_transpose=True, bitonic=True)
    top8_ph3_st4_to_1(u, descending=False, end_transpose=True)
    print("      -> the last 8 of the ascending result (L2/L3) are this half's top-8; left in place")

    # ---- (c) merge the two runs ---------------------------------------------
    print("\n########## (c) merge: desc 8 (L0/L1) + asc 8 (L2/L3) = bitonic 16 -> top-8 of all 32 ##########")
    d.load8(u)
    u.dump("load8   read (a)'s descending run back into L0/L1; L2/L3 still hold (b)'s ascending run")
    top8_ph3_st4_to_1(u, descending=True, end_transpose=True, label="ph3 merge   -> L0/L1 = top-8 of the group (desc)")

    top8 = u.seq()[:8]
    print("\n      >>> top-8 = " + " ".join(f"e{i:02d}:{v:2d}" for i, v in top8))

    # ---- tail: top-2 sum + broadcast ----------------------------------------
    print("\n########## tail: top-2 sum (TRANSP / SFPADD / TRANSP) + broadcast down the column ##########")
    u.sfptransp()
    top2_sum = u.key[0][0] + u.key[1][0]  # SFPADD(L0 = L0 + L1): row 0 = rank0 + rank1
    u.sfptransp()
    print(f"      top2_sum = {top8[0][1]} + {top8[1][1]} = {top2_sum}  (broadcast to interm+0/+4)")

    return top8, top2_sum


def make_keys(argv):
    """Build the 32 keys. Always a permutation of 1..32, so there are never ties."""
    mode = argv[1] if len(argv) > 1 else "ordered"

    if mode == "ordered":
        return list(range(1, 33))  # expert i has key i+1

    if mode == "shuffle":
        # A fixed permutation, so the top-8 of the two halves interleave and the merge step is
        # not degenerate.
        return [(i * 13 + 5) % 32 + 1 for i in range(32)]

    if mode == "random":
        seed = int(argv[2]) if len(argv) > 2 else random.randrange(1_000_000)
        random.seed(seed)
        keys = list(range(1, 33))
        random.shuffle(keys)
        print(f"random permutation, seed = {seed}   (reproduce with: {argv[0]} random {seed})")
        return keys

    sys.exit(f"unknown mode {mode!r}; expected one of: ordered | shuffle | random [SEED]")


def main():
    keys = make_keys(sys.argv)
    top8, top2_sum = sum_top2_one_group(keys)

    print("\n" + "=" * 92)
    ref = sorted(range(32), key=lambda i: -keys[i])[:8]
    got = [i for i, _ in top8]
    print("reference top-8 (sorted) :", " ".join(f"e{i:02d}:{keys[i]:2d}" for i in ref))
    print("bitonic network output   :", " ".join(f"e{i:02d}:{v:2d}" for i, v in top8))
    print("match                    :", got == ref)
    print(f"top2_sum: network={top2_sum}  reference={keys[ref[0]] + keys[ref[1]]}")
    print("=" * 92)


if __name__ == "__main__":
    main()
