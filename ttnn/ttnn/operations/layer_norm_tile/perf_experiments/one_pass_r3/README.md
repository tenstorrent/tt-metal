# one_pass_r3: delete the two-pass `row_stats_block` on the R3 (unbalanced) split

Copy of the op at 9daa4c30ecf in `ln_variant/` (real op untouched). Knobs added to the copy's descriptor:
- `ONE_PASS_ALL`: one-pass variance wherever fp32 DEST + HiFi4, also on splits the op keeps resident. There
  the stats window is the whole row (G = Wt, no partial round trips) and normalize stays whole-row.
- `SHIFT_MODE` (compute CT arg 23, one-pass + residual only): 0 = shifted d = (x - x[:,0]) + res (op today),
  1 = unshifted z with the op's pairwise S1, 2 = unshifted with symmetric S1 = sum(z_A + z_B).

Files: `test_one_pass_r3.py` (perf rows with the golden gate, ported large-mean and constant-row tests),
`bench.sh <label> <variants> <cases>` (one profiled pass, results in `logs/bench.txt`, device log in
`logs/<label>.device.csv`), `rowzones.py` (per-class zones + latest-core timeline), `row1.py` (heavy cores'
last, resident row: TRISC_1 zone medians).

## Results (BH p100a, device kernel us)

| variant | focus1 | focus2 | focus3 | large mean, residual fp32 |
|---|---|---|---|---|
| base (resident two-pass) | 35.30 (35.24-35.42) | 47.96 (47.56-48.16) | 33.34 (33.06-33.59) | rel 0.0203 (unbalanced), 0.0021 (balanced, shifted) |
| shift (resident shifted one-pass) | 36.49 (36.43-36.78) | 47.73 (46.80-48.88) | 33.39 (32.94-33.45) | rel 0.0021 |
| sym / unshift (no shift) | 35.81 (1 run) | 48.10 | 33.20 | INCORRECT: pcc 0.705, rel 316 (variance clamped to 0) |

Other rows, one run (base / shift / sym): allL1_res 24.89 / 25.60 / 24.62, L1_nores 20.79 / 20.60 / 21.11,
8x384 36.70 / 36.36 / 36.00, 32x32 18.70 / 18.72 / 17.89, affTILE 48.06 / 47.73 / 48.18, focus1 bf16 35.25 /
35.61 / 35.11, 3x500 18.66 / 18.79 / 17.88, 64x17 21.22 / 21.22 / 21.30, 2x512 13.47 / 13.71 / 13.55.
`shift` leaves L1_nores, 8x384, 32x32, 3x500, 64x17 and bf16 on the base code, so those deltas are noise.
`sym` does change 8x384 / 32x32 / 3x500 (balanced streamed residual rows): -0.7..-0.8 us, which is the cost of
the shift pass there. That variant is incorrect, so the number is not usable.

Heavy cores' last (resident) row, TRISC_1 zones, us:

| row | base: add_and_stats + row_stats | one-pass add_and_stats | row sum base -> one-pass |
|---|---|---|---|
| focus1 / allL1_res (residual) | 1.92 + 2.25 = 4.17 | shift 5.28, sym 4.05 | 8.66 -> 9.92 (shift), 8.66 (sym) |
| focus3 (no residual) | 0.73 + 2.27 = 3.00 | 2.69 | 7.39 -> 7.22 |

- The 24 HiFi4 squares cost ~1.9-2.0 us per row in either place, so one-pass only drops the separate
  acquire / collapse / rsqrt (~0.3 us). normalize is ~0.1 us longer in one-pass because nothing else separates
  it from center.
- The shifted residual add (sub col-bcast + DEST-reuse add) costs ~1.1 us per resident row.
- Heavy row 0 waits for the whole row to land (~16 us), so one-pass does not move it earlier either: normalize
  starts at 20.7 (base) vs 20.8 (shift).

Precision: symmetric S1 gives the same clamped variance as pairwise S1. The S2 bias is therefore not an S1/S2
operand mismatch between SrcA and SrcB; it is in the squares themselves. Unverified hypothesis: the HiFi4
product of fp32-page (tf32-in-Src) operands is biased low by ~2^-12 relative, larger than var / mean^2 =
2e-4 at |mean| = 100. Only shifting the operands to O(std) fixes it.

Verdict: REGRESSION on focus1 (shift +1.2 us), flat on focus2 / focus3. The incorrect no-shift variant is not a
win either.
