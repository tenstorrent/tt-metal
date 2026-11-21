# whole_k_rowstream_v2 (Perf 19): the whole-K row stream on the round-19 tree, without a store pacer

Perf 18's whole-K row-streamed walk (`../whole_k_rowstream/`), ported onto the current working tree (C_out fold, ring
startup split, kh tile reuse) with no wall-clock pacer. Per batch of `rows` tile-rows, compute tilizes the batch's
units, accumulates all of K in DEST against the resident weights, adds the bias and packs once. There are no fp32
partials and no per-K-block packs. The output leaves at the compute rate instead of in the last K-block.

## Change (`integration.patch`, op_base -> op_copy)
- **Host.**
  - `rowstream_plan` re-spends the partials' and in0's L1 on in0 `[rows, K]`, a unit FIFO lag, ring growth and a
    deeper output CB, as Perf 18 did. The blocking solve is unchanged.
  - New: the domain rule `ROWSTREAM_RULE = "store_rate"`, below.
  - New: `ROWSTREAM_KHR`. `"replace"` (the default) means select_blocking does not take kh reuse where the row stream
    applies.
  - New: `rowstream_khr_plan`, the `"fuse"` option.
  - The pacer is gone.
- **Compute (`conv2d_3x3_rowstream.hpp`).** `walk` takes an in0 policy:
  - `BlockIn0`: the batch's block at the CB read pointer.
  - `RingIn0`: kh reuse's tile ring, ring row t + kh W/32, batches flat across M-blocks.
  - Bias: `batch` (fp32 sums, then the row-broadcast add per batch; the default) or `kstep` (a ones-column tile times
    the bias tile as K-step 0).
- **Reader.**
  - Tile-row-major units.
  - Per tile-row, the ring is released keyed on the FIFO depth.
  - New `RmBankRing::need_stream`: the wait that issues whole chunks and handles the RM_SPLIT window. Perf 18's
    reader was not split-aware, and the split now applies to every ring window >= 512 KB.
  - The ones tile is written before kh reuse's unit stream.
- **Writer.** Bank-covering store windows of row groups, popped one row group at a time. No pacing.
- **Reach.** Everything sits under `CONV2D_3X3_ROWSTREAM`, so other programs are unchanged: P1 gated off is the same
  program, bit-identical. It reaches only the DRAM-interleaved ROW_MAJOR ring with resident weights. That excludes the
  C_out fold, TILE input, sharded input (so no Panoptic program) and one M-block per core.

## Domain rule: the output stream rate, not the roofline
The row stream emits one output tile per K tile-MACs on every work core at once. That gives a stream rate of:

    cores x out_tile_bytes x clock / (K x 64)

which is 4.75e12 / K B/s for bf16 output on 110 cores at HiFi4. The rule applies the row stream only where this rate
is at most the DRAM bandwidth (450 GB/s). Past that rate, every core stores continuously, and NoC1 starves the far
columns, as Perf 18 found.

The rule depends only on K (C_in), so its margins are wide: 528 GB/s at C_in 32 against 264 GB/s at C_in 64, a factor
of 2.

The whole-op roofline would be the wrong rule. 1026x2048 64->64 has DRAM time 1.17x its matmul time, yet it wins 7.5%.

| side | case | output stream | base -> row stream (us) |
|---|---|---|---|
| out (> 450 GB/s) | P1 1026x2048 32->128 | 528 GB/s | 1833 -> 2891 (+58%) |
| out | P4 514x1024 32->128 | 528 | 598 -> 744 (+24%) |
| in | r4 1026x2048 64->64 (roofline 1.17) | 264 | 1728 -> 1599 (-7.5%) |
| in | r5 1026x2048 64->96 (roofline 0.98) | 264 | 2324 -> 2153 (-7.4%) |
| in | P5 / P6 / P2 / P8 (C_in 64) | 259-264 | -4.9% to -7.3% |
| in | P12 / P13 / P7 / P10 / P14 (C_in 128) | 130-132 | -10.1% to +1.9% |

## Results
Setup: Blackhole p150b, 110 cores, bf16, HiFi4, fp32 DEST, bias, padding (0, 0, 1, 1), device kernel us, one call per
run. Recommended configuration (descriptor defaults, variant `rsr` / `rspb`): batch bias, the row stream replacing kh
reuse, the store-rate rule.

| idx | case | calls | base | candidate | delta | runs |
|---|---|---|---|---|---|---|
| 47 P13 | 514x1024 128->512 | 1 | 5335.4 | 4796.6 | -10.1% | b3 |
| 45 P12 | 514x1024 128->256 | 1 | 2756.2 | 2483.9 | -9.9% | b4 |
| 46 P6 | 1026x2048 64->256 | 1 | 5284.0 | 4898.8 | -7.3% | b3 |
| 44 P5 | 1026x2048 64->128 | 1 | 2893.1 | 2685.2 | -7.2% | b2 |
| 37 P8 | 258x512 64->256 (base: kh reuse) | 5 | 398.7 / 398.4 | 371.1 / 370.4 | -7.0% | b2, f1 |
| 42 P2 | 514x1024 64->256 (base: kh reuse) | 6 | 1385.7 / 1386.9 | 1317.6 / 1319.3 | -4.9% | b2, f1 |
| 43 P7 | 258x512 128->512 | 8 | 1309.3 | 1303.9 | -0.4% (flat) | b2 |
| 38 P10 | 130x256 128->512 (base: kh reuse) | 17 | 374.7 / 374.6 / 374.6 | 376.8 / 377.0 / 377.1 | +0.6% (flat) | b1, b2, f1 |
| 40 P14 | 258x512 128->256 | 1 | 745.7 / 745.9 / 746.4 | 760.8 / 760.6 / 759.0 | **+2.0%** (median of 3) | b1, b2, f1 |
| 39 P11 | 514x1024 64->128 (base: kh reuse) | 1 | 748.8 / 757.0 | 799.3 / 789.1 | **+4.2..+6.7%** | b1, b4 |
| 41 P1, 36 P4 | C_in 32 | 10 | | | gated off: same program (f1: P1 bit-identical) | |
| 35 P3, 33 P9 | C_out fold | 2 | | | not reached (the fold keeps its path) | |
| 34, 32, 31 | one M-block per core | 12 | | | not reached | |

- **Call-weighted over the 17:** 61.13 -> 59.23 ms (-3.1%). Keeping kh reuse where it applies gives 59.69 ms (-2.3%).
- **Extra regimes, all 258x512 64->256:**
  - padding 1, no bias, ReLU (r2): 401.3 -> 364.2 (-9.3%).
  - padding 1, bias, ReLU (r3): 399.4 -> 373.0 (-6.6%).
  - Both have PCC equal to base.

## Menu (us; base in the same run)
| variant | P13 | P12 | P6 | P5 | P2 | P8 | P7 | P10 | P11 | P14 |
|---|---|---|---|---|---|---|---|---|---|---|
| base | 5334-5335 | 2746-2756 | 5283-5284 | 2891-2893 | 1386-1387 | 398-399 | 1309-1310 | 374-375 | 749-761 | 745-746 |
| rspb: batch bias, replace kh reuse (recommended) | 4797 | 2484 | 4899 | 2685 | 1318-1319 | 370-371 | 1304 | 377 | 789-799 | 759-761 |
| rsp: kstep, replace | 4850 | 2515 | 4975 | 2724 | 1332-1336 | 383.5 | 1308 | 385 | 795-796 | 763-764 |
| rs: kstep, fused over kh reuse's ring | - | - | - | - | 1332-1337 | 385-386 | - | 391-392 | 779 | - |
| rsb: batch, fused | - | - | - | - | - | - | - | 386 | 764 | - |
| keep kh reuse (rsx / rsxb) | rs | rs | rs | rs | = base | = base | rs | = base | = base | rs |

Notes on the menu:
- The batch bias is faster than the K-step everywhere measured: -1.4% (P2, P5) to -3.9% (P8, P10). The K-step adds one
  HiFi4 tile-MAC per output tile, which is 1/K more matmul.
- Fusing with kh reuse's tile ring is bit-identical to the plain-ring row stream. It is slower than the plain ring on
  P10, and still slower than kh reuse's K-blocked walk on P10 and P11.

## Precision (same config everywhere)
- **batch (recommended):**
  - PCC vs torch over the whole output is 0.9999986, equal to base on every case.
  - 0.007-0.016% of outputs differ from base, by at most 1 bf16 ulp where |ref| >= mean|ref| / 16.
  - The larger ulp distances are near-zero values, including 0-14 sign flips per case.
  - The worst error in reference ulps equals base's on every case.
- **kstep:**
  - PCC is equal.
  - About 2.8% of outputs differ from base, all by at most 1 ulp where significant.
  - kstep is closer to the fp32 reference: its worst error in reference ulps is below base's on every case (P8 0.647 vs
    0.682, P13 0.665 vs 0.707).
  - The brief prefers the variant closest to fp32, so kstep is the alternative at +1.4..+3.9% time.
- **No bias (r2):** about 4.5% of outputs differ, and the worst error is 0.863 vs base 0.867 reference ulps. Base
  reloads its partials through srcA (tf32).

## Domain
- **Applies to:** the DRAM-interleaved ROW_MAJOR ring walk with resident weights, where the output stream rate is at
  most the DRAM bandwidth. That is every C_in >= 64 ring case at bf16, HiFi4, 110 cores. It replaces kh reuse there.
  - Covered: with or without bias, ReLU, padding (0, 0, 1, 1) or 1, with or without the N split, with or without the
    ring startup split.
- **Measured regressions inside it:**
  - P11 (+4.2% / +6.7%, 2 runs): kh reuse's K-blocked walk beats every row-stream variant there. Fused + batch bias is
    still +2.1%.
  - P14 (+2.0%, median of 3; the K-step is +2.4%).
  - No geometric rule found for either. kh reuse's tilize saving should scale with 1/C_out, and P11 is the only kh
    reuse case at C_out 128, but one point does not make a rule.
- **Flat:** P7, P10.
- **Excluded by the rule, measured:** P1, P4 (C_in 32).
- **Not reached:** the C_out fold (P3, P9; the fold is far faster there than the row stream measured in Perf 18), one
  M-block per core, TILE input, sharded input.
- **Untested:** bf8_b weights or output (the K-step needs bf16 in0 and the bias in the weights' format, otherwise it
  falls back to the batch bias), dilation > 1, W not a multiple of 32.

## Raw LLK / helpers bypassed
`compute_kernel_lib::matmul_block` is bypassed for capability reasons:
- it accumulates K-blocks through packer L1 accumulation into a partials CB;
- it has no per-K-step in0 source: the K-step reads the ones CB, and the fused variant reads the tile ring by row.

The walk uses `ckernel::matmul_block` / `matmul_block_init` and `compute_kernel_lib::pack_subblock_row_strided`, as the
K-blocked walk does.

The batch bias pass uses `add_bcast_rows_init` / `add_tiles_bcast_rows`, waiting and popping per row group.
`compute_kernel_lib::add_bias_bcast_rows` with a per-batch shape is probably expressible (ergonomics). It was not
tried.

The reader's ring fill keeps the base's raw NoC de-interleave. `need_stream` is a new wait mode on it.

## Re-run
- `run.sh <label> <cases> <variants> [ENV=...]`:
  - runs in the foreground;
  - stops with exit 75 if the device lock is not acquired within `LOCK_WAIT_S`.
- Variants and regimes are defined in `test_rs.py`.
- `python3 -I summ.py <label>` summarizes PCC, ulp and the plan per case.
- `python3 -I rs_plan.py op_copy` prints the host plan and the output stream rate per focus case.
