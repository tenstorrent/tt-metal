# post_stats_chain

Idea: cut the per-row math cost of the resident whole-row chain center -> row_stats_block -> normalize ->
affine_scale -> affine_shift (two-pass path, the R3 heavy cores on focus1) at the same precision contract
(fp32 DEST, HiFi4, approx off, bf16 I/O). BH p100a, base = 9daa4c30ecf.

## Files
- `chain_variants.hpp`, `bench_kernels/`, `test_bench.py`, `zones.py`: isolated one-core bench, 16 iterations of
  one 24-tile row (fp32 z = x + res, fp32 mean, bf16 gamma/beta tiles, bf16 out). Correctness vs float64 torch.
  `PSC_VARIANTS=0,1,2 scripts/run_safe_pytest.sh --run-all --profile <dir>/test_bench.py::test_timed --import-mode=prepend -s`
  then `python3 zones.py` (per-TRISC zone means, iteration 0 skipped).
- `op_kernels/base` (op kernels at HEAD), `op_kernels/v2` (compute changed only), `test_whole_op.py`, `whole.sh <variant> [-k]`,
  `percore.py <dev.csv>`, `swap_kernel.py` (pytest plugin: `LN_VARIANT=v2 PYTHONPATH=<dir> ... -p swap_kernel`).
- `graduate/compute.diff`: base -> v2 on `kernels/layer_norm_tile_compute.cpp` (the only file to change).

## Mechanism (bench, math thread TRISC1, ns per 24-tile row)
center 948 | row_stats 2240 (c*c 1462, sfpu_reduce ROW 423 incl. FPU drain, rsqrt col0 225, pack/sync ~130) |
normalize 1465 | affine_scale 1446 | affine_shift 800. Per tile: HiFi4 ELWMUL ~60 ns (32 ELWMUL instrs:
4 fidelity phases x 2 per face x 4 faces), ELWADD/SUB 33-40 ns. Muls are math-bound: bf16 work pages
(half the unpack/pack bytes) move the chain only ~3%. The chain is 5 FPU ops per tile (sub, 3 x mul, add),
which is the minimum for two-pass at FPU level; SFPU replacements of any pass cost more per tile
(~100-200 cycles/tile vs 45-80), DEST-reuse fusions were measured slower in Perf 1 / R2.

## Variants (bench, 1 row x 24 tiles, 16 iterations, kernel ns)
| variant | kernel ns | chain/row | PCC / rel-RMS (fp32 DEST, randn) | mean 100 fp32 | bf16 DEST |
|---|---|---|---|---|---|
| 0 baseline | 116856 | 7.30 us | 0.999998 / 0.001921 | 0.008362 | 0.004746 |
| 1 cum_wait (c * c waits c per 4-tile group) | 113663 (-2.7%) | 7.10 | 0.999998 / 0.001921 | - | - |
| 2 gamma_first (cum_wait + out = ((c*g)*s)+b) | 113008 (-3.3%) | 7.06 | 0.999998 / 0.001908 | 0.008345 | 0.004676 |
| diag: work pages bf16 (precision trade) | 112996 (-3.3%) | - | 0.999997 / 0.002669 | 0.008478 | - |
rows=2 per block: 224818 -> 222279 (v1). Widths Wt 1/3/8/48 and rows 2: v2 within +-0.0001 rel-RMS of v0.
bf16 DEST at mean 100 fails the bench's 0.02 gate equally on v0 and v2 (0.0306): base two-pass behaviour.

## Whole op (device kernel us, one run each, base vs v2)
| row | base | v2 |
|---|---|---|
| focus1 | 35.35 / 35.55 / 35.59 | 35.56 / 35.23 / 35.36 |
| focus2 | 47.29 | 46.33 |
| focus3 | 33.31 | 33.19 |
| allL1_res | 24.93 | 24.19 |
| L1_nores | 20.57 | 20.89 |
| 8x384 | 36.51 | 36.17 |
| 32x32 | 18.70 | 18.71 |
| affineTILE_DRAM | 47.88 | 47.10 |
| focus1_bf16dest | 36.14 | 35.56 |
| 3x500 | 18.71 | 18.73 |
| 64x17 | 21.45 | 21.09 |
| 2x512 | 13.46 | 13.38 |
focus1 per core: heavy-core (2-block) chain 13.48 -> 13.07 us, TRISC1 end median 32.24 -> 31.88-32.09;
only the heavy cores' last row is on the critical path, so ~0.2 us/row reaches the op as ~0.2-0.3 us (noise band).
Precision (whole op, focus shape): identical to base to 4 digits (fp32 0.001905, mean100 0.01275, bf16 0.00477).
Correctness with v2 swapped in: test_layer_norm_tile + half_band + stream_knobs 156/156, row_stats + precision 24/24, all 12 perf rows.
