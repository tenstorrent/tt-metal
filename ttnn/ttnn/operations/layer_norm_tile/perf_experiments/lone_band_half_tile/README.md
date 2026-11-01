# lone_band_half_tile

Idea: an R3 core's lone band fills only faces 0-1 of its compute tile-row. Run every pass of that row on a
16x32 TensorShape (num_faces = 2) in unpack, math and pack, so faces 2-3 are never computed.

Base commit 9daa4c30ecf, BH p100a (110 cores). fp32_dest_acc_en / math_fidelity / approx / dtypes unchanged.

## Files
- `kernels/layer_norm_tile_compute.cpp`: copy of the op's compute kernel plus namespace `lone` (raw-LLK
  half-tile binary / DEST-reuse / pack primitives, `pass`, `shifted_add`, `stats_window`, `row_stats`) and the
  per-phase split in `kernel_main` (`by_rows`: full rows before the lone row, the lone row, full rows after,
  so every phase still moves the whole block through each CB). CT arg 23 `half_lone`, RT arg 2 `lone_row`.
  `half_lone = 0` compiles to the op's path.
- `kernels/layer_norm_tile_{reader,writer}.cpp`, `layer_norm_tile_stream.hpp`: unchanged copies.
- `lbh_descriptor.py`: copy of the descriptor; `HALF_LONE` (env `LBH_VARIANT=base|half`), `_lone_row`
  (mirrors `RowSources`: first row when an odd run starts on a bottom band, else last), CT/RT args.
- `lbh_swap.py`: pytest plugin, routes the real entry and the op's descriptor module to the copy.
- `test_lbh.py`: perf + guard rows with PCC / rel-RMS; `test_large_mean_precision` (|mean| = 100).
- `bench.sh <base|half> <label> [rows]`: profiled run, appends to `logs/bench.txt`.
- `core_timeline.py`: per-core end times and zone timeline of the slowest cores.

## Why raw LLK
The CB-id compute API and the eltwise_chain / convenience helpers read the tile shape from the CB's
`unpack_tile_num_faces` / `pack_tile_num_faces` tables, which are `constexpr` per CB id. The lone row uses the
same CBs as the full rows, so a per-row face count is inexpressible through them. The kernel calls the
`*_impl` cores the CB-id API wraps (`llk_unpack_AB_init_impl` / `llk_unpack_AB_impl`,
`llk_unpack_A_init_impl` / `llk_unpack_A_impl`, `llk_math_eltwise_binary_init_impl` /
`llk_math_eltwise_binary_impl`, `llk_pack_init_impl`) with `TensorShape{16, 16, 1, 2}`, and restores the
packer with `llk_pack_init(cb)`. Alias CB ids with a 16x32 tile would need lockstep pointer moves on every
producer and consumer.

## Results (device kernel us, final kernel = v3)
| row | base | half | delta |
|---|---|---|---|
| focus1 inL1/resDRAM/outDRAM | 35.51 (35.02-35.82, n=5) | 35.40 (35.00-36.53, n=5) | flat |
| focus2 all DRAM + res | 47.45 (n=5) | 47.54 (n=5) | flat |
| focus3 all DRAM no res | 33.56 (n=5) | 33.42 (n=5) | flat |
| allL1_res | 24.41 (24.32-26.19, n=3) | 23.84 (23.50-24.01, n=3) | -0.6 (-2.3%); v1 run: 26.47 -> 23.03 |
| L1_nores | 20.70 | 20.80 | flat |
| 32x32 all DRAM res | 18.93 | 19.02 | flat |
| affineTILE_DRAM | 47.51 (n=5) | 48.13 (n=5) | flat (+1.3%) |
| focus1 bf16 DEST | 35.22 (n=5) | 35.66 (n=5) | flat (+1.2%) |
| 3x500 inL1/resDRAM | 18.86 | 19.23 | +2.0%, in band (v1: 18.78 vs 18.78) |
| 2x512 all L1 res | 13.54 | 11.84 | -1.7 (-12.6%) |
| 8x384, 64x17 | R1, no lone band | same code path | flat |

n=5 rows pool the v1 sweeps (same lone-row math for R = 1 blocks) with the two v3 sweeps.

Precision: identical to base on every row (PCC / rel-RMS equal to 5-6 digits, randn and |mean| = 100, both
DEST modes). Unit dir 237/237 + bench 12/12, golden 545/545 through the copy.

Per-pass math-thread time of the lone row (focus1 critical core): center 0.75 -> 0.77, row_stats 2.25 -> 1.50,
normalize 1.48 -> 0.88, affine_scale 1.42 -> 0.93, affine_shift 0.84 -> 0.84 (us). HiFi4 multiplies scale
with faces; add / sub passes are bound by per-tile overhead (~43 cycles/tile), not faces.

Why focus1 does not move: the critical core (13, 4) is lone-first. Its compute end moves 33.6 -> 32.1 us,
but the end is the writer RISC: it reads the next row's residual until ~23 us, then the lone row's DRAM
output stores take ~10 us under all-core write contention. Light cores also end at ~35.5 us on the same
write tail.

## Variants tried
- v1: lone row as its own compute block (sub-block split). Broke multi-row blocks (7x3200, PCC 0.10): the
  resident-indexed CBs assume each block starts at the CB base. Replaced by the per-phase split.
- v2: per-pass half-pack init / restore. 32x32 18.96 -> 20.25, 3x500 18.91 -> 20.56 (per-window rows run 12
  passes per row, latency-bound).
- v3 (final): the half packer MOP is held across the lone row's add_and_stats loop and its per-window run.
- v4: also held across lone-only whole-row blocks; no measurable change, reverted.
