# ragged_tail (Perf 20)

A core's last M-block runs only its valid tile-rows. The walk used to run the compile-time `bm / sb_h` row groups on
every M-block, so when a core's tile-row count is not a multiple of bm, its last M-block tilized, multiplied, bias-added
and packed padding rows that the writer then dropped.

## What changed (op_copy vs op_base, `integration.patch`; apply from the op dir with `patch -p1`)
- `conv2d_3x3_kblock_walk.hpp`: `walk(..., row_groups)` takes the runtime number of row groups (default: all of
  them). The matmul, output packs and bias pass use that count. in0 and partials CB counts stay nominal: with a bias,
  the region's unpushed tail is pushed and popped after the bias pass.
- `conv2d_3x3_compute.cpp`: under `RAGGED_ARG`, compute reads the core's valid tile-rows. The walk then runs
  `ceil(valid / sb_h)` row groups on the last block. The tap-direct FIFO tilize handles only those units, pops the
  padding units' table entries (zero units, no gathered pages) and pushes their in0 pages unwritten. kh reuse: lane 2
  ends at `2 wt + actual` instead of `2 wt + nmb * bm`.
- `conv2d_3x3_khr_band_order.hpp`, `conv2d_3x3_rm_ring.hpp`: the reader's band order and its fill runs end lane 2 at
  the same row. The pixel stream does not change, since those rows had empty pixel ranges.
- `conv2d_3x3_writer.cpp`: under `RAGGED_ARG`, the interleaved store drains `ceil(valid / sb_h)` row groups per block.
- Descriptor: `ragged` = the K-blocked walk with the writer's interleaved store: not kw-fold, not K split, not N-slice,
  not slab-direct, not C_out fold, not sharded output, one in0 group. The core's valid tile-rows go in as the last
  compute arg. `RAGGED_ARG` = its index without the TILE ring's per-M-block args, and the kernel adds `2 * nmb` itself.
  kh reuse asserts ragged. The blocking and L1 are unchanged.

## Results (Blackhole p150b, device kernel us, median of 3 calls, alternating order; all outputs bit-identical, sha1)
| case | base | ragged | delta |
|---|---|---|---|
| P10 idx38 (x17) | 358.3 | 329.7 | -8.0% |
| P14 idx40 | 731.9 | 672.4 | -8.1% |
| P8 idx37 (x5) | 397.2 | 382.6 | -3.7% |
| P11 idx39 | 721.7 | 703.3 | -2.5% |
| P7 idx43 (x8) | 1297.1 | 1268.6 | -2.2% |
| P13 idx47 (full blocks) | 5339.0 | 5339.3 | flat |
| other 11 focus cases | | | within +-0.5% (P9 -1.5%, noise) |
| 17 focus, call-weighted | 58.899 ms | 57.972 ms | -1.57% |
| r2 130x512 128->512 | 736.2 | 674.4 | -8.4% |
| r8 130x1024 64->256 (kh reuse, cores < 1 image row) | 370.7 | 363.9 | -1.8% |
| r6 66x2048 32->512 / r9 66x512 64->64 (sb 2x2) | 467.7 / 73.8 | 470.8 / 73.5 | flat |
| TILE t38 / t43 / x10 | 414.0 / 1562.0 / 1675.5 | 399.8 / 1470.1 / 1622.6 | -3.4 / -5.9 / -3.2% |
| TILE x15 (s2, bm 1) | 1430.3 | 1445.9 | flat (bm 1 has no padding rows) |

No bias + ReLU (`v1`) and bias + ReLU (`v2`) on 38, 40, t38: bit-identical. P14 no-bias -4.3%, t38 no-bias -3.9%.

## Harness
`run.sh <label> <cases> base,rt [GR_ITERS=3 GR_ORDER=alt GR_BIAS=0 GR_RELU=1]` (from `graduation_r19/test_final.py`).
Results go to `runs/<label>.{txt,jsonl}`. `domain_plan.py` (host-only) prints each regime's blocking and padding rows.
