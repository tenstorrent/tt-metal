# khr_kernel_row_kblocks (Perf 20)

kh reuse (`KH_REUSE`) with one K-block per kernel row: with the weights resident, compute merges the three taps
(kh, 0..2) into one K-block of bk = 3 C_in / 32. The ring row t + kh W/32 is [kw][c], one contiguous in0 run. Writer
rounds 3kh .. 3kh+2 sit next to each other in cb_weights, so in1 is one contiguous [3 tw, bn] block. That gives 3
K-blocks per M-block instead of 9, and 3x fewer fp32 partials packs and per-K-block inits.
This is a compute-only change, `#ifdef KH_REUSE`. The host blocking, the writer's 9 rounds, and L1 are unchanged.
With streamed weights a K-block stays one tap, because a kernel row would need 3 rounds fronted and WEIGHT_DEPTH is 2.

Blackhole p150b, 110 cores, `l1_small_size=65536`, bf16 / HiFi4 / fp32 DEST. Numbers are device kernel us, the
median of 3 calls per run; "runs" is the number of sessions.

| case | base | krow (recommended) | ftr | ft | sb2 |
|---|---|---|---|---|---|
| P2 idx42 514x1024 64->256 | 1268.8 (3 runs) | 1245.3 (-1.9%, 3 runs) | 1249.1 | 1241.2 | - |
| P8 idx37 258x512 64->256 | 397.0 (5) | 384.6 (-3.1%, 7) | 391.9 | fails under profiler | 389.1 |
| P10 idx38 130x256 128->512 | 358.4 (4) | 360.6 (+0.6%, 6) | 356.1 (-0.6%) | fails under profiler | - |
| P11 idx39 514x1024 64->128 | 725.7 (3) | 727.8 (+0.3%, 3) | 719.7 | 714.2 | - |

The variants:
- **krow**: `op_copy/`, also `integration.patch`. Kernel-row K-blocks with compile-time bk / nkb.
- **ftr**: `op_ftr/` and `variants/ftr.patch`. The first M-block keeps one tap per K-block; later M-blocks use kernel
  rows. The walk takes a runtime `taps`, so it is instantiated once.
- **ft**: `variants/ft.patch`. Same schedule as ftr, but with two walk instantiations. Under `--profile` it overflows
  the kernel config buffer on P8 and P10 (TRISC text 6.5 KB -> 13.6 KB). Without the profiler it fits on P8.
- **rt3**: ftr's runtime walk at taps 3 everywhere. Control for the runtime cost: P8 +1.0 us, P10 +2.3 us vs krow.
- **ftc**: ftr with compile-time k loops. Same as ftr, so the k loop is not where the runtime cost comes from.
- **sb2**: krow with 2x2 subblocks where bm and W/32 are even, so a subblock never straddles the ring end. Measured on
  P8 only (P11's bm drops to 11 under it). Correct, but slower than 1x4. `_llk_unpack_AB_matmul_` unpacks
  rt x (1 + ct) tiles per k step, which is 6 for 2x2 vs 5 for 1x4.

Why P10 / P11 do not gain:
- **P10**: K-block 0 now waits for 3 weight rounds instead of 1. TRISC_0 `compute_weight_wait` is 5.0 us p50 vs
  0.27 us. ftr removes that wait, but costs P8 about 7 us, because the first M-block there is not weight-paced.
- **P11**: the max core is input-stream-bound. Its tilize waits on the reader, and max - p50 is 44 us. krow cuts the
  p50 walk by 10.5 us (693.0 -> 682.5) but leaves the max core where it was (730.5 -> 726.4).

Precision: not bit-identical, because three taps now accumulate in DEST before the packer's L1 add. PCC and mean
error vs fp64 torch are equal to 9 digits on every case (P2 0.9999985523, MAE 0.0011694 for both). Outputs that
differ: P2 7109 / 134.2M, P8 1790 / 33.5M, P10 1587 / 16.8M, P11 3474 / 67.1M. About 89% of them are 1 bf16 ulp. The
largest absolute difference is 1 ulp of the largest outputs (0.0156 at |y| ~ 4). The large ulp counts (max 128) only
occur at small magnitudes. LoFi is bit-identical.

Domain sweep (`runs/sweepA/B/C`, base vs krow, one run each):

| regime | base | krow | result |
|---|---|---|---|
| r8 258x256 64->1024 (N split, bn 16) | 701.1 | 693.2 | -1.1% |
| r9 1026x256 128->64 (bn 2, bm 19) | 385.4 | 373.1 | -3.2% |
| r10 1026x256 64->256 (bm 15) | 706.1 | 685.6 | -2.9% |
| r11 130x256 128->512, pad 1 | 358.6 | 358.3 | flat |
| 42 no bias | 1270.5 | 1230.8 | -3.1% |
| 37 + ReLU | 397.4 | 384.9 | -3.1% |
| 37 no bias + ReLU | 402.1 | 387.2 | -3.7% |
| 42 bf8_b weights | 1267.7 | 1240.6 | -2.1% |
| 39 bf8_b weights | 698.7 | 707.8 | +1.3%, inside the noise band; krow calls ranged 704-718 |
| 38 LoFi | 216.1 | 215.6 | flat, bit-identical |
| streamed weights (r7, 38 no-bias + ReLU, 38 bf8_b) | | | taps 1: bit-identical, flat |
| idx 31, 34 (no kh reuse) | | | bit-identical, flat |

The no-bias rows have ~5-10% of outputs differing. That is because the existing no-bias walk reloads the fp32 partials
through SrcA, which truncates them to TF32. krow only changes how many taps the reloaded sum holds; MAE changes by
+0.05%. `fp32_dest_acc_en=False` is rejected by the op.

Reach: only kh-reuse programs, which need DRAM-interleaved ROW_MAJOR bf16 input (`bank_run_input`). It never reaches
TILE or sharded input (`kh_reuse_capable`: `rm_ring_px > 0`, `not tile_input`, `not sharded_output`).

Interaction with the ragged-tail fix: the two changes are independent. That fix limits the rows of the last M-block;
this one changes the K-blocks. On the walk call site, krow changes only template arguments, so the merge is simple.
ftr's runtime `taps` argument would sit next to ragged_tail's runtime `row_groups` argument.

Files:
- `op_base/`: the working tree at the round start. One reader comment is re-wrapped to avoid a profiler zone-hash
  collision.
- `op_copy/`: krow.
- `op_ftr/`: ftr.
- `variants/*.patch`: the other variants, as patches against op_base.
- `test_khrk.py` and `run.sh`: the harness (`run.sh <label> <cases> <variants>`). Case modifiers: `:nb :relu :w8
  :lofi :nofp32`.
- `summ.py`: prints a summary of a run's jsonl.
- `khrk_plan.py`: host-only blocking and a small kh-reuse geometry sweep.
