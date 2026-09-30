# mhc_pre — cross-block pipeline (Perf 1, round 1)

Box: BH p150, card 2. Metric: in-process `DEVICE KERNEL DURATION [ns]` (ReadDeviceProfiler). Variants are
interleaved in the same process; medians of N calls, samples in us. Precision contract fixed: fp32_dest_acc_en=True,
HiFi4, approx off, dtypes as given; every variant runs under the identical config.

## Layout

- `mhc_pre.py`, `mhc_pre_program_descriptor.py`, `kernels/`: a copy of the op with every schedule variant behind
  descriptor knobs (`PIPELINE`, `PIPE_S_AHEAD`, `PIPE_S_NOHS`, `PIPE_ROOT_COEF_FIRST`, `PIPE_TAIL_ONLY`, `PIPE_X_DEPTH`,
  ...). The knobs are bench-only.
- `grad/`: the graduated op, with no knobs. `graduation.patch` is `grad/` minus the real op
  (descriptor, compute, writer; reader unchanged), and `git apply --check` passes on the current tree.
- `bench/test_cbp.py` + `bench/run.sh`: correctness (`run.sh correct`) and perf (`run.sh perf`). Selected by the env
  vars `CBP_SHAPES`, `CBP_DTYPES` (bx_fw, fx_fw, fx_bw, bx_bw), `CBP_VARIANTS`, `CBP_REPEAT` and `CBP_SEEDS`.
  `bench/cbp_grad_plugin.py` runs the unmodified golden or unit suites against `grad/`:
  `PYTHONPATH=<bench> scripts/run_safe_pytest.sh --device 2 eval/golden_tests/mhc_pre/ -p cbp_grad_plugin`.

## Graduated schedule (grad/)

Step b (proj(b) done on entry). `pipe_at(b) = b+1 < nb && (depth >= 3 || b + depth >= nb)`:
- pipelined step: compute runs proj(b+1) first. X(b+1) sits behind X(b), and an `XView` adds a wrap-aware tile
  offset. The root folds b (if needed), then computes coef(b), then folds b+1, then runs ymix(b). The other ranks
  run tail(b). The writer sends P(b+1), then the root does gather+fold+mcast of S(b+1), then y(b)/pc(b) are stored,
  then the non-root ranks receive S(b+1).
- serial step: exactly the old order.
- With depth 2 (the default) only the LAST step pipelines. Pipelining earlier steps holds X(b) past proj(b+1), which
  delays the reader's X(b+2) prefetch. That was measured as a regression (see menu option B).
- Depth-1 plans are always serial.
- cb_coef_in holds 2 blocks (+8 KB at bt=1). The group mcast has NO consumer-ready handshake (Flag):
  - the landing is write-once, because the root mcasts S(b+1) only after all P(b+1) arrived;
  - each rank sends P(b+1) after freeing S(b-1)/S(b) and after resetting the round-b flag.
  - A Counter data-ready signal HANGS here. Its multicast atomic waits for an ack from the looped-back root
    (triage: root BRISC in the send's `noc_async_atomic_barrier`).
- fp32 X: cb_grid is popped right after the projection instead of after the y-mix.

## Correctness (only pass/fail)

All variants were checked against the `helpers.pytorch_mhc_pre` reference and TOLERANCES, plus the doubly-stochastic
gate. Coverage: seeds 0,1 (plus 2 for `grad` on the small shapes), shapes 1280x4096, 4096x1792, 2048x5120, 640x7168,
640x1792, 1000x1792 (ragged), 100x1792, 1x7168, 64x4096, 32x128 and 1x2x256x8192, each with all 4 dtype pairs.

- **Every candidate output is BITWISE identical to base**, in every cell and on every seed (`bitwise=base`).
- The one golden miss is base-level: 1x7168 bx_bw on seed 1, which is not a golden seed. The post rms is
  5.08e-4 against a limit of 5e-4, and it is identical in all variants.
- The golden suite on `grad/` passes 206/206. Unit tests `test_mhc_pre.py` 14/14 and `test_mhc_pre_blocking.py` 10/10
  pass. The blocking test's pd monkeypatches hit the real pd, so its knob coverage is partial.

Worst rel-RMS (y/post/comb), the same for base and every variant:
- bf16 X: 1.68e-3 / 3.0e-4 / 2.8e-4
- fp32 X: 8.5e-4 / 5.9e-5 / 5.6e-5

## Menu at the focus shape 1280x4096, bf16 X / fp32 W (G=5, 2 blocks per core)

| option | us (median) | samples |
|---|---|---|
| base (op as committed) | 147.7 | 146.9 151.0 148.1 147.7 147.6 |
| pipe (proj(b+1) before tail(b), cb_coef_in x1, handshake) | 146.0 | 144.6 146.0 147.0 146.8 145.5 |
| pipe_rcf (root folds b before proj(b+1)) | 146.1 | 147.0 145.4 146.1 145.2 147.0 |
| pipe_s1 / s2 / s3 (split proj(b+1) around tail(b)) | 150.0 / 149.7 / 147.8 | (regressions / flat) |
| sahead (S(b+1) mcast before tail(b), 2-block cb_coef_in, handshake) | 142.8 | 143.3 143.4 142.8 141.0 142.2 |
| nohs (sahead, no handshake) | 143.5 | 143.9 146.0 141.8 143.5 142.9 |
| sahead_rcoef (root: coef(b) before folding b+1) | 142.1 | 141.3 140.8 143.5 142.9 142.1 |
| **nohs_rcoef = grad (last step)** | **136.8** | 139.0 136.8 136.8 136.6 137.7 |
| nohs_rcf_rcoef | 139.9 | 139.9 139.1 139.1 142.4 140.9 |
| +depth 3 (x_block_depth 3) | 136.5-141.0 | noisy across runs; no gain at 2 blocks |

The focus shape has 2 blocks, so the graduated rule equals the full pipeline there. Precision is bitwise equal
to base for every option.

## Domain sweep: base -> grad (us, medians; 5 calls where listed, else 3)

| shape | bx_fw | fx_fw | fx_bw | bx_bw |
|---|---|---|---|---|
| 1280x4096 | 147.7 -> **136.8** | 379.3 -> **371.1** | 362.4 -> **348.2** (noisy) | 182.8 -> 182.3 (1 blk, d1) |
| 4096x1792 | 241.1 -> **232.0** | 541.7 -> 553.8 (**+2.2%, 10 calls**) | 537.0 -> **512.0** | 260.8 -> 260.7 (d1) |
| 2048x5120 | 313.1 -> **304.6** | 710.3 -> **702.3** | 652.3 -> **642.1** | 354.0 -> 353.6 (d1) |
| 640x7168 | 144.9 -> 146.5 (1 blk, noise) | 411.7 -> 408.1 | 355.1 -> **345.3** | 147.4 -> 147.6 |
| 640x1792 | 43.7 -> 43.5 | 121.7 -> **117.9** | 115.0 -> **112.3** | 43.7 -> 44.0 |
| 1000x1792 | 152.7 -> 153.6 (1 blk) | 190.4 -> **183.1** | 183.5 -> **176.9** | 104.2 -> 104.5 |
| 1x7168 | 45.9 -> 45.8 | 119.4 -> **115.0** | 80.2 -> 80.0 | 40.2 -> 40.4 |
| 64x4096 | 42.8 -> 42.8 | 92.2 -> **89.3** | 64.6 -> 64.4 | 38.2 -> 38.2 |
| 32x128 | 18.7 -> 18.8 | 23.8 -> 23.3 | 19.9 -> 20.0 | 18.8 -> 18.9 |

The 3-5 us gains on 1-block fp32-X cells have an identical schedule, so they come from L1 placement: cb_coef_in
doubles and the CBs behind it move.

## Other options measured (not graduated)

A. **Full pipeline at depth 2** (`nohs_rcoef`). It wins more on the S-wait-bound shape: 4096x1792 bx_fw
   220 (vs grad 228-232). But it regresses the DRAM-bound multi-block cells because it loses the X(b+2) prefetch:
   - 1280x4096 fx_fw: 378 -> 400
   - 2048x5120 fx_fw: 711 -> 756
   - 2048x5120 bx_fw: 315 -> 317-322
   - 4096x1792 fx_fw: 542 -> 560

B. **Depth 3 + full pipeline where depth 3 fits** (`nohs_rcoef_tail_d3` / `sahead_rcoef_tail_d3`). Big wins on
   small-slice fp32-X cells:
   - 4096x1792 fx_fw 541.8 -> 523.8, fx_bw 536.7 -> 491.3
   - 1000x1792 fx_fw 189 -> 176, fx_bw 184 -> 165
   Regressions on large-slice fp32-X cells:
   - 1280x4096 fx_fw 377 -> 394
   - 2048x5120 fx_fw 711 -> 779 (nohs) / 721 (handshake)
   It also costs one more X block of L1, so it is not a single-path candidate.

C. Split proj(b+1) around tail(b) (`pipe_s*`): regression or flat. Root fold order (`rcf`, `rtail`): no better than
   coef-first.

## Exceptions (grad)

- **measured-regression**: 4096x1792 fp32 X / fp32 W (13 blocks, G=11): 541.7 -> 553.8 us (+2.2%). It is a
  bimodal mode shift: base spends 8 of 10 calls at ~541, grad spends 8 of 10 at ~554. The same shape with a bf16 W
  wins (537 -> 512).
- **inexpressible**: depth-1 plans (1280x4096 / 4096x1792 / 2048x5120 bx_bw) have no room for X(b+1), so they stay
  serial automatically (`pipe_at` is false).
- Flat: every 1-block cell (the schedule is unchanged).
- Untested: RM layout (not in SUPPORTED); `bt > 1` plans (BLOCK_TOKEN_TILES_CAP = 1 today; the code is bt-generic).

## Raw LLK

`project_block_pieces<1>` (the existing raw matmul_tiles block op) is used only for the pipelined proj(b+1) on the
bf16 X / bf16 W path. `matmul_block` reads in0 from the CB front and has no tile-index base, so it cannot project a
block sitting behind the resident X(b). The output is bitwise identical to the helper. The front-block projection
keeps the helper. The indexed reads of X(b+1) at a modular (wrapped) page offset work because the unpacker's address
math is uint32 (`cb_access_within_bounds` also holds).
