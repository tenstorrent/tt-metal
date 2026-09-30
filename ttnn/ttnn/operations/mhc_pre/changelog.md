# Changelog: mhc_pre

## Phase 0 — Core Implementation
- **Date**: 2026-09-30
- **What was done**: Initial implementation via the incremental pipeline (planner → implementer → verifier).
  - Regime R1 `group_ksplit_resident`: tokens go to groups and nC is split over group ranks; the X block
    stays resident from the projection to the y-mix (X read once).
  - Group combine: push gather to the root, an fp32 SFPU rank-ordered fold, and an `mcast_pipe` broadcast.
  - Custom fp32 SFPU coefficient and Sinkhorn block ops; FPU y-mix; the Sinkhorn is owned round-robin.
  - Implementer perf pass (BH p150, fp32): 640×7168 went from 415 to 382 µs; the composite baseline is 2396 µs.
- **SUPPORTED at Phase 0**: dtype=[float32], layout=[TILE], weight_dtype=[float32, bfloat16],
  fp32_dest_acc_en=[True], alignment=[tile_aligned, h_non_aligned]
- **Accuracy achieved** (fp32 X, 4 shapes, `test_mhc_pre_precision_baseline.py`):
  - y: PCC ≥ 0.9999998, max_abs ≤ 9.5e-3, rel_rms ≈ 8.6e-4 (ratio median 0.99932: FPU truncation of fp32 X)
  - post: PCC ≥ 0.9999998, max_abs ≤ 9.4e-4, rel_rms ≤ 5.3e-4
  - comb: PCC ≥ 0.9999999, max_abs ≤ 4.0e-4, rel_rms ≤ 4.5e-4
  - The Sinkhorn in isolation is fp32-exact (1.9e-7 vs fp64). Repeat calls are bitwise-identical.
- **Golden suite at Phase 0**: 106 supported_pass, 98 xfail_expected (all `dtype = bfloat16`), 1
  supported_fail. The failure is `test_large_sinkhorn_logits[T64_nC4096]`, a precision failure tracked in
  Refinement 2. Drift is 0 (`verifier_report.json`).
- **Issues encountered / verifier fixes**:
  - DRY: CB page counts were restated in `_l1_bytes`. Both the L1 fit and the ProgramDescriptor now read
    `_cb_table` (behaviour unchanged).
  - `weight_dtype += bfloat16` on golden evidence (no kernel change).
  - `dtype = bfloat16` runs, but 11 bf16-X × fp32-W cells and the bf16 depth chain miss the 5e-4 coefficient
    gate. Root cause, measured: the FPU keeps ~9 bits of a tf32 W, and its in-tile matmul accumulation is not
    fp32-exact. It is not claimed; it moves to Refinement 1 (W hi/lo split).
- **Tests added**: `test_mhc_pre_precision_baseline.py` (verifier), plus probes 002–007. The acceptance
  suite (`test_mhc_pre.py`), `test_mhc_pre_blocking.py` and `test_mhc_pre_perf.py` were already in place.

## Refinement 1 — bf16 residual streams (lands the perf-focus contract)
- **Date**: 2026-09-30
- **What was done**:
  - `SUPPORTED["dtype"]` gains `bfloat16`.
  - For an fp32 W, the compute kernel runs `w_split_block` once, before block 0:
    - Each resident fp32 W tile is loaded via `copy_tile`, with `cb_weight` tagged UnpackToDestFp32.
    - The SFPU computes `W_hi = W & 0xFFFF0000` and `W_lo = W − W_hi`.
    - Both are packed as bf16, in place, into `cb_weight_split`, a second buffer index aliasing
      `cb_weight`'s allocation. There is 0 extra L1, and both are declared in `_cb_table`.
  - `project_block_pieces<2>` (a thin block op over `matmul_tiles`; `matmul_block` cannot take two in1
    operands into one DEST window) accumulates X@W_hi (HiFi4) and X@W_lo (`W_LO_FIDELITY` = LoFi) in the
    same DEST window. The X block is still retained for Σx² and the y-mix.
  - A bf16 W keeps the unchanged `matmul_block` helper path (`w_pieces == 1`).
  - Perf levers: the reader pushes W in `W_CHUNK_TILES = 8` chunks, so the split overlaps the W DRAM read.
    The LoFi lo pass saves 3 of the 4 fidelity phases on half of the doubled matmul.
  - Knobs `W_CHUNK_TILES`, `W_LO_FIDELITY` and `w_pieces()` are single-source in the descriptor.
- **Reused**: the op file gate, `_cb_table`, the reader W load, the `matmul_block` path (bf16 W), and every
  other phase.
- **Added**: the `cb_weight_split` alias, `w_split_block`, and `project_block_pieces`.
- **Accuracy achieved** (bf16 X × fp32 W, `test_mhc_pre_bf16_stream_fp32_weight_precision`):
  - post rel-RMS 2.4–3.1e-4 and comb 2.5–2.6e-4 on 17×512, 1000×7168 and 256×6144, against the 5e-4 gate.
    Before this refinement they were 5.0–5.5e-4.
  - The fp32-X baseline shapes are unchanged or better (post/comb ≤ 4.0e-4 rel-RMS, PCC ≥ 0.99999993).
- **Golden test progress**: 205/206 (was 106 pass + 98 xfail + 1 fail).
  - All 98 bf16 cells pass, plus `test_comb_depth_chain[bf16]`.
  - The only failure is `test_large_sinkhorn_logits[T64_nC4096]` (Refinement 2, unchanged at worst row
    0.0674).
- **Perf** (BH, device kernel ns, bf16 X / fp32 W, before → after):

  | Shape (T×C) | Before | After |
  |---|---|---|
  | 640×7168 | 267.3 µs | 269.6 µs |
  | 640×1792 | 103.6 µs | 105.1 µs |
  | 1280×4096 | 261.9 µs | 264.6 µs |
  | 4096×1792 | 371.6 µs | 373.4 µs |

  fp32 X, after: 640×7168 378 µs (was 382–383), 4096×1792 523 µs (was 521).

  The first split measured 295 µs. Ablation showed ~20 µs from the SFPU split sitting serially after the W
  read, and ~6 µs from the doubled HiFi4 matmul. The chunked W push and the LoFi lo pass removed both.
- **Issues encountered**: `MATH_FIDELITY` exists only on TRISC_MATH. The lo-pass layout decision must match
  on every TRISC, so the main fidelity is passed as a CT arg.
- **Tests added**:
  - `test_mhc_pre_precision_baseline.py::test_mhc_pre_bf16_stream_fp32_weight_precision` (3 shapes).
  - `test_mhc_pre_perf.py` is parametrized over X dtype.
