# Changelog: groupnorm_sc_N_1_HW_C

## Phase 0 — Core Implementation
- **Date**: 2026-06-10
- **What was done**: Initial implementation via incremental pipeline (planner → implementer →
  verifier). Single-core GroupNorm over (N, 1, HW, C): per-(n,g) three streaming passes
  (mean → centered variance → normalize + optional affine) using kernel-lib helpers throughout.
- **SUPPORTED at Phase 0**: dtype=[bfloat16], layout=[TILE, ROW_MAJOR],
  alignment=[tile_aligned], groups_alignment=[aligned],
  affine=[gamma_beta, gamma_only, no_affine], affine_dtype=[bfloat16],
  affine_layout=[ROW_MAJOR, TILE]
- **Accuracy achieved**: PCC ≥ 0.999992, max_abs_err ≤ 0.080, rel_rms_err ≤ 0.0036
  (measured on 4 shapes via test_groupnorm_sc_N_1_HW_C_precision_baseline.py, bf16 gamma_beta)
- **Golden suite at Phase 0**: 300 / 7236 cells passing (3385 xfail_expected, 3551
  invalid_skipped; supported_fail = xpass_drift = xfail_wrong_mode = 0, per `verifier_report.json`)
- **Issues encountered**: verifier fixes — (1) reader/writer per-tile NoC barriers batched per
  Wg row chunk; (2) affine_layout=TILE under-claim promoted into SUPPORTED on probe evidence
  (PCC ≥ 0.99999, +120 golden cells). Known boundary: G=1 + C ≥ 2048 with gamma_beta exceeds
  L1 (no golden cell affected; see verification_report.md).
- **Tests added**: test_groupnorm_sc_N_1_HW_C.py (planner),
  test_groupnorm_sc_N_1_HW_C_precision_baseline.py,
  test_groupnorm_sc_N_1_HW_C_extended.py (verifier)

## Refinement 1 — Numerical configurability + multi-core distribution
- **Date**: 2026-06-10
- **What was done**:
  - `SUPPORTED["dtype"]` and `SUPPORTED["affine_dtype"]` extended with `float32`, `bfloat8_b`
    (incl. all mixed-precision combinations). Zero compute-kernel changes for dtype work —
    input/output/affine CB formats were already dtype-derived; helpers carry data-format reconfig.
  - Intermediate stat CBs (mean/var/centered/xhat/scaled): `Float32` when `fp32_dest_acc_en`
    or fp32 input, else `bfloat16` (incl. bf8b input — block-float intermediates lose precision
    for no L1 win). No `UnpackToDestFp32` tags: every intermediate feeds FPU helpers
    (sub/mul/square/reduce), which the tag forbids.
  - `compute_kernel_config` exposed on the entry point (WormholeComputeKernelConfig-style:
    math_fidelity, fp32_dest_acc_en, math_approx_mode, dst_full_sync_en). Defaults match the
    Phase-0 hard-coded behavior for bf16/bf8b; fp32 input defaults `fp32_dest_acc_en=True`
    (dtype-driven; fp32 had no prior behavior — measured rel_rms 0.0112→0.0075 on
    (1,1,1024,256) G=8 + bf8b gamma; flipped the only 4 failing golden cells).
  - Multi-core: N·G (n,g) groups split over the full grid via `split_work_to_cores`
    (one (n,g) group per work unit); per-core `[start_group, num_groups_here]` runtime args
    in all three kernels; gamma/beta read once per core; per-core CB footprint unchanged.
- **Accuracy achieved**: precision matrix 384/384 (8 shapes × 3 dtypes × 4 fidelities ×
  2 acc × 2 distributions): HiFi4 PCC ≥ 0.99986 all dtypes; worst overall PCC 0.9977 (LoFi).
  See tests/.../precision_matrix_results.md.
- **Golden test progress**: 1650 / 7236 supported_pass (Phase 0: 300), 2035 xfail_expected,
  3551 invalid_skipped, 0 supported_fail / 0 xpass_drift / 0 xfail_wrong_mode. EXCLUSIONS
  remains empty.
- **Issues encountered**: FLOAT32 + gamma_only + bf8b gamma initially failed rms by ~5%
  (0.0104–0.0106 vs 0.01) on default config. Probes showed bf8b-gamma quantization alone is
  only 0.0055–0.0066; the gap was TF32 dest rounding of the fp32 stats — fixed by the
  dtype-driven fp32_dest_acc_en default (no EXCLUSIONS needed). LoFi rel-RMS ≈ 0.021 vs
  0.02 HiFi band — expected fidelity tradeoff, asserted via PCC only.
- **Tests added**: test_groupnorm_sc_N_1_HW_C_refinement1.py (26 cases: new dtypes, mixed
  affine, compute config sweep, multicore split regimes incl. uneven 100-group + distinct-group
  routing), test_groupnorm_sc_N_1_HW_C_precision_matrix.py (384 cases) +
  precision_matrix_results.md, probes/probe_004.py.

## Refinement 2 — Non-tile-aligned shapes (HW and C tails)
- **Date**: 2026-06-10
- **What was done**:
  - `SUPPORTED["alignment"]` extended with `hw_non_aligned`, `c_non_aligned`.
  - `tag_groups_alignment` now reports `aligned` for `num_groups == 1` (a single group cannot
    straddle a tile boundary — only the C-tail tile needs masking, which is the alignment
    axis's job). All `c_non_aligned` golden cells have G == 1; G > 1 C tails stay gated on
    `groups_alignment=non_aligned` until Refinement 3.
  - Host: padded tile counts via ceil (Ht/Wt/Wg); CT args `hw_tail`, `c_tail` (+ `mask_output`)
    to reader/compute; scaler stays `1/sqrt(HW·Cg)` on logical sizes.
  - Masking (REDUCE_SCALAR has no partial-scaler support — `prepare_partial_reduce_scalers`
    static-rejects it): reader fills persistent bf16 0/1 mask rows (`cb_mask_interior` Wg tiles
    for C tail; `cb_mask_tail` Wg tiles for HW tail, corner combined); compute multiplies
    `cb_centered` by the held mask after `x − mean` in pass 2 (variance correctness) and —
    only when output dtype is bf8b — in pass 3 (block-exponent hygiene; for fp32/bf16 the
    padding is unread garbage and the extra mul cost rms). Pass 1 needs no masking — host
    tilize zero-pads, so the SUM over the padded slab is already logical.
  - Scaler CB format now follows the stat format (Float32 when fp32 stats): tail element
    counts are non-power-of-two, so a bf16 1/sqrt(N) quantizes at 2^-9 relative and shifts
    the mean by ~0.4% (DPRINT-confirmed: mean 0.996 on all-ones); fp32 scaler flipped 4 of
    6 failing fp32 + bf8b-gamma golden cells.
- **Accuracy achieved**: rel_rms 0.002–0.006 bf16, ≤ 0.005 fp32, ≤ 0.015 bf8b; PCC ≥ 0.995
  on all 28 refinement-2 unit cases (shapes incl. 17x64, 50x128, 47x256, 2x1x100x128, 64x17,
  64x50, 128x100, 2x1x64x47, 17x17, 50x100, 2x1x47x50, hw tails with G ∈ {4,8}).
- **Golden test progress**: shape-sharded full sweep: 2,361 supported_pass across shards
  (1,923 in disjoint shape shards + 438 in the tail-shape shard), 0 xpass_drift,
  0 xfail_wrong_mode, 2 supported_fail (see Issues). Regression suite 35/35. Up from 1,650
  supported_pass at Refinement 1.
- **Issues encountered**:
  - All-ones probing initially looked like a masking bug (constant 0.7773 output). DPRINT
    showed correct masking; root cause was bf16 scaler quantization amplified by rstd at
    var = 0 — degenerate input, documented in the debug test.
  - 2 irreducible golden fails: `1x1x64x17 gamma_only bf8b TILE/RM` rms 0.01056 vs target
    0.01 — the bf8b quantization floor of gamma alone (perfect torch math) is 0.01064
    (probe_006), i.e. the op is BELOW the floor (op_vs_floor rms 0.0009). NOT in EXCLUSIONS
    (precision near-miss). Follow-up belongs test-side: golden tolerance is keyed on input
    dtype only and doesn't budget bf8b affine quantization on tiny C.
- **Tests added**: test_groupnorm_sc_N_1_HW_C_refinement2.py (28 cases),
  test_groupnorm_sc_N_1_HW_C_refinement2_debug.py (6 deterministic cases), probes/probe_005.py
  (DPRINT mean/rstd), probes/probe_006.py (bf8b gamma quantization floor).

## Refinement 3 — Non-tile-aligned group widths (SD / SDXL regime)
- **Date**: 2026-06-10
- **What was done**:
  - `SUPPORTED["groups_alignment"]` extended with `non_aligned` (Cg % 32 != 0, G > 1 — the
    SD/SDXL regime, Cg ∈ {10..80}; also unblocks the 8 coupled C-tail + straddle cells).
  - New cluster kernel path (all three kernels, CT-gated `GROUPS_NA`; aligned path untouched):
    work unit = (n, cluster), cluster = lcm(Cg, 32) channels capped at C — group/tile
    boundaries re-align at cluster edges, so output tiles stay disjoint across cores.
    Passes 1/2 run per group over its tile span (≤ ceil(Cg/32)+1 tiles) with reader-generated
    0/1 column masks; pass 1 masks BEFORE the reduce (neighbor groups share tiles). Compute
    exports per-group mean/rstd scalars; the reader scatters them into per-column row vectors
    (zeros in padding cols). Pass 3 is one Row-broadcast sweep over the cluster:
    (x − mean_row)·rstd_row·γ+β — no partial-tile writes; padding cols zero via rstd_row=0;
    bf8b + HW-tail rows masked with two scalar tiles (row masks are column-independent).
  - CB-wrap rule learned the hard way (deadlock at first Wsg=2 group): multi-tile frames must
    have uniform width per CB. Input/output stream per tile; mask frames pad to Ws_max;
    row vectors push full Wc_full frames.
  - Precision at Ht=512 (SDXL 16384x320): per-row reload-accumulate truncates through TF32
    srcA toward zero — variance deficit 8.7% (slope 1.046 on all groups). Fixed by 16-row
    chunked reduce blocks (512→32 reloads) + `fp32_dest_acc_en` default ON for the cluster
    path (Float32 stat CBs). L1 OOM from chunking fixed by sizing only cb_centered for full
    chunks (in/out/xhat/scaled stay 2·max(Ws,Wc)).
- **Accuracy achieved**: PCC ≥ 0.9994, rel_rms 0.002–0.009 bf16 on 33 refinement cases
  (SD widths 10–80, C-tail+straddle, HW-tail+straddle, all dtypes, multibatch); SDXL
  16384x320 bf16 PCC 0.99975.
- **Golden test progress**: full sweep (7 shape shards): 3683 supported_pass / 3551
  invalid_skipped / 0 xfail remaining / 0 xpass_drift / 0 xfail_wrong_mode / 2 supported_fail
  — both the pre-existing R2 irreducible `1x1x64x17 gamma_only bf8b` cells (op below the
  bf8b gamma quantization floor; test-side tolerance issue, NOT in EXCLUSIONS). Up from
  2361 at Refinement 2. EXCLUSIONS remains empty.
- **Issues encountered**: (1) CB wrap deadlock with variable-width frames (DPRINT-isolated to
  first 2-tile group span); (2) TF32 reload truncation at Ht=512; (3) L1 OOM 1.57MB after
  16-row chunking; one transient infra failure (host disk full → MPI segfault, pruned stale
  JIT caches). Two prior gate tests asserting NotImplementedError for straddle flipped to
  assert support.
- **Tests added**: test_groupnorm_sc_N_1_HW_C_refinement3.py (33 cases); probes/probe_007–012
  (hang isolation, per-group bias and slope decomposition).
