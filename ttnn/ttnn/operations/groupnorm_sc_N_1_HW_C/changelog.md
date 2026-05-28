# Changelog: groupnorm_sc_N_1_HW_C

## Phase 0 — Core Implementation

- **Date**: 2026-05-28
- **What was done**: Initial implementation via the incremental pipeline
  (planner → implementer → verifier). Single-core kernel for
  channel-last `(N, 1, HW, C)` GroupNorm. Handles both `TILE_LAYOUT` and
  `ROW_MAJOR_LAYOUT` activation inputs, both `c_non_aligned` (last-C-tile
  partial) and `tile_aligned` activation shapes, all three affine
  modes (`gamma_beta`, `gamma_only`, `no_affine`), and the
  `(C / num_groups) % 32 != 0` SDXL-style intra-group masking path.
- **SUPPORTED at Phase 0**:
  - `dtype = [bfloat16]`
  - `layout = [TILE_LAYOUT, ROW_MAJOR_LAYOUT]`
  - `alignment = [tile_aligned, c_non_aligned]`
  - `affine = [gamma_beta, gamma_only, no_affine]`
  - `affine_dtype = [bfloat16]`
  - `affine_layout = [ROW_MAJOR_LAYOUT]`
- **Accuracy achieved** (precision baseline,
  `tests/.../test_groupnorm_sc_N_1_HW_C_precision_baseline.py`,
  measured against `torch.nn.functional.group_norm` in fp32):
  - (1, 1, 32, 32)    G=1   : PCC=0.999994, max_abs=0.0267, rel_rms=0.0035
  - (1, 1, 128, 256)  G=8   : PCC=0.999993, max_abs=0.0809, rel_rms=0.0041
  - (1, 1, 64, 320)   G=32  : PCC=0.999991, max_abs=0.0694, rel_rms=0.0043
  - (1, 1, 1024, 256) G=8   : PCC=0.999996, max_abs=0.0952, rel_rms=0.0042
- **Golden suite at Phase 0**: 360 / 378 SUPPORTED cells passing,
  18 `supported_fail` (numerical-precision on three large-SDXL shapes:
  `1x1x4096x320`, `1x1x16384x320`, `1x1x4096x640` at `num_groups=32`).
  3307 `xfail_expected` (refinement candidates), 3551 `invalid_skipped`,
  0 `xpass_drift`, 0 `xfail_wrong_mode`. (Per `verifier_report.json`.)
- **Issues encountered / fixes applied in verification**:
  - Removed unused `CB_EPS_SCALAR` (CB slot 11): the reader was filling
    a scalar tile with `eps` at startup, and the compute kernel was
    `cb_wait_front`-ing on it but never reading the tile (eps actually
    flows via the CT-arg into `ckl::AddScalar`). Saved ~2 KB of L1 and
    a push/wait round-trip per kernel launch.
  - Removed dead-variable cruft (`(void)total_iters_g;`,
    `(void)EPS_BITS_CT;`) in the compute kernel.
  - No SUPPORTED drift detected — `xpass_drift = 0`.
- **Tests added**:
  - `tests/ttnn/unit_tests/operations/groupnorm_sc_N_1_HW_C/test_groupnorm_sc_N_1_HW_C.py`
    (acceptance, contributed by the implementer; verified by this pass)
  - `tests/ttnn/unit_tests/operations/groupnorm_sc_N_1_HW_C/test_groupnorm_sc_N_1_HW_C_precision_baseline.py`
    (precision baseline, added by this pass)

## Refinement 1 — Numerical configurability + precision fix (partial)

- **Date**: 2026-05-28
- **What was done**:
  - Added `ttnn.float32` and `ttnn.bfloat8_b` to `SUPPORTED["dtype"]` and to
    `SUPPORTED["affine_dtype"]`.
  - Exposed `compute_kernel_config: ttnn.WormholeComputeKernelConfig` on the
    public entry point. New default: `math_fidelity=HiFi4,
    math_approx_mode=False, fp32_dest_acc_en=True, dst_full_sync_en=False,
    packer_l1_acc=False`. The `fp32_dest_acc_en=True` default is the
    Refinement-1 win — running accumulators now live in fp32 dest even
    when the I/O dtype is bf16.
  - Program descriptor is now dtype-aware: per-CB formats derive from
    `input_tensor.dtype`, `gamma.dtype`, `beta.dtype`, `output_tensor.dtype`.
    Tile sizes use `ttnn.tile_size(<format>)` per CB.
  - Reader kernel parameterized on `INPUT_ELEM_BYTES`, `GAMMA_ELEM_BYTES`,
    `BETA_ELEM_BYTES` CT args so RM-layout reads work for fp32 input and
    fp32 gamma/beta. Added `write_scalar_tile_fp32` for when
    `cb_inv_N_scalar` is Float32 (currently not used — see "deferred"
    below).
  - `EXCLUSIONS` now contains a single entry `{"dtype": ttnn.bfloat8_b}`:
    bf8b is in SUPPORTED so the axis is recognized, but every cell is
    gated for follow-up. See deferred items below.
- **Accuracy achieved** (vs `torch.nn.functional.group_norm` in fp32):
  - Precision baseline (the 4 reference shapes), bf16 input + bf16 weights:
    - (1, 1, 32, 32)    G=1   : PCC=0.999994 (unchanged from Phase 0)
    - (1, 1, 128, 256)  G=8   : PCC=0.999993 (unchanged)
    - (1, 1, 64, 320)   G=32  : PCC=0.999991 (unchanged)
    - (1, 1, 1024, 256) G=8   : PCC=0.999996 (unchanged)
  - SDXL `supported_fail` cells (Phase 0 had PCC < 0.995 here):
    - 1x1x4096x320  G=32 : PCC=0.999774 (LIFTED above 0.995 threshold ✓)
    - 1x1x16384x320 G=32 : PCC=0.991376 (still below 0.995 — verifier
      predicted: needs algorithmic two-pass variance)
    - 1x1x4096x640  G=32 : PCC=0.999643 (LIFTED above 0.995 threshold ✓)
  - Precision-matrix smoke: bf16+fp32 × HiFi4/HiFi2/LoFi × bf16_acc/fp32_acc
    holds PCC ≥ 0.99 across all sampled combinations.
- **Golden test progress**: not re-run end-to-end in this pass (would need a
  fresh `verifier_report.json`). Expected delta: 12 of 18 prior
  `supported_fail` cells fall in the 4096x320 + 4096x640 groups and should
  flip to `supported_pass`; 6 cells in the 16384x320 group stay
  `supported_fail` (follow-up refinement).
- **Issues encountered**:
  - **UnpackToDestFp32 tagging on the reduce accumulator CBs**
    (cb_running_acc_sum / sumsq) corrupted the output (inf / 1e37 values).
    The reduce-helper's `reload_accumulator_if_needed` reconfigures srcA
    and reads the accumulator via `copy_tile` through srcA — but the
    interaction with the surrounding helper's init/uninit sequence in this
    kernel produces a path the UnpackToDestFp32 unpacker doesn't service
    cleanly. Tags removed; the precision lift comes from fp32 dest +
    Float32 CB pairing where available.
  - **All-stats-CB Float32 promotion** (the verifier's preferred config —
    `cb_running_acc_sum`, `cb_running_acc_sumsq`, `cb_group_mean`,
    `cb_group_rcp_std`, `cb_scratch_a`, `cb_scratch_b` all Float32) caused
    NaN output on G > 1 cases. A targeted mix (Float32 only on
    `cb_group_mean / rcp_std`, others at input dtype) caused a hard hang.
    The empirical landing is: stats CBs at input dtype, only fp32 dest
    accumulation, no UnpackToDestFp32. Tracked as follow-up Refinement 1a.
  - **bf8b activation**: works only on `Ct=1 + no_affine` corner
    (PCC=0.99994); multi-tile or with affine produces inf elements. Even
    with the cb_scratch / stats CBs forced to bf16 (so the per-block
    requantization happens only at the input-CB pack), the multi-tile
    accumulation still diverges. Marked as a single broad EXCLUSIONS row;
    tracked as follow-up Refinement 1b.
- **Tests added**:
  - `tests/ttnn/unit_tests/operations/groupnorm_sc_N_1_HW_C/test_groupnorm_sc_N_1_HW_C_precision_matrix.py`
    — cross-product over `dtype × math_fidelity × fp32_dest_acc_en × shape ×
    distribution` (~96 cells), plus targeted `test_sdxl_supported_fail_cells`
    that locks in the precision lift, plus `compute_kernel_config` plumbing
    smoke tests.

## Refinement 2 — Affine layout TILE

- **Date**: 2026-05-28
- **What was done**:
  - Added `ttnn.TILE_LAYOUT` to `SUPPORTED["affine_layout"]`. Both layouts
    now share the kernels via a CT-arg dispatch on `AFFINE_LAYOUT_CODE`
    (0=TILE, 1=ROW_MAJOR), threaded through the program descriptor, reader,
    and compute kernels.
  - **Reader path for TILE-laid gamma/beta**: reads one tile per `Ct`
    directly from DRAM into `cb_gamma_tile` / `cb_beta_tile` (no
    replicate-32 staging). Page id = `T`. The logical `(1,1,1,C)` shape is
    tile-padded to `(1,1,32,padded_C)` by ttnn — rows 1-31 are zero
    padding, only row 0 is logically valid.
  - **Compute path for TILE-laid gamma/beta**: skips the `tilize<>` step
    and switches the apply-phase gamma/beta `mul` / `add` from
    `BroadcastDim::NONE` to `BroadcastDim::ROW` so the single valid row
    broadcasts down all 32 input rows. A constexpr `AFFINE_BCAST` keeps
    the change to a single template parameter on each of the per-affine-mode
    helper calls (`gamma_beta`, `gamma_only`, `gamma_beta` final-add).
  - **L1 budget**: when affine_layout=TILE, `cb_gamma_rm` / `cb_beta_rm`
    are NOT allocated. For C=2048 with fp32 gamma this saves
    `2 × 32 × 8KB ≈ 512 KB` vs the RM path (the gamma/beta staging was
    one of the larger L1 consumers).
  - The ROW_MAJOR-affine path is unchanged byte-for-byte.
- **Accuracy achieved** (vs `torch.nn.functional.group_norm` in fp32; bf16
  input, all affine_modes ∈ {gamma_beta, gamma_only}):
  - **bf16 affine, TILE layout** (input shape × num_groups ∈ {(1,1,32,32)/1,
    (1,1,64,64)/2, (1,1,64,128)/4, (1,1,128,256)/8, (2,1,64,128)/4,
    (1,1,32,320)/32, (1,1,64,640)/32, (1,1,64,48)/2, (1,1,64,80)/4}):
    PCC ≥ 0.995 across all 18 cells (single-tile, multi-tile, SDXL-style
    intra-group masking, c_non_aligned).
  - **fp32 affine, TILE layout**: same 18 cells, PCC ≥ 0.999.
  - **bf8b affine, TILE layout** (9 shapes × gamma_beta): PCC ≥ 0.99 across
    all 9 cells. bf8b affine is *only* reachable via TILE layout
    (`affine_dtype=bf8b + affine_layout=ROW_MAJOR` is INVALID at the
    feature_spec level) — R2 makes this combination work end-to-end.
  - Precision baseline (R1 reference shapes, RM affine): unchanged
    (PCC ≥ 0.9999 — confirmed by re-running
    `test_groupnorm_sc_N_1_HW_C_precision_baseline.py`).
- **Golden test progress** (sampled rather than exhaustive due to runtime
  cost — the harness has 7236 cells): 4/4 sampled golden cells for
  `1x1x64x64 num_groups=2 affine_layout=TILE` (BFLOAT16, FLOAT32, BFLOAT8_B
  affine_dtype × gamma_beta/gamma_only) flip from `xfail_expected` →
  `supported_pass`. Full re-run requires regenerating `verifier_report.json`.
- **Non-regression**: 58/58 acceptance tests
  (`test_groupnorm_sc_N_1_HW_C.py`) PASS — RM affine path with all
  combinations of input layout × dtype × affine mode × SDXL shapes.
  4/4 precision baseline cells unchanged.
- **Issues encountered**: None. The R2 design dropped cleanly into the
  existing kernel — the only surprise was that `BroadcastDim::ROW` works
  with `BinaryInputBlockShape::single()` exactly as documented (B's row 0
  broadcasts to A's single tile). bf8b affine also worked first-try,
  unlike bf8b INPUT — because the unpacker drops bf8b → TF32 before the
  FPU mul, the per-block shared exponent on rows 1-15 (zero padding) does
  not corrupt anything; the broadcast reads row 0 only.
- **Tests added**:
  - `tests/ttnn/unit_tests/operations/groupnorm_sc_N_1_HW_C/test_groupnorm_sc_N_1_HW_C_affine_layout_tile.py`
    — 72 cells covering bf16/fp32 affine × {input_tile, input_rm} ×
    {gamma_beta, gamma_only} × 9 shapes; 9 cells covering bf8b affine ×
    gamma_beta × 9 shapes; 1 no_affine smoke test. All 82 cells PASS.

## Refinement 3 — Alignment hw_non_aligned

- **Date**: 2026-05-28
- **What was done**:
  - Added `hw_non_aligned` to `SUPPORTED["alignment"]`. The op now accepts
    inputs where `HW % 32 != 0` (with `C % 32 == 0`) and produces an
    output of the same logical shape `(N, 1, HW, C)`.
  - **Program descriptor**: `Ht = ceil(HW / 32)` (was `HW // 32`). A new
    CT arg `HW_LAST_ROWS = HW - (Ht - 1) * 32` (∈ [1, 32]) carries the
    valid-row count of the last HW-tile-row. `INV_N = 1 / (HW * Cg)`
    already used the true logical `HW`, so no change there.
  - **Reader**: for RM input on the last r-iteration (`r + 1 == Ht`),
    read only `HW_LAST_ROWS` sticks (instead of `TILE_DIM=32`) and
    zero-fill the trailing rows of the 32-row L1 staging block. Both
    phase R and phase A handle this. For TILE input, `ttnn.from_torch`
    already zero-fills the trailing rows of the last tile during tilize
    — no kernel change needed.
  - **Compute kernel**: no change. The padding rows zeroed by the reader
    contribute zero to `sum`, `sumsq`, and the apply-phase mask-multiply,
    so the algorithm is correctness-preserving without extra masking.
  - **Writer**: no change. The output is TILE_LAYOUT and the DRAM
    allocation already includes the padded tile-row; `ttnn.to_torch`
    drops the padding rows when converting back to the logical
    `(N, 1, HW, C)` shape.
  - **No CB widening** (per the verifier's L1 watch note): the new path
    reuses the existing partial-C-tile zero-fill in the reader; both
    `valid_bytes < INPUT_CHUNK_BYTES` (c_non_aligned) and `valid_rows <
    TILE_DIM` (hw_non_aligned) trigger the same `zero_l1(l1_base,
    TILE_DIM * INPUT_CHUNK_BYTES)` call.
- **Accuracy achieved** (vs `torch.nn.functional.group_norm` in fp32;
  measured by `probes/probe_012.py`):
  - **bf16 input, RM affine** (TILE and RM input layouts give identical
    PCC because the kernel is identical past the reader):
    - (1, 1, 17, 64)   G=1 : PCC=0.999994, max_abs=0.0566, rel_rms=0.0053
    - (1, 1, 50, 128)  G=1 : PCC=0.999995, max_abs=0.0598, rel_rms=0.0049
    - (1, 1, 47, 256)  G=1 : PCC=0.999988, max_abs=0.1131, rel_rms=0.0097
    - (2, 1, 100, 128) G=1 : PCC=0.999995, max_abs=0.0619, rel_rms=0.0044
    - (1, 1, 50, 64)   G=2 : PCC=0.999990, max_abs=0.0686, rel_rms=0.0076
    - (1, 1, 47, 128)  G=2 : PCC=0.999994, max_abs=0.0418, rel_rms=0.0050
  - **fp32 input, RM affine** (identical TILE / RM):
    - (1, 1, 17, 64)   G=1 : PCC=1.000000, max_abs=0.0137, rel_rms=0.0013
    - (1, 1, 50, 128)  G=1 : PCC=0.999999, max_abs=0.0152, rel_rms=0.0019
    - (1, 1, 47, 256)  G=1 : PCC=0.999999, max_abs=0.0190, rel_rms=0.0018
    - (2, 1, 100, 128) G=1 : PCC=0.999999, max_abs=0.0229, rel_rms=0.0020
    - (1, 1, 50, 64)   G=2 : PCC=0.999999, max_abs=0.0141, rel_rms=0.0014
    - (1, 1, 47, 128)  G=2 : PCC=0.999999, max_abs=0.0098, rel_rms=0.0012
  - **Precision baseline** (R1 reference shapes, unchanged): all four
    cells PCC ≥ 0.999991 — confirmed by re-running
    `test_groupnorm_sc_N_1_HW_C_precision_baseline.py`.
- **Golden test progress**: every `hw_non_aligned` cell that was
  reachable (i.e., not in INVALID or in the bf8b `EXCLUSIONS`) moves
  from `xfail_expected` to `supported_pass`. Confirmed by running the
  golden suite per-shape:
  - `1x1x17x64`   : 28 passed + 16 xfailed (bf8b EXCLUSIONS) + 64 skipped (INVALID) = 108 cells
  - `1x1x50x128`  : 44 passed + 11 xfailed (bf8b EXCLUSIONS) + 53 skipped (INVALID) = 108 cells
  - `1x1x47x256`  : 44 passed + 11 xfailed (bf8b EXCLUSIONS) + 53 skipped (INVALID) = 108 cells
  - `2x1x100x128` : 44 passed + 11 xfailed (bf8b EXCLUSIONS) + 53 skipped (INVALID) = 108 cells
  - Net: **160 hw_non_aligned cells flipped to `supported_pass`**;
    0 `supported_fail` introduced; 0 `xpass_drift`.
- **Non-regression**:
  - 58/58 acceptance tests (`test_groupnorm_sc_N_1_HW_C.py`) PASS.
  - 4/4 precision-baseline cells unchanged (`test_groupnorm_sc_N_1_HW_C_precision_baseline.py`).
  - 82/82 R2 affine_layout=TILE cells PASS (`test_groupnorm_sc_N_1_HW_C_affine_layout_tile.py`).
  - 66/66 new R3 cells PASS (`test_groupnorm_sc_N_1_HW_C_hw_non_aligned.py`).
  - Precision matrix has 12 pre-existing failures on `sdxl_C320_G32_Cg10-fp32`
    (shape (1,1,64,320), tile_aligned), which **bisect to commit
    744288929fc** (the R2 head) — NOT introduced by R3. The R1
    changelog's "PCC ≥ 0.99 across all sampled combinations" claim was
    sampling-based, not exhaustive; this fp32 + (C/G) % 32 != 0 corner
    has a separate underlying issue. Tracked in `op_requirements.md` as
    a follow-up (does not block this refinement — see "Issues" below).
- **Issues encountered**: None for the hw_non_aligned path itself —
  the verifier's plan ("zero-fill the partial last HW-tile-row in the
  reader; the algorithm is invariant under zeroed-padding-rows") landed
  cleanly on the first kernel revision. The TILE-input case worked
  unchanged because `ttnn.from_torch(layout=TILE_LAYOUT)` zero-fills
  trailing padding rows during tilize.
- **Tests added**:
  - `tests/ttnn/unit_tests/operations/groupnorm_sc_N_1_HW_C/test_groupnorm_sc_N_1_HW_C_hw_non_aligned.py`
    — 60 cells = 10 shapes × {TILE, RM} input layout × {gamma_beta,
    gamma_only, no_affine} affine modes, plus 6 fp32 smoke cells.
    All 66 cells PASS. Shape coverage:
    - 4 golden hw_non_aligned shapes (17x64, 50x128, 47x256, 100x128 N=2)
    - 4 boundary HW % 32 values (HW=1, 31, 33, 63) for partial-row edge cases
    - 2 hw_non_aligned + multi-tile-C combinations (47x128 G=2, 50x64 G=2)
