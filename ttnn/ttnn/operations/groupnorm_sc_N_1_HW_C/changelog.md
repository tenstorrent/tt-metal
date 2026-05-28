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
