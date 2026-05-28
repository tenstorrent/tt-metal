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
