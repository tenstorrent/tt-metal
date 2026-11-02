# Changelog: conv2d_nhwc

## Phase 0 — Core Implementation
- **Date**: 2026-11-02
- **What was done**: Initial implementation via the incremental pipeline (planner → implementer → verifier).
  - Regime R0: an im2col gather (reader) → tilize → K-blocked `matmul_block` with packer-L1 accumulation → in-place bias → untilize → DRAM write.
  - 2-D (output-pixel × output-channel) split over the full grid.
  - One `ttnn.generic_op` dispatch per call.
- **Verifier changes**:
  - **R10 column-clipped input window.** A 2-D clip for single-output-row blocks, and a trimmed contiguous range for multi-row blocks. This fixed 9 SUPPORTED tt-forge cells that failed with "window exceeds L1".
  - **Two-tier L1 budget.** The 768 KiB preference first, then the hard cap `get_max_worker_l1_unreserved_size() − 128 KiB`. This fixed SDXL 32×32×1280 s2 at fp32.
  - **SUPPORTED `stride_mode += strided`.** The kernels were already stride-parametric; probed and golden-verified.
  - **DRY fixes:**
    - compute CB indices are now CT args;
    - `DST_TILES`, `ACC_DTYPE` and the compute config derive from `MATH_FIDELITY` / `FP32_DEST_ACC_EN`;
    - weight tile bytes come from the weight dtype.
  - **Writer:** `writes_flushed` per tile-row and one final write barrier.
  - **`validate()`:** checks the prepared weight's K rows against C_in / kernel_size / groups.
- **SUPPORTED at Phase 0**:
  - dtype = [float32], weight_dtype = [float32], layout = [ROW_MAJOR], memory_layout = [INTERLEAVED];
  - bias_mode = [with_bias, no_bias], activation = [none];
  - channel_alignment = [tile_aligned], stride_mode = [unit, strided], groups_mode = [dense], dilation_mode = [unit].
- **Accuracy achieved**: PCC ≥ 0.999997, max_abs_err ≤ 1.1e-2, rel_rms_err ≈ 1.1–1.2e-3, ULP p50 ≈ 1.3e4 (tf32 operand truncation). Measured on 5 shapes via `test_conv2d_nhwc_precision_baseline.py`.
- **Golden suite at Phase 0**: 172 supported_pass, 0 supported_fail / xpass_drift / xfail_wrong_mode, 2262 xfail_expected, 1104 invalid_skipped (per `verifier_report.json`). As handed over: 133 pass, 9 supported_fail.
- **Issues encountered**:
  - The 9 window-OOM `supported_fail` cells, fixed in place.
  - The planner's claim that no shape reaches R10 did not hold for the translated tt-forge sweep.
- **Tests added**:
  - `test_conv2d_nhwc.py`: strided and wide-row column-clip cases; the stride rejection check moved to dilation.
  - `test_conv2d_nhwc_precision_baseline.py`.
