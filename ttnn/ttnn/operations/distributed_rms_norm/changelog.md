# Changelog: distributed_rms_norm

## Phase 0 — Core Implementation
- **Date**: 2026-10-14
- **What was done**: Initial implementation via incremental pipeline (planner → implementer → verifier).
  - One `ttnn.generic_op(MeshProgramDescriptor)` per call.
  - x is resident in L1 across a flat fp32 record all-gather over Fabric (port cores, ready fence, arrival counters, one-hop relay).
  - Rows × W-chunks are spread over the full worker grid.
- **SUPPORTED at Phase 0** (after verifier drift fixes):
  - dtype=[float32, bfloat16], fp32_dest_acc_en=[True], layout=[TILE]
  - alignment=[tile_aligned, h_non_aligned], rank=[2, 3, 4]
  - gamma_mode=[gamma, no_gamma], gamma_dtype=[float32, bfloat16, "none"]
  - memory_layout=[INTERLEAVED], cluster_axis=[0, 1], topology=[Linear, Ring], num_links=[1, 2, None]
- **Accuracy achieved** (worst device, 4 shapes, `test_distributed_rms_norm_precision_baseline.py`):
  - bf16: PCC=0.999998, max_abs_err=0.043, rel_rms_err=0.0019, max 1 bf16 ULP.
  - fp32: PCC=1.000000, max_abs_err=0.031, rel_rms_err=0.0015. There is a systematic −0.1 % bias (tf32 FPU operand truncation), noted on Refinement 4.
- **Golden suite at Phase 0**: 931 supported_pass, 0 supported_fail, 0 xpass_drift, 1 xfail_wrong_mode (harness L1-profiling artifact on a correct refusal), 7125 xfail_expected, 7776 invalid_skipped (per `verifier_report.json`). Acceptance 27/27.
- **Device time** (bf16 + gamma, axis 1, L=2, max over 4 devices): 640×1792 24.9 µs, 32×1792 10.9 µs, 2048×1280 44.6 µs.
- **Issues encountered / verifier fixes**:
  - **Capacity failure fixed.** Per-device (2, 2048, 4096) bf16 + gamma raised the op's capacity error: 1.39 MB, of which 0.70 MB was the fp32 `cb_xg` prescale block. Added a per-call `GAMMA_PRESCALE` regime choice: when the prescale does not fit, gamma is fused into normalize via DEST-reuse Mul against row-replicated gamma tiles, and the footprint drops to 0.68 MB.
  - **Registry drift fixed:** `num_links=None` was missing from SUPPORTED (197 xpass_drift).
  - **Promoted on evidence:**
    - `Ring` (ring mock 16/16, plus fp32 / no-gamma / repeat bit-identical probe);
    - `h_non_aligned` (144/144 cells).
  - **DRY fixes:**
    - the writer's output batch now comes from `out_batch_tiles_knob` (RT arg), not a hard-coded `cw`;
    - `RECORD_BYTES` is passed as a CT arg instead of a literal in two kernels.
  - **Caller errors added:** non-ROW_MAJOR gamma, and an explicit `num_links` beyond the port-column cap. The latter was silently clamped before.
  - `l1_ledger.md` was brought current (`MIN_CORE_W_TILES`=8, conditional `cb_xg`, worst-case footprints on the 11×10 grid).
- **Tests added**: test_distributed_rms_norm.py (planner), test_distributed_rms_norm_precision_baseline.py, test_distributed_rms_norm_perf_bench.py (device-time bench for the LOOSE_CASES configs), probes/probe_001.py (Ring verification probe).
