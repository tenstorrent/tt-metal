# Changelog: typecast

## Phase 0 — Core Implementation
- **Date**: 2026-11-21
- **What was done**: initial implementation via the incremental pipeline (planner → implementer → verifier). Regime `tile_stream_float` (op_design.md): reader → `cb_input_tiles` → compute (`compute_kernel_lib::typecast`, 32-bit DEST; UnpackToDest for fp32 input, UnpackToSrc for bf16) → `cb_output_tiles` → writer. One Metal 2.0 `ProgramSpec`, one `ttnn.generic_op` dispatch, all clusters via `split_work_to_cores` (row-wise, contiguous ranges), thread plan (1, 1, 1) from `THREAD_PLAN`.
- **SUPPORTED at Phase 0**: dtype=[bfloat16, float32], target_dtype=[bfloat16, float32], layout=[TILE], memory_config=[DRAM], sharding=[interleaved], alignment=[tile_aligned], rank=[4] (= PHASE0)
- **Accuracy achieved**: bit-exact on all four pairs (0 mismatches against torch `Tensor.to`). fp32→bf16 versus the float64 input: PCC=0.9999986, max_abs_err=1.54e-2, rel_rms=1.66e-3, ≤ 0.5 ULP (torch's RNE rounding). The three other pairs have zero error. Measured on 4 shapes via `test_typecast_precision_baseline.py`.
- **Golden suite at Phase 0**: 24 / 24 supported cells passing; 2285 xfail_expected; 832 skipped (INVALID + bfloat8_b unbuildable on quasar); 0 supported_fail / xpass_drift / xfail_wrong_mode; queue_gaps 0 (per `verifier_report.json`)
- **Issues encountered**:
  - Implementer: `block_tiles` is capped at one DEST section (4 tiles). The eltwise chain clamps a block to DEST and syncs the rings per clamped window, so a larger block let a window straddle the ring end. Details in `l1_ledger.md`.
  - Verifier: `op_design.md` corrected to match that cap (Mechanism caps row, `block_tiles_cap`, worked values). Added the R7 sharded-input refusal test the specification requires the op to carry. Removed a stray `noqa`. No drift fixes were needed.
- **Tests added**: `test_typecast.py` (acceptance, generated); `test_typecast_mechanisms.py` covers:
  - the thread plan read off the ProgramSpec, and other plans bit-exact;
  - RNE ties;
  - fp32 keeps every bit;
  - multi-block / ragged tails / ring wrap;
  - the device session (tag hand-back);
  - output placement;
  - refusals, including a sharded input refused by `sharding`.

  `test_typecast_precision_baseline.py` was added by the verifier.
- **Performance**: not measured (ttsim has no clock). Structural counts:
  - each input and output tile crosses DRAM once;
  - zero cross-core traffic;
  - all 32 clusters are engaged once tensor_tiles ≥ 32;
  - 1 NEO, 1 reader hart and 1 writer hart per cluster.
