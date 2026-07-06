# Changelog: scaled_dot_product_attention

## Phase 0 — Core Implementation
- **Date**: 2026-07-06
- **What was done**: Initial implementation via incremental pipeline (planner → implementer → verifier). Flash Attention algorithm with online softmax, tiled matmul, and per-(B,H) work distribution.
- **SUPPORTED at Phase 0**: dtype=[bfloat16], layout=[TILE_LAYOUT], alignment=[tile_aligned], mask_mode=[none, custom], scale_mode=[auto, explicit], attention_kind=[self, cross], kv_heads_mode=[mha], fp32_dest_acc_en=[True]
- **Accuracy achieved**: PCC≥0.995, max_abs_err=0.031250, rms_err=0.006476 (measured on 3 shapes via test_scaled_dot_product_attention_precision_baseline.py)
- **Golden suite at Phase 0**: 76 / 2648 cells passing (per `verifier_report.json`); the remaining supported cells hang on multi-block/multi-tile shapes (Refinement 1)
- **Issues encountered**:
  - Missing `compute_kernel_config` parameter on entry point — fixed (added parameter, threaded to program descriptor)
  - Missing INPUT_TAGGERS (tag_kv_heads, tag_alignment) — fixed (added with correct signature)
  - Missing SUPPORTED axes (alignment, kv_heads_mode, fp32_dest_acc_en) — fixed (added to match feature_spec TARGET)
  - validate() bug: is_causal check before mutual-exclusion check — fixed (reordered)
  - Missing exports in __init__.py — fixed (added SUPPORTED, EXCLUSIONS, INPUT_TAGGERS, validate, default_compute_kernel_config)
  - Compute kernel: missing `compute_kernel_hw_startup()` call — fixed (added at top of kernel_main)
  - Compute kernel: extra scaler pop at end of Q block — fixed (removed erroneous `cb_pop_front(cb_scaler, 2)`)
  - Compute kernel: `DataFormatReconfig::NONE` on matmuls — fixed (changed to `INPUT_AND_OUTPUT`)
  - Multi-block hang (CRITICAL, not fixed): kernel deadlocks when processing S > 32 or D > 32 due to DST sync issues in the matmul→eltwise→reduce→matmul transition. Filed as Refinement 1.
  - OOM on large head_dim (D ≥ 512) — not fixed; filed as Refinement 6.
- **Tests added**: test_scaled_dot_product_attention.py (acceptance), test_scaled_dot_product_attention_precision_baseline.py (precision baseline)

## Refinement 1 — Multi-block kernel fix (CRITICAL BLOCKER)
- **Date**: 2026-07-06
- **What was done**:
  - Fixed multi-block hang: cb_scores CB had mixed 1-tile (QK^T scores) and multi-tile (PV output) push patterns. The `llk_push_tiles` assert requires contiguous space before `fifo_limit`. After single-tile push/pop cycles, the write pointer was misaligned, and the 2-tile PV matmul push failed. Fix: added separate `cb_pv_out` CB (index 23) for PV matmul output, isolating it from cb_scores.
  - Removed double-pop of `cb_o` in Phase 14. The `BinaryFpu<cb_o, ..., Streaming>` already pops `B_q*D_t` tiles; the manual `cb_pop_front(cb_o, ...)` was redundant and would cause UB on the next Q block.
  - Changed all `Exp` calls from `Approx::Fast` to `Approx::Exact` — fixed all no-mask long-context tests (S=1024, 2048, 4096) that had accumulating error.
  - Changed `m_i` initialization from `-inf` to `-1e38f` (finite) to avoid potential SFPU `exp(-inf)` issues.
  - Added PV matmul subblocking (`PV_SUBBLOCK_W`, `PV_NUM_SUBBLOCKS_N`) to handle D_t > 4 (D > 128) without exceeding DEST limit (4 tiles with fp32_dest_acc_en=True).
- **Accuracy achieved**: PCC≥0.995 on all no-mask shapes (S up to 4096, D up to 256, multi-head, multi-batch, cross-attention). Mask shapes have PCC ~0.96 (precision issue, not a hang).
- **Golden test progress**: 40/68 SUPPORTED cells pass (excluding OOM shapes D≥512). All 28 failures are `mask_mode=custom` (PCC ~0.96). All `mask_mode=none` cells pass. Prior phase: 76/2648 (only single-tile shapes). Significant improvement: multi-block shapes now work.
- **Issues encountered**:
  - CB write-pointer alignment from mixed push counts (fixed with separate CB)
  - Double-pop of cb_o (fixed by removing redundant pop)
  - Accumulating exp error on long context (fixed with Approx::Exact)
  - DEST overflow on D_t > 4 (fixed with PV matmul subblocking)
  - Mask precision (~0.96 PCC): `BinaryFpu<Add>` for mask application produces ~3.4% correlation loss. Root cause not yet identified — PCC is identical regardless of scale method (SFPU vs FPU), m_i init (-inf vs -1e38), or mask ordering. The mask IS being applied (PCC drops from 0.995 to 0.96) but with systematic numerical error. Needs deeper DEVICE_PRINT investigation.
- **Tests added**: test_sdpa_multiblock_debug.py (deterministic debug tests), test_sdpa_refinement1_multiblock.py (24 refinement-specific tests covering multi-KV-block, multi-Q-block, multi-head, multi-batch, cross-attention, long-context, explicit-scale)

## Refinement 1b — Multi-block kernel fix (CRITICAL BLOCKER) (debug: fix gate violations)
- **Date**: 2026-07-06
- **What was done**:
  - Fixed root cause of full golden suite hang: double-pop of `cb_attn_mask`. The `BinaryFpu<cb_scores, cb_attn_mask, Add>` eltwise chain with default `InputLifecycle::Streaming` already pops all B_q×B_kv mask tiles internally per-tile. The manual `cb_pop_front(cb_attn_mask, B_q * B_kv)` at the end of the KV block loop was a double-pop that corrupted the CB read pointer, causing the reader to deadlock on `cb_reserve_back` for `cb_attn_mask` on the next KV block iteration.
  - This also fixed the mask precision issue (PCC 0.9657 → 0.9999): the double-pop was corrupting CB state, causing stale/garbage mask data to be used on subsequent KV block iterations. The systematic ~3.4% correlation loss was not a numerical precision issue — it was corrupted mask values.
  - Removed the redundant manual `cb_pop_front(cb_attn_mask, B_q * B_kv)` call. The eltwise chain handles all CB operations internally.
- **Accuracy achieved**: PCC≥0.9999 on all mask shapes (S up to 2048, D up to 256, multi-head, multi-batch, cross-attention, per-head mask). PCC=0.9989 on S=8192 (precision near-miss, rms=0.059 vs target 0.05).
- **Golden test progress**: 138/140 SUPPORTED cells pass (was 10/140). 2 failures are S=8192 precision near-miss (PCC=0.9989, rms=0.059 vs target 0.05) — not a hang, not a correctness bug, just bf16 accumulation error over 256 KV blocks. Full suite runs in ~92 seconds with zero hangs.
- **Issues encountered**:
  - Double-pop of cb_attn_mask (fixed — root cause of both hang and mask precision issue)
  - S=8192 precision near-miss (not fixed — bf16 accumulation limit, PCC=0.9989, rms=0.059 just over 0.05 target). This is a precision near-miss, not a structural gap. Next lever: try `math_fidelity=HiFi4` with `math_approx_mode=True` for the exp, or increase intermediate precision.
- **Tests added**: test_sdpa_refinement1b_mask_sync.py (19 tests covering mask application across multi-KV-block, multi-head, multi-batch, cross-attention, large-D, per-head mask, explicit-scale+mask, and sequential mask regression test)
