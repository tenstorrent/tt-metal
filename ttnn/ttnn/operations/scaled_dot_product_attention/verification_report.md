# Verification Report: scaled_dot_product_attention

## Code Review

### Fixed in this pass

1. **Missing `compute_kernel_config` parameter on entry point** — The golden test helper (`helpers.py`) calls `scaled_dot_product_attention(..., compute_kernel_config=...)` but the op's entry point didn't accept this parameter. Added `compute_kernel_config: ttnn.ComputeConfigDescriptor | None = None` to the signature and threaded it through to `create_program_descriptor`.

2. **Missing INPUT_TAGGERS** — The feature_spec declares three tagger-derived axes (`attention_kind`, `kv_heads_mode`, `alignment`) but the op file only had `tag_attention_kind`. Added `tag_kv_heads` (mha/gqa/mqa) and `tag_alignment` (tile_aligned/w_non_aligned/h_non_aligned) taggers with the correct `(inputs, axes)` signature.

3. **Missing SUPPORTED axes** — Added `alignment`, `kv_heads_mode`, and `fp32_dest_acc_en` axes to SUPPORTED to match the feature_spec TARGET. Phase 0 SUPPORTED: alignment=tile_aligned only, kv_heads_mode=mha only, fp32_dest_acc_en=True only.

4. **validate() bug: is_causal check order** — The original code checked `is_causal` and raised `UnsupportedAxisValue` BEFORE checking the `is_causal + attn_mask` mutual exclusion. Fixed: the mutual-exclusion ValueError is now checked first, then `is_causal=True` is mapped to `mask_mode=causal` (which is not in SUPPORTED, so validate() rejects it via the SUPPORTED check).

5. **Missing exports in `__init__.py`** — The op's `__init__.py` only exported `scaled_dot_product_attention`. Added exports for `SUPPORTED`, `EXCLUSIONS`, `INPUT_TAGGERS`, `default_compute_kernel_config`, and `validate` (needed by golden test `test_golden.py` and `axes.py`).

6. **Compute kernel: missing `compute_kernel_hw_startup()` call** — The compute kernel included `compute_kernel_hw_startup.h` but never called `compute_kernel_hw_startup()`, which is required before any compute API per the helper documentation. Added the call at the top of `kernel_main()`.

7. **Compute kernel: extra scaler pop** — The compute kernel popped 2 scaler tiles per KV block (correct: one after MAX reduce, one after SUM reduce) but ALSO popped 2 more at the end of each Q block (`cb_pop_front(cb_scaler, 2)`). This caused a scaler CB underflow on multi-Q-block shapes. Removed the erroneous extra pop.

8. **Compute kernel: `DataFormatReconfig::NONE` on matmuls** — Both matmul_block calls (QK^T and PV) used `DataFormatReconfig::NONE`, which skips the unpacker/packer reconfiguration needed when transitioning between eltwise/reduce and matmul operations. Changed to `DataFormatReconfig::INPUT_AND_OUTPUT` (the default) so the hardware is properly reconfigured for matmul mode after eltwise chains.

### Known issues (not fixed in this pass)

1. **Multi-block hang (CRITICAL)** — The compute kernel hangs when processing more than 1 Q block, more than 1 KV block, or D_t > 1 (head dim > 32). The hang manifests as a deadlock between:
   - The unpacker (trisc0) waiting for a CB to be filled (e.g., cb_scores or cb_psum)
   - The packer (trisc2) stuck in `llk_push_tiles` → `TTI_STALLWAIT(PACK)` waiting for the pack hardware
   - The math (trisc1) stuck in `mop_sync` waiting for the math hardware to finish a previous MOP

   Root cause is likely a DST sync issue: the unpacker runs ahead of the math/packer pipeline, and the `tile_regs_wait()`/`tile_regs_commit()` sync mechanism doesn't properly serialize the multi-stage transitions (matmul → eltwise → reduce → matmul). The single-block case (S=32, D=32) works because there's only one Q block, one KV block, and D_t=1 (PV matmul produces 1 output tile).

   This blocks all shapes with S > 32 or D > 32. Filed as a refinement (Refinement 1).

2. **OOM on large head_dim (D ≥ 512)** — CBs are sized as `B_q × D_t` tiles, which for D=1024 (D_t=32) produces large L1 allocations. The `cb_o` CB (`B_q × D_t × tile_size`) exceeds the 1.5 MB L1 budget around D_t=16 (D=512). Filed as a refinement (Refinement 3).

3. **Dead `cb_scale` CB** — The `cb_scale` CB (index 4) is declared in the program descriptor but never used — the compute kernel uses `MulUnary<>{scale_bits}` (compile-time scalar) instead of a runtime scale tile. This is harmless (1 page, never pushed) but wasteful.

4. **Reader re-pushes Q per KV block** — The reader kernel re-reads Q tiles from DRAM for every KV block iteration. This is correct but inefficient; Q tiles are the same across KV blocks and could be cached in L1.

## Registry Conformance

- Confirmed: INPUT_TAGGERS (3 taggers: attention_kind, kv_heads_mode, alignment), SUPPORTED (8 axes), EXCLUSIONS (empty), validate() all present and correctly wired in the op file.
- Confirmed: op file does NOT declare INVALID (it's a test-suite concept in feature_spec.py).
- Confirmed: `validate()` checks SUPPORTED per-axis first, then EXCLUSIONS (cell-level), in the correct order.
- Confirmed: public entry point calls `validate()` as its first line.
- Confirmed: all INPUT_TAGGERS functions have the `(inputs, axes)` signature.
- No auto-fixes applied to SUPPORTED based on XPASS evidence — the xpass_drift cells from the initial golden run were due to the old op file lacking axes; the updated op file correctly rejects them.

### INVALID audit (feature_spec.py)

- `INVALID = []` — the feature_spec declares no structurally-impossible cells.
- Rationale: TARGET["layout"] is TILE-only (SDPA is TILE-only by design), so the canonical bf8b+ROW_MAJOR rule is vacuous — no ROW_MAJOR cell exists in the cartesian product to forbid.
- Well-formedness: no cross-tensor-axis entries, no "my kernel doesn't support this yet" entries.
- The INVALID list is correct for this op.

## Precision Baseline

| Shape | PCC | Max Abs Err | Mean Abs Err | Relative RMS Err |
|-------|-----|-------------|--------------|------------------|
| (1,1,32,32) | ≥0.995 | 0.023438 | 0.005083 | 0.021625 |
| (1,4,32,32) | ≥0.995 | 0.031250 | 0.004489 | 0.021730 |
| (2,4,32,32) | ≥0.995 | 0.031250 | 0.004239 | 0.021171 |

**Assessment**: The single-tile path produces correct results with PCC ≥ 0.995 and relative RMS error ~2%. Error is dominated by bf16 matmul rounding in the QK^T and PV matmuls.

**Recommended tolerances**: PCC >= 0.995, rtol=0.1, atol=0.05

## Verifier CLI Summary

The golden test suite was run with the updated op file. Due to the multi-block hang, only single-tile shapes (S=32, D=32) produce passing results.

- supported_pass: 76 (single-tile bf16 shapes across mask/scale variations)
- xfail_expected: 2508 (correctly rejected by validate())
- invalid_skipped: 0
- supported_fail: 64 (multi-tile shapes that hang/OOM + large head_dim OOM)
- xpass_drift: 0
- xfail_wrong_mode: 0

Note: The supported_fail cells are all either:
- Multi-block hangs (S > 32 or D > 32) — timeout classified as numerical-precision
- Large head_dim OOM (D ≥ 512)

## Recommendations

1. **Multi-block hang (Refinement 1)** — This is the critical blocker. The kernel needs a fundamental fix to the DST sync and matmul→eltwise→matmul transition. The `compute_kernel_hw_startup` should be called at boot, and the eltwise chain's `binary_op_init_common` or equivalent should be called before each stage that transitions between matmul and eltwise/reduce operations.

2. **L1 budget for large head_dim (Refinement 3)** — The `cb_o` CB scales as `B_q × D_t` tiles. For D=1024 (D_t=32), this is 32 tiles × 2 KB = 64 KB, which is fine. But `cb_pv` is `B_q × B_kv` and `cb_scores` is `max(B_q × B_kv, B_q × D_t)`. The real OOM comes from the Q/K/V stream CBs which scale with `D_t`. The `/memory-budget-metal` skill's streaming-reduce wrapper could help.

3. **Dead `cb_scale` CB** — Remove the `cb_scale` CB declaration and its page allocation. The `MulUnary<>{scale_bits}` compile-time scalar approach is correct and doesn't need a runtime tile.

4. **Reader Q re-push optimization** — Cache Q tiles in L1 across KV block iterations instead of re-reading from DRAM. This is a performance optimization, not a correctness fix.
