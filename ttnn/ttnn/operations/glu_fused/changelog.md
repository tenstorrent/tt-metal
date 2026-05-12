# Changelog: glu_fused

## Phase 0 — Core Implementation

- **Date**: 2026-05-12
- **What was done**: Initial implementation via incremental pipeline
  (planner → implementer → verifier). Single fused TTNN kernel folding
  the GLU composite (`slice + slice + sigmoid + multiply`) into one
  `ttnn.generic_op` dispatch. Split happens at the tile-id level inside
  the reader; chain `Load A → Load B → Sigmoid<Exact> → SfpuMul` runs
  inside `sfpu_pipeline` on DEST.
- **Accuracy achieved** (vs `torch.nn.functional.glu`, fp64 reference):
  - PCC ≈ 1.0 (≥ 0.9999999999999991 across all measured shapes)
  - max_abs ≈ 1–4 × 10⁻⁷ (essentially fp32 eps)
  - mean_abs ≈ 4 × 10⁻⁹
  - relative RMS ≈ 3 × 10⁻⁸
  - Measured on 4 shapes: `(1,1,32,64)`, `(1,1,32,128)`,
    `(1,1,256,128)`, `(2,2,128,256)`.
  - Achieved precision is at the fp32 floor — error is dominated by the
    Wormhole sigmoid LUT (6-piece PLU), not by accumulation or pack
    truncation.
- **Issues encountered and fixed during verification**:
  - **Reader**: original kernel issued one `noc_async_read_barrier` per
    tile read (two barriers per output tile). Fixed: issue both A and B
    reads concurrently on NoC0, single barrier per output tile. Halves
    the per-iter barrier count; mirrors the canonical two-input reader
    pattern (`backward_softmax_reader.cpp:86-97`).
  - **Compute**: `sfpu_pipeline` was called with all four template
    parameters explicitly listed, all matching the helper's defaults.
    Simplified to default-args call with a comment recording which
    defaults are being relied on. Behavior unchanged.
- **Tests added**:
  - `test_glu_fused.py` (acceptance — 20 tests covering correctness,
    call patterns, structural split-offset check, and the negative
    validation matrix).
  - `test_glu_fused_precision_baseline.py` (4 tests recording PCC,
    max/mean abs, and relative RMS into the captured pytest log for
    refinement agents to reference).
  - `test_glu_fused_extended.py` (5 tests: L1 memory_config
    preservation, wide-W structural check, sigmoid saturation at
    ±20, determinism).
  - Total: 29 tests, all passing post-fix.
