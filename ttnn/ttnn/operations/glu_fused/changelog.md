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

## Refinement 2 — bfloat16 support

- **Date**: 2026-05-12
- **What was done**: Relaxed the dtype validator in `glu_fused.py` to accept
  both `float32` and `bfloat16`. Made the compute config in
  `glu_fused_program_descriptor.py` dtype-aware:
  - fp32 path retains the Phase 0 lock-in: `HiFi4 + fp32_dest_acc_en=True +
    UnpackToDestFp32` on input CBs.
  - bf16 path uses `LoFi + fp32_dest_acc_en=False`, default unpack mode.
    Justification: with a bf16 input, `UnpackToDestFp32` just zero-extends
    the mantissa (no precision gain) and `fp32_dest_acc_en=True` halves
    DEST capacity from 8 to 4 tiles in half-sync — measured ~1.3× slower
    than the bf16-tuned config, with `max_abs_diff = 0.0` either way.
  - Kernel itself is unchanged: the SFPU chain (`Load A`, `Load B`,
    `Sigmoid<Exact>`, `SfpuMul`) is format-agnostic; the CB unpack/pack
    reconfig handles the dtype switch.
- **Accuracy achieved** (bf16, vs `torch.nn.functional.glu`):
  - PCC ≥ 0.999 on the 4 baseline shapes.
  - `max_abs_diff` against Makora's bf16 `glu` kernel: 0.00e+00
    (bit-identical output).
- **Performance**: at bf16, head-to-head vs Makora `glu`:
  - GMEAN speedup ratio 1.04× (Makora marginally faster, within noise).
  - vs Phase 0 fp32 config running on bf16 inputs (the wrong setting):
    ~1.3× faster — the dtype-aware config closed the gap from 1.33× to
    1.04× without changing any kernel code.
- **Issues encountered**: None. The kernel was already bf16-ready
  (CBs use `input_tensor.dtype`); only the validator and compute config
  needed updates.
- **Tests added**:
  - `test_glu_fused_bf16.py` (5 tests: 4 correctness shapes at bf16 +
    1 validation case confirming non-{fp32, bf16} dtypes still rejected).
    All 5 pass.
