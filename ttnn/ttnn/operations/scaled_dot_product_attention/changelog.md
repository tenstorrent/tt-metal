# Changelog: scaled_dot_product_attention

## Phase 0 — Core Implementation
- **Date**: 2026-06-12
- **What was done**: Initial Flash Attention implementation via the incremental
  pipeline (planner → implementer → verifier). Online-softmax recurrence
  (running max / sum / output), O(S) memory — the full S_q × S_kv score matrix
  is never materialized. Multi-core via `split_work_to_cores` over
  `(b, h, q_block)` work units. `generic_op` + `ProgramDescriptor`; reader
  (NCRISC) / compute (TRISC) / writer (BRISC); helper-based compute
  (`matmul_block`, `reduce`, `eltwise_chain`, `binary_sfpu` BinaryMax).
- **SUPPORTED at Phase 0**: dtype=[bfloat16], layout=[TILE], alignment=[tile_aligned],
  attention_kind=[self, cross], kv_heads_mode=[mha], mask_mode=[none, custom],
  scale_mode=[auto, explicit]. EXCLUSIONS=[]. INVALID lives in feature_spec.py
  (`[]` — SDPA is TILE-only, so the bf8b+ROW_MAJOR rule is vacuous).
- **Accuracy achieved**: PCC ≈ 0.9999, max_abs_err ≈ 0.009–0.024,
  mean_abs_err ≈ 0.0006–0.0035, relative RMS ≈ 0.012–0.016 (measured on 4 shapes
  via `test_scaled_dot_product_attention_precision_baseline.py`; bf16,
  tile-aligned, random-normal inputs).
- **Golden suite at Phase 0**: 140 / 1156 cells passing; 976 xfail_expected;
  loud categories all 0 (supported_fail / xpass_drift / xfail_wrong_mode), per
  `verifier_report.json`.
- **Issues encountered (fixed during verification)**:
  - `__init__.py` did not re-export `INPUT_TAGGERS` / `SUPPORTED` / `EXCLUSIONS`,
    causing a golden-suite collection error. Added the re-exports.
  - Running accumulators (`m_i`, `l_i`, `O_i`) and their scratch were stored in
    bf16 CBs, violating the design's fp32-accumulation requirement and
    compounding rounding across KV blocks (error grew with S; S=8192 cells
    failed numerical-precision). Switched the accumulator/scratch CBs to
    `float32` and fixed the `cb()` page-size helper to size by data format.
    Result: `supported_pass` 138 → 140 (S=8192 now passes); regression-canary
    PCC improved (e.g. `test_negative_input[B1_H12_S512]` 0.866 → 0.936).
  - Reader rebuilt the mask `TensorAccessor` every KV iteration; hoisted it to
    construct-once alongside the Q/K/V accessors.
- **Tests added**: `test_scaled_dot_product_attention_precision_baseline.py`
  (PCC + abs/RMS across 4 shapes). The acceptance suite
  (`test_scaled_dot_product_attention.py`, 24 cases) and the golden suite were
  already present; both pass.

## Refinement 1 — Numerical configurability (dtype + compute_kernel_config)  [~] partial
- **Date**: 2026-06-12
- **What was done**:
  - `SUPPORTED["dtype"]` += `ttnn.float32`, `ttnn.bfloat8_b` (was `[bfloat16]`).
  - Public entry point exposes optional `compute_kernel_config` (a
    `ttnn.*ComputeKernelConfig`): `math_fidelity`, `fp32_dest_acc_en`,
    `math_approx_mode`, `dst_full_sync_en` are now caller-controlled.
  - Program descriptor derives ALL CB data formats from dtype + config
    (subsuming the Phase-0 hard-coded f32 accumulator formats): input/output CBs
    follow the tensor dtype; intermediate/accumulator CBs (running m_i/l_i/O_i,
    their scratch, AND the score/prob blocks cb_qk/cb_p — the verifier's
    score-path precision lever) are fp32 when `fp32_dest_acc_en`, else input
    dtype. Reduce scalers stay bf16 (packed-scaler format).
  - **Default `math_fidelity` is dtype-aware**: bf16/bf8b keep Phase-0 HiFi2
    (bf16 path byte-identical to Phase 0); fp32 defaults to HiFi4. fp32 matmul
    operands truncate to TF32 in srcA/srcB, so HiFi2 leaves fp32 at ~bf16
    precision; HiFi4's multi-pass matmul recovers the mantissa to meet fp32's
    tight golden RMS target. This is a new default for a newly-added dtype, not
    a change to prior bf16 behavior.
  - **No `UnpackToDestFp32` tags**: every intermediate/accumulator CB feeds an
    FPU op (matmul/reduce/FPU-binary), which is incompatible with the tag. The
    fp32 storage already gives the precision win; FPU inputs land in TF32
    regardless (unavoidable srcA/srcB drop).
  - Defensive: bf8b intermediate CBs are floored to bf16 when
    `fp32_dest_acc_en=False` (block-float is unusable for the running stats).
- **Accuracy achieved** (default config, golden randn inputs):
  - bf16: PCC ≥ 0.995, rms ≤ 0.05 — all golden cells pass.
  - bf8b: PCC ≥ 0.97 (rand) / 0.9998 (randn) on tile-aligned, rms ≤ 0.027 — all
    golden cells pass.
  - fp32: PCC ≈ 0.9996–1.0; rms ≤ 0.012 for S ≤ 2048 (HiFi4). 134/140 fp32
    golden cells pass.
  - Precision matrix (8 shapes × 3 dtypes × 4 fidelities × fp32_acc × {uniform,
    normal}): 320 passed, 64 skipped (bf8b+lp_acc). Full table in
    `precision_matrix_results.md`.
- **Golden test progress**: **414 / 1156 passing** (was 140 at Phase 0), 696
  xfailed (was 976), 1 skipped, **6 failed** — all fp32 corners (see below).
  Regression canaries: 14 fail (down from Phase-0 19) — 4 are GQA/MQA correctly
  rejected by validate() (Refinement 2 scope); 10 are uniform/negative
  distribution canaries that are relative-RMS metric artifacts (max_abs tiny,
  0.01–0.03; near-uniform softmax → near-constant output → rms denominator
  collapses). The score-path fp32 lever (cb_qk/cb_p) was applied but does not
  move these because the bottleneck is the metric, not precision.
- **Issues encountered / deferred (left failing per protocol, NOT excluded)**:
  - **fp32 D=1024 (`Q1x1x128x1024`, 4 cells): L1 OOM.** Explicit
    `program.cpp` throw — fp32 CBs (4 B/elem, footprint scales with D_t and
    double-buffering) exceed the 1.5 MB L1 budget at D_t=32. Orthogonal to
    numerics; bf16/bf8b D=1024 fit. → **Refinement 5** (memory-budget: chunk D
    or single-buffer large-D fp32 CBs).
  - **fp32 S=8192 (`Q1x1x8192x64`, 2 cells): precision near-miss** (PCC 0.9996,
    rms 0.0284 vs 0.02; max_abs 0.0034). Root cause confirmed by the
    ttnn-expert-debugger via host-side simulation: it is **SFPU `exp` precision
    accumulated across the 256 online-softmax KV blocks** (rms grows
    ~√num_blocks: 32blk 0.0059 → 256blk 0.0284), NOT TF32 matmul (full-fp32
    matmul sim = 0.0000; TF32 sim = 0.0004, flat in S). Kernel already runs at
    the max precision the descriptor exposes (HiFi4 + fp32_dest_acc + accurate
    exp); no descriptor/helper-level lever closes it. → **Refinement 6**
    (algorithmic: two-pass softmax for the fp32 long-context path, or wider
    Q-block to amortize fewer recurrence steps — net-new, outside R1's
    descriptor-level scope). This is at/below the hardware floor of the
    online-softmax + Wormhole-SFPU stack for the 256-block cell.
  - The canonical `bf8b × non-aligned` EXCLUSION is NOT added yet — the
    `alignment` axis is still `tile_aligned`-only; it arms with Refinement 4.
- **Tests added**:
  - `test_scaled_dot_product_attention_precision_matrix.py` (the authoritative
    precision-characterization matrix) + `precision_matrix_results.md`.
  - `test_scaled_dot_product_attention_debug.py` (CPU-only flash-recurrence
    precision simulation, authored by the expert-debugger — pinpoints exp as
    the S=8192 fp32 error source; preserved as a regression/analysis artifact).
  - Probes `probe_001`–`probe_004` (dtype sweep, HiFi sweep, S-sweep).

## Refinement 2 — GQA / MQA (KV head broadcast)  [x] complete
- **Date**: 2026-06-12
- **What was done**:
  - `SUPPORTED["kv_heads_mode"]` += `"gqa"`, `"mqa"` (was `["mha"]`).
  - Reader (`scaled_dot_product_attention_reader.cpp`): single index remap —
    each Q head `h` now reads K/V from head `h_kv = h / kv_group`, where
    `kv_group = H_q / H_kv`. New CT arg (index 7) carries `H_kv`; `kv_base`
    strides over `H_kv` heads instead of `H_q`. MHA is unchanged
    (`H_kv == H_q` ⇒ `kv_group == 1` ⇒ identity remap, byte-identical to
    Phase 0/R1). Output, Q, and mask indexing stay on `H_q` (mask broadcasts
    over Q heads exactly as before). Accessor CT offsets bumped 7 → 8.
  - Program descriptor: passes `h_kv = int(key.shape[1])` as reader CT arg 7.
  - `validate()`: now rejects `H_q` not a multiple of `H_kv` with `ValueError`
    (undefined head broadcast — matches the reference's `repeat_interleave`
    contract). No new compute, no new CB.
- **Accuracy achieved** (default config, randn inputs, GQA/MQA ratios 8:2,
  32:8, 8:1, 32:1, 8:2 long, 2:1 multibatch):
  - bf16: PCC ≥ 0.995. fp32: PCC ≥ 0.999. bf8b: PCC ≥ 0.99. All new
    `test_scaled_dot_product_attention_gqa_mqa.py` cases pass (rtol/atol via
    PCC gate per dtype).
- **Golden test progress**: **647 / 1156 passing** (was 414 at Refinement 1),
  492 xfailed, 1 skipped, 16 failed. The +233 are exactly the registry cells
  carrying `kv_heads_mode ∈ {gqa, mqa}` (now all green) plus the 4
  `test_gqa_mqa_forward` regression canaries (were failing `other` =
  validate() rejection at R1, now pass). **Zero GQA/MQA failures.** The 16
  remaining failures are all pre-existing, out-of-scope R1 deferrals:
  4× `Q1x1x128x1024` fp32 L1 OOM (→ Refinement 5), 2× `Q1x1x8192x64` fp32
  SFPU-exp precision floor (→ Refinement 6), 10× uniform/negative regression
  canaries (relative-RMS metric artifacts on near-uniform softmax, documented
  in R1). None touch the GQA/MQA path.
- **Issues encountered**: None. The head remap was a clean reader-side change
  as the verifier predicted; the existing `tag_kv_heads` tagger and
  `validate()` routing required no change beyond the divisibility guard.
- **Tests added**: `tests/ttnn/unit_tests/operations/scaled_dot_product_attention/test_scaled_dot_product_attention_gqa_mqa.py`
  — 20 cases: GQA/MQA over 6 head-ratio shapes × {bf16, fp32, bf8b},
  MQA composed with a custom additive mask, and a non-divisible-heads
  rejection check.

## Refinement 3 — Causal masking (on-device triangular bias)  [x] complete
- **Date**: 2026-06-12
- **What was done**:
  - `SUPPORTED["mask_mode"]` += `"causal"` (was `["none", "custom"]`).
  - `EXCLUSIONS` += `{"mask_mode": "causal", "attention_kind": "cross"}` —
    causal requires a square score matrix (S_q == S_kv, decoder self-attn);
    the on-device generation path assumes it. Rectangular causal is refused
    (raises `ExcludedCell`/`NotImplementedError` → observed as xfail_expected).
  - **On-device triangular bias, generated ONCE.** Because the kernel runs
    `q_chunk_t == k_chunk_t == 1` and causal forces S_q == S_kv, the
    diagonal-straddling KV block is always `j == qi` and its per-element mask
    (element (r,c) = 0 if c <= r else -inf) is the SAME constant 32x32 tile for
    every work unit. The reader generates it once at kernel start into a
    dedicated `cb_causal_mask` (1 bf16 page, allocated only when `is_causal`)
    via a manual L1 fill that honors the 32x32 4-face (16x16) tile layout —
    so the bytes are byte-identical to a TILE-layout DRAM mask read. bf16 is
    exact for {0, -inf}; -inf bit pattern 0xFF80.
  - **Three-region control flow.** Both reader and compute cap the KV loop at
    `j <= qi`: future blocks (`j > qi`) are never read or computed (the
    ~half-KV-work causal speedup, and keeps cb_k_in/cb_v_in push==wait). Past
    blocks (`j < qi`) get no mask (fully attended). The diagonal block
    (`j == qi`) adds `cb_causal_mask` (HeldStream) to the score block at the
    Phase E slot, then the unchanged online-softmax recurrence runs.
  - **New args**: reader CT8 `is_causal` (accessor CT offsets 8→9); compute
    CT3 `is_causal` + CT4 `S_q_t` + RT1 `start_unit` so compute decodes
    `qi = (start_unit + u) % S_q_t`. Program descriptor gained an `is_causal`
    param threaded from the entry point.
  - The masking math is structurally identical to the already-validated
    custom-mask path (the golden "custom" cell uses the same triangular -inf
    additive bias), so correctness was pre-proven; the new work is the
    on-device generation + block-skipping control flow.
- **Accuracy achieved** (randn inputs, default config):
  - bf16: PCC >= 0.995. fp32: PCC >= 0.999. bf8b: PCC >= 0.99. All shapes in
    `test_scaled_dot_product_attention_causal.py` (single-tile, multi-tile,
    multi-head, multi-batch, GQA 4:1, MQA 8:1, Llama3 GQA) pass.
  - Deterministic diagonal-block check (`test_causal_first_row_...`): with a
    single Q tile-row, output row 0 attends only to key 0 — verified exact.
- **Golden test progress**: **880 / 1156 passing** (was 647 at Refinement 2),
  228 xfailed, 1 skipped, **8 failed**. **+233 newly passing** = the
  `mask_mode=causal, attention_kind=self, tile_aligned` registry cells across
  bf16/fp32/bf8b and mha/gqa/mqa. **Zero causal-specific failures; zero XPASS
  drift.** The 8 failures are all pre-existing, out-of-scope deferrals:
  - 6× `Q1x1x128x1024` fp32 → L1 OOM (`program.cpp:1450`). Fails IDENTICALLY
    for `mask_mode ∈ {none, custom, causal}` at this shape, confirming it is
    the fp32 D=1024 large-head-dim OOM, **orthogonal to causal** → Refinement 5.
    (Causal adds only one bf16 mask tile; the OOM is the D_t=32 fp32 CBs.)
  - 2× `Q1x1x8192x64` fp32 (mask_mode=none) → SFPU-exp precision floor
    (rms 0.0284 vs 0.02) → Refinement 6. Note: causal+S=8192+fp32 is NOT among
    the failures — causal processes ~half the KV blocks, so the
    √num_blocks exp-accumulation error stays below the 0.02 target.
- **Issues encountered**: None. Passed on the first kernel implementation
  (35/35 new causal tests under both --dev and non-dev; no regression on the
  44 acceptance + GQA/MQA cases).
- **Tests added**: `tests/ttnn/unit_tests/operations/scaled_dot_product_attention/test_scaled_dot_product_attention_causal.py`
  — 35 cases: causal self-attn over 10 shapes × {bf16, fp32, bf8b}, explicit
  vs auto scale, a deterministic diagonal-block correctness check, the
  causal+cross EXCLUSION (NotImplementedError), and is_causal+attn_mask
  mutual-exclusion (ValueError).

## Refinement 4 — Non-tile-aligned sequence / head dim  [x] complete
- **Date**: 2026-06-12
- **What was done**:
  - `SUPPORTED["alignment"]` += `"w_non_aligned"`, `"h_non_aligned"` (was
    `["tile_aligned"]`).
  - `EXCLUSIONS` += `{dtype: bfloat8_b, alignment: w_non_aligned}` and
    `{dtype: bfloat8_b, alignment: h_non_aligned}` — the canonical bf8b ×
    non-aligned EXCLUSION flagged in Refinement 1. Block-float shares one
    exponent per 16x16 face; from_torch's tilization computes that exponent
    over the (non-zero) garbage padding, so the live mantissa is crushed and
    the reader's post-read zeroing cannot repair an already-packed tile.
  - **Root cause (probe-confirmed).** from_torch's TILE padding of a partial
    last tile is NOT zero (probe: bf16 non-aligned out-of-box PCC 0.03–0.75,
    one nan). Two independent edges, both at the data-access boundary (math
    stays on full tiles):
    1. **D % 32 (w_non_aligned).** QKᵀ contracts over D, so Q/K's padded D
       columns must be ZEROED (0·x = 0) or garbage pollutes the dot product.
    2. **S_kv % 32.** The partial last KV tile's padded key columns would
       otherwise enter the online-softmax (score ≈ 0, not −inf) and inflate the
       running max / sum.
    S_q % 32 (partial query rows) needs nothing — rows are independent in every
    tile op and padded output rows are dropped on readback.
  - **Reader** (`..._reader.cpp`): new `zero_tile_tail()` (honors the 4-face
    16x16 tile layout, bf16/fp32 via `elem_bytes`) zeros (a) Q/K's padded D
    columns of the last D tile (`cols >= D%32`) and (b) K/V's padded S_kv rows
    of the last KV block (`rows >= S_kv%32`, overflow/nan safety so a masked
    P·V term is 0·0). Also generates a constant `cb_kv_pad_mask` once: a
    {0,−inf} bf16 column tile (`c < S_kv%32 ? 0 : −inf`).
  - **Compute** (`..._compute.cpp`): adds `cb_kv_pad_mask` (HeldStream) to the
    score block on the last KV block (`j == num_kv_blocks-1`) when S_kv
    non-aligned — composes additively with the custom / causal masks. The
    padded keys become −inf → softmax weight 0 → drop out of max/sum and P·V.
  - **Descriptor**: ceil-div tile counts (`D_t`, `S_q_t`, `S_kv_t`); new CT
    args `d_valid` / `kv_valid` / `elem_bytes` (reader) and `kv_valid`
    (compute); allocates `cb_kv_pad_mask` (CB index 6, 1 bf16 page) when
    `S_kv % 32 != 0`. The tile-aligned path is unchanged: `d_valid == kv_valid
    == 0` makes every new code path an `if constexpr (0 != 0)` (compiled out),
    and ceil-div == floor-div for aligned dims — bf16/fp32/bf8b aligned cells
    are byte-identical to Refinement 3.
- **Accuracy achieved** (randn inputs, default config):
  - bf16: PCC ≥ 0.9999, max_abs ≤ 0.052 across all 10 non-aligned shapes
    (w/h/both, self/cross, mha/gqa/mqa, multi-head, multi-batch).
  - fp32: PCC ≥ 0.999 on all non-aligned golden cells.
  - No NaN in any output (padding fully neutralized).
- **Golden test progress**: **996 / 1156 passing** (was 880 at Refinement 3),
  112 xfailed, 1 skipped, **8 failed**. **+116 newly passing** = exactly the
  `alignment ∈ {w_non_aligned, h_non_aligned}` registry cells across bf16/fp32
  × {none, custom, causal} × {mha, gqa, mqa, cross}. **Zero non-aligned
  failures; zero XPASS drift.** The 8 failures are ALL pre-existing,
  out-of-scope fp32 `tile_aligned` deferrals, unchanged by this refinement:
  - 6× `Q1x1x128x1024` fp32 → L1 OOM (`program.cpp`) → Refinement 5.
  - 2× `Q1x1x8192x64` fp32 → SFPU-exp precision floor (rms 0.0284 vs 0.02) →
    Refinement 6.
  bf8b × non-aligned (24 cells) and causal × cross non-aligned are correctly
  xfailed via EXCLUSIONS.
- **Issues encountered**: None blocking. The two-edge masking (D-zero +
  kv-pad −inf) passed on the first kernel implementation; both --dev and
  non-dev pass (no race conditions).
- **Tests added**: `tests/ttnn/unit_tests/operations/scaled_dot_product_attention/test_scaled_dot_product_attention_non_aligned.py`
  — 70 cases: non-aligned (w/h/both, self/cross, gqa/mqa, multi-head/batch) ×
  {bf16, fp32} × {none, custom mask, causal} × scale modes, with NaN-free +
  PCC + shape assertions, plus the bf8b × non-aligned EXCLUSION
  (NotImplementedError). Probes 005–008 (from_torch padding inspection, before/
  after the fix on the non-aligned shape set).
