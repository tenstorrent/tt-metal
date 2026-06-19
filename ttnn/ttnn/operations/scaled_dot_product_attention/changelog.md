# Changelog: scaled_dot_product_attention

## Phase 0 — Core Implementation
- **Date**: 2026-06-18
- **What was done**: Initial Flash-Attention (online-softmax) implementation
  via the incremental pipeline (planner → implementer → verifier). Fused
  reader/compute/writer kernels built on the helper library
  (`matmul_block`, `reduce<MAX/SUM,REDUCE_ROW>`, `eltwise_chain`,
  `add`/`mul`/`copy`/`unary`/`binary_sfpu`, `transform_in_place`). Score CBs
  sized to one `B_q × B_kv` block (the load-bearing Flash constraint — the
  full `S_q × S_kv` matrix is never materialized). Multi-core from the start
  via `split_work_to_cores` (one `(b, h_q, qb)` work item per stamp,
  interleaved DRAM, no inter-core communication). Host applies an L1-aware
  block cap (`B·DHt ≤ 16`) so large `D` and long sequences fit L1.
- **SUPPORTED at Phase 0**: dtype=[bfloat16], layout=[TILE],
  alignment=[tile_aligned], attention_kind=[self, cross],
  kv_heads_mode=[mha, gqa, mqa], mask_mode=[none, causal],
  scale_mode=[auto, explicit]. EXCLUSIONS=[]. (INVALID lives in
  feature_spec.py and is []; not declared in the op file.)
- **Accuracy achieved** (bf16, well-conditioned randn, fp32 reference,
  `test_scaled_dot_product_attention_precision_baseline.py`):
  PCC=0.99992–0.99996, max_abs_err≈0.012, mean_abs_err≈0.001,
  relative_rms_err≈0.009–0.012 across 4 shapes
  [(1,1,32,64), (1,4,128,64), (1,8,256,64), (2,4,512,64)].
- **Golden suite at Phase 0**: 207 / 208 supported cells passing (per
  `verifier_report.json`); 536 xfail_expected; xpass_drift=0;
  xfail_wrong_mode=0. The 1 red cell is `Q1x1x128x1024` bf16 explicit-scale,
  category `numerical-precision` (rel-RMS 0.0505 vs 0.05, PCC 0.9987) —
  queued as Refinement 1, not silenced.
- **Issues encountered**:
  - **Code-review fix**: the reader rebuilt the mask `TensorAccessor` (and
    re-called `get_tile_size`) inside the per-KV-block loop — hoisted to a
    single loop-invariant construction before the work loop. Acceptance
    34/34 still passes (mask + no-mask paths).
  - **Regression suite** (`test_regression.py`, numerics-tagged, not
    registry-gated): 15 failures on `negative`/`uniform`/`large` input
    distributions. Diagnosed as **metric conditioning, not algorithmic
    bugs** — all-same-sign V yields a near-constant softmax-averaged output
    (std≈0), so relative-RMS/PCC are ill-conditioned even though absolute
    error is at the bf16 floor (≤0.03). The genuine-precision subset
    (large ×10) is addressed by Refinement 1's higher-precision config.
    Golden-suite files were not modified (upstream-authored).
  - Three Phase-0 bring-up bugs were already found+fixed by the
    implementer/expert-debugger: UNPACK hang from in-place pre-scale of the
    reader-fed `cb_q_in` (scale moved to the locally-produced scores);
    `j>0` α-correction wrong for `B_q>1` (held operands switched to
    `OperandKind::Block`+`HeldBulk`); large-head-dim (D≥512) L1 overflow
    (L1-aware block cap).
- **Tests added**: `test_scaled_dot_product_attention_precision_baseline.py`
  (PCC + abs/RMS across 4 shapes). Existing:
  `test_scaled_dot_product_attention.py` (acceptance, 32 cases),
  `test_scaled_dot_product_attention_debug.py` (single-tile no-hang
  regression, 2 cases). Full op test suite: 34 passed.

## Refinement 1 — Numerical configurability expansion
- **Date**: 2026-06-19
- **What was done**:
  - `SUPPORTED["dtype"]` += `float32`, `bfloat8_b` (was bf16-only). EXCLUSIONS
    stays `[]` — every aligned fp32/bf8b golden cell passes.
  - Exposed `compute_kernel_config` on the public entry point (optional
    `ttnn.*ComputeKernelConfig`): threads `math_fidelity`, `fp32_dest_acc_en`,
    `math_approx_mode`, `dst_full_sync_en` into the program descriptor's
    `ComputeConfigDescriptor`.
  - **Default config changed** (deliberate, R1's purpose is accuracy): Phase-0
    `HiFi2 / fp32_dest_acc OFF` → `HiFi4 / fp32_dest_acc_en=True`. fp32 dest
    accumulation parks the online-softmax running stats (m, l, O) and the QK/PV
    matmul partials in fp32; HiFi4 maximizes matmul pass count. Callers wanting
    throughput pass an explicit HiFi2/LoFi config.
  - Intermediate/scratch CBs are now dtype-aware: `Float32` when
    `fp32_dest_acc_en`, AND forced `Float32` whenever the input dtype is not
    bf16 (fp32/bf8b) — a bf16 intermediate fed from a non-bf16 input through the
    matmul→reduce→eltwise helper chain yields uncorrelated output when
    fp32_dest_acc is off (fixes fp32+fp32_acc=False: PCC 0.37 → 0.9999). bf16
    intermediates only for the bf16-input + fp32_acc-off case (formats match).
  - bf8b forces `fp32_dest_acc_en=True` regardless of the caller's flag — the
    block-float matmul has no valid bf16-dest mode (PCC ~0.06 otherwise);
    prevents a silent-garbage footgun.
  - Block cap is now a byte budget (`MAX_INPUT_BLOCK_BYTES`), dtype-aware so fp32
    inputs get half the tile budget for the same L1 footprint (bf16 unchanged).
  - **No UnpackToDestFp32 tags**: every intermediate CB feeds an FPU op (matmul /
    reduce / FPU binary), which is incompatible with the tag (skill §1.5). The
    precision lever is fp32 dest accumulation + fp32 scratch, not lossless reload.
  - **Verifier #38306 concern resolved by direct test, not avoidance.** The
    Phase-0 note flagged HiFi4 + fp32_dest_acc + bf16 matmul-path SUM reduce as
    the known-bad combo. Tested on this kernel (bf16 multi-KV, SUM reduce every
    block): PCC=1.00000, rel-RMS 0.0024 — NO corruption. The generic
    `compute_kernel_lib::reduce` threads `MATH_FIDELITY` as a template arg (does
    not hardcode HiFi4), and the SUM-reduce intermediates here are fp32, so the
    bf16-SUM-reduce failure path of #38306 is never reached. HiFi4 is therefore
    the default.
- **Accuracy achieved** (default config, normal inputs, fp32 reference):
  - bf16: PCC ≥ 0.9999, rel-RMS ≤ 0.0023, max_abs ≤ 0.0028 across matrix shapes.
  - fp32: PCC = 1.00000, rel-RMS ≤ 0.0032, max_abs ≤ 0.0045.
  - bf8b: PCC ≥ 0.99987, rel-RMS ≤ 0.016, max_abs ≤ 0.015.
  - Phase-0 registry red cell `Q1x1x128x1024` bf16 explicit-scale: rel-RMS
    0.0505 → **0.0138** (golden) / 0.0033 (matrix) — now passing.
  - Full precision matrix (384 cells, all fidelities × fp32_acc × dist):
    **384 passed, 0 skipped**. See `precision_matrix_results.md`.
- **Golden test progress**: bf16 **208/208** (Phase-0 was 207/208), fp32
  **208/208**, bf8b **208/208** aligned cells passing — **624 passed**, 120
  xfailed (non-aligned, Refinement-2 scope), **0 failed**, xpass_drift=0.
  Regression suite (`test_regression.py`, not registry-gated, upstream-authored,
  NOT modified): **15 → 7 failures**. The genuine-precision `large` (×10) cases
  now pass; the 7 remaining are `uniform`/`negative` near-constant-output cases
  with max_abs ≤ 0.0078 (bf16 floor) failing only on rms-normalized-by-≈0-std —
  a metric-conditioning artifact (Phase-0-documented), not an algorithmic bug,
  and not fixable by any precision config.
- **Issues encountered**:
  - fp32 / bf8b + `fp32_dest_acc_en=False` produced uncorrelated output
    (PCC 0.06–0.37, rel-RMS >1) through the bf16 intermediate. Root-caused via
    probes: a non-bf16 input cannot round-trip through a bf16 intermediate in the
    helper chain. Fixed for fp32 (fp32 intermediates); bf8b additionally needs
    fp32 dest accumulation (forced on). Characterized in the precision matrix.
  - The verifier-flagged #38306 combo was empirically disproven (above) rather
    than worked around.
- **Tests added**:
  `test_scaled_dot_product_attention_precision_matrix.py` (the authoritative
  precision characterization: 8 shapes × 3 dtypes × 4 fidelities × 2 fp32_acc ×
  2 distributions = 384 cells), and `precision_matrix_results.md`. All prior op
  tests still green (acceptance 32, debug 2, baseline 4 = 38 passed).

## Refinement 2 — Non-tile-aligned shape support
- **Date**: 2026-06-19
- **What was done**:
  - `SUPPORTED["alignment"]` += `w_non_aligned` (D not %32), `h_non_aligned`
    (S_q not %32, D aligned). EXCLUSIONS stays `[]` — every non-aligned cell
    passes, including bf8b × non-aligned (the refinement had pre-authorized an
    EXCLUSION for it; not needed).
  - **Program-descriptor tile-count fix (load-bearing bug)**: `DHt`, `Sq_t`,
    `Sk_t` were computed with FLOOR division, silently dropping the partial last
    tile and processing only the tile-aligned prefix. Switched to CEIL division
    (`_ceil_div`). For tile-aligned dims ceil == floor, so this is a no-op on the
    aligned path. Added `kv_partial = S_kv % 32` as the masking trigger.
  - **KV-column softmax-denominator masking** (the `h_non_aligned` core change):
    K's padded rows are zero, so the padded KV columns of the last score
    tile-column score 0; without masking, the row-max can latch onto 0 and the
    row-sum gains `exp(0 - m)` per padded column — both corrupt the softmax.
    The reader now generates a persistent additive `-inf` pad-mask block (one
    `B_q × B_kv` block, acc_dtype, face-major element fill, dtype-aware fp32/bf16)
    ONCE at startup and never pops it. The compute (Phase 2c) adds it held
    (Block-indexed, mirroring the proven Phase-5 operand structure) on the LAST
    kv block, before the row-max. The whole path is gated on
    `if constexpr (kv_partial != 0)`, so tile-aligned inputs compile it out
    entirely (the pad-mask CB is not even allocated).
  - **`w_non_aligned` needs nothing beyond the ceil fix**: from_torch zero-pads
    the TILE inputs, so the QK contraction and PV matmul over the partial D-tile's
    padded lanes contribute `0·x = 0`. The masking is keyed on `S_kv % 32 != 0`,
    independent of the alignment tag, so `both`-non-aligned and cross-attention
    with aligned-S_q-but-non-aligned-S_kv are covered too.
  - **`h_non_aligned` S_q (output) padding**: the last seq tile is partial; the
    writer writes full tiles and ttnn's logical-shape slicing drops the padding
    output rows. Padded Q rows produce finite garbage (softmax of zero-scores =
    uniform over valid columns; denominator ≥ 1), never NaN, and are discarded.
  - bf8b × non-aligned reuses R1's dtype-aware CB derivation: bf8b forces
    fp32 dest acc + fp32 intermediates, so the pad mask is fp32 and the
    block-float input's zero-padding survives the matmul cleanly.
- **Accuracy achieved** (fp32 reference; probe + golden + unit tests):
  - bf16: PCC = 1.00000, max_abs ≤ 0.008, norm-RMS ≤ 0.05 across the non-aligned
    shape sweep.
  - fp32: PCC = 1.00000, max_abs ≤ 0.008, norm-RMS ≤ 0.02.
  - bf8b: PCC ≥ 0.9999, max_abs ≤ 0.029, norm-RMS ≈ 0.015 (well under the 0.12
    bf8b tolerance).
  - Direct before/after on `h_non_aligned`: PCC 0.99926 → **1.00000**, max_abs
    **0.153 → 0.003** (the pad mask removing the polluted denominator).
- **Golden test progress**: **744 passed, 1 skipped, 0 failed, 0 xfailed**
  (Refinement 1 was 624 passed + 120 xfailed). All 120 previously-xfailed
  non-aligned cells (10 shapes × 3 dtypes × mask × scale) now pass. The full
  TARGET × INPUTS cartesian is supported; xpass_drift = 0.
- **Issues encountered**: None blocking. The only subtlety was the face-major
  tile element layout for the in-reader mask fill (a 32×32 tile is 4 row-major
  16×16 faces ordered TL/TR/BL/BR; element (r,c) lives at
  `face*256 + (r%16)*16 + (c%16)`) — verified correct by the PCC jump to 1.0.
  bf8b × non-aligned passed out of the box, so the pre-authorized EXCLUSION was
  not needed.
- **Tests added**:
  `test_scaled_dot_product_attention_non_aligned.py` — 11 non-aligned shapes
  (w / h / both / cross-attn-non-aligned-S_kv, across MHA/GQA/MQA/multi-batch)
  × bf16/fp32/bf8b × mask none/causal × scale auto/explicit (132 cells) + a
  3-dtype aligned no-regression guard (135 passed under --dev, 135 non-dev).
  All prior op tests still green (acceptance 32, debug 2, baseline 4,
  precision matrix 384 = 422 passed).
