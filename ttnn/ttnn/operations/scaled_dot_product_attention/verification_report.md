# Verification Report: scaled_dot_product_attention

Flash-Attention SDPA (online softmax). Verified on Wormhole, branch
`llk_helper_library`. Date: 2026-06-18.

## Code Review

### Fixed

- **Reader: mask `TensorAccessor` rebuilt every KV-block → hoisted out.**
  `scaled_dot_product_attention_reader.cpp` constructed
  `TensorAccessor(mask_args, mask_addr, get_tile_size(cb_mask_in))` *inside*
  the `for j` KV loop (and called `get_tile_size` twice). The accessor and
  tile size are loop-invariant — moved to a single construction before the
  work loop (guarded so `mask_tile_bytes = has_mask ? get_tile_size : 0`).
  Verified both mask and no-mask acceptance paths still pass (34/34) — the
  unconditional construction with empty `TensorAccessorArgs()` on the
  no-mask path is safe.

### Reviewed — correct, no change needed

- **CB push/pop balance** — traced every CB (cb_q_in, cb_k_in, cb_v_in,
  cb_mask_in, cb_max/sum_scaler, cb_m_blk, cb_m_run, cb_m_prev, cb_alpha,
  cb_l_run, cb_l_blk, cb_qk, cb_p, cb_o_blk, cb_o_run, cb_l_recip, cb_out).
  Every producer push count equals consumer wait/pop count per work item.
  The persistent accumulators (`cb_m_prev` leftover after the last commit,
  `cb_q_in` retained across the KV loop) are explicitly drained in the
  post-loop cleanup (`cb_pop_front(cb_q_in)`, `cb_wait_front`/`cb_pop_front`
  on `cb_m_prev`). Balanced.
- **Helper usage** — all compute phases route through helpers
  (`matmul_block`, `reduce<MAX/SUM,REDUCE_ROW>`, `eltwise_chain`,
  `add`/`mul`/`copy`/`unary`/`binary_sfpu`, `transform_in_place`). No
  hand-rolled FPU/SFPU sequences. `WaitAndRetainOnLastBlock` correctly
  retains Q across the KV loop; reduces use `WaitUpfrontNoPop` so the score
  block survives for the downstream sub-exp / PV matmul (exactly one popper
  each). Held vs streamed operand indexing (`OperandKind::Block`+`HeldBulk`
  vs `Scalar`+`Streaming`) is correct for `B_q > 1` — this was bug #2 the
  implementer fixed during bring-up.
- **TensorAccessor** (not deprecated `InterleavedAddrGen`), `void
  kernel_main()` syntax, and `api/dataflow/dataflow_api.h` /
  `api/compute/*` include paths — all correct in all three kernels.
- **Broadcast dims** — Col-broadcast of the per-row max/recip/alpha
  (REDUCE_ROW → Col0) across the free columns is correct in phases 5/6/7/10b;
  None-broadcast for the same-shape adds. Matches the design's broadcast
  verification table.

### Advisory deviations from `op_design.md` (sound, documented in-kernel)

- **Scale folded into scores, not Q.** The design pre-scaled Q
  (`transform_in_place<cb_q_in>`); the implementation folds scale into the
  locally-produced scores (`transform_in_place<cb_qk>`). Rationale (kernel
  header + `test_scaled_dot_product_attention_debug.py`): an in-place SFPU
  transform on the reader-fed `cb_q_in` nets zero tiles visible to the QK
  matmul's unpacker → UNPACK hang. `(Q·scale)·Kᵀ == scale·(Q·Kᵀ)`, so the
  result is identical and the in-place transform on a locally-framed CB is
  the legal pattern. Correct.
- **Running max via `reduce<MAX>` + `binary_sfpu<BinaryMax>` instead of
  `reduce<MAX>+Accumulate`.** The Accumulate path *pops* its accumulator CB
  on reload, which would destroy `m_{j-1}` before α (phase 4) can read it.
  `op_design.md`'s gotchas document this exact separate-op alternative.
  Correct.
- **Boot init is `mm_block_init()` only** (not
  `compute_kernel_hw_startup()` + `mm_block_init()`). Kernel comment: the
  former is a superset (it issues the same `llk_*_hw_configure` +
  pack/dest init) plus matmul init; calling both double-issues
  `pack_sync_init`. Acceptance + golden pass, so the single init is
  sufficient.

### Recommendations (perf — not refinements, no failing cell to point at)

- **Matmul sub-block sizing is hard-coded to 1×1** (`MatmulBlockShape::of(
  B_q, B_kv, 1, 1, DHt, 1)` and the PV equivalent). With `B_q,B_kv ≤ 4` the
  DEST limit (8 tiles, bf16 half-sync) allows larger sub-blocks (e.g. 2×2),
  which would cut the number of matmul passes. Correct as-is, just not
  packing DEST to its limit. Host-side `determine_largest_subblock_size`
  (per the design) would derive the optimal sub-block — deferred as a perf
  follow-up, not a SUPPORTED-advancing change.
- **Writer barriers per output tile.** `scaled_dot_product_attention_writer.cpp`
  does `cb_wait_front(cb_out,1)` → write → `noc_async_write_barrier()` →
  `cb_pop_front(cb_out,1)` per tile because compute streams `cb_out` one tile
  at a time. Correct, but a per-tile barrier serializes the NoC writes. A
  batched drain (wait `o_tiles`, write all, one barrier) would pipeline the
  block. Perf-only; left as-is.

## Registry Conformance

- **Confirmed all four declarations present and correctly wired** in
  `scaled_dot_product_attention.py`:
  - `INPUT_TAGGERS` — `tag_alignment`, `tag_attention_kind`,
    `tag_kv_heads`, each with the `(inputs, axes)` signature.
  - `SUPPORTED` — covers every axis the kernel gates on (dtype, layout,
    alignment, attention_kind, kv_heads_mode, mask_mode, scale_mode).
  - `EXCLUSIONS = []` — the generic additive-mask path handles
    causal+cross with no `S_q==S_kv` assumption, so nothing to exclude.
  - `validate()` — shape-contract `ValueError`s first (rank, head_dim Q vs
    K, K==V, batch, `H_q % H_kv`, mask dims), *then* SUPPORTED per-axis,
    *then* EXCLUSIONS. Order correct. Public entry point calls
    `validate()` as its first line.
- **Confirmed the op file does NOT declare `INVALID`** (it's a test-suite
  concept). Correct.
- **No auto-fixes to SUPPORTED were needed** — `xpass_drift = 0` and
  `xfail_wrong_mode = 0`, so SUPPORTED is honest about Phase-0 behavior.

### INVALID audit (`eval/golden_tests/.../feature_spec.py`)

- `INVALID = []`. Well-formed: SDPA is TILE-only by design
  (`TARGET["layout"] = [ttnn.TILE_LAYOUT]`, no ROW_MAJOR in the universe),
  so the canonical bf8b+ROW_MAJOR rule is **vacuous** — no ROW_MAJOR cell
  exists in the cartesian to forbid. There are no weight axes (no
  norm-style no-weight canonicalization needed) and no cross-tensor-axis
  coupling. Passes all three sanity rules. No change requested.

## Precision Baseline

`tests/.../test_scaled_dot_product_attention_precision_baseline.py`, bf16,
well-conditioned `randn` inputs (auto scale, no mask), fp32 reference:

| Shape (B,H,S,D) | PCC | Max Abs Err | Mean Abs Err | Relative RMS Err |
|-----------------|--------|-------------|--------------|------------------|
| (1,1,32,64)     | 0.99996 | 0.01199 | 0.00172 | 0.00894 |
| (1,4,128,64)    | 0.99994 | 0.00999 | 0.00111 | 0.01097 |
| (1,8,256,64)    | 0.99993 | 0.01355 | 0.00086 | 0.01173 |
| (2,4,512,64)    | 0.99992 | 0.01180 | 0.00066 | 0.01218 |

**Assessment**: Excellent for bf16. PCC ≥ 0.9999 across the range; relative
RMS ~1% and slowly rising with sequence length (more KV-blocks → more
online-softmax accumulation steps in bf16). Max abs error ~0.01 is at the
bf16 quantization floor.

**Recommended tolerances**: PCC ≥ 0.995, rel-RMS ≤ 0.05 (the golden-suite
bf16 band) — comfortable headroom on well-conditioned inputs.

### Adversarial-distribution note (regression suite)

`test_regression.py` (numerics-tagged, **not** registry-gated) has 15
failures on `test_negative_input` / `test_uniform_input` /
`test_large_magnitude_input`. These are **not algorithmic bugs**:

| Distribution | Max Abs Err | Relative RMS | PCC | Why it "fails" |
|---|---|---|---|---|
| negative (−[0.5,1.5]) | 0.023–0.027 | 0.10–0.46 | 0.905–0.995 | all-same-sign V → softmax-averaged output is **near-constant** (std ≈ 0); rel-RMS = abs_rms/std blows up and PCC is ill-conditioned on a near-zero-variance signal |
| uniform [0,1] | 0.008–0.016 | 0.05–0.24 | 0.975–0.999 | same near-constant-output effect |
| large (×10) | 1.7–47 | 0.01–0.11 | 0.994–0.9999 | PCC fine; only the largest (B1H12S512) rel-RMS 0.107 > 0.05 — bf16 accumulation over 16 KV-blocks |

The **absolute** error is at the bf16 floor (≤ 0.03 for negative/uniform);
the metric is misleading because the reference output has near-zero variance
when all V rows share a sign. `small_magnitude` (×0.01) and the acceptance
`randn` cases pass at PCC 0.99999. The real lever for the genuine-precision
ones (large ×10, and the one registry `supported_fail`) is higher-precision
accumulation — addressed by **Refinement 1** (`compute_kernel_config` /
fp32_dest_acc / fp32 dtype). Golden-suite files are upstream-authored and
were not modified.

## Verifier CLI Summary

From `eval/results/scaled_dot_product_attention/verifier_report.json`
(golden suite: 784 collected; 744 registry cartesian + 40 regression):

- supported_pass: **207**
- xfail_expected: **536**  (validate() correctly rejected every out-of-SUPPORTED cell)
- invalid_skipped: 0  (INVALID = [])
- **supported_fail: 1**  — `Q1x1x128x1024` bf16 explicit-scale, category
  `numerical-precision` (rel-RMS 0.0505 vs 0.05 target, PCC 0.9987). Per the
  registry-model policy, a `numerical-precision` supported_fail **stays
  failing and becomes a refinement entry** (Refinement 1) rather than being
  silenced via EXCLUSIONS. This is the only registry red cell.
- **xpass_drift: 0**  ✓ (SUPPORTED does not under-claim)
- **xfail_wrong_mode: 0**  ✓ (every rejection is the typed support-refusal)
- no_axes_found: 40  — the `test_regression.py` numerics cases (not part of
  the registry cartesian); see the adversarial-distribution note above.

`xpass_drift` and `xfail_wrong_mode` are both 0 — the SUPPORTED block is
honest. The single `supported_fail` is a precision near-miss queued as a
refinement, not a kernel bug.

## Refinement Gap Analysis

`TARGET − SUPPORTED` (per `feature_spec.py`):

| Axis | TARGET | SUPPORTED | Gap |
|---|---|---|---|
| dtype | float32, bfloat16, bfloat8_b | bfloat16 | **float32, bfloat8_b** |
| layout | TILE | TILE | — |
| alignment | tile_aligned, w_non_aligned, h_non_aligned | tile_aligned | **w_non_aligned, h_non_aligned** |
| attention_kind | self, cross | self, cross | — |
| kv_heads_mode | mha, gqa, mqa | mha, gqa, mqa | — |
| mask_mode | none, causal | none, causal | — |
| scale_mode | auto, explicit | auto, explicit | — |

The 536 `xfail_expected` cells decompose exactly onto these two gaps:
`('dtype',)` 416 + `('alignment',)` 40 + `('alignment','dtype')` 80. Every
`(axis, missing_value)` pair is covered by a refinement in
`op_requirements.md` (R1 = dtype, R2 = alignment; the 80 intersection cells
clear when both land). No queue gaps.

**Multi-core is already done** — the program descriptor uses
`split_work_to_cores` and stamps per-core runtime args, so the
embarrassingly-parallel `/interleaved-parallel` work is complete. **L1 is
already bounded** — the flash design sizes score CBs to `B_q·B_kv` and the
host applies an L1-aware block cap (`B·DHt ≤ 16`), so S=4096/8192 and
D=1024 golden cells pass with no OOM. No memory-budget refinement is needed.

## Recommendations

1. **Refinement 1 (numeric configurability) lands first.** It expands dtype
   to {fp32, bf8b} and exposes `compute_kernel_config`, which is the only
   in-scope lever for the one `supported_fail` (the 128×1024 bf16
   explicit-scale precision near-miss) and the genuine-precision regression
   cases. Watch the implementer constraint flagged in bring-up: enabling
   `fp32_dest_acc_en` collides with the hard-coded HiFi4 in the matmul-path
   SUM reduce (issue #38306) — the skill's config plumbing must thread this
   carefully or that one cell lands in EXCLUSIONS.
2. **Refinement 2 (non-tile-alignment) is the structurally harder one** and
   should land second. The softmax denominator must exclude padded KV
   columns (mask the partial last `S_kv` tile to −∞ before the row-sum), and
   the QK contraction must zero-pad the partial last `D` tile. Make the
   non-aligned reader/compute path dtype-aware so the 80 intersection cells
   (fp32/bf8b × non-aligned) also clear — though bf8b × non-aligned may need
   EXCLUSIONS per the standard bf8b-block-format limitation.
