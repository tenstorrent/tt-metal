# Operation Requirements: scaled_dot_product_attention

## Definition
- **Formula**: `output = softmax(Q @ K^T * scale + mask) @ V` (Flash Attention v2 online-softmax recurrence with O(S) memory)
- **PyTorch Reference**: `torch.nn.functional.scaled_dot_product_attention(Q, K, V, attn_mask=mask, is_causal=is_causal, scale=scale)`
- **Import Path**: `from ttnn.operations.scaled_dot_product_attention import scaled_dot_product_attention`
- **Function Signature**: `scaled_dot_product_attention(query: ttnn.Tensor, key: ttnn.Tensor, value: ttnn.Tensor, *, attn_mask: ttnn.Tensor | None = None, is_causal: bool = False, scale: float | None = None, compute_kernel_config=None, memory_config: ttnn.MemoryConfig = None) -> ttnn.Tensor`

## Phases

> **Non-regression rule**: Every refinement must pass all tests from prior phases.
> **Drift signal**: XPASS-strict failures mean the implementer added support but forgot to update SUPPORTED. The implementer fixes by updating SUPPORTED.
> **Checkbox protocol**: Implementer marks `[x]` when the refinement is complete and all tests pass, `[~]` when real work landed but at least one named axis value is deferred (treated as completed by the queue, surfaced as partial), `[ ]` only when nothing usable was produced.
> **Refinement ID + follow-up naming (mandatory — the runner parses this)**: Primary refinements are `Refinement N` (e.g. `Refinement 1`, `Refinement 2`). When you ship `[~]` partial and file the sharper follow-up the partial-tick protocol requires, name it by appending a lowercase letter to the parent's number: `Refinement 1b`, `Refinement 1c`, … (never `Refinement 1.5`, `Refinement 1 (follow-up)`, or a fresh number). Order follow-ups immediately after their parent so the queue runs them before later refinements — a partial's remaining-blocker follow-up must be picked next, not leapfrogged. The runner's parser matches exactly `Refinement \d+[a-z]?`; any other shape is invisible to the queue and silently skipped.

### [x] Phase 0 — Core Implementation

- **SUPPORTED dtype**: [bfloat16]
- **SUPPORTED layout**: [TILE]
- **SUPPORTED shape-derived axes**: alignment=tile_aligned only
- **SUPPORTED op-specific axes**: attention_kind ∈ {self, cross}, kv_heads_mode ∈ {mha, gqa, mqa}, mask_mode ∈ {none, custom}, scale_mode ∈ {auto, explicit}, fp32_dest_acc_en ∈ {True}
- **Cores**: single-core per (B,H) work unit, embarrassingly parallel via split_work_to_cores
- **Compute config**: hard-coded HiFi4 + fp32_dest_acc_en=True
- **Golden baseline**: 200 / 2767 cells passing (per verifier CLI); 8 OOM on D=512/D=1024; 2440 xfail_expected

### [x] Refinement 1 — Numerical configurability expansion

**Goal**: add `ttnn.float32`, `ttnn.bfloat8_b` to `SUPPORTED["dtype"]`, add `False` to `SUPPORTED["fp32_dest_acc_en"]`, and expose `compute_kernel_config: ttnn.ComputeKernelConfig` on the entry point (already accepted but only True is in SUPPORTED). Cells that fail out of the box (typically `bfloat8_b + fp32_dest_acc_en=False`) land in `EXCLUSIONS`, not in their own refinement. Also add the `EXCLUSIONS` entry for `{dtype: float32, fp32_dest_acc_en: False}` (maxed input + non-maxed acc is rejected, mirrors softmax convention per feature_spec.py).

**Implementation skill**: /numeric-formats-metal

**Verifier notes**: should land first — the dtype-driven CB format derivation (fp32 vs bf16 intermediate CBs) and the `UnpackToDestFp32` tagging are needed before the layout refinement can handle non-aligned shapes correctly with the full dtype set. The tolerance matrix in `helpers.py` already defines per-(dtype, fp32_dest_acc_en) thresholds. The current kernel uses fp32 intermediate CBs for scores/max/sum/O — the `/numeric-formats-metal` skill will determine whether those stay fp32 for all dtypes or switch to dtype-driven formats.

**Done when**: `SUPPORTED["dtype"]` contains `[ttnn.float32, ttnn.bfloat16, ttnn.bfloat8_b]` and `SUPPORTED["fp32_dest_acc_en"]` contains `[True, False]`, with all cells in the SUPPORTED rectangle passing (except any in EXCLUSIONS).

### [x] Refinement 2 — Non-tile-aligned shape support

**Goal**: add `"w_non_aligned"` and `"h_non_aligned"` to `SUPPORTED["alignment"]` via in-kernel edge-tile handling. `w_non_aligned` (D not divisible by 32) requires zero-padding the last D-tile column in the reader and masking the PV matmul output. `h_non_aligned` (S_q not divisible by 32) requires zero-padding the last Q-tile row and masking the output.

**Implementation skill**: /memory-layouts

**Verifier notes**: SDPA is TILE-only (no ROW_MAJOR in TARGET), so this is purely about non-aligned dimensions within TILE layout — the reader must zero-pad partial tiles and the writer must mask the valid region. The current kernel assumes all dimensions are tile-aligned; the `B_q_t` divisor logic in the program descriptor only handles tile-aligned S_q. The `/memory-layouts` skill's "non-aligned rule" (last-tile H/W zero-pad / mask done in the reader or compute) applies. Bundles with the multi-core distribution naturally since the reader rewrite is the primary change.

### [x] Refinement 3 — Causal masking

**Goal**: add `"causal"` to `SUPPORTED["mask_mode"]`. When `is_causal=True`, the compute kernel generates a triangular -inf mask on-device per (Q-block, KV-block) pair instead of reading an additive mask tensor. Add the `{"mask_mode": "causal", "attention_kind": "cross"}` EXCLUSION (causal masking requires S_q == S_kv). Also add `is_causal=True` to the SUPPORTED path (currently rejected because mask_mode="causal" is not in SUPPORTED).

**Verifier notes**: This is an algorithm-fundamental change — the causal mask is generated per-block in the compute kernel, not read from DRAM. Three regions per (Q-block, KV-block) pair: fully-past (no mask), fully-future (skip entire block — the causal perf win), diagonal-straddling (per-element triangular -inf mask). The block-skip for fully-future blocks requires the compute kernel to conditionally skip QK^T/softmax/PV for blocks where all positions are above the causal diagonal. `is_causal=True` + `attn_mask is not None` is already a ValueError in validate(). The `default_compute_kernel_config()` already exists and is the single source of truth. No skill pointer — the causal mask generation is a compute-kernel algorithm change outside the current skill inventory's scope.

**Done when**: `SUPPORTED["mask_mode"]` contains `["none", "custom", "causal"]` and `is_causal=True` cells pass for self-attention (S_q == S_kv). The `{mask_mode: causal, attention_kind: cross}` EXCLUSION is declared.

### [x] Refinement 4 — L1 budget fit for large head dims

**Goal**: rewrite the output accumulation and score CB sizing so the per-core L1 CB footprint is bounded by a constant (not by `D_t`), so the op stops OOMing on D=512 (D_t=16) and D=1024 (D_t=32). The 8 currently-failing OOM cells all have D ≥ 512 within the SUPPORTED rectangle. No SUPPORTED axis is added — this is a resource boundary, not a kernel-level branch.

**Implementation skill**: /memory-budget-metal

**Verifier notes**: The OOM is caused by `cb_o` and `cb_o_accum`, each sized `B_q_t * D_t` pages × fp32_tile_size. For D=1024: 4 * 32 = 128 tiles × 4 KB = 512 KB each → 1 MB for just these two CBs, exceeding 1.5 MB with the other CBs. The `/memory-budget-metal` skill's K-blocking pattern applies: chunk the PV matmul's D dimension into sub-blocks, accumulate into a smaller cb_o, and write partial results to DRAM. The matmul_block helper's `num_k_blocks > 1` + weights restreaming pattern is the natural fit. This refinement is independent of the dtype refinement — the OOM occurs at bf16 (Phase 0 baseline).

**Done when**: every Phase 0 cell currently in the `OOM` category passes.

### [x] Refinement 5 — Large unequal-seqlen cross-attention

**Goal**: stop the op from hanging the device on cross-attention with large, unequal sequence lengths (S_q ≠ S_kv, large S). The op currently times out (`TT_THROW: TIMEOUT: device timeout in fetch queue wait, potential hang detected`) on `test_sdpa_noncausal_unequal_seqlen__nightly[1-8-1-4096-2048-128-k256-q128-bfp8]` — S_q=4096, S_kv=2048, d=128, bf8b, noncausal. No SUPPORTED axis is added — this is a resource/correctness boundary for large asymmetric cross-attention shapes that the translated (blind) suite exposes but the golden suite does not cover.

**Implementation skill**: /memory-budget-metal

**Verifier notes**: R4's K-blocking chunks the PV matmul's D dimension, bounding CBs against large head dims — but it does not chunk the S_kv (sequence) dimension the attention loops over. The crash cell has small D (128, no D-pressure) but large asymmetric S (S_q=4096 ≠ S_kv=2048). The hang is a fetch-queue timeout, indicating the kernel's KV-block loop over S_kv deadlocks or exhausts a resource when S is large and Q/KV tile counts don't divide evenly. Investigate the `num_kv_blocks` loop + the reader's KV streaming for large unequal S_q/S_kv; the `/memory-budget-metal` streaming-reduce pattern (chunk S_kv, accumulate partial softmax stats, spill to DRAM) likely applies. Note: translated-suite cells are run only as the final blind pass (non-gating); this refinement targets the real defect the blind pass surfaced.

**Done when**: `test_sdpa_noncausal_unequal_seqlen__nightly[1-8-1-4096-2048-128-k256-q128-bfp8]` (and the other large unequal-seqlen cross-attention cells) run to completion without a fetch-queue timeout.

**Target test file**: `eval/golden_tests/scaled_dot_product_attention/test_translated.py` — the failing cell lives here. Run it directly to verify your fix: `scripts/run_safe_pytest.sh "eval/golden_tests/scaled_dot_product_attention/test_translated.py::test_sdpa_noncausal_unequal_seqlen__nightly[1-8-1-4096-2048-128-k256-q128-bfp8]"`. Note: `test_translated.py` is EXCLUDED from the gate's golden runs (it's a blind, non-gating suite), so the mechanical completion gate cannot verify this refinement — you MUST verify the cell yourself by running it directly and confirming it passes without a timeout. The gate's bullets 1 (no hangs in golden) and 3 (golden majority) will pass vacuously for this refinement since the target isn't in the gated golden suite; your direct verification of the translated cell is the real done criterion.
