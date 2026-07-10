# Operation Requirements: scaled_dot_product_attention

## Definition
- **Formula**: `O[b,h,:,:] = softmax( (Q[b,h]·K[b,h]ᵀ)·scale + M ) · V[b,h]`, softmax
  over the `S_kv` (last) axis. `M` = additive mask (custom tensor, on-device causal
  triangle, or none). Flash-Attention: online-softmax over the KV sequence, so the
  full `S_q × S_kv` score matrix is never materialized.
- **PyTorch Reference**:
  ```python
  def reference(Q, K, V, attn_mask=None, is_causal=False, scale=None):
      Qf, Kf, Vf = Q.float(), K.float(), V.float()
      H_q, H_kv = Qf.shape[1], Kf.shape[1]
      if H_q != H_kv:                       # GQA / MQA broadcast
          r = H_q // H_kv
          Kf, Vf = Kf.repeat_interleave(r, 1), Vf.repeat_interleave(r, 1)
      return torch.nn.functional.scaled_dot_product_attention(
          Qf, Kf, Vf, attn_mask=attn_mask, is_causal=is_causal, scale=scale)
  ```
- **Import Path**: `from ttnn.operations.scaled_dot_product_attention import scaled_dot_product_attention`
- **Function Signature**:
  ```python
  scaled_dot_product_attention(
      query: ttnn.Tensor,                       # (B, H_q,  S_q,  D)
      key: ttnn.Tensor,                         # (B, H_kv, S_kv, D)
      value: ttnn.Tensor,                       # (B, H_kv, S_kv, D)
      *,
      attn_mask: ttnn.Tensor = None,            # (B, 1|H_q, S_q, S_kv) additive
      is_causal: bool = False,                  # mutually exclusive with attn_mask
      scale: float = None,                      # None => 1/sqrt(D)
      compute_kernel_config: ttnn.ComputeConfigDescriptor = None,
      memory_config: ttnn.MemoryConfig = None,
  ) -> ttnn.Tensor                             # (B, H_q, S_q, D)
  ```

## Phases

> **Non-regression rule**: Every refinement must pass all tests from prior phases.
> **Drift signal**: XPASS-strict failures mean the implementer added support but forgot to update SUPPORTED. The implementer fixes by updating SUPPORTED.
> **Checkbox protocol**: Implementer marks `[x]` when the refinement is complete and all tests pass, `[~]` when real work landed but at least one named axis value is deferred (treated as completed by the queue, surfaced as partial), `[ ]` only when nothing usable was produced.
> **Refinement ID + follow-up naming (mandatory)**: Primary refinements are `Refinement N`. Partial-tick follow-ups append a lowercase letter to the parent's number (`Refinement 1b`, …) and are ordered immediately after their parent.

### [x] Phase 0 — Core Implementation

- **SUPPORTED dtype**: [bfloat16]
- **SUPPORTED fp32_dest_acc_en**: [True, False]
- **SUPPORTED layout**: [TILE]
- **SUPPORTED alignment**: tile_aligned only
- **SUPPORTED attention_kind**: [self, cross]
- **SUPPORTED kv_heads_mode**: [mha, gqa, mqa]
- **SUPPORTED mask_mode**: [none, custom]
- **SUPPORTED scale_mode**: [auto, explicit]
- **Cores**: multi-core (`split_work_to_cores`-equivalent contiguous Q-block runs; embarrassingly parallel, no inter-core comms)
- **Compute config**: `compute_kernel_config` exposed; default HiFi4 + `fp32_dest_acc_en=True`
- **Golden baseline**: 520 / 571 supported cells passing (per verifier CLI). 51 supported_fail = 24 OOM (large head_dim) + 27 numerical-precision (extreme-length translated tests); both queued below / documented.
- **Precision**: PCC ≥ 0.99999 across 4 shapes (`test_scaled_dot_product_attention_precision_baseline.py`)

### [x] Refinement 1 — Numerical configurability expansion (dtype)

**Goal**: add `ttnn.float32` and `ttnn.bfloat8_b` to `SUPPORTED["dtype"]`. The
entry point already exposes `compute_kernel_config`; wire dtype-driven CB formats
(tile-format derivation for the streaming Q/K/V/mask/out CBs; the fp32 running-state
CBs already exist), and add `UnpackToDestFp32` tagging where it applies. Keep the
`{dtype: float32, fp32_dest_acc_en: False}` EXCLUSION (already declared). Per the
design: fp32 requires HiFi4 + `fp32_dest_acc_en`; bf16 with `num_k_blocks==1` single-
K-block matmuls needs `fp32_dest_acc_en=True` for full correctness. Cells that fail
out of the box — typically `bfloat8_b + w_non_aligned` / `bfloat8_b + h_non_aligned`
(block-format + partial tile) — land in `EXCLUSIONS`, not their own refinement.

**Implementation skill**: /numeric-formats-metal

**Verifier notes**: land this first — it is foundational, and Refinements 2 and 4
reuse the dtype-driven CB-format derivation introduced here. Unlocks 452 (bf8b) +
424 (fp32) xfail_expected cells (minus the bf8b×non-aligned cells, which depend on
Refinement 2's alignment path and stay excluded until then). Do **not** split bf8b
into its own entry.

### [x] Refinement 2 — Non-tile-aligned sequence / head dim

**Goal**: add `"w_non_aligned"` (D % 32 ≠ 0) and `"h_non_aligned"` (S_q % 32 ≠ 0, D
aligned) to `SUPPORTED["alignment"]`, natively in the kernel — no `ttnn.tilize` /
`to_layout` wrapper. Handle the partial last tile along S/D: zero-pad / mask the
padded score columns to −∞ before the row-max (so softmax ignores padding), and use
the partial-scaler pair for the reductions (`ReducePartialScaler::last_tile_at(...)`
+ `calculate_and_prepare_partial_reduce_scalers`, `reduce_helpers_dataflow.hpp:149`).
Applies to both Q's (S_q, D) and the KV (S_kv, D) legs through the same reader/compute
edge-handling.

**Implementation skill**: /memory-layouts

**Verifier notes**: unlocks the 113 (h_non_aligned) + 40 (w_non_aligned) pure-alignment
xfail cells, plus the bf8b×non-aligned and fp32×non-aligned combos once Refinement 1
has landed the dtypes. Depends on R1 only for the dtype×alignment cross cells; the
bf16 alignment cells are independent and can be validated first. The design's
Refinements section (`op_design.md` §198) has the partial-scaler recipe. This is the
edge-padding case, not a structural reduction-shape change — no new INPUT_TAGGER.

### [ ] Refinement 3 — Native causal masking (`is_causal=True`)

**Goal**: add `"causal"` to `SUPPORTED["mask_mode"]`. Derive the triangular bias
**on-device**, block-by-block — never materialize the full `S_q × S_kv` mask. Three
regions per `(Q-block, KV-block)`: past blocks unmasked; future blocks are whole-tile
−∞ and **should be skipped entirely** (don't run QKᵀ/softmax/PV — the ≈½ KV-work
causal win); the diagonal-straddling block gets a per-element triangular −∞ generated
on-device (SFPU fill / iota-compare), applied additively onto scores before the
row-max (same additive primitive as the custom mask). When adding causal, also declare
`EXCLUSION {mask_mode: causal, attention_kind: cross}` (causal requires `S_q == S_kv`)
and raise `ValueError` when `is_causal=True` and `attn_mask is not None` (already
enforced in `validate()`).

**Verifier notes**: standalone — this is an algorithm-fundamental change (on-device
mask generation + block-skipping control flow), not covered by any current skill.
Independent of R1/R2 for the bf16 tile-aligned path; the causal × {fp32, bf8b} and
causal × non-aligned cross-product cells (242 + 212 + 92 + 20×… xfail cells) light up
only after R1/R2 respectively, so schedule after them to avoid re-touching the causal
path. Work from `op_design.md` §196 (Refinements & Rules → mask_mode=causal).

**Done when**: the `mask_mode=causal` xfail cells for bf16 tile-aligned pass, and the
`{causal, cross}` cells remain properly rejected (xfail via the new EXCLUSION).

### [ ] Refinement 4 — L1 budget fit for large head_dim

**Goal**: rewrite the QKᵀ / PV data flow so the per-core L1 CB footprint is bounded
regardless of `Dt` (head_dim tiles), so the op stops OOMing on the head-dim-scaling
golden cells `Q(1,1,128,{256,512,1024})` (`D ∈ {256, 512, 1024}`, `Dt ∈ {8,16,32}`).
Phase 0 sizes `cb_q`/`cb_qs`/`cb_k`/`cb_v` and the fp32 `cb_o_run`/`cb_o_new`/`cb_pv`
at `q_chunk_t·Dt` (or `kv_chunk_t·Dt`), which exceeds the ~1.5 MB budget around
`D = 512`. Block the QKᵀ contraction over `Dt` (`num_k_blocks > 1`, K restreamed) and
bound the PV output-width CBs. No SUPPORTED axis is added — head_dim size is a resource
boundary, not a kernel branch; a `shape_size` tagger would only hide the gap.

**Implementation skill**: /memory-budget-metal

**Verifier notes**: memory-pressure refinement — schedule last, after R1–R3 stabilize
the SUPPORTED rectangle (D=256/512/1024 cells must then also carry the R1 dtypes and
R2 alignment). The skill's matmul K-blocking pattern (`num_k_blocks > 1`, weights
restreaming, `PreKBlockFn`) is the natural fit for the QKᵀ `Dt` contraction; the PV
matmul's output width (`q_chunk_t·Dt`) is the other lever. This is **not** a multi-core
refinement — the op is already multi-core and stamping more cores does not shrink the
per-core CBs. The extreme-length `fp32_dest_acc_en=False` numerical-precision failures
(translated `full-grid` S=45056, S=4096 tight-RMSE) are a **separate** concern with no
clean in-kernel lever — documented in `verification_report.md`, not bundled here.

**Done when**: every Phase-0 golden cell currently in the `OOM` category
(`Q(1,1,128,256/512/1024)`, both `fp32_dest_acc_en` values, both mask/scale modes)
passes.
