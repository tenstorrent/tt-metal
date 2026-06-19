# Operation Requirements: scaled_dot_product_attention

## Definition
- **Formula**: `O[b,h,i,:] = Σ_j softmax_j( (Q[b,h,i,:]·K[b,h,j,:]ᵀ)·scale + mask[b,h,i,j] ) · V[b,h,j,:]`
  computed with the Flash-Attention online (running) softmax — the full
  `S_q × S_kv` score matrix is never materialized.
- **PyTorch Reference**:
  ```python
  def sdpa(Q, K, V, attention_mask=None, scale=None):
      # Q (B,H_q,S_q,D); K,V (B,H_kv,S_kv,D), H_q % H_kv == 0
      Qf, Kf, Vf = Q.float(), K.float(), V.float()
      if Q.shape[1] != K.shape[1]:                 # GQA / MQA head broadcast
          r = Q.shape[1] // K.shape[1]
          Kf = Kf.repeat_interleave(r, dim=1); Vf = Vf.repeat_interleave(r, dim=1)
      s = scale if scale is not None else 1.0 / (Qf.shape[-1] ** 0.5)
      scores = (Qf @ Kf.transpose(-2, -1)) * s
      if attention_mask is not None:
          scores = scores + attention_mask.float()
      return (torch.softmax(scores, dim=-1) @ Vf).to(Q.dtype)
  ```
- **Import Path**: `from ttnn.operations.scaled_dot_product_attention import scaled_dot_product_attention`
- **Function Signature**:
  ```python
  scaled_dot_product_attention(
      Q: ttnn.Tensor,                     # (B, H_q,  S_q,  D)
      K: ttnn.Tensor,                     # (B, H_kv, S_kv, D)
      V: ttnn.Tensor,                     # (B, H_kv, S_kv, D)
      *,
      attention_mask: ttnn.Tensor = None, # (B, 1|H_q, S_q, S_kv) additive
      scale: float = None,                # 1/sqrt(D) when None
  ) -> ttnn.Tensor                        # (B, H_q, S_q, D)
  ```

## Phases

> **Non-regression rule**: Every refinement must pass all tests from prior phases.
> **Drift signal**: XPASS-strict failures mean the implementer added support but forgot to update SUPPORTED. The implementer fixes by updating SUPPORTED.
> **Checkbox protocol**: Implementer marks `[x]` when the refinement is complete and all tests pass, `[~]` when real work landed but at least one named axis value is deferred (treated as completed by the queue, surfaced as partial), `[ ]` only when nothing usable was produced.

### [x] Phase 0 — Core Implementation

- **SUPPORTED dtype**: [bfloat16]
- **SUPPORTED layout**: [TILE]
- **SUPPORTED shape-derived axes**: alignment=tile_aligned only
- **SUPPORTED op-specific axes**: attention_kind ∈ {self, cross}; kv_heads_mode ∈ {mha, gqa, mqa}; mask_mode ∈ {none, causal}; scale_mode ∈ {auto, explicit}
- **Cores**: multi-core (embarrassingly parallel; `split_work_to_cores`, one `(b, h_q, qb)` work item per stamp, interleaved DRAM) — already done at Phase 0.
- **Compute config**: hard-coded `MathFidelity.HiFi2`, `fp32_dest_acc_en` off (avoids the HiFi4+fp32+bf16 SUM-reduce combo). Not caller-configurable.
- **Golden baseline**: 207 / 208 supported cells passing (per verifier CLI); 536 correctly xfailed; the 1 failing cell is the `Q1x1x128x1024` bf16 explicit-scale precision near-miss (category `numerical-precision`).

### [x] Refinement 1 — Numerical configurability expansion

**Goal**: add `ttnn.float32` and `ttnn.bfloat8_b` to `SUPPORTED["dtype"]`,
expose `compute_kernel_config: ttnn.ComputeKernelConfig` on the entry point
(`math_fidelity`, `fp32_dest_acc_en`, `math_approx_mode`), and correct
intermediate-CB precision (dtype-aware CB formats; `UnpackToDestFp32`
tagging where it applies). This also moves the one Phase-0 `supported_fail`
(`Q1x1x128x1024` bf16 explicit-scale, `numerical-precision`) to passing —
the precision lever (fp32_dest_acc / higher fidelity on the
matmul→softmax→matmul chain) is exactly this refinement's scope. Cells that
fail out of the box (typically `bfloat8_b` on the wider-`D` /
deeper-KV-accumulation shapes) land in `EXCLUSIONS`, not their own
refinement.

**Implementation skill**: /numeric-formats-metal

**Verifier notes**: lands first. (1) It clears the only registry red cell
and the genuine-precision regression cases (`test_large_magnitude_input`,
near-constant-output `negative`/`uniform`). (2) **Constraint flagged during
Phase-0 bring-up**: the matmul-path SUM reduce hard-codes HiFi4, and
`fp32_dest_acc_en` + that path is the known-bad combo (issue #38306) — the
config plumbing must thread `fp32_dest_acc` so it does *not* reach that
reduce, or the 128×1024 cell stays in `EXCLUSIONS`. (3) Refinement 2 reuses
the dtype-driven CB-format derivation introduced here, so this must precede
it.

### [x] Refinement 2 — Non-tile-aligned shape support

**Goal**: add `"w_non_aligned"` (D not %32) and `"h_non_aligned"` (S_q not
%32, D aligned) to `SUPPORTED["alignment"]` via **in-kernel** data-access
changes — the partial last tile is zero-padded / masked in the reader or
compute, math stays on tiles. Two distinct masking concerns:
- **D non-aligned**: the QK contraction (`in0_block_k = DHt = ceil(D/32)`)
  and the V/output `D` dim include a partial last tile — zero-pad the unused
  lanes in the reader so the dot product and the PV matmul don't accumulate
  garbage.
- **S_kv non-aligned**: the score block's free dim is partial — the padded
  KV columns MUST be masked to −∞ before the row-max / row-sum, or the
  softmax denominator picks up `exp(0 − m)` from padding. **S_q
  non-aligned**: the output's last seq tile is partial — the writer simply
  skips the padding rows.

**Implementation skill**: /memory-layouts

**Verifier notes**: lands second — structurally harder than R1 and depends
on R1's dtype-aware CB formats. Make the non-aligned reader/compute path
dtype-aware so the 80 `(alignment, dtype)` intersection cells
(fp32/bf8b × non-aligned) also clear; per the standard block-format
limitation, `bfloat8_b × non_aligned` may need `EXCLUSIONS` rather than a
fight. This is the **softmax-denominator masking** flavor of non-alignment
(mask the partial last tile), not a structural reduction-unit change —
`/memory-layouts`'s "non-aligned rule (last-tile H/W zero-pad / mask in the
reader or compute)" is the right pattern. **Do not** wrap the op in
`ttnn.to_layout` / `ttnn.fill_implicit_tile_padding` and forward to the
existing kernel — the native in-kernel masking is the target; a
manipulation-op wrapper is only the `[~]` partial-tick escape hatch (name
the specific blocked kernel work in the changelog if you take it, and file
the in-kernel masking as the follow-up).

**Done when**: every Phase-0 `xfail_expected` cell with `alignment ∈
{w_non_aligned, h_non_aligned}` and a SUPPORTED dtype passes (bf8b ×
non-aligned permitted in EXCLUSIONS).
