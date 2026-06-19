# Operation Requirements: scaled_dot_product_attention

## Definition
- **Formula**: `O[b,h,:,:] = softmax( (Q[b,h]·K[b,h]ᵀ)·scale + mask[b,·] , dim=-1 ) · V[b,h]`
- **PyTorch Reference**:
  ```python
  def sdpa(Q, K, V, *, attention_mask=None, scale=None):
      Qf, Kf, Vf = Q.float(), K.float(), V.float()
      H_q, H_kv = Qf.shape[1], Kf.shape[1]
      if H_q != H_kv:                       # GQA / MQA head broadcast
          r = H_q // H_kv
          Kf, Vf = Kf.repeat_interleave(r, 1), Vf.repeat_interleave(r, 1)
      s = scale if scale is not None else 1.0 / math.sqrt(Qf.shape[-1])
      scores = (Qf @ Kf.transpose(-2, -1)) * s
      if attention_mask is not None:
          scores = scores + attention_mask.float()
      return (torch.softmax(scores, dim=-1) @ Vf).to(Q.dtype)
  ```
- **Import Path**: `from ttnn.operations.scaled_dot_product_attention import scaled_dot_product_attention`
- **Function Signature**:
  ```python
  scaled_dot_product_attention(
      Q: ttnn.Tensor,                       # (B, H, S_q, D)
      K: ttnn.Tensor,                       # (B, H_kv, S_kv, D)
      V: ttnn.Tensor,                       # (B, H_kv, S_kv, D)
      *,
      attention_mask: ttnn.Tensor = None,   # (B, 1, S_q, S_kv) or (B, H, S_q, S_kv), additive
      scale: float = None,                  # if None, 1/sqrt(D)
      memory_config: ttnn.MemoryConfig = None,
  ) -> ttnn.Tensor                          # (B, H, S_q, D)
  ```

## Phases

> **Non-regression rule**: Every refinement must pass all tests from prior phases.
> **Drift signal**: XPASS-strict failures mean the implementer added support but forgot to update SUPPORTED. The implementer fixes by updating SUPPORTED.
> **Checkbox protocol**: `[x]` complete and all tests pass; `[~]` real work landed but at least one named axis value deferred (treated as completed, surfaced as partial); `[ ]` nothing usable produced.

### [x] Phase 0 — Core Implementation

- **SUPPORTED dtype**: [ttnn.bfloat16]
- **SUPPORTED layout**: [ttnn.TILE_LAYOUT]  (TILE-only by design — ROW_MAJOR is not in TARGET)
- **SUPPORTED alignment**: ["tile_aligned"]
- **SUPPORTED attention_kind**: ["self", "cross"]
- **SUPPORTED kv_heads_mode**: ["mha"]
- **SUPPORTED mask_mode**: ["none", "causal"]
- **SUPPORTED scale_mode**: ["auto", "explicit"]
- **EXCLUSIONS**: [] (causal+cross is supported — kernel adds a dense additive mask block-by-block, never assumes S_q==S_kv)
- **Cores**: multi-core, `split_work_to_cores` over `B·H·Sq_t` units (embarrassingly parallel, no inter-core comm)
- **Compute config**: hard-coded HiFi2 + `fp32_dest_acc_en=True`, `dst_full_sync_en=False`; bf16 intermediate CBs (fp32 CBs hang this LLK — Issue #13364)
- **Algorithm**: Flash Attention — tiled online softmax, `cb_scores`/`cb_p` sized to one `(1 × kv_chunk_t)` block; running m/l/O are the only KV-surviving state (O(S) memory)
- **Golden baseline**: **140 / 140** supported cells passing (verifier CLI: supported_fail=0, xpass_drift=0, xfail_wrong_mode=0). 604 cells correctly xfail toward TARGET.
- **Precision (randn, bf16)**: PCC ≥ 0.99996, relative RMS ~1% across (1,1,32,32)…(1,8,512,64).

---

### [ ] Refinement 1 — Numerical configurability (float32 + bfloat8_b + compute_kernel_config)

**Goal**: add `ttnn.float32` and `ttnn.bfloat8_b` to `SUPPORTED["dtype"]`, expose a
`compute_kernel_config: ttnn.ComputeKernelConfig` argument on the public entry point,
and correct intermediate-CB / output precision so the op runs at each input dtype.
Unblocks the 248 `dtype=float32` xfail cells and the 248 `dtype=bfloat8_b` xfail cells.
Cells that fail out of the box (typically `bfloat8_b + non_tile_aligned`, and possibly
`bfloat8_b + causal` if the block-exponent `−inf` mask misbehaves) land in
`EXCLUSIONS`, not in their own refinement.

**Implementation skill**: /numeric-formats-metal

**Verifier notes**:
- Land this **first** — it is the descriptor-level change (CB format derivation +
  config plumbing) that R2 and R3 both build on.
- **Hard constraint from Phase 0**: the intermediate matmul/reduce-output CBs
  (`cb_scores`, `cb_pv`, `cb_out_accum`, `cb_max`, `cb_sum`) must stay bf16 even for
  the float32 input dtype — fp32 *CB storage* hits the Issue #13364
  `pack_reconfig`→`TTI_STALLWAIT(PACK|THCON)` hang (see kernel header +
  `test_scaled_dot_product_attention_debug.py`). Precision stays in the fp32 DEST
  (`fp32_dest_acc_en`). For float32 inputs this means input tiles are unpacked to
  bf16 for the matmuls; that is the production SDPA behavior, not a regression.
  Treat the `compute_kernel_config` surface (math_fidelity / math_approx_mode) as
  the configurable lever; do **not** try to flip the intermediate CBs to fp32.
- **Output dtype**: the program descriptor currently hard-codes the output to
  `ttnn.bfloat16`. Make the output dtype follow the input dtype (and honor any
  caller `memory_config`/dtype).
- This refinement does **not** address the bf16-accumulator precision limitation on
  adversarial distributions (uniform/negative) — that is a separate, currently
  lever-less issue tracked in `verification_report.md`, not here.

**Done when**: every `dtype ∈ {float32, bfloat8_b}` cell either passes or is in
`EXCLUSIONS`; the 24 acceptance tests and 140 Phase-0 golden cells still pass.

---

### [ ] Refinement 2 — GQA / MQA (kv_heads_mode)

**Goal**: add `"gqa"` and `"mqa"` to `SUPPORTED["kv_heads_mode"]`. Unblocks the 132
`kv_heads_mode=gqa` and 96 `kv_heads_mode=mqa` xfail cells (Q has more heads than
K/V; each Q head reads its grouped KV head, `h_kv = h / (H_q/H_kv)`).

**Verifier notes**: (verifier-authored — no inventory skill covers head-index
remapping; it is neither a layout/dtype change nor a cross-core-dependency
algorithm.)
- The reader **already** implements the mapping: `head_group = H / H_kv` and
  `h_kv = h / head_group`, with `kv_head_base = (b*H_kv + h_kv)*Skv_t`. The work
  split is already over `B·H·Sq_t` units (one per Q head), so GQA/MQA is
  embarrassingly parallel with **no** new multi-core distribution and **no**
  inter-core communication — do not reach for `/interleaved-parallel`.
- Expected scope: remove the `kv_heads_mode` gate in `validate()` (extend
  `SUPPORTED`), then verify. The structural check `Q.shape[1] % K.shape[1] == 0`
  is already in `_check_structural`. The mask head index already handles
  `mask_H ∈ {1, H}`.
- The standing `test_regression.py::test_gqa_mqa_forward` cases (currently
  `UnsupportedAxisValue` rejections, 4 of the 18 regression "failures") should pass
  once the gate is lifted — use them plus the golden GQA/MQA cells to verify.
- Independent of R1; the crossed `dtype×gqa` / `dtype×mqa` cells need *both* R1 and
  R2 before they pass.

**Done when**: every `kv_heads_mode ∈ {gqa, mqa}` cell at supported dtype passes;
`test_gqa_mqa_forward` passes; prior phases still green.

---

### [ ] Refinement 3 — Non-tile-aligned shapes (alignment w / h)

**Goal**: add `"w_non_aligned"` and `"h_non_aligned"` to `SUPPORTED["alignment"]` via
**in-kernel** edge handling (zero-pad / mask the partial last tile in the reader or
compute — no `ttnn.to_layout`/`tilize` wrapper). Unblocks the 60 `w_non_aligned`
and 60 `h_non_aligned` xfail cells. SDPA stays TILE-layout; only the alignment axis
expands, so this is the `/memory-layouts` non-aligned methodology, not a new Layout.

**Implementation skill**: /memory-layouts

**Verifier notes**:
- Land **last** — edge-masking is the trickiest path and interacts with both the
  dtype set (R1) and the KV reduction. By this point SUPPORTED dtype/kv_heads should
  be stable.
- Three distinct edges, all must be handled:
  - **`w_non_aligned` (D not %32)**: D is the QKᵀ *contraction* dim and the PV *free*
    dim. The padded columns of the last D-tile must be zero in Q and K so they
    contribute 0 to `Q·Kᵀ`; the PV output's padded D-columns must be dropped on
    write-back.
  - **`h_non_aligned` (S_q not %32)**: the last query tile-row has < 32 valid rows.
    Math is unaffected per-row; the writer must not emit the padding rows.
  - **S_kv not %32 (the subtle one)**: `tag_alignment` only inspects Q's `(S_q, D)`,
    but the non-aligned INPUTS bundle non-aligned `S_kv` too (e.g.
    `((1,4,100,50),(1,4,47,50),…)`). The padded key columns of the last KV block
    must be `−inf` **before** the row-max / row-sum, or the softmax denominator is
    wrong. This is the structural part the skill's "mask in the reader/compute"
    rule targets — handle it in the reduce input, not by post-hoc correction.
- **Partial-tick fallback**: if native in-kernel KV-edge masking is more than a
  focused pass, `[~]`-partial by landing `h_non_aligned` (writer row-trim only,
  cheapest) and deferring `w_non_aligned` + the `S_kv` edge to a follow-up; call out
  the deferral in `changelog.md`. Do **not** substitute a `ttnn.to_layout`/`tilize`
  wrapper — SUPPORTED must reflect real kernel capability.
- `bfloat8_b + non_aligned` cells are expected to land in `EXCLUSIONS` (set in R1),
  not become passing here.

**Done when**: every `alignment ∈ {w_non_aligned, h_non_aligned}` cell at supported
dtype/kv_heads passes or is in `EXCLUSIONS`; prior phases still green.
