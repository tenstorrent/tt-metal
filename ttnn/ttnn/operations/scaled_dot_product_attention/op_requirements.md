# Operation Requirements: scaled_dot_product_attention

## Definition
- **Formula**: `O[b,h,:,:] = softmax( (Q·Kᵀ)·scale + M , dim=-1 ) · V`, computed
  block-by-block via the Flash Attention online-softmax recurrence (running
  max / running sum / running output; the full S_q × S_kv score matrix is never
  materialized).
- **PyTorch Reference**:
  ```python
  def sdpa(q, k, v, attn_mask=None, is_causal=False, scale=None):
      scale = scale if scale is not None else 1.0 / math.sqrt(q.shape[-1])
      scores = torch.matmul(q, k.transpose(-2, -1)) * scale
      if is_causal:
          s_q, s_kv = q.shape[-2], k.shape[-2]
          scores = scores + torch.triu(torch.full((s_q, s_kv), float("-inf")), diagonal=1)
      elif attn_mask is not None:
          scores = scores + attn_mask
      return torch.matmul(torch.softmax(scores, dim=-1), v)
  ```
- **Import Path**: `from ttnn.operations.scaled_dot_product_attention import scaled_dot_product_attention`
- **Function Signature**:
  ```python
  scaled_dot_product_attention(
      query: ttnn.Tensor,                  # (B, H, S_q, D)
      key:   ttnn.Tensor,                  # (B, H, S_kv, D)
      value: ttnn.Tensor,                  # (B, H, S_kv, D)
      *,
      attn_mask: ttnn.Tensor = None,       # (B, 1, S_q, S_kv) or (B, H, S_q, S_kv) additive
      is_causal: bool = False,
      scale: float = None,                 # None → 1/sqrt(D)
  ) -> ttnn.Tensor                         # (B, H, S_q, D)
  ```

## Phases

> **Non-regression rule**: Every refinement must pass all tests from prior phases.
> **Drift signal**: XPASS-strict failures mean the implementer added support but forgot to update SUPPORTED. The implementer fixes by updating SUPPORTED.
> **Checkbox protocol**: `[x]` complete + all tests pass; `[~]` real work landed but ≥1 named axis value deferred; `[ ]` nothing usable produced.

### [x] Phase 0 — Core Implementation

- **SUPPORTED dtype**: [bfloat16]
- **SUPPORTED layout**: [TILE] (TARGET is TILE-only — SDPA has no ROW_MAJOR path)
- **SUPPORTED alignment**: [tile_aligned]
- **SUPPORTED attention_kind**: [self, cross]
- **SUPPORTED kv_heads_mode**: [mha]
- **SUPPORTED mask_mode**: [none, custom]
- **SUPPORTED scale_mode**: [auto, explicit]
- **Cores**: multi-core, embarrassingly parallel — `split_work_to_cores` over
  `(b, h, q_block)` work units across the full compute grid (already done).
- **Per-core memory**: O(1) in sequence length (`q_chunk_t == k_chunk_t == 1`);
  long-context S=4096/8192 runs without OOM.
- **Compute config**: hard-coded HiFi2 + `fp32_dest_acc_en=True`; running
  accumulators (`m_i`, `l_i`, `O_i`) and their scratch held in fp32 CBs (fixed
  during verification).
- **Golden baseline**: 140 / 1156 cells passing; 976 xfail_expected; loud
  categories all 0 (per `verifier_report.json`).
- **Precision**: PCC ≈ 0.9999, rel-RMS ≈ 1.3–1.6 % on random-normal bf16 inputs
  (single-tile → S=1024).

---

### [ ] Refinement 1 — Numerical configurability (dtype + compute_kernel_config)

**Goal**: add `ttnn.float32` and `ttnn.bfloat8_b` to `SUPPORTED["dtype"]`, expose
`compute_kernel_config: ttnn.ComputeKernelConfig` on the entry point (so
`math_fidelity`, `fp32_dest_acc_en`, `math_approx_mode` are caller-controlled
instead of hard-coded HiFi2), and derive all CB data formats from the input
dtype (subsuming the fp32-accumulator formats wired in during verification).
Add `UnpackToDestFp32` tagging where the fp32 path needs it. Cells that fail
out of the box land in `EXCLUSIONS` — the canonical one here is
`{"dtype": ttnn.bfloat8_b, "alignment": "w_non_aligned"}` /
`{"dtype": ttnn.bfloat8_b, "alignment": "h_non_aligned"}` (bf8b block-float on a
partial last tile), which only becomes reachable once Refinement 4 adds the
non-aligned alignment values — add the EXCLUSION when both land.

**Implementation skill**: /numeric-formats-metal

**Verifier notes**: land this first. The CB-format-from-dtype derivation it
introduces should replace the explicit `f32` CB formats added during
verification, so the two precision concerns are handled in one place. This is
also the home for the score-path precision lever (`cb_qk`/`cb_p` fp32, or
exposed `math_fidelity`/`fp32_dest_acc_en`) that lifts the non-registry
`test_negative_input` / `test_uniform_input` canaries — no separate entry for
those. Covers `(dtype, FLOAT32)` and `(dtype, BFLOAT8_B)`; the dtype axis
appears in 744 of the 976 xfail_expected cells, so this is the largest unlock.

---

### [ ] Refinement 2 — GQA / MQA (KV head broadcast)

**Goal**: add `"gqa"` and `"mqa"` to `SUPPORTED["kv_heads_mode"]`. The
`tag_kv_heads` tagger and `validate()` routing already exist; the kernel change
is reader-side only — work unit `(b, h_q, i)` must read K/V from head
`h_kv = h_q // (H_q / H_kv)` instead of `h_q`. No new compute, no new CB. Pass
the KV-head count (or the group ratio) as a reader CT/RT arg and remap the
`kv_base` head index. Output and Q indexing stay on H_q.

**Verifier notes**: high value (modern LLMs ship GQA/MQA), low risk — a single
index remap in the reader. Independent of the other refinements; can land in
any order after Refinement 1 (so the new dtype set is already handled before
GQA shapes exercise it). Unblocks the 4 `test_gqa_mqa_forward` regression
canaries (currently failing `other` = validate() rejection) in addition to the
342 registry cells carrying `kv_heads_mode ∈ {gqa, mqa}`. The reference
(`helpers.pytorch_scaled_dot_product_attention`) already broadcasts K/V via
`repeat_interleave`, so correctness is directly comparable.

**Done when**: every registry cell with `kv_heads_mode ∈ {gqa, mqa}` (and
`alignment=tile_aligned`, `dtype` per current SUPPORTED) passes, and the
`test_gqa_mqa_forward` canaries pass.

---

### [ ] Refinement 3 — Causal masking (on-device triangular bias)

**Goal**: add `"causal"` to `SUPPORTED["mask_mode"]`. Generate the triangular
−inf bias **on-device** from `is_causal` (never a caller tensor, never a
materialized full S_q × S_kv mask), added to each score block before the
running max (the Phase E slot). Exploit the three-region structure per
(Q-block i, KV-block j): past blocks unmasked; future blocks whole-tile −inf and
**skipped outright** (no QKᵀ/softmax/PV — the ≈½-KV-work causal speedup); the
single diagonal-straddling block gets a per-element triangular mask generated
on-device. Add `EXCLUSIONS += [{"mask_mode": "causal", "attention_kind": "cross"}]`
(causal requires S_q == S_kv) raising the support-refusal. (`validate()` already
raises `ValueError` when `is_causal and attn_mask is not None`.)

**Verifier notes**: algorithm-fundamental (new on-device mask-generation path +
block-skipping control flow) — stands alone, no current skill covers on-device
causal mask generation. See `op_design.md` "Causal refinement contract" for the
binding requirements. Covers `(mask_mode, causal)` — 372 xfail_expected cells
carry it. Order after Refinement 1 so the mask path is written once against the
final dtype/CB-format machinery. Independent of Refinement 2 (causal is
typically MHA decoder self-attn, but the head-broadcast remap composes cleanly).

**Done when**: registry cells with `mask_mode=causal, attention_kind=self`
(current dtype/kv_heads SUPPORTED) pass; `attention_kind=cross` causal cells are
EXCLUDED (raise the support-refusal, observed as xfail_expected).

---

### [ ] Refinement 4 — Non-tile-aligned sequence / head dim

**Goal**: add `"h_non_aligned"` (S_q not a multiple of 32) and `"w_non_aligned"`
(D not a multiple of 32) to `SUPPORTED["alignment"]`, natively in-kernel. The
last tile along the partial dimension must be zero-padded / masked in the reader
(for the partial K/V or partial-D edge) and the partial rows excluded from the
softmax reduction so they contribute neither to the running max nor the running
sum. Math always stays on full tiles; the masking happens at the data-access
boundary.

**Implementation skill**: /memory-layouts

**Verifier notes**: the `/memory-layouts` "non-aligned rule" (last-tile H/W
zero-pad / mask in reader or compute) is the relevant pattern, even though the
`layout` axis itself has no gap (TILE-only). Land last: it is the trickiest
correctness work (partial-tile masking that must not pollute the online-softmax
reductions) and benefits from the dtype machinery (R1) and any causal
edge-masking infrastructure (R3) being settled first. Covers
`(alignment, h_non_aligned)` and `(alignment, w_non_aligned)` — 180
xfail_expected cells. When this lands together with R1, add the bf8b ×
non-aligned EXCLUSION named in Refinement 1.

**Done when**: registry cells with `alignment ∈ {h_non_aligned, w_non_aligned}`
(current dtype/kv_heads/mask SUPPORTED) pass, minus any documented EXCLUSION.
