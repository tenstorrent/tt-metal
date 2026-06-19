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

### [~] Refinement 1 — Numerical configurability (float32 + bfloat8_b + compute_kernel_config)

> **R1 outcome (partial)**: `bfloat8_b` fully landed (incl. bf8b+causal — the
> block-float additive −inf survives, PCC 0.9999, no EXCLUSIONS needed).
> `float32` landed for all tile-aligned/MHA cells **except two narrow corners**:
> (1) `float32 + head_dim=1024` (`Q1x1x128x1024`, 4 cells) hit a **hard L1 OOM** —
> the fp32 input CBs double the footprint (cb_k/cb_v ≈ 1 MB each at Dt=32),
> static CBs grow to 2,767,616 B > 1,572,864 B. **Lever proven**: temporarily
> capping `kv_chunk_t=1` shrinks cb_k/cb_v and all 4 cases run+pass — deferred to
> the memory-budget follow-up (R4 below) rather than bolting a fragile hardcoded-
> L1 heuristic into this numeric-formats refinement. (2) `float32 + S=8192 +
> no-mask` (`Q1x1x8192x64`, 2 cells) miss the tight fp32 rms target (rms=0.0206 >
> 0.02, PCC=0.9999) — the known bf16-CB softmax-accumulator limitation (Issue
> #13364, fp32 CBs hang), explicitly out of R1 scope per the verifier note and
> tracked in `verification_report.md` (lever-less, not a refinement). Causal +
> S=8192 fp32 passes (masking halves the accumulation depth). Golden: **414/420**
> supported cells pass (Phase-0 140 bf16 all green; 324 correctly xfail toward R2/R3).


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

### [~] Refinement 2 — GQA / MQA (kv_heads_mode)

> **R2 outcome (partial)**: `gqa` and `mqa` both **fully landed** in
> `SUPPORTED["kv_heads_mode"]` — the reader's existing Q-head→KV-head remapping
> is bit-exact correct across every supported dtype. Golden: **203 / 204**
> tile-aligned gqa/mqa cells pass (the 24 non-aligned gqa/mqa cells stay xfail
> toward R3; total 228 gqa/mqa cells). `test_gqa_mqa_forward` (4 cases, were
> `UnsupportedAxisValue` rejections) now passes; all 24 acceptance + 50 new
> R2-unit cases pass; the 140 Phase-0 bf16 golden cells stay green (MHA
> `validate()` is byte-identical — widening SUPPORTED cannot regress MHA).
> **One red corner, NOT silenced in EXCLUSIONS** (precision-near-miss protocol):
> `float32 + Q1x8x4096x128 (GQA) + no-mask + explicit-scale` — rms=0.0222 > 0.02
> fp32 target, PCC=0.9999. Bounded precisely: the *same* shape passes at
> bf16 (4/4), bf8b (4/4), fp32+causal (2/2), and fp32+none+auto — only
> fp32+none+explicit tips over the tight 0.02 target. This is **not** a
> head-remapping gap; it is the identical **bf16-CB softmax-accumulator
> limitation (Issue #13364)** that R1 already declared lever-less for the MHA
> equivalent (`Q1x1x8192x64` fp32 no-mask, rms=0.0206), now surfacing on one
> GQA long-context shape. Lever-less for the default-config golden (fp32 CBs
> hang; math_fidelity is accumulation-dominated; default must stay HiFi2).
> Tracked in the bottom "Not a refinement" note below, **not** filed as a
> standalone refinement (no in-scope lever — same rationale as R1/R4).

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

---

### [ ] Refinement 4 — L1 budget for large head_dim (footprint-aware kv_chunk_t)

**Goal**: make the **6** currently-failing `float32 + head_dim=1024` golden cells
(`Q1x1x128x1024`, mask ∈ {none, causal} × scale ∈ {auto, explicit}) run without an
L1 over-allocation, by sizing the KV streaming CBs to the per-core L1 budget instead
of an unconditional `kv_chunk_t = largest_divisor_leq(Skv_t, DEST_LIMIT)`. Spun out
of R1 — the fp32 dtype doubled the input-CB footprint (cb_k/cb_v ≈ 1 MB each at
Dt=32), pushing static CBs to 2,767,616 B > the 1,572,864 B L1 ceiling.

**Implementation skill**: /memory-budget-metal

**Verifier notes / proven lever** (from the R1 run):
- The failure is a clean allocator throw (`program.cpp:1487`,
  "Statically allocated circular buffers grow to 2767616 B beyond max L1 size of
  1572864 B"), reproducible on `Q1x1x128x1024 dtype=FLOAT32`. Not a kernel bug — a
  resource bound from R1's fp32 input CBs.
- **Lever already proven in R1**: temporarily setting `KV_CHUNK_MAX = 1` (so
  `kv_chunk_t = 1`) makes **all 4** D=1024 fp32 cells run and pass (golden, both
  mask/scale combos). kv_chunk_t reduction shrinks `cb_k`/`cb_v` (= `2·kv_chunk_t·Dt`
  tiles) and `cb_scores`/`cb_p`/`cb_mask` proportionally; the compute kernel already
  accepts any `kv_chunk_t` via compile-time args, so no kernel change is needed.
- **Do NOT** apply a global `kv_chunk_t=1` (it serializes KV streaming for every
  shape). The fix is a **footprint-aware** reduction in the program descriptor:
  estimate the static CB total for the current `kv_chunk_t`; while it exceeds the
  per-core L1 budget (minus a page-alignment margin — the allocator added ~4% over
  the naive byte sum in R1), step `kv_chunk_t` down to the next smaller divisor of
  `Skv_t` (floor 1). bf16 `Q1x1x128x1024` stays at `kv_chunk_t=4` (its ~1.43 MB
  footprint fits), so existing cases do not regress. Use the skill's L1-budget query
  rather than a hardcoded Blackhole constant.
- Also re-derive `osw_qk`/`in1sb_qk` from the reduced `kv_chunk_t` (they already are
  — `osw_qk = largest_divisor_leq(kv_chunk_t, DEST_LIMIT)`).

**Done when**: the 6 `float32 + Q1x1x128x1024` golden cells pass; the 414 R1-passing
golden cells and 24 acceptance tests stay green; no new OOM on any supported shape.

> **Not a refinement — tracked separately**: the bf16-CB softmax-accumulator
> precision limitation (running `l`/`O` round-tripped through bf16 across the KV
> chunks; fp32 CBs hang this LLK — Issue #13364) surfaces on every `float32 +
> long-context + no-mask` shape whose accumulation depth pushes rms just over the
> tight fp32 0.02 target. Known data points:
> - R1 (MHA): `float32 + Q1x1x8192x64 + no-mask` (2 cells), rms=0.0206, PCC=0.9999.
> - R2 (GQA): `float32 + Q1x8x4096x128 + no-mask + explicit-scale` (1 cell),
>   rms=0.0222, PCC=0.9999. (Same shape passes at bf16/bf8b/fp32-causal/
>   fp32-none-auto — only the explicit-scale variant of the no-mask fp32 case
>   tips over; explicit 0.125 > auto 0.0884 ⇒ sharper scores ⇒ marginally more
>   accumulation error.)
>
> No in-scope lever (math_fidelity is accumulation-dominated here, not
> matmul-dominated; the default must stay HiFi2; the golden harness uses the
> default config so the R1 `compute_kernel_config` surface can't help these
> cells). Documented in `verification_report.md` and R1/R2 `changelog.md`.
> Revisit when Issue #13364 is resolved, not as a standalone refinement.
