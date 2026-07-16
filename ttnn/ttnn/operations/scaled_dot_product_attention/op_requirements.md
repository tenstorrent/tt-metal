# Operation Requirements: scaled_dot_product_attention

## Definition
- **Formula**: `O = softmax(Q·Kᵀ·scale + mask) · V`, computed via FlashAttention-2
  (tiled online-softmax over KV blocks; the `S_q × S_kv` score matrix is never
  materialized). `scale` defaults to `1/sqrt(D)`.
- **PyTorch Reference**:
  ```python
  def sdpa(q, k, v, attn_mask=None, scale=None):
      B, H, Sq, D = q.shape
      Hkv = k.shape[1]
      if scale is None:
          scale = 1.0 / math.sqrt(D)
      if Hkv != H:                                   # GQA / MQA
          rep = H // Hkv
          k = k.repeat_interleave(rep, dim=1)
          v = v.repeat_interleave(rep, dim=1)
      scores = torch.matmul(q.float(), k.float().transpose(-2, -1)) * scale
      if attn_mask is not None:
          scores = scores + attn_mask.float()
      return torch.matmul(torch.softmax(scores, dim=-1), v.float())
  ```
- **Import Path**: `from ttnn.operations.scaled_dot_product_attention import scaled_dot_product_attention`
- **Function Signature**:
  ```python
  scaled_dot_product_attention(
      query: ttnn.Tensor,                          # (B, H, S_q, D)
      key: ttnn.Tensor,                            # (B, H_kv, S_kv, D)
      value: ttnn.Tensor,                          # (B, H_kv, S_kv, D)
      *,
      attn_mask: ttnn.Tensor | None = None,        # (B, {1,H}, S_q, S_kv) additive
      is_causal: bool = False,                     # mutually exclusive with attn_mask
      scale: float | None = None,                  # None → 1/sqrt(D)
      compute_kernel_config: ttnn.ComputeConfigDescriptor | None = None,
      memory_config: ttnn.MemoryConfig | None = None,
  ) -> ttnn.Tensor                                 # (B, H, S_q, D)
  ```

## Phases

> **Non-regression rule**: Every refinement must pass all tests from prior phases.
> **Drift signal**: XPASS-strict failures mean the implementer added support but forgot to update SUPPORTED. The implementer fixes by updating SUPPORTED.
> **Checkbox protocol**: Implementer marks `[x]` when the refinement is complete and all tests pass, `[~]` when real work landed but at least one named axis value is deferred (treated as completed by the queue, surfaced as partial), `[ ]` only when nothing usable was produced.
> **Refinement ID + follow-up naming (mandatory — the runner parses this)**: Primary refinements are `Refinement N`. A `[~]` partial's sharper follow-up appends a lowercase letter to the parent's number (`Refinement 1b`, `Refinement 1c`, …), ordered immediately after its parent. The runner's parser matches exactly `Refinement \d+[a-z]?`.

> **TARGET − SUPPORTED gap covered by this queue** (everything else in TARGET is
> already SUPPORTED and green):
> - `dtype`: float32, bfloat8_b → **R2**
> - `fp32_dest_acc_en`: False → **R2**
> - `alignment`: w_non_aligned, h_non_aligned → **R1**
> - `mask_mode`: causal → **R4**
>
> No `memory_layout`/sharding axis exists in the golden TARGET, so flash-decode
> (cross-core `S_kv` split) and GQA KV-multicast are **not** queued — they have no
> golden cell to unlock (see verification_report.md "Observations").

### [x] Phase 0 — Core Implementation

- **SUPPORTED dtype**: [bfloat16]
- **SUPPORTED fp32_dest_acc_en**: [True]
- **SUPPORTED layout**: [TILE]
- **SUPPORTED alignment**: [tile_aligned]
- **SUPPORTED attention_kind**: [self, cross]
- **SUPPORTED kv_heads_mode**: [mha, gqa, mqa]
- **SUPPORTED mask_mode**: [none, custom]
- **SUPPORTED scale_mode**: [auto, explicit]
- **EXCLUSIONS**: none
- **Cores**: multi-core (`split_work_to_cores` over `B·H·n_q_chunks`, interleaved DRAM)
- **Blocking**: `Sq_chunk_t`/`Skv_chunk_t`/`KV_DEPTH` fitted once by `_fit_l1`
- **Compute config**: HiFi4 + fp32 DEST (default), knobs threaded from `compute_kernel_config`
- **Golden baseline**: **212 / 212** supported cells passing (verifier CLI: supported_fail=0, xpass_drift=0, xfail_wrong_mode=0)

---

### [ ] Refinement 1 — Non-tile-aligned shapes (w_non_aligned + h_non_aligned)

**Goal**: add `"w_non_aligned"` and `"h_non_aligned"` to `SUPPORTED["alignment"]`,
handled **natively in the kernel** (no `ttnn.tilize`/`to_layout` wrapper). Covers
both non-aligned legs in one refinement (they share the reader/compute edge
machinery):
- **`w_non_aligned` (D % 32 ≠ 0)**: zero-pad the last D-tile in the reader so the
  Q·Kᵀ contraction (over `Dt`) and the P·V free-dim are correct; drop the padded
  output columns of the last D-tile in the writer.
- **`h_non_aligned` (S_q or S_kv % 32 ≠ 0)**: the last Q-chunk writes only valid
  rows; **the last KV tile's padding rows must be masked to −∞ before the
  row-max/exp/sum** so they fall out of the softmax denominator — use the reduce
  helper's partial-scaler / mask path (`ReducePartialScaler::last_tile_at` +
  `calculate_and_prepare_partial_reduce_scalers` / `prepare_reduce_mask` on the
  reader). This is the structural part — attention's softmax reduction spans the
  padded KV axis, so a plain edge-pad is not enough.

**Implementation skill**: /memory-layouts

**Verifier notes**: **hardest generality refinement — land it first, on the
smallest test surface** (before R2 adds float32/bf8b, so the edge machinery is
validated on bf16 alone). While here, replace the descriptor's `_chunk_size`
largest-divisor trick with a coarse chunk (`min(axis_t, 4)`) + a masked partial
remainder (the divisor trick avoids partial chunks today but would collapse to a
1-tile chunk for a prime tile-count > 4 — see verification_report.md). The skill
body owns the last-tile zero-pad/mask pattern; do not re-derive it here.
`bfloat8_b + non_tile_aligned` is expected to fail and becomes an EXCLUSION armed
by R2, not by this refinement.

**Done when**: the `alignment ∈ {w_non_aligned, h_non_aligned}` golden cells
(currently `xfail_expected`) pass; the mask-reduction correctly excludes KV
padding (verify on `test_regression`-style partial-S_kv shapes); prior phases green.

### [ ] Refinement 2 — Numerical configurability (dtype + compute-config + intermediate precision)

**Goal**: add `ttnn.float32` and `ttnn.bfloat8_b` to `SUPPORTED["dtype"]`, add
`False` to `SUPPORTED["fp32_dest_acc_en"]`, expose the caller's
`compute_kernel_config` (`math_fidelity`, `fp32_dest_acc_en`, `math_approx_mode`)
end-to-end, and correct intermediate-CB precision (`cb_scores`/`cb_exp` fp32 where
it pays, `UnpackToDestFp32` tagging). Cells that fail out of the box land in
`EXCLUSIONS`, **not** their own refinement — arm at minimum:
- `{"dtype": ttnn.float32, "fp32_dest_acc_en": False}` — maxed input + non-maxed
  accumulation is lossy; refuse it (honor the caller's flag, don't silently force
  True). The golden `TOLERANCES` omits this combo for the same reason.
- `{"dtype": ttnn.bfloat8_b, "alignment": "w_non_aligned"}` (and `h_non_aligned`)
  if bf8b + non-aligned misses tolerance — the canonical `/numeric-formats-metal`
  EXCLUSION. (Depends on R1 having added the non-aligned values first.)

**Implementation skill**: /numeric-formats-metal

**Verifier notes**: **pulled ahead of the (harder) causal refinement (R4)** because
the perf-flagged loose case `1×10×9472×128` pins `fp32_dest_acc_en=False` — that
shape is `xfail` until this refinement lands it in SUPPORTED, and **R3 (perf)
cannot run against it until then**. Also fixes/relaxes the 9 `test_regression.py`
precision failures (float32 dtype + fp32 intermediates push the adversarial
distributions back under tolerance — see verification_report.md; they are genuine
bf16 precision, ruled not-a-bug). Order after R1 so the bf8b + non-aligned
EXCLUSIONS have their axis values to reference. Pass condition per the skill: zero
kernel changes when helpers/descriptor precision are wired correctly.

### [ ] Refinement 3 — Speed up the perf-flagged profile (data-movement)

**Type**: perf

**Goal**: `feature_spec.LOOSE_CASES` flags `1×10×9472×128` (bf16, MHA, self-attn,
`fp32_dest_acc_en=False`, HiFi2) as the mandatory perf target, with
`expected_math_util = 0.35` (≈2.16 ms kernel time) as the goal and a **soft**
`pcc_threshold = 0.997` as the sole gate. Optimize **this** shape. The dominant
lever is data movement: the reader currently issues **one async read + one barrier
+ one push per tile** (the `double_buffer` anti-pattern) even though the KV CBs are
double-buffered, leaving the NoC latency-bound. Apply the relevant patterns from
`ttnn/ttnn/operations/examples/master.md` — `double_buffer` (batch a block of
async reads then one barrier; the CBs are already `KV_DEPTH`-deep) and
`reader_placement` (the split already uses `row_wise=True` — confirm it's optimal
here) look applicable. No SUPPORTED change.

**Verifier notes**: **depends on R2** — the flagged shape's `fp32_dest_acc_en=False`
must be in SUPPORTED (added by R2) before this shape can run at all (it is `xfail`
today). This is the reason R2 precedes R4 in the queue. Headroom is confirmed
qualitatively: the largest measured supported cell (`1×8×4096×128`, 3.47 ms /
110 cores) extrapolates to well above the 2.16 ms target for the ~5× larger
flagged shape, and the shape is data-movement-bound. `/perf-roofline-dm` can
confirm before investing.

**Done when**: measured device-ns improves on the flagged shape (moving
`expected_math_util` toward 0.35), its soft `pcc_threshold=0.997` still holds, the
golden suite is green, and no regression across the config-spanning guard set (one
representative per distinct kernel path × layout × placement — here: mask
none/custom × a small and a large shape, DRAM and L1 output).

### [ ] Refinement 4 — Causal masking (mask_mode = causal)

**Goal**: add `"causal"` to `SUPPORTED["mask_mode"]`, generating the triangular
bias **on-device** (no mask tensor) driven by the `is_causal` compile-time flag.
Two parts, both reachable from phase-0's per-chunk KV loop:
1. **Block-skip** whole future KV chunks (`j·Skv_chunk_t > qc_end` ⇒ the chunk is
   fully masked) — roughly halves the KV work for causal self-attention.
2. **Per-element diagonal mask** on the single straddling KV chunk (the additive
   `−∞` upper-triangular bias), applied to the scaled scores *before* the row-max —
   reuse the existing `has_mask` additive-mask compute path (phase 3), fed by an
   on-device-generated mask instead of a streamed tensor.

Arm `EXCLUSIONS += {"mask_mode": "causal", "attention_kind": "cross"}` (causal
requires `S_q == S_kv`; the rectangular case corresponds to no real workload). The
`is_causal ∧ attn_mask` ValueError is already in `validate()`.

**Verifier notes**: standalone scheme-change (design Lamp 1) — no inventory skill
covers on-device causal masking, so this is verifier-authored. Land **after** R2
so the causal path is validated against the full dtype set already in SUPPORTED
(bf16 first is smallest, but R2's dependency pull-up forces R2 before this; the
causal path is dtype-agnostic and inherits R2's cells). Reuses R1's KV-edge masking
for the straddling-chunk diagonal when `S_q`/`S_kv` are non-aligned. Reference the
design's "Lamps → 1. Causal masking" section and
`tech_reports/FlashAttention/FlashAttention.md` (causal load-balancing).

**Done when**: `mask_mode=causal` self-attention golden cells pass; `causal + cross`
is xfail via the new EXCLUSION; block-skip verified to reduce device-ns on causal
vs an equivalent full-mask custom run; prior phases green.

### [ ] Refinement 5 — Speed up the perf-flagged profile (compute-side)

**Type**: perf

**Goal**: continue optimizing the flagged `1×10×9472×128` profile toward
`expected_math_util = 0.35`, this time with **compute-side** levers distinct from
R3's data-movement work. From `ttnn/ttnn/operations/examples/master.md`:
`matmul_output_subblock` (grow the Q·Kᵀ and P·V output subblocks toward the DEST
budget — with `fp32_dest_acc_en=False` the budget is the full 8 bf16 tiles, so the
`_matmul_subblocks` cap can widen), `compute_block_size` (co-tune the
`Sq_chunk_t`/`Skv_chunk_t` block factors — coarser amortizes per-phase
reconfig/init over more tiles, whole-tile-minimum floor) and its reconfig-ablation
second lever (drop the per-phase data-format reconfig where the format is constant
across the boundary). These are knob-tunes on the block surface `_fit_l1` already
exposes. No SUPPORTED change.

**Verifier notes**: generality is exhausted after R4, so the remaining phases are
all perf (still on the flagged shape — it is the sole flagged case, so R5 uses a
different lever class than R3). Co-tune block size against L1 (the `_fit_l1` budget)
and DEST budget together; the `fp32_dest_acc_en=False` regime (flagged) doubles the
DEST budget vs the phase-0 default, which is exactly what unlocks a wider matmul
subblock. Gate on measurement — if `/perf-roofline-dm` shows the shape is already
data-movement-saturated after R3, fold any residual compute win into R3's guard-set
re-measure instead of shipping a no-op phase.

**Done when**: measured device-ns improves further on the flagged shape beyond R3
(toward the 0.35 util goal), soft `pcc_threshold=0.997` holds, golden green, and no
regression across the config-spanning guard set.
