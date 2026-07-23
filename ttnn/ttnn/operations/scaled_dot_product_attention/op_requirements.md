# Operation Requirements: scaled_dot_product_attention (Flash Attention)

## Definition
- **Formula**: `O = softmax(Q·Kᵀ·scale [+ mask], dim=-1) · V`, per (batch, head), computed
  online over KV-blocks (the S_q×S_kv score matrix is never materialized).
- **PyTorch Reference**:
  ```python
  def reference(Q, K, V, attn_mask=None, is_causal=False, scale=None):
      import torch
      H_q, H_kv = Q.shape[1], K.shape[1]
      if H_q != H_kv:  # GQA / MQA head broadcast
          r = H_q // H_kv
          K = K.repeat_interleave(r, dim=1)
          V = V.repeat_interleave(r, dim=1)
      return torch.nn.functional.scaled_dot_product_attention(
          Q.float(), K.float(), V.float(),
          attn_mask=(attn_mask.float() if attn_mask is not None else None),
          is_causal=is_causal, scale=scale)
  ```
- **Import Path**: `from ttnn.operations.scaled_dot_product_attention import scaled_dot_product_attention`
- **Function Signature**:
  ```python
  scaled_dot_product_attention(
      query: ttnn.Tensor,                       # (B, H_q, S_q, D)
      key: ttnn.Tensor,                         # (B, H_kv, S_kv, D)
      value: ttnn.Tensor,                       # (B, H_kv, S_kv, D)
      *,
      attn_mask: ttnn.Tensor = None,            # (B, 1, S_q, S_kv) or (B, H_q, S_q, S_kv), additive
      attention_mask: ttnn.Tensor = None,       # alias for attn_mask
      is_causal: bool = False,
      scale: float = None,                      # None -> 1/sqrt(D)
      compute_kernel_config: ttnn.ComputeConfigDescriptor = None,
  ) -> ttnn.Tensor                              # (B, H_q, S_q, D), bfloat16, TILE
  ```

## Phases

> **Non-regression rule**: Every refinement must pass all tests from prior phases.
> **Drift signal**: XPASS-strict failures mean the implementer added support but forgot to update SUPPORTED. The implementer fixes by updating SUPPORTED.
> **Checkbox protocol**: Implementer marks `[x]` when the refinement is complete and all tests pass, `[~]` when real work landed but at least one named axis value is deferred (treated as completed by the queue, surfaced as partial), `[ ]` only when nothing usable was produced.
> **Refinement ID + follow-up naming (mandatory — the runner parses this)**: Primary refinements are `Refinement N`. Partial follow-ups append a lowercase letter to the parent's number (`Refinement 1b`, `Refinement 1c`, …), ordered immediately after the parent.

### [x] Phase 0 — Core Implementation

- **SUPPORTED dtype**: [bfloat16]
- **SUPPORTED fp32_dest_acc_en**: [True]
- **SUPPORTED layout**: [TILE]
- **SUPPORTED shape-derived axes**: alignment=tile_aligned; attention_kind ∈ {self, cross}; kv_heads_mode ∈ {mha, gqa, mqa}
- **SUPPORTED op-specific axes**: mask_mode ∈ {none, custom}; scale_mode ∈ {auto, explicit}
- **Cores**: multi-core (flat `B·H_q·q_num_chunks` work-list over the grid, embarrassingly parallel)
- **Compute config**: HiFi2 + fp32_dest_acc_en=True (single source via `default_compute_kernel_config()`)
- **Golden baseline**: 206 supported_pass / 6 supported_fail (all OOM, D∈{512,1024}) / 2113 xfail_expected; xpass_drift=0, xfail_wrong_mode=0 (per `verifier_report.json`)

### [ ] Refinement 1 — Numerical configurability expansion

**Goal**: add `ttnn.float32` and `ttnn.bfloat8_b` to `SUPPORTED["dtype"]` and add
`False` to `SUPPORTED["fp32_dest_acc_en"]`; the entry point already exposes
`compute_kernel_config` (resolved via `default_compute_kernel_config()`). Set the
intermediate-CB formats / `UnpackToDestFp32` tagging per dtype, and make the two
matmuls carry the correct fidelity (bf16 Kt>1 → HiFi2 + fp32-DEST; fp32 → HiFi4 +
fp32-DEST; **never** HiFi4 + fp32-DEST with bf16 inputs, issue #38306). Cells that
fail out of the box go to `EXCLUSIONS`, not their own refinement: **`{dtype: float32,
fp32_dest_acc_en: False}`** is legal-but-lossy and refused (mirrors softmax);
`{dtype: bfloat8_b, fp32_dest_acc_en: False}` is a known-lossy corner (reference
lands PCC ~0.9996) — keep it supported if it clears the golden `(bf8b, False)`
tolerance (0.99 / 0.12), else EXCLUDE.

**Implementation skill**: /numeric-formats-metal

**Verifier notes**: **Lands first — this is the perf-1 contract anchor.** The
perf-flagged loose case `(1,10,9472,128)` runs bf16 @ `fp32_dest_acc_en=False`, which
Phase-0 does not support; this refinement unlocks exactly that config so Refinements 3
and 5 optimize the real target, never a `fp32_dest_acc_en=True` proxy (different
datapath). Also the concrete lever for the adversarial-distribution regression
failures (`test_regression.py` uniform/negative/large-magnitude — genuine bf16 noise,
not a scale bug; see verification_report.md): float32 + fp32 intermediate CBs clear
them. Build it performantly (keep the multi-core grid fill + reader batching from
Phase-0). Adding float32 doubles CB bytes → worsens the D-scaling L1 pressure, so
Refinement 2 must follow immediately.

**Done when**: golden `dtype ∈ {float32, bfloat8_b}` and `fp32_dest_acc_en=False` cells
move from xfail_expected to supported_pass (minus the EXCLUDED corner(s)); zero kernel
changes beyond CB-format / config wiring where helpers are used correctly; the
bf16 @ `fp32_dest_acc_en=False` path passes for the flagged shape's config.

### [ ] Refinement 2 — Per-core L1 budget fit for large head_dim

**Goal**: bound the per-core CB footprint so the D∈{512,1024} shapes stop OOMing.
Phase-0 sizes `cb_q_in`, `cb_k_in`, `cb_v_in`, `cb_out`, and the running-`O`
accumulator CBs linearly in `dht = D/32` (D is never blocked), so per-core L1 grows to
1.58 MB @ D=512 and 2.83 MB @ D=1024 (L1 max 1.57 MB). Move the 6 `supported_fail`
OOM cells (`Q1x1x128x512`, `Q1x1x128x1024`, ×{none,custom}×{auto,explicit}) to
passing by budgeting the chunk knobs against available L1 (shrink `sq_chunk_t` /
`sk_chunk_t` as `dht` grows, and/or block the QKᵀ D-contraction so `cb_k_in` holds
`sk_chunk_t·block_k` instead of `sk_chunk_t·dht`). No SUPPORTED axis is added — this
is a resource boundary, not a kernel-branch axis; `shape_size` bucketing would hide
the gap.

**Implementation skill**: /memory-budget-metal

**Verifier notes**: **Must follow Refinement 1** — R1 adds float32 (2× tile bytes),
which lowers the D at which OOM strikes, so the budget must account for the resolved
dtype's tile size. The Q-block is inherently resident across the whole KV loop
(flash-attention invariant), so the realistic lever is capping `sq_chunk_t`/`sk_chunk_t`
by an L1-budget calc in the program descriptor (a knob-turn on the block factors the
planner already exposed), optionally with D-blocking on QKᵀ (`num_k_blocks = DHt`,
which the design's Blocking Model names but Phase-0 collapsed to `num_k_blocks=1`).
The flagged perf shape (D=128) fits already, so this does not touch the perf path.

**Done when**: every Phase-0 cell currently in the `OOM` category (the 6
`supported_fail`) passes, at every supported dtype; no regression on the D≤256 cells.

### [ ] Refinement 3 — Speed up the perf-flagged profile (K/V reuse multicast)

**Type**: perf

**Goal**: `feature_spec.LOOSE_CASES` flags `(1,10,9472,128)` bf16 @
`fp32_dest_acc_en=False`, mask none, auto scale, self/MHA as the mandatory perf
target (perf goal `expected_math_util ≥ 0.35` against the HiFi2 FPU roofline; soft
gate PCC ≥ 0.997, RMSE 4e-2 record-only). At this shape ~740 Q-blocks spread over
~110 cores, and every core owning a Q-block of a given (batch, head) re-reads the same
~2.4 MB K and ~2.4 MB V from DRAM — the dominant bottleneck. Read each KV-block once on
an injector and NoC-multicast it to the cores sharing that (batch, head) instead of
per-core DRAM pulls, per the relevant pattern in
`ttnn/ttnn/operations/examples/master.md` (`shared_input_reuse`, T3, ~1.71× at 22
cores; this is design lamp #3, the reuse-shared-operand broadcast). No SUPPORTED change.

**Verifier notes**: **Validate by building the mcast variant and measuring device-ns —
do not infer its value from a remove-one-stage ablation** (a broadcast's benefit is
under-shown by such an ablation). Depends on Refinement 1 (the target's
`fp32_dest_acc_en=False` config must be supported first). Uses the `mcast_pipe`
`SenderPipe`/`ReceiverPipe` helper (`dataflow_kernel_lib`) + `ttnn.Mcast2D` host wiring;
double-buffer so the injector prefetches while consumers drain. This is a scheme-change
(new one-injector-plus-broadcast dataflow topology), so it stands alone as a whole
perf phase. `/perf-roofline-dm` first if unsure there is headroom — but the redundant-
read count here makes this the high-confidence lever.

**Done when**: measured device-ns improves on the flagged shape (moving it toward the
0.35 math-util floor), its soft PCC gate (0.997) still holds, the golden suite is
green, and no regression across the config-spanning guard set (one representative per
distinct kernel path × layout × placement).

### [ ] Refinement 4 — Causal masking (mask_mode=causal)

**Goal**: add `"causal"` to `SUPPORTED["mask_mode"]`. When `is_causal=True` the op
generates the triangular −inf bias **on device** (no mask tensor) and truncates the
KV loop — block-skip all strictly-future KV-blocks (≈ half the KV work for a decoder)
and stamp the triangular mask only on the single diagonal-straddling block. Arm the
`{mask_mode: causal, attention_kind: cross}` EXCLUSION (causal requires S_q==S_kv) and
keep the `is_causal + attn_mask` mutual-exclusion ValueError (re-armed now that causal
is reachable).

**Verifier notes**: No inventory skill covers on-device causal-mask generation + KV-loop
truncation — this is a verifier-authored scheme-change (design lamp #1). It adds a
third compile-time mask regime alongside `none`/`custom`; the KV-loop upper bound is
already a parameter, so causal only *truncates* it (no new loop nest). Independent of
the perf path and of Refinements 1–2; ordered here (after the first perf pass) because
it is the last remaining generality axis and it extends the now dtype-aware, budgeted
kernel once rather than being reworked under it. Golden `causal` cells (and the
`{causal, cross}` EXCLUSION → xfail) become the acceptance signal.

**Done when**: golden `mask_mode=causal` self-attention cells move from xfail_expected
to supported_pass; `{causal, cross}` cells are cleanly refused (xfail via ExcludedCell);
`is_causal + attn_mask` raises ValueError; no regression on none/custom.

### [ ] Refinement 5 — Speed up the perf-flagged profile (block-size / buffer-depth co-tune)

**Type**: perf

**Goal**: with the redundant-read bottleneck relieved by Refinement 3, co-tune the
remaining block-surface knobs on the same flagged shape `(1,10,9472,128)` toward the
`expected_math_util ≥ 0.35` goal, using the relevant patterns in
`ttnn/ttnn/operations/examples/master.md`: matmul output-subblock size (`matmul_output_subblock`,
cap at 4 tiles under fp32-DEST), compute block / chunk granularity
(`compute_block_size`, `q_chunk_tiles`/`k_chunk_tiles` — whole tiles minimum, coarser
amortizes the per-helper reconfig/init overhead), CB buffer depth and reader-barrier
batching (`double_buffer` — interleave the K/V/mask read streams behind fewer
barriers). No SUPPORTED change.

**Verifier notes**: These are T2 knob-turns on the block factors the planner exposed —
one phase, since they are cheap and compound. Ordered after Refinement 3 because a
double-buffer / batching knob is a near-no-op while the shape is DRAM-bandwidth-bound
on redundant reads (`double_buffer` gist: "no gain once bandwidth-bound"); relieve the
read count first, then chase compute-side amortization. `/perf-roofline-dm` to confirm
residual headroom before filing effort here.

**Done when**: measured device-ns improves further on the flagged shape with the golden
suite green, its soft PCC gate holds, and no regression across the config-spanning
guard set (one representative per distinct kernel path × layout × placement).
