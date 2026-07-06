# Operation Requirements: scaled_dot_product_attention

## Definition
- **Formula**: `output[b,h,i,d] = (Σ_j exp((Q[b,h,i,:]·K[b,h,j,:]*scale + mask[i,j]) - m_i) / l_i) * V[b,h,j,:]` where `m_i` and `l_i` are the running row-max and row-sum maintained by online softmax across KV blocks.
- **PyTorch Reference**: `torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=..., is_causal=..., scale=...)`
- **Import Path**: `from ttnn.operations.scaled_dot_product_attention import scaled_dot_product_attention`
- **Function Signature**: `scaled_dot_product_attention(query: Tensor, key: Tensor, value: Tensor, *, attn_mask: Tensor | None = None, is_causal: bool = False, scale: float | None = None, memory_config: MemoryConfig | None = None, compute_kernel_config: ComputeConfigDescriptor | None = None) -> Tensor`

## Phases

> **Non-regression rule**: Every refinement must pass all tests from prior phases.
> **Drift signal**: XPASS-strict failures mean the implementer added support but forgot to update SUPPORTED. The implementer fixes by updating SUPPORTED.
> **Checkbox protocol**: Implementer marks `[x]` when the refinement is complete and all tests pass, `[~]` when real work landed but at least one named axis value is deferred (treated as completed by the queue, surfaced as partial), `[ ]` only when nothing usable was produced.
> **Refinement ID + follow-up naming (mandatory — the runner parses this)**: Primary refinements are `Refinement N` (e.g. `Refinement 1`, `Refinement 2`). When you ship `[~]` partial and file the sharper follow-up the partial-tick protocol requires, name it by appending a lowercase letter to the parent's number: `Refinement 1b`, `Refinement 1c`, … (never `Refinement 1.5`, `Refinement 1 (follow-up)`, or a fresh number). Order follow-ups immediately after their parent so the queue runs them before later refinements — a partial's remaining-blocker follow-up must be picked next, not leapfrogged. The runner's parser matches exactly `Refinement \d+[a-z]?`; any other shape is invisible to the queue and silently skipped.

### [x] Phase 0 — Core Implementation

- **SUPPORTED dtype**: [bfloat16]
- **SUPPORTED layout**: [TILE_LAYOUT]
- **SUPPORTED alignment**: [tile_aligned] only
- **SUPPORTED mask_mode**: [none, custom]
- **SUPPORTED scale_mode**: [auto, explicit]
- **SUPPORTED attention_kind**: [self, cross]
- **SUPPORTED kv_heads_mode**: [mha]
- **SUPPORTED fp32_dest_acc_en**: [True]
- **Cores**: single (8×8 grid, `split_work_to_cores`)
- **Compute config**: hard-coded HiFi4 + fp32_dest_acc_en=True + math_approx_mode=False
- **Golden baseline**: 76 / 2648 cells passing (per verifier CLI on single-tile shapes; multi-block hang blocks the rest)

### [ ] Refinement 1 — Multi-block kernel fix (CRITICAL BLOCKER)

**Goal**: Fix the compute kernel hang that occurs when processing more than 1 Q block, more than 1 KV block, or D_t > 1 (head dim > 32). The hang is a DST sync deadlock between the unpacker, math, and packer threads when transitioning between matmul and eltwise/reduce operations across loop iterations. This refinement moves all Phase 0 cells currently in the `supported_fail` (hang/timeout) category to passing.

**Verifier notes**: This is the critical blocker — without it, no other refinement can be verified on multi-block shapes. The root cause is likely that the eltwise chain's per-element init (`copy_tile_to_dst_init_short_with_dt`, `binary_op_init_common`, etc.) doesn't fully transition the math engine from matmul MOP to eltwise MOP, causing `mop_sync` to deadlock on subsequent iterations. The fix likely requires calling `binary_op_init_common` or `compute_kernel_hw_startup` before each stage transition (matmul → eltwise → reduce → matmul). The single-block case works because there's only one transition per stage.

**Done when**: Every Phase 0 cell (bf16, TILE, tile_aligned, mha, fp32_dest_acc_en=True, mask_mode ∈ {none, custom}, scale_mode ∈ {auto, explicit}) with S > 32 or D > 32 passes without hanging. Specifically the `supported_fail` cells from the verifier report move to `supported_pass`.

### [ ] Refinement 2 — Numerical configurability expansion

**Goal**: add `ttnn.float32`, `ttnn.bfloat8_b` to `SUPPORTED["dtype"]`, add `False` to `SUPPORTED["fp32_dest_acc_en"]`, and expose `compute_kernel_config: ttnn.ComputeKernelConfig` on the entry point (already done — the parameter exists and is threaded to the program descriptor). Cells that fail out of the box (typically `bfloat8_b + non_tile_aligned_dim`) land in `EXCLUSIONS`, not in their own refinement. Pass condition: zero kernel changes when helpers are wired correctly.

**Implementation skill**: /numeric-formats-metal

**Verifier notes**: Should land after Refinement 1 — the multi-block fix is needed to verify dtypes on non-trivial shapes. The op already accepts `compute_kernel_config` and the `default_compute_kernel_config()` function is the single source of truth. The `fp32_dest_acc_en` axis is already in SUPPORTED. Missing values: dtype={FLOAT32, BFLOAT8_B}, fp32_dest_acc_en={False}. The `{"dtype": ttnn.float32, "fp32_dest_acc_en": False}` EXCLUSION from the design doc should be added when this refinement lands.

### [ ] Refinement 3 — GQA / MQA head broadcasting

**Goal**: add `gqa` and `mqa` to `SUPPORTED["kv_heads_mode"]` by implementing head broadcasting in the reader kernel (replicating K/V tiles for H_kv < H_q) or in the work distribution (assigning the same K/V tile range to multiple Q head cores).

**Verifier notes**: The golden test INPUTS already include GQA (4:1, 8:1, 3:1 ratios) and MQA (H_kv=1) shapes. The reference helper handles head broadcasting via `repeat_interleave`. The op needs to either broadcast K/V in the reader kernel or adjust the work distribution so multiple Q-head cores read from the same K/V tiles. Missing values: kv_heads_mode={gqa, mqa}.

### [ ] Refinement 4 — Causal masking

**Goal**: add `causal` to `SUPPORTED["mask_mode"]` by generating the triangular causal mask on-device (never from a caller tensor or materialized full mask). Three regions per (Q-block, KV-block): blocks entirely in the past are unmasked; blocks entirely in the future are whole-tile -inf and should be skipped; only the block straddling the diagonal needs a per-element triangular mask.

**Verifier notes**: Per the design doc rules, `{"mask_mode": "causal", "attention_kind": "cross"}` should be declared as an EXCLUSION (causal requires S_q == S_kv). Also, `is_causal=True` combined with a non-None `attn_mask` should raise ValueError (already implemented in validate). Missing value: mask_mode={causal}.

### [ ] Refinement 5 — Non-tile-aligned shapes

**Goal**: add `w_non_aligned` and `h_non_aligned` to `SUPPORTED["alignment"]` by handling non-tile-aligned dimensions (D % 32 != 0 or S % 32 != 0) with zero-padding / masking in the reader kernel.

**Verifier notes**: The golden test INPUTS include non-aligned shapes (D=50, S=47, etc.). These currently xfail because the kernel assumes tile-aligned dimensions. Missing values: alignment={w_non_aligned, h_non_aligned}.

### [ ] Refinement 6 — L1 budget fit for large head_dim

**Goal**: rewrite the reduction and streaming phases so the per-core L1 CB footprint is bounded by a constant (chunking on the head_dim and KV dimensions), so the op stops OOMing on the large head_dim shapes in `feature_spec.INPUTS` (D ∈ {512, 1024}). Phase 0 leaves these cells failing with `OOM`; this refinement moves them to passing.

**Implementation skill**: /memory-budget-metal

**Verifier notes**: The OOM occurs because the Q/K/V stream CBs scale as `2 × (B_q × D_t)` pages, and for D_t=32 (D=1024) this exceeds the L1 budget. The `/memory-budget-metal` skill's K-blocking pattern (`num_k_blocks > 1`, weights restreaming) is the natural fit — split the D dimension into K-blocks and restream. This refinement does NOT add a SUPPORTED axis — `shape_size` is not a kernel-level branch, just a resource boundary.
