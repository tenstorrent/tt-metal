# Operation Requirements: scaled_dot_product_attention

## Definition

- **Formula**: `O[b, h, i, :] = sum_j softmax_j(((Q[b,h,i,:] · K[b,h_kv,j,:]) * scale + mask[b,*,i,j])) * V[b,h_kv,j,:]`
- **PyTorch Reference**:
  ```python
  def torch_sdpa(q, k, v, *, attention_mask=None, scale=None):
      qf = q.to(torch.float32)
      kf = k.to(torch.float32)
      vf = v.to(torch.float32)
      # GQA / MQA head broadcast (H_q must be a multiple of H_kv).
      if qf.shape[1] != kf.shape[1]:
          reps = qf.shape[1] // kf.shape[1]
          kf = kf.repeat_interleave(reps, dim=1)
          vf = vf.repeat_interleave(reps, dim=1)
      s = scale if scale is not None else 1.0 / math.sqrt(qf.shape[-1])
      scores = torch.matmul(qf, kf.transpose(-2, -1)) * s
      if attention_mask is not None:
          scores = scores + attention_mask.to(torch.float32)
      attn = torch.softmax(scores, dim=-1)
      return torch.matmul(attn, vf).to(q.dtype)
  ```
- **Import Path**: `from ttnn.operations.scaled_dot_product_attention import scaled_dot_product_attention`
- **Function Signature**:
  ```python
  scaled_dot_product_attention(
      q: ttnn.Tensor,                       # (B, H, S_q,  D)
      k: ttnn.Tensor,                       # (B, H_kv, S_kv, D)
      v: ttnn.Tensor,                       # (B, H_kv, S_kv, D)
      *,
      attention_mask: Optional[ttnn.Tensor] = None,  # (B, 1 | H, S_q, S_kv)
      scale: Optional[float] = None,                 # None → 1 / sqrt(D)
      memory_config: Optional[ttnn.MemoryConfig] = None,
  ) -> ttnn.Tensor                                   # (B, H, S_q, D)
  ```

## Phases

> **Non-regression rule**: every refinement must pass all tests from
> prior phases.
> **Drift signal**: XPASS-strict failures mean the implementer added
> support but forgot to update SUPPORTED. The implementer fixes by
> updating SUPPORTED.
> **Checkbox protocol**: implementer marks `[x]` when the refinement is
> complete and all tests pass, `[~]` when real work landed but at least
> one named axis value is deferred (treated as completed by the queue,
> surfaced as partial), `[ ]` only when nothing usable was produced.

### [x] Phase 0 — Core Implementation

- **SUPPORTED dtype**: `[ttnn.bfloat16]`
- **SUPPORTED layout**: `[ttnn.TILE_LAYOUT]`
- **SUPPORTED alignment**: `[tile_aligned]` (S_q, S_kv, D all multiples of 32)
- **SUPPORTED attention_kind**: `[self, cross]`
- **SUPPORTED kv_heads_mode**: `[mha]` (Q.H == K.H == V.H)
- **SUPPORTED mask_mode**: `[none, causal]`
- **SUPPORTED scale_mode**: `[auto, explicit]`
- **EXCLUSIONS**: `[{"mask_mode": "causal", "attention_kind": "cross"}]`
  — causal mask on a rectangular S_q × S_kv block is well-defined math
  but not a real workload; refinements must consciously enable it.
- **Cores**: single-core (the program descriptor stamps work on (0, 0)).
  Multi-core distribution is design-ready (`split_work_to_cores` is
  documented in op_design.md) but not wired yet.
- **Compute config**: hard-coded `fp32_dest_acc_en=True`, math fidelity
  is the platform default. No `compute_kernel_config` kwarg exposed.
- **Golden baseline**: 128 / 130 supported cells pass. 2 supported_fail
  cells at the long-context boundary (S = 8192 self-attention, bf16,
  mask_mode=none) — both `severity=precision`, `pcc=0.999731`,
  `rms=0.0558` vs target 0.05. Carried into Refinement 1 as the
  precision-floor target.

### [~] Refinement 1 — Numerical configurability + multi-core distribution

**Goal**: extend `SUPPORTED["dtype"]` with `ttnn.float32` and
`ttnn.bfloat8_b` (Phase 0 already has `ttnn.bfloat16`); expose
`compute_kernel_config: Optional[ttnn.ComputeKernelConfig] = None` on
the entry point (covering `math_fidelity`, `fp32_dest_acc_en`,
`math_approx_mode`, `dst_full_sync_en`); set intermediate-CB formats
for the running-state CBs (`cb_max_acc`, `cb_sum_acc`, `cb_cur_mm_out`,
`cb_prev_mm_out`) so the bf16 in-place updates accumulate against an
fp32 reload and the long-context S = 8192 cells move from
`numerical-precision` to passing; tag the unpack/dest pipeline with
`UnpackToDestFp32` where it applies. Distribute the per-query-tile-row
work over the available core grid via `ttnn.split_work_to_cores`
(interleaved DRAM, embarrassingly parallel — no inter-core
communication, the design's per-row pipeline already runs core-local
and the runtime-arg pattern `start_row_id + num_rows` is already in the
kernel). Cells that fail out of the box (typically
`bfloat8_b + non_tile_aligned_dim`) land in `EXCLUSIONS`, not their own
refinement.

**Implementation skill**: /numeric-formats-metal, /interleaved-parallel

**Verifier notes**: must land first — Refinement 2 (KV-broadcast) reuses
the reader's runtime-arg multi-core pattern this refinement introduces,
and Refinement 3 (alignment) reuses the per-dtype CB-format derivation.
The S = 8192 precision lift is the rationale for fp32-ing the running-
state CBs (intermediate-CB format), not just for exposing the kernel
config. If the intermediate-CB lift doesn't fully close S = 8192 (RMS
sits at 0.0558 today, target 0.05), the partial-tick is acceptable: the
gap is narrow, the residual lever is a two-pass output normalization
which isn't covered by the current skill set and would file as its own
refinement.

**Done when**:
- `SUPPORTED["dtype"]` includes `ttnn.bfloat16`, `ttnn.float32`,
  `ttnn.bfloat8_b`.
- Entry-point accepts `compute_kernel_config` and threads it into the
  `ComputeConfigDescriptor`.
- The two S = 8192 `supported_fail` cells flip to `supported_pass` for
  at least the fp32 path; the bf16 path either also flips (preferred)
  or lands a documented partial tick with the gap quantified.
- Multi-core distribution: program descriptor uses
  `ttnn.split_work_to_cores` over `B × H × Qt` and the dispatched grid
  matches `min(total_rows, compute_with_storage_grid_size())`.
- All Phase 0 tests still pass; no new `xpass_drift` or
  `xfail_wrong_mode` cells.

### [~] Refinement 2 — KV-head broadcast (MQA + GQA)

**Goal**: add `"mqa"` and `"gqa"` to `SUPPORTED["kv_heads_mode"]`. The
Phase 0 reader computes the KV-row base index as
`kv_base = (b * H + h) * Kt` where `H` is Q's head count; this assumes
`H_kv == H_q` (MHA only). MQA / GQA need the reader to compute
`h_kv = h_q * H_kv / H_q` (integer divide, valid because
`H_q % H_kv == 0` per the prompt's contract) and address K / V tiles
with `H_kv` in place of `H`. The mask reader stays unchanged
(`MASK_PER_HEAD` is Q-head-indexed). Compute and writer unchanged. New
CT args: pass `H_kv` separately (rename current `H` → `H_q` everywhere
for clarity).

**Verifier notes**: algorithmic-reader rewrite — no skill in the
inventory covers it (not numerical config, not layout, not L1 memory
budget, not embarrassingly-parallel split). The Llama-3 8B/70B
GQA ratio (`H_q=32`, `H_kv=8`) and Whisper-style MQA (`H_kv=1`) are
covered by `feature_spec.INPUTS` — 228 xfail_expected cells will flip
to `supported_pass` when this lands (132 gqa + 96 mqa). Cleanest order
in the queue is for this to follow R1: the multi-core reader changes
in R1 land first, then this refinement edits only the K/V index math
(not the per-core work distribution).

**Done when**:
- `SUPPORTED["kv_heads_mode"]` includes `["mha", "mqa", "gqa"]`.
- Reader passes both `H_q` and `H_kv` as CT args and addresses K/V
  with `H_kv`.
- All 132 + 96 = 228 GQA / MQA cells in
  `eval/golden_tests/scaled_dot_product_attention/test_golden.py` move
  from `xfail_expected` to `supported_pass`.
- `test_regression.py::test_gqa_mqa_forward` cases pass.
- All Phase 0 + Refinement 1 tests still pass.

### [x] Refinement 3 — Non-tile-aligned shapes (W and H)

**Goal**: add `"w_non_aligned"` and `"h_non_aligned"` to
`SUPPORTED["alignment"]`. Phase 0 only handles `D`, `S_q`, `S_kv` all
multiples of 32. For `w_non_aligned` (D not divisible by 32): pad
head_dim on tile read with zero (Q tile pad is benign because the
QK^T inner product picks up `q_real · 0 = 0`, and the V tile pad is
benign because attention weights × 0 = 0 — head_dim is reduction in
the first matmul and broadcast-then-write in the second). For
`h_non_aligned` (S_q and/or S_kv not divisible by 32): pad the
seq-len dim on tile read; in compute, the last partial K tile needs an
additive mask of `-inf` over the padded key positions (otherwise the
softmax includes the padded zeros as real keys with score = scale * 0
= 0, polluting the normalization). The last partial Q tile reads junk
output rows that the writer drops by masking the last-row-write to
the real shape.

**Implementation skill**: /memory-layouts

**Verifier notes**: the `/memory-layouts` skill covers in-kernel
edge-tile padding + last-tile masking. The seq-len side (`h_non_aligned`)
needs the kernel to either (a) inject a synthetic mask covering padded
keys when `S_kv % 32 != 0`, or (b) carry a `valid_keys_in_last_tile`
runtime arg and gate the score in the compute kernel before softmax.
Option (a) keeps compute simple — pre-cook the mask in the reader (or
fuse with the user mask in the K-loop) and let the existing
`HAS_MASK` path handle it. Either way it's the in-kernel path, not a
`ttnn.pad`-then-call wrapper.

The S/D non-alignment must compose with R1's dtype set: `bfloat8_b +
non_tile_aligned_dim` is a well-known incompatible combination
(bf8b's per-tile-exponent scheme requires fully populated tiles); that
specific cell should land in `EXCLUSIONS` rather than blocking R3 (or
land here without it, depending on what's already in `EXCLUSIONS` from
R1).

**Done when**:
- `SUPPORTED["alignment"]` includes `["tile_aligned", "w_non_aligned",
  "h_non_aligned"]`.
- All 60 + 60 = 120 non-aligned cells in
  `eval/golden_tests/scaled_dot_product_attention/test_golden.py` move
  from `xfail_expected` to `supported_pass` (modulo any `bfloat8_b +
  non_tile_aligned` cells deferred to `EXCLUSIONS`).
- Phase 0, Refinement 1, Refinement 2 cells still pass.

### [~] Refinement 4 — fp32 long-context precision via two-pass output normalization

**Goal**: close the precision-near-miss gap on fp32 + S ≥ 4096 cells
(currently failing Refinement 1's fp32 tolerance of PCC=0.999 /
RMS=0.02). Measured at Refinement 1 (extended at Refinement 2 with
the newly-supported MQA + GQA shapes that fall into the same fp32
long-context floor):

| Cell | PCC | RMS | Target | Gap |
|---|---|---|---|---|
| `Q1x1x4096x64 fp32 self auto` | 0.999656 | 0.026267 | (0.999, 0.02) | RMS over by 0.006 |
| `Q1x1x4096x64 fp32 self explicit` | 0.999656 | 0.026267 | (0.999, 0.02) | RMS over by 0.006 |
| `Q1x4x4096x64 fp32 self auto` | 0.999662 | 0.026222 | (0.999, 0.02) | RMS over by 0.006 |
| `Q1x4x4096x64 fp32 self explicit` | 0.999662 | 0.026222 | (0.999, 0.02) | RMS over by 0.006 |
| `Q1x1x8192x64 fp32 self auto` | 0.998610 | 0.053144 | (0.999, 0.02) | PCC + RMS |
| `Q1x1x8192x64 fp32 self explicit` | 0.998610 | 0.053144 | (0.999, 0.02) | PCC + RMS |
| `Q1x4x4096x64_KV1x1x4096x64 fp32 mqa auto` *(new, R2)* | 0.999696 | 0.024784 | (0.999, 0.02) | RMS over by 0.005 |
| `Q1x4x4096x64_KV1x1x4096x64 fp32 mqa explicit` *(new, R2)* | 0.999696 | 0.024784 | (0.999, 0.02) | RMS over by 0.005 |
| `Q1x8x4096x128_KV1x2x4096x128 fp32 gqa auto` *(new, R2)* | 0.999684 | 0.025178 | (0.999, 0.02) | RMS over by 0.005 |
| `Q1x8x4096x128_KV1x2x4096x128 fp32 gqa explicit` *(new, R2)* | 0.999649 | 0.026515 | (0.999, 0.02) | RMS over by 0.007 |

The Refinement 1 intermediate-CB lift (Float32 running-state CBs)
closed the bf16 S=8192 cells but the fp32 cells' tighter tolerance
exposes a remaining floor: each K-iteration's
`cur_mm_out = prev_mm_out * exp_max_diff + partial` correction
accumulates rounding error that's quadratic in the number of K-blocks.

**Verifier notes**: Refinement 1's verifier explicitly named this as
the follow-up: "the residual lever is a two-pass output normalization
which isn't covered by the current skill set and would file as its
own refinement." The two-pass approach computes the running max and
sum across all K-blocks first (as today), then in a second pass
recomputes `exp(score - global_max) @ V / global_sum` — avoiding the
multiplicative cascade of `exp_max_diff` corrections.

The kernel's L1 footprint for the two-pass approach: need to either
(a) re-read K/V from DRAM in pass 2 (cheap), or (b) park attention
weights in L1 (`Kt * Wt` extra tile-slots — expensive at S=8192).
Pass (a) is the cleanest path.

**Done when**:
- All 6 fp32 + S ≥ 4096 cells in `test_golden.py` move from
  `supported_fail` to `supported_pass`.
- Phase 0, Refinement 1, Refinement 2, Refinement 3 cells still pass.

### [x] Refinement 5 — fp32 + large head_dim L1 capacity via K-blocking

**Goal**: enable fp32 + D=1024 cells (currently failing with
`Statically allocated circular buffers grow to 1590624 B which is
beyond max L1 size of 1499136 B`). The Refinement 1 program
descriptor sizes `cb_prev_mm_out` and `cb_cur_mm_out` as
`2 * Dt * sizeof(Float32) = 2 * 32 * 4096 = 262144 B each`, plus the
double-buffered Q/K/V/output CBs (each `2 * Dt * sizeof(Float32) =
262144 B`). On a per-core L1 budget of ~1.5 MB, fp32 + D=1024 exhausts
the budget.

The fix is the `/memory-budget-metal` recipe: K-block the matmul so
the intermediate `mm_out` CBs hold only a chunk of head_dim at a
time, not the full Dt. Per-core CB footprint becomes O(D_chunk),
not O(D).

**Failing cells** (4 in `test_golden.py`):
- `Q1x1x128x1024 fp32 self {auto,explicit} × mask={none,causal}`

bf16 and bf8b paths at D=1024 work fine — they use half the
per-tile bytes, fitting in L1. Only fp32 hits the cap.

**Implementation skill**: /memory-budget-metal

**Done when**:
- All 4 fp32 + D=1024 cells in `test_golden.py` move from
  `supported_fail` to `supported_pass`.
- Phase 0, Refinement 1, Refinement 2, Refinement 3 cells still pass.

### [x] Refinement 6 — S=8192 fp32 precision lift via UnpackToDestFp32 on running-state CBs + SFPU-based final divide

**Goal**: close the residual S=8192 fp32 cells that R4's two-pass design
brought to 0.027 RMS but couldn't push under the 0.02 target. R4 dropped
the cur_mm_out + sum_exp cascades; the residual is the per-K-iter SFPU
unpack of cb_cur_sum_exp / cb_cur_mm_out, which currently routes through
the TF32 SrcA/SrcB path (~10 mantissa bits preserved per read). For
Kt=256 the sqrt(Kt) × TF32-ULP floor is ~0.016 RMS in expectation, and
we land at 0.027 — close to but not under the strict 0.02 fp32 target.

**Failing cells** (2 in `test_golden.py`):
- `Q1x1x8192x64 fp32 self mha none × {auto, explicit}` — RMS=0.0272,
  PCC=0.999631 (target 0.999, 0.02). Severity=precision near-miss.

**Implementation skill**: /numeric-formats-metal (specifically the
UnpackToDestFp32 surface; consult the skill for the safety rules around
tagged CBs and which downstream consumers are compatible).

**Verifier notes**: R4's iter3 already restructured pass 1 to max-only
and pass 2 to direct sum_exp + output accumulation (no correction
cascade in either). The remaining lever is to upgrade the per-K-iter
unpack of the running-state CBs from TF32 (10 mantissa bits) to full
fp32 (24 mantissa bits) — which drops the cumulative cascade RMS by
roughly 3 orders of magnitude in theory. Sketch of the code lever:

1. Tag `cb_cur_sum_exp` and `cb_cur_mm_out` with `UnpackToDestFp32` in
   `scaled_dot_product_attention_program_descriptor.py`. (Both CBs are
   read by `copy_tile` (SFPU) in `update_cur_sum_exp_pass2` and
   `matmul_attn_by_v_accumulate` — those reads will gain the full
   fp32 unpack.)

2. Both CBs are also read by `mul_tiles_bcast_cols` (FPU) in the final
   divide — which corrupts silently under `UnpackToDestFp32` (per the
   R1 op file's safety note). Fix: introduce an UNTAGGED intermediate
   CB (`cb_cur_mm_out_for_divide`, Dt tiles) and copy
   `cb_cur_mm_out` → `cb_cur_mm_out_for_divide` once after pass 2 ends,
   before the final divide. The single TF32 copy adds one ULP of error
   for one tile-set, vs eliminating the per-K-iter cascade's
   `sqrt(Kt) × TF32 ULP` accumulation. Same trick for `cb_cur_sum_exp`
   → `cb_cur_sum_exp_for_divide` (1 tile).

3. Alternatively, replace the FPU `mul_tiles_bcast_cols` with an
   SFPU-based column-broadcast multiply — requires building a
   broadcast helper (column-0 of one tile × all-cols of another). The
   intermediate-CB approach in (2) is the minimal-change path; the
   SFPU final-divide is the cleaner long-term design.

`cb_cur_max` is also FPU-read (sub_tiles_bcast_cols in
`apply_exp_inplace_with_global_max`); tagging it would corrupt
that pass-2 inner-loop sub. Leaving `cb_cur_max` untagged for now is
fine — its per-iter precision floor doesn't show up in the final
output (it's a max operand for exp, not a cascade accumulator).

**Done when**:
- Both `Q1x1x8192x64 fp32 self mha none × {auto, explicit}` cells in
  `test_golden.py` move from `supported_fail` to `supported_pass`.
- All R4-passing cells still pass (RMS ≤ 0.02 fp32 target).
- Phase 0 + Refinement 1 + Refinement 2 + Refinement 3 cells still pass.
