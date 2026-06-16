# Operation Requirements: scaled_dot_product_attention

## Definition
- **Formula**: `O[b,h,i,:] = Σ_j softmax_j((Q[b,h,i,:]·K[b,h_kv,j,:])·scale + mask[i,j]) · V[b,h_kv,j,:]`
  computed via the FlashAttention online-softmax recurrence (running max / sum /
  output per query row; the S_q×S_kv score matrix is never materialized).
- **PyTorch Reference**:
  ```python
  def reference_sdpa(Q, K, V, *, attn_mask=None, is_causal=False, scale=None):
      Qf, Kf, Vf = Q.float(), K.float(), V.float()
      if Kf.shape[1] != Qf.shape[1]:           # GQA/MQA head broadcast
          r = Qf.shape[1] // Kf.shape[1]
          Kf = Kf.repeat_interleave(r, dim=1); Vf = Vf.repeat_interleave(r, dim=1)
      s = scale if scale is not None else 1.0 / math.sqrt(Qf.shape[-1])
      scores = torch.matmul(Qf, Kf.transpose(-2, -1)) * s
      if is_causal:
          scores = scores + triu(-inf)         # S_q==S_kv
      elif attn_mask is not None:
          scores = scores + attn_mask.float()
      return torch.matmul(torch.softmax(scores, dim=-1), Vf).to(Q.dtype)
  ```
- **Import Path**: `from ttnn.operations.scaled_dot_product_attention import scaled_dot_product_attention`
- **Function Signature**:
  ```python
  scaled_dot_product_attention(
      query: ttnn.Tensor,                       # (B, H_q, S_q, D)
      key: ttnn.Tensor,                         # (B, H_kv, S_kv, D)
      value: ttnn.Tensor,                       # (B, H_kv, S_kv, D)
      *,
      attn_mask: ttnn.Tensor = None,            # ({1|B}, {1|H_q}, S_q, S_kv) additive
      is_causal: bool = False,
      scale: float = None,                      # None -> 1/sqrt(D)
      compute_kernel_config: ttnn.ComputeKernelConfig = None,
      memory_config: ttnn.MemoryConfig = None,
  ) -> ttnn.Tensor                              # (B, H_q, S_q, D)
  ```

## Phases

> **Non-regression rule**: Every refinement must pass all tests from prior phases.
> **Drift signal**: XPASS-strict failures mean the implementer added support but forgot to update SUPPORTED. The implementer fixes by updating SUPPORTED.
> **Checkbox protocol**: Implementer marks `[x]` when the refinement is complete and all tests pass, `[~]` when real work landed but at least one named axis value is deferred (treated as completed by the queue, surfaced as partial), `[ ]` only when nothing usable was produced.

### [x] Phase 0 — Core Implementation

- **SUPPORTED dtype**: [bfloat16]
- **SUPPORTED fp32_dest_acc_en**: [True, False]
- **SUPPORTED layout**: [TILE] (TILE-only by design — no ROW_MAJOR in TARGET)
- **SUPPORTED alignment**: tile_aligned only
- **SUPPORTED attention_kind**: [self, cross]
- **SUPPORTED kv_heads_mode**: [mha]
- **SUPPORTED mask_mode**: [none, custom] (custom = caller additive mask,
  now incl. batch/head broadcast)
- **SUPPORTED scale_mode**: [auto, explicit]
- **Cores**: multi-core, embarrassingly parallel (one `(b, h, q-chunk)` per
  work-unit, `split_work_to_cores`); no inter-core communication.
- **Compute config**: `compute_kernel_config` honored; default HiFi2 +
  fp32_dest_acc_en=True when None.
- **Golden baseline**: 328 supported_pass; 20 supported_fail (all
  numerical-precision long-context, owned by R1); xpass_drift=0,
  xfail_wrong_mode=0.

### [~] Refinement 1 — Numerical configurability + fp32 accumulator precision

**Goal**: add `ttnn.float32` and `ttnn.bfloat8_b` to `SUPPORTED["dtype"]`,
derive per-CB data formats from input dtype (input CBs, score CB, mask CB,
output CB), and — the load-bearing change — store the online-softmax running
statistics (`cb_o`, `cb_l`, `cb_m`, and the per-iteration `cb_pv`/`cb_o_resc`
accumulation) in **fp32** so the recurrence stops re-rounding to bf16 every
KV-chunk. This both opens the dtype axis and moves the 20 long-context
`numerical-precision` `supported_fail` cells to passing. Arm EXCLUSION
`{"dtype": ttnn.float32, "fp32_dest_acc_en": False}` (maxed input + non-maxed
DEST acc is rejected). Cells that fail out of the box
(`bfloat8_b + non_tile_aligned_dim`, once R3 lands) go to `EXCLUSIONS`, not a
separate refinement.

**Implementation skill**: /numeric-formats-metal

**Verifier notes**: land **first**. (a) The fp32-accumulator change is what
clears the existing `supported_fail` set — it is not optional dtype polish.
(b) R3 (non-aligned) and the bf8b path both reuse the dtype-driven CB-format
derivation introduced here, so this must precede R3. (c) `compute_kernel_config`
is already plumbed through to the `ComputeConfigDescriptor`
(`math_fidelity` / `fp32_dest_acc_en`); the remaining work is CB formats +
accumulator precision, not config exposure. (d) The intermediate CBs are
currently hard-coded `ttnn.bfloat16` in the program descriptor `cb()` helper —
that is the seam to open.

**Done when**: every Phase-0 `supported_fail` cell (the S∈{2048,4096,8192}
long-context bf16 cells) passes, and `SUPPORTED["dtype"]` contains
`float32` + `bfloat8_b` with the float32+acc=False EXCLUSION armed.

### [x] Refinement 2 — Attention variants: causal masking + GQA / MQA

**Goal**: two attention-semantic additions that share the validate / reader
surface:
- add `"causal"` to `SUPPORTED["mask_mode"]` — generate the triangular −inf
  bias **on-device** from `is_causal` (no caller tensor, no materialized full
  mask): skip fully-future KV-chunks, pass fully-past KV-chunks unmasked, and
  apply a per-element triangular −inf only to the diagonal-straddling block.
  Arm EXCLUSION `{"mask_mode": "causal", "attention_kind": "cross"}` (causal
  requires S_q == S_kv) and keep the `is_causal` + `attn_mask` mutual-exclusion
  `ValueError`.
- add `"gqa"` and `"mqa"` to `SUPPORTED["kv_heads_mode"]`. The kernel already
  implements this correctly (reader computes `h_kv = h / group`,
  `group = H_q/H_kv`; verified at PCC 0.99997/0.99996 by temporarily widening
  SUPPORTED) — so this portion is a `validate()` + `SUPPORTED` change with
  **no kernel work**.

**Verifier notes**: no skill in the inventory covers either piece — causal is
an algorithm-specific on-device bias (work from `op_design.md` §"Refinement
contracts → mask_mode=causal"), and GQA/MQA is validate-only. Bundled because
GQA/MQA alone is far too small to be its own refinement, and both touch the
same `validate()` + reader-index region. Independent of R1; can land before or
after, but file after R1 so the causal block path inherits the dtype-aware CB
formats. Unblocks the 4 `test_gqa_mqa_forward` regression failures.

**Done when**: `SUPPORTED["mask_mode"]` contains `causal` and
`SUPPORTED["kv_heads_mode"]` contains `gqa` + `mqa`; the causal+cross EXCLUSION
is armed; the GQA/MQA and causal golden cells move from `xfail_expected` to
`supported_pass` with no `xpass_drift`.

### [x] Refinement 3 — Non-tile-aligned shapes (S_q / S_kv / D)

**Goal**: add `"w_non_aligned"` (D % 32 ≠ 0) and `"h_non_aligned"`
(S_q % 32 ≠ 0, D aligned) to `SUPPORTED["alignment"]` via **in-kernel** edge
handling: zero-pad / mask the last partial tile in the reader (or compute), and
use partial reduce-scalers (`prepare_partial_reduce_scalers` /
`ReducePartialScaler::last_tile_at`) so the padding lanes do not pollute the
softmax row-max / row-sum. Head-dim non-alignment pads `D` up to a tile. Add
EXCLUSION `{"dtype": ttnn.bfloat8_b, "alignment": "w_non_aligned"}` (and
`h_non_aligned` if it fails out of the box) per the /numeric-formats-metal
bf8b+non-aligned rule.

**Implementation skill**: /memory-layouts

**Verifier notes**: order **last**. (a) Depends on R1 — the non-aligned INPUTS
are exercised across all dtypes, so the dtype-aware CB formats and the
bf8b+non-aligned EXCLUSION must exist first. (b) This is the "pad/mask the edge
tile" case the /memory-layouts non-aligned rule covers, **not** a layout-family
change (SDPA stays TILE-only). The masked-softmax-edge interaction (padding
lanes must read as −inf into the row-max, 0 into the row-sum after exp) is the
subtle part — verify against the `47`/`50`/`100`/`33` non-aligned INPUTS.

**Done when**: `SUPPORTED["alignment"]` contains `w_non_aligned` +
`h_non_aligned`; the non-aligned golden cells (e.g. `Q1x1x32x50`,
`Q1x1x47x64`, `Q1x12x33x50`) move from `xfail_expected` to `supported_pass`
(minus any bf8b+non-aligned cells parked in EXCLUSIONS), no `xpass_drift`.

### [x] Refinement 4 — fp32 L1 budget: bound per-core CB footprint for D=1024

**Goal**: fp32 input @ `D=1024` (`d_t=32`) OOMs L1 — the input CBs
(`cb_q_in/k_in/v_in`, sized `2*d_t` pages) plus the fp32 accumulators
(`cb_o/cb_pv/cb_o_resc`, `d_t` pages each) and `cb_out` together exceed the
1.5 MB L1 budget when each tile is fp32 (4 KB). Failing golden cells:
`Q1x1x128x1024_KV1x1x128x1024` × {fp32, acc=True} × {none,custom} ×
{auto,explicit} (4 cells, `RuntimeError: circular buffers grow beyond max L1
size`, program.cpp:1450). bf16 at the same shape fits, so this is fp32-only.

**Implementation skill**: /memory-budget-metal

**Verifier notes**: do NOT EXCLUDE these (OOM is the signal, per the
numeric-formats OOM rule). The fix is the K-blocking / bounded-footprint
pattern: stream the head dim in `d_chunk` tiles instead of sizing
`cb_q_in/cb_out`/accumulators by the full `d_t`, OR drop the input-CB
double-buffer (`2*d_t → d_t`) for fp32. Per-core CB footprint must stop
scaling with `d_t`. Independent of R2/R3.

### [x] Refinement 5 — long-context precision: lift acc=False / fp32 S≥4096 misses

**Goal**: the remaining long-context `supported_fail` cells are precision
near-misses at S∈{4096,8192}, NOT structural gaps (left red by R1, not
EXCLUDED):
- `bf16` + `acc=False` and `bf8b` + `acc=False` @ S≥4096: rel_rms 0.15–0.77
  vs golden 0.12. **Proven** (R1 probe: fp32-CB-always gives identical rms) to
  be the 16-bit DEST register floor — the golden hard-codes
  `fp32_dest_acc_en=False` + HiFi2 for this path, so neither CB format nor
  fidelity is a lever. The only remaining lever is **algorithmic**: reduce the
  KV-chunk count via `Bkv_t = 2/4` blocking (fewer online-softmax rescale
  rounds → less 16-bit-DEST error accumulation), per op_design.md
  "Performance: consider Bq_t/Bkv_t = 2 blocking for long context".
- `fp32` + `acc=True` @ S≥4096: rel_rms 0.028–0.053 vs tight golden 0.02. The
  fp32→TF32 srcA/srcB truncation in the QK^T/PV matmuls is the floor (HiFi4 is
  already max). Lever: Kahan/compensated accumulation of the running output,
  or wider-than-tile reduce blocking — investigate whether the 0.02 fp32 bound
  is reachable at all at S≥4096 or whether the golden tolerance is the limit.

**Verifier notes**: depends on R1. These are `severity=precision` misses with
the PCC/RMS baseline recorded in `changelog.md` / `precision_matrix_results.md`.
Order after R4 (the OOM blocks fp32 from even building at the largest D).
