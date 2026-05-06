# Numerical Stability Analysis: linear (Phase 0)

Single-core 2D matmul plus optional row-broadcast bias on bf16 inputs.

```
output = input @ weight              # no-bias path
output = input @ weight + bias[0,:]  # bias path (row-0 broadcast across M)
```

Shapes: `input [1,1,M,K]`, `weight [1,1,K,N]`, optional `bias [1,1,32,N]`. M, K, N
divisible by 32. `Mt = M/32`, `Kt = K/32`, `Nt = N/32`.

---

## Algorithm Summary

Two compute phases:

1. **Matmul** (FPU-heavy). For each of the `Mt × Nt` output tiles, accumulate the
   inner-product across `Kt` weight tiles, with each tile contributing `32`
   elementwise products that accumulate into a single output element. The
   per-output accumulation depth is therefore `K = Kt × 32` bf16 multiply-adds.

2. **Bias add** (FPU add, only when bias is supplied). Read each matmul output
   sub-block back from `cb_partials`, add the corresponding row-broadcast bias
   tile, pack to `cb_output_tiles`. One add per output element; depth = 1.

Precision-sensitive phases:

- **Matmul K-accumulation** — depth grows linearly with K. This is the dominant
  accumulation in the op.
- **bf16 pack between matmul and bias add** — `cb_partials` is bf16, so the
  fp32 matmul result is rounded to 7-mantissa-bit bf16 before the bias is added.
- **bf16 pack to output** — the final output tensor is bf16; rounding once at
  the end is unavoidable for any bf16 output op.

There is no SFPU work, no division, no exponential, no max-subtraction, and no
catastrophic-cancellation risk in this op. The numerical envelope is dominated
purely by the matmul K-accumulation.

---

## Error Source Inventory

| # | Source | Location | Severity | Mitigation |
|---|--------|----------|----------|-----------|
| 1 | bf16 × bf16 multiply at HiFi4 (full mantissa, 4 passes) | `matmul_block_helpers.inl:269` (`ckernel::matmul_block`) | Low — HiFi4 captures full bf16 mantissa | `math_fidelity=HiFi4` set in descriptor (line 250) |
| 2 | K-accumulation of `K = Kt*32` bf16 products | inner-K loop, `matmul_block_helpers.inl:266-286` | Moderate at large K (>~1k elements) but bounded by fp32 DEST | `fp32_dest_acc_en=True` (descriptor line 251); fully held in DEST per output tile |
| 3 | fp32 → bf16 pack from matmul DEST into `cb_partials` (bias path only) | `matmul_block_helpers.inl:321` (`pack_tile_block`) | Moderate — single round-trip per output sub-block in bias path | `cb_partials` is `output_tensor.dtype` (= bf16) — **NOT FP32**; no `UnpackToDestFp32` configured (descriptor sets neither). Round-to-nearest-even at pack |
| 4 | bf16 → bf16 add (FPU, fidelity-insensitive) for bias broadcast | `bias_add_helpers.inl:143` (`add_tiles_bcast_rows`) | Low — FPU add is exact for representable inputs | None needed |
| 5 | fp32 → bf16 pack from bias DEST into `cb_output_tiles` | `bias_add_helpers.inl:161` | Low — single rounding per output tile | Final-output rounding inherent to bf16 output |
| 6 | fp32 → bf16 pack from matmul DEST directly into `cb_output_tiles` (no-bias path) | `matmul_block_helpers.inl:321` | Low — single rounding per output tile | Final-output rounding inherent to bf16 output |

Bookkeeping observations not present as risks:

- No SFPU calls anywhere on the path (no `exp_tile`, `recip_tile`, `sqrt_tile`, `log_tile`).
- No division — `recip_tile` never invoked, no epsilon guard needed because none is required.
- No `sub_tiles` — no catastrophic cancellation surface.
- No max-subtraction step — none is needed because nothing exponentiates.

---

## Accumulation Analysis

- **What is accumulated**: inner-product `sum_{k=0..K-1} input[m,k] * weight[k,n]` for each output element `(m,n)`.
- **Accumulation depth**: `K = Kt × 32` bf16 multiply-add terms per output element.
- **DEST precision**: **fp32** (`fp32_dest_acc_en=True`, descriptor line 251). Each
  product is computed at HiFi4 (full bf16×bf16 mantissa) then accumulated into
  fp32 DEST. The 23-bit fp32 mantissa absorbs the running sum without
  truncating the small terms that bf16 would otherwise round away.
- **DEST tiles in flight per matmul iteration**: `out_subblock_h * out_subblock_w = 1 * 1 = 1`. fp32 half-sync DEST capacity is 4 tiles (per `dest_helpers.hpp`), so the kernel uses 25% of DEST.
- **K-blocking**: `num_k_blocks = 1` — the full K dimension reduces in a single block.
  The K-loop runs `Kt` times entirely inside DEST without any `pack_tile`/reload
  cycle. **Zero L1 round-trips during the K-reduction.**
- **Intermediate CB format (matmul → bias)**: `cb_partials` is allocated with
  `data_format=output_tensor.dtype` (descriptor line 154) = **bf16**.
  `UnpackToDestMode::UnpackToDestFp32` is **not** configured (the descriptor does
  not pass an `unpack_to_dest_mode` vector at all — defaults apply). The matmul
  result is therefore rounded to bf16 once before bias add re-unpacks.
  This is the single round-trip in the bias path; the no-bias path avoids it.
- **Round-trips through L1**: `0` for the K-reduction itself; `1` for the
  matmul→bias hand-off (bias path only); `0` for the no-bias path.
- **Order of operations**: matmul is a sum-of-products with no inner divisions;
  divide-then-sum vs sum-then-divide is not applicable.
- **Assessment**: The choice to put K-accumulation entirely in fp32 DEST and
  use HiFi4 for the bf16 multiplies is the single most important precision
  decision and is correctly made. The remaining precision limit is the bf16
  storage of `cb_partials` in the bias path, which truncates 23-bit fp32 to
  7-bit bf16 mantissa once before the bias add. In typical neural-network
  bias magnitudes (small relative to `input @ weight`) this loss is
  per-output-tile rounding noise rather than systematic error.

---

## Numerical Guards

| Guard | Present | Location | Notes |
|-------|:-------:|----------|-------|
| Max subtraction before exp | n/a | — | No `exp_tile` in this op |
| ReLU clamp for approx exp | n/a | — | No `exp_tile`; `LastBlockTarget::Out` is used with no relu (compute kernel line 109). `OutWithRelu` is available but unused |
| Epsilon before reciprocal | n/a | — | No `recip_tile` in this op |
| Non-tile-aligned masking | n/a | — | Validation enforces M, K, N divisible by 32 (`linear.py:96-97`). No mask tile or partial scaler needed |
| Welford's algorithm | n/a | — | No mean/variance computation |
| fp32 DEST accumulation | Yes | `linear_program_descriptor.py:251` | `fp32_dest_acc_en=True` |
| HiFi math fidelity | Yes (HiFi4) | `linear_program_descriptor.py:250` | Captures full bf16 mantissa across 4 FPU passes |
| Logical-vs-padded shape correctness | Yes | `linear_program_descriptor.py:50-55` and `linear.py:44-46` | Uses `shape[-2]/shape[-1]` (logical), not padded. Validation at `linear.py:96` rejects non-32-multiple shapes outright, so the tile grid `(Mt, Kt, Nt)` matches both logical and padded shapes |

---

## Math Fidelity Profile

| Compute phase | FPU/SFPU | Fidelity-sensitive | Default setting |
|---------------|----------|:------------------:|-----------------|
| `ckernel::matmul_block` (bf16 × bf16 multiply-add) | FPU | **Yes** | HiFi4 (4 passes, full bf16 mantissa on both operands) |
| `add_tiles_bcast_rows` (bias add) | FPU (add) | **No** (FPU add is exact for representable inputs) | HiFi4 (inherited; irrelevant for add) |
| `pack_tile_block` (DEST → L1) | Pack | n/a | round-to-nearest-even fp32→bf16 conversion at every pack |

Notes:

- Wormhole HiFi4 + `fp32_dest_acc_en=true` is flagged in the reference as a
  known-bad combination on Wormhole B0 (bug #38306). The code unconditionally
  picks `HiFi4 + fp32_dest_acc_en=True`. This is a precision/risk tradeoff
  worth flagging if the op runs on WH B0 — the recommended workaround per the
  reference is HiFi3 with fp32 accumulation.
- `math_approx_mode` is not set in the descriptor; SFPU has no codepath here, so
  the default (whatever the platform default is) is irrelevant.

- **User-configurable**: **No.** `linear()` (`linear.py:19-25`) accepts
  `input_tensor`, `weight_tensor`, optional `bias`, and `memory_config` only.
  No `compute_kernel_config` parameter is exposed; HiFi4 + fp32 DEST is hard-coded
  in the program descriptor.
- **Recommended minimum fidelity**: HiFi2 covers full bf16-input matmul precision
  on most hardware (both 7-bit mantissas in 2 passes). HiFi4 (4 passes) is
  conservative for bf16 inputs and only differs from HiFi2 if the inputs were
  fp16/fp32. The descriptor comment at lines 236-237 calls out that LoFi loses
  too many mantissa bits once K crosses one tile, but HiFi2 vs HiFi4 is not
  motivated; HiFi2 would likely be a throughput win at no precision cost for
  bf16×bf16, and HiFi3 specifically avoids the WH B0 fp32-DEST bug.

---

## Tile-Boundary Precision

- **Tiles per matmul output element**: each output element accumulates `Kt` tiles
  worth of products = `K / 32` multiply-add terms.
- **DEST capacity**: 4 tiles at fp32 half-sync (`dest_helpers.hpp:97`).
- **Tiles held in DEST per matmul iteration**: 1 (`out_subblock_h × out_subblock_w = 1 × 1`).
- **L1 round-trips per matmul output sub-block**:
  - During K-reduction: `0` — `num_k_blocks = 1`, the entire K-accumulation
    happens within DEST in fp32. No interm spill, no reload.
  - At sub-block boundary: 1 pack to L1 (the only one). In the no-bias path this
    pack lands in `cb_output_tiles` (final). In the bias path it lands in
    `cb_partials` (bf16 intermediate).
- **L1 round-trips per output (bias path)**: matmul→partials (bf16) +
  partials→bias-DEST + bias-DEST→cb_output_tiles = **1 intermediate-CB
  round-trip in bf16**. The bf16 storage at `cb_partials` is the precision
  bottleneck of the bias path; if the bias magnitude is comparable to the
  matmul output magnitude (typical case) the impact is one rounding event per
  output element, dominated by the inherent bf16-output rounding.
- **Intermediate format**: `cb_partials` is **bf16** (descriptor line 154
  reuses `output_tensor.dtype`). `UnpackToDestMode::UnpackToDestFp32` is **not**
  set — the descriptor never builds an `unpack_to_dest_mode` list, so all CBs
  inherit the default unpack path. This is consistent with the bf16 CB format
  but means even if `cb_partials` were promoted to Float32, an additional
  config change would be required to read it back into DEST without
  intermediate truncation.
- **Assessment**: For Phase 0 the K-reduction is essentially as precise as
  bf16-input matmul allows — fp32 DEST plus HiFi4 plus single-block K
  reduction. The bias path adds one bf16 rounding step that is small relative
  to typical output magnitudes but is a recognizable precision cost compared
  to a hypothetical fp32 `cb_partials`.

---

## Configuration Exposure

| Setting | Hard-coded value | Exposed to user | Where to change today |
|---------|------------------|:---------------:|-----------------------|
| `math_fidelity` | `MathFidelity.HiFi4` | No | `linear_program_descriptor.py:250` |
| `fp32_dest_acc_en` | `True` | No | `linear_program_descriptor.py:251` |
| `math_approx_mode` | not set (platform default) | No | n/a — no SFPU path |
| `packer_l1_acc` | not set on `ComputeConfigDescriptor` (defaults to false); template arg explicitly `false` (`linear_compute.cpp:80, 108`) | No | `linear_compute.cpp:80, 108` |
| `dst_full_sync_en` | not set (default = half-sync) | No | descriptor would need `dst_full_sync_en=True` |
| `cb_partials` data format | `output_tensor.dtype` (= bf16) | No | `linear_program_descriptor.py:154` |
| `UnpackToDestMode` per CB | not configured (all default) | No | descriptor would need an `unpack_to_dest_mode` list |
| Input dtypes | bf16 only — enforced by validation | No (forces bf16) | `linear.py:111-112` |
| Output dtype | bf16 (matches input) | No | `linear.py:51` |

The op currently ships a fixed precision posture; users have no knob.

---

## Key Observations

- **K-accumulation is correctly placed in fp32 DEST.** With `num_k_blocks=1`,
  `out_subblock_h=out_subblock_w=1`, and `fp32_dest_acc_en=True`, the entire
  per-output-element sum of `Kt × 32` bf16 products lands in a 23-bit fp32
  mantissa with zero L1 round-trips. This is the single biggest precision
  decision, and the descriptor comment (lines 236-242) documents the
  acceptance-test failure observed without it.

- **Bias path stores partials in bf16, not fp32.** `cb_partials` inherits
  `output_tensor.dtype` (= bf16). This costs one rounding-to-nearest-even
  event per output sub-block before the bias is added. Promoting `cb_partials`
  to `Float32` plus configuring `UnpackToDestMode::UnpackToDestFp32` for it
  would push the post-matmul precision through to the bias add intact, at the
  cost of 2× L1 size for that CB. For Phase 0 single-core sizes this is not
  a critical accuracy ceiling, but it is the next precision lever to pull if
  bias-path accuracy ever underperforms.

- **HiFi4 may be over-conservative for bf16-input matmul.** Both operands have
  7-bit mantissas; HiFi2 (2 FPU passes) covers full bf16×bf16 precision. The
  descriptor jumps directly to HiFi4 (4 passes — the LoFi→HiFi2 jump was
  defended in the comment, the HiFi2→HiFi4 jump was not). On Wormhole B0,
  HiFi4 + `fp32_dest_acc_en=true` is flagged as buggy (#38306); HiFi3 +
  fp32-DEST is the documented workaround. Worth re-evaluating at multi-core
  scale-out where the FPU-throughput cost matters.

- **No precision exposure to the user.** `linear()` does not accept a
  `compute_kernel_config` parameter. A user who wants LoFi for throughput on
  small K, or who wants to validate against HiFi3 to skirt the WH B0 bug,
  has to edit the program descriptor. This is fine for Phase 0 (op is single-
  core, one specific posture is correct) but should be opened up before this
  op is used as a general drop-in for the existing matmul + bias fused path.

- **No SFPU surface, no division, no subtraction.** The op has none of the
  classical numerical-stability hazards (catastrophic cancellation, underflow
  near zero, exp overflow, division-by-zero). The full stability story
  reduces to "fp32 K-accumulation + HiFi4 multiplies + bf16 round-trip into
  the bias-path intermediate CB", which is documented above.
