# Numerical Stability Analysis: rms_norm

## Algorithm Summary

`rms_norm` computes, along the innermost (W) dimension:

```
output[..., i, j] = input[..., i, j] / sqrt(mean(input[..., i, :]^2) + epsilon) * gamma[j]
```

Two-pass streaming reduce per 32-row chunk × `Wt` tiles wide (`Wt = ceil(W/32)`):

- **Pass 1 (Stages A+B+C)**: square every input tile → row-wise SUM/REDUCE_ROW with scaler `1/W` → add `eps` → rsqrt. Produces one `cb_mean_sq` tile per chunk holding `rsqrt(mean(x^2)+eps)` in column-0 layout.
- **Pass 2 (Stage D)**: stream input tiles, multiply by the rsqrt scalar broadcast across the column dimension. If gamma is present, **Stage E** then multiplies by `gamma` broadcast across the row dimension.

Precision-sensitive phases:

1. **Reduction sum over `W` elements** — accumulation depth grows with last-dim size.
2. **Squaring** — doubles dynamic range; values near the bf16 max grow toward overflow, values near bf16 min underflow to 0.
3. **`1/W` divide-then-sum scaler** — scales every element of the sum before accumulation.
4. **Add `eps` + rsqrt** — SFPU transcendental; ill-conditioned if `mean(x^2) ≈ -eps` (effectively impossible since `mean(x^2) ≥ 0`).
5. **bf16 packs after square / rsqrt / Stage D** — every intermediate CB round-trip is a precision event when the CB is bf16.

## Error Source Inventory

| # | Source | Location | Severity | Mitigation |
|---|--------|----------|----------|------------|
| 1 | Square (FPU mul) of input tiles | `rms_norm_compute.cpp:116-120` (Stage A `eltwise_chain` with `Square`) | Moderate (squaring expands range; bf16 mantissa rounds the result) | Affected by `math_fidelity` (HiFi4 default). fp32 dest helps when fp32 input |
| 2 | Pack `x^2` to `cb_x_sq` (input_dtype) | `rms_norm_compute.cpp:120` + descriptor lines 211-223 | High for bf16 inputs (squared values lose 16 mantissa bits per pack); none for fp32 (lossless) | Float32 intermediate CB only when input is fp32; no fp32 override possible |
| 3 | Accumulation `Σ x²` (FPU mul-add via `reduce<SUM, REDUCE_ROW>`) | `rms_norm_compute.cpp:125-133` | Moderate–High: depth = `W` elements (formula: `Wt × 32` if `W%32==0`, else `(Wt-1)×32 + partial_w`). For typical `W ∈ {32 … 4096}` this is 32–4096 terms | `fp32_dest_acc_en = (input.dtype==fp32)` only. For bf16 inputs, accumulation happens in bf16 dest — depth >32 risks precision loss |
| 4 | Divide-then-sum scaler `1/W` | `rms_norm_program_descriptor.py:289` (`_fp32_bits(1.0/W)`), applied per-element by FPU during reduce | Moderate: every term is rounded after multiply by `1/W` before being summed. Small terms can flush to zero in bf16 when `1/W < 2⁻¹²⁷` (only at astronomical `W`) | None — would require scaler=1 + post-multiply by `1/W` to switch to sum-then-divide |
| 5 | bf16 scaler tile precision | `rms_norm_reader.cpp:65-67`, descriptor line 203 (`data_format=bfloat16`) | Low: `1/W` for typical W rounds to ~7 mantissa bits ≈ 3.9×10⁻³ relative error | Scaler CB is fixed bfloat16 regardless of input dtype |
| 6 | Add `eps` (SFPU `add_unary_tile`) | `rms_norm_compute.cpp:139-140` | Low (additive guard, exact within precision of dest) | Guards against `1/sqrt(0)` |
| 7 | `rsqrt_tile<false>` (SFPU) | `rms_norm_compute.cpp:141-142` | Low–Moderate. `legacy_compat=false`, `FAST_APPROX=false`, but `APPROX` macro is gated by `math_approx_mode` (defaults to **false** → precise path) | Precise path: 2 Newton–Raphson iterations |
| 8 | Pack `cb_mean_sq` after rsqrt | `rms_norm_compute.cpp:138-143` (`transform_in_place`) | Moderate for bf16 input | CB format = input dtype (fp32 lossless, bf16 lossy) |
| 9 | Stage D mul `x * rsqrt_scalar` (FPU, BroadcastDim::Col) | `rms_norm_compute.cpp:150-200` | Moderate (single FPU mul; sensitive to `math_fidelity`) | HiFi4 default; affects significand passes |
| 10 | Pack normalized output `cb_x_norm` / `cb_output_tiles` | `rms_norm_compute.cpp:164, 183, 199` | Moderate for bf16 output; lossless for fp32 | CB format follows tensor dtype |
| 11 | Stage E mul `x_norm * gamma` (FPU, BroadcastDim::Row) | `rms_norm_compute.cpp:169-183` (gamma path) | Moderate, fidelity-sensitive | HiFi4 default |
| 12 | Partial-W tile garbage columns (when `W%32 ≠ 0` and input is RM) | `rms_norm_compute.cpp:81-82`, reader lines 58-67 | Mitigated | `ReducePartialScaler::last_tile_at(1)` — partial scaler tile zeros padded columns during reduce |
| 13 | Partial-H garbage rows (last chunk with `H%32 ≠ 0`) | reader lines 105-115, writer lines 37-42 | Mitigated for output (writer skips invalid rows). Compute still squares + Stage D multiplies the garbage rows; their `rsqrt` is computed from valid rows only because REDUCE_ROW is row-wise | Writer writes only `total_units` valid rows |
| 14 | Catastrophic cancellation | (none) | N/A | No subtraction of similar magnitudes anywhere in the pipeline (RMSNorm is mean-of-squares, not mean+variance) |
| 15 | Approx-exp / overflow | (none) | N/A | No exp/log in the pipeline |

## Accumulation Analysis

- **What is accumulated**: row-wise sum of `x²` for the RMS denominator.
- **Accumulation depth**: `W` elements per row when `W%32==0`; `(Wt-1)·32 + partial_w` elements when row-major + partial-W. Bounded only by the user's last-dim size.
- **Dest precision**:
  - `input.dtype == float32`: DEST is fp32 (program descriptor sets `fp32_dest_acc_en=True`). `DEST_AUTO_LIMIT` halves to 4 tiles (half-sync default).
  - `input.dtype == bfloat16`: DEST is fp16 b. `fp32_dest_acc_en=False`. Capacity 8 tiles (half-sync default).
- **Intermediate CB format**: `cb_x_sq`, `cb_mean_sq`, `cb_x_norm` all use `input_dtype` (descriptor lines 211-254). Float32 only when input is fp32; bf16 otherwise.
- **`UnpackToDestMode::UnpackToDestFp32` configured**: **No.** Even for fp32 input, the helper-library reload from L1 goes through SrcA/SrcB and may suffer TF32 truncation on intermediate reloads (per reference §2.7).
- **Round-trips through L1**: in pass 1, exactly one pack-and-reload between `cb_x_sq` and the reducer (Stage A packs `Wt` tiles, then Stage B's `BulkWaitBulkPop` reads them back as one bulk). The reduce helper internally handles `ceil(Wt / DEST_AUTO_LIMIT) − 1` reload-tile-from-CB events when `Wt` exceeds dest capacity (= 8 for bf16, 4 for fp32). In Stage D and E, no accumulation roundtrip — each output tile is produced and packed independently.
- **Order of operations**: **divide-then-sum** (scaler `1/W` baked into reduce). Adds `W − 1` extra rounding events vs sum-then-divide and risks small-term flushing at extreme `W`. For typical neural-net widths this is acceptable but is a known suboptimal choice.
- **Assessment**: For bf16 inputs with `W > 256`, the combination of (a) divide-then-sum, (b) bf16 dest, (c) bf16 `cb_x_sq` roundtrip, and (d) bf16 scaler tile compounds error. Quality of `mean(x²)` is the dominant error source under bf16. For fp32 inputs the situation is much healthier because dest, intermediates, and DESTAUTO halving are all fp32, but the missing `UnpackToDestFp32` mode is a residual loss.

## Numerical Guards

| Guard | Present | Location | Notes |
|-------|:-------:|----------|-------|
| Max subtraction before exp | N/A | — | No exp in pipeline |
| ReLU clamp for approx exp | N/A | — | No exp in pipeline |
| Epsilon before reciprocal/sqrt | Yes | `rms_norm_compute.cpp:139-140` (`add_unary_tile(dst, eps_bits)` before `rsqrt_tile`) | `epsilon` default 1e-6 from entry point; user-configurable. eps is converted to fp32 bits in the descriptor (line 350) and added via SFPU regardless of input dtype |
| Non-tile-aligned masking (W axis) | Yes (partial scaler) | `rms_norm_compute.cpp:81-82`; reader lines 58-67 | Uses partial-scaler technique (no mask CB, no extra pack roundtrip). Only active for RM input (TILE input requires `W%32==0`) |
| Non-tile-aligned handling (H axis) | Yes (writer-side row count) | reader lines 105-115; writer lines 37-42 | Compute kernel still operates on `Wt` tiles regardless; out-of-bounds rows produce arithmetic garbage that is then discarded by the writer's `write_sticks_after_untilize` (which writes only `total_units` valid rows). Row-wise REDUCE_ROW means out-of-bounds rows do not contaminate valid rows' RMS |
| Welford's algorithm | No | — | Not applicable (RMSNorm has no mean-subtraction; Welford is unnecessary) |
| `logical_shape()` for W | Yes (implicit) | `rms_norm.py` uses `input_tensor.shape`; `1/W` uses logical W | Descriptor uses `shape[-1]` = logical W when computing the scaler (line 289). No padded-shape contamination |

## Math Fidelity Profile

| Compute phase | FPU/SFPU | Fidelity-sensitive | Default setting |
|--------------|----------|:------------------:|-----------------|
| Square (`Square<Dst::D0>` via `eltwise_chain`) | FPU mul | **Yes** | HiFi4 (descriptor default) |
| Row-wise reduce SUM (`reduce<SUM, REDUCE_ROW>`) | FPU mul (scaler · operand) + add | **Yes** | HiFi4 |
| `add_unary_tile(eps)` | SFPU add | No | n/a |
| `rsqrt_tile<false>` | SFPU | No (controlled by `math_approx_mode`) | `math_approx_mode=False` → precise path (2 Newton-Raphson iterations, ≤1 ULP fp32) |
| Stage D mul `x · rsqrt` (BroadcastDim::Col) | FPU mul | **Yes** | HiFi4 |
| Stage E mul `x_norm · gamma` (BroadcastDim::Row) | FPU mul | **Yes** | HiFi4 |
| Tilize / untilize | Data movement | No | n/a |

- **User-configurable**: **No.** The entry point `rms_norm(input_tensor, *, gamma, epsilon)` does not accept a `compute_kernel_config`. The descriptor hard-codes `ComputeConfigDescriptor(fp32_dest_acc_en=fp32_dest)` only, leaving every other field at the `ComputeConfigDescriptor` defaults (`math_fidelity=HiFi4`, `math_approx_mode=False`, `dst_full_sync_en=False`, `bfp8_pack_precise=False`).
- **Recommended minimum fidelity**: HiFi2 for normalization workloads. HiFi4 (the current default) is the safe choice but costs 4× FPU passes per multiply vs LoFi. Operations like LayerNorm/RMSNorm typically run fine at HiFi2 with negligible PCC drop.
- **Wormhole B0 caveat**: HiFi4 + `fp32_dest_acc_en=true` triggers hardware bug #38306. The fp32 path of this op (input dtype fp32) currently hits exactly this combination. **HiFi3 is the recommended alternative for the fp32 path on Wormhole.**

## Tile-Boundary Precision

- **Tiles in reduction**: `Wt = ceil(W/32)` tiles per row, processed by REDUCE_ROW in one bulk.
- **Dest capacity** (half-sync, the default):
  - bf16 input: `DEST_AUTO_LIMIT = 8` tiles.
  - fp32 input: `DEST_AUTO_LIMIT = 4` tiles.
- **L1 round-trips per reduce** (helper-library internal): `ceil(Wt / DEST_AUTO_LIMIT) − 1` reload events from `cb_x_sq`. For `Wt ≤ DEST_AUTO_LIMIT` (= 8 bf16 or 4 fp32) no reload is needed.
  - Example: `W = 1024` (`Wt = 32`) ⇒ 3 reload events in bf16 mode, 7 reload events in fp32 mode.
- **Stage A → Stage B handoff** (`cb_x_sq`): single mandatory pack of `Wt` tiles. CB format equals input dtype, so the squared values traverse bf16 only when input is bf16.
- **Stage B → Stage C handoff** (`cb_mean_sq`): one tile carrying `mean(x²)`. Same format as input dtype. fp32 input ⇒ lossless; bf16 input ⇒ 7-mantissa-bit rounding event.
- **Stage C → Stage D handoff**: `cb_mean_sq` is held with `WaitNoPop` for all `Wt` Stage D iterations — no per-iteration reload; only one unpack at Stage D entry.
- **Stage D → Stage E handoff** (`cb_x_norm`, gamma path): pack-and-reload of `Wt` normalized tiles. CB format = input dtype.
- **Stage D/E → output**: `cb_output_tiles` format = output dtype (= input dtype here).
- **Missing UnpackToDestFp32**: `unpack_to_dest_mode` is left at default for every CB. When fp32 is active, the helper library still unpacks via SrcA/SrcB. Per the reference, this may truncate to TF32 on the reload path. A safe improvement would be `unpack_to_dest_mode[cb_x_sq] = UnpackToDestFp32` and `unpack_to_dest_mode[cb_mean_sq] = UnpackToDestFp32` when `fp32_dest_acc_en=true`.
- **Assessment**: bf16 path has 3 mandatory bf16 packs in flight (`cb_x_sq`, `cb_mean_sq`, `cb_output_tiles`/`cb_x_norm`). For `W ≤ 256` (`Wt ≤ 8`) the reduce stays entirely in dest, so only the explicit handoff packs hurt. For `W = 1024+`, additional intra-reduce reloads compound. fp32 path is largely lossless except for the SrcA/SrcB reload path (mitigatable with UnpackToDestFp32).

## Configuration Exposure

| Setting | Exposed to user | Default | Recommendation |
|---------|:---------------:|---------|----------------|
| `fp32_dest_acc_en` | No (auto-derived) | `True` iff `input.dtype == float32`, else `False` | Expose `compute_kernel_config` so callers can enable fp32 accumulation for bf16 inputs too. Today, bf16 inputs cannot opt into fp32 dest, even though it would help reductions over large `W` |
| `math_fidelity` | No | HiFi4 (via descriptor binding default) | Expose. Many normalization callers are happy with HiFi2; current HiFi4 default is the slowest matmul throughput tier. Also exposes the Wormhole HiFi4 + fp32 dest workaround to users |
| `math_approx_mode` | No | False | Expose. Current precise rsqrt is the right default; users with tight perf budgets and tolerant PCC may want `True` |
| `dst_full_sync_en` | No | False (half-sync) | Low priority. Enabling it would double dest capacity but at sync cost — generally not worth exposing |
| `packer_l1_acc` | No (not bound) | n/a | Not used here — no L1 accumulation pattern in the kernel. Would conflict with fp32 dest anyway. No action |
| `unpack_to_dest_mode` | No | Default for every CB | For fp32 path, setting `UnpackToDestFp32` on `cb_x_sq` and `cb_mean_sq` would close the residual reload-precision gap. Internal change; no user surface needed |
| `epsilon` | Yes | `1e-6` | Already exposed and propagated as fp32 bits |

## Key Observations

- **fp32 accumulation tied 1-to-1 to input dtype.** The descriptor enables `fp32_dest_acc_en` only when the user supplies a float32 input. There is no path for bf16 callers to opt into fp32 accumulation, even though the reduction depth (`W` elements) makes this the single highest-leverage precision improvement. Refinement opportunity: expose `compute_kernel_config` end-to-end and let callers pin `fp32_dest_acc_en` independently of input dtype.
- **Divide-then-sum, not sum-then-divide.** The reduce scaler is `1/W` (`rms_norm_program_descriptor.py:289`), which multiplies every element before accumulation. Switching to scaler `1.0` plus a post-reduction `mul_tiles_bcast_scalar(1/W)` (or fusing `1/W` into the rsqrt step) would remove `W − 1` extra rounding events and bring the operator in line with the accuracy-tips reference. The bf16 scaler tile (also a precision event) becomes unnecessary if `1/W` is folded into the SFPU stage.
- **Wormhole HiFi4 + fp32 dest hits a known hardware bug.** For float32 inputs the program currently asks for HiFi4 (descriptor default) AND `fp32_dest_acc_en=True` — exactly the combination flagged in tt_metal bug #38306. The fp32 path may produce silently incorrect numerics on Wormhole B0. Either expose `math_fidelity` to callers, or hard-cap to HiFi3 when `fp32_dest_acc_en` is on.
- **Numerical guards are minimal but correct for the algorithm.** Epsilon is added before rsqrt, partial-W is handled with the efficient partial-scaler technique (no mask CB), and `1/W` uses logical (not padded) `W`. RMSNorm has no catastrophic cancellation site, so the absence of Welford's algorithm is correct; the precise rsqrt path (`fast_and_approx=false`) is conservative and a reasonable default.
- **Missing `UnpackToDestMode::UnpackToDestFp32` for the fp32 path.** Even when the user supplies float32 input, intermediate CB reloads go through SrcA/SrcB and may truncate to TF32. Setting `UnpackToDestFp32` on `cb_x_sq` (and on `cb_mean_sq` for the held rsqrt scalar) would close this residual gap with no API change.
