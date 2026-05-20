# Numerical Stability Analysis: layer_norm_rm

## Algorithm Summary

Per-row LayerNorm over the last dimension, computed as a **3-pass streaming** algorithm on a single Tensix core. The reader streams the input tensor three times from DRAM (one full sweep per pass), the compute kernel processes one tile-row (32 logical rows × `Wt` tiles wide) per outer iteration:

| Pass | Math | Kernel call(s) |
|------|------|----------------|
| 1 | `mean = (1/W) · Σ x` | `reduce<SUM, REDUCE_ROW>(cb_input_tiles, cb_scaler, cb_mean)` with scaler `1/W` |
| 2 | `var = (1/W) · Σ (x − mean)²`; `inv_std = 1/√(var + ε)` | `sub<COL>(cb_input_tiles, cb_mean, cb_centered)` → `square_in_place(cb_centered)` → `reduce<SUM, REDUCE_ROW>(cb_centered, cb_scaler, cb_inv_std)` → `transform_in_place(cb_inv_std)` running `add_unary_tile(eps_bits)` + `rsqrt_tile` |
| 3 | `y = (x − mean) · inv_std · γ + β` | `sub<COL>(cb_input_tiles, cb_mean, cb_centered)` → `mul_in_place<COL>(cb_centered, cb_inv_std)` → optional `mul_in_place<ROW>(cb_centered, cb_gamma_tiles)` → optional `add_in_place<ROW>(cb_centered, cb_beta_tiles)` → drain |

**Precision-sensitive phases:**
1. Pass 1 row sum across `W` elements (mean) — the reduction depth is `O(W)`, scaler is `1/W` injected at every multiply-accumulate ⇒ **divide-then-sum**, NOT sum-then-divide.
2. Pass 2 centering `(x − mean)` followed by square — catastrophic-cancellation risk when `x ≈ mean`, then squaring amplifies the residual error.
3. Pass 2 row sum of squared centered values — same `O(W)` depth, same `1/W` divide-then-sum.
4. `add(var, eps)` + `rsqrt` — division-by-near-zero protected only by `epsilon` (single `1e-5` default).
5. Pack of `mean`, `inv_std`, `centered` to L1 between passes — every tile boundary truncates to the intermediate-CB format (= input dtype).

`mean`, `inv_std`, `centered`, and gamma/beta tiles are all stored in the **input tensor's dtype** (bfloat16, float32, or bfloat8_b). The scaler CB is **always bfloat16** (a fixed contract from `reduce_helpers_dataflow.hpp`).

## Error Source Inventory

| # | Source | Location | Severity (bf16 input) | Mitigation |
|---|--------|----------|-----------------------|------------|
| 1 | Pass 1 row-sum over `W=Wt·32` elements, scaler `1/W` baked into the matmul | `layer_norm_rm_compute.cpp:87-95` → `reduce<SUM, REDUCE_ROW>` | High for `W ≥ 256`; divide-then-sum compounds rounding | DEST acc *configurable* (`fp32_dest_acc_en`, off by default); matmul path forces HiFi4 fidelity regardless of user setting |
| 2 | Bf16 pack of `mean` to `cb_mean` (input-dtype) | reduce → `pack_tile` → cb_mean (bf16 if input bf16) | Medium: mean's relative precision is bounded by 7-bit mantissa | None. cb_mean is always input dtype; no `UnpackToDestMode::UnpackToDestFp32` is configured |
| 3 | Catastrophic cancellation `x − mean` in Pass 2 | `layer_norm_rm_compute.cpp:102-105` → `sub<COL>` | High when input dimension is near-uniform / has large bias (typical for LayerNorm inputs in attention residual streams) | None; algorithm is two-pass, not Welford's |
| 4 | `square_in_place(cb_centered)` amplifies any cancellation error to `O(err²)` magnitude | `layer_norm_rm_compute.cpp:107` | High (paired with #3) | None |
| 5 | Pass 2 row-sum of `(x−mean)²` over `W` elements, scaler `1/W` | `layer_norm_rm_compute.cpp:109-117` | High for `W ≥ 256`, same as #1 | DEST acc configurable; HiFi4 forced for the matmul-reduce |
| 6 | Bf16 pack of `inv_std` to `cb_inv_std` (input-dtype) | reduce → cb_inv_std → `transform_in_place` | Medium | None |
| 7 | `add_unary_tile(var, eps_bits)` — eps is an fp32 bit-pattern but operates on whatever format is in DST | `layer_norm_rm_compute.cpp:120-125` | Low: eps=1e-5 is representable in bf16; only matters when var ≈ 0 | epsilon guard present (configurable, default `1e-5`) |
| 8 | `rsqrt_tile` SFPU on `(var + eps)` | `layer_norm_rm_compute.cpp:123-124` | Low–Medium: precise rsqrt is ≤1 ULP; approximate mode is ~3 ULP | `math_approx_mode=False` by default (precise mode); user can flip via `compute_kernel_config` |
| 9 | Pass 3 `sub<COL>` recomputes `(x − mean)` — same cancellation pattern as #3, but result drives the **output**, not a downstream square | `layer_norm_rm_compute.cpp:131-134` | High (drives final output) | None |
| 10 | `mul_in_place<COL>(centered, inv_std)` — FPU multiply | `layer_norm_rm_compute.cpp:136-137` | Fidelity-sensitive (eltwise mul on FPU); DEST narrow at fp16 by default | User-configurable `math_fidelity`; default HiFi4 |
| 11 | Optional `mul_in_place<ROW>(centered, gamma)` | `layer_norm_rm_compute.cpp:139-142` | Fidelity-sensitive eltwise mul | Same as #10 |
| 12 | Optional `add_in_place<ROW>(centered, beta)` | `layer_norm_rm_compute.cpp:143-146` | Eltwise add is exact on FPU; only the bf16 pack of the result is lossy | Pack to output dtype |
| 13 | `untilize` → row-major output (for RM-out path) | `layer_norm_rm_compute.cpp:150` | None (movement, not arithmetic) | n/a |
| 14 | Reader read-three-times (DRAM bandwidth × 3); same bits each pass — no precision loss but compounds the rounding from L1-format reload | `layer_norm_rm_reader.cpp:87-119` | n/a for precision | n/a |
| 15 | Non-tile-aligned `W` (i.e. `W % 32 ≠ 0`) garbage in last-tile columns | reader emits `prepare_partial_reduce_scalers` → compute uses `ReducePartialScaler::last_tile_at(1)` | Cleanly handled | Partial-scaler mask (NOT a multiply mask) — no extra round-trip |

## Accumulation Analysis

**What is accumulated (Pass 1 mean and Pass 2 variance):**
- Each row reduction sums `W` elements scaled by `1/W` via the **matmul LLK path** (because REDUCE_ROW + SUM dispatches through `reduce_uses_matmul()` → matmul with the scaler tile as srcB). `reduce_helpers_compute.inl:215` selects the matmul branch.

| Property | Value |
|----------|-------|
| Accumulation depth (per row) | `W = Wt × 32` elements |
| Tiles streamed per reduction | `Wt` |
| Input policy | `WaitAndPopPerTile` (single-tile streaming) |
| DEST precision | **bf16 by default** (`ComputeConfigDescriptor()` default `fp32_dest_acc_en=False`). User-settable to fp32 |
| Intermediate CB (`cb_mean`, `cb_inv_std`) format | **`input_tensor.dtype`** — bf16 / fp32 / bf8_b. **Never forced to fp32**, even when `fp32_dest_acc_en=True` |
| `UnpackToDestMode::UnpackToDestFp32` | **NOT configured.** No `compute_kernel_config.unpack_to_dest_mode` is set in `create_program_descriptor` |
| DEST round-trips per reduction | 0 within a single Wt-tile reduce: the accumulator stays in DEST[0] for all `Wt` tiles. (The matmul accumulates srcA·srcB into the same dest register.) However, `mean` is packed to `cb_mean` and unpacked by `sub<COL>` later — that's **1 round-trip** for the mean. Similarly **1 round-trip** for `inv_std`. |
| Scaler position in the math | `1/W` is the **srcB** of each matmul tile → divide-then-sum: every element is multiplied by `1/W` before being summed |
| Order-of-operations | **divide-then-sum** (`1/W` baked into the scaler). Sum-then-divide is NOT used. |

**Assessment:**
- Within a single `reduce<>` call the accumulator never leaves DEST — there is no per-tile L1 round-trip, so the per-row depth-`W` sum is the only relevant rounding chain.
- The product-then-sum order means each of the `W` partial products is rounded once at the multiply, then summed. With bf16 dest, the running sum becomes lossy for `W ≥ ~256` (each new term of order `1/W` is small compared to the running sum that approaches `μ`). With fp32 dest, accumulation is essentially exact for any practical `W`.
- The bf16 storage of `mean` and `inv_std` (when input is bf16) is a hard cap on Pass-2/3 precision: even if Pass 1 ran with `fp32_dest_acc_en=True`, the mean is packed to bf16 before re-use ⇒ `(x − mean)` cancellation uses a 7-bit-mantissa mean. **`fp32_dest_acc_en` therefore only partially mitigates the precision loss**; setting it does NOT make Pass-2 cancellation any safer unless the entry point is also extended to override `cb_mean` / `cb_inv_std` to Float32 and add `unpack_to_dest_mode = UnpackToDestFp32` for those CBs.

## Numerical Guards

| Guard | Present | Location | Notes |
|-------|:-------:|----------|-------|
| Max subtraction before exp | n/a | — | No exp in the kernel |
| ReLU clamp for approx exp | n/a | — | — |
| Epsilon before reciprocal/rsqrt | **✓** | `layer_norm_rm_compute.cpp:120-125` | `add_unary_tile(dst, eps_bits)` before `rsqrt_tile`. Default `eps = 1e-5` (configurable via `epsilon` kwarg). Eps is plumbed as a fp32 bit pattern through compile-time arg 7. |
| Non-tile-aligned-W handling | **✓** | reader: `dataflow_kernel_lib::prepare_partial_reduce_scalers<…, partial_w>(inv_W)`; compute: `partial_scaler = ReducePartialScaler::last_tile_at(1)` | Partial-scaler tile (zero out-of-bounds positions) — no mask CB, no extra pack/unpack round-trip. Reads `origin_W = int(input_shape[-1])` (the logical shape), not the padded shape ⇒ `1/W` divisor is correct. |
| Welford's algorithm | ✗ | — | Two-pass mean+variance is used. `(x − mean)` cancellation is unmitigated. |
| Subtraction of similar values | unguarded | Pass 2, Pass 3 `sub<COL>` | The dominant precision risk for typical LayerNorm inputs where activations are near-zero-mean. |

## Math Fidelity Profile

| Compute phase | FPU/SFPU | Fidelity-sensitive | Effective setting |
|---------------|----------|:------------------:|-------------------|
| Pass 1 `reduce<SUM, REDUCE_ROW>` (mean) | FPU matmul | Yes | **Forced HiFi4** — `REDUCE_MATMUL_FIDELITY = HiFi4` in `reduce_helpers_compute.inl:22`, **overrides the user's `math_fidelity`** for this op |
| Pass 2 `sub<COL>(x, mean, centered)` | FPU eltwise add (sub) | No (add/sub is exact) | n/a |
| Pass 2 `square_in_place` (eltwise mul of cb_centered with itself) | FPU eltwise mul | **Yes** | User-configurable; default HiFi4 |
| Pass 2 `reduce<SUM, REDUCE_ROW>` (variance) | FPU matmul | Yes | Forced HiFi4 (same as Pass 1) |
| `add_unary_tile(eps)` | SFPU eltwise unary | No | n/a |
| `rsqrt_tile` | SFPU | Affected by `math_approx_mode` (NOT by `math_fidelity`) | Default `math_approx_mode=False` → precise rsqrt (≤1 ULP for fp32 inputs) |
| Pass 3 `sub<COL>` | FPU eltwise add | No | n/a |
| Pass 3 `mul_in_place<COL>` (× inv_std) | FPU eltwise mul | Yes | User-configurable; default HiFi4 |
| Pass 3 `mul_in_place<ROW>` (× gamma) | FPU eltwise mul | Yes | User-configurable; default HiFi4 |
| Pass 3 `add_in_place<ROW>` (+ beta) | FPU eltwise add | No | n/a |
| Tilize / untilize | Data movement | No | n/a |

- **User-configurable**: Yes — `layer_norm()` exposes `compute_kernel_config: ttnn.ComputeConfigDescriptor`. The default if `None` is `ttnn.ComputeConfigDescriptor()` ⇒ `math_fidelity=HiFi4`, `fp32_dest_acc_en=False`, `math_approx_mode=False`.
- **Recommended minimum fidelity**: HiFi2 in principle; in practice the dominant reductions are *already* forced to HiFi4 by the kernel-lib matmul wrapper, so the user-visible `math_fidelity` only affects Pass-2 `square_in_place` and Pass-3 multiplies. LoFi for those should still be reasonable since the inputs to those multiplies are already-reduced scalars or small affine factors.

## Tile-Boundary Precision

| Property | Value |
|----------|-------|
| Tiles per row-reduction | `Wt` (= `ceil(W/32)`) |
| Accumulator behavior **within** a single reduce<> | Stays in DEST[0] for all `Wt` matmul-accumulate steps. **0 L1 round-trips inside a reduce.** |
| Round-trips of `mean` between Pass 1 and Pass 2 | **1** (pack to `cb_mean` after Pass 1; unpack into srcB of `sub<COL>` in Pass 2). Format: `input_tensor.dtype`. |
| Round-trips of `mean` between Pass 2 and Pass 3 | **0 additional** — `cb_mean` is consumed via `BinaryInputPolicy::WaitUpfrontNoPop` and remains in L1, then popped once at end of row. The same bf16 mean tile is reused by Pass 3's `sub<COL>`. |
| Round-trips of `inv_std` between Pass 2 reduce and `transform_in_place` | **1** (pack to `cb_inv_std`, then `transform_in_place` does `copy_tile → rsqrt → pack` ⇒ a second round-trip in `cb_inv_std`). |
| Round-trips of `inv_std` into Pass 3 | **0 additional** — `WaitUpfrontNoPop` reuse. |
| Round-trips of `centered` within Pass 2 / Pass 3 | Each of `sub`, `square_in_place`, `mul_in_place<COL>`, `mul_in_place<ROW>`, `add_in_place<ROW>` packs back to `cb_centered`. For Pass 3: **3–5 round-trips per tile** (sub → mul × inv_std → optional mul × γ → optional add β → drain). All at `input_tensor.dtype`. |
| DEST capacity | `8 / (1 + fp32_dest_acc_en)` half-sync = 8 tiles bf16 / 4 tiles fp32 — but irrelevant for the dominant per-row reduce which uses DEST[0] only |
| Intermediate CB format | `cb_mean`, `cb_inv_std`, `cb_centered`, `cb_gamma_tiles`, `cb_beta_tiles` are all `input_tensor.dtype`. **No CB is forced to Float32.** |
| `UnpackToDestFp32` configured | **No** — `ComputeConfigDescriptor.unpack_to_dest_mode` is not set. Reload of `cb_mean` / `cb_inv_std` into DEST for `sub<COL>` / `mul<COL>` goes through SrcA/B at the CB's stored format. |

**Assessment.** The reduction itself is precision-tight: the accumulator stays in DEST across all `Wt` matmul steps, so the dominant accumulation error is determined by DEST mode (bf16 by default). The **weak link** is the per-row `mean`/`inv_std` which always pass through L1 at the input dtype. For bf16 input, this caps mean precision at 7 mantissa bits regardless of `fp32_dest_acc_en`. The `centered` CB taking multiple bf16 round-trips through Pass 3 is also notable but each round-trip is a single representation conversion of an `O(1)` magnitude value, so cumulative error there is modest.

## Configuration Exposure

| Setting | Exposed to user | Default | Notes / Recommendation |
|---------|:--------------:|---------|------------------------|
| `fp32_dest_acc_en` | ✓ (via `compute_kernel_config`) | `False` | Strongly improves Pass 1 and Pass 2 reduction precision **inside DEST**. Has limited effect on Pass-2 cancellation because `cb_mean` is still packed at input dtype (no companion change to CB format). DEST limit halves 8 → 4 tiles when enabled — no impact here because reductions use DEST[0] only. |
| `math_fidelity` | ✓ | `HiFi4` | Pass-1 and Pass-2 *reductions* ignore this — they are hard-coded to HiFi4 via `REDUCE_MATMUL_FIDELITY`. Only `square_in_place`, Pass-3 `mul × inv_std`, and Pass-3 `mul × γ` actually honor the user setting. |
| `math_approx_mode` | ✓ | `False` (precise) | Affects only the SFPU `rsqrt_tile`. Precise mode is the safe default for layer norm. |
| `dst_full_sync_en` | ✓ (via `compute_kernel_config`) | `False` | Half-sync. Reductions don't need full-sync. |
| `unpack_to_dest_mode` | ✓ (settable on `ComputeConfigDescriptor`) | unset | **Not used.** Would be needed if `cb_mean` / `cb_inv_std` were ever Float32-formatted to allow precision-preserving reload. |
| `bfp8_pack_precise` | ✓ | unset | Relevant only for bfloat8_b output packing path. |
| `epsilon` | ✓ (op kwarg) | `1e-5` | Plumbed as fp32 bit pattern → `add_unary_tile`. |
| `packer_l1_acc` | ✗ | n/a | Not used; reductions stay in DEST. |
| Scaler CB dtype | ✗ | bfloat16 (fixed) | Per `reduce_helpers_dataflow.hpp` contract: scaler tile is always bf16. Affects the precision of the `1/W` constant — for `W = 2048`, `1/W ≈ 4.88e-4` rounds to bf16 with ~0.4% relative error, which the matmul then propagates. |

## Key Observations

1. **The dominant reductions are pinned at HiFi4 fidelity regardless of user input.** Both row sums (mean and variance) go through `reduce_with_matmul_init` which forces `REDUCE_MATMUL_FIDELITY = HiFi4` — the operation's `math_fidelity` config only affects Pass-2 `square_in_place` and Pass-3 elementwise multiplies. This is a precision-positive default; it also explains why the per-dtype tolerances in `eval/golden_tests/layer_norm_rm/helpers.py` (`bf16: pcc=0.995 rms=0.04`, `fp32: pcc=0.9999 rms=0.02`) are achievable even with `fp32_dest_acc_en=False`.

2. **The dominant precision risk is two-pass cancellation in `(x − mean)`, not the reductions.** The kernel does not use Welford's algorithm; for LayerNorm inputs that are near-zero-mean per row (typical for residual-stream activations), the subtraction has limited significant bits. The error is squared in Pass 2 and propagates linearly into Pass 3's output. The looser `bfloat8_b` tolerance (`pcc=0.99, rms=0.10`) reflects this compounded loss when the centered values are stored at bf8_b in `cb_centered`.

3. **`fp32_dest_acc_en=True` is only a partial fix because intermediate CBs are stored at input dtype.** Even with fp32 DEST accumulation, the `mean` and `inv_std` scalars are packed back to `cb_mean` / `cb_inv_std` at `input_tensor.dtype` (bf16 for the common case) and reloaded through SrcA/B without `UnpackToDestFp32`. To realize the full benefit of `fp32_dest_acc_en`, the program descriptor would need to (a) override `cb_mean`, `cb_inv_std` (and likely `cb_centered`) to `ttnn.float32`, and (b) set `compute_kernel_config.unpack_to_dest_mode` for those CB indices to `UnpackToDestMode::UnpackToDestFp32`. Neither is currently done.

4. **Mean uses divide-then-sum, not the more numerically-stable sum-then-divide.** The reader plumbs `1/W` into the scaler tile, which the matmul multiplies into every element before summation. Each of the `W` products is rounded once before contributing to the sum. Sum-then-divide would batch the `W − 1` extra rounding errors into a single post-reduction multiply, and would keep the running sum at full magnitude for accumulation. The current choice trades a modest precision drop for not needing an extra `mul_tiles_bcast_scalar` pass.

5. **Non-tile-aligned `W` is correctly handled with the partial-scaler pattern.** `origin_W = int(input_shape[-1])` (logical width) sets `inv_W = 1.0 / W` and the partial scaler zeroes the out-of-bounds positions of the last tile — no contamination of the reduction. The scaler CB also holds *two* tiles in this case (`cb_scaler_pages = 2`), with `last_tile_at(1)` picking the partial tile. This is the recommended pattern from the accuracy tips reference and is more efficient than a mask-tile multiply (no extra L1 round-trip).

6. **The 3-pass design re-reads the input from DRAM three times.** This is a bandwidth cost, not a precision cost — the same input bits are observed each pass, so there is no rounding between passes for the input itself. (A one-pass Welford implementation would halve the DRAM bandwidth and *improve* numerical stability simultaneously, at the cost of a more complex compute kernel.)
