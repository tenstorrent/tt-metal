# Numerical Stability Analysis: backward_softmax

## Algorithm Summary

`backward_softmax` computes the vector–Jacobian product (VJP) of softmax:

    grad_input = output * (grad_output - sum(output * grad_output, dim))

with `dim ∈ {-1, -2}`. Inputs are float32, rank-4, tile-aligned `(N, C, H, W)`.

The implementation is a **two-pass streaming algorithm** over the reduction
axis, with per-lane re-reads of both inputs from DRAM:

- **Pass 1** (per block of `BLOCK_SIZE` tiles along the reduce axis):
  1. `mul(cb_grad_output, cb_output, cb_prod)`  — element-wise `dy ⊙ y`
  2. `accumulate_reduce_block<SUM, REDUCE_ROW|REDUCE_COL>(cb_prod, cb_scaler, cb_sum, b, NUM_BLOCKS)`
     — block-streamed SUM-reduce that reloads/repacks the running accumulator
       from `cb_sum` between blocks via `Accumulate::at(cb_sum, b)`.
- **Pass 2** (per block):
  1. `sub<SCALAR>(cb_grad_output, cb_sum, cb_centered)` — broadcast subtract
     of the lane sum `s` from `dy` (this is the **catastrophic-cancellation
     site**: when `dy_i ≈ s` the relative error of `(dy_i − s)` blows up).
  2. `mul(cb_output, cb_centered, cb_grad_input)` — gates the centered
     gradient by `y` (which is in `[0, 1]` for softmax outputs, so does not
     amplify error).

**Precision-sensitive phases**:

| Phase | Why precision matters |
|-------|-----------------------|
| Pass-1 element-wise multiply `dy ⊙ y` | FPU multiply, fidelity-sensitive |
| Pass-1 streaming SUM over `Wt` (or `Ht`) tiles | Reduction depth = `W` (or `H`) elements per lane; runs through fp32 dest with a per-block round-trip through `cb_sum` (Float32 CB). |
| Pass-2 `sub<SCALAR>(dy, s)` | Catastrophic cancellation when `dy_i ≈ s`. The result drives the gradient magnitude. |
| Pass-2 multiply `y ⊙ (dy − s)` | FPU multiply, fidelity-sensitive |

The accumulation depth grows linearly with the reduction dimension (`W` for
`dim=-1`, `H` for `dim=-2`). For typical softmax usage in transformer attention
(`W` up to a few thousand) this is the dominant precision concern.

---

## Error Source Inventory

| # | Source | Location | Severity | Mitigation |
|---|--------|----------|----------|------------|
| 1 | FPU multiply `dy ⊙ y` (fidelity-sensitive) | `backward_softmax_compute.cpp:78` (`ckl::mul` pass 1) | Low — `HiFi4` consumes the full fp32 mantissa over 4 passes | math_fidelity = HiFi4 (hard-coded) |
| 2 | Accumulation of products into the lane sum | `backward_softmax_compute.cpp:80–81` (`ckl::accumulate_reduce_block`) | Moderate — depth = `W` (or `H`) elements per lane; can reach 1000s for transformer attention | fp32_dest_acc_en=True; intermediate `cb_sum` allocated as **Float32** (`grad_output.dtype`) so reload through L1 between blocks is lossless modulo float32 round |
| 3 | Tile-boundary repack between blocks | `accumulate_reduce_block` reloads `cb_sum` via `Accumulate::at(cb_sum, b)` for each non-first block | Low — `cb_sum` is Float32 (4 B/elem); the round-trip is fp32→fp32 in L1 | Float32 CB format (program descriptor lines 161–170) |
| 4 | Pack `dy ⊙ y` from fp32 dest to `cb_prod` | `ckl::mul` writes results to `cb_prod` (Float32) | Low — Float32 CB format avoids the bf16 truncation | `cb_prod` is Float32 (program descriptor lines 149–159) |
| 5 | bf16 scaler tile unpacked into fp32 reduce path | `cb_scaler` is bfloat16 (descriptor line 78); reduce LLK consumes it | Negligible — scaler is exactly representable (1.0) in bf16 | scaler value = 1.0; format reconfig handled by `BinaryDataFormatReconfig::INPUT_AND_OUTPUT` and reduce helper |
| 6 | **Catastrophic cancellation**: `sub<SCALAR>(dy, s)` when `dy_i ≈ s` | `backward_softmax_compute.cpp:90–94` | **High for adversarial inputs / numerically saturated softmax outputs**; otherwise moderate. fp32 dest preserves what precision is available, but no algorithmic guard exists (no Kahan, no rearrangement). | fp32_dest_acc_en=True keeps both operands in fp32 in dest. **No structural mitigation** — the algorithmic form `dy − s` is computed directly. |
| 7 | FPU multiply `y ⊙ (dy − s)` | `backward_softmax_compute.cpp:96` | Low — `y ∈ [0, 1]` does not amplify error; `HiFi4` consumes full fp32 mantissa | math_fidelity = HiFi4 |
| 8 | Pack `cb_grad_input` to DRAM | end of pass 2 mul | None — writer is fp32→fp32 (output dtype = fp32, `cb_grad_input` is Float32) | `cb_grad_input` Float32 (descriptor line 144) |
| 9 | Underflow on near-zero `y` | pass-2 mul, when `y_i ≈ 0` | None inherent — fp32 has subnormals; but `y * (dy − s)` may underflow for very small `y_i` | none required: underflow to 0 is the mathematically correct gradient when y_i = 0 |

---

## Accumulation Analysis

- **What is accumulated**: per-lane `s = sum_j(output_j * grad_output_j)`. One sum per row (`dim=-1`) or one sum per column (`dim=-2`).
- **Accumulation depth (elements)**:
  - `dim=-1`: `Wt × 32 = W` elements per row.
  - `dim=-2`: `Ht × 32 = H` elements per column.
- **Accumulation depth (tiles)**:
  - `dim=-1`: `Wt` tiles per row, processed as `NUM_BLOCKS = Wt / BLOCK_SIZE` blocks of `BLOCK_SIZE ≤ 8` tiles each.
  - `dim=-2`: `Ht` tiles per column, structured analogously.
- **Dest precision**: **fp32** (`fp32_dest_acc_en=True` set in the `ComputeConfigDescriptor`, `backward_softmax_program_descriptor.py:254–257`). Effective DST capacity = 4 tiles per acquire (half-sync + fp32 acc). The kernel-lib helpers consult `DEST_AUTO_LIMIT` and batch internally so `BLOCK_SIZE` up to 8 is safe.
- **Intermediate CB format**:
  - `cb_prod` (Pass-1 product): **Float32** (`grad_output.dtype`, 4 KB/page, 4 B/elem).
  - `cb_sum` (running accumulator): **Float32** (4 KB/page).
  - `cb_centered` (Pass-2 `dy − s`): **Float32** (4 KB/page).
- **UnpackToDestFp32 configured**: **No.** The program descriptor does not pass an `unpack_to_dest_mode` vector. `cb_sum` is reloaded for the second/third/... block via `Accumulate::at(cb_sum, b)`, which goes through the standard SrcA path. **Risk**: even though `cb_sum` is stored in Float32 in L1, the SrcA path can truncate to TF32 (10-bit mantissa) when unpacking back into dest. This is a real but typically minor precision leak between blocks; it is not catastrophic because each block contributes at most `BLOCK_SIZE × 32 ≤ 256` newly-summed terms before the next reload, and TF32 mantissa is still finer than bf16's 7 bits.
- **Round-trips through L1**: `NUM_BLOCKS − 1` reloads of `cb_sum` per lane (the first block does no reload). For typical transformer shapes (e.g. `Wt = 32`, `BLOCK_SIZE = 8`) this is `4 − 1 = 3` round-trips per lane.
- **Order of operations**: **sum-then-divide is N/A** — the formula has no division. The reduce scaler is exactly **1.0** (`SUM_AND_MAX_REDUCE_FACTOR=1`, scaler tile prepared by `calculate_and_prepare_reduce_scaler<…, PoolType::SUM, …>`). No `1/N` is applied at any point, so there is no divide-then-sum risk.
- **Assessment**: Accumulation depth scales with the reduce dimension and can be large (transformer attention often has `W` in the thousands). The Float32 dest + Float32 intermediate CB combination is the right choice. The lone gap is `UnpackToDestMode::UnpackToDestFp32` not being set on `cb_sum`, which means the inter-block reload may go through TF32. Given typical block counts (≤ ~64) this is a negligible vs catastrophic-cancellation issue further downstream.

---

## Numerical Guards

| Guard | Present | Location | Notes |
|-------|:-------:|----------|-------|
| Max subtraction before exp | N/A | — | No `exp_tile` in this op (forward softmax does this; the backward formula uses `y` directly). |
| ReLU clamp for approx exp | N/A | — | No exp. |
| Epsilon before reciprocal | N/A | — | No reciprocal / division. |
| Non-tile-aligned masking | ✗ | `backward_softmax.py:112–115` | The Python entry point **rejects** non-tile-aligned `H`/`W` with `ValueError`. This sidesteps the issue rather than masking it. The reader uses `calculate_and_prepare_reduce_scaler` (single-tile path), not `calculate_and_prepare_partial_reduce_scalers`. |
| Welford's algorithm | N/A | — | This is not a mean/variance op; Welford does not apply. |
| **Catastrophic-cancellation mitigation** for `dy − s` | ✗ | `backward_softmax_compute.cpp:90–94` | No reformulation, no Kahan correction. fp32 dest preserves available precision but cannot recover bits lost when `dy_i` and `s` are nearly equal. |
| `logical_shape()` vs `padded_shape()` | N/A — gated by tile-alignment validation | `backward_softmax.py:110–115` | Since H and W must be multiples of 32, `logical_shape == padded_shape`. |

---

## Math Fidelity Profile

| Compute phase | FPU/SFPU | Fidelity-sensitive | Default setting |
|--------------|----------|:-----------------:|-----------------|
| `mul(cb_grad_output, cb_output, cb_prod)` (pass 1) | FPU | **Yes** | HiFi4 |
| `accumulate_reduce_block<SUM>` (pass 1 reduce) | FPU | **Yes** | HiFi4 |
| `sub<SCALAR>(cb_grad_output, cb_sum, cb_centered)` (pass 2) | FPU (add/sub path) | No (eltwise add/sub is exact for representable inputs) | HiFi4 (irrelevant for sub) |
| `mul(cb_output, cb_centered, cb_grad_input)` (pass 2) | FPU | **Yes** | HiFi4 |

- **User-configurable**: **No.** The entry point `backward_softmax(grad_output, output, *, dim, memory_config)` does **not** accept `compute_kernel_config`. The compute config is hard-coded inside `create_program_descriptor` (`backward_softmax_program_descriptor.py:254–257`) as `MathFidelity.HiFi4`, `fp32_dest_acc_en=True`. `math_approx_mode` and `packer_l1_acc` are not specified (default to `True` and `False` respectively per `WormholeComputeKernelConfig`), but neither is exercised — there are no SFPU ops, and `packer_l1_acc` is moot because L1 acc does not work with `fp32_dest_acc_en=True` anyway.
- **Recommended minimum fidelity**: HiFi3 or HiFi4. Two FPU multiplies feed the result, and one of them feeds an accumulating reduce; LoFi/HiFi2 would silently drop precision below fp32. The current HiFi4 default is the conservative choice.
- **Hardware-bug note**: HiFi4 + `fp32_dest_acc_en` is reported on Wormhole B0 as bug #38306. The op uses both. If running on Wormhole and observing systematic precision regressions, HiFi3 + fp32 dest would be the safer combination — but this op does not let the caller select it.

---

## Tile-Boundary Precision

- **Tiles in reduction (per lane)**:
  - `dim=-1`: `Wt = W / 32` tiles.
  - `dim=-2`: `Ht = H / 32` tiles.
- **Block size**: `BLOCK_SIZE = pick_block_size(reduce_tiles)` — largest divisor of `reduce_tiles` that is ≤ 8 (default policy in `_pick_block_size`). So `BLOCK_SIZE ∈ {1, 2, 3, 4, 5, 6, 7, 8}` with `BLOCK_SIZE | reduce_tiles`.
- **`NUM_BLOCKS`** = `reduce_tiles / BLOCK_SIZE`.
- **Dest capacity**: 4 tiles per acquire (fp32 dest + half-sync). The helpers honor this internally via `DEST_AUTO_LIMIT`, batching `BLOCK_SIZE` up to 8 tiles across multiple acquire/release cycles when needed.
- **L1 round-trips per reduction (per lane)**: **`NUM_BLOCKS − 1`** reloads of `cb_sum`. The first block writes the fresh accumulator; every subsequent block reloads `cb_sum`, adds the new block's reduced contribution, and writes back.
- **Intermediate format**: All three intermediate CBs (`cb_prod`, `cb_sum`, `cb_centered`) are **Float32**. No bf16 truncation between phases.
- **Assessment**: Round-trip count is small (≤ `Wt/1 − 1` worst case if `BLOCK_SIZE=1`, typically ≤ ~16 for transformer-sized shapes). The accumulator stays in Float32 in L1, so each round-trip preserves all 23 mantissa bits modulo a single float32 round-to-nearest at pack time. The only soft spot is the missing `UnpackToDestMode::UnpackToDestFp32` on `cb_sum` — without it, the unpack on reload may go through SrcA at TF32 (10-bit mantissa) precision rather than full fp32. This is a small additional rounding error per block, but does not compound the way bf16 round-trips would.

---

## Configuration Exposure

| Setting | Exposed to user | Default | Recommendation |
|---------|:--------------:|---------|----------------|
| `fp32_dest_acc_en` | ✗ (hard-coded `True`) | True | Correct for this op given the reduction depth and the `(dy − s)` cancellation. Should remain True even if exposed. |
| `math_fidelity` | ✗ (hard-coded `HiFi4`) | HiFi4 | HiFi4 is conservative; HiFi3 is the reasonable tradeoff (matches WH Wormhole B0 recommendation when used with fp32 dest). HiFi2 would degrade the multiply precision below what fp32 inputs deserve. |
| `math_approx_mode` | ✗ (not specified) | True (struct default) | Irrelevant — no SFPU ops in this kernel. |
| `packer_l1_acc` | ✗ (not specified) | False (struct default) | Cannot be enabled together with `fp32_dest_acc_en=True` (hardware limitation, issue #28800). The op already avoids L1 acc by routing accumulation through `cb_sum` reload. |
| `UnpackToDestMode::UnpackToDestFp32` for `cb_sum` | ✗ (not configured at all) | `Default` for every CB | **Gap**: setting `UnpackToDestFp32` on `cb_sum` (and arguably `cb_prod` and `cb_centered`) would make the inter-block reload bypass SrcA's TF32 truncation. Currently small-magnitude error, but the op-design comment block claims "max-precision configuration" — this is the one knob still unused. |

---

## Key Observations

- **Precision configuration is conservative and well-matched to the math**: float32 inputs/outputs, fp32 dest accumulation, Float32 intermediate CBs (`cb_prod`, `cb_sum`, `cb_centered`), and HiFi4 fidelity together preserve fp32 precision through the multiply→sum→subtract→multiply pipeline. The op-design's "Phase-0 maximum-precision configuration" claim is broadly accurate.

- **The `(dy − s)` subtraction is the dominant precision risk and is unmitigated**. When `grad_output_i ≈ s` (e.g. in saturated-softmax regimes where one row entry of `y` dominates and `dy` aligns with that entry), the subtraction can lose most significant bits. The implementation does not reformulate, does not apply Kahan summation, and does not detect/clamp. fp32 dest is the only cushion. This is intrinsic to the chosen algorithmic form `output * (grad_output − sum(output * grad_output))` — the alternative form `output * grad_output − output * s` has the same cancellation. The genuinely cancellation-free form `sum_j (output_j * (grad_output_i − grad_output_j))` is `O(N²)` and not viable.

- **`UnpackToDestMode::UnpackToDestFp32` is not set on `cb_sum`** despite the program descriptor declaring `cb_sum` as Float32 and `fp32_dest_acc_en=True` being set on the compute config. This means the inter-block reload of the running sum may travel through SrcA's TF32 precision rather than full fp32. The error is small (one TF32 round per block boundary, `NUM_BLOCKS − 1` ≤ ~16 for typical shapes), but the configuration is inconsistent with the rest of the precision setup.

- **No precision configurability at the API surface**. `backward_softmax(grad_output, output, *, dim, memory_config)` does not accept a `compute_kernel_config` — callers cannot trade fidelity/throughput. This is consistent with Phase-0 scope but should be revisited if the op enters a perf-sensitive path. Notably, on Wormhole B0 the HiFi4 + fp32_dest_acc combination has a known bug (#38306); a config override (or default change to HiFi3) would let callers work around this.

- **Reduction accumulation is structurally safe**. Because (a) the scaler is exactly 1.0 (no `1/N` divide-then-sum), (b) each block's partial sum is added into a Float32 dest accumulator, and (c) the running sum lives in a Float32 CB across block boundaries, the running sum sees only float32-level rounding error per element. Even at `W = 4096` (`Wt = 128`, `NUM_BLOCKS = 16` with `BLOCK_SIZE = 8`) the cumulative accumulation error is well below the `(dy − s)` cancellation error for any non-trivial input. Accumulation is therefore *not* the limiting factor — the subtraction is.
