# Numerical Stability Analysis: atan_mean

Static analysis of the fused `torch.atan(x).mean(dim=-1)` kernel as configured in Phase 0 (float32 input, `fp32_dest_acc_en=True`, `MathFidelity::HiFi4`, H/W tile-aligned).

Source files inspected:
- `ttnn/ttnn/operations/atan_mean/atan_mean.py`
- `ttnn/ttnn/operations/atan_mean/atan_mean_program_descriptor.py`
- `ttnn/ttnn/operations/atan_mean/kernels/atan_mean_compute.cpp`
- `ttnn/ttnn/operations/atan_mean/kernels/atan_mean_reader.cpp`
- `ttnn/cpp/ttnn/kernel_lib/reduce_helpers_compute.{hpp,inl}` (the `reduce<>()` helper)
- `ttnn/cpp/ttnn/kernel_lib/reduce_helpers_dataflow.inl` (the scaler-prep helper)
- `ttnn/cpp/ttnn/kernel_lib/reduce_helpers_common.hpp` (matmul-path predicate)
- `tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/ckernel_sfpu_trigonometry.h` (`sfpu_atan`, `calculate_atan`)

## Algorithm Summary

For an input tensor `x` of logical shape `(N, C, H, W)` with `W` divisible by 32, the operation computes

```
y[n, c, h] = (1/W) * sum_{w=0..W-1} atan(x[n, c, h, w])
```

i.e., row-wise mean of `atan(x)` along the last axis. There is one output per `(n, c, h)`.

The compute kernel runs one "row-tile" at a time (one `(n, c, ht)` triple), consuming `Wt = W/32` input tiles and producing a single output tile (results land in column 0 of the output tile per the matmul-mode REDUCE_ROW convention; the trailing `1` dim is materialised as a tile with width 32, and is later squeezed metadata-only).

Each row-tile is two precision-sensitive phases:

1. **SFPU `atan`** — element-wise transcendental over `Wt` input tiles, written to an intermediate fp32 CB. Polynomial approximation, no FPU multiply involvement.
2. **Matmul-mode REDUCE_ROW with AVG** — `Wt × 32 = W` element accumulation per row. Because `pool_type == AVG && reduce_dim == REDUCE_ROW`, the kernel-lib helper dispatches to the **matmul path** (`reduce_uses_matmul()` in `reduce_helpers_common.hpp` returns `true`). The scaler tile in column 0 of SrcB is `1/W` in bf16, and each `matmul_tiles` invocation accumulates one fp32 input tile times the col-0 scaler into the fp32 dest register. Across `Wt` invocations under a single `tile_regs_acquire/commit/wait`, this produces `(1/W) * sum(atan(x))` per row.

Precision-sensitive phases:
- The atan polynomial itself (SFPU, fp32 dest path).
- Accumulation of `W` partial products in the matmul path (FPU, fidelity-sensitive).
- Per-tile multiply by a bf16-quantised `1/W` scaler (fidelity-sensitive; bf16 mantissa floor on the scaler value).
- Final pack from fp32 dest to the fp32 output CB (lossless for fp32→fp32).

## Error Source Inventory

| # | Source | Location | Severity | Mitigation |
|---|--------|----------|----------|-----------|
| 1 | SFPU `atan` polynomial approximation (9-term Sollya minimax in fp32 path) | `ckernel_sfpu_trigonometry.h` `sfpu_atan` (lines 319-371, fp32-dest branch lines 345-358); invoked from `atan_mean_compute.cpp:58` | Low — `atan` is bounded `|atan(x)| < π/2 ≈ 1.5708`, fit error ≤2⁻⁴⁰ relative on `[2⁻⁴⁰, 1]`, range reduction `atan(x) = π/2 − atan(1/x)` for `|x|>1` introduces one reciprocal + one subtract | fp32 polynomial branch auto-selected because `DST_ACCUM_MODE=true`; no extra `float_to_fp16b` rounding after evaluation (the bf16 round at `ckernel_sfpu_trigonometry.h:380` is gated on `!is_fp32_dest_acc_en`) |
| 2 | Catastrophic-cancellation risk in atan range-reduction (`π/2 − atan(1/\|x\|)` when `\|x\|→1⁺`) | `ckernel_sfpu_trigonometry.h:363` | Low — only loses precision very close to `\|x\|=1`, and the subtrahend `atan(1/\|x\|)` is itself fully evaluated in fp32 dest | Inherent to the SFPU implementation; not mitigated at the operation level |
| 3 | FPU multiply-accumulate of `W` terms in REDUCE_ROW matmul path | `reduce_helpers_compute.inl::reduce_matmul_tiles` (lines 37-41, `llk_math_matmul<REDUCE_MATMUL_FIDELITY, …>`); driven from the `wt` loop at lines 372-401 of the same file | Moderate — depth = `W` elements per row, but accumulation is entirely in fp32 dest with HiFi4 (4-pass FPU) | `fp32_dest_acc_en=True` (program descriptor line 185); `math_fidelity=HiFi4` (line 184); single `tile_regs_acquire` spans all `Wt` matmul calls, so the running sum never round-trips L1 |
| 4 | bfloat16 quantisation of the `1/W` scaler | reader: `dataflow_kernel_lib::calculate_and_prepare_reduce_scaler<cb_scaler, AVG, REDUCE_ROW, W>()` (`atan_mean_reader.cpp:39-40`); scaler value `1.0f / W` at `reduce_helpers_dataflow.inl:313`; CB format `ttnn.bfloat16` (`atan_mean_program_descriptor.py:89`) | Low–moderate — `1/W` is rounded to 7-mantissa-bit bf16 before being broadcast through SrcB column 0; max relative error ≈ 2⁻⁸; this same rounded value multiplies every `atan(x)` term, so the error is a single multiplicative scale on the row mean, not an accumulating bias | None at the operation level (scaler CB format is hard-coded bf16). Acceptable because `atan` output range is bounded and the per-row multiplier is identical. |
| 5 | Pack-stage rounding `fp32 dest → fp32 cb_atan_tiles` | `sfpu_atan` helper packing the `atan` result into `cb_atan_tiles` (descriptor format `input_tensor.dtype = fp32`, `atan_mean_program_descriptor.py:117`) | None — fp32→fp32 pack is lossless | fp32 intermediate CB (`cb_atan_tiles` allocated with `data_format=input_tensor.dtype` which is fp32) |
| 6 | Unpack `cb_atan_tiles → SrcA` for matmul-mode REDUCE_ROW input | Implicit in `reduce_matmul_tiles` (`reduce_helpers_compute.inl:39`), driven by SrcA format set by `reconfig_data_format(scaler_cb, input_cb)` at line 232 (matmul mode places scaler in SrcA, input in SrcB at init, then ordinary matmul uses SrcA for left operand) | Possible — SrcA path for the input-CB side traditionally truncates fp32 to TF32 (19 mantissa bits) unless `UnpackToDestFp32` is configured | **NOT mitigated.** The program descriptor does not set `UnpackToDestMode::UnpackToDestFp32` on `cb_atan_tiles`, and matmul-path inputs do not go via `copy_tile()` (which is what `UnpackToDestFp32` enables). The standard SrcA/SrcB matmul path is used, which is the documented LLK behavior; precision loss here is at most TF32 mantissa truncation per operand before the 4-pass HiFi4 multiplier consumes the full 19 bits — bounded and well-defined. |
| 7 | Pack `fp32 dest → cb_output_tiles` after reduce | Inside the `reduce<>()` helper at `reduce_helpers_compute.inl:316`; output CB format `output_tensor.dtype = fp32` (`atan_mean_program_descriptor.py:101`) | None — fp32→fp32 pack is lossless | fp32 output CB |
| 8 | Non-tile-aligned handling | Entry point validates `W % 32 == 0` and `H % 32 == 0` (`atan_mean.py:80-84`) | N/A in Phase 0 | Hard-error in the entry point; no partial-scaler or mask path is wired up. |

## Accumulation Analysis

- **What is accumulated**: row-wise sum of `atan(x_w)` weighted by a per-element `1/W` scaler, fused into a matmul-mode REDUCE_ROW.
- **Accumulation depth**: `W = Wt × 32` elements per output, where `Wt` is the compile-time number of input tiles along the row.
- **Dest precision**: **fp32** (`fp32_dest_acc_en=True` in `ComputeConfigDescriptor`).
- **Intermediate CB format**: `cb_atan_tiles` is **Float32** (taken from `input_tensor.dtype`, validated to be fp32 in the entry point).
- **UnpackToDestFp32 configured**: **No.** No `unpack_to_dest_mode` override is created in the program descriptor. The reduce uses the matmul SrcA/SrcB path (not `copy_tile()`), so `UnpackToDestFp32` would not change the matmul ingress behavior. The unpack into SrcA for `cb_atan_tiles` therefore follows the standard LLK matmul ingress (TF32 internal width on Wormhole B0).
- **Round-trips through L1**: **0 round-trips per row.** The compute kernel issues a single `tile_regs_acquire/commit/wait/release` cycle inside `reduce<>()` covering all `Wt` `reduce_matmul_tiles` calls for one row (`reduce_helpers_compute.inl:365-405`, REDUCE_ROW path). The running accumulator therefore lives in fp32 dest for the full `W`-element sum and is only packed to L1 once, into the fp32 output CB.
  - There is one prior round-trip per row through `cb_atan_tiles` (sfpu_atan pack → matmul unpack), but that CB is fp32, so the round-trip is precision-preserving up to the SrcA TF32 width on the matmul ingress.
- **Order of operations**: The matmul col-0 scaler encodes the `1/W` multiplier into every multiply-accumulate step. Algebraically this is "divide-then-sum" (each multiply by `1/W` happens before its add into dest), but because each per-term multiply uses the fp32 dest accumulator and the scaler is a single rounded constant, this is numerically equivalent to a single-precision sum-then-divide with one extra multiplicative rounding from the bf16 scaler. There is no underflow risk for typical W (≤ 2¹⁶) because `atan(x) / W` stays well within fp32 normal range.
- **Assessment**: Strong. The full `W`-wide sum is performed in fp32 dest with HiFi4 fidelity and never spills to L1. The only meaningful precision floor is the bf16 quantisation of `1/W` in the col-0 scaler and the TF32 ingress width on SrcA for the post-atan tiles — neither accumulates.

## Numerical Guards

| Guard | Present | Location | Notes |
|-------|:-------:|----------|-------|
| Max subtraction before exp | N/A | — | No `exp` in the computation |
| ReLU clamp for approx exp | N/A | — | No `exp` in the computation |
| Epsilon before reciprocal | N/A | — | No `recip` in the user-visible flow; `sfpu_atan` does call `sfpu_reciprocal<false>` for the `|x|>1` range reduction, but only on `t0 = |x| > 1`, so the denominator is bounded ≥ 1 and no epsilon is needed |
| NaN propagation in atan | ✓ | `ckernel_sfpu_trigonometry.h:325-327` | `sfpu_atan` explicitly detects NaN inputs (`exponent == 255 && mantissa != 0`) and emits `quiet_NaN`. Non-finite inputs propagate correctly. |
| Non-tile-aligned masking (partial scaler / mask tile) | ✗ | `atan_mean.py:80-84` rejects | Phase 0 hard-validates `W % 32 == 0`; no partial-scaler path exists. The kernel-lib `reduce<>()` and `calculate_and_prepare_reduce_scaler` both support partial scalers (`ReducePartialScaler::last_tile_at(...)`, `prepare_partial_reduce_scalers`), but neither is wired up. |
| Welford's algorithm | ✗ | — | Not applicable; only mean is computed (no variance). The single-pass mean here has the same stability profile as Welford's for a mean alone (Welford only differentiates itself when variance is also produced). |
| Logical-vs-padded shape for `N` in `1/N` | ✓ | `atan_mean_program_descriptor.py:39` reads `N, C, H, W = input_shape` (from `input_tensor.shape`, which is the logical shape); `W` is then forwarded as the `reduce_factor` template parameter to `calculate_and_prepare_reduce_scaler` so the scaler is `1/W_logical` | Correct because Phase 0 forces `W_logical == W_padded`. If Phase 1 ever relaxes the `% 32 == 0` constraint, this must switch to `logical_shape()` + a partial scaler. |

## Math Fidelity Profile

| Compute phase | FPU/SFPU | Fidelity-sensitive | Default setting |
|---------------|----------|:------------------:|-----------------|
| SFPU `atan` polynomial (`calculate_atan`) | SFPU | No (controlled by `APPROXIMATION_MODE` / `is_fp32_dest_acc_en` template params, not `math_fidelity`) | Phase 0: fp32-dest branch (9-term Sollya minimax). `APPROX=true` is passed via `SFPU_INIT_KERNEL_CALL(atan, …, true)` but `sfpu_atan` does not branch on `APPROXIMATION_MODE` for the polynomial choice — only `is_fp32_dest_acc_en` matters. |
| Reciprocal inside `sfpu_atan` (range reduction for `|x|>1`) | SFPU | No | `sfpu_reciprocal<false>` (precise, 2 Newton-Raphson iterations, ≤1 ULP for fp32). Fixed at SFPU-helper level; not user-controllable. |
| Matmul-mode REDUCE_ROW (every `reduce_matmul_tiles` call) | FPU | **Yes** | Phase 0: **HiFi4** (4 passes, 1 TFLOPS matmul throughput; full mantissa precision on a 5×7 multiplier). Set in `ComputeConfigDescriptor(math_fidelity=ttnn.MathFidelity.HiFi4, …)` in `atan_mean_program_descriptor.py:184`. |

- **User-configurable**: **No.** The entry-point function `atan_mean(input_tensor)` accepts no `compute_kernel_config` parameter (`atan_mean.py:20`). Both `math_fidelity` and `fp32_dest_acc_en` are hard-coded in `create_program_descriptor`. `math_approx_mode` and `packer_l1_acc` are not set on the `ComputeConfigDescriptor` at all (defaults inherited from the descriptor — likely `approx_mode=true`, `packer_l1_acc=false`).
- **Recommended minimum fidelity for this op**: HiFi2 would still give acceptable accuracy for atan-mean (atan output is bounded and the bf16 scaler is the dominant precision floor anyway). HiFi4 is conservative and matches what op_design.md commits to in Phase 0.

## Tile-Boundary Precision

- **Tiles in reduction**: `Wt = W / 32` input tiles per output row; `N × C × H / 32` total rows across the tensor.
- **Dest capacity** (Wormhole, fp32-dest, half-sync — default for compute kernels): **4 tiles**. The reduce helper only ever needs a single dest slot per row (the running accumulator at `dst_idx=0`), so dest pressure is `1/4` of capacity — never a bottleneck for this op.
- **L1 round-trips per reduction**: **0 round-trips of the accumulator.** The `Wt` `reduce_matmul_tiles` calls live inside one `tile_regs_acquire/commit/wait/release` block (see `reduce_helpers_compute.inl` REDUCE_ROW path lines 365-417), so the row accumulator stays in fp32 dest from the first multiply-accumulate to the final `pack_tile` of the row result.
- **Intermediate CB format used by the reduce input** (`cb_atan_tiles`): **Float32**.
- **Intermediate CB format of the scaler** (`cb_scaler`): **Float16_b (bfloat16)** — fixed by the matmul col-0 scaler convention in `prepare_reduce_scaler` (the dataflow helper only supports `Float16_b` and `Float32` for matmul-mode reduce; the program descriptor picked bf16).
- **`UnpackToDestFp32` configured for `cb_atan_tiles`**: No. This would only matter if the reduce path used `copy_tile()` to reload an accumulator from L1; the matmul REDUCE_ROW path here does not — the accumulator never leaves dest within a row.
- **Assessment**: Excellent for a row-mean of this depth. The only precision floor inside the inner loop is (a) HiFi4 multiplier width and (b) bf16 quantisation of the single `1/W` scaler that is held column-broadcast in SrcB. Both are constant w.r.t. `W`; the per-row error does not grow with the reduction depth.

## Configuration Exposure

| Setting | Exposed to user | Default (effective) | Recommendation |
|---------|:--------------:|----------------------|----------------|
| `fp32_dest_acc_en` | ✗ | `True` (hard-coded, `atan_mean_program_descriptor.py:185`) | Keep `True` for accuracy. If a future "fast" variant is wanted, this is the most natural lever — and would need to be paired with a downgrade to `cb_atan_tiles` of `Float16_b` since the matmul ingress already truncates to TF32 on SrcA. |
| `math_fidelity` | ✗ | `HiFi4` (hard-coded, `atan_mean_program_descriptor.py:184`) | Acceptable. HiFi3 would likely match accuracy at higher throughput (4-pass HiFi4 vs 3-pass HiFi3), but Phase 0 spec calls for HiFi4. Note: the reference guide flags a Wormhole-B0 HW issue (#38306) where `HiFi4 + fp32_dest_acc_en` may produce incorrect results, recommending HiFi3 with fp32 accum on WH instead — this combination is exactly what Phase 0 uses, so verify the test suite catches any such regression. |
| `math_approx_mode` | ✗ | Inherits `ComputeConfigDescriptor` default (effectively unused for atan; `sfpu_atan` polynomial selection keys off `is_fp32_dest_acc_en`, not `APPROXIMATION_MODE`) | No action needed at the op level. |
| `packer_l1_acc` | ✗ | Inherits `ComputeConfigDescriptor` default (likely `false`) | Not applicable — the algorithm has no use for packer L1 accumulation since each output row is produced exactly once. Moreover, packer L1 accumulation is incompatible with `fp32_dest_acc_en=True` (reference §2.6), so it must remain off. |
| `compute_kernel_config` parameter on `atan_mean(...)` | ✗ | Not in function signature (`atan_mean.py:20`) | Acceptable for a Phase 0 fused op. If this graduates beyond Phase 0, expose the standard `compute_kernel_config` parameter so callers can choose precision/throughput tradeoffs. |

## Key Observations

- **Stability profile is intentionally strong.** Phase 0 pins the maximum-precision combination available on Wormhole/Blackhole: `fp32_dest_acc_en=True`, `math_fidelity=HiFi4`, fp32 input CB, fp32 intermediate (`cb_atan_tiles`), fp32 output CB. The full `W`-wide row accumulation lives in fp32 dest with no L1 round-trips of the accumulator — a near-best-case profile for a row mean.
- **The dominant precision floor is the bf16 `1/W` scaler, not the accumulator.** Because AVG/REDUCE_ROW takes the matmul path (`reduce_uses_matmul()` → true), `1/W` is encoded into column 0 of a bf16 scaler tile and broadcast through SrcB into every matmul-accumulate step. The bf16 round of `1/W` is a multiplicative scale on the entire row mean (max ~2⁻⁸ relative bias), not an accumulating term — its impact does **not** grow with `W`.
- **No `UnpackToDestMode::UnpackToDestFp32` override on `cb_atan_tiles`.** This is fine for the current code path because the reduce uses matmul (SrcA/SrcB ingress, not `copy_tile()`), where matmul ingress truncation to TF32 (19 mantissa bits) on SrcA is the LLK contract. It is, however, a latent correctness consideration: if anyone refactors the reduce to use the non-matmul reduce path with an in-CB accumulator (e.g., to support partial scalers in Phase 1+), the lack of `UnpackToDestFp32` on `cb_atan_tiles` would silently lose precision on accumulator reloads.
- **No user knob for precision/throughput tradeoff.** The entry point hard-codes both `math_fidelity` and `fp32_dest_acc_en` and accepts no `compute_kernel_config`. This is consistent with the Phase 0 spec but means any callers needing higher throughput must wait for a Phase 1 API change. The implementation already cleanly passes through the kernel-lib helpers, so plumbing `compute_kernel_config` later is straightforward.
- **Phase 0 hits the `HiFi4 + fp32_dest_acc_en` combination flagged by the reference (Wormhole #38306).** The reference document warns this combination can produce incorrect results on Wormhole B0 and recommends HiFi3 with fp32 accumulation instead. Currently the op_design and program descriptor commit to HiFi4. Whether this materially matters for matmul-mode REDUCE_ROW on the current LLK version should be confirmed by the existing acceptance suite (the implementer log indicates 20/20 acceptance pass), but it is worth recording as a known sensitivity in case future hardware revisions or LLK refactors expose it.
- **Non-tile-aligned shapes are out of scope for Phase 0.** The entry point hard-validates `W % 32 == 0` and `H % 32 == 0`, so the well-supported partial-scaler path in `reduce_helpers_*` is unused here. When Phase 1 lifts the alignment requirement, the natural fix is to switch the reader to `calculate_and_prepare_partial_reduce_scalers<…, valid_w_in_last_tile, W, false>()` and the compute kernel to `ReducePartialScaler::last_tile_at(1)`, with `W` in the scaler computation read from the **logical** shape (`logical_shape()[-1]`).
