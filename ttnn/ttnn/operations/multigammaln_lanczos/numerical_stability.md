# Numerical Stability Analysis: multigammaln_lanczos (p = 4)

## Algorithm Summary

`ttnn.multigammaln_lanczos` computes the order-4 multivariate log-gamma function elementwise, using the **Lanczos 6-term polynomial** as the underlying `lgamma` approximation (NOT the Stirling-with-reflection variant used by `ttnn.multigammaln`):

$$\text{multigammaln\_lanczos}(a) = L(a) + L(a-0.5) + L(a-1.0) + L(a-1.5) + 3\ln\pi$$

For one scalar argument `y = a - offset`, the Lanczos sub-recipe is (kernel lines 25–37):

```
input  = y - 1                                          (= a - (offset + 1))
series = 1 + sum_{j=1..6} coef[j] / (input + j)
t      = input + 5.5                                    (= a - (offset - 4.5))
L(y)   = (input + 0.5) * log(t) + log(series)
       - input - 4.581061468643f                        // == log(sqrt(2π)) - 5.5
```

Pole zeroing at the integer poles `y == 1, y == 2` (where the polynomial form has `1/0` blowups but `lgamma` is exactly `0`) is implemented as a binary mask `mask = (a != offset+1) * (a != offset+2)` and multiplied onto the result — no input-value branching.

There is **no reduction**, **no matmul**, and **no cross-tile data dependency** — every output tile depends on exactly one input tile. Inputs and outputs are float32, tile layout. The operation is decomposed into two compute sub-phases per input tile:

1. **Sub-phase A x4 — per-offset Lanczos polynomial.** For each `offset ∈ {0.0, 0.5, 1.0, 1.5}`, run the Lanczos recipe and pack to a per-offset intermediate CB (`cb_lgamma_a`, `_half`, `_one`, `_three_halves`). The input tile is held (`cb_wait_front`, NOT popped) across all four invocations.
2. **Sub-phase B — 4-way sum + constant.** Loads the four intermediates into D0..D3, three `add_binary_tile` folds into D0, then `add_unary_tile` of `3·log(π) ≈ 3.434189657547f` (bit-cast `THREE_LOG_PI_BITS`). Packs to `cb_output_tiles`.

**Precision-sensitive phases** in this op (in approximate order of contribution to end-to-end error):

| Phase | Concern |
|-------|---------|
| `recip_tile(3)` of `(input + j)` near `j = 1..6` poles | Approximate-mode reciprocal (~3 ULP); `(input + j) → 0` when `a` approaches `(offset + 1 − j)` |
| Series sum `1 + Σ coef[j] / (input + j)` | The Lanczos coefficients alternate sign and have magnitudes up to `~86.5`; for `input ≈ 0..6` the partial sums grow then cancel back toward `1` — catastrophic cancellation candidate |
| `log_tile(2)` on the series | Approximate-mode `log`; if the series is small/near-1 the log derivative is large, amplifying upstream cancellation |
| `log_tile(1)` on `t = a - (offset - 4.5)` | Approximate-mode `log`; `t = 0` is reachable when `a = offset - 4.5` (an out-of-domain pole) |
| `(input + 0.5) * log(t)` SFPU multiply | Dominant magnitude term — but SFPU multiplies are full precision, fidelity-inert |
| Final 4-way sum + `3·log(π)` add | Trivial; 5 fp32 adds in DEST |

---

## Error Source Inventory

| # | Source | Location (multigammaln_lanczos_compute.cpp) | Severity | Mitigation |
|---|--------|---------------------------------------------|----------|------------|
| 1 | `copy_tile(cb_input_tiles, 0, …)` — repeated reload of the input tile (4 times in pole zeroing + 1 for input + 1 for input+0.5 + 1 for t + 6 in the series loop = **9 reloads per offset × 4 offsets = 36 reloads/tile**) | lines 142, 161, 186, 192, 215, 225 (per offset) | Low | `cb_input_tiles` is Float32 (`multigammaln_lanczos_program_descriptor.py:73`) AND configured `UnpackToDestMode::UnpackToDestFp32` (line 174–183). Bypasses SrcA/SrcB TF32 truncation. Bit-exact fp32→fp32. |
| 2 | `sub_unary_tile(1, K::SUB_FOR_T_BITS)` — `t = a - (offset - 4.5)` | line 145 | Low | fp32 binop_with_scalar in fp32 DEST. Exact for representable inputs. |
| 3 | `log_tile(1)` of `t` | line 148 | Moderate | SFPU log, `fast_and_approx=false` (template default from `compute_kernel_api.h:113`). **Precise log path.** No epsilon — if `a ≤ offset − 4.5` then `t ≤ 0` and `log` returns NaN/-inf (out-of-domain). |
| 4 | `fill_tile(2, 1.0f)` — series accumulator init | line 154 | None | Exact constant fill. |
| 5 | 6× series-term loop: `copy → sub_unary → recip → mul_unary → add_binary` (D2 accumulator) | lines 158–177 (macro), 172–177 (6 invocations per offset) | **High** in subdomain | (a) **`recip_tile(3)` is in approximate mode** by template (`recip.h:36–40`, `APPROX` is the global compute-kernel APPROX define). For `(input + j) → 0` (i.e., `a → offset + 1 − j`), reciprocal error blows up; no epsilon guard. (b) **Series cancellation**: 6 alternating-sign terms with magnitudes up to `~86.5`, summed in fp32 DEST. Cancellation depth ≤ 6 terms — bounded but the absolute magnitudes are well above `1`, so a few mantissa bits can be lost when the sum collapses back near 1 for large `a`. fp32 DEST (23-bit mantissa) keeps this manageable. |
| 6 | `mul_unary_tile(3, coef_bits)` — scalar multiply of `coef[j]` onto `1/(input+j)` | line 167 (inside macro) | Low | SFPU multiply — full precision, NOT FPU multiply; **`math_fidelity` does not affect it.** |
| 7 | `add_binary_tile(2, 3, 2)` — fold term into D2 series accumulator | line 169 (inside macro) | Low | fp32 add in DEST, exact for representable operands. Accumulator depth = 7 (1 init + 6 terms). |
| 8 | `log_tile(2)` of the series | line 183 | Moderate | SFPU log, default `fast_and_approx=false` (precise mode). Series ≥ 0 by construction on domain; near-zero series is reachable for inputs near the negative-half-line; no log-of-non-positive guard. |
| 9 | `mul_binary_tile(3, 1, 3)` — `(input + 0.5) · log(t)` | line 199 | Low | SFPU multiply, fp32 DEST. Full precision. Dominant magnitude term — but no fidelity exposure since it's SFPU. |
| 10 | `add_binary_tile(3, 2, 3)` — `+ log(series)` | line 203 | Low | fp32 add in DEST, exact. |
| 11 | `sub_binary_tile(3, 0, 3)` — `D3 -= input` | line 207 | Moderate | Catastrophic-cancellation candidate when `(input + 0.5) · log(t) + log(series) ≈ input` (i.e., asymptotic regime). For large `a`, both sides grow ≈ `a · log(a)` and `a`; the subtraction `≈ (a · log(a))` is well-conditioned, but for small/mid-range `a` the magnitudes can be close. fp32 DEST mitigates. |
| 12 | `sub_unary_tile(3, LOG_TWO_PI_HALF_MINUS_5_5_BITS)` — `D3 -= 4.581061468643` | line 211 | Low | Single fp32 scalar sub in DEST. The algebraic identity used here replaces `+log(sqrt(2π)) - t` with `-input - 4.581...`, which **avoids an extra DEST slot for `t`** (since `log_tile(1)` already overwrote D1 with `log(t)`). This is a stability win — it removes a separate `(log(sqrt(2π)) - t)` computation that would have introduced its own cancellation. |
| 13 | `unary_eq_tile(0, K::POLE_AT_ONE_BITS)` + `rsub_unary_tile(0, ONE_F_BITS)` + `mul_binary_tile(3, 0, 3)` — pole zero at `y == 1` | lines 215–222 | None | Exact equality on fp32. The mask is `{0.0, 1.0}`; `1.0 - mask` flips. Multiplication by 0.0 or 1.0 is exact. **One concern**: equality on float requires `a` to be **bitwise** equal to `offset + 1.0`. Any fp32 rounding upstream (none here — `a` enters the kernel directly from L1) means the mask only fires when `a` is exactly that pole. For inputs perturbed by epsilon, `1 / (input + j)` produces a very large but finite value rather than `∞`, so the pole-mask alone may not catch near-misses — but `torch.lgamma` itself returns `inf` at exact integer ≤ 0, so this behavior matches the spec at exact poles. |
| 14 | `unary_eq_tile(0, K::POLE_AT_TWO_BITS)` + `rsub_unary_tile` + `mul_binary_tile` — pole zero at `y == 2` | lines 225–232 | None | Same as #13. |
| 15 | `pack_tile(3, cb_out)` — pack per-offset L result to fp32 intermediate CB | line 238 | None | All four `cb_lgamma_*` CBs are Float32 (`multigammaln_lanczos_program_descriptor.py:108`, `data_format=input_tensor.dtype` which is fp32). fp32 DEST → fp32 CB is bit-exact. |
| 16 | Sub-phase B: 4× `copy_tile(cb_lgamma_*, 0, slot)` — reload intermediates | lines 261–268 | None | All four `cb_lgamma_*` CBs are configured `UnpackToDestMode::UnpackToDestFp32` (`multigammaln_lanczos_program_descriptor.py:174–183`). Bypasses SrcA/SrcB TF32-truncation path. Bit-exact fp32 reload. **This is the explicit precision lever called out in the program descriptor comment lines 168–172.** |
| 17 | Three sequential `add_binary_tile(0, n, 0)` for n=1,2,3 — 4-way sum into D0 | lines 271–274 | Low | All four terms already in fp32 DEST. 3 fp32 adds, accumulation depth 4 — negligible (~4·2⁻²³ relative ≈ 4.8e-7). |
| 18 | `add_unary_tile(0, THREE_LOG_PI_BITS)` — `+ 3·log(π)` | line 278 | Low | Single fp32 add of pre-baked IEEE-754 bit pattern. Exact. |
| 19 | `pack_tile(0, cb_output_tiles)` — final pack to fp32 output CB | line 283 | None | Output CB is fp32 (`data_format=output_tensor.dtype`, validated as fp32 in `multigammaln_lanczos.py:65–68`). fp32 → fp32 pack is bit-exact. |

---

## Accumulation Analysis

This operation has **no cross-tile reduction**. The only accumulations are intra-tile, in fp32 DEST:

- **What is accumulated**:
  1. The **series accumulator** D2 = `1 + Σ_{j=1..6} coef[j] / (input + j)` — 7 fp32 terms (1 init + 6 alternating-sign Lanczos coefficient terms), per offset, per output element.
  2. The **per-offset L assembly** — 4 fp32 ops: `D3 = (input+0.5)*log(t) + log(series) - input - 4.581...` (1 mul, 1 add, 2 subs).
  3. The **sub-phase B 4-way sum** — 5 fp32 terms (4 L(a − offset) + `3·log(π)` constant).
- **Total accumulation depth per output element**: ≤ 7 (series) + 4 (assembly) + 5 (final) = **16 fp32 adds/subs in DEST**, independent of tensor shape. Each occurs in fp32 DEST with the operands ranging from O(10⁻⁵) (smallest coef) to O(10²) (largest coef magnitude).
- **Dest precision**: **fp32** (`fp32_dest_acc_en=True`, `multigammaln_lanczos_program_descriptor.py:187`).
- **Intermediate CB format**: **Float32** for all six precision-sensitive CBs (`cb_input_tiles`, `cb_output_tiles`, `cb_lgamma_a`, `cb_lgamma_a_half`, `cb_lgamma_a_one`, `cb_lgamma_a_three_halves`) — set via `data_format=input_tensor.dtype` (fp32) at `multigammaln_lanczos_program_descriptor.py:73, 86, 108`.
- **UnpackToDestFp32 configured**: **Yes**, for all six CBs (`multigammaln_lanczos_program_descriptor.py:174–183`). This is the explicit fix relative to the Stirling cousin (`multigammaln`), which leaves `UnpackToDestMode::Default` (see `multigammaln/numerical_stability.md` §Tile-Boundary Precision).
- **Round-trips through L1**: **Exactly 1 per output tile** — sub-phase A packs 4 intermediate tiles to L1, sub-phase B reads all 4 back. The input tile makes 9 reads (per offset × 4 offsets = 36 reads total) from the same held L1 tile but never round-trips.
- **Order of operations**: Sum-only; no division (the only division-like op is `recip_tile` in the series, but the result is multiplied by `coef[j]`, not divided by). Not applicable.
- **Assessment**: Accumulation depth is shape-independent and bounded at 16 fp32 ops/element. With fp32 mantissa (23 bits) the worst-case rounding bound is ~16 · 2⁻²³ relative ≈ 1.9e-6, well below the upstream SFPU approximation errors (recip ~3 ULP, log precise ~1 ULP). The series cancellation in step 5 is the dominant absolute-error source — coefficients up to `~86.5` summing back toward 1 means ~`log₂(86.5) ≈ 7 bits` of cancellation room is consumed, leaving ~16 mantissa bits in the worst case, still ample.

---

## Numerical Guards

| Guard | Present | Location | Notes |
|-------|:-------:|----------|-------|
| Max subtraction before exp | n/a | — | Operation contains no `exp_tile`. |
| ReLU clamp for approx exp | n/a | — | Operation contains no `exp_tile`. |
| Epsilon before reciprocal | ✗ | line 165 (`recip_tile(3)`) | No `(input + j) + ε` guard. The six pole points for the series (j=1..6) are `a == offset + 1 − j ∈ {offset, offset−1, ..., offset−5}`. For `offset = 0`, that's `a ∈ {0, −1, −2, −3, −4, −5}`; for `offset = 1.5`, that's `a ∈ {1.5, 0.5, ..., −3.5}`. **Note**: The pole zeroing (steps 13–14) only masks `y == 1` and `y == 2` (i.e., `a == offset + 1` and `a == offset + 2`) — it does NOT mask the series internal `(input + j) == 0` poles at `a ∈ {offset+1−j}` for j=1..6. For inputs that hit those poles, the recip produces `+inf`, then `+inf · coef[j]` accumulates into the series, then `log(inf) = inf`, then propagates to NaN downstream. This matches `torch.lgamma` semantics at non-positive integers but is not what the pole-zero mask is for. |
| Non-tile-aligned masking | ✗ (and unnecessary) | `multigammaln_lanczos.py:81–84` | Entry point validates `shape[-1] % 32 == 0 and shape[-2] % 32 == 0` and raises `ValueError` otherwise. The operation is elementwise and aligned by contract; no per-tile masking is required. |
| Welford's algorithm | n/a | — | No mean/variance computation. |
| Pole zeroing at `y == 1, y == 2` | ✓ | lines 215–232 | `unary_eq_tile` + `rsub_unary_tile` + `mul_binary_tile`. Zeros L(y) at `a == offset + 1` and `a == offset + 2` where `lgamma(1) = lgamma(2) = 0`. Exact-equality test on fp32 — only fires for bitwise-exact poles; near-pole inputs are left to the polynomial. |
| Algebraic identity to skip un-logged `t` | ✓ | lines 36–37 of kernel comment; implemented at line 211 | Uses `L(y) = (input+0.5)·log(t) + log(series) − input − 4.581...` (with `4.581 = 5.5 − 0.918938...`) instead of `(input+0.5)·log(t) + log(series) + 0.918938... − t`. This avoids needing both `t` and `log(t)` in DEST simultaneously — only `log(t)` lives in D1 after `log_tile(1)`. **Stability win**: removes a `(log(sqrt(2π)) - t)` subtraction that would have introduced its own cancellation when `t ≈ log(sqrt(2π)) ≈ 0.92`. |
| Log-of-non-positive guard for `t` | ✗ | line 148 | `t = a − (offset − 4.5)`; if `a ≤ offset − 4.5` (deep negative for `offset = 0.0`, or `a ≤ −3.0` for `offset = 1.5`) then `t ≤ 0` and `log` returns NaN/-inf. No domain check. Matches `torch.lgamma` semantics — `lgamma` itself is undefined for `a ≤ 0`. |
| Log-of-non-positive guard for `series` | ✗ | line 183 | The series `1 + Σ coef[j]/(input+j)` can in principle go non-positive in deep negative subdomains. No guard. |

---

## Math Fidelity Profile

| Compute phase | FPU/SFPU | Fidelity-sensitive | Default / actual setting |
|--------------|----------|:-----------------:|--------------------------|
| `copy_tile()` (CB → DEST) | Unpacker → DEST | No (data movement) | n/a; routed direct to DEST via `UnpackToDestFp32` |
| `binop_with_scalar` family: `add_unary_tile`, `sub_unary_tile`, `mul_unary_tile`, `rsub_unary_tile` | SFPU | No (SFPU) | n/a |
| `fill_tile(2, 1.0f)` | SFPU | No | n/a |
| `recip_tile(3)` | SFPU | No — controlled by `math_approx_mode` not fidelity | **Approximate mode** (the `APPROX` template param at `recip.h:36–39` is the global compute-kernel APPROX define, sourced from `math_approx_mode` in the compute config; **not explicitly set in `ComputeConfigDescriptor`** so it falls to the platform default `true` for `WormholeComputeKernelConfig`). ~3 ULP. |
| `log_tile(1)` on `t` and `log_tile(2)` on series | SFPU | No — controlled by `math_approx_mode` | **Precise mode** at the second template arg level: `fast_and_approx = false` (the default of `log_tile<bool fast_and_approx = false>` at `compute_kernel_api.h:113`). But the first template arg `APPROX` (the global compute-kernel default) is still `true`. Note: per the LLK convention, `fast_and_approx` is the dominant precision selector for `log` — the **precise (non-fast-approx) `log` path is in effect.** |
| `add_binary_tile`, `sub_binary_tile`, `mul_binary_tile` (all from `eltwise_binary_sfpu.h`) | SFPU | No — SFPU multiply is full precision regardless of `math_fidelity` | n/a |
| `unary_eq_tile` | SFPU | No (comparison) | n/a |
| `pack_tile` (fp32 → fp32) | Packer | No (bit-exact for matched format) | n/a |

- **User-configurable**: **No.** The entry point `multigammaln_lanczos(input_tensor, *, memory_config=None)` (`multigammaln_lanczos.py:24–28`) does not accept a `compute_kernel_config` argument. `math_fidelity` and `fp32_dest_acc_en` are hard-coded in `multigammaln_lanczos_program_descriptor.py:186–188`.
- **Math fidelity in effect**: **`MathFidelity.HiFi4`** (4 FPU passes for any FPU multiply).
- **`math_approx_mode` in effect**: **Default `true`** (not explicitly set in `ComputeConfigDescriptor` — falls to the platform default per `WormholeComputeKernelConfig`). This means `recip_tile` is approximate (~3 ULP); `log_tile` is also approximate via the global `APPROX` flag, but the explicit `fast_and_approx=false` template default on `log_tile` selects the slower-but-more-accurate path.
- **`packer_l1_acc`**: Default `false` (not set). Correct — no L1 accumulation is used, and L1-acc does not work with `fp32_dest_acc_en=True` anyway.
- **Recommended minimum fidelity**: The compute kernel contains **zero FPU multiply ops** (`matmul_tiles`, `mul_tiles`, `reduce_tile`). All binary multiplies are SFPU (`mul_binary_tile`, `mul_unary_tile`), which are unaffected by `math_fidelity`. **HiFi4 is functionally inert here** — choosing LoFi would not change correctness. The choice is documented as defensive in `op_design.md:64`; in practice the actual precision lever for this op is `fp32_dest_acc_en` (engaged) plus `UnpackToDestFp32` on the intermediate CBs (engaged).

---

## Tile-Boundary Precision

- **Tiles in reduction**: 0 (no reduction). The relevant tile-boundary path is the round-trip from DEST → intermediate CB → DEST that occurs once per output element, per offset.
- **Dest capacity**: **4 tiles** (`fp32_dest_acc_en=True` + half-sync default — see reference §2.2 and `dest_helpers.hpp:88–102`). The per-offset Lanczos recipe uses exactly D0..D3 per `tile_regs_acquire`/`release` block; sub-phase B also uses exactly D0..D3 for the 4-way sum — both pack within budget.
- **L1 round-trips per output tile**: **1** — sub-phase A packs 4 intermediate tiles to L1, sub-phase B reads all 4 back. The input tile is held in `cb_input_tiles` (waited but not popped) across all four sub-phase-A invocations and re-`copy_tile`'d 9 times per offset (36 reads of the same L1 tile per output tile), but it never round-trips through pack.
- **Intermediate format**: **Float32** for all four `cb_lgamma_*` CBs (`multigammaln_lanczos_program_descriptor.py:108`).
- **UnpackToDestMode**: **`UnpackToDestFp32`** explicitly set on all six fp32 CBs (`multigammaln_lanczos_program_descriptor.py:174–183`). This bypasses the SrcA/SrcB TF32-truncation path on `copy_tile()` and routes the unpack directly to DEST in full fp32. **This is the explicit fix relative to the Stirling cousin (`multigammaln`)**, which leaves `UnpackToDestMode::Default` and exposes a 23→10-bit mantissa truncation on intermediate reloads.
- **Assessment**: With every fp32 CB declared Float32 AND `UnpackToDestFp32`, AND `fp32_dest_acc_en=True`, **the round-trip is bit-exact end-to-end**. No precision is lost at any tile boundary. This is the maximal precision configuration available on Wormhole for an elementwise fp32 kernel.

---

## Configuration Exposure

| Setting | Exposed to user | Default / hard-coded | Recommendation |
|---------|:--------------:|----------------------|----------------|
| `fp32_dest_acc_en` | ✗ | `True` (hard-coded at `multigammaln_lanczos_program_descriptor.py:187`) | Correct and necessary for fp32 inputs and the multi-step Lanczos recipe. Should remain on by default. |
| `math_fidelity` | ✗ | `HiFi4` (hard-coded at `multigammaln_lanczos_program_descriptor.py:186`) | HiFi4 is functionally inert for this kernel — no FPU multiply path is active. LoFi would not change correctness. The choice is defensive and harmless; matches the cousin's choice. |
| `math_approx_mode` | ✗ | Not set in `ComputeConfigDescriptor` ⇒ falls to platform default (`true` per `WormholeComputeKernelConfig`). | `recip_tile` runs in approximate mode (~3 ULP). `log_tile<false>` defaults force the precise log path regardless of this flag. The setting is implicit and not user-overridable. |
| `packer_l1_acc` | ✗ | Not set ⇒ default `false`. | Correct: no L1 accumulation is performed, and L1-acc does not work with `fp32_dest_acc_en=True` regardless (reference §2.6). |
| `unpack_to_dest_mode` | ✗ | `UnpackToDestFp32` on all six fp32 CBs (`multigammaln_lanczos_program_descriptor.py:174–183`); `Default` on all other CB indices. | **Correctly configured.** This is the key precision lever that distinguishes this op's stability from the Stirling cousin's. |
| `compute_kernel_config` (full struct) | ✗ | Not accepted by entry point | Phase-0 deliberately omits this for simplicity; the public API surface is `multigammaln_lanczos(input_tensor, *, memory_config=None)`. |
| `dtype` | ✗ | `float32` only (validated; raises `ValueError` otherwise — `multigammaln_lanczos.py:65–68`) | Phase-0 contract. The Lanczos recipe is designed for fp32 DEST — bf16 dest would lose mantissa to the alternating-sign cancellation in the series. |
| `p` (order) | ✗ | Hard-coded to 4 | Documented in `op_design.md` and the entry-point docstring (`multigammaln_lanczos.py:10`). |

---

## Key Observations

1. **Pure elementwise SFPU operation with no cross-tile reduction.** Most standard stability concerns (`Wt`-deep accumulation, divide-then-sum, tile-boundary precision loss in a multi-tile reduction) do not apply. The dominant numerical concerns are intra-tile: the alternating-sign series cancellation and the approximate-mode reciprocal at series poles.

2. **`UnpackToDestFp32` IS configured for all intermediate CBs — fixing the precision gap flagged in the Stirling cousin.** This is the most significant numerical-stability difference vs. `ttnn.multigammaln`. The cousin op leaves `UnpackToDestMode::Default` (see `multigammaln/numerical_stability.md` line 109), exposing a TF32-truncation path on `copy_tile()`. This op explicitly sets `UnpackToDestFp32` on every fp32 CB (`cb_input_tiles`, `cb_output_tiles`, and all four `cb_lgamma_*`) at `multigammaln_lanczos_program_descriptor.py:174–183`, with a comment block (lines 168–172) documenting why. **Result: the tile-boundary round-trip is bit-exact end-to-end in fp32.**

3. **The series-sum cancellation is the dominant intra-tile error source, but it's bounded.** Six alternating-sign Lanczos coefficients with magnitudes up to `~86.5` summed back toward `1` — consumes ~7 bits of fp32 mantissa worst-case, leaving ~16 mantissa bits of effective precision after the series. This is the structural difference vs. the Stirling cousin: Stirling's stability concerns are dominated by the **reflection branch** (`reflection_adj − res_stirling` near `x_off < 0.5`); Lanczos's are dominated by the **series alternating-sum cancellation**. Both stay within fp32 budget with `fp32_dest_acc_en=True`.

4. **`recip_tile` runs in approximate mode (~3 ULP), with no epsilon guard at the series poles.** For inputs `a ∈ {offset + 1 − j}` (j=1..6) the reciprocal produces `±inf`, which propagates through `coef[j] · ∞ → ∞`, then `log(∞) → ∞`, then NaN at the final assembly — matching `torch.lgamma` at non-positive integers but **NOT** the same as the explicit pole-zero mask. The mask only covers `y == 1, y == 2` (the points where `lgamma = 0`); the inner-series poles are intentionally left to propagate as inf/NaN. This is a deliberate domain choice, not an accuracy bug.

5. **HiFi4 fidelity is functionally inert.** The compute kernel contains **zero FPU multiply ops** — every multiply (`mul_unary_tile`, `mul_binary_tile`) is SFPU, which is unaffected by `math_fidelity`. The HiFi4 setting in `multigammaln_lanczos_program_descriptor.py:186` is defensive but does not contribute to the precision target. The two real precision levers — `fp32_dest_acc_en` and `UnpackToDestFp32` on intermediate CBs — are both engaged. The op_design.md "required for fp32 FPU precision" claim is the same overstatement as in the Stirling cousin: the actual fidelity-sensitive operations don't exist in this kernel.

6. **Algebraic identity removes a stability hazard.** The kernel uses `L(y) = (input + 0.5)·log(t) + log(series) − input − 4.581061468643f` (lines 36–37 of kernel comment, implemented at line 211) instead of the textbook `+log(sqrt(2π)) − t`. Beyond saving a DEST slot, this avoids a `(log(sqrt(2π)) − t)` subtraction that would have introduced its own cancellation when `t ≈ log(sqrt(2π)) ≈ 0.92` (i.e., when `a ≈ offset − 4.58`). The replacement subtraction `−input − 4.581...` is well-conditioned for all in-domain `a`.
