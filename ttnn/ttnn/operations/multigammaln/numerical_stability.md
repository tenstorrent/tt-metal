# Numerical Stability Analysis: multigammaln (p = 4)

## Algorithm Summary

`ttnn.multigammaln` computes the order-4 multivariate log-gamma function elementwise:

$$\text{multigammaln}(a) = \ln\Gamma(a) + \ln\Gamma(a - 0.5) + \ln\Gamma(a - 1.0) + \ln\Gamma(a - 1.5) + 3\ln\pi$$

There is **no reduction**, **no matmul**, and **no cross-tile data dependency** — every output tile depends on exactly one input tile. Inputs and outputs are float32, tile layout.

The operation is decomposed into two compute sub-phases per input tile:

1. **Sub-phase A — four inlined lgamma recipes.** For each `offset ∈ {0.0, 0.5, 1.0, 1.5}`, the kernel runs the line-for-line copy of the fp32 reference `lgamma_kernel.cpp` (Stirling with Taylor bridges near `z=1` and `z=2`, followed by the reflection adjustment for inputs `< 0.5`). The result of each offset is packed to its own intermediate CB (`cb_lgamma_a`, `cb_lgamma_a_half`, `cb_lgamma_a_one`, `cb_lgamma_a_three_halves`).
2. **Sub-phase B — 4-way add + constant.** Loads the four intermediate tiles into D0..D3, performs three `add_binary_tile` folds into D0, then adds the compile-time constant `3·log(π) ≈ 3.434189657547f` (encoded as fp32 bit pattern `0x405BA32E`) via `add_unary_tile`. Packs to `cb_output_tiles`.

**Precision-sensitive phases** (in order of likely contribution to end-to-end error):

| Phase | Concern |
|-------|---------|
| Stirling Bernoulli polynomial / Taylor bridge | Approximate `_sfpu_reciprocal_<2>()` in `1/z` evaluation; polynomial conditioning near branch joins (`abs(z - 1) = 0.25`, `abs(z - 2) = 0.25`, `abs(z - 0.5) = 0.01`) |
| `frac(x_off)` for large `x_off` | `frac` loses absolute precision as `\|x\|` grows — the fractional bits get pushed out of the fp32 mantissa |
| `sin(π · frac(x_off))` near integers | Approximation error of `sin` near zero amplifies `log\|sin\|` |
| `log\|sin(π · frac(x_off))\|` | The `log` near 0 has unbounded derivative; combined with `sin`-approx error, the reflection term has high relative error |
| `reflection_adj − res_stirling` (`x_off < 0.5` branch) | Subtraction of two values of similar magnitude — catastrophic-cancellation candidate |
| 4-way Sfpu-add of four lgamma terms | Trivial — only 4 terms, all in fp32 DEST |

---

## Error Source Inventory

| # | Source | Location (multigammaln_compute.cpp) | Severity | Mitigation |
|---|--------|-------------------------------------|----------|------------|
| 1 | Pack of intermediate lgamma tile to L1 (one per offset) | line 184 — `pack_tile(0, cb_out)` | Low | Intermediate CBs are **Float32** (`multigammaln_program_descriptor.py:105–108`, `data_format=input_tensor.dtype` which is fp32). DEST is fp32. Pack is fp32→fp32 ⇒ bit-exact. |
| 2 | Unpack of intermediate lgamma tile from L1 to DEST | lines 221–227 — four `copy_tile()` from `cb_lgamma_*` | Low | Default `UnpackToDestMode::Default` is used (not `UnpackToDestFp32`). With fp32 source CB and fp32 DEST, unpack-via-SrcA path is the default; precision exposure depends on the LLK unpack-fp32 path. See *Tile-Boundary Precision* below. |
| 3 | Approximate reciprocal `_sfpu_reciprocal_<2>(z)` inside Stirling | `ckernel_sfpu_lgamma.h:108` (called from `lgamma_stirling_float_tile` at line 119, 179) | Moderate | Iteration count **2** (template arg `<2>`) — Newton-Raphson 2-iter, ~1 ULP in fp32. Not the LoFi 0-iter form. Fixed at the LLK level; not user-configurable. |
| 4 | `frac_tile(1)` of `x_off` | line 134 | Moderate→High for large `\|x\|` | None. As `\|x_off\|` grows past ~`2^23 ≈ 8.4e6`, fp32 cannot resolve the unit place; `frac` returns 0 and the reflection log-sin path collapses. Domain note in the kernel comment (lines 28–33) accepts this. |
| 5 | `sin_tile(1)` at `π·frac(x_off)` | line 139 | Moderate | SFPU `sin` (approximate by default — no `_init<...>` template arg given). Worst near zeros (i.e., when `frac(x_off) → 0` or `→ 1`), which is exactly where the reflection branch is taken. |
| 6 | `log_tile(1)` of `\|sin(π·frac)\|` (default mode = approx) | line 167 | Moderate | Approximate log. Output magnitude is unbounded near zero — small errors in `sin` are *amplified* by the log derivative `1/sin`. |
| 7 | Subtraction `reflection_adj − res_stirling` inside `lgamma_adjusted_tile` (for `x_off < 0.5`) | `ckernel_sfpu_lgamma.h:147` | High (in subset of domain) | Catastrophic cancellation when `reflection_adj ≈ res_stirling`. Done in fp32 DEST (mantissa = 23 bits), so up to ~7 decimal digits of relative precision survive — usable but data-dependent. No additional mitigation present. |
| 8 | Polynomial branch joins at `abs(d1) = 0.25` / `abs(d2) = 0.25` | `ckernel_sfpu_lgamma.h:86, 96` | Low | The Taylor and Stirling-Bernoulli forms agree well at the 0.25 cutoff by construction (fixed at LLK level; designed for 0–3 ULP target — see header comment). |
| 9 | `lgamma_stirling_float_tile(0, 1, 0)` — the `(z - 0.5) · log(z)` FPU/SFPU multiply | `ckernel_sfpu_lgamma.h:107` | Low (single multiply, fp32 DEST) | Math fidelity = HiFi4. Note: this multiply is SFPU-internal (sfpi `*` op on `vFloat`), **not** FPU; HiFi4 does not directly affect it — SFPU multiplies are full-precision regardless of `math_fidelity`. |
| 10 | Three sequential `add_binary_tile` folds D0 = D0 + Dn for n=1,2,3 | lines 231–233 | Low | All terms already in fp32 DEST. 3 additions of 4 fp32 terms — accumulation depth 4, fp32 mantissa — negligible. |
| 11 | `add_unary_tile(0, 3·log(π))` for the constant offset | line 237 | Low | Scalar passed as IEEE-754 bit pattern (`0x405BA32E`). Single fp32 add in DEST. |
| 12 | Final `pack_tile(0, cb_output_tiles)` to fp32 output CB | line 242 | Low | Output CB is fp32 (`data_format=output_tensor.dtype`, validated as fp32 in `multigammaln.py:59–62`). fp32→fp32 pack ⇒ bit-exact. |
| 13 | Subtle: `init_sfpu(cb_input_tiles, cb_output_tiles)` called once, but four intermediate CBs are written | line 195 + lines 184 (four invocations) | Low | All five output CBs have identical format (fp32, fp32, fp32, fp32, fp32). No reconfig needed; the kernel comments call this out (`op_design.md` line 258). |
| 14 | Subtraction of large input from constant 0.5 in reflection-mask: `(x_off - 0.5) < 0`  | line 99 (`sub_binary_tile(1,2,1)`) | Low | Used only for a comparison (`ltz_tile`). Sign-bit-only extraction — magnitude loss in subtraction does not affect the mask outcome. |

---

## Accumulation Analysis

This operation has **no reduction**; the only accumulation is the 4-way add in sub-phase B.

- **What is accumulated**: The four scalar `lgamma(x - offset)` terms plus the constant `3·log(π)`.
- **Accumulation depth**: **5 fp32 terms per output element** (4 lgamma terms + 1 constant). Independent of tensor shape.
- **Dest precision**: **fp32** (`fp32_dest_acc_en=True`).
- **Intermediate CB format**: **Float32** for all four `cb_lgamma_*` CBs (set explicitly to `input_tensor.dtype = ttnn.float32` in `multigammaln_program_descriptor.py:107`).
- **UnpackToDestFp32 configured**: **No** — `UnpackToDestMode` is left at `Default` (the descriptor does not pass an `unpack_to_dest_mode` override). See *Tile-Boundary Precision* for the implication.
- **Round-trips through L1**: **Exactly 1 per output tile** — sub-phase A packs each of the 4 intermediate tiles to L1 once; sub-phase B reads all four back to DEST once. No multi-tile loop spans the round-trip.
- **Order of operations**: Sum-only (no division). Not applicable.
- **Assessment**: Accumulation error is negligible. With 4 terms in fp32 DEST (23-bit mantissa), the worst-case bound is `~4 · 2⁻²³` relative = ~4.8e-7 — well below the precision of the upstream Stirling/reflection approximations (which target 0-3 ULP per the LLK header comment).

---

## Numerical Guards

| Guard | Present | Location | Notes |
|-------|:-------:|----------|-------|
| Max subtraction before exp | n/a | — | Operation contains no `exp_tile`. |
| ReLU clamp for approx exp | n/a | — | Operation contains no `exp_tile`. |
| Epsilon before reciprocal | ✗ | `ckernel_sfpu_lgamma.h:108` | `_sfpu_reciprocal_<2>(z)` is called with `z` directly. For `x_off ∈ (-0.5, 0.5)` the reflection moves `z = 1 - x_off ∈ (0.5, 1.5)`, so `z` is bounded away from zero by construction. **No explicit ε is needed** because the reflection guarantees `z ≥ 0.5` for all reachable `x_off`. The boundary at `z = 0.5` is handled by an explicit `v_if(abs(z - 0.5) < 0.01)` clamp to `LOG_SQRT_PI` (`ckernel_sfpu_lgamma.h:117`). |
| Non-tile-aligned masking | ✗ (and unnecessary) | `multigammaln.py:75–78` | Entry point validates `shape[-1] % 32 == 0 and shape[-2] % 32 == 0` and raises `ValueError` otherwise. The operation is elementwise and aligned by contract, so no per-tile masking is required. |
| Welford's algorithm | n/a | — | No mean/variance computation. |
| Integer-mask before `log\|sin\|` | ✓ | lines 154–161 of compute kernel; `ckernel_sfpu_lgamma.h:138–148` | Detects `floor(x_off) == x_off`, zeros out `sin(π·frac)` for that mask, then takes `abs` then `log`. `log(0) = -∞` is intentional — propagates to NaN downstream via `reflection_adj - res_stirling = +∞ - finite = +∞`, then summed with the other lgamma terms to produce torch-compatible NaN/inf at integer arguments. |
| Polynomial bridges near `z = 1`, `z = 2` | ✓ | `ckernel_sfpu_lgamma.h:86–104` | 5-term Taylor expansions inserted to avoid the Stirling form's cancellation near `(z - 0.5)·log(z) ≈ z - log(sqrt(2π))` when `log z` is small. |
| Boundary clamp at `z ≈ 0.5` | ✓ | `ckernel_sfpu_lgamma.h:117–118` | Returns `LOG_SQRT_PI` directly when `\|z − 0.5\| < 0.01`. |
| Output forcing `+inf` at `\|x\| = ∞` | ✓ (fp32-acc path only) | `ckernel_sfpu_lgamma.h:152–157` | When `is_fp32_dest_acc_en=true` and the input is `±∞` (exp == 128, mantissa == 0), `lgamma_adjusted_tile` writes `+∞` explicitly. This path is active here (config has `fp32_dest_acc_en=True`). |
| Domain protection for `a ≤ 1.5` | ✗ (by design) | — | The kernel does NOT branch on `a`. For `a ∈ (-∞, 1.5]`, one or more lgamma arguments are non-positive; NaN propagates via `log` of a negative or zero `\|sin\|`. Documented in compute kernel comments (lines 25–33). |

---

## Math Fidelity Profile

| Compute phase | FPU/SFPU | Fidelity-sensitive | Default setting |
|--------------|----------|:-----------------:|-----------------|
| `copy_tile()` (CB → DEST) | Unpacker → DEST | No (data movement) | n/a |
| `sub_unary_tile` (per-offset subtract) | SFPU | No | n/a (uses `binop_with_scalar`) |
| `fill_tile` | SFPU | No | n/a |
| `sub_binary_tile`, `add_binary_tile` | SFPU | No (elementwise add/sub is exact) | n/a |
| `mul_binary_tile` (only in `frac(x_off) · π`) | SFPU | **Yes** in principle; SFPU `*` is full-precision. `math_fidelity` is documented to affect **FPU** multiplies (matmul/reduce/mul_tiles). | HiFi4 |
| `ltz_tile`, `eq_binary_tile`, `where_tile`, `abs_tile`, `frac_tile`, `floor_tile` | SFPU | No | n/a |
| `log_tile<false>(1)` (the `log(z)` feeding Stirling) | SFPU | No — controlled by `math_approx_mode`, not fidelity | **Precise mode** (template arg `false`) |
| `log_tile(1)` (the `log\|sin\|` term) | SFPU | No — `math_approx_mode` only | **Default mode** (no template arg — resolves to the kernel's default, typically approximate when `math_approx_mode=true`, but `math_approx_mode` is not set in `ComputeConfigDescriptor` and so falls to the default for that field, which is `true` per `WormholeComputeKernelConfig`). See *Key Observations* below. |
| `sin_tile`, `lgamma_stirling_float_tile`, `lgamma_adjusted_tile` | SFPU | No — internally use `math_approx_mode` template via LLK | Whatever propagates from `ComputeConfigDescriptor` (`math_approx_mode` not explicitly set ⇒ takes the descriptor default). |
| `pack_tile` (fp32 → fp32) | Packer | No (bit-exact for same format) | n/a |

- **User-configurable**: **No.** The entry point `multigammaln(input_tensor, *, memory_config=None)` (`multigammaln.py:19–23`) does not accept a `compute_kernel_config` argument. `math_fidelity` and `fp32_dest_acc_en` are hard-coded in `multigammaln_program_descriptor.py:171–174`.
- **Math fidelity in effect**: `MathFidelity.HiFi4` (4 FPU passes for any FPU multiply).
- **Recommended minimum fidelity**: The compute kernel contains no FPU multiply (no `mul_tiles`, no `matmul_tiles`, no `reduce_tile`). All multiplies are SFPU-internal. **HiFi4 is overkill** for the actual FPU workload here; it does not change correctness for SFPU operations. The choice is defensive — it matches the reference `lgamma` precision target and keeps the door open for an FPU-resident binary multiply that might be substituted in a future refactor. The op_design.md "Key Risks" table (line 262) explicitly asserts HiFi4 is required, but the only FPU-touching path in the compute kernel is the unpack/pack data movement, which is not throughput-bounded by fidelity.

---

## Tile-Boundary Precision

- **Tiles in reduction**: 0 (no reduction). The relevant tile-boundary path is the round-trip from DEST → intermediate CB → DEST that occurs once per output element, per term.
- **Dest capacity**: 4 tiles (fp32_dest_acc_en + half-sync default — see reference §2.2 and `dest_helpers.hpp:88–102`).
- **L1 round-trips per output tile**: **1** — each of the 4 lgamma sub-phases packs 1 tile, then sub-phase B reads all 4 back. The input tile makes 4 reads (kept in front, not popped between sub-phases — see compute kernel lines 198–207) but never round-trips through the kernel.
- **Intermediate format**: **Float32** for all 4 `cb_lgamma_*` CBs (`multigammaln_program_descriptor.py:107`, `data_format=input_tensor.dtype`).
- **UnpackToDestMode**: **Default** (not `UnpackToDestFp32`). The descriptor does not pass an `unpack_to_dest_mode` override. With fp32 source CBs and fp32 DEST, the standard unpack path goes through SrcA/SrcB. The reference §2.7 notes that SrcA/SrcB may truncate to TF32 (≈10-bit mantissa); whether this happens in practice depends on the LLK unpack path for `copy_tile` on fp32 CBs. **This is a potential precision exposure point.**
- **Assessment**: Of all error sources in the pipeline, the round-trip through SrcA-on-`copy_tile` is the most defensible candidate for a refinement: setting `unpack_to_dest_mode[cb_lgamma_*] = UnpackToDestFp32` for all four intermediate CBs would route the unpack directly to DEST, avoiding any TF32 truncation. The cost is modest (just descriptor lines), and the benefit is most relevant for the catastrophic-cancellation branch in `lgamma_adjusted_tile` where a 10-bit-mantissa intermediate could compound the precision loss.

---

## Configuration Exposure

| Setting | Exposed to user | Default | Recommendation |
|---------|:--------------:|---------|----------------|
| `fp32_dest_acc_en` | ✗ | `True` (hard-coded) | Correct for fp32 inputs and the multi-step lgamma recipe. Should remain on by default. |
| `math_fidelity` | ✗ | `HiFi4` (hard-coded) | HiFi4 is the safe default but overkill for this kernel — no FPU multiply path is active. HiFi2 would not change correctness; the choice is documented as defensive in `op_design.md`. |
| `math_approx_mode` | ✗ | Not set in `ComputeConfigDescriptor` ⇒ falls to the underlying default (`true` per the `WormholeComputeKernelConfig` reference default). | All SFPU calls except the explicit `log_tile<false>` (line 116) inherit the approx-mode default. For `sin`, `log\|sin\|`, the implicit reciprocal inside Stirling, this means **approximate mode is in effect**. |
| `packer_l1_acc` | ✗ | Not set ⇒ default `false`. | Correct: no L1 accumulation is performed, and L1-acc does not work with `fp32_dest_acc_en=True` regardless (reference §2.6). |
| `compute_kernel_config` (full struct) | ✗ | Not accepted by entry point | Phase-0 deliberately omits this for simplicity; the public API surface is `multigammaln(input_tensor, *, memory_config=None)`. |
| `dtype` | ✗ | `float32` only (validated; raises `ValueError` otherwise — `multigammaln.py:59–62`) | Phase 0 contract. |
| `p` (order) | ✗ | Hard-coded to 4 | Documented in `op_design.md` and the entry-point docstring. |

---

## Key Observations

1. **Pure elementwise SFPU operation with no reduction or accumulation across tiles.** Most of the standard stability concerns (Wt-deep accumulation, divide-then-sum, tile-boundary precision loss in a multi-tile reduction) do not apply. The only L1 round-trip per output tile is the 4-term intermediate hop, which stays in fp32 end-to-end on the CB side.

2. **The dominant error sources are inside the LLK lgamma routines, not in the multigammaln kernel itself.** Stirling's Bernoulli correction uses 2-iter Newton-Raphson reciprocal (`_sfpu_reciprocal_<2>`, ~1 ULP), and Taylor bridges near `z=1` / `z=2` target a 0–3 ULP overall precision per the LLK header comment. The kernel comments at lines 25–33 also explicitly note a known LLK-level issue where inputs very close to `0.5` can produce `+inf` when `lgamma_adjusted_tile` is inlined four times (not seen when `ttnn.lgamma` is called in isolation) — a precision pathology specific to the multi-inline structure of this kernel.

3. **`UnpackToDestMode` is NOT configured to `UnpackToDestFp32` for the four intermediate CBs.** This is the only realistic precision-improvement lever in the program descriptor. Setting `unpack_to_dest_mode[24..27] = UnpackToDestFp32` would route `copy_tile()` directly from L1 to DEST in fp32, bypassing the SrcA path that may truncate to TF32. Worth investigating as a future refinement, particularly for accuracy-tuning the `x_off < 0.5` reflection branch where catastrophic cancellation already eats into the bit budget.

4. **Mixed approximate/precise mode for `log`.** The first `log` (computing `log(z)` that feeds Stirling) is called with the explicit precise template arg (`log_tile<false>(1)`, line 116). The second `log` (taking `log|sin|`) is called without a template arg (line 167) and resolves to the kernel default — approximate. This mirrors `lgamma_kernel.cpp:67–68` and `:118–119` exactly. Both choices are intentional: Stirling's `(z - 0.5)·log(z)` is the dominant magnitude term and benefits most from precision, whereas `log|sin|` is unbounded and the leading-order error contribution comes from the `sin` approximation rather than the `log` itself.

5. **HiFi4 fidelity is functionally inert in this kernel.** The compute kernel contains zero FPU multiply ops (`matmul_tiles`, `mul_tiles`, `reduce_tile`); all binary multiplies are SFPU (`mul_binary_tile`), which are unaffected by `math_fidelity`. Choosing HiFi4 costs no precision but also gains none — the documented rationale in `op_design.md:262` ("required for the precision target") is overstated. The actual precision lever for this op is `fp32_dest_acc_en`, which IS engaged and IS necessary (the LLK `lgamma_adjusted_tile` explicitly switches on this flag at `ckernel_sfpu_lgamma.h:150–157` for both the +∞ handling at `\|x\| = ∞` and to skip the bf16-round-trip that the `!is_fp32_dest_acc_en` path performs).
