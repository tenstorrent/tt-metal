# Numerical Stability Analysis: multigammaln_lanczos

## Algorithm Summary

`multigammaln_lanczos` computes `torch.special.multigammaln(x, p=4)` as a single fused TTNN
elementwise unary kernel. Mathematically, per element:

```
output[i] = sum_{k=0..3} lgamma_lanczos(input[i] - 0.5*k) + 3*log(pi)
```

where each `lgamma_lanczos(a)` is the Stirling/Lanczos 6-term form algebraically rearranged
so it fits inside 4 fp32 DST slots:

```
temp   = 1 + sum_{i=0..5} C[i] / (a + i)            # 6 reciprocals, 6 mul-by-scalar, 6 adds
result = (a - 0.5)*log(a + 4.5) + log(temp) - a - 3.581061468642829
if (a == 1.0) or (a == 2.0): result = 0.0           # pole zero-mask via unary_ne(a)
```

The implementation is **elementwise unary** — there is **no inter-element reduction**, no
broadcast, and no inter-core communication. The "accumulation" is intra-tile and intra-element:
each output element is the sum of 4 lgamma evaluations of the same input element, sequenced as
4 separate `tile_regs_acquire`/`pack_tile` blocks that ping-pong through a Float32 intermediate
CB (`cb_accumulator`).

**Precision-sensitive phases (per input tile, per Lanczos sub-evaluation `k`):**

1. **Reciprocal cluster** — 6 × `recip_tile()` on `a + i` for `i = 0..5` (SFPU; precise mode by default).
2. **Per-term scalar multiply** — 6 × `mul_unary_tile(C[i])`, where `|C[i]|` spans 5 orders of
   magnitude (`5.4e-6` ... `86.5`).
3. **Local sum of 6 Lanczos terms** in D0 via `add_binary_tile` (SFPU eltwise add, NOT FPU-fidelity-sensitive).
4. **Two `log_tile()` calls** — `log(temp)` and `log(a + 4.5)` (SFPU; precise mode by default).
5. **Subtraction `(log(temp) - (a + 3.581...))`** — potential catastrophic cancellation
   when the two operands have similar magnitude (kernel line 145).
6. **Multiply `(a - 0.5) * log(a + 4.5)`** — FPU? No: implemented via `mul_binary_tile` (SFPU eltwise),
   so it is **NOT affected by `math_fidelity`**.
7. **Pole mask** — `unary_ne_tile(a, 1.0)` / `unary_ne_tile(a, 2.0)` on a copy of `a` (not on the
   result), then `mul_binary_tile(result, mask, result)`. Designed to avoid `NaN * 0 = NaN`.
8. **Inter-iteration accumulator round-trip** — at the end of each `k`, D0 is packed to
   `cb_accumulator` (Float32 CB); the next `k` reloads it via `copy_tile(cb_accumulator, 0, 2)`.

The accumulation depth is fixed at **4 terms per element** (one per multivariate sub-evaluation)
plus 6 sub-terms inside each lgamma — both are constant in the operation's shape.

---

## Error Source Inventory

Per-tile counts assume Phase 0 fixed config: `p=4`, half-sync, `fp32_dest_acc_en=True`,
so DST holds 4 tiles. Line numbers reference `kernels/multigammaln_lanczos_compute.cpp`.

| # | Source | Location | Per output element | Severity | Mitigation |
|---|--------|----------|-------------------|----------|------------|
| 1 | SFPU reciprocal — `1/(a+i)` for the 6 Lanczos terms | line 125–126 (per `k`) | 6 × 4 = 24 reciprocals | Low–Moderate | Precise mode (default `math_approx_mode=false` in Python `ComputeConfigDescriptor`); `recip_tile<APPROX=false>` runs the 2-iteration Newton step (≤1 ULP fp32) |
| 2 | SFPU multiply-by-scalar with mixed-magnitude C[i] | line 128 | 6 × 4 = 24 mul-by-scalar | Moderate | None per-term. Bit-cast at compile time (line 42 `f2u`) — no runtime scalar rounding. C[5]=5.4e-6 vs C[0]=76 sums create rounding cancellation in `D0` (mitigated by fp32 dest acc). |
| 3 | SFPU `add_binary_tile` sum of 6 Lanczos terms into D0 | line 130 | 6 × 4 = 24 adds | Low | fp32_dest_acc_en=True; D0 stays in fp32 across all 6 adds without ever leaving DST (no L1 round-trip inside one `k`). SFPU add is bit-exact for representable inputs. |
| 4 | SFPU `log_tile(log(temp))` | line 137 | 4 logs | Low | Precise mode (`fast_and_approx=false`, the template default at `compute_kernel_api.h:92`). `temp` is approximately 1 + small, so log uses the well-conditioned regime. |
| 5 | **Catastrophic cancellation** `D0 -= (a + 3.581)` (where `D0` was `log(temp)` ≈ small) | line 145 | 4 subtractions | Moderate–High | None explicit. Both operands are finite; relative error in D0 grows from this subtraction. fp32_dest_acc helps preserve the ~23-bit mantissa, but the subtraction itself still loses leading bits if magnitudes happen to align. |
| 6 | SFPU `log_tile(log(a + 4.5))` | line 153 | 4 logs | Low | Precise mode; argument `a + 4.5 ∈ [6.5, 14.5]` on the safe domain — well away from the log pole at 0. |
| 7 | SFPU `(a - 0.5) * log(a + 4.5)` via `mul_binary_tile` | line 159 | 4 multiplies | Low | `mul_binary_tile` is SFPU — **not affected by `math_fidelity`**. Both operands are fp32 in DST; full mantissa retained. |
| 8 | SFPU `(D0 += (a-0.5)·log(a+4.5))` | line 161 | 4 adds | Low | Operands are in DST in fp32; clean fp32 add. |
| 9 | Pole-mask multiplies `D0 *= (a != 1); D0 *= (a != 2)` | lines 172, 179 | 4 × 2 = 8 multiplies | None (designed) | **Mask is computed on D2 = copy of `a`, NOT on D0/result.** Since `a` is finite throughout the safe domain `[2.0, 10.0]` (and the shifted forms `a − 0.5·k` remain finite), `(a != 1.0)` cleanly evaluates to `{0.0, 1.0}`. Avoids the `NaN * 0 = NaN` pitfall that would occur if the polynomial actually hit a pole and we masked the result. |
| 10 | **Inter-`k` accumulator round-trip through L1** — `pack_tile(0, cb_accumulator)` then `copy_tile(cb_accumulator, 0, 2)` | lines 191, 183 | 4 packs + 4 reloads per output element (5 packs + 4 reloads counting init-zero and final) | **Low–Moderate** | `cb_accumulator` is **Float32** (program descriptor line 98). Pack from fp32 DST to fp32 L1 is lossless. **However**, `UnpackToDestMode::UnpackToDestFp32` is **NOT** set for `cb_accumulator` (the program descriptor never assigns `unpack_to_dest_mode`). The reload uses `copy_tile`, which under fp32_dest_acc=true with the default UnpackToDestMode goes through SrcA/SrcB and may truncate to TF32 (~10-bit mantissa). See "Tile-Boundary Precision" for the precision impact. |
| 11 | Final `D0 += 3·log(π)` constant add | line 207 | 1 add | None | fp32 dest; constant baked at compile time. |
| 12 | Final pack to `cb_output_tiles` (Float32 output CB) | line 211 | 1 pack | None | Output CB is Float32 (program descriptor line 87 uses `output_tensor.dtype` which is fp32 per validator). |
| 13 | DRAM round-trip on the output | writer kernel | 1 NoC write per tile | None | Float32 throughout the data path; no dtype conversion. |

**Total operation count per output element**: 24 SFPU recip + 24 SFPU mul-by-scalar + 24 SFPU add (Lanczos terms) + 8 SFPU log + 8 SFPU pole-mask multiplies + 4 catastrophic-cancellation-prone subs + 4 FPU-free multiplies + miscellaneous DST→DST copies. Roughly **~110 SFPU ops per output element**.

---

## Accumulation Analysis

- **What is accumulated**: The 4 per-multivariate-`k` lgamma results, each itself the sum of 6
  Lanczos polynomial terms. The accumulation is **across the 4 lgamma iterations** (`k = 0..3`)
  for a single input element — it is *not* a tensor-dimension reduction.
- **Accumulation depth**: **4 terms** at the outer level (`k=0..3`) + **6 terms** at the inner level
  (Lanczos polynomial), independent of tensor shape. Constant, shape-independent.
- **Dest precision**: **fp32** (`fp32_dest_acc_en=True`, hard-coded in
  `multigammaln_lanczos_program_descriptor.py:168`). DST holds 4 tiles per acquire.
- **Intermediate CB format**: `cb_accumulator` is **`ttnn.float32`** (program descriptor line 98).
  Page size 4096 B = 1 fp32 tile. 2 pages — minimum for the read-modify-write ping-pong (front
  + back simultaneously live).
- **UnpackToDestFp32 configured**: **No.** `unpack_to_dest_mode` is not assigned in the program
  descriptor; the binding default at `program_descriptors.cpp` leaves it as `UnpackToDestMode::Default`
  for every CB. This means the `copy_tile(cb_accumulator, 0, 2)` reload (line 183) goes through
  the standard unpack path (SrcA/SrcB), which under `fp32_dest_acc=true` may truncate fp32 → TF32
  (~10 mantissa bits) on the unpack edge.
- **Round-trips through L1**: **4 fp32 round-trips per output element** — one per `k = 0..3`,
  exactly (5 packs in counting init-zero, 4 reloads, plus 1 final read for the `+3·log(π)`).
- **Order of operations**: This is not a mean — there is no divide. The 4 lgamma values are
  summed; the constant `3·log(π)` is added once at the end. No "divide-then-sum vs sum-then-divide"
  choice applies.
- **Assessment**: The accumulation **chain** is well-conditioned at the algorithm level: only 4
  outer terms and 6 inner terms, fixed at compile time. The risk is concentrated in two places:
  (a) the four-times-repeated catastrophic-cancellation subtraction `log(temp) - (a + 3.581)` on
  line 145, and (b) the precision-on-reload story for `cb_accumulator` (item #10). The op stays
  inside the documented safe domain `a ∈ [2.0, 10.0]` per `op_design.md`, where the Lanczos
  6-term polynomial is at its sweet spot; tolerances are explicitly wide
  (`rtol=0.1, atol=0.5`) reflecting the fp32-Lanczos vs libm-double baseline gap.

---

## Numerical Guards

| Guard | Present | Location | Notes |
|-------|:-------:|----------|-------|
| Max subtraction before exp | n/a | — | No `exp_tile` in this kernel — only `log_tile` and `recip_tile`. |
| ReLU clamp for approx exp | n/a | — | No `exp_tile`. |
| Epsilon before reciprocal | ✗ | line 126 | No `+ε` guard. Safe-domain assumption: smallest `a + i` on `[2.0, 10.0]` with `k=3, i=0` is `(2.0 − 1.5) + 0 = 0.5` → `1/0.5 = 2.0` finite. With `k=3, i=5`, denominator is `0.5 + 5 = 5.5` — well away from zero. **Out-of-domain inputs** (`a ≤ 0` with `i = 0` at `k = 0`, etc.) will produce ±Inf and propagate. Documented as "NaN/Inf propagates naturally" in `op_design.md` → Risk #7. |
| Non-tile-aligned masking | ✗ | entry point validator | Not needed — operation is elementwise, no reduction. Validator at `multigammaln_lanczos.py:74-78` rejects non-tile-aligned shapes (`H % 32 != 0` or `W % 32 != 0`). |
| Welford's algorithm | n/a | — | No mean/variance; no need. |
| **Pole zero-mask on input, not on result** | ✓ | lines 167–179 | **Intentional and load-bearing.** The mask is computed as `(a != 1.0)` and `(a != 2.0)` on a copy of `a` (D2), not on the polynomial result (D0). Reason: `a` is always finite within the safe domain, so `(a != C)` is a clean `{0.0, 1.0}` evaluation. If a future input drives the polynomial to NaN/Inf (e.g., a `1/(a + i)` hitting `1/0`), then `NaN * 0 = NaN` would still leak through if the mask were applied to D0. By masking on `a` first, `0.0 * NaN_in_D0 = NaN` — **wait, this still propagates NaN.** The guard is effective only if the polynomial in D0 is **finite** at the input value `a == 1` or `a == 2`. Within `[2.0, 10.0]` and the shifted Lanczos arguments, this is the case (smallest denominator `a + i = 0.5` is finite). For inputs outside the safe domain that drive D0 to NaN, the mask does not rescue the result — the kernel relies on the user staying inside the documented domain. (Comment at line 165–166 in the kernel acknowledges this.) |
| **Algebraic re-grouping to free a DST slot** | ✓ | constants at lines 60–61 | `LANCZOS_OFFSET = 4.5 − 0.918938531357171 = 3.581061468642829` is precomputed at compile time. Mathematically equivalent to the standard form — does not change numerical conditioning. |
| **Bit-cast scalars at compile time** | ✓ | `f2u()` at line 42 | `__builtin_bit_cast` evaluates at compile time. No runtime float→u32 conversion error; the exact IEEE-754 representation of each Lanczos coefficient reaches the SFPU intact. |

---

## Math Fidelity Profile

The operation is **SFPU-dominant** — every arithmetic op in the kernel except the pack/unpack
hardware is an SFPU primitive. There are **no `matmul_tiles`, no `reduce_tile`, no FPU
`mul_tiles`** calls. Therefore `math_fidelity` has **almost no effect** on the result of this op.

| Compute phase | Engine | Fidelity-sensitive | Approx-mode-sensitive | Default in this op |
|--------------|:------:|:-----------------:|:--------------------:|--------------------|
| `add_unary_tile(D, scalar)` (scalar add/sub) | SFPU | No | No | — |
| `mul_unary_tile(D, scalar)` (Lanczos coefficient mul) | SFPU | No | No | — |
| `recip_tile(D)` | SFPU | No | **Yes** | `math_approx_mode=false` ⇒ 2-iter Newton, ≤1 ULP fp32 |
| `log_tile(D)` | SFPU | No | **Yes** | `math_approx_mode=false` ⇒ precise polynomial |
| `add_binary_tile / sub_binary_tile / mul_binary_tile` (DST×DST) | SFPU | No | No | — |
| `unary_ne_tile(D, scalar)` | SFPU | No | No | — |
| `copy_dest_values<Float32>(src, dst)` | DST internal | No | No | — |
| `fill_tile(D, 0.0f)` | SFPU | No | No | — |
| `copy_tile(cb, …)` (CB → DST) | unpack | No | No | format-driven only |
| `pack_tile(D, cb)` | pack | No | No | format-driven only |

- **User-configurable**: **No.** The entry point `multigammaln_lanczos(input_tensor)` accepts
  no `compute_kernel_config` argument. Compute config is hard-coded in
  `multigammaln_lanczos_program_descriptor.py:166-169` to `MathFidelity.HiFi4` and
  `fp32_dest_acc_en=True`. `math_approx_mode` is left unset → binding default `false` (precise).
  `dst_full_sync_en` is left unset → default `false` (half-sync). `bfp8_pack_precise`,
  `unpack_to_dest_mode` are also left at defaults.
- **Math fidelity HiFi4 is over-specified for this op**: every fidelity-sensitive op (FPU
  multiply / matmul / reduce) is absent. HiFi4 has zero precision benefit here vs LoFi — but
  also no throughput cost, because the FPU is idle. (The kernel exclusively uses SFPU and
  unpacker/packer.) The HiFi4 setting is a documented, defensive "max precision" choice from
  the Phase 0 spec; it does not actually drive precision in this kernel.
- **Math approx mode is the precision lever**: defaults to `false` (precise). `recip_tile` and
  `log_tile` consequently run in their high-precision variants. **If the implementer ever flips
  `math_approx_mode` to true to reclaim throughput, the precision regime changes substantially**
  (recip ~3 ULP, log polynomial degree reduced).

**Known Wormhole hardware bug**: HiFi4 + `fp32_dest_acc_en` can produce incorrect results
on Wormhole B0 (bug #38306, per `numerical_stability_analysis_reference.md` §2.3). The
recommended workaround is HiFi3 + fp32 dest. **For this op, the FPU is unused** — the
fidelity setting is moot in practice — but a strict reading of the bug note would say HiFi3
is the "safer" choice for any kernel running with fp32 dest on Wormhole. No mitigation in the
current code.

---

## Tile-Boundary Precision

- **Tiles in reduction**: **N/A.** Elementwise unary, no across-tile reduction. Each output
  tile is computed independently from one input tile.
- **Accumulator round-trips per output element**: **4 fp32 round-trips through `cb_accumulator`**
  (the per-`k` ping-pong, lines 191 / 183), plus 1 init-zero pack (line 95) and 1 final reload
  (line 205) for the `+3·log(π)` add. Net: **5 packs / 5 pops on `cb_accumulator` per output tile**.
- **Dest capacity vs work**: DST holds 4 fp32 tiles (half-sync + fp32 dest). The kernel uses
  D0 / D1 / D2 / D3 simultaneously within each `tile_regs_acquire` block — exactly at the cap.
  The algebraic re-grouping in `op_design.md` was specifically chosen to free the global
  accumulator out of DST (into `cb_accumulator`) so the local 4-slot budget is sufficient.
- **Intermediate CB format**: `cb_accumulator` is **Float32** (program descriptor line 98).
  fp32 dest → fp32 L1 is a **lossless** pack (no mantissa truncation, no shared-exponent loss).
- **UnpackToDestFp32**: **Not configured.** The program descriptor does not set
  `unpack_to_dest_mode` for any CB. Under `fp32_dest_acc_en=true` and default
  UnpackToDestMode, the reload `copy_tile(cb_accumulator, 0, 2)` may route through SrcA/SrcB
  and truncate fp32 → TF32 (10-bit mantissa) on the unpack edge. This is the **single most
  significant precision gap** in the op as currently written.
- **Realized precision per round-trip**: **fp32 on pack**, **possibly TF32 (~10 mantissa
  bits) on unpack** — call it ~3 decimal digits per round-trip × 4 round-trips. Empirically,
  the test suite passes at `rtol=0.1, atol=0.5` on `a ∈ [2.0, 10.0]`, indicating the precision
  is acceptable for this domain even without `UnpackToDestFp32`. The "Lanczos 6-term at fp32"
  baseline error is itself ~1e-3, so the TF32 reload edge is comparable in magnitude — not
  catastrophic, but not optimal.
- **Assessment**: 4 round-trips with a Float32 intermediate CB and TF32 reload edge. Adding
  `unpack_to_dest_mode[cb_accumulator] = UnpackToDestMode::UnpackToDestFp32` to the program
  descriptor would close this gap at zero algorithmic cost (the same `copy_tile` calls would
  bypass SrcA/SrcB). This is a documented best practice from the reference doc §2.7 and
  `accuracy_tips.md`. It is the cleanest available numerical improvement.

---

## Configuration Exposure

| Setting | Exposed to user | Default in this op | Recommendation |
|---------|:--------------:|--------------------|----------------|
| `fp32_dest_acc_en` | ✗ | **True** (hard-coded, `multigammaln_lanczos_program_descriptor.py:168`) | Keep True — critical for the 24-term Lanczos sum in D0 (item #2). |
| `math_fidelity` | ✗ | **HiFi4** (hard-coded, line 167) | Effectively over-specified — kernel uses no FPU multiplies. LoFi would be equivalent in precision and throughput here. Setting HiFi4 also incurs the documented HiFi4+fp32-dest WH B0 risk (#38306), though that risk is theoretical here since the FPU is idle. |
| `math_approx_mode` | ✗ | **False** (binding default — not set in descriptor) | This is the **real precision lever** for this op. Keep at False — flipping to True would substantially degrade `recip_tile` (24 calls/elem) and `log_tile` (8 calls/elem). |
| `dst_full_sync_en` | ✗ | False (default — half-sync) | Half-sync with fp32 dest gives 4-slot DST; the algorithm is specifically designed to fit. No reason to change. |
| `unpack_to_dest_mode[cb_accumulator]` | ✗ | **Default (NOT UnpackToDestFp32)** | **Gap.** Setting this to `UnpackToDestFp32` would make the reload edge truly lossless (matches the lossless pack edge). One-line change in the program descriptor. |
| `packer_l1_acc` | ✗ | N/A (not a field on this descriptor; would be incompatible with fp32 dest anyway per reference §2.6) | Not applicable. |
| `bfp8_pack_precise` | ✗ | False (default) | Not applicable — no bfp8 in this op. |
| `p` (multigammaln parameter) | ✗ | Fixed at 4 | Spec — permanently fixed. |
| `compute_kernel_config` argument | ✗ | n/a (no such argument on the entry point) | Phase 0 deliberately does not expose this; documented as a future refinement in `op_design.md` → "Out of Scope". |

---

## Key Observations

1. **The op is SFPU-only — math_fidelity is irrelevant in practice.** Every multiply in the
   kernel (`mul_unary_tile`, `mul_binary_tile`) is SFPU, not FPU. There are no `matmul_tiles`,
   no `reduce_tile`, no FPU `mul_tiles`. The hard-coded `HiFi4` setting is a defensive
   max-precision choice that has zero effect on the actual result. The real precision lever is
   `math_approx_mode`, which the program descriptor leaves at the binding default `false`
   (precise mode) — that is the correct setting for this op.

2. **The pole-zero guard is correctly applied to `a`, not to the result, but only effective
   inside the safe domain.** Lines 167–179 mask via `unary_ne_tile(D2 = copy of a, 1.0)` and
   `unary_ne_tile(D2, 2.0)`, then `mul_binary_tile(D0, D2, D0)`. Because `a` is finite on the
   documented domain `[2.0, 10.0]`, the mask cleanly evaluates to `{0.0, 1.0}` and the multiply
   into D0 produces an exact zero. The design choice avoids the `NaN × 0 = NaN` pitfall — but
   only as long as D0 itself is finite at `a = 1` or `a = 2`. The op explicitly opts out of
   handling adversarial inputs that drive D0 to NaN/Inf via the input validator + documented
   safe domain.

3. **The accumulator-CB reload edge is missing the `UnpackToDestFp32` setting.** The pack edge
   (fp32 DST → fp32 L1 in `cb_accumulator`) is lossless because the CB is declared `ttnn.float32`
   in the program descriptor (line 98). The reload edge (`copy_tile(cb_accumulator, 0, 2)` at
   line 183) routes through the default unpacker, which under `fp32_dest_acc_en=true` may
   truncate fp32 → TF32 (~10 mantissa bits) on the way back into DST. Setting
   `unpack_to_dest_mode[cb_accumulator] = UnpackToDestMode::UnpackToDestFp32` in the program
   descriptor would close this — and only this — gap. This is the single highest-leverage
   numerical improvement available without changing the algorithm.

4. **The single sub-Lanczos catastrophic-cancellation candidate is `log(temp) − (a + 3.581)`
   on line 145.** Both operands are finite, and `log(temp)` is approximately `log(1 + small) ≈ small`,
   while `(a + 3.581)` on `a ∈ [2.0, 10.0]` is approximately `5.58` to `13.58`. The two operands
   are **not** of similar magnitude over the documented safe domain — the subtraction is well-
   conditioned in practice. The risk only materializes for inputs that drive `log(temp)` close to
   `a + 3.581`, which the safe-domain assumption rules out.

5. **The 4-tile DST budget is tight and load-bearing.** Within each `k` iteration the kernel
   simultaneously needs D0 (local accumulator), D1 (`a`, persistent across the iteration),
   D2 (per-Lanczos-term scratch + reload of previous global accumulator), and D3 (`a − 0.5`
   only in step 2.k.5). The algebraic re-grouping in `op_design.md` that moves the global
   accumulator into `cb_accumulator` exists precisely to free a slot. This makes the 4
   `cb_accumulator` L1 round-trips a *consequence of the DST budget*, not a precision oversight
   — the alternative (keeping the global accumulator in DST) would have required 5+ slots and
   was structurally impossible under half-sync + fp32 dest.

6. **Test tolerances are wide and intentional.** `op_design.md` Risk #8 documents that
   "Lanczos at fp32 is meaningfully less accurate than torch's libm double-precision reference"
   and that the acceptance test runs at `rtol=0.1, atol=0.5` on `a ∈ [2.0, 10.0]`. The numerical
   profile of this op is "fp32-Lanczos baseline, dominated by algorithmic error, not by
   pack/unpack precision loss". The dominant error budget is the 6-term polynomial truncation,
   not the 4 L1 round-trips. That said, observation #3 above is still a free win — closing the
   TF32 unpack edge would shrink the implementation-side error contribution toward zero
   without touching the irreducible Lanczos polynomial error.
