# Numerical Stability Analysis: glu_fused

## Algorithm Summary

`glu_fused` computes the Gated Linear Unit along the last dimension:

```
out[..., j] = x[..., j] * sigmoid(x[..., j + W/2])      for j ∈ [0, W/2)
```

It is a **pure pointwise (elementwise) operation** — there is no reduction, no accumulation across tiles, no row-wise broadcasting, and no inter-tile data dependency. Each output element depends on exactly two input elements (one from each half of the W dimension), and each input element is consumed exactly once.

**Precision-sensitive phases** (in order of appearance per output tile):

1. **Sigmoid of the second-half tile (B)** — a transcendental SFPU operation; the Wormhole/Blackhole implementation uses a 6-piece piecewise-linear LUT, not `1/(1+exp(-x))`. This LUT is the dominant source of numerical error in this op.
2. **Element-wise multiply A·sigmoid(B)** — performed on the SFPU as `mul_binary_tile` between two DEST slots; effectively exact at fp32 (one float multiply per lane).
3. **Pack of D0 to output CB** — fp32 → fp32 (CB is allocated with `output_tensor.dtype = float32`); no truncation here.

There are no precision-sensitive phases that require accumulation guards, max-subtraction, epsilon, or masking.

---

## Error Source Inventory

| # | Source | Location | Severity | Mitigation present? |
|---|--------|----------|----------|---------------------|
| 1 | Sigmoid 6-piece LUT (piecewise-linear approximation of sigmoid) | `glu_fused_compute.cpp:41` (`Sigmoid<Approx::Exact, D1>`) → `sigmoid_tile<RC, false>` → `_calculate_sigmoid_` LUT2 | **Moderate-Low**: piecewise-linear fit, bounded ≲ 1% absolute error of an output in [0,1]; coefficients are FP16-encoded so even the "exact" mode is not bit-accurate fp32 sigmoid | Yes — `Approx::Exact` (i.e. `APPROXIMATION_MODE=false`) is selected, which on WH picks the more conservative LUT path. No further mitigation is possible without switching to `1/(1+exp(-x))` |
| 2 | SFPU multiply D0 = D0 × D1 | `glu_fused_compute.cpp:42` (`SfpuMul<D0,D1,D0>` → `mul_binary_tile`) | **Negligible**: single fp32 lane multiply on values already in DEST, fp32_dest_acc_en=true | Implicit (single fp32 op, no rounding chain) |
| 3 | Unpack of A and B tiles from L1 → DEST | `glu_fused_compute.cpp:39-40` (`Load<cb_input_a/b, …>`) | **None**: lossless, both CBs are Float32 and `UnpackToDestMode::UnpackToDestFp32` is set | `UnpackToDestFp32` on both `cb_input_a` and `cb_input_b` (`glu_fused_program_descriptor.py:170-172`) |
| 4 | Pack of D0 → `cb_output_tiles` | implicit in `sfpu_pipeline<…>` epilogue | **None**: output CB is Float32 (same dtype as input), so the pack is fp32→fp32 (no mantissa truncation) | Output dtype = input dtype = float32 (`glu_fused.py:48-52` allocates output with `input_tensor.dtype`) |
| 5 | Sigmoid LUT for very large |x|: saturates at 0.4998 for x>4 and at ~−0.4998+0.5 ≈ 0.0002 for x<−4 (LUT branch returns 0.4998+0.5 ≈ 1.0 / 0.0 region) | same as #1 | **Low**: the LUT's saturation behaviour is *numerically benign* for sigmoid (true sigmoid also saturates to 1/0); no NaN/Inf paths, and no negative-output-for-very-negative-input pathology that plagues approximate `exp_tile` | Inherent to the LUT design |
| 6 | Math fidelity passes on `SfpuMul` | n/a | **None**: `mul_binary_tile` is an SFPU op, **not** an FPU op — `math_fidelity` does *not* gate SFPU multiply. HiFi4 is configured but has no effect on this op's compute path | — |

There are no `add_tiles`, `sub_tiles`, `reduce_tile`, `recip_tile`, `exp_tile`, or `matmul_tiles` in this kernel. The classic stability concerns (accumulation depth, catastrophic cancellation, max-subtract guard, ε guard, ReLU clamp for approx exp, Welford's, partial scaler) are all **not applicable**.

---

## Accumulation Analysis

| Aspect | Value |
|--------|-------|
| What is accumulated | **Nothing** — pure pointwise op |
| Accumulation depth | 0 |
| DEST precision | fp32 (`fp32_dest_acc_en=True`) |
| Intermediate CB format | n/a — no intermediate CBs between compute phases |
| `UnpackToDestFp32` configured | Yes, on `cb_input_a` and `cb_input_b` (inputs only — no intermediate accumulator CB to need it) |
| Round-trips through L1 (per output tile) | 0 between compute phases — each output tile is produced in one DEST acquire/commit cycle, no spill to L1 |
| Order of operations | n/a (no sum or mean) |
| Assessment | No accumulation error possible. The entire fused chain (`Load A; Load B; Sigmoid(B); A*B`) executes within a single DEST tile-batch, so even the `pack_tile` happens exactly once per output tile, directly from fp32 DEST to an fp32 output CB |

---

## Numerical Guards

| Guard | Present | Location | Notes |
|-------|:-------:|----------|-------|
| Max subtraction before exp | n/a | — | No `exp_tile` in the kernel — sigmoid is computed via LUT, not via `1+exp(-x)` |
| ReLU clamp for approximate exp | n/a | — | No approximate exp used |
| Epsilon before reciprocal | n/a | — | No reciprocal in the kernel |
| Non-tile-aligned masking | ✗ (not needed) | `glu_fused.py:78-85` | Entry point **rejects** non-aligned shapes: `W % 64 != 0` and `H % 32 != 0` raise. Phase-0 design avoids the masking question entirely by requiring tile-aligned inputs |
| Welford's algorithm | n/a | — | No mean/variance |
| Sigmoid saturation (LUT → 0.4998 for x>4, ~0 for x<−4) | Implicit | `ckernel_sfpu_sigmoid.h:54-59` | The LUT itself is its own clamp — no Inf/NaN can be produced for finite inputs |
| FP32 precision on unpack (avoid TF32 truncation through SrcA/SrcB) | ✓ | `glu_fused_program_descriptor.py:170-172` | `UnpackToDestMode::UnpackToDestFp32` on both input CBs |

---

## Math Fidelity Profile

| Compute phase | Hardware unit | Fidelity-sensitive | Configured setting |
|--------------|--------------|:------------------:|--------------------|
| Load A (`Load<cb_input_a, D0>`) | Unpack (L1→DEST) | No (unpack precision is controlled by `UnpackToDestMode`, not by `math_fidelity`) | UnpackToDestFp32 |
| Load B (`Load<cb_input_b, D1>`) | Unpack (L1→DEST) | No | UnpackToDestFp32 |
| Sigmoid(D1) | **SFPU** (LUT2 instruction) | **No** — `math_fidelity` gates the FPU only; SFPU LUT is unaffected | `math_approx_mode` is **not explicitly set** in `ComputeConfigDescriptor`. The hard-coded `Approx::Exact` template arg on `Sigmoid<>` in the kernel selects the non-approximate LUT path *regardless* of `math_approx_mode` |
| D0 × D1 (`SfpuMul`) | **SFPU** (`mul_binary_tile`) | **No** — SFPU multiply is not fidelity-gated; `math_fidelity=HiFi4` has no effect on this op | HiFi4 (irrelevant for this kernel) |
| Pack D0 → output CB | Pack (DEST→L1) | No | fp32 → fp32 (no truncation) |

- **User-configurable**: **No.** The entry point `glu_fused(input_tensor)` does not accept a `compute_kernel_config` parameter; `math_fidelity`, `fp32_dest_acc_en`, and `unpack_to_dest_mode` are hard-coded in `glu_fused_program_descriptor.py:166-173`.
- **Recommended minimum fidelity**: irrelevant — the configured `HiFi4` is unused by this kernel's compute path. (HiFi4 + fp32_dest_acc_en is documented to have a hardware bug on Wormhole B0 (#38306) for FPU mul/matmul, but since this op only uses SFPU ops, that bug is not triggered.)

---

## Tile-Boundary Precision

| Aspect | Value / formula |
|--------|-----------------|
| Total output tiles | `total_output_tiles = (N · C · H · (W/2)) / (32·32) = (N·C·H · Wt_half) / 32` where `Wt_half = W/64` |
| Tiles consumed in a single reduction | n/a (no reduction) |
| DEST capacity | 4 tiles (half-sync + `fp32_dest_acc_en=true`) per `tile_regs_acquire` |
| Chain stride | 2 DEST slots used (`D0` for A, `D1` for B/sigmoid(B)) |
| Auto-batch factor | `DEST_AUTO_LIMIT / chain_stride = 4 / 2 = 2` chain iterations per acquire (i.e. up to 2 output tiles processed per acquire/commit cycle) |
| L1 round-trips per output tile (between compute phases) | **0** — entire chain stays in DEST until final pack |
| Intermediate CB format | n/a — no intermediate accumulator CB |
| Assessment | Pointwise op with a single fp32 → fp32 pack per output tile. There is **no tile-boundary precision loss** because there are no tile boundaries that an intermediate value has to survive |

---

## Configuration Exposure

| Setting | Exposed to user | Default in this op | Notes |
|---------|:--------------:|-------------------|-------|
| `fp32_dest_acc_en` | ✗ | `True` (hard-coded) | Important for fp32 input — without this, every DEST write would truncate to bf16 |
| `math_fidelity` | ✗ | `HiFi4` (hard-coded) | **Has no observable effect on this kernel** — no FPU multiply path is used. Could be safely lowered to LoFi for compile-time consistency without changing numerical results, but doing so would not improve throughput either, since SFPU ops are not gated by fidelity |
| `math_approx_mode` | ✗ | Not set (defaults from `ComputeConfigDescriptor`); **overridden** at the call site by `Sigmoid<Approx::Exact, …>` template arg, which forces the precise sigmoid LUT regardless of the global setting | The `Approx::Exact` template arg on `Sigmoid<>` is the effective gate for sigmoid precision in this kernel |
| `packer_l1_acc` | ✗ | Not set (defaults to false) | n/a — no L1 accumulation needed for a pointwise op |
| `UnpackToDestFp32` (per-CB) | ✗ | Set on both input CBs | Essential to preserve fp32 precision on the unpack path |
| Phase-0 shape constraints (`W%64==0`, `H%32==0`, `rank==4`) | ✗ | Hard-coded validation | Eliminates the non-tile-aligned masking question by construction |

---

## Key Observations

1. **No accumulation, no transcendental-other-than-sigmoid → minimal stability risk.** The op has exactly one precision event of any consequence: the 6-piece piecewise-linear sigmoid LUT. Everything else (unpack, multiply, pack) is bit-exact fp32. The "high accuracy" config knobs (HiFi4, fp32_dest_acc_en, UnpackToDestFp32) are mostly defensive — only `fp32_dest_acc_en=True` and `UnpackToDestFp32` are doing real work for this op; `HiFi4` is unused because the only multiply (`SfpuMul`) bypasses the FPU.

2. **The "Exact" sigmoid is still a LUT approximation, not `1/(1+exp(-x))`.** Users expecting `torch.sigmoid`-level precision should be aware: the WH/BH `_calculate_sigmoid_` implementation uses a 6-piece linear fit with FP16-encoded coefficients (≲ 1% absolute error on each segment, by construction of the linear fit). For typical neural-network activations this is acceptable; for stability-critical workloads (gating signals near saturation), the LUT's slope discontinuities at the segment boundaries (0.5, 1.0, 1.5, 2.0, 4.0) introduce small kinks. The LUT does saturate cleanly (no Inf/NaN, no negative-for-very-negative pathology), so the multiply A·sigmoid(B) is safe.

3. **The hard-coded `HiFi4 + fp32_dest_acc_en=true` combination is documented to be buggy on Wormhole B0 (#38306) for FPU mul/matmul** — but this op uses neither, so the bug is not triggered. The `HiFi4` setting is effectively dead configuration for this kernel.

4. **Configuration is not user-exposed.** The entry point `glu_fused(input_tensor)` takes only the tensor — no `compute_kernel_config`, no override path. This is reasonable for Phase 0 (one shape regime, one dtype, one precision target), but means future float16/bfloat16/fp8 input support cannot ride the current entry point without an API change. If lower-precision inputs are ever supported, the same kernel structure would still be safe (fp32 DEST + UnpackToDestFp32 protects against the SrcA/SrcB TF32 truncation that hits naive bf16-pass-through kernels).

5. **Phase-0 shape constraints (`W%64==0`, `H%32==0`) eliminate the non-tile-aligned hazard at validation time.** If those guards are ever relaxed, the compute kernel would need partial-scaler or mask-tile handling on the W boundary (since each *half* of the input must be tile-aligned for the reader's split arithmetic to be correct). The `W%64==0` constraint specifically enforces that *each half* is tile-aligned, not just the full W. This is a correctness gate, not just a precision gate.
