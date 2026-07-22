<!--
SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
SPDX-License-Identifier: Apache-2.0
-->

# Quasar ↔ Wormhole LLK support matrix — eltwise binary ops, fusable activations & subtile broadcast

**Audience: the LLK team.** **Wormhole is the baseline:** every row below is a binary op, fusable
activation, or broadcast primitive that `binary_ng` can emit/use and that **Wormhole supports**. The
**Quasar** column is the live status. This is a *matrix, not a gaps list* — the **rows are stable and do
not change as support lands; only the Quasar cell flips** (`kernel` → `bridge` → `✓`). A "gap" is simply
any Quasar cell that is not yet `✓`. Over time the WH and Quasar columns converge.

- **How to maintain:** when the LLK team lands a primitive, flip its Quasar cell — do **not** delete the
  row. A new op WH gains becomes a new row (WH `✓`, Quasar per its state). Re-derive live any time with
  `qualify_quasar_binary.py --coverage` / `--supports <op>`.

## Legend

| Quasar cell | Meaning |
|---|---|
| `✓` | Supported — present at all 3 layers with an explicit Quasar branch **and it compiles**. |
| `✓*` | Supported **and sim-certified** on the QSR simulator (craq-sim). |
| `✓!` | LLK primitive **compiles**, but the op is **runtime-wrong** end-to-end on the `binary_ng` DFB path (see Caveats). NOT a working claim. |
| `bridge` | **Tier-1, cheap.** The Quasar ckernel **and** `SfpuType` slot already exist; only the layer-1 compute-API gate and/or the layer-2 `calculate_<op>` bridge is missing. |
| `kernel` | **Tier-2, real LLK work.** No Quasar `_calculate_<op>_` ckernel and/or no `SfpuType` slot. |
| `broken` | The Quasar bridge/ckernel EXISTS but **fails to JIT-compile** — a bug to fix, not a port. |
| `format` | Op exists, but a dtype (uint16/uint32/int32/block-float) is `static_assert`-blocked. |
| `—` | Not applicable (op/dtype not defined on that target). |

> **Static 3-layer presence is necessary but NOT sufficient — only a compile/sim run confirms `✓`.**
> An op can have all three layers present yet still fail to compile, so a static `✓` can be wrong until a
> sim run confirms it. Prefer `✓*` (sim-certified) claims over static-inspection `✓`.

WH column is `✓` throughout by construction (it is the baseline); a non-`✓` WH cell would flag an op WH
itself lacks.

## Scope & method

- **Binary rows** = every `BinaryOpType` `binary_ng`'s `OpConfig` ctor routes (FPU `add/sub/mul`, the
  activation-decomposed "derived" ops, and the `is_binary_sfpu_op` SFPU subset). `ADDALPHA`/`SUBALPHA` are
  the only two enum members `binary_ng` does **not** route (the ctor `TT_THROW`s — separate composite
  ops), so they are out of scope.
- **Activation rows** = every `UnaryOpType` `binary_ng` can fuse as an lhs/rhs/post activation — i.e.
  every op `unary_op_utils.cpp::get_op_init_and_func` handles (nearly the whole enum).
- **Broadcast rows** (Table 3) = the `unary_bcast<BroadcastType>` primitive `binary_ng`'s Quasar DFB
  factory uses to realize `SubtileBroadcastType != NONE` for a **single** operand
  (`SCALAR`/`ROW`/`COL`); the mixed types (`ROW_A_COL_B`/`ROW_B_COL_A`) aren't op-wired yet, so they have
  no row here (tracked in `../QUASAR_PARITY_GAPS.md`).
- **Which `tt_llk_quasar` tree is authoritative** — there are **two** and they disagree. The build uses
  **`tt_metal/tt-llk/tt_llk_quasar/`** (on the kernel `-I` path — `tt_metal/hw/CMakeLists.txt`). The other,
  `tt_metal/third_party/tt_llk/tt_llk_quasar/`, is **NOT** on the include path (IDE glob only) and is
  staler-but-richer → inventorying it gives **false positives**. Classified against the build tree only.
- **3-layer test for `✓`** (all present, none behind a bare `#ifndef ARCH_QUASAR`):
  1. Compute API `tt_metal/hw/inc/api/compute/…` — the `<op>_tile()` the kernel calls. **Available** =
     `#ifndef ARCH_QUASAR … #else <quasar> #endif`; **gated out** = `#ifndef` with no `#else`, or a
     dedicated header with zero `ARCH_QUASAR` mentions. The big gated blocks in `compute_kernel_api.h` are
     ~`263-671` and ~`828-1273`.
  2. Quasar LLK-API bridge `tt_metal/hw/ckernels/quasar/metal/llk_api/llk_sfpu/` — `ckernel::sfpu::calculate_<op>`.
  3. Quasar ckernel `tt_metal/tt-llk/tt_llk_quasar/common/inc/{sfpu,experimental}/` — `_calculate_<op>_`,
     plus a `SfpuType` entry in `…/llk_lib/llk_defs.h`.
- `bridge` vs `kernel` is decided by layers 2-3: ckernel + `SfpuType` present ⇒ at worst `bridge`.

## Table 1 — Binary ops (WH baseline)

Dtype is called out in the Quasar cell where it differs (bf16 is the model-relevant path).

### Arithmetic
| Op | Route | WH | Quasar | Evidence / to-close |
|---|---|:--:|---|---|
| `add` | FPU (bf16) · SFPU (fp32/int) | ✓ | `✓*` bf16 · `✓!` int32 · `✓*` fp32 | bf16/fp32 sim-certified via `binary_ng`'s no-bcast AND tensor-scalar suites; int32 `ckernel_sfpu_add.h` compiles but is silent-wrong on the DFB compute path — see Caveats |
| `sub` | FPU (bf16) · SFPU (fp32/int) | ✓ | `✓*` bf16 · `kernel` int32 · `✓*` fp32 | bf16/fp32 sim-certified via `binary_ng`'s no-bcast AND tensor-scalar suites; fp32 `sub_binary_tile` (Quasar branch) + `calculate_sfpu_binary` SUB; int32 `sub_int_sfpu.h` WH-only |
| `mul` | FPU (bf16) · SFPU | ✓ | `✓*` bf16 · `✓*` fp32 · `✓!` int32 | bf16/fp32 sim-certified via `binary_ng`'s no-bcast AND tensor-scalar suites; `mul_binary_tile` (Quasar branch); int32 `ckernel_sfpu_mul_int32.h` compiles but is silent-wrong on the DFB compute path — see Caveats |
| `div` | SFPU | ✓ | `✓*` bf16/fp32 · `kernel` int32 | bf16/fp32 also sim-certified via the tensor-scalar suite; `div_binary_tile` ✓; `div_int32_tile` unported |
| `rsub` | FPU+`NEG` (bf16) · SFPU | ✓ | `kernel` | bf16 blocked by `NEG` (Tier-2 activation); fp32/int SFPU rsub gated |

### Comparison
| Op | Route | WH | Quasar | Evidence / to-close |
|---|---|:--:|---|---|
| `gt` | SFPU (int) / post-`GTZ` | ✓ | `kernel` float/bf16 · `bridge` int32 | int `gt_int` wrapper exists; float-compare ckernel absent |
| `lt` `le` `ge` | SFPU (int) / post-`{L,LE,GE}TZ` | ✓ | `kernel` float/bf16 · `bridge` int32 | int ckernel `ckernel_sfpu_binary_comp.h` exists; add `lt/le/ge_int` wrappers |
| `eq` `ne` | SFPU (fp) / post-`{EQ,NE}Z` | ✓ | `kernel` | float `eq/ne` unported; int path routes `sub_int` (also unported) |

### Integer arithmetic (int32; uint16/uint32 → `format`)
| Op | Route | WH | Quasar | Evidence / to-close |
|---|---|:--:|---|---|
| `add_int` | SFPU | ✓ | `✓!` | `ckernel_sfpu_add.h` compiles but is silent-wrong on the DFB — see Caveats |
| `mul_int` | SFPU | ✓ | `✓!` | `ckernel_sfpu_mul_int32.h` compiles but is silent-wrong on the DFB — see Caveats |
| `sub_int` / `rsub_int` | SFPU | ✓ | `kernel` | WH-only header |
| `gcd` / `lcm` | SFPU | ✓ | `kernel` | |
| `div_int` / `floor_div` / `trunc_div` | SFPU | ✓ | `kernel` | `div_int32_sfpu.h`, `div_int32_floor.h` |
| `remainder` / `fmod` | SFPU | ✓ | `kernel` | |
| `bitwise_and` / `or` / `xor` | SFPU | ✓ | `kernel` | `binary_bitwise_sfpu.h` WH-only |
| `left_shift` / `right_shift` / `logical_right_shift` | SFPU | ✓ | `kernel` | `binary_shift.h` WH-only |

### Reductions
| Op | Route | WH | Quasar | Evidence / to-close |
|---|---|:--:|---|---|
| `maximum` / `minimum` | SFPU | ✓ | `✓` float+int32 · `format` uint | `ckernel_sfpu_binary_max_min.h` bridge; uint `static_assert` |

### Transcendental / misc binary (fp32 SFPU)
| Op | Route | WH | Quasar | Evidence / to-close |
|---|---|:--:|---|---|
| `power` | SFPU | ✓ | `kernel` | `power_binary_tile` gated |
| `xlogy` | SFPU | ✓ | `kernel` | no XLOGY branch in Quasar `calculate_sfpu_binary` (needs the `log` helper) |
| `atan2` | SFPU | ✓ | `kernel` | |
| `isclose` | SFPU | ✓ | `kernel` | |

### Quantization
| Op | Route | WH | Quasar | Evidence / to-close |
|---|---|:--:|---|---|
| `quant` / `requant` / `dequant` | SFPU | ✓ | `kernel` | no quant ckernels under `quasar/…/llk_sfpu/` |

### Ternary
| Op | Route | WH | Quasar | Evidence / to-close |
|---|---|:--:|---|---|
| `where` (`tst` / `tts`) | SFPU | ✓ | `✓` | `ckernel_sfpu_where.h` bridge |

### Derived (FPU + fused activation — Quasar status follows the activation pieces, see Table 2)
| Op | Decomposition | WH | Quasar (bf16) | Blocking piece |
|---|---|:--:|---|---|
| `squared_difference` | `SUB` + post `SQUARE` | ✓ | `✓` | `SQUARE` ✓ |
| `bias_gelu` | `ADD` + post `GELU` | ✓ | `✓` | `ADD` + post `GELU`, both supported |
| `hypot` | `SQUARE`·`SQUARE`·`ADD`·post `SQRT` | ✓ | `✓` | all ✓ |
| `logical_and` / `or` / `xor` | `NEZ`·`NEZ`·(`MUL`/`ADD`)·post `NEZ` | ✓ | `✓` | `NEZ` ✓ |
| `logaddexp` | `EXP`·`EXP`·`ADD`·post `LOG` | ✓ | `kernel` | `LOG` (Tier-2) |
| `logaddexp2` | `EXP2`·`EXP2`·`ADD`·post `LOG2` | ✓ | `kernel` | `EXP2`, `LOG2` (Tier-2) |
| `ldexp` | rhs `EXP2` · `MUL` | ✓ | `kernel` | `EXP2` (Tier-2) |

## Table 2 — Fusable activations (WH baseline)

Rows grouped by family (fixed). Currently `✓` on Quasar: **relu\*, silu\*, sigmoid\*, tanh\*, square\*,
gelu\*, exp, sqrt, rsqrt, reciprocal, eqz/nez/gtz/ltz/gez/lez** (\* sim-certified).
Everything else is `bridge` or `kernel`.

### relu family
| Activation | WH | Quasar | Evidence / to-close |
|---|:--:|---|---|
| `relu` | ✓ | `✓*` | `eltwise_unary/relu.h` Quasar branch; ResNet50-certified |
| `relu_max` | ✓ | `bridge` | `_relu_max_` + `SfpuType::relu_max` exist; un-gate `relu.h:52-177` |
| `relu_min` | ✓ | `bridge` | `_relu_min_` + `SfpuType::relu_min` exist; un-gate |
| `leaky_relu` | ✓ | `bridge` | `_calculate_lrelu_` + `SfpuType::lrelu` exist; un-gate |
| `relu6` | ✓ | `kernel` | no ckernel / `SfpuType` |

### gelu / sigmoid / silu / tanh / square
| Activation | WH | Quasar | Evidence / to-close |
|---|:--:|---|---|
| `gelu` | ✓ | `✓*` | bridge + `#else` branch (`gelu.h:42-55`) + `SfpuType::gelu`; compiles and sim-certified (interleaved/height × post/lhs). |
| `gelu_tanh` | ✓ | `kernel` | inside `gelu.h` `#ifndef` (57-140); no `gelu_tanh` ckernel/`SfpuType` |
| `sigmoid` | ✓ | `✓*` | `compute_kernel_api.h:133` `#ifdef ARCH_QUASAR` branch; sim-certified |
| `silu` | ✓ | `✓*` | `:162` branch; Llama SwiGLU-certified |
| `tanh` | ✓ | `✓*` | `:215-228` `#else` branch + `ckernel_sfpu_tanh.h` + `SfpuType::tanh`; sim-certified |
| `square` | ✓ | `✓*` | `:244-250` `#else` branch + `ckernel_sfpu_square.h` + `SfpuType::square`; sim-certified |

### exp / log
| Activation | WH | Quasar | Evidence / to-close |
|---|:--:|---|---|
| `exp` | ✓ | `✓` | `eltwise_unary/exp.h` Quasar path |
| `exp2` | ✓ | `kernel` | no ckernel/`SfpuType` (also blocks `ldexp`/`logaddexp2`) |
| `expm1` | ✓ | `kernel` | |
| `log` | ✓ | `kernel` | blocks `logaddexp` |
| `log2` | ✓ | `kernel` | blocks `logaddexp2` |
| `log10` | ✓ | `kernel` | |
| `log1p` | ✓ | `kernel` | |

### reciprocal / roots
| Activation | WH | Quasar | Evidence / to-close |
|---|:--:|---|---|
| `reciprocal` | ✓ | `✓` | `ckernel_sfpu_recip.h` |
| `sqrt` | ✓ | `✓` | `ckernel_sfpu_sqrt.h` |
| `rsqrt` | ✓ | `✓` | `ckernel_sfpu_rsqrt.h` |
| `cbrt` | ✓ | `kernel` | |

### compare-to-zero  (SUPPORTED) / scalar-compare (gap)
| Activation | WH | Quasar | Evidence / to-close |
|---|:--:|---|---|
| `eqz` `nez` `gtz` `ltz` `gez` `lez` | ✓ | `✓` | `eltwise_unary/comp.h` `#else` branches |
| `unary_eq/ne/gt/lt/ge/le` (vs a scalar) | ✓ | `kernel` | only compare-to-**zero** exists on Quasar |

### trig (full family)
| Activation | WH | Quasar | Evidence / to-close |
|---|:--:|---|---|
| `sin` `cos` `tan` `asin` `acos` `atan` `sinh` `cosh` `asinh` `acosh` `atanh` | ✓ | `kernel` | `trigonometry.h` — no `ARCH_QUASAR` path |

### error / bessel / gamma
| Activation | WH | Quasar | Evidence / to-close |
|---|:--:|---|---|
| `erf` `erfc` `erfinv` | ✓ | `kernel` | |
| `i0` `i1` | ✓ | `kernel` | |
| `lgamma` `digamma` `polygamma` | ✓ | `kernel` | |

### activation shapes
| Activation | WH | Quasar | Evidence / to-close |
|---|:--:|---|---|
| `elu` `celu` `selu` | ✓ | `kernel` | |
| `hardsigmoid` `hardswish` `hardtanh` `hardshrink` `softshrink` `softsign` | ✓ | `kernel` | |
| `softplus` `tanhshrink` `mish` `hardmish` `logsigmoid` `xielu` | ✓ | `kernel` | |

### sign / misc math
| Activation | WH | Quasar | Evidence / to-close |
|---|:--:|---|---|
| `abs` | ✓ | `bridge` | `common/inc/experimental/ckernel_sfpu_abs.h` + `SfpuType::abs`; un-gate `abs_tile` (~`:437`) |
| `abs_int32` | ✓ | `format` | int32 abs path inside the gated int block (~`:462`) |
| `neg` | ✓ | `kernel` | blocks `rsub` (FPU path) |
| `sign` `signbit` | ✓ | `kernel` | |
| `heaviside` | ✓ | `kernel` | |
| `power` `power_iterative` `rpow` `rdiv` | ✓ | `kernel` | |
| `logit` | ✓ | `kernel` | |

### rounding / rem-mod
| Activation | WH | Quasar | Evidence / to-close |
|---|:--:|---|---|
| `floor` `ceil` `round` `trunc` `frac` | ✓ | `kernel` | `rounding.h` |
| `remainder` `fmod` | ✓ | `kernel` | |

### is-checks / logic
| Activation | WH | Quasar | Evidence / to-close |
|---|:--:|---|---|
| `isinf` `isposinf` `isneginf` `isnan` `isfinite` | ✓ | `kernel` | `isinf_isnan.h` |
| `logical_not` | ✓ | `kernel` | |

### bitwise / shift (int dtype)
| Activation | WH | Quasar | Evidence / to-close |
|---|:--:|---|---|
| `bitwise_and` `or` `xor` `not` | ✓ | `kernel` | |
| `left_shift` `right_shift` | ✓ | `kernel` | |

### scalar-parameterized arithmetic
| Activation | WH | Quasar | Evidence / to-close |
|---|:--:|---|---|
| `mul_unary_sfpu` (`× const`) | ✓ | `bridge` | `ckernel_sfpu_binop_with_unary.h` exists, **Mul-only** |
| `add_/sub_/div_unary_sfpu`, scalar `rsub` | ✓ | `kernel` | binop-with-unary `static_assert`s Mul-only |
| `maximum` / `minimum` (vs a scalar) | ✓ | `kernel` | unary scalar max/min gated (~`:1110`/`:1237`) |

### infra (not math activations — out of scope)
`identity`, `fill`, `typecast`, `bitcast`, `where_tss`, `clamp_tss`, `threshold`, `prelu_sfpu`,
`alt_complex_rotate90`, `tiled_prod`. (`where` and `typecast` do have Quasar LLK paths + bridges, but they
are ternary-select / dtype-cast infra, not activations.)

## Table 3 — Subtile broadcast primitive (`unary_bcast`)

Unlike Table 1/2 (`BinaryOpType` / `UnaryOpType` rows), this tracks the **single-operand broadcast
primitive** `unary_bcast<BroadcastType::{SCALAR,ROW,COL}>` (`tt_metal/hw/inc/api/compute/bcast.h`) that
`binary_ng`'s Quasar DFB factory uses to pre-broadcast the smaller operand into DST before the FPU/SFPU
binary op runs on it. WH/BH carry the same header's non-Quasar branch. An op-level `SubtileBroadcastType`
(e.g. `SCALAR_A` vs `SCALAR_B`) picks which operand feeds this primitive; the primitive itself only cares
about the broadcast dimension, hence one row per dimension.

| Broadcast dimension | WH | Quasar | Evidence / to-close |
|---|:--:|---|---|
| `unary_bcast<SCALAR>` | ✓ | `✓*` | Quasar `#else` branch (`bcast.h`) + `llk_unpack_unary_broadcast_operands.h` / `llk_math_unary_broadcast.h` SCALAR body; sim-certified standalone (`test_unary_broadcast_quasar.py`) **and** through `binary_ng` (`test_binary_ng_bcast.py`, `SubtileBroadcastType::SCALAR_A/B`) |
| `unary_bcast<ROW>` | ✓ | `✓*` | same evidence, ROW body; op-certified via `SubtileBroadcastType::ROW_A/B` |
| `unary_bcast<COL>` | ✓ | `✓*` | same evidence, COL body; op-certified via `SubtileBroadcastType::COL_A/B` |

All three dimensions lower to the same Quasar LLK path — the MOVB2D srcB→dest datacopy
(`llk_math_eltwise_unary_datacopy<DataCopyType::B2D, ...>`), differentiated only by broadcast constants
(`dst_lo`/`bcast0`/`srcb_col_inc`). This is a different mechanism from the two-operand
`add_tiles_bcast_rows`/`mul_tiles_bcast_cols`/etc. shorthand family further down `bcast.h` (`ELWADD`-style,
`llk_unpack_AB<BroadcastType>`), which is WH/BH-only and not used by the Quasar DFB kernels.

- **32-bit formats are gated off:** the Quasar branch's `enable_unpack_to_dest` check omits
  `DataFormat::UInt32` — Quasar has no uint32 device format (its 32-bit formats are `Float32`/`Int32`; the
  enum slot WH/BH use for `UInt32` is `MxFp4_2x_B` on Quasar) — and asserts if a 32-bit format would need
  the A2D unpack-to-dest path, which isn't implemented on Quasar. bf16 only for now.
- `reconfigure_unary_bcast` (mid-program bcast-type/format switch) is `#ifndef ARCH_QUASAR`-only; Quasar
  re-`init`s per broadcast type instead.
- No sim/LLK bug surfaced while certifying SCALAR/ROW/COL through the op — all 112 broadcast cases in
  `test_binary_ng_bcast.py` pass on the QSR sim alongside the 88-case no-bcast regression suite.

## Priorities (by model impact)

1. **int `lt`/`le`/`ge`** (binary) — `bridge`, trivial (ckernel already handles lt/gt/le/ge).
2. **`abs`, `leaky_relu`, `relu_max`, `relu_min`** (activations) — `bridge`, cheap (un-gate; ckernels + slots exist).
3. **`gelu_tanh`** — transformer MLPs using tanh-GELU (`kernel`; plain `gelu` is supported).
4. **`log` / `exp2` / `log2`** — softmax-adjacent; unblock `logaddexp`/`logaddexp2`/`ldexp` (`kernel`).
5. Everything else as op/model demand arises.

**Already done — no LLK work:** `relu` (ResNet50), `silu` (Llama SwiGLU), `sigmoid`, `tanh`, `square`, `gelu`
— all sim-certified — plus the arithmetic/`where`/compare-to-zero core (bf16/fp32 add/sub/mul/div now
sim-certified through both `binary_ng`'s tensor-tensor no-broadcast path AND its tensor-scalar path, the
latter via a writer-filled RHS tile) and single-operand subtile broadcast `unary_bcast` SCALAR/ROW/COL
(Table 3), sim-certified through `binary_ng` itself.

## Closing a gap — the pattern

- **`bridge` → `✓`:** add `ckernel::sfpu::calculate_<op>` in `hw/ckernels/quasar/…/llk_sfpu/ckernel_sfpu_<op>.h`
  (copy `ckernel_sfpu_silu.h` / `ckernel_sfpu_gelu.h`), then change the compute-API bare `#ifndef ARCH_QUASAR`
  around `<op>_tile()` into an `#ifndef … #else … #endif` branch (mirror `gelu.h:42-55`).
- **`kernel` → `bridge`:** add `_calculate_<op>_` in `tt_llk_quasar/common/inc/sfpu/` + a `SfpuType` entry
  (`tt_llk_quasar/llk_lib/llk_defs.h`).
- **No ops-side change needed:** `binary_ng` already emits the correct `<op>_tile()` / `<op>_binary_tile()`
  calls; every gap here is below the compute-API line.
- **Verify:** add the op to the Quasar LLK test, run on the QSR sim (`run_test.sh`), then re-run
  `qualify_quasar_binary.py` to confirm the Quasar cell flips.

## Key files

- Op routing / SFPU-fn map: `ttnn/cpp/ttnn/operations/experimental/quasar/binary_ng/device/binary_ng_utils.cpp`
  (`OpConfig` ctor `:152-399`, `get_sfpu_init_fn` `:402`, `add_activation_defines` `:570`),
  `…/binary_ng_device_operation.cpp` (`is_binary_sfpu_op` `:18`).
- Activation emit: `binary_ng_utils.cpp::add_activation_defines` → `unary/common/unary_op_utils.cpp::get_op_init_and_func`.
- Compute-API gates: `tt_metal/hw/inc/api/compute/eltwise_binary_sfpu.h` (binary SFPU),
  `compute_kernel_api.h` (gated blocks ~`:263-671`, ~`:828-1273`), `eltwise_unary/*.h`, per-op SFPU headers.
- Quasar LLK-API bridges: `tt_metal/hw/ckernels/quasar/metal/llk_api/llk_sfpu/`.
- Quasar ckernels + `SfpuType`: `tt_metal/tt-llk/tt_llk_quasar/common/inc/{sfpu,experimental}/`,
  `…/llk_lib/llk_defs.h`. **(Authoritative build tree — NOT `third_party/tt_llk/tt_llk_quasar/`.)**
- Quasar LLK tests: `tt_metal/tt-llk/tests/python_tests/quasar/test_eltwise_{binary,binary_sfpu,unary_sfpu}_quasar.py`
  (WH baselines: `tests/python_tests/test_eltwise_binary.py`, `test_eltwise_unary_sfpu.py`).
- Broadcast primitive (Table 3): `tt_metal/hw/inc/api/compute/bcast.h` (`unary_bcast`/`_init`/`_uninit`) ·
  Quasar LLK-API bridges `tt_metal/hw/ckernels/quasar/metal/llk_api/{llk_unpack_A_api.h,
  llk_math_unary_datacopy_api.h}` · core LLK
  `tt_metal/tt-llk/tt_llk_quasar/llk_lib/{llk_unpack_unary_broadcast_operands.h,
  llk_math_unary_broadcast.h}`.
- Broadcast tests: standalone LLK `tt_metal/tt-llk/tests/python_tests/quasar/test_unary_broadcast_quasar.py`;
  through the op `tests/ttnn/nightly/unit_tests/operations/experimental/quasar/test_binary_ng_bcast.py`.

## Caveats

- Quasar cells are static compile-availability, except `✓*` (sim-certified correct) and `✓!` (sim-observed runtime-wrong).
- **Two `tt_llk_quasar` trees** — classify against `tt_metal/tt-llk/tt_llk_quasar/` (build path), not
  `tt_metal/third_party/tt_llk/tt_llk_quasar/` (IDE-only, staler-but-richer → false positives).
- `format` cells (block-float / uint16 / uint32 / int32) may be intentional Quasar arch choices (MX) —
  confirm with the arch team before porting.
- **`add`/`mul` int32 are the `✓!` exception:** both compile and run on `binary_ng`'s Quasar DFB path, but
  return wrong output at runtime (all-zero tiles on the tensor-scalar path; garbage on the tensor-tensor
  no-broadcast path) — a compute-path bug (suspected locus: the op factory's `set_unpack_mode`, which only
  emits an SFPU unpack mode for `Float32`, never `Int32`), not a compile or LLK-primitive gap. The
  tensor-**scalar** AND tensor-**tensor** no-broadcast gates both exclude int32 (→ descriptor, a clean
  "unsupported on Quasar" throw), so int32 no longer reaches the broken DFB path; the underlying compute bug
  is unfixed (int32 would be wrong if run there) (see `../QUASAR_PARITY_GAPS.md` §2 and §7). Do not read the
  `✓!` cells as a working claim.
- Keep current on tt-llk / craq-sim pin bumps: re-run `qualify_quasar_binary.py --coverage` and reconcile.
  The LLK team's `quasar-llk.yml` on the QSR sim is the upstream source of truth.
