# `_with_dt` LLK Init Function Tree

Decomposition tree of all `_with_dt` compute kernel API functions and their wrapper expansions. All bodies are inline `ALWI`. **22 functions total.**

Pattern: `_with_dt` = `reconfig_data_format[_srca/_srcb]` (or `pack_reconfig_data_format`) + plain init.

## Master index

Three different gating mechanisms exist across the codebase. When picking a function, check which gating applies — it determines whether reconfig is always emitted.

| Gating | Meaning |
|---|---|
| `#ifndef ARCH_QUASAR` | Always emit reconfig (skipped only on Quasar — TODO in code). |
| `#if defined FP32_DEST_ACC_EN` | Reconfig **only** in fp32-accum builds. Non-fp32 builds reduce wrapper to plain init. |
| _none_ | Always emit reconfig. Uses `DST_ACCUM_MODE` template arg on the LLK call directly. |

| Function | Header | Gating |
|---|---|---|
| `add_bcast_cols_init_short_with_dt` | `ttnn/cpp/ttnn/kernel/compute/moreh_common.hpp` | `FP32_DEST_ACC_EN` |
| `add_bcast_rows_init_short_with_dt` | `ttnn/cpp/ttnn/kernel/compute/moreh_common.hpp` | `FP32_DEST_ACC_EN` |
| `add_bcast_scalar_init_short_with_dt` | `ttnn/cpp/ttnn/kernel/compute/moreh_common.hpp` | `FP32_DEST_ACC_EN` |
| `add_tiles_init_with_dt` | `ttnn/cpp/ttnn/kernel/compute/moreh_common.hpp` | `FP32_DEST_ACC_EN` |
| `copy_tile_init_with_dt` | `ttnn/cpp/ttnn/kernel/compute/moreh_common.hpp` | `FP32_DEST_ACC_EN` |
| `copy_tile_to_dst_init_short_with_dt` | `tt_metal/hw/inc/api/compute/tile_move_copy.h` | `#ifndef ARCH_QUASAR` |
| `fast_tilize_init_with_dt` | `tt_metal/hw/inc/api/compute/tilize.h` | `#ifndef ARCH_QUASAR` |
| `mm_block_init_short_with_dt` | `tt_metal/hw/inc/api/compute/matmul.h` | `#ifndef ARCH_QUASAR` |
| `mm_init_short_with_dt` | `tt_metal/hw/inc/api/compute/matmul.h` | `#ifndef ARCH_QUASAR` |
| `mul_bcast_cols_init_short_with_dt` | `ttnn/cpp/ttnn/kernel/compute/moreh_common.hpp` | `FP32_DEST_ACC_EN` |
| `mul_bcast_rows_init_short_with_dt` | `ttnn/cpp/ttnn/kernel/compute/moreh_common.hpp` | `FP32_DEST_ACC_EN` |
| `mul_tiles_bcast_scalar_init_short_with_dt` | `ttnn/cpp/ttnn/kernel/compute/moreh_common.hpp` | `FP32_DEST_ACC_EN` |
| `mul_tiles_init_with_dt` | `ttnn/cpp/ttnn/kernel/compute/moreh_common.hpp` | `FP32_DEST_ACC_EN` |
| `pack_tile_with_dt` | `ttnn/cpp/ttnn/kernel/compute/moreh_common.hpp` | `FP32_DEST_ACC_EN` |
| `reduce_init_short_with_dt` | `ttnn/cpp/ttnn/kernel_lib/reduce_helpers_compute.inl` | _none_ |
| `reduce_with_matmul_init_with_dt` | `ttnn/cpp/ttnn/kernel_lib/reduce_helpers_compute.inl` | _none_ |
| `sub_bcast_cols_init_short_with_dt` | `ttnn/cpp/ttnn/kernel/compute/moreh_common.hpp` | `FP32_DEST_ACC_EN` |
| `sub_bcast_rows_init_short_with_dt` | `ttnn/cpp/ttnn/kernel/compute/moreh_common.hpp` | `FP32_DEST_ACC_EN` |
| `sub_tiles_bcast_scalar_init_short_with_dt` | `ttnn/cpp/ttnn/kernel/compute/moreh_common.hpp` | `FP32_DEST_ACC_EN` |
| `sub_tiles_init_with_dt` | `ttnn/cpp/ttnn/kernel/compute/moreh_common.hpp` | `FP32_DEST_ACC_EN` |
| `tilize_init_short_with_dt` | `tt_metal/hw/inc/api/compute/tilize.h` | `#ifndef ARCH_QUASAR` |
| `tilize_uninit_with_dt` | `tt_metal/hw/inc/api/compute/tilize.h` | `#ifndef ARCH_QUASAR` |

---

## By task

### Tile copy

```
copy_tile_to_dst_init_short_with_dt(old_cb, new_cb, transpose=0)   [tile_move_copy.h]
├── reconfig_data_format_srca(old_cb, new_cb)
│   ├── UNPACK: llk_unpack_reconfig_data_format_srca(old, new)
│   └── MATH:   llk_math_reconfig_data_format_srca(old, new)
└── copy_tile_to_dst_init_short(new_cb, transpose)

copy_tile_init_with_dt(icb, transpose=0)                           [moreh_common.hpp]
├── if FP32_DEST_ACC_EN:
│   └── reconfig_data_format_srca(icb)
│       ├── UNPACK: llk_unpack_reconfig_data_format_srca(icb)
│       └── MATH:   llk_math_reconfig_data_format_srca(icb)
└── copy_tile_to_dst_init_short(icb, transpose)
```

`copy_tile_init_with_dt` is the single-arg wrapper around `copy_tile_to_dst_init_short_with_dt`'s plain init half (no `old_cb` tracking, conditional on fp32 accum).

### Tilize

```
tilize_init_short_with_dt(old_icb, new_icb, block, ocb)            [tilize.h]
├── MATH: llk_math_eltwise_unary_datacopy_init<A2D, tilize=true>(new_icb)
├── reconfig_data_format_srca(old_icb, new_icb)
│   ├── UNPACK: llk_unpack_reconfig_data_format_srca(old, new)
│   └── MATH:   llk_math_reconfig_data_format_srca(old, new)
└── UNPACK: llk_unpack_tilize_init(new_icb, block)

tilize_uninit_with_dt(old_icb, new_icb, ocb)                       [tilize.h]
├── UNPACK: llk_unpack_tilize_uninit(old_icb)
├── reconfig_data_format_srca(old_icb, new_icb)
│   ├── UNPACK: llk_unpack_reconfig_data_format_srca(old, new)
│   └── MATH:   llk_math_reconfig_data_format_srca(old, new)
└── PACK: llk_pack_init(ocb)                  [BLACKHOLE only]

fast_tilize_init_with_dt(icb, full_dim, ocb)                       [tilize.h]
├── reconfig_data_format(icb, icb)            [BOTH srca + srcb form]
│   ├── UNPACK: llk_unpack_reconfig_data_format(icb, icb)
│   └── MATH:   llk_math_reconfig_data_format<true,true>(icb, icb)
└── fast_tilize_init(icb, full_dim, ocb)
    ├── state_configure<SRCA, PACK>(icb, ocb)
    ├── UNPACK: llk_unpack_fast_tilize_init(icb, full_dim)
    ├── MATH:   llk_math_fast_tilize_init(icb, ...)
    └── PACK:   llk_pack_fast_tilize_init(icb, ocb, ...)
```

### Matmul

```
mm_init_short_with_dt(in0, in1, c_in_old_srca, transpose=0)        [matmul.h]
├── reconfig_data_format_srca(c_in_old_srca, in1)
│   ├── UNPACK: llk_unpack_reconfig_data_format_srca(old, in1)
│   └── MATH:   llk_math_reconfig_data_format_srca(old, in1)
└── mm_init_short(in0, in1, transpose)

mm_block_init_short_with_dt(in0, in1, old_in1, transpose, ct, rt, kt)  [matmul.h]
├── state_configure(in1, in0)
├── reconfig_data_format_srca(old_in1, in1)
│   ├── UNPACK: llk_unpack_reconfig_data_format_srca(old, in1)
│   └── MATH:   llk_math_reconfig_data_format_srca(old, in1)
└── mm_block_init_short(in0, in1, transpose, ct, rt, kt)
```

Both reconfigure srcA only — `in1_cb_id` is the new srcA operand for matmul.

### Eltwise binary (add / sub / mul, including bcast)

All 12 wrappers in `moreh_common.hpp` follow the same shape:

```
{add,sub,mul}_tiles_init_with_dt(c0, c1)                           [moreh_common.hpp]
{add,sub,mul}_bcast_{rows,cols,scalar}_init_short_with_dt(c0, c1)  [moreh_common.hpp]
├── if FP32_DEST_ACC_EN:
│   └── reconfig_data_format(c0, c1)
│       ├── UNPACK: llk_unpack_reconfig_data_format(c0, c1)
│       └── MATH:   llk_math_reconfig_data_format(c0, c1)
└── plain init: {add,sub,mul}_tiles_init / *_bcast_*_init_short
```

Plain-init names map 1:1 by stripping `_with_dt`. Two exceptions:
- `add_bcast_scalar_init_short_with_dt` calls `add_bcast_scalar_init_short` (note: source code; the bcast scalar variant for `add` is `add_bcast_scalar_init_short`).
- `sub_bcast_rows_init_short_with_dt` is **hand-inlined** — calls `llk_math_eltwise_binary_init<ELWSUB, ROW>` + `llk_unpack_AB_init<ROW>` directly instead of `sub_bcast_rows_init_short`. There is a FIXME comment in source noting the API-update need.

### Pack

```
pack_tile_with_dt(dst, icb)                                        [moreh_common.hpp]
├── if FP32_DEST_ACC_EN:
│   └── pack_reconfig_data_format(icb)
│       └── PACK: llk_pack_reconfig_data_format(icb)
└── pack_tile(dst, icb)
    └── PACK: llk_pack(dst, icb, ...)
```

This is the **only** `_with_dt` that touches PACK reconfig. All others handle UNPACK + MATH only — output dtype changes need this wrapper or an explicit `pack_reconfig_data_format` call.

### Reduce

```
reduce_with_matmul_init_with_dt(in0, in1, c_in_old_srca)           [reduce_helpers_compute.inl]
├── UNPACK: llk_unpack_reconfig_data_format_srca<DST_ACCUM_MODE>(c_in_old_srca, in1)
├── MATH:   llk_math_reconfig_data_format_srca<DST_ACCUM_MODE>(c_in_old_srca, in1)
└── reduce_with_matmul_init(in0, in1)
    ├── state_configure(in1, in0)
    ├── MATH:   llk_math_matmul_init<HiFi4, MM_THROTTLE>(in0, in1, 0)
    └── UNPACK: llk_unpack_AB_matmul_init(in0, in1, 0)

reduce_init_short_with_dt<reduce_type, reduce_dim>(old_cb, input_cb, scaler_cb)  [reduce_helpers_compute.inl]
├── UNPACK: llk_unpack_reconfig_data_format_srca<DST_ACCUM_MODE>(old_cb, input_cb)
├── MATH:   llk_math_reconfig_data_format_srca<DST_ACCUM_MODE>(old_cb, input_cb)
├── UNPACK: llk_unpack_AB_reduce_init<reduce_type, reduce_dim>(input_cb, scaler_cb)
└── MATH:   llk_math_reduce_init<reduce_type, reduce_dim, DST_ACCUM_MODE, MATH_FIDELITY>()
```

Live in `compute_kernel_lib::` namespace, not `ckernel::`. `reduce_init_short_with_dt` skips PACK reconfig — comment notes packer format remains valid from initial `reduce_init`. `reduce_with_matmul_init` uses `REDUCE_MATMUL_FIDELITY = HiFi4` (not the kernel-default `MATH_FIDELITY`).

---

## `reconfig_data_format` family (`reconfig_data_format.h`)

Each variant fans out into UNPACK + MATH calls. PACK-side reconfig is separate (`pack_reconfig_data_format`, in `pack.h`) — never auto-included by any `_with_dt` except `pack_tile_with_dt`.

```
reconfig_data_format(srca_new, srcb_new)              ← both operands, unconditional
├── UNPACK: llk_unpack_reconfig_data_format(srca_new, srcb_new)
└── MATH:   llk_math_reconfig_data_format(srca_new, srcb_new)

reconfig_data_format(srca_old, srca_new, srcb_old, srcb_new)   ← both, conditional on diff
├── UNPACK: llk_unpack_reconfig_data_format(srca_old, srca_new, srcb_old, srcb_new)
└── MATH:   llk_math_reconfig_data_format(srca_old, srca_new, srcb_old, srcb_new)

reconfig_data_format_srca(srca_new)                   ← srca only, unconditional
reconfig_data_format_srca(srca_old, srca_new)         ← srca only, conditional on diff
├── UNPACK: llk_unpack_reconfig_data_format_srca(...)
└── MATH:   llk_math_reconfig_data_format_srca(...)

reconfig_data_format_srcb(srcb_new)                   ← srcb only, unconditional
reconfig_data_format_srcb(srcb_old, srcb_new)         ← srcb only, conditional on diff
├── UNPACK: llk_unpack_reconfig_data_format_srcb(...)
└── MATH:   llk_math_reconfig_data_format_srcb(...)

pack_reconfig_data_format(new_cb)                     [pack.h, separate from above]
pack_reconfig_data_format(old_cb, new_cb)
└── PACK: llk_pack_reconfig_data_format(...)
```

Template parameters on all `reconfig_data_format` wrappers:
- `to_from_int8 = false` — set `true` for int8 paths
- `is_tile_dim_reconfig_en = false` — when `true`, also reconfigures dim/stride to `FACE_ROW_MAJOR`

---

## Decision rules

| Scenario | Action |
|---|---|
| Same op, same CBs, same dtype as last tile | Nothing. |
| New CB feeding srcA mid-kernel — copy / tilize / matmul | Use the matching `_with_dt`, OR call `reconfig_data_format_srca[(old,new)]` + plain init. |
| Eltwise binary (add/sub/mul/bcast) with new operand dtype | Use the matching `_with_dt` from `moreh_common.hpp` (conditional on `FP32_DEST_ACC_EN`), OR call `reconfig_data_format(c0, c1)` + plain init. |
| Reduce op (matmul-based or unpack/math reduce) with new input CB | Use `reduce_with_matmul_init_with_dt` or `reduce_init_short_with_dt<ptype, pdim>`. |
| Output CB dtype changed | Use `pack_tile_with_dt`, OR call `pack_reconfig_data_format(new_cb)` then `pack_tile(...)`. No other `_with_dt` touches PACK reconfig. |
| Switching kernel-wide engine mode (eltwise ↔ matmul) | One-time `*_op_init_common` / `mm_init` at kernel start (most-restrictive). Do **not** repeat per tile. |

## Hard rules

- **`hw_configure` is one-time.** Never inside per-tile loop — races compute pipeline, causes hangs/corruption.
- Skip reconfig only if same CB **and** same dtype as previous tile.
- All `_with_dt` cover UNPACK + MATH format metadata for srcA (and `fast_tilize_init_with_dt` covers both srcA and srcB). Output (PACK) format is touched only by `pack_tile_with_dt`.
- `mm_init_short_with_dt`, `mm_block_init_short_with_dt`, and `reduce_with_matmul_init_with_dt` only reconfigure srcA — `in1_cb_id` is the new srcA operand for matmul.
- `sub_bcast_rows_init_short_with_dt` is hand-inlined (does not call `sub_bcast_rows_init_short`) — see FIXME comment in source.
- Reconfig under `FP32_DEST_ACC_EN`-gated wrappers is a no-op in non-fp32-accum builds — caller must verify build config when relying on the wrapper for dtype correctness.
