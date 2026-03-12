# SDPA Streaming v2: Why Attribute Removal Costs 2x More on Wormhole than Blackhole

Date: 2026-03-12

## Summary

Removing `__attribute__((noinline, noclone))` from the 4 hot functions in `compute_streaming.hpp`
causes **+4.0 KB code growth on Blackhole** but **+8.3 KB on Wormhole** — despite near-identical
baseline sizes (~57.5 KB vs ~58.2 KB). This document explains why.

The root cause is that Blackhole uses **custom, hand-optimized LLK functions** with direct
instruction sequences (no MOP generation, 1–2 template levels), while Wormhole uses **generic,
deeply-templated LLK functions** with full MOP programming and 4–6 template levels. When GCC
is free to inline/clone, the WH templates expand ~2x more per call site than the BH custom code.

## Measured Code Growth (q256/k256, d=128)

### Baselines are nearly identical

| Core | BH (s=2240) | WH (Mochi) | Delta |
|------|--------:|--------:|------:|
| trisc0 (UNPACK) | 17,440 | 17,752 | +312 |
| trisc1 (MATH) | 12,472 | 13,528 | +1,056 |
| trisc2 (PACK) | 16,968 | 15,692 | -1,276 |
| brisc (WRITER) | 6,076 | 6,292 | +216 |
| ncrisc (READER) | 4,496 | 4,936 | +440 |
| **Grand Total** | **57,452** | **58,200** | **+748** |

### Attribute removal: per-core code growth

| Core | BH delta | WH delta | WH/BH ratio |
|------|--------:|--------:|------:|
| trisc0 (UNPACK) | +828 | +1,800 | 2.2x |
| trisc1 (MATH) | **-428** | **+1,580** | — |
| trisc2 (PACK) | +3,620 | +4,916 | 1.4x |
| **Compute Total** | **+4,020** | **+8,296** | **2.1x** |

### Grand total with headroom

| Variant | BH | WH (Mochi) | WH headroom (70,656 limit) |
|---------|------:|------:|------:|
| Baseline | 57,452 | 58,200 | 12,456 |
| No attrs | 61,472 | 66,496 | 4,160 |
| No attrs + no unroll | 63,424 | 69,152 | **-192 (OVERFLOW)** |

## Architecture-Specific LLK Implementations

The `#ifdef ARCH_BLACKHOLE` blocks in `compute_streaming.hpp` select different LLK
implementations for the two most impactful functions. These implementations have fundamentally
different code generation characteristics.

### `blocked_matmul_and_pack` (5 call sites)

**Blackhole** — `matmul_block_no_mop()`:
- Defined in `api/compute/experimental/matmul_custom.h`
- Calls `llk_math_matmul_no_mop<MATH_FIDELITY, MM_THROTTLE>()`
- Implementation: pre-loads instruction sequences into a **replay buffer** at init time;
  the inner loop is a thin `lltt::replay()` call
- Template depth: 1–2 levels
- No MOP (machine operation) generation at call time
- Source: `tt_llk_blackhole/llk_lib/experimental/llk_math_matmul_custom_no_mop.h` (571 lines)

**Wormhole** — `matmul_block()`:
- Defined in `api/compute/matmul.h`
- Calls `llk_math_matmul<MATH_FIDELITY, MM_THROTTLE>()`
- Implementation: creates `ckernel_template` objects, programs MOPs with nested loops,
  configures address modifiers at each call
- Template depth: 4–6 levels (init → configure_mop → configure_addrmod → ...)
- Full MOP init/run cycle per call
- Source: `tt_llk_wormhole_b0/llk_lib/llk_math_matmul.h` (874 lines — 53% larger)

**Impact**: WH's `ckernel_template` constructor + method calls + address modifier configuration
expand significantly when inlined across 5 call sites. BH's replay-buffer approach inlines to
almost nothing.

### `sub_exp_block_bcast_cols` (5 call sites, dominant contributor)

This function accounts for ~92% of code growth on BH and is the largest single contributor on
both architectures.

**Blackhole** — custom sub + exp:
- `sub_bcast_cols_init_short_custom()` → 6 direct `TTI_*` instructions for unpack, minimal
  address mod config for math
- `sub_tiles_bcast_cols_custom()` → ~8 explicit `TTI_ELWSUB` instructions inline, custom
  unpack function with direct `TTI_UNPACR`/`TTI_INCADCZW` sequences
- No MOP generation, no template dispatch
- Sources:
  - `tt_llk_blackhole/llk_lib/experimental/llk_math_eltwise_binary_custom.h` (115 lines)
  - `tt_llk_blackhole/llk_lib/experimental/llk_unpack_AB_sub_bcast_col_custom.h` (77 lines)
  - `hw/ckernels/blackhole/metal/llk_api/experimental/llk_math_eltwise_binary_custom_api.h` (35 lines)

**Wormhole** — standard sub + exp:
- `sub_bcast_cols_init_short()` → standard template functions expanding to full MOP
  initialization
- `sub_tiles_bcast_cols()` → `llk_math_eltwise_binary_init_with_operands<ELWSUB, BroadcastType::COL, LoFi>`
  → `_llk_math_eltwise_binary_init_<>` → `eltwise_binary_configure_mop_standard<>` →
  `eltwise_binary_configure_addrmod<>` (4-level template chain)
- Full MOP programming with face-row/column loops, fidelity handling, addr_mod stack operations
- Sources:
  - `tt_llk_wormhole_b0/llk_lib/llk_math_eltwise_binary.h` (400+ lines)
  - `tt_llk_wormhole_b0/llk_lib/llk_unpack_AB.h`
  - `hw/ckernels/wormhole_b0/metal/llk_api/llk_math_binary_api.h`

**Impact**: Each WH call site inlines ~1.7 KB of expanded template code vs ~0.8 KB for BH's
direct instruction sequences. Over 5 call sites, this accounts for most of the 2x difference.

### `exp_packthread_tile` (shared)

The exp/pack thread tile implementation is the same SFPU template on both architectures
(identical `exp_packthread_tile<true, true, false, false, InputClamping::None, iterations>()`).
This is why trisc2 (PACK) growth, while larger on WH (+4.9 KB vs +3.6 KB), has a smaller
ratio (1.4x) than trisc0/trisc1.

### Pack path differences

The BH `blocked_pack` specializations in both `blocked_matmul_and_pack` and
`sub_exp_block_bcast_cols` use a row-skip pattern (`pack_tile<true>(dst_idx, out_cb, offset);
dst_idx += SUBBLOCK_W;`) that packs one tile per row and skips the rest. This produces less
code than WH's tile-by-tile double loop. However, this difference is modest compared to the
LLK template expansion.

## Per-Site Expansion Estimate

| Aspect | BH Custom | WH Standard |
|--------|-----------|-------------|
| Function body (lines) | ~130 (direct + API) | ~400+ (template dispatch) |
| Template depth | 1–2 levels | 4–6 levels |
| MOP programming | No (direct instructions) | Yes (full MOP init/run) |
| Estimated per-site expansion | ~800 bytes | ~1,700 bytes |
| Call sites (sub_exp) | 5 | 5 |
| **Expected growth** | **~4 KB** | **~8.5 KB** |

Measured: BH +4.0 KB, WH +8.3 KB. The estimate aligns well.

## Why trisc1 (MATH) Shrinks on BH but Grows on WH

This is the most striking per-core difference: BH trisc1 **shrinks** by 428 bytes, while WH
trisc1 **grows** by 1,580 bytes.

**BH**: The custom math functions (`llk_math_matmul_no_mop`, `llk_math_eltwise_binary_bcast_reuse_custom`)
are already thin wrappers around replay buffer calls. When GCC inlines them, it can **eliminate
call/return overhead and dead branches** — the remaining code is smaller than the function
prologue/epilogue it replaced. The replay buffer does the real work at runtime; the math
function is just a dispatch shim.

**WH**: The standard `ckernel_template` + MOP machinery is substantial. Each inlined copy brings
the full `configure_mop` → `configure_addrmod` → `ckernel_template::run()` chain. With 5 call
sites and IPA-CP creating specialized clones for different constant arguments (`TRANSPOSE`,
`IN1_STRIDE`, `SUBBLOCK_W`), the trisc1 code grows by +1.6 KB.

## Conclusion

The 2x code growth asymmetry is an inherent consequence of the LLK architecture:

- **BH custom APIs were designed for compactness**: direct instruction sequences, no MOP
  generation, shallow templates. They are naturally tolerant of inlining — the expanded code
  is small, and GCC can often optimize it further.
- **WH standard APIs were designed for generality**: deep template chains, full MOP
  programming, `ckernel_template` class machinery. They are not inlining-friendly — each
  instantiation drags hundreds of bytes of supporting infrastructure.

This makes the conditional `SDPA_NOINLINE` approach the right solution:
- **BH**: remove the attribute → +1pp math utilization, +4 KB code (ample headroom)
- **WH**: keep `noinline,noclone` → no code growth, safe for Mochi (12 KB headroom)

## Source Files

### SDPA kernel
- `ttnn/cpp/ttnn/operations/transformer/sdpa/device/kernels/compute/compute_streaming.hpp`

### Blackhole custom LLK (experimental)
- `tt_metal/hw/inc/api/compute/experimental/matmul_custom.h`
- `tt_metal/hw/inc/api/compute/experimental/sdpa_sub_custom.h`
- `tt_llk_blackhole/llk_lib/experimental/llk_math_matmul_custom_no_mop.h` (571 lines)
- `tt_llk_blackhole/llk_lib/experimental/llk_math_eltwise_binary_custom.h` (115 lines)
- `tt_llk_blackhole/llk_lib/experimental/llk_unpack_AB_sub_bcast_col_custom.h` (77 lines)
- `hw/ckernels/blackhole/metal/llk_api/experimental/llk_math_eltwise_binary_custom_api.h` (35 lines)
- `hw/ckernels/blackhole/metal/llk_api/experimental/llk_unpack_AB_sub_bcast_col_custom_api.h`

### Wormhole standard LLK
- `tt_metal/hw/inc/api/compute/matmul.h`
- `tt_metal/hw/inc/api/compute/bcast.h`
- `tt_llk_wormhole_b0/llk_lib/llk_math_matmul.h` (874 lines)
- `tt_llk_wormhole_b0/llk_lib/llk_math_eltwise_binary.h` (400+ lines)
- `tt_llk_wormhole_b0/llk_lib/llk_unpack_AB.h`
- `hw/ckernels/wormhole_b0/metal/llk_api/llk_math_binary_api.h`

### Measurement data
- `code_size_analysis_bh-qb.md` — BH quiet box: full sweep, per-function, noinline vs noclone
- `code_size_analysis_t3000.md` — T3000: SD3.5, Mochi, Wan 14B configs
- Gist: https://gist.github.com/djordjenTT/d25de4d53f79775704dd6e2587ab3fd1
