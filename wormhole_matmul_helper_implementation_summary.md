# Wormhole Matmul Helper Implementation Summary

## What Was Built

**New files:**
- `ttnn/cpp/ttnn/kernel_lib/matmul_helpers_compute.hpp` — Declarations following the helper library conventions (free functions in `compute_kernel_lib` namespace, `MatmulMode::TILE`/`BLOCK` template parameter, `MatmulConfig` struct with factory methods, `.hpp` + `.inl` split)
- `ttnn/cpp/ttnn/kernel_lib/matmul_helpers_compute.inl` — All implementations

**API levels:**
- **Low-level:** `matmul_init`, `matmul_init_short`, `matmul_init_short_with_dt`, `matmul_init_short_with_both_dt`, `matmul_tile`
- **Mid-level:** `matmul_accumulate`, `matmul_accumulate_subblock`, `matmul_acquire_dst`, `matmul_pack_output`, `matmul_pack_partials`, `matmul_reload_partials`, `matmul_accumulate_and_pack`
- **High-level:** `matmul` (full auto loop), `matmul_inner_block`, `matmul_compute_tile`
- **Specialized:** `matmul_reduce_w`, `matmul_attention`, `matmul_reduce_subblock_inplace`, `matmul_moe_with_bias`, `matmul_moe_w2_dm1_cycling`, `matmul_moe_w2_dm1_linear`

## Files Migrated (27+ kernel files)

All production compute kernel files that included `api/compute/matmul_op.h` were migrated:

- `bmm.cpp`, `bmm_large_block_zm.cpp`, `bmm_large_block_zm_fused_bias_activation.cpp` (x2 copies), `bmm_large_block_zm_fused_bias_activation_gathered.cpp` (x2 copies)
- `reduce_w.cpp`, `moreh_matmul.cpp`, `moreh_mean_w.cpp`, `moreh_sum_w.cpp`
- `transformer_attn_matmul.cpp`, `transformer_group_attn_matmul.cpp`
- `conv3d/compute.cpp`, `minimal_matmul/compute.cpp`
- `moe_compute/compute.cpp`, `moe_gpt/compute.cpp`, `moe_gate_mm/compute.cpp`
- `matmul_wo/compute.cpp`, `topk_router_gpt/compute.cpp`
- `compute_common.hpp` (SDPA), `compute_streaming.hpp` (SDPA)
- `bmm_tilize_untilize.cpp`, `conv_bmm_tilize.cpp`
- `all_gather_minimal_matmul_async/compute.cpp`, `llama_all_gather_matmul_async/compute/bmm...gathered.cpp`
- `rope.hpp` (DeepSeek v3 b1)
- `sdpa_compute_utils.hpp`, `sdpa_fw_compute_kernel.cpp`, `sdpa_bw_compute_utils.hpp` (tt-train)

## Test Results (Wormhole n150, all 0 failures)

| Suite | Passed | Time |
|---|---|---|
| test_matmul.py | 589 | 11m46s |
| test_sum.py | 383 | 3m04s |
| test_moreh_matmul.py | 89 | 4m36s |
| test_moreh_mean.py | 76 | 5m12s |
| test_moreh_sum.py | 227 | 1m44s |
| test_reduction_mean.py | 175 | 59s |
| test_sdpa_decode.py | 9 | 5s each |
| test_sdpa_prefill.py | 3 | 8.91s |
| **Total** | **1,551** | |

## Bug Found and Fixed

**`matmul_reduce_subblock_inplace` was hardcoded to `MatmulMode::TILE`** — the original `MatmulOp::reduce_subblock_inplace` was a member of the templated class, so it inherited whatever mode the class was instantiated with. The SDPA uses it with block-mode configuration (`mm_block_init_short`), but the initial non-templated free function called `matmul_tiles()` instead of `matmul_block()`. This mismatch caused the hardware to deadlock — all 64 cores hung with TRISC2 (packer) stuck in kernel code, BRISC/NCRISC blocked on CBs.

**Diagnosis method:** Watcher logs showed all worker cores with BRISC in `CWFW` (CB write front wait), NCRISC in `CRBW` (CB read back wait), and TRISC2 in `K` (in kernel). This pattern indicated a compute pipeline deadlock. Bisecting by reverting individual files isolated the bug to `compute_common.hpp`'s `matmul_reduce` function, which called `matmul_reduce_subblock_inplace`. Code inspection revealed the hardcoded TILE mode in the implementation.

**Fix:** Made `matmul_reduce_subblock_inplace` templated on `MatmulMode`. Call sites pass `<MatmulMode::BLOCK>`. The declaration in `.hpp` and implementation in `.inl` were both updated. After fix, SDPA prefill tests complete in 8.91s (previously hung indefinitely).

**Implication for other functions:** `matmul_compute_tile`, `matmul_reduce_w`, and `matmul_attention` also hardcode `MatmulMode::TILE` internally. This is correct for these three because they are inherently tile-mode operations (the original class only used them from `TileMatmulOp` instances). But any future function that could be used in either mode must be templated.

## Build Notes

- Compute kernels are JIT-compiled at runtime, so `./build_metal.sh` doesn't need to be re-run for kernel header changes
- The JIT cache at `generated/kernels/` must be cleared (`rm -rf generated/kernels/`) after modifying kernel headers to force recompilation
- `ARCH_NAME` env var on the test system was incorrectly set to `blackhole` despite being Wormhole n150 hardware — all tests were run with `ARCH_NAME=wormhole_b0`
- Two initial compilation errors were fixed:
  1. Missing `common_types.hpp` include (doesn't exist on this branch) — removed since matmul helpers don't use `NoAccumulation`/`NoOp`
  2. Undefined `ALWI` macro — fixed by including `api/compute/matmul.h` in the `.hpp` file, matching how `tilize_helpers.hpp` includes `api/compute/tilize.h`

## Blackhole Considerations

- `#ifdef ARCH_BLACKHOLE` sections exist for: `matmul_tile_no_mop`, `matmul_accumulate_no_mop` (in both `.hpp` and `.inl`)
- The SDPA `compute_streaming.hpp` `blocked_matmul_and_pack` has a `#ifdef ARCH_BLACKHOLE` branch using `matmul_accumulate_no_mop` — migrated but untested on this Wormhole system
- `matmul_init_short_with_both_dt` (block mode only, used by `conv_bmm_tilize.cpp`) — non-templated, should work on both architectures
- The Blackhole instance should verify all `#ifdef ARCH_BLACKHOLE` paths, particularly `matmul_accumulate_no_mop` in SDPA and the `matmul_tile_no_mop` function

## Design Decisions

1. **`MatmulMode` template param** replaces `IsBlockMode` — `MatmulMode::TILE` and `MatmulMode::BLOCK` are more readable than `true`/`false`
2. **Free functions** in `compute_kernel_lib` namespace instead of a class — matches the existing helper library conventions (`reduce_helpers_compute.hpp`, `binary_op_helpers.hpp`)
3. **`MatmulConfig` struct** with factory methods (`.tile()`, `.block()`) and builder pattern (`.with_transpose()`, `.with_partials()`) — same fields as old `MatmulOpConfig`, compatible with designated initializers
4. **`MatmulBlockShape` struct** for full-auto mode parameters — encapsulates the 9 loop dimension parameters with `::of()` factory
5. **No `NoAccumulation`/`NoOp` types** — matmul spill/reload uses explicit `matmul_reload_partials()` instead of compile-time accumulation dispatch
6. **`#include "api/compute/matmul.h"` in `.hpp`** — required to bring in `ALWI` macro definition, matching how other helpers include their corresponding LLK API header
7. **`matmul_op.h` kept intact** — old header remains for `.matmul_op_project/` PoC test files; no production code uses it
8. **Specialized functions templated only when needed** — `matmul_reduce_w`, `matmul_attention`, `matmul_compute_tile` are tile-mode-only (non-templated). `matmul_reduce_subblock_inplace` is templated because SDPA uses it in block mode
9. **`MoeDm1State` moved to `compute_kernel_lib` namespace** — same struct, same fields, new namespace to match the helper library conventions
