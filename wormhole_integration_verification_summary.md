# Wormhole Integration Verification Summary

**Branch:** `wransom/matmulop_integ_impl` | **Verified on:** Wormhole B0 n150 | **Date:** 2026-04-10

---

## Overview

This document records the Wormhole verification of the integrated matmul helper library
on the `wransom/matmulop_integ_impl` branch. The integrated branch was produced by a
previous Blackhole instance that merged the WH and BH implementation branches, using the
BH branch's code (cleaner API with factory methods, convenience aliases, `static_assert`
guards) while preserving all WH bug fixes. That instance verified the merged code on
Blackhole P100A but could not verify it on Wormhole hardware.

This verification confirms that the integrated helper library works correctly on Wormhole,
including all `#else` (non-BLACKHOLE) code paths in architecture-guarded sections.

---

## Environment

- **Hardware:** Wormhole B0 n150 (single chip, 8x8 = 64 worker cores)
- **Confirmed via:** `ttnn.open_device()` reports `Arch.WORMHOLE_B0`
- **ARCH_NAME note:** The system had `ARCH_NAME=blackhole` set incorrectly in the
  environment. All tests were run with explicit `ARCH_NAME=wormhole_b0` override.
- **JIT cache:** Cleared (`rm -rf generated/kernels/`) before testing to force fresh
  kernel compilation against the new helper headers.

---

## Test Results

| Suite | Passed | Skipped | Failed | Time |
|-------|--------|---------|--------|------|
| test_matmul.py (core unit) | 589 | 104 | **0** | 8m49s |
| SDPA decode | 9 | 0 | **0** | ~10s each |
| SDPA prefill | 3 | 2 | **0** | ~9s |
| moreh_matmul | 89 | 84 | **0** | 1m31s |
| moreh_mean + moreh_sum | 303 | 227 | **0** | 51s |
| reduction sum + mean | 558 | 1 | **0** | 14s |
| conv2d | 161 | 48 | **0** | 3m32s |
| linear + addmm + sparse + experimental + batch_mismatch | 494 | 34 | **0** | 5m45s |
| **TOTAL** | **2,206** | **500** | **0** | **~21m** |

All skips are pre-existing (WH-specific limitations, `bfloat8_b` exclusions, grid size
requirements, profiling/OOM guards). No test was skipped due to our changes.

### Comparison with Previous Instance Results

| Suite | This run (WH integ) | Previous WH instance | Previous BH instance |
|-------|---------------------|---------------------|---------------------|
| test_matmul.py | 589 | 589 | 557 |
| SDPA decode | 9 | 9 | 9 |
| SDPA prefill | 3 | 3 | 3 |
| moreh_matmul | 89 | 89 | 89 |
| moreh_mean + moreh_sum | 303 | 303 | 303 |
| reduction sum + mean | 558 | 558 | — |
| conv2d | 161 | — | 161 |
| linear + addmm + sparse + experimental + batch_mismatch | 494 | — | 493 |
| **Total passed** | **2,206** | **1,551** | **2,762** |

Pass counts match or exceed previous per-architecture results. Differences in totals
reflect different test scope (BH instance ran nightly matmul suites; this run included
linear/addmm/sparse/experimental which were not in the WH instance's scope).

---

## Migrated Kernels Exercised During Testing

The following migrated compute kernels were confirmed compiling and executing correctly
on Wormhole via the test suites:

| Kernel File | Test Suite | Helper Functions Exercised |
|-------------|-----------|---------------------------|
| `bmm_large_block_zm_fused_bias_activation.cpp` | test_matmul.py | `matmul_init<BLOCK>`, `matmul_accumulate<BLOCK>`, `matmul_reload_partials<BLOCK>`, `matmul_init_short_with_dt<BLOCK>` |
| `compute_streaming.hpp` (SDPA) | test_sdpa_decode.py | `matmul_accumulate<BLOCK>` (WH `#else` path), `matmul_tile<BLOCK>` |
| `compute_common.hpp` (SDPA) | test_sdpa_prefill.py | `matmul_reduce_subblock_inplace<BLOCK>`, `matmul_init_short<BLOCK>` |
| `moreh_matmul.cpp` | test_moreh_matmul.py | `matmul_compute_one_tile<TILE>`, `matmul_init<TILE>` |
| `moreh_mean_w.cpp` | test_moreh_mean.py | `matmul_reduce_w_with_init<TILE>` |
| `moreh_sum_w.cpp` | test_moreh_sum.py | `matmul_reduce_w_with_init<TILE>` |
| `reduce_w.cpp` | test_sum.py | `matmul_reduce_w<TILE>`, `matmul_init<TILE>` |
| `conv_bmm_tilize.cpp` | test_conv2d.py | `matmul_init_short_with_both_dt<BLOCK>`, `matmul_accumulate<BLOCK>`, `matmul_accumulate_and_pack<BLOCK>` |
| `bmm.cpp` | test_matmul.py | `matmul<TILE>` (full automated), `matmul_init<TILE>` |
| `bmm_large_block_zm.cpp` | test_matmul.py | `matmul<BLOCK>` (full automated), `matmul_init<BLOCK>` |

---

## Critical Bug Fix Verified

### `matmul_reduce_subblock_inplace` Deadlock Fix

**Background:** The initial implementation hardcoded `MatmulMode::TILE` in
`matmul_reduce_subblock_inplace`. The original `MatmulOp::reduce_subblock_inplace` was a
member of the templated class, inheriting whatever mode the class was instantiated with.
SDPA uses it with block-mode configuration, but the free function called `matmul_tiles()`
instead of `matmul_block()`, causing all 64 cores to hang (found and fixed by the WH
instance in commit `ddc136db09`).

**Verification:** The SDPA prefill tests exercise `compute_common.hpp:1292` which calls
`matmul_reduce_subblock_inplace<BLOCK>`. All 3 SDPA prefill tests passed in ~9 seconds
each on this Wormhole system. No deadlocks observed.

The fix (making the function templated on `MatmulMode`) is present in the integrated
branch at `matmul_helpers_compute.inl:314-328`.

---

## Architecture-Specific Code Paths Verified

### WH `#else` Paths in `compute_streaming.hpp`

The SDPA streaming kernel has extensive `#ifdef ARCH_BLACKHOLE` / `#else` guards. On
Wormhole, the `#else` paths are taken:

| Code Path | BH (ARCH_BLACKHOLE) | WH (#else) — verified here |
|-----------|--------------------|-----------------------------|
| Blocked matmul | `matmul_accumulate_no_mop<BLOCK>` | `matmul_accumulate<BLOCK>` |
| QKT init | `mm_no_mop_init_short` | `mm_block_init_short` |
| QKT reinit | `mm_no_mop_reinit_short` | `mm_block_init_short` |
| Pack MOP config | `llk_pack_mop_config` calls | Not present (standard pack) |
| Reciprocal | `recip_tile<false>` | `recip_tile_init` + `recip_tile(0)` |

All WH paths executed successfully through the SDPA decode and prefill test suites.

### `#ifndef ALWI` Guard

The `.hpp` header uses `#ifndef ALWI` / `#define ALWI` to ensure the macro is available
when the header is the first include in a kernel file (before `common_globals.h`). This
was verified to work on Wormhole — all kernel files compiled successfully through JIT.

---

## Testing Notes

### Parallel Execution and Device Contention

Initial test runs attempted to execute matmul, SDPA decode, and SDPA prefill in parallel.
Since the Wormhole device is exclusive (only one process can hold it at a time), the SDPA
tests waited for the matmul suite to finish. Two tests showed "ERROR" status in pytest
(not "FAILED") due to device contention during parallel startup:

- `test_sdpa_decode[b=8-nh=8-nkv=1-s=32768-d=128-grid_size=(8, 6)-...]` — **PASSED on
  isolated rerun** (5.26s)
- `test_sdpa_tt[b=1-nh=8-nkv=1-s=2048-d=128-k128-q128-dram_interleaved-bfp8]` — **SKIPPED
  on isolated rerun** ("Can cause OOM if profiling is enabled" — DPRINT was active)

Neither error was related to our changes.

### DPRINT Active

The test environment had DPRINT enabled on device 0, worker core (x=0,y=0). This caused
one SDPA prefill test to skip with "Can cause OOM if profiling is enabled." This is a
pre-existing skip condition, not related to the helper library migration.

---

## What Remains

1. **MoE/CCL kernels** — `moe_compute/compute.cpp`, `moe_gpt/compute.cpp`,
   `topk_router_gpt/compute.cpp`, `all_gather_minimal_matmul_async/compute.cpp` were
   migrated but require multi-device setups (T3000/Galaxy) for device-level validation.
   Compilation succeeded and the helper calls are structurally identical to verified
   kernels.

2. **tt-train kernels** — `sdpa_fw_compute_kernel.cpp`, `sdpa_compute_utils.hpp`, and
   `sdpa_bw_compute_utils.hpp` were migrated but tt-train tests were not run (separate
   test infrastructure).

3. **Cleanup pass** — The next instance should:
   - Remove the old `matmul_op.h` header if no kernel includes it
   - Clean up `.matmul_op_project/` directory (PoC migration artifacts)
   - Remove the summary `.md` files from the repo root
   - Verify the branch is clean for PR submission
