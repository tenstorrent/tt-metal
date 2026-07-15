# CB→DFB Kernel Audit: `all_gather_matmul_async`

**Date:** 2026-07-15
**Op root:** `ttnn/cpp/ttnn/operations/experimental/ccl/all_gather_matmul_async/`

**Scope:** No kernels of its own — pure host wrapper. `device/all_gather_matmul_async_program_factory.cpp` composes two donor program factories and creates no `device/kernels/`:
- `all_gather_async` — `build_all_gather_async_minimal_default_program_artifacts` (donor: `ttnn/cpp/ttnn/operations/experimental/ccl/all_gather_async/`).
- `matmul` — `matmul_multi_core_reuse_mcast_2d_optimized_helper` / `matmul_multi_core_reuse_mcast_1d_optimized_helper` (donor: `ttnn/cpp/ttnn/operations/matmul/`), **standard** mcast path (not the `bmm_*_gathered` variant).

## Overall verdict: YELLOW (inherited)

**Summary:** This op has **no device kernels** — it delegates entirely to the `all_gather_async` and `matmul` program factories. Verdict inherits from the donors. `all_gather_async` is **GREEN** (canonical fabric dataflow, zero litmus hits — see its audit). The standard `matmul` mcast_1d/2d donor slice is **YELLOW** (two mechanical getter swaps: `writer_unary_interleaved_start_id.cpp` `fifo_page_size` → `get_entry_size()`, and the ring reader `fifo_num_pages` → `get_total_num_entries()`; see `ttnn/cpp/ttnn/operations/matmul/CB_DFB_KERNEL_AUDIT.md`). The RED `bmm_*_gathered` matmul variant is **not** invoked on this path (gather is done by the ccl half). Combined rollup: **YELLOW**.

## CB portability

| CB | Class | Kernel(s) | 1xx status | 1xx notes | 2xx status | 2xx notes |
|----|-------|-----------|------------|-----------|------------|-----------|
| (all CBs) | — | delegated — no own kernels | — | inherits `all_gather_async` (GREEN) + `matmul` standard mcast (YELLOW: 2 mechanical getter swaps) | — | same |

## GATE hits (must be empty to merge)

- (none of its own) — see `matmul` donor audit for the two mechanical `fifo_page_size` / `fifo_num_pages` field-read swaps (getters exist).

## Blocked on runtime (2xx rollup)

- (none) — donors have no missing-getter dependency; matmul fixes are mechanical.

## Recommended path

No kernel work in this op. Land the two mechanical getter swaps in the `matmul` standard mcast donor (clears its GATE → GREEN); `all_gather_async` ports mechanically. This wrapper then inherits GREEN.
