# CB→DFB Kernel Audit: `matmul_reduce_scatter_async`

**Date:** 2026-07-15
**Op root:** `ttnn/cpp/ttnn/operations/experimental/ccl/matmul_reduce_scatter_async/`

**Scope:** No kernels of its own — pure host wrapper. `device/matmul_reduce_scatter_async_program_factory.cpp` composes two donor program factories and creates no `device/kernels/`:
- `reduce_scatter_minimal_async` — `build_ring_reduce_scatter_minimal_async_program_artifacts` (donor: `ttnn/cpp/ttnn/operations/experimental/ccl/reduce_scatter_minimal_async/`).
- `matmul` — `matmul_multi_core_reuse_mcast_2d_optimized_helper` (donor: `ttnn/cpp/ttnn/operations/matmul/`), **standard** mcast_2d path (not the `bmm_*_gathered` variant).

## Overall verdict: YELLOW (inherited)

**Summary:** This op has **no device kernels** — it delegates to the `reduce_scatter_minimal_async` and `matmul` program factories. Verdict inherits from the donors. `reduce_scatter_minimal_async` is **GREEN** (litmus-scanned: zero GATE / silent-wrong / ptr-surgery / field-read hits — canonical fabric reduce-scatter dataflow). The standard `matmul` mcast_2d donor slice is **YELLOW** (two mechanical getter swaps; see `ttnn/cpp/ttnn/operations/matmul/CB_DFB_KERNEL_AUDIT.md`). The RED `bmm_*_gathered` matmul variant is **not** invoked here. Combined rollup: **YELLOW**.

## CB portability

| CB | Class | Kernel(s) | 1xx status | 1xx notes | 2xx status | 2xx notes |
|----|-------|-----------|------------|-----------|------------|-----------|
| (all CBs) | — | delegated — no own kernels | — | inherits `reduce_scatter_minimal_async` (GREEN) + `matmul` standard mcast_2d (YELLOW: 2 mechanical getter swaps) | — | same |

## GATE hits (must be empty to merge)

- (none of its own) — see `matmul` donor audit for the two mechanical `fifo_page_size` / `fifo_num_pages` field-read swaps (getters exist).

## Blocked on runtime (2xx rollup)

- (none) — donors have no missing-getter dependency; matmul fixes are mechanical.

## Recommended path

No kernel work in this op. Land the two mechanical getter swaps in the `matmul` standard mcast donor; `reduce_scatter_minimal_async` ports mechanically. This wrapper then inherits GREEN.
