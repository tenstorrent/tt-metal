# CB→DFB Kernel Audit: `llama_reduce_scatter_matmul`

**Date:** 2026-07-15
**Op root:** `ttnn/cpp/ttnn/operations/experimental/ccl/llama_reduce_scatter_matmul/`

**Scope:** No kernels of its own — pure host wrapper. `device/rs_matmul_program_factory.cpp` composes two donor program factories and creates no `device/kernels/`:
- `llama_reduce_scatter` — `LlamaReduceScatterDeviceOperation::LlamaReduceScatterAdd::create_at_program_processing` (donor: `ttnn/cpp/ttnn/operations/experimental/ccl/llama_reduce_scatter/`).
- `matmul` — `matmul_multi_core_reuse_mcast_1d_optimized_helper` (donor: `ttnn/cpp/ttnn/operations/matmul/`), **standard** mcast_1d path (not the `bmm_*_gathered` variant).

## Overall verdict: YELLOW (inherited)

**Summary:** This op has **no device kernels** — it delegates to the `llama_reduce_scatter` and `matmul` program factories. Verdict inherits from the donors. `llama_reduce_scatter` is **GREEN** (canonical fabric reduce-scatter dataflow, zero litmus hits — see its audit). The standard `matmul` mcast_1d donor slice is **YELLOW** (two mechanical getter swaps; see `ttnn/cpp/ttnn/operations/matmul/CB_DFB_KERNEL_AUDIT.md`). The RED `bmm_*_gathered` matmul variant is **not** invoked here. Combined rollup: **YELLOW**.

## CB portability

| CB | Class | Kernel(s) | 1xx status | 1xx notes | 2xx status | 2xx notes |
|----|-------|-----------|------------|-----------|------------|-----------|
| (all CBs) | — | delegated — no own kernels | — | inherits `llama_reduce_scatter` (GREEN) + `matmul` standard mcast_1d (YELLOW: 2 mechanical getter swaps) | — | same |

## GATE hits (must be empty to merge)

- (none of its own) — see `matmul` donor audit for the two mechanical `fifo_page_size` / `fifo_num_pages` field-read swaps (getters exist).

## Blocked on runtime (2xx rollup)

- (none) — donors have no missing-getter dependency; matmul fixes are mechanical.

## Recommended path

No kernel work in this op. Land the two mechanical getter swaps in the `matmul` standard mcast donor; `llama_reduce_scatter` ports mechanically. This wrapper then inherits GREEN.
