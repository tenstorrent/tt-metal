# CB→DFB Kernel Audit: `deepseek_prefill/masked_bincount`

**Date:** 2026-07-15
**Op root:** `ttnn/cpp/ttnn/operations/experimental/deepseek_prefill/masked_bincount/`

**Scope:** `device/kernels/reader_masked_bincount.cpp`

## Overall verdict: GREEN

**Summary:** Zero litmus hits — no `get_local_cb_interface(...)` field access, no `read_tile_value`/`get_tile_address`, no `get_pointer_to_cb_data`, no `get_cb_tiles_*_ptr`, no ptr surgery, no `fifo_*` field reads. The kernel uses linear-FIFO input/output CBs plus scratch/gather CBs addressed only through bare `get_read_ptr()`/`get_write_ptr()`, with semaphore-based cross-core coordination. Mechanical `CircularBuffer` → `DataflowBuffer` rename; safe to port on both arches.

## CB portability

| CB | Class | Kernel(s) | 1xx status | 1xx notes | 2xx status | 2xx notes |
|----|-------|-----------|------------|-----------|------------|-----------|
| `cb_id_in`, `cb_id_out` | 1 | `reader_masked_bincount.cpp` | Portable | linear FIFO → `DataflowBuffer` | Portable | — |
| `cb_gather_tmp`, `cb_mask` | 6 | `reader_masked_bincount.cpp` | Portable | scratch/gather region, bare `get_read_ptr()`/`get_write_ptr()` + semaphores — autoportable (`ScratchpadSpec` + sems cleaner end-state) | Portable | same |

## GATE hits (must be empty to merge)

- (none)

## Blocked on runtime (2xx rollup)

- (none)
