# CB→DFB Kernel Audit: `deepseek_prefill/insert`

**Date:** 2026-07-15
**Op root:** `ttnn/cpp/ttnn/operations/experimental/deepseek_prefill/insert/`

**Scope:** `device/kernels/dataflow/{reader_insert.cpp, writer_insert.cpp}`

## Overall verdict: GREEN

**Summary:** Zero litmus hits across both kernels — no `get_local_cb_interface(...)` field access, no `read_tile_value`/`get_tile_address`, no `get_pointer_to_cb_data`, no `get_cb_tiles_*_ptr`, no ptr surgery, no `fifo_*` field reads. `cb_tile` is a canonical linear FIFO; the `*_scratch` CBs are private scratch addressed only through bare `get_read_ptr()`/`get_write_ptr()`. Mechanical `CircularBuffer` → `DataflowBuffer` rename; safe to port on both arches.

## CB portability

| CB | Class | Kernel(s) | 1xx status | 1xx notes | 2xx status | 2xx notes |
|----|-------|-----------|------------|-----------|------------|-----------|
| `cb_tile` | 1 | `reader_insert.cpp`, `writer_insert.cpp` | Portable | linear FIFO → `DataflowBuffer` | Portable | — |
| `cb_start_scratch`, `cb_counts_scratch`, `cb_global_expert_idx_scratch` | 6 | `reader_insert.cpp`, `writer_insert.cpp` | Portable | private scratch, bare `get_read_ptr()`/`get_write_ptr()` only — autoportable (`ScratchpadSpec` cleaner end-state, not port-gating) | Portable | same |

## GATE hits (must be empty to merge)

- (none)

## Blocked on runtime (2xx rollup)

- (none)
