# CB→DFB Kernel Audit: `deepseek_prefill/offset_cumsum`

**Date:** 2026-07-15
**Op root:** `ttnn/cpp/ttnn/operations/experimental/deepseek_prefill/offset_cumsum/`

**Scope:** `device/kernels/reader_offset_cumsum_interleaved.cpp`

## Overall verdict: GREEN

**Summary:** Zero litmus hits — no `get_local_cb_interface(...)` field access, no `read_tile_value`/`get_tile_address`, no `get_pointer_to_cb_data`, no `get_cb_tiles_*_ptr`, no ptr surgery, no `fifo_*` field reads. Input/output CBs are canonical linear FIFO; `cb_id_local` is a private scratch region addressed only through bare `get_read_ptr()`/`get_write_ptr()`. Mechanical `CircularBuffer` → `DataflowBuffer` rename; safe to port on both arches.

## CB portability

| CB | Class | Kernel(s) | 1xx status | 1xx notes | 2xx status | 2xx notes |
|----|-------|-----------|------------|-----------|------------|-----------|
| `cb_id_in0`, `cb_id_out0` | 1 | `reader_offset_cumsum_interleaved.cpp` | Portable | linear FIFO → `DataflowBuffer` | Portable | — |
| `cb_id_local` | 6 | `reader_offset_cumsum_interleaved.cpp` | Portable | private scratch, bare `get_read_ptr()`/`get_write_ptr()` only — autoportable (`ScratchpadSpec` cleaner end-state) | Portable | same |

## GATE hits (must be empty to merge)

- (none)

## Blocked on runtime (2xx rollup)

- (none)
