# CB→DFB Kernel Audit: `deepseek_prefill/per_token_cast_back`

**Date:** 2026-07-15
**Op root:** `ttnn/cpp/ttnn/operations/experimental/deepseek_prefill/per_token_cast_back/`

**Scope:** `device/kernels/dataflow/{reader_per_token_cast_back.cpp, writer_per_token_cast_back.cpp}`, `device/kernels/compute/compute_per_token_cast_back.cpp`

## Overall verdict: GREEN

**Summary:** Zero litmus hits across all three kernels — no `get_local_cb_interface(...)` field access, no `read_tile_value`/`get_tile_address`, no `get_pointer_to_cb_data`, no `get_cb_tiles_*_ptr`, no ptr surgery, no `fifo_*` field reads. The reader→compute→writer pipeline is canonical linear FIFO; `cb_scale_scratch` is a per-row scratch region addressed only through bare `get_read_ptr()`/`get_write_ptr()`. Mechanical `CircularBuffer` → `DataflowBuffer` rename; safe to port on both arches.

## CB portability

| CB | Class | Kernel(s) | 1xx status | 1xx notes | 2xx status | 2xx notes |
|----|-------|-----------|------------|-----------|------------|-----------|
| `cb_input_e4m3`, `cb_scale_bcast` | 1 | `reader_*.cpp`, `compute_*.cpp` | Portable | linear FIFO inputs → `DataflowBuffer` | Portable | — |
| `cb_in_rm`, `cb_in_tile`, `cb_out_tile` | 1 | `compute_per_token_cast_back.cpp` | Portable | intra-compute linear FIFO stages | Portable | — |
| `cb_out_fp32` | 1 | `compute_*.cpp`, `writer_*.cpp` | Portable | pack → output, linear FIFO / bare L1 addr | Portable | — |
| `cb_scale_scratch` | 6 | `reader_per_token_cast_back.cpp` | Portable | per-row scratch, bare `get_read_ptr()`/`get_write_ptr()` only — autoportable (`ScratchpadSpec` cleaner end-state) | Portable | same |

## GATE hits (must be empty to merge)

- (none)

## Blocked on runtime (2xx rollup)

- (none)
