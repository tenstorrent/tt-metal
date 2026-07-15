# CB→DFB Kernel Audit: `rotate_half`

**Date:** 2026-07-15
**Op root:** `ttnn/cpp/ttnn/operations/experimental/transformer/rotate_half/`

**Scope:** `rotate_half` program factory → kernels: `device/kernels/dataflow/reader_rotate_half_interleaved_start_id.cpp`, `device/kernels/dataflow/writer_rotate_half_interleaved_start_id.cpp`. Includes are all standard API headers.

## Overall verdict: GREEN

**Summary:** Dataflow rotate-half op: reader streams the two halves of each row (`cb_in_no_mul`, `cb_in_mul`) and seeds a scalar (-1) tile; writer negates/reorders via the mul path and writes out. All CBs use canonical `reserve_back`/`get_write_ptr()`/`push_back` (reader) and `wait_front`/`get_read_ptr()`/`pop_front` (writer) on the `CircularBuffer` object API — pointers used only as L1/NoC addresses. No `get_local_cb_interface` field access, no ptr surgery, no runtime-blocked APIs. All Class 1 → mechanical `CircularBuffer` → `DataflowBuffer` rename.

## CB portability

| CB | Class | Kernel(s) | 1xx status | 1xx notes | 2xx status | 2xx notes |
|----|-------|-----------|------------|-----------|------------|-----------|
| `cb_in_no_mul` | 1 | reader/writer | Portable | first-half tiles, linear FIFO | Portable | — |
| `cb_in_mul` | 1 | reader/writer | Portable | second-half tiles (multiplied), linear FIFO | Portable | — |
| `cb_in_scalar` | 1 | reader | Portable | scalar (-1) tile; reader seeds via `get_write_ptr()` L1 addr then `push_back` | Portable | — |
| `cb_out_no_mul` / `cb_out_mul` | 1 | writer | Portable | output halves, pack/`get_read_ptr()` as NoC addr | Portable | — |

## GATE hits (must be empty to merge)

- (none)

## Blocked on runtime (2xx rollup)

- (none)
