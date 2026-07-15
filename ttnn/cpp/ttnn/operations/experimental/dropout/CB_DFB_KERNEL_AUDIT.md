# CB→DFB Kernel Audit: `dropout`

**Date:** 2026-07-15
**Op root:** `ttnn/cpp/ttnn/operations/experimental/dropout/`

**Scope:** `dropout_program_factory` → kernels: `device/kernels/dataflow/reader_dropout_interleaved_start_id.cpp`, `device/kernels/compute/dropout_kernel.cpp`, `device/kernels/dataflow/writer_dropout_interleaved_start_id.cpp`.

## Overall verdict: GREEN

**Summary:** Elementwise dropout (reader → dropout compute → writer). Litmus scans find **zero** `get_local_cb_interface`/`cb_interface.` access, **zero** `get_cb_tiles_*_ptr`, **zero** `read_tile_value`/`get_tile_address`, **zero** `get_pointer_to_cb_data`, and **zero** `fifo_*` surgery or field reads. All CBs are canonical Class 1 linear FIFO; RNG seed coordination uses `CoreLocalMem` (sanctioned). Mechanical `CircularBuffer` → `DataflowBuffer` rename only.

## CB portability

CBs collapsed by role (`cb_id_*` aliases are the CT-arg id of the same buffer).

| CB | Class | Kernel(s) | 1xx status | 1xx notes | 2xx status | 2xx notes |
|----|-------|-----------|------------|-----------|------------|-----------|
| `cb_in0` | 1 | `reader_dropout_interleaved_start_id.cpp`, `dropout_kernel.cpp` | Portable | input tiles, linear FIFO → `DataflowBuffer` | Portable | — |
| `cb_out` | 1 | `dropout_kernel.cpp`, `writer_dropout_interleaved_start_id.cpp` | Portable | pack → output, `get_read_ptr()`/`get_write_ptr()` as L1/NoC addr only | Portable | — |

## GATE hits (must be empty to merge)

- (none)

## Blocked on runtime (2xx rollup)

- (none)
