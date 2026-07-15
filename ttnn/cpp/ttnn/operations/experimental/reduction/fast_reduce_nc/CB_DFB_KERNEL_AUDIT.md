# CB→DFB Kernel Audit: `fast_reduce_nc`

**Date:** 2026-07-15
**Op root:** `ttnn/cpp/ttnn/operations/experimental/reduction/fast_reduce_nc/`

**Scope:** `fast_reduce_nc_program_factory` → kernels: `device/kernels/reader_reduce_nc.cpp`, `device/kernels/reduce_nc.cpp`, `device/kernels/writer_reduce_nc.cpp`. Include closure: `kernel_lib/l1_helpers.hpp`.

## Overall verdict: GREEN

**Summary:** Plain NC reduction (reader → reduce compute → writer). Litmus scans find **zero** `get_local_cb_interface`/`cb_interface.` access, **zero** `get_cb_tiles_*_ptr`, **zero** `read_tile_value`/`get_tile_address`, **zero** `get_pointer_to_cb_data`, and **zero** `fifo_*` surgery or field reads. All CBs are canonical Class 1 linear FIFO. Mechanical `CircularBuffer` → `DataflowBuffer` rename only.

## CB portability

| CB | Class | Kernel(s) | 1xx status | 1xx notes | 2xx status | 2xx notes |
|----|-------|-----------|------------|-----------|------------|-----------|
| `cb_in0` | 1 | `reader_reduce_nc.cpp`, `reduce_nc.cpp` | Portable | input tiles, linear FIFO → `DataflowBuffer` | Portable | — |
| `cb_in1` (reduce scalar) | 1 | `reader_reduce_nc.cpp`, `reduce_nc.cpp` | Portable | scalar operand, linear FIFO | Portable | — |
| `cb_out0` | 1 | `reduce_nc.cpp`, `writer_reduce_nc.cpp` | Portable | pack → output, `get_read_ptr()`/`get_write_ptr()` as L1/NoC addr only | Portable | — |

## GATE hits (must be empty to merge)

- (none)

## Blocked on runtime (2xx rollup)

- (none)
