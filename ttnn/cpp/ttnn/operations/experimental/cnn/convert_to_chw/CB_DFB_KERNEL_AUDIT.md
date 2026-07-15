# CB→DFB Kernel Audit: `convert_to_chw`

**Date:** 2026-07-15
**Op root:** `ttnn/cpp/ttnn/operations/experimental/cnn/convert_to_chw/`

**Scope:** `convert_to_chw_program_factory` → kernels: `device/kernels/reader_convert_to_chw.cpp`, `device/kernels/convert_to_chw.cpp`, `device/kernels/writer_convert_to_chw.cpp`.

## Overall verdict: GREEN

**Summary:** HWC→CHW layout conversion (reader → transpose compute → writer). Litmus scans find **zero** `get_local_cb_interface`/`cb_interface.` access, **zero** `get_cb_tiles_*_ptr`, **zero** `read_tile_value`/`get_tile_address`, **zero** `get_pointer_to_cb_data`, and **zero** `fifo_*` surgery or field reads. All CBs are canonical Class 1 linear FIFO. Mechanical `CircularBuffer` → `DataflowBuffer` rename only.

## CB portability

CBs collapsed by role (`_cb`/`_obj` aliases are the same buffer's handle / CT-arg id).

| CB | Class | Kernel(s) | 1xx status | 1xx notes | 2xx status | 2xx notes |
|----|-------|-----------|------------|-----------|------------|-----------|
| `cb_in` | 1 | `reader_convert_to_chw.cpp`, `convert_to_chw.cpp` | Portable | input tiles, linear FIFO → `DataflowBuffer` | Portable | — |
| `cb_transpose` | 1 | `convert_to_chw.cpp` | Portable | transpose intermediate, canonical `reserve/push` ↔ `wait/pop` | Portable | — |
| `cb_out` | 1 | `convert_to_chw.cpp`, `writer_convert_to_chw.cpp` | Portable | pack → output, `get_read_ptr()`/`get_write_ptr()` as L1/NoC addr only | Portable | — |

## GATE hits (must be empty to merge)

- (none)

## Blocked on runtime (2xx rollup)

- (none)
