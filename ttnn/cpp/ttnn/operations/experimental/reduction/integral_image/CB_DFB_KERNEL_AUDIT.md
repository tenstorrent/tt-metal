# CB→DFB Kernel Audit: `integral_image`

**Date:** 2026-07-15
**Op root:** `ttnn/cpp/ttnn/operations/experimental/reduction/integral_image/`

**Scope:** `intimg_program_factory` → kernels: `device/kernels/intimg_reader.cpp`, `device/kernels/intimg_compute.cpp`, `device/kernels/intimg_writer.cpp`. Include closure: `device/kernels/common.hpp`, `device/kernels/common_dataflow.hpp` (op-local shared headers).

## Overall verdict: GREEN

**Summary:** Cumulative-sum (integral image) pipeline: reader → cumsum compute (2-axis staged) → writer. Litmus scans over all kernels + shared headers find **zero** `get_local_cb_interface`/`cb_interface.` access, **zero** `get_cb_tiles_*_ptr`, **zero** `read_tile_value`/`get_tile_address`, **zero** `get_pointer_to_cb_data`, and **zero** `fifo_*` surgery or field reads. All CBs are canonical Class 1 linear FIFO. The `ReadCBGuard`/`WriteCBGuard` RAII helpers in `common.hpp` wrap the standard `wait_front`/`pop_front` and `reserve_back`/`push_back` calls only — no field access. Mechanical `CircularBuffer` → `DataflowBuffer` rename only.

## Scope notes

- CB names such as `cb_read_guard`/`cb_write_guard` are local `ReadCBGuard`/`WriteCBGuard` RAII objects (canonical sync), not distinct buffers.

## CB portability

CBs collapsed by role.

| CB | Class | Kernel(s) | 1xx status | 1xx notes | 2xx status | 2xx notes |
|----|-------|-----------|------------|-----------|------------|-----------|
| `cb_input` | 1 | `intimg_reader.cpp`, `intimg_compute.cpp` | Portable | input tiles, linear FIFO → `DataflowBuffer` | Portable | — |
| `cb_cumsum_stage_*`, `cb_cumsum_output`, `cb_acc` | 1 | `intimg_compute.cpp` | Portable | staged cumsum intermediates, canonical `reserve/push` ↔ `wait/pop` | Portable | — |
| `cb_axis_2_buffer`, `cb_axis_3_buffer_read`, `cb_axis_3_buffer_write` | 1 | `intimg_compute.cpp`, `common_dataflow.hpp` | Portable | per-axis intermediate FIFOs | Portable | — |
| `cb_output` | 1 | `intimg_compute.cpp`, `intimg_writer.cpp` | Portable | pack → output, `get_read_ptr()`/`get_write_ptr()` as L1/NoC addr only | Portable | — |

## GATE hits (must be empty to merge)

- (none)

## Blocked on runtime (2xx rollup)

- (none)
