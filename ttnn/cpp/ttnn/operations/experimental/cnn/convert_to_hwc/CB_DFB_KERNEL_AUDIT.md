# CB→DFB Kernel Audit: `convert_to_hwc`

**Date:** 2026-07-15
**Op root:** `ttnn/cpp/ttnn/operations/experimental/cnn/convert_to_hwc/`

**Scope:** `convert_to_hwc_program_factory` → kernels: `device/kernels/convert_to_hwc.cpp` (compute), `device/kernels/writer_convert_to_hwc.cpp`. Include closure: `kernel_lib/tilize_helpers.hpp`.

## Overall verdict: GREEN

**Summary:** CHW→HWC layout conversion (transpose compute + writer; reader folded into the sharded factory). Litmus scans find **zero** `get_local_cb_interface`/`cb_interface.` access, **zero** `get_cb_tiles_*_ptr`, **zero** `read_tile_value`/`get_tile_address`, **zero** `get_pointer_to_cb_data`, and **zero** `fifo_*` surgery or field reads. All CBs are canonical Class 1 linear FIFO. The writer's `get_write_ptr() + offset` / `get_read_ptr() + src_offset_bytes` calls are bare L1/NoC byte addressing (WEIRD-OK / already portable). Mechanical `CircularBuffer` → `DataflowBuffer` rename only.

## CB portability

CBs collapsed by role (`_cb`/`_obj` aliases are the same buffer's handle / CT-arg id).

| CB | Class | Kernel(s) | 1xx status | 1xx notes | 2xx status | 2xx notes |
|----|-------|-----------|------------|-----------|------------|-----------|
| `cb_in`, `cb_in_batch`, `cb_tiled_in` | 1 | `convert_to_hwc.cpp` | Portable | input / batched / tilized inputs, linear FIFO → `DataflowBuffer` | Portable | — |
| `cb_transpose_in` (`cb_transpose_in0/1`) | 1 | `convert_to_hwc.cpp` | Portable | transpose intermediates, canonical `reserve/push` ↔ `wait/pop` | Portable | — |
| `cb_out` | 1 | `convert_to_hwc.cpp`, `writer_convert_to_hwc.cpp` | Portable (workaround) | **undesirable but OK hack:** writer uses `get_write_ptr() + offset` / `get_read_ptr() + src_offset_bytes` for HWC-layout NoC write (bare L1/NoC addressing, not `fifo_*` surgery) | Portable (workaround) | same |

## GATE hits (must be empty to merge)

- (none)

## Blocked on runtime (2xx rollup)

- (none)
