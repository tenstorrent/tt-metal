# CB→DFB Kernel Audit: `experimental/quasar/move`

**Date:** 2026-07-15
**Op root:** `ttnn/cpp/ttnn/operations/experimental/quasar/move/`

**Scope:** All device kernels under `device/kernels/`: dataflow (`move_interleaved_with_overlap.cpp`, `move_interleaved_with_overlap_writer.cpp`, `move_stick_layout_interleaved_with_overlap.cpp`, `move_stick_layout_interleaved_with_overlap_writer.cpp`, `reader_unary_local_l1_copy_backwards.cpp`).

## Overall verdict: GREEN

**Summary:** Clean. All Step-4 litmus scans return **zero** hits — no GATE field access, no silent-wrong pointers, no `read_tile_value`/`get_tile_address`, no `get_pointer_to_cb_data`, no `fifo_*` ptr surgery or field reads. The move kernels stream through canonical linear FIFOs (including the overlap-aware and backwards local-L1 copy paths); `get_read_ptr()`/`get_write_ptr()` are used only as L1/NoC byte addresses. Mechanical `CircularBuffer` → `DataflowBuffer` rename on both arches.

## CB portability

| CB | Class | Kernel(s) | 1xx status | 1xx notes | 2xx status | 2xx notes |
|----|-------|-----------|------------|-----------|------------|-----------|
| `cb_in` / `cb_out` (move stream) | 1 | `move_interleaved_with_overlap*.cpp`, `move_stick_layout_interleaved_with_overlap*.cpp`, `reader_unary_local_l1_copy_backwards.cpp` | Portable | overlap-aware / backwards local-L1 copy; linear FIFO, `get_read_ptr()`/`get_write_ptr()` as addr only | Portable | — |

## GATE hits (must be empty to merge)

- (none)

## Blocked on runtime (2xx rollup)

- (none)
