# CB→DFB Kernel Audit: `experimental/quasar/transpose`

**Date:** 2026-07-15
**Op root:** `ttnn/cpp/ttnn/operations/experimental/quasar/transpose/`

**Scope:** All device kernels under `device/kernels/`: compute (`transpose_wh.cpp`, `transpose_wh_rm.cpp`, `transpose_wh_rm_sharded.cpp`, `transpose_wh_sharded.cpp`) and the dataflow reader/writer family for WH / HC / CN transpose across interleaved, sharded, and row-major layouts (18 files).

## Overall verdict: GREEN

**Summary:** Clean. All Step-4 litmus scans return **zero** hits across all compute and dataflow kernels — no GATE field access, no silent-wrong pointers, no `read_tile_value`/`get_tile_address`, no `get_pointer_to_cb_data`, no `fifo_*` ptr surgery or field reads. Transpose readers restage tiles/sticks into canonical linear FIFOs, the WH/HC/CN compute stages consume and pack, and writers drain; `get_read_ptr()`/`get_write_ptr()` are used only as L1/NoC byte addresses for the layout-reordering NoC transfers. Mechanical `CircularBuffer` → `DataflowBuffer` rename on both arches.

## CB portability

| CB | Class | Kernel(s) | 1xx status | 1xx notes | 2xx status | 2xx notes |
|----|-------|-----------|------------|-----------|------------|-----------|
| `cb_in` (restaged input) | 1 | `reader_unary_transpose_*`, `reader_unary_sharded.cpp` | Portable | tile/stick restage, linear FIFO → `DataflowBuffer`; `get_read_ptr()` as addr only | Portable | — |
| `cb_out` (transposed output) | 1 | `transpose_wh*.cpp`, `writer_unary_transpose_*`, `writer_unary_sharded.cpp`, `writer_unary_interleaved_start_id.cpp` | Portable | pack/drain → output, `get_write_ptr()` as addr only | Portable | — |

## GATE hits (must be empty to merge)

- (none)

## Blocked on runtime (2xx rollup)

- (none)
