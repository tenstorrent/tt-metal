# CB→DFB Kernel Audit: `experimental/quasar/pad`

**Date:** 2026-07-15
**Op root:** `ttnn/cpp/ttnn/operations/experimental/quasar/pad/`

**Scope:** All device kernels under `device/kernels/`: dataflow readers/writers for RM interleaved (`reader/writer_pad_dims_rm_interleaved{,_sc,_v2}.cpp`), RM sharded (`reader/writer_pad_dims_rm_sharded{,_stickwise}.cpp`), tiled (`reader/writer_pad_tiled.cpp`), and unary (`reader_unary_interleaved_start_id.cpp`, `writer_unary_pad_dims_interleaved.cpp`), plus shared `common.hpp`.

## Overall verdict: GREEN

**Summary:** Clean. All Step-4 litmus scans return **zero** hits across all 15 kernel/header files — no GATE field access, no silent-wrong pointers, no `read_tile_value`/`get_tile_address`, no `get_pointer_to_cb_data`, no `fifo_*` ptr surgery or field reads. Pad readers stage input and pad values into canonical linear FIFOs and writers drain them; `get_read_ptr()`/`get_write_ptr()` are used only as L1/NoC byte addresses. Mechanical `CircularBuffer` → `DataflowBuffer` rename on both arches.

## CB portability

| CB | Class | Kernel(s) | 1xx status | 1xx notes | 2xx status | 2xx notes |
|----|-------|-----------|------------|-----------|------------|-----------|
| `cb_in` / pad-value CB | 1 | `reader_pad_*`, `reader_unary_interleaved_start_id.cpp`, `common.hpp` | Portable | input + pad-value fill, linear FIFO → `DataflowBuffer` | Portable | — |
| `cb_out` | 1 | `writer_pad_*`, `writer_unary_pad_dims_interleaved.cpp` | Portable | drain → output, `get_write_ptr()` as addr only | Portable | — |

## GATE hits (must be empty to merge)

- (none)

## Blocked on runtime (2xx rollup)

- (none)
