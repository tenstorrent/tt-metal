# CB→DFB Kernel Audit: `experimental/quasar/slice`

**Date:** 2026-07-15
**Op root:** `ttnn/cpp/ttnn/operations/experimental/quasar/slice/`

**Scope:** All device kernels under `device/kernels/`: dataflow readers/writers for multicore 4d/nd slice (`reader/writer_multicore_slice_{4d,nd}.cpp`), unpad-dims interleaved (`reader_unary_unpad_dims_interleaved_start_id{,_tensor_args}.cpp`), RM interleaved/sharded (`slice_reader_unary_unpad_dims_rm_{interleaved_start_id,sharded}.cpp`, `slice_writer_unary_stick_layout_interleaved_start_id.cpp`), strided slice (`strided_slice_reader_rm_interleaved_nd.cpp`, `strided_slice_writer_rm_interleaved.cpp`), and `writer_unary_interleaved_start_id.cpp`.

## Overall verdict: GREEN

**Summary:** Clean. All Step-4 litmus scans return **zero** hits across all 12 kernels — no GATE field access, no silent-wrong pointers, no `read_tile_value`/`get_tile_address`, no `get_pointer_to_cb_data`, no `fifo_*` ptr surgery or field reads. Slice/unpad/strided readers stream selected pages into canonical linear FIFOs and writers drain them; `get_read_ptr()`/`get_write_ptr()` are used only as L1/NoC byte addresses. Mechanical `CircularBuffer` → `DataflowBuffer` rename on both arches.

## CB portability

| CB | Class | Kernel(s) | 1xx status | 1xx notes | 2xx status | 2xx notes |
|----|-------|-----------|------------|-----------|------------|-----------|
| `cb_in` (sliced input) | 1 | `reader_*`, `slice_reader_*`, `strided_slice_reader_*` | Portable | selected-page stream, linear FIFO → `DataflowBuffer` | Portable | — |
| `cb_out` | 1 | `writer_*`, `slice_writer_*`, `strided_slice_writer_*` | Portable | drain → output, `get_write_ptr()` as addr only | Portable | — |

## GATE hits (must be empty to merge)

- (none)

## Blocked on runtime (2xx rollup)

- (none)
