# CB→DFB Kernel Audit: `slice_write`

**Date:** 2026-07-15
**Op root:** `ttnn/cpp/ttnn/operations/experimental/slice_write/`

**Scope:** `slice_write_rm_interleaved_program_factory.cpp`, `slice_write_rm_sharded_input_program_factory.cpp`, `slice_write_tiled_sharded_input_program_factory.cpp` → kernels: `dataflow/slice_write_reader_interleaved.cpp`, `dataflow/slice_write_writer_interleaved.cpp`, `dataflow/slice_write_writer_interleaved_strided.cpp`; **donor** `eltwise/unary/device/kernels/dataflow/reader_unary_sharded.cpp`; donor header `data_movement/common/kernels/common.hpp`.

## Overall verdict: GREEN

**Summary:** All CBs are canonical Class 1 linear FIFOs. Step-4 litmus scans return **zero** hits across the interleaved/strided reader/writer kernels, the `reader_unary_sharded.cpp` donor, and the `common.hpp` header — no GATE, no silent-wrong, no ptr surgery, no field reads. Writers use bare `get_read_ptr()`/`get_write_ptr()` only as L1/NoC byte addresses (strided output writes are NoC-address arithmetic, not CB field surgery). Mechanical `CircularBuffer` → `DataflowBuffer` rename only.

## CB portability

| CB | Class | Kernel(s) | 1xx status | 1xx notes | 2xx status | 2xx notes |
|----|-------|-----------|------------|-----------|------------|-----------|
| `cb_id_in0` / `cb_id_in` | 1 | `slice_write_reader_interleaved.cpp`, `reader_unary_sharded.cpp` (donor) | Portable | input tiles/sticks, canonical FIFO | Portable | — |
| `cb_id_out` / `cb_id_out0` | 1 | `slice_write_writer_interleaved.cpp`, `slice_write_writer_interleaved_strided.cpp` | Portable | strided/interleaved output write, `get_read_ptr()`/`get_write_ptr()` as L1/NoC addr only | Portable | — |

## GATE hits (must be empty to merge)

- (none)

## Blocked on runtime (2xx rollup)

- (none)
