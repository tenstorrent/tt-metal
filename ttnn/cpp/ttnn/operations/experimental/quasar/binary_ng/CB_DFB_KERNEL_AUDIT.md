# CB→DFB Kernel Audit: `experimental/quasar/binary_ng`

**Date:** 2026-07-15
**Op root:** `ttnn/cpp/ttnn/operations/experimental/quasar/binary_ng/`

**Scope:** All device kernels under `device/kernels/`: compute (`eltwise_binary.cpp`, `eltwise_binary_no_bcast.cpp`, `eltwise_binary_scalar.cpp`, `eltwise_binary_sfpu*.cpp`, `eltwise_where_*.cpp`, headers `eltwise_utils*.hpp`) and dataflow (`reader_interleaved_no_bcast.cpp`, `writer_interleaved_scalar.cpp`, `fill_tile_utils.hpp`).

## Overall verdict: GREEN

**Summary:** Clean. All Step-4 litmus scans return **zero** hits across every compute variant and both dataflow kernels — no GATE field access, no silent-wrong pointers, no `read_tile_value`/`get_tile_address`, no `get_pointer_to_cb_data`, no `fifo_*` ptr surgery or field reads. Operand/scalar inputs and the output are canonical linear FIFOs; `get_read_ptr()`/`get_write_ptr()` are used only as L1/NoC byte addresses. Mechanical `CircularBuffer` → `DataflowBuffer` rename on both arches.

## CB portability

| CB | Class | Kernel(s) | 1xx status | 1xx notes | 2xx status | 2xx notes |
|----|-------|-----------|------------|-----------|------------|-----------|
| `cb_in0`, `cb_in1` | 1 | `reader_interleaved_no_bcast.cpp`, compute `eltwise_binary*.cpp`, `eltwise_where_*.cpp` | Portable | operand inputs, linear FIFO → `DataflowBuffer` | Portable | — |
| scalar/broadcast operand CB | 1 | `writer_interleaved_scalar.cpp`, `eltwise_binary_scalar.cpp`, `eltwise_*_sfpu_scalar.cpp`, `fill_tile_utils.hpp` | Portable | scalar fill, linear FIFO | Portable | — |
| `cb_out0` | 1 | compute `eltwise_*` kernels | Portable | pack → output, `get_write_ptr()` as L1/NoC addr only | Portable | — |

## GATE hits (must be empty to merge)

- (none)

## Blocked on runtime (2xx rollup)

- (none)
