# CB→DFB Kernel Audit: `experimental/ccl/strided_reduce_scatter_async`

**Date:** 2026-07-15
**Op root:** `ttnn/cpp/ttnn/operations/experimental/ccl/strided_reduce_scatter_async/`

**Scope:** `device/kernels/minimal_ring_strided_reduce_scatter_async_reader.cpp`, `minimal_ring_strided_reduce_scatter_async_writer.cpp`, `minimal_ring_reduction.cpp`, `strided_ring_reduce_scatter_common.hpp`.

## Overall verdict: GREEN

**Summary:** Strided ring reduce-scatter dataflow + a small addcmul/reduction compute kernel. All CBs (`cb_input_id`, `cb_intermediate_id`, `cb_reader_output_id`, `cb_compute_output_id`, plus compute temps `addcmul_a_cb`, `addcmul_b_cb`, `addcmul_temp_cb`) are canonical Class 1 linear FIFOs using `cb_reserve_back`/`cb_wait_front`/`cb_push_back`/`cb_pop_front`; `get_read_ptr()`/`get_write_ptr()` are NoC/L1 addresses only. All six Step-4 litmus scans return **zero hits**. Mechanical `CircularBuffer` → `DataflowBuffer` on both arches.

## CB portability

| CB | Class | Kernel(s) | 1xx status | 1xx notes | 2xx status | 2xx notes |
|----|-------|-----------|------------|-----------|------------|-----------|
| `cb_input_id` | 1 | `*_reader.cpp`, `minimal_ring_reduction.cpp` | Portable | input tensor reader → compute, canonical FIFO | Portable | — |
| `cb_intermediate_id` | 1 | `*_reader.cpp`, `minimal_ring_reduction.cpp` | Portable | ring partials, linear FIFO | Portable | — |
| `cb_reader_output_id` | 1 | `*_reader.cpp`, `*_writer.cpp` | Portable | reader-produced output slice (first step), linear FIFO | Portable | — |
| `cb_compute_output_id` | 1 | `minimal_ring_reduction.cpp`, `*_writer.cpp` | Portable | reduced output from compute → writer, canonical FIFO | Portable | — |
| `addcmul_a_cb`, `addcmul_b_cb`, `addcmul_temp_cb` | 1 | `*_reader.cpp`, `minimal_ring_reduction.cpp` | Portable | addcmul fused-reduction compute temps, canonical FIFO | Portable | — |

## GATE hits (must be empty to merge)

- (none)

## Blocked on runtime (2xx rollup)

- (none)

## Recommended path

Port freely on both arches — canonical strided reduce-scatter dataflow + compute FIFOs, no field surgery, no runtime API dependency, no LTA prerequisite.
