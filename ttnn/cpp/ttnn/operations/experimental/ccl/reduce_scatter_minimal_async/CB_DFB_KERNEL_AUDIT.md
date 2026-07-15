# CB→DFB Kernel Audit: `experimental/ccl/reduce_scatter_minimal_async`

**Date:** 2026-07-15
**Op root:** `ttnn/cpp/ttnn/operations/experimental/ccl/reduce_scatter_minimal_async/`

**Scope:** `device/kernels/{ring,line,dim_zero_ring,dim_zero_line}_reduce_scatter_minimal_async_{reader,writer}.cpp` + matching `*_reduction.cpp` compute kernels (12 files).

## Overall verdict: GREEN

**Summary:** Ring/line reduce-scatter dataflow + a small elementwise reduction compute kernel. Every CB (`cb_input_id`, `cb_interm_id`, `cb_interm2_id`, `cb_compute_output_id`, `cb_reader_output_id`) is a canonical Class 1 linear FIFO using standard `cb_reserve_back`/`cb_wait_front`/`cb_push_back`/`cb_pop_front`; readers/writers use `get_read_ptr()`/`get_write_ptr()` only as NoC/L1 byte addresses. All six Step-4 litmus scans return **zero hits** across all 12 kernels. Mechanical `CircularBuffer` → `DataflowBuffer` on both arches.

## CB portability

| CB | Class | Kernel(s) | 1xx status | 1xx notes | 2xx status | 2xx notes |
|----|-------|-----------|------------|-----------|------------|-----------|
| `cb_input_id` | 1 | `*_reader.cpp`, `*_reduction.cpp` | Portable | input tensor reader → compute, canonical FIFO | Portable | — |
| `cb_interm_id`, `cb_interm2_id` | 1 | `*_reader.cpp`, `*_reduction.cpp` | Portable | intermediate partials for ring/line reduce, linear FIFO | Portable | — |
| `cb_compute_output_id` | 1 | `*_reduction.cpp`, `*_writer.cpp` | Portable | reduced output from compute → writer, canonical FIFO | Portable | — |
| `cb_reader_output_id` | 1 | `*_reader.cpp`, `*_writer.cpp` | Portable | reader-produced output slice (first step), linear FIFO | Portable | — |

## GATE hits (must be empty to merge)

- (none)

## Blocked on runtime (2xx rollup)

- (none)

## Recommended path

Port freely on both arches — canonical reduce-scatter dataflow + compute FIFOs, no field surgery, no runtime API dependency, no LTA prerequisite.
