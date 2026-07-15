# CB→DFB Kernel Audit: `experimental/ccl/strided_all_gather_async`

**Date:** 2026-07-15
**Op root:** `ttnn/cpp/ttnn/operations/experimental/ccl/strided_all_gather_async/`

**Scope:** `device/kernels/minimal_default_reader.cpp`, `minimal_default_writer.cpp`, `strided_all_gather_common.hpp`, `fused_receiver_utils.hpp`.

## Overall verdict: GREEN

**Summary:** Strided all-gather dataflow op (fabric NoC + semaphore movement). The `cb_output` is a canonical Class 1 linear FIFO — `cb_output.reserve_back(max_tiles_per_packet)` on the producer side and `cb_output.wait_front(max_tiles_per_packet)` on the consumer side (`strided_all_gather_common.hpp:147,248`) — with `get_read_ptr()`/`get_write_ptr()` used only as NoC/L1 addresses. All six Step-4 litmus scans return **zero hits** across the reader, writer, and both shared headers. Mechanical `CircularBuffer` → `DataflowBuffer` on both arches.

## CB portability

| CB | Class | Kernel(s) | 1xx status | 1xx notes | 2xx status | 2xx notes |
|----|-------|-----------|------------|-----------|------------|-----------|
| `cb_output` | 1 | `minimal_default_reader.cpp`, `minimal_default_writer.cpp`, `strided_all_gather_common.hpp` | Portable | per-packet gather staging, canonical `reserve_back`/`wait_front`/`push_back`/`pop_front` FIFO | Portable | — |

## GATE hits (must be empty to merge)

- (none)

## Blocked on runtime (2xx rollup)

- (none)

## Recommended path

Port freely on both arches — canonical strided all-gather dataflow FIFO, no field surgery, no runtime API dependency, no LTA prerequisite. (This op's kernels are also the donor set for the composite `strided_all_gather_minimal_matmul_async`.)
