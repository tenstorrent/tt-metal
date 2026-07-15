# CB→DFB Kernel Audit: `experimental/ccl/slice_reshard_async`

**Date:** 2026-07-15
**Op root:** `ttnn/cpp/ttnn/operations/experimental/ccl/slice_reshard_async/`

**Scope:** `device/kernels/minimal_default_reader.cpp`, `minimal_default_writer.cpp`.

## Overall verdict: GREEN

**Summary:** Slice + reshard dataflow op (fabric NoC + semaphore movement). The single `cb_output` is a canonical Class 1 linear FIFO using `cb_output.reserve_back(1)` (reader) / `cb_output.wait_front(1)` (writer) with `get_read_ptr()`/`get_write_ptr()` as NoC/L1 addresses. All six Step-4 litmus scans return **zero hits**. Mechanical `CircularBuffer` → `DataflowBuffer` on both arches.

## CB portability

| CB | Class | Kernel(s) | 1xx status | 1xx notes | 2xx status | 2xx notes |
|----|-------|-----------|------------|-----------|------------|-----------|
| `cb_output` | 1 | `minimal_default_reader.cpp`, `minimal_default_writer.cpp` | Portable | reader→writer slice staging, canonical `reserve_back`/`wait_front` FIFO | Portable | — |

## GATE hits (must be empty to merge)

- (none)

## Blocked on runtime (2xx rollup)

- (none)

## Recommended path

Port freely on both arches — canonical slice/reshard dataflow FIFO, no field surgery, no runtime API dependency, no LTA prerequisite.
