# CB→DFB Kernel Audit: `experimental/ccl/neighbor_pad_async`

**Date:** 2026-07-15
**Op root:** `ttnn/cpp/ttnn/operations/experimental/ccl/neighbor_pad_async/`

**Scope:** `device/kernels/minimal_default_reader.cpp`, `minimal_default_writer.cpp`, `local_copy_reader.cpp`, `local_copy_writer.cpp`, `phase2_w_reader.cpp`.

## Overall verdict: GREEN

**Summary:** Pure dataflow neighbor-halo pad op (fabric NoC + semaphore movement). CBs (`cb_output`, `recv_cb`) are canonical Class 1 linear FIFO / bare-pointer L1 staging with `reserve_back`/`wait_front`/`push_back`/`pop_front`. All six Step-4 litmus scans return **zero hits** in scope. Mechanical `CircularBuffer` → `DataflowBuffer` on both arches.

## CB portability

| CB | Class | Kernel(s) | 1xx status | 1xx notes | 2xx status | 2xx notes |
|----|-------|-----------|------------|-----------|------------|-----------|
| `cb_output` | 1 | `minimal_default_reader.cpp`, `minimal_default_writer.cpp`, `local_copy_reader.cpp`, `local_copy_writer.cpp`, `phase2_w_reader.cpp` | Portable | output/halo staging, canonical FIFO; `get_read_ptr()`/`get_write_ptr()` as NoC addresses only | Portable | — |
| `recv_cb` | 1 | `minimal_default_reader.cpp`, `minimal_default_writer.cpp` | Portable | received-neighbor staging, linear FIFO | Portable | — |

## GATE hits (must be empty to merge)

- (none)

## Blocked on runtime (2xx rollup)

- (none)

## Recommended path

Port freely on both arches — canonical fabric dataflow, no field surgery, no runtime API dependency, no LTA prerequisite.
