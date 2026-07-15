# CB→DFB Kernel Audit: `experimental/ccl/send_recv_async/recv_async_h2d`

**Date:** 2026-07-15
**Op root:** `ttnn/cpp/ttnn/operations/experimental/ccl/send_recv_async/recv_async_h2d/`

**Scope:** `device/kernels/h2d_receiver_writer.cpp`.

## Overall verdict: GREEN

**Summary:** Single-kernel host-to-device fabric receive writer. CBs are reserved-region staging / packet-header buffers used with bare `get_read_ptr()`/`get_write_ptr()` NoC/L1 addressing and canonical FIFO sync. All six Step-4 litmus scans return **zero hits**. Mechanical `CircularBuffer` → `DataflowBuffer` on both arches.

## CB portability

| CB | Class | Kernel(s) | 1xx status | 1xx notes | 2xx status | 2xx notes |
|----|-------|-----------|------------|-----------|------------|-----------|
| all in-scope CBs (payload staging / packet-header) | 1/6 | `h2d_receiver_writer.cpp` | Portable | reserved-region L1 + canonical FIFO; `get_write_ptr()` as NoC/L1 address only | Portable | — |

## GATE hits (must be empty to merge)

- (none)

## Blocked on runtime (2xx rollup)

- (none)

## Recommended path

Port freely on both arches — canonical H2D fabric receive, no field surgery, no runtime API dependency, no LTA prerequisite.
