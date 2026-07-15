# CB→DFB Kernel Audit: `experimental/ccl/send_recv_async/send_async_d2h`

**Date:** 2026-07-15
**Op root:** `ttnn/cpp/ttnn/operations/experimental/ccl/send_recv_async/send_async_d2h/`

**Scope:** `device/kernels/d2h_sender_reader.cpp`.

## Overall verdict: GREEN

**Summary:** Single-kernel device-to-host fabric send reader. Uses a `scratch_cb` staging region with bare `get_read_ptr()`/`get_write_ptr()` NoC/L1 addressing and canonical FIFO sync. All six Step-4 litmus scans return **zero hits**. Mechanical `CircularBuffer` → `DataflowBuffer` on both arches.

## CB portability

| CB | Class | Kernel(s) | 1xx status | 1xx notes | 2xx status | 2xx notes |
|----|-------|-----------|------------|-----------|------------|-----------|
| `scratch_cb` | 1/6 | `d2h_sender_reader.cpp` | Portable | payload staging region; `get_read_ptr()`/`get_write_ptr()` as NoC/L1 address only (optional `ScratchpadSpec` hardening) | Portable | — |

## GATE hits (must be empty to merge)

- (none)

## Blocked on runtime (2xx rollup)

- (none)

## Recommended path

Port freely on both arches — canonical D2H fabric send, no field surgery, no runtime API dependency, no LTA prerequisite.
