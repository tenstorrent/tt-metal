# CB→DFB Kernel Audit: `experimental/ccl/send_recv_async/send_async`

**Date:** 2026-07-15
**Op root:** `ttnn/cpp/ttnn/operations/experimental/ccl/send_recv_async/send_async/`

**Scope:** `device/kernels/sender_reader.cpp`, `sender_writer.cpp`.

## Overall verdict: GREEN

**Summary:** Fabric send op. CBs are `cb0`/`data_cb` (payload stream) and `fabric_packet_header_cb` (reserved packet-header region) — canonical Class 1 FIFO / reserved-region L1 with `get_read_ptr()`/`get_write_ptr()` as NoC/L1 addresses. All six Step-4 litmus scans return **zero hits** in scope. Mechanical `CircularBuffer` → `DataflowBuffer` on both arches.

## CB portability

| CB | Class | Kernel(s) | 1xx status | 1xx notes | 2xx status | 2xx notes |
|----|-------|-----------|------------|-----------|------------|-----------|
| `cb0` / `data_cb` | 1 | `sender_reader.cpp`, `sender_writer.cpp` | Portable | payload input stream, canonical FIFO; `get_read_ptr()` as NoC source | Portable | — |
| `fabric_packet_header_cb` | 1/6 | `sender_writer.cpp` | Portable | reserved fabric packet-header region, no field surgery | Portable | — |

## GATE hits (must be empty to merge)

- (none)

## Blocked on runtime (2xx rollup)

- (none)

## Recommended path

Port freely on both arches — canonical fabric send dataflow, no field surgery, no runtime API dependency, no LTA prerequisite.
