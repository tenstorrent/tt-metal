# CB→DFB Kernel Audit: `all_to_all_async_generic`

**Date:** 2026-07-15
**Op root:** `ttnn/cpp/ttnn/operations/experimental/ccl/all_to_all_async_generic/`

**Scope:** All in-scope device kernels under `device/kernels/`: `all_to_all_sender_reader.cpp`, `all_to_all_sender_writer.cpp`.

## Overall verdict: GREEN

**Summary:** Pure fabric NoC/semaphore all-to-all movement (sender reader + writer only, no compute). All Step-4 litmus scans return **zero** hits — no GATE, no silent-wrong, no `read_tile_value`/`get_tile_address`, no `get_pointer_to_cb_data`, no ptr surgery, no `fifo_*` field reads. Only a reserved fabric packet-header scratch CB appears; movement uses `get_read_ptr()`/`get_write_ptr()` as L1/NoC addresses. Mechanical rename.

## CB portability

| CB | Class | Kernel(s) | 1xx status | 1xx notes | 2xx status | 2xx notes |
|----|-------|-----------|------------|-----------|------------|-----------|
| `reserved_packet_header_cb_id` / `cb_packet_header` | 1/6 | `all_to_all_sender_reader.cpp`, `all_to_all_sender_writer.cpp` | Portable | fabric packet-header scratch; `reserve_back` + `get_write_ptr()`, no field access | Portable | — |
| data staging CB(s) | 1 | `all_to_all_sender_reader.cpp`, `all_to_all_sender_writer.cpp` | Portable | canonical FIFO; `get_read_ptr()`/`get_write_ptr()` as L1/NoC addr | Portable | — |

## GATE hits (must be empty to merge)

- (none)

## Blocked on runtime (2xx rollup)

- (none)

## Recommended path

Port freely — mechanical `CircularBuffer` → `DataflowBuffer` rename. No field surgery, no runtime API dependency, no LTA prerequisite.
