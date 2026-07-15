# CB→DFB Kernel Audit: `deepseek_prefill/outbound_socket_service_sync`

**Date:** 2026-07-15
**Op root:** `ttnn/cpp/ttnn/operations/experimental/deepseek_prefill/outbound_socket_service_sync/`

**Scope:** `device/kernels/outbound_socket_service_sync_writer.cpp`

## Overall verdict: GREEN

**Summary:** Zero litmus hits — no `get_local_cb_interface(...)` field access, no `read_tile_value`/`get_tile_address`, no `get_pointer_to_cb_data`, no `get_cb_tiles_*_ptr`, no ptr surgery, no `fifo_*` field reads. This is a socket/semaphore synchronization writer: it uses a single scratch CB (`scratch_cb_index`) as a staging L1 region addressed only through bare pointers, with `noc_semaphore` handshakes for the real cross-endpoint sync. Mechanical `CircularBuffer` → `DataflowBuffer` rename; safe to port on both arches.

## CB portability

| CB | Class | Kernel(s) | 1xx status | 1xx notes | 2xx status | 2xx notes |
|----|-------|-----------|------------|-----------|------------|-----------|
| `scratch_cb` (`scratch_cb_index`) | 6 | `outbound_socket_service_sync_writer.cpp` | Portable | staging scratch, bare pointer + `noc_semaphore` sync — autoportable (`ScratchpadSpec` + sems cleaner end-state, not port-gating) | Portable | same |

## GATE hits (must be empty to merge)

- (none)

## Blocked on runtime (2xx rollup)

- (none)
