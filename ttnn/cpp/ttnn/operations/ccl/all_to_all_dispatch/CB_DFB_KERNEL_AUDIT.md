# CB→DFB Kernel Audit: `all_to_all_dispatch` [factory: `AllToAllDispatchSparse`]

**Date:** 2026-07-15
**Op root:** `ttnn/cpp/ttnn/operations/ccl/all_to_all_dispatch/`

**Scope:** `AllToAllDispatchSparse` (`all_to_all_dispatch_program_factory.cpp`, via helper `build_dispatch_program_descriptor`) → kernels: `device/kernels/dataflow/reader_all_to_all_dispatch.cpp`, `device/kernels/dataflow/writer_all_to_all_dispatch.cpp`. No donor kernels. Shared headers scanned: `ccl/common/kernels/moe_utils.hpp`, `data_movement/common/kernels/common.hpp`, `ccl/common/kernels/minimal_ccl_common.hpp`, `ccl/kernel_common/worker_routing_utils.hpp` (all zero hits).

## Overall verdict: GREEN

**Summary:** Three data CBs (`c_0`/`c_1`/`c_2`) are Class 1 cross-kernel linear FIFOs. Three CBs (`c_3` packet-header, `c_4` send-prep, `c_5` metadata) are Class 6 structural non-FIFO private-L1 scratch → autoportable to `ScratchpadSpec` (+ semaphores where a real handshake exists). Zero GATE / silent-wrong / quasar-blocked / LTA-prereq / ptr-surgery hits. All pointer use is bare `get_read_ptr`/`get_write_ptr` L1 addressing.

## CB portability

| CB | Class | Kernel(s) | 1xx status | 1xx notes | 2xx status | 2xx notes |
|----|-------|-----------|------------|-----------|------------|-----------|
| `c_0` `input_tensor_cb_id` | 1 | reader (producer), writer (consumer) | Portable | linear FIFO → `DataflowBuffer` | Portable | — |
| `c_1` `indices_tensor_cb_id` | 1 | reader (producer), writer (consumer) | Portable | linear FIFO; writer also reads via `get_read_ptr` as index scratch | Portable | — |
| `c_2` `mapping_tensor_cb_id` | 1 | reader (producer), writer (consumer) | Portable | linear FIFO | Portable | — |
| `c_3` `packet_header_cb_id` | 6 | writer only | Portable | autoportable: **ScratchpadSpec** — private L1, no FIFO ops; `get_read_ptr` reinterpreted as packet-header scratch | Portable | same |
| `c_4` `send_preparation_buffer_cb_id` | 6 | writer only | Portable | autoportable: **ScratchpadSpec** — private L1 bookkeeping scratch, zeroed via `zero_buffer_async`; no FIFO ops | Portable | same |
| `c_5` `metadata_buffer_id` (FullPacket impl only) | 6 | reader (write-staging), writer (read-out) | Portable | autoportable: **ScratchpadSpec**; reader→writer handoff not via CB credits — confirm existing semaphore/fabric barrier covers it, else **ScratchpadSpec + SemaphoreSpec** (still autoportable) | Portable | same |

## GATE hits (must be empty to merge)

- (none)

## Blocked on runtime (2xx rollup)

- (none)

## Notes

- `c_3`/`c_4`/`c_5` are allocated as CBs but never call `cb_reserve_back`/`cb_push_back`/`cb_wait_front`/`cb_pop_front` — Class 6 structural non-FIFO. Backing is private L1 (not borrowed from a tensor), so the end-state is `ScratchpadSpec`, not `LocalTensorAccessor`. Per the spec, ScratchpadSpec is autoportable and keeps the op GREEN.
- `writer:zero_buffer_async` builds a `CircularBuffer` handle purely to `noc.async_write_zeros` into `c_4` — no `LocalCBInterface` field access.
- `c_5` exists only when `impl == FullPacket` (`metadata_buffer_id = CTA34`; reader write-stages at `writer/reader.cpp` via `get_write_ptr`, writer reads out via `get_noc_addr(get_read_ptr(...))`).
