# CB→DFB Kernel Audit: `experimental/ccl/ring_attention_all_gather_async`

**Date:** 2026-07-13
**Op root:** `ttnn/cpp/ttnn/operations/experimental/ccl/ring_attention_all_gather_async/`

**Scope:** `device/kernels/ring_attention_all_gather_reader.cpp`, `device/kernels/ring_attention_all_gather_writer.cpp`.

## Overall verdict: RED

**Summary:** The **writer is clean** (canonical consumer, linear FIFO). The **reader** implements a manual ring/staging pipe on `cb_output`: it snapshots `fifo_limit` and `fifo_size` via `get_local_cb_interface(cb_output_id)` and hand-wraps the write address (`l1_write_addr -= cb_fifo_size` when `>= cb_fifo_limit`) while posting per-packet `push_back` credits. This is a **Class 4 credit/address decoupling** pattern. The `fifo_size`/`fifo_limit` reads have **no existing DFB getter** (Runtime "Needed" — `get_total_buffer_size_bytes()` / ring-span getters), so this is a hard GATE that blocks the port until (a) getters are filed to Almeet **and** (b) the manual ring wrap is redesigned. Even with getters, the ring-wrap logic wants scratchpad + semaphores rather than a hacked DFB.

## CB portability

| CB | Class | Kernel(s) | 1xx status | 1xx notes | 2xx status | 2xx notes |
|----|-------|-----------|------------|-----------|------------|-----------|
| `cb_output` (staging ring) | 4 | `ring_attention_all_gather_reader.cpp` | Blocked | GATE: `fifo_limit`/`fifo_size` reads (`:148,149`) — **no getter**; manual ring wrap (`:70-71`) + per-packet `push_back` (`:86`). Prefer scratchpad + semaphores. | Blocked | same; on Quasar disable implicit sync if kept on DFB ptr surgery |
| `cb_output` (consumer) | 1 | `ring_attention_all_gather_writer.cpp` | Portable | writer drains canonically (`wait_front`/`pop_front`) → `DataflowBuffer` | Portable | — |

## GATE hits (must be empty to merge)

- `ring_attention_all_gather_reader.cpp:148` — `get_local_cb_interface(cb_output_id).fifo_limit` **read** — **no DFB getter today** → file issue to Almeet (ring-span getter).
- `ring_attention_all_gather_reader.cpp:149` — `get_local_cb_interface(cb_output_id).fifo_size` **read** — **no DFB getter today** → `get_total_buffer_size_bytes()` (Runtime "Needed").

Context: `cb_output.reserve_back(batch_pages)` at `:64`, hand-wrap at `:70-71` (`if (l1_write_addr >= cb_fifo_limit) l1_write_addr -= cb_fifo_size;`), `cb_output.push_back(packet_size_in_pages)` at `:86`. The comment at `:44-45` acknowledges bypassing the CB's contiguous-write assumption.

## Blocked on runtime (2xx rollup)

- `get_total_buffer_size_bytes()` / ring-span getters (**Needed — file to Almeet**) for the `fifo_size`/`fifo_limit` reads. This matches the `ring_attention` row in the guide's "Runtime fixes in flight" table.

## Recommended path

Audit default is **scratchpad + semaphores**: hold the packet ring slots in a `ScratchpadSpec` with per-slot/per-packet `SemaphoreSpec` handshakes, replacing the manual `fifo_size`/`fifo_limit` wrap and per-packet `push_back` credits. If the DFB ptr path is retained as a v1 fallback, it needs the Almeet getters **and** implicit-sync disabled on Quasar (`Gen2Config::disable_implicit_sync_for`) — record as **Portable (workaround)** only after getters land. The writer needs no change beyond the mechanical `CircularBuffer` → `DataflowBuffer` rename.
