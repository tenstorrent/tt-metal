# CB→DFB Kernel Audit: `ccl`

**Date:** 2026-07-15
**Op root:** `ttnn/cpp/ttnn/operations/ccl/`

**Scope:** all in-scope EDM/fabric dataflow + compute kernels across `all_gather`, `all_broadcast`, `all_reduce`, `all_to_all_combine`, `all_to_all_dispatch`, `broadcast`, `reduce_scatter`, `reduce_to_root`, `mesh_partition`, plus shared `kernel_common/worker_edm_utils.hpp`, `common/kernels/`, `kernels/edm/`.

## Overall verdict: RED

**Summary:** CCL is overwhelmingly Class 1 canonical FIFO (fabric `fetch_chunk`/`send_chunk` = `reserve_back`/`push_back`/`wait_front`/`pop_front` with `get_read_ptr`/`get_write_ptr` NOC cursors) plus semaphore/fabric sync — all portable. The **only** illegal pattern in the entire tree is a `fifo_num_pages` field read inside two debug `ASSERT`s in the shared `worker_edm_utils.hpp`. That is a hard GATE but a mechanical `get_total_num_entries()` swap (PR #49197, merged). One header fix clears every ccl subop that includes it → GREEN.

## CB portability

| CB | Class | Kernel(s) | 1xx status | 1xx notes | 2xx status | 2xx notes |
|----|-------|-----------|------------|-----------|------------|-----------|
| filler-page CBs | 1 | `kernel_common/worker_edm_utils.hpp` (`push_filler_pages_to_cb` / `pop_filler_pages_from_cb`) | Blocked | GATE: `worker_edm_utils.hpp:21,26` `.fifo_num_pages` in `ASSERT` → `get_total_num_entries()`; otherwise `reserve/push` + `wait/pop` linear FIFO | Blocked | same |
| chunk transfer CBs (`fetch_chunk`/`send_chunk` and per-op reader/writer CBs) | 1 | `worker_edm_utils.hpp`, all_gather/reduce_scatter/broadcast/all_to_all_* reader+writer kernels | Portable | canonical fabric FIFO, `get_write_ptr()`/`get_read_ptr()` NOC cursors only | Portable | — |
| reduction/combine compute CBs | 1 | reduce_scatter / all_reduce / reduce_to_root compute kernels | Portable | linear producer/consumer; no field access | Portable | — |

## GATE hits (must be empty to merge)

- `ccl/kernel_common/worker_edm_utils.hpp:21` — `get_local_cb_interface(cb_id).fifo_num_pages` (ASSERT) — → `get_total_num_entries()`
- `ccl/kernel_common/worker_edm_utils.hpp:26` — `get_local_cb_interface(cb_id).fifo_num_pages` (ASSERT) — → `get_total_num_entries()`

## Blocked on runtime (2xx rollup)

- (none — `get_total_num_entries()` already merged)
