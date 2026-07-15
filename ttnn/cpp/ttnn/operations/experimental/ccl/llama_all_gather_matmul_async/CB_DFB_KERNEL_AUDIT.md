# CB→DFB Kernel Audit: `llama_all_gather_matmul_async`

**Date:** 2026-07-15
**Op root:** `ttnn/cpp/ttnn/operations/experimental/ccl/llama_all_gather_matmul_async/`

**Scope:** All in-scope device kernels under `device/kernels/`: `compute/bmm_large_block_zm_fused_bias_activation_gathered.cpp`, `reader_bmm_tile_layout_in0_ring_all_gather.cpp`, `reader_bmm_tile_layout_in1_ring_all_gather.cpp`, `worker_reader.cpp`, `worker_receiver.cpp`, `worker_writer.cpp`.

## Overall verdict: RED

**Summary:** The all-gather fabric dataflow kernels (`worker_*`, `reader_bmm_tile_layout_in0/in1_ring_all_gather.cpp`) are clean canonical FIFO — zero litmus hits. But the fused compute kernel `bmm_large_block_zm_fused_bias_activation_gathered.cpp` (the same "gathered" matmul offender flagged in the matmul audit) treats `cb_id_in1` as a manually-indexed ring/window: it grabs `LocalCBInterface& local_cb = get_local_cb_interface(cb_id)` and **writes** `local_cb.fifo_rd_ptr` (ring-index wrap) and **reads** `local_cb.fifo_size` / `local_cb.fifo_limit` (no getter today). That is a hard **GATE** + Class 2/4 pointer surgery, so the CB→DFB port is blocked until the ring window is redesigned. The op-level rollup is **RED**.

## CB portability

| CB | Class | Kernel(s) | 1xx status | 1xx notes | 2xx status | 2xx notes |
|----|-------|-----------|------------|-----------|------------|-----------|
| **`cb_id_in1` (gathered ring window)** | **2 + 4** | **`compute/bmm_large_block_zm_fused_bias_activation_gathered.cpp`** | **Blocked** | **GATE:** `get_local_cb_interface(cb_id).fifo_rd_ptr` **writes** (ring-index wrap, lines 50, 87, 124, 129, 132) + `fifo_size`/`fifo_limit` **reads** (lines 55–57, 64, 120–126, no getter). Class 2 window / Class 4 manual ring wrap — redesign (scratchpad + semaphores), not a getter swap. | **Blocked** | same — prefer scratchpad+sems or strided multi-producer DFB; `fifo_rd_ptr` write cannot use implicit sync |
| `cb_id_in0` / `mm_partials_cb` (compute) | 1 | `compute/bmm_large_block_zm_fused_bias_activation_gathered.cpp` | Portable | activation input + partials via canonical `wait_front`/`pop_front`/`copy_block_matmul_partials` — no ptr surgery | Portable | — |
| `cb_in0` / `cb_in2`, sync CBs (ring readers) | 1 | `reader_bmm_tile_layout_in0_ring_all_gather.cpp`, `reader_bmm_tile_layout_in1_ring_all_gather.cpp` | Portable | ring-all-gather staging + `noc_semaphore` sync, canonical FIFO | Portable | — |
| all-gather worker CBs + packet header | 1/6 | `worker_reader.cpp`, `worker_receiver.cpp`, `worker_writer.cpp` | Portable | canonical FIFO + fabric packet-header scratch; `get_read_ptr()`/`get_write_ptr()` as L1/NoC addr | Portable | — |
| `cb_out` (final pack) | 1 | `compute/bmm_large_block_zm_fused_bias_activation_gathered.cpp`, `worker_writer.cpp` | Portable | pack → output, `get_write_ptr()` as L1/NoC addr | Portable | — after ring window redesigned |

## GATE hits (must be empty to merge)

- `compute/bmm_large_block_zm_fused_bias_activation_gathered.cpp:50` — `get_local_cb_interface(cb_id).fifo_rd_ptr = val` (**write**, via `update_local_cb_rd_ptr`) — Class 2/4 redesign, not a mechanical swap.
- `compute/bmm_large_block_zm_fused_bias_activation_gathered.cpp:87, 124, 129, 132` — additional `local_cb.fifo_rd_ptr` **writes** (ring wrap in `calculate_next_block_index_and_update_rd_ptr` / `update_rd_ptr_to_ring_index`).
- `compute/bmm_large_block_zm_fused_bias_activation_gathered.cpp:45, 63, 81, 83, 119` — `local_cb.fifo_rd_ptr` **reads** (ring base bookkeeping).
- `compute/bmm_large_block_zm_fused_bias_activation_gathered.cpp:55–57, 64, 120–126` — `local_cb.fifo_size` / `local_cb.fifo_limit` **reads** — **no existing DFB getter** → file issue to Almeet (`get_total_buffer_size_bytes()` / ring-span getters, "Needed" in Runtime fixes table) before port proceeds.

## Blocked on runtime (2xx rollup)

- `fifo_size` / `fifo_limit` getters (`get_total_buffer_size_bytes()` / ring-span) — **needed** for the `cb_id_in1` ring wrap. Even with getters, the `fifo_rd_ptr` **writes** still require a Class 2/4 redesign, so this variant stays RED regardless of the getter landing.

## Recommended path

Same as the matmul `bmm_*_gathered` variant: the `cb_id_in1` ring-window `fifo_rd_ptr` surgery must be redesigned — audit default is **scratchpad + semaphores** for the window layout (or a strided multi-producer DFB on Quasar). File the `fifo_size`/`fifo_limit` getter issue to Almeet. Do **not** port the gathered compute with a ptr hack: the field access is on the compute UNPACK side and credits are decoupled from addresses. The surrounding all-gather dataflow kernels are already clean and can port mechanically once the compute redesign lands.
