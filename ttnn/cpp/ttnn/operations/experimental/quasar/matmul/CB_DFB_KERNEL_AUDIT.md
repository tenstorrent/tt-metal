# CB→DFB Kernel Audit: `experimental/quasar/matmul`

**Date:** 2026-07-15
**Op root:** `ttnn/cpp/ttnn/operations/experimental/quasar/matmul/`

**Scope:** All in-scope device kernels under `device/kernels/` referenced by the quasar matmul program factories (compute `bmm.cpp`, `bmm_large_block_zm.cpp`, `bmm_large_block_zm_fused_bias_activation{,_metal2}.cpp`, `bmm_large_block_zm_fused_bias_activation_gathered.cpp`, `bmm_fused_activation.hpp`; the `reader_bmm_tile_layout_*` reader family incl. sender/receiver, dram-sharded, ring-all-gather, and `_metal2` variants; `writer_bmm_tile_layout.cpp`, `reader_writer_bmm_tile_layout_in1.cpp`, `writer_unary_interleaved_start_id.cpp`). The gathered variant is referenced by `device/factory/matmul_multicore_reuse_mcast_1d_program_factory.cpp`.

## Overall verdict: RED

**Summary:** The **standard matmul path is nearly clean** — canonical linear-FIFO CBs, with only one mechanical getter-swap (`writer_unary_interleaved_start_id.cpp:18` `fifo_page_size` → `get_entry_size()`). But the **ring-all-gather "gathered" variant** (`bmm_large_block_zm_fused_bias_activation_gathered.cpp`) treats `cb_in1` as a manually-indexed ring/window on the UNPACK side: it **writes** `get_local_cb_interface(cb_in1).fifo_rd_ptr` (ring wrap, lines 50/87/124/129/132) and **reads** `fifo_size`/`fifo_limit` (no getter today, lines 55-57/64/83/119-126). That is a hard **GATE** (Class 2 window + Class 4 manual ring wrap) and blocks the CB→DFB port until redesigned. This mirrors the non-quasar `matmul` gathered variant. The gathered variant makes the op-level rollup **RED**; everything else is YELLOW → GREEN after the one getter swap.

## CB portability

| CB | Class | Kernel(s) | 1xx status | 1xx notes | 2xx status | 2xx notes |
|----|-------|-----------|------------|-----------|------------|-----------|
| `cb_in0` (activation) | 1 | `bmm*.cpp`, `reader_bmm_tile_layout_in0*.cpp` (+`_metal2`) | Portable | canonical `reserve/push`/`wait/pop` → `DataflowBuffer` | Portable | — |
| `cb_in1` (standard weights) | 1 | `bmm_large_block_zm_fused_bias_activation{,_metal2}.cpp`, `reader_bmm_tile_layout_in1*.cpp` (+`_metal2`) | Portable | weights, linear FIFO | Portable | — |
| `cb_out` | 1 | `bmm*.cpp`, `writer_bmm_tile_layout.cpp`, `reader_writer_bmm_tile_layout_in1.cpp` | Portable | pack → output, `get_write_ptr()` as L1/NoC addr only | Portable | — |
| `cb_intermed0` (mm_partials) | 1 | `bmm_large_block_zm_fused_bias_activation*.cpp`, `bmm_large_block_zm.cpp` | Portable | partials via canonical pop/reserve — **no** rd/wr ptr surgery | Portable | — |
| `cb_bias` | 1 | `bmm_large_block_zm_fused_bias_activation*.cpp` | Portable | bias broadcast, linear FIFO | Portable | — |
| `cb_id_out` (interleaved writer) | 1 | `writer_unary_interleaved_start_id.cpp:18` | Portable | **NEEDS-FIX:** `get_local_cb_interface(cb_id_out).fifo_page_size` → `get_entry_size()` (getter exists) | Portable | same |
| ring-all-gather sync CBs (`cb_in0`/`cb_in2`/`cb_sync*`) | 1 | `reader_bmm_tile_layout_in0_ring_all_gather.cpp`, `reader_bmm_tile_layout_in1_ring_all_gather.cpp` | Portable | canonical FIFO + `noc_semaphore` | Portable | — |
| **`cb_in1` (gathered ring/window)** | **2 + 4** | **`bmm_large_block_zm_fused_bias_activation_gathered.cpp`** | **Blocked** | **GATE:** `fifo_rd_ptr` **writes** (ring wrap, lines 50/87/124/129/132) + `fifo_size`/`fifo_limit` **reads** (no getter). Class 2 window / Class 4 manual ring wrap — redesign before port. | **Blocked** | same |

## GATE hits (must be empty to merge)

Field **writes** (pointer surgery — require redesign, not getter swaps):

- `bmm_large_block_zm_fused_bias_activation_gathered.cpp:50` — `get_local_cb_interface(cb_id).fifo_rd_ptr = val` (`update_local_cb_rd_ptr`) — Class 2/4 ring wrap.
- `bmm_large_block_zm_fused_bias_activation_gathered.cpp:87,124,129,132` — additional `fifo_rd_ptr` writes (`calculate_next_block_index_and_update_rd_ptr`, `update_rd_ptr_to_ring_index`).

Field **reads** with **no existing getter** (file issue to Almeet — `get_total_buffer_size_bytes()` / ring-span getters, "Needed" in Runtime fixes table):

- `bmm_large_block_zm_fused_bias_activation_gathered.cpp:55-57,64,83,119-126` — `fifo_size` / `fifo_limit` reads (ring start-addr, split detection, wrap math).

Field **read** (mechanical, getter exists today):

- `writer_unary_interleaved_start_id.cpp:18` — `get_local_cb_interface(cb_id_out).fifo_page_size` → `get_entry_size()`. Clears with a trivial swap.

## Blocked on runtime (2xx rollup)

- `fifo_size` / `fifo_limit` getters (`get_total_buffer_size_bytes()` / ring-span) — **needed** for the gathered ring wrap. Even with getters, the `fifo_rd_ptr` **writes** still require a Class 2/4 redesign (scratchpad + semaphores for the ring window preferred; strided/multi-producer DFB on Quasar), so this variant stays RED regardless.

## Recommended path

1. **Standard matmul + ring-all-gather (YELLOW → GREEN):** one mechanical getter swap (`writer_unary_interleaved_start_id.cpp:18`). All other CBs are canonical Class 1 → mechanical `CircularBuffer` → `DataflowBuffer`.
2. **Gathered variant (RED):** redesign the `cb_in1` ring-window `fifo_rd_ptr` surgery — audit default is **scratchpad + semaphores** for the ring window (or strided/multi-producer DFB on Quasar). File the `fifo_size`/`fifo_limit` getter issue to Almeet. Do **not** port with a ptr hack — the field access is on the compute UNPACK side and credits are decoupled from addresses.
