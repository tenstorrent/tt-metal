# CB→DFB Kernel Audit: `matmul`

**Date:** 2026-07-13
**Op root:** `ttnn/cpp/ttnn/operations/matmul/`

**Scope:** All in-scope device kernels referenced by the matmul + sparse_matmul program factories (`matmul_multicore`, `matmul_multicore_reuse_optimized`, `matmul_multicore_reuse_mcast_1d` incl. ring-all-gather/gathered variant, `matmul_multicore_reuse_mcast_2d`, `matmul_multicore_reuse_mcast_dram_sharded`, `matmul_multicore_reuse_batched_hs_dram_sharded`, `sparse/.../sparse_matmul_multicore_reuse_mcast_1d_optimized`).

## Overall verdict: RED

**Summary:** The **standard matmul path is nearly clean** — canonical linear-FIFO CBs that are mechanically portable, with only two mechanical getter-swap fixes (`fifo_page_size` → `get_entry_size()`, `fifo_num_pages` → `get_total_num_entries()`). However, the **ring-all-gather "gathered" variant** (`bmm_large_block_zm_fused_bias_activation_gathered.cpp`) treats `cb_in1` as a manually-indexed ring/window and **writes** `fifo_rd_ptr` via `get_local_cb_interface(...)` and reads `fifo_size`/`fifo_limit` (no getter today). That is a **hard GATE** and blocks the CB→DFB port until redesigned. Everything except the gathered variant is **YELLOW** (mechanical getter swaps); the gathered variant makes the op-level rollup **RED**.

## Scope notes

- **Unreferenced (out of scope, informational):** `kernels/compute/bmm_large_block_zm.cpp`, `kernels/dataflow/reader_bmm_tile_layout.cpp`, `kernels/dataflow/writer_bmm_tile_layout.cpp` — referenced only by `tt_metal/programming_examples`, tests, and docs, **not** by any matmul op factory. Not scanned / not gated.
- **`sparse_matmul`** shares kernels with the standard 1d path (`reader_bmm_tile_layout_in0_sender_padding.cpp`, `reader_bmm_tile_layout_in0_receiver.cpp`, `bmm_large_block_zm_fused_bias_activation.cpp`) — no new offenders; folds into the standard-path rows.

## CB portability

| CB | Class | Kernel(s) | 1xx status | 1xx notes | 2xx status | 2xx notes |
|----|-------|-----------|------------|-----------|------------|-----------|
| `cb_in0` / `cb_in0_transposed` | 1 | `bmm*.cpp`, `reader_bmm_tile_layout_in0*.cpp` | Portable | activation input, canonical `reserve/push`/`wait/pop` → `DataflowBuffer` | Portable | — |
| `cb_in1` (standard) | 1 | `bmm_large_block_zm_fused_bias_activation.cpp`, `reader_bmm_tile_layout_in1*.cpp` | Portable | weights, linear FIFO | Portable | — |
| `cb_out` | 1 | `bmm*.cpp`, `reader_*_writer_padding.cpp`, `reader_writer_bmm_tile_layout_in1.cpp` | Portable | pack → output, `get_write_ptr()` as L1/NOC addr only | Portable | — |
| `cb_intermed0` (mm_partials) | 1 | `bmm_large_block_zm_fused_bias_activation.cpp`, `bmm_large_block_zm.cpp`(unref) | Portable | partials accumulate via canonical pop/reserve — **no** rd/wr ptr surgery | Portable | — |
| `cb_bias` | 1 | `bmm_large_block_zm_fused_bias_activation.cpp` | Portable | bias broadcast, linear FIFO | Portable | — |
| `cb_id_out` (output) | 1 | `writer_unary_interleaved_start_id.cpp` | Portable | **NEEDS-FIX:** `get_local_cb_interface(cb_id_out).fifo_page_size` → `get_entry_size()` (getter exists) | Portable | same |
| `cb_id_in1` (ring reader view) | 1 | `reader_bmm_tile_layout_in1_ring_all_gather.cpp` | Portable | **NEEDS-FIX:** `get_local_cb_interface(cb_id_in1).fifo_num_pages` → `get_total_num_entries()` (merged, PR #49197) | Portable | same |
| `cb_in0` (ring) / `cb_in2`, `cb_sync`, `cb_sync2` | 1 | `reader_bmm_tile_layout_in0_ring_all_gather.cpp`, `reader_bmm_tile_layout_in1_ring_all_gather.cpp` | Portable | ring-all-gather sync CBs, canonical FIFO + `noc_semaphore` | Portable | — |
| `cb_remote` | 6 | `reader_bmm_tile_layout_in1_ring_all_gather.cpp` | Portable (prereq: review) | `api/remote_circular_buffer.h` RemoteCB — host-side RemoteCB spec, not a local FIFO; verify against host audit | Portable (prereq: review) | same |
| **`cb_in1` (gathered)** | **2 + 4** | **`bmm_large_block_zm_fused_bias_activation_gathered.cpp`** | **Blocked** | **GATE:** `get_local_cb_interface(cb_in1).fifo_rd_ptr` **write** (ring-index wrap, lines 51/88/125/130/133) + `fifo_size`/`fifo_limit` reads (no getter). Class 2 window / Class 4 manual ring wrap. Resolve before port. | **Blocked** | same |

## GATE hits (must be empty to merge)

- `bmm_large_block_zm_fused_bias_activation_gathered.cpp:51` — `get_local_cb_interface(cb_id).fifo_rd_ptr = val` (**write**) — Class 2/4 redesign (scratchpad + semaphores for the ring window, or strided/multi-DFB), not a mechanical getter swap.
- `bmm_large_block_zm_fused_bias_activation_gathered.cpp:56-57, 65, 121-122, 127` — `fifo_size` / `fifo_limit` **reads** — **no existing DFB getter** → file issue to Almeet (`get_total_buffer_size_bytes()` / ring-span getters, "Needed" in Runtime fixes table) before port proceeds.
- `bmm_large_block_zm_fused_bias_activation_gathered.cpp:88, 120, 125, 130, 133` — additional `fifo_rd_ptr` writes (ring wrap).
- `writer_unary_interleaved_start_id.cpp:19` — `get_local_cb_interface(cb_id_out).fifo_page_size` **read** — **mechanical**: → `get_entry_size()` (getter exists today). Clears GATE with a trivial swap.
- `reader_bmm_tile_layout_in1_ring_all_gather.cpp:167` — `get_local_cb_interface(cb_id_in1).fifo_num_pages` **read** — **mechanical**: → `get_total_num_entries()` (merged, PR #49197).

## Blocked on runtime (2xx rollup)

- `fifo_size` / `fifo_limit` getters (`get_total_buffer_size_bytes()` / ring-span) — **needed** for `bmm_large_block_zm_fused_bias_activation_gathered.cpp` ring wrap. Even with getters, the `fifo_rd_ptr` **writes** still require a Class 2/4 redesign (scratchpad + semaphores preferred; on Quasar consider strided/multi-producer DFB), so this variant stays RED regardless.

## Recommended path

1. **Standard matmul + sparse_matmul (YELLOW → GREEN):** two mechanical getter swaps clear both GATE field reads. All other CBs are canonical Class 1 → mechanical `CircularBuffer` → `DataflowBuffer`. Safe to port after the two swaps.
2. **Ring-all-gather / gathered variant (RED):** the `cb_in1` ring-window `fifo_rd_ptr` surgery must be redesigned — audit default is **scratchpad + semaphores** for the window layout (or strided/multi-DFB on Quasar). File the `fifo_size`/`fifo_limit` getter issue to Almeet. Do **not** port the gathered variant with a ptr hack because the field access is on the compute UNPACK side and the credits are decoupled from addresses.
