# CB→DFB Kernel Audit: `experimental/quasar/conv2d`

**Date:** 2026-07-13
**Op root:** `ttnn/cpp/ttnn/operations/experimental/quasar/conv2d/`

**Scope:** All in-scope device kernels under `device/kernels/`. Fused compute (`conv_bmm_tilize.cpp` + `conv_bmm_tilize_metal2.cpp`), depthwise compute (`compute_depthwise_conv1d{,_metal2}.cpp`), activation/weight readers (`reader_conv_activations_*`, `activation_reader_width_sharded*`, `weights_reader_width_sharded*`, `reader_depthwise_conv1d*`), mcast reader/writers (`reader_writer_tiled_out_1d_*`, `writer_tiled_out_2d_*`), and shared header `conv_reader_common.hpp`. Both legacy and `_metal2` variants are in scope.

## Overall verdict: RED

**Summary:** conv2d is the guide's **P0 reference bad example** and it holds here. One fused compute kernel combines Class 2 (activation-reuse window), Class 3 (split-reader scatter), Class 4 (held-base restore), and Class 5 (`matmul_partials_cb` L1-accumulation save/restore), all implemented via **`get_local_cb_interface(...).fifo_wr_ptr`/`fifo_rd_ptr` writes** (42 across the fused compute + readers). These are field **writes** (pointer surgery), not mechanical getter swaps, so the CB→DFB port is **hard-GATE blocked** until the window/scatter/partials patterns are redesigned. Notably the `_metal2` variants already rename buffers to `dfb::*` but **retain the field writes** — the GATE is unresolved in the ported code too.

## CB portability

| CB | Class | Kernel(s) | 1xx status | 1xx notes | 2xx status | 2xx notes |
|----|-------|-----------|------------|-----------|------------|-----------|
| `dfb::act` / `cb_id_act` | 2 | `reader_conv_activations_*`, `conv_reader_common.hpp`, `activation_reader_width_sharded*` | Portable (workaround) | **undesirable but OK hack:** Class 2 activation-reuse window via `fifo_wr_ptr` jump (`conv_reader_common.hpp:91,108`); uplift: scratchpad + semaphores | Blocked | Class 2 window — `fifo_wr_ptr` write is a GATE; prefer scratchpad+sems (or strided DFB), not ptr surgery |
| `dfb::act_second_reader` / `cb_id_act_second_reader` | 3 | `reader_writer_tiled_out_1d_*`, `writer_tiled_out_2d_*`, `reader_conv_..._metal2.cpp` | Portable (workaround) | **undesirable but OK hack:** Class 3 split-reader scatter `fifo_wr_ptr = l1_write_addr_act`; WH/BH keep ptr scatter | Blocked | Class 3 scatter — Quasar: NEEDS-DESIGN-DECISION (strided multi-producer DFB vs multi-DFB combine vs disable split reader) |
| `dfb::act_tilized` / `tilized_in0_cb_id` | 3 + 4 | `conv_bmm_tilize{,_metal2}.cpp`, `compute_depthwise_conv1d*` | Blocked | GATE: `fifo_wr_ptr` held-base restore (`conv_bmm_tilize.cpp:369`, `_metal2:413`) + scatter tilize | Blocked | same — held-base restore is credit/address decoupling |
| `matmul_partials_cb` / `dfb::matmul_partials` | 5 | `conv_bmm_tilize{,_metal2}.cpp` | Blocked | GATE: `fifo_rd_ptr`/`fifo_wr_ptr` save/restore across inner dim (`conv_bmm_tilize.cpp:296-632`), `packer_l1_acc` L1 accumulation | Blocked | Class 5 in-place accumulator; LLK-driven partials — needs DFB ptr save/restore (WEIRD-OK, disable implicit sync) or dest-only acc |
| `in0_cb_id` / `dfb::act` (compute view) | 2 | `conv_bmm_tilize{,_metal2}.cpp` | Blocked | GATE: `fifo_rd_ptr` read as activation-reuse base (`conv_bmm_tilize.cpp:264,270`) | Blocked | Class 2 window read base |
| `in_cb_id` / `out_cb_id` (depthwise) | 4/5 | `conv_bmm_tilize{,_metal2}.cpp`, `compute_depthwise_conv1d*` | Blocked | GATE: `fifo_rd_ptr`/`fifo_wr_ptr` set+restore (`conv_bmm_tilize.cpp:71,77,111,163`) | Blocked | held-base restore before `push_back` |
| `dfb::weights` / `dfb::bias` | 1 | `weights_reader_width_sharded_metal2.cpp`, `conv_bmm_tilize_metal2.cpp`, mcast writers | Portable | linear FIFO weights/bias → `DataflowBuffer` | Portable | — |
| `dfb::out` / `out_cb_id` (final pack) | 1 | `conv_bmm_tilize{,_metal2}.cpp`, `compute_depthwise_conv1d_metal2.cpp` | Portable | pack → output, `get_write_ptr()` as L1/NOC addr | Portable | — after partials path redesigned |
| `dfb::act_row_major` / `cb_id_act_row_major_bfloat` | 1 | `reader_conv_activations_*`, `activation_reader_width_sharded*` | Portable | row-major staging, linear FIFO | Portable | — |
| `dfb::act_sharded` | 1 | `reader_writer_tiled_out_1d_*`, `writer_tiled_out_2d_*` | Portable | resident sharded input | Portable | — |
| `dfb::reader_indices` | 1 | reader/writer mcast kernels | Portable | index stream, linear FIFO | Portable | — |
| `dfb::num_pages` read (pool-style) | 1 | `conv_reader_common.hpp:25`, `conv_bmm_tilize.cpp` | Portable | **NEEDS-FIX:** `fifo_num_pages` → `get_total_num_entries()` (merged, PR #49197) | Portable | same |

## GATE hits (must be empty to merge)

Field **writes** (pointer surgery — require redesign, not getter swaps):

- `conv_reader_common.hpp:91,108` — `get_local_cb_interface(cb_id_act).fifo_wr_ptr = ...` (Class 2 activation-reuse window)
- `reader_conv_activations_padded_with_halo_3x3_weights_v2{,_metal2}.cpp:89,116` — `.fifo_wr_ptr = l1_write_addr_act` (Class 3 scatter)
- `reader_writer_tiled_out_1d_mcast_{sender,receiver}_..._metal2.cpp:180,137` and legacy `:155,113` — `.fifo_wr_ptr = l1_write_addr_act`
- `conv_bmm_tilize.cpp:71,77,111,163,369,519-632` and `conv_bmm_tilize_metal2.cpp:104,110,140,184,413,558-669` — `matmul_partials_cb`/`out_cb_id`/`tilized_in0_cb_id` `fifo_rd_ptr`/`fifo_wr_ptr` save/restore (Class 4/5)

Field **reads** (mechanical, getter exists):

- `conv_reader_common.hpp:25`, `conv_bmm_tilize.cpp` — `fifo_num_pages` → `get_total_num_entries()`

## Blocked on runtime (2xx rollup)

- No missing-getter dependency for the writes; these are **redesigns**, not API gaps. Recommended v1 path per class:
  - **Class 2** activation-reuse window → **scratchpad + semaphores** (audit default).
  - **Class 3** split-reader scatter → **WH/BH:** keep ptr scatter (WEIRD-OK); **Quasar:** strided multi-producer DFB (`stride_in_entries`) or disable split reader.
  - **Class 4/5** held-base restore + `matmul_partials_cb` L1-acc → LLK-coupled → **DFB ptr/credit surgery (WEIRD-OK)** with implicit sync disabled on Quasar, or dest-only accumulation / disable `packer_l1_acc`.

## Recommended path

conv2d cannot be mechanically ported. Sequence: (1) clear the mechanical `fifo_num_pages` read; (2) redesign Class 2 window to scratchpad+sems; (3) decide Class 3 arch fork (strided DFB on Quasar vs ptr scatter on WH/BH); (4) keep Class 4/5 partials on documented DFB ptr save/restore (WEIRD-OK) or disable L1-acc for v1. Until (2)–(4) land, the op stays RED.
