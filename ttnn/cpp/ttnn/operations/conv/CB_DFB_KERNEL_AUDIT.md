# CB→DFB Kernel Audit: `conv` (conv2d)

**Date:** 2026-07-15
**Op root:** `ttnn/cpp/ttnn/operations/conv/conv2d/`

**Scope:** conv2d multi-core factories → kernels: `conv_bmm_tilize.cpp`, `conv_reader_common.hpp`, `reader_conv_activations_padded_with_halo_3x3_weights_v2.cpp`, `reader_conv_activations_2d_mcast_padded_with_halo_3x3_weights_v2.cpp`, `reader_writer_tiled_out_1d_mcast_{sender,receiver}_conv_weights_tiled_col_to_rm_blocks.cpp`, `writer_tiled_out_2d_mcast_{sender,receiver}_*.cpp`, `activation_reader_width_sharded.cpp`, `weights_reader_width_sharded.cpp`, `{reader,compute}_depthwise_conv1d.cpp`.

**Note:** `conv1d` and `conv_transpose2d` are host-only wrappers with **no device kernels** — they reuse these conv2d kernels, so this report covers all three.

## Overall verdict: RED

**Summary:** conv2d is the guide's documented "worst in-scope offender." The fused compute kernel `conv_bmm_tilize.cpp` plus the halo/split readers perform extensive `get_local_cb_interface(...).fifo_wr_ptr` / `.fifo_rd_ptr` **writes** (pointer surgery) across Classes 2 (activation-reuse window), 3 (split-reader scatter tilize), 4 (held-base restore before `push_back`), and 5 (partials L1 accumulator save/restore). These are field *writes* → NEEDS-DESIGN-DECISION, not mechanical getter swaps. Hard-blocked until redesigned. The pure input/output/weight/bias FIFOs are Class 1 portable.

## CB portability

| CB | Class | Kernel(s) | 1xx status | 1xx notes | 2xx status | 2xx notes |
|----|-------|-----------|------------|-----------|------------|-----------|
| `in0_cb` (activation) | 2 | `conv_bmm_tilize.cpp`, `conv_reader_common.hpp`, `reader_conv_activations_*halo*.cpp` | Blocked | GATE: `.fifo_rd_ptr` read (`conv_bmm_tilize.cpp:265`), `.fifo_wr_ptr` writes (`conv_reader_common.hpp:91,109`; reader:89). Window reuse → **scratchpad+sems** (uplift) | Blocked | same; Class 2 window → scratchpad+sems preferred |
| `in0_cb_second_reader` | 2/3 | `conv_bmm_tilize.cpp`, `reader_writer_tiled_out_1d_mcast_{sender,receiver}_*.cpp` | Blocked | GATE: `.fifo_rd_ptr` read (`:271`), `.fifo_wr_ptr` writes (sender:155, receiver:113). Split-reader scatter | Blocked | strided multi-producer DFB (NEEDS-DESIGN) |
| `tilized_in0_cb` | 3 | `conv_bmm_tilize.cpp` | Blocked | GATE: `.fifo_wr_ptr` scatter writes (`:269,364`). Arch fork: 1xx ptr scatter (WEIRD-OK) **after** removing field access; 2xx strided DFB | Blocked | strided DFB vs multi-DFB vs disable split reader (NEEDS-DESIGN) |
| `matmul_partials_cb` | 5 | `conv_bmm_tilize.cpp` | Blocked | GATE: `.fifo_rd_ptr`/`.fifo_wr_ptr` save/restore throughout inner dim (`:298,299,310,311,502–615`). In-place L1 acc | Blocked | DFB ptr save/restore (WEIRD-OK) once field access removed, or disable `packer_l1_acc`; Quasar SELF-LOOP-CANDIDATE if PACK→UNPACK fits |
| generic size read | 1 | `conv_reader_common.hpp` | Blocked | GATE: `.fifo_num_pages` read (`:25`) → `get_total_num_entries()` (mechanical) | Blocked | same |
| `out_cb` | 1 | `conv_bmm_tilize.cpp`, `writer_tiled_out_2d_mcast_*.cpp` | Portable | linear pack → output FIFO | Portable | — |
| `in1_cb` (weights) | 1 | `conv_bmm_tilize.cpp`, `weights_reader_width_sharded.cpp`, weight readers | Portable | linear FIFO | Portable | — |
| `in0_pretilize_cb` | 1 | `conv_bmm_tilize.cpp` | Portable | linear FIFO | Portable | — |
| `bias_cb` | 1 | `conv_bmm_tilize.cpp` | Portable | linear FIFO | Portable | — |
| depthwise conv1d CBs | 1 | `reader_depthwise_conv1d.cpp`, `compute_depthwise_conv1d.cpp` | Portable | no field access in scan | Portable | — |

## GATE hits (must be empty to merge)

- `conv_bmm_tilize.cpp:72,78,112,164,265,269,271,298,299,310,311,364,502,503,514,515,526,527,540,541,545,548,558,614,615` — `get_local_cb_interface(...).fifo_{rd,wr}_ptr` reads/writes (partials save/restore, activation-reuse, scatter tilize)
- `conv_reader_common.hpp:25` — `.fifo_num_pages` read → `get_total_num_entries()`; `:91,109` — `.fifo_wr_ptr` writes (window reset)
- `reader_conv_activations_padded_with_halo_3x3_weights_v2.cpp:89` — `.fifo_wr_ptr` write
- `reader_writer_tiled_out_1d_mcast_sender_conv_weights_tiled_col_to_rm_blocks.cpp:155` — `.fifo_wr_ptr` write
- `reader_writer_tiled_out_1d_mcast_receiver_conv_weights_tiled_col_to_rm_blocks.cpp:113` — `.fifo_wr_ptr` write

## Blocked on runtime (2xx rollup)

- Class 3 split-reader scatter and Class 5 partials require design decisions (strided DFB / scratchpad+sems / disable opt) on 2xx after the GATE field access is removed.
