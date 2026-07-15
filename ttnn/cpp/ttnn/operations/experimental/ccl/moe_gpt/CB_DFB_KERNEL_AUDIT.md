# CB→DFB Kernel Audit: `experimental/ccl/moe_gpt`

**Date:** 2026-07-15
**Op root:** `ttnn/cpp/ttnn/operations/experimental/ccl/moe_gpt/`

**Scope:** `device/kernels/dm0.cpp`, `dm1.cpp`, `combine_dm1.cpp`, `compute.cpp`, `tilize_reader.cpp`, `tilize_writer.cpp`, `tilize_compute.cpp`, `moe_gpt_ring_common.h`, `swiglu_sfpu.h`.

## Overall verdict: YELLOW

**Summary:** Sibling of `moe_compute` with the same MoE ring/tilize/SwiGLU structure. CBs are overwhelmingly **canonical Class 1 linear FIFOs** (weight/activation streaming, ready-flag handshakes, routing metadata) → mechanical `CircularBuffer` → `DataflowBuffer`. The **only** litmus offenders are two `get_tile_address(...)` scalar reads on control CBs — `cb_total_chunks` (`tilize_compute.cpp:44`) and `cb_w2c_md` (`compute.cpp:179`) — reading a chunk count / per-expert metadata pointer from L1 after a `wait_front`. These are **QUASAR-BLOCKED** (need the DFB `read_tile_value`/`get_tile_address` API on 2xx, in progress on Runtime), so **2xx is Blocked (runtime)** with **1xx clear** → op rollup **YELLOW**. **No** GATE hits, no silent-wrong hits, no pointer surgery.

## CB portability

| CB | Class | Kernel(s) | 1xx status | 1xx notes | 2xx status | 2xx notes |
|----|-------|-----------|------------|-----------|------------|-----------|
| `cb_total_chunks` | 6 | `tilize_compute.cpp`, `tilize_reader.cpp`, `tilize_writer.cpp` | Portable (workaround) | **undesirable but OK hack:** `get_tile_address(cb_total_chunks.get_cb_id(), 0)` scalar read of chunk count (`tilize_compute.cpp:44`) after `wait_front` — works as raw L1 read on 1xx; uplift: DFB `read_tile_value` | Blocked (runtime) | needs `read_tile_value`/`get_tile_address` on DFB (Runtime "in progress"); LLK/DFB id required, LTA insufficient |
| `cb_w2c_md` | 6 | `compute.cpp`, `dm1.cpp`, `combine_dm1.cpp` | Portable (workaround) | **undesirable but OK hack:** `get_tile_address(cb_w2c_md_id, 0)` reads per-expert token counts + chunk-ready semaphore address (`compute.cpp:179`) after `wait_front` | Blocked (runtime) | same — needs DFB read API on 2xx |
| `cb_r2c_w0_w1`, `cb_r2c_w2` | 1 | `dm0.cpp`, `compute.cpp` | Portable | weight input streams, canonical FIFO | Portable | — |
| `cb_s2c_in`, `cb_s2c_in2` | 1 | `dm1.cpp`, `compute.cpp` | Portable | activation input streams, linear FIFO | Portable | — |
| `cb_c2w_rdy`, `cb_w2c_rdy` | 1 | `dm1.cpp`, `combine_dm1.cpp`, `compute.cpp` | Portable | compute↔writer ready-flag handshakes, canonical FIFO | Portable | — |
| `cb_c2s_out` | 1 | `dm1.cpp`, `combine_dm1.cpp`, `compute.cpp` | Portable | compute → sender output stream | Portable | — |
| `cb_c2c_ones_tile` | 1 | `compute.cpp` | Portable | constant ones tile | Portable | — |
| `cb_tilize_input`, `cb_tilize_output` | 1 | `tilize_reader.cpp`, `tilize_writer.cpp`, `tilize_compute.cpp` | Portable | row-major → tilized stream; output single-buffered via canonical `reserve_back` | Portable | — |
| `cb_per_expert_total_tokens` | 1 | `tilize_reader.cpp`, `tilize_writer.cpp` | Portable | per-expert token-count transfer, linear FIFO | Portable | — |
| `cb_indices_tensor`, `cb_scores_tensor`, `cb_mapping_tensor` | 1 | `tilize_reader.cpp`, `tilize_writer.cpp` | Portable | routing metadata tensors, linear FIFO | Portable | — |
| `cb_e_t`, `cb_expert_activation`, `cb_brisc_e_t`, `cb_brisc_expert_counts`, `cb_brisc_expert_activation`, `cb_brisc_activated_count`, `cb_remote_counts` | 1 | `tilize_reader.cpp`, `tilize_writer.cpp` | Portable | expert-routing bookkeeping CBs, canonical FIFO | Portable | — |

## GATE hits (must be empty to merge)

- (none) — no `get_local_cb_interface(...).<field>` access anywhere in scope.

## Blocked on runtime (2xx rollup)

- `read_tile_value` / `get_tile_address` on `DataflowBuffer` (Quasar, **in progress**) → `cb_total_chunks` (`tilize_compute.cpp:44`) and `cb_w2c_md` (`compute.cpp:179`). 1xx clear; 2xx blocked until API lands.

## Recommended path

1. **1xx (WH/BH): GREEN-equivalent now.** Mechanical port; the two `get_tile_address` control-scalar reads work as raw L1 reads.
2. **2xx (Quasar): wait on Runtime API.** Reclassify `cb_total_chunks` / `cb_w2c_md` to Portable once `read_tile_value`/`get_tile_address` land on `DataflowBuffer`. Same QUASAR-BLOCKED family as `moe_compute` and sdpa control reads — one API unblocks both moe ops.
