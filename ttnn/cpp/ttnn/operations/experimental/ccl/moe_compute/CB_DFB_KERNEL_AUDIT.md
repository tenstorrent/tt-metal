# CB→DFB Kernel Audit: `experimental/ccl/moe_compute`

**Date:** 2026-07-15
**Op root:** `ttnn/cpp/ttnn/operations/experimental/ccl/moe_compute/`

**Scope:** `device/kernels/dm0.cpp`, `dm1.cpp`, `compute.cpp`, `tilize_reader.cpp`, `tilize_writer.cpp`, `tilize_compute.cpp`, `moe_ring_common.h`.

## Overall verdict: YELLOW

**Summary:** The MoE compute pipeline is a large multi-kernel op (reader/writer/compute + tilize reader/writer/compute) whose CBs are overwhelmingly **canonical Class 1 linear FIFOs** (activation/weight streaming, ready-flag handshakes, per-expert metadata transfer) — all mechanically `CircularBuffer` → `DataflowBuffer`. The **only** litmus offenders are two `get_tile_address(...)` scalar reads on control CBs — `cb_total_chunks` (`tilize_compute.cpp:102`) and `cb_w2c_md` (`compute.cpp:285`) — used to read a chunk count / metadata pointer out of L1. These are **QUASAR-BLOCKED** (need the `read_tile_value`/`get_tile_address` DFB read API on 2xx, in progress on the Runtime team), so **2xx is Blocked (runtime)** while **1xx is clear** → op rollup **YELLOW**. There are **no** GATE (`get_local_cb_interface(...).<field>`) hits, no silent-wrong hits, and no pointer surgery. The one `fifo_rd_ptr` occurrence (`tilize_compute.cpp:114`, plus the comment at `:14`) is a **commented-out DPRINT debug block** — not a live hit.

## CB portability

| CB | Class | Kernel(s) | 1xx status | 1xx notes | 2xx status | 2xx notes |
|----|-------|-----------|------------|-----------|------------|-----------|
| `cb_total_chunks` | 6 | `tilize_compute.cpp`, `tilize_reader.cpp`, `tilize_writer.cpp` | Portable (workaround) | **undesirable but OK hack:** `get_tile_address(total_chunks_cb_id, 0)` scalar read of chunk count (`tilize_compute.cpp:102`) after `wait_front` — L1 ptr read works on 1xx; uplift: DFB `read_tile_value` when API lands | Blocked (runtime) | needs `read_tile_value`/`get_tile_address` on DFB (Runtime "in progress") — LLK/DFB id required, LTA insufficient |
| `cb_w2c_md` | 6 | `compute.cpp`, `dm1.cpp` | Portable (workaround) | **undesirable but OK hack:** `get_tile_address(cb_w2c_md_id, 0)` reads per-expert token counts + semaphore addresses (`compute.cpp:285`) after `wait_front`/`pop_front` | Blocked (runtime) | same — needs DFB read API on 2xx |
| `cb_r2c_w0_w1`, `cb_r2c_w2` | 1 | `dm0.cpp`, `compute.cpp` | Portable | weight input streams, canonical reserve/push → wait/pop | Portable | — |
| `cb_s2c_in`, `cb_s2c_in2` | 1 | `dm1.cpp`, `compute.cpp` | Portable | activation input streams, linear FIFO | Portable | — |
| `cb_c2w_rdy`, `cb_w2c_rdy` | 1 | `dm1.cpp`, `compute.cpp` | Portable | compute↔writer ready-flag handshake CBs, canonical FIFO | Portable | — |
| `cb_c2s_out` | 1 | `dm1.cpp`, `compute.cpp` | Portable | compute → sender output stream | Portable | — |
| `cb_c2c_ones_tile` | 1 | `compute.cpp` | Portable | constant ones tile, linear FIFO | Portable | — |
| `cb_tilize_input`, `cb_tilize_output` | 1 | `tilize_reader.cpp`, `tilize_writer.cpp`, `tilize_compute.cpp` | Portable | row-major → tilized stream; output single-buffered via `reserve_back(all pages)` (canonical, no ptr surgery) | Portable | — |
| `cb_per_expert_total_tokens` | 1 | `dm0.cpp`, `dm1.cpp`, `tilize_*` | Portable | per-expert token-count transfer, linear FIFO | Portable | — |
| `cb_indices_tensor`, `cb_scores_tensor`, `cb_mapping_tensor` | 1 | `tilize_reader.cpp`, `tilize_writer.cpp` | Portable | routing metadata tensors, linear FIFO | Portable | — |
| `cb_e_t`, `cb_expert_activation`, `cb_brisc_e_t`, `cb_brisc_expert_counts`, `cb_brisc_expert_activation`, `cb_brisc_activated_count`, `cb_remote_counts` | 1 | `tilize_reader.cpp`, `tilize_writer.cpp` | Portable | expert-routing bookkeeping CBs, canonical FIFO | Portable | — |

## GATE hits (must be empty to merge)

- (none) — no `get_local_cb_interface(...).<field>` access. `tilize_compute.cpp:114` `fifo_rd_ptr` is inside a commented-out DPRINT debug block (see `:110-118`); not a live hit.

## Blocked on runtime (2xx rollup)

- `read_tile_value` / `get_tile_address` on `DataflowBuffer` (Quasar, **in progress** on Runtime team) → affects `cb_total_chunks` (`tilize_compute.cpp:102`) and `cb_w2c_md` (`compute.cpp:285`). 1xx path is clear (raw L1 ptr read); 2xx blocked until the API lands.

## Recommended path

1. **1xx (WH/BH): GREEN-equivalent now.** All CBs port mechanically; the two `get_tile_address` scalar reads work as raw L1 pointer reads on Gen1. Document them as **Portable (workaround)**.
2. **2xx (Quasar): wait on Runtime API.** Reclassify `cb_total_chunks` and `cb_w2c_md` to Portable once `read_tile_value`/`get_tile_address` land on `DataflowBuffer`. These are control-scalar reads (chunk counts, semaphore addresses), not FIFO data — same family as the sdpa/moe_gpt QUASAR-BLOCKED reads.
