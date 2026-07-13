# CB→DFB Kernel Audit: `transformer` [all kernel-bearing ops]

**Date:** 2026-07-13
**Op root:** `ttnn/cpp/ttnn/operations/transformer/`

**Scope:** Combined device-kernel audit of the three transformer ops that own program factories + kernels:
- `gated_delta_attn` → `device/kernels/{compute,dataflow}/*`
- `sdpa_decode` → `device/kernels/{compute,dataflow}/*`
- `sdpa` (+ variants: `sparse_sdpa`, `sparse_sdpa_msa`, `joint`, `ring_joint`, `exp_ring_joint`, `ring_distributed`) → `device/kernels/{compute,dataflow,*.hpp}/*`

Composite ops `attention_softmax`, `concatenate_heads`, and `split_query_key_value_and_split_heads` have **no program factory / device kernels** (no `CreateKernel`) — **N/A** for this kernel audit; they inherit readiness from the ops they call.

---

## Overall verdict: RED

The `sdpa` family is **RED** due to a hard **GATE** in the shared streaming helper `compute_streaming.hpp::cb_push_back_hold_wr_ptr` (writes `fifo_wr_ptr`, reads `fifo_limit`/`fifo_size`/`fifo_page_size` via `get_local_cb_interface`). `sdpa_decode` is **YELLOW** (2xx `read_tile_value` runtime dependency, 1xx clear). `gated_delta_attn` is **GREEN** (clean linear-FIFO kernels).

Per-op rollup:

| Op | Rollup | Reason |
|----|--------|--------|
| `gated_delta_attn` | **GREEN** | No GATE / field / ptr / `read_tile_value` hits — all Class 1 |
| `sdpa_decode` | **YELLOW** | `read_tile_value` on ctrl CB → 2xx `Blocked (runtime)`, 1xx clear |
| `sdpa` (+ all variants) | **RED** | GATE `get_local_cb_interface(...).fifo_wr_ptr` write + ring-span field reads with no getter; plus 2xx `read_tile_value` on ctrl CBs |

---

## CB portability

| CB | Class | Kernel(s) | 1xx status | 1xx notes | 2xx status | 2xx notes |
|----|-------|-----------|------------|-----------|------------|-----------|
| `cb_nm_P_a/b`, `cb_nm_R_a`, `cb_L_unit`, `cb_L_inv_*`, `cb_L_inv_row_i`, `cb_S`, `cb_v_beta_sc`, `cb_v_cor`, `cb_k_bd_sc`, `cb_k_dt`, `cb_k_cum`, `cb_q_decay`, `cb_intra_att`, `cb_dl_exp` | 1 | `gated_delta_attn.cpp`, `reader_gated_delta_attn.cpp`, `writer_gated_delta_attn.cpp` | Portable | linear FIFO → `DataflowBuffer` (mechanical rename) | Portable | — |
| `cb_cur_pos` | 6 | `sdpa_flash_decode.cpp` | Portable (workaround) | **undesirable but OK hack:** mailbox `wait_front` + `read_tile_value` scalar read (issue #27979) | Blocked (runtime) | needs `read_tile_value` on DFB (in flight); LTA insufficient — LLK-coupled ctrl read |
| all other `sdpa_decode` reader/writer/compute CBs | 1 | `reader_decode_all.cpp`, `writer_decode_all.cpp`, `sdpa_flash_decode.cpp` | Portable | linear FIFO | Portable | — |
| `cb_qkt_im` / `cb_qk_im` | 4 | `compute_streaming.hpp`, `sdpa.cpp`, `sparse_sdpa_compute.cpp`, `sparse_sdpa_msa_compute.cpp` (+ `compute_common.hpp`, `exp_ring_joint_sdpa.cpp`, `ring_joint_sdpa.cpp`, `joint_sdpa.cpp`) | **Blocked** | GATE: `cb_push_back_hold_wr_ptr` writes `get_local_cb_interface(cb).fifo_wr_ptr` and reads `fifo_limit`/`fifo_size` (no getter). Resolve before port | **Blocked** | same GATE |
| `cb_ctrl` | 6 | `sparse_sdpa_compute.cpp`, `sparse_sdpa_msa_compute.cpp`, `sparse_sdpa_reader.cpp`, `sparse_sdpa_msa_reader.cpp` | Portable (workaround) | **undesirable but OK hack:** `read_tile_value` scalar ctrl read (mailbox-distributed UNPACK) | Blocked (runtime) | needs `read_tile_value` on DFB (in flight) |
| `cb_chunk_start_idx` | 6 | `sdpa.cpp`, `writer_interleaved.cpp` | Portable (workaround) | **undesirable but OK hack:** `read_tile_value` scalar read | Blocked (runtime) | needs `read_tile_value` on DFB (in flight) |
| all other `sdpa` reader/writer/compute CBs (Q/K/V/mask/out streams, joint/ring dataflow) | 1 | `reader_interleaved.cpp`, `writer_interleaved.cpp`, `joint_*`, `ring_joint_*`, `exp_ring_joint_*`, `sparse_sdpa_*` | Portable | linear FIFO | Portable | — |

---

## GATE hits (must be empty to merge)

- `sdpa/device/kernels/compute/compute_streaming.hpp:84` — `get_local_cb_interface(cb_id)` grabbed, then:
  - `:85` `intf.fifo_wr_ptr -= …` — **write** to `fifo_wr_ptr` (hard GATE)
  - `:85` read `intf.fifo_page_size` — mechanical, → `get_entry_size()`
  - `:86` read `intf.fifo_limit`, `intf.fifo_size` — **no getter today** (ring-span; file to Almeet)
  - `:88` `intf.fifo_wr_ptr += intf.fifo_size` — write + read

  Helper `cb_push_back_hold_wr_ptr` is Class 4 (credit/address decoupling: `push_back` posts credits, then wr_ptr is rewound so `pack_tile<true>` offsets stay relative to a stable base). Reached by `sdpa.cpp`, `sparse_sdpa_compute.cpp`, `sparse_sdpa_msa_compute.cpp` directly and pulled into the ring/joint compute kernels via `compute_common.hpp`.

**Resolution path (before `sdpa` can port):**
1. Rewrite the wr_ptr manipulation off `get_local_cb_interface` — use `get_write_ptr()` + a sanctioned DFB write-pointer setter (Class 4 → *Portable (workaround)*, `WEIRD-OK`, disable implicit sync on Quasar).
2. Replace `fifo_page_size` with `get_entry_size()`.
3. **File issue to Almeet** for ring-span getters (`fifo_limit` / `fifo_size` → `get_total_buffer_size_bytes()` / ring-span) — the wrap math has no getter today.

## Blocked on runtime (2xx rollup)

- `read_tile_value` on DFB (Runtime team, in progress) → unblocks: `cb_cur_pos` (`sdpa_decode`), `cb_ctrl` (`sparse_sdpa`, `sparse_sdpa_msa`), `cb_chunk_start_idx` (`sdpa`). 1xx paths already clear via mailbox `read_tile_value`.
