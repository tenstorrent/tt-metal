# CB→DFB Kernel Audit: `experimental/` (master rollup)

**Date:** 2026-07-15
**Op root:** `ttnn/cpp/ttnn/operations/experimental/`
**Method:** Full per-op audit. Every sub-op (dir containing `device/`) was audited per the [`CB_DFB_Check.md`](../../../../../CB_DFB_Check.md) 5-step recipe — scope discovery, Step-4 litmus scans on each op's kernel closure (incl. transitive `#include` and cross-op donor kernels), CB classification, and a per-op `CB_DFB_KERNEL_AUDIT.md`. This file is the index; **each op root has its own detailed report.**

## Overall verdict: RED (4 ops), YELLOW (19), GREEN (96), OUT-OF-SCOPE (5)

**Summary:** The experimental tree is **overwhelmingly portable**. Of 124 audited ops, **96 are GREEN** (canonical Class 1 FIFO / bare-pointer L1 addressing — mechanical `CircularBuffer` → `DataflowBuffer` rename), **19 are YELLOW** (mostly one-line `fifo_page_size` → `get_entry_size()` getter swaps; plus 1 LTA prereq and 3 QUASAR-BLOCKED 2xx-only), and **only 4 are RED**. All 4 REDs are the **same anti-pattern**: a CB used as a hand-indexed ring/window via `fifo_wr_ptr`/`fifo_rd_ptr` **writes** with wrap math from no-getter `fifo_size`/`fifo_limit` reads. **5 ops are OUT-OF-SCOPE** (moe-gate firmware reconfig + DeepSeek-prefill routing).

| Verdict | Count | Ops |
|---------|-------|-----|
| 🔴 RED | 4 | `quasar/conv2d`, `quasar/matmul` (gathered), `ccl/ring_attention_all_gather_async`, `ccl/llama_all_gather_matmul_async` (gathered) |
| 🟡 YELLOW | 19 | see table below |
| 🟢 GREEN | 96 | all others |
| ⬜ OUT-OF-SCOPE | 5 | `deepseek/moe/deepseek_moe_gate`, `deepseek/moe/generalized_moe_gate`, `deepseek_prefill/{combine,dispatch,post_combine_reduce}` |

---

## RED — not safe to port (structural GATE)

| Op | Driver | Class | Detail |
|----|--------|-------|--------|
| `quasar/conv2d` | 42 `fifo_wr_ptr`/`fifo_rd_ptr` writes (window/scatter/held-base/partials) | 2+3+4+5 | [doc](quasar/conv2d/CB_DFB_KERNEL_AUDIT.md) |
| `quasar/matmul` | gathered variant `bmm_*_gathered.cpp` `cb_in1` ring window: `fifo_rd_ptr` writes + no-getter `fifo_size`/`fifo_limit` reads | 2+4 | [doc](quasar/matmul/CB_DFB_KERNEL_AUDIT.md) |
| `ccl/ring_attention_all_gather_async` | reader manual ring wrap: no-getter `fifo_limit`/`fifo_size` reads | 4 | [doc](ccl/ring_attention_all_gather_async/CB_DFB_KERNEL_AUDIT.md) |
| `ccl/llama_all_gather_matmul_async` | reuses gathered-matmul ring window (`fifo_rd_ptr` writes + no-getter reads) | 2+4 | [doc](ccl/llama_all_gather_matmul_async/CB_DFB_KERNEL_AUDIT.md) |

All four share **one fix**: replace the manual ring/window with **scratchpad + semaphores** (or strided/multi-producer DFB on Quasar), and file the Almeet ring-span getter (`get_total_buffer_size_bytes()` / `fifo_size`/`fifo_limit`) for the read side.

## YELLOW — safe after a mechanical fix

| Op | Fix | Kind |
|----|-----|------|
| `quasar/interleaved_to_sharded` | `fifo_page_size` → `get_entry_size()` (scratch `dfb::in1`) | getter swap |
| `quasar/pool_generic` | `fifo_num_pages` → `get_total_num_entries()` (PR #49197); `fifo_page_size` → `get_entry_size()` | getter swap |
| `quasar/reduction/generic` | `fifo_page_size` → `get_entry_size()` (reader_rm/writer) | getter swap |
| `quasar/tilize` | `fifo_page_size` → `get_entry_size()` (writer) | getter swap |
| `quasar/tilize_with_val_padding` | `fifo_page_size` → `get_entry_size()` (writer{,_metal2}) | getter swap |
| `quasar/typecast` | `fifo_page_size` → `get_entry_size()` (reader+writer) | getter swap |
| `quasar/untilize` | `fifo_page_size` → `get_entry_size()` (reader{,_metal2}) | getter swap |
| `quasar/untilize_with_unpadding` | `fifo_page_size` → `get_entry_size()` (reader) | getter swap |
| `matmul/attn_matmul` | `fifo_page_size` → `get_entry_size()` (shared `writer_unary_interleaved_start_id.cpp:19` donor) | getter swap |
| `unary_backward/gelu_backward` | same shared donor writer as attn_matmul | getter swap |
| `padded_slice` | `fifo_page_size` → `get_entry_size()` (`:97` live; `:85` is a DEBUG DPRINT) | getter swap |
| `deepseek_prefill/update_padded_kv_cache` | `fifo_page_size` → `get_entry_size()` (writer:66) | getter swap |
| `ccl/all_gather_matmul_async` | inherits standard `matmul` mcast getter swaps (no own kernels) | getter swap (donor) |
| `ccl/llama_reduce_scatter_matmul` | inherits standard `matmul` mcast_1d getter swaps (no own kernels) | getter swap (donor) |
| `ccl/matmul_reduce_scatter_async` | inherits standard `matmul` mcast_2d getter swaps (no own kernels) | getter swap (donor) |
| `transformer/dit_layernorm_pre_all_gather` | `get_pointer_to_cb_data` (Welford recip LUT via `memory.h`) → **LocalTensorAccessor** | **LTA prereq** |
| `deepseek_prefill/unified_routed_expert_ffn` | `fifo_rd_ptr<<4` mailbox read → `get_read_ptr()` (1xx); `read_tile_value` (2xx) | fix + **2xx runtime** |
| `ccl/moe_compute` | `get_tile_address` control-scalar reads | **2xx runtime** |
| `ccl/moe_gpt` | `get_tile_address` control-scalar reads | **2xx runtime** |

## OUT-OF-SCOPE — track separately (do not gate other ports)

| Op | Reason |
|----|--------|
| `deepseek/moe/deepseek_moe_gate` | `reconfig_cbs_for_mask` full `LocalCBInterface` rewrite + `get_cb_tiles_*_ptr` writes (SILENT-WRONG, firmware reinit) |
| `deepseek/moe/generalized_moe_gate` | same firmware-style reconfig in `unified_kernels/kernel_utils.hpp` |
| `deepseek_prefill/combine` | DeepSeek-prefill routing — `read_tile_value` |
| `deepseek_prefill/dispatch` | DeepSeek-prefill routing — `read_tile_value` |
| `deepseek_prefill/post_combine_reduce` | DeepSeek-prefill routing — `read_tile_value_uint16` + `fifo_rd_ptr`/`fifo_page_size` reads |

## GREEN — safe to port (96 ops)

No GATE / silent-wrong / ptr-surgery / runtime blocker. Notable families entirely GREEN: **transformer** (all 22 except `dit_layernorm_pre_all_gather`), **ccl** send/recv/reduce-scatter/all-gather dataflow (except the 2 REDs + 2 moe YELLOWs + 3 composite YELLOWs), **deepseek_prefill** in-scope ops (10/13), **reduction/ssm/cnn/conv3d/dropout/fusion/bcast_to** (all 14), and most `quasar` structural ops (`binary`, `binary_ng`, `fold`, `halo`, `move`, `pad`, `reshape_view`, `reshard`, `sharded_to_interleaved`, `slice`, `transpose`). See each op's own `CB_DFB_KERNEL_AUDIT.md` for its CB table.

> **Caveat:** GREEN = "no hard blockers in the scan." A per-op Step-5 buffer inventory (in each doc) confirms Class-1 status; none carry a GATE, silent-wrong, or runtime blocker.

---

## Cross-cutting conclusions

1. **The RED problem is singular and shared.** `conv2d`, `matmul`-gathered, `ring_attention`, and `llama_all_gather_matmul_async` are the same manual-ring/window pointer-surgery bug (Patterns A+B in [`DFB_BLOCKED_SUMMARY.md`](../../../../../DFB_BLOCKED_SUMMARY.md)). One reusable **scratchpad + semaphores** ring pattern + one **Almeet ring-span getter** issue clears all four.
2. **YELLOW is almost entirely one line.** 16 of 19 YELLOWs are `fifo_page_size` → `get_entry_size()` (or `fifo_num_pages` → `get_total_num_entries()`, PR #49197) mechanical swaps. `attn_matmul` + `gelu_backward` share a single donor writer line. The remaining 3 are the 2xx-runtime `read_tile_value`/`get_tile_address` cluster (`moe_compute`, `moe_gpt`, `unified_routed_expert_ffn`) plus the single LTA prereq (`dit_layernorm_pre_all_gather`).
3. **The vast majority (96/124) is a no-op rename.** Most experimental kernels already use the new `CircularBuffer` object API with canonical `reserve_back`/`push_back`/`wait_front`/`pop_front` and bare `get_read_ptr()`/`get_write_ptr()` L1/NoC addressing.

## GATE hits requiring redesign (RED)

- `quasar/conv2d/*` — 42 `fifo_wr_ptr`/`fifo_rd_ptr` writes (see op doc).
- `quasar/matmul` `bmm_large_block_zm_fused_bias_activation_gathered.cpp` — `cb_in1` `fifo_rd_ptr` writes + `fifo_size`/`fifo_limit` reads.
- `ccl/ring_attention_all_gather_async/.../ring_attention_all_gather_reader.cpp:148,149` — `fifo_limit`/`fifo_size` reads (no getter).
- `ccl/llama_all_gather_matmul_async` — gathered-matmul ring window (same as `quasar/matmul`).

## Blocked on runtime (2xx rollup)

- `read_tile_value` / `get_tile_address` on Quasar DFB (**in progress**) — `ccl/moe_compute`, `ccl/moe_gpt`, `deepseek_prefill/unified_routed_expert_ffn` (2xx only).
- `get_total_buffer_size_bytes()` / ring-span getters (**needed — file to Almeet**) — the 4 RED ring/window ops.
