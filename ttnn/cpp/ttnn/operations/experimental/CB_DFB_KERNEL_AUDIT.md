# CB→DFB Kernel Audit: `experimental/` (survey rollup)

**Date:** 2026-07-13
**Op root:** `ttnn/cpp/ttnn/operations/experimental/`
**Method:** Triage pass — [`CB_DFB_Check.md`](../../../../../CB_DFB_Check.md) Step-4 litmus scans across all **600** kernel files (`**/kernels/**/*.{cpp,hpp,h}`) in 33 op families. This survey finds hard blockers (GATE / QUASAR-BLOCKED / NEEDS-FIX); it is **not** a full per-op Step-5 buffer classification. Per-op CB tables exist for the two RED ops (see linked docs).

## Overall verdict: RED (2 ops), YELLOW (cluster), GREEN (remainder)

**Summary:** Most experimental ops are safe to port. Only **`quasar/conv2d`** and **`ccl/ring_attention_all_gather_async`** carry hard GATEs (field *writes* / no-getter field reads). A cluster of ops need mechanical getter swaps, one LTA migration, and two `moe` ops wait on the Quasar `read_tile_value` / `get_tile_address` runtime API. All other families show zero illegal CB patterns.

---

## Per-op rollup

### RED — not safe to port (structural GATE)

| Op | Rollup | Blocker | Class | Detail |
|----|--------|---------|-------|--------|
| `quasar/conv2d` | **RED** | 42 `fifo_wr_ptr`/`fifo_rd_ptr` **writes** via `get_local_cb_interface(...)` (activation-reuse window, split-reader scatter, held-base restore, `matmul_partials_cb` L1-acc save/restore) | 2+3+4+5 | [`quasar/conv2d/CB_DFB_KERNEL_AUDIT.md`](quasar/conv2d/CB_DFB_KERNEL_AUDIT.md) |
| `ccl/ring_attention_all_gather_async` | **RED** | `fifo_limit` + `fifo_size` **reads** — **no DFB getter exists** (manual ring wrap); Class 4 staging | 4 | [`ccl/ring_attention_all_gather_async/CB_DFB_KERNEL_AUDIT.md`](ccl/ring_attention_all_gather_async/CB_DFB_KERNEL_AUDIT.md) |

### YELLOW — safe after a mechanical fix

| Op | Fix | Kernel(s) / field | Status |
|----|-----|-------------------|--------|
| `quasar/untilize` | `fifo_page_size` → `get_entry_size()` | `reader_unary_interleaved_start_id{,_metal2}.cpp` | Portable after swap |
| `quasar/untilize_with_unpadding` | `fifo_page_size` → `get_entry_size()` | `reader_unary_interleaved_start_id.cpp:16` | Portable after swap |
| `quasar/tilize` | `fifo_page_size` → `get_entry_size()` | `dataflow/writer_unary_interleaved_start_id.cpp:16` | Portable after swap |
| `quasar/tilize_with_val_padding` | `fifo_page_size` → `get_entry_size()` | `writer_unary_interleaved_start_id{,_metal2}.cpp` | Portable after swap |
| `quasar/typecast` | `fifo_page_size` → `get_entry_size()` | reader + writer `unary_interleaved_start_id.cpp` | Portable after swap |
| `quasar/matmul` | `fifo_page_size` → `get_entry_size()` | `dataflow/writer_unary_interleaved_start_id.cpp:18` | Portable after swap |
| `quasar/reduction/generic` | `fifo_page_size` → `get_entry_size()` | `reader_unary_reduce_rm.cpp`, writer `_start_id{,_metal2}.cpp` | Portable after swap |
| `quasar/interleaved_to_sharded` | `fifo_page_size` → `get_entry_size()` | `reader_unary_stick_layout_...start_id.cpp:54` (scratch CB) | Portable after swap |
| `quasar/pool_generic` | `fifo_num_pages` → `get_total_num_entries()` (PR #49197); `fifo_page_size` → `get_entry_size()` | `pool_kernels_common.hpp:47,48,76` | Portable after swap |
| `padded_slice` | `fifo_page_size` → `get_entry_size()` (one hit is a `DPRINT` debug line) | `padded_slice_reader_rm_interleaved_start_id.cpp:85,97` | Portable after swap |
| `deepseek_prefill/update_padded_kv_cache` | `fifo_page_size` → `get_entry_size()` | `writer_update_padded_kv_cache.cpp:66` | Portable after swap |
| `transformer/dit_layernorm_pre_all_gather` | `get_pointer_to_cb_data` → **LocalTensorAccessor** (Welford reciprocal LUT) | `compute/layernorm_pre_allgather_welford.cpp:40` | **Portable (prereq: LTA)** |
| `deepseek_prefill/unified_routed_expert_ffn` (fused_swiglu) | `fifo_rd_ptr << 4` mailbox read → `get_read_ptr()` (1xx) / `read_tile_value` (2xx) | `compute/fused_swiglu.cpp:609-610` | 1xx Portable after swap; **2xx Blocked (runtime)** |

### YELLOW (2xx only) — QUASAR-BLOCKED on runtime API

`read_tile_value` / `get_tile_address` on DFB is **in progress** (Runtime team). Portable on 1xx via documented ptr hack; blocked on 2xx until API lands.

| Op | Field | Kernel(s) |
|----|-------|-----------|
| `ccl/moe_compute` | `get_tile_address` | `tilize_compute.cpp:102`, `compute.cpp:285` |
| `ccl/moe_gpt` | `get_tile_address` | `tilize_compute.cpp:44`, `compute.cpp:179` |

### OUT-OF-SCOPE — track separately (do not gate other ports)

Per [`CB_DFB_Check.md` § Audit scope](../../../../../CB_DFB_Check.md) path-exclusions (firmware-style reconfig / MOE routing / DeepSeek prefill dispatch pipeline):

| Op | Reason | Signals seen |
|----|--------|--------------|
| `deepseek_prefill/combine` | DeepSeek prefill combine | `read_tile_value` (`untilize_combine.cpp:91`) |
| `deepseek_prefill/dispatch` | DeepSeek prefill dispatch | `read_tile_value` (`untilize_dispatch.cpp:54`) |
| `deepseek_prefill/post_combine_reduce` | DeepSeek prefill post-combine-reduce | `read_tile_value_uint16` + `fifo_rd_ptr`/`fifo_page_size` reads |

### GREEN — no illegal CB patterns found (safe to port)

Zero litmus hits (no GATE, no silent-wrong, no `read_tile_value`/`get_tile_address`/`get_pointer_to_cb_data`, no ptr surgery). Mechanical `CircularBuffer` → `DataflowBuffer`:

`adaptive_pool`, `bcast_to`, `cnn`, `conv3d`, `copy`, `core_subset_write`, `dropout`, `fusion`, `indexer_score`, `isin`, `minimal_matmul`, `multi_scale_deformable_attn`, `paged_cache`, `plusone`, `reshape`, `slice_write`, `ssm`, `tensor_prefetcher`, `topk_large_indices`, `topk_router_gpt`, `unary_backward`, `reduction` (non-quasar), `deepseek`, `deepseek_moe_post_combine_tilize`, `test`, and all remaining `quasar/*`, `ccl/*`, and `transformer/*` sub-ops not listed above.

> **Caveat:** GREEN = "no hard blockers in the scan." A full per-op Step-5 buffer inventory is still needed before merge to confirm each CB is Class 1, but none of these carry a GATE, silent-wrong, or runtime blocker.

---

## GATE hits (must be empty to merge)

**Mechanical (existing getter — trivial swap clears GATE):**

- `quasar/untilize/.../reader_unary_interleaved_start_id.cpp:21`, `..._metal2.cpp:22` — `fifo_page_size` → `get_entry_size()`
- `quasar/untilize_with_unpadding/.../reader_unary_interleaved_start_id.cpp:16` — `fifo_page_size`
- `quasar/tilize/.../writer_unary_interleaved_start_id.cpp:16` — `fifo_page_size`
- `quasar/tilize_with_val_padding/.../writer_unary_interleaved_start_id{,_metal2}.cpp:20,22` — `fifo_page_size`
- `quasar/typecast/.../{reader,writer}_unary_interleaved_start_id.cpp:20,21` — `fifo_page_size`
- `quasar/matmul/.../writer_unary_interleaved_start_id.cpp:18` — `fifo_page_size`
- `quasar/reduction/generic/.../reader_unary_reduce_rm.cpp:83`, `writer_unary_interleaved_start_id{,_metal2}.cpp:20,22` — `fifo_page_size`
- `quasar/interleaved_to_sharded/.../reader_unary_stick_layout_sharded_blocks_interleaved_start_id.cpp:54` — `fifo_page_size`
- `quasar/pool_generic/.../pool_kernels_common.hpp:47,48,76` — `fifo_num_pages` / `fifo_page_size`
- `padded_slice/.../padded_slice_reader_rm_interleaved_start_id.cpp:85,97` — `fifo_page_size` (`:85` is inside a `DPRINT`)
- `deepseek_prefill/update_padded_kv_cache/.../writer_update_padded_kv_cache.cpp:66` — `fifo_page_size`

**Mailbox scalar read (fix or `read_tile_value`):**

- `deepseek_prefill/unified_routed_expert_ffn/.../fused_swiglu.cpp:609,610` — `fifo_rd_ptr << 4` L1 address

**Structural — field WRITES / no-getter reads (RED, redesign required):**

- `quasar/conv2d/*` — 42 `fifo_wr_ptr`/`fifo_rd_ptr` writes (see per-op doc)
- `ccl/ring_attention_all_gather_async/.../ring_attention_all_gather_reader.cpp:148,149` — `fifo_limit`/`fifo_size` reads, **no getter**

## Blocked on runtime (2xx rollup)

- `read_tile_value` / `get_tile_address` on Quasar DFB (**in progress**) — `ccl/moe_compute`, `ccl/moe_gpt`, `deepseek_prefill/unified_routed_expert_ffn` (2xx path only).
- `get_total_buffer_size_bytes()` / ring-span getters (**needed — file to Almeet**) — `ccl/ring_attention_all_gather_async` `fifo_size`/`fifo_limit`.

## Notes / false positives ruled out

- `ccl/moe_compute/.../tilize_compute.cpp:114` `fifo_rd_ptr` hit is inside a **commented-out** DEBUG block — not a live access.
- `moe_compute`/`moe_gpt`/`post_combine_reduce`/`combine`/`dispatch` matches on `//` lines are comments referencing `read_tile_value`.
- `quasar/conv2d` ships both a legacy (`conv_bmm_tilize.cpp`) and a `_metal2.cpp` variant; the metal2 variant already renames CBs to `dfb::*` but **still performs the field writes** — the port attempt exists but the GATE is unresolved.
