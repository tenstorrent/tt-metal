# CB→DFB Kernel Audit: `nlp_create_qkv_heads_decode`

**Date:** 2026-07-15
**Op root:** `ttnn/cpp/ttnn/operations/experimental/transformer/nlp_create_qkv_heads_decode/`

**Scope:** `device/kernels/reader_tm_tile_layout_nlp_create_qkv_heads_decode.cpp` (sharded), `device/kernels/reader_interleaved_tm_tile_layout_nlp_create_qkv_heads_decode.cpp` (interleaved + DRAM-aligned staging), `device/kernels/reader_tm_tile_layout_nlp_create_qkv_heads_decode_on_subcoregrids.cpp`. Donor include: `ttnn/operations/data_movement/common/kernels/common.hpp` (`tt_memmove`; scanned — clean).

## Overall verdict: GREEN

**Summary:** Decode-mode QKV head split. Q/K/V output shards are filled by per-batch NoC scatter (`get_write_ptr() + offset`). The interleaved path adds `cb_aligned_scratch`, a private DRAM-alignment staging region used only via `get_write_ptr()` (round-up base + per-tile slots for the aligned NoC read → `tt_memmove` copy) — a sync-free scratch with no consumer. `cb_batch_offset` is a small index CB read via `get_write_ptr()` reinterpret. No `get_local_cb_interface`, no `fifo_*` field access, no `get_pointer_to_cb_data` — Step-4 litmus scans return **zero** hits.

## CB portability

| CB | Class | Kernel(s) | 1xx status | 1xx notes | 2xx status | 2xx notes |
|----|-------|-----------|------------|-----------|------------|-----------|
| `cb_id_q_out` / `cb_id_k_out` / `cb_id_v_out` | 1/3 | all three readers | Portable (workaround) | **undesirable but OK hack:** `get_write_ptr() + per-batch/per-head offset` scatter into reserved output shards; uplift: strided/multi-producer DFB on Quasar | Portable (workaround) | same |
| `cb_aligned_scratch` | 6 | `reader_interleaved_*_decode.cpp` | Portable | sync-free private DRAM-align staging (`get_write_ptr()` base + slots, no consumer) → `ScratchpadSpec` (autoportable) | Portable | same |
| `cb_batch_offset` | 6 | `reader_*_decode.cpp`, `reader_*_decode_on_subcoregrids.cpp` | Portable | sync-free index CB; `get_write_ptr()` reinterpret read (not `get_pointer_to_cb_data`) → `ScratchpadSpec`/LTA candidate | Portable | same |

## GATE hits (must be empty to merge)

- (none)

## Blocked on runtime (2xx rollup)

- (none)
