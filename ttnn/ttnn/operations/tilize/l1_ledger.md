# L1 Ledger: tilize

Schema: `.claude/references/l1-footprint-discipline.md`. Block axes (from `op_design.md`): `rows`, `cols`, `stick_in_tile`.

| CB | Capacity (pages) | Live set | Axis accounting | Page format | Producer | Consumer | Lifetime | Shares with / why not |
|----|------------------|----------|-----------------|-------------|----------|----------|----------|-----------------------|
| `cb_rm_sticks` | `in_depth_tile_rows · block_width_tiles`, page = `tile_h · 32 · in_elem_bytes` | `block_width_tiles` pages (one tile-row being unpacked) | `{rows: streams → window of in_depth_tile_rows tile-rows, cols: spans → block_width_tiles, stick_in_tile: spans → tile_h (page size)}` | input dtype; Phase 0 `Float16_b` (no DEST accumulation — the page carries input bytes; fp32/32-bit int refinements use `Float32`/`Int32`/`UInt32` with `fp32_dest_acc_en` on and `UnpackToDestFp32`) | reader | compute | whole kernel | Cannot share with `cb_tiles_out`: concurrent lifetime (reader fills tile-row k+1 while compute packs tile-row k into the other CB), and the helper requires distinct input/output DFBs (tilize_helpers.hpp:99). Capacity over live set (2×) is the double-buffer decision (`in_depth_tile_rows = 2`). |
| `cb_tiles_out` | `out_depth_tile_rows · block_width_tiles`, page = output tile size (`tile_h × 32`, output dtype) | `block_width_tiles` pages (one packed tile-row) | `{rows: streams → window of out_depth_tile_rows tile-rows, cols: spans → block_width_tiles, stick_in_tile: spans → tile_h (page size)}` | output dtype; Phase 0 `Float16_b` with `fp32_dest_acc_en = False` (16-bit DEST ↔ 16-bit page: consistent) | compute | writer | whole kernel | Cannot share with `cb_rm_sticks`: concurrent lifetime (above). Capacity over live set (2×) is the double-buffer decision (`out_depth_tile_rows = 2`). Cannot be zero-copy over the output in interleaved regimes (output pages spread over banks); in the sharded-output regimes it IS backed on the output shard (no extra L1). |
| `cb_rm_sticks_tail` (slot 1, implementer-added) | `in_depth_tile_rows · last_block_width_tiles`, page as `cb_rm_sticks` | `last_block_width_tiles` pages | as `cb_rm_sticks`, `cols: spans → last_block_width_tiles` | as `cb_rm_sticks` | reader | compute | allocated only when `last_block_width_tiles ≠ block_width_tiles` (ragged last column block, i.e. `num_col_blocks > 1`) | Cannot share `cb_rm_sticks`' ring: the tail quantum `last_block_width_tiles` does not divide `in_depth·block_width_tiles`, so a contiguous tail tile-row could straddle the ring end. Own ring = one quantum per ring (the design's ring-wrap note). Could alias `cb_rm_sticks` by lifetime (main blocks finish before the tail starts), but CB descriptors have no alias API here; cost ≤ main CB size, only on ragged shapes. |
| `cb_tiles_out_tail` (slot 17, implementer-added) | `out_depth_tile_rows · last_block_width_tiles`, page as `cb_tiles_out` | `last_block_width_tiles` pages | as `cb_tiles_out`, `cols: spans → last_block_width_tiles` | as `cb_tiles_out` | compute | writer | same as `cb_rm_sticks_tail` | Same reason as `cb_rm_sticks_tail`. |

## Symbol table

| Symbol | Bound | Predicate establishing it |
|--------|-------|---------------------------|
| `block_width_tiles` | `≤ max_block_width_tiles ≤ min(255, L1_BLOCK_BUDGET_BYTES // per_col_tile_bytes)`; `≤ 8` when `low_l1` | host clamp in `max_block_width_tiles` (op_design.md Blocking Model); independent of tensor dims under `low_l1` |
| `last_block_width_tiles` | `1 ≤ … ≤ block_width_tiles` | derived |
| `in_depth_tile_rows`, `out_depth_tile_rows` | constant 2 (Phase 0) | host constants |
| `tile_h` | `∈ {1,2,4,8,16,32}` | validate() (`tile` check) |
| `in_elem_bytes`, `out_elem_bytes` | `≤ 4` | dtype set in TARGET |
| `L1_BLOCK_BUDGET_BYTES` | 512 KiB | host constant |
| `per_col_tile_bytes` | `in_depth·in_page_bytes + out_depth·out_page_bytes` ≤ `2·4096 + 2·4096` | derived |

No tensor dimension appears in any capacity expression: `block_width_tiles` is `min(core_cols_tiles, cap)` and the cap is dimension-independent, so the footprint is bounded for every shape at both `low_l1` settings, and O(1) in the tensor's dims under `low_l1=True` (cap = 8).

## Total per-core footprint

```
L1 = block_width_tiles · (in_depth_tile_rows · tile_h·32·in_elem_bytes
                          + out_depth_tile_rows · out_tile_bytes)
   ≤ L1_BLOCK_BUDGET_BYTES                       (low_l1 = False)   = 512 KiB
   (+ tail rings ≤ the same again, only on ragged column blocking: ≤ 1 MiB total)
   ≤ 8 · per_col_tile_bytes ≤ 8 · 16 KiB = 128 KiB  (low_l1 = True)
```

Scales linearly with `block_width_tiles` (knob) and the two depth knobs; independent of `block_height_tiles` (streamed) and of every tensor dimension. Phase 0 bf16 example: `[1,1,2048,64]` → `block_width_tiles = 2` → `2 · (2·2048 + 2·2048) = 16 KiB`.

## Data-movement budget (chosen split: `interleaved_row_split`)

| Tensor | DRAM crossings | Why that many | Cross-core traffic added |
|--------|----------------|---------------|--------------------------|
| `input_tensor` | 1 | each core reads only its own tile-rows' sticks, each stick segment once; nothing is re-read (no second pass, no shared operand) | none |
| `output` | 1 | each tile packed once and written once by the core that produced it | none |

Total per tier: DRAM `N·in_elem_bytes` read + `N_padded·out_elem_bytes` written (N = element count); NoC cross-core 0; local L1: each byte written once to `cb_rm_sticks`, read once by the unpacker, written once to `cb_tiles_out`, read once by the writer.

> Cheapest-traffic split considered: `rows only` — bytes are identical across every candidate split (each crosses DRAM once); rows-only additionally issues the fewest read transactions (one per whole stick). Implemented: `rows only` (`interleaved_row_split`) — the implemented split is the cheapest. The 2-D split (`interleaved_2d_split`, same bytes, `cores_c×` more read transactions) is deferred because it serves only `short_wide` / `square_large` (and `tall_narrow` with `R < grid`), outside Phase 0's declared domain; the per-core rectangle RT args keep it a host-only change.
