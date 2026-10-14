# L1 Ledger: tilize

Schema: `.claude/references/l1-footprint-discipline.md`. Block axes (from `op_design.md`): `rows`, `cols`, `stick_in_tile`.

| CB | Capacity (pages) | Live set | Axis accounting | Page format | Producer | Consumer | Lifetime | Shares with / why not |
|----|------------------|----------|-----------------|-------------|----------|----------|----------|-----------------------|
| `cb_rm_sticks` | `in_depth_tile_rows · block_width_tiles`, page = `tile_h · 32 · in_elem_bytes` | `block_width_tiles` pages (one tile-row being unpacked) | `{rows: streams → window of in_depth_tile_rows tile-rows, cols: spans → block_width_tiles, stick_in_tile: spans → tile_h (page size)}` | input dtype; Phase 0 `Float16_b` (no DEST accumulation — the page carries input bytes; fp32/32-bit int refinements use `Float32`/`Int32`/`UInt32` with `fp32_dest_acc_en` on and `UnpackToDestFp32`) | reader | compute | whole kernel | Cannot share with `cb_tiles_out`: concurrent lifetime (reader fills tile-row k+1 while compute packs tile-row k into the other CB), and the helper requires distinct input/output DFBs (tilize_helpers.hpp:99). Capacity over live set (2×) is the double-buffer decision (`in_depth_tile_rows = 2`). |
| `cb_tiles_out` | `out_depth_tile_rows · block_width_tiles`, page = output tile size (`tile_h × 32`, output dtype) | `block_width_tiles` pages (one packed tile-row) | `{rows: streams → window of out_depth_tile_rows tile-rows, cols: spans → block_width_tiles, stick_in_tile: spans → tile_h (page size)}` | output dtype; Phase 0 `Float16_b` with `fp32_dest_acc_en = False` (16-bit DEST ↔ 16-bit page: consistent) | compute | writer | whole kernel | Cannot share with `cb_rm_sticks`: concurrent lifetime (above). Capacity over live set (2×) is the double-buffer decision (`out_depth_tile_rows = 2`). Cannot be zero-copy over the output in interleaved regimes (output pages spread over banks); in the sharded-output regimes it IS backed on the output shard (no extra L1). |
| `cb_rm_sticks_tail` (slot 1, implementer-added) | `in_depth_tile_rows · last_block_width_tiles`, page as `cb_rm_sticks` | `last_block_width_tiles` pages | as `cb_rm_sticks`, `cols: spans → last_block_width_tiles` | as `cb_rm_sticks` | reader | compute | allocated only when `last_block_width_tiles ≠ block_width_tiles` (ragged last column block, i.e. `num_col_blocks > 1`) | Cannot share `cb_rm_sticks`' ring: the tail quantum `last_block_width_tiles` does not divide `in_depth·block_width_tiles`, so a contiguous tail tile-row could straddle the ring end. Own ring = one quantum per ring (the design's ring-wrap note). Could alias `cb_rm_sticks` by lifetime (main blocks finish before the tail starts), but CB descriptors have no alias API here; cost ≤ main CB size, only on ragged shapes. |
| `cb_tiles_out_tail` (slot 17, implementer-added) | `out_depth_tile_rows · last_block_width_tiles`, page as `cb_tiles_out` | `last_block_width_tiles` pages | as `cb_tiles_out`, `cols: spans → last_block_width_tiles` | as `cb_tiles_out` | compute | writer | same as `cb_rm_sticks_tail` | Same reason as `cb_rm_sticks_tail`. |
| `cb_stick_staging` (slot 2, implementer-added, perf) | 1 page of `2 · num_banks · ceil(staging_tile_rows·tile_h / num_banks) · stick_bytes`; `staging_tile_rows = clamp(STAGING_SLOT_BUDGET_BYTES (16 KiB) // (tile_h·stick_bytes), 1, max core rows)` | one slot being copied out while the other slot's DRAM reads land | `{rows: streams → staging_tile_rows per slot, cols: spans the whole stick (only when num_col_blocks = 1), stick_in_tile: spans}` | input dtype (raw bytes) | reader | reader (local L1→L1 NoC copies into `cb_rm_sticks`) | whole kernel; allocated only when the bank-coalesced reader is active (interleaved source, one column block, aligned images, cores ≥ 2·num_banks) | Reader-private scratch, no CB handshake. Cannot be `cb_rm_sticks` itself: the wide per-bank reads land bank-grouped, and tilize needs row order, so a permuting copy between two buffers is required. ≈ 2×16 KiB (one slot may round up to a whole tile-row when a tile-row exceeds 16 KiB, i.e. up to 2 · tile_h · stick_bytes ≤ 2 · per-tile-row input bytes of the block). |
| `cb_pad_fill` (slot 3, Refinement 1) | 1 page of `fill_row_bytes + 64`, `fill_row_bytes = block_width_tiles · 32 · in_elem_bytes + 64` | the whole fill row (read-only after setup) | `{rows: n/a (one stick), cols: spans → block_width_tiles (+64 B phase slack), stick_in_tile: n/a}` | input dtype (the pad value replicated) | reader (once, at kernel start: 64 B RISC seed + doubling local NoC copies) | reader (source of local L1→L1 fill copies into `cb_rm_sticks`) | whole kernel; allocated only when the output is padded | Reader-private, no CB handshake. Cannot be `cb_rm_sticks`: it must outlive every tile-row. Not a zero-tile helper case — the fill is an arbitrary value, and filling via NoC copies (not a RISC store loop per segment) keeps the DM engine moving the bytes. Mutually exclusive with `cb_stick_staging` (coalescing requires no padding). ≤ 255·128 + 128 B. |

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
| `stick_bytes` (in the staging term only) | `≤ block_width_tiles · 32 · in_elem_bytes` (+ alignment pad) | staging exists only when `num_col_blocks = 1`, i.e. the whole stick is one block, so the stick is bounded by the block cap |
| `staging_tile_rows` | `max(1, min(core_rows, STAGING_SLOT_BUDGET_BYTES // (tile_h · stick_bytes)))`; slot ≤ `max(16 KiB, tile_h · stick_bytes)` rounded up to `num_banks` sticks | host clamp (`STAGING_SLOT_BUDGET_BYTES = 16 KiB`) |
| `num_banks` | DRAM banks of the input buffer (BH 8, WH 12); coalescing gated on `cores_used ≥ 2·num_banks` | `ttnn.get_memory_view(...)`; L1-interleaved sources never meet the gate in practice |

No tensor dimension appears in any capacity expression: `block_width_tiles` is `min(core_cols_tiles, cap)` and the cap is dimension-independent, so the footprint is bounded for every shape at both `low_l1` settings, and O(1) in the tensor's dims under `low_l1=True` (cap = 8).

## Total per-core footprint

```
L1 = block_width_tiles · (in_depth_tile_rows · tile_h·32·in_elem_bytes
                          + out_depth_tile_rows · out_tile_bytes)
     + [num_col_blocks = 1 and coalesced]  2 · slot_bytes,
           slot_bytes = num_banks · ceil(staging_tile_rows·tile_h / num_banks) · stick_bytes
     + [ragged column blocking]  tail rings (≤ the main rings)
     + [padded output]  fill row (≤ block_width_tiles · 32 · in_elem_bytes + 128 B)
   ≤ L1_BLOCK_BUDGET_BYTES (512 KiB) + staging (≤ 2 · tile_h · block_width_tiles·32·in_elem_bytes,
                                               ≤ 256 KiB at the bf16/fp32 caps)        (low_l1 = False)
   ≤ 8 · per_col_tile_bytes + staging(C ≤ 8: ≤ 2·32·256·4 = 64 KiB) ≤ 192 KiB          (low_l1 = True)
```

Staging and the tail rings are mutually exclusive (staging needs `num_col_blocks = 1`, a tail needs
`num_col_blocks > 1`), so the worst case is `max(staging, tail rings)` on top of the main rings.
(Verifier, Phase 0 audit: the staging term was missing from this total; added.)

Scales linearly with `block_width_tiles` (knob) and the two depth knobs; independent of `block_height_tiles` (streamed) and of every tensor dimension. Phase 0 bf16 example: `[1,1,2048,64]` → `block_width_tiles = 2` → `2 · (2·2048 + 2·2048) = 16 KiB`.

Per-group CBs (Refinement 1): under `interleaved_2d_split` cores with different column extents
(at most two: full and ragged last column group) get their own CB set sized by their own
`block_width_tiles`, so every bound above holds per core.

## Data-movement budget (chosen split: `interleaved_row_split`, `interleaved_2d_split` when R < grid)

| Tensor | DRAM crossings | Why that many | Cross-core traffic added |
|--------|----------------|---------------|--------------------------|
| `input_tensor` | 1 | each core reads only its own rectangle's stick segments (2-D split: `cores_c` segments per stick across cores, same bytes); pad regions are never read — they are filled by local copies from `cb_pad_fill`, each stick segment once; nothing is re-read (no second pass, no shared operand). Coalesced reader: same bytes in `num_banks` wide reads per staging chunk instead of one read per stick, plus one local L1→L1 copy per stick (no extra DRAM crossing) | none |
| `output` | 1 | each tile packed once and written once by the core that produced it | none |

Total per tier: DRAM `N·in_elem_bytes` read + `N_padded·out_elem_bytes` written (N = element count); NoC cross-core 0; local L1: each byte written once to `cb_rm_sticks`, read once by the unpacker, written once to `cb_tiles_out`, read once by the writer.

> Cheapest-traffic split considered: `rows only` — bytes are identical across every candidate split (each crosses DRAM once); rows-only additionally issues the fewest read transactions (one per whole stick). Implemented: `rows only` whenever `R ≥ grid`, and `interleaved_2d_split` (same bytes, `cores_c×` more, shorter read transactions) only when `R < grid`, where the selector minimizes the busiest core's tile count (ties → more row cores). Refinement 1 measured on BH (110 cores): `[1,1,32,2048]` 5.6 → 4.1 µs, `[1,1,32,8192]` 12.2 → 6.1 µs, `[1,1,2048,2048]` 43.2 → 39.6 µs.
