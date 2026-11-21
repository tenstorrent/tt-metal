# L1 ledger: paged_update_cache

Schema and audits: `.claude/references/l1-footprint-discipline.md`. Design: `op_design.md` (regime `direct_row_write`). Worked numbers are for Quasar (4 MiB L1, `ttnn.get_max_worker_l1_unreserved_size()` = 3,963,328 B probed on ttsim, DRAM alignment `A` = 64 B).

**Inventory.** The op allocates **one** buffer per engaged core: `index_scratch`. Before it was accepted, every other candidate was removed:

| Candidate buffer | Removed by | Why it does not exist |
|------------------|------------|-----------------------|
| a staging ring for cache tiles (`tile_rmw`) | rule 3.1 (pack into the destination) | the face-row segments are written straight into the cache page at their offset; the cache is never read |
| a staging ring for the input rows | rule 3.1 | the input shard is already resident on the engaged core: it is the NoC write source, in place (`LocalTensorAccessor`) |
| a DFB borrowed from the input shard | not needed (and a DM-only self-loop ring is forbidden on Quasar) | the shard is addressed directly; a ring would only add a copy |
| a separate positions buffer and page-table buffer | rule 3.3 (one allocation) | both are read by the same thread in the same barrier window and read back together: one scratchpad, two regions |
| a reader→writer handoff ring for the resolved indices | thread plan (one DM thread) | the index reads and the writes are in the same thread; values stay in registers |

| CB | Capacity (pages) | Live set | Axis accounting | Page format | Producer | Consumer | Lifetime | Shares with / why not |
|----|------------------|----------|-----------------|-------------|----------|----------|----------|-----------------------|
| `index_scratch` (ScratchpadSpec, 1-byte pages) | `ROW_WRITER_THREADS · (pos_page_bytes + block_users_max · pt_row_bytes)` (one slot per writer thread, so threads never share bytes; Phase 0 W = 1), with `pos_page_bytes = align_up(users·4, A)`, `pt_row_bytes = align_up(blocks_per_seq·4, A)`, `block_users_max = users_per_shard` | equal to capacity: every region is written by one read barrier and read by one `scoped_lock` window; peak = all of it | `{users: spans → block_users (one page-table row slot per user of the block, plus the one positions page), heads: streams → not held (the write loop walks heads with the indices in registers), dtiles: streams → not held, face_half: streams → not held}` | none (DM-only bytes; int32 values; uint32/uint16 are not buffer formats on Quasar and none is needed) | `row_writer` (tagged NoC reads) | `row_writer` (CPU loads under `scoped_lock`) | `load_index_block` → end of `write_rows_block`: the kernel resolves each user (`resolve_block`) just before that user's writes, under one `scoped_lock` held across the user loop, so the slot stays live through the writes (no other op buffer exists to reuse the bytes, so the longer lifetime costs nothing) | **Shares with nothing**: it is the op's only allocation, so there is no other lifetime to alias with. Its positions region and row region are deliberately one allocation (inventory above). Capacity exceeds the strict need (4 B position, 4 B entry per user) by the alignment padding of the position page and by the rest of the page-table row: the row is read whole to hide the second dependent DRAM round trip (stall shadow; `dependent_entry_read` is the deferred opposite trade) |
| `input` shard (tensor-owned, accessed via `LocalTensorAccessor`) | 0 allocated by the op (caller's `users_per_shard · dtiles · input_page_bytes`) | the `block_users × heads` head rows actually read: `2 · block_users · heads · dtiles` segments of `face_row_bytes` — rows `heads..31` are never touched | `{users: spans → block_users (the shard's tile rows), heads: spans → heads of 32 rows, dtiles: spans → dtiles, face_half: spans → 2}` | cell dtype (bfloat16 Phase 0; float32 TARGET) | caller (before launch) | `row_writer` (NoC write source) | whole program | Not an op buffer: consumed in place as the write source, which is why no input ring exists |

## Symbol table

| Symbol | Bound | Predicate establishing it |
|--------|-------|---------------------------|
| `A` | `ttnn.get_dram_alignment()` (64 on Quasar) | device query |
| `users` | 1 in TARGET/PHASE0 | SUPPORTED `users = [1]` |
| `users_per_shard` | `shard_shape[0] // 32`; = 1 for every shape the golden runner builds (padded height 32 → one shard) | the input's shard spec |
| `block_users_max` | `users_per_shard` | as above |
| `blocks_per_seq` | `≤ blocks` (X6); INPUTS max 128 | `validate()` X6 + the scratch predicate below |
| `pos_page_bytes` | `align_up(users·4, A)` = 64 at users = 1 | — |
| `pt_row_bytes` | `align_up(blocks_per_seq·4, A)`; INPUTS max 512 | scratch predicate |
| `budget` | `ttnn.get_max_worker_l1_unreserved_size() − l1_reserve_bytes` = 3,963,328 − 262,144 = 3,701,184 B on Quasar | device query + host constant `l1_reserve_bytes` |
| `index_scratch_bytes` | `≤ budget` | host predicate in `validate()`; beyond it `NotImplementedError` (deferred regime `dependent_entry_read`) |

## Total per-core footprint

`footprint = ROW_WRITER_THREADS · (pos_page_bytes + block_users_max · pt_row_bytes)` (implemented; W = 1 in Phase 0)

- scales with `block_users` (the shard's users) and with `blocks_per_seq` (through `pt_row_bytes`);
- constant in `heads`, `dtiles` and the dtype: the data never passes through an op buffer.

Worked on Quasar: Llama traced call 64 + 1 · 512 = **576 B**; Qwen traced call 64 + 1 · 64 = **128 B**; largest INPUTS entry 576 B. That is 0.016 % of the budget.

## Data-movement budget

Per call, per user (`E` = element bytes, 2 for bf16 / 4 for fp32; `T` = tile bytes):

| Tensor | DRAM crossings | Why that many | Cross-core traffic added |
|--------|----------------|---------------|--------------------------|
| `cache_tensor` | reads: **0**; writes: `heads·D·E` B (the touched rows only, a `1/block_size` sliver of one block per head), in `2·heads·dtiles` transfers of `16·E` B | written in place from the input; no tile is read back because the NoC writes a face row at its byte offset | 0 |
| `input_tensor` | **0** DRAM reads (it lives in L1); its `heads` rows cross the NoC once, L1 → DRAM, which is the cache write above | resident shard consumed in place on the core that holds it | 0 |
| `update_idxs_tensor` | 1 page (`pos_page_bytes` = 64 B) per engaged core | one engaged core per user; the whole page because a position is 4 B of a 64-B-aligned page | 0 |
| `page_table` | 1 row (`pt_row_bytes`) per user | the whole row, read together with the position, so the entry's read does not wait on the position (stall shadow) | 0 |

Totals per tier, Llama traced call (bf16, 8 heads, D = 64, 128-entry table): DRAM writes 1,024 B (32 transfers), DRAM reads 576 B (2 transfers), cross-core 0 B, core-local 0 B copied. Qwen traced call (8 heads, D = 128, 8-entry table): DRAM writes 2,048 B (64 transfers), DRAM reads 128 B, cross-core 0 B. The rejected `tile_rmw` would read and write `2 · heads · dtiles · T` = 65,536 B (Llama) and 131,072 B (Qwen).

> Cheapest-traffic split considered: users across the caller's shard cores with every head and D-tile of a user on its shard core (`direct_row_write`) — the cache and input cross DRAM exactly once at their minimum, 0 B cross-core. Implemented: the same split. The implemented split is the cheapest; the next one, `grid_spread_pull`, adds `heads·D·E` B cross-core and `(G − 1)·(pos_page_bytes + pt_row_bytes)` B of DRAM reads per user, and is a `deferred` regime row in `op_design.md` with its reachability argument.
