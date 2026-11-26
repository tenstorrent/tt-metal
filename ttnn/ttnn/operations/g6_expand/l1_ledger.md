# L1 Ledger: g6_expand

Schema and audits: `.claude/references/l1-footprint-discipline.md`. Blocking Model: `op_design.md`.
Block axes: `n` (images, `block_images`), `h` (out tile rows, `block_tile_rows`), `w` (out tile columns, `block_tile_cols`).

| CB | Capacity (pages) | Live set | Axis accounting | Page format | Producer | Consumer | Lifetime | Shares with / why not |
|----|------------------|----------|-----------------|-------------|----------|----------|----------|-----------------------|
| `cb_bcast_row_tiles` | `block_tile_cols · bcast_buffer_depth` (page = `tile_bytes` = 2048 B) | `block_tile_cols` (one block's B tiles: each is held from its read until its last out write is flushed; up to all of a block's tiles at once while the writer drains) | `{n: streams → 1 image per block (block_images = 1), h: streams → one B tile serves all this_block_tile_rows writes, w: spans → block_tile_cols}` | bfloat16 (`Float16_b`) — input dtype; raw bytes moved by the NoC, never unpacked, never through DEST (no `fp32_dest_acc_en` exists; audit 2 n/a: dataflow-only payload) | reader | writer | whole kernel; per page: reserve → partial x read → in-place replication → push → `this_block_tile_rows` writes → flush → pop | the only CB. Capacity − live set = `block_tile_cols · (bcast_buffer_depth − 1)`: double buffering across blocks (reader prepares block k+1 while writer drains block k), mechanism reason. The x landing area is the B tile itself (in-place transform, Rule 3.2) and out is written straight from it (no staging CB, Rule 3.1), so no second buffer exists to share with. |

## Symbol table

| Symbol | Bound | Predicate establishing it |
|--------|-------|---------------------------|
| `tile_bytes` | 2048 | bfloat16 TILE (SUPPORTED dtype/layout) |
| `bcast_buffer_depth` | 2 (host constant) | Phase 0 knob |
| `block_tile_cols` | `1 ≤ block_tile_cols ≤ max_block_tile_cols` | host clamp `block_tile_cols = clamp(ceil(Wt / num_cores), 1, max_block_tile_cols)`; production = 1 |
| `max_block_tile_cols` | `floor((1347776 − reserved_margin) / (bcast_buffer_depth · tile_bytes))` (≈ 328 at margin 4 KB) | BB-05 budget; wider assignments become more blocks per core, never a refusal |
| `block_tile_rows` | `≤ tensor_out_tile_rows`; not in any capacity expression | streams (writes from one page) |
| `Wt`, `Ht`, `tensor_images` | unbounded (W has no upper limit) — appear in **no** capacity expression | — |

## Footprint

Total per core = `block_tile_cols · bcast_buffer_depth · tile_bytes` = **4096 B in every production case** (1 · 2 · 2048), ≤ `1347776 − reserved_margin` for every W by the clamp. Scales only with `block_tile_cols` (w-extent knob) and `bcast_buffer_depth`; independent of W, H and the image count. No semaphores, no compute CBs.

## Data-movement budget

Chosen split: R1 (columns first, then rows; per-core partial read of x). Counts on a 110-core grid (`row_groups` = 6 / 4 / 4 / 4 / 3).

| Tensor | DRAM crossings | Why that many | Cross-core traffic added |
|--------|----------------|---------------|--------------------------|
| out (W=512: 256 pages, 524 288 B) | 1 (write only; never read) | every page written exactly once from a resident B tile; no intermediate tensor in DRAM | 0 |
| x (W=512: 16 pages) | `row_groups` = 6 partial reads per page, 576 B each (28 % of a page): 96 requests, 55 296 B | each core that writes column w builds its own B(n, w); one read per (core, column), reused for all `block_tile_rows` writes of that column | 0 |
| x (W=256 / 207 / 196 / 185) | 4 / 4 / 4 / 3 partial reads per page: 32 / 28 / 28 / 18 requests (18 432 / 16 128 / 16 128 / 10 368 B) | same; `min_block_tile_rows = 2` keeps it ≤ one read per two out tiles of the column | 0 |

Totals per tier at W=512: DRAM write 524 288 B (minimum), DRAM read 55 296 B (minimum 9 216 B), cross-core 0, core-local replication 16 × 2 KB per row-group core (96 B tiles = 192 KB of L1→L1 self-copies, 62 transfers per tile).

> Cheapest-traffic split considered: columns only (R4, `row_groups = 1`) — DRAM read 9 216 B instead of 55 296 B (−46 080 B DRAM, −80 requests), same 524 288 B written, but 16 cores issuing 16 writes each instead of 96 cores issuing ≤ 3. Read-once + mcast (R2) reaches the same DRAM read but adds 80 × 2048 B = 163 840 B of cross-core multicast and a semaphore handshake per column group. Implemented: R1 with `min_block_tile_rows = 2`. R4 is not deferred — it is the same built code at `min_block_tile_rows = Ht` (a knob-turn); R2 is deferred because it places the sender's read + replication + handshake in front of every receiver's first write in a write-bound op, while the reads it removes are single 576 B requests issued concurrently at kernel start.
