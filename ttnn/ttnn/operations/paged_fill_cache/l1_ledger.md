# L1 Ledger: paged_fill_cache

Per cluster, Phase 0, target **quasar** (4 MiB L1 per cluster, 32 clusters). Schema and audits:
`.claude/references/l1-footprint-discipline.md`. Thread plan (R, C, W) = (1, 0, 1): no compute
kernel, so no capacity is multiplied by a compute-thread count.

Block axes of the Blocking Model: `input_tile` (the fold of `head` x `seq_tile_row` x `d_tile`),
`page_table_entry`, `cache_block` (output scatter target, never traversed).

| CB | Capacity (pages) | Live set | Axis accounting | Page format | Producer | Consumer | Lifetime | Shares with / why not |
|----|------------------|----------|-----------------|-------------|----------|----------|----------|-----------------------|
| `dfb_input_tiles` | `TILE_RING_ENTRIES` (= 8, even) x `tile_bytes` | `TILE_RING_ENTRIES`: one txn-id half filling while the other drains (implicit-sync double buffer); the tile itself is live only between its read landing and its write leaving L1 | `{input_tile: streams → window TILE_RING_ENTRIES, page_table_entry: streams → 0 (not held here), cache_block: streams → 0 (destination only)}` (`head`, `seq_tile_row`, `d_tile`: streamed as part of `input_tile`) | input dtype (bfloat16 at Phase 0; float32 when TARGET's fp32 is unlocked). Dataflow-only payload, never passes through DEST (no compute kernel), so the DEST-width audit does not apply; the width must equal the tensors' for a bit-exact copy (R1) | reader | writer | the whole tile stream (start of `stream_mapped_tiles_in` to the end of `scatter_tiles_to_cache`) | none: concurrent with `dfb_page_table_row` for its whole lifetime (the row is the lookup table for every tile), and a different page format (int32 vs. tile dtype). Capacity = live set: the 2x is the double buffer, the mechanism reason (txn-id halves). No phase boundary exists to fold away: the entry read **is** the entry written (pack-into-destination already holds: zero intermediate buffers) |
| `dfb_page_table_row` | `PT_RING_ENTRIES` (= 2) x `pt_entry_bytes`, `pt_entry_bytes = align_up(4 * blocks_per_seq, dram_alignment)` | 2: entry 0 held by the writer, entry 1 held (reserved) by the reader, both for the whole stream | `{page_table_entry: spans → whole row (blocks_per_seq entries), input_tile: streams → 0 (looked up once per tile row, not stored per tile), cache_block: streams → 0}` | int32 (raw page-table values; `-1` = skip). Never through DEST | reader | writer | whole kernel lifetime of both DM kernels | none: concurrent with `dfb_input_tiles` (above). The two entries cannot share one: a DM kernel cannot read an entry it has pushed without breaking single ownership, and Quasar offers no self-bound DM scratch ring (`dataflow_buffer.cpp:2238-2254`), so each kernel owns one copy. Capacity = live set |

## Symbol table

| Symbol | Bound | Predicate establishing it |
|--------|-------|---------------------------|
| `TILE_RING_ENTRIES` | 8 at Phase 0; even; `TILE_RING_ENTRIES * tile_bytes <= 1,048,560` | host constant; host assert (even, ring cap) |
| `tile_bytes` | 2048 (bfloat16), 4096 (float32) | `input_tensor.buffer_page_size()`; TARGET dtypes; `validate()` refuses others |
| `PT_RING_ENTRIES` | 2 | structural constant (one copy per kernel) |
| `blocks_per_seq` | `<= 131,056` (so `2 * pt_entry_bytes <= 1,048,560`) | `validate()` refuses a page table wider than that; INPUTS maximum is 6 (24 B → 64 B entry) |
| `dram_alignment` | 64 B on quasar | `ttnn.get_dram_alignment()` (queried, never typed in) |
| `l1_budget` | `ttnn.get_max_worker_l1_unreserved_size() - l1_reserve_bytes` (just under 4 MiB on quasar) | run-time query; the footprint below is a fixed window ≪ budget, so no predicate is evaluated against it |

## Total per-cluster footprint

```
footprint = TILE_RING_ENTRIES * tile_bytes  +  PT_RING_ENTRIES * align_up(4 * blocks_per_seq, dram_alignment)
          = 8 * 2048 + 2 * 64 = 16,512 B   (Phase 0, bfloat16, blocks_per_seq <= 16 on INPUTS)
          = 8 * 4096 + 2 * 64 = 32,896 B   (float32)
```

- Term 1 scales with the buffer-depth knob `TILE_RING_ENTRIES` and the dtype; it does **not** scale with
  any op dimension (H, seq, D, blocks, users): the block streams through a fixed window.
- Term 2 scales with `blocks_per_seq` only (bounded above); it does not scale with any block knob.
- No term scales with `core_tiles` (the block extent) or the grid.

## Data-movement budget

Chosen split: R-filter (tile-granular over the full grid, page-table row read per cluster, `-1` rows
filtered at the source). `n_mapped` = input tiles whose tile row's entry is not `-1`
(= `H * seq_t * D_t` when no entry is `-1`).

| Tensor | DRAM crossings | Why that many | Cross-core traffic added |
|--------|----------------|---------------|--------------------------|
| `input_tensor` | 1 (read), only its `n_mapped` tiles; skipped rows 0 | each tile is owned by one cluster and copied once; no tile is needed twice, so nothing is held for reuse; filtering at the source removes the skipped rows' reads | none |
| `cache_tensor` | 1 (write) of `n_mapped` tiles; 0 reads | a write-only scatter: every mapped input tile lands once in its page; the rest of the cache is never touched (R2) | none |
| `page_table` | 2 reads of one row page per engaged cluster (`<= 2 * 32` reads of `pt_page_bytes`, each a 64-B-aligned burst) | each of the two DM kernels needs the row for its whole stream and must own its copy; the row is reuse-shared across clusters but read per cluster (broadcast rejected: ≤ 4 KB saved vs. a handshake on the critical path) | none |

Totals per call (bfloat16):
- DRAM reads: `2048 * n_mapped + 2 * clusters * pt_page_bytes` B (Llama call: 524,288 + 2 * 32 * 64 = 528,384 B).
- DRAM writes: `2048 * n_mapped` B (Llama call: 524,288 B).
- Cross-core NoC (cluster↔cluster): 0 B.
- Core-local L1 re-reads: 0 (each ring entry is written once by a NoC read and read once by a NoC write; the row is read by the RISC once per tile row).

> Cheapest-traffic split considered: R-filter — it is the minimum for both tensors (each mapped input
> tile read once, each written cache tile written once, the cache never read). Implemented: R-filter.
> The one excess is the page-table row: 2 DRAM reads of the same page per engaged cluster. Removing
> *both* down to one per call would need a cross-cluster handshake (R-mcast-pt, rejected), but the
> *second* read per cluster needs none: the second copy can be taken from the first one in the
> cluster's own L1 (a NoC L1->L1 self-read, or a RISC copy with the DM data-cache maintenance the API
> requires). Verifier, 2026-11-21: **deferred** — folded into Refinement 2 of `op_requirements.md`
> (page-table DRAM reads per call: 2 x engaged clusters -> 1 x engaged clusters). Not fixed in place
> because the DM cores keep an L1 data cache and a RISC-side copy handed to another hart needs a
> verified flush/invalidate sequence; the refinement owns that.
