# L1 Ledger: nomic_moe_dispatch

Schema: `.claude/references/l1-footprint-discipline.md`. The block axes are those of `op_design.md`'s Blocking Model:

| Axis | Block extent |
|---|---|
| `tile_row` | `block_tile_rows = 1` (shard height), `row_slots` blocks resident |
| `column` | `block_width_tiles` w |
| `row_in_block` | 32 rows, `rows_noc0` / `rows_noc1` per reader |
| `token` | not blocked by the op: the table is the caller's |
| `index` | the whole index image a core (a row's round span lands per block; Refinement 3) |

Compute config: `fp32_dest_acc_en = False`, so 16-bit DEST. Every tile page is Float16_b.

| CB | Capacity (pages) | Live set | Axis accounting | Page format | Producer | Consumer | Lifetime | Shares with / why not |
|----|------------------|----------|-----------------|-------------|----------|----------|----------|-----------------------|
| `cb_rows_head` (idx 0) | `row_slots · w` pages of 2048 B; `row_slots = max_blocks_per_core` (≤ 13, a bounded symbol) | `row_slots · w`: all slots may hold rows at once, because the readers run ahead of compute by design | `{tile_row: spans → row_slots, column: spans → w, row_in_block: spans → 32 (rows 0..rows_noc0-1 written by this index's producer), token: streams → one piece per row, index: streams}` | Float16_b | reader_noc0 | compute | whole kernel | **Aliased with `cb_rows_tail`** (one `CBDescriptor`, two format descriptors): the two readers fill disjoint row halves of one slot, so one allocation, two credit counters. **Not shared with `cb_output_tiles`:** an in-place tilize was considered and rejected. The packer writes tile 0 (2 KB) over RM rows 0-2 of the slot while the unpacker still reads those rows for tiles 1..w-1, which corrupts data. **Capacity > a 2-slot ring:** explicit pipelining decision (full residency; perf lamp L3). |
| `cb_rows_tail` (idx 1) | the same allocation as `cb_rows_head` (0 extra bytes) | same | `{tile_row: spans → row_slots, column: spans → w, row_in_block: spans → 32 (rows rows_noc0..31 written by this index's producer), token: streams, index: streams}` | Float16_b | reader_noc1 | compute | whole kernel | Aliased with `cb_rows_head`, kept in lockstep: page 2048 B on both, push / pop `w` per block, both from base |
| `cb_output_tiles` (idx 2) | `max_blocks_per_core · w` pages of 2048 B, backed on the output tensor (0 CB bytes beyond the io tensor) | `num_blocks_this_core · w` (each tile written once, kept) | `{tile_row: spans → max_blocks_per_core, column: spans → w, row_in_block: spans → 32 (tile height), token: streams, index: streams}` | Float16_b | compute | none (it is the output tensor) | whole kernel | **Is** the output: "pack into the destination" (Rule 3.1). It cannot share with the rows CBs (see above). |
| `cb_gather_ids` (idx 3), Refinement 3 | `R·4 + (max_blocks_per_core + 1)·64` B in 64 B pages: a full-size gather_index image (tile row t's ids at t·128 B), one 64 B count cell a round, and 64 B of alignment slack | per core: its grid row's round spans, ≤ `max_blocks_per_core · (ceil(11/k) + 1) · 128` B of ids (≤ 6.5 KB), plus on a row sender its round-count cells; nothing is ever overwritten (each tile row's slot is written once a call) | `{tile_row: spans → all tile rows (an image; only the core's row's spans are written), column: streams (the same ids serve every piece), row_in_block: spans → 32 ids per tile row, token: streams, index: spans → R}` | UInt32 | the row sender's multicast (or, on the sender core, its own DRAM read) | reader_noc0 and reader_noc1 (both read, neither pushes or pops: self-owned scratch) | whole kernel | **One CB for both readers (was one per reader):** the row multicast lands a round's span at the same address on every core, and both RISCs read their halves of it. **Full image, not a ring:** a ring would need a reuse handshake from all 11 row members back to the sender before a slot could be rewritten; the image costs ≤ 34 KB a core (R = 8512) and stays within the plan (156.9 KB total at R = 8416, under the 256 KiB cap and beside the perf cases' 96 KB reservation). **Not in the rows CBs:** fully resident, no free page. **Count cells:** the sender's multicast source for each round's count is its own never-rewritten cell, so no publish waits for an earlier one to leave L1. |

## Symbol table

| Symbol | Bound | Established by |
|---|---|---|
| `R` | 2240 ≤ R ≤ 8512, R % 32 == 0 | `validate()` |
| `tile_rows = R/32` | 70 ≤ tile_rows ≤ 266 | follows from R |
| `k = pieces_per_row(R)` | k ∈ {2, 3, 4, 6} over the range | the selection rule in `op_design.md` (minimum legal k), with k = 1 removed from the candidates by the verifier (measured; see `verification_report.md`) |
| `w = block_width_tiles = 24/k` | w ∈ {4, 6, 8, 12} | follows from k |
| `max_blocks_per_core = ceil(tile_rows·k/110)` | 1 ≤ max_blocks_per_core ≤ 13, with `max_blocks_per_core·w ≤ ceil(24·tile_rows/110) + 4 ≤ 63` | the selection rule (the L1 budget predicate) |
| `row_slots` | = `max_blocks_per_core` in Phase 0; 1 ≤ row_slots ≤ max_blocks_per_core | host plan (perf lamp L3) |
| `rows_noc0`, `rows_noc1` | 16, 16 (sum 32); on grid column 0 (the ids row senders) 14, 18 (Refinement 4: `SENDER_ROW_SHED_*` = 2, the sender reader sheds 2 rows) | host plan |
| `ids_lookahead` | 1 (1 ≤ ids_lookahead ≤ 6: ≤ 8 ids rounds in flight, kernel `static_assert`) | host plan `IDS_LOOKAHEAD`: rounds the row sender fetches ahead of publishing |
| `reads_in_flight` | `min(READS_IN_FLIGHT = 12, READ_BYTES_IN_FLIGHT = 4096 // piece_bytes)`: 5 / 8 / 10 / 12 at k = 2 / 3 / 4 / 6 (not L1: the NoC read window) | host plan (Refinement 3: the k = 2 congestion cap; Refinement 4: re-measured per k after the cheaper issue path) |

## Total per-core footprint (CBs beyond the io tensors)

```
circular_buffer_bytes(R) = row_slots·w·2048                       # rows (head+tail alias): scales with row_slots and w
                         + R·4 + (max_blocks_per_core + 1)·64     # ids image + round-count cells (Refinement 3)
```

- The term `row_slots·w·2048` is ≤ 63·2048 = 129,024 B. In range the maximum is 120 KiB, at R ∈ [8096, 8512] (5·12 tiles).
- The total in range is ≤ 157,312 B (R = 8512: 122,880 B of rows + 34,048 B of ids image + 384 B of cells), which is
  ≤ the 256 KiB cap (156,928 B at R = 8416). Refinement 3 added up to ~34 KB a core (the image) over Phase 0's 384 B.
- `cb_output_tiles` adds 0: it is the output tensor's own shard region. Its size is `max_blocks_per_core·w·2048` ≤ the interleaved share + 8 KB, by the selection rule.

## Data-movement budget (R1, per call)

| Tensor | DRAM crossings | Why that many | Cross-core traffic added |
|--------|----------------|---------------|--------------------------|
| table (L1) | 0. Each gathered row's piece crosses the NoC once, R·1536 B in total, half on NOC0 and half on NOC1 (column 0: 14 / 18 rows a block, Refinement 4). | No row is re-read: a piece is read straight into its block slot, and all slots stay resident until tilized. | none beyond the gather itself |
| gather_index (DRAM, one page) | ~1 per id (Refinement 3): each grid row's round span is read once by the row's sender, R·4 B plus the ≤ 1 tile row a round where a k-sibling pair straddles two grid rows, ≈ 34 KB at R = 8416 (was R·4·k = 67 KB in 1052 64 B requests; now 10 requests of ≤ 896 B a round) | one sender a grid row reads its row's span and multicasts it along the row (R10 `ids_row_mcast`) | row multicasts: ≈ R·4 B of payload in total (≈ R·4/10 B a grid row, each span multicast once to the row's 10 other cores, on disjoint row paths; ~34 KB at R = 8416), plus one 4 B count multicast a round a row |
| output (L1) | 0. Packed into the core's own shard: 0 NoC bytes. | The output layout is chosen so that the core that tilizes a tile also holds it. | none |

**Totals at R = 8416:**

| Tier | Bytes |
|---|---|
| NoC reads | 12.93 MB of table pieces (6.46 MB per NoC) + ~34 KB of ids from 1 DRAM bank (Refinement 3; was 67 KB) |
| NoC writes | ~34 KB of row multicasts (one span a round a grid row, to its 10 other cores) + 4 B counts |
| DRAM | ~34 KB (ids) |

> Cheapest-traffic split considered: R1 itself (output shards local, rows split over both NoCs). It moves 12.9 MB less NoC traffic than the incumbent's interleaved output (R2), and 6.5 MB less than the whole-row pair forward (R5). Implemented: R1. The only bytes above the minimum are the k-fold ids reads (R·4·(k−1) B ≤ 143 KB). Their cheaper alternative, R7 `ids_sibling_forward`, is deferred because those bytes are ≤ 1.3% of row bytes, and their latency is hidden behind block 0's row reads (ids of block 0 first). R7 stays reachable: a forward fills the same per-RISC ids scratch before the block's rows issue.

> **Implementer measurement (gather_index fetch).** The CB inventory and byte counts above are as built. The premise
> that the k-fold ids reads are hidden behind block 0 did not hold as first scheduled: issuing every block's ids at
> kernel start queued about 800-1050 64 B reads on the one gather_index DRAM bank before block 1 could start
> (ablation with the ids stubbed: R = 4320 4.43 us against 7.99 us). The built schedule fetches block j's ids one block
> ahead (`ids_lookahead = 1`), which keeps that bank's queue in global block order: 11.92 / 7.99 / 4.20 us ->
> 9.47 / 6.69 / 4.19 us at R = 8416 / 4320 / 2272. The ids still cost about 1.5-2.2 us of critical path (R = 2272 /
> 4320, against the ids-stubbed run), mostly the grid's block-0 wave on one bank, so R7 `ids_sibling_forward`
> (k-fold fewer DRAM requests) is now the measured next lever, not a ≤1.3%-of-bytes footnote.

> **Verifier audit (2026-11-04).** Two changes, both applied in place: (1) the ids scratch was sized to every block
> (`max_blocks_per_core + 1` pages) while the lookahead fetch keeps only `ids_lookahead + 1` blocks live, so it is
> now a ring of `ids_slots` pages (≤ 896 B → 192 B a reader); (2) k = 1 is no longer a column-split candidate, a
> measured departure from the coarsest-block default (R = 3296: 6.59 → 4.86 us; R = 6368: 14.97 → 7.47 us). A
> measured finding that bears on R7: at R = 8416, stubbing the ids fetch with local ids made the op *slower*
> (9.66 → 12.8-14.3 us), while R = 2272 / 4320 sped up (4.60 → 2.9, 6.99 → 4.4-4.5 us). At the largest R the ids
> fetch is effectively staggering the grid's first row reads, so a cheaper ids path must come with an explicit issue
> stagger (Refinement 3 in `op_requirements.md`).

> **Refinement 3 (2026-11-04): gather_index fetch topology.** The two per-reader ids rings (2 × 192 B) are replaced by one
> `cb_gather_ids` holding a full gather_index image (≤ 34 KB) that the grid row's sender fills by multicast, one round's
> span per block, behind a count semaphore (`ids_ready`, semaphore 0). DRAM ids traffic drops from R·4·k B in one 64 B
> request per reader per block (220 per round on one bank) to ≈ R·4 B in 10 requests a round. The ring alternative (a
> row-wide reuse handshake) was not taken: the image fits, and no slot is ever rewritten. Measured (device us, mean of
> 5 calls): R = 8416 9.81 → 9.17, 2272 4.61 → 3.70-3.79, 8064 12.36 → 11.30; see the changelog.
