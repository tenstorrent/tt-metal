# L1 Ledger: g6_interleaved_to_sharded

Schema and audits: `.claude/references/l1-footprint-discipline.md`. Blocking decisions: `op_design.md` → Blocking Model.

Block axes accounted in every row: `tile_row`, `tile_col`, `image`, `row_in_tile`, `col_in_tile`.

## Buffers

The op adds **no circular buffer and no scratch L1**. The only L1 it writes is the output tensor's own shard. It is listed here because BB-05 (D15 of the bundle) counts it inside the per-core bound, and because it is the destination every read lands in.

| CB | Capacity (pages) | Live set | Axis accounting | Page format | Producer | Consumer | Lifetime | Shares with / why not |
|----|------------------|----------|-----------------|-------------|----------|----------|----------|-----------------------|
| `y_shard` (output tensor's L1 shard; a tensor buffer, not a CB; allocated by the entry point from `memory_config`) | `shard_tile_rows * shard_tile_cols` pages of `tile_bytes`. This is the caller's placement, not an op choice. p1: 1*5 = 5 pages (10,240 B); p2: 1*4 = 4 pages (8,192 B). | `block_tile_rows * block_tile_cols` tiles written per core (≤ capacity; = capacity in both production cases). A partial last shard leaves the slots past the tile grid unwritten. | `{tile_row: spans → block_tile_rows; tile_col: spans → block_tile_cols; image: spans → folded into tile_row; row_in_tile: spans → span_bytes(r) of each tile written (rows past the span stay unwritten padding); col_in_tile: spans → 32 (whole face rows)}` | bfloat16 tile (Float16_b), the input's own bytes. No DEST is involved (no compute kernel), so audit 2 has nothing to compare. | reader kernel(s) (NCRISC; + BRISC when `issuers_per_core = 2`), by NoC read responses | none inside this program; the model's next program reads it in place | from the op's call to the consumer's end (it is the result) | none: it is the result itself, and no op buffer exists to share with. Inventory is 0 because each NoC read lands at its final address (Rule 3, pattern 1: pack into the destination); a staging CB would hold the same bytes a second time. |

Audit notes:

| Audit | Finding |
|-------|---------|
| 1. Capacity vs live set | Over: capacity exceeds the live set only for a partial last shard; the gap is the caller's shard geometry (placement), not an op choice. Under: the live set spans `tile_row`, `tile_col`, and capacity scales with both. |
| 2. Page format vs DEST width | n/a: no DEST on the path; page format = input format (bit-exact copy). |
| 3. Disjoint lifetimes without sharing | n/a: one buffer. |
| 4. Bounds and closed form | every symbol below is bounded; the total is closed-form. |

## Symbol table

| Symbol | Meaning | Bound | Predicate establishing the bound |
|--------|---------|-------|----------------------------------|
| `tile_bytes` | bytes of one bf16 32x32 tile page | 2048 | `validate()`: dtype bfloat16, TILE, 32x32 tile |
| `shard_tile_rows`, `shard_tile_cols` | `shard_shape / 32` | `shard_tile_rows * shard_tile_cols * tile_bytes ≤ L1_BUDGET_BYTES = 1347776` | `validate()` BB-05 check (ValueError otherwise); production: 5 and 4 pages |
| `block_tile_rows`, `block_tile_cols` | one issuer's rectangle | `≤ shard_tile_rows`, `≤ shard_tile_cols` | `min()` clamps in the host work split |
| `span_bytes(r)` | bytes read per tile of tile row `r` | `≤ tile_bytes` | regime selection function (op_design.md, Regimes) |
| `issuers_per_core` | reader kernels per core | {1, 2} | `KNOBS` value set |
| `partial_tile_read` | span shape of a partial tile row | {`valid_rows`, `face_rows`, `whole_tile`} | `KNOBS` value set; every span ≤ `tile_bytes` (kernel `static_assert`) |

## Footprint

```
L1_op(core) = shard_tile_rows * shard_tile_cols * tile_bytes     (output shard; caller's placement)
            + 0                                                  (op-added CBs / scratch)
p1: 10,240 B per core on 32 cores;  p2: 8,192 B on 1 core   (bound 1,347,776 B)
```

No term scales with a knob. Block extents, `issuers_per_core` and `partial_tile_read` change what is written and how, never what is allocated. Kernel-config region per core: fixed-count CT args + 2 common RT args (input, output address) + one fixed-count rectangle of per-core RT args, independent of the tile count (BB-04).

## Data-movement budget

Chosen split: owner-fetch along the target's shard axis (width: tile columns; height: tile rows), with the valid-row span on partial tile rows.

| Tensor | DRAM crossings | Why that many | Cross-core traffic added |
|--------|----------------|---------------|--------------------------|
| `x` (input, DRAM interleaved) | 1 per page. p1: 160 reads x 544 B = 87,040 B (327,680 B tiled; 10,240 B logical). p2: 4 x 544 B = 2,176 B. A fully valid tile row costs 2048 B per page. | Each page is read once, directly into its final L1 address. Nothing is staged, so nothing is re-read. | 0 |
| `y` (output, L1 sharded) | 0 | It lands in the owner's L1 as the read response and never touches DRAM | 0: the read response terminates on its owner; no core writes another core's L1 |

Totals per tier:

| Case | DRAM bytes | NoC bytes (read responses) | Core-to-core bytes |
|------|------------|----------------------------|--------------------|
| p1 | 87,040 | 87,040 into 32 cores (2,720 B per core, 5 requests per core) | 0 |
| p2 | 2,176 | 2,176 into 1 core (4 requests) | 0 |

> Cheapest-traffic split considered: owner-fetch along the target's shard axis — 1 DRAM crossing per page, 0 core-to-core bytes. Implemented: the same split. The implemented split is the cheapest. Inside it, the two-span transfer shape (perf lamp L2) moves fewer bytes, reaching the logical floor (p1: 10,240 B, i.e. −76,800 DRAM bytes) at 2 requests per partial tile. That is a transfer shape within the chosen split, measured as a lamp, not a different split.

Implemented knob alternatives (same split, no L1 change; p1 device kernel duration, 5-call medians, Blackhole):

| `partial_tile_read` | DRAM bytes (p1) | Requests per core | 1 issuer | 2 issuers |
|---------------------|-----------------|-------------------|----------|-----------|
| `valid_rows` (default) | 87,040 | 5 | 1096 ns | 1031 ns |
| `face_rows` (lamp L2) | 10,240 | 10 | 1280 ns | 1121 ns |
| `whole_tile` | 327,680 | 5 | 1675 ns | 1495 ns |

Read-ablated floor: 543 ns, of which ~450 ns is first-to-last core launch stagger. `face_rows` reaches the
logical byte floor but loses on request count; `valid_rows` stays the default.
