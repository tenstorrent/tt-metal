# L1 Ledger: g6_permute (regime `rows_tile`, Phase 0)

Schema and audits: `.claude/references/l1-footprint-discipline.md`. Block axes are those of
`op_design.md` "Axes": n, a, b, w. No compute kernel runs, so `fp32_dest_acc_en` does not
exist. Every page is a bfloat16 dataflow payload that never passes through DEST.

| CB | Capacity (pages) | Live set | Axis accounting | Page format | Producer | Consumer | Lifetime | Shares with / why not |
|----|------------------|----------|-----------------|-------------|----------|----------|----------|-----------------------|
| `cb_input_rows` (idx 0, 2048 B pages) | `depth_in * a_tile_rows` | `depth_in` blocks x `a_live(ta)` slot pages. In each page only the block's `block_b_count` rows are live (`2 * block_b_count` segments of 32 B), each at its own intra-tile offset | `{n: streams -> 1 image per block, a: spans -> a_tile_rows slots, b: streams -> block_b_count rows inside each slot, w: streams -> 1 tile column}` | bfloat16 (input dtype; dataflow-only payload) | reader | writer | `load_rows_block` -> `exchange_rows_block` of one block | **Not shared** with `cb_output_staging`: both are live together during `exchange_rows_block` (read one, write the other). **Capacity above live bytes**: a slot is a whole tile page because each row must land at its own tile offset to satisfy the Blackhole DRAM-read mod-64 rule (`noc_parameters.h:379`); compacting would save at most `depth_in * a_tile_rows * (2048 - 64)` B (<= 124 KiB), which no shape needs against the 1347776-B budget. **Depth**: `depth_in = min(2, max_blocks_per_core)`, so a single-block core holds one slot set |
| `cb_output_staging` (idx 1, 2048 B pages) | `block_b_rows` (`depth_out = 1`) | `block_b_count` pages, each with `a_live(ta)` live rows; rows `a_live..31` are stale (output padding, unspecified) | `{n: streams -> 1, a: spans -> a_live rows inside each page, b: spans -> block_b_rows pages, w: streams -> 1}` | bfloat16 | writer | writer (private scratch) | `exchange_rows_block` -> `store_tiles_block` of one block; refilled only after `noc_async_writes_flushed()` | **Not shared** with `cb_input_rows`: concurrent lifetime (above). **Not removable**: the copy destination must be a whole output page in L1 so the page leaves in one write (`rows_tile_scatter`, which drops it, is deferred in the regime table). **Not aliased** onto the input slots: the exchange reads slots while writing staging |

## Symbol table

| Symbol | Definition | Bound | Predicate establishing the bound |
|--------|------------|-------|----------------------------------|
| `a_tile_rows` | `min(32, A)` | <= 32 | tile geometry: an a-tile is 32 rows (regime `rows_tile`: TILE layout) |
| `a_live(ta)` | `min(32, A - 32*ta)` | <= `a_tile_rows` | same |
| `block_b_rows` | `min(32, B, units_per_core_g1)` | <= 32 | mechanism cap "one b-tile per block" (block rule) |
| `block_b_count` | actual run of a block | <= `block_b_rows` | block rule |
| `depth_in` | `min(2, max_blocks_per_core)` | <= 2 | Phase 0 depth knob |
| page size | `ttnn.tile_size(bfloat16)` = 2048 B | constant | dtype bfloat16 (Phase 0 SUPPORTED) |

## Footprint per core

`L1(core) = (depth_in * a_tile_rows + block_b_rows) * 2048 B`

- The `depth_in * a_tile_rows` term scales with the depth knob and with the a-tile (<= 32 slots).
- The `block_b_rows` term scales with the b knob.
- Upper bound for every shape of the cell: `(2*32 + 32) * 2048 = 196608 B`, below the 1347776 B the model leaves.
- Phase 0 is DRAM-only, so no output page sits in L1.
- No refusal is needed (BB-05).

| Production shape | `a_tile_rows` | `block_b_rows` | `depth_in` | L1 per core |
|------------------|---------------|----------------|------------|-------------|
| 301 (1, 1, 16, 64) | 1 | 1 | 1 | 4096 B |
| 302 (1, 16, 1, 64) | 16 | 1 | 1 | 34816 B |
| 303 (1, 6, 16, 64) | 6 | 1 | 1 | 14336 B |
| 304 (1, 16, 6, 64) | 16 | 1 | 1 | 34816 B |
| 305 (1, 1, 8, 64) | 1 | 1 | 1 | 4096 B |
| 306 (1, 8, 1, 64) | 8 | 1 | 1 | 18432 B |
| 307 (1, 6, 8, 64) | 6 | 1 | 1 | 14336 B |
| 308 (1, 8, 6, 64) | 8 | 1 | 1 | 18432 B |

## Data-movement budget (split S2, `rows_tile`)

| Tensor | DRAM crossings | Why that many | Cross-core traffic added |
|--------|----------------|---------------|--------------------------|
| x | 1, live rows only: `N*A*B*W*2` B in `<= 4` reads per (live input tile, block) | each live input row belongs to exactly one output tile, so to exactly one core's b-run; padding rows are never addressed | none |
| out | 1, whole tiles: `N*B*tensor_a_tiles*tensor_w_tiles*2048` B in one write per tile | each output tile is written once, whole, by its owner from L1 staging | none |
| (core-local) | — | the row exchange: `2*N*A*B*tensor_w_tiles` copies of 32 B, each live row once | none |

Per production shape (one block per core):

| Shape | x bytes / read transactions | out bytes / write transactions | local copies (32 B) | cores |
|-------|-----------------------------|--------------------------------|---------------------|-------|
| 301 (1, 1, 16, 64) | 2048 / 64 | 65536 / 32 | 64 | 32 |
| 302 (1, 16, 1, 64) | 2048 / 64 | 4096 / 2 | 64 | 2 |
| 303 (1, 6, 16, 64) | 12288 / 384 | 65536 / 32 | 384 | 32 |
| 304 (1, 16, 6, 64) | 12288 / 384 | 24576 / 12 | 384 | 12 |
| 305 (1, 1, 8, 64) | 1024 / 32 | 32768 / 16 | 32 | 16 |
| 306 (1, 8, 1, 64) | 1024 / 32 | 4096 / 2 | 32 | 2 |
| 307 (1, 6, 8, 64) | 6144 / 192 | 32768 / 16 | 192 | 16 |
| 308 (1, 8, 6, 64) | 6144 / 192 | 24576 / 12 | 192 | 12 |

Totals per tier (formula):
- DRAM read: `N*A*B*W*2` B.
- DRAM write: `N*B*tensor_a_tiles*tensor_w_tiles*2048` B.
- Cross-core: 0.
- Core-local: `N*A*B*W*2` B in 32-B copies.

> Cheapest-traffic split considered: `rows_tile_scatter` (S4: same row-span reads, 32-B
> output segments written straight from the input slots). DRAM bytes written fall to the live
> bytes (303: 65536 -> 12288 B; DRAM read unchanged), and the 2R core-local copies disappear.
> DRAM write transactions rise from one per output page to `2*a_live` per page (303: 32 -> 384).
> Implemented: `rows_tile` (S2). Deferred because crossings are equal (one per tensor) and the
> bytes saved are output padding inside a page that `rows_tile` writes in one transaction
> anyway. The scatter multiplies DRAM write transactions per page by up to 64, the transfer
> shape the `tile_reorder` entry measured slower. `rows_tile` serves every shape the scatter
> would serve, and the structure stays reachable: same reader, same unit split, with the
> writer's exchange + tile write swapped for segment writes.
>
> Fewest DRAM transactions: S1 (whole-band reads, whole-tile writes; 303: 44). It is `rows_tile`
> at `block_b_rows = b_live`, reached by the plan whenever a core's units cover a whole b-tile
> run, and kept as the grid-synchronization lamp for the Qwen shapes.
