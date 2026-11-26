# L1 Ledger: g6_interleaved_to_sharded

Schema and audits: `.claude/references/l1-footprint-discipline.md`. Blocking decisions: `op_design.md` → Blocking Model.

The op allocates **no CB, no semaphore and no scratch buffer**. The only L1 it touches is the output shard: the entry point allocates it and the reader's NoC read responses write it in place. That shard counts inside the BB-05 bound (`docs/explorations/g6_interleaved_to_sharded/sources/local/contract_r3.md:70,75`), so it is listed below as an accounting row. It is not a CB.

Block axes (from the Blocking Model): `tile_rows` (folded, extent `block_rows`), `tile_cols` (extent `block_cols`), `rows_in_tile` (extent `transfer_rows`).

| CB | Capacity (pages) | Live set | Axis accounting | Page format | Producer | Consumer | Lifetime | Shares with / why not |
|----|------------------|----------|-----------------|-------------|----------|----------|----------|-----------------------|
| output shard (tensor buffer, not a CB; allocated by the entry point from `memory_config`) | `shard_rows_tiles * shard_cols_tiles` (caller-pinned shard shape, ≥ `block_rows * block_cols`) | `block_rows * block_cols` pages written by this program. Slots beyond the clipped block exist only on a clipped last shard and are never touched | `{tile_rows: spans → shard_rows_tiles, tile_cols: spans → shard_cols_tiles, rows_in_tile: spans → 32 (whole tile allocated; only transfer_rows written)}` | bfloat16 tile (`Float16_b`), 2048 B. It never passes through DEST, so audit 2 does not apply | reader (NoC read responses) | none in this program (the model's next op) | from allocation through the rest of the model step | It is the op's result, so it cannot share. Every former staging candidate is folded into it: the DRAM read lands at the final slot (pack-into-destination, Rule 3.1) |
| (no CB) | 0 | 0 | `{tile_rows: streams → no buffer, tile_cols: streams → no buffer, rows_in_tile: streams → no buffer}` | — | — | — | — | Inventory minimised to zero. A CB aliased on the output shard was rejected: it would only provide an address the RT arg already provides, and it would add a CB config entry and a push with no consumer |

## Symbol table

| Symbol | Bound | Predicate that establishes it |
|--------|-------|-------------------------------|
| `shard_rows_tiles` | ≥ 1. `shard_rows_tiles * shard_cols_tiles * page_bytes ≤ 1347776` | `memory_config.shard_spec.shape[0] % 32 == 0` (Validation). For WIDTH, `shard_rows_tiles ≥ R`. The BB-05 host check |
| `shard_cols_tiles` | ≥ 1, same product bound | `shard_spec.shape[1] % 32 == 0`. For HEIGHT, `shard_cols_tiles ≥ C`. The BB-05 host check |
| `block_rows`, `block_cols` | `≤ shard_rows_tiles`, `≤ shard_cols_tiles` | `min` clipping in `_plan` |
| `page_bytes` | 2048 | bfloat16 32x32 tile (dtype and tile validated) |
| `transfer_rows` | 1..32 rows per page, written as ≤ 2 segments, each ≤ `page_bytes` | segment table in `_plan`. It sizes NoC transfers, not storage |

## Footprint

Per active core: `L1_op = shard_rows_tiles * shard_cols_tiles * page_bytes` (output shard) `+ 0` (op buffers).

| Case | Output shard | Op buffers | Total | BB-05 bound |
|------|--------------|------------|-------|-------------|
| p1 `(1,5120)` → width `(32,160)` | 5 × 2048 = 10240 B | 0 | 10240 B | 1347776 B |
| p2 `(1,1,1,128)` → height `(32,128)` | 4 × 2048 = 8192 B | 0 | 8192 B | 1347776 B |

What scales with which knob: the only term scales with the caller-pinned shard shape. No op knob (`partial_tile_transfer`, `reads_in_flight`) changes L1.

## Data-movement budget

| Tensor | DRAM crossings | Why that many | Cross-core traffic added |
|--------|----------------|---------------|--------------------------|
| `input_tensor` | 1, valid bytes only: full tile rows 2048 B/page; partial tile rows 2 × `align_up(rows*32, 64)` B/page. p1: 160 × 128 = 20480 B. p2: 4 × 128 = 512 B | owner-fetch: each page is read once by the core that owns its shard. The valid-rows transfer skips the padding rows the contract excludes from the result. The 64 B DRAM read granule and the two-face split of a row set the excess over 64 logical B/tile | 0 |
| output | 0 (lives in L1) | written once, by the read response, at its final slot | 0 (every page lands in the L1 of the core that read it) |

Totals per call: DRAM → L1: p1 20480 B, p2 512 B. Core-to-core: 0 B. L1 → L1: 0 B.

> Cheapest-traffic split considered: owner-fetch with the valid-rows transfer (split by output shard, each owner reads its own pages, padding rows skipped) — p1 20480 B DRAM / 0 B NoC, against 327680 B DRAM / 0 B NoC for whole pages and 327680 B DRAM + 317440 B NoC for one mover core. Implemented: the same split. It is the cheapest-traffic split, so nothing is deferred for traffic. Reader-and-forward (regime C) is deferred only for issue parallelism, which no production shape needs.
