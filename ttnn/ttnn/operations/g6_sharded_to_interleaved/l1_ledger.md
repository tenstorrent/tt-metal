# L1 Ledger: g6_sharded_to_interleaved

Schema and audits: `.claude/references/l1-footprint-discipline.md`. Block axes from `op_design.md` → Blocking Model: `tile_rows`, `tile_cols`, `lane`.

**Inventory.** The op owns no scratch, staging or intermediate buffer. The store reads each tile in place from the resident shard and writes it straight to its output page (Rule 3, pattern 1: the destination is the output page itself). The only CBs are two zero-copy views of the input shard, one per writer lane. They exist only so that each lane kernel has its own CB.

| CB | Capacity (pages) | Live set | Axis accounting | Page format | Producer | Consumer | Lifetime | Shares with / why not |
|----|------------------|----------|-----------------|-------------|----------|----------|----------|-----------------------|
| `cb_input_shard_lane0` | `shard_tiles_h · shard_tiles_w` (= `aligned_size_per_bank / page_bytes`, set by `cb_descriptor_from_sharded_tensor`). Globally allocated **on the input buffer**: 0 B of op L1 | the whole shard (`rows_k · cols_k` valid tiles), resident before the call and owned by the producer. Lane 0 reads `ceil(rows_k·cols_k / L)` of its pages in place | `{tile_rows: spans → shard_tiles_h, tile_cols: spans → shard_tiles_w, lane: streams → lane 0's sub-range of ceil(n_k/L) tiles}` | input dtype: Float16_b (bfloat16) / Bfp8_b (bfloat8_b). A dataflow-only payload that never passes through DEST, so no DEST-width question exists (audit 2 n/a) | none — the resident shard, written by the program that produced `input_tensor`; never pushed | writer lane 0 | whole call | **Shares the same bytes with `cb_input_shard_lane1`** (both backed on the input buffer at offset 0). Two CB entries rather than one so each consumer kernel owns its CB (one-consumer rule). The alias costs no L1. Capacity = live set (the shard), so audit 1 holds in both directions |
| `cb_input_shard_lane1` | same expression; same buffer; 0 B of op L1 | the whole shard, resident. Lane 1 reads the remaining `floor`-side pages of its sub-range | `{tile_rows: spans → shard_tiles_h, tile_cols: spans → shard_tiles_w, lane: streams → lane 1's sub-range}` | same as lane 0 | none — resident shard | writer lane 1 | whole call | Shares the same bytes with `cb_input_shard_lane0` (same reason). Absent when `num_writer_lanes = 1` |

Non-CB L1 that counts against the per-core budget (spec D16 / BB-05: counted before the output is allocated):

| Item | Per-core bytes | Scales with | When |
|------|----------------|-------------|------|
| Output pages of an L1-interleaved output | `ceil(P / N_L1_banks) · page_bytes` (the same range is reserved in every L1 bank) | P, `page_bytes` | `memory_config.buffer_type == L1` only (p1). 0 for DRAM output (p2) |
| Input shard | `shard_tiles_h · shard_tiles_w · page_bytes` | shard spec | **Outside the budget**: already allocated by the producer when the call starts (spec) |

### Symbol table

| Symbol | Meaning | Bound | Predicate establishing it |
|--------|---------|-------|---------------------------|
| `shard_tiles_h`, `shard_tiles_w` | input shard shape in tiles | p2: 1 x 5; p2 other: 1 x 3; p1: 10 x 1; p1 other: 11 x 1 | read from `input_tensor.shard_spec`; validation requires whole tiles. No op buffer depends on them (the CBs alias the producer's shard) |
| `rows_k`, `cols_k`, `n_k` | valid extent of shard k (ragged last shard) | `≤ shard_tiles_h`, `≤ shard_tiles_w`, `n_k ≤ shard_tiles_h·shard_tiles_w` | `min` formulas in `op_design.md` → Work Distribution |
| `L` = `num_writer_lanes` | writer lanes per core | {1, 2}; Phase 0 = 2 | host constant `NUM_WRITER_LANES` |
| `page_bytes` | aligned tile page | 2048 (bfloat16), 1088 (bfloat8_b) | TARGET dtypes |
| `P` | output pages = R·C | ≤ 1024 over the golden INPUTS (p1); 160 (p2) | INPUTS; no op buffer scales with P, only the L1 output share does |
| `N_L1_banks` | L1 interleaved bank count of the device (`NUM_L1_BANKS`) | ≥ 103 on the production Blackhole, so `ceil(1024 / N_L1_banks) ≤ 10` (INFERENCE: one L1 bank per worker core, 110 worker cores per spec section 3; more banks only shrink the share) | device constant |

### Total per-core footprint

`op_L1 = 0` (CB entries alias the input shard) `+ [buffer == L1] · ceil(P / N_L1_banks) · page_bytes`

- p2 (DRAM output): **0 B**.
- p1 (L1 output): ≤ `10 · 1088 = 10880 B` per core, 0.8 % of the 1347776 B budget.
- No term scales with a block knob. `L` changes only the number of zero-byte CB entries. The single output-share term scales with P and `page_bytes` alone.
- No budget predicate, safety fraction or blocking search exists. No shape in INPUTS approaches the bound, so the op never refuses a shape on L1 grounds.

## Data-movement budget

| Tensor | DRAM crossings | Why that many | Cross-core traffic added |
|--------|----------------|---------------|--------------------------|
| `input_tensor` (sharded L1) | 0 | Resident shard read in place through the zero-copy CB. It never crosses the NoC as a read and never touches DRAM | 0 |
| output, DRAM interleaved (p2: 160 pages, 327680 B) | 1 (write) | Each page written exactly once, by the core holding its source tile | none beyond the write itself (each tile's one NoC crossing *is* the DRAM write) |
| output, L1 interleaved (p1: 1024 pages, 1114112 B) | 0 | L1 destination | 1114112 B: each page crosses the NoC once to its L1 bank (a loopback where the bank is the source core) |

Totals per call:

| Case | DRAM read | DRAM write | NoC | Core-local L1 reads |
|------|-----------|------------|-----|---------------------|
| p2 | 0 | 327680 B | 327680 B (160 writes from 32 cores, 64 issuing RISCs) | 327680 B |
| p1 | 0 | 0 | 1114112 B (1024 writes from 103 cores, 206 issuing RISCs) | 1114112 B |

> Cheapest-traffic split considered: the input's own shard split (owner writes) — every tile crosses the NoC once, zero NoC or DRAM reads, the output written once. Implemented: the same split. It is the cheapest; every alternative (gather, redistribution, DRAM staging) adds P·`page_bytes` or more per tier, and none is deferred.
