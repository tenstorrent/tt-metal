# L1 Ledger: g6_reshard

Schema and audits: `.claude/references/l1-footprint-discipline.md`. Blocking decisions: `op_design.md` → Blocking Model.

The op allocates **no circular buffer and no scratch**. Every NoC read lands directly in out's shard (Rule 3.1, pack into the destination), and every source tile is read where it lives. The one L1 allocation charged to the op is out's own shard: the lead counts it against the free L1 (`docs/explorations/g6_reshard/sources/local/contract_r3.md:68`, D11). in0's shard belongs to the model and is outside the budget.

Block axes (from the Blocking Model): `tile_rows` (extent `block_height_tiles`, split over `issuers_per_core` by rows) and `tile_cols` (extent `block_width_tiles`).

| CB | Capacity (pages) | Live set | Axis accounting | Page format | Producer | Consumer | Lifetime | Shares with / why not |
|----|------------------|----------|-----------------|-------------|----------|----------|----------|-----------------------|
| out shard: out's tensor buffer, not a CB, reached by address | `block_height_tiles · block_width_tiles` slots of `aligned_page_bytes` each, set by `memory_config`'s shard spec (P: 8·4 = 32; O1: 8·5 = 40) | all slots: each holds one result tile from its write until the next program consumes it | `{tile_rows: spans → block_height_tiles, tile_cols: spans → block_width_tiles}` | Bfp8_b as stored, 1088 B. No DEST anywhere on the path, so the DEST-width audit does not apply | issuer 0 (rows `[block_row_begin, block_row_begin + ⌈rows/2⌉)`) and issuer 1 (the remaining rows): disjoint rows, NoC reads, no CB handshake | none inside the program (the next program reads out) | from allocation (host, before dispatch) through the end of the program and beyond: it is the result | none. It is the output itself. It cannot share with in0's shard because out must be a new tensor (rule 1, R8), and nothing else exists to share with |
| (no other buffer) | 0 | 0 | `{tile_rows: streams → 0, tile_cols: streams → 0}`: no staging, because sources are read in place | — | — | — | — | Every phase boundary is pack-into-destination. No intermediate exists to alias or fold |

## Symbol table

| Symbol | Meaning | Bound | Predicate establishing it |
|--------|---------|-------|---------------------------|
| `block_height_tiles` | out shard height / 32 | `≥ 1`, `≤ tensor_height_tiles` rounded up to the shard | `memory_config` validation: shard dims multiples of 32; BB-05 refusal below |
| `block_width_tiles` | out shard width / 32 | `≥ 1`, `≤ tensor_width_tiles` rounded up to the shard | same |
| `aligned_page_bytes` | tile slot stride, `buffer_aligned_page_size()` | 1088 for bfloat8_b (Phase 0 SUPPORTED dtype) | `validate()`: dtype ∈ SUPPORTED; host asserts in0 and out report the same value |
| `L1_FREE_BYTES` | L1 per core the model leaves to the op | 1347776 | lead's environment parameter (`contract_r3.md`, section 5) |
| out shard bytes | `block_height_tiles · block_width_tiles · aligned_page_bytes` | `≤ L1_FREE_BYTES` | host refuses with `RuntimeError` above it (BB-05). Golden INPUTS max: 43520 B (O1) |

## Total per-core footprint

`out_shard_bytes = block_height_tiles · block_width_tiles · aligned_page_bytes` (+ 0 scratch, + 0 CB)

- P / I0 / I3 / I5: 8 · 4 · 1088 = **34816 B** on each of 64 cores
- O1: 8 · 5 · 1088 = **43520 B** on each of 56 cores
- Scales with: the out shard shape only, which `memory_config` sets and the op does not tune. `issuers_per_core` and every lamp in the Blocking Model add **0 B** of L1.
- Outside the L1 data budget: each issuer's RT args (O(1) per core) and in0's accessor CT args (≤ in0's core count) live in the kernel-config region (BB-04).

## Data-movement budget

Split: PULL (the out block holder issues reads).

| Tensor | DRAM crossings | Why that many | Cross-core traffic added |
|--------|----------------|---------------|--------------------------|
| in0 | 0 | resident in L1 before the program; each tile is read once, where it lives | each tile whose source core differs from its target core crosses the NoC once: P 1880 tiles = 2045440 B (168 tiles = 182784 B copied on their own core by a self-aimed read); I0 0 B (2048 local); I3 1949696 B (256 local); I5 2228224 B (0 local); O1 2181440 B (43 local) |
| out | 0 | written in place, once per tile, by the core that holds it | none beyond in0's line (the same bytes, counted once) |

Totals per tier for the production pair P: **DRAM 0 B; cross-core NoC 2045440 B (1024 runs, 256 core pairs); core-local 182784 B.** No handshake traffic: no semaphore, no write ack, no multicast.

> Cheapest-traffic split considered: PULL, PUSH and BALANCED tie (each non-local tile crosses the NoC once, DRAM 0 B; PUSH and BALANCED add write acks). Implemented: PULL, which is a cheapest-traffic split. PUSH and BALANCED are `deferred` rows of the regime table because PULL already covers every placement pair they would serve. Their delta is 0 B on every tier, and they change only which end issues each transfer.
