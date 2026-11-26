# L1 Ledger: g6_sharded_to_interleaved

Schema and audits: `.claude/references/l1-footprint-discipline.md`. Block axes (op_design.md, Blocking Model): `tile_rows` (extent `block_tile_rows`), `tile_cols` (extent `block_tile_cols`). No `fp32_dest_acc_en`: the op has no compute kernel and nothing passes through DEST.

## CB table

| CB | Capacity (pages) | Live set | Axis accounting | Page format | Producer | Consumer | Lifetime | Shares with / why not |
|----|------------------|----------|-----------------|-------------|----------|----------|----------|-----------------------|
| `cb_input_shard` | `shard_tile_rows * shard_tile_cols` — the input's sharded buffer itself (`aligned_size_per_bank`), globally allocated, 0 B taken from the op's budget | `block_tile_rows * block_tile_cols` (the whole shard is resident at once; on a short last shard the capacity exceeds the live set by the shard's padding pages, which the buffer already holds — mechanism reason: shard geometry) | `{tile_rows: spans → block_tile_rows (capacity: shard_tile_rows), tile_cols: spans → block_tile_cols (capacity: shard_tile_cols)}` | input dtype, unconverted (Float16_b / Bfp8_b): a dataflow-only payload that never goes through DEST | none in this program (the shard is resident before launch) | writer | whole program | aliases the input tensor's buffer (zero-copy) — this IS the input's storage; no second CB exists to share with |

Inventory before solve (Rule 1, Rule 3): the op has no scratch CB. Pack-into-destination is the whole design — every tile goes from its resident shard straight to its output page; there is no phase boundary, no staging tile, no intermediate. No budget predicate, safety fraction or blocking search exists.

## Non-CB L1 the op uses

| Item | Bytes per core | Scales with | Counted against `L1_FREE_BYTES` |
|------|----------------|-------------|---------------------------------|
| output pages when `buffer = l1` (the output's own L1 banks) | `ceil(num_pages / num_l1_banks) * out_aligned_page_bytes` | `num_pages` (tensor), page bytes (dtype) — not a block knob | yes (contract §8: the output's own L1 pages come out of the 1347776 B) |
| output pages when `buffer = dram` | 0 | — | — |
| input shard | `shard_tile_rows * shard_tile_cols * in_aligned_page_bytes` | shard spec | no (allocated by the producer before the call; outside the budget) |

## Symbol table

| Symbol | Meaning | Bound | Predicate establishing it |
|--------|---------|-------|---------------------------|
| `shard_tile_rows`, `shard_tile_cols` | nominal shard extent in tiles | ≥ 1; the shard already fits in L1 (allocated by the producer) | input is a valid sharded L1 tensor (`validate()` structural checks) |
| `block_tile_rows`, `block_tile_cols` | this core's actual shard extent | `1 ≤ block ≤ shard` | `min()` clamp in the host plan |
| `num_pages` | `R * C` output pages | p2: 160; p1: 1024 (golden INPUTS) | shape of the input |
| `num_l1_banks` | L1 banks of the interleaved allocator | device property (one per worker core on the campaign board, 110 — INFERENCE, tensix.md) | device |
| `out_aligned_page_bytes`, `in_aligned_page_bytes` | page bytes | 2048 (bfloat16), 1088 (bfloat8_b) | dtype; host asserts both equal the page size |
| `L1_FREE_BYTES` | free L1 per core at the call | 1347776 | model environment (contract §5, §8) |

## Total per-core footprint

`op_L1(core) = 0 [CBs] + (buffer == l1 ? ceil(num_pages / num_l1_banks) * out_aligned_page_bytes : 0)`

No term scales with a block knob: the only CB aliases the input, and the output share depends on the tensor and the bank count. Golden INPUTS: p2 (DRAM output) 0 B; p1 (L1 output) `ceil(1024 / 110) * 1088 = 10880 B` at 110 banks (17408 B even at 64 banks), 0.8 % of `L1_FREE_BYTES`. The BB-05 host guard (`op_L1 > L1_FREE_BYTES` → `RuntimeError`) cannot trigger on any golden shape.

## Data-movement budget

Chosen split: owner writes whole shard. P = number of pages, `page_bytes` as above.

| Tensor | DRAM crossings | Why that many | Cross-core traffic added |
|--------|----------------|---------------|--------------------------|
| input (sharded L1) | 0 | every tile is read in place from the L1 of the core that holds its shard (`cb_input_shard`, zero-copy); no NoC read | 0 |
| output, `buffer = dram` (p2) | 1 (327680 B: 160 pages × 2048 B, whole tiles incl. 31 padding rows each) | each page written once by its owner core; whole-tile writes (logical-rows-only writes rejected, regime table) | the writes themselves: 160 transfers from 32 cores |
| output, `buffer = l1` (p1) | 0 | the output lives in L1 banks | 1114112 B over the NoC (1024 pages × 1088 B), each page once, from 103 cores |

Totals per tier — p2: DRAM 327680 B written, NoC 327680 B (the same writes), L1 reads 327680 B local. p1: DRAM 0 B, NoC 1114112 B, L1 reads 1114112 B local.

> Cheapest-traffic split considered: owner writes whole shard — 0 B above the minimum on every tier (each tile read locally once, one NoC crossing per tile). Implemented: owner writes whole shard. The implemented split is the cheapest; every other split adds at least one NoC crossing per moved tile (regime table: gather, rebalance, DRAM staging — all rejected).
