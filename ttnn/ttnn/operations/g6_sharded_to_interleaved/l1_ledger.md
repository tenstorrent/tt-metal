# L1 Ledger: g6_sharded_to_interleaved

Schema and audits: `.claude/references/l1-footprint-discipline.md`. Block axes from `op_design.md` → Blocking Model: `tile_row`, `tile_col`, tile interior, device.

| CB | Capacity (pages) | Live set | Axis accounting | Page format | Producer | Consumer | Lifetime | Shares with / why not |
|----|------------------|----------|-----------------|-------------|----------|----------|----------|-----------------------|
| `cb_input_shard` | `shard_tile_rows · shard_tile_cols` pages of `page_size`, globally allocated on `input_tensor`'s own shard buffer (`ttnn.cb_descriptor_from_sharded_tensor`, total size = the input's bank size). **0 bytes of new L1.** | `shard_tile_rows · shard_tile_cols`: the whole shard is resident before launch and stays resident (it is the caller's tensor); the op reads one page at a time from it and keeps nothing else live | `{tile_row: spans → shard_tile_rows (block uses block_tile_rows ≤ it), tile_col: spans → shard_tile_cols (block uses block_tile_cols ≤ it), tile interior: spans → one whole page, device: n/a (per-device program; every chip has its own copy at the same address)}` | input dtype: `Float16_b` (bf16) / `Bfp8_b` (bf8b). A dataflow-only payload that never passes through DEST, so audit 2 does not apply; the bytes move unchanged | none in this program: resident, written by the program that produced `input_tensor` | writer | whole program; read-only; no `wait_front` / `pop_front` / `push_back` | Aliases `input_tensor`'s shard buffer itself: the capacity is the caller's allocation, not the op's. No other CB exists to share with. The output is written directly to its interleaved pages, never staged, so neither the op nor the inventory adds a buffer. |

Capacity vs live set (audit 1): equal. The shard is the input itself, and the op allocates none of it. The CB costs 0 bytes and spans the axes it is sized on, so no axis is over- or under-scaled.

Inventory (Rule 1, Rule 3): one CB, allocating nothing. Pattern 1 (pack into the destination) is realized in its strongest form: the source writes straight into the output's pages, so no staging buffer and no phase boundary exists. Patterns 2-4 have nothing to act on (no phase, no intermediate, no accumulator).

## Symbol table

| Symbol | Meaning | Bound | Predicate establishing it |
|--------|---------|-------|---------------------------|
| `shard_tile_rows`, `shard_tile_cols` | nominal shard extents in tiles, `shard_spec.shape / 32` | the caller's shard already fits L1: it is allocated before the call, and the 1347776 B model budget excludes the input shards | `validate()` checks 4-6 (sharded, L1, whole tiles) |
| `block_tile_rows`, `block_tile_cols` | this core's block (= its clipped shard) | `≤ shard_tile_rows`, `≤ shard_tile_cols` | clamp in the Mechanism caps table |
| `page_size` | `ttnn.tile_size(dtype)` | ∈ {2048, 1088} B, `≤ NOC_MAX_BURST_SIZE = 16384` B | dtype ∈ TARGET (`SUPPORTED` gate); host assert |
| `tensor_tile_rows · tensor_tile_cols` | output pages | L1 output: `ceil(pages / n_l1_banks) · page_size ≤ 1347776` B, else a host `RuntimeError` (BB-05). INPUTS maximum: 1024 × 1088 = 1,114,112 B, inside the bound even at one bank | host check on a cache miss, `buffer == "l1"` |
| `n_l1_banks` | L1 banks of the device | ≥ 1 | device L1 memory view |

## Total per-core footprint

```
op_allocated_L1(core)   = 0                                                    # cb_input_shard aliases the input
model_budget_used(core) = op_allocated_L1(core)
                        + [buffer == "l1"] · ceil(tensor_tile_rows · tensor_tile_cols / n_l1_banks) · page_size
```

No term scales with a block knob, because no block-sized buffer exists. The only term is the output's own L1 pages, which the contract charges to the model budget. p1: `ceil(1024 / n_l1_banks) · 1088` B, e.g. 10,880 B at 110 banks. p2 (DRAM output): 0 B.

## Data-movement budget

| Tensor | DRAM crossings | Why that many | Cross-core traffic added |
|--------|----------------|---------------|--------------------------|
| `input_tensor` | 0 | resident in L1 shards; each slot is the source of its owner's single write, read in place (no NoC read, no staging copy) | 0 |
| output, p2 (1, 5120) bf16 → DRAM | 1 (written once: 160 × 2048 = 327,680 B) | each page written once by the core that owns its tile | 0 core→core; 327,680 B core→DRAM |
| output, p1 (1, 128, 256, 32) bf8b → L1 | 0 | L1 interleaved destination | 1,114,112 B core→core (1024 × 1088 B, one write per page) |

Totals per tier:

| Case | DRAM read | DRAM write | NoC (core→core) | Local L1 reads (source side of the writes) |
|------|-----------|------------|-----------------|--------------------------------------------|
| p2 | 0 | 327,680 B | 0 | 327,680 B |
| p1 | 0 | 0 | 1,114,112 B | 1,114,112 B |

> Cheapest-traffic split considered: the physical-shard (owner) split — one NoC crossing per tile, zero reads, zero DRAM reads; `destination_pull` ties at one crossing only for L1 outputs (delta 0 B per tier there) and does not exist for DRAM outputs. Implemented: the owner split (`owner_writes`). **The implemented split is also the cheapest.** `destination_pull` is a `rejected` regime row (superseded by `owner_writes` across both buffer types), not a deferred one.
