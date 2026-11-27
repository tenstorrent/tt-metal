# L1 Ledger: g6_sharded_to_interleaved

Schema and audits: `.claude/references/l1-footprint-discipline.md`. Block axes (from `op_design.md`, Blocking Model): `tile_rows` (extent `block_extent_rows`) and `tile_cols` (extent `block_extent_cols`); the block is the core's whole resident input shard.

Inventory: two CB indices, both bound zero-copy to the input's sharded buffer. No staging, scratch, scaler or output CB exists: the store sources each tile from its shard slot and writes it straight to its output page, so the only phase boundary (shard → output page) needs no storage.

## CBs

| CB | Capacity (pages) | Live set | Axis accounting | Page format | Producer | Consumer | Lifetime | Shares with / why not |
|----|------------------|----------|-----------------|-------------|----------|----------|----------|-----------------------|
| `cb_shard_lane0` | `shard_rows_tiles * shard_cols_tiles` (`S_pages`), globally allocated at the input shard's address — **0 new bytes** | `S_pages`: the whole shard is resident from program start (pre-placed input); lane 0 reads `ceil(n_k / num_lanes)` of those slots | `{tile_rows: spans → block_extent_rows ≤ shard_rows_tiles, tile_cols: spans → block_extent_cols ≤ shard_cols_tiles}` | input dtype: `Float16_b` (bfloat16) or `Bfp8_b` (bfloat8_b); dataflow-only payload, never through DEST, so the DEST-width audit does not apply | none — pre-placed by the producer op; no kernel pushes | writer lane 0 (BRISC) | whole program | shares the input's sharded buffer with `cb_shard_lane1` (same bytes); a second index exists only so each CB has one consumer; capacity = live set (the shard), pinned by the input |
| `cb_shard_lane1` | `S_pages`, same buffer — **0 new bytes** | `S_pages` resident; lane 1 reads `n_k − ceil(n_k / num_lanes)` slots | `{tile_rows: spans → block_extent_rows, tile_cols: spans → block_extent_cols}` | same as lane 0 | none — pre-placed | writer lane 1 (NCRISC) | whole program | shares the input's sharded buffer with `cb_shard_lane0`; separate index for single-consumer ownership |

Sync: push count = wait count = 0 on both (address binding via `get_read_ptr`; no FIFO pointer moves).

## Non-CB L1 the call adds

| Item | Bytes per core (per L1 bank) | Live | Why it exists |
|------|------------------------------|------|---------------|
| output pages, when `memory_config` is L1 interleaved | `[buffer == l1] * ceil(P / N_l1_banks) * page_bytes` | from allocation until the caller frees the output | the op's result; counts against the 1347776 B the model leaves free (spec, Device section) |

## Symbol table

| Symbol | Meaning | Bound | Predicate establishing it |
|--------|---------|-------|---------------------------|
| `S_pages` | pages per shard = `(s_h / 32) * (s_w / 32)` | fixed by the input's allocation, which lies outside the 1347776 B budget; costs the op 0 B | `validate()`: whole-tile shard shape; TT-Metal shard validation (`tensor_spec.cpp:50-72`). Golden: 5 (p2), 10 (p1), 3 and 11 (other specs) |
| `n_k` | tiles of core k's shard = `block_extent_rows * block_extent_cols` | `≤ S_pages` | per-core rectangle formula (`op_design.md`, Work Distribution) |
| `num_lanes` | issue lanes per core | ∈ {1, 2}; Phase 0 = 2 | host constant |
| `P` | output pages = `R * C` | golden max 1024 | golden `INPUTS`; for an L1 output the host refuses `ceil(P / N_l1_banks) * page_bytes > 1347776` with `RuntimeError` (BB-05) |
| `N_l1_banks` | L1 banks of the device's interleaved allocator | read from the device (`ttnn.TensorSpec.compute_consumed_memory_bytes_per_bank(page_alignment, num_banks)`, `ttnn/cpp/ttnn-nanobind/tensor.cpp:494-498`, gives the per-bank bytes directly); INFERENCE: one per worker core, ≥ 103 on the production grid | device property |
| `page_bytes` | tile bytes | ∈ {2048 (bfloat16), 1088 (bfloat8_b)} | `TARGET` dtypes; `tt_backend_api_types.hpp:129`, `:139` |

## Total per-core footprint

    L1_op(core) = 0                                              # both CBs alias the input shard
                + [buffer == l1] * ceil(P / N_l1_banks) * page_bytes   # output pages in this core's bank

No term scales with a block extent (the block is the resident input, already paid for), and none scales with `num_lanes`. p2 (DRAM output): 0 B. p1 (L1 output): `ceil(1024 / N_l1_banks) * 1088` = 10880 B at `N_l1_banks ≥ 103`, about 0.8 % of 1347776 B.

## Data-movement budget

Chosen split: the input's own shard split (owner-writes).

| Tensor | DRAM crossings | Why that many | Cross-core traffic added |
|--------|----------------|---------------|--------------------------|
| `input_tensor` (x) | 0 | resident in L1 shards; each tile is the local source of its own output write, never fetched | none (no fetch over the NoC) |
| output (p2, DRAM interleaved) | 1 (327680 B written) | each page written once, from the core that holds its tile | 160 unicast page writes (327680 B) from 32 cores into 8 DRAM banks |
| output (p1, L1 interleaved) | 0 | L1 destination; each page written once | 1024 unicast page writes (1114112 B) from 103 cores into the L1 banks |

Totals per tier — p2: DRAM 327680 B (write), NoC 327680 B, core-local L1 reads 327680 B (the write engine sourcing the shard). p1: DRAM 0 B, NoC 1114112 B, core-local L1 reads 1114112 B, remote L1 writes 1114112 B.

> Cheapest-traffic split considered: the input's own shard split (owner-writes) — every tile crosses the NoC once and the input crosses no tier, which is the named-boundary minimum. Implemented: owner-writes. The implemented split is the cheapest; every alternative (output-page split, output-bank split, gathering cores) adds one NoC fetch per tile (2P transfers) and is `rejected` in the regime table, not deferred.
