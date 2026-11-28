# L1 Ledger: g6_sharded_to_interleaved

Block axes (from `op_design.md` Blocking Model): `tile_row` (`block_rows`), `tile_col` (`block_cols`), shard/core axis (`num_blocks_this_core` = 1).

| CB | Capacity (pages) | Live set | Axis accounting | Page format | Producer | Consumer | Lifetime | Shares with / why not |
|----|------------------|----------|-----------------|-------------|----------|----------|----------|-----------------------|
| `cb_input_shard` | `shard_row_tiles * shard_col_tiles` — globally allocated ON the input's shard buffer (`cb_descriptor_from_sharded_tensor`); **0 bytes of new L1** | the whole shard (every slot is resident before launch and read in place) | `{tile_row: spans → block_rows ≤ shard_row_tiles, tile_col: spans → block_cols ≤ shard_col_tiles, shard axis: spans → 1 block}` | input dtype (Float16_b / Bfp8_b); dataflow-only payload, never passes DEST | host placement (pre-resident shard) | writer | whole program | Shares the input tensor's own allocation (zero-copy alias); no other CB exists to share with. Capacity = live set (no gap). |

No other CB. No compute kernel; no DEST use (`fp32_dest_acc_en` not applicable).

## Symbol table

| Symbol | Bound | Predicate establishing it |
|--------|-------|---------------------------|
| `shard_row_tiles`, `shard_col_tiles` | the input shard, already resident in L1 outside the op's budget | input is a valid L1-sharded tensor (validate) |
| `num_l1_banks` | device L1 interleaved bank count | device property |
| `output_l1_pages_per_bank` | `ceil(tensor_row_tiles * tensor_col_tiles / num_l1_banks)` | only when `memory_config.buffer_type == L1` |

## Per-core footprint

`op_L1(core) = 0` (CBs) + `[buffer == l1] * output_l1_pages_per_bank * out_aligned_page_size` (output allocation, counted against the 1347776 B the model leaves).

- p1 (1024 tiles × 1088 B → L1, ~100+ banks): ≈ `ceil(1024/num_l1_banks) * 1088` ≈ 10–11 KB per bank ≪ 1347776 B.
- p2 (160 tiles × 2048 B → DRAM): 0 B L1.

No term scales with a block knob; the only scaling term is the output itself.

## Data-movement budget

| Tensor | DRAM crossings | Why that many | Cross-core traffic added |
|--------|----------------|---------------|--------------------------|
| `input_tensor` (sharded L1) | 0 | read in place from the owning core's L1 (zero-copy CB) | 0 |
| `output` (DRAM dest) | 1 (each page written once) | owner-writes: the write is the only crossing | 0 beyond the write itself |
| `output` (L1 dest) | 0 | L1 banks; each page crosses the NoC once | 0 beyond the write itself |

Totals: p2 → DRAM: 160 × 2048 B = 327 680 B over NoC→DRAM, 0 B DRAM reads. p1 → L1: 1024 × 1088 B = 1 114 112 B over NoC→L1, 0 B DRAM.

> Cheapest-traffic split considered: owner-writes (each shard core writes its resident tiles) — delta 0 (it is the minimum: one NoC crossing per tile). Implemented: owner-writes. The implemented split is also the cheapest.
