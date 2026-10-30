# L1 Ledger: layer_norm (layer_norm_tile)

Schema and audits: `.claude/references/l1-footprint-discipline.md`. Block axes: `row` (extent `block_rows`) and `w` (extent `block_w_tiles`), from `op_design.md` Blocking Model.

Page sizes: `T_in = T_aff = T_out = 2048` (bf16 tile). `T_int` = 4096 when `fp32_dest_acc_en` (Float32), else 2048 (Float16_b).

| CB | Capacity (pages) | Live set | Axis accounting | Page format | Producer | Consumer | Lifetime | Shares with / why not |
|----|------------------|----------|-----------------|-------------|----------|----------|----------|-----------------------|
| `cb_input_tiles` | `input_depth_blocks * block_rows * block_w_tiles` | `block_rows * block_w_tiles` (P1 waits the whole block), plus 1 prefetched block when `input_depth_blocks = 2` | `{row: spans -> block_rows, w: spans -> block_w_tiles}` | bf16 (input tensor data; carries no accumulation, so it is not subject to the DEST-width rule) | reader | compute | load_block to P2 | Not with `cb_centered`: under fp32 DEST the formats differ (bf16 vs Float32). Under bf16 DEST an in-place P2 would make compute a second producer of a CB the reader is refilling (ownership violation). Capacity above the live set is double buffering for block prefetch (`input_depth_blocks`), and is set to 1 when the core has a single block. |
| `cb_gamma_rows` | `block_w_tiles` | `block_w_tiles` (row 0 of each tile) | `{row: streams (constant across all blocks), w: spans -> block_w_tiles}` | gamma dtype (bf16; tensor data) | reader | compute | kernel | Could merge with `cb_beta_rows` into one `2*block_w_tiles` CB via `TileAddressing::Offset`. That saves 0 bytes (same capacity sum), so they stay separate (one tensor per CB). Not aliasable with anything: live for the whole kernel. |
| `cb_beta_rows` | `block_w_tiles` | `block_w_tiles` | `{row: streams (constant), w: spans -> block_w_tiles}` | beta dtype (bf16; tensor data) | reader | compute | kernel | see `cb_gamma_rows` |
| `cb_row_stat` | `block_rows` | `block_rows` | `{row: spans -> block_rows, w: streams (collapsed to 1 tile per row)}` | `intermediate_format` (Float32 if fp32 DEST, else Float16_b) | compute | compute | P1 to P2 (mean), then P3 to P4 (sumsq partial) | **Shares with itself across two disjoint uses** (mean, then sumsq): same producer/consumer, same shape and format, sequential lifetimes with balanced push/pop. Not with `cb_rstd`: P4 reads `cb_row_stat` while writing `cb_rstd` (concurrent; the reduce asserts input != output). |
| `cb_rstd` | `block_rows` | `block_rows` | `{row: spans -> block_rows, w: streams (collapsed)}` | `intermediate_format` | compute | compute | P4 to P5 | Concurrent with `cb_row_stat` during P4 (see above). Disjoint from P1's mean in time, but that use is already folded into `cb_row_stat`. |
| `cb_centered` | `block_rows * block_w_tiles` | `block_rows * block_w_tiles` (P3 reads the whole block resident; P5/P6 rewrite it in place) | `{row: spans -> block_rows, w: spans -> block_w_tiles}` | `intermediate_format` | compute | compute | P2 to P7 | Already absorbs three phase boundaries in place (P5 scale, P6 gamma; P7 packs straight to the destination). Not with `cb_input_tiles` (format / ownership, above). Not with `cb_output_tiles`: the writer is its consumer, so an in-place final pass would give it two consumers. Single-buffered: one compute thread cannot pipeline blocks. |
| `cb_output_tiles` | `output_depth_rows * block_w_tiles` | `block_w_tiles` (one writer batch) + one tile-row being packed | `{row: streams (window of output_depth_rows tile-rows), w: spans -> block_w_tiles}` | bf16 (op output dtype) | compute | writer | P7 to store_block | Capacity above the live set is double buffering (`output_depth_rows = 2`), so P7 packs row r+1 while the writer drains row r. Packed per tile, drained per tile-row, so it never needs to span `row`. No aliasing: live for every block. |

No reduce-scaler CB: AccumulateViaAdd on the aligned W never reads the scaler (`reduce_helpers_compute.inl:320-337`), and 1/W is an SFPU scalar multiply.

## Symbol table

| Symbol | Bound | Predicate establishing it |
|--------|-------|---------------------------|
| `block_w_tiles` = `tensor_w_tiles` | 24 under contract; `<= W_TILES_RESIDENT_MAX = 32` | `validate()`: `W % 32 == 0 and ceil(W/32) <= 32` (regime `row-split-resident-w`) |
| `block_rows` | `<= BLOCK_ROWS_MAX = 2` | host: `block_rows = min(core_rows_g1, BLOCK_ROWS_MAX)` |
| `input_depth_blocks` | in {1, 2} | host: `min(2, max_num_blocks)`, with `max_num_blocks = ceil(core_rows_g1 / block_rows)` |
| `output_depth_rows` | 2 | host constant |
| `T_int` | in {2048, 4096} | `fp32_dest_acc_en` |

## Total per-core footprint

```
F = input_depth_blocks * block_rows * block_w_tiles * T_in        # scales with block_rows, block_w_tiles, input depth
  + 2 * block_w_tiles * T_aff                                     # scales with block_w_tiles only
  + 2 * block_rows * T_int                                        # scales with block_rows
  + block_rows * block_w_tiles * T_int                            # scales with block_rows, block_w_tiles
  + output_depth_rows * block_w_tiles * T_out                     # scales with block_w_tiles, output depth
```

| Case (fp32 DEST, `block_w_tiles = 24`) | F |
|----------------------------------------|---|
| `block_rows = 1`, depth 1 (R <= num_cores) | 48 + 96 + 8 + 96 + 96 = **344 KB** |
| `block_rows = 2`, depth 1 (R in (64, 128], single block) | 96 + 96 + 16 + 192 + 96 = **496 KB** |
| `block_rows = 2`, depth 2 (R > 2 * num_cores, e.g. 248) | 192 + 96 + 16 + 192 + 96 = **592 KB** |
| bound (`block_w_tiles = 32`, `block_rows = 2`, depth 2) | 256 + 128 + 16 + 256 + 128 = **784 KB** |

Under bf16 DEST, the `T_int` terms halve.

## Data-movement budget

`R = tensor_row_tiles`, `P` = participating cores (`min(R, grid cores)`), tile = 2 KB.

| Tensor | DRAM crossings | Why that many | Cross-core traffic added |
|--------|----------------|---------------|--------------------------|
| input | 0 (L1-interleaved). Each tile crosses the NoC from its L1 bank **once**: `R * 48 KB`. | whole `w` resident per block, so the mean, center, variance and normalize passes all read core-local L1 (no re-read) | 0 |
| output | 0 (L1-interleaved). Each tile is written over the NoC once: `R * 48 KB`. | packed straight to `cb_output_tiles` from the last phase | 0 |
| gamma | `affine-tile`: `P * 24 * affine_read_bytes * 2` from DRAM (`P * 3 KB` on BH, 64 B granule). `affine-rm`: 0 DRAM, `P * 1.5 KB` over the NoC from L1. | row-0-only read: 64 valid B per tile, read once per core and resident for the kernel | 0 |
| beta | same as gamma | same | 0 |
| mean / sumsq / rstd / xc | 0 | core-local intermediates | 0 |

Totals on the 110-core Blackhole grid at R = 128, P = 110: NoC-from-L1 input 6 MB, output 6 MB; affine 330 KB from L1 (RM form), or 660 KB from DRAM (TILE form, 64 B read granule = BH DRAM alignment). At R = 32, P = 32: 1.5 MB, 1.5 MB, 96 / 192 KB.

> Cheapest-traffic split considered: `row` (tile-rows across cores, whole W resident). Implemented: `row`. It is the cheapest: the only split with zero cross-core traffic and single-crossing input/output. The `w-split-combine` alternative adds about `R * k * 4 KB` of partial-tile gather/mcast plus 2 semaphore rounds per block and saves only `P * 3 KB * (1 - 1/k)` of affine reads. It is deferred because its only gain is occupancy in the `R < num_cores` class (positive reason in the regime table). The affine replication (`P * 3 KB` vs. 3 KB minimum) is the one above-minimum term. Mcast was rejected because those reads are latency-bound and already hidden behind block-0 compute.
