# L1 Ledger: layer_norm_tile

Schema and audits: `.claude/references/l1-footprint-discipline.md`. Block axes from `op_design.md`:
`tile_rows` (extent `block_height`, runtime extent `block_rows <= block_height`) and `W` (extent
`Wt = tensor_w_tiles`, whole, never split). `intra_tile_rows` is fixed at the tile height (32) and is
spanned by every tile-shaped page; it is not repeated per row below.

`T_in` = bf16 tile = 2048 B. `T_acc` = 4096 B (Float32) when `fp32_dest_acc_en`, else 2048 B (Float16_b).

| CB | Capacity (pages) | Live set | Axis accounting | Page format | Producer | Consumer | Lifetime | Shares with / why not |
|----|------------------|----------|-----------------|-------------|----------|----------|----------|-----------------------|
| `cb_input_tiles` | `input_depth * block_height * Wt` | no residual: `block_rows * Wt` (whole block resident through `mean_block`); residual: 1 tile on the consumer side | `{tile_rows: spans -> block_height, W: spans -> Wt}` | bf16 (tensor dtype) | reader | compute | `load_block` .. `center_block` (no residual) / `add_residual_block` (residual) | Not shared: it is reader-produced, and every candidate partner (`work`, `stats`) is compute-produced, so aliasing would give the bytes two producers. Capacity above the residual-path live set is the reader prefetch of a whole block in one pass; `input_depth = 2` only when a next block exists (double buffering). |
| `cb_residual_tiles` | `HAS_RESIDUAL * input_depth * block_height * Wt` | 1 tile consumer-side (streamed by `add_residual_block`) | `{tile_rows: spans -> block_height, W: spans -> Wt}` | bf16 | reader | compute | `load_block` .. `add_residual_block` | Not shared: concurrent with `cb_input_tiles` (both streamed by the same phase) and reader-produced. Capacity above live set = whole-block reader prefetch so the reader fills input and residual tile-rows in one barrier; a `Wt`-tile window is the nearby alternative (overlap lamp scope). |
| `cb_gamma_rows` | `Wt` | `Wt` (all tile-columns re-read by every `scale_block`) | `{tile_rows: streams -> 0 (constant over rows), W: spans -> Wt}` | bf16 | writer | compute | kernel | Not shared: live for the whole kernel, overlapping every other CB. Also serves as the never-read scaler id of `mean_block`/`rstd_block` (no allocation). |
| `cb_beta_rows` | `Wt` | `Wt` | `{tile_rows: streams -> 0, W: spans -> Wt}` | bf16 | writer | compute | kernel | Not shared: kernel lifetime, concurrent with `cb_gamma_rows` (both resident). |
| `cb_xsum` (alias group `work`) | `block_height * Wt` (group total) | `block_rows * Wt` | `{tile_rows: spans -> block_height, W: spans -> Wt}` | `T_acc` | compute | compute | `add_residual_block` .. `center_block` | **Shares** the `work` allocation with `cb_centered`, `cb_normed`, `cb_scaled` (lockstep alias, identical page size and quanta). No traffic without residual. |
| `cb_centered` (alias group `work`) | same allocation | `block_rows * Wt` (resident through `sumsq_block`) | `{tile_rows: spans -> block_height, W: spans -> Wt}` | `T_acc` | compute | compute | `center_block` .. `normalize_block` | **Shares** `work`. Not aliased with `cb_input_tiles` (reader-produced, see above). |
| `cb_normed` (alias group `work`) | same allocation | `block_rows * Wt` at most (streamed into `scale_block`) | `{tile_rows: spans -> block_height, W: spans -> Wt}` | `T_acc` | compute | compute | `normalize_block` .. `scale_block` | **Shares** `work`. |
| `cb_scaled` (alias group `work`) | same allocation | `block_rows * Wt` at most | `{tile_rows: spans -> block_height, W: spans -> Wt}` | `T_acc` | compute | compute | `scale_block` .. `shift_block` | **Shares** `work`. Not packed straight into `cb_output_tiles`: `shift_block` is a second FPU binary that needs its operand in a CB (dest-reuse fusion is the Phase E perf lamp). |
| `cb_mean` (alias group `stats`) | `block_height` (group total) | `block_rows` (helper bulk-reserves all outputs) | `{tile_rows: spans -> block_height, W: streams -> 1 (collapsed)}` | `T_acc` | compute | compute | `mean_block` .. `center_block` | **Shares** the `stats` allocation with `cb_sumsq`, `cb_rstd`. Not folded into `work`: different page count and it is live while `work` is full. |
| `cb_sumsq` (alias group `stats`) | same allocation | `block_rows` | `{tile_rows: spans -> block_height, W: streams -> 1}` | `T_acc` | compute | compute | `sumsq_block` .. `rstd_block` | **Shares** `stats`. Not folded into DEST across both phases: the within-tile collapse is a separate helper call (no SFPU reduce chain element exists). |
| `cb_rstd` (alias group `stats`) | same allocation | `block_rows` | `{tile_rows: spans -> block_height, W: streams -> 1}` | `T_acc` | compute | compute | `rstd_block` .. `normalize_block` | **Shares** `stats`; entry `o` is packed only after entry `o` of `cb_sumsq` was unpacked (same output). |
| `cb_output_tiles` | `OUTPUT_DEPTH * Wt` | `Wt` per side (compute fills tile-row `r+1` while the writer drains `r`) | `{tile_rows: streams -> OUTPUT_DEPTH, W: spans -> Wt}` | bf16 | compute | writer | per tile-row | Not shared: writer-consumed and concurrent with every compute phase of the next tile-row; capacity above `Wt` is double buffering. |

Allocations: 7 with residual (`cb_input_tiles`, `cb_residual_tiles`, `cb_gamma_rows`, `cb_beta_rows`, `work`, `stats`, `cb_output_tiles`), 6 without. No scaler / mask CB exists (aligned `AccumulateViaAdd` never waits one).

## Symbol table

| Symbol | Bound | Predicate establishing it |
|--------|-------|---------------------------|
| `Wt = tensor_w_tiles` | 24 | validate: `W == 768` domain, `W % 32 == 0` |
| `block_height` | `1 <= block_height <= min(max_core_rows_tiles, block_height_cap(input_depth))` | `derive_blocking()` in `op_design.md`; validate requires `block_height_cap(INPUT_DEPTH_MAX) >= 1` |
| `block_rows` | `1 <= block_rows <= block_height` | per-block runtime extent (`last_block_extent` on the final block) |
| `input_depth` | `{1, 2}` | 1 iff `max_core_rows_tiles <= block_height_cap(1)` |
| `HAS_RESIDUAL` | `{0, 1}` | `residual_input_tensor is not None` |
| `OUTPUT_DEPTH` | 2 | host constant |
| `T_acc` | `{2048, 4096}` | `fp32_dest_acc_en` |
| `CB_L1_BUDGET_BYTES` | 786,432 | host constant; leaves >= 600 KiB of the ~1.4 MiB usable BH L1 for L1-interleaved tensor banks (~56 KiB per tensor per core at `B*S = 4096`) |

## Footprint

```
total = block_height * ( Wt * (input_depth * T_in * (1 + HAS_RESIDUAL) + T_acc) + T_acc )   # scales with block_height, input_depth, HAS_RESIDUAL, T_acc
      + Wt * T_in * (2 + OUTPUT_DEPTH)                                                       # fixed: gamma, beta, output window
```

| Configuration | `block_height` | `input_depth` | Bytes |
|---------------|----------------|---------------|-------|
| fp32, residual (worst Phase 0) | 2 | 1 | 598,016 |
| fp32, no residual | 2 | 1 | 499,712 |
| bf16 DEST, residual | 2 | 1 | 495,616 |
| fp32, residual, multi-block (`(256,1,16,768)` on 110 cores) | 1 | 2 | 495,616 |

## Data-movement budget

Tier: input / residual / output pages live in remote L1 banks, so the expensive crossing is the NoC hop
to the computing core; TILE gamma/beta are in DRAM. Example shape `(8, 1, 512, 768)` with residual:
`tensor_bytes = 128 tile-rows * 24 * 2048 = 6,291,456 B`; 110-core grid, all 110 cores active (128 tile-rows).

| Tensor | DRAM crossings | Why that many | Cross-core traffic added |
|--------|----------------|---------------|--------------------------|
| input | 0 DRAM; 1 NoC crossing per tile (6 MiB) | whole `W` resident per block, so mean, variance and normalize all read the one resident copy | none |
| residual | 0 DRAM; 1 NoC crossing per tile (6 MiB) | folded into `cb_xsum` once, never re-read | none |
| gamma | TILE/DRAM: 1 crossing per active core of 24 x 128 B = 3 KiB (330 KiB total); RM/L1: 24 x 64 B = 1.5 KiB per core (165 KiB total) | reuse-shared operand fetched per core (row 0 only, not whole 2 KiB tiles: whole tiles would be 48 KiB per core) | none |
| beta | same as gamma | same | none |
| output | 0 DRAM; 1 NoC crossing per tile (6 MiB) | written once from the resident block | none |

Totals for the example: NoC 18 MiB (in + res + out) + 0.33-0.66 MiB affine rows; DRAM 0 (RM gamma) or 0.66 MiB (TILE gamma).

> Cheapest-traffic split considered: `tile_rows` across cores (`row_resident`) - delta 0 bytes per tier.
> Implemented: `tile_rows` across cores. It is also the cheapest-traffic split. The only above-minimum
> term is the per-core affine-row fetch (`affine_mcast`, deferred: the fetch runs on the writer during
> the reader's block-0 fetch and compute's pre-affine phases, so it is off the critical path). The
> `w_split_combine` regime adds traffic (+2 (k-1) T_acc per core per tile-row) and is deferred for the
> occupancy reason stated in the regime table, not for traffic.
