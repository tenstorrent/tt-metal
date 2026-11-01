# L1 Ledger: layer_norm_tile (regime R1 `row_resident`)

Schema and audits: `.claude/references/l1-footprint-discipline.md`. Block axes from `op_design.md`
Blocking Model: `row` (extent `R` = `block_row_tiles`), `w` (extent `core_w_tiles` = `Wt` in R1),
`w`-transfer quantum (`C` = `io_chunk_tiles`), `band` (`band_rows` = 32, unused by R1 buffers).

`tb` = 2048 B (bf16 tile). `wb` = 4096 B when `fp32_dest_acc_en` else 2048 B. `wfmt` = Float32 when
`fp32_dest_acc_en` else Float16_b. `n = R_b * Wt` (tiles of one block).

| CB | Capacity (pages) | Live set | Axis accounting | Page format | Producer | Consumer | Lifetime | Shares with / why not |
|----|------------------|----------|-----------------|-------------|----------|----------|----------|-----------------------|
| `cb_input_tiles` | `in_depth * R * Wt` | no residual: `R * Wt` resident (mean + center) plus up to `R * Wt` of the next block in flight; residual: streamed, peak = whatever the reader has prefetched (up to `in_depth * R * Wt`) | `{row: spans -> R, w: spans -> Wt, w-chunk: streams -> C, band: streams -> 1}` | Float16_b (input dtype, reader-filled; not a packed intermediate) | reader | compute | per block | none: concurrent with `cb_residual_tiles` (both filled by the same chunk loop) and with the work pair (prefetch of block b+1 overlaps compute of b). Capacity above the residual-path live set is the `in_depth` double buffer, a pipelining decision |
| `cb_residual_tiles` (only `HAS_RESIDUAL`) | `in_depth * R * Wt` | streamed; peak = reader prefetch, up to `in_depth * R * Wt` | `{row: spans -> R, w: spans -> Wt, w-chunk: streams -> C, band: streams -> 1}` | Float16_b | reader | compute | per block | none: concurrent with `cb_input_tiles` (consumed tile-for-tile together by `residual_add_block`). Block-sized so the reader runs a whole block ahead (pipelining decision; `split_reader` lamp may move its producer to the writer RISC) |
| `cb_gamma_tiles` | `Wt` | `Wt` (row 0 of each tile) for the whole kernel | `{row: streams -> 1 (shared by every row), w: spans -> Wt, w-chunk: streams -> 1, band: streams -> 1}` | Float16_b (affine dtype) | reader | compute | whole kernel | none: live for the whole kernel, concurrent with everything. Also bound as the (never-read) reduce scaler id; no separate scaler CB exists |
| `cb_beta_tiles` | `Wt` | `Wt` for the whole kernel | `{row: streams -> 1, w: spans -> Wt, w-chunk: streams -> 1, band: streams -> 1}` | Float16_b | reader | compute | whole kernel | none: concurrent with `cb_gamma_tiles` (both used in back-to-back phases of every block) |
| `cb_row_work_a` | `R * Wt` | `n`: `z` (or normalized block) resident across mean / center, or across normalize -> affine_scale | `{row: spans -> R, w: spans -> Wt, w-chunk: streams -> 1, band: streams -> 1}` | wfmt | compute | compute | per block (two disjoint occupancies) | SHARES the allocation with `cb_row_work_b` (alias group, one `CBDescriptor`, `total_size = R * Wt * wb`): each phase moves tile i from one id to page i of the other, so the two ids never need more than one block of bytes |
| `cb_row_work_b` | `R * Wt` (same bytes) | `n`: `c` resident across sumsq, or the scaled block | `{row: spans -> R, w: spans -> Wt, w-chunk: streams -> 1, band: streams -> 1}` | wfmt | compute | compute | per block | aliased with `cb_row_work_a` (above). Not merged into `cb_input_tiles`: that would give the input CB a second producer (compute), and it has a different page format under fp32 DEST |
| `cb_row_mean` | `R` | `R_b` (one col-0 tile per tile-row) | `{row: spans -> R, w: streams -> 1 (collapsed), w-chunk: streams -> 1, band: streams -> 1}` | wfmt | compute | compute | `row_mean_block` -> end of `center_block` | none with `cb_row_sumsq` although lifetimes are disjoint: one shared FIFO would place the second indexed window at `[R_b, 2*R_b)` and wrap on a ragged final block (Mechanism caps); two `R`-tile CBs cost the same bytes as one `2R` CB. Not shared with `cb_inv_std`: different phase pair, same argument |
| `cb_row_sumsq` | `R` | `R_b` raw DEST-accumulated tiles | `{row: spans -> R, w: streams -> 1 (accumulated in DEST), w-chunk: streams -> 1, band: streams -> 1}` | wfmt | compute | compute | `row_sumsq_block` -> `row_inv_std_block` | none with `cb_inv_std`: concurrent (the reduce reads it while writing `cb_inv_std`; reduce asserts input != output). With `cb_row_mean`: see above |
| `cb_inv_std` | `R` | `R_b` col-0 tiles | `{row: spans -> R, w: streams -> 1, w-chunk: streams -> 1, band: streams -> 1}` | wfmt | compute | compute | `row_inv_std_block` -> end of `normalize_block` | none: concurrent with `cb_row_sumsq` (above); disjoint from `cb_row_mean` but sharing would put its indexed window behind the mean's (wrap on ragged block) |
| `cb_output_tiles` | `out_depth * R * Wt`, `out_depth = 1` | up to `n` while the writer drains block b and compute runs block b+1's first phases | `{row: spans -> R, w: spans -> Wt, w-chunk: streams -> C (writer pops C), band: streams -> 1}` | Float16_b (output dtype) | compute | writer | per block | none: concurrent with the work pair of the next block (the writer drains while compute proceeds; pipelining decision). Not packed straight into a work CB: the writer would become a second consumer of the alias pair |

No scaler, mask, or zero CB exists: both reduces run `ReduceAlgorithm::AccumulateViaAdd` on tile-aligned W
(`reduce_helpers_compute.inl:320-337` never waits the scaler on that path).

## Symbol table

| Symbol | Meaning | Bound | Predicate establishing it |
|--------|---------|-------|---------------------------|
| `Wt` | `tensor_w_tiles = W / 32` | 24 in TARGET (W = 768); any value with `F(1) <= budget` | validate: `W % 32 == 0`; R1 predicate `F(1) <= budget` |
| `R` | `block_row_tiles` | `1 <= R <= min(ceil(core_row_tiles_max / MIN_BLOCKS_PER_CORE), R_fit)` | `select_blocking()` |
| `R_b` | runtime rows of a block | `1 <= R_b <= R` | `last_block_extent` derivation |
| `in_depth` | input prefetch depth (blocks) | 1 or 2 | `2 if ceil(core_row_tiles_max / R) >= 2 else 1` |
| `out_depth` | output depth (blocks) | 1 | host constant `OUT_DEPTH_BLOCKS` |
| `C` | `io_chunk_tiles` | divides `Wt`, `<= 8` | host: largest divisor of `Wt` not above `IO_CHUNK_TARGET_TILES` |
| `has_res` | residual present | 0 or 1 | call signature |
| `tb`, `wb` | tile bytes | 2048; 2048 or 4096 | dtype bf16; `fp32_dest_acc_en` |
| `budget` | CB bytes available per core | `largest_contiguous_bytes_free_per_bank(L1) - 64 KiB` | queried on host after output allocation |

## Footprint (per core, bytes)

```
F(R) = R * Wt * ( in_depth * tb * (1 + has_res)     # input, residual
                + wb                                 # work alias pair (one allocation)
                + out_depth * tb )                   # output
     + 3 * R * wb                                    # mean, sumsq, inv_std
     + 2 * Wt * tb                                   # gamma, beta
```

Scaling: every term but the last is linear in `R`; the first term scales with `in_depth` and doubles
with the residual; `wb` doubles the work and stat terms under fp32 DEST; the affine term is constant
in every block knob. Production (Wt = 24, residual, fp32 DEST, in_depth = 2): `F(1)` = 96 + 96 + 96 +
48 + 12 + 96 = 444 KiB; `F(2)` = 792 KiB. `C` does not appear: chunking changes transfer granularity,
not capacity.

## Data-movement budget (chosen split R1, headline 8x512x768 bf16)

Per-tensor size: `X = Rs = O = 8 * 512 * 768 * 2 B` = 6.29 MB. `A` = 110 active cores.

| Tensor | DRAM crossings | Why that many | Cross-core traffic added |
|--------|----------------|---------------|--------------------------|
| `input` | 1 (from DRAM, or 1 NoC read from L1 banks when the input is in L1) | the whole row block stays resident in `cb_input_tiles` / the work pair for every pass | 0 |
| `residual` | 1 | consumed once by `residual_add_block`; `z` stays resident | 0 |
| `output` | 1 write | packed once by `affine_shift_block` | 0 |
| `gamma` | `A` sparse re-reads of row 0: 24 x 64 B (RM) or 24 x 128 B (TILE) per core = 1.5 / 3 KiB | reuse-shared operand read per core (R4 deferred) | 0 |
| `beta` | same as gamma | same | 0 |

Totals per tier, headline:

| Row | DRAM bytes | L1-bank NoC bytes |
|-----|------------|-------------------|
| input L1, residual DRAM, out DRAM, affine RM L1 | 12.58 MB | 6.29 MB + 0.34 MB affine |
| all DRAM with residual, affine RM L1 | 18.87 MB | 0.34 MB affine |
| all DRAM, no residual | 12.58 MB | 0.34 MB |
| all DRAM with residual, affine TILE DRAM | 18.87 MB + 0.68 MB | 0 |

> Cheapest-traffic split considered: R1 `row_resident` (tile-rows across cores, whole row resident), tied
> on bytes with R3 `half_band_split` (same bytes, 2x NoC transactions, less when all-pad bands are
> skipped). Implemented: R1. It is the cheapest-traffic split. R3 is deferred because its gain is load
> balance and pad skipping, not bytes, and it doubles the transaction count; R2 is deferred because it
> adds `2 * R * (k-1) * tile` NoC bytes per core per block on top of the same DRAM bytes. Both stay
> reachable (op_design.md Regimes).
