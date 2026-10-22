# L1 Ledger: rms_norm

Schema and audits: `.claude/references/l1-footprint-discipline.md`. Blocking axes (from `op_design.md` → Blocking Model): `tile_rows` (extent `R` = `block_rows`), `width_reduce` (extent `Wp` = `width_padded_tiles`, the whole padded row, reduction stage), `width_stream` (extent `C` = `w_chunk_tiles`, streaming stages). Regimes: R1 (row-parallel, row-resident) and, since Refinement 4, R3 (W-split + cross-core Σx² combine; see the Refinement 4 section — every row below applies to R3 with `Wp` / `Wt` read as the core's W-slice). `fp32_dest_acc_en = True` → every compute-produced intermediate that carries an un-finalized value is `Float32`; `fp32_dest_acc_en = False` (Refinement 2) → `Float16_b` (DEST is 16-bit, so a `Float32` page would store no extra bits), for every input dtype including `bfloat8_b` (a `Bfp8_b` intermediate would also share one exponent across 16 raw Σx² lanes / x·γ products).

Per-CB presence: `is_rm` = input `Layout::ROW_MAJOR`; `has_gamma` = γ supplied. Capacities are in pages; page bytes are in the `Page format` column's tile size unless stated.

| CB | Capacity (pages) | Live set | Axis accounting | Page format | Producer | Consumer | Lifetime | Shares with / why not |
|----|------------------|----------|-----------------|-------------|----------|----------|----------|-----------------------|
| `cb_x_rm_sticks` (is_rm) | `stream_depth · C` (page = `T_in`) | `C` pages (32 stick segments of one chunk) while compute tilizes; capacity − live = one chunk in flight (double buffering) | `{tile_rows: streams → 1 tile-row, width_reduce: streams, width_stream: spans → C}` | input dtype (RM bytes, tile-sized pages) | reader | compute (`tilize`) | load/tilize, per (tile-row, chunk) | No share: concurrent with `cb_x_tiles` (tilize reads one, writes the other) and with `cb_output` (reader runs ahead while the writer drains); `alias_with` would require lockstep with a buffer that is live in the same instant. |
| `cb_x_tiles` | `x_depth · R · Wp` (page = `T_in`) | `R_b · Wp` (whole block: read once, used by `square_accumulate` and every `scale_*` chunk); capacity − live = the next block (`x_depth = 2`, prefetch) or 0 (`x_depth = 1`) | `{tile_rows: spans → R, width_reduce: spans → Wp, width_stream: streams (indexed by chunk offset, not sized by C)}` | input dtype — tilize (RM) / DRAM read (TILE) of input-dtype values is lossless, so no wider page is needed (audit 2: this buffer is not an accumulation boundary) | reader (TILE) / compute `tilize` (RM) | compute | `load_block` → `release_block`, per block | No share: it is the only Wt-proportional buffer and is live across the entire block; every other CB is live inside its lifetime. Pad tiles (`Wp − Wt` < C per row) are zeroed once and are what lets every streaming CB use nominal C quanta. |
| `cb_reduce_scaler` | `1` (page = 2048 B) — constant live set | 1 | `{tile_rows: streams (not scaled), width_reduce: streams, width_stream: streams}` | bfloat16 (packed scaler, `calculate_and_prepare_reduce_scaler`) — value 1.0 is exact | reader (once) | compute (`reduce`, never popped until kernel end) | kernel | No share: lives for the whole kernel. |
| `cb_sumsq_partial` | `R` (page = `T_interm`) | `R_b` (one raw Σx² tile per tile-row, all produced before `rms_finalize` runs) | `{tile_rows: spans → R, width_reduce: streams (collapsed into 1 tile), width_stream: streams}` | `T_interm` format: Float32 under fp32 DEST (carries an un-finalized fp32 DEST accumulation), Float16_b under 16-bit DEST (audit 2) | compute (`square_accumulate`) | compute (`rms_finalize`) | `square_accumulate` → `rms_finalize`, per block | **Shares-with candidate:** `cb_inv_rms` (same `R × T_interm` shape, back-to-back lifetimes). Not aliased: `rms_finalize` reads tile-row r of `cb_sumsq_partial` while packing row r of `cb_inv_rms` — the two are live in the same instant, and with `BulkWaitBulkPop` per row the reduce reserves the output before popping the input. Cost of keeping both: `R · T_interm`. **Disjoint-lifetime pair with `cb_x_gamma` / `cb_y_tiles`** (verifier audit): within a block `cb_sumsq_partial` is drained by `rms_finalize` before the first `scale_*` chunk packs, and the next block's `square_accumulate` cannot start before `normalize_block` ends (same compute thread), so the lifetimes are disjoint. Not aliased: the saving is at most `R · T_interm` (≤ 1/C of `cb_x_gamma`, < 1 % of the footprint at every Phase 0 shape), while aliasing needs the `alias_with` lockstep discipline across three helpers with different push/pop quanta. |
| `cb_inv_rms` | `R` (page = `T_interm`) | `R_b` (all rows of the block, read by every chunk) | `{tile_rows: spans → R, width_reduce: streams, width_stream: streams (re-read per chunk)}` | `T_interm` format: Float32 under fp32 DEST (`rsqrt` result consumed by an FPU multiply), Float16_b under 16-bit DEST (audit 2) | compute (`rms_finalize`) | compute (`scale_by_inv_rms`) | `rms_finalize` → `release_block`, per block | No share: concurrent with `cb_sumsq_partial` (above) and with every per-chunk CB. |
| `cb_gamma_row` (has_gamma, **not** allocated on the TILE-γ resident path since Refinement 3 — `gamma_row_pages()` = 0 there) | `Wt` (page = `gamma_row_page`: `32 · es_g` B for `Layout::ROW_MAJOR` γ; `segs · GAMMA_SEG_SLOT` for `Layout::TILE` γ: 2 segments = 128 B for Float32 / Float16_b, 3 segments = 192 B for Bfp8_b) | `Wt` pages (the whole staging row, read by every expansion) | `{tile_rows: streams (reused by every block), width_reduce: spans → Wt (not Wp; pad tiles are never expanded from it), width_stream: streams (read C at a time)}` | γ dtype (raw bytes) | writer (RM: one NoC read of the stick; TILE: the row-0 segments of each γ tile (`gamma_row_segments()`: face 0 / face 1 row 0; for Bfp8_b the 32 B exponent head holding both rows' shared exponents plus the two 32 B mantissa segments at byte 64 and 320), each into its own 64 B-aligned slot so DRAM source and L1 destination keep matching low address bits) | writer (scratch source of the expansion; no push/pop) | kernel | No share: resident for the kernel; it is the compact form (`1/32` of `Wt` γ tiles for RM, `1/16` at bf16 for TILE) that makes γ residency affordable when `cb_gamma_tiles` streams. TILE γ rows ≥ 1 never leave DRAM — `BroadcastDim::Row` reads row 0 only. **Deleted on the TILE-γ resident path (Refinement 3, `GAMMA_DIRECT_READ`)**: the writer reads each γ tile's row-0 bytes straight into its `cb_gamma_tiles` page, so the staging row has no consumer (pattern 1: pack/read into the destination). It stays for RM γ (one stick read, expanded) and streaming γ (re-expanded per (block, chunk) from L1 instead of re-reading DRAM). **Also deleted under the streaming L1 fallback (Refinement 5, `Geometry.gamma_stream_direct`)**: when the blocking search cannot fit the S-page TILE-γ staging row (e.g. a HEIGHT shard of 1 × 344 tiles at W = 11008: input + output shards 704 KB each leave ~52 KB, the staging row alone is 44 KB), `choose_blocking` retries without it and the writer re-reads each streaming chunk's row-0 prefix from DRAM per (block, chunk). Taken only when the staging-row search refuses, so no shape that fit before changes. |
| `cb_gamma_tiles` (has_gamma) | **resident** (`gamma_resident`): `Wp`; **streaming**: `stream_depth · C` (page = `T_g`) | resident: all `Wp` row-0 tiles for the whole kernel (pushed C at a time during the prologue, popped at kernel end); streaming: `C` (one chunk, reused across the `R_b` rows), capacity − live = the next chunk | `{tile_rows: streams (Row mapping reuses the same tiles for all R rows), width_reduce: resident → Wp / streams, width_stream: indexed by chunk offset (resident, `TileAddressing::Offset`) / spans → C (streaming)}` | γ dtype | **writer** (was reader in Phase 0 — perf lamp L4) | compute (`scale_by_gamma`) | resident: kernel; streaming: per (block, chunk) | No share: concurrent with `cb_x_gamma` and `cb_x_tiles`. Resident is chosen by `choose_gamma_resident()` only when `F(R, x_depth, resident)` fits at the blocking already chosen with streaming γ — it never costs block rows, `x_depth` or chunk width, so `_WIDE` shapes (W ≥ 12288) stay on the streaming path. |
| `cb_x_gamma` (has_gamma) | `R · C` (page = `T_interm`) | `R_b · C` (whole sub-block packed by `scale_by_gamma` before `scale_by_inv_rms` starts — sequential helpers on the same TRISCs) | `{tile_rows: spans → R, width_reduce: streams, width_stream: spans → C}` | `T_interm` format: Float32 under fp32 DEST (un-finalized product consumed by another FPU multiply), Float16_b under 16-bit DEST (audit 2) | compute | compute | per (block, chunk) | **Shares-with candidate:** `cb_y_tiles` (both `R·C`, compute-internal). Not aliased: `scale_by_inv_rms` reads `cb_x_gamma` and writes `cb_y_tiles` in the same instant. In-place (`cb_x_gamma` → `cb_x_gamma`) was rejected: the eltwise chain documents no CB in-place lifecycle (`chain.inl` has DEST-slot in-place only, `:603-626`), and with the CB full the output reserve would precede the input pop → deadlock. Different page format too (Float32 vs output dtype). |
| `cb_y_tiles` (is_rm) | `R · C` (page = `T_out`) | `R_b · C` (whole sub-block packed before `untilize` runs — sequential helpers) | `{tile_rows: spans → R, width_reduce: streams, width_stream: spans → C}` | output dtype — this pack **is** the op's final rounding point; `untilize` downstream is a lossless re-layout of output-dtype values, so a `Float32` page would carry bits that are discarded one step later (audit 2 reasoned, not defaulted) | compute (`scale_by_inv_rms`) | compute (`untilize`) | per (block, chunk) | No share: concurrent with `cb_x_gamma` (above) and with `cb_output` (`untilize` reads one, writes the other). TILE path does not allocate it: `scale_by_inv_rms` packs straight into `cb_output` (pack-into-destination, Rule 3 #1). |
| `cb_output` | `stream_depth · C` (page = `T_out`) | `C` (one tile-row × chunk being written); capacity − live = next quantum (double buffering) | `{tile_rows: streams → 1 tile-row per quantum, width_reduce: streams, width_stream: spans → C}` | output dtype | compute (`scale_by_inv_rms` TILE / `untilize` RM) | writer | per (block, chunk) | No share: concurrent with every compute-internal CB of the chunk; only CB the writer consumes. |
| `cb_partial_col` (R3, G > 1) | `R` (page = `T_interm`) | `R_b`: this core's collapsed Σx² partial per tile-row (column 0 valid) | `{tile_rows: spans → R, width_reduce: streams (collapsed), width_stream: streams}` | `T_interm` format (an un-finalized partial sum, audit 2) | compute (`collapse_block`, `reduce<SUM, REDUCE_ROW>` without post-op) | reader (one write per owner of the owner's rows — contiguous here — into its gather slots, own rows by loopback; Refinement 6) | per block | No share: it is the reader-consumed twin of `cb_inv_rms` (single-producer/single-consumer forbids packing the partial into a CB compute also reads), and it is live while `cb_sumsq_partial` is drained by the same reduce. |
| `cb_gather` (R3, G > 1) | `G · q`, `q = ceil(R / owners)` (`gather_pages()`; page = `T_interm`) | owner: `G · q_b` — contributor-major, contributor j's partials of the owned rows at slots `[j·q_b, j·q_b + n)` (Refinement 6; was `r·G + index`); non-owners of a block: not pushed — the CB exists on every core so the base address every contributor targets is the same everywhere | `{tile_rows: spans → q (this owner's share), contributors: spans → G}` | `T_interm` format | reader (owner: G−1 remote writes + its own loopback write, gated by `SEM_GATHER`; pushes the uniform `G · q_b` quantum so full blocks return the pointer to the base) | compute (owner: `reduce<SUM, REDUCE_COL, AccumulateViaAdd, Skip>` over `(G, owned)`, row pitch `q_b`, `BulkWaitBulkPop`) | per block | No share: the symmetric allocation IS the addressing mechanism of the push gather. Refinement 6 reduce-scatter: `owners` members each own `q` rows, so the buffer shrinks from `G · R` to `G · ceil(R / owners)` ≈ `R + G` pages per core — no longer the term that bounds R on wide groups (owners = 1, the root topology, keeps `G · R`). |
| `cb_rms_root` (R3, G > 1) | `q = ceil(R / owners)` (`scatter_rows()`; page = `T_interm`) | owner: `owned ≤ q` finalized `inv_rms` tiles awaiting the all-gather | `{tile_rows: spans → q}` | `T_interm` format | compute (owner) | reader (owners = 1: `McastArgs` sender → every member's `cb_inv_rms`; owners > 1: unicast of the owned run to every member + `SEM_INV`) | per block | No share: every member's `cb_inv_rms` is produced by the reader (the all-gather), so an owner cannot pack straight into it. The last owner of a ragged split pushes `owned < q` per full block, so its read pointer drifts: the reader sends a run that straddles the ring end as two writes and pops one page at a time (CB pointers wrap only on exact equality with the ring end). |
| `cb_sumsq_last` (TILE shard, W % 32 ≠ 0) | `R` (page = `T_interm`) | masked core: `R_b` raw x² tiles of the partial last tile | `{tile_rows: spans → R}` | `T_interm` format | compute | compute (`reduce` with `ReducePartialScaler::with_partial()`) | per block | No share: live together with `cb_sumsq_acc`. The caller's shard is read-only, so the pad columns are dropped by the partial scaler instead of being zeroed (lamp L5). |
| `cb_sumsq_acc` (TILE shard, W % 32 ≠ 0) | `R` (page = `T_interm`) | masked core: `R_b` collapsed Σx² of the full tiles — the reduce accumulator the partial tile is folded onto | `{tile_rows: spans → R}` | `T_interm` format | compute | compute (`Accumulate::at(cb_sumsq_acc, 1)` reload) | per block | No share: the accumulated reduce reads it while packing `cb_partial_col` / `cb_inv_rms`. |

**HEIGHT_SHARDED (regime R1-HS, Refinement 5)**: the R3 sharded inventory with G = 1 — every Tensix core holds full-W rows, `Topology.combine = False`, so `cb_partial_col` / `cb_gather` / `cb_rms_root` and the semaphore + multicast channel are not allocated; TILE: zero-copy `cb_x_tiles` / `cb_output`, one block = the whole shard unless L1 forces a split (`choose_block_rows` sharded branch); ROW_MAJOR: the R3 RM-shard re-layout. Data movement: 0 DRAM bytes of x and of the output (both resident), γ as in R1. Work split = the shard spec (`ttnn.corerange_to_cores(grid, n_shards, orientation)`), a ragged last shard carries fewer valid rows.
**Sharded TILE placements (R3)**: `cb_x_tiles` = `ttnn.cb_descriptor_from_sharded_tensor(input)` and `cb_output` = `ttnn.cb_descriptor_from_sharded_tensor(output)` — zero-copy, 0 B of their own; the reader pushes the core's rows once and compute packs strided (`TileAddressing::Strided`, row stride = shard width `S`) straight into the output shard. `cb_reduce_scaler` holds the `[full, partial]` scaler pair (2 pages) when W % 32 ≠ 0. **ROW_MAJOR shards (R3)**: shards are element-granular (e.g. 8- or 24-element slices, 205-stick shards), so `tilize<C>` cannot read them in place; `cb_x_rm_sticks` / `cb_x_tiles` / `cb_y_tiles` / `cb_output` keep their interleaved roles and the reader / writer copy sticks between them and the core's own shard with local L1 NoC transfers (no TensorAccessor, no DRAM). RM γ staging (`cb_gamma_row`) gets one extra page: an RM slice starts at any element, so its DRAM read starts at the aligned-down address.

### Inventory decisions (Rule 1 / Rule 3) recorded

| Phase boundary | Decision |
|----------------|----------|
| x → x² → Σx² | **No x² buffer.** Square and cross-tile sum fold into one FPU multiply with DEST accumulation per tile-row (Rule 3 #4); only one raw partial tile per row leaves DEST. A materialized x² row would be `Wt · T32` (1 MiB at W = 8192). |
| Σx² → mean → +ε → rsqrt | No intermediates: the three SFPU steps are the reduce helper's `post_reduce_op` on the DEST result. |
| x·γ → ·inv_rms (TILE) | One intermediate (`cb_x_gamma`, C-chunk sized, not row sized); final product packs into `cb_output` (Rule 3 #1). |
| ·inv_rms → untilize (RM) | One intermediate (`cb_y_tiles`), needed because `untilize` reads a CB. |
| γ | Compact resident row + C-chunk expansion instead of `Wt` resident γ tiles (32× smaller). |
| x | Resident once per block (the residency that keeps x at 1 DRAM crossing) — the only Wt-scaled buffer. |

### Symbol table

| Symbol | Meaning | Bound | Predicate establishing the bound |
|--------|---------|-------|----------------------------------|
| `R` = `block_rows` | tile-rows per block | `1 ≤ R ≤ ceil(rows_g1 / MIN_BLOCKS_PER_CORE)` | closed form in `op_design.md` → Work Distribution, additionally capped so a core with ≥ 2 tile-rows walks ≥ `MIN_BLOCKS_PER_CORE = 2` blocks (lamp L1: x_depth = 2 then actually prefetches); R1 predicate `F(1, 1) ≤ L1_budget` |
| `R_b` | rows in the current (possibly ragged) block | `1 ≤ R_b ≤ R` | `min(R, rows_this_core − block_idx·R)` |
| `C` = `w_chunk_tiles` | streaming chunk width (tiles) | `1 ≤ C ≤ cap ≤ W_CHUNK_TILES = 32` | implementation: `cap` = the first of `32, 16, 8, …, 1` whose footprint admits `x_depth = 2` (else the first that fits at `x_depth = 1`); `num_w_chunks = ceil(Wt / cap)`, `C = ceil(Wt / num_w_chunks)` (balanced chunks) |
| `Wt` | `ceil(W / 32)` tiles | bounded by the R1 predicate (Phase 0 INPUTS: ≤ 256) | `F(1, 1) ≤ L1_budget` |
| `Wp` = `width_padded_tiles` | `num_w_chunks · C` | `Wt ≤ Wp < Wt + num_w_chunks` | definition |
| `x_depth` | blocks of x in flight | `{1, 2}` | `1` when every core walks a single block (`rows_g1 ≤ 1`: nothing to prefetch, the second buffer would be dead L1); else `2` iff `R2 ≥ 1` (closed form) |
| `stream_depth` | depth of the three streaming CBs | `2` | host constant |
| `T_in`, `T_out` | input / output tile bytes | `{2048, 4096}` B | dtype ∈ {bfloat16, float32} |
| `T_g`, `es_g` | γ tile bytes, γ element bytes | `{2048, 4096}` B, `{2, 4}` B | γ dtype ∈ {bfloat16, float32} |
| `T_interm` | page bytes of the compute-internal CBs (`cb_sumsq_partial`, `cb_inv_rms`, `cb_x_gamma`) | 4096 B (Float32) under fp32 DEST, 2048 B (Float16_b) under 16-bit DEST | derived ONCE in `intermediate_format()` (`rms_norm_program_descriptor.py`) — Float32 under fp32 DEST, Float16_b otherwise (any input dtype) — and used by both the CB declarations and `footprint()` |
| `L1_budget` | bytes available for CBs per Tensix core | device-reported | `ttnn.get_memory_view(device, ttnn.BufferType.L1).largest_contiguous_bytes_free_per_bank` |

### Total per-core footprint (bytes)

```
F(R, x_depth) =   x_depth · R · Wp · T_in                       # cb_x_tiles          — scales with R, Wp (W), x_depth
                + R · T_interm                                  # cb_sumsq_partial    — scales with R
                + R · T_interm                                  # cb_inv_rms          — scales with R
                + has_gamma · R · C · T_interm                  # cb_x_gamma          — scales with R, C
                + is_rm     · R · C · T_out                     # cb_y_tiles          — scales with R, C
                + 2048                                          # cb_reduce_scaler    — constant
                + stream_depth · C · T_out                      # cb_output           — scales with C
                + is_rm     · stream_depth · C · T_in           # cb_x_rm_sticks      — scales with C
                + has_gamma · (gamma_resident ? Wp : stream_depth · C) · T_g   # cb_gamma_tiles — C (streaming) / W (resident)
                + has_gamma · !direct · Wt · gamma_row_page     # cb_gamma_row        — scales with W only; 0 when
                                                                #   direct = TILE γ ∧ gamma_resident (Refinement 3)
```

Only `cb_x_tiles` (and, at 1/32 the rate, `cb_gamma_row`) scale with W; everything else is bounded by `R` and `C`.

R3 adds `combine_bytes(R) = (R + G · ceil(R / owners) + ceil(R / owners)) · T_interm` (`cb_partial_col`, `cb_gather`, `cb_rms_root`; Refinement 6 — was `(2 + G) · R · T_interm` per row) and, per tile-row, `masked · 2 · T_interm` (`cb_sumsq_last`, `cb_sumsq_acc`); on a TILE shard the `x_depth · R · Wp · T_in` (`cb_x_tiles`) and `stream_depth · C · T_out` (`cb_output`) terms are 0 (zero-copy) and W is the core's slice. Sharded cores take the coarsest R that fits (up to the whole shard); interleaved cores keep the `MIN_BLOCKS_PER_CORE` cap.

Worst Phase 0 INPUTS cell — `(1, 1, 32, 8192)` float32, float32 γ, `Layout::ROW_MAJOR` (Wt = Wp = 256, C = 8 — the chunk-cap search falls back from 32/16 to 8 here — R = 1, `x_depth = 1`): `1,048,576 + 4,096 + 4,096 + 32,768 + 32,768 + 2,048 + 65,536 + 65,536 + 65,536 + 32,768 = 1,353,728 B` (1.29 MiB); same cell in `Layout::TILE`: `1,255,424 B` (1.20 MiB). With `x_depth = 2` the TILE cell would need 2.20 MiB, so the closed form selects `x_depth = 1` there; every narrower INPUTS row gets `x_depth = 2`.

## Data-movement budget

R1 (built). `N_tiles = tensor_tile_rows · Wt`; `active_cores = min(grid cores, tensor_tile_rows)`.

| Tensor | DRAM crossings | Why that many | Cross-core traffic added |
|--------|----------------|---------------|--------------------------|
| input x | 1 (`N_tiles · T_in` B, TILE; `num_sticks · W · es` B, RM) | `cb_x_tiles` keeps each block resident across the reduction pass and the scaling pass | none |
| γ | `active_cores` × one row. **TILE γ, resident path (Refinement 3, default `GAMMA_DIRECT_PREFIX`)**: `active_cores · Wt` reads of `gamma_row0_prefix_bytes` (544 B at bf16) straight into `cb_gamma_tiles` — 8192×7168 moves 120 · 224 · 544 B = 14.6 MB, 6 % of the 235 MB x + output traffic (measured: still faster on prefill than the 2·Wt-request segment forms, see the Refinement 3 notes). RM γ: `active_cores · W · es_g` B in one read per core (the single γ page lives in one DRAM bank). TILE γ: `active_cores · segs·Wt` reads of `seg_bytes` B (row-0 segments only: 2 × `16 · es_g` B for Float32 / Float16_b; 3 × 32 B — exponent head + 2 mantissa rows — for Bfp8_b) | each core fills its staging row once (writer RISC-V) and expands it locally; bytes ≤ 1/32 of x. **Measured cost on many-row prefill is the request count, not the bytes**: with TILE γ, 120 cores × 2·Wt tiny reads ≈ the x read request count at W = 7168, +3–10 % device-ns over RM γ; ablating the fill alone returns TILE γ to the no-γ time (8192×5120: 399 vs 398 µs). The fix is G-MC (read once + multicast), Refinement 6 | none (multicast = G-MC, deferred) |
| output | 1 (same bytes as x) | written once, per C-tile quantum | none |
| scaler, partials, inv_rms | 0 | generated and consumed on-core | none |

Totals per tier: DRAM = `2 · |x| + active_cores · W · es_g` B (≤ `2.03 · |x|`); cross-core NoC = 0 B; core-local L1 reads of x: 2 passes over the resident block (square_accumulate reads each tile once as both FPU operands; `scale_by_gamma` / `scale_by_inv_rms` read it once more).

R3 (built, Refinement 4). `G` = cores per row-group, `rows` = tensor tile-rows, `T_i` = `T_interm`.

| Tensor | DRAM crossings | Why that many | Cross-core traffic added |
|--------|----------------|---------------|--------------------------|
| input x | interleaved: 1 (each core reads only its W-slice of its tile-rows, resident for both passes); sharded: 0 (already in L1; TILE zero-copy, RM re-laid locally) | slices are resident | none |
| γ | one W-slice per core: `active_cores · S · (row-0 prefix)` B (TILE) — a row-group reads γ exactly once in total | each core needs only its slice | none (G-MC still deferred, Refinement 6) |
| output | interleaved: 1; sharded: 0 (packed into the output shard) | | none |
| Σx² partials | 0 | | per tile-row: `G − 1` partial tiles to the row's owner (`(G−1) · T_i` B), sent as one write per (contributor, owner) per block + one `SEM_GATHER` increment per contributor per owner per block |
| inv_rms | 0 | | owners = 1: per tile-row one `T_i` tile multicast to the `G − 1` other members (+ the receiver handshake per block); owners > 1 (Refinement 6 reduce-scatter): per tile-row `G − 1` unicast `T_i` tiles from its owner (one write per member per owner per block) + one `SEM_INV` increment per owner per member per block. Same bytes into each member as the multicast, `G − 1`× the injected bytes from the owners — measured cheaper than G concurrent multicasts into one row-group |

Example: 32×7168 interleaved (G = 23, 1 tile-row, bf16 intermediates): 22 × 2 KiB gathered + one 2 KiB multicast vs 917 KB of x + output DRAM traffic.

> Cheapest-traffic split considered: `tile_rows` across cores, rows resident (R1) — it is the cheapest: 1 DRAM crossing per tensor and 0 cross-core bytes. Implemented: R1. The W-split R3 moves the same DRAM bytes plus `G_w` partial tiles (4 KiB each) to a root and one 4 KiB `inv_rms` multicast per tile-row of each row-group; it is deferred because every Phase-0-SUPPORTED cell satisfies R1's residency predicate and the only cells beyond it are outside Phase 0 SUPPORTED (`_WIDE`: `gamma_layout=TILE`; perf cases: `fp32_dest_acc_en=False`); its gain is occupancy on short-`tile_rows` shapes, a measured trade (lamp L2).

## Implementation notes (ttnn-implementer)

- **Chunk width.** The design fixed `C = min(Wt, 8)`; measurement (Blackhole, 120 Tensix cores) showed
  the RM path starved on 512 B stick-segment reads (`(1,1,8192,1024)` bf16 RM: 125 µs at C = 8,
  100.5 µs at C = 16, 93 µs at C = 32; TILE flat). `W_CHUNK_TILES` is now a cap of 32 and C is chosen
  per call from the same footprint `F(R, d)` (symbol table above). Every CB capacity formula in this
  ledger is unchanged — only the value of C differs.
- **Block rows.** `R` is capped at `ceil(rows_g1 / MIN_BLOCKS_PER_CORE)` (lamp L1). Together with the
  C change: RM `(1,1,8192,1024)` 125 → 89.5 µs, TILE 89.5 → 86 µs.

## Refinement 1 notes (TILE γ, γ supply on the writer)

- **γ supply moved to the writer RISC-V** (lamp L4) for both γ `Layout`s: the reader now moves only x,
  so its duration with γ equals its no-γ duration (decode 32×7168: 7.5 µs vs 7.4 µs; was 32.7 µs).
- **Resident vs streaming γ** (`GAMMA_RESIDENT` knob, `choose_gamma_resident()`): resident = `Wp`
  expanded tiles once per kernel, pushed chunk by chunk (fill of chunk c+1 overlaps expansion of chunk
  c; compute waits cumulatively); streaming = one chunk per (block, chunk) step, pushed one step ahead
  of the output drain, TILE staging filled lazily one chunk ahead. Expansion uses stateful
  single-packet NoC reads.
- **`x_depth = 1` on single-block cores** frees `R·Wp·T_in` B, which is what lets decode / 640-row
  shapes (one tile-row per core) take the resident path.

## Refinement 3 notes (perf at the exact interleaved `_perf_case` config)

- **CB inventory change**: `cb_gamma_row` is gone on the TILE-γ resident path (`gamma_direct()` ≠ OFF). The writer reads row 0 of every γ tile from DRAM into its `cb_gamma_tiles` page:
  - `GAMMA_DIRECT_PREFIX` (default): one read of bytes `[0, gamma_row0_prefix_bytes)` per γ tile — 544 B bf16, 1088 B fp32, 352 B Bfp8_b;
  - `GAMMA_DIRECT_SEGMENTS`: one read per row-0 segment, to the same page offset.

  No other CB changes capacity. Removing the staging row lowers `F(R, x_depth, resident=True)`, so `choose_gamma_resident()` admits resident γ on slightly more shapes.
- **Compute block, not a buffer**: `SCALE_BLOCK_TILES` (8, clamped by the chain to `DEST_AUTO_LIMIT`) is the DEST window of `scale_by_gamma` / `scale_by_inv_rms`. It adds no L1. `cb_x_gamma` is now waited upfront and popped at the end of the inv_rms chain, and the outputs are reserved / pushed per tile-row (C pages). Both quanta fit the existing `R·C` / `stream_depth·C` capacities.
- **Measured trade, `PREFIX` vs `SEGMENTS`** (bf16, 16-bit DEST, HiFi2, bf16 TILE γ, Blackhole 1350 MHz, device ns):

  | shape | PREFIX | SEGMENTS |
  |---|---:|---:|
  | 640×7168 | 56.1 µs | 55.6 µs |
  | 7168×1024 | 76.8 µs | 81.4 µs |
  | 8192×1024 | 85.5 µs | 89.1 µs |
  | 8192×7168 | 580.9 µs | 585.3 µs |

  On prefill the request count costs more than the extra bytes, so PREFIX ships. Decode is identical with either mode: the γ supply (11 k cycles at 32×7168) hides behind `square_accumulate`.

## Verifier notes (Phase 0 verification)

- **DRY fix.** `footprint()` priced the compute-internal CBs at a hard-coded Float32 tile while the CB
  declarations derived their format from `fp32_dest_acc_en`; both now read `Geometry.T_interm`
  (`intermediate_format()`), so the `fp32_dest_acc_en=False` refinement changes one place.
- **γ expansion mechanism.** The per-(block, chunk) row-0 expansion now uses local L1→L1 NoC reads
  (2 per tile, one barrier per chunk) instead of RISC-V word-by-word `volatile` copies: 1.15–1.28× on
  every γ-bearing decode / 640-row perf shape, no regression elsewhere (Blackhole, 1350 MHz). No CB
  change; the γ path still serializes the reader (it cannot prefetch block k+1 until compute has drained
  block k's γ chunks) — carried by Refinement 1 in `op_requirements.md`.
- **`MIN_BLOCKS_PER_CORE = 2` re-measured in isolation** (TILE input, bf16, HiFi2, fp32 DEST): vs `1`,
  1.02–1.05× on the many-row γ shapes (7168×1024, 7168×2048, 8192×1024), flat (±1 %) elsewhere. The
  departure from "coarsest block that fits" stands on measurement.

## Refinement 4 notes (regime R3: cross-core Σx² combine)

- **New CBs**: `cb_partial_col`, `cb_gather`, `cb_rms_root` (combine), `cb_sumsq_last`, `cb_sumsq_acc` (TILE-shard W-pad mask) — rows above. `cb_inv_rms` keeps its shape; its producer is the reader (the multicast) whenever G > 1.
- **Zero-copy sharded TILE**: `cb_x_tiles` / `cb_output` backed on the input / output shard — the largest TILE-shard footprint terms disappear; the gather (`G · R · T_interm`) is what bounds R on wide groups.
- **Block size (sharded)**: the coarsest R that fits, up to the whole shard (no `MIN_BLOCKS_PER_CORE` cap: nothing to prefetch, and every extra block is one more combine round, lamp L2).
- **Interleaved layout knobs** (`R3_ROWS_PER_GROUP = 2`, `R3_MAX_GROUP_CORES = 24`, `R3_MIN_SLICE_TILES = 4`, occupancy gate `R3_OCCUPANCY_GAIN = 2` + `R3_MIN_GROUP_CORES = 4`) change only the slice width `S` and the core count; every CB formula is the R1 one over the slice.

## Refinement 6 notes (R3 combine as a reduce-scatter; column-0 finalize)

- **Topology knob** `R3_SCATTER_ROWS_PER_OWNER` (default 1): `owners = min(G, rows_max // knob)`, then trimmed to
  `block_rows // knob` when the footprint at the trimmed value still fits. owners = 1 is the Refinement 4 root
  reduce + multicast (decode: one tile-row per group). Resized CBs: `cb_gather` `G · R` → `G · ceil(R / owners)`,
  `cb_rms_root` `R` → `ceil(R / owners)` (rows above). New semaphore `SEM_INV` (id 1, shared with the absent
  multicast channel's first id). No CB added or removed.
- **Block-size consequence**: the gather no longer bounds R on wide groups, so e.g. BLOCK 8192×1024 (G = 8, 32
  tile-rows per core) runs one block instead of 21 + 11.
- **`choose_block_rows`** is now a monotone search over `footprint(R)` (`_max_block_rows`) — the combine term is no
  longer linear in R. For every footprint without a combine the search returns the old closed form exactly.
- **Data movement**: DRAM bytes unchanged (sharded x / output never leave L1; interleaved R3 reads each slice
  once). Cross-core bytes per tile-row unchanged for the gather; the all-gather injects `G − 1` copies per row
  from its owner instead of one multicast (see the table above).
