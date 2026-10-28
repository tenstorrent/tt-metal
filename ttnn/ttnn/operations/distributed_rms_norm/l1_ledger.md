# L1 Ledger: distributed_rms_norm

Schema and audits: `.claude/references/l1-footprint-discipline.md`. Blocking axes (from `op_design.md` → Blocking Model):
`tile_rows` (block `R`, wave `q_w`, owner run `q`), `width` (slice `S`, padded `Wp`, chunk `C`), `devices` (`G`),
`slices_per_group` (`G_w`). Every row accounts for all four.

Per **worker** core. Port cores hold no CBs: their only L1 is the landing shard (last section).

| CB | Capacity (pages) | Live set | Axis accounting | Page format | Producer | Consumer | Lifetime | Shares with / why not |
|----|------------------|----------|-----------------|-------------|----------|----------|----------|-----------------------|
| `cb_x_tiles` | interleaved `x_depth · R · Wp` (`x_depth = 1` when `R = rows_this_core`); sharded: the input shard (zero-copy, not allocated) | `R · Wp`: read once, consumed by square, γ pass and inv pass of every wave | `{tile_rows: spans → R, width: spans → Wp, devices: — (none), slices_per_group: — (none)}` | input dtype (Float32 / Float16_b / Bfp8_b) | reader | compute | block | none: live for the whole block, concurrent with everything; sharded = the caller's shard, already zero-copy |
| `cb_reduce_scaler` | 1 (2 on the sharded W-pad mask path: full + partial scaler) | 1 (2) — constant | `{tile_rows: —, width: —, devices: —, slices_per_group: —}` constant | Float16_b (the reduce scaler is bf16 by contract) | reader | compute | kernel | none: constant for the kernel's life |
| `cb_sumsq_partial` | `q_w` | `q_w`: one raw Σx² tile per tile-row of the wave, between the square chain and the collapse reduce | `{tile_rows: spans → q_w, width: streams (accumulated in DEST), devices: —, slices_per_group: —}` | intermediate (Float32 fp32 DEST / Float16_b 16-bit DEST) | compute | compute | wave | not `cb_partial_col`: same lifetime boundary but the collapse reads all `q_w` raw tiles (BulkWaitBulkPop) while writing their collapses — in place would need real credit on one CB for two phases (`l1-footprint-discipline.md` Rule 3.2); 4 KiB at the focus case |
| `cb_inv_rms` | `R` | `R`: every wave's inv until `release_block` (waves land in `q_w` quanta) | `{tile_rows: spans → R, width: —, devices: —, slices_per_group: —}` | intermediate | reader (pushes after `SEM_INV`) | compute | block | none: remote writes land at fixed slots while earlier waves are still being consumed |
| `cb_gamma_tiles` (γ) | `Wp` | `Wp`: the slice's row-0 γ tiles, resident | `{tile_rows: —, width: spans → Wp, devices: —, slices_per_group: —}` | γ dtype | writer | compute | kernel | none: resident for the kernel |
| `cb_x_gamma` (γ) | `R · Wp` | `R · Wp`: the γ pass runs over the whole block before any inv arrives (the stall shadow) | `{tile_rows: spans → R, width: spans → Wp, devices: —, slices_per_group: —}` | intermediate | compute | compute | block | not in place in `cb_x_tiles`: different page format (input dtype vs intermediate) — packing x·γ into a bf16 / Bfp8_b page would round before the inv multiply; explicit pipelining decision (lamp L4 measures a per-wave γ pass that would shrink it to `q_w · Wp`) |
| `cb_partial_col` | `q_w` | `q_w`: collapsed slice partials of one wave until the reader has sent them | `{tile_rows: spans → q_w, width: —, devices: —, slices_per_group: —}` | intermediate | compute | reader | wave | not `cb_chip_partial`: same thread pair but distinct poll-loop events (contribute every wave vs publish only on owned waves); merging would force one interleaved stream order |
| `cb_gather` | `G_w · q` | `G_w · q`: the owner's contributor-major gather slots of one wave (remote faces 0/2 writes) | `{tile_rows: spans → q, width: —, devices: —, slices_per_group: spans → G_w}` | intermediate | reader | compute | wave (owner) | not `cb_stats`: wave w+1's gather fills while wave w's stats wait for the fabric (concurrent by the wave pipeline); allocated on every worker because owners rotate (same L1 address on every member) |
| `cb_chip_partial` | `q` | `q`: chip-local Σx² of the owner's rows of one wave | `{tile_rows: spans → q, width: —, devices: —, slices_per_group: —}` | intermediate | compute | reader | wave (owner) | not `cb_rms_root`: wave w+1's chip partial is produced while wave w's root is still in flight (pipelining) |
| `cb_stats` | `G · q` | `G · q`: the G devices' partials of each owned row, column 0, position order | `{tile_rows: spans → q, width: —, devices: spans → G, slices_per_group: —}` | intermediate; unpacked to DEST with `UnpackToDestFp32` under fp32 DEST | reader | compute | wave (owner) | not `cb_gather`: concurrent (above); pages pre-zeroed once, fixed ring positions (push quantum `G · q` = capacity) |
| `cb_rms_root` | `q` | `q`: finalized inv of the owner's rows before the all-gather | `{tile_rows: spans → q, width: —, devices: —, slices_per_group: —}` | intermediate | compute | writer (reader when γ streams — base rule) | wave (owner) | not `cb_chip_partial`: concurrent across waves (pipelining) |
| `cb_sumsq_last` (sharded masked) | `q_w` | `q_w`: x² of the partial last tile | `{tile_rows: spans → q_w, width: — (one tile), devices: —, slices_per_group: —}` | intermediate | compute | compute | wave | not allocated unless sharded and `W_local % 32 != 0`; not shared with `cb_sumsq_partial`: both live inside one collapse (full tiles + masked tile accumulated) |
| `cb_sumsq_acc` (sharded masked) | `q_w` | `q_w`: collapse accumulator of the full tiles | `{tile_rows: spans → q_w, width: —, devices: —, slices_per_group: —}` | intermediate | compute | compute | wave | as above (base mechanism) |
| `cb_output` | interleaved `stream_depth · C`; sharded: the output shard (zero-copy, not allocated) | `C` per tile-row in flight; capacity `2·C` = double buffer | `{tile_rows: streams, width: spans → C, devices: —, slices_per_group: —}` | output dtype | compute | writer | chunk | none: streaming to the writer; sharded = the caller's output shard |
| `cb_record_scratch` | `G · q` | `G · q` records of 128 B: own (published, then reused as the own entry of the fold) + `G − 1` peers read from the ports | `{tile_rows: spans → q, width: —, devices: spans → G, slices_per_group: —}` | raw fp32 records (not a tile format) | reader | reader (scratch, never pushed) | kernel | not `cb_stats`: different page size and layout (packed 128 B records vs column-0 tile pages); the scatter reads one and writes the other |

## Symbol table

| Symbol | Bound | Predicate that establishes it |
|--------|-------|-------------------------------|
| `R` | `rows_this_core ≤ ceil(T / G_h)`; halves with `x_depth = 2` if the footprint does not fit | `fused_split()` footprint check (Work Distribution) |
| `q_w` | `ceil(rows_max / min(rows_max, WAVES_MAX))`, `WAVES_MAX = 8` | `fused_split()` |
| `q` | `ceil(q_w / owners)`, `owners = min(G_w, q_w)` | `fused_split()` / sharded split |
| `S`, `Wp` | `S ≤ Wt`; `Wp = ceil(S/C)·C`, `C = min(S, W_CHUNK_TILES = 32)` | `fused_split()` (interleaved) / shard width (sharded) |
| `G_w` | interleaved `≤ MAX_GROUP_CORES = 24`; WIDTH-sharded: shard count (≤ 86 for every INPUTS geometry) | `fused_split()` / shard spec + footprint check (refuse otherwise) |
| `G` | mesh dim size (≤ 8 on every target system) | mesh shape |
| `T_in`, `T_out` | 4096 (fp32) / 2048 (bf16) / 1088 (Bfp8_b) | dtype |
| `T_g` | 4096 / 2048 | γ dtype |
| `T_i` | 4096 under fp32 DEST, 2048 under 16-bit DEST | `fp32_dest_acc_en` (`intermediate_format()`, base single source) |
| `stream_depth` | 2 | host constant |

## Total per-core footprint

```
F = x_depth·R·Wp·T_in                          [interleaved only]          scales with R, S
  + 2048·(1 + masked)                                                      constant
  + T_i·( q_w + R + q_w + G_w·q + q + G·q + q )   sumsq, inv, partial_col, gather, chip, stats, root
  + has_gamma·( Wp·T_g + R·Wp·T_i )                                         scales with S (and R)
  + masked·2·q_w·T_i                                [sharded masked only]
  + stream_depth·C·T_out                            [interleaved only]     scales with C
  + G·q·128                                                                 scales with G
```

Focus case (640×1792 bf16, fp32 DEST, bf16 TILE γ; R 3, Wp 4, C 4, q_w 1, q 1, G_w 14, G 4):
x 24,576 + scaler 2,048 + sumsq 4,096 + inv 12,288 + γ 8,192 + x·γ 49,152 + partial 4,096 + gather 57,344
+ chip 4,096 + stats 16,384 + root 4,096 + output 16,384 + records 512 = **203,264 B (≈ 199 KiB)** of ~1.4 MiB.
The two largest terms are `cb_gather` (`G_w`-scaled, lamp L5 would cut it to 128 B rows) and `cb_x_gamma`
(`R·Wp`-scaled, lamp L4).

**Port core:** landing shard `num_waves · G · seg_stride` bytes, `seg_stride = align_up(max_l n_pub(l) · q · 128, 64)`.
Focus: 3 · 4 · 512 = 6 KiB. No CBs.

## Data-movement budget

Per device and call. `pfx` = TILE γ row-0 prefix bytes per tile (544 bf16 / 1088 fp32, base `gamma_row0_prefix_bytes`).

### Focus: 640 × 1792, bf16, TILE γ bf16, Ring G = 4, L = 2 (S 4, G_w 14, G_h 7, 3 waves)

| Tensor | DRAM crossings | Why that many | Cross-core traffic added |
|--------|----------------|---------------|--------------------------|
| x | 1 (2,293,760 B read) | resident in `cb_x_tiles` for square, γ pass and inv pass (R = the whole assignment) | — |
| out | 1 (2,293,760 B write) | written once per tile, padding tiles never stored | — |
| γ | `G_h = 7` reads of each tile's prefix (7 · 56 · 544 = 213,248 B, 9.3 % of x) | reuse-shared, read per row-group (G-MC deferred) | — |
| Σx² partials (on chip) | 0 | L1 only | per tile-row 13 contributor tiles × faces 0/2 (2,048 B) = 26,624 B → 532,480 B per chip |
| inv_rms (on chip) | 0 | L1 only | per tile-row 13 × 2,048 B = 26,624 B → 532,480 B per chip |
| records (chip ↔ port) | 0 | L1 only | publish 21 records (20 + 1 zero) × 128 B × 2 ports = 5,376 B; collect 21 × 3 × 128 = 8,064 B |
| records (fabric) | 0 | — | own 2,688 B to each direction + 1 relayed set FWD; busiest link (link 0 FWD) 2 × 12 records × 128 = 3,072 B |

Totals: DRAM 4,800,768 B; on-chip NoC ≈ 1.08 MB; fabric ≈ 8 KiB sent per device.

### Decode: 32 × 1792 (S 4, G_w 14, G_h 1, 1 wave, L = 1 used)

| Tensor | DRAM crossings | Why that many | Cross-core traffic added |
|--------|----------------|---------------|--------------------------|
| x / out | 1 / 1 (114,688 B each) | resident | — |
| γ | 1 (56 · 544 = 30,464 B, 27 % of x) | one row-group | — |
| partials + inv (on chip) | 0 | | 2 · 13 · 2,048 = 53,248 B |
| records | 0 | | 128 B published to 2 ports; 3 × 128 B collected; fabric 128 B own + 128 B relayed |

### 2048 × 1280 (S 5, G_w 8, G_h 13, 5 waves)

| Tensor | DRAM crossings | Why that many | Cross-core traffic added |
|--------|----------------|---------------|--------------------------|
| x / out | 1 / 1 (5,242,880 B each) | resident (25 tiles per core) | — |
| γ | 13 (13 · 40 · 544 = 282,880 B, 5.4 % of x) | one read per row-group | — |
| partials + inv (on chip) | 0 | | per tile-row 2 · 7 · 2,048 = 28,672 B → 1,835,008 B |
| records | 0 | | 65 records (64 + 1 zero) × 128 = 8,320 B own per device |

> Cheapest-traffic split considered: D3 + G-MC (γ read once per slice column and multicast down the `G_h`
> row-groups) — DRAM −183 KB (−3.8 %) at the focus case, −253 KB (−2.4 %) at 2048×1280; adds one NoC multicast of
> `S` γ tiles per slice. Implemented: D3 with per-row-group γ reads (the same split; G-MC is a stepping stone on
> top of it). Deferred because the Phase 0 core order makes same-slice cores strided across the grid and
> `ttnn.Mcast` forms only equal consecutive groups (base changelog Perf 1, helper friction); reaching it is a
> column-major core reorder per slice, the CBs are unchanged. Among the splits that need no new mechanism, the
> implemented D3 is the cheapest at the DRAM tier (D1 reads γ 20× at the focus case; the `G_h = 1` D3 variant
> saves the same γ bytes as G-MC but adds 3.4 MB of on-chip combine traffic and a fan-in of 56).
