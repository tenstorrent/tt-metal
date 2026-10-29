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
| `cb_inv_rms` | `num_waves · q_w` (≥ R) | every wave's inv until `release_block`; pushed in `q_w` quanta once SEM_INV covers the wave | `{tile_rows: spans → R, width: —, devices: —, slices_per_group: —}` | intermediate | reader (pushes after `SEM_INV`) | compute | block | none: remote writes land at fixed slots (no wrap) while earlier waves are still being consumed |
| `cb_gamma_tiles` (γ) | `Wp` | `Wp`: the slice's row-0 γ tiles, resident | `{tile_rows: —, width: spans → Wp, devices: —, slices_per_group: —}` | γ dtype | writer | compute | kernel | none: resident for the kernel |
| `cb_x_gamma` (γ) | `R · WS` (whole block, default); `q_w · WS` with the per-wave γ fallback (`plan.gamma_per_wave`, chosen only when the whole block does not fit) | `R · WS`: the γ pass runs over the whole block before any inv arrives (the stall shadow) | `{tile_rows: spans → R (fallback: q_w), width: spans → WS, devices: —, slices_per_group: —}` | intermediate | compute | compute | block (fallback: wave) | not in place in `cb_x_tiles`: different page format (input dtype vs intermediate); the fallback trades the stall shadow for L1 (measured need: WIDTH_SHARDED 2048×2048 with γ, fp32 DEST) |
| `cb_partial_col` | `2 · q_w` | `q_w`, two waves deep: compute collapses wave w+1 while the reader still sends wave w (chip_sum lags the square by one wave) | `{tile_rows: spans → q_w, width: —, devices: —, slices_per_group: —}` | intermediate | compute | reader | wave | not `cb_chip_partial`: same thread pair but distinct events (contribute every wave vs publish only on owned waves) |
| `cb_gather` | `max_owned · G_w · q`, `max_owned = ceil(num_waves · owners / G_w)` | one `G_w · q` region per wave this member owns (ordinal o), at fixed addresses | `{tile_rows: spans → q, width: —, devices: —, slices_per_group: spans → G_w}` | intermediate | reader | compute | wave (owner) | not one wave quantum: remote contributors can run a wave ahead of the owner's compute, so an owner of two waves needs both regions (WAR); SEM_GATHER keeps one bit field per ordinal |
| `cb_chip_partial` | `2 · q` | `q`, two deep: D1 collapses wave w+1 into it while the reader still publishes wave w | `{tile_rows: spans → q, width: —, devices: —, slices_per_group: —}` | intermediate | compute | reader | wave (owner) | not `cb_rms_root`: wave w+1's chip partial is produced while wave w's root is still in flight |
| `cb_stats` | `G · q` | `G · q`: the G devices' records of each owned row, column 0, position order | `{tile_rows: spans → q, width: —, devices: spans → G, slices_per_group: —}` | **Float32 always** (records are exact fp32; `UnpackToDestFp32` under fp32 DEST, unpacker conversion under 16-bit DEST) | reader | compute | wave (owner) | not `cb_gather`: concurrent; pages pre-zeroed once after Phase A, fixed ring positions (push quantum = capacity) |
| `cb_rms_root` | `q` | `q`: finalized inv of the owner's rows before the all-gather | `{tile_rows: spans → q, width: —, devices: —, slices_per_group: —}` | intermediate | compute | writer (reader when γ streams — base rule) | wave (owner) | not `cb_chip_partial`: concurrent across waves (pipelining) |
| `cb_sumsq_last` (sharded masked) | `q_w` | `q_w`: x² of the partial last tile | `{tile_rows: spans → q_w, width: — (one tile), devices: —, slices_per_group: —}` | intermediate | compute | compute | wave | not allocated unless sharded and `W_local % 32 != 0`; not shared with `cb_sumsq_partial`: both live inside one collapse (full tiles + masked tile accumulated) |
| `cb_sumsq_acc` (sharded masked) | `q_w` | `q_w`: collapse accumulator of the full tiles | `{tile_rows: spans → q_w, width: —, devices: —, slices_per_group: —}` | intermediate | compute | compute | wave | as above (base mechanism) |
| `cb_output` | interleaved output: `stream_depth · C`; sharded output: the output shard (zero-copy, not allocated) | `C` (shard input: the chunk width) per tile-row in flight | `{tile_rows: streams, width: spans → C, devices: —, slices_per_group: —}` | output dtype | compute | writer | chunk | none: streaming to the writer; keyed on the OUTPUT placement (a sharded input may stream to an interleaved output) |
| `cb_record_scratch` | `(G + 1) · q` pages of 128 B | `q` publish staging + `G · q` collected records (own record re-read from the port landing, so wave w+1's publish never overwrites it) | `{tile_rows: spans → q, width: —, devices: spans → G, slices_per_group: —}` | raw fp32 records | reader | reader (scratch) | kernel | not `cb_stats`: packed 128 B records vs column-0 tile pages |

> **Implementation deltas (Phase 0, implementer):** rows above updated to what the kernels allocate —
> `cb_inv_rms` = `num_waves · q_w`, `cb_partial_col` 2 waves deep, `cb_chip_partial` 2 deep, `cb_gather` per owned
> wave, `cb_stats` always Float32, `cb_record_scratch` `(G + 1) · q`, `cb_x_gamma` with a per-wave fallback,
> `cb_output` keyed on the output placement. `x_depth = 2` (blocks) is **not built**: a shape whose whole block does
> not fit is refused (verifier: filed as Refinement 1, a port of the base op's L1 fallbacks) (`NotImplementedError`): 128×8192 HEIGHT_SHARDED, 4096×1024 WIDTH_SHARDED, prefill ≥ 8192 rows ×
> 3584. Single source: `footprint()` in `distributed_rms_norm_program_descriptor.py` mirrors this table.

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

Mirrors `footprint()` in `distributed_rms_norm_program_descriptor.py` term for term (verifier: re-derived from the
code; the Phase 0 formula had drifted — `cb_chip_partial` / `cb_partial_col` are two deep, the record scratch is
`(G + 1)·q`, and the combine CBs exist only when `G_w > 1`).

```
F = R·Wp·T_in                                         [interleaved input only; x_depth = 1]   scales with R, S
  + 2048·(1 + masked)                                 cb_reduce_scaler                         constant
  + T_i·( q_w + NW·q_w + 2·q + q )                    sumsq, inv, chip_partial, rms_root
  + [G_w > 1] T_i·( 2·q_w + max(1, max_owned)·G_w·q ) partial_col, gather                      scales with G_w
  + 4096·G·q                                          cb_stats (Float32 always)                scales with G
  + (G + 1)·q·128                                     cb_record_scratch                        scales with G
  + has_gamma·( Wp·T_g + (R | q_w)·WS·T_i )           gamma_tiles, x_gamma (block | per-wave)  scales with S, R
  + masked·2·q_w·T_i                                  sumsq_last, sumsq_acc [sharded masked]
  + STREAM_DEPTH·C·T_out                              cb_output [interleaved output only]      scales with C
```

Focus case (640×1792 bf16, fp32 DEST, bf16 TILE γ; R 3, Wp 4, C 4, NW 3, q_w 1, q 1, G_w 14, max_owned 1, G 4):
x 24,576 + scaler 2,048 + sumsq 4,096 + inv 12,288 + chip 8,192 + root 4,096 + partial_col 8,192 + gather 57,344
+ stats 16,384 + records 640 + γ 8,192 + x·γ 49,152 + output 16,384 = **211,584 B (≈ 207 KiB)** of ~1.4 MiB
(`footprint()` returns the same number). Decode 32×1792: 154,240 B. 2048×1280 (R 5, S 5, G_w 8, NW 5): 281,216 B.
The two largest terms are `cb_gather` (`G_w`-scaled; lamp L5 would cut it to 128 B rows) and `cb_x_gamma`
(`R·WS`-scaled; lamp L4 / the per-wave fallback).

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


## Refinement 1 deltas (implementer) — `cb_table()` is now the single source of both the CB descriptors and `footprint()`

| CB | Phase 0 | With the L1 ladder (Refinement 1) |
|----|---------|-----------------------------------|
| `cb_x_tiles` (interleaved) | `R·Wp` | `x_depth·R_pass·WS`, `R_pass = min(rows_max, PW·q_w)` (passes; `x_depth` 2 when it fits) |
| `cb_inv_rms` | `num_waves·q_w` | `PW·q_w` (one pass; slots reused after the `SEM_PASS` row-group barrier) |
| `cb_gather` | `max_owned·G_w·q` | `regions·G_w·q`, `regions = max_owned_waves(PW, owners, G_w)` (owned-wave ordinal mod regions) |
| `cb_partial_col` | `2·q_w` | `combine_depth·q_w` (lean rung: 1) |
| `cb_sumsq_partial` (+ `_last`, `_acc` masked) | `q_w` | `SQ = min(sq_rows, q_w)` (lean rung: 1 tile-row per square→collapse round) |
| `cb_gamma_tiles` | `Wp` | `Wp`, or `STREAM_DEPTH·C` when γ streams |
| `cb_x_gamma` | `R·WS` / `q_w·WS` | block `R_pass·WS` / wave `q_w·WS` / chunk `q_w·C` |
| `cb_output` | `STREAM_DEPTH·C` (interleaved out) | same, keyed on `out_resident` (input's own shard placement); resharded outputs stream |
| all CBs (last resort) | static region | carved out of one scratch L1 tensor in the largest free hole (dynamic CB addresses) |

`C` follows the halved chunk cap. Phase 0 plans are option 0 of the ladder: their CB sizes are unchanged.

**Budget**: gap below the lowest allocated L1 block (CBs grow up from the bank base), re-measured after the plan's
exchange scratch exists. **Port core**: landing `num_waves·G·seg_stride` (floor 4 KiB), one per (mesh, ports, G),
grown / shrunk on demand.

**Data movement** (fallback rungs only; Phase 0 unchanged): streamed γ reads each slice chunk's row-0 prefix once per
(wave with rows) instead of once per kernel — γ DRAM bytes × waves (≤ 12) on the cores that stream; x, out: one DRAM
crossing each in every rung (passes keep x resident per pass block; no re-read).

## Refinement 1a deltas (implementer) — the x re-read rung (interleaved only, last resort)

| CB | Before | x re-read rung (`plan.reread`) | Shares with / why not |
|----|--------|--------------------------------|-----------------------|
| `cb_x_tiles` (interleaved) | `x_depth·R_pass·WS` (x resident for the pass) | `x_depth·SQ·WS` (`SQ = 1`, `x_depth` 2 then 1): x streamed per square→collapse round, popped after the collapse | not `cb_x_reread`: Phase A streams it while the writer already prefetches Phase B items (concurrent) |
| `cb_x_reread` (new, index 8) | — | `STREAM_DEPTH·C` pages, input dtype; writer → compute, one (wave, chunk, tile-row) item of `C` tiles | not `cb_x_tiles`: different producer (writer vs reader) — single-producer rule; not `cb_output`: concurrent (consumed into it) |
| `cb_x_gamma` | chunk `q_w·C` | `C` (x·γ of one tile-row chunk) | as before |
| `cb_partial_col` | `combine_depth·q_w` | kept at `COMBINE_DEPTH = 2`: the streamed reader pushes wave w+1 before servicing wave w — a single-wave `cb_partial_col` would deadlock | — |

Data-movement budget on this rung: **x crosses DRAM twice** (Phase A square, Phase B scale), out once, γ as before.
Taken only when no other rung fits (target cell: 131072 × 3584 per device, bf16, RM γ, 16-bit DEST: the landing
scratch of ~540 KB per bank leaves ~918 KB, and one wave of resident x needs ≥ 720 KB on top of the other CBs).

## Refinement 2 — no CB change

The LOCAL perf levers (reader prologue position, `READ_LOOKAHEAD_WAVES = 0`, `COMBINE_LAG = 0`, `NOC_SWAP_COLUMNS = 3`)
change only the order and the NoC of existing transfers: no CB is added, resized or removed, `footprint()` is unchanged,
and the data-movement budget above is unchanged (x and out cross DRAM once, γ once per row-group; the swapped columns
move the same bytes on the other NoC).
