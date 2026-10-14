# L1 Ledger: distributed_rms_norm (Phase 0, regime R1)

Schema and audits are defined in `.claude/references/l1-footprint-discipline.md`. Block axes come from `op_design.md`.

Block axes:
- `rows` (extent `block_rows`)
- `w_local` (extent `core_w_tiles`)
- `sources` (extent `n_src_tiles`, stage data)

`group` (G) is a device axis, so no worker CB spans it. Its only footprint is through `n_src_tiles` and the port scratch.

Phase 0 runs with `fp32_dest_acc_en = True`, so every DEST-produced intermediate is Float32. R4: with `fp32_dest_acc_en = False` only cb_xg narrows (Float16_b); the stats CBs (cb_sq_acc, cb_partial, cb_stats_combine, cb_rstd, cb_lc, cb_lc_out) and the wire records stay Float32 — the spec requires the exchanged partials and their sum in fp32 (page-format audit mechanism reason). bfloat8_b x / output tiles are 1088 B pages (x CB and cb_output_tiles rows scale with `ttnn.tile_size`).

## Worker cores

| CB | Capacity (pages) | Live set | Axis accounting | Page format | Producer | Consumer | Lifetime | Shares with / why not |
|----|------------------|----------|-----------------|-------------|----------|----------|----------|-----------------------|
| cb_x_resident | `block_rows · core_w_tiles` | `block_rows · core_w_tiles`. The whole block is resident from load until it is popped by prescale (gamma) or normalize (no gamma). | `{rows: spans → block_rows, w_local: spans → core_w_tiles, sources: streams → 0}` | x dtype (Float32 / Float16_b) | reader | compute | load → prescale / normalize | **No share.** Its residency across the exchange is the whole reason x crosses DRAM once. It cannot alias cb_xg: prescale reads x tile i while packing xg tile i, both live, and an in-place transform would make compute a second producer of a reader-produced CB. |
| cb_gamma_tiles | `core_w_tiles` (allocated only when `HAS_GAMMA`) | `core_w_tiles`, held for the whole prescale (or, on the fused path, the whole normalize). Row 0 valid on the prescale path (Row bcast); all 32 rows = row 0 on the fused path (DEST-reuse Mul has no broadcast). | `{rows: streams → reused for every row, w_local: spans → core_w_tiles, sources: streams → 0}` | gamma dtype (Float32 / Float16_b). This is input data, not DEST output, so it keeps its own format. | reader | compute | whole call | **No share.** It is reuse-shared across rows (read once per call). Every lifetime it could share overlaps it. |
| cb_scaler | 1 (constant) | 1 | `{rows: streams, w_local: streams, sources: streams}` (constant) | Float16_b (scaler convention; 1.0 is exact) | reader | compute | whole call | **No share.** It is read by two reduces (sumsq and combine), so it must persist across both. |
| cb_sq_acc | `block_rows` | `block_rows`. One DEST-accumulated Σ_w x² tile per tile row. | `{rows: spans → block_rows, w_local: streams → collapsed in DEST, sources: streams → 0}` | Float32 | compute | compute | sumsq | **Disjoint lifetime with cb_rstd** (same size and format), but kept separate on purpose. Deferred regime R2 (multi-round) runs round b+1's sumsq while round b's rstd is live, and aliasing would foreclose that. It totals ≤ `block_rows · 4 KiB`. Folding into DEST is already done: W is collapsed in DEST, and only one tile per row is packed. |
| cb_partial | `block_rows` | `block_rows` | `{rows: spans → block_rows, w_local: streams → 0, sources: streams → 0}` | Float32 (stats stay fp32 on the wire) | compute | writer | sumsq → publish | **No share with cb_sq_acc:** the reduce reads cb_sq_acc while it packs cb_partial (concurrent), and the consumers differ (compute vs writer). **No share with cb_rstd:** different consumer kernel, and R2 overlap. |
| cb_xg | `block_rows · core_w_tiles` (allocated only when `GAMMA_PRESCALE` = `HAS_GAMMA ∧ F_worker(prescale) ≤ L1_BUDGET`) | `block_rows · core_w_tiles` from prescale until normalize | `{rows: spans → block_rows, w_local: spans → core_w_tiles, sources: streams → 0}` | Float32 when `fp32_dest_acc_en`, else Float16_b (R4: follows the DEST width; `xg_dtype_of`) | compute | compute | prescale → normalize | It exists only because of the **explicit stall-shadow decision**: x·gamma is computed during the exchange wait. Its lifetime overlaps cb_x_resident (prescale) and cb_output_tiles (normalize). The perf lamp "gamma prescale" measures the version without it. |
| cb_stats_combine | `n_src_tiles · block_rows` | `n_src_tiles · block_rows`. The reduce waits for the whole block (BulkWait). | `{rows: spans → block_rows, w_local: streams → 0, sources: spans → n_src_tiles}` | Float32 | reader | compute | gather → finalize | **No share.** It could alias cb_sq_acc only when `n_src_tiles = 1`, and that also forecloses R2. It totals ≤ 4 KiB · `n_src_tiles · block_rows`. |
| cb_rstd | `block_rows` | `block_rows`. The Col-mapped Upfront window is held across normalize. | `{rows: spans → block_rows, w_local: streams → broadcast along it, sources: streams → 0}` | Float32 | compute | compute | finalize → normalize | See cb_sq_acc (R2 overlap). |
| cb_output_tiles | `OUT_DEPTH · out_batch_tiles` (`OUT_DEPTH = 2`; `out_batch_tiles = out_batch_tiles_knob(core_w_tiles)`, Phase 0 `= core_w_tiles`, must divide it; passed to the writer as an RT arg) | `out_batch_tiles` per stage | `{rows: streams → 1 tile row at a time, w_local: spans → core_w_tiles per batch, sources: streams → 0}` | x dtype | compute | writer | normalize → store | Capacity exceeds the live set by 2×. That is **double buffering**: the pack of tile-row j+1 overlaps the write of j (catalog `double_buffer`). It cannot pack into the destination because the destination is interleaved DRAM. It cannot alias cb_x_resident: compute would become a second producer. |

### Symbol table (worker)

| Symbol | Bound | Predicate establishing it |
|--------|-------|---------------------------|
| `block_rows` | ≤ `core_rows = ceil(tensor_row_tiles / S_r)` | `_plan_blocks()`; Phase 0 sets `block_rows = core_rows` under `P_resident` |
| `core_w_tiles` | ≤ `ceil(tensor_w_tiles / S_w)`; ≥ `min(tensor_w_tiles, MIN_CORE_W_TILES = 8)` | `_choose_split()` |
| `n_src_tiles` | `ceil(G · S_w / 32)`; G ≤ 8 (Galaxy axis 0), `S_w ≤ ceil(tensor_w_tiles/8)` | `_choose_split()` plus mesh shape |
| `out_batch_tiles` | divides `core_w_tiles`; Phase 0 `= core_w_tiles` | `out_batch_tiles_knob()` (single source; host asserts divisibility per chunk group) |
| `GAMMA_PRESCALE` | 0 / 1 | `HAS_GAMMA ∧ F_worker(GAMMA_PRESCALE = 1) ≤ L1_BUDGET` (host, per call) |
| `T_x` | x tile bytes: 4096 (fp32) / 2048 (bf16) / 1088 (bfp8, TARGET) | dtype |
| `T_g` | gamma tile bytes: 4096 / 2048 | gamma dtype |
| `T_f` | 4096 (Float32 tile) | — |
| `HAS_GAMMA` | 0 / 1 | gamma given |
| `L1_BUDGET` | 1 MiB per worker for CBs (host constant `WORKER_L1_BUDGET`) | — |

### Footprint (per worker, bytes)

```
F_worker = block_rows · core_w_tiles · (T_x + GAMMA_PRESCALE · T_f)   # cb_x_resident + cb_xg       -> scales with block_rows x core_w_tiles
         + HAS_GAMMA · core_w_tiles · T_g                              # cb_gamma_tiles              -> scales with core_w_tiles
         + 2048                                                       # cb_scaler                   -> constant
         + block_rows · T_f · (3 + n_src_tiles)                        # sq_acc, partial, rstd, combine -> scales with block_rows (x n_src_tiles)
         + OUT_DEPTH · out_batch_tiles · T_x                           # cb_output_tiles             -> scales with out_batch_tiles
GAMMA_PRESCALE := HAS_GAMMA ∧ F_worker(GAMMA_PRESCALE = 1) <= L1_BUDGET     # stall-shadow prescale only when it fits
P_resident     := F_worker(block_rows = core_rows, GAMMA_PRESCALE) <= L1_BUDGET   # else RuntimeError (R2 deferred)
```

Worst cases on the measured 11×10 Blackhole grid (N = 99 workers, G = 2):

| Case | Split | F (prescale) | F (fused) | Path |
|------|-------|--------------|-----------|------|
| 2048×2048 fp32, fp32 gamma (largest INPUTS) | S_r 33, S_w 3, cw 22, rows 2 | 665 600 B | 485 376 B | prescale |
| (2, 2048, 4096) bf16, bf16 gamma (translated `8192-2048-2`) | S_r 33, S_w 3, cw 43, rows 4 | 1 388 544 B | 684 032 B | **fused** (prescale > 1 MiB) |

The fused path was added by the verifier: before it, the second case raised the P_resident RuntimeError.

## Port cores

No CBs. Each port core holds:
- its shard of the per-call **exchange scratch** tensor: `G · slot_bytes`, with `slot_bytes = 128 · max_l recs_link[l]`;
- the fabric packet-header pool (firmware-owned, 12 headers per DM RISC);
- global semaphores `arrival` and `ready`, 4 B each.

| Buffer | Capacity | Live set | Axis accounting | Format | Producer | Consumer | Lifetime | Shares with / why not |
|--------|----------|----------|-----------------|--------|----------|----------|----------|-----------------------|
| exchange scratch (L1 tensor shard) | `G · slot_bytes` | all G slots until the stats-ready mcast. Workers pull after it, so the slots must hold for the whole call. | `{rows: spans → link's rows, w_local: spans → S_w records per row, sources: spans → G slots}` | raw fp32 records (128 B) | local workers (slot p, NoC write) plus the upstream port (Fabric) | peer port (relay source) plus workers (NoC read) | one call | **No share.** It is call-scoped by design: no credits and no round trips. It is a per-call tensor (freed after the call), not a persistent allocation. |

`P_scratch := G · slot_bytes ≤ 1 MiB`. The worst INPUTS case is 4096×1024 on Galaxy axis 0 (G = 8, L = 1, `S_w = 4`, 128 rows): 8 · 512 · 128 B = 512 KiB.

## Data-movement budget

Per device, per call, for split R1. B = x bytes = `batch·ceil(H/32)·ceil(W_local/32)·T_x`.

| Tensor | DRAM crossings | Why that many | Cross-core traffic added |
|--------|----------------|---------------|--------------------------|
| x | 1 (read) | x is resident in cb_x_resident across the exchange (`P_resident`). No second pass. | — |
| output | 1 (write) | written once from cb_output_tiles | — |
| gamma | S_r (each worker reads its W segment once per call) | Stepping stone. The broadcast (R8) would make it 1. | — |
| stats records | 0 | They live in L1 only. The stats never round-trip through DRAM, unlike today's composite. | NoC: `Ht·S_w·128 B` worker→port (×2 ports per link at interior devices) plus `S_r·S_w·G·S_w·128 B·block_rows/S_r` port→worker pulls. Fabric: per link direction, ≤ (G−1) entries of `recs_link·128 B`. |

Totals at the perf-focus shape (640×1792 bf16, G = 2, L = 2, 117 workers, S_w = 5):

| Tier | Bytes |
|------|-------|
| DRAM | 2.29 MB read + 2.29 MB written (minimum) + gamma 20 × 3.6 KB = 72 KB |
| NoC | 12.8 KB of records to the ports, 100 workers × 10 records × 128 B = 128 KB of pulls |
| Fabric | 6.4 KB per link direction, one packet each |

> Cheapest-traffic split considered: rows × W-chunks with x resident and flat records (R1). The delta is 0: DRAM is at the x-once and out-once minimum. Implemented: R1. The implemented split **is** the cheapest-traffic split. The only above-minimum term is gamma (S_r reads of a W_local stick); its broadcast is deferred as R8 because it is ≤ 3 % of DRAM bytes.

## Implementation deltas (ttnn-implementer, Phase 0)

| Item | Planned | Implemented | Why |
|------|---------|-------------|-----|
| CB capacity of `cb_x_resident`, `cb_output_tiles`, `cb_gamma_tiles`, `cb_xg` | nominal `core_w_tiles` on every worker | one CBDescriptor per **chunk group** (full chunks / the ragged last chunk, `tensor_w_tiles − (S_w−1)·core_w_tiles`), capacity derived from that group's chunk width | CB pointers wrap only on an exact hit of the ring end, so every batch (one tile row of the chunk) must divide the capacity. With nominal sizing the ragged chunk's 3rd output batch ran past the CB (golden 4096×1024 failure). Footprint per core is ≤ the nominal one. |
| `min_core_w_tiles` | 4 | 8 | Record pulls per device are `Ht·G·S_w²·128 B` (quadratic in S_w). At 640×1792 on the 11×10 grid, 4 gives S_w = 14 (1 MB of pulls through 4 port cores); 8 gives S_w = 4. Measured (perf lamp "grid synchronization"): 640×1792 28.7 → 21.7 µs, 32×4096 12.3 → 10.1 µs, decode / 2048×1280 unchanged. |
| Record layout in the scratch slot | record = 128 B contiguous | per (row group, tile row) a block of `S_w·128 B`: `[half 0 of w = 0..S_w−1][half 1 of w = 0..S_w−1]` | Consecutive sources s = q·S_w + w are consecutive combine-tile face rows, so the gather is 2 NoC reads per (q, tile row, 16-row face run) instead of 2 per record (perf lamp "record pull mechanism"; 640×1792 gather 23 → 1.1 µs). Slot sizes and record counts unchanged. |
| cb_stats_combine zero fill | whole reserved block every round | only the last source tile of each tile row, and only when `G·S_w mod 32 ≠ 0` | All other rows are overwritten by records. |
| Links | `L` | `L_eff = min(L, S_r)` | A link with no row group would carry no records (and no arrival increment). |
| Worker rectangle | `(0,1)..(grid_x−1, grid_y−1)` | `(0,1)..(grid_x−1, y_last)` with y_last the last row holding a used worker; idle cores of that rectangle run zero-row kernels | The stats-ready multicast must only land on cores of this program. |

Data-movement budget at 640×1792 bf16 (G = 2, L = 2, 11×10 grid → S_w = 4, S_r = 20, 80 workers): DRAM x 2.29 MB read once, out 2.29 MB written once, gamma 20 × 3.6 KB; NoC record pulls `20·2·16·128 B` = 80 KB; Fabric one packet per link direction.

## Verifier deltas (Phase 0 verification)

| Item | Before | After | Why |
|------|--------|-------|-----|
| `cb_xg` | always allocated with gamma | allocated only when `GAMMA_PRESCALE` (prescale footprint fits `L1_BUDGET`); otherwise gamma is fused into normalize as a DEST-reuse Mul (`cb_gamma_tiles` row-replicated by the reader) | (2, 2048, 4096) bf16 + gamma needed 1.39 MB with the fp32 `cb_xg` block. The fused path needs 0.68 MB. No more RuntimeError on any golden / translated shape. The perf-focus shapes keep the prescale path (unchanged). |
| `out_batch_tiles` | host knob for the CB size only; the writer hard-coded `cw` as its batch | writer RT arg `out_batch`, walks each tile row in `out_batch` batches; host asserts it divides the chunk | DRY: one source for the batch. Verified live by running the acceptance suite at `out_batch = 1`. |
| `RECORD_BYTES` | host constant + literal `128` in reader and writer | host constant passed as a CT arg | DRY. |

## Refinement 1 deltas (native sharded placement, regimes R9 / R10 + R5 pre-combine)

Sharded calls (HEIGHT / WIDTH, L1) run the same CB set on the **shard grid**. The per-call knobs are chosen by
`_choose_knobs` against the CB room the allocator actually leaves: `_l1_free_below_buffers` reads the L1 block table
(`ttnn._ttnn.device.GetMemoryView`), so it counts the shards, the scratch, the inbox, and any other live L1 tensor.
The interleaved path is unchanged: `stat_rows = block_rows`, `gamma_chunk = core_w_tiles`, and the budget stays
`WORKER_L1_BUDGET`.

| CB / buffer | Sharded capacity | Producer | Consumer | Shares with / why not |
|-------------|------------------|----------|----------|-----------------------|
| cb_x_resident | **the input shard** (`cb_descriptor_from_sharded_tensor`): 0 CB bytes | reader (publishes `core_rows·cw_phys` resident tiles, no NoC read) | compute (Strided, row stride `cw_phys`, never popped) | **Is** the shard: zero-copy. No TensorAccessor read of a core's own shard exists. |
| cb_output_tiles | **the output shard** when its shard spec equals the input's (0 CB bytes); otherwise `OUT_DEPTH·out_batch` streamed to the writer (TensorAccessor store, any output memory config) | compute (Strided pack in place; reserve once, push once) | — (native) / writer | Pack into the destination: the writer's store disappears on the native path. |
| cb_sq_acc, cb_partial, cb_rstd | `stat_rows` | as Phase 0 | as Phase 0 | **Streamed in `stat_rows` batches.** A WIDTH shard holds every tile row (Ht up to 128), so a full block of per-row stats would be 0.5 MB per CB. `stat_rows` is the largest power-of-two fraction of the block that fits. |
| cb_stats_combine | `n_src_tiles·stat_rows` | reader | compute | as above; `n_src_tiles = ceil(G·rec_w/32)`, with `rec_w = 1` under R5 |
| cb_gamma_tiles | `gamma_chunk` (== cw: held for the call) or `2·gamma_chunk` (a double-buffered stream of `gamma_chunk`-page chunks; the ragged last chunk still occupies `gamma_chunk` pages so the ring wraps on a chunk boundary) | reader | compute | **Streamed** when a HEIGHT shard's full W_local of gamma does not fit next to the in/out shards: 32×8192 bf16 + fp32 gamma would need 1 MB of gamma tiles beside 2×512 KB of shards. Only the fused normalize streams (prescale needs `cb_xg`, which is larger than gamma and so never fits when gamma does not). |
| cb_xg | `block_rows·cw` (prescale only) | compute | compute | unchanged; dropped (fused path) when it does not fit |
| cb_lc (new, combiners) | `n_lc_tiles·lc_per`, `n_lc_tiles = ceil(S_w/32)`, `lc_per = ceil(core_rows/S_w)` | reader (local gather from my inbox) | compute (REDUCE_COL) | R5 only. **No share**: it is live while cb_partial streams. |
| cb_lc_out (new, combiners) | `lc_per` | compute | writer (publishes to the ports) | R5 only. Row 0 of each tile is the combined record (REDUCE_COL, no transpose). |
| R5 inbox (L1 sharded tensor over the worker set) | `lc_per·S_w·128 B` per core | every W chunk's writer (NoC write of its record into the owning combiner) | combiner's reader | Same address on every core (lockstep allocation). It is a buffer, not a CB, because remote writers address it. |

**Why R5 (device-local pre-combine) is built now.** A WIDTH shard pins `S_w` to the shard core count (up to 86 on
the 11×10 grid). With flat records, the exchange scratch is `G·S_w·Ht·128 B`: 1 MB for 4096×1024. It is allocated
in lockstep on every core, and each worker pulls `G·S_w` records per tile row (quadratic in `S_w`). With R5 the
scratch is `G·Ht·128 B` (32 KB) and each worker pulls G records per row. The design's deferral reason ("≤ 3 packets
per link") assumed `S_w ≤ 4` from `_choose_split`; it does not hold for a pinned WIDTH `S_w`. R5 is path-gated to
sharded `S_w > 1`. The interleaved split keeps flat records.

**Footprint (sharded).** `F = HAS_GAMMA·gamma_chunk·(1|2)·T_g + GAMMA_PRESCALE·block_rows·cw·T_f + 2048 +
stat_rows·T_f·(3 + n_src_tiles) + R5·lc_per·T_f·(n_lc_tiles + 1) + (¬native)·OUT_DEPTH·out_batch·T_x`. x and a
native output cost 0.

**Ports (sharded).** These are the first `2·L_eff` grid cores the shard grid leaves free. The stats-ready
increment is multicast per rectangle of the shard grid, so it only reaches cores of this program.

**Data-movement budget (sharded).**

| Tensor | DRAM crossings | Why |
|--------|----------------|-----|
| x | 0 | It is read in place from the shard. |
| output | 0 | Compute packs into the shard (native). |
| gamma | `S_r` reads per call (HEIGHT: one per shard; WIDTH: one per W chunk), or `S_r·stat batches` when fused + streamed | — |
| records | 0 | They stay in L1. WIDTH NoC traffic: `Ht·S_w·128 B` worker→combiner, `Ht·128 B` combiner→port, `S_w·Ht·G·128 B` port→worker pulls. |

## Refinement 3 — R2 multi-round exchange (interleaved): status **built**

`block_rows < core_rows` is now live on the interleaved path. The plan picks `rounds` and derives `block_rows = ceil(core_rows_max / rounds)`:
- **Split.** `_rounds_split` uses `EXCHANGE_ROUNDS = 2`. It caps `S_r` at `ceil(Ht / 2)` and relaxes the W-chunk floor so `S_w` can double. It is taken only when the tiles per worker do not exceed the one-round split.
  - 640×1792: `S_r` 20→10, `S_w` 4→8, still 14 tiles per worker.
- **L1 fit (`_fit_rounds`).** If the footprint fails, `rounds` grows until it fits. The prescale path is preferred over the fused one.

The worker kernels run a lookahead schedule with `ROUND_LOOKAHEAD = 1`: `front(b+1)` (load / sumsq / publish / prescale) runs before `back(b)` (gather / finalize / normalize / store).

**CBs whose capacity changes with `block_rows` / rounds:**

| CB | Capacity | Why |
|----|----------|-----|
| cb_x_resident | prescale: `(1 + ROUND_LOOKAHEAD) · block_rows · cw` (`x_blocks()`). No prescale (no gamma, or fused gamma): `num_blocks · block_rows · cw` (the whole assignment). | With prescale, x is popped every round, so the CB is a plain FIFO of the blocks in flight. Without prescale, round b's x stays at the CB front until its normalize, and round b+1's sumsq reads past it at a tile offset. That offset is linear from the read pointer and not wrap-safe. A `(1+LA)`-block ring wraps from round 3 on: fp32 4096×1024 at 4 rounds failed (rel-RMS 0.008 / 0.019) before this sizing. So the CB never wraps within a call. |
| cb_xg | `(1 + ROUND_LOOKAHEAD) · block_rows · cw` | Prescale of round b+1 runs before round b's normalize pops round b. |
| cb_sq_acc, cb_partial, cb_rstd, cb_stats_combine | unchanged formula (`stat_rows = block_rows`) | These CBs shrink with `block_rows`. Their earlier "kept separate for R2" notes now hold: round b+1's sumsq runs while round b's rstd is live. |
| cb_output_tiles, cb_gamma_tiles | unchanged | — |

**Footprint (interleaved):**
```
F = X_BLOCKS_IN_FLIGHT · x_blocks · block_rows·cw·T_x + PRESCALE · (1+LA)·block_rows·cw·T_f + (unchanged rest)
```
**P_resident:** the RuntimeError now fires only if no number of rounds fits.
- With prescale, the x and xg blocks shrink with more rounds. (2, 2048, 4096) bf16 + gamma used to need the fused fallback. It now runs 4 rounds of one-row blocks on the prescale path: ~0.8 MB, 346 → ~200 µs.
- Without prescale, x is whole-assignment resident (same as R1), so rounds do not lift that path.

**Semaphores (all cores).**
- `SEM_LOCAL_ARRIVALS + b % (1+LA)`: rotating local-arrival counters on the ports. A single counter let a fast worker's round b+1 increment complete a slow worker's round b, so a port sent round b before all of its records had landed. This was observed: round-0 rows wrong on 4096×1024 / 2048×1280.
- `SEM_LOCAL_COMBINE`.
- `SEM_STATS_READY_BASE + 2l + d`: one stats-ready counter per port. A worker waits on both ports of its link for ≥ round + 1.
- Total `3 + LA + 2L ≤ 16`, checked on the host.

**Exchange scratch:** the slot is round-major.
- `slot_bytes = 128 · num_blocks · max_l round_recs[l]`.
- `round_recs[l] = rec_w · Σ_{g∈l} min(block_rows, core_rows[g])`.
- One round reproduces the dense R1 layout. A ragged last round leaves an unread gap.

**Data-movement budget:** unchanged. x crosses DRAM once (never re-read) and the output once. Stats bytes are the same per tile row. Fabric carries one entry per round per source, each `round_recs·128 B`.
