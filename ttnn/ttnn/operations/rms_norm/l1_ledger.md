# L1 Ledger: rms_norm (regime R1 row-resident + R6 HEIGHT_SHARDED in place + R3 W-split)

Schema and audits: `.claude/references/l1-footprint-discipline.md`. Block axes (from
`op_design.md` Blocking Model): **rows** (extent `r = block_rows`), **w** (whole row per core,
`Wc = core_w_tiles`, stored at stride `Wp = row_stride_tiles = K·c`), **w_chunk** (streaming
extent `c = w_chunk_tiles`, `K = num_w_chunks`).

**Refinement 2 — page formats follow the DEST width** (`_Geometry.interm_fmt`, single source):
`fp32_dest_acc_en=True` → `cb_rstd` / `cb_normed` are `Float32`; `fp32_dest_acc_en=False` → they are
`Float16_b` (16-bit DEST already rounded the value, a wider page stores nothing; halves `cb_normed`, so
the block selector can pick a larger `r`/`c`). **`cb_sumsq` stays `Float32` in both modes**
(`_Geometry.sumsq_fmt`) — mechanism reason: it is a *packer L1 accumulator*, the Σx² adds happen in L1
at the page format, not in DEST, so Float32 pages carry real bits even with 16-bit DEST (measured: bf16
pages bias rstd, relative RMS 0.017 → 0.006 at W=7168). Cost ≤ 2 tiles/row. `cb_reduce_mask` follows
`cb_sumsq`'s format (srcB constraint below). bfloat8_b input / gamma / output: `cb_x`, `cb_gamma`,
`cb_out` carry the tensor's own format (Bfp8_b pages, `ttnn.tile_size`).

(Phase 0 text:) `fp32_dest_acc_en = True` in Phase 0 → every compute intermediate page is `Float32` (audit 2,
"under" direction: a 16-bit intermediate would truncate fp32 DEST at the phase boundary).
Buffers holding *raw input data* (`cb_x`, `cb_x_rm`), constants (`cb_reduce_mask`) and *final
output* (`cb_out`, `cb_out_rm`) carry the tensor's own format — they never carry a value wider
than the tensor that produced/consumes them.

| CB | Capacity (pages) | Live set | Axis accounting | Page format | Producer | Consumer | Lifetime | Shares with / why not |
|----|------------------|----------|-----------------|-------------|----------|----------|----------|-----------------------|
| `cb_x_rm` (RM only) | `2·c` | `c` (one row-chunk of 32 sticks) | `{rows: streams → 1 tile-row, w: streams → chunk, w_chunk: spans → c}` | input dtype | reader | compute | load phase of each block | capacity 2× live = double buffering (reader fetches the next row-chunk while compute tilizes). Cannot share with `cb_out_rm`: both live concurrently across blocks (reader prefetch vs writer drain), different producers |
| `cb_x` | `x_block_depth·r·Wp` (`x_block_depth` = 1 shipped; R3 measured 2 slower on prefill, see data-movement note) | `rows_this_block·Wp` (whole block resident from load to release) | `{rows: spans → r, w: spans → Wp, w_chunk: spans (contained in w)}` | input dtype | reader (TILE) / compute tilize (RM) | compute | load → release, per block | the residency buffer; no other buffer has a disjoint lifetime with it (live through every compute phase of the block). `Wp − Wc < K` pad pages per row are the nominal-chunk mechanism reason (uniform CT chunk width, ring quantum `c`) |
| `cb_reduce_mask` (`w_partial_elems ≠ 0` only) | `1` | `1` | `{rows: streams (constant), w: streams (constant), w_chunk: streams (constant)}` | **Float32** (0/1 exact; implementer deviation — the reduce helper's masked fold `fold_partial_last` unpacks the mask as srcB without reconfiguring srcB away from `cb_sumsq`'s Float32, so the mask must carry `cb_sumsq`'s format) | reader | compute | whole kernel | constant tile; aliasing a constant into a ring with live data would destroy it |
| `cb_sumsq` | `r·n_sq` (own allocation) | `rows_this_block·n_sq` (`n_sq ≤ 2`) | `{rows: spans → r, w: streams (collapsed to ≤ 2 tiles by L1 accumulation), w_chunk: streams}` | Float32 | compute | compute | sumsq → rstd, per block | **not aliased with `cb_normed`** (implementer deviation): the rstd reduce reads its `r·n_sq` window by index from the read pointer, and indexed windows must not straddle the ring's `fifo_limit`; with the alias capacity `round_up(r·max(c,n_sq), lcm(c,n_sq))` the sumsq window drifts across blocks and eventually straddles it. As its own ring of exactly `r·n_sq` pages every full block returns both pointers to base. Cost: `r·n_sq·4 KiB` (≤ 8 KiB at r = 1) |
| `cb_rstd` | `r` | `rows_this_block` | `{rows: spans → r, w: streams (one col-0 tile per row), w_chunk: streams}` | Float32 | compute | compute | rstd → release, per block | live through every normalize chunk, concurrent with `cb_normed` and `cb_x` → no alias |
| `cb_gamma` (gamma only) | `2·c` | `c` | `{rows: streams (broadcast), w: streams → chunk, w_chunk: spans → c}` | gamma dtype | reader | compute | per chunk of the normalize phase | 2× live = double buffering (prefetch during the previous chunk / during sumsq). Concurrent with `cb_x_rm` across blocks (reader prefetches next x while current block scales) → no alias |
| `cb_normed` (gamma or RM) | `r·c` | `rows_this_block·c` | `{rows: spans → r, w: streams → chunk, w_chunk: spans → c}` | Float32 | compute | compute | normalize → scale/untilize, per chunk | not aliased (see `cb_sumsq`). Full chunk-block, not a window: normalize and scale/untilize are sequential helpers on one thread; each full chunk returns the ring to base |
| `cb_out` (TILE only) | `2·c` | `c` (one row-chunk) | `{rows: streams → 1 tile-row, w: streams → chunk, w_chunk: spans → c}` | output dtype | compute | writer | per row-chunk | 2× live = double buffering compute vs writer. Pack-into-destination: with no gamma, normalize packs straight here (no `cb_normed` at all) |
| `cb_out_rm` (RM only) | `2·c` | `c` | `{rows: streams → 1 tile-row, w: streams → chunk, w_chunk: spans → c}` | output dtype | compute | writer | per row-chunk | 2× live = double buffering. Cannot share with `cb_x_rm` (concurrent, different producers/consumers) |

### R6 HEIGHT_SHARDED (Refinement 1) — CB placement deltas

Placement `RESIDENT` (TILE x; RM x when `W % 32 == 0` and `shard_h % 32 == 0` and a whole-row chunk
fits): the input / output **shards are the CBs** (`ttnn.cb_descriptor_from_sharded_tensor`, page
size overridden to the tile size so CB credit counts tiles) — zero-copy, no DRAM, no NoC.

| CB | R6 RESIDENT | R6 LOCAL_STICKS (RM, shard not tilize-consumable in place) |
|----|-------------|----------------------------|
| `cb_x` (TILE) | **= input shard** (`shard_h/32 · Wc` pages, not allocated) | n/a |
| `cb_x` (RM) | `r·Wp`, as R1 (tilize output) | `r·Wp`, as R1 |
| `cb_x_rm` (RM) | **= input shard** (whole tile-rows of `Wc` pages at stick stride `Wc·32`) | `2·c` streaming window, filled by self-aimed L1→L1 restride reads of the resident sticks |
| `cb_out` (TILE) | **= output shard** (compute packs in place; writer only drains credit) | n/a |
| `cb_out_rm` (RM) | **= output shard** (untilize writes whole tile-rows in place) | `2·c` window; writer copies the valid sticks into the output shard (local L1 write) |
| others | as R1 | as R1 |

Why LOCAL_STICKS exists: tilize/untilize consume / produce whole 32-stick tile-rows at stride
`c·32` elements. An RM height shard has stride `align16(W·elem)` (≠ `Wc·32·elem` when `W % 32 ≠ 0`)
and an arbitrary stick count (`auto_shard_config` gives e.g. 18 sticks/core for 2048 rows on 120
cores) — untilizing in place would overrun the shard. Those shards are restrided locally (L1 only).

R6 block selection (`_select_resident_blocks`): `K = 1` with the largest `r` that fits (the whole
shard when it fits). A resident row cannot be re-chunked at a stride other than `Wc`, and with
`K > 1` the chunk-major pack order equals the shard's row-major order only for `r = 1`, so the wide
fallback is `r = 1` with `c | Wc`. RM resident needs `K = 1` (falls back to LOCAL_STICKS otherwise).
`F` drops the `cb_x` (TILE) / `cb_x_rm` + `cb_out[_rm]` terms that are the shards; `B` is already
net of both resident shards (allocator view).

Reuse patterns applied (Rule 3): **pack into destination** — TILE no-gamma normalize → `cb_out`;
**transform in place** — RM+gamma scale runs in place on `cb_normed` with real per-tile credit;
~~alias `cb_sumsq` ↔ `cb_normed`~~ (dropped by the implementer — ring-wrap of indexed windows, see the `cb_sumsq` row); **fold into an accumulator** — Σx² accumulated by the
packer into one L1 tile per row instead of materializing `r·Wc` squared tiles (the DEST variant is
ruled out: ELWMUL ignores `acc_to_dest`). Rejected: `cb_out` in-place on `cb_normed` for TILE (would
give `cb_normed` two consumers, compute and writer).

## Symbol table

| Symbol | Meaning | Bound | Predicate establishing it |
|--------|---------|-------|---------------------------|
| `r` | `block_rows` | `1 ≤ r ≤ rows_cap` | block-selection function (`op_design.md`): largest r with `F(r, K_floor) ≤ B` |
| `c` | `w_chunk_tiles = ceil(Wc / K)` | `min(Wc, 8) ≤ c ≤ Wc` | block-selection function: smallest K with `F(r, K) ≤ B`, `K ≤ K_floor` |
| `K` | `num_w_chunks` | `1 ≤ K ≤ K_floor = ceil(Wc / min(Wc, 8))` | same |
| `Wc` | `core_w_tiles = ceil(W/32)` (R1: whole row) | `F(1, K_floor) ≤ B` (R1 legality) | R1 predicate; otherwise `NotImplementedError` (R3 deferred) |
| `Wp` | `K·c` | `Wc ≤ Wp < Wc + K` | definition |
| `n_sq` | sumsq tiles per row | `1 ≤ n_sq ≤ 2` | `(Wc − p > 0) + p`, `p = (W % 32 ≠ 0)` — host `_n_sq(g, core_w_tiles)` and the compute kernel both derive it from the `core_w_tiles` knob |
| `rows_cap` | max tile-rows per core | `ceil(Rt / num_cores)` | `split_work_to_cores` |
| `s_x`, `s_g`, `s_o` | input / gamma / output tile bytes | 2048 (bf16) or 4096 (fp32) | dtype |
| `s_f` | fp32 tile bytes | 4096 | — |
| `B` | per-core L1 bytes available to CBs | `get_memory_view(device, L1).largest_contiguous_bytes_free_per_bank − 32 KiB` (net of any L1-resident tensor, incl. the output) | `rms_norm_program_descriptor.py` |
| `[g]`, `[rm]`, `[tile]`, `[p]` | 1 if has_gamma / ROW_MAJOR / TILE / partial W, else 0 | — | — |

## Total per-core footprint

```
F(r, K) =   x_block_depth·r·Wp·s_x                     # cb_x            — scales with r and W (depth parked at 1)
          + [rm]·2c·s_x                                # cb_x_rm         — scales with c
          + [p]·s_f                                    # cb_reduce_mask  — constant (Float32)
          + r·s_f                                      # cb_rstd         — scales with r
          + r·n_sq·s_f                                 # cb_sumsq        — scales with r
          + ([g] or [rm])·r·c·s_f                      # cb_normed       — scales with r·c
          + [g]·2c·s_g                                 # cb_gamma        — scales with c
          + 2c·s_o                                     # cb_out | cb_out_rm — scales with c
```

Measured knobs (Blackhole p150, 12×10 grid, `B = 1 427 520 B` = largest free L1 bank region − 32 KiB;
verifier probe 2026-11-12, `_select_blocks` output):

| Cell | `Rt` | `rows_cap` | `r, c, K` | `F(r, K)` |
|------|------|------------|-----------|-----------|
| bf16 TILE + bf16 gamma, 8192 × 1024 | 256 | 3 | 3, 32, 1 | 876 544 B |
| bf16 TILE + bf16 gamma, 8192 × 4096 | 256 | 3 | 3, 26, 5 | 1 355 776 B |
| bf16 TILE + bf16 gamma, 8192 × 7168 | 256 | 3 | 2, 28, 8 (2 blocks/core) | 1 392 640 B |
| fp32 RM + fp32 gamma, 32 × 8192 (worst Phase 0 cell) | 1 | 1 | 1, 11, 24 | 1 404 928 B |
| bf16 TILE + bf16 gamma, 64 × 12288 (loose) | 2 | 1 | 1, 48, 8 | 1 384 448 B |
| bf16 TILE + bf16 gamma, 32 × 16384 (loose) | 1 | 1 | 1, 27, 19 | 1 390 592 B |

On the wide prefill cells the `r`-first selection pushes `K` to 5–8 and `cb_normed` (`r·c` **Float32**
pages) is the largest term after `cb_x`; once `fp32_dest_acc_en=False` lands (bf16 intermediates,
see the page-format audit) that term halves. 32 × 32768 bf16 (`Wc = 1024`, 2 MiB of x) does not fit
R1 at any block size → `NotImplementedError` naming R3.

## Data-movement budget (R1, chosen split = rows across the full grid)

| Tensor | DRAM crossings | Why that many | Cross-core traffic added |
|--------|----------------|---------------|--------------------------|
| input `x` | 1 | whole rows resident in `cb_x` from sumsq through normalize — never re-read | none |
| output | 1 | each tile/stick written once from a streaming window | none |
| gamma | `Σ_cores num_blocks_this_core` (= active cores when each core has one block) | streamed per chunk, re-read once per block per core (reuse-shared operand, not held) | none |
| reduce mask | 0 | generated in L1 by the reader | none |
| x / output, HEIGHT_SHARDED (R6) | 0 | resident shards consumed / produced in place (RM LOCAL_STICKS: one L1→L1 restride per stick, no DRAM) | none |
| gamma, TILE layout | same as RM gamma | per gamma tile only the broadcast row is read: the 64 B (DRAM-alignment) head of face 0 and face 1 (`g_row0_bytes` / `g_face1_offset` CT args; bf16 / fp32). Block-float (bf8b) gamma still reads whole pages. Refinement 3: the whole-page read re-fetched 2 KiB per tile per block, as many bytes as one x tile-row per block (+33–50% DRAM reads on 2–3-block prefill cores) | none |

Totals per tier: DRAM = `|x| + |out| + Σ_blocks·|gamma_read|` (with `|gamma_read| ≤ |x_block| / (16·r)` — RM gamma: W elements; TILE bf16/fp32 gamma: 2×64 B per tile; TILE bf8b gamma: whole pages = `|x_block| / r`);
cross-core = 0; core-local L1 = x read twice from L1 (sumsq, normalize), intermediates once each.

> Cheapest-traffic split considered: rows across cores (R1) **is** the cheapest for x and output
> (each crosses DRAM exactly once, no cross-core bytes). The only above-minimum term is gamma:
> R4 (read gamma once + multicast) would save `(Σ_blocks − 1)·|gamma|` DRAM bytes (≤ 3% of x bytes
> at r = 1) and add the same bytes on the NoC. Implemented: R1. R4 deferred because the saving is
> bounded by 1/(32·r) of the x traffic while R1's per-core gamma read is the stepping stone R4
> layers onto (the reader already addresses gamma by chunk offset). R3 (W-split + combine) moves
> the same DRAM bytes as R1 plus a per-block gather/multicast; it is deferred because every Phase 0
> shape fits R1's residency predicate and its benefit is occupancy, not traffic.

## R3 W-split (Refinement 4) — CB deltas and data-movement budget

A row-group of `G` cores shares each tile-row; core `slot` owns the W slice
`[w_elem_start, w_elem_end)` (`core_w_tiles` tiles, its own CT kernel group when the slice width or its
partial-tile width differs — e.g. the ragged last shard column). Placement: **WIDTH/BLOCK_SHARDED TILE** →
`cb_x` / `cb_out` are the resident shards (row stride = shard tile width, `K = 1` preferred, as R6);
**WIDTH/BLOCK_SHARDED RM** → LOCAL_STICKS restride of the granule-wide slice sticks (as R6 RM);
**interleaved** (only when `F(1, K_floor) > B` for the whole row, or the `FORCE_W_SPLIT` test hook) →
the R1 streaming CBs over the slice, `G = ceil(Wc / ceil(Wc / grid.x))` cores per grid row.

| CB | Capacity (pages) | Live set | Axis accounting | Page format | Producer | Consumer | Lifetime | Shares with / why not |
|----|------------------|----------|-----------------|-------------|----------|----------|----------|-----------------------|
| `cb_partial` (R3) | `r` | `rows_this_block` | `{rows: spans → r, w: collapsed (slice Σx², col-0 valid), w_chunk: streams}` | DEST width (`interm_fmt`) — leaves DEST once, no L1 accumulation | compute | writer | partial reduce → gather, per block | concurrent with `cb_sumsq` only inside the reduce call (input vs output) — could alias `cb_sumsq` in principle, but it is consumed by a different kernel (writer) while compute moves on; separate ring keeps one producer/one consumer |
| `cb_gather` (R3) | `G·r` | `G·rows_this_block` (root only; contributors only use its address) | `{rows: spans → r, w: spans → G partials, w_chunk: —}` | DEST width | writer (root: after all G arrivals) | compute (root) | gather → combine, per block | must exist at the same L1 address on every core (contributors write into the root's copy by their own write pointer) → allocated on every core; sized exactly `G·r` so every block starts at base. Largest R3 term: `G·r·s_f` (e.g. G = 32, bf16: 64 KiB at r = 1). Float32 pages under 16-bit DEST doubled it and failed `13×777×1023` WIDTH (budget 96 KiB with both 650 KiB shards resident) |
| `cb_rstd_root` (R3) | `r` | `rows_this_block` | `{rows: spans → r}` | DEST width | compute (root) | writer (root) | combine → multicast, per block | separate from `cb_rstd`: the multicast's loopback writes the root's own `cb_rstd`, so `cb_rstd` has one producer (the writer) on every core |
| `cb_rstd` (R3) | `r` | as R1 | as R1 | as R1 | **writer** (rstd multicast receive / loopback) | compute | as R1 | as R1 |

`F_R3(r, K) = F(r, K) + r·s_f (partial) + G·r·s_f (gather) + r·s_f (rstd_root)`; the block selector
(`_select_blocks` / `_select_resident_blocks`) solves against it unchanged.

Data movement (R3): x 1 DRAM crossing (interleaved) / 0 (sharded); out 1 / 0; gamma: each core reads
only its slice → 1 per row-group per block. Cross-core per block per row-group: `(G − 1)·rows`
partial tiles unicast to the root (+ 1 semaphore increment per contributor) and one multicast of `rows`
rstd tiles to the `G` group cores. Semaphores: `SEM_GATHER` (cumulative count, never reset — the root
waits for `(G − 1)·(block + 1)`), plus the `ttnn.Mcast` channel's data-ready / consumer-ready pair.

## R3 decode / few-row tuning (Refinement 5) — deltas

- **Interleaved R3 placement**: a row-group is a `gx × gy` core rectangle (`_r3_group_rect`) at grid rows
  `[i·gy, (i+1)·gy)`; `G = gx·gy ≤ Wc / R3_MIN_SLICE_TILES (5)`, even W split (the first `Wc mod G` slices
  get one extra tile). Taken when `F(1, K_floor) > B` **or** R3 occupies `≥ 1.2×` R1's cores
  (`_take_r3_interleaved`). `cb_gather` is still `G·r` pages on every core: at G = 120 (`32×32768` fp32
  DEST) that is 120 × 4 KiB = 480 KiB per core at r = 1. It fits, because the slice CBs are tiny. Only
  the root reads it.
- **Gather payload**: contributors write faces 0 and 2 of each partial (the column-0 data), not the whole
  tile. That is half the root's incast. Faces 1/3 of `cb_gather` keep stale bytes. They are summed into
  faces 1/3 of `cb_rstd_root` / `cb_rstd`, which are never read: the post-op is scoped to faces 0/2, and the
  normalize multiply broadcasts column 0.
- **`cb_normed` under the stall shadow** (R3 + TILE + gamma, not resident multi-row shards): `r·Wp`
  pages, not `r·c`. It holds the whole block's x·γ from the shadow until the ·rstd pass. With `K = 1`
  (every sharded R3 geometry and most decode slices) `r·Wp = r·c`, so there is no extra L1. The selector
  solves with the block-wide term and falls back to no shadow if that does not fit. `cb_x` is released
  right after the shadow; the block's x is never read again.
- Data movement is unchanged (x, out, gamma crossings as R3). The cross-core gather moves `(G−1)·rows`
  half-tiles instead of full tiles.

## R3 row-wise reduce-scatter combine (Refinement 6) — deltas

`q = combine_rows = ceil(r / G)` (`_combine_rows`; `R3_SCATTER_COMBINE = False` restores `q = r`, the
single-root combine). Row `i` of a block is combined by slot `o = i / q` (the *owner*); the root still
multicasts the whole block's rstd.

| CB | Capacity (pages) | Live set | Axis accounting | Page format | Producer | Consumer | Lifetime | Shares with / why not |
|----|------------------|----------|-----------------|-------------|----------|----------|----------|-----------------------|
| `cb_gather` (R3) | `G·q` (was `G·r`) | `G·q` on an owner (always the full quantum, so each owned block returns the ring to base; `G·(q − n_own)` pages are unused on a ragged owner) | `{rows: spans → q (owned run), w: spans → G partials, slot-major: contributor j's rows at [j·q, j·q + q)}` | DEST width | writer (owner, after the G − 1 remote arrivals) | compute (owner: `reduce<SUM, REDUCE_COL, AccumulateViaAdd, Skip>` over `(G, q)`) | gather → combine, per owned block | same uniform-address requirement as R4 (contributors write by their own write pointer) → every core. Slot-major so a contributor's run for one owner is contiguous. Shrinks the R4 term by `G`: 7168×1024 BLOCK (G = 8, r = 23) 368 KiB → 48 KiB |
| `cb_rstd_root` (R3) | `q` (was `r`) | `q` (n_own real rows + `q − n_own` unused columns of the reduce) | `{rows: spans → q}` | DEST width | compute (owner) | writer (owner) | combine → relay / multicast, per owned block | as R4 |
| `cb_rstd_gather` (R3, `q < r` only) | `r` | `r` (raw scratch on the root; never pushed/popped) | `{rows: spans → r}` | DEST width | writers of the owners (NoC writes of `n_own` tiles at tile `o·q`) | root writer (multicast source) | relay → multicast, per block | must have a uniform address (owners write into the root's copy) → every core. Cannot reuse `cb_rstd` (the multicast's loopback destination on the root) or `cb_rstd_root` (single compute producer) |
| `cb_reduce_scaler` (R3, no partial-tail slice) | `1` | `1` | constant | bfloat16 | reader (`calculate_and_prepare_reduce_scaler<SUM, REDUCE_ROW>`, with block 0's reads) | compute (`reduce<…, ReduceTile>` partial) | whole kernel | constant tile; `cb_reduce_mask` holds the 0/1 mask of the AccumulateViaAdd path, which keeps serving slices with a partial tail |

`F_R3(r, K) = F(r, K) + r·s_f (partial) + G·q·s_f (gather) + q·s_f (rstd_root) + [q < r]·r·s_f (rstd_gather)`
(+ one bf16 scaler tile, below the safety margin).

Data movement (R3, per block per row-group): the `(G − 1)·rows` half-tile partials now go to `num_owners`
destinations instead of one root (a row all-to-all; each contributor sends owner `o`'s run as soon as the
owner's rows are reduced — pipelined with sumsq — and bumps `o`'s `SEM_GATHER`); then `num_owners − 1`
relays of `n_own` rstd tiles to the root (`SEM_RSTD_GATHER`, cumulative), then the same one multicast of
`rows` rstd tiles. With one owner (e.g. decode: `rows = 1`) the relay is skipped and the root
multicasts straight from `cb_rstd_root`, as R5. DRAM crossings unchanged.
