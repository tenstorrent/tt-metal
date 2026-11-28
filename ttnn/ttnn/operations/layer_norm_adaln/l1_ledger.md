# L1 Ledger: layer_norm_adaln

Schema and audits: `.claude/references/l1-footprint-discipline.md`. Block axes (from `op_design.md`
Blocking Model): **rows** (`block_height` = bh), **W** (`block_width` = bw), **slices** (`num_w_slices` = S,
the cross-core axis the gathered stats span). `T16` = 2048 B (bf16 tile). `TA` = acc tile: Float32
4096 B when `fp32_dest_acc_en`, Float16_b 2048 B otherwise.

| CB | Capacity (pages) | Live set | Axis accounting | Page format | Producer | Consumer | Lifetime | Shares with / why not |
|----|------------------|----------|-----------------|-------------|----------|----------|----------|-----------------------|
| `cb_input_tiles` | `INPUT_BLOCK_DEPTH·bh·bw` | `bh·bw` (whole block: read by mean, dev_sq and normalize) | `{rows: spans → bh, W: spans → bw, slices: n/a (one slice per core)}` | bf16 (input dtype; never packed from DEST) | reader | compute | load → end of normalize | none — live across every compute phase of the block; the output cannot pack into it (different consumer: writer vs compute) |
| `cb_reduce_scaler` | 1 | 1 (constant 1.0) | `{rows: none, W: none, slices: none}` — constant | bf16 (scaler format required by reduce LLK; dataflow-written, never through DEST) | reader | compute | kernel | none — constant read by every reduce of the kernel |
| `cb_local_mean` | `bh` | `bh` (r for every row, needed by pivot and publish) | `{rows: spans → bh, W: reduced, slices: own slice}` | acc | compute | compute | mean → publish | not merged into `cb_local_stats`: compute reads r (pivot, M2) while the writer must be the sole consumer of `cb_local_stats` |
| `cb_pivot` | `bh` | `bh` | `{rows: spans → bh, W: reduced, slices: own}` | **bf16** — mechanism: pivot must be exactly src-register-representable so `x − p` is exact and the combine reads the same value (audit 2 *under* answered) | compute | compute | pivot → end of normalize | none — live through the whole block, overlapping every other per-block CB |
| `cb_dev_sq_acc` | `bh` | `bh` (one L1-accumulated tile per row) | `{rows: spans → bh, W: accumulated (streams over bw into 1 tile), slices: own}` | acc (fp32 L1 accumulation) | compute | compute | dev_sq accumulate → dev_sq row reduce | **implementer deviation**: not reduced in place — `reduce()` asserts `input_dfb_id != output_dfb_id` (`reduce_helpers_compute.inl`), so the row reduce writes `cb_dev_sq`. Disjoint lifetime from `cb_row_stats`/`cb_shift_full`/`cb_rstd_full` (alias rejected: dual-role buffer, `bh·TA` ≤ 8 KB) |
| `cb_dev_sq` | `bh` | `bh` (s2 col tile per row) | `{rows: spans → bh, W: reduced, slices: own}` | acc | compute | compute | dev_sq row reduce → publish | could alias `cb_dev_sq_acc`'s storage (disjoint from its accumulate phase only after the reduce pops it) but the helper forbids in==out; kept separate, `bh·TA` |
| `cb_dev_sum_acc` (16-bit DEST only) | `bh` | `bh` (one L1-accumulated tile of `x − p` per row) | `{rows: spans → bh, W: accumulated (streams over bw into 1 tile), slices: own}` | acc (= Float16_b; DEST is 16-bit) | compute | compute | dev_sum accumulate → dev_sum row reduce | not shared with `cb_dev_sq_acc`: both accumulate in the same block (sequential chains) and their reduce outputs are read together by publish; `bh·2 KB` |
| `cb_dev_sum` (16-bit DEST only) | `bh` | `bh` (Σ(x−p) col tile per row) | `{rows: spans → bh, W: reduced, slices: own}` | acc | compute | compute | dev_sum row reduce → publish | reduce helper forbids in==out (see `cb_dev_sq_acc`); `bh·2 KB` |
| `cb_local_stats` | `spr·bh` (`spr` = `stats_per_row`: 2 fp32 DEST, 3 16-bit DEST) | `spr·bh` (fp32 DEST: r, M2; 16-bit DEST: p, d, M2 per row — one send payload) | `{rows: spans → bh, W: reduced, slices: own}` | acc | compute | writer | publish → end of own send round | none — distinct consumer (writer) from all compute-consumed CBs |
| `cb_gathered_stats` | `spr·S·bh·depth` | `spr·S·bh` per block (combine needs all S payloads: μ before M2); `depth` = 2 only for a multi-block core on the one-shot exchange (a peer may run one block ahead and land block b+1 while this core still combines b), else 1 (`gather_depth`, Perf 1 round 3) | `{rows: spans → bh, W: reduced, slices: spans → S}` | acc | writer | compute | exchange → combine | none — written by peers' NoC traffic; disjoint from `cb_local_stats` in role and producer |
| `cb_row_stats` | `2·bh` | `2·bh` (m, rstd per row) | `{rows: spans → bh, W: reduced, slices: combined}` | acc | compute | compute | combine → bcast | disjoint from `cb_dev_sq_acc` / `cb_local_mean` — see `cb_dev_sq_acc` row (dual-role alias rejected) |
| `cb_shift_full` | `bh` | `bh` | `{rows: spans → bh, W: broadcast within tile, slices: combined}` | acc | compute | compute | bcast → normalize | none — live concurrently with `cb_rstd_full` and the x block during normalize |
| `cb_rstd_full` | `bh` | `bh` | `{rows: spans → bh, W: broadcast within tile, slices: combined}` | acc | compute | compute | bcast → normalize | none — concurrent with `cb_shift_full` |
| `cb_weight_rows` | `affine_stream_pages(bw)` = `AFFINE_STREAM_DEPTH · min(AFFINE_STREAM_CHUNK, bw)` (2·8 = 16 at bw ≥ 8) | one chunk (+1 chunk double buffer: the reader lands chunk c+1 under one NoC barrier while compute bcasts chunk c) | `{rows: none, W: streams over bw in chunks, slices: own}` | bf16 (dataflow-written row 0) | reader | compute | first block (affine only) | none — streaming window; capacity > live set by the double-buffer chunk (verifier: was 2 pages with one DRAM barrier per tile — 64 serialized round trips at bw=32; chunking cut 2304×3072 affine 122.9 → 110.8 µs) |
| `cb_bias_rows` | `affine_stream_pages(bw)` | one chunk (+1 chunk double buffer) | `{rows: none, W: streams over bw in chunks, slices: own}` | bf16 | reader | compute | first block (affine only) | could alias `cb_weight_rows` (w and b *are* streamed sequentially, all w then all b), but compute also consumes them sequentially through two separate `unary_bcast` calls bound to distinct CB ids; aliasing saves 32 KB at bw ≥ 8 on a ≤ 574 KB footprint — kept separate |
| `cb_weight_full` | `bw` | `bw` (every row of every block multiplies by all of them) | `{rows: none (reused by all), W: spans → bw, slices: own}` | bf16 — exact copy of bf16 data via unpack→DEST→pack, no accumulation (audit 2 *under* answered: no precision lost) | compute | compute | call | none — resident for the whole call, concurrent with every per-block CB |
| `cb_bias_full` | `bw` | `bw` | `{rows: none, W: spans → bw, slices: own}` | bf16 (same reason) | compute | compute | call | none — same as `cb_weight_full` |
| `cb_output_tiles` | `OUTPUT_ROW_DEPTH·bw` (2·bw) | `bw` (+`bw` double buffer: pack row r+1 while writer drains row r) | `{rows: streams (one row in flight + one buffered), W: spans → bw, slices: own}` | bf16 (output dtype) | compute | writer | normalize → store | none — cannot be `cb_input_tiles` (writer vs compute consumer); capacity > live set by the double-buffer row |

Allocated only when `HAS_AFFINE`: `cb_weight_rows`, `cb_bias_rows`, `cb_weight_full`, `cb_bias_full`.
Allocated only when `fp32_dest_acc_en=False`: `cb_dev_sum_acc`, `cb_dev_sum` (verifier numerics fix — see below).

## Symbol table

| Symbol | Bound | Predicate that establishes it |
|--------|-------|-------------------------------|
| `bh` (`block_height`) | `1 ≤ bh ≤ core_h_tiles`; INPUTS max 2 | `select_blocking`: `footprint ≤ L1_BUDGET_BYTES` |
| `bw` (`block_width`) | `= tensor_w_tiles / S`; INPUTS max 96 (S=1 shapes have bw ≤ 4; S>1 shapes bw ≤ 32) | `S | tensor_w_tiles`; `bw ≥ MIN_SLICE_TILES` when S>1; footprint predicate |
| `S` (`num_w_slices`) | `S | grid_x` (≤ 12 on BH), `S | tensor_w_tiles` | `select_blocking` candidate filter; `spr·S·bh ≤ GATHER_TILE_CAP = 48` |
| `INPUT_BLOCK_DEPTH` | 1 | host constant |
| `OUTPUT_ROW_DEPTH` | 2 | host constant |
| `AFFINE_STREAM_DEPTH` | 2 (chunks) | host constant |
| `AFFINE_STREAM_CHUNK` | 8 (tiles per barrier), clamped to `bw` by `affine_stream_chunk(bw)` | host constant; the reader receives the clamped value as a CT arg |
| `spr` (`stats_per_row`) | 2 (fp32 DEST) / 3 (16-bit DEST) | `stats_per_row(fp32_dest_acc_en)` — single source for CB sizes, writer payload, compute layout, gather cap |
| `TA` | 4096 (fp32 DEST) or 2048 | `compute_kernel_config.fp32_dest_acc_en` |

## Total per-core footprint

```
footprint(bh, bw, S) =
    T16 · ( INPUT_BLOCK_DEPTH·bh·bw            # x block           — scales with bh·bw
          + 1 + bh                              # scaler, pivot
          + OUTPUT_ROW_DEPTH·bw                 # output            — scales with bw
          + HAS_AFFINE·(2·affine_stream_pages(bw) + 2·bw) )   # w/b chunked streams + full — scales with bw
  + TA  · bh · ( 1 + 1 + 1 + spr + spr·S + 2 + 1 + 1 + (fp32_dest ? 0 : 2) )
          # local_mean, dev_sq_acc, dev_sq, local_stats, gathered, row_stats, shift, rstd [, dev_sum_acc, dev_sum]
```

Computed by `layer_norm_adaln.footprint_bytes` (the same function `select_blocking` gates on):

| Shape (selected) | bh, bw, S | fp32 DEST, affine | 16-bit DEST, affine |
|------------------|-----------|-------------------|---------------------|
| (1,1,2048,3072) / (1,1,2304,3072) | 2, 32, 3 | 574 KB | 538 KB |
| (1,1,256,3072) / (1,1,32,3072) | 1, 8, 12 | 280 KB | 244 KB |
| (1,1,256,1024) | 1, 8, 4 | 216 KB | 196 KB |
| (1,1,32,64) | 1, 2, 1 | 84 KB | 70 KB |

All far below the ~1.4 MB BH L1 per core: no budget solve is exercised for any INPUTS shape (the
predicate exists only for the out-of-INPUTS multi-block fallback).

## Data-movement budget

`B = H·W·2` bytes (x and out each). Numbers for (1,1,2048,3072): `B` = 12.58 MB; `num_row_groups` = 40, S = 3.

| Tensor | DRAM crossings | Why that many | Cross-core traffic added |
|--------|----------------|---------------|--------------------------|
| x | 1 (12.58 MB) | whole block resident through mean, dev_sq and normalize (two-pass without re-read) | 0 |
| out | 1 (12.58 MB) | written once from `cb_output_tiles` | 0 |
| weight | `num_row_groups` × (row-0 bytes of the core's slice): each core fetches `bw` tiles × 2 × 64 B → 40 × 96 × 128 B ≈ 0.49 MB | reuse-shared operand, fetched once per core per call and kept resident (`cb_weight_full`) for all blocks; row-0-only fetch (32× fewer bytes than full pages) | 0 |
| bias | same as weight ≈ 0.49 MB | same | 0 |
| per-row stats | 0 | never leave L1 | per group per block: S rotating mcasts of `2·bh·TA` = 3 × 16 KB, each delivered to S−1 = 2 receivers ⇒ 40 × 96 KB ≈ 3.8 MB delivered (on-chip, row-local) |

Totals per tier (2048 rows): **DRAM ≈ 26.1 MB** (x 12.58 + out 12.58 + w/b 0.98); **NoC cross-core ≈ 3.8 MB delivered**
(1.9 MB injected); core-local re-reads: x block read 3× from L1 by compute (mean, dev_sq, normalize).

At (1,1,256,3072) (S=12, 8 groups, bh=1, bw=8): DRAM ≈ 1.57 + 1.57 + 2·(8·96·128 B ≈ 0.1 MB) ≈ 3.3 MB;
NoC: 8 groups × 12 mcasts × 8 KB × 11 receivers ≈ 8.4 MB delivered (0.77 MB injected) — the reason for the
grid-synchronization and stats-transport lamps in `op_design.md`.

Stats payload at fp32 DEST is unchanged (`2·bh·TA` per round). At 16-bit DEST it is `3·bh·TA` with `TA` = 2 KB,
i.e. 0.75× the fp32-DEST bytes.

> Cheapest-traffic split considered: rows × W-slices with resident slices (`rows_x_wslices`) — DRAM at the
> minimum for x/out and the smallest w/b row-0 total (each slice's w/b is fetched by `num_row_groups` cores
> instead of every row-owning core fetching the full width: 0.98 MB vs ~1.6 MB for `rows_only` at 64 cores).
> Implemented: `rows_x_wslices` wherever `select_blocking` picks S>1 (all large shapes), `rows_only` where
> W is too narrow to slice. The implemented split is the cheapest-DRAM split for every INPUTS shape; the only
> cheaper alternative on any tier is `affine_mcast` (w/b once per call, −0.9 MB DRAM at 2048 rows, < 4%),
> recorded as a `deferred` regime: the row-0 fetch already covers it, and the reader-produced
> `cb_weight_rows`/`cb_bias_rows` contract keeps it reachable by swapping the producer only.

## Implementer notes (Phase 0 build)

- `cb_dev_sq` added (slot 15): see the `cb_dev_sq_acc` row. `select_blocking`'s footprint includes it
  (`layer_norm_adaln.footprint_bytes`); no INPUTS shape's selection changes (largest: ≈ 0.52 MB at 2048/2304 rows).
- `unpack_to_dest_mode = UnpackToDestFp32` on `cb_local_mean`, `cb_dev_sq_acc`, `cb_dev_sq`, `cb_gathered_stats`
  (all read by `copy_tile` into SFPU math). **Not** on `cb_row_stats`: its consumer is `unary_bcast<Col>` (srcB
  path), and every downstream use of m / rstd goes through FPU srcB (tf32) anyway.
- Data-movement budget unchanged: x and out cross DRAM once; w/b row-0 per core; stats all-gather row-local.

## Verifier notes (verification pass)

- **Affine stream chunking**: `cb_weight_rows` / `cb_bias_rows` now hold `AFFINE_STREAM_DEPTH` chunks of
  `min(AFFINE_STREAM_CHUNK, bw)` tiles, one NoC barrier per chunk (was one per tile). `footprint_bytes` and the CB
  descriptors read the same `affine_stream_pages(bw)`.
- **16-bit DEST numerics**: with `fp32_dest_acc_en=False` the first-pass mean r is accumulated in bf16 DEST
  (partial sums ~5e4 at mean 50, ulp 256), so `d = r − p` was off by ~1 std and large-offset inputs failed
  (rel-RMS up to 0.998). Under 16-bit DEST only, a `dev_sum` pass measures `d = Σ(x−p)/n` on the O(std) deviations
  (`cb_dev_sum_acc`, `cb_dev_sum`), and the payload becomes `[p, d, M2]` (`spr = 3`) so the combine forms
  `δ_j = (p_j − p_own) + d_j` without rounding a large value. The fp32-DEST path is unchanged (`spr = 2`).
- Audit results (both directions): no fp32 page under 16-bit DEST (all acc CBs follow `TA`); no 16-bit page holding an
  fp32-DEST accumulation except `cb_pivot` (bf16 by mechanism). No `Shares with` cell is blank.

## Perf 1 notes (perf tournament round 1)

- `cb_weight_rows` / `cb_bias_rows` no longer exist as separate allocations: they alias `cb_weight_full` / `cb_bias_full`
  (`bw` bf16 pages each; the column multicast lands row 0 of each tile there and the Row broadcast runs in place).
  Saves `2·AFFINE_STREAM_DEPTH·AFFINE_STREAM_CHUNK` bf16 pages per core (64 KB at bw = 32).
- `cb_local_stats` / `cb_gathered_stats` pages per row: `stats_payload_tiles` = 1 (fp32 DEST, `[r | M2]`) / 2 (16-bit
  DEST, `[p | d]`, `[M2 | −]`) instead of 2 / 3.
- Budget: `footprint_bytes` (== the real CB total, verified against the descriptor) is compared with the device's
  `cb_limit` (CB region from the allocator's unreserved base to the L1 top; 1,460,288 B on BH), not
  `get_max_worker_l1_unreserved_size()` (1,530,976 B, which includes the kernel-config ring).

## Perf 2 notes (perf tournament round 2)

- Deleted: `cb_shift_full`, `cb_rstd_full` (bh acc pages each) and `cb_weight_full` / `cb_bias_full` (formerly aliased
  by the row-0 CBs). normalize broadcasts m / rstd (Col) from `cb_row_stats` and w / b (Row) from `cb_weight_rows` /
  `cb_bias_rows` via srcB hardware broadcast, so `cb_weight_rows` / `cb_bias_rows` (`bw` bf16 pages each, call-resident)
  are now the only affine CBs. Per-core saving: `2·bh·TA` (16 KB at bh = 2, fp32 DEST).
- `cb_output_tiles` is unchanged in size (`OUTPUT_ROW_DEPTH·bw`); it is now produced per DEST window and consumed per
  tile (popped per row).
- `footprint_bytes` updated to match (acc rows: `1 + 1 + 1 + spr + spr·S + 2 [+ 2 under 16-bit DEST]`).

## Perf 2 notes (4th tournament: second round of the current series; supersedes the stats rows above)
- **Removed:** `cb_local_mean` (slot 2), `cb_dev_sum_acc` (16), `cb_dev_sum` (17). The stats are one merged pass in both DEST modes: per DEST window, `D[j] = x − p`, `D[NW+j] = D[j]²` (SFPU), then L1-acc packs.
- **`cb_dev_sq_acc`** is now **2** acc tiles, call-constant: the `[Σ(x−p), Σ(x−p)²]` L1 accumulators of the current row. One Accurate row reduce of both tiles per row writes **`cb_dev_sq` (2·bh)**.
- **`cb_pivot` (bh bf16)** is unchanged, but p is now bf16(mean of the row's first chunk = `STATS_CHUNK` = 4 tiles), not the full-row mean.
- **`unpack_to_dest_mode = UnpackToDestFp32`** on `cb_dev_sq_acc`, `cb_dev_sq`, `cb_gathered_stats` (fp32 DEST).
- **Normalize (fp32 DEST)** holds c′ = −(p+m)·rstd in SFPU LREG0..7 per tile row. That costs no L1 and no CB.
- **`footprint_bytes`** is exact: `acc_tile_bytes · (2 + bh · (2 + spr + spr·S·depth + 2))`. Every INPUTS shape's blocking is unchanged. Some L1-limited 23040-row shapes (cb_limit 1 460 288 B on BH) now pick a taller block (e.g. 23040×2048 wb fp32: S=2, bh=12). All of them fit and are correct.
