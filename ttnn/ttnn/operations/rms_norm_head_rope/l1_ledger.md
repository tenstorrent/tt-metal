# L1 Ledger: rms_norm_head_rope

Schema and audits: `.claude/references/l1-footprint-discipline.md`. Block axes (from `op_design.md`
Blocking Model): `r` (row-tile, block extent `block_row_tiles = 1`), `h` (head, block extent `bh =
block_heads`), `d` (head-dim tile, extent `HEAD_TILES = 4`). `BT = bh·HEAD_TILES`.
`DF` = Float32 when `fp32_dest_acc_en`, else Float16_b. Sizes: bf16 tile 2 KB, Float32 tile 4 KB.

| CB | Capacity (pages) | Live set | Axis accounting | Page format | Producer | Consumer | Lifetime | Shares with / why not |
|----|------------------|----------|-----------------|-------------|----------|----------|----------|-----------------------|
| `cb_input_tiles` | `input_depth·BT` | `BT` (x must stay through sum_squares → reduce → normalize) | `{r: streams → 1, h: spans → bh, d: spans → 4}` | Float16_b (input dtype; dataflow payload, never through DEST before landing) | reader | compute | per block | Capacity 2× live set: double buffering (`input_depth = 2`) so reader fetches block b+1 under compute of b. Cannot share: concurrent with every per-block CB |
| `cb_rope_cos` | `rope_depth·HEAD_TILES` | 4 | `{r: streams → 1, h: — (shared over all heads, not scaled), d: spans → 4}` | Float16_b (dataflow payload) | reader | compute | per row-tile | 2× live set: prefetch next row-tile. Separate from `cb_rope_sin` because both are FPU operands live in the same rope chain |
| `cb_rope_sin` | `rope_depth·HEAD_TILES` | 4 | `{r: streams → 1, h: — (shared), d: spans → 4}` | Float16_b | reader | compute | per row-tile | as `cb_rope_cos` |
| `cb_weight_row` | `HC·HEAD_TILES` | `HC·4` (all of the core's weight tiles, row 0 valid) | `{r: — (resident over r), h: spans → HC (core's head group, ⊇ bh), d: spans → 4}` | Float16_b (dataflow payload) | reader | compute | prologue only | Disjoint lifetime with every per-block CB (dead after `expand_weight_rows`); alias candidate with `cb_rotated`. **Not aliased in Phase 0**: it is NoC-written by the reader while `cb_rotated` is compute-internal, so aliasing needs `alias_with` lockstep across two different producer threads, and worst-case total is 996 KB of ~1.4 MB. Recorded as the first reuse to take if L1 tightens |
| `cb_trans_mat` | 1 | 1 (constant; bare literal legal) | `{r: —, h: —, d: — (one 32×32 tile serves every tile)}` | Float16_b (±1 exact) | reader | compute | whole kernel | Constant, live for the whole kernel → cannot share with anything per-block; cannot share with scaler (different contents, both live) |
| `cb_reduce_scaler` | 1 | 1 (constant) | `{r: —, h: —, d: —}` | Float16_b (pool-type-aware scaler, bf16 packed) | reader | compute | whole kernel | Constant, live whole kernel |
| `cb_weight` | `HC·HEAD_TILES` | `HC·4` | `{r: — (resident over r), h: spans → HC, d: spans → 4}` | Float16_b — exact copy of bf16 weight; wider page adds no bits (values are bf16 by construction; dataflow-origin payload passing through DEST unchanged) | compute | compute | whole kernel | Must be resident across all rows (reuse-shared over r). Cannot alias `cb_weight_row` — both live during `expand_weight_rows` (read one, write the other) |
| `cb_sumsq` | `bh` | `bh` | `{r: streams → 1, h: spans → bh, d: reduced away (DEST-accumulated over 4)}` | DF | compute | compute | sum_squares → rsqrt_mean | Live concurrently with x and (written then read) with `cb_rstd`; in-place reduce into itself rejected — reduce helper's pop-vs-reserve order inside `BulkWaitBulkPop` is not a documented contract (`reduce_helpers_compute.hpp:113-114` only says "each output tile is reserved and pushed individually"). Lifetime is disjoint from `cb_normed` / `cb_rotated` (dead before normalize starts) — not aliased: the descriptor API gives each CB index its own allocation (no alias mechanism on this path), and the saving is ≤ `bh`·4 KB = 32 KB of a ≤ 996 KB worst case. Cost 4 KB/head |
| `cb_rstd` | `bh` | `bh` | `{r: streams → 1, h: spans → bh, d: — (col-0 scalar per row, broadcast over d)}` | DF | compute | compute | rsqrt_mean → normalize | See `cb_sumsq` |
| `cb_normed` | `BT` | `BT` (y read by matmul, then again by rope) | `{r: streams → 1, h: spans → bh, d: spans → 4}` | DF | compute | compute | normalize → apply_rope [HAS_ROPE only] | Not allocated without rope (normalize packs into `cb_output_tiles` — pack into destination). With rope: cannot be in-place over x (x is bf16-paged and reader-fed; y needs DF) and is live concurrently with `cb_rotated` |
| `cb_rotated` | `BT` | `BT` | `{r: streams → 1, h: spans → bh, d: spans → 4}` | DF | compute | compute | rotate_pairs → apply_rope [HAS_ROPE only] | Concurrent with `cb_normed` (both operands of the rope chain). Could alias `cb_weight_row` (see that row). Perf lamp (op_design): RoPE reformulation that removes this CB |
| `cb_output_tiles` | `output_depth·BT` | `BT` (writer drains per `STORE_CHUNK_HEADS` heads; tail blocks padded to `BT` by compute) | `{r: streams → 1, h: spans → bh, d: spans → 4}` | Float16_b (output ABI: bf16) | compute | writer | per block | 2× live set: double buffering so the writer drains block b while compute fills b+1 |

## Symbol table

| Symbol | Bound | Predicate establishing it |
|--------|-------|---------------------------|
| `HEAD_TILES` | = 4 | `HEAD_DIM = 128` fixed by the op; W % 128 == 0 validated |
| `bh` (`block_heads`) | 1 ≤ bh ≤ `MAX_BLOCK_HEADS` = 8 | host: `bh = min(HC, MAX_BLOCK_HEADS)` |
| `HC` (`ceil(H/G)`) | 1 ≤ HC ≤ `MAX_RESIDENT_HEADS` = 24 | `select_grid_split` skips any G with `ceil(H/G) > 24` (G = H always qualifies) |
| `input_depth`, `output_depth`, `rope_depth` | = 2 (Phase 0) | host constants |
| `block_row_tiles` | = 1 (Phase 0) | host constant, asserted |
| `DF` size | 4 KB (fp32 DEST) / 2 KB | `fp32_dest_acc_en` |

## Total per-core footprint

```
L1 = 2KB·(input_depth·BT + 2·rope_depth·4 + 2·HC·4 + 2 + output_depth·BT)
   + DF·(2·bh + 2·BT)
```
- Scales with `bh` (block knob): `cb_input_tiles`, `cb_output_tiles`, `cb_sumsq`, `cb_rstd`, `cb_normed`, `cb_rotated`.
- Scales with `HC` (core-assignment knob G): `cb_weight_row`, `cb_weight`.
- Scales with depth knobs: input/output/rope CBs.
- Constant: trans_mat, scaler.

Worst case (bh=8, BT=32, HC=24, DF=4 KB, depths 2): 2 KB·(64+16+192+2+64) + 4 KB·(16+64) = 676 + 320 = **996 KB** (< ~1.4 MB usable).
Target shapes (R ≥ 2048: bh=HC=8): 2 KB·(64+16+64+2+64) + 320 KB = **740 KB**. R=256 (bh=HC=2): **~190 KB**.
No budget solve exists; the caps are mechanism caps on HC/bh, and the inventory is minimal: every phase boundary
that can pack into its destination does (no-rope normalize → output; weight folded into normalize via DEST reuse,
no separate weighted-y buffer).

## Data-movement budget

Chosen split: rows × head-groups, `(G, Rg)` from `select_grid_split`. Numbers for R=2304, W=3072, rope + weight
(G=3, Rg=40, 120 cores). x/out 14.16 MB each; cos, sin 0.59 MB each; weight tile-padded 196 KB (6 KB useful).

| Tensor | DRAM crossings | Why that many | Cross-core traffic added |
|--------|----------------|---------------|--------------------------|
| x | 1 | block resident in `cb_input_tiles` across sum_squares, reduce, normalize — never re-read | none |
| out | 1 | written once per block | none |
| cos | G = 3 | each head-group core of a row-group reads its row-tiles' cos; resident across all its heads (`cb_rope_cos` spans h) | none |
| sin | G = 3 | as cos | none |
| weight | Rg = 40 reads of the row-0 bytes of each core's head group (128 B per tile); = 0.5 MB total | resident per core across all its rows (`cb_weight` spans h, resident over r); only row 0 of each tile pulled | none |
| transformation_mat | 0 | generated in L1 by the reader | none |

Totals per tier: DRAM ≈ 14.16 + 14.16 + 3.54 + 0.5 = **32.4 MB** (minimum ≈ 29.5 MB); cross-core NoC: 0; core-local:
x read from L1 twice per block (sum_squares, normalize), sumsq/rstd once, y twice (matmul, rope), rot once, weight once per tile.

> Cheapest-traffic split considered: rows-only (G=1, 72 cores) — 30.4 MB DRAM (−2.0 MB vs chosen), no cross-core.
> Also R2 (chosen G + cos/sin multicast) — 30.9 MB DRAM (−2.4 MB of cos/sin), adds (G−1)·1.18 MB NoC multicast.
> Implemented: rows × head-groups with G from `select_grid_split`, which already has G=1 in its search space
> (a knob-turn, not a different algorithm). G=1 loses on the per-core-max objective (25 vs 18 units/core,
> 48 of 120 cores idle). R2 deferred because the measured `shared_input_reuse` threshold (window ≥ 0.5× own input)
> is not met (32 KB vs 128 KB = 0.25×) and lines of ≤ 4 receivers measured as losses (`mcast_topology`); the
> cos/sin CBs are reader-fed and compute-consumed only, so a receiver swap keeps R2 reachable.

## Implementation notes (ttnn-implementer, Phase 0)

- Tail blocks (`valid_heads < bh`): `cb_input_tiles` / `cb_output_tiles` are pushed/popped in whole `BT` windows
  (reader pushes `BT`, compute pads the output window); compute-internal `cb_rstd`, `cb_normed`, `cb_rotated` are
  realigned to a block boundary after each block. Keeps every multi-tile window contiguous in the ring (no wrap).
  Capacities are unchanged from the table above.
- Writer streams each output block in chunks of `STORE_CHUNK_HEADS` (host knob, default 1) heads with one flush per
  chunk and one barrier per block. Data-movement budget unchanged.

## Verifier audit notes (Phase 0)

- Ledger currency re-checked against `rms_norm_head_rope_program_descriptor.py`: all 12 CBs present with matching
  size expressions and formats; no undeclared CB.
- Page-format audit: DEST-width (`DF`) pages only on compute-internal CBs that carry DEST results
  (`cb_sumsq`, `cb_rstd`, `cb_normed`, `cb_rotated`); all bf16 when `fp32_dest_acc_en=False`. `cb_weight` stays
  bf16 under fp32 DEST because its values are an exact copy of bf16 weight. The op_design "Intermediate page format"
  perf lamp (bf16 `cb_normed` / `cb_rotated` under fp32 DEST) would be an **under**-width page — it truncates y and
  rot at a phase boundary and is out of bounds as a perf lever.
- FPU operand placement (compute kernel): the DEST-width operand is fed on srcB, the bf16 operand on srcA — a
  precision fix applied by the verifier (no L1 change).
