# L1 Ledger: mhc_post

Schema: `.claude/references/l1-footprint-discipline.md`.

**Block axes** (Blocking Model):
- `t`: token-tile row, extent `block_rows` = 1.
- `c`: C-column tiles, extent `bc = block_cols`.
- `s`: streams, extent n = 4, whole.

**Abbreviations:**
- `pgF` = F page (4096 fp32 / 2048 bf16).
- `pgX` = X / X' page.
- `ind` = in_depth, `outd` = out_depth, `cd` = coef_depth, `nr` = norm_rows_in_flight.
- `ncoef` = n + n² (+ n when collapse) = 20 or 24.

**Format rule:** `fp32_dest_acc_en` is always on. Every page that carries a DEST result forward is Float32, except where the contract rounds: X' to X.dtype, y_b and y to bf16, and `cb_mixed_bf16`, which is by definition the bf16 X' the projection and collapse must read.

| CB | Capacity (pages) | Live set | Axis accounting | Page format | Producer | Consumer | Lifetime | Shares with / why not |
|----|------------------|----------|-----------------|-------------|----------|----------|----------|-----------------------|
| `cb_sublayer_tiles` | `ind` | 1 (one unit's F) | {t: streams, c: streams → 1/unit, s: n/a (F has no stream axis)} | F.dtype | reader | compute | block | Capacity is above the live set by `ind − 1`, for double buffering (overlap). Concurrent with every other streaming CB, so no alias. |
| `cb_stream_tiles` | `n·ind` | n (all streams of one unit; X'_j needs all X_i) | {t: streams, c: streams → 1/unit, s: spans → n} | X.dtype | reader | compute | block | Kept separate from `cb_sublayer_tiles` because their page formats differ when the sublayer and stream dtypes differ. Double-buffered. |
| `cb_coef_raw` | 3, or 1 when post/comb/P are one buffer | 1–3 raw tiles of the current row | {t: spans → 1 row, c: none (constant across c), s: none} | Float32 | reader | reader (private landing) | per row | Cannot alias `cb_coef_cols`: the expansion reads raw and writes the column pages at the same time. It is only 4–12 KB. |
| `cb_coef_cols` | `ncoef·cd` | `ncoef` (the row's coefficients, all referenced by every unit of the row) | {t: spans → 1 row ×cd window, c: none (broadcast over c), s: spans → n (post_j, comb_ij, P_i)} | Float32 (UnpackToDestFp32) | reader | compute | per row | `cd = 2` overlaps the next row's expansion with this row's mix. Each page uses only column 0 (128 B of 4 KB), because unpack-to-dest COL bcast reads column 0 of faces 0/2 only (mechanism cap). A denser packing is impossible without an unaligned unpack offset. |
| `cb_mixed_out_tiles` | `n·outd` | n (one unit's X'_0..n-1) | {t: streams, c: streams → 1/unit, s: spans → n} | X.dtype | compute | writer | block | Not shared with `cb_mixed_bf16`: that would give two consumers (the writer and compute), which is a CB-ownership violation. In fp32 mode the formats also differ. |
| `cb_mixed_bf16` (R2) | `n·bc` | n·bc (the projection K-block and y need the whole block) | {t: spans → 1, c: spans → bc, s: spans → n} | Float16_b | compute | compute | block | This is the contract's "bf16 X' as written". It replaces the forbidden DRAM re-read (rejected regime X2). It cannot be the writer's CB (ownership rule). |
| `cb_w_slice` (R2) | W-resident: `n·core_c_tiles`; W-streamed: `n·bc` | same as capacity | {t: none (resident across t), c: spans → core_c_tiles / bc, s: spans → n} | Float32 (srcB holds W's 9-bit mantissa exactly) | reader | compute | kernel (resident) / block (streamed) | Resident so that W crosses DRAM once per core (Rule 1b). The predicate `n·core_c_tiles·4096 ≤ W_RESIDENT_BUDGET` selects the variant. |
| `cb_sumsq_acc` (R2) | 1 | 1 | {t: spans → 1 row, c: streams (folded by packer L1-acc), s: streams (folded)} | Float32 | compute | compute | per row | Folding into one tile through packer L1 accumulation (Rule 3.4) removes any per-unit storage. It cannot alias `cb_ysq_acc`, because both are live across the row. |
| `cb_ysq_acc` (norm) | 1 | 1 | {t: spans → 1, c: streams (folded), s: none} | Float32 | compute | compute | per row | As `cb_sumsq_acc`: concurrent lifetime. |
| `cb_proj_interm` (R2) | 1 | 1 (the partial X'@W of the row across column passes) | {t: spans → 1, c: streams (K accumulated), s: streams (K)} | Float32 | compute | compute | per row | It is the matmul helper's interm. Its lifetime overlaps the acc tiles, so it cannot alias them. |
| `cb_partial_out` (R2) | 2 (+1 norm) | 2–3 (the projection tile plus the column-0 reductions the writer merges into cols 24/25) | {t: spans → 1, c: none (already reduced), s: none} | Float32 | compute | writer | per row | The merge happens on the writer RISC (32-word copies) so that no compute-side column shift is needed. The pages are consumed once per row. |
| `cb_partial_slots` (R2, leaders/roots) | `fanin` | fanin (every contributor's slot; a fixed order needs all of them before the fold ends) | {t: spans → 1 row at a time, c: none, s: none} | Float32 (UnpackToDestFp32) | writer (remote writes land; it pushes after the semaphore count) | compute | per row | Slot-indexed for determinism. Capacity scales with `fanin` (≤ 16, symbol table), not with the core count. |
| `cb_combined` (R2) | 1 | 1 | {t: spans → 1, c: none, s: none} | Float32 | compute | writer | per row | Leader → root send source / root → stats DRAM. Kept separate from the slots because both are live during the fold. |
| `cb_hc_lanes_in` (collapse, root) | 1 | 1 | {t: spans → 1 (32 tokens as lanes), c: none, s: none} | Float32 (UnpackToDestFp32) | writer (permuter) | compute | per row | Could alias `cb_hc_lanes_out` in lifetime, but the in-place sfpi op runs in DEST, so the output pack and the next row's input permutation would contend. 4 KB; kept separate. |
| `cb_hc_lanes_out` (collapse, root) | 1 | 1 | {t: spans → 1, c: none, s: none} | Float32 | compute | writer | per row | As above. |
| `cb_y_hold` (collapse) | `bc·nr` (norm) / unused without norm (y goes straight to `cb_y_out`) | bc·nr (y_b of the rows still waiting for rstd) | {t: spans → nr rows (stall shadow), c: spans → bc, s: none} | Float16_b (y_b is bf16 by contract) | compute | compute | per row (until rstd) | `nr = 2` is the stall-shadow depth (Blocking Model). It cannot be `cb_y_out`: the producer and consumer stages differ, and the writer owns `cb_y_out`. |
| `cb_y_out` (collapse) | `bc` | bc | {t: streams, c: spans → bc, s: none} | Float16_b | compute | writer | block | It is the writer's source, and with the feed also the port's source in R4 (via the y_output staging). |
| `cb_rstd_col` (norm) | `nr` | nr | {t: spans → nr rows, c: none (broadcast), s: none} | Float32 (UnpackToDestFp32) | reader (the root's unicast lands here) | compute | per row | Column-0 page, like the coefficients. Live alongside `cb_y_hold`. |
| `cb_norm_w` (norm) | `core_c_tiles` | core_c_tiles (the whole w slice, resident) | {t: none (resident across t), c: spans → core_c_tiles, s: none} | Float32 (UnpackToDestFp32, Row bcast) | reader | compute | kernel | Resident means w crosses DRAM once per core (C·4 B in total). Bounded by `core_c_tiles` (symbol table). |

## Symbol table

| Symbol | Bound | Predicate establishing it |
|--------|-------|---------------------------|
| n | = 4 | contract (validated) |
| `ncoef` | ≤ 24 | n = 4 |
| `bc = block_cols` | `≤ block_cols_cap = floor(BLOCK_L1_BUDGET / (n·2048 + [W-streamed] n·4096 + (nr+1)·2048))` and `≤ core_c_tiles` | host `choose_block_cols()` |
| `core_c_tiles` | `ceil(tensor_c_tiles / Q)` | split chooser |
| `W_RESIDENT_BUDGET` | host constant 256 KiB → `n·core_c_tiles ≤ 64` | regime predicate R2 vs R2-W-streamed |
| `cb_norm_w` pages | `core_c_tiles ≤ 64` under the same predicate (norm implies R2). When it is exceeded, w streams with `bc` like W (same predicate). | R2 predicate |
| `fanin` | ≤ 16 | host constant `COMBINE_FANIN` (Phase 0 of R2: the grid row width, 11 on an 11×10 grid) |
| `ind`, `outd`, `cd`, `nr` | 2 each (Phase 0) | host constants |
| `BLOCK_L1_BUDGET` | 384 KiB | host constant; total below stays < 1 MiB |

## Total per-core footprint

```
plain (Phase 0):
  ind·pgF + n·ind·pgX + n·outd·pgX + ncoef·cd·4096 + 3·4096
  fp32 worst case: 2·4K + 8·4K + 8·4K + 20·2·4K + 12K = 252 KiB   (bc-independent: streams over c)

collapse + norm (R2), additional:
  n·bc·2048 + n·core_c_tiles·4096 + 3·4096 (accs/interm) + 3·4096 (partial_out)
  + fanin·4096 + 3·4096 (combined, hc in/out) + (bc·nr + bc)·2048 + nr·4096 + core_c_tiles·4096
  T=160, C=5120, bf16, R=1 (bc = core_c_tiles = 2, fanin = 11):
     16K + 32K + 12K + 12K + 44K + 12K + 12K + 8K + 8K = 156 KiB  (+ plain part with bf16 pages ≈ 230 KiB)
```

How each term scales:
- `bc` scales `cb_mixed_bf16` and `cb_y_hold`/`cb_y_out`.
- `core_c_tiles` scales the resident W and w.
- `fanin` scales the slots.
- Everything else is constant per row or per unit.

## Data-movement budget

Chosen split for stats/collapse: R = 1 when `tensor_c_tiles ≥ |L|`, else the chooser's R. Plain: the chooser's balanced R; the traffic is identical for every R.

| Tensor | DRAM crossings | Why that many | Cross-core traffic added |
|--------|----------------|---------------|--------------------------|
| F | 1 | each unit has exactly one owner | — |
| X | 1 | same; X'_j needs all n streams, and all n live in one unit | — |
| post / comb / P | 1 per row group that owns the row = 1 (rows are partitioned). Packed: one tile read serves all three. | the row's tile is read by each core owning a column range of that row: **Q reads of a 4 KB tile per row** (≈ Q·Ht·4 KB; 2.2 MB at T=160, Q=106). This is above the minimum, and the cost is per-transaction rather than per-byte. | — (each core reads its own copy; an mcast per row group is the deferred alternative, but it is negligible next to X) |
| X' | 1 write; **0 re-reads** (projection and y use the in-L1 bf16 copy) | `cb_mixed_bf16` residency | — |
| W | R (1 when R = 1) | resident per core across its rows | R3 would make it 1 at any R |
| w | 1 per row group (resident) | resident | — |
| stats | 1 write (root) | — | Ht·Q·4 KB partials to leaders, plus Ht·ceil(Q/fanin)·4 KB to roots |
| hc | 1 write (root) | — | — |
| y / D | 1 write; R4 (feed) adds (G−1)/G write + read of the staging | R4 stepping stone | rstd: Ht·Q·128 B. Fabric: (G−1)/G·T·C·2 B out per chip. |

Totals at T=160, C=5120, bf16, collapse + norm, R=1:
- DRAM ≈ 1.6 (F) + 6.6 (X) + 6.6 (X') + 2.6 (W) + 1.6 (y) + 0.2 (small) ≈ 19.2 MB, which is the roofline bytes.
- Coefficient reads add ~2.2 MB of 4 KB transactions.
- NoC cross-core ≈ 2.2 MB (partials) + 0.2 MB.

> Cheapest-traffic split considered: C-columns-only (R = 1). W is read once, X' is never re-read, and nothing is added at DRAM. Implemented: the same (the chooser picks R = 1 whenever `tensor_c_tiles ≥ |L|`; the extra W bytes make every R > 1 lose on DRAM). Where R > 1 is forced (small C), the cheapest is R3 W-mcast, saving `(R−1)·n·C·128 B` of DRAM. It is deferred because W-resident R2 is already at the minimum on every shape where R = 1 fills the grid, including both perf-focus cases. The column ranges are identical across row groups, so the R3 family stays reachable without restructuring.
