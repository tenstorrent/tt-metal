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
| `cb_sublayer_tiles` | `ind·bc` (implemented: IN_BLOCKS = 2 blocks) | bc (one block's F; mix_block holds one coefficient broadcast in DEST across the block's units) | {t: streams, c: spans → bc per block, s: n/a (F has no stream axis)} | F.dtype | reader | compute | block | Double-buffered in blocks (overlap). A ragged pass still occupies bc slots (filler) so every consumer window starts on a block boundary and never wraps. Concurrent with every other streaming CB, so no alias. |
| `cb_stream_tiles` | `n·ind·bc` (IN_BLOCKS = 2) | n·bc (all streams of the block's units, unit-major) | {t: streams, c: spans → bc, s: spans → n} | X.dtype | reader | compute | block | Kept separate from `cb_sublayer_tiles` because their page formats differ when the sublayer and stream dtypes differ. Double-buffered in blocks; filler slots on a ragged pass as above. |
| `cb_coef_raw` | Phase 0 (plain): 2 (post, comb; only 1 used when they are one buffer). Collapse: 3 (post, comb, P), 1 used when one packed buffer serves all | 1–3 raw tiles of the current row | {t: spans → 1 row, c: none (constant across c), s: none} | Float32 | reader | reader (private landing) | per row | Cannot alias `cb_coef_cols`: the expansion reads raw and writes the column pages at the same time. It is only 4–12 KB. |
| `cb_coef_cols` | `ncoef·cd` | `ncoef` (the row's coefficients, all referenced by every unit of the row) | {t: spans → 1 row ×cd window, c: none (broadcast over c), s: spans → n (post_j, comb_ij, P_i)} | Float32 (UnpackToDestFp32) | reader | compute | per row | `cd = 2` overlaps the next row's expansion with this row's mix. Each page uses only column 0 (128 B of 4 KB), because unpack-to-dest COL bcast reads column 0 of faces 0/2 only (mechanism cap). A denser packing is impossible without an unaligned unpack offset. |
| `cb_mixed_out_tiles` | `n·outd·bc` (OUT_BLOCKS = 2) | n·bc (one block's X', j-major) | {t: streams, c: spans → bc, s: spans → n} | X.dtype | compute | writer | block | Not shared with `cb_mixed_bf16`: that would give two consumers (the writer and compute), which is a CB-ownership violation. In fp32 mode the formats also differ. |
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

## R2 as implemented (Refinement 1)

These rows supersede the planned R2 rows above where they differ. `nc` = number of chunks = ceil(Q / fanin).
`mr` = rows a core roots = ceil(rows_per_group / Q).

| CB | Capacity (pages) | Page format | Producer | Consumer | Shares with / why not |
|----|------------------|-------------|----------|----------|-----------------------|
| `cb_coef_raw` | `2 · (2 \| 3)` (COEF_RAW_BUFS × raw tiles) | Float32 | reader | reader (private) | Double-buffered by row parity so that row t+1's tiles land while row t expands. Cannot alias `cb_coef_cols` (read while written). |
| `cb_mixed_bf16` | `n·bc` | Float16_b | compute (mix pack) | compute (matmul in0, collapse) | Single consumer thread. It is the bf16 X' as written. Cannot be the writer's `cb_mixed_out_tiles` (two consumers). |
| `cb_w_slice` | resident `passes·n·bc`, streamed `2·n·bc` | Float32 (not UnpackToDestFp32: FPU srcA) | reader | compute (matmul in1) | Layout per pass in K order j·bc+u. Filler rows of a ragged pass are zeroed. |
| `cb_sumsq_acc` | 1 | Float32 (UnpackToDestFp32) | compute | compute | In-place reload accumulator, zeroed per row. Packer L1-acc is not used (fp32 DEST). Live across the whole row, so it cannot alias. |
| `cb_proj_interm` | 1 | Float32 (UnpackToDestFp32) | compute | compute | Matmul spill across passes. Live during the row's K-loop. |
| `cb_proj_out` / `cb_sumsq_col` | PARTIAL_DEPTH = 2 each | Float32 | compute | writer | Depth 2 lets compute produce row t+1's partial while the writer does the lagged leader duty of row t. |
| `cb_leader_slots` | fanin (whole CB waited / popped) | Float32 (UnpackToDestFp32) | writer (remote writes land) | compute | The FIFO pointer stays on the base address the remote writers target. Reuse is credited by the ready bitmask. |
| `cb_chunk_sum` | 1 | Float32 | compute | writer | The leader's fold goes to the root. |
| `cb_root_slots` | `nc · mr` (one region per rooted row) | Float32 (UnpackToDestFp32) | writer | compute | Deferred root, so regions are never reused within a call. |
| `cb_stats` | 1 | Float32 | compute | writer | |
| `cb_hc_in` / `cb_hc_out` | 1 each | Float32 (in: UnpackToDestFp32) | writer / compute | compute / writer | Lane layout. |
| `cb_hc_tile` | 1 | Float32 | writer (private) | writer | Row-layout landing of hc for the DRAM write. |
| `cb_y_out` | Y_BLOCKS·bc | Float16_b | compute | writer | Filler y is dropped by the writer. |

`cb_y_hold`, `cb_ysq_acc`, `cb_rstd_col` and `cb_norm_w` belong to Refinement 2 (norm) and are not allocated yet.
L1 at T160 C5120 (R=2, Q=55, bc=3, W resident) is well under budget: the R2 additions are ≈ 24 + 48 + 16 + 44 +
20 + 12·4 KB ≈ 200 KB.

## R2-norm and R4 feed as implemented (Refinement 2)

These rows supersede the planned `cb_y_hold` / `cb_ysq_acc` / `cb_rstd_col` / `cb_norm_w` rows above where they
differ. `mr_rows` = rows of the core's row group; `ccp` = the core's columns padded to whole passes
(passes · bc).

| CB | Capacity (pages) | Page format | Producer | Consumer | Shares with / why not |
|----|------------------|-------------|----------|----------|-----------------------|
| `cb_y_hold` | `mr_rows · ccp` (all the core's rows, not `nr`) | Float16_b | compute (collapse pack) | compute (Σy² reload, norm_block) | The roots run after their own rows (Refinement 1's deferred root), so rstd of any row returns only at the end of the call: every y_b of the core is held. Bounded by `Y_HOLD_BUDGET` = 512 KiB (largest golden shape T2048 C7168: 64 rows × 3 = 384 KiB). Cannot be `cb_y_out` (different consumer stage; the writer owns `cb_y_out`). Shrinks to `nr · ccp` once the root is lagged instead of deferred (Refinement 3/4 lever). |
| `cb_ysq_acc` | 1 | Float32 (UnpackToDestFp32) | compute | compute | Reload accumulator of Σ y_b² (the bf16 y_b reloaded: an exact fp32 sum of the rounded values), zeroed with `cb_sumsq_acc` by the same fill chain. Live across the row with `cb_sumsq_acc`: no alias. |
| `cb_sumsq_col` | `PARTIAL_DEPTH · 2` with norm | Float32 | compute | writer | Page 2 of a row carries Σ y_b² in column 0; the writer merges it into partial column ng+1 (25). Reused CB, not a new one. |
| `cb_rstd_col` | `NORM_ROWS_IN_FLIGHT` = 2 | Float32 (UnpackToDestFp32) | reader | compute | Column-0 page (COL broadcast), expanded row by row from the landing. |
| `cb_norm_w` | `ccp` | Float32 (UnpackToDestFp32, Row broadcast) | reader | compute | Resident w slice, filler zeroed. |
| `cb_rstd_land` | `mr_rows` × 128 B | raw | root writers (remote NoC writes) | reader (private) | One slot per row: rows arrive from different roots in any order. 128 B = 32 fp32 tokens. |
| `cb_rstd_send` | 1 × 128 B | raw | root writer | root writer | Lane vector → token order before the Q unicasts. |
| `cb_feed_chunks` (port cores) | `2 · FEED_CHUNKS_IN_FLIGHT · run_pages` (= 8 × 7 bf16 tiles = 112 KiB) | Float16_b | port reader | port sender | fabric_all_gather's 112 KiB ring; up to `FEED_CHUNKS_IN_FLIGHT` packets in flight. |
| `cb_feed_poll` (port cores) | 1 × 16 B · #workers | raw | port reader (private) | port reader | Landing of the workers' SEM_FEED_ROWS reads. |

Port cores are not workers, so the port CBs do not add to a worker's footprint.

## Symbol table

| Symbol | Bound | Predicate establishing it |
|--------|-------|---------------------------|
| n | = 4 | contract (validated) |
| `ncoef` | ≤ 24 | n = 4 |
| `bc = block_cols` | `≤ core_c_tiles`; `≤ floor(BLOCK_L1_BUDGET / bytes_per_block_col)` with plain `bytes_per_block_col = 2·(pgF + n·pgX) + 2·n·pgX` (R2 adds `n·2048 + [W-streamed] n·4096 + (nr+1)·2048`); `≤ DEST fp32 slots − 2` (= 6, full-sync DEST: one accumulator per unit + coefficient + operand) | host `choose_block_cols()` |
| `core_c_tiles` | `ceil(tensor_c_tiles / Q)` | split chooser |
| `W_RESIDENT_BUDGET` | host constant 256 KiB → `n·core_c_tiles ≤ 64` | regime predicate R2 vs R2-W-streamed |
| `cb_norm_w` pages | `core_c_tiles ≤ 64` under the same predicate (norm implies R2). When it is exceeded, w streams with `bc` like W (same predicate). | R2 predicate |
| `fanin` | ≤ 16 | host constant `COMBINE_FANIN` = 11 (implemented) |
| `bc` with W | ≤ `MAX_BLOCK_COLS_W` = 4 | kernel-config buffer (70 656 B): the fused program at bc = 5 measured 74 KB |
| `ind`, `outd`, `cd`, `nr` | 2 each (Phase 0); `ind`/`outd` are counted in blocks of `bc` units (IN_BLOCKS / OUT_BLOCKS) | host constants |
| `BLOCK_L1_BUDGET` | 384 KiB | host constant; total below stays < 1 MiB |

## Total per-core footprint

```
plain (Phase 0, as implemented):
  bc·(ind·pgF + n·ind·pgX + n·outd·pgX) + ncoef·cd·4096 + 2·4096   (2·4096 = cb_coef_raw, 2 pages)
  fp32 worst case (bc = 5, the L1 cap at BLOCK_L1_BUDGET = 384 KiB): 5·72K + 160K + 8K = 528 KiB

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

Chosen split for stats/collapse (as implemented): the W-path chooser (per-core compute model, single-pass splits) — R = 2 at T160 / T1280, C5120 on 110 cores, so W crosses DRAM twice (+2.6 MB, ≈ 5 µs of DRAM time; it measured 35 % faster at T1280 than R = 1). Plain: the chooser's balanced R; the traffic is identical for every R.

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
| y / D | 1 write; R4 (feed) adds (G−1)/G write + read of the staging | R4 stepping stone (as implemented: the own block straight to D, the G−1 remote blocks to y_output then read by the ports) | rstd: Ht·Q·128 B (Refinement 2: one 128 B unicast per member per row). Fabric: (G−1)/G·T·C·2 B out per chip, ≤ 7-tile packets (bank-contiguous runs of the destination's D). |

Totals at T=160, C=5120, bf16, collapse + norm, R=1:
- DRAM ≈ 1.6 (F) + 6.6 (X) + 6.6 (X') + 2.6 (W) + 1.6 (y) + 0.2 (small) ≈ 19.2 MB, which is the roofline bytes.
- Coefficient reads add ~2.2 MB of 4 KB transactions.
- NoC cross-core ≈ 2.2 MB (partials) + 0.2 MB.

> Cheapest-traffic split considered: C-columns-only (R = 1). W is read once, X' is never re-read, and nothing is added at DRAM. Implemented: the same (the chooser picks R = 1 whenever `tensor_c_tiles ≥ |L|`; the extra W bytes make every R > 1 lose on DRAM). Where R > 1 is forced (small C), the cheapest is R3 W-mcast, saving `(R−1)·n·C·128 B` of DRAM. It is deferred because W-resident R2 is already at the minimum on every shape where R = 1 fills the grid, including both perf-focus cases. The column ranges are identical across row groups, so the R3 family stays reachable without restructuring.
