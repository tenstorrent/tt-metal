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
| `cb_kpages` (Perf 1) | `1 + n` = 5 (20 KB) | Float32 (UnpackToDestFp32) | compute (`build_kpages`, row start) | compute (fused mix sweep) | The row's K pages: page s holds coef(s, j)[token R] at (R, C), j = 2·(C ≥ 16) + (C & 1), so one SFPU load serves the 4 vectors of a row group. Built from `cb_coef_cols` once per row, popped with it. Cannot alias `cb_coef_cols` (built from it, and collapse still broadcasts P from it). |
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
| `cb_y_hold` | `mr_rows · ccp` (all the core's rows, not `nr`) | Float16_b | compute (collapse pack) | compute (Σy² reload, norm_block) | The roots run after their own rows (Refinement 1's deferred root), so rstd of any row returns only at the end of the call: every y_b of the core is held. Bounded by `Y_HOLD_BUDGET` = 512 KiB (largest golden shape T2048 C7168: 64 rows × 3 = 384 KiB). Cannot be `cb_y_out` (different consumer stage; the writer owns `cb_y_out`). Refinement 3: with the lagged-root knob (`NORM_ROOT_LAG` > 0) it holds `min(mr_rows, NORM_ROWS_IN_FLIGHT) · ccp` (a wrapping ring: the Σy² reload addresses wrapped pages as `idx − hold_pages`); the knob is parked at 0 (deferred: every row held), because lagging the roots loses at T160 and T1280 (see changelog). |
| `cb_ysq_acc` | 1 | Float32 (UnpackToDestFp32) | compute | compute | Reload accumulator of Σ y_b² (the bf16 y_b reloaded: an exact fp32 sum of the rounded values), zeroed with `cb_sumsq_acc` by the same fill chain. Live across the row with `cb_sumsq_acc`: no alias. |
| `cb_sumsq_col` | `PARTIAL_DEPTH · 2` with norm | Float32 | compute | writer | Page 2 of a row carries Σ y_b² in column 0; the writer merges it into partial column ng+1 (25). Reused CB, not a new one. |
| `cb_rstd_col` | `RSTD_COL_DEPTH` = 2 | Float32 (UnpackToDestFp32) | reader (deferred, default) / writer (lagged knob) — one producer per build | compute | Column-0 page (COL broadcast), expanded row by row from the landing. |
| `cb_norm_w` | `ccp` | Float32 (UnpackToDestFp32, Row broadcast) | reader | compute | Resident w slice, filler zeroed. |
| `cb_rstd_land` | `mr_rows` × 128 B | raw | root writers (remote NoC writes) | reader (private; the writer in the lagged knob) | One slot per row: rows arrive from different roots in any order. 128 B = 32 fp32 tokens. |
| `cb_rstd_send` | 1 × 128 B | raw | root writer | root writer | Lane vector → token order before the Q unicasts. |
| `cb_feed_chunks` (port cores) | `(FEED_CHUNKS_IN_FLIGHT + FEED_READ_BATCH) · run_pages` (= 8 × 7 bf16 tiles = 112 KiB) | Float16_b | port reader | port sender | fabric_all_gather's 112 KiB ring: up to `FEED_CHUNKS_IN_FLIGHT` packets in flight on the sender while the reader fills `FEED_READ_BATCH` chunks behind one read barrier (Refinement 3). |
| `cb_feed_poll` (workers + port cores, one address) | 1 × 16 B · #workers | raw | workers (remote inline writes of their published-row count, after the port armed them) | port reader (local scan) | Refinement 3 push publication: allocated on the workers too only so that a worker's `get_write_ptr` is the ports' slot address (1.7 KiB on a worker, never touched there). The port zeroes its slots before arming. |
| `cb_feed_yscr` (port cores, T % 32 ≠ 0) | `2 · run_pages` bf16 tiles (28 KiB) | Float16_b | port reader (private) | port reader | Refinement 2b: the ≤ 2 y tiles each staged D tile / piece column spans, before its face rows are copied into the ring slot. Cannot share `cb_feed_chunks`: the ring slot is the copy's destination, live at the same time. |
| `cb_feed_land` (port cores, T % 32 ≠ 0) | `nimg · nsh · Cb` bf16 tiles (nsh ≤ G−1 shared D tile rows a image; C 7168: 168 tiles = 336 KiB; `FEED_LAND_BUDGET` 768 KiB, host refusal above) | raw (compact contributor blocks) | every chip's senders (fabric fused write + inc) and my own (local writes) | landing port's sender (merge) | Refinement 2b: shared D tile rows, written to D by this chip's landing port only. Allocated on every port core (one CB, one address on every core and chip) though only port (0,0) receives; port cores are otherwise idle L1. The merge reuses the drained `cb_feed_chunks` as its tile scratch. |

Port cores are not workers, so the port CBs do not add to a worker's footprint.

## Refinement 4 changes (as implemented)

| CB | Capacity (pages) | Page format | Producer | Consumer | Shares with / why not |
|----|------------------|-------------|----------|----------|-----------------------|
| `cb_w_slice` | unchanged | unchanged | reader (my 1/R share from DRAM) **+ the R−1 peer readers** (NoC writes of their shares into the same L1, counted on `SEM_W_SHARE`) — the reader pushes once all R shares landed, so the CB keeps one producer | compute | Slice sharing (`SLICE_SHARE`): the R cores of a column member hold the same slice; the peers' writes land before this core's reader pushes (one push, one producer for the CB protocol). Issued behind the first block's barrier instead of up front; the compute waits for it only at its first matmul K-block (in `MixPassFn`). |
| `cb_norm_w` | unchanged | unchanged | reader + peer readers (same mechanism, same semaphore) | compute | Same sharing as W; needed only in norm_block. |
| `cb_rstd_lane` (norm) | 1 | Float32 | compute (root: the rstd step, run before the hc step) | writer (root: lane vector ng+1 → token order → the Q unicasts) | New. Cannot be `cb_hc_out`: that would need the rstd and hc tiles in one 1-page CB in sequence; a shared single-instantiation variant (`hc_chain(rstd_only)`) measured a larger program (71.8 KB vs 71.2 KB, the kernel-config buffer). 4 KiB on every worker. |

Kernel-config ring (not L1 but the binding budget at bc = 4): under trace replay a launch overlaps the previous one
only if binaries + 2 × (args + CB / semaphore config) ≤ 70 656 B. The feed program at R = 5 / bc = 4 under the device
profiler is at that edge: reader built -Os on the W path (`MHC_READER_OS`), writer's cold root / leader paths
noinline -Os, the row group's coordinates as the writer's RT args (not the 105-entry table as common args). Without
these the feed call is DISPATCH-BOUND (12.8 µs launch gap). Any growth of the worker program must re-check it.
Refinement 5 (the coefficient multicast, +632 B reader, +1 semaphore id): made room by moving the ports' 106-word
worker table from per-core to common RT args of the port reader (the program's unique-RTA region is the max over
kernel groups, set by the port group: 928 → 688 B, counted twice). Measured (`TT_METAL_LOG_JIT_TELEMETRY=1`, feed
program): kernel text 67 744 B + 2 × (688 RTA + 112 sem + 480 CB + 32) = 70 368 B ≤ 70 656 (288 B left). The
mcast_pipe helper version (McastFamily) cost 828 B of reader text, 13 RT words and 2 semaphore ids: 416 B over.

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
| `bc` with W | ≤ `MAX_BLOCK_COLS_W` = 4, balanced passes `ceil(core_c_tiles / ceil(core_c_tiles / 4))` (Refinement 4) | kernel-config buffer (70 656 B): the fused program at bc = 5 measured 74 KB |
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

Chosen split for stats/collapse (as implemented, Refinement 4): the W-path chooser (per-core compute model counting the filler units of balanced passes, multi-pass splits allowed, W-resident splits preferred) — R = 5, Q = 21 (feed mesh, 105 of 106 workers) / Q = 22 (single chip) at T160 C5120: 1 row × 8 units (2 passes of 4) a core; T1280 C5120: R = 4 on 110 cores. Slice sharing makes W and w cross DRAM once at any R (each of the R cores of a column member reads 1/R and NoC-writes it to the other R−1). Plain: the chooser's balanced R; the traffic is identical for every R.

| Tensor | DRAM crossings | Why that many | Cross-core traffic added |
|--------|----------------|---------------|--------------------------|
| F | 1 | each unit has exactly one owner | — |
| X | 1 | same; X'_j needs all n streams, and all n live in one unit | — |
| post / comb / P | 1 per chunk of the row group that owns the row (Refinement 5: a row group = 1–4 one-rectangle chunks of consecutive cores; 14 chunks for the 5 row groups on the feed mesh at T160, 5 single-chip). Packed: one tile read serves all three. | `COEF_MCAST`: the chunk's first core reads row t's tiles and NoC-multicasts them into the same `cb_coef_raw` landing of the chunk (one ack per receiver per row, counter semaphore `SEM_COEF`). Before: **Q reads of the same 4 KB tile per row**, 21–22 cores on one DRAM bank at the call's start. | (chunk size − 1) · span per row and chunk over the NoC multicast (span = 4–12 KB); + 1 ack atomic per receiver per row |
| X' | 1 write; **0 re-reads** (projection and y use the in-L1 bf16 copy) | `cb_mixed_bf16` residency | — |
| W | 1 (Refinement 4 slice sharing; R before) | resident per core across its rows; each of the R cores of a column member reads 1/R | (R−1)/R · n·C·128 B per member set over the NoC (≈ 2.1 MB at R = 5) |
| w | 1 (slice sharing; R before) | resident | (R−1)/R of the core's w tiles to its peers |
| stats | 1 write (root) | — | Ht·Q·4 KB partials to leaders, plus Ht·ceil(Q/fanin)·4 KB to roots |
| hc | 1 write (root) | — | — |
| y / D | 1 write; R4 (feed) adds (G−1)/G write + read of the staging; T % 32 ≠ 0 (Refinement 2b): the own block is staged too (+1/G write + read) and the ports read ≤ 2 y tiles per shifted D tile (staging read ≤ 2× for chips q ≥ 1) | R4 stepping stone (as implemented: the own block straight to D, the G−1 remote blocks to y_output then read by the ports) | rstd: Ht·Q·128 B (Refinement 2: one 128 B unicast per member per row). Fabric: (G−1)/G·T·C·2 B out per chip, ≤ 7-tile packets (bank-contiguous runs of the destination's D). Feed publication (Refinement 3): one 4 B inline write per (worker, row, port) into the ports' `cb_feed_poll` + one arm atomic per (port, worker) — replaces the ports' NoC polling. DRAM crossings unchanged (R5 would remove the staging). |

Totals at T=160, C=5120, bf16, collapse + norm (R=1 at Refinement 3; at R=5 with slice sharing the DRAM total is the same and the coefficient reads drop from Q·Ht = 530 to 105 tile reads — one row a core):
- DRAM ≈ 1.6 (F) + 6.6 (X) + 6.6 (X') + 2.6 (W) + 1.6 (y) + 0.2 (small) ≈ 19.2 MB, which is the roofline bytes.
- Coefficient reads: 105 tile reads (Refinement 4, one row a core) → 14 on the feed mesh / 5 single-chip (Refinement 5 multicast), 56 KB.
- NoC cross-core ≈ 2.2 MB (partials) + 0.2 MB.

> Cheapest-traffic split considered: C-columns-only (R = 1). W is read once, X' is never re-read, and nothing is added at DRAM. Implemented: the same (the chooser picks R = 1 whenever `tensor_c_tiles ≥ |L|`; the extra W bytes make every R > 1 lose on DRAM). Where R > 1 is forced (small C), the cheapest is R3 W-mcast, saving `(R−1)·n·C·128 B` of DRAM. It is deferred because W-resident R2 is already at the minimum on every shape where R = 1 fills the grid, including both perf-focus cases. The column ranges are identical across row groups, so the R3 family stays reachable without restructuring.
