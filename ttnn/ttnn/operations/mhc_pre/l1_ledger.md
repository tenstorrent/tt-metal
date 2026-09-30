# L1 Ledger: mhc_pre

Schema and audits: `.claude/references/l1-footprint-discipline.md`. Block axes (from `op_design.md` Blocking Model):
**token** (extent `block_token_tiles`), **K** = stream-column × stream (extent `core_k_tiles`, the whole rank slice; its y-side projection is **C** with extent `core_c_tiles`), **rank** (group ranks, `group_cores`), **slot** (coefficient slots, whole tile), **stream** (n, whole).

Page sizes: `xT` = X tile bytes (fp32 4096, bf16 2048); `wT` = W tile bytes; `yT = xT`; `fT = 4096` (fp32); `hT = 2048` (bf16).

| CB | Capacity (pages) | Live set | Axis accounting | Page format | Producer | Consumer | Lifetime | Shares with / why not |
|----|------------------|----------|-----------------|-------------|----------|----------|----------|-----------------------|
| `cb_x_resident` | `x_block_depth · block_token_tiles · core_k_tiles_max` | same. Block b is held from projection through y-mix while block b+1 prefetches. | {token: spans → block_token_tiles·x_block_depth, K: spans → core_k_tiles_max, rank: streams (one rank per core), slot: n/a (0), stream: spans (inside K)} | X dtype | reader | compute | per block | none. It is the residency that makes X read once. Capacity above one block is the prefetch depth (stall shadow of the combine round trip). |
| `cb_weight` | `core_k_tiles_max` | same (resident all kernel). Pushed by the reader in `W_CHUNK_TILES` chunks. | {token: streams (reused by every block), K: spans → core_k_tiles_max, rank: streams, slot: 0, stream: spans (inside K)} | W dtype (fp32 W: UnpackToDestFp32, read only by the split's `copy_tile`) | reader | compute | whole kernel | none. Concurrent with every other CB. Re-reading per block would double DRAM traffic. |
| `cb_weight_split` (fp32 W only) | `2 · core_k_tiles_max` bf16 pages | same (resident all kernel) | {token: streams, K: spans → core_k_tiles_max (× 2 pieces), rank: streams, slot: 0, stream: spans} | Float16_b, pages [W_hi(k), W_lo(k)] | compute (`w_split_block`) | compute (`project_block_pieces`) | whole kernel, from before block 0 | **Aliases `cb_weight`** (a second buffer index in the same CBDescriptor, so it costs 0 extra bytes). Tile k of fp32 W is rewritten in place into its bf16 pair, which occupies the same 4096 B. That is safe because tile k is fully unpacked into DEST before its pair is packed. |
| `cb_bias_coef` | 1 | 1 (constant: 24 slots × 32 lanes of b_k) | {token: streams (constant), K: 0, rank: 0, slot: spans → 1 tile, stream: 0} | Float32 (UnpackToDestFp32) | reader | compute | whole kernel | none: it is a persistent operand of every `coefficients_block`. Different lifetime from all per-block CBs. |
| `cb_reduce_scaler` | 1 | 1 (constant) | {token: 0, K: 0, rank: 0, slot: 0, stream: 0} | Float16_b | reader | compute | whole kernel | none: its page format differs from every fp32 CB, and it is read by every reduce. |
| `cb_sq_acc` | fp32 X: `bt` (filled exactly by `x_stats_block` before the projection, reduced after it); bf16 X: `bt · X_STREAM_CHUNKS` (one partial per (row, K chunk) from `project_sumsq_streamed`, fp32 W; a bf16 W uses one page per row) | same | {token: spans → bt, K: spans → X_STREAM_CHUNKS partials (each collapsed in DEST), rank: 0, slot: 0, stream: streams} | Float32 (UnpackToDestFp32: the Accurate SFPU row collapse copies Q to DEST; Default would truncate it to tf32) | compute | compute | `project_sumsq_streamed` → the per-row reduce after the last mix row | Not in place: Q differs in shape from `cb_partial`'s pages (32×32 elementwise sum vs col-0 result). The partials must wait until every mix row is in `cb_partial` ([mix rows \| sumsq rows] order the writer relies on), so they cannot fold into DEST (the projection windows interleave). 4 tiles at bt=1 (Refinement 4: was 1). |
| `cb_partial` | `2 · block_token_tiles` | same (the block's [mix rows \| Σx² rows] partials, sent in one transfer) | {token: spans → block_token_tiles, K: streams (reduced), rank: 0, slot: 0, stream: 0} | Float32 | compute | writer | per block | Could alias `cb_combined` (same size): on the root both are live at once (partial in flight while combine waits), so no. Non-roots could, but the address must stay uniform. |
| `cb_gathered` | `group_cores · 2 · block_token_tiles` | same on the root (all ranks' partials); **0 on non-roots** | {token: spans → block_token_tiles, K: 0, rank: spans → group_cores, slot: 0, stream: 0} | Float32 (UnpackToDestFp32) | writer (root; remote fill) | compute (root) | per block | Allocated grid-wide so its address is uniform: remote senders address it with their own base. Non-root capacity is unused, which is the stated cost of the uniform address. It cannot alias `cb_x_resident` / `cb_weight` (concurrent). Capped by `group_cores ≤ 32`. |
| `cb_combined` | `2 · block_token_tiles` | same | {token: spans → block_token_tiles, K: 0, rank: 0 (already summed), slot: 0, stream: 0} | Float32 | root: compute / non-root: remote mcast | writer | per block | **Shared**: the root's compute output and the non-roots' mcast landing are one allocation. The roles are disjoint per core, and the mcast destination address equals the source address. |
| `cb_coef_in` | `COEF_IN_BLOCKS (= 2) · 2 · block_token_tiles` (Perf 1: was `2 · bt`) | S(b) held for coefficients / y-mix while S(b+1) lands (cross-block pipeline; the handshake-free group multicast relies on the second slot) | {token: spans → block_token_tiles, K: 0, rank: 0, slot: spans → 1 tile, stream: 0} | Float32 (UnpackToDestFp32) | writer | compute | per block | Not `cb_combined` in place: on the root `cb_combined` is the multicast source while the loopback copy lands here. The row-major → coefficient-major transform runs in DEST (transpose + SFPU gather), so no scatter CB is needed. |
| `cb_comb_coef` | `2 · block_token_tiles` | owned rows × [post, comb] | {token: spans, K: 0, rank: 0, slot: spans, stream: 0} | Float32 | compute (fused owned block) | writer | per block | none: already the row-major output tiles (the writer stores them as they are). Not `cb_pre_cols`: different consumer (the writer). |
| `cb_coef_keep` | 1 | the owned row's coefficient-major tile after coefficients + Sinkhorn (pre slots untouched) | {token: 1 row, K: 0, rank: 0, slot: spans → 1 tile, stream: 0} | Float32 (UnpackToDestFp32: exact reload) | compute | compute | within one owned row: packed in the Sinkhorn window, reloaded by the pre-tile window | Refinement 4. Replaces a second S gather + coefficients pass on owned rows. Not DEST: the pre tiles need DEST tiles 0..3, which the Sinkhorn window's post/comb tiles occupy. Not `cb_comb_coef` (writer-consumed). 4 KB. |
| `cb_pre_cols` | `n · block_token_tiles` | same | {token: spans → block_token_tiles, K: 0, rank: 0, slot: 0, stream: spans → n} | Float32 | compute | compute | `coefficients_block` → `ymix_block` | none: concurrent with `cb_x_resident` (both are y-mix operands), and the layout differs from the coefficient tile (FPU col-bcast needs col 0). |
| `cb_y_out` | `y_depth · y_chunk_tiles` | `y_chunk_tiles` (one window being packed while one drains) | {token: streams (block rows one after another), K→C: streams → y_chunk_tiles window, rank: 0, slot: 0, stream: streams (accumulated in DEST)} | y dtype | compute | writer | per chunk | none. Streaming window: sizing it to the block (`block_token_tiles·core_c_tiles`) would add up to 84 KB with no reuse. Double buffering = overlap of pack and DRAM write. |
| `cb_x_fp32` (fp32 X only) | = `cb_x_resident` | same | same as `cb_x_resident` | Float32, UnpackToDestFp32 | compute (lockstep reserve/push) | compute (`x_stats_block`, `x_split_window`) | per block, in lockstep with `cb_x_resident` | **Aliases `cb_x_resident`** (second buffer index, 0 bytes). The FPU y-mix keeps reading the Default-mode index; the SFPU stages need the exact fp32 through DEST, and the unpack-to-dest mode is per buffer index. |
| `cb_x_pieces` (fp32 X only) | `X_PIECE_DEPTH · 3 · X_CHUNK_K_TILES · min(bt, 4)` bf16 | one K-chunk window of [x0, x1_hi, x1_mid] | {token: streams (sub-block rows), K: streams → X_CHUNK_K_TILES, rank: 0, slot: 0, stream: spans (inside K)} | Float16_b | compute (`x_split_window`) | compute (projection) | per chunk window | none. Recomputed per K chunk from the resident fp32 block — never a second resident copy (3 pieces of the whole block would be 1.5× the block). Pushed at the nominal window size so it never wraps mid-window. |
| `cb_mix_run` | fp32 X: `min(bt, 4)`; bf16 X: 1 | the running fp32 mix of the current sub-block (bf16 X: one row) | {token: spans → sub-block, K: streams (accumulated), rank: 0, slot: 0, stream: 0} | Float32, UnpackToDestFp32 | compute | compute | between K chunk windows | Not `cb_partial`: that CB's consumer is the writer (single-consumer rule), and the running value is reloaded by compute. Not DEST: DEST does not persist across the windows that interleave with the chunks (fp32 X: the split windows; bf16 X: the Σx² windows, Refinement 4). |
| `cb_max_lanes` (fp32 X only) | 1 | one lane-wise bound tile (x row-tile Σx², or W max\|W\|) | {all: 0 / 1 tile} | Float32 (Default: FPU reduce reads it) | compute | compute (reduce MAX) | per row-tile | Not `cb_sq_acc`: that page is consumed later (after the projection) by the SUM reduce; the MAX reduce needs its own copy of the same DEST tile, packed twice. |
| `cb_max_scalar` (fp32 X only) | 1 | the reduce<MAX, REDUCE_SCALAR> result | 1 tile | Float32, UnpackToDestFp32 (exact scalar broadcast) | compute | compute | per row-tile | none (1 tile, transient). |
| `cb_grid` (fp32 X only) | `bt` | the block's per-row-tile grid-constant tiles (and the W one, once, before block 0) | {token: spans → bt, rest 0} | Float32, UnpackToDestFp32 | compute | compute (splits) | per block | W's grid tile reuses it before block 0 (disjoint lifetime). |
| `cb_max_scaler` (fp32 X only) | 1 | constant | 1 tile | Float16_b | reader | compute | whole kernel | Separate from `cb_reduce_scaler` (pool-type-aware fill: <MAX, REDUCE_SCALAR> vs <SUM, REDUCE_ROW>). |
| `cb_w_own_ready` / `cb_w_own_split` | 1 each (32 B token pages) | 1 token | no payload | Float16_b | writer / compute | compute / writer | once, before block 0 (W column all-gather) | none (tokens: 64 B). |
| `cb_w_share_landed` | 1 (32 B token page) | 1 token | no payload | Float16_b | reader | writer | once, before block 0 (W column all-gather, `W_SHARE_ON_READER`) | Refinement 5. Not `cb_w_own_ready`: that one is writer → compute (single producer / consumer each). The share bytes themselves land in the writer-owned `cb_weight` region (no new buffer). 32 B. |

## Symbol table

| Symbol | Bound | Predicate establishing it |
|--------|-------|---------------------------|
| `n` | = 4 (W: n(n+2)=24); host asserts n(n+2)+1 ≤ 32 | W shape; Mechanism caps (coefficient slots) |
| `group_cores` | 1 ≤ G_k ≤ min(32, Ct) | group geometry derivation, `group_cores_cap = 32` |
| `core_k_tiles_max` | `n · ceil(Ct / group_cores)`; ≤ 84 for every TARGET/INPUTS shape on an 11-wide grid (C=7168); ≤ 112 on an 8-wide grid | group geometry |
| `core_c_tiles_max` | `ceil(Ct / group_cores)` | same |
| `block_token_tiles` | 1 ≤ bt ≤ min(`BLOCK_TOKEN_TILES_CAP`, `core_token_tiles_max`), and the total below ≤ `l1_budget` | selection function (op_design Mechanism caps). Implementation: `BLOCK_TOKEN_TILES_CAP = 1` (measured, perf lamp L1: finer token blocks overlap the X read of block b+1 with block b's combine / y-mix / y stores; the K slice stays whole) |
| `x_block_depth` | ∈ {2, 1}; 1 only when the total with 2 and bt=1 exceeds `l1_budget` | selection function |
| `y_chunk_tiles` | `min(core_c_tiles_max, 8)` | host constant (catalog `double_buffer`: 4–8 in flight) |
| `y_depth` | 2 | host constant |
| `l1_budget` | device L1 per core minus allocator-reserved base (queried from device/allocator at call time; BH ≈ 1.4 MB usable) | host query |

## Total per-core footprint (identical on every core; the uniform-address CBs are counted everywhere)

```
L1 = x_block_depth · block_token_tiles · core_k_tiles_max · xT         # scales with token and K knobs, depth
   + core_k_tiles_max · wT                                              # scales with K
   + block_token_tiles · fT · (2 + 2·group_cores + 2 + 4 + 2 + n + S)  # partial, gathered, combined, coef_in (2 blocks, Perf 1), comb, pre_cols,
                                                                        #   sq_acc (S = X_STREAM_CHUNKS bf16 X, 1 fp32 X)
   + y_depth · y_chunk_tiles · yT
   + fT·(1 + 1 + 1) + hT                                                # bias, coef_keep, mix_run (bf16 X), scaler
```

Worst TARGET case (T=640/4096, C=7168, fp32, 11-wide group, bt=1, depth 2): 688,128 + 344,064 + 4096·(2+22+2+4+4) = 139,264 + 65,536 + 16,384 + 2,048 ≈ **1.26 MB** (fits BH). The same case on an 8-wide grid (WH): `core_k_tiles_max` = 112 → 1.60 MB with depth 2 → the selection falls back to depth 1 → 1.14 MB.

## Data-movement budget

Reference shape T=640, C=7168, n=4, fp32 X and W, 11×10 grid (G_k = 11, G_t = 10).

| Tensor | DRAM crossings | Why that many | Cross-core traffic added |
|--------|----------------|---------------|--------------------------|
| X (73.4 MB) | 1 | Each rank's block stays resident in `cb_x_resident` from projection to y-mix, and blocks are pipelined per group. The resident unit is `block_token_tiles × core_k_tiles`, so read-once holds for every T. | none |
| W (3.67 MB) | 10 (= G_t) | Each group's rank r reads its own W slice once and keeps it resident all kernel (`cb_weight`). It is not broadcast across groups in Phase 0 (R2 deferred). | none (R2 would add a 3.67 MB column mcast) |
| b (4 KB) | 110 (once per core) | read into `cb_bias_coef` once | none |
| y (18.4 MB) | 1 | written once from `cb_y_out` | none |
| post (80 KB tiles) | 1 | written once by the owning rank | none |
| comb (80 KB tiles) | 1 | written once by the owning rank | none |
| partials | 0 | on-chip only | gather: `Mt·G_k·2` = 440 tiles = 1.8 MB (≈ 0.18 MB into each of 10 roots) |
| combined S | 0 | on-chip only | mcast: `Mt·2` tiles × (G_k−1) receivers = 400 tile deliveries = 1.6 MB |

Totals: DRAM ≈ 73.4 + 36.7 + 0.4 + 18.4 + 0.16 ≈ **129 MB** (fp32). The DRAM minimum is 95.9 MB. The bf16-X perf-focus case is 36.7 + 36.7 + 9.2 + … ≈ 83 MB vs a 50 MB minimum. NoC: ≈ 3.4 MB combine traffic (≈ 5 % of X).

> Cheapest-traffic split considered: tokens × stream-column **with W broadcast down the group-rank columns** (R2) — −33 MB DRAM (W 36.7 MB → 3.67 MB), +3.67 MB × (G_t−1) NoC column mcast, one-shot at kernel start. Implemented: tokens × stream-column (R1) with direct per-group W reads. Deferred because R2 is a stepping-stone successor: it changes only the fill path of `cb_weight` (reader DRAM read → `Mcast1D(PerColumn)` receive). Compute, every other CB and the group combine are unchanged. Phase 0 validates the single cross-core mechanism (the group combine) in isolation first. The structure keeps R2 reachable: `cb_weight` is filled once, before block 0, by the reader alone, and the same-rank cores of all groups already share an identical W slice (identical `c_start`/`core_c_tiles` per rank).

## Implementation notes (ttnn-implementer)

- `l1_budget` = `ttnn.get_max_worker_l1_unreserved_size()` − 96 KB margin (1,531,904 B unreserved on BH p150; the margin
  was 64 KB until Refinement 4, see below).
- Every CB push is ring-aligned: `cb_x_resident` is pushed/popped by the NOMINAL `block_token_tiles · core_k_tiles_max`
  pages (uneven ranks and the ragged last block write fewer tiles into the same slot); `cb_combined` / `cb_gathered` are
  pushed at the nominal `2 · block_token_tiles` per rank, so the root's read pointer and every non-root's landing are the
  CB base on every block. `cb_y_out` is drained in `y_chunk_tiles` windows clipped at the ring end (wrap-aware writer).
- The bias tile is staged in the first, not-yet-pushed `cb_x_resident` slot (disjoint lifetime: before block 0's X read).
- Data-movement budget unchanged: X once, W once per group (G_t×, R2 still deferred), outputs once. Measured: the reader
  is DRAM-bound (~110 MB in ~300 µs at T=640, C=7168 fp32), so the W re-read (≈ 1/3 of the bytes) is the largest
  remaining lever.

## Verifier notes (Phase 0 verification)

- **Single source of CB sizes.** Every row above is now produced by one host function,
  `mhc_pre_program_descriptor._cb_table(...)`. Both the L1 selection function (`_l1_bytes`, which is affine
  in `block_token_tiles`) and the ProgramDescriptor's CB list read it. Before this, `_l1_bytes` restated the
  page counts by hand, so a knob turn could change the CBs without changing the fit test.
- **Currency check.** The table matches the code row by row. The measured `device_l1_peak_bytes` on the golden
  run is 1,255,424 B at C=7168 fp32 (kmax=84, G=11, bt=1, depth 2), which equals the closed form above.
- **bf16 W** (now in SUPPORTED): `cb_weight` pages are 2048 B. At C=7168 that is 172 KB instead of 344 KB.

## Refinement 1 (bf16 streams, W hi/lo split)

- `cb_weight_split` aliases `cb_weight`, so the footprint formula is unchanged. For bf16 X, `xT` = 2048, which
  halves the `cb_x_resident` term. The data-movement budget is also unchanged: the split is on-chip only.

## Refinement 2 (fp32 streams, exact-grid projection)

- New fp32-X-only CBs (rows above): `cb_x_fp32` (alias, 0 B), `cb_x_pieces` (2 · 3 · 8 · 1 · 2 KB = 96 KB at
  bt = 1), `cb_mix_run` 4 KB, `cb_max_lanes` / `cb_max_scalar` / `cb_grid` 4 KB each, `cb_max_scaler` 2 KB.
  `cb_weight_split` is reused unchanged for the W grid pieces [W0, W − W0] (same 2 bf16 pages per fp32 W tile).
- Footprint at C = 7168 fp32 (kmax 84, bt 1, depth 2): 1,372,160 B of the 1,466,368 B budget. The x_block
  depth-2 prefetch survives on every TARGET shape.
- The `_l1_bytes` affine solve over-estimates the `min(bt, 4)` terms for bt > 4 (conservative).
- Data-movement budget unchanged: the pieces, grids and running partial are on-chip only.

## Refinement 3 (W column broadcast, regime R2)

- No CB change: `cb_weight` is filled by the column mcast instead of a DRAM read on receivers (same address on
  every core, write-once, pushed in the same `W_CHUNK_TILES` chunks). One more semaphore (`SEM_W_READY`).
- Data-movement budget, when R2 applies (`group_h == 1`, ≥ 2 full group rows active — e.g. T=640, C=7168):
  **W crosses DRAM once** (was G_t = 10×): fp32 W 36.7 MB → 3.67 MB. Added NoC: 3.67 MB × (G_t − 1) column
  mcast deliveries, one-shot at kernel start. Totals at T=640, C=7168: fp32 X ≈ 96 MB (minimum), bf16 X ≈ 50 MB
  (minimum). Shapes with `group_h > 1` (decode, small Mt) keep the per-group W read.

## Refinement 4 (T=640, C=1792 bf16 perf focus)

- Removed (earlier R4 step): `cb_coef_out`, `cb_logits_coef`, `cb_out_stage`. The row-major ↔ coefficient-major
  transforms moved into DEST (transpose_tile / transpose_dest + SFPU subvector transpose), so the writer no longer
  scatters S or stages post / comb tiles. `cb_coef_in` and `cb_comb_coef` grew to `2 · bt`.
- Added: `cb_coef_keep` (1 fp32 tile, the fused owned block), `cb_mix_run` for bf16 X (1 fp32 tile, the streamed
  projection's running mix). Grew: `cb_sq_acc` to `bt · X_STREAM_CHUNKS` for bf16 X (streamed Σx² partials).
  Net at bt = 1, bf16 X vs Refinement 3: added 4 + 4 + 12 KB, `cb_coef_in` / `cb_comb_coef` +4 KB each, removed
  4 + 4 + 8 KB → **+12 KB**.
- `L1_SAFETY_MARGIN` 64 → 96 KB. The CB base sits ~70.7 KB above the allocator's unreserved base (the kernel
  config ring includes the kernel binaries, and the compute binary grew). 1×1×2048×20480 bf16 had its
  footprint 2 KB under the old budget and overflowed L1 by 3.2 KB.
- Owner C discount: rank 0 holds `even − OWNER_C_DISCOUNT` stream columns when it owns every Sinkhorn row, so
  `core_k_tiles_max` = n · max(per-rank C tiles). That is +1 C tile on the other ranks at C=1792: 13 vs 12.
  The fit reads the same split (`_c_split`), so the CB sizes stay single-source.
- Data-movement budget unchanged: X once, W once per physical column (column all-gather), outputs once. The
  NoC-flip knob (`READER_NOC_FLIP_ROWS`, default 0) changes only the NoC, not the bytes.

## Refinement 5 (T=1280, C=4096 bf16 perf focus: block × depth co-tune + NoC placement)

- Added `cb_w_share_landed` (one 32 B token, reader → writer). Nothing else changed in the inventory:
  `block_token_tiles` stays 1 and `x_block_depth` stays 2 (co-tune measured below), so the bf16-X L1 headroom that
  bf16 frees (≈ 200–350 KB at C = 4096–7168) is left unspent — a coarser block or a deeper prefetch did not pay.
- With the W column all-gather, the reader (not the writer) DRAM-reads this core's W share into `cb_weight` at the
  writer's slots (transaction id 15, issued before the X burst) and hands it over by token. The writer still owns
  the CB (it reserves / publishes it; the split, multicast and bias are unchanged).
- Data-movement budget unchanged: X once, W once per physical column (column all-gather), outputs once. Only the NoC
  that carries each byte moved (reader NoC flip for the top `round(0.4 · grid_y)` rows; the W share now rides the
  reader's NoC).
