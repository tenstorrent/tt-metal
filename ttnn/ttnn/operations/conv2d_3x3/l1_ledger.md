# L1 Ledger: conv2d_3x3 (regimes R0 interleaved-slab, R2 HEIGHT-sharded, R3 TILE-input untilize, R4 BLOCK K split)

Schema and audits: `.claude/references/l1-footprint-discipline.md`. Block axes (from `op_design.md`
Blocking Model): **M** (`block_m_tiles` = bm), **N** (`block_n_tiles` = bn), **K** (`block_k_tiles` = bk;
its spatial components kh, kw and its channel component c). Tile bytes: `T_bf16 = 2048`,
`T_fp32 = 4096`, `T_in0` (im2col tile format = activation dtype; bf16 in Phase 0), `T_w`
(weight_dtype), `T_out` (output dtype = input dtype). `sbh` = `out_subblock_h` (derived, divides bm).

Inventory decisions made before any solve (Rule 1 / Rule 3):

| Candidate buffer | Decision | Why |
|------------------|----------|-----|
| zero-source buffer for padding taps | **removed** | Zero reads come from the hardware `MEM_ZEROS_BASE` region (512 B) — no CB. |
| full-image or contiguous-footprint staging | **replaced** by the 3-band slab | Bands are stored at stride `min(d*W, band_len_px)`: identical to the contiguous footprint at d=1, 3x smaller at large d (32x64, d=18, bm=1: 204 px vs 2372 px). |
| tilize output in place over `cb_im2col_rm` | **rejected** | `tilize` requires distinct input/output buffers (`tilize_helpers.hpp:101`), and the two have different page formats/semantics (RM bytes vs tiles). |
| separate no-bias copy pass (partials -> out) | **removed** | no-bias packs the last K-block straight into `cb_output_tiles` (`LastBlockTarget::Out`): pack-into-destination. |
| bias-add output in place over `cb_matmul_partials` | **rejected** | Different page formats (Float32 partials vs output dtype) and the TileRowMajor interm must be its own region (`matmul_block_helpers.hpp:247-252`). |
| `cb_im2col_tiles` aliased with `cb_output_tiles` | **rejected** | Concurrent across M-blocks: the writer drains block j while compute tilizes block j+1 (explicit pipelining decision). The saving (`bm*bk*T_in0`) is not needed by any Phase 0 shape (max total 1256 KB, below). |

The remaining L1 solve (`select_blocking`) exists for one reason that inventory cannot remove: the weight
operand alone exceeds L1 (K = 18432 x N = 256 bf16 = 9.4 MB). It shrinks `bk` first (costs no bytes), then
`bm` (re-streams weights and halo), never `bn` — `op_design.md` Mechanism caps #1.

## Ledger

| CB | Capacity (pages) | Live set | Axis accounting | Page format | Producer | Consumer | Lifetime | Shares with / why not |
|----|------------------|----------|-----------------|-------------|----------|----------|----------|-----------------------|
| `cb_halo_slab` (ROW_MAJOR: page = `c_in_p*2` B, one pixel. TILE: page = `T_bf16`, 32 slab pixels of 32 channels) | ROW_MAJOR: `2*band_stride_px + band_len_px` pixels. TILE: `slab_tile_rows * Ct` pages (`Ct = c_in_p/32`) | = capacity: any row of any K-block may read any pixel of its band, so the whole footprint is simultaneously live | `{M: spans -> band_len_px, affine in bm (dilated by s, d, W; face-direct reaches tap 3, one pixel further); TILE: bands rounded out to 32-pixel input tile-rows, a band starting inside the previous band's tile-rows shares them; N: — (activation does not vary with N); K: spans kh (3 bands), kw (2d+1 margin inside band_len_px) and c (all c_in_p channels per pixel)}` | bfloat16 RM | ROW_MAJOR: reader; HEIGHT RM (`SLAB_SPLIT_NOCS`): reader (own and lower-index shards' runs, NoC0) + writer (higher-index shards' runs, NoC1), disjoint pixels, handed over by the core-local `slab_fwd_done` / `slab_free` semaphores. TILE: compute (untilize, `slab_tile_rows` tile-rows per M-block) | reader, and the writer under the split face-direct gather (ROW_MAJOR: scratch, no FIFO handshake. TILE: waits per band, pops all after the last K-block, so every M-block starts at the CB base) | one M-block (all its K-blocks) | none — concurrent with every other CB of the block (reader gathers from it while compute consumes `cb_im2col_rm`/`cb_im2col_tiles` and accumulates partials). TILE: `slab_tile_rows` is a whole number of untilize batches (dummy tail rows are pushed unread). |
| `cb_input_tiles` (TILE only; page = `T_in` = input tile) | `INPUT_TILE_DEPTH_BATCHES * ub * Ct` = `2*ub*Ct` (`ub` = `untilize_batch_rows`) | `ub * Ct` per party | `{M: streams -> ub footprint tile-rows (depth 2); N: —; K: spans c (one whole tile-row of channels: the untilize block)}` | input dtype (bf16 / bf8_b) | reader (one trid fence per batch) | compute (untilize, one tile-row of `Ct` tiles per block) | one batch | none — 2x gap is double buffering (reader reads batch b+1 while compute untilizes b). `ub` defaults to the tile-rows one band can touch (about one fence per band) and is the FIRST extent the L1 solve shrinks (to 1): a smaller batch costs a fence per tile-row, a smaller `bk` a whole K-block. Measured 8x16x2048 TILE bf16: `ub` 3 -> 1 lets `bk` stay at 16 instead of 3, 526 -> 463 us. |
| `cb_im2col_rm` (page = `T_bf16`, RM content: 32 rows x 32 cols; tap-major: `bk / Ct` blocks of 32 pixels x `c_in_p`, same bytes) | `im2col_rm_depth_rows * bk` = `2*bk` | `bk` per party (one tile-row being filled, one being tilized) | `{M: streams -> 1 tile-row window (depth 2); N: —; K: spans -> bk}` | bfloat16 (RM bytes in tile-sized pages) | reader | compute (tilize) | one tile-row | none — gap to live set (2x) is double buffering (reader fills t+1 while compute tilizes t); concurrent with `cb_im2col_tiles` (tilize reads one while writing the other). **Not allocated on face-direct shapes** (the gather writes tiles directly; removed by inventory, not solved). |
| `cb_im2col_tiles` (page = `T_in0`) | `bm * bk` | `bm * bk` (matmul waits the whole in0 block, `matmul_block_helpers.inl:265`) | `{M: spans -> bm; N: —; K: spans -> bk}` | in0 = activation dtype (bfloat16 in Phase 0). An operand, not an accumulator: values are exact in this format and srcB unpacks at most tf32, so audit 2's under-case does not apply. | compute (tilize, in PreKBlockFn); face-direct: the reader (reserve once per K-block, push per row group of `in0_stream_rows`; the writer writes the odd row groups into the reserved block, so capacity must be exactly one K-block, starting at the CB base every time) | compute (matmul; streamed: one call per row group) | one K-block | none — concurrent with `cb_matmul_partials` (matmul reads it while accumulating); alias with `cb_output_tiles` rejected above (cross-block pipelining). Face-direct stem.conv1: `bk = tensor_k_tiles` = 3 makes it 228 KB, the whole K walk in one block (P1). |
| `cb_weights` (page = `T_w`) | `weight_depth * bk * bn` (`weight_depth` = 2; 1 only if `num_k_blocks * num_m_blocks_this_core == 1`) | `bk * bn` per party | `{M: — (re-streamed per M-block; R5 would make it span K whole); N: spans -> bn; K: spans -> bk, streams K-blocks}` | weight_dtype (bf16 Phase 0) | writer (R1: the sender's writer reads from DRAM, every other participant's writer receives the multicast into its own slot) | compute (matmul) | one K-block per slot | none — 2x gap is double buffering (writer streams K-block k+1 during matmul k); concurrent with all compute CBs. |
| `cb_bias` (page = `T_w`; allocated only when `has_bias`) | `ns` (= `bn`, except under R4: the core's output N-slice) | `ns` (fronted for the whole kernel) | `{M: — (shared by every M-block); N: spans -> ns; K: —}` | weight_dtype | writer | compute (bias add) | whole kernel | none — live for the whole kernel. |
| `cb_matmul_partials` (page = `T_fp32`) | exactly `bm * bn` (mechanism cap #2: L1-acc lands on the same addresses every K-block) | `bm * bn` (accumulated in place across all K-blocks) | `{M: spans -> bm; N: spans -> bn; K: streams (reduced over, never materialized per K-block)}` | Float32 — `fp32_dest_acc_en=True`, so the accumulation keeps fp32 across K-blocks (audit 2: correct width) | compute (matmul pack; R4: one caller-owned reserve/push per M-block) | compute (matmul drain/reload; bias add). R4: the writer (scatter, one wait/pop per M-block; the reader reads its higher-column slices in place between the writer's send_go and its own send_done) | one M-block | none — TileRowMajor interm must be its own region; Float32 vs output format; concurrent with `cb_output_tiles` during the bias add. |
| `cb_output_tiles` (page = `T_out`) | `out_depth_row_groups * sbh * bn` = `2*sbh*bn` | `sbh * bn` per party | `{M: streams -> one row group of sbh tile-rows (depth 2); N: spans -> bn; K: —}` | output dtype = input dtype (the output tensor's format by contract; not an intermediate) | compute | writer | one row group | none — 2x gap is double buffering (writer drains row group g while compute packs g+1). **R2 (HEIGHT): backed on the output shard** (`cb_descriptor_from_sharded_tensor`, capacity = `core_m_tiles * bn` pages = the shard, not CB-region bytes); compute packs every M-block of the core into it in order and nothing pops it, so `bm` must divide `core_m_tiles` (pushes = capacity exactly). |
| `cb_combine_slots` (R4, `k_cols > 1`; page = `T_fp32`) | `k_cols * bm * ns` (`ns` = `out_n_tiles` = `ceil(tensor_n_tiles / k_cols)`) | = capacity: the sum needs every row peer's slice of every output tile at once | `{M: spans -> bm; N: spans -> ns (the core's output columns); K: spans the row's k_cols channel slices (one slot each), reduced here}` | Float32 — partial sums keep the matmul's fp32 accumulation across cores (audit 2) | writer (reserve + push of the whole CB; the bytes are NoC-written by every row peer's writer/reader) | compute (`reduce` AccumulateViaAdd + Skip, BulkWaitBulkPop) | one M-block | none — cannot alias `cb_matmul_partials`: the writer is still scattering this core's own partials when peers' slices land. Reserved/pushed whole every round, so its base address is the same on every core. |
| `cb_combined` (R4 with bias; page = `T_fp32`) | `bm * ns` | `bm * ns` (the reduce and the bias add are sequential helpers on the same TRISCs) | `{M: spans -> bm; N: spans -> ns; K: —}` | Float32 (an fp32 sum feeding the bias add) | compute (reduce) | compute (bias add) | one M-block | none — no bias: the reduce packs straight into the shard-backed `cb_output_tiles` and this CB is not allocated (pack into destination). With bias it cannot be the output shard: Float32 sum vs output dtype. |

## Symbol table

| Symbol | Meaning | Bound | Predicate establishing it |
|--------|---------|-------|---------------------------|
| `bm` = `block_m_tiles` | M extent of a block | `1 <= bm <= core_m_tiles = ceil(tensor_m_tiles / grid_cores)`; Phase 0 INPUTS: 1; LOOSE: <= 38 | `select_blocking` L1 solve |
| `bk` = `block_k_tiles` | K extent of a block | divides `tensor_kkh_tiles <= 192` (`C_in_p <= 2048`) | mechanism cap #3 + L1 solve |
| `bn` = `block_n_tiles` | N extent | `= tensor_n_tiles <= 16` (`C_out <= 512`, TARGET) | fixed by the traffic ranking |
| `sbh` | `out_subblock_h` | `1..4`, divides `bm`, `sbh * out_subblock_w <= 4` | DEST cap (mechanism cap #7) |
| `c_in_p` | per-pixel channel width | `<= 2048` (BLOCK slices: `<= 192`) | `C_in_p` rule (op_design Parameters) |
| `band_len_px` | pixels in one kh band | `<= min(H*W, (bm*32 - 1)*s + dh_max*s*(W - W_out) + 2d + 1)`, `dh_max = floor((bm*32 + W_out - 2) / W_out)`; host computes the exact max over the core's blocks | follows `bm`, s, d, W |
| `band_stride_px` | slab offset between bands | `= min(d*W, band_len_px)` | d, W, `band_len_px` |
| `slab_tile_rows` | TILE slab capacity in tile-rows | max over the core's M-blocks of the reader's band plan, rounded up to a multiple of `ub` | `_tile_slab_rows` (host mirror of the reader plan) |
| `ub` = `untilize_batch_rows` | TILE footprint tile-rows per reader push / compute untilize call | `1 <= ub <= min(raw slab tile-rows, ceil(band_len_px/32) + 1)` | L1 solve (shrunk first) |
| `im2col_rm_depth_rows`, `weight_depth`, `out_depth_row_groups` | buffer depths | 2, {1, 2}, 2 | Blocking Model depth table |
| `L1_CB_BUDGET_BYTES` | CB budget per core | `get_max_worker_l1_unreserved_size() - 128 KB headroom - RESIDENT_SHARD_SETS * (input shard + output shard)`; `RESIDENT_SHARD_SETS = 2` in R2 (this call's pair plus the previous call's, which a chained model and the golden harness's cache-hit dispatch keep alive; every enqueue re-checks the CB region against the lowest live L1 buffer), 0 shard bytes in R0/R3 | `l1_cb_budget_bytes`, `select_blocking` |

## Total per-core footprint

```
F(bm, bk) = (2*band_stride_px(bm) + band_len_px(bm)) * c_in_p * 2       # slab (RM)   — scales with bm (affine), c_in_p, d, s, W
          | slab_tile_rows(bm) * Ct * T_bf16 + 2 * ub * Ct * T_in          # slab + input window (TILE)
          + im2col_rm_depth_rows * bk * T_bf16                           # im2col RM   — scales with bk
          + bm * bk * T_in0                                              # im2col tile — scales with bm x bk
          + weight_depth * bk * bn * T_w                                 # weights     — scales with bk x bn
          + has_bias * bn * T_w                                          # bias        — scales with bn
          + bm * bn * T_fp32                                             # partials    — scales with bm x bn
          + out_depth_row_groups * sbh * bn * T_out                      # output      — scales with bn (sbh <= 4); 0 in R2/R4 (shard-backed)
          + k_split * (k_cols * bm * ns + has_bias * bm * ns) * T_fp32   # R4 combine slots + summed block — scale with bm x ns x k_cols
          <= L1_CB_BUDGET_BYTES
```
(R4: the bias term is `ns * T_w`.)

Worked values, R4 BLOCK LOOSE layers (bf8_b, Blackhole, pinned grids; budget after 2 x (input + output shard)).
Every layer keeps one M-block per core (`bm = core_m_tiles`):

| Layer | grid (cols x rows) | k_cols, ns | bm | bk / tensor_kkh_tiles | footprint / budget |
|-------|--------------------|------------|----|-----------------------|--------------------|
| res4.0.conv2 (s 2) | 8 x 8 | 8, 1 | 8 | 3 / 3 | 817 / 1282 KB |
| res4.{1..5}.conv2 | 8 x 8 | 8, 1 | 8 | 3 / 3 | 694 / 1333 KB |
| res5.0.conv2 (d 2) | 8 x 10 | 8, 2 | 7 | 6 / 6 | 1307 / 1307 KB |
| res5.2.conv2 (d 8) | 8 x 10 | 8, 2 | 7 | 3 / 6 | 1236 / 1307 KB |
| aspp.convs.1 (C_in 2048, d 6) | 11 x 8 | 11, 1 | 8 | 6 / 18 | 1241 / 1248 KB |
| aspp.convs.3 (C_in 1024, d 18, no bias) | 11 x 8 | 11, 1 | 8 | 9 / 9 | 1075 / 1299 KB |

Worked values, R2 HEIGHT LOOSE layers (bf8_b weights, with bias; 11x10 Blackhole, pinned shard grids; budget after
subtracting 2 x the core's input + output shard):

| Layer | cores | core_m_tiles | bm | bk / tensor_kkh_tiles | 2 x shards | footprint / budget |
|-------|-------|--------------|----|-----------------------|------------|--------------------|
| stem.conv1 (RM bf16, C_in 8, s 2) | 108 of 110 | 38 | 38 | 3 / 1 (face-direct, whole K) | 452 KB | 658 / 914 KB (slab 118 + in0 228 + partials 304 + weights/bias 9; no `cb_im2col_rm`). Before Refinement 6: bk 1, 508 KB. |
| stem.conv2 | 108 of 110 | 38 | 38 | 6 / 6 | 323 KB | 1038 / 1044 KB |
| stem.conv3 | 108 of 110 | 38 | 19 | 6 / 6 | 484 KB | 845 / 882 KB |
| res2.x.conv2 | 103 | 10 | 10 | 6 / 6 | 85 KB | 390 / 1282 KB |
| semantic decoder.res3.fuse_conv.0 | 86 | 3 | 3 | 30 / 30 | 114 KB | 1236 / 1252 KB |
| semantic decoder.res2.fuse_conv.0 | 103 | 10 | 5 | 9 / 27 | 361 KB | 917 / 1005 KB (bm 10 at bk 9 needs 1136 KB; L1 left after both shard pairs: 1134 KB) |
| semantic res2.fuse_conv.1 | 103 | 10 | 5 | 12 / 24 | 340 KB | 939 / 1027 KB |

Worked values, Phase 0 INPUTS (bf16, with bias; every core runs one M-block, `bm = 1`). Budget on Blackhole p150
(11x10 grid): `get_max_worker_l1_unreserved_size()` 1530976 B - 128 KB headroom = 1399904 B (1367 KB). Verified
against `select_blocking` on device (verification_report.md); bf8_b weights shrink every row by the `T_w` terms
(deep-K bf8_b solves to `bk = 32`, 1072 KB):

| Shape | cores | bk / tensor_kkh_tiles | slab px | weights | total |
|-------|-------|-----------------------|---------|---------|-------|
| 16x34x32 -> 32, p0 | 14 | 3 / 3 | 102 | 12 KB | 46 KB |
| 32x64x128 -> 128, p1 | 64 | 12 / 12 | 102 | 192 KB | 329 KB |
| 32x64x128 -> 128, p18 d18 | 64 | 12 / 12 | 204 | 192 KB | 355 KB |
| 8x16x2048 -> 256, p1 | 4 | 24 / 192 (below tap floor 64: one kh block of weights is 2 x 1.5 MB) | 66 | 768 KB | 1256 KB |

## Data-movement budget (chosen split: M, with the R1 weight multicast since Refinement 3)

| Tensor | DRAM crossings | Why that many | Cross-core traffic added |
|--------|----------------|---------------|--------------------------|
| input | ROW_MAJOR: `sum over M-blocks of slab_px / (H*W)`: ~1 + halo. 32x64 at `bm = 1`: 102 px per 32-px block -> 3.2x (bands disjoint, block narrower than an image row); one core: 1.06x. TILE: the same bands rounded out to 32-pixel tile-rows: 32x64 at `bm = 1`: ~7 tile-rows (224 px) per 32-px block -> ~7x; 128x256x64 at `bm = 10`: 1152 vs 834 px per block (1.38x RM) | Each M-block stages its own 3-band footprint once and gathers all 9 taps from L1; neighbouring blocks re-read the halo rows. TILE pays tile-row rounding but reads 2 KB (bf16) / 1088 B (bf8_b) pages instead of one `c_in_p*2` B page per pixel: 128x256x64 measured 227 us (RM) -> 111 / 102 us (TILE bf16 / bf8_b), because the RM fill is DRAM-transaction-bound | 0 |
| weights | R1: `max_m_blocks_per_core` (+ `num_m_blocks` of a ragged last core, which streams its own). R0 (channel absent): `num_m_groups * num_m_blocks_this_core` | R1: the sender (work core 0) reads each `bk*bn` K-block once per M-block pass and multicasts it into every participant's `cb_weights` slot; no residency across M-blocks, so `bm < core_m_tiles` re-streams the weight once per extra M-block, now as multicast instead of `num_m_groups` DRAM copies | R1: `(participants - 1) * K_p * C_out_p * elem` per M-block pass of multicast payload (one sender-serialized transfer per K-block), plus one ack per receiver and one data-ready flag per round |
| bias | `num_m_groups` x `C_out_p` row (one tile row) | Read once per core | 0 |
| output | 1 | Each output tile is written once by its owning core | 0 |

R4 (BLOCK_SHARDED, `k_cols > 1`): the input is read like R2 but per channel slice (each core reads only its column's
slice of its bands, one read per owner-shard run); weights cross DRAM once per grid column per M-block pass (since
Refinement 5 the first participating row of each column reads that column's K-slice, `K_p / k_cols` rows x all of
C_out, and multicasts it down the column into the same `cb_weights` slot; a ragged last row with fewer M-blocks streams
its own). res4.x: 5 MB -> 625 KB of DRAM reads, 4.4 MB of column multicast; the 64-core DRAM stream had become the
bound (21 us) once the gather stopped hiding it. The output is packed in place. Cross-core traffic added: each core sends `bm * tensor_n_tiles` fp32 tiles per M-block (its own
slice included, as a self-write) and receives `k_cols * bm * ns`. res4.x: 256 KB out + 256 KB in per core per call,
which bounds the round on one NoC interface; split over NoC0 (reader, higher columns) and NoC1 (writer, own and
lower columns) it measured 86.8 -> 78.7 us (scatter writes stubbed: 66.1 us).

R2 (HEIGHT_SHARDED) differs only in the activation and output rows: there is no DRAM crossing for either.
ROW_MAJOR HEIGHT (Refinement 6): the slab's bytes are unchanged, but the runs owned by higher-index shards cross
on NoC1 (the writer) instead of NoC0, so no read response wraps the ring; the face-direct gather is local L1 only
(4 x 512 B reads per in0 tile instead of 32 x 48 B row reads per tile-row per kh).
The input crosses the NoC once plus halo (each core's three kh bands, read as page runs from whichever cores own
them, one read per shard a band touches; the own-shard part is a self-aimed L1 copy into the slab, the layout
transform into kh bands contiguous across shard edges). The output never moves: compute packs straight into the
output shard. Weights are as R0, multiplied by `num_m_blocks_this_core` where the solve shrinks `bm`
(semantic decoder.res2.fuse_conv.0: bm 5 of 10 doubled its 705 KB per-core weight stream, 145 MB of DRAM reads; with R1 it is 1.4 MB of DRAM reads plus the multicast).

Totals, INPUTS `32x64x128 -> 128, p1` (64 cores): input 1.6 MB (tensor 512 KB), weights 288 KB DRAM + 18.1 MB multicast (R0: 18.4 MB DRAM), output 512 KB. INPUTS `8x16x2048 -> 256` (4 cores): input 1.1 MB (tensor 512 KB), weights 9.4 MB DRAM + 28.3 MB multicast (R0: 37.7 MB DRAM), output 64 KB.

R1 adds no CB and changes no capacity: the multicast lands in `cb_weights` (quantum `bk*bn`, depth `weight_depth`) at the same address on every participant because every CB has the same core range and every participant pushes the same quanta in the same order. Its only L1 cost is the channel's two semaphores per participating core.

> Cheapest-traffic split considered: **M split + weight broadcast (R1)** — weights `num_m_groups` -> 1 DRAM crossing (32x64x128: 18.4 MB -> 288 KB DRAM; adds 18.1 MB of multicast payload on the NoC, one sender-serialized transfer per K-block); input and output unchanged. Implemented: **M split with per-core weight stream (R0)**. Deferred because the Phase 0 weight path is prompt-directed (each core reads its own kh blocks; multicast is a later perf refinement) and R0's `cb_weights` receive side — K-block quantum `bk*bn`, depth 2 — is exactly the multicast receiver, so R1 layers onto R0 without changing any CB, block extent or compute phase.

## Perf 1 additions

| CB | Index | Size | When |
|----|-------|------|------|
| `cb_slice_r` | 10 | `NSLICE_SLICE_DEPTH * bm * out_n` fp32 tiles | R4 N-slice-outer (reader's slice sends) |
| `cb_zero_src` | 11 | `32 * C_in_p * 2` B | tap-major gather |
| `cb_im2col_tiles` | 2 | `bm * K` (was `bm * bk`) | R4 N-slice-outer (resident in0) |
| `cb_weights` | 3 | `NSLICE_WEIGHT_DEPTH * K * out_n` (was `weight_depth * bk * bn`) | R4 N-slice-outer |
| `cb_matmul_partials` | 5 | `NSLICE_SLICE_DEPTH * bm * out_n` (was `bm * bn`) | R4 N-slice-outer (writer's slice sends) |
| `cb_im2col_rm` | 1 | depth `min(8, bm * num_k_blocks)` units (was 2) | split row gather |

`_footprint` counts all of them. Measured footprints, in bytes:
- res5.1: 1,266,368 (K-blocked, bk 3) -> 1,219,840 (N-slice-outer, bk 6);
- res4.0: 836,928 -> 781,888;
- aspp.convs.1/2: unchanged at 1,257,792 (the N-slice-outer in0 does not fit).

The split row gather adds 6 core-local semaphores (`row_go`, `row_done`, `row_limit`, `row_assigned`, 2 mailbox words).
