# L1 Ledger: conv2d_3x3 (Phase 0, regime R0 interleaved-slab)

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
| `cb_halo_slab` (page = `c_in_p*2` B, one pixel) | `2*band_stride_px + band_len_px` | = capacity: any row of any K-block may read any pixel of its band, so the whole footprint is simultaneously live | `{M: spans -> band_len_px, affine in bm (dilated by s, d, W); N: — (activation does not vary with N); K: spans kh (3 bands at band_stride_px), kw (2d+1 margin inside band_len_px) and c (all c_in_p channels per pixel)}` | bfloat16 RM | reader | reader (private scratch; no FIFO handshake in R0) | one M-block (all its K-blocks) | none — concurrent with every other CB of the block (reader gathers from it while compute consumes `cb_im2col_rm`/`cb_im2col_tiles` and accumulates partials). In R3 it becomes a compute->reader CB; same capacity. |
| `cb_im2col_rm` (page = `T_bf16`, RM content: 32 rows x 32 cols) | `im2col_rm_depth_rows * bk` = `2*bk` | `bk` per party (one tile-row being filled, one being tilized) | `{M: streams -> 1 tile-row window (depth 2); N: —; K: spans -> bk}` | bfloat16 (RM bytes in tile-sized pages) | reader | compute (tilize) | one tile-row | none — gap to live set (2x) is double buffering (reader fills t+1 while compute tilizes t); concurrent with `cb_im2col_tiles` (tilize reads one while writing the other). |
| `cb_im2col_tiles` (page = `T_in0`) | `bm * bk` | `bm * bk` (matmul waits the whole in0 block, `matmul_block_helpers.inl:265`) | `{M: spans -> bm; N: —; K: spans -> bk}` | in0 = activation dtype (bfloat16 in Phase 0). An operand, not an accumulator: values are exact in this format and srcB unpacks at most tf32, so audit 2's under-case does not apply. | compute (tilize, in PreKBlockFn) | compute (matmul) | one K-block | none — concurrent with `cb_matmul_partials` (matmul reads it while accumulating); alias with `cb_output_tiles` rejected above (cross-block pipelining). |
| `cb_weights` (page = `T_w`) | `weight_depth * bk * bn` (`weight_depth` = 2; 1 only if `num_k_blocks * num_m_blocks_this_core == 1`) | `bk * bn` per party | `{M: — (re-streamed per M-block; R5 would make it span K whole); N: spans -> bn; K: spans -> bk, streams K-blocks}` | weight_dtype (bf16 Phase 0) | writer | compute (matmul) | one K-block per slot | none — 2x gap is double buffering (writer streams K-block k+1 during matmul k); concurrent with all compute CBs. |
| `cb_bias` (page = `T_w`; allocated only when `has_bias`) | `bn` | `bn` (fronted for the whole kernel) | `{M: — (shared by every M-block); N: spans -> bn; K: —}` | weight_dtype | writer | compute (bias add) | whole kernel | none — live for the whole kernel. |
| `cb_matmul_partials` (page = `T_fp32`) | exactly `bm * bn` (mechanism cap #2: L1-acc lands on the same addresses every K-block) | `bm * bn` (accumulated in place across all K-blocks) | `{M: spans -> bm; N: spans -> bn; K: streams (reduced over, never materialized per K-block)}` | Float32 — `fp32_dest_acc_en=True`, so the accumulation keeps fp32 across K-blocks (audit 2: correct width) | compute (matmul pack) | compute (matmul drain/reload; bias add) | one M-block | none — TileRowMajor interm must be its own region; Float32 vs output format; concurrent with `cb_output_tiles` during the bias add. |
| `cb_output_tiles` (page = `T_out`) | `out_depth_row_groups * sbh * bn` = `2*sbh*bn` | `sbh * bn` per party | `{M: streams -> one row group of sbh tile-rows (depth 2); N: spans -> bn; K: —}` | output dtype = input dtype (the output tensor's format by contract; not an intermediate) | compute | writer | one row group | none — 2x gap is double buffering (writer drains row group g while compute packs g+1). In R2-R4 this CB is backed on the output shard (capacity = the shard). |

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
| `im2col_rm_depth_rows`, `weight_depth`, `out_depth_row_groups` | buffer depths | 2, {1, 2}, 2 | Blocking Model depth table |
| `L1_CB_BUDGET_BYTES` | CB budget per core | per-arch host constant minus L1-resident tensor bytes on the core (0 in R0) | `select_blocking` |

## Total per-core footprint

```
F(bm, bk) = (2*band_stride_px(bm) + band_len_px(bm)) * c_in_p * 2       # slab        — scales with bm (affine), c_in_p, d, s, W
          + im2col_rm_depth_rows * bk * T_bf16                           # im2col RM   — scales with bk
          + bm * bk * T_in0                                              # im2col tile — scales with bm x bk
          + weight_depth * bk * bn * T_w                                 # weights     — scales with bk x bn
          + has_bias * bn * T_w                                          # bias        — scales with bn
          + bm * bn * T_fp32                                             # partials    — scales with bm x bn
          + out_depth_row_groups * sbh * bn * T_out                      # output      — scales with bn (sbh <= 4)
          <= L1_CB_BUDGET_BYTES
```

Worked values, Phase 0 INPUTS (bf16, with bias, 1300 KB budget assumed; every core runs one M-block, `bm = 1`):

| Shape | cores | bk / tensor_kkh_tiles | slab px | weights | total |
|-------|-------|-----------------------|---------|---------|-------|
| 16x34x32 -> 32, p0 | 14 | 3 / 3 | 102 | 12 KB | 46 KB |
| 32x64x128 -> 128, p1 | 64 | 12 / 12 | 102 | 192 KB | 329 KB |
| 32x64x128 -> 128, p18 d18 (refinement) | 64 | 12 / 12 | 204 | 192 KB | 355 KB |
| 8x16x2048 -> 256, p1 | 4 | 24 / 192 (below tap floor 64: one kh block of weights is 2 x 1.5 MB) | 66 | 768 KB | 1256 KB |

## Data-movement budget (chosen split: M only, R0)

| Tensor | DRAM crossings | Why that many | Cross-core traffic added |
|--------|----------------|---------------|--------------------------|
| input | `sum over M-blocks of slab_px / (H*W)`: ~1 + halo. 32x64 at `bm = 1`: 102 px per 32-px block -> 3.2x (bands disjoint, block narrower than an image row); one core: 1.06x | Each M-block stages its own 3-band footprint once and gathers all 9 taps from L1; neighbouring blocks re-read the halo rows | 0 |
| weights | `num_m_groups * num_m_blocks_this_core` (each core streams all of `K_p x C_out_p` per M-block) | Per-core weight stream (prompt-directed Phase 0); no residency across cores | 0 |
| bias | `num_m_groups` x `C_out_p` row (one tile row) | Read once per core | 0 |
| output | 1 | Each output tile is written once by its owning core | 0 |

Totals, INPUTS `32x64x128 -> 128, p1` (64 cores): input 1.6 MB (tensor 512 KB), weights 18.4 MB, output 512 KB, cross-core 0. INPUTS `8x16x2048 -> 256`: input 1.1 MB (tensor 512 KB), weights 37.7 MB, output 64 KB.

> Cheapest-traffic split considered: **M split + weight broadcast (R1)** — weights `num_m_groups` -> 1 DRAM crossing (32x64x128: 18.4 MB -> 288 KB DRAM; adds 18.1 MB of multicast payload on the NoC, one sender-serialized transfer per K-block); input and output unchanged. Implemented: **M split with per-core weight stream (R0)**. Deferred because the Phase 0 weight path is prompt-directed (each core reads its own kh blocks; multicast is a later perf refinement) and R0's `cb_weights` receive side — K-block quantum `bk*bn`, depth 2 — is exactly the multicast receiver, so R1 layers onto R0 without changing any CB, block extent or compute phase.
