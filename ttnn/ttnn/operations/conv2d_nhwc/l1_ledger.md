# L1 Ledger: conv2d_nhwc (Phase 0, regime R0)

Block axes (from `op_design.md` → Blocking Model): `m` (bm = `block_m_tiles`), `n` (bn = `block_n_tiles`), `k` (bk = `block_k_tiles`), `g` (`block_groups` = 1). The window is the m-block's footprint and is not a separate axis.

`T` = tile bytes of the activation compute format. `T_w` = weight tile bytes. `T_acc` = accumulator tile bytes. Phase 0 (fp32, `fp32_dest_acc_en = True`): `T = T_w = T_acc = 4096`. Refinement 2: CB formats follow the tensors — activation-side CBs (`cb_act_window`, `cb_act_rm`, `cb_act_tiles`, `cb_act_send`, `cb_out_rm`, `cb_out_shard`) use the input dtype (bf16: `T = 2048`, `tap_width` granule 8), `cb_weight_tiles` / `cb_bias_tiles` the weight dtype (bf8b: `T_w = 1088`), and `cb_partials` / `cb_biased` follow `fp32_dest_acc_en` from `compute_policy(dtype, weight_dtype)` (currently on for fp32 and bf16 inputs, so `T_acc = 4096`). The "Float32" column entries below are the fp32-input instance.

| CB | Capacity (pages) | Live set | Axis accounting | Page format | Producer | Consumer | Lifetime | Shares with / why not |
|----|------------------|----------|-----------------|-------------|----------|----------|----------|-----------------------|
| `cb_act_window` | `window_sticks_max` (page = `tap_width·elem_bytes`) | `window_sticks(m_block)`, i.e. the whole footprint, all simultaneously read by the gather | `{m: spans → window_sticks(bm), affine in bm (rows (R_out(bm)−1)·s + d(k−1) + 1 per image crossed × W); n: streams (reused for every n-block, not scaled); k: streams (every K-block's gather reads it); g: spans → 1}` | Float32 (input dtype; data is copied, not computed) | reader | reader (private scratch) | fetch of m-block b → last gather of b | Cannot share with `cb_act_rm`: the gather reads the window while writing `cb_act_rm`, so they are concurrent. Cannot alias the compute-side CBs: the reader fetches window b+1 while compute still holds b's tiles (stall-shadow overlap is an explicit pipelining decision). |
| `cb_act_rm` | `act_rm_depth · bk` (= 2·bk) | `bk` (one tile-row of one K-block) | `{m: streams (one tile-row per quantum); n: streams (re-gathered per n-block); k: spans → bk; g: spans → 1}` | Float32 | reader | compute (tilize) | per tile-row | Capacity exceeds the live set by `act_rm_depth`: double buffering lets the reader gather tile-row t+1 during the tilize of t. Cannot alias `cb_act_tiles` because tilize reads one and writes the other at the same time. |
| `cb_weight_tiles` | `weight_depth · bk · bn` (= 2·bk·bn) | `bk·bn` (one K-block) | `{m: streams (re-streamed per m-block); n: spans → bn; k: spans → bk; g: spans → 1}` | Float32 (`weight_dtype`) | writer | compute (matmul in1) | per K-block | Capacity exceeds the live set by `weight_depth`: double buffering prefetches K-block j+1 under matmul j. Live during the matmul, so it is concurrent with every compute CB. |
| `cb_bias_tiles` | `core_n_tiles` (= bn·num_n_blocks). Allocated only if `has_bias`. | `core_n_tiles` (fronted for the kernel's life) | `{m: streams (reused every m-block); n: spans → core_n_tiles (whole core slice, ≥ bn); k: streams (absent); g: spans → 1}` | Float32 (`weight_dtype`) | writer | compute (bias add) | whole kernel | Persistent across all blocks, so it overlaps every other CB. Holding the whole core slice (not bn) avoids a re-read per n-block. Phase 0 has `num_n_blocks = 1`, so it equals bn. |
| `cb_act_tiles` | `bm · bk` | `bm · bk` (the whole in0 K-block is fronted for the matmul) | `{m: spans → bm; n: streams (refilled per n-block); k: spans → bk; g: spans → 1}` | Float32 | compute (tilize) | compute (matmul in0) | per K-block | Could alias with `cb_out_rm`? No: `cb_out_rm` is drained by the writer asynchronously while the next block's tilize runs. Could alias with `cb_partials`? No: the matmul reads in0 while accumulating into the partials, so they are concurrent. |
| `cb_partials` | `bm · bn` | `bm · bn` (the accumulator) | `{m: spans → bm; n: spans → bn; k: streams (accumulated in place, packer L1 acc); g: spans → 1}` | Float32 (= DEST width; `fp32_dest_acc_en` on) | compute (matmul) | compute (bias add, or untilize when there is no bias) | per (m,n) block | **Shares** its bytes with `cb_biased` (alias pair, accumulate-then-finalize in place). |
| `cb_biased` | `bm · bn` (alias of `cb_partials`: one `CBDescriptor`, two `CBFormatDescriptor`s, `total_size = bm·bn·T_acc`) | `bm · bn` | `{m: spans → bm; n: spans → bn; k: streams (none left); g: spans → 1}` | Float32 | compute (bias add) | compute (untilize) | per (m,n) block, bias path only | **Shares** with `cb_partials`. The alias gives the bias phase real credit edges with zero extra L1. |
| `cb_out_rm` | `out_rm_depth · bn` (= 2·bn) | `bn` (one tile-row) | `{m: streams (one tile-row per quantum); n: spans → bn; k: streams (absent); g: spans → 1}` | Float32 (output dtype) | compute (untilize) | writer | per tile-row | Capacity exceeds the live set by `out_rm_depth`: double buffering overlaps the untilize with the DRAM write. Untilizing in place into the accumulator is impossible because the TILE→RM byte layouts differ and the writer drains concurrently with the next block's matmul packs. |

## Symbol table

| Symbol | Bound | Predicate establishing it |
|---|---|---|
| `window_sticks_max` | Block in one output row: `(d(kH−1)+1) · min(W, (bm·32−1)·s + d(kW−1) + 1)` (2-D column clip). Block over several rows: `≤ ((ceil(bm·32/W_out) + 1)·s + d(k−1) + 1)·W` within one image crossing, first/last row trimmed. Exactly the host-computed maximum over all blocks (`_window_bounds`, mirrored by the reader). | the `select_blocking()` fit check: `footprint ≤ CB_BUDGET_BYTES`, else `≤ l1_cap_bytes()` |
| `tap_width` | `≤ C_in_g + l1_granule_elems − 1` | definition |
| `bm` | `1 ≤ bm ≤ core_m_tiles = ceil(M_t / min(M_t, num_cores))` | selection rule steps 2–3 |
| `bn` | divides `core_n_tiles`, which divides `N_t` | selection rule step 4 |
| `bk` | divides `K_t = kH·band_tiles` | mechanism cap (uniform `in0_block_k`) |
| `core_n_tiles` | `N_t / g_n` | Work Distribution |
| `act_rm_depth`, `weight_depth`, `out_rm_depth` | 2, 2, 2 | host constants |
| `CB_BUDGET_BYTES` | 768 KiB, preferred | module constant (perf lamp L5) |
| `l1_cap_bytes()` | `get_max_worker_l1_unreserved_size() − 128 KiB` (≈ 1.33 MiB on BH) | hard cap, used only when nothing blocks under `CB_BUDGET_BYTES` (verifier addition) |

## Total per-core footprint

```
F(bm, bk, bn) = window_sticks_max(bm) · tap_width · elem_bytes      # scales with bm (affine footprint) and W·C_in
              + act_rm_depth · bk · T                                # scales with bk
              + bm · bk · T                                          # scales with bm·bk
              + weight_depth · bk · bn · T_w                         # scales with bk·bn
              + bm · bn · T_acc                                      # scales with bm·bn (partials ≡ biased)
              + has_bias · core_n_tiles · T_w                        # scales with the core's n slice
              + out_rm_depth · bn · T                                # scales with bn
```

Over the 62 Phase 0-eligible golden shapes (110 cores, fp32), F ranges from 162 to 676 KiB. Input and output tensors stay in DRAM (no L1 tensors), so the device peak equals F.

Verifier re-measure (device `device_l1_peak_bytes`, golden run after the column clip and `strided` were added): 32 KiB – 1.27 MiB. The only shape above the 768 KiB preference is SDXL 32×32×1280→1280 s2 (fp32, 1.26 MiB: its one-tile-row window spans 5 input rows × 32 columns × 5 KiB sticks), which blocks only under the hard cap.

## Data-movement budget (chosen split R0)

`ρ` = `Σ_blocks window_sticks / (N·H·W)`, the halo duplication factor (computed on host; 1.0–6.4 over the Phase 0 golden set). `nb_m = ceil(core_m_tiles / bm)` is the number of m-blocks per core (1 for all but one Phase 0 shape).

| Tensor | DRAM crossings | Why that many | Cross-core traffic added |
|--------|----------------|---------------|--------------------------|
| input | `g_n · ρ` | Each n-core of an m range fetches the same window, which is reuse-shared across n and not multicast in R0. Each m-block re-fetches its `d(k−1)` halo rows, and so does each neighbouring core. | none |
| weight | `g_m · nb_m` (each core reads its `K_t × core_n_tiles` slice once per m-block) | The weight is reuse-shared across m and not multicast in R0. It is streamed per K-block and not resident across m-blocks. | none |
| bias | `g_m` (per core, its slice, once) | persistent `cb_bias_tiles` | none |
| output | 1 | written once from `cb_out_rm` | none |

Total bytes, DRAM tier: `g_n·ρ·|in| + g_m·nb_m·|W| + g_m·|b| + |out|`. NoC cross-core tier: 0. Core-local tier: the gather reads each window stick `kH·kW` times on average, as L1→L1 copies of `kW·tap_width` runs, and every im2col tile is written once and tilized once.

> Cheapest-traffic split considered: **m-split with resident window plus neighbour halo over the NoC (R7/R3) and weight multicast (R1)**. Its delta: the input DRAM tier drops from `g_n·ρ·|in|` to `|in|`, with `(ρ−1)·|in|` of halo moving to the NoC tier. The weight DRAM tier drops from `g_m·nb_m·|W|` to `|W|`, and in exchange `(g_m−1)·|W|` of multicast payload is added on the NoC tier.
> Example, 14×14×256→512 fp32: DRAM falls from ≈ 5.6 MiB + 22.5 MiB to 0.2 MiB + 4.5 MiB.
> Implemented: **R0 (m×n split, DRAM window, per-core weight reads)**.
> Deferred because R1/R3/R7 are layered onto R0's unchanged CB set and gather code (only the window-fill and weight-fill sources change), and they arrive with the sharded-HEIGHT refinement (R3). That refinement pins the multicast receiver group and halo neighbours that R7 then reuses for interleaved inputs. R0 is the stepping stone, not a dead end: no R0 buffer or loop is discarded when they land.

---

# L1 Ledger addendum: Refinement 1 — regimes R4 (BLOCK_SHARDED) / R5 (WIDTH_SHARDED)

Block axes: `m` (bm = a divisor of `core_m_tiles`; whole shard by default), `n` (bn = `nt_s = ceil(N_t/S)`, the whole output-shard row: pinned, the untilize writes shard sticks directly), `k` (bk = a divisor of the slice's `kH·band_tiles`, the largest that fits — each K-block is one multicast handshake), `S` = channel slices (BLOCK grid columns / WIDTH cores). `tap_width` = the shard's slice width; K order `[slice][kh][kw × slice_width]`, each (slice, kh) band padded to `band_tiles` tiles.

| CB | Capacity (pages) | Live set | Axis accounting | Page format | Producer | Consumer | Lifetime | Shares with / why not |
|----|------------------|----------|-----------------|-------------|----------|----------|----------|-----------------------|
| `cb_in_shard` (8) | the input shard (zero-copy, `cb_descriptor_from_sharded_tensor`) | whole shard | `{m: pinned by the shard; k: the core's channel slice}` | input dtype (stick pages) | — (tensor) | reader (gather reads in place; remote cores' halo reads) | kernel | Is the tensor; no CB bytes allocated. Never re-read through a TensorAccessor. |
| `cb_act_window` (0, reused) | `halo_max` sticks (≥ 1) | the m-block's window sticks outside the resident shard | `{m: spans → halo(bm); n: streams; k: streams}` | input dtype | reader | reader (private) | per m-block | Same CB index/role as R0's window, now holding only the halo (one NoC read per owning same-column shard run). WIDTH: 0 sticks (1-page stub). |
| `cb_act_rm` (1) | `act_rm_rows·bk`, `act_rm_rows = max(ACT_RM_DEPTH, ACT_RM_KBLOCKS·bm)` when S > 1 (= one whole own K-block), else `2·bk` | `bk` per tile-row; up to one own K-block ahead | `{m: spans → bm (S > 1); k: spans → bk}` | input dtype | reader | compute (tilize) | per own K-block | Refinement 3: one own K-block deep, so the reader gathers the next own block during the other S−1 slices' turns instead of inside the owner's tilize. Cannot alias `cb_act_send` (tilize reads one, writes the other). |
| `cb_act_send` (10) | `act_send_depth·bm·bk`; depth = `min(ACT_SEND_DEPTH, kblocks_per_slice)` = 1 (Refinement 3 removed the whole-slice prefetch: 468 KiB on SDXL; depth ≥ kblocks_per_slice is the same thing as a knob value) | own tilized K-blocks not yet multicast | `{m: spans → bm; k: spans → bk × depth}` | input dtype | compute (tilize) | writer (multicast source) | per own K-block | Cannot alias `cb_act_tiles`: the owner's own block is in flight (send) while the previous received block is still being matmul'd. Only allocated when S > 1. |
| `cb_act_tiles` (4) | `ACT_RECV_DEPTH·bm·bk` (= 2) when S > 1, else `bm·bk` | `bm·bk` (in0 of the current K-block) | `{m: spans → bm; k: spans → bk}` | input dtype | writer (multicast landing, every group core incl. WIDTH sinks) / compute tilize when S = 1 | compute (matmul in0) | per K-block | Double-buffered so the next block's multicast overlaps the current matmul. Allocated on the multicast rectangle (WIDTH sinks included) so the landing address is identical. |
| `cb_weight_tiles` (2) | `SHARDED_WEIGHT_DEPTH·bk·bn` (= 6) | `bk·bn` per K-block; up to depth − 2 blocks ahead | as R0 (column tiles ≥ N_t zero-filled in-kernel, technique 5) | weight dtype | writer (BLOCK, g_m > 1: row-0 DRAM read + column multicast; others receive) | compute | per K-block | Refinement 3: the weight stream runs depth − 2 K-blocks ahead of the act exchange so its DRAM latency / column handshake leave the row chain (6.5 KB per block at bf8b, bk = 3). Landing address lockstep down the column. |
| `cb_bias_tiles` (3) | `nt_s` | `nt_s` | as R0 (tiles ≥ N_t zero-filled) | weight dtype | writer | compute | kernel | unchanged |
| `cb_partials` / `cb_biased` (5/6) | `bm·bn` | `bm·bn` | as R0 | Float32 | compute | compute | per m-block | alias pair, unchanged |
| `cb_out_shard` (9) | the output shard's full tile-rows (zero-copy) | — | `{m: spans → core_m_tiles; n: spans → nt_s}` | output dtype, tile-sized pages (bn pages = 32 shard sticks) | compute (untilize) | — (tensor) | kernel | Is the output tensor; no CB bytes. |
| `cb_out_rm` (7, reused) | `bn` (WIDTH with `M % 32 ≠ 0` only) | the partial last tile-row | — | output dtype | compute (untilize) | writer (copies the valid rows into the shard) | once | A WIDTH output shard is exactly M rows tall, so the last partial tile-row cannot be untilized in place. |

Footprint `F = max(halo,1)·tap·eb + act_rm_rows·bk·T + [S>1]·depth_send·bm·bk·T + depth_recv·bm·bk·T + weight_depth·bk·nt_s·T_w + bm·nt_s·T_acc + has_bias·nt_s·T_w + out_tail·nt_s·T`, chosen under `SHARDED_CB_BUDGET_BYTES` (480 KiB, Refinement 3), then `CB_BUDGET_BYTES` (768 KiB), then the hard cap. Input and output shards live in L1 on top of F (measured: graph-capture peak = F + both shards).

Measured (fp32, SDXL 64×64×640→640 3×3, pinned 10×10 BLOCK, shard (410, 64)): bm = 13, bk = 3, 60 K-blocks, F = 702,208 B (over the 618,992 B bf16 production peak — fp32 tiles are 2× the bytes; Refinement 3 re-solves this at bf16 × bf8b).

Refinement 3 (bf16 × bf8b, same config): bm = 13, bk = 3, 60 K-blocks, F = halo 17,280 + act_rm 79,872 + send 79,872 + recv 159,744 + weights 39,168 + partials 106,496 + bias 2,176 = 484,608 B; graph-capture `device_l1_peak_bytes` = 590,336 B (≤ 618,992 B target; + input shard 52,480 + output shard 53,248). The budget leaves 16 KiB for an L1_SMALL region. bk = 6 would need ≈ 777 KB of CBs at bm = 13 (every bk tile costs 13·(1 rm + 1 send + 2 recv)·T), so bk = 3 is the largest divisor of the slice K (18) under the target.

## Data-movement budget (R4 / R5)

| Tensor | DRAM crossings | Cross-core traffic added |
|--------|----------------|--------------------------|
| input | 0 (L1 resident) | halo: `(ρ_halo − 1)·|in|`-ish same-column NoC reads (BLOCK only); the own slice is read in place |
| im2col | — | each core's tilized slice multicast to the S−1 others in its group: `(S−1)·|im2col slice|` per group. K order interleaves the slices (K-block j is slice j mod S's own block j div S; Refinement 3) — weight rows are mapped in the writer, the prepared layout is unchanged |
| weight | BLOCK g_m > 1: 1 per grid column per m-block (row-0 read + column multicast); WIDTH/single row: 1 per core | column multicast `(g_m − 1)·|W column slice|` |
| bias | 1 per core | none |
| output | 0 (written into the L1 output shard; WIDTH tail rows copied core-locally) | none |

## Refinement 4 — depthwise (regime R11) deltas

Same CB set; no CB added or removed. `taps = kH·kW`; the depthwise K-block is `bk = taps · bn` (one per n-block, `num_k_blocks = 1`).

| CB | Depthwise capacity (pages) | Live set | Axis accounting | Shares with / why not |
|----|----------------------------|----------|-----------------|-----------------------|
| `cb_act_window` | `window_sticks_max`, page = `core_n_tiles·32·elem_bytes` (R0: only the core's own channel tiles, read at channel offset `n_tile_start·32`); sharded: the halo, page = the slice stick | as dense | `{m: spans → window(bm); n: spans → core_n_tiles (whole core range, every n-block reads it); taps: streams}` | as dense |
| `cb_act_rm` | `ACT_RM_DEPTH · taps · bn` | one tile-row `taps · bn` | `{m: streams (tile-row); n: spans → bn; taps: spans → taps}` | as dense (double-buffered gather vs tilize) |
| `cb_act_tiles` | `DW_ACT_TILE_ROWS · taps · bn` (= one tile-row) | one tile-row | `{m: streams (one tile-row: no in0 reuse across rows); n: spans → bn; taps: spans}` | Smaller than dense (`bm·bk`): the tap accumulation consumes a tile-row before the next is tilized (same TRISCs), so a depth of 1 suffices. |
| `cb_weight_tiles` | `DW_WEIGHT_DEPTH · taps · bn` (`weight_k_rows = taps`), weight dtype; each tile's row 0 = the tap's per-channel weights | `taps · bn` (fronted for the whole (m, n) block) | `{m: streams (re-read per m-block); n: spans → bn; taps: spans}` | Depth 2 prefetches the next (m, n) block's tap weights. Rows 1..31 of every tile are zero (the row-broadcast operand reads row 0 only): 32× the bytes of the real weights, still ≤ `2·taps·bn` tiles. |
| `cb_partials` / `cb_biased` | `bm · bn` | `bm · bn` | `{m: spans → bm (one tile-row pushed per pass); n: spans → bn; taps: accumulated in DEST (WholeShape), packed once per output tile}` | as dense; no packer L1 accumulation on this path |
| `cb_act_send` | not allocated (no K exchange: each depthwise channel slice is self-contained) | — | — | — |

Depthwise footprint: `F_dw = window·tap_bytes + (ACT_RM_DEPTH + DW_ACT_TILE_ROWS)·taps·bn·T + DW_WEIGHT_DEPTH·taps·bn·T_w + bm·bn·T_acc + bias + out`. Walk: bn (whole core range first) then bm.

### Data-movement budget (depthwise)

| Tensor | DRAM crossings | Why |
|--------|----------------|-----|
| input (R0) | `ρ` (each byte read by one n-core only: windows hold only the core's channel tiles) | halo rows re-read by neighbouring m-cores |
| input (BLOCK/WIDTH) | 0 (resident shard); halo from same-column shards over the NoC | no row multicast |
| weight | `g_m · nb_m` per channel slice (`taps · core_n_tiles` tiles each) | no column multicast (tiny) |
| output | 1 | |

## Refinement 5 — HEIGHT_SHARDED (regime R3) + weight multicast (R1) deltas

No CB added, removed or resized: HEIGHT runs the R4 sharded path with one channel slice
(`S = 1`, `act_mcast = False`, no `cb_act_send`). Rows that change meaning:

| CB | HEIGHT capacity / meaning | Shares with / why not |
|----|---------------------------|-----------------------|
| `cb_in_shard` (8) | the core's resident HEIGHT shard (`shard_h` sticks × C_in, zero-copy) | is the input tensor; no CB bytes |
| `cb_act_window` (0) | the halo: window sticks outside the resident shard, from any input shard (shard order over the grid), one NoC read per owner run | the output split (`core_m_tiles = ceil(M_t / num_shards)`) need not align with the input shards (strided convs), so the halo can be most of the window; sized by `halo_max(bm)` |
| `cb_out_shard` (9) | the output HEIGHT shard: `core_m_tiles` whole tile-rows × all `N_t` channel tiles, on the first `g_m` input cores | is the output tensor. Allocated per core at the full shard size, so the last core's ragged tile-rows untilize in place (no `out_tail`) |
| `cb_weight_tiles` (2) | as R4 (`SHARDED_WEIGHT_DEPTH · bk · N_t`; depthwise `DW_WEIGHT_DEPTH · taps · N_t`) — lockstep slot on every core for the multicast | unchanged |

Measured plans (bf16 × bf8b, bias; F = CB footprint, shards extra):
- ResNet50 layer1 3×3 (480-stick shards, 105 cores): bm = 15, bk = 6, 3 K-blocks, halo 114 sticks, F = 426,880 B.
- layer2 downsample s2 (457, 110): bm = 4, bk = 6, 6 K-blocks, halo 649 sticks, F = 466,432 B.
- layer2 3×3 (128, 98): bm = 4, bk = 9, 4 K-blocks, halo 58, F = 430,336 B.
- 1×1 s2 shortcut (457, 110): bm = 4, bk = 2, 1 K-block, halo 535, F = 337,280 B.

### Data-movement budget (HEIGHT)

| Tensor | DRAM crossings | Cross-core traffic added |
|--------|----------------|--------------------------|
| input | 0 (L1 resident) | halo sticks from neighbour shards, one NoC read per owner run (strided: most of the window) |
| weight | 1 per m-block (core 0 reads; `ceil(core_m_tiles/bm)` = 1 at the whole-shard default) | multicast `(g_m − 1)·|W|` per m-block (dense and depthwise); off when the core set needs > 3 rectangles or m-block counts differ |
| bias | 1 per core | none |
| output | 0 (untilized into the L1 output shard) | none |

## Refinement 6 — HEIGHT resident weights, single-K pipeline, tap-major gather

| CB | HEIGHT capacity / meaning (R6) | Shares with / why not |
|----|--------------------------------|-----------------------|
| `cb_weight_tiles` (2) | the whole weight slice, `K_t · N_t` tiles (weight dtype), resident for the kernel's life. Allocated on the bounding rectangle of the input cores (multicast landing slot, sinks included). | Concurrent with every compute CB. It replaces the depth-6 stream (6·bk·N_t), so it is smaller whenever K_t < 6·bk. Not resident when no blocking fits: the streamed path is the fallback. |
| `cb_act_rm` (1) | `bm · bk` (single-K pipeline: one whole m-block, bk = K_t); else `ACT_RM_DEPTH · bk` | The extra m-block of capacity is what overlaps the gather of m-block i+1 with compute of m-block i. Tap-major pages hold 32-pixel × tap blocks: same bytes per page. |
| `cb_act_tiles` (4) | `bm · K_t` | unchanged role |
| `cb_partials` / `cb_biased` | `bm · N_t` fp32 | alias pair, unchanged |
| `cb_act_window2` (11) / `cb_act_rm2` (12) | only with `SPLIT_GATHER` (off): the writer RISC's halo window and RM im2col for odd m-blocks | Separate from 0 / 1 because a CB has one producer. |

Footprint (single-K): `F = halo(bm)·tap_bytes + bm·K_t·T (rm) + bm·K_t·T (act) + K_t·N_t·T_w + bm·N_t·T_acc + N_t·T_w (bias)`. The walk takes the largest bm with ≥ `HEIGHT_MIN_M_BLOCKS` m-blocks under `SHARDED_CB_BUDGET_BYTES`.

Measured plans and `device_l1_peak_bytes` (bf16 × bf8b, bias):

| Case | bm | bk | halo sticks | peak L1 (B) | achievable (B) |
|------|----|----|-------------|-------------|----------------|
| layer1 3×3 | 5 | 18 | 57 | 581,120 | 723,456 |
| layer2 downsample | 1 | 36 | 233 | 534,272 | 670,576 |
| layer2 3×3 | 1 | 36 | 29 | 397,824 | 450,928 |
| 1×1 s2 | 2 | 2 | ≤265 | 264,320 | 317,040 |

### Data-movement budget (HEIGHT, R6)

| Tensor | DRAM crossings | Cross-core traffic |
|--------|----------------|--------------------|
| input | 0 (L1 resident) | halo from neighbour shards, one NoC read per owner run. Output shards sit on the input core covering their window, so strided convs move the window minus the resident overlap, not the whole window. |
| weight | 1 (read once by the first compute core) | one multicast of `K_t·N_t` tiles to the input grid's bounding rectangle (once per kernel, not per m-block) |
| bias | 1 per core | none |
| output | 0 | none |

## Refinement 7 — TILE activations (+ bf8b)

The RM pipeline CBs (`cb_act_window`, `cb_act_rm`, `cb_act_tiles`, `cb_act_send`) take `rm_dtype(dtype)`: bf8b untilizes to bf16 (`T = 2048`), other dtypes as before. TILE outputs are packed as tiles of the output dtype (`T_out = tile_bytes(dtype)`; bf8b 1088).

| CB | Capacity (pages) | Live set | Axis accounting | Page format | Producer | Consumer | Lifetime | Shares with / why not |
|----|------------------|----------|-----------------|-------------|----------|----------|----------|-----------------------|
| `cb_act_window` (TILE) | `win_tile_rows · c_tiles` (page = RM tile `T`; one tile-row = 32 sticks of `tap_width = c_tiles·32`) | sharded: the core's whole untilized window (own shard ∪ window of its whole m range); DRAM: one m-block's window tile-rows | `{m: sharded spans the core range (once per kernel) / DRAM spans one m-block (affine in bm); n, k: streams (reused)}` | `rm_dtype` | compute (untilize) | reader (gather source) | sharded: whole kernel; DRAM: per m-block | Replaces the RM halo scratch (same index, now compute-produced). Cannot alias `cb_win_tiles`: untilize reads one while writing the other. |
| `cb_win_tiles` (13, TILE only) | sharded: `max(halo_before + halo_after) · c_tiles`; DRAM: `win_tile_rows · c_tiles` | the halo (sharded) / window (DRAM) tiles awaiting untilize | `{m: as cb_act_window}` | input dtype (bf8b 1088) | reader (NoC: owner-shard runs / DRAM tiles) | compute (untilize) | until untilized | The own shard is never copied: compute untilizes it in place from `cb_in_shard` (zero-copy). Only halo tile-rows are staged here. |
| `cb_out_shard` (TILE) | `core_m_tiles · bn` (page `T_out`) | — | as before | output dtype | compute (bias add / copy pack) | — (it is the output tensor) | whole kernel | Zero-copy on the output shard. TileRowMajor partials are the shard's tile order. |
| `cb_out_rm` (R0 TILE) | `max(2, 2·sb_h) · bn` (page `T_out`) | one row group (`sb_h·bn`) | `{m: streams}` | output dtype | compute (bias add / copy pack) | writer (tile pages `t·N_t + n`) | per row group | Two row groups double-buffer the pack against the DRAM write. |
| R9 (`direct_in0`) | `cb_act_window`, `cb_act_rm`, `cb_act_send`, `cb_win_tiles`: 1 page each (idle); HEIGHT `cb_act_tiles`: 1 page | — | — | — | — | — | — | in0 is `cb_in_shard` itself (HEIGHT: compute's matmul; BLOCK/WIDTH: the writer multicasts shard K-blocks into `cb_act_tiles`, which keeps the input dtype). |

Footprint additions (TILE): sharded `+ win_tile_rows·32·tap_width·elem_bytes + halo_tile_rows·c_tiles·T_in` (replacing the per-block RM halo scratch); R0 `+ win_tile_rows·c_tiles·T_in` (the window term becomes whole tile-rows).

Data movement (TILE): sharded input as before (shard resident; halo once per kernel as whole tile-rows, one NoC read per owner run); DRAM TILE input `g_n · ρ_tile` crossings (ρ over whole window tile-rows per m-block); output 1. R9: the input never leaves L1 on HEIGHT, and is multicast once along the row on BLOCK/WIDTH.

## Refinement 8 — channel alignment, dilation, grouped convs

| CB | Capacity (pages) | Live set | Axis accounting | Page format | Producer | Consumer | Lifetime | Shares with / why not |
|----|------------------|----------|-----------------|-------------|----------|----------|----------|-----------------------|
| `cb_stick_stage` (14) | `2 · stage_row_sticks` (page = `stage_stride` = C_in·elem_bytes rounded up to the DRAM read alignment; `stage_row_sticks = min(STAGE_ROW_STICKS = 64, W)`). R0 RM only, only when `tap_bytes` is not a DRAM-alignment multiple (unaligned C_in). | one half (`stage_row_sticks` sticks) receives DRAM reads while the other half is copied into `cb_act_window` | `{m: streams (per window row chunk); n: streams; k: absent; g: spans → 1}` | input dtype (copied, not computed) | reader (DRAM reads) | reader (L1→L1 copies into the window) | per window chunk | Cannot be the window itself: a DRAM read must land at the source's alignment (64 B on BH), while window sticks are packed at `tap_bytes` (16 B granule) so a kernel-row band stays one contiguous run. Zeroed once, so the copies carry zero pad lanes into the window. ≤ 2·64·64 B = 8 KiB. |

Other deltas (no new CB):
- `cb_act_window` slot = `src_bytes` (the source stick: `tap_bytes` on R0 RM; the HEIGHT shard stick, which may be wider than the granule-padded C_in; the TILE window's whole tile columns). Direct-read R0 windows with `stick_bytes < tap_bytes` are zeroed once at kernel start (`window_zero_bytes`).
- Dilated R0 block inside one output row: the window holds only its `kH` kernel rows (`kH · win_w` sticks instead of `(d(kH−1)+1) · win_w`). This is what lets the d = 18 fp32 256-channel cell fit.
- Grouped: native groups on R0 shrink K to one group's `kH·band(C_in/g)`; the window stick is the group's `C_in/g` channels. Block-diagonal groups (sharded, or groups that are not whole tiles) are the dense footprint over all of C_in.

### Data-movement budget (Refinement 8)
- input (R0, native groups): `g_n · ρ` as before, but each core reads only its group's `C_in/g` channels of each stick (`1/g` of the bytes). Unaligned C_in: same crossings; each stick also costs one L1→L1 copy.
- input (R0, dilated, block inside one output row): `kH` input rows per output row instead of `d(kH−1)+1`.
- weight: unchanged. Native groups read `[K_g, C_out]` (`1/g` of a block-diagonal weight). Block-diagonal reads `g`× the per-group bytes.

## Refinement 9 — ResNet50 BLOCK perf (R4) deltas

No CB was added, removed or resized at the default knob values; the footprint is unchanged. Rows that change meaning:

| CB | Capacity | Change | Shares with / why not |
|---|---|---|---|
| `cb_act_tiles` (4) | `(ACT_RECV_DEPTH + act_channels − 1)·bm·bk`; `act_channels = 1` (default) → 2 blocks, as before | `ACT_CHANNELS = 2` (parked) adds a third landing slot: two rounds in flight while compute holds a third. Measured slower (concurrent multicasts in one row), so it stays at 1. | unchanged |
| `cb_weight_tiles` (2) | `SHARDED_WEIGHT_DEPTH·bk·bn` (= 6) | Now a ring the writer addresses ahead of its write pointer: DRAM reads run up to `depth − 3` blocks ahead of the landing (one read transaction id per block), receivers land one block ahead of the matmul. BLOCK prepared weights are slice-major (`weight_slice_tiles`): a column's K-block is one contiguous page run over all DRAM banks. | unchanged |
| `cb_act_rm` (1) | unchanged (one own K-block) | BLOCK / WIDTH stride-1 convs whose taps are one tile (`tap_tiles = 1`, or when it does not shrink bk) gather tap-major: same bytes, ~3× fewer local reads. | Shrinking it below one own K-block would put the gather (1.3–1.5k cycles per tile-row) inside compute's tilize. |

`device_l1_peak_bytes` (graph capture) vs `achievable_l1_peak_bytes`: layer3 downsample 413,696 / 370,416; layer3 3×3 352,384 / 238,640; layer4 3×3 392,064 / 161,296 B — unchanged by this refinement, above target. The act CBs (`cb_act_rm` + `cb_act_send` + 2 landing slots = 4 own-K-block copies, 245 KB at bm = 10, bk = 3 for layer3) are the excess; a smaller block multiplies exchange rounds (bm = 5 / bk = 3: 100k vs 77k cycles on layer3 3×3).

### Data-movement budget (R4, Refinement 9)

DRAM crossings per tensor are unchanged: input 0 (resident shard; halo L1→L1), weights 1× per grid column (now spread over every bank), output 0 (resident shard). NoC traffic per core is unchanged: one act K-block per round from its owner, one weight K-block per K-block down the column. The weight multicast source rotates down the column (block w from row w mod g_m) when a weight K-block is ≤ `WEIGHT_ROTATE_MAX_RATIO` (0.25) of the act K-block's bytes, else stays on row 0.
