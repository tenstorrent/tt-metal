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
