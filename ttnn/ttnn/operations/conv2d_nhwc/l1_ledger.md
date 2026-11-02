# L1 Ledger: conv2d_nhwc (Phase 0, regime R0)

Block axes (from `op_design.md` → Blocking Model): `m` (bm = `block_m_tiles`), `n` (bn = `block_n_tiles`), `k` (bk = `block_k_tiles`), `g` (`block_groups` = 1). The window is the m-block's footprint and is not a separate axis.

`T` = tile bytes of the activation compute format. `T_w` = weight tile bytes. `T_acc` = accumulator tile bytes. Phase 0 (fp32, `fp32_dest_acc_en = True`): `T = T_w = T_acc = 4096`.

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
