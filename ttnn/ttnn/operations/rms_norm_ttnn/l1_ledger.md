# L1 Ledger: rms_norm_ttnn

Companion to `op_design.md`. Schema and audits: `.claude/references/l1-footprint-discipline.md`.

**Named block axes** (from the Blocking Model): `row`, `width`, `channel`, `pass`,
`width-group slot`. Every row's `Axis accounting` cell names all five, each tagged `spans` or
`streams`.

**Inventory before solve.** The buffer count was minimized before any budget predicate was
consulted. What that produced, concretely:

| Phase boundary | Could it be storage-free? | Decision |
|----------------|--------------------------|----------|
| `normalize_block` → `scale_block` → `bias_block` | yes, for all but the first | `normalize_block` packs into `cb_normalized`; `scale_block` **transforms it in place**; `bias_block` **packs into the destination** `cb_output_tiles`. This is patterns 1+2 of Rule 3 and it is what makes bias cost **zero** block-sized buffers |
| `residual_add_block` → `square_block` → `normalize_block` | no | `cb_x_sum` is genuinely required: `t = x + r` is read by *both* passes, and fusing the add into pass B's broadcast multiply is inexpressible (`DestReuseBinary` carries no broadcast parameter, `chain.hpp:526`). Pattern 4 (fold into DEST) covers the pass-A half only — recorded as Lamp L-RES-FUSE, not as a buffer |
| `square_block` → `reduce_accumulate_block` | partly | Pattern 4: the D12 DEST fold accumulates the chunk's width tiles inside DEST, collapsing `cb_x_squared` from `BR*WC` pages to `BR*1`. Gated on `WC ≤ 8` and `PARTIAL_W == 0` |
| root fold → finalize | yes | Fused into one DEST window (D22); `ROOT_FOLD_OUT` and every combine-path use of `cb_row_stat` were **deleted**, saving 256 kB/core at `BR = 32` |
| combine partial hand-off | yes | D27's compact transpose took `cb_partials_gathered` from `GS * BR` pages to `GS`, removing the `GROUP_SIZE × BLOCK_ROWS` term from the block solve entirely |
| `cb_input_tiles` / `cb_output_tiles` / `cb_residual_tiles` on a resident shard | yes | Pattern 1 taken to its limit: **zero-copy** CB descriptors over the shard buffers, 0 arena bytes each |

Only after all six was a budget predicate introduced.

---

## Symbol table

| Symbol | Meaning | Bound | Predicate establishing the bound |
|--------|---------|-------|----------------------------------|
| `BR` | `BLOCK_ROWS`, tile-rows per block | `1 ≤ BR ≤ min(core_row_tiles, brmax)`; **`BR ≤ 32` on any combine path** | `_solve_blocking`: `brmax = (budget − fixed) // per_tilerow`; the 32 clamp is D27's compact-tile mechanism cap (host `assert`, kernel `static_assert`) |
| `WC` | `WT_CHUNK`, width tiles per block | `1 ≤ WC ≤ WPC`, and **`WC | WPC`** | D1: `tilize`/`untilize` take `block_width_tiles` as a compile-time template param; `reduce`'s `BulkWaitBulkPop` asserts `num_pages % cols == 0`; a multi-page reserve must not straddle the ring |
| `WPC` | `wt_per_core`, this core's width tiles | `WPC = Wt` (row split) or `Wt/gw` (width split) or `shard_w_t` (shard) | `_plan_placement` |
| `XH` | `x_hold_wt` | `XH = WPC` when `X_RESIDENT`, else `XH = WC` | `_solve_blocking`; `assert XH == WC*NUM_W_CHUNKS if X_RESIDENT else XH == WC` |
| `XSW` | `X_SQUARED_WT` | `XSW ∈ {1, WC}` | `1` iff `PARTIAL_W == 0 and WC ≤ DEST_ACC_SQUARE_MAX_WT (8)`; host `assert XSW in (1, WC)` |
| `DX`, `DO` | `CB_X_DEPTH`, `CB_OUT_DEPTH` | `∈ CB_DEPTH_CANDIDATES = (2,)` on TILE; forced to `1` on ROW_MAJOR | the regime search walks the candidates coarsest-first; RM's producer/consumer is a sequential tilize/untilize so depth buys no overlap |
| `DR` | `CB_R_DEPTH` | `DR = DX` | tied by construction so one knob moves both streams (Lamp L-RES-DEPTH is the measurement that could untie it) |
| `DS` | `CB_RM_STAGE_DEPTH` | `= 2` | primary knob |
| `SD` | `CB_ROW_STAT_DEPTH` | `= 2` | **correctness floor**, not a perf depth (D6) |
| `FD` | `CB_COMBINE_FLAT_DEPTH` | `= 2` | primary knob; one round in flight |
| `SP` | `scaler_pages` | `∈ {1, 2}` | `2` iff `kernel_partial_w != 0` |
| `G` | `group_size` | `1 ≤ G ≤ WIDTH_SPLIT_MAX_GROUP_CORES (16)` on the interleaved width split; `= shard grid extent` on a WIDTH/BLOCK shard, itself `≤ grid.x * grid.y` | `_auto_width_split` / `_plan_placement` |
| `GS` | `GATHER_SLOTS` | `= G + G%2 ≤ G+1` | D22's pairwise DEST walk needs an even window |
| `f0`, `f1` | combine-tree fan-ins | `f0 = 4`; `f1 = ceil(G/f0) ≥ 2`; `f0*f1 ≥ G` | `_combine_tree_arity`; returns `None` (flat) when `f1 < 2` or too few fold tiles are deleted |
| `bt` | `tile_size(input.dtype)` | `∈ {1088 (bf8b), 2048 (bf16), 4096 (fp32)}` | dtype ∈ SUPPORTED |
| `gt`, `bit` | `tile_size(weight.dtype)`, `tile_size(bias.dtype)` | same set; **independent of `bt` and of each other** | operand dtype ∈ SUPPORTED |
| `st`, `ft` | `tile_size(bf16)` = 2048, `tile_size(fp32)` = 4096 | constants | — |
| `budget` | bytes the CBs may take | `L1_SAFETY_FRACTION (0.85) * max(0, usable_L1 − l1_reserved − L1_CB_ARENA_BASE_RESERVE)` where the last two terms apply only when a shard is resident | `_solve_blocking`; `l1_reserved = shard_bytes(in) + shard_bytes(out) + shard_bytes(residual)` |

`HAS_G`, `HAS_B`, `HAS_R` are `0/1` compile-time presence flags; `RM`, `PC_RM`, `NAT_IN`,
`NAT_OUT`, `NAT_R`, `CMB`, `CMP` (compact = `CMB and BR>1`), `TREE` are `0/1` build flags.

---

## The table

| CB | Capacity (pages) | Live set | Axis accounting | Page format | Producer | Consumer | Lifetime | Shares with / why not |
|----|------------------|----------|-----------------|-------------|----------|----------|----------|-----------------------|
| `cb_input_sticks` | `RM * DS * WC` | `WC` (one tile-row's staging) | `{row: streams → 32 sticks per push, width: spans → WC, channel: —, pass: streams → re-staged per pass when !X_RESIDENT, slot: —}` | input dtype | reader | compute | whole kernel, RM only | **Cannot share.** Concurrent with `cb_output_sticks` (the untilize of block *b* overlaps the tilize of *b+1*) and with `cb_residual_sticks` (both feed one `residual_add_block`). Capacity exceeds live set by `DS` — the reader↔tilize double buffer |
| `cb_input_tiles` | `NAT_IN ? shard_h_t*shard_w_t : DX * BR * (HAS_R ? WC : XH)` | same as capacity when zero-copy; `BR*(HAS_R?WC:XH)` otherwise | `{row: spans → BR, width: spans → WC or XH, channel: —, pass: HAS_R ? streams : spans (held across both passes when X_RESIDENT), slot: —}` | input dtype | reader | compute | whole kernel | **Cannot share.** Zero-copy over the input shard when `NAT_IN` (0 arena bytes, and aliasing anything else would corrupt the caller's tensor). Otherwise concurrent with `cb_output_tiles` by the `DX`/`DO` pipelining decision. Capacity exceeds live set by `DX` — double buffering |
| `cb_x_squared` | `BR * XSW` | `BR * XSW` | `{row: spans → BR, width: spans → XSW (=1 under the DEST fold, else WC), channel: —, pass: pass A only, slot: —}` | input dtype | compute | compute | pass A | **Could share with `cb_normalized`** (pass A vs. pass B, disjoint) — **not taken**: the D25 combine pipeline issues block `b+1`'s pass A before block `b`'s combine, which makes the two concurrent. Recorded per Rule 3's pipelining clause |
| `cb_scaler` | `SP` (1 or 2) | `SP` | `{row: —, width: —, channel: —, pass: —, slot: —}` — constant | **bfloat16**, always | reader | compute | whole kernel | **Cannot share.** Live for the whole kernel; `1.0` exactly, never `1/W` |
| `cb_row_stat` | `!CMB * SD * BR` | `BR` (one block's stats) | `{row: spans → BR, width: —, channel: —, pass: spans → written in A, read in B, slot: —}` | **float32 always** — deliberately overrides the "page format follows DEST width" default, because this is the cross-chunk accumulator `reduce`'s `Accumulate::at` reloads; a 16-bit page would make the STREAM reload lossy at exactly the widths this op cares about | compute | compute | pass A → pass B | **Cannot share.** Capacity is `SD =` 2× the live set and that gap is a **correctness** requirement, not double buffering: `transform_in_place` rotates the ring and a partial final block's finalized tiles would otherwise straddle the wrap (D6). Not allocated at all on a combine path |
| `cb_gamma_sticks` | `HAS_G * PC_RM * WC` | `WC` | `{row: —, width: spans → WC, channel: spans → WC*32 weights, pass: streams, slot: —}` | weight dtype | reader | compute | `HAS_G && PC_RM` | **Cannot share with `cb_bias_sticks`** — differing page format (the two operands' dtypes are independent) and concurrent (both staged per chunk before their tilizes) |
| `cb_gamma_tiles` | `HAS_G * XH` | `XH` | `{row: **streams** → the same vector feeds every row (reuse-shared), width: spans → XH, channel: spans → XH*32, pass: spans → held across both, slot: —}` | weight dtype | reader (TILE build) / compute (RM build) | compute | `HAS_G` | **Cannot share with `cb_bias_tiles`** — differing page format, and both are live simultaneously in pass B |
| `cb_normalized` | `(!HAS_R) * (HAS_G \| HAS_B) * BR * WC` | `BR * WC` | `{row: spans → BR, width: spans → WC, channel: —, pass: pass B only, slot: —}` | input dtype | compute | compute | pass B | **Cannot share with `cb_x_squared`** — the D25 pipeline makes them concurrent (above). **Not allocated when `HAS_R`**: `cb_x_sum` already holds a block-shaped input-dtype buffer that pass B can pack into and transform in place |
| `cb_output_tiles` | `NAT_OUT ? out_shard_pages : DO * BR * WC` | as capacity when zero-copy; `BR*WC` otherwise | `{row: spans → BR, width: spans → WC, channel: —, pass: pass B only, slot: —}` | output dtype (= input dtype) | compute | writer | pass B | **Cannot share.** Zero-copy over the output shard when `NAT_OUT`; under `inplace` that shard **is** the input's, which is the aliasing the caller asked for and not a ledger sharing. Capacity exceeds live set by `DO` — compute↔writer double buffer |
| `cb_output_sticks` | `RM * DS * WC` | `WC` | `{row: streams → 32 sticks per pop, width: spans → WC, channel: —, pass: pass B only, slot: —}` | output dtype | compute | writer | RM only | **Cannot share with `cb_input_sticks`** — concurrent across the pipelined block boundary |
| `cb_sum_handoff` | `CMB * SD * BR` | `BR` | `{row: spans → BR, width: —, channel: —, pass: pass A, slot: streams → this core's own partial}` | float32 | compute | writer | combine only | **Replaces `cb_row_stat` on the combine path** — that is the sharing, and it is why `cb_row_stat` is not allocated there. Capacity `SD` is the D25 pipeline depth |
| `cb_partials_gathered` | `CMB * (TREE ? f0 + f0%2 : GS)` | same | `{row: **streams** — one compact tile carries all BR stats since D27, width: —, channel: —, pass: —, slot: spans → GS or f0}` | float32 | writer | compute | combine only | **Cannot share with `cb_gather_l1`** — a level-0 gatherer is simultaneously a level-1 member, so both rings are live in the same round. Capacity = live set (no depth: deepening the gather ring regressed twice, D25) |
| `cb_stat_handoff` | `CMB * FD` | `1` | `{row: streams — one compact tile, width: —, channel: —, pass: —, slot: —}` | float32 | compute | writer | combine only | **Cannot share.** Capacity `FD = 2` is the one-round-in-flight depth |
| `cb_row_final` | `CMB * SD * BR` | `BR` | `{row: spans → BR, width: —, channel: —, pass: pass B, slot: —}` | float32 | compute | compute | combine only | **Replaces `cb_row_stat` as pass B's stat source** on the combine path — same substitution as `cb_sum_handoff`. Capacity `SD` mirrors `cb_row_stat`'s ring-rotation floor |
| `cb_bank` | `CMP * BR` | `BR` | `{row: spans → BR (one one-hot basis vector per tile-row), width: —, channel: —, pass: —, slot: —}` | **bfloat16** (exact `1.0` at `0x3F80`) | reader | compute | compact combine only | **Cannot share.** Live for the whole kernel and **never popped** — it is a constant basis, not a stream |
| `cb_compact_handoff` | `CMP * FD` | `1` | `{row: streams, width: —, channel: —, pass: —, slot: —}` | float32 | compute | writer | compact combine only | **Cannot share** with `cb_mcast_in` — the outbound partial and the inbound stat are live in the same round |
| `cb_mcast_in` | `CMP * FD` | `1` | `{row: streams, width: —, channel: —, pass: —, slot: —}` | float32 | writer | compute | compact combine only | **Cannot share.** Declared on **every** core in the multicast box, inactive ones included, so its L1 address is identical group-wide — a sharing decision would break that invariant |
| `cb_gather_l1` | `CMB * TREE * (f1 + f1%2)` | same | `{row: streams, width: —, channel: —, pass: —, slot: spans → f1}` | float32 | writer | compute | tree combine only | **Cannot share with `cb_partials_gathered`** (above) |
| `cb_node_out` | `CMB * TREE * FD` | `1` | `{row: streams, width: —, channel: —, pass: —, slot: —}` | float32 | compute | writer | tree combine only | **Cannot share.** Carries a level-0 node's *raw* (unfinalized) run sum while the level-0 ring is still being refilled |
| **`cb_residual_sticks`** | `HAS_R * RM * DS * WC` | `WC` | `{row: streams → 32 sticks per push, width: spans → WC, channel: —, pass: streams, slot: —}` | input dtype | reader | compute | `HAS_R` && RM | **Cannot share with `cb_input_sticks`** — both are operands of one `residual_add_block` and are therefore live at the same instant. Same page format, so the *only* obstacle is concurrency, and it is stated rather than assumed |
| **`cb_residual_tiles`** | `HAS_R * (NAT_R ? shard_h_t*shard_w_t : DR * BR * WC)` | as capacity when zero-copy; `BR*WC` otherwise | `{row: spans → BR, width: spans → WC, channel: —, pass: streams → consumed in pass A, and again in pass B only when !X_RESIDENT, slot: —}` | input dtype | reader | compute | `HAS_R` | **Cannot share with `cb_input_tiles`** — simultaneous operands. Zero-copy over the residual's own shard when `NAT_R` (the residual carries the input's shard spec, so it is already in this core's L1 and must never cross the NoC). Capacity exceeds live set by `DR` — double buffering |
| **`cb_x_sum`** | `HAS_R * BR * XH` | `BR * XH` | `{row: spans → BR, width: spans → XH, channel: —, pass: **spans** → written in A, read in B (this is why its width extent is XH and not WC), slot: —}` | input dtype | compute | compute | `HAS_R`, pass A → pass B | **Takes over `cb_input_tiles`'s held role** — that is the sharing: when `HAS_R`, `cb_input_tiles` drops from `BR*XH` held to `DX*BR*WC` streaming. **Cannot** additionally share with `cb_normalized`, and does not need to: `cb_normalized` is not allocated when `HAS_R`, because pass B packs into `cb_x_sum`'s successor slot and transforms in place |
| **`cb_bias_sticks`** | `HAS_B * PC_RM * WC` | `WC` | `{row: —, width: spans → WC, channel: spans → WC*32 biases, pass: streams, slot: —}` | bias dtype | reader | compute | `HAS_B && PC_RM` | **Cannot share with `cb_gamma_sticks`** — differing page format (independent dtypes) and concurrent staging |
| **`cb_bias_tiles`** | `HAS_B * XH` | `XH` | `{row: **streams** → reuse-shared across every row, width: spans → XH, channel: spans → XH*32, pass: spans → held, slot: —}` | bias dtype | reader (TILE) / compute (RM) | compute | `HAS_B` | **Cannot share with `cb_gamma_tiles`** — differing page format and both live in pass B |

### Audit notes

- **Audit 1 (capacity vs. live set).** Every gap is accounted: `DX`/`DR`/`DO`/`DS` are double
  buffering, `FD` is one-round-in-flight, and `SD` on `cb_row_stat`/`cb_row_final` is the D6
  **correctness** floor rather than overlap. No CB whose live set `spans` an axis has a capacity
  that fails to scale with it — the one that looks like a counterexample, `cb_partials_gathered`,
  genuinely `streams` over `row` since D27 replaced the `GS × BR` ring with one compact tile.
- **Audit 2 (page format vs. DEST width).** Two deliberate `Float32` pages while
  `fp32_dest_acc_en` may be off: `cb_row_stat` / `cb_sum_handoff` / `cb_row_final` (the cross-chunk
  and cross-core accumulators — the wide page is what keeps the *reload* lossless, which is a
  property of the CB round-trip and not of DEST) and the combine's transport CBs (a partial summed
  across up to 16 cores). Both are argued at their rows. Every other CB carries the dtype of the
  tensor it holds, which is what makes the X-09 mixed-dtype cell correct rather than merely
  accepted. `cb_bank` is bfloat16 because an exact `1.0` is all a permutation basis needs.
- **Audit 3 (disjoint lifetime).** The one disjoint pair — `cb_x_squared` (pass A) and
  `cb_normalized` (pass B) — is **not** shared, and the reason is the D25 cross-block pipeline,
  recorded at the row per Rule 3's explicit pipelining clause.
- **Audit 4 (bounds and closed form).** Every symbol above is bounded with its predicate; the
  total below is closed-form in them.

---

## Total per-core footprint

```
arena_bytes =
  # --- activations ---
    RM     * DS * WC * bt                            # cb_input_sticks
  + RM     * DS * WC * bt                            # cb_output_sticks
  + (NAT_IN  ? 0 : DX * BR * (HAS_R ? WC : XH) * bt)  # cb_input_tiles
  + (NAT_OUT ? 0 : DO * BR * WC * bt)                 # cb_output_tiles
  + HAS_R * RM * DS * WC * bt                        # cb_residual_sticks
  + HAS_R * (NAT_R ? 0 : DR * BR * WC * bt)          # cb_residual_tiles
  + HAS_R * BR * XH * bt                             # cb_x_sum
  + BR * XSW * bt                                    # cb_x_squared
  + (!HAS_R) * (HAS_G | HAS_B) * BR * WC * bt        # cb_normalized

  # --- per-channel operands (reuse-shared: no BR term anywhere) ---
  + HAS_G * XH * gt      + HAS_G * PC_RM * WC * gt   # cb_gamma_tiles  + cb_gamma_sticks
  + HAS_B * XH * bit     + HAS_B * PC_RM * WC * bit  # cb_bias_tiles   + cb_bias_sticks

  # --- statistics ---
  + SP * st                                          # cb_scaler
  + (!CMB) * SD * BR * ft                            # cb_row_stat
  + CMB    * SD * BR * ft                            # cb_sum_handoff
  + CMB    * SD * BR * ft                            # cb_row_final
  + CMP    * BR * st                                 # cb_bank

  # --- combine, flat in BR ---
  + CMB * ( (TREE ? (f0 + f0%2) + (f1 + f1%2) + FD : GS) + FD + CMP * 2*FD ) * ft

l1_reserved = shard_bytes(input) + shard_bytes(output) + HAS_R * shard_bytes(residual)
budget      = 0.85 * max(0, usable_L1 - (l1_reserved ? l1_reserved + 70656 : 0))
constraint  : arena_bytes <= budget
```

### Which terms scale with which knob

| Knob | Terms it moves |
|------|----------------|
| `BR` | every activation CB, `cb_x_squared`, `cb_row_stat` / `cb_sum_handoff` / `cb_row_final`, `cb_bank`. **Not** the per-channel operands (reuse-shared) and **not** the combine ring (flat since D27) |
| `WC` | the stick CBs, the streaming tile CBs, `cb_normalized`, `cb_x_squared` (only when the DEST fold is off), the per-channel *stick* CBs |
| `XH` | `cb_x_sum` and the per-channel *tile* CBs — the three buffers that are held across both passes |
| `DX` / `DR` / `DO` / `DS` | the double-buffered streams only |
| `SD` | the fp32 accumulators only — a correctness knob whose cost the budget nonetheless prices |
| `G` / `f0` / `f1` | the combine ring only, and it is `O(G)` tiles rather than `O(G·BR)` |
| `HAS_R` | adds `cb_residual_*` and `cb_x_sum`; **removes** `cb_normalized`; **demotes** `cb_input_tiles` from held (`XH`) to streaming (`DX*WC`) |
| `HAS_B` | adds `cb_bias_tiles` (+ `cb_bias_sticks` on RM). **Adds no block-sized buffer** — the in-place `scale_block` is what buys that |

### Worked deltas at the two extremes

Both at `BR=1`, `bt=gt=bit=2048` (bf16), TILE, `DX=DO=DR=2`, no combine, `XSW=1`.

| Configuration | RESIDENT (`WC = XH = WPC`) | STREAM (`XH = WC`) |
|---|---|---|
| `no_gamma` (the seed's baseline) | `(2·WPC + 1)·bt + …` | `(2·WC + 1)·bt + …` |
| `gamma` | `+ (WC + 1)·bt` (`cb_normalized` + `cb_gamma_tiles` at gt=bt) | same |
| `gamma_bias` | `+ 1·bt` over `gamma` — **only `cb_bias_tiles`** | same |
| `gamma_bias_residual` | `+ (2·WC + WPC)·bt − WC·bt` over `gamma_bias`: adds `cb_residual_tiles` (`2·WC`) and `cb_x_sum` (`WPC`), removes `cb_normalized` (`WC`), and `cb_input_tiles` drops from `2·WPC` to `2·WC` (equal at RESIDENT) | `+ (2·WC + WC − WC)·bt = +2·WC·bt` |

The `gamma_bias` row is the point of the in-place `scale_block`: adding a bias costs **one row
vector**, not one block.

---

## Data-movement budget

For the chosen split. `C` = active cores, `nb` = row-blocks per core, `nc` = `NUM_W_CHUNKS`.
"DRAM crossings" counts how many times the *whole tensor's worth of bytes* moves across DRAM.

### ROWS · RESIDENT / ROW_RESIDENT — the row split (rank 1 of the traffic ranking)

| Tensor | DRAM crossings | Why that many | Cross-core traffic added |
|--------|----------------|---------------|--------------------------|
| `input_tensor` | **1** | held in `cb_input_tiles` (RESIDENT) or in `cb_x_sum` / the held x CB (ROW_RESIDENT) across both passes | none |
| `residual_input_tensor` | **1** | consumed once into `cb_x_sum`, which is what both passes then read | none |
| `weight` | **C** | reuse-shared: it does not vary along `row`, which is the axis the split cuts, so every core reads the identical bytes. Held per core (`X_RESIDENT`), so once each and not once per block | none |
| `bias` | **C** | identical reasoning | none |
| output | **1** | written once | none |

Total ≈ `(2 + HAS_R)·Sx + C·(HAS_G·Sg + HAS_B·Sb)` bytes across DRAM, where `Sx` is the activation
size and `Sg`/`Sb` the operand sizes. Cross-core: **zero**.

### ROWS · STREAM — the fallback

| Tensor | DRAM crossings | Why that many | Cross-core |
|--------|----------------|---------------|-----------|
| `input_tensor` | **2** | nothing is held, so pass B re-reads it | none |
| `residual_input_tensor` | **2** | same — and this is the term that makes the residual expensive in this regime specifically | none |
| `weight` / `bias` | **C · nb · nc** each | chunked and not held, so re-read for every pass-B chunk of every row-block | none |
| output | **1** | | none |

On `(1,1,8192,7168)` with a weight this was measured at 470 MB and 1 043 918 ns ≈ 450 GB/s — the
part's DRAM roofline, i.e. the only lever left was the byte count. ROW_RESIDENT moves 285 MB.
**With a residual the STREAM figure gains another 234 MB** (two more full crossings), which is why
ROW_RESIDENT matters more in this op than it did in the seed.

### ROWS + WIDTH SPLIT — the dependent-axis split (rank 2)

| Tensor | DRAM crossings | Why that many | Cross-core traffic added |
|--------|----------------|---------------|--------------------------|
| `input_tensor` | **1** | each core reads only its `Wt/gw` slice | one compact fp32 tile (4096 B) per member per row-block, up the tree |
| `residual_input_tensor` | **1** | same slice | — |
| `weight` / `bias` | **gh** each (not `C`) | a core reads only the slice of the vector its width slice needs, so the whole vector is read once per *row-group* rather than once per core — **strictly less** than the row split | — |
| output | **1** | | one multicast fp32 tile per row-block, down to `G−1` receivers |

Cross-core total per row-block: `G · 4096` B up (or `(f0 + f1) · 4096` B with the tree, spread
across `f1` cores) + `4096` B multicast to `G−1` receivers.

### The one line that matters

> Cheapest-traffic split considered: **`row` across the grid + `width` across a group** — it ties
> the pure row split on every activation (1 crossing each) and **beats** it on the reuse-shared
> operands by a factor of `C/gh`, at a cost of `G` compact 4 kB tiles up and one multicast tile
> down per row-block. Implemented: **that split**, under the `_auto_width_split` AUTO policy.
> Not universally applied because the `WIDTH_SPLIT_MIN_GAIN = 4` gate was **measured**: at
> `MIN_GAIN = 2` the `(1024,1024)` case split 32 → 80 cores and regressed 0.92×, so a combine
> round that is not paid for by ≥4× more cores at work costs more than the operand bytes it saves.
> Every shape that does split under the shipped gate measured faster, up to 3.46×.

The cheapest-traffic split **is** implemented, so there is no `deferred` regime row carrying a
traffic delta. The two `deferred` rows in `op_design.md` — `GAMMA_MCAST` and
`RAGGED WIDTH CHUNK` — carry their own reachability arguments there: `GAMMA_MCAST` would take the
per-channel crossings from `C` (or `gh`) to `1`, and the width split already removes that term
structurally wherever the grid is under-filled, which is exactly where it would bite hardest.

### Occupancy was not treated as sufficient

Rank 4 of the traffic ranking — one core, whole width — moves the **fewest** total bytes of any
candidate and is nonetheless never chosen. That is the ordering of Rule 2 doing its job: fill the
grid first, then take the coarsest block. Conversely rank 2 is chosen over rank 1 on the decode
profiles *even though it adds cross-core traffic*, because rank 1 there leaves 109 of 110 cores
idle — and it is chosen over rank 1 on the operand traffic even where both fill the grid. Neither
decision was made by counting busy cores alone.
</content>
