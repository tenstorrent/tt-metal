# L1 Ledger: rms_norm_ttnn

Companion to `op_design.md`. Schema and audits: `.claude/references/l1-footprint-discipline.md`.

**Named block axes** (from the Blocking Model): `row`, `width`, `channel`, `pass`,
`width-group slot`. Every row's `Axis accounting` cell names all five, each tagged `spans` or
`streams`.

**Inventory before solve.** The buffer count was minimized before any budget predicate was
consulted. What that produced, concretely:

| Phase boundary | Could it be storage-free? | Decision |
|----------------|--------------------------|----------|
| `normalize_block` → `scale_block` → `bias_block` | yes, for all but the first | `normalize_block` packs into `cb_normalized`; `scale_block` **transforms it in place**; `bias_block` **packs into the destination** `cb_output_tiles`. Patterns 1+2 of Rule 3: three stages, ONE block-shaped buffer between them. **What in-place actually costs, stated honestly:** the in-place chain ROTATES `cb_normalized`'s ring (pop `PASS_B_BLK`, then reserve `PASS_B_BLK`), so a block advances the front by `rows*WC`. That is a whole revolution for a FULL block but not for the partial final one, and `bias_block`'s bulk wait + linear indexing would then straddle the wrap — D6's hazard on a different CB. So `cb_normalized` is **two blocks deep when `BR > 1` and both stages are present** (`_norm_cb_depth`), which is the same L1 a dedicated `cb_scaled` would have cost. In-place is therefore not an L1 win over the fallback at `BR > 1`; it wins at `BR == 1` (every ROW_RESIDENT / STREAM / BAND / width-shard build), where no block can be partial and depth 1 is exactly correct. Lamp L-BIAS-INPLACE is the measurement that would decide the `BR > 1` case on ns rather than bytes |
| `residual_add_block` → `square_block` → `normalize_block` | no | `cb_x_sum` is genuinely required: `t = x + r` is read by *both* passes, and fusing the add into pass B's broadcast multiply is inexpressible (`DestReuseBinary` carries no broadcast parameter, `chain.hpp:526`). Pattern 4 (fold into DEST) covers the pass-A half only — recorded as Lamp L-RES-FUSE, not as a buffer |
| `normalize_block` packing back into `cb_x_sum` when `HAS_R` | **no — and `op_design.md`'s CB table is wrong here** | The design's `cb_normalized` row says it is "allocated only when `!HAS_RESIDUAL`" because "with a residual, `cb_x_sum` plays this role too". It cannot: `cb_x_sum` is pass B's *srcA* at `WaitPolicy::Upfront` / `PopPolicy::None` (it is the HELD operand, indexed at a tile base), and an in-place chain requires an **incrementally popping** input and an **incrementally reserving** output — `chain.inl:82-85` and `inplace_chain.cpp:5-21`. Aliasing it would DEADLOCK on the packer's reserve rather than return a wrong answer. The design's own Key-Risks row says the same thing ("`cb_x_sum` … is never aliased"), so the two statements contradict each other and this ledger follows the risk row. `cb_normalized` is therefore allocated whenever `HAS_G \| HAS_B`, residual or not |
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
| `WC` | `WT_CHUNK`, width tiles per block | `1 ≤ WC ≤ WPC`, and **`WC | WPC`** (D1) — or, since D33, `NC·WC ≥ WPC` with the last chunk padded, which keeps the chunk UNIFORM and so satisfies all three mechanisms verbatim | D1: `tilize`/`untilize` take `block_width_tiles` as a compile-time template param; `reduce`'s `BulkWaitBulkPop` asserts `num_pages % cols == 0`; a multi-page reserve must not straddle the ring. D33 pads rather than raggedizes precisely so none of the three moves |
| `NC` | `NUM_W_CHUNKS`, width chunks per row-block | `NC = ceil(WPC / WC)`; `NC·WC = WPC` exactly whenever `WP == 0` | `_width_chunk` |
| `WPC` | `wt_per_core`, this core's width tiles | `WPC = Wt` (row split) or `Wt/gw` (width split) or `shard_w_t` (shard) | `_plan_placement` |
| `WP` | `wt_pad` — pad width tiles in the LAST chunk (D33) | `WP = NC·WC − WPC`; `0` on every divisor build, and `WP < NC` by construction | `_width_chunk` → reader CT 29 / writer CT 2 high half |
| `XH` | `x_hold_wt` | `XH = NC·WC` (the PADDED per-core width) when `X_RESIDENT`, else `XH = WC`. Equals `WPC` whenever `WP == 0` — i.e. everywhere but a ragged chunk | `_solve_blocking` |
| `XSW` | `X_SQUARED_WT` | `XSW ∈ {1, WC}` | `1` iff `PARTIAL_W == 0 and WC ≤ DEST_ACC_SQUARE_MAX_WT (8)`; host `assert XSW in (1, WC)` |
| `DX`, `DO` | `CB_X_DEPTH`, `CB_OUT_DEPTH` | `∈ CB_DEPTH_CANDIDATES = (2,)` on TILE; forced to `1` on ROW_MAJOR | the regime search walks the candidates coarsest-first; RM's producer/consumer is a sequential tilize/untilize so depth buys no overlap |
| `DR` | `CB_R_DEPTH` | `DR = DX` | tied by construction so one knob moves both streams (Lamp L-RES-DEPTH is the measurement that could untie it) |
| `DS` | `CB_RM_STAGE_DEPTH` | `= 2`, **searched down to 1 on the BAND scheme** | primary knob; the band branch of `_solve_blocking` walks `(2, 1)` because a band has THREE activation staging rings and its block is already pinned at one tile-row, so the ring depth is the only extent left to give back. Local-L1 reads are the cheapest overlap in the op to sacrifice |
| `ND` | `_norm_cb_depth(HAS_G, HAS_B, BR)` — blocks of `cb_normalized` | `0` when no post-stage; `1` with exactly one, or with both at `BR == 1`; `2` with both at `BR > 1` | the in-place `scale_block` rotates the ring by `rows*WC` per block; `2` is the D6-class contiguity floor for a PARTIAL final block, and at `BR == 1` no block is ever partial |
| `PS` | per-channel staging pages (`pc_stage_pages`) | `WC` normally; `DS` under the D30 narrow fallback | D30: a per-channel operand is ONE stick, but a `tilize<WC>` ring must reserve 32 rows' worth of pages to carry it. The narrow form stages one tile COLUMN per page and consumes it as `tilize<1>(WC)` — bit-identical tiles at `1/WC` of the L1. Taken only when the budget asks (the band search's second axis) |
| `SD` | `CB_ROW_STAT_DEPTH` | `= 2` | **correctness floor**, not a perf depth (D6) |
| `FD` | `CB_COMBINE_FLAT_DEPTH` | `= 2` | primary knob; one round in flight |
| `SP` | `scaler_pages` | `∈ {1, 2}` | `2` iff `kernel_partial_w != 0` |
| `G` | `group_size` | `1 ≤ G ≤ WIDTH_SPLIT_MAX_GROUP_CORES (16)` on the interleaved width split; `= shard grid extent` on a WIDTH/BLOCK shard, itself `≤ grid.x * grid.y` | `_auto_width_split` / `_plan_placement` |
| `GS` | `GATHER_SLOTS` | `= G + G%2 ≤ G+1` | D22's pairwise DEST walk needs an even window |
| `f0`, `f1` | combine-tree fan-ins | `f0` = the largest **divisor** of `G` in the measured band `[COMBINE_TREE_F0_MIN, _MAX] = [4, 10]` that both gates admit (the cap itself as a ragged fallback); `f1 = ceil(G/f0) ≥ 2`; `f0*f1 ≥ G` | `_combine_tree_arity`; returns `None` (flat) when no candidate has `f1 ≥ 2` and deletes enough fold tiles. Refinement 1 lever 1 made `f0` derived; it never enlarges the two rings (`f0 + f1` is minimised near `sqrt`-ish arities), and at `G = 64` it SHRINKS them from `4 + 16` to `8 + 8` pages = 32 KB/core given back |
| `bt` | `tile_size(input.dtype)` — the TENSORS' format (`cb_input_*`, `cb_residual_*`, `cb_output_*`) | `∈ {1088 (bf8b), 2048 (bf16), 4096 (fp32)}` | dtype ∈ SUPPORTED |
| `it` | `tile_size(_intermediate_dtype(input.dtype))` — the COMPUTE INTERMEDIATES' format (`cb_x_sum`, `cb_x_squared`, `cb_normalized`) | `= bt` at bf16 / fp32; `= 2048` (bf16) at bf8b | a value the kernel itself produced is never re-quantized to a block float — three stacked block-float roundings measured 1.39e-2 relative Frobenius against bf16's 1.01e-2 on the residual+gamma bf8b cell, either side of the reference suite's 1.6e-2 threshold |
| `gt`, `bit` | `tile_size(weight.dtype)`, `tile_size(bias.dtype)` | same set; **independent of `bt` and of each other** | operand dtype ∈ SUPPORTED |
| `st`, `ft` | `tile_size(bf16)` = 2048 (`cb_bank`), `tile_size(fp32)` = 4096 | constants | — |
| `sct` | `tile_size(scaler_dtype)`; `scaler_dtype = it` when `PARTIAL_W != 0`, else bf16 | `∈ {2048, 4096}` | the AccumulateViaAdd partial fold reads the 0/1 mask through the reduce INPUT's unpack format (`reduce_accumulate_via_add` reconfigs BOTH operands from the input CB), so the mask must be WRITTEN in that format or its lanes land at the wrong pitch |
| `budget` | bytes the CBs may take | `L1_SAFETY_FRACTION (0.85) * max(0, usable_L1 − l1_reserved − L1_CB_ARENA_BASE_RESERVE)` where the last two terms apply only when a shard is resident | `_solve_blocking`; `l1_reserved` sums the **distinct** resident tensors among {input, output, residual} — `inplace` makes the output *be* the input, so charging both would price one buffer twice |

`HAS_G`, `HAS_B`, `HAS_R` are `0/1` compile-time presence flags; `RM`, `PC_RM`, `NAT_IN`,
`NAT_OUT`, `NAT_R`, `CMB`, `CMP` (compact = `CMB and BR>1`), `TREE` are `0/1` build flags.

---

## The table

| CB | Capacity (pages) | Live set | Axis accounting | Page format | Producer | Consumer | Lifetime | Shares with / why not |
|----|------------------|----------|-----------------|-------------|----------|----------|----------|-----------------------|
| `cb_input_sticks` | `RM * DS * WC` | `WC` (one tile-row's staging) | `{row: streams → 32 sticks per push, width: spans → WC, channel: —, pass: streams → re-staged per pass when !X_RESIDENT, slot: —}` | input dtype | reader | compute | whole kernel, RM only | **Cannot share.** Concurrent with `cb_output_sticks` (the untilize of block *b* overlaps the tilize of *b+1*) and with `cb_residual_sticks` (both feed one `residual_add_block`). Capacity exceeds live set by `DS` — the reader↔tilize double buffer |
| `cb_input_tiles` | `NAT_IN ? shard_h_t*shard_w_t : DX * BR * (HAS_R ? WC : XH)` | same as capacity when zero-copy; `BR*(HAS_R?WC:XH)` otherwise | `{row: spans → BR, width: spans → WC or XH, channel: —, pass: HAS_R ? streams : spans (held across both passes when X_RESIDENT), slot: —}` | input dtype | reader | compute | whole kernel | **Cannot share.** Zero-copy over the input shard when `NAT_IN` (0 arena bytes, and aliasing anything else would corrupt the caller's tensor). Otherwise concurrent with `cb_output_tiles` by the `DX`/`DO` pipelining decision. Capacity exceeds live set by `DX` — double buffering |
| `cb_x_squared` | `BR * XSW` | `BR * XSW` | `{row: spans → BR, width: spans → XSW (=1 under the DEST fold, else WC), channel: —, pass: pass A only, slot: —}` | intermediate dtype (`it`) | compute | compute | pass A | **Could share with `cb_normalized`** (pass A vs. pass B, disjoint) — **not taken**: the D25 combine pipeline issues block `b+1`'s pass A before block `b`'s combine, which makes the two concurrent. Recorded per Rule 3's pipelining clause |
| `cb_scaler` | `SP` (1 or 2) | `SP` | `{row: —, width: —, channel: —, pass: —, slot: —}` — constant | **bfloat16** when `PARTIAL_W == 0`; the INTERMEDIATE format `it` when `PARTIAL_W != 0`, because the partial mask is unpacked in the reduce input's format | reader | compute | whole kernel | **Cannot share.** Live for the whole kernel; `1.0` exactly, never `1/W` |
| `cb_row_stat` | `!CMB * SD * BR` | `BR` (one block's stats) | `{row: spans → BR, width: —, channel: —, pass: spans → written in A, read in B, slot: —}` | **float32 always** — deliberately overrides the "page format follows DEST width" default, because this is the cross-chunk accumulator `reduce`'s `Accumulate::at` reloads; a 16-bit page would make the STREAM reload lossy at exactly the widths this op cares about | compute | compute | pass A → pass B | **Cannot share.** Capacity is `SD =` 2× the live set and that gap is a **correctness** requirement, not double buffering: `transform_in_place` rotates the ring and a partial final block's finalized tiles would otherwise straddle the wrap (D6). Not allocated at all on a combine path **unless `FIN_SPREAD`** (Refinement 2), where it comes back as the spread finalize's `FD`-page output — flat in `BR`, because D27 made the round's unit ONE compact tile. That re-use IS a sharing (the same index, disjoint lifetimes: pass-A accumulator off the combine path, post-multicast finalize on it) and it is why the spread needs no new buffer index. `FIN_SPREAD` is parked at 0, so the term is 0 in every shipped build |
| `cb_gamma_sticks` | `HAS_G * PC_RM * PS` | `PS` | `{row: —, width: spans → WC (as `PS = WC`) or **streams** → one tile column per page under D30, channel: spans → WC*32 weights, pass: streams, slot: —}` | weight dtype | reader | compute | `HAS_G && PC_RM` | **Cannot share with `cb_bias_sticks`** — differing page format (the two operands' dtypes are independent) and concurrent (both staged per chunk before their tilizes). Capacity `PS` is the D30 knob: `WC` for one wide `tilize<WC>(1)`, or `DS` pages for `tilize<1>(WC)` |
| `cb_gamma_tiles` | `HAS_G * XH` | `XH` | `{row: **streams** → the same vector feeds every row (reuse-shared), width: spans → XH, channel: spans → XH*32, pass: spans → held across both, slot: —}` | weight dtype | reader (TILE build) / compute (RM build) | compute | `HAS_G` | **Cannot share with `cb_bias_tiles`** — differing page format, and both are live simultaneously in pass B |
| `cb_normalized` | `ND * BR * WC` (`ND = 0` when neither post-stage is present) | `BR * WC` | `{row: spans → BR, width: spans → WC, channel: —, pass: pass B only, slot: —}` | intermediate dtype (`it`) | compute | compute | pass B | **Cannot share with `cb_x_squared`** — the D25 pipeline makes them concurrent (above). **Cannot share with `cb_x_sum`** either, and the design's contrary claim is refuted in the inventory table above: `cb_x_sum` is pass B's held `Upfront`/`None` srcA and an in-place chain needs an incrementally-popping input, so aliasing it deadlocks. Capacity `ND` is the in-place rotation floor, not double buffering |
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
| **`cb_x_sum`** | `HAS_R * BR * XH` | `BR * XH` | `{row: spans → BR, width: spans → XH, channel: —, pass: **spans** → written in A, read in B (this is why its width extent is XH and not WC), slot: —}` | intermediate dtype (`it`) | compute | compute | `HAS_R`, pass A → pass B | **Takes over `cb_input_tiles`'s held role** — that is the sharing: when `HAS_R`, `cb_input_tiles` drops from `BR*XH` held to `DX*BR*WC` streaming. **Cannot** additionally share with `cb_normalized` — see the inventory table: it is pass B's HELD srcA (`Upfront`/`None`, indexed at a tile base), which is exactly the operand shape an in-place chain forbids as its output. Depth 1 is nonetheless correct: a FULL block's push/pop is a whole ring revolution, and only a core's LAST block can be partial, so the D6 straddle cannot arise |
| **`cb_bias_sticks`** | `HAS_B * PC_RM * PS` | `PS` | `{row: —, width: spans → WC or **streams** under D30, channel: spans → WC*32 biases, pass: streams, slot: —}` | bias dtype | reader | compute | `HAS_B && PC_RM` | **Cannot share with `cb_gamma_sticks`** — differing page format (independent dtypes) and concurrent staging. Same `PS` knob as gamma's |
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
  + HAS_R * BR * XH * it                             # cb_x_sum      (INTERMEDIATE format)
  + BR * XSW * it                                    # cb_x_squared  (INTERMEDIATE format)
  + ND * BR * WC * it                                # cb_normalized (INTERMEDIATE format)

  # --- per-channel operands (reuse-shared: no BR term anywhere) ---
  + HAS_G * XH * gt      + HAS_G * PC_RM * PS * gt   # cb_gamma_tiles  + cb_gamma_sticks
  + HAS_B * XH * bit     + HAS_B * PC_RM * PS * bit  # cb_bias_tiles   + cb_bias_sticks

  # --- statistics ---
  + SP * sct                                         # cb_scaler
  + (!CMB) * SD * BR * ft                            # cb_row_stat
  + CMB    * SD * BR * ft                            # cb_sum_handoff
  + CMB    * SD * BR * ft                            # cb_row_final
  + CMP    * BR * st                                 # cb_bank

  # --- combine, flat in BR ---
  + CMB * ( (TREE ? (f0 + f0%2) + (f1 + f1%2) + FD : GS) + FD + CMP * 2*FD ) * ft
  + CMB * FIN_SPREAD * FD * ft                       # cb_row_stat, re-used as the
                                                     # spread finalize's landing CB
                                                     # (Refinement 2; FIN_SPREAD = 0 in
                                                     #  every shipped build, so 0 bytes)

l1_reserved = sum of shard_bytes over the DISTINCT resident tensors among
              {input, output, residual}          # `inplace` makes output IS input,
                                                 # and double-charging one buffer
                                                 # would shrink the block for nothing
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
| `HAS_B` | adds `cb_bias_tiles` (+ `cb_bias_sticks` on RM). Adds **no** block-sized buffer at `BR == 1` — the in-place `scale_block` is what buys that — and **one** at `BR > 1`, where the in-place ring needs `ND = 2` for the partial-block contiguity floor |
| `ND` | `cb_normalized` only |
| `PS` | the per-channel *stick* CBs only |

### Worked deltas at the two extremes

Both at `BR=1`, `bt=gt=bit=2048` (bf16), TILE, `DX=DO=DR=2`, no combine, `XSW=1`.

| Configuration | RESIDENT (`WC = XH = WPC`) | STREAM (`XH = WC`) |
|---|---|---|
| `no_gamma` (the seed's baseline) | `(2·WPC + 1)·bt + …` | `(2·WC + 1)·bt + …` |
| `gamma` | `+ (WC + 1)·bt` (`cb_normalized` + `cb_gamma_tiles` at gt=bt) | same |
| `gamma_bias` | `+ 1·bt` over `gamma` at `BR == 1` — **only `cb_bias_tiles`**; `+ (WC + 1)·bt` at `BR > 1`, where `ND` goes 1 → 2 | same (`BR == 1` by construction in STREAM) |
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

Total ≈ `(2 + HAS_R)·Sx + C·(HAS_G·Sg' + HAS_B·Sb')` bytes across DRAM, where `Sx` is the activation
size and `Sg'`/`Sb'` the operand **bytes actually fetched**. Cross-core: **zero**.

> **`Sg'` is not `Sg`, and the difference is 16× — added by the verifier, because every
> figure below this line was written against whole tiles.** A per-channel operand is a
> `(1,1,1,W)` vector whose only consumer reads TILE ROW 0, so D23 trims the TILE-layout read
> to **two face-rows**: `GAMMA_TRIM = 2` fetches `2 · TILE_DIM · elem_bytes` = **128 B of a
> 2048 B bf16 tile** (`GAMMA_TRIM = 1`, the bf8b demotion, fetches half a page). So
> `Sg' = Sg / 16` at bf16 TILE, `Sg / 2` at bf8b, and `Sg' = Sg` only for a ROW_MAJOR
> operand, which is one stick and already minimal. The whole-tile figures quoted in the two
> sections below (and the `118 MB` / `50 MB` gamma numbers in `op_design.md`'s `GAMMA_MCAST`
> row and D14) predate that trim and **over-count the reuse-shared term by that factor**.
> Corrected: on `(1,1,8192,7168)` ROW_RESIDENT the operand term is ~3 MB, not 50 MB, against
> 234 MB of x+out — i.e. ~1%, so `GAMMA_MCAST`'s deferral is *more* clearly right than its
> own row argues, not less. Nothing about the chosen split changes; only the price of the
> deferred alternative does.

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
| output | **1** | | one multicast fp32 tile per row-block, down to `G−1` receivers — **3072 B on the identity path since Refinement 2** |

Cross-core total per row-block: `G · 4096` B up (or `(f0 + f1) · 4096` B with the tree, spread
across `f1` cores) + `4096` B multicast to `G−1` receivers — **`3072` B when `BLOCK_ROWS == 1`**.

Refinement 2 (Lamp L-FIN) is the only change so far that moves a byte count here, and it moves
exactly one: on the IDENTITY path (`BLOCK_ROWS == 1`) the stat multicast carries faces 0..2 rather
than the whole tile — 3072 B instead of 4096 B to each of `G−1` receivers — because the landing CB's
only reader there is pass B's column broadcast (column 0 lives in faces 0 and 2), which is D26's
already-measured licence applied to the other direction. The COMPACT path is unchanged at 4096 B:
its un-permute matmul sums 32 products, so every column must be finite. The gather's byte counts
are untouched in both directions.

Refinement 1 lever 2 changes **which NoC** those bytes ride, never how many there are. On a
`native_in` plan the reader has no activation stream at all (x is an aliased resident shard), so the
whole combine moves to **NOC_0** and the reader takes NOC_1 — a swap of both kernels, because the
two data-movement RISCs must not share one engine. On a streamed plan the reader owns every
activation byte and NOC_0 stays its; see `_combine_noc`. Byte counts in every row above are
unaffected.

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

---

## Deviations from `op_design.md`, recorded here because this is where they are priced

Five, all in the advisory half of the design (CB sizing / knob selection); the scheme, topology,
work split and helper mapping are unchanged.

| # | Deviation | Why | Effect on an operand-free build |
|---|-----------|-----|--------------------------------|
| **D29** | `cb_normalized` is allocated whenever `HAS_G \| HAS_B`, at `ND` blocks deep, rather than "only when `!HAS_RESIDUAL`" | The design's CB table wants pass B to pack back into `cb_x_sum`; that CB is pass B's HELD `Upfront`/`None` srcA and an in-place chain requires an incrementally-popping input (`chain.inl:82-85`, `inplace_chain.cpp:5-21`), so the alias would DEADLOCK. The design's own Key-Risks row says `cb_x_sum` is never aliased, so the two statements contradict and this follows the risk row. The `ND = 2` depth is the same D6-class contiguity floor: the in-place `scale_block` rotates the ring by `rows*WC`, which is a whole revolution only for a full block | **none** — `ND = 0` with no operand, `1` with gamma only, which is byte-identical to the seed |
| **D30** | A ROW_MAJOR per-channel operand's staging ring can be `DS` pages instead of `WC` (`tilize<1>(WC)` instead of `tilize<WC>(1)`) | The design gives the BAND scheme no L1 fallback ("metal is the arbiter"). At the seed's two operands that held; a third activation ring plus `cb_x_sum` plus a second per-channel operand does not. **Measured**: `(128, 8192)` fp32 ROW_MAJOR BLOCK_SHARDED with `gamma_bias_residual` (a 13×748 shard on an 11×10 grid, `WC = 25`) built 1 406 976 B of CBs against a 1 344 512 B ceiling — a 62 464 B overshoot, i.e. a hard launch failure on 2 golden cells. A per-channel operand is ONE stick; the wide ring reserves 25 whole fp32 tiles (100 kB) to carry 3 200 B of it, twice over. The narrow form costs `WC` LLK block calls instead of one, paid once per core in the resident regimes | **none** — the band search takes it only after the wide form and both ring depths have failed the budget; every band build that already fit takes candidate 1 |
| **D31** | `DS` is searched `(2, 1)` on the BAND scheme | Same cause. It is ordered BEFORE D30 because a band's activation reads come from the core's own L1, so the overlap it gives up is the cheapest in the op | **none** — same reason |
| **D33** | `WC` may be the coarsest **fitting** chunk with the last chunk PADDED, not the coarsest fitting **divisor**; every held CB then spans `NC · WC` (the padded width) rather than `wt_per_core` | D1's divisor clamp is a granularity cliff at a prime `Wt`: 127 has no divisor between 1 and 127, so any cap below the whole row collapses `WC` to 1 and repays every per-phase init / reconfig / pipeline fill-and-drain 127 times per block. The padded form keeps the chunk UNIFORM, so all three of D1's mechanisms hold verbatim and the compute kernel is byte-for-byte unchanged. **L1 cost**: `WP = NC·WC − wt_per_core` extra tiles on the held CBs only, and `WP < NC` by construction — ONE tile in 128 at `Wt = 127`. The L5 solve re-caps against its own pad (the held CBs grew), and falls back to D1 only if no padded candidate fits. **Measured** 1.34–10.46× on the prime-`Wt` resilience shapes, 1.05× on `(1,1,1024,16384)` STREAM, at better pcc | **none** — `WP = 0` on every width whose coarsest fitting chunk is already a divisor, which is every `INPUTS` and every perf-group shape |
| **D32** | The D25 combine pipeline (`PIPE_A`) is gated off when `HAS_R` | The hoisted pass A for block `b+1` would write `cb_x_sum`, whose ring is ONE block deep and whose front block `b`'s pass B still owns — so the hoist would either overwrite live data or self-deadlock on the reserve. Making it legal needs `cb_x_sum` at depth 2 **and** a runtime tile base on the *pack*, and `output(...)` carries no tile base, so it is not expressible without a new chain seam. Recorded as a follow-up with the measurement to take, not as a finished trade | **none** — `PIPE_A` is unchanged without a residual |

### The band worked example, in full

At the failing cell above (`WC = 25`, `BR = 1`, `bt = gt = bit = 4096`, `G = 11`, flat combine,
`XSW = WC` because `WC > DEST_ACC_SQUARE_MAX_WT`), in fp32 tiles:

| Term | wide staging, `DS = 2` | wide, `DS = 1` | **narrow (`PS = DS = 1`)** |
|------|------------------------|----------------|---------------------------|
| 3 activation stick rings (`3·DS·WC`) | 150 | 75 | 75 |
| `cb_input_tiles` + `cb_residual_tiles` + `cb_x_sum` + `cb_x_squared` + `cb_output_tiles` (`5·WC`) | 125 | 125 | 125 |
| `cb_normalized` (`ND·WC`, `ND = 1` at `BR = 1`) | 25 | 25 | 25 |
| per-channel tiles (`2·XH`) | 50 | 50 | 50 |
| per-channel sticks (`2·PS`) | 50 | 50 | **2** |
| combine (`SD·BR·2 + GS + FD`) | 18 | 18 | 18 |
| **total tiles** | 418 | 343 | **295** |
| **bytes** | 1 712 128 | 1 404 928 | **1 208 320** |
| vs. the 1 344 512 B hard ceiling | **+368 kB** | **+60 kB** | **−136 kB** |

`ND = 1` here is D29's `BR == 1` case doing real work: at the design's flat `2` this column would
be 320 tiles / 1 310 720 B, still inside the ceiling but only by 34 kB.

---

## Measured cost of each operand

Blackhole p150b, 110-core grid, bf16 / HiFi2 / `fp32_dest_acc_en=False`, interleaved DRAM, TILE,
one fresh-cache profiled run per variant (`--profile`, DEVICE KERNEL DURATION ns).

| Shape | `no_gamma` seed → ttnn | `gamma` seed → ttnn | `gamma_bias` | `residual` | all three |
|-------|------------------------|---------------------|--------------|-----------|-----------|
| `(1,1,32,1024)` decode | 3955 → 3899 (**1.01×**) | 4859 → 4884 (**1.00×**) | 5829 | 4605 | 6427 |
| `(1,1,8192,1024)` prefill | 84741 → 81493 (**1.04×**) | 89287 → 89544 (**1.00×**) | 96349 | 125387 | 140905 |
| `(1,1,32,7168)` wide decode | 7455 → 7499 (**0.99×**) | 9180 → 9248 (**0.99×**) | 10744 | 9337 | 12670 |

**Seed parity holds** on every operand-free and gamma-only cell: 0.99×–1.04×, inside the ~2% noise
band.

But a perf ratio is only *evidence* for "the operand-free program is the seed's", so that claim is
also **asserted structurally**, deterministically, on the host —
`test_rms_norm_ttnn_perf.py::test_program_is_structurally_the_seeds` builds BOTH descriptors from
the same tensors and compares the CB set page-for-page (`{index → (total_size, page_size)}` — the
whole L1 footprint and the whole blocking decision made visible) plus the CT args. 28 cells over 14
geometries, one per internal scheme: row split, interleaved width split, HEIGHT (local reduce),
WIDTH identity **and** compact, BLOCK, the ROW_MAJOR BAND, the masked-reduce shapes and the
L1-tight wide ones. All identical.

The three kernels are compared differently, and the asymmetry is structural:
the **writer** takes no operand, so not one of its args may move (asserted byte-identical); the
**compute** kernel carries no accessor block, so the seed's 19 args are a plain prefix of this op's
23; the **reader**'s scalars are followed by `TensorAccessorArgs` BLOCKS, so the operands' scalars
necessarily sit *before* them — its two halves are checked separately (scalars 0..20 identical at
the seed's own indices, and the seed's `(x, gamma)` accessor blocks the leading blocks of this op's
`(x, gamma, bias, residual)`).

Where the numbers do differ, the attribution is known rather than guessed: on the two SMALLEST
width-sharded decode shapes — `(1,1,32,7168)` 28c and `(1,1,32,1024)` 8c, 3.8–5.8 µs kernels — the
op runs 1.2–2.0% slower than the seed across three runs each, while the CB set and every CT arg are
**identical** (verified above) and the larger shapes are at parity to 0.1–0.3%. With the program
proven the same, the residue is kernel-binary size (this file's kernels carry the bias / residual /
program-config branches even where `if constexpr` elides them), i.e. i-cache fill on a kernel short
enough to notice it. The only way to remove it would be to split the kernels per operand set, which
trades one 1.5% decode regression for a combinatorial build matrix.

**The residual is at the DRAM roofline on the prefill profile**: it takes the activation crossings
from 2 (x in, out) to 3 (x, r, out), i.e. 1.50× the bytes, and measures 1.42× the time
(88 598 → 125 387 ns). There is no lever there — the bytes are the wall. On the decode profiles it
is FLAT to slightly faster (4884 → 4605, 9248 → 9337), because those shapes are latency-bound on
the width-split combine rather than byte-bound.

**The bias costs 1.09×–1.23×**, largest at decode where a whole extra chain over the block is not
hidden behind DRAM. Lamp L-OPERAND-TRIM (`BIAS_TRIM ∈ {0, GAMMA_TRIM}`) is **TAKEN and CLOSED in
Refinement 3**: sweeping each operand's granularity independently, everything coarser than D23's
derived two-face-row form LOSES — half page 0.93–1.00×, whole tile **0.76–0.96×**, and
`gamma=2 / bias=0` 0.85–1.00×. So the trim is a BYTE-count win, not a transaction-count one, and
two trimmed reads per chunk instead of one does not flip it. `BIAS_TRIM` copying `GAMMA_TRIM`'s
policy — derived from its OWN tile size — is the measured optimum; both are now overridable
(`PER_CHANNEL_TRIM_GAMMA` / `_BIAS`) and both ship derived.

### Levers taken and refused on the residual path

| Lever | Verdict | Numbers (median of THREE fresh-cache profiled runs per variant) |
|-------|---------|------------------------------------------------------------------|
| `cb_x_sum`'s pack policy: `Upfront`/`AtEnd` instead of pass B's `PerBlockSize` pair | **TAKEN** | `(1,1,8192,1024)` gamma_bias_residual 139 758 → 136 361 (**1.025×**, and the two run distributions are *disjoint*, which is why 2.5% is reportable on a shape whose spread is ~2%); `(1,1,32,5120)` WIDTH 32c 6 824 → 6 730 (**1.014×**); residual-only prefill 125 154 → 124 890 (1.002×, flat); `(1,1,7168,1024)` BLOCK 64c 33 822 → 33 876 (0.998×, flat). cb_x_sum is compute-private and BOTH its readers wait `Upfront`, so the incremental page handover D21 kept for pass B (whose consumer is the ROW_MAJOR `untilize`) buys nothing here — at `WT_CHUNK = 32` / `PASS_B_BLK = 8` it was 4 reserve/push pairs per (block, chunk) where 1 does. It wins where a bias makes pass B long enough for pass A's own flow control to be visible, and is flat where the shape is DRAM-bound |
| **L-RES-FUSE** — fuse `residual_add_block` + `square_block` into ONE DEST window | **BUILT AND MEASURED in Refinement 3; the multi-`PackTile` reading below was WRONG, and the correct form LOSES** | The row's original claim — that `chain.inl`'s support for multiple `PackTile` elements makes `BinaryFpu<Add> -> PackTile<cb_x_sum> -> Square -> PackTile<cb_x_squared>` legal, keeping `t` AND dropping a pack+unpack — is **false**, and the failure is silent rather than structural. In `eltwise_chain` **pack is its own cohort**, disjoint from math-MOP/SFPU (`chain.inl elem_pack_init`), so EVERY pack in a chain runs after EVERY compute element: the first `PackTile` publishes the SQUARE into `cb_x_sum` and pass B normalizes `t^2`. Built and measured at **pcc 0.260** on `(1,1,8192,5120)` ROW_RESIDENT `gamma_bias_residual`, and 0.947x, so it was not even fast. Publishing `t` and squaring it in one DEST window would need a DEST->DEST copy element the chain does not expose. `op_design.md`'s STREAM-only framing was therefore RIGHT: `t` need not survive only where pass B rebuilds it. That three-element form (`Add -> Square -> Pack`) is shipped as the `RES_FUSE` knob, is correct (pcc 0.999980), and measures **0.989x** on `(1,1,1024,16384)` STREAM `gamma_bias_residual` -- the SFPU `square_tile` costs more than the saved unpack and pack. **Parked at 0, kept live.** The anti-correlation the row identified (the fold is on exactly where the op is latency-bound) still holds and is why the knob's reach is small either way |
| **D32** — reinstating the D25 combine pipeline with a residual | **REFUSED as inexpressible at bounded L1**, and the cost measured at zero | The hoist needs a TWO-BLOCK sliding window in `cb_x_sum` at a tile offset. A ring cannot hold one: the front advances by exactly one block per iteration, so for any ring size `k·BR·XH` the window straddles the wrap at `f = (k−1)·BR·XH`. The seed's `cb_input_tiles` escapes this only because under `NATIVE_IN` it is the whole shard (every page popped once, monotonically, never wrapping) — so the equivalent for `cb_x_sum` is a buffer sized to the whole per-core assignment, i.e. a second resident shard's worth of L1. **Measured cost of giving the pipeline up:** `(1,1,7168,1024)` BLOCK-sharded 64c with `gamma_bias_residual` runs **33 813 ns against the feature spec's 34 569 ns achievable** — it beats the reference *without* the pipeline, so there is nothing to buy back |


---

## Refinement 3 — no inventory change, and why

Refinement 3 added **no CB, resized none, deleted none, and moved no DRAM crossing**: every one of
its five knobs ships at a byte-identical default except `PASS_A_SQ_BLOCK`, which changes only the
DEST-lane block size and the reserve/push *granularity* of a pack into `cb_x_squared` — not that
CB's page count, its format, its producer, its consumer or its lifetime. The table above and the
data-movement budget therefore stand unchanged.

One knob *would* have moved the table and is parked because of it. **`CB_SQ_EXACT`** corrects
`_cb_block_mult`'s over-pricing of `cb_x_squared`: under the D12 DEST fold that CB is
`BLOCK_ROWS × 1` tiles, but the solve charges it the full chunk width. The error is conservative —
it can only shrink `BLOCK_ROWS`, never overflow L1 — and correcting it does admit a coarser block
where the fold is on and L1 binds: `(1,1,8192,1024)` BLOCK `[1024,128]` 64c goes `BLOCK_ROWS`
20 → 25 (CB region 1 165 → 1 309 kB), `(1,1,7168,1024)` 11 → 12, `(1,1,1024,512)` WIDTH 21 → 25.
It measures **0.987×** on the first of those — one of the two shards the verifier flagged for thin
margin — so the coarser block costs more in ring pressure and pipeline fill than the fewer combine
rounds buy. The conservative price ships (`CB_SQ_EXACT = 0`), which is also what keeps
`test_program_is_structurally_the_seeds` byte-identical; the exact price stays a live knob with
this number attached.

**The prefill's data-movement budget is now confirmed against the machine rather than a nominal
roofline.** For `(1,1,8192,W)` INTERLEAVED bf16 the op moves exactly `2 · R · W · 2` bytes plus the
per-channel row, and at that traffic it runs at **384 GB/s** (W=1024) and **409 GB/s** (W=7168)
against `ttnn.clone` of the same tensors at 400 and 398 GB/s — i.e. the wide prefill is 3% faster
than a pure DRAM→DRAM copy. There is no byte left to remove on this path; the only non-DRAM residue
is the per-channel operand's own read, which all cores issue simultaneously against the same few
DRAM pages (`no_gamma` 83 087 ns vs `gamma` 87 372 ns).
