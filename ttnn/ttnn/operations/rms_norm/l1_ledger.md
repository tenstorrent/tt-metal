# L1 Ledger: rms_norm

Companion to `op_design.md`. Schema and audits: `.claude/references/l1-footprint-discipline.md`.

Named block axes (from the Blocking Model): **`row`** (extent `block_row_tiles`) and **`hidden`**
(extent `core_w_tiles`; the block always spans the core's entire hidden slice). Every row below
accounts for **both**, tagged `spans` or `streams`.

## Inventory justification (Rule 1 — before any budget solve)

The solve for `block_row_tiles` at the bottom of this file is a last step, not the design. The
inventory was minimized first:

| Buffer that does **not** exist | Why not — which Rule 3 pattern removed it |
|--------------------------------|--------------------------------------------|
| `cb_input_squared` (a block-sized `x²` scratch) | **Pattern 4, fold into the accumulator.** `sumsq_block` accumulates `x·x` across the block's hidden tiles inside a persistent fp32 DEST accumulator and packs one tile per tile-row. The `x²` values never reach L1. This removes an `R·C`-tile buffer — the single largest saving in the op — and is also the measured-fastest shape (`eltwise_l1_vs_dest_accumulate`: 10.59× over an L1 read-modify-write accumulator). |
| `cb_stat_reduced` (a separate output for the finalize) | **Pattern 1, pack into the destination.** The `×1/W → +eps → rsqrt` finalize is a terminal chain that packs straight into `cb_rstd_send`, the multicast source. `compute_fusion` measures the SFPU-consumer fusion at 1.01–1.07× and it removes a whole standalone pass. |
| A second gamma buffer for the RM path | `cb_gamma_rm` is tilized once into `cb_gamma_tiles` at kernel start; both layouts then share one CB contract (`C` tiles, row-0 valid). No per-block gamma buffer exists at all. |
| A per-block gamma copy | **Pattern 1 again.** `cb_gamma_tiles` is waited upfront and never popped, so one allocation serves every block instead of `num_blocks_this_core` re-reads. |
| A separate output staging buffer on the RM path | `untilize` writes directly into `cb_output_rm`, which the writer drains; there is no tiled staging copy beyond `cb_output_tiles`, which the TILE path needs anyway. |

Two reuses were **considered and rejected**, each for a concrete reason recorded in the table below:
in-placing `gamma_block` onto `cb_output_tiles` (would give the writer and compute two concurrent
consumers), and folding `cb_rstd_send` into `cb_rstd` (would make one CB a compute product, a writer
multicast source, and a compute input — three parties).

## CB table

Symbols: `R` = `block_row_tiles`, `C` = `core_w_tiles`, `G` = `w_group_size`, `WC` = `w_chunk_tiles`
(the hidden-axis chunk; `WC == C`, one chunk, on every geometry that fits without chunking, and `NC = ceil(C / WC)`).
`T_in` = `tile_size(input_dtype)`, `T_γ` = `tile_size(gamma_dtype)`, `T_f32` = 4096, `T_bf16` = 2048.
Indicators: `rm_in`, `rm_out`, `rm_γ`, `has_gamma`, `has_tail` ∈ {0,1}.

| CB | Capacity (pages) | Live set | Axis accounting | Page format | Producer | Consumer | Lifetime | Shares with / why not |
|----|------------------|----------|-----------------|-------------|----------|----------|----------|-----------------------|
| `cb_input_rm` | `rm_cb_depth · WC` = `2·WC` | `WC` | `{row: streams → one 32-row block, hidden: spans → C}` | `T_in` | reader | compute | whole kernel, RM input only | **Cannot share.** Concurrent with `cb_input_tiles` by construction — `tilize` reads one while writing the other. Capacity exceeds live set by 2× deliberately: depth 2 overlaps stick reads with `tilize`. |
| `cb_input_tiles` | `input_cb_depth · R · (rm_in ? NC·WC : C)` = `2·R·C` unchunked | `R · C` | `{row: spans → R, hidden: spans → C}` | `T_in` | reader (TILE) / compute-`tilize` (RM) | compute | one block, two in flight | **Cannot share.** Live from `sumsq_block` through `scale_block`, overlapping every other block-scoped buffer. Capacity is 2× the live set: the depth-2 prefetch is an explicit pipelining decision (`double_buffer`, 2.78×) and is what stops the DRAM read serializing against compute. |
| `cb_scaler` | `1` | `1` | `{row: streams → 0, hidden: streams → 0}` — a constant, scales with neither | `T_bf16` (mandated, `reduce_helpers_dataflow.inl:185-187`) | reader | compute | whole kernel | **Cannot share.** Lifetime is the whole kernel, so it overlaps everything. Bare literal capacity is correct: the live set is genuinely one tile. |
| `cb_wmask` | `has_tail · 1` | `1` | `{row: streams → 0, hidden: streams → 0}` — one column mask for the ragged tile, independent of both extents | `T_bf16` | reader | compute | whole kernel, only when `has_tail` | **Cannot share.** Whole-kernel lifetime. Not allocated at all when `partial_w == 0` or this core does not own the last hidden tile. |
| `cb_zero_tile` | `1` | `1` | `{row: streams → 0, hidden: streams → 0}` | `T_f32` | reader | compute | whole kernel | **Cannot share.** Whole-kernel lifetime; it is the identity `B` operand of the `combine_stat_block` accumulation. |
| `cb_stat_sq` | `R · (NC + has_tail)` = `R` unchunked | `R · nc` | `{row: spans → R, hidden: streams → the whole slice passes through DEST, never through this buffer}` | `T_f32` | compute | compute | `sumsq_block` → `reduce_stat_block` | **Cannot share** with `cb_tail_masked`: `mask_tail_block` accumulates *into* `cb_stat_sq` while reading `cb_tail_masked`, so the two are simultaneously live. |
| `cb_tail_masked` | `has_tail · R` | `R` | `{row: spans → R, hidden: streams → window of exactly 1 (the ragged tile)}` | `T_f32` | compute | compute | `mask_tail_block` only | **Cannot share** with `cb_stat_sq` (concurrent, above). Could share with `cb_stat_sum` — disjoint lifetimes, identical format and size — but is left separate because it is `has_tail`-conditional and the saving is `R` fp32 tiles (≤ 32 KiB at the `MAX_GATHER_TILES` bound); recorded as an available, deliberately unclaimed reuse. |
| `cb_stat_partial` | `R` | `R` | `{row: spans → R, hidden: streams → fully folded into a column}` | `T_f32` | compute | writer | `reduce_stat_block` → the gather write | **Cannot share.** Live across the gather, concurrently with `cb_stat_gather` (this core's own slot is written from it). |
| `cb_stat_gather` | `R · G` | `R · G` | `{row: spans → R, hidden: spans → the cross-core extent G — all G slices of a tile-row must be resident at once, which is the whole purpose}` | `T_f32` | writer | compute | `combine_stat_block` only | **Cannot share.** Allocated on **every** group member, not just roots, because `mcast_pipe.hpp:44-45` requires an identical `dst_l1` on all receivers and a root-only CB would shift every later CB's address on the roots. The non-root copies are dead space — the accounted price of a uniform L1 map. Capacity equals live set exactly. |
| `cb_stat_sum` | `R` | `R` | `{row: spans → R, hidden: streams → G partials folded in DEST}` | `T_f32` | compute | compute | `combine_stat_block` only | **Cannot share** with `cb_stat_gather` (concurrent — it is the destination of that reduction) or with `cb_rstd_send` (concurrent — the finalize chain reads one and packs the other). |
| `cb_rstd_send` | `R` | `R` | `{row: spans → R, hidden: streams → 0}` | `T_f32` | compute | writer | `combine_stat_block` only, root-meaningful | **Cannot share** with `cb_rstd`: the multicast uses `src != dst` so the root loops back into its own `cb_rstd` (`mcast_pipe.inl:84`), which requires two distinct addresses. Merging them would also make one CB a compute product, a writer multicast source and a compute input — three parties, which the CB ownership invariant forbids. |
| `cb_rstd` | `R` | `R` | `{row: spans → R, hidden: streams → 0}` | `T_f32` | writer | compute | combine → `scale_block` | **Cannot share.** Must sit at an identical L1 address on every group member (multicast destination), and is live across `scale_block`. |
| `cb_gamma_rm` | `rm_γ · has_gamma · WC` | `WC` | `{row: streams → 0, hidden: spans → C}` | `T_γ` | reader | compute | until `load_gamma_slice` completes | **Cannot share.** Concurrent with `cb_gamma_tiles` (`tilize` reads one, writes the other). Its lifetime is disjoint from every block-scoped buffer, but it is freed conceptually rather than reallocated; sharing it with `cb_normed` was rejected because the page formats differ (`T_γ` vs `T_in`) whenever `gamma_dtype != input_dtype`, which is an explicit TARGET axis. |
| `cb_gamma_tiles` | `has_gamma · WC` | `WC` | `{row: streams → 0 — one copy serves every block, hidden: spans → C}` | `T_γ` | reader (TILE) / compute-`tilize` (RM) | compute | whole kernel | **Cannot share.** Whole-kernel lifetime; waited upfront by `gamma_block` every block and never popped. This is what keeps gamma at one DRAM read per core rather than one per block. |
| `cb_normed` | `has_gamma · R · WC` | `R · WC` | `{row: spans → R, hidden: spans → C}` | `T_in` | compute | compute | `scale_block` → `gamma_block` | **Cannot share.** Concurrent with `cb_input_tiles` (`scale_block` reads that and writes this) and with `cb_output_tiles` (`gamma_block` reads this and writes that). Not allocated when gamma is absent — `scale_block` then packs straight into `cb_output_tiles` (Rule 3 pattern 1). Sharing with `cb_output_tiles` via an in-place transform was **rejected**: the writer is already a consumer of `cb_output_tiles`, so an in-place second pass would race the writer for pages the first pass pushed. |
| `cb_output_tiles` | `output_cb_depth · WC` = `2·WC` (`R · WC` on the RM-out leg) | `WC` | `{row: streams → one tile-row at a time, hidden: spans → C}` | `T_in` | compute | writer (TILE out) / compute-`untilize` (RM out) | one block | **Cannot share.** Concurrent with `cb_normed`. Capacity exceeds the live set by 2×: depth 2 both overlaps the drain with compute and gives the writer a ≥ 4–8-tile batch behind one barrier (`double_buffer`). |
| `cb_output_rm` | `rm_out · rm_cb_depth · WC` = `2·WC` | `WC` | `{row: streams → one tile-row, hidden: spans → C}` | `T_in` | compute | writer | one block, RM output only | **Cannot share.** Concurrent with `cb_output_tiles` (`untilize` reads one, writes the other). |

## Symbol table

Every non-block parameter in a capacity expression, with its bound and the predicate establishing it.

| Symbol | Meaning | Bound | Predicate establishing the bound |
|--------|---------|-------|----------------------------------|
| `R` = `block_row_tiles` | block extent along `row` | `1 ≤ R ≤ min(core_row_tiles, MAX_GATHER_TILES / G)` and `R ≤ (l1_cb_budget − fixed_bytes) / per_row_bytes` | The residency solve below (a closed form, not a search) plus the declared `MAX_GATHER_TILES` mechanism cap. |
| `C` = `core_w_tiles` | block extent along `hidden` | `1 ≤ C ≤ max_core_w_tiles`, where `max_core_w_tiles` is the largest `C` satisfying `fixed_bytes(C) + per_row_bytes(C) ≤ l1_cb_budget` at `R = 1` | The regime-selection function raises `G` until `C = ceil(tensor_w_tiles / G)` clears this bound. This is the **residency predicate** — it is what makes regimes R1/R2 reachable and R3 unnecessary within the declared universe. |
| `G` = `w_group_size` | cores per reduction group | `1 ≤ G ≤ min(tensor_w_tiles, grid_x · grid_y)`, and `G = w_group_cols · w_group_rows` with `w_group_cols \| grid_x`, `w_group_rows \| grid_y` | Mechanism caps: a group must be a rectangle for `Mcast2D`, and a core owning zero hidden tiles would hang the gather. |
| `WC` = `w_chunk_tiles` | block extent along `hidden` of the buffers that only *stream* over it | `1 ≤ WC ≤ C`; `WC = C` (one chunk) unless a resident shard's `C` does not fit | Refinement 2b. A shard spec pins `G` **and** `C`, so when the depth ladder still does not fit, the hidden axis is chunked and the *coarsest* `WC` that fits is taken. Interleaved geometries never reach it — `_select_regime` still has `G`. |
| `NC` = `ceil(C / WC)` | hidden chunks per block | `1` unchunked | Derived, in the kernels too (`CB_CHUNK_TILES` is the only CT arg). Sets `cb_stat_sq`'s column count: one partial `Σ x²` column per chunk, folded by the reduce that already sums a tile-row's columns. |
| `input_cb_depth`, `output_cb_depth`, `rm_cb_depth` | buffer depths | `2` in Phase 0 | Explicit pipelining knobs; perf lamp P1 measures alternatives. |
| `MAX_GATHER_TILES` | cap on `R · G` | `64` (fp32 tiles = 256 KiB) | Declared mechanism cap; bounds `cb_stat_gather`, the only buffer whose capacity is a product of two extents. |
| `l1_cb_budget` | bytes available to CBs | `device.l1_size_per_core() − L1_RESERVE_BYTES`, `L1_RESERVE_BYTES = 131072` | Named host constant covering kernel binaries, stack, semaphores and allocator alignment — **not** a safety fraction. |
| `tensor_row_tiles`, `tensor_w_tiles`, `partial_w` | tensor geometry | derived from the shape, alignment-aware (`ceil`, per-image on the TILE path) | Formulas in `op_design.md` → Blocking Model → Axes. |

## Total per-core footprint

```
# WC = w_chunk_tiles (= C, one chunk, unless a resident shard forced chunking);
# NC = ceil(C / WC); pin_in / pin_out = the block CB is the resident shard itself,
# which costs the CB arena nothing (its bytes come off the budget once, up front).
per_row_bytes  = T_in · (!pin_in · input_cb_depth · (rm_in ? NC·WC : C) + has_gamma · WC)
               + T_f32 · (4 + NC + has_tail + G)                    # cb_stat_sq (NC[+1] cols), _partial,
                                                                    #   _sum, cb_rstd_send, cb_rstd, cb_stat_gather
fixed_bytes    = T_in · (!pin_out · output_cb_depth · WC + rm_in·rm_cb_depth·WC + rm_out·rm_cb_depth·WC)
               + T_γ  · has_gamma · WC · (1 + rm_γ)                 # cb_gamma_tiles [+ cb_gamma_rm]
               + T_f32                                              # cb_zero_tile
               + T_bf16 · (1 + has_tail)                            # cb_scaler [+ cb_wmask]

footprint(R)   = fixed_bytes + R · per_row_bytes

block_row_tiles = clamp( floor((l1_cb_budget − fixed_bytes) / per_row_bytes),
                         1,
                         min(core_row_tiles, MAX_GATHER_TILES / G) )
```

**The chunking trade, stated as a formula.** `WC` moves two terms in opposite directions:
`T_in·(2 + has_gamma)·WC` (the streaming buffers) falls with a finer chunk while
`T_f32·NC = T_f32·ceil(C/WC)` (one stat column per chunk) rises, so the footprint is minimized near
`WC ≈ sqrt(C · T_f32 / (T_in·(2+has_gamma)))` and the solve simply takes the **coarsest** `WC` that
fits — chunking is a residency fallback, not a target. Worked: a HEIGHT-sharded `(1,1,96,6144)` bf16
TILE core holds `C = 192`; the two resident shards take `2 · 393 216 B` of the `1 441 792 B` budget,
leaving `655 360 B` against an unchunked `cb_gamma_tiles + cb_normed` of `2 · 393 216 B` — it does not
fit, and `WC = 128` does (`2 · 262 144 + 2 · 4096 + 26 624 = 559 104 B`).

**Which terms scale with which knob.** Everything in `per_row_bytes` scales with `R`. Within it,
`T_in · (input_cb_depth + has_gamma) · C` also scales with `C` (the two block-sized buffers), and
`T_f32 · G` also scales with `G` (the gather). `fixed_bytes` scales with `C` only. Nothing scales with
`tensor_row_tiles`, `tensor_w_tiles` or any other whole-op dimension — the only op dimension that
reaches a capacity expression is `tensor_w_tiles`, and only through `C = ceil(tensor_w_tiles / G)`,
which the residency predicate bounds.

### Worked check — the widest prefill perf case

`(1, 1, 8192, 7168)`, bfloat16, TILE, gamma bfloat16 TILE, 64-core grid, `l1_cb_budget = 1 441 792 B`.
`tensor_row_tiles = 256`, `tensor_w_tiles = 224`, `partial_w = 0`.

| Candidate `G` | `C` | `fixed_bytes` | `per_row_bytes` | `R` | `active cores` | score `(active, −G, R)` |
|---------------|-----|---------------|-----------------|-----|----------------|--------------------------|
| 1 | 224 | 1 382 400 | 1 400 832 | **0 — does not fit** | — | discarded |
| **2** | **112** | **694 272** | **716 800** | **1** | **64** | **(64, −2, 1) ← chosen** |
| 4 | 56 | 350 208 | 380 928 | 2 | 64 | (64, −4, 2) |

Chosen footprint `694 272 + 1 · 716 800 = 1 411 072 B` ≤ `1 441 792 B`. Note that `G = 1` is
**discarded by the residency predicate, not by occupancy** — a 224-tile hidden slice plus its
intermediate cannot be held resident on one core, and the alternative to splitting `hidden` would be
re-reading the whole input from DRAM (regime R3).

### Worked check — the widest decode perf case

`(1, 1, 32, 7168)`: `tensor_row_tiles = 1`, so `active = G` and the selection maximizes `G` to 64.
`C = 4`, `R` clamped to 1 by both `core_row_tiles` and `MAX_GATHER_TILES / G = 1`.
Footprint `30 720 + 307 200 = 337 920 B` — of which `cb_stat_gather` alone is `64 · 4096 = 262 144 B`.
The block buffers and the stat buffers scale **inversely** in `G`, which is why no separate cap on `C`
is needed once `MAX_GATHER_TILES` bounds `R · G`.

---

## Data-movement budget

For the chosen hybrid split. `N` = tensor bytes, `W_bytes` = `W · gamma_element_bytes`,
`num_row_groups = (grid_x·grid_y) / G`.

| Tensor | DRAM crossings | Why that many | Cross-core traffic added |
|--------|----------------|---------------|--------------------------|
| `input_tensor` | **1×** (`N` bytes), **0×** on a resident shard | `cb_input_tiles` holds the whole block resident from `sumsq_block` through `scale_block`, so the apply pass re-reads it from L1, not DRAM. This is the residency decision the hidden-axis split exists to enable; without it the count would be 2× (regime R3). | none |
| `gamma` | **`num_row_groups ×`**, or **`num_row_groups · num_blocks_this_core ×`** when the hidden axis is chunked | Structurally unreachable minimum: gamma does not vary along `row`, so each of the `num_row_groups` disjoint row-groups must have its own copy. *Within* a group it is read exactly once — the hidden split partitions it across the group's members. Read once per core for the whole kernel, never per block, because `cb_gamma_tiles` is never popped — **except** under hidden-axis chunking (Refinement 2b), where only one chunk of the vector fits L1 at a time and it is re-fed per block (`W_bytes` per block, on geometries whose whole slice would not fit at all; `num_blocks_this_core` is 1 on every HEIGHT-sharded case measured). | none (scheme lamp G1 would convert `num_row_groups − 1` of these reads into a multicast) |
| `output` | **1×** (`N` bytes) | Written once, streamed out of `cb_output_tiles` as it is produced. | none |
| per-block statistics | 0 | Never touch DRAM. | Per block: `(G−1) · R` fp32 tiles unicast into the root's gather slots, plus `R` fp32 tiles multicast to `G` receivers. Over a core's whole assignment: `num_blocks_this_core · R · 4096 · G` bytes of gather-plus-broadcast, i.e. `core_row_tiles · G · 4096` bytes independent of `R`. |

**Totals, widest prefill case** (`(1,1,8192,7168)` bf16, `G = 2`, `num_row_groups = 32`):
DRAM `2N + 32·W_bytes = 234 881 024 + 458 752 = 235 339 776 B`, against a named-boundary minimum of
`2N + W_bytes = 234 895 360 B` — **0.19 % above minimum**. Cross-core ≈ `256 · 2 · 4096 = 2 MiB` in
total across the grid, i.e. under 1 % of DRAM traffic, on the cheaper tier.

**Totals, widest decode case** (`(1,1,32,7168)` bf16, `G = 64`, `num_row_groups = 1`):
DRAM `2N + W_bytes = 931 840 B` — **exactly the minimum**, because a single row-group cannot replicate
gamma. Cross-core `63·4096 + 64·4096 = 520 192 B`, which is 56 % of the DRAM bytes. That ratio is
uncomfortable and is exactly what **perf lamp P2** exists to settle: capping `G` at 16 or 32 cuts the
multicast fan-out proportionally while still filling most of the grid, and the measured-fastest
geometries in `feature_spec.py` for these shapes use 28–32 cores rather than 64.

> Cheapest-traffic split considered: **pure hidden split with `num_row_groups = 1`** —
> `−(num_row_groups−1)·W_bytes` DRAM (≈ 0.44 MiB, or 0.19 % of DRAM traffic, on the widest prefill
> case), `+(tensor_row_tiles/R − core_row_tiles/R)` additional combine rounds on the NoC tier.
> Implemented: **the hybrid split** (row across groups × hidden within a group), which *is* the
> cheapest-traffic split in the decode regime (it degenerates to `num_row_groups = 1` there) and is
> within 0.19 % of it in the prefill regime. Lamped as **perf lamp P3** because the combine-round cost
> it would add is unmeasured and is predicted to exceed the bytes it saves by more than an order of
> magnitude; the structure keeps it reachable by forcing `w_group_size = grid_x · grid_y` in the
> selection function's `score` tuple, a one-line change.
>
> The alternative that is *not* cheapest and *is* lamped for a structural reason is regime **R3**
> (`+N` DRAM bytes, one extra full read of the input). It is reachable through the `Accumulate` path
> already present in the reduce helper and the reader's `byte_offset_within_page` chunk hook.


---

## Implementation deltas (recorded by the implementer)

The inventory above was built as designed — no buffer in it was added, and the two
"available, deliberately unclaimed" reuses stayed unclaimed. Four rows changed shape.

| Change | Row(s) affected | Why |
|--------|-----------------|-----|
| **`C` is now GROUP-UNIFORM, not per-core.** `C` = `cb_w_tiles` = `ceil(tensor_w_tiles / G)` on **every** core of a group; a core's ragged share rides as the runtime `core_w <= C`. | every `C`-scaled row | Two hard mechanisms, not a preference. (1) A CB's page capacity must be an exact multiple of its push/pop quantum (`dataflow_api.h:216-221`, "no other wrap is legal"); a ragged per-core quantum does not divide the one uniform capacity the multicast/gather L1 map already forced. (2) `cb_rstd` / `cb_stat_gather` are addressed by a peer's LOCAL pointer. The statistics phases still walk only `core_w` columns (at row stride `C`), so the pad tiles never enter the reduction; the apply phases cover them and the writer drops them. This also collapsed the design's two kernel core-ranges into one. |
| **`cb_tail_masked` does not exist.** | `cb_tail_masked` (deleted) | `mask_tail_block` is now ONE fused chain — `BinaryFpu(x_tail, wmask, Mul, Row) -> Square<> -> PackTile` — so the masked value never lands in L1. Rule 3 pattern 4 (fold into the accumulator), the same pattern that removed `cb_input_squared`. Saves `has_tail * R` fp32 tiles. |
| **`cb_stat_sq` is `R * (1 + has_tail)` pages, not `R`.** | `cb_stat_sq` | The bulk accumulation and the masked tail write **two stat columns per tile-row** (`r * nc + col`, caller-managed strided packs into one caller-reserved window), and `reduce<SUM, REDUCE_ROW>` folds both columns in one call over `of(R, nc)`. This replaces the design's `L1Accumulation::Enabled` re-pack, which `eltwise_chain.inl:1007-1017` pins to a SINGLE output tile (`walk` is false whenever L1 accumulation is on), so it cannot accumulate into `R` distinct tiles. Net vs. the design: `+has_tail * R` fp32 tiles here, `-has_tail * R` from the deleted `cb_tail_masked` — a wash. |
| **`cb_output_tiles` is `R * C` pages on the ROW_MAJOR-output path** (still `output_cb_depth * C` on the tiled path). | `cb_output_tiles` | `gamma_block` and `untilize_out_block` are sequential compute helpers — both own all three TRISCs, so they cannot pipeline through a depth-2 buffer: the producer would block on `cb_reserve_back` before the consumer ever runs. Same rule as `cb_normed` (`ttnn-cb-memory-fundamentals.md`). The term therefore moves from `fixed_bytes` to `per_row_bytes` on that path. |
| **`cb_input_tiles` is `1 · R · C` pages, not `input_cb_depth · R · C`, when the busiest core's row assignment is a single block** (verifier change). | `cb_input_tiles` | `input_cb_depth` buys exactly one thing — the reader prefetching block `b+1` while compute runs block `b`. When `num_blocks == 1` on every core (which is what the selection picks for e.g. `(1,1,8192,1024)`: `core_row_tiles = 3 = R`) there is no block `b+1`, so the second buffer is dead L1 and the row's "capacity is 2× the live set for pipelining" justification is simply false there. The residency SOLVE still uses the full knob (the conservative value), so this only ever lowers the footprint and cannot change the selected `(G, C, R)`. |
| **The CB inventory is stated ONCE** (verifier change). | all rows | `_cb_specs()` in `rms_norm_program_descriptor.py` is now the single statement of `(index, num_pages, page_size, format)`; the descriptor's CB list is a `map` over it and `_cb_bytes()` (the residency solve) *derives* `fixed_bytes` / `per_row_bytes` from it by differencing at `R = 1, 2` (every page count is affine in `R`). Previously the footprint expressions below were a second, independent statement of this table and could drift from what was actually allocated. Verified byte-identical to the previous closed form over 4608 `(dtype, gamma dtype, rm_in, rm_out, rm_γ, has_gamma, has_tail, C, G)` combinations. |
| **`cb_stat_sq` carries `R · (1 + has_tail_global)` pages on every core, including cores that do not own the tensor's last hidden tile** (live set `R` there). | `cb_stat_sq` | Same uniform-L1-map requirement as `cb_stat_gather`: every CB is allocated identically group-wide because two of them are addressed by a peer's local pointer. The dead `R` fp32 pages on non-tail-owning cores are the accounted price; they are bounded by `MAX_GATHER_TILES` like every other `R`-scaled row. |
| **No row changed shape (Refinement 1).** `T_in` / `T_γ` gain a third value — `tile_size(bfloat8_b) = 1088` — and every `T_f32` / `T_bf16` row is untouched at BOTH `fp32_dest_acc_en` settings. | none (values only) | Refinement 1 added `dtype`/`gamma_dtype` = `bfloat8_b` and `fp32_dest_acc_en = False`, and neither is a footprint change. (1) Block-float only *narrows* the `T_in`/`T_γ`-scaled rows (1088 vs 2048/4096 bytes per tile), so the residency predicate is strictly easier to satisfy and the selected `(G, C, R)` can only get coarser — never a new OOM. (2) The whole `cb_stat_*` chain (`cb_stat_sq`, `_partial`, `_gather`, `_sum`, `cb_rstd_send`, `cb_rstd`) plus `cb_zero_tile` are pinned to `T_f32` **unconditionally** — they do NOT follow `fp32_dest_acc_en`, and `cb_scaler` / `cb_wmask` stay `T_bf16` (`reduce_helpers_dataflow.inl:185-187` `static_assert`s the scaler format). That was already true at Phase 0; Refinement 1 is what makes it load-bearing, because `fp32_dest_acc_en = False` narrows only the *in-DEST* accumulation while `Σ x²` still crosses L1 in fp32. Demoting any of those rows to `T_in` would be the `row_reduce_accumulate` failure mode, not a saving. (3) DEST capacity is never spelled as a literal — the helpers read `DEST_AUTO_LIMIT`, which doubles 4 → 8 when fp32 DEST accumulation is off — so no capacity in this ledger assumes the halved value. |

### Refinement 2 deltas — physical shard placements

Three rows change shape, and one previously-unclaimed reuse is now claimed. The
**live sets and axis accounting are unchanged**: a shard supplies the same `(R, C, G)`
geometry `_select_regime` would otherwise choose, so this is a placement change, not
a blocking change.

| Change | Row(s) affected | Why |
|--------|-----------------|-----|
| **`cb_input_tiles` costs the CB arena NOTHING on a TILE-layout sharded input.** Its capacity becomes the shard's own bank size (`shard_row_tiles · C` pages ≥ `R · C`), and its address is the tensor's L1 buffer address. | `cb_input_tiles` | This is what "sharding implemented" means: `ttnn.cb_descriptor_from_sharded_tensor` pins the CB zero-copy over the resident shard, so `load_block` is the CB handshake alone and the reader's DRAM leg disappears (`_sharded_cb`). The shard's pages are already tile-row-major at row stride `C`, i.e. exactly the block layout compute expects, and total pushes over the kernel (`core_row_tiles · C`) never exceed capacity, so the pointer never wraps. The bytes do not vanish — they move out of the arena and into the budget, which now subtracts both shards' `aligned_size_per_bank` before the solve runs. |
| **`cb_output_tiles` likewise on a TILE-layout sharded output.** | `cb_output_tiles` | Symmetric: compute packs straight into the block's final home and `store_block` is the CB handshake alone. The pad columns / padded tile-rows compute also writes land in the shard's own padding, outside the logical tensor, and are never read back. |
| **`cb_gamma_rm` now SHARES `cb_input_rm`** when `gamma_dtype == input_dtype` (so the page formats match). | `cb_gamma_rm` (conditionally deleted) | Rule 3 pattern 3 (alias disjoint lifetimes) — the reuse the Phase 0 row recorded as rejected, now available: `cb_gamma_rm` is filled once and dies the instant `load_gamma_slice` has tilized it, strictly before `cb_input_rm`'s first push, and **both have the same producer (the reader) and the same consumer (compute's `tilize`)**, so single-producer/single-consumer holds. Saves `C` tiles on every ROW_MAJOR-input × ROW_MAJOR-gamma case. The Phase 0 rejection stands for the mixed-precision case it was written about: one CB index carries one data format, so `gamma_dtype != input_dtype` keeps its own buffer. |
| **The buffer depths are now a LADDER, not fixed knobs** (`_DEPTH_LADDER`): `(input_cb_depth, rm_cb_depth)` steps `(2,2) → (2,1) → (1,1)`, and the solve takes the first step that fits. | `cb_input_rm`, `cb_output_rm`, `cb_input_tiles` | A shard spec pins BOTH `G` and `C`, so `R` and the depths are the only residency knobs left — and `R` is often already 1 (a HEIGHT shard of one tile-row). Depth buys overlap, so it is spent first and surrendered last: step 0 is the knobs, and no interleaved geometry ever leaves it, so the default path is byte-identical. |

**Where a physical shard leaves the footprint standing.** For HEIGHT_SHARDED the shard
cuts the independent `row` axis, so `G == 1` by construction and `C == tensor_w_tiles` —
the one op dimension that reaches a capacity is no longer bounded by "raise `G`", because
the caller pinned `G`. At `C ≳ 127` (bf16) or `C ≳ 64` (fp32, or an fp32 gamma) the
resident input shard + resident output shard + `cb_gamma_tiles` (`C` tiles) alone exceed
the 1.44 MiB budget, and the solve raises rather than allocate. That is the sole remaining
gap in this refinement and it is a **capacity** limit, not a tuning one: no depth or block
setting reaches it. See `op_requirements.md` Refinement 2b — the fix is to chunk the hidden
axis inside a core, which for a *resident* shard is a nearly free version of regime R3
(R3's fatal cost is a second DRAM read of the input; a resident shard has none).

### Corrected footprint expressions (what the host actually computes)

These are what `_cb_bytes()` *derives* from `_cb_specs()` — they are recorded here as the closed
form, not as a second source of truth (see the last two implementation deltas above).

```
per_row_bytes  = T_in  * C * (input_cb_depth + has_gamma + rm_out)     # cb_input_tiles, cb_normed, [cb_output_tiles]
               + T_f32 * (4 + (1 + has_tail) + G)                      # cb_stat_partial/_sum/_rstd_send/_rstd,
                                                                       #   cb_stat_sq (1+has_tail cols), cb_stat_gather
fixed_bytes    = T_in  * C * ((1 - rm_out) * output_cb_depth
                              + rm_in * rm_cb_depth + rm_out * rm_cb_depth)
               + T_gamma * has_gamma * C * (1 + rm_gamma)              # cb_gamma_tiles [+ cb_gamma_rm]
               + T_bf16 * (1 + has_tail)                               # cb_scaler [+ cb_wmask]
               + T_f32                                                 # cb_zero_tile

block_row_tiles = clamp( floor((l1_cb_budget - fixed_bytes) / per_row_bytes),
                         1, min(core_row_tiles, MAX_GATHER_TILES / G) )
```

`l1_cb_budget = l1_size_per_core(arch) - L1_RESERVE_BYTES`, `L1_RESERVE_BYTES = 131072`.
Nothing scales with `tensor_row_tiles` or `tensor_w_tiles`; the only op dimension reaching a
capacity is `tensor_w_tiles`, and only through `C = ceil(tensor_w_tiles / G)`, which the
residency predicate bounds by raising `G`.

## Data-movement budget — measured selections

Unchanged from the table above in kind: **input 1x DRAM, output 1x DRAM, gamma
`num_row_groups x`**, statistics never touch DRAM. Two selection facts, measured on a
blackhole_p150b 11x10 grid:

| Shape | `G` | row-groups | `C` | `R` | blocks/core | cores | DRAM crossings | Cross-core per block |
|-------|-----|-----------|-----|-----|-------------|-------|----------------|----------------------|
| `(1,1,8192,7168)` prefill | 2 | 55 | 112 | 1 | 5 | 110 | in 1x, out 1x, gamma 55x | `1*R` fp32 tile gathered + `R` multicast to 2 |
| `(1,1,8192,2304)` prefill | 1 | 110 | 72 | 2 | 2 | 110 | in 1x, out 1x, gamma 110x | degenerate (local copy) |
| `(1,1,32,7168)` decode | 22 | 1 | 11 | 1 | 1 | 22 | in 1x, out 1x, gamma **1x** (the minimum) | `21*R` gathered + `R` multicast to 22 |
| `(1,1,8192,1024)` prefill | 1 | 110 | 32 | 3 | 1 | 110 | in 1x, out 1x, gamma 110x | degenerate (local copy) |

Re-read off `_select_regime` **on the device** (blackhole, 11×10 grid, `l1_cb_budget = 1 441 792 B`)
during verification; the first row previously recorded `G = 5, C = 45, R = 3`, which the current
score tuple does not select (both `G = 2` and `G = 5` fill all 110 cores, and the `−G` tiebreak —
fewest combine partners — takes `G = 2`). The `(1,1,8192,1024)` row is the `num_blocks == 1` case
that the verifier's `input_cb_depth` change now sizes at depth 1.

The decode row is the design's predicted best case: a single row-group cannot replicate gamma,
so DRAM traffic is exactly the named-boundary minimum. The `G = 22` (rather than 110) choice is
**perf lamp P2, now measured and adopted** — see `MAX_W_GROUP_SIZE` in
`rms_norm_program_descriptor.py` for the numbers.

### Data-movement budget — a physically SHARDED input/output (Refinement 2)

The activation crossings drop from 1× DRAM to **0×**: the input is already in L1 and the
output's final home is L1, so neither activation tensor crosses DRAM at all inside the op.

| Tensor | DRAM crossings, INTERLEAVED | DRAM crossings, `*_SHARDED` | Why |
|--------|----------------------------|-----------------------------|-----|
| `input_tensor` | 1× (`N` bytes) | **0×** | The shard IS the block, resident in the consuming core's own L1. TILE: consumed through a pinned zero-copy CB, so not one byte moves. ROW_MAJOR: re-strided CORE-LOCALLY (L1 → L1) into the tile-row stride `tilize` requires, because the shard's stick stride is its own width — still no DRAM crossing, and one bulk transfer per 32-row group when the strides already agree. |
| `output` | 1× (`N` bytes) | **0×** | Symmetric. TILE: compute packs into the pinned output shard. ROW_MAJOR: `untilize` emits the uniform stride, the writer re-strides it core-locally into the shard. |
| `gamma` | `num_row_groups ×` | `num_row_groups ×` — **1×** for WIDTH_SHARDED | Unchanged in kind: gamma stays interleaved DRAM and is read once per core for the whole kernel. WIDTH_SHARDED is the structural minimum, because the whole shard grid is ONE reduction group (`num_row_groups == 1`), so the group's members read disjoint slices and gamma crosses DRAM exactly once. HEIGHT_SHARDED is the worst case (one row-group per core). |
| per-block statistics | 0 | 0 | Unchanged. Per block: `(G−1)·R` fp32 tiles unicast into the root's gather slots + `R` tiles multicast to the group rectangle. HEIGHT_SHARDED has `G == 1`, so the round degenerates to a local copy and there is **no cross-core traffic at all**. |

A core's own shard is never addressed through a `TensorAccessor`; the accessor keeps owning
interleaved I/O and gamma. `tests/.../test_rms_norm_sharded.py::test_rms_norm_tile_shard_is_consumed_in_place`
asserts the pinning on the descriptor, because an accessor read of a local shard is
numerically indistinguishable from the zero-copy path and would pass every value check.

**Filler cores.** A WIDTH shard grid is not always a rectangle (16 slices on an 11-wide grid
is a full row plus a 5-core row), and the `rstd` broadcast needs one. The group rectangle is
therefore the shard grid's bounding box, and the cores in it that own no shard stay program
cores — they hold the identical CB map (so the broadcast lands in a reserved `cb_rstd` rather
than in unowned L1) and nothing else. They receive `R` fp32 tiles per block and never ack,
which is why the mcast is emitted with an explicit `num_active = G − 1`. Cost: at most
`grid_x − 1` cores' worth of otherwise-idle L1 map per group.

---

## Refinement 3 deltas — the two-stage grid combine

The combine is now a **two-level tree** (`_tree_for_box`), not a flat root-gather. Level 1 folds
one grid ROW of the group on that row's leader; level 2 folds the row totals on the root.
`stage2_span == 1` is the degenerate flat gather and is the Phase 0 inventory byte-for-byte, which
is what every group that is not a fully populated multi-row rectangle keeps (a shard grid with
filler cores, any 1-D group, `G == 1`, and `nx == 1` — where level 1 would be a self-write).

| CB | Pages (capacity) | Live set | Spans / streams | Format | Producer | Consumer | Lifetime | Shares with / why not |
|----|------------------|----------|-----------------|--------|----------|----------|----------|-----------------------|
| `cb_stat_gather` (**resized**) | `R · S1` | `R · S1` | `{row: spans → R, hidden: spans → the LEVEL-1 cross-core extent S1}` | `T_f32` | writer | compute | `combine_stat_block` level 1 | Unchanged in kind; `S1` is now one grid row of the group (`nx`) instead of the whole group (`G`), so this CB **shrinks** by a factor `ny` whenever the tree is taken. |
| `cb_stat_gather2` (**new**, id 15) | `R · S2` | `R · S2` | `{row: spans → R, hidden: spans → the LEVEL-2 extent S2 = ny}` | `T_f32` | writer (leaders → root) | compute (root) | `combine_stat_block` level 2 | **Cannot share.** Concurrent with `cb_stat_gather` (level 1's result feeds it) and with `cb_stat_sum` (its destination). Not allocated at all when the tree is flat. |
| `cb_branch_sum` (**new**, id 18) | `R` | `R` | `{row: spans → R, hidden: streams → S1 partials folded in DEST}` | `T_f32` | compute (leader) | writer (leader) | `combine_stat_block`, between the levels | **Cannot share** with `cb_stat_partial` (both are compute→writer, but a leader holds this one while the next block's partial is being produced) or with `cb_stat_sum` (that one is the root's level-2 destination, live at the same time on the root). Not allocated when the tree is flat. |

**Net L1: the tree COSTS NOTHING and usually saves.** Flat holds `R·G = R·nx·ny` fp32 stat-gather
tiles; the tree holds `R·(nx + ny + 1)`. At the decode geometry `11 × 2` that is 14 tiles against
22, and at `11 × 10` it is 22 against 110 — i.e. the deeper the group, the larger the saving. The
`MAX_GATHER_TILES` cap correspondingly moves from `R·G` to `R · max(S1, S2)`, which is the largest
buffer that actually exists.

### Data-movement budget — cross-core traffic per block (updated)

Activation and gamma DRAM crossings are **unchanged** (input 1×, output 1×, gamma
`num_row_groups ×`; 0× for the activations on a resident shard). Only the statistics traffic
changes shape:

| Combine | Unicasts into ONE core's L1 | fp32 tile adds on the critical path | Multicast |
|---------|-----------------------------|--------------------------------------|-----------|
| Flat (Phase 0, still used where the tree does not apply) | `G − 1`, all into the root | `G`, all on the root | `R` tiles to the group rectangle |
| Two-stage grid (Refinement 3) | `nx − 1` into each row leader, then `ny − 1` into the root | `nx` (in parallel on every leader) + `ny` (root) | unchanged |

Total bytes on the NoC are the same to within one extra `R`-tile hop per leader; what changes is
the **fan-in at any one destination** and the **serial tile-add count on the root**, which is what
the measurement showed to be the binding term (3630 ns of a 12467 ns decode op at `G = 22`).

### Measured selections — updated for `MAX_W_GROUP_SIZE = 0`

| Shape | `G` (was) | `S1 × S2` | `C` | `R` | cores (was) | DRAM crossings |
|-------|-----------|-----------|-----|-----|-------------|----------------|
| `(1,1,32,7168)` decode | 110 (22) | 11 × 10 | 3 | 1 | 110 (22) | in 1×, out 1×, gamma **1×** (still the minimum: one row-group) |
| `(1,1,32,5120)` decode | 110 (22) | 11 × 10 | 2 | 1 | 110 (22) | in 1×, out 1×, gamma 1× |
| `(1,1,32,2304)` decode | 55 (22) | 11 × 5 | 2 | 1 | 55 (22) | in 1×, out 1×, gamma 1× |
| `(1,1,32,1024)` decode | 22 (22) | 11 × 2 | 2 | 1 | 22 (22) | unchanged |
| every `(1,1,8192,W)` prefill | 1–2 (unchanged) | flat | unchanged | unchanged | 110 | unchanged |

The prefill rows are untouched: the score ties on occupancy for every `G` that fills the grid and
the `−G` tiebreak still takes the smallest, so the cap was never binding there (measured: 99 559 /
212 595 / 425 990 / 592 081 ns against 103 076 / 220 005 / 425 343 / 591 707 before).

---

## Refinement 4 deltas — the critical-path admissibility band

**No CB is added, removed or resized.** The inventory above is unchanged; every `Num Pages`
expression still reads `R`, `C`, `G`/`S1`, `S2` and the depth knobs. What changed is which
`(G, C, R)` the selection function *picks* for a shape whose `row` split leaves a materially
unbalanced critical path — `_admissible_by_balance`, gated by `BALANCE_SLACK_PCT = 15` and
`MIN_CORE_W_TILES = 16`. Because the per-core footprint is a function of `(C, G, R)` and every
candidate the band chooses among **already cleared the residency solve**, the L1 accounting
needs no new row: the band can only ever pick a geometry that fits.

Two second-order footprint consequences, both favourable, both already expressed by the table:

* `C` halves (or better) wherever the band fires, so `cb_input_tiles` (`depth·R·C`), `cb_normed`
  (`R·C`), `cb_output_tiles` (`depth·C`) and `cb_gamma_tiles` (`C`) all shrink per tile-row —
  which is what lets the solve return a **coarser** `R` (`(1,1,8192,1024)`: `R` 3 → 5,
  `(1,1,8192,2304)`: 2 → 4). Coarser `R` is fewer combine rounds, i.e. the knob moving in the
  direction `op_design.md` wants.
* `cb_stat_gather` grows from `R·1` to `R·G`, i.e. from `R` to `2R` fp32 tiles at `G = 2` — 4 KiB
  extra per tile-row against the ~64 KiB per tile-row the block CBs give back.

### Data-movement budget — updated selections

Activation crossings are **unchanged in kind** (input 1×, output 1×; 0× on a resident shard).
`gamma` improves: it crosses DRAM `num_row_groups ×`, and the band *halves* `num_row_groups` on
the shapes where it fires, because a group is now 2 cores instead of 1.

| Shape | `G` (was) | row-groups (was) | `C` (was) | `R` (was) | blocks/core (was) | cores | DRAM crossings | Cross-core per block |
|-------|-----------|------------------|-----------|-----------|-------------------|-------|----------------|----------------------|
| `(1,1,8192,1024)` prefill | **2** (1) | **55** (110) | **16** (32) | **5** (3) | 1 (1) | 110 | in 1×, out 1×, gamma **55×** (was 110×) | `1·R` fp32 tile gathered + `R` multicast to 2 |
| `(1,1,8192,2304)` prefill | **2** (1) | **55** (110) | **36** (72) | **4** (2) | 2 (2) | 110 | in 1×, out 1×, gamma **55×** (was 110×) | `1·R` gathered + `R` multicast to 2 |
| `(1,1,8192,5120)` prefill | 2 | 55 | 80 | 1 | 5 | 110 | unchanged | unchanged |
| `(1,1,8192,7168)` prefill | 2 | 55 | 112 | 1 | 5 | 110 | unchanged | unchanged |
| every `(1,1,32,W)` decode | unchanged | unchanged | unchanged | unchanged | unchanged | unchanged | unchanged | unchanged (the band never fires: one candidate survives the occupancy key) |

So the total DRAM bytes **fall** on the two shapes that moved — by `55 · W · 2` bytes of gamma
(0.11 MiB at `W = 1024`) — while the activation traffic, which is 99.9 % of the total, is
untouched. The measured win (1.098× / 1.115×) is not that saving; it is the shorter critical
path (see the changelog), and the gamma saving is incidental.

**Statistics traffic appears where there was none.** At `G = 1` the combine degenerates to a
local copy; at `G = 2` each block really unicasts `R` fp32 tiles to the partner and multicasts
`R` back. That is `2·R·4096` bytes per block per group — 40 KiB per block at `R = 5` — against
the ~2.6 MiB of activation bytes the same block moves, i.e. 1.5 %. It is the cost the band pays
and the measurement says it is worth it.
