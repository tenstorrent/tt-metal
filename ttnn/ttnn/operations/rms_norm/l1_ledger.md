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

Symbols: `R` = `block_row_tiles`, `C` = `core_w_tiles`, `G` = `w_group_size`.
`T_in` = `tile_size(input_dtype)`, `T_γ` = `tile_size(gamma_dtype)`, `T_f32` = 4096, `T_bf16` = 2048.
Indicators: `rm_in`, `rm_out`, `rm_γ`, `has_gamma`, `has_tail` ∈ {0,1}.

| CB | Capacity (pages) | Live set | Axis accounting | Page format | Producer | Consumer | Lifetime | Shares with / why not |
|----|------------------|----------|-----------------|-------------|----------|----------|----------|-----------------------|
| `cb_input_rm` | `rm_cb_depth · C` = `2·C` | `C` | `{row: streams → one 32-row block, hidden: spans → C}` | `T_in` | reader | compute | whole kernel, RM input only | **Cannot share.** Concurrent with `cb_input_tiles` by construction — `tilize` reads one while writing the other. Capacity exceeds live set by 2× deliberately: depth 2 overlaps stick reads with `tilize`. |
| `cb_input_tiles` | `input_cb_depth · R · C` = `2·R·C` | `R · C` | `{row: spans → R, hidden: spans → C}` | `T_in` | reader (TILE) / compute-`tilize` (RM) | compute | one block, two in flight | **Cannot share.** Live from `sumsq_block` through `scale_block`, overlapping every other block-scoped buffer. Capacity is 2× the live set: the depth-2 prefetch is an explicit pipelining decision (`double_buffer`, 2.78×) and is what stops the DRAM read serializing against compute. |
| `cb_scaler` | `1` | `1` | `{row: streams → 0, hidden: streams → 0}` — a constant, scales with neither | `T_bf16` (mandated, `reduce_helpers_dataflow.inl:185-187`) | reader | compute | whole kernel | **Cannot share.** Lifetime is the whole kernel, so it overlaps everything. Bare literal capacity is correct: the live set is genuinely one tile. |
| `cb_wmask` | `has_tail · 1` | `1` | `{row: streams → 0, hidden: streams → 0}` — one column mask for the ragged tile, independent of both extents | `T_bf16` | reader | compute | whole kernel, only when `has_tail` | **Cannot share.** Whole-kernel lifetime. Not allocated at all when `partial_w == 0` or this core does not own the last hidden tile. |
| `cb_zero_tile` | `1` | `1` | `{row: streams → 0, hidden: streams → 0}` | `T_f32` | reader | compute | whole kernel | **Cannot share.** Whole-kernel lifetime; it is the identity `B` operand of the `combine_stat_block` accumulation. |
| `cb_stat_sq` | `R` | `R` | `{row: spans → R, hidden: streams → the whole slice passes through DEST, never through this buffer}` | `T_f32` | compute | compute | `sumsq_block` → `reduce_stat_block` | **Cannot share** with `cb_tail_masked`: `mask_tail_block` accumulates *into* `cb_stat_sq` while reading `cb_tail_masked`, so the two are simultaneously live. |
| `cb_tail_masked` | `has_tail · R` | `R` | `{row: spans → R, hidden: streams → window of exactly 1 (the ragged tile)}` | `T_f32` | compute | compute | `mask_tail_block` only | **Cannot share** with `cb_stat_sq` (concurrent, above). Could share with `cb_stat_sum` — disjoint lifetimes, identical format and size — but is left separate because it is `has_tail`-conditional and the saving is `R` fp32 tiles (≤ 32 KiB at the `MAX_GATHER_TILES` bound); recorded as an available, deliberately unclaimed reuse. |
| `cb_stat_partial` | `R` | `R` | `{row: spans → R, hidden: streams → fully folded into a column}` | `T_f32` | compute | writer | `reduce_stat_block` → the gather write | **Cannot share.** Live across the gather, concurrently with `cb_stat_gather` (this core's own slot is written from it). |
| `cb_stat_gather` | `R · G` | `R · G` | `{row: spans → R, hidden: spans → the cross-core extent G — all G slices of a tile-row must be resident at once, which is the whole purpose}` | `T_f32` | writer | compute | `combine_stat_block` only | **Cannot share.** Allocated on **every** group member, not just roots, because `mcast_pipe.hpp:44-45` requires an identical `dst_l1` on all receivers and a root-only CB would shift every later CB's address on the roots. The non-root copies are dead space — the accounted price of a uniform L1 map. Capacity equals live set exactly. |
| `cb_stat_sum` | `R` | `R` | `{row: spans → R, hidden: streams → G partials folded in DEST}` | `T_f32` | compute | compute | `combine_stat_block` only | **Cannot share** with `cb_stat_gather` (concurrent — it is the destination of that reduction) or with `cb_rstd_send` (concurrent — the finalize chain reads one and packs the other). |
| `cb_rstd_send` | `R` | `R` | `{row: spans → R, hidden: streams → 0}` | `T_f32` | compute | writer | `combine_stat_block` only, root-meaningful | **Cannot share** with `cb_rstd`: the multicast uses `src != dst` so the root loops back into its own `cb_rstd` (`mcast_pipe.inl:84`), which requires two distinct addresses. Merging them would also make one CB a compute product, a writer multicast source and a compute input — three parties, which the CB ownership invariant forbids. |
| `cb_rstd` | `R` | `R` | `{row: spans → R, hidden: streams → 0}` | `T_f32` | writer | compute | combine → `scale_block` | **Cannot share.** Must sit at an identical L1 address on every group member (multicast destination), and is live across `scale_block`. |
| `cb_gamma_rm` | `rm_γ · has_gamma · C` | `C` | `{row: streams → 0, hidden: spans → C}` | `T_γ` | reader | compute | until `load_gamma_slice` completes | **Cannot share.** Concurrent with `cb_gamma_tiles` (`tilize` reads one, writes the other). Its lifetime is disjoint from every block-scoped buffer, but it is freed conceptually rather than reallocated; sharing it with `cb_normed` was rejected because the page formats differ (`T_γ` vs `T_in`) whenever `gamma_dtype != input_dtype`, which is an explicit TARGET axis. |
| `cb_gamma_tiles` | `has_gamma · C` | `C` | `{row: streams → 0 — one copy serves every block, hidden: spans → C}` | `T_γ` | reader (TILE) / compute-`tilize` (RM) | compute | whole kernel | **Cannot share.** Whole-kernel lifetime; waited upfront by `gamma_block` every block and never popped. This is what keeps gamma at one DRAM read per core rather than one per block. |
| `cb_normed` | `has_gamma · R · C` | `R · C` | `{row: spans → R, hidden: spans → C}` | `T_in` | compute | compute | `scale_block` → `gamma_block` | **Cannot share.** Concurrent with `cb_input_tiles` (`scale_block` reads that and writes this) and with `cb_output_tiles` (`gamma_block` reads this and writes that). Not allocated when gamma is absent — `scale_block` then packs straight into `cb_output_tiles` (Rule 3 pattern 1). Sharing with `cb_output_tiles` via an in-place transform was **rejected**: the writer is already a consumer of `cb_output_tiles`, so an in-place second pass would race the writer for pages the first pass pushed. |
| `cb_output_tiles` | `output_cb_depth · C` = `2·C` | `C` | `{row: streams → one tile-row at a time, hidden: spans → C}` | `T_in` | compute | writer (TILE out) / compute-`untilize` (RM out) | one block | **Cannot share.** Concurrent with `cb_normed`. Capacity exceeds the live set by 2×: depth 2 both overlaps the drain with compute and gives the writer a ≥ 4–8-tile batch behind one barrier (`double_buffer`). |
| `cb_output_rm` | `rm_out · rm_cb_depth · C` = `2·C` | `C` | `{row: streams → one tile-row, hidden: spans → C}` | `T_in` | compute | writer | one block, RM output only | **Cannot share.** Concurrent with `cb_output_tiles` (`untilize` reads one, writes the other). |

## Symbol table

Every non-block parameter in a capacity expression, with its bound and the predicate establishing it.

| Symbol | Meaning | Bound | Predicate establishing the bound |
|--------|---------|-------|----------------------------------|
| `R` = `block_row_tiles` | block extent along `row` | `1 ≤ R ≤ min(core_row_tiles, MAX_GATHER_TILES / G)` and `R ≤ (l1_cb_budget − fixed_bytes) / per_row_bytes` | The residency solve below (a closed form, not a search) plus the declared `MAX_GATHER_TILES` mechanism cap. |
| `C` = `core_w_tiles` | block extent along `hidden` | `1 ≤ C ≤ max_core_w_tiles`, where `max_core_w_tiles` is the largest `C` satisfying `fixed_bytes(C) + per_row_bytes(C) ≤ l1_cb_budget` at `R = 1` | The regime-selection function raises `G` until `C = ceil(tensor_w_tiles / G)` clears this bound. This is the **residency predicate** — it is what makes regimes R1/R2 reachable and R3 unnecessary within the declared universe. |
| `G` = `w_group_size` | cores per reduction group | `1 ≤ G ≤ min(tensor_w_tiles, grid_x · grid_y)`, and `G = w_group_cols · w_group_rows` with `w_group_cols \| grid_x`, `w_group_rows \| grid_y` | Mechanism caps: a group must be a rectangle for `Mcast2D`, and a core owning zero hidden tiles would hang the gather. |
| `input_cb_depth`, `output_cb_depth`, `rm_cb_depth` | buffer depths | `2` in Phase 0 | Explicit pipelining knobs; perf lamp P1 measures alternatives. |
| `MAX_GATHER_TILES` | cap on `R · G` | `64` (fp32 tiles = 256 KiB) | Declared mechanism cap; bounds `cb_stat_gather`, the only buffer whose capacity is a product of two extents. |
| `l1_cb_budget` | bytes available to CBs | `device.l1_size_per_core() − L1_RESERVE_BYTES`, `L1_RESERVE_BYTES = 131072` | Named host constant covering kernel binaries, stack, semaphores and allocator alignment — **not** a safety fraction. |
| `tensor_row_tiles`, `tensor_w_tiles`, `partial_w` | tensor geometry | derived from the shape, alignment-aware (`ceil`, per-image on the TILE path) | Formulas in `op_design.md` → Blocking Model → Axes. |

## Total per-core footprint

```
per_row_bytes  = T_in · (input_cb_depth + has_gamma) · C            # cb_input_tiles + cb_normed
               + T_f32 · (5 + has_tail + G)                         # cb_stat_sq, _partial, _sum, cb_rstd_send,
                                                                    #   cb_rstd, [cb_tail_masked], cb_stat_gather
fixed_bytes    = T_in · C · (output_cb_depth + rm_in·rm_cb_depth + rm_out·rm_cb_depth)
               + T_γ  · has_gamma · C · (1 + rm_γ)                  # cb_gamma_tiles [+ cb_gamma_rm]
               + T_f32                                              # cb_zero_tile
               + T_bf16 · (1 + has_tail)                            # cb_scaler [+ cb_wmask]

footprint(R)   = fixed_bytes + R · per_row_bytes

block_row_tiles = clamp( floor((l1_cb_budget − fixed_bytes) / per_row_bytes),
                         1,
                         min(core_row_tiles, MAX_GATHER_TILES / G) )
```

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
| `input_tensor` | **1×** (`N` bytes) | `cb_input_tiles` holds the whole block resident from `sumsq_block` through `scale_block`, so the apply pass re-reads it from L1, not DRAM. This is the residency decision the hidden-axis split exists to enable; without it the count would be 2× (regime R3). | none |
| `gamma` | **`num_row_groups ×`** (`num_row_groups · W_bytes`) | Structurally unreachable minimum: gamma does not vary along `row`, so each of the `num_row_groups` disjoint row-groups must have its own copy. *Within* a group it is read exactly once — the hidden split partitions it across the group's members. Read once per core for the whole kernel, never per block, because `cb_gamma_tiles` is never popped. | none (scheme lamp G1 would convert `num_row_groups − 1` of these reads into a multicast) |
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

### Corrected footprint expressions (what the host actually computes)

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

| Shape | `G` | row-groups | `C` | `R` | cores | DRAM crossings | Cross-core per block |
|-------|-----|-----------|-----|-----|-------|----------------|----------------------|
| `(1,1,8192,7168)` prefill | 5 | 22 | 45 | 3 | 110 | in 1x, out 1x, gamma 22x | `4*R` fp32 tiles gathered + `R` multicast to 5 |
| `(1,1,32,7168)` decode | 22 | 1 | 11 | 1 | 22 | in 1x, out 1x, gamma **1x** (the minimum) | `21*R` gathered + `R` multicast to 22 |
| `(1,1,8192,1024)` prefill | 1 | 110 | 32 | 3 | 110 | in 1x, out 1x, gamma 110x | degenerate (local copy) |

The decode row is the design's predicted best case: a single row-group cannot replicate gamma,
so DRAM traffic is exactly the named-boundary minimum. The `G = 22` (rather than 110) choice is
**perf lamp P2, now measured and adopted** — see `MAX_W_GROUP_SIZE` in
`rms_norm_program_descriptor.py` for the numbers.
