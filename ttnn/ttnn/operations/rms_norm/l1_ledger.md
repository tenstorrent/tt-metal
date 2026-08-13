# L1 Ledger: rms_norm

Schema and audits: `.claude/references/l1-footprint-discipline.md`. Block semantics:
`.claude/references/blocking-model.md` §6. Consumes the Blocking Model in `op_design.md`; restates
no blocking decision.

## Named block axes (every row below accounts for all of them)

| Axis | Extent knob | Phase 0 value |
|------|-------------|---------------|
| `row` | `block_rows` | `core_row_tiles` (whole per-core assignment) |
| `hidden` | `block_hidden_tiles` | `slice_hidden_tiles = ceil_div(tensor_hidden_tiles, num_hidden_slices)` |
| `contributor` (intermediate axis of `cb_gathered_partials`) | `num_hidden_slices` | selection function |

Shorthand in the expressions: `B = block_rows`, `S = block_hidden_tiles`, `s = num_hidden_slices`.

## Inventory-before-solve record (Rule 1 + Rule 3)

Applied **before** the budget predicate below was written. Buffers that a naive phase-per-buffer
layout would have created, and why they do not exist:

| Buffer that does not exist | Rule 3 pattern applied | Saving |
|---|---|---|
| `cb_masked_x` (W-masked copy of x) | **3.2 transform in place** — `mask_tail_block` reads and packs the same `cb_input_tiles` pages at `TileOffset::Strided{base=S−1, row_stride=S}`. Legal because the masked-out lanes lie outside the logical shape, so computing the whole pipeline on `x·mask` is *equivalent* to masking only the Σx² input. | `B·S` tiles |
| `cb_squared` (x² for a `reduce`-based Σx²) | **3.4 fold into the accumulator** — `sum_of_squares` accumulates `x·x` in DEST across each row (`DestAccumulation::PerRow`) and packs one tile per row, so the `S` squared tiles per row are never materialized. Also the measured-faster path (2.94× at 32 tiles). | `B·S` tiles |
| `cb_normalized` (`x·r` before the gamma multiply) | **3.2 transform in place** — `scale_block` packs `x·r` back into `cb_input_tiles`; `apply_gamma_block` reads it from there. | `B·S` tiles |
| `cb_output_tiles` **in the ROW_MAJOR path** | **3.2 transform in place** — the consumer there is `untilize`, another *compute* helper, and two sequential compute helpers own all three TRISCs and cannot overlap, so a CB between them would have to hold the producer's whole `B·S` output (`ttnn-cb-memory-fundamentals.md:122-154`). Instead the last normalize phase packs in place into `cb_input_tiles` and `untilize` reads it from there. This is a *correctness* fix as much as a footprint one: a `2S`-page buffer there would deadlock on `cb_reserve_back`. | `B·S` tiles on the heaviest layout |
| `cb_mean_sq` (Σx²/W + eps, before rsqrt) | **3.1 pack into the destination** + the `post_reduce_op` extension slot — `×1/W`, `+eps` and `rsqrt` all run on the reduce's DEST tile, so only `r` is ever packed. | `B` tiles, and one whole compute phase |
| `cb_slice_stat` when `s == 1` | **3.1 pack into the destination** — with no combine, `collapse_partial_block` fuses the finalize and packs straight into `cb_rms_recip`. The CB is not created. | `B` tiles |
| `cb_gamma_tiles` when `gamma is None` | not created; `apply_gamma_block` is elided and `scale_block` packs straight to `cb_output_tiles` (a `copy` pass would have been pure waste). | `S` tiles + one phase |

Three block-sized buffers removed. Only after that is the fit predicate below introduced, and it is
the **only** budget solve in the design — there is no safety fraction and no blocking search beyond
"largest `B ≤ core_row_tiles` that fits".

## Implementation deltas (recorded against the design's inventory)

Four departures from the design's inventory, all forced by a mechanism and all
verified on device:

| Delta | Why | Footprint effect |
|---|---|---|
| **`cb_slice_stat` does not exist.** The per-slice within-tile collapse is FUSED into the root's combine: contributors ship the RAW `cb_sq_partials` tile (all 32 columns carry a partial sum) and the root runs ONE `reduce<SUM, REDUCE_ROW>` over the gathered `(B, s)` block. | `ReduceWithinTile::Skip` is unreachable through `compute_kernel_lib::reduce()`: its "Skip is AccumulateViaAdd-only" `static_assert` (`reduce_helpers_compute.inl:886-891`) sits AFTER the `if constexpr (AccumulateViaAdd) { … return; }` block, so it is not in a discarded statement and fires for the AccumulateViaAdd instantiation too. Sum-then-collapse == collapse-then-sum, so the fused form is arithmetically identical and ships the same `B` tiles per contributor. | **−`B` tiles** on every core with `s > 1`, and one whole compute phase removed from every non-root core. `cb_sq_partials`' consumer becomes the writer (still exactly one producer, one consumer). |
| **`cb_thread_sync` added: 1 page, `bfloat16` tile.** Carries no data. One `cb_reserve_back`/`cb_push_back` (PACK-only) + `cb_wait_front`/`cb_pop_front` (UNPACK-only) round trip between two in-place stages. | Two consecutive chains addressing `cb_input_tiles` with caller-managed `(None, None)` policies exchange no CB handshake, so nothing orders chain *N*'s pack against chain *N+1*'s unpack of the same tile. The dst-sync window bounds the skew to a couple of tiles, so the race only bites at `B*S ≤ 2` — it passed under `--dev` and failed in production on exactly the narrow-W / single-tile-row shapes. | **+1 tile** (2 KB), constant — scales with nothing. |
| **Every CB is declared on ONE common core-range set** (the union of the active row-group rects), so `cb_gathered_partials` and `cb_rms_bcast` are allocated on non-root cores too. | Both cross-core mechanisms need a peer's L1 address to be derivable from the local one: the gather's destination is the root's `cb_gathered_partials` base, and `mcast_pipe` *requires* `dst_l1` (the `cb_rms_recip` landing) to be identical on every receiver (`mcast_pipe.hpp:44-45`). A per-core-varying CB set would give per-core-varying addresses. | **+`(s+1)·B` tiles** on non-root cores when `s > 1`. The design already told the host to size on the union (`fits()` is evaluated with `is_root=True`), so the *budget* is unchanged — only the allocation is now literal rather than notional. |
| **`cb_input_tiles` capacity is EXACTLY `B*S`, never more.** | The in-place rewrite needs `get_write_ptr() == get_read_ptr()`, which only holds when the CB is exactly full. A depth-2 `cb_input_tiles` would silently write `x·r` to the wrong half. This *reinforces* the design's `in_cb_depth = 1`; the overlap perf lamp must therefore be measured as "smaller `block_rows` + a second buffer", not "same block, deeper CB". | none (this is the design's value, now load-bearing) |

Additionally the reader pays a **one-time** NoC zero-fill of the regions the
per-block reads never touch — the tail tiles of a ragged hidden slice
(`cb_input_tiles`) and the W-gap inside every `cb_rm_stage_in` stick. It is
once-per-kernel and costs no capacity, because those bytes are inside buffers
the ledger already accounts for and are never rewritten (capacity == live set,
so every block reuses the same physical pages).

## The ledger

| CB | Capacity (pages) | Live set | Axis accounting | Page format | Producer | Consumer | Lifetime | Shares with / why not |
|----|------------------|----------|-----------------|-------------|----------|----------|----------|-----------------------|
| `cb_input_tiles` | `B * S` | `B * S` | `{row: spans → B, hidden: spans → S, contributor: n/a (not a cross-core buffer)}` | `tile_size(input.dtype)` | reader (TILE) / compute-`tilize` (ROW_MAJOR) | compute | whole block: `load_block` → `apply_gamma_block` (TILE) / `untilize_block` (ROW_MAJOR) | **Cannot share.** Live across *every* other block-scoped CB's lifetime by construction — holding x resident across the Σx² pass and the normalize pass is what keeps the input at one DRAM crossing. Capacity == live set (no double buffering: `in_cb_depth = 1`, see the overlap perf lamp). It **absorbs** three would-be buffers by in-place rewriting (`cb_masked_x`, `cb_normalized`, and `cb_output_tiles` on the ROW_MAJOR path). |
| `cb_gamma_tiles` | `S` | `S` | `{row: streams → the same S tiles serve every row, hidden: spans → S, contributor: n/a}` | `tile_size(gamma.dtype)` | reader | compute | whole kernel | **Cannot share.** Lifetime is the whole kernel, so it overlaps every other buffer. Also a different page format from every fp32 stat CB. Not created when `gamma is None`. |
| `cb_sq_partials` | `B` | `B` | `{row: spans → B, hidden: streams → S tiles folded into each page inside DEST, contributor: n/a}` | `tile_size(float32)` | compute | compute (`s == 1`) / writer (`s > 1`, it *is* the gather payload — delta #1) | `square_accumulate_block` → `collapse_partial_block` (`s == 1`) or → gather (`s > 1`) | **Cannot share with `cb_rms_recip`**: `reduce` runtime-asserts `input_dfb != output_dfb` (`reduce_helpers_compute.inl:895-897`), and the assert is compiled out in non-`--dev` builds, so aliasing them yields corrupt data rather than a halt. Lifetimes are adjacent, not disjoint (it is the reduce's *input* while the other is its *output*). |
| `cb_gathered_partials` | `s * B` (declared on **every** core of the rect when `s > 1`; live only on the root — delta #3) | `s * B` | `{row: spans → B, hidden: streams → the slice is summed away, contributor: spans → s}` | `tile_size(float32)` | reader | compute | `combine_block` | **Cannot share.** Its pages are the landing zone for `s−1` remote NoC writes whose destination addresses are baked into the contributors' runtime args; overlaying another buffer would let a peer overwrite live data. It is also the only buffer that spans the contributor axis. |
| `cb_rms_bcast` | `B` (declared on every core of the rect when `s > 1`; live only on the root — delta #3) | `B` | `{row: spans → B, hidden: streams, contributor: streams → it is the sum over s, held once}` | `tile_size(float32)` | compute | reader | `combine_block` | **Cannot share with `cb_rms_recip`**, which is the deliberate reason it exists: the mcast source is consumed by the reader and the mcast landing is produced by the reader. One CB would be a cross-kernel handoff aliased onto a compute buffer — the exact two-consumer race called out at `ttnn-cb-memory-fundamentals.md:104`. `mcast_pipe`'s loopback mode additionally *requires* `src_l1 != dst_l1` (`mcast_pipe.inl:84-90`). |
| `cb_rms_recip` | `B` | `B` | `{row: spans → B, hidden: streams → one r per row serves all S tiles, contributor: streams → already combined}` | `tile_size(float32)` | reader (`s>1`) / compute (`s==1`) | compute | `combine_block` → `scale_block` | **Cannot share with `cb_sq_partials`** (reduce in/out assert) or `cb_rms_bcast` (producer/consumer kernels differ). Lifetime overlaps `cb_input_tiles` (it is `scale_block`'s srcB while x is live). |
| `cb_scaler` | `1` — a bare literal, and the live set is genuinely constant (one scaler tile for the kernel's life) | `1` | `{row: streams, hidden: streams, contributor: streams}` — scales with **nothing** | `tile_size(bfloat16)` | reader | compute | whole kernel | **Cannot share.** Waited but **never popped** by `reduce` (`reduce_helpers_compute.inl:955`), so it is live for the whole kernel, and its bf16 format is a hard `static_assert` (`reduce_helpers_dataflow.inl:185-187`) that differs from every other CB's format when the input is fp32. |
| `cb_w_mask` | `1` — bare literal, constant live set | `1` | `{row: streams, hidden: streams → applies to one tile per row, contributor: streams}` — scales with nothing | `tile_size(bfloat16)` | reader | compute | whole kernel | **Cannot share with `cb_scaler`** even though both are 1-page bf16 constants and both are pushed once: `reduce` waits `cb_scaler` while `mask_tail_block`'s chain reads `cb_w_mask` as an FPU srcB with `BroadcastDim::Row`, so both are simultaneously live for the whole kernel. **Created only when `mask_enabled`** (`layout == TILE && W % 32 != 0`); the reader gates `prepare_reduce_mask` on the same predicate as a `if constexpr`, because that helper `static_assert`s on this CB's page *format* and so must sit in a discarded statement when the CB is absent. |
| `cb_thread_sync` | `1` — bare literal, carries no data | `1` (never read) | `{row: streams, hidden: streams, contributor: streams}` — scales with nothing | `tile_size(bfloat16)` | compute (PACK) | compute (UNPACK) | whole kernel, one push/wait round trip per in-place handoff | **Cannot share with `cb_scaler` / `cb_w_mask`**: those are waited-and-never-popped constants, whereas this one is pushed and popped on every in-place boundary; aliasing would consume a constant's page. It exists to supply the PACK→UNPACK ordering edge two caller-managed `(None, None)` chains on `cb_input_tiles` do not exchange (delta #2). |
| `cb_output_tiles` | `out_cb_depth * S` (`out_cb_depth = 2`) — **TILE layout only** | `S` | `{row: streams → window of out_cb_depth tile-rows, hidden: spans → S, contributor: n/a}` | `tile_size(output.dtype)` | compute | writer | per tile-row, across the whole block | **Capacity (2S) exceeds the live set (S) deliberately** — this is the double-buffer depth knob, whose 2.78× overlap win is measured at `examples/double_buffer/report.md:30-42`, and it is legal here *because* the consumer is the writer (a different processor). **Cannot share with `cb_input_tiles`**: both are live simultaneously throughout `apply_gamma_block` (source and destination), and merging them would give the writer a second consumer's view of a compute-owned CB. Does not exist on the ROW_MAJOR path — see the inventory record above. |
| `cb_rm_stage_in` | `rm_in_depth * S` **tile-sized** pages (`rm_in_depth = 2`) = the same bytes as `rm_in_depth * 32` sticks of pitch `S*32*elem`; the reader writes stick `k` at `k * RM_STICK_PITCH` inside the window it reserved, and `tilize` consumes `S` pages per tile-row (symmetric-page mode) | `S` pages = 32 sticks (one tile-row) | `{row: streams → window of rm_in_depth tile-rows, hidden: spans → a whole padded stick of S*32 elements, contributor: n/a}` | input dtype, stick-sized pages | reader | compute | per tile-row (ROW_MAJOR only) | **Capacity exceeds the live set by the depth knob** (overlaps 32 stick reads for tile-row `i+1` against `tilize` of tile-row `i`; legal because producer and consumer are different processors). **Cannot share with `cb_rm_stage_out`**: opposite direction, both live across the same block, different page granularity (sticks vs tiles), and merging them would give one CB two producers and two consumers. |
| `cb_rm_stage_out` | `rm_out_depth * S` tile pages (`rm_out_depth = 2`) | `S` | `{row: streams → window of rm_out_depth tile-rows, hidden: spans → S, contributor: n/a}` | output dtype, tile-sized pages (`untilize` has no asymmetric-page mode, `untilize_helpers.hpp:109-110`) | compute | writer | per tile-row (ROW_MAJOR only) | **Capacity exceeds the live set by the depth knob** (overlaps `untilize` against the writer's stick writes; legal because the consumer is the writer). **Cannot share with `cb_input_tiles`**: `untilize` reads one and writes the other in the same call (`untilize_helpers.inl:227-233`), so both are live, and `cb_input_tiles` has no dataflow consumer while this one does. |

## Symbol table — every non-block parameter, its bound, and the predicate establishing it

| Symbol | Meaning | Bound | Predicate establishing the bound |
|--------|---------|-------|----------------------------------|
| `s` = `num_hidden_slices` | cores combining over one row-group | `1 ≤ s ≤ min(tensor_hidden_tiles, grid_x · grid_y / num_row_groups)` | Mechanism cap: a row-group is exactly `grid_x × (grid_y / num_row_groups)` cores because `Mcast2D` needs a rectangle (`host/mcast_host.hpp:460-465`); and `s ≤ tensor_hidden_tiles` because a slice cannot be empty. `grid_x`, `grid_y` come from `device.compute_with_storage_grid_size()`. |
| `num_row_groups` | independent combine groups | divisor of `grid_y`, `≤ max(1, tensor_row_tiles)` | Same rectangle cap; and no more row-groups than tile-rows to hand out. |
| `out_cb_depth` | `cb_output_tiles` depth | `≥ 1`, default `2` | Host constant; a buffer-depth knob, not an op dimension. |
| `rm_in_depth`, `rm_out_depth` | ROW_MAJOR staging depths | `≥ 1`, default `2` | Host constants. |
| `elem`, `tile_bytes` | element / tile size of the relevant tensor | `elem ∈ {2, 4}`; `tile_bytes = ttnn.tile_size(dtype) ∈ {2048, 4096}` | `SUPPORTED["dtype"] ⊆ {bfloat16, float32}` in Phase 0. Stat CBs are pinned to `float32` ⇒ 4096 B. |
| `l1_align` | L1 alignment | `ttnn.get_l1_alignment()` (16 on WH/BH) | HAL query, never hardcoded. |
| `l1_working_budget` | CB budget per core | `L1_SIZE_PER_CORE_FALLBACK (1 MB) - L1_RESERVE (96 KB)` = **928 KB** | `device.l1_size_per_core()` is **not bound to Python** on this build, so the host falls back to one named constant. `ttnn.get_max_worker_l1_unreserved_size()` *is* bound (1 532 032 B on this part) but is keyed off `KERNEL_CONFIG` and overshoots by the kernel-config ringbuffer (`ttnn/core/services/h2d_socket_service.cpp:138`), so adopting it needs its own reserve and a measured re-run — recorded as a perf lever (a bigger budget means a coarser `block_rows`), not taken here. The current value is a **conservative** bound: every fit decision is made against less L1 than the part has. |
| `tensor_hidden_tiles`, `tensor_row_tiles` | whole-op dimensions | **bounded only inside the fit predicate**: they appear in no capacity expression. `S = ceil_div(tensor_hidden_tiles, s)` is bounded by the fit predicate below; `B ≤ core_row_tiles` is bounded by `fits(B, S)`. | The fit predicate is the bound. No CB capacity contains an unbounded op dimension. |

## Total per-core footprint

```
footprint_tiles(B, S, s, layout, has_gamma, is_root)
  =            B * S                      # cb_input_tiles          scales with row × hidden
  + (has_gamma ? S : 0)                   # cb_gamma_tiles          scales with hidden
  +            B                          # cb_sq_partials          scales with row
  + (is_root && s > 1 ? s * B : 0)        # cb_gathered_partials    scales with row × contributor
  + (s > 1 ? B : 0)                       # cb_rms_bcast            scales with row
  +            B                          # cb_rms_recip            scales with row
  +            3                          # cb_scaler + cb_w_mask + cb_thread_sync (scales with nothing).
                                          # Counted unconditionally even though cb_w_mask exists only
                                          # when mask_enabled — a deliberate 2 KB conservatism in the
                                          # predicate, never an over-allocation on device.
  + (layout == ROW_MAJOR                  # the two paths are exclusive:
       ? (rm_in_depth + rm_out_depth) * S #   rm staging            scales with hidden × depth
       : out_cb_depth * S)                #   cb_output_tiles       scales with hidden × depth
                                          # ROW_MAJOR has NO cb_output_tiles — the last normalize
                                          # phase packs in place into cb_input_tiles (see inventory).

footprint_bytes = footprint_tiles * tile_bytes            # stat CBs pinned fp32; see note
fits(B, S)      = footprint_bytes <= l1_working_budget
```

Note on formats: the expression is written in tiles for readability, but the host computes bytes
per-CB because the formats differ — `cb_input_tiles` / `cb_output_tiles` / RM staging use the tensor
dtype, the six stat CBs are `float32` (4096 B), and `cb_scaler` / `cb_w_mask` are `bfloat16`
(2048 B). The `float32`-everywhere reading is the conservative upper bound and is what the regime
reachability argument in `op_design.md` uses.

Which terms scale with which knob:

| Term | Scales with | Dominance |
|------|-------------|-----------|
| `B * S` | both block extents | **dominant** for every realistic shape — this is the term the fit predicate solves against |
| `out_cb_depth * S` (TILE) *or* `(rm_in_depth + rm_out_depth) * S` (ROW_MAJOR) | `hidden` extent × a depth knob | secondary; `2S` (TILE) or `4S` (ROW_MAJOR) — mutually exclusive terms |
| `s * B` | `row` extent × `contributor` extent | root cores only; bounded by `s ≤ grid_x·grid_y` and `B ≤ core_row_tiles` |
| `S`, `4B`, `2` | one extent, or nothing | negligible |

The root core carries `s*B + B` tiles more than a non-root core in the same rect. The host must size
CBs on the **union** of core ranges, i.e. every core in a rect is allocated the root's footprint (or
the root's rect is declared as its own `CBDescriptor` core range) — otherwise the fit predicate is
evaluated against the wrong core. `fits()` above is evaluated with `is_root=True`, which is the
conservative choice.

---

## Data-movement budget

Named memory boundary: **DRAM**. Minimum = *each input crosses once, each output crosses once*, i.e.
`input + output + gamma = (2·Rt·Wt + Wt) · T` where `Rt = tensor_row_tiles`, `Wt = tensor_hidden_tiles`,
`T = tile bytes`, `g = num_row_groups`, `s = num_hidden_slices`.

| Tensor | DRAM crossings | Why that many | Cross-core traffic added |
|--------|----------------|---------------|--------------------------|
| `input_tensor` (x) | **1** (`Rt·Wt·T` bytes) | `cb_input_tiles` holds the whole block resident across `square_accumulate_block`, `scale_block` and `apply_gamma_block`. The residency is bought by the hidden split, which shrinks each core's reduced extent — this is the residency half of the dependent-axis split, not just its parallelism half. | none |
| `gamma` | **`g`** (`g·Wt·T` bytes) | Each row-group's line of cores collectively reads all of gamma exactly once (slices partition the hidden axis), but the `g` row-groups do not share — gamma is invariant along the row axis, so it is reuse-shared *by construction of the row split*. `g = 1` in the wide/decode regime ⇒ minimum; `g = grid_y`-divisor in the row-heavy regime ⇒ the excess. | none (the **GammaBroadcast** scheme lamp would move this term onto the NoC and reach the minimum) |
| `output` | **1** (`Rt·Wt·T` bytes) | `cb_output_tiles` is drained by the writer as compute packs; nothing is re-read or read-modify-written. | none |
| partial Σx² (intermediate, never in DRAM) | **0** | Lives entirely in L1 and on the NoC. | gather `Rt·(s−1)·T` bytes (unicast, `B` single-tile writes per contributor per block, stride `s` pages); mcast `Rt·T` bytes with fan-out `s−1`; `g · num_blocks_this_core` semaphore round-trips |

**Totals per tier**

| Tier | Bytes |
|------|-------|
| DRAM (most expensive) | `(2·Rt·Wt + g·Wt) · T` |
| Cross-core NoC (moderate, latency-dominated) | `Rt·s·T` payload + `g · num_blocks_this_core` handshakes |
| Core-local L1 (free) | the two in-place rewrites of `cb_input_tiles` (`mask_tail_block`, `scale_block`) and the root's copy of its own partial into `cb_gathered_partials` |

Worked, bf16 (`T = 2 KB`), for the two decisive `feature_spec.py` perf shapes:

| Shape | `Rt`, `Wt` | Chosen `g`, `s` | DRAM | NoC |
|---|---|---|---|---|
| decode `(1,1,32,7168)` | 1, 224 | 1, 56 | `2·224·2K + 1·224·2K` = **1.34 MB** = the minimum | `1·56·2K` = 112 KB payload, 1 handshake |
| prefill `(1,1,8192,1024)` | 256, 32 | 8, 8 | `2·256·32·2K + 8·32·2K` = 32.0 + 0.5 = **32.5 MB** | `256·8·2K` = 4.0 MB payload, 8 handshakes |
| prefill, `s = 1` corner (pure row-parallel) | 256, 32 | 64, 1 | `32.0 + 4.0` = **36.0 MB** (+11 %) | 0 |

> **Cheapest-traffic split considered:** `num_row_groups = 1`, `num_hidden_slices = min(Wt, grid_x·grid_y)`
> — gamma partitions across one group, so DRAM reaches the minimum `(2·Rt·Wt + Wt)·T` exactly.
> **Delta vs. implemented:** in the wide/decode regime the implemented split **is** this split
> (`g = 1`) — delta **0 bytes on every tier**, and the cheapest split is implemented. In the row-heavy
> prefill regime the implemented split is `g = 8`, `s = 8`, which costs **+0.44 MB DRAM**
> (`(8−1)·32·2K`) and **saves 28 MB of NoC** relative to `g = 1, s = 64` (`256·64·2K` = 32 MB vs
> `256·8·2K` = 4 MB) — so the *global* cheapest-traffic split is not `g = 1` there once the moderate
> tier is counted, and the implemented `g = 8` is the cheapest point on the combined budget.
> **Implemented:** the 2D partition (`g` = the largest divisor of `grid_y` that is `≤ Rt` and leaves
> `≥ hidden_tiles_per_core_floor` hidden tiles per core).
> **Lamped, with its delta:** (a) **GammaBroadcast** — removes the residual `(g−1)·Wt·T` DRAM term
> (0.44 MB / 1.4 % for prefill at `g=8`; **3.5 MB / 11 %** against the `s=1` corner a floor-of-1
> heuristic would pick), at the cost of a second `Mcast1D(PerColumn)` family on disjoint semaphore
> ids; unmeasured because the catalog's mcast entries show the device win landing well below the
> read-count reduction (`mcast_topology/report.md:8-10`: 1.91× device for 8× fewer reads).
> (b) **Reduce-scatter the contributor axis** — same bytes, redistributes the root's
> `B·(s−1)` add_tiles across the rect; unmeasured because at our 1–8-tile payload the measured latency
> floor favours root/tree (`tensix_all_reduce/report.md:85-99`).
> **The structure keeps both reachable:** gamma is loaded by a separate once-per-kernel named
> operation over a per-core hidden slice with its own CB, and the combine is a single named operation
> whose contributor page layout (`row·s + contributor`) is stated — neither change touches the loop
> nest, the block extents, or any other CB.
