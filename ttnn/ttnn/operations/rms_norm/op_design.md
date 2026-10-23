# Operation Design: rms_norm

Terminology used throughout (qualified once here, bare afterwards):
*core* = Tensix core (`CoreCoord`); *layout* = `ttnn.Layout` (`ROW_MAJOR_LAYOUT` / `TILE_LAYOUT`);
*placement* = `TensorMemoryLayout`; *tile* = 32×32 elements, *tile bytes* = 2048 B (bf16) /
4096 B (fp32); *page* = a CB/buffer page (a tile unless stated "stick"); *block* = the blocking-model
block of §Blocking Model (never a bfp8 block, never `BLOCK_SHARDED`); *DEST* = the DEST register
file. Every count names its unit.

## Overview

| Field | Value |
|-------|-------|
| Classification | fused (reduction + eltwise), single native dispatch (`ttnn.generic_op`, one `ProgramDescriptor`) |
| Goal | Normalize every row of the last dimension by its root-mean-square and optionally scale by a per-column weight, natively for `TILE_LAYOUT` and `ROW_MAJOR_LAYOUT`, tile-aligned and non-aligned shapes, bf16 and fp32, on the whole Tensix grid from Phase 0. |
| Math | `out[..., i, w] = x[..., i, w] * rsqrt( (1/W) * Σ_{w'<W} x[..., i, w']² + epsilon ) * (gamma[w] if gamma else 1)` |
| Mode | Derivative (generic_op + kernel-lib helpers) |
| References | `.claude/references/blocking-model.md`; `.claude/references/l1-footprint-discipline.md`; `.claude/references/precision_convention.md`; `ttnn/ttnn/operations/examples/master.md` (entries listed in §Catalog evidence); `eval/golden_tests/rms_norm/feature_spec.py`; `eval/golden_tests/rms_norm/helpers.py` (reference `pytorch_rms_norm`, `run_rms_norm` call contract) |

### Catalog evidence (which measured entry informed which knob)

| Catalog entry (`examples/master.md`) | Knob / decision it informs |
|---|---|
| `width_split` (BH pays from Wt ≥ 2 tile-columns; 7.22× at Wt=64) | the W-split regime R2 exists to fill the grid when tile-rows < cores |
| `tensix_all_reduce` (1 tile per core: root reduce beats reduce-scatter, BH 0.75×) | R2 combine = root gather + multicast; reduce-scatter / tree deferred (R4) |
| `tensix_all_reduce_compute` (pairwise FPU add into fp32 DEST, 9.37× BH) | root combine uses `reduce<…, AccumulateViaAdd, ReduceWithinTile::Skip>` (the helper's realization of that pattern) |
| `row_reduce_accumulate`, `reduce_accumulate`, `reduce_block` (ReduceTile wins at ≤ 8 tiles per output on BH) | per-core row-sum uses `ReduceAlgorithm::ReduceTile` over ≤ 2 tiles per row |
| `eltwise_l1_vs_dest_accumulate` (DEST accumulation 2.01× at 64 steps BH) | sum-of-squares accumulates across the W slice in DEST, never through L1 |
| `compute_fusion` (FPU consumer: pack-to-L1 beats DEST reuse, 0.93× BH for reuse) | γ-multiply and inv-multiply are two FPU chains joined by `cb_x_gamma` (L1), not a DEST-reuse chain |
| `compute_block_size` (fixed ~1.4 µs per pass on BH; coarsest block wins) | `block_rows` and `chunk_w_tiles` default to the coarsest that fits L1 |
| `double_buffer`, `noc_placement` (batch ≥ 4–8 reads per barrier; row-major core order) | readers/writers issue one barrier per chunk of tiles; cores enumerated row-major |
| `shared_input_reuse`, `mcast_topology` (multicast pays only with ≥ 8–16 tiles per handshake on BH) | gamma broadcast across row groups deferred (R5) |
| `sfpu_tile_scope` (rsqrt on row/col-scoped vectors 7.3×) | perf lamp L4 on the post-reduce SFPU scope |

## Parameters

| Name | Type | Required | Valid Range | Default | CT/RT |
|------|------|----------|-------------|---------|-------|
| `input_tensor` | `ttnn.Tensor` | yes | rank ≥ 2, on device | — | buffer address RT |
| `gamma` (keyword-only) | `Optional[ttnn.Tensor]` | no | logical volume == W and `shape[-1] == W` (canonical shape `(1,1,1,W)`) | `None` | presence + layout CT, address RT |
| `epsilon` (keyword-only) | `float` | no | finite, ≥ 0 | `1e-6` | fp32 bit pattern RT |
| `compute_kernel_config` (keyword-only) | `ttnn.ComputeConfigDescriptor` | no | `fp32_dest_acc_en=True` in Phase 0; any `math_fidelity` / `math_approx_mode` | `default_compute_kernel_config()` (HiFi4, fp32_dest_acc_en=True, math_approx_mode=False) | passed as-is: `config=compute_kernel_config` on the compute `KernelDescriptor` |
| `memory_config` (keyword-only, extension) | `Optional[ttnn.MemoryConfig]` | no | Phase 0: `None` or an interleaved config | `None` → `ttnn.DRAM_MEMORY_CONFIG` | host only |

`memory_config` is not in the mandated signature but **must be accepted**: `eval/golden_tests/rms_norm/helpers.py`
passes `memory_config=ttnn_input.memory_config()` for every sharded cell, and those cells must reach
`validate()` (NotImplementedError → xfail), not die with a `TypeError`. All mandated call patterns are unaffected.

Entry-point contract (in this order): (1) structural validation → `ValueError` for rank < 2, gamma `shape[-1] != W`,
gamma logical volume ≠ W; (2) resolve `compute_kernel_config` through the exported
`default_compute_kernel_config()` factory (never inline the default); (3) `validate()` support gate
(NotImplementedError family) reading `input_tensor.dtype`, `.layout`, `.memory_config().memory_layout`, the
resolved `fp32_dest_acc_en`, gamma presence/dtype/layout, and the `INPUT_TAGGERS`; (4) build one
`ProgramDescriptor`; (5) one `ttnn.generic_op([input, (gamma,) output], descriptor)`. No `to_layout`, `tilize`,
`pad`, `slice` or any second dispatch anywhere.

Taggers the op must export (names the golden suite already uses): `tag_alignment` → `"tile_aligned"` |
`"w_non_aligned"` | `"h_non_aligned"` (feature_spec docstring: W checked first, then H); `tag_rank` → `len(shape)`;
`tag_gamma_dtype` / `tag_gamma_layout` → `"none"` when `axes["gamma_mode"] == "no_gamma"`, else pass the axis value
through. `"none"` is always legal for both gamma axes.

## Tensors

### Input

| Property | Requirement |
|----------|-------------|
| Shape | rank ≥ 2; `W = shape[-1]`, `H = shape[-2]`, leading dims arbitrary (rank 2/3/4 in TARGET) |
| Dtype | Phase 0: `bfloat16`, `float32` (`bfloat8_b` is a TARGET refinement) |
| Layout | `TILE_LAYOUT` and `ROW_MAJOR_LAYOUT`, both native |
| Memory | Phase 0: `TensorMemoryLayout.INTERLEAVED` (DRAM or L1); sharded placements are TARGET refinements (§Dataflow Strategy, placement table) |
| Gamma | optional; `(1,1,1,W)`; `ROW_MAJOR_LAYOUT` (primary) or `TILE_LAYOUT`; dtype bf16 / fp32, may differ from the input dtype; interleaved |

### Output

| Property | Value |
|----------|-------|
| Shape | identical to input (logical and padded) |
| Dtype | input dtype |
| Layout | input layout |
| Memory | `memory_config` if given, else `ttnn.DRAM_MEMORY_CONFIG` (interleaved) |

## Blocking Model

The first design section. Everything below it is downstream of these decisions.
Semantics: `.claude/references/blocking-model.md`.

### Index-space geometry (alignment-aware, the one place these are defined)

| Symbol | Definition | Note |
|--------|------------|------|
| `tensor_w_elems` (W) | `shape[-1]` | logical, used for `1/W` |
| `tensor_w_tiles` | `ceil(W / 32)` | tile-columns of a row |
| `last_tile_valid_cols` | `W − 32·(tensor_w_tiles − 1)` ∈ [1, 32] | 32 ⇔ W tile-aligned |
| `tensor_rows_tiles` (TILE layout) | `prod(shape[:-2]) · ceil(H / 32)` | every `(…, H, W)` image tile-padded independently — never `floor(prod·H/32)` |
| `tensor_rows_sticks` (RM layout) | `prod(shape[:-1])` | sticks; rows are independent, so 32 consecutive sticks form a tile-row across image boundaries |
| `tensor_rows_tiles` (RM layout) | `ceil(tensor_rows_sticks / 32)` | last tile-row holds `tensor_rows_sticks − 32·(tensor_rows_tiles−1)` valid sticks |
| `num_cores_available` | `grid.x · grid.y`, `grid = device.compute_with_storage_grid_size()` | runtime query, never a constant |
| `l1_budget_bytes` | `device.l1_size_per_core() − ttnn.device.GetAllocatorBaseAddress(device, ttnn.BufferType.L1)` | worker-L1 bytes available to CBs (Phase 0 tensors live in DRAM); no safety fraction |

### Axes

| Axis | Character (+ one-clause reason) | Extent knob | Phase 0 value | Knob source | Core-assignment | Later unlock |
|------|--------------------------------|-------------|---------------|-------------|-----------------|--------------|
| row (tile-rows of the folded `[rows, W]` view) | **independent** — every output row depends only on its own row | `block_rows` (tile-rows per block) | coarsest that fits: `max b ≤ rows_hi` with `footprint(b, …) ≤ l1_budget_bytes`; forced to 1 when `chunks_per_row > 1` (mechanism cap M5) | `BlockingPlan.block_rows` (host `_select_blocking`) | balanced split of `tensor_rows_tiles` over `num_row_groups` row groups (two classes, `ttnn.split_work_to_cores` semantics); all `w_group_size` cores of a group share its rows | knob-turn (depth, block size) |
| W (hidden, the reduced dimension) | **dependent** — the mean spans it (cross-block combine required) | `core_w_tiles` (tile-columns per core slice = the block's W extent; the whole slice is resident) | `ceil(tensor_w_tiles / w_group_size)` | `BlockingPlan.core_w_tiles` | split across the `w_group_size` cores of a W-group: core `g` owns tile-columns `[g·core_w_tiles, min((g+1)·core_w_tiles, tensor_w_tiles))`; `w_group_size = 1` ⇒ not split | built in Phase 0 (R2); sharded realizations are placement knob-turns |
| W, streaming quantum | (same axis — the quantum every *streaming* buffer moves along W; the resident block is not cut) | `chunk_w_tiles` | `core_w_tiles` if it fits, else the largest balanced chunk ≥ `chunk_floor_tiles` (= 8) that fits | `BlockingPlan.chunk_w_tiles`; `chunks_per_row = ceil(core_w_tiles / chunk_w_tiles)`, `row_stride_tiles = chunks_per_row · chunk_w_tiles` derived | per core (local) | knob-turn |
| gamma operand vs. row axis | **reuse-shared** — gamma does not vary along the row axis, which is the primary split (operand-reuse check) | — (resident for the whole kernel: `row_stride_tiles` tiles) | read once per core | `BlockingPlan.core_w_tiles` | each core reads only its W slice once | gamma multicast across row groups (R5, deferred) |
| intermediate: per-row partial sums, row axis | independent (inherited) | `block_rows` (same knob) | same | same | stays with the row group that produced it | — |
| intermediate: per-row partial sums, W-group member axis (`g`) | **dependent** — the RMS needs all `w_group_size` partials | `w_group_size` | from `_select_blocking` | `BlockingPlan.w_group_size` | **communication role:** member 0 of each group (root) collects, sums, finalizes `rsqrt`, multicasts. The compute is `block_rows` output tiles (1 tile in every grid-filling R2 plan, since each row group then owns ≤ 2 tile-rows — see R4) summed over `w_group_size` inputs: there is no row work to spread, and a `w_group_size`-way sum of one tile is not divisible across cores without a second round. Recorded, not hidden: the other members idle for the root's sum + `rsqrt` | tree / reduce-scatter combine (R4, deferred) |

Single source of truth: one host function `_select_blocking(geometry, dtypes, layout, has_gamma, gamma_layout, grid, l1_budget_bytes) -> BlockingPlan` (frozen dataclass). Every CB page count, loop trip count, kernel CT/RT arg and core range is derived from the `BlockingPlan` fields; no block, chunk, depth or core-count literal appears anywhere else (host or kernel).

### Buffer-depth knobs

| CB | Depth knob | Phase 0 value | What the depth buys |
|----|------------|---------------|---------------------|
| `cb_x_tiles` | `x_depth_blocks` | 1 block (`block_rows × row_stride_tiles` tiles) | pops are per chunk (`block_rows × chunk_w_tiles` tiles, contiguous front — cap M5), so the reader refills the next block while the γ / normalize phases drain this one; depth 2 = full read/compute overlap (lamp L1) |
| `cb_x_rm_sticks` (RM) | `stream_depth` | 2 × `chunk_w_tiles` tiles | reader fetches the next (row, chunk) stick slab while compute tilizes the current one |
| `cb_out_tiles` (TILE layout) | `stream_depth` | 2 × `block_rows × chunk_w_tiles` tiles | writer drains one chunk while compute produces the next |
| `cb_out_tiles` (RM layout) | — (sequential helpers on compute) | 1 × `block_rows × chunk_w_tiles` | normalize → untilize are sequential on the compute thread: full window, no depth |
| `cb_out_rm_sticks` (RM) | `stream_depth` | 2 × `block_rows × chunk_w_tiles` tiles | writer drains one chunk of sticks while compute untilizes the next |
| `cb_x_gamma` | `xg_lookahead_chunks` | `chunks_per_row` if `w_group_size > 1` and it fits, else 1 (selector falls back to 1 before widening the W-group) | lookahead = how many chunks of `x·γ` are computed before the inverse-RMS is needed — in R2 the whole block is scaled inside the combine stall (stall-shadow S1) |

### Mechanism caps

| Mechanism | Cap on which extent | Clamp | What happens unclamped |
|-----------|--------------------|-------|------------------------|
| M1 cross-core addressing of `cb_rowsum_gather` / `cb_inv_rms` (remote NoC writes target the *local* CB address) | none on extent; requires every CB to be declared **once, over the whole used-core range, at one nominal size** | all CB capacities use nominal quanta (`core_w_tiles`, `block_rows`, `chunk_w_tiles`) on every core, including the ragged last W slice and the ragged last block | per-class CB sizes shift later CB addresses per core class → partials/inverse-RMS land in the wrong L1 bytes (silent corruption) |
| M2 empty W slices | `w_group_size` | after `core_w_tiles = ceil(tensor_w_tiles / gw)`, recompute `w_group_size = ceil(tensor_w_tiles / core_w_tiles)` | `gw=4, tensor_w_tiles=9` gives a member with 0 valid tiles → divide-by-nothing slice, hangs on its partial |
| M3 grid capacity | `w_group_size ≤ num_cores_available` | `_select_blocking` raises `RuntimeError("rms_norm: W too wide for the grid")` if no `gw ≤ num_cores_available` fits L1 | allocation failure / CB clash at program build |
| M4 `tilize` / `untilize` width is a template parameter (`tilize_helpers.hpp:187-197`, `untilize_helpers.hpp:145-154`) | `chunk_w_tiles` is a compile-time arg | one CT value per program (nominal); ragged chunks are handled by narrowing the NoC transfers, not the helper width | different per-core widths would need per-class kernels and break M1 |
| M5 FIFO pops are front-contiguous | x is popped per chunk (`block_rows × chunk_w_tiles` tiles) | if `chunks_per_row > 1` then `block_rows = 1` (a chunk of a multi-row block is not contiguous in row-major `[row][col]` order) | popping a non-contiguous region frees tiles still needed / reader overwrites live x |
| M6 CB ring wrap (pointer resets only on exact `fifo_limit`) | every push/pop quantum | nominal quanta everywhere: x pushes/pops `block_rows × chunk_w_tiles` (capacity = `x_depth_blocks × chunks_per_row ×` that), streaming CBs push the nominal chunk even when its last tiles are not valid, ragged last block keeps nominal `block_rows` | pointer walks past the CB into neighbouring L1 on a core's second block |
| M7 BH DRAM read alignment 64 B (`NOC_DRAM_READ_ALIGNMENT_BYTES`) | RM stick sub-reads, TILE-gamma row-0 reads | RM slice / chunk byte offsets are multiples of `32 × elem` ≥ 64 B; TILE-gamma reads land at in-page offsets 0 and `256·elem` (512 / 1024 B), each `round_up(16·elem, ttnn.get_dram_alignment())` bytes | misaligned DRAM read returns shifted data |
| M8 reduce partial scaler covers only the **last** tile of a `reduce()` row (`reduce_helpers_compute.hpp:332-336`) | the masked tile must stay uncollapsed | sum-of-squares keeps `P_last` (the last valid tile's x², per row) separate from `P_full` (Σ of the others) | masking a collapsed tile erases valid lanes / or padding (poisoned to 1000 by the golden suite) leaks into the RMS |
| M9 stale L1 under the RM tail (`read_sticks_for_tilize` never clears the padded stride: `tilize_helpers_dataflow.inl:122-123`) | bytes `[valid_bytes, round_up(valid_bytes, 32·elem))` of each stick in the last valid tile | reader zero-fills them (`fill_l1_range`, `l1_helpers.hpp:89-90`) before push | NaN/Inf bit patterns × 0 mask = NaN in the row sum |
| M10 DEST capacity at fp32 (4 tiles half-sync) | none on block extents — every phase keeps ≤ 1 accumulator per output tile (PerRow DEST accumulation, one reduce output per row) | chains clamp their lane block internally (`chain.hpp:435-440`); untilize picks block-based pack when width > DEST (`untilize_helpers.hpp:76-77`) | — |
| M11 `prepare_reduce_scaler` valid count ∈ [1, 32] | `last_tile_valid_cols` | passed at runtime per core (32 for cores whose last valid tile is full) | — |

### Regimes

| Regime | Status | Predicate | Block | Data movement vs. minimum (boundary: DRAM; minimum = x read once, out written once, gamma read once) | What a bigger block buys |
|--------|--------|-----------|-------|---------------------------|--------------------------|
| R1 `row_local` | **built** | `_select_blocking(...).w_group_size == 1` | `block_rows × core_w_tiles` (= `tensor_w_tiles`), resident | x 1×, out 1×; gamma `num_row_groups`× (each core re-reads its slice once — reuse-shared operand, ≤ 6.25 % of x bytes, R5) | per block: tilize/sum-of-squares/reduce/γ/normalize inits and data-format reconfigs (intended: once per block per phase), one `cb_x` wait, `chunks_per_row` chain pairs; per chunk: one read barrier and one write barrier |
| R2 `w_split_root_combine` | **built** | `w_group_size ≥ 2` | `block_rows × core_w_tiles` (the core's resident slice) | x 1×, out 1×, gamma `num_row_groups`×; **plus cross-core per block:** `(w_group_size−1) · block_rows` fp32 partial tiles unicast to the root, `block_rows` fp32 inverse-RMS tiles multicast to `w_group_size` cores | R1's terms **plus** one combine round per block (semaphore round trip, root's `w_group_size`-way sum + rsqrt, one multicast): larger `block_rows` pays the round once for more rows |
| R3 `two_pass_restream` (row split, x streamed twice: once for Σx², once to normalize) | **rejected** — dead end; superseded by R2, which keeps every slice resident at x 1× | — | `block_rows × chunk` streamed | x **2×** DRAM | — |
| R4 `w_split_tree_or_reduce_scatter_combine` | **deferred** — the built root combine is the measured winner for the payload R2 produces: `block_rows` tiles per core per round (1 tile whenever R2 is chosen to fill the grid, because then each row group has ≤ 1–2 tile-rows), where catalog `tensix_all_reduce` measures reduce-scatter at 0.75× (BH, 1 tile/core) and tree ≈ root on 1-D groups (W-groups are consecutive row-major runs). Reachable: the gather CB, semaphores and `Mcast` channel are already per-group objects; a tree adds one intermediate leader stage in the writer | `w_group_size ≥ 2` and (future) `w_group_size ≥ tree_threshold` | as R2 | as R2, fewer root ingress tiles per round | as R2 |
| R5 `gamma_row_group_mcast` (reuse-shared broadcast of gamma) | **deferred** — saving bounded: gamma bytes per core ≤ `row_stride_tiles × 2·round_up(16·e_g, 64)` B vs. ≥ `row_stride_tiles × 1024·e_x` B of x per core, i.e. ≤ 6.25 % of x traffic in every dtype pair, and only when a core holds a single tile-row; catalog `shared_input_reuse` needs ≥ 8–16 tiles per handshake to pay. Reachable: `cb_gamma_tiles` is produced by the writer's one-time gamma stage; swapping that stage's source from DRAM to a multicast receive touches nothing else | `num_row_groups ≥ 2` | — | gamma 1× instead of `num_row_groups`× | — |
| R6 `w_split_block_pipelined` (stall-shadow S2: compute sum-of-squares of block b+1 while block b's combine is in flight) | **deferred** — applies only when `w_group_size ≥ 2` **and** `num_blocks_this_core ≥ 2`, which `_select_blocking` produces only when L1 forces the W-split on a many-row tensor; there the combine is paid once per block whose slice is the largest that fits, so it is amortized over the maximal per-core work. Reachable: needs `x_depth_blocks = 2` (a knob) and two-slot `cb_rowsum_gather` / `cb_inv_rms` addressing (`slot = block_idx % 2`), both expressible without changing the phase structure | `w_group_size ≥ 2 ∧ num_blocks_this_core ≥ 2` | as R2 | as R2 | hides the combine latency entirely |
| R7 sharded placements (HEIGHT → R1 with pinned geometry; WIDTH / BLOCK → R2 with pinned geometry) | **deferred** — the logical regimes are built in Phase 0, so these are placement refinements, not algorithms; the Phase 0 SUPPORTED rectangle is `memory_layout = INTERLEAVED` (op requirement scope) | input `memory_config().memory_layout != INTERLEAVED` | **one block = the whole resident shard** (re-chunk only under L1 pressure, mechanism caps apply) | x 0× DRAM (already in L1), out 0× when output sharded; consumed zero-copy | — |

Selection function (exact; pinned, regime-pinned tests required — §Work Distribution):

```
CHUNK_FLOOR_TILES      = 8      # BlockingPlan constant; grid-sync / per-call lamp L3
MIN_W_TILES_PER_CORE   = 8      # BlockingPlan constant; grid-sync lamp L2
C, R, Wt = num_cores_available, tensor_rows_tiles, tensor_w_tiles
gw_fill = min(C // R, Wt // MIN_W_TILES_PER_CORE) if (R < C and C // R >= 2 and Wt >= 2*MIN_W_TILES_PER_CORE) else 1
for gw_try in range(max(gw_fill, 1), C + 1):                 # smallest W-group first: fewest combines
    n  = ceil(Wt / gw_try); gw = ceil(Wt / n)                 # M2
    gr = min(R, C // gw); rows_hi = ceil(R / gr)
    for k in 1, 2, ...:                                       # largest balanced chunk first
        chunk = ceil(n / k)
        if chunk < min(n, CHUNK_FLOOR_TILES): break
        for L in ([k, 1] if gw >= 2 else [1]):                # R2: whole-block x·γ lookahead first (stall fill S1)
            if footprint(block_rows=1, n, chunk, k, L, gw) <= l1_budget_bytes:
                block_rows = 1 if k > 1 else max b in [1, rows_hi] with footprint(b, n, chunk, 1, L, gw) <= l1_budget_bytes   # M5
                return BlockingPlan(gw, n, gr, rows_hi, block_rows, chunk, chunks_per_row=k, xg_lookahead_chunks=L, ...)
raise RuntimeError("rms_norm: W too wide for the grid")     # M3
```

`footprint(...)` is the closed form in `l1_ledger.md` (§Footprint). Regime R1 ⇔ `gw == 1`, R2 ⇔ `gw ≥ 2`. Order of
responses to L1 pressure, by design: shrink `block_rows`, then `chunk_w_tiles` (down to 8 tiles; in R2 the whole-block x·γ lookahead is dropped to 1 chunk at each chunk size before going smaller), then raise `w_group_size`
— a smaller block or chunk only adds per-pass fixed cost; a larger W-group adds a cross-core round per block.

Domain notes per row: R1 covers every tensor whose tile-rows fill the grid (or whose W is too narrow to split:
`tensor_w_tiles < 16`) and whose slice fits; R2 covers few-row tensors (decode `32×7168`, the `_WIDE` loose cases
`32×16384`, `32×32768`, `64×12288`) and L1-forced widths (e.g. fp32 `4096×8192` with gamma). No other regime applies to
INTERLEAVED inputs: R3 is strictly worse than R2 on DRAM bytes, and R4–R6 are refinements on top of R2's structure.

### Traffic ranking

| Candidate split | DRAM crossings (x / out / gamma) | Cross-core traffic | Verdict |
|---|---|---|---|
| row split, slice resident (R1) | 1 / 1 / `num_row_groups` slices | none | **cheapest** (DRAM minimum; gamma replication ≤ 6.25 % of x) — default whenever it fills the grid and fits |
| W split + root combine, slice resident (R2) | 1 / 1 / `num_row_groups` slices (fewer groups than R1) | per block per group: `(gw−1)·block_rows` × 4096 B unicast + `block_rows` × 4096 B multicast to `gw` cores | ties R1 on DRAM, adds a few fp32 tiles of NoC per block; chosen when the row split under-fills the grid (occupancy tiebreak — DRAM is equal, so it is legal) or when L1 forces it (residency: the dependent-axis split is what keeps x at 1× instead of 2×) |
| row split, two-pass restream (R3) | **2** / 1 / … | none | rejected: re-reads x |
| split the 32 rows inside a tile-row across cores | 1 / 1 / … | — | impossible at tile granularity (TILE layout) and would fragment RM transfers below 1 tile row; not a candidate |

Implemented split = cheapest split (R1) wherever R1 fills the grid; R2 everywhere else. Both are at the DRAM minimum
for x and out.

### Block schedule

Logical schedule (reader, compute, writer realize their parts asynchronously):

```cpp
// once per kernel
prepare_scalers();           // reader:  cb_reduce_scaler ← [ones(32 valid), ones(last_tile_valid_cols of this core)]
load_gamma();                // writer:  cb_gamma_tiles   ← this core's gamma slice, row 0 of each tile valid (resident)

for (uint32_t block_idx = 0; block_idx < num_blocks_this_core; ++block_idx) {
    load_block(block_idx);           // reader (+compute tilize for RM): cb_x_tiles ← block_rows × core_w_tiles
    sumsq_block(block_idx);          // compute: cb_sumsq ← per row [P_full = Σ_{j<last} x_j², P_last = x_last²]
    rowsum_block(block_idx);         // compute: R1 → cb_inv_rms ← rsqrt(S/W + eps);  R2 → cb_rowsum_partial ← S
    scale_gamma_block(block_idx);    // compute (gamma): cb_x_gamma ← x ⊙ γ for the first xg_lookahead_chunks chunks
    combine_block(block_idx);        // R2 only — writer: partial → root; root compute: Σ_g S_g → rsqrt → root writer mcast → cb_inv_rms
    normalize_block(block_idx);      // compute: per chunk  cb_out_tiles ← (x⊙γ or x) ⊙ inv  (+ next x⊙γ chunk)
    store_block(block_idx);          // RM: compute untilize per chunk; writer: valid tiles / valid stick bytes → DRAM
}
```

| Block operation | Block shape it acts on | Resident across it | Intended fixed-cost frequency |
|---|---|---|---|
| `prepare_scalers` | 2 bf16 tiles | — | once per kernel |
| `load_gamma` | `core_w_tiles` tiles (row 0 valid) | `cb_gamma_tiles` until kernel end | once per kernel; RM gamma: one DRAM read of the slice + `2·slice_w_tiles` local face-row copies; TILE gamma: 2 row-0 reads per tile, one barrier |
| `load_block` | `block_rows × core_w_tiles` tiles, pushed per chunk (`block_rows × chunk_w_tiles` tiles) | x for the whole block | one read barrier per chunk; RM: one `tilize` call per block (`num_blocks = block_rows × chunks_per_row` tile-rows of width `chunk_w_tiles`), init once per block |
| `sumsq_block` | `block_rows × slice_w_tiles_this_core` (valid columns only) | x (not popped) | ≤ 2 chain calls per block (full tiles, last tile); one `mul` init each; DEST accumulates across the slice (PerRow) |
| `rowsum_block` | `block_rows × {1,2}` fp32 tiles | x | one `reduce` call per block (ReduceTile, partial scaler on the last tile of each row); post-op (×1/W, +eps, rsqrt) fused in R1 |
| `scale_gamma_block` | `block_rows × chunk_w_tiles` per chunk, first `xg_lookahead_chunks` chunks | x (popped per chunk after its γ pass), γ | one `mul` init per chunk call |
| `combine_block` (R2) | `block_rows` fp32 tiles per member | x·γ in `cb_x_gamma` | **one** combine round per block: `block_rows` unicasts + 1 semaphore inc per member, one root `reduce` call (AccumulateViaAdd + Skip, fused post-op), **one multicast** |
| `normalize_block` | `block_rows × chunk_w_tiles` per chunk | `cb_inv_rms` (waited once, popped once per block) | one `mul` init per chunk call (+ one γ `mul` call per chunk beyond the lookahead) |
| `store_block` | `block_rows × chunk_w_tiles` per chunk | — | one write barrier per chunk; RM: one `untilize` call per chunk (`num_blocks = block_rows`) |

Compute-thread order per block (this order **is** the stall-shadow decision S1): `tilize (RM)` → `sumsq` → `rowsum` →
[root: `scale_gamma(lookahead)` → `combine reduce`] / [others: `scale_gamma(lookahead)`] → wait `cb_inv_rms` → for each
chunk: `normalize(c)`, `scale_gamma(c + lookahead)` if any, `untilize(c)` (RM).

Work narrowing on ragged extents (nominal CB counts always — M1/M6):
* last W slice: `sumsq` / `rowsum` use `slice_w_tiles_this_core` (valid); `scale_gamma` / `normalize` / `tilize` /
  `untilize` run the nominal chunk width (results on invalid tiles are never stored); readers and writers transfer
  only valid tiles / valid stick bytes.
* last block of a core (`last_block_rows < block_rows`): compute runs all `block_rows` rows (stale rows produce
  never-stored results; every phase is row-independent); dataflow transfers only valid rows.
* RM partial last tile-row: missing sticks are not read; untouched rows are never written back.

### Stall-shadow analysis

| Stage that waits | Waits on | Independent work scheduled into the stall | Permission | Status |
|---|---|---|---|---|
| S1: compute, R2, before `normalize_block` | `cb_inv_rms` (peer partials → root → multicast) | `x ⊙ γ` for `xg_lookahead_chunks` chunks (= `chunks_per_row`, the whole block, whenever it fits — always for grid-filling R2 plans, whose slices are small) | multiplication commutes and associates: `x·inv·γ = (x·γ)·inv`; changes rounding order only (fp32 intermediate `cb_x_gamma`), gated on the golden PCC baseline | **built** (γ-first normalize is the only normalize form) |
| S1': root compute, R2, before `combine reduce` | `cb_rowsum_gather` (all members' partials) | its own `x ⊙ γ` lookahead (placed before the combine reduce) | as S1 | **built** |
| S2: compute, R2 with ≥ 2 blocks per core | as S1 | `sumsq_block(b+1)` + its partial | blocks are independent along the row axis | deferred → R6 |
| writer, before first `store_block` | first `cb_out_tiles` chunk | `load_gamma` (whole gamma stage runs here) | gamma does not depend on x | **built** |
| reader, while `cb_x_tiles` full (depth 1) | compute's per-chunk pops | nothing on the reader (its only other job, scalers, is done first) | — | lamp L1 (depth 2) |
| no-gamma R2 | `cb_inv_rms` | nothing: `x ⊙ inv` is the only remaining work and needs the wait | — | stated: no fill exists |

### Perf lamps

| Lamp | Why the default may be wrong here | Nearby alternative to measure |
|------|-----------------------------------|-------------------------------|
| L1 overlap | `x_depth_blocks = 1` and `block_rows` = whole assignment: the reader cannot start block b+1 until γ/normalize pops begin; with one block per core reads and writes serialize (BH DRAM is shared, so this may be free at the full grid) | `x_depth_blocks = 2` with `block_rows` halved; `block_rows = 1` |
| L2 grid synchronization | `MIN_W_TILES_PER_CORE = 8` decides how wide R2 groups get; at decode shapes the combine round may dominate per-core work of 8 tiles | `MIN_W_TILES_PER_CORE ∈ {4, 16}`; R1 forced (`gw = 1`, fewer cores) for `tensor_rows_tiles ≥ num_cores_available / 2` |
| L3 chunk floor | `CHUNK_FLOOR_TILES = 8`: below it the selector raises `w_group_size` (adds a combine) instead of chunking finer | floor 4 / 16; for wide R1 tensors compare chunk 16 vs. `gw = 2` |
| L4 SFPU scope of the post-reduce op | ×1/W, +eps, rsqrt run on all 32×32 lanes though only column 0 is meaningful (catalog `sfpu_tile_scope`) — once per tile-row | column-scoped `VectorMode` / iterations for the three SFPU calls |
| L5 combine payload | partial and inverse-RMS tiles move as whole 4096 B fp32 tiles; only column 0 (faces 0 and 2) carries data | send faces 0 and 2 only (2 × 1024 B) |
| L6 combine placement | the combine runs on the writer, behind `store_block(b−1)`'s tail when a core has ≥ 2 blocks | run the combine on the reader, or split it (send on reader, receive on writer) |
| L7 RM output pass | normalize packs tiles, then a separate `untilize` pass re-reads them | pack-untilize directly from the normalize DEST (removes `cb_out_tiles` for RM) |
| L8 FPU→FPU via L1 | `cb_x_gamma` round trip chosen per catalog `compute_fusion`; it costs a fp32 chunk of L1 | DEST-reuse form: `x⊙γ` then `DestReuseBinary` × a column-expanded inverse-RMS tile (`unary_bcast<Col>`), no `cb_x_gamma` |

## Dataflow Strategy

| Stage | Format | Mechanism | Notes |
|-------|--------|-----------|-------|
| x DRAM → L1 (TILE) | input dtype tiles | reader, `TensorAccessor` page reads of this core's valid `[row, col]` tiles into `cb_x_tiles` (row stride `row_stride_tiles`), one barrier per chunk | page id = `tile_row · tensor_w_tiles + col` |
| x DRAM → L1 (RM) | sticks | reader: for each (tile-row, chunk) 32 stick sub-reads of `valid_bytes` at byte offset `(slice_w_start_tile + c·chunk_w_tiles)·32·e_x`, L1 stride `chunk_w_tiles·32·e_x`, zero-fill tail (M9), push `chunk_w_tiles` pages into `cb_x_rm_sticks` | a raw loop, not `read_sticks_for_tilize` (§API Mapping, rejection R-a) |
| RM sticks → tiles | input dtype tiles | compute `tilize<chunk_w_tiles>` per block | `Fp32Mode::Fast` (downstream consumers are FPU) |
| gamma → `cb_gamma_tiles` | gamma dtype tiles, row 0 valid | writer, once: RM — read the slice of the single gamma stick into `cb_gamma_rm_stick`, then per valid tile copy 16 elems → face 0 row 0, 16 elems → face 1 row 0 (local self-NoC reads); TILE — 2 DRAM reads per tile (row 0 of faces 0 and 1) | rows 1–31 stay stale; consumed only through `BroadcastDim::Row` |
| constants | bf16 tiles | reader, once: `calculate_and_prepare_reduce_scaler<cb_reduce_scaler, SUM, REDUCE_ROW>(32)` then `(last_tile_valid_cols_this_core)` | 2 pages |
| compute phases | fp32 intermediates | §Block Operation Realization | `fp32_dest_acc_en=True` |
| R2 partial → root | fp32 tile per row | writer: `noc_async_write` of row r's partial to the root's `cb_rowsum_gather` base `+ (r·w_group_size + g)·4096`, write flush, `noc_semaphore_inc(root, sem_gather_arrivals, 1)`; the root writes its own partial to itself the same way (no inc) | all writes target the **root's** CB address = the local CB address (M1) |
| R2 root gather → compute | fp32 tiles | root writer: `cb_reserve_back(gather, gw·block_rows)`, `noc_semaphore_wait_min(sem_gather_arrivals, (gw−1)·(block_idx+1))` (cumulative, never reset), own write barrier, `cb_push_back` | |
| R2 inverse-RMS → group | fp32 tile per row | root writer: `ttnn.Mcast` fixed sender (member 0), `send(src = cb_inv_rms_root read ptr, dst = cb_inv_rms write ptr, block_rows·4096)` — loopback includes the root; receivers `receive()` then `cb_push_back(cb_inv_rms, block_rows)`; every member reserves `cb_inv_rms` **before** sending its partial | `McastConfig(noc = writer NoC, handshake = False)`: readiness is implied — a member's partial for block b is produced only after its compute popped `cb_inv_rms` of block b−1, and the root multicasts only after all partials arrived |
| tiles → DRAM (TILE) | output dtype tiles | writer: valid tiles of each chunk, one barrier per chunk | |
| tiles → sticks → DRAM (RM) | sticks | compute `untilize<chunk_w_tiles>(block_rows)` per chunk → `cb_out_rm_sticks`; writer: per tile-row, valid sticks × valid bytes at byte offset, L1 stride `chunk_w_tiles·32·e_out` | raw loop, not `write_sticks_after_untilize` (rejection R-b) |

Inter-Tensix contract for R2 (built): semaphores — `sem_gather_arrivals` (one per core, used on roots) plus the
`Mcast` channel's own semaphores (`mcast.attach(descriptor, "inv_mcast", [writer], first_semaphore_id)`); groups =
consecutive runs of `w_group_size` cores in row-major order over the used cores (`ttnn.Mcast(device, cfg, used_cores,
w_group_size, ttnn.McastFixedSenderConfig(sender_index=0), RowMajor)` — runs that wrap a grid row become ≤ 3
rectangles, handled by the helper). Peer (root) coordinates passed as RT args through
`device.worker_core_from_logical_core`.

Placement axis (TARGET `memory_layout`) as the physical realization of the logical shards:

| `memory_layout` value | Cut axis (this op's character) | Maps onto | Cost class | Physical realization |
|---|---|---|---|---|
| `INTERLEAVED` | — | R1 / R2 by `_select_blocking` | built (Phase 0) | DRAM pages via `TensorAccessor` |
| `HEIGHT_SHARDED` | row axis (independent) | R1, `num_row_groups` = shard grid, `block_rows` = shard tile-rows (one block = whole shard) | knob-turn (placement) | `cb_x_tiles` / `cb_out_tiles` backed on the shard via `ttnn.cb_descriptor_from_sharded_tensor` (zero-copy, no NoC re-read); RM shard → tilize from the shard CB |
| `WIDTH_SHARDED` | W axis (dependent) | R2 with `num_row_groups = 1`, `w_group_size` = shard count, `core_w_tiles` = shard width | knob-turn — the combine it needs is built in Phase 0 | as above; Mcast groups = the shard grid's rows-of-cores order |
| `BLOCK_SHARDED` | both | R2, row groups = shard-grid rows, W-groups = shard-grid columns | knob-turn (combine built) | as above; ragged shards follow the nominal-quanta rule (M1/M6) |

## Work Distribution

| Field | Value |
|-------|-------|
| Work unit | a block: `block_rows` tile-rows × one W slice of `core_w_tiles` tiles |
| Grid | `used_cores = num_row_groups · w_group_size` cores, the first ones of `device.compute_with_storage_grid_size()` in **row-major** order (`ttnn.num_cores_to_corerangeset(used, grid, row_wise=True)`); core `i` → row group `i // w_group_size`, member `g = i % w_group_size`; member 0 = root |
| Per-core work | rows: `split_work_to_cores` semantics over `num_row_groups` groups (two classes `rows_hi = ceil(R/gr)`, `rows_lo`); `num_blocks_this_core = ceil(core_rows_tiles / block_rows)`, `last_block_rows` = remainder; W: `slice_w_start_tile = g·core_w_tiles`, `slice_w_tiles_this_core = min(core_w_tiles, tensor_w_tiles − slice_w_start_tile)`; `last_tile_valid_cols_this_core = last_tile_valid_cols` if the core owns the last tensor tile-column else 32 |
| Remainder | rows: two-class split (no core empty since `gr ≤ R`); W: ragged last slice (M2 guarantees ≥ 1 valid tile); H padding: `ceil` per image (TILE) / `ceil` over sticks (RM) |
| Phase 0 multi-core | yes: R1 spreads the independent row axis across up to all cores; R2 spreads both axes |

Regime-pinned tests (required; shapes pin the regime for any grid with ≤ 256 cores and L1 ≤ 1.5 MiB/core):

| Pinned regime / path | Shape (dtype, layout, gamma) | Why it is pinned |
|---|---|---|
| R1, `chunks_per_row = 1`, multi-block | `(1,1,8192,64)` bf16 TILE + RM gamma | `tensor_rows_tiles = 256 ≥ C` ⇒ `gw_fill = 1`; tiny slice fits |
| R1, `chunks_per_row > 1` | `(1,1,8192,4096)` bf16 TILE + RM gamma | `R = 256 ≥ C`; whole-slice chunk needs ≥ 1.5 MiB (x 256 KiB + gamma 256 KiB + x·γ 512 KiB + out 512 KiB) ⇒ chunking |
| R2 fill (`R = 1`) | `(1,1,32,16384)` bf16 TILE and RM, with/without gamma, non-aligned `(1,1,32,16300)` | `R = 1 < C`, `Wt ≥ 16` ⇒ `gw ≥ 2` on any grid |
| R2 L1-forced, multi-block | `(1,1,4096,8192)` fp32 TILE + fp32 RM gamma | `R = 128 ≥ C`, slice 1 MiB x + 1 MiB gamma > L1 ⇒ `gw ≥ 2`, `num_blocks_this_core ≥ 2` |

## Circular Buffers

`T_in`, `T_out` = tile bytes of the input/output dtype; `T_g` = tile bytes of the gamma dtype; `F32` = 4096 B;
`b` = `block_rows`, `n` = `core_w_tiles`, `c` = `chunk_w_tiles`, `s` = `row_stride_tiles`, `gw` = `w_group_size`,
`L` = `xg_lookahead_chunks`. Every CB is declared once over all `used_cores` (M1). Full ledger: `l1_ledger.md`.

| Semantic Name | Index | Page Size | Num Pages | Sizing rationale | Format | Producer | Consumer | Lifetime |
|---------------|-------|-----------|-----------|------------------|--------|----------|----------|-----------|
| `cb_x_rm_sticks` (RM only) | 0 | `T_in` (32 sticks × 32 elems) | `2·c` | streams (row, chunk) slabs; spans a W chunk | input dtype | reader | compute (tilize) | per chunk |
| `cb_x_tiles` | 1 | `T_in` | `x_depth_blocks · b · s` | spans the whole block (resident) | input dtype | reader (TILE) / compute (RM, tilize) | compute | per block, popped per chunk |
| `cb_gamma_rm_stick` (RM gamma only) | 2 | `round_up(n·32·e_g, ttnn.get_dram_alignment())` | 1 | landing pad for the gamma slice | gamma dtype (raw bytes) | writer | writer | kernel start only |
| `cb_gamma_tiles` (gamma only) | 3 | `T_g` | `s` | resident gamma slice (row 0 valid) | gamma dtype | writer | compute | whole kernel |
| `cb_reduce_scaler` | 4 | 2048 | 2 | [full, partial] scaler pair | bf16 | reader | compute | whole kernel |
| `cb_sumsq` | 5 | `F32` | `2·b` | `[P_full, P_last]` per row (strided pack, M8) | fp32 | compute | compute | per block |
| `cb_rowsum_partial` (R2) | 6 | `F32` | `b` | per-row partial S (column 0) | fp32 | compute | writer | per block |
| `cb_rowsum_gather` (R2) | 7 | `F32` | `gw·b` | `[row][member]` partials; symmetric address (M1) | fp32 | writer (root: remote writes land here; root pushes) | compute (root) | per block |
| `cb_inv_rms_root` (R2) | 8 | `F32` | `b` | root's finalized inverse-RMS, multicast source | fp32 | compute (root) | writer (root) | per block |
| `cb_inv_rms` | 9 | `F32` | `b` | per-row inverse-RMS (column 0); multicast destination in R2 | fp32 | compute (R1) / writer (R2) | compute | per block |
| `cb_x_gamma` (gamma only) | 10 | `F32` | `L·b·c` | x⊙γ lookahead window | fp32 | compute | compute | per block |
| `cb_out_tiles` | 11 | `T_out` | TILE: `2·b·c`; RM: `b·c` | streams chunks | output dtype | compute | writer (TILE) / compute (RM, untilize) | per chunk |
| `cb_out_rm_sticks` (RM only) | 12 | `T_out` | `2·b·c` | streams untilized chunk slabs | output dtype | compute | writer | per chunk |

Producer/consumer notes: a CB whose producer depends on layout or regime has exactly one producer *per program*
(layout and regime are fixed per dispatch). `cb_gamma_rm_stick` is a writer-private scratch region (reserve, use,
push/pop by the writer itself). No CB is transformed in place.

## Block Operation Realization

| # | Block operation | Block shape | Helper? | Input CB (semantic name, pages, state) | Output CB (semantic name, pages) | CB state after |
|---|-----------------|-------------|---------|----------------------------------------|----------------------------------|----------------|
| 0a | `prepare_scalers` | 2 tiles | helper (dataflow) | — | `cb_reduce_scaler`, 2 | resident, popped by compute at kernel end |
| 0b | `load_gamma` | `n` tiles | raw dataflow (R-c) | DRAM gamma; `cb_gamma_rm_stick` (RM) | `cb_gamma_tiles`, `s` (valid: `slice_w_tiles_this_core`) | resident, popped by compute at kernel end |
| 1 | `load_block` (TILE) | `b × n` | raw dataflow | DRAM x | `cb_x_tiles`, `b·c` per chunk push | full block resident |
| 1' | `load_block` (RM) | `b × n` | raw dataflow (R-a) + helper `tilize` | `cb_x_rm_sticks`, `c` per (row, chunk), consumed | `cb_x_tiles`, `c` per tilize tile-row | full block resident |
| 2 | `sumsq_block` | `b × slice_w_tiles_this_core` | helper (`eltwise_chain`) | `cb_x_tiles`, `b·s`, caller-waited, **not popped** | `cb_sumsq`, `2·b` (or `b` when `slice_w_tiles_this_core == 1`), caller reserve/push | x still resident |
| 3 | `rowsum_block` | `b × {2 or 1}` | helper (`reduce`) | `cb_sumsq`, consumed | R1: `cb_inv_rms`, `b`; R2: `cb_rowsum_partial`, `b` | — |
| 4 | `scale_gamma_block` (gamma) | `b × c` per chunk, chunks `[0, L)` | helper (`mul`, Row bcast) | `cb_x_tiles` (chunk popped after), `cb_gamma_tiles` (no pop) | `cb_x_gamma`, `b·c` per chunk | x chunk freed for the reader |
| 5 | `combine_block` (R2) | `b` tiles per member; root reduces `b × gw` | raw dataflow (gather) + helper `Mcast`; root compute helper `reduce` | root: `cb_rowsum_gather`, `gw·b`, consumed | root: `cb_inv_rms_root`, `b` → multicast → every member's `cb_inv_rms`, `b` | `cb_inv_rms` holds the block's inverse-RMS |
| 6 | `normalize_block` | `b × c` per chunk | helper (`mul`, Col bcast) | `cb_x_gamma` (or `cb_x_tiles` without gamma; chunk popped), `cb_inv_rms` (caller-waited once, popped after last chunk) | `cb_out_tiles`, `b·c` per chunk | — |
| 7 | `store_block` | `b × c` per chunk | TILE: raw dataflow; RM: helper `untilize` + raw dataflow (R-b) | `cb_out_tiles` (→ `cb_out_rm_sticks` for RM) | DRAM output | — |

## API Mapping

| Block operation | Type | Function | File:Line | Template Params / Args | Input CB | Output CB | Which params are block knobs |
|-----------------|------|----------|-----------|------------------------|----------|-----------|------------------------------|
| kernel boot | raw_api | `compute_kernel_hw_startup(icb0, icb1, ocb)` | required by every helper: `reduce_helpers_compute.hpp:31-35`, `chain.hpp:34-40`, `tilize_helpers.hpp:89-93` | first statement of `kernel_main`, once; first stage's CBs (RM: `cb_x_rm_sticks, cb_x_rm_sticks, cb_x_tiles`; TILE: `cb_x_tiles, cb_x_tiles, cb_sumsq`) | — | — | — |
| `prepare_scalers` | helper | `dataflow_kernel_lib::calculate_and_prepare_reduce_scaler<cb_reduce_scaler, PoolType::SUM, ReduceDim::REDUCE_ROW>(valid)` ×2 | `reduce_helpers_dataflow.hpp:97-99` | `valid = 32`, then `valid = last_tile_valid_cols_this_core` (RT); pool-type-aware overload | — | `cb_reduce_scaler` | — |
| `load_block` (RM) compute part | helper | `compute_kernel_lib::tilize<chunk_w_tiles, cb_x_rm_sticks, cb_x_tiles>(b · chunks_per_row)` | `tilize_helpers.hpp:187-197` | symmetric pages; `Fp32Mode::Fast`; default init/uninit (one call per block) | `cb_x_rm_sticks` | `cb_x_tiles` | `chunk_w_tiles` (CT), `b·chunks_per_row` (RT) |
| `sumsq_block` | helper | `compute_kernel_lib::eltwise_chain(IterationShape::grid(b, cols), BinaryFpu<BinaryFpuOp::Mul, input(cb_x_tiles, WaitPolicy::None, PopPolicy::None, InputTileMapping::Block, …, TileAddressing::Strided), input(cb_x_tiles, … same …), Dst::D0, DestAccumulation::PerRow>{StridedTileRange{base, s}, StridedTileRange{base, s}}, PackTile<output(cb_sumsq, ReservePolicy::None, PushPolicy::None, DataFormatReconfig::Enabled, TileAddressing::Strided, DestAccumulation::PerRow)>{StridedTileRange{pack_base, cols_per_row}})` — call A: `cols = slice_w_tiles_this_core − 1`, `base = 0`, `pack_base = 0`; call B: `cols = 1`, `base = slice_w_tiles_this_core − 1`, `pack_base = 1`; `cols_per_row = 2` (1 when the slice has one tile: call A skipped) | `chain.hpp:121-148, 208-217, 232-237, 261-266, 283-287, 485-494, 503-504, 532-533`; same-CB operands: `chain.inl:1168-1169`; Mul with DEST accumulation: `chain.inl:1176-1184`; PerRow + (None,None) pack legal: `chain.inl:969-976` | caller: `cb_wait_front(cb_x_tiles, b·s)` once per block, `cb_reserve_back(cb_sumsq, b·cols_per_row)` / `cb_push_back` | `cb_x_tiles` | `cb_sumsq` | `b`, slice width (RT) |
| `rowsum_block` | helper | `compute_kernel_lib::reduce<PoolType::SUM, ReduceDim::REDUCE_ROW, cb_sumsq, cb_reduce_scaler, OUT, ReduceInputPolicy::BulkWaitBulkPop, ReduceDataFormatReconfigMode::INPUT_AND_OUTPUT, ReduceFp32Mode::Fast, ReduceAlgorithm::ReduceTile>(ReduceInputBlockShape::of(b, cols_per_row), ReduceInputMemoryLayout::contiguous(), NoAccumulation{}, post_op, ReducePartialScaler::with_partial())`; `OUT = cb_inv_rms` (R1) / `cb_rowsum_partial` (R2); R1 `post_op = [](uint32_t d){ binop_with_scalar_tile_init(); mul_unary_tile(d, inv_w_bits); add_unary_tile(d, eps_bits); rsqrt_tile_init(); rsqrt_tile(d); }`, R2 `post_op = NoOp{}` | `reduce_helpers_compute.hpp:620-641, 294-305, 344-356, 116, 160`; SFPU: `binop_with_scalar.h:156, 60, 35`; `rsqrt.h:21, 44` | `inv_w_bits` = fp32 bits of `1/W` (logical W), `eps_bits` = fp32 bits of epsilon (RT) | `cb_sumsq`, `cb_reduce_scaler` | `cb_inv_rms` / `cb_rowsum_partial` | `b` |
| `scale_gamma_block` | helper | `compute_kernel_lib::mul<input(cb_x_tiles, WaitPolicy::None, PopPolicy::None, InputTileMapping::Block, DataFormatReconfig::Enabled, TileAddressing::Offset), input(cb_gamma_tiles, BroadcastDim::Row, WaitPolicy::None, PopPolicy::None, InputTileMapping::Row, DataFormatReconfig::Enabled, TileAddressing::Offset), output(cb_x_gamma)>(IterationShape::grid(b, c))` (or the equivalent `eltwise_chain(BinaryFpu<Mul, …>{x_base, γ_base}, PackTile…)` to pass the chunk offsets) | `convenience.hpp:50-51`, `chain.hpp:356-377, 383-391, 305-310`, `TileAddressing::Offset` `chain.hpp:253-256` | x base = 0 (every earlier chunk of the block was already popped, so the next chunk is the FIFO front — contiguous by M5), γ base = `c_idx·c`; caller `cb_wait_front(cb_gamma_tiles, s)` once per kernel, `cb_pop_front(cb_x_tiles, b·c)` after each chunk | `cb_x_tiles`, `cb_gamma_tiles` | `cb_x_gamma` | `b`, `c` |
| `combine_block` root compute | helper | `compute_kernel_lib::reduce<PoolType::SUM, ReduceDim::REDUCE_ROW, cb_rowsum_gather, cb_reduce_scaler, cb_inv_rms_root, ReduceInputPolicy::BulkWaitBulkPop, ReduceDataFormatReconfigMode::INPUT_AND_OUTPUT, ReduceFp32Mode::Fast, ReduceAlgorithm::AccumulateViaAdd, ReduceWithinTile::Skip>(ReduceInputBlockShape::of(b, gw), contiguous(), NoAccumulation{}, post_op)` with the R1 `post_op` | `reduce_helpers_compute.hpp:133-159, 165-195, 197-209` (the documented cross-core combine), `620-641` | Skip ⇒ post-op must run its own SFPU inits (`reduce_helpers_compute.hpp:185-187`) | `cb_rowsum_gather` | `cb_inv_rms_root` | `b`, `gw` |
| `combine_block` multicast | helper | host `ttnn.Mcast(device, ttnn.McastConfig(noc=<writer NoC>, handshake=False), used_cores, w_group_size, ttnn.McastFixedSenderConfig(sender_index=0))` + `attach(descriptor, "inv_mcast", [writer], first_sem_id)`; device `McastArgs<get_named_compile_time_arg_val("inv_mcast_ct_offset"), get_named_compile_time_arg_val("inv_mcast_rt_offset")>()`, root `sender(noc).send(src, dst, b·4096)`, members `receiver(noc).receive()` | host `mcast.hpp:86-96, 98-102, 123-131, 139-143`; device `mcast_args.hpp:38-56`; loopback to the in-group sender `mcast_pipe.inl:139-146` | R1: no channel (`ttnn.attach_absent_mcast(writer, "inv_mcast")`, `mcast.hpp:69`) | `cb_inv_rms_root` | `cb_inv_rms` | `b` |
| `combine_block` gather | raw_api | `noc_async_write`, `noc_semaphore_inc`, `noc_semaphore_wait_min`, `get_semaphore` | `dataflow_api.h:2397, 2141, 1689` | sem `sem_gather_arrivals` (`ttnn.SemaphoreDescriptor`, initial 0, all used cores) | `cb_rowsum_partial` | root `cb_rowsum_gather` | `b`, `gw` |
| `normalize_block` | helper | `compute_kernel_lib::mul<input(A, WaitPolicy::PerTile, PopPolicy::PerTile, InputTileMapping::Block), input(cb_inv_rms, BroadcastDim::Col, WaitPolicy::None, PopPolicy::None, InputTileMapping::Col), output(cb_out_tiles, ReservePolicy::PerTile, PushPolicy::PerTile)>(IterationShape::grid(b, c))`; `A = cb_x_gamma` (gamma) or `cb_x_tiles` with `WaitPolicy::None/PopPolicy::None` + `TileAddressing::Offset` and a caller pop per chunk (no gamma) | `convenience.hpp:50-51`, `chain.hpp:222-224` (Col mapping), `305-310` (Col bcast) | caller `cb_wait_front(cb_inv_rms, b)` before chunk 0, `cb_pop_front(cb_inv_rms, b)` after the last chunk | `cb_x_gamma` / `cb_x_tiles`, `cb_inv_rms` | `cb_out_tiles` | `b`, `c` |
| `store_block` (RM) compute part | helper | `compute_kernel_lib::untilize<chunk_w_tiles, cb_out_tiles, cb_out_rm_sticks>(b)` | `untilize_helpers.hpp:145-154` | default init/uninit per call | `cb_out_tiles` | `cb_out_rm_sticks` | `chunk_w_tiles` (CT), `b` |
| RM tail zero-fill | helper | `dataflow_kernel_lib::fill_l1_range<elem>(addr, bytes, 0)` | `l1_helpers.hpp:89-90` | per stick of the last valid tile, `bytes = round_up(valid, 32·e) − valid` | — | `cb_x_rm_sticks` | — |
| RM gamma face-row scatter | helper | `dataflow_kernel_lib::set_read_state<16·e_g>(noc, src)` + `read_with_state<16·e_g>(noc, dst, src)` | `local_copy_helpers_dataflow.hpp:74-75, 94-95` | 2 per valid gamma tile, one barrier | `cb_gamma_rm_stick` | `cb_gamma_tiles` | — |
| tile / stick DRAM transfers | raw_api | `TensorAccessor` + `noc_async_read` / `noc_async_write` (+ page/byte offsets) | `tech_reports/tensor_accessor/tensor_accessor.md` | | | | chunk = transfer batch |

Helpers considered and rejected (every `raw_api` / hand-written dataflow entry above):

| Id | Where | Candidate | Mismatch (file:line) | Concrete reason |
|---|---|---|---|---|
| R-a | RM `load_block` reader | `dataflow_kernel_lib::read_sticks_for_tilize` | `tilize_helpers_dataflow.inl:92-93` (L1 stride `padded_row_bytes` and push count `width_in_tiles` derived from the valid `row_bytes`), `:127` (pushes `width_in_tiles`), `:122-123` (padded bytes never written) | the ragged last chunk / last W slice must keep the **nominal** stride `chunk_w_tiles·32·e` and the nominal push of `chunk_w_tiles` pages (M4, M6) while reading only the valid bytes, and needs the tail zero-fill (M9); the helper derives stride and push count from the valid width. One raw loop (modelled on the helper body) serves every chunk so the CB contract is uniform |
| R-b | RM `store_block` writer | `dataflow_kernel_lib::write_sticks_after_untilize` | `tilize_helpers_dataflow.inl:208-209` (stride / pop count from `row_bytes`), `:239` (pops `width_in_tiles`) | same nominal-stride / nominal-pop requirement on the ragged chunk |
| R-c | `load_gamma` | `read_sticks_for_tilize` + `tilize` of a 32-stick slab | `tilize_helpers.hpp:121-125` (tilize consumes 32 sticks per tile-row) | gamma is one stick; tilizing it needs a 32-row slab (`n` tiles of extra L1) and a compute pass for 1/32 useful data; the face-row scatter needs 2 small local copies per tile and no compute |
| R-d | partial gather | `ttnn.Mcast` / `McastArgs` | `mcast.hpp:119-120` ("partitions the ordered receiver cores into equal consecutive groups" — one sender fans out to its group) | the gather is many-to-one; the helper is one-to-many. Multicast is used for the return leg |
| R-e | TILE x / output transfers | (no tile-streaming dataflow helper exists in `ttnn/cpp/ttnn/kernel_lib/`) | — | plain `TensorAccessor` page reads/writes batched per chunk |

## Broadcast Verification

| Phase | Op | CB_A (semantic name) Valid Region | CB_B (semantic name) Valid Region | Broadcast Dim |
|-------|-----|-----------------------------------|-----------------------------------|---------------|
| `sumsq_block` | Mul (x·x, DEST accumulate PerRow) | `cb_x_tiles`: All (padding columns of the last tile excluded later by the partial scaler on `P_last`) | `cb_x_tiles` (same tile): All | None |
| `scale_gamma_block` | Mul | `cb_x_tiles`: All | `cb_gamma_tiles`: Row0 (1-D `[W]` operand) | Row (`BroadcastDim::Row`, `InputTileMapping::Row`) |
| `normalize_block` | Mul | `cb_x_gamma` / `cb_x_tiles`: All | `cb_inv_rms`: Col0 (REDUCE_ROW output) | Col (`BroadcastDim::Col`, `InputTileMapping::Col`) |
| `rowsum_block` / root combine | reduce ROW | `cb_sumsq`: All / `cb_rowsum_gather`: Col0 | `cb_reduce_scaler` (row-0 scaler layout) | — (Skip on the combine: inputs already Col0) |

## Key Risks and Gotchas

| Risk | Why it bites here | Mitigation in this design |
|------|-------------------|---------------------------|
| Padding leaks into the RMS | golden poisons TILE padding with 1000 (`helpers.py`, `poison_padding`); RM tail bytes are stale L1 | `P_last` kept separate and masked by the partial scaler (M8); RM tails zero-filled (M9); `1/W` uses the **logical** W; sum-of-squares never walks tiles beyond `slice_w_tiles_this_core` |
| Remote writes hit the wrong address | partials and inverse-RMS are written into another core's CB at the local CB address | every CB declared once over all used cores at nominal size (M1); no per-class CB descriptors |
| Ring-wrap overshoot on ragged slices/blocks | last slice has fewer valid tiles, last block fewer rows | nominal push/pop counts everywhere (M6); work narrowed in transfers and in the sum-of-squares only |
| Multicast without handshake overwrites a live inverse-RMS | `McastConfig(handshake=False)` | protocol invariant: a member reserves `cb_inv_rms` and its compute has popped block b−1's inverse-RMS before its block-b partial exists; the root multicasts only after all block-b partials arrived |
| Semaphore races | gather counter reused across blocks | cumulative `noc_semaphore_wait_min(sem, (gw−1)·(block_idx+1))`, never reset |
| Gamma row broadcast reads stale rows | `cb_gamma_tiles` rows 1–31 are stale/poisoned | gamma consumed only via `BroadcastDim::Row` (srcB row 0); TILE gamma reads land row 0 of faces 0/1 at in-page offsets 0 and `256·e_g` |
| fp32 + `fp32_dest_acc_en=False` | precision convention | EXCLUSION (NotImplementedError); bf16 + False outside Phase 0 SUPPORTED; config read from the resolved descriptor, never overridden |
| Reduce scaler format | the reduce LLK expects the scaler in bf16 row-0 layout | `cb_reduce_scaler` bf16, filled by the pool-type-aware `calculate_and_prepare_reduce_scaler<…, SUM, REDUCE_ROW>`; `1/W` applied in the fp32 post-op, never baked into the bf16 scaler |
| Regime only on some grids | R1/R2 choice depends on `num_cores_available` and L1 size | selection function pinned above; regime-pinned tests use shapes that select the same regime on any grid ≤ 256 cores and L1 ≤ 1.5 MiB |
| x popped while still needed | x is read by sum-of-squares, γ-scale and (no gamma) normalize | sum-of-squares/rowsum use `None/None` (no pop); x popped per chunk only after its last consumer; M5 keeps each pop contiguous |
| The `memory_config` kwarg | golden passes it for sharded cells | accepted; Phase 0 refuses sharded placements in `validate()` (NotImplementedError) |

### Structural impossibilities

None beyond `feature_spec.py` INVALID. (No index axes: the op has no `dim` parameter.)
