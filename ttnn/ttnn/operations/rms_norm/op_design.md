# Operation Design: rms_norm

## Overview

| Field | Value |
|-------|-------|
| Classification | fused (row reduction + broadcast apply, optional cross-Tensix-core combine) |
| Goal | One native `ttnn.generic_op` dispatch computing RMSNorm along the last dim, for `Layout::TILE` and `Layout::ROW_MAJOR` inputs of any rank ≥ 2, tile-aligned or not, filling the Tensix grid from Phase 0 — by row-groups, and by splitting W across up to 8 Tensix cores (partials combined over the NoC) when there are too few tile-rows or W is too wide to keep resident. |
| Math | `out[..., h, w] = x[..., h, w] * rsqrt( (1/W) * Σ_{w'<W} x[..., h, w']² + eps ) * gamma[w]` (gamma term omitted when `gamma is None`). W = logical last dim; padding never enters Σ or the 1/W. |
| Mode | Hybrid — compute = kernel-lib helpers (`eltwise_chain`, `reduce`, `tilize`, `untilize`); dataflow = TensorAccessor reads/writes + one raw NoC unicast all-gather for the W-split combine. Designed from first principles + the measured catalog (`ttnn/ttnn/operations/examples/master.md`). |
| References | `.claude/references/blocking-model.md`, `.claude/references/l1-footprint-discipline.md`, `.claude/references/precision_convention.md`, `ttnn/cpp/ttnn/kernel_lib/{eltwise/api/chain.hpp, reduce_helpers_compute.hpp, tilize_helpers.hpp, untilize_helpers.hpp, tilize_helpers_dataflow.hpp, l1_helpers.hpp}`, catalog entries `width_split`, `tensix_all_reduce`, `row_reduce_accumulate`, `eltwise_l1_vs_dest_accumulate`, `compute_fusion`, `compute_block_size`, `double_buffer`, `shared_input_reuse`, `resident_shard_locality`, `sfpu_tile_scope`; golden suite `eval/golden_tests/rms_norm/` |

### Catalog entries → knobs they informed

| Catalog entry | Measured fact used | Knob / decision it set |
|---|---|---|
| `width_split` | Reduce+combine W-split pays when R_t ≤ ~1/8 grid, k ≤ 8, ≥ 4 tiles per slice; k ≥ 16 combine grows to ~4 µs; RM pieces must stay ≥ 512 B | occupancy term of `w_split_cores` (k) selection, `K_CAP = 8`, RM piece floor |
| `tensix_all_reduce` (per-row vector) | K ≤ 8: symmetric unicast round of 1-tile column-0 partials; multicast return only pays from K ≥ 12 | combine topology = symmetric unicast all-gather; k > 8 owner+mcast is a `deferred` row |
| `eltwise_l1_vs_dest_accumulate` | DEST-resident accumulation 2.01× over L1 accumulation at 64 steps (BH), pays from ≥ 4 steps | pass 1 squares and accumulates in DEST (no x² buffer) |
| `row_reduce_accumulate` | pairwise `add_tiles` into DEST + one finalize beats `reduce_tile` from 8 tiles (BH) | `ReduceAlgorithm::AccumulateViaAdd` for collapse / combine |
| `compute_fusion` | DEST→SrcA reuse costs ~40 ns/tile vs a second pass | DestReuse gamma multiply chosen for L1 (op is DRAM-bound), recorded as perf lamp |
| `compute_block_size` | ~1.4 µs per extra helper pass (BH) | block = whole per-core assignment when it fits |
| `double_buffer` | 2-deep CBs + several reads in flight per barrier still add ~7% at full BH grid | `x_block_depth = 2` when > 1 block; row-granular read barriers |
| `shared_input_reuse` (2) | resident window broadcast is null for r ≤ 0.125 | gamma multicast is `deferred` (r ≤ 1/32) |
| `resident_shard_locality` | balanced ≤ 32-tile column chunks; never "widest divisor ≤ 32" | `rm_chunk_w_tiles` balanced chunking; sharded regimes consume the shard in place |
| `sfpu_tile_scope` | rsqrt on col-0-only result wastes 3.8× SFPU work | perf lamp on the finalize post-op |

## Parameters

| Name | Type | Required | Valid Range | Default | CT/RT |
|------|------|----------|-------------|---------|-------|
| input_tensor | ttnn.Tensor | yes | rank ≥ 2; dtype ∈ SUPPORTED; on device | — | — |
| gamma | Optional[ttnn.Tensor] (keyword-only) | no | shape (1,1,1,W) (any leading 1s), last dim == input W; `Layout::ROW_MAJOR` or `Layout::TILE` | None | `has_gamma` CT; buffer address RT |
| epsilon | float (keyword-only) | no | ≥ 0, finite | 1e-6 | RT (fp32 bits) to compute |
| compute_kernel_config | ttnn.ComputeConfigDescriptor (keyword-only) | no | `fp32_dest_acc_en` gated (Phase 0: True); `math_fidelity` / `math_approx_mode` any value, honored | `default_compute_kernel_config()` → HiFi4, fp32_dest_acc_en=True, math_approx_mode=False | passed as-is as `config=` of the compute KernelDescriptor |
| memory_config | Optional[ttnn.MemoryConfig] (keyword-only) | no | output placement; Phase 0: must be `TensorMemoryLayout::INTERLEAVED` | input's memory_config | host only |

Entry-point contract:

| Item | Decision |
|------|----------|
| `default_compute_kernel_config()` | exported from `ttnn/operations/rms_norm/__init__.py` as a factory (fresh descriptor per call); `None` resolves through it, nowhere else. `eval/golden_tests/rms_norm/axes.py` imports it. |
| Signature | `rms_norm(input_tensor, *, gamma=None, epsilon=1e-6, compute_kernel_config=None, memory_config=None)` — `memory_config` must be accepted because the golden runner passes it for sharded cells (they must reach `validate()` and raise `NotImplementedError`, not `TypeError`). |
| Validation errors (ValueError) | input rank < 2; gamma last dim ≠ input last dim; gamma volume ≠ W. |
| Support gate (NotImplementedError) | axis outside SUPPORTED; EXCLUSION `{dtype: float32, fp32_dest_acc_en: False}` (precision convention, natively rejected). |
| Capacity error (RuntimeError) | W so wide that even `w_split_cores = K_CAP` does not fit L1 (see Mechanism caps) — not reachable by any golden shape on WH/BH. |
| INPUT_TAGGERS | `tag_alignment` → `"tile_aligned"` / `"w_non_aligned"` / `"h_non_aligned"` (W checked first); `tag_rank` → `len(shape)`; `tag_gamma_dtype` / `tag_gamma_layout` → real value, or `"none"` when gamma is absent. |
| Recommended Phase 0 SUPPORTED (implementer owns it) | dtype {float32, bfloat16}; fp32_dest_acc_en {True}; layout {TILE, ROW_MAJOR}; alignment {all three}; rank {2,3,4}; gamma_mode {gamma, no_gamma}; gamma_dtype {float32, bfloat16, "none"}; gamma_layout {TILE, ROW_MAJOR, "none"}; memory_layout {INTERLEAVED}. Refinement candidates: bfloat8_b input, bfloat8_b gamma, `bfloat16 + fp32_dest_acc_en=False`, HEIGHT/WIDTH/BLOCK_SHARDED. |
| Index axes | none (the reduced dim is fixed at -1). |
| Single dispatch | host: validate → derive params → `select_blocking()` → allocate output → build one ProgramDescriptor → one `ttnn.generic_op([input, (gamma), output], desc)`. No ttnn op is called on the tensors. |

## Tensors

### Input

| Property | Requirement |
|----------|-------------|
| Shape | rank ≥ 2, `(..., H, W)`; leading dims fold into the row axis |
| Dtype | float32 or bfloat16 (Phase 0) |
| Layout | `Layout::TILE` or `Layout::ROW_MAJOR`, natively (no host transform) |
| Memory | `TensorMemoryLayout::INTERLEAVED`, DRAM or L1 `BufferType` (Phase 0) |
| Gamma | logical shape `(1,1,1,W)`; dtype float32/bfloat16; `Layout::ROW_MAJOR` (one stick of W elements) or `Layout::TILE` (row 0 of tile-row 0 valid) |

### Output

| Property | Value |
|----------|-------|
| Shape | identical logical shape to input |
| Dtype | input dtype |
| Layout | input `Layout` |
| Memory | `memory_config` if given, else the input's (Phase 0: INTERLEAVED) |

## Blocking Model

The op's data has two axes after folding: the **row axis** (every leading dim folded with tile-rows of H) and the **W axis** (the reduced dim). Leading dims fold honestly: the op treats every row identically and the tile-rows of consecutive images are consecutive in the tile index space (`TILE`) / stick index space (`ROW_MAJOR`).

Geometry, alignment-aware and per image (single source: host `derive_geometry()`):

| Symbol | Formula | Unit |
|---|---|---|
| `W` | `shape[-1]` | elements |
| `H` | `shape[-2]` | elements |
| `num_images` | `prod(shape[:-2])` (1 for rank 2) | images |
| `image_row_tiles` | `ceil(H / 32)` | tile-rows per image |
| `tensor_row_tiles` | `num_images * image_row_tiles` (never `floor(num_images*H/32)`) | tile-rows |
| `tensor_w_tiles` | `ceil(W / 32)` | tiles |
| `w_valid_in_last_tile` | `W - 32*(tensor_w_tiles-1)` ∈ [1, 32] | elements |

### Axes

| Axis | Character (+ one-clause reason) | Extent knob | Phase 0 value | Knob source | Core-assignment | Later unlock |
|------|--------------------------------|-------------|---------------|-------------|-----------------|--------------|
| row (tile-rows, incl. folded leading dims) | **independent** — each output row depends only on its own input row | `block_row_tiles` | coarsest that fits: the whole per-group assignment `core_row_tiles` when it fits at `x_block_depth = 1`, else the largest value fitting at `x_block_depth = 2`, ≥ 1 | host `select_blocking()` → RT arg `block_row_tiles` (+ `last_block_row_tiles`, `num_blocks_this_core`) | spread over **row-groups** across the whole grid (`ttnn.split_work_to_cores` on the virtual group grid); contiguous tile-row range per group | knob-turn (`block_row_tiles`, grid) |
| W (tiles) | **dependent** — the Σx² spans the whole row | block extent along W = the core's whole W share `core_w_tiles` (resident); split across `w_split_cores` (k) cores | k = `max(k_fit, k_occ)` (see Work Distribution); `core_w_tiles = slice_w_tiles` (or the remainder for the last member) | host `select_blocking()` → RT args `w_tile_start`, `core_w_tiles`, `member_idx`; CT `w_split_cores` | k consecutive Tensix cores in one grid row share every tile-row of their group; member m owns W tiles `[m*slice_w_tiles, min((m+1)*slice_w_tiles, tensor_w_tiles))` | knob-turn for k ∈ [1, 8] (combine built); scheme-change for k > 8 (deferred row) |
| W streaming quantum (`ROW_MAJOR` only) | — (a pipeline quantum inside the W block, not a block extent) | `rm_chunk_w_tiles` | `chunks = ceil(slice_w_tiles/32)`, `rm_chunk_w_tiles = ceil(slice_w_tiles/chunks)` (balanced, ≤ 32) | host → CT arg | n/a (per core) | knob-turn |

Gamma is **reuse-shared** along the row axis: it does not vary by row, so every row-group re-reads it. Operand-reuse check per (operand, chosen split):

| Operand | Split across cores | Varies along it? | Consequence |
|---|---|---|---|
| x | row-groups | yes | none — each byte read once |
| x | W across k members | yes | none |
| gamma | row-groups | **no** → reuse-shared by the split | each group reads its W slice once into a resident CB; broadcast (read once + multicast) is a `deferred` regime row |
| gamma | W across k members | yes | each member reads only its slice (gamma DRAM bytes fall by k) |
| inv_rms (stage output) | W across k members | **no** | every member needs the same per-row value → the combine (all-gather) |

Step-1 table — one row per axis of every data object, no blanks:

| Data | Axis | Character | Block factor (Phase 0) | Core assignment | Buffer depth | Why |
|---|---|---|---|---|---|---|
| x (input) | row | independent | `block_row_tiles` (coarsest fitting) | row-groups over the full grid | `x_block_depth` = 1 if `num_blocks_this_core == 1` else 2 | fills the grid with no communication |
| x (input) | W | dependent | whole `core_w_tiles` (resident) | k members (k = 1 unless occupancy or fit demands) | same CB as above | residency avoids a second DRAM read of x |
| gamma | W | reuse-shared over row | whole `core_w_tiles`, resident for the program | each core its own slice | 1 (program lifetime) | read once per core; ≤ 1/32 of x bytes per block |
| gamma | row | (absent — gamma has no row extent) | n/a | n/a | n/a | gamma is one row |
| sumsq tiles (pass-1 output) | row | independent | `block_row_tiles` | same core as x | 1 block | one fp32 tile per tile-row |
| sumsq tiles | W | collapsed (reduced away) | n/a | n/a | n/a | DEST accumulation folds W |
| partials (k > 1) | row | independent | `block_row_tiles` | every member computes its own | 1 block | one column-0 tile per tile-row |
| gathered partials (k > 1) | row × member | row independent; member axis dependent (summed) | `block_row_tiles × k` | **every member combines all rows redundantly** (symmetric all-gather) | 2 blocks (mechanism cap) | the combine is k·block_row_tiles tile-adds; scattering it would add a second NoC round for < 1 µs of work — no member idles while another computes |
| inv_rms | row | independent | `block_row_tiles` | every member of the group holds it | 1 block | consumed by pass 2 of the same block |
| out | row, W | as x | same block as x | same core as x | streams through `cb_out` (`out_depth = 2`) | written once |

### Buffer-depth knobs

| CB | Depth knob | Phase 0 value | What the depth buys |
|----|------------|---------------|---------------------|
| `cb_x` (TILE path) | `x_block_depth` | 2 when `num_blocks_this_core > 1` (and it fits), else 1 | reader loads block b+1 while compute runs block b; enables the combine lookahead |
| `cb_x` (RM path) | `x_block_depth` | 1 (compute is its own producer: tilize of block b+1 cannot start before pass 2 of block b) | — |
| `cb_x_rm` | `rm_in_depth` | 2 | reader fetches the next tile-row chunk of sticks while compute tilizes |
| `cb_out` (TILE path) | `out_depth` × `out_unit_tiles` | 2 × `min(core_w_tiles, 16)` | writer drains while pass 2 packs |
| `cb_out_rm` | `out_depth` × `rm_chunk_w_tiles` | 2 | writer drains one untilized chunk while compute untilizes the next |
| `cb_gathered_sumsq` | fixed 2 blocks | 2 | **correctness**, not overlap — see Mechanism caps |
| `cb_partial_sumsq`, `cb_sumsq_tiles`, `cb_inv_rms` | 1 block | 1 | — (producer/consumer handoff within one block) |

### Mechanism caps

| Mechanism | Cap on which extent | Clamp | What happens unclamped |
|-----------|--------------------|-------|------------------------|
| Unicast all-gather in one grid row | `w_split_cores` (k) | symmetric all-gather for `k ≤ K_SYM_MAX_INTERLEAVED = 8`; **Refinement 5**: owner combine beyond, `k ≤ K_CAP = min(K_CAP_MAX, grid cores)` with `K_CAP_MAX = (341 − 13) / 2` (writer RT args: one (x, y) pair per member); k ≤ grid_x: a group = k consecutive Tensix cores of one grid row; k > grid_x: groups packed row-major across grid rows | owner combine grows ~linearly in k on the owner (~0.15 µs per member, BH p150b): priced in `_occ_cost` via `OWNER_MEMBER_COST` |
| Lockstep slot reuse in `cb_gathered_sumsq` (peers write into my CB without my `cb_reserve_back`) | its depth | exactly 2 blocks (`2·k·block_row_tiles` pages) | depth 1: a peer's block b+1 partial overwrites my block b slot before my compute pops it → silently wrong inv_rms |
| Identical CB addresses across group members | every CB | all CBs declared on the **full active core range** (same CBDescriptor list for every core) | a peer computes my `cb_gathered_sumsq` slot address from its own write pointer → writes to the wrong L1 |
| Monotonic arrival semaphore | blocks per core | wait for `sem ≥ k·(block_idx+1)`; never reset | resetting races the next block's arrivals → hang or early release |
| W split at tile granularity | `slice_w_tiles` | integral tiles; `k = ceil(tensor_w_tiles / ceil(tensor_w_tiles / k))` so no member gets 0 tiles | a member with 0 tiles deadlocks the gather (peers wait for its partial) |
| BH DRAM read alignment 64 B | RM stick piece offsets, gamma row reads | piece offsets are `w_tile_start·32·elem_bytes` (multiple of 64 B for bf16/fp32); TILE-gamma row 0 read as two 64 B reads (face 0 row 0 at +0, face 1 row 0 at +`face_bytes`) | misaligned reads return wrong bytes on BH |
| `tilize` / `untilize` width is a compile-time template arg | `rm_chunk_w_tiles` | one CT value per program; slices padded in L1 to `core_w_tiles_padded = ceil(core_w_tiles / rm_chunk_w_tiles) · rm_chunk_w_tiles` (pad tiles are zeros) | a ragged last chunk needs a second instantiation **and** breaks the CB ring-wrap invariant (pushes of different sizes do not close the ring) |
| Balanced RM chunk | `rm_chunk_w_tiles` | ≤ 32 tiles, `ceil(slice/ceil(slice/32))` | "widest divisor ≤ 32" turns prime widths into 1-tile chunks (catalog: 6.7× slower) |
| W-padding must be zero before Σx² | last W tile of the last member (and RM pad tiles) | reader zero-fills columns ≥ W in L1 (`fill_l1_range`, `l1_helpers.hpp:90`) after the read, before `cb_push_back` | poisoned/stale padding (golden `poison_padding = 1000`, or stale L1 that may be NaN) leaks into Σx² |
| `eltwise_chain` PerRow DEST accumulation | pass-1 output lifecycle | `PackTile` with `(ReservePolicy::PerOuter, PushPolicy::PerOuter)`, `Dst::D0` (`chain.inl:970`, `:1092`) | static_assert |
| DEST capacity (fp32 DEST: 4 tiles half-sync) | none on the block — the chain clamps its own `block_size`; reduce uses one DST per output | n/a | n/a (no accumulator set must stay live beyond 1 tile) |

### Regimes

Named memory boundary: DRAM. Minimum = x read once, out written once, gamma read once (W·elem_bytes).

| Regime | Status | Predicate | Block | Data movement vs. minimum | What a bigger block buys |
|--------|--------|-----------|-------|---------------------------|--------------------------|
| **R1 row_parallel_resident** | built | `select_blocking()` returns k == 1 | `block_row_tiles × core_w_tiles` (core_w_tiles = tensor_w_tiles) | x, out at minimum; gamma read once per active Tensix core (G copies of W·elem_bytes for RM gamma; G·tensor_w_tiles·128 B for TILE gamma, row 0 only) — the reuse-shared excess, ≤ 1/(32·block_row_tiles) of x bytes per core | per block: one `cb_wait_front`/`cb_pop_front` on cb_x, one pass-1 chain init + DEST acquire/pack per tile-row, one collapse `reduce` call (SFPU macro load), one pass-2 chain init (mul + dest-reuse inits), RM: one tilize/untilize init+uninit; pipeline fill/drain once per block |
| **R2 w_split_combine** (k ∈ [2, 8], symmetric unicast all-gather) | built | k > 1: `k_occ > 1` (tensor_row_tiles·8 ≤ G, tensor_w_tiles ≥ 8, RM: W·elem_bytes/k ≥ 512 B) or `k_fit > 1` (k = 1 does not fit L1) | `block_row_tiles × core_w_tiles` with `core_w_tiles ≈ tensor_w_tiles / k` | x, out at minimum; gamma total falls to G/k copies; **adds** cross-core: per block k² · block_row_tiles fp32 tiles (4 KB each) + k² semaphore increments | everything in R1, plus: one gather round (k·block_row_tiles unicast writes + barrier + k semaphore incs) and one combine `reduce` call per block instead of per tile-row |
| R2 lookahead (schedule variant of R2) | built | R2 and `num_blocks_this_core ≥ 2` (then `x_block_depth = 2`) | same | same as R2 | lifts pass 1 of block b+1 into the gather wait of block b (stall-shadow) |
| two_pass_dram_restream (W chunked in-core, x re-read from DRAM for pass 2) | **rejected** — dead end, superseded by R2 everywhere: R2 keeps the slice resident, R2's combine (k²·block_row_tiles·4 KB NoC) replaces a second full read of x from DRAM | — | chunk × rows | +1 full DRAM crossing of x | — |
| single_core | **rejected** — Phase 0 must be multi-core; R1/R2 fill the grid | — | — | — | — |
| R3 wide_combine (per-row owner gathers its rows' k partials, combines + finalizes, returns inv_rms to all k members) | **implemented (Refinement 2)** for WIDTH / BLOCK_SHARDED, used for every sharded k > 1 (`K_SYM_MAX_SHARDED = 1`: measured 2.2× over the symmetric all-gather on tall BLOCK shards, flat on 1-tile-row shards). Row r of a block → owner member r % k; uniform `ceil(rows/k)·k` gathered pages per member per block (lockstep slot addressing); faces 0/2 payload both ways; return by unicast (multicast return and the row-partial/transpose payload of catalog `tensix_all_reduce` are the next levers). Interleaved: used for k > 8 (Refinement 5). | sharded `k > 1`, interleaved `k > 8` | same | cross-core: k partials into each owner + k inv_rms tiles back per owned row | — |
| gamma_broadcast (one injector reads gamma, `ttnn.Mcast` to the group) | **deferred** — catalog `shared_input_reuse` (2): a resident window with r = window/x-per-core ≤ 0.125 measured 0.96–1.03×; here r ≤ 1/32. Reachable: cb_gamma is already a resident program-lifetime CB with one producer (reader); the Mcast channel only replaces its DRAM read. | — | — | gamma read once instead of G/k times | — |
| HEIGHT_SHARDED resident | **landed in Refinement 4** (k = 1 case of the R2 shard-resident path; gamma multicast k = 1; streamed-gamma / RM chunk-cap fallbacks for wide W — see l1_ledger.md) — placement refinement (Phase 0 SUPPORTED is INTERLEAVED). Knob-turn on R1: core-assignment = shard spec, block = whole shard, cb_x backed on the shard (`ttnn.cb_descriptor_from_sharded_tensor`, zero-copy, no NoC re-read), cb_out backed on the output shard. | `memory_layout == HEIGHT_SHARDED` | shard rows × tensor_w_tiles | input/output never cross DRAM (already in L1) | — |
| WIDTH / BLOCK_SHARDED (k from the shard grid) | **implemented (Refinement 2)**: members, slices and NoC coordinates from the shard spec (orientation-aware; WIDTH groups may span grid rows); block = the resident shard; TILE cb_x / cb_out backed on the shards (zero-copy); RM stick shards copied locally into the tilize staging; combine = R3 owner topology. | `memory_layout ∈ {WIDTH, BLOCK}_SHARDED` | shard | x / out never cross DRAM; R3 cross-core | — |
| reduce_scatter combine (member m finalizes rows r ≡ m mod k, then returns them) | **rejected** — superseded by the symmetric all-gather for k ≤ 8: payload is block_row_tiles one-tile partials; a second NoC round costs more than the redundant k·block_row_tiles tile-adds it saves (catalog `tensix_all_reduce`: small groups/tiny payloads → keep the one-round form) | — | — | — | — |

### Regime selection (exact, host-side, `select_blocking()`)

```
G         = grid_x * grid_y                    # device.compute_with_storage_grid_size()
K_CAP     = min(K_CAP_MAX, grid_x * grid_y)     # Refinement 5 (was min(8, grid_x)); see _choose_k / _occ_cost
budget    = L1 bytes available to CBs per Tensix core (device L1 size − allocator base)
footprint(k, block_row_tiles, x_block_depth)  = expression in l1_ledger.md
slice(k)  = ceil(tensor_w_tiles / k);  k := ceil(tensor_w_tiles / slice(k))   # normalize: no empty member

k_fit = smallest k in [1, K_CAP] with footprint(k, 1, 1) <= budget          # else RuntimeError (W too wide)
k_occ = 1
if 8 * tensor_row_tiles <= G and tensor_w_tiles >= 8:
    k_occ = min(K_CAP, G // tensor_row_tiles, tensor_w_tiles // 4)
    if layout == ROW_MAJOR: k_occ = min(k_occ, max(1, (W * elem_bytes) // 512))
k = normalize(max(k_fit, k_occ))

# Refinements 3 + 5: k_occ above is superseded by _choose_k(): the k >= k_fit minimizing _occ_cost =
#   rows_per_group * (slice + COMBINE_TILE_COST) [+ ceil(rows/k) * k * OWNER_MEMBER_COST for k > 8],
#   candidates while slice >= 4 tiles (RM: pieces >= 512 B) and the owner's gathered CB fits L1.
groups_per_grid_row = grid_x // k ; num_groups = groups_per_grid_row * grid_y   # k > grid_x: (grid_x*grid_y) // k, row-major
split tensor_row_tiles over min(num_groups, tensor_row_tiles) groups → core_row_tiles (2 classes)
if footprint(k, max core_row_tiles, 1) <= budget:
    block_row_tiles = max core_row_tiles ; x_block_depth = 1      # whole assignment, one block
else:
    block_row_tiles = largest b >= 1 with footprint(k, b, 2) <= budget ; x_block_depth = 2
    (if none: block_row_tiles = 1, x_block_depth = 1)
RM: x_block_depth = 1 always.
combine_lookahead = (k > 1 and x_block_depth == 2)
```

Regime-pinned tests are required (the acceptance test pins one shape per regime): R1 many-rows, R2 by occupancy (1 tile-row, W = 4096), R2 by fit (W = 16384), R2 with ≥ 2 blocks per group (lookahead, W = 12288 with 6144 rows: k = 2 by fit, ≥ 3 tile-rows per group on WH and BH), each for `Layout::TILE` and `Layout::ROW_MAJOR`. The predicates use only `grid_x ≥ 2`, so every pinned shape lands in its regime on WH (8×8) and BH (12×10 / 13×10).

### Traffic ranking

Counted as crossings per tensor (DRAM most expensive, then cross-core NoC, then core-local L1 reads).

| Candidate split | DRAM crossings (x / out / gamma) | Cross-core NoC | Core-local | Verdict |
|---|---|---|---|---|
| row-groups only, W resident (R1) | 1 / 1 / G·(W bytes) | 0 | x read twice from L1 (pass 1, pass 2) | **cheapest when it fits and fills the grid** |
| row-groups × W split k, resident slice, combine (R2) | 1 / 1 / (G/k)·(W bytes) | k²·block_row_tiles·4 KB per block | same | **same DRAM as R1, less gamma**; adds a small NoC payload; the only split that fills the grid when tensor_row_tiles ≪ G, and the only one that keeps the slice resident when W is wide |
| row-groups, W chunked in-core, x re-streamed (two-pass) | 2 / 1 / … | 0 | — | rejected: +1 x crossing |
| W split only (all cores share one row) | 1 / 1 / 1·W | k² per row | — | subsumed: R2 with one group |

Chosen: R1 when k = 1 suffices; R2 otherwise. R1 vs R2 tie on x/out DRAM; R2 is preferred only by occupancy (few tile-rows) or by residency (W too wide), both of which are traffic-neutral on DRAM and cost a NoC payload — measured to pay in the catalog's band (`width_split`). Independent-axis-first is the tiebreaker that makes R1 the default.

### Block schedule

Logical schedule per Tensix core (reader / compute / writer realize their parts asynchronously):

```cpp
// once per program
load_gamma_slice();                                   // reader → cb_gamma (resident, expanded rows)
for (uint32_t block_idx = 0; block_idx < num_blocks_this_core; ++block_idx) {
    load_x_block(block_idx);                          // reader: DRAM → cb_x (TILE) / cb_x_rm (RM), zero W-padding
    tilize_x_block(block_idx);                        // compute, RM only: cb_x_rm → cb_x
    sum_squares_block(block_idx);                     // compute: cb_x → cb_sumsq_tiles (DEST-accumulated Σ x², per tile-row)
    collapse_sumsq_block(block_idx);                  // compute: within-tile row collapse; k==1: + finalize → cb_inv_rms
                                                      //                                     k>1: → cb_partial_sumsq
    exchange_partials_block(block_idx);               // writer, k>1: all-gather partials → cb_gathered_sumsq
    combine_partials_block(block_idx);                // compute, k>1: Σ over k members + finalize → cb_inv_rms
    normalize_block(block_idx);                       // compute: x * inv_rms (bcast col) [* gamma] → cb_out
    untilize_out_block(block_idx);                    // compute, RM only: cb_out → cb_out_rm
    store_block(block_idx);                           // writer: cb_out / cb_out_rm → DRAM
}
```

Compute order with `combine_lookahead` (k > 1, ≥ 2 blocks): `sum_squares(0), collapse(0)`, then for each b: `[sum_squares(b+1), collapse(b+1)]`, `combine(b)`, `normalize(b)`. Writer order is unchanged: `exchange(b)` → `store(b)`.

| Block operation | Block shape it acts on | Resident across it | Intended frequency of fixed costs |
|---|---|---|---|
| `load_gamma_slice` | `core_w_tiles` (RM: `core_w_tiles_padded`) gamma tiles | — | once per program: one reserve/push; RM gamma: one contiguous read of the slice per core; TILE gamma: 2 × 64 B reads per tile (row 0); reader expands row 0 into rows 1–31 of every tile in L1 |
| `load_x_block` | `block_row_tiles × core_w_tiles` tiles (RM: × `core_w_tiles_padded`, delivered in `rm_chunk_w_tiles` × 32-stick pieces) | cb_gamma | TILE: one `cb_reserve_back` per block, reads issued for a whole tile-row (`core_w_tiles` reads) per barrier, one push per tile-row; RM: one reserve/barrier/push per (tile-row, chunk); W-padding zero-fill only on the member owning the last W tile |
| `tilize_x_block` (RM) | `block_row_tiles × core_w_tiles_padded`, as `block_row_tiles·chunks` calls of `tilize<rm_chunk_w_tiles>(1)` | cb_gamma | tilize init once per block (`InitOnly` / `Neither` / `UninitOnly`) |
| `sum_squares_block` | `grid(block_row_tiles, core_w_tiles)` (RM: `core_w_tiles_padded`) of cb_x, both operands the same tile | cb_x block (not popped), cb_gamma | one chain call per block: mul init (acc_to_dest) once, one DEST acquire + one pack per tile-row |
| `collapse_sumsq_block` | `ReduceInputBlockShape::of(block_row_tiles, 1)` | cb_x, cb_gamma | one `reduce` call per block (SFPU-macro load once); k == 1: finalize post-op per tile-row |
| `exchange_partials_block` (k > 1) | `block_row_tiles` column-0 partial tiles → `k × block_row_tiles` gathered | cb_x, cb_gamma | one gather round per block: k·block_row_tiles unicast tile writes, one write barrier, k semaphore increments, one semaphore wait |
| `combine_partials_block` (k > 1) | `ReduceInputBlockShape::of(block_row_tiles, k)` with `ReduceWithinTile::Skip` | cb_x, cb_gamma | one `reduce` call per block; finalize post-op per tile-row |
| `normalize_block` | TILE: `grid(block_row_tiles, core_w_tiles)` one chain call; RM: per chunk c `grid(block_row_tiles, rm_chunk_w_tiles)` (Strided over cb_x) | cb_x (popped at end), cb_inv_rms (popped at end), cb_gamma | TILE: mul-bcast-col init + dest-reuse init once per block; RM: once per chunk |
| `untilize_out_block` (RM) | per chunk `untilize<rm_chunk_w_tiles>(block_row_tiles)` | — | untilize init once per block (`InitOnly` / `Neither` / `UninitOnly`) |
| `store_block` | TILE: `block_row_tiles × core_w_tiles` tiles; RM: valid sticks × valid bytes per chunk | — | TILE: per-tile write with flush (lamp: batch); RM: one stick-piece write per stick per chunk, one barrier per chunk |

**Stall-shadow check.** The only stage that waits on something it does not produce is `combine_partials_block` (waits for k−1 peers' partials). Work independent of that wait: (1) **pass 1 of the next block** — independent of inv_rms(b), no FP-order change → built as `combine_lookahead` (requires `x_block_depth = 2`, which R2 already has whenever there is a next block); (2) **`x · gamma`** — legal by commutativity (`x·r·γ = (x·γ)·r`), but needs a block-sized fp32 buffer (or an in-place bf16 rewrite of cb_x that adds a rounding) for the whole wait — rejected on L1 and precision grounds; (3) when a core owns a single block (the occupancy-selected R2 case, ≤ 1 tile-row per group), nothing is independent of the wait — stated, accepted, and bounded by the catalog's ~0.5–1.5 µs combine at k ≤ 8.

### Perf lamps

| Lamp | Why the default may be wrong here | Nearby alternative to measure |
|------|-----------------------------------|-------------------------------|
| Overlap (one-block assignment) | When the whole per-group assignment is one block (`x_block_depth = 1`), the read of x does not overlap pass 1 and the write of the last tiles does not overlap the next read | `block_row_tiles = ceil(core_row_tiles/2)` with `x_block_depth = 2`  — R3: built as `OVERLAP_BLOCKS` (default 1); 2 measured slower on BH prefill (DRAM-write-bound) |
| Overlap at the fit boundary | Wide-W prefill (e.g. 8192×7168) fits k = 1 only at `x_block_depth = 1` → no inter-block overlap | k = 2 with `x_block_depth = 2` |
| Grid synchronization (occupancy k) | `k_occ` takes up to 8 members with ≥ 4 tiles each; at small W the combine (~0.5–1.5 µs) may exceed the per-core time saved | k_occ / 2, and k = 1 (catalog: Wt = 8, k = 2 only 1.17×) |
| Partial payload | Phase 0 ships whole 4 KB fp32 partial tiles; only column 0 (faces 0 and 2) is meaningful | ship faces 0/2 only (2 × 1 KB) — catalog `tensix_all_reduce`: never ship the whole tile |
| Signal fence | write barrier before each semaphore increment | `noc_async_writes_flushed()` before the increment, one barrier at kernel exit (catalog 1.19× at 1–6 tiles/core) |
| Finalize SFPU scope | mul/add/rsqrt post-op runs on all 32 vectors of a tile whose result is column 0 | `c_skip` scope (catalog `sfpu_tile_scope`, 3.84×) |
| Gamma fusion | DestReuse gamma multiply costs ~40 ns/tile DEST→SrcA move (`compute_fusion`) | two-pass (x·inv_rms → scratch, then ·gamma bcast Row) — only worth it if compute-bound |
| Gamma residency | Expanded gamma tiles cost `core_w_tiles` full tiles of L1 for one row of data; it is the term that forces k > 1 earliest for wide W | gamma streamed per W chunk (re-read per block, ≤ 1/32 of x bytes) |
| Writer batching (TILE) | per-tile write + flush | wait/pop in `out_unit_tiles` units with one barrier per unit |
| Gamma row-0 reads (TILE gamma) | 2 × 64 B reads per tile are transaction-count-bound | read whole gamma tiles (2 KB) once per core when `core_w_tiles` is small |

## Dataflow Strategy

| Stage | Format | Mechanism | Notes |
|-------|--------|-----------|-------|
| gamma DRAM → cb_gamma | gamma dtype (Float16_b / Float32), TILE format | reader, TensorAccessor; RM gamma: one read of `core_w_tiles·32·elem_bytes` from stick page 0 at offset `w_tile_start·32·elem_bytes` into tile 0's face area, then RISC-V L1 copies: elems [32j, 32j+16) → tile j face 0 rows 0..15, elems [32j+16, 32j+32) → face 1 rows 0..15, and the same 16-element rows replicated into faces 2/3 (rows 16..31); scatter in descending j so the staging region in tile 0 is consumed last. TILE gamma: per tile, read face 0 row 0 and face 1 row 0 (64 B each; bfloat8_b: the 64 B shared-exponent section rides with the first read and is re-patterned so every face row carries row 0's exponent — Refinement 1), then replicate. Columns ≥ W zero-filled. Pad tiles (RM) all zero. | "expanded gamma": every row of tile j equals gamma[32j .. 32j+31]; enables a non-broadcast DestReuse multiply |
| x DRAM → cb_x (TILE) | input dtype tiles | reader; tile id `t·tensor_w_tiles + w_tile_start + j` for global tile-row t; one barrier per tile-row; last member zero-fills columns ≥ `w_valid_in_last_tile` of its last tile (tile-face addressing: v ≤ 16 → face 0/2 cols v..15 and all of faces 1/3; v > 16 → faces 1/3 cols v−16..15) | cb_x is row-major `[r][j]` within a block |
| x DRAM → cb_x_rm (RM) | input dtype sticks, tile-sized pages | reader; `read_sticks_for_tilize<cb_x_rm>` (`tilize_helpers_dataflow.hpp:88`) per (tile-row, chunk) with `byte_offset_within_page = (w_tile_start + c·rm_chunk_w_tiles)·32·elem_bytes`; stick id `image·H + tr·32 + s`, only `s < H − tr·32` read (stale H-tail rows are harmless: row-local). Chunks with a W tail (last member only) use a raw path: zero-fill the tail columns with `fill_l1_range` then read the valid bytes; pad-only chunks are zero-filled entirely. | H-tail rows never written back |
| cb_x_rm → cb_x (RM) | tiles | compute `tilize<rm_chunk_w_tiles>` | Fp32Mode::Fast (fp32 → tf32 in DEST, same as every FPU consumer downstream) |
| compute pass 1 / collapse | T_acc-format tiles (Float32 / Float16_b per `fp32_dest_acc_en`) | eltwise chain DEST accumulation + reduce helper | see Block Operation Realization |
| partial exchange (k > 1) | T_acc-format tiles (Float32 / Float16_b per `fp32_dest_acc_en`) | writer: `cb_wait_front(cb_partial_sumsq, block_row_tiles)`; `cb_reserve_back(cb_gathered_sumsq, k·block_row_tiles)`; `slot_base = get_write_ptr(cb_gathered_sumsq)` (identical on all members by lockstep); for each member m' (including self) and row r: `noc_async_write(partial[r] → noc(m'), slot_base + (r·k + member_idx)·tile_bytes)`; `noc_async_write_barrier()`; `noc_semaphore_inc(noc(m'), sem_partials_arrived, 1)` for each m'; `cb_pop_front(cb_partial_sumsq)`; `noc_semaphore_wait_min(sem, k·(block_idx+1))`; `cb_push_back(cb_gathered_sumsq, k·block_row_tiles)` | peers' NoC coords from `device.worker_core_from_logical_core`, passed as RT args (k (x, y) pairs) |
| cb_out → DRAM (TILE) | output dtype tiles | writer, TensorAccessor, per tile: wait 1, write, flush, pop 1 | writes all tiles incl. H-padding rows (they are part of the tile pages) |
| cb_out → cb_out_rm → DRAM (RM) | output dtype sticks | compute `untilize<rm_chunk_w_tiles>`; writer `write_sticks_after_untilize<cb_out_rm>` (`tilize_helpers_dataflow.hpp:130`) per (tile-row, chunk) with valid rows only and `row_bytes` = valid bytes of the chunk; pad-only chunks popped without writing | |

Placement axis (`memory_layout`) as the physical realization of the logical shards above:

| `TensorMemoryLayout` value | Axis it cuts | Character | Unlock class | Realization |
|---|---|---|---|---|
| INTERLEAVED | — | — | built (Phase 0) | R1/R2 as above |
| HEIGHT_SHARDED | row | independent | **knob-turn** on R1 | core assignment and per-core extent from the shard spec; block = whole shard (sub-blocked only under L1 pressure); cb_x and cb_out backed on the shards via `ttnn.cb_descriptor_from_sharded_tensor` (never re-read through a TensorAccessor). W-padding zero-fill must not mutate the caller's shard: the last W tile per tile-row is copied into a 1-tile scratch where it is zeroed. RM shards: local stick copy + tilize per `resident_shard_locality`. |
| WIDTH_SHARDED | W | dependent | **scheme-change, implemented (Refinement 2)** | k = number of W shards (up to the whole grid); one group; member i = i-th shard-grid core in orientation order; slices = shard widths (ragged last); W tail masked in compute (`cb_tail_mask` + reduce `Accumulate`), never zeroed in the caller's shard |
| BLOCK_SHARDED | row and W | row independent, W dependent | **implemented (Refinement 2)** | row-groups = shard-grid rows, k = shard-grid columns (orientation-aware: ROW_MAJOR shard (g, i) at (x0+i, y0+g), COL_MAJOR at (x0+g, y0+i)); ragged last shard row / column own only their valid tile-rows / W tiles |

Structural impossibilities: none beyond the INVALID entries already in `feature_spec.py`.

## Work Distribution

| Field | Value |
|-------|-------|
| Work unit | a block: `block_row_tiles` tile-rows × the member's W slice |
| Grid | `device.compute_with_storage_grid_size()` = (grid_x, grid_y); a runtime parameter, never a literal |
| Row-groups | virtual grid `(grid_x // k, grid_y)`; `ttnn.split_work_to_cores(virtual_grid, tensor_row_tiles)` → two group classes with `core_row_tiles` = ceil / floor; virtual group (vx, vy) → physical Tensix cores `(vx·k + m, vy)`, m ∈ [0, k) |
| Per-core work | tile-rows `[group_row_start, group_row_start + core_row_tiles)`, W tiles `[w_tile_start, w_tile_start + core_w_tiles)`; `num_blocks_this_core = ceil(core_row_tiles / block_row_tiles)` |
| Remainder | rows: two `split_work_to_cores` classes; last block of a core has `last_block_row_tiles = core_row_tiles − (num_blocks_this_core−1)·block_row_tiles` and is always the core's final ring use (actual counts pushed/waited/popped are safe); W: last member's `core_w_tiles = tensor_w_tiles − (k−1)·slice_w_tiles`; geometry uses `ceil` per image (`tensor_row_tiles = num_images·ceil(H/32)`) |
| Idle Tensix cores | not in the program's core range (no kernels launched) |
| Kernel descriptors | one reader, one writer, one compute KernelDescriptor over the active core range; per-core values are RT args. CT args identical across cores (`w_split_cores`, `has_gamma`, `is_row_major`, `rm_chunk_w_tiles`, `combine_lookahead`, CB ids, accessor args). |

## Circular Buffers

T_in / T_out / T_g = tile bytes of the input / output / gamma dtype; T_acc = tile bytes of the DEST-width intermediate format: Float32 (4096 B) when `fp32_dest_acc_en=True`, Float16_b (2048 B) when False (Refinement 1).

| Semantic Name | Index | Page Size | Num Pages | Sizing rationale | Format | Producer | Consumer | Lifetime |
|---------------|-------|-----------|-----------|------------------|--------|----------|----------|-----------|
| `cb_x_rm` (RM only) | 0 | T_in | `rm_in_depth · rm_chunk_w_tiles` | streams over row and W in (32 sticks × chunk) units | input dtype | reader | compute | per (tile-row, chunk) |
| `cb_x` | 1 | T_in | `x_block_depth · block_row_tiles · core_w_tiles_max` (RM: `core_w_tiles_padded`) | spans the block (row × W): resident across pass 1 and pass 2 | input dtype | reader (TILE) / compute (RM) | compute | per block |
| `cb_gamma` (has_gamma) | 2 | T_g | `core_w_tiles_max` (RM: padded) | spans W; row axis absent | gamma dtype | reader | compute | program |
| `cb_sumsq_tiles` | 3 | T_acc | `block_row_tiles` | spans row; W reduced away | T_acc format | compute | compute | pass 1 → collapse |
| `cb_partial_sumsq` (k > 1) | 4 | T_acc | `block_row_tiles` | spans row | T_acc format | compute | writer | collapse → exchange |
| `cb_gathered_sumsq` (k > 1) | 5 | T_acc | `2 · w_split_cores · block_row_tiles` | spans row × member, 2 blocks (mechanism cap) | T_acc format | writer | compute | exchange → combine |
| `cb_inv_rms` | 6 | T_acc | `block_row_tiles` | spans row | T_acc format | compute | compute | finalize → end of normalize |
| `cb_out` | 7 | T_out | TILE: `out_depth · out_unit_tiles`; RM: `block_row_tiles · rm_chunk_w_tiles` | TILE streams; RM spans one chunk column of the block (untilize input) | output dtype | compute | writer (TILE) / compute (RM) | per tile / per chunk |
| `cb_out_rm` (RM only) | 8 | T_out | `out_depth · rm_chunk_w_tiles` | streams | output dtype | compute | writer | per (tile-row, chunk) |

`core_w_tiles_max` = `slice_w_tiles` (the non-last members' share; CB capacity is one value for every core). Semaphore: `sem_partials_arrived` (k > 1), initial 0, on the active core range. Every CB is declared on the full active core range (Mechanism caps). Push/wait parity: cb_x pushes per tile-row (`core_w_tiles`), compute waits the whole block (`block_rows_this · core_w_tiles`); cb_gathered pushes `k·block_row_tiles`, compute's reduce waits/pops `k·block_row_tiles`; cb_sumsq/cb_partial/cb_inv_rms `block_rows_this` per block; cb_out per tile (TILE) / per chunk (RM).

## Block Operation Realization

| # | Block operation | Block shape | Helper? | Input CB (semantic name, pages, state) | Output CB (semantic name, pages) | CB state after |
|---|-----------------|-------------|---------|----------------------------------------|----------------------------------|----------------|
| 0 | `compute_kernel_hw_startup` | — | API | cb_x, cb_x, cb_out (RM: cb_x_rm, cb_x) | — | once, first statement of MAIN |
| 1 | `tilize_x_block` (RM) | `block_row_tiles × chunks` calls of width `rm_chunk_w_tiles` | yes | cb_x_rm, `rm_chunk_w_tiles` per call, popped | cb_x, `rm_chunk_w_tiles` per call | cb_x holds the block |
| 2 | `sum_squares_block` | `IterationShape::grid(block_rows_this, core_w_tiles[_padded])` | yes | cb_x, caller `cb_wait_front(block)`; operands A = B = cb_x, `WaitPolicy::None / PopPolicy::None`, `InputTileMapping::Block`; with lookahead `TileAddressing::Offset` base = `block_row_tiles·core_w_tiles` for block b+1 | cb_sumsq_tiles, 1 per tile-row (`PerOuter/PerOuter`) | cb_x **not** popped |
| 3a | `collapse_sumsq_block` (k == 1) | `ReduceInputBlockShape::of(block_rows_this, 1)` | yes | cb_sumsq_tiles, `block_rows_this`, BulkWaitBulkPop | cb_inv_rms, `block_rows_this` | finalize post-op applied |
| 3b | `collapse_sumsq_block` (k > 1) | same | yes | same | cb_partial_sumsq, `block_rows_this` | raw Σx² per row in column 0 |
| 4 | `exchange_partials_block` (writer) | `block_row_tiles` → `k·block_row_tiles` | raw dataflow | cb_partial_sumsq (popped) | cb_gathered_sumsq (pushed `k·block_row_tiles`) | slot half = block parity |
| 5 | `combine_partials_block` (k > 1) | `ReduceInputBlockShape::of(block_rows_this, k)` (tile `r·k + m`) | yes | cb_gathered_sumsq, BulkWaitBulkPop | cb_inv_rms, `block_rows_this` | finalize post-op applied |
| 6 | `normalize_block` | TILE: `grid(block_rows_this, core_w_tiles)`; RM: per chunk `grid(block_rows_this, rm_chunk_w_tiles)`, cb_x `Strided{base = c·rm_chunk_w_tiles, row_stride = core_w_tiles_padded}`, cb_gamma `Offset{c·rm_chunk_w_tiles}` | yes | cb_x (None/None, Block), cb_inv_rms (`BroadcastDim::Col`, None/None, `InputTileMapping::Col`), cb_gamma (None/None, `InputTileMapping::Row`) | cb_out (PerTile/PerTile) | after the block: caller pops cb_x (`block_rows_this·core_w_tiles`) and cb_inv_rms (`block_rows_this`) |
| 7 | `untilize_out_block` (RM) | per chunk `untilize<rm_chunk_w_tiles>(block_rows_this)` | yes | cb_out, `rm_chunk_w_tiles` per tile-row | cb_out_rm, `rm_chunk_w_tiles` per tile-row | — |

Finalize post-op (used by 3a and 5): on DST[0] — `binop_with_scalar_tile_init(); mul_unary_tile(dst, bits(1/W)); add_unary_tile(dst, bits(eps)); rsqrt_tile_init(); rsqrt_tile(dst);` (1/W and eps as fp32 bit patterns, RT args; `W` logical). In step 5 (`ReduceWithinTile::Skip`) the reduce leaves the SFPU untouched, so the post-op's own inits are mandatory (`reduce_helpers_compute.hpp:185-187`).

Scaler-CB argument of `reduce`: the AccumulateViaAdd path with no partial tile and the default `CopySeedPairs` reload never reads the scaler CB (`reduce_helpers_compute.hpp:150`; zero tile only for `CopySeedZeroPair`, `.inl:511-518`). Pass `cb_x` (allocated, distinct from input and output; asserted at `.inl:335`). No scaler or mask CB is allocated — padding is zeroed in L1 by the reader instead.

## API Mapping

| Block operation | Type | Function | File:Line | Template Params / Args | Input CB | Output CB | Which params are block knobs |
|-----------------|------|----------|-----------|------------------------|----------|-----------|------------------------------|
| startup | raw_api (required boot) | `compute_kernel_hw_startup(icb0, icb1, ocb)` | `tt_metal/hw/inc/api/compute/compute_kernel_hw_startup.h:60` | (cb_x, cb_x, cb_out) / RM (cb_x_rm, cb_x_rm, cb_x) | — | — | — |
| `tilize_x_block` | helper | `compute_kernel_lib::tilize` | `ttnn/cpp/ttnn/kernel_lib/tilize_helpers.hpp:197` | `<rm_chunk_w_tiles, cb_x_rm, cb_x, InitOnly/Neither/UninitOnly, WaitBlock, UnpackAndPackReconfigure, Fp32Mode::Fast>(1)` | cb_x_rm | cb_x | `rm_chunk_w_tiles` |
| `sum_squares_block` | helper | `eltwise_chain` + `BinaryFpu<BinaryFpuOp::Mul, A, B, Dst::D0, DestAccumulation::PerRow>` + `PackTile<output(cb_sumsq_tiles, PerOuter, PerOuter, Enabled, Direct, DestAccumulation::PerRow)>` | `eltwise/api/chain.hpp:533`, `:491`, `:504`; `DestAccumulation` `:283`; acc_to_dest mul init `eltwise/core/chain.inl:1184`; lifecycle rule `chain.inl:970` | A = B = `input(cb_x, WaitPolicy::None, PopPolicy::None, InputTileMapping::Block, Enabled, Direct | Offset)` | cb_x | cb_sumsq_tiles | shape `grid(block_rows_this, core_w_tiles)` |
| `collapse_sumsq_block` | helper | `compute_kernel_lib::reduce` | `reduce_helpers_compute.hpp:636` | `<PoolType::SUM, ReduceDim::REDUCE_ROW, cb_sumsq_tiles, cb_x, cb_inv_rms | cb_partial_sumsq, ReduceInputPolicy::BulkWaitBulkPop, INPUT_AND_OUTPUT, ReduceFp32Mode::Fast, ReduceAlgorithm::AccumulateViaAdd, ReduceWithinTile::Collapse>(ReduceInputBlockShape::of(block_rows_this, 1), contiguous(), NoAccumulation{}, finalize_or_NoOp)` | cb_sumsq_tiles | cb_inv_rms / cb_partial_sumsq | `block_rows_this` |
| `combine_partials_block` | helper | `compute_kernel_lib::reduce` | `reduce_helpers_compute.hpp:636`; Skip `:195` | `<SUM, REDUCE_ROW, cb_gathered_sumsq, cb_x, cb_inv_rms, BulkWaitBulkPop, INPUT_AND_OUTPUT, Fast, AccumulateViaAdd, ReduceWithinTile::Skip>(of(block_rows_this, w_split_cores), contiguous(), NoAccumulation{}, finalize)` | cb_gathered_sumsq | cb_inv_rms | `block_rows_this`, `w_split_cores` |
| finalize post-op | raw_api inside a helper callback | `mul_unary_tile`, `add_unary_tile`, `binop_with_scalar_tile_init`, `rsqrt_tile`, `rsqrt_tile_init` | `tt_metal/hw/inc/api/compute/eltwise_unary/binop_with_scalar.h:60, :35, :156`; `eltwise_unary/rsqrt.h:44, :21` | dst index from the helper; fp32 bit RT args | DST | DST | — |
| `normalize_block` | helper | `eltwise_chain` + `BinaryFpu<Mul, input(cb_x,…), input(cb_inv_rms, BroadcastDim::Col, None, None, InputTileMapping::Col)>` + `DestReuseBinary<Mul, input(cb_gamma, None, None, InputTileMapping::Row), DestReuseType::DEST_TO_SRCA>` (has_gamma) + `PackTile<output(cb_out)>` | `chain.hpp:491`, `:500`, `:504`, `:533`; Strided ctor `chain.inl:1164` | RM: cb_x `TileAddressing::Strided`, cb_gamma `TileAddressing::Offset` | cb_x, cb_inv_rms, cb_gamma | cb_out | shape `grid(block_rows_this, core_w_tiles)` / `grid(block_rows_this, rm_chunk_w_tiles)` |
| `untilize_out_block` | helper | `compute_kernel_lib::untilize` | `untilize_helpers.hpp:154` | `<rm_chunk_w_tiles, cb_out, cb_out_rm, InitOnly/Neither/UninitOnly>(block_rows_this)` | cb_out | cb_out_rm | `rm_chunk_w_tiles` |
| RM reader | helper | `dataflow_kernel_lib::read_sticks_for_tilize` | `tilize_helpers_dataflow.hpp:88` | `<cb_x_rm>(accessor, valid_rows, chunk_bytes, start_stick, byte_offset)` | DRAM | cb_x_rm | `rm_chunk_w_tiles` |
| RM writer | helper | `dataflow_kernel_lib::write_sticks_after_untilize` | `tilize_helpers_dataflow.hpp:130` | `<cb_out_rm>(accessor, valid_rows, valid_bytes, start_stick, byte_offset)` | cb_out_rm | DRAM | `rm_chunk_w_tiles` |
| W-padding / pad-tile zero fill | helper | `fill_l1_range<val_size>` | `ttnn/cpp/ttnn/kernel_lib/l1_helpers.hpp:90` | byte ranges of the tail columns / pad tiles | cb_x / cb_x_rm / cb_gamma (in reader) | same | — |
| TILE x / out / gamma DRAM access | raw_api (dataflow) | `TensorAccessor`, `noc_async_read`/`noc_async_write`, barriers | `tech_reports/tensor_accessor/tensor_accessor.md` | — | DRAM | CBs | — |
| `exchange_partials_block` | raw_api (dataflow) | `noc_async_write`, `noc_async_write_barrier`, `noc_semaphore_inc`, `noc_semaphore_wait_min`, `get_noc_addr(x, y, addr)` | `tt_metal/hw/inc/api/dataflow/dataflow_api.h` | — | cb_partial_sumsq | cb_gathered_sumsq | `w_split_cores`, `block_row_tiles` |

Helpers considered and rejected (raw_api entries):

| raw_api entry | Candidate helper | Mismatch (file:line) | Reason |
|---|---|---|---|
| finalize post-op | `eltwise_chain` SFPU elements (`Rsqrt`, `eltwise/unary/math.hpp:38`) | `reduce()` exposes the post-reduce hook as a `uint32_t dst_idx` callable (`reduce_helpers_compute.hpp:483`, `:636`), not a chain | a chain call would need the reduced tile packed to L1 and re-read (an extra CB and pass); the hook is the helper's own extension point |
| `exchange_partials_block` | `ttnn.Mcast` with `McastRotatingSenderConfig` (`mcast/host/mcast.hpp:105`) | one sender per round: k members ⇒ k serialized multicast rounds with a handshake each | catalog `tensix_all_reduce`: for K ≤ 8 a single symmetric unicast round wins; multicast only pays from K ≥ 12 (R3) |
| `exchange_partials_block` | `ccl_helpers_dataflow.hpp` | multi-device CCL transport, not intra-chip Tensix peers | wrong scope |
| `sum_squares_block` alternative | `compute_kernel_lib::square` (`eltwise/api/convenience.hpp:61`) + `reduce` AccumulateViaAdd | needs a block-sized fp32 x² CB and a second pass | DEST accumulation in the chain folds x² into the accumulator (Rule 3 pattern 4; catalog 2.01×) |
| W-padding masking | `prepare_reduce_mask` + `ReducePartialScaler::only_partial()` (`reduce_helpers_dataflow.hpp:74`, `reduce_helpers_compute.hpp:344`) | the mask applies only inside `reduce()`'s input walk; pass 1 is a chain, not a reduce | the reader zero-fill makes padding 0 before any compute; also makes the RM path NaN-safe (a 0×NaN mask would not) |

## Broadcast Verification

| Phase | Op | CB_A (semantic name) Valid Region | CB_B (semantic name) Valid Region | Broadcast Dim |
|-------|-----|-----------------------------------|-----------------------------------|---------------|
| sum_squares | Mul (acc_to_dest) | cb_x: All | cb_x: All | None |
| normalize (step 1) | Mul | cb_x: All | cb_inv_rms: Col0 (REDUCE_ROW output) | `BroadcastDim::Col` |
| normalize (step 2) | DestReuse Mul | DEST: All | cb_gamma: All (rows expanded by the reader) | None |
| combine | AccumulateViaAdd + Skip | cb_gathered_sumsq: Col0 | — | — |

## Key Risks and Gotchas

| Risk | Why it bites here | Mitigation in this design |
|------|-------------------|---------------------------|
| Padding leaks into Σx² | golden poisons TILE padding with 1000; RM tails are stale L1 (possibly NaN) | reader zero-fills columns ≥ W (and RM pad tiles) in L1 before `cb_push_back`; 1/W uses the logical W |
| H-padding rows / stale RM sticks | NaN in unused rows | all compute is row-local (DEST accumulation is elementwise; REDUCE_ROW collapses within a row); those rows are never written (RM) or are tile padding (TILE) |
| Gathered-slot overwrite | peers write into my CB without credit | depth exactly 2 blocks + lockstep argument (a peer's partial(b+2) needs my partial(b+1), produced only after I popped gathered(b)); holds with lookahead too because the writer order stays `exchange(b) → store(b)` |
| CB addresses differ across cores | slot address is computed from the sender's own write pointer | identical CBDescriptor list on the whole active core range |
| Semaphore race | counts from block b+1 can arrive before block b is consumed | monotonic target `k·(block_idx+1)`, never reset |
| Ring-wrap with ragged widths | `tilize`/`untilize` CT widths and linear block reads | RM slices padded in L1 to a multiple of `rm_chunk_w_tiles`; TILE cb_out streams per tile; cb_x pushes per tile-row and capacity is a multiple of `block_row_tiles·core_w_tiles`; the ragged row block is always the core's last |
| DEST-accumulating multiply | `mul_init(..., acc_to_dest=1)` path is less exercised than add | verify pass 1 alone early on a 1-tile shape against `sum(x²)`; fallback: `square` → fp32 scratch CB (chunked) → `reduce` AccumulateViaAdd with `Accumulate::at` |
| fp32 precision | FPU paths truncate srcA/srcB to tf32 (tilize Fast also) | fp32 PCC threshold 0.999 tolerates ~5e-4 relative; DEST stays fp32; if RMS tolerance fails, move the square to SFPU (`CopyTile` + `Square<>`, `eltwise/unary/misc.hpp:35`) with `UnpackToDestFp32` on cb_x |
| `fp32_dest_acc_en` contract | precision convention | read the caller's flag; `float32 + False` → EXCLUSION (NotImplementedError); `bfloat16 + False` outside Phase 0 SUPPORTED; intermediate CBs follow the DEST width (`acc_dtype()`: Float32 / Float16_b); Refinement 1 adds `bfloat16 + False` and `bfloat8_b + {True, False}` |
| Regime only on some grids | k depends on grid size | predicates need only `grid_x ≥ 2`; acceptance test pins one shape per regime |
| `memory_config` kwarg | golden runner passes it for sharded cells | accept it; Phase 0 validate refuses non-INTERLEAVED with NotImplementedError |
| Gamma with leading dims (1,1,1,W) in TILE | tile-row 0 holds gamma, rows 1..31 padding (possibly poisoned) | read only row 0 of each gamma tile and expand in L1 |
