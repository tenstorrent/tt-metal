# Operation Design: layer_norm (layer_norm_tile)

## Overview

| Field | Value |
|-------|-------|
| Classification | compute (row-wise normalization, fused reduce + eltwise) |
| Goal | One-dispatch LayerNorm over the last dim of a TILE-layout, L1-interleaved bf16 tensor `(B, 1, S, 768)`, with required gamma/beta, any `S` (tile-aligned or not), multi-core from Phase 0, caller-selectable compute config. |
| Math | `mean[r] = (1/W) sum_j x[r,j]`; `xc[r,j] = x[r,j] - mean[r]`; `var[r] = (1/W) sum_j xc[r,j]^2`; `out[r,j] = gamma[j] * xc[r,j] * rsqrt(var[r] + eps) + beta[j]` |
| Mode | Derivative (helpers-first; no pre-existing implementation consulted) |
| Package | `ttnn/ttnn/operations/layer_norm_tile/` (module `ttnn.operations.layer_norm_tile`, exports `layer_norm`, `INPUT_TAGGERS`, `SUPPORTED`, `EXCLUSIONS`, `validate`, `default_compute_kernel_config`). The mandated import path `from ttnn.operations.layer_norm import layer_norm` is provided by a one-line re-export package `ttnn/ttnn/operations/layer_norm/__init__.py` (`from ttnn.operations.layer_norm_tile import layer_norm, default_compute_kernel_config`). |
| Dispatch | Exactly one `ttnn.generic_op` per call. Host-side work is validation, parameter derivation and output allocation only. |
| References | `.claude/references/blocking-model.md`, `.claude/references/l1-footprint-discipline.md`, `.claude/references/precision_convention.md`, `ttnn/cpp/ttnn/kernel_lib/reduce_helpers_compute.hpp`, `ttnn/cpp/ttnn/kernel_lib/eltwise/api/{chain,convenience}.hpp`, `tt_metal/tt-llk/tt_llk_wormhole_b0/llk_lib/llk_unpack_AB.h`, catalog `ttnn/ttnn/operations/examples/{compute_block_size,compute_fusion,reduce_accumulate,row_reduce_accumulate,double_buffer,read_coalescing,mcast_topology,shared_input_reuse,width_split,eltwise_l1_vs_dest_accumulate}/README.md` |

### feature_spec.py status

`eval/golden_tests/layer_norm_tile/feature_spec.py` is **absent** in this tree. `/golden-tests` must author it; the planner does not. This design names the TARGET axes it is written against, by the names `validate()` gates on:

| Axis | Values (TARGET) | Phase 0 intent |
|------|-----------------|----------------|
| `dtype` | `ttnn.bfloat16` | bf16 |
| `fp32_dest_acc_en` | `True`, `False` | both. The only thing that changes is the page format of the intermediate CBs (one derived parameter). |
| `layout` | `ttnn.TILE_LAYOUT` | TILE |
| `alignment` | `"tile_aligned"`, `"h_non_aligned"` | both. W is pinned at 768 by the contract, so `w_non_aligned` is not in the universe (see regime `w-partial-masked`). |
| `affine_layout` | `ttnn.TILE_LAYOUT`, `ttnn.ROW_MAJOR_LAYOUT` | both: one mechanism, two address generators (Dataflow Strategy). |
| `memory_layout` | `INTERLEAVED` | interleaved. The buffer may be L1 (the contract) or DRAM; TensorAccessor handles both. |

INPUT_TAGGERS (op file): `tag_alignment(inputs, axes)` returns `"tile_aligned"` if `shape[-2] % 32 == 0 and shape[-1] % 32 == 0`, `"h_non_aligned"` if only W is aligned, and `"w_non_aligned"` otherwise. `"w_non_aligned"` is outside SUPPORTED and raises `UnsupportedAxisValue`.

Structural impossibilities: none beyond the defaults.

The requirement's call pattern `layer_norm(input, compute_kernel_config=...)` omits gamma/beta, but both are **required positional** parameters. That pattern is a `TypeError` by signature and is not a supported call.

## Parameters

| Name | Type | Required | Valid Range | Default | CT/RT |
|------|------|----------|-------------|---------|-------|
| `input_tensor` | `ttnn.Tensor` | yes | bf16, TILE, interleaved, `shape[-1] % 32 == 0`, `ceil(W/32) <= W_TILES_RESIDENT_MAX (32)` | - | - |
| `gamma` | `ttnn.Tensor` | yes | bf16, `volume == W`, last dim divides W; TILE (any rank, row 0 valid) or ROW_MAJOR with last dim `Wg`, `Wg % 32 == 0`; interleaved DRAM or L1 | - | reader CT: `affine_is_tile`, `affine_stick_elems` |
| `beta` | `ttnn.Tensor` | yes | same as gamma, same layout as gamma | - | same CT args as gamma |
| `epsilon` | `float` (kw-only) | no | `> 0` | `1e-5` | compute **RT** arg (fp32 bits), so changing epsilon does not recompile |
| `compute_kernel_config` | `ttnn.ComputeConfigDescriptor` (kw-only) | no | any fidelity / approx; `fp32_dest_acc_en` in {True, False} | `default_compute_kernel_config()` = HiFi4, `fp32_dest_acc_en=True`, `math_approx_mode=False` | `ComputeConfigDescriptor` passed through to the compute kernel |

Entry-point order: (1) resolve `None` config through `default_compute_kernel_config()`, the single source; (2) input dtype not bf16 raises **`ValueError`** whose message contains the word `dtype` (requirement: outside the op's dtype universe); (3) `validate()` runs SUPPORTED / EXCLUSIONS and raises `UnsupportedAxisValue` / `ExcludedCell` (`NotImplementedError` subclasses); (4) allocate output; (5) build descriptor; (6) one `ttnn.generic_op`. No `to_layout`, `tilize`, `pad`, `slice` or any other host-side tensor op is allowed.

## Tensors

### Input

| Property | Requirement |
|----------|-------------|
| Shape | `(B, 1, S, 768)` per contract; `1024 <= B*S <= 4096`, `S <= 512`, S arbitrary (non-tile-aligned allowed). The design folds all leading dims, so any rank >= 2 works. |
| Dtype | bfloat16 |
| Layout | TILE (32x32). Each image `(n, c)` is tile-padded independently. |
| Memory | interleaved, L1 per contract (DRAM also works) |

### Gamma / Beta

| Form | Shape | Layout / memory | Pages | Valid bytes per tile of W |
|------|-------|-----------------|-------|---------------------------|
| `affine-tile` | `(768,)` or `(1,1,1,768)` | TILE, DRAM interleaved | 24 tiles (row 0 valid) | face0-row0 (32 B at tile offset 0) + face1-row0 (32 B at tile offset 512); read as `affine_read_bytes` each (64 B on Blackhole DRAM; row 1 of the face comes along and is ignored) |
| `affine-rm` | `(1,1,24,32)` (or `(1,1,1,768)`) | ROW_MAJOR, L1 interleaved | 24 sticks of 64 B (or 1 stick of 1536 B) | 64 contiguous bytes at `stick = 32*j / Wg`, `offset = (32*j mod Wg) * 2` |

### Output

| Property | Value |
|----------|-------|
| Shape | identical to input (logical and padded) |
| Dtype | bfloat16 (= input) |
| Layout | TILE |
| Memory | identical to input's memory config (L1 interleaved per contract) |

## Blocking Model

### Axes

Index space: `tensor_row_tiles = prod(shape[:-2]) * ceil(H/32)` tile-rows (per-image ceil) x `tensor_w_tiles = ceil(W/32)` = 24 tile-columns. Under the contract `tensor_row_tiles = B * ceil(S/32)`, which is 32..128 for power-of-two S and up to 248 for heavily padded S (e.g. S=33).

| Axis | Character (+ one-clause reason) | Extent knob | Phase 0 value | Knob source | Core-assignment | Later unlock |
|------|--------------------------------|-------------|---------------|-------------|-----------------|--------------|
| tile-row (`row`) | **independent**: each output row depends only on its own input row (mean/var/normalize never mix rows). Padding rows of a partial tile-row are rows like any other. | `block_rows` | `min(max_core_rows, BLOCK_ROWS_MAX)` with `BLOCK_ROWS_MAX = 2`. This is the whole per-core assignment for every power-of-two-S input on any grid of >= 64 cores (target device: Blackhole, 11x10 = 110 cores, so R <= 128 gives core_rows <= 2). | host constant `BLOCK_ROWS_MAX`; `block_rows` is a compute + reader + writer **CT** arg derived once on host; `num_blocks_this_core = ceil(core_rows / block_rows)` is an **RT** arg | `ttnn.split_work_to_cores(grid, tensor_row_tiles)` over the **full** `compute_with_storage_grid_size()`; group-1 cores own `core_rows_g1`, group-2 cores own `core_rows_g2`. The grid is a host parameter, not a literal. | knob-turn (raise `BLOCK_ROWS_MAX`; shrink to 1 + depth 2, see overlap lamp) |
| tile-column (`w`) | **dependent**: mean and variance span the whole row, and the output needs both. | `block_w_tiles` | `tensor_w_tiles` (24): whole extent, not split. Two-pass LN re-touches `xc` after the variance, and a resident row turns the re-read into an L1-local pass. | `block_w_tiles = tensor_w_tiles`, computed once on host, CT arg to all three kernels | not assigned across cores. Every core owns whole rows. | scheme-change (`w-split-combine`, deferred) |
| affine vector (`gamma`/`beta` along `w`) | **reuse-shared**: does not vary along the split `row` axis, so every core needs the identical 768-wide vectors (operand-reuse check on the chosen split). | `block_w_tiles` (same knob; the affine CBs span `w`) | 24 tiles, resident for the kernel lifetime, row 0 valid only | same as `w` | replicated: each participating core reads its own copy **once per kernel** (only 64 valid B per tile) | read-once + mcast is `rejected` (regime table) |
| intra-tile rows (32 rows of a tile-row) | independent, but **not splittable**: FPU/SFPU/pack operate on whole tiles. | none (tile-atomic) | 32 (hardware) | hardware tile shape | - | - |

Intermediate data re-run of this table:

| Intermediate | Axes it carries | Assignment |
|--------------|-----------------|------------|
| `mean`, `sqsum`, `rstd` (1 tile per tile-row, column 0 valid) | `row` (spans `block_rows`); `w` collapsed | computed on the core that owns the row. No collector core, no stage where one core works while the others wait. |
| `xc` (centered, then scaled in place) | `row` x `w` (spans the block) | same core, resident across phases P2..P7 |

**Stall-shadow check.**

| Stall | What waits | Work scheduled into it | Permission |
|-------|------------|------------------------|------------|
| Compute waits on reader for block 0 | compute P1 | none possible in compute (nothing resident yet) | - |
| Affine fetch (48 + 48 small reads, possibly DRAM latency) | compute P6 of block 0 | **reader issues block-0 input reads first, then gamma/beta reads**, so the affine fetch overlaps P1..P5 of block 0 | no FP reorder; only DM ordering |
| Reader waits on CB space for block i+1 | reader | with `input_depth_blocks = 2` the reader fills the second slot during compute of block i | - |
| (deferred `w-split-combine`) waiting on peers' partial sumsq | compute | `xc * gamma` can run in the wait, because `(xc*gamma)*rstd == (xc*rstd)*gamma` (commutative). This changes FP rounding order, so gate it on the precision baseline. | algebraic (commutativity) |

### Buffer-depth knobs

| CB | Depth knob | Phase 0 value | What the depth buys |
|----|------------|---------------|---------------------|
| `cb_input_tiles` | `input_depth_blocks` | `min(2, max_num_blocks)` (host-derived; 1 when every core has a single block, because a second slot could never be filled) | reader prefetches block i+1 while compute runs block i |
| `cb_output_tiles` | `output_depth_rows` (window = `output_depth_rows * block_w_tiles`) | 2 | P7 packs tile-row r+1 while the writer drains tile-row r; the writer batches one tile-row (24 writes) per barrier |
| `cb_centered` | none (1 block) | 1 | sequential helpers on one TRISC set. A second block would only matter with cross-block compute pipelining, which compute-on-one-thread cannot do. |
| `cb_gamma_rows`, `cb_beta_rows` | none | 1 x `block_w_tiles` | persistent constants |
| `cb_row_stat`, `cb_rstd` | none | 1 x `block_rows` | produced and consumed by compute within a block |

### Mechanism caps

| Mechanism | Cap on which extent | Clamp | What happens unclamped |
|-----------|--------------------|-------|------------------------|
| Resident-W working set (L1) | `block_w_tiles` | `validate()`: `tensor_w_tiles <= W_TILES_RESIDENT_MAX = 32` (contract: 24) | CB allocation overflows L1 (program build failure) or collides with L1-resident input/output banks |
| L1 footprint, `row` extent | `block_rows` | `block_rows = min(max_core_rows, BLOCK_ROWS_MAX = 2)` (ledger total <= ~600 KB, leaving >= 700 KB of L1 for the L1-interleaved input/output banks and firmware) | allocation failure on heavily padded shapes (tile_rows up to 248 -> core_rows 4) |
| Affine half-row NoC read | read size `affine_read_bytes` | TILE source: `affine_read_bytes = max(32, alignment(affine buffer))` <= 512 (one face; BH DRAM = 64 B reads rows 0-1 of the face, row 1 ignored). RM source: halves start at `o` and `o + 32`, so `validate()` requires `32 % alignment(affine buffer) == 0` (L1 16 B passes; RM in BH DRAM, 64 B, is rejected as outside the contract) and `Wg % 32 == 0` | a misaligned NoC read returns shifted bytes, giving silently wrong gamma/beta |
| ROW broadcast reads only row 0 of srcB faces 0/1 (`llk_unpack_AB.h:96-106,233-260`) | affine CB content | reader writes only row 0 (2 x 32 B of valid data per tile; TILE-DRAM reads drag in row 1); other rows never initialized | n/a: rows 1..31 are never read. If a later realization switches to non-broadcast (DestReuse lamp), the tiles must be fully replicated first. |
| Packer L1-accumulate (P3) | one accumulator tile per tile-row | one chain call per tile-row of the block (internal loop of `sqsum_block`) | n/a: `PackTile` L1 accumulation is one-output-tile per chain (`chain.inl:1133-1140`) |
| DEST capacity | none at block level | helpers own subblocking (`DEST_AUTO_LIMIT`, `dest_helpers.hpp:103`) | - |

### Regimes

| Regime | Status | Predicate | Block | Data movement vs. minimum | What a bigger block buys |
|--------|--------|-----------|-------|---------------------------|--------------------------|
| `row-split-resident-w` | **built** | `W % 32 == 0 and tensor_w_tiles <= 32` (always true under the contract) | `block_rows x block_w_tiles` | **minimum** for input/output: each input tile crosses the NoC (L1 banks to core) once, each output tile once, 0 DRAM crossings for x/out. Affine: `P x 3 KB` (P participating cores) vs. 3 KB minimum (replication, see the two `affine-*` rows). Intermediates are core-local. | Per block, at intended frequency once per block: 6 helper-phase inits + format reconfigs (P1..P7, about 0.3 us/phase per `compute_block_size`), 7 CB handshakes per phase boundary, pipeline fill/drain. `block_rows = 2` halves these vs. 1 per tile-row. No regime-added terms. |
| `w-streaming-reread` | **rejected**: superseded by `row-split-resident-w` everywhere in the contract (W = 768 means 24 tiles always fit) | `tensor_w_tiles > 32` | `block_rows x w_chunk` | re-reads x 2x (mean pass, center/var pass, normalize pass), i.e. 3x input NoC traffic | - (dead end: no shape in the universe needs it) |
| `w-split-combine` | **deferred**: occupancy-only scheme-change. Its domain is `tensor_row_tiles <= num_cores / 2` (k >= 2 slices must fit the grid): on the 110-core Blackhole grid that is R <= 55, i.e. the B*S = 1024 aligned class (R = 32 -> k = 3, 96 cores) and some non-aligned shapes (R = 44, 48). R = 64 / 96 / 128 gain nothing (k = 2 would exceed the grid). It moves **more** bytes than the built regime on every tier (adds 2 combine rounds), and the built regime is correct and at the traffic minimum for every input. Positive reason it is not Phase 0: the cross-core combine reorders the mean/variance summation across cores, so it is gated on the precision baseline that the built regime establishes. Reachable: `block_w_tiles` is already a knob, 1/W uses the **logical** W (not the slice), and the stats live in their own CBs (`cb_row_stat`, `cb_rstd`) so a combine op slots between P1/P2 and P3/P4 without touching other phases. | would be `tensor_row_tiles <= num_cores / 2`; `k` = largest divisor of `tensor_w_tiles` with `k * tensor_row_tiles <= num_cores` | `block_rows x (tensor_w_tiles / k)` on k cores per row | input/output unchanged; affine per core / k; adds per tile-row 2 rounds x (k-1) x 1 tile (partial-sum tiles, column 0 valid) gather + 1 tile mcast back, each gated on a semaphore. Combine: `reduce<SUM, ..., AccumulateViaAdd, ..., ReduceWithinTile::Skip>` over gathered partials (`reduce_helpers_compute.hpp:195-237`); transport: `ttnn.Mcast` + `McastArgs` | cuts per-core compute k-fold (3x at R = 32 on BH); adds 2 semaphore round trips per block (grid-sync lamp applies) |
| `affine-full-tile-read` | **rejected**: superseded by `affine-row0-read` (same CB, same compute, 32x fewer bytes) | - | 24 full tiles per tensor per core | `P x 96 KB` DRAM (10.5 MB at P = 110, more than the whole input at R = 128) | - |
| `affine-row0-read` (prologue, part of the built regime) | **built** | always | 24 tiles, 2 x `affine_read_bytes` per tile per tensor | `P x 3 KB` (`P x 6 KB` for TILE-DRAM on BH) (DRAM for `affine-tile`, L1 NoC for `affine-rm`) | once per kernel; hidden behind block-0 compute (stall-shadow) |
| `affine-read-once-mcast` | **rejected**: superseded by `affine-row0-read`. The per-core payload is 3 KB of latency-bound small reads already overlapped with block-0 compute. Mcast puts a semaphore rendezvous on every core's critical path to save bytes that are not on it (catalog `mcast_topology`: the gain shrinks to nothing as slices get small). | - | - | would be 3 KB + mcast to P cores | - |
| `height-sharded-in-place` | **deferred**: outside the contract (`memory_layout` TARGET = INTERLEAVED). Reachable as a knob-turn: a height shard = the `row` split pinned by the caller. Consume via `ttnn.cb_descriptor_from_sharded_tensor` on `cb_input_tiles` / `cb_output_tiles` (zero-copy), `block_rows = shard rows` clamped by `BLOCK_ROWS_MAX`. | `memory_layout == HEIGHT_SHARDED` | shard rows x `block_w_tiles` | input/output 0 NoC crossings | - |
| `width/block-sharded` | **deferred**: outside the contract. Scheme-change: cuts the dependent `w` axis, so it is `w-split-combine` with caller-pinned k. | `memory_layout in {WIDTH, BLOCK}_SHARDED` | - | - | - |
| `w-partial-masked` | **deferred**: the contract pins W = 768 (aligned). Mechanism if W ever becomes non-aligned: `ReducePartialScaler::only_partial()` + `prepare_reduce_mask` mask CB for P1/P4 (`reduce_helpers_compute.hpp:308-357`) and a masked P3 last column. | `W % 32 != 0` | unchanged | unchanged | - |

Regime-selection function: a single built compute regime; `validate()` rejects inputs outside its predicate. The affine form (`affine-tile` vs `affine-rm`) is a **reader CT selection** (`affine_is_tile`), not a compute regime. Tests are still pinned on both forms.

### Traffic ranking

Per call, `R = tensor_row_tiles`, tile = 2 KB (bf16), `P` = participating cores.

| Rank | Split | Input (NoC from L1 banks) | Output | Affine | Cross-core | Verdict |
|------|-------|---------------------------|--------|--------|------------|---------|
| 1 | `row` (built) | `R * 48 KB`, once | `R * 48 KB`, once | `P * 3 KB` | 0 | chosen: minimum on every tier except affine replication (3 KB/core, hidden) |
| 2 | `w` into k slices + combine | `R * 48 KB` | `R * 48 KB` | `P * 3 KB / k` | `R * 2 rounds * k tiles` (about `R * k * 4 KB`) + 2 semaphore rendezvous per block | deferred: strictly more bytes; only buys occupancy when `R < num_cores` |
| 3 | `row` with streaming W | `3 * R * 48 KB` | `R * 48 KB` | `P * 3 KB` | 0 | rejected (dead end) |

Occupancy on the 110-core Blackhole grid (kept separate from the ranking): `R` = 32 / 64 / 96 / 128 puts 32 / 64 / 96 / 110 cores to work with 1 / 1 / 1 / 1-2 rows each (R = 128: 18 cores own 2 rows and set the critical path). Idle cores remain for R < 110, but splitting `w` only fits the grid for R <= 55, the deferred row's domain.

### Block schedule

```cpp
// prologue (once per kernel): load_affine() - reader, issued AFTER block 0's input reads
for (uint32_t block_idx = 0; block_idx < num_blocks_this_core; ++block_idx) {
    load_block(block_idx);        // reader: block_rows x block_w_tiles tiles -> cb_input_tiles
    mean_block(block_idx);        // P1
    center_block(block_idx);      // P2
    sqsum_block(block_idx);       // P3
    rstd_block(block_idx);        // P4
    scale_block(block_idx);       // P5
    gamma_block(block_idx);       // P6
    shift_block(block_idx);       // P7 -> cb_output_tiles
    store_block(block_idx);       // writer: real tile-rows only
}
```

| Op | Block shape | Resident across it | Intended fixed-cost frequency |
|----|-------------|--------------------|-------------------------------|
| `load_affine` | `block_w_tiles` tiles x 2 tensors, 2 x `affine_read_bytes` each | written into `cb_gamma_rows` / `cb_beta_rows`, held for kernel lifetime | once per kernel; one barrier for all 96 reads; one push of `block_w_tiles` per CB |
| `load_block` | `block_rows x block_w_tiles` | - | per tile-row: reserve `block_w_tiles`, issue `block_w_tiles` page reads, one barrier, push. Ragged tile-rows (beyond `core_rows`) are reserved and pushed at the **nominal** count without reading (stale data, never written out). |
| `mean_block` (P1) | `ReduceInputBlockShape::of(block_rows, block_w_tiles)` | `cb_input_tiles` block stays resident (WaitUpfrontNoPop) | one reduce setup per block (AccumulateViaAdd light reconfig + sfpu_reduce init) |
| `center_block` (P2) | `IterationShape::grid(block_rows, block_w_tiles)` | `cb_row_stat` (mean) held for the block (Upfront, popped AtEnd); pops `cb_input_tiles` tile by tile | one sub-bcast-col init per block |
| `sqsum_block` (P3) | `block_rows` x `IterationShape::tiles(block_w_tiles)` (one L1-accumulated output tile per tile-row) | `cb_centered` resident, not popped | one mul init per tile-row call (the init is hoisted when the chain is uniform). This is the internal loop of a real block op: no per-tile init. |
| `rstd_block` (P4) | `ReduceInputBlockShape::of(block_rows, 1)` | `cb_centered` resident | one reduce setup per block; post-op SFPU inits once per output tile (`block_rows` per block) |
| `scale_block` (P5) | `grid(block_rows, block_w_tiles)` | `cb_rstd` held (Upfront, AtEnd); `cb_centered` rewritten in place | one mul-bcast-col init per block |
| `gamma_block` (P6) | `grid(block_rows, block_w_tiles)` | `cb_gamma_rows` held (Upfront, never popped until kernel end) | one mul-bcast-row init per block |
| `shift_block` (P7) | `grid(block_rows, block_w_tiles)` | `cb_beta_rows` held; consumes `cb_centered` | one add-bcast-row init per block; pushes per tile (writer streams) |
| `store_block` | `block_rows x block_w_tiles` | - | per tile-row: wait `block_w_tiles`, issue writes if the row is real, one barrier, pop `block_w_tiles` (nominal) |

### Perf lamps

| Lamp | Why the default may be wrong here | Nearby alternative to measure |
|------|-----------------------------------|-------------------------------|
| **Overlap** | `block_rows = core_rows = 2` puts each core's whole assignment in one block, so read, compute and write serialize per core (no next block to prefetch) | `BLOCK_ROWS_MAX = 1` with `input_depth_blocks = 2`: block i+1's read overlaps block i's compute, at the cost of 2x the per-block fixed cost |
| **Grid synchronization** | none in the built regime (no collectives). It applies to the deferred `w-split-combine`, where 1 tile-row per core leaves ~12 tiles of work against 2 semaphore rounds. | (for `w-split-combine`) k = 2 vs. leaving 32 cores idle |
| Normalize-tail fusion | P5/P6/P7 are three L1 passes over the fp32 block (catalog `compute_fusion`: FPU-after-FPU DEST reuse is 0.94-1.02x, so L1 passes are the default) | one chain: `BinaryFpu<Mul>(xc, rstd bcast Col) -> DestReuseBinary<Mul>(gamma_full) -> DestReuseBinary<Add>(beta_full) -> PackTile(cb_output_tiles)` (`chain.hpp:580`). Needs fully row-replicated gamma/beta tiles (DestReuse has no broadcast), so `load_affine` must replicate row 0 down the tile. |
| Variance accumulation site | P3 accumulates sum(xc^2) in L1 through the packer (`L1Accumulation::Enabled`), which is correct for any FPU op | `BinaryFpu<Mul, ..., DestAccumulation::PerRow>` over `grid(block_rows, block_w_tiles)` (one call per block, no L1 RMW; catalog `eltwise_l1_vs_dest_accumulate`: DEST 2x faster than packer-L1-acc for Add). **Valid only after a probe confirms ELWMUL accumulates into DEST**: the WH MOP passes `acc_to_dest = 0` for ELWMUL (`llk_math_eltwise_binary.h:153-183`). |
| Writer batch | one tile-row (24 writes) per barrier | 8-tile batches with `output_depth` 2 (smaller CB, earlier first write) |

## Dataflow Strategy

| Stage | Format | Mechanism | Notes |
|-------|--------|-----------|-------|
| input pages (L1 banks) to core | bf16 tile (2 KB) | reader: `TensorAccessor(input)`, `noc_async_read_page(page, acc, l1)`; tile id = `row_tile * tensor_w_tiles + w` | one barrier per tile-row; no coalescing possible (interleaved pages, catalog `read_coalescing`) |
| gamma/beta to core | bf16, row 0 of each tile | reader: `TensorAccessor(gamma).get_noc_addr(page, offset)` (`tensor_accessor.h:155`), 2 reads of `affine_read_bytes` per tile into CB page offsets `0` and `512`. TILE: `page = j`, src offsets `0` / `512`, `affine_read_bytes = max(32, buffer alignment)`. RM: `page = 32*j / Wg`, src offset `o = (32*j mod Wg)*2`, then `o` / `o + 32`, 32 B each. | one barrier for both tensors; issued after block-0 input reads |
| reader to compute | `cb_input_tiles` bf16; `cb_gamma_rows`, `cb_beta_rows` bf16 | CB | - |
| compute internal | `cb_row_stat`, `cb_rstd`, `cb_centered`: **Float32** if `fp32_dest_acc_en` else **Float16_b** | CB (compute to compute) | single derived parameter `intermediate_format` |
| compute to writer | `cb_output_tiles` bf16 | CB, per-tile push | - |
| core to output pages | bf16 tile | writer: `TensorAccessor(output)`, `noc_async_write_page`; one barrier per tile-row; ragged rows skipped | - |

Placement (TARGET `memory_layout`) per scheme: INTERLEAVED (built). HEIGHT_SHARDED is a **knob-turn** (cuts the independent `row` axis; zero-copy CB on the shard). WIDTH/BLOCK_SHARDED is a **scheme-change** (cuts the dependent `w` axis; needs the `w-split-combine` gather/mcast contract: each core reduces its slice to a column-0-valid partial tile, a designated row-leader gathers k partials, sums them with `reduce<..., AccumulateViaAdd, ..., ReduceWithinTile::Skip>`, and mcasts the result tile back to the k cores; two rounds, mean then sumsq).

Non-tile-aligned H: no masking. Padding rows of the last tile-row of each image are independent rows (per-row stats never mix rows). Whatever values they hold, they produce values only in padding rows of the output. Reader and writer move the padded tile footprint unchanged.

## Work Distribution

| Field | Value |
|-------|-------|
| Work unit | a block of `block_rows` tile-rows x `block_w_tiles` tiles |
| Grid | `device.compute_with_storage_grid_size()` (host parameter); `ttnn.split_work_to_cores(grid, tensor_row_tiles)` gives `(num_cores, all_cores, group_1, group_2, core_rows_g1, core_rows_g2)` |
| Per-core work | RT args: `start_row_tile`, `core_rows`, `num_blocks_this_core = ceil(core_rows / block_rows)`. CT: `block_rows = min(core_rows_g1, BLOCK_ROWS_MAX)`, `block_w_tiles`. |
| Remainder | `tensor_row_tiles = prod(shape[:-2]) * ceil(H/32)` (per-image ceil, never `floor(N*H/32)`); `tensor_w_tiles = ceil(W/32)`. A core whose `core_rows` is not a multiple of `block_rows` runs its last block at **nominal** CB counts; reader/writer skip the I/O of tile-rows `>= core_rows`. |
| Idle cores | `tensor_row_tiles < grid cores` leaves `grid - tensor_row_tiles` cores idle (deferred `w-split-combine`). Kernels are created only on `all_cores`. |

## Circular Buffers

`T_in = tile_size(bf16) = 2048`, `T_aff = tile_size(gamma.dtype) = 2048`, `T_int = tile_size(intermediate_format)` = 4096 (fp32 DEST) or 2048, `T_out = 2048`.

| Semantic Name | Index | Page Size | Num Pages | Sizing rationale | Format | Producer | Consumer | Lifetime |
|---------------|-------|-----------|-----------|------------------|--------|----------|----------|-----------|
| `cb_input_tiles` | 0 | `T_in` | `input_depth_blocks * block_rows * block_w_tiles` | spans `row` (block_rows) and `w` (block_w_tiles): P1 needs the whole block before P2 starts; depth for prefetch | input dtype (bf16) | reader | compute | load_block to P2 |
| `cb_gamma_rows` | 1 | `T_aff` | `block_w_tiles` | spans `w`; constant across `row` | gamma dtype (bf16) | reader | compute | kernel |
| `cb_beta_rows` | 2 | `T_aff` | `block_w_tiles` | spans `w`; constant across `row` | beta dtype (bf16) | reader | compute | kernel |
| `cb_row_stat` | 3 | `T_int` | `block_rows` | spans `row`; `w` collapsed. Holds mean (P1 to P2), then sumsq partial (P3 to P4). | intermediate | compute | compute | per block, two sequential uses |
| `cb_rstd` | 4 | `T_int` | `block_rows` | spans `row` | intermediate | compute | compute | P4 to P5 |
| `cb_centered` | 5 | `T_int` | `block_rows * block_w_tiles` | spans `row` x `w`: read by P3, P5, P6, P7 | intermediate | compute | compute | P2 to P7 (in place in P5, P6) |
| `cb_output_tiles` | 6 | `T_out` | `output_depth_rows * block_w_tiles` | streams over `row`; spans `w` (one tile-row writer batch) | output dtype (bf16) | compute | writer | P7 to store_block |

Push/wait balance per block (nominal): `cb_input_tiles` push `block_rows*block_w_tiles` (per tile-row) / P1 waits all, P2 pops 1 at a time. `cb_row_stat` P1 pushes `block_rows` / P2 pops `block_rows` (AtEnd); P3 pushes `block_rows` (1 per tile-row call) / P4 pops `block_rows`. `cb_rstd` P4 pushes `block_rows` / P5 pops `block_rows` (AtEnd). `cb_centered` P2 pushes N; P3 waits N, no pop; P5 pops N, pushes N; P6 pops N, pushes N; P7 pops N (`N = block_rows*block_w_tiles`). `cb_output_tiles` P7 pushes N per tile / writer pops `block_w_tiles` x `block_rows`. `cb_gamma_rows` / `cb_beta_rows` push `block_w_tiles` once / waited Upfront each block, popped once after the last block.

## Block Operation Realization

| # | Block operation | Block shape | Helper? | Input CB (semantic name, pages, state) | Output CB (semantic name, pages) | CB state after |
|---|-----------------|-------------|---------|----------------------------------------|----------------------------------|----------------|
| 0 | `load_affine` | 24 tiles x 2 | dataflow (raw NoC) | gamma/beta tensors | `cb_gamma_rows` (24), `cb_beta_rows` (24) | both resident until kernel end |
| 1 | `load_block` | `block_rows x block_w_tiles` | dataflow | input tensor | `cb_input_tiles` (N) | N pushed |
| 2 | `mean_block` | `of(block_rows, block_w_tiles)` | yes: `reduce_mean` | `cb_input_tiles` (N, WaitUpfrontNoPop) | `cb_row_stat` (block_rows) | input still resident |
| 3 | `center_block` | `grid(block_rows, block_w_tiles)` | yes: `sub` | `cb_input_tiles` (Scalar map, PerTile/PerTile), `cb_row_stat` (Col map, bcast Col, Upfront/AtEnd) | `cb_centered` (N) | input and mean popped |
| 4 | `sqsum_block` | `block_rows` x `tiles(block_w_tiles)` | yes: `eltwise_chain` (BinaryFpu Mul + PackTile L1-acc) | `cb_centered` (A=B, Block map, Upfront/None, `TileAddressing::Offset` base `r*block_w_tiles`) | `cb_row_stat` (1 per call, OneUpfront/OneAtEnd, L1Accumulation::Enabled) | centered still resident |
| 5 | `rstd_block` | `of(block_rows, 1)` | yes: `reduce` + post-op | `cb_row_stat` (block_rows, BulkWaitBulkPop) | `cb_rstd` (block_rows) | row_stat popped |
| 6 | `scale_block` | `grid(block_rows, block_w_tiles)` | yes: `mul` in place | `cb_centered` (Scalar, PerTile/PerTile), `cb_rstd` (Col, bcast Col, Upfront/AtEnd) | `cb_centered` (PerTile/PerTile) | centered rewritten; rstd popped |
| 7 | `gamma_block` | `grid(block_rows, block_w_tiles)` | yes: `mul` in place | `cb_centered` (Scalar, PerTile/PerTile), `cb_gamma_rows` (Row map, bcast Row, Upfront/None) | `cb_centered` | gamma not popped |
| 8 | `shift_block` | `grid(block_rows, block_w_tiles)` | yes: `add` | `cb_centered` (Scalar, PerTile/PerTile), `cb_beta_rows` (Row map, bcast Row, Upfront/None) | `cb_output_tiles` (PerTile/PerTile) | centered empty |
| 9 | `store_block` | `block_rows x block_w_tiles` | dataflow | `cb_output_tiles` | output tensor | - |

Kernel boot: `compute_kernel_hw_startup(cb_input_tiles, cb_gamma_rows, cb_row_stat)` exactly once as the first statement of compute `MAIN()`, before any helper. Every later phase keeps helper-owned format reconfig enabled (`ReduceDataFormatReconfigMode::INPUT_AND_OUTPUT`, `DataFormatReconfig::Enabled`), because adjacent phases change formats (bf16 to fp32 and back). After the last block, compute pops `block_w_tiles` from `cb_gamma_rows` and `cb_beta_rows`.

In-place P5/P6 correctness: the chain pops each input tile before reserving the output tile (`chain.inl:2806-2820`: compute-side wait/exec/pop, then pack-side reserve/pack/push), so an in-place pass over a full `cb_centered` ring makes progress and preserves order. Each in-place phase carries real CB credit (PerTile pop + push), which satisfies the cross-phase ordering rule in `l1-footprint-discipline.md` Rule 3.2.

## API Mapping

| Block operation | Type | Function | File:Line | Template Params / Args | Input CB | Output CB | Which params are block knobs |
|-----------------|------|----------|-----------|------------------------|----------|-----------|------------------------------|
| `mean_block` | helper | `compute_kernel_lib::reduce_mean` | `ttnn/cpp/ttnn/kernel_lib/reduce_helpers_compute.hpp:691` (impl `reduce_helpers_compute.inl:1300-1335`) | `<REDUCE_ROW, cb_input_tiles, cb_gamma_rows /*scaler id, never waited on the aligned path, inl:320-337*/, cb_row_stat, ReduceInputPolicy::WaitUpfrontNoPop, INPUT_AND_OUTPUT, ReduceFp32Mode::Fast, ReduceAlgorithm::AccumulateViaAdd>(ReduceInputBlockShape::of(block_rows, block_w_tiles), W_logical)` | `cb_input_tiles` | `cb_row_stat` | `block_rows`, `block_w_tiles` (shape) |
| `center_block` | helper | `compute_kernel_lib::sub` | `eltwise/api/convenience.hpp:61` | `<input(cb_input_tiles, PerTile, PerTile, Scalar), input(cb_row_stat, BroadcastDim::Col, Upfront, AtEnd, Col), output(cb_centered)>(IterationShape::grid(block_rows, block_w_tiles), ...)` | `cb_input_tiles`, `cb_row_stat` | `cb_centered` | grid shape |
| `sqsum_block` | helper | `compute_kernel_lib::eltwise_chain` with `BinaryFpu<BinaryFpuOp::Mul, A=B=input(cb_centered, Upfront, None, Block, Enabled, Offset)>` + `PackTile<output(cb_row_stat, OneUpfront, OneAtEnd, Enabled, Direct, DestAccumulation::Disabled, L1Accumulation::Enabled)>` | `eltwise/api/chain.hpp:568` (BinaryFpu), `:588` (PackTile), `:620` (eltwise_chain); L1-acc lifecycle `eltwise/core/chain.inl:1127-1140`; same-CB A/B dedup `chain.inl:1284-1291`; legal Block policies `chain.inl:59-63` | `IterationShape::tiles(block_w_tiles)`, base `r*block_w_tiles` for both operands, one call per `r < block_rows` | `cb_centered` | `cb_row_stat` | `block_w_tiles` (shape), `block_rows` (call count) |
| `rstd_block` | helper | `compute_kernel_lib::reduce` | `reduce_helpers_compute.hpp:637` (AccumulateViaAdd path `reduce_helpers_compute.inl:540-600`) | `<PoolType::SUM, REDUCE_ROW, cb_row_stat, cb_gamma_rows /*unused scaler*/, cb_rstd, BulkWaitBulkPop, INPUT_AND_OUTPUT, Fast, AccumulateViaAdd, ReduceWithinTile::Collapse>(ReduceInputBlockShape::of(block_rows, 1), contiguous(), NoAccumulation{}, post_op)`; `post_op(dst)`: `mul_unary_tile(dst, inv_w_bits); add_unary_tile(dst, eps_bits); rsqrt_tile_init(); rsqrt_tile(dst);` (`binop_with_scalar.h:35,60`, `rsqrt.h:44`; the scalar ops free-ride on the finalize's SFPU init, `reduce_helpers_compute.inl:1324-1331`) | `cb_row_stat` | `cb_rstd` | `block_rows` |
| `scale_block` | helper | `compute_kernel_lib::mul` (in place) | `eltwise/api/convenience.hpp:70` | `<input(cb_centered, PerTile, PerTile, Scalar), input(cb_rstd, BroadcastDim::Col, Upfront, AtEnd, Col), output(cb_centered)>(grid(block_rows, block_w_tiles))` | `cb_centered`, `cb_rstd` | `cb_centered` | grid shape |
| `gamma_block` | helper | `compute_kernel_lib::mul` (in place) | `eltwise/api/convenience.hpp:70`; ROW bcast semantics `chain.hpp:330-335`, `llk_unpack_AB.h:96-106` | `<input(cb_centered, PerTile, PerTile, Scalar), input(cb_gamma_rows, BroadcastDim::Row, Upfront, None, Row), output(cb_centered)>(grid(block_rows, block_w_tiles))` | `cb_centered`, `cb_gamma_rows` | `cb_centered` | grid shape |
| `shift_block` | helper | `compute_kernel_lib::add` | `eltwise/api/convenience.hpp:52` | `<input(cb_centered, PerTile, PerTile, Scalar), input(cb_beta_rows, BroadcastDim::Row, Upfront, None, Row), output(cb_output_tiles)>(grid(block_rows, block_w_tiles))` | `cb_centered`, `cb_beta_rows` | `cb_output_tiles` | grid shape |
| `load_block` / `store_block` | raw_api (dataflow) | `TensorAccessor` + `noc_async_read_page` / `noc_async_write_page` + barriers | `tt_metal/hw/inc/api/tensor/tensor_accessor.h:155` | per tile-row batch | input tensor | output tensor | `block_rows`, `block_w_tiles` |
| `load_affine` | raw_api (dataflow) | `TensorAccessor::get_noc_addr(page, offset)` + `noc_async_read` (`affine_read_bytes`) | `tensor_accessor.h:155` | 2 reads per tile per tensor | gamma/beta | `cb_gamma_rows`, `cb_beta_rows` | `block_w_tiles` |

Helpers considered and rejected:

| Raw entry | Candidate helper | Mismatch (file:line) | Reason |
|-----------|------------------|----------------------|--------|
| `load_block` / `store_block` | `local_copy_helpers_dataflow.hpp`, `tilize_helpers_dataflow.hpp` | `tilize_helpers_dataflow.hpp` (RM-stick to tile path) | input/output are already TILE pages from an interleaved tensor; the only operation is page-granular NoC read/write, for which no dataflow helper exists beyond `TensorAccessor` itself |
| `load_affine` | `reduce_helpers_dataflow.hpp` (`prepare_reduce_scaler`) / `eltwise/generators/fill.hpp` | `reduce_helpers_dataflow.hpp:1-157` builds constant scaler tiles | gamma/beta are tensor data scattered into row 0 of each tile, not a generated constant; no helper performs a "stick/face-row to tile row 0" scatter |
| (no scaler CB) | `prepare_reduce_scaler` (pool-type-aware) | `reduce_helpers_compute.inl:320-337` | AccumulateViaAdd on an aligned reduce never waits the scaler CB; 1/W is applied as `mul_unary_tile` (P1 via `reduce_mean`, P4 via post-op). **No reduce-scaler CB exists in this design**, so the bf16-packed scaler rule has nothing to apply to. |

## Broadcast Verification

| Phase | Op | CB_A Valid Region | CB_B Valid Region | Broadcast Dim |
|-------|-----|-------------------|-------------------|---------------|
| P2 `center_block` | sub | `cb_input_tiles`: All | `cb_row_stat` (REDUCE_ROW out): Col0 | Col |
| P3 `sqsum_block` | mul | `cb_centered`: All | `cb_centered`: All | None |
| P5 `scale_block` | mul | `cb_centered`: All | `cb_rstd` (REDUCE_ROW out + SFPU post-op): Col0 | Col |
| P6 `gamma_block` | mul | `cb_centered`: All | `cb_gamma_rows` (1D `[W]`): Row0 | Row |
| P7 `shift_block` | add | `cb_centered`: All | `cb_beta_rows` (1D `[W]`): Row0 | Row |

## Key Risks and Gotchas

| Risk | Why it bites here | Mitigation in this design |
|------|-------------------|---------------------------|
| Block mapping is illegal with streaming policies | `chain.inl:59-63`: `Block` requires PerBlockSize, Cumulative or Upfront/None wait | streamed operands use `InputTileMapping::Scalar` + PerTile/PerTile (front tile each step); only P3 (resident, Upfront/None) uses `Block` |
| Gamma/beta CB rows 1..31 uninitialized | reader writes only row 0 | ROW broadcast reads only row 0 of faces 0/1 (`llk_unpack_AB.h:96-106,233`). Any switch to a non-broadcast consumer (DestReuse lamp) must replicate rows first. |
| ELWMUL ignores `acc_to_dest` in the WH MOP | `llk_math_eltwise_binary.h:153-183` hardcodes `acc_to_dest = 0` for ELWMUL | P3 accumulates in L1 via the packer; the DEST-accumulate variant is a lamp gated on a probe |
| fp32 intermediates vs bf16 I/O | P2 (bf16 A, fp32 B), P5..P7 (fp32 A, fp32/bf16 B), P7 packs bf16 | every helper call keeps format reconfig enabled; `intermediate_format` is the single source for `cb_row_stat`, `cb_rstd`, `cb_centered` |
| Ragged last block on a core | `core_rows % block_rows != 0` (e.g. R = 96 or 248) | nominal CB counts in all three kernels; reader/writer skip I/O for tile-rows `>= core_rows`; compute is raggedness-agnostic. Stale data in skipped rows may produce NaN, which is never written. |
| Non-tile-aligned S | padding rows inside the last tile-row of each image | no masking needed (rows independent); tile-row count uses per-image `ceil` |
| Affine RM vs TILE source offsets | TILE: halves at src offsets 0 / 512 of page j; RM: 64 contiguous bytes at `(32*j / Wg, (32*j mod Wg)*2)` | one reader routine, CT `affine_is_tile` + `affine_stick_elems`; destination offsets are always 0 / 512 in the CB page |
| Unused scaler id in reduce | AccumulateViaAdd without partial never touches the scaler CB (`reduce_helpers_compute.inl:320-337`) but the template slot needs an id distinct from input/output | pass `cb_gamma_rows` (allocated, never waited by the reduce) |
| L1-resident input/output + CBs | input/output L1-interleaved banks (up to ~190 KB/bank each at R = 248) compete with CBs | `BLOCK_ROWS_MAX = 2` keeps CBs <= ~600 KB (see ledger) |
| Mandated import path | `ttnn.operations.layer_norm` vs package dir `layer_norm_tile` | re-export package `ttnn/ttnn/operations/layer_norm/__init__.py` |
| Dtype rejection type | requirement says `ValueError`; registry refusals are `NotImplementedError` | dtype check (bf16 only) raises `ValueError` before `validate()` |
