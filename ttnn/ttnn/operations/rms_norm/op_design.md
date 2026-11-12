# Operation Design: rms_norm

## Overview

| Field | Value |
|-------|-------|
| Classification | compute (row reduction + broadcast eltwise), one `ttnn.generic_op` dispatch |
| Goal | `RMSNorm(x) = x / sqrt(mean(x², dim=-1) + eps) * gamma`, natively for TILE and ROW_MAJOR, tile-aligned and non-aligned H/W, bf16 and fp32, multi-core from Phase 0 |
| Math | `rstd[r] = rsqrt( (1/W)·Σ_{c<W} x[r,c]² + eps )`; `out[r,c] = x[r,c] · rstd[r] · gamma[c]` (gamma term omitted when `gamma is None`) |
| References | `eval/golden_tests/rms_norm/feature_spec.py` (TARGET/INPUTS/INVALID/LOOSE_CASES), `eval/golden_tests/rms_norm/helpers.py` (reference + TOLERANCES), `.claude/references/blocking-model.md`, `.claude/references/l1-footprint-discipline.md`, `.claude/references/precision_convention.md`, `.claude/references/generic_op_template/`, kernel lib `ttnn/cpp/ttnn/kernel_lib/` (reduce, eltwise chain, tilize/untilize, dataflow tilize helpers), catalog `ttnn/ttnn/operations/examples/master.md` |
| Companion artifact | `ttnn/ttnn/operations/rms_norm/l1_ledger.md` (CB ledger + data-movement budget) |

## Parameters

| Name | Type | Required | Valid Range | Default | CT/RT |
|------|------|----------|-------------|---------|-------|
| `input_tensor` | `ttnn.Tensor` | yes | rank ≥ 2; bf16/fp32; TILE or ROW_MAJOR; Phase 0: interleaved (DRAM or L1) | — | — |
| `gamma` (kw-only) | `Optional[ttnn.Tensor]` | no | shape `(1,1,1,W)` (numel == W, last dim == input last dim), ROW_MAJOR, bf16/fp32 | `None` | `has_gamma` CT |
| `epsilon` (kw-only) | `float` | no | > 0 | `1e-6` | fp32 bit pattern, compute CT |
| `compute_kernel_config` (kw-only) | `ttnn.ComputeConfigDescriptor` | no | Phase 0: `fp32_dest_acc_en=True`; any `math_fidelity` / `math_approx_mode` | `default_compute_kernel_config()` (HiFi4, fp32 dest on, approx off) | passed as-is as the compute kernel `config=` |
| `memory_config` (kw-only) | `Optional[ttnn.MemoryConfig]` | no | Phase 0: interleaved | DRAM interleaved | — |
| `program_config` (kw-only) | accepted for call-signature compatibility (`axes.classify_call` passes it); ignored in Phase 0 | no | `None` | `None` | — |

Derived host constants (all computed once in the program-descriptor builder; single source each):

| Constant | Definition |
|----------|------------|
| `W` | `input.shape[-1]` |
| `tensor_w_tiles` (`Wc`) | `ceil(W / 32)` |
| `w_partial_elems` | `W % 32` (0 = aligned; else valid elems in the last W tile) |
| `inv_w_bits` | fp32 bits of `1.0 / W` (logical W — never `Wc·32`) |
| `eps_bits` | fp32 bits of `epsilon` |
| `tensor_rows_tiles` (`Rt`) | TILE: `prod(shape[:-2]) · ceil(H / 32)` (each image tile-padded independently); ROW_MAJOR: `ceil(prod(shape[:-1]) / 32)` (sticks are contiguous across images, no per-image padding) |
| `num_valid_sticks` | ROW_MAJOR only: `prod(shape[:-1])` (the last tile-row may hold < 32 valid sticks) |

## Tensors

### Input

| Property | Requirement |
|----------|-------------|
| Shape | rank ≥ 2, any leading dims; `[..., H, W]` |
| Dtype | bfloat16, float32 |
| Layout | TILE or ROW_MAJOR (native, no host conversion) |
| Memory | Phase 0: interleaved (DRAM/L1). Sharded = refinement (see Dataflow Strategy) |

### Gamma

| Property | Requirement |
|----------|-------------|
| Shape | `(1, 1, 1, W)` |
| Dtype | bfloat16 or float32 (independent of input dtype; its CB carries its own format) |
| Layout | ROW_MAJOR (Phase 0). TILE gamma = knob-turn refinement (reader reads tile pages directly; no row-0 build) |
| Memory | interleaved |

### Output

| Property | Value |
|----------|-------|
| Shape | identical to input |
| Dtype | input dtype |
| Layout | input layout |
| Memory | `memory_config` or DRAM interleaved |

## Blocking Model

The op is one independent axis (rows) and one dependent axis (W, the reduced axis). A block is a
2-D rectangle: `block_rows` tile-rows × the core's whole W slice. Phase 0 = regime **R1
row-resident**: each core owns a contiguous range of tile-rows, holds each block's full rows
resident in L1 (x crosses DRAM exactly once), reduces, then normalizes from L1.

### Axes

| Axis | Character (+ one-clause reason) | Extent knob | Phase 0 value | Knob source | Core-assignment | Later unlock |
|------|--------------------------------|-------------|---------------|-------------|-----------------|--------------|
| `rows` (all leading dims × H, in tile-rows) | **independent** — each row's result depends only on that row | `block_rows` (`r`) — tile-rows per block | coarsest that fits: `min(rows_per_core_max, r_max)` from the block-selection function below (whole per-core assignment whenever it fits) | host constant → compute/reader/writer CT arg `block_rows` | `ttnn.split_work_to_cores(grid, Rt, row_wise=True)`: contiguous tile-row ranges over the **full** compute grid (`device.compute_with_storage_grid_size()`, a host parameter, not a literal) | knob-turn (raise/lower `r`, change grid) |
| `w` (last dim, in tiles) — reduction view | **dependent** — `rstd[r]` spans the whole row | `core_w_tiles` — W slice owned by one core | `= tensor_w_tiles` (whole row; R1) | host constant → CT `core_w_tiles`, plus RT `w_tile_start = 0` | not split across cores in Phase 0 (R1); split + combine is R3 (`deferred`) | scheme-change (R3) |
| `w` — streaming view inside a block (normalize / gamma / output / RM staging) | dependent for the reduce, but the *post-rstd* phases are independent along W | `w_chunk_tiles` (`c`) — tile-columns per normalize/stream chunk; `num_w_chunks` (`K`) = `ceil(core_w_tiles / c)` | coarsest that fits: `K` = smallest chunk count that fits (usually 1 → `c = core_w_tiles`) | host constant → CT `w_chunk_tiles`, `num_w_chunks`; `row_stride_tiles = K·c` derived | n/a (within core) | knob-turn |
| `gamma` columns | **reuse-shared** — gamma varies only along W, so splitting rows across cores makes every core read identical gamma bytes (reuse-shared by construction of the row split) | streams with `w_chunk_tiles` | `c` per chunk | same as above | every core reads its own copy (stepping stone) | R4 multicast (`deferred`), R5 per-core resident (`deferred`) |

Per-axis knob table (Step 1.4 — one row per axis, no blanks):

| Axis | Character | Block factor (Phase 0) | Core assignment | Buffer depth | Why |
|---|---|---|---|---|---|
| rows | independent | `block_rows` = coarsest fit (whole core assignment when it fits) | split over full grid, row_wise | `cb_x` depth 1 block (rows resident across reduce→normalize); streaming windows depth 2 | rows are free to parallelize; residency of whole rows is what keeps x at one DRAM crossing |
| w (reduce) | dependent | whole row per core (`core_w_tiles = Wc`) | **not assigned across cores** in Phase 0 — every Phase 0 shape's row fits one core (predicate below); cross-core W split = R3 | `cb_sumsq` holds ≤ 2 tiles per row (accumulated), not Wc | splitting W needs a cross-core combine; Phase 0 shapes don't need it for residency |
| w (stream) | independent after rstd | `w_chunk_tiles` = whole row unless L1 forces chunks (floor 8 tiles) | n/a | `cb_out`/`cb_out_rm`/`cb_gamma`/`cb_x_rm` depth 2 (2·c) | chunking only to bound the per-chunk windows; whole tiles, ≥ 8 per chunk |
| gamma cols | reuse-shared | per chunk `c` | each core reads its copy, once per block | `cb_gamma` 2·c | `|gamma| ≤ |x per block| / (32·r)`; mcast is the deferred successor |
| *stage data:* `sumsq` partials (per row, ≤2 tiles) | indexed by rows → independent | `block_rows` rows per reduce call | same core as the rows (no designated collector core) | full block (`r·n_sq`) | intermediate of sequential helpers on one thread |
| *stage data:* `rstd` (1 tile per row) | independent (rows) | `block_rows` | same core; R1 has no coordinator role | full block (`r`) | consumed by every chunk of normalize |

No stage computes on one core while others wait: R1 has no collector/coordinator core at all.
(In R3 the group root also owns a W slice — "root works too", per catalog `tensix_all_reduce`.)

**Every knob is a parameter.** `block_rows`, `w_chunk_tiles`, `num_w_chunks`, `core_w_tiles`,
`w_tile_start`, the grid, `w_partial_elems`, buffer depths — each set once on the host and
passed down; every CB page count, loop trip count and per-core RT arg is derived from them. The
kernels loop `for (block_idx = 0; block_idx < num_blocks_this_core; ++block_idx)` with
`num_blocks_this_core = ceil(num_tile_rows_this_core / block_rows)` and
`rows_this_block = min(block_rows, remaining)`; no tensor dimension is inlined.

#### Block-selection function (pinned; host)

```
B          = per-core L1 bytes available to CBs (device query; interleaved tensors put nothing else in L1)
rows_cap   = max tile-rows assigned to any core (split_work_to_cores group 1)
c_floor    = min(core_w_tiles, 8)                       # granularity floor (whole tiles, ≥ 8: catalog)
F(r, K)    = footprint from l1_ledger.md with c = ceil(core_w_tiles / K), row_stride_tiles = K·c
R1 legal   ⇔ F(1, K_floor) ≤ B, where K_floor = ceil(core_w_tiles / c_floor)
r          = largest r in [1, rows_cap] with F(r, K_floor) ≤ B
K          = smallest K in [1, K_floor] with F(r, K) ≤ B          # → c = ceil(core_w_tiles / K)
```
`r` is maximized first because every per-block fixed cost (reduce init, `cb_x` handshake, the
gamma DRAM re-read) is paid once per block; `K` is then minimized so chunk-level inits are as few
as possible. If R1 is not legal the op raises `NotImplementedError("row too wide for single-core
residency; needs W-split regime R3")` — no Phase 0 SUPPORTED cell reaches it (worst case fp32
W=8192, fp32 gamma, ROW_MAJOR: ~1.1–1.3 MB at r=1, K=16..32, under the Wormhole/Blackhole CB budget).

### Buffer-depth knobs

| CB | Depth knob | Phase 0 value | What the depth buys |
|----|------------|---------------|---------------------|
| `cb_x` | `x_block_depth` | 1 block (`block_rows · row_stride_tiles` pages) | residency across reduce→normalize; intra-block overlap comes from per-chunk pushes + cumulative waits, not depth |
| `cb_x_rm` (RM) | `stream_depth` | 2 × `c` | reader fetches the next row-chunk of sticks while compute tilizes the current |
| `cb_gamma` | `stream_depth` | 2 × `c` | reader prefetches the next gamma chunk while compute scales the current |
| `cb_out` (TILE) / `cb_out_rm` (RM) | `stream_depth` | 2 × `c` | compute packs row-chunk i+1 while writer drains row-chunk i |
| `cb_sumsq`, `cb_rstd`, `cb_normed`, `cb_reduce_mask` | none (full block / constant) | — | sequential helpers on one thread cannot pipeline; full block is the contract |

`stream_depth` is one host constant (2) feeding all four streaming CBs.

### Mechanism caps

| Mechanism | Cap on which extent | Clamp | What happens unclamped |
|-----------|--------------------|-------|------------------------|
| `compute_kernel_lib::tilize` / `untilize` take `block_width_tiles` as a **template** param (`tilize_helpers.hpp:194-207`, `untilize_helpers.hpp:152-163`) | `w_chunk_tiles` must be compile-time and uniform | every chunk (incl. the last) runs at the nominal width `c`; rows are stored at `row_stride_tiles = K·c` (≤ `K−1` pad tile-columns per row, results dropped by the writer) | a narrower tail chunk would need a second instantiation and break the uniform CB quantum below |
| CB ring wrap — pointers reset only on an exact `fifo_limit` hit; multi-tile windows must not straddle it | every windowed CB's push/pop quantum | quanta are `c` tiles (one row-chunk) or 1 tile; every capacity is a multiple of its quantum; ragged last *block* only shortens `rows_this_block` (ring returns to base after each full block) | silent corruption of the wrapped window |
| `dataflow_kernel_lib::read_sticks_for_tilize` / `write_sticks_after_untilize` derive the L1 row stride from `row_bytes` (`tilize_helpers_dataflow.inl:92-93`, `:208-209`) | RM multi-chunk tail chunk (valid bytes < `c·32·elem`) | use the helpers for every chunk whose valid width is the full `c` (always the case when `K = 1`); the tail chunk of a `K > 1` RM row uses a hand-written read/write with stride `c·32·elem` and nominal push/pop `c` | stride mismatch with `tilize<c>` → garbled tail columns |
| `prepare_reduce_mask(valid_elems)` requires `valid_elems ∈ [1, 31]` (`reduce_helpers_dataflow.hpp:84-85`) | partial W tile | push the mask only when `w_partial_elems != 0`; reduce uses `ReducePartialScaler::only_partial()` only then | assert / wrong mask |
| Packer L1 accumulation is fp32-DEST only (catalog `row_reduce_accumulate`, `eltwise_l1_vs_dest_accumulate`) | sumsq mechanism vs `fp32_dest_acc_en` | Phase 0 gates `fp32_dest_acc_en=True`; the `bf16 + fp32_dest_acc_en=False` refinement must swap sumsq to a squared-chunk window + `reduce<…, AccumulateViaAdd>` with `Accumulate` | wrong sums |
| *Refinement 2 correction:* measured: packer L1 accumulation is correct with 16-bit DEST too (into Float32 or bf16 pages). The sumsq mechanism is unchanged for `fp32_dest_acc_en=False`; `cb_sumsq` stays Float32 so the L1 adds keep fp32 precision | — | no swap needed | — |
| FPU `ELWMUL` ignores `acc_to_dest` in its MOP (`tt_metal/tt-llk/tt_llk_blackhole/llk_lib/llk_math_eltwise_binary.h:160-191` — all ELWMUL MOPs pass `0 /*acc_to_dest*/`) | sum of squares cannot use `BinaryFpu<Mul, …, DestAccumulation::…>` | sumsq accumulates in **L1** via `PackTile<…, L1Accumulation::…>` instead | silently overwritten DEST → Σ = last tile only |
| AccumulateViaAdd partial path: BulkWaitBulkPop, ROW, NoAccumulation (`reduce_helpers_compute.hpp:138-163`) | rstd reduce | rstd reduce is one standalone call per block, `BulkWaitBulkPop` | static_assert / ASSERT |
| `TileAddressing::Strided` requires caller-managed `(None, None)` wait/pop (`chain.hpp:265-286`, `chain.inl` BinaryFpu asserts) | normalize reading `cb_x` | compute waits `cb_x` explicitly and pops it once per block | compile error |
| L1 CB budget `B` | `block_rows`, `w_chunk_tiles` | block-selection function above | allocation failure |

DEST capacity enters nowhere as a block bound: every chain clamps its internal block to
`DEST_AUTO_LIMIT` (`dest_helpers.hpp:103`), the reduce uses one DST per output, and no mechanism
needs N live accumulators.

### Regimes

| Regime | Status | Predicate | Block | Data movement vs. minimum (boundary: DRAM; minimum = x read once, out written once, gamma read once) | What a bigger block buys |
|--------|--------|-----------|-------|---------------------------|--------------------------|
| **R1 row-resident, row-split** | **built** (Phase 0) | interleaved input AND `F(1, K_floor) ≤ B` | `block_rows × row_stride_tiles` resident; normalize in `block_rows × w_chunk_tiles` chunks | x: 1 crossing (min); out: 1 (min); gamma: `Σ_cores num_blocks_this_core` crossings (above min by that factor; ≤ 1/(32·r) of each block's x bytes); no cross-core | per block: one rstd-reduce init, one `cb_x` wait/pop, one gamma pass (fewer gamma DRAM re-reads), one tilize init (RM); per chunk (bigger `c`): fewer normalize/scale/untilize inits + data-format reconfigs, fewer writer barriers |
| R3 W-split + cross-core combine (the block-sharded logical scheme) | **built (Refinement 4)** — WIDTH/BLOCK_SHARDED (shard grid = the split; row-group = shard row, root = slot 0) and interleaved rows that fail R1 residency (pinned predicate `not _r1_fits`; `G = grid.x`-wide groups, row-groups down y; `FORCE_W_SPLIT` test hook). Topology as in the "R3 Tensix-to-Tensix contract" below; stall shadow (x·γ during the rstd wait) not built — see changelog. Original rationale:  every Phase 0 SUPPORTED shape satisfies R1's predicate and R3 moves the same DRAM bytes plus combine traffic; its gains are occupancy when `Rt < grid` (decode `32×W`, `_WIDE` loose cases — which pin TILE gamma, outside Phase 0) and residency for rows that do not fit one core (`W > ~10K` fp32). It is the scheme WIDTH/BLOCK_SHARDED require; it is the next scheme-change refinement | `F(1,K_floor) > B` OR `Rt < num_cores / 2` (to be pinned when built) OR WIDTH/BLOCK sharded | per core: `block_rows × core_w_tiles` with `core_w_tiles = ceil(Wc / G)` for a W-group of G cores | x: 1 (min); out: 1; gamma: each core reads only its W slice → 1 total per row-group; cross-core: G·`block_rows` partial tiles gathered to the group root + `block_rows` rstd tiles multicast back, per block | per block: one gather + one multicast round per row-group, so larger `block_rows` amortizes the rendezvous |
| R4 gamma read-once + multicast | **deferred** — successor to R1's per-core gamma read (stepping stone). Saves `(Σ blocks − 1)·|gamma|` DRAM bytes, i.e. ≤ 1/(32·r) of x traffic (≤ 3% at r = 1, less at larger r), and replaces them with the same bytes on the NoC | `has_gamma` AND blocks-total ≫ 1 | as R1 | gamma: 1 DRAM crossing + mcast to all cores | — |
| R5 gamma per-core resident | **deferred** — Phase 0 typically assigns one block per core, where resident gamma reads gamma exactly as often as R1 does; differs only when `num_blocks_this_core > 1` | `F_resident(r,K) ≤ B` AND `num_blocks_this_core > 1` | as R1 + `cb_gamma` sized `core_w_tiles` | gamma: 1 per core | — |
| R6 HEIGHT_SHARDED in place | **built (Refinement 1)** — knob-turn on R1: shard = rows × full W already on the core; `cb_x` backed by the shard (`ttnn.cb_descriptor_from_sharded_tensor`, zero-copy), output CB backed by the output shard; block = whole shard | `memory_layout == HEIGHT_SHARDED` | whole shard (one block), `K = 1` required (shard width = `Wc`) | x: 0 DRAM; out: 0 DRAM; gamma per core | one block per shard = one reduce/normalize pass |
| R2 two-pass streaming (re-read x for normalize) | **rejected** — superseded by R3 for every shape in INPUTS and LOOSE_CASES (R3 keeps each slice resident and reads x once; R2 reads x twice). Dead end: no good scheme passes through it | — | — | x: 2 crossings | — |
| R0 single-core | **rejected** — superseded by R1 (same kernels, grid = 1) | — | — | — | — |

Regime-selection function (Phase 0): `memory_layout == INTERLEAVED and F(1, K_floor) ≤ B → R1`;
anything else → `NotImplementedError` from `validate()` / the builder.
**Refinement 4**: interleaved → R1 iff `F(1, K_floor) ≤ B`, else R3 (pinned; no grid/row-count term —
Refinement 5 owns the occupancy predicate); sharded with one W slice → R6; WIDTH/BLOCK with ≥ 2 slices → R3. Only one regime is built,
so there is no grid-dependent regime switch to pin yet; when R3 lands, its predicate must be
pinned exactly and regime-pinned tests added (a W-split that triggers only for some grid sizes
passes on one device and fails on another).

### Traffic ranking

| Rank | Split | DRAM crossings | Cross-core | Notes |
|------|-------|----------------|------------|-------|
| 1 | **rows across cores, whole row resident (R1)** | x 1, out 1, gamma once per block per core | none | chosen: minimum x/out traffic, no combine |
| 2 | rows + W across cores with combine (R3) | x 1, out 1, gamma 1 per row-group | G·r partial tiles + r rstd tiles per block | ties R1 on DRAM, adds a rendezvous; wins only on occupancy/residency → deferred |
| 3 | rows across cores, gamma mcast (R4) | x 1, out 1, gamma 1 | gamma bytes on NoC | ≤ 3% DRAM saving; deferred |
| 4 | rows across cores, x re-read (R2) | x 2, out 1 | none | rejected |

Operand-reuse check for the chosen split (rows): `x` varies along rows (no reuse); `gamma` does
**not** vary along rows → reuse-shared by construction → its broadcast regime is R4 (deferred).
`mask` is generated in L1 per core (no DRAM). `rstd` is per-row, produced locally.

Stall-shadow check:

| Stage that waits | Waits for | Work scheduled into the wait |
|------------------|-----------|------------------------------|
| `sumsq_block` (compute) | x tiles from the reader (DRAM latency) | sumsq consumes x chunk-by-chunk as each chunk lands (cumulative waits on `cb_x`; reader pushes per `c`-tile chunk), so the reduction runs under the read |
| `normalize_block` | `rstd` of the block | nothing independent in R1: rstd is produced on the same thread immediately before, no wait exists. The reader uses that window to prefetch gamma chunks (`cb_gamma` depth 2) |
| reader, next block | `cb_x` pop at end of current block (depth 1) | nothing in Phase 0 → perf lamp `x_block_depth = 2` |
| R3 (deferred): every core | combined rstd from the group root | `x·gamma` for the local slice into `cb_normed` (multiplication commutes: `x·rstd·γ = (x·γ)·rstd`), leaving one multiply by rstd after the multicast lands. Changes FP rounding order → gate on the precision baseline when built |

### Block schedule

```cpp
for (uint32_t block_idx = 0; block_idx < num_blocks_this_core; ++block_idx) {
    load_x_block(block_idx);       // reader (TILE) | reader + tilize_x_block (RM) -> cb_x
    sumsq_block(block_idx);        // per row: Σ_t x_t² accumulated in one L1 tile (+ masked tail tile)
    rstd_block(block_idx);         // one reduce over (rows_this_block × n_sq): ·1/W, +eps, rsqrt -> cb_rstd
    for (chunk = 0; chunk < num_w_chunks; ++chunk) {   // inner traversal owned by normalize_block
        normalize_chunk(block_idx, chunk);  // x · rstd(col-bcast)
        scale_chunk(block_idx, chunk);      // · gamma(row-bcast)          [has_gamma]
        untilize_chunk(block_idx, chunk);   // tiles -> sticks              [ROW_MAJOR]
        store_chunk(block_idx, chunk);      // writer
    }
    release_block(block_idx);      // pop cb_x (rows_this_block·row_stride_tiles) and cb_rstd (rows_this_block)
}
```

| Operation | Block shape | Resident across it | Intended init / sync frequency |
|-----------|-------------|--------------------|--------------------------------|
| `load_x_block` (TILE) | `rows_this_block × core_w_tiles` tiles (stored at stride `row_stride_tiles`) | — | reader: per row-chunk, ≤ `c` async tile reads then **one** barrier, one push of `c` pages (nominal, pad pages unwritten) |
| `load_x_block` + `tilize_x_block` (RM) | 32 sticks × `c·32` elems per row-chunk → `c` tiles | `cb_x` accumulates the block | reader: one barrier per row-chunk (32 stick reads); compute: tilize init once per block (`InitOnly` / `Neither` / `UninitOnly` across the block's calls), one `tilize<c>` per row-chunk |
| `sumsq_block` | per row: `n_full = Wc − (w_partial_elems ? 1 : 0)` tiles → 1 accumulated tile; + 1 squared tail tile if partial | `cb_x` (whole block, not popped) | mul init once per block (hoisted: `InitReconfigOwner::Caller` once, or one chain per row), one `cb_sumsq` push per row per output tile |
| `rstd_block` | `ReduceInputBlockShape::of(rows_this_block, n_sq)` | `cb_x`, mask | one reduce init per block; post-op SFPU inits per row (callback contract) |
| `normalize_chunk` | `grid(rows_this_block, c)` | `cb_x`, `cb_rstd` (both popped only in `release_block`) | one binary-bcast init per chunk; outputs pushed per row (`c` tiles) |
| `scale_chunk` | `grid(rows_this_block, c)` | `cb_rstd`, `cb_x`; `cb_gamma` chunk (popped at chunk end) | one bcast-row init per chunk |
| `untilize_chunk` | `untilize<c>` × `rows_this_block` | — | one untilize init per chunk; one `c`-tile push per row |
| `store_chunk` | per row: `c` tiles (TILE, write only cols `< Wc`) / 32 sticks × valid chunk bytes (RM, only valid sticks) | — | writer: ≤ `c` (TILE) or ≤ 32 (RM) async writes then one barrier per row-chunk |

### Perf lamps

| Lamp | Why the default may be wrong here | Nearby alternative to measure |
|------|-----------------------------------|-------------------------------|
| Overlap — `x_block_depth = 1` | with > 1 block per core the reader cannot fetch block b+1 until block b is popped | `x_block_depth = 2` with `block_rows` halved (same L1) |
| Overlap — whole assignment as one block | one block per core means no cross-block read/compute overlap beyond the per-chunk cumulative waits | two blocks of half height |
| Grid synchronization / occupancy | `Rt` small (decode `32×W`: 1 tile-row) leaves most of the grid idle under R1 | R3 (deferred regime) — not a knob |
| rstd reduce datapath | `AccumulateViaAdd` chosen for fp32 finalize accuracy; catalog `row_reduce_accumulate` measures the matmul `ReduceTile` path faster at 1–2 tiles per output | `ReduceAlgorithm::ReduceTile` with `[full, partial]` bf16 1.0 scalers (`with_partial()`), 1/W still applied in the post-op |
| gamma combine mechanism | L1 round-trip (`cb_normed`) chosen per catalog `compute_fusion` (FPU consumer → pack + unpack beats DEST reuse) | `DestReuseBinary<Mul, gamma, …>` fused into the normalize chain (needs gamma tiles replicated to 32 rows) — removes `cb_normed` and one init per chunk |
| `r`-first block selection | maximizing `block_rows` can force more W chunks (more chunk inits) on wide rows | `block_rows = 1` with `K = 1` where both fit |
| reconfig elision | sumsq/normalize/scale mix bf16 x, fp32 intermediates and output dtype; helpers reconfig at every boundary by default | catalog `compute_block_size` reconfig ablation on the all-same-format edges |
| NoC placement | `split_work_to_cores(row_wise=True)` per catalog `noc_placement` | `row_wise=False` |

Catalog entries that informed knobs: `width_split` (R3 row; W-split for few-row shapes),
`double_buffer` (stream depth 2, one barrier per ≥ 4–8 tile batch → `c_floor = 8`),
`compute_block_size` (coarse blocks, per-pass fixed cost), `row_reduce_accumulate` +
`eltwise_l1_vs_dest_accumulate` (packer L1 accumulation for Σx², fp32-only; pairwise add),
`reduce_block` (AccumulateViaAdd partial masked tail), `compute_fusion` (L1 round trip for the FPU
gamma multiply), `noc_placement` (row_wise), `tensix_all_reduce` (R3 topology: root reduce +
multicast back, root also works), `sfpu_tile_scope` (future lamp: rsqrt on column 0 only).

## Dataflow Strategy

| Stage | Format | Mechanism | Notes |
|-------|--------|-----------|-------|
| x DRAM → `cb_x` (TILE) | input dtype tiles | reader, `TensorAccessor`, tile page `t·Wc + w` for tile-row `t` | per row-chunk: ≤ `c` reads, 1 barrier, push `c` |
| x DRAM → `cb_x_rm` (RM) | input dtype sticks, L1 stride `c·32·elem` | reader, `read_sticks_for_tilize<cb_x_rm>(acc, rows, row_bytes, start_stick, chunk·c·32·elem)` | `rows` = 32 or the valid remainder of the last tile-row; chunk byte offset is a multiple of 64 B (DRAM-aligned); tail chunk of a K>1 row hand-written (cap table) |
| `cb_x_rm` → `cb_x` (RM) | tiles | compute `tilize<c>` | fp32: `Fp32Mode::Fast` (downstream consumers are FPU; tf32 either way) |
| reduce mask → `cb_reduce_mask` | bf16 0/1 tile | reader, `prepare_reduce_mask<cb_reduce_mask, REDUCE_ROW>(w_partial_elems)` once per kernel | only when `w_partial_elems != 0`; compute pops once at the end |
| gamma DRAM → `cb_gamma` | gamma dtype; tile row 0 valid | reader reads the contiguous gamma segment `[chunk·c·32, …)` (valid bytes only) into the reserved window, then lays it out **backwards** (last tile first) into each tile's row 0: elems 0–15 → face 0 row 0, 16–31 → face 1 row 0 | backward order makes the in-place relayout overlap-safe; rows 1–31 unused by row broadcast; push `c` (nominal) per chunk, every block |
| compute | fp32 intermediates (`cb_sumsq`, `cb_rstd`, `cb_normed`) | helpers (API Mapping) | fp32 DEST |
| `cb_out` → DRAM (TILE) | output dtype tiles | writer, tile pages, skips pad columns `≥ Wc` | |
| `cb_out_rm` → DRAM (RM) | output sticks | writer, `write_sticks_after_untilize<cb_out_rm>(acc, valid_rows, valid_chunk_bytes, start_stick, chunk byte offset)` | only valid sticks/bytes written |

Placement values of TARGET `memory_layout`, classified against this op's axis characters:

| `memory_layout` | Cuts | Character of cut axis | Class | Realization |
|-----------------|------|-----------------------|-------|-------------|
| INTERLEAVED | — | — | Phase 0 | R1 |
| HEIGHT_SHARDED | rows | independent | **knob-turn** (R6) | block = whole shard; `cb_x` and the output CB backed by the shards (`ttnn.cb_descriptor_from_sharded_tensor`), never re-read through a `TensorAccessor`; writer becomes a no-op on the output side |
| WIDTH_SHARDED | W | dependent | **scheme-change** (R3) | each core's slice is resident in place; partial Σx² per row combined across the width group |
| BLOCK_SHARDED | rows and W | rows independent, W dependent | **scheme-change** (R3 per grid row) | as WIDTH within each grid row |

R3 Tensix-to-Tensix contract (kept reachable): per row-group of G cores (one `ttnn.Mcast` channel
grouped along the W split): (1) every core runs `sumsq_block` on its slice and a local
`reduce<SUM, REDUCE_ROW, …, AccumulateViaAdd>` (Collapse, **no** post-op) into `cb_partial`
(`block_rows` col-0-valid tiles); (2) contributors write their `cb_partial` pages into the root's
`cb_gather` slot `contributor·block_rows` (uniform stride) and increment the root's semaphore;
(3) the root runs `reduce<SUM, REDUCE_ROW, …, AccumulateViaAdd, ReduceWithinTile::Skip>` over
`(block_rows, G)` with the same post-op (·1/W, +eps, rsqrt) into `cb_rstd`, then multicasts
`cb_rstd` to the group (`McastArgs::sender().send()` / `receiver().receive()`); (4) meanwhile every
core computes `x·γ` (stall shadow). Phase 0's structure keeps this reachable: the compute kernel
already takes `core_w_tiles` / `w_tile_start` as parameters, the reader already reads gamma by
chunk offset, and `rstd_block` is a separate block operation whose post-op only moves to the root.

**Refinement 6 — combine distribution.** Steps (2)–(3) are now a row-wise reduce-scatter
(`tensix_all_reduce`): with `q = ceil(block_rows / G)`, slot `o` owns rows `[o·q, (o+1)·q)`. Every core
sends owner `o`'s run of partials (faces 0/2) into `o`'s slot-major `cb_gather` (`G·q` pages) as soon as
that run is reduced — the per-run partial (`ReduceTile`, one Σx² tile per row) is interleaved into
`sumsq_block` so the all-to-all overlaps sumsq. Each owner combines its rows
(`reduce<SUM, REDUCE_COL, AccumulateViaAdd, Skip>` over `(G, q)` + post-op) and relays them to the root's
`cb_rstd_gather`; the root's single multicast (step 3) is unchanged. With one owner (`rows ≤ q`, e.g.
decode) this is exactly the R4/R5 root combine. Knobs: `R3_SCATTER_COMBINE`, `R3_PARTIAL_REDUCE_TILE`,
`R3_PARTIAL_INTERLEAVE`, `R3_SHADOW_RESIDENT_MIN_ROWS`.

## Work Distribution

| Field | Value |
|-------|-------|
| Work unit | one tile-row (32 rows × whole W); blocks of `block_rows` tile-rows |
| Grid | `device.compute_with_storage_grid_size()` (host parameter); active cores = `min(Rt, grid cores)` |
| Per-core work | `ttnn.split_work_to_cores(grid, Rt, row_wise=True)` → group 1 cores get `n+1` tile-rows, group 2 get `n`; RT args `start_tile_row`, `num_tile_rows` per core |
| Remainder | handled by the two split groups; per core the last block has `rows_this_block = num_tile_rows − (num_blocks−1)·block_rows`. Tile geometry is alignment-aware: TILE `Rt = prod(shape[:-2])·ceil(H/32)` (per-image), RM `Rt = ceil(num_valid_sticks/32)` with the last tile-row's valid sticks `num_valid_sticks − 32·(Rt−1)`; `Wc = ceil(W/32)` |
| CB sizing | uniform across cores, from `rows_cap` (the larger group) |

Regime: only R1 is built (see Regimes). Acceptance tests cover grid-independent behaviour;
R3 will require regime-pinned tests when added.

## Circular Buffers

Symbols: `r = block_rows`, `c = w_chunk_tiles`, `Wp = row_stride_tiles = num_w_chunks·c`,
`n_sq = (Wc − p > 0 ? 1 : 0) + p` with `p = (w_partial_elems != 0)`. Full ledger: `l1_ledger.md`.

| Semantic Name | Index | Page Size | Num Pages | Sizing rationale | Format | Producer | Consumer | Lifetime |
|---------------|-------|-----------|-----------|------------------|--------|----------|----------|-----------|
| `cb_x_rm` (RM only) | 0 | input tile size | `2·c` | streams rows & W chunks; spans one row-chunk | input dtype | reader | compute | load phase |
| `cb_x` | 1 | input tile size | `r·Wp` | spans rows (block) and W (whole row) | input dtype | reader (TILE) / compute-tilize (RM) — exactly one per layout variant | compute | load → release, per block |
| `cb_reduce_mask` (partial W only) | 2 | bf16 tile | 1 | constant | Float16_b | reader | compute | whole kernel |
| `cb_sumsq` | 3 | fp32 tile | shared allocation with `cb_normed` (see ledger); live `r·n_sq` | spans rows; W collapsed to ≤2 tiles | Float32 | compute | compute | sumsq → rstd |
| `cb_rstd` | 4 | fp32 tile | `r` | spans rows | Float32 | compute | compute | rstd → release |
| `cb_gamma` (gamma only) | 5 | gamma tile size | `2·c` | streams W chunks | gamma dtype | reader | compute | normalize phase, per chunk |
| `cb_normed` (gamma or RM) | 6 | fp32 tile | `cap_alias = round_up(r·max(c, n_sq), lcm(c, n_sq))` (shared with `cb_sumsq`) | spans rows × one chunk | Float32 | compute | compute | normalize → scale / untilize, per chunk |
| `cb_out` (TILE only) | 7 | output tile size | `2·c` | streams rows & chunks (quantum `c`) | output dtype | compute | writer | per row-chunk |
| `cb_out_rm` (RM only) | 8 | output tile size | `2·c` | streams rows & chunks (quantum `c`) | output dtype | compute | writer | per row-chunk |

CB sync (push = wait per CB): `cb_x` push `c` per row-chunk ⇄ compute waits cumulatively, pops
`rows_this_block·Wp` once; `cb_x_rm` push/pop `c` per row-chunk (tilize WaitBlock); `cb_sumsq`
push 1 per output tile ⇄ reduce waits/pops `n_sq` per row; `cb_rstd` push 1 per row ⇄ compute
waits `rows_this_block` once, pops once; `cb_gamma` push/pop `c` per chunk; `cb_normed` push `c`
per row ⇄ consumer per tile (scale) / per `c` (untilize); `cb_out`, `cb_out_rm` push/pop `c`
per row-chunk; `cb_reduce_mask` push 1 once / pop 1 once.

## Block Operation Realization

| # | Block operation | Block shape | Helper? | Input CB (semantic name, pages, state) | Output CB (semantic name, pages) | CB state after |
|---|-----------------|-------------|---------|----------------------------------------|----------------------------------|----------------|
| 0 | kernel prologue | — | `compute_kernel_hw_startup(cb_x, cb_x, cb_rstd)` first statement (later stages re-boot per the chain's per-stage rule only if required) | — | — | — |
| 1 | `tilize_x_block` (RM) | `rows_this_block × num_w_chunks` calls of `tilize<c>(1)` | yes | `cb_x_rm`, `c`, consumed | `cb_x`, `c` per call | `cb_x` holds the block |
| 2 | `sumsq_block` | per row: `IterationShape::tiles(n_full)` | yes (eltwise_chain) | `cb_x`, row base `i·Wp`, not popped | `cb_sumsq`, 1 tile (L1-accumulated) | — |
| 2b | `sumsq_tail` (partial W) | per row: `IterationShape::one_tile()` at tile `i·Wp + Wc − 1` | yes (`square` chain) | `cb_x`, not popped | `cb_sumsq`, 1 tile | per row `n_sq` tiles queued |
| 3 | `rstd_block` | `ReduceInputBlockShape::of(rows_this_block, n_sq)` | yes (reduce) | `cb_sumsq`, `rows_this_block·n_sq`, popped; `cb_reduce_mask` waited, not popped | `cb_rstd`, `rows_this_block` | `cb_rstd` persistent through block |
| 4 | `normalize_chunk` | `IterationShape::grid(rows_this_block, c)` | yes (eltwise_chain) | `cb_x` Strided `{chunk·c, Wp}`, not popped; `cb_rstd` Col-mapped, not popped | `cb_out` (TILE, no gamma) or `cb_normed`, `c` per row | — |
| 5 | `scale_chunk` (gamma) | `grid(rows_this_block, c)` | yes (eltwise_chain) | `cb_normed` per tile popped; `cb_gamma` Row-mapped `c`, popped at chunk end | TILE: `cb_out`; RM: `cb_normed` **in place** (per-tile pop/push credit) | RM: `cb_normed` holds scaled tiles |
| 6 | `untilize_chunk` (RM) | `untilize<c>(rows_this_block)` | yes | `cb_normed`, `c` per row, popped | `cb_out_rm`, `c` per row | — |
| 7 | `release_block` | — | raw `cb_pop_front` (no helper owns a caller-managed window) | `cb_x` `rows_this_block·Wp`, `cb_rstd` `rows_this_block` | — | empty, ring back at base |

## API Mapping

| Block operation | Type | Function | File:Line | Template Params / Args | Input CB | Output CB | Which params are block knobs |
|-----------------|------|----------|-----------|------------------------|----------|-----------|------------------------------|
| boot | raw_api | `compute_kernel_hw_startup` | required by `chain.hpp:24-40`, `reduce_helpers_compute.hpp:33-37` | `(cb_x, cb_x, cb_rstd)` once at `MAIN()` start | — | — | — |
| `tilize_x_block` | helper | `compute_kernel_lib::tilize` | `tilize_helpers.hpp:194-207` | `<w_chunk_tiles, cb_x_rm, cb_x, InitUninitMode::{InitOnly,Neither,UninitOnly}, WaitMode::WaitBlock, …, Fp32Mode::Fast>(in, out, 1)` | `cb_x_rm` | `cb_x` | `block_width_tiles = w_chunk_tiles` |
| `sumsq_block` | helper | `eltwise_chain` + `BinaryFpu<Mul>` + `PackTile` | `chain.hpp:568`, `:588`, `:620`; L1 acc `chain.hpp:297-304`, lifecycle rule `chain.inl:1131-1135` | `BinaryFpu<Mul, input(cb_x, WaitPolicy::Cumulative (or None), PopPolicy::None, Block, Enabled, Offset), same>{x, x, i·Wp, i·Wp}`; `PackTile<output(cb_sumsq, OneUpfront, OneAtEnd, Enabled, Direct, DestAccumulation::Disabled, L1Accumulation::Enabled)>` — or caller-managed `(None, None)` + `L1Accumulation::AddToExisting` per chunk if Cumulative+Offset is rejected | `cb_x` | `cb_sumsq` | shape `tiles(n_full)` |
| `sumsq_tail` | helper | `compute_kernel_lib::square` | `eltwise/api/convenience.hpp:80`, body `convenience.inl:33-38` | input `cb_x` `(None, None, Offset)` base `i·Wp+Wc−1`; output `cb_sumsq` per tile | `cb_x` | `cb_sumsq` | — |
| `rstd_block` | helper | `compute_kernel_lib::reduce` | `reduce_helpers_compute.hpp:673` (decl), algorithm `:165`, partial `:350` | `<SUM, REDUCE_ROW, cb_sumsq, cb_reduce_mask, cb_rstd, ReduceInputPolicy::BulkWaitBulkPop, INPUT_AND_OUTPUT, ReduceFp32Mode::Fast, ReduceAlgorithm::AccumulateViaAdd, ReduceWithinTile::Collapse>(sumsq, mask, rstd, of(rows_this_block, n_sq), contiguous(), NoAccumulation{}, post_op, p ? only_partial() : none())` | `cb_sumsq`, `cb_reduce_mask` | `cb_rstd` | `rows` = `rows_this_block` |
| `rstd_block` post-op | raw_api (inside the helper's documented `post_reduce_op` callback, `reduce_helpers_compute.hpp:548-553,579`) | `binop_with_scalar_tile_init`, `mul_unary_tile(d, inv_w_bits)`, `add_unary_tile(d, eps_bits)`, `rsqrt_tile_init`, `rsqrt_tile(d)` | `tt_metal/hw/inc/api/compute/eltwise_unary/binop_with_scalar.h:61,36,157`, `rsqrt.h:21,45` | `dst_idx` from the helper | DEST | DEST | — |
| `normalize_chunk` | helper | `eltwise_chain` + `BinaryFpu<Mul, …, BroadcastDim::Col>` + `PackTile` | `chain.hpp:568`, `:330-335`, `:588`, `:620` | A = `input(cb_x, None, None, Block, Enabled, Strided)` with `StridedTileRange{chunk·c, Wp}` (`chain.hpp:288`); B = `input(cb_rstd, BroadcastDim::Col, None, None, Col)`; out `output(cb_normed | cb_out, PerOuter, PerOuter)` | `cb_x`, `cb_rstd` | `cb_normed` / `cb_out` | shape `grid(rows_this_block, c)` |
| `scale_chunk` | helper | `eltwise_chain` + `BinaryFpu<Mul, …, BroadcastDim::Row>` + `PackTile` (convenience `mul` with Row bcast where it expresses the policies) | `chain.hpp:568`, `convenience.hpp:70` | A = `input(cb_normed, PerTile, PerTile, Block)`; B = `input(cb_gamma, BroadcastDim::Row, Upfront, AtEnd, Row)`; out `cb_out` (TILE) / `cb_normed` in place (RM), per tile | `cb_normed`, `cb_gamma` | `cb_out` / `cb_normed` | shape `grid(rows_this_block, c)` |
| `untilize_chunk` | helper | `compute_kernel_lib::untilize` | `untilize_helpers.hpp:152-163` | `<w_chunk_tiles, cb_normed, cb_out_rm>(in, out, rows_this_block)` | `cb_normed` | `cb_out_rm` | `block_width_tiles = w_chunk_tiles` |
| `release_block` | raw_api | `cb_pop_front` (via the kernel's `DataflowBuffer`) | — | `rows_this_block·Wp`, `rows_this_block` | `cb_x`, `cb_rstd` | — | — |
| reader mask | helper | `dataflow_kernel_lib::prepare_reduce_mask<cb_reduce_mask, REDUCE_ROW>` | `reduce_helpers_dataflow.hpp:84-85` | `(w_partial_elems)` | — | `cb_reduce_mask` | — |
| reader RM sticks | helper | `dataflow_kernel_lib::read_sticks_for_tilize<cb_x_rm>` | `tilize_helpers_dataflow.hpp:88` | `(acc, rows, row_bytes, start_stick, chunk·c·32·elem)` | DRAM | `cb_x_rm` | `row_bytes = c·32·elem` |
| writer RM sticks | helper | `dataflow_kernel_lib::write_sticks_after_untilize<cb_out_rm>` | `tilize_helpers_dataflow.hpp:130` | `(acc, rows, row_bytes, start_stick, chunk byte offset)` | `cb_out_rm` | DRAM | `row_bytes = c·32·elem` |
| reader/writer RM tail chunk (K > 1) | raw_api | `noc_async_read` / `noc_async_write` + nominal `cb_reserve_back/push_back(c)` | — | stride `c·32·elem`, valid bytes only | DRAM / `cb_out_rm` | `cb_x_rm` / DRAM | — |
| reader TILE x / writer TILE out / reader gamma | raw_api | `TensorAccessor` + `noc_async_read/write` + one barrier per row-chunk | `tech_reports/tensor_accessor/tensor_accessor.md` | page ids from tile-row and column | DRAM / CBs | CBs / DRAM | `c` per batch |

Helpers considered and rejected:

| Raw use | Candidate | File:line of mismatch | Reason |
|---------|-----------|-----------------------|--------|
| post-op SFPU calls | chain `MulUnary`/`AddUnary`/`Rsqrt` structs (`eltwise/unary/scalar.hpp:26-33`, `math.hpp:38`) | `reduce_helpers_compute.hpp:673-681` — `post_reduce_op` is a callable on a DEST index, not a chain | the chain owns its own dst-sync window and CB lifecycle; inside the reduce's window only raw DEST ops are legal |
| sumsq via `BinaryFpu<Mul, …, DestAccumulation::WholeShape>` | DEST accumulation | `llk_math_eltwise_binary.h:160-191` (ELWMUL MOPs hard-code `acc_to_dest = 0`) | would overwrite instead of accumulate; replaced by packer L1 accumulation (still a helper) |
| sumsq via `square` into a Wc-wide x² CB + `reduce` | full x² materialization | — | an extra `r·Wc` fp32 buffer (1 MB at W=8192); L1 accumulation folds it to 1 tile/row |
| RM tail chunk | `read_sticks_for_tilize` / `write_sticks_after_untilize` | `tilize_helpers_dataflow.inl:92-93`, `:208-209` | stride derived from `row_bytes`; the tail needs stride `c·32·elem` with fewer valid bytes |
| gamma row-0 build | `read_sticks_for_tilize` + `tilize` | — | a compute tilize per chunk per block plus a staging CB, to produce one valid row; the reader relayout is two small local copies per tile |
| `release_block` pops | — | caller-managed `(None, None)` windows (`chain.hpp:265-286`) | the windows span several helper calls; no helper owns them |

## Broadcast Verification

| Phase | Op | CB_A (semantic name) Valid Region | CB_B (semantic name) Valid Region | Broadcast Dim |
|-------|-----|-----------------------------------|-----------------------------------|---------------|
| sumsq | mul (x·x) | `cb_x` All | `cb_x` All | None |
| rstd partial fold (inside reduce) | masked mul of tail x² | `cb_sumsq` All | `cb_reduce_mask` Row0 | Row (helper-internal) |
| normalize | mul | `cb_x` All | `cb_rstd` Col0 (REDUCE_ROW output) | **Col** |
| scale | mul | `cb_normed` All | `cb_gamma` Row0 (1-D `[W]`) | **Row** |

## Key Risks and Gotchas

| Risk | Why it bites here | Mitigation in this design |
|------|-------------------|---------------------------|
| Padding folded into Σx² | TILE padding can hold anything (golden poison = 1000); RM tail columns of `cb_x` are stale L1 | the last W tile is excluded from the L1-accumulated sum and folded in only through the reduce's 0/1 mask (`only_partial`); `inv_w_bits = 1/W` uses logical W |
| Wrong divisor | `1/(Wc·32)` instead of `1/W` passes aligned shapes only | single host constant `inv_w_bits` from logical W; acceptance tests include W = 50, 17, 100 |
| ELWMUL + DEST accumulation | looks legal in the chain API, silently overwrites | L1 accumulation via `PackTile` (cap table) |
| In-place scale on `cb_normed` (RM+gamma) | rule-3.2 hazard: a credit-less in-place pass lets untilize read pre-scale tiles | scale uses real per-tile wait/pop + reserve/push on `cb_normed`; never `(None, None)` there |
| Ring wrap with ragged chunks / blocks | multi-tile windows straddling `fifo_limit` corrupt silently | nominal chunk width `c` everywhere (row stride `Wp = K·c`), quanta `c` or 1 tile, capacities multiples of quanta; `cb_sumsq`/`cb_normed` alias capacity `round_up(r·max(c,n_sq), lcm(c,n_sq))` |
| Alias `cb_sumsq` ↔ `cb_normed` | two CB indices over one allocation | lifetimes are disjoint on the single compute thread (rstd reduce fully unpacks `cb_sumsq` before normalize packs `cb_normed`); declared as one `CBDescriptor` with two format descriptors |
| RM last tile-row | `< 32` valid sticks; rows from different images share a tile-row | reader reads/writer writes only `num_valid_sticks` sticks; compute processes full tiles (garbage rows are row-independent) |
| H-non-aligned TILE | padding rows produce garbage rstd/out rows | row-independent math; padding rows are never read back logically |
| fp32 precision | FPU srcA/srcB are tf32; L1 accumulation needs fp32 DEST | Phase 0 gates `fp32_dest_acc_en=True`; `float32 + False` is an EXCLUSION; tolerance from `helpers.TOLERANCES` |
| gamma relayout overlap | reading the segment into the reserved window and spreading it to row 0 of each tile overwrites unread source bytes if done forwards | relayout backwards (last tile first) |
| gamma different dtype from input | mixed srcA/srcB formats | `cb_gamma` carries gamma dtype; chains reconfig per operand |
| Wide-row residency | fp32 W=8192 + fp32 gamma + RM is the tightest Phase 0 cell | block-selection function shrinks `c` (K up to `Wc/8`); predicate failure raises `NotImplementedError` naming R3 |
| One dispatch | multi-phase op tempts sequencing ttnn ops | everything inside one `ttnn.generic_op` ProgramDescriptor; no host-side to_layout/pad/slice |

### Phase 0 SUPPORTED rectangle (for the implementer's `SUPPORTED` block)

| Axis | Phase 0 values | Refinement candidates (TARGET − Phase 0) |
|------|----------------|------------------------------------------|
| `dtype` | float32, bfloat16 | bfloat8_b (TILE only) |
| `fp32_dest_acc_en` | True | False (bf16; needs sumsq mechanism swap — cap table); `{float32, False}` is an EXCLUSION |
| `layout` | TILE, ROW_MAJOR | — |
| `alignment` | tile_aligned, w_non_aligned, h_non_aligned | — |
| `rank` | 2, 3, 4 | — |
| `gamma_mode` | gamma, no_gamma | — |
| `gamma_dtype` | float32, bfloat16, "none" | bfloat8_b |
| `gamma_layout` | ROW_MAJOR, "none" | TILE (knob-turn: read tile pages, skip row-0 build) |
| `memory_layout` | INTERLEAVED | HEIGHT_SHARDED (knob-turn, R6), WIDTH/BLOCK_SHARDED (scheme-change, R3) |

INPUT_TAGGERS: `tag_alignment` (three-valued per feature_spec docstring), `tag_rank`,
`tag_gamma_dtype` / `tag_gamma_layout` returning `"none"` when gamma is None. Validation raises
`ValueError` for rank < 2 and gamma last-dim mismatch; `NotImplementedError` for cells outside
SUPPORTED. `default_compute_kernel_config()` is exported and is the only place the default lives.

### Structural impossibilities

None beyond `feature_spec.INVALID`.
