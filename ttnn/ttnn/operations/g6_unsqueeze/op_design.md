# Operation Design: g6_unsqueeze

## Overview

| Field | Value |
|-------|-------|
| Classification | data_movement — no compute kernel, no arithmetic, no TRISC binary |
| Goal | Return `torch.unsqueeze(x, r)` for a one-row float32 TILE input `(1,…,1,W)` as a new DRAM-interleaved TILE tensor `(1,…,1,W,1)`, bit for bit, in one device program that moves row 0 of every input tile into column 0 of the matching output tile. |
| Math | `out[i_0,…,i_{r-1},0] = x[i_0,…,i_{r-1}]`. On tiles, with `w = 32·t + c`, `h = c div 16`, `j = c mod 16`: output page `t`, face `2h`, row `j`, column 0 ← input page `t`, face `h`, row 0, column `j`. |
| References | Spec of this op (rules BB-01…BB-22); `eval/golden_tests/g6_unsqueeze/{feature_spec,helpers,test_regression}.py`; `.claude/references/{blocking-model,l1-footprint-discipline,op-design-template}.md`; catalog `ttnn/ttnn/operations/examples/master.md` entries `width_split`, `double_buffer`, `trid_double_issue`, `tile_reorder`, `split_reader`, `noc_placement`, `bank_stagger`; `ttnn/cpp/ttnn/kernel_lib/tile_column_dataflow.hpp`; `tt_metal/hw/inc/api/dataflow/dataflow_api.h`; `tt_metal/hw/inc/internal/tt-1xx/blackhole/noc/noc_parameters.h:375-380`. The bundle `docs/explorations/g6_unsqueeze/` is absent; designed from the spec. |

## Parameters

| Name | Type | Required | Valid Range | Default | CT/RT |
|------|------|----------|-------------|---------|-------|
| `input_tensor` | `ttnn.Tensor` | yes | on device; rank 1–3; every dim but the last = 1; `W = shape[-1] ≥ 1`; float32; TILE; DRAM interleaved | — | host; its buffer address is per-call data |
| `dim` | `int`, positional or `dim=` | yes | `-(r+1) ≤ dim ≤ r` and canonical `k = dim if dim ≥ 0 else dim + r + 1` equals `r` (so `dim ∈ {r, -1}`) | none | host only — both spellings are the same move, so `dim` never reaches a kernel and is not in the program-cache key |

### Registry contract (op file)

| Item | Value |
|------|-------|
| `INPUT_TAGGERS` | `{"dtype": tag_dtype, "layout": tag_layout}`. `tag_dtype(inputs, axes) → inputs[0]["dtype"]`; `tag_layout(inputs, axes) → inputs[0]["layout"]`. `inputs[0]` is the scenario dict `{"input_shape", "dim", "dtype", "layout"}`; axis names are TARGET's. |
| `SUPPORTED` | `{"dtype": [ttnn.float32], "layout": [ttnn.TILE_LAYOUT]}` — equal to TARGET |
| `EXCLUSIONS` | `[]`. No alignment or size axis, no tagger that gates shapes: the 1-row input tiles and 1-column output tiles are the shape class itself. |
| `validate(input_tensor, dim)` | First line of `g6_unsqueeze`. Order: (1) SUPPORTED per axis from `input_tensor.dtype` / `input_tensor.layout` → `UnsupportedAxisValue`; (2) EXCLUSIONS → `ExcludedCell` (none); (3) the request checks below → `ValueError`. Everything runs before allocation and dispatch, so a refusal builds no program. |
| INVALID | not declared, not checked (`feature_spec.INVALID == []`) |

| Malformed request | Host check, in this order | Raises |
|-------------------|---------------------------|--------|
| tensor not on device | storage is not device storage | `ValueError` |
| rank outside 1–3 | `len(shape) not in (1, 2, 3)` | `ValueError` |
| no elements | any dim is 0 | `ValueError` |
| more than one logical row | any of `shape[:-1]` is not 1 | `ValueError` |
| not DRAM interleaved | `memory_config().buffer_type != DRAM` or `memory_layout != INTERLEAVED` | `ValueError` |
| `dim` not an integer | `operator.index(dim)` raises | `ValueError` |
| `dim` out of range | `dim < -(r+1)` or `dim > r` | `ValueError` |
| `dim` not the last position | canonical `k != r` | `ValueError` |

Every refusal message starts with `g6_unsqueeze:`; the acceptance test matches on it.

## Tensors

### Input

| Property | Requirement |
|----------|-------------|
| Shape | `(W,)`, `(1, W)` or `(1, 1, W)`, any `W ≥ 1`; production `(1, 512)` and `(1, 256)` |
| Dtype | float32 |
| Layout | TILE (32×32, four 16×16 faces, face-major, row-major inside a face) |
| Memory | DRAM interleaved |

### Output

| Property | Value |
|----------|-------|
| Shape | `input.shape + (1,)`: `(W, 1)`, `(1, W, 1)`, `(1, 1, W, 1)` |
| Dtype | input dtype (float32) — no cast |
| Layout | input layout (TILE) |
| Memory | input memory config (DRAM interleaved). A new buffer from `ttnn.allocate_tensor_on_device`; the only device allocation of the call. |

### Tile geometry — the move in pages

Constants (single source, never restated as literals): `tile_w = tile_h = 32`; `face_w = face_h = 16` (`tt_metal/hw/inc/api/tile_constants.h:21-23`); `elem_bytes = input.element_size()` (4). Derived: `face_bytes = face_h·face_w·elem_bytes` (1024), `face_row_bytes = face_w·elem_bytes` (64), `faces_per_face_row = tile_w / face_w` (2), `tile_halves = tile_w / face_w` (2).

| Quantity | Formula | p1 `(1,512)` | p2 `(1,256)` |
|----------|---------|--------------|--------------|
| `tensor_w_tiles` | `ceil(W / tile_w)` — per image; every leading dim is 1, so one image. Host asserts it equals `input.buffer_num_pages()` and `output.buffer_num_pages()`. | 16 | 8 |
| Input page `t` | tile column `t`; row 0 holds `w = 32t + c`, `c ∈ [0, 32)`; rows 1–31 are padding | | |
| Output page `t` | tile row `t`; column 0 holds `w = 32t + c` in row `c`; columns 1–31 are padding; in the last page, rows `≥ W − 32t` are padding | | |
| Value `c = 16h + j` | input byte `h·face_bytes + j·elem_bytes` → output byte `h·faces_per_face_row·face_bytes + j·face_row_bytes` | | |

Each tile half `h` is a self-contained transfer: the 64-B row 0 of input face `h` becomes column 0 of output face `2h` (1024 B). Input rows 1–31 and output faces 1 and 3 are padding: they are neither read nor written. In the last tile, input row-0 padding columns (`c ≥ W − 32t`) land in output rows of the same index, which are output padding, so no padding value reaches a logical output position.

## Blocking Model

### Axes

| Axis | Character (+ one-clause reason) | Extent knob | Phase 0 value | Knob source | Core-assignment | Later unlock |
|------|--------------------------------|-------------|---------------|-------------|-----------------|--------------|
| `lead` — leading dims `d_0…d_{r-2}` of the input (its one logical row) | independent — a row moves on its own | `block_lead` | 1, the whole tensor extent: the contract fixes it at 1 (`validate` refuses more rows), so there is nothing to block | derived: `prod(shape[:-1])` | not split (extent 1) | scheme-change: more rows is "another movement on tiles" per the spec and outside TARGET |
| `w_tile` — tile columns of the input = tile rows of the output, extent `tensor_w_tiles` | independent — output page `t` depends on input page `t` only | `block_w_tiles` | `min(max_core_w_tiles, block_w_tiles_cap)` = 1 for p1 and p2. Occupancy first gives each core one tile; the block then takes the whole per-core assignment. It is not a collapsed knob: it rises with tiles per core up to the L1 cap (658). | host plan function (once per cache key), threaded as a compile-time constant of both movers; CB capacity and block-loop bounds are derived from it | contiguous page ranges over `num_cores = min(tensor_w_tiles, max_mover_cores)` cores in one rectangle (Work Distribution) | knob-turn (`max_mover_cores`, `block_w_tiles`) |
| `tile_half` — `h = c div 16`: input face `h` row 0 ↔ output face `2h` column 0, extent `tile_halves` = 2 | independent — the two halves share no byte of either tensor | `block_tile_halves` | 1 per mover: each core's two halves are split across its two data-movement RISCs (`movers_per_core = 2`), so a mover's block is its whole half-assignment | derived: `tile_halves / movers_per_core`; `movers_per_core` is a host constant | across the two data-movement RISCs of every mover core (half 0 → RISCV_1 on NoC0, half 1 → RISCV_0 on NoC1), not across cores | knob-turn (`movers_per_core = 1`: one mover takes both halves) |
| `face_lane` — `j = c mod 16`, the 16 values of one face row | independent per value, but its 64 B are one DRAM read granule | `block_face_lanes` | 16, the whole face row: any cut re-reads the same 64-B granule (`NOC_DRAM_READ_ALIGNMENT_BYTES = 64`, noc_parameters.h:379) | derived: `face_w` | not split | none — a split adds only re-reads |
| `new_axis` — the appended output dim (extent 1, output column 0) | independent — extent 1, no work along it | `block_new_axis` | 1, fixed by the op's definition | op definition | not split | none |

Intermediate data. The scheme has one intermediate: the staged face slot of a `(w_tile, tile_half)` pair (16 values). Its axes are `w_tile`, `tile_half` and `face_lane`, assigned as above. The RISC that gathers a half also scatters and stores it. No core or RISC collects, coordinates or hands out work, and no stage computes on one core while others wait.

### Buffer-depth knobs

| CB | Depth knob | Phase 0 value | What the depth buys |
|----|------------|---------------|---------------------|
| `cb_mover0_face_slots`, `cb_mover1_face_slots` | `staging_depth` (blocks resident per mover; host constant) | 1 | Depth 2 lets the gather of block `i+1` land while block `i` is scattered and stored. That matters only when `num_blocks_this_core > 1`, i.e. `max_core_w_tiles > block_w_tiles_cap` (`tensor_w_tiles > 658·num_cores`), which no golden shape reaches. Production has one block per core: depth 2 would buy nothing and double the L1. Overlap lamp below. |

### Mechanism caps

| Mechanism | Cap on which extent | Clamp | What happens unclamped |
|-----------|--------------------|-------|------------------------|
| DRAM read alignment 64 B (`NOC_DRAM_READ_ALIGNMENT_BYTES`, `noc_parameters.h:379`) | gather granule `face_row_bytes`; source = page + `h·face_bytes`; destination = slot base | float32: `face_row_bytes = 64` exactly and slot stride `face_bytes = 1024` (a multiple of 64); CB bases are DRAM-aligned by the program allocator (`tt_metal/impl/program/program.cpp:2505-2509` passes the DRAM alignment). Host asserts `face_row_bytes % dram_alignment == 0` and `face_bytes % dram_alignment == 0`. | bytes land shifted inside the slot: wrong values, no error |
| L1 left to the op (BB-05: 1347776 B per core) | `block_w_tiles` | `block_w_tiles_cap = floor(op_l1_budget_bytes / (movers_per_core · staging_depth · block_tile_halves · face_bytes))` = 658 at Phase 0; `block_w_tiles = min(max_core_w_tiles, block_w_tiles_cap)`; more tiles per core become more blocks (W has no upper limit) | CB allocation beyond the budget: a refused launch, or overlap with the model's L1 tensors |
| NoC single-packet size (`NOC_MAX_BURST_SIZE`, `noc_parameters.h:290`) | face write of `face_bytes` | none needed: 1024 B is below the burst size, so one packet per face | n/a |
| In-place row → column move inside one face slot | face geometry | needs `face_h == face_w` and destinations `j·face_w` (`j ≥ 1`) disjoint from sources `1…face_w−1`; true for 16×16 faces. The mover `static_assert`s it. | sources overwritten before they are read: wrong values |
| Blackhole RISC-V L1 cache (write-through, 4×16 B lines; `tt_metal/hw/inc/internal/tt-1xx/cache.h:7-15`) | when the scatter of a slot may start | scatter only after a read barrier, which fences (`noc_async_read_barrier`, `dataflow_api.h:1932`); a trid-scoped variant must call `invalidate_l1_cache()` itself | stale words from an earlier block: wrong values that differ between runs (BB-06) |

### Regimes

| Regime | Status | Predicate | Block | Data movement vs. minimum | What a bigger block buys |
|--------|--------|-----------|-------|---------------------------|--------------------------|
| `half_split_face_scatter` | **built** | every input `validate()` accepts | per mover: `block_w_tiles × block_tile_halves × block_face_lanes` (Phase 0: 1 × 1 × 16); per core: both halves | **Minimum at the DRAM boundary**: each logical input value crosses once, inside its 64-B face row (2 × 64 B per tile; the read granule is the face row, so no padding row is read). Each output face that holds a value crosses once (2 × 1024 B per tile): a face is the smallest single-transaction write that holds its 16 column-0 rows (see `column_strip_writes`). No cross-core traffic. Input rows 1–31 and output faces 1 and 3 never cross. p1: 2048 B read + 32768 B written in 64 transactions; p2 half of that. | Per block: one read barrier (one exposed DRAM read round trip), and with more than one block, one write flush before slot reuse. Per kernel run: accessor setup and one write barrier. Production has one block per core, so each is paid once per call. Domain: all accepted inputs. No other regime applies: the op has no dependent axis (no combine) and no reuse-shared operand (no multicast); every other row moves more bytes or transactions for the same result. |
| `halves_across_cores` | rejected — superseded by `half_split_face_scatter` | — | 1 tile half per core | same bytes and transactions | Same per-RISC path (one 64-B read, 15 word moves, one 1024-B write) on `2·tensor_w_tiles` cores (32 for p1). That doubles per-core runtime args and core ranges, while the built regime uses the idle second data-movement RISC of each core. |
| `whole_tile_move` | rejected (dead end) — superseded by `half_split_face_scatter` everywhere | — | whole 4096-B pages | input 32× (padding rows included, so poisoned padding enters L1); output 2× | none: same transaction count per direction, and at DRAM-bound W about 4× the time |
| `compute_engine_transpose` (reader → unpack/transpose → DEST → pack → writer) | rejected — superseded by `half_split_face_scatter` | — | whole tiles | input 32×, output 2× | n/a. float32 through the FPU operand registers is 19-bit (spec rule 1). A bit-exact route needs unpack-to-DEST, a DEST transpose and three TRISC binaries. |
| `single_core_mover` | rejected (dead end) — Phase 0 is multi-core | — | all tiles on one core | same bytes | n/a: all `2·tensor_w_tiles` reads, `30·tensor_w_tiles` word moves and all writes serialize on one core |
| `column_strip_writes` (16 B per output row holding column 0) | rejected — superseded by `half_split_face_scatter` | — | per output row | output 512 B instead of 2048 B per tile, but 32 write transactions per tile instead of 2 | n/a: transaction-bound at every golden shape |
| `role_split_pipeline` (reader RISC on NoC0 gathers both halves; writer RISC on NoC1 scatters and stores; CB credit hand-off) | deferred — no golden or production shape reaches its domain. Every case is ≤ 16 tiles at 1 tile per core, so latency-bound. The catalog's NoC-direction gain (`noc_placement`, 2.5–4.8×) was measured on bandwidth-bound grid-filling copies. | `tensor_w_tiles ≫ num_cores` (DRAM-bandwidth-bound) | per core: `block_w_tiles × tile_halves × block_face_lanes` | same bytes; changes only which NoC carries reads vs. writes | Reachable: `gather_rows_block` and `scatter_store_block` keep their contracts. Only their RISC assignment moves, plus a CB hand-off between them. |
| `staged_or_viewed` (via a DRAM or host copy, or the input pages returned under the new shape) | rejected — forbidden by the contract | — | — | 2× DRAM traffic, or values left in rows | n/a |

### Traffic ranking

Candidate splits, ranked by DRAM crossings, then cross-core traffic, then core-local work. Counts are for p1; p2 is half.

| Rank | Split | DRAM in | DRAM out | Transactions | Cross-core | Core-local |
|------|-------|---------|----------|--------------|------------|------------|
| 1 (chosen) | `w_tile` across cores × `tile_half` across each core's 2 data-movement RISCs | 2048 B | 32768 B | 32 R + 32 W | none | 15 word moves per (tile, half), two RISCs in parallel |
| 1 (tie on bytes) | `w_tile` × `tile_half`, both across cores | 2048 B | 32768 B | 32 R + 32 W | none | same; twice the cores |
| 1 (tie on bytes) | `w_tile` across cores, both halves on one RISC | 2048 B | 32768 B | 32 R + 32 W | none | 30 word moves per tile on one RISC |
| 4 | `face_lane` across cores | up to 16× (each 64-B granule read by every lane owner) | 16 writes of 64 B per face | ≥ 16× | none | — |
| n/a | dependent axis + cross-core combine | no axis is dependent: each output value depends on exactly one input value | | | | |
| n/a | broadcast of a reuse-shared operand | operand-reuse check against the chosen split: the only input varies along `w_tile` (page `t` feeds page `t` only) and along `tile_half`, so no core re-reads another core's bytes | | | | |

Chosen: rank 1 with halves on the two RISCs. Among the three equal-byte splits, the RISC split takes parallelism the core already has (its second data-movement RISC is otherwise idle) and adds no cores, core ranges or runtime args. That occupancy argument is kept apart from the byte ranking; the grid-synchronization and RISC-split lamps below measure it.

### Block schedule

```cpp
// One mover per data-movement RISC of every mover core. Mover m owns tile halves
// [m·block_tile_halves, (m+1)·block_tile_halves) of the core's pages [first_tile, first_tile + core_w_tiles).
for (uint32_t block_idx = 0; block_idx < num_blocks_this_core; ++block_idx) {
    gather_rows_block(block_idx);    // DRAM → face slots (row 0 of each slot)
    scatter_store_block(block_idx);  // each slot: row 0 → column 0 in place, then slot → DRAM
}
drain_writes();
```

| Block operation | Block shape it acts on | Resident across it | Intended frequency of fixed costs |
|-----------------|------------------------|--------------------|-----------------------------------|
| `gather_rows_block` | `block_w_tiles × block_tile_halves` slots. One `face_row_bytes` read per (page, half), from input page `t` at `h·face_bytes` into row 0 of the pair's slot. | the block's slots | All reads of the block are issued back to back, then **one** read barrier per block (catalog `double_buffer`: batch, then barrier). When `block_idx > 0` and `staging_depth == 1`: **one** `noc_async_writes_flushed` before the first read, since the slots are reused. |
| `scatter_store_block` | same slots. Per slot: `slot[j·face_w] = slot[j]` for `j = 1…face_w−1` (15 word moves), then **one** `face_bytes` write to output page `t` at `h·faces_per_face_row·face_bytes`, issued at once | slots until their writes have left L1 | No barrier inside the block. Write issue is interleaved per slot, so slot `k`'s write serializes on the NoC while slot `k+1` is moved. No CB handshake: the slots are single-thread scratch. |
| `drain_writes` | every write the mover issued | — | **one** `noc_async_write_barrier` per kernel run (BB-03) |

Once per kernel run, not per block: input and output `TensorAccessor` construction and the slot base (`get_write_ptr`).

#### Stall shadows

| Wait | What runs in its shadow | Why legal |
|------|-------------------------|-----------|
| read barrier in `gather_rows_block` | computing the output NoC addresses of the block's slots (`out_accessor.get_noc_addr(t, offset)`), placed before the barrier. With `staging_depth = 2`, block `i+1`'s gather is issued before block `i`'s scatter. | Addresses depend only on page ids. A pure move has no floating-point order, so any reorder of independent pages gives identical bits; no precision baseline gates it. |
| write barrier in `drain_writes` | nothing — it is the last stage and BB-03 requires it | — |
| one page per mover (production) | nothing beyond the address math: the scatter needs the read data and the write needs the scatter | data dependence |

### Perf lamps

| Lamp | Why the default may be wrong here | Nearby alternative to measure |
|------|-----------------------------------|-------------------------------|
| Grid synchronization | Production gives 1 page per core (16 cores for p1, 8 for p2), and each core's work is one DRAM round trip each way. Per-core dispatch work (unique runtime args, range size) may outweigh the parallelism. | `max_mover_cores ∈ {4, 8}` for p1 (2–4 pages per core, reads still one batch per block), against the full-grid default |
| Overlap | `staging_depth = 1`: with `num_blocks_this_core > 1`, each block's read round trip is exposed | `staging_depth = 2` with `block_w_tiles = block_w_tiles_cap`; only above `658·num_cores` pages |
| RISC split (catalog `split_reader`: pays only when a data-movement RISC is on the critical path) | The second mover takes ≈ 15 word moves and 2 NoC issues per page off the path, which is small against two DRAM round trips | `movers_per_core = 1`: one kernel binary, both halves per page on RISCV_1 |
| Per-core runtime args | Every mover core gets two unique args (`first_tile`, `core_w_tiles`), written per core by dispatch | derive `first_tile` in the kernel from the core's logical coordinates and the rectangle origin, so only common args remain |
| Barrier granularity (catalog `trid_double_issue`) | One read barrier per block holds every scatter until the block's last read lands | trid-tagged sub-batches with `async_read_barrier_with_trid` (`local_copy_helpers_dataflow.hpp:147,158`) plus `invalidate_l1_cache`; only meaningful when `block_w_tiles ≥ 4` |
| Face writes vs. page writes (catalog `tile_reorder`: whole pages beat face writes on DRAM-bound moves) | At DRAM-bound W: two 1024-B face writes per page from two RISCs vs. one 4096-B page write | `movers_per_core = 1` with a whole-tile slot written as one page; only at DRAM-bound W, since production is latency-bound |

## Dataflow Strategy

| Stage | Format | Mechanism | Notes |
|-------|--------|-----------|-------|
| DRAM input page `t` → mover L1 | 64 B = 16 float32 (face `h`, row 0) | `noc_async_read` from `in_accessor.get_noc_addr(t, h·face_bytes)` into row 0 of the mover's slot | only row 0 of an input tile crosses; padding rows are never read |
| inside the slot | float32 bit patterns handled as `uint32_t` | RISC-V word loads and stores, no FP instruction | row 0 → column 0 of the same face. Row-0 columns 1–15 keep their stale copies, which are output padding. |
| mover L1 → DRAM output page `t` | 1024 B = one 16×16 float32 face | `noc_async_write` (non-posted) to `out_accessor.get_noc_addr(t, h·faces_per_face_row·face_bytes)` | output face `2h`; faces 1 and 3 are not written (padding) |
| between the two RISCs of a core | none | — | the two movers share no CB, no semaphore and no L1 byte |
| between cores | none | — | — |

TARGET placement: TARGET has no `memory_layout` axis. DRAM interleaved is the only placement, so no sharded scheme is in scope. A physical shard of this op would be a placement of the `w_tile` split, which is independent, so it would be a knob-turn.

Structural impossibilities: none beyond `INVALID = []`.

## Work Distribution

| Field | Value |
|-------|-------|
| Work unit | one block: `block_w_tiles` consecutive pages × the mover's tile halves |
| Grid | `device.compute_with_storage_grid_size()` = `grid_x × grid_y`. `max_mover_cores` defaults to `grid_x·grid_y` (Phase 0) and is a host constant, never inlined. |
| Mover cores | `num_cores = min(tensor_w_tiles, max_mover_cores)` — p1 16, p2 8 |
| Per-core work | core `k` (row-major index inside the rectangle): `core_w_tiles_k = tensor_w_tiles div num_cores + (k < tensor_w_tiles mod num_cores ? 1 : 0)`, `first_tile_k = k·(tensor_w_tiles div num_cores) + min(k, tensor_w_tiles mod num_cores)` → 1 page per core for p1 and p2 |
| Blocks per core | `num_blocks_this_core = ceil(core_w_tiles / block_w_tiles)`, `last_block_extent = core_w_tiles − (num_blocks_this_core − 1)·block_w_tiles` → 1 and 1 for p1 and p2 |
| Remainder | ragged split in whole pages by the `mod` above. A partial tile (`W mod 32 ≠ 0`) is not a remainder: its padding lanes move with it into output padding. Tile counts use `ceil` per image. |
| Core placement (BB-17) | One rectangle when possible: `cols` = the largest divisor of `num_cores` with `cols ≤ grid_x` and `num_cores / cols ≤ grid_y`, giving `CoreRange((0,0), (cols−1, num_cores/cols − 1))`. If no divisor qualifies, fill rows row-major: full rows plus one partial row (2 ranges). p1 → 8×2, p2 → 8×1, one range each on any grid with `grid_x ≥ 8`. Wide rectangles per catalog `noc_placement`. Not `split_work_to_cores`' row-wise fill, which gives 13 + 3 for 16 cores on a 13-wide grid. Both movers use the same range. |
| Per-program vs. per-call data | Per program (cache key): extents, `staging_depth`, `movers_per_core`, the core rectangle, each core's `(first_tile, core_w_tiles)`. Per call: the input and output buffer addresses, and nothing else. |

Regime selection: one built regime. The plan function below is the only selector of extents; pin it in the op file and use it both for dispatch and for reporting.

```
plan(input_tensor, device):
    tensor_w_tiles     = ceil(W / tile_w)
    num_cores          = min(tensor_w_tiles, max_mover_cores)          # max_mover_cores = grid_x·grid_y
    max_core_w_tiles   = ceil(tensor_w_tiles / num_cores)
    block_tile_halves  = tile_halves / movers_per_core                 # movers_per_core = 2
    block_w_tiles_cap  = floor(op_l1_budget_bytes / (movers_per_core·staging_depth·block_tile_halves·face_bytes))
    block_w_tiles      = min(max_core_w_tiles, block_w_tiles_cap)      # staging_depth = 1, op_l1_budget_bytes = 1347776
    rectangle          = placement(num_cores, grid_x, grid_y)
```

| Obligation | Design |
|------------|--------|
| Plan report (BB-08) | `g6_unsqueeze_plan(input_tensor)`, a pure host function in the op file, returns the regime name, every extent above, the core rectangle and each core's page range. The entry point builds from the same result. |
| Role dump (BB-21) | The same report lists per core: RISCV_1 = `mover0` (halves `[0, block_tile_halves)`, NoC0) and RISCV_0 = `mover1` (the remaining halves, NoC1), with the core's page range. |
| Profiler zones (BB-21) | Each mover wraps `gather_rows_block` issue, the read-barrier wait, `scatter_store_block` and the final write-barrier wait in their own `DeviceZoneScopedN` zones. The zones compile to nothing in the plain build. |
| No rebuild on repeat (BB-07, BB-16) | The host caches the built program per key = (device id, logical input shape, padded input shape, dtype, layout, buffer type, memory layout, grid size, knob values). On a hit it rewrites only the two addresses, with no new kernel, CB or descriptor objects. Use either a cached `ProgramDescriptor` with `custom_program_hash` set to a hash of the key salted with the op name (`ttnn/cpp/ttnn-nanobind/program_descriptors.cpp:1140-1143`, the `custom_program_hash` property), or the `ProgramSpec` + tensor-params path of `ttnn.generic_op` (`ttnn/cpp/ttnn/operations/generic/generic_op.cpp:38-48`), whichever lets a hit patch addresses without a rebuild. Everything that changes the program is in the key; addresses are not. |
| Kernel size (BB-04, BB-18) | Two small dataflow binaries (one source compiled per RISC), default optimisation level — no size-optimising build |

## Circular Buffers

| Semantic Name | Index | Page Size | Num Pages | Sizing rationale | Format | Producer | Consumer | Lifetime |
|---------------|-------|-----------|-----------|------------------|--------|----------|----------|-----------|
| `cb_mover0_face_slots` | 0 | `face_bytes` (1024) | `staging_depth · block_w_tiles · block_tile_halves` (p1/p2: 1) | spans `w_tile` (`block_w_tiles · staging_depth`), `tile_half` (`block_tile_halves`) and `face_lane` (the face's 16 rows); `lead` and `new_axis` have extent 1. No op dimension appears. | Float32 | `mover0` (RISCV_1) | `mover0` (RISCV_1) | whole kernel run |
| `cb_mover1_face_slots` | 1 | `face_bytes` (1024) | `staging_depth · block_w_tiles · block_tile_halves` (p1/p2: 1) | same as `cb_mover0_face_slots` | Float32 | `mover1` (RISCV_0) | `mover1` (RISCV_0) | whole kernel run |

Both CBs are single-thread scratch allocations, addressed as `get_write_ptr(cb) + slot·face_bytes`. Producer and consumer are the same thread, so the CB-ownership invariant holds. There is no reserve, push, wait or pop: push count = wait count = 0, and the ring-wrap invariant does not apply. Both exist only on the mover rectangle. With `movers_per_core = 1` (lamp), only `cb_mover0_face_slots` exists, with `block_tile_halves = 2`. Ledger: `l1_ledger.md`.

## Block Operation Realization

| # | Block operation | Block shape | Helper? | Input CB (semantic name, pages, state) | Output CB (semantic name, pages) | CB state after |
|---|-----------------|-------------|---------|----------------------------------------|----------------------------------|----------------|
| 1 | `gather_rows_block` | `block_w_tiles × block_tile_halves` face rows of `face_row_bytes` | none — raw NoC reads through `TensorAccessor` (API Mapping) | DRAM input pages `[first_tile + block_idx·block_w_tiles, +extent)` | `cb_mover{m}_face_slots`, `block_w_tiles · block_tile_halves` slots | row 0 of each slot holds its 16 values; the rest of the slot is stale |
| 2 | `scatter_store_block` | the same slots | none — custom block operation (row → column move); gap recorded in API Mapping | `cb_mover{m}_face_slots`, same slots, transformed in place | DRAM output pages (`TensorAccessor`) at face `2h` | column 0 of each slot holds its 16 values and has been written; the slot is reusable once flushed |
| 3 | `drain_writes` | every write the mover issued | none | — | — | all writes acknowledged (BB-03) |

## API Mapping

| Block operation | Type | Function | File:Line | Template Params / Args | Input CB | Output CB | Which params are block knobs |
|-----------------|------|----------|-----------|------------------------|----------|-----------|------------------------------|
| `gather_rows_block`, `scatter_store_block` | raw_api | `TensorAccessor::get_noc_addr(page_id, offset, noc)`; host `ttnn.TensorAccessorArgs(tensor)` | `tt_metal/hw/inc/api/tensor/tensor_accessor.h:71,165`; `ttnn/cpp/ttnn-nanobind/tensor_accessor_args.cpp:15` | input: `(t, h·face_bytes)`; output: `(t, h·faces_per_face_row·face_bytes)` | DRAM | — | page range from `first_tile`, `block_w_tiles` |
| `gather_rows_block` | raw_api | `noc_async_read` | `tt_metal/hw/inc/api/dataflow/dataflow_api.h:547` | `(src_noc_addr, slot_addr, face_row_bytes)` | DRAM input | `cb_mover{m}_face_slots` | reads per barrier = `block_w_tiles · block_tile_halves` |
| `gather_rows_block` | raw_api | `noc_async_read_barrier` | `dataflow_api.h:1921` (fence at `:1932`) | once per block | — | — | frequency = per block |
| `gather_rows_block` (slot reuse) | raw_api | `noc_async_writes_flushed` | `dataflow_api.h:1980` | once per block boundary when `staging_depth == 1` | — | — | `staging_depth` |
| `scatter_store_block` | raw_api | `get_write_ptr` + volatile `uint32_t` L1 pointer (`addr_to_l1_ptr`) | `dataflow_api.h:325`; `ttnn/cpp/ttnn/kernel_lib/l1_helpers.hpp:30` | `slot[j·face_w] = slot[j]`, `j = 1…face_w−1` | `cb_mover{m}_face_slots` | same CB, in place | slot count `block_w_tiles · block_tile_halves` |
| `scatter_store_block` | raw_api | `noc_async_write` (non-posted) | `dataflow_api.h:876` | `(slot_addr, dst_noc_addr, face_bytes)` | `cb_mover{m}_face_slots` | DRAM output | one per slot |
| `drain_writes` | raw_api | `noc_async_write_barrier` | `dataflow_api.h:1951` | once per kernel run | — | — | — |
| all | raw_api | `DeviceZoneScopedN` | `tt_metal/hw/inc/api/debug/kernel_profiler.hpp:1012` | zones per block operation and per barrier wait | — | — | — |
| host | raw_api | `ttnn.DataMovementConfigDescriptor(processor, noc)` | `ttnn/cpp/ttnn-nanobind/program_descriptors.cpp:599-609` | `mover0`: RISCV_1 / NOC_0; `mover1`: RISCV_0 / NOC_1 | — | — | `movers_per_core` |

### Helpers considered and rejected

| Candidate | File:Line of the mismatch | Why it cannot express the shape |
|-----------|---------------------------|---------------------------------|
| `copy_tile_column<dfb_id, T>` | `ttnn/cpp/ttnn/kernel_lib/tile_column_dataflow.hpp:92` | Copies column `src_col` to column `dst_col`: both sides walk `column_base + row_offset` (`:47-56`). The move here is row 0 → column 0. Its layout also needs a DFB tile shape (`get_tile_r_dim<dfb_id>`, `:38`), while a slot is one face. **Gap**: a `copy_tile_row_to_column` (row source, column destination, in place allowed) in `tile_column_dataflow.hpp` would cover the move in `scatter_store_block`. |
| `fill_tile_column<dfb_id, T>` | `tile_column_dataflow.hpp:70` | writes one constant, not a row of values |
| self-aimed NoC L1 → L1 copies (`local_addr`, `set_read_state`, `read_with_state`) | `ttnn/cpp/ttnn/kernel_lib/local_copy_helpers_dataflow.hpp:7-35` | A NoC copy moves contiguous ranges. Each value would be its own 4-B transfer whose source offset `4j` and destination offset `64j` differ modulo the 16-B L1 NoC alignment (`noc_parameters.h:375-376`). That is illegal for `j mod 4 ≠ 0`, and costs 15 transactions per face where 15 word moves suffice. |
| compute helpers (`tilize`, `untilize`, `eltwise_*`, `reduce`, `matmul_block`) and an unpack-transpose | — | The op has no compute phase. The only compute route (`compute_engine_transpose`) is rejected for bit exactness. |
| `ttnn.Mcast` / `McastArgs` | `ttnn/cpp/ttnn/kernel_lib/mcast/` | no reuse-shared operand and no cross-core step |

## Key Risks and Gotchas

| Risk | Why it bites here | Mitigation in this design |
|------|-------------------|---------------------------|
| Input padding in a logical output position | Rows 1–31 of every input tile are padding, and the `poisoned_padding` case fills them with NaN. A whole-tile copy or a wrong face offset moves them into column 0. | Read only face `h` row 0 (offset `h·face_bytes`, `face_row_bytes` long). The acceptance test poisons the padding with NaN. |
| A value converted on the way | NaN payloads, −0.0, infinities and subnormals must survive (`special_values` case) | only `uint32_t` loads and stores plus NoC copies; no FP instruction, no compute kernel |
| Wrong output face | Column 0 of rows 16–31 is face 2, not face 1 (faces form a row-major 2×2) | output offset `h·faces_per_face_row·face_bytes`, derived from tile and face dims; the tests cover both halves of every tile |
| Stale L1-cache words on Blackhole | NoC writes into slots a RISC has read before (slot reuse across blocks) | scatter only after `noc_async_read_barrier`, which fences; a trid variant calls `invalidate_l1_cache()` |
| Slot overwritten while its write is in flight | `staging_depth = 1` with more than one block | `noc_async_writes_flushed` before the next gather into the slots |
| NoC traffic still in flight when the kernel returns (BB-03) | another program starts right after the call | non-posted writes and a final `noc_async_write_barrier`; every read is retired by its block's barrier |
| Incomplete program-cache key (BB-07) | p2 then p1 differ only in W (8 vs. 16 pages) — the `cache_two_widths` case | the key holds the logical shape, which fixes `tensor_w_tiles`, the rectangle and per-core ranges |
| A program built per call or after capture (BB-07, BB-16) | the host would otherwise build a descriptor every call | cache per key; a hit rewrites only the two addresses |
| Two movers on one CB | CB ownership is one thread per role | one CB per mover |
| DRAM read misalignment | Blackhole DRAM reads align at 64 B | face rows are 64 B, slot bases are DRAM-aligned, slot stride is 1024 B; host asserts |
| Scattered core ranges (BB-17) | `split_work_to_cores` row-wise fill splits 16 cores 13 + 3 on a 13-wide grid | rectangle placement rule in Work Distribution |
