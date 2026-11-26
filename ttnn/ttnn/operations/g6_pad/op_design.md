# Operation Design: g6_pad

## Overview

| Field | Value |
|-------|-------|
| Classification | data_movement (no arithmetic, no compute kernel) |
| Goal | Constant padding at the high end of the trailing dims, bit-identical, as exactly one device program per call. A `tile_padding_extension` call rewrites only the last tile row of each slice of x in place and returns a metadata-only view of x's buffer; a `new_storage` call writes every output tile once from at most one read of each input tile. |
| Math | `D_i = d_i + h_i`; `out[j] = x[j] if j_i < d_i for every i else c`, `c` = `value` converted to x's dtype as torch converts a Python scalar |
| References | `docs/explorations/g6_pad/DOSSIER.md`, `docs/explorations/g6_pad/tensix.md` (geometry, storage classes, view routes), `docs/explorations/g6_pad/risks.md` (R1–R10), `eval/golden_tests/g6_pad/feature_spec.py`, `eval/golden_tests/g6_pad/helpers.py` (oracle, storage_class), catalog `ttnn/ttnn/operations/examples/{width_split,noc_placement,double_buffer,tile_reorder,bank_stagger}` |

## Parameters

| Name | Type | Required | Valid Range | Default | CT/RT |
|------|------|----------|-------------|---------|-------|
| `input_tensor` | ttnn.Tensor | yes | rank 2–4, interleaved (DRAM or L1); sharded → `ValueError` | — | — |
| `padding` | list/tuple of k 2-element lists/tuples of `int` | yes (positional or keyword) | `1 <= k <= rank`; every pair `(0, h)` with `h >= 0`; anything else → `ValueError` | — | geometry derived from it is **CT** (part of the program hash) |
| `value` | `int` or `float` (not `bool`) | no (positional or keyword) | for int32 / uint32 inside the dtype's range, else `ValueError` | `0.0` | its bit pattern `fill_bits` is an **RT** arg (overridden every call, never baked into the cached program) |

Padding canonicalisation (entry point, before anything else): pairs align to the trailing dims —
`highs = [0]*(rank-k) + [h for (_, h) in padding]` (`helpers._highs` form). The torch flat list is
never used on device.

`fill_bits` derivation (host, scalar only, no device op): `torch.tensor(value, dtype=torch_dtype)`
then `.view(int16|int32)` → masked to 16 / 32 bits. bfloat16 by float32 RNE (torch's own
conversion), int32/uint32 by truncation after the range check. `-1.7014118346046923e38` → bf16
`0xFF00` (−2^127, finite).

## Tensors

### Input

| Property | Requirement |
|----------|-------------|
| Shape | `(d_0, …, d_{r-1})`, r ∈ {2,3,4}; any dim any size |
| Dtype | bfloat16 (Phase 0); float32, int32, uint32 (refinements) |
| Layout | TILE (Phase 0); ROW_MAJOR (refinement) |
| Memory | DRAM interleaved (Phase 0); L1 interleaved (refinement); sharded refused (`ValueError`) |

### Output

| Property | Value |
|----------|-------|
| Shape | `(d_0+h_0, …, d_{r-1}+h_{r-1})` |
| Dtype | x's dtype |
| Layout | x's layout |
| Memory | x's memory config. `tile_padding_extension`: x's own buffer, viewed at the output logical shape with x's padded shape. `new_storage`: `ttnn.allocate_tensor_on_device(out_shape, dtype, layout, device, x.memory_config())` |

### Storage class (host predicate, decided by shapes alone — same as `helpers.storage_class`)

```
extension = (layout == TILE)
            and highs[-2] > 0 and all(h == 0 for i, h in enumerate(highs) if i != rank-2)
            and ceil(d_{r-2}/32) == ceil(D_{r-2}/32)
```
Every other call is `new_storage`. Note: `extension` implies `d_{r-2} % 32 != 0` (dim −2 grows
inside its last tile), so the last tile row of each slice always holds logical rows.

### The extension view (pinned route — launches no device program)

`out = ttnn.reshape(input_tensor, ttnn.Shape(out_shape), input_tensor.padded_shape)`

Paper trace (implementer must confirm with a graph-capture probe on every extension production
shape, see Key Risks): the logical volume changes, so the routed codegen plan is refused
(`reshape_view/codegen/reshape_codegen_supported.cpp:402-403`, "explicit padded shape or volume
change"); `this_is_view` is false (`reshape_view/reshape.cpp:760-766`, dim −2 is non-aligned);
the volume-change branch (`reshape.cpp:778-788`) takes `ttnn::experimental::view(tensor, logical,
padded)` because the padded shape is tile-aligned and its last dim equals x's; that is
`tt::tt_metal::view` (`ttnn/core/tensor/tensor_ops.cpp:465-477`), a new tensor on x's mesh buffer
address — no allocation, no program. The graph tracker records a `Tensor::reshape`
function node without `program_factory_type`, which the golden one-program check does not count.
The view is made **after** `generic_op` returns (the program reads and writes x's buffer through
x). The ProgramDescriptor is dispatched with `io_tensors = [input_tensor]` (in-place: the one
tensor is both source and destination).

## Blocking Model

### Axes

The op's output tile index space (TILE layout) has three axis groups. Every one is
**independent**: each output tile depends on at most the input tile at the same index (or none),
so no result spans any axis and no axis needs a combine. The fill constant `c` is the only operand
shared by many tiles; it is generated per core from an RT scalar (zero DRAM, zero NoC), so it is
reuse-shared without any transfer to reuse.

The three groups are **folded into one work axis** — the *work tile* index `w` over the touched
tile set, tile column fastest — because (a) all are independent, (b) the op treats every tile
identically except for its per-tile class (copy / boundary / fill-only), and (c) interleaved tile
pages are separate NoC transfers to separate banks, so a 2-D rectangle of tiles has no
transfer-shape advantage over a run of equal length. The block is a run of `block_tiles`
consecutive work tiles.

Touched tile set (`first_tile_row` is the single place the storage class enters the work space):

| Storage class | `first_tile_row` | Work tiles `num_work_tiles` |
|---|---|---|
| tile_padding_extension | `Ht_out - 1` | `S_out * 1 * Wt_out` (last tile row of every slice; nothing else is read or written) |
| new_storage | `0` | `S_out * Ht_out * Wt_out` (every output tile) |

with `S_out = prod(D_0..D_{r-3})`, `Ht_out = ceil(D_{r-2}/32)`, `Wt_out = ceil(D_{r-1}/32)` (per
image, `ceil`, never `floor`/`//` of a flattened height); input analogues `S_in`, `Ht_in`, `Wt_in`
from `d`. Rank 2: `S = 1`.

| Axis | Character (+ one-clause reason) | Extent knob | Phase 0 value | Knob source | Core-assignment | Later unlock |
|------|--------------------------------|-------------|---------------|-------------|-----------------|--------------|
| slice (leading dims d_0..d_{r-3}, flattened) | independent — an output tile of slice s reads only slice s of x (or nothing when s lies in the leading padding) | folded into `block_tiles` (a block may cross slice boundaries) | — (folded) | `block_tiles` | spread: part of the folded run split over the grid | knob-turn (leading-dim padding = fill-only tiles, already classified) |
| tile row (dim −2) | independent — no result spans rows | folded into `block_tiles`; the touched range is `[first_tile_row, Ht_out)` | extension: 1 tile row; new_storage: all | `first_tile_row` (host, from storage class) | spread: part of the folded run | knob-turn |
| tile column (dim −1) | independent — no result spans columns | folded into `block_tiles` | — (folded) | `block_tiles` | spread: part of the folded run (this is what fills the grid on p2–p4, 1×1940 tiles; catalog `width_split`) | knob-turn |
| **work tile `w`** (the fold of the three) | independent | `block_tiles` | `min(max_tiles_per_core, block_tiles_cap)` — the whole per-core assignment as one block in every INPUTS shape | host constant computed once in the program-descriptor builder; CT arg to reader and writer | `ttnn.split_work_to_cores(grid, num_work_tiles, row_wise=True)`: contiguous `w` ranges, ≥ `min_tiles_per_core` tiles each, all cores of the device grid eligible | knob-turn |
| intra-tile rows/cols (the 32×32 tile) | independent — elementwise | whole tile (a tile is the page; never split) | 32×32 | fixed by the layout | not split: one tile is one NoC page | — |
| fill constant `c` | reuse-shared — the same bits in every padded position of every tile | one tile (writer-side fill page) / per-range fills (mask) | 1 tile | RT arg `fill_bits` | every core makes its own; no mcast | — (mcast rejected, regime table) |

Derived per call (single source each, everything else computed from them):

| Symbol | Definition | Source |
|---|---|---|
| `grid` | `device.compute_with_storage_grid_size()` | host, read every build; part of the descriptor-cache key |
| `min_tiles_per_core` | Phase 0 = 1 | host constant (grid-sync lamp) |
| `num_cores` | `min(grid.x*grid.y, ceil(num_work_tiles / min_tiles_per_core))` | host |
| `max_tiles_per_core` | `ceil(num_work_tiles / num_cores)` (core_group_1 of the split) | host |
| `block_tiles_cap` | `floor((L1_OP_BUDGET - fill_tile_bytes) / (2 * tile_bytes))`, `L1_OP_BUDGET = 1347776` | host constant expression |
| `block_tiles` | `min(max_tiles_per_core, block_tiles_cap)` | host → CT arg |
| `num_blocks_this_core` | `ceil(tiles_this_core / block_tiles)` | host → RT arg |
| `buffer_depth` | `2 if ceil(max_tiles_per_core / block_tiles) > 1 else 1` | host → CB size only |

### Buffer-depth knobs

| CB | Depth knob | Phase 0 value | What the depth buys |
|----|------------|---------------|---------------------|
| `cb_work_tiles` | `buffer_depth` | 1 when a core has one block (every INPUTS shape), 2 as soon as a core has > 1 block | depth 2 overlaps the reader's next block (read + mask) with the writer's current block; with one block per core there is nothing to overlap, so a second slot would be dead L1 |
| `cb_fill_tile` | none (constant, 1 page) | 1 | — (built once, read-only afterwards) |

### Mechanism caps

| Mechanism | Cap on which extent | Clamp | What happens unclamped |
|-----------|--------------------|-------|------------------------|
| L1 budget of the model (BB-05: 1347776 B per core) | `block_tiles × buffer_depth` | `block_tiles <= block_tiles_cap` (above); larger assignments stream in more blocks | CB allocation beyond the model's free L1: launch failure inside the model only |
| CB ring wrap (pointers reset only on an exact `fifo_limit` hit) | push/pop quantum | reader pushes and writer pops exactly `block_tiles` per block, including a ragged last block (only the NoC transfers narrow to `last_block_extent`) | ring corruption on the second block of a ragged core |
| `fill_l1_range<val_size>` (`ttnn/cpp/ttnn/kernel_lib/l1_helpers.hpp:100-101`): start and length multiples of `val_size`, `val_size ∈ {1,2,4}` | mask ranges | `val_size = elem_bytes` (2 for bf16, 4 for fp32/int32/uint32); every mask range is a whole number of elements by construction | head/tail elements silently under-filled (l1_helpers.hpp:96-98) → x's tile padding leaks (R1) |
| NoC transfer = one interleaved page | per-tile transfer | one whole tile per read and per write (2048 B bf16, 4096 B 32-bit); never a transfer across two pages | a transfer spanning two banks writes the wrong bank |
| DRAM/L1 alignment of whole-tile transfers | page start | whole tiles start at page boundaries (aligned for every arch); no sub-tile DRAM transfer in any built regime | — (the sub-tile write-only regime is deferred precisely on this cap) |
| Fill pattern width | `fill_bits` | 32-bit RT arg; bf16 uses its low 16 bits | — |
| Tile index width | `num_work_tiles`, page ids | uint32; max in INPUTS 9680 | — |

### Regimes

Minimum named against the DRAM (or L1-interleaved) boundary: *each input tile holding a logical
value crosses at most once, each output tile that must change is written at most once, and no
fill-only page is read*.

| Regime | Status | Predicate | Block | Data movement vs. minimum | What a bigger block buys |
|--------|--------|-----------|-------|---------------------------|--------------------------|
| **A. extension, in place, read-mask-write the last tile row** | built | `extension` (host predicate above) | `block_tiles` work tiles from `S_out × Wt_out` (one tile row per slice) | reads each last-tile-row tile once and writes it once; every other tile row: zero crossings. Above the strict minimum (only rows `d%32..31` need writing, x needs no read): +1 read and +(rows above `d%32`) write per touched tile — the structural cost of whole-tile transfers; minimum reachable only by regime C | fewer read barriers, CB reserve/push/wait/pop handshakes and write flushes per tile (intended once per block); mask-range setup computed once per kernel |
| **B. new storage, one pass with in-L1 assembly** | built | `not extension` and layout TILE | `block_tiles` work tiles from `S_out × Ht_out × Wt_out` | minimum: copy and boundary tiles read once and written once; fill-only tiles never read, written once from the core's `cb_fill_tile` | same fixed costs as A; the fill tile is built once per kernel, not per block |
| C. extension, write-only new rows | deferred — layers on A (same work set, same per-tile walk; only the boundary-tile action changes from read+mask+write to sub-tile writes of `c` from `cb_fill_tile`); deferred because the sub-tile DRAM writes start at row offset `(d%32)·16·elem_bytes` inside a face, which violates DRAM write alignment for odd `d%32` on Blackhole (64 B) and turns 1 write/tile into 2–3 transfers of 64–1024 B (count-bound, catalog `tile_reorder`); must be measured on p2–p4 (offsets 64 B / 256 B, aligned) before it is worth its predicate | `extension and ((d%32)·16·elem_bytes) % dram_write_alignment == 0` | same as A | the minimum: zero reads; writes rows `d%32..31` only (p2: 3.72 MB instead of 3.97 MB read + 3.97 MB written) | — |
| D. ROW_MAJOR, stick pages | deferred — refinement of the `layout` axis; reuses the folded work-index + per-page classification skeleton with page = one stick (output rows `S_out × D_{r-2}`, each copy / boundary (columns `d_{r-1}..D_{r-1}-1` filled in L1) / fill-only); needs its own handling of sticks below DRAM alignment (4 B and 32 B rows at a 64 B stride, R5) | `layout == ROW_MAJOR` | `block_sticks` | minimum (same one-pass assembly) | same |
| E. two-pass fill then copy | rejected — forbidden by the contract (second full-tensor pass); superseded by B's in-L1 assembly | — | — | output written twice, fill-only pages written then overwritten | — |
| F. extension copied into new storage | rejected — forbidden for multi-row slices; superseded by A (in place, last tile row only) | — | — | moves every tile row (121× the needed bytes on [1,3844,C]) | — |
| G. masking on the compute engine (unpack → fill/where → pack) | rejected — any unpack/pack in another format breaks bit equality for int32/uint32/fp32 (R2) and costs a compute kernel; superseded by A/B's RISC-V/L1 mask on raw bytes | — | — | same DRAM traffic plus an L1 round trip through SRC/DEST | — |
| H. read-once-and-mcast the fill tile | rejected — `c` is generated locally from a 32-bit RT arg at zero traffic; any multicast adds NoC traffic and a semaphore round trip | — | — | — | — |

Regime selection function (host, exact): `regime = "A" if extension else ("B" if layout == TILE else "D")`.
Regime-pinned tests: the acceptance test carries both built regimes (A on single- and multi-tile-row
slices, B with `h` and `hw` padding); the regime is a pure function of shapes, independent of the
device grid.

### Traffic ranking

Candidate splits, ranked on DRAM crossings, then cross-core traffic:

| Split | DRAM crossings | Cross-core traffic | Notes |
|---|---|---|---|
| folded work-tile run over the full grid (**chosen**) | minimum of the regime (A / B above) | none | fills the grid for tall (`[1,3844,C]` new-storage-like), wide (p2–p4: 1 × 1940) and batched (`[16,1,C]`) shapes alike |
| tile-row split | same bytes | none | ties on bytes but strands p2–p4 and every extension call (one touched tile row per slice) on `S` cores — on p2–p4 a single core (catalog `width_split`: up to 7.8× slower) |
| slice split | same bytes | none | `S = 1` for most cases → one core |
| tile-column split | same bytes | none | ties on p2–p4, fails tall new-storage shapes (`Wt = 4` on `[1,256,128]`) |
| split a dependent axis + combine | n/a | n/a | the op has no dependent axis |

Every split moves the same bytes (no operand is shared across tiles except `c`, which moves
nothing), so the ranking is a tie on traffic and the tiebreaker is the only one left: the split
that keeps every INPUTS shape on the grid. Core placement `row_wise=True` (catalog
`noc_placement`: a column line of DRAM-facing cores is ~2.9× slower); reader on NoC0, writer on
NoC1 (default configs).

### Block schedule

Per core, reader (NCRISC, NoC0) and writer (BRISC, NoC1); no compute kernel.

```cpp
make_fill_tile();                                   // writer, once per kernel, only if the call has fill-only tiles
for (uint32_t block_idx = 0; block_idx < num_blocks_this_core; ++block_idx) {
    load_block(block_idx);                          // reader
    mask_block(block_idx);                          // reader
    store_block(block_idx);                         // writer
}
drain_writes();                                     // writer, once per kernel
```

| Block operation | Block shape | Acts on | Resident across it | Intended frequency of fixed costs |
|---|---|---|---|---|
| `make_fill_tile` | 1 tile | `cb_fill_tile`: every element = `fill_bits` (`elem_bytes` wide) | `cb_fill_tile` for the whole kernel | once per kernel launch; skipped (and CB not allocated) when `has_fill_only_tiles == 0` |
| `load_block` | `block_tiles` work tiles (ragged last block: `last_block_extent`) | reserve `block_tiles` slots of `cb_work_tiles`; for every work tile with an input tile (`has_input`), one whole-tile `noc_async_read` into its slot; fill-only tiles leave their slot untouched | — | one `cb_reserve_back` and **one** read barrier per block |
| `mask_block` | same block | for every boundary tile in the block, overwrite in its slot the tile positions that lie in x's tile padding but inside out's logical box with `c` (ranges below); copy tiles untouched; then push `block_tiles` | mask byte ranges (computed once per kernel from CT args) | one `cb_push_back(block_tiles)` per block; mask ranges per boundary tile |
| `store_block` | same block | wait `block_tiles`; for each work tile one whole-tile write to output page `out_page(w)`: from its slot (copy/boundary) or from `cb_fill_tile` (fill-only); flush; pop `block_tiles` | `cb_fill_tile` | one `cb_wait_front`, one `noc_async_writes_flushed`, one `cb_pop_front` per block |
| `drain_writes` | — | `noc_async_write_barrier()` | — | once per kernel (BB-03: nothing in flight at return) |

Per-tile classification inside a block (both kernels compute it identically from the same CT/RT
args; counters `(s, tr, tc)` are advanced incrementally, no division per tile):

| Quantity | Definition |
|---|---|
| work tile → output coords | `w = w_begin + i`; `tr = first_tile_row + (w / Wt_out) % (Ht_out - first_tile_row)`, `tc = w % Wt_out`, `s = w / (Wt_out*(Ht_out-first_tile_row))` |
| `out_page` | `s*Ht_out*Wt_out + tr*Wt_out + tc` |
| `slice_in` | decompose `s` over the output leading dims `(D_0..D_{r-3})`; `slice_exists` iff every leading coord `< d_i`; `s_in` = the same coords linearised over `(d_0..d_{r-3})` (Phase 0: leading dims never padded → `s_in = s`) |
| `has_input` | `slice_exists and tr < Ht_in and tc < Wt_in` |
| `in_page` | `s_in*Ht_in*Wt_in + tr*Wt_in + tc` (extension: equals `out_page`; same buffer) |
| `row_mask` | `has_input and tr == Ht_in-1 and D_{r-2} > d_{r-2} and r0 != 0`, `r0 = d_{r-2} % 32` |
| `col_mask` | `has_input and tc == Wt_in-1 and D_{r-1} > d_{r-1} and c0 != 0`, `c0 = d_{r-1} % 32` |
| class | fill-only = `!has_input`; boundary = `row_mask or col_mask`; copy otherwise |

Mask byte ranges inside a tile (`e = elem_bytes`; face = 16×16, face row = `16e` B, face = `256e` B;
faces 0,1 = rows 0–15 (cols 0–15, 16–31), faces 2,3 = rows 16–31). Rows/cols beyond out's
logical box are out's own tile padding and may take `c` too; filling them keeps the ranges simple.

| Mask | Condition | Byte ranges filled with `c` (offsets inside the tile) |
|---|---|---|
| row, `r0 < 16` | `row_mask` | face0 `[r0·16e, 256e)`, face1 `[256e + r0·16e, 512e)`, faces 2–3 `[512e, 1024e)` |
| row, `r0 >= 16` | `row_mask` | face2 `[512e + (r0-16)·16e, 768e)`, face3 `[768e + (r0-16)·16e, 1024e)` |
| col, `c0 < 16` | `col_mask` | faces 1 and 3 whole (`[256e,512e)`, `[768e,1024e)`); in each of the 32 face rows of faces 0 and 2, `[row_base + c0·e, row_base + 16e)` |
| col, `c0 >= 16` | `col_mask` | in each of the 32 face rows of faces 1 and 3, `[row_base + (c0-16)·e, row_base + 16e)` |

### Perf lamps

| Lamp | Why the default may be wrong here | Nearby alternative to measure |
|------|-----------------------------------|-------------------------------|
| Overlap | one block per core (the default in every INPUTS shape) serialises the core's reads and writes: the writer idles until the whole block is read and masked | `block_tiles = 4..8` with `buffer_depth = 2` (catalog `double_buffer`: sweet spot 4–8, "no gain once DRAM-bound") on p2–p4 and p1 |
| Grid synchronization | `min_tiles_per_core = 1` puts 1 tile on each of 80 cores for `[1,3844,2560]` extension, 1–2 tiles per core on the small new-storage cases: launch and per-core fixed cost may dominate | `min_tiles_per_core ∈ {2, 4, 8}` on case 250 (80 tiles), case 254 (768), case 65 (64) |
| Mask realisation | the default mask writes `c` with RISC-V stores (`fill_l1_range`, ~1.5 KB per bf16 boundary tile; every tile of A is a boundary tile) on the reader's critical path | (a) local NoC L1→L1 copies of the same byte ranges from a reader-owned fill tile; (b) for A only (all tiles share `r0`): pre-fill every slot with `c` once and read only the logical face rows of each tile (2 reads of `r0·16e` B, faces 0–1) so the masked rows are never overwritten — saves the mask and 1–1.9 KB of read per tile, at 2 transfers per tile |
| Bank stagger | all cores start their run on consecutive tile ids → bank pattern fixed by `w_begin` | catalog `bank_stagger` start offset per core (rotate within the core's run) on p2–p4 |

## Dataflow Strategy

| Stage | Format | Mechanism | Notes |
|-------|--------|-----------|-------|
| host: validate, classify, view | — | Python | `validate()` first; storage class; extension view after dispatch |
| x (DRAM/L1 interleaved) → `cb_work_tiles` | raw tile bytes, x's dtype | reader: `TensorAccessor` + whole-tile `noc_async_read` per `has_input` tile, one barrier per block | no unpack, no format conversion anywhere |
| in-slot assembly | raw bytes | reader: `fill_l1_range<elem_bytes>` over the mask ranges | assembles boundary pages in L1 (allowed one-pass assembly) |
| `cb_work_tiles` → out | raw bytes | writer: `TensorAccessor` on out (extension: on x, same buffer) + whole-tile `noc_async_write` | flush per block, barrier at end |
| `cb_fill_tile` → out | raw bytes | writer: whole-tile write of the fill page for fill-only tiles | no DRAM read of fill-only pages |

TARGET axes seen through the regimes:

| TARGET axis value | Cost class | Why |
|---|---|---|
| dtype float32 / int32 / uint32 | knob-turn | `elem_bytes` CT arg, tile bytes from `ttnn.tile_size(dtype)`; raw-byte path is dtype-agnostic |
| memory_config "l1" | knob-turn | `TensorAccessorArgs` covers L1 interleaved; extension writes in place in L1 |
| rank 2 | knob-turn | `S = 1` |
| alignment tile_aligned / padded_dims leading, w | knob-turn | all covered by the per-tile classification (leading padding → fill-only slices; `w` → col mask + fill-only columns) |
| layout ROW_MAJOR | scheme-change (regime D) | page = stick, alignment handling for sub-64 B sticks |
| memory_layout sharded | not in TARGET (refused) | — |

No inter-Tensix communication in any regime.

## Work Distribution

| Field | Value |
|-------|-------|
| Work unit | one work tile `w` of the touched set; blocks of `block_tiles` |
| Grid | the device's `compute_with_storage_grid_size()`; `num_cores = min(grid cores, ceil(num_work_tiles / min_tiles_per_core))` |
| Per-core work | `ttnn.split_work_to_cores(grid, num_work_tiles, row_wise=True)`: core_group_1 gets `max_tiles_per_core`, core_group_2 one fewer; each core a contiguous `[w_begin, w_begin + tiles_this_core)` assigned in row-major core order |
| Remainder | group 2 + ragged last block per core (`last_block_extent = tiles_this_core - (num_blocks_this_core-1)*block_tiles`); geometry from per-image `ceil` |
| Kernel groups | ONE reader and ONE writer `KernelDescriptor` over the union of both core groups (per-core differences are RT args) → the used cores are a row-major prefix of the grid: ≤ 2 rectangles per RISC (BB-17) |

Phase 0 examples (grid 13×10 = 130 assumed for illustration only; the grid is read at run time):

| Case | Regime | `num_work_tiles` | cores | tiles/core |
|---|---|---|---|---|
| p2–p4 `[1,1,R,62080]` | A | 1940 | 130 | 15 |
| 247 `[1,3844,1536]` | A | 48 | 48 | 1 |
| 254 `[16,1,1536]` | A | 768 | 130 | 6 |
| 65 `[1,1,196,196]→256²` | B | 64 | 64 | 1 |
| 64 `[1,196,128]→[1,256,128]` | B | 32 | 32 | 1 |

## Circular Buffers

| Semantic Name | Index | Page Size | Num Pages | Sizing rationale | Format | Producer | Consumer | Lifetime |
|---------------|-------|-----------|-----------|------------------|--------|----------|----------|-----------|
| `cb_work_tiles` | 0 | `tile_bytes = ttnn.tile_size(dtype)` | `block_tiles * buffer_depth` | spans the work-tile axis by `block_tiles`; depth for overlap only when a core has > 1 block | x's dtype (raw bytes; DM only) | reader | writer | whole kernel |
| `cb_fill_tile` | 1 | `tile_bytes` | 1 | constant: one fill page | x's dtype | writer | writer (self-owned scratch) | whole kernel; allocated only when `has_fill_only_tiles` |

## Block Operation Realization

| # | Block operation | Block shape | Helper? | Input CB (semantic name, pages, state) | Output CB (semantic name, pages) | CB state after |
|---|-----------------|-------------|---------|----------------------------------------|----------------------------------|----------------|
| 0 | `make_fill_tile` | 1 tile | `fill_l1_range` | — | `cb_fill_tile`, 1 (writer reserves once, fills, never pushes/pops; uses `get_write_ptr`) | filled, resident to kernel end |
| 1 | `load_block` | `block_tiles` | raw dataflow | x via accessor | `cb_work_tiles`, `block_tiles` reserved | reserved, bytes landed after the block barrier |
| 2 | `mask_block` | `block_tiles` | `fill_l1_range` | `cb_work_tiles` (reserved by reader) | `cb_work_tiles`, push `block_tiles` | pushed |
| 3 | `store_block` | `block_tiles` | raw dataflow | `cb_work_tiles`, wait `block_tiles`; `cb_fill_tile` | out via accessor | popped after flush |
| 4 | `drain_writes` | — | raw dataflow | — | — | all writes landed |

## API Mapping

| Block operation | Type | Function | File:Line | Template Params / Args | Input CB | Output CB | Which params are block knobs |
|-----------------|------|----------|-----------|------------------------|----------|-----------|------------------------------|
| `make_fill_tile`, `mask_block` | helper | `dataflow_kernel_lib::fill_l1_range<val_size>(start_addr, n_bytes, val)` | `ttnn/cpp/ttnn/kernel_lib/l1_helpers.hpp:100-101` | `val_size = elem_bytes`; `val = fill_bits` | — / `cb_work_tiles` | `cb_fill_tile` / `cb_work_tiles` | none (range lengths fixed by `r0`, `c0`, `elem_bytes`) |
| `load_block` | raw_api | `TensorAccessor` + `noc_async_read_page` / `noc_async_read_tile`, `noc_async_read_barrier` | `tt_metal/hw/inc/api/tensor/tensor_accessor.h:71`, `tt_metal/hw/inc/api/dataflow/dataflow_api.h:1243`, `:1302`, `:1921` | accessor CT args from `ttnn.TensorAccessorArgs(input)` | — | `cb_work_tiles` | `block_tiles` (reads per barrier) |
| `store_block`, `drain_writes` | raw_api | `TensorAccessor` + `noc_async_write_page`, `noc_async_writes_flushed`, `noc_async_write_barrier` | `dataflow_api.h:1418`, `:1980`, `:1951` | accessor CT args from `ttnn.TensorAccessorArgs(output)` (extension: the input) | `cb_work_tiles`, `cb_fill_tile` | — | `block_tiles` (writes per flush) |

Helpers considered and rejected for the raw dataflow rows:
- `dataflow_kernel_lib` mcast (`mcast/kernel/mcast_args.hpp`): no cross-core data in any built regime (regime H rejected).
- `zero_tile` / `prepare_zero_tile` (`ttnn/cpp/ttnn/kernel_lib/l1_helpers.hpp:53`, `:67`): zeros only; `c` is per-call data (−1, −2^127, 0.1→bf16, 7.0 in the suite). `fill_l1_range` covers zero too.
- `Noc::async_write_zeros` (`tt_metal/hw/inc/api/dataflow/noc.h:849`): zeros only, same reason.
- compute helpers (`eltwise_convenience.hpp`, `fill.h:29/51/81`): would route values through unpack/DEST/pack (regime G, rejected for bit equality); the op has no compute phase.
- No kernel_lib helper wraps interleaved page reads/writes for a block; the per-block batching (one barrier/flush per block) is the block operation.

`compute_kernel_hw_startup()`: not applicable — the program has no compute kernel.

## Host-side contract (entry point, op file)

| Item | Decision |
|---|---|
| First line | `validate(input_tensor, padding, value)` |
| validate order | (1) argument form → `ValueError` (padding sequence of 1..r pairs of ints, low == 0, high ≥ 0; `value` int/float not bool; integer range for int32/uint32; sharded input; non-device tensor); (2) build scenario dict `{shape, padding, value, dtype, layout, memory_config: "dram"/"l1"}`; (3) taggers → SUPPORTED per axis → `UnsupportedAxisValue`; (4) EXCLUSIONS (none) → `ExcludedCell` |
| INPUT_TAGGERS | `tag_alignment` (W%32 → "w_non_aligned"; else H%32 → "h_non_aligned"; else "tile_aligned"), `tag_rank`, `tag_padded_dims` (highs of −2/−1 after trailing alignment: both → "hw", −2 → "h", −1 → "w", else "leading"), `tag_dtype`, `tag_layout`, `tag_memory_config` |
| SUPPORTED (Phase 0) | dtype [bfloat16], layout [TILE], memory_config ["dram"], alignment ["w_non_aligned","h_non_aligned"], rank [3,4], padded_dims ["h","hw"] |
| EXCLUSIONS | none |
| Output | extension: `generic_op([x], desc)` then the pinned view; new_storage: allocate, `generic_op([x, out], desc)` |
| Program-description cache (BB-07) | module-level dict keyed by `(shape, padded_shape, highs, dtype, layout, buffer_type, grid, regime)` → ProgramDescriptor; on a hit only RT values are rewritten (buffer addresses, `fill_bits`); geometry (`d`, `D`, `first_tile_row`, `r0`, `c0`, `block_tiles`, `elem_bytes`, `has_fill_only_tiles`) is CT, so the generic_op hash (`generic_op_device_operation.cpp:79-110`, CT args hashed, RT values overridden on hit) separates calls with different padding while `value` never enters a cached program |
| Plan report (BB-08) | `g6_pad.last_plan` (or a `plan(input, padding)` function) returns `{storage_class, regime, num_cores, block_tiles, buffer_depth, num_work_tiles}` |
| Profiler zones (BB-21) | `DeviceZoneScopedN` around `make_fill_tile`, each `load_block` (incl. its barrier), `mask_block`, the `cb_wait_front` in `store_block`, the writes+flush, `drain_writes`; zones are compiled out of the plain build |
| No host transformation | no to_layout / tilize / untilize / pad / slice / typecast / to_memory_config; the extension view is the only shape change |

### Structural impossibilities

None beyond `feature_spec.py`'s (`INVALID = []`).

## Key Risks and Gotchas

| Risk | Why it bites here | Mitigation in this design |
|------|-------------------|---------------------------|
| View route dispatches a program (R10) | `ttnn.reshape` has routed / codegen paths; one taken = 2 programs per call | pinned route traced on paper (above); **first implementation step**: a `tt-probe` running every Phase 0 extension shape under `ttnn.graph.begin_graph_capture` asserting zero `program_factory_type` nodes for the view and `out.buffer_address() == x.buffer_address()`, `out.padded_shape == x.padded_shape` |
| Input tile padding leaks (R1) | poisoned padding in loose cases; every A tile and every B boundary tile carries x's padding | mask covers rows `[r0,32)` and cols `[c0,32)` wholly; masks applied *after* the read barrier of the block, *before* the push |
| Extension touches other tile rows (R7) | `untouched_rows` loose case compares raw buffer rows | `first_tile_row = Ht_out-1`; work set is exactly the last tile row of every slice; reads and writes are both indexed by `out_page` of that set |
| In-place hazard | x is both source and destination | each tile owned by one core; the writer writes a tile only after the reader's barrier for its block and the push; no other core touches it |
| Ragged blocks and CB ring | group-2 cores and last blocks are shorter | push/pop nominal `block_tiles`; only NoC transfers narrow |
| Fill-value cache (R4, BB-07) | case 253 run with value 0 then 7.0 | `fill_bits` RT, rewritten on every call including cache hits |
| bf16 fill conversion | 0.1 is inexact; −2^127 must stay finite | host converts with torch to the dtype and views the bits; no float math on device |
| fill-only CB allocation | extension calls never need `cb_fill_tile` | allocate iff `has_fill_only_tiles` (any output tile with `!has_input`), CT flag so the writer skips `make_fill_tile` |
| Write completion (BB-03) | another program starts right after | `noc_async_write_barrier()` at writer end; reader ends with its last read barrier |
