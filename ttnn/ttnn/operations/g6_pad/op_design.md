# Operation Design: g6_pad

## Overview

| Field | Value |
|-------|-------|
| Classification | data_movement — no arithmetic, no compute kernel, no unpack/pack anywhere |
| Goal | Constant padding at the high end of the trailing dims, bit-identical, one device program per call. A `tile_padding_extension` call writes `c` into the new logical rows of the last tile row of each slice of x **without reading anything** and returns a metadata-only view of x's buffer. A `new_storage` call writes every output tile once, reads every input tile at most once, and never reads a fill-only page. |
| Math | `D_i = d_i + h_i`; `out[j] = x[j]` if `j_i < d_i` for every `i`, else `c`; `c` = `value` converted to x's dtype the way torch fills a tensor of that dtype |
| References | contract `eval/prompts/g6_pad.txt`; golden suite `eval/golden_tests/g6_pad/{feature_spec.py,helpers.py}` (oracle, `storage_class`); NoC alignment `tt_metal/hw/inc/internal/tt-1xx/blackhole/noc/noc_parameters.h:375-380`, `.../wormhole/noc/noc_parameters.h:291-296`; view route `ttnn/cpp/ttnn/operations/data_movement/reshape_view/reshape.cpp:760-791`; catalog `ttnn/ttnn/operations/examples/{width_split,noc_placement,double_buffer,tile_reorder,split_reader,bank_stagger}` |

## Parameters

| Name | Type | Required | Valid Range | Default | CT/RT |
|------|------|----------|-------------|---------|-------|
| `input_tensor` | `ttnn.Tensor` | yes | on device, rank 2–4, interleaved (DRAM or L1); sharded → `ValueError` | — | buffer address: common RT arg |
| `padding` | list/tuple of `k` 2-element lists/tuples of `int` (not `bool`) | yes, positional or keyword | `1 <= k <= rank`; every pair `(0, h)`, `h >= 0`; anything else → `ValueError` | — | geometry derived from it: CT (part of the program hash) |
| `value` | `int` or `float` (not `bool`) | no, positional or keyword | int32/uint32: finite, `trunc(value)` inside the dtype range, else `ValueError` | `0.0` | bit pattern `fill_bits`: common RT arg (never in the program hash) |

Padding canonicalisation (host, first thing after the argument checks): pairs align to the trailing
dims — `highs = [0] * (rank - k) + [h for (_, h) in padding]`. The torch flat list is never formed.

`fill_bits` (host scalar, no device op):

| dtype | Conversion | Bits |
|---|---|---|
| bfloat16 | `torch.full((1,), value, dtype=torch.bfloat16)` (torch's own conversion, RNE through float32) viewed as `int16` | low 16 bits of a uint32 RT arg; `-1.7014118346046923e38` → `0xFF00` (−2^127, finite); `0.1` → `0x3DCD` |
| float32 (refinement) | `torch.full((1,), value, dtype=torch.float32)` viewed as `int32` | 32 bits |
| int32 / uint32 (refinement) | `c = math.trunc(value)`, range-checked, `c & 0xFFFFFFFF` | 32 bits; no float detour |

## Tensors

### Input

| Property | Requirement |
|----------|-------------|
| Shape | `(d_0, …, d_{r-1})`, `r ∈ {2,3,4}`, any size per dim |
| Dtype | bfloat16 (Phase 0); float32, int32, uint32 (refinements) |
| Layout | TILE (Phase 0); ROW_MAJOR (refinement, regime R) |
| Memory | DRAM interleaved (Phase 0); L1 interleaved (refinement); sharded refused (`ValueError`) |

### Output

| Property | Value |
|----------|-------|
| Shape | `(d_0+h_0, …, d_{r-1}+h_{r-1})` |
| Dtype | x's dtype |
| Layout | x's layout |
| Memory | x's memory config. `tile_padding_extension`: x's own buffer, viewed at the output logical shape with x's padded shape. `new_storage`: `ttnn.allocate_tensor_on_device(ttnn.Shape(out_shape), x.dtype, x.layout, device, x.memory_config())` (positional args) |

### Storage class (host predicate, shapes alone — identical to `helpers.storage_class`)

```
extension = layout == TILE
            and highs[-2] > 0 and all(h == 0 for i, h in enumerate(highs) if i != rank - 2)
            and ceil(d_{r-2} / 32) == ceil(D_{r-2} / 32)
```

Every other call is `new_storage`. `extension` implies `d_{r-2} % 32 != 0`, so the last tile row of
every slice holds logical rows `0 .. in_row_rem-1` and the new rows are `in_row_rem .. out_row_end-1`
of that same tile row (symbols in the Blocking Model).

### The extension view (pinned route — launches no device program, moves no data)

```
generic_op([input_tensor], extension_descriptor)       # the one device program; writes x's buffer in place
out = ttnn.reshape(input_tensor, ttnn.Shape(out_shape), input_tensor.padded_shape)
```

| Step in `reshape_dispatch` | File:line | Outcome for an extension call |
|---|---|---|
| no-op / zero-volume / host / same-shape gates | `reshape.cpp:636-688` | not taken (logical shape differs, device tensor) |
| ND-sharded paths | `reshape.cpp:700-757` | not taken (interleaved) |
| `this_is_view` | `reshape.cpp:760-766` | false: dim −2 changes and `d_{r-2} % 32 != 0` |
| routed codegen plan | `reshape.cpp:772-776` → `codegen/reshape_codegen_supported.cpp:402-403` | refused: "explicit padded shape or volume change" |
| volume-change branch | `reshape.cpp:778-788` | `tile_tensor_view_reshape_possible` (TILE, padded dims −2/−1 multiples of 32, padded last dim equal) → `ttnn::experimental::view(tensor, logical, padded)` |
| `tt::tt_metal::view` | `ttnn/core/tensor/tensor_ops.cpp:533-546`, `:466-478` | new `Tensor` on a `MeshBuffer` created at x's address; graph tracker records `Tensor::reshape` without `program_factory_type` (not counted as a program) |

Confirmed on device (Blackhole, probe under graph capture) for `[1,100,1536]→[1,128,1536]`,
`[1,1,8,62080]→[1,1,16,62080]` and `[16,1,1536]→[16,32,1536]`: zero device programs,
`out.buffer_address() == x.buffer_address()`, `out.padded_shape == x.padded_shape`, x's logical shape
unchanged. The acceptance test re-checks the one-program property on every call.
`generic_op` with a single in-place io tensor is legal (`generic_op_device_operation.cpp:330-341`).

## Blocking Model

### Axes

The op's output tile index space has three axis groups — slice (leading dims flattened), tile row
(dim −2), tile column (dim −1) — plus the intra-tile 32×32. **Every axis is independent**: an output
tile depends on at most the input tile at the same coordinates (or on nothing), so no result spans
any axis and no split needs a combine. The fill constant `c` is the one operand shared by every
padded position; it is made per core from a 32-bit RT scalar, so it is reuse-shared with nothing to
move.

The three tile-index groups are **folded into one work axis** `w`, tile column fastest, over the
**touched tile set** of the regime. Reasons: all three are independent and treated identically; an
interleaved tile is its own NoC page on its own bank, so a 2-D rectangle of tiles has no
transfer-shape advantage over a run of equal length; and the fold is what keeps every INPUTS shape on
the grid (wide-short p2–p4 have one touched tile row of 1940 tiles; tall shapes have 4–80 tile
columns).

Geometry (per image, `ceil`, never a flattened `floor`):

| Symbol | Definition |
|---|---|
| `out_tensor_slices`, `in_tensor_slices` | `prod(D_0..D_{r-3})`, `prod(d_0..d_{r-3})`; rank 2 → 1 |
| `out_tensor_row_tiles`, `in_tensor_row_tiles` | `ceil(D_{r-2}/32)`, `ceil(d_{r-2}/32)` |
| `out_tensor_col_tiles`, `in_tensor_col_tiles` | `ceil(D_{r-1}/32)`, `ceil(d_{r-1}/32)` |
| `in_row_rem`, `in_col_rem` | `d_{r-2} % 32`, `d_{r-1} % 32` (0 = aligned) |
| `out_row_end` | `D_{r-2} % 32`, or 32 when that is 0 (extension: exclusive end of the new rows inside the last tile row) |
| `elem_bytes`, `tile_bytes` | 2 / 2048 (bf16), 4 / 4096 (fp32, int32, uint32) |

Touched tile set per regime:

| Regime | `w` enumerates | `num_work_tiles` | Output page of `w` |
|---|---|---|---|
| E (extension) | `(s, tc)`, `s < out_tensor_slices`, `tc < out_tensor_col_tiles`; the tile row is fixed to the last one | `out_tensor_slices * out_tensor_col_tiles` | `(s*out_tensor_row_tiles + out_tensor_row_tiles-1)*out_tensor_col_tiles + tc` (equal shapes in and out) |
| N (new storage) | every output tile `(s, tr, tc)` | `out_tensor_slices * out_tensor_row_tiles * out_tensor_col_tiles` | `w` itself |

| Axis | Character (+ one-clause reason) | Extent knob | Phase 0 value | Knob source | Core-assignment | Later unlock |
|------|--------------------------------|-------------|---------------|-------------|-----------------|--------------|
| slice (leading dims) | independent — an output tile of slice `s` reads only slice `s` of x, or nothing when `s` lies in leading padding | folded into `block_tiles` (a block may cross slices) | folded | `block_tiles` | spread: part of the folded run | knob-turn (leading padding = fill-only slices, already in the per-tile class) |
| tile row (dim −2) | independent — no result spans rows | folded; E touches only the last row of each slice (a decision fixed by the contract, not a collapsed extent) | E: 1 row per slice; N: all rows | `regime` (host) | spread: part of the folded run | knob-turn |
| tile column (dim −1) | independent — no result spans columns | folded into `block_tiles` | folded | `block_tiles` | spread: part of the folded run; fills the grid on p2–p4 (catalog `width_split`) | knob-turn |
| **work tile `w`** (the fold) | independent | `block_tiles` | E: `max_core_work_tiles` (the whole run, one block per core; a group-2 core's block is one shorter). N: `min(max_core_work_tiles, block_tiles_cap)` — the whole run in every INPUTS shape | host, computed once per descriptor; CT arg | `ttnn.split_work_to_cores(grid, num_work_tiles, row_wise=True)`; contiguous `w` runs on a row-major prefix of the grid | knob-turn |
| intra-tile (32×32) | independent — elementwise | whole tile in N (a tile is the page, never split); in E the sub-tile write segments are fixed by `in_row_rem`/`out_row_end`, not a knob | 32×32 | layout | not split: one tile is one NoC page | — |
| fill constant `c` | reuse-shared — the same bits in every padded position | one fill page per core | 1 page | common RT arg `fill_bits` | every core builds its own; nothing moves | — (multicast rejected, regime table) |

Derived per call (single source each; everything else is computed from these):

| Symbol | Definition | Source |
|---|---|---|
| `grid` | `device.compute_with_storage_grid_size()` | host; part of the descriptor-cache key |
| `min_core_work_tiles` | Phase 0 = 1 | host constant (grid-sync lamp) |
| `num_cores` | `min(grid.x*grid.y, ceil(num_work_tiles / min_core_work_tiles))` | host |
| `max_core_work_tiles` | `ceil(num_work_tiles / num_cores)` (core group 1 of the split) | host |
| `core_work_begin`, `core_work_tiles` | this core's run from the split | host → per-core RT args |
| `block_tiles_cap` (N) | `floor((L1_OP_BUDGET - tile_bytes) / (2 * tile_bytes))`, `L1_OP_BUDGET = 1347776` | host constant expression |
| `block_tiles` | E: `max_core_work_tiles`; N: `min(max_core_work_tiles, block_tiles_cap)` | host → CT arg |
| `num_blocks_this_core` | `ceil(core_work_tiles / block_tiles)`; E: always 1 | host → per-core RT arg |
| `last_block_extent` | `core_work_tiles - (num_blocks_this_core - 1) * block_tiles` | kernel, from the two above |
| `buffer_depth` (N) | `2 if ceil(max_core_work_tiles / block_tiles) > 1 else 1` | host → CB size only |

### Buffer-depth knobs

| CB | Depth knob | Phase 0 value | What the depth buys |
|----|------------|---------------|---------------------|
| `cb_work_tiles` (N) | `buffer_depth` | 1 in every INPUTS shape (one block per core); 2 as soon as the L1 clamp gives a core more than one block | depth 2 lets the reader fill block `k+1` while the writer drains block `k`; with one block per core a second slot is dead L1 |
| `cb_fill_tile` (E, N) | none — constant | 1 | built once, read-only afterwards |

### Mechanism caps

| Mechanism | Cap on which extent | Clamp | What happens unclamped |
|-----------|--------------------|-------|------------------------|
| Model L1 budget (BB-05: 1347776 B per core) | N: `block_tiles * buffer_depth` | `block_tiles <= block_tiles_cap`; longer runs stream as more blocks | CB allocation past the model's free L1: launch failure inside the model only |
| CB ring wrap (pointers reset only on an exact `fifo_limit` hit) | N push/pop quantum | reader pushes and writer pops exactly `block_tiles` per block, ragged last block included; only the NoC transfers narrow to `last_block_extent` | second block of a ragged core walks past the CB into neighbouring L1 |
| NoC write alignment to DRAM and L1: 16 B on both archs (`blackhole/noc_parameters.h:376,380`, `wormhole/noc_parameters.h:292,296`) | E sub-tile write segments | segment offsets and lengths are multiples of one face row (`16 * elem_bytes` ≥ 32 B) by construction; host asserts `offset % 16 == 0 and length % 16 == 0` | misaligned write: corrupted or dropped bytes |
| NoC DRAM read alignment: 64 B BH (`blackhole/noc_parameters.h:379`), 32 B WH (`:295`) | N reads | N reads whole tiles only (page-aligned starts); no sub-tile read in any built regime | — (a sub-tile read lamp must keep 64-B-aligned source offsets) |
| `fill_l1_range<val_size>` (`ttnn/cpp/ttnn/kernel_lib/l1_helpers.hpp:96-101`): start and length multiples of `val_size`, `val_size ∈ {1,2,4}` | N mask ranges, fill-page build | `val_size = elem_bytes`; every range is whole elements by construction | head/tail elements silently under-filled → x's tile padding leaks into a logical output position |
| One NoC page per tile, one bank per page | every transfer | a transfer never spans two pages; E computes the page's NoC address once and issues its ≤3 segments inside it | a transfer crossing a page boundary writes the wrong bank |
| `fill_bits` width | `c` | 32-bit common RT arg; bf16 uses the low 16 bits | — |
| Tile index width | `num_work_tiles`, page ids | uint32; INPUTS max 1940 (E), 1536 (N, p1) | — |

### Regimes

Minimum, named against the DRAM (or L1-interleaved) boundary: *every input byte that a logical
output position needs crosses at most once, every output byte that must change is written once,
nothing else is read, and no fill-only page is read at all.*

| Regime | Status | Predicate | Block | Data movement vs. minimum | What a bigger block buys |
|--------|--------|-----------|-------|---------------------------|--------------------------|
| **E. extension, write-only new rows, in place** | built | `extension` | `block_tiles = max_core_work_tiles` touched tiles (last tile row of each slice), one block per core | **the minimum**: zero reads; per touched tile writes rows `in_row_rem .. out_row_end-1` × all 32 tile columns as ≤3 face-row segments from the core's fill page. p2: 3,724,800 B written vs 7,946,240 B moved by read-mask-write; p3: 993,280 B vs 7,946,240 B. The only bytes above the strict minimum are the column padding of the last tile column inside those rows (output tile padding, written to keep one segment per face run) | one final write barrier per block (intended: once per kernel); fill page built once per kernel; page NoC address computed once per tile; no CB handshake at all |
| **N. new storage, one pass, in-L1 assembly** | built | `not extension and layout == TILE` | `block_tiles` work tiles of `out_tensor_slices × out_tensor_row_tiles × out_tensor_col_tiles` | minimum on pages: copy and boundary tiles read once and written once; fill-only tiles never read, written once from the fill page. Above the byte minimum: a boundary tile reads its input tile padding rows/columns (whole-tile read; a sub-tile read would split one transfer into 2–34) | per block: one `cb_reserve_back` + one read barrier + one `cb_push_back` (reader), one `cb_wait_front` + one `noc_async_writes_flushed` + one `cb_pop_front` (writer); fill page and mask byte ranges once per kernel |
| R. ROW_MAJOR, stick pages | deferred — refinement of the `layout` axis (no Phase 0 cell is ROW_MAJOR). Reuses N's skeleton unchanged: folded output-page index, per-page class (copy / boundary / fill-only), reader → `cb_work_tiles` → writer, fill page; only the page becomes one stick (`D_{r-1} * elem_bytes`) and the boundary action fills bytes `[d_{r-1}*e, D_{r-1}*e)` of the stick in L1. Needs its own handling of sticks below the DRAM read alignment (4 B and 32 B rows on a 64-B stride) | `layout == ROW_MAJOR` | `block_sticks` | minimum (same one-pass assembly) | same handshakes per block of sticks |
| A. extension, read-mask-write of the last tile row | rejected — superseded by E on every extension call: E needs no read, no reader kernel, no CB, no mask, and moves 2–48 % of A's bytes (p3: 12.5 %, p2: 47 %); nothing of A survives in E (dead end) | — | — | +1 whole-tile read and the write of the rows outside `[in_row_rem, out_row_end)` per touched tile | — |
| F. extension copied into new storage | rejected — forbidden by the contract for multi-row slices; for one-row slices superseded by E (in place, no read) | — | — | moves every tile row (121× the touched bytes on `[1,3844,C]`) | — |
| P. two passes: fill the output, then copy x over it | rejected — forbidden (second full-tensor DRAM pass); superseded by N's in-L1 assembly | — | — | output written twice, fill-only pages written then overwritten | — |
| G. masking on the compute engine (unpack → select → pack) | rejected — any unpack/pack can convert a value (int32/uint32, fp32 specials) and costs a compute kernel; superseded by N's raw-byte RISC-V mask | — | — | same DRAM traffic plus an L1 round trip through SRC/DEST | — |
| M. read once and multicast the fill page | rejected — `c` is generated locally from a 32-bit RT arg; a multicast adds NoC traffic and a semaphore round trip to save nothing | — | — | — | — |
| D. split a dependent axis + cross-core combine | rejected — the op has no dependent axis (every output tile depends on at most one input tile) | — | — | — | — |

**Regime selection function** (host, exact):
`regime = "E" if extension else ("N" if layout == TILE else "R")`. It is a pure function of the
shapes and the layout, independent of the device grid. Regime-pinned tests: the acceptance test
carries E on one-row slices, multi-row slices, wide-short slices and batched slices, and N with `h`
and `hw` padding (boundary rows, boundary columns, corner, fill-only rows and columns).

### Traffic ranking

| Split of the touched tile set | DRAM crossings | Cross-core traffic | Occupancy on INPUTS |
|---|---|---|---|
| folded work-tile run over the full grid (**chosen**) | the regime minimum | none | fills the grid on wide-short (p2–p4: 1 × 1940), batched (`[16,1,C]`: 16 × 48) and tall N shapes (`[1,256,128]`: 32 tiles) alike |
| tile-row split | same bytes | none | E touches one tile row per slice → `out_tensor_slices` cores; p2–p4 run on **one core** (catalog `width_split`: up to 7.8× slower) |
| slice split | same bytes | none | `out_tensor_slices = 1` for most cases → one core |
| tile-column split | same bytes | none | ties on p2–p4; N `[1,256,128]` has 4 tile columns → 4 cores |
| dependent-axis split + combine | — | — | no dependent axis |

Every split moves identical bytes (no shared operand moves; `c` is made per core), so the split
ranking is a tie and occupancy breaks it. Core placement: `row_wise=True` (catalog `noc_placement`:
a column line of DRAM-facing cores is ~2.9× slower than a row); writes on NoC1, reads on NoC0 (the
default writer/reader configs; catalog `noc_placement`: 2.5–4.8× better than the reverse).

The cheapest-traffic **regime** decision is not a tie: E (zero reads, ≤3 sub-tile writes per touched
tile) beats read-mask-write A on bytes and on latency (no read round trip on the critical path) and
is legal for every `in_row_rem` because NoC writes need only 16-B alignment on both architectures.
Catalog `tile_reorder` (whole tiles beat face-sized writes) compares equal bytes moved in fewer
transactions; it does not apply to E, which moves fewer bytes in as many transactions as A.

### Stall-shadow check

| Waiting stage | Waits on | Work independent of the wait | Decision |
|---|---|---|---|
| N writer, `cb_wait_front(cb_work_tiles)` | the reader's reads of the block | the block's **fill-only** tiles (they depend on nothing but the fill page) | **reorder**: per block, the writer writes every fill-only tile first, then waits for the block and writes the input-backed tiles. Pure data movement, no floating-point order involved |
| N reader, `noc_async_read_barrier()` | its own reads | none: the mask overwrites part of the slot the read is landing in, so it must follow the barrier | nothing to schedule (state in kernel: mask strictly after the barrier) |
| E writer, final `noc_async_write_barrier()` | write acks | none left: every write of the core is already issued | nothing to schedule |
| E writer, first segment write | the fill page being built | none: every write sources the fill page | nothing to schedule (fill only the segment byte ranges, ≤1984 B bf16) |

### Block schedule

**Regime E** — one kernel per used core: `extension_writer` (BRISC, NoC1). No reader, no compute.

```cpp
build_fill_segments();                               // once per kernel
for (uint32_t block_idx = 0; block_idx < num_blocks_this_core; ++block_idx) {   // = 1
    fill_new_rows_block(block_idx);
}
drain_writes();                                      // once per kernel (BB-03)
```

**Regime N** — two kernels per used core: `new_storage_reader` (NCRISC, NoC0) and
`new_storage_writer` (BRISC, NoC1). No compute.

```cpp
build_fill_page();                                   // writer, once per kernel, only if this core has fill-only tiles
for (uint32_t block_idx = 0; block_idx < num_blocks_this_core; ++block_idx) {
    load_block(block_idx);                           // reader
    assemble_block(block_idx);                       // reader
    store_fill_tiles_block(block_idx);               // writer (stall shadow: before the wait)
    store_input_tiles_block(block_idx);              // writer
}
drain_writes();                                      // writer, once per kernel (BB-03)
```

| Block operation | Kernel | Block shape | Acts on | Resident across it | Intended frequency of fixed costs | Zone (BB-21) |
|---|---|---|---|---|---|---|
| `build_fill_segments` | E writer | the ≤3 segment byte ranges of one tile | `cb_fill_tile`: bytes of every segment = `fill_bits` (`elem_bytes` wide) | `cb_fill_tile` for the whole kernel | once per kernel | `G6PAD_FILL_BUILD` |
| `fill_new_rows_block` | E writer | `block_tiles` touched tiles (ragged: `core_work_tiles`) | for each tile: page NoC address once, then one write per segment from `cb_fill_tile + seg_offset` to `page + seg_offset`, `seg_len` bytes | `cb_fill_tile` | ≤3 write issues per tile; **no** barrier inside the block | `G6PAD_FILL_ROWS` |
| `build_fill_page` | N writer | 1 tile | `cb_fill_tile`: every element = `fill_bits` | `cb_fill_tile` for the whole kernel | once per kernel; skipped when RT `core_has_fill_only == 0` | `G6PAD_FILL_BUILD` |
| `load_block` | N reader | `block_tiles` work tiles (ragged: `last_block_extent`) | reserve `block_tiles` slots; slot `i` ↔ block tile `i`; one whole-tile read per input-backed tile into its slot; fill-only slots untouched | — | one `cb_reserve_back(block_tiles)`, one `noc_async_read_barrier` per block | `G6PAD_LOAD` (barrier inside its own `G6PAD_READ_WAIT`) |
| `assemble_block` | N reader | same block | for each boundary slot, write `c` over the byte ranges that lie in x's tile padding (row ranges, column ranges below); copy slots untouched; then `cb_push_back(block_tiles)` | mask byte ranges (computed once per kernel from CT args) | one push per block | `G6PAD_ASSEMBLE` |
| `store_fill_tiles_block` | N writer | same block | one whole-tile write from `cb_fill_tile` to `out_page(w)` for every fill-only tile of the block | `cb_fill_tile` | one write per fill-only tile; no CB handshake | `G6PAD_STORE_FILL` |
| `store_input_tiles_block` | N writer | same block | `cb_wait_front(block_tiles)`; one whole-tile write from slot `i` for every input-backed tile; `noc_async_writes_flushed`; `cb_pop_front(block_tiles)` | — | one wait, one flush, one pop per block | `G6PAD_STORE` (wait inside `G6PAD_WRITE_WAIT`) |
| `drain_writes` | writer (E, N) | — | `noc_async_write_barrier()` | — | once per kernel | `G6PAD_DRAIN` |

Both N kernels classify tiles identically from the same CT args and their per-core RT range;
counters `(s, tr, tc)` and leading coordinates advance incrementally (no division per tile; one
decomposition of `core_work_begin` per kernel).

| Quantity | Definition (N) |
|---|---|
| output coords of `w` | `tc = w % out_tensor_col_tiles`, `tr = (w / out_tensor_col_tiles) % out_tensor_row_tiles`, `s = w / (out_tensor_row_tiles * out_tensor_col_tiles)` |
| slice mapping | decompose `s` over the output leading dims `(D_0..D_{r-3})`; `slice_exists` iff every leading coord `< d_i`; `s_in` = the same coords linearised over `(d_0..d_{r-3})`. Phase 0 (no leading padding): `s_in = s`, always exists |
| `has_input` | `slice_exists and tr < in_tensor_row_tiles and tc < in_tensor_col_tiles` |
| `in_page` | `(s_in*in_tensor_row_tiles + tr)*in_tensor_col_tiles + tc` |
| `row_mask` | `has_input and tr == in_tensor_row_tiles-1 and in_row_rem != 0 and D_{r-2} > d_{r-2}` |
| `col_mask` | `has_input and tc == in_tensor_col_tiles-1 and in_col_rem != 0 and D_{r-1} > d_{r-1}` |
| class | fill-only = `!has_input`; boundary = `row_mask or col_mask`; copy otherwise |

When `D_{r-2} == d_{r-2}` the input's padding rows are also output padding (same tile-row count), so
no row mask is needed there; the same holds for columns.

Tile byte layout (`e = elem_bytes`): faces of 16×16 in order face0 (rows 0–15, cols 0–15), face1
(rows 0–15, cols 16–31), face2 (rows 16–31, cols 0–15), face3 (rows 16–31, cols 16–31); face row =
`16e` B, face = `256e` B; element `(r, c)` of a face at `(r*16 + c)*e`.

**E write segments** — rows `[a, b) = [in_row_rem, out_row_end)`, all 32 columns, computed on the host
once per call and passed as CT args `(num_segments, off_0, len_0, …, off_2, len_2)`:

| Case | Segments (byte offsets inside the tile) | Production instance (bf16) |
|---|---|---|
| `a < 16`, `b <= 16` | face0 `[16e·a, 16e·b)`; face1 `[256e + 16e·a, 256e + 16e·b)` | p3 (`a=8, b=16`): `[256,512)`, `[768,1024)` → 2 × 256 B |
| `a < 16 < b < 32` | face0 `[16e·a, 256e)`; face1+face2 `[256e + 16e·a, 512e + 16e·(b-16))`; face3 `[768e, 768e + 16e·(b-16))` | — |
| `a < 16`, `b = 32` | face0 `[16e·a, 256e)`; face1+face2+face3 `[256e + 16e·a, 1024e)` | p2 (`a=2`): `[64,512)`, `[576,2048)`; p4 (`a=8`): `[256,512)`, `[768,2048)` |
| `a >= 16`, except `a = 16, b = 32` | face2 `[512e + 16e·(a-16), 512e + 16e·(b-16))`; face3 `[768e + 16e·(a-16), 768e + 16e·(b-16))` | — |
| `a = 16`, `b = 32` | faces 2+3 `[512e, 1024e)` | case 257: `[1024,2048)` |

Rows `out_row_end .. 31` (output tile padding) are not written: they keep whatever x's producer left,
which the contract allows. Rows `0 .. in_row_rem-1` (logical values of x) are never touched.

**N mask ranges** — filled with `c` in a boundary slot; rows/columns beyond the output's logical box
are output padding and may take `c` too:

| Mask | Condition | Byte ranges inside the tile |
|---|---|---|
| row, `in_row_rem < 16` | `row_mask` | face0 `[16e·in_row_rem, 256e)`; face1+2+3 `[256e + 16e·in_row_rem, 1024e)` |
| row, `in_row_rem >= 16` | `row_mask` | face2 `[512e + 16e·(in_row_rem-16), 768e)`; face3 `[768e + 16e·(in_row_rem-16), 1024e)` |
| col, `in_col_rem < 16` | `col_mask` | faces 1 and 3 whole (`[256e,512e)`, `[768e,1024e)`); in each of the 32 face rows of faces 0 and 2: `[row_base + e·in_col_rem, row_base + 16e)` |
| col, `in_col_rem >= 16` | `col_mask` | in each of the 32 face rows of faces 1 and 3: `[row_base + e·(in_col_rem-16), row_base + 16e)` |

### Perf lamps

| Lamp | Why the default may be wrong here | Nearby alternative to measure |
|------|-----------------------------------|-------------------------------|
| Overlap (N) | one block per core serialises each core's reads against its writes: the writer drains only after the whole block is read and assembled | `block_tiles ∈ {4, 8}` with `buffer_depth = 2` (catalog `double_buffer`: sweet spot 4–8, no gain once DRAM-bound) on p1 and case 65 |
| Grid synchronization | `min_core_work_tiles = 1` puts 1 tile on each of 48 cores (case 247, E: 2 writes per core), 64 cores (case 65, N) — per-core launch and fixed cost may dominate | `min_core_work_tiles ∈ {2, 4, 8}` on cases 247, 254 (768 E tiles), 65 |
| E issue rate (both DM RISCs) | E leaves NCRISC idle; each BRISC issues 2–3 writes per tile (~36–54 per core on p2–p4 at 110 cores) and may be issue-bound | split each core's run between BRISC/NoC1 and NCRISC/NoC0, each building its own fill segments (catalog `split_reader`; `noc_placement` warns writes on NoC0 can be slower) on p2–p4 |
| N class balance (p1) | equal-tile contiguous runs put the 960 copy tiles (read + write) on the first ~69 of 110 cores and the 576 fill-only tiles (write only) on the rest: up to 2× byte imbalance, and copy-only cores have no fill work to hide their read wait | split input-backed tiles and fill-only tiles separately over all cores (two runs per core: one over the input tile space, one over the fill boxes) on p1 |
| N mask realization | RISC-V stores (`fill_l1_range`, ~1.5 KB per bf16 row-boundary tile, 34 ranges per column-boundary tile) sit on the reader's critical path | (a) local NoC L1→L1 copies from a reader-side fill page; (b) row-only boundary tiles: read only the logical face rows (2 reads, 64-B-aligned starts) into a slot pre-filled with `c` — on cases 65, 69, 72 |
| Bank stagger (N p1) | runs start at `core_work_begin = k · max_core_work_tiles`; on p1 at 110 cores that is `14k`, so the cores' first pages fall on half of the 8 BH DRAM banks | catalog `bank_stagger`: rotate each core's walk to start at `core % nbanks` within its run, on p1 |

## Dataflow Strategy

| Stage | Format | Mechanism | Notes |
|-------|--------|-----------|-------|
| host | — | Python | `validate()` first; storage class; descriptor from cache; one `generic_op`; E: view after dispatch |
| E: fill segments → x's last tile row | raw bytes, x's dtype | `extension_writer`: `TensorAccessor(x)`; per tile `get_noc_addr(page)` once, then `noc_async_write(fill + off_k, page_addr + off_k, len_k)`; one barrier at the end | zero reads; in place; only the last tile row of each slice is written |
| N: x → `cb_work_tiles` | raw tile bytes | `new_storage_reader`: `TensorAccessor(x)` + whole-tile `noc_async_read_page` per input-backed tile; one barrier per block | no format conversion anywhere |
| N: in-slot assembly | raw bytes | reader: `fill_l1_range<elem_bytes>` over the mask ranges after the barrier | the page is assembled in L1 before it is written (the allowed one-pass form) |
| N: `cb_work_tiles` / `cb_fill_tile` → out | raw bytes | `new_storage_writer`: `TensorAccessor(out)` + whole-tile `noc_async_write_page`; flush per block, barrier at the end | fill-only pages never read from DRAM |

No inter-Tensix communication in any regime (no semaphores, no multicast).

TARGET axes seen through the regimes:

| TARGET axis value | Cost class | Why |
|---|---|---|
| dtype float32 / int32 / uint32 | knob-turn | `elem_bytes` CT arg; tile bytes from `ttnn.tile_size(dtype)`; segment offsets scale with `16e` (64 B multiples, still 16-B aligned); raw-byte path is dtype-agnostic; `fill_bits` 32-bit |
| memory_config "l1" | knob-turn | `TensorAccessorArgs` covers L1 interleaved; E writes in place in L1; N output allocated with x's memory config |
| rank 2 | knob-turn | `out_tensor_slices = 1`, no leading coords |
| alignment tile_aligned | knob-turn | no masks; only copy and fill-only tiles |
| padded_dims leading | knob-turn | leading coords already in the per-tile class (fill-only slices) — p1 |
| padded_dims w | knob-turn | column mask + fill-only columns (N) |
| layout ROW_MAJOR | scheme-change (regime R) | page = stick, sub-alignment sticks |
| sharded placement | not in TARGET; refused | — |

### Structural impossibilities

None beyond `feature_spec.py` (`INVALID = []`): every dtype exists in both layouts and TARGET holds
no block format.

## Work Distribution

| Field | Value |
|-------|-------|
| Work unit | one touched tile `w` (E: last-tile-row tile; N: output tile); blocks of `block_tiles` |
| Grid | `device.compute_with_storage_grid_size()`; `num_cores = min(grid cores, ceil(num_work_tiles / min_core_work_tiles))` |
| Per-core work | `ttnn.split_work_to_cores(grid, num_work_tiles, row_wise=True)`: core group 1 gets `max_core_work_tiles`, group 2 one fewer; each core a contiguous run `[core_work_begin, core_work_begin + core_work_tiles)` in row-major core order |
| Remainder | group 2 plus a ragged last block per core (`last_block_extent`); geometry from per-image `ceil` |
| Kernel groups | one `KernelDescriptor` per role over the union of both groups (per-core differences are RT args) → the used cores form a row-major prefix of the grid: ≤ 2 rectangles per role (BB-17) |

Phase 0 examples (the probed Blackhole device has an 11×10 = 110-core compute grid; the grid is read at run time, never assumed):

| Case | Regime | `num_work_tiles` | Cores | Tiles per core | Bytes read / written |
|---|---|---|---|---|---|
| p2 258 `[1,1,2,62080]`→32 | E | 1940 | 110 | 18 (70 cores), 17 (40) | 0 / 3,724,800 |
| p3 253 `[1,1,8,62080]`→16 | E | 1940 | 110 | 18 / 17 | 0 / 993,280 |
| p4 291 `[1,1,8,62080]`→32 | E | 1940 | 110 | 18 / 17 | 0 / 2,979,840 |
| 247 `[1,3844,1536]` | E | 48 | 48 | 1 | 0 / 86,016 |
| 254 `[16,1,1536]` | E | 768 | 110 | 7 (108), 6 (2) | 0 / 1,523,712 |
| 248 `[1,3844,12]` | E | 1 | 1 | 1 (the only touched page) | 0 / 1,792 |
| 64 `[1,196,128]`→256 | N | 32 | 32 | 1 | 57,344 / 65,536 |
| 65 `[1,1,196,196]`→256² | N | 64 | 64 | 1 | 100,352 / 131,072 |

## Circular Buffers

| Semantic Name | Index | Page Size | Num Pages | Sizing rationale | Format | Producer | Consumer | Lifetime |
|---------------|-------|-----------|-----------|------------------|--------|----------|----------|-----------|
| `cb_work_tiles` (N only) | 0 | `tile_bytes` | `block_tiles * buffer_depth` | spans the work axis by `block_tiles`; depth only when a core has > 1 block | x's dtype (raw; DM only) | reader | writer | whole kernel |
| `cb_fill_tile` (E and N) | 1 | `tile_bytes` | 1 | constant: one fill page | x's dtype (raw) | writer | writer (self-owned scratch: built once via the write pointer, never pushed or popped) | whole kernel; in N allocated only when the call has fill-only tiles |

## Block Operation Realization

| # | Block operation | Block shape | Helper? | Input CB (semantic name, pages, state) | Output CB (semantic name, pages) | CB state after |
|---|-----------------|-------------|---------|----------------------------------------|----------------------------------|----------------|
| E0 | `build_fill_segments` | ≤3 byte ranges | `fill_l1_range` | — | `cb_fill_tile`, 1 (write pointer) | segment bytes hold `c`, resident to kernel end |
| E1 | `fill_new_rows_block` | `block_tiles` | raw dataflow | `cb_fill_tile` (read by the NoC) | x via accessor | — |
| E2 | `drain_writes` | — | raw dataflow | — | — | all writes acked |
| N0 | `build_fill_page` | 1 tile | `fill_l1_range` | — | `cb_fill_tile`, 1 | filled, resident to kernel end |
| N1 | `load_block` | `block_tiles` | raw dataflow | x via accessor | `cb_work_tiles`, `block_tiles` reserved | reserved; bytes landed after the block barrier |
| N2 | `assemble_block` | `block_tiles` | `fill_l1_range` | `cb_work_tiles` (still reserved by the reader) | `cb_work_tiles`, push `block_tiles` | pushed |
| N3 | `store_fill_tiles_block` | `block_tiles` (fill-only subset) | raw dataflow | `cb_fill_tile` | out via accessor | — |
| N4 | `store_input_tiles_block` | `block_tiles` (input-backed subset) | raw dataflow | `cb_work_tiles`, wait `block_tiles` | out via accessor | popped after the flush |
| N5 | `drain_writes` | — | raw dataflow | — | — | all writes acked |

## API Mapping

| Block operation | Type | Function | File:Line | Template Params / Args | Input CB | Output CB | Which params are block knobs |
|-----------------|------|----------|-----------|------------------------|----------|-----------|------------------------------|
| `build_fill_segments`, `build_fill_page`, `assemble_block` | helper | `dataflow_kernel_lib::fill_l1_range<val_size>(start_addr, n_bytes, val)` | `ttnn/cpp/ttnn/kernel_lib/l1_helpers.hpp:100-101` | `val_size = elem_bytes`; `val = fill_bits` | — / `cb_work_tiles` | `cb_fill_tile` / `cb_work_tiles` | none (ranges fixed by `in_row_rem`, `in_col_rem`, `out_row_end`, `elem_bytes`) |
| `fill_new_rows_block` | raw_api | `TensorAccessor::get_noc_addr(page_id, offset)`; `noc_async_write(src_l1, dst_noc_addr, size)` | `tt_metal/hw/inc/api/tensor/tensor_accessor.h:165`; `tt_metal/hw/inc/api/dataflow/dataflow_api.h:876` | accessor CT args from `ttnn.TensorAccessorArgs(input_tensor)`; one address per tile, ≤3 writes | `cb_fill_tile` | — | `block_tiles` (writes per barrier) |
| `load_block` | raw_api | `noc_async_read_page(id, accessor, dst_l1)`, `noc_async_read_barrier()` | `dataflow_api.h:1243`, `:1921` | accessor CT args from `ttnn.TensorAccessorArgs(input_tensor)` | — | `cb_work_tiles` | `block_tiles` (reads per barrier) |
| `store_fill_tiles_block`, `store_input_tiles_block` | raw_api | `noc_async_write_page(id, accessor, src_l1)`, `noc_async_writes_flushed()` | `dataflow_api.h:1418`, `:1980` | accessor CT args from `ttnn.TensorAccessorArgs(output_tensor)` | `cb_fill_tile`, `cb_work_tiles` | — | `block_tiles` (writes per flush) |
| `drain_writes` | raw_api | `noc_async_write_barrier()` | `dataflow_api.h:1951` | — | — | — | — |

Helpers considered and rejected for the raw dataflow rows:

| Candidate | File:line | Why it cannot express the operation |
|---|---|---|
| `zero_tile` / `prepare_zero_tile` | `ttnn/cpp/ttnn/kernel_lib/l1_helpers.hpp:52-77` | zeros only; `c` is per-call data (−1, −2^127, 0.1 → bf16, 7.0 in the suite). `fill_l1_range` covers zero too |
| `Noc::async_write_zeros` | `tt_metal/hw/inc/api/dataflow/noc.h:849`, `:893` | zeros only (same reason); the DRAM overload still needs a pre-zeroed L1 scratch, so it saves nothing over writing from the fill page |
| `fill_tile_column` | `ttnn/cpp/ttnn/kernel_lib/tile_column_dataflow.hpp:69-70` | one column per call, 32 strided element stores each; a column mask of `32 - in_col_rem` columns is cheaper as 2 whole faces + 32 contiguous face-row ranges with `fill_l1_range` |
| `dataflow_kernel_lib` mcast (`mcast/kernel/mcast_args.hpp`) | — | no cross-core data in any built regime (regime M rejected) |
| local copy helpers (`local_copy_helpers_dataflow.hpp:57-158`) | — | L1→L1 self-reads; no L1→L1 step exists in the built regimes (they are the realization of the mask lamp (a)) |
| compute helpers (`eltwise_convenience.hpp`, `eltwise_chain.hpp`) | — | route values through unpack/DEST/pack (regime G rejected for bit equality); the op has no compute phase |
| — | — | no kernel_lib helper wraps interleaved page reads/writes for a block; the per-block batching (one barrier/flush per block) is the block operation |

`compute_kernel_hw_startup()`: not applicable — no compute kernel in any regime.

## Host-side contract (op file)

| Item | Decision |
|---|---|
| Files | `ttnn/ttnn/operations/g6_pad/__init__.py` exports `g6_pad`, `INPUT_TAGGERS`, `SUPPORTED`, `EXCLUSIONS`; `g6_pad.py` (entry point, taggers, validate, plan); program-descriptor module; `kernels/` (three roles: `extension_writer`, `new_storage_reader`, `new_storage_writer`) |
| Signature | `g6_pad(input_tensor, padding, value=0.0)` — `padding` and `value` positional or keyword |
| First line | `validate(input_tensor, padding, value)` |
| validate order | (1) argument form → `ValueError`: `padding` a list/tuple of 1..rank 2-element lists/tuples of `int` (not `bool`), every `low == 0`, every `high >= 0`; `value` an `int` or `float`, not `bool`; int32/uint32: finite and `trunc(value)` in range; input on device; input not sharded. (2) scenario dict `{shape, padding, value, dtype, layout, memory_config: "l1" if buffer type is L1 else "dram"}`. (3) taggers → `SUPPORTED` per axis → `UnsupportedAxisValue`. (4) `EXCLUSIONS` (empty) → `ExcludedCell` |
| INPUT_TAGGERS | `tag_alignment`: `W % 32 != 0` → `"w_non_aligned"`; else `H % 32 != 0` → `"h_non_aligned"`; else `"tile_aligned"`. `tag_rank`: `len(shape)`. `tag_padded_dims`: highs aligned to the trailing dims, `hh = highs[-2] if len(highs) >= 2 else 0`, `hw = highs[-1]`; both > 0 `"hw"`, only `hh` `"h"`, only `hw` `"w"`, neither `"leading"`. `tag_dtype`, `tag_layout`: the enum. `tag_memory_config`: `s["memory_config"]`. Taggers are **total**: the harness tags malformed refusal scenarios (5 pairs on rank 4, `low != 0`) at collection, so they never raise |
| SUPPORTED (Phase 0) | `dtype: [bfloat16]`, `layout: [TILE]`, `memory_config: ["dram"]`, `alignment: ["w_non_aligned", "h_non_aligned"]`, `rank: [3, 4]`, `padded_dims: ["h", "hw"]` (implementer owns the block) |
| EXCLUSIONS | `[]` |
| Dispatch | E: `ttnn.generic_op([input_tensor], desc)` then the view. N: allocate `out`, `ttnn.generic_op([input_tensor, out], desc)` |
| Descriptor cache (BB-07) | module-level dict keyed by `(regime, tuple(x.shape), tuple(highs), dtype, layout, buffer_type, grid)`; the value is the built `ProgramDescriptor` (CBs, kernels, CT args, per-core RT args). Per-call values live **only** in common RT args: E writer `[x_addr, fill_bits]`; N reader `[x_addr, fill_bits]`; N writer `[out_addr, fill_bits]`. A hit writes those and dispatches; nothing else is rebuilt. `generic_op` hashes RT arg counts, not values (`generic_op_device_operation.cpp:78-139`), and re-applies common RT args on a program-cache hit (`tt_metal/impl/program/program_descriptors.cpp:188-215`). Implementer verifies by probe that the write reaches the dispatched descriptor (nanobind list properties return copies) |
| Plan report (BB-08, BB-21) | `g6_pad.last_plan` (or an equivalent function) records per call: regime, `num_work_tiles`, `num_cores`, `block_tiles`, `buffer_depth`, and the role per core range (`E: extension_writer`, `N: reader+writer`) |
| Trace (BB-16) | no device allocation besides N's output; no scratch tensors; descriptors cached before capture |

## Key Risks and Gotchas

| Risk | Why it bites here | Mitigation in this design |
|------|-------------------|---------------------------|
| Mask applied before the read lands (N) | the in-flight read overwrites the masked bytes; zero-padded test inputs and `c = 0` hide it | mask strictly after the block's `noc_async_read_barrier()`; acceptance test runs every case with the input's tile padding poisoned to 1000 |
| Whole-tile copy of a boundary tile without the mask (N) | returns the producer's padding where the output holds `c` | class `boundary` always masks; poisoned-padding cases 64/65/68/72 |
| E segment off by one face row | rows `< in_row_rem` hold x's logical values: writing them destroys input data in place | segments computed once on the host from `in_row_rem`, `out_row_end`; bit-exact check of every logical output position (rows `< in_row_rem` are x's values) and of the raw rows above the last tile row |
| View route launches a program | a reshape internals change could route the volume-changing view through the codegen plan or a device op | pinned route table above; acceptance test counts device programs per call by graph capture (must be exactly 1) |
| Returning x itself instead of the view | output shape is wrong and the golden storage check refuses `out is x` | the entry point always returns the `ttnn.reshape` view at `out_shape` |
| Program cache keyed too loosely / `c` baked in | the same shape is called with values 0 then 7.0, then with different padding | `fill_bits` is a common RT arg; geometry and segments are CT args (hashed); the cache key holds shape and highs |
| `c` conversion | 0.1 is inexact in bf16; −2^127 must stay finite (`0xFF00`); ints truncate | torch's own conversion on the host (`torch.full`); int path without a float detour |
| bf16 fill width | `fill_l1_range<2>` replicates the low 16 bits into every word; `val_size = 4` with a bf16 pattern would write a word whose high half is not `c` | `val_size = elem_bytes`; host masks `fill_bits` to 16 bits for bf16 |
| Ragged last block (N) | a short push breaks the ring when a core has > 1 block | nominal `block_tiles` push/pop every block; only the transfers narrow |
| Taggers raising on refusal scenarios | the harness tags every LOOSE case, including 5 pairs on rank 4 and `low != 0`, before calling the op | taggers use trailing-aligned highs and negative indices only; argument checks live in `validate()` |
| Sharded input reaching the support gate | the golden refusal case is a supported cell (bf16 TILE); a `SupportRefusal` there is a failure | the sharded check is an argument-form `ValueError` raised before tagging |
| E output tile padding keeps the producer's values | rows `out_row_end..31` of the last tile row are not written | allowed by the contract (output tile padding may hold anything); deliberate — writing them would add bytes for nothing |

## Catalog entries used

| Entry | Knob it informed |
|---|---|
| `width_split` | fold the tile-column axis into the work run so one-tile-row p2–p4 fill the grid |
| `noc_placement` | `row_wise=True`; reads NoC0 / writes NoC1 |
| `double_buffer` | N overlap lamp (block 4–8, depth 2); one barrier per block, never per tile |
| `tile_reorder` | N moves whole tiles; E's sub-tile writes are justified by moving fewer bytes, not equal bytes |
| `split_reader` | E dual-RISC issue lamp |
| `bank_stagger` | N p1 bank lamp |
