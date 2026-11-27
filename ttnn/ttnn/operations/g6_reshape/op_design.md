# Operation Design: g6_reshape

## Overview

| Field | Value |
|-------|-------|
| Classification | data_movement. Two data-movement kernels per core (the two **lanes**) and **no compute kernel**: no value passes an unpacker, DEST or a packer, so the op is bit-exact by construction |
| Goal | Return `input_tensor` under shape `e`: every logical value moves bit for bit to the position with the same row-major linear index, laid out again in the input's own layout (TILE in Phase 0). The op is ONE device program that reads only valid input bytes and writes only valid output bytes. |
| Math | `out[j] = x[i]` whenever `lin(j; e) = lin(i; d)`. In the page view (leading dims folded into rows, `g` = global row): `Y[g_o, c_o] = X[L div W_in, L mod W_in]` with `L = g_o·W_out + c_o` |
| References | `eval/golden_tests/g6_reshape/{feature_spec,helpers,axes}.py`; `eval/op_template.py`; `ttnn/ttnn/operations/_op_contract.py`; `.claude/references/{blocking-model,l1-footprint-discipline,op-design-template,ttnn-cb-memory-fundamentals}.md`; `.claude/references/generic_op_template/`; catalog `ttnn/ttnn/operations/examples/master.md` (entries in the table below); `tt_metal/hw/inc/internal/tt-1xx/blackhole/noc/noc_parameters.h:375-380`; `tt_metal/hw/inc/internal/debug/sanitize.h:319-341,746`; `tt_metal/impl/tensor/spec/layout/tensor_layout.cpp:261-346`; `tt_metal/impl/program/program.cpp:2505-2509`; `tt_metal/hw/inc/internal/tt-1xx/cache.h:7-16`. `docs/explorations/g6_reshape/` does not exist in this tree, so this design works from the spec alone. |

The op is a bijection between the valid **face-row segments** of the two tensors. A segment is at most 16 consecutive logical elements of one tile row inside one face: one contiguous DRAM run of at most `16·esize` bytes. One tensor, the **tile side**, is partitioned across lanes and moved as merged face runs at identical intra-tile offsets in DRAM and L1. The other, the **map side**, is addressed segment by segment through the index map. The regime choice (§Regimes) decides which tensor is the tile side: the one whose tiles need fewer coalesced transfers.

### Catalog entries that set knobs

| Entry | What it decided here |
|-------|----------------------|
| `split_reader` | Both data-movement RISC-Vs issue map-side transactions (two **lanes** per core): the map side is issue-bound, with tens of 32-B transfers per core against 1–3 tile-side ones |
| `noc_placement` | Core set comes from `num_cores_to_corerangeset(..., row_wise=True)`. Lane 0 (NCRISC) is on NoC0 and lane 1 (BRISC) on NoC1, the `ReaderConfigDescriptor` / `WriterConfigDescriptor` defaults. The NoC-direction lamp covers lane 1's reads on NoC1 |
| `double_buffer` | One read barrier per block, never one per transfer. Buffer depth is the `SLOT_DEPTH` knob |
| `trid_double_issue` | The read-ahead lamp: `SLOT_DEPTH = 2` with trid-scoped barriers for multi-block lanes |
| `tile_reorder` | Byte-adjacent face runs merge into one transfer (a 16-row tile moves as one 1-KiB transfer). The span-write lamp compares this with one transfer that spans padding rows |
| `width_split` | The split unit is sub-tile (strip), so a tensor with 2 tile-side tiles still spreads over the lanes the knob chooses |
| `bank_stagger` | Bank-stagger lamp |
| `read_coalescing` | Face-pair read lamp (one read covering both valid face rows of a 1-row input tile) |

## Parameters

| Name | Type | Required | Valid Range | Default | CT/RT |
|------|------|----------|-------------|---------|-------|
| `input_tensor` | `ttnn.Tensor` | yes | on device, DRAM interleaved, rank 1..4, every dim ≥ 1. Phase 0 SUPPORTED: `bfloat16` × `TILE_LAYOUT` | — | geometry → CT; buffer address → common RT |
| `shape` | `list[int]` or `tuple[int]`, positional | yes | length 1..4; integer entries (not `bool`), each ≥ 1, or exactly one `-1`; product after resolving `-1` = N = element count of `input_tensor` | — | resolved `e` → CT (through the geometry symbols) |

### Registry and host validation

| Item | Decision |
|------|----------|
| `INPUT_TAGGERS` | `{"dtype": tag_dtype, "layout": tag_layout}`. `tag_dtype(inputs, axes)` returns `inputs[0]["dtype"]`; `tag_layout(inputs, axes)` returns `inputs[0]["layout"]` (scenario dict `{"input_shape", "shape", "dtype", "layout"}`). Axis names and values match `feature_spec.TARGET`. |
| `SUPPORTED` (Phase 0) | `{"dtype": [ttnn.bfloat16], "layout": [ttnn.TILE_LAYOUT]}` |
| `EXCLUSIONS` | `[]` |
| `validate(input_tensor, shape)` | First line of `g6_reshape`. Builds the scenario dict from the call (`tuple(input_tensor.shape)`, `shape`, `input_tensor.dtype`, `input_tensor.layout`) and applies `INPUT_TAGGERS` in order. Then raises `UnsupportedAxisValue` for an axis value outside `SUPPORTED`, then `ExcludedCell` for an `EXCLUSIONS` match. No INVALID check. No size-based refusal and no shape tagger (the int32/uint32 ROW_MAJOR cases are refused only through the `dtype` / `layout` axes). |
| `_resolve_shape(input_tensor, shape)` (right after `validate`) | Raises `ValueError` when: `shape` is not a list/tuple; its length is outside 1..4; an entry is non-integer or `bool`; an entry is `0`; an entry is `< -1`; more than one entry is `-1`; with a `-1`, `N mod prod(other entries) ≠ 0`; the product ≠ N; the input rank is outside 1..4; the input is not on device or not DRAM interleaved; `N ≥ 2**32` (index-width cap, §Mechanism caps). Returns `e`. |
| Output allocation | `ttnn.allocate_tensor_on_device(ttnn.Shape(e), input_tensor.dtype, input_tensor.layout, device, input_tensor.memory_config())`, positional args. This is the ONLY device allocation. An identity reshape (`e == d`) still dispatches the program, because the output is a new tensor. |
| Dispatch | `ttnn.generic_op([input_tensor, output], descriptor)`: one program per call. The entry point never calls `to_layout`, `tilize`, `untilize`, `pad`, `slice`, `reshape` or `typecast`, and never reads tensor values on the host. |
| Plan report (BB-08, BB-21) | `g6_reshape_plan(input_tensor, shape) -> dict`, exported from `ttnn.operations.g6_reshape`. It runs on the host only, dispatches nothing, and returns the same plan object the program builder uses (§Work Distribution). |

## Tensors

### Input

| Property | Requirement |
|----------|-------------|
| Shape | `d = (d_0..d_{n-1})`, n ∈ 1..4, every `d_i ≥ 1` |
| Dtype | Phase 0: `bfloat16`. TARGET adds `int32` and `uint32` (ROW_MAJOR), and `float32` (TILE) |
| Layout | Phase 0: `TILE_LAYOUT`. A 32×32 tile is four 16×16 faces, face-major and row-major within a face |
| Memory | DRAM interleaved |

### Output

| Property | Value |
|----------|-------|
| Shape | `e` = `shape` with `-1` resolved |
| Dtype | `input_tensor.dtype` |
| Layout | `input_tensor.layout` |
| Memory | `input_tensor.memory_config()` (DRAM interleaved) |

### Geometry symbols

These are derived once per call signature on the host. Their single source is the plan builder in `g6_reshape_program_descriptor.py`.

| Symbol | Definition (same rule for `_in` and `_out`) |
|--------|---------------------------------------------|
| `W` | last logical dim |
| `H` | second-to-last logical dim; `1` for rank 1. A rank-1 TILE tensor pads to `[32, 32]` (`tensor_layout.cpp:311-346`) |
| `IMG` | product of the dims before the last two; `1` for rank ≤ 2 |
| `Wt`, `Ht` | `ceil(W/32)`, `ceil(H/32)`. Tile padding is per image: the physical height is `IMG·round_up(H, 32)` (`tensor_layout.cpp:261-290`) |
| `num_tiles` | `IMG·Ht·Wt`. The host checks it equals `tensor.buffer_num_pages()` and raises RuntimeError otherwise |
| tile id | `t = (img·Ht + tr)·Wt + tc` |
| `v(tr)`, `vc(tc)` | valid rows / columns of a tile: `min(32, H − 32·tr)`, `min(32, W − 32·tc)` |
| global row, linear index | `g = img·H + 32·tr + r`, `L = g·W + c` |
| `esize` | `input_tensor.element_size()`: 2 for bf16, 4 for fp32/int32/uint32 |
| `face_row_bytes`, `face_bytes`, `tile_bytes` | `16·esize`, `256·esize`, `1024·esize` |
| `face_off(r, c)` | `((r div 16)·2 + c div 16)·face_bytes + (r mod 16)·face_row_bytes + (c mod 16)·esize`, for `r, c ∈ [0, 32)` |
| `granule_bytes` | `max(ttnn.get_dram_alignment(), face_row_bytes)`. On Blackhole this is 64 for every TARGET dtype (`NOC_DRAM_READ_ALIGNMENT_BYTES = 64`, `noc_parameters.h:379`); on Wormhole it is 32 for bf16 |
| `strip_rows` | `granule_bytes div face_row_bytes`: 2 for bf16 on BH, 1 for fp32 on BH, 1 for bf16 on WH. These are the tile rows whose face rows share one DRAM granule |
| `strips(tr)` | `ceil(v(tr) / strip_rows)` |
| `tile_runs(r_lo, r_hi, vc)` | Number of transfers that move rows `[r_lo, r_hi)` of one tile. Each face `f` (in byte order 0..3, skipping faces 1 and 3 when `vc ≤ 16`) that intersects the rows contributes the byte range of its intersecting rows, and byte-adjacent ranges merge. The result is ≤ 4: `[0, 16)` with `vc > 16` gives 1 (faces 0+1), `[0, 32)` gives 1 (whole tile), `[0, 6)` gives 2 |
| `tile_side_transfers(T)` | `Σ_{img, tr, tc} tile_runs(0, v(tr), vc(tc))` |
| `aligned` | `W_in mod 16 == 0 ∧ W_out mod 16 == 0`. Then every segment of either tensor is a whole 16-element face row, and the two tensors' segments coincide one to one |

## Blocking Model

### Work units

| Term | Definition |
|------|-----------|
| **Tile side** `T_s` | The output in `gather_*` regimes and the input in `scatter_aligned`. Its tiles are partitioned, and its transfers are merged face runs at identical intra-tile offsets in DRAM and in the lane's L1 tile image. |
| **Map side** `T_m` | The other tensor, addressed per segment / piece through the index map. It is never partitioned on its own: each lane touches exactly the map-side bytes its tile-side segments name. |
| **Strip** (the work unit) | Rows `[p·strip_rows, min((p+1)·strip_rows, v(tr)))` of tile-side tile `t`, `p < strips(tr)`. Strips are ordered tile-major: `t` ascending, then `p`. Count: `U = IMG_s · Wt_s · Σ_{tr<Ht_s} strips(tr)`. A strip never shares a DRAM granule with another strip of the same tile, so no two lanes write one tile-side granule. |
| **Segment** | One row `r` of a strip × column half `fh ∈ {0, 1}` with `16·fh < vc(tc)`: columns `[c0, c0 + n)`, `c0 = 32·tc + 16·fh`, `n = min(16, W_s − c0)`. Within a strip, segments are walked rows ascending, then `fh`. |
| **Piece** (`gather_general`) | The maximal sub-range of an output segment that is contiguous in one input face row. It ends at `c_i mod 16 == 15`, at `c_i == W_in − 1`, or at the segment end. `aligned` ⇒ exactly one piece per segment, a whole face row. Otherwise there are at most `P = min(16, ceil(16 / W_in) + 2)` pieces per segment. |
| **Lane** | One data-movement RISC-V of a core running the full read → assemble → write pipeline for its own contiguous strip run. Lane 0 is NCRISC on NoC0, lane 1 is BRISC on NoC1. A lane owns its L1 scratch, and no CB credit passes between lanes. |
| Strip `s` → `(img, tr, tc, p)` (closed form, host and kernel) | `sf = 32 div strip_rows`; `S_img = Wt·(sf·(Ht − 1) + strips(Ht − 1))`; `img = s div S_img`; `s1 = s mod S_img`; `tr = min(s1 div (sf·Wt), Ht − 1)`; `s2 = s1 − tr·sf·Wt`; `tc = s2 div strips(tr)`; `p = s2 mod strips(tr)`. Kernels walk incrementally from the lane's start strip: `p += 1`; when `p == strips(tr)`, set `p = 0` and `tc += 1`; when `tc == Wt`, `tc = 0` and `tr += 1`; when `tr == Ht`, `tr = 0` and `img += 1`. |

### Axes

| Axis | Character (+ one-clause reason) | Extent knob | Phase 0 value | Knob source | Core-assignment | Later unlock |
|------|--------------------------------|-------------|---------------|-------------|-----------------|--------------|
| `t_s`: tile-side tile sequence (`img`, `tr`, `tc` folded; contiguous in tile-id order and treated identically) | independent: the map is a bijection, so no logical value depends on another and no result spans an axis | `block_tiles` | `min(max_lane_tiles, BLOCK_TILES_CAP)`: the whole lane run, which is 1–3 tiles on every golden shape | `BLOCK_SLOT_BYTES_CAP` → `BLOCK_TILES_CAP = max(1, BLOCK_SLOT_BYTES_CAP div tile_bytes)` → `block_tiles` (CT) | contiguous strip runs over `n_lanes = LANES_PER_CORE·n_cores` lanes (Work Distribution) | knob-turn |
| `strip`: strip index inside a tile-side tile | independent: each strip's segments read and write only their own linear ranges | strips per block = every strip of the lane's run that lies in the block's tiles (derived, never split further) | all `strips(tr)` of each tile the block covers; fewer only in a lane's first and last tile | lane run `(start_tile, start_strip, num_strips)`, per-lane RT args derived on the host | split across lanes at strip granularity. This lets the lane count exceed the tile count: `[16,1,64]→[1,1,16,64]` has 2 output tiles and 16 strips | knob-turn |
| `r`: row inside a strip | independent | `strip_rows` | 2 (bf16 on BH), the whole strip, never split | derived from `granule_bytes` and `esize` (CT) | not split across lanes: the rows of one strip share a DRAM granule (mechanism cap) | none |
| `fh`: column half of a row | independent: two disjoint 16-column segments | segments per row = `1 + [vc(tc) > 16]` | whole, never split | derived from `W_s` (CT) | not split: both halves of a row stay in one lane | none: a face row is the DRAM-contiguous atom of the tile format |
| `e`: element inside a segment | independent | 16 (whole face row) | 16 | fixed by the tile format | not split | none |
| `t_m`, `r_m`, `c_m`: map-side tiles / rows / columns | independent and **not reuse-shared**: exactly one segment names each map-side element, so no operand is read twice | none: a block's map-side footprint is exactly the image of its segments, so there is no extent to choose | the image of the block's strips | — (no knob, by construction) | follows tile-side ownership (owner moves) | `scatter_aligned` is this same model with the two tensors' roles swapped |
| staged granules (`gather_*` intermediate data) | independent: each is consumed only by the copies of its own lane | `staging_slots`, `copy_entries` (flush window) | `min(cap, staged_bound)`: one flush per block on every golden shape | `STAGING_SLOTS_CAP`, `COPY_ENTRIES_CAP` | never leaves its lane. No core collects, coordinates or hands out work: every lane runs the same role on its own run, so no stage leaves the rest of the group waiting | knob-turn |

### Knob sources (single source of truth)

All knobs are module-level constants of `ttnn/ttnn/operations/g6_reshape/g6_reshape_program_descriptor.py`. The plan builder reads them as **module globals at call time**, never copies them into another module at import, and puts every one of them in the descriptor memo key. Every quantity below them is derived once on the host; kernels receive only derived values.

| Knob | Phase 0 value | Derived quantities |
|------|---------------|--------------------|
| `LANES_PER_CORE` | 2 | lane kernels instantiated per core (1 = lane 0 only); `n_lanes` cap `= LANES_PER_CORE·grid_cores` |
| `MIN_STRIPS_PER_LANE` | 1 (full occupancy) | `n_lanes = min(LANES_PER_CORE·grid_cores, ceil(U / MIN_STRIPS_PER_LANE))`, `n_cores = ceil(n_lanes / LANES_PER_CORE)` |
| `BLOCK_SLOT_BYTES_CAP` | 131072 | `BLOCK_TILES_CAP`, `block_tiles`, `num_blocks_this_lane`, `max_blocks_per_lane`, `max_block_rows = min(max_lane_rows, 32·block_tiles)` |
| `SLOT_DEPTH` | 1 | `slot_depth = min(SLOT_DEPTH, max_blocks_per_lane)` |
| `STAGING_SLOTS_CAP` | 512 | `staged_bound = 2·max_block_rows·P` (`P = 1` for `gather_aligned`); `staging_slots = min(STAGING_SLOTS_CAP, staged_bound)` |
| `COPY_ENTRIES_CAP` | 1024 | `copy_entries = min(COPY_ENTRIES_CAP, staged_bound)` |
| `REGIME_OVERRIDE` | `None` | `regime` (§Regimes). Non-`None` values are legal only where the forced regime is correct (`gather_aligned` / `scatter_aligned` require `aligned`); otherwise the builder raises `ValueError` |

### Buffer-depth knobs

| CB | Depth knob | Phase 0 value | What the depth buys |
|----|------------|---------------|---------------------|
| `cb_lane0_tile_images`, `cb_lane1_tile_images` | `SLOT_DEPTH` → `slot_depth` | 1 (1 block per lane on every golden shape) | Depth 1 already overlaps block `b`'s write acks with block `b+1`'s reads: writes are async and only `noc_async_writes_flushed()` (data out of L1) precedes slot reuse. Depth 2 lets a lane issue block `b+1`'s reads before it waits on block `b`'s (the read-ahead lamp) |
| `cb_lane0_staging`, `cb_lane1_staging` | `STAGING_SLOTS_CAP`, `COPY_ENTRIES_CAP` → `staging_slots`, `copy_entries` | 512 / 1024, effectively `staged_bound`: ≤ 24 on the production shapes, ≤ 120 on the `gather_general` pins | staged reads in flight between two flushes (barrier + copies). One flush per block whenever the bound fits |

### Mechanism caps

| Mechanism | Cap on which extent | Clamp | What happens unclamped |
|-----------|--------------------|-------|------------------------|
| DRAM read congruence `(l1_dst & (A−1)) == (dram_src & (A−1))`, with `A = NOC_DRAM_READ_ALIGNMENT_BYTES`: 64 on BH, 32 on WH (`noc_parameters.h:379`; watcher check at `sanitize.h:746`) | A direct map-side face-row read in `gather_aligned`. A bf16 32-B face row lands at a 32-B row stride, so source and destination parity differ when `r_in ≢ r_out (mod 2)`: about half the segments of the 9 production `gather_aligned` shapes | Per segment at run time: the read is **direct** iff the congruence holds, else **staged** (read the `granule_bytes` granule that contains the face row into a staging slot, then copy after the barrier). `gather_general` stages every piece. fp32 face rows are 64 B and are always congruent | NoC alignment hang under the watcher; silently shifted rows without it |
| The same congruence for tile-side reads (`scatter_aligned`) | none | Always congruent: tile images sit at `cb_base + k·tile_bytes`, the CB base is DRAM-aligned (`program.cpp:2505-2509`), DRAM pages sit at bank base + `k·round_up(tile_bytes, 64)`, and the intra-tile offset is identical on both sides. The kernel asserts it under `--dev` | — |
| DRAM write congruence, 16 B (`noc_parameters.h:380`) | every write | Writes start at face-row boundaries (multiples of 32 B) on both sides. Partial pieces never go to DRAM directly; they pass through the tile image | NoC error / hang |
| Shared DRAM granule between lanes | tile-side strip boundaries | The strip is `strip_rows` rows, so lane boundaries fall on granule boundaries and tile-side runs start 64-B aligned | two lanes' byte-masked writes into one 64-B granule (Key Risks) |
| CB-bounds sanitizer: a NoC transfer that starts inside a CB must end inside it (`sanitize.h:319-341`) | staged granule reads | Each granule read lands in exactly one `granule_bytes` staging slot. The copy table occupies the CB tail and only RISC-V loads/stores touch it | watcher `DebugSanitizeCBOutOfBounds` |
| Kernel L1 budget, 1,347,776 B (BB-05) | `block_tiles`, `slot_depth`, `staging_slots`, `copy_entries`, `LANES_PER_CORE` | `BLOCK_SLOT_BYTES_CAP`, `SLOT_DEPTH ≤ 2`, the two staging caps, `LANES_PER_CORE ≤ 2` ⇒ ≤ 606,208 B per core for any shape and any N (`l1_ledger.md`) | OOM at launch for large N |
| 32-bit linear index / byte offset in the kernels | N | The host refuses `N ≥ 2**32` with `ValueError`. No golden or production shape is within 10⁴× of it | wrapped indices → wrong values |

### Regimes

Boundary: DRAM. The minimum is every logical input byte read once and every logical output byte written once. The transaction floor is set by the map side: one transfer per valid segment.

| Regime | Status | Predicate | Block | Data movement vs. minimum | What a bigger block buys |
|--------|--------|-----------|-------|---------------------------|--------------------------|
| **`gather_aligned`** | **built** (Phase 0): 9 of the 14 production bf16 cases | TILE ∧ `aligned` ∧ `tile_side_transfers(in) > tile_side_transfers(out)` | `block_tiles` output tiles × the lane's strips in them × whole segments | Output: minimum, one write per merged face run of the lane's rows in a tile. Input: every valid face row once. A staged face row reads its 64-B granule for 32 B, so input bytes are ≤ 1.5× minimum on BH bf16 (fp32: 1.0×, all direct). Transactions: `segs` reads + Σ lane merged-run writes (`[16,1,5120]→[1,1,16,5120]`: 5120 reads + ≤ 880 writes, ≤ 4 per lane) | Fixed costs per block: 1 slot-reuse flush, 1 read barrier, 1 assemble pass, 1 fence. Each block pays one DRAM read round trip, amortized over its segments. Intended frequency: once per lane on every golden shape |
| **`scatter_aligned`** | **built** (Phase 0): the other 5 production bf16 cases (`[1,1,H,4608]→[H,1,4608]`, `[1,H,64]→[H,1,64]`, `[1,1,6,256]→[1,1,1536]`) | TILE ∧ `aligned` ∧ `tile_side_transfers(in) ≤ tile_side_transfers(out)` (a tie goes here: no staging) | `block_tiles` input tiles × the lane's strips in them × whole segments | Minimum on both tensors, with no staging: input face runs land congruent in input tile images, and output face rows are written straight from them. Transactions: Σ lane merged-run reads + `segs` writes (`[1,1,16,4608]→[16,1,4608]`: ≤ 880 + 4608, against 4608 + 4608 as a gather) | Fixed costs per block: 1 slot-reuse flush + 1 read barrier, plus one DRAM read round trip. Intended: once per lane |
| **`gather_general`** | **built** (Phase 0). No production bf16 case selects it, but `SUPPORTED` admits every bf16 TILE shape, and the float32 FLUX cases (`[1,W]→[1,1,W,1]`, W ∈ {185, 196, 207, 256, 512}) land here once fp32 is enabled | TILE ∧ ¬`aligned` | as `gather_aligned` | Input: each distinct granule that contains a piece, read once per flush window per lane (single-entry memo). That exceeds the minimum by granule rounding; the excess is structural when `W_in < 16`, because each row's few valid elements sit alone in a face row. Output: face runs that include the padding columns of partial face rows (`W_out = 1`: one write carries 16 valid elements, 16× bytes for 1/16 the transactions of element writes) | as `gather_aligned`, plus granule reuse across consecutive strips (`W_out = 1`: 16 consecutive pieces per granule) |
| `page_identity_copy` | deferred: no golden or production shape satisfies the predicate, and `scatter_aligned`/`gather_*` are correct there | TILE ∧ `IMG_in·Ht_in == IMG_out·Ht_out` ∧ `H_in == H_out` ∧ `W_in == W_out` (output page `t` ≡ input page `t` byte for byte) | `block_tiles` whole tiles | minimum in whole-tile transfers: 1 read + 1 write per tile, instead of the per-segment map side | Reachable: the tile side already moves merged runs at identical offsets; this regime makes both sides tile-side with whole-tile runs, on the same lane skeleton |
| `rm_page_gather` | deferred: ROW_MAJOR enters `SUPPORTED` only together with int32/uint32 (spec rule: dtype and its production layout in one change). Phase 0 is TILE only | ROW_MAJOR | `block_pages` output pages, each page one output row at its aligned page stride | Input: one read per (output-page piece, input page) span. The 64-B congruence is handled by the same granule staging as `gather_general`. Output: one write per output page | Reachable: the same lane skeleton, with the slot holding output pages at aligned page stride and the map side holding contiguous input spans; only the address function changes (no faces) |
| `rm_bank_slot_writes` | deferred: the successor of `rm_page_gather` for the 4 flagged int32 `[1,W]→[W,1]` cases (W = 4096, 3844, 1024, 1000), built in the same ROW_MAJOR refinement | ROW_MAJOR ∧ `W_out·esize < aligned_page_bytes(out)` | per lane, a run of consecutive slots of ONE DRAM bank (page `p` → bank `p mod num_banks`, slot `p div num_banks`) | Output: one write per lane covering its bank-slot run. The alignment gap of each 64-B slot belongs to the output buffer and is written as part of it. For p1, 4096 four-byte pages become ~`n_lanes` writes instead of 4096. Input: one contiguous read per lane of the span that holds its pages' elements | Reachable: the lane slot already holds pages at aligned stride; this regime only orders lanes by bank and merges slot spans, layered on `rm_page_gather` |
| `scatter_general` | rejected: superseded by `gather_general`. Unaligned map-side pieces are 2–30 B and cannot be written to DRAM under the 16-B write congruence without a read-modify-write of output face rows (twice the output traffic, plus cross-lane races) | — | — | ≥ 2× output | — |
| `reader_writer_pipeline` (one RISC-V reads, the other writes, CB credit between them) | rejected: superseded by the two lanes. Bytes are equal, but every map-side transaction lands on ONE RISC-V while the other issues 1–3 tile-side transfers. The lanes halve the per-RISC issue count, and the pipeline's cross-block overlap is recovered inside a lane by async writes + flush-before-reuse | — | — | minimum bytes; 2× issue on the critical RISC-V | — |
| `whole_tiles_in_out` | rejected: a dead end, superseded everywhere by the three built regimes | — | whole tiles | reads every padding row (`[16,1,5120]`: 2560 tiles = 5.24 MB for 160 KiB of values) and writes every padding row (`[16,1,4608]`: 4.7 MB for 144 KiB), 16–32× minimum | — |
| `engine_untilize_regroup_tilize` | rejected: it routes values through unpack/DEST/pack on a bit-exact contract (NaN payloads, denormals and int32 above 2²⁴ are not guaranteed to survive). It also reads whole padding tiles, adds a compute kernel and two CBs, and still needs this design's index-map copies for the regroup | — | — | ≥ whole-tile traffic + two engine passes | — |
| `staged_through_dram_or_host` | rejected: the contract forbids it (no further DRAM tensor, no host read) | — | — | 2× minimum | — |
| `cross_core_segment_routing` | rejected: DRAM → core A → NoC → core B → DRAM. Each segment is already read and written by its owner, so the extra hop saves no DRAM byte | — | — | minimum DRAM + one NoC crossing per segment | — |
| `single_core` | rejected: Phase 0 is multi-core by contract, and the strip split is free | — | — | minimum | — |

**Regime selection function.** It runs on the host, is exact, and depends only on the shapes and `REGIME_OVERRIDE`, never on the grid:

```
if layout != TILE:                                    # unreachable in Phase 0 (validate refuses it)
    regime = rm_page_gather / rm_bank_slot_writes    (deferred)
elif REGIME_OVERRIDE is not None:                     # legality checked, ValueError otherwise
    regime = REGIME_OVERRIDE
elif not aligned:                                     regime = "gather_general"
elif tile_side_transfers(in) <= tile_side_transfers(out): regime = "scatter_aligned"
else:                                                 regime = "gather_aligned"
```

Every regime compiles from ONE kernel source, `kernels/g6_reshape_lane.cpp`, with exactly one define: `G6_REGIME_GATHER_ALIGNED`, `G6_REGIME_SCATTER_ALIGNED` or `G6_REGIME_GATHER_GENERAL`. Dead paths are compiled out (BB-18).

**Regime-pinned tests.** The acceptance test asserts `g6_reshape_plan(...)["regime"]` before it checks values:

| Regime | Pinned cases |
|--------|--------------|
| `gather_aligned` | the 9 production cases whose `tile_side_transfers(in) > tile_side_transfers(out)` |
| `scatter_aligned` | the other 5 production cases, plus `[1,1,30,64]→[10,3,64]` (lanes writing two halves of one DRAM granule) |
| `gather_general` | the production geometries of the int32/float32 cases, run in bf16: `[1,185]→[1,1,185,1]`, `[1,77]→[77,1]`, `[1,4096]→[4096,1]`, `[8]→[8,1]`, `[8,1]→[8]` |
| all three | forced through `REGIME_OVERRIDE` on `[1,16,6,256]→[1,1,16,1536]` |

### Traffic ranking

Candidate splits are ranked by DRAM crossings per tensor, then cross-core traffic, then transaction count on the critical RISC-V. The ranking is qualitative; there are no ns estimates. `segs` = valid segments, equal on both sides when `aligned`.

| Rank | Split | Input DRAM crossings | Output DRAM crossings | Cross-core | Transactions | Verdict |
|------|-------|----------------------|-----------------------|------------|--------------|---------|
| 1 | **contiguous tile-side strip runs over 2 lanes per core, owner moves; tile side = the tensor with fewer `tile_side_transfers`** | 1 × valid (`gather_aligned` staged face rows read a 64-B granule) | 1 × valid | none | `segs + ≈min(tile_side_transfers)`, split over 2 RISC-Vs per core | **chosen**: minimum bytes on both tensors, no combine, fills the grid at strip granularity, and halves the per-RISC issue count |
| 2 | same split with a reader/writer pipeline per core | 1 × valid | 1 × valid | none | same total, but all map-side transfers on one RISC-V | superseded (regime `reader_writer_pipeline`) |
| 3 | same split, tile side fixed to the output (gather only) | 1 × valid (+ staging) | 1 × valid | none | `segs + tile_side_transfers(out)` | equal bytes, ~2× the transactions on the 5 `scatter_aligned` shapes, plus avoidable staging → superseded by the predicate |
| 4 | whole tile-side tiles per lane (tile granularity) | 1 × valid | 1 × valid | none | as rank 1 | equal bytes, but the tile count caps the lane count (2 lanes for `[16,1,64]→[1,1,16,64]`); strip runs subsume it |
| 5 | whole tiles in, whole tiles out | all input tiles incl. padding | all output tiles incl. padding | none | `num_tiles_in + num_tiles_out` | 16–32× bytes → rejected |
| 6 | split + cross-core routing | 1 × valid | 1 × valid | +1 NoC crossing per segment | 3 per segment | rejected |
| — | dependent-axis split + combine | — | — | — | — | not applicable: the op has no dependent axis, since no logical value spans an axis |

**Operand-reuse check.** There is one input. Along the chosen split every lane reads a disjoint set of input face rows (the map is a bijection), so the input is not reuse-shared by construction of the split. The only shared bytes are a `gather_general` granule that straddles a lane boundary: at most 1 granule per boundary. No broadcast/multicast regime exists.

**Stall-shadow check.**

| Wait | What can run in it |
|------|--------------------|
| A lane's read barrier in `gather_*` / `scatter_aligned` | Work independent of the wait: (a) the previous block's tile-side writes, which are async and already in flight; (b) copies of staged granules that have already landed, which needs scoped barriers (trid lamp); (c) the next block's reads, which needs `slot_depth = 2` (read-ahead lamp). Every reorder is legal because pure moves are bit-identical under any order, so no precision gate applies. |
| A lane's slot-reuse flush (`noc_async_writes_flushed`) | It waits only for write data to leave L1, not for acks, so nothing is worth scheduling into it. |
| The final `noc_async_write_barrier` | It closes the program (BB-03). Nothing in the lane is independent of it. |

### Block schedule

The schedule below runs per lane: the same source on NCRISC/NoC0 (lane 0) and BRISC/NoC1 (lane 1), on disjoint strip runs.

```cpp
// gather_aligned / gather_general                 // scatter_aligned
for (block_idx = 0; block_idx < num_blocks_this_lane; ++block_idx) {   for (block_idx ...) {
    gather_block(block_idx);    // map-side reads                       load_block(block_idx);     // tile-side reads
    assemble_block(block_idx);  // barrier + staged copies              scatter_block(block_idx);  // map-side writes
    store_block(block_idx);     // tile-side writes (async)         }
}                                                                       noc_async_write_barrier();
noc_async_write_barrier();
```

| Block operation | Regime | Block shape | What it does | Resident across it | Intended frequency of fixed costs |
|-----------------|--------|-------------|--------------|--------------------|-----------------------------------|
| `gather_block` | `gather_*` | the block's strips: ≤ `block_tiles` output tiles × the lane's strips in them × segments | Slot `s = block_idx mod slot_depth`. If `block_idx ≥ slot_depth`, first `noc_async_writes_flushed()` (slot reuse). Then walk the block's segments. Per piece, compute the source (input tile id + byte offset) by the index map and the destination `slot + (t − block_first_tile)·tile_bytes + face_off(r, 16·fh) + piece offset`. `gather_aligned`: a **direct** `noc_async_read` of the face row when the congruence holds, else **staged**. `gather_general`: every piece is staged. Staged means: read the granule that contains the piece into the next staging slot unless it equals the last staged granule (memo), then append `(staging_addr + offset, dst_addr, bytes)` to the copy table. When the slots or the table are full, flush (barrier + table copies, reset slots, table and memo) and continue; a flush may fall at any piece boundary | the slot (`block_tiles` tile images) | 1 slot-reuse flush per block (none on golden shapes); extra flushes `= ceil(staged pieces / min(staging_slots, copy_entries)) − 1`, which is 0 on golden shapes; TensorAccessors built once per kernel |
| `assemble_block` | `gather_*` | the block's copy table | `noc_async_read_barrier()`, which also invalidates the BH RISC-V L1 cache (`dataflow_api.h:1921-1935`). Execute the copy table with RISC-V loads/stores (`uint32_t` words for whole face rows; `esize`-wide elements for partial pieces). Then one `fence` so the stores are ordered before the NoC writes that source them | slot + staging | 1 barrier + 1 copy pass + 1 fence per block |
| `store_block` | `gather_*` | the block's tiles × the lane's rows in each | Per tile, the lane's row range `[r_lo, r_hi)` gives `tile_runs(r_lo, r_hi, vc)` merged face runs. Each run is ONE `noc_async_write` from `slot + (t − first)·tile_bytes + run_off` to output page `t` + `run_off`. No flush here: the next block's `gather_block` flushes before it reuses a slot | the slot being drained | ≤ 4 writes per tile (1 for a lane holding all 16 or all 32 rows) |
| `load_block` | `scatter_aligned` | ≤ `block_tiles` input tiles × the lane's rows in each | Slot reuse as in `gather_block`. Per tile, `tile_runs(r_lo, r_hi, vc)` merged runs; each is ONE `noc_async_read` from input page `t` + `run_off` to `slot + (t − first)·tile_bytes + run_off` (always congruent). Then `noc_async_read_barrier()` | the slot | 1 slot-reuse flush + 1 barrier per block; ≤ 4 reads per tile |
| `scatter_block` | `scatter_aligned` | the block's segments | Per segment, compute the output tile id + byte offset of its whole destination face row by the index map, then `noc_async_write` `face_row_bytes` from the slot face row (always 16-B congruent) | the slot being drained | no barrier per block; the next block's slot reuse flushes |
| kernel end | all | — | `noc_async_write_barrier()` once (BB-03). Reads end with the block's barrier | — | once per lane |

**Index map.** One header, `kernels/g6_reshape_index_map.hpp`, used by both regimes. Divisors are CT constants, so division compiles to multiply-high; the implementer may walk `L` incrementally instead.

| Direction | Map |
|-----------|-----|
| output segment → input pieces (`gather_*`) | `g_o = img·H_out + 32·tr + r`; `c0 = 32·tc + 16·fh`; `n = min(16, W_out − c0)`; `L = g_o·W_out + c0`. For `q ∈ [L, L+n)`: `g_i = q div W_in`, `c_i = q mod W_in`, `img_i = g_i div H_in`, `h_i = g_i mod H_in`; tile `(img_i·Ht_in + h_i div 32)·Wt_in + c_i div 32`, offset `face_off(h_i mod 32, c_i mod 32)`. A piece ends at `c_i mod 16 == 15`, at `c_i == W_in − 1`, or at `n`. |
| input segment → output face row (`scatter_aligned`) | `g_i = img·H_in + 32·tr + r`; `L = g_i·W_in + 32·tc + 16·fh`; `g_o = L div W_out`, `c_o = L mod W_out`; tile `((g_o div H_out)·Ht_out + (g_o mod H_out) div 32)·Wt_out + c_o div 32`, offset `face_off((g_o mod H_out) mod 32, c_o mod 32)`. `aligned` gives `c_o mod 16 == 0`. |

**Padding rule** (spec rule 2). Only logical positions are ever addressed as sources. A staged granule may contain padding bytes, but only piece bytes are copied out of it. A partial destination segment (`n < 16`, `gather_general`) leaves face-row bytes `n..15` of the slot as stale L1; those bytes land in output padding columns, which the contract leaves unspecified. Output padding rows are never written.

### Perf lamps

| Lamp | Why the default may be wrong here | Nearby alternative to measure |
|------|-----------------------------------|-------------------------------|
| Grid synchronization (`MIN_STRIPS_PER_LANE = 1`, `LANES_PER_CORE = 2`) | 10 of the 14 production shapes put 1–2 strips on a lane, so per-core fixed costs (launch, per-lane RT args, accessor setup, one DRAM round trip, the final barrier) may dominate | `MIN_STRIPS_PER_LANE ∈ {2, 4, 8, 16}`; judge on unprofiled traced-replay time |
| NoC direction (lane 1 reads on NoC1) | catalog `noc_placement` measured reads on NoC1 2.5–4.8× slower for bandwidth-bound spread lines; this op's transfers are small and issue-bound, so the penalty may or may not apply | `LANES_PER_CORE = 1` at the same `n_lanes` (twice the cores) on `[16,1,5120]→[1,1,16,5120]` and `[1,1,16,4608]→[16,1,4608]` |
| Read-ahead (catalog `trid_double_issue`; the stall-shadow reorder) | a multi-block lane idles one DRAM round trip per block at its read barrier | `SLOT_DEPTH = 2`: issue block `b+1`'s reads under trid `B` before `async_read_barrier_with_trid(A)` for block `b` (`local_copy_helpers_dataflow.hpp:147,158`). Measure only on shapes with `max_blocks_per_lane > 1` |
| Trid split inside a block | `assemble_block` waits for every read, including direct reads it does not depend on | staged reads on trid A, direct reads on trid B: barrier A → copies → barrier B |
| Issue cost per transfer | map-side transfers are fixed-size face rows, but `noc_async_read/write` reprogram the whole command buffer each call | `noc_async_read_one_packet_set_state` / `_with_state` (`dataflow_api.h:575,599`) and the write equivalents for the face-row stream |
| Regime predicate near-ties | `[1,8,6,256]→[1,1,8,1536]` compares 128 against 96 tile-side transfers and selects `gather_aligned`, which stages about half its 768 reads | `REGIME_OVERRIDE = "scatter_aligned"` on `[1,H,6,256]→[1,1,H,1536]` and `[2,1,5120]→[1,1,2,5120]` |
| Face-pair read (catalog `read_coalescing`) | 1-row input tiles (`gather_aligned` on `[H,1,W]→[1,1,H,W]`) cost 2 reads of 32 B per input tile | one 544-B read (face 0 row 0 … face 1 row 0) into staging + 2 copies: half the reads, 8.5× the bytes on those tiles |
| Span vs merged face-run writes (catalog `tile_reorder`) | a lane holding 1–15 rows of a tile writes 2 face runs; one larger write may beat two small ones | one write spanning the run's first to last byte (the padding rows in between are output padding) |
| Bank stagger (catalog `bank_stagger`) | in `[H,1,W]→[1,1,H,W]` (W ≡ 0 mod 256), all H sources of one output tile lie in ONE bank (`(h·Wt + tc) mod 8 = tc mod 8`), so a lane's reads serialize on one bank | rotate each lane's segment walk start by `lane_idx mod strips_in_block` |

## Dataflow Strategy

| Stage | Format | Mechanism | Notes |
|-------|--------|-----------|-------|
| DRAM → `cb_laneK_tile_images` (tile side, `scatter_aligned`) | input merged face runs (≤ 2048 B bf16) | `noc_async_read(TensorAccessor(input).get_noc_addr(t, run_off), slot_addr, bytes)` | identical intra-tile offset on both sides ⇒ always congruent |
| DRAM → `cb_laneK_tile_images` (map side, `gather_aligned` direct) | one face row (32 B bf16) | `noc_async_read` | only when `(dst ^ src) & (NOC_DRAM_READ_ALIGNMENT_BYTES − 1) == 0` |
| DRAM → `cb_laneK_staging` (map side, staged) | one `granule_bytes` granule | `noc_async_read` of the granule containing the piece into the next staging slot | single-entry memo on the granule's NoC address |
| `cb_laneK_staging` → `cb_laneK_tile_images` | piece bytes | RISC-V loads/stores after the barrier, then a `fence` | lane-private |
| `cb_laneK_tile_images` → DRAM (tile side, `gather_*`) | output merged face runs of the lane's rows | `noc_async_write(slot_addr, TensorAccessor(output).get_noc_addr(t, run_off), bytes)` | one write per merged run |
| `cb_laneK_tile_images` → DRAM (map side, `scatter_aligned`) | one face row per segment | `noc_async_write` | the 16-B write congruence holds for every face row |
| (no compute kernel) | — | — | bit-exact by construction |

Placement: lane 0 is `g6_reshape_lane.cpp` on NCRISC / NoC0 (`ttnn.ReaderConfigDescriptor()`), and lane 1 is the same source on BRISC / NoC1 (`ttnn.WriterConfigDescriptor()`); the source learns its lane from a CT arg. No regime, built or deferred, uses inter-Tensix communication: no semaphores, no multicast, no cross-lane credit.

**TARGET as regimes.**

| TARGET value | Regime(s) | Class |
|--------------|-----------|-------|
| `bfloat16` × `TILE_LAYOUT` | `gather_aligned`, `scatter_aligned`, `gather_general` | built (Phase 0) |
| `float32` × `TILE_LAYOUT` (FLUX `[1,W]→[1,1,W,1]`) | the same three with `esize = 4`: granule = face row = 64 B, so `gather_aligned` is all-direct and `strip_rows = 1`. The 5 FLUX cases select `gather_general` | knob-turn: `esize` is a CT arg and every offset derives from it |
| `int32`, `uint32` × `ROW_MAJOR_LAYOUT` | `rm_page_gather` + `rm_bank_slot_writes` | regime addition on the same lane skeleton (a new address function, no new pipeline); arrives together with ROW_MAJOR in one change |

TARGET has no placement axis (every case is DRAM interleaved), so no sharded realization is designed. No index axis (`dim`) exists, so nothing is canonicalized.

**Structural impossibilities:** none. This agrees with `feature_spec.INVALID = []`.

## Work Distribution

| Field | Value |
|-------|-------|
| Work unit | the strip of the tile-side tensor. A block is ≤ `block_tiles` consecutive tile-side tiles of the lane's run |
| Grid | `device.compute_with_storage_grid_size()`, a runtime query and never a literal; `grid_cores = x·y` |
| Lanes and cores | `n_lanes = min(LANES_PER_CORE·grid_cores, ceil(U / MIN_STRIPS_PER_LANE))`; `n_cores = ceil(n_lanes / LANES_PER_CORE)`; cores = `ttnn.num_cores_to_corerangeset(n_cores, grid, row_wise=True)`, which gives ≤ 2 rectangular ranges (BB-17). Both lane kernels and all CBs sit on that one CoreRangeSet. Global lane index = `LANES_PER_CORE·core_idx + lane`, with `core_idx` in the row-wise order of that set |
| Per-lane work | contiguous strip runs. The first `U mod n_lanes` lanes take `ceil(U / n_lanes)` strips and the rest take `floor(U / n_lanes)`. With an odd `n_lanes`, the last core's lane 1 gets 0 strips and returns at once. The host computes each lane's `(start_tile, start_strip)` by the closed form, plus its `num_strips` and `num_blocks_this_lane` |
| Remainder | strip counts differ by ≤ 1 across lanes. Tile geometry uses `ceil` and per-image `Ht` (`num_tiles = IMG·ceil(H/32)·ceil(W/32)`), never `floor(IMG·H/32)` |
| Production consequence (110-core grid, 220 lanes) | `[16,1,5120]→[1,1,16,5120]` 1280 strips → 110 cores; `[8,1,5120]→…` 640 → 110; `[1,1,16,4608]→[16,1,4608]` 1152 → 110; `[1,1,8,4608]→…` 576 → 110; `[1,16,6,256]→[1,1,16,1536]` and `[16,1,1536]→…` 384 → 110; `[1,8,6,256]→…` and `[8,1,1536]→…` 192 → 96; `[2,1,5120]→[1,1,2,5120]` 160 → 80; `[1,1,6,256]→[1,1,1536]` 24 → 12; `[16,1,64]→…`, `[1,16,64]→…` 16 → 8; `[8,1,64]→…`, `[1,8,64]→…` 8 → 4 |
| Plan report (BB-08, BB-21) | `g6_reshape_plan(input_tensor, shape)` returns at least: `regime`, `tile_side` (`"output"`/`"input"`), `aligned`, `work_units` (U), `strip_rows`, `lanes_per_core`, `min_strips_per_lane`, `grid_cores`, `n_lanes`, `n_cores`, `num_core_ranges`, `core_ranges`, `block_tiles`, `max_blocks_per_lane`, `slot_depth`, `staging_slots`, `copy_entries`, `strips_per_lane` (min, max), `l1_bytes_per_core`, and `roles` (`{"NCRISC": "lane0", "BRISC": "lane1"}`, the same on every core) |
| Profiling zones (BB-21) | `DeviceZoneScopedN` around every piece of work and every wait, with the same names on both lanes: `G6_SLOT_REUSE`, `G6_MAP_ISSUE`, `G6_TILE_ISSUE`, `G6_READ_BARRIER`, `G6_ASSEMBLE`, `G6_FINAL_WRITE_BARRIER`. Without the profiler the macro is `(void(sizeof(name)))` (`kernel_profiler.hpp:1047`), so the plain build is unchanged |
| Program description cache (BB-07, BB-16) | CT args: geometry of both tensors (`W, H, IMG, Wt, Ht`), `esize`, `strip_rows`, `granule_bytes`, `block_tiles`, `slot_depth`, `staging_slots`, `copy_entries`, lane id, CB indices, `TensorAccessorArgs(input)`, `TensorAccessorArgs(output)`; plus the regime define. Per-lane RT args: `start_tile`, `start_strip`, `num_strips`, `num_blocks_this_lane`. **Common RT args: `[input_address, output_address]`**, the same on every core. The Python side memoizes `(ProgramDescriptor, plan)` in a module dict keyed by `(tuple(d), tuple(e), dtype, layout, memory_config, device id, grid size, LANES_PER_CORE, MIN_STRIPS_PER_LANE, BLOCK_SLOT_BYTES_CAP, SLOT_DEPTH, STAGING_SLOTS_CAP, COPY_ENTRIES_CAP, REGIME_OVERRIDE)`, never by a buffer address. A hit rewrites only the two common RT args of both kernels. There is no host read of tensor values and no allocation besides the output; the first (eager) call builds and compiles, so a traced call allocates only its output and builds nothing |

## Circular Buffers

All four CBs are **lane-private scratch**. Each lane takes its CBs' base once with `get_write_ptr` and never calls `cb_reserve_back` / `cb_push_back` / `cb_wait_front` / `cb_pop_front`. A CB carries no credit; it is an L1 allocation that one RISC-V both fills and drains.

| Semantic Name | Index | Page Size | Num Pages | Sizing rationale | Format | Producer | Consumer | Lifetime |
|---------------|-------|-----------|-----------|------------------|--------|----------|----------|-----------|
| `cb_lane0_tile_images` | 0 | `tile_bytes` (2048 B bf16) | `slot_depth · block_tiles` | spans `t_s` (`block_tiles`), `strip` and `r` (a tile image holds all 32 row positions), `fh`, `e`; streams over blocks; map-side axes are never resident. Holds output tile images in `gather_*` and input tile images in `scatter_aligned` | input dtype as raw bytes (never unpacked) | lane 0 (NCRISC) | lane 0 (NCRISC) | whole kernel; ring of `slot_depth` block slots |
| `cb_lane1_tile_images` | 1 | `tile_bytes` | `slot_depth · block_tiles` | as `cb_lane0_tile_images` | raw bytes | lane 1 (BRISC) | lane 1 (BRISC) | whole kernel |
| `cb_lane0_staging` (`gather_*` only; not allocated for `scatter_aligned`) | 2 | `granule_bytes` (64 B on BH) | `staging_slots + ceil(8·copy_entries / granule_bytes)` | the granule ring (one page per staged granule) plus the copy table in the tail (8 B per entry). Live set = staged pieces between two flushes; streams over strips and blocks | raw bytes | lane 0 (NCRISC) | lane 0 (NCRISC) | whole kernel |
| `cb_lane1_staging` (`gather_*` only) | 3 | `granule_bytes` | as `cb_lane0_staging` | as `cb_lane0_staging` | raw bytes | lane 1 (BRISC) | lane 1 (BRISC) | whole kernel |

When `LANES_PER_CORE = 1`, the lane-1 CBs are not allocated. The full ledger, axis accounting and data-movement budget are in `l1_ledger.md`.

## Block Operation Realization

| # | Block operation | Block shape | Helper? | Input CB (semantic name, pages, state) | Output CB (semantic name, pages) | CB state after |
|---|-----------------|-------------|---------|----------------------------------------|----------------------------------|----------------|
| 1 | `gather_block` | ≤ `block_tiles` output tiles × the lane's strips × segments | no compute helper applies (dataflow only): TensorAccessor + `noc_async_read` | input tensor (DRAM); `cb_laneK_staging` ring | `cb_laneK_tile_images`, slot `block_idx mod slot_depth` | direct reads and staged granules in flight |
| 2 | `assemble_block` | the block's copy table | RISC-V copies via `addr_to_l1_ptr` | `cb_laneK_staging` (landed granules) | `cb_laneK_tile_images` slot | slot complete; staging reset |
| 3 | `store_block` | the block's tiles × the lane's rows | TensorAccessor + `noc_async_write` | `cb_laneK_tile_images` slot | output tensor (DRAM) | writes in flight; flushed before slot reuse |
| 1' | `load_block` | ≤ `block_tiles` input tiles × the lane's rows | TensorAccessor + `noc_async_read` | input tensor (DRAM) | `cb_laneK_tile_images` slot | slot complete after the barrier |
| 2' | `scatter_block` | the block's segments | TensorAccessor + `noc_async_write` | `cb_laneK_tile_images` slot | output tensor (DRAM) | writes in flight; flushed before slot reuse |

## API Mapping

| Block operation | Type | Function | File:Line | Template Params / Args | Input CB | Output CB | Which params are block knobs |
|-----------------|------|----------|-----------|------------------------|----------|-----------|------------------------------|
| all transfers | raw_api | `TensorAccessor::get_noc_addr(page_id, offset)` | `tt_metal/hw/inc/api/tensor/tensor_accessor.h:165` (interleaved specialization `:554`) | page = tile id; offset = `face_off(...)`, `run_off`, or the granule-aligned offset | — | — | — |
| `gather_block`, `load_block` | raw_api | `noc_async_read` | `tt_metal/hw/inc/api/dataflow/dataflow_api.h:547` | `(src_noc_addr, dst_l1, bytes)` | — | `cb_laneK_tile_images` / `cb_laneK_staging` | — |
| `assemble_block`, `load_block` | raw_api | `noc_async_read_barrier` | `dataflow_api.h:1921` | once per block (+ once per flush) | — | — | frequency set by `block_tiles`, `staging_slots`, `copy_entries` |
| `assemble_block` | helper | `addr_to_l1_ptr` | `ttnn/cpp/ttnn/kernel_lib/l1_helpers.hpp:30` | `uint32_t` word copies for whole face rows; `uint16_t`/`uint32_t` element copies for partial pieces (as `tile_column_dataflow.hpp:92` does) | `cb_laneK_staging` | `cb_laneK_tile_images` | — |
| `store_block`, `scatter_block` | raw_api | `noc_async_write`, `noc_async_writes_flushed`, `noc_async_write_barrier` | `dataflow_api.h:876`, `:1980`, `:1951` | one write per merged run / per segment; flush before slot reuse; barrier at kernel end | `cb_laneK_tile_images` | — | `block_tiles`, `slot_depth` set the flush frequency |
| all | raw_api | `get_write_ptr`, `get_arg_val`, `get_common_arg_val` | `dataflow_api.h:325`, `:148`, `:168` | CB base once per kernel; per-lane and common RT args | — | — | — |
| all | raw_api | `DeviceZoneScopedN` | `tt_metal/hw/inc/api/debug/kernel_profiler.hpp:1012`, `:1047` | zone names above | — | — | — |
| (lamp: read-ahead, trid split) | helper | `dataflow_kernel_lib::set_read_trid`, `async_read_barrier_with_trid` | `ttnn/cpp/ttnn/kernel_lib/local_copy_helpers_dataflow.hpp:147`, `:158` | one trid per block / read class | — | — | `SLOT_DEPTH` |
| (lamp: issue cost) | raw_api | `noc_async_read_one_packet_set_state`, `noc_async_read_one_packet_with_state` | `dataflow_api.h:575`, `:599` | fixed face-row size | — | — | — |

**Helpers considered and rejected.** These cover every `raw_api` row above.

| Candidate | File:Line | Why it cannot express this op's block operation |
|-----------|-----------|-----------------------------------------------|
| `compute_kernel_lib::untilize` | `ttnn/cpp/ttnn/kernel_lib/untilize_helpers.hpp:163` | It consumes whole input tiles through unpack → DEST → pack. Padding tiles are read (regime `whole_tiles_in_out`), values pass the compute engine on a bit-exact contract, and the regroup between untilize and tilize still needs this design's index-map copies |
| `compute_kernel_lib::tilize` | `ttnn/cpp/ttnn/kernel_lib/tilize_helpers.hpp:206` | the mirror of the row above, for the same three reasons |
| `read_sticks_for_tilize`, `write_sticks_after_untilize` | `ttnn/cpp/ttnn/kernel_lib/tilize_helpers_dataflow.hpp:88`, `:130` | They move whole row-major sticks of a ROW_MAJOR tensor into/out of a CB with credit. Phase 0 tensors are TILE and the move unit is a face-row segment inside a tile page, so no stick exists on either side (they are candidates for `rm_page_gather`) |
| `copy_tile_column`, `fill_tile_column` | `ttnn/cpp/ttnn/kernel_lib/tile_column_dataflow.hpp:92`, `:70` | They copy one *column* of one tile into a column of another, and the template needs a DataflowBuffer `dfb_id` for tile dims. A `gather_general` piece is a contiguous run of a *face row* of a staged granule, not a tile column |
| `dataflow_kernel_lib::local_addr`, `set_read_state` / `read_with_state` | `local_copy_helpers_dataflow.hpp:57`, `:75`, `:95` | NoC self-copies need an issue + a barrier per batch. A staged copy is 32 B (8 words) or a few elements, which is cheaper as RISC-V stores, and `read_with_state` fixes one size per loop while `gather_general` pieces vary. The implementer may use them for long pieces |
| `ttnn.Mcast` / `mcast/kernel/mcast_args.hpp` | — | no regime has a reuse-shared operand or a cross-core step |
| eltwise / reduce / matmul helpers | — | the op has no arithmetic |

## Key Risks and Gotchas

| Risk | Why it bites here | Mitigation in this design |
|------|-------------------|---------------------------|
| Input padding reaching a logical output | The golden suite poisons input padding with NaN (`fill_implicit_tile_padding`); the zero padding `from_torch` writes hides a leak | Only logical positions are addressed as sources. Staged granules contribute only piece bytes, and the `aligned` regimes move whole valid face rows only. Acceptance test poisons one case per regime |
| BH 64-B DRAM read congruence | 32-B face rows sit at 32-B row stride, and source/destination parity differs on ~50 % of `gather_aligned` segments | runtime check per segment, staging otherwise; new kernels run first with `--dev` (NoC sanitizer, CB bounds) |
| RISC-V copies vs the NoC write that sources them | `store_block` NoC-reads slot bytes that `assemble_block` wrote with RISC-V stores | BH L1 cache is write-through (`cache.h:7-16`). One `fence` after the copy pass orders the stores before the write commands. The lane's own read barrier invalidates its cache before it loads staged bytes |
| Two lanes writing one 64-B DRAM granule | The tile side never does (strips are granule-aligned). The map side of `scatter_aligned` can, for non-production shapes: output rows `r` and `r^1` of one face may come from different lanes' strips | Byte-masked NoC writes to DRAM. Pinned by the acceptance case `[1,1,30,64]→[10,3,64]`, where image-1 rows 0 and 1 come from strips 1 and 2 on different lanes. If it fails, the fix is predicate-level: route such shapes to `gather_aligned`, whose tile side is granule-aligned |
| Lane boundary inside a tile | strip granularity: several lanes write rows of one output tile (`[16,1,64]→[1,1,16,64]`: 4 lanes per tile) | each lane writes only its own strips' merged runs; byte ranges are disjoint and granule-aligned |
| Program cache replaying a wrong map (BB-07) | `[16,1,64]→[1,1,16,64]` then `→[1,1,1,1024]` on one input share the input geometry | output geometry and the regime define are CT; the memo key holds `e` and every knob; acceptance test calls both targets and re-calls the first |
| Stale buffer address on a memo hit | the memo returns a descriptor built for another tensor | addresses live only in the two common RT args, rewritten on every call; acceptance test reuses a signature with fresh tensors |
| Rank-1 / rank-2 TILE tensors | rank 1 has `H = 1` and padded shape `[32, 32]` | derive `IMG, H, W` from the logical shape exactly as the geometry table says; assert `IMG·Ht·Wt == buffer_num_pages()` on both tensors |
| Staging overflow mid-segment (`gather_general`, `W_in < 16`) | up to 16 pieces per segment, one granule each before memo hits | flush at any piece boundary when the slots or the table are full; the multi-block knob test drives both caps to 4 |
| Lane with zero strips | odd `n_lanes` leaves the last core's lane 1 empty | the kernel returns before any transfer; no CB credit exists, so nothing waits on it |
| Multi-block path untested by production shapes | full occupancy puts every golden shape at one block per lane | acceptance test shrinks `BLOCK_SLOT_BYTES_CAP` and the staging caps, raises `MIN_STRIPS_PER_LANE`, sets `SLOT_DEPTH = 2`, and asserts `max_blocks_per_lane > 1` per regime |
| Transactions in flight at kernel exit (BB-03) | another program starts right after | each lane ends with `noc_async_write_barrier()`; every read is retired by its block's barrier before use |
| Kernel text size (BB-04, BB-18) | three regimes in one source, two instances per core | one regime define per build; no compute kernel; geometry as CT constants keeps the address math small |
