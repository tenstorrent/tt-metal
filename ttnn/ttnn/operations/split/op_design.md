# Operation Design: split

## Overview

| Field | Value |
|-------|-------|
| Classification | data_movement (no compute kernel: every chunk element is an input element, moved unchanged — spec R1) |
| Goal | Cut `input_tensor` along `dim` into `ceil(shape[dim] / split_size)` consecutive chunks, each a tensor of its own in the input's buffer (spec R2, R5), in **one** `ttnn.generic_op` dispatch that writes every chunk. |
| Math | `chunk[c][..., h', ...] = input[..., c*split_size + h', ...]` for `0 <= h' < min(split_size, shape[dim] - c*split_size)` along `dim`; every other index unchanged |
| References | Spec: `eval/specs/split/spec.py` (governs). Target: `.claude/references/targets/quasar.md`. Backend: `.claude/references/backends/ttsim.md`. Host program pattern: `.claude/references/quasar_op_examples/` (README, `toy_thread_plan/toy_thread_plan_program_artifacts.py`, `toy_thread_plan/kernels/reader_implicit.cpp`, `writer_implicit.cpp`, `reader.cpp`). Rings: `tt_metal/impl/dataflow_buffer/DataflowBuffer.md` Part A. Blocking: `.claude/references/blocking-model.md`. Ledger: `ttnn/ttnn/operations/split/l1_ledger.md`. DM-only `ProgramSpec` precedent (reader + writer, no compute kernel): `ttnn/cpp/ttnn/operations/experimental/quasar/transpose/device/transpose_cn_program_factory.cpp`. |

## Target

| Fact | Value | Source |
|------|-------|--------|
| Target | `quasar` on the ttsim simulator | `.claude/references/targets/quasar.md` + `.claude/references/backends/ttsim.md` |
| Arch | quasar | target page |
| Compute grid | 8 × 4 = 32 clusters (worked numbers below use 32); the op queries `device.compute_with_storage_grid_size()` | target page |
| L1 per core | 4,194,304 B (4 MiB) per cluster | target page |
| Unreserved budget | `ttnn.get_max_worker_l1_unreserved_size() - l1_reserve_bytes`; the value on quasar is not recorded in the profile. **The op evaluates no budget predicate**: its per-core footprint is bounded by constants and the NoC-burst cap (≤ 128 KiB, `l1_ledger.md`), far below 4 MiB | `unknown` (value); target page (query) |
| Host program | `ttnn.ProgramSpec` (Metal 2.0) through `ttnn.generic_op(io_tensors, spec, run_args, tensor_args)` — reference `.claude/references/quasar_op_examples/` | target page; `ttnn/cpp/ttnn/operations/generic/generic_op_nanobind.cpp:70-81` |
| Threads per core | 4 NEOs (C ∈ {1,2,4}), 6 user DM harts (R + W ≤ 6; on STRIDED rings R and W each divide C). This op runs **no compute kernel**: plan (R, C, W) = (1, 0, 1), Work Distribution | target page *Thread plan* |
| Unavailable here | `ttnn.ProgramDescriptor` and the legacy program-factory device op (Metal 1 kernels refused on this part); `bfloat8_b` / `bfloat4_b` (no path; not in TARGET anyway); uint16/uint32 as DFB formats (not needed: bfloat16 / float32 only). Binding here: **the sticky NoC transaction tag**: an implicit-sync kernel hands the tag back as its last statement; an explicit-sync kernel tags every transfer. | target page *Not available* |

## Parameters

| Name | Type | Required | Valid Range | Default | CT/RT |
|------|------|----------|-------------|---------|-------|
| `input_tensor` | `ttnn.Tensor` | yes | on device, interleaved, DRAM or L1 (R4, R6) | — | host (tensor parameter `input`) |
| `split_size` | `int` (not `bool`, not a list: X3) | yes | `> 0` (X1) | — | host only; kernels see the derived geometry |
| `dim` | `int` | no | `[-rank, rank-1]` (X2); canonicalized to `dim if dim < 0 else dim - rank` **before** the support check (R3) | `0` | host only |

Host-side order in `split()`: (1) `split_size` type check (list/tuple/bool/non-int → `TypeError`, X3); (2) `split_size <= 0` → `ValueError` (X1); (3) `dim` out of range → `IndexError` (X2); (4) canonicalize `dim` (R3); (5) derive axes with `INPUT_TAGGERS = AXES` (imported from the spec, never redefined) on `(tuple(shape), split_size, dim)` plus `eval.memory.memory_axes(input_tensor)` (R6) plus dtype/layout; (6) `validate()` raises `NotImplementedError` outside SUPPORTED (R7). SUPPORTED after Phase 0 = spec `PHASE0` exactly.

## Tensors

### Input

| Property | Requirement |
|----------|-------------|
| Shape | rank 3 `[B, H, W]` in Phase 0 (TARGET: rank 2–4) |
| Dtype | bfloat16 (Phase 0); TARGET adds float32 |
| Layout | TILE or ROW_MAJOR |
| Memory | interleaved, DRAM or L1 |

### Output

| Property | Value |
|----------|-------|
| Shape | a Python `list` of `n_chunks = ceil(S / split_size)` tensors, `S = shape[dim]`; chunk `c` has `shape[dim] = chunk_extent[c] = min(split_size, S - c*split_size)`, every other dim unchanged |
| Dtype | input's |
| Layout | input's |
| Memory | interleaved, same buffer type as the input (`ttnn.DRAM_MEMORY_CONFIG` / `ttnn.L1_MEMORY_CONFIG`), each chunk allocated with `ttnn.allocate_tensor_on_device(TensorSpec(chunk_shape, dtype, layout, memory_config), device)` (R5). All chunks go into one `io_tensors = [input, chunk_0, …, chunk_{n-1}]` with `tensor_args = {"input": 0, "chunk_0": 1, …}`; `generic_op`'s return value (the last tensor) is ignored and the list is returned. |

## Blocking Model

The first design section. Everything below it is downstream of these decisions.
Semantics: `.claude/references/blocking-model.md`.

### Canonical page view (single source of every geometry quantity)

Every regime works on **pages** of the interleaved buffers (a stick in ROW_MAJOR, a 32×32 tile in TILE) and on one flattened **walk** over the units it moves. The host derives, once, in `split_program_artifacts.py`:

| Symbol | ROW_MAJOR, `d ≤ -2` | TILE, `d = -2` | TILE, `d ≤ -3` (later) |
|---|---|---|---|
| `tensor_outer` | `prod(shape[:d])` | `prod(shape[:-2])` | `prod(shape[:d])` |
| `tensor_cut_pages` (page-rows along the cut) | `shape[d]` | `tensor_h_tiles = ceil(H/32)` (per image) | `shape[d]` |
| `tensor_inner_pages` (pages per page-row) | `prod(shape[d+1:-1])` (1 for `d=-2`) | `tensor_w_tiles = ceil(W/32)` | `prod(shape[d+1:-2]) * ceil(H/32) * ceil(W/32)` |
| `page_bytes` | `input.buffer_aligned_page_size()` | `input.buffer_aligned_page_size()` (2048 bf16 / 4096 fp32) | same |

Input page id `(o, p, i) = (o*tensor_cut_pages + p)*tensor_inner_pages + i`. Chunk `c`, row `h'` of the cut axis (in elements) maps to input row `c*split_size + h'`. All tile counts are `ceil`, per image — never `floor(B*H/32)` or `W // 32`.

### Axes

Phase 0 is rank 3 `[B, H, W]`, `dim = -2`. The op's data axes and the walk they fold into:

| Axis | Character (+ one-clause reason) | Extent knob | Phase 0 value | Knob source | Core-assignment | Later unlock |
|------|--------------------------------|-------------|---------------|-------------|-----------------|--------------|
| `outer` (B; all dims before `dim`) | independent — an output page reads input pages of the same `o` only | folded into the walk; per-core extent = what the core's walk range covers | whole range assigned | `split_work_to_cores` result → RT args `start_unit`, `num_units` | contiguous walk ranges over the full grid (`row_wise=True`) | knob-turn |
| `cut` (H; the split axis, incl. its chunk index `c = h // split_size`) | independent — every output row is one input row; no result spans the axis. In `tile_row_shift` the footprint **overlaps**: an output tile-row reads up to 2 input tile-rows | `block_h_tiles` (tile_row_shift: output tiles of one column whose input reads share one DRAM barrier); page_remap: folded into the walk | tile_row_shift: `min(BLOCK_H_TILES=8, tiles left in the column segment)`; page_remap: whole range | host constant `BLOCK_H_TILES` → CT arg `block_h_tiles` | contiguous walk ranges (a range may start mid-column: ≤ 1 halo tile per core) | knob-turn |
| `width` (W; pages per page-row) | independent — a column of tiles / a stick never mixes with another | page_remap: folded into the walk (whole page per unit); tile_row_shift: **1 tile column per block** | 1 column (tile_row_shift) | structural (walk order) | contiguous walk ranges; columns are the walk's outer index | knob-turn |
| `within-page` (rows of a tile; bytes of a stick) | independent | not split: a page moves whole (page_remap) or is assembled whole (tile_row_shift) | whole page | `page_bytes` (queried) | not assigned across cores — a page is the NoC transaction unit; cutting it buys no DRAM channel (a page lives in one bank) | scheme-change only for `d = -1` (deferred regimes) |

Why the walk folds the axes to 1-D: in `page_remap` every page is handled identically (one page in, one page out) and the input pages are contiguous in walk order, which is the blocking-model §1 licence to fold. In `tile_row_shift` the walk is **column-major** `(o, tw, c, j)` — for each tile column, the output tiles of chunk 0, then chunk 1, … — because the reuse (the shared input tile at a cut inside a tile, and at a chunk boundary) lies along the cut axis of one column.

Why `width` = 1 column per block in `tile_row_shift` (an extent at 1 needs a reason): the residency that saves bytes is along `cut`; holding several columns at once multiplies the window without removing any read (each page is its own NoC transaction in every case) and there is no per-block init to amortize in a DM-only kernel. The DRAM-barrier amortization the block exists for is bought along `cut` (`block_h_tiles`).

Why `BLOCK_H_TILES = 8` and not the whole column segment: the fixed per-block cost is one DRAM read round trip (barrier) — the catalog (`ttnn/ttnn/operations/examples/double_buffer`, `trid_double_issue`) measures the reads-per-barrier gain saturating at ~4–8; above that the window grows with the op's H (an op-sized scratchpad). 8 is the coarsest value with a measured gain and a constant footprint; the perf lamp below names 16 / whole segment.

Intermediate data: the only intermediate is the assembled output tile (`tile_row_shift`), whose axes are the output's: it is produced and drained one tile at a time on the core that owns it; nothing is computed on a designated core while others wait (no coordinator role exists).

### Buffer-depth knobs

| CB | Depth knob | Phase 0 value | What the depth buys |
|----|------------|---------------|---------------------|
| `cb_output_pages` | `RING_DEPTH` (entries per thread; × 1 thread) | 2 | reader fills entry k+1 while the writer drains entry k (implicit sync splits the ring into two txn-id halves, `DataflowBuffer.md` "Double buffering") |
| `scratch_input_window` (tile_row_shift only) | `WINDOW_TILES = 2 × BLOCK_H_TILES` (derived, not a knob of its own; verifier amendment, below) | 16 | one block's input tiles — a block may span several short tile columns, ≤ m + 1 input tiles per column segment of m output tiles — including the one shared with the previous block, so each input tile is read once per core range |

> **Verifier amendment (Phase 0 verification).** The `width` = 1 column per block rule above is withdrawn for
> `tile_row_shift`. Holding several columns *does* remove barriers: with blocks clamped to one column, every
> Phase 0 `tile_row_shift` entry (all have H ≤ 2 tile-rows per chunk image, the YOLO calls included) read one
> input tile per DRAM barrier, the read-one/barrier trap. A block is now `block_h_tiles` consecutive output
> tiles of the column-major walk, across column boundaries; the window holds `2 × block_h_tiles` tiles indexed
> by read sequence number. Counts and the reasoning are in `l1_ledger.md` (Verifier amendment) and
> `verification_report.md`.

### Mechanism caps

| Mechanism | Cap on which extent | Clamp | What happens unclamped |
|-----------|--------------------|-------|------------------------|
| Implicit-sync transfer = exactly one ring entry, one NoC transaction (`noc.h:922-943`: no size argument) | `page_bytes` ≤ `NOC_MAX_BURST_SIZE` = 256 words × 256 B = 64 KiB (`tt_metal/hw/inc/internal/tt-2xx/quasar/noc/noc_parameters.h:396`) | host: `NotImplementedError` if `page_bytes > 65536` (ROW_MAJOR W > 32768 bf16; unreachable in TARGET, max W 8192) | — |
| Ring size ≤ 1 MiB − 16 B (target page) | `RING_DEPTH × page_bytes` | follows from the 64 KiB cap with `RING_DEPTH = 2` | DFB creation refused |
| Ring entry = whole aligned page | entry size | `entry_size = input.buffer_aligned_page_size()`; host asserts every chunk's `buffer_aligned_page_size()` equals it | a write of a larger entry spills into the next page of the bank (corrupts a neighbouring chunk page) |
| Writer binds one accessor per chunk tensor | `n_chunks` ≤ `MAX_CHUNKS = 4` | TARGET's `chunks` ≤ 4; `validate()` refuses more | missing accessor → build failure / wrong tensor |
| Row runs in `tile_row_shift` are face-row granular | copy offsets/sizes are multiples of `face_row_bytes = page_bytes / 64` (32 B bf16, 64 B fp32) | ≥ L1 alignment 16 B (`ttnn.get_l1_alignment()`) by construction; host asserts `face_row_bytes % l1_alignment == 0` | misaligned self-aimed NoC copy |
| DRAM alignment 64 B (`ttnn.get_dram_alignment()`) | any sub-page DRAM transfer | none issued: DRAM/L1-bank transfers are whole pages only | 32-B-offset DRAM reads — why `dram_subtile_direct` is rejected |

### Regimes

| Regime | Status | Predicate | Block | Data movement vs. minimum | What a bigger block buys |
|--------|--------|-----------|-------|---------------------------|--------------------------|
| `page_remap` | **built** | `layout == ROW_MAJOR and d <= -2`, or `layout == TILE and (d <= -3 or split_size % 32 == 0)` (the selection function below). Phase 0 builds and tests the `d = -2` part; the walk map is written in the general `[outer, cut_pages, inner_pages]` form, so the other dims are a SUPPORTED change plus tests | the core's whole contiguous range of input pages (`num_units`); one pass, one accessor setup | **minimum**: each input page crosses the input tier (DRAM or L1 banks) once, each output page once; every output page is exactly one input page | nothing per block beyond one kernel launch + accessor setup per core (paid once); per page: one tagged read + one tagged write; ISR credit per txn-id half of the ring — `RING_DEPTH` amortizes that (lamp) |
| `tile_row_shift` | **built** (the traced `[1,4,400]` TILE call cuts inside a tile) | `layout == TILE and d == -2 and split_size % 32 != 0` | `block_h_tiles` consecutive output tiles of one tile column (walk `(o, tw, c, j)`), input window `block_h_tiles + 1` tiles | input: each input tile once per column segment → minimum + **≤ 1 halo tile per engaged core** (a range that starts mid-column re-reads the tile the previous core also read; ≤ grid − 1 tiles total, independent of tensor size); output: once; plus core-local L1→L1 copies of each output tile's valid rows once | one DRAM barrier per block instead of per tile (intended: one barrier per `block_h_tiles` outputs); window priming once per column segment |
| `tile_col_shift` | **built** (Refinement 1) — the mirror of `tile_row_shift`: lines are tile rows (`[outer = images·h_tiles, cut = w_tiles, inner = 1]`), walk line-major `(line, c, j)`, the same window / read-sequence logic. Column shifts are element-granular (`w0 % 32` elements), so assembly is RISC-V copies (window read through its `scoped_lock` after the block barrier; entry written through the ring's `scoped_write_lock`): ≤ 4 column runs per tile, the same for every row, over the tile row's valid rows only; shift 0 → one whole-tile self-aimed NoC copy | `layout == TILE and d == -1 and split_size % 32 != 0` | `block_h_tiles` output tiles of the line-major walk (may span tile rows) | minimum + ≤ 1 halo tile per core | as `tile_row_shift` |
| `rm_substick` | **built** (Refinement 1) — a line is one stick (`[outer = sticks, cut = 1 page, inner = 1]`, `page_elems = W`, `unit_elems = split_size`); whole input sticks are read into the window (`block_h_tiles` output sticks per DRAM barrier, each input stick read once per core range) and each chunk's element range is realigned into its ring entry: one self-aimed NoC copy when it starts 16-B aligned (`ttnn.get_l1_alignment()`), else RISC-V element copies. Ring entry = chunk 0's aligned page; when the last chunk's aligned page is shorter (ROW_MAJOR in L1, 16-B pages), the writer switches to explicit sync with writes sized per chunk (an implicit-sync write has no size argument and would spill into the next page of the bank) | `layout == ROW_MAJOR and d == -1` (TILE `d = -1` with `split_size % 32 == 0` is `page_remap` with `tensor_outer = prod(shape[:-2])*ceil(H/32)`, `tensor_cut_pages = ceil(W/32)`, `tensor_inner_pages = 1`) | `block_h_tiles` output sticks | input once (+ ≤ 1 halo stick per core), output once | as `tile_row_shift`; perf lamp: when every chunk boundary is DRAM-aligned (`split_size·elem % 64 == 0`), read each segment straight into its entry (implicit sync, `offset_bytes`), skipping the window |
| `rm_w_segments` (cut sticks into W segments to engage more cores when sticks < cores, e.g. the traced `[1,4,8400]` RM: 4 sticks) | **deferred** — positive reason: an interleaved page lives in one bank, so W-segments of a stick add requesters but no DRAM channel / L1 bank; the 4 traced sticks already span both of the part's 2 DRAM channels. A ragged last segment also breaks the one-entry implicit-sync transfer (no size argument), forcing explicit sync. Reachable: unit becomes `(stick, segment)` with `offset_bytes` in the accessor args (`api/tensor/noc_traits.h:136-143`) | `layout == ROW_MAJOR and tensor_outer*tensor_cut_pages*tensor_inner_pages < num_cores` | `segment_bytes` (multiple of 64 B) | same bytes, more transactions | — |
| `per_output_tile_refetch` | **rejected** — dead end, superseded by `tile_row_shift` everywhere: each output tile fetches its 1–2 input tiles with no residency, so input crosses DRAM up to 2× (`split_size ≥ 32`) or up to `n_chunks`× (`split_size < 32`, several chunks in one input tile) | — | — | up to 4× input | — |
| `dram_subtile_direct` (read row runs straight from DRAM into the output entry) | **rejected** — runs are 32-B (bf16 face-row) granular, below the 64-B DRAM alignment, and ≤ 8 small transactions per tile vs 1 whole page (catalog `tile_reorder`: whole-page transfers win on a DRAM-bound move) | — | — | minimum bytes, 8× transactions | — |
| `program_per_chunk` | **rejected** — violates the one-dispatch-per-invocation rule | — | — | — | — |
| cross-core combine / multicast | **not applicable** — no dependent axis (nothing spans the cut), and no operand is reuse-shared (single input; each input byte lands in exactly one output position) | — | — | — | — |

**Regime-selection function** (host, exact, grid-independent):

```
d = canonical dim
if layout == ROW_MAJOR:      regime = "rm_substick" if d == -1 else "page_remap"
elif d <= -3:                regime = "page_remap"
elif split_size % 32 == 0:   regime = "page_remap"            # d in {-2, -1}
else:                        regime = "tile_row_shift" if d == -2 else "tile_col_shift"
```
Phase 0 reaches `page_remap` (RM, and TILE with `split_size % 32 == 0`) and `tile_row_shift`; Refinement 1 adds `tile_col_shift` and `rm_substick` (one generalized `ShiftWalk`, `kernels/split_walk.hpp`, serves all three shift regimes). The predicate depends on shape and `split_size` only, never on the grid. Regime-pinned cases from INPUTS (all in PHASE0): `page_remap`/TILE — `(1,64,416) s32`, `(4,128,512) s64`, `(1,1024,64) s512`; `page_remap`/RM — every PHASE0 entry in ROW_MAJOR; `tile_row_shift` — `(1,4,400) s2`, `(1,2,64) s1`, `(1,32,50) s16`, `(2,6,512) s3`, `(1,34,4096) s17`, `(1,4,8400) s2`. The implementer adds a test asserting which regime each of these selects (call the host selection function directly).

### Traffic ranking

Counted per tensor, per tier (input tier = DRAM or the L1 banks the interleaved tensor lives in):

| Candidate split | Input crossings | Output crossings | Cross-core | Notes |
|---|---|---|---|---|
| contiguous walk ranges, `page_remap` | 1× | 1× | none | minimum |
| contiguous column-major output-tile ranges with a window (`tile_row_shift`) | 1× + ≤ (cores − 1) tiles | 1× | none | **chosen**; the halo is bounded by the core count, not the tensor |
| whole tile columns dealt to cores (no halo) | 1× | 1× | none | cheapest bytes, but under-fills the grid when `outer × tensor_w_tiles < cores` (`[1,4,400]`: 13 columns, `[1,32,50]`: 2). **Trade stated**: the chosen split pays ≤ 31 extra tile reads (≤ 62 KiB bf16) for full occupancy; for tensors with ≥ cores columns the two coincide except at range boundaries. Grid-sync lamp measures it |
| per-output-tile refetch | up to 4× | 1× | none | rejected |
| W-segments of sticks (RM) | 1× | 1× | none | same bytes, more transactions, no extra bank — deferred |
| split the cut axis + combine | n/a | | | no dependent axis: nothing to combine |

Operand-reuse check: the one input does vary along every split axis (each core reads disjoint pages, except the bounded halo) → nothing is reuse-shared → no broadcast row.

Stall-shadow check: the `tile_row_shift` reader waits at each DRAM read barrier. Work independent of that wait: assembling the previous block's output tiles (they need only resident tiles). Phase 0 does not overlap them (the writer hart already overlaps DRAM writes with the reader's stall, and 32 cores keep both DRAM channels busy); the reorder — issue block k+1's reads under a second transaction id, assemble block k, then barrier — needs a window of `2·block_h_tiles + 1` and is recorded as a perf lamp. No floating-point order is involved (pure copy). The `page_remap` reader never waits on its own reads (implicit sync; only ring credits gate it).

### Block schedule

Per core, one program (the regime's kernels). `page_remap`:

```cpp
// reader (1 hart)                                   // writer (1 hart)
for (unit = start_unit; unit < start_unit + num_units; ++unit)
    load_page(unit);           // input page -> cb_output_pages entry
                                                      for (unit = ...) store_page(unit);   // entry -> chunk c page
```

`tile_row_shift` (walk `(o, tw, c, j)` over output tiles, ranges cut into column segments):

```cpp
for (block_idx = 0; block_idx < num_blocks_this_core; ++block_idx) {   // block = <= block_h_tiles outputs of one column
    load_window_block(block_idx);        // missing input tiles of this block -> scratch_input_window, ONE barrier
    assemble_block(block_idx);           // per output tile: reserve 1 entry, <= 8 row-run copies, barrier, push 1
    store_block(block_idx);              // writer: each entry -> chunk c page (implicit sync)
}
```

| Operation | Block shape | Resident across it | Intended frequency of fixed costs |
|---|---|---|---|
| `load_page` | 1 page (stick or tile) of the core's range | nothing | one tagged implicit-sync read per page; no barrier; ISR credit per ring half |
| `store_page` | 1 page | nothing | one tagged implicit-sync write per page; no barrier |
| `load_window_block` | up to `block_h_tiles + 1` input tiles of one column (the first may already be resident) | the window keeps the last tile of the previous block of the same column | DRAM read barrier **once per block** (trid 0); window reset once per column change |
| `assemble_block` | `block_h_tiles` output tiles, each from ≤ 2 resident input tiles | window | per output tile: `reserve_back(1)`, ≤ 8 self-aimed copies, one local barrier (trid 0), `push_back(1)`; whole-tile copy (1 transfer) when the shift is 0 |
| `store_block` | the same tiles | nothing | one implicit-sync write per tile |

Output-tile row-run map (for `assemble_block`, shift `σ = (c*split_size + 32j) % 32`, input tile row `ta = (c*split_size + 32j) // 32`, valid rows `v = min(32, chunk_extent[c] - 32j)`): output row `r ∈ [0, v)` comes from input row `r + σ` of tile `ta` (if `< 32`) else row `r + σ - 32` of tile `ta + 1`. Byte offset of row `r`, column half `h ∈ {0,1}` inside a tile: `((r / 16)*2 + h)*face_bytes + (r % 16)*face_row_bytes`, `face_bytes = page_bytes/4`, `face_row_bytes = page_bytes/64`. Split `[0, v)` at output rows 16 and at input-row boundaries 16 and 32 into maximal runs; each run is two contiguous copies (h = 0, 1) of `run_rows × face_row_bytes` bytes → ≤ 8 copies per tile. Rows `v..31` of the output tile are padding: not written (unspecified by the spec's GUIDANCE "Padding"). Tile `ta + 1` is fetched only when `σ + v > 32` (it then exists: every valid row is `< H`).

### Perf lamps

| Lamp | Why the default may be wrong here | Nearby alternative to measure |
|------|-----------------------------------|-------------------------------|
| Grid synchronization | tiny inputs (traced `[1,4,400]`: 26 output tiles on 26 cores, 1 each, 13 halo reads) are launch-dominated | deal whole tile columns (13 cores, 0 halo); and engaging ≤ 8 cores for < 64 units |
| Overlap (`RING_DEPTH`) | 2 entries per ring may leave the NoC idle between ISR credits | `RING_DEPTH` = 4 and 8 (check on ROW_MAJOR sticks of 16 KiB) |
| `block_h_tiles` | 8 taken from WH/BH catalog measurements, not Quasar | 4, 16, whole column segment |
| Stall shadow (tile_row_shift) | the reader idles during each DRAM barrier | prefetch block k+1 under a second trid while assembling block k (window `2·block_h_tiles + 1`) |
| Copy engine (tile_row_shift) | ≤ 8 small self-aimed NoC copies + a barrier per tile may cost more than CPU stores | RISC-V word copies through the uncached L1 alias (`DataflowBuffer::get_write_ptr`, `dataflow_buffer.h:407`) |
| Thread plan (tile_row_shift) | the reader does fetch + assembly; it may become the per-core bound | (2, 0, 1) with columns dealt to two reader harts (window per hart) |

On ttsim these are gated on structural counts (transfers per barrier, tile reads per output tile, cores engaged), not time (backend page, *Performance measurement*).

## Dataflow Strategy

| Stage | Format | Mechanism | Notes |
|-------|--------|-----------|-------|
| input pages → reader | page as stored (stick / tile, bf16) | `page_remap`: `noc.async_read<NocOptions::TXN_ID>(acc_input, cb_output_pages, {.page_id}, {})` (implicit sync). `tile_row_shift`: tagged explicit read into `scratch_input_window` (`{.trid = 0}`), `async_read_barrier` on trid 0 once per block | DRAM and L1 inputs both through `TensorAccessor(tensor::input)` (spec GUIDANCE "Placement") |
| window → `cb_output_pages` (tile_row_shift) | tile rows, face-row granular | self-aimed `noc.async_read<TXN_ID>(UnicastEndpoint, cb_output_pages, run_bytes, dataflow_kernel_lib::local_addr(src, noc.get_noc_id()), {.offset_bytes = dst_off}, {.trid = 0})`, then barrier, `push_back(1)` | ring opted out of implicit sync on the producer side (`disable_dfb_implicit_sync_for=["cb_output_pages"]` on the reader); consumer side stays implicit (per-side rule, `dfb.cpp:297`) |
| `cb_output_pages` → chunk tensors | page | writer: `noc.async_write<NocOptions::TXN_ID>(cb_output_pages, acc_chunk_c, {}, {.page_id})` (implicit sync) on the accessor of chunk `c` | writer binds `chunk_0 … chunk_{n-1}`; specialized by a define `SPLIT_NUM_CHUNKS` (`KernelSpec.CompilerOptions(defines=…)`) so only bound names are referenced |
| end of kernel | — | implicit-sync reader: `noc_async_read_set_trid(0, noc.get_noc_id())`; writer: `noc_async_write_set_trid(0, noc.get_noc_id())` as the last statement | sticky-tag rule (target page *Ring protocols*) |

Placement axis (TARGET `sharding` = `["interleaved"]` only): no sharded scheme is in TARGET; R6 refuses a sharded input by name. Were one added, a height shard along the cut would be a knob-turn (whole pages per core; consume the shard in place through a ring on the shard), a width shard likewise — split has no dependent axis, so no placement is a scheme-change.

## Work Distribution

| Field | Value |
|-------|-------|
| Work unit | `page_remap`: one input page (= one output page). `tile_row_shift`: one output tile in the column-major walk `(o, tw, c, j)`; `num_units = tensor_outer × tensor_w_tiles × Σ_c ceil(chunk_extent[c]/32)` |
| Grid | `device.compute_with_storage_grid_size()` (8×4 on quasar); `ttnn.split_work_to_cores(grid, num_units, row_wise=True)` (catalog `noc_placement`: row-wise spread); cores with 0 units get no kernel (target_nodes = the core set returned) |
| Per-core work | RT args `start_unit`, `num_units` (one contiguous range); geometry (`tensor_outer`, `tensor_cut_pages`, `tensor_inner_pages`, `split_size` in page-rows or rows, `n_chunks`, `page_bytes`/`face_row_bytes`, `block_h_tiles`) from the single host derivation |
| Remainder | `split_work_to_cores` two groups (`per_core_1`, `per_core_2`); ragged last chunk via `chunk_extent[c] = min(split_size, S - c*split_size)` and `ceil(…/32)` tile rows per chunk image; partial tiles handled by the valid-row count `v` |
| Bound | **data movement** — zero compute per unit (a copy; ~2 KiB in + 2 KiB out per tile, ≤ 8 local copies in tile_row_shift) |
| Thread plan | **(R, C, W) = (1, 0, 1)**: one reader hart, no compute kernel, one writer hart; host constant `THREAD_PLAN = (1, 0, 1)` passed as CT args `reader_threads`, `writer_threads` (kernels loop with stride = their count and `get_my_thread_id()`) |
| Unit dealt to compute threads | none (no compute kernel). DM threads deal the ring entry itself (1 reader, 1 writer; no padding) |
| Tests the plan needs | (a) a test that every output page was written — tag check: run the op on an input with distinct per-page values and compare every chunk page (with 1+1 harts a correct full result shows both ran; the threads' loops must still be sized from the CT counts so a missing thread leaves pages unwritten); (b) one device session: another program (e.g. `ttnn.to_layout` of some tensor) then split; and split then `ttnn.to_layout(chunk, ttnn.ROW_MAJOR_LAYOUT)` on a TILE chunk, for both regimes |

Worked Phase 0 numbers (quasar, 32 cores, bf16):

| Entry | Layout | Regime | `num_units` | Cores × units | Input tile/stick reads (min) |
|---|---|---|---|---|---|
| `(1,4,8400) s2` | RM | page_remap | 4 sticks (16,832 B aligned DRAM page) | 4 × 1 | 4 (4) |
| `(1,4,400) s2` | TILE | tile_row_shift | 13 cols × 2 = 26 | 26 × 1 | 26 (13) |
| `(1,2,64) s1` | TILE | tile_row_shift | 2 × 2 = 4 | 4 × 1 | 4 (2) |
| `(1,32,50) s16` | TILE | tile_row_shift | 2 × 2 = 4 | 4 × 1 | 4 (2) |
| `(2,6,512) s3` | TILE | tile_row_shift | 32 × 2 = 64 | 32 × 2 (whole columns) | 32 (32) |
| `(1,64,416) s32` | TILE | page_remap | 2 × 13 = 26 | 26 × 1 | 26 (26) |
| `(4,128,512) s64` | TILE | page_remap | 256 | 32 × 8 | 256 (256) |
| `(1,34,4096) s17` | TILE | tile_row_shift | 128 × 2 = 256 | 32 × 8 (4 whole columns) | 256 (256) |
| `(1,1024,64) s512` | TILE | page_remap | 64 | 32 × 2 | 64 (64) |
| `(1,1024,64) s512` | RM | page_remap | 1024 sticks | 32 × 32 | 1024 (1024) |

## Circular Buffers

Dataflow buffers (DFB rings) and one scratchpad. Exactly one ring per program.

| Semantic Name | Index | Page Size | Num Pages | Sizing rationale | Format | Producer | Consumer | Sync | Lifetime |
|---------------|-------|-----------|-----------|------------------|--------|----------|----------|------|-----------|
| `cb_output_pages` | DFB `unique_id="cb_output_pages"` | `page_bytes` = `input.buffer_aligned_page_size()` | `RING_DEPTH × 1` (1 = reader threads = writer threads) = 2 | streams over every axis; live set 1 entry each side; depth 2 for overlap | `data_format_metadata = input.dtype` (bf16) | reader | writer | page_remap: implicit / implicit. tile_row_shift: **explicit** producer (opted out, every reader transfer trid 0) / implicit consumer | whole program |
| `scratch_input_window` (tile_row_shift only) | `ScratchpadSpec(unique_id="scratch_input_window")` | `page_bytes` | `WINDOW_TILES = BLOCK_H_TILES + 1` = 9 → `size_per_node = 9 × page_bytes` | spans `cut` (block_h_tiles + 1 input tiles of one column); streams over `outer`, `width` | metadata omitted (DM-only scratchpad) | reader (private) | reader (private) | — (no ring credits; the reader's own trid-0 barrier orders read → copy) | whole program |

## Block Operation Realization

| # | Block operation | Block shape | Helper? | Input CB (semantic name, pages, state) | Output CB (semantic name, pages) | CB state after |
|---|-----------------|-------------|---------|----------------------------------------|----------------------------------|----------------|
| 1 | `load_page` | 1 page | no (raw NoC API; no kernel-lib helper for a DRAM→ring page read) | `input` tensor page | `cb_output_pages`, 1 entry (implicit post) | entry posted |
| 2 | `store_page` | 1 page | no | `cb_output_pages`, 1 entry | chunk `c` tensor page | entry acked |
| 3 | `load_window_block` | ≤ `block_h_tiles + 1` tiles of one column | no | `input` tensor pages | `scratch_input_window` slots `tile_row % WINDOW_TILES` | window holds the block's input tiles |
| 4 | `assemble_block` | `block_h_tiles` output tiles | **yes** — `dataflow_kernel_lib::local_addr` | `scratch_input_window` | `cb_output_pages`, 1 entry per output tile (reserve/push) | entries posted |
| 5 | `store_block` | same tiles | no | `cb_output_pages` | chunk tensor pages | entries acked |

## API Mapping

| Block operation | Type | Function | File:Line | Template Params / Args | Input CB | Output CB | Which params are block knobs |
|-----------------|------|----------|-----------|------------------------|----------|-----------|------------------------------|
| `load_page` | raw_api | `Noc::async_read<NocOptions::TXN_ID>(src, DataflowBuffer&, src_args, dst_args)` (implicit) | `tt_metal/hw/inc/api/dataflow/noc.h:922-927`; pattern `.claude/references/quasar_op_examples/toy_thread_plan/kernels/reader_implicit.cpp` | `acc_input`, `cb_output_pages`, `{.page_id = input page}` | `input` | `cb_output_pages` | none (page = transaction) |
| `store_page`, `store_block` | raw_api | `Noc::async_write<NocOptions::TXN_ID>(DataflowBuffer&, dst, src_args, dst_args)` (implicit) | `noc.h:938-943`; `toy_thread_plan/kernels/writer_implicit.cpp` | `cb_output_pages`, `acc_chunk_c`, `{}`, `{.page_id = chunk page}` | `cb_output_pages` | chunk tensors | none |
| `load_window_block` | raw_api | `Noc::async_read<NocOptions::TXN_ID>(acc_input, dst, size, {.page_id}, dst_args, {.trid = 0})` + `async_read_barrier<NocOptions::TXN_ID>({.trid = 0})` | `noc.h:247-258`, `noc.h:739`; pattern `toy_thread_plan/kernels/reader.cpp` | dst = the window slot address (`Scratchpad<uint8_t>(scratch::scratch_input_window)`, `tt_metal/hw/inc/api/scratchpad.h:54`) | `input` | `scratch_input_window` | `block_h_tiles` (reads per barrier) |
| `assemble_block` | helper + raw_api | `dataflow_kernel_lib::local_addr(addr, noc.get_noc_id())` as the `UnicastEndpoint` source of `Noc::async_read<TXN_ID>(UnicastEndpoint{}, cb_output_pages, run_bytes, src_args, {.offset_bytes = dst_off}, {.trid = 0})`; `DataflowBuffer::reserve_back(1)` / `push_back(1)` | `ttnn/cpp/ttnn/kernel_lib/local_copy_helpers_dataflow.hpp:57` (`.inl:17-30` example); `tt_metal/hw/inc/api/dataflow/endpoints.h:76`; `noc.h:252` | run sizes vary per tile → plain `async_read`, not the stateful `set_read_state` / `read_with_state` (single fixed size, `local_copy_helpers_dataflow.hpp:74-95`) | `scratch_input_window` | `cb_output_pages` | none |
| host | raw_api | `ttnn.experimental.ProgramSpec`, `KernelSpec(num_threads=…, compiler_options=CompilerOptions(defines=…))`, `DataflowBufferSpec`, `ScratchpadSpec`, `ttnn.generic_op(io_tensors, spec, run_args, tensor_args)` | `toy_thread_plan_program_artifacts.py`; `ttnn/cpp/ttnn-nanobind/program_spec.cpp:365` (ScratchpadSpec), `:586-601` (CompilerOptions); `generic_op_nanobind.cpp:70-81` | `DataMovement2XXConfig(disable_dfb_implicit_sync_for=["cb_output_pages"])` on the tile_row_shift reader only | | | `THREAD_PLAN`, `RING_DEPTH`, `BLOCK_H_TILES` |

Helpers considered and rejected (no compute phase exists; the compute-side helpers are listed for completeness):

| Candidate | Why not | Citation |
|---|---|---|
| `tilize` / `untilize` helpers (row assembly via untilize → shift → tilize on the NEOs) | values would pass through unpack/DEST/pack: R1 (no element through another dtype; float32 in TARGET needs fp32 DEST and `UnpackToDest`) and two extra L1 passes per tile; the shift is a row move, which the DM copy does exactly | spec R1 |
| `eltwise` `copy` helper | a compute-side copy maps one input tile to one output tile (`copy(shape, in, out)`): it cannot move rows across tiles, so it would be an extra NEO hop around the DM assembly that is needed anyway | `ttnn/cpp/ttnn/kernel_lib/eltwise/api/convenience.hpp:113` |
| `dataflow_kernel_lib::set_read_state` / `read_with_state` | one fixed transfer size baked into NoC state; runs vary (1–16 face rows) | `local_copy_helpers_dataflow.hpp:66-75` |
| `ttnn.Mcast` / `McastArgs` | no reuse-shared operand | traffic ranking above |

## Key Risks and Gotchas

| Risk | Why it bites here | Mitigation in this design |
|------|-------------------|---------------------------|
| Sticky transaction tag | the implicit reader/writer leave a ring's id in the tag register; the explicit tile_row_shift reader issues self-aimed copies and scratch reads | implicit kernels end with `noc_async_read_set_trid(0, …)` / `noc_async_write_set_trid(0, …)`; the explicit reader tags **every** transfer (scratch reads and local copies) with `{.trid = 0}`; session tests in Work Distribution |
| DM→DM ring with explicit producer / implicit consumer | not shown in the Quasar examples (they pair DM with compute); `transpose_cn` uses explicit on both sides | per-side sync is legal (`dfb.cpp:288-297`); if the mixed ring misbehaves on ttsim, fall back to explicit on the writer side too (opted out, tagged writes, `async_write_barrier` before `pop_front`) — record the change |
| Self-aimed copy addresses | `get_write_ptr()` returns the uncached alias (`dataflow_buffer.h:407`); the scratchpad base is a plain L1 address; `local_addr` must use the issuing NoC's coordinates | pass `noc.get_noc_id()` to `local_addr`; use the DFB as the typed destination with `{.offset_bytes}` (never a raw write pointer); test `assemble_block` alone first on `(1,4,400)` |
| Unbalanced ring at exit | implicit writer must issue exactly one write per entry the reader posted | reader and writer derive the identical walk from the same RT args (shared header `kernels/split_walk.hpp`) |
| Chunk-count-dependent accessor set | `tensor::chunk_k` exists only for bound chunks | define `SPLIT_NUM_CHUNKS`; `#if` around `chunk_2` / `chunk_3`; Phase 0 tests n = 2 only, so add a probe at n = 1 and n = 3 before claiming more |
| Halo correctness at a range starting mid-column | the window is empty at the start of a range | `load_window_block` fetches every tile a block needs that is not resident, including the first tile of the range |
| Padding rows of assembled tiles | rows `v..31` keep stale ring contents | allowed by GUIDANCE "Padding" (only logical elements are compared); do not read an extra input tile to fill them |
| Shift-0 tiles inside `tile_row_shift` (chunk 0 always) | a whole-tile copy is cheaper than 8 runs | one `page_bytes` copy when `σ == 0` |
| ROW_MAJOR page size not a multiple of the alignment (W = 17, 50, 100…) | entry = `buffer_aligned_page_size()`, not `W*2` | host asserts input and every chunk share the aligned page size; ring entry uses it |

### Structural impossibilities

None to add: every TARGET cell is buildable (spec INVALID is empty, and bfloat8_b/bfloat4_b are not in TARGET).
