# Operation Design: g6_pad

## Overview

| Field | Value |
|-------|-------|
| Classification | data_movement (no compute kernel, no arithmetic) |
| Goal | Constant high-end padding of the trailing dims, bit-exact, in exactly one device program per call. An extension call (TILE, only dim -2 grows inside the existing tile rows) writes c into the input's own buffer and returns a metadata-only view. Every other call assembles each output page once in L1 and writes it once. |
| Math | `D_i = d_i + h_i`; `out[j] = x[j]` if `j_i < d_i` for every i, else `c`; `c` = `value` converted to the dtype as torch converts a Python number for a fill (bf16 through fp32 with RNE; integers by truncation) |
| References | Contract `eval/prompts/g6_pad.txt`; golden suite `eval/golden_tests/g6_pad/{feature_spec.py,helpers.py}` (oracle `pytorch_g6_pad`, `storage_class`); `.claude/references/{blocking-model.md,l1-footprint-discipline.md,noc-alignment-in-tt-metal.md,noc-alignment-theory.md,ttnn-cb-memory-fundamentals.md}`; catalog `ttnn/ttnn/operations/examples/master.md` entries `noc_placement`, `width_split`, `double_buffer`, `trid_double_issue`, `tile_reorder`, `split_reader`, `bank_stagger` |

### Notation (used everywhere below)

| Symbol | Definition |
|--------|-----------|
| `r` | input rank, 2..4 |
| `es` | element size in bytes: 2 (bfloat16), 4 (float32, int32, uint32) |
| `page_bytes` | TILE: `1024 * es` (one 32x32 tile) |
| `row_bytes`, `face_bytes` | `16 * es` (one face row), `256 * es` (one 16x16 face); tile = face0 (rows 0-15, cols 0-15), face1 (rows 0-15, cols 16-31), face2 (rows 16-31, cols 0-15), face3 (rows 16-31, cols 16-31), row-major inside a face |
| `(L0, L1)` / `(M0, M1)` | input / output leading dims: rank 4 `(d0, d1)`, rank 3 `(1, d0)`, rank 2 `(1, 1)`; same with `D` for the output |
| `S_in = L0*L1`, `S_out = M0*M1` | input / output slice counts (a slice = one index of the dims before -2) |
| `Ht_in, Wt_in, Ht_out, Wt_out` | `ceil(d[-2]/32)`, `ceil(d[-1]/32)`, `ceil(D[-2]/32)`, `ceil(D[-1]/32)` — always `ceil`, per slice |
| `a`, `b` (R1) | `a = d[-2] - 32*(Ht_in-1)` (logical rows in the last tile row, 1..31), `b = D[-2] - 32*(Ht_in-1)` (end of the new rows, a+1..32) |
| `row_mask` (R2) | `d[-2] - 32*(Ht_in-1)` if `d[-2] % 32 != 0 and D[-2] > d[-2]`, else 32 (no row mask) |
| `col_mask` (R2) | `d[-1] - 32*(Wt_in-1)` if `d[-1] % 32 != 0 and D[-1] > d[-1]`, else 32 (no column mask) |
| `R`, `P`, `F` (R2) | staged pages `S_in*Ht_in*Wt_in`; output pages `S_out*Ht_out*Wt_out`; fill pages `P - R` |
| `U` (R1) | band units `S_in * Wt_in` (every tile of the last tile row of every slice) |
| `c_bits` | the raw bit pattern of c in the dtype (low 16 bits for bfloat16) |

## Parameters

| Name | Type | Required | Valid Range | Default | CT/RT |
|------|------|----------|-------------|---------|-------|
| `input_tensor` | `ttnn.Tensor` | yes | rank 2-4, interleaved (DRAM or L1), TILE or ROW_MAJOR | — | buffer address: common RT arg |
| `padding` | sequence of k pairs `(low, high)` (list/tuple of 2-element lists/tuples) | yes | `1 <= k <= r`; ints (bool excluded); `low == 0`; `high >= 0`; pair m applies to dim `r-k+m` | — | host-derived: output shape, R1 band runs, R2 masks/walk geometry — all RT args |
| `value` | `int` or `float` (positional or keyword) | no | any int/float; integer dtypes: truncated value inside the dtype range | `0.0` | `c_bits`: common RT arg; `FILL_IS_ZERO = (c_bits == 0)`: CT arg |

### Host validation order (entry point: `validate()` is the first line)

| Step | Check | Raises |
|------|-------|--------|
| 1 | `padding` is a list/tuple of 1..r pairs, each a 2-element list/tuple of ints; `low == 0`; `high >= 0` | `ValueError` |
| 2 | `value` is int or float; for int32 / uint32 the truncated value is in range | `ValueError` |
| 3 | input is sharded (`memory_config().is_sharded()`) | `ValueError` — must precede step 4: `SupportRefusal` subclasses `NotImplementedError` (`ttnn/ttnn/operations/_op_contract.py:23`), and the golden sharded refusal expects `ValueError`/`RuntimeError` on a SUPPORTED cell |
| 4 | build the scenario dict (`shape`, `padding`, `value`, `dtype`, `layout`, `"dram"`/`"l1"` from the buffer type), apply `INPUT_TAGGERS`, gate per axis on `SUPPORTED`, then on `EXCLUSIONS` | `UnsupportedAxisValue` / `ExcludedCell` |

`INPUT_TAGGERS` (names and values match `feature_spec.TARGET`): `tag_alignment` (W%32≠0 → `"w_non_aligned"`; else H%32≠0 → `"h_non_aligned"`; else `"tile_aligned"`), `tag_rank` (`len(shape)`), `tag_padded_dims` (highs of dims -2/-1 after aligning the pairs to the trailing dims: both > 0 `"hw"`, only -2 `"h"`, only -1 `"w"`, neither `"leading"`), `tag_dtype`, `tag_layout`, `tag_memory_config`. **Taggers must be total**: the harness tags before `validate()` runs, including on the refusal scenarios (`low = 1`, `high = -4`, 5 pairs on rank 4) — align only the last `min(k, r)` pairs and count a dim as padded only when its high is `> 0`.

Phase 0 support rectangle (the implementer writes `SUPPORTED`): dtype `bfloat16`; layout `TILE`; memory_config `"dram"`; alignment `"w_non_aligned"`, `"h_non_aligned"`; rank 3, 4; padded_dims `"h"`, `"hw"`. `EXCLUSIONS = []`. The design below is general over TARGET; Phase 0 builds R1 and R2 for bfloat16.

Structural impossibilities: none beyond `feature_spec.INVALID` (empty).

## Tensors

### Input

| Property | Requirement |
|----------|-------------|
| Shape | `(d_0, ..., d_{r-1})`, r = 2..4, any sizes; dims -2/-1 need not be multiples of 32; the input's tile padding holds unspecified values and is never copied into a logical output position |
| Dtype | bfloat16 (Phase 0); float32, int32, uint32 (refinement) |
| Layout | TILE (Phase 0); ROW_MAJOR (refinement, regime R3) |
| Memory | interleaved DRAM (Phase 0); interleaved L1 (refinement); sharded → `ValueError` |

### Output

| Property | Value |
|----------|-------|
| Shape | `(d_0 + h_0, ..., d_{r-1} + h_{r-1})` |
| Dtype | input dtype (no cast) |
| Layout | input layout |
| Memory | input memory config. R1: a metadata-only view of the input's buffer — same address, same padded shape. R2: a new tensor allocated with `ttnn.allocate_tensor_on_device(shape, dtype, layout, device, input.memory_config())` |

## Blocking Model

Every axis of this op is independent: an output page depends on at most one input page (the same tile index) and on the scalar c. There is no reduction, no scan and no shared operand. The blocking problem is therefore which pages a core owns and how many it keeps in flight. Data movement per page is fixed: each input page crosses DRAM at most once, each output byte that must change crosses once.

### Regime selection (exact, host-side, grid-independent)

```
highs        = [0]*(r-k) + [h for (_, h) in padding]
is_extension = layout == TILE
               and highs[r-2] > 0
               and all(highs[i] == 0 for i != r-2)
               and ceil((d[-2] + highs[r-2]) / 32) == ceil(d[-2] / 32)
regime       = "extension_band"    if is_extension
               "new_storage_tiles" if layout == TILE
               "new_storage_rows"  if layout == ROW_MAJOR   (deferred, R3)
```

This is the contract's storage-class rule (`helpers.storage_class`). It depends only on shapes and layout, never on the grid or on free L1. `g6_pad_plan(input_tensor, padding, value=0.0)` returns the selected plan (BB-08). It is a pure host function, launches nothing and is exported from `ttnn.operations.g6_pad`. Fields: `regime`, `num_cores`, `core_ranges`, `block_pages`, `staging_depth`, `roles` (per core: role and its page counts, used for the BB-21 role dump).

### Axes

| Axis | Character (+ one-clause reason) | Extent knob | Phase 0 value | Knob source | Core-assignment | Later unlock |
|------|--------------------------------|-------------|---------------|-------------|-----------------|--------------|
| `page` — flattened page index in page order (`lead` × `tile_row` × `tile_col`); R1 enumerates band units (last-row tiles), R2 enumerates staged pages in input page order and fill pages in output page order | independent — each output page is a function of one input page (same tile index) or of c only | R1: `units_this_core` (the whole assignment, no staging); R2: `block_pages` (staged pages per block) | R1: all of the core's `U_k` units. R2: `block_pages = min(r_max, block_pages_cap)`, `r_max = ceil(R / num_cores)`; one block per core on every INPUTS shape | host `G6PadPlan` (`block_pages`, per-core counts) → CB capacity + RT args | contiguous balanced ranges (split_work_to_cores' balanced formula) over `num_cores = min(grid_cores, ceil(max_units / MIN_PAGES_PER_CORE))` cores, row-wise order; R2 splits staged and fill pages separately over the same cores, fill ranges in reverse core order | knob-turn (`MIN_PAGES_PER_CORE`, grid, `block_pages`) |
| `lead` — up to two dims before -2 (`L0`, `L1` → `M0`, `M1`) | independent — no value crosses slices | folded into `page`: a block covers however many slices its page run spans | n/a (folded) | `G6PadPlan` walk start coordinates (RT args) | via `page`; never assigned per slice, because `S_in = 1` for p2–p4 would cap the work at one core (Work geometry rule) | knob-turn (leading padding only adds fill slices in R2) |
| `tile_row` (dim -2 in tiles) | independent — rows of tiles never combine | folded into `page` | n/a (folded) | walk start coordinates | via `page`; a tile-row split strands one-tile-row tensors on one core (catalog `width_split`) | knob-turn |
| `tile_col` (dim -1 in tiles) | independent | folded into `page` | n/a (folded) | walk start coordinates | via `page`; a column split caps W = 12 tensors (`Wt = 1`) at one core | knob-turn |
| `tile_interior` (32×32 elements in face layout) | independent — each element is moved or replaced by c alone | whole tile per operation: R1 writes the band rows `[a, b)` of the tile as ≤3 contiguous byte runs; R2 moves whole tiles and masks the complement of the logical rectangle `[0,row_mask) × [0,col_mask)` in edge tiles | whole tile | band runs / masks: host-derived per call (RT args) | not assigned across cores: a tile is the smallest DRAM transaction worth a core (regime `split_tile_across_cores`, rejected) | — |

**Why flattening is honest.** Interleaved pages are never contiguous with their neighbours (they live on different banks), so every page is its own NoC transaction whatever rectangle holds it (`noc-alignment-theory.md` §7.2). A run of pages in page order costs the same as any rectangle of the same size. It also balances to ±1 page on every shape, including `S_in = 1, Ht = 1` (p2–p4) and `Wt = 1` (W = 12).

**Stages and their data.** The only intermediates are (1) the fill page, a constant page that every participating core builds for itself; no core coordinates or distributes it, and it has no axes; and (2) staged tiles, which have exactly the input page axes and are covered by the `page` row. No stage runs on one core while others wait.

**Every knob is a parameter:** `MIN_PAGES_PER_CORE` (host constant, Phase 0 = 1), the grid (`device.compute_with_storage_grid_size()`, never a literal), `L1_BUDGET_BYTES` (host constant 1347776, BB-05), `block_pages` and `staging_depth` (derived once in `G6PadPlan`, fed to the CB descriptor and the RT args), band runs and masks (derived once in `G6PadPlan`). No kernel inlines a page count, tile count or core count.

### Buffer-depth knobs

| CB | Depth knob | Phase 0 value | What the depth buys |
|----|------------|---------------|---------------------|
| `cb_staged_tiles` | `staging_depth` | `1` if `r_max * page_bytes <= staging_budget` (one block holds the core's whole assignment); else `2` with `block_pages = block_pages_cap` | `2` overlaps block k+1's DRAM reads with block k's DRAM writes; with one block per core there is no block k+1, so a second slot would be dead L1 |
| `cb_fill_page_reader` | none (one constant page) | 1 | nothing: read-only source of every mask copy |
| `cb_fill_page_writer` | none (one constant page) | 1 | nothing: read-only source of every fill/band write |

`staging_budget = L1_BUDGET_BYTES - num_fill_pages * page_bytes`; `block_pages_cap = floor(staging_budget / (staging_depth * page_bytes))`.

### Mechanism caps

| Mechanism | Cap on which extent | Clamp | What happens unclamped |
|-----------|--------------------|-------|------------------------|
| Model L1 budget (BB-05: 1347776 B/core) | `block_pages` × `staging_depth` | `block_pages = min(r_max, block_pages_cap)`; `staging_depth = 2` when the clamp binds | launch refusal / overlap with the model's L1 on shapes whose per-core assignment is large |
| Tile face layout (4 faces of 16×16, row-major inside a face) | R1 band and R2 mask geometry | runs computed per face, adjacent runs merged; a band `[a, b)` gives at most 3 runs (`b <= 16`: 2; `a >= 16`: 1–2; `a < 16 < b`: 3, or 2 when `b == 32`); fixed 3 (offset, length) RT-arg slots | rows outside `[a, b)` overwritten: logical input rows destroyed, or bytes outside the last tile row written (contract violation) |
| NoC endpoint alignment (BH: DRAM read offsets congruent mod 64, DRAM write and L1 mod 16) | run/mask offsets | every partial transfer sources the fill page at the **same in-page offset** it targets (offsets are multiples of `row_bytes` ≥ 32 B); DRAM reads are whole pages at page base | silent data corruption (watcher stops it) |
| `NOC_MAX_BURST_SIZE` (BH: 256 words × 64 B = 16 KB, `noc_parameters.h:289-290`) | bytes per transaction | `page_bytes <= 4 KB` (TILE): one packet per page; stateful one-packet writes are legal | multi-packet transfers through a one-packet stateful path truncate |
| Kernel-config region (BB-04, BB-18) | RT-arg count | shape geometry, `c_bits`, band runs and masks are **common** RT args; per-core RT args are ≤ 8 words (start coordinates and counts) | refused launch in profiler/NoC-trace builds |
| 32-bit RT args | `c_bits`, page ids | `c_bits` fits one word for every dtype; page ids < 2^32 | — |

### Regimes

| Regime | Status | Predicate | Block | Data movement vs. minimum | What a bigger block buys |
|--------|--------|-----------|-------|---------------------------|--------------------------|
| `extension_band` (R1) | **built** | `is_extension` | `units_this_core` last-row tiles; band rows `[a, b)` of each, as ≤3 runs | Minimum. Boundary: DRAM. Minimum = the new logical rows written once; c needs no read. Actual: 0 B read; `(b-a) * 2 * row_bytes` written per unit (whole 32-column rows: the ≤ 31 output-padding columns of the last tile column sit inside the same contiguous runs, so they cost no extra transaction) | No CB, so the block is the whole assignment. Fixed costs: one fill-page build per kernel (NoC zero path when `c_bits == 0`), one final `noc_async_write_barrier` per kernel. Neither depends on `U_k` |
| `new_storage_tiles` (R2) | **built** | `layout == TILE and not is_extension` | `block_pages` staged pages (input page order) per block; fill pages written per tile-row run | Minimum. Boundary: DRAM. Each input page is read exactly once, each output page written exactly once, and no fill page is ever read (fill comes from L1). Edge-tile masking is L1-local (self-aimed NoC copies) and adds no DRAM bytes | Per block: one `noc_async_read_barrier` + one mask barrier + one `cb_reserve_back`/`cb_push_back` (reader); one `cb_wait_front`/`cb_pop_front` + one `noc_async_writes_flushed` (writer). Once per kernel: fill-page build per kernel, final write barrier. At `staging_depth = 1` (every INPUTS shape) each of these runs once per core |
| `new_storage_rows` (R3, ROW_MAJOR) | **deferred** — ROW_MAJOR is outside the Phase 0 rectangle (no bfloat16 TILE DRAM case has it); reachable on R2's structure unchanged: the page walkers and `cb_staged_tiles` are page-generic | `layout == ROW_MAJOR` | `block_pages` sticks; page = output stick (`D[-1]*es`, DRAM-aligned stride). Staged stick: read `d[-1]*es` bytes into the slot, then copy the W tail `[d[-1]*es, D[-1]*es)` from a fill row at the matching offset. Fill sticks come from the fill row | Minimum (each input stick read once, each output stick written once). Sticks wider than `staging_budget / 3` need a W-chunk extent (new mechanism cap) — `[5, 8192]` int32 is 32 KB, well inside | as R2, plus the per-stick transaction count (4 B sticks in `[5, 1]` are transaction-dominated) |
| `extension_as_copy` | **rejected** — superseded by R1 | — | — | Copy into new storage reads and writes each last-row tile (and every tile row when Ht > 1, which the contract forbids). R1 writes `(b-a)/32` of the last row and reads nothing | — |
| `extension_rmw` (read each last-row tile, write c rows in L1, write the whole tile back) | **rejected** — superseded by R1 | — | — | `2 * page_bytes` per unit against R1's `(b-a)/32 * page_bytes`, plus a staging CB | — |
| `extension_assemble` (read the logical rows `[0, a)`, write the assembled whole tile) | **rejected** — superseded by R1 | — | — | `(a/32 + 1) * page_bytes` per unit, always more than R1's `(b-a)/32 * page_bytes`, and it adds a DRAM read per unit | — |
| `fill_then_copy` (fill the whole output, then copy the input over it) | **rejected** — contract forbids it (second full-tensor DRAM pass); dead end, superseded by R2 | — | — | `P + R` page writes instead of `P` | — |
| `split_tile_across_cores` | **rejected** — superseded by the page split | — | — | same bytes; adds a core's launch for ≤ 4 KB of work | — |
| `compute_masking` (unpack → select against c → pack) | **rejected** — superseded by R2's dataflow masking | — | — | adds an L1 unpack/pack round trip per staged tile and passes values through a format path the contract forbids for int32/fp32 | — |
| `fill_page_multicast` (one core builds the fill page and multicasts it) | **rejected** — superseded by per-core local build | — | — | adds `num_cores * page_bytes` of NoC traffic plus a semaphore round trip to save a ≤ 0.5 µs local build of a page that comes from one scalar | — |

### Traffic ranking

| Candidate split | DRAM crossings | Cross-core traffic | Occupancy on INPUTS | Verdict |
|-----------------|---------------|--------------------|---------------------|---------|
| Flattened page split (R1: band units; R2: staged + fill pages, two balanced sub-splits on the same cores) | minimum: input pages ≤ 1×, output bytes 1× | none | fills the grid whenever work units ≥ grid (p2–p4: 1940 units; p1: 960 staged / 576 fill) | **chosen** |
| Split by slices (`lead`) | same | none | `S_in = 1` for p2–p4 and every `[1, T, C]` case → one core | ties on bytes, loses on occupancy (forbidden by the Work geometry rule) |
| Split by tile rows | same | none | R1 touches one tile row per slice → capped at `S_in` | ties on bytes, loses on occupancy |
| Split by tile columns (catalog `width_split`) | same | none | `Wt = 1` for every C = 12 case → one core | ties on bytes, loses on occupancy |
| Dependent-axis split + combine | n/a — the op has no dependent axis | — | — | not applicable |
| Broadcast of a reuse-shared operand | n/a — the only operand shared by every page is c, a 32-bit RT arg; no tensor is re-read along any split (the operand-reuse check finds no input invariant along the split axis: input page (n, ht, wt) feeds exactly one output page) | — | — | not applicable |

All splits move the same DRAM bytes: there is no shared operand and no overlapping footprint. The extent is invisible to the split ranking (`blocking-model.md` §4). Occupancy and balance break the tie: only the flattened page split fills the grid on every INPUTS geometry. Within R2, staged pages (read + write) and fill pages (write only) are split separately over the same cores, so each core moves `≈ (2R + F)/num_cores` page-crossings. A single split over output pages would leave cores holding copy ranges with twice the bytes of cores holding fill ranges.

### Block schedule

R1 `extension_band` — one kernel (writer, BRISC / NoC1):

```cpp
build_fill_page();                       // once per kernel; c_bits == 0 → NoC zero path, else word stores
write_band_block(units_this_core);       // ≤3 runs per unit, fill_page+off → page_addr+off; one final write barrier
```

R2 `new_storage_tiles` — reader (NCRISC / NoC0) and writer (BRISC / NoC1):

```cpp
// writer
build_fill_page();                                   // once per kernel
write_fill_pages(fill_range_this_core);              // BEFORE the first staged wait: stall-shadow fill
for (uint32_t block_idx = 0; block_idx < num_blocks_this_core; ++block_idx) {
    store_block(block_idx);                          // wait block_pages, write block_extent pages, flush, pop
}
// final noc_async_write_barrier

// reader
for (uint32_t block_idx = 0; block_idx < num_blocks_this_core; ++block_idx) {
    load_block(block_idx);                           // reserve block_pages, issue block_extent page reads
    if (block_idx == 0) build_fill_page();           // in the shadow of block 0's reads (needed only by the mask)
    // read barrier
    mask_edge_tiles(block_idx);                      // edge pages: complement of the logical rectangle ← c
    // mask barrier, push block_pages
}
```

| Block operation | Regime / kernel | Block shape | Resident across it | Intended frequency of fixed costs |
|-----------------|-----------------|-------------|--------------------|-----------------------------------|
| `build_fill_page` | R1 writer; R2 writer; R2 reader | one page of `c_bits` (R1 needs only the span of the band runs to be valid) | the page stays valid for the whole kernel | once per kernel per RISC; `FILL_IS_ZERO` → `async_write_zeros` + one `write_zeros_l1_barrier`, else one `fill_l1_range` pass |
| `write_band_block` | R1 writer | `units_this_core` units × ≤3 runs | fill page | one write per run; no CB handshake; **one** `noc_async_write_barrier` at the end of the kernel (BB-03) |
| `write_fill_pages` | R2 writer | `fill_count_this_core` pages, walked as per-tile-row column runs | fill page | one whole-page write per page (same source and size: stateful one-packet writes allowed); no barrier until the final one |
| `load_block` | R2 reader | `block_extent ≤ block_pages` staged pages, input page order | — | one `cb_reserve_back(block_pages)` + one `noc_async_read_barrier` per block |
| `mask_edge_tiles` | R2 reader | the edge pages of the block (last input tile row and/or last input tile column) | fill page (reader's copy) | runs issued back to back, then one barrier per block that contains an edge page, then one `cb_push_back(block_pages)` |
| `store_block` | R2 writer | `block_extent` pages | — | one `cb_wait_front(block_pages)`, one whole-page write per page, one `noc_async_writes_flushed`, one `cb_pop_front(block_pages)` per block |

A ragged last block keeps the nominal `block_pages` in every reserve/push/wait/pop and narrows only the NoC transfers to `block_extent`. That closes the CB ring cycle (`ttnn-cb-memory-fundamentals.md`, Ring-Wrap Invariant).

**Stall-shadow check.**

| Waiting stage | What it waits for | Work moved into the wait | Why it is legal |
|---------------|-------------------|--------------------------|-----------------|
| R2 writer, first `cb_wait_front` | the reader's DRAM reads of block 0 | all of the core's fill-page writes | fill pages and staged pages are disjoint output pages, so write order is free; no floating-point order is involved |
| R2 reader, block-0 read barrier | DRAM read latency | building the reader's fill page | the page is first used by `mask_edge_tiles`, which runs after the barrier |
| R2 reader, mask barrier | local L1→L1 copies | nothing independent remains in the block (the push needs every mask) | — |
| R1 writer | nothing until its own final barrier | — | R1 has no wait before its writes |

### Perf lamps

| Lamp | Why the default may be wrong here | Nearby alternative to measure |
|------|-----------------------------------|-------------------------------|
| Overlap (R2) | One block per core puts all of a core's staged reads before all of its staged writes; only the fill writes overlap the reads | `block_pages = ceil(r_max / 2)`, `staging_depth = 2` (same L1), on p1 |
| Grid synchronization / per-core fixed cost (R1, R2) | `MIN_PAGES_PER_CORE = 1` gives tiny shapes one page per core (case 65: 49 cores with 1–2 pages each). Per-core launch and RT-arg dispatch can outweigh the work | `MIN_PAGES_PER_CORE = 2, 4` on cases 65 and 254 |
| Issue rate (R1) | One BRISC issues every band write. p3 has 2 × 256 B runs per unit, so transaction count is high relative to bytes (catalog `split_reader`: only helps when one RISC is issue-bound) | split each core's units between BRISC/NoC1 and NCRISC/NoC0, on p3 |
| Bank alignment (R1, R2) | Each core's range starts on bank `start_page mod num_banks`. When per-core counts are multiples of the bank count, every core starts on the same bank (catalog `bank_stagger`) | rotate each core's walk start by `core_idx mod num_banks`, on p1 and p2 |
| Mask mechanism (R2) | Mask runs are self-aimed NoC copies from `cb_fill_page_reader`. On shapes with very few edge pages, RISC word stores could cost less and would drop that CB | `fill_l1_range` stores for all mask runs, on case 65 (13 edge pages of 64) |

Catalog evidence used: `noc_placement` → row-wise core order and reads on NoC0 / writes on NoC1; `width_split` → flatten so one-tile-row tensors fill the grid; `double_buffer` + `trid_double_issue` → batch a block of reads behind one barrier and skip trids at a large CB; `tile_reorder` → whole-page writes for staged and fill pages, never per-face writes; `split_reader` → issue-rate lamp; `bank_stagger` → bank lamp.

## Dataflow Strategy

| Stage | Format | Mechanism | Notes |
|-------|--------|-----------|-------|
| Host: `value` → `c_bits` | dtype bit pattern | `torch.full((1,), value, dtype=<torch dtype>)` viewed as int16/int32 | This is the oracle's own conversion: -1.7014118346046923e38 → bf16 `0xFF00` (finite), 0.1 → RNE. `FILL_IS_ZERO` is keyed on `c_bits == 0`; `-0.0` (bf16 `0x8000`) is **not** zero |
| Host R1: output view | metadata only | `ttnn.reshape(input_tensor, ttnn.Shape(out_shape), input_tensor.padded_shape)` | Volume change → codegen refuses (`reshape_codegen_supported.cpp:402-403`) → `tile_tensor_view_reshape_possible` → `ttnn::experimental::view(tensor, logical, padded)` (`reshape.cpp:778-787`) → `view_device` builds a new MeshBuffer at the same address (`tensor_ops.cpp`). No program. The implementer verifies with graph capture: exactly one program per call (the op's) and the same `buffer_address()` |
| R1: fill page → DRAM band | raw bytes | `noc_async_write(fill_page + off_j, accessor.get_noc_addr(page_id, off_j), len_j)` for j < `num_runs`, NoC1 | `page_id = (n * Ht_in + Ht_in - 1) * Wt_in + wt`; never touches another tile row (contract); 0 reads |
| R2: input page → `cb_staged_tiles` | raw tile bytes | `noc_async_read(accessor.get_noc_addr(p_in), slot, page_bytes)`, NoC0, one barrier per block | whole page at page base: DRAM read alignment satisfied by construction |
| R2: edge mask | raw bytes | self-aimed NoC copies from `cb_fill_page_reader` at the same in-page offset (`local_copy_helpers_dataflow.hpp`) | Mask region of an edge tile, per face `f = (fr, fc)`: `a_f = clamp(row_mask - 16 fr, 0, 16)`, `w_f = clamp(col_mask - 16 fc, 0, 16)`. If `a_f == 0` or `w_f == 0`, the whole face is masked (one run). Otherwise rows `[a_f, 16)` form one run and, when `w_f < 16`, each row `< a_f` has a tail `[w_f*es, row_bytes)`. A tile is an edge tile iff (`ht == Ht_in-1` and `row_mask < 32`) or (`wt == Wt_in-1` and `col_mask < 32`) |
| R2: `cb_staged_tiles` → output page | raw tile bytes | `noc_async_write(slot, out_accessor.get_noc_addr(p_out), page_bytes)`, NoC1 | `p_out = ((i0*M1 + i1) * Ht_out + ht) * Wt_out + wt` for input slice `(i0, i1)` |
| R2: fill page → output page | raw tile bytes | whole-page write from `cb_fill_page_writer`, NoC1 | Fill enumeration in output page order: in an image slice (`i0 < L0`, `i1 < L1`), rows `ht < Ht_in` contribute cols `[Wt_in, Wt_out)` and rows `ht >= Ht_in` contribute `[0, Wt_out)`; a fill slice contributes every page |

No inter-Tensix communication in any regime, built or deferred. No compute kernel: no value passes through unpack/DEST/pack, so every dtype is bit-exact by construction.

### Host sequence (one device program per call, BB-01/BB-07)

| Step | R1 | R2 |
|------|----|----|
| 1 | `validate()` | `validate()` |
| 2 | `plan = g6_pad_plan(...)` (cached) | same |
| 3 | `out = view(input)` (no program) | `out = allocate_tensor_on_device(...)` |
| 4 | descriptor from the Python cache, keyed on `(shape, padded_shape, highs, c_bits, dtype, layout, buffer_type, device id, grid)`; on a hit, patch only the buffer-address common RT args | same |
| 5 | `ttnn.generic_op([input, out], descriptor)` | `ttnn.generic_op([input, out], descriptor)` |

Everything shape-, padding- or value-dependent lives in RT args. CT args are limited to `es`, `FILL_IS_ZERO` and the `TensorAccessorArgs`. The device program cache (hash of sources, CT args, core ranges, CB sizes, RT-arg counts: `generic_op_device_operation.cpp:78-147`) therefore reuses binaries across shapes, and the RT args of each call's descriptor carry the values. The golden follow-up case (value 0, then 7.0, then another shape) checks exactly this.

### TARGET axes → regimes

| TARGET axis value | Regime | Cost class | What changes |
|-------------------|--------|------------|--------------|
| dtype float32 / int32 / uint32 | R1, R2 | knob-turn | `es = 4` (CT), `page_bytes = 4096`, `c_bits` 32-bit, integer range check on host; band/mask offsets scale with `es` |
| layout ROW_MAJOR | R3 | regime addition on R2's structure | page = stick; W-tail assembly; deferred row above |
| memory_config `"l1"` | R1, R2 | knob-turn | the `TensorAccessorArgs` encode the buffer type; L1 offsets are already congruent mod 16 |
| alignment `"tile_aligned"` | R2 (an aligned H that grows always adds tile rows) | knob-turn | no edge pages (`row_mask = col_mask = 32`) |
| rank 2 | R1, R2 | knob-turn | `L0 = L1 = 1` |
| padded_dims `"leading"`, `"w"` | R2 | knob-turn | fill slices / fill columns in the fill enumeration; `"leading"` with all highs 0 is an all-copy R2 call |
| `*_SHARDED` | — | not in TARGET | the contract refuses sharded inputs with `ValueError` |

## Work Distribution

| Field | R1 `extension_band` | R2 `new_storage_tiles` |
|-------|---------------------|------------------------|
| Work unit | one last-row tile (band unit) | one page (staged or fill) |
| Grid | `device.compute_with_storage_grid_size()`; cores in row-wise order; used cores = `ttnn.num_cores_to_corerangeset(num_cores, grid, row_wise=True)`: ≤ 2 rectangles (BB-17), one kernel descriptor per role over that set | same |
| `num_cores` | `min(grid_cores, ceil(U / MIN_PAGES_PER_CORE))` | `min(grid_cores, ceil(max(R, F, 1) / MIN_PAGES_PER_CORE))` |
| Per-core work | core k: `U_k = U // N + (k < U % N)` units from `k*(U//N) + min(k, U%N)` | core k: staged `r_k = R // N + (k < R % N)` from `k*(R//N) + min(k, R%N)`; fill index `j = N-1-k`: `f_k = F // N + (j < F % N)` from `j*(F//N) + min(j, F%N)`; reversed so that the extra staged and extra fill pages land on different cores |
| Per-core RT args | start `(n, wt)`, `U_k` | reader: start input page / coordinates, `r_k`, `num_blocks_this_core`; writer: the same plus fill start coordinates and `f_k` |
| Remainder | balanced ±1 unit; `ceil` geometry per slice (`Ht = ceil(H/32)` per image, never `floor(batch*H/32)`) | same; ragged last block keeps nominal CB counts |
| Single-unit shapes | `U = 1` (e.g. `[1, 3844, 12]`): one core. The work is one tile's band, and splitting a tile is the rejected `split_tile_across_cores` regime | `max(R, F) = 1`: one core |
| Phase 0 INPUTS examples (grid-independent) | p2/p3/p4: `U = 1940`; `[1, 3844, 1536]`: 48; `[16, 1, 1536]`: 768; `[1, 1000, 12]`: 1 | case 64: R=28, F=4; case 65: R=49, F=15; case 69: R=36, F=28; p1 (refinement): R=960, F=576 |

Regime-pinned tests: the acceptance test asserts `g6_pad_plan(...).regime` for every case. The predicate is grid-independent, so the same input lands in the same regime on every device.

## Circular Buffers

| Semantic Name | Index | Page Size | Num Pages | Sizing rationale | Format | Producer | Consumer | Lifetime |
|---------------|-------|-----------|-----------|------------------|--------|----------|----------|-----------|
| `cb_staged_tiles` | 0 | `page_bytes` | `staging_depth * block_pages` | spans `page` (`block_pages`); streams over blocks; spans `tile_interior` (whole tile) | input dtype | reader | writer | R2, whole kernel |
| `cb_fill_page_reader` | 1 | `page_bytes` | 1 | constant (c); source of mask runs at any in-tile offset | input dtype | reader | reader | R2, built once (in block 0's read shadow), read until the last mask |
| `cb_fill_page_writer` | 2 | `page_bytes` | 1 | constant (c); source of band runs (R1) and whole fill pages (R2) | input dtype | writer | writer | R1 and R2, built once, read until the last write |

R1 allocates only `cb_fill_page_writer`. The two fill pages are separate because one CB read by two kernels breaks the ownership invariant. The duplicate costs one page per core. Full accounting is in `l1_ledger.md`.

## Block Operation Realization

| # | Block operation | Block shape | Helper? | Input CB (semantic name, pages, state) | Output CB (semantic name, pages) | CB state after |
|---|-----------------|-------------|---------|----------------------------------------|----------------------------------|----------------|
| 1 | `build_fill_page` (writer) | one page | yes (`fill_l1_range` / `Noc::async_write_zeros`) | — | `cb_fill_page_writer`, 1 (reserved once, never pushed: scratch owned by the writer) | valid `c_bits` page for the rest of the kernel |
| 2 | `write_band_block` (R1) | `U_k` units × ≤3 runs | raw (see API Mapping) | `cb_fill_page_writer`, 1, read-only | DRAM (input buffer = output view) | unchanged |
| 3 | `build_fill_page` (reader, R2) | one page | yes | — | `cb_fill_page_reader`, 1 (scratch owned by the reader) | valid page |
| 4 | `write_fill_pages` (R2 writer) | `f_k` pages | raw | `cb_fill_page_writer`, 1, read-only | DRAM output | unchanged |
| 5 | `load_block` (R2 reader) | `block_extent` pages | raw | DRAM input | `cb_staged_tiles`, reserve `block_pages` | slots filled, not yet pushed |
| 6 | `mask_edge_tiles` (R2 reader) | edge pages of the block | yes (`local_copy_helpers_dataflow`) | `cb_fill_page_reader`, 1, read-only | `cb_staged_tiles` (in place, before push) | pushed `block_pages` after the mask barrier |
| 7 | `store_block` (R2 writer) | `block_extent` pages | raw | `cb_staged_tiles`, wait `block_pages` | DRAM output | popped `block_pages` after `noc_async_writes_flushed` |

## API Mapping

| Block operation | Type | Function | File:Line | Template Params / Args | Input CB | Output CB | Which params are block knobs |
|-----------------|------|----------|-----------|------------------------|----------|-----------|------------------------------|
| `build_fill_page` (`c_bits != 0`) | helper | `dataflow_kernel_lib::fill_l1_range<val_size>(start_addr, n_bytes, val)` | `ttnn/cpp/ttnn/kernel_lib/l1_helpers.hpp:101` | `val_size = es`; `val = c_bits`; `n_bytes = page_bytes` (R1 may limit it to the run span) | — | fill page | none |
| `build_fill_page` (`c_bits == 0`) | helper | `Noc::async_write_zeros(dst, size_bytes)` + `Noc::write_zeros_l1_barrier()` (wrapped for a DFB by `dataflow_kernel_lib::zero_tile`) | `tt_metal/hw/inc/api/dataflow/noc.h:849`, `:904`; `l1_helpers.hpp:53` | `size_bytes = page_bytes`; no other NoC write may sit between the two calls (`noc.h:418`, the zero-mode assert) | — | fill page | none |
| `write_band_block` | raw_api | `noc_async_write(src_l1, dst_noc, size)` + `TensorAccessor::get_noc_addr(page_id, offset)`; `noc_async_write_barrier()` once at kernel end | `tt_metal/hw/inc/api/dataflow/dataflow_api.h:876`; `tt_metal/hw/inc/api/tensor/tensor_accessor.h:165`; `dataflow_api.h:1951` | `src = fill_page + off_j`, `dst = get_noc_addr(page_id, off_j)`, `size = len_j` | `cb_fill_page_writer` | DRAM | `U_k` (RT) |
| `write_fill_pages` | raw_api | `noc_async_write_one_packet_set_state` / `_with_state` (same source and size every page) or `noc_async_write` | `dataflow_api.h:982`, `:1008`, `:876` | `size = page_bytes ≤ NOC_MAX_BURST_SIZE` | `cb_fill_page_writer` | DRAM | `f_k` (RT) |
| `load_block` | raw_api | `noc_async_read(src_noc, dst_l1, size)` + `get_noc_addr(page_id)`; `noc_async_read_barrier()` | `dataflow_api.h:547`; `tensor_accessor.h:165`; `dataflow_api.h:1921` | `size = page_bytes`, `block_extent` reads per barrier | DRAM | `cb_staged_tiles` | `block_pages` (RT) |
| `mask_edge_tiles` | helper | `dataflow_kernel_lib::local_addr`, `set_read_state<transfer_size>`, `read_with_state`, or `Noc::async_read` with a self-aimed source; `noc_async_read_barrier()` | `ttnn/cpp/ttnn/kernel_lib/local_copy_helpers_dataflow.hpp:57`, `:75`, `:95`; `dataflow_api.h:1921` | source `fill_page + off`, destination `slot + off` (same offset); `local_addr(addr, noc.get_noc_id())` when the Noc is explicit | `cb_fill_page_reader` | `cb_staged_tiles` (in place) | none |
| `store_block` | raw_api | `noc_async_write` + `get_noc_addr(p_out)`; `noc_async_writes_flushed()` per block; `noc_async_write_barrier()` at kernel end | `dataflow_api.h:876`, `:1980`, `:1951`; `tensor_accessor.h:165` | `size = page_bytes` | `cb_staged_tiles` | DRAM | `block_pages` (RT) |

**Helpers considered and rejected.**

| Raw entry | Candidate helper | File:Line | Concrete mismatch |
|-----------|------------------|-----------|-------------------|
| `write_band_block` | `Noc::async_write_zeros(accessor, size_bytes, {page_id, offset_bytes}, scratch)` | `noc.h:893` (contract `:870-880`) | Writes zeros only: its scratch must be pre-zeroed (`noc.h:880`), so it cannot express `c != 0` (golden follow-up value 7.0). A second, zero-only walk reads the same L1 scratch and moves the same bytes, so it saves nothing |
| `write_band_block`, `mask_edge_tiles` | `dataflow_kernel_lib::fill_tile_column<dfb_id, T>` | `ttnn/cpp/ttnn/kernel_lib/tile_column_dataflow.hpp:69-70` | Sets one **column** of an L1 tile with RISC stores. R1 writes **rows** of a DRAM page; R2's mask is a row band plus row tails, which `fill_tile_column` would cover only at up to 32 calls × 32 stores per tile |
| `load_block`, `store_block`, `write_fill_pages` | none — kernel_lib's dataflow headers (`l1_helpers.hpp`, `local_copy_helpers_dataflow.hpp`, `tile_column_dataflow.hpp`, `index_tile_dataflow.hpp`, `tilize_helpers_dataflow.hpp`, `reduce_helpers_dataflow.hpp`, `dfb_helpers_dataflow.hpp`) have no interleaved-page batch reader or writer | `ttnn/cpp/ttnn/kernel_lib/` listing | There is no helper to call. The page transfers are the TensorAccessor + `noc_async_*` primitives themselves |
| (no compute kernel) | `compute_kernel_lib::copy<Input, Output>` (CopyTile → PackTile) | `ttnn/cpp/ttnn/kernel_lib/eltwise/api/convenience.hpp:113` | Routes every value through unpack → DEST → pack. The contract forbids "an unpack and pack in another format" for int32/fp32, and the op has no arithmetic to amortize the L1 round trip. Without a compute kernel, `compute_kernel_hw_startup` does not apply |

## Key Risks and Gotchas

| Risk | Why it bites here | Mitigation in this design |
|------|-------------------|---------------------------|
| Input tile padding copied into a logical output position | Test inputs are zero-padded and c is 0 in most cases, so a whole-tile copy passes the suite and leaks the producer's leftovers in the model (rule 2) | R2: `row_mask`/`col_mask` derived from "non-aligned **and** that dim grows", so every edge tile is masked. R1: writes all band rows `[a, b)`. Golden `poison_padding` cases (253, 247, 68, 63, 65) gate it |
| A copy tile wrongly classified when a dim is non-aligned but not padded | positions past `d` in that dim are output tile padding (don't care), not logical | edge test uses the row mask only on `ht == Ht_in-1` and the column mask only on `wt == Wt_in-1`, each only when that dim grows |
| R1 writing outside the last tile row | contract violation; the golden `untouched_rows` case compares every earlier tile row byte for byte | band runs lie inside `[0, page_bytes)` of last-row pages only; `page_id` is built from `Ht_in - 1` |
| Band runs wrong for `a < 16 < b < 32` (3 runs) | no production case reaches it, so no golden cell does | the host run builder merges per-face runs. Check it host-side (no device) against a row→face byte map for `a < 16 < b < 32`, `b <= 16`, `a >= 16`, `b == 32` |
| The view dispatches a program | BB-01/BB-02: a reshape program fails the golden one-program check | the view path is cited above. Assert one program and an equal `buffer_address()` |
| Cache key missing padding or value | BB-07 follow-up case (value 0 → 7.0 → other shape) replays a wrong program | all such data in RT args. The Python descriptor cache key includes `highs` and `c_bits`; on a hit only buffer addresses are patched |
| `FILL_IS_ZERO` keyed on `value == 0` | `-0.0 == 0` in Python, but its bits are `0x8000` | key on `c_bits == 0` |
| Validation order | `SupportRefusal` is a `NotImplementedError`; the sharded and argument refusals must be `ValueError` | `ValueError` checks run before the support gate |
| Taggers crash on malformed padding | the harness tags before `validate()` on the refusal scenarios | taggers are total |
| Fill page not visible to the NoC before its first read | RISC stores in `fill_l1_range` must land before the NoC reads the page | after the store loop and before the first NoC transfer that sources the page, the kernel must make the stores visible (Blackhole L1 stores from the DM RISCs are uncached; zero path: `write_zeros_l1_barrier`). Verify under `--dev` |
| Transactions in flight at kernel exit | BB-03: the next program hangs | R1 and R2 writers end with `noc_async_write_barrier`; the reader's last barrier precedes its last push |
| Ragged per-core block | the CB ring advances by the pushed counts | nominal `block_pages` on every reserve/push/wait/pop; only NoC sizes shrink |
| Output aliases input (R1) | `generic_op([input, out], ...)` with two tensors on one buffer | the kernel writes through `out`'s address (equal to the input's). The caller must keep the input alive (contract) |
| Profiling (BB-21) | time outside a zone reads as waiting | `DeviceZoneScopedN` per block operation and per blocking wait (`G6PAD-FILL-PAGE`, `G6PAD-BAND`, `G6PAD-READ-BLOCK`, `G6PAD-MASK`, `G6PAD-FILL-WRITES`, `G6PAD-WAIT-STAGED`, `G6PAD-STORE-BLOCK`, `G6PAD-FINAL-BARRIER`); zones compile out of the plain build (`tt_metal/hw/inc/api/debug/kernel_profiler.hpp:1047`); `g6_pad_plan(...).roles` is the role dump |
