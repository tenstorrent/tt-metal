# Operation Design: reshape

The specification is `eval/specs/reshape/spec.py`; where this design and it disagree, the specification governs.

## Overview

| Field | Value |
|-------|-------|
| Classification | data_movement (no compute kernel: every output element is an input element, unchanged; DEFINITION, rule R1) |
| Goal | `torch.reshape` on a device tensor: the output has the new shape, the input's dtype and layout, and lives where the input does (R7). One `ttnn.generic_op` dispatch per call, always a copy into a new buffer. |
| Math | `flat(output)[k] = flat(input)[k]` for every k, where `flat` counts with the last dimension fastest (logical elements only; tile padding is not an element) |
| References | `eval/specs/reshape/spec.py`; `.claude/references/targets/quasar.md`; `.claude/references/backends/ttsim.md`; `.claude/references/quasar_op_examples/` (README, `toy_thread_plan/`); `ttnn/ttnn/operations/toy_spec_single_tensor/` (a ProgramSpec with two DM kernels, a DM→DM ring and no compute kernel); `tt_metal/impl/dataflow_buffer/DataflowBuffer.md` Part A; perf catalog `ttnn/ttnn/operations/examples/master.md` (entries `double_buffer`, `tile_reorder`, `read_coalescing`, `width_split`, `split_reader`) |

## Target

| Fact | Value | Source |
|------|-------|--------|
| Target | `quasar` on the ttsim simulator | `.claude/references/targets/quasar.md` + `.claude/references/backends/ttsim.md` |
| Arch | quasar | target page |
| Compute grid | 8 × 4 = 32 clusters (probe: `compute_with_storage_grid_size()` = 8-4) | target page; probe on this backend |
| L1 per core | 4,194,304 B | target page |
| Unreserved budget | `ttnn.get_max_worker_l1_unreserved_size() - L1_RESERVE_BYTES` = 3,963,328 − 1,048,576 = 2,914,752 B on this target (the query returned 3,963,328 in a probe) | probe on this backend |
| Host program | `ttnn.ProgramSpec` (`ttnn.experimental.ProgramSpec` / `KernelSpec` / `DataflowBufferSpec` / `WorkUnitSpec`, dispatched by `ttnn.generic_op`) — `.claude/references/quasar_op_examples/` | target page |
| Threads per core | 4 NEO compute engines (C ∈ {1,2,4}); 6 user DM harts (R + W ≤ 6); STRIDED rings need R \| C and W \| C. This op uses no NEO (no compute kernel) and 2 (regime A) or 3 (regime B) DM harts | target page *Thread plan* |
| Alignment | L1 16 B, DRAM 64 B (probe: `get_l1_alignment()` = 16, `get_dram_alignment()` = 64) | target page; probe |
| DFB limits | 32 DFBs per cluster; a ring ≤ 1,048,560 B; one producer kernel and one consumer kernel per ring; DM kernel binary ≤ 48 KiB | target page *Cluster model* |
| Unavailable here | `ttnn.ProgramDescriptor` and the legacy C++ device op (Metal 1 kernels refused at creation, target page). `bfloat8_b` / `bfloat4_b` (no path on this part; not in TARGET anyway). A program-less view of the input is excluded by the run's one-dispatch rule, not by the target (regime table) | target page |

Every worked number below is for `quasar` (32 clusters, the budget above). Every run-time predicate uses the query, never the value.

## Parameters

| Name | Type | Required | Valid Range | Default | CT/RT |
|------|------|----------|-------------|---------|-------|
| `input_tensor` | `ttnn.Tensor` | yes | on device, interleaved, DRAM or L1 (R5, R8) | — | — |
| `shape` | `list` or `tuple` of `int` (R3) | yes | at most one `-1` (X2); product equals the input volume after resolving `-1` (X1, X3) | — | host only |

The entry point's signature equals the specification's stub: `reshape(input_tensor, shape) -> ttnn.Tensor`. No compute config (DEFINITION). There is no index axis to canonicalize.

### Host-side order of checks (entry point)

| Step | Check | Raises |
|------|-------|--------|
| 1 | `shape` is a list or tuple of ints; convert to a tuple (R3) | `ValueError` |
| 2 | more than one `-1` | `ValueError` (X2) |
| 3 | resolve `-1` (R2): `rest = prod(d for d != -1)`; `rest == 0` or `volume % rest != 0` → no size fits | `ValueError` (X3) |
| 4 | `prod(resolved) != volume` | `ValueError` (X1) |
| 5 | axes: `dtype`, `layout`, `eval.memory.memory_axes(input_tensor)` → `memory_config`, `sharding` (R8); then every function of the spec's `AXES` (`INPUT_TAGGERS = AXES`) on `(tuple(input_tensor.shape), tuple(shape))` | — |
| 6 | SUPPORTED / EXCLUSIONS gate (R6) | `NotImplementedError` (`UnsupportedAxisValue` / `ExcludedCell`) |
| 7 | mechanism caps (table below): tile shape 32×32; row-major page ≤ `NOC_MAX_BURST_SIZE` | `NotImplementedError` |
| 8 | regime selection (Regimes), output allocation (R7), program build, one `ttnn.generic_op` | — |

## Tensors

### Input

| Property | Requirement |
|----------|-------------|
| Shape | rank 1–4 in TARGET; PHASE0: ranks 1, 3, 4. `H = shape[-2]` if rank > 1 else 1; `W = shape[-1]` |
| Dtype | PHASE0: bfloat16. TARGET: bfloat16, float32, int32, uint32 |
| Layout | ROW_MAJOR (page = one row of W elements) or TILE (page = one 32×32 tile, four 16×16 faces) |
| Memory | interleaved, DRAM or L1 |

### Output

| Property | Value |
|----------|-------|
| Shape | the resolved new shape |
| Dtype | the input's |
| Layout | the input's; in TILE padded for the new shape (R4); the padding rows of an output tile are not written (their content is unspecified: DECISIONS, `pad_value=None`) |
| Memory | a new interleaved buffer of the input's buffer type: `ttnn.allocate_tensor_on_device(ttnn.TensorSpec(new_shape, input.dtype, input.layout, <input.memory_config()>), device)` (R7, GUIDANCE "Placement") |

## Blocking Model

### Index space and the two regimes it splits into

In PHASE0 W keeps its size (`changes` ∈ {leading, h}), so the op is a re-grouping of whole rows: global logical row `g` (0 ≤ g < `tensor_rows = volume / W`) is row `g % H_in` of input plane `g // H_in` and row `g % H_out` of output plane `g // H_out`.

- **Row-major**: a page is a row, pages follow each other in row order in both tensors, so output page p = input page p.
- **Tile, `leading`, or `h` with both `H_in` and `H_out` multiples of 32**: every tile holds the same 32 rows × 32 columns in both tensors, so output page p = input page p.
- **Tile, `h` otherwise**: output tile-row k (plane `k // out_plane_tile_rows`, tile-row `k % out_plane_tile_rows` of it) holds `valid_rows(k) = min(32, H_out − 32·(k % out_plane_tile_rows))` logical rows, which lie in one or more input tile-rows at other row offsets; whole face-rows move between tiles. Column `w` of the output depends on column `w` of the input only.

Scope names (blocking-model §3): `tensor_pages`, `tensor_w_tiles = ceil(W/32)`, `in_plane_tile_rows = ceil(H_in/32)`, `out_plane_tile_rows = ceil(H_out/32)`, `in_planes = volume / (H_in·W)`, `out_planes = volume / (H_out·W)`, `tensor_in_tile_rows = in_planes · in_plane_tile_rows`, `tensor_out_tile_rows = out_planes · out_plane_tile_rows`; per core `core_pages`, `core_tile_cols`, `core_out_tile_rows`; per block `block_pages`, `block_tile_cols`, `block_out_tile_rows`, `num_blocks_this_core`, `block_idx`. Every tile count uses `ceil` and is per plane (per image), never `floor(batch·H/32)`.

### Axes

| Axis | Character (+ one-clause reason) | Extent knob | Phase 0 value | Knob source | Core-assignment | Later unlock |
|------|--------------------------------|-------------|---------------|-------------|-----------------|--------------|
| **A. `page`** (regime A: flattened over every leading dim × H × W-tiles; output page p ← input page p) | independent — each output page is one input page, no cross-page dependency | `block_pages` | `clamp(floor(BLOCK_BYTES_TARGET / entry_bytes), 1, max(core_pages))`; 32 KiB of pages (16 bf16 tiles; 1 page of a 16,800 B row) | host constant `BLOCK_BYTES_TARGET` = 32,768 → `block_pages` derived once on host, passed as CT arg | contiguous page ranges over the full grid: `ttnn.split_work_to_cores(grid, tensor_pages, row_wise=True)`; grid queried | knob-turn |
| **B1. `out_tile_row`** (regime B: output tile-rows, `tensor_out_tile_rows`) | independent — an output tile-row is built from its own input rows, no combine; adjacent output tile-rows can share one boundary input tile-row (an overlapping footprint, blocking-model §4: a coarser block reads it fewer times) | `block_out_tile_rows` | the core's whole assignment `core_out_tile_rows`; shrunk only if the window does not fit after `block_tile_cols` reached 1 | host, from the fit search (Regimes); CT/RT arg | contiguous ranges in `row_groups` groups (after the column groups); `row_groups = max(1, min(tensor_out_tile_rows, tensor_in_tile_rows, num_cores // col_groups))` | knob-turn |
| **B2. `tile_col`** (regime B: W-tiles, `tensor_w_tiles`) | independent, and with no overlap at all — output column w reads input column w only | `block_tile_cols` | the core's whole assignment `core_tile_cols`; shrunk first when the window does not fit (it adds no traffic) | host, from the fit search; CT/RT arg | contiguous ranges in `col_groups = min(tensor_w_tiles, num_cores)` groups, split first (cheapest: zero duplication) | knob-turn |
| **sub-tile rows / face-rows** (regime B, inside a tile) | not a block axis: rows inside one tile are moved by the regroup operation, which walks them | — (whole tile; the run walk covers all 32 rows) | 32 rows | tile geometry (`page_bytes / 64` = bytes per face-row) | not assigned across cores: a tile is indivisible across cores (splitting it would fragment 2 KiB transfers into 32 B ones) | — |
| **W within a row-major page** (regime A) | not a block axis: a row-major page is moved whole | — (whole page) | whole row | `input.buffer_page_size()` | not assigned across cores: a page is one NoC transfer; cutting it fragments the transfer | knob-turn only with regime `w_regroup_rm` |

No axis is **dependent**: every output element is exactly one input element, so no split needs a combine. **Reuse-shared check** (each operand × chosen split): the input is the only operand; under the B1 split a boundary input tile-row is needed by the two cores on either side of a row-group boundary — at most one input tile-row × `core_tile_cols` per boundary, bounded by the `row_groups ≤ tensor_in_tile_rows` cap (Traffic ranking). No operand is read by every core.

Phase 0 is multi-core: both regimes spread their independent axes over the whole grid from the first build; the grid comes from `device.compute_with_storage_grid_size()`; every extent is a parameter with one host source; no kernel loop bound is a literal.

### Buffer-depth knobs

| CB | Depth knob | Phase 0 value | What the depth buys |
|----|------------|---------------|---------------------|
| `dfb_pages` (A) | `PAGE_RING_DEPTH` | 2 (ring = 2 × `block_pages`) | the reader fills one block while the writer drains the other; implicit sync's txn-id batches fall on the halves |
| `dfb_input_window` (B) | `input_window_depth` (derived) | `2 if max(num_blocks_this_core) > 1 else 1` | with several blocks per core, the reader loads block i+1's window while the relayout regroups block i; with one block per core a second window would never be filled |
| `dfb_output_tiles` (B) | `OUTPUT_RING_DEPTH` | 2 (ring = 2 × `output_quantum`, `output_quantum = min(block_tile_cols, max(1, floor(BLOCK_BYTES_TARGET / page_bytes)))`) | the relayout assembles the next output tiles while the writer sends the previous ones |

### Mechanism caps

| Mechanism | Cap on which extent | Clamp | What happens unclamped |
|-----------|--------------------|-------|------------------------|
| Implicit-sync NoC transfer = one transaction per ring entry; `NOC_MAX_BURST_SIZE` = 256 words × 256 B = 64 KiB (`tt_metal/hw/inc/internal/tt-2xx/quasar/noc/noc_parameters.h:395-396`) | row-major page bytes (`W · element_size`) | refuse `page_bytes > NOC_MAX_BURST_SIZE` with `NotImplementedError` (TARGET's largest: 8,400 × 4 = 33,600 B) | a multi-packet read per entry |
| DFB entry alignment (L1 16 B) and the implicit overload transferring a whole entry (`noc.h:913-940`, "size … provided by the DataflowBuffer") | `entry_bytes` | `entry_bytes = round_up(page_bytes, ttnn.get_l1_alignment())`; holds `entry_bytes ≤ buffer_aligned_page_size()` for input and output (same buffer type, DRAM 64 B / L1 16 B alignment), so the extra bytes read/written stay inside the page's own slot (probe: W = 47 bf16 → page 94 B, entry 96 B, DRAM slot 128 B) | misaligned ring entries; a transfer that spills into the next page's slot |
| DFB ring ≤ 1,048,560 B (target page) | `dfb_input_window` = `input_window_depth × window_entries × page_bytes`; `dfb_pages` = `2 × block_pages × entry_bytes` | regime B fit search shrinks `block_tile_cols`, then `block_out_tile_rows`; regime A: `block_pages ≤ BLOCK_BYTES_TARGET / entry_bytes` (≤ 64 KiB + one page) | ring creation fails |
| `reserve_back` / `wait_front` / `push_back` / `pop_front` take `uint16_t` (`tt_metal/hw/inc/api/dataflow/dataflow_buffer.h:243-246`) | `window_entries` | implied by the 1 MiB ring cap (≤ 512 tiles of 2 KiB) | count truncation: hang |
| Window contiguity: one `scoped_read_lock(window_entries)` pointer covers the window only if it never wraps (`dataflow_buffer.h:447-466`) | `dfb_input_window` capacity vs. its quantum | capacity = `input_window_depth × window_entries`, and every push and pop is exactly `window_entries` (nominal, also for a ragged block whose real tile count is smaller) | the regroup reads across the ring end: silent wrong rows |
| Implicit-sync txn-id batching: a side's entries split evenly over the txn ids (`DataflowBuffer.md` C4, "How the ISR threshold is programmed") | ring entry counts of `dfb_pages`, `dfb_output_tiles` | both are `2 × quantum`: even | host rejects the ring, or uneven credits |
| Tile geometry: 32×32 tiles of four 16×16 faces | face-row offsets of the regroup | refuse a tile shape other than 32×32 (`input_tensor.tile.tile_shape`); `face_row_bytes = page_bytes / 64`, `face_bytes = page_bytes / 4`, derived on host and passed as CT args | rows copied to wrong offsets |
| Run table of one output tile-row | runs per output tile-row | ≤ 32 (a run is ≥ 1 row; a tile has 32 rows): a fixed 32-entry local array | — (cannot overflow) |
| Quasar XIP DM build: `/` and `%` pull libgcc divide routines and break the ELF layout (comment at `ttnn/cpp/ttnn/operations/experimental/quasar/move/device/kernels/dataflow/reader_unary_local_l1_copy_backwards.cpp:35-40`) | the kernels' row / tile-row cursors | the host computes every block's start cursors; kernels advance them by increments and compares only | kernel build failure (`tt_elffile.cpp` TT_THROW) |
| Uneven `wait_front` sizes without matching pops are a listed DFB hang cause (`DataflowBuffer.md:685`) | `dfb_input_window` wait quantum | one `wait_front(window_entries)` and one `pop_front(window_entries)` per block, never growing partial waits | hang |

### Regimes

| Regime | Status | Predicate | Block | Data movement vs. minimum | What a bigger block buys |
|--------|--------|-----------|-------|---------------------------|--------------------------|
| `page_identity` | built | `layout == ROW_MAJOR_LAYOUT` (W kept), **or** `layout == TILE_LAYOUT` and (`H_in == H_out` or (`H_in % 32 == 0` and `H_out % 32 == 0`)) | `block_pages` consecutive pages of the core's range (`core_pages`) | minimum for a copying op: each input page crosses the input's memory boundary (DRAM or L1 banks) once in, each output page once out; no cross-core traffic. (The absolute minimum, no traffic, is a view: rejected below) | fewer txn-id batches / ISR credit events per core (one per block, intended), more bytes in flight per batch; pipeline fill/drain once per core. No init and no reconfig exists to amortize (no compute) |
| `tile_row_regroup` | built | `layout == TILE_LAYOUT` and `W_in == W_out` and `H_in != H_out` and not (`H_in % 32 == 0` and `H_out % 32 == 0`) | `block_out_tile_rows × block_tile_cols` output tiles, with its input window of `n_in(block) × block_tile_cols` whole input tiles resident | output: minimum (each output tile written once). Input: each input tile read once per block that overlaps it; blocks along `tile_col` never overlap, blocks along `out_tile_row` share at most one input tile-row per boundary, so total reads ≤ `(tensor_in_tile_rows + row_groups − 1 + extra row blocks) × tensor_w_tiles` < 2 × minimum (cap `row_groups ≤ tensor_in_tile_rows`); with one block per core the extra row blocks are 0. No cross-core traffic | fewer boundary input tile-rows re-read (overlapping footprint along `out_tile_row`); one reserve + one trid barrier + one push (reader) and one wait + one pop (relayout) per block, intended; the run walk is computed once per output tile-row and reused over the block's `block_tile_cols` tiles |
| `view` (return the input's buffer under the new shape, no program) | rejected — violates the run's rule of exactly one native dispatch per invocation; superseded by `page_identity`, which is the same result with one copy | — | — | zero traffic (below the copy minimum) | — |
| `w_regroup_rm` (row-major, W changes: output row j = elements `[j·W_out, (j+1)·W_out)` of the flat stream) | **built (Refinement 1)**: `plan_w_regroup_rm` / `_w_regroup_rm_artifacts`; the window reader of `tile_row_regroup` on a one-column grid of row pages, a byte-run relayout (`RESHAPE_W_REGROUP_RM`), regime A's page writer; output pages split in `row_groups = min(out_pages, in_pages, num_cores)` balanced ranges, block `block_out_rows` (whole range, fit search by bisection). Originally: deferred — no PHASE0 cell (`changes = w` is outside PHASE0; no traced caller changes W). Scheme-change: rows are cut at byte offsets that are not page- or 16 B-aligned, so input pages are staged whole in a window and a relayout DM kernel copies byte runs into output pages. Reachable: the same reader → relayout → writer topology and window ring as `tile_row_regroup`; only the run walker changes (byte runs between row pages instead of face-row runs between tiles) | `ROW_MAJOR` and `W_in != W_out` | `block_out_rows × 1` output pages with their input-page window | minimum + a window-boundary re-read per block (same bound as `tile_row_regroup`) | as `tile_row_regroup` |
| `w_regroup_tile` (tile, W changes: output row g = elements `[g·W_out, (g+1)·W_out)`, padded for the new shape) | **built (Refinement 2)**: `plan_w_regroup_tile` / `_w_regroup_tile_artifacts`; regime B's window reader (input tile-rows × a per-cluster input tile-column window `[win_col0, win_col0 + win_cols)`), an element-run relayout (`RESHAPE_W_REGROUP_TILE`: one input cursor per output row, runs cut at input row, input face-row and output face-row ends, phase-aware CPU copy), regime B's tile writer (`RESHAPE_TILE_WRITER`); output tile-rows split in `row_groups = min(t_out, t_in, num_cores)` balanced ranges, output tile columns in the largest `col_groups` whose reads stay ≤ max(2 × minimum, column-unsplit reads); block `block_out_tile_rows` (whole range; fit search: largest fitting block, then more column groups). Originally: deferred — no PHASE0 cell. Scheme-change: elements move inside face-rows at element granularity (2 or 4 B), so the relayout copies element runs (CPU copy; NoC needs 16 B alignment). | `TILE` and `W_in != W_out` | `block_out_tile_rows × col_count` output tiles, the window spanning the input tile-rows and tile-columns that feed them | minimum + a boundary input tile-row per row range + the overlap of column windows | as `tile_row_regroup` |
| `subtile_dram_gather` (read face-row spans straight from DRAM into output tiles, no window) | rejected — 32 B transfers at 32 B offsets against the 64 B DRAM alignment, and 64 transactions per tile instead of one; superseded by `tile_row_regroup` (whole-tile reads, catalog `tile_reorder`: move whole pages) | — | — | input read once in theory, at 64× the transaction count | — |
| `untilize_move_tilize` (NEOs untilize input tile-rows, a DM drops/inserts padding rows, NEOs tilize) | rejected — two compute passes and still a DM row move; superseded by `tile_row_regroup`, which moves the same bytes with no compute. (A dead end: no good scheme passes through it) | — | — | same DRAM traffic + two L1 round trips through the NEOs | — |
| `boundary_tile_row_mcast` (a core multicasts its last input tile-row to the next row group instead of both reading it) | rejected — the payload is at most one input tile-row × `core_tile_cols` per core boundary, the same bytes as the DRAM re-read it would replace, plus a semaphore rendezvous; superseded by `tile_row_regroup` with the `row_groups ≤ tensor_in_tile_rows` cap | — | — | replaces ≤ 1 tile-row re-read per boundary with a NoC multicast of the same size | — |

**Regime-selection function** (host, exact; PHASE0 has already refused `W_in != W_out`):

```
H_in  = in_shape[-2]  if len(in_shape)  > 1 else 1
H_out = out_shape[-2] if len(out_shape) > 1 else 1
if layout == ROW_MAJOR_LAYOUT:                               -> page_identity
elif H_in == H_out or (H_in % 32 == 0 and H_out % 32 == 0):  -> page_identity
else:                                                         -> tile_row_regroup
```

`page_identity` additionally asserts on host that `input.buffer_num_pages() == output.buffer_num_pages()` and `input.buffer_page_size() == output.buffer_page_size()` (true by the predicate; a rank-1 TILE `[512]` is stored as `[32, 512]`, 16 tiles, the same as `[1,1,1,512]` — probe).

**Regime-pinned tests (required, in addition to the acceptance test).** The regime does not depend on the grid, so each test pins it by shape:

| Test | Shape → new shape | Layout | Regime |
|------|-------------------|--------|--------|
| RM `h`, wide page | `(1,64,8400) → (1,4,16,8400)` | ROW_MAJOR, DRAM | page_identity |
| RM non-aligned W (94 B page, 96 B entry) | `(4,128,47) → (2,2,128,47)` | ROW_MAJOR, L1 | page_identity |
| TILE `leading`, rank 1 | `(512,) → (1,1,1,512)` | TILE, L1 | page_identity |
| TILE `h`, both H multiples of 32 | `(1,64,128) → (1,2,32,128)` | TILE, DRAM | page_identity |
| TILE `h`, H_out < 32 (traced call) | `(1,64,400) → (1,4,16,400)` | TILE, L1 | tile_row_regroup |
| TILE `h`, H_in non-aligned, rows cross planes | `(1,40,40,512) → (1,1,1600,512)` | TILE, DRAM | tile_row_regroup |
| TILE `h`, H_in < 32 (window of up to 32 input tile-rows per output tile-row) | `(1,64,4,64) → (1,8,32,64)` and `(2,3,5,64) → (2,1,15,64)` | TILE, DRAM | tile_row_regroup |
| TILE `h`, W non-aligned, ragged blocks | `(2,1,100,47) → (2,4,25,47)` | TILE, L1 | tile_row_regroup |
| a test that forces several blocks per core (`block_tile_cols < core_tile_cols`, window depth 2) through a host-side knob override | `(1,64,8400) → (1,4,16,8400)` with `block_tile_cols = 2` | TILE, DRAM | tile_row_regroup |

### Traffic ranking

Counted in crossings per tensor of its memory boundary (DRAM, or the L1 banks for an L1 tensor). No nanoseconds.

| Rank | Candidate split | Input crossings | Output crossings | Cross-core | Notes |
|------|-----------------|-----------------|------------------|------------|-------|
| 0 | none (a view) | 0 | 0 | 0 | rejected: one-dispatch rule |
| 1 | A: pages over cores (`page_identity`) | 1 | 1 | 0 | the minimum for a copy; every core-range cut is free |
| 1 | B: `tile_col` only | 1 | 1 | 0 | minimum, but engages only `min(tensor_w_tiles, grid)` cores (4 of 32 for W = 128) |
| 2 | B: `tile_col` first, then `out_tile_row` in `row_groups ≤ tensor_in_tile_rows` groups (**chosen**) | ≤ `1 + (row_groups − 1) / tensor_in_tile_rows` < 2 | 1 | 0 | fills the grid; the extra reads are one boundary input tile-row per row-group boundary |
| 3 | B: `out_tile_row` uncapped (`row_groups = tensor_out_tile_rows`) | up to `(tensor_in_tile_rows + tensor_out_tile_rows − 1) / tensor_in_tile_rows` (3× for `(1,1,50,128)→(1,5,10,128)`) | 1 | 0 | more cores, more re-reads: a perf lamp's alternative |
| — | split a dependent axis + combine | — | — | — | does not exist: no axis is dependent |

**Chosen and why.** Regime A splits its single independent axis; regime B cuts columns first (no duplication), then rows only until the grid is filled, capped so that input reads stay below twice the minimum. This is an explicit trade (blocking-model §4): the column-only split is cheaper in bytes but leaves most clusters idle whenever `tensor_w_tiles < 32`. The cheaper split stays reachable by setting `row_groups` to 1, a one-line knob-turn.

### Block schedule

Regime `page_identity` (every core, over its `core_pages`):

```cpp
for (uint32_t block_idx = 0; block_idx < num_blocks_this_core; ++block_idx) {  // block: block_pages pages (last: last_block_pages)
    load_pages_block(block_idx);   // reader: one implicit-sync read per page into dfb_pages
    store_pages_block(block_idx);  // writer: one implicit-sync write per page out of dfb_pages
}
```

| Operation | Block shape | Resident across it | Intended fixed-cost frequency |
|-----------|-------------|--------------------|-------------------------------|
| `load_pages_block` | `block_pages` pages, page ids `start_page + block_idx·block_pages + i` | nothing | no barrier, no reserve/push: credits posted by the DM0 ISR once per txn-id batch (≈ one block); one `noc_async_read_set_trid(0, …)` at kernel exit |
| `store_pages_block` | the same `block_pages` pages, same ids in the output | nothing | no barrier, no wait/pop: acks by the ISR once per batch; one `noc_async_write_set_trid(0, …)` at kernel exit |

Regime `tile_row_regroup` (every core, over its `core_out_tile_rows × core_tile_cols`; blocks ordered column-block outer, row-block inner):

```cpp
for (uint32_t block_idx = 0; block_idx < num_blocks_this_core; ++block_idx) {  // block: block_out_tile_rows x block_tile_cols output tiles
    load_window_block(block_idx);    // reader: n_in(block) x wb whole input tiles into dfb_input_window
    regroup_rows_block(block_idx);   // relayout: build every output tile of the block from the window
    store_tiles_block(block_idx);    // writer: one implicit-sync write per output tile
}
```

| Operation | Block shape | Resident across it | Intended fixed-cost frequency |
|-----------|-------------|--------------------|-------------------------------|
| `load_window_block` | input tile-rows `[window_first, window_first + n_in)` × columns `[w0, w0 + wb)`, read tile-row-major into window entry `e = (tr − window_first)·wb + (w − w0)` | — | one `reserve_back(window_entries)`, all reads tagged `{.trid = 0}`, one `async_read_barrier<TXN_ID>({.trid = 0})`, one `push_back(window_entries)` per block (nominal count, also when `n_in·wb < window_entries`) |
| `regroup_rows_block` | `block_out_tile_rows × wb` output tiles; per output tile-row k one run table (≤ 32 runs: `{src tile-row offset, src row, dst row, length}`, split at input tile-row ends and at row 16 of source and destination), reused for all wb tiles of the row | the whole input window, for the whole block | one `wait_front(window_entries)` and one `scoped_read_lock(window_entries)` per block; per output tile one `reserve_back(1)` + `scoped_write_lock(1)` + `push_back(1)` (the `dfb_output_tiles` quantum; see the Circular Buffers table); per run two contiguous copies (left face column: faces 0/2, right: 1/3) of `length · face_row_bytes`; one `pop_front(window_entries)` per block |
| `store_tiles_block` | the block's output tiles, ids `k·tensor_w_tiles + w`, k-major | — | no barrier, no wait/pop (implicit sync); one `noc_async_write_set_trid(0, …)` at kernel exit |

Run walk (per output tile-row k, inside `regroup_rows_block`): for `r` in `[0, valid_rows(k))` the destination row is `r` and the source is the input cursor `(src tile-row, src row)`, advanced by one row per step: `src row += 1`; at the end of an input tile-row (`src row == 32`, or the input plane's last valid row passed) move to the next input tile-row and row 0. The cursor at the block's first row is computed on host and passed in; the kernels never divide (Mechanism caps). Rows `valid_rows(k)..31` of an output tile are not written.

**Stall-shadow check.** Stages that wait: the relayout waits for the whole window; the writer waits for the relayout. The work that does not depend on the current window is loading the next block's window. It runs in that stall whenever a core has more than one block (`input_window_depth = 2`). Inside one block, regrouping the output tile-rows whose input tile-rows have already landed would also be independent. It is not used, because it needs growing partial `wait_front` counts, a listed DFB hang cause (`DataflowBuffer.md:685`). The perf lamp below measures the block split that gets this overlap with no extra traffic. No reorder changes any value: the op is an exact copy.

### Perf lamps

| Lamp | Why the default may be wrong here | Nearby alternative to measure |
|------|-----------------------------------|-------------------------------|
| Overlap (regime B) | one block per core reads the whole window before any regroup starts, so read and regroup do not overlap | two column blocks per core (`block_tile_cols = ceil(core_tile_cols / 2)`, `input_window_depth = 2`): costs no traffic (columns never overlap) |
| Grid synchronization / duplication (regime B) | filling the grid with row groups re-reads one boundary input tile-row per group; on tiny tensors (`(2,1,100,47)`: 16 clusters, 2 tiles each) launch and handshake cost may dominate | `row_groups = 1` (column-only, minimum traffic) and `row_groups = tensor_out_tile_rows` (uncapped) |
| Grid synchronization (regime A) | small tensors (`(1,1,4,8400)` RM: 4 pages) spread one page per cluster | a minimum of `block_pages` pages per engaged cluster |
| Transaction size (regime A) | implicit sync moves one page per transaction; for small row-major pages (288 B for W = 144) per-transaction cost dominates | bank-run coalescing: input and output have identical per-bank page layouts (same page size, count and buffer type, both interleaved from bank 0), so one transfer can move k pages of one bank (explicit sync, entry = k pages); catalog `read_coalescing` |
| Bytes in flight (regime A) | `BLOCK_BYTES_TARGET` = 32 KiB is twice the catalog's 4–8-tile sweet spot (`double_buffer`), chosen for small row-major pages | 16 KiB and the whole core assignment |
| Thread plan (both) | (1, 0, 1) assumes one hart per side issues fast enough; for small pages a hart may be issue-bound (catalog `split_reader`), and in regime B the single relayout hart copies every output byte | (2, 0, 2) for regime A on row-major pages ≤ 1 KiB; two relayout harts splitting a block's output tile-rows in regime B. Measure on a timed backend |
| Relayout copy engine (regime B) | a CPU word copy through the uncached alias moves 4 B per load and store | self-aimed NoC reads (`dataflow_kernel_lib::local_addr`, `local_copy_helpers_dataflow.hpp:57`) into the output entry, once a public NoC address of the window's read pointer exists |

## Dataflow Strategy

| Stage | Format | Mechanism | Notes |
|-------|--------|-----------|-------|
| A1 input → `dfb_pages` | pages (row-major rows or tiles), unchanged bytes | reader: `noc.async_read<NocOptions::TXN_ID>(acc_input, dfb_pages, {.page_id = p}, {})`, implicit sync | `TensorAccessor` serves DRAM and L1 interleaved alike (R5) |
| A2 `dfb_pages` → output | same | writer: `noc.async_write<NocOptions::TXN_ID>(dfb_pages, acc_output, {}, {.page_id = p})`, implicit sync | the output page id equals the input page id |
| B1 input → `dfb_input_window` | whole input tiles | reader: explicit sync; `noc.async_read<NocOptions::TXN_ID>(acc_input, dfb_input_window, page_bytes, {.page_id}, {.offset_bytes = e·page_bytes}, {.trid = 0})`, one barrier on trid 0 per block | explicit, because the window is assembled from several transfers and pushed at its nominal count; ring listed in the reader's `disable_dfb_implicit_sync_for` |
| B2 `dfb_input_window` → `dfb_output_tiles` | face-row spans, same dtype bytes | relayout DM kernel: CPU copy (uint32 words, both ends 16 B aligned) through `scoped_read_lock` / `scoped_write_lock` pointers (uncached alias on Quasar DM; the write lock flushes on release) | no NoC transfer in this kernel; both rings in its `disable_dfb_implicit_sync_for` |
| B3 `dfb_output_tiles` → output | whole output tiles | writer: implicit sync, one tagged write per tile | the producer side is explicit and the consumer side implicit: the protocol is chosen per side |

No stage communicates between clusters; no deferred regime needs it.

**TARGET axes as regimes.**

| TARGET value | Status | Class |
|--------------|--------|-------|
| `dtype` float32 / int32 / uint32 | not in PHASE0 | knob-turn: page bytes and face-row bytes are derived from `buffer_page_size()`; no value passes through another dtype (R1), so the copy is exact for every dtype |
| `rank` 2 | not in PHASE0 | knob-turn: `H = shape[-2]` already covers it; only the validate gate changes |
| `changes` w | not in PHASE0 | scheme-change: regimes `w_regroup_rm`, `w_regroup_tile` |
| `memory_config` DRAM / L1 | built | — |
| `sharding` interleaved | built (the only TARGET value; R8 refuses a sharded input by name) | — |
| `layout` TILE / ROW_MAJOR, `alignment` all three | built | — |

No structural impossibility to add: the spec's `INVALID = []` is right (no block-float dtype in TARGET).

## Work Distribution

| Field | Value |
|-------|-------|
| Work unit | A: a page. B: one output tile (`out_tile_row`, `tile_col`), grouped into blocks |
| Grid | `device.compute_with_storage_grid_size()` (8 × 4 on quasar); `num_cores = grid.x · grid.y` |
| Per-core work | A: `ttnn.split_work_to_cores(grid, tensor_pages, row_wise=True)` → contiguous `(start_page, core_pages)`. B: `col_groups = min(tensor_w_tiles, num_cores)`; `row_groups = max(1, min(tensor_out_tile_rows, tensor_in_tile_rows, num_cores // col_groups))`; core index `rg·col_groups + cg`, placed row-major on the grid; `tile_col` and `out_tile_row` each cut into balanced contiguous ranges |
| Remainder | balanced ranges whose sizes differ by ≤ 1 (the first `rem` groups take one more); regime B's ragged blocks keep the nominal window push/pop count; every tile count is `ceil` and per plane |
| Worked (quasar, bf16) | A `(1,64,8400)` RM: 64 pages of 16,800 B on 32 clusters × 2, `block_pages = 1`, ring 2 × 16,800 B. A `(1,40,40,512)` RM: 1,600 pages of 1 KiB, 50 per cluster, `block_pages = 32`, 2 blocks per cluster. A `(1,1,2048,256)→(1,4,512,256)` TILE: 512 tiles, 16 per cluster, `block_pages = 16`. B `(1,64,400)→(1,4,16,400)`: `tensor_w_tiles = 13`, `col_groups = 13`, `row_groups = min(4, 2, 2) = 2` → 26 clusters, each 2 output tile-rows × 1 column, window 1 tile. B `(1,40,40,512)→(1,1,1600,512)`: `col_groups = 16`, `row_groups = 2` → 32 clusters, each 25 output tile-rows × 1 column, window 40 tiles (80 KiB), input read exactly once. B `(1,64,8400)→(1,4,16,8400)`: `col_groups = 32`, `row_groups = 1`, 8–9 columns per cluster, window 2 × 9 = 18 tiles |
| Bound | data movement: the op computes nothing; each unit is bytes in, the same bytes out (regime B adds one L1-to-L1 copy per output byte on a DM hart) |
| Thread plan | `THREAD_PLAN = {"reader_threads": 1, "relayout_threads": 1, "writer_threads": 1}` (one named host constant): (R, C, W) = (1, 0, 1): no NEO, because there is no compute kernel (the ProgramSpec precedent is `toy_spec_fill`, a DM producer and a DM consumer); regime B adds one relayout hart (3 of 6 user DM harts). The counts are passed as CT args and the kernels stride their loops by them |
| Unit dealt to compute threads | none (no compute thread). With one hart per DM kernel every ring entry goes to that hart; the host deals whole blocks to clusters |
| Tests the plan needs | (1) the engaged-cluster count read off the built `ProgramSpec` (work unit `target_nodes`) equals the split's for one shape per regime: a correct result alone does not show that every core ran. (2) In one device session, reshape after another program (e.g. a `ttnn.add`, or a reshape of the other regime). (3) In one device session, reshape followed by `ttnn.to_layout(output, ttnn.ROW_MAJOR_LAYOUT)` on a TILE output and a `ttnn.to_layout(output, ttnn.TILE_LAYOUT)` on a row-major output, for both regimes (regime B's reader is explicit-sync, its writer implicit-sync) |

## Circular Buffers

Each regime is its own `ProgramSpec`; buffers of the two never coexist.

| Semantic Name | Index | Page Size | Num Pages | Sizing rationale | Format | Producer | Consumer | Sync | Lifetime |
|---------------|-------|-----------|-----------|------------------|--------|----------|----------|------|----------|
| `dfb_pages` | DFB `pages` (A) | `entry_bytes = round_up(input.buffer_page_size(), l1_alignment)` | `PAGE_RING_DEPTH × block_pages` | spans `page` → `block_pages` (one block per txn-id batch) × depth 2; streams the core's range | input dtype (bytes copied, never unpacked) | reader | writer | implicit / implicit | whole program |
| `dfb_input_window` | DFB `input_window` (B) | `page_bytes` (one tile) | `input_window_depth × window_entries`, `window_entries = n_in_cap × block_tile_cols` | spans `out_tile_row` → `n_in_cap` (input tile-rows feeding `block_out_tile_rows`) and `tile_col` → `block_tile_cols`; streams blocks | input dtype | reader | relayout | explicit (reader: reserve / trid-tagged reads / barrier / push) / explicit (relayout: wait / pop) | whole program; one window per block |
| `dfb_output_tiles` | DFB `output_tiles` (B) | `page_bytes` | `OUTPUT_RING_DEPTH × output_quantum` | spans neither block axis: a streaming window of output tiles; streams `out_tile_row` and `tile_col` | input dtype | relayout | writer | explicit (relayout: reserve 1 / push 1) / implicit (writer) | whole program |

`n_in_cap` = the maximum over every block of the program of `n_in(block) = in_tile_row_of(last row of the block) − in_tile_row_of(first row of the block) + 1`, with `in_tile_row_of(g) = (g // H_in)·in_plane_tile_rows + (g % H_in) // 32` and the block's first row `first_global_row(k0) = (k0 // out_plane_tile_rows)·H_out + 32·(k0 % out_plane_tile_rows)`, computed on host. The relayout's `reserve_back(1)` per output tile keeps the `dfb_output_tiles` cycle closed for any block width (quantum 1), and the writer's implicit side needs one transfer per entry. Every ring has one producer kernel and one consumer kernel; producer push count = consumer wait count on every ring (A: one entry per page on both sides; B window: `window_entries` per block on both sides; B output: one per tile on both sides).

## Block Operation Realization

| # | Block operation | Block shape | Helper? | Input CB (semantic name, pages, state) | Output CB (semantic name, pages) | CB state after |
|---|-----------------|-------------|---------|----------------------------------------|----------------------------------|----------------|
| A1 | `load_pages_block` | `block_pages` pages | no (dataflow NoC API) | input tensor (accessor) | `dfb_pages`, `block_pages` posted by the ISR | — |
| A2 | `store_pages_block` | `block_pages` pages | no (dataflow NoC API) | `dfb_pages`, `block_pages` | output tensor (accessor) | entries acked by the ISR |
| B1 | `load_window_block` | `n_in(block) × wb` input tiles (pushed as `window_entries`) | no (dataflow NoC API) | input tensor (accessor) | `dfb_input_window`, `window_entries` | window resident until B2 pops it |
| B2 | `regroup_rows_block` | `block_out_tile_rows × wb` output tiles | no (raw CPU copy; helpers considered below) | `dfb_input_window`, `window_entries`, waited once, held for the block | `dfb_output_tiles`, 1 per output tile, `block_out_tile_rows × wb` per block | window popped at block end |
| B3 | `store_tiles_block` | `block_out_tile_rows × wb` output tiles | no (dataflow NoC API) | `dfb_output_tiles`, 1 per tile | output tensor (accessor) | — |

## API Mapping

| Block operation | Type | Function | File:Line | Template Params / Args | Input CB | Output CB | Which params are block knobs |
|-----------------|------|----------|-----------|------------------------|----------|-----------|------------------------------|
| A1, B3 reads/writes (implicit) | raw_api | `Noc::async_read<NocOptions::TXN_ID>(src, DataflowBuffer&, src_args, dst_args)` / `Noc::async_write<NocOptions::TXN_ID>(DataflowBuffer&, dst, src_args, dst_args)` | `tt_metal/hw/inc/api/dataflow/noc.h:913-940` | accessor + `{.page_id}`; size = the DFB's entry | input / `dfb_output_tiles` | `dfb_pages` / output | none (one entry per call) |
| B1 reads (explicit) | raw_api | `Noc::async_read<NocOptions::TXN_ID>(src, dst, size_bytes, src_args, dst_args, {.trid = 0})`, `Noc::async_read_barrier<NocOptions::TXN_ID>({.trid = 0})` | `noc.h:246-277` (async_read); pattern `.claude/references/quasar_op_examples/toy_thread_plan/kernels/reader.cpp` | `page_bytes`, `{.page_id}`, `{.offset_bytes = e · page_bytes}` | input | `dfb_input_window` | `block_tile_cols`, `block_out_tile_rows` (through `window_entries`) |
| ring credits (explicit sides) | raw_api | `DataflowBuffer::reserve_back / push_back / wait_front / pop_front` | `tt_metal/hw/inc/api/dataflow/dataflow_buffer.h:243-246` | `window_entries` (B window), 1 (B output) | — | — | `window_entries` |
| B2 copy | raw_api | `DataflowBuffer::scoped_read_lock(n)` / `scoped_write_lock(1)` → `get_ptr<uint32_t>()`, word copy | `dataflow_buffer.h:447-466`; pattern `ttnn/ttnn/operations/toy_spec_single_tensor/kernels/fill.cpp` | `window_entries`; spans of `length · face_row_bytes` | `dfb_input_window` | `dfb_output_tiles` | `block_out_tile_rows`, `block_tile_cols` |
| tile geometry | helper | `dataflow_kernel_lib::get_tile_r_dim<dfb>()` / `get_tile_c_dim<dfb>()` (optional static check of the 32×32 assumption) | `ttnn/cpp/ttnn/kernel_lib/dfb_helpers_dataflow.hpp:14-18` | DFB id | — | — | — |
| trid hand-back (implicit kernels) | raw_api | `noc_async_read_set_trid(0, noc.get_noc_id())` / `noc_async_write_set_trid(0, noc.get_noc_id())` as the kernel's last statement | target page *Ring protocols*; `toy_thread_plan/kernels/reader_implicit.cpp` | — | — | — | — |
| host | — | `ttnn.experimental.ProgramSpec`, `KernelSpec(num_threads=…)`, `DataflowBufferSpec`, `DataMovementHardwareConfig.DataMovement2XXConfig(disable_dfb_implicit_sync_for=[…])`, `WorkUnitSpec`, `ttnn.generic_op` | `.claude/references/quasar_op_examples/toy_thread_plan/toy_thread_plan_program_artifacts.py` | — | — | — | — |

**Helpers considered and rejected (every raw_api row).**

| Raw API use | Candidate helper | File:line of the mismatch | Why it cannot express the shape |
|-------------|------------------|---------------------------|---------------------------------|
| B2 copy | `dataflow_kernel_lib::local_addr` + self-aimed `Noc::async_read` (`local_copy_helpers_dataflow.hpp:57`, `:116-133`) | `dataflow_buffer.h:403-408` (`get_read_ptr()` returns the uncached CPU alias, `+ L1_UNCACHED_OFFSET`, on Quasar DM) and `:468, :482-484` (`get_noc_read_addr()` is private: "NOC APIs do not accept uncached addresses, but this is private") | the copy's source is a byte offset inside the input window; the window has no public NoC-address accessor, so a self-aimed read would have to be sourced at a private address or at the CPU alias minus a private constant. Kept as the copy-engine perf lamp |
| B2 copy | `set_read_state` / `read_with_state` (`local_copy_helpers_dataflow.hpp:74-75`) | `:74` (`template <uint32_t transfer_size>`; single packet, size baked at compile time) | run lengths vary per run (1–16 face-rows), so a compile-time size cannot express them |
| B2 copy | `tile_column_dataflow.hpp` (column writers) | `tile_column_dataflow.hpp:7-13` (moves one column, 16 elements at a stride of FACE_WIDTH) | the regroup moves contiguous face-row spans, not columns |
| B2 (whole regime) | `compute_kernel_lib::untilize` / `tilize` | `untilize_helpers.hpp:152-163`, `tilize_helpers.hpp:194-208` (NEO helpers, DFB → DFB on a compute kernel) | they change the layout but cannot drop or insert padding rows; a DM row move would still be needed, plus two compute passes (regime `untilize_move_tilize`, rejected) |
| A1/A2/B1/B3 NoC transfers | none in `kernel_lib` covers an interleaved page copy | — (the library's dataflow helpers are scalers, mcast, tile columns, local copies) | the toy examples' accessor + DFB NoC calls are the target's prescribed pattern |
| ring credits | none | — | ring protocol calls are the API itself |

No compute phase exists, so there is no `compute_kernel_hw_startup()`, no compute helper, no scaler and no `fpu_math_fidelity`.

## Key Risks and Gotchas

| Risk | Why it bites here | Mitigation in this design |
|------|-------------------|---------------------------|
| A DM→DM ring with implicit sync on both sides has no in-tree precedent (`toy_spec_fill` and Quasar `move` use explicit sync on a DM→DM ring; the toy examples pair implicit DM sides with a NEO) | regime A's only ring | implicit/implicit is the target page's default for NoC-filled and NoC-drained rings. If needed, switch the writer side to explicit sync (`wait_front(1)`, write tagged `{.trid = 0}`, `async_write_barrier<TXN_ID>({.trid = 0})`, `pop_front(1)`, ring in the writer's `disable_dfb_implicit_sync_for`). That is a local change of one side |
| Sticky transaction tag | implicit kernels leave a ring's trid in the tag register | the implicit reader and writer end with `noc_async_*_set_trid(0, …)`; the explicit reader tags every transfer `{.trid = 0}`; the relayout issues no NoC transfer; the session tests of Work Distribution |
| Window wrap | `scoped_read_lock(window_entries)` hands out one pointer; a window across the ring end would be read as garbage | capacity = `depth × window_entries` and every push/pop is exactly `window_entries` (nominal, also for ragged blocks) |
| Division in DM kernels | the Quasar XIP DM build breaks when a kernel pulls libgcc divide routines | host computes every block's start cursors (first output tile-row, its valid rows, the input plane row and tile-row of its first logical row, `window_first`, `n_in`); kernels advance by increments; tile ids by multiply-add |
| Row-major pages that are not 16 B multiples | W = 47 bf16 → 94 B pages | entry = `round_up(page, l1_alignment)` (96 B); the implicit transfer moves the entry, which stays inside the page's aligned slot (128 B in DRAM, 96 B in L1) |
| Padding rows of the regrouped tiles | output tiles of `H_out % 32 != 0` hold `valid_rows(k) < 32` rows | rows `valid_rows(k)..31` are never written; the logical elements are exact (R4); padding content is unspecified (DECISIONS) |
| Input planes shorter than a tile (`H_in < 32`) | one output tile-row then gathers from up to 32 input tile-rows, and `n_in` grows with `block_out_tile_rows · 32 / H_in` | `n_in_cap` is computed exactly on host from the real blocks; the fit search shrinks `block_tile_cols`, then `block_out_tile_rows`; regime-pinned tests with `H_in` = 4 and 5 |
| Rank-1 TILE tensors | `[512]` is stored as `[32, 512]` | regime A asserts equal page counts and page sizes on host; `H = 1` for rank 1 everywhere (the spec's `_hw`) |
| The spec's `AXES` take the call's `shape` | `tag_changes` resolves a `-1` itself | the entry point passes `(tuple(input_tensor.shape), tuple(shape))` after the X1–X3 checks, and never redefines a tagger |
| Output placement (R7) is review-checked only | the golden runner does not compare memory configs | the output spec takes `input_tensor.memory_config()`; the verifier reads `output.memory_config()` |
