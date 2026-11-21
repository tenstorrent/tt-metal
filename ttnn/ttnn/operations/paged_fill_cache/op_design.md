# Operation Design: paged_fill_cache

## Overview

| Field | Value |
|-------|-------|
| Classification | data_movement (pure tile copy with a device-side indirection; no compute kernel) |
| Goal | Write the `seq` positions of one user's K or V sequence into a paged KV cache in place, through row `batch_idx` of an int32 page table read on the device, in one `ttnn.generic_op` dispatch of a Metal 2.0 `ttnn.ProgramSpec`. |
| Math | for every input tile row `t` (positions `32t..32t+31`) of head `h` whose block `b = page_table[batch_idx, t // bs_t]` is not `-1`, and every tile column `d`: `cache_tile[((b*H + h)*bs_t + t % bs_t)*D_t + d] = input_tile[(h*seq_t + t)*D_t + d]` (whole 32x32 tiles, bit-exact) |
| References | Specification `eval/specs/paged_fill_cache/spec.py` (authoritative); `.claude/references/targets/quasar.md`; `.claude/references/backends/ttsim.md`; `.claude/references/quasar_op_examples/` (`toy_thread_plan` implicit/mixed readers, `toy_spec_single_tensor` for the in-place pattern); `tt_metal/impl/dataflow_buffer/DataflowBuffer.md` Part A, C4 and C6; Quasar's own DM-only tile op: `ttnn/cpp/ttnn/operations/experimental/quasar/slice/device/slice_program_factory_tile.cpp:97-123,198` (reader + writer, no compute kernel); perf catalog `ttnn/ttnn/operations/examples/master.md` (`double_buffer`, `width_split`, `noc_placement`, `tile_reorder`) |

## Target

| Fact | Value | Source |
|------|-------|--------|
| Target | `quasar` on the ttsim simulator | `.claude/references/targets/quasar.md` + `.claude/references/backends/ttsim.md` |
| Arch | `quasar` | target page |
| Compute grid | 8 x 4 = 32 clusters (fixed on this part); the op queries `device.compute_with_storage_grid_size()` | target page |
| L1 per core | 4,194,304 B (4 MiB) per cluster | target page |
| Unreserved budget | `ttnn.get_max_worker_l1_unreserved_size() - l1_reserve_bytes`; this design spends ~16.1 KB per cluster at Phase 0 (see `l1_ledger.md`), so no budget predicate is needed (Rule 1: inventory first; the footprint is a fixed window, not op-sized) | target page; `l1_ledger.md` |
| Host program | `ttnn.ProgramSpec` + `ttnn.generic_op` (Metal 2.0, DataflowBuffer rings) — `.claude/references/quasar_op_examples/` (README, `toy_thread_plan/toy_thread_plan_program_artifacts.py`, `toy_spec_single_tensor` in `ttnn/ttnn/operations/toy_spec_single_tensor/`) | target page |
| Threads per core | 4 NEOs (C ∈ {1,2,4}), 6 user DM harts (R + W ≤ 6), STRIDED rings need R \| C and W \| C when a compute kernel is on the ring; ≤ 32 DFBs per cluster, each ring ≤ 1,048,560 B; 16 semaphores; DM kernel binary ≤ 48 KiB | target page *Cluster model* / *Thread plan* |
| Unavailable here | (1) `ttnn.ProgramDescriptor` / Metal 1 kernels and the legacy C++ program factory — refused at kernel creation (`tt_metal/impl/kernels/kernel.hpp`). (2) `bfloat8_b` / `bfloat4_b` — no path on Quasar; refused in `validate()`, no refinement queued. (3) A data-movement kernel binding one ring as both producer and consumer (a DM "scratch" ring) — only Tensix INTRA rings support self-binding (`tt_metal/impl/dataflow_buffer/dataflow_buffer.cpp:2238-2254`); this is why the page-table row travels reader → writer (see Dataflow Strategy). | target page *Not available*; DataflowBuffer host code |

All worked numbers below are for **quasar (32 clusters, 4 MiB L1)**. Every predicate the op evaluates
queries the device (`compute_with_storage_grid_size()`, `get_max_worker_l1_unreserved_size()`,
`get_dram_alignment()`, `get_l1_alignment()`).

## Parameters

| Name | Type | Required | Valid Range | Default | CT/RT |
|------|------|----------|-------------|---------|-------|
| `cache_tensor` | `ttnn.Tensor` | yes | rank 4 `[blocks, H, block_size, D]`, TILE, DRAM interleaved, dtype = input's | — | tensor binding (writer) |
| `input_tensor` | `ttnn.Tensor` | yes | rank 4 `[1, H, seq, D]`, TILE, interleaved (Phase 0: DRAM) | — | tensor binding (reader) |
| `page_table` | `ttnn.Tensor` | yes | rank 2 `[users, blocks_per_seq]`, int32, ROW_MAJOR, DRAM interleaved | — | tensor binding (reader) |
| `batch_idx` | `int` (keyword-only; `bool` and non-int refused) | no | `0 <= batch_idx < users` | `0` | RT arg (reader: the page id of the row) |

Derived host quantities (single source each, computed once in the program-artifacts factory):

| Quantity | Formula (alignment-aware) | Passed as |
|----------|---------------------------|-----------|
| `num_heads` (H) | `input.shape[1]` (= `cache.shape[1]`) | RT arg, reader + writer |
| `seq_tiles` (seq_t) | `ceil(seq / 32)` | RT arg, reader + writer |
| `block_tiles_in_cache` (bs_t) | `ceil(block_size / 32)` | RT arg, reader + writer |
| `d_tiles` (D_t) | `ceil(D / 32)` | RT arg, reader + writer |
| `tensor_input_tiles` | `H * seq_t * D_t` (the input's tile count; per-image tile padding is moot: one image, H and W multiples of 32 in TARGET, `ceil` kept for the refinement that relaxes it) | host only (split) |
| `core_start_tile`, `core_tiles` | from `ttnn.split_work_to_cores(grid, tensor_input_tiles, row_wise=True)` | RT args per core, reader + writer |
| `pt_page_bytes` | `page_table.buffer_page_size()` (= `4 * blocks_per_seq`) | RT arg (reader: read size) |
| `pt_entry_bytes` | `align_up(pt_page_bytes, ttnn.get_dram_alignment())` | DFB entry size |
| `tile_bytes` | `input_tensor.buffer_page_size()` (2048 for bf16, 4096 for fp32) | DFB entry size |

## Tensors

### Input

| Property | `input_tensor` | `page_table` | `cache_tensor` (written only, never read) |
|----------|----------------|--------------|------------------------------------------|
| Shape | `[1, H, seq, D]`; `seq % 32 == 0`, `D % 32 == 0` | `[users, blocks_per_seq]`; `ceil(seq/block_size) <= blocks_per_seq` | `[blocks, H, block_size, D]`; `block_size % 32 == 0` |
| Dtype | cache's dtype (Phase 0: bfloat16) | int32 | bfloat16 (Phase 0) |
| Layout | TILE | ROW_MAJOR (one page = one user's row) | TILE |
| Memory | interleaved; Phase 0: DRAM (L1 is a TARGET knob-turn) | DRAM interleaved | DRAM interleaved |

### Output

| Property | Value |
|----------|-------|
| Shape | `cache_tensor.shape` |
| Dtype | `cache_tensor.dtype` |
| Layout | TILE |
| Memory | DRAM interleaved — **the caller's `cache_tensor` buffer itself** (rule R3). `io_tensors = [input_tensor, page_table, cache_tensor]`; `generic_op` returns `io_tensors.back()` (`ttnn/cpp/ttnn/operations/generic/device/generic_op_device_operation.cpp:342`); the entry point returns `cache_tensor`. No tensor is allocated. |

### Entry-point validation (R6–R9, REJECTIONS X1–X11)

`validate()` gates on the axes `dtype`, `layout`, `memory_config`, `sharding` (from
`eval.memory.memory_axes(input_tensor)`, R7) and `alignment` (`INPUT_TAGGERS = AXES`, imported from the
specification, never redefined), raising `NotImplementedError` outside SUPPORTED. Before that, the
structural checks raise `ValueError`/`TypeError` (all in `_REJECTED_WITH`): ranks 4/4/2 (X10); input
`shape[0] == 1` (X3); heads equal (X1); D equal (X2); `block_size % 32` (X7); `seq % 32` (X8); `D % 32`
(X9); `batch_idx` an `int` and not `bool` (X11), `>= 0` (X5), `< users` (X4); `ceil(seq/block_size) <=
blocks_per_seq` (X6). R8 refusals with `NotImplementedError`: cache not DRAM interleaved, page table not
int32 / ROW_MAJOR / DRAM interleaved, input dtype ≠ cache dtype. Mechanism-cap refusals (see Mechanism
caps): cache tile count ≥ 2^32, `pt_entry_bytes * 2 > 1,048,560`. Page-table *values* are not checked
(they live on the device; R4/R5).

### Structural impossibilities

None to add: TARGET is `{bf16, fp32} x TILE x {DRAM, L1} x interleaved x tile_aligned`, all buildable.

## Blocking Model

The op is a copy of whole tiles with a per-tile-row indirection. **No axis is dependent** (nothing
spans an axis: no reduction, no scan), so there is no "split the dependent axis + combine" candidate;
the only cross-core-shared operand is the page-table row.

### Axes

| Axis | Character (+ one-clause reason) | Extent knob | Phase 0 value | Knob source | Core-assignment | Later unlock |
|------|--------------------------------|-------------|---------------|-------------|-----------------|--------------|
| `head` (H) | independent — each head's tiles go to their own head slot of the cache, no cross-head term | folded into `block_extent[input_tile]` (the three input axes `head`, `seq_tile_row`, `d_tile` are contiguous in the input's tile order and treated identically, so they fold into one axis — `blocking-model.md` §1) | covered by the fold | — | spread with the fold | knob-turn |
| `seq_tile_row` (seq_t) | independent — a tile row's destination depends only on its own page-table entry; the entry is an indirection, not a cross-block dependency | folded (as above) | covered by the fold | — | spread with the fold | knob-turn |
| `d_tile` (D_t) | independent — elementwise copy along D; contiguous page ids in both tensors within a tile row | folded (as above) | covered by the fold | — | spread with the fold (a core may start/end mid tile-row: required for the 4096/8192-wide, one-tile-row INPUTS) | knob-turn |
| `input_tile` = flatten(`head`, `seq_tile_row`, `d_tile`), page id of the input | independent (all three constituents are) | `core_tiles` (block extent = this core's whole contiguous range; `num_blocks_this_core = 1`) | whole per-core assignment: 1–8 tiles for every INPUTS entry on 32 clusters (worked table in Work Distribution) | host: `split_work_to_cores` result → RT args `core_start_tile`, `core_tiles` | `split_work_to_cores(grid, tensor_input_tiles, row_wise=True)` over the full grid, unit = 1 tile | knob-turn (unit size `split_unit_tiles` for the grid-sync lamp) |
| `page_table_entry` (the row's `blocks_per_seq` int32 values) | reuse-shared — every core's walk looks entries up; the row does not vary along the split axis, so every core needs the same bytes | whole row (`pt_page_bytes`) — one page, one NoC read per copy | the whole row: 1 page | host: `page_table.buffer_page_size()` | not split: every engaged cluster reads the row (2 copies, one per kernel) | scheme-change (read once + multicast) — `rejected`, see Regimes |
| `cache_block` (the cache's `blocks` axis) | independent output axis, scatter target — never traversed; reached only through page-table values | none: not walked (0 tiles read; written tiles are exactly the mapped input tiles) | — | — | not assigned: destinations follow the input tiles' owners | — |

Every axis has a decision; there is no intermediate data produced by any stage (the ring entry *is* the
input tile, written out unchanged), so there is no second table to re-run.

**Why `core_tiles` (the whole assignment) and not a smaller block:** no tile is used twice, so the block
extent changes no byte count (`blocking-model.md` §4: extent invisible to the split ranking). What the
block extent amortizes is the per-block fixed cost: the page-table load (one per block → one per core)
and the pipeline fill/drain. Holding the block *resident* buys nothing for a copy, so the L1 window is
the separate buffer-depth knob below; the block streams through it.

### Buffer-depth knobs

| CB | Depth knob | Phase 0 value | What the depth buys |
|----|------------|---------------|---------------------|
| `dfb_input_tiles` | `TILE_RING_ENTRIES` (host constant; must be even, see caps) | 8 entries (= two implicit-sync txn-id halves of 4) | tiles in flight between reader and writer: the runtime splits an even ring into 2 txn-id batches (`DataflowBuffer.md` C4, `compute_optimal_txn_id_count`), so the reader fills one half while the writer drains the other — 4 NoC reads outstanding per batch, the `double_buffer` catalog sweet spot (block 4–8, ring = 2 x block). Covers every INPUTS core (≤ 8 tiles) without the reader ever stalling on space. |
| `dfb_page_table_row` | `PT_RING_ENTRIES` = 2 (structural, not a tuning knob) | 2 | one copy of the row for each kernel: entry 0 is pushed to the writer, entry 1 stays reserved by the reader as its own lookup table and is pushed at the reader's end |

### Mechanism caps

| Mechanism | Cap on which extent | Clamp | What happens unclamped |
|-----------|--------------------|-------|------------------------|
| Implicit-sync txn-id batching (`DataflowBuffer.md` C4) | `TILE_RING_ENTRIES` | even (divisible by 2 x 1 producer x 1 TC) | odd → 1 txn id: threshold = whole ring, reader and writer serialize (no overlap); not wrong, but the double buffer silently disappears. Host asserts `TILE_RING_ENTRIES % 2 == 0`. |
| Ring size ≤ 1,048,560 B (target page) | `TILE_RING_ENTRIES * tile_bytes`; `2 * pt_entry_bytes` | 8 x 4096 B (fp32) = 32 KB ≪ cap; `pt_entry_bytes ≤ 524,280` ⇒ `blocks_per_seq ≤ 131,056` — `validate()` refuses beyond | DFB creation fails at launch |
| Implicit-sync read/write size = one ring entry (`noc.h:923,938`: no size argument) | entry size of `dfb_input_tiles` | `entry_size = input_tensor.buffer_page_size()` (= cache's page size; dtypes equal by R8) | partial-tile copies → wrong cache values |
| One NoC transaction per interleaved page | transfer unit | 1 tile per read / write (consecutive page ids sit in different DRAM banks: no coalescing across pages) | — (no larger contiguous unit exists on an interleaved tensor) |
| uint32 page index in `TensorAccessor` / kernel arithmetic | cache tile id `((b*H + h)*bs_t + r)*D_t + d` | `validate()`: `blocks * H * bs_t * D_t < 2^32` | silent wrap → writes into the wrong cache tile |
| Page-table value range | `b` from the device | not checked (spec: values other than a block or `-1` are unspecified) | out-of-range block writes outside the cache — caller contract, documented |
| DRAM read alignment | page-table row read | ring entry sized `align_up(pt_page_bytes, dram_alignment)`; read size = `pt_page_bytes` from the page start (DRAM-aligned); second copy lands at `offset_bytes = pt_entry_bytes` (16-B L1-aligned since 64 is) | misaligned L1 destination for the second copy |

### Regimes

| Regime | Status | Predicate | Block | Data movement vs. minimum | What a bigger block buys |
|--------|--------|-----------|-------|---------------------------|--------------------------|
| **R-filter: tile-granular split over the full grid, page-table row read per core, unmapped tile rows filtered at the source** | built | every valid call (the only regime; all INPUTS, all TARGET dtypes/placements) | `core_tiles` (whole per-core contiguous input-tile range), streamed through a `TILE_RING_ENTRIES` window | **minimum** for the tensors against the DRAM boundary: each *mapped* input tile crosses DRAM once (read), each written cache tile once (write), the cache is never read, unmapped (`-1`) tile rows are neither read nor written. Above minimum only by the page-table row: 2 page reads per engaged cluster (≤ 64 x 64 B = 4 KB vs. ≥ 2 KB per tile moved) | one page-table load + one DFB init/drain + one pipeline fill per block (intended frequency: once per core per call); per-tile cost (one NoC read, one NoC write, txn-id credit per half-ring) is extent-invariant |
| R-drop: read every input tile, drop the tiles of `-1` rows in the writer (tt-metal's scheme per spec GUIDANCE) | rejected — superseded by R-filter | — | same | above minimum: skipped rows' tiles cross DRAM once for nothing; and a dropped entry is drained by no NoC write, so the writer's tile-ring side must be explicit-sync (wait/barrier/pop) | — |
| R-row: whole tile rows dealt to cores | rejected — superseded by R-filter (tile-granular) | — | `rows_per_core x D_t` | same bytes; strands the one-tile-row INPUTS (`[1,1,32,4096]`, `[1,1,32,8192]`: 1 row → 1 cluster of 32) — the `width_split` catalog failure | — |
| R-seqblock: whole sequence blocks (`k`) dealt to cores | rejected — superseded by R-filter | — | `bs_t x H x D_t` | same bytes; occupancy capped by `ceil(seq/block_size)` (Qwen: 2 clusters) | — |
| R-mcast-pt: page-table row read once by one cluster and multicast | rejected — superseded by the per-core read in R-filter | — | — | saves ≤ 63 reads of 64 B (≤ 4 KB) at the price of a semaphore round trip on every engaged cluster before its first tile, i.e. adds latency to the critical path it would shorten; a stepping stone that would never be built on | — |
| R-host-pt: read the page table back on the host and pass destinations as RT args | rejected — superseded by the device-side lookup | — | — | one host↔device round trip per call, a host sync, and not replayable from a captured trace with a new page table (spec GUIDANCE) | — |
| R-L1-input: input interleaved in L1 (TARGET `memory_config: L1`) | deferred — knob-turn of R-filter, not a scheme: PHASE0 pins DRAM | `memory_axes(input) == {L1, interleaved}` | same | reads cross the NoC from L1 banks instead of DRAM; `TensorAccessor` resolves both, kernels unchanged; only `validate()`/SUPPORTED change | same |

Single-built-regime proof obligation: R-filter covers the whole TARGET domain; no other regime applies
because no axis is dependent (no combine regime exists), no operand is reused except the tiny page-table
row (multicast rejected above), and no residency question arises (no tile is read twice). Minimum vs.
actual: stated in the row; the only excess is the page-table row.

### Traffic ranking

| Rank | Candidate split | DRAM crossings | Cross-core traffic | Occupancy (accounted separately, §4) | Verdict |
|------|-----------------|----------------|--------------------|--------------------------------------|---------|
| 1 | R-filter (tile-granular, filter at source) | input mapped tiles x1 read, cache x1 write, page-table row 2 x engaged clusters | none | `min(32, tensor_input_tiles)` clusters | **chosen** — cheapest bytes and full grid for every INPUTS entry with ≥ 32 tiles |
| 2 | R-row / R-seqblock | same as 1 | none | worse (1 or 2 clusters on several INPUTS) | rejected on occupancy at equal bytes (tiebreak) |
| 3 | R-mcast-pt | page-table row x1 instead of x2·clusters | one multicast of ≤ 64 B + semaphore handshake per cluster | same as 1 | rejected: bytes saved are negligible, the handshake adds latency |
| 4 | R-drop | input *all* tiles x1 read (+ skipped rows), cache x1 write | none | same as 1 | rejected: strictly more DRAM reads when any entry is `-1` |
| — | split a dependent axis + combine | n/a: the op has no dependent axis | — | — | not a candidate |

Cheapest-traffic split = implemented split (R-filter).

**Operand-reuse check** (every operand vs. the chosen split along `input_tile`): `input_tensor` varies
along it (each tile read once by its owner); `cache_tensor` is written only, never read; `page_table`
row does **not** vary along it → reuse-shared by construction → per-core read (built), broadcast
(rejected above, with reason).

**Stall-shadow check:** the reader blocks once per call on the page-table row read (one DRAM round trip)
before its first tile read, and the writer blocks on the row's arrival. What could run in that window:
the input tile reads — but under R-filter *whether* a tile is read depends on the row (filtering), and
every tile ring entry must be drained by a NoC write for implicit sync, so a speculative read of a
possibly-skipped tile cannot be issued. Nothing in the reader is independent of the wait; the writer has
nothing to write before both the row and a tile arrive. The stall is one DRAM latency per call per
cluster, recorded as a perf lamp (speculative first window) rather than reordered. No reorder changes
any arithmetic (there is none), so no precision gate applies.

### Block schedule

```cpp
// Per cluster. One reader thread, one writer thread, no compute kernel.
load_page_table_row();                       // reader: reserve 2 entries of dfb_page_table_row, read page
                                             //   `batch_idx` of page_table into both (user trid 0), 1 barrier,
                                             //   push entry 0 -> writer; entry 1 stays reserved (reader's copy)
for (uint32_t block_idx = 0; block_idx < num_blocks_this_core; ++block_idx) {   // = 1 at Phase 0
    stream_mapped_tiles_in(block_idx);       // reader: walk [core_start_tile, +core_tiles) in input order,
                                             //   skip whole tile rows whose entry is -1, one implicit-sync
                                             //   read per mapped tile into dfb_input_tiles
    scatter_tiles_to_cache(block_idx);       // writer: wait for the row (once), same walk with the same skip,
                                             //   one implicit-sync write per tile to its cache page id
}
release_page_table_row();                    // reader: push entry 1; writer: pop entry 0, wait+pop entry 1
hand_back_noc_tags();                        // reader: noc_async_read_set_trid(0,...); writer: noc_async_write_set_trid(0,...)
```

| Block operation | Block shape | Resident across it | Intended frequency of fixed costs |
|-----------------|-------------|--------------------|-----------------------------------|
| `load_page_table_row` | 1 page (the row), 2 copies | — | once per cluster per call: 1 reservation, 2 reads, 1 barrier (trid 0), 1 push |
| `stream_mapped_tiles_in` | `core_tiles` input tiles (1-D range of the folded axis) | reader's row copy (entry 1, reserved) | page-table lookup once per tile row entered (not per tile); one NoC read per mapped tile (the interleaved page is the transfer unit — necessary traversal inside the block op); credit posts once per half-ring (4 tiles) by the DM0 ISR; no barrier, no push |
| `scatter_tiles_to_cache` | same range, same skip | writer's row copy (entry 0, waited once) | destination row base recomputed once per tile row; one NoC write per tile; credit acks once per half-ring by the DM0 ISR; no barrier, no pop |
| `release_page_table_row` | 2 entries | — | once per cluster per call |
| `hand_back_noc_tags` | — | — | once per kernel, last statement |

The walk (both kernels, identical): decompose `core_start_tile` once into `(h, t, d)` (one division
chain per cluster), then advance `d → t → h` with counters; on entering a tile row read
`block = row[t / bs_t]` (`t / bs_t` and `t % bs_t` also maintained as counters); if `block == -1` advance
`d` to the end of the row (or the end of the range) without a transfer; else the row's destination base
is `((block*H + h)*bs_t + t % bs_t) * D_t` and each tile goes to `base + d`.

### Perf lamps

| Lamp | Why the default may be wrong here | Nearby alternative to measure |
|------|-----------------------------------|-------------------------------|
| Grid synchronization | Most INPUTS give 1 tile per cluster (Qwen: 16 tiles on 16 clusters); per-cluster fixed cost (launch, DFB init, page-table round trip, exit drain) may exceed the tile's transfer | `split_unit_tiles` = 4 (engage `ceil(tensor_input_tiles/4)` clusters) vs. 1 — count: clusters engaged and tiles per cluster |
| Overlap (ring depth) | With `TILE_RING_ENTRIES` = 8 the writer starts after 4 tiles land (or at reader exit) | `TILE_RING_ENTRIES` ∈ {2, 8, 16} — count: transactions in flight per txn-id batch |
| Page-table stall (stall-shadow) | One DRAM round trip per cluster precedes the first tile read | read-and-drop for the first ring window only (issue the first `TILE_RING_ENTRIES` tile reads before the row resolves; needs an explicit-sync writer side) — measure only on a timed backend |
| Thread plan | DM-bound copy: one reader + one writer hart with implicit sync | (2, 0, 2): 2 reader + 2 writer harts on STRIDED 2→2 rings (each pair walks the mapped tiles of its parity; 2 page-table entries per reader thread) — on `ttsim-perf` |

## Dataflow Strategy

| Stage | Format | Mechanism | Notes |
|-------|--------|-----------|-------|
| page table row → reader's L1 | int32 row, `pt_page_bytes`, 2 copies in `dfb_page_table_row` | `noc.async_read<NocOptions::TXN_ID>(pt_accessor, dfb_page_table_row, pt_page_bytes, {.page_id = batch_idx}, {.offset_bytes = 0 / pt_entry_bytes}, {.trid = 0})` (`noc.h:252`), `async_read_barrier<TXN_ID>({.trid = 0})` (`noc.h:739`) | explicit sync (reserve_back(2) / push_back(1) … push_back(1)); ring listed in the reader's and writer's `disable_dfb_implicit_sync_for` |
| reader → writer (row) | entry 0 | `wait_front(1)` at start; `get_read_ptr()` (`dataflow_buffer.h:245,408`) | writer holds entry 0 for the whole stream |
| input DRAM → `dfb_input_tiles` | TILE, input dtype, 1 tile per entry | implicit sync: `noc.async_read<NocOptions::TXN_ID>(input_accessor, dfb_input_tiles, {.page_id = i}, {})` (`noc.h:923`) | NoC0 (reader config default); DM0 ISR posts credits per half-ring; skipped rows: no read |
| `dfb_input_tiles` → cache DRAM | TILE, unchanged bytes | implicit sync: `noc.async_write<NocOptions::TXN_ID>(dfb_input_tiles, cache_accessor, {}, {.page_id = dest})` (`noc.h:938`) | NoC1 (writer config default); whole-tile writes (`tile_reorder` catalog: never sub-tile faces) |
| return | — | `generic_op([input, page_table, cache], spec, run_args, tensor_indices)`; return `cache_tensor` | in place (R3); nothing allocated |

**Why the row goes reader → writer and is read twice:** both kernels need the row for their whole stream
(the reader to filter, the writer to address). A DM kernel cannot own a self-bound scratch ring on
Quasar (Unavailable row (3)), and a ring the writer produced for the reader would deadlock (the reader
would wait for the writer's push at its end, while the writer waits for the reader's last partial tile
batch, which DM0 credits only after the reader returns — `DataflowBuffer.md` C6). So the reader, the
one kernel that waits on nothing at its end, produces both copies: one it pushes at the start, one it
keeps reserved (producer-owned L1) and pushes as its last ring action. The writer pops both at its end.
Balanced: 2 pushed, 2 popped.

**Placement axes (TARGET) as blocking regimes:** `memory_config: DRAM` (built), `L1` interleaved —
knob-turn (the accessor resolves either; no kernel change, R-L1-input row). `sharding`: TARGET has only
`interleaved`; a sharded input would cut the independent `input_tile` axis, so its block is the shard,
consumed in place through a ring backed on the shard (`ttnn.cb_descriptor_from_sharded_tensor`
analogue), never re-read over the NoC — a knob-turn on the reader side should TARGET ever grow it.
`dtype: float32` — knob-turn (entry size from `buffer_page_size()`; ring 8 x 4096 B; `validate()` only).

## Work Distribution

| Field | Value |
|-------|-------|
| Work unit | 1 input tile (`split_unit_tiles` = 1, host constant); a cluster's block = its whole contiguous range |
| Grid | `device.compute_with_storage_grid_size()` (8x4 = 32 on quasar); `ttnn.split_work_to_cores(grid, tensor_input_tiles, row_wise=True)` (`noc_placement` catalog: row-wise spreads along the DRAM-facing axis) |
| Per-core work | group 1: `ceil` share, group 2: `floor` share; `core_start_tile` = running prefix over the groups in core order (as `toy_spec_single_tensor._split_tiles`) |
| Remainder | handled by the two groups of `split_work_to_cores`; tile counts use `ceil(seq/32)`, `ceil(D/32)`, `ceil(block_size/32)`; a cluster's range may start and end mid tile row (the walk handles both) |
| Bound | **data movement**: zero compute per tile (a bit-exact copy, R1) against two 2–4 KB NoC transfers per tile |
| Thread plan | **(R, C, W) = (1, 0, 1)**, host constant `THREAD_PLAN = (1, 0, 1)`; no compute kernel (nothing to compute, as Quasar's own tile `slice`: `slice_program_factory_tile.cpp:198`). R and W are passed as compile-time args `reader_threads` / `writer_threads`; the kernels stride by them |
| Unit dealt to compute threads | n/a (C = 0). DM→DM ring `dfb_input_tiles` is STRIDED 1→1: the unit is the ring entry (1 tile); no padding |
| Tests the plan needs | (a) a test that reads the plan off the host program (`ProgramSpec` kernels: reader `num_threads == 1`, writer `num_threads == 1`, no compute kernel) — with one thread per kind a correct result over a multi-cluster split shows every thread ran; (b) one device session: run another program, then `paged_fill_cache`, then `ttnn.to_layout(cache, ttnn.ROW_MAJOR_LAYOUT)` (tt-metal's own, untagged transfers) and check it (target page *Ring protocols*; backend *On Quasar*) |

Worked Phase 0 split on quasar (32 clusters):

| INPUTS entry (cache / input) | H x seq_t x D_t = tiles | clusters engaged | tiles per cluster |
|-----|-----|-----|-----|
| (8,2,32,128) / (1,2,64,128) | 2x2x4 = 16 | 16 | 1 |
| (128,8,32,64) / (1,8,512,64) | 8x16x2 = 256 | 32 | 8 |
| (4,1,32,32) / (1,1,32,32) | 1 | 1 | 1 |
| (8,2,32,64) / (1,2,96,64) | 12 | 12 | 1 |
| (8,4,64,64) / (1,4,128,64) | 32 | 32 | 1 |
| (6,3,64,96) / (1,3,96,96) | 27 | 27 | 1 |
| (4,8,128,64) / (1,8,32,64) | 16 | 16 | 1 |
| (16,1,32,256) / (1,1,160,256) | 40 | 32 | 8 x 2, 24 x 1 |
| (12,2,96,32) / (1,2,288,32) | 18 | 18 | 1 |
| (8,16,32,64) / (1,16,64,64) | 64 | 32 | 2 |
| (2,1,32,4096) / (1,1,32,4096) | 128 | 32 | 4 |
| (2,1,32,8192) / (1,1,32,8192) | 256 | 32 | 8 |

Single regime: no selection predicate and no regime-pinned tests are needed.

## Circular Buffers

| Semantic Name | Index | Page Size | Num Pages | Sizing rationale | Format | Producer | Consumer | Sync | Lifetime |
|---------------|-------|-----------|-----------|------------------|--------|----------|----------|------|-----------|
| `dfb_input_tiles` | `"input_tiles"` (DFB unique_id) | `tile_bytes` = `input_tensor.buffer_page_size()` | `TILE_RING_ENTRIES` (8) — per compute thread: n/a (C = 0); STRIDED 1→1 | streams over `input_tile` through a fixed window; spans no axis (no tile is used twice) | input dtype (bfloat16 Phase 0) — raw payload, never through DEST | reader | writer | implicit / implicit | whole stream |
| `dfb_page_table_row` | `"page_table_row"` | `pt_entry_bytes` = `align_up(pt_page_bytes, ttnn.get_dram_alignment())` | `PT_RING_ENTRIES` = 2 | spans `page_table_entry` (whole row, both copies live for the whole stream); streams over nothing | int32 (`data_format_metadata = ttnn.int32`) — raw payload | reader | writer | explicit / explicit (listed in both kernels' `disable_dfb_implicit_sync_for`) | whole kernel lifetime |

## Block Operation Realization

| # | Block operation | Block shape | Helper? | Input CB (semantic name, pages, state) | Output CB (semantic name, pages) | CB state after |
|---|-----------------|-------------|---------|----------------------------------------|----------------------------------|----------------|
| 1 | `load_page_table_row` | 1 row x 2 copies | no (raw dataflow API; no kernel-lib helper reads a page into a ring) | page_table DRAM page `batch_idx` | `dfb_page_table_row`, 2 reserved | entry 0 pushed (writer-owned), entry 1 reserved (reader-owned) |
| 2 | `stream_mapped_tiles_in` | `core_tiles` | no (raw `Noc` implicit-sync read) | input DRAM pages `[core_start_tile, +core_tiles)` minus `-1` rows | `dfb_input_tiles`, 1 per mapped tile | credits posted by DM0 ISR |
| 3 | `scatter_tiles_to_cache` | `core_tiles` | no (raw `Noc` implicit-sync write) | `dfb_input_tiles`, 1 per mapped tile; `dfb_page_table_row` entry 0 (waited once) | cache DRAM pages | tile ring drained (ISR acks) |
| 4 | `release_page_table_row` | 2 entries | no | — | — | reader pushed entry 1; writer popped both: ring balanced (2/2) |
| 5 | `hand_back_noc_tags` | — | no | — | — | reader read-tag and writer write-tag = 0 |

## API Mapping

| Block operation | Type | Function | File:Line | Template Params / Args | Input CB | Output CB | Which params are block knobs |
|-----------------|------|----------|-----------|------------------------|----------|-----------|------------------------------|
| `load_page_table_row` | raw_api | `DataflowBuffer::reserve_back` / `push_back` | `tt_metal/hw/inc/api/dataflow/dataflow_buffer.h:243-244` | `reserve_back(2)`, `push_back(1)` | — | `dfb_page_table_row` | none (structural 2) |
| `load_page_table_row` | raw_api | `Noc::async_read` (explicit, sized) | `tt_metal/hw/inc/api/dataflow/noc.h:252` | `<NocOptions::TXN_ID>(pt_accessor, dfb_page_table_row, pt_page_bytes, {.page_id = batch_idx}, {.offset_bytes = k * pt_entry_bytes}, {.trid = 0})`, k = 0, 1 | page_table | `dfb_page_table_row` | — |
| `load_page_table_row` | raw_api | `Noc::async_read_barrier` | `noc.h:739` | `<NocOptions::TXN_ID>({.trid = 0})` | — | — | — |
| `stream_mapped_tiles_in` | raw_api | `Noc::async_read` (implicit-sync DFB overload) | `noc.h:923` | `<NocOptions::TXN_ID>(input_accessor, dfb_input_tiles, {.page_id = i}, {})` | input | `dfb_input_tiles` | range `core_start_tile`, `core_tiles` (RT) |
| `stream_mapped_tiles_in` | raw_api | `DataflowBuffer::get_write_ptr` + `addr_to_l1_ptr` | `dataflow_buffer.h:407`; `ttnn/cpp/ttnn/kernel_lib/l1_helpers.hpp:30` | reader's row copy = `get_write_ptr()` after the first `push_back(1)` (entry 1), read as `volatile tt_l1_ptr int32_t*` | `dfb_page_table_row` | — | — |
| `scatter_tiles_to_cache` | raw_api | `DataflowBuffer::wait_front` + `get_read_ptr` | `dataflow_buffer.h:245,408` | `wait_front(1)` once, entry 0 | `dfb_page_table_row` | — | — |
| `scatter_tiles_to_cache` | raw_api | `Noc::async_write` (implicit-sync DFB overload) | `noc.h:938` | `<NocOptions::TXN_ID>(dfb_input_tiles, cache_accessor, {}, {.page_id = dest})` | `dfb_input_tiles` | cache | — |
| `release_page_table_row` | raw_api | `push_back(1)` (reader); `pop_front(1)`, `wait_front(1)`, `pop_front(1)` (writer) | `dataflow_buffer.h:244-246` | — | `dfb_page_table_row` | — | — |
| `hand_back_noc_tags` | raw_api | `noc_async_read_set_trid(0, noc.get_noc_id())` / `noc_async_write_set_trid(0, noc.get_noc_id())` | as used in `.claude/references/quasar_op_examples/toy_thread_plan/kernels/reader_implicit.cpp` / `writer_implicit.cpp` (last statement) | — | — | — | — |
| host | raw_api | `DataMovement2XXConfig(disable_dfb_implicit_sync_for=["page_table_row"])` on both DM kernels | `ttnn/cpp/ttnn-nanobind/program_spec.cpp:439-447`; usage `toy_thread_plan_program_artifacts.py` `_dm_config` | — | — | — | — |
| host | raw_api | `ttnn.split_work_to_cores(grid, n, row_wise=True)`, `buffer_page_size()` | `ttnn/cpp/ttnn-nanobind/pytensor.cpp:1582` | — | — | — | `split_unit_tiles` |

**Helpers considered and rejected** (applies to every `raw_api` row; there is no compute phase, so no
compute helper and no `compute_kernel_hw_startup` applies):

| Candidate | File:Line | Why it cannot express this |
|-----------|-----------|----------------------------|
| `tilize`/`untilize`/`reduce`/`eltwise`/`matmul_block` compute helpers | `ttnn/cpp/ttnn/kernel_lib/*_helpers.hpp` | the op has no compute phase (R1: every element moved unchanged; no compute kernel exists) |
| `local_copy_helpers_dataflow` (`set_read_state` / `read_with_state`) | `ttnn/cpp/ttnn/kernel_lib/local_copy_helpers_dataflow.hpp:57-95` | self-aimed L1→L1 reads (`local_addr`, `:57`); this op's transfers are DRAM↔L1 through a `TensorAccessor` with per-tile page ids |
| `dfb_helpers_dataflow` | `ttnn/cpp/ttnn/kernel_lib/dfb_helpers_dataflow.hpp:14-18` | only tile-dimension queries (`get_tile_r_dim`/`get_tile_c_dim`); no transfer helper |
| `tile_column_dataflow`, `l1_helpers::prepare_zero_tile` | `tile_column_dataflow.hpp:70,92`; `l1_helpers.hpp:67` | tile-content writers; the op never edits a tile (`addr_to_l1_ptr`, `l1_helpers.hpp:30`, **is** used for the row lookup) |
| `mcast` (`McastArgs`, `ttnn.Mcast`) | `ttnn/cpp/ttnn/kernel_lib/mcast/` | the only reuse-shared operand is ≤ 64 B; its broadcast regime is rejected (R-mcast-pt) |

## Key Risks and Gotchas

| Risk | Why it bites here | Mitigation in this design |
|------|-------------------|---------------------------|
| Reader and writer walks diverge | Both kernels must skip exactly the same tile rows and visit tiles in the same order, or the writer writes tile *k* to tile *j*'s destination | one shared kernel-side header (e.g. `kernels/paged_fill_cache_walk.hpp`) holding the `(h, t, d)` decomposition and the row-skip rule, included by both; the two row copies are reads of the same DRAM page in the same program, so their values agree |
| Deadlock on the page-table ring | The writer needs the row before any write; the reader may block on a full tile ring | the reader pushes entry 0 **before** its first tile read and pushes entry 1 before it returns; the writer waits for entry 1 only after all its writes (see Dataflow Strategy) |
| Implicit sync on a ring with a skipped entry | an implicit consumer acks only through writes; an entry that is not written would never be acked | filtering at the source: every entry pushed into `dfb_input_tiles` is drained by exactly one write |
| Sticky NoC transaction tag | a later tt-metal program (`ttnn.to_layout`) issues untagged transfers | `hand_back_noc_tags` as each kernel's last statement (reader: read tag; writer: write tag); the one-session test of Work Distribution covers it |
| Explicit-sync transfers untagged | the reader's page-table reads use explicit sync; an untagged read would reuse a ring's txn id | every explicit transfer carries `{.trid = 0}` and its barrier uses the same id |
| Partial last tile batch | a cluster's mapped tile count is unknown on the host (depends on `-1` entries) and rarely a multiple of the half-ring | implicit sync needs no nominal counts: DM0 credits the last partial batch when the reader returns (`DataflowBuffer.md` C6); a cluster with 0 mapped tiles pushes nothing |
| `-1` comparison | the row is int32; `-1` is `0xFFFFFFFF` | read entries as `int32_t` and compare to `-1` (`SKIP`) |
| Not in place | returning a new tensor or copying the cache fails every cell's R3 check | no output allocation; `cache_tensor` is the last io tensor and is returned |
| Shape constants as compile-time args | every INPUTS shape would compile a new kernel pair — slow on the simulator | all shape quantities (`H`, `seq_t`, `bs_t`, `D_t`, range, `batch_idx`, `pt_page_bytes`) are runtime args; compile-time args only the plan counts |
| A cluster starting mid tile row | 4096/8192-wide inputs split one tile row over 32 clusters | the walk starts at `d = core_start_tile % D_t` and looks the row's entry up on entry |

## Knob summary (single sources)

| Knob | Kind | Phase 0 | Source |
|------|------|---------|--------|
| `split_unit_tiles` | core-assignment granularity | 1 tile | host constant in the program-artifacts module |
| `core_tiles` (block extent of `input_tile`) | block factor | whole per-core range | `split_work_to_cores` → RT arg |
| `TILE_RING_ENTRIES` | buffer depth | 8 (even) | host constant |
| `PT_RING_ENTRIES` | structural | 2 | host constant → CT arg `page_table_row_copies` (the kernels' reserve / push / pop counts on the row ring; verifier, 2026-11-21) |
| `THREAD_PLAN` | thread plan | (1, 0, 1) | host constant → CT args `reader_threads`, `writer_threads` |
| grid | core count | queried | `device.compute_with_storage_grid_size()` |
