# Operation Design: g6_indexed_fill

## Overview

| Field | Value |
|-------|-------|
| Classification | data_movement (index-driven page selection; no compute engine, no compute kernel) |
| Goal | Return a new uint32 ROW_MAJOR DRAM-interleaved tensor equal to `input_tensor` except at the batches `batch_ids` names, which take `value_tensor`'s batches in order. The positions are read **on the device, in every call, unsigned**; an entry `>= B` writes nothing. One device program, one data-movement kernel per core. |
| Math | `y[b,r,w] = s[i,r,w]` if `b == ids[i] < B` for some `i` (the **largest** such `i`), else `y[b,r,w] = x[b,r,w]` |
| References | `docs/explorations/g6_indexed_fill/{DOSSIER,math,tensix,risks,lessons}.md`; `eval/golden_tests/g6_indexed_fill/{feature_spec,helpers}.py`; `.claude/references/{blocking-model,l1-footprint-discipline,op-design-template}.md`; catalog `ttnn/ttnn/operations/examples/master.md` (entries listed under Perf lamps); `l1_ledger.md` (this directory) |

Symbols used throughout:

| Symbol | Meaning | Production |
|--------|---------|------------|
| `B, d1, d2, W` | extents of `input_tensor` | 7, 1, 1, 32 |
| `K` | length of `batch_ids` (= dim 0 of `value_tensor`) | 1 |
| `R = d1*d2` | rows (pages) per batch | 1 |
| `P = B*R` | pages of `input_tensor` and of the result | 7 |
| `Q = K*R` | pages of `value_tensor` | 1 |
| `page_bytes = 4*W` | logical bytes of one ROW_MAJOR page | 128 |
| `A` | `ttnn.get_dram_alignment()` (DRAM read alignment, 64 B on Blackhole, `noc_parameters.h:379`) | 64 |

## Parameters

| Name | Type | Required | Valid Range | Default | CT/RT |
|------|------|----------|-------------|---------|-------|
| `batch_ids` | `ttnn.Tensor` (positional 1) | yes | rank 1 `[K]`, `K >= 1`, uint32, ROW_MAJOR, DRAM interleaved; **values unrestricted** (any uint32) | — | buffer address only (RT); values are read by the kernel, never by the host |
| `input_tensor` | `ttnn.Tensor` (positional 2) | yes | rank 4 `[B,d1,d2,W]`, all extents `>= 1`, uint32, ROW_MAJOR, DRAM interleaved | — | buffer address (RT); shape → plan (CT) |
| `value_tensor` | `ttnn.Tensor` (positional 3) | yes | rank 4 `[K,d1,d2,W]` (d1, d2, W those of `input_tensor`), uint32, ROW_MAJOR, DRAM interleaved | — | buffer address (RT) |
| `dim` | `int`, keyword-only | no | `0` only (no canonicalization: `-4` is refused like any value `!= 0`) | `0` | host only; never reaches the program |

## Tensors

### Input

| Property | `batch_ids` | `input_tensor` | `value_tensor` |
|----------|-------------|----------------|----------------|
| Shape | `[K]` | `[B, d1, d2, W]` | `[K, d1, d2, W]` |
| Dtype | uint32 | uint32 | uint32 |
| Layout | RM — one page of `4K` bytes | RM — `P` pages of `page_bytes` | RM — `Q` pages of `page_bytes` |
| Memory | DRAM interleaved | DRAM interleaved | DRAM interleaved |

### Output

| Property | Value |
|----------|-------|
| Shape | `[B, d1, d2, W]` (= `input_tensor.shape`) |
| Dtype | uint32 |
| Layout | ROW_MAJOR (`P` pages; no padding, every element logical) |
| Memory | DRAM interleaved, a **new** buffer allocated by the op every eager call (once at trace capture) |

## Blocking Model

### Axes

The op has five logical axes. `b`, `r` and `i` fold into one flattened **page** axis because pages
of `x`/`y` are contiguous in `(b, r)`, pages of `s` are contiguous in `(i, r)`, and every page is
moved identically (blocking-model §1, honest flattening). The axis that is actually split across
cores is **`u`**, the work-unit axis:

```
page_idx in [0, P + Q):   page_idx <  P  -> x page p = page_idx        (x unit, b = p / R)
                          page_idx >= P  -> value page q = page_idx - P (value unit, i = q / R, r = q % R)
unit = page_idx * row_chunks_per_page + c,   c in [0, row_chunks_per_page)   (c = row chunk, the w axis)
```

| Axis | Character (+ one-clause reason) | Extent knob | Phase 0 value | Knob source | Core-assignment | Buffer depth | Why | Later unlock |
|------|--------------------------------|-------------|---------------|-------------|-----------------|--------------|-----|--------------|
| `b` (dim 0 of x / y, extent B) | independent — each `y[b]` comes whole from one batch of x or s; no result spans `b` (math.md axes table) | folded into `u` → `block_units` | whole per-core assignment | `plan.block_units` (host `derive_plan`) | contiguous unit ranges over the full grid (`u` row) | `staging_depth` (on `u`) | the selection per batch needs only the ids values, so batches are freely assignable | knob-turn |
| `r` (rows of a batch, `R = d1*d2`) | independent — rows of one batch never combine | folded with `b` (`p = b*R + r`) and with `i` (`q = i*R + r`) → `block_units` | whole per-core assignment | `plan.block_units` | as `u` | as `u` | a batch's rows may straddle two cores; each row decides alone from the same ids | knob-turn |
| `i` (dim 0 of value_tensor, extent K) | independent for the **moved value pages** — entry `i`'s rows go to one destination batch | folded into `u` as value pages `P + q` → `block_units` | whole per-core assignment | `plan.block_units` | value units are the tail of `u`; split with it | as `u` | the moved payload of entry `i` is owned by one unit range; its destination comes from the ids | knob-turn |
| `w` (W elements of a row) | independent — elements never combine; a page is the natural transfer unit | `row_chunk_bytes` | whole page (128 B): page `< min_row_chunk_bytes`, a split would only fragment one transfer | `plan.row_chunk_bytes` | spread only when pages under-fill the grid: `row_chunks_per_page = max(1, min(cores_cap // (P+Q), page_bytes // min_row_chunk_bytes))`, clamped by `max_row_chunk_bytes` | n/a (part of the slot size) | the width split of catalog `width_split`, applied only when it fills idle cores and never below the granularity floor | knob-turn |
| `j` (entries of batch_ids, **scanned** by the decision) | **reuse-shared** — every unit's decision reads all K entries (x unit: "is my batch named?"; value unit: "is my entry the last with its value?"); the split of `u` re-reads them on every core by construction | `ids_chunk_entries` | `K` (= 1): the whole vector in one window | `plan.ids_chunk_entries` | **not split**: every work core reads every entry (replicated read; multicast is the `deferred` row `ids_mcast`) | 1 (`cb_batch_ids`) | the decision depends on the whole vector; at `K*4 <= 64 B` a replicated read costs one DRAM granule per core, issued in parallel with the page loads | scheme-change (`ids_mcast`) |
| `u` (flattened work axis = the split) | independent (union of the rows above) | `block_units` | `max_units_per_core` (= 1 in production) | `plan.block_units` | `split_work_to_cores(core_set, units, row_wise=True)` over `num_cores_to_corerangeset(cores_cap, grid, row_wise=True)`; `cores_cap` = full grid in Phase 0 | `staging_depth` | spread over the full grid first, then the coarsest block that fits (l1-footprint Rule 2) | knob-turn (`max_work_cores`, `block_units`, `staging_depth`) |

**Stage data.** The only intermediate the scheme creates is the per-unit **decision** (destination
page or SKIP) of `select_block`. It has the axis `u` and is computed by the core that owns the unit,
for its own units only. No core collects, coordinates or hands out work; no stage runs on one core
while others wait.

**Every knob is a parameter.** Each value above is produced once by the host function
`derive_plan` (algorithm under Work Distribution) from the shapes, the device grid and the dict
`PLAN_CAPS`; every CB size, compile-time argument and per-core runtime argument is derived from that
plan. No kernel contains a literal standing for B, K, R, W, a block, a chunk or a core count.

### Buffer-depth knobs

| CB | Depth knob | Phase 0 value | What the depth buys |
|----|------------|---------------|---------------------|
| `cb_staged_pages` | `staging_depth` | 1 when `max_blocks_per_core == 1` (production); `PLAN_CAPS["max_staging_depth"]` (2) otherwise | at 2, block `k+1`'s loads land while block `k`'s writes drain; at one block per core there is nothing to overlap and a second slot would be dead L1 |
| `cb_batch_ids` | `ids_depth` (fixed 1) | 1 | `ids_resident`: one window, read once per kernel — nothing to overlap. `ids_streamed`: a second window would prefetch chunk `c+1` while chunk `c` is scanned (perf lamp) |
| `cb_block_dest` | fixed 1 | 1 | the table is written by `select_block(k)` and fully read by `store_block(k)`'s issue loop on the same RISC before `select_block(k+1)` starts — one RISC cannot overlap them |

### Mechanism caps

| Mechanism | Cap on which extent | Clamp | What happens unclamped |
|-----------|--------------------|-------|------------------------|
| DRAM read address alignment `A` (`NOC_DRAM_READ_ALIGNMENT_BYTES`, `tt_metal/hw/inc/internal/tt-1xx/blackhole/noc/noc_parameters.h:379`) | row-chunk offsets, ids window start | `row_chunk_bytes` is a multiple of `A`; an ids window covering entries `[j0, j1)` is read as `[align_down(4*j0, A), align_up(4*j1, A))` and the entries are addressed at offset `4*j0 - window_start` | a misaligned DRAM read returns neighbouring bytes: a wrong index or a wrong row, no error (R6) |
| L1 slot / DRAM address congruence modulo `A` | every staging slot, the ids window, the dest table | every CB `total_size` and every slot stride is a multiple of `A`, so every slot starts `A`-aligned | the NoC copies into a misaligned slot; the payload shifts |
| Read size vs. page slot | the last row chunk of a page; the ids window | reads move `align_up(chunk_bytes, A)` (ends inside the page's aligned DRAM slot, `buffer_aligned_page_size()`); **writes move the logical chunk bytes** | an aligned-size write would touch the padding of the result's page slot; a logical-size read under the alignment may move a partial granule |
| Single NoC transfer `<= 1 MB` (`noc_async_write` doc, `dataflow_api.h:868`) | `row_chunk_bytes`, ids window | `max_row_chunk_bytes = 64 KiB`; window `<= 4*ids_chunk_cap_entries + A` = 64 KiB + 64 | size field overflow, truncated transfer |
| Unsigned 32-bit index (contract) | the ids values | read as `uint32_t`, compared `t < B` unsigned; destination page `t*R + r` formed **only after** `t < B` holds; no signed type anywhere on the path | `4294967295` read signed is `-1`: a write before the result into another tensor (R3) |
| RISC-V L1 data cache (Blackhole) | ids values read by the core | load ids through `volatile tt_l1_ptr uint32_t*` only after `noc_async_read_barrier`, which invalidates the cache (`dataflow_api.h:1921-1934`) | a stale index from the cache; passes a single call |
| Per-core L1 bound 1347776 B (BB-05) | total per-core footprint | caps keep the footprint `<= 1179712` B for every shape (ledger); host asserts `l1_bytes_per_core <= 1347776` | launch failure or overlap with the model's L1 tensors |
| 32-bit page / unit indices | `units = (P+Q)*row_chunks_per_page` | host asserts `units < 2**32` | wrapped page id writes the wrong page |

### Regimes

Minimum, named boundary **DRAM**: every input page crosses once, every result page crosses once.
Domain: every shape meeting the input requirements falls in exactly one built regime, selected by
`K` alone; production (`K = 1`) is `ids_resident`.

| Regime | Status | Predicate | Block | Data movement (vs. minimum) | What a bigger block buys |
|--------|--------|-----------|-------|-----------------------------|--------------------------|
| `ids_resident` (scatter, unconditional loads) | **built** — production: 8 units on 8 cores, 1 block each | `ceil(K / ids_chunk_entries) == 1` ⇔ `K <= ids_chunk_cap_entries` (16384) | `block_units × row_chunk_bytes` on `u × w`; the ids window (all K entries) resident for the core's whole run | x: once (all `P` pages loaded; the `<= min(K,B)*R` named pages are loaded and dropped); s: once (`Q` pages); y: once (`P` writes, each page exactly once); ids: once **per work core** (`num_cores` aligned windows). Above the minimum by `num_cores - 1` ids windows and `<= K*R` dropped pages — the price of issuing every load in the ids read's shadow (2 serial DRAM round trips instead of 3) | per block: one read barrier (an exposed DRAM round trip at depth 1), one decision pass over the K resident entries, one `writes_flushed` before slot reuse. Intended frequency: once per block; production: once per call |
| `ids_streamed` (scatter, ids re-streamed) | **built** — required for correctness at every K within the L1 bound (rule 9); same kernel, compile-time flag | `K > ids_chunk_cap_entries` | as `ids_resident`; the ids are read in windows of `ids_chunk_entries` inside `select_block`, once per block | ids: `num_cores × max_blocks_per_core × 4K` bytes — above the minimum by that factor; x, s, y as `ids_resident` | every block re-streams the whole ids vector: a bigger block divides the ids traffic directly, plus the `ids_resident` terms |
| `ids_mcast` | **deferred** — at every `ids_resident` shape the window is one 64-B granule per core, read in parallel with that core's page loads; a multicast puts injector read → multicast → semaphore wait in series on every receiver's critical path to save `(num_cores-1)` granules (7 × 64 B in production). It pays where the replicated ids bytes are large: `ids_streamed` with many cores and blocks. Reachable: `cb_batch_ids` has the same address and layout on every work core, the work cores form `<= 2` rectangles, and the contract in Dataflow Strategy replaces only the DRAM read inside `load_ids_window` | as the built regime | as the built regime | ids: 1 DRAM read per call (minimum) + `(num_cores-1)` multicast window payloads + 1 semaphore round per window | as the built regime, minus the per-core ids read |
| `gather_after_index` (each core owns result pages, loads the chosen source after the ids land) | **deferred** — this is the cheapest-traffic split: it loads exactly `P` source pages instead of `P + Q`, saving `Q` page reads (128 B in production). Not built because it puts a full DRAM round trip in series on every core (ids → chosen source → write: 3 serial round trips instead of 2) for 128 B in a latency-bound op (tensix.md: ~16 transfers, no bandwidth term). It pays only where `Q` pages are a large share of a bandwidth-bound move — no INPUTS shape. Reachable: same kernel and CBs; `load_block` moves after `select_block` (which already produces the per-unit destination) and value units are re-keyed by destination | `K*R` pages a large share of `P` and the move bandwidth-bound | result pages per core | minimum for x and s; ids as `ids_resident` | as the built regime |
| `gather_speculative` (each core owns result pages, loads its x pages and **all** value pages up front) | **rejected** — superseded by `ids_resident`, which has the same 2-round-trip critical path and loads value pages once; this loads all `Q` value pages on every work core (`Q × num_cores`) because a result page's owner cannot know which entry names it before the ids land | — | result pages per core | s: `num_cores` crossings | — |
| `copy_then_overwrite` (math.md F2) | **rejected** — superseded by `ids_resident`. Writes `K*R` pages twice; on one core the overwrite waits a write barrier after the copy (a third serial round trip), across cores the two writes to one address race (R4, BB-06). The built regime writes every result page exactly once | — | — | y: `P + K*R` writes | — |
| `dependent_split_combine` | **rejected** — no axis is dependent (math.md "Axes and dependence": no result spans `b`, `r`, `w` or `i`), so there is no partial to combine; the only coupling between units (named-batch skip, repeated entries) is resolved locally from the replicated ids | — | — | — | — |

Regime selection function (host, shape-only, grid-independent):
`regime = "ids_resident" if ceil(K / ids_chunk_entries) == 1 else "ids_streamed"`, passed to the
kernel as the compile-time flag `ids_resident`. Regime-pinned tests force `ids_streamed` on the
rule-9 shape through `PLAN_CAPS["ids_chunk_cap_entries"] = 1` (the aligned-window read makes a
1-entry window legal), so both regimes run without adding a shape (BB-22).

### Traffic ranking

Production figures (`[1]`, `[7,1,1,32]`, `[1,1,1,32]`); bytes on the DRAM wire (an ids window is
one 64-B granule). Fan-out is per call.

| # | Candidate split | DRAM reads | DRAM writes | Cross-core | Serial DRAM round trips per core | Verdict |
|---|-----------------|-----------|-------------|------------|----------------------------------|---------|
| A | `u` = x pages ++ value pages over cores, loads unconditional, ids replicated | ids 8×64, x 896, s 128 = 1536 | 896 | 0 | 2 (loads ∥ ids → store) | **built** (`ids_resident` / `ids_streamed`) |
| B | A + ids read once and multicast | ids 64, x 896, s 128 = 1088 | 896 | 7 × 64 B multicast + semaphore | 3 (ids → multicast → store) | `deferred` (`ids_mcast`) |
| C | result pages over cores, source chosen after the ids land | ids 7×64, x 768, s 128 = 1344 | 896 | 0 | 3 | `deferred` (`gather_after_index`) |
| C+B | C with the ids multicast | ids 64, x 768, s 128 = 960 | 896 | multicast + semaphore | 4 | cheapest bytes, longest chain — not a separate row: it is the two deferred rows combined |
| D | result pages over cores, every candidate loaded up front | ids 7×64, x 896, s 7×128 = 2240 | 896 | 0 | 2 | `rejected` |
| E | copy all, then overwrite the named batch | ids, x, s as A | 1024 | 0 (one core) / race (two) | 3 | `rejected` |
| F | `w` split (row chunks over cores) | as A, more transactions | as A | 0 | 2 | part of A (`row_chunks_per_page`); for 128-B pages nothing to split |
| G | split a dependent axis + combine | — | — | — | — | no dependent axis |

Decision. By DRAM bytes, C (and C+B) rank above A: A loads `Q` more pages and replicates the ids
window. A is chosen because the trade is bytes for a serial round trip, stated explicitly
(blocking-model §4): every core of C must wait for the ids before it can even issue its load, so
C's chain is ids → load → store, while A issues all loads in the ids read's shadow and its chain is
(ids ∥ loads) → store. The extra bytes are `Q` pages plus one granule per core — 128 + 448 B in
production, against an op whose cost is per-transfer latency, not bandwidth. Both cheaper rows are
`deferred` with their delta, and A's structure reaches both (same CBs, same decision table).

### Stall shadow

| Wait | Work independent of the wait | Placed into the shadow | Permission | Floating-point order |
|------|------------------------------|------------------------|------------|----------------------|
| ids window landing (every core, every call) | every source load of block 0 — the address of a unit's source depends only on the unit, never on the ids | `load_block(0)` issued in the same burst as `load_ids_window`; one read barrier covers both | selection commutes with loading: a load has no side effect, so loading a page whose write will be dropped is legal; the choice is applied at `store_block` | unchanged — no arithmetic |
| `store_block(k)` writes draining (multi-block shapes) | `load_block(k+1)` into the other slot | issued before `store_block(k)` when `staging_depth == 2` | disjoint L1 slots | unchanged |
| final write barrier (BB-03) | nothing: every write's destination depends on the ids, and the barrier is the kernel's last act | — | — | — |
| `ids_streamed`: window `c` read inside `select_block` | the scan of window `c-1` | not in Phase 0 (perf lamp `ids_depth`) | windows are independent reads | — |

### Block schedule

```cpp
load_ids_window(0);                  // ids_resident only: the core's single ids read (ids_streamed reads
                                     // every window inside select_block)
load_block(0);                       // same burst: one read barrier covers ids + block 0
for (uint32_t block_idx = 0; block_idx < num_blocks_this_core; ++block_idx) {
    wait_loads(block_idx);           // read barrier: block_idx's units (+ the ids window) have landed
    select_block(block_idx);         // destination per unit from all K entries (ids_streamed: windows read here)
    load_block(block_idx + 1);       // staging_depth 2: here; staging_depth 1: after store_block (slot reuse)
    store_block(block_idx);          // one write per kept unit: slot -> its destination page/chunk
}
drain_writes();                      // write barrier, BB-03
```

| Block operation | Block shape | Resident across it | Intended frequency of fixed costs |
|-----------------|-------------|--------------------|-----------------------------------|
| `load_ids_window` | one window of `ids_chunk_entries` entries (aligned superset) | the window: whole kernel (`ids_resident`) | 1 NoC read per window; no own barrier (shares `wait_loads(0)`); TensorAccessor built once per kernel |
| `load_block` | `block_units × row_chunk_bytes` | the block's slot until `store_block` has issued its writes and they are flushed | 1 NoC read per unit (one whole chunk, never a sub-chunk); no barrier inside; before reusing a slot: one `noc_async_writes_flushed` |
| `wait_loads` | — | — | 1 read barrier per block |
| `select_block` | `block_units` decisions | `cb_block_dest` until `store_block(block_idx)` has issued | 0 NoC transactions (`ids_resident`); `ids_chunks` window reads + barriers per block (`ids_streamed`); one pass over the K entries per block |
| `store_block` | `<= block_units` writes | — | 1 NoC write per kept unit; no barrier inside |
| `drain_writes` | — | — | 1 write barrier per kernel |

**`select_block` decision contract** (the only logic in the op):

| Unit kind | Destination |
|-----------|-------------|
| x unit (page `p < P`, chunk `c`), `b = p / R` | SKIP if any entry `j` has `ids[j] == b`; else page `p`, chunk `c` |
| value unit (page `P + q`, chunk `c`), `i = q / R`, `r = q % R`, `t = ids[i]` | SKIP if `t >= B` (unsigned) or some `j > i` has `ids[j] == t`; else page `t*R + r`, chunk `c` |

Invariant: across the whole grid every result page-chunk has **exactly one writer** — no address
is written twice, so no write needs ordering against another (BB-06), and repeated entries resolve
to the **largest** `i` (the oracle's and torch CPU's choice) deterministically. Worst-case work is
`O(block_units × K)` comparisons per block; the realization (e.g. a range check that marks the
named batches of the block in `O(K)`) is the implementer's.

### Perf lamps

| Lamp | Why the default may be wrong here | Nearby alternative to measure |
|------|-----------------------------------|-------------------------------|
| Grid synchronization: full grid → 8 cores × 1 unit | per-core launch cost (runtime-arg writes, start skew of 8 cores) may exceed the issue time it saves (one core would issue 9 reads + 7 writes in series) | `PLAN_CAPS["max_work_cores"]` ∈ {1, 2, 4, 8}; judge on unprofiled host time of blocking trace replays and the model step, not profiler kernel duration alone (lesson RB-053) |
| One DM kernel on RISCV_0 / NOC_0 | a reader (NCRISC/NoC0) + writer (BRISC/NoC1) split puts the ids read and the page read on two RISCs; it pays only if one RISC is issue-bound (catalog `split_reader`) | reader + writer kernels with a CB hand-off of the staged pages and the decision table |
| Placement: row `(0,0)–(7,0)` | catalog `noc_placement`: row/diagonal lines beat columns for spread DRAM traffic; the ids page and the result pages sit in fixed banks | another one-rectangle placement (e.g. the row nearest the DRAM column); NOC_1 for the writes |
| Overlap (non-production shapes) | the coarsest block (up to 1 MiB of staging) issues thousands of reads before one barrier; catalog `double_buffer` measured the sweet spot at ~4–8 transfers per barrier with depth 2 | `PLAN_CAPS["staging_budget_bytes"]` = `2 × 8 × row_chunk_bytes` |
| ids window depth in `ids_streamed` | windows are read and barriered serially inside `select_block` | `ids_depth = 2` (prefetch window `c+1` while scanning `c`) |

Catalog entries and lessons that set the knobs:

| Entry | Knob it informed |
|-------|------------------|
| `double_buffer` | issue a whole block of reads, then one barrier; `staging_depth = 2` only when a core has more than one block |
| `tile_reorder` | one transfer per whole page/chunk, barriers batched per block; never sub-page scatter |
| `width_split` | split `w` across cores only when the pages under-fill the grid |
| `noc_placement` | `row_wise=True` placement; NoC lamp |
| `trid_double_issue` | not used: the block is the whole core assignment ("the trivial one-block loop is within 3%") |
| `split_reader` | not used: 3 transactions per core are not issue-bound (lamp) |
| `bank_stagger` | not used: one unit per core in production; no common walk order to stagger |
| `mcast_topology`, `shared_input_reuse` | the `ids_mcast` deferred row (ids are constant along `u`) |
| lessons RB-036 / RB-053 | one rectangular core range per kernel group in production; measurement method for the core-count lamp |

## Dataflow Strategy

| Stage | Format | Mechanism | Notes |
|-------|--------|-----------|-------|
| batch_ids DRAM page → `cb_batch_ids` | raw uint32 bytes, one `A`-aligned window | `noc_async_read(ids_accessor.get_noc_addr(0, window_start), window_slot, window_bytes)` | every work core, every call, every replay; the values never reach the host, a compile-time or a runtime argument |
| x / s DRAM pages → `cb_staged_pages` | raw bytes, one row chunk per slot | `noc_async_read(accessor.get_noc_addr(page, c*row_chunk_bytes), slot, align_up(chunk_bytes, A))` | issued before the ids are known (stall shadow) |
| `cb_batch_ids` → RISC-V registers → `cb_block_dest` | uint32 scalars | `volatile tt_l1_ptr uint32_t*` loads after the read barrier | the only values a core interprets; page payloads never pass through a register, unpacker, DEST or packer — bit-exact for all 2^32 values (R5) |
| `cb_staged_pages` → result DRAM pages | raw bytes | `noc_async_write(slot, y_accessor.get_noc_addr(dest_page, c*row_chunk_bytes), chunk_bytes)` (non-posted, acked) | one write per kept unit; `noc_async_write_barrier` before the kernel returns (BB-03) |
| `ids_mcast` (deferred) contract | raw window | `ttnn.Mcast` with a fixed sender = the first work core over the work-core rectangle(s); `McastArgs` `sender(noc).send()` writes the window into every receiver's `cb_batch_ids` at the same address; receivers `receiver(noc).receive()` in place of the DRAM read barrier of `load_ids_window` | page loads stay unconditional and are still issued before the wait |

Placement axis: TARGET has no `memory_layout` axis; every tensor is DRAM interleaved, so there is
no physical shard to realize and no sharded value to classify as knob-turn or scheme-change.

## Work Distribution

| Field | Value |
|-------|-------|
| Work unit | one row chunk of one page: `unit = page_idx * row_chunks_per_page + c`; `page_idx < P` → x page, else value page `page_idx - P` |
| Grid | `cores_cap = grid.x*grid.y` (Phase 0) or `min(grid.x*grid.y, PLAN_CAPS["max_work_cores"])`; `grid = device.compute_with_storage_grid_size()`; core set `ttnn.num_cores_to_corerangeset(cores_cap, grid, True)` |
| Per-core work | `ttnn.split_work_to_cores(core_set, units, True)`: one contiguous range `[unit_begin, unit_begin + unit_count)` per core (runtime args); group 1 holds `ceil`, group 2 one unit fewer |
| Remainder | ragged group 2; the last block of a core has a shorter runtime extent (same block operation); the last chunk of a page is `page_bytes - (row_chunks_per_page-1)*row_chunk_bytes`; ROW_MAJOR — the page count is exactly `B*d1*d2`, no tile padding |
| Production | `units = 8` → 8 cores `(0,0)–(7,0)`, one CoreRange; cores 0–6 own x pages 0–6, core 7 owns the value page; 1 block, depth 1 |
| Regime | `ids_resident` iff `ceil(K / ids_chunk_entries) == 1`; shape-only, so no grid-dependent regime |

`derive_plan` — the single source of every knob (host Python; runs on a descriptor-cache miss and in
`describe_plan`):

```python
caps = PLAN_CAPS                        # read at call time, never copied at import
A = ttnn.get_dram_alignment()
page_bytes, P, Q = 4 * W, B * R, K * R
pages = P + Q
grid = device.compute_with_storage_grid_size()
cores_cap = grid.x * grid.y if caps["max_work_cores"] is None else min(grid.x * grid.y, caps["max_work_cores"])
n_split = max(1, min(cores_cap // pages, page_bytes // caps["min_row_chunk_bytes"]))
max_chunk = max(A, align_down(caps["max_row_chunk_bytes"], A))
row_chunk_bytes = min(align_up(ceil_div(page_bytes, n_split), A), max_chunk)
row_chunks_per_page = ceil_div(page_bytes, row_chunk_bytes)
units = pages * row_chunks_per_page
core_set = ttnn.num_cores_to_corerangeset(cores_cap, grid, True)
num_cores, all_cores, g1, g2, n1, n2 = ttnn.split_work_to_cores(core_set, units, True)
if n1 * row_chunk_bytes <= caps["staging_budget_bytes"]:
    staging_depth, block_units = 1, n1                      # one block: the whole core assignment
else:
    staging_depth = caps["max_staging_depth"]
    block_units = max(1, caps["staging_budget_bytes"] // (staging_depth * row_chunk_bytes))
max_blocks_per_core = ceil_div(n1, block_units)
ids_chunk_entries = min(K, caps["ids_chunk_cap_entries"])
ids_chunks = ceil_div(K, ids_chunk_entries)
regime = "ids_resident" if ids_chunks == 1 else "ids_streamed"
ids_window_bytes = align_up(4 * ids_chunk_entries, A) + (A if ids_chunks > 1 and (4 * ids_chunk_entries) % A else 0)
dest_bytes = align_up(4 * block_units, A)
l1_bytes_per_core = ids_window_bytes + staging_depth * block_units * row_chunk_bytes + dest_bytes
assert l1_bytes_per_core <= 1347776 and units < 2**32
```

`PLAN_CAPS` (module-level dict, exported by the package; defaults):

| Key | Default | Bounds what |
|-----|---------|-------------|
| `max_work_cores` | `None` (full grid) | core-assignment knob |
| `staging_budget_bytes` | `1 << 20` | `staging_depth × block_units × row_chunk_bytes` |
| `max_staging_depth` | `2` | `staging_depth` when a core has more than one block |
| `min_row_chunk_bytes` | `2048` | granularity floor of the `w` split |
| `max_row_chunk_bytes` | `65536` | `row_chunk_bytes` (one NoC transfer) |
| `ids_chunk_cap_entries` | `16384` | `ids_chunk_entries` (regime predicate) |

## Circular Buffers

All three CBs are L1 allocations of the **single** data-movement kernel (`dm`), which both fills and
drains them; there is no CB credit (no reserve/push/wait/pop — push count = wait count = 0) and
ordering comes from NoC barriers: read barrier before a slot or window is read, `writes_flushed`
before a slot is refilled. Base addresses via `get_write_ptr`. `UInt32` is a label only: no payload
passes through DEST.

| Semantic Name | Index | Page Size | Num Pages | Sizing rationale | Format | Producer | Consumer | Lifetime |
|---------------|-------|-----------|-----------|------------------|--------|----------|----------|-----------|
| `cb_batch_ids` | 0 | `ids_window_bytes` | 1 | spans `j` (`ids_chunk_entries` + alignment superset); streams `u`, `w` | UInt32 | dm (NoC read) | dm (scalar loads) | `ids_resident`: whole kernel; `ids_streamed`: one window inside `select_block` |
| `cb_staged_pages` | 1 | `row_chunk_bytes` | `staging_depth * block_units` | spans `u` (`block_units` slots × `staging_depth`) and `w` (`row_chunk_bytes`); streams `j` | UInt32 | dm (NoC read) | dm (NoC write source) | one block per slot set |
| `cb_block_dest` | 2 | `align_up(4*block_units, A)` | 1 | spans `u` (one destination word per unit); streams `w`, `j` | UInt32 | dm (`select_block` stores) | dm (`store_block` loads) | one block |

Production: 64 + 128 + 64 = 256 B per core. Full ledger, sharing decisions and data-movement
budget: `l1_ledger.md`.

## Block Operation Realization

| # | Block operation | Block shape | Helper? | Input CB (semantic name, pages, state) | Output CB (semantic name, pages) | CB state after |
|---|-----------------|-------------|---------|----------------------------------------|----------------------------------|----------------|
| 1 | `load_ids_window` | 1 window of `ids_chunk_entries` | no (raw NoC) | DRAM `batch_ids` page 0 | `cb_batch_ids`, 1 page (in flight) | window lands at `wait_loads(0)` (`ids_resident`) or at the window's own barrier (`ids_streamed`) |
| 2 | `load_block` | `block_units × row_chunk_bytes` | no (raw NoC) | DRAM x / s pages | `cb_staged_pages`, slot set `block_idx % staging_depth` (in flight) | in flight until `wait_loads(block_idx)` |
| 3 | `wait_loads` | — | no | — | — | slot set and (block 0) the window readable |
| 4 | `select_block` | `block_units` decisions | no (scalar RISC-V code) | `cb_batch_ids`, 1 page, landed | `cb_block_dest`, 1 page | `cb_batch_ids` kept (`ids_resident`) / overwritten by the next window (`ids_streamed`) |
| 5 | `store_block` | `<= block_units` writes | no (raw NoC) | `cb_staged_pages` slot set, landed; `cb_block_dest`, written | DRAM result pages | slot set free for reuse after `noc_async_writes_flushed` |
| 6 | `drain_writes` | — | no | — | — | every write acked; kernel returns |

Profiling (BB-21): each row above is one `DeviceZoneScopedN` zone, and every blocking wait has its
own: `IF-IDS-ISSUE`, `IF-LOAD-ISSUE`, `IF-READ-WAIT`, `IF-SELECT`, `IF-SLOT-WAIT` (writes_flushed),
`IF-STORE-ISSUE`, `IF-WRITE-WAIT`. The macro compiles to nothing in the plain build
(`kernel_profiler.hpp:1047`).

## API Mapping

| Block operation | Type | Function | File:Line | Template Params / Args | Input CB | Output CB | Which params are block knobs |
|-----------------|------|----------|-----------|------------------------|----------|-----------|------------------------------|
| all (addressing) | raw_api | `TensorAccessor::get_noc_addr(page_id, offset)` | `tt_metal/hw/inc/api/tensor/tensor_accessor.h:165` | one accessor per tensor (ids, x, s, y) from `TensorAccessorArgs` CT args; `offset = c*row_chunk_bytes` or the window start | — | — | `row_chunk_bytes` (offset) |
| `load_ids_window`, `load_block` | raw_api | `noc_async_read(src_noc_addr, dst_l1, size, noc)` | `tt_metal/hw/inc/api/dataflow/dataflow_api.h:547` | size `align_up(chunk_bytes, A)` / `ids_window_bytes` | DRAM | `cb_batch_ids`, `cb_staged_pages` | `row_chunk_bytes`, `ids_chunk_entries` |
| all | raw_api | `get_write_ptr(cb)` | `dataflow_api.h:325` | CB base address | — | — | — |
| `wait_loads` | raw_api | `noc_async_read_barrier(noc)` | `dataflow_api.h:1921` | also invalidates the RISC-V L1 cache | — | — | — |
| `select_block` | raw (scalar) | loads through `volatile tt_l1_ptr uint32_t*` | — | unsigned compares only | `cb_batch_ids` | `cb_block_dest` | `block_units`, `ids_chunk_entries` |
| `store_block` | raw_api | `noc_async_write(src_l1, dst_noc_addr, size, noc)` | `dataflow_api.h:876` | non-posted (default); size = logical chunk bytes | `cb_staged_pages`, `cb_block_dest` | DRAM | `block_units`, `row_chunk_bytes` |
| slot reuse | raw_api | `noc_async_writes_flushed(noc)` | `dataflow_api.h:1980` | once per block before a slot set is refilled | — | — | `staging_depth` |
| `drain_writes` | raw_api | `noc_async_write_barrier(noc)` | `dataflow_api.h:1951` | once per kernel (BB-03) | — | — | — |
| zones | raw_api | `DeviceZoneScopedN(name)` | `tt_metal/hw/inc/api/debug/kernel_profiler.hpp:1012` (plain build: `:1047`) | one per operation and per wait | — | — | — |
| host split | raw_api | `ttnn.split_work_to_cores(core_grid, units, row_wise)` | `ttnn/cpp/ttnn-nanobind/operations/core.cpp:505` | CoreRangeSet overload | — | — | `max_work_cores` |
| host core set | raw_api | `ttnn.num_cores_to_corerangeset(n, grid, row_wise)` | `ttnn/cpp/ttnn-nanobind/operations/core.cpp:460` | — | — | — | `max_work_cores` |

**Helpers considered and rejected** (every entry above is `raw_api`: the op has no compute phase and
the helper library's dataflow helpers do not address DRAM pages by index):

| Candidate | File:Line | Why it cannot express this shape |
|-----------|-----------|----------------------------------|
| `noc_async_read_page(id, addrgen, dst, offset, noc)` | `dataflow_api.h:1243-1258` | always moves `get_aligned_page_size()` bytes (no size argument), so it cannot move a row chunk smaller than the page; `noc_async_read` + `get_noc_addr(page, offset)` covers whole pages and chunks with one code path |
| eltwise `copy(shape, in, out)` | `ttnn/cpp/ttnn/kernel_lib/eltwise/api/convenience.hpp:113` | a compute-kernel copy through unpack → DEST → pack over tiles: the op is ROW_MAJOR with no tiles, has no arithmetic, and must not put a 32-bit payload on an engine path (R5) |
| `tilize` / `untilize` | `ttnn/cpp/ttnn/kernel_lib/tilize_helpers.hpp`, `untilize_helpers.hpp` | layout conversion; nothing is tiled |
| `local_addr`, `set_read_state`, `read_with_state` | `ttnn/cpp/ttnn/kernel_lib/local_copy_helpers_dataflow.hpp:57,74-75,94-133` | self-aimed L1 → L1 reads with one fixed source and size per state; the sources here are DRAM pages in varying banks |
| `set_read_trid`, `async_read_barrier_with_trid` | `local_copy_helpers_dataflow.hpp:147,158` | pays only when a block cannot be widened (catalog `trid_double_issue`); here the block is the whole core assignment |
| `McastArgs` sender/receiver; `ttnn.Mcast` | `ttnn/cpp/ttnn/kernel_lib/mcast/kernel/mcast_args.hpp`; `mcast/host/mcast.hpp:131` | the mechanism of the `deferred` row `ids_mcast`; no built regime multicasts |
| reduce / matmul / bias / reblock / SFPU helpers | `kernel_lib/*_helpers*.hpp` | no arithmetic |

## Host Contract

| Item | Decision |
|------|----------|
| Package | `ttnn/ttnn/operations/g6_indexed_fill/` exports `g6_indexed_fill`, `INPUT_TAGGERS`, `SUPPORTED`, `EXCLUSIONS`, `describe_plan`, `PLAN_CAPS` (the acceptance test imports the last two) |
| Entry point | `g6_indexed_fill(batch_ids, input_tensor, value_tensor, *, dim=0)`; first line `validate(...)`; then allocate the result, take the cached descriptor (or build it), dispatch `ttnn.generic_op([batch_ids, input_tensor, value_tensor, output], descriptor)` — one program, output last |
| `INPUT_TAGGERS` | `{"alignment": tag_alignment}`; `tag_alignment(inputs, axes)` reads `inputs[1] = [B, d1, d2, W]`: `W % 32 != 0` → `"w_non_aligned"`, elif `d2 % 32 != 0` → `"h_non_aligned"`, else `"tile_aligned"`; `inputs = (ids_shape, input_shape, value_shape)` |
| `validate()` order | (1) `dim != 0` → ValueError; (2) ranks (`batch_ids` 1, `input_tensor` 4, `value_tensor` 4) → ValueError — before the tagger, which reads `shape[2:]`; (3) axes `{dtype: input_tensor.dtype, layout: input_tensor.layout, alignment: tag}` vs SUPPORTED → `UnsupportedAxisValue`, then EXCLUSIONS → `ExcludedCell`; (4) `batch_ids` / `value_tensor` dtype uint32 and ROW_MAJOR → ValueError; (5) `value_tensor.shape[0] == batch_ids.shape[0] >= 1`, `value_tensor.shape[1:] == input_tensor.shape[1:]`, every extent `>= 1` → ValueError; (6) every tensor on the device, the same device, `buffer_type == DRAM` and `memory_layout == INTERLEAVED` → ValueError. No check reads a value of `batch_ids` |
| Output | `ttnn.allocate_tensor_on_device(ttnn.Shape(input_tensor.shape), ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT, device, ttnn.DRAM_MEMORY_CONFIG)`, a new buffer per eager call; inputs are only read (rule 4) |
| Program arguments | compile-time: the plan (`B, R, K, P, page_bytes, row_chunk_bytes, row_chunks_per_page, block_units, staging_depth, ids_chunk_entries, ids_chunks, ids_resident, A`, CB indices) + `TensorAccessorArgs` of ids, x, s, y. Runtime per core: the four buffer addresses + `unit_begin`, `unit_count`. **Nothing derives from a value of `batch_ids`** (rule 1) |
| Kernel | one data-movement kernel on `RISCV_0` / `NOC_0` (`ttnn.DataMovementConfigDescriptor`), on `all_cores`; no compute kernel; default optimisation level (no size-optimised build, BB-18) |
| Descriptor cache (BB-07) | module dict keyed by `(device id, the three shapes, dtypes, layouts, memory configs, tuple(sorted(PLAN_CAPS.items())))` → `(plan, descriptor)`; a hit patches only runtime args 0–3 (ids, x, s, y addresses) of every core and dispatches; `custom_program_hash` may be set to the key's hash to skip re-hashing |
| Trace (BB-16) | at capture the eager path runs once (allocation + recorded program); a replay re-runs the recorded program, whose kernel reads the rewritten ids buffer; nothing is allocated or built after capture (the cache is warm from the eager call before capture) |
| `describe_plan(batch_ids, input_tensor, value_tensor, *, dim=0)` (BB-08, BB-21) | returns a dict with `regime`, `num_cores`, `core_ranges` (number of rectangles of `all_cores`), `units`, `block_units`, `staging_depth`, `max_blocks_per_core`, `row_chunk_bytes`, `row_chunks_per_page`, `ids_chunk_entries`, `ids_chunks`, `l1_bytes_per_core`, `roles` (one entry per work core: `core` `(x, y)`, `unit_begin`, `unit_count`, `x_units`, `value_units`) — the same plan the dispatch uses |
| `PROPERTIES` (optional) | `multi_core` declared |

**Note for the lead (structural, not an INVALID candidate).** TARGET and Phase 0 SUPPORTED list
only `alignment = "h_non_aligned"`, so `validate()` refuses an input whose `W % 32 != 0`
(`"w_non_aligned"`) or whose `d2 % 32 == 0` (`"tile_aligned"`), while rule 9 asks for every W and
d2. The kernels above are alignment-agnostic (ROW_MAJOR, no padding): widening is a SUPPORTED change
with no kernel change, once TARGET lists the values (a `/golden-tests` action).

## Acceptance Test

`tests/ttnn/unit_tests/operations/g6_indexed_fill/test_g6_indexed_fill.py` (+ the directory
`conftest.py`). Deviations from the generic template, each forced by this op's contract:

| Template default | Here | Why |
|------------------|------|-----|
| `torch.randn` | `torch.randint` over `[0, 2^32)` with `torch.manual_seed(42)`, `UINT32_MAX` and `2^24 + 1` forced into every batch | uint32 payloads; a float generator never exercises the upper bits (R5) |
| ≥ 4 shapes | the production shape and the rule-9 shape only | BB-22 |
| PCC + RMS gate | `torch.equal` on int64 views | `helpers.TOLERANCES[uint32] == ("exact", None)` |
| no per-file `pytestmark` | `pytestmark = use_module_device({"trace_region_size": 1 << 20})` | the module-scoped device takes its parameters from the module's marker; rule 1 is shown by a trace replay |
| one regime | `PLAN_CAPS` overrides pin `ids_streamed`, multi-block at depth 1 and 2, and row chunks on the rule-9 shape | regime-pinned tests without new shapes |

## Key Risks and Gotchas

| Risk | Why it bites here | Mitigation in this design |
|------|-------------------|---------------------------|
| Frozen position (R2) | a host read, CT or RT arg of the index passes every single-call test | the kernel reads the ids buffer every run; CT/RT args carry shapes, plan and addresses only; cache key has no ids value; tested by in-place rewrite + trace replay + an unchanged program-cache count |
| Stale address on a cache hit | a hit that forgets to patch the result address writes into the previous call's output — the returned tensor then holds stale but often right values | patch all four addresses on every hit; the test keeps every result alive and checks each |
| Signed or unchecked index (R3) | `4294967295` as `-1`, or `7`, writes outside the result | `uint32_t` only; `t < B` before forming `t*R + r`; guard-tensor test with 7 and 4294967295 |
| 4-B ids page in a 64-B DRAM granule (R6) | a tile-sized or misaligned read returns neighbour bytes | aligned-window read (`[align_down, align_up)` of the entries' bytes), entry offset inside the window |
| Blackhole L1 data cache | the index loaded before the barrier, or through a cached pointer, is stale | volatile loads only after `noc_async_read_barrier` |
| Two writers of one page (R4) | copy-then-overwrite or a duplicate entry written by two cores races | exactly-one-writer invariant of `select_block`: named x units SKIP; only the last equal entry writes |
| Value unit straddling an ids window (`ids_streamed`) | the winner test of entry `i` needs `ids[i]` and every later entry, possibly in later windows | the decision state per unit persists in `cb_block_dest` across windows of one block |
| Ragged tails | last block of a core, last row chunk of a page, the 1-entry ids window at a 4-B offset | runtime extents into the same block operations; tests force each on the rule-9 shape |
| Values through a narrow path (R5) | a compute-kernel copy or float staging drops bits above 2^24 | no compute kernel; payloads move NoC → L1 → NoC untouched |
| In-flight transactions at return (BB-03) | a pending write hangs the next program | `noc_async_write_barrier` is the kernel's last act; every read is barriered before use |
