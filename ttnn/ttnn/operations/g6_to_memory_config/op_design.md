# Operation Design: g6_to_memory_config

## Overview

| Field | Value |
|-------|-------|
| Classification | data_movement (no compute kernel, no arithmetic) |
| Goal | Return the input's tile pages, bit for bit, in the placement `memory_config` names. Every page is read once from its source location and written once to its target location, in one program, by the core that holds one side of it. |
| Math | `y[i] = x[i]` for every logical index `i`; `shape(y) = shape(x)`, `dtype(y) = dtype(x)`, `layout(y) = TILE`, `placement(y) = memory_config` |
| Unit of work | one tile **page** (bf16 2048 B; bfloat8_b 1088 B, bfloat4_b 576 B in the refinement). A page is never cut: a sub-page transfer re-forms the shared-exponent groups of a block format. |
| References | `eval/golden_tests/g6_to_memory_config/feature_spec.py` (TARGET, 105 INPUTS, LOOSE_CASES); `eval/golden_tests/g6_to_memory_config/helpers.py` (oracle = `clone`, exact gate, placement gate); `.claude/references/blocking-model.md`; `.claude/references/l1-footprint-discipline.md`; `tech_reports/tensor_accessor/tensor_accessor.md`, `tensor_accessor_iterator.md`; `tt_metal/impl/buffers/buffer_distribution_spec.cpp` (shard→core order); `ttnn/ttnn/operations/examples/master.md` (catalog entries listed under Perf lamps) |

## Parameters

| Name | Type | Required | Valid Range | Default | CT/RT |
|------|------|----------|-------------|---------|-------|
| `input_tensor` | `ttnn.Tensor` | yes, positional | on device, TILE, rank 2–4, any placement (DRAM/L1 × interleaved/height/width/block, row-major shard orientation) | — | buffer address: RT; geometry/placement: CT (TensorAccessorArgs) |
| `memory_config` | `ttnn.MemoryConfig` | yes, positional | DRAM or L1; interleaved or height/width/block sharded with a row-major shard spec that covers the padded tile grid | — | output buffer address: RT; placement: CT (TensorAccessorArgs of the allocated output) |

There is no dtype, layout or compute-config parameter. Nothing else is accepted.

## Tensors

### Input

| Property | Requirement |
|----------|-------------|
| Shape | `(..., H, W)`, rank 2–4. `H` any positive value; `W` a multiple of 32 in every production case (a non-multiple is still legal: pages are whole tiles). |
| Tile grid | `tensor_rows_tiles = prod(shape[:-2]) * ceil(H/32)` (per-image ceil: each image is tile-padded on its own), `tensor_cols_tiles = ceil(W/32)`, `tensor_pages = tensor_rows_tiles * tensor_cols_tiles`. Page id = `row * tensor_cols_tiles + col` — identical under every placement (the buffer distribution folds leading dims into the outer shard dim: `buffer_distribution_spec.cpp:442-444`). |
| Dtype | Phase 0: bfloat16. Refinement: bfloat8_b, bfloat4_b (pages moved whole; no kernel change). |
| Layout | TILE |
| Memory | any: DRAM or L1; interleaved, height, width or block sharded (uneven last shard, more cores than shards, multi-range and off-origin grids all legal) |

### Output

| Property | Value |
|----------|-------|
| Shape | `input_tensor.shape` (logical; padded shape identical) |
| Dtype | `input_tensor.dtype` |
| Layout | TILE |
| Memory | exactly `memory_config` (buffer type, memory layout, shard grid with its ranges in the given order, shard shape, orientation). Allocated with `ttnn.allocate_tensor_on_device(shape, dtype, TILE, device, memory_config)` — never the input returned, never another placement. Rows `H..32*ceil(H/32)-1` of the last tile row of each image are copied as part of the whole page; their content is unspecified. |

## Blocking Model

### Axes

| Axis | Character (+ one-clause reason) | Extent knob | Phase 0 value | Knob source | Core-assignment | Later unlock |
|------|--------------------------------|-------------|---------------|-------------|-----------------|--------------|
| `tile_row` (leading dims folded × `ceil(H/32)` per image) | **independent** — output page `(r, c)` depends on input page `(r, c)` alone | `core_rows_tiles` (rows of the owned block) | **whole owned extent**: sharded owner → its shard's rows (`shard_rows_tiles = shard_shape[0]/32`, ragged last shard = `tensor_rows_tiles − (n_shard_rows−1)·shard_rows_tiles`); L1-bank owner → not cut separately (flattened with `tile_col`, see `bank_pages`); bounce → flattened (see `bounce_block_pages`) | the owned placement's shard spec (host), or the regime knob named in the cell | by the **mover side's placement** (see Regimes): height/block shards → one row band per core; width shards → not split (every core spans all rows); L1 interleaved → page id mod `num_l1_banks`; bounce → contiguous target-order chunks over the full compute grid | knob-turn |
| `tile_col` (`ceil(W/32)`) | **independent** — same reason | `core_cols_tiles` | **whole owned extent**: sharded owner → its shard's columns (`shard_cols_tiles = shard_shape[1]/32`, ragged last shard = remainder); L1-bank owner / bounce → flattened as above | as above | width/block shards → one column band per core; height shards → not split; L1 interleaved / bounce → as above | knob-turn |
| flattened page index (`tile_row × tile_col`) — only for the two owners that have no rectangle | independent | `bank_pages` (L1-bank owner) / `bounce_block_pages` (bounce) | `bank_pages` = every page of the bank, `ceil((tensor_pages − b)/num_l1_banks)` (one block); `bounce_block_pages = min(ceil(max_chunk_pages / bounce_depth), floor(MODEL_L1_BUDGET_BYTES / (bounce_depth · aligned_page_size)))` | host constants `MODEL_L1_BUDGET_BYTES = 1347776`, `bounce_depth`; derived once in the plan | as above | knob-turn |

No axis is **dependent** (nothing is reduced, scanned or combined) and no operand is **reuse-shared**:
operand-reuse check for the chosen split — the single input varies along every split (each moved
page is a distinct source page), so no core re-reads bytes another core also reads; no broadcast row
exists.

Every intermediate stage's data is the page itself: there is no derived tensor to re-block.

Transfer granularity inside a block (not a block axis — a mechanism of the move): a **segment** is
the maximal run of consecutive pages **in the mover's walk order** (R1/R2: row-major slot order of
the owned shard, or increasing page id of the owned bank; R3: the worker's target-order chunk) whose
address advances by exactly `aligned_page_size` from page to page on **both** sides, clipped to the
block and to `max_segment_pages`. One NoC transaction moves one segment. Phase 0 value of `max_segment_pages` = `floor(MAX_TRANSFER_BYTES / aligned_page_size)`
with `MAX_TRANSFER_BYTES = 1 MiB` — i.e. every maximal run in one call; single source: one host
constant passed as a CT arg.

### Buffer-depth knobs

| CB | Depth knob | Phase 0 value | What the depth buys |
|----|------------|---------------|---------------------|
| `cb_staging_pages` (bounce regime only) | `bounce_depth` | 2 | reader gathers block k+1 while writer scatters block k. With `bounce_block_pages = ceil(max_chunk_pages / 2)` the two slots hold the whole chunk, so depth 2 costs the same L1 as one whole-chunk block and buys the overlap. |
| — (pull / push regimes: no CB) | NoC transactions in flight | all of a core's segments issued back to back, **one** barrier per core per call; in-flight count bounded only by NoC command-buffer back-pressure | issue/latency overlap across the whole owned block. Catalog `double_buffer`: one tile per barrier is "the trap" (6.5 GB/s); `trid_double_issue`: several blocks per barrier is "the bigger win". |

### Mechanism caps

| Mechanism | Cap on which extent | Clamp | What happens unclamped |
|-----------|--------------------|-------|------------------------|
| Whole-page transfer (block formats) | segment | ≥ 1 whole page; sizes are `n · aligned_page_size` | a sub-page cut re-forms bfp8/bfp4 exponent groups → values change, equality gate fails |
| NoC call size (`noc_async_read`/`noc_async_write` document `size` 0..1MB, `dataflow_api.h:540-541`) | segment | `max_segment_pages = floor(1 MiB / aligned_page_size)` | out-of-range size argument |
| Single-packet stateful APIs (`read_with_state`, `*_one_packet*`), if the implementer uses them | segment | `≤ NOC_MAX_BURST_SIZE` (16 KiB on BH, `noc_parameters.h:290`; `local_copy_helpers_dataflow.inl:66` static_asserts it) | static_assert / truncated transfer |
| Contiguity | segment | `min(run(side_a), run(side_b), block remainder, max_segment_pages)` with, per side: staging slot → the whole block; a sharded side walked in its own slot order (R1/R2 owned shard, R3 sharded target) → to the end of the shard when its rows are full (`core_cols_tiles == shard_cols_tiles`), else to the end of the current row; a sharded side walked by consecutive page ids (the remote side of R1/R2) → `acc.num_contiguous_pages(p, row_end)` when `acc.contiguous_page_stride() == 1`, else 1; an interleaved side → 1 (consecutive pages sit in different banks) | a segment crossing a shard edge, a ragged row's padding slots, or a bank boundary reads/writes the wrong slots → wrong pages |
| Equal page stride on both sides | segment > 1 | multi-page segments only when `input.buffer_aligned_page_size() == output.buffer_aligned_page_size()` (true for 2048/1088/576 B in DRAM and L1); otherwise `max_segment_pages = 1` | pages land at the wrong stride |
| Sharded slot stride | owned-block addressing | the slot of page `(r, c)` in shard `(si, sj)` is `(r − si·shard_rows_tiles)·shard_cols_tiles + (c − sj·shard_cols_tiles)` — the **full** shard width, also in a ragged last shard (`buffer_distribution_spec.cpp:41-52` strides by the full shard shape) | ragged last shard (e.g. 40 of 52 columns in Qwen 2071) packed at the ragged width → every row after the first misplaced |
| One shard per core | owned set | validation refuses a grid with fewer cores than shards (block: fewer columns/rows than shard columns/rows); host asserts `n_shards ≤ len(core list)` | round-robin wrap gives a core two shards; a one-shard walk misses pages |
| L1-bank ownership | `bank_pages` blocks | predicate `l1_bank_owners := ttnn._ttnn.reports.get_device_info(device).l1_num_banks == grid.x · grid.y` (`grid = device.compute_with_storage_grid_size()`); when false an L1-interleaved side has 0 holders (falls to another regime) | pages of a bank hosted outside the launch grid are never moved |
| Staging capacity | `bounce_block_pages` | `≤ floor(MODEL_L1_BUDGET_BYTES / (bounce_depth · aligned_page_size))` | BB-05 overflow; launch failure in the model |
| CB ring wrap | bounce push/pop count | every push/pop is the nominal `bounce_block_pages`, capacity `bounce_depth · bounce_block_pages`; a ragged last block narrows only the NoC transfers | pointers never reset on `fifo_limit` → slot drift, hang or corruption |
| Page-id width | `tensor_pages` | `< 2^32` (host assert) | page-id wraparound |
| Static accessor rank (`shard_pages` static_asserts it, `tensor_accessor.h:336`) | accessor args | `ttnn.TensorAccessorArgs(tensor)` default config (all compile-time) | compile error |

### Regimes

Holder count of a side (host, exact):
`holders(side) = 0` for DRAM (any scheme);
`min(tensor_pages, num_l1_banks)` for L1 interleaved if `l1_bank_owners`, else 0;
`ceil(tensor_rows_tiles / shard_rows_tiles) · ceil(tensor_cols_tiles / shard_cols_tiles)` (non-empty shards) for L1 sharded.

**Regime-selection function** (`h_in = holders(input)`, `h_out = holders(memory_config)`):

```
regime = "bounce" if h_in == 0 and h_out == 0
       = "pull"   if h_out >= h_in          (ties go to pull)
       = "push"   otherwise
```

| Regime | Status | Predicate | Block | Data movement vs. minimum | What a bigger block buys |
|--------|--------|-----------|-------|---------------------------|--------------------------|
| **R1 pull** — each core holding target pages reads them from the source straight into its own target slots | **built** (57 bf16 + 6 bf8 production cases at 110 L1 banks; DRAM→L1 and L1→L1 with `h_out ≥ h_in`) | `h_out > 0 and h_out >= h_in` | the core's whole owned target set: sharded → `core_rows_tiles × core_cols_tiles` (its shard); L1 bank → `bank_pages` | **minimum** against the DRAM boundary and the NoC: each page crosses its source tier once and lands in its final slot; no staging, no second hop. L1→L1 on the same core is a NoC loopback (still one move). | one kernel launch, one accessor construction, one bank discovery (bank owners), and **one** `noc_async_read_barrier` per core per call; maximal segments amortize the ~115 ns per-transaction issue cost (catalog `double_buffer`). Nothing scales per page except the transaction itself. |
| **R2 push** — each core holding source pages writes them straight to their target slots | **built** (29 bf16 + 6 bf8 production cases at 110 L1 banks; L1→DRAM and L1→L1 with `h_in > h_out`) | `h_in > h_out` | the core's whole owned source set (shard or `bank_pages`) | **minimum**, same accounting as R1 | same as R1 with one `noc_async_write_barrier` per core per call |
| **R3 bounce** — full-grid workers gather a block from the source into an L1 staging slot, then scatter it to the target | **built** — positive reason: `validate()` admits DRAM→DRAM cells in Phase 0 (SUPPORTED holds `dram` on both sides; EXCLUSIONS removes only DRAM width-sharded targets), so they need a regime; it is also the stepping stone of the bfloat4_b DRAM width-sharded refinement (the PERF FOCUS case Qwen 2183), which then only drops the exclusion | `h_in == 0 and h_out == 0` (DRAM↔DRAM, or an L1-interleaved side when `l1_bank_owners` is false) | `bounce_block_pages` consecutive pages of the worker's target-order chunk | **minimum for DRAM→DRAM**: each page crosses DRAM once read, once written; the L1 bounce is structural (no core holds either side, a NoC read lands only in the issuer's L1). 2 NoC transactions per segment. | per block: one CB reserve/push + one read barrier (reader), one wait/pop + one write flush (writer); one final write barrier. Bigger block = fewer handshakes/barriers and longer coalesced target writes (a full-row DRAM width shard is contiguous in slot order, so one block is one write — Qwen 2183: 373 pages, 215 KB). |
| **R4 helper fan-out** — holders keep their pages; extra grid cores take a share by bouncing (pull from the source into their staging, push to the target) | **deferred** — R1/R2 already serve every shape R4 would, at the minimum crossing count; R4 is the measured trade "+1 L1→L1 NoC crossing per helper-moved page (`+helper_pages · aligned_page_size` bytes) for issue parallelism beyond the holder count". The busiest Phase 0 mover holds 94 pages (Qwen 2078, all 110 banks busy, DRAM-bound) and the busiest narrow mover 64 pages (8 width-shard holders pushing 1 MiB to DRAM: Qwen 2072/2080/2135/2171). Reachable unchanged: a helper is an R3 worker whose chunk is a subset of the owner's pages; only the host assignment changes. | — | — | +1 cross-core crossing per helper page | — |
| **R5 grid-split bounce for every pair** (`split_work_to_cores` over pages, every page staged) | **rejected** — superseded by R1/R2 wherever a side is in L1: same DRAM crossings, +1 NoC crossing per page, +1 staging CB per core. Dead end: nothing of it survives under R1/R2. | — | — | +1 crossing/page | — |
| **R6 two-step through a third placement** (interleaved intermediate, or an existing ttnn data-movement op) | **rejected** — forbidden by the spec (one program, read once/write once, BB-01); doubles the traffic | — | — | 2× | — |
| **R7 dependent-axis split + cross-core combine** | **rejected** — no axis is dependent; there is nothing to combine | — | — | — | — |
| **R8 move through compute** (unpack → DEST → pack, e.g. `copy_tile`) | **rejected** — not bit-exact for block formats (re-derives exponents), adds a compute kernel and two CBs per core; superseded by R1/R2/R3 everywhere | — | — | +2 L1 passes/page | — |

Regime-pinned tests are required and present in the acceptance test (`select_plan(...).regime`
asserted per case, with the expected value recomputed from the predicate above).

### Traffic ranking

Per call, `B = tensor_pages · aligned_page_size`. DRAM crossings count whole-tensor moves across
the DRAM boundary.

| Rank | Candidate split | DRAM crossings | Cross-core NoC bytes | Core-local staging | Verdict |
|------|-----------------|----------------|----------------------|--------------------|---------|
| 1 | Owner split (R1/R2): the L1-side holder moves each page in one transaction | input 1× if in DRAM, output 1× if in DRAM | `B` when both sides are L1 on different cores (one hop per page); 0 extra otherwise | none | **chosen** — the minimum: every page crosses once, lands in place |
| 1 (DRAM↔DRAM only) | Bounce (R3) | input 1×, output 1× | 0 cross-core (DRAM→worker L1→DRAM) | `bounce_depth · bounce_block_pages` pages | chosen where no L1 side exists — the minimum there |
| 2 | Owner + helpers (R4) | same | `B + helper_pages · page` | helpers only | deferred (measured trade vs. issue parallelism) |
| 3 | Grid split + bounce for every pair (R5) | same | `2B`-class: every page makes two hops | every core | rejected |
| 4 | Two-step via an intermediate placement (R6) | +1× read and +1× write of the whole tensor | +`B` | intermediate tensor | rejected (forbidden) |
| — | Dependent-axis split + combine | n/a — no dependent axis | — | — | not applicable |

Occupancy is ranked separately and does not override the table: R1/R2 occupancy = the holder count
of the mover side (1–110 cores in production); the full-grid alternative is R4.

### Block schedule

R1 pull — one kernel per mover core, RISCV_1 on NoC0 (catalog `noc_placement`: reads on NoC0):

```cpp
resolve_owned_set();                              // shard (id, core_rows_tiles, core_cols_tiles) or own L1 bank(s)
for (uint32_t block_idx = 0; block_idx < num_blocks_this_core; ++block_idx) {   // Phase 0: 1 (whole owned set)
    gather_block(block_idx);                      // one NoC read per segment: source -> own target slots
}
drain_reads();                                    // one noc_async_read_barrier
```

R2 push — one kernel per mover core, RISCV_0 on NoC1 (catalog `noc_placement`: writes on NoC1):

```cpp
resolve_owned_set();
for (uint32_t block_idx = 0; block_idx < num_blocks_this_core; ++block_idx) {   // Phase 0: 1
    scatter_block(block_idx);                     // one NoC write per segment: own source slots -> target
}
drain_writes();                                   // one noc_async_write_barrier
```

R3 bounce — reader on RISCV_1/NoC0 and writer on RISCV_0/NoC1 of every worker, staging CB between:

```cpp
for (uint32_t block_idx = 0; block_idx < num_blocks_this_core; ++block_idx) {   // Phase 0: 2 (ceil(chunk/2) per block)
    reserve_slot(block_idx);   gather_block(block_idx);  publish_slot(block_idx);    // reader
    await_slot(block_idx);     scatter_block(block_idx); release_slot(block_idx);    // writer
}
drain_writes();                                                                       // writer
```

| Block operation | Block shape it acts on | Resident across it | Intended frequency of fixed costs |
|-----------------|------------------------|--------------------|-----------------------------------|
| `resolve_owned_set` | — | — | once per kernel: read RT args; bank owners scan `NUM_L1_BANKS` entries once to find their bank(s) |
| `gather_block` (R1, R3 reader) | R1: whole owned set (`core_rows_tiles × core_cols_tiles` or `bank_pages`); R3: `bounce_block_pages` target-order pages | R1: the target slots (they are the output); R3: the staging slot | one source `get_noc_addr` + one `noc_async_read` per **segment**; no barrier inside (R1) / one read barrier per block (R3) |
| `scatter_block` (R2, R3 writer) | R2: whole owned set; R3: `bounce_block_pages` | R2: the source slots (the input, untouched); R3: the staging slot | one target `get_noc_addr` + one `noc_async_write` per segment; R3: one `noc_async_writes_flushed` per block before release |
| `reserve_slot` / `publish_slot` / `await_slot` / `release_slot` (R3) | `bounce_block_pages` | staging slot | one CB handshake pair per block per side, nominal count even for a ragged last block |
| `drain_reads` / `drain_writes` | — | — | exactly once per kernel, last statement (BB-03) |

Stall-shadow check: the only waits are the final barrier (R1/R2), the per-block slot waits and the
read barrier (R3). Everything a mover could do is issued before its wait; no work is independent of
the awaited transfers. Nothing to reorder.

### Perf lamps

| Lamp | Why the default may be wrong here | Nearby alternative to measure |
|------|-----------------------------------|-------------------------------|
| Two-RISC issue split | R1/R2 use one data-movement RISC per core; narrow movers are issue-bound (8 holders × 64 pages pushed to DRAM in Qwen 2072/2080/2135/2171; ~115 ns/transaction, catalog `double_buffer`) | split each mover's segments between RISCV_1/NoC0 and RISCV_0/NoC1 (catalog `split_reader`: 1.21× at 2048 B when issue-bound) |
| Pull-vs-push tie rule for L1↔L1 | max-holder rule picks push for Qwen 2064/2127/2161 (≈110 bank holders × 1–2 pages) and 2070/2133/2166/2073/2136/2172/2081 (40 holders × 4 pages); catalog `tensix_all_reduce`: pull wins ≤ 24 tiles/core, push is monotone | always-pull for L1↔L1 |
| Grid synchronization (bank-owner launch) | R1/R2 with an L1-interleaved mover side launch the full compute grid even when `tensor_pages` is tiny (Qwen 2155/2156: 8 pages, 110-core launch) | pick the side with fewer cores when its per-core pages stay ≤ a few (here: push from the 1 block-shard holder) |
| Bank discovery on device | each bank-owner core scans `NUM_L1_BANKS` (110) table entries per call | host-provided bank id per core (needs a host bank→core map) |
| Segment cap | `max_segment_pages` = maximal run; one long transfer serializes on one source/target pair (catalog `read_coalescing`: coalescing wins, but "does not prove one full-block call is the best transfer size") | cap at 4–8 KiB on L1↔L1 width-shard moves (Qwen 2067/2070/2163/2166) |
| Overlap (R3) | `bounce_block_pages = ceil(chunk/2)` gives only two blocks per worker; first gather and last scatter do not overlap | smaller blocks with `bounce_depth` 3–4 at equal L1 (catalog `trid_double_issue`: `ahead = cb_blocks/2`) — PERF FOCUS case Qwen 2183 |
| DRAM bank affinity (R1 bank owners from DRAM, R3) | workers read interleaved DRAM in page-id order; the PERF FOCUS writes land in one DRAM bank per worker chunk | stagger each core's start page by `core % num_dram_banks` (catalog `bank_stagger`: 1.04–1.07× interleaved, 1.13–1.32× width-sharded); place R3 workers of shard k near DRAM bank k via `ttnn.get_optimal_dram_bank_to_logical_worker_assignment` (catalog `read_coalescing`: affinity 1.22–1.48× alone) |

Catalog entries that informed the knobs: `double_buffer` (no per-page barrier; one barrier per
block), `trid_double_issue` (several blocks per barrier; DRAM→DRAM bounce structure and depth),
`noc_placement` (reads NoC0 / writes NoC1; `split_work_to_cores(row_wise=True)` for R3),
`split_reader` (two-RISC lamp), `read_coalescing` (segment coalescing legal only on physically
contiguous bytes — never across interleaved pages), `bank_stagger` (affinity lamp),
`tensix_all_reduce` (pull vs push), `tile_reorder` (one transfer per whole 2 KiB tile, no face split).

## Dataflow Strategy

| Stage | Format | Mechanism | Notes |
|-------|--------|-----------|-------|
| R1: source page(s) → own target slots | the page bytes, untouched | one `noc_async_read(src_noc_addr, dst_local_addr, n·aligned_page)` per segment; `src_noc_addr` from the source TensorAccessor, `dst_local_addr` = local address of the owned target slot (from the output TensorAccessor's `get_noc_addr(page)` local part, or `shard_pages`) | no CB: the read lands in the output buffer itself (pack-into-destination); a resident target shard is written in place, never re-read |
| R2: own source slots → target page(s) | untouched | one `noc_async_write(src_local_addr, dst_noc_addr, n·aligned_page)` per segment | no CB: the resident source shard/bank is read in place by the write itself |
| R3: source → staging slot → target | untouched | reader `noc_async_read` per source segment into `cb_staging_pages`, read barrier, push; writer `noc_async_write` per target segment from the slot, `noc_async_writes_flushed`, pop | the only CB of the op; dataflow-only payload, never through DEST |
| Page → core ownership (sharded side) | — | shard id `k` (row-major over the shard grid `ceil(R/shard_rows_tiles) × ceil(C/shard_cols_tiles)`) lives on `cores[k]`, with `cores = ttnn.corerange_to_cores(grid, grid.num_cores(), True)` for height/width (ROUND_ROBIN_1D, `buffer_distribution_spec.cpp:162-175`) and the grid's first range trimmed to `n_shard_cols × n_shard_rows`, row-major, for block (GRID_2D, `:177-201`) | the host must use the buffer's own order, never a re-derived regular grid |
| Page → core ownership (L1 interleaved side) | — | page `p` lives in L1 bank `p mod NUM_L1_BANKS`; a mover core finds its bank(s) on device by testing `is_local(bank_address<false>(b, 0, noc), noc)` for every `b` (`noc_address_backend_xy.h:50-66`) | gated by `l1_bank_owners` |

TARGET placement axes, as regime rows (no inter-Tensix protocol exists or is needed: no semaphores, no multicast):

| TARGET value | Cost class | Why |
|--------------|-----------|-----|
| `in_scheme`/`out_scheme` = `height_sharded`, `width_sharded`, `block_sharded` | knob-turn | every cut leaves each page's move self-contained (both axes independent); the scheme only changes which cores hold which pages (owned-set description) |
| `in_buffer`/`out_buffer` = `dram` / `l1` | knob-turn | the buffer type only changes holder count and so the regime choice; all regimes are built |
| `{out_scheme: width_sharded, out_buffer: dram}` (Phase 0 EXCLUSION) | knob-turn | R3 bounce already serves DRAM→DRAM; drop the exclusion |
| `dtype` = `bfloat8_b`, `bfloat4_b` | knob-turn | page size comes from the buffer (`aligned_page_size` in the accessor); whole-page moves are bit-exact |

Structural impossibilities: none to add (feature_spec `INVALID = []` is correct).

## Work Distribution

| Field | Value |
|-------|-------|
| Work unit | a block (whole owned set for R1/R2; `bounce_block_pages` for R3); a segment inside it is one NoC transaction |
| Grid | R1/R2 sharded mover side: that placement's shard grid, as given, merged into maximal rectangles (BB-17: ≤ 3 ranges for every production grid — the 16 and 8 single-core ranges of Qwen 2058/2121 merge to 2); cores of the grid with no shard get zero-work args. R1/R2 L1-interleaved mover side and R3: `device.compute_with_storage_grid_size()` as one rectangle (R3: the `split_work_to_cores(..., row_wise=True)` prefix). |
| Per-core work | R1/R2 sharded: shard id + `core_rows_tiles`, `core_cols_tiles` (ragged on the last shard row/column). R1/R2 bank: every page `p ≡ b (mod NUM_L1_BANKS)`, `p < tensor_pages`, of each owned bank `b`. R3: contiguous chunk `[chunk_start, chunk_start + chunk_pages)` of the target-order sequence (interleaved target: page ids; sharded target: shards in shard-id order, row-major over the real pages of each shard), from `ttnn.split_work_to_cores(grid, tensor_pages, row_wise=True)`; `num_blocks_this_core = ceil(chunk_pages / bounce_block_pages)`. |
| Remainder | ragged last shard via `ceil`; partial tile rows are whole pages (`tensor_rows_tiles = prod(shape[:-2]) · ceil(H/32)`, never `floor(prod·H/32)`); R3 ragged last block narrows only NoC transfers |
| Mover count | R1 = `h_out`, R2 = `h_in`, R3 = `min(tensor_pages, grid cores)` — chosen by the regime function, not by the shape |
| Plan report (BB-08/BB-21) | `select_plan(input_tensor, memory_config)` returns the plan (regime, holders, `l1_bank_owners`, kernel cores, per-core role) — see Host Contract |

## Circular Buffers

| Semantic Name | Index | Page Size | Num Pages | Sizing rationale | Format | Producer | Consumer | Lifetime |
|---------------|-------|-----------|-----------|------------------|--------|----------|----------|-----------|
| `cb_staging_pages` (R3 only; absent in R1/R2) | 0 | `input.buffer_aligned_page_size()` | `bounce_depth · bounce_block_pages` | live set spans `bounce_block_pages` of the flattened target-order axis per slot, `bounce_depth` slots; streams over `tile_row`/`tile_col` | `input.dtype` | reader | writer | whole R3 kernel |

R1/R2 declare **no CB** (inventory minimized to zero): the mover owns one side of every page it
moves and the NoC transfer itself is the copy. A `cb_descriptor_from_sharded_tensor` alias would add
no handshake (there is no second kernel on the core to hand off to) and is deliberately not used;
the owned shard/bank is addressed through its TensorAccessor. Full ledger: `l1_ledger.md`.

## Block Operation Realization

| # | Block operation | Block shape | Helper? | Input CB (semantic name, pages, state) | Output CB (semantic name, pages) | CB state after |
|---|-----------------|-------------|---------|----------------------------------------|----------------------------------|----------------|
| 1 | `resolve_owned_set` | — | no (raw: RT args; bank scan) | — | — | — |
| 2 | `gather_block` (R1) | owned set | no (raw NoC + TensorAccessor) | — (source tensor via accessor) | — (target slots of the output) | — |
| 3 | `scatter_block` (R2) | owned set | no | — (own source slots) | — (target via accessor) | — |
| 4 | `reserve_slot` → `gather_block` → `publish_slot` (R3 reader) | `bounce_block_pages` | no | — | `cb_staging_pages`, `bounce_block_pages` | +`bounce_block_pages` published |
| 5 | `await_slot` → `scatter_block` → `release_slot` (R3 writer) | `bounce_block_pages` | no | `cb_staging_pages`, `bounce_block_pages`, waited | — | −`bounce_block_pages` popped after `noc_async_writes_flushed` |
| 6 | `drain_reads` / `drain_writes` | — | no | — | — | — |

## API Mapping

| Block operation | Type | Function | File:Line | Template Params / Args | Input CB | Output CB | Which params are block knobs |
|-----------------|------|----------|-----------|------------------------|----------|-----------|------------------------------|
| accessor setup (all) | raw_api | `TensorAccessorArgs<cta, crta>()`, `TensorAccessor(args, base_addr)` | `tt_metal/hw/inc/api/tensor/tensor_accessor.h:71`; host `ttnn.TensorAccessorArgs(tensor)` `ttnn/cpp/ttnn-nanobind/tensor_accessor_args.cpp:26` | CT args from both tensors (scalars first, accessors last); base addresses RT | — | — | — |
| page address | raw_api | `get_noc_addr(page_id)` | `tensor_accessor.h:165` | page id | — | — | — |
| segment length | raw_api | `num_contiguous_pages(page_id, end_page_id)`, `contiguous_page_stride()` | `tensor_accessor.h:192`, `:181` | end = owned-block remainder | — | — | `max_segment_pages` clamps the result |
| sharded owned walk | raw_api | `shard_pages(shard_id, start, end)` (or slot arithmetic per Mechanism caps) | `tensor_accessor.h:331` | shard id from RT args | — | — | owned extents |
| bank discovery | raw_api | `bank_address<false>(b, 0, noc)`, `is_local(addr, noc)`, `extract_local_address` | `tt_metal/hw/inc/internal/dataflow/noc_address_backend_xy.h:50-66` | `NUM_L1_BANKS` | — | — | — |
| `gather_block` | raw_api | `noc_async_read` | `tt_metal/hw/inc/api/dataflow/dataflow_api.h:547` | `(src_noc_addr, dst_local_addr, n·aligned_page)` | — / `cb_staging_pages` slot (R3) | — | segment `n` |
| `scatter_block` | raw_api | `noc_async_write` | `dataflow_api.h:876` | `(src_local_addr, dst_noc_addr, n·aligned_page)` | — / `cb_staging_pages` slot (R3) | — | segment `n` |
| `drain_reads` / R3 read barrier | raw_api | `noc_async_read_barrier` | `dataflow_api.h:1921` | — | — | — | — |
| `drain_writes` | raw_api | `noc_async_write_barrier` | `dataflow_api.h:1951` | — | — | — | — |
| `release_slot` precondition (R3) | raw_api | `noc_async_writes_flushed` | `dataflow_api.h:1980` | — | — | — | — |
| R3 slot handshake | raw_api | `cb_reserve_back` / `cb_push_back` / `cb_wait_front` / `cb_pop_front` | `dataflow_api.h:406` / `:208` / `:479` / `:260` | `bounce_block_pages` | `cb_staging_pages` | `cb_staging_pages` | `bounce_block_pages` |
| profiler zones (BB-21) | raw_api | `DeviceZoneScopedN(name)` | `tt_metal/hw/inc/api/debug/kernel_profiler.hpp:1012` | zone names in Host Contract | — | — | — |

**Helpers considered and rejected** (applies to every `raw_api` row above — the op has no compute phase):

| Candidate | File:Line of the mismatch | Why it cannot express the shape |
|-----------|---------------------------|---------------------------------|
| `local_addr` / `set_read_state` / `read_with_state` (`local_copy_helpers_dataflow.hpp`) | `local_copy_helpers_dataflow.hpp:57` (source is always this core's own L1), `local_copy_helpers_dataflow.inl:66` (`transfer_size <= NOC_MAX_BURST_SIZE`) | self-aimed L1→L1 copies only; the move's source is a remote core or DRAM bank and segments exceed one packet |
| `McastArgs` / `ttnn.Mcast` (`mcast/kernel/mcast_args.hpp`) | `mcast_args.hpp:15-29` (one sender delivers the same payload to a receiver group) | no page goes to more than one destination; nothing is reuse-shared |
| `TensorAccessor::is_local_bank` for bank ownership | `tensor_accessor.h:582` ("not supported by the interleaved tensor accessor") | cannot identify the owned L1 bank of an interleaved buffer; hence the firmware bank-table scan |
| compute helpers (`eltwise_convenience.hpp` `copy`, `tilize`, `untilize`, `reduce`, `matmul_block`) | — | no compute phase exists; a DEST round trip is regime R8 (rejected: not bit-exact for block formats) |
| `tilize_helpers_dataflow.hpp`, `reduce_helpers_dataflow.hpp`, `tile_column_dataflow.hpp`, `index_tile_dataflow.hpp`, `dfb_helpers_dataflow.hpp` | headers' @brief (`tile_column_dataflow.hpp:9`, `index_tile_dataflow.hpp:45`, `dfb_helpers_dataflow.hpp:9`) | sub-tile fills / index tiles / tile-dimension queries; none moves whole pages between placements |

`compute_kernel_hw_startup()`: not applicable — the op has no compute kernel.

## Key Risks and Gotchas

| Risk | Why it bites here | Mitigation in this design |
|------|-------------------|---------------------------|
| Shard→core order differs from the buffer's | multi-range grids (Qwen 2058: 16 single-core ranges; 2065: two ranges, 34 shards on 36 cores) and off-origin grids (2154 at core (1,0)) read back right only if placed by the buffer's own order | host derives the core list with the buffer's own function (`corerange_to_cores(grid, grid.num_cores(), row_wise=True)`; GRID_2D trim for block); an `ASSERT(acc.is_local_shard(shard_id))` in the `--dev` build |
| Ragged last shard slot stride | Qwen 2071/2079/2134/2170: last width shard holds 40 of 52 columns; Panoptic 7 last block-shard row 1 of 7 | slots use the full shard width (Mechanism caps); the walk covers only real pages |
| Kernel group is not rectangular | BB-17: sharded mover grids given as single-core lists | merge the mover grid into maximal rectangles; idle members get zero work |
| L1-interleaved page ownership | no host binding maps L1 bank → core | device-side discovery, gated by `l1_bank_owners` on host; when false the side has 0 holders and the regime function falls back (correct, slower) |
| NoC coordinate space | the pull kernel issues on NoC0, the push kernel on NoC1; bank discovery and any self-address must use the issuing NoC's coordinates (`local_copy_helpers_dataflow.hpp:48-51` warning) | pass the issuing NoC id to `bank_address`/`is_local` |
| Source and target on the same core | Qwen 2157/2158 (1-core he → 1-core bl), L1-interleaved ↔ L1-interleaved | NoC loopback read is the one move; never a raw RISC-V memcpy (would bypass the NoC ordering the barrier relies on) |
| In-flight transactions at return | BB-03: the next program starts right after | every kernel ends with its barrier, including idle cores (trivially satisfied) and the R3 writer (`noc_async_write_barrier`, not only `writes_flushed`) |
| Page size assumption | refinement pages are 1088 B and 576 B | every size from `aligned_page_size` (accessor / `buffer_aligned_page_size()`); no literal 2048 |
| Non-aligned H | 67 of 105 cases; leading dims fold per image (Qwen 2058 `[1,16,1,256]` = 16 tile rows) | `tensor_rows_tiles = prod(shape[:-2]) · ceil(H/32)`; whole tiles moved, padding rows ride along |
| Program cache key incomplete | BB-07: two placements with the same shape (e.g. 2067 vs 2163 grids) | key = every field that changes the program (see Host Contract); addresses are runtime args only |
| Returning the input or a wrong placement | values read back right either way | always allocate the output with `memory_config` and dispatch; never short-circuit identical placements |

## Host Contract (registry, validation, plan, caching)

### Registry declarations (op file `g6_to_memory_config.py`)

| Item | Value |
|------|-------|
| `INPUT_TAGGERS` (order) | `dtype → scenario["dtype"]`; `layout → scenario["layout"]`; `rank → int(len(scenario["shape"]))`; `alignment → "tile_aligned" if H % 32 == 0 and W % 32 == 0 else "h_non_aligned"`; `in_scheme`, `out_scheme → scenario["in_memory"/"out_memory"]["scheme"]`; `in_buffer`, `out_buffer → ...["buffer"]`. `inputs` is `(scenario,)`. |
| `SUPPORTED` (Phase 0) | `dtype [bfloat16]`, `layout [TILE_LAYOUT]`, `rank [2, 3, 4]`, `alignment ["tile_aligned", "h_non_aligned"]`, `in_scheme`/`out_scheme` all four, `in_buffer`/`out_buffer ["dram", "l1"]` |
| `EXCLUSIONS` (Phase 0) | exactly `[{"out_scheme": "width_sharded", "out_buffer": "dram"}]` |
| Exports of the package `ttnn.operations.g6_to_memory_config` | `g6_to_memory_config`, `select_plan`, `INPUT_TAGGERS`, `SUPPORTED`, `EXCLUSIONS` |

### `validate(input_tensor, memory_config)` — first line of the entry point

| Order | Check | Raises |
|-------|-------|--------|
| 1 | input is a device tensor (`ttnn.is_tensor_storage_on_device`) | `ValueError` |
| 2 | input layout is TILE | `ValueError` |
| 3 | rank in 2..4 | `ValueError` |
| 4 | every shard spec (input's and `memory_config`'s) has ROW_MAJOR orientation | `ValueError` |
| 5 | `memory_config` shard spec covers the padded tile grid: shard dims multiples of 32; height → shard width = padded W; width → shard height = `tensor_rows_tiles · 32`; `n_shards ≤ grid.num_cores()` (block: one range with ≥ `n_shard_cols` columns and ≥ `n_shard_rows` rows) | `ValueError` |
| 6 | build the scenario dict from the tensor and `memory_config` (`shape`, `dtype`, `layout`, `in_memory`/`out_memory` `buffer`+`scheme`, as `eval/golden_tests/g6_to_memory_config/axes.py:40-54` does), run `INPUT_TAGGERS` | — |
| 7 | SUPPORTED per axis | `UnsupportedAxisValue` |
| 8 | EXCLUSIONS per cell | `ExcludedCell` |
| 9 | BB-05: `out_l1_bytes_per_core + staging_bytes ≤ MODEL_L1_BUDGET_BYTES` (`out_l1_bytes_per_core` = `ceil(tensor_pages/num_l1_banks)·aligned_page` for L1 interleaved, `shard_rows_tiles·shard_cols_tiles·aligned_page` for L1 sharded, 0 for DRAM; `staging_bytes` = R3 CB or 0). Every golden shape passes (largest: Qwen 2078, 94 pages = 188 KiB/core). | `ValueError` (never an EXCLUSION, never a size tag) |

### Plan report — `select_plan(input_tensor, memory_config) -> G6Plan`

Frozen dataclass, pure host function, deterministic for a given placement pair and device
(BB-08); the entry point uses the same function.

| Field | Meaning |
|-------|---------|
| `regime` | `"pull"` / `"push"` / `"bounce"` from the selection function |
| `holders_in`, `holders_out` | `h_in`, `h_out` |
| `l1_bank_owners` | the L1-bank ownership predicate |
| `kernel_cores` | `ttnn.CoreRangeSet` per kernel group (merged rectangles) |
| `roles` | `{(x, y): role}` with role one of `"pull"`, `"push"`, `"bounce_reader+writer"`, `"idle"`, for every launched core (BB-21 role dump) |
| `bounce_block_pages`, `bounce_depth` | R3 knobs (`None` otherwise) |

### One dispatch, caching, zones

| Concern | Decision |
|---------|----------|
| Dispatch (BB-01) | exactly one `ttnn.generic_op([input_tensor, output_tensor], descriptor)` per call; no other device op, no host readback, no host-side padding/slicing/layout change |
| Descriptor cache (BB-07, BB-16) | module dict keyed by (device id, compute grid, logical + padded shape, dtype, input memory config, output memory config — buffer type, layout, shard grid ranges in order, shard shape, orientation). Value: plan + `ProgramDescriptor` with `custom_program_hash = hash(key)`. Per call only the two buffer base addresses change — keep them in runtime args (common runtime args where the kernel shares them) and patch them on the cached descriptor; nothing else is rebuilt. |
| Kernels (BB-18) | two small sources: a gather kernel (R1, R3 reader) and a scatter kernel (R2, R3 writer); local side (owned buffer vs staging slot) selected by a CT flag |
| Profiler zones (BB-21) | `G6-RESOLVE` (RT args + bank scan), `G6-ISSUE` (segment loop), `G6-DRAIN` (final barrier); R3 adds `G6-SLOT-WAIT` (reserve/wait) and `G6-BLOCK-BARRIER` (per-block read barrier / write flush). Zones compile out of the plain build. |
