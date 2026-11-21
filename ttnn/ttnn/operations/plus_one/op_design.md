# Operation Design: plus_one

## Overview

| Field | Value |
|-------|-------|
| Classification | data_movement (the only arithmetic is a 32-bit integer add done by the data-movement harts; no compute kernel) |
| Goal | Increment every 32-bit element of a row-major interleaved int32/uint32 tensor **in its own buffer**, wrapping; with `skip_negative_entries=True` increment only elements whose bits, read as int32, are in `[0, INT32_MAX)`. One `ttnn.generic_op` dispatch, multi-core, multi-thread. |
| Math | `x[i] <- (uint32)x[i] + 1u` for every element; with the flag: `x[i] <- ((uint32)x[i] < 0x7FFFFFFFu) ? (uint32)x[i] + 1u : x[i]` |
| References | Specification `eval/specs/plus_one/spec.py` (authoritative: DEFINITION, RULES R1–R5, PHASE0, GUIDANCE); target `.claude/references/targets/quasar.md`; backend `.claude/references/backends/ttsim.md`; host-program examples `.claude/references/quasar_op_examples/` (`toy_thread_plan/` host + explicit-sync kernels); scratchpad API `tt_metal/api/tt-metalium/workload/specification/scratchpad_spec.hpp:22-66`, `tt_metal/hw/inc/api/scratchpad.h:41-139`; Quasar scratchpad+TensorAccessor copy kernel `tests/tt_metal/tt_metal/data_movement/quasar_examples/quasar_addrgen/kernels/ta_api_copy.cpp:28-55`; in-place generic op pattern `ttnn/ttnn/operations/toy_spec_single_tensor/toy_spec_single_tensor.py` |

## Target

| Fact | Value | Source |
|------|-------|--------|
| Target | `quasar` on ttsim simulator (`libttsim.so`, slow dispatch, untimed) | `.claude/references/targets/quasar.md` + `.claude/references/backends/ttsim.md` |
| Arch | quasar | target page |
| Compute grid | 8 × 4 = 32 cores (clusters), fixed on this part; queried at run time with `device.compute_with_storage_grid_size()` | target page |
| L1 per core | 4,194,304 B (4 MiB), shared by the cluster's 4 NEOs and 8 DM harts | target page |
| Unreserved budget | `ttnn.get_max_worker_l1_unreserved_size() - L1_RESERVE_BYTES` (`L1_RESERVE_BYTES = 0`, see Blocking Model) = slightly under 4 MiB on this target (the query's value; never typed in) | target page, query column |
| Host program | `ttnn.ProgramSpec` via `ttnn.experimental` (`E.ProgramSpec`, `E.KernelSpec`, `E.ScratchpadSpec`, `E.TensorParameter`, `E.WorkUnitSpec`, `E.ProgramRunArgs`) dispatched by one `ttnn.generic_op([input_tensor], spec, run_args, {"t": 0})` — reference `.claude/references/quasar_op_examples/toy_thread_plan/toy_thread_plan_program_artifacts.py` | target page |
| Threads per core | 4 NEO compute engines (C ∈ {1,2,4}) and 6 user DM harts (R + W ≤ 6; R, W divide C on STRIDED rings). This op has **no ring**, so the divisibility rule does not apply; its plan is one DM kernel with `WORKER_THREADS` threads and C = 0 (Work Distribution) | target page *Thread plan* |
| Alignment | L1 16 B (`ttnn.get_l1_alignment()`), DRAM 64 B (`ttnn.get_dram_alignment()`); the Quasar NoC itself has **no** L1/DRAM transfer alignment restriction (`tt_metal/hw/inc/internal/tt-2xx/quasar/noc/noc_parameters.h:482-491`) | target page; noc_parameters.h |
| DM cache | DM harts read L1 through a shared, cacheable L2 with 64-B lines (`L2_CACHE_LINE_SIZE`, `tt_metal/hw/inc/internal/tt-2xx/quasar/overlay/overlay_addresses.h:44`); NoC writes into L1 do not invalidate it, and invalidation **discards dirty lines** (`tt_metal/hw/inc/internal/tt-2xx/risc_common.h:263-296`) | source |
| Unavailable here | (1) `ttnn.ProgramDescriptor` / Metal 1 kernels — refused at kernel creation (target page). (2) Legacy C++ device op with a program factory — same refusal (target page). (3) `bfloat8_b`, `bfloat4_b` — no path (irrelevant: op is int-only). (4) **uint32 as a buffer format**: a `DataflowBufferSpec`/`ScratchpadSpec` with `data_format_metadata=uint32` is refused (`tt_metal/common/tt_backend_api_types.cpp:96-122` lists Int32/RawUInt32, not UInt32) → the scratchpad here carries **no** format metadata. (5) A **DM kernel self-looping a DFB** (producer and consumer of one ring) is refused on Gen2 (`tt_metal/impl/metal2_host_api/program_spec/validation/resource/dfb.cpp:638-652`, message: "Consider using a scratchpad") → scratch L1 is a `ScratchpadSpec`. (6) **Untagged NoC transfers**: not used → every transfer of this op carries user trid 0. | target page; cited source |

All worked numbers below are for this target (32 cores, 4 MiB L1). Every predicate the op evaluates queries the device.

## Parameters

| Name | Type | Required | Valid Range | Default | CT/RT |
|------|------|----------|-------------|---------|-------|
| `input_tensor` | `ttnn.Tensor` | yes (positional) | int32 / uint32, ROW_MAJOR, interleaved, rank 1–4 (Phase 0: rank 1, DRAM) | — | tensor binding (accessor `t`), read **and** written |
| `skip_negative_entries` | `bool` | no (keyword-only) | `False`, `True` | `False` | CT (`skip_negative_entries` ∈ {0,1}) |

Signature exactly as the spec stub: `plus_one(input_tensor, *, skip_negative_entries=False) -> ttnn.Tensor`. No compute config (spec DECISIONS). No `sub_core_grids`.

## Tensors

### Input

| Property | Requirement |
|----------|-------------|
| Shape | any rank 1–4, `W = shape[-1]`, `rows = volume / W` (rank 1 → `rows = 1`) |
| Dtype | `ttnn.int32` or `ttnn.uint32` (both handled as raw 32-bit words) |
| Layout | ROW_MAJOR (one page = one row of `W` elements, `W*4` bytes; the buffer slot is padded to its alignment, the padding is not part of the tensor) |
| Memory | interleaved; Phase 0: DRAM; L1 is a later knob-turn (validate gate only) |

### Output

| Property | Value |
|----------|-------|
| Shape | same tensor |
| Dtype | same |
| Layout | same |
| Memory | **the input's own buffer** (R3): the op allocates nothing and returns the `input_tensor` object it was given |

### Validation (Python side, R4/R5)

| Check | Refusal |
|-------|---------|
| axes from `eval.memory.memory_axes(input_tensor)` (`memory_config`, `sharding`), `rank` from `INPUT_TAGGERS = AXES` (imported from the spec, never redefined), `dtype`, `layout`, `skip_negative_entries` gated against `SUPPORTED` (= `PHASE0`) | `NotImplementedError` naming the axis and value |
| any other dtype (no float form), TILE layout, rank > 4, sharded | `NotImplementedError` (falls out of the SUPPORTED gate) |

No index axes. No structural impossibilities to add to `INVALID`.

## Blocking Model

Semantics: `.claude/references/blocking-model.md`. The op is a read–modify–write of every element; every result depends on one element only.

### Axes

| Axis | Character (+ one-clause reason) | Extent knob | Phase 0 value | Knob source | Core-assignment | Later unlock |
|------|--------------------------------|-------------|---------------|-------------|-----------------|--------------|
| `row` — the page index, all leading dims flattened (`rows = volume / W`; rank 1 → 1) | **independent** — `x[i]+1` reads no other element, so rows compute in isolation | `block_units`: units of one thread's share driven through one read → increment → write pass | **the thread's whole share** (`ceil(max units per core / WORKER_THREADS)`) clamped to the per-thread scratch slot: `min(that, floor(max_block_bytes / slot_stride_bytes))` | host, derived once in `create_program_artifacts` from `WORKER_THREADS`, the split and `max_block_bytes`; passed to the kernel as one arg | units (row segments) dealt contiguously to cores by `ttnn.split_work_to_cores`, then round-robin to the core's `WORKER_THREADS` threads (Work Distribution) | knob-turn (more cores / threads, smaller blocks) |
| `col` — element within a row (`W`) | **independent** — same reason; a row may be cut anywhere on an alignment granule | `seg_elems`: elements per unit (row segment) | `W` (whole row) when `rows >= workers_wanted`; otherwise `align_up(ceil(W / ceil(workers_wanted / rows)), seg_granule_elems)`; always clamped to `≤ floor_to(max_block_bytes / 4, seg_granule_elems)` | host, derived once from `MIN_WORKER_BYTES`, the grid query, `WORKER_THREADS`, the buffer alignment query and `max_block_bytes` | segments of one row go to different workers when rows alone under-fill the wanted worker count (rank-1 Phase 0: the **only** way to use more than one core) | knob-turn |
| (operand reuse check) | the op has one operand; the constant `1` is an immediate. The input varies along both split axes, so **no operand is reuse-shared** by the split and no broadcast regime exists | — | — | — | — | — |
| (dependent axes) | **none** — no result spans any axis, so there is no combine and no cross-core communication of any kind | — | — | — | — | — |

Derived quantities (each from the single sources above, never restated):

| Quantity | Expression |
|----------|------------|
| `seg_granule_elems` | `buffer_alignment // 4`, `buffer_alignment = ttnn.get_dram_alignment()` for a DRAM tensor, `ttnn.get_l1_alignment()` for L1 (16 elems / 4 elems on this target) |
| `workers_wanted` | `clamp(ceil(rows * W * 4 / MIN_WORKER_BYTES), 1, grid_cores * WORKER_THREADS)` |
| `segs_per_row` | `ceil(W / seg_elems)` |
| `num_units` | `rows * segs_per_row` |
| `num_workers` | `min(workers_wanted, num_units)` |
| `num_cores` | `ceil(num_workers / WORKER_THREADS)` (≤ `grid_cores`) |
| unit `u` → (row, element offset, length) | `row = u // segs_per_row`, `off = (u % segs_per_row) * seg_elems`, `n = min(seg_elems, W - off)` (ragged last segment) |
| `slot_stride_bytes` | `align_up(seg_elems * 4, ttnn.get_l1_alignment())` |
| `max_block_bytes` | `floor_to((budget - CACHE_LINE_BYTES) / (WORKER_THREADS * SCRATCH_DEPTH), CACHE_LINE_BYTES)`, `budget = ttnn.get_max_worker_l1_unreserved_size() - L1_RESERVE_BYTES` |
| `region_stride_bytes` | `align_up(SCRATCH_DEPTH * block_units * slot_stride_bytes, CACHE_LINE_BYTES)` |
| scratchpad `size_per_node` | `WORKER_THREADS * region_stride_bytes + CACHE_LINE_BYTES` (slack to line-align the base, see Mechanism caps) |

Host constants (one named source each, in the program-artifacts module):

| Constant | Phase 0 value | Why |
|----------|---------------|-----|
| `WORKER_THREADS` | 4 | the thread plan (Work Distribution) |
| `MIN_WORKER_BYTES` | 1024 | grid-synchronization floor: below ~1 KiB a worker's time is its launch plus two DRAM round trips, not the increment, so another worker does not shorten the critical path. Perf lamp below. |
| `SCRATCH_DEPTH` | 1 | blocks resident per thread; threads supply the overlap (perf lamp) |
| `L1_RESERVE_BYTES` | 0 | the scratchpad is the program's only L1 allocation; the query already excludes firmware regions |
| `CACHE_LINE_BYTES` | 64 | mechanism cap (L2 line); Python has no query, cite `overlay_addresses.h:44` beside it |

Worked Phase 0 values on this target (grid 32, T = 4, DRAM tensors):

| Shape | bytes | workers_wanted | seg_elems | units | workers / cores | per-thread block |
|-------|-------|----------------|-----------|-------|-----------------|------------------|
| `(1,)` … `(100,)` | 4 … 400 | 1 | `W` | 1 | 1 / 1 | 1 unit, one row |
| `(4096,)` | 16 KiB | 16 | 256 (1 KiB) | 16 | 16 / 4 | 1 unit of 256 elems |
| later: `(512, 8)` | 16 KiB | 16 | 8 (whole row) | 512 | 16 / 4 | 32 rows (`block_units = 32`) |
| later: `(1,1,2,8192)` | 64 KiB | 64 | 256 | 64 | 64 / 16 | 1 unit |

### Buffer-depth knobs

| CB | Depth knob | Phase 0 value | What the depth buys |
|----|------------|---------------|---------------------|
| `scratch_units` (ScratchpadSpec, per-thread region) | `SCRATCH_DEPTH` | 1 | depth 2 would let one thread read block k+1 while incrementing / writing block k; at depth 1 the overlap comes from the core's other threads, whose phases are independent |

### Mechanism caps

| Mechanism | Cap on which extent | Clamp | What happens unclamped |
|-----------|--------------------|-------|------------------------|
| DM L2 cache, 64-B lines, invalidate discards dirty data (`risc_common.h:263-296`) | per-thread scratch region base and stride | every thread's region starts at a **64-B aligned L1 address** (`align_up(pad base address, 64) + t * region_stride_bytes`) and `region_stride_bytes` is a multiple of 64; scratchpad gets `+64` B slack because its base is only 16-B aligned | thread A's lock acquire invalidates a line holding thread B's not-yet-flushed increments → B's results silently lost |
| Buffer alignment granule (DRAM 64 B / L1 16 B, queried) | `seg_elems` when `segs_per_row > 1` | `seg_elems` a multiple of `seg_granule_elems`; only a row's last segment is ragged | two cores write parts of one alignment granule of a DRAM page; harmless on the Quasar NoC by its spec, but not proven for the DRAM controller — the clamp costs nothing |
| Per-thread scratch slot (`max_block_bytes`) | `seg_elems` and `block_units` | `seg_elems ≤ floor_to(max_block_bytes/4, granule)`; `block_units ≤ floor(max_block_bytes / slot_stride_bytes)`; a thread with more units loops over blocks | scratchpad exceeds L1 → program build fails |
| int32 arithmetic | element value | all arithmetic on `uint32_t` (wrap is defined); compare `x < 0x7FFFFFFFu` for the flag | signed overflow on `INT32_MAX + 1` is UB in C++ — the compiler may fold or trap; R1 fails |
| NoC transaction-id credits (`noc.h:259-264`: async_read busy-waits while fewer than half the trid's transactions are free) | reads issued before one barrier (`block_units`) | none needed — the API throttles; noted so nobody adds a manual cap | — |
| NoC burst (`NOC_MAX_BURST_SIZE`, `noc_parameters.h:395-396`) | bytes per transfer | none — the default `max_page_size` template splits larger transfers (`noc.h:246-251`) | — |

### Regimes

| Regime | Status | Predicate | Block | Data movement vs. minimum | What a bigger block buys |
|--------|--------|-----------|-------|---------------------------|--------------------------|
| **A. `segment_split_in_place`** — units = row segments (`seg_elems`, whole rows when rows suffice), dealt to `num_cores × WORKER_THREADS` workers; each thread reads its block of units from the tensor into its scratch region, increments in L1 with its own integer ALU, writes the same bytes back to the same page offsets | **built** | every cell of TARGET (int32/uint32, RM, DRAM or L1 interleaved, rank 1–4, both flag values); Phase 0 gates it to PHASE0. No other regime is needed there: no dependent axis (no combine), no shared operand (no multicast), and the data is not tile-shaped (no NEO path without a re-layout) | per thread: `block_units` units × `seg_elems` elements, one scratch region | Boundary = DRAM (the tensor's buffer). Minimum for an in-place read–modify–write: every element crosses DRAM once in and once out (2 × `rows·W·4` B). Actual: **exactly the minimum** — only tensor bytes move (transfers are exactly `n*4` B; page padding is never read or written). No cross-core traffic. | per block, per thread: one read barrier, one lock acquire (an L2 invalidate loop over the region), one lock release (an L2 flush loop), one write barrier. Intended frequency: **once per thread** in Phase 0 (whole share = one block); once per block only when a share exceeds the scratch slot |
| B. `single_worker_whole_tensor` (one core, one hart, whole tensor) | **rejected** — superseded by A, which **reduces to it** when `rows·W·4 ≤ MIN_WORKER_BYTES` (one worker) and spreads beyond it; building it separately would duplicate A | — | whole tensor | minimum | — |
| C. `neo_sfpu_int_add` — rows treated as 1024-element blocks through reader → ring → NEO integer add (SFPU `add_int`, `ttnn/cpp/ttnn/kernel_lib/eltwise/binary/sfpu/sfpu.inl:13`) → ring → writer | **deferred** — it pays only when a core's share is large enough that the DM increment loop, not DRAM, is the limit even with 4 harts per core, which is far above the op's purpose (positions and indices, ≤ ~100 elements; largest INPUTS entry 64 KiB); and int32 exactness through unpack→DEST→pack (R1 forbids any float pass) is unverified on Quasar, and uint32 is not a buffer format (Unavailable (4)) | would be `share_bytes_per_core > <measured crossover>` | 1024-elem blocks | minimum (same bytes) | LLK init amortization |
| | Reachability: A's units, split and in-place binding are unchanged; the worker's read half becomes a reader feeding an int32 ring, its write half a writer draining one, and the increment moves to a compute kernel between them. | | | | |
| D. `pipelined_reader_writer_ring` — split the worker into an incrementing reader (explicit-sync producer) and an implicit-sync writer on a DM→DM ring, so one unit's write overlaps the next unit's read inside a thread pair | **deferred** — the in-core overlap it buys is already supplied by `WORKER_THREADS` independent threads, and the explicit-producer / implicit-consumer DM→DM mix is untested on this base (`tests/tt_metal/tt_metal/api/dataflow_buffer/dfb_test_common.hpp:162-165` always uses one protocol on both sides) | — | same units | minimum | — |
| | Reachability: the unit list per thread is unchanged; only the transport between read and write changes. | | | | |
| E. `l1_bank_local_in_place` — for an **L1**-interleaved tensor, the core whose L1 bank holds a page increments it where it lies (zero NoC bytes, no scratch) | **deferred** — L1 placement is not in PHASE0; when it is, A already serves it as a knob-turn (the accessor addresses L1 banks; granule becomes the L1 alignment). E is the perf successor: it needs units assigned by bank owner instead of contiguously | `memory_config == "L1"` | pages of the core's own bank | below the DRAM minimum (no NoC bytes) | — |
| | Reachability: A's unit → (row, offset) mapping is unchanged; only the core-assignment function changes (bank owner instead of contiguous range) and the read/write become local L1 accesses. | | | | |

Regime-selection function: one built regime, so none. Pinned tests still cover both shapes of A's split (whole-row units and multi-segment rows with a ragged last segment): Work Distribution.

### Traffic ranking

| Candidate split | DRAM bytes | NoC transfers | Cross-core | Verdict |
|-----------------|-----------|---------------|------------|---------|
| `row` only (whole-row units) | 2 × tensor | 2 per row (fewest) | none | cheapest; used whenever `rows ≥ workers_wanted` |
| `row` + `col` (row segments) | 2 × tensor (same) | 2 per segment: `2·rows·(segs_per_row − 1)` more than row-only | none | same bytes; extra transfers only where rows alone under-fill the wanted workers (rank-1 Phase 0) — **chosen there**, it is the only parallelism a single-row tensor has |
| `col` only | 2 × tensor | as above, for every row | none | dominated by the two rows above (adds transfers where rows already fill) |
| split a dependent axis + combine | — | — | — | does not exist: no dependent axis |
| broadcast a shared operand | — | — | — | does not exist: no shared operand |

All splits move the same bytes across DRAM (the in-place minimum); the ranking is decided by transfer count, and the implemented split (row-first, segments only when rows under-fill) is the cheapest. For rank-1 Phase 0 tensors the whole tensor is one page in one DRAM bank, so every segment hits the same bank: inherent in the placement, not in the split.

### Block schedule

Per thread `t` of the worker kernel on a core owning units `[unit_start, unit_start + unit_count)`; the thread's units are `unit_start + t, + t + T, …` (T = `WORKER_THREADS`):

```cpp
for (uint32_t block_idx = 0; block_idx < num_blocks_this_thread; ++block_idx) {
    load_block(block_idx);       // ≤ block_units tagged reads (trid 0), page `row`, offset `off*4`, `n*4` bytes,
                                 // into this thread's region, slot k at k*slot_stride_bytes; ONE read barrier
    increment_block(block_idx);  // ONE scoped_lock over the region (L2 invalidate), add 1 (or the flagged add)
                                 // to the n words of every slot as uint32, lock released (L2 flush)
    store_block(block_idx);      // the same ≤ block_units tagged writes back to the same page offsets; ONE write barrier
}
// last statements: hand the hart's tag back (read and write) as user id 0
```

| Operation | Block shape | Resident across it | Intended frequency |
|-----------|-------------|--------------------|--------------------|
| `load_block` | `block_units × seg_elems` (ragged last segment `n`) | this thread's scratch region | 1 read barrier per block (Phase 0: once per thread); reads are issued back-to-back before it |
| `increment_block` | same | same region, now cache-visible | 1 lock acquire + 1 release per block (never per unit or per element) |
| `store_block` | same | region held until the write barrier returns, then reused by the next block | 1 write barrier per block |

### Perf lamps

| Lamp | Why the default may be wrong here | Nearby alternative to measure |
|------|-----------------------------------|-------------------------------|
| Grid synchronization: `MIN_WORKER_BYTES = 1024` | the floor trades occupancy for per-worker fixed cost (launch, runtime args, two DRAM round trips); the right crossover is unmeasured, and slow dispatch on ttsim inflates per-core launch | `MIN_WORKER_BYTES = buffer_alignment` (max occupancy: 64 B per worker) and `= ∞` (one worker, tt-metal's scheme) |
| Threads-first dealing (whole rounds of `WORKER_THREADS` units per core) | with no ring there is no padding penalty for spreading cores first; fewer cores means fewer launches, more cores means more NoC endpoints | cores-first: `num_cores = min(grid, num_workers)`, one unit per core before any second thread is used |
| Thread count `WORKER_THREADS = 4` | the increment loop is per-hart work; 6 user harts exist and nothing else in the program needs one | `WORKER_THREADS = 1` and `= 6` (one constant) |
| Overlap: one block per thread, `SCRATCH_DEPTH = 1` | a thread's read, increment and write are serial; at large shares the threads of a core all read at once, then all increment, then all write | `SCRATCH_DEPTH = 2` with `block_units` capped to a few KiB per block |
| Shared trid 0 across the core's threads | if the trid counters are shared between harts, one thread's barrier also waits for the others' transfers (correct, slower) | per-thread user trid (`1 + t`), with the tag handed back as 0 at the end |
| Writes on the reader's NoC | the catalog's `noc_placement` measured reads on NoC0 / writes on NoC1 as the faster pairing for spread placements (WH) | issue `store_block` on the other NoC, if a Quasar DM hart can address it |

Catalog entries that informed the knobs (`ttnn/ttnn/operations/examples/master.md`): `width_split` (split a wide, short tensor along its width — the `col` segmentation for rank-1), `double_buffer` (issue a block of reads, then one barrier — `load_block`/`store_block`; "no gain once DRAM-bound"), `read_coalescing` (a no-compute RM DM op; whole transfers, not fragments), `tile_reorder` (move whole pages; do not fragment below the page when not needed — row-first split), `noc_placement` (`row_wise=True` in `split_work_to_cores`; write-NoC lamp).

## Dataflow Strategy

| Stage | Format | Mechanism | Notes |
|-------|--------|-----------|-------|
| DRAM page `row`, bytes `[off*4, (off+n)*4)` → scratch slot | raw 32-bit words (int32 or uint32, identical handling) | `noc.async_read<NocOptions::TXN_ID>(acc_t, pad, n*4, {.page_id = row, .offset_bytes = off*4}, {.offset_bytes = slot_offset}, {.trid = 0})` (`tt_metal/hw/inc/api/dataflow/noc.h:246-277`, accessor args `tt_metal/hw/inc/api/tensor/noc_traits.h:135-143`, scratchpad dst `tt_metal/hw/inc/internal/local_mem/noc_traits.h:73-95`) | explicit protocol: tagged, one `async_read_barrier<TXN_ID>({.trid=0})` per block (`noc.h:739`) |
| scratch slot → scratch slot | raw words | DM hart integer ALU inside `pad.scoped_lock(elem_offset, elem_count)` (`tt_metal/hw/inc/api/scratchpad.h:136-139` → `internal/scoped_lock_cache_ops.h`: invalidate on acquire, flush on release) | `uint32_t` arithmetic only; the lock's scope closes **before** `store_block` |
| scratch slot → the same DRAM bytes | raw words | `noc.async_write<NocOptions::TXN_ID>(pad, acc_t, n*4, {.offset_bytes = slot_offset}, {.page_id = row, .offset_bytes = off*4}, {.trid = 0})` (`noc.h:411`) | one `async_write_barrier<TXN_ID>({.trid=0})` per block (`noc.h:760`) before the region is reused or the kernel returns |
| tag hand-back | — | `noc_async_read_set_trid(0, noc.get_noc_id())`, `noc_async_write_set_trid(0, noc.get_noc_id())` (`tt_metal/hw/inc/api/dataflow/dataflow_api.h:2558,2579`) as the kernel's last statements | all transfers already use trid 0; kept so the hand-back is explicit for tt-metal programs that follow |

Tensix-to-Tensix contract: none in any built or deferred regime (no combine, no multicast).

TARGET placement axes as regimes:

| TARGET value | Cost class | How |
|--------------|-----------|-----|
| `memory_config: DRAM` | built (Phase 0) | regime A |
| `memory_config: L1` (interleaved) | **knob-turn** | regime A unchanged: the TensorAccessor addresses L1 banks; `seg_granule_elems` from `ttnn.get_l1_alignment()`; validate gate widened. Regime E is its deferred perf successor |
| `sharding: interleaved` | built | the only value in TARGET; sharded tensors are refused by name (R4) |
| `rank: 2, 3, 4` | **knob-turn** | `rows = volume / W` already folds leading dims; whole-row units appear when `rows ≥ workers_wanted`; validate gate widened |

## Work Distribution

| Field | Value |
|-------|-------|
| Work unit | a row segment (`seg_elems` elements of one page; a whole row when rows suffice); a block is up to `block_units` of one thread's units |
| Grid | `device.compute_with_storage_grid_size()` (8×4 on this target); engaged cores `ttnn.num_cores_to_corerangeset(num_cores, grid, row_wise=True)` |
| Per-core work | `ttnn.split_work_to_cores(engaged_core_set, num_units, row_wise=True)` → two groups; each core gets runtime args `unit_start`, `unit_count` (contiguous range). Inside the core thread `t` takes units `unit_start + t + k·WORKER_THREADS`. A core is never launched with zero units |
| Remainder | `ceil` everywhere: `segs_per_row = ceil(W/seg_elems)`, last segment `n = W − off`; `rows = volume / W` from the **logical** shape (RM has no tile padding; per-image `ceil(H/32)` does not apply). `split_work_to_cores` handles the unit remainder (group 2); a core's `unit_count` need not divide by `WORKER_THREADS` — a thread with fewer (or zero) units does fewer (or no) blocks; there is no ring, so no padding round is needed |
| Bound | **compute on the DM harts**: per 4-byte word a hart executes a load, a compare, an add and a store from L1, while the NoC moves 256 B per cycle per link; at the 1-KiB floor the increment work is already the same order as the two DRAM round trips and it grows with the share while the transfer time barely does. NEOs are not used (no tile layout; R1 forbids a float pass; Regime C) |
| Thread plan | one DM kernel `worker` with `num_threads = WORKER_THREADS = 4`, C = 0 NEOs, no separate reader/writer: each thread is its own reader, incrementer and writer for its units. Host constant `WORKER_THREADS`; passed as a compile-time arg, and the kernel strides by it (never by `get_num_threads()`), so a missing thread leaves units unincremented. `hw_config = E.create_reader_datamovement_config()`, no `disable_dfb_implicit_sync_for` (no DFB exists) |
| Unit dealt to threads | the work unit itself (a row segment). Threads take disjoint unit lists and disjoint, line-aligned scratch regions; no thread touches another's units or region. The host deals whole rounds (`num_cores = ceil(num_workers / WORKER_THREADS)`), so only the last core can be short of a round |
| Tests the plan needs | (1) **every thread ran**: a probe option of the program builder (not of the public signature), `probe_tag_threads`, makes thread `t` add `1 + t`; on `(4096,)` (16 units, 4 cores × 4 threads) the test recomputes each unit's owner thread from the host split function and checks every element, and asserts all four thread ids occur. (2) **one device session**: plus_one run after a tt-metal program and before one: `ttnn.to_layout(<bfloat16 TILE tensor>, ttnn.ROW_MAJOR_LAYOUT)` (an untilize, explicit sync, untagged) → plus_one → the same untilize again → plus_one a second time on the same tensor (expects `x + 2`), all checked, in one test. (3) **pinned split shapes**: `(4096,)` (multi-core, 16 segments), `(100,)` (one worker), and `(100,)` with the builder's `min_worker_bytes=64` probe (7 segments of 16, ragged last of 4) — run with the flag both ways and with EDGE values at segment boundaries |

## Circular Buffers

No DataflowBuffer exists (a DM self-loop is refused on Gen2, and the op has one kernel). The only L1 buffer is a scratchpad.

| Semantic Name | Index | Page Size | Num Pages | Sizing rationale | Format | Producer | Consumer | Sync | Lifetime |
|---------------|-------|-----------|-----------|------------------|--------|----------|----------|------|-----------|
| `scratch_units` (`E.ScratchpadSpec(unique_id="scratch_units", size_per_node=…)`, bound as accessor `scratch_units`; kernel view `Scratchpad<uint32_t>`) | — (scratchpad, no ring index) | `slot_stride_bytes` per slot | `WORKER_THREADS × SCRATCH_DEPTH × block_units` slots, laid out as `WORKER_THREADS` regions of `region_stride_bytes`, + 64 B slack: `size_per_node = WORKER_THREADS·region_stride_bytes + CACHE_LINE_BYTES` | spans `row` → `block_units` per thread × `WORKER_THREADS`; spans `col` → `seg_elems` per slot; never an op dimension | none (`data_format_metadata` omitted: uint32 is refused, DM-only use needs none) | `worker` (NoC reads, thread `t` into region `t` only) | `worker` (increment, then NoC writes, same thread, same region) | explicit: user trid 0 on every transfer + one barrier per block; cache via `scoped_lock` | the whole program; each region reused block after block |

One kernel both fills and drains `scratch_units`: legal for a scratchpad (no credit machinery; `scratchpad_spec.hpp:22-66`), and race-free because each thread owns one line-aligned region exclusively and orders read barrier → lock → increment → unlock → write → write barrier itself.

## Block Operation Realization

| # | Block operation | Block shape | Helper? | Input (semantic name, pages, state) | Output (semantic name, pages) | State after |
|---|-----------------|-------------|---------|------------------------------------|-------------------------------|-------------|
| 1 | `load_block` | `≤ block_units × seg_elems` | no (raw NoC API; no helper fits, see API Mapping) | `input_tensor` pages via accessor `t` | `scratch_units` region `t`, `≤ block_units` slots | slots hold the original words in L1, not yet visible to the hart's cache |
| 2 | `increment_block` | same | no | `scratch_units` region `t` | `scratch_units` region `t` (in place) | incremented words flushed to L1 at lock release |
| 3 | `store_block` | same | no | `scratch_units` region `t` | `input_tensor` pages via accessor `t` (same offsets) | DRAM holds the result; region free for the next block after the write barrier |

## API Mapping

| Block operation | Type | Function | File:Line | Template Params / Args | Input | Output | Which params are block knobs |
|-----------------|------|----------|-----------|------------------------|-------|--------|------------------------------|
| host program | host | `E.ProgramSpec`, `E.KernelSpec(num_threads=WORKER_THREADS, scratchpad_bindings=[E.ScratchpadBinding("scratch_units","scratch_units")], tensor_bindings=[E.TensorBinding("t","t")])`, `E.ScratchpadSpec`, `E.TensorParameter("t", input_tensor.spec)`, `E.WorkUnitSpec(kernels=["worker"], target_nodes=engaged_core_set)` | `ttnn/cpp/ttnn-nanobind/program_spec.cpp:364-398` (ScratchpadSpec), `:704-718` (binding); DM-only WorkUnit legal: `tt_metal/impl/metal2_host_api/program_spec/validation/placement/work_unit.cpp:131-144`; example `.claude/references/quasar_op_examples/toy_thread_plan/toy_thread_plan_program_artifacts.py:72-158` | — | — | — | `num_threads`, `size_per_node` |
| dispatch | host | `ttnn.generic_op([input_tensor], spec, run_args, {"t": 0})`; return the **`input_tensor` object** | `ttnn/cpp/ttnn/operations/generic/generic_op_nanobind.cpp:34-82` (single-element list = in-place; returns the last tensor) | — | — | — | — |
| split | host | `ttnn.num_cores_to_corerangeset`, `ttnn.split_work_to_cores(core_set, num_units, row_wise=True)` | `ttnn/ttnn/core.py:54-64`; `ttnn/cpp/ttnn-nanobind/operations/core.cpp:504-509` | — | — | — | `num_units` (from `seg_elems`) |
| `load_block` | raw_api | `Noc::async_read<NocOptions::TXN_ID>` + `async_read_barrier<NocOptions::TXN_ID>` | `tt_metal/hw/inc/api/dataflow/noc.h:246-277,739`; accessor offset args `tt_metal/hw/inc/api/tensor/noc_traits.h:135-143`; scratchpad endpoint `tt_metal/hw/inc/internal/local_mem/noc_traits.h:73-95` | `size = n*4`, `{.page_id=row,.offset_bytes=off*4}`, `{.offset_bytes=slot_offset}`, `{.trid=0}` | `input_tensor` | `scratch_units` | `block_units` (reads per barrier), `seg_elems` (`n`) |
| `increment_block` | raw_api | `Scratchpad<uint32_t>::scoped_lock(offset, num_elements)` + `operator[]` | `tt_metal/hw/inc/api/scratchpad.h:68-72,136-139`; cache ops `tt_metal/hw/inc/internal/scoped_lock_cache_ops.h` | lock spans the thread's whole block (`block_units·slot_stride_bytes/4` elements) | `scratch_units` | `scratch_units` | `block_units`, `seg_elems` |
| `store_block` | raw_api | `Noc::async_write<NocOptions::TXN_ID>` + `async_write_barrier<NocOptions::TXN_ID>` | `tt_metal/hw/inc/api/dataflow/noc.h:411,760` | mirror of `load_block` | `scratch_units` | `input_tensor` | same |
| tag hand-back | raw_api | `noc_async_read_set_trid(0, noc.get_noc_id())`, `noc_async_write_set_trid(0, noc.get_noc_id())` | `tt_metal/hw/inc/api/dataflow/dataflow_api.h:2558,2579` | — | — | — | — |

**Helpers considered and rejected** (all three raw entries share the reason set):

| Candidate | File:Line of the mismatch | Why it cannot express this op |
|-----------|---------------------------|-------------------------------|
| `eltwise` convenience / chain (`add`, `unary`, `binary_sfpu<AddInt…>`) | `ttnn/cpp/ttnn/kernel_lib/eltwise/api/convenience.hpp:34-36` (built on `chain.hpp` + `dfb_helpers_compute.hpp`: TRISC compute API on DFB tiles); int add `ttnn/cpp/ttnn/kernel_lib/eltwise/binary/sfpu/sfpu.inl:13` (`api/compute/add_int_sfpu.h`) | compute-kernel helpers: they need a NEO kernel fed by rings through unpack→DEST→pack. This op has no compute kernel (Regime C deferred): the increment runs on the DM hart that already holds the words, which no helper offers |
| `local_copy_helpers_dataflow.hpp` | `ttnn/cpp/ttnn/kernel_lib/local_copy_helpers_dataflow.hpp:8-34` | L1→L1 self-aimed copies into a typed DFB; the op moves DRAM↔scratchpad and has no DFB |
| `dfb_helpers_dataflow.hpp` | (whole header: DFB reserve/push helpers) | no DFB exists (Gen2 refuses a DM self-loop, `dfb.cpp:638-652`) |
| `l1_helpers.hpp` `addr_to_l1_ptr` | `ttnn/cpp/ttnn/kernel_lib/l1_helpers.hpp:30` (`addr_to_l1_ptr`) | a raw volatile pointer bypasses the scratchpad's `scoped_lock` cache maintenance, which this design requires (Mechanism caps) |
| tilize / untilize / reduce / matmul / mcast | — | no tiles, no reduction, no product, no shared operand |

`compute_kernel_hw_startup()`: not applicable — no compute kernel. `fpu_math_fidelity`: not applicable (no compute `KernelSpec`).

## Key Risks and Gotchas

| Risk | Why it bites here | Mitigation in this design |
|------|-------------------|---------------------------|
| Stale or lost words through the DM L2 cache | NoC reads land in L1 behind the hart's cache; the hart's stores sit in the cache until flushed; invalidation discards dirty lines | every hart access to scratch inside **one** `scoped_lock` per block, acquired after the read barrier and released (scope closed) before the first write; thread regions 64-B line-aligned by address with 64 B slack |
| Signed overflow | `INT32_MAX + 1` is UB on `int32_t` | `Scratchpad<uint32_t>`, `x + 1u`, flag test `x < 0x7FFFFFFFu` (equivalent to int32 view in `[0, INT32_MAX)`, also for uint32 tensors ≥ 2^31−1, R2) |
| uint32 as a buffer format | refused on Quasar | scratchpad without `data_format_metadata`; the tensor's own dtype is untouched (tensor parameters are not format-checked: `tt_metal/impl/metal2_host_api/program_spec/validation/resource/tensor.cpp:25-37`: only accessor names are checked) |
| Sticky NoC tag | an untagged transfer reuses the hart's last tag | every transfer `NocOptions::TXN_ID` with `{.trid = 0}`, barriers on the same id, tag handed back as 0 at the end; session test required |
| Double increment or lost update | two workers touching one byte | units partition the elements exactly (contiguous per core, round-robin per thread); segment boundaries on the alignment granule; each core writes only bytes it read |
| Writing page padding | GUIDANCE: padding is not part of the tensor | transfers are exactly `n*4` bytes from offset `off*4`; the slot's padding to `slot_stride_bytes` stays in L1 |
| R3 in-place identity | the check accepts the object or a same-buffer handle | `io_tensors=[input_tensor]`, no `allocate_tensor_on_device`, return `input_tensor` itself |
| Rank-1 shapes | `rows` must come from the logical shape | `W = shape[-1]`, `rows = volume // W`; assert `rows == input_tensor.buffer_num_pages()` on host |
| Threads ran? | a correct result does not prove four threads worked | loops stride by the CT `WORKER_THREADS`; `probe_tag_threads` test |
| Recompiles per shape on a slow simulator | shape-derived values as compile-time args recompile per shape | only `WORKER_THREADS` and `skip_negative_entries` are compile-time; geometry (`W`, `seg_elems`, `segs_per_row`, `slot_stride_bytes`, `region_stride_bytes`, `block_units`) is runtime (common or per-core), `unit_start`/`unit_count` per core |
| Test output hidden under `--forked` workers | ttsim backend page | measurements and probes through `scripts/tt-probe.sh` or `record_property` |
