# Operation Design: clone

## Overview

| Field | Value |
|-------|-------|
| Classification | data_movement (no compute kernel) |
| Goal | A new interleaved tensor in the buffer `memory_config` names (the input's buffer when `None`) that holds the input's pages byte for byte: same shape, dtype, layout, values. |
| Math | `output[i] = input[i]` for every element (and every padding byte of every page); nothing is converted (R1) |
| References | Specification `eval/specs/clone/spec.py` (governs). Host pattern: `.claude/references/quasar_op_examples/toy_thread_plan/` (`toy_thread_plan_program_artifacts.py`, `kernels/reader_implicit.cpp`, `kernels/writer_implicit.cpp`). DM→DM ring with implicit sync on both ends, in tt-metal's own test: `tests/tt_metal/tt_metal/data_movement/quasar_examples/quasar_addrgen/test_tensor_accessor_addrgen.cpp:1964-2015` (`TensorAccessorAddrgenThreads.CopiesThroughDfb`) with `kernels/ta_mt_reader_to_dfb.cpp`, `kernels/ta_mt_writer_from_dfb.cpp`. Ring semantics: `tt_metal/impl/dataflow_buffer/DataflowBuffer.md` Part A. |

## Target

| Fact | Value | Source |
|------|-------|--------|
| Target | `quasar` on the ttsim simulator | `.claude/references/targets/quasar.md` + `.claude/references/backends/ttsim.md` |
| Arch | quasar | target page |
| Compute grid | 8×4 = 32 clusters (worked numbers below use 32); queried at run time with `device.compute_with_storage_grid_size()` | target page |
| L1 per core | 4,194,304 B (4 MiB) per cluster | target page |
| Unreserved budget | `ttnn.get_max_worker_l1_unreserved_size() - l1_reserve_bytes`. The page does not give the value. This design evaluates no budget predicate: the only buffer is capped by the DFB ring limit (1,048,560 B), far below the budget. | target page (value: unknown, queried) |
| Host program | `ttnn.ProgramSpec` (Metal 2.0, `ttnn.experimental.*`, one `ttnn.generic_op` call). Reference: `.claude/references/quasar_op_examples/` | target page |
| Threads per core | 4 NEO compute engines, 6 user DM harts (DM0 and DM1 reserved), R + W ≤ 6. On a STRIDED DM→DM ring this design requires R = W (one tile counter per producer/consumer thread pair, `DataflowBuffer.md:251-256`). C does not apply: there is no compute kernel. | target page *Thread plan*; `DataflowBuffer.md` |
| Unavailable here | `ttnn.ProgramDescriptor` / Metal 1 kernels and the legacy C++ device op with a program factory: refused at kernel creation (`tt_metal/impl/kernels/kernel.hpp`). `bfloat8_b`, `bfloat4_b`: no path on this part (not in TARGET anyway). An untagged NoC transfer after a tagged one: counted against a ring, which hangs or fails with `tile counter overflowed capacity`. So every kernel hands its trid back. The library-gaps entries (`quasar-library-gaps.md`: reduce Accurate, AccumulateViaAdd, MAX+REDUCE_ROW Accumulate, reduce post-op, tf32 reduce, reduce indexed access under a plan, compute DPRINT) are all compute-side, and this op plans no compute, so none of them is planned. An implicit-sync transfer larger than one NoC packet (65,536 B, `noc_parameters.h:395-396`): the ISR would fire on the first fragment (`dataflow_buffer.cpp:560`), so pages above that are refused, see Mechanism caps. | target page, library-gaps reference, cited lines |

## Parameters

| Name | Type | Required | Valid Range | Default | CT/RT |
|------|------|----------|-------------|---------|-------|
| `input` | `ttnn.Tensor` | yes | on device, interleaved, DRAM or L1, bfloat16 (Phase 0) / float32 (TARGET), TILE or ROW_MAJOR, rank 2–4 | — | tensor binding |
| `dtype` (keyword-only) | `ttnn.DataType` or `None` | no | `None` or `input.dtype`. Read into the `dtype_argument` axis with the spec's `canonical_dtype`. Any other dtype is refused (R2/X2). | `None` | host only |
| `memory_config` (keyword-only) | `ttnn.MemoryConfig` or `None` | no | `None`, or an interleaved DRAM or L1 config. Read into the `output_memory` axis with the spec's `canonical_memory`, which raises `TypeError` for a non-config such as the word `"L1"` (X1). A sharded config reads as `"<buf>/<sharding>"` and is refused by name (R7). | `None` | host only |

The entry point's signature equals the specification's stub exactly: `clone(input, *, dtype=None, memory_config=None)`. The op has no index axis to canonicalize.

**Host-side order of operations** (all before any device work):

1. Read `memory_config` into `output_memory` with `canonical_memory`. This raises `TypeError` for a non-config (X1).
2. Read `dtype` into `dtype_argument` with `canonical_dtype`.
3. Build the axes: `dtype`, `layout`, then `memory_config` / `sharding` from `eval.memory.memory_axes(input)` (R7), then `dtype_argument`, `output_memory`, then `INPUT_TAGGERS = AXES` (`alignment`, `rank`).
4. Run `validate()` against SUPPORTED/EXCLUSIONS, raising `NotImplementedError` (R8). A `dtype` other than the input's lands here, since no cell has that value (X2).
5. Check the page-size cap (Mechanism caps).
6. Allocate the output (R3, R5).
7. Build the ProgramSpec and dispatch once.

## Tensors

### Input

| Property | Requirement |
|----------|-------------|
| Shape | rank 2, 3 or 4, any H and W (`alignment` ∈ tile_aligned / w_non_aligned / h_non_aligned) |
| Dtype | bfloat16 (Phase 0); float32 in TARGET |
| Layout | TILE or ROW_MAJOR (both in Phase 0) |
| Memory | interleaved, DRAM or L1 (`memory_config` axis from `eval.memory.memory_axes`) |

### Output

| Property | Value |
|----------|-------|
| Shape | `input.shape` (logical), with the same tile padding: built as `ttnn.TensorSpec(input.shape, input.dtype, input.layout, out_mem, tile=input.tile)` (`ttnn/cpp/ttnn-nanobind/tensor.cpp:289-305`) |
| Dtype | `input.dtype` |
| Layout | `input.layout` |
| Memory | `out_mem` = `memory_config` when given, else `ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, input.memory_config().buffer_type)`. Always interleaved (R3). Always a fresh allocation via `ttnn.allocate_tensor_on_device(spec, device)`, also when `out_mem` names the input's buffer. The op never returns the input (R5). |

Host assertions after allocation: `output.buffer_num_pages() == input.buffer_num_pages()` and `output.buffer_page_size() == input.buffer_page_size()`. The copy is page for page, so equal page geometry is a precondition, not a hope.

## Blocking Model

Semantics: `.claude/references/blocking-model.md`.

The op's data is one buffer of pages, and clone maps page `p` of the input to page `p` of the output. A page is a 32×32 tile in TILE layout and one W-element row (a "stick") in ROW_MAJOR. Every tensor axis is independent: no output element depends on more than the one input element at its index. Pages are contiguous in buffer order, and the op treats every leading axis the same way. So the leading axes fold into one **page axis**, and the within-page axis is the transfer unit (`blocking-model.md` §1, "leading axes fold into one when the op treats them identically and they are contiguous").

### Axes

| Axis | Character (+ one-clause reason) | Extent knob | Phase 0 value | Knob source | Core-assignment | Later unlock |
|------|--------------------------------|-------------|---------------|-------------|-----------------|--------------|
| `outer` (product of dims before −2: N·C, or B, or 1) | independent: each output element reads only its own index | folded into `page` | folded | — | via `page` | — |
| `rows` (H; TILE: `ceil(H/32)` tile-rows per image; RM: H sticks per image) | independent: same reason | folded into `page` | folded | — | via `page` | — |
| `cols` in TILE (`ceil(W/32)` tile-columns) | independent: same reason | folded into `page` | folded | — | via `page` | — |
| **`page`** (flattened, buffer order: TILE `tensor_pages = outer·ceil(H/32)·ceil(W/32)`; RM `tensor_pages = outer·H`) | independent: page p of the output is page p of the input, with no cross-page combine | `core_pages`: the block is this core's **whole** contiguous page range, driven through one uninterrupted pipeline pass (`num_blocks_this_core = 1`) | the `ttnn.split_work_to_cores` share, e.g. 25 tiles for `[1,1,1600,512]` TILE on 32 clusters | host split: `(start_page, core_pages)` per core as runtime args. `tensor_pages = input.buffer_num_pages()` is the single source of the total. | contiguous page ranges over the **full grid**, `row_wise=True` (catalog `noc_placement`). Consecutive cores start on different banks (natural `bank_stagger`). | knob-turn (grid size, split units). Bank-affine assignment is the `owner_local` regime (scheme-change). |
| **`in_page`** (TILE: the 32×32 tile interior; RM: the W-element stick, W·elem bytes, including the W axis of RM) | independent, but **never split**: a page is one NoC transaction and one ring entry | `entry_bytes = align_up(page_bytes, ttnn.get_l1_alignment())` with `page_bytes = input.buffer_page_size()` | TILE bf16: 2,048 B. RM bf16: `align_up(2·W, 16)`, e.g. W=17 → 48 B, W=8192 → 16,384 B. | host, from `page_bytes` | whole extent, not assigned across cores: splitting a page into sub-page transfers multiplies transactions for the same bytes (catalog `tile_reorder`), and an implicit-sync STRIDED entry is exactly one transaction (Mechanism caps) | scheme-change (`oversized_page` regime) only for pages above one NoC packet |

**Every stage's own data.** There is one stage, `load → store`, and its only intermediate is the staged page in the ring. That data has the same `page`/`in_page` axes with the same characters. Its `page` axis streams through the ring window and is never assigned separately. No core collects, coordinates or hands out work: every cluster owns its range end to end, so no stage runs on one core while others wait.

**Bandwidth ranking.** See *Traffic ranking*.

**Operand-reuse check.** There is one input operand, and it varies along the split axis (`page`): each core reads disjoint pages. Nothing is reuse-shared, so there is no broadcast regime.

**Stall-shadow check.** The writer waits on the reader's entries, and the reader waits on NoC/DRAM latency. The work independent of a waited-for page is the next pages' reads. The ring depth puts that work into the stall: up to `ring_entries_per_thread·R` reads in flight while earlier entries drain, and implicit sync splits the ring into ≥ 2 transaction-id slices so credit updates overlap the next slice's traffic (`DataflowBuffer.md`, Part A "Implicit sync (Quasar DM)", *Double buffering*). There is no arithmetic, so no reorder needs an algebraic permission and no floating-point order changes.

**Committed scheme (Phase 0).** Each engaged cluster copies one contiguous page range of the input to the same range of the output:

- a reader kernel (R threads) issues one implicit-sync tagged read per page into the STRIDED ring `cb_staged_pages`;
- a writer kernel (W = R threads) issues one implicit-sync tagged write per entry to the output page.

The pages are spread over the full grid from day 1. Every knob is a parameter defined once on the host: `THREAD_PLAN`, `ring_entries_per_thread`, the grid (queried), `core_pages`/`start_page` (from the split) and `entry_bytes` (derived). None is an inlined constant. The kernels loop `for (i = get_my_thread_id(); i < core_pages; i += R)` with `core_pages` a runtime arg and R a compile-time arg.

### Buffer-depth knobs

| CB | Depth knob | Phase 0 value | What the depth buys |
|----|------------|---------------|---------------------|
| `cb_staged_pages` | `ring_entries_per_thread` (entries per reader/writer thread pair). Total `num_entries = ring_entries_per_thread · R`. | 4 (so 4 entries at R = 1) | Reads in flight ahead of the writer, and ≥ 2 implicit-sync transaction-id slices (2 entries each) so credit posting by the DM0 ISR overlaps the next slice's NoC traffic. Catalog `double_buffer`: ~4–8 outstanding transfers is the sweet spot. Quasar README: depth 2 per thread already reached the DRAM floor for tiles. 4 is the minimal value giving each txn-id half more than one transaction in flight, and costs 8 KiB at TILE bf16. |

### Mechanism caps

| Mechanism | Cap on which extent | Clamp | What happens unclamped |
|-----------|--------------------|-------|------------------------|
| Implicit-sync STRIDED read/write moves exactly `get_entry_size()` bytes per call, one entry per NoC transaction (`tt_metal/hw/inc/internal/tt-2xx/dataflow_buffer.inl:782,808` read; `:829` and the STRIDED branch write) | `in_page`: one page per entry, never several pages per entry | entry = one page | Multi-page entries would need one source address per entry, but consecutive interleaved pages live in different banks. The reads would be wrong. |
| DFB entry size must be a multiple of L1 alignment (`tt_metal/impl/dataflow_buffer/dataflow_buffer.cpp:1617-1632`, entry stored in 16-B units) | `entry_bytes` | `entry_bytes = align_up(page_bytes, ttnn.get_l1_alignment())` | A ROW_MAJOR stick of 34 B (W=17) would be truncated to 32 B in 16-B units: silent data loss. |
| Write of `entry_bytes` must stay inside the output page's slot. Interleaved slot = `buffer_aligned_page_size()` = `align_up(page_bytes, buffer alignment)`, with alignment 16 B in L1 and 64 B in DRAM, both ≥ L1 alignment. | `entry_bytes` | `entry_bytes ≤ min(input.buffer_aligned_page_size(), output.buffer_aligned_page_size())`, true by construction; asserted on host | Overwrites the first bytes of the next page in that bank (written by another core): wrong results far from the cause. |
| Quasar NoC transfer alignment is 1 B for L1 and DRAM (`tt_metal/hw/inc/internal/tt-2xx/quasar/noc/noc_parameters.h:483-491`) | none (sticks of any byte length transfer directly) | — | — (no restriction on this target) |
| Implicit sync counts NoC transactions; a transfer above one packet would fire the ISR on its first fragment (`dataflow_buffer.cpp:560`; `NOC_MAX_BURST_SIZE` = 256 words × 256 B = 65,536 B, `noc_parameters.h:395-396`) | `entry_bytes` | Host refuses `entry_bytes > NOC_MAX_BURST_BYTES` (65,536, a named constant citing those lines; there is no Python query) with `NotImplementedError`. Reached only by ROW_MAJOR rows wider than 32,768 bf16 / 16,384 fp32 elements, and no INPUTS shape comes close (max 8192 × 4 B = 32 KiB). | Credits posted before the data lands: the writer sends stale bytes. |
| DFB ring at most 1,048,560 B (target page *Cluster model*) | `ring_entries_per_thread` | `ring_entries_per_thread = min(knob, floor(1,048,560 / (entry_bytes · R)))`, and ≥ 2 | DFB creation fails at program build. Unreachable with the caps above (4 × 3 × 65,536 = 768 KiB). |
| Implicit-sync batch must divide evenly over txn ids and threads (`DataflowBuffer.md:410-424`) | `num_entries` | `num_entries = ring_entries_per_thread · R` with `ring_entries_per_thread` even | Host rejects the DFB when no txn-id count works. |
| Thread plan on a DM→DM STRIDED ring: R + W ≤ 6. This design sets R = W so producer thread t and consumer thread t share one tile counter and each moves `ceil((core_pages − t)/R)` entries. | `THREAD_PLAN` | R = W ∈ {1, 2, 3}, validated on host | A consumer thread waiting on entries no producer thread owes it: a hang at kernel exit (`AAW`). |

### Regimes

| Regime | Status | Predicate | Block | Data movement vs. minimum | What a bigger block buys |
|--------|--------|-----------|-------|---------------------------|--------------------------|
| `interleaved_page_stream`: per-core contiguous page range, reader → STRIDED DM→DM ring → writer, implicit sync both ends | **built** | input interleaved (DRAM or L1) ∧ output interleaved (DRAM or L1) ∧ `entry_bytes ≤ NOC_MAX_BURST_BYTES`. Covers every TARGET and INPUTS cell. | `core_pages` pages (the whole assignment, one block) × `entry_bytes` | Boundary: the tensor's home tier. The input crosses its tier once, either DRAM, or a NoC read from its L1 bank (local for 1/n_banks of pages). The output crosses once, as a NoC write to its bank. Above the named minimum by `entry_bytes − page_bytes` per page (≤ 15 B of slot padding on RM sticks, 0 on tiles). Each tensor is moved exactly once. For an L1 endpoint this is one NoC crossing more than `owner_local` (below). | The block is already the whole per-core assignment. Its fixed costs are each paid **once per core**: kernel launch and DFB setup, the implicit-sync final-batch flush at kernel exit (DM0), and the trid hand-back. Per page, only the transaction itself (one read, one write). Per txn-id slice (every `ring_entries_per_thread/2` entries), one ISR credit post or ack. |
| `owner_local`: bank-affine assignment. The core that owns a page's L1 bank reads the input page straight into the output page's local address (L1 output), or reads its local input page and writes it out (L1 input). No ring for the L1→L1 case. | **deferred** | output is L1 (owner-reads) ∨ input is L1 (owner-writes) | all pages of one L1 bank on its owning core (`pages p with bank(p) = my bank`) | Saves one full NoC crossing of the tensor for each L1 endpoint. `[1,1,1600,512]` TILE DRAM→L1: DRAM 1,638,400 B (unchanged) and NoC writes −1,638,400 B. RM `[1,1,400,64]` L1→L1: NoC traffic 2 × 51,200 B → 51,200 B. | Same per-core fixed costs. Successor on top of it: bank-run coalescing (one transaction for a run of same-bank pages when both slots match), which pays when `page_bytes ≪` one packet. |
| `oversized_page`: entries split into ≤ 1-packet chunks under explicit sync (reserve/push, tagged reads and writes, one trid barrier per page) | **deferred** | `entry_bytes > NOC_MAX_BURST_BYTES` (ROW_MAJOR rows above 64 KiB) | 1 page in ⌈entry_bytes / 65,536⌉ chunks | Same bytes as built, more transactions | — |
| `sharded_in_place`: consume a sharded input through a DFB backed on its shard (`ttnn.cb_descriptor_from_sharded_tensor` analogue for ProgramSpec), write the shard out | **deferred** | input or output sharded | the whole resident shard (one block) | Removes the NoC read of a local shard | one block per shard |
| `single_kernel_copy`: one DM kernel per core reads a page into scratch, barriers, writes it, barriers | **rejected**: superseded by `interleaved_page_stream`, which moves the same bytes with read and write overlapped across two harts and implicit credits | — | 1 page | same bytes, but read latency and write latency serialize per page | — |
| `compute_passthrough`: reader → NEO `copy_tile` → writer | **rejected**: superseded by `interleaved_page_stream` | — | tiles | Adds two L1 ring crossings (unpack + pack) per tile for no transformation. ROW_MAJOR sticks are not tiles. A float32 value through 16-bit DEST breaks R1. | — |

**Deferred-row reasons and reachability:**

- `owner_local`:
  - **Why deferred:** it needs the L1-bank → core map on the host to assign pages by bank. No ttnn Python binding exposes it: `ttnn-python-utility-bindings.md` lists none, and `split_work_to_cores` deals by count. It also needs proof that every L1 bank of the output lies on a cluster of the compute grid; otherwise a bank-local scan in the kernel misses pages. It cannot replace the built regime, because DRAM→DRAM (`memory_config` DRAM × `output_memory` DRAM, in TARGET) has no local endpoint. So `interleaved_page_stream` is the base this layers onto, not a dead end.
  - **How it stays reachable:** the host already builds the per-core page assignment and passes it as runtime args. Kernel selection is per-regime on the host, the same `validate()`, output allocation and single `generic_op` dispatch apply, and the kernels address pages through `TensorAccessor` page ids, which give the bank NoC address the owner check needs.
- `oversized_page`:
  - **Why deferred:** no TARGET×INPUTS shape reaches it. The widest row is 8192 × 4 B = 32 KiB (fp32, TARGET), half a packet.
  - **How it stays reachable:** the host-side cap check is exactly its predicate. The explicit-sync form is the documented one (`toy_thread_plan/kernels/reader.cpp`; target page *Ring protocols*), and it slots in as a second reader/writer pair selected by that predicate.
- `sharded_in_place`:
  - **Why deferred:** TARGET's `sharding` axis has the single value `interleaved`, so no cell reaches it. Sharded inputs and configs are refused by name (R7).
  - **How it stays reachable:** the page-range split is the logical shard (a height-style shard on the `page` axis, independent, so a knob-turn), and the ring is already the consumer interface a shard-backed DFB would replace.

**Regime selection function (Phase 0).** One built regime, so there is no selection. The `entry_bytes > NOC_MAX_BURST_BYTES` check is a refusal (`NotImplementedError`), not a second regime.

### Traffic ranking

Totals are per tensor, counted as crossings (T = `tensor_pages · entry_bytes` bytes).

| Candidate split | DRAM crossings | Cross-core NoC | Notes | Rank |
|-----------------|----------------|----------------|-------|------|
| bank-affine `page` sets (`owner_local`) | each DRAM endpoint once | T per DRAM endpoint only. An L1 endpoint is core-local. | needs bank→core map (deferred, above) | 1 (cheapest wherever an endpoint is L1, so all of PHASE0) |
| contiguous `page` ranges (built) | each DRAM endpoint once | T per L1 endpoint (L1 input read from its bank; L1 output written to its bank) | ties rank 1 on DRAM→DRAM | 2 |
| `page` split by batch / tile-row / tile-column boundaries | same as contiguous ranges | same | buys nothing over contiguous ranges, which already align to any boundary the counts allow | 2 (equal) |
| sub-page split of `in_page` | same bytes | same bytes | ×k transactions for the same bytes (catalog `tile_reorder`) | rejected |
| split a dependent axis + combine | — | — | **no dependent axis exists**: clone has no cross-element dependency, so there is no combine to price | n/a |

Chosen: contiguous `page` ranges over the full grid. It is rank 2: the rank-1 split is `owner_local`, recorded as a `deferred` row with its byte delta and positive reason. Occupancy was not the deciding term. Both candidates fill the grid equally; the difference is the second NoC crossing for L1 endpoints.

### Block schedule

```cpp
// per engaged cluster; num_blocks_this_core == 1 (the block is the whole contiguous range)
for (uint32_t block_idx = 0; block_idx < num_blocks_this_core; ++block_idx) {
    load_block(block_idx);    // reader: pages [start_page, start_page + core_pages) -> cb_staged_pages
    store_block(block_idx);   // writer: cb_staged_pages -> output pages [start_page, start_page + core_pages)
}
```

| Block operation | Block shape | Resident across it | Intended frequency of fixed costs |
|-----------------|-------------|--------------------|-----------------------------------|
| `load_block` | `core_pages` pages × `entry_bytes`, streamed through a window of `ring_entries_per_thread·R` entries. Thread t takes pages `start_page + t, + t + R, …`. | nothing beyond the ring window | `Noc`, `DataflowBuffer` and `TensorAccessor` construction once per kernel. One tagged read per page, with no reserve/push/barrier. ISR credit post once per txn-id slice. Final-batch flush once at kernel exit (firmware). Read trid hand-back (`noc_async_read_set_trid(0, noc)`) once, as the last statement. |
| `store_block` | the same `core_pages` pages × `entry_bytes`. Thread t writes entries `t, t + R, …` to output pages `start_page + t, …`. | nothing | Construction once per kernel. One tagged write per entry, with no wait/pop/barrier. ISR ack once per txn-id slice. Firmware waits for the writes to land at exit. Write trid hand-back (`noc_async_write_set_trid(0, noc)`) once, as the last statement. |

### Perf lamps

| Lamp | Why the default may be wrong here | Nearby alternative to measure |
|------|-----------------------------------|-------------------------------|
| Thread plan (1, –, 1) | With small ROW_MAJOR sticks (48–128 B) the cost is per transaction, not per byte. One hart may be issue-bound (catalog `split_reader`) even though tiles sit on the DRAM floor with one hart (Quasar README). | (2, –, 2) and (3, –, 3) on `[1,1,400,64]` RM and `[2,1,100,47]` RM, measured on a timed backend. Turning it is one constant (`THREAD_PLAN`). |
| Ring depth `ring_entries_per_thread = 4` | Deeper rings keep more transactions in flight per hart. Shallower rings were already at the floor for tiles. | 2 and 8 |
| Grid synchronization (full-grid occupancy) | Small tensors give 1 page per core (`[32,17]` RM: 32 sticks on 32 clusters; `[1,1,32,8192]` RM: 1 stick each), so launch and DFB setup per cluster dominate the one transfer. | a minimum of `min_pages_per_core` (e.g. 4 or 8 pages) per engaged cluster, i.e. fewer clusters with more pages each |
| NoC assignment (reader on the reader-config NoC, writer on the writer-config NoC: the `create_reader/writer_datamovement_config` defaults) | Catalog `noc_placement` measured WH; the Quasar NoC routing may favour another pairing. | both kernels on one NoC; swapped NoCs |
| Overlap lamp | Does not apply: the block (the whole assignment) never has to be resident. The ring window decouples reader and writer at any block size. | — |

## Dataflow Strategy

| Stage | Format | Mechanism | Notes |
|-------|--------|-----------|-------|
| Input page → cluster L1 | raw page bytes (tile or stick; dtype untouched) | reader: `noc.async_read<NocOptions::TXN_ID>(acc_input, cb_staged_pages, {.page_id = start_page + i}, {})` | Implicit sync: the DM0 ISR posts credits. `entry_bytes` per call (page + ≤ 15 B slot padding on sticks; the read stays inside the input slot). |
| Ring | `entry_bytes` per entry, `ring_entries_per_thread·R` entries, STRIDED both ends | `cb_staged_pages` (DM → DM) | one producer kernel (reader), one consumer kernel (writer) |
| Cluster L1 → output page | raw page bytes | writer: `noc.async_write<NocOptions::TXN_ID>(cb_staged_pages, acc_output, {}, {.page_id = start_page + i})` | Implicit sync: the ISR acks. Firmware waits for writes at exit. |
| Placement | output in DRAM or L1 interleaved per `out_mem` | `TensorParameter(spec=output.spec)` | The program is placement-agnostic. The `memory_config` × `output_memory` values change only the tensor specs, so extending SUPPORTED to `output_memory` None/DRAM, or to float32, is a **knob-turn** (no kernel change). |

**TARGET placement axes against the blocking scheme:**

| TARGET value | Cost class | Why |
|--------------|-----------|-----|
| `memory_config` DRAM / L1 (input) | built in Phase 0 | `TensorAccessor` resolves either. The page-range split is unchanged. |
| `output_memory` L1 | built in Phase 0 | — |
| `output_memory` None / DRAM | knob-turn | output spec only, plus SUPPORTED |
| `dtype` float32 | knob-turn | `page_bytes` doubles and `entry_bytes` is derived. Bytes are copied raw, so R1 holds exactly. |
| `sharding` interleaved only | — | sharded is outside TARGET (`sharded_in_place` deferred) |

**Structural impossibilities:** none found. The spec's `INVALID = []` is consistent with this design.

## Work Distribution

| Field | Value |
|-------|-------|
| Work unit | one page (tile or stick). The per-core block is the cluster's whole contiguous range. |
| Grid | `device.compute_with_storage_grid_size()` (32 clusters on this target), `ttnn.split_work_to_cores(grid, tensor_pages_in_rounds, row_wise=True)` |
| Per-core work | `core_pages` contiguous pages from `start_page`, as runtime args to reader and writer (same values for both) |
| Remainder | `split_work_to_cores` gives two groups (`per_core_1`, `per_core_2`). `tensor_pages` comes from `input.buffer_num_pages()`, which already counts each image's tile padding: TILE `outer·ceil(H/32)·ceil(W/32)`, RM `outer·H`. Never `floor`. Clusters beyond `tensor_pages` get no work and are not in `target_nodes`. |
| Bound | **data movement**: zero compute per page against `entry_bytes` read + written per page. |
| Thread plan | `THREAD_PLAN = (R, W) = (1, 1)`, one host constant. R and W are passed to the kernels as compile-time args (`reader_threads`, `writer_threads`) and set `KernelSpec(num_threads=…)`. No compute kernel, so the plan is (1, 0, 1). Per the target page, an op bound by data movement takes one thread of each kind, and implicit sync with one hart reaches the DRAM floor. Legal values R = W ∈ {1, 2, 3} (perf lamp). |
| Unit dealt to threads | The ring entry itself (one page), dealt round-robin: reader thread t and writer thread t both handle the cluster's pages t, t+R, …, so a ragged last round is legal (no padding). When R > 1 the host deals pages to clusters in whole rounds of R (`split_work_to_cores` over `ceil(tensor_pages / R)` rounds, last round short only on the last cluster), so no cluster has fewer pages than threads. Phase 0: R = 1, rounds = pages. |
| Artifacts factory | `create_program_artifacts(input, output, *, threads=THREAD_PLAN, ring_entries_per_thread=RING_ENTRIES_PER_THREAD, grid=None)` returns `(spec, run_args, tensor_indices)`. `grid=None` means `device.compute_with_storage_grid_size()`. The public `clone()` always calls it with the defaults. The keyword knobs exist so that the thread test and the perf lamps turn a knob without editing code; each default is the one host constant that is the knob's single source. |
| Tests the plan needs | (1) **Every thread ran.** Build the program through the artifacts factory with an explicit `threads=(R, R)` and a pre-allocated output pre-filled with a sentinel (a value absent from the input, e.g. bf16 `-3.0e38`). Then check that no page still holds the sentinel and the output equals the input, for R ∈ {1, 2, 3}, on a shape whose per-core page count is not divisible by R (e.g. `[1,1,224,32]` TILE = 7 tiles on one core with a 1×1 grid override, or 7 pages per core). A missing thread leaves its stride of pages at the sentinel. (2) **Device-session tests**, each in one device session: run a tt-metal program first (`ttnn.to_layout` of some tensor), then clone and check; and clone, then `ttnn.to_layout(output, ROW_MAJOR_LAYOUT)` (TILE output) or `ttnn.to_layout(output, TILE_LAYOUT)` (RM output) and check the values. Together they catch a missing trid hand-back. |

**Worked Phase 0 numbers (quasar, 32 clusters, bf16):**

| Input (traced / INPUTS) | Layout | `tensor_pages` | `entry_bytes` | Split | Ring bytes per cluster |
|-------------------------|--------|----------------|---------------|-------|------------------------|
| `[1,1,400,512]` DRAM | TILE | 1·13·16 = 208 | 2,048 | 16×7 + 16×6 | 8,192 |
| `[1,1,1600,512]` DRAM | TILE | 1·50·16 = 800 | 2,048 | 32×25 | 8,192 |
| `[1,1,400,256]` DRAM | TILE | 1·13·8 = 104 | 2,048 | 8×4 + 24×3 | 8,192 |
| `[1,1,1600,256]` DRAM | TILE | 400 | 2,048 | 16×13 + 16×12 | 8,192 |
| `[1,1,400,64]` L1 | RM | 400 | 128 | 16×13 + 16×12 | 512 |
| `[32,17]` | RM | 32 | 48 (34 B page) | 32×1 | 192 |
| `[1,1,32,8192]` | RM | 32 | 16,384 | 32×1 | 65,536 |
| `[2,512,1024]` | TILE | 2·16·32 = 1,024 | 2,048 | 32×32 | 8,192 |

## Circular Buffers

| Semantic Name | Index | Page Size | Num Pages | Sizing rationale | Format | Producer | Consumer | Sync | Lifetime |
|---------------|-------|-----------|-----------|------------------|--------|----------|----------|------|-----------|
| `cb_staged_pages` | DFB unique_id `"staged_pages"` (no numeric slot under ProgramSpec) | `entry_bytes = align_up(input.buffer_page_size(), ttnn.get_l1_alignment())` | `ring_entries_per_thread · R` (Phase 0: 4) | spans `in_page` (whole page per entry), streams `page` (window of `ring_entries_per_thread` entries per thread pair) | `input.dtype` (`data_format_metadata`). Payload is moved by the NoC only and never enters DEST. | reader | writer | implicit both ends: tagged `async_read<TXN_ID>` / `async_write<TXN_ID>`, no reserve/push/wait/pop, ring not in `disable_dfb_implicit_sync_for` | whole program |

That is the full inventory. One buffer is the minimum for a NoC copy on this part: a NoC read lands in the issuing cluster's L1 and a write leaves from it, so a staging slot is needed (except in `owner_local`). There are no constants, no scalers and no intermediates.

## Block Operation Realization

| # | Block operation | Block shape | Helper? | Input CB (semantic name, pages, state) | Output CB (semantic name, pages) | CB state after |
|---|-----------------|-------------|---------|----------------------------------------|----------------------------------|----------------|
| 1 | `load_block` (reader kernel) | `core_pages` × `entry_bytes`, thread stride R | no (raw dataflow API; no kernel-lib helper covers an interleaved page copy, see API Mapping) | — (input tensor via `TensorAccessor`) | `cb_staged_pages`, 1 entry per call | all of this thread's entries posted (final partial batch posted by DM0 at exit) |
| 2 | `store_block` (writer kernel) | same | no | `cb_staged_pages`, 1 entry per call | — (output tensor via `TensorAccessor`) | ring empty; tile counters drained at exit |

## API Mapping

| Block operation | Type | Function | File:Line | Template Params / Args | Input CB | Output CB | Which params are block knobs |
|-----------------|------|----------|-----------|------------------------|----------|-----------|------------------------------|
| `load_block` | raw_api | `Noc::async_read<NocOptions::TXN_ID>(src, DataflowBuffer& dst, src_args, dst_args = {})` (implicit-sync overload) | decl `tt_metal/hw/inc/api/dataflow/noc.h:913-927`; def `tt_metal/hw/inc/internal/tt-2xx/dataflow_buffer.inl:773-813` | `src` = `TensorAccessor(tensor::input)`, `{.page_id = start_page + i}`, `{}` | — | `cb_staged_pages` | none (size = entry size, set on host) |
| `store_block` | raw_api | `Noc::async_write<NocOptions::TXN_ID>(DataflowBuffer& src, dst, src_args, dst_args)` | decl `noc.h:929-942`; def `dataflow_buffer.inl:819-876` | `{}`, `{.page_id = start_page + i}`, `dst` = `TensorAccessor(tensor::output)` | `cb_staged_pages` | — | none |
| both | raw_api | `TensorAccessor(tensor::<name>)` from the binding token | `tt_metal/hw/inc/api/tensor/tensor_accessor.h:112-113` | interleaved DRAM or L1, page size from the tensor spec | — | — | — |
| both | raw_api | `get_my_thread_id()` | `tt_metal/hw/inc/api/kernel_thread_globals.h:155` | loop start; stride R or W (CT args) | — | — | `THREAD_PLAN` |
| `load_block` end | raw_api | `noc_async_read_set_trid(0, noc.get_noc_id())`, the last statement | `tt_metal/hw/inc/api/dataflow/dataflow_api.h:2558` | — | — | — | — |
| `store_block` end | raw_api | `noc_async_write_set_trid(0, noc.get_noc_id())`, the last statement | `dataflow_api.h:2579` | — | — | — | — |
| host | raw_api | `ttnn.split_work_to_cores(grid, n, row_wise=True)`, `ttnn.corerange_to_cores` | pattern: `.claude/references/quasar_op_examples/toy_thread_plan/toy_thread_plan_program_artifacts.py` (`split_tiles`) | `n = ceil(tensor_pages / R)` rounds | — | — | grid, R |
| host | raw_api | `ttnn.experimental.ProgramSpec`, `KernelSpec(num_threads=…, dfb_bindings=[producer_of / consumer_of], tensor_bindings, compile_time_args, runtime_arg_schema)`, `DataflowBufferSpec(unique_id, entry_size, num_entries, data_format_metadata)`, `TensorParameter`, `WorkUnitSpec`, `ProgramRunArgs`, `create_reader_datamovement_config` / `create_writer_datamovement_config`, `ttnn.generic_op([input, output], spec, run_args, tensor_indices)` | same reference file; DM-only program precedent: `ttnn/ttnn/operations/toy_spec_single_tensor/toy_spec_single_tensor.py` (`create_fill_artifacts`: two DM kernels, no compute) | `entry_size = entry_bytes`, `num_entries = ring_entries_per_thread·R` | — | — | `ring_entries_per_thread`, `THREAD_PLAN` |
| host | raw_api | `ttnn.TensorSpec(shape, dtype, layout, memory_config, tile)`, `ttnn.allocate_tensor_on_device(spec, device)` | `ttnn/cpp/ttnn-nanobind/tensor.cpp:289-305`; `ttnn/cpp/ttnn-nanobind/operations/core.cpp:275` | output spec with `out_mem` | — | — | — |
| host | raw_api | `ttnn.get_l1_alignment()`, `tensor.buffer_page_size()`, `tensor.buffer_aligned_page_size()`, `tensor.buffer_num_pages()` | `ttnn/cpp/ttnn-nanobind/device.cpp:806`; `.claude/references/ttnn-python-utility-bindings.md` §`buffer_aligned_page_size` | `entry_bytes` derivation and slot assertion | — | — | — |

**Helpers considered and rejected (every raw_api entry above):**

| Candidate | File:Line | Why it cannot express the shape |
|-----------|-----------|---------------------------------|
| compute helpers (`eltwise_convenience.hpp` `copy`, `eltwise_chain` `CopyTile`) | `ttnn/cpp/ttnn/kernel_lib/eltwise_convenience.hpp` | They run on a compute kernel. clone has none by design: `compute_passthrough` is rejected because it adds two L1 ring crossings per tile, cannot carry ROW_MAJOR sticks, and a float32 tile through 16-bit DEST violates R1. |
| `local_copy_helpers_dataflow.hpp` (`local_addr`, `set_read_state`, `read_with_state`, `set_read_trid`) | `ttnn/cpp/ttnn/kernel_lib/local_copy_helpers_dataflow.hpp:9,46-57,74-95` | Self-aimed L1→L1 reads within one core ("@brief Local L1 -> L1 copy helpers: self-aimed NoC reads", `:9`). clone's pages live in DRAM banks or other clusters' L1 banks and are addressed by `TensorAccessor` page id. |
| `dfb_helpers_dataflow.hpp` | `ttnn/cpp/ttnn/kernel_lib/dfb_helpers_dataflow.hpp:14-18` | Only tile r/c dimension getters; no transfer. |
| `mcast/kernel/mcast_args.hpp` | `ttnn/cpp/ttnn/kernel_lib/mcast/kernel/mcast_args.hpp` | Multicast of a shared operand. clone has no reuse-shared operand (operand-reuse check above). |
| `tilize_helpers_dataflow.hpp` / `untilize_helpers.hpp` | `ttnn/cpp/ttnn/kernel_lib/` | Layout conversion. clone never changes layout. |

No compute phase exists, so `compute_kernel_hw_startup()` does not apply.

## Key Risks and Gotchas

| Risk | Why it bites here | Mitigation in this design |
|------|-------------------|---------------------------|
| Sticky transaction tag | Both kernels issue only tagged transfers. A tt-metal program after clone (`ttnn.to_layout` of its output, the obvious next call) would have its untagged transfers counted against a ring: a hang or `tile counter overflowed capacity`. | The last statement of the reader is `noc_async_read_set_trid(0, noc.get_noc_id())`, and of the writer `noc_async_write_set_trid(0, noc.get_noc_id())`. Device-session tests (Work Distribution) catch a regression. |
| ROW_MAJOR stick not a multiple of 16 B (W odd, e.g. 17 → 34 B) | The DFB entry is stored in 16-B units, so the raw page size would be truncated. A DRAM-aligned (64-B) entry would overrun the output's 16-B L1 slot into the neighbouring page. | `entry_bytes = align_up(page_bytes, ttnn.get_l1_alignment())`, and the host asserts `entry_bytes ≤ min(in, out).buffer_aligned_page_size()`. Regression shapes: `[32,17]`, `[1,1,17,50]`, `[2,1,100,47]` RM, with input DRAM and output L1. |
| Using `buffer_aligned_page_size()` of one tensor as the entry | DRAM input (64-B slots) → L1 output (16-B slots): the 64-B entry overwrites the next L1 page's head. | The entry is derived from the page size, never from one side's slot. |
| Returning the input or a view when `out_mem` equals the input's placement | R5. The golden suite cannot see it. | Always `allocate_tensor_on_device` and always dispatch. Test: `output.buffer_address() != input.buffer_address()` for an L1 input with `memory_config=L1`. |
| Output placement unchecked by the golden suite | R3 is a review check. | Test `output.memory_config().buffer_type` and `.memory_layout == INTERLEAVED` for DRAM and L1 inputs with `memory_config=ttnn.L1_MEMORY_CONFIG`. The None and DRAM cases are refused in Phase 0 (`NotImplementedError`) until SUPPORTED grows. |
| X1 / X2 ordering | `canonical_memory` must run before any tensor work so that the word `"L1"` raises `TypeError`. A `dtype` ≠ the input's reads as itself and must be refused. | The host order of operations in Parameters puts canonicalization first, then validate. |
| Sharded input or sharded `memory_config` | R7: must be refused by name, and the generic runner never builds one. | The `sharding` axis comes from `eval.memory.memory_axes(input)` and `output_memory` from `canonical_memory`, both gated in `validate()`. The op carries its own test with a height-sharded input and a sharded config, expecting `NotImplementedError`. |
| Tile padding of non-aligned TILE tensors | The buffer holds `ceil`-padded tiles. A page count derived from the logical shape with `floor` drops tiles. | `tensor_pages = input.buffer_num_pages()` (single source). Output spec from the same logical shape and tile, asserted equal in page count and page size. |
| R ≠ W or R + W > 6 | A STRIDED DM→DM ring with unequal thread counts and a ragged count can leave a consumer waiting at exit. | Host validates R = W ∈ {1, 2, 3}. The `THREAD_PLAN` constant is the only place it is set. |
| DFB region against L1 tensors | The output (up to 1.6 MB for `[1,1,1600,512]`) is allocated in L1 banks while the program's ring is placed in each cluster's L1. | The ring is ≤ 64 KiB on every INPUTS shape (8 KiB on tiles), and 1.6 MB over 32 banks is ~51 KB per bank. Not tight on 4 MiB, but the session tests run the largest traced shape with an L1 output. |
| Thread plan invisible in a passing test | A single thread could do all the work (it cannot here, because of the stride, but a regression to `get_num_threads()` or a loop without the stride would hide it). | The sentinel-prefilled thread test in Work Distribution, run at R ∈ {1, 2, 3}. |
