# Operation Design: argmax

## Overview

| Field | Value |
|-------|-------|
| Classification | data_movement (a compare-and-index scan run on the data-movement harts; no compute kernel) |
| Goal | One device program per call that returns, for every row of the input along `dim = -1`, the index of its largest element (lowest index on ties, `+0.0 == -0.0`), as a uint32 row-major tensor placed where the input is. |
| Math | `out[r] = min{ i : key(x[r, i]) = max_j key(x[r, j]) }` for every row `r` of the input flattened to `rows × W`. `key` is the order-preserving unsigned key of the stored bits (below). No arithmetic on the values. |
| Specification | `eval/specs/argmax/spec.py` (authoritative: RULES R1 to R10, REJECTIONS X1 and X2, PHASE0) |
| References | `.claude/references/targets/quasar.md`, `.claude/references/backends/ttsim.md`, `.claude/references/blocking-model.md`, `.claude/references/l1-footprint-discipline.md`, `.claude/references/quasar_op_examples/` (README, `toy_thread_plan`), `ttnn/ttnn/operations/toy_spec_mcast/` (a ProgramSpec with DM kernels only), `tt_metal/hw/inc/api/{scratchpad.h,semaphore.h,dataflow/noc.h,dataflow/endpoints.h,core_local_mem.h}` |

**The comparison key** (R1, R3). For a bf16 word `b` (16 bits): `if (b == 0x8000) b = 0` (so -0 equals +0); `key = (b & 0x8000) ? (~b & 0xFFFF) : (b | 0x8000)`. Unsigned order of `key` is the float order of the stored value. float32 (a later refinement) is the same transform on 32 bits. A key never passes through a narrower type (R3). NaN is unspecified (`NUMERIC["special_values"]` is UNRESOLVED). A positive NaN gets the largest key and wins, as in torch. Nothing in the suite passes one.

**Tie rule** (R1). Within a scan, the running best is replaced only when `key > best_key`. A merge of partials of the same row takes them in ascending column order with the same strict `>`. Equivalently, it takes the lexicographic maximum of `(key, -index)`. That order is total, commutative and associative, so the merge order cannot change the result.

## Target

| Fact | Value | Source |
|------|-------|--------|
| Target | `quasar` on the ttsim simulator | `.claude/references/targets/quasar.md` + `.claude/references/backends/ttsim.md` |
| Arch | `quasar` | target page |
| Compute grid | 8 × 4 = 32 clusters (probe on this run: `compute_with_storage_grid_size()` = 8×4) | target page; queried at run time |
| L1 per core | 4,194,304 B (4 MiB) per cluster, shared by its NEOs and DM harts | target page |
| Unreserved budget | `ttnn.get_max_worker_l1_unreserved_size() - L1_RESERVE_BYTES` = 3,963,328 − 65,536 = **3,897,792 B** on this target (query value read by a probe on this run) | probe of `ttnn.get_max_worker_l1_unreserved_size()` |
| L1 / DRAM alignment | 16 B / 64 B (`ttnn.get_l1_alignment()`, `ttnn.get_dram_alignment()`). NoC transfers need no alignment on this part: `NOC_{L1,DRAM}_{READ,WRITE}_ALIGNMENT_BYTES = 1` (`tt_metal/hw/inc/internal/tt-2xx/quasar/noc/noc_parameters.h:486-491`) | target page; noc_parameters.h |
| Host program | `ttnn.ProgramSpec` (Metal 2.0, `ttnn.experimental.*`) dispatched by one `ttnn.generic_op` call. Reference: `.claude/references/quasar_op_examples/` (README first); a DM-only ProgramSpec: `ttnn/ttnn/operations/toy_spec_mcast/toy_spec_mcast.py` | target page |
| Threads per core | 4 NEOs (C ∈ {1,2,4}) and 6 user DM harts (R + W ≤ 6; a DM `KernelSpec` takes `num_threads ≤ 6`, `tt_metal/impl/metal2_host_api/program_spec/validation/kernel_spec.cpp:124-130`). R and W must divide C only on STRIDED rings, and this op has no rings. User transaction ids are 0..7 (`noc.h:113`, `DFB_TXN_ID_BASE = 8`) | target page; kernel_spec.cpp; noc.h |
| Unavailable here | `ttnn.ProgramDescriptor` and the legacy program-factory device op (Metal 1, refused at kernel creation). `bfloat8_b` / `bfloat4_b` (no path on this part). uint16/uint32 are not buffer formats (the scratchpads here carry no format metadata). Untagged NoC transfers: not used, every transfer carries a user trid and the last statements reset the read and write trids to 0. Not in the LLK: an index-carrying max (no such primitive under `tt_metal/tt-llk/tt_llk_quasar/llk_lib/`), so there is no NEO path for argmax today. A `Scratchpad` cannot be a multicast destination (`static_assert`, `tt_metal/hw/inc/internal/local_mem/noc_traits.h:97-101`). | target page; LLK listing |

Every worked number below is for this target: 32 clusters, 6 scan threads each, S_total = 192 scanners. Every predicate the op evaluates calls the queries: `compute_with_storage_grid_size()` (or the `sub_core_grids` set), `get_max_worker_l1_unreserved_size()` and `get_l1_alignment()`.

## Parameters

| Name | Type | Required | Valid Range | Default | CT/RT |
|------|------|----------|-------------|---------|-------|
| `input_tensor` | `ttnn.Tensor` | yes | Phase 0: bf16, ROW_MAJOR, DRAM interleaved, rank 4 | — | tensor binding `input` |
| `dim` | `int` or `None` | no | `[-ndim, ndim-1]` (X1 otherwise). Canonicalised to `dim if dim < 0 else dim - ndim` **before** the support check (R4). Only `-1` is supported; `None` and other dims raise NotImplementedError. | `None` | host only |
| `keepdim` | `bool` | no | Phase 0: `False` | `False` | host only: it sets `out_rows_per_page` (common RT) |
| `sub_core_grids` | `ttnn.CoreRangeSet` or `None` | no (kw-only) | any. Not gated (R5). When given, it is the set of clusters the program runs on, in place of the device grid. The result does not depend on it. | `None` | host only: the core set |
| `output_tensor` | `ttnn.Tensor` or `None` | no (kw-only) | `None` only. Any other value raises (R6, X2). | `None` | — |

Validation order in `validate()`. (1) `output_tensor is not None` raises `ValueError` (X2). (2) `dim` outside `[-ndim, ndim-1]` raises `IndexError` (X1); `dim is None` raises NotImplementedError. (3) Canonicalise `dim` (R4). (4) Derive `memory_config` and `sharding` with `eval.memory.memory_axes(input_tensor)` (R9) and the specification's `AXES` (`INPUT_TAGGERS = AXES`, imported, never redefined). (5) Gate every axis against `SUPPORTED`, raising NotImplementedError (R10). `sub_core_grids` is never read by `validate()` (R5).

## Tensors

### Input

| Property | Requirement |
|----------|-------------|
| Shape | rank 4 `(N, C, H, W)` in Phase 0; the op flattens it to `tensor_rows = N·C·H` rows of `tensor_w = W` elements. The code takes any rank ≥ 2: `tensor_rows = prod(shape[:-1])` |
| Dtype | bfloat16 (elem_bytes = 2) |
| Layout | ROW_MAJOR: one page per row, page size `W · elem_bytes`; any W (alignment is irrelevant in row-major) |
| Memory | DRAM, interleaved |

### Output

| Property | Value |
|----------|-------|
| Shape | `spec.output_shape`: the input shape with `dim` removed (keepdim=False) or kept at size 1 |
| Dtype | uint32 |
| Layout | ROW_MAJOR. Page = the output's last dimension: `out_rows_per_page = output.shape[-1]`. This is H for rank-4 keepdim=False and 1 for keepdim=True. Row `r` lives at page `r // out_rows_per_page`, byte `4·(r % out_rows_per_page)` |
| Memory | interleaved, in the input's buffer (R8): `ttnn.allocate_tensor_on_device(ttnn.TensorSpec(shape, ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT, memory_config=input.memory_config()), device)` |

## Blocking Model

The op's data is a `tensor_rows × tensor_w` matrix of rows that are independent of each other. Each row is reduced along W to one (key, index) pair. The scan runs on DM harts: each **scanner** (one DM thread on one cluster) owns a contiguous range of **units** and walks it in blocks.

### Axes

| Axis | Character (+ one-clause reason) | Extent knob | Phase 0 value | Knob source | Core-assignment | Later unlock |
|------|--------------------------------|-------------|---------------|-------------|-----------------|--------------|
| `rows` (N·C·H folded; contiguous page order, the op treats them alike) | **independent**: no row's result depends on another row | `block_rows`: rows per block. When a row fits the block, `= min(rows_this_scanner, floor(block_bytes / row_pitch))`; otherwise 1 | the whole scanner assignment when it fits. Worked: (4,8,32,256) → 6 (5 on 128 scanners); (1,1,2048,256) → 11; (1,1,32,64) → 1, since each scanner owns 1 row. SEGMENTED → 1, a unit being one segment of one row | derived from `block_bytes` (host, one function `derive_blocks()`) | ROWS: `ttnn.split_work_to_cores(grid, tensor_rows)` gives contiguous row ranges to clusters, and each cluster's range is cut contiguously over its `SCAN_THREADS` threads | knob-turn |
| `w` (the searched dim, `dim = -1`) | **dependent**: the result spans the whole row (an argmax over it) | `block_cols`: elements per block along W `= min(piece_cols, floor(block_bytes / elem_bytes))`. Across scanners: `segments_per_row` (spr) | whole row / whole segment: (1,1,2,151936) → 1583-element segments, one block each; (4,8,32,256) → 256 | `block_cols` from `block_bytes`; `segments_per_row` from `select_regime()` (host) | ROWS: not split across scanners (a scanner carries the running (key, idx) across column blocks). SEGMENTED: cut into `spr` segments dealt to separate scanners and combined by a gather to the finalizer | scheme-change (already built: SEGMENTED) |
| `partials` (data of the combine stage: `tensor_rows × spr` records, SEGMENTED only) | rows: **independent**; segments of a row: **dependent** (merged in column order) | whole stage, one block (U ≤ S_total records) | all U records in one wait | `num_records = U` (common RT) | **not spread**: the finalizer (unit 0's scanner) merges all U ≤ 192 records. The merge is U compares, less than one block scan, and its wait ends with the slowest scanner. The finalizer's role is communication; the computation left on it is negligible. Spreading it is a `deferred` regime (PER-ROW-COMBINERS) | scheme-change |
| `scan_threads` (DM harts of a cluster) | — (the thread plan, the 4th knob) | `SCAN_THREADS` | 6 | host constant `SCAN_THREADS`, passed as CT arg `scan_threads` | each cluster's unit range is cut contiguously over its threads | knob-turn |

**Operand-reuse check.** The single input varies along every axis split (`rows` and `w`), so no operand is reuse-shared and no broadcast regime exists. Multicast has no role in this op.

**Single sources (DRY).** Each knob is defined once, as a host constant or derived in one host function, and handed to the kernel as a CT or RT arg:

| Knob | Phase 0 | Defined in | Everything derived from it |
|------|---------|------------|----------------------------|
| `SCAN_THREADS` | 6 | host constant | `num_threads`, S_total = NC·6, per-thread scratchpad slices, RT array lengths |
| `IN_DEPTH` | 2 (max) | host constant | effective depth `in_depth = min(IN_DEPTH, max_blocks_per_scanner)`; `sp_blocks` size |
| `OUT_DEPTH` | 2 | host constant | `sp_out_stage` size, write trids 2..3 |
| `MIN_SEGMENT_BYTES` | 2048 | host constant | `segments_per_row` |
| `L1_RESERVE_BYTES` | 65,536 | host constant | the budget |
| `RECORD_BYTES` | 16 | host constant (= L1 alignment of this part; assert `>= ttnn.get_l1_alignment()`) | `sp_partials` size, slot offsets |
| `block_bytes` | `min(need, cap)` (below) | host `derive_blocks()` | `block_rows`, `block_cols`, `row_pitch`, `num_blocks_this_scanner`, `out_stage_rows` |

`row_pitch = align_up(min(tensor_w, block_bytes/elem_bytes)·elem_bytes, l1_alignment)`. `need` = the largest block any scanner asks for: ROWS `rows_per_scanner_max · align_up(tensor_w·elem_bytes, l1_align)`; SEGMENTED `align_up(seg_cols·elem_bytes, l1_align)`. `cap = align_down((budget − sp_partials_bytes) / SCAN_THREADS / (IN_DEPTH + OUT_DEPTH/4), l1_align)` = **259,632 B** on this target. The `/4` covers output staging: `block_rows ≤ block_bytes/16` rows of 4 B each.

### Buffer-depth knobs

| Buffer | Depth knob | Phase 0 value | What the depth buys |
|----|------------|---------------|---------------------|
| `sp_blocks` (per-thread input block slots) | `in_depth = min(IN_DEPTH, max_blocks_per_scanner)` | 2 when any scanner walks ≥ 2 blocks; otherwise 1. Every PHASE0 shape on 32×6 is 1, because the whole assignment is one block | the read of block k+1 (trid `(k+1) % in_depth`) runs while block k is scanned |
| `sp_out_stage` (per-thread output index slots) | `OUT_DEPTH` | 2 | the write of block k's indices (trid `2 + k%2`) runs while block k+1 is scanned. A slot is reused only after a barrier on its trid |
| `sp_partials` | — | 1 slot per unit | a gather landing zone; one write per slot per program |

### Mechanism caps

| Mechanism | Cap on which extent | Clamp | What happens unclamped |
|-----------|--------------------|-------|------------------------|
| L1 budget per scanner thread | `block_bytes` | `≤ cap` (above), from the device query | scratchpad allocation fails, or two threads' slices overlap |
| uint32 index and record fields | `tensor_w`, `tensor_rows` | `validate()` refuses `tensor_w ≥ 2^32` or `tensor_rows ≥ 2^32` (NotImplementedError) | the index wraps and is silently wrong |
| User transaction ids 0..7 (`noc.h:113`) | read slots `in_depth` + write slots `OUT_DEPTH` + 1 record trid ≤ 8 | `IN_DEPTH + OUT_DEPTH + 1 ≤ 8`, asserted on the host | a trid ≥ 8 is a ring's id |
| Gather landing zone (`sp_partials`) | records per program U | SEGMENTED predicate guarantees `U = tensor_rows·spr ≤ S_total`; zone sized `S_total` slots | records overwrite each other or fall outside the zone |
| Finalizer output staging (SEGMENTED) | `tensor_rows` staged at once | the predicate `spr ≥ 2` with `spr ≤ floor(S_total / tensor_rows)` gives `tensor_rows ≤ S_total/2 = 96` | unbounded staging |
| Semaphore count | `num_records` | `U ≤ S_total` (uint32) | — |
| DM cache coherence (`scoped_lock`, `core_local_mem.h:145`) | the range the scanner reads or sources | every block is invalidated before it is scanned; every staged range is flushed before a NoC write reads it | stale data scanned, or stale bytes written to DRAM |

### Regimes

Selection function (host, `select_regime(tensor_rows, tensor_w, elem_bytes, num_clusters)`, also exported for tests):

```
S_total = num_clusters · SCAN_THREADS          # num_clusters: sub_core_grids if given, else the device grid (query)
spr     = min(S_total // tensor_rows, (tensor_w · elem_bytes) // MIN_SEGMENT_BYTES)
regime  = SEGMENTED if spr >= 2 else ROWS
U       = tensor_rows · spr  (SEGMENTED)   |   tensor_rows  (ROWS)
seg_cols = ceil(tensor_w / spr)            # SEGMENTED; the last segment is tensor_w − (spr−1)·seg_cols
```

Minimum traffic, against the DRAM boundary: each input element crosses DRAM once and each output index crosses once.

| Regime | Status | Predicate | Block | Data movement vs. minimum | What a bigger block buys |
|--------|--------|-----------|-------|---------------------------|--------------------------|
| **ROWS**: rows split over all scanners; each scanner scans whole rows | **built** | `spr < 2`. In PHASE0 on 32×6: (4,8,32,256), (1,1,2048,256), (1,1,32,64), (1,1,32,50), (4,8,32,47), (1,1,17,64), (1,1,17,50), (2,1,100,47), (2,4,64,1) | `block_rows × block_cols` = `min(rows_this_scanner, floor(block_bytes/row_pitch))` × `tensor_w` when a row fits; otherwise `1 × floor(block_bytes/elem_bytes)` | **minimum**: input once, output once, no cross-core traffic | per block: one trid barrier, one cache invalidate (`scoped_lock`) over the block, and one output write per output page the block's rows touch, plus its flush. Intended frequency: once per block, and Phase 0 has one block per scanner. The NoC reads stay one per row piece whatever the block |
| **SEGMENTED**: W split into `spr` segments over scanners; a gather of one 16-B record per unit to the finalizer, which merges and writes | **built** | `spr ≥ 2`. In PHASE0: (1,1,2,151936) spr=96 U=192, (1,1,32,4096) spr=4 U=128, (1,1,32,8192) spr=6 U=192 | `1 × block_cols`, `block_cols = min(seg_cols, floor(block_bytes/elem_bytes))`; one block per scanner in Phase 0 | input once and output once, **plus** U × 16 B of records over the NoC and U remote semaphore increments (≤ 3,072 B and 192 atomics). This is the only cost above the minimum, and it is the price of using the scanners that ROWS would leave idle (2 rows → 2 of 192) | per block: as for ROWS. Per scanner: one record write and one semaphore increment. Per program: one wait, one merge of U records, and ⌈rows/out_rows_per_page⌉ (+1) output writes |
| BALANCED-RANGES: contiguous multi-unit ranges with partial head and tail rows, U > S_total allowed, so rows in `(S_total/2, S_total)` with wide W fill every scanner | **deferred**: no PHASE0 shape is in that band on 32×6. The first shape that is, (128, 8192), is rank 2 and outside PHASE0. Reachable: the scanner's piece walk and the record sink both exist; this adds a head and a tail record per scanner (2 slots per scanner in `sp_partials`) | `S_total/2 < tensor_rows < S_total` and `tensor_w·elem_bytes ≥ 2·MIN_SEGMENT_BYTES` | as SEGMENTED | as SEGMENTED, with ≤ 2 records per scanner | as SEGMENTED |
| PER-ROW-COMBINERS: each row's segment-0 scanner merges its own row's records | **deferred**: the single finalizer's merge is U ≤ 192 compares and ≤ 96 indices written, both below one block scan. The critical path is the slowest scanner's finish either way. Reachable: change the slot owner from the finalizer to `row·spr`'s scanner, with one semaphore per thread (6 ≤ 16 per cluster) | — | — | same bytes as SEGMENTED | — |
| INTRA-CLUSTER-W: rows to clusters, W segments to that cluster's threads, combined in local L1 | **rejected**: superseded by SEGMENTED. It saves ≤ 3 KB of NoC records but caps the scanners at `rows × 6` (12 of 192 for the model's 2-row call) | — | — | minimum (combine core-local) | — |
| NEO-SCAN: reinterpret or tilize row chunks, then an SFPU index-carrying max | **deferred**: Quasar has no index-carrying max primitive (`tt_metal/tt-llk/tt_llk_quasar/llk_lib/` has none) and no kernel-library helper computes one. `reduce<MAX>` returns values only. Reachable: it would replace only the `scan_block` operation, and the split, the gather and the stores are kept | — | — | minimum DRAM; adds an L1→DEST unpack per element | — |
| TWO-PASS: a max pass, then an index-of-first-max pass | **rejected** (dead end): it reads the input from DRAM twice. The single-pass running (key, idx) scan is exact and supersedes it everywhere | — | — | input ×2 | — |
| TWO-DISPATCH: partials to DRAM, then a second program | **rejected**: it violates the one-dispatch rule. SEGMENTED's in-program gather supersedes it | — | — | — | — |

Resident vs streaming is **not** a separate regime here. The scan is single-pass whatever the size, and `block_bytes = min(need, cap)` makes the whole assignment one resident block when it fits and streams column or row blocks when it does not. No path re-reads anything.

### Traffic ranking

| Candidate split | DRAM crossings (input / output) | Cross-core | Rank |
|-----------------|---------------------------------|------------|------|
| `rows` across scanners (ROWS) | 1 / 1 | none | 1 (cheapest; independent axis) |
| `rows` to clusters + `w` within a cluster (INTRA-CLUSTER-W) | 1 / 1 | none (L1 combine) | 1 (ties ROWS, but fills only rows×6 scanners) |
| `w` across scanners + gather (SEGMENTED) | 1 / 1 | U × 16 B + U atomics, one fan-in to one cluster | 2 (≤ 3 KB above the minimum) |
| two passes | 2 / 1 | none | rejected |
| two dispatches | 1 / 1 + partials round trip via DRAM | — | rejected (rule) |

Every candidate moves the input across DRAM exactly once. The independent split (ROWS) is primary. The tiebreak goes to the split with no combine. SEGMENTED is the work assignment for the shapes ROWS leaves on fewer than half the scanners. This is an occupancy decision, settled by the predicate above and not by traffic. It adds ≤ 3 KB.

**Stall-shadow check.** (1) A scanner waits for a block read. The previous block's scan fills that wait whenever a scanner walks ≥ 2 blocks (`in_depth = 2`). With one block per scanner the wait is the pipeline fill (overlap lamp). (2) The finalizer waits for U records. Its own segment's scan is scheduled before the wait. Nothing else is independent of the records: the merge needs them, and writing a row needs its merge. The merge order is commutative (lexicographic `(key, −index)`), so records could be folded in arrival order, but that would shorten only a ≤ 192-compare tail. There is no floating-point arithmetic anywhere, so no reorder needs a precision gate.

### Block schedule

Per scanner thread (cluster `c`, thread `t`, unit range `[unit_start[t], unit_start[t] + unit_count[t])`):

```cpp
for (uint32_t block_idx = 0; block_idx < num_blocks_this_scanner; ++block_idx) {
    load_block(block_idx);          // issue block_idx + in_depth − 1 ahead; barrier on block_idx's trid
    scan_block(block_idx);          // fold the block into the running (key, idx) of each row piece in it
    emit_block_results(block_idx);  // ROWS: stage the indices of rows completed in this block, write the run(s)
}
send_partial();                     // SEGMENTED only: one 16-B record -> finalizer slot u, remote semaphore up
combine_partials();                 // SEGMENTED, finalizer thread only: wait U, merge spr records per row in order
store_results();                    // SEGMENTED, finalizer thread only: write tensor_rows indices
release_noc();                      // barrier own write trids; hand trid 0 back for reads and writes
```

| Block operation | Block shape | Resident across it | Intended frequency of fixed costs |
|-----------------|-------------|--------------------|-----------------------------------|
| `load_block` | `block_rows × block_cols` elements: one NoC read per row piece, `block_cols·elem_bytes` bytes each, from page `row` at byte offset `col_lo·elem_bytes`, landing at `row_pitch` strides in this thread's slot `block_idx % in_depth` | the other depth slot (being scanned) | one trid barrier per block. Reads issued back to back with no per-read barrier |
| `scan_block` | the same rectangle, read through one `scoped_lock` over the slot (cache invalidate) | running `(best_key, best_idx)` per row piece, carried across column blocks of the same row | one invalidate per block. Init of the running pair once per row piece (from its first element), not per block |
| `emit_block_results` (ROWS) | the `block_rows` indices of rows that finished in this block | — | one flush (`scoped_lock` release) and one write per output page touched, per block. Trid `2 + block_idx % 2`, barrier before slot reuse |
| `send_partial` (SEGMENTED) | one record `{key, index, 0, 0}` (16 B) | — | once per scanner: flush, write to the finalizer's `sp_partials + 16·u` (trid 4), barrier on trid 4, `up(noc, fx, fy, 1)` |
| `combine_partials` (finalizer) | all U records: `tensor_rows × spr` | the staged output indices | once per program: `wait(U)`, then `set(0)` (re-run safety), one invalidate over U·16 B. Each row is merged over slots `[r·spr, (r+1)·spr)` in order with strict `>` |
| `store_results` (finalizer) | `tensor_rows` indices | — | once per program: one flush, one write per output page |
| `release_noc` | — | — | once per thread: `async_write_barrier` on trids 2..4, then `noc_async_read_set_trid(0, noc_id)` and `noc_async_write_set_trid(0, noc_id)` as the last statements |

### Perf lamps

| Lamp | Why the default may be wrong here | Nearby alternative to measure |
|------|-----------------------------------|-------------------------------|
| Overlap | Phase 0 makes each scanner's whole assignment one block (`in_depth = 1`). The read of the assignment fully precedes its scan, so no read/scan overlap happens within a thread | `block_bytes = need/2` and `need/4` with `IN_DEPTH = 2` (the structural count: blocks per scanner 1 → 2..4) |
| Grid synchronization | SEGMENTED engages up to 192 scanners for 2 rows. Each segment (~1.5K elements) pays a DRAM read latency, a record write and a semaphore round trip that may rival its scan | `MIN_SEGMENT_BYTES` 2 KiB (default) against 8 KiB (fewer, fatter segments; for the 2-row call, scanners 192 → 74) |
| Thread count | 6 harts per cluster share the cluster's NoC endpoints and L1 ports. Their L2 caches are invalidated per block | `SCAN_THREADS = 6` against 4 (one constant) |
| Scan granularity | one halfword load + key + compare per element | 64-bit loads processing 4 bf16 per load in registers (same results, fewer loads) |
| Dealing | `split_work_to_cores` puts contiguous rows on clusters before stacking threads. For ROWS with ≤ 32 rows only thread 0 runs on each cluster | contiguous dealing that fills threads first (whole rounds of 6 per cluster: fewer clusters, more harts each) |

## Dataflow Strategy

| Stage | Format | Mechanism | Notes |
|-------|--------|-----------|-------|
| DRAM → scanner slot | raw bf16 bytes of a row piece | `noc.async_read<NocOptions::TXN_ID>(input_accessor, UnicastEndpoint/Scratchpad slot, bytes, {.page_id = row, .offset_bytes = col_lo·2}, {.offset_bytes = slot_off + i·row_pitch}, {.trid = block_idx % in_depth})` | byte-granular NoC on this part. One read per row piece; `async_read_barrier<TXN_ID>({.trid})` once per block |
| slot → registers | bf16 words through the DM cache | `scoped_lock` (invalidate) on the slot, then scalar loads | the key transform and strict-`>` fold |
| running result → `sp_out_stage` | uint32 indices | scalar stores, then `scoped_lock` release (flush) | ROWS: per block. SEGMENTED: finalizer only |
| `sp_out_stage` → DRAM output | uint32 run | `noc.async_write<TXN_ID>(stage, output_accessor, bytes, {}, {.page_id = r // out_rows_per_page, .offset_bytes = 4·(r % out_rows_per_page)}, {.trid = 2 + k%2})` | one write per output page the run touches. Neighbouring scanners write disjoint 4-B ranges of the same page |
| record → finalizer `sp_partials` (SEGMENTED) | `{key:u32, index:u32, 0, 0}` | flush the local send slot, `noc.async_write<TXN_ID>` to `UnicastEndpoint{fx, fy, sp_partials_base + 16·u}` (trid 4), barrier, `Semaphore(sem::records_arrived).up(noc, fx, fy, 1)` | the scratchpad base is a common RT arg, so it is the same on every cluster. `fx, fy` = `device.worker_core_from_logical_core(finalizer_core)` (common RT) |
| finalizer merge | records | `wait(U)`, `set(0)`, `scoped_lock` over the zone, merge | — |

**TARGET axes against the scheme.** Each axis's character decides the cost of growing it:

| TARGET value outside PHASE0 | Class | What changes |
|-----------------------------|-------|--------------|
| `memory_config: L1` (interleaved) | knob-turn | none in the kernels: `TensorAccessor` addresses L1 banks. The output goes to L1 by R8 |
| `dtype: float32` | knob-turn | `elem_bytes = 4` and a 32-bit key transform (one template parameter of `scan_block`). R3 holds because keys are full width |
| `keepdim: True` | knob-turn | `out_rows_per_page = 1`, so every row is its own output page and runs split per row |
| `rank: 2, 3` | knob-turn | `tensor_rows = prod(shape[:-1])` is already general. Rank 2's output is 1-D with one page of `tensor_rows` |
| `layout: TILE` | load-path refinement, same scheme | `load_block` reads tile-rows (32 rows × the tiles of a segment), and `scan_block` walks face order, skipping padded columns ≥ W and rows ≥ H (GUIDANCE). The unit becomes a tile-row or a tile-segment. The split, gather and store do not change |
| `sharding` | only `interleaved` is in TARGET | — |

**Structural impossibilities**: none to add; the specification's `INVALID = []` stands.

## Work Distribution

| Field | Value |
|-------|-------|
| Work unit | ROWS: one row. SEGMENTED: one (row, segment), unit id `u = row·spr + seg` |
| Grid | `sub_core_grids` if given, else every cluster of `device.compute_with_storage_grid_size()` (32 here). `num_clusters` comes from that set |
| Per-core work | `ttnn.split_work_to_cores(grid, U, row_wise=True)` gives each engaged cluster a contiguous unit range `[core_start, core_start + core_count)`. Within a cluster, thread `t` takes `core_count // T + (t < core_count % T)` units starting at `core_start + t·(core_count // T) + min(t, core_count % T)`. The host passes per-cluster RT arrays `unit_start[T]` and `unit_count[T]` (`RuntimeArgSchema(runtime_array_args=[ArrayArgDecl(name, size=T)])`). The kernel reads `get_arg(args::unit_start)[get_my_thread_id()]`. The split lives on the host only |
| Remainder | balanced by `split_work_to_cores` (group 1 / group 2) and by `% T` inside a cluster. A thread with `unit_count = 0` does nothing but `release_noc`. Row geometry: `tensor_rows = prod(shape[:-1])` with no padding in row-major. Segment geometry `seg_cols = ceil(tensor_w / spr)`, last segment shorter. Every count uses `ceil` |
| Finalizer | SEGMENTED: the cluster holding unit 0, thread 0. RT arg `is_finalizer` per cluster; common RT `finalizer_noc_x/y`, `num_records = U` |
| Engaged clusters on 32×6 | (1,1,2,151936): U=192 → 32 clusters × 6 threads. (1,1,32,4096): U=128 → 32 × 4. (4,8,32,256): U=1024 → 32 clusters × 6 threads (5–6 rows each). (1,1,2048,256): 32 × 6 (10–11 rows). (1,1,32,64): 32 × 1. (1,1,17,50): 17 × 1 |
| Bound | **compute, run on the DM harts**: each 2-byte element costs a scalar load, a key transform, a compare and a branch (several hart cycles), while one hart's NoC read delivers those 2 bytes in a small fraction of a cycle. The NEOs cannot help, because no index-carrying max exists for them (NEO-SCAN, deferred) |
| Thread plan | **(R, C, W) = (6, 0, 0)**: one DM kernel `argmax_scan` with `num_threads = SCAN_THREADS = 6`, each thread the reader, scanner and writer of its own range. No compute kernel. Every user DM hart works because the bound is the scan. R + W = 6 ≤ 6; the divisibility rules concern STRIDED rings, and the op has none. Host constant `SCAN_THREADS`, CT arg `scan_threads`. The kernel sizes its loops from the CT count and `get_my_thread_id()`, never `get_num_threads()` |
| Unit dealt to compute threads | none dealt to NEOs. Units are dealt to DM threads as contiguous ranges (above), with no ring between threads. Nothing is padded: a thread with fewer units does less work and no thread waits for another, except the finalizer's gather |
| Tests the plan needs | (1) **Threads ran**: a host probe flag `probe_scanner_ids=True` on `create_program_artifacts`. In ROWS each output index is replaced by the global scanner id `g = cluster_ordinal·T + t` of the producing thread. In SEGMENTED the finalizer writes, per row, the sum over merged records of `(g+1)`. The test checks both against the host split on (4,8,32,256) and (1,1,2,151936), and checks that every thread id 0..5 appears. (2) **One device session after another program**: `ttnn.to_layout(x, ttnn.TILE_LAYOUT)` (a tt-metal program), then `argmax(x)`, then the result is checked. (3) **Before one of tt-metal's own**: `argmax(x)`, then `ttnn.to_layout(ttnn.to_layout(x, ttnn.TILE_LAYOUT), ttnn.ROW_MAJOR_LAYOUT)` in the same session, with both results checked. (4) **Regime-pinned**: assert `select_regime()` returns ROWS for (4,8,32,256) and SEGMENTED for both model shapes on the device grid, and run one shape of each regime with `sub_core_grids` = a 1-cluster set (S_total = 6: (1,1,2,151936) → SEGMENTED spr=3, (1,1,32,64) → ROWS) and a 4×2 set. The results must equal the full-grid ones (R5) |

## Buffers (scratchpads and semaphore; this op has no DataflowBuffers)

| Semantic Name | Kind | Entry size | Count | Sizing rationale | Format | Producer | Consumer | Sync | Lifetime |
|---------------|------|-----------|-------|------------------|--------|----------|----------|------|----------|
| `sp_blocks` | `ScratchpadSpec`, bound to `argmax_scan` | `block_bytes` | `SCAN_THREADS × in_depth` slots; thread t owns `[t·in_depth·block_bytes, (t+1)·in_depth·block_bytes)` | spans `rows` → `block_rows` and `w` → `block_cols` (the block itself). Streams over the scanner's range | none (raw bf16 bytes; scratchpad touched only by DM) | NoC reads issued by thread t | thread t's scan | explicit: user trids `0..in_depth−1`, barrier per block, `scoped_lock` invalidate | whole program, per block |
| `sp_out_stage` | `ScratchpadSpec` | `align_up(out_stage_rows·4, l1_align)`; `out_stage_rows` = `block_rows` (ROWS) or `tensor_rows` (SEGMENTED; ≤ S_total/2) | `SCAN_THREADS × OUT_DEPTH` | spans `rows` → `block_rows`; does not scale with `w` (one index per row) | none (uint32 is not a buffer format) | thread t's stores (flushed) | NoC writes by thread t | explicit: trids `2..3`, barrier before slot reuse | per block (ROWS) / end of program (finalizer) |
| `sp_partials` | `ScratchpadSpec` | 16 B (`RECORD_BYTES`) | `S_total` landing slots + `SCAN_THREADS` send slots | landing: one slot per unit, `U ≤ S_total` by predicate. Send: one per thread | none | landing slot u: the one scanner owning u (remote NoC write); send slot t: thread t | landing: finalizer thread; send slot t: thread t's NoC write | explicit: trid 4 plus the `records_arrived` semaphore | SEGMENTED only (allocated in both regimes, ≤ 3,168 B) |
| `records_arrived` | `SemaphoreSpec(target_nodes = engaged clusters)`, bound to `argmax_scan` | 1 word | 1 per cluster | counts records landed at the finalizer | — | every scanner (remote `up(noc, fx, fy, 1)`) | finalizer `wait(U)`, then `set(0)` | semaphore | SEGMENTED |

Every per-thread slice has one owning thread. Every landing slot has one writer (the scanner owning unit u) and one reader (the finalizer). No buffer has two consumers. The full ledger is in `l1_ledger.md`.

## Block Operation Realization

| # | Block operation | Block shape | Helper? | Input (name, size, state) | Output (name, size) | State after |
|---|-----------------|-------------|---------|---------------------------|---------------------|-------------|
| 1 | `load_block` | `block_rows × block_cols` | no (raw NoC) | input tensor pages `row` at `col_lo` | `sp_blocks[t][block_idx % in_depth]`, `block_rows·row_pitch` B | slot filled after the trid barrier |
| 2 | `scan_block` | same | no (scalar loop) | `sp_blocks` slot (invalidated) | running `(best_key, best_idx)` per row piece, in registers | slot free for the next `load_block` |
| 3 | `emit_block_results` | `block_rows` indices | no | completed rows' pairs | `sp_out_stage[t][block_idx % 2]` → output pages | write in flight on trid 2/3 |
| 4 | `send_partial` | 1 record | no | the segment's pair | `sp_partials` send slot t → finalizer landing slot u | record landed, semaphore +1 |
| 5 | `combine_partials` | `tensor_rows × spr` records | no | `sp_partials` landing zone (invalidated) after `wait(U)` | `sp_out_stage[t=0][0]`, `tensor_rows` indices | semaphore reset to 0 |
| 6 | `store_results` | `tensor_rows` indices | no | `sp_out_stage` (flushed) | output pages | — |

## API Mapping

| Block operation | Type | Function | File:Line | Template Params / Args | Input | Output | Which params are block knobs |
|-----------------|------|----------|-----------|------------------------|-------|--------|------------------------------|
| host program | raw_api | `ttnn.experimental.ProgramSpec`, `KernelSpec(num_threads=…)`, `ScratchpadSpec`, `SemaphoreSpec`, `KernelSpec.ScratchpadBinding/SemaphoreBinding/TensorBinding`, `RuntimeArgSchema(runtime_array_args=[ArrayArgDecl])`, `ttnn.generic_op` | `ttnn/cpp/ttnn-nanobind/program_spec.cpp:350-398` (semaphore, scratchpad), `:605-646` (RT arrays), `:688-718` (bindings), `:862-893` (ProgramSpec) | DM kernel hw config `ttnn.experimental.create_reader_datamovement_config()` | — | — | `num_threads = SCAN_THREADS`; scratchpad `size_per_node` from `block_bytes`, `in_depth`, `OUT_DEPTH`, `S_total` |
| `load_block` | raw_api | `Noc::async_read<NocOptions::TXN_ID>` + `async_read_barrier<TXN_ID>` | `tt_metal/hw/inc/api/dataflow/noc.h:252`, `:739` | src `TensorAccessor` args `{page_id, offset_bytes}` (`tt_metal/hw/inc/api/tensor/noc_traits.h:135-143`); dst `Scratchpad` args `{offset_bytes}` (`tt_metal/hw/inc/internal/local_mem/noc_traits.h:75-95`); `{.trid}` | input tensor | `sp_blocks` | `block_rows`, `block_cols`, `in_depth` |
| `scan_block` | raw_api | `Scratchpad<uint16_t>::local_mem()` / `CoreLocalMem::scoped_lock(n)` | `tt_metal/hw/inc/api/scratchpad.h:42`, `:129`; `tt_metal/hw/inc/api/core_local_mem.h:145-153` (invalidate on acquire, flush on release: `internal/scoped_lock_cache_ops.h`) | element count `block_rows·row_pitch/2` | `sp_blocks` | registers | `block_rows`, `block_cols` |
| `emit_block_results`, `store_results` | raw_api | `Noc::async_write<NocOptions::TXN_ID>` + `async_write_barrier<TXN_ID>` | `noc.h:411`, `:760` | src `Scratchpad` `{offset_bytes}`; dst `TensorAccessor` `{page_id, offset_bytes}`; `{.trid = 2 + k%2}` | `sp_out_stage` | output tensor | `out_stage_rows` (= `block_rows`), `OUT_DEPTH` |
| `send_partial` | raw_api | `Noc::async_write<TXN_ID>` to `UnicastEndpoint`, then `Semaphore::up(noc, x, y, 1)` | `tt_metal/hw/inc/api/dataflow/endpoints.h:13`, `:76-107`; `tt_metal/hw/inc/api/semaphore.h:201-208` | dst `{noc_x, noc_y, addr = sp_partials base + 16·u}`; `{.trid = 4}` | `sp_partials` send slot | finalizer landing slot | — |
| `combine_partials` | raw_api | `Semaphore::wait(U)`, `Semaphore::set(0)` | `semaphore.h:127`, `:170` | `U` = common RT `num_records` | `sp_partials` | `sp_out_stage` | `segments_per_row` |
| `release_noc` | raw_api | `noc_async_read_set_trid(0, noc_id)`, `noc_async_write_set_trid(0, noc_id)` | `.claude/references/quasar_op_examples/toy_thread_plan/kernels/reader_implicit.cpp` (last statement) | — | — | — | — |
| thread id | raw_api | `get_my_thread_id()` | `tt_metal/hw/inc/api/kernel_thread_globals.h:155` | — | — | — | `scan_threads` (CT) |

**Helpers considered and rejected** (all phases are raw_api because the op has no compute kernel. The kernel library's compute helpers act on tiles in DEST, and the scan runs on DM harts over row-major bytes):

| Candidate | File:Line | Why it cannot express this op |
|-----------|-----------|-------------------------------|
| `compute_kernel_lib::reduce<PoolType::MAX, REDUCE_ROW>` | `ttnn/cpp/ttnn/kernel_lib/reduce_helpers_compute.hpp:526-531`, `:654-670` | it returns the maximum **value** per row of tiles; argmax needs the **index**, with lowest-index ties. It also needs tiled DFB input and a scaler; the input is row-major and the op has no NEO stage (NEO-SCAN deferred) |
| `compute_kernel_lib::tilize` | `ttnn/cpp/ttnn/kernel_lib/tilize_helpers.hpp:206` | it would only feed a compute path that has no index-carrying max on Quasar (Target, Unavailable here) |
| `dataflow_kernel_lib::McastArgs` / `ttnn.Mcast` | `ttnn/cpp/ttnn/kernel_lib/mcast/kernel/mcast_args.hpp:18-28` | it is a 1→N broadcast channel; the combine here is an N→1 gather of distinct 16-B records. No operand is reuse-shared (operand-reuse check). A `Scratchpad` cannot be a multicast destination either (`internal/local_mem/noc_traits.h:97-101`) |
| `reduce_helpers_dataflow` scaler prep | `ttnn/cpp/ttnn/kernel_lib/reduce_helpers_dataflow.hpp` | no reduce helper is used, so no scaler exists |

Missing block operation, for the record: a DM-side "argmax over a row-major L1 range" helper does not exist. `scan_block` is that operation, written once in the kernel as an inline function that takes `(slot, block_rows, block_cols, row_pitch, col_base)` and folds into the per-row pairs. The kernel library would gain from such a helper, and nothing in this op depends on its absence.

## Key Risks and Gotchas

| Risk | Why it bites here | Mitigation in this design |
|------|-------------------|---------------------------|
| DM L2 cache vs NoC | Quasar DM harts read L1 through a cache (`scoped_lock_cache_ops.h`: `invalidate_l2_cache_range` / `flush_l2_cache_range`). A block landed by the NoC can be read stale, and staged indices can reach DRAM unflushed | `scoped_lock` over the slot before every scan and over the landing zone before the merge. Staged indices and records are written inside a `scoped_lock` whose release (flush) precedes the NoC write |
| Transaction tag | An untagged tt-metal program after this one may inherit a tag | every read, write and record transfer carries a user trid (0..4 ≤ 7). The semaphore atomic follows a tagged write. The last statements reset the read and write trids to 0. Required tests (2) and (3) of the plan |
| Ties and signed zeros (R1) | the `ties` and `signed_zeros` regression cases. A merge that is not in column order, or a raw-bit compare without −0 normalisation, picks a wrong index | key transform with `0x8000 → 0`. Strict `>` in the scan. The merge walks slots `r·spr..` in ascending segment order with strict `>`. The running pair starts from the piece's first element |
| Disjoint sub-page writes to one DRAM page | ROWS scanners write neighbouring 4-B index runs of the same output page (e.g. 6 scanners into one 128-B page for (4,8,32,256)). NoC writes are byte-granular here, and the DRAM side's handling of concurrent masked writes is unverified | the acceptance shapes exercise it. If any write is lost, deal ROWS ranges in multiples of `ttnn.get_dram_alignment()/4` rows aligned to page starts, so no two scanners share a DRAM word. That is a change to `split` only |
| Semaphore persistence across a cached re-run | a program-cache hit may not re-zero the semaphore | the finalizer calls `set(0)` right after `wait(U)`. No other writer touches it after its `up` |
| Remote scratchpad address | the sender writes into the finalizer's `sp_partials` | the scratchpad base comes from a common RT arg (`Scratchpad` ctor, `scratchpad.h:55-59`), so it is the same on every cluster, and sender address = own base + 16·u. The finalizer's NoC coordinates come from `device.worker_core_from_logical_core` on the host |
| Regime depends on the grid | S_total = clusters × 6. A board or `sub_core_grids` with fewer clusters flips shapes between ROWS and SEGMENTED | `select_regime()` is one host function. Regime-pinned tests run both regimes on fixed sub-grids, and R5 requires equal results |
| Threads with no units | for ≤ 32 units only thread 0 of each cluster works | `unit_count[t] = 0` → the thread skips straight to `release_noc`. The finalizer waits for exactly U ups, never per thread |
| Formats of scratchpads | uint16/uint32 are refused as buffer formats | the scratchpads carry no `data_format_metadata` (DM-only access) |
| One dispatch | the rules forbid sequencing programs | split, scan, gather, merge and store all run in one `ttnn.generic_op` call. Validation and allocation stay on the host |
| W = 1 and tiny rows | 2-byte pages, many reads per block | row pitch aligned to 16 B. The block takes `floor(block_bytes/16)` rows. The result is index 0 by the first-element init |
