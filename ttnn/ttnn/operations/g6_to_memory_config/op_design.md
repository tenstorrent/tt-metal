# Operation Design: g6_to_memory_config

## Overview

| Field | Value |
|-------|-------|
| Classification | data_movement (no compute kernel, no CB between RISCs except the staged regime's one staging ring) |
| Goal | Return a new TILE tensor with the input's shape, dtype and stored bits, placed exactly as `memory_config` names. One program; each tile page is read once from its source place and written once to its target place, through the L1 of the core that moves it. |
| Math | `y[p] = x[p]` for every tile page `p`; page `p = r * tensor_tile_cols + c` has the same index in both tensors (same shape, same per-image tile padding); only its address changes, `addr_in(p) -> addr_out(p)` |
| References | `docs/explorations/g6_to_memory_config/{DOSSIER.md,math.md,tensix.md,risks.md,lessons.md}`; `eval/golden_tests/g6_to_memory_config/{feature_spec.py,helpers.py}`; `.claude/references/{blocking-model.md,l1-footprint-discipline.md}`; catalog `ttnn/ttnn/operations/examples/{double_buffer,trid_double_issue,read_coalescing,split_reader,bank_stagger,noc_placement,width_split,tile_reorder}/README.md` |

## Parameters

| Name | Type | Required | Valid Range | Default | CT/RT |
|------|------|----------|-------------|---------|-------|
| `input_tensor` | `ttnn.Tensor` | yes (positional) | on device, TILE, rank 2..4, any placement, shard spec ROW_MAJOR if sharded | — | buffer address RT; page size, accessor args CT |
| `memory_config` | `ttnn.MemoryConfig` | yes (positional) | DRAM/L1; INTERLEAVED or HEIGHT/WIDTH/BLOCK sharded; shard shape multiples of 32, ROW_MAJOR, grid holding ≥ the shards the padded tile grid needs | — | output accessor args CT; output address RT; shard rectangles RT |

No dtype, layout or compute_kernel_config parameter. Index axes: none.

## Tensors

### Input

| Property | Requirement |
|----------|-------------|
| Shape | `(..., H, W)`, rank 2..4; `tensor_tile_rows = prod(leading) * ceil(H/32)`, `tensor_tile_cols = ceil(W/32)` (per-image padding, never `floor(L*H/32)`) |
| Dtype | Phase 0: bfloat16. Refinement: bfloat8_b, bfloat4_b (page sizes 1088 / 576 B; same kernels, page size is a CT arg) |
| Layout | TILE |
| Memory | DRAM or L1; interleaved or height/width/block sharded (any grid form: several ranges, single-core ranges, off-origin, more cores than shards, uneven last shard) |

### Output

| Property | Value |
|----------|-------|
| Shape | `input_tensor.shape` (same padded shape) |
| Dtype | `input_tensor.dtype` |
| Layout | TILE |
| Memory | exactly `memory_config` (allocated with `ttnn.allocate_tensor_on_device(shape, dtype, TILE, device, memory_config)`; never the input object) |

## Blocking Model

The op's index space is the **tile-page grid** `tensor_tile_rows × tensor_tile_cols`. Every output page depends on exactly one input page, so both axes are independent; the only indivisible unit is the page itself (block formats share an exponent per 16-datum face row — a page moves whole). There is **no compute**: the "fixed per-block cost" a block amortizes is the NoC barrier, CB handshakes and per-transaction command issue, and *the work-split is decided by locality*: which core touches a page determines how many NoC crossings the page pays.

The scheme: **the mover of a page is a core that already holds one end of it.** If the target is L1-sharded the target shard owner pulls; else if the source is L1-sharded the source shard owner pushes; else (both ends are banks — interleaved DRAM/L1, or DRAM-sharded) a full-grid set of movers stages pages through their own L1. These are three regimes of one scheme (each is the logical shard of the placement that owns a resident end), selected by an exact host predicate (Regimes table).

### Axes

| Axis | Character (+ one-clause reason) | Extent knob | Phase 0 value | Knob source | Core-assignment | Later unlock |
|------|--------------------------------|-------------|---------------|-------------|-----------------|--------------|
| `tile_row` (leading dims folded × `ceil(H/32)`) | independent — no output page spans rows; leading dims fold because the op treats every image identically and pages are image-major contiguous ids | PULL/PUSH: `block_tile_rows` = this core's shard tile rows (actual, ≤ shard nominal); STAGE: folded into `stage_block_pages` (flat page range) | PULL/PUSH: whole shard rows of the owner (one block per core); STAGE: see `stage_block_pages` | PULL: per-core RT arg from the **output** shard spec; PUSH: per-core RT arg from the **input** shard spec; STAGE: host constant `STAGE_BLOCK_PAGES_CAP` + per-core page range from `split_work_to_cores` | PULL: output shard owners (height/block shards cut rows); PUSH: input shard owners; STAGE: contiguous page ranges spread over the full compute grid | knob-turn (STAGE grid size, block size); owner-assist = scheme-change (deferred row) |
| `tile_col` (`ceil(W/32)`) | independent — no output page spans columns | PULL/PUSH: `block_tile_cols` = this core's shard tile cols (actual; uneven last shard narrower); STAGE: folded into the flat page range | PULL/PUSH: whole shard cols (one block per core); STAGE: flat range | as above | PULL/PUSH: width/block shards cut cols; STAGE: flat page range | knob-turn |
| `page` (within-tile bytes) | indivisible — a block-format group of 16 datums shares one exponent; cutting it re-quantizes; bfloat16 trimming is a perf lamp only | `page_bytes` (transfer granularity) | whole page (`input_tensor.buffer_page_size()`) | CT arg `page_size` = `input_tensor.buffer_page_size()` (== output's) | never split across cores | none (mechanism cap) |
| `run` (contiguous pages coalesced into one NoC transfer) | derived, not split — pages adjacent in **both** source and destination memory | `run_pages` = `min(src contiguous run, dst row remainder)` computed per run in-kernel | maximal run (1 for an interleaved end, up to a shard row for sharded↔sharded) | computed in kernel from `TensorAccessor::num_contiguous_pages` + `contiguous_page_stride()`; never a literal | inside the mover | none |

The three-knob table, one row per axis, no blanks:

| Axis | Character | Block factor (Phase 0) | Core assignment | Buffer depth | Why |
|---|---|---|---|---|---|
| `tile_row` | independent | PULL/PUSH: whole resident shard rows (one block); STAGE: part of a `stage_block_pages` flat range | PULL: output shard owner; PUSH: input shard owner; STAGE: `split_work_to_cores(full grid, num_pages, row_wise=True)` | PULL/PUSH: none — the shard itself is the destination/source (zero-copy), no ring; STAGE: `stage_depth = 2` | the shard hands the block size (one block, sharded default); STAGE's ring is sized by the overlap lamp, not by the tensor |
| `tile_col` | independent | same as `tile_row` | same as `tile_row` | same | width shards cut this axis; STAGE flattens both axes into a page range because interleaved pages of consecutive ids sit in consecutive banks (no 2D locality to exploit) |
| `page` | indivisible | 1 whole page | never split | — | bit-exactness of block formats; whole-page transfers are the catalog's faster granularity (`tile_reorder`) |
| `run` | derived | maximal | inside mover | — | one NoC transaction per maximal contiguous run (`read_coalescing`: 3.24× from coalescing) |

**Stage data:** the scheme introduces no intermediate result — the STAGE ring holds the same pages in flight, with the same axes, owned by the mover; no core collects or coordinates. No stage computes on one core while others wait.

**Stall-shadow check.** Waiting stages: (a) a mover's read barrier before it pushes a staged block (STAGE reader) — the writer on the other RISC drains the previous block in that window (depth 2 is exactly this reorder); (b) PULL/PUSH end-of-block barrier — nothing else is left on that core; the only work independent of the wait would be another core's pages (owner-assist, deferred row). No floating-point order exists to gate.

**Operand-reuse check.** One operand; it varies along every split axis (each page read once). No broadcast row.

**Commit.** Phase 0 builds all three regimes (PULL, PUSH, STAGE) and the selection predicate; Phase 0 is multi-core in every regime (shard grid or full compute grid). Every knob above is a parameter: `stage_block_pages` and `stage_depth` are host constants threaded as CT args, per-core extents are RT args, the STAGE grid is `device.compute_with_storage_grid_size()` passed through `split_work_to_cores` (never inlined).

### Buffer-depth knobs

| CB | Depth knob | Phase 0 value | What the depth buys |
|----|------------|---------------|---------------------|
| `cb_staged_pages` (STAGE only) | `stage_depth` (host constant `STAGE_DEPTH`) | 2 | reader fills block k+1 (NoC0 reads) while writer drains block k (NoC1 writes); catalog `double_buffer`: barrier batching + 2×block ring stack to 2.78× |
| `cb_output_shard` (PULL only, zero-copy on the output shard) | none — capacity is the shard | n/a | the destination itself; no ring, no handshake |
| `cb_input_shard` (PUSH only, zero-copy on the input shard) | none — capacity is the shard | n/a | the source itself; no ring, no handshake |

### Mechanism caps

| Mechanism | Cap on which extent | Clamp | What happens unclamped |
|-----------|--------------------|-------|------------------------|
| Block-format shared exponent per 16-datum face row | `page_bytes` | transfer whole pages only (bfloat16 trim is a lamp, never applied to block formats) | re-quantized values; exact-equality cell fails |
| Coalesced run must be contiguous in **both** memories | `run_pages` | `run = (src.contiguous_page_stride() == 1) ? min(src.num_contiguous_pages(p, row_end_page), dst_row_remaining) : 1` (`tensor_accessor.h:180,192`; interleaved accessor's stride is the bank count, `tensor_accessor.h:561`) and the symmetric clamp on the destination side for PUSH/STAGE writes to a sharded target | an interleaved "run" read as consecutive bytes copies neighbouring bank pages → wrong values |
| Shard L1 layout uses the **nominal** shard shape | local page offset in an uneven last shard | local offset = `(r_local * shard_tile_cols_nominal + c_local) * page_size`; loop bounds use the actual extents | uneven shards (2063, 2069, 2071, Panoptic 0/2/7/36) land at wrong offsets |
| Ring-wrap invariant of `cb_staged_pages` | pushed/popped count | every push/pop is the nominal `stage_block_pages`; a ragged last block transfers `last_block_pages` but still pushes/pops nominal | CB pointers desync → hang on the next block or next program |
| DRAM read alignment 64 B, L1 16 B | page offsets | all tile page sizes (2048, 1088, 576) are multiples of 64; `buffer_aligned_page_size() == buffer_page_size()` asserted on host for coalescing | misaligned reads corrupt data |
| BB-05 L1 budget 1347776 B / core | `stage_block_pages × stage_depth` plus output L1 pages | host predicate in Work Distribution; `STAGE_BLOCK_PAGES_CAP = 16` | refused launch beside the model's tensors |

### Regimes

Boundary named for "minimum": **each input page that lives in DRAM crosses DRAM once, each output page that lives in DRAM crosses once; each page whose source core ≠ target core crosses the NoC once.**

| Regime | Status | Predicate | Block | Data movement vs. minimum | What a bigger block buys |
|--------|--------|-----------|-------|---------------------------|--------------------------|
| **PULL** — target-shard owner reads its pages from wherever they are, straight into its own shard | built | `out.is_sharded() and out.buffer_type == L1` | one block per owner core = its whole output shard, `block_tile_rows × block_tile_cols` (actual extents) | **minimum**: one NoC read per maximal run (DRAM pages read once; remote L1 pages read once; same-core pages a loopback read), zero writes | it *is* the coarsest (whole shard): one read barrier per call per core instead of per page; one init of the accessor walk |
| **PUSH** — source-shard owner writes its pages from its own shard to wherever they go | built | `not PULL and in.is_sharded() and in.buffer_type == L1` | one block per owner = its whole input shard, `block_tile_rows × block_tile_cols` (actual) | **minimum**: one NoC write per maximal run (target interleaved → run 1), zero remote reads; DRAM pages written once | coarsest: one write barrier per core per call |
| **STAGE** — full-grid movers read a page range into their own L1 ring and write it out | built | `not PULL and not PUSH` (interleaved↔interleaved in Phase 0; DRAM-sharded target in the refinement) | `stage_block_pages = min(core_pages_max, STAGE_BLOCK_PAGES_CAP=16)` pages per block, `stage_depth = 2` | DRAM minimum met (1 read per DRAM input page, 1 write per DRAM output page). NoC: **2 crossings per page** (bank → mover, mover → bank); minimum 1 only when one end is an L1 bank owned by the mover (bank-owner row below); for DRAM→DRAM (refinement) 2 is the structural minimum (no NoC path bypasses a core's L1) | fewer read barriers and CB handshakes per page (catalog: flat between 4 and 64 pages/barrier; 8–16 sweet spot) — capped by the overlap lamp |
| **BANK-OWNER** — for an L1-interleaved end, the mover of page `p` is the core of L1 bank `p mod num_l1_banks`, reading/writing its own bank locally | deferred | would replace STAGE when `out` (or `in`) is L1-interleaved | per-core: pages of its own bank, coarsest run | saves one NoC crossing per page vs STAGE (27 bf16 + 3 bf8 cases: 1 transfer per page instead of 2) | same as STAGE |
| **OWNER-ASSIST** — non-owner cores move part of a large shard (staged) into/out of the owner's shard | deferred | PULL/PUSH with `pages_per_owner` ≫ cores-per-owner ratio (e.g. 2072: 64 pages on each of 8 owners) | STAGE block on helper cores | adds 1 NoC crossing per assisted page (owner path is 1, assist is 2) in exchange for occupancy | — |
| **VALID-BYTES** — move only the valid rows of the partial last tile row | rejected as a regime — superseded by the "partial-tile trim" perf lamp: it changes one transfer's size inside the same block operation, no structure change | — | — | bytes only, same transaction count | — |
| **TWO-STEP** (to an interleaved intermediate, then to target) | rejected — dead end, forbidden by the spec (traffic ×2); PULL/PUSH/STAGE supersede it everywhere | — | — | 2× every tier | — |

Deferred reasons (positive):
- **BANK-OWNER:** the host must know the logical core of each L1 bank to put the mover of bank `b` on it; the Python program-descriptor API exposes only the DRAM-bank→worker map (`ttnn/cpp/ttnn-nanobind/device.cpp:811`), not an L1-bank→core map, so it cannot be built in the generic-op layer without a new binding. STAGE keeps it reachable: the same reader with the write replaced by a local landing (`TensorAccessor::is_local_page`, `tensor_accessor.h:312`) and a per-bank page stride is exactly this regime; only the core-assignment changes.
- **OWNER-ASSIST:** every Phase 0 owner moves at most 64 pages (2072/2080/2135/2171: 64 bf16 pages per source shard), i.e. one block that the NoC issues in a single burst series; the regime adds a NoC crossing per page to every assisted page and a second kernel group on helper cores (BB-17 ranges). It stays reachable because STAGE already accepts a sharded L1 accessor on either side: assisting is STAGE run over a sub-range of an owner's pages, selected by a predicate on `pages_per_owner`. The grid-synchronization lamp names the experiment.

**Regime-selection function (pinned):**
```
def select_regime(input_tensor, memory_config):
    if memory_config.is_sharded() and memory_config.buffer_type == ttnn.BufferType.L1: return "pull"
    if input_tensor.memory_config().is_sharded() and input_tensor.memory_config().buffer_type == ttnn.BufferType.L1: return "push"
    return "stage"
```
Phase 0 case counts (bf16, 86 cases): PULL 31 (DRAM-int→L1 height/width 16, L1-int→L1 width 3, L1 width→width 10, L1 height→block 2); PUSH 28 (L1 height/width/block → DRAM/L1 interleaved); STAGE 27 (DRAM-int→L1-int). Refinement: 12 bf8 cases (3 STAGE, 9 PUSH), 7 bf4 DRAM width targets (STAGE). The op exposes the plan a call takes (BB-08): `g6_to_memory_config.select_regime(input_tensor, memory_config)` is public and returns the same string the dispatch uses. Regime-pinned tests: the acceptance test pins at least two cases per regime.

### Traffic ranking

| Candidate split (who moves page p) | DRAM crossings | NoC crossings per page | Fan-out / pattern | Rank |
|---|---|---|---|---|
| target-shard owner (PULL) when target is L1-sharded | minimum | 1 (0 bytes across the mesh when source is on the same core) | many readers, each reading its own pages; reads land locally | 1 for L1-sharded targets |
| source-shard owner (PUSH) when source is L1-sharded | minimum | 1 | each owner writes its own pages | 1 for L1-sharded sources (tie with PULL for sharded→sharded; PULL taken: the consumer's own core waits for its own data, and the destination shard row is the contiguous side, so runs are bounded only by the source shard edge) |
| L1-bank owner (BANK-OWNER) when one end is L1-interleaved | minimum | 1 | per-bank stride walk | 1 for L1-interleaved ends — deferred (API) |
| full-grid stager (STAGE) | minimum | 2 | contiguous page range per core, row-wise core line | 2; the only option when both ends are DRAM |
| split by `tile_col` / `tile_row` into a fixed rectangle unrelated to either placement | minimum | 2 | — | dominated by STAGE (no locality gained, worse balance) — not listed as a regime |
| dependent-axis split + combine | n/a — no dependent axis | — | — | n/a |

Chosen: the locality-ordered predicate above. Occupancy is not used to override it; the grid-sync lamp records where occupancy may still win.

### Block schedule

PULL (one kernel on each output-shard core, NCRISC / NoC0):
```cpp
// num_blocks_this_core = (core holds a shard) ? 1 : 0   (shard_blocks_per_core knob, Phase 0 = 1)
for (uint32_t block_idx = 0; block_idx < num_blocks_this_core; ++block_idx) {
    pull_block(block_idx);   // block_tile_rows × block_tile_cols pages: one noc read per maximal run, src→local shard
    settle_block(block_idx); // one noc_async_read_barrier
}
```
PUSH (one kernel on each input-shard core, BRISC / NoC1):
```cpp
for (uint32_t block_idx = 0; block_idx < num_blocks_this_core; ++block_idx) {
    push_block(block_idx);   // block_tile_rows × block_tile_cols pages: one noc write per maximal run, local shard→dst
    settle_block(block_idx); // one noc_async_write_barrier (BB-03: also guarantees arrival before the program ends)
}
```
STAGE (reader NCRISC/NoC0 + writer BRISC/NoC1 on every core of the split):
```cpp
for (uint32_t block_idx = 0; block_idx < num_blocks_this_core; ++block_idx) {
    load_block(block_idx);   // reader: reserve stage_block_pages, issue reads of this block's pages, one read barrier, push nominal
    store_block(block_idx);  // writer: wait nominal, issue writes, one writes-flushed (barrier at kernel end), pop nominal
}
```

| Operation | Block shape | Resident across it | Intended frequency of fixed costs |
|---|---|---|---|
| `pull_block` | `block_tile_rows × block_tile_cols` (actual shard extents; nominal shard stride for offsets) | the output shard (it is the destination) | accessor construction once per kernel; one NoC command per run; zero CB handshakes; **one read barrier per block = per call per core** |
| `push_block` | `block_tile_rows × block_tile_cols` (actual input-shard extents) | the input shard (it is the source) | one NoC command per run; zero CB handshakes; one write barrier per call per core |
| `load_block` | `stage_block_pages` pages (`last_block_pages` on the ragged last block) | none beyond the ring | one `cb_reserve_back`/`cb_push_back` and one read barrier per block |
| `store_block` | same | none | one `cb_wait_front`/`cb_pop_front` per block; `noc_async_writes_flushed` before each pop (the slot may be reused); one `noc_async_write_barrier` at kernel end |

Every kernel ends with its barrier issued after the last transfer (BB-03), including cores with zero work (they return without issuing anything).

### Perf lamps

| Lamp | Why the default may be wrong here | Nearby alternative to measure |
|------|-----------------------------------|-------------------------------|
| Grid-synchronization (STAGE occupancy) | decode-sized STAGE calls are not in Phase 0, but small prefill ones are (e.g. 2089: 2×48 = 96 pages; 2105 3×136) — 1 page per core on 110 cores leaves launch/finish skew dominant (R2) | `split_work_to_cores` over fewer cores with ≥ 4 pages each (the catalog's per-barrier sweet spot) |
| Grid-synchronization (owner-only PULL/PUSH) | 8 owners × 64 pages (2072, 2135, 2171) at ~115 ns per tile transaction per core (`double_buffer`) vs a full grid | OWNER-ASSIST (deferred row): STAGE helpers on a sub-range |
| Overlap (STAGE block) | `STAGE_BLOCK_PAGES_CAP = 16` vs whole per-core range (94 pages for 2078) — the whole range serializes read against write | 8, 16, 32 pages, and depth 3 with `ahead = depth/2` (`trid_double_issue`) |
| Split issue across both RISCs (PULL/PUSH) | one RISC issues every NoC command of the owner; whole-tile split gave ~1.2× (`split_reader`) | halve each owner's run list between NCRISC (NoC0) and BRISC (NoC1) — a second kernel on the same cores, same core ranges |
| Partial-tile trim (bfloat16 only) | 67 of 105 cases have `H mod 32 ≠ 0`; when `H mod 32 ∈ [1,16]` the valid rows of a last-tile-row page are exactly faces 0–1 = its first 1024 B | transfer `page_size/2` for those pages (bfloat16 only; block formats keep whole pages); same transaction count, half the bytes on decode-shaped tensors |
| Bank stagger (PUSH/STAGE writes to DRAM interleaved, STAGE reads) | all movers start at bank 0 (`bank_stagger`: 1.01–1.07× interleaved) | start mover k's walk at page offset `k mod num_dram_banks` within its range, wrap |
| STAGE core line placement | `split_work_to_cores(row_wise=False)` gives columns (`noc_placement`: column 2.9× slower) | `row_wise=True` is the default chosen; measure diagonal on Blackhole |

## Dataflow Strategy

| Stage | Format | Mechanism | Notes |
|-------|--------|-----------|-------|
| locate | page ids | host computes per-core shard rectangle `(row0, block_tile_rows, col0, block_tile_cols)` from the shard spec; kernel walks `p = r * tensor_tile_cols + c` | input and output share page ids (same padded shape) |
| PULL move | whole tiles, byte copy | `noc_async_read(src_accessor.get_noc_addr(p), shard_base + local_offset, run_pages * page_size)` | `shard_base = get_write_ptr(cb_output_shard)` (CB backed on the output shard, zero-copy); source may be DRAM-int, L1-int or a remote/local L1 shard |
| PUSH move | whole tiles | `noc_async_write(shard_base + local_offset, dst_accessor.get_noc_addr(p), run_pages * page_size)` | `shard_base = get_read_ptr(cb_input_shard)` (CB backed on the input shard, zero-copy — the resident shard is never re-read through an accessor) |
| STAGE move | whole tiles | reader `noc_async_read(src.get_noc_addr(p), cb slot)` → `cb_staged_pages` → writer `noc_async_write(cb slot, dst.get_noc_addr(p))` | ring depth 2; output accessor may be interleaved (Phase 0) or DRAM-sharded (refinement: same kernel, sharded accessor CT args) |
| completion | — | read barrier (PULL, STAGE reader), write barrier (PUSH, STAGE writer) before each kernel returns | BB-03 |

**TARGET placement axis as regime rows.** Every `*_SHARDED` value cuts only independent axes, so each is a **knob-turn** (a different core-assignment of the same block operation), never a scheme-change: L1 height/width/block targets → PULL with the output shard as the block; L1-sharded sources → PUSH with the input shard as the block; DRAM width-sharded targets (refinement, bf4 weights) → STAGE with a sharded DRAM output accessor — no new kernel, the EXCLUSION is lifted by removing it once the bf4 page path is tested. Block formats are a knob-turn too (page size is a CT arg; whole pages move).

Inter-Tensix contract: none (no semaphores, no multicast). Deferred OWNER-ASSIST would write into the owner's shard with plain NoC writes; completion is program end, no handshake needed.

## Work Distribution

| Field | Value |
|-------|-------|
| Work unit | PULL/PUSH: the owner's whole shard (one block); STAGE: a `stage_block_pages` block of a contiguous page range |
| Grid | PULL: output shard grid (`memory_config.shard_spec.grid`, merged into maximal rectangles with `CoreRangeSet.merge` (`ttnn/cpp/ttnn-nanobind/tensor.cpp:652`) for the kernel's core ranges, BB-17); PUSH: input shard grid, same treatment; STAGE: `split_work_to_cores(device.compute_with_storage_grid_size(), num_pages, row_wise=True)` |
| Per-core work | PULL/PUSH: shard `i` ↔ core: height/width sharded — `i`-th core of `ttnn.corerange_to_cores(shard_grid, row_wise=True)` in the stored range order; block sharded (ROW_MAJOR) — core at bounding-box offset `(x - x0, y - y0)` holds shard `(shard_row = y - y0, shard_col = x - x0)`. Actual extents: `block_tile_rows = min(shard_tile_rows, tensor_tile_rows - row0)`, `block_tile_cols = min(shard_tile_cols, tensor_tile_cols - col0)`; a core whose shard index ≥ number of shards gets `num_blocks_this_core = 0`. STAGE: `start_page`, `core_pages` from `split_work_to_cores` (two groups); `num_blocks_this_core = ceil(core_pages / stage_block_pages)`, `last_block_pages = core_pages - (num_blocks_this_core-1)*stage_block_pages` |
| Remainder | `tensor_tile_rows = prod(shape[:-2]) * ceil(H/32)`; `tensor_tile_cols = ceil(W/32)`; number of shards along rows = `ceil(tensor_tile_rows / shard_tile_rows)`, along cols `ceil(tensor_tile_cols / shard_tile_cols)`; ragged STAGE tails via `last_block_pages` |

Host validation (ValueError) is the first thing `validate()` does — before the INPUT_TAGGERS / SUPPORTED / EXCLUSIONS checks, so a rank-5 or host tensor raises ValueError rather than a support refusal — and the entry point calls `validate(input_tensor, memory_config)` as its first line. The taggers project the scenario dict from the tensor (`shape`, `dtype`, `layout`, `input_tensor.memory_config()`) and from `memory_config` with the axis names of `feature_spec.TARGET`. Checks: input on device and TILE; rank 2..4; for input and target shard specs — ROW_MAJOR, shard shape multiples of 32, and coverage: height `cores ≥ ceil(rows/shard_rows)`; width `cores ≥ ceil(cols/shard_cols)`; block `bbox_rows ≥ shard rows needed and bbox_cols ≥ shard cols needed` (more cores than shards is valid). BB-05 host refusal: `op_l1_bytes + output_l1_bytes_per_core > 1347776` raises ValueError (never hit by the golden INPUTS).

Program-description cache (BB-07): the host memoizes the built descriptor in a dict keyed by `(regime, padded shape, dtype, input memory_config incl. shard spec grid/shape/orientation/buffer type, memory_config likewise, device grid)`; on a hit only the buffer-address RT args are patched. generic_op's own hash covers kernel sources, CT args, core ranges and RT-arg counts, not RT values (`ttnn/cpp/ttnn/operations/generic/device/generic_op_device_operation.cpp:78`), and replays overridden RT args from the passed descriptor (`generic_op_spec_factory.cpp:44`) — so per-core shard rectangles stay correct as long as the passed descriptor is the right one, which the Python key guarantees. Addresses are never CT args.

## Circular Buffers

| Semantic Name | Index | Page Size | Num Pages | Sizing rationale | Format | Producer | Consumer | Lifetime |
|---------------|-------|-----------|-----------|------------------|--------|----------|----------|-----------|
| `cb_staged_pages` (STAGE only) | 0 | `page_size` | `stage_depth * stage_block_pages` | spans `stage_block_pages` (flat `tile_row×tile_col` range); streams over the core's range | input dtype (bit copy, never unpacked) | reader | writer | whole kernel |
| `cb_output_shard` (PULL only) | 1 | `page_size` | output shard pages (via `ttnn.cb_descriptor_from_sharded_tensor(1, output_tensor)`) | the output shard itself — zero-copy, no allocation of its own | output dtype | pull kernel (writes via NoC read landing; no push) | none in-program (consumer is the next op) | whole program |
| `cb_input_shard` (PUSH only) | 2 | `page_size` | input shard pages (`ttnn.cb_descriptor_from_sharded_tensor(2, input_tensor)`) | the input shard itself — zero-copy | input dtype | none in-program (the model's tensor) | push kernel | whole program |

`cb_output_shard` / `cb_input_shard` are used only for their base address (`get_write_ptr` / `get_read_ptr`); no reserve/push/wait/pop happens on them, so push count = wait count = 0 trivially, and each is touched by exactly one kernel. `cb_staged_pages`: reader pushes `stage_block_pages` per block × `num_blocks_this_core`; writer waits/pops the same.

## Block Operation Realization

| # | Block operation | Block shape | Helper? | Input CB (semantic name, pages, state) | Output CB (semantic name, pages) | CB state after |
|---|-----------------|-------------|---------|----------------------------------------|----------------------------------|----------------|
| 1 | `pull_block` | `block_tile_rows × block_tile_cols` | no (raw dataflow; see API Mapping) | source via `TensorAccessor` (no CB) | `cb_output_shard` (address only) | shard filled after `settle_block` |
| 2 | `settle_block` (PULL) | — | no | — | — | all reads landed |
| 3 | `push_block` | `block_tile_rows × block_tile_cols` | no | `cb_input_shard` (address only) | target via `TensorAccessor` | — |
| 4 | `settle_block` (PUSH) | — | no | — | — | all writes acknowledged |
| 5 | `load_block` | `stage_block_pages` | no | source via `TensorAccessor` | `cb_staged_pages`, push `stage_block_pages` | block visible to writer |
| 6 | `store_block` | `stage_block_pages` | no | `cb_staged_pages`, wait/pop `stage_block_pages` | target via `TensorAccessor` | slot freed after writes flushed |

## API Mapping

| Block operation | Type | Function | File:Line | Template Params / Args | Input CB | Output CB | Which params are block knobs |
|-----------------|------|----------|-----------|------------------------|----------|-----------|------------------------------|
| all | raw_api | `TensorAccessor::get_noc_addr(page_id)` | `tt_metal/hw/inc/api/tensor/tensor_accessor.h:165` | accessor from `TensorAccessorArgs<cta_offset>()` (CT args at the end of each kernel's CT list, host `ttnn.TensorAccessorArgs(tensor)`, `ttnn/cpp/ttnn-nanobind/tensor_accessor_args.cpp:25`) | — | — | none |
| `pull_block` / `push_block` / `load_block` | raw_api | `TensorAccessor::num_contiguous_pages(page_id, end_page_id)`, `contiguous_page_stride()` | sharded `tensor_accessor.h:180,192`; interleaved `tensor_accessor.h:561,568` | coalesce only when stride == 1 | — | — | `run_pages` (derived) |
| `pull_block`, `load_block` | raw_api | `noc_async_read(src_noc_addr, dst_l1, bytes)` | `tt_metal/hw/inc/api/dataflow/dataflow_api.h:547` | bytes = `run_pages * page_size` | — | — | run length |
| `push_block`, `store_block` | raw_api | `noc_async_write(src_l1, dst_noc_addr, bytes)` | `dataflow_api.h:876` | same | — | — | run length |
| `settle_block` (PULL), `load_block` | raw_api | `noc_async_read_barrier()` | `dataflow_api.h:1921` | once per block | — | — | block size sets frequency |
| `settle_block` (PUSH), writer end | raw_api | `noc_async_write_barrier()` | `dataflow_api.h:1951` | once per block / kernel end | — | — | — |
| `store_block` | raw_api | `noc_async_writes_flushed()` | `dataflow_api.h:1980` | before each `cb_pop_front` (departure, not arrival — not a substitute for the final barrier) | — | — | — |
| PULL / PUSH shard base | raw_api | `ttnn.cb_descriptor_from_sharded_tensor` (host) + `get_write_ptr`/`get_read_ptr` | `ttnn/cpp/ttnn-nanobind/program_descriptors.cpp:518` | cb index 1 / 2 | `cb_input_shard` | `cb_output_shard` | — |
| STAGE grid | raw_api (host) | `ttnn.split_work_to_cores(grid, num_pages, row_wise=True)` | `ttnn/ttnn/core.py:19` | — | — | — | grid size (knob) |

**Helpers considered and rejected** (apply to every raw_api row — there is no compute phase, so no compute helper is a candidate):

| Candidate | File:Line of the mismatch | Why it cannot express the shape |
|---|---|---|
| `dataflow_kernel_lib::set_read_state` / `read_with_state` (local copy) | `ttnn/cpp/ttnn/kernel_lib/local_copy_helpers_dataflow.hpp:74-75, 94-95` | self-aimed L1→L1 only, with the **source address baked into the NoC state** once per loop and single-packet size; every move here has a different remote source (or remote destination) per run and multi-page runs exceed one packet |
| `ttnn.Mcast` / `McastArgs` | `ttnn/cpp/ttnn/kernel_lib/mcast/kernel/mcast_args.hpp` | no operand is reuse-shared (operand-reuse check: each page has exactly one destination) |
| tilize / untilize / eltwise `copy` | `ttnn/cpp/ttnn/kernel_lib/eltwise_convenience.hpp` | they unpack and repack through DEST — forbidden (repacking re-quantizes block formats) and a needless compute pass |
| `PagesAddressIteratorSharded` | `tt_metal/hw/inc/api/tensor/pages_address_iterator.h:21` | walks one tensor's page ids in order; the mover walks a 2D shard rectangle of the *other* tensor and needs runs clamped by both sides — the implementer may use it inside a block op where it fits, not required |

## Key Risks and Gotchas

| Risk | Why it bites here | Mitigation in this design |
|------|-------------------|---------------------------|
| Shard-to-core mapping wrong for irregular grids | 2058 (16 single-core ranges), 2121, 2065/2128 (34 shards on 36 cores), 2154 (core (1,0)), block shards use 2D bbox positions | mapping rules pinned in Work Distribution; under `--dev` the pull kernel `ASSERT`s `dst_accessor.is_local_page(first_page)` / push kernel `src.is_local_page(first_page)` (`tensor_accessor.h:312`); acceptance test pins 2058, 2065, 2154, 2157 |
| Uneven last shard offsets | the shard's L1 stride is the nominal width | nominal-stride offset rule in Mechanism caps; acceptance pins 2063, 2071 |
| Coalescing an interleaved "run" | interleaved `num_contiguous_pages` counts pages at bank stride, not consecutive ids (`tensor_accessor.h:561-568`) | run = 1 unless `contiguous_page_stride() == 1` |
| Transfers in flight at kernel end (BB-03) | the op is nothing but NoC transfers; next program starts at once | every kernel ends in its barrier; zero-work cores issue nothing |
| Cache replaying wrong placement (BB-07, R8) | 2070/2073 differ only in target shard spec | Python descriptor cache key includes both full memory configs; CB/kernel core ranges differ anyway |
| Kernel core ranges fragmented (BB-17) | shard grids written as many single-core ranges | merge ranges into rectangles for kernel/CB core ranges; zero-work cores of a grid stay inside their rectangle instead of being carved out |
| Kernel text size (BB-18) | three regimes in one op | one small kernel file per role (`pull`, `push`, `stage_reader`, `stage_writer`); accessor kinds resolved at compile time by CT args, no runtime regime switch inside a kernel |
| Profiling zones (BB-21) | idle-wait vs work must be visible | zones `pull_block`, `push_block`, `load_block`, `store_block`, `wait_staged_block`, `wait_free_slot`, `settle_block` under the profiler define only; host can dump `{core: role}` per call |
| Returning the input when placements are equal | values right either way | always allocate a new output and dispatch |

## Structural impossibilities

None beyond `feature_spec.py`'s empty INVALID.

## Registry (for the implementer's op file)

| Item | Phase 0 value (from the spec; implementer declares) |
|---|---|
| INPUT_TAGGERS | `dtype`, `layout`, `rank`, `alignment` ("tile_aligned" when H and W are multiples of 32, else "h_non_aligned"), `in_scheme`, `in_buffer`, `out_scheme`, `out_buffer` — names and values exactly as `feature_spec.TARGET` |
| SUPPORTED | dtype [bfloat16]; layout [TILE]; rank [2,3,4]; alignment both; in/out scheme all four; in/out buffer dram, l1 |
| EXCLUSIONS | `{"out_scheme": "width_sharded", "out_buffer": "dram"}` only |
| Refinement path | bfloat8_b / bfloat4_b: page size is already a CT arg — PUSH and STAGE serve the 12 bf8 cases unchanged; DRAM width-sharded target: STAGE with a sharded DRAM output accessor, then drop the EXCLUSION |
