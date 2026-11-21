# Operation Design: embedding_gen

## Overview

| Field | Value |
|-------|-------|
| Classification | data_movement (a gather: no compute kernel, no compute config) |
| Goal | One device dispatch that copies, for every index, one row of the table into one row of the output, byte for byte. It reads the rows straight out of a row-major or tile-layout table and places them straight into a row-major or tile-layout output. |
| Math | `out[b, s, i] = weight[min(input_tensor[b, …, s], V-1), i]` for `b < B, s < S, i < D` (`[S, D]` for indices `[S]`). The clamp is the R2 guard: valid indices are never clamped. |
| References | `eval/specs/embedding_gen/spec.py` (authoritative); `.claude/references/targets/quasar.md`; `.claude/references/backends/ttsim.md`; `.claude/references/quasar_op_examples/` (host-program pattern); `tt_metal/impl/dataflow_buffer/DataflowBuffer.md` Part A; `ttnn/cpp/ttnn/operations/experimental/quasar/slice/device/slice_program_factory_rm.cpp:255-373` (an existing Quasar program with only data-movement kernels and a DM→DM ring) |

## Target

| Fact | Value | Source |
|------|-------|--------|
| Target | `quasar` on ttsim simulator | `.claude/references/targets/quasar.md` + `.claude/references/backends/ttsim.md` |
| Arch | quasar | target page |
| Compute grid | 8 × 4 = 32 clusters, fixed on this part (run time: `device.compute_with_storage_grid_size()`) | target page |
| L1 per core | 4,194,304 B (4 MiB) per cluster, shared by its NEOs and DM cores | target page |
| Unreserved budget | `ttnn.get_max_worker_l1_unreserved_size() - L1_RESERVE_BYTES`. On this target that is below 4 MiB. It never binds in this design: the per-ring cap of 1,048,560 B is the tighter limit (Mechanism caps). | target page |
| Host program | `ttnn.ProgramSpec` (Metal 2.0) launched by one `ttnn.generic_op` call. Pattern: `.claude/references/quasar_op_examples/toy_thread_plan/toy_thread_plan_program_artifacts.py` | target page |
| Threads per core | 4 NEO compute engines (C ∈ {1,2,4}); 6 user DM harts (R + W ≤ 6); on STRIDED rings R and W each divide C. This op has no compute kernel, so C = 0 and the divisibility rule does not apply. A DM→DM STRIDED ring needs integer thread ratios. | target page *Thread plan*; `tt_metal/impl/metal2_host_api/program_spec/validation/placement/work_unit.cpp:132,144` (a work unit needs kernels, not a compute kernel) |
| Unavailable here | **(1) `ttnn.ProgramDescriptor` / Metal 1 kernels and the legacy C++ program factory:** refused at kernel creation (target page). **(2) bfloat8_b and bfloat4_b:** no path on this part; never in SUPPORTED, refused by validate(). **(3) uint32 / uint16 as a DFB format:** `dfb.cpp:148`. This op's buffers are touched only by DM kernels, so they carry **no** `data_format_metadata` (allowed for DM-only rings, `dfb.cpp:272-282`). **(4) A DM kernel as producer and consumer of one ring:** `dfb.cpp:651-657`; the reader's private index storage is therefore a `ScratchpadSpec`. **(5) A NoC transfer without a transaction id:** not used. Every transfer of both kernels carries user trid 0. **(6) An ALL-pattern DM consumer with implicit sync:** `tests/tt_metal/tt_metal/api/dataflow_buffer/dfb_test_common.hpp:343-351`; not used. | target page *Not available* |

All worked numbers below are for this target: 32 clusters, bfloat16 (`esize = 2`), ring cap 1,048,560 B.

## Parameters

| Name | Type | Required | Valid Range | Default | CT/RT |
|------|------|----------|-------------|---------|-------|
| `input_tensor` | ttnn.Tensor uint32, ROW_MAJOR | yes | `[B, S]` or `[B, 1, 1, S]` (P0); `[S]` later (X2 refuses a non-1 middle dim) | — | tensor binding `indices` |
| `weight` | ttnn.Tensor | yes | `[1, 1, V, D]` (P0); `[V, D]` later. bfloat16 (P0), TILE or ROW_MAJOR. X1 refuses a non-1 leading dim. | — | tensor binding `table` |
| `padding_idx` | int / None | no, keyword | None or a row of the table. Changes no result (R6) and only feeds `tag_embeddings_type`. | None | host only |
| `layout` | ttnn.Layout / None | no, keyword | TILE, ROW_MAJOR; None → the table's layout (`canonical_layout`, R9) | None | CT flag `out_tiled` |
| `embeddings_type` | ttnn.EmbeddingsType | no, keyword | GENERIC / PADDED (needs `padding_idx`, X3) / BINARY (needs V == 2 when `padding_idx` is None, X4). Read with `type_word`. Changes no result (R7). | GENERIC | host only |
| `memory_config` | ttnn.MemoryConfig / None | no, keyword | None (→ the indices' buffer, R12) or `ttnn.DRAM_MEMORY_CONFIG`. Read with `canonical_memory`; a non-config is a TypeError (X5); L1 or sharded is refused by name (R13). | None | host only (output allocation) |

**Entry-point canonicalization.** This runs before the support check and uses the spec's own functions. `INPUT_TAGGERS = AXES`, imported and never redefined.
- `V, D = weight.shape[-2], weight.shape[-1]`. `B, S` come from `input_tensor`: `[B,S]` → `(B,S)`; `[B,1,1,S]` → `(B,S)`; `[S]` → `(1,S)` (later).
- The axes `dtype`, `layout` come from `weight`. `memory_config`, `sharding` come from `eval.memory.memory_axes(input_tensor)` (R14). `output_layout = canonical_layout(layout, …)`; `output_memory = canonical_memory(memory_config, …)`.
- The derived axes come from `AXES` applied to `(input_tensor.shape, weight.shape, padding_idx, embeddings_type)`.

No index axis (`dim`) exists in this op.

## Tensors

### Input

| Property | `input_tensor` (indices) | `weight` (table) |
|----------|--------------------------|------------------|
| Shape | `[B, S]` / `[B, 1, 1, S]` | `[1, 1, V, D]`, D % 32 == 0 in P0 |
| Dtype | uint32 | bfloat16 (P0); float32 later (knob-turn: `esize`) |
| Layout | ROW_MAJOR: page = one batch row of S uint32, aligned page size `align(4S, dram_alignment)` | ROW_MAJOR (page = one table row, `D·esize` B) or TILE (page = one 32×32 tile with four 16×16 faces; tile `(tr, tc)` is page `tr·Dt + tc`) |
| Memory | DRAM interleaved (P0); L1 interleaved later (R3, knob-turn) | DRAM interleaved (every cell) |

### Output

| Property | Value |
|----------|-------|
| Shape | `[B, S, D]` for indices of rank 2 or 4 (R10); `[S, D]` for rank 1 |
| Dtype | the table's |
| Layout | `canonical_layout(layout)`: TILE (each `[S, D]` image tile-padded to `ceil(S/32)` tile-rows × `Dt` tile-cols; padding holds anything, R8) or ROW_MAJOR (page = one output row; flattened token `f = b·S + s` is page `f`) |
| Memory | `memory_config` if given, otherwise `input_tensor.memory_config()`; interleaved (R12). Allocated with `ttnn.allocate_tensor_on_device(TensorSpec(shape, dtype, layout, memory_config), device)`. |

## Blocking Model

### Axes

| Axis | Character (+ reason) | Extent knob | Phase 0 value | Knob source | Core-assignment | Later unlock |
|------|---------------------|-------------|---------------|-------------|-----------------|--------------|
| **token** (flattened `B·S`; blocked per batch) | independent: each output row depends only on its own index | `block_tokens`: tokens per block, never crossing a batch | TILE out: **32**, the tile height (a cap: one tile-row of the output). RM out: `clamp(ceil(B·S / num_cores), 1, BLOCK_TOKENS_MAX)`, which is the whole per-core assignment when that fits. | host `plan_blocks()`; cap `BLOCK_TOKENS_MAX = 32` (host constant) | **Primary split.** Token groups (`n_token_groups = B · ceil(S / block_tokens)`) are dealt in contiguous ranges over the grid. | knob-turn |
| **column** (D, in 32-element units: `tensor_col_units = Dt = ceil(D/32)`) | independent: output column `i` depends only on table column `i` | `block_cols`: 32-element column units per block | `ceil(Dt / num_col_chunks)`, i.e. the whole width unless the grid-fill or ring-cap clamp cuts it | host `plan_blocks()` | **Secondary split**, engaged only when `n_token_groups < num_cores`: `num_col_chunks = clamp(floor(num_cores / n_token_groups), 1, Dt)`, raised to `ceil(Dt / cols_cap)` by the ring cap | knob-turn |
| **lookup** (V, the table's rows) | reuse-shared, data-dependent: the index picks the row, any token may pick any row, and repeated indices re-select the same bytes. No result spans V, so it is not dependent. | none: never blocked. A block reads only the rows its indices select. | whole extent, not split. The reason: a split along V means every core scans every index (regime *lookup-axis ownership*, rejected). | — | not assigned across cores | scheme-change (regime *resident rows*, deferred) |
| **batch** (B) | independent | folded into **token** (groups never cross a batch, so a TILE block is exactly one output tile-row and an index slice is one page read) | folded | — | via token groups | knob-turn |
| **work unit** (derived: `(token group, column chunk)`) | independent | `num_units_this_core` | `split_work_to_cores(grid, n_token_groups · num_col_chunks, row_wise=True)` | host | contiguous unit ranges, row-wise core order (catalog `noc_placement`) | knob-turn (grid size is a runtime value) |

**Re-running the table on the intermediate data.** The one intermediate is the gathered block in the ring. It has the token × column axes of the block that produced it. It is consumed by the writer on the same core, with no stage, no coordinator core and no collect step, so nothing waits on one core.

**The scheme Phase 0 builds.** *Gather-direct*, multi-core from day 1. Token groups go over the whole grid, column chunks fill the cores the tokens leave idle, and each core walks its units one block at a time through a reader → ring → writer pipeline. Every knob (`block_tokens`, `block_cols`, `num_col_chunks`, `RING_DEPTH`, `BLOCK_TOKENS_MAX`, `THREAD_PLAN`, the grid) is a parameter. `plan_blocks()` is their single source, and every page count, entry size, loop bound and runtime argument is derived from its return value.

**`plan_blocks(B, S, Dt, esize, out_tiled, grid, l1_budget, dfb_max_bytes=DFB_MAX_BYTES)`.** This is the only place blocking is decided. The `dfb_max_bytes` argument exists so a test can force the clamp path.

| Step | Formula |
|------|---------|
| 1 occupancy | `num_cores = grid.x · grid.y`; `block_tokens = 32 if out_tiled else clamp(ceil(B·S / num_cores), 1, BLOCK_TOKENS_MAX)` |
| 2 groups | `groups_per_batch = ceil(S / block_tokens)`; `n_token_groups = B · groups_per_batch` |
| 3 fill | `num_col_chunks = clamp(floor(num_cores / n_token_groups), 1, Dt)` |
| 4 cap | `cols_cap = floor(min(dfb_max_bytes, l1_budget - scratch_bytes) / (RING_DEPTH · block_tokens · 32 · esize))`; `num_col_chunks = max(num_col_chunks, ceil(Dt / cols_cap))` |
| 5 balance | `block_cols = ceil(Dt / num_col_chunks)`; `num_col_chunks = ceil(Dt / block_cols)` (only the last chunk is ragged) |
| 6 deal | `num_units = n_token_groups · num_col_chunks`; `split_work_to_cores(grid, num_units, row_wise=True)` gives each core `(start_unit, num_units_this_core)` |
| derived | `entry_bytes = block_tokens · block_cols · 32 · esize`; `scratch_bytes = READER_THREADS · INDEX_SLOTS · BLOCK_TOKENS_MAX · 4` |

**Unit → block geometry**, computed on device from runtime arguments:
- `g = u / num_col_chunks` and `cc = u % num_col_chunks`;
- `b = g / groups_per_batch` and `tg = g % groups_per_batch`;
- `s0 = tg · block_tokens` and `n_tok = min(block_tokens, S - s0)`;
- `c0 = cc · block_cols` and `n_cols = min(block_cols, Dt - c0)`.

### Buffer-depth knobs

| CB | Depth knob | Phase 0 value | What the depth buys |
|----|------------|---------------|---------------------|
| `cb_gathered_block` | `RING_DEPTH` (entries; one entry = one block) | 2 | The reader assembles block *k+1* while the writer drains block *k*. |
| `scratch_token_indices` | `INDEX_SLOTS` (index slices per reader thread) | 2 | The next block's index slice lands during the current block's gather (stall-shadow reorder below). |

### Mechanism caps

| Mechanism | Cap on which extent | Clamp | What happens unclamped |
|-----------|--------------------|-------|------------------------|
| Tile height: a TILE output block is one output tile-row | `block_tokens` (TILE out) | `= 32` exactly | The rows of two tile-rows mix in one entry and the tile writes land on the wrong pages. |
| Tile geometry 32×32 with 16×16 faces: face-row piece = `16·esize` B | piece size on every non-RM→RM path | validate() refuses a non-32×32 tile (`weight.tile`) | Pieces land at the wrong offsets: wrong values, no error. |
| DFB ring size ≤ 1,048,560 B (target page cluster model; host constant `DFB_MAX_BYTES`, Python has no query) | `block_cols · block_tokens` | `cols_cap` in step 4 | Program build fails. Example: D = 8192 TILE with 32 tokens needs 2 × 512 KiB. |
| L1 budget `ttnn.get_max_worker_l1_unreserved_size() - L1_RESERVE_BYTES` | ring + scratchpad | the `min(...)` in step 4 | Allocation failure. |
| Index slice slot = `BLOCK_TOKENS_MAX · 4` B | `block_tokens ≤ BLOCK_TOKENS_MAX` | step 1 clamp | The next slot is overwritten and the wrong rows are gathered. |
| Table rows: an index ≥ V addresses outside the table (R2) | the row address | device clamp `v = idx < V ? idx : V-1` before any address is formed | A DRAM read outside the tensor. |
| `uint16_t` entry counts in DFB calls (`dataflow_buffer.h:243-246`) | entries per reserve | always 1 entry per block | n/a |
| DM self-loop DFB forbidden (`dfb.cpp:651-657`) | the reader's private storage | `ScratchpadSpec` instead of a ring | Host validation error. |

### Transfer shapes

These are part of the scheme. Source and destination addressing are fixed by `(table layout, out_tiled)`, and both kernels select the code path with CT flags.

| Table → out | Source of token row `v`, column unit `c`, half `h ∈ {0,1}` | Destination in the entry | Reads per token (`pieces`) | Writes per block |
|---|---|---|---|---|
| RM → RM | one piece: page `v`, offset `c0·32·esize`, `n_cols·32·esize` B | slab `k` at `k · block_cols·32·esize` | **1** | `n_tok` slab writes: page `b·S + s0 + k`, dst offset `c0·32·esize`, `n_cols·32·esize` B |
| RM → TILE | page `v`, offset `(c·32 + h·16)·esize`, `16·esize` B | tile `jj` at `jj·tile_bytes` + `face_off(r, h)` | `2·n_cols` | `n_cols` whole-tile writes: page `(b·ceil(S/32) + tg)·Dt + c0 + jj` |
| TILE → RM | page `(v/32)·Dt + c`, offset `face_off(v%32, h)`, `16·esize` B | slab `k` + `(jj·32 + h·16)·esize` | `2·n_cols` | as RM → RM |
| TILE → TILE | page `(v/32)·Dt + c`, offset `face_off(v%32, h)` | tile `jj` + `face_off(r, h)` | `2·n_cols` | as RM → TILE |

The face offset is `face_off(r, h) = (r/16)·2·face_bytes + h·face_bytes + (r%16)·16·esize`, with `face_bytes = 16·16·esize` and `r` the row inside the tile. The NoC on this part has 1-byte alignment for reads and writes, with no source/destination congruence rule: `tt_metal/hw/inc/internal/tt-2xx/quasar/noc/noc_parameters.h:482-491`. The watcher's congruence mask is therefore 0 (`tt_metal/hw/inc/internal/debug/sanitize.h:746-755`). So every piece is one NoC read from DRAM to its final L1 position: no staging copy and no local reshuffle. The bytes are copied unchanged, so R1 holds for every dtype by construction.

### Regimes

| Regime | Status | Predicate | Block | Data movement vs. minimum (DRAM boundary) | What a bigger block buys |
|--------|--------|-----------|-------|-------------------------------------------|--------------------------|
| **gather-direct** | **built** | Every SUPPORTED cell. The domain is all four (table layout × output layout) pairs, at any B, S, V, D the caps admit. | `block_tokens × block_cols` in output layout, one ring entry | **Minimum.** Minimum = each output byte written once, plus each selected row-slab read once, plus each index read once. Actual: output written once. The selected row bytes are read once per token occurrence, so a repeated index re-reads its row: regime *resident rows*. Each index is read `num_col_chunks` times, `≤ (num_col_chunks-1)·4` B extra per token and only when the column split engages. Transactions per token = `pieces` (Transfer shapes); a TILE table cannot go below `2·n_cols`, because a table row there is `2·Dt` non-contiguous 32 B runs in DRAM. | Fixed cost per block: 1 index-slice read; 1 `reserve_back`/`push_back` and 1 read barrier on the reader; 1 `wait_front`/`pop_front` and 1 write barrier on the writer; pipeline fill/drain once per core. Against `n_tok · pieces` reads and `n_tok` or `n_cols` writes per block, this is ≤ 1/32 of the transactions at `block_tokens = 32`. No other regime applies to this domain: the alternatives below either re-read bytes or add a dispatch. |
| **resident rows** (hot rows and small tables in L1: the `padding_idx` row, a BINARY 2-row table, any table whose `V·D·esize` fits a scratchpad when `V < tokens per core`) | **deferred.** Positive reason: R6 and R7 make `padding_idx` and `embeddings_type` result-neutral, and GUIDANCE says nothing asks for the cache. Gather-direct already serves every shape it would serve, and the DRAM it saves is the duplicate-row reads only. For the PHASE0 tables that is ≤ `(tokens per core − V)` reads of ≤ 512 B. Reachable: the reader's per-token piece loop has one source expression. A second source, a scratchpad row read through a loopback, slots in behind a host predicate without touching the ring, the writer or the split. | `V·D·esize ≤ scratch budget ∧ V < tokens_per_core`, or a `padding_idx` hot row | same block | table: `V` rows once per core instead of one row per token | — |
| **staged row + local scatter** (RM table → TILE out: 1 DRAM read per token into a scratchpad row, then `2·n_cols` loopback copies) | not a regime; it is the perf lamp *RM-row staging* below (same scheme, different transfer shape) | — | — | — | — |
| **compute tilize / untilize** (gathered RM rows tilized by a NEO; a TILE table untilized by whole tile-rows) | **rejected**, superseded by gather-direct. The untilize half reads a 32-row tile-row per token (32× the minimum: a dead end). The tilize half adds a compute kernel, a second ring and a NEO stage to a byte copy, and float32 through the unpacker risks R1. Gather-direct reaches the byte minimum on all four layout pairs with one mechanism. | — | — | table: ×32 for TILE tables | — |
| **convert the table to row-major first** | **rejected**, superseded by gather-direct. It is two dispatches (forbidden: one native dispatch per call) and moves `2·V·D·esize` B to look up `B·S` rows (V = 8192 for 32 tokens in the Llama call). | — | — | table: `V·D·esize` ×2 extra | — |
| **lookup-axis ownership** (each core owns a V range and scans every index) | **rejected**, superseded by the token split. Every core reads all `B·S` indices (×32), the load follows the index distribution, and a resident-slab variant reads `V·D·esize` ≫ the selected rows. | — | — | indices ×num_cores; table ≥ selected bytes | — |
| **index broadcast under the column split** (one core reads a group's index slice and multicasts it to its `num_col_chunks` cores) | **rejected**, superseded by independent re-reads. The payload is ≤ 128 B per group. A multicast plus a semaphore rendezvous costs more than `num_col_chunks` parallel 128 B DRAM reads, which land during the gather anyway (prefetch). | — | — | saves `(num_col_chunks-1)·n_tok·4` B | — |

### Traffic ranking

| Rank | Split | DRAM crossings | Cross-core | Transactions | Verdict |
|------|-------|----------------|------------|--------------|---------|
| 1 | **token** (independent) | table: selected bytes once; indices once; output once | none | minimum (`pieces` per token, one write per page) | **primary split**, cheapest |
| 2 | **column** (independent) on top of token | as rank 1, plus indices × `num_col_chunks` (≤ 31 × 128 B per group) | none | RM→RM row reads and writes split into `num_col_chunks` slabs (more, smaller transfers); face-row shapes unchanged | engaged only when `n_token_groups < num_cores`, to fill the grid |
| 3 | lookup-axis ownership (the data-dependent axis) | indices × num_cores; or table ≫ selected | none | — | rejected |
| 4 | token split with a whole-table layout conversion | + `2·V·D·esize` | — | + a dispatch | rejected |

There is no dependent axis to split with a combine: no result spans any axis. Occupancy is handled inside rank 1 (step 1 of `plan_blocks`) and rank 2 (step 3). It never re-orders the ranking.

### Block schedule

Per core, per kernel thread (thread `t` of `N` takes this core's blocks `t, t+N, …`; P0 has N = 1):

```cpp
// reader
load_index_slice(first_block);                                  // index slot 0, trid 0, barrier
for (uint32_t block_idx = t; block_idx < num_units_this_core; block_idx += READER_THREADS) {
    prefetch_index_slice(block_idx + READER_THREADS);           // other slot, trid 0, no barrier (before the reserve: verifier, Phase 0)
    reserve_gathered_block();                                   // cb_gathered_block.reserve_back(1)
    gather_block(block_idx);                                    // n_tok × pieces reads, trid 0
    land_and_publish_block();                                   // one read barrier (trid 0), push_back(1)
}
// writer
for (uint32_t block_idx = t; block_idx < num_units_this_core; block_idx += WRITER_THREADS) {
    store_block(block_idx);                                     // wait_front(1); n_cols tile or n_tok slab writes (trid 0); write barrier; pop_front(1)
}
```

| Operation | Block shape | Resident across it | Intended frequency of fixed costs |
|-----------|-------------|--------------------|-----------------------------------|
| `load_index_slice` | `n_tok` uint32 (one page, offset `s0·4`) | — | once per core and thread |
| `prefetch_index_slice` | `n_tok` uint32 of the next block | the current slot, read by `gather_block` | once per block, no barrier of its own: the block's barrier covers it |
| `gather_block` | `n_tok × n_cols` column units: `n_tok · pieces` reads into one entry | the index slot; the reserved entry | **no barrier inside**. All reads of the block are in flight together (≥ 32 per barrier at `block_tokens = 32`; catalog `double_buffer` / `trid_double_issue`: ≥ 4–8 reads per barrier is past the knee). Per token: one index load from L1 and one clamp. |
| `land_and_publish_block` | one entry | — | 1 read barrier + 1 push per block |
| `store_block` | one entry: `n_cols` tiles or `n_tok` slabs | — | 1 wait + all writes in flight + 1 write barrier + 1 pop per block |

**Stall-shadow check.**
- **Reader waiting on the index slice.** Every gather address depends on it. The next block's slice depends only on the block number, so it is read **into** the current block's gather window (`prefetch_index_slice`). That is a loop-nest reorder, built in P0; the only block whose index read is exposed is the first.
- **Reader barrier before push.** The writer drains the previous entry in that window (`RING_DEPTH = 2`).
- **Writer barrier before pop.** Nothing on the writer is independent of the slot it holds except the next block's writes. Deferring the pop by one block is the perf lamp *writer pop deferral*.
- No floating point is involved, so no reorder needs the precision baseline.

### Perf lamps

| Lamp | Why the default may be wrong here | Nearby alternative to measure |
|------|-----------------------------------|-------------------------------|
| **Thread plan** `(1, 0, 1)` | The face-row shapes issue `2·n_cols` 32 B reads per token, so the reader hart's issue rate, not DRAM bandwidth, may bound them. Catalog `split_reader` (WH, L1 source): 32 B transactions are issue-bound, and two readers gave 1.74×. | `(2, 0, 2)` and `(4, 0, 2)` on `ttsim-perf`, model cycles. Each kernel deals its blocks `t, t+N`, so this edits one constant. |
| **RM-row staging** (RM table → TILE out) | `2·n_cols` small DRAM reads per token, where one `n_cols·64` B read could fetch the row | One DRAM read per token into a scratchpad row, then `2·n_cols` loopback reads into the tile positions (`tt_memmove`, `ttnn/cpp/ttnn/operations/data_movement/common/kernels/common.hpp:218`) |
| **Grid synchronization: column split at small widths** | Qwen `(1,1) × D=128`, RM out → 4 cores × one 64 B slab; a launch on 4 cores to move 256 B | `num_col_chunks = 1` below a width floor (one core, one 256 B read) |
| **Grid synchronization: RM out one token per core** | `(1,32)` RM out puts one token on each of 32 cores; the per-core fixed cost (index read, handshakes, launch) is paid 32 times for 32 rows | `block_tokens` floor (e.g. 4 tokens per core on 8 cores) |
| **Overlap: one block per core** | Every PHASE0 entry has `num_units_this_core = 1`. The writer starts only when the whole block has landed, so the reader and writer do not overlap within a core. | Halve `block_cols` (2 blocks per core, depth 2) when `num_units_this_core == 1` |
| **Writer pop deferral** | The write barrier per block idles the writer while its writes drain | Issue block `k+1`'s writes before barrier + pop of block `k` (`RING_DEPTH = 3`) |
| **Ring depth** | Depth 2 may not cover DRAM latency jitter of scattered reads | `RING_DEPTH = 3` |

## Dataflow Strategy

| Stage | Format | Mechanism | Notes |
|-------|--------|-----------|-------|
| Index slice → `scratch_token_indices` | uint32, `n_tok·4` B | reader: `noc.async_read<TXN_ID>(indices_accessor, scratchpad, n_tok·4, {.page_id = b, .offset_bytes = s0·4}, {.offset_bytes = slot·BLOCK_TOKENS_MAX·4}, {.trid = 0})` | Read the values only after a trid-0 read barrier. `noc_async_read_barrier_with_trid` calls `invalidate_l1_cache()` (`tt_metal/hw/inc/api/dataflow/dataflow_api.h:2602-2611`), so the cached view is coherent. |
| Table pieces → `cb_gathered_block` entry | the table's bytes, placed in output layout | reader: `noc.async_read<TXN_ID>(table_accessor, cb_gathered_block, size, {.page_id, .offset_bytes}, {.offset_bytes = dst_off}, {.trid = 0})`, per Transfer shapes | Explicit sync: the entry is assembled from many reads (target page *Ring protocols*). The ring is listed in the reader's `disable_dfb_implicit_sync_for`. |
| `cb_gathered_block` entry → output | output pages | writer: `noc.async_write<TXN_ID>(cb_gathered_block, out_accessor, size, {.offset_bytes = src_off}, {.page_id, .offset_bytes}, {.trid = 0})` | Explicit sync: an entry is drained by several writes. The ring is listed in the writer's `disable_dfb_implicit_sync_for`. With RM out and a column split, cores write byte-disjoint slabs of one row page; the NoC's byte granularity allows it. |
| Kernel exit | — | both kernels end with `noc_async_read_set_trid(0, noc.get_noc_id())` (reader) / `noc_async_write_set_trid(0, …)` (writer) (`dataflow_api.h:2558,2579`) | Every transfer already carries trid 0. The reset is the target page's closing rule, kept as insurance. |

No inter-core communication in any built or deferred regime.

**TARGET placement / support axes as regimes.** Each is classified against this op's axis characters.

| TARGET value beyond PHASE0 | Cost class | Why |
|----------------------------|-----------|-----|
| `memory_config: L1` (indices) | knob-turn | The indices are read through their TensorAccessor wherever they are. Output default follows them (R12, `output_memory: None` → L1; validate() then allocates in L1, interleaved). |
| `sharding: interleaved` only | — | TARGET has no sharded value. A sharded index tensor is refused by name (R14). |
| `dtype: float32` | knob-turn | `esize = 4`: face-row pieces become 64 B, the slabs scale. Byte copies keep R1. |
| `dtype: bfloat8_b` | — | unbuildable on quasar (target page family facts) |
| `alignment: w_non_aligned` | knob-turn | Clip every piece and slab to `D` columns (last column unit holds `D % 32` valid elements; a face-row half may be partial or empty). TILE output padding columns hold anything (R8). Reads never pass column `D` of an RM row. |
| `rank: 1`, `table_rank: 2` | knob-turn | canonicalization only (`B = 1`; `V, D` from the last two dims) |

## Work Distribution

| Field | Value |
|-------|-------|
| Work unit | one block = `(token group, column chunk)`, `block_tokens × block_cols` |
| Grid | `device.compute_with_storage_grid_size()` (8×4 here); all of it is offered to `split_work_to_cores(..., row_wise=True)` |
| Per-core work | `num_units_this_core` contiguous units from `start_unit`, with runtime args `start_unit`, `num_units`. Shape geometry `(B, S, V, Dt, block_tokens, block_cols, num_col_chunks, groups_per_batch)` is runtime (common) args: one compile serves every shape. Layout flags, `esize` and the thread counts are CT args. |
| Remainder | ragged last token group per batch (`n_tok = min(block_tokens, S - s0)`, `S` not a multiple of `block_tokens`), ragged last column chunk (`n_cols`), uneven unit counts per core from `split_work_to_cores`. Every count is `ceil`-based and per image: `ceil(S/32)` tile-rows per batch, never `floor(B·S/32)`. A ragged block still reserves, pushes, waits and pops exactly 1 entry; only the reads and writes narrow. |
| Bound | **data movement.** The op computes nothing; per token it moves `D·esize` B in and out with `pieces` reads, so the issuing DM hart and DRAM are the only resources in play. |
| Thread plan | **(R, C, W) = (1, 0, 1)**, held in the host constant `THREAD_PLAN`. There is no compute kernel: C = 0, as an op with no compute per unit calls for (target page *Choosing the plan*: data-movement bound → one thread of each kind). `READER_THREADS` and `WRITER_THREADS` are CT args; both kernels stride their block loops by them from `get_my_thread_id()` (`tt_metal/hw/inc/api/kernel_thread_globals.h:155`). The scratchpad is sized per reader thread. Raising the plan is the perf lamp *thread plan*. |
| Unit dealt to compute threads | n/a: no compute threads. The unit dealt to DM threads is the ring entry (one block), so no round padding exists. Thread `t` of each kernel takes this core's blocks `t, t+N, …`, which on a STRIDED 1P/1C-per-thread ring is the entry dealing the ring itself does. |
| Tests the plan needs | (1) A host-side probe that reads the plan off the built `ProgramSpec`: reader `num_threads == 1`, writer `num_threads == 1`, no compute kernel. It also reads the per-core `num_units` runtime args and checks they sum to `num_units` with no core empty, which shows every unit was dealt. (2) One device session: `embedding_gen` after another program (e.g. a `ttnn.to_layout`, or a second `embedding_gen` with another shape), then `ttnn.to_layout(output, ttnn.ROW_MAJOR_LAYOUT)` after it, and the result checked. |

**Assignment sub-modes and the tests that pin them.** One regime, three host-selected core-assignment shapes. The selection is `plan_blocks` steps 3–4.

| Sub-mode | Predicate | Pinned by |
|----------|-----------|-----------|
| token split only | `n_token_groups ≥ num_cores` and `Dt ≤ cols_cap` | op-carried cell `(2, 1, 1, 1024) × (1,1,64,128)`, both table and output layouts: 64 groups → 2 blocks per core, which exercises the ring cycling and the index prefetch across blocks. No PHASE0 INPUTS entry reaches it. |
| token + column split (grid fill) | `n_token_groups < num_cores`, `Dt > 1` | PHASE0 INPUTS, e.g. `(1,32) × (1,1,8192,64)` TILE out (2 cores), `(1,8) × (1,1,32,8192)` (32 cores, `block_cols = 8`) |
| ring-cap column split | `ceil(Dt / cols_cap) > fill value` | op-carried cell with `plan_blocks(..., dfb_max_bytes=<small>)` forcing `cols_cap = 1` on `(1,1,1,32) × (1,1,64,128)`; plus a host-only unit test of `plan_blocks` at `(1, 1024, 8192)` TILE (cap → `num_col_chunks = 2`, `block_cols = 128`) |
| R2 guard | an index ≥ V | op-carried test: indices with `V + 100` mixed in; no fault, and every valid position exact |

**Worked Phase 0 plans** (8×4, bf16; `plan_blocks` output):

| Entry (indices × table) | out | block_tokens | n_token_groups | num_col_chunks | block_cols | cores | entry B |
|---|---|---|---|---|---|---|---|
| (1,32) × (1,1,8192,64) | TILE | 32 | 1 | 2 | 1 | 2 | 2,048 |
| (1,32) × (1,1,8192,64) | RM | 1 | 32 | 1 | 2 | 32 | 128 |
| (1,1,1,32) × (1,1,64,128) (also pad 5, PADDED 63, BINARY V=2) | TILE | 32 | 1 | 4 | 1 | 4 | 2,048 |
| same | RM | 1 | 32 | 1 | 4 | 32 | 256 |
| (1,1) × (1,1,4096,128) | TILE | 32 | 1 | 4 | 1 | 4 | 2,048 |
| (1,1) × (1,1,4096,128) | RM | 1 | 1 | 4 | 1 | 4 | 64 |
| (2,16) × (1,1,1000,64) | TILE | 32 | 2 | 2 | 1 | 4 | 2,048 |
| (4,1,1,128) × (1,1,2048,512) | TILE | 32 | 16 | 2 | 8 | 32 | 16,384 |
| (4,1,1,128) × (1,1,2048,512) | RM | 16 | 32 | 1 | 16 | 32 | 16,384 |
| (1,512) × (1,1,1024,128) | TILE / RM | 32 / 16 | 16 / 32 | 2 / 1 | 2 / 4 | 32 | 4,096 |
| (2,32) × (1,1,64,4096) | TILE / RM | 32 / 2 | 2 / 32 | 16 / 1 | 8 / 128 | 32 | 16,384 |
| (1,8) × (1,1,32,8192) | TILE / RM | 32 / 1 | 1 / 8 | 32 / 4 | 8 / 64 | 32 | 16,384 / 4,096 |
| (2,1,1,64) × (1,1,2,256) BINARY | TILE / RM | 32 / 4 | 4 / 32 | 8 / 1 | 1 / 8 | 32 | 2,048 |

## Circular Buffers

| Semantic Name | Index | Page Size | Num Pages | Sizing rationale | Format | Producer | Consumer | Sync | Lifetime |
|---------------|-------|-----------|-----------|------------------|--------|----------|----------|------|----------|
| `cb_gathered_block` (DFB `unique_id="gathered_block"`, STRIDED) | DFB 0 | `entry_bytes = block_tokens · block_cols · 32 · esize` | `RING_DEPTH` (· `READER_THREADS` once R > 1, so each producer thread has its own depth) | spans token (`block_tokens`) and column (`block_cols`); streams over units; independent of V | none (`data_format_metadata` unset: DM-only ring, `dfb.cpp:272-282`) | reader | writer | explicit on both sides: `reserve_back(1)`/`push_back(1)`, `wait_front(1)`/`pop_front(1)`. Listed in both kernels' `disable_dfb_implicit_sync_for`; every transfer of both kernels tagged trid 0. | whole program |
| `scratch_token_indices` (`ScratchpadSpec`, bound to the reader only) | scratchpad | 4 B per index | `READER_THREADS · INDEX_SLOTS · BLOCK_TOKENS_MAX` indices (256 B in P0) | spans token (one slot per block's indices, ≤ `BLOCK_TOKENS_MAX`); constant in column and V | none (DM-only) | reader (NoC writes into it) | reader (loads) | — (one kernel; ordering by trid-0 read barrier) | whole program |

## Block Operation Realization

| # | Block operation | Block shape | Helper? | Input CB (semantic name, pages, state) | Output CB (semantic name, pages) | CB state after |
|---|-----------------|-------------|---------|----------------------------------------|----------------------------------|----------------|
| 1 | `load_index_slice` | `n_tok` indices | no (DM raw API) | indices tensor (DRAM) | `scratch_token_indices` slot 0 | slot 0 valid after barrier |
| 2 | `reserve_gathered_block` | 1 entry | no | — | `cb_gathered_block`, 1 entry reserved | reader owns the write entry |
| 3 | `prefetch_index_slice` | `n_tok'` indices of the next own block | no | indices tensor | `scratch_token_indices`, other slot | in flight until op 5's barrier |
| 4 | `gather_block` | `n_tok × n_cols` column units: `n_tok·pieces` reads | no | table (DRAM); current index slot (read) | `cb_gathered_block` reserved entry | in flight |
| 5 | `land_and_publish_block` | 1 entry | no | — | `cb_gathered_block`, push 1 | entry visible to the writer; next slot valid |
| 6 | `store_block` | 1 entry → `n_cols` tiles or `n_tok` slabs | no | `cb_gathered_block`, wait 1 | output tensor (DRAM) | entry popped after the write barrier |

## API Mapping

| Block operation | Type | Function | File:Line | Template Params / Args | Input CB | Output CB | Which params are block knobs |
|-----------------|------|----------|-----------|------------------------|----------|-----------|------------------------------|
| host program | host | `ttnn.experimental.ProgramSpec(name, kernels, dataflow_buffers, scratchpads, tensor_parameters, work_units)`; `ttnn.generic_op` | `ttnn/cpp/ttnn-nanobind/program_spec.cpp:888-894`; example `.claude/references/quasar_op_examples/toy_thread_plan/toy_thread_plan_program_artifacts.py` | two `KernelSpec`s (reader, writer) via `E.create_reader_datamovement_config` / `create_writer_datamovement_config`, each with `config_2xx = DataMovement2XXConfig(disable_dfb_implicit_sync_for=["gathered_block"])`; `num_threads` from `THREAD_PLAN`; one `WorkUnitSpec(kernels=["reader","writer"], target_nodes=core_set)` | — | — | `num_threads`, `num_entries`, `entry_size`, core set |
| host buffers | host | `DataflowBufferSpec(unique_id, entry_size, num_entries)`; `ScratchpadSpec(unique_id, size_per_node)` + `ScratchpadBinding(scratchpad_spec_name, accessor_name)` | `program_spec.cpp:365-398,705-718`; `tt_metal/impl/metal2_host_api/program_spec/validation/resource/scratchpad.cpp:18-37` | no `data_format_metadata` on either | — | — | `entry_size = entry_bytes`, `num_entries = RING_DEPTH`, `size_per_node = scratch_bytes` |
| host split | host | `ttnn.split_work_to_cores(grid, num_units, row_wise=True)` | used as in `toy_thread_plan_program_artifacts.py` `split_tiles` | `num_units` from `plan_blocks` | — | — | `num_units` |
| 1, 3 | raw_api | `Noc::async_read<NocOptions::TXN_ID>(TensorAccessor, Scratchpad, size, {.page_id, .offset_bytes}, {.offset_bytes}, {.trid = 0})` | `tt_metal/hw/inc/api/dataflow/noc.h:252`; TensorAccessor args `tt_metal/hw/inc/api/tensor/noc_traits.h:136-143`; scratchpad endpoint `tt_metal/hw/inc/internal/local_mem/noc_traits.h:74-102` | size `n_tok·4` | indices | `scratch_token_indices` | `block_tokens` (slice length) |
| 1, 5 | raw_api | `Noc::async_read_barrier<NocOptions::TXN_ID>({.trid = 0})` | `noc.h:739` | — | — | — | once per block |
| 1, 4 | raw_api | `Scratchpad<uint32_t>(scratch::token_indices).local_mem()` | `tt_metal/hw/inc/api/scratchpad.h:55,129` | load index `k` of the slot after the barrier | `scratch_token_indices` | — | — |
| 2, 5 | raw_api | `DataflowBuffer::reserve_back(1)` / `push_back(1)` | `tt_metal/hw/inc/api/dataflow/dataflow_buffer.h:243-244` | 1 entry = 1 block | — | `cb_gathered_block` | entry = block |
| 4 | raw_api | `Noc::async_read<NocOptions::TXN_ID>(TensorAccessor, DataflowBuffer, size, {.page_id, .offset_bytes}, {.offset_bytes = dst_off}, {.trid = 0})` | `noc.h:252`; `DataflowBufferArgs` `noc.h:22-24`; dst = write ptr + offset `dataflow_buffer.h:604-611` | Transfer shapes table | table | `cb_gathered_block` | `block_tokens`, `block_cols` (loop extents) |
| 6 | raw_api | `DataflowBuffer::wait_front(1)` / `pop_front(1)`; `Noc::async_write<NocOptions::TXN_ID>(DataflowBuffer, TensorAccessor, size, {.offset_bytes = src_off}, {.page_id, .offset_bytes}, {.trid = 0})`; `Noc::async_write_barrier<NocOptions::TXN_ID>({.trid = 0})` | `dataflow_buffer.h:245-246`; `noc.h:411,760` | — | `cb_gathered_block` | output | `block_cols` (tile writes) / `block_tokens` (slab writes) |
| exit | raw_api | `noc_async_read_set_trid(0, noc.get_noc_id())`, `noc_async_write_set_trid(0, …)` | `tt_metal/hw/inc/api/dataflow/dataflow_api.h:2558,2579` | — | — | — | — |
| thread dealing | raw_api | `get_my_thread_id()` | `tt_metal/hw/inc/api/kernel_thread_globals.h:155` | CT `READER_THREADS` / `WRITER_THREADS` | — | — | thread plan |

**Helpers considered and rejected** (for every `raw_api` row above). This op has no compute phase, so the compute helpers have nothing to cover. Three data-movement helpers were considered for these rows:
- **`tilize_helpers.hpp` / `untilize_helpers.hpp`**: they compile on Quasar (`ttnn/cpp/ttnn/kernel_lib/untilize_helpers.inl:32-53`, `tilize_helpers.inl:156,176`). They are compute-kernel helpers acting on a ring of whole 32-row tiles or sticks (`tilize_helpers.inl:66-69`). They cannot read one row out of a tile, so a TILE table would need whole tile-rows (regime *compute tilize / untilize*, rejected). The direct face-row placement makes them unnecessary on all four layout pairs.
- **`mcast/kernel/mcast_args.hpp`**: no operand is shared across cores (regime *index broadcast*, rejected).
- **`tt_memmove`** (`ttnn/cpp/ttnn/operations/data_movement/common/kernels/common.hpp:218`): a local copy is never needed, because the 1-byte NoC alignment (`noc_parameters.h:482-491`) puts every piece in place from DRAM. It is the mechanism of the perf lamp *RM-row staging*.

No `compute_kernel_hw_startup()` applies: there is no compute kernel.

## Structural impossibilities

None found beyond the specification's `INVALID`.

## Key Risks and Gotchas

| Risk | Why it bites here | Mitigation in this design |
|------|-------------------|---------------------------|
| Stale index values | The reader loads indices that a NoC read wrote into L1, through the DM data cache | Read a slot only after the trid-0 read barrier that covered its transfer (it calls `invalidate_l1_cache()`, `dataflow_api.h:2602-2611`). The prefetched slot is valid only after the *next* block's barrier. |
| Index-slot overwrite | The prefetch writes the slot the gather reads if the slots are not alternated | Slot = `(own block ordinal) % INDEX_SLOTS`, per reader thread at offset `t · INDEX_SLOTS · BLOCK_TOKENS_MAX · 4` |
| An index ≥ V reads outside the table (R2) | The address is formed from the raw index | Clamp `v = min(idx, V-1)` before any address. V is a runtime arg. Op-carried test. |
| Tile padding rows in the output | `S % 32 ≠ 0`: an entry's rows `n_tok..31` hold stale bytes | Allowed by R8 (padding holds anything); the reader writes rows `< n_tok` only. |
| Ragged block counts | Reserving or pushing `n_tok` entries would break the ring cycle | One entry per block always. Only reads and writes narrow (`n_tok`, `n_cols`). |
| `data_format_metadata` on a DM-only buffer | uint32 is refused (`dfb.cpp:148`), and a format is meaningless for byte moves | Leave it unset on the ring and the scratchpad (`dfb.cpp:272-282`). |
| Face offset formula | It is the single place layout correctness lives, and any error is silent | Pass `face_bytes` and the face-row size from the host (derived from `esize` and the 32×32 / 16×16 tile). Cover all four layout pairs and ragged S in the acceptance cells (PHASE0 already crosses both table layouts with both output layouts). |
| Sub-16-B transfers | The index slice of a 1-token block is 4 B. The Quasar NoC states 1-byte alignment, but no minimum size is stated. | If the simulator rejects it, read the 16 B window containing the slice. That stays inside the 64-B-aligned page allocation (`buffer.cpp:880`). |
| `memory_config=None` placement | The output must go where the *indices* are (R12), not where the table is | validate() and the allocation use `input_tensor.memory_config()` when `memory_config` is None. |
| Assignment sub-modes only some shapes reach | The multi-block and ring-cap paths are in no PHASE0 INPUTS entry | The op-carried pinned cells in Work Distribution. |
