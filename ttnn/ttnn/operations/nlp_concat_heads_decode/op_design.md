# Operation Design: nlp_concat_heads_decode

## Overview

| Field | Value |
|-------|-------|
| Classification | data_movement (a sub-tile row gather across cores; no compute kernel) |
| Goal | Put the first `num_heads` head rows of each user's decode-step tile row side by side in that user's output row, with the output width-sharded one head per core (spec R2, R4). One `ttnn.generic_op` dispatch of one `ttnn.ProgramSpec`. |
| Math | `output[0, 0, u, h*head_dim + d] = input[0, u, h, d]` for `u < users`, `h < num_heads`, `d < head_dim`; output rows `users..31` are padding (R3). Bytes are moved unchanged (R1). |
| References | Spec: `eval/specs/nlp_concat_heads_decode/spec.py` (governs). Target: `.claude/references/targets/quasar.md`. Backend: `.claude/references/backends/ttsim.md`. Host program pattern: `.claude/references/quasar_op_examples/toy_thread_plan/toy_thread_plan_program_artifacts.py`. A DFB-free data-movement `ProgramSpec`: `ttnn/cpp/ttnn/operations/experimental/quasar/reshard/device/nd_reshard_program_factory_copy_local.cpp:116-195`. NoC API: `tt_metal/hw/inc/api/dataflow/noc.h`. Local shard handle: `tt_metal/hw/inc/api/tensor/local_tensor_accessor.h`. Perf catalog: `ttnn/ttnn/operations/examples/master.md` (`double_buffer`, `split_reader`, `bank_stagger`, `read_coalescing`). |

## Target

| Fact | Value | Source |
|------|-------|--------|
| Target | `quasar` on ttsim simulator | `.claude/references/targets/quasar.md` + `.claude/references/backends/ttsim.md` |
| Arch | `quasar` | target page |
| Compute grid | 8 × 4 = 32 clusters (queried at run time: `device.compute_with_storage_grid_size()`) | target page |
| L1 per core | 4,194,304 B (4 MiB) per cluster | target page |
| Unreserved budget | `ttnn.get_max_worker_l1_unreserved_size() - l1_reserve_bytes`; **no predicate of this op consumes it**: the op allocates no L1 of its own (no DFB, no scratchpad). The value on quasar is not recorded here. | unknown (not needed) |
| Host program | `ttnn.ProgramSpec` (Metal 2.0) through `ttnn.generic_op`: `.claude/references/quasar_op_examples/` | target page |
| Threads per core | 4 NEO compute engines (C ∈ {1,2,4}), 6 user DM cores (R + W ≤ 6; on STRIDED rings R \| C and W \| C). This op uses **no ring and no compute kernel**, so the divisibility rule does not bind it. | target page *Thread plan* |
| Unavailable here | (1) `ttnn.ProgramDescriptor` / Metal 1 kernels and the legacy program-factory device op: refused at kernel creation. (2) `bfloat8_b`, `bfloat4_b`: no path on this family (not in TARGET anyway). (3) **Untagged NoC transfers: not used**: every transfer of this op is tagged `NocOptions::TXN_ID` with `{.trid = 0}`, barriers on the same id, and the kernel hands the tag back as its last statement. (4) uint16/uint32 buffer formats are refused (no buffer here). | target page *Not available* |

All worked numbers below are for **quasar (8×4 = 32 clusters)**; every run-time predicate queries the device.

## Parameters

| Name | Type | Required | Valid Range | Default | CT/RT |
|------|------|----------|-------------|---------|-------|
| `input_tensor` | `ttnn.Tensor` | yes | rank 4, `[1, users, H, head_dim]`, `H ≤ 32`, `head_dim % 32 == 0`, TILE, L1 height-sharded, shard `[32, head_dim]` | — | — |
| `num_heads` | `int`, keyword-only | yes | `1 ≤ num_heads ≤ H` (X1, X2); `num_heads ≤ grid.x·grid.y` else R5 `RuntimeError` | — | host: sets the core set; kernel RT `head_idx` per node |
| `sub_core_grids` | `ttnn.CoreRangeSet` or `None`, keyword-only | no | `None` only (R7, X4) | `None` | host only |

No compute config (the op has no compute kernel). No index axis to canonicalize (there is no `dim` argument).

## Tensors

### Input

| Property | Requirement |
|----------|-------------|
| Shape | `[1, users, H, head_dim]`; Phase 0: `users = 1`, `H = 32`, `head_dim ∈ 32·ℕ` |
| Dtype | Phase 0 `bfloat16`; TARGET adds `float32` |
| Layout | TILE (32×32 tiles, four 16×16 faces) |
| Memory | L1, HEIGHT_SHARDED, one shard per user, shard shape `[32, head_dim]` (R6) |

### Output

| Property | Value |
|----------|-------|
| Shape | `[1, 1, 32, num_heads * head_dim]` (32 rows always, R3) |
| Dtype | the input's |
| Layout | TILE |
| Memory | `ttnn.MemoryConfig(WIDTH_SHARDED, L1, ttnn.ShardSpec(cores, [32, head_dim], ROW_MAJOR))` with `cores = ttnn.num_cores_to_corerangeset(num_heads, device.compute_with_storage_grid_size(), row_wise=True)` taken **verbatim** from that call (R4; the callers compare with `==`). Allocated with `ttnn.allocate_tensor_on_device(ttnn.TensorSpec(shape, dtype, TILE_LAYOUT, memory_config), device)`. |

Tile geometry (all formulas alignment-aware, per user): `head_tiles = ceil(head_dim / 32)` (= `head_dim / 32`, since `head_dim % 32 == 0` is gated); the input of user `u` is tile row `u` of the height-sharded tensor (its `H` rows tile-padded to 32 per user, `ceil(H/32) = 1` by X7), so input page id of (user `u`, tile `t`) is `u * head_tiles + t`. Output core `i` holds output pages `i*head_tiles .. (i+1)*head_tiles - 1`, contiguous in its shard; local tile `t` is at byte `t * tile_bytes` of the shard.

## Blocking Model

### Axes

| Axis | Character (+ one-clause reason) | Extent knob | Phase 0 value | Knob source | Core-assignment | Later unlock |
|------|--------------------------------|-------------|---------------|-------------|-----------------|--------------|
| `head` (h < num_heads) | independent — output head `h` depends only on input row `h`, no combine | `block_heads` | 1 per core | fixed by the output contract R4 (shard width = one head); host derives nothing from it | **spread across the grid**: head `i` → the i-th core of `num_cores_to_corerangeset(num_heads, grid, row_wise=True)`; all `num_heads` cores engaged in Phase 0 (2 cores for the 2-head call, 32 clusters for the 32-head call on quasar) | none needed — already one head per core, the contract forbids another assignment |
| `head_dim_tiles` (t < head_tiles) | independent — each tile's row runs move in isolation | `block_tiles` (tiles one thread gathers in one block) | `head_tiles` (whole head: the core's entire share, one block) | host: `block_tiles = ceil(head_tiles / reader_threads)`, from `head_tiles = head_dim // 32` and the plan constant `READER_THREADS` | not split across cores (the contract places the whole head on one core); split across the core's reader threads as contiguous ranges when `READER_THREADS > 1` | knob-turn (raise `READER_THREADS`: the perf lamp below) |
| `user` (u < users) | independent — user `u`'s row is a separate source shard and a separate output row | `block_users` | `num_users` (= 1 in TARGET; whole extent) | host CT arg `num_users = input.shape[1]` | not split: every output core holds row `u` of every user by the contract; each core walks all users | knob-turn (users > 1 is outside TARGET; the kernel already loops `num_users`) |
| `head_row` (the H rows of a user's tile) | independent selection axis — the core takes exactly row `head_idx`; rows `≥ num_heads` and tile padding rows are never read (R8) | none: one row per core is the op's definition, not a blocking choice | 1 | RT arg `head_idx` per node | row `i` → core `i` (same as `head`) | — |
| `face_half` (k ∈ {0,1}: the two 16-element runs of a tile row) | independent — two disjoint contiguous runs | `runs_per_row = tile_w / face_w` | 2 (a mechanism fact, see caps) | derived on host from `tile.tile_shape` / `tile.face_shape` | in the same thread as its tile | — |

Intermediate data: none. The gathered rows land in their final place (the output shard); no stage produces data that a later stage reads, so there is no second table to run. No core collects or coordinates: there is no designated core and no idle-while-one-computes stage.

### Buffer-depth knobs

| CB | Depth knob | Phase 0 value | What the depth buys |
|----|------------|---------------|---------------------|
| (none) | — | — | The op has no DFB: reads land directly in the output shard. The pipelining knob that remains is **transactions in flight per barrier** = `2 · block_tiles · num_users` reads per thread, all issued before one barrier (catalog `double_buffer`: never read-one/barrier). It is not a buffer and needs no L1. |

### Mechanism caps

| Mechanism | Cap on which extent | Clamp | What happens unclamped |
|-----------|--------------------|-------|------------------------|
| Tile face layout: a row `r` of a 32×32 tile is two runs of `face_w = 16` elements, in faces `(r/16)*2` and `(r/16)*2 + 1`, at byte `(r%16)·run_bytes` of each face | `face_half` = 2, run size `run_bytes = 16·elem_bytes` | `validate()` refuses any tile other than `tile_shape == [32, 32]`, `face_shape == [16, 16]` | wrong elements gathered silently |
| One tile row per user (X7: `H ≤ 32`) | input page id `u·head_tiles + t` | X7 rejection (`H > 32`) | reads land on another user's or another tile row's data |
| `head_dim % 32 == 0` (`alignment` has no `w_non_aligned`) | `head_tiles` | validate (alignment axis gate) | padding columns between heads in the output |
| Output rows = 32 (`PADDED_USERS`) | `num_users` | validate: `users ≤ 32` (TARGET users = 1) | row offset past the tile |
| `num_heads ≤ grid cores` (R5) | `head` core count | `RuntimeError("Target number of cores {n} is greater than total number of available cores {c}")` raised before any allocation | no output placement exists |
| NoC TXN_ID credit: `async_read<TXN_ID>` busy-waits while the trid has fewer than half its transaction slots free (`noc.h:259-264`) | reads in flight per thread | self-clamping; no action | — (throttles, never wrong) |
| iDMA zero borrows write command buffer 0 until `write_zeros_l1_barrier()` (`noc.h:832-836`) | ordering of zero fill vs other writes | the kernel issues no NoC write; it barriers the zero before the first read | corrupted writes (none exist here) |
| `READER_THREADS` vs `head_tiles` | `block_tiles ≥ 1` | host: `reader_threads = min(READER_THREADS, head_tiles)` | a thread with an empty range (wasted hart) |
| NoC L1 read alignment on quasar = 1 B (`tt_metal/hw/inc/internal/tt-2xx/quasar/noc/noc_parameters.h:486`) | run offsets | none needed (offsets are multiples of 32 B anyway) | — |

### Regimes

| Regime | Status | Predicate | Block | Data movement vs. minimum | What a bigger block buys |
|--------|--------|-----------|-------|---------------------------|--------------------------|
| `pull_rows` — each output core pulls its head's row runs from every user's input shard straight into its own output shard | **built** (the only regime) | every call that passes validate | `block_heads = 1` × `block_tiles = head_tiles / reader_threads` × `block_users = num_users`; `num_blocks_this_core = 1` per thread | **Minimum.** Boundary: the NoC between clusters. Each defined input element (`h < num_heads`, `u < users`) crosses the NoC exactly once, into its final output location; no DRAM crossing at all; no element read that is not output (R8). Undefined rows are written by a core-local iDMA zero, not moved. | One kernel launch, one zero-fill + one zero barrier, one trid setup and **one** read barrier per thread per op, instead of per tile or per user. Domain: all of TARGET (bf16/fp32 only change `run_bytes`; h_non_aligned only changes which rows exist). Why no other regime applies: no axis is dependent and no operand is reuse-shared (each core needs disjoint bytes), so no combine or broadcast regime has work to do. |
| `push_rows` — the input core(s) write each head's rows to the output cores | rejected — superseded by `pull_rows` | — | — | Same NoC bytes, but all `2·head_tiles·num_heads` transfers issue from the `users` input cores (one in TARGET): issue serialized on one hart instead of spread over `num_heads` cores. | — |
| `pull_tiles_extract_local` — each output core reads the user's whole tile row, then extracts its row in L1 | rejected (dead end) — superseded by `pull_rows` | — | — | NoC bytes ×32 (whole tiles instead of one row of each), plus an L1 scratch of `head_tiles` tiles per core and a local copy pass. No better scheme passes through it. | — |
| `staged_ring` — read runs into a DFB, a second kernel copies them into the shard | rejected — superseded by `pull_rows` | — | — | An extra L1 hop and a buffer; the read can land in the shard directly (`noc_traits_t<LocalTensorAccessor>` destination). | — |
| `broadcast_input` — multicast the input shard to all head cores | rejected — not applicable | — | — | The input is not reuse-shared under the head split (core `i` needs only row `i`): a broadcast moves `num_heads`× the bytes. | — |

### Traffic ranking

| Candidate split | NoC bytes (Phase 0, bf16, `num_heads = 32`, `head_dim = 128`, quasar) | Transfers | DRAM | Verdict |
|-----------------|------------------------------------------|-----------|------|---------|
| Split `head` across cores, pull rows (`pull_rows`) | `num_heads·users·head_dim·elem` = 32·1·128·2 = 8 KiB | `2·head_tiles·users` per core = 8 per core, 256 in all, 32 B each, issued in parallel on 32 clusters | 0 | **chosen**: minimum bytes, issue spread over every engaged core |
| Same split, push from the input core | 8 KiB | 256, all from one hart | 0 | rejected: same bytes, serialized issue |
| Same split, pull whole tiles and extract | `num_heads·users·head_tiles·tile_bytes` = 32·4·2 KiB = 256 KiB | 128 | 0 | rejected: 32× bytes |
| Split `head_dim_tiles` across cores (several cores per head) | 8 KiB | same | 0 | not legal: the output contract places a whole head on one core; that axis is split across the core's threads instead (lamp) |
| Split a dependent axis + combine | — | — | — | no dependent axis exists |

The cheapest-traffic split is the implemented one. Occupancy is fixed by the contract (`num_heads` cores), not chosen.

**Stall shadow.** The kernel waits twice: on the zero-fill barrier (local iDMA) and on the read barrier. Between them nothing is independent of the wait: the reads must not land before the zero completes (they overwrite row `u` of the zeroed tiles), and after the reads are issued no work remains. The only reorder available is to zero just the padding ranges concurrently with the reads; it is a perf lamp below, not Phase 0, because it turns one zero call into up to six per tile.

### Block schedule

Per output core `i` (`head_idx = i`), per reader thread `r` of `reader_threads` (Phase 0: one thread):

```cpp
// thread r owns tiles [t_begin, t_end) = [r*block_tiles, min((r+1)*block_tiles, head_tiles))
for (uint32_t block_idx = 0; block_idx < num_blocks_this_core; ++block_idx) {   // num_blocks_this_core = 1
    zero_block(block_idx);          // padding rows defined (R3 allows; callers read them)
    gather_rows_block(block_idx);   // the op: row head_idx of every user's tiles -> row u of local tiles
}
// store is implicit: the reads land in the output shard. Last statement: hand the trid back.
```

| Block operation | Block shape | Resident across it | Intended frequency of fixed costs |
|-----------------|-------------|--------------------|-----------------------------------|
| `zero_block` | this thread's contiguous tile range of the local output shard: `(t_end - t_begin) · tile_bytes` bytes, all 32 rows | the output shard (tensor-owned) | **one** `noc.async_write_zeros` call over the whole contiguous range + **one** `write_zeros_l1_barrier()` per thread per op. Skipped when CT `zero_padding_rows == 0`. |
| `gather_rows_block` | `num_users × (t_end - t_begin) × 2 runs`, each run `run_bytes` from input page `u·head_tiles + t` at `src_offset(head_idx, k)` to the local shard at `t·tile_bytes + dst_offset(u, k)` | the output shard; the input shards on the users' cores (read only) | all `2·block_tiles·num_users` tagged reads issued back to back, **one** `async_read_barrier<TXN_ID>({.trid = 0})` per thread per op; trid set once per read by the API (no per-read barrier). |

Offsets (bytes; derived once on host from the tile geometry, see Knob sources):
- `src_offset(h, k) = ((h / face_h) · runs_per_row + k) · face_bytes + (h % face_h) · run_bytes`
- `dst_offset(u, k) = ((u / face_h) · runs_per_row + k) · face_bytes + (u % face_h) · run_bytes`
- `elem_bytes = tile_bytes / (tile_h · tile_w)`, `face_bytes = elem_bytes · face_h · face_w`, `run_bytes = elem_bytes · face_w`, `runs_per_row = tile_w / face_w` (bf16: 2048 / 512 / 32 / 2; fp32: 4096 / 1024 / 64 / 2).

### Perf lamps

| Lamp | Why the default may be wrong here | Nearby alternative to measure |
|------|-----------------------------------|-------------------------------|
| Reader threads (catalog `split_reader`) | One hart issues `2·head_tiles·users` (≤ 16 in INPUTS) 32-64 B reads; if issuing, not the NoC, is the limit, a second hart halves it. Expected not to pay at ≤ 16 reads. | `READER_THREADS = 2` (contiguous tile halves, each thread zeroing its own range); count reads per thread and barriers per thread as the structural measure. |
| Zero fill (overlap) | The whole-shard zero (`head_tiles·tile_bytes`, 8 KiB/core at head_dim 128 bf16) is 32× the gathered bytes and serializes before the reads; R3 does not require it. | (a) `zero_padding_rows = 0` (tt-metal's behaviour: padding left as allocated); (b) zero only the padding ranges of each tile (rows `≥ users`: per tile two partial-face ranges plus the two lower faces) issued concurrently with the reads, one zero barrier at the end. |
| Input-core hot spot (catalog `bank_stagger`) | Every engaged core reads the same input core's L1 at the same instant, all starting at tile 0. | Start each core's tile walk at `head_idx % head_tiles` and wrap. |
| Stateful reads | Every read has the same size `run_bytes`; per-read command-register setup may dominate a 32 B transfer. | `noc.async_read_set_state` + `async_read_with_state` (`noc.h:283-360`) with the size programmed once. |

Grid-synchronization lamp: not attached — the core count is fixed by the output contract (R4), there is no collective, and no alternative split is legal. Overlap lamp beyond the zero fill: not attached — there is no compute to overlap with.

## Dataflow Strategy

| Stage | Format | Mechanism | Notes |
|-------|--------|-----------|-------|
| Input at rest | TILE pages in L1 of the user's core (height shard `[32, head_dim]`) | tensor-owned | addressed by `TensorAccessor(tensor::input)` with `{.page_id = u·head_tiles + t, .offset_bytes = src_offset}` (`noc_traits.h:135-139`); the accessor resolves which core holds the page. |
| Zero fill | output shard bytes | core-local iDMA zero: `noc.async_write_zeros(out_shard, bytes, {.offset_bytes})` then `noc.write_zeros_l1_barrier()` (`noc.h:848-849, 904`) | `out_shard` is `LocalTensorAccessor<uint32_t>(tensor::output)` (`local_tensor_accessor.h:58-77`); no NoC traffic. |
| Gather | raw bytes, `run_bytes` per transfer | `noc.async_read<NocOptions::TXN_ID>(input_acc, out_shard, run_bytes, {.page_id, .offset_bytes}, {.offset_bytes}, {.trid = 0})` (`noc.h:246-282`), destination resolved as local L1 by `noc_traits_t<LocalTensorAccessor<T>>::dst_addr` (`tt_metal/hw/inc/internal/local_mem/noc_traits.h:106-122`) | remote L1 → local L1; for head 0 of user 0 the source may be the same core (a NoC self-read, legal). No unpack/pack: float32 stays bit-exact (R1). |
| Completion | — | `noc.async_read_barrier<NocOptions::TXN_ID>({.trid = 0})` (`noc.h:738-746`), then `noc_async_read_set_trid(0, noc.get_noc_id())` as the last statement (`dataflow_api.h:2558`) | the trid hand-back is required by the target page's ring-protocol rules even without rings. |
| Output at rest | TILE pages, width shard `[32, head_dim]` on core `i` | tensor-owned | rows `0..users-1` defined, rows `users..31` zero when `zero_padding_rows` (R3 permits either). |

**Placement as TARGET axes.** The input placement (`memory_config = "L1"`, `sharding = "height"`) is the only TARGET value: it is the physical realization of the `user` split (independent axis → no combine). The output's width sharding is the physical realization of the `head` split (independent axis → the knob-turn already built in Phase 0). No other placement value exists in TARGET, so no scheme-change is pending. Both shards are consumed in place: the input through a `TensorAccessor` page read from its own core (the data must cross to another core, so a NoC read is inherent, not a re-read), the output through `LocalTensorAccessor` (zero-copy).

**TARGET − PHASE0 (refinement candidates, both knob-turns):**

| Axis value | Class | What changes |
|------------|-------|--------------|
| `dtype: float32` | knob-turn | nothing in the kernel: `tile_bytes` → 4096 and the derived `face_bytes`/`run_bytes` CT args follow; add the value to SUPPORTED; R1 holds because no element passes through DEST or a packer. |
| `alignment: h_non_aligned` (H < 32) | knob-turn | nothing in the kernel: the input page is still a 32-row padded tile; rows `num_heads..31` are never read (R8, X2 bounds `num_heads ≤ H`); add the value to SUPPORTED. |

Structural impossibilities: none to add to INVALID.

## Work Distribution

| Field | Value |
|-------|-------|
| Work unit | one head: `head_tiles` tiles × `num_users` users × 2 runs, i.e. one block per thread |
| Grid | `cores = ttnn.num_cores_to_corerangeset(num_heads, device.compute_with_storage_grid_size(), row_wise=True)`; the program's `WorkUnitSpec.target_nodes = cores`. Worked on quasar: 1 / 2 / 3 / 4 / 8 / 16 / 32 clusters for the INPUTS head counts. |
| Per-core work | core = the i-th entry of `ttnn.corerange_to_cores(cores, None, True)` (row-major, matching the ROW_MAJOR shard orientation of the output); RT arg `head_idx = i`. The gather of head `i`: `2·head_tiles·num_users` reads of `run_bytes`. |
| Remainder | none across cores (one head each, by contract). Across threads: `block_tiles = ceil(head_tiles / reader_threads)`, thread `r` takes `[r·block_tiles, min((r+1)·block_tiles, head_tiles))`; the last range may be shorter (ragged runtime extent, same block operation). |
| Bound | **data movement** — there is no compute at all: every unit is a 32-64 B copy. |
| Thread plan | **(R, C, W) = (1, 0, 0)**: one reader DM kernel (`num_threads = READER_THREADS = 1`), no compute kernel, no writer kernel (the reads land in the output, nothing is written out). Host constant `READER_THREADS = 1`, passed to the kernel as CT arg `reader_threads` (clamped `min(READER_THREADS, head_tiles)`). For an op bound by data movement the target page's plan is one thread of each kind; with no compute and no output stream the kinds reduce to the reader. |
| Unit dealt to compute threads | n/a (no compute kernel). Unit dealt to reader threads: a contiguous tile range (whole tiles; each thread zeroes and gathers only its own range, so threads need no mutual synchronization). |
| Tests the plan needs | (1) **Every thread ran**: a test calling the op with `READER_THREADS ∈ {1, 2}` (host override, e.g. a module-level constant the test patches) on `head_dim = 256` (8 tiles) asserts the full result is right; because each thread zero-fills only its own range, a thread that did not run leaves its tiles zero — a wrong result, not a silent pass. (2) **Session order**: in one device session, run a tt-metal program, then the op (e.g. `ttnn.to_layout` of an unrelated interleaved tensor, then `nlp_concat_heads_decode`), and run the op followed by a tt-metal program on its result (`ttnn.to_layout(ttnn.to_memory_config(out, ttnn.DRAM_MEMORY_CONFIG), ttnn.ROW_MAJOR_LAYOUT)`), checking values after each. |

Only one regime exists, so there is no selection predicate and no regime-pinned test.

## Circular Buffers

| Semantic Name | Index | Page Size | Num Pages | Sizing rationale | Format | Producer | Consumer | Sync | Lifetime |
|---------------|-------|-----------|-----------|------------------|--------|----------|----------|------|-----------|

**None.** The op allocates no DataflowBuffer, scratchpad or semaphore (inventory = 0: the only phase boundary, "read rows" → "place rows", is expressed as reads landing in the destination). The kernel touches two tensor-owned L1 regions through tensor bindings:

| Binding (accessor name) | Tensor | Handle in kernel | Access |
|-------------------------|--------|------------------|--------|
| `input` | input (height-sharded, remote) | `TensorAccessor(tensor::input)` | NoC read source, page + byte offset |
| `output` | output (width-sharded, local shard) | `LocalTensorAccessor<uint32_t>(tensor::output)` | NoC read destination + iDMA zero destination, byte offset |

The DM kernel's hardware config: `E.create_reader_datamovement_config()` (no ring, so `disable_dfb_implicit_sync_for` is irrelevant); every transfer is tagged (Unavailable-here item 3).

## Block Operation Realization

| # | Block operation | Block shape | Helper? | Input CB (semantic name, pages, state) | Output CB (semantic name, pages) | CB state after |
|---|-----------------|-------------|---------|----------------------------------------|----------------------------------|----------------|
| 1 | `zero_block` | this thread's tile range of the output shard, `block_tiles · tile_bytes` bytes | no (raw Noc API; see API Mapping) | — | output shard via `output` binding | tiles `[t_begin, t_end)` all zero |
| 2 | `gather_rows_block` | `num_users × block_tiles × 2` runs of `run_bytes` | no (raw Noc API) | input shards via `input` binding (read only) | output shard via `output` binding | row `u` of each local tile `t` = row `head_idx` of input page `u·head_tiles + t`; other rows zero |

## API Mapping

| Block operation | Type | Function | File:Line | Template Params / Args | Input CB | Output CB | Which params are block knobs |
|-----------------|------|----------|-----------|------------------------|----------|-----------|------------------------------|
| (kernel entry) | raw_api | `TensorAccessor(tensor::input)`, `LocalTensorAccessor<uint32_t>(tensor::output)`, `Noc noc;` | `tt_metal/hw/inc/api/tensor/local_tensor_accessor.h:58-77`; `tt_metal/hw/inc/api/tensor/noc_traits.h:135-139` | accessor names = the host `TensorBinding.accessor_name`s | — | — | — |
| `zero_block` | raw_api | `noc.async_write_zeros(out_shard, size_bytes, {.offset_bytes})`; `noc.write_zeros_l1_barrier()` | `tt_metal/hw/inc/api/dataflow/noc.h:848-849, 904`; quasar impl `tt_metal/hw/inc/internal/tt-2xx/noc_zero_l1.inl:25-80`; `noc_zero_l1_endpoint_v<LocalTensorAccessor<T>> = true` at `tt_metal/hw/inc/internal/local_mem/noc_traits.h:138` | `size_bytes = (t_end - t_begin)·tile_bytes`, `offset_bytes = t_begin·tile_bytes` | — | output shard | `block_tiles` (size), CT `zero_padding_rows` (on/off) |
| `gather_rows_block` | raw_api | `noc.async_read<NocOptions::TXN_ID>(input_acc, out_shard, run_bytes, {.page_id, .offset_bytes}, {.offset_bytes}, {.trid = 0})`; `noc.async_read_barrier<NocOptions::TXN_ID>({.trid = 0})`; last statement `noc_async_read_set_trid(0, noc.get_noc_id())` | `noc.h:246-282` (read), `noc.h:738-746` (barrier), `tt_metal/hw/inc/api/dataflow/dataflow_api.h:2558` (set_trid); destination trait `tt_metal/hw/inc/internal/local_mem/noc_traits.h:106-122` | CT: `tile_bytes`, `face_bytes`, `run_bytes`, `face_h`, `runs_per_row`, `head_tiles`, `num_users`, `reader_threads`, `zero_padding_rows`; RT per node: `head_idx` | input shards | output shard | `block_tiles` (from `head_tiles`, `reader_threads`), `num_users` |

**Helpers considered and rejected (all three raw_api rows):**

| Candidate | File:Line | Why it cannot express this block |
|-----------|-----------|----------------------------------|
| Compute eltwise `copy` / `CopyTile` (`eltwise/` chain) | `ttnn/cpp/ttnn/kernel_lib/eltwise/` (a compute helper on whole tiles in DFBs) | The op moves one *row* of a tile into another row of another tile: a tile-granular unpack→DEST→pack cannot select or relocate a sub-tile row, and it would add a compute kernel, two rings and a DEST round trip (a float32 value through 16-bit DEST breaks R1). |
| `dataflow_kernel_lib::set_read_state` / `read_with_state` | `ttnn/cpp/ttnn/kernel_lib/local_copy_helpers_dataflow.hpp:57, 75, 95` | Self-aimed (local L1 → local L1) only: `local_addr()` builds a source at this core's own coordinates (`:43-57`); every source here is on the input core. |
| `dataflow_kernel_lib::zero_tile` / `prepare_zero_tile` | `ttnn/cpp/ttnn/kernel_lib/l1_helpers.hpp:53-71` | Zeroes one tile at a DataflowBuffer's write pointer; this op has no DFB and zeroes a contiguous multi-tile range of a tensor shard in one call (the underlying `noc.async_write_zeros` it wraps is used directly). |
| `dataflow_kernel_lib::fill_l1_range` | `ttnn/cpp/ttnn/kernel_lib/l1_helpers.hpp:101` | RISC-V stores into the DM data cache (quasar needs a flush before the NoC or a reader sees them, `tile_column_dataflow.hpp:22-23`); the iDMA zero needs neither. |
| `tile_column_dataflow.hpp` (`fill_tile_column`) | `ttnn/cpp/ttnn/kernel_lib/tile_column_dataflow.hpp:70-78` | Columns of a DFB-shaped tile by CPU stores; this op moves rows between cores. Its face arithmetic (`:36-56`) is the same as `src_offset`/`dst_offset` above. |
| `mcast` (`McastArgs`) | `ttnn/cpp/ttnn/kernel_lib/mcast/` | No operand is reuse-shared under the head split (each core needs disjoint bytes). |

No compute phase exists, so `compute_kernel_hw_startup()` and the compute helpers do not apply.

### Host program (implementer's checklist, from the patterns cited)

| Item | Decision |
|------|----------|
| Program | `E.ProgramSpec(name="nlp_concat_heads_decode", kernels=[gather], tensor_parameters=[TensorParameter("input", input.spec), TensorParameter("output", output.spec)], work_units=[E.WorkUnitSpec(name="main", kernels=["gather"], target_nodes=cores)])` — no `dataflow_buffers`, no semaphores (pattern: `nd_reshard_program_factory_copy_local.cpp:186-195`). |
| Kernel | `E.KernelSpec(unique_id="gather", source=<op dir>/kernels/gather_head_rows.cpp, hw_config=E.create_reader_datamovement_config(), num_threads=reader_threads, tensor_bindings=[TensorBinding("input","input"), TensorBinding("output","output")], compile_time_args={...}, runtime_arg_schema=RuntimeArgSchema(runtime_arg_names=["head_idx"]))`. |
| Run args | `E.ProgramRunArgs(kernel_run_args=[E.KernelRunArgs(kernel="gather", runtime_arg_values={"head_idx": {core: i}})])`. |
| Dispatch | exactly one `ttnn.generic_op([input_tensor, output], spec, run_args, {"input": 0, "output": 1})`. |
| CT-arg derivation | `tile_h, tile_w = input.tile.tile_shape`; `face_h, face_w = input.tile.face_shape`; `tile_bytes = input.buffer_page_size()` (assert equal to the output's); the rest by the formulas in Block schedule — each derived once from these. |

### validate() and entry-point order

| Step | Check | Raises |
|------|-------|--------|
| 1 | rank 4 (X5); `shape[0] == 1` (X3); `shape[2] ≤ 32` (X7); `1 ≤ num_heads ≤ shape[2]` (X1, X2); `sub_core_grids is None` (X4, R7) | `ValueError` |
| 2 | axes: `dtype`, `layout`, `memory_axes(input)` → `memory_config`, `sharding` (R6), `AXES` taggers on `(shape, num_heads)` → `users`, `alignment`; each against SUPPORTED (= PHASE0), then EXCLUSIONS (empty) | `NotImplementedError` (R9; X6 lands here) |
| 3 | shard shape `== [32, head_dim]` (R6); tile 32×32 / faces 16×16 (mechanism cap) | `NotImplementedError` |
| 4 | `num_heads > grid.x·grid.y` with `grid = device.compute_with_storage_grid_size()` (R5), before any allocation | `RuntimeError("Target number of cores {num_heads} is greater than total number of available cores {grid.x*grid.y}")` |
| 5 | build output memory config (R4), allocate, build program, one `generic_op` | — |

`INPUT_TAGGERS = AXES` imported from `eval.specs.nlp_concat_heads_decode.spec`; never redefined.

## Key Risks and Gotchas

| Risk | Why it bites here | Mitigation in this design |
|------|-------------------|---------------------------|
| Tag register | the op runs after tt-metal's own quasar ops (implicit sync) in model sessions | every read `NocOptions::TXN_ID` `{.trid = 0}`, barrier on trid 0, `noc_async_read_set_trid(0, ...)` last; session-order tests (Work Distribution) |
| iDMA zero interacting with the ring ISR / tag state | the zero uses write cmd buffer 0 with `CMDBUF_DEF_TRID` (`noc_zero_l1.inl:68`); not exercised in a session with other programs before | the session-order tests cover it; if the zero misbehaves there, set `zero_padding_rows = 0` (R3 permits unwritten padding) and record it |
| Head ↔ core order mismatch | the output's ROW_MAJOR shard order over a multi-range core set (e.g. 16 heads = rows 0-1 of an 8-wide grid) must equal the `head_idx` enumeration | `head_idx` from `ttnn.corerange_to_cores(cores, None, True)` on the very set used in the ShardSpec; INPUTS with 16 and 32 heads (multi-row sets) fail loudly on a mismatch |
| Memory config compared with `==` by callers | a set with the same cores written as other ranges is unequal | use the `num_cores_to_corerangeset(..., row_wise=True)` result unchanged in the ShardSpec |
| R5 message | callers turn exactly `Target number of cores` into a skip | explicit `RuntimeError` with tt-metal's wording before allocation (step 4) |
| Reading rows beyond `num_heads` | R8 | the kernel reads only row `head_idx < num_heads` of each tile; no whole-tile reads |
| Source on the same core | head 0 of user 0 when the input shard is on core (0,0) | a NoC self-read is legal; no special case |
