# Operation Design: paged_update_cache

The specification `eval/specs/paged_update_cache/spec.py` governs wherever this design and it disagree.

## Overview

| Field | Value |
|-------|-------|
| Classification | data_movement (no compute stage: R1, "the op moves values and computes nothing") |
| Goal | Write one decode step's head rows of one user (or several) into a paged, tile-layout KV cache in DRAM, in place, and return the caller's `cache_tensor`. |
| Math | for each user `u` with `p = update_idxs_tensor[u] != -1`, each head `h < heads`, each `d < D`: `cache[page_table[u, p // block_size], h, p % block_size, d] = input[0, u, h, d]`; every other cache element unchanged |
| Mechanism in one line | Every (user, head, D-tile) unit is **two face-row segments** (16 elements each) that the user's shard core writes **directly from its resident input shard into the cache page at a byte offset**: no cache read, no staging, no compute. |
| References | `eval/specs/paged_update_cache/spec.py` (DEFINITION, RULES R1-R9, REJECTIONS X1-X11, GUIDANCE); `.claude/references/targets/quasar.md`; `.claude/references/backends/ttsim.md`; `.claude/references/quasar_op_examples/` (ProgramSpec host pattern, ring protocols, trid hand-back); `tt_metal/hw/inc/api/dataflow/noc.h`; `tt_metal/hw/inc/api/tensor/local_tensor_accessor.h`; `tt_metal/hw/inc/api/scratchpad.h`; `ttnn/ttnn/operations/examples/master.md` (`trid_double_issue`: one barrier per block, not per transfer; `read_coalescing`: why the two segments of a unit cannot be merged, they are 512 B apart) |

## Target

| Fact | Value | Source |
|------|-------|--------|
| Target | `quasar` on the ttsim simulator | `.claude/references/targets/quasar.md` + `.claude/references/backends/ttsim.md` |
| Arch | `quasar` | target page |
| Compute grid | 8 × 4 = 32 clusters (probe: `device.compute_with_storage_grid_size()` = 8-4) | target page; queried at run time |
| L1 per core | 4,194,304 B (4 MiB) | target page |
| Unreserved budget | `ttnn.get_max_worker_l1_unreserved_size() - l1_reserve_bytes` = 3,963,328 − 262,144 = 3,701,184 B on this target (query value probed on ttsim 2026-11-21) | probe |
| Alignment | L1 16 B (`ttnn.get_l1_alignment()`), DRAM buffer/page 64 B (`ttnn.get_dram_alignment()`); NoC transfer alignment on Quasar is 1 B for L1 and DRAM, read and write (`tt_metal/hw/inc/internal/tt-2xx/quasar/noc/noc_parameters.h:486-491`) | target page + header |
| Host program | `ttnn.ProgramSpec` (`ttnn.experimental.ProgramSpec`, `KernelSpec`, `ScratchpadSpec`, `TensorParameter`, `WorkUnitSpec`, one `ttnn.generic_op`) — reference `.claude/references/quasar_op_examples/toy_thread_plan/toy_thread_plan_program_artifacts.py` | target page |
| Threads per core | 4 NEO compute engines (C ∈ {1, 2, 4}); 6 user DM harts (R + W ≤ 6; R, W divide C on STRIDED rings). This op uses **one DM kernel with one thread and no compute kernel** (Work Distribution) | target page *Thread plan* |
| Unavailable here | `ttnn.ProgramDescriptor` / Metal 1 kernels and the legacy C++ program factory (refused at kernel creation); `bfloat8_b`, `bfloat4_b` (no path on Quasar; not in TARGET anyway); a DFB borrowed from the input shard with DM-only endpoints (a self-loop ring on DM kernels is forbidden in Metal 2.0 — `ttnn/cpp/ttnn/operations/experimental/quasar/move/device/kernels/dataflow/reader_unary_local_l1_copy_backwards.cpp:24-26`, so the shard is reached with `LocalTensorAccessor`); untagged NoC transfers: not used (every transfer carries `{.trid = 0}`); uint16/uint32 buffer formats (the scratchpad carries no format metadata) | target page |

Every worked number below is for Quasar (8 × 4 grid, 4 MiB L1). Every run-time predicate uses the queries, never the numbers.

## Parameters

| Name | Type | Required | Valid Range | Default | CT/RT |
|------|------|----------|-------------|---------|-------|
| `cache_tensor` | `ttnn.Tensor` | yes (positional) | rank 4 `[blocks, heads, block_size, D]`, TILE, DRAM interleaved, dtype of the cell; `block_size % 32 == 0` (X7), `D % 32 == 0` (X10) | — | tensor binding `cache` (TensorAccessor, written) |
| `input_tensor` | `ttnn.Tensor` | yes (positional) | rank 4 `[1, users, H, D]`, TILE, L1 HEIGHT_SHARDED, dtype == cache dtype; `heads ≤ H ≤ 32` (X1, X8); `shape[0] == 1` (X3); last dim == cache's (X2) | — | tensor binding `input` (LocalTensorAccessor, read) |
| `update_idxs_tensor` | `ttnn.Tensor` | yes (keyword-only) | `[users]`, int32, ROW_MAJOR, DRAM interleaved (X4, R8) | — | tensor binding `positions` (TensorAccessor, read) |
| `page_table` | `ttnn.Tensor` | yes (keyword-only) | `[users, blocks_per_seq]`, int32, ROW_MAJOR, DRAM interleaved; `blocks_per_seq ≤ blocks` (X5, X6, R8) | — | tensor binding `page_table` (TensorAccessor, read) |
| `heads` | derived | — | `cache.shape[1]`, 1..32 | — | CT (`block_heads`) |
| `block_size_tiles` | derived | — | `cache.shape[2] // 32` | — | CT (a compile-time divisor: see Risks) |
| `dtiles` | derived | — | `D // 32` | — | CT (`block_dtiles`) |
| `face_row_bytes` | derived | — | `16 × element_size(dtype)`: 32 (bf16), 64 (fp32) | — | CT |
| `face_bytes` | derived | — | `256 × element_size(dtype)` | — | CT |
| `input_page_bytes` | derived | — | input tile page size aligned up to `ttnn.get_l1_alignment()` | — | CT |
| `pos_page_bytes` | derived | — | `update_idxs_tensor` aligned page size = `align_up(users × 4, ttnn.get_dram_alignment())` | — | CT |
| `pt_row_bytes` | derived | — | `page_table` aligned page size = `align_up(blocks_per_seq × 4, ttnn.get_dram_alignment())` | — | CT |
| `first_user`, `num_users_this_core` | derived per core | — | from the input's shard spec | — | RT |

The op has no non-tensor argument (`CALL["arguments"] == {}`) and no compute config. The entry point `paged_update_cache(cache_tensor, input_tensor, *, update_idxs_tensor, page_table)` equals the specification's stub and returns **`cache_tensor` itself** (R3).

### Host validation (`validate()`, before any program is built)

The op file imports `INPUT_TAGGERS = AXES` from the specification (`users`, `alignment`) and derives `memory_config` / `sharding` with `eval.memory.memory_axes(input_tensor)` (R7). `SUPPORTED` = `PHASE0`. Order: malformed calls first (any of `NotImplementedError, ValueError, RuntimeError, TypeError` is accepted; use `ValueError`), then the support gate (`NotImplementedError`).

| Check | Raises | Covers |
|-------|--------|--------|
| cache and input rank 4 | ValueError | X11 |
| `input.shape[0] == 1` | ValueError | X3 |
| `input.shape[-1] == cache.shape[-1]` | ValueError | X2 |
| `cache.shape[-1] % 32 == 0` (D) | ValueError (and `alignment = w_non_aligned` is outside SUPPORTED) | X10 |
| `cache.shape[2] % 32 == 0` (block size) | ValueError | X7 |
| `input.shape[2] >= cache.shape[1]` (rows ≥ heads) | ValueError | X1 |
| `input.shape[2] <= 32` | ValueError | X8 |
| `update_idxs_tensor.shape == [users]` | ValueError | X4 |
| `page_table.shape[0] == users` | ValueError | X5 |
| `page_table.shape[1] <= cache.shape[0]` | ValueError | X6 |
| cache: TILE, DRAM interleaved; index tensors: int32, ROW_MAJOR, DRAM interleaved; `input.dtype == cache.dtype` | NotImplementedError | R8 |
| input and cache tile shape `[32, 32]`, face shape `[16, 16]` | NotImplementedError | mechanism cap |
| cell axes (`dtype`, `layout`, `memory_config`, `sharding`, `users`, `alignment`) ∈ SUPPORTED | NotImplementedError | R7, R9, X9 |
| `index_scratch_bytes <= ttnn.get_max_worker_l1_unreserved_size() - l1_reserve_bytes` | NotImplementedError (names the bound) | mechanism cap |

`l1_reserve_bytes` = 262,144 (host constant, single source): headroom for the co-resident L1 tensors on the shard core (the input shard itself is ≤ 16 tiles × 4 KiB = 64 KiB at the widest TARGET shape) and the program's own allocations.

## Tensors

### Input

| Property | `cache_tensor` | `input_tensor` | `update_idxs_tensor` | `page_table` |
|----------|----------------|----------------|----------------------|--------------|
| Shape | `[blocks, heads, block_size, D]` | `[1, users, H, D]`, `heads ≤ H ≤ 32` | `[users]` | `[users, blocks_per_seq]` |
| Dtype | bfloat16 (Phase 0), float32 (TARGET) | == cache dtype | int32 | int32 |
| Layout | TILE | TILE (each user is one tile row: H pads to 32 per image) | ROW_MAJOR (one page of `users` int32) | ROW_MAJOR (one page per user row) |
| Memory | DRAM interleaved | L1 HEIGHT_SHARDED, any core(s), any grid | DRAM interleaved | DRAM interleaved |

### Output

| Property | Value |
|----------|-------|
| Shape | `cache_tensor.shape` |
| Dtype | `cache_tensor.dtype` |
| Layout | TILE |
| Memory | DRAM interleaved — **the caller's own buffer**: `cache_tensor` is the last element of `io_tensors` of the one `ttnn.generic_op`, and the entry point returns `cache_tensor` (R3). No allocation. |

## Blocking Model

The first design section. Everything below it is downstream of these decisions. Semantics: `.claude/references/blocking-model.md`.

**Work unit.** The atomic unit is one `(user, head, D-tile)` triple: row `h` of input tile `j` of user `u` goes to row `r = p % 32` of cache tile `j` of tile row `(page_table[u, p // block_size] · heads + h) · block_size_tiles + (p % block_size) // 32`. In tile layout a 32-element tile row is two 16-element **face rows** (faces `2·(row // 16)` and `2·(row // 16) + 1`, at `row % 16` inside each face), `face_bytes` apart; so one unit is **two segment transfers** of `face_row_bytes`. Nothing is computed.

### Axes

| Axis | Character (+ one-clause reason) | Extent knob | Phase 0 value | Knob source | Core-assignment | Later unlock |
|------|--------------------------------|-------------|---------------|-------------|-----------------|--------------|
| `users` (input dim 1) | independent — each user writes its own positions of its own blocks; nothing spans users | `block_users` | `num_users_this_core` = the users held in this core's shard (Phase 0: 1) | host, from the input's shard spec: `users_per_shard = shard_height // 32`, `num_users_this_core = min(users_per_shard, users − first_user)` → RT `num_users_this_core`, `first_user` | **the caller's shard**: the core holding user `u`'s shard owns user `u` (the shard fixes core-assignment and per-core extent; the block is the whole shard) | knob-turn (more users per shard / more shard cores need no change) |
| `heads` (cache dim 1; input rows `0..heads−1`) | independent — each head writes its own cache tile row | `block_heads` | `heads` (whole axis) | host CT `heads` = `cache.shape[1]` | **not split across cores**: all heads of a user on that user's shard core — the data is resident there, and any other core would first pull it over the NoC (Traffic ranking); rows `heads..H−1` are never read (R6) | knob-turn within the shard core (thread plan lamp); cross-core = `grid_spread_pull` regime (deferred) |
| `dtiles` (D / 32) | independent — each D-tile is its own cache page | `block_dtiles` | `dtiles` (whole axis) | host CT `dtiles` = `D // 32` | **not split across cores**, same reason as `heads` | same as `heads` |
| `face_half` (the two 16-element segments of a tile row) | independent — two disjoint byte runs | — (fixed 2, a tile-format constant) | 2 | `TILE_FACES_PER_ROW = 2` (tile 32 × 32, faces 16 × 16; host checks the tile shape) | with its unit | none |
| `blocks` (cache dim 0) | not a work axis — selected per user by the page-table entry the position picks | — | — | — | — | none |
| `block_size` positions (cache dim 2) | not a work axis — one position per user is written; selects the tile row `(p % block_size) // 32` and the in-tile row `p % 32` | `block_size_tiles` (CT divisor, not an extent) | `cache.shape[2] // 32` | host CT | — | none |
| `blocks_per_seq` (page-table dim 1) | not a work axis — only the entry `p // block_size` is used (R5); the row is read whole to lift the second dependent DRAM read into the first one's stall (Stall shadow) | `pt_row_bytes` (CT, the row's aligned page) | `align_up(blocks_per_seq·4, dram_alignment)` | host CT, from the tensor's aligned page size | read by the core owning the user | regime `dependent_entry_read` (deferred) past the scratch cap |

**One block per core.** The block is the rectangle `block_users × block_heads × block_dtiles × face_half`, i.e. this core's whole assignment (the resident shard's users, every head, every D-tile). `num_blocks_this_core = 1` by construction (`blocking-model.md` §1 "the core's own assignment" is the bound; the block's L1 working set is the index scratch only, a few hundred bytes — L1 never binds). The only sub-division is the declared transaction-counter cap below, which no Phase 0 or TARGET shape reaches.

**Intermediate data.** The scheme produces one internal datum per user: the resolved `(skip, cache_tile_row_base, row_in_tile)` triple from `load_index_block`. It has the `users` axis only, is computed on the core that owns the user, and is consumed there; no stage runs on a designated core while others wait.

### Buffer-depth knobs

| CB | Depth knob | Phase 0 value | What the depth buys |
|----|------------|---------------|---------------------|
| `index_scratch` (ScratchpadSpec) | none — one slot | 1 | The block is the whole assignment and the scratch is read once per program; there is no second block to overlap. |
| (no DFB rings) | — | — | The op has no producer/consumer pair: the input is written from where it sits, and the cache is never staged. |

### Mechanism caps

| Mechanism | Cap on which extent | Clamp | What happens unclamped |
|-----------|--------------------|-------|------------------------|
| Tile face geometry (32 × 32 tile, 16 × 16 faces, faces row-major inside a tile) | the segment split of a unit: exactly 2 runs of 16 elements, `face_bytes` apart | host refuses (`NotImplementedError`) any input or cache whose `tile.tile_shape != [32, 32]` or face shape ≠ `[16, 16]` | segments land at wrong offsets: silently corrupt other rows of the cache (R2) |
| NoC transfer alignment | segment size and offsets (`face_row_bytes` = 32 / 64 B at multiples of themselves) | kernel `static_assert(face_row_bytes % NOC_DRAM_WRITE_ALIGNMENT_BYTES == 0 && face_row_bytes % NOC_L1_READ_ALIGNMENT_BYTES == 0)`; Quasar = 1 B (`noc_parameters.h:486-491`), WH/BH ≤ 16 B for writes — holds on every generation | a misaligned write would be widened by hardware and overwrite neighbours |
| Outstanding transactions per transaction id (`NOC_MAX_TRANSACTION_ID_COUNT = 0xFFFF`, `noc_parameters.h:36`) | writes issued between two barriers | `writes_per_barrier = min(2·block_users·heads·dtiles, (NOC_MAX_TRANSACTION_ID_COUNT + 1) / 2)` — host constant `WRITE_TXN_CAP = 32768`; Phase 0 max is 2·1·32·16 = 1,024 ≪ cap, so one barrier per block | — |
| User transaction ids on Quasar (`USER_TXN_ID_MAX = 7`, `dataflow_buffer_config.h:17`) | which trid the kernel may use | trid 0 for every transfer | ids ≥ 8 belong to the DFB runtime |
| Index scratch (`ScratchpadSpec.size_per_node`) | `block_users · pt_row_bytes + pos_page_bytes` | host predicate `index_scratch_bytes ≤ ttnn.get_max_worker_l1_unreserved_size() − l1_reserve_bytes`; beyond it refuse with `NotImplementedError` naming the bound (regime `dependent_entry_read` is the deferred successor) | program allocation fails at build |
| Compile-time divisors (Quasar DM build, see Risks) | `block_size_tiles`, `heads`, `dtiles` used as divisors | pass as CT args; walk `(h, j)` with nested loops, never `div/mod` of a flattened index | runtime `__udivsi3` can break the Quasar XIP link (`reader_unary_local_l1_copy_backwards.cpp:33-38`) |

### Regimes

| Regime | Status | Predicate | Block | Data movement vs. minimum | What a bigger block buys |
|--------|--------|-----------|-------|---------------------------|--------------------------|
| `direct_row_write` | **built** (Phase 0, and every TARGET cell: bf16 and fp32 differ only in `face_row_bytes`; `tile_aligned` vs `h_non_aligned` only in H, which no kernel reads) | `index_scratch_bytes ≤ budget` (always true at INPUTS: max 576 B vs 3.7 MB) | `block_users × block_heads × block_dtiles × 2` segments, the whole shard | **Minimum against the DRAM boundary is reached for the cache and the input**: each written element crosses into DRAM exactly once (`heads·D·elem` B per non-skipped user), the cache is never read, the input never leaves L1 except into its destination. Above minimum only on the index side: the whole page-table row (`pt_row_bytes`) is read instead of its one 4-byte entry, and the 4-byte position arrives in a 64-B aligned page — bought deliberately to take one dependent DRAM round trip off the critical path | One index load (2 reads, one read barrier) and one write barrier per block instead of per user; program launch amortized over the shard. Intended frequency: 1 read barrier + 1 write barrier + 2 trid hand-backs per core per call |
| `dependent_entry_read` | **deferred** — positive reason: only needed when one page-table row exceeds the L1 budget (> ~925 K entries per user at 3.7 MB), which no INPUTS entry, model or realistic cache approaches (largest: 128 entries = 512 B); `direct_row_write` serves every shape it would. Reachable: `load_index_block` becomes two dependent reads (position, then the `dram_alignment`-byte window holding the entry at `align_down((p // block_size)·4, A)`), nothing else changes | `index_scratch_bytes > budget` | same | index bytes drop to `pos_page_bytes + A` per user; adds one dependent DRAM round trip per call | same |
| `grid_spread_pull` | **deferred** — positive reason: it moves strictly more bytes on both tiers (every input byte crosses the NoC twice: shard core → worker, worker → DRAM; index reads repeat on every engaged core, all against the one or two banks holding `update_idxs_tensor` page 0 and the user's row) and cannot shorten the critical path, which is the index round trip every engaged core repeats; `direct_row_write` covers every shape. What it would buy: write issue spread over G cores (≤ 1,024 transfers per user at the widest TARGET shape). Reachable: the write loop already takes its unit range from `(h, j)` bounds; swapping the source endpoint from `LocalTensorAccessor` to the input's sharded `TensorAccessor` (same `offset_bytes` arithmetic) and splitting `(heads × dtiles)` with `ttnn.split_work_to_cores(row_wise=True)` is the whole change; the segment pull is independent of the index read and goes into its stall | `heads·dtiles ≥ G·k` for a measured `k` | per worker: a contiguous range of `(h, j)` units | + `heads·D·elem` B cross-core per user; + (G − 1) × (`pos_page_bytes + pt_row_bytes`) DRAM reads | fewer cores, more units each: the grid-sync trade |
| `index_broadcast` | **deferred** with `grid_spread_pull` (only meaningful under it) — positive reason: with one engaged core per user there is no second reader to serve. Reachable: one injector reads the position and row and multicasts them with `ttnn.Mcast` / `McastArgs` (`ttnn/cpp/ttnn/kernel_lib/mcast/`), workers `receive()` into `index_scratch` | `grid_spread_pull` built | — | removes (G − 1) index reads, adds one multicast + semaphore round trip | — |
| `tile_rmw` | **rejected** — superseded by `direct_row_write`: read each touched cache tile, patch row `r` in L1, write the tile back. Moves `2·tile_bytes` per unit instead of `2·face_row_bytes` (32× the DRAM bytes: 4,096 B vs 64 B per bf16 unit), adds a scratch ring of `block_dtiles·block_heads` tiles and a dependent read→patch→write chain; it exists only where a NoC cannot write a 32-B run into DRAM, which no supported generation has | — | — | 2·tile_bytes per unit | — |
| `untilize_patch_tilize` (tt-metal's own approach, per GUIDANCE) | **rejected** — superseded by `direct_row_write`: a compute kernel untilizes the tile row, the row is overwritten, the row is tilized back. Same 32× DRAM bytes as `tile_rmw`, plus two unpack/pack passes; a float32 value through SrcA is at risk of a tf32 round trip, which R1 forbids; nothing in the op needs the compute engines | — | — | as `tile_rmw` + 2 L1 passes | — |

Only one regime is built, so there is no regime-selection function beyond the scratch predicate, which is a refusal, not a second path.

### Traffic ranking

Counted per user and call; `E` = element bytes (2 / 4), `A` = DRAM alignment (64 on Quasar), `G` = cores engaged per user.

| Rank | Split | DRAM bytes | Cross-core bytes | Transfers | Note |
|------|-------|------------|------------------|-----------|------|
| 1 (chosen) | users across the caller's shard cores; `heads × dtiles` whole on the shard core (`direct_row_write`) | writes `heads·D·E`; reads `pos_page_bytes + pt_row_bytes` | 0 | `2·heads·dtiles` writes + 2 reads | the named-boundary minimum for both data tensors |
| 2 | `grid_spread_pull`: `heads × dtiles` over G cores | writes `heads·D·E`; reads `G·(pos_page_bytes + pt_row_bytes)` (same 1–2 banks) | `heads·D·E` | same writes + `2·heads·dtiles` pulls + 2G reads | more bytes on both tiers; the pull hides in the index stall, the bank contention does not |
| 3 | `grid_spread_pull` + `index_broadcast` | as rank 1 for reads | `heads·D·E` + multicast of `pos_page_bytes + pt_row_bytes` to G − 1 | + semaphore round trip | |
| 4 | `tile_rmw` on any split | reads + writes `2·heads·dtiles·tile_bytes` | 0 (shard-local) | `2·heads·dtiles` | 32× the minimum |
| — | splitting a dependent axis | none exists: no result spans any axis | | | |

Llama traced call (8 heads, D = 64, bf16, 128-entry page table) on Quasar: rank 1 writes 1,024 B in 32 transfers and reads 64 + 512 B; rank 4 would move 65,536 B. Qwen (8 heads, D = 128, 8-entry table): 2,048 B in 64 writes, reads 64 + 64 B.

**Operand-reuse check** (every operand vs. the chosen split, which is by user): `cache_tensor` — varies by user (distinct blocks per the page table), not shared. `input_tensor` — varies by user (its own shard). `update_idxs_tensor` — one page holds every user's position: with one engaged core per user each core reads the page once; shared only under `grid_spread_pull` (→ `index_broadcast`). `page_table` — one row per user, read by that user's core only.

**Stall-shadow check.** The one wait is `load_index_block`: the write destinations depend on the position and on the page-table entry it selects. The naive order is two dependent DRAM round trips (position, then entry). Reading the **whole page-table row** does not depend on the position, so both reads are issued together and one round trip is hidden in the other — the `pt_row_bytes − 4` extra bytes are the price (`dependent_entry_read` is the opposite trade, deferred). Nothing else can fill the stall: the source data is already resident and every write needs its destination. No floating-point order is involved (nothing is computed).

### Block schedule

```cpp
// On each core that holds a user's shard (Phase 0: one core). num_blocks_this_core == 1.
for (uint32_t block_idx = 0; block_idx < num_blocks_this_core; ++block_idx) {
    load_index_block(block_idx);    // positions page + block_users page-table rows -> index_scratch; ONE read barrier
    resolve_block(block_idx);       // per user: skip if p == -1, else (cache tile-row base, row in tile)
    write_rows_block(block_idx);    // every non-skipped user x heads x dtiles x 2 segments: shard -> cache page+offset; ONE write barrier
}
release_tags();                     // hand the hart's transaction tag back (reads and writes)
```

| Operation | Block shape | Resident across it | Intended frequency of fixed costs |
|-----------|-------------|--------------------|-----------------------------------|
| `load_index_block` | `1` positions page (`pos_page_bytes`) + `block_users` page-table rows (`pt_row_bytes` each) | the input shard (caller-owned, untouched) | all `1 + block_users` reads tagged trid 0 issued back to back, **one** `async_read_barrier<TXN_ID>({.trid = 0})` per block |
| `resolve_block` | `block_users` positions | `index_scratch` (read by the CPU under one `scoped_lock` of the touched bytes — the DM core's cache must be invalidated after the NoC wrote them) | one scoped lock per block; per user: one compare against −1, divisions only by CT constants (`p >> 5`, `& 31`, `/ block_size_tiles`) |
| `write_rows_block` | `block_users × block_heads × block_dtiles × 2` segments of `face_row_bytes` | the input shard (source, in place via `LocalTensorAccessor`), `index_scratch` values held in registers | all writes tagged trid 0 issued back to back (non-posted); **one** `async_write_barrier<TXN_ID>({.trid = 0})` per `writes_per_barrier` (= whole block at every shape the spec has) |
| `release_tags` | — | — | once per kernel: `noc_async_read_set_trid(0, noc.get_noc_id())` and `noc_async_write_set_trid(0, noc.get_noc_id())` as the last statements |

Per write, with `r = p & 31`, `row_tile = (p >> 5) − k·block_size_tiles` where `k = (p >> 5) / block_size_tiles`, `blk = page_table_row[k]`, and `s ∈ {0, 1}`:

| Quantity | Expression |
|----------|------------|
| cache page id | `((blk · heads + h) · block_size_tiles + row_tile) · dtiles + j` |
| cache offset | `(2·(r >> 4) + s) · face_bytes + (r & 15) · face_row_bytes` |
| source offset in the local shard | `((u − first_user) · dtiles + j) · input_page_bytes + (2·(h >> 4) + s) · face_bytes + (h & 15) · face_row_bytes` |
| size | `face_row_bytes` |

`h` runs over `0 .. heads − 1` only (R6: rows `heads..H−1` and the tile padding are never read). The page-table entries other than `k` are never interpreted (R5), and a skipped user issues no write (R4).

### Perf lamps

| Lamp | Why the default may be wrong here | Nearby alternative to measure |
|------|-----------------------------------|-------------------------------|
| Grid synchronization (inverted): engaged cores = the shard's cores | With one user the whole op runs on one cluster while 31 idle; issuing up to 1,024 small writes from one hart could outlast the index round trip at the widest shapes | `grid_spread_pull` with G = 2, 4, 8 at heads = 32, D = 512 (fp32) vs. the built regime. Structural count on ttsim: writes issued per hart; time only on ttsim-perf / hardware |
| Writer threads | One DM thread issues every segment write; the cluster has 5 more user harts | the same kernel with `num_threads = W ∈ {2, 3}`, units `(h, j)` dealt round-robin `t, t+W, …`, each thread reading the index itself; count: writes per hart |
| Page-table row prefetch | Reading the whole row hides one round trip but makes the read longer as `blocks_per_seq` grows | `dependent_entry_read` at `blocks_per_seq` = 4,096 (16 KiB row) vs. the built prefetch |

No overlap lamp: there is one phase per core and no compute to overlap with.

## Dataflow Strategy

| Stage | Format | Mechanism | Notes |
|-------|--------|-----------|-------|
| Positions | int32 RM page, DRAM → `index_scratch[0 : pos_page_bytes)` | `noc.async_read<TXN_ID>(positions_accessor, index_scratch, pos_page_bytes, {.page_id = 0}, {.offset_bytes = 0}, {.trid = 0})` | the whole page (`users` int32, aligned to `A`); user `u`'s position is word `u` |
| Page-table rows | int32 RM pages, DRAM → `index_scratch[pos_page_bytes + i·pt_row_bytes : …)` | one tagged read per user of the block, page id `first_user + i` | issued with the position read, one barrier for all |
| Resolve | int32 in L1 → registers | CPU loads inside `Scratchpad::scoped_lock` (`scratchpad.h:136`) | −1 → skip the user |
| Row write | tile-layout face rows, input shard (L1, local) → cache page in DRAM at an offset | `noc.async_write<TXN_ID>(input_local, cache_accessor, face_row_bytes, {.offset_bytes = src_off}, {.page_id = pg, .offset_bytes = dst_off}, {.trid = 0})` | zero-copy consumption of the resident shard; no ring, no staging, no cache read |
| Completion | — | `async_write_barrier<TXN_ID>({.trid = 0})`, then trid hand-back | |

**Placement axes as regimes.** TARGET's placement is `memory_config = L1`, `sharding = height` only. Height sharding cuts the `users` axis, which is independent: each shard core owns whole users, so the physical shard **is** the built regime's core-assignment, a knob-turn and not a scheme-change. The shard is consumed in place through `LocalTensorAccessor` (a DFB borrowed from the shard with DM-only endpoints is a forbidden self-loop on Quasar, and a ring would only add a copy: every byte the op needs leaves the shard straight into its destination). Under the deferred `grid_spread_pull`, the cores not holding the shard would read their segments with the input's sharded `TensorAccessor` (cross-core unicast pull, no semaphore: the input is complete before the program starts); the shard core itself keeps `LocalTensorAccessor`.

**TARGET − PHASE0 as refinements.**

| Axis value | Class | What changes |
|------------|-------|--------------|
| `dtype = float32` | knob-turn | `face_row_bytes` / `face_bytes` / page sizes derive from the element size; SUPPORTED + validate only. R1 holds trivially (bytes are moved, never converted) |
| `alignment = tile_aligned` (H = 32) | knob-turn | none in the kernels (they never read H); validate only. R6 rows beyond `heads` are already ignored |
| `users > 1` | not in TARGET | already general: one core per shard, `block_users` per core |

**Structural impossibilities.** None to add: the specification's empty `INVALID` is right for this rectangle.

**Index axes.** The op has no `dim`/`axis` argument; nothing to canonicalize.

## Work Distribution

| Field | Value |
|-------|-------|
| Work unit | a block = one shard core's users × all heads × all D-tiles × 2 segments |
| Grid | the cores of `input_tensor.memory_config().shard_spec.grid` that hold at least one user, in the shard orientation's order (`ttnn.corerange_to_cores(grid, None, orientation == ROW_MAJOR)`), first `ceil(users / users_per_shard)` of them; never assumed to be (0, 0) and never assumed to be more than one core |
| Per-core work | `first_user = i · users_per_shard`, `num_users_this_core = min(users_per_shard, users − first_user)` for the i-th data core; `users_per_shard = shard_shape[0] // 32` |
| Remainder | the last data core takes the ragged `num_users_this_core`; tile-row count per user is `ceil(H / 32) = 1` (per-image padding, X8 keeps H ≤ 32); `dtiles = ceil(D / 32)` with D % 32 == 0 enforced |
| Phase 0 on Quasar | users = 1 → one engaged cluster (the one holding the shard; core (0, 0) in the golden runner, any core in general); 2·heads·dtiles writes there (16–1,024 by INPUTS) |
| Bound | **data movement** — there is no compute at all; per unit 2 transfers of 32/64 B, and the critical path is the index round trip followed by the write acknowledgements |
| Thread plan | `(R, C, W) = (0, 0, 1)`: **one data-movement kernel `row_writer` with `num_threads = 1`, no compute kernel**. The index reads live in the same thread because every write depends on them (a separate reader would add only a handoff). No compute kernel because nothing is computed (R1). Host constant `ROW_WRITER_THREADS = 1`, passed as CT `writer_threads`; the kernel loops are written against it (units `(h, j)` with `idx % writer_threads == thread_id`), so the writer-threads lamp is a one-constant change |
| Unit dealt to compute threads | none (no compute threads). Units are not dealt to rings: there are no rings |
| Tests the plan needs | (1) the generated acceptance test and the golden suite; (2) **the op in one device session after another program** (e.g. a `ttnn.to_layout` or the op called twice) **and before one of tt-metal's own** (`ttnn.to_layout(cache, ttnn.ROW_MAJOR_LAYOUT)`), each in one session — the trid hand-back is what it checks; (3) **the shard on a core other than (0, 0)** (e.g. a height-sharded input built on `CoreRange((5, 2), (5, 2))`), result compared with the reference — shows the op follows the shard and does not assume the core; (4) a refusal test with a lowered scratch cap (or a monkeypatched `l1_reserve_bytes`) for the `index_scratch_bytes` predicate; (5) with `writer_threads = 1` a correct result shows the one thread ran (a later W > 1 must add a test where a thread's units are visibly its own, e.g. skipping one thread leaves exactly its units unwritten) |

## Circular Buffers

There are **no dataflow buffers (rings)**. The one allocation is a kernel-private scratchpad.

| Semantic Name | Index | Page Size | Num Pages | Sizing rationale | Format | Producer | Consumer | Sync | Lifetime |
|---------------|-------|-----------|-----------|------------------|--------|----------|----------|------|-----------|
| `index_scratch` (`ScratchpadSpec`, accessor `scratch::index_scratch`) | — (scratchpad, not a ring) | 1 byte (`size_per_node` in bytes) | `pos_page_bytes + block_users_max · pt_row_bytes` where `block_users_max = users_per_shard` | spans `users` (one row slot per user of the block); streams nothing else; constant in `heads`, `dtiles` | no `data_format_metadata` (DM-only; int32 bytes) | `row_writer` (NoC reads) | `row_writer` (CPU loads) | explicit: tagged trid-0 reads + read barrier, then `scoped_lock` for CPU visibility | `load_index_block` → `resolve_block` |
| `input` shard (tensor-owned; not allocated by the op) | — | `input_page_bytes` | `users_per_shard · dtiles` (the caller's shard) | the caller's allocation | cell dtype | caller | `row_writer` (NoC write source) | none needed: complete before launch | whole program |

Full accounting: `l1_ledger.md`.

## Block Operation Realization

| # | Block operation | Block shape | Helper? | Input CB (semantic name, pages, state) | Output CB (semantic name, pages) | CB state after |
|---|-----------------|-------------|---------|----------------------------------------|----------------------------------|----------------|
| 1 | `load_index_block` | `1 + block_users` int32 pages | no (raw NoC; see API Mapping) | `positions`, `page_table` DRAM accessors | `index_scratch` (`pos_page_bytes + block_users·pt_row_bytes` B) | filled, read barrier done |
| 2 | `resolve_block` | `block_users` positions | no (CPU) | `index_scratch` | registers: `skip[u]`, `blk[u]`, `row_tile[u]`, `r[u]` | `index_scratch` dead |
| 3 | `write_rows_block` | `block_users × block_heads × block_dtiles × 2` segments | no (raw NoC) | `input` shard (LocalTensorAccessor), registers | `cache` DRAM accessor (pages written at offsets) | writes acknowledged |
| 4 | `release_tags` | — | no | — | — | tag register = 0 |

## API Mapping

| Block operation | Type | Function | File:Line | Template Params / Args | Input CB | Output CB | Which params are block knobs |
|-----------------|------|----------|-----------|------------------------|----------|-----------|------------------------------|
| `load_index_block` | raw_api | `Noc::async_read<NocOptions::TXN_ID>(TensorAccessor, Scratchpad, size, {.page_id, .offset_bytes}, {.offset_bytes}, {.trid = 0})` | `tt_metal/hw/inc/api/dataflow/noc.h:252`; src args `api/tensor/noc_traits.h:136-139`; Scratchpad endpoint `internal/local_mem/noc_traits.h:75-89` | size `pos_page_bytes` / `pt_row_bytes`; page ids `0` / `first_user + i` | `positions`, `page_table` | `index_scratch` | `block_users` (count of row reads) |
| `load_index_block` | raw_api | `Noc::async_read_barrier<NocOptions::TXN_ID>({.trid = 0})` | `noc.h:739` | trid 0 | | | — |
| `resolve_block` | raw_api | `Scratchpad<int32_t>::scoped_lock(offset, n)` + `operator[]` | `tt_metal/hw/inc/api/scratchpad.h:136`, `:55` (ctor from `scratch::index_scratch`) | `n = pos_page_bytes/4` then per row | `index_scratch` | — | `block_users` |
| `write_rows_block` | raw_api | `LocalTensorAccessor<uint32_t>(tensor::input)` | `tt_metal/hw/inc/api/tensor/local_tensor_accessor.h:51`; NoC endpoint `internal/local_mem/noc_traits.h:106-124` | src `offset_bytes` per the table above | `input` shard | — | `block_users`, `block_heads`, `block_dtiles` (loop bounds) |
| `write_rows_block` | raw_api | `Noc::async_write<NocOptions::TXN_ID>(LocalTensorAccessor, TensorAccessor, face_row_bytes, {.offset_bytes}, {.page_id, .offset_bytes}, {.trid = 0})` | `noc.h:411`; dst args `api/tensor/noc_traits.h:140-143` | non-posted | `input` shard | `cache` | — |
| `write_rows_block` | raw_api | `Noc::async_write_barrier<NocOptions::TXN_ID>({.trid = 0})` | `noc.h:760` | once per `writes_per_barrier` | | | `writes_per_barrier` |
| `release_tags` | raw_api | `noc_async_read_set_trid(0, noc.get_noc_id())`, `noc_async_write_set_trid(0, noc.get_noc_id())` | `tt_metal/hw/inc/api/dataflow/dataflow_api.h:2558`, `:2579` | — | | | — |
| host | raw_api | `ttnn.experimental.ProgramSpec`, `KernelSpec(hw_config=E.create_writer_datamovement_config(), num_threads=ROW_WRITER_THREADS, scratchpad_bindings=[E.ScratchpadBinding(scratchpad_spec_name="index_scratch", accessor_name="index_scratch")], tensor_bindings=[input, positions, page_table, cache])`, `ScratchpadSpec(unique_id="index_scratch", size_per_node=…)`, `TensorParameter` ×4, `WorkUnitSpec(target_nodes=<data cores>)`, `ttnn.generic_op([input, positions, page_table, cache], spec, run_args, {"input": 0, "positions": 1, "page_table": 2, "cache": 3})` | `ttnn/cpp/ttnn-nanobind/program_spec.cpp:365-398` (ScratchpadSpec), `:704-718` (ScratchpadBinding), `:839-901` (WorkUnitSpec, ProgramSpec), `:476` (writer DM config); `ttnn/cpp/ttnn/operations/generic/generic_op_nanobind.cpp` (Metal 2.0 overload; in-place = output last) | — | | | `ROW_WRITER_THREADS`, `l1_reserve_bytes`, `WRITE_TXN_CAP` |

**Helpers considered and rejected** (the op has no compute phase, so no compute helper applies; for the data-movement operations):

| Operation | Candidate | Mismatch (file:line) |
|-----------|-----------|----------------------|
| `write_rows_block` | `dataflow_kernel_lib::local_addr` / `set_read_state` / `read_with_state` | `ttnn/cpp/ttnn/kernel_lib/local_copy_helpers_dataflow.hpp:7-33`: L1 → L1 self-aimed **reads** only; the destination here is a DRAM page at an offset, reached with a write |
| `write_rows_block` | `fill_tile_column` / `copy_tile_column` | `ttnn/cpp/ttnn/kernel_lib/tile_column_dataflow.hpp:61-92`: CPU stores into a tile **column** of a tile resident in L1; the target here is a tile **row** of a tile in DRAM, and no tile is staged |
| `load_index_block` | `ttnn/cpp/ttnn/kernel_lib/l1_helpers.hpp:30-41` (`addr_to_l1_ptr`, `local_noc_addr`) | raw volatile pointer without the Quasar cache invalidation; `Scratchpad::scoped_lock` (`scratchpad.h:136`) is the API that performs it |
| (deferred `index_broadcast`) | `ttnn.Mcast` / `McastArgs` (`ttnn/cpp/ttnn/kernel_lib/mcast/`) | not used in Phase 0: only one engaged core per user; it is the mechanism named for the deferred regime |

## Key Risks and Gotchas

| Risk | Why it bites here | Mitigation in this design |
|------|-------------------|---------------------------|
| Returning a new tensor instead of the cache (R3) | `HOOKS["check"]` calls the op again with the returned tensor and reads the caller's | `cache_tensor` is the last `io_tensors` element and its own `TensorParameter`; the entry point returns `cache_tensor`; nothing is allocated |
| Rows past `heads` written (R6) | the input has H ≥ heads rows; an op taking the head count from the input would write into the next head's (another block's) tile rows | `heads` comes from `cache.shape[1]` only; the INPUTS entry with 2 heads in 8 rows catches it |
| A skipped user (−1) still writes (R4) | the check is on device | `resolve_block` compares the int32 against −1 before any address is formed |
| CPU reads stale index bytes | Quasar DM cores reach L1 through a cache (`core_local_mem.h:28`, `scoped_lock` at `:145`) | read `index_scratch` only inside `scoped_lock` after the read barrier |
| Untagged transfer after an implicit-sync program, or a ring id left in the tag register | — | every transfer `TXN_ID` with `{.trid = 0}`, barriers on trid 0, `release_tags` as the last statements; test (2) in Work Distribution |
| The shard is not on (0, 0) / the device has one core | GUIDANCE and R7 | target nodes come from the shard spec; test (3) |
| Program built on a node subset of the input's shard grid | Metal 2.0 might require a `TensorBinding`'s nodes to cover the sharded tensor's grid | if validation refuses, put the kernel on the whole shard grid with `num_users_this_core = 0` on the cores without data (they return immediately) |
| Runtime division in a Quasar DM kernel | `__udivsi3` pulled in has broken the Quasar XIP link (`reader_unary_local_l1_copy_backwards.cpp:33-38`) | `block_size_tiles`, `heads`, `dtiles` are CT args (division by a constant compiles to a multiply); `p >> 5`, `p & 31`; nested `(h, j)` loops instead of `div/mod` of a flattened index |
| Faces assumed 16 × 16 | segment offsets are computed from the face layout | host refuses any other tile/face shape (Mechanism caps) |
| A page-table entry out of range | the spec leaves it unspecified (R5, DECISIONS) | not checked, as specified; only entry `k` is read |
| `int32` position read as unsigned | −1 must compare as −1 | read the scratch as `int32_t`; compare before shifting |
| ttsim gives no timing | perf claims | perf lamps are structural counts (writes per hart, bytes per tier); time only on ttsim-perf / hardware |
