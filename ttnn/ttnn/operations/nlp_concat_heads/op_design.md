# Operation Design: nlp_concat_heads

## Overview

| Field | Value |
|-------|-------|
| Classification | data_movement (a whole-tile permutation in Phase 0; no compute kernel) |
| Goal | Put the heads of an attention output side by side: `[B, H, S, D]` → `[B, 1, S, H·D]`, every element moved unchanged, in one `ttnn.generic_op` dispatch of one `ttnn.ProgramSpec`. |
| Math | `output[b, 0, s, h·D + d] = input[b, h, s, d]`. Tile form (D % 32 == 0): output tile `(b, r, h·Dt + c)` = input tile `(b, h, r, c)`, with `Dt = D/32` and `St = ceil(S/32)` tile rows per image. |
| References | Specification `eval/specs/nlp_concat_heads/spec.py` (governs). Target `.claude/references/targets/quasar.md`; backend `.claude/references/backends/ttsim.md`. Host pattern `.claude/references/quasar_op_examples/toy_thread_plan/` (`toy_thread_plan_program_artifacts.py`, `kernels/reader_implicit.cpp`, `kernels/writer_implicit.cpp`). DM→DM ring with no compute kernel: `ttnn/ttnn/operations/toy_spec_single_tensor/toy_spec_single_tensor.py:86-107` (fill → writer). Ring semantics `tt_metal/impl/dataflow_buffer/DataflowBuffer.md`. Catalog `ttnn/ttnn/operations/examples/master.md` (entries `tile_reorder`, `double_buffer`, `noc_placement`, `width_split`, `bank_stagger`). |

## Target

| Fact | Value | Source |
|------|-------|--------|
| Target | `quasar` on the ttsim simulator (`libttsim.so`, untimed) | `.claude/references/targets/quasar.md` + `.claude/references/backends/ttsim.md` |
| Arch | `quasar` | target page |
| Compute grid | 8 × 4 = 32 clusters (worked numbers below); the op queries `device.compute_with_storage_grid_size()` | target page |
| L1 per core | 4,194,304 B (4 MiB) per cluster | target page |
| Unreserved budget | `ttnn.get_max_worker_l1_unreserved_size()`; the exact value on quasar was not established here (< 4,194,304 B). The op's footprint is shape-independent and ≤ 32 KiB, so no budget predicate gates a regime; the host only asserts the ring fits the query. | `unknown` (exact number); target page (bound) |
| Host program | `ttnn.ProgramSpec` + `ttnn.generic_op` — `.claude/references/quasar_op_examples/` | target page |
| Threads per core | 4 NEO compute engines (C ∈ {1,2,4}); 6 user DM harts (R + W ≤ 6); STRIDED rings deal entries round-robin; capacity per thread-endpoint `num_entries / max(P, C)` (`DataflowBuffer.md:255`). This op runs **no compute kernel**: plan (R, C, W) = (1, 0, 1). | target page *Thread plan*; `DataflowBuffer.md:250-256` |
| Unavailable here | `ttnn.ProgramDescriptor` / Metal 1 kernels / legacy C++ program factory (refused at kernel creation, `tt_metal/impl/kernels/kernel.hpp`). `bfloat8_b`, `bfloat4_b` (no path on this family; not in TARGET anyway). Every kernel here ends by handing its tag back; uint16/uint32 are not DFB formats (irrelevant: TARGET dtypes are bfloat16, float32). | target page *Not available* |

Every worked number below is for quasar (32 clusters). Every predicate the op evaluates queries the device.

## Parameters

| Name | Type | Required | Valid Range | Default | CT/RT |
|------|------|----------|-------------|---------|-------|
| `input_tensor` | `ttnn.Tensor` | yes | rank 4 `[B, H, S, D]`, TILE layout, interleaved; Phase 0: bfloat16, DRAM, `S % 32 == 0`, `D % 32 == 0`, `B == 1` | — | host |
| `memory_config` (keyword-only) | `ttnn.MemoryConfig` or `None` | no | interleaved DRAM / L1 config; `None` = the input's buffer, interleaved. Phase 0: an interleaved DRAM config only | `None` | host (allocation only) |

No index axis exists, so no sign canonicalization is needed. No compute config, no `runtime_rows` (spec `TT_METAL["drop"]`).

## Tensors

### Input

| Property | Requirement |
|----------|-------------|
| Shape | `[B, H, S, D]` (rank 4; X1 refuses any other rank with `ValueError`) |
| Dtype | TARGET bfloat16, float32. Phase 0: bfloat16 |
| Layout | TILE (row-major refused by name, R7 / X3) |
| Memory | interleaved, DRAM or L1 (R5). Phase 0: DRAM. Sharded refused by name (R6) |

### Output

| Property | Value |
|----------|-------|
| Shape | `[B, 1, S, H·D]` |
| Dtype | the input's |
| Layout | TILE |
| Memory | interleaved, in the buffer `memory_config` names; `None` → the input's buffer (R3). Allocated with `ttnn.allocate_tensor_on_device(ttnn.TensorSpec(shape, dtype, ttnn.TILE_LAYOUT, memory_config=out_mc), device)` (`ttnn/cpp/ttnn-nanobind/tensor.cpp:290-305`) |

## Blocking Model

The op computes nothing: every output tile is exactly one input tile (Phase 0, `D % 32 == 0`). No axis carries a dependency, no operand is read by more than one block, and the traffic minimum (each input tile read once from DRAM, each output tile written once) is reached by every split. So the split is chosen on occupancy and transfer shape, and all four tensor axes fold into one flat axis in **output page order** (`p = ((b·St + r)·H + h)·Dt + c`), which is contiguous in the output and treats every axis identically (`blocking-model.md` §1, "leading axes fold into one when the op treats them identically and they are contiguous").

### Axes

| Axis | Character (+ one-clause reason) | Extent knob | Phase 0 value | Knob source | Core-assignment | Later unlock |
|------|--------------------------------|-------------|---------------|-------------|-----------------|--------------|
| `out_tile` (flat output page index over b·r·h·c) | independent — each output tile is one input tile, no cross-tile term | `block_tiles` | `core_tiles` (the core's whole contiguous range: `num_blocks_this_core = 1`) | host split: `core_tiles` from `ttnn.split_work_to_cores`; `block_tiles` derived from it | contiguous ranges of `out_tile` over all engaged cores, `split_work_to_cores(grid, tensor_out_tiles, row_wise=True)` | knob-turn |
| `batch` (b, B) | independent — sequences never mix | folded into `out_tile` (outermost); per-block extent = what the range spans | — (Phase 0 B = 1) | `tensor_batch` = `shape[0]`, host | via `out_tile` | knob-turn (validate widening only: index math is per-image from day 1) |
| `token_row` (r, St = ceil(S/32)) | independent — each tile row moves on its own | folded into `out_tile`; per-block extent = what the range spans | — | `tensor_token_rows = ceil(S/32)`, host | via `out_tile` | knob-turn (h_non_aligned: padding rows travel with their tiles) |
| `head` (h, H) | independent — heads are concatenated, not combined | folded into `out_tile` | — | `tensor_heads = shape[1]`, host | via `out_tile` (a token row may be cut between cores mid-row: no reuse, so no cost) | knob-turn |
| `head_dim_tile` (c, Dt = D/32) | independent — a tile column moves whole | folded into `out_tile` | — | `tensor_head_dim_tiles = D/32`, host | via `out_tile` | knob-turn; D % 32 ≠ 0 is a **scheme-change** (`subtile_assemble` regime) |

Why no axis carries its own extent knob beyond the flat one: no buffer, transfer or fixed cost scales with any of b, r, h, c separately — each tile is one NoC page read and one NoC page write wherever it sits in the rectangle. The one extent that buys anything is how much a core owns (launch and arg setup are paid once per core), and that is `block_tiles = core_tiles`. The `subtile_assemble` regime is where an N-D block becomes real (a block must stay within one token row there); its row says how.

Re-run on the stage's own data: the op has one stage (relay a tile through L1). The ring holds tiles in output order and has no axis of its own. There is no designated collecting core, no combine stage and no stage that runs on one core while others wait.

### Buffer-depth knobs

| CB | Depth knob | Phase 0 value | What the depth buys |
|----|------------|---------------|---------------------|
| `cb_relay_tiles` | `RING_DEPTH_TILES` (host constant; `num_entries = RING_DEPTH_TILES × max(R, W)`) | 8 | Reads in flight: with implicit sync the reader issues the next tagged read without waiting, bounded only by free ring entries, and the writer drains on the other NoC. Catalog `double_buffer`: ~4–8 reads per batch and a buffer of 2× that saturate a core's NoC; 8 = two implicit-sync halves of 4 (`DataflowBuffer.md:111`). |

### Mechanism caps

| Mechanism | Cap on which extent | Clamp | What happens unclamped |
|-----------|--------------------|-------|------------------------|
| DFB ring size (`quasar.md` cluster model: ≤ 1,048,560 B per ring) | `RING_DEPTH_TILES × page_bytes` | host assert `RING_DEPTH_TILES × page_bytes ≤ min(1_048_560, ttnn.get_max_worker_l1_unreserved_size())`; 8 × 4096 = 32 KiB at float32 | program build fails |
| Implicit-sync batch divisibility (`DataflowBuffer.md:410-425`: `entry_threshold = num_entries / num_txn_ids` must divide by producers / consumers and TCs) | `RING_DEPTH_TILES` | `RING_DEPTH_TILES` a multiple of 4 (covers 1, 2 or 4 txn ids at R = W = 1); host asserts it | program build refuses the descriptor |
| One implicit-sync call = one NoC transaction = one ring entry (`DataflowBuffer.md:130`) | entry size | `entry_size = input.buffer_page_size()` (one tile page); host asserts `input.buffer_page_size() == output.buffer_page_size()` | a partial tile moved per entry → wrong results |
| Page index width (`uint32_t` page ids in `TensorAccessor`) | `tensor_out_tiles` | host asserts `tensor_out_tiles < 2**32` (the models' largest call, [1, 32, 4096, 128], is 16,384 tiles) | wrapped source page id → wrong tile |
| Ragged per-core range | `block_tiles` on the last core group | none needed: the ring is FIFO, each entry is one tile, a core's count may stop anywhere (target page: "tiles dealt round-robin may stop anywhere"); firmware credits the last partial batch at kernel exit (`DataflowBuffer.md:134`) | — |

### Regimes

| Regime | Status | Predicate | Block | Data movement vs. minimum | What a bigger block buys |
|--------|--------|-----------|-------|---------------------------|--------------------------|
| `tile_relay` | **built** | `D % 32 == 0` (covers tile_aligned, h_non_aligned, single and batched, both dtypes, DRAM/L1 in and out). Phase 0's `validate()` narrows it to PHASE0. | `block_tiles = core_tiles` contiguous output pages | Minimum at the DRAM boundary: each input tile crosses DRAM once (read), each output tile once (write). No cross-core traffic. L1: each tile written into the ring once by its NoC read and read once by its NoC write. | Kernel launch, runtime-arg load, the one division that seeds the (b, r, h, c) counters, and the end-of-kernel drain and tag reset — once per core. Per-tile cost is one NoC read + one NoC write and does not amortize (interleaved pages are in different banks; no coalescing across pages exists). The ISR credit batch is `RING_DEPTH_TILES / num_txn_ids` tiles, set by depth, not block. |
| `subtile_assemble` | **deferred** — outside PHASE0 (`alignment = w_non_aligned`); built by the refinement that widens to it | `D % 32 != 0` | a run of output tiles within **one** token row: `block_tiles = min(core_tiles_in_row, …)`; `num_blocks_this_core` = number of token rows the core's range touches | Above minimum by at most one input tile per block boundary: an input tile whose columns straddle two output tiles is read once per block it feeds; inside a block a sliding window holds it. Input padding columns (`D..32·ceil(D/32)` of each head) are never copied; output padding columns beyond `H·D` are zero-filled. | Same fixed costs, plus fewer duplicated boundary reads (overlapping footprints, `blocking-model.md` §4): bigger blocks re-read fewer straddling tiles. |
| `row_major_round_trip` (untilize → permute → tilize, tt-metal's fallback for D % 32 ≠ 0) | **rejected** — superseded by `subtile_assemble` everywhere; a dead end | — | — | Each tensor crosses DEST twice (unpack/pack): a float32 value passes through the unpacker's format path, against R1; as separate ops it is also ≥ 2 dispatches, against the one-dispatch rule | — |
| `head_split` (unit = one input image (b, h); reader reads it contiguously, writer scatters) | **rejected** — superseded by `tile_relay`'s flat split | — | `Dt·St` tiles | Same bytes as `tile_relay`, but only `B·H` units: 4 clusters of 32 for the traced `[1, 4, 32, 64]` | — |
| `token_row_split` (unit = one token tile row (b, r), GUIDANCE's unit) | **rejected** for `tile_relay` — superseded by the flat split; the row boundary returns as the *block* boundary inside `subtile_assemble` | — | `H·Dt` tiles | Same bytes, `B·St` units: 1 cluster of 32 for both traced calls (catalog `width_split`: a one-row-tall tensor split by rows runs at single-core speed) | — |
| `sharded_in_place` (input shard consumed through a DFB borrowed on the shard, no NoC re-read) | **rejected** — outside TARGET: `sharding` is `["interleaved"]` only and R6 refuses a sharded input by name | — | the shard | would remove the DRAM read when the input is already in L1 | — |

Regime selection: `tile_relay` iff `input.shape[-1] % 32 == 0`; else `subtile_assemble` (refused with `NotImplementedError` until built). A regime-pinned test pair is required when `subtile_assemble` lands (one `w_non_aligned` cell and one aligned cell, asserting which program `name` was dispatched). The predicate depends on the shape only, never on the grid.

### Traffic ranking

| Candidate split | DRAM crossings (input / output) | Cross-core | Units for occupancy | Verdict |
|-----------------|-------------------------------|-----------|---------------------|---------|
| Flat output pages (`tile_relay`) | 1 / 1 | none | `B·St·H·Dt` tiles: 8 for `[1,4,32,64]`, 16 for `[1,4,32,128]`, 256 for `[1,32,128,64]` | **chosen** — minimum traffic, finest balance (± 1 tile) |
| Input images (`head_split`) | 1 / 1 | none | `B·H` | rejected: same bytes, under-fills the grid |
| Token rows (`token_row_split`) | 1 / 1 | none | `B·St` | rejected: same bytes, 1 cluster for the traced calls |
| Dependent axis + combine | — | — | — | not applicable: the op has no dependent axis |
| Broadcast of a reuse-shared operand | — | — | — | not applicable: one operand, each tile read by exactly one core (operand-reuse check: the input varies along `out_tile`, the split axis) |

All splits move the same bytes, so the split ranking ties and occupancy decides; transfer shape is identical (one whole tile page per NoC transaction — catalog `tile_reorder`: move whole pages, never faces). The cheapest-traffic split is the implemented one.

Stall-shadow check: the writer waits for the reader's tiles and the reader waits for free ring entries; the op has one stage and no work that is independent of the wait. Nothing to reorder.

### Block schedule

```cpp
// per core: [start_page, start_page + core_tiles) of the output; Phase 0: num_blocks_this_core = 1
for (uint32_t block_idx = 0; block_idx < num_blocks_this_core; ++block_idx) {
    load_block(block_idx);   // reader
    store_block(block_idx);  // writer
}
```

| Operation | Kernel | Block shape | Resident across it | Intended frequency of fixed costs |
|-----------|--------|-------------|--------------------|-----------------------------------|
| `load_block` | reader | `block_tiles` output pages, in output order | nothing: a streaming window of `RING_DEPTH_TILES` | counters (c, h, r, b) seeded by division **once per kernel**, then advanced by increments per tile (no per-tile division); one tagged implicit-sync read per tile (`noc.async_read<NocOptions::TXN_ID>(acc_in, cb_relay_tiles, {.page_id = src}, {})`); **no barrier, reserve or push** in the kernel; ring credits posted by the DM0 ISR once per `RING_DEPTH_TILES / num_txn_ids` tiles; tag handed back (`noc_async_read_set_trid(0, noc.get_noc_id())`) **once**, as the last statement |
| `store_block` | writer | `block_tiles` output pages, contiguous `start_page …` | nothing | one tagged implicit-sync write per tile (`noc.async_write<NocOptions::TXN_ID>(cb_relay_tiles, acc_out, {}, {.page_id = p})`); no barrier/wait/pop; firmware waits for the writes at exit; tag handed back once (`noc_async_write_set_trid(0, noc.get_noc_id())`) |

Source page of output page `p` (both regimes share the walk): `c = p mod Dt; h = (p / Dt) mod H; row = p / (Dt·H)` (`row = b·St + r`); `src = ((b·H + h)·St + r)·Dt + c`. Walking `p` upward advances `c`, wrapping into `h`, then `r`, then `b`.

### Perf lamps

| Lamp | Why the default may be wrong here | Nearby alternative to measure |
|------|-----------------------------------|-------------------------------|
| Grid synchronization | Full occupancy leaves 1 tile per cluster for the traced calls (8 and 16 clusters at 1 tile each) and 1–8 tiles for every PHASE0 shape: launch and arg setup can dominate. | `min_tiles_per_core` (host constant, Phase 0 = 1: engage `min(grid, ceil(tensor_out_tiles / min_tiles_per_core))` clusters) at 4 and 8. |
| Thread plan | (1, 0, 1) assumes one hart per direction is enough. | (2, 0, 2) — `THREAD_PLAN` constant; two readers and two writers on the STRIDED ring, `num_entries = RING_DEPTH_TILES × 2`. Measure on `ttsim-perf`. |
| Ring depth | 8 is the catalog sweet spot on Wormhole/Blackhole. | `RING_DEPTH_TILES` = 4 and 16. |
| Bank order | Every cluster starts its walk at its own contiguous range; consecutive pages alternate DRAM banks, so cores may collide on one bank at the same moment (catalog `bank_stagger`). | Rotate each cluster's start within its range by `core_index mod num_dram_channels` (writer and reader rotate identically). |

Overlap lamp: not attached — the block is a streaming window, never held whole, so a large block cannot serialize read against write.

## Dataflow Strategy

| Stage | Format | Mechanism | Notes |
|-------|--------|-----------|-------|
| Input DRAM/L1 → ring | TILE pages of the input dtype, unchanged | reader DM kernel, NoC0 (`create_reader_datamovement_config`), `TensorAccessor(tensor::input)`, implicit-sync tagged reads, one page per entry | source page from the permutation walk |
| ring → output DRAM/L1 | same bytes | writer DM kernel, NoC1 (`create_writer_datamovement_config`), `TensorAccessor(tensor::output)`, implicit-sync tagged writes | contiguous output pages per core |
| compute | — | **none** | no element passes through DEST, so R1 holds for float32 bit-for-bit with no precision configuration; the ring's `data_format_metadata` is the input dtype only as a label |

Placement and TARGET axes, as the regime rows see them:

| TARGET axis value | Phase 0 | Cost to unlock | Why |
|-------------------|---------|----------------|-----|
| `dtype` float32 | no | knob-turn | entry size and format come from the tensor; DM copy is bit-exact |
| `memory_config` L1 (input) | no | knob-turn | `TensorAccessor` reads an L1 interleaved tensor the same way |
| `output_memory` None / L1 | no | knob-turn | allocation only (`TensorSpec(..., memory_config=out_mc)`); None → `MemoryConfig(INTERLEAVED, input buffer type)` |
| `alignment` h_non_aligned | no | knob-turn | `St = ceil(S/32)` per image already; padding rows travel with their tiles |
| `batch` batched | no | knob-turn | `b` is in the walk from day 1 |
| `alignment` w_non_aligned | no | scheme-change | `subtile_assemble`: an assembly stage between reader and writer |
| `sharding` | interleaved only | — | TARGET has no sharded value |

`subtile_assemble` contract (deferred, so the structure stays reachable): same host split over output pages, cut into blocks at token-row boundaries; reader → `cb_head_tiles` (implicit, input tiles in (h, c) order for the block's columns) → **assembler** DM kernel (explicit sync on both rings, listed in `disable_dfb_implicit_sync_for`; copies face-row element runs with RISC-V stores — `ttnn/cpp/ttnn/kernel_lib/tile_column_dataflow.hpp:11-24` documents the face layout and that a DM core's stores sit in its data cache and must be flushed before the NoC reads the tile; issues no NoC transfer) → `cb_output_tiles` → writer (implicit, unchanged). No inter-core communication in any regime.

Structural impossibilities: none beyond the specification's `INVALID = []`.

## Work Distribution

| Field | Value |
|-------|-------|
| Work unit | one output tile page; a core's block is its contiguous range |
| Grid | `device.compute_with_storage_grid_size()` (8 × 4 on quasar), passed to the split; `min_tiles_per_core` caps engaged cores (Phase 0 = 1) |
| Per-core work | `ttnn.split_work_to_cores(grid, tensor_out_tiles, row_wise=True)` (`ttnn/cpp/ttnn-nanobind/operations/core.cpp:471-486`; `row_wise=True` per catalog `noc_placement`); cores ordered row-major, `start_page` = running sum. Worked (quasar): `[1,4,32,64]` → 8 tiles → 8 clusters × 1; `[1,4,32,128]` → 16 × 1; `[1,8,64,64]` → 32 × 1; `[1,32,128,64]` → 256 → 32 × 8; `[1,64,32,128]` → 256 → 32 × 8; `[1,32,4096,128]` (model) → 16,384 → 32 × 512 |
| Remainder | the split's two groups differ by one tile; geometry is per image and `ceil`: `tensor_out_tiles = B · ceil(S/32) · H · (D/32)` in `tile_relay` (`B · ceil(S/32) · ceil(H·D/32)` in `subtile_assemble`) |
| Bound | **data movement** — zero compute per tile against one 2–4 KiB DRAM read and one write per tile |
| Thread plan | `THREAD_PLAN = (R, C, W) = (1, 0, 1)`: one reader hart, no compute kernel, one writer hart. Bound by data movement; C = 0 because there is nothing to compute. The host passes R and W to the kernels as compile-time args; each kernel strides its tile loop by its own count from `get_my_thread_id()` so the (2, 0, 2) lamp is a constant edit. Rule R \| W or W \| R holds for STRIDED DM→DM (`DataflowBuffer.md:250-255`). |
| Unit dealt to compute threads | n/a (no compute kernel). Between DM threads the unit is the ring entry (one tile), so counts may stop anywhere and no round padding exists. |
| Tests the plan needs | (1) every hart ran: with R = W = 1 the full-output check covers it; when the lamp raises R or W, a test with `tensor_out_tiles` not divisible by the thread count checks every page. (2) One device session: the op after another program (e.g. a `ttnn.add`, or the op itself twice) and before `ttnn.to_layout(output, ttnn.ROW_MAJOR_LAYOUT)`; both must complete and the untilized result match. (3) Engaged-core count: the host split helper returns `min(grid_cores, tensor_out_tiles)` cores for a shape in each split group (e.g. 8 for `[1,4,32,64]`, 32 for `[1,32,128,64]`). |

## Circular Buffers

| Semantic Name | Index | Page Size | Num Pages | Sizing rationale | Format | Producer | Consumer | Sync | Lifetime |
|---------------|-------|-----------|-----------|------------------|--------|----------|----------|------|-----------|
| `cb_relay_tiles` | DFB unique_id `"relay"` | `input.buffer_page_size()` (2048 B bf16, 4096 B fp32) | `RING_DEPTH_TILES × max(R, W)` = 8 | streams over `out_tile` (window of `RING_DEPTH_TILES` reads in flight); spans no block axis | input dtype (label only; bytes are never unpacked) | reader | writer | implicit on both sides (tagged NoC fill and drain; no `disable_dfb_implicit_sync_for`) | whole program |

STRIDED access pattern on both endpoints (the default). One ring is the whole inventory: the tile goes DRAM → L1 → DRAM and L1 staging is unavoidable for a NoC read.

## Block Operation Realization

| # | Block operation | Block shape | Helper? | Input CB (semantic name, pages, state) | Output CB (semantic name, pages) | CB state after |
|---|-----------------|-------------|---------|----------------------------------------|----------------------------------|----------------|
| 1 | `load_block` | `block_tiles` pages (Phase 0 = `core_tiles`) | no (raw `Noc` API; no dataflow helper wraps an implicit-sync page read) | input tensor via `TensorAccessor` | `cb_relay_tiles`, one entry per tile, ISR-posted | every tile posted; firmware posts the last partial batch at exit |
| 2 | `store_block` | `block_tiles` pages | no (raw `Noc` API) | `cb_relay_tiles`, one entry per tile | output tensor via `TensorAccessor` | ring empty at exit; writes landed (firmware waits) |

## API Mapping

| Block operation | Type | Function | File:Line | Template Params / Args | Input CB | Output CB | Which params are block knobs |
|-----------------|------|----------|-----------|------------------------|----------|-----------|------------------------------|
| `load_block` | raw_api | `Noc::async_read<NocOptions::TXN_ID>(src_accessor, dfb, {.page_id}, {})` (implicit-sync overload) | `tt_metal/hw/inc/api/dataflow/noc.h:915-922` | `NocOptions::TXN_ID`; page id from the walk | input tensor | `cb_relay_tiles` | none (one entry per call); loop bound `core_tiles` (RT), stride R (CT) |
| `load_block` (exit) | raw_api | `noc_async_read_set_trid(0, noc.get_noc_id())` | `tt_metal/hw/inc/api/dataflow/dataflow_api.h:2558` | — | — | — | — |
| `store_block` | raw_api | `Noc::async_write<NocOptions::TXN_ID>(dfb, dst_accessor, {}, {.page_id})` | `tt_metal/hw/inc/api/dataflow/noc.h:930-937` | `NocOptions::TXN_ID`; `page_id = start_page + i` | `cb_relay_tiles` | output tensor | loop bound `core_tiles` (RT), stride W (CT) |
| `store_block` (exit) | raw_api | `noc_async_write_set_trid(0, noc.get_noc_id())` | `tt_metal/hw/inc/api/dataflow/dataflow_api.h:2579` | — | — | — | — |
| host: ring | host | `ttnn.experimental.DataflowBufferSpec(unique_id="relay", entry_size=page_bytes, num_entries=RING_DEPTH_TILES*max(R,W), data_format_metadata=input.dtype)` | `ttnn/cpp/ttnn-nanobind/program_spec.cpp:302` | — | — | — | `RING_DEPTH_TILES` |
| host: kernels | host | `KernelSpec(num_threads=R / W, hw_config=create_reader_datamovement_config() / create_writer_datamovement_config(), dfb_bindings=[producer_of("relay", …)] / [consumer_of("relay", …)])` | `ttnn/cpp/ttnn-nanobind/program_spec.cpp:470-477, 798-811` | — | — | — | `THREAD_PLAN` |
| host: dispatch | host | `ttnn.generic_op([input, output], spec, run_args, {"input": 0, "output": 1})`; `WorkUnitSpec(kernels=["reader", "writer"], target_nodes=core_set)` | `ttnn/cpp/ttnn/operations/generic/generic_op_nanobind.cpp:83` | — | — | — | — |

Helpers considered and rejected (all phases): the op has **no compute phase**, so no compute helper applies. `compute_kernel_lib` `copy` (`ttnn/cpp/ttnn/kernel_lib/eltwise/api/convenience.hpp:113`) would add a compute kernel whose only effect is to unpack each tile into DEST and pack it back: a second L1 hop per tile and, for float32, a pass through the unpacker's format path that R1 forbids an element to take; the DM relay is bit-exact by construction. The dataflow helpers (`local_copy_helpers_dataflow.hpp:8-34`: L1 → L1 self-aimed reads; `tile_column_dataflow.hpp:91-92`: column copies) do not express a DRAM page → ring transfer; `tile_column_dataflow.hpp` is the building block named for `subtile_assemble`.

## Key Risks and Gotchas

| Risk | Why it bites here | Mitigation in this design |
|------|-------------------|---------------------------|
| Sticky NoC tag after implicit sync | The golden suite runs one process per cell and cannot see it. | Both kernels end with the trid reset; the one-session test (after another program, before `ttnn.to_layout`) is required. |
| DM → DM implicit sync on both ring ends | tt-metal's examples run DM→DM rings with explicit sync (`toy_spec_single_tensor`), and the implicit-sync 1×1 DM→DM config is covered by `tests/tt_metal/tt_metal/api/dataflow_buffer/test_dataflow_buffer_configs.cpp:460-474` (serialization) | Phase 0 uses implicit sync on both ends (the target page's default). If the simulator rejects or hangs it, fall back to explicit sync on the ring end that fails: list the ring in that kernel's `disable_dfb_implicit_sync_for`, `reserve_back/push_back` or `wait_front/pop_front` per tile, and every NoC transfer of that kernel tagged `{.trid = 0}` with the barrier on the same id. |
| Wrong source page | The walk crosses four nested wraps (c → h → r → b); a counter advanced in the wrong order produces a plausible-looking permutation | Seed by division once, then increment; `[1,8,64,64]` (two rows, 8 heads) and `[1,32,128,64]` (four rows, ranges cut mid-row) exercise every wrap; batched cells exercise `b` later. |
| Validate wider than PHASE0 | Supporting `output_memory=None` or L1 now costs nothing in the kernels, but `SUPPORTED` must equal PHASE0 and outside cells are `xfail(strict)`: an accepted cell XPASSes and fails the suite | `validate()` gates exactly on PHASE0 (`output_memory` "DRAM" only, `memory_config` "DRAM", bfloat16, tile_aligned, single); widening is the refinements' job. |
| Ring under-filled cores | A core with 1 tile leaves its implicit-sync batch (4 tiles) partial | Firmware credits the last partial batch at kernel exit (`DataflowBuffer.md:134, 571`); no kernel-side flush. |
| `memory_config` read by identity | Callers build `ttnn.MemoryConfig(INTERLEAVED, DRAM)` afresh (R4) | Read it with the spec's `canonical_memory` (`spec.py:157-168`); a non-config raises `TypeError` (X2); the input's axes come from `eval.memory.memory_axes(input_tensor)` (R6). |
