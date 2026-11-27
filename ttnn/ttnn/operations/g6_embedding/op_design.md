# Operation Design: g6_embedding

## Overview

| Field | Value |
|-------|-------|
| Classification | data_movement (index-driven gather + tilize; no arithmetic) |
| Goal | One device program per call that reads the uint32 index on device, reads from the bfloat16 table only the rows the indices name, and writes them as a TILE bfloat16 result, bit for bit. |
| Math | `out[p, d] = W[idx[p], d]` for every logical position `p` of `idx` and every `d ∈ [0, D)`; a `[1, V, D]` weight is `W = weight[0]`, read in place (its pages are the `V` rows). |
| References | `eval/golden_tests/g6_embedding/feature_spec.py` (TARGET, INPUTS, LOOSE_CASES), `eval/golden_tests/g6_embedding/helpers.py` (oracle `F.embedding`, `TOLERANCES = ("exact", None)`), `.claude/references/blocking-model.md`, `.claude/references/l1-footprint-discipline.md`, `.claude/references/noc-alignment-in-tt-metal.md`, `ttnn/cpp/ttnn/kernel_lib/tilize_helpers.hpp`, catalog `ttnn/ttnn/operations/examples/master.md` (entries: `noc_placement`, `width_split`, `double_buffer`, `trid_double_issue`, `tile_reorder`, `split_reader`, `bank_stagger`) |

## Parameters

| Name | Type | Required | Valid Range | Default | CT/RT |
|------|------|----------|-------------|---------|-------|
| `input_tensor` (idx) | `ttnn.Tensor` | yes, positional | uint32, rank 1 `[N]` or rank 2 `[B, S]`, on device, DRAM interleaved; layout ROW_MAJOR (Phase 0), TILE (refinement) | — | shape-derived values CT; buffer address common RT |
| `weight` (W) | `ttnn.Tensor` | yes, positional | bfloat16, ROW_MAJOR, `[V, D]` or `[1, V, D]`, `D % 32 == 0`, on device, DRAM interleaved | — | shape-derived values CT; buffer address common RT |
| `layout` | `ttnn.Layout` | no, keyword | `ttnn.TILE_LAYOUT` (axis `output_layout`) | `ttnn.TILE_LAYOUT` | host only |
| `memory_config` | `ttnn.MemoryConfig` | no, keyword | `None` or DRAM interleaved | `None` → `ttnn.DRAM_MEMORY_CONFIG` | host only (output placement) |
| `runtime_rows` | `bool` | no, keyword | `False` (axis `runtime_rows`; `True` → `UnsupportedAxisValue`) | `False` | host only |

No `compute_kernel_config`: the op does no arithmetic. The compute kernel's config is fixed (see Key Risks: `fp32_dest_acc_en = False`, 16-bit DEST, UInt16 data path).

## Tensors

### Input — index (`input_tensor`)

| Property | Requirement |
|----------|-------------|
| Shape | `[B, S]` or `[N]` (rank-1 is one image: `B_img = 1`, `S_img = N`) |
| Dtype | `ttnn.uint32`; every value in `[0, V)` by contract (never checked, never read on host) |
| Layout | ROW_MAJOR (Phase 0): page `b` = row `b`, `4·S_img` bytes. TILE (refinement R2): uint32 tiles, positions in face order, padding positions arbitrary and never used |
| Memory | DRAM interleaved |

### Input — table (`weight`)

| Property | Requirement |
|----------|-------------|
| Shape | `[V, D]` or `[1, V, D]`; `V ≥ 1`; `D % 32 == 0` |
| Dtype | `ttnn.bfloat16` (axis `dtype`) |
| Layout | ROW_MAJOR: page `v` = table row `v`, `2·D` bytes (a multiple of 64, so `aligned_page_size == page_size` on Blackhole). A `[1, V, D]` tensor has the same `V` pages; it is addressed as is, never reshaped |
| Memory | DRAM interleaved |

### Output

| Property | Value |
|----------|-------|
| Shape | `idx.shape + [D]`: `[B, S, D]` or `[N, D]`; padded shape `[B, 32·ceil(S/32), D]` / `[32·ceil(N/32), D]` |
| Dtype | `weight.dtype` (bfloat16) |
| Layout | TILE (32×32, 2048-byte pages) |
| Memory | `memory_config`, `None` → DRAM interleaved; tile id of (tile row `tr`, tile column `w`) = `tr·Dt + w` |

## Support Contract (registry model)

| Item | Decision |
|------|----------|
| `INPUT_TAGGERS` | `alignment`: `inputs[0]["index_shape"][-1] % 32 == 0` → `"tile_aligned"`, else `"h_non_aligned"`. `layout`: `inputs[0]["index_layout"]` `"ROW_MAJOR"` → `ttnn.ROW_MAJOR_LAYOUT`, `"TILE"` → `ttnn.TILE_LAYOUT` |
| Scenario built by `validate()` | `({"index_shape": tuple(input_tensor.shape), "weight_shape": tuple(weight.shape), "index_layout": "ROW_MAJOR"/"TILE", "memory_config": "dram_interleaved" if memory_config is a DRAM-interleaved config else None},)` |
| Free axes derived in `validate()` | `dtype = weight.dtype`, `output_layout = layout`, `runtime_rows = bool(runtime_rows)` |
| `SUPPORTED` (Phase 0) | `dtype: [bfloat16]`, `layout: [ROW_MAJOR]`, `alignment: ["tile_aligned", "h_non_aligned"]`, `output_layout: [TILE]`, `runtime_rows: [False]` |
| `EXCLUSIONS` | `[]` — no size-tagged cell, ever (every golden shape fits; see L1 ledger) |
| `validate()` order | (1) input requirements → `ValueError`: idx rank ∉ {1, 2} or dtype ≠ uint32; weight rank ∉ {2, 3}, rank 3 with leading extent ≠ 1, layout ≠ ROW_MAJOR; `D % 32 ≠ 0`; a tensor not on device or not DRAM interleaved; `memory_config` not DRAM interleaved; per-core L1 footprint (ledger expression) > 1,347,776 B → `ValueError` naming the BB-05 bound. (2) SUPPORTED per axis → `UnsupportedAxisValue` whose message names the axis and value (e.g. `g6_embedding: runtime_rows=True not in SUPPORTED [False]`). (3) EXCLUSIONS → `ExcludedCell`. `validate()` is the entry point's first line |
| TARGET − SUPPORTED | `layout = TILE` (index): **knob-turn** — regime R2, same blocking, only `fill_index_window` changes |
| Placement axis | TARGET has no `memory_layout` axis; sharded inputs/outputs are outside the contract (see Dataflow Strategy, last row) |
| Structural impossibilities | none beyond `INVALID = []` |

## Blocking Model

Semantics: `.claude/references/blocking-model.md`. Scope names follow its §3.

**Derived extents (host, once per program):**
`S_img` = last index dim; `B_img` = `B` (1 for rank-1); `tensor_tile_rows_per_image = ceil(S_img / 32)`; `tensor_tile_rows T = B_img · tensor_tile_rows_per_image` (per-image ceil — tile rows never straddle images); `tensor_width_tiles Dt = D / 32`; `width_blocks_per_row = Dt / block_width_tiles`; `total_units = T · width_blocks_per_row`.

**Work unit** = (tile row `tr`, width block `wb`) = 32 token positions × `block_width_tiles` tile columns, numbered `u = tr · width_blocks_per_row + wb` (tile-row-major, width blocks inner). For `tr = b·tensor_tile_rows_per_image + h`: `rows_valid(tr) = min(32, S_img − 32·h)`.

### Axes

| Axis | Character (+ one-clause reason) | Extent knob | Phase 0 value | Knob source | Core-assignment | Later unlock |
|------|--------------------------------|-------------|---------------|-------------|-----------------|--------------|
| Token tile rows `tr ∈ [0, T)` (index axes `b`, `h` folded: `tr = b·Ht + h`) | **independent** — each output tile row is a copy of the rows its own 32 indices name; no result spans tile rows | `block_tile_rows` (units per block) | **1** — (a) no operand is re-read across tile rows, so bytes moved are constant in this extent and it buys only fixed per-block cost: one read barrier, one `cb_gathered_rows` reserve/push, one `cb_output_tiles` wait/pop, one writer flush — each against 32 row reads and `block_width_tiles` whole-tile writes; the tilize LLK runs once per tile row whatever the block (`tilize_helpers.inl:214-239`); (b) overlap lamp: per-core assignments are ≤ 19 units on every production shape, and a `k`-unit block delays each core's first write by `k` units of reads; (c) L1: at `block_width_tiles = 40` one unit is 80 KiB per slot, the extent multiplies all four slots, and the whole assignment (19 units, 1.5 MiB per buffer) cannot be resident. Chip-wide, extent 1 keeps `cores × 32` row reads in flight (3,520 at 110 cores) | host constant `BLOCK_TILE_ROWS` → reader CT arg; CB capacities derive from it | contiguous unit ranges: `ttnn.split_work_to_cores(grid, total_units, row_wise=True)`; spread across the full grid from Phase 0 | knob-turn (raise `BLOCK_TILE_ROWS`; clamp in Mechanism caps) |
| Token position in a tile row `r ∈ [0, 32)` | **independent** — each position is its own row read | tile height (fixed 32); runtime extent `rows_valid(tr)` | 32 (`rows_valid` on the last tile row of each image) | hardware tile shape | inside a unit; not assigned | none (mechanism-fixed) |
| Table row length, tile columns `w ∈ [0, Dt)` | **independent** — output column `w` is column `w` of the gathered rows; no result spans columns | `block_width_tiles` | `select_block_width(T, Dt, C)` (Work Distribution); production values 1–40 | host function `select_block_width` → CT arg of reader, compute (tilize template param), writer | width blocks interleaved into the unit order: `width_blocks_per_row` consecutive units share a tile row and land on consecutive cores | knob-turn (it is the selection function's output) |
| Table rows `v ∈ [0, V)` | **independent, gather-addressed** — never iterated: a block touches exactly the rows its indices name; reuse between blocks happens only through repeated index values (data-dependent, none by construction) | none — a block touches ≤ `32 · block_tile_rows` table rows, derived from the token extent | — | — | not assigned: every core reads the whole table address space through the interleaved accessor, one named row slice at a time | none (a whole-table pass is forbidden by the contract) |

**Operand-reuse check (primary split = token tile rows).** Index: varies along `tr` → each core reads only its own slice. Table: the bytes a unit reads vary with its indices → no structural reuse. **Index under the secondary D split:** does not vary along `w` → reuse-shared *by construction of that split* → broadcast regime R3.

**Intermediates re-run.** `cb_index_window` (tile-row axis), `cb_gathered_rows` (32 positions × `block_width_tiles` columns per unit), `cb_output_tiles` (`block_width_tiles` tiles per unit) carry the same independent axes and stay on the core that owns the unit. No core collects, coordinates or hands out work; every core runs fill → gather → tilize → store on its own units; no stage computes on one core while others wait.

**Stall-shadow check.**

| Wait | What could run in its shadow | Decision |
|------|------------------------------|----------|
| Reader: per-block read barrier | the next block's gather — units are independent copies, so issuing block `k+1` before retiring block `k` is legal; no floating-point order exists to disturb | Phase 0 leaves it unfilled (depth-2 CBs overlap it with compute and writer); filling it is perf lamp LP3 (trid ring) |
| Reader: `fill_index_window` barrier | nothing — every row read needs an index value | none to schedule; compute and writer have no work before the first block |
| Compute: input wait | nothing — tilize has no other input | none |
| Writer: flush before pop | the next unit's tile writes (already issued after the flush) | flush once per block, write barrier once per kernel |

### Buffer-depth knobs

| CB | Depth knob | Phase 0 value | What the depth buys |
|----|------------|---------------|---------------------|
| `cb_index_window` | `index_window_tile_rows = min(INDEX_WINDOW_MAX_TILE_ROWS = 64, max core tile-row span)` | ≤ 19 on every production shape (one fill per core) | the indices of every tile row the core touches are resident before its first row read, so no DRAM index read sits inside the block loop; the cap bounds L1 for unbounded token counts (refill per window) |
| `cb_gathered_rows` | `GATHER_DEPTH_BLOCKS` | 2 | reader gathers block `k+1` while compute tilizes block `k` |
| `cb_output_tiles` | `OUTPUT_DEPTH_BLOCKS` | 2 | compute tilizes block `k+1` while writer stores block `k` |

### Mechanism caps

| Mechanism | Cap on which extent | Clamp | What happens unclamped |
|-----------|--------------------|-------|------------------------|
| Tilize helper: `block_width_tiles` is a template parameter (`tilize_helpers.hpp:194-207`); helper asserts input/output CB capacity ≥ `block_width_tiles` (`tilize_helpers.inl:197-203`) | `block_width_tiles` | CT arg; CB capacities `= depth · block_tile_rows · block_width_tiles` ≥ `block_width_tiles` | assert under `--dev`; CB overrun otherwise |
| Tilize unit = exactly 32 rows per tile row | token extent of a unit | every unit pushes the nominal `block_width_tiles` pages; positions `r ≥ rows_valid` are not written (stale L1 = padding rows, unspecified by contract) | n/a |
| Bit-exact copy: CB data format | format of `cb_gathered_rows`, `cb_output_tiles` | both `ttnn.uint16` (DataFormat::UInt16): standard tilize path, integer unpack/datacopy/pack, 16-bit DEST. Float16_b would select fast tilize (`tilize_helpers.inl:30-35, 65-69`) and the float route, which flushes subnormals and rewrites NaN (`data_movement_test_utils.py:27-35` leaves both out of bf16 checks; UINT16 is checked over its full range) | NaN / subnormal / sign-of-zero rewrites — invisible on random data |
| Blackhole DRAM read alignment: `local_addr ≡ noc_addr (mod 64)` (`noc-alignment-in-tt-metal.md` §1) | row-slice offset, row-slice size, stick stride, index offsets | row-slice offset `wb·block_width_tiles·64` and stick stride `block_width_tiles·64` are multiples of 64; CB bases are allocated at DRAM alignment (`tt_metal/impl/program/program.cpp:2505-2509`); index offsets `128·h` and window slots `128·j` are multiples of 64 | silent data corruption (watcher stop under `--dev`) |
| L1 budget (BB-05, 1,347,776 B) | `block_width_tiles` | `block_width_tiles` = a divisor of `Dt` ≤ `MAX_BLOCK_WIDTH_TILES = 80` (4 slots × 80 × 2048 B = 640 KiB) | footprint past the bound for D > 2560 |
| Index window capacity | tile rows resident in `cb_index_window` | `INDEX_WINDOW_MAX_TILE_ROWS = 64` (8 KiB); refill when a block's tile row leaves the window | unbounded L1 for unbounded `B·S` |
| Block rectangle | `block_tile_rows` | `block_tile_rows > 1` only when `width_blocks_per_row == 1` (else clamp to 1) | a block that is a raster run, not a rectangle (still correct) |
| CB ring wrap (`ttnn-cb-memory-fundamentals.md`, Ring-Wrap) | push/pop quanta | every push and pop is `block_width_tiles` pages (one unit); capacity is a multiple; a short block exists only as a core's final block | pointer overrun into neighbouring L1 |
| Index value domain | table page id | none: contract guarantees `idx ∈ [0, V)`; no host or device check | unspecified result (contract) |

### Regimes

| Regime | Status | Predicate | Block | Data movement vs. minimum | What a bigger block buys |
|--------|--------|-----------|-------|---------------------------|--------------------------|
| **R1 gather → tilize → store**, units = tile row × width block, cores split units | **built** | `index layout == ROW_MAJOR` | `block_tile_rows` (= 1) units × (32 positions × `block_width_tiles` tiles) | Minimum (named boundary DRAM): index `4·B·S` B once, each named row `2·D` B once per position, each output tile (2048 B, padding rows of partial tile rows included — a TILE page is whole) once. Exactly met when `width_blocks_per_row = 1`. When `> 1`: table bytes unchanged but each row read as `width_blocks_per_row` slices of `block_width_tiles·64` B, and each tile row's index slice read `width_blocks_per_row` times (≤ 3.1 % of the call's DRAM bytes on every production shape, worst `[1, 64]×[248320, 1280]`; ≤ 0.24 % on every call of ≥ 3,844 tokens; ≤ 384 extra reads per call) | per block: one DRAM round trip at the reader barrier, one reserve/push and one wait/pop on each streaming CB, one writer flush. Once per kernel: `compute_kernel_hw_startup`, `tilize_init`/`uninit`, final write barrier. Once per window (= once per core on production): index fill |
| **R2 TILE-index fill** | **deferred** — Phase 0 SUPPORTED holds `layout = ROW_MAJOR` only (TILE is 2 of 49 production cases); R1's blocking, CBs, compute and writer are unchanged, only `fill_index_window` reads the tile row's 32 indices as two 64-B face-row runs (faces 0/1 or 2/3 at offset `(b mod 16)·64` of index tile `(b/32, h)`), so R1 keeps it reachable | `index layout == TILE` | as R1 | as R1 (two index transfers per tile row instead of one); positions past the logical shape never read | as R1 |
| **R3 index broadcast per tile-row group** | **deferred** — it does not shorten a receiver's critical path: it replaces one ≤128-B DRAM read issued before the core's first row read with a sender DRAM read → multicast → semaphore chain that still contains that read; the bytes it saves are ≤ 3.1 % of every production call (≤ 0.24 % from 3,844 tokens up), and on every call where that share exceeds 1 % each core owns exactly one unit, so the multicast's rendezvous lands on the only work the receiver has. Reachable: only the source of `fill_index_window` changes; group = the `width_blocks_per_row` cores owning one tile row (consecutive cores, `ttnn.Mcast` ChainUnicast for non-rectangular groups) | `width_blocks_per_row > 1` | as R1 | removes `(width_blocks_per_row − 1)` index-slice reads per tile row; adds one multicast + semaphore per group | as R1 |
| R4 whole index page per core | **rejected** — superseded by R1's per-core slice read | — | — | `+(cores − 1)·4·S` B per page, all on one DRAM bank (RB-010) | — |
| R5 dataflow tilize (row pieces placed into face positions by NoC) | **rejected** — superseded by R1's compute tilize | — | 32-B face-row pieces | `2·block_width_tiles` transfers per row instead of one; direct DRAM→face reads also break 64-B congruence for odd face rows | — |
| R6 Float16_b (fast) tilize | **rejected** — superseded by R1's UInt16 tilize; the float route is not bit-exact | — | — | — | — |
| R7 table pass (convert/tilize W, then gather tiles) | **rejected** — forbidden by contract; superseded by R1 | — | — | `+V·2·D` B (up to 671 MB) per call | — |
| R8 duplicate-index dedupe (read a repeated row once) | **rejected** — superseded by R1: savings depend on index values (zero for distinct indices), and the compare work sits on the reader's issue path | — | — | data-dependent | — |
| Dependent-axis split + cross-core combine | **not applicable** — no axis carries a reduction, scan or any cross-block combine | — | — | — | — |

Single built regime: the predicate is the index layout; R1's width blocking is a parameter of R1, selected by `select_block_width` (pinned in Work Distribution), not a second regime.

### Traffic ranking

Counted per tensor and tier for a call with `B·S` tokens, row bytes `2D`, `T·Dt` output tiles.

| Rank | Candidate split | Index DRAM crossings | Table DRAM bytes / transfer size | Output | Cross-core | Verdict |
|------|-----------------|----------------------|----------------------------------|--------|------------|---------|
| 1 | token tile rows across cores (`width_blocks_per_row = 1`) | once (each core reads only its tile rows' slice) | `B·S·2D` once, full-row transfers of `2D` B | once | 0 | **primary** — the minimum |
| 2 | token tile rows × width blocks (`width_blocks_per_row > 1`) | `width_blocks_per_row` × per tile row | `B·S·2D` once, transfers of `block_width_tiles·64` B | once | 0 | **built as R1's parameter**, used only where the selection function finds the token axis under-fills or unbalances the grid (`width_split` catalog: up to 7.8× on short-wide tensors) |
| 3 | width blocks with index broadcast | once | as rank 2 | once | `T·(width_blocks_per_row − 1)` × ≤128 B multicast + semaphores | R3, deferred |
| 4 | whole index page per core | `cores` × per page | as rank 1 | once | 0 | R4, rejected |
| 5 | table pass | once | `≥ V·2D` (≤ 671 MB) | once | 0 | R7, rejected (contract) |
| — | dependent axis + combine | — | — | — | — | none exists |

The traffic ranking orders the axes (tokens before D); occupancy is the selection function's job and never overrides it — `select_block_width` cuts D only when the token units alone cost more per core.

### Block schedule

```cpp
// Per core: units [core_first_unit, core_first_unit + core_num_units).
fill_index_window(window_idx = 0);                       // reader
for (uint32_t block_idx = 0; block_idx < num_blocks_this_core; ++block_idx) {
    gather_rows_block(block_idx);                        // reader  -> cb_gathered_rows
    tilize_block(block_idx);                             // compute -> cb_output_tiles
    store_tiles_block(block_idx);                        // writer  -> DRAM
}
// fill_index_window(window_idx + 1) runs before a block whose first tile row is past the window
// (never on a production shape: max span 19 < 64).
```

| Block operation | Block shape | Resident across it | Intended frequency of fixed costs |
|-----------------|-------------|--------------------|-----------------------------------|
| `fill_index_window` | tile rows `[window_first_tr, window_first_tr + index_window_tile_rows)` ∩ core span; one NoC read per image segment: bytes `[128·h_first, min(4·S_img, 128·(h_last + 1)))` of index page `b` → window slot `(tr − window_first_tr)·128` | the whole window, for every block that uses it | one read per image segment, one barrier per window (once per core on production) |
| `gather_rows_block` | `block_tile_rows` units; per unit `rows_valid` reads of `block_width_tiles·64` B from table page `idx[r]` at byte offset `wb·block_width_tiles·64` into slot row `r` (stride `block_width_tiles·64`) | `cb_index_window` | one `cb_reserve_back(block_tile_rows·block_width_tiles)`, one read barrier, `block_tile_rows` pushes of `block_width_tiles` per block |
| `tilize_block` | `block_tile_rows` tile rows × `block_width_tiles` tiles, UInt16 | nothing (pure streaming) | `compute_kernel_hw_startup` + `tilize_init` once per kernel (one helper call over `core_num_units` tile rows); per tile row one wait/reserve/push/pop inside the helper |
| `store_tiles_block` | `block_tile_rows · block_width_tiles` whole 2048-B tiles to tile ids `tr·Dt + wb·block_width_tiles + w` | nothing | one `noc_async_writes_flushed` per block before its pops; one `noc_async_write_barrier` per kernel |

### Perf lamps

| Lamp | Why the default may be wrong here | Nearby alternative to measure |
|------|-----------------------------------|-------------------------------|
| LP1 overlap (unit width) | shapes where each core owns 1–2 units (`[1, 3844]`, `[1, 4096]`, every decode call) serialize read → tilize → write per core; a whole-chip tilize bubble sits between the read and write phases of a DRAM-bound call | `UNIT_OVERHEAD_TILES ∈ {1, 4}` (shifts `select_block_width` toward narrower / wider units); measure p4 and `[1, 4096]×[248320, 1280]` |
| LP2 grid synchronization (decode) | `T = 1` calls spread one tile per core over 40–80 cores; launch and per-core fixed cost may dominate one 2-KiB tile | cap the participating cores (e.g. `block_width_tiles ∈ {4, 8}` → 10 / 5 cores) on `[1, 1]`, `[1, 8]`, `[1, 16]` × `[248320, 1280]` |
| LP3 read-ahead (`trid_double_issue`) | the reader's per-block barrier drains its reads; `GATHER_DEPTH_BLOCKS = 2` leaves only compute overlap | trid ring of 2–3 blocks: `set_read_trid` / `async_read_barrier_with_trid` (`local_copy_helpers_dataflow.hpp:147,158`) at equal CB size; the README's landing-slot warning applies |
| LP4 writer bank rotation (`bank_stagger`) | `Dt % 8 == 0` (D = 1280, 2560, 5120) puts every unit's first tile on bank `(wb·block_width_tiles) mod 8`; cores walking in step hit one bank | start each core's tile walk within a unit at `core_index mod 8` and wrap |
| LP5 split reader (`split_reader`) | narrow slices (64–128 B, D = 64 tables and decode width splits) may make the reader RISC issue-bound | issue half of each unit's row reads from the writer RISC on NoC1 into the same slot; only if the reader is measured issue-bound |
| LP6 buffer depth | 2-deep CBs may not cover DRAM latency variance on p1 (80-KiB units) | `GATHER_DEPTH_BLOCKS`, `OUTPUT_DEPTH_BLOCKS ∈ {3, 4}` (p1 footprint 330 KiB → 495 / 660 KiB) |
| LP7 narrow-row block height | D ≤ 128 units carry 4–8 KiB per barrier | `BLOCK_TILE_ROWS ∈ {4, 8}` where `width_blocks_per_row == 1` and `block_width_tiles ≤ 4` (`double_buffer` catalog: ~4–8 transfers per barrier) |

## Dataflow Strategy

| Stage | Format | Mechanism | Notes |
|-------|--------|-----------|-------|
| index DRAM → `cb_index_window` | uint32 RM | reader (RISCV_1, NoC0): `TensorAccessor(index).get_noc_addr(b, 128·h_first)` + `noc_async_read`, one read per image segment, one barrier | only this core's tile rows; read size `4·count` B exactly; window bytes past a partial tile row are never used as addresses |
| table DRAM → `cb_gathered_rows` | bfloat16 bits in a UInt16 CB, RM sticks of `block_width_tiles·32` values | reader (NoC0): `TensorAccessor(weight).get_noc_addr(idx[r], wb·block_width_tiles·64)` + `noc_async_read(…, block_width_tiles·64)` per valid position; one barrier per block | `[1, V, D]` weight: page id = `v` directly; no reshape |
| `cb_gathered_rows` → `cb_output_tiles` | RM → TILE, UInt16 | compute: `compute_kernel_lib::tilize` standard path (integer unpack/datacopy/pack, 16-bit DEST) | bit-exact for every 16-bit pattern; padding rows of partial tile rows carry stale L1 |
| `cb_output_tiles` → output DRAM | TILE bfloat16 bits | writer (RISCV_0, NoC1): `TensorAccessor(output)`, one whole 2048-B tile per write (`tile_reorder` catalog), flush per block, barrier at end | tile id `tr·Dt + wb·block_width_tiles + w` |
| (R3, deferred) index broadcast | uint32 | the core owning `wb = 0` of a tile row reads the slice and multicasts it to the other `width_blocks_per_row − 1` owners (`ttnn.Mcast`, ChainUnicast for non-rectangular groups, `McastArgs` sender/receiver); receivers wait on the channel's data-ready semaphore before their first row read | changes only `fill_index_window`'s source; `cb_index_window` layout unchanged |
| (R2, deferred) TILE index | uint32 tiles | `fill_index_window` reads two 64-B runs per tile row from index tile `(b/32)·ceil(S/32) + h`, faces `2·⌊(b mod 32)/16⌋` and `+1`, row offset `(b mod 16)·64` | positions ≥ `S_img` or batch rows ≥ `B` never read (rule 3) |
| Placement (not in TARGET) | — | — | a height shard of the output along tile rows would be the physical form of the token split (knob-turn: writer packs into a CB backed on the shard); a width shard would be the physical form of the D split (knob-turn). Neither is a TARGET value; `memory_config` other than DRAM interleaved is a `ValueError` |

## Work Distribution

| Field | Value |
|-------|-------|
| Work unit | one unit = tile row `tr` × width block `wb` (32 positions × `block_width_tiles` tiles); a block = `block_tile_rows` consecutive units |
| Grid | `device.compute_with_storage_grid_size()`; `C = grid.x · grid.y` (11 × 10 = 110 on the development board); never inlined |
| Per-core work | `ttnn.split_work_to_cores(grid, total_units, row_wise=True)` → cores used `= min(total_units, C)`; `core_group_1` holds `ceil(total_units / C)` units, `core_group_2` one fewer; each core gets the contiguous range `[core_first_unit, core_first_unit + core_num_units)` as per-core RT args; `row_wise=True` spreads the line across the DRAM-facing axis (`noc_placement` catalog) and keeps the used cores in ≤ 2 rectangles per group (BB-17) |
| Remainder | tile geometry per image with `ceil`: `T = B_img · ceil(S_img/32)`; `rows_valid = min(32, S_img − 32·h)`; uneven unit counts handled by `split_work_to_cores`' two groups; a core's final block may hold fewer than `block_tile_rows` units |

**Block-width selection (exact, host):**

```python
MAX_BLOCK_WIDTH_TILES = 80      # L1 cap (Mechanism caps)
UNIT_OVERHEAD_TILES = 2         # per-unit fixed cost in tile-equivalents (perf lamp LP1)

def select_block_width(T, Dt, C):
    candidates = [bw for bw in range(1, Dt + 1) if Dt % bw == 0 and bw <= MAX_BLOCK_WIDTH_TILES]
    def cost(bw):  # critical-path load of the busiest core
        return ceil(T * (Dt // bw) / C) * (bw + UNIT_OVERHEAD_TILES)
    return min(candidates, key=lambda bw: (cost(bw), -bw))   # ties -> wider block
```

Production selections at `C = 110`:

| Index shapes (tables) | T | Dt | `block_width_tiles` | `width_blocks_per_row` | units | cores | max units/core | footprint B |
|-----------------------|---|----|---------------------|------------------------|-------|-------|----------------|-------------|
| [16,4096] (D=64) | 2048 | 2 | 2 | 1 | 2048 | 110 | 19 | 18,816 |
| [16,4096] (D=1280) — p1 | 2048 | 40 | 40 | 1 | 2048 | 110 | 19 | 330,112 |
| [8,4096] (D=64) | 1024 | 2 | 2 | 1 | 1024 | 110 | 10 | 17,664 |
| [8,4096] (D=1280) — p2 | 1024 | 40 | 20 | 2 | 2048 | 110 | 19 | 165,120 |
| [16,1024] (D=64) | 512 | 2 | 2 | 1 | 512 | 110 | 5 | 17,024 |
| [16,1024] (D=1280) — p3 | 512 | 40 | 40 | 1 | 512 | 110 | 5 | 328,320 |
| [8,1024] (D=64 / 1280) | 256 | 2 / 40 | 2 / 20 | 1 / 2 | 256 / 512 | 110 | 3 / 5 | 16,768 / 164,224 |
| [1,4096] (D=64 / 1280) | 128 | 2 / 40 | 2 / 10 | 1 / 4 | 128 / 512 | 110 | 2 / 5 | 16,640 / 82,176 |
| [1,3844] (D=64 / 1280) — p4 | 121 | 2 / 40 | 2 / 10 | 1 / 4 | 121 / 484 | 110 | 2 / 5 | 16,640 / 82,176 |
| [16,128] (D=64 / 1280) | 64 | 2 / 40 | 2 / 8 | 1 / 5 | 64 / 320 | 64 / 110 | 1 / 3 | 16,512 / 65,792 |
| [1,1000], [1,1024], [8,128] (D=64 / 1280) | 32 | 2 / 40 | 1 / 4 | 2 / 10 | 64 / 320 | 64 / 110 | 1 / 3 | 8,320 / 33,024 |
| [1,512] (D=2560) | 16 | 80 | 16 | 5 | 80 | 80 | 1 | 131,200 |
| [2,128], [1,256] (D=64 / 1280) | 8 | 2 / 40 | 1 / 4 | 2 / 10 | 16 / 80 | 16 / 80 | 1 | 8,320 / 32,896 |
| [1,256] TILE (D=2560, R2) | 8 | 80 | 8 | 10 | 80 | 80 | 1 | 65,664 |
| [1,129] (D=64 / 1280) | 5 | 2 / 40 | 1 / 2 | 2 / 20 | 10 / 100 | 10 / 100 | 1 | 8,320 / 16,512 |
| [1,100], [1,128] (D=64 / 1280) | 4 | 2 / 40 | 1 / 2 | 2 / 20 | 8 / 80 | 8 / 80 | 1 | 8,320 / 16,512 |
| [1,77] (D=64 / 1280) | 3 | 2 / 40 | 1 / 2 | 2 / 20 | 6 / 60 | 6 / 60 | 1 | 8,320 / 16,512 |
| [1,64], [1,33] (D=64 / 1280) | 2 | 2 / 40 | 1 / 1 | 2 / 40 | 4 / 80 | 4 / 80 | 1 | 8,320 |
| [1,1], [1,8], [1,16], [1,31] (D=64 / 1280) | 1 | 2 / 40 | 1 / 1 | 2 / 40 | 2 / 40 | 2 / 40 | 1 | 8,320 |
| [1,1] (D=5120) | 1 | 160 | 2 | 80 | 80 | 80 | 1 | 16,512 |
| [1,1] ([1,256,128]) | 1 | 4 | 1 | 4 | 4 | 4 | 1 | 8,320 |
| [1] TILE (D=2560, R2) | 1 | 80 | 1 | 80 | 80 | 80 | 1 | 8,320 |

Tests must cover both `width_blocks_per_row = 1` and `> 1` on any grid ≥ 40 cores; the acceptance test does (`[16,4096]`, `[16,1024]` → 1; `[1,8]`, `[16,128]`, `[1,3844]` → > 1).

**Host program contract (BB-07, BB-08, BB-16, BB-21).**

| Item | Decision |
|------|----------|
| Descriptor cache | module-level dict keyed by `(tuple(idx.shape), idx.layout, tuple(weight.shape), weight.dtype, output memory_config, (grid.x, grid.y), device id)`; a hit rebuilds nothing — it reuses CBs, kernel CT args, per-core RT args and core ranges, and sets only the three buffer addresses |
| Per-call arguments | buffer addresses (index, weight, output) as **common runtime args** of the kernels that use them; nothing else changes per call; index values never reach the host |
| `custom_program_hash` | hash of the cache key (every CT-affecting quantity is a function of it) |
| Plan report (BB-08) | `describe_plan(input_tensor, weight, memory_config=None) -> dict` with `block_width_tiles`, `width_blocks_per_row`, `block_tile_rows`, `total_units`, `num_cores`, units per core per group, CB capacities, footprint bytes, regime name (`"R1"`) |
| Role dump (BB-21) | the same dict lists every used core with its role (`gather+tilize+store`) and unit range; all used cores share one role |
| Profile zones (BB-21) | reader: `IDX_FILL`, `GATHER_ISSUE`, `GATHER_WAIT` (barrier), `GATHER_SLOT_WAIT` (reserve); compute: `TILIZE` (helper-owned CB waits included); writer: `STORE_WAIT_IN`, `STORE_ISSUE`, `STORE_FLUSH`, `STORE_BARRIER`; zones compile out of the plain build |
| Dispatch | exactly one `ttnn.generic_op([input_tensor, weight, output], descriptor)` per call; output allocated by `ttnn.allocate_tensor_on_device(Shape(idx.shape + [D]), weight.dtype, TILE_LAYOUT, device, memory_config)` |

## Circular Buffers

| Semantic Name | Index | Page Size | Num Pages | Sizing rationale | Format | Producer | Consumer | Lifetime |
|---------------|-------|-----------|-----------|------------------|--------|----------|----------|-----------|
| `cb_index_window` | 0 | 128 B (32 uint32 = one tile row's indices) | `index_window_tile_rows` | spans the window's tile rows; streams nothing per block (read in place) | UInt32 | reader | reader (scratch: reserved once, addressed by write pointer, no push/pop) | whole kernel |
| `cb_gathered_rows` | 1 | 2048 B (one tile's worth of sticks: a unit is 32 sticks × `block_width_tiles·64` B = `block_width_tiles` pages) | `GATHER_DEPTH_BLOCKS · block_tile_rows · block_width_tiles` | spans `block_width_tiles` (D) and `block_tile_rows` (tokens) per slot × depth; streams table rows | UInt16 | reader | compute | per block |
| `cb_output_tiles` | 2 | 2048 B (one tile) | `OUTPUT_DEPTH_BLOCKS · block_tile_rows · block_width_tiles` | spans `block_width_tiles`, `block_tile_rows` per slot × depth | UInt16 | compute | writer | per block |

Push = wait on every CB: `cb_gathered_rows` reader pushes `block_width_tiles` per unit, compute waits/pops `block_width_tiles` per unit; `cb_output_tiles` compute pushes `block_width_tiles` per unit, writer waits/pops `block_width_tiles` per unit. Full ledger: `l1_ledger.md`.

## Block Operation Realization

| # | Block operation | Block shape | Helper? | Input CB (semantic name, pages, state) | Output CB (semantic name, pages) | CB state after |
|---|-----------------|-------------|---------|----------------------------------------|----------------------------------|----------------|
| 1 | `fill_index_window` | `index_window_tile_rows` × 32 indices | no (raw NoC; helper gap recorded in API Mapping) | index tensor (DRAM) | `cb_index_window`, `index_window_tile_rows` pages, reserved once | resident for all blocks of the window |
| 2 | `gather_rows_block` | `block_tile_rows` × (32 positions × `block_width_tiles` tiles) | no (raw NoC; helper gap recorded) | `cb_index_window` (read in place), table (DRAM) | `cb_gathered_rows`, `block_tile_rows·block_width_tiles` pages | pushed after the block barrier |
| 3 | `tilize_block` | `block_tile_rows` tile rows × `block_width_tiles` tiles | yes: `compute_kernel_lib::tilize` | `cb_gathered_rows`, `block_width_tiles` pages per tile row, waited/popped by the helper | `cb_output_tiles`, `block_width_tiles` pages per tile row | input freed per tile row |
| 4 | `store_tiles_block` | `block_tile_rows · block_width_tiles` tiles | no (raw NoC; no tile-store helper) | `cb_output_tiles`, `block_width_tiles` pages per unit | output tensor (DRAM) | popped after the block's flush |

## API Mapping

| Block operation | Type | Function | File:Line | Template Params / Args | Input CB | Output CB | Which params are block knobs |
|-----------------|------|----------|-----------|------------------------|----------|-----------|------------------------------|
| `fill_index_window` | raw_api | `TensorAccessor::get_noc_addr(page_id, offset)` | `tt_metal/hw/inc/api/tensor/tensor_accessor.h:165` | `page_id = b`, `offset = 128·h_first` | — | `cb_index_window` | `index_window_tile_rows` (segment length) |
| `fill_index_window` | raw_api | `noc_async_read` / `noc_async_read_barrier` | `tt_metal/hw/inc/api/dataflow/dataflow_api.h:547`, `:1921` | `size = 4·count` | — | `cb_index_window` | — |
| `gather_rows_block` | raw_api | `TensorAccessor::get_noc_addr(page_id, offset)` | `tensor_accessor.h:165` | `page_id = idx[r]`, `offset = wb·block_width_tiles·64` | `cb_index_window` | — | `block_width_tiles` |
| `gather_rows_block` | raw_api | `noc_async_read`, `noc_async_read_barrier`, `cb_reserve_back` / `cb_push_back` | `dataflow_api.h:547`, `:1921`, `:406`, `:208` | `size = block_width_tiles·64`, dst stride `block_width_tiles·64` | — | `cb_gathered_rows` | `block_width_tiles`, `block_tile_rows` |
| `tilize_block` | helper | `compute_kernel_hw_startup(cb_gathered_rows, cb_output_tiles)` | `tt_metal/hw/inc/api/compute/compute_kernel_hw_startup.h:113` | once per kernel | `cb_gathered_rows` | `cb_output_tiles` | — |
| `tilize_block` | helper | `compute_kernel_lib::tilize<block_width_tiles, cb_gathered_rows, cb_output_tiles>(in, out, core_num_units)` | `ttnn/cpp/ttnn/kernel_lib/tilize_helpers.hpp:194-207` (loop `tilize_helpers.inl:214-239`) | defaults: `InitAndUninit`, `WaitBlock`, `UnpackAndPackReconfigure`; UInt16 CBs select the standard path (`tilize_helpers.inl:65-69`) | `cb_gathered_rows` | `cb_output_tiles` | `block_width_tiles` (template), `num_blocks = core_num_units` |
| `store_tiles_block` | raw_api | `noc_async_write_page` (or `noc_async_write` + `get_noc_addr`), `noc_async_writes_flushed`, `noc_async_write_barrier` | `dataflow_api.h:1418`, `:876`, `:1980`, `:1951` | page id `tr·Dt + wb·block_width_tiles + w`, 2048 B | `cb_output_tiles` | — | `block_width_tiles`, `block_tile_rows` |
| perf lamp LP3 | raw_api | `dataflow_kernel_lib::set_read_trid` / `async_read_barrier_with_trid` | `ttnn/cpp/ttnn/kernel_lib/local_copy_helpers_dataflow.hpp:147`, `:158` | trid per block | — | `cb_gathered_rows` | `GATHER_DEPTH_BLOCKS` |
| R3 (deferred) | helper | `McastArgs` sender/receiver; host `ttnn.Mcast` | `ttnn/cpp/ttnn/kernel_lib/mcast/kernel/mcast_args.hpp`, `ttnn/cpp/ttnn/kernel_lib/mcast/host/mcast.hpp` | group = owners of one tile row | — | `cb_index_window` | — |

**Helpers considered and rejected.**

| Raw-API operation | Candidate helper | File:line of the mismatch | Concrete reason |
|-------------------|------------------|---------------------------|-----------------|
| `gather_rows_block` | `dataflow_kernel_lib::read_sticks_for_tilize` | `tilize_helpers_dataflow.hpp:88`; page id `start_page + block_row + row` at `tilize_helpers_dataflow.inl:121` | reads consecutive pages; the gather's page id is a per-row value loaded from L1 (`idx[r]`). Gap: a `read_sticks_for_tilize` variant taking a page-id span — build `gather_rows_block` as the block operation |
| `fill_index_window` | `dataflow_kernel_lib::read_sticks_for_tilize` | `tilize_helpers_dataflow.hpp:88`, `tilize_helpers_dataflow.inl:121` | reads whole sticks into a tilize-input CB; the window needs a byte sub-range `[128·h_first, 4·S_img)` of one page into a non-tilize scratch CB |
| `store_tiles_block` | `dataflow_kernel_lib::write_sticks_after_untilize` | `tilize_helpers_dataflow.hpp:130` | writes row-major sticks after untilize; the output here is whole TILE pages |

## Key Risks and Gotchas

| Risk | Why it bites here | Mitigation in this design |
|------|-------------------|---------------------------|
| Bit-exactness through tilize | the contract grades bit patterns (-0.0, subnormals, ±inf, NaN payloads); the float tilize route rewrites NaN and flushes subnormals; random data never shows it | both CBs `ttnn.uint16`; `fp32_dest_acc_en = False`, default unpack mode (UINT16 is how `tilize_with_val_padding` runs uint16: `tilize_with_val_padding_multi_core_default_program_factory.cpp:33-34`, checked over the full uint16 range by `test_tilize_with_val_padding.py::test_tilize_with_val_padding_docstring_supported_inputs`); acceptance test `test_g6_embedding_special_bit_patterns` on both Qwen tables. If it fails on this board, the fallback is R5 (rejected row) — CB layout unchanged |
| `compute_kernel_hw_startup` with the wrong CBs | srcA/pack formats come from the CBs passed; a bf16-labelled CB anywhere in the chain re-enters the float route | pass `cb_gathered_rows`, `cb_output_tiles` (both UInt16); no other compute CB exists |
| Partial tile rows (23 of 49 production cases) | the last tile row of every image has `rows_valid < 32`; for B > 1 this is mid-stream, not only at the end | the reader issues exactly `rows_valid` row reads per unit and still pushes the nominal `block_width_tiles` pages; padding rows hold stale L1 (unspecified by contract); the writer writes whole tiles |
| Index positions past the logical shape | rule 3 (TILE refinement): padding positions hold arbitrary values in the model | every row read is bounded by `rows_valid`; window bytes past `4·S_img` are never dereferenced; R2 never reads batch rows ≥ `B` |
| Multi-image core ranges | a core's units can span two images (`[16, 4096]`: 19 units, 128 tile rows per image) | `tr → (b, h)` per unit; the window fill issues one read per image segment; output tile ids `tr·Dt + …` are contiguous across images because each image is tile-padded independently |
| 64-B DRAM congruence | a 16-B-aligned L1 destination is not enough on Blackhole | every DRAM read has source and destination ≡ 0 mod 64 (Mechanism caps) |
| Ring wrap | a short block mid-range would leave the write pointer off a slot boundary | push/pop quantum is always one unit; the only short block is a core's last |
| Program cache (BB-07) and trace (BB-16) | a Python-side rebuild per call costs host time on every untraced call; a missing CT quantity in the key replays a wrong program | descriptor cache keyed as in Work Distribution, `custom_program_hash` from the key, addresses as common RT args; no allocation other than the output, nothing built after the first call |
| Grid-dependent selection | `select_block_width` depends on `C`; a shape can change `width_blocks_per_row` across boards | tests assert values, never the plan; the plan is reported by `describe_plan` |
| NoC completion before return (BB-03) | the next program starts right after the call | reader barriers every block (and the window fill); writer ends with `noc_async_write_barrier` |
| Kernel image size (BB-04, BB-18) | three kernels per core with profiler/NoC-trace builds | `block_width_tiles` is the only template-shaped CT quantity; the gather loop stays a runtime loop over `rows_valid` |
