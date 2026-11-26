# Operation Design: g6_concat

## Overview

| Field | Value |
|-------|-------|
| Classification | data_movement (pure move, no compute kernel) |
| Goal | Join K interleaved device tensors along one dim in one device program, moving every logical value bit for bit, reading each logical input row once and writing each logical output row once. |
| Math | `d = dim mod r`; `o_k = n_0 + … + n_{k-1}`; `out[i_0,…,o_k + j,…,i_{r-1}] = x_k[i_0,…,j,…,i_{r-1}]`, `0 ≤ j < n_k` |
| References | `eval/golden_tests/g6_concat/feature_spec.py` (TARGET / INPUTS / LOOSE_CASES), `eval/golden_tests/g6_concat/helpers.py` (bit-exact grading), `.claude/references/blocking-model.md`, `.claude/references/l1-footprint-discipline.md`, `.claude/references/noc-alignment-in-tt-metal.md`, `ttnn/cpp/ttnn/kernel_lib/local_copy_helpers_dataflow.hpp`, `ttnn/cpp/ttnn/kernel_lib/tile_column_dataflow.hpp:11-14` (face layout), catalog `ttnn/ttnn/operations/examples/master.md` (entries cited in Perf lamps) |
| Device facts used | Blackhole, worker grid 11×10 = **G = 110** cores on this board (`device.compute_with_storage_grid_size()`), 8 DRAM banks, NoC alignment: DRAM read 64 B, DRAM write 16 B, L1 16 B (`tt_metal/hw/inc/internal/tt-1xx/blackhole/noc/noc_parameters.h:375-380`); CB base addresses are aligned to the DRAM alignment (`tt_metal/impl/program/program.cpp:2505-2509`); per-core CB limit 1 460 288 B ≥ the op's budget 1 347 776 B. |
| Output artifacts | `op_design.md` (this), `l1_ledger.md` (CB ledger + data-movement budget), `tests/ttnn/unit_tests/operations/g6_concat/test_g6_concat.py` |

## Parameters

| Name | Type | Required | Valid Range | Default | CT/RT |
|------|------|----------|-------------|---------|-------|
| `tensors` | `list[ttnn.Tensor]` (positional) | yes | K ≥ 1; rank 3 or 4; one dtype, one layout; equal sizes except along `d`; interleaved; on one device; K ≤ `max_inputs` (mechanism cap) | — | input base addresses: common RT (reader); shapes: host plan |
| `dim` | `int` (positional) | no | `[-r, r-1]` | `0` | host only (selects plan; canonicalized `d = dim + r if dim < 0`) |
| `memory_config` | `ttnn.MemoryConfig` (keyword-only) | no | `None` or an INTERLEAVED config (DRAM or L1) | `None` → `ttnn.DRAM_MEMORY_CONFIG` | host only (output placement; output accessor CT args) |

No `compute_kernel_config`: nothing is computed.

## Tensors

### Input

| Property | Requirement |
|----------|-------------|
| Shape | rank 3 or 4, all inputs equal rank; sizes equal in every dim except `d`; `n_k ≥ 1` along `d` |
| Dtype | shared by all inputs. Phase 0: `bfloat16`. TARGET adds `float32`, `bfloat8_b` |
| Layout | shared by all inputs. Phase 0: `TILE_LAYOUT`. TARGET adds `ROW_MAJOR_LAYOUT` |
| Memory | INTERLEAVED. Phase 0: all DRAM. TARGET adds all-L1 and mixed; each input is read where it lives |

### Output

| Property | Value |
|----------|-------|
| Shape | shape of `x_0` with `N = Σ n_k` in dim `d` |
| Dtype | inputs' dtype |
| Layout | inputs' layout |
| Memory | INTERLEAVED in `memory_config.buffer_type`; DRAM when `memory_config is None` (also when inputs live in L1). Allocated by the entry point with `ttnn.allocate_tensor_on_device`; padding content unspecified |

## Blocking Model

All decisions below are made on the **output's tile-page index space**. Notation (tensor scope):
`A` = product of output dims before `d`; `Ht_out = ceil(H_out/32)` and `Wt_out = W_out/32` per image
(`H_out`, `W_out` = output's last two dims; W is a multiple of 32 everywhere in TARGET);
`total_out_pages = output.buffer_num_pages()`; `tile_bytes`, `face_bytes = tile_bytes/4`,
`row_bytes = face_bytes/16` (bf16: 2048 / 512 / 32 B). Face layout: face 0 = rows 0-15 × cols 0-15,
face 1 = rows 0-15 × cols 16-31, face 2 = rows 16-31 × cols 0-15, face 3 = rows 16-31 × cols 16-31,
row-major inside a face (`tile_column_dataflow.hpp:11-14`). So **tile row `r` is two 16-element
segments**: face `2·(r≥16)` and face `2·(r≥16)+1`, each at in-face offset `(r mod 16)·row_bytes`.

### Axes

| Axis | Character (+ one-clause reason) | Extent knob | Phase 0 value | Knob source | Core-assignment | Later unlock |
|------|--------------------------------|-------------|---------------|-------------|-----------------|--------------|
| `lead` — output dims before `d` (extent `A`) | independent — each slice of the output is built from the same slice of each input, no cross-slice dependency | folded into `out_page` (block extent `block_tiles`) | whatever `block_tiles` consecutive output pages cover | `plan.block_tiles` | with `out_page` | knob-turn |
| `join` — dim `d`, extent `N = Σ n_k`, partitioned into K input segments `[o_k, o_k+n_k)` | independent — every output position along `d` has exactly one source position (`o_k+j ← j` of `x_k`); nothing spans it, so splitting it needs no combine | folded into `out_page` | as above | `plan.block_tiles` | with `out_page` | knob-turn |
| `trail` — output dims strictly between `d` and the two tile dims (exists only for concat_dim `outer`; extent 1 otherwise) | independent — same reason as `lead` | folded into `out_page` | as above | `plan.block_tiles` | with `out_page` | knob-turn |
| `tile_row` — `Ht_out` tile rows per image (this IS `join` for concat_dim `h`) | independent — an output tile row draws rows from ≤ 2 source tile rows per input; row-granular reads give each source row to exactly one output tile, so there is no shared footprint to duplicate | folded into `out_page` | as above | `plan.block_tiles` | with `out_page` | knob-turn |
| `tile_col` — `Wt_out` tile columns (this IS `join` for concat_dim `w`) | independent — tile columns never mix (W is tile-aligned in TARGET) | folded into `out_page` | as above | `plan.block_tiles` | with `out_page` | knob-turn |
| **`out_page`** — the fold: flat output page index, row-major over `lead, join/trail, tile_row, tile_col` | independent — a fold of independent axes that are contiguous in output page order and treated identically (every output page is produced by the same block operation), so flattening is honest (`blocking-model.md` §1) | **`block_tiles`** (output pages per block) | `min(max_core_tiles, block_tiles_l1_cap)` — the coarsest block that fits: the core's whole assignment when it fits, else the L1 cap (see Mechanism caps) | `plan.block_tiles` → CB capacities + common RT arg of both kernels | contiguous page ranges over the **full grid**: `ttnn.split_work_to_cores(grid, total_out_pages, row_wise=True)`; `num_cores = min(G, total_out_pages)`; grid is a runtime query, never a constant | knob-turn |
| `row_in_tile` — the 32 rows inside one tile; `m` = logical rows of the tile (32 except the last tile row of an image) | independent — but it is the axis row_gather re-indexes: an output tile's row `r` may come from any input's row | whole tile (32 rows): a tile is both the NoC page and the CB page. `partial_rows_max` sets the **transfer shape** of a tile with `m < 32` logical rows | 32 rows per CB page; `partial_rows_max = 16` | `plan.partial_rows_max` (CT arg) | not split — splitting rows across cores turns one transfer into 2 × 32 B transfers per row on two cores, never cheaper | n/a (not split) |
| `source` — row_gather only: the shifted inputs feeding one output tile (extent ≤ `min(K, 32)`) | independent — each source's rows land on disjoint rows of the output tile; no combine | `staging_slots_per_tile` (`S`): every shifted source of a tile is staged in the same block | `S = max over output tile rows T of #{k : o_k mod 32 ≠ 0 and rows of x_k intersect T}` (host-computed, exact) | `plan.staging_slots_per_tile` → staging CB capacity + common RT arg | not split — all sources of a tile land on the core that owns the tile | n/a (not split) |

**Intermediate data re-check.** The only intermediate is the staged source rows (`cb_row_staging`,
row_gather). Its axes are `out_page` (split exactly like the output — the staged rows of an output
tile live on the core that owns that tile), `source` and `row_in_tile` (not split, reasons above).
No core collects, coordinates or hands out work: every core runs the same two stages on its own
page range, so no stage leaves the rest of the grid idle.

### Buffer-depth knobs

| CB | Depth knob | Phase 0 value | What the depth buys |
|----|------------|---------------|---------------------|
| `cb_concat_tiles` | `cb_depth` (effective `depth_eff = min(cb_depth, num_blocks_max)`, `num_blocks_max = ceil(max_core_tiles / block_tiles)`) | `cb_depth = 2` | reader fills block `i+1` while the writer places/stores block `i`; `depth_eff = 1` when a core has a single block (a second slot would never be used) |
| `cb_row_staging` | same `cb_depth` / `depth_eff` (lockstep with `cb_concat_tiles`: one push of each per block) | `2` | the reader stages block `i+1`'s shifted rows while the writer copies block `i`'s |

### Mechanism caps

| Mechanism | Cap on which extent | Clamp | What happens unclamped |
|-----------|--------------------|-------|------------------------|
| Op L1 budget (BB-05): 1 347 776 B per core | `block_tiles` | `block_tiles_l1_cap = floor(1347776 / (cb_depth · tile_bytes · (1 + S)))`; `block_tiles = max(1, min(max_core_tiles, block_tiles_l1_cap))`; host asserts `depth_eff·block_tiles·tile_bytes·(1+S) ≤ 1347776` | CB allocation fails at launch only when model tensors share L1 (a standalone test hides it), or CBs overlap model L1 |
| Blackhole DRAM read alignment: L1 and DRAM addresses must be equal mod 64 | where a DRAM row segment may land in L1 | every DRAM read lands at the **same intra-tile offset** it has in the source tile (slot bases are 64-B aligned, tile pages too). A shifted row therefore goes to a staging slot, never straight to its destination row | silent data corruption without watcher (`noc-alignment-in-tt-metal.md` §1) |
| L1↔L1 / L1→DRAM write alignment: equal mod 16 | size and offset of a row segment | row segments are multiples of `row_bytes` (bf16 32 B, fp32 64 B) — always legal. bfloat8_b exponents are 1 B per face row: **`partial_rows_max` is clamped to 0 for bfloat8_b** (whole tiles only, the exponent section moves with the tile) and row_gather is not available for bfloat8_b (deferred row) | torn rows / wrong exponents |
| Staging-slot sharing | rows staged per slot | one slot per (shifted input, output tile) holds ≤ 2 pieces of that input (tail rows `[a,32)` of source tile `t`, head rows `[0,b)` of tile `t+1`, with `b ≤ a`, so they are disjoint in tile offset space). **Staged reads move exactly rows `[a,b)` — never widened to the whole tile** | two in-flight DRAM reads overlap bytes in one slot → nondeterministic output (BB-06) |
| Sources per output tile | `S` | `S ≤ min(K, 32)` — each source contributes ≥ 1 of the tile's 32 rows | — (bound used by the ledger) |
| CB ring wrap (pointers reset only on an exact `fifo_limit` hit) | per-block push/pop quanta | every block pushes and pops the **nominal** `block_tiles` pages of `cb_concat_tiles` and `S·block_tiles` pages of `cb_row_staging`; a ragged last block narrows only the transfers | the write pointer walks past the CB into neighbouring L1 — wrong data, no hang (`ttnn-cb-memory-fundamentals.md` "CB Ring-Wrap Invariant") |
| Runtime-arg count: 341 words per core per kernel (`tt_metal/api/tt-metalium/kernel_types.hpp:50`) | K | 2 words per input (base address, run length / height) + ≤ 20 fixed words; `max_inputs = 160`; host raises `RuntimeError` above it | refused launch (BB-04) |

### Regimes

| Regime | Status | Predicate | Block | Data movement vs. minimum | What a bigger block buys |
|--------|--------|-----------|-------|---------------------------|--------------------------|
| `page_run` | **built** | `concat_dim ∈ {outer, w}` **or** (`concat_dim == h` **and** `n_k mod 32 == 0` for every `k < K-1`) | `block_tiles` consecutive output pages; each page has exactly one source page (page map is a bijection: output page `g = p·L_total + c`, source = page `p·L_k + (c − C_k)` of `x_k` for `C_k ≤ c < C_k + L_k`; `L_k` = `n_k·B_trail·Ht·Wt` (outer), `Ht_k·Wt` (h), `Wt_k` (w); `P = A` (outer, h) or `A·Ht` (w)) | **Minimum** at the DRAM boundary: each logical input row-segment read once, each logical output row-segment written once, zero cross-core traffic. Tiles with `m > partial_rows_max` logical rows move whole (their ≤ 15 padding rows ride along into output padding — allowed); tiles with `m ≤ partial_rows_max` move only rows `[0,m)` (≤ 2 transfers). Production: Qwen 17 = 201 326 592 B read + 201 326 592 B written; Qwen 160 (H = 1) = 124 160 B each way instead of 3 973 120 | per block, once each: `cb_reserve_back` + one read barrier + `cb_push_back` (reader), `cb_wait_front` + one writes-flushed + `cb_pop_front` (writer); this regime adds no term |
| `row_gather` | **built** | `concat_dim == h` **and** ∃ `k < K-1` with `n_k mod 32 ≠ 0` (rows of several inputs share output tiles at non-tile offsets) | `block_tiles` consecutive output pages × their staged sources (`S` staging slots per page) | **Minimum** at the DRAM boundary — every logical source row read once at face-row granularity (rows `[a,b)` of each source tile it meets), every logical output row written once; input padding is never read into a logical row. Adds **core-local** L1→L1 copies for shifted rows (no DRAM, no inter-core NoC). Qwen 79: reads 1 986 560 B (62 080 × 32 B), local copies 1 862 400 B, writes 1 986 560 B (1 940 × 1 KB) | as `page_run`, plus once per block: the staging CB's reserve/push/wait/pop and one self-aimed-read barrier in the writer; per output tile row (not per tile): the piece list is recomputed |
| `row_gather_column_walk` | deferred — no production input with a shifted row offset is taller than one tile (Qwen 79/89/143 all have `n_k = 1`), and on those shapes it moves the identical transfers | `concat_dim == h`, shifted inputs with `n_k > 32` | cores own whole output tile columns `(p, wt)` and walk `tile_row`; the straddling source tile stays resident between consecutive output tiles | same bytes as `row_gather`; saves DRAM **transactions** only: a shifted full source tile is 1 × `tile_bytes` read instead of ≤ 8 partial reads | reachable: `row_gather`'s per-output-tile piece list is unchanged; only the core assignment (column ranges instead of flat ranges) and one resident source slot per input change |
| `whole_tile_row_gather` | rejected — superseded by `row_gather` (same structure, row-granular reads) | `concat_dim == h`, shifted | read whole source tiles, place rows | 32× the input bytes on height-1 inputs: Qwen 79 reads 63 569 920 B instead of 1 986 560 B | — |
| `input_scatter` | rejected — superseded by `row_gather` | as `row_gather` | cores own input tiles and write each row segment straight to its output DRAM position | same bytes, 32× the DRAM write transactions on Qwen 79 (62 080 × 32 B writes vs 1 940 × 1 KB) | — |
| `tile_row_split` (height-only work split) | rejected — superseded by the flat `out_page` split | — | cores own whole tile rows / images | same bytes; strands one-tile-row outputs (Qwen 79/80/89/90/143/144/160: `Ht_out = 1`) on one core (catalog `width_split`) | — |
| `per_input_core_groups` | rejected — superseded by the flat `out_page` split | — | one core group per input | same bytes; uneven `n_k` (Qwen 18: 108 vs 84; Qwen 36: 15 vs 1) unbalances the groups | — |
| `stick_run` (ROW_MAJOR) | deferred — `ROW_MAJOR_LAYOUT` is not in Phase 0 SUPPORTED | layout ROW_MAJOR, concat_dim ∈ {outer, h} (every RM production case) | `page_run` with stick pages (page = one row of W elements, `buffer_aligned_page_size()` stride) | minimum: each stick read once, written once | as `page_run`. Reachable: same run table, same kernels with page = stick and no `row_in_tile` axis (RM h-concat is a stick-run copy: no row gather exists in RM) |
| `row_gather_bfp8` | deferred — `bfloat8_b` is not in Phase 0 SUPPORTED and no production bfloat8_b case joins along h | dtype bfloat8_b and row_gather predicate | as `row_gather` | as `row_gather` plus exponent bytes | reachable: mantissa row segments (16 B) move through the same staging/self-copy path; the 1-byte face-row exponents move by RISC-V byte stores in `place_rows_block` |
| placement variants (`input_buffer ∈ {l1, mixed}`, `output_buffer = l1`) | deferred — not in Phase 0 SUPPORTED | per-call placement | unchanged | unchanged (an L1 input is read over the NoC from its L1 banks instead of DRAM) | reachable by knob-turn: each input's address generator takes its own bank type; output accessor takes the output's; the L1 budget subtracts the output's per-bank share when the output is in L1 |

**Regime selection function** (host, exact, grid-independent):
`regime = "row_gather" if d == r-2 and any(n_k % 32 != 0 for k < K-1) else "page_run"`.
Regime-pinned tests are required and present in the acceptance test (`plan_of(...)["regime"]`).

`w_non_aligned` is not in TARGET (no production cell); no regime is designed for it.

### Traffic ranking

Counted at the DRAM boundary (each logical input row read once + each logical output row written
once is the floor) and the inter-core NoC (zero is the floor):

| Rank | Split | DRAM crossings per tensor | Cross-core traffic | Verdict |
|------|-------|---------------------------|--------------------|---------|
| 1 | flat `out_page` ranges over the full grid (`page_run` / `row_gather`) | each input logical row once, output logical row once | none (row_gather's placement copies are core-local) | **chosen** |
| 1 (tie on bytes) | `per_input_core_groups`, `tile_row_split`, `row_gather_column_walk` | same | none | rejected / deferred on occupancy, balance or transaction count (regime table) |
| 2 | `input_scatter` | same bytes | none | rejected: 32× write transactions |
| 3 | `whole_tile_row_gather` | up to 32× input bytes (height-1 inputs) | none | rejected |
| — | "split a dependent axis + combine" | — | — | does not exist: no result spans any axis |

**Operand-reuse check.** Split = output page ranges. For every input `x_k`: its pages vary along
`out_page` (each output page has a distinct source page, or distinct source rows in row_gather), so
no core re-reads bytes another core reads. No operand is reuse-shared; no broadcast row exists.
(`[x]*16` passes one buffer 16 times — 16 different output segments, each read once; inherent.)

**Stall-shadow check.** Waiting stages: the reader's per-block DRAM read barrier and the writer's
`cb_wait_front`. Placement of shifted rows (`place_rows_block`) is scheduled on the **writer**, so it
runs in the shadow of the reader's DRAM latency for the next block and splits NoC issue work across
both data-movement RISC-Vs (reader: DRAM reads; writer: self-copies + DRAM writes). Nothing else is
independent of the awaited data. No floating-point order exists to change.

### Block schedule

```cpp
for (uint32_t block_idx = 0; block_idx < num_blocks_this_core; ++block_idx) {
    load_block(block_idx);        // reader : DRAM -> cb_concat_tiles (direct rows) + cb_row_staging (shifted rows)
    place_rows_block(block_idx);  // writer : cb_row_staging -> cb_concat_tiles, core-local (row_gather only)
    store_block(block_idx);       // writer : cb_concat_tiles -> output DRAM
}
```

| Block operation | Block shape | Resident across it | Intended frequency of fixed costs |
|-----------------|-------------|--------------------|-----------------------------------|
| `load_block` | `this_block_tiles ≤ block_tiles` output pages; per page all its pieces (`page_run`: 1 piece; `row_gather`: ≤ 1 direct + ≤ `S` staged sources) | nothing from earlier blocks (pure stream); per-input address generators live for the whole kernel | 1 `cb_reserve_back` per CB, 1 `noc.async_read_barrier()`, 1 `cb_push_back` per CB — **per block**; address generators constructed once per kernel; piece list recomputed once per output tile row, not per tile |
| `place_rows_block` | same pages; for each staged piece: source rows `[a,b)` → destination rows `[d, d+b−a)` | the block's staged rows and output slots (held by the writer between wait and pop) | 1 `cb_wait_front` per CB, 1 self-aimed-read barrier — per block |
| `store_block` | same pages; output rows `[0, m_out)` of each page | — | 1 `noc.async_writes_flushed()` + 1 `cb_pop_front` per CB — per block; 1 `noc.async_write_barrier()` per kernel (end) |

Transfer shapes inside the operations (these are the unit traversals that belong *inside* a block
operation, per `blocking-model.md` §2):

| Transfer | Rows | Segments (all at identical intra-tile offsets unless stated) |
|----------|------|-------------------------------------------------------------|
| zero-shift tile (`page_run` read and write, `row_gather` direct read, every output write) | `[0, m)` | `m > partial_rows_max` → 1 × `tile_bytes`; else the segments of rows `[0,m)` by the staged-read rule below (with `partial_rows_max = 16`: face 0 rows `[0,m)` + face 1 rows `[0,m)`, merged into 1 × `2·face_bytes` when `m == 16`) |
| staged read (row_gather, shifted) | exactly `[a, b)` of one source tile | top part rows `[a, min(b,16))` in faces 0 and 1, bottom part rows `[max(a,16), b)` in faces 2 and 3 — ≤ 4 segments (a whole top or bottom face pair merges into one `2·face_bytes` segment), never widened |
| self-copy (staging slot → output slot) | source `[a,b)` → destination `[d, d+b−a)` | split at source row 16 and destination row 16 → ≤ 3 row segments × 2 face halves → ≤ 6 self-aimed reads; offsets differ (multiples of `row_bytes`, legal mod 16) |

Piece list of output tile `(p, T, wt)` in row_gather (both kernels derive it from the same common
RT args with one shared header — single source): output rows `[32T, 32T + m_T)`, `m_T = min(32, N − 32T)`;
for each `k` with `[o_k, o_k+n_k)` meeting them: source rows `j ∈ [max(0, 32T−o_k), min(n_k, 32T+m_T−o_k))`,
split at multiples of 32 into source tiles `t = j/32`, source page `(p·Ht_k + t)·Wt + wt`,
`Ht_k = ceil(n_k/32)`, destination row `o_k + j − 32T`. Shift `≡ o_k mod 32`: **zero → direct read
into the output slot; non-zero → staged read into slot `rank_T(k)`** (rank of `k` among the shifted
inputs meeting row `T`). At most one piece per output tile is zero-shift and it starts at row 0, so a
direct read never collides with another read; rows it widens over (whole-tile transfer) are
overwritten by `place_rows_block` after the push.

### Perf lamps

| Lamp | Why the default may be wrong here | Nearby alternative to measure |
|------|-----------------------------------|-------------------------------|
| Overlap (block size) | `block_tiles` = whole assignment / L1 cap; a single-block core (`depth_eff = 1`: e.g. FLUX 29, 231 pages/core) serializes its read against its write. At grid scale DRAM stays saturated in both phases, so the default is expected to hold | `block_tiles ∈ {ceil(core_tiles/2), 64, 16}` with `cb_depth = 2` on Qwen 17 (p2), FLUX 29, Qwen 25 |
| Grid synchronization | tiny outputs (FLUX 22/23: 8 pages; Qwen 109: 48) get 1 page per core; launch + barrier cost may dominate | floor of `min_tiles_per_core ∈ {2, 4, 8}` (fewer cores) on outputs `< 4·G` pages |
| Bank stagger (catalog `bank_stagger`, 1.04–1.32× on BH) | when `max_core_tiles` is a multiple of 8, every core starts on the same DRAM bank | rotate each core's walk start by `core_index mod 8` within its range |
| Barrier pipelining (catalog `trid_double_issue`) | one read barrier per block idles the NoC between blocks on a core | trid-tagged reads, retire the oldest block (`set_read_trid` / `async_read_barrier_with_trid`, `local_copy_helpers_dataflow.hpp:147,158`) with `cb_depth = 3` |
| `partial_rows_max` (catalog `tile_reorder`: whole-tile transfers beat face transfers on full tiles) | the 16-row threshold trades bytes against transaction count on partial tiles | `partial_rows_max ∈ {0, 16, 31}` on Qwen 160 (m = 1), Qwen 80 (m = 16), Qwen 109 (m = 31) |
| Residue-matched direct read (row_gather) | bf16 rows with an even shift (and every fp32 row) satisfy the mod-64 DRAM rule at the destination offset, so their staging + self-copy is avoidable | read pieces with `(dst_offset − src_offset) mod 64 == 0` straight into the output slot; stage only the rest (halves staging and self-copies on Qwen 79) |
| Issue split (catalog `split_reader`) | Qwen 79 issues 576 DRAM reads per core on one RISC-V | move half the inputs' staged reads to the writer RISC-V when the reader is measured issue-bound |

Catalog entries that set the defaults: `noc_placement` (row-wise core order, reads NoC0 / writes
NoC1), `width_split` (the flat split covers one-tile-row outputs), `double_buffer` + `trid_double_issue`
(depth 2; at grid scale the copy is DRAM-bound and a large-CB one-block loop is within 3 % of the best),
`tile_reorder` (whole-tile transfers for full tiles). `read_coalescing` does not apply: interleaved
pages are not contiguous.

## Dataflow Strategy

| Stage | Format | Mechanism | Notes |
|-------|--------|-----------|-------|
| `x_k` in DRAM | TILE pages (bf16 2048 B), interleaved over 8 banks | — | `K` base addresses in the reader's common RT args |
| DRAM → `cb_concat_tiles` slot | whole tile or face-row segments, same intra-tile offsets | reader (NCRISC, NoC0): `Noc::async_read(TensorAccessor, cb, size, {page_id, offset_bytes}, {offset_bytes})` | all `page_run` reads; the zero-shift piece in `row_gather` |
| DRAM → `cb_row_staging` slot | face-row segments at the **source** offsets | reader (NoC0), same call | shifted pieces (row_gather) |
| `cb_row_staging` → `cb_concat_tiles` slot | face-row segments re-indexed to destination rows | writer (BRISC, NoC1): self-aimed `Noc::async_read` with `dataflow_kernel_lib::local_addr(addr, noc.get_noc_id())` | row_gather only; barriered before any write of that block |
| `cb_concat_tiles` → output | whole tile or rows `[0, m_out)` | writer (NoC1): `Noc::async_write(cb, TensorAccessor, size, {offset}, {page_id, offset_bytes})` | flush before pop; final write barrier |

No Tensix-to-Tensix traffic in any regime (built or deferred); no semaphores, no multicast.

**TARGET placement axes as regime rows.** TARGET has no sharded values; its placement axes are
`input_buffer ∈ {dram, l1, mixed}` and `output_buffer ∈ {dram, l1}`, all INTERLEAVED. Each is a
**knob-turn** (no axis character changes): an L1 input is read through its own address generator with
the L1 bank type, the output accessor takes the output's bank type, and an L1 output's per-bank share
is subtracted from the CB budget. No input or output is moved to another placement.

## Work Distribution

| Field | Value |
|-------|-------|
| Work unit | an output page; a block is `block_tiles` consecutive output pages |
| Grid | `device.compute_with_storage_grid_size()` (11×10 here); `ttnn.split_work_to_cores(grid, total_out_pages, row_wise=True)` → `num_cores = min(G, total_out_pages)`, `all_cores`, `core_group_1` (`max_core_tiles = ceil(total_out_pages/num_cores)` pages each), `core_group_2` (one fewer). Kernels are created once per role on `all_cores` (≤ 2 rectangles: full rows + one partial row — BB-17) |
| Per-core work | pages `[start_page, start_page + num_pages)` (per-core RT args); `num_blocks_this_core = ceil(num_pages / block_tiles)` |
| Remainder | group 2 gets `max_core_tiles − 1` pages; the last block of a core is ragged (`last_block_tiles = num_pages − (num_blocks_this_core − 1)·block_tiles`) and still pushes/pops the nominal `block_tiles` |
| Tile geometry | per image: `Ht = ceil(H/32)`, `Wt = W/32`; row_gather output `Ht_out = ceil(N/32)` (≠ `Σ ceil(n_k/32)`); `m` of output page `g` = `32` unless `(g / Wt_out) mod Ht_out == Ht_out − 1`, then `H_out − 32·(Ht_out − 1)`; `total_out_pages = output.buffer_num_pages()` |
| Examples (G = 110) | Qwen 17: 98 304 pages → 894/893 per core, `block_tiles = 329`, 3 blocks; Qwen 18 (fp32): 894 per core, `block_tiles = 164`, 6 blocks; Qwen 79: 1 940 pages → 18/17, `S = 15`, `block_tiles = 18`, 1 block; FLUX 23: 8 pages → 8 cores × 1 |

## Host Plan and Program Cache

| Item | Decision |
|------|----------|
| Plan function | `_plan(scenario, device)` → `{regime, num_cores, core_ranges, per-core (start_page, num_pages), block_tiles, cb_depth, depth_eff, staging_slots_per_tile, partial_rows_max, tile_bytes, l1_bytes_per_core}`. Depends only on shapes, dim, dtype, layout, placements and the grid — never on live free L1, so a production shape always gets the same plan (BB-08) |
| Plan reporting (BB-08, BB-21) | `plan_of(tensors, dim=0, *, memory_config=None) -> dict`, exported from `ttnn.operations.g6_concat`, returns the plan above plus `roles = {"reader": "NCRISC/NoC0", "writer": "BRISC/NoC1"}` for every core in `core_ranges` (cores outside are idle) |
| Program cache (BB-07) | the ProgramDescriptor is built once per structural key `(device id, input shapes, canonical d, dtype, layout, input buffer types, output buffer type)` and kept in a module-level dict; a repeat call patches only the buffer addresses (common RT args: input addresses in the reader, output address in the writer) and dispatches. Shape-derived values are RT args or CT args; generic_op re-applies every RT value of the passed descriptor on a cache hit because no `buffer_bindings` are declared (`ttnn/api/ttnn/mesh_device_operation_adapter.hpp:905-962`, slow-path `apply_descriptor_runtime_args`). Do **not** add `buffer_bindings`/`emplace_runtime_args`: that fast path patches only addresses and would leave shape RT args stale between structures with equal hashes |
| CT vs RT | CT: regime flag, tile geometry (`tile_bytes`, `face_bytes`, `row_bytes`), `partial_rows_max`, TensorAccessorArgs (one input class + the output). RT (common): `K`, per-input base address + run length `L_k` (page_run) or height `n_k` (row_gather), output geometry (`L_total`, `Wt_out`, `Ht_out`, `last_tile_rows`, `N`), `block_tiles`, `staging_slots_per_tile`, output address. RT (per core): `start_page`, `num_pages`. Keeping shapes in RT keeps the number of distinct kernel binaries to one per (regime, dtype) |
| Trace (BB-16) | the output is allocated in the entry point at call (capture) time; nothing else is allocated; a replay reuses the captured RT args |
| Profiling (BB-21) | `DeviceZoneScopedN` zones: reader `G6C-RD-RESERVE`, `G6C-RD-ISSUE`, `G6C-RD-BARRIER`; writer `G6C-WR-WAIT`, `G6C-WR-PLACE`, `G6C-WR-ISSUE`, `G6C-WR-FLUSH`, `G6C-WR-FINAL-BARRIER`. Zones compile out of the plain build |

## Entry Point and Registry Contract

| Item | Decision |
|------|----------|
| Signature | `g6_concat(tensors, dim=0, *, memory_config=None) -> ttnn.Tensor`; first line `validate(tensors, dim, memory_config=memory_config)` |
| validate() order | (1) structural → `ValueError`: empty list; rank ∉ {3, 4} or ranks differ; dtypes differ; layouts differ; `dim ∉ [-r, r-1]`; sizes differ in a dim other than `d`; a tensor not on device / on another device; a sharded input or sharded `memory_config`. (2) build the scenario dict `{"shapes", "dim", "dtype", "layout", "input_buffers", "memory_config": "l1" if memory_config is L1 else None}` and run INPUT_TAGGERS in declaration order. (3) SUPPORTED per axis → `UnsupportedAxisValue`. (4) EXCLUSIONS → `ExcludedCell`. (5) `K > max_inputs` → `RuntimeError`. Structural checks first, so a dtype mismatch is a `ValueError`, never a support refusal. Every message the op raises starts with `g6_concat:` (the acceptance test matches it through the `expect_error` fixture) |
| INPUT_TAGGERS | `dtype` (shared dtype), `layout` (shared layout), `concat_dim` (`d = dim + r if dim < 0`; `"w"` if `d == r-1`, `"h"` if `d == r-2`, else `"outer"` — canonicalized before tagging, never the raw literal), `alignment` (`"w_non_aligned"` if any `shape[-1] % 32`, else `"h_non_aligned"` if any `shape[-2] % 32`, else `"tile_aligned"`), `input_buffer` (`"dram"` / `"l1"` / `"mixed"`), `output_buffer` (`"l1"` iff `memory_config == "l1"`, else `"dram"`) |
| Phase 0 SUPPORTED | `dtype: [bfloat16]`, `layout: [TILE_LAYOUT]`, `concat_dim: ["outer", "h", "w"]`, `alignment: ["tile_aligned", "h_non_aligned"]`, `input_buffer: ["dram"]`, `output_buffer: ["dram"]` (the implementer writes the block) |
| EXCLUSIONS | none. h-concat with rows of several inputs in one tile is `row_gather` — never excluded or gated |
| Dispatch | `ttnn.generic_op([*tensors, output], descriptor)` — output last; exactly one program; no `to_layout` / `tilize` / `pad` / `slice` / typecast / memory move anywhere |
| Structural impossibilities | none beyond feature_spec's INVALID |

## Circular Buffers

| Semantic Name | Index | Page Size | Num Pages | Sizing rationale | Format | Producer | Consumer | Lifetime |
|---------------|-------|-----------|-----------|------------------|--------|----------|----------|----------|
| `cb_concat_tiles` | 0 | `tile_bytes` | `depth_eff · block_tiles` | spans `out_page` (`block_tiles`) and `row_in_tile` (whole tile per page); streams over blocks; depth buys read/write overlap | input dtype (`Float16_b` in Phase 0) — dataflow-only payload, never through DEST | reader | writer | whole kernel, both regimes |
| `cb_row_staging` | 1 | `tile_bytes` | `depth_eff · block_tiles · staging_slots_per_tile` | spans `out_page` (`block_tiles`), `source` (`S`) and `row_in_tile`; streams over blocks; lockstep with `cb_concat_tiles` | input dtype | reader | writer | whole kernel, **row_gather only** (not created in page_run) |

The writer writes into the front pages of `cb_concat_tiles` it has waited on (self-copies) before
storing them; it is that CB's only consumer, so ownership stays single-producer / single-consumer.
Full ledger: `l1_ledger.md`.

## Block Operation Realization

| # | Block operation | Block shape | Helper? | Input CB (semantic name, pages, state) | Output CB (semantic name, pages) | CB state after |
|---|-----------------|-------------|---------|----------------------------------------|----------------------------------|----------------|
| 1 | `load_block` | `block_tiles` output pages (+ `S·block_tiles` staging pages) | no (raw NoC API; see API Mapping) | — (DRAM `x_k`) | `cb_concat_tiles` `block_tiles` reserved → pushed; `cb_row_staging` `S·block_tiles` reserved → pushed (row_gather) | pushed with only the block's real pieces landed; ragged pages untouched |
| 2 | `place_rows_block` | same pages; ≤ `S` staged sources each | yes — `dataflow_kernel_lib::local_addr` | `cb_row_staging` `S·block_tiles` waited (front) | `cb_concat_tiles` `block_tiles` waited (front), modified in place | both still held by the writer |
| 3 | `store_block` | same pages, rows `[0, m_out)` | no (raw NoC API) | `cb_concat_tiles` `block_tiles` (front) | — (DRAM output) | `cb_concat_tiles` and `cb_row_staging` popped by the nominal counts after `async_writes_flushed()` |

## API Mapping

| Block operation | Type | Function | File:Line | Template Params / Args | Input CB | Output CB | Which params are block knobs |
|-----------------|------|----------|-----------|------------------------|----------|-----------|------------------------------|
| `load_block` | raw_api | `Noc::async_read` | `tt_metal/hw/inc/api/dataflow/noc.h:252` | `src` = per-input `TensorAccessor` with `{.page_id, .offset_bytes}` (`tt_metal/hw/inc/api/tensor/noc_traits.h:135-171`); `dst` = CB with `{.offset_bytes}` (slot + intra-tile offset); `size_bytes` = segment | — | `cb_concat_tiles`, `cb_row_staging` | `block_tiles` (reads per barrier), `partial_rows_max` (segment shape) |
| `load_block` | raw_api | `Noc::async_read_barrier` | `noc.h:739` | once per block | — | — | `block_tiles` |
| `load_block` | raw_api | `cb_reserve_back` / `cb_push_back` | `tt_metal/hw/inc/api/dataflow/dataflow_api.h:406` / `:208` | nominal `block_tiles`, `S·block_tiles` | — | both | `block_tiles`, `S` |
| `place_rows_block` | helper | `dataflow_kernel_lib::local_addr` + `Noc::async_read` | `ttnn/cpp/ttnn/kernel_lib/local_copy_helpers_dataflow.hpp:57`, `.inl:30-34` (usage example `.inl:17-28`) | `local_addr(src_l1_addr, noc.get_noc_id())`, dst = `cb_concat_tiles` with `{.offset_bytes}` | `cb_row_staging` | `cb_concat_tiles` | `S` |
| `place_rows_block` | raw_api | `cb_wait_front` / `Noc::async_read_barrier` | `dataflow_api.h:479` / `noc.h:739` | once per block | both | — | `block_tiles` |
| `store_block` | raw_api | `Noc::async_write` | `noc.h:411` | `src` = CB `{.offset_bytes}`, `dst` = output `TensorAccessor` `{.page_id, .offset_bytes}` | `cb_concat_tiles` | — | `block_tiles`, `partial_rows_max` |
| `store_block` | raw_api | `Noc::async_writes_flushed` / `cb_pop_front` / `Noc::async_write_barrier` | `noc.h:783` / `dataflow_api.h:260` / `noc.h:760` | flush + pop per block; barrier once at kernel end (BB-03) | both | — | `block_tiles` |

**Helpers considered and rejected (`load_block`, `store_block`):**

| Candidate | File:Line | Why it cannot express this block operation |
|-----------|-----------|--------------------------------------------|
| `copy` (eltwise convenience) | `ttnn/cpp/ttnn/kernel_lib/eltwise/api/convenience.hpp:109-113` (requires `compute_kernel_hw_startup`, `:29`) | a compute-kernel CopyTile → PackTile through DEST: the op computes nothing, a DEST round trip is a format pass the contract forbids ("MUST NOT convert … flush"), and it keeps every row at its tile position — it cannot re-index rows. It also does not read DRAM: the DRAM transfer still needs a dataflow kernel |
| `tilize` / `untilize` | `tilize_helpers.hpp:91-95,206`, `untilize_helpers.hpp:81` | format conversions on the compute engines; data arrives and leaves in TILE layout; untilize → place → tilize is two extra passes through unpack/pack |
| `read_sticks_for_tilize` | `tilize_helpers_dataflow.hpp:29-36` | reads row-major sticks for a tilize; inputs are tiles |
| `set_read_state` / `read_with_state` (stateful self-copy) | `local_copy_helpers_dataflow.hpp:74-95`, `.inl:60` ("Changing the source address or the size means calling set_read_state() again") | every self-copy segment has a different source address and size; the stateful path pays its setup per segment. Plain `Noc::async_read` with `local_addr` is the helper's documented form for that (`.inl:17-28`) |
| `Mcast` / `McastArgs` | `ttnn/cpp/ttnn/kernel_lib/mcast/kernel/mcast_args.hpp` | no reuse-shared operand exists |

There is no compute kernel, so `compute_kernel_hw_startup()`, reduce scalers and `fp32_dest_acc_en`
do not apply. The gap that would make `load_block` / `store_block` a helper call is a dataflow helper
"move logical rows `[a,b)` of a tile between an accessor page and an L1 slot at matching offsets";
its body is the segment rule of the transfer table above.

## Key Risks and Gotchas

| Risk | Why it bites here | Mitigation in this design |
|------|-------------------|---------------------------|
| Padding copied into logical rows | row_gather shifts rows across tile boundaries; test inputs are zero-padded and hide it | only logical rows are ever read for shifted pieces; widened transfers happen only on zero-shift tiles, whose extra rows are padding→padding or overwritten by `place_rows_block`. Poisoned-padding cases in the acceptance test (Qwen 109 with 1000.0, Qwen 89 with NaN) |
| Staged reads widened to a whole tile | two pieces of one input share a staging slot at disjoint rows; a widened read overwrites the other piece's rows while both are in flight | staged reads move exactly rows `[a,b)` (Mechanism caps) |
| DRAM read landing at the destination row | Blackhole compares DRAM read addresses mod 64; a bf16 row at odd row parity differs by 32 B | every DRAM read lands at its source intra-tile offset; only `place_rows_block` changes offsets (L1↔L1, mod 16) |
| Self-copy on the wrong NoC coordinates | the writer issues on NoC1; `my_x[0]` is the NoC0 coordinate | `local_addr(addr, noc.get_noc_id())` (`local_copy_helpers_dataflow.hpp:48-51`) |
| Write issued before self-copies land | `store_block` reads the output slot the self-copies are still writing | one self-aimed-read barrier per block before the first write of that block |
| Ragged block breaks the CB ring | `num_pages` is rarely a multiple of `block_tiles` | nominal push/pop counts; only transfers narrow |
| Stale shape RT args on a program-cache hit | different structures share a generic_op hash when CT args, CB sizes and core ranges coincide — FLUX 23 `(1,1,1,128)×2` and Qwen 108 `(1,1,31,64)+(1,1,31,192)` both give 8 pages on 8 cores with `block_tiles = 1`, but run lengths `(4,4)` vs `(2,6)` and logical rows 1 vs 31 | no buffer bindings → generic_op re-applies all RT values; the acceptance test runs FLUX 23 → Qwen 108 → FLUX 23 on one device |
| Same tensor passed K times (`[x] * 16`) | one buffer appears 16 times in `generic_op`'s tensor list | inputs are read-only; each occurrence is a different output segment |
| L1 budget at the cap | Qwen 17 uses 1 347 584 B of the 1 347 776 B budget for `cb_concat_tiles` | cap formula + host assert; for L1-output refinements the output's per-bank share is subtracted first |
| Partial writes leave stale DRAM in output padding | `m ≤ partial_rows_max` tiles write only logical rows | allowed by contract (output padding unspecified); graders compare logical values only |
| One program, one pass | the easy fix-up (second program, host pad/slice, DRAM staging) is forbidden | both regimes are inside one program; no host tensor transformation |
| `bfloat8_b` partial tiles | exponents are 1 B per face row | `partial_rows_max` clamped to 0 for bfloat8_b; `row_gather_bfp8` deferred |
