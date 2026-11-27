# Operation Design: g6_embedding

## Overview

| Field | Value |
|-------|-------|
| Classification | data_movement (index-driven row gather + bit-exact row-major → tile relayout; no arithmetic) |
| Goal | Gather one table row per index on the device and write the rows as TILE pages of `idx.shape + [D]`, every value bit-identical to the table, in one device program per call. |
| Math | `out[i, d] = W[idx[i], d]` for every position `i` of `idx`, `d ∈ [0, D)`; `W = weight` (`[V, D]`) or `weight[0]` (`[1, V, D]`) |
| Route (decided) | reader gathers table-row segments into a row-major staging CB → compute tilizes with **both CBs declared `UInt16`** and 16-bit DEST (bit passthrough) → writer writes tiles |
| References | `docs/explorations/g6_embedding/{DOSSIER,math,tensix,risks,lessons}.md`; `eval/golden_tests/g6_embedding/{feature_spec,helpers}.py`; `.claude/references/{blocking-model,l1-footprint-discipline,op-design-template,ttnn-cb-memory-fundamentals,noc-alignment-in-tt-metal}.md`; helper `ttnn/cpp/ttnn/kernel_lib/tilize_helpers.hpp`; catalog `ttnn/ttnn/operations/examples/master.md` entries `double_buffer`, `trid_double_issue`, `width_split`, `noc_placement`, `bank_stagger`, `split_reader`, `tile_reorder`, `compute_block_size` |

Catalog entries → knobs they informed:

| Entry | Knob / decision |
|-------|-----------------|
| `width_split` | `col_tile` is spread across cores when tile-rows cannot fill the grid (regime `column_blocked`) |
| `noc_placement` | `split_work_to_cores(..., row_wise=True)`; reader NoC0, writer NoC1 (defaults) |
| `double_buffer`, `trid_double_issue` | one read barrier per block over 32 independent row reads, `rows_depth = 2`; trid as the overlap lamp |
| `tile_reorder` | writer moves whole 2 KB tiles, one write per tile, never faces |
| `compute_block_size` | one tilize helper call per core (init/uninit once), not one per block |
| `bank_stagger` | writer bank-rotation lamp |
| `split_reader` | reader-issue-bound lamp for the 128-B-row table |

## Parameters

| Name | Type | Required | Valid Range | Default | CT/RT |
|------|------|----------|-------------|---------|-------|
| `input_tensor` (positional) | `ttnn.Tensor` | yes | uint32, rank 1 `[N]` or rank 2 `[B, S]`, DRAM interleaved, ROW_MAJOR (Phase 0) / TILE (refinement) | — | shape → host plan; values never leave the device |
| `weight` (positional) | `ttnn.Tensor` | yes | bfloat16, ROW_MAJOR, `[V, D]` or `[1, V, D]`, `D % 32 == 0`, DRAM interleaved | — | shape → host plan |
| `layout` (kw) | `ttnn.Layout` | no | `ttnn.TILE_LAYOUT` only (axis `output_layout`) | `ttnn.TILE_LAYOUT` | host |
| `memory_config` (kw) | `ttnn.MemoryConfig \| None` | no | `None` or DRAM interleaved | `None` (= DRAM interleaved) | host |
| `runtime_rows` (kw) | `bool` | no | `False` only (axis `runtime_rows`; `True` → `UnsupportedAxisValue` whose message names `runtime_rows`) | `False` | host |

No `compute_kernel_config`, no `padding_idx` / `max_norm` / `norm_type` / `scale_grad_by_freq` / `sparse`.

## Tensors

### Input

| Property | `input_tensor` (idx) | `weight` (W) |
|----------|----------------------|--------------|
| Shape | `[B, S]` or `[N]` (S, N ≥ 1, unbounded) | `[V, D]` or `[1, V, D]`; read as it arrives, no reshape (BB-02) |
| Dtype | uint32 | bfloat16 |
| Layout | ROW_MAJOR (Phase 0): one page per row of B, `4·S` bytes (rank 1: one page of `4·N` bytes). TILE (refinement): uint32 tiles, positions in face order | ROW_MAJOR: one page per table row, `2·D` bytes (V pages for both ranks) |
| Memory | DRAM interleaved | DRAM interleaved |

### Output

| Property | Value |
|----------|-------|
| Shape | `idx.shape + [D]`: `[B, S, D]` or `[N, D]` |
| Dtype | bfloat16 (= weight dtype; bytes are the table's bytes) |
| Layout | TILE; per image `ceil(S/32)` tile-rows × `D/32` tile-columns; tail rows of a partial last tile-row are unspecified padding |
| Memory | DRAM interleaved (`memory_config=None` or explicit DRAM interleaved) |

### Support contract (registry; the implementer writes SUPPORTED)

| Axis | Source in `validate()` | TARGET | Phase 0 |
|------|------------------------|--------|---------|
| `dtype` | `weight.dtype` | bfloat16 | bfloat16 |
| `layout` | tagger `tag_layout`: scenario `index_layout` (`"ROW_MAJOR"` → `ttnn.ROW_MAJOR_LAYOUT`, `"TILE"` → `ttnn.TILE_LAYOUT`) | ROW_MAJOR, TILE | ROW_MAJOR |
| `alignment` | tagger `tag_alignment`: `index_shape[-1] % 32 == 0` → `"tile_aligned"`, else `"h_non_aligned"` | both | both (no EXCLUSIONS, no size split) |
| `output_layout` | `layout` keyword | TILE | TILE |
| `runtime_rows` | `bool(runtime_rows)` | False | False |

`validate()` builds `(scenario,)` with keys `index_shape`, `weight_shape`, `index_layout`, `memory_config` exactly as `feature_spec.py` documents. It raises `ValueError` for: idx rank ∉ {1, 2} or dtype ≠ uint32; weight rank ∉ {2, 3}, rank 3 with leading extent ≠ 1, not ROW_MAJOR; `D % 32 ≠ 0`; a tensor not on device / not DRAM interleaved; a `memory_config` not DRAM interleaved; `select_plan(...).l1_bytes_per_core > 1_347_776` (message names the bound; unreachable by construction, see Mechanism caps). Every refusal message names the offending tensor (`input_tensor` / `weight`), keyword or axis. EXCLUSIONS: none. Structural impossibilities (INVALID): none to add.

TARGET − SUPPORTED = `layout: TILE` (index). Cost class: **knob-turn** — only `fetch_index_block`'s address math changes (Dataflow Strategy); no other block operation, CB or split changes. TARGET has no `memory_layout` axis: every tensor of this contract is DRAM interleaved, so no physical-shard scheme exists.

## Blocking Model

Semantics: `.claude/references/blocking-model.md`. Grid on the bring-up chip: 11 × 10 = 110 cores (Blackhole); the design reads it from `device.compute_with_storage_grid_size()`, never a literal.

Index space: output tiles `(g, k)`, `g ∈ [0, tensor_tile_rows)` = `image · tile_rows_per_image + tr`, `k ∈ [0, tensor_col_tiles)`. Scope names (§3):

| Scope | Names |
|-------|-------|
| tensor | `tensor_images = B` (1 for rank 1), `tensor_token_extent = S` (N for rank 1), `tile_rows_per_image = ceil(S/32)`, `tensor_tile_rows = tensor_images · tile_rows_per_image`, `tensor_col_tiles = D/32` |
| block | `block_tile_rows`, `block_col_tiles`, `col_blocks_per_row = tensor_col_tiles / block_col_tiles`, `valid_rows(g) = min(32, S − 32·tr)` |
| core | `first_block_this_core`, `num_blocks_this_core`, `block_idx` |

### Axes

| Axis | Character (+ one-clause reason) | Extent knob | Phase 0 value | Knob source | Core-assignment | Later unlock |
|------|--------------------------------|-------------|---------------|-------------|-----------------|--------------|
| `image` (B) | independent — each output row reads only its own index | folded into `tile_row` (no own knob) | — | — | via `tile_row` | — |
| | *Fold reason:* images are contiguous in `g` and treated identically; a block never straddles an image because a block is one tile-row and tile-rows are per image (`tile_rows_per_image = ceil(S/32)`). | | | | | |
| `tile_row` (g, `tensor_tile_rows`) | independent — no result spans tokens (math.md, F1) | `block_tile_rows` | **1** | host constant `BLOCK_TILE_ROWS = 1` (scales CB capacities and the reserve quantum) | contiguous runs of blocks over all cores (`split_work_to_cores(grid, total_blocks, row_wise=True)`) | knob-turn |
| | *Why 1 (not the bound):* (a) the compute block along this axis is the core's whole run — ONE tilize helper call with `num_blocks = num_blocks_this_core`, init/uninit once; the CB only *streams* tile-rows through it, so a coarser value buys no init amortization; (b) one tile-row already puts 32 independent DRAM reads in flight per core before its single barrier (32·C = 3 520 grid-wide), so batching more tile-rows adds no usable DRAM parallelism while it delays the first push by a whole batch (overlap lamp). L1 is not the binding reason. | | | | | |
| `col_tile` (k, `tensor_col_tiles = D/32`) | independent — column d of the result reads column d of the table only (math.md, F2); **idx is reuse-shared along it** (idx does not vary along D) | `block_col_tiles` | `tensor_col_tiles / col_blocks_per_row` from `select_plan` (= `tensor_col_tiles` whenever tile-rows fill the grid) | `select_plan()` on host → one CT arg to all three kernels (it is the tilize helper's `block_width_tiles` template parameter) | spread over cores only when `tensor_tile_rows` cannot fill the grid (regime `column_blocked`) | knob-turn |
| `table_row` (V) | gathered — not an output axis; rows chosen by index values, nothing to combine | none — a block touches exactly `valid_rows(g) ≤ 32` table rows, fixed by the tile height | — | — | not assigned (data-dependent) | — |

There is **no dependent axis**: every output value is a copy of one input value. "Split the dependent axis + combine" does not exist for this op.

Operand-reuse check (per operand × split axis):

| Operand | Varies along `tile_row`? | Varies along `col_tile`? | Consequence |
|---------|--------------------------|--------------------------|-------------|
| `idx` | yes (disjoint 128-B segments per tile-row) | **no** | splitting `col_tile` re-reads each index segment `col_blocks_per_row` times → reuse-shared by construction → regime row `index_page_multicast` |
| `W` | only through repeated index values (data-dependent) | yes (disjoint segments) | repeat-dedup is regime row `repeated_index_dedup` |
| `out` | yes | yes | written once |

Stall-shadow check (stages that wait on something they do not produce):

| Waiting stage | What it waits for | Work scheduled into the window | Legality |
|---------------|-------------------|--------------------------------|----------|
| `gather_rows_block` row reads | the tile-row's index values (DRAM latency) | the **next** tile-row's `fetch_index_block` is issued after the current block's row reads and before the current block's barrier — one barrier retires both; from the second tile-row on, index latency is off the critical path | tile-rows are independent; no arithmetic, so no FP-order concern |
| `gather_rows_block` barrier | 32 row reads landing | nothing in Phase 0 (one barrier per block); trid-scoped barriers would let the next block's reads ride across it → overlap lamp | — |
| first block of every core | index read → row reads (two dependent DRAM latencies) | nothing independent of the index exists; the index read is the core's first transaction (RB-012) | — |
| `tilize_block`, `store_block` | CB credit from the upstream kernel | nothing (single-role kernels, depth-2 CBs overlap adjacent blocks) | — |

### Buffer-depth knobs

| CB | Depth knob | Phase 0 value | What the depth buys |
|----|------------|---------------|---------------------|
| `cb_index_values` | `INDEX_DEPTH` (slots of one tile-row of indices) | 2 | current tile-row's indices + the prefetched next tile-row's (stall-shadow reorder) |
| `cb_gathered_rows` | `ROWS_DEPTH` (blocks) | 2 | reader fills block k+1 while compute tilizes block k |
| `cb_output_tiles` | `OUT_DEPTH` (blocks) | 2 | compute packs block k+1 while writer drains block k |

All three are host constants in the op module; every capacity and the `col_tile_cap` below derive from them.

### Mechanism caps

| Mechanism | Cap on which extent | Clamp | What happens unclamped |
|-----------|--------------------|-------|------------------------|
| tilize helper block width is a **template** parameter (`tilize_helpers.hpp:194-207`) | `block_col_tiles` is compile-time and identical on every core; no ragged column block | `col_blocks_per_row` is a divisor of `tensor_col_tiles` (select_plan) | a ragged last column block would need a second compute binary or a mis-sized tilize → wrong tiles |
| L1 per core ≤ 1 347 776 B (BB-05) | `block_col_tiles` | `block_col_tiles ≤ col_tile_cap = floor((1_347_776 − INDEX_DEPTH·128) / ((ROWS_DEPTH + OUT_DEPTH)·BLOCK_TILE_ROWS·2048))` = **164** at Phase 0 depths, via `q_l1` in select_plan | launch refused inside the model |
| Blackhole DRAM read alignment: `l1_addr % 64 == noc_addr % 64` (`noc-alignment-in-tt-metal.md` §1) | row-segment width and offset | segments are whole tile-columns: offset `c·block_col_tiles·64` B, length `block_col_tiles·64` B; L1 row slot `r·block_col_tiles·64` B; CB bases are allocated at DRAM alignment (`tt_metal/impl/program/program.cpp:2505-2509`); index segment offset `128·tr`; TILE-index face rows at multiples of 64 B | silent corruption (no watcher) |
| A tile-row holds rows of one image only | `block_tile_rows` | a block is one tile-row `g` of image `g / tile_rows_per_image`; `valid_rows(g) = min(32, S − 32·(g mod tile_rows_per_image))` | rows of two images in one tile → wrong output |
| Fast tilize exists only for Float32/Float16_b (`tilize_helpers.inl:30-35, 65-69`) and is not bit-exact for bf16 | data format of `cb_gathered_rows` / `cb_output_tiles` | both declared `UInt16`; `fp32_dest_acc_en = False` (16-bit DEST), `UnpackToDestMode::Default` — the stock UINT16 tilize configuration (`tilize_multi_core_default_program_factory.cpp:36-38, 146`) | Float16_b → fast tilize → NaN rewritten, −0.0 → +0.0 (stock tilize test excludes exactly these: `test_tilize_with_val_padding.py:1155, 1161, 1177`) |
| tilize helper requires `num_blocks > 0` (`tilize_helpers.inl:95`) | cores in the program | `num_cores = min(total_blocks, C)`; a core with zero blocks is not in the core set | assert / hang |
| Index value is a 32-bit page id | V | none needed (V ≤ 262 144 in production; values outside `[0, V)` are outside the contract and are not checked) | — |

### Regimes

Selection function (host, exact; `C = grid.x · grid.y`):

```python
q_l1    = min(q for q in divisors(K) if K // q <= col_tile_cap)      # L1 clamp (164 at Phase 0)
q_fill  = max(q for q in divisors(K) if q <= max(1, C // R))         # fill the grid, one block per core
col_blocks_per_row = max(q_l1, q_fill)
block_col_tiles    = K // col_blocks_per_row
total_blocks       = R * col_blocks_per_row       # block id = g * col_blocks_per_row + c
num_cores          = min(total_blocks, C)
regime = "whole_row" if col_blocks_per_row == 1 else "column_blocked"
# K = tensor_col_tiles, R = tensor_tile_rows
```

| Regime | Status | Predicate | Block | Data movement vs. minimum (boundary: DRAM; minimum = idx once, each named table row once per index, out once) | What a bigger block buys |
|--------|--------|-----------|-------|---------------------------|--------------------------|
| `whole_row` | **built** | `col_blocks_per_row == 1` ⟺ `K ≤ col_tile_cap` and `C // R < p(K)` (`p(K)` = smallest divisor of K above 1; K = 1 → always). Domain: every shape with `R > C / p(K)` up to rounding, i.e. tile-rows alone fill more than `1/p(K)` of the grid (at C = 110: all perf cases p1–p4 and 14 of the 49 production cases; the other 35 are `column_blocked`) | `BLOCK_TILE_ROWS × K` (one full tile-row) | **minimum**: idx 4N B once; table N reads of `2D` B (one transaction per token — the floor; repeats read again, see dedup row); out `2·D·32·R` B once (tile-padded) | the block is already the coarsest along `col_tile`; per block it pays 1 read barrier, 1 reserve/push on the reader, 1 wait/pop + 1 reserve/push in compute (tilize_block MOP per tile-row; tilize init **once per core**), 1 wait/pop + 1 write flush on the writer, ≤ 1 index fetch per tile-row (prefetched). Bigger `D` per row = fewer reads per byte |
| `column_blocked` | **built** | `col_blocks_per_row > 1` (tile-rows alone leave cores idle: `C // R ≥ p(K)`; or the L1 clamp binds) | `BLOCK_TILE_ROWS × (K / col_blocks_per_row)` | table bytes = minimum, but **N·col_blocks_per_row** reads of `block_col_tiles·64` B; each idx segment read by `col_blocks_per_row` cores (≤ `col_blocks_per_row·128` B per tile-row); all segments of one table row hit that row's single bank | chosen as the **coarsest** column block that still gives every core ≤ 1 block (one wave); a coarser block = fewer, larger row-segment reads and fewer duplicate index reads, at the price of idle cores |
| `index_page_multicast` | **deferred** — stepping stone already built: every core reads its own index segment from DRAM; this successor layers on it by replacing only the producer of `cb_index_values` (`fetch_index_block` is its own named operation with its own CB). Positive reason not built in Phase 0: the cost it removes is bounded to ≤ C requests of ≤ 128 B on one bank per call, each issued as its core's **first** transaction (ahead of any bulk read of that core, RB-012) and prefetched one tile-row ahead thereafter, so it touches only the start of each core's first block; the multicast instead puts a sender read + mcast + semaphore round on every receiver's first block | B = 1 or rank-1 idx (one page, one bank) in either built regime; or `column_blocked` | same as host regime | DRAM idx requests: `num_cores` → `#idx pages`; adds one ≤ `4·S` B (≤ 16 KB) multicast per page to the cores whose blocks lie in it + one semaphore rendezvous (spec rule 8 / RB-010). Mechanism: `ttnn.Mcast` + `McastArgs` (`ttnn/cpp/ttnn/kernel_lib/mcast/kernel/mcast_args.hpp`) | — |
| `row_injector_scatter` | **deferred** — successor of `column_blocked`; reachable because `gather_rows_block` only changes its source (DRAM page → a peer's L1 slot). Positive reason: column_blocked calls are one-wave calls (≤ 1 block per core) whose time *is* each core's serial chain; the scatter inserts a DRAM→injector→peer hop plus a semaphore into that chain | `column_blocked` | `BLOCK_TILE_ROWS × K` on the injector, scattered as `block_col_tiles` segments | removes `(col_blocks_per_row − 1)·N` DRAM reads and the single-bank queue per table row; adds `2·D·N·(col_blocks_per_row−1)/col_blocks_per_row` B of core-to-core traffic | — |
| `repeated_index_dedup` | **deferred** — positive reason: unreachable for every measured case. The suite draws uniform indices: expected repeats inside one core's run at p1 ≈ 608²/(2·248 320) ≈ 0.74 per core; a cross-core dedup needs a global sort (a second pass over idx). The gather's per-row read is the unit dedup would skip, so the structure stays reachable | — | — | saves `2D` B per repeated index within a core's run | — |
| `column_only_split` | **rejected** — superseded by `whole_row` + `column_blocked`: it re-reads every index `col_blocks_per_row` times and multiplies row reads on shapes whose whole rows already fill the grid | — | — | N·cols reads, idx × cols | — |
| `token_only_single_wave` (R < C on ≤ R cores) | **rejected** — superseded by `column_blocked`; spec rule 6 (cores are the blocking decision's choice, not the shape's) | — | — | minimum bytes, but R ≤ 32 cores for decode calls | — |
| route `float16b_fast_tilize` | **rejected** — superseded by the UInt16 tilize route: rewrites NaN payloads and the sign of zero (Mechanism caps, row 5) | — | — | — | — |
| route `dm_face_permute` (L1→L1 32-B copies, `local_copy_helpers_dataflow.hpp:57`) | **rejected** — dead end: 64 NoC copies per tile (5.2 M at p1) on the same RISCs that issue the gather; the engine tilize is otherwise idle hardware | — | — | — | — |

Production plans at C = 110 (`ROWS_DEPTH = OUT_DEPTH = INDEX_DEPTH = 2`):

| idx shape | table | R | K | regime | `block_col_tiles` (segment) | blocks | cores | max blocks/core | L1/core (B) |
|-----------|-------|---|---|--------|----------------------------|--------|-------|-----------------|-------------|
| [16,4096] | 248320×1280 (p1) | 2048 | 40 | whole_row | 40 (2560 B) | 2048 | 110 | 19 | 327 936 |
| [8,4096] | 248320×1280 (p2) | 1024 | 40 | whole_row | 40 | 1024 | 110 | 10 | 327 936 |
| [16,1024] | 248320×1280 (p3) | 512 | 40 | whole_row | 40 | 512 | 110 | 5 | 327 936 |
| [1,3844] | 248320×1280 (p4) | 121 | 40 | whole_row | 40 | 121 | 110 | 2 | 327 936 |
| [1,4096] | 248320×1280 | 128 | 40 | whole_row | 40 | 128 | 110 | 2 | 327 936 |
| [16,128] | 248320×1280 | 64 | 40 | whole_row | 40 | 64 | 64 | 1 | 327 936 |
| [1,1024], [1,1000], [8,128] | 248320×1280 | 32 | 40 | column_blocked | 20 (1280 B) | 64 | 64 | 1 | 164 096 |
| [2,128], [1,256] | 248320×1280 | 8 | 40 | column_blocked | 4 (256 B) | 80 | 80 | 1 | 33 024 |
| [1,129] / [1,100], [1,128] / [1,77] | 248320×1280 | 5 / 4 / 3 | 40 | column_blocked | 2 (128 B) | 100 / 80 / 60 | same | 1 | 16 640 |
| [1,64], [1,33] / [1,1], [1,8], [1,16], [1,31] | 248320×1280 | 2 / 1 | 40 | column_blocked | 1 (64 B) | 80 / 40 | same | 1 | 8 448 |
| [16,4096] … [8,1024], [1,4096], [16,128] | 262144×64 | ≥ 64 | 2 | whole_row | 2 (128 B) | R | min(R,110) | ≤ 19 | 16 640 |
| R ≤ 32 shapes | 262144×64 | ≤ 32 | 2 | column_blocked | 1 (64 B) | 2R | 2R | 1 | 8 448 |
| [1,512] | 131072×2560 | 16 | 80 | column_blocked | 16 (1024 B) | 80 | 80 | 1 | 131 328 |
| [1,256] TILE idx | 131072×2560 | 8 | 80 | column_blocked | 8 (512 B) | 80 | 80 | 1 | 65 792 |
| [1] TILE idx | 131072×2560 | 1 | 80 | column_blocked | 1 (64 B) | 80 | 80 | 1 | 8 448 |
| [1,1] | 32×5120 | 1 | 160 | column_blocked | 2 (128 B) | 80 | 80 | 1 | 16 640 |
| [1,1] | 1×256×128 | 1 | 4 | column_blocked | 1 (64 B) | 4 | 4 | 1 | 8 448 |

Regime-pinned tests (required; both regimes must run on any grid): `whole_row` — `[16,1024]` and `[1,3844]` × 248320×1280 (whole_row for every C < 242); `column_blocked` — `[1,77]` × 248320×1280 and `[1,1]` × 262144×64 (column_blocked for every C ≥ 6). The op exports `select_plan(input_tensor, weight)` so tests and BB-08 reporting read the selected plan.

### Traffic ranking

Counts are DRAM crossings per tensor (the dominant tier; no cross-core traffic in any built regime).

| Rank | Candidate split | idx | table | out | Cross-core | Verdict |
|------|-----------------|-----|-------|-----|------------|---------|
| 1 | `tile_row` only (whole_row) | 1× (4N B, one 128-B segment per tile-row) | 1× per index, N reads of 2D B | 1× | none | **cheapest**; chosen wherever it fills the grid |
| 2 | `tile_row` × `col_tile` (column_blocked) | `col_blocks_per_row`× (bytes ≤ 128 per tile-row per core) | same bytes, `col_blocks_per_row`× the transactions | 1× | none | chosen only when rank 1 leaves cores idle (occupancy step); one block per core keeps the multiplier at the smallest value that fills the grid |
| 3 | rank 2 + index multicast | 1× | as rank 2 | 1× | ≤ 16 KB mcast per idx page | deferred row |
| 4 | `col_tile` only | `cols`× on every shape | `cols`× transactions on every shape | 1× | none | rejected |
| — | dependent-axis split + combine | — | — | — | — | does not exist (no dependent axis) |

Occupancy did not override the ranking: rank 2 is used only where rank 1 cannot reach the grid, and it moves the same bytes; its added cost is transaction count and duplicate 128-B index reads, recorded in its row.

### Block schedule

Logical schedule (reader, compute and writer realize their parts asynchronously):

```cpp
fetch_index_block(tile_row_of(first_block_this_core));            // reader prologue: core's first transaction
for (uint32_t block_idx = 0; block_idx < num_blocks_this_core; ++block_idx) {
    gather_rows_block(block_idx);    // reader: issue valid_rows row-segment reads into cb_gathered_rows
    fetch_index_block(next_tile_row); // reader: only if the next block opens a new tile-row; issued BEFORE
                                      //         gather's barrier so one barrier retires both
    tilize_block(block_idx);         // compute: realized as ONE helper call over all blocks of the core
    store_block(block_idx);          // writer
}
```

| Block operation | Block shape | Resident across it | Intended frequency of fixed costs |
|-----------------|-------------|--------------------|-----------------------------------|
| `fetch_index_block` | 1 tile-row: `valid_rows(g)` uint32 indices (≤ 128 B) of image `g / tile_rows_per_image`, tokens from `32·tr` | the slot stays live until `gather_rows_block` of the last column block of that tile-row has *issued* its reads | one fetch per distinct tile-row of the core (1 read for a ROW_MAJOR idx, 2 face-row reads for a TILE idx; 1 fetch per core in column_blocked); no barrier of its own after the prologue (shares gather's) |
| `gather_rows_block` | `BLOCK_TILE_ROWS × block_col_tiles`: `valid_rows(g)` row segments of `block_col_tiles·64` B from `W[idx, c·block_col_tiles·32 : (c+1)·block_col_tiles·32]`, row r at slot offset `r·block_col_tiles·64` | the reserved `cb_gathered_rows` slot until pushed | 1 `cb_reserve_back(block_col_tiles)` + `valid_rows` reads + **1 read barrier** + 1 `cb_push_back(block_col_tiles)` per block (nominal count even when `valid_rows < 32`) |
| `tilize_block` | `BLOCK_TILE_ROWS × block_col_tiles`, UInt16 | nothing across blocks | `compute_kernel_hw_startup` once per kernel; tilize init/uninit **once per core per call** (one helper call, `num_blocks = num_blocks_this_core`); 1 wait/pop + 1 reserve/push of `block_col_tiles` per block |
| `store_block` | `block_col_tiles` tiles → output pages `g·K + c·block_col_tiles + j` | nothing across blocks | `block_col_tiles` tile writes + 1 `noc_async_writes_flushed` + 1 `cb_pop_front` per block; **1 write barrier per kernel** at the end |

No multicast, semaphore or cross-core barrier exists in Phase 0. Each named operation gets its own `DeviceZoneScopedN` zone in the zone build only (BB-21); every core has the same role set (reader/compute/writer), which is what the role dump reports.

### Perf lamps

| Lamp | Why the default may be wrong here | Nearby alternative to measure |
|------|-----------------------------------|-------------------------------|
| Overlap (reader barrier granularity) | One barrier per block drains the core's read pipe; at K = 2 (128-B rows) a block is only 4 KB, so each drain idles the wire for a DRAM latency | `ROWS_DEPTH = 4` with transaction-id-scoped barriers keeping 2 blocks in flight (`trid_double_issue`; `noc_async_read_barrier_with_trid`, `dataflow_api.h:2602`) |
| Grid synchronization | `column_blocked` spreads decode calls over 40–100 cores with 1–4 tiles each; per-core launch + two dependent DRAM latencies + single-bank queueing of segments of one table row may dominate | cap `col_blocks_per_row` so `block_col_tiles ≥ 4` (fewer cores, larger segments); judge by host time of blocking trace replays, not profiled kernel time (RB-053) |
| Load balance (whole_row at `C/2 < R < 2C`) | p4 (R = 121) and `[1,4096]`×1280 (R = 128, 112 calls) give 11/18 cores a second block; `[16,128]` (R = 64) uses 64 of 110 cores | `col_blocks_per_row = 2` or `4` for these shapes (242–512 blocks), same kernels |
| Bank stagger (writer) | K = 40 ≡ 0 (mod 8): every whole-row block's first output page is on bank 0 and all cores walk banks in lockstep | rotate the tile order inside `store_block` by core index (`bank_stagger`) |
| Reader issue rate | K = 2: 32 reads per 2 output tiles; on `[16,4096]`×64 one RISC issues ~600 reads per core and may be issue-bound | split each block's row reads across NCRISC and BRISC (`split_reader`) — needs a second producer handshake, so measure first |

## Dataflow Strategy

| Stage | Format | Mechanism | Notes |
|-------|--------|-----------|-------|
| idx DRAM → `cb_index_values` | uint32, `valid_rows(g)·4` B | `noc_async_read(idx_accessor.get_noc_addr(page = image, offset = 128·tr), slot, 4·valid_rows)` | ROW_MAJOR rank 2: page = image row; rank 1: page 0. Never reads past position S−1 of a page |
| idx (TILE, refinement) → `cb_index_values` | uint32 | per tile-row: tile page `(b/32)·ceil(S/32) + tr`; face row `rb = b mod 32`: tokens 0–15 at byte `((rb/16)·2)·1024 + (rb mod 16)·64`, tokens 16–31 at `+1024`; read `min(16, valid)·4` B then `max(0, valid−16)·4` B into slot bytes 0 and 64 | only logical positions are read; padding positions never address the table (rule 3). Rank-1 `[N]`: b = 0 |
| index values → row addresses | uint32 in L1 | reader RISC loads via `addr_to_l1_ptr` (`l1_helpers.hpp:30`) | values never reach the host, args, or cache key (rule 1) |
| W DRAM → `cb_gathered_rows` | bf16 bytes, row-major, row pitch `block_col_tiles·64` B | per row r < `valid_rows`: `noc_async_read(w_accessor.get_noc_addr(idx[r], c·block_col_tiles·64), slot + r·block_col_tiles·64, block_col_tiles·64)`; one barrier per block | `[1, V, D]` weight: same accessor, V pages (no reshape). Rows ≥ `valid_rows` keep stale bytes → unspecified output padding |
| `cb_gathered_rows` → `cb_output_tiles` | **UInt16** on both CBs (the bf16 bits) | `compute_kernel_lib::tilize<block_col_tiles, cb_gathered_rows, cb_output_tiles>(num_blocks_this_core)`; 16-bit DEST | integer passthrough: every bf16 pattern incl. NaN payloads, ±0, subnormals, ±inf survives |
| `cb_output_tiles` → out DRAM | 2048-B tiles | `noc_async_write(slot + j·2048, out_accessor.get_noc_addr(g·K + c·block_col_tiles + j), 2048)` | whole-tile writes (`tile_reorder`); NoC1 |

Deferred-regime contract (`index_page_multicast`): one injector core per idx page reads the page once into its `cb_index_values`-sized landing area and multicasts it to the rectangle of cores whose blocks lie in that page (`ttnn.Mcast` host config + `McastArgs` sender/receiver, one semaphore); receivers take their 128-B slice. Only the producer side of `fetch_index_block` changes.

## Work Distribution

| Field | Value |
|-------|-------|
| Work unit | one block = (tile-row `g`, column block `c`), `block id = g·col_blocks_per_row + c` |
| Grid | `device.compute_with_storage_grid_size()` (110 here); `C = grid.x·grid.y` |
| Per-core work | `ttnn.split_work_to_cores(grid, total_blocks, row_wise=True)`: group 1 gets `ceil`, group 2 `floor`; blocks assigned as contiguous id ranges in row-wise core order → `first_block_this_core`, `num_blocks_this_core` |
| Remainder | block remainder via the two groups; token remainder per image: `tile_rows_per_image = ceil(S/32)`, `valid_rows(g) = min(32, S − 32·(g mod tile_rows_per_image))` (never `floor(B·S/32)`) |
| Core ranges (BB-17) | one kernel per RISC over `all_cores` (≤ 2 rectangles: full rows + one partial row); group differences are runtime args only |
| Regime selection | `select_plan` above; tests pinned per regime (Regimes section) |

## Host Side

| Concern | Decision |
|---------|----------|
| Entry sequence | `validate(...)` first line → `select_plan` → `ttnn.allocate_tensor_on_device(Shape(idx.shape + [D]), bfloat16, TILE, device, DRAM interleaved)` → cached program artifacts → `ttnn.generic_op([input_tensor, weight, output], ...)` (output last) |
| Program cache (BB-07) | module-level dict keyed by `(tuple(idx.shape), idx.layout, tuple(weight.shape), weight.dtype, grid.x, grid.y)`; on a hit nothing is rebuilt. Cached artifacts hold **no buffer address**: use the ProgramSpec path (`TensorParameter` + `TensorBinding`, pattern of `ttnn/ttnn/operations/toy_spec_single_tensor/toy_spec_single_tensor.py`) so each call binds its own tensors; a ProgramDescriptor path is acceptable only if it patches the three address slots per call without rebuilding |
| Trace (BB-16) | the only allocation is the output in the entry point; no kernel build after the first call of a key |
| Plan report (BB-08) | `select_plan(input_tensor, weight)` exported from `ttnn.operations.g6_embedding`; returns `regime`, `col_blocks_per_row`, `block_col_tiles`, `tensor_tile_rows`, `tensor_col_tiles`, `total_blocks`, `num_cores`, `l1_bytes_per_core` |
| Knob sources | `BLOCK_TILE_ROWS`, `INDEX_DEPTH`, `ROWS_DEPTH`, `OUT_DEPTH`, `L1_BUDGET_BYTES = 1_347_776` are module constants; `block_col_tiles` (CT, all kernels) and everything else derive from `select_plan` |
| Compute config | `fp32_dest_acc_en = False`, `dst_full_sync_en = False`, default unpack-to-dest mode; math fidelity irrelevant |

## Circular Buffers

| Semantic Name | Index | Page Size | Num Pages | Sizing rationale | Format | Producer | Consumer | Lifetime |
|---------------|-------|-----------|-----------|------------------|--------|----------|----------|-----------|
| `cb_index_values` | 0 | 128 B (32 uint32) | `INDEX_DEPTH` | spans nothing of `col_tile` (idx does not vary along D); streams `tile_row` with a window of `INDEX_DEPTH` slots | UInt32 | reader | reader (private scratch; ordered by the reader's own read barrier, no CB credit) | whole kernel |
| `cb_gathered_rows` | 1 | 2048 B (one UInt16 tile) | `ROWS_DEPTH · BLOCK_TILE_ROWS · block_col_tiles` | spans `col_tile` (`block_col_tiles`) and the ≤ 32 gathered rows of a tile-row; streams `tile_row` | UInt16 | reader | compute | whole kernel, block-pipelined |
| `cb_output_tiles` | 2 | 2048 B | `OUT_DEPTH · BLOCK_TILE_ROWS · block_col_tiles` | spans `col_tile`; streams `tile_row` | UInt16 | compute | writer | whole kernel, block-pipelined |

Every push and pop on `cb_gathered_rows` / `cb_output_tiles` is exactly `block_col_tiles` pages, so the ring closes (capacity = depth × quantum). Full ledger: `l1_ledger.md`.

## Block Operation Realization

| # | Block operation | Block shape | Helper? | Input CB (semantic name, pages, state) | Output CB (semantic name, pages) | CB state after |
|---|-----------------|-------------|---------|----------------------------------------|----------------------------------|----------------|
| 1 | `fetch_index_block` | ≤ 32 indices of one tile-row | no (raw NoC read) | idx DRAM page | `cb_index_values`, 1 slot (128 B) | slot valid after the next read barrier; reused 2 tile-rows later |
| 2 | `gather_rows_block` | `1 × block_col_tiles` | no (raw NoC reads, index-driven) | `cb_index_values` slot of tile-row g (read by RISC) + W DRAM | `cb_gathered_rows`, `block_col_tiles` pages pushed | — |
| 3 | `tilize_block` | `1 × block_col_tiles` (one helper call spans `num_blocks_this_core` of them) | yes: `compute_kernel_lib::tilize` | `cb_gathered_rows`, `block_col_tiles` pages waited/popped per block | `cb_output_tiles`, `block_col_tiles` pages pushed per block | — |
| 4 | `store_block` | `block_col_tiles` tiles | no (raw NoC writes) | `cb_output_tiles`, `block_col_tiles` pages waited, popped after `noc_async_writes_flushed` | out DRAM pages | — |

## API Mapping

| Block operation | Type | Function | File:Line | Template Params / Args | Input CB | Output CB | Which params are block knobs |
|-----------------|------|----------|-----------|------------------------|----------|-----------|------------------------------|
| `tilize_block` | helper | `compute_kernel_lib::tilize` | `ttnn/cpp/ttnn/kernel_lib/tilize_helpers.hpp:194-207` (impl `tilize_helpers.inl:75-251`; standard path `tilize_init` `:185`, `tilize_block` `:230`) | `<block_col_tiles, cb_gathered_rows, cb_output_tiles>` (other params default; `NoReconfigure` permitted since formats never change), `num_blocks = num_blocks_this_core`, symmetric mode (no `total_input_pages`) | `cb_gathered_rows` | `cb_output_tiles` | `block_width_tiles = block_col_tiles`; `num_blocks = num_blocks_this_core` |
| (prerequisite) | helper | `compute_kernel_hw_startup(icb, ocb)` | `tt_metal/hw/inc/api/compute/compute_kernel_hw_startup.h:113` | `(cb_gathered_rows, cb_output_tiles)` | — | — | — |
| `fetch_index_block` | raw_api | `TensorAccessor::get_noc_addr(page_id, offset)` + `noc_async_read` | `tt_metal/hw/inc/api/tensor/tensor_accessor.h:165`; `tt_metal/hw/inc/api/dataflow/dataflow_api.h:547` | page = image (RM) / tile page (TILE), offset `128·tr` (RM), size `4·valid_rows` | idx DRAM | `cb_index_values` | slot count `INDEX_DEPTH` |
| `gather_rows_block` | raw_api | `get_noc_addr(idx_value, c·block_col_tiles·64)` + `noc_async_read` + `noc_async_read_barrier`; `cb_reserve_back` / `get_write_ptr` / `cb_push_back`; index loads via `addr_to_l1_ptr` | `tensor_accessor.h:165`; `dataflow_api.h:547, 1921, 406, 325, 208`; `ttnn/cpp/ttnn/kernel_lib/l1_helpers.hpp:30` | segment `block_col_tiles·64` B, `valid_rows` reads per block | `cb_index_values` | `cb_gathered_rows` | `block_col_tiles` (segment and push size) |
| `store_block` | raw_api | `cb_wait_front` / `get_read_ptr` / `noc_async_write` / `noc_async_writes_flushed` / `cb_pop_front`; final `noc_async_write_barrier` | `dataflow_api.h:479, 346, 876, 1980, 260, 1951` | page `g·K + c·block_col_tiles + j` | `cb_output_tiles` | out DRAM | `block_col_tiles` (wait/pop size) |

Helpers considered and rejected for the raw_api entries:

| Block operation | Candidate | Mismatch (file:line) | What would close the gap |
|-----------------|-----------|----------------------|--------------------------|
| `gather_rows_block` | `dataflow_kernel_lib::read_sticks_for_tilize` | page id is `start_page + block_row + row` — contiguous sticks only (`tilize_helpers_dataflow.inl:121`); the gather's page id is a data-dependent value read from L1. Its reserve → reads → one barrier → push structure (`:117-127`) is exactly the planned one | an index-driven variant taking a page-id source (functor or L1 index slot) |
| `fetch_index_block` | `read_sticks_for_tilize` | pushes `width_in_tiles` tile pages into a tilize CB with its own barrier (`tilize_helpers_dataflow.inl:117-127`); the index slot is reader-private scratch read by the RISC, not a tilize input | none needed (one read) |
| `fetch_index_block` | `dataflow_kernel_lib::generate_index_tile` | generates position tiles for sort ops; reads nothing (`index_tile_dataflow.hpp:74`) | — |
| `store_block` | `dataflow_kernel_lib::write_sticks_after_untilize` | writes row-major sticks from an untilize CB (`tilize_helpers_dataflow.hpp:125-131`); output here is tile pages. No tile-page writer helper exists in `ttnn/cpp/ttnn/kernel_lib/` | a block tile-writer helper |

## Key Risks and Gotchas

| Risk | Why it bites here | Mitigation in this design |
|------|-------------------|---------------------------|
| Special values altered by the tilize path (R3) | declaring the staging CB `bfloat16` (the natural choice) makes the helper pick fast tilize through float registers (`tilize_helpers.inl:30-35, 65-69`); NaN payloads and −0.0 change, randn data never shows it | `cb_gathered_rows` and `cb_output_tiles` are **UInt16**, `fp32_dest_acc_en = False` — the stock UINT16 tilize configuration, tested over the full 16-bit range (`data_movement_test_utils.py:21, 32`); acceptance test grades special rows bit for bit |
| Partial last tile-row (23 cases) | reading 32 indices past `S` reads the next image's page or page padding; pushing `valid_rows` pages breaks CB counts | read exactly `4·valid_rows` index bytes and `valid_rows` table segments; always push/pop `block_col_tiles` pages |
| Index values on the host or in the cache (R7, rule 1) | a replayed trace brings new index values | no index value is read on the host or enters args/key; acceptance test re-calls with new indices on the same program |
| TILE-index padding as addresses (R2, refinement) | test tensors hold zeros (row 0 is valid) in padding; the model holds garbage | only face-row bytes of logical positions are read (Dataflow Strategy); golden loose case poisons the padding |
| Prefetch past the last block | an index prefetch for a tile-row the core does not own reads out of range and leaves a transaction pending at return (BB-03) | prefetch only when `block_idx + 1 < num_blocks_this_core` and the next block opens a new tile-row |
| Writer pops before its writes leave L1 | compute overwrites a slot still being read by the NoC → run-to-run differences (BB-06) | `noc_async_writes_flushed` before every `cb_pop_front`; `noc_async_write_barrier` before return |
| Regime depends on grid size | `whole_row` vs `column_blocked` is a function of C | regime-pinned tests on shapes whose regime is fixed for every real grid (Regimes section) |
| Per-image tile padding of the output | `[B, S, D]` with `S % 32 ≠ 0` pads each image separately | output page of `(g, k)` is `g·K + k` with `g = image·ceil(S/32) + tr` |
| `[1, V, D]` weight | a reshape would violate BB-02 | the accessor is built on the rank-3 buffer as is: V pages of `2·D` B |
| Duplicate index reads on one bank (rule 8) | B = 1 / rank-1 idx is one page in one bank; column_blocked reads each segment `col_blocks_per_row` times | index read is each core's first transaction (RB-012); multicast is the deferred `index_page_multicast` row |
| Program rebuilt per call (BB-07) | Python-side descriptor building per call costs host time on every untraced call | keyed artifact cache with no buffer address in it (Host Side) |
