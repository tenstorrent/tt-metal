# Operation Design: g6_permute

## Overview

| Field | Value |
|-------|-------|
| Classification | data_movement — no compute kernel, no arithmetic |
| Goal | `out = x.permute(dims)` as one device program, bit-exact for bfloat16. Phase 0 builds dims `(0, 2, 1, 3)` on bfloat16 TILE tensors interleaved in DRAM (the eight Qwen shapes `[1, A, B, 64]`, 2176 of 2203 traced calls) and records every other TARGET regime with its status |
| Math | `out[o_0..o_{r-1}] = x[i_0..i_{r-1}]`, `i_{dims[k]} = o_k`. Phase 0 instance: `out[n, b, a, w] = x[n, a, b, w]` — rows of W move whole (F1, `docs/explorations/g6_permute/math.md#reformulations`) |
| References | request (= `docs/explorations/g6_permute/spec/g6_permute.txt`); `docs/explorations/g6_permute/{DOSSIER,math,tensix,risks,lessons}.md`; `eval/golden_tests/g6_permute/{feature_spec,helpers,test_regression}.py`; `.claude/references/{blocking-model,l1-footprint-discipline,noc-alignment-in-tt-metal}.md`; catalog `ttnn/ttnn/operations/examples/master.md` (entries listed under *Catalog entries*) |

## Parameters

| Name | Type | Required | Valid Range | Default | CT/RT |
|------|------|----------|-------------|---------|-------|
| `input_tensor` | `ttnn.Tensor` | yes | on device, interleaved, 32×32 tiles when TILE; Phase 0: rank 4, bfloat16, TILE, DRAM, `W % 32 == 0`, `H % 32 != 0` | — | buffer address: common RT arg; shape geometry: CT args |
| `dims` | sequence of `int`, positional | yes | length = rank; entries in `[-r, r-1]`; a permutation after wrapping `d < 0 → d + r`. Phase 0: canonical `(0, 2, 1, 3)` | — | host only: selects the regime / index map; part of the program cache key |

No `memory_config`, `dtype`, output tensor or compute config: the output takes the input's dtype, layout and placement.

## Tensors

### Input

| Property | Requirement |
|----------|-------------|
| Shape | rank 4 `[N, A, B, W]` (Phase 0); production `N = 1`, `(A, B) ∈ {(1,16),(16,1),(6,16),(16,6),(1,8),(8,1),(6,8),(8,6)}`, `W = 64` |
| Dtype | bfloat16 |
| Layout | TILE (32×32 tiles, 16×16 faces) |
| Memory | interleaved, DRAM |

### Output

| Property | Value |
|----------|-------|
| Shape | `[x.shape[d] for d in dims]` → Phase 0 `[N, B, A, W]` |
| Dtype | input's |
| Layout | input's |
| Memory | input's memory config (interleaved, same buffer type), allocated by the entry point |

Padding: the input's padding rows (`b ≥ B` inside a tile) may hold anything and never reach a logical output position; the output's padding rows (`a ≥ A` inside a tile) are not written and hold whatever the fresh buffer held (D7 of the dossier).

## Blocking Model

Semantics: `.claude/references/blocking-model.md`. Everything below this section realizes it.

### The Phase 0 index map in tile terms

| Quantity | Definition |
|----------|------------|
| `tensor_n, tensor_a, tensor_b, tensor_w` | logical input dims 0..3 |
| `tensor_bt` | `ceil(tensor_b / 32)` — input tile rows per `(n, a)` |
| `tensor_at` | `ceil(tensor_a / 32)` — output tile rows per `(n, b)` |
| `tensor_wt` | `ceil(tensor_w / 32)` — tile columns (equal in input and output: W is kept) |
| `tensor_in_tiles` | `tensor_n · tensor_a · tensor_bt · tensor_wt` = `input_tensor.buffer_num_pages()` |
| `tensor_out_tiles` | `tensor_n · tensor_b · tensor_at · tensor_wt` = `output_tensor.buffer_num_pages()` |
| input page | `p_in(n, a, bt, wt) = ((n · tensor_a + a) · tensor_bt + bt) · tensor_wt + wt`; tile row `r` holds `b = 32·bt + r` |
| `rows_live(bt)` | `min(32, tensor_b − 32·bt)` — tile rows `r ≥ rows_live(bt)` are input padding |
| output page | `p_out(n, b, at, wt) = ((n · tensor_b + b) · tensor_at + at) · tensor_wt + wt`; tile row `ro` holds `a = 32·at + ro` |
| the move | tile row `r` of input tile `(n, a, bt, wt)` → tile row `ro = a mod 32` of output tile `(n, 32·bt + r, a div 32, wt)` |
| destination stride | consecutive `r` of one input tile land in output pages `tensor_at · tensor_wt` apart, all at the same row `ro` |
| half-row offset | half `h ∈ {0, 1}` (columns 0–15 / 16–31) of tile row `t` is at `face_off(t, h) = ((t div 16)·2 + h) · FACE_BYTES + (t mod 16) · FACE_ROW_BYTES`; faces 0/1 hold rows 0–15, faces 2/3 rows 16–31 (`ttnn/cpp/ttnn/kernel_lib/tile_column_dataflow.hpp:10-15`). bfloat16: `FACE_BYTES = 512`, `FACE_ROW_BYTES = 32` |
| live prefix | `prefix_bytes(rows) = top·2·FACE_BYTES + FACE_BYTES + (rows − 16·top)·FACE_ROW_BYTES`, `top = (rows − 1) div 16` — the shortest byte prefix of a tile holding both halves of rows `0..rows−1` (16 rows → 1024 B, 32 rows → 2048 B, 1 row → 544 B) |

A row of an input tile and its destination row are never adjacent to their neighbours on both sides at once (row `r+1` goes to another output tile; output row `ro+1` comes from another input tile), so the **32 B half-row is the irreducible transfer unit** of this permutation in TILE layout.

### Axes

The four tile-grid axes are all independent and the scatter treats every input tile identically (its destinations are a closed-form function of its page id), and they are contiguous in input page order — so they **fold honestly into one axis, `in_tile`** (the input page id). The block is a run of `block_in_tiles` consecutive input pages.

| Axis | Character (+ one-clause reason) | Extent knob | Phase 0 value | Knob source | Core-assignment | Later unlock |
|------|--------------------------------|-------------|---------------|-------------|-----------------|--------------|
| `n` (in dim 0 → out dim 0) | independent — the batch index is copied | folded into `in_tile` → `block_in_tiles` | fold | host (fold order) | contiguous `in_tile` ranges | knob-turn |
| `a` (in dim 1 → out dim 2) | independent — each `a` owns row slot `a mod 32` of its output tiles; no two `a` write the same byte | folded into `in_tile` | fold | host | contiguous `in_tile` ranges | knob-turn |
| `bt` (tile rows of in dim 2 → out dim 1) | independent — each live row `b` goes to its own output tile | folded into `in_tile` | fold | host | contiguous `in_tile` ranges | knob-turn |
| `wt` (W tile columns, kept) | independent — a half-row keeps its 16 columns | folded into `in_tile` | fold | host | contiguous `in_tile` ranges | knob-turn |
| `in_tile` (fold of `n, a, bt, wt`) | independent — no input tile depends on another | `block_in_tiles` | `min(max_core_in_tiles, block_cap)` = **1 for every production shape**: `tensor_in_tiles ≤ 32` is below the grid, so each core's whole assignment is one tile (bound 2 of blocking-model §1, not a collapsed knob) | host program-descriptor function, passed as CT arg; every page count and trip count derives from it | `split_work_to_cores(core_grid, tensor_in_tiles, row_wise=True)` over the full grid | knob-turn (block, grid) |
| `r` (rows of one input tile, `b mod 32`) | independent — each live row has its own destination | `rows_live(bt)` — the whole live tile, never cut | `min(32, tensor_b − 32·bt)`: 16, 1, 16, 6, 8, 1, 8, 6 (cases 301–308) | derived from `tensor_b` (CT) | not split across cores: the tile is the DRAM read unit; a sub-tile split needs 64 B-aligned row-pair reads, two reads per core per tile | scheme-change |
| `h` (half-row: tile columns 0–15 / 16–31) | independent — each half lands in the same half of the output row | 2 (whole row) | 2 | tile geometry (CT) | never split: 32 B is the transfer unit | none (transfer unit) |

Every stage's data: there is one stage (the move) and no intermediate result; no core collects or coordinates, so no stage computes on one core while others wait.

### Buffer-depth knobs

| CB | Depth knob | Phase 0 value | What the depth buys |
|----|------------|---------------|---------------------|
| `cb_input_tiles` | `input_depth` (host constant, CT) | 2; the allocated depth is `min(input_depth, max_blocks_per_core)`, which is 1 for every production shape (one block per core) | reading block `k+1` while block `k` is scattered, on cores that own more than one block |

### Mechanism caps

| Mechanism | Cap on which extent | Clamp | What happens unclamped |
|-----------|--------------------|-------|------------------------|
| DRAM read alignment — Blackhole compares local and NoC address mod 64 (`tt_metal/hw/inc/internal/tt-1xx/blackhole/noc/noc_parameters.h:379`; `noc-alignment-in-tt-metal.md` §1) | the byte window read from an input tile | every read starts at the tile's page base into a CB slot base; CB bases are aligned to the DRAM alignment (`tt_metal/impl/program/program.cpp:2505-2509`) and slots are 2048 B apart → the read is the prefix `[0, prefix_bytes(rows_live(bt)))`, never a mid-tile window | a 32 B read at an odd face row lands 32 B off mod 64: shifted data, or a watcher stop |
| DRAM write / L1 alignment — mod 16 (`noc_parameters.h:375-380`) | the half-row write | 32 B half-rows at `face_off(t, h) ≡ 0 (mod 16)` on both sides | holds for bfloat16; the bfloat8_b half-row carries a 1-byte exponent that no 16 B-aligned write can move — why bfloat8_b is a separate regime |
| Tile geometry (32×32 tile, 16×16 faces) | rows per tile, half-row size | `validate()` raises `ValueError` for an input whose tile is not 32×32; `FACE_BYTES`, `FACE_ROW_BYTES` derived on host from the input's tile size and element size | face offsets for the wrong geometry move wrong bytes silently |
| L1 left by the model: 1347776 B per core, output's own L1 pages included (BB-05, spec rule 10) | `block_in_tiles` | `block_cap = floor((L1_OP_BUDGET_BYTES − out_l1_bytes_per_core) / (input_depth · in_page_bytes))`; Phase 0 (output in DRAM): `floor(1347776 / 4096) = 329` tiles | refused launch inside the model only; a test that runs alone has the whole L1 |
| Input tile padding (h_non_aligned) | rows scattered per tile | the scatter walks exactly `rows_live(bt)` rows | padding (garbage in the model) reaches logical output rows (risk R4) |
| `NOC_MAX_BURST_SIZE` | none binds: reads ≤ 2048 B, writes 32 B, all single-packet | — | — |

### Regimes

Selection is structural, by `(layout, canonical dims, dtype)` — exact, host-checkable, grid-independent. Phase 0's `validate()` admits only row 1.

| Regime | Status | Predicate | Block | Data movement vs. minimum | What a bigger block buys |
|--------|--------|-----------|-------|---------------------------|--------------------------|
| **`rows_tile_scatter`** | **built** (Phase 0) | `layout == TILE ∧ dims == (0,2,1,3) ∧ dtype == bfloat16` | `block_in_tiles` input tiles × `rows_live(bt)` rows × 2 halves | Boundary DRAM; minimum = every logical input value read once, every logical output value written once. **x**: read once, the live prefix of each tile in one transaction — above the minimum only by padding rows inside the prefix, ≤ 15 × 32 B per tile (B = 1: 544 B read for 64 B live). **out**: exactly the logical bytes, once, as 32 B half-rows; output padding rows never written. Cross-core: 0. Domain: every dims-(0,2,1,3) bfloat16 TILE shape (tile_aligned and w_non_aligned too: half-rows carry padding columns into output padding columns because W is kept) | intended once per block: `cb_reserve_back` + `cb_push_back` (reader), `cb_wait_front` + `cb_pop_front` (writer), one `noc_async_read_barrier`, one `noc_async_writes_flushed`. Per-tile (one read) and per-row (one destination setup, two writes) costs are set by the regime and do not shrink with the block. Production: one block per core, so each fixed cost is paid once per core |
| `rows_tile_group_assemble` | deferred — `rows_tile_scatter` computes every shape this one would serve; this one wins only where one whole-tile DRAM write beats 2·rows 32 B DRAM writes, i.e. output tiles with many live rows (`tensor_a ≥ 32`) on a DRAM-bound shape. Every production output tile holds ≤ 16 live rows and the largest production call writes 12 KB. Reachable: same reader and `cb_input_tiles`; the writer gains `cb_output_tiles`, assembles rows in L1 with self-aimed NoC reads, writes whole tiles; the host unit becomes the group | `tensor_a ≥ 32` (crossover to be set by measurement) | closed group `(n, at, bt) × block_wt`: ≤ 32 input tiles + ≤ 32 output tiles per `wt` | x once; out once as whole tiles including padding rows (2048 B per output tile); the 2·rows 32 B DRAM writes become 2·rows 32 B core-local copies + one 2 KB write per output tile; occupancy limited to `tensor_n·tensor_at·tensor_bt·tensor_wt / block_wt` groups | amortizes one output-tile write over up to 32 rows; costs an output CB of `block_wt · 32` tiles |
| `rows_tile_gather` | rejected — superseded by `rows_tile_scatter` | — | one output tile | x: a 32 B half-row read must keep its offset mod 64, which odd/even row pairs (`r`, `ro`) violate → 64 B bounce read + local copy per half-row: 2× read transactions, 2× read bytes; out as whole tiles with padding | — |
| `rows_tile_reread` | rejected (dead end) — superseded by `rows_tile_scatter` | — | one output tile, its input tiles read whole | x crosses DRAM `rows_live` times (up to 32×) — once per output tile of its group | — |
| `rows_tile_mcast` | rejected — superseded by `rows_tile_scatter` | — | output tiles split, input tiles multicast to their group | x once from DRAM, then multicast to ≤ 32 cores that each use 1/32 of it: 32× cross-core bytes where scatter has none | — |
| `rows_tile_bfp8` | deferred — no INPUTS entry pairs bfloat8_b with dims (0,2,1,3). Reachable on `rows_tile_group_assemble`'s structure | `layout == TILE ∧ dims == (0,2,1,3) ∧ dtype == bfloat8_b` | as `rows_tile_group_assemble` | a half-row is 16 mantissa bytes (NoC-movable) + 1 exponent byte (not movable at 16 B alignment) → assemble output tiles in L1 (exponent section by RISC-V byte stores), write whole; bit-exact, the 16-value group is preserved | as `rows_tile_group_assemble` |
| `rows_rm_page_remap` | deferred — no INPUTS entry pairs ROW_MAJOR with dims (0,2,1,3) | `layout == ROW_MAJOR ∧ dims == (0,2,1,3)` | `block_in_pages` pages | page = one row of W, moves whole: page `(n,a,b)` → page `(n,b,a)`; one transaction each side — the minimum | per-block handshakes and barriers |
| **`blocks_rm_gather`** | **built** (Refinement 1): same host split machinery (units `(n, c, h)`), own reader/writer pair, no cross-core step | `layout == ROW_MAJOR ∧ dims == (0,1,4,2,5,3)` | `block_h` consecutive `h` of one `(n, c)`: `W·I·block_h` input pages of `J` values → `I·J·block_h` output pages of `W` values | x: one read per 4 B page into its own 64 B-aligned slot (DRAM read alignment), logical bytes once but 524288 transactions on p1; local `[W, I·J] → [I·J, W]` halfword gather; out once, 64 B pages | per-block handshakes/barriers; the gather per `(n, c, h)` is fixed |
| `blocks_rm_bank_coalesced` | deferred — alternative to `blocks_rm_gather` on p1/p2 (the PERF FOCUS case), settled by one experiment against it | as `blocks_rm_gather` | `k` adjacent 64 B slots of one DRAM bank (pages `p, p+NB, p+2·NB, …`) | x: 16× the logical bytes (whole 64 B slots) in `k·64 B` transactions instead of one transaction per 4 B page; out as `blocks_rm_gather` | fewer, larger reads |
| **`nhwc_rm_transpose`** | **built** (Refinement 4): `blocks_rm_gather`'s reader (rows of units with a short last unit), its own writer, the shared `transpose_pages` (`kernels/g6_permute_rm_transpose.hpp`); DRAM or L1 | `layout == ROW_MAJOR ∧ dims == (0,3,1,2)` | unit `(n, h, wb)`: `block_w` input pages of `C` values (the last unit of a row `tail_w = W − (⌈W/block_w⌉ − 1)·block_w`) → `C` partial output rows; a block is `block_units = ⌈BLOCK_PAGES / block_w⌉` units under the L1 and core caps. `block_w` is `W` (whole rows) or a multiple of 8 values, picked by `_choose_block_w` (p4: 64) | x once, one read per input page into its own slot (bank stride); local `[width, C] → [C, width]` word transpose; out once as `C` writes of `width·2` B at offset `wb·block_w·2` (16 B-aligned); the output's L1 pages count against the budget | handshakes per block; a wider unit means fewer, longer writes but coarser balance over the grid |
| **`nhwc_tile_bfp8_transpose`** | **built** (Refinement 2): own reader / compute / writer, DRAM or L1. Realization: transpose each input tile into DEST, `pack_untilize` it in bfloat16 as row `h` of a `[32, 32·32]` block (`cb_row_block`), then the tilize helper builds output tile `c` from column chunk `c` of every row and packs bfloat8_b — the row exchange is the tilize, no dataflow copies. The reader pushes each group in chunks of `READ_CHUNK_TILES` so the transpose overlaps the read | `layout == TILE ∧ dims == (0,3,1,2) ∧ dtype == bfloat8_b` | group `(n, ht, wt)`: 32 input tiles (`h` in the `ht` block) → 32 output tiles (`c`, `ht`, `wt`) | x once, out once, whole tiles in L1; in the core: unpack bfloat8_b → `transpose_wh` → pack bfloat16 (exact), then the `rows_tile_scatter` row exchange core-local, then unpack bfloat16 → pack bfloat8_b (the packer sets each group's shared exponent to its maximum = E) | one init set per group of 32 tiles |
| other TARGET combinations (e.g. (0,3,1,2) on bfloat16 TILE, (0,1,4,2,5,3) on TILE) | deferred — no production case; each is one of the index maps above on another layout | the remaining `(layout, dims, dtype)` cells | — | — | — |

Regime-pinned tests: the predicate is the cell, so each regime's tests are its cell's golden cases plus its acceptance shapes.

### Traffic ranking

Units: bytes and transactions per tier. Worked case 303 `[1, 6, 16, 64]` (12 input tiles, 32 output tiles, 12288 B logical).

| Rank | Split (candidate) | DRAM read | DRAM write | Cross-core | Core-local copies | Cores (303) | Status |
|------|-------------------|-----------|-----------|------------|-------------------|-------------|--------|
| 1 | **input tiles, rows scattered to the output** (`rows_tile_scatter`) | live prefix once: 12288 B, 12 reads | logical once: 12288 B, 384 × 32 B | 0 | 0 | 12 | **built** |
| 2 | closed groups, rows assembled in L1 (`rows_tile_group_assemble`) | 12288 B, 12 reads | whole tiles: 65536 B, 32 writes | 0 | 384 × 32 B | 2 | deferred |
| 3 | output tiles, rows gathered (`rows_tile_gather`) | 384 × 64 B bounce = 24576 B, 384 reads | 65536 B, 32 writes | 0 | 384 × 32 B | 32 | rejected |
| 4 | output tiles, input multicast (`rows_tile_mcast`) | 12288 B | 65536 B | 12 tiles × 16 receivers | 384 × 32 B | 32 | rejected |
| 5 | output tiles, input tiles re-read (`rows_tile_reread`) | 192 prefix reads = 196608 B | 65536 B | 0 | 384 × 32 B | 32 | rejected (dead end) |
| — | split a dependent axis + combine | not applicable: no result spans any axis (`math.md#axes-and-dependence`), so no axis is dependent and no combine exists | | | | | |

Split 1 moves the fewest bytes on every tier; it is the cheapest-traffic split and it is the one implemented. Splits 3–5 occupy more cores on the `A = 1` shapes (301, 305: 32 and 16 output tiles against 2 input tiles) at 2× the read bytes or more; occupancy does not override the traffic ranking, and two cores each moving one tile already sit at the launch floor (grid-synchronization lamp). General formulas for split 1: DRAM read `Σ_tiles prefix_bytes(rows_live(bt))`; DRAM write `tensor_n·tensor_a·tensor_b·tensor_w·2` B in `2·tensor_n·tensor_a·tensor_b·tensor_wt` transactions; cross-core 0.

**Operand-reuse check.** One input operand, `x`. Under split 1 it varies along the split axis (each core reads distinct tiles), so no core re-reads another's bytes and there is no broadcast row. Under the output-tile splits (3–5) `x` is reuse-shared by construction — the reason those splits re-read, bounce or multicast — and they are rejected.

**Stall-shadow check.** Waits: the writer on `cb_input_tiles` (the reader's block), the reader on its read barrier, the writer on `noc_async_writes_flushed` before popping, and the final `noc_async_write_barrier`. Work independent of the awaited data: the reader's next block (taken by `input_depth = 2` when a core owns ≥ 2 blocks); the writer's per-tile destination bases (cheap, computed before `cb_wait_front`). No algebraic reorder applies — the op has no arithmetic. Nothing else is independent of the waits.

### Block schedule

```cpp
for (uint32_t block_idx = 0; block_idx < num_blocks_this_core; ++block_idx) {
    load_block(block_idx);           // reader (NCRISC, NoC0)
    scatter_rows_block(block_idx);   // writer (BRISC, NoC1)
}
```

| Block operation | Block shape | Resident across it | Intended fixed-cost frequency |
|-----------------|-------------|--------------------|-------------------------------|
| `load_block` | `block_in_tiles` input tiles (`last_block_extent` on a ragged last block), each the live prefix `[0, prefix_bytes(rows_live(bt)))` | the block's tile slots in `cb_input_tiles`, until `scatter_rows_block` pops them | one `cb_reserve_back`; one read per tile; **one** `noc_async_read_barrier` per block; one `cb_push_back` of the **nominal** `block_in_tiles` pages, also on a ragged last block |
| `scatter_rows_block` | the same tiles × `rows_live(bt)` rows × 2 halves | the block's slots, read in place | one `cb_wait_front`; per tile one destination base `p_out(n, 32·bt, a div 32, wt)` and offsets `face_off(a mod 32, h)`; per row one destination page (base + `r · tensor_at · tensor_wt`), two 32 B writes from `face_off(r, h)`; rows walked from `r0 = a mod rows_live(bt)`, wrapping (stagger); **one** `noc_async_writes_flushed` per block, then one `cb_pop_front` of `block_in_tiles` pages; **one** `noc_async_write_barrier` per kernel, at the end (BB-03) |

No compute phase, so no inits and no format reconfigs exist. No multicast, no semaphores, no cross-core barrier.

### Perf lamps

| Lamp | Why the default may be wrong here | Nearby alternative to measure |
|------|-----------------------------------|-------------------------------|
| Overlap | `block_in_tiles` = a core's whole assignment, so on large shapes the whole block is read before the first scatter write | `block_in_tiles` 4–8 with `input_depth` 2 (catalog `double_buffer`) on the acceptance shape `[1, 64, 100, 256]` (≈ 16–19 tiles per core) |
| Grid synchronization | production runs 1 tile per core on 2–32 cores; per-core work is 1 read + ≤ 32 writes, so per-core launch cost may dominate | the same split over `ceil(tensor_in_tiles / 4)` cores; judge on host trace-replay time, not profiled kernel time (`lessons.md` RB-053) |
| Scatter issue rate | the writer issues 2 writes per live row (64 per full tile) while the reader issues 1 per tile: on dense shapes the writer RISC is issue-bound | split each core's tiles between both data-movement RISCs, each reading and scattering its share on its own NoC (catalog `split_reader`); `noc_async_write_one_packet_set_state` once per row + `noc_async_write_one_packet_with_state` per half (`dataflow_api.h:982`, `:1008`) |
| Row-walk stagger | default starts each tile's walk at `r0 = a mod rows_live(bt)` so cores with neighbouring `a` hit different output pages / banks at once (catalog `bank_stagger`); with one tile per core the gain may be nil | `r0 = 0` (CT flag `stagger_rows = 0`) |
| Read barrier granularity | one drain-all barrier per block idles the read stream between blocks on cores with many blocks | transaction-id barriers (catalog `trid_double_issue`) |
| Core placement | `row_wise=True` spreads the active cores along the DRAM-facing axis (catalog `noc_placement`) | `row_wise=False` |
| Prefix vs exact-row reads | for `rows_live < 16` the prefix carries `(16 − rows_live) · 32` B of padding rows (B = 1: 480 of 544 B) in its one transaction | two reads per tile (face 0 and face 1 live rows only) |

### Catalog entries that set the knobs

| Entry | Knob it set |
|-------|-------------|
| `double_buffer` | `load_block` issues every read of the block before one barrier; `input_depth = 2`; the overlap lamp |
| `tile_reorder` | reads are whole-prefix transfers (one per tile); the whole-tile-write alternative for DRAM-bound dense shapes is the deferred `rows_tile_group_assemble` |
| `width_split` | `wt` is folded into the work unit, so short wide shapes (`A = 1`) still spread over `tensor_wt` cores |
| `noc_placement` | `split_work_to_cores(..., row_wise=True)`; reader on NoC0, writer on NoC1 |
| `bank_stagger` | row-walk stagger default and its lamp |
| `split_reader`, `trid_double_issue` | lamps only — they pay only once a RISC is issue-bound or blocks are many |

## Dataflow Strategy

| Stage | Format | Mechanism | Notes |
|-------|--------|-----------|-------|
| x in DRAM | bfloat16 tiles, 4 faces of 16×16 | interleaved pages `p_in` | input padding rows may hold garbage |
| DRAM → `cb_input_tiles` | live prefix of each tile | reader (NCRISC, NoC0): `noc_async_read(x_accessor.get_noc_addr(p_in), slot, prefix_bytes(rows_live(bt)))` | one read per tile, one barrier per block |
| `cb_input_tiles` → DRAM out | 32 B half-rows | writer (BRISC, NoC1): `noc_async_write(slot + face_off(r, h), out_accessor.get_noc_addr(p_out, face_off(a mod 32, h)), 32)` | non-posted (posted writes have no completion to wait on before the next program, BB-03); flushed per block, barrier at kernel end |
| out in DRAM | bfloat16 tiles | — | logical rows written; padding rows untouched |

No value passes through an unpacker, DEST or packer, so every bfloat16 bit pattern (NaN payloads, −0, denormals, ±inf) arrives unchanged (risk R3, spec rule 1).

No regime in the table needs Tensix-to-Tensix communication: no axis is dependent and no operand is reuse-shared under the chosen splits.

**TARGET placement axis (`buffer`).** TARGET has `"dram"` and `"l1"`, both interleaved; no `*_SHARDED` value exists (contract r3: no tensor of this op is sharded). For `rows_tile_scatter`, `"l1"` is a **knob-turn**: the accessors address L1 banks; `block_cap` subtracts `out_l1_bytes_per_core = ceil(tensor_out_tiles / num_l1_banks) · out_page_bytes`; the L1 read alignment (16) only relaxes the prefix constraint. `tag_buffer` gives a sharded input its own label so `validate()` refuses it instead of mis-addressing it.

## Work Distribution

| Field | Value |
|-------|-------|
| Work unit | one input tile on the `in_tile` axis; a block is `block_in_tiles` consecutive input pages |
| Grid | `core_grid` = `device.compute_with_storage_grid_size()` — a host parameter, never inlined; active cores = `min(tensor_in_tiles, grid cores)` |
| Split | `split_work_to_cores(core_grid, tensor_in_tiles, row_wise=True)` → `(num_cores, all_cores, core_group_1, core_group_2, tiles_g1, tiles_g2)`; cores walked in `all_cores` row-major order; core `k` starts at page `Σ` of earlier cores' shares |
| Per-core work | `core_in_tiles ∈ {tiles_g1, tiles_g2}`; `max_core_in_tiles = tiles_g1`; `block_in_tiles = min(max_core_in_tiles, block_cap)`; `num_blocks_this_core = ceil(core_in_tiles / block_in_tiles)`; `last_block_extent = core_in_tiles − (num_blocks_this_core − 1) · block_in_tiles` |
| Remainder | `tiles_g1 = tiles_g2 + 1` where the split is ragged; a ragged last block keeps the nominal push / wait / pop counts and narrows only its reads and writes |
| Start coordinates | the host decomposes each core's start page into `(n0, a0, bt0, wt0)` and passes them as RT args; kernels advance the four counters (`wt` innermost) without a division per tile |
| Geometry | alignment-aware per image: `tensor_bt = ceil(tensor_b / 32)`, `tensor_at = ceil(tensor_a / 32)`, `tensor_wt = ceil(tensor_w / 32)` — never `floor(N·H/32)` |
| Production | 2, 32, 12, 32, 2, 16, 12, 16 active cores for cases 301–308, one tile each; `core_group_2` empty; `all_cores` is 1–3 rectangles (BB-17) |
| Regime selection | `select_regime(axes)` by `(layout, dims, dtype)`; one built regime in Phase 0; grid-independent |

## Circular Buffers

| Semantic Name | Index | Page Size | Num Pages | Sizing rationale | Format | Producer | Consumer | Lifetime |
|---------------|-------|-----------|-----------|------------------|--------|----------|----------|-----------|
| `cb_input_tiles` | 0 | `in_page_bytes = input_tensor.buffer_page_size()` (2048) | `min(input_depth, max_blocks_per_core) · block_in_tiles` (1 in production) | spans `in_tile` (`block_in_tiles`) and the whole tile slot (`r`, `h`); streams over nothing else | input dtype (Float16_b) | reader | writer | whole program |

The op has one buffer. The output in DRAM needs none: rows go from the input slot straight to their destination (pack-into-destination, `l1-footprint-discipline.md` Rule 3.1). Ledger: `l1_ledger.md`.

## Block Operation Realization

| # | Block operation | Block shape | Helper? | Input CB (semantic name, pages, state) | Output CB (semantic name, pages) | CB state after |
|---|-----------------|-------------|---------|----------------------------------------|----------------------------------|----------------|
| 1 | `load_block` | `block_in_tiles` tiles, prefix each | no — raw dataflow (no kernel_lib helper reads into a CB from a TensorAccessor) | — (x in DRAM) | `cb_input_tiles`, `block_in_tiles` pages pushed | block resident until op 2 pops it |
| 2 | `scatter_rows_block` | same tiles × `rows_live(bt)` × 2 | no — raw dataflow (gap: no row-scatter block helper) | `cb_input_tiles`, `block_in_tiles` pages, waited | — (out in DRAM) | popped after `noc_async_writes_flushed` |

## API Mapping

| Block operation | Type | Function | File:Line | Template Params / Args | Input CB | Output CB | Which params are block knobs |
|-----------------|------|----------|-----------|------------------------|----------|-----------|------------------------------|
| `load_block` | raw_api | `cb_reserve_back` / `cb_push_back` | `tt_metal/hw/inc/api/dataflow/dataflow_api.h:406` / `:208` | `(cb_input_tiles, block_in_tiles)` | — | `cb_input_tiles` | `block_in_tiles` |
| `load_block` | raw_api | `get_write_ptr` | `dataflow_api.h:325` | slot `k` = write ptr + `k · in_page_bytes` | — | `cb_input_tiles` | — |
| `load_block` | raw_api | `TensorAccessor::get_noc_addr(page_id, offset)` | `tt_metal/hw/inc/api/tensor/tensor_accessor.h:165` | `p_in`; host `ttnn.TensorAccessorArgs(input_tensor)` | — | — | — |
| `load_block` | raw_api | `noc_async_read` | `dataflow_api.h:547` | `(src, slot, prefix_bytes(rows_live(bt)))` | — | `cb_input_tiles` | — |
| `load_block` | raw_api | `noc_async_read_barrier` | `dataflow_api.h:1921` | once per block | — | — | `block_in_tiles` sets its frequency |
| `scatter_rows_block` | raw_api | `cb_wait_front` / `cb_pop_front` | `dataflow_api.h:479` / `:260` | `(cb_input_tiles, block_in_tiles)` | `cb_input_tiles` | — | `block_in_tiles` |
| `scatter_rows_block` | raw_api | `get_read_ptr` | `dataflow_api.h:346` | slot `k` = read ptr + `k · in_page_bytes` | `cb_input_tiles` | — | — |
| `scatter_rows_block` | raw_api | `TensorAccessor::get_noc_addr(page_id, offset)` | `tensor_accessor.h:165` | `(p_out, face_off(a mod 32, h))`; host `ttnn.TensorAccessorArgs(output_tensor)` | — | — | — |
| `scatter_rows_block` | raw_api | `noc_async_write` (or `noc_async_write_one_packet_set_state` + `_with_state`) | `dataflow_api.h:876` (`:982`, `:1008`) | `(slot + face_off(r, h), dst, FACE_ROW_BYTES)`, non-posted | `cb_input_tiles` | — | — |
| `scatter_rows_block` | raw_api | `noc_async_writes_flushed` | `dataflow_api.h:1980` | once per block, before `cb_pop_front` (source slots must not be overwritten while writes read them) | — | — | `block_in_tiles` sets its frequency |
| `scatter_rows_block` | raw_api | `noc_async_write_barrier` | `dataflow_api.h:1951` | once, at kernel end (BB-03) | — | — | — |

**Helpers considered and rejected** (apply to every row above):

| Candidate | File:Line | Why it cannot express this op |
|-----------|-----------|-------------------------------|
| compute helpers (`tilize_helpers.hpp`, `untilize_helpers.hpp`, `eltwise_convenience.hpp` copy, `transpose_wh`) | `ttnn/cpp/ttnn/kernel_lib/` | there is no compute phase; routing bfloat16 through unpack → DEST → pack can change bits (risk R3, spec rule 1 forbids it) |
| `reshuffle_rows_tile` | `tt_metal/hw/inc/api/compute/reshuffle.h:31` | moves rows inside one tile in DEST (SFPU); the Phase 0 moves cross tiles, and it is a compute path |
| `set_read_state` / `read_with_state` | `ttnn/cpp/ttnn/kernel_lib/local_copy_helpers_dataflow.hpp:75` | core-local L1→L1 copies; this regime copies nothing locally — it writes from L1 straight to DRAM. Reserved for `rows_tile_group_assemble` |
| `tile_column_dataflow.hpp` | `ttnn/cpp/ttnn/kernel_lib/tile_column_dataflow.hpp:1-30` | builds a tile column by RISC-V stores in local L1; this op moves rows to remote pages |
| mcast (`McastArgs`) | `ttnn/cpp/ttnn/kernel_lib/mcast/kernel/mcast_args.hpp:18` | no reuse-shared operand under the chosen split |
| `reduce_helpers_dataflow.hpp` | `ttnn/cpp/ttnn/kernel_lib/reduce_helpers_dataflow.hpp` | no reduction |

Gap recorded: kernel_lib has no block helper that scatters tile rows to remote pages. `scatter_rows_block` is the raw dataflow calls above; the implementer may wrap them as one block function. Closing the gap: a dataflow helper `scatter_tile_rows(slot, rows, dst_base_page, page_stride, dst_offsets)`.

## Key Risks and Gotchas

| Risk | Why it bites here | Mitigation in this design |
|------|-------------------|---------------------------|
| Input padding reaches output (R4) | every production input tile is 1–16 of 32 rows live; the model's producer leaves garbage in the rest | the scatter walks exactly `rows_live(bt)` rows; acceptance tests poison the padding (inf, 1000) |
| A mid-tile DRAM read | a "read only the live rows" optimization that starts a read at an odd face row breaks the mod-64 rule silently | reads are prefixes from the tile base only (mechanism cap 1) |
| `rows_live` taken from the padded shape | `input_tensor.padded_shape[-2]` is 32, the logical B is not | geometry from `input_tensor.shape` (logical); padded shape only for page counts |
| `tensor_a > 32` or `tensor_b > 32` | production never reaches them, so a map fitted to A, B ≤ 16 passes the golden suite and fails the next caller (spec rule 9) | `at = a div 32`, `ro = a mod 32`, `bt` loop with per-`bt` `rows_live`; acceptance shapes `[1, 40, 6, 64]`, `[1, 6, 40, 64]`, `[2, 34, 47, 96]` |
| Source slot overwritten in flight | popping `cb_input_tiles` before the half-row writes left L1 lets the reader overwrite their source | `noc_async_writes_flushed` before `cb_pop_front` |
| Writes in flight at kernel exit (BB-03) | the next program starts right after; a write still in flight corrupts or hangs it | non-posted writes; `noc_async_write_barrier` at the end of the writer |
| Per-row bank decomposition cost | `get_noc_addr(p_out)` resolves bank and offset per row; with `2·rows` writes per tile this is the writer's hot path | output page per row = base + `r · tensor_at · tensor_wt`; the implementer may advance bank/offset incrementally instead of calling the accessor per row |
| Cache key incomplete (R10, BB-07) | shapes with other dims or another dtype would replay the wrong index map | the key holds every input of `select_regime` and of the CT args (see *Host contract*) |
| Kernel text per role (BB-04, BB-18; R8) | later regimes add index maps, layouts, dtypes | one reader/writer pair **per regime**, selected on host; no role carries another regime's code |

## Host Contract

| Item | Decision |
|------|----------|
| Entry flow | `validate(input_tensor, dims)` first → canonical dims → output shape → allocate output (`ttnn.allocate_tensor_on_device`, input's dtype, layout, memory config) → cached descriptor for the key, patched with the two buffer addresses → `ttnn.generic_op([input_tensor, output_tensor], descriptor)` — one dispatch |
| Taggers (request's names) | `tag_dtype`, `tag_layout`, `tag_buffer` (`"dram"` / `"l1"` for interleaved; a distinct label for sharded so it is refused), `tag_rank`, `tag_alignment` (W non-aligned wins over H), `tag_dims` (wraps negatives; raises `ValueError` for wrong length, out-of-range entry, non-permutation — before any support check) |
| validate order | malformed dims → `ValueError` (from `tag_dims`); non-32×32 tile → `ValueError`; SUPPORTED per axis → `UnsupportedAxisValue`; EXCLUSIONS (empty) → `ExcludedCell`. `dims` is canonicalized to non-negative entries before the support check |
| Program cache (BB-07, BB-16) | descriptor cached per key `(input logical shape, input padded shape, canonical dims, dtype, layout, buffer type, memory layout, tile shape, core_grid)`; buffer addresses are runtime args (common RT args recommended: two values) rewritten in the cached descriptor on every call; the program hash covers runtime args by count only (`ttnn/cpp/ttnn/operations/generic/device/generic_op_device_operation.cpp:78-100`), so a repeat call hits the compiled program with the new addresses; CT args hold geometry, `block_in_tiles`, `input_depth`, `stagger_rows`, face sizes |
| Plan report (BB-08) | a module function returns, for `(input_tensor, dims)`, the regime name, active core count, `block_in_tiles` and `num_blocks` per core group, without dispatching |
| Zones (BB-21) | `load_block`, `scatter_rows_block`, and each wait (`cb_wait_front`, read barrier, writes flushed, final write barrier) inside named zones, compiled in only in the zone build; the plan report doubles as the role dump: every active core runs reader + writer, other cores are idle |
| Core ranges (BB-17) | `all_cores` from `split_work_to_cores(row_wise=True)`: 1–3 rectangles for every production shape |

### Structural impossibilities

Cells pairing a dims value with the other rank — `{rank: 4, dims: (0,1,4,2,5,3)}`, `{rank: 6, dims: (0,2,1,3)}`, `{rank: 6, dims: (0,3,1,2)}` — cannot exist (dims length equals rank). `feature_spec.py` needs no INVALID entry for them because every axis is projected from a scenario; if a free axis is ever added, they belong in INVALID.

## Acceptance Test

`tests/ttnn/unit_tests/operations/g6_permute/test_g6_permute.py` (immutable for the implementer). Bit-exact gate (`helpers.TOLERANCES[bfloat16] = ("exact", None)`), compared as int16 bit patterns. Cases (BB-22: production cases and cases the rules require):

| Group | Cases | Rule it covers |
|-------|-------|----------------|
| production | the eight Qwen shapes | Phase 0 cell |
| other shapes of the cell | `[1,1,5,32]` (one input tile), `[2,6,16,64]` (N > 1), `[1,40,6,64]` (A > 32), `[1,6,40,64]` (B > 32, rows past face row 16), `[2,34,47,96]` (every axis past one tile, `tensor_wt = 3`), `[1,64,100,256]` (many tiles per core) | spec rule 9 |
| special values | inf, −inf, NaN (two payloads), −0, denormals at `[1,6,16,64]` | spec rule 1 |
| poisoned padding | inf at `[1,6,16,64]`, 1000 at `[1,16,1,64]`, NaN at `[1,6,40,64]` | spec rule 2 |
| negative dims | `(0, -2, 1, -1)` equals `(0, 2, 1, 3)` bit for bit | Validation |
| malformed dims | wrong length, repeated entry, out-of-range entry → `ValueError` / `RuntimeError` | Validation |
| repeat call | same shape twice with new data: no new program cache entry, fresh values (not a stale replay), identical bits on identical input | BB-06, BB-07 |
| placement | every case checks output shape, dtype, layout, DRAM, interleaved | Output |
