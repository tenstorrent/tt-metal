# Operation Design: g6_slice

## Overview

| Field | Value |
|-------|-------|
| Classification | data_movement — a pure relocation of bytes. No compute kernel, no unpack/DEST/pack round trip, no arithmetic. |
| Goal | Return `x[starts:ends:steps]` (PyTorch basic indexing) bit for bit, in x's dtype, layout and memory placement, as **one** `ttnn.generic_op` dispatch whose reader touches only the parts of x that hold selected elements and whose writer touches only out. |
| Math | `out[i_0..i_{r-1}] = x[b_0 + s_0·i_0, …, b_{r-1} + s_{r-1}·i_{r-1}]`, `M_d = ceil((e_d − b_d)/s_d)`, `b_d = clamp(starts[d] (+N_d if < 0), 0, N_d)`, `e_d = clamp(ends[d] (+N_d if < 0), b_d, N_d)`. Device-tensor form (refinement): `out = x.narrow(sd, r0, L)`, `r0 = floor(starts[sd] / 32)·32`, `L = N_sd / num_devices`. |
| References | `eval/golden_tests/g6_slice/feature_spec.py` (TARGET / INPUTS / INVALID / LOOSE_CASES), `eval/golden_tests/g6_slice/helpers.py` (oracle, `TOLERANCES` = exact for bfloat16), `eval/op_template.py` (registry model), `.claude/references/blocking-model.md`, `.claude/references/l1-footprint-discipline.md`, `.claude/references/noc-alignment-in-tt-metal.md`, `.claude/references/generic_op_template/`, `tech_reports/tensor_accessor/tensor_accessor.md`, `tech_reports/tensor_layouts/tensor_layouts.md`, catalog `ttnn/ttnn/operations/examples/master.md` (entries listed under *Catalog evidence used*), `l1_ledger.md` (this directory) |

## Parameters

| Name | Type | Required | Valid Range | Default | CT/RT |
|------|------|----------|-------------|---------|-------|
| `input_tensor` | `ttnn.Tensor` | yes | rank 2–4. Phase 0: bfloat16, TILE (32×32 tiles, 16×16 faces), DRAM interleaved | — | TensorAccessor args CT; buffer address common RT |
| `starts` | `list[int]` of length r, or `ttnn.Tensor` uint32 ROW_MAJOR DRAM shape (r,) | yes | any int; normalized and clamped per *Math* | — | host list: host-normalized; enters the kernels as the regime define and `src_slots` (CT, from `row_shift`/`col_shift`) and as RT values (per-core first source page, `row_shift`, `col_shift`). Tensor form: read on device by the reader (refinement), never on the host, never CT |
| `ends` | same form as `starts` | yes | any int; clamped to `[b_d, N_d]`. A tensor is never read | — | host list: through `M` (RT). Tensor: unread |
| `steps` | `list[int]` of length r, or `None` | no | ≥ 1. SUPPORTED: all 1 (`step_mode` `"strided"` is outside TARGET) | `None` ≡ all 1 | host only (selects `step_mode`) |
| `slice_dim` | `int` | only with `num_devices`, only with tensor begins | `[−r, r−1]`, canonicalized to `slice_dim + r` before any check | `None` | refinement: CT (which stride the device begin scales) |
| `num_devices` | `int` | only with `slice_dim` | ≥ 1 and divides `N_{slice_dim}` | `None` | refinement: host (gives `L`, part of the output shape) |

There is no `memory_config`, `dtype` or `compute_kernel_config` parameter.

## Tensors

### Input

| Property | Requirement |
|----------|-------------|
| Shape | `(N_0, …, N_{r-1})`, r ∈ {2, 3, 4}; any dim may be non-tile-aligned |
| Dtype | Phase 0: bfloat16. TARGET: float32, int32, uint32, bfloat8_b, bfloat4_b |
| Layout | Phase 0: TILE. TARGET: ROW_MAJOR |
| Memory | Phase 0: DRAM interleaved. TARGET: L1 interleaved, L1 width-sharded |

### Output

| Property | Value |
|----------|-------|
| Shape | `(M_0, …, M_{r-1})` (host-list form); `N` with `N_sd` replaced by `L` (tensor form) |
| Dtype | `input_tensor.dtype` |
| Layout | `input_tensor.layout` |
| Memory | `input_tensor.memory_config()` — same buffer type, interleaved; width-sharded keeps x's shard spec unchanged (refinement) |
| Allocation | `ttnn.allocate_tensor_on_device(ttnn.Shape(M), x.dtype, x.layout, device, x.memory_config())` — positional args (host steps: *Host Entry Contract*) |
| TILE padding | positions of out beyond its logical shape are not part of the result and are **not guaranteed to be written** (R2 writes only the selected rows of a tile). No input padding ever lands on a logical output position |

## Blocking Model

Semantics: `.claude/references/blocking-model.md`. Everything below this section realizes these decisions.

**Index space.** The op's tile index space is the **output's**: canonical 4D `out_shape = (M0, M1, M2, M3)`; `tensor_out_row_tiles = ceil(M2/32)`, `tensor_out_col_tiles = ceil(M3/32)` per image; one output page per `(i0, i1, tr, tc)`. Input geometry: `tensor_in_row_tiles = ceil(N2/32)`, `tensor_in_col_tiles = ceil(N3/32)`; page strides `in_stride_col = 1`, `in_stride_row = tensor_in_col_tiles`, `in_stride_lead1 = tensor_in_row_tiles·tensor_in_col_tiles`, `in_stride_lead0 = N1·in_stride_lead1`. Shifts: `row_shift = b2 mod 32`, `col_shift = b3 mod 32`, `row_tile_begin = b2 div 32`, `col_tile_begin = b3 div 32`.

**Source map.** Output page `(i0, i1, tr, tc)` draws from top-left source page
`q = (b0+i0)·in_stride_lead0 + (b1+i1)·in_stride_lead1 + (row_tile_begin+tr)·in_stride_row + (col_tile_begin+tc)`,
plus `q + in_stride_row` iff `row_shift + valid_rows(tr) > 32`, plus `q + 1` iff `col_shift + valid_cols(tc) > 32` (and `q + in_stride_row + 1` iff both), where `valid_rows(tr) = min(32, M2 − 32·tr)`, `valid_cols(tc) = min(32, M3 − 32·tc)`. A page is read **only** when that condition holds — reading it unconditionally reads past the end of x (row 4095 of 4096).

**Fold.** All four axes are independent and the output pages are contiguous in row-major `(i0, i1, tr, tc)` order, so core assignment and blocks act on the folded index `out_page = ((i0·M1 + i1)·tensor_out_row_tiles + tr)·tensor_out_col_tiles + tc`. The fold is honest for this op: the page is the transfer unit (interleaved pages are never coalesced across pages), no helper consumes a block shape, and the per-block fixed cost (one read barrier, one CB handshake, one write flush) is the same wherever in the tile grid the block lies. The source side follows by an incremental odometer — no div/mod per page.

### Axes

| Axis | Character (+ one-clause reason) | Extent knob | Phase 0 value | Knob source | Core-assignment | Later unlock |
|------|--------------------------------|-------------|---------------|-------------|-----------------|--------------|
| `lead0` (dim r−4; extent 1 for r < 4) | independent — each output element copies one input element, nothing spans the axis | `block_tiles` via the fold (no separate extent: any `block_tiles` consecutive pages are one block) | covered by the fold | host constant `BLOCK_BYTES_CAP` (see `out_page`) | contiguous `out_page` ranges | knob-turn |
| `lead1` (dim r−3; extent 1 for r < 3) | independent — same | `block_tiles` via the fold | covered by the fold | same | same | knob-turn |
| `tile_row` (dim r−2, `tensor_out_row_tiles`) | independent — R2's two source tile rows per output tile row are a footprint, not a combine; the bands are disjoint, so nothing is shared between neighbours | `block_tiles` via the fold | covered by the fold | same | same | knob-turn |
| `tile_col` (dim r−1, `tensor_out_col_tiles`) | independent — R3's neighbouring output tiles share a source tile column (overlapping footprint, read twice in Phase 0) | `block_tiles` via the fold | covered by the fold | same | same | knob-turn |
| `out_page` (row-major fold of the four) | independent | `block_tiles` = output pages per block | `max(1, min(core_pages_max, BLOCK_BYTES_CAP // (page_bytes · src_slots)))` → bfloat16: R1 32, R2 single-slot 32, R2 two-slot 16, R3 8–16 | host constant `BLOCK_BYTES_CAP = 64 KiB` in the program-descriptor module; `block_tiles` derived once and passed as a CT arg; CB pages, push/pop quanta, `num_blocks_this_core` all derive from it | `ttnn.split_work_to_cores(grid, tensor_out_pages, row_wise=True)` over the **full** `compute_with_storage_grid_size()` grid; cores = `min(tensor_out_pages, grid cores)` | knob-turn |
| `src_slot` (staged source pages per output page) | not a data axis — the regime's footprint per output page | `src_slots` | R1: 1; R2: 2 iff `row_shift + min(M2, 32) > 32` else 1; R3: `(col_shift + min(M3, 32) > 32 ? 2 : 1) · (row_shift + min(M2, 32) > 32 ? 2 : 1)` | host, from the regime predicate; CT arg | — | — |

**Why `block_tiles` is below the L1 bound.** The coarsest block that fits would be ≈ 329 bfloat16 pages (1,347,776 B / (2 · 2048 B)). It is capped at 64 KiB because this is a pure copy whose per-block fixed cost is one barrier + one CB handshake + one flush (sub-µs), while pipeline fill (the first block is read before any write issues) and drain (the last block is written after the last read) grow with the block (overlap lamp, §4). Catalog `double_buffer` measured the gain of batching reads saturating at 4–8 pages per barrier; `trid_double_issue` measured ≈ 77–98 GB/s on **one** core at block 16 — far above the ≈ 3.5 GB/s per-core share of DRAM bandwidth when the full grid copies (≈ 460 GB/s, the roofline the lead quotes for p1). 32 pages is 2× above the largest block those entries saw help; the lamp names the sweep.

**Interleaved ordering.** First spread `tensor_out_pages` over the full grid (occupancy); then take the block above within each core's range. The block never exceeds the core's assignment (`core_pages_max`).

### Operand-reuse check

| Operand | Varies along the split (`out_page`)? | Reuse-shared by construction? | Consequence |
|---------|--------------------------------------|-------------------------------|-------------|
| `x` | yes — every selected input element feeds exactly one output element | no | no broadcast; each core reads disjoint source bytes (R3 excepted: a shared source column between neighbouring output tiles, see R3 row) |
| `starts` tensor (refinement) | no — every core needs the same `starts[sd]` word | **yes** | broadcast regime row `R1-dev-bcast` (deferred, below) |
| `ends` tensor | never read | — | — |

### Buffer-depth knobs

| CB | Depth knob | Phase 0 value | What the depth buys |
|----|------------|---------------|---------------------|
| `cb_src_stage` | `stage_depth_blocks` (host constant, single source; CB pages = `stage_depth_blocks · block_tiles · src_slots`) | 2 | the reader fills block k+1 while the writer drains block k (read/write overlap on NoC0/NoC1) |

### Mechanism caps

| Mechanism | Cap on which extent | Clamp | What happens unclamped |
|-----------|--------------------|-------|------------------------|
| CB ring wrap (`ttnn-cb-memory-fundamentals.md`, *CB Ring-Wrap Invariant*) | push/pop quantum | every push and pop uses the **nominal** `block_tiles · src_slots`; a ragged last block moves only `last_block_extent` pages but still pushes/pops nominal; capacity = `stage_depth_blocks ·` nominal | the pointer steps past `fifo_limit` and the kernels read/write neighbouring L1 — silent corruption in release builds |
| Core assignment | `block_tiles` | `block_tiles ≤ core_pages_max` (the largest per-core count of the split) | blocks larger than the work; ring quanta that never close |
| DRAM read alignment (Blackhole: source and L1 destination equal mod 64; `NOC_DRAM_READ_ALIGNMENT_BYTES`, `tt_metal/hw/inc/internal/tt-1xx/blackhole/noc/noc_parameters.h:379`) | R2 sub-page reads | each band read lands in its stage slot at the **same intra-page offset** it has in DRAM; envelopes are `align_down/align_up` to `NOC_DRAM_READ_ALIGNMENT_BYTES` (kernel-side constant — never a host literal); stage slots are whole pages and CB bases are aligned to the DRAM alignment (`tt_metal/impl/program/program.cpp:2505-2509`); `face_bytes % NOC_DRAM_READ_ALIGNMENT_BYTES == 0` keeps every envelope inside its face | data lands at the wrong offset — silently wrong rows (the watcher only flags it under `--dev`) |
| DRAM write alignment (Blackhole: equal mod 16) | R2 row-run writes; R3 | runs start at multiples of `face_row_bytes` (32 B bfloat16) on both sides; element-granular column shifts (R3) are assembled in L1 first and written as whole pages | misaligned scatter writes corrupt out |
| Face geometry | R2/R3 segment math | tiles must be 32×32 with 16×16 faces (`x.tile.tile_shape == (32, 32)`; host `RuntimeError` otherwise); block formats (exponent section) are outside the R2/R3 math (row `R2/R3-bfp`) | rows/elements taken from the wrong face |
| Source bounds | which source pages are read | bottom page only iff `row_shift + valid_rows(tr) > 32`; right page only iff `col_shift + valid_cols(tc) > 32` | reads past the last page of x |
| Device begin (refinement) | source page offset | `r0 = min(floor(idx/32)·32, r0_max)`, `r0_max = floor((N_sd − L)/32)·32` for a tiled dim, `N_sd − L` for a leading dim | an arbitrary `starts` value reads outside x |
| L1 budget (BB-05) | total CB bytes | `stage_depth_blocks · block_tiles · src_slots · page_bytes ≤ 2 · max(BLOCK_BYTES_CAP, src_slots · page_bytes)` = 128 KiB ≪ 1,347,776 B | launch refused in the model |

### Regimes

Selection (host, exact; depends only on normalized begins and `M`, never on the grid):

```
def select_regime(b, M):              # canonical 4D, host-list form
    row_shift, col_shift = b[2] % 32, b[3] % 32
    if col_shift != 0:                       return "R3_col_shift"
    if row_shift != 0 or M[2] < 32:          return "R2_row_band"
    return "R1_page_copy"
```

Minimum (named boundary): every byte of x that holds a selected element crosses DRAM once (at DRAM read granularity), every byte of out that holds a selected element crosses DRAM once.

| Regime | Status | Predicate | Block | Data movement vs. minimum | What a bigger block buys |
|--------|--------|-----------|-------|---------------------------|--------------------------|
| `R1_page_copy` | **built** — 748 of 868 Phase 0 cells incl. p1–p4 (≤ 278528 pages) | `col_shift == 0 and row_shift == 0 and M2 ≥ 32` | `block_tiles` output pages, `src_slots = 1`; the staged source page **is** the output page | minimum at page granularity: each selected input page read once, each output page written once. Last tile row/col of out carry whole tiles (≤ 31 extra rows/cols of x, input data or input padding landing only on output padding positions) | one `cb_reserve_back`/`cb_push_back` + one `noc_async_read_barrier` (reader) and one `cb_wait_front`/`cb_pop_front` + one `noc_async_writes_flushed` (writer) per block instead of per page; pipeline fill/drain grow with it (lamp L1) |
| `R2_row_band` | **built** — 120 Phase 0 cells: every in-tile dim r−2 begin (all with `M2 = 1`) and every `M2 < 32` selection; ≤ 2080 pages | `col_shift == 0 and (row_shift != 0 or M2 < 32)` | `block_tiles` output pages × `src_slots ∈ {1, 2}` staged source pages each | minimum at DRAM read granularity: reads only the face-row bands of the selected rows (64 B envelopes on Blackhole), writes only the selected rows (32 B runs per face half). Case 587: 640 pages × (2 × 64 B read + 2 × 32 B write) = 81,920 B read / 40,960 B written vs. 1,310,720 B / 1,310,720 B for whole pages. The bands of neighbouring output tile rows are disjoint, so no source byte is read twice | same per-block handshake/barrier/flush; the band/run segment tables are computed once per kernel launch for the ≤ 2 valid-row classes, not per block |
| `R3_col_shift` | **built** — contract correctness only: SUPPORTED `w_non_aligned` admits `b[-1] % 32 != 0`; no production case has one (every TILE begin on dim r−1 is a multiple of 32) | `col_shift != 0` | `block_tiles` output pages × `src_slots ∈ {2, 4}` whole source pages | above minimum: each source page read up to 2× (4× with `row_shift ≠ 0`) because neighbouring output tiles share a source column; out written as whole pages. Accepted: no production cell | same handshake amortization; the element-granular assembly is per page and does not amortize |
| `R3_reuse` (keep the shared source column resident across the block) | **deferred** — no production shape has a dim r−1 begin inside a tile, so it would serve no golden cell | `col_shift != 0` | as R3 with a sliding slot window along `tile_col` | removes R3's 2× re-read → `(block_tiles+1)/block_tiles` | reachable: blocks walk consecutive `tc`, so the right slot of page k is the left slot of page k+1; only the slot indexing changes |
| `R2/R3-bfp` (row/col shift for bfloat8_b/bfloat4_b) | **deferred** — dtype axis values outside Phase 0 SUPPORTED; every production block-format case is `tile_aligned` (R1). Must ship in the same refinement that admits a block format, because `h_non_aligned` cannot be excluded | `dtype ∈ {bfloat8_b, bfloat4_b}` and R2/R3 predicate | whole source pages; in-place L1 assembly (the R3 assembler) extended to move the per-face-row exponent bytes with their mantissa rows | as R3 | reachable: R3's writer-side in-place assembler is the mechanism; only the byte map of a tile changes |
| `RM_stick_gather` | **deferred** — layout ROW_MAJOR outside Phase 0 SUPPORTED (312 production cases) | `layout == ROW_MAJOR` | `block_tiles` output sticks | minimum: each selected stick's byte range `[b3·es, (b3+M3)·es)` read once (64 B envelope; a realigning L1→L1 copy when `(b3·es) % 16 != 0`), each output stick written once | same handshake amortization | reachable: same `out_page` fold and odometer with the stick as the page and `tensor_*_row_tiles` replaced by row counts |
| `R1-dev` (device-tensor begin) | **deferred** — `begins_source` `"device_tensor"` outside Phase 0 SUPPORTED (2 production cases + loose cases) | tensor begins | as R1 (or R2 with `row_shift = 0` when `L < 32` on dim r−2) | as R1/R2; plus one read of the starts page per core | as R1 | reachable: the begin enters only as a constant source-page offset `Δq = (r0 in units of sd) · in_stride_sd` added to every source page id; Phase 0 folds the host begin into the per-core first source page instead. The writer never depends on the begin |
| `R1-dev-bcast` (one reader fetches `starts[sd]` and multicasts `Δq`) | **deferred** — the shared operand is one 4-byte word; the multicast replaces one 64 B DRAM read per core with a semaphore rendezvous of the same latency on every core's critical path and adds a sender role. `R1-dev` covers every shape it would serve | tensor begins | as R1-dev | removes `num_cores − 1` reads of one DRAM word; adds one multicast + flag semaphore | — | reachable: `McastArgs` sender/receiver (`ttnn/cpp/ttnn/kernel_lib/mcast/kernel/mcast_args.hpp`) on the reader, payload = `Δq` |
| `R1-shard` (width-sharded L1) | **deferred** — `memory_config` `"l1_width_sharded"` outside Phase 0 SUPPORTED (2 production cases) | x and out width-sharded, same shard spec | one block = the whole output shard of the core (8 pages in production) | core-local when `b3` is shard-aligned (production `b3 = 0`: output shard k = input shard k); otherwise pulls the neighbouring shard's pages over the NoC; out written in place (zero copy) | one block per core, no per-block handshake | reachable: same `out_page` → source map; core assignment pinned by the output shard spec; out consumed through a CB backed on the output shard (`ttnn.cb_descriptor_from_sharded_tensor`), so the writer's DRAM path disappears |
| `in_page_owner` (cores own input pages and scatter) | **rejected** — superseded by the output-owner split | — | — | R2/R3 output pages need 2–4 input pages owned by different cores → cross-core gather or duplicate reads; cores owning unselected pages idle | — |
| `strided_gather` | **rejected** — `step_mode` `"strided"` is outside TARGET (every production step is 1); `validate()` refuses it | any step > 1 | — | — | — |

Regime-pinned tests are required and present in the acceptance test (`R1`, `R2` with `row_shift ∈ {0, 3, 7, 17, 25, 31}`, `R3`). The predicate does not depend on the grid, so no regime is grid-reachable only.

### Traffic ranking

Bytes per tier for the candidate splits (no time estimates). x is read-only and out is write-only, so there is no dependent axis and no combine to rank.

| Rank | Candidate split | DRAM | Cross-core | Notes |
|------|-----------------|------|------------|-------|
| 1 (chosen) | `out_page`-owner, contiguous ranges over the full grid | minimum (R1, R2); ≤ 2×/4× source pages for R3 only | 0 | fills the grid for every shape with ≥ grid-many pages; one-row outputs (e.g. 1940 pages × 1 tile row) still spread by column |
| 2 | tile-row owner (whole output tile rows per core) | same bytes | 0 | the single-row class has 1–16 tile rows → ≤ 16 busy cores (catalog `width_split`); dominated by rank 1 |
| 3 | `in_page`-owner | same for R1; R2/R3 need a gather of 2–4 pages from other cores | ≥ 1 page per output page for R2/R3 | rejected row |
| — | dependent-axis split + combine | — | — | no dependent axis exists |
| — | read-once-and-multicast of x | — | — | x has no reuse-shared axis; only the device-form starts word is reuse-shared (`R1-dev-bcast`) |

### Stall-shadow check

| Waiting stage | Waits on | Work independent of the wait | Decision |
|---------------|----------|------------------------------|----------|
| reader `noc_async_read_barrier` (once per block) | its own block-k reads | issuing block k+1's reads — legal unconditionally (a pure copy: no data dependence, no floating-point order to preserve, so no precision gate) | Phase 0: `stage_depth_blocks = 2` lets the reader run a full block ahead of the writer; the barrier still drains the reader's queue once per block. The tagged reorder (issue k+1, retire only k) is lamp L2 — it changes only which barrier retires a block, not the loop nest |
| writer `cb_wait_front` | the reader's block k | none: every write needs staged bytes. R3's per-page assembly of block k overlaps the reader's block k+1 reads by construction | — |
| writer `noc_async_writes_flushed` (once per block, before `cb_pop_front`) | block k's write data leaving L1 | assembling/issuing block k+1 if already staged | kept: the slot must not be refilled before its data left L1 |
| reader starts-page read (refinement) | `starts[sd]` | nothing on the reader (every source address depends on the begin); the writer has no data yet | recorded: no fill exists |

### Block schedule

```cpp
for (uint32_t block_idx = 0; block_idx < num_blocks_this_core; ++block_idx) {
    stage_source_block(block_idx);   // reader
    assemble_block(block_idx);       // writer, R3 only
    store_block(block_idx);          // writer
}
// writer: noc_async_write_barrier() before returning (BB-03)
```

| Block operation | Block shape | Resident across it | Intended frequency of fixed costs |
|-----------------|-------------|--------------------|-----------------------------------|
| `stage_source_block` | `block_tiles` output pages (`last_block_extent` on the last block) × `src_slots` slots | odometer state; R2 band tables (per valid-row class) | one `cb_reserve_back` + one `cb_push_back` (nominal count) + one `noc_async_read_barrier` per block. Reads: R1 one page read per output page; R2 one read per merged band envelope (2 for a single row, ≤ 4 per slot); R3 one page read per needed source page. Accessor construction and band tables once per kernel launch |
| `assemble_block` (R3) | `block_tiles` output pages, each a 32×32 element raster | the staged source slots of the block | no handshake of its own (operates between the writer's `cb_wait_front` and its writes); one in-place raster pass per output page |
| `store_block` | `block_tiles` output pages | R2 run tables (per valid-row class) | one `cb_wait_front` + one `cb_pop_front` (nominal) + one `noc_async_writes_flushed` per block. Writes: R1/R3 one page write per output page; R2 one write per merged row run (2 for a single row, ≤ 8 per page). One `noc_async_write_barrier` per kernel launch |

### Perf lamps

| Lamp | Why the default may be wrong here | Nearby alternative to measure |
|------|-----------------------------------|-------------------------------|
| L1 overlap / fill (block size, depth) | p1 moves 2142 pages per core; a 64 KiB block costs one block of fill and one of drain per core, a smaller block more barriers | p1 (case 703) and p3 (case 733): `BLOCK_BYTES_CAP ∈ {16, 32, 64, 128} KiB` × `stage_depth_blocks ∈ {2, 3, 4}` |
| L2 read pipelining across the barrier | the per-block drain-everything barrier idles this core's read queue once per block (stall-shadow) | `dataflow_kernel_lib::set_read_trid` / `async_read_barrier_with_trid` (`ttnn/cpp/ttnn/kernel_lib/local_copy_helpers_dataflow.hpp:147,158`) with 2–3 blocks in flight on p1; catalog `trid_double_issue` |
| L3 grid synchronization (participation) | outputs of 1–2 pages per core (8-, 12-, 160-page cases) pay per-core launch and one DRAM round trip each; fewer cores with a few pages each may finish sooner | `min_pages_per_core ∈ {1, 2, 4, 8}` on cases 191 (8 pages), 1443 (12), 815 (160) |
| L4 bank stagger | every core walks its range in the same direction over 8 round-robin banks | rotate each core's start within its range (catalog `bank_stagger`), p1 and p3 |
| L5 R2 band vs. whole page | when a slot needs ≥ 16 rows the band is 2–4 reads where one whole-page read would do (two_tiles rule case, `M2 = 31` cases) | per-slot switch: whole-page read when the slot's needed rows ≥ 16 |

### Catalog evidence used

| Catalog entry | Knob it informed |
|---------------|------------------|
| `double_buffer` | `stage_depth_blocks = 2`; batching reads per barrier saturates at a small block |
| `trid_double_issue` | `BLOCK_BYTES_CAP` (single-core GB/s at block 16 ≫ per-core share at full grid); lamp L2 |
| `tile_reorder` | R1/R3 move whole 2 KB pages in one transaction; sub-page transfers only where they cut bytes (R2) |
| `noc_placement` | `split_work_to_cores(..., row_wise=True)`; reader on NoC0, writer on NoC1 (descriptor defaults) |
| `width_split` | split by `out_page`, not by tile row, so one-row outputs fill the grid |
| `bank_stagger` | lamp L4 |
| `split_reader`, `read_coalescing` | not used: the reader is not issue-bound at grid scale, and interleaved pages cannot be coalesced across pages |

## Dataflow Strategy

| Stage | Format | Mechanism | Notes |
|-------|--------|-----------|-------|
| x in DRAM → `cb_src_stage` | x's storage bytes (bfloat16 tile pages; R2: partial bands at source offsets) | reader (NCRISC, NoC0): `noc_async_read` from `TensorAccessor::get_noc_addr(page, offset)` | R1 whole page into the slot; R2 band envelopes at the source offsets; R3 whole pages into their slots |
| `cb_src_stage` (R3) | same bytes | writer (BRISC) RISC-V element copies, in place in slot 0, raster order | valid rows × valid cols only |
| `cb_src_stage` → out in DRAM | same bytes | writer (BRISC, NoC1): `noc_async_write` to `TensorAccessor::get_noc_addr(out_page, offset)` | R1/R3 whole page; R2 row runs at output offsets |
| Compute | — | none | there is no compute kernel: any unpack/DEST/pack pass converts formats (int32 above 2^24, block-format exponents) |

TARGET placement (`memory_config`) — the same logical split seen as a support axis:

| Value | Cut | Character of the cut axis | Class | Realization |
|-------|-----|---------------------------|-------|-------------|
| `dram_interleaved` | contiguous `out_page` ranges | independent | built | R1 / R2 / R3 |
| `l1_interleaved` | same | independent | knob-turn | accessor from the tensor (buffer type L1, 16 B alignment); out resident in L1 counts against 1,347,776 B per core: `ceil(out_pages / num_l1_banks) · page_bytes` plus `cb_bytes_per_core` |
| `l1_width_sharded` | output width shards, pinned by the shard spec | `tile_col`: independent | knob-turn (no combine) | `R1-shard`: core = output shard owner (out shard spec = x's, unchanged); out through a CB backed on the output shard (zero copy, no DRAM writer); sources through the sharded TensorAccessor (core-local when `b3` is shard-aligned, NoC pull from the neighbour otherwise; x is immutable during the program, so no semaphore); cores with no output shard run nothing |

Other TARGET axes:

| Axis value | Class | What changes |
|------------|-------|--------------|
| dtype float32 / int32 / uint32 (TILE) | knob-turn | `page_bytes` 4096, `face_row_bytes` 64 (still a multiple of every alignment); kernels unchanged |
| dtype bfloat8_b / bfloat4_b | knob-turn for R1; scheme addition `R2/R3-bfp` | R1 copies whole pages bit-exactly |
| layout ROW_MAJOR | new regime `RM_stick_gather` | page = stick |
| begins_source device_tensor | new regime `R1-dev` | reader prologue reads `starts[sd]`; descriptor key without the begin |

## Work Distribution

| Field | Value |
|-------|-------|
| Work unit | an output page; a block = `block_tiles` consecutive output pages of a core's range |
| Grid | `device.compute_with_storage_grid_size()` — a runtime parameter (part of the descriptor key), never a literal |
| Per-core work | `(num_cores, all_cores, group_1, group_2, pages_g1, pages_g2) = ttnn.split_work_to_cores(grid, tensor_out_pages, row_wise=True)`; cores enumerated in the same row-wise order receive consecutive ranges `[core_first_page, core_first_page + core_pages)` |
| Remainder | group-2 cores own `pages_g1 − 1` pages; every core's last block may be ragged (`last_block_extent`), pushed/popped at the nominal count |
| Tile geometry | per image and with `ceil`: `tensor_out_row_tiles = ceil(M2/32)` for every `(i0, i1)`, never `floor(M0·M1·M2/32)`; `tensor_out_pages = M0·M1·ceil(M2/32)·ceil(M3/32)` |
| Per-core start state | host decomposes `core_first_page` into `(i0, i1, tr, tc)` and computes the first source page `q` (host div/mod, once per core) |
| Kernel groups (BB-17) | one reader and one writer kernel over `all_cores` (≤ 2 rectangular ranges from the row-wise split); per-core differences are runtime args only |
| Regime selection | `select_regime` above; regime chosen per call, compiled in via a define so the other regimes' code is absent from the binary |
| Values that must never be CT | anything derived from a device-tensor begin (refinement) |

## Circular Buffers

| Semantic Name | Index | Page Size | Num Pages | Sizing rationale | Format | Producer | Consumer | Lifetime |
|---------------|-------|-----------|-----------|------------------|--------|----------|----------|----------|
| `cb_src_stage` | 0 | `x.buffer_page_size()` (2048 B bfloat16 tile) | `stage_depth_blocks · block_tiles · src_slots` | spans `out_page` (`block_tiles`) and `src_slot` (`src_slots`); streams over `lead0`, `lead1`, `tile_row`, `tile_col` | x's dtype (raw storage bytes; never unpacked) | reader | writer | whole program |

Phase 0 has exactly one CB. R1 needs no output CB (the staged page is the output page), R2 writes runs straight from the stage (no assembly buffer), R3 assembles in place in slot 0. Refinements add `cb_begin_scratch` (reader-private, one aligned starts page, `R1-dev`) and the shard-backed output CB (`R1-shard`). Full ledger: `l1_ledger.md`.

## Block Operation Realization

| # | Block operation | Block shape | Helper? | Input CB (semantic name, pages, state) | Output CB (semantic name, pages) | CB state after |
|---|-----------------|-------------|---------|----------------------------------------|----------------------------------|----------------|
| 1 | `stage_source_block` | `block_tiles` × `src_slots` | no (dataflow) | x (DRAM pages) | `cb_src_stage`, `block_tiles · src_slots` reserved → pushed | block pushed; slot `t·src_slots + j` holds source slot j of output page t |
| 2 | `assemble_block` (R3) | `block_tiles` pages | no | `cb_src_stage`, waited by the writer | `cb_src_stage` slot 0 of each page, in place | still owned by the writer (not popped) |
| 3 | `store_block` | `block_tiles` pages | no | `cb_src_stage`, waited (R1/R2) or already waited (R3) | out (DRAM pages) | flushed, then popped (nominal) |

Slot layout per output page: R2 slot 0 = top (`q`), slot 1 = bottom (`q + in_stride_row`); R3 slot index = `row_sel · col_slots + col_sel` with `col_slots = 1 + (col_shift + min(M3, 32) > 32)`. Unneeded slots are reserved but neither read nor written.

### Sub-page geometry (TILE, Phase 0)

`es` = element bytes; `face_row_bytes = 16·es` (32), `face_bytes = 256·es` (512), `page_bytes = 1024·es` (2048). Face `(fr, h)` = index `2·fr + h` at `(2·fr + h)·face_bytes`; tile row ρ, column χ is in face `(ρ div 16, χ div 16)` at `((ρ mod 16)·16 + (χ mod 16))·es`. `s = row_shift`, `v = valid_rows(tr)`; there are at most two valid-row classes per call (`v = 32`, and `last_row_valid = M2 − 32·(tensor_out_row_tiles − 1)`), so both tables are built once per launch.

| Item | Rule |
|------|------|
| R2 rows needed per slot | top: in-tile rows `[s, min(32, s + v))`; bottom (iff `s + v > 32`): `[0, s + v − 32)` |
| R2 band envelopes (reader) | for each slot range `[ρa, ρb)` and `fr ∈ {0, 1}`: local rows `[a, b) = [max(ρa, 16fr), min(ρb, 16fr + 16)) − 16fr`; if non-empty, for `h ∈ {0, 1}`: bytes `[(2fr+h)·face_bytes + a·face_row_bytes, (2fr+h)·face_bytes + b·face_row_bytes)` → `align_down`/`align_up` to `NOC_DRAM_READ_ALIGNMENT_BYTES`; merge touching envelopes; read `[lo, hi)` from source page offset `lo` into the slot at offset `lo` |
| R2 row runs (writer) | split output rows `[0, v)` at `k = 16` and at every `k` with `(s + k) mod 16 == 0` (≤ 4 runs). Run `[k0, k1)`: `g0 = s + k0`, `sel = g0 div 32`, `ρ0 = g0 mod 32`; for `h ∈ {0, 1}`: source `slot_sel + (2·(ρ0 div 16) + h)·face_bytes + (ρ0 mod 16)·face_row_bytes`, destination offset `(2·(k0 div 16) + h)·face_bytes + (k0 mod 16)·face_row_bytes` in output page p, length `(k1 − k0)·face_row_bytes`. Merge the two halves when the run is a whole face on both sides |
| R2 example, case 587 (`s = 31`, `v = 1`) | reads: `[1472, 1536)` and `[1984, 2048)` of one source page; writes: 32 B to offsets 0 and 512 of the output page |
| R3 in-place assembly (writer) | for `y` in `[0, valid_rows(tr))` ascending, `x` in `[0, valid_cols(tc))` ascending: element `(row_shift + y, col_shift + x)` of the 2×2 source block (slot by `div 32`, position by `mod 32`) → element `(y, x)` of slot 0. Safe in place: the source position is strictly later in raster order than `(y, x)` whenever the source is slot 0, so it is never overwritten before it is read. Copy in runs contiguous within one source face row and one destination face row, element-wise ascending (`es`-wide loads/stores), then write slot 0 as page p |

## API Mapping

| Block operation | Type | Function | File:Line | Template Params / Args | Input CB | Output CB | Which params are block knobs |
|-----------------|------|----------|-----------|------------------------|----------|-----------|------------------------------|
| `stage_source_block` | raw_api | `cb_reserve_back` / `cb_push_back` | `tt_metal/hw/inc/api/dataflow/dataflow_api.h:406` / `:208` | `(cb_src_stage, block_tiles · src_slots)` | — | `cb_src_stage` | `block_tiles`, `src_slots` |
| `stage_source_block` | raw_api | `get_write_ptr` | `dataflow_api.h:325` | `(cb_src_stage)` | — | `cb_src_stage` | — |
| `stage_source_block` | raw_api | `TensorAccessor::get_noc_addr(page_id, offset)` | `tt_metal/hw/inc/api/tensor/tensor_accessor.h:165` | source page `q` (+ slot offset), band offset `lo` | — | — | — |
| `stage_source_block` | raw_api | `noc_async_read` | `dataflow_api.h:546` | `(src_noc_addr, slot_addr + lo, hi − lo)`; R1/R3: whole page | — | `cb_src_stage` | — |
| `stage_source_block` | raw_api | `noc_async_read_barrier` | `dataflow_api.h:1921` | once per block | — | — | — |
| `store_block` | raw_api | `cb_wait_front` / `cb_pop_front` | `dataflow_api.h:479` / `:260` | `(cb_src_stage, block_tiles · src_slots)` | `cb_src_stage` | — | `block_tiles`, `src_slots` |
| `store_block` | raw_api | `get_read_ptr` | `dataflow_api.h:346` | `(cb_src_stage)` | `cb_src_stage` | — | — |
| `store_block` | raw_api | `noc_async_write` | `dataflow_api.h:875` | `(slot_addr + src_off, out.get_noc_addr(p, dst_off), len)` | `cb_src_stage` | out | — |
| `store_block` | raw_api | `noc_async_writes_flushed` | `dataflow_api.h:1979-1980` | once per block, before `cb_pop_front` | — | — | — |
| end of writer | raw_api | `noc_async_write_barrier` | `dataflow_api.h:1950-1951` | once per launch (BB-03) | — | — | — |
| `assemble_block` | raw_api | RISC-V `volatile tt_l1_ptr` loads/stores of width `es` | — | element runs per R3 rule | `cb_src_stage` | `cb_src_stage` slot 0 | — |
| host | — | `ttnn.split_work_to_cores` | `ttnn/cpp/ttnn-nanobind/operations/core.cpp:471` | `(grid, tensor_out_pages, row_wise=True)` | — | — | core assignment |
| host | — | `ProgramDescriptor.custom_program_hash`, `ttnn.compute_program_descriptor_hash` | `ttnn/cpp/ttnn-nanobind/program_descriptors.cpp:1140-1143`, `ttnn/cpp/ttnn/operations/generic/generic_op_nanobind.cpp:85-104` | set once per memoized descriptor | — | — | — |

**Helpers considered and rejected** (apply to every `raw_api` row above):

| Candidate | File:Line | Why it cannot express the shape |
|-----------|-----------|----------------------------------|
| `copy` (eltwise convenience) | `ttnn/cpp/ttnn/kernel_lib/eltwise/api/convenience.hpp:109-113` | a `CopyTile → PackTile` chain through DEST: a compute pass that unpacks and repacks. The contract forbids passing values through another format (int32 above 2^24, block-format exponents), cannot shift rows within a tile, and adds a compute kernel to a DRAM-bound copy |
| `tilize` / `untilize` | `ttnn/cpp/ttnn/kernel_lib/untilize_helpers.hpp:163`, `ttnn/cpp/ttnn/kernel_lib/tilize_helpers.hpp:206` | layout conversion inside the program is the several-program chain in disguise and also goes through the unpacker; the TILE layout is preserved here |
| `read_sticks_for_tilize` | `ttnn/cpp/ttnn/kernel_lib/tilize_helpers_dataflow.hpp:87-93` | reads ROW_MAJOR sticks into a tilize CB; the source here is TILE pages and band envelopes |
| `set_read_state` / `read_with_state` | `ttnn/cpp/ttnn/kernel_lib/local_copy_helpers_dataflow.hpp:74-75, 94-95` | stateful single-packet reads from one fixed **local** L1 source (`.inl:66` asserts the size); the reader's sources are DRAM pages at varying addresses and sizes |
| `set_read_trid` / `async_read_barrier_with_trid` | `local_copy_helpers_dataflow.hpp:147, 158` | applicable — it is lamp L2 (measured alternative), not the Phase 0 default |
| `McastArgs` | `ttnn/cpp/ttnn/kernel_lib/mcast/kernel/mcast_args.hpp` | no reuse-shared operand in Phase 0; mechanism of the deferred `R1-dev-bcast` row |

No compute kernel exists, so `compute_kernel_hw_startup()` and reduce-scaler rules do not apply.

## Host Entry Contract

The entry point does no tensor transformation of its own: no `to_layout`, `tilize`, `untilize`, `pad`, `slice`, `reshape`, `typecast` or memory conversion on x or out, and no host read of a device tensor's values.

| Step | What | Detail |
|------|------|--------|
| 1 | `validate(...)` — first line of `g6_slice` | **(a) structural checks → `ValueError`:** `len(starts/ends/steps) != r`; any step < 1; any `M_d == 0` after clamping; `slice_dim` without `num_devices` or the reverse; either given with list begins; `steps` given with tensor begins; `slice_dim ∉ [−r, r−1]`; a starts/ends tensor that is not uint32 of shape `(r,)`; `N_sd % num_devices != 0`. These come first because the scenario dict cannot be built from malformed arguments. **(b)** build the scenario dict exactly as `eval/golden_tests/g6_slice/axes.py:classify_call` does and apply `INPUT_TAGGERS` in declaration order. **(c)** SUPPORTED per axis → `UnsupportedAxisValue`. **(d)** EXCLUSIONS (empty in Phase 0) → `ExcludedCell`. |
| 2 | Normalize | host list → `b`, `e`, `M` per *Math*; canonical 4D view: prepend `4 − r` dims of size 1 with begin 0 |
| 3 | Select regime | `select_regime` (see *Regimes*) |
| 4 | Allocate out | per *Output* |
| 5 | Descriptor | memoized per call signature (*Descriptor cache* below); on a hit only the two buffer addresses are patched |
| 6 | Dispatch | `ttnn.generic_op([x, out], descriptor)` — the only device op of the call |

| Tagger (TARGET axis name) | Returns |
|---------------------------|---------|
| `tag_dtype` → `"dtype"` | `scenario["dtype"]` |
| `tag_layout` → `"layout"` | `scenario["layout"]` |
| `tag_memory_config` → `"memory_config"` | `scenario["memory"]` ∈ `"dram_interleaved"`, `"l1_interleaved"`, `"l1_width_sharded"` |
| `tag_alignment` → `"alignment"` | `b` = normalized begins (tensor form: all 0 — read `M` only), `M` = `scenario["out_shape"]`. `"w_non_aligned"` if `b[-1] % 32 or M[-1] % 32`; else `"h_non_aligned"` if `b[-2] % 32 or M[-2] % 32`; else `"tile_aligned"` |
| `tag_rank` → `"rank"` | `len(scenario["shape"])` |
| `tag_begins_source` → `"begins_source"` | `scenario["begins_source"]` |
| `tag_step_mode` → `"step_mode"` | `"unit_step"` if `steps is None` or all 1, else `"strided"` |

| Registry item | Phase 0 value |
|---------------|---------------|
| `SUPPORTED` | `dtype: [ttnn.bfloat16]`, `layout: [ttnn.TILE_LAYOUT]`, `memory_config: ["dram_interleaved"]`, `alignment: ["tile_aligned", "w_non_aligned", "h_non_aligned"]`, `rank: [2, 3, 4]`, `begins_source: ["host_list"]`, `step_mode: ["unit_step"]` |
| `EXCLUSIONS` | `[]`. `"h_non_aligned"` never appears in any EXCLUSIONS entry, in any phase |
| `INVALID` | not declared by the op (lives in `feature_spec.py`) |
| Structural impossibilities | none beyond the two INVALID entries already in `feature_spec.py` |

| Host mechanism | Contract |
|----------------|----------|
| Descriptor cache (BB-07) | module-level dict keyed by `(device id, grid size, tuple(x.shape), tuple(x.padded_shape), x.dtype, x.layout, x.memory_config(), normalized b, M, regime)`. Tensor form (refinement): the key holds `slice_dim`, `num_devices` and **no begin value**. Miss: build the descriptor, then set `descriptor.custom_program_hash = ttnn.compute_program_descriptor_hash(descriptor)` once. Hit: overwrite the two buffer addresses (common runtime args of reader and writer) and dispatch — no rebuild. |
| Why patching is mandatory | the generic-op program hash covers CT args, CB configs and RT-arg **counts**, not RT values (`ttnn/cpp/ttnn/operations/generic/device/generic_op_device_operation.cpp:78-147, 307-324`); on a program-cache hit the RT values of the descriptor passed on *this* call are written into the cached program (`tt_metal/impl/program/program_descriptors.cpp:188-243`). A stale address in a reused descriptor is a wrong-data bug, not a cache miss. |
| Plan report (BB-08, BB-21) | `describe_plan(input_tensor, starts, ends, steps=None, *, slice_dim=None, num_devices=None) -> dict` with `regime`, `num_cores`, `core_ranges`, `block_tiles`, `src_slots`, `stage_depth_blocks`, `cb_bytes_per_core`, and per core its role (`reader+writer`) and `[core_first_page, core_first_page + core_pages)`. Same function the entry point uses, so the report is the plan that runs. |
| L1 refusal (BB-05) | host raises `RuntimeError` if `cb_bytes_per_core` (+ the core's share of an L1 out, refinement) exceeds 1,347,776 B. Unreachable in Phase 0 (≤ 128 KiB); never used to refuse a golden shape. |

## Key Risks and Gotchas

| Risk | Why it bites here | Mitigation in this design |
|------|-------------------|---------------------------|
| Reading past x for in-tile begins | row 4095 of 4096 (case 587) has no tile row below it; R3 likewise for the last tile column | bottom/right slots read only under the *Source bounds* conditions |
| Ragged last block corrupts L1 | `core_pages` is rarely a multiple of `block_tiles` | nominal push/pop always; only the moved page count varies |
| Sub-page reads misaligned on Blackhole | band reads are 64-B-granular and land at source offsets | slots are whole pages at 64-B-aligned CB bases; offsets copied unchanged; alignment constant from the kernel headers |
| Stale runtime args on a program-cache hit | begins of leading dims (e.g. 16 slices `[k:k+1]` of the same `(16, 6, 4096, 256)`) share one compiled program; only RT values differ | descriptor cache keyed by the normalized begins; addresses patched every call; acceptance test runs two begins on one x |
| Odometer carry with leading-dim slices | cases cut dims r−4 and r−1 together (296 cells); the carry from `tc` → `tr` → `i1` → `i0` jumps by different strides | host computes the start state; kernel carries add `in_stride_row`, `in_stride_lead1`, `in_stride_lead0` at each wrap; regime tests include rank-4 dims −4 and −1 |
| Output padding rows unwritten in R2 | consumers that read padding see stale DRAM | contract: padding is not part of the result; repeat and trace replay are compared on logical values (BB-06, BB-16) |
| Non-aligned `N2` with an aligned begin | input padding rows sit inside the last input tile | R1 copies them only onto output padding positions (`b + M ≤ N`); R2 reads only selected rows |
| Row-wise core order mismatch | `split_work_to_cores(..., row_wise=True)` orders cores row-major | runtime args assigned by iterating the same row-wise order |
| Kernel text size under trace (BB-04, BB-18) | three regimes in one source | regime selected by a define; unused paths are not compiled; no size-optimizing build flag |
| Zones (BB-21) | profiler build must show work and waits per role | zones `g6-stage-block`, `g6-assemble-block`, `g6-store-block`, `g6-wait-read`, `g6-wait-stage`, `g6-flush`; `describe_plan` dumps each core's role and range; zones compile to nothing in the plain build |
| Device begin baked into the program (refinement) | eager is right, trace replay is wrong | begin read on device every call; key without it; clamp per *Mechanism caps* |
