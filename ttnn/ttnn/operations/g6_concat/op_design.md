# Operation Design: g6_concat

## Overview

| Field | Value |
|-------|-------|
| Classification | data_movement (no compute kernel; values move as bytes, never through unpack/math/pack) |
| Goal | Join K interleaved device tensors along one dim into a newly allocated interleaved tensor, bit for bit, in ONE device program: every input element read once from its buffer, every output element written once. |
| Math | `d = dim mod r`; `o_k = n_0 + … + n_{k-1}`; `out[i_0,…,o_k+j,…,i_{r-1}] = x_k[i_0,…,j,…,i_{r-1}]`, `0 ≤ j < n_k` |
| References | `docs/explorations/g6_concat/{DOSSIER,math,tensix,risks,lessons}.md`; `eval/golden_tests/g6_concat/{feature_spec,helpers}.py`; `.claude/references/{blocking-model,l1-footprint-discipline,op-design-template}.md`; catalog `ttnn/ttnn/operations/examples/master.md` (entries `double_buffer`, `trid_double_issue`, `width_split`, `noc_placement`, `bank_stagger`, `tile_reorder`) |

## Parameters

| Name | Type | Required | Valid Range | Default | CT/RT |
|------|------|----------|-------------|---------|-------|
| `tensors` | `list[ttnn.Tensor]` | yes (positional) | `1 ≤ K ≤ K_MAX` (= 128, Mechanism caps); equal rank ∈ {3,4}, dtype, layout; equal sizes except along `d` | — | per-input base address: common RT arg; per-input extent along `d`: CT arg (part of the program) |
| `dim` | `int` | no (positional) | `[-r, r-1]`, canonicalized `d = dim + r if dim < 0` **before** any tagging/gating | `0` | host only (folds into the per-input run lengths / row counts) |
| `memory_config` | `ttnn.MemoryConfig \| None` | no (keyword) | interleaved DRAM or L1 | `None` → `ttnn.DRAM_MEMORY_CONFIG` (also when inputs live in L1) | host (output allocation); output accessor args CT |

## Tensors

### Input

| Property | Requirement |
|----------|-------------|
| Shape | rank 3 or 4; `x_k.shape[a] == x_0.shape[a]` for all `a ≠ d`; `n_k = x_k.shape[d] ≥ 1` |
| Dtype | shared by all inputs. Phase 0: `bfloat16`; TARGET adds `float32`, `bfloat8_b` |
| Layout | shared by all inputs. Phase 0: `TILE`; TARGET adds `ROW_MAJOR` |
| Memory | interleaved; each input read **in its own placement** (DRAM or L1, may differ per input). Phase 0: all DRAM |

### Output

| Property | Value |
|----------|-------|
| Shape | `x_0.shape` with `N = Σ n_k` at index `d` |
| Dtype | inputs' dtype (no cast) |
| Layout | inputs' layout |
| Memory | interleaved in `memory_config`'s buffer; DRAM interleaved when `memory_config is None`. Allocated by the entry point with `ttnn.allocate_tensor_on_device` (positional args), passed last to `ttnn.generic_op` |

### Registry axes (names exactly as `feature_spec.py` TARGET)

| Axis | Tagger (from the scenario dict / call) | TARGET | Phase 0 SUPPORTED |
|------|----------------------------------------|--------|-------------------|
| `dtype` | `tag_dtype`: shared dtype | bf16, f32, bf8b | bf16 |
| `layout` | `tag_layout`: shared layout | TILE, ROW_MAJOR | TILE |
| `concat_dim` | `tag_concat_dim`: `d = dim % r` first; `"w"` if `d == r-1`, `"h"` if `d == r-2`, else `"outer"` | outer, h, w | outer, h, w |
| `alignment` | `tag_alignment`: `"w_non_aligned"` if any `shape[-1] % 32`, elif any `shape[-2] % 32` → `"h_non_aligned"`, else `"tile_aligned"` | tile_aligned, h_non_aligned | tile_aligned, h_non_aligned |
| `input_buffer` | `tag_input_buffer`: all dram / all l1 / mixed | dram, l1, mixed | dram |
| `output_buffer` | `tag_output_buffer`: `memory_config`'s buffer, `"dram"` for None | dram, l1 | dram |

EXCLUSIONS (Phase 0): none. `validate()` builds the scenario dict
`{"shapes", "dim", "dtype", "layout", "input_buffers", "memory_config"}` from the call, applies the
taggers in the order above, checks SUPPORTED per axis then EXCLUSIONS, raising
`UnsupportedAxisValue` / `ExcludedCell`. Structural validation (empty list, rank ∉ {3,4}, rank /
dtype / layout mismatch, size mismatch off `d`, `dim ∉ [-r, r-1]`, `K > K_MAX`) raises `ValueError`
and runs before the tagging (the taggers need a well-formed list). The regime (below) is **not** a
registry axis — it is derived from shapes and never gates support.

Structural impossibilities beyond `feature_spec.py`'s INVALID: none found.

## Blocking Model

Semantics: `.claude/references/blocking-model.md`. Everything below this section realizes it.

**The op in one sentence:** every output page has exactly one source region (one input page, or —
only in the row-gather regime — a set of input row-pieces), and no output page depends on any
other. There is **no dependent axis and no reuse-shared operand**: the operand-reuse check over
(every input, the chosen split) finds that no input element feeds more than one output element, so
no input is re-read by any split and there is nothing to broadcast. All axes are independent and,
in output page order, contiguous — so they fold into **one linear output-page axis**, and the block
is a run of consecutive output pages (`blocking-model.md` §1: leading axes fold when the op treats
them identically and they are contiguous).

### Canonical address map (host-derived, per regime)

TILE geometry is per image and alignment-aware: `Ht(x) = ceil(x.shape[-2]/32)`,
`Wt(x) = ceil(x.shape[-1]/32)`, `slices(x) = prod(x.shape[:-2])`.

| Regime | `tensor_slices` (P) | Per-input run `run_k` (pages of `x_k` per slice) | `tensor_out_pages` |
|--------|--------------------|---------------------------------------------------|--------------------|
| page_relocation, `concat_dim == "outer"` | `prod(shape[:d])` | `n_k · prod(shape[d+1:r-2]) · Ht · Wt` | `P · Σ run_k` |
| page_relocation, `concat_dim == "h"` (offsets aligned) | `prod(shape[:r-2])` | `ceil(n_k/32) · Wt` | `P · Σ run_k` (= `P · ceil(N/32) · Wt`) |
| page_relocation, `concat_dim == "w"` | `prod(shape[:r-2]) · Ht` | `n_k / 32` (every production `n_k` is a multiple of 32; `w_non_aligned` is not in TARGET) | `P · Σ run_k` |
| row_gather (`concat_dim == "h"`, misaligned offsets) | `prod(shape[:r-2])` | rows: `n_k`; tiles `tensor_k_h_tiles = ceil(n_k/32)` | `P · ceil(N/32) · Wt` |

page_relocation: output page `g = p·R + offset_k + j` (`R = Σ run_k`, `offset_k = Σ_{m<k} run_m`)
← input page `p·run_k + j` of `x_k`. row_gather: output tile `g = (p·tensor_out_h_tiles + ht)·Wt + wt`;
output row `ρ = 32·ht + t` (`ρ < N`) ← row `ρ − o_k` of `x_k` in slice `p`, column-tile `wt`.

### Axes

| Axis | Character (+ one-clause reason) | Extent knob | Phase 0 value | Knob source | Core-assignment | Later unlock |
|------|--------------------------------|-------------|---------------|-------------|-----------------|--------------|
| `out_page` — the linear output-page axis (fold of slice × concat × inner, below) | independent — every output page has its own source, no cross-page combine | `block_pages` | `min(core_pages, BLOCK_PAGES_MAX)`; `BLOCK_PAGES_MAX = 32` | host constant `BLOCK_PAGES_MAX` in the program-descriptor module → CT arg `block_pages` of reader and writer; every CB page count and loop bound derives from it | contiguous ranges of `core_pages` output pages per core over the **full** compute grid (`split_work_to_cores(grid, tensor_out_pages, row_wise=True)`); grid is a host parameter (`device.compute_with_storage_grid_size()`), never inlined | knob-turn |
| `slice` (dims before `d`, or before the tile dims) | independent — a slice of the output is a slice of the inputs | folded into `block_pages` (contiguous in output page order) | folded | `block_pages` | folded into `out_page`'s split; a core range may cross slices | knob-turn |
| `concat` (dim `d`: the K source runs) | independent — each output position along `d` has exactly one source `(k, j)`; no result spans `d` | folded into `block_pages`; a block may cross input boundaries | folded | `block_pages` | folded; a core's range may start mid-input and span several inputs | knob-turn |
| `inner` (dims after `d` down to the tile grid; `Wt` for "h"; none for "w") | independent — per-element move | folded into `block_pages` | folded | `block_pages` | folded | knob-turn |
| `tile_row` (32 rows inside one output tile — row_gather only) | independent per row, but **atomic per tile**: an output tile is written whole, so its rows meet in one core's L1 before the write | whole tile (32 rows), not split | 32 rows | mechanism (TILE page) | never split across cores: a core assembles each output tile it owns from all its sources | none (splitting a tile across cores is a combine — rejected regime `split_tile_across_cores`) |
| `input_list` (k = 0..K-1) | independent — inputs are disjoint sources | not a block axis; walked inside a block as `(k, j)` advances | — | CT arg `num_inputs` | not assigned separately (rejected regime `core_group_per_input`) | none |

Every stage the scheme introduces: the only intermediate is the row_gather **staging window**
(per output tile, private to the reader core that owns the tile); it has the same `out_page` axis and
is assigned with it. No core collects, coordinates or hands out work — every core is a peer running
the same reader/writer on its own range; no stage computes on one core while others wait.

### Buffer-depth knobs

| CB | Depth knob | Phase 0 value | What the depth buys |
|----|------------|---------------|---------------------|
| `cb_out_tiles` | `out_buffer_depth` (host constant → CB capacity `out_buffer_depth · block_pages`) | 2 | reader fills block `b+1` (NoC0 reads) while writer drains block `b` (NoC1 writes) — read/write overlap (catalog `double_buffer`) |
| `cb_row_stage` (row_gather only) | none — reader-private scratch, consumed inside the block that filled it | 1 block | nothing to overlap: its producer and consumer are the same reader thread, sequentially |

### Mechanism caps

| Mechanism | Cap on which extent | Clamp | What happens unclamped |
|-----------|--------------------|-------|------------------------|
| NoC DRAM-read alignment (`NOC_DRAM_READ_ALIGNMENT_BYTES` = 64 B on Blackhole, 32 B on Wormhole; `tt_metal/hw/inc/internal/tt-1xx/blackhole/noc/noc_parameters.h:379`; sanitizer requires `src ≡ dst (mod align)`, `tt_metal/hw/inc/internal/debug/sanitize.h:746`) | row_gather row-piece reads: a bf16 face row is 32 B, a f32 face row 64 B | DRAM reads cover the aligned window `[align_down(src_off, A), align_up(src_off+len, A))` landing in `cb_row_stage` at the **same offset mod A**; rows are then placed with an L1→L1 self-read (16 B alignment). `A` = `hal` DRAM alignment passed as CT arg — never a literal | NoC alignment fault under `--dev`; silently shifted rows otherwise (risk R3) |
| Staging window size | `cb_row_stage` bytes per output tile | ≤ `2 · tile_bytes` (each window ≤ 2× its live bytes when `2·face_row_bytes ≥ A`: bf16/f32 on BH and WH) → `STAGE_PAGES_PER_TILE = 2`; host asserts `2·face_row_bytes ≥ A` (fails only for bf8b, whose row_gather is a deferred row) | staging overflow into the next tile's window → wrong rows |
| CB base alignment | every CB slot address | CBs are allocated at DRAM alignment (`tt_metal/impl/program/program.cpp:2509`); every page size used (`tile_bytes`, `2·tile_bytes`) is a multiple of 64 B, so every slot base ≡ every DRAM page base (mod A) | — (holds by construction; reader `ASSERT`s `slot_addr % A == 0` under `--dev`) |
| Output tile atomicity | `tile_row` extent | a block covers whole output tiles; a core owns whole tiles | two cores writing parts of one tile → race / combine |
| Runtime-arg capacity | `K` | per-input data lives in **common** runtime args (1 address per input) and CT args (1 extent per input): `K_MAX = 128` refused on host with `ValueError` beyond it (no golden INPUT exceeds 16) | refused launch / truncated args |
| Ring-wrap of `cb_out_tiles` | push/pop quantum | every push/pop is the nominal `block_pages` (a ragged last block narrows only the NoC transfers, never the push count) → capacity `out_buffer_depth · block_pages` is closed by whole blocks | CB pointer desync on the block after a short one |

### Regimes

| Regime | Status | Predicate | Block | Data movement vs. minimum | What a bigger block buys |
|--------|--------|-----------|-------|---------------------------|--------------------------|
| `page_relocation` | **built** | `concat_dim != "h"` **or** `all(o_k % 32 == 0 for k in 1..K-1)` (≡ every `n_k`, `k < K-1`, is a multiple of 32) | `block_pages` consecutive output pages; each page = one whole input page | **minimum**: each input page crosses its buffer's NoC boundary once (read), each output page once (write); zero cross-core. Input padding rows (h_non_aligned with "w"/"outer", or the last input of an aligned "h") land only in output padding rows (per-image tile padding), never in logical rows | per block: one read barrier (reader), one writes-flushed (writer), one CB reserve/push + wait/pop pair, one `(k,j)` walk restart; pipeline fill/drain = one block per core |
| `row_gather` | **built** | `concat_dim == "h"` **and** `any(o_k % 32 != 0 for k in 1..K-1)` (production: Qwen 79/89/143, 16/2/8 inputs of height 1) | `block_pages` consecutive output tiles; each tile assembled from row pieces of up to 32 `(input, input tile)` sources | each **logical** input row crosses its buffer once, inside an aligned window ≤ 2× its bytes (bf16 on BH: a 32 B face row arrives in a 64 B window; Qwen 79 reads 3.97 MB vs 1.99 MB logical vs 63.6 MB for whole tiles); each output tile written once; plus one core-local L1→L1 pass (staging → slot). Minimum is unreachable for bf16 on BH because a 32 B piece cannot be read from DRAM at a 32 B-misaligned offset (Mechanism caps) | per block: one stage-read barrier + one place-read barrier + one CB handshake; the row-piece schedule is computed per tile (not per block) — bigger blocks amortize only the barriers |
| `row_gather_direct` | deferred — successor of `row_gather` on the same structure: a piece whose `src_off ≡ dst_off (mod A)` (and every f32 piece) can be read from DRAM straight into the slot, skipping staging and the L1 pass. Not built in Phase 0 because the window must not clobber a neighbour row of the slot, which needs a per-piece overlap rule; the staging path is always correct. Reachable: same reader walk, a per-piece branch | as `row_gather`, per piece | per piece | removes the core-local pass for congruent pieces (half the pieces of Qwen 79) | as `row_gather` |
| `whole_tile_gather` (read each contributing input tile whole into L1, place rows locally) | rejected — superseded by `row_gather` | — | — | Qwen 79: 63.6 MB DRAM read for 1.99 MB of live rows (32×); also forces one core to own every output tile an input tile straddles to keep "read once" | — |
| `input_stationary` (split input pages across cores; each core writes its pages to their output positions) | rejected — superseded by the output-stationary split | — | — | equal bytes for page_relocation; for row_gather an output tile's rows come from several cores → cross-core combine (semaphores + NoC payload) or a read-modify-write second pass on the output (forbidden) | — |
| `core_group_per_input` (each input to its own core group) | rejected — superseded by output-stationary | — | — | equal bytes; imbalanced (Qwen 18: 108 vs 84 slices) and K-dependent core ranges (BB-17) | — |
| `split_tile_across_cores` (cores own row bands of one output tile) | rejected — no shape needs it: the smallest row_gather output (Qwen 89) has 1940 tiles, > grid | — | — | adds a cross-core combine per tile | — |
| `one_program_per_input` / delegate to an existing op / host round trip | rejected — BB-01, spec rule 1 | — | — | K programs, or a second pass | — |
| ROW_MAJOR page relocation / stick assembly | deferred — `layout = ROW_MAJOR` is a TARGET refinement; Phase 0 SUPPORTED is TILE. Reachable: "outer"/"h" are `page_relocation` with pages = sticks (same walk); "w" is `row_gather`'s structure with byte pieces of a stick instead of face rows (same staging + place) | layout RM | `block_pages` sticks | minimum (outer/h); window overhang ≤ A per piece (w) | as above |
| bf8b `row_gather` | deferred — no TARGET cell (production bf8b is "w", tile-aligned → `page_relocation`); a bf8b face row's 1-byte exponents are below every NoC alignment, needing a RISC-V byte copy | `dtype = bf8b ∧ row_gather predicate` | — | — | — |

Regime selection is a pure host function `select_plan(shapes, d, dtype, layout)` → `{regime,
block_pages, num_cores, core ranges}`; the op exposes the last selected plan (`g6_concat.last_plan`
attribute or `g6_concat.plan_for(tensors, dim, memory_config)`) so a call can report it (BB-08).
Regime-pinned tests: `page_relocation` ← `(1,1,2048,128)+(1,1,256,128)` along dim 2 (an "h" call
that must NOT take row_gather) and every "w"/"outer" case; `row_gather` ← Qwen 79/89 and one
straddling case (`(1,1,31,64)+(1,1,33,64)+(1,1,1,64)` along dim 2). Both regimes are reachable on
every grid (the predicate is shape-only).

### Traffic ranking

Counted in crossings per tensor (DRAM for Phase 0 tensors), no time estimates.

| Rank | Candidate split | Input crossings | Output crossings | Cross-core | Verdict |
|------|-----------------|-----------------|------------------|------------|---------|
| 1 | **Output-stationary over the linear output-page axis** (contiguous output ranges, full grid, row-wise) | 1 per input page (page_relocation); 1 per logical row inside ≤2× windows (row_gather) | 1 | none | **chosen** — minimum for page_relocation, the best reachable for row_gather; one split serves both regimes |
| 1 (tie) | Input-stationary over the concatenated input pages | 1 | 1 (page_relocation); row_gather: > 1 or a combine | combine for row_gather | rejected: ties only where row_gather is absent; tie-break = one split for both regimes |
| 3 | Split along the slice axis only / along the concat axis only | 1 | 1 | none | under-fills the grid on `(1,1,16,62080)` (1 slice) and on Qwen 80 (1 tile row) — `width_split` catalog finding; strictly dominated by rank 1 |
| 4 | "Dependent split + combine" | n/a — no axis is dependent; the nearest analogue is splitting one output tile's rows across cores (`split_tile_across_cores`) | 1 | a combine per tile | rejected: adds traffic, saves none |

Occupancy is the second, separate step: `split_work_to_cores(grid, tensor_out_pages, row_wise=True)`
spreads pages over `min(grid_cores, tensor_out_pages)` cores (row-wise = the DRAM-facing axis,
catalog `noc_placement`), then the coarsest block within each core's range is taken.

### Block schedule

Per core, `num_blocks_this_core = ceil(core_pages / block_pages)`, `last_block_extent = core_pages − (num_blocks_this_core − 1)·block_pages`.

```cpp
for (uint32_t block_idx = 0; block_idx < num_blocks_this_core; ++block_idx) {
    // reader (NCRISC, NoC0)
    load_block(block_idx);          // page_relocation: whole input pages -> cb_out_tiles slots
    // -- or, row_gather --
    stage_rows_block(block_idx);    // aligned DRAM windows of every row piece -> cb_row_stage
    place_rows_block(block_idx);    // L1->L1 self-reads: stage -> cb_out_tiles slots; zero rows >= N
    // writer (BRISC, NoC1)
    store_block(block_idx);         // cb_out_tiles slots -> output pages
}
```

| Block operation | Block shape | Resident across it | Intended frequency of fixed costs |
|-----------------|-------------|--------------------|-----------------------------------|
| `load_block` | `block_pages` output pages (actual `last_block_extent` on the last block) | the `(p, k, j)` cursor (carried, not recomputed) | 1 `cb_reserve_back(block_pages)` · all page reads issued back-to-back · **1** `noc_async_read_barrier` · 1 `cb_push_back(block_pages)` per block. Accessors (K + 1) constructed once per kernel |
| `stage_rows_block` | `block_pages` output tiles × their row pieces (≤ 2 windows per piece: column faces 0/1) | the `(p, ht, wt)` cursor; `cb_row_stage` region of the block | all window reads of the block issued back-to-back · **1** read barrier per block |
| `place_rows_block` | same tiles; pieces split at src/dst 16-row half and input-tile boundaries | `cb_row_stage` (read), `cb_out_tiles` block (written) | 1 `cb_reserve_back` before staging, all L1→L1 piece reads issued back-to-back · **1** read barrier · zero-fill of rows `≥ N` (last tile row of a slice only) by RISC-V stores · 1 `cb_push_back(block_pages)` |
| `store_block` | `block_pages` output pages | — | 1 `cb_wait_front(block_pages)` · page writes issued back-to-back · **1** `noc_async_writes_flushed` · 1 `cb_pop_front(block_pages)` per block; **1** `noc_async_write_barrier` at kernel end (BB-03) |

**Stall-shadow check.** The reader's read barrier and the writer's write flush are the only waits;
the double-buffered `cb_out_tiles` already schedules the other kernel's block into each one. Within
row_gather, the place step depends on the stage step's data, so the stage barrier has no
independent work of the *same* block to shadow it; the next block's staging is independent and can
be issued before the current block's place barrier — that is the `trid` perf lamp below, not a
Phase 0 reorder (no floating-point order exists to gate: nothing is computed).

**Profiling (BB-21).** Each block operation and each blocking wait (`cb_reserve_back`,
`cb_wait_front`, barriers) sits inside its own `DeviceZoneScopedN` zone, compiled in only for the
zone build; the host can dump `{core: role}` (every active core: reader+writer peer; inactive
cores: none).

### Perf lamps

| Lamp | Why the default may be wrong here | Nearby alternative to measure |
|------|-----------------------------------|-------------------------------|
| Overlap (block size) | `BLOCK_PAGES_MAX = 32` is below the L1-coarsest (≈ 160 bf16 pages at depth 2 in 1.35 MB) on purpose: at p2/p3 (~756 pages/core on 130 cores) a 160-page block makes pipeline fill + drain ~40% of a core's stream, while catalog `trid_double_issue` measured a large-CB one-block loop within 3% of best — barrier cost is already amortized at 32. | `BLOCK_PAGES_MAX ∈ {8, 16, 64, 128}` on p2/p3/p1 and Qwen 79 |
| Grid synchronization (few-tile calls) | Qwen 108/157/158 have 8 output pages → 8 cores of 1 page each; per-core launch and per-range dispatch may dominate (risk R9) | `min_pages_per_core ∈ {1, 4, 8}` (cores = `ceil(tensor_out_pages / min_pages_per_core)`) measured as trace replay host time (lesson RB-053) |
| Bank stagger | every core's range starts at `i·core_pages`; when `core_pages % num_dram_banks == 0` all cores start on the same bank | rotate each core's first page within its range (catalog `bank_stagger`) |
| Read issue depth / trid | `load_block` drains every read at the block barrier; `stage_rows_block` → `place_rows_block` serializes | `set_read_trid`/`async_read_barrier_with_trid` over 2–4 blocks in flight (`local_copy_helpers_dataflow.hpp:147,158`) |
| Row-piece issue cost | Qwen 79: 1940 tiles × 16 inputs × 2 windows = 62080 DRAM reads of 64 B + as many L1 placements; may be issue-bound on NCRISC | split staging reads across NCRISC/BRISC (catalog `split_reader`); stateful local reads `set_read_state`/`read_with_state` (`local_copy_helpers_dataflow.hpp:74,94`) for the 32 B placements |

## Dataflow Strategy

| Stage | Format | Mechanism | Notes |
|-------|--------|-----------|-------|
| Input buffer → reader L1 (page_relocation) | whole TILE pages, bytes untouched | `noc_async_read_page(page, accessor_k, slot_addr)` per page, one barrier per block | accessor per input: one shared `TensorAccessorArgs` set (Phase 0: all inputs DRAM interleaved, equal page size) + per-input base address from common RT args. Mixed placement (refinement): two arg sets (DRAM, L1) and a per-input placement CT flag |
| Input buffer → `cb_row_stage` (row_gather) | aligned byte windows of face rows | `noc_async_read(accessor_k.get_noc_addr(page, win_off), stage_addr + win_off_in_stage, win_len)` | `win_off ≡ stage slot offset (mod A)` by construction (Mechanism caps) |
| `cb_row_stage` → `cb_out_tiles` slot (row_gather) | face-row runs | L1→L1 self-read via `dataflow_kernel_lib::local_addr` (`local_copy_helpers_dataflow.hpp:57`) + `noc_async_read` | 16 B alignment; pieces split at 16-row face halves (src and dst) and at input-tile boundaries |
| `cb_out_tiles` → output buffer | whole TILE pages | `noc_async_write_page(page, out_accessor, slot_addr)`; `noc_async_writes_flushed` per block, `noc_async_write_barrier` at end | output accessor from the output tensor (DRAM or L1 interleaved per `memory_config`) |
| Tensix ↔ Tensix | — | none | no regime (built or deferred) needs inter-core communication |

**Placement axis (`input_buffer`, `output_buffer`) as regimes:** interleaved only — there is no
physical shard in TARGET. L1 inputs / L1 output change the accessor's bank map only (knob-turn:
accessor args + output allocation). Neither changes a split, so neither is a scheme-change.

**Row-gather address detail (one piece).** For output tile `(p, ht, wt)`, input `k` with rows
`[o_k, o_k+n_k)` intersecting `[32ht, min(32ht+32, N))`: input row `s_abs = ρ − o_k`, input tile
`(p·tensor_k_h_tiles + s_abs/32)·Wt + wt`, in-tile row `s = s_abs % 32`, out row `t = ρ % 32`. A piece
is a maximal run of `m` rows with constant `s/16`, `t/16` and input tile. For column face `c ∈ {0,1}`:
`src_off = ((s/16)·2 + c)·face_bytes + (s%16)·face_row_bytes`, `dst_off` likewise with `t`, length
`m·face_row_bytes` (`face_bytes = 256·elem`, `face_row_bytes = 16·elem`). Only rows `ρ < N` of the
input's logical rows are ever placed; input padding never reaches the output; output rows `≥ N`
(only in the last tile row of each slice) are written as zeros (deterministic output bits, BB-06).

## Work Distribution

| Field | Value |
|-------|-------|
| Work unit | one output page (a block = `block_pages` consecutive output pages) |
| Grid | `device.compute_with_storage_grid_size()` (full compute grid; host parameter, never inlined) |
| Per-core work | `(num_cores, all_cores, g1, g2, pages_g1, pages_g2) = ttnn.split_work_to_cores(grid, tensor_out_pages, row_wise=True)`; core `i` (row-major order over `all_cores`) gets `core_start_page_i = Σ_{c<i} core_pages_c`, `core_pages_i ∈ {pages_g1, pages_g2}`. ≤ 2 groups → few rectangular core ranges per kernel (BB-17) |
| Remainder | `split_work_to_cores` balances to ±1 page; the in-core ragged last block has `last_block_extent ≤ block_pages`. `tensor_out_pages` uses per-image `ceil` (Canonical address map) |
| Per-core RT args | `core_start_page`, `core_pages` (reader and writer). Common RT args: K input base addresses (reader), output base address (writer). CT args: `num_inputs`, per-input `run_k` (page_relocation) or `n_k` rows (row_gather), `P`, `Wt`, `tensor_out_h_tiles`, `block_pages`, `tile_bytes`, `A`, accessor args |
| Reader cursor init | page_relocation: `p = start / R`, `rem = start % R`, `k` = the input with `offset_k ≤ rem < offset_k + run_k`, `j = rem − offset_k`; then advance `j → k → p`. row_gather: `p, ht, wt` from `start` by div/mod |
| Program cache (BB-07, BB-16) | host memoizes the built `ProgramDescriptor` keyed on `(per-input shapes, d, dtype, layout, per-input buffer type, output memory_config, device grid)`; a hit only rewrites the K + 1 address common RT args. Trace replay allocates nothing beyond the output allocated at capture |

## Circular Buffers

| Semantic Name | Index | Page Size | Num Pages | Sizing rationale | Format | Producer | Consumer | Lifetime |
|---------------|-------|-----------|-----------|------------------|--------|----------|----------|-----------|
| `cb_out_tiles` | 0 | `tile_bytes` (= output `buffer_page_size()`, 2048 B bf16) | `out_buffer_depth · block_pages` | spans `out_page` → `block_pages`; depth 2 for read/write overlap; streams over `slice`/`concat`/`inner` (folded) and spans `tile_row` (whole tiles) | inputs' dtype (bytes are never interpreted) | reader | writer | whole program |
| `cb_row_stage` (row_gather only; not allocated in page_relocation) | 1 | `STAGE_PAGES_PER_TILE · tile_bytes` (= 4096 B bf16) | `block_pages` | spans `out_page` → `block_pages` (all windows of a block staged before one barrier); spans `tile_row` (≤ 2× a tile's live bytes) | inputs' dtype (raw bytes) | reader | reader (same thread; scratch addressed via `get_write_ptr`, no push/pop) | within one block (`stage_rows_block` → `place_rows_block`) |

Full ledger: `l1_ledger.md`.

## Block Operation Realization

| # | Block operation | Block shape | Helper? | Input CB (semantic name, pages, state) | Output CB (semantic name, pages) | CB state after |
|---|-----------------|-------------|---------|----------------------------------------|----------------------------------|----------------|
| 1 | `load_block` (page_relocation) | `block_pages` pages | no (dataflow, raw API) | — (input buffers via accessors) | `cb_out_tiles`, reserve/push `block_pages` | `block_pages` pushed (nominal even when ragged) |
| 2a | `stage_rows_block` (row_gather) | `block_pages` tiles × pieces | no | — | `cb_row_stage` (scratch, `block_pages` pages) | staged windows, read barrier complete |
| 2b | `place_rows_block` (row_gather) | `block_pages` tiles | `local_addr` helper for the self-read source | `cb_row_stage` | `cb_out_tiles`, reserve (before 2a) / push `block_pages` | `block_pages` pushed |
| 3 | `store_block` | `block_pages` pages | no | `cb_out_tiles`, wait/pop `block_pages` | — (output buffer) | popped after `noc_async_writes_flushed` |

## API Mapping

| Block operation | Type | Function | File:Line | Template Params / Args | Input CB | Output CB | Which params are block knobs |
|-----------------|------|----------|-----------|------------------------|----------|-----------|------------------------------|
| all (reader/writer) | raw_api | `TensorAccessor`, `TensorAccessor::get_noc_addr(page_id, offset)` | `tt_metal/hw/inc/api/tensor/tensor_accessor.h:71`, `:165` | `TensorAccessorArgs<CTA_OFFSET>` + base address | — | — | none |
| `load_block` | raw_api | `noc_async_read_page` | `tt_metal/hw/inc/api/dataflow/dataflow_api.h:1243` | `(page_id, accessor_k, slot_addr)` | — | `cb_out_tiles` | count per block = `block_pages` |
| `stage_rows_block`, `place_rows_block` | raw_api | `noc_async_read` | `tt_metal/hw/inc/api/dataflow/dataflow_api.h:547` | `(src_noc_addr, dst_l1_addr, len)` | — / `cb_row_stage` | `cb_row_stage` / `cb_out_tiles` | count per block = `block_pages` × pieces |
| `place_rows_block` | helper | `dataflow_kernel_lib::local_addr` | `ttnn/cpp/ttnn/kernel_lib/local_copy_helpers_dataflow.hpp:57` | `(addr, noc_id = noc_index)` | `cb_row_stage` | `cb_out_tiles` | none |
| `load_block`, `stage_rows_block`, `place_rows_block` | raw_api | `noc_async_read_barrier` | `tt_metal/hw/inc/api/dataflow/dataflow_api.h:1921` | — | — | — | once per block |
| `store_block` | raw_api | `noc_async_write_page` | `tt_metal/hw/inc/api/dataflow/dataflow_api.h:1418` | `(page_id, out_accessor, slot_addr)` | `cb_out_tiles` | — | count per block = `block_pages` |
| `store_block` | raw_api | `noc_async_writes_flushed`, `noc_async_write_barrier` | `tt_metal/hw/inc/api/dataflow/dataflow_api.h:1980`, `:1951` | — | — | — | flushed once per block; barrier once at end |

**Helpers considered and rejected (every raw_api entry):**

| Candidate | File:Line | Why it cannot express this op |
|-----------|-----------|-------------------------------|
| `tilize` / `untilize` / `eltwise` `copy` / `reduce` / `matmul_block` (compute helpers) | `ttnn/cpp/ttnn/kernel_lib/eltwise_convenience.hpp` (`copy`), `tilize_helpers.hpp`, `untilize_helpers.hpp` | all route values through unpack → DEST → pack, a format path that is forbidden (spec rule 2: f32 low bits, bf8b groups would change) and that moves bytes the reader can place directly; nothing is computed, so there is no compute kernel |
| `tile_column_dataflow.hpp`, `tilize_helpers_dataflow.hpp` | `ttnn/cpp/ttnn/kernel_lib/tile_column_dataflow.hpp`, `tilize_helpers_dataflow.hpp` | read a single tensor's tile column / RM sticks for a tilize; neither walks K source buffers by a per-page source map nor places sub-tile row pieces |
| `read_with_state` / `set_read_state` | `ttnn/cpp/ttnn/kernel_lib/local_copy_helpers_dataflow.hpp:74`, `:94` | single fixed source per state; row_gather pieces vary in source and length per piece. Recorded as the "row-piece issue cost" perf lamp for the fixed-length 32 B case |
| `Mcast` / `McastArgs` | `ttnn/cpp/ttnn/kernel_lib/mcast/mcast_protocol.hpp` | no reuse-shared operand and no combine: nothing to multicast |

## Key Risks and Gotchas

| Risk | Why it bites here | Mitigation in this design |
|------|-------------------|---------------------------|
| Row-gather pieces misaligned for DRAM reads (R3) | Blackhole: 64 B DRAM read alignment vs 32 B bf16 face rows | aligned windows into `cb_row_stage` at congruent offsets, then L1→L1 placement; `A` is a CT arg from `hal`, not a literal; validated under `--dev` (NoC sanitizer) |
| Input padding in logical output (R2) | h-concat of height-1 inputs: 31 padding rows per input tile; test inputs are zero-padded | row_gather places only rows `j < n_k`; the poisoned-padding test (NaN / 1000.0) is in the acceptance test |
| Regime predicate taken from the alignment tag | an "h" call with h_non_aligned **only in the last input** is page_relocation; one with aligned last offsets but a misaligned middle input is row_gather | predicate uses `o_k % 32` for `k ≥ 1`, never the tag; regime-pinned tests both ways |
| Wrong row_gather offsets at face / tile boundaries | pieces must split at src half (16), dst half (16) and input-tile boundaries; production only exercises height-1 inputs | straddling test `(1,1,31,64)+(1,1,33,64)+(1,1,1,64)` dim 2 (rows cross both halves and tile edges) |
| `(k, j)` cursor across block and core boundaries | a core range can start mid-input, mid-slice; inputs with `run_k` smaller than a block (Qwen 157: runs of 2 and 6) | cursor initialized by div/mod from `core_start_page`, advanced per page; K = 2..16 and K = 1 tested |
| Ragged last block | `core_pages % block_pages ≠ 0` | push/pop nominal `block_pages`; NoC transfers limited to `last_block_extent` |
| BB-03 | writer returns with writes in flight | `noc_async_write_barrier` at writer end; reader ends on a read barrier |
| Cache key / K-dependence (R6) | K = 1..16, per-input run lengths are CT args | memoized descriptor keyed on every shape, placement and dtype; a hit patches only addresses |
| Output padding non-determinism | row_gather slots are reused; rows `≥ N` would hold stale bytes | zero-fill rows `≥ N` in `place_rows_block` (only the last tile row of each slice) |
