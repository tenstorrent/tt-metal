# Operation Design: g6_slice

## Overview

| Field | Value |
|-------|-------|
| Classification | data_movement (pure copy: no compute kernel, no arithmetic, no format conversion) |
| Goal | Copy the elements of `x` selected by per-dim begin/end/step into a new tensor with `x`'s dtype, layout and placement, in **one** device program that reads only the bytes holding selected elements and writes only `out`. |
| Math | `out[i_0..i_{r-1}] = x[b_0 + s_0*i_0, ..., b_{r-1} + s_{r-1}*i_{r-1}]`, `M_d = ceil((e_d - b_d)/s_d)`, `b_d = clamp(starts[d] (+N_d if < 0), 0, N_d)`, `e_d = clamp(ends[d] (+N_d if < 0), b_d, N_d)`. Device-tensor form: `out = x[..., r0 : r0 + L, ...]` on dim `sd`, `r0 = floor(starts[sd]/32)*32`, `L = N_sd / num_devices`. |
| Precision | Bit copy. Every byte of `out` that is a logical element is a byte of `x`, moved by the NoC (or, for a W begin inside a tile, by RISC-V integer loads/stores). No byte passes through unpack/DEST/pack. NaN/inf/denormals/integers > 2^24 are preserved by construction. |
| References | `eval/golden_tests/g6_slice/feature_spec.py` (TARGET/INPUTS/INVALID), `eval/golden_tests/g6_slice/helpers.py` (oracle, grading = bitwise for bf16), `.claude/references/blocking-model.md`, `.claude/references/l1-footprint-discipline.md`, `.claude/references/noc-alignment-in-tt-metal.md`, `ttnn/ttnn/operations/examples/master.md` (entries cited under *Catalog evidence*), `ttnn/cpp/ttnn/kernel_lib/tile_column_dataflow.hpp`, `ttnn/cpp/ttnn/kernel_lib/local_copy_helpers_dataflow.hpp`, `tt_metal/hw/inc/api/dataflow/dataflow_api.h`, `tt_metal/hw/inc/api/tensor/tensor_accessor.h` |

## Parameters

| Name | Type | Required | Valid Range | Default | CT/RT |
|------|------|----------|-------------|---------|-------|
| `input_tensor` (x) | `ttnn.Tensor` | yes | rank 2–4; on device | — | tensor (accessor CT args; buffer address RT) |
| `starts` | `list[int]` of length r, or `ttnn.Tensor` uint32 ROW_MAJOR DRAM shape (r,) | yes | any int (normalized PyTorch-style) | — | host list → derived offsets are **RT**; tensor → read on device (never on host) |
| `ends` | same form as `starts` | yes | any int (clamped); tensor form: never read | — | host only |
| `steps` | `list[int]` of length r, or `None` | no | each ≥ 1; Phase 0 supports all-1 only | `None` = all 1 | host only (gates `step_mode`) |
| `slice_dim` (kw) | `int` | device form only | `[-r, r-1]`, canonicalized to `slice_dim + r` before any check | `None` | CT (selects the page stride the runtime begin is applied to) |
| `num_devices` (kw) | `int` | device form only | divides `N_sd` | `None` | CT (via output shape) |

There is no `memory_config`, `dtype` or `compute_kernel_config` parameter: `out` takes `x.dtype`, `x.layout`, `x.memory_config()`.

### Derived quantities (host, single source each)

All kernels see the op as **rank 4**: shape, begins and output shape are left-padded with size-1 / begin-0 dims (`N4`, `b4`, `M4`). Tile counts are per image and use `ceil`.

| Symbol | Definition | CT/RT | Consumers |
|--------|------------|-------|-----------|
| `tensor_in_ht_tiles`, `tensor_in_wt_tiles` | `ceil(N4[2]/32)`, `ceil(N4[3]/32)` | CT | reader (input page ids) |
| `tensor_out_ht_tiles`, `tensor_out_wt_tiles` | `ceil(M4[2]/32)`, `ceil(M4[3]/32)` | CT | reader, writer (walk carries) |
| `N4[1]`, `M4[1]` | input / output extent of `lead1` | CT | reader (input stride), walk carry |
| `rows_last_tile_row` | `M4[2] - 32*(tensor_out_ht_tiles - 1)` ∈ [1, 32] | CT | reader, writer (row class of the last tile row of each image) |
| `total_out_pages` | `M4[0]*M4[1]*tensor_out_ht_tiles*tensor_out_wt_tiles` | host | split |
| `lead_begin0`, `lead_begin1` | `b4[0]`, `b4[1]` | **RT** | reader |
| `first_input_tile_row` (q) | `b4[2] // 32` | **RT** | reader |
| `row_offset_in_tile` (off) | `b4[2] % 32` | **RT** | reader, writer |
| `first_input_tile_col` | `b4[3] // 32` | **RT** | reader |
| `col_offset_in_tile` (c) | `b4[3] % 32` | **RT** | reader |
| `core_start_page`, `core_pages` | from `split_work_to_cores` | **RT** (per core) | reader, writer |
| `core_start_coord` | `(o0, o1, th, tw)` of `core_start_page`, decomposed **on host** (no device division) | **RT** (per core) | reader, writer |
| `ROW_RUNS`, `W_SHIFT` | regime flags (see *Regimes*) | CT (compile out unused paths, BB-04/BB-18/RB-038) | reader, writer |
| `block_pages`, `cb_depth`, `page_bytes`, `face_row_bytes` | *Blocking Model*; `page_bytes = x.buffer_page_size()` (tile size of x's dtype), `face_row_bytes = page_bytes / 64` for 16/32-bit dtypes | CT | reader, writer, CB sizes |

Begins are **RT**, regime flags and shapes are **CT**: the generic-op program hash covers CT args, defines, core ranges and CB sizes but only the *count* of RT args (`ttnn/cpp/ttnn/operations/generic/device/generic_op_device_operation.cpp:83-103`), and RT values are overridden on every hit. So the 47 single-row slices of `(1,1,256,256)` (cases 191–235) or the 16 batch-index slices of `(16,6,4096,256)` share one compiled program each.

### Host contract

| Item | Requirement |
|------|-------------|
| Entry point order | `validate()` is the first line. Inside `validate()`: **(1)** structural checks raise `ValueError` — wrong rank (not 2–4), `starts`/`ends`/`steps` length ≠ r, a step < 1, mixed forms (one tensor, one list), `slice_dim` xor `num_devices`, either given with list begins, `steps` given with tensor begins, `slice_dim` outside `[-r, r-1]`, begin tensors not uint32 shape (r,), `num_devices` not dividing `N_sd`, any `M_d == 0`. **(2)** build the scenario dict (same keys as `feature_spec.py`), apply `INPUT_TAGGERS` in declaration order. **(3)** SUPPORTED per axis → `UnsupportedAxisValue`. **(4)** EXCLUSIONS (empty) → `ExcludedCell`. Structural errors therefore never surface as a support refusal. |
| `INPUT_TAGGERS` | `tag_dtype` → `scenario["dtype"]`; `tag_layout` → `scenario["layout"]`; `tag_memory_config` → `scenario["memory"]`; `tag_alignment` → with `b` the normalized begins and `M = scenario["out_shape"]`: `"w_non_aligned"` if `b[-1] % 32 or M[-1] % 32`, else `"h_non_aligned"` if `b[-2] % 32 or M[-2] % 32`, else `"tile_aligned"`; device form: `b ≡ 0`, read `M` only; `tag_rank` → `len(shape)`; `tag_begins_source` → `scenario["begins_source"]`; `tag_step_mode` → `"unit_step"` iff `steps is None` or all 1, else `"strided"`. Taggers read only the scenario dict (they are also driven by `eval/golden_tests/g6_slice/axes.py`). |
| `SUPPORTED` (Phase 0) | dtype `[bfloat16]`, layout `[TILE_LAYOUT]`, memory_config `["dram_interleaved"]`, alignment `["tile_aligned","w_non_aligned","h_non_aligned"]`, rank `[2,3,4]`, begins_source `["host_list"]`, step_mode `["unit_step"]`. `EXCLUSIONS = []`. `"h_non_aligned"` is never excluded in any phase. |
| One dispatch | Exactly one `ttnn.generic_op([x, out], descriptor)` (device form: `[x, starts_tensor, out]`). No other ttnn op touches `x` or `out` (no `to_layout`, `tilize`, `untilize`, `pad`, `slice`, `reshape`, `typecast`, memory conversion). `out` is allocated with `ttnn.allocate_tensor_on_device(ttnn.Shape(M), x.dtype, x.layout, device, x.memory_config())`. |
| Plan cache (BB-07) | The built `ProgramDescriptor` is cached in a module dict keyed by `(shape, padded_shape, dtype, layout, memory_config, b4, M4, device id, grid, effective BLOCK_PAGES_CAP, CB_DEPTH, MAX_CORES, slice_dim, num_devices)`. A hit rewrites only the RT slots that hold buffer addresses (x, out, starts tensor); nothing else is rebuilt. |
| Knob module constants | In `ttnn/ttnn/operations/g6_slice/g6_slice.py`: `BLOCK_PAGES_CAP = None` (None → derived from the L1 budget), `CB_DEPTH = 2`, `MAX_CORES = None` (None → full `compute_with_storage_grid_size()`), `L1_BUDGET_BYTES = 1347776`. Read at call time (not captured at import) so a test can override them. |
| Plan report (BB-08, BB-21) | `describe_plan(input_tensor, starts, ends, steps=None, *, slice_dim=None, num_devices=None) -> dict`, importable from `ttnn.operations.g6_slice`, runs the same validation and planning without dispatch and returns at least `{"regime", "num_cores", "block_pages", "cb_depth", "num_blocks_max", "l1_bytes_per_core", "core_roles"}`; `regime` ∈ `{"page_copy", "row_runs", "col_realign"}` in Phase 0; `core_roles` maps every used core to `"reader+writer"`. |
| Zones (BB-21) | Zone build only (no change to the plain build): reader zones `G6_FETCH_ISSUE`, `G6_READ_BARRIER`, `G6_CB_RESERVE`, `G6_COL_REALIGN`; writer zones `G6_CB_WAIT`, `G6_WRITE_ISSUE`, `G6_WRITE_FLUSH`, `G6_WRITE_BARRIER`. |
| Trace (BB-16) | Nothing on the call path allocates besides `out`, nothing is compiled after the first eager call of a signature; the begin of the device form is read by the kernel, so replay with a new begin needs no host work. |
| Completion (BB-03) | Writer ends with `noc_async_write_barrier()`; reader's last block ends with a read barrier before its last push. |

## Tensors

### Input

| Property | Requirement |
|----------|-------------|
| Shape | `(N_0, ..., N_{r-1})`, r ∈ {2,3,4}; no alignment requirement on any dim (TILE pages are padded per image) |
| Dtype | Phase 0: bfloat16. TARGET: + float32, int32, uint32, bfloat8_b, bfloat4_b |
| Layout | Phase 0: TILE. TARGET: + ROW_MAJOR |
| Memory | Phase 0: DRAM interleaved. TARGET: + L1 interleaved, L1 width-sharded |

### Output

| Property | Value |
|----------|-------|
| Shape | `(M_0, ..., M_{r-1})` (device form: `N` with `N_sd` replaced by `L`) |
| Dtype | `x.dtype` |
| Layout | `x.layout`; TILE positions beyond the logical shape are not part of the result and may hold anything (the row-run writer leaves them unwritten) |
| Memory | `x.memory_config()`; width-sharded: same shard spec (grid, shard shape, orientation), out fills the first `ceil(M_w/shard_w)` shards in orientation order |

## Blocking Model

Semantics: `.claude/references/blocking-model.md`. This op has one operand, no reduction and no reuse: **every axis is independent**, every selected byte is used exactly once, and the whole design question is which pages each core moves, in what transfer shape, and how many per handshake.

### Axes

| Axis | Character (+ one-clause reason) | Extent knob | Phase 0 value | Knob source | Core-assignment | Later unlock |
|------|--------------------------------|-------------|---------------|-------------|-----------------|--------------|
| `lead0` (dim r−4; extent 1 when r < 4) | independent — output index i reads input index `b4[0]+i`, no value spans the axis | `block_pages` (folded: contiguous in output page order, identical per-page work) | covered by the `block_pages` run | `_plan()` (fold is structural) | inside the contiguous page ranges | knob-turn |
| `lead1` (dim r−3; extent 1 when r < 3) | independent — same | `block_pages` (folded) | covered by the run | `_plan()` | inside the page ranges | knob-turn |
| `h` = output tile rows (dim r−2), `tensor_out_ht_tiles` per image | independent — each output row is one input row; a shifted output tile reads two input tiles but no value spans tiles | `block_pages` (folded) | covered by the run | `_plan()` | inside the page ranges | knob-turn |
| `w` = output tile columns (dim r−1), `tensor_out_wt_tiles` | independent — each output column is one input column | `block_pages` (folded) | covered by the run | `_plan()` | inside the page ranges | knob-turn |
| **`out_page`** = fold of `lead0·lead1·h·w` in output page order | independent | **`block_pages`** | `min(core_pages_max, block_pages_cap)` — the coarsest block that fits the L1 budget, or the whole per-core assignment | `BLOCK_PAGES_CAP` (None → `block_pages_cap = floor((L1_budget − fixed_bytes) / (slots_per_page × page_bytes))`, see *Mechanism caps*) → `_plan()` → CT arg; CB capacities derived from it | contiguous ranges `[core_start_page, core_start_page + core_pages)` over the **full grid** via `split_work_to_cores(grid, total_out_pages, row_wise=True)`; grid = `MAX_CORES` knob | knob-turn |
| stage data: fetched staging page (content of `cb_slice_pages`) | independent — one staging page per output page | `block_pages` (same axis) | same | same | same core as its output page | knob-turn |
| stage data: right-column staging page (`col_realign` only) | independent — one per output page | `block_pages` (same axis) | same | same | same core | knob-turn |
| stage data: device-form begin (refinement) | **reuse-shared by construction** — every core needs the same `starts[sd]`, it does not vary along `out_page` | none (one 4-B value) | 1 | — | replicated: each core reads it itself (no designated core; the broadcast row is `rejected`, see *Regimes*) | knob-turn |
| intra-tile rows / columns | not a block axis — per-page transfer geometry (`row_offset_in_tile`, `rows_this_tile_row`, `col_offset_in_tile`) | none: a block always covers whole tiles | 32 × 32 | — | follows the page | — |

Why the fold is honest: in every regime one output page is produced from a fixed, per-page computable set of input bytes, and output pages are contiguous in `(lead0, lead1, h, w)` order. Interleaved pages live on different DRAM banks, so no 2-D block shape can coalesce transfers that a 1-D run cannot; the block's only job is to amortize the per-handshake costs below, and a 1-D run does that with ±1-page balance and no rectangle factorization.

**Operand-reuse check** (split = `out_page` ranges): `x` varies along `out_page` (each core reads disjoint input bytes) → not reuse-shared. The device-form `starts` tensor does not vary → reuse-shared by construction → its broadcast row is in the regime table (`rejected`, 4-B payload).

**Stall-shadow check.** Stages that wait: (a) reader on its read barrier — independent work: issuing the next block's reads, which `cb_depth = 2` already allows and the trid lamp extends; (b) writer on `cb_wait_front` — it has no other work; (c) reader on `cb_reserve_back` — no other work; (d) device form, reader on the `starts` read — every `x` read depends on the begin, so nothing can fill it (one DRAM round trip per call). No algebraic reorder exists (no arithmetic); no floating-point order is involved.

### Buffer-depth knobs

| CB | Depth knob | Phase 0 value | What the depth buys |
|----|------------|---------------|---------------------|
| `cb_slice_pages` | `cb_depth` (`CB_DEPTH`) | 2 | reader fetches block k+1 (NoC0) while the writer stores block k (NoC1) — catalog `double_buffer` |
| `cb_right_col_tile` | none (fixed 1 block) | 1 | consumed by the reader itself between the block's barrier and its push; a second slot would sit idle (same thread) |

### Mechanism caps

| Mechanism | Cap on which extent | Clamp | What happens unclamped |
|-----------|--------------------|-------|------------------------|
| L1 budget BB-05 (1347776 B/core, minus L1 `out` bytes the core holds) | `block_pages` | `block_pages_cap = floor((L1_budget − fixed_bytes) / (slots_per_page × page_bytes))`, `L1_budget = min(L1_BUDGET_BYTES, ttnn.device.get_max_worker_l1_unreserved_size())`, `fixed_bytes = out_l1_bytes_per_core + 64 × num_cbs`, `slots_per_page = cb_depth` (`+1` in `col_realign`); bf16: `floor(1347712 / 4096) = 329` | launch refused, or a clash with the model's L1 that only the model shows |
| Core assignment | `block_pages` | `≤ core_pages_max = ceil(total_out_pages / num_cores)` | idle L1 (not wrong) |
| DRAM read alignment (BH 64 B, WH 32 B; `tt_metal/hw/inc/internal/tt-1xx/blackhole/noc/noc_parameters.h:379`): L1 dst ≡ NoC src (mod 64) | placement of every read run | every run lands at its **natural in-tile byte offset** in a page whose base ≡ 0 mod 64: CBs are allocated at DRAM alignment (`tt_metal/impl/program/program.cpp:2505-2509`) and every tile size (2048, 4096, 1088, 576) is a multiple of 64 | silent corruption (watcher: sanitize error) |
| DRAM write alignment (BH 16 B): src ≡ dst (mod 16) | writer run offsets | writer runs are whole face-rows: `face_row_bytes` = 32 B (bf16) / 64 B (32-bit) — both ends multiples of 32 | silent corruption |
| Element granularity of a W begin inside a tile | `col_offset_in_tile` ≠ 0 | not expressible as NoC runs (needs `(c·elem_bytes) % 16 == 0`); realigned by RISC-V column copies (`copy_tile_column`) after the read barrier (which invalidates the L1 cache, `tt_metal/hw/inc/api/dataflow/dataflow_api.h:1927-1932`) | wrong columns |
| Block-format shared exponents (refinement) | row/col runs for bfloat8_b/bfloat4_b | one exponent byte per face-row: `page_copy` only until `block_format_runs` is built | wrong values |
| CB ring wrap | per-handshake page count | every push/pop is the nominal `block_pages`; a ragged last block narrows only its NoC work (capacity = `cb_depth × block_pages` closes the ring) | pointer desync → hang |
| Page-id width | all ids are uint32 | largest golden input: `16·128·272 = 557056` pages | — |
| `NOC_MAX_BURST_SIZE` | single transfer size | `noc_async_read/write` split internally; pages ≤ 4096 B | — |

### Regimes

Domain counts are over the 868 Phase 0 production cases (bf16, TILE, DRAM interleaved, host list).

| Regime | Status | Predicate | Block | Data movement vs. minimum | What a bigger block buys |
|--------|--------|-----------|-------|---------------------------|--------------------------|
| `page_copy` | **built** — 670 cases (2.81 M output pages) | TILE ∧ host_list ∧ `off == 0` ∧ `c == 0` ∧ `M4[2] % 32 == 0`. CT: `ROW_RUNS=0, W_SHIFT=0` | `block_pages` whole tiles | Minimum named against DRAM: each selected input page crosses once (one 2-KB read), each output page once. Above logical bytes only when `M4[3] % 32 ≠ 0` (the last tile column carries ≤ 31 unselected columns into output padding); unavoidable at page granularity. | Per block, intended once: 1 `cb_reserve_back` + 1 `cb_push_back` + 1 read barrier (reader), 1 `cb_wait_front` + 1 `noc_async_writes_flushed` + 1 `cb_pop_front` (writer), one fill/drain bubble. p1 (703): 2143 pages/core at 130 cores → 7 blocks of 329. |
| `row_runs` | **built** — 198 cases (163730 pages), incl. all 77 begins inside a tile and the "two_tiles" rule case | TILE ∧ host_list ∧ `c == 0` ∧ (`off ≠ 0` ∨ `M4[2] % 32 ≠ 0`). CT: `ROW_RUNS=1, W_SHIFT=0` | `block_pages` whole tiles; per page ≤ 8 read runs + ≤ 8 write runs of whole face-rows | **Below** the page minimum: each logical row-slice (whole tile width, face-row granular) crosses DRAM once on read and once on write; unselected rows never move. A tile with `rv` logical rows moves `rv × 64` B each way (bf16) instead of 2048 — 32× fewer bytes for the single-row cases (e.g. `(1,1,16,62080)` row 15: 124 KB vs 4 MB). Full tile rows with `off == 0` are one 2-KB run (= `page_copy`). | Same per-block costs as `page_copy`. Run lists depend only on `(off, rv)` → derived once per kernel per row class (≤ 2 classes: `rv = 32`, `rv = rows_last_tile_row`), never per page. |
| `col_realign` | **built** (correctness path) — 0 production cases (every production TILE begin on the last dim is a multiple of 32); required because `"w_non_aligned"` is SUPPORTED with no EXCLUSIONS and a begin may lie anywhere inside a tile | TILE ∧ host_list ∧ `c ≠ 0`. CT: `W_SHIFT=1`, `ROW_RUNS=1` | `block_pages` tiles + `block_pages` right-column tiles | x: each output tile reads its left input tile column (face halves holding cols `[c,32)`) and, only when its logical columns reach past `32−c`, its right one (halves holding `[0,c)`) → ≤ 2× the selected bytes; out: logical rows only. | + one shared barrier for the block's left and right fetches; the realign pass (32 column copies per page, RISC-V stores) scales with pages, not amortized. |
| `device_begin_copy` | **deferred** — `begins_source "device_tensor"` is TARGET − SUPPORTED. Reachable without restructuring: it is `page_copy`'s walk with one runtime additive input-page offset `begin_page_offset = min(floor(idx/32), floor((N_sd − L)/32)) × page_stride_sd` (`page_stride` = 1 tile for sd = r−1, `tensor_in_wt_tiles` for sd = r−2, the product of later page extents for a leading dim; a leading-dim begin is `r0 = floor(idx/32)*32` elements), computed by a reader prologue that reads the `starts` page into `cb_slice_pages` slot 0 before the first fetch | `begins_source == device_tensor` | as `page_copy` (partial last tile row copied whole; extra rows land in output padding) | as `page_copy` + one 64-B DRAM read of `starts` per core per call; `ends` never read. The clamp keeps every read inside `x` for any stored value. | as `page_copy` |
| `stick_copy` | **deferred** — ROW_MAJOR is TARGET − SUPPORTED. Reachable: same `out_page` axis (pages = sticks), same reader→`cb_slice_pages`→writer topology; only the page geometry changes | layout ROW_MAJOR | `block_pages` sticks; per stick one read of `M4[3]·es` bytes at byte `b4[3]·es` of the input stick, landing at the same offset mod 64 in an aligned slot; RISC realign when `(b4[3]·es) % 16 ≠ 0` | minimum: each selected stick byte once each way | same handshakes; sticks are 20 B–17 KB, so handshake amortization weighs more than for tiles |
| `shard_gather` | **deferred** — `l1_width_sharded` is TARGET − SUPPORTED. The caller's shard spec pins core-assignment (the output shard owners: 34 of 36 cores in the model) and the per-core extent; block = the whole output shard (8 tiles). Reachable: `fetch_block` with an L1 source (local or remote input shard via the sharded accessor), and the writer's DRAM runs replaced by in-place local run copies into a CB backed on `out` (`ttnn.cb_descriptor_from_sharded_tensor`, zero copy; never re-read over the NoC) | memory_config == l1_width_sharded | one block = whole output shard | x: each selected byte crosses the NoC once, 0 hops when local (model: `b4[3] = 0`, every output shard is the same core's input shard); out: never crosses | one block per core: one handshake, one barrier |
| `block_format_runs` | **deferred** — bfloat8_b/bfloat4_b are TARGET − SUPPORTED and all 6 production block-format cases are `page_copy` (a knob-turn: page_bytes 1088/576). Shifted or partial rows need a byte-granular exponent realign step (new block op, no scheme change) | dtype ∈ {bf8_b, bf4_b} ∧ (`off ≠ 0` ∨ `M4[2] % 32 ≠ 0` ∨ `c ≠ 0`) | as `row_runs` | as `row_runs` | as `row_runs` |
| `col_realign_sliding` | **deferred** — successor of `col_realign`; no production TILE case has `c ≠ 0`, so its ≤ 2× read touches no production case. Reachable: keep output tile `tw`'s right-column page as `tw+1`'s left (same CBs, same walk) | `c ≠ 0` | as `col_realign` | x: 1× (each input tile column read once) | as `col_realign` |
| `whole_tile_bounce` | **rejected** — superseded by `row_runs` | `off ≠ 0` | read both whole input tiles, rotate in L1, write a whole page | x: 2 pages per output tile; 32× the logical bytes for single-row cases | — |
| `tile_row_read_drop` | **rejected** — forbidden (reads whole tile rows to drop columns); doubles x bytes on p1 | — | — | 2× x on p1 | — |
| `program_chain` | **rejected** — forbidden (BB-01/BB-02): untilize/slice/tilize/pad around a copy | — | — | ≥ 3 DRAM round trips | — |
| `compute_path_copy` | **rejected** — superseded by the DMA-only copy: an unpack→DEST→pack hop adds a TRISC stage and a format path (DEST conversion alters int32 > 2^24, may canonicalize NaN/denormals) for zero benefit | — | — | same bytes + L1 round trip | — |
| `begin_broadcast` | **rejected** — superseded by the per-core `starts` read: 4-B payload; a semaphore rendezvous costs more than the 64-B read it saves and adds a designated-core stage | device form | — | saves 64 B × cores | — |
| `dependent_split_combine` | **rejected** — no axis is dependent; there is nothing to combine | — | — | — | — |
| `resident_fast_path` | **rejected** — no byte is re-read, so residency saves nothing | — | — | — | — |
| `single_core` | **rejected** — forbidden; one core cannot draw the DRAM bandwidth p1 needs | — | — | — | — |

Regime-selection function (host, after normalization and 4-D padding; host-list TILE form):

```
off = b4[2] % 32 ; c = b4[3] % 32
regime = "col_realign" if c != 0 else ("page_copy" if off == 0 and M4[2] % 32 == 0 else "row_runs")
```

Regime-pinned tests are required (the acceptance test asserts `describe_plan(...)["regime"]` per case). The selection depends only on begins and output shape — never on the grid.

### Traffic ranking

Every split moves the same bytes (each selected byte crosses DRAM once each way, no cross-core traffic); the ranking is therefore decided by transfer shape, balance and occupancy, in that order.

| Candidate split | x DRAM crossings | out DRAM crossings | Cross-core traffic | Occupancy / balance | Verdict |
|-----------------|------------------|--------------------|--------------------|---------------------|---------|
| A. contiguous **output-page ranges** (`out_page`) | 1× selected | 1× | none | full grid whenever `total_out_pages ≥ cores`; ±1 page | **chosen** |
| B. `lead0·lead1` only | 1× | 1× | none | ≤ `M4[0]·M4[1]` cores (1 for every `(1, H, W)` case) | ties on bytes, loses occupancy |
| C. tile rows only | 1× | 1× | none | ≤ rows of tiles (1 for the `(1,1,1,62080)` outputs) | ties, loses occupancy (catalog `width_split`) |
| D. tile columns only | 1× | 1× | none | ≤ `tensor_out_wt_tiles` | ties, loses occupancy |
| E. 2-D rectangles (h × w) | 1× | 1× | none | full, imbalanced unless factorizable | ties; A balances without factorization |
| F. input-page ownership (a core writes every output row its input tiles feed) | 1× | 1×, but a shifted output page written by two cores (two partial writes) | none | full | ties on bytes, doubles write transactions, splits page ownership |
| G. split a dependent axis + combine | — | — | — | — | N/A: no dependent axis |

### Block schedule

Logical schedule per core (reader and writer are separate asynchronous kernels; block k+1's fetch overlaps block k's store):

```cpp
// device form only (refinement): read_begin();  once per core, before block 0
for (uint32_t block_idx = 0; block_idx < num_blocks_this_core; ++block_idx) {
    fetch_block(block_idx);            // reader: DRAM x -> cb_slice_pages (+ cb_right_col_tile), natural in-tile offsets
    realign_columns_block(block_idx);  // reader, col_realign only: in place on the block's cb_slice_pages pages
    store_block(block_idx);            // writer: cb_slice_pages -> DRAM out, row-rotated runs
}
```

| Block operation | Block shape | Resident across it | Intended frequency of fixed costs |
|-----------------|-------------|--------------------|-----------------------------------|
| `fetch_block` | `block_pages` output pages (last block: `last_block_pages`) | the other `cb_slice_pages` slot (being stored) | 1 reserve + 1 read barrier per block (+ 1 push when not `col_realign`); run lists per kernel; reads for all pages of the block issued back to back before the barrier |
| `realign_columns_block` | `block_pages` pages, 32 column copies each | the block's slot + `cb_right_col_tile` | once per block, after `fetch_block`'s barrier; then 1 push |
| `store_block` | `block_pages` pages | the other slot (being fetched) | 1 wait + 1 `noc_async_writes_flushed` + 1 pop per block; one `noc_async_write_barrier` per kernel (end) |
| `read_begin` (refinement) | 1 page of `starts` | — | once per kernel |

What `fetch_block` does per output page `(o0, o1, th, tw)` (`rv = 32`, or `rows_last_tile_row` for `th = tensor_out_ht_tiles − 1`; `nu = min(32 − off, rv)`; `nl = rv − nu`):

| Source | Input page | Rows read (natural positions in the staging page) |
|--------|-----------|---------------------------------------------------|
| upper tile | `((lead_begin0+o0)·N4[1] + lead_begin1+o1)·tensor_in_ht_tiles·tensor_in_wt_tiles + (q+th)·tensor_in_wt_tiles + (first_input_tile_col + tw)` | `[off, off + nu)` |
| lower tile (only if `nl > 0`) | upper + `tensor_in_wt_tiles` | `[0, nl)` |
| `col_realign`: right column (only if the page's logical columns `min(32, M4[3] − 32·tw)` exceed `32 − c`) | the same rows of input page + 1, into `cb_right_col_tile` | same row bands |

The two bands are complementary row sets, so one staging page holds both without collision; staging row `s` holds output row `(s − off) mod 32`. A row band maps to ≤ 4 contiguous byte runs (rows `[r0, r1)` ∩ `[0,16)` in faces 0/1, ∩ `[16,32)` in faces 2/3; a band covering a whole half tile merges its two faces into one 1024-B run; a whole tile is one 2048-B run). `store_block` writes output rows `[0, rv)` from staging rows `(i + off) mod 32`, split wherever the source or destination crosses a 16-row face boundary or the 32-row wrap: ≤ 8 runs per page (two column halves × ≤ 4 pieces, merged when both sides are whole half tiles). `realign_columns_block` (`col_realign`) rewrites each page in place, ascending `j = 0..31`: column `j` ← staging column `j + c` when `j + c < 32`, else `cb_right_col_tile` column `j + c − 32`; ascending order is safe because column `j + c ≥ j` is read before it is overwritten.

### Perf lamps

| Lamp | Why the default may be wrong here | Nearby alternative to measure |
|------|-----------------------------------|-------------------------------|
| Overlap — single-block cores | With `core_pages ≤ block_pages_cap` (every case under ~43 k pages on 130 cores) a core reads its whole assignment, then writes it: no read/write overlap inside the core. Irrelevant when the grid is DRAM-bound, visible when latency-bound. | `block_pages = ceil(core_pages / 4)` with `cb_depth` 2 and 4, on 179 (6144 pages) and 1112 (40960 pages) |
| Overlap — L1-filling block on p1 | 329-page blocks make each read barrier wait on 329 transactions; the reader cannot issue during it | `block_pages ∈ {32, 64, 128}` × `cb_depth ∈ {2, 4}` on 703 |
| Trid issue | The per-block read barrier drains the reader's queue (catalog `trid_double_issue`: 1.11–1.63× on BH single core) | trid-tagged double issue (`set_read_trid` / `async_read_barrier_with_trid`) on 703 |
| Bank stagger | Cores whose `core_start_page` coincide mod the DRAM bank count start on the same bank (catalog `bank_stagger`: 1.0–1.07× on interleaved sources) | rotate each core's walk start within its range on 703 and 730 |
| Grid synchronization | Outputs with ≤ ~2 × grid pages put 1 page on each of up to 130 cores; launch/per-core fixed cost may dominate | `num_cores = ceil(total_out_pages / 4)` on 213 (8 pages) and 1946 (160 pages) |
| Shifted full rows | For `off ≠ 0, rv = 32` the writer issues 8 face-run writes per page (catalog `tile_reorder`: whole-page writes beat face writes when DRAM-bound) — only the "two_tiles" rule case in Phase 0 | local L1 rotate (self-aimed NoC run copies, `local_copy_helpers_dataflow.hpp`) + one 2-KB page write |
| Partial rows near 32 | For `off == 0` and `rv ≥ 16` row runs save < 2× bytes for 2–3× the transactions | whole-page copy when `rv ≥ 16` (cases 1463/1478, `rv = 31`) |
| Left half only | The last tile column of a 12-wide output (730, 731) carries 20 unselected columns; its right face half need not move | runs restricted to face column half 0 when `min(32, M4[3] − 32·tw) ≤ 16` (half the bytes, 2 × 512-B runs per page) |

### Catalog evidence

| Catalog entry | Knob it informed |
|---------------|------------------|
| `noc_placement` | `split_work_to_cores(..., row_wise=True)`; reader on NCRISC/NoC0, writer on BRISC/NoC1 (descriptor defaults) |
| `double_buffer` | batch a whole block of reads per barrier (never read-one/barrier); `cb_depth = 2`; no further gain once the grid is DRAM-bound |
| `trid_double_issue` | trid lamp; "a large CB with the trivial one-block loop is within 3% of best" supports the coarse default |
| `tile_reorder` | whole-page transfers wherever the page is whole (`page_copy`, full rows with `off == 0`); the shifted-full-rows lamp |
| `width_split` | the flattened split fills the grid for one-tile-row outputs (`(1,1,1,62080)`) |
| `bank_stagger` | bank-stagger lamp |
| `split_reader` | not applied: only pays when a data-movement RISC-V is measured issue-bound |

## Dataflow Strategy

| Stage | Format | Mechanism | Notes |
|-------|--------|-----------|-------|
| DRAM x → reader | x's tile pages (bf16, 2048 B) | `noc_async_read` of byte runs at `accessor.get_noc_addr(page_id, run_offset)` on NoC0 | runs land at their natural in-tile offset (DRAM read alignment) |
| reader → writer | staging page = output page with rows rotated by `off` (identity when `off == 0`) | `cb_slice_pages`, `block_pages` per handshake, `cb_depth` slots | `col_realign`: realigned in place before the push |
| writer → DRAM out | output tile pages | `noc_async_write` of face-row runs at `out_accessor.get_noc_addr(page_id, run_offset)` on NoC1 | only logical rows written; `noc_async_writes_flushed` before each pop, `noc_async_write_barrier` at the end |
| (device form) DRAM starts → reader | uint32 page of `starts` | one `noc_async_read` into `cb_slice_pages` slot 0, before the first fetch | value kept in a register; slot then reused for data |

No compute kernel exists; there is no Tensix-to-Tensix traffic in any built or deferred interleaved regime. `shard_gather` reads remote input shards with plain NoC reads (no semaphores: the input is static for the program's duration).

### TARGET axes as regimes

| TARGET value | Phase 0 | Regime | Cost class | What changes |
|--------------|---------|--------|------------|--------------|
| dtype bfloat16 | built | `page_copy` / `row_runs` / `col_realign` | — | — |
| dtype float32, int32, uint32 | refinement | same three | knob-turn | `page_bytes` 4096, `face_row_bytes` 64, `copy_tile_column<…, uint32_t>`; no conversion anywhere |
| dtype bfloat8_b, bfloat4_b | refinement | `page_copy` (all production cases) / `block_format_runs` | knob-turn / new block op | page 1088 / 576 B; exponent realign for shifted or partial rows |
| layout ROW_MAJOR | refinement | `stick_copy` | regime addition on the same scheme (same axis, topology, CB) | page = stick; byte-offset reads |
| memory_config l1_interleaved | refinement | same three | knob-turn (placement) | buffer type; `out_l1_bytes_per_core = ceil(total_out_pages / num_l1_banks) × aligned_page_bytes` enters the L1 budget |
| memory_config l1_width_sharded | refinement | `shard_gather` | knob-turn class (the caller cut the independent `w` axis; no combine), placement change | CB on `out`'s shard; core set = output shard owners (≤ 2 rectangles in the model grid) |
| begins_source device_tensor | refinement | `device_begin_copy` | knob-turn | reader prologue + runtime page offset; `slice_dim` canonicalized (`+r` if negative) before checks |
| alignment tile_aligned / w_non_aligned / h_non_aligned | built | all | — | `h_non_aligned` never excluded |
| rank 2 / 3 / 4 | built | 4-D padding | — | — |
| step_mode unit_step | built | — | — | `"strided"` is not in TARGET; refused by `validate()` |

### Structural impossibilities

None beyond `feature_spec.py`'s INVALID (block formats in ROW_MAJOR).

## Work Distribution

| Field | Value |
|-------|-------|
| Work unit | a block of `block_pages` consecutive output pages |
| Grid | full `device.compute_with_storage_grid_size()` (110–130 Tensix on Blackhole), or `MAX_CORES` when set |
| Per-core work | `split_work_to_cores(grid, total_out_pages, row_wise=True)` → group 1 cores get `core_pages_max`, group 2 one fewer; cores in row-major order so the used set is ≤ 2 rectangles (BB-17); one reader and one writer kernel over the union, per-core RT args carry `core_start_page`, `core_pages`, `core_start_coord` |
| Blocks per core | `num_blocks_this_core = ceil(core_pages / block_pages)`; `last_block_pages = core_pages − (num_blocks_this_core − 1)·block_pages`; push/pop counts stay nominal (`block_pages`) |
| Remainder | alignment-aware: `total_out_pages = M4[0]·M4[1]·ceil(M4[2]/32)·ceil(M4[3]/32)` — tile padding per image, never `floor(batch·H/32)`; the per-page row class uses `rows_last_tile_row` on the last tile row of every image |
| Walk | incremental `(o0, o1, th, tw)` with carries `tw → th → o1 → o0`, started from the host-decomposed `core_start_coord`; input page ids incremental from the same counters |
| Regime selection | the function in *Regimes*; regime-pinned tests required |

## Circular Buffers

| Semantic Name | Index | Page Size | Num Pages | Sizing rationale | Format | Producer | Consumer | Lifetime |
|---------------|-------|-----------|-----------|------------------|--------|----------|----------|----------|
| `cb_slice_pages` | 0 | `page_bytes` (= `x.buffer_page_size()`; bf16 2048) | `cb_depth × block_pages` | spans `out_page` (`block_pages` per slot, `cb_depth` slots); `lead0/lead1/h/w` folded into `out_page` | x's dtype | reader | writer | whole program |
| `cb_right_col_tile` | 1 (allocated only when `W_SHIFT`) | `page_bytes` | `block_pages` | spans `out_page` (one right-column tile per page of the block in flight); single block | x's dtype | reader | reader (private scratch, no handshake) | `col_realign` programs; between a block's fetch and its realign |

## Block Operation Realization

| # | Block operation | Block shape | Helper? | Input CB (semantic name, pages, state) | Output CB (semantic name, pages) | CB state after |
|---|-----------------|-------------|---------|----------------------------------------|----------------------------------|----------------|
| 1 | `fetch_block` | `block_pages` pages | no — raw NoC (see API Mapping) | — (DRAM `x`) | `cb_slice_pages`, `block_pages` reserved; `col_realign`: + `cb_right_col_tile`, `block_pages` | pushed `block_pages` (not `col_realign`); reserved and filled (`col_realign`) |
| 2 | `realign_columns_block` (`col_realign` only) | `block_pages` pages × 32 columns | yes — `copy_tile_column` | `cb_slice_pages` (reserved slot, filled) + `cb_right_col_tile` | `cb_slice_pages`, same slot, in place | pushed `block_pages` |
| 3 | `store_block` | `block_pages` pages | no — raw NoC | `cb_slice_pages`, wait `block_pages` | — (DRAM `out`) | popped `block_pages` after `noc_async_writes_flushed` |

## API Mapping

| Block operation | Type | Function | File:Line | Template Params / Args | Input CB | Output CB | Which params are block knobs |
|-----------------|------|----------|-----------|------------------------|----------|-----------|------------------------------|
| `fetch_block` | raw_api | `cb_reserve_back` / `get_write_ptr` / `cb_push_back` | `tt_metal/hw/inc/api/dataflow/dataflow_api.h:406` / `:325` / `:208` | `(cb_slice_pages, block_pages)` | — | `cb_slice_pages` | `block_pages` |
| `fetch_block` | raw_api | `TensorAccessor::get_noc_addr(page_id, offset)` | `tt_metal/hw/inc/api/tensor/tensor_accessor.h:165` | input page id, run offset within the page | — | — | — |
| `fetch_block` | raw_api | `noc_async_read(src_noc_addr, dst_l1_addr, size)` | `tt_metal/hw/inc/api/dataflow/dataflow_api.h:547` | one call per run (whole page = one run) | — | `cb_slice_pages` / `cb_right_col_tile` | — |
| `fetch_block` | raw_api | `noc_async_read_barrier()` | `tt_metal/hw/inc/api/dataflow/dataflow_api.h:1921` | once per block | — | — | frequency set by `block_pages` |
| `realign_columns_block` | helper | `dataflow_kernel_lib::copy_tile_column<dfb_id, T>(src_addr, src_col, dst_addr, dst_col)` | `ttnn/cpp/ttnn/kernel_lib/tile_column_dataflow.hpp:92` | `dfb_id = cb_slice_pages` (tile shape), `T = uint16_t` (bf16) / `uint32_t` (32-bit); src = same page or `cb_right_col_tile` page | `cb_slice_pages`, `cb_right_col_tile` | `cb_slice_pages` (in place) | none (per-column traversal inside the block op) |
| `store_block` | raw_api | `cb_wait_front` / `get_read_ptr` / `cb_pop_front` | `tt_metal/hw/inc/api/dataflow/dataflow_api.h:479` / `:346` / `:260` | `(cb_slice_pages, block_pages)` | `cb_slice_pages` | — | `block_pages` |
| `store_block` | raw_api | `noc_async_write(src_l1_addr, dst_noc_addr, size)` + `TensorAccessor::get_noc_addr` | `tt_metal/hw/inc/api/dataflow/dataflow_api.h:876`, `tensor_accessor.h:165` | one call per run | `cb_slice_pages` | — | — |
| `store_block` | raw_api | `noc_async_writes_flushed()` / `noc_async_write_barrier()` | `tt_metal/hw/inc/api/dataflow/dataflow_api.h:1980` / `:1951` | flushed once per block before the pop; barrier once at kernel end | — | — | frequency set by `block_pages` |
| trid lamp (not Phase 0) | helper | `dataflow_kernel_lib::set_read_trid` / `async_read_barrier_with_trid` | `ttnn/cpp/ttnn/kernel_lib/local_copy_helpers_dataflow.hpp:147` / `:158` | trid per block | — | — | — |
| host | — | `ttnn.split_work_to_cores`, `ttnn.TensorAccessorArgs`, `ttnn.CBDescriptor`, `ttnn.KernelDescriptor` (`ReaderConfigDescriptor` / `WriterConfigDescriptor`), `ttnn.generic_op`, `ttnn.allocate_tensor_on_device` | `.claude/references/ttnn-python-utility-bindings.md`, `.claude/references/generic_op_template/` | — | — | — | — |

**Helpers considered and rejected (raw_api rows).** No compute phase exists, so the compute helper table (tilize/untilize/reduce/eltwise/matmul) has nothing to cover and there is no `compute_kernel_hw_startup` call (no compute kernel). Dataflow helpers in `ttnn/cpp/ttnn/kernel_lib/` checked:

| Candidate | File:Line | Why it cannot express `fetch_block` / `store_block` |
|-----------|-----------|------------------------------------------------------|
| `set_read_state` / `read_with_state` / `local_addr` | `local_copy_helpers_dataflow.hpp:74`, `:95`, `:57` | self-aimed L1→L1 reads only (`local_addr` builds a source at `my_x/my_y`), and `read_with_state` bakes one transfer size into the NoC state; the fetch reads DRAM pages of other banks with per-run sizes |
| `read_sticks_for_tilize` / `write_sticks_after_untilize` | `tilize_helpers_dataflow.hpp:88`, `:130` | ROW_MAJOR sticks from one contiguous page range packed in tilize stride, pushing tile-height groups; Phase 0 moves tile pages, and the slice's stick sequence (refinement) jumps between images and needs no tile grouping |
| `fill_tile_column` / `copy_tile_column` | `tile_column_dataflow.hpp:70`, `:92` | RISC-V stores within L1; used for `realign_columns_block`, cannot move DRAM bytes |
| `Mcast` / `McastArgs` | `ttnn/cpp/ttnn/kernel_lib/mcast/` | no operand is multicast (the only shared operand, `starts`, is a 4-B read per core) |

The gap that would let a helper own `fetch_block`/`store_block`: a "page-run gather" helper taking a TensorAccessor, a page-id sequence and a ≤ 8-entry run list, issuing the runs into a CB block and returning after one barrier.

## Key Risks and Gotchas

| Risk | Why it bites here | Mitigation in this design |
|------|-------------------|---------------------------|
| Lower tile read past `x` | Row 4095 of `(16,4096,1280)` (case 587) is the last row of the last tile; there is no lower tile | lower band read only when `nl > 0`; `rv` clipped by `rows_last_tile_row`; never read a tile for rows it does not supply |
| Rows past `N_{r-2}` in a partial input tile | `(1,1,16,62080)` has one 16-row tile; rows 16..31 are input padding | only rows `< M4[2]` of the output (hence `< N4[2]` of the input) are read or written |
| Mod-64 DRAM read alignment | 32-B face-row runs at odd row offsets (`off` odd, e.g. 1390 `off = 1`) | runs always land at their natural in-tile offset; the writer — whose alignment is mod 16 — applies the rotation |
| Writer pops before writes leave L1 | the reader overwrites a slot still being written → nondeterministic output (BB-06) | `noc_async_writes_flushed()` before every `cb_pop_front` |
| In-flight writes at program end | the next program in the model starts immediately (BB-03) | writer ends with `noc_async_write_barrier()` |
| Realign reads stale L1 | RISC-V loads after NoC writes into L1 | realign runs strictly after `noc_async_read_barrier()` (which invalidates the L1 cache); stores become visible before the push |
| Ragged last block | `core_pages` not a multiple of `block_pages` | push/pop nominal `block_pages`; only the NoC work narrows (ring closes at `cb_depth × block_pages`) |
| Begins baked into CT args | one compile per distinct begin (dozens of programs; BB-07 cost on the first call of each) | begins are RT; only regime flags and shapes are CT; plan cache keyed on the full signature |
| Large CB vs co-resident L1 | `block_pages_cap` fills the BB-05 budget (~1.35 MB at depth 2 for bf16) | the budget is `min(L1_BUDGET_BYTES, get_max_worker_l1_unreserved_size())` minus L1-`out` bytes; a CB/L1-buffer clash at launch is fixed in the budget term, never with a literal |
| `copy_tile_column` element type | bf16 needs 16-bit moves; a 32-bit `T` swaps column pairs | `T` derived from `page_bytes` (2048 → `uint16_t`, 4096 → `uint32_t`) |
| Validation order | a malformed device-form call must raise `ValueError`, not the Phase 0 refusal | structural checks precede tagging and the SUPPORTED gate |
| Negative / out-of-range begins | PyTorch semantics: `-33` on N=64 → 31, end `10**6` → N | normalize on host before tagging, regime selection and planning |
