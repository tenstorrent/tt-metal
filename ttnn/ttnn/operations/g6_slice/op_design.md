# Operation Design: g6_slice

## Overview

| Field | Value |
|-------|-------|
| Classification | data_movement (no compute kernel, no arithmetic: a page gather with sub-tile relocation) |
| Goal | Copy the elements of `x` selected by per-dim begin/end/step into a new tensor of `x`'s dtype, layout and placement, bit for bit, in one device program that reads only the selected pages of `x`. |
| Math | `out[i_0..i_{r-1}] = x[b_0 + s_0*i_0, …, b_{r-1} + s_{r-1}*i_{r-1}]`, `M_d = ceil((e_d - b_d)/s_d)`, `b_d = clamp(starts[d] (+N_d if <0), 0, N_d)`, `e_d = clamp(ends[d] (+N_d if <0), b_d, N_d)`. Device-tensor form: `out = x.narrow(sd, floor(starts[sd]/32)*32, N_sd/num_devices)`. |
| References | `eval/golden_tests/g6_slice/feature_spec.py` (TARGET/INPUTS/INVALID/LOOSE_CASES), `eval/golden_tests/g6_slice/helpers.py` (oracle, `TOLERANCES` = exact for bf16), `.claude/references/blocking-model.md`, `.claude/references/l1-footprint-discipline.md`, `.claude/references/ttnn-cb-memory-fundamentals.md`, catalog `ttnn/ttnn/operations/examples/{double_buffer,trid_double_issue,tile_reorder,noc_placement,bank_stagger,width_split,read_coalescing}`, `tt_metal/hw/inc/api/dataflow/dataflow_api.h`, `tt_metal/hw/inc/api/tensor/tensor_accessor.h`, `ttnn/cpp/ttnn/kernel_lib/{tile_column_dataflow.hpp,local_copy_helpers_dataflow.hpp}`, `tt_metal/hw/inc/internal/tt-1xx/blackhole/noc/noc_parameters.h:375-380` (NoC alignment), `tt_metal/impl/program/program.cpp:2504-2508` (CBs allocated at DRAM alignment) |

### Catalog entries that set knobs

| Entry | Knob it informs | What it says for this op |
|-------|-----------------|--------------------------|
| `double_buffer` | block_pages ≥ 4, buffer_depth 2 | one barrier per block, CB = depth × block; no gain once DRAM-bound (every large Phase 0 case is) |
| `trid_double_issue` | barrier granularity (perf lamp) | at a large CB the one-block loop is within 3 % of the best; trids buy L1, not peak bandwidth |
| `tile_reorder` | R1/R3 store transfer shape | whole-page writes beat sub-tile face writes on a DRAM-bound move |
| `noc_placement` | core assignment | `split_work_to_cores(..., row_wise=True)`; reads NoC0 (reader), writes NoC1 (writer) |
| `width_split` | why the split is over the flat output-page index | a tile-row split strands one-row outputs (1940-page single-row case) on one core |
| `bank_stagger` | perf lamp | lock-step bank collisions when every core's range starts on the same bank |
| `read_coalescing` | refinement (sharded) only | coalescing is invalid across interleaved pages; applies to the width-sharded shard copy |

## Parameters

| Name | Type | Required | Valid Range | Default | CT/RT |
|------|------|----------|-------------|---------|-------|
| `input_tensor` | `ttnn.Tensor` | yes | rank 2–4, on device | — | buffer address RT; dtype/page size CT |
| `starts` | `list[int]` (len r) or `ttnn.Tensor` uint32 ROW_MAJOR DRAM `(r,)` | yes | any int; negative counts from the end; clamped to `[0, N_d]` | — | host list → normalized begins as RT args; tensor → buffer address RT (refinement) |
| `ends` | same form as `starts` | yes | clamped to `[b_d, N_d]`; a tensor is never read | — | host only (→ `M`, passed as RT extents) |
| `steps` | `list[int]` (len r) | no | each ≥ 1; only all-1 supported (`step_mode` "unit_step") | `None` (all 1) | host only |
| `slice_dim` | `int` | device form only | `[-r, r-1]`, canonicalized to `slice_dim + r` before any check | `None` | RT (refinement) |
| `num_devices` | `int` | device form only | divides `N_slice_dim` | `None` | host (`L = N_sd/num_devices` → RT) |

Regime flags (`row_offset != 0`, `col_offset != 0`), `sources_per_output`, `block_pages`, `buffer_depth`, the device-begin switch and the element type are CT (they change the kernel text or the CB layout); every begin value, extent, stride and per-core range is RT. A new begin of the same shape class therefore hits the program cache (`generic_op` hashes CT args, defines and RT-arg *counts*, not RT values: `ttnn/cpp/ttnn/operations/generic/device/generic_op_device_operation.cpp:83-101`).

Entry-point validation (before tagging; raises `ValueError`): `len(starts|ends|steps) != r`; any step < 1; any `M_d == 0`; `slice_dim` without `num_devices` or the reverse, either with list begins, `steps` with tensor begins, `slice_dim ∉ [-r, r-1]`; starts/ends tensors not uint32 of shape `(r,)`; `N_sd % num_devices != 0`. Then the taggers build the axes cell and `validate()` gates SUPPORTED (`UnsupportedAxisValue`) and EXCLUSIONS (`ExcludedCell`). A step > 1 tags `step_mode = "strided"` (not in TARGET) and is refused with `UnsupportedAxisValue`. Every `ValueError` and every support refusal message starts with `g6_slice:` (the acceptance test matches on it through the `expect_error` fixture).

## Tensors

### Input

| Property | Requirement |
|----------|-------------|
| Shape | rank 2–4, any dims (no alignment requirement on any dim) |
| Dtype | bfloat16 (Phase 0); TARGET adds float32, int32, uint32, bfloat8_b, bfloat4_b |
| Layout | TILE (Phase 0); TARGET adds ROW_MAJOR |
| Memory | DRAM interleaved (Phase 0); TARGET adds L1 interleaved, L1 width-sharded |

### Output

| Property | Value |
|----------|-------|
| Shape | `(M_0, …, M_{r-1})` (device form: `M_sd = N_sd/num_devices`, other dims whole) |
| Dtype | `input_tensor.dtype` |
| Layout | `input_tensor.layout` |
| Memory | `input_tensor.memory_config()` (same buffer type; width-sharded: same shard spec, grid, shape, orientation) |

Allocation: `ttnn.allocate_tensor_on_device(ttnn.Shape(M), x.dtype, x.layout, device, x.memory_config())`; `generic_op([x, (starts_tensor,) out], descriptor)` — exactly one dispatch.

### Canonical index space (used by every section below)

The op folds rank to 4 with leading unit dims: dims `(D0, D1, H, W)`; rank 3 → `D0 = 1`, rank 2 → `D0 = D1 = 1`.

| Symbol | Definition |
|--------|-----------|
| `in_h_tiles`, `in_w_tiles` | `ceil(N_H/32)`, `ceil(N_W/32)` — per image (= `padded_shape[-2]/32`, `padded_shape[-1]/32`) |
| `out_h_tiles`, `out_w_tiles` | `ceil(M_H/32)`, `ceil(M_W/32)` — per image |
| `total_out_pages` | `M_D0 · M_D1 · out_h_tiles · out_w_tiles` |
| `hb, row_offset` | `b_H // 32`, `b_H % 32` |
| `wb, col_offset` | `b_W // 32`, `b_W % 32` |
| out page `p` | row-major `(o0, o1, i, j)`, `j` fastest |
| source page `A(p)` | `((b_D0+o0)·N_D1 + b_D1+o1)·in_h_tiles·in_w_tiles + (hb+i)·in_w_tiles + (wb+j)`; below `B = A + in_w_tiles`, right `R = A + 1`, below-right `A + in_w_tiles + 1` |
| `valid_rows(i)`, `valid_cols(j)` | `min(32, M_H − 32 i)`, `min(32, M_W − 32 j)` |
| `needs_below(i)` | `row_offset > 0 ∧ valid_rows(i) > 32 − row_offset` |
| `needs_right(j)` | `col_offset > 0 ∧ valid_cols(j) > 32 − col_offset` |
| `e, FR, FB` | element bytes (bf16: 2); face-row bytes `16·e`; face bytes `16·FR` |
| `row_byte(r, c)` | `(2·[r ≥ 16] + c)·FB + (r mod 16)·FR` — byte offset of row `r`, column half `c`, inside one tile page |

Every logical output element maps to a logical input element (`b_d + i_d < N_d`), so input padding can only land in output padding; output positions beyond `M` may hold anything and are not written where the regime has no reason to write them.

## Blocking Model

The first design section. Everything below it is downstream of these decisions.
Semantics: `.claude/references/blocking-model.md`.

### Axes

Every output element is one input element; nothing spans any axis, so every axis is **independent**. The four tensor axes fold honestly into one flattened output-page axis: the op treats every output page identically (a gather keyed on its own index), output pages are contiguous in `out`, and the input's non-contiguity buys nothing on interleaved storage (consecutive page ids live in different banks; every page is its own NoC transaction). The block is therefore an interval of that axis.

| Axis | Character (+ one-clause reason) | Extent knob | Phase 0 value | Knob source | Core-assignment | Later unlock |
|------|--------------------------------|-------------|---------------|-------------|-----------------|--------------|
| `D0` (dim r−4; 1 below rank 4) | independent — each output element copies one input element | folded into `block_pages` (the block interval covers it) | whole extent inside the interval | `plan_blocks()` | via `out_page` ranges | knob-turn |
| `D1` (dim r−3; 1 below rank 3) | independent — same | folded into `block_pages` | whole extent inside the interval | `plan_blocks()` | via `out_page` ranges | knob-turn |
| `H` (tile rows, dim r−2) | independent; under `row_offset > 0` an output tile's footprint is a 2-tile stencil (`A`, `B`) that overlaps its neighbour's — a halo, not a combine | folded into `block_pages`; halo factor `1 + needs_below_any` | whole extent inside the interval; halo factor 1 for every production case | `plan_blocks()` | via `out_page` ranges | knob-turn |
| `W` (tile cols, dim r−1) | independent; under `col_offset > 0` the stencil gains the right tile (`R`) — a halo | folded into `block_pages`; halo factor `1 + needs_right_any` | whole extent inside the interval | `plan_blocks()` | via `out_page` ranges | knob-turn |
| `out_page` (fold of the four, row-major) | independent | `block_pages` | `ceil(pages_per_core / ceil(pages_per_core / block_pages_cap))`; p1 at 110 cores: 2533 pages → 8 blocks of 317 | `plan_blocks()` host function → one CT arg; CB capacity, kernel trip counts derived | `split_work_to_cores(grid, total_out_pages, row_wise=True)` over the full `device.compute_with_storage_grid_size()` grid: contiguous ranges, ±1 page balance | knob-turn |
| `stencil` (source slots per output page) | not a data axis — footprint factor of the halos above | `sources_per_output = (1 + needs_right_any)(1 + needs_below_any)` | 1 for every production case | `plan_blocks()` → CT | per page | — |

`needs_below_any = row_offset > 0 ∧ M_H > 32 − row_offset`; `needs_right_any = col_offset > 0 ∧ M_W > 32 − col_offset`.

Knob definitions (single source: `plan_blocks()` on the host; kernels receive `block_pages`, `sources_per_output`, `buffer_depth` as CT args and derive `num_blocks_this_core = ceil(num_pages_this_core / block_pages)`, `last_block_extent` themselves):

| Knob | Expression |
|------|-----------|
| `pages_per_core` | `units_per_core_group_1` of `split_work_to_cores` (= `ceil(total_out_pages / num_cores)`) |
| `block_pages_cap` | `floor((L1_OP_BUDGET − fixed_bytes) / (MAX_BUFFER_DEPTH·sources_per_output·in_page_bytes + assembled_per_output·out_page_bytes))`, `L1_OP_BUDGET = 1347776`, `MAX_BUFFER_DEPTH = 2`, `assembled_per_output = (1 + needs_below_any)` in element_shift else 0, `fixed_bytes` = 0 in Phase 0 |
| `num_blocks_max` | `ceil(pages_per_core / block_pages_cap)` |
| `block_pages` | `ceil(pages_per_core / num_blocks_max)` — the coarsest block that fits, equalized so the last block is not a sliver (same block count as the coarsest choice) |
| `buffer_depth` | `1` if `num_blocks_max == 1` (nothing to prefetch) else `MAX_BUFFER_DEPTH` |

Phase 0 caps (bf16, 2048-B pages): page_copy / row_shift (S = 1): **329**; row_shift (S = 2): 164; element_shift (S = 2, no below): 131; element_shift (S = 4): 65.

### Buffer-depth knobs

| CB | Depth knob | Phase 0 value | What the depth buys |
|----|------------|---------------|---------------------|
| `cb_staged_tiles` | `buffer_depth` | 1 when a core's whole assignment is one block, else 2 | reader fills block k+1 while the writer drains block k (read and write streams on different NoCs overlap) |
| `cb_assembled_tiles` (element_shift only) | fixed 1 block | 1 | none needed: the writer composes and drains it itself; one flush per block frees it |
| `cb_begin_index` (device form, refinement) | fixed 1 | 1 | read once per launch |

### Mechanism caps

| Mechanism | Cap on which extent | Clamp | What happens unclamped |
|-----------|--------------------|-------|------------------------|
| BB-05 L1 budget (1347776 B incl. an L1 `out`'s share) | `block_pages` | `block_pages ≤ block_pages_cap` (expression above) | launch fails only inside the model, or CBs overlap the model's L1 tensors |
| CB ring wrap (`ttnn-cb-memory-fundamentals.md` "CB Ring-Wrap Invariant") | push/pop quantum | every push and pop is the nominal `sources_per_output·block_pages` (staged) / `assembled_per_output·block_pages` (assembled), the ragged last block included; only the NoC transfers narrow | pointer walks past `fifo_limit` on the third block of a 2-deep ring → silent corruption of neighbouring L1 |
| NoC address congruence: DRAM read L1≡src mod 64 (BH) / 32 (WH); DRAM write and L1↔L1 mod 16 (`noc_parameters.h:375-380`, sanitizer `sanitize.h:746`) | sub-page transfer offsets | sub-page reads land at the source's own in-page offset inside a page-sized slot (CB bases are DRAM-aligned, `program.cpp:2504-2508`; tile pages are multiples of 64 B), so every read is congruent; the row shift is applied on the **write** side, whose congruence is 16 B and face rows are `FR = 32 B` (bf16) / 64 B (4-byte types) | an odd `row_offset` read straight into its shifted position is 32 B off mod 64 on Blackhole → dropped/garbled bytes, watcher error under `--dev` |
| Face-row granularity of the NoC row shift | dtypes for row_shift's NoC segments | `FR % 16 == 0` (bf16, fp32, int32, uint32); block floats (1-byte exponent per face row, 8-B bfp4 mantissa rows) route row shifts through the element_shift composer (refinement) | exponent bytes moved at 16-B granularity carry neighbours' exponents → wrong values |
| Shared exponent of block floats | `col_offset` for bfloat8_b / bfloat4_b | `col_offset % 16 == 0` (face-granular shift keeps each 16-element exponent group whole); other offsets refused (refinement) | a split group would need requantization — not a copy |
| uint32 page ids / RT args | pages of `x` | largest INPUTS input is 557056 pages ≪ 2³² | — |
| Device-form begin (refinement) | begin on dim `sd` | tile dims: `t0 = min(idx/32, in_tiles_sd − out_tiles_sd)`; leading dims: `min(floor(idx/32)·32, N_sd − L)` | an out-of-contract begin reads outside `x` |

### Regimes

| Regime | Status | Predicate | Block | Data movement vs. minimum | What a bigger block buys |
|--------|--------|-----------|-------|---------------------------|--------------------------|
| **page_copy** (R1) | built | TILE ∧ interleaved ∧ (`begins_source == device_tensor` ∨ (`row_offset == 0` ∧ `col_offset == 0`)) | `block_pages` out pages, 1 staged page each | **minimum** against DRAM: each selected `x` page crosses once, each `out` page once, 0 cross-core. Domain: 791 of 868 Phase 0 cases incl. all 8 flagged perf cases (p1 703: 278528 pages, 570.4 MB read + 570.4 MB written) | one read barrier per block (each exposes one DRAM round trip with an empty read pipe), one reserve/push/wait/pop per block per kernel, one write flush per block; p1 pays 8 of each per core instead of 2533 |
| **row_shift** (R2) | built | TILE ∧ interleaved ∧ host_list ∧ `row_offset != 0` ∧ `col_offset == 0` ∧ `FR % 16 == 0` | `block_pages` out pages × `1 + needs_below_any` staged slots | `x`: one covering sub-page read per needed source page (`[row_byte(r0,0), row_byte(r1,1)+FR)`); production (all 77 cases have `M_H = 1`): 544 B/page vs. 64 B of selected data (+480 B: one transaction covers both column halves instead of two). `out`: only valid rows written (`valid_rows·2·FR` per page) = minimum. Multi-row selections re-read interior pages as `B` of tile i and `A` of tile i+1 (≤ 2×, rule case only) | as page_copy, plus one flush per block for the ≤ 8 segment writes per page |
| **element_shift** (R3) | built | TILE ∧ interleaved ∧ host_list ∧ `col_offset != 0` (any `row_offset`) | `block_pages` out pages × `sources_per_output` staged slots + `(1 + needs_below_any)` assembled pages each | `x`: `sources_per_output` whole pages per out page (≤ 4× minimum: a page shared by up to 4 output pages is re-staged per output page); `out`: whole pages once (rows_offset 0) or valid-row segments (row_offset > 0). No production TILE case (0 of 1310); built because SUPPORTED `w_non_aligned` admits any `b_W` | barrier/handshake/flush per block as page_copy; the column copy is per element and is not amortized |
| row_shift_window (R2w) | deferred: no production row_shift case ever reads a `B` page (all 77 select one row), so the ≤ 2× re-read it removes is paid by the rule case only | row_shift ∧ `needs_below_any` | an `H`-run of out pages of one `(o0,o1,j)` column, `B` of page i kept as `A` of page i+1 | removes the interior re-read → minimum | — |
| element_shift_reuse (R3r) | deferred: no TILE production case has `col_offset != 0` | element_shift | a `W`-run inside one tile row; right page of `j` kept as left page of `j+1` | `x` read ≈ once (+1 page per run) | — |
| device_begin (R1-dev) | deferred: `begins_source` "device_tensor" is outside the Phase 0 SUPPORTED rectangle (2 production cases); reachable because page_copy takes the dim-`sd` begin only as a base offset of `A(p)`, which a kernel-start load supplies in place of an RT arg | `begins_source == device_tensor` (always page_copy: the begin is a whole tile) | as page_copy | as page_copy + one `align_up(4r, DRAM_ALIGNMENT)` read of `starts` per core | as page_copy |
| begin_broadcast | rejected: superseded by R1-dev's per-core read — the payload is `4r ≤ 16` B (one 64-B transaction per core); a multicast puts a semaphore rendezvous on every core's critical path to save that one read | device form | — | — | — |
| shard_copy (RS) | deferred: `l1_width_sharded` is outside Phase 0; knob-turn (it cuts `W`, an independent axis): core-assignment and block are pinned by `out`'s shard spec (block = the whole output shard, 8 pages in production), `out`'s CB is backed on `out`'s shard (zero-copy), each out shard reads from `x`'s shard(s) — a self-aimed local read when `b_W % shard_w == 0` and the shard is on the same core (production: `b_W = 0`), a remote NoC read otherwise | `memory_config == l1_width_sharded` | whole output shard | each out shard byte crosses L1→L1 once, no DRAM | one transfer per shard (shard rows are contiguous: `read_coalescing`) |
| l1_interleaved placement | deferred: outside Phase 0; knob-turn — R1/R2/R3 unchanged, `TensorAccessor` resolves L1 banks, `fixed_bytes += ceil(total_out_pages / num_l1_banks)·out_page_bytes` | `memory_config == l1_interleaved` | as R1–R3 | as R1–R3 with L1 instead of DRAM | as R1–R3 |
| stick_gather (RM) | deferred: ROW_MAJOR is outside Phase 0; reachable on the same flat page split — an RM page is a stick: out stick ← bytes `[b_W·e, (b_W+M_W)·e)` of input stick, read from the 64-B-aligned floor of that range into a slot at the congruent offset, written from `slot + (b_W·e mod 64)`; a RISC-V realign of < 16 B only when `b_W·e % 16 != 0` (no production case) | `layout == ROW_MAJOR` | `block_sticks` out sticks | each selected stick segment once (≤ 63 B extra per stick for alignment) | as page_copy |
| block-float row shift | deferred: no production block-float case has `b_H % 32 != 0` or `b_W % 32 != 0` (all 6 are page_copy) | bfloat8_b/bfloat4_b ∧ (`row_offset != 0` ∨ `col_offset != 0`) | as element_shift | as element_shift (exponent bytes moved with their face rows) | as element_shift |
| whole_row_read | rejected: forbidden ("MUST NOT read x whole or whole tile rows of it") and 2× the DRAM bytes of p1/p2; superseded by page_copy | — | — | 2× | — |
| program_chain (untilize → slice → tilize) | rejected: BB-01 and the contract (one program) | — | — | ≥ 3× | — |
| compute_copy (unpack→DEST→pack) | rejected: `copy` (`kernel_lib/eltwise/api/convenience.hpp:109-113`) is tile→tile at the same element positions — it cannot shift rows or columns — and routes values through a format path (int32/uint32 > 2²⁴, NaN payloads, denormals) | — | — | +L1 round trip | — |
| direct_shifted_read | rejected: superseded by row_shift — reading face rows straight into their shifted output position breaks the 64-B read congruence on Blackhole for odd `row_offset` | — | — | — | — |
| strided | rejected: `step_mode` "strided" is not in TARGET (no production step ≠ 1); refused with `UnsupportedAxisValue` | any step > 1 | — | — | — |

**Regime-selection function** (host, exact; exported as `g6_slice_plan(...)["regime"]`):

```
if begins_source == "device_tensor":        regime = "page_copy"
elif b_W % 32 != 0:                         regime = "element_shift"
elif b_H % 32 != 0:                         regime = "row_shift"
else:                                       regime = "page_copy"
```
(`b` normalized: negatives + `N_d`, clamped.) Regime-pinned tests: page_copy (production 1478, 182, 1484, 621, 588, 701), row_shift (587, 1183, rule `two_tiles`), element_shift (two rule-shape cases). The predicate depends only on the call's begins, never on the grid, so no regime is grid-dependent.

### Traffic ranking

Counted per tensor and tier for the flagged p1 (703, page_copy) and in general.

| Rank | Candidate split | `x` DRAM crossings | `out` DRAM crossings | Cross-core | Verdict |
|------|-----------------|--------------------|----------------------|------------|---------|
| 1 | **S1**: contiguous ranges of the flat output-page index over `min(total_out_pages, G)` cores | each selected page once (p1: 570.4 MB) | once (570.4 MB) | 0 | **chosen** |
| 1= | S2: split by output tile rows only | same | same | 0 | tie on bytes; strands one-row outputs (1183: 1940 pages in one tile row → 1 core; `width_split`) |
| 1= | S3: split by output tile columns only | same | same | 0 | tie on bytes; strands one-column outputs (730/731: `out_w_tiles = 1`) |
| 1= | S4: contiguous ranges of input pages, reading only the selected ones | same | same | 0 | tie on bytes; unselected ranges leave cores idle (p1: half the cores) |
| 5 | S5: contiguous input ranges read whole, unselected pages dropped in L1 | `x` whole (p1: 2× = 1140.9 MB) | once | 0 | rejected (forbidden; +570 MB) |
| — | split a dependent axis + cross-core combine | — | — | — | none exists: no output element depends on more than one input element |
| — | halo forwarding of a shared source page to the neighbouring core instead of a DRAM re-read (row_shift / element_shift) | −1 page per core boundary | — | +1 page L1→L1 per boundary | not built: in-core reuse (R2w/R3r) removes the bulk first, and no production case reads a shared page at all |

S1 moves the minimum; the tiebreak against S2–S4 is occupancy and balance (±1 page). **Operand-reuse check**: `x` varies along `out_page` (every core reads different pages) — not reuse-shared. The device-form `starts` tensor does not vary along any split → reuse-shared by construction → its broadcast is the `begin_broadcast` row (rejected, 16-B payload).

**Stall-shadow check**:

| Stage that waits | Waits on | Work scheduled into the window |
|------------------|----------|--------------------------------|
| writer `wait` on `cb_staged_tiles` | reader's block reads | nothing of its own; at depth 2 the reader's next block is already in flight |
| reader read barrier per block | DRAM round trip of its block | the next block's issue could run here — trid perf lamp (no algebraic permission needed: pure copy) |
| reader `reserve` on `cb_staged_tiles` | writer draining block k−1 | none needed: depth 2 |
| reader prologue in the device form (refinement) | the `starts` read | nothing: every source page depends on the begin when `sd` is a tile dim; the writer's setup is independent and already concurrent |

No reorder changes floating-point order: there is no arithmetic.

### Block schedule

```cpp
// reader: optional prologue (device form, refinement): load_begin();  -> begin offset on dim sd
for (uint32_t block_idx = 0; block_idx < num_blocks_this_core; ++block_idx) {
    stage_sources_block(block_idx);    // reader: x -> cb_staged_tiles
    assemble_columns_block(block_idx); // writer, element_shift only: cb_staged_tiles -> cb_assembled_tiles
    store_block(block_idx);            // writer: cb_staged_tiles | cb_assembled_tiles -> out
}
// writer epilogue: noc_async_write_barrier() (BB-03)
```

| Operation | Block shape | Acts | Resident across it | Intended frequency of fixed costs |
|-----------|-------------|------|--------------------|-----------------------------------|
| `load_begin` (refinement) | 1 page of `starts` | read `starts[sd]`, round down to a tile, clamp | begin offset, for the whole launch | once per launch |
| `stage_sources_block` | `block_pages` out pages × `sources_per_output` slots | page_copy: one whole-page read per out page; row_shift: one covering sub-page read per needed source page (`A`, and `B` iff `needs_below(i)`), placed at the source's own in-page offset; element_shift: one whole-page read per needed stencil page; unneeded slots left unread | nothing (slots are released by the writer) | 1 reserve + 1 read barrier + 1 push per block |
| `assemble_columns_block` (element_shift) | `block_pages` out pages × `(1 + needs_below_any)` assembled pages | for each assembled page (top = from `A`/`R`, bottom = from `B`/below-right): for every valid output column `x < valid_cols(j)`, copy source column `(x + col_offset) mod 32` of the left (`x + col_offset < 32`) or right page into column `x` (all 32 rows) | the staged block | 1 wait on staged + 1 reserve/push/wait on assembled per block |
| `store_block` | `block_pages` out pages | row_offset == 0: one whole-page write per out page (staged page in page_copy, top assembled page in element_shift); row_offset > 0: segment writes (below) | — | 1 write flush + 1 pop per block (+1 pop of assembled in element_shift) |

**Row-shift segment plan** (row_shift and element_shift with `row_offset > 0`): output row `y < valid_rows(i)` comes from source row `s = y + row_offset` of the top source (`s < 32`) or row `s − 32` of the bottom source. Runs of consecutive `y` break at `y = 16` (destination face) and where `s` reaches 16, 32 or 48 (source face/page). For each run `[y0, y1)` and column half `c ∈ {0, 1}`: one `noc_async_write_page(out_page, out_accessor, slot + row_byte(s0 mod 32, c), (y1−y0)·FR, row_byte(y0, c))`. At most 4 runs → at most 8 writes per page; for the production case (`valid_rows = 1`) exactly 2 writes of `FR` bytes. Covering read range for source rows `[r0, r1]` of one page: `[row_byte(r0, 0), row_byte(r1, 1) + FR)`.

### Perf lamps

| Lamp | Why the default may be wrong here | Nearby alternative to measure |
|------|-----------------------------------|-------------------------------|
| Overlap | the coarsest block (up to 329 pages) makes every core with ≤ 329 pages a single block at depth 1 — read all, then write all (851 of 868 Phase 0 cases at 110 cores); at ≤ 8 blocks per core the first fill and last drain are each ≥ 1/8 of the work. Harmless while DRAM-bound (all large cases), possibly not for mid-size cases | `block_pages ∈ {8, 16, 32}` × `buffer_depth ∈ {2, 3, 4}` on p1 (703), 733 and 588 (6144 pages, single block per core) |
| Grid synchronization | at 110 cores, 601 of 868 Phase 0 cases have ≤ 8 pages per core (722 have ≤ 16); per-core launch and RT-arg cost may exceed the work | `num_cores ∈ {min(total, G), ceil(total/8), ceil(total/32)}` on 182, 621, 587 |
| Barrier granularity (stall shadow) | one read barrier per block drains the read pipe | trid double issue (`local_copy_helpers_dataflow.hpp:147,158`), T = 2–3 blocks in flight, at `block_pages` 16–32, iso-L1 vs. the default |
| Bank stagger | contiguous per-core ranges whose length is a multiple of the 8 DRAM banks start every core on the same bank in lock step | rotate each core's walk to start at page `core_idx % num_dram_banks` of its range and wrap (`bank_stagger`), p1/p2/p4 |
| row_shift read granularity | one covering read (544 B) per page vs. two 32-B face-row reads | two reads per page on 1183 (1940 pages) |
| row_shift store for full tiles | ≤ 8 segment writes per page when `valid_rows = 32` (rule case only); `tile_reorder` favours whole pages | compose in L1 with self-aimed reads, one whole-page write |
| element_shift with `col_offset % 16 == 0` | per-column RISC-V copies when whole faces could move | 4 face-granular self-aimed NoC reads per page (`local_copy_helpers_dataflow.hpp:57`) |

## Dataflow Strategy

| Stage | Format | Mechanism | Notes |
|-------|--------|-----------|-------|
| `x` DRAM → `cb_staged_tiles` (page_copy, element_shift) | whole tile pages, `x`'s storage format | reader (NCRISC, NoC0): `noc_async_read_page(page, x_accessor, slot)` per needed page, one barrier per block | no conversion anywhere |
| `x` DRAM → `cb_staged_tiles` (row_shift) | covering byte range of one page at its own in-page offset | reader: `noc_async_read(x_accessor.get_noc_addr(page, lo), slot + lo, hi − lo)` | congruent by construction |
| `cb_staged_tiles` → `cb_assembled_tiles` (element_shift) | element columns | writer (BRISC): `copy_tile_column<cb_staged_tiles, T>` RISC-V loads/stores | T = `uint16_t` (bf16) |
| staged/assembled → `out` DRAM | whole page (row_offset 0) or face-row runs | writer (NoC1): `noc_async_write_page(page, out_accessor, src, size, offset)`; one `noc_async_writes_flushed` per block; `noc_async_write_barrier` at exit | output padding is not written by row-shift stores |
| `starts` DRAM → `cb_begin_index` (refinement) | `4r` B uint32 stick | reader prologue, every core | never read on host, never in RT/CT args |
| width-sharded (refinement) | shard pages | `out` CB backed on `out`'s shard (`ttnn.cb_descriptor_from_sharded_tensor`); local self-aimed read from `x`'s shard, remote NoC read only when the source shard lives on another core | no DRAM; no `TensorAccessor` re-read of a core's own shard |

TARGET placement values as regimes: `dram_interleaved` → page_copy/row_shift/element_shift (built); `l1_interleaved` → same kernels, L1-bank accessor, budget minus `out`'s share (knob-turn); `l1_width_sharded` → shard_copy (knob-turn: cuts `W`, independent). Layout `ROW_MAJOR` → stick_gather. `begins_source` `device_tensor` → device_begin. Dtypes float32/int32/uint32 → page size 4096, `FR = 64`, `T = uint32_t` (knob-turn); bfloat8_b/bfloat4_b → page_copy as is (pages 1088/576 B, multiples of 64), shifts via the block-float row-shift row.

## Work Distribution

| Field | Value |
|-------|-------|
| Work unit | one output page; the block is `block_pages` consecutive output pages |
| Grid | `device.compute_with_storage_grid_size()` (RT parameter; never a constant) |
| Per-core work | `split_work_to_cores(grid, total_out_pages, row_wise=True)` → group 1: `pages_per_core` pages, group 2: `pages_per_core − 1`; core `k` (row-major order) owns `[start_page_k, start_page_k + num_pages_k)`; one kernel per role over `all_cores` (≤ 2 rectangles: full rows + one partial row, BB-17) |
| Remainder | `total_out_pages` uses per-image `ceil` tile counts; a core's last block has `last_block_extent = num_pages_k − (num_blocks_this_core − 1)·block_pages` real pages but pushes/pops the nominal count |
| Plan report (BB-08) | `g6_slice_plan(input_tensor, starts, ends, steps=None, *, slice_dim=None, num_devices=None) -> dict` with keys `regime`, `row_offset`, `col_offset`, `sources_per_output`, `num_cores`, `pages_per_core`, `block_pages`, `buffer_depth` — same function the entry point uses to build the descriptor |

The reader walks its pages with an `(o0, o1, i, j)` counter (no per-page division) and derives `A(p)` incrementally; `needs_below(i)` / `needs_right(j)` and `valid_rows(i)` / `valid_cols(j)` change only at tile-row / tile-column boundaries.

RT args per core (both kernels): `start_page`, `num_pages`; geometry (shared values): `M_D1, out_h_tiles, out_w_tiles, N_D1, in_h_tiles, in_w_tiles, b_D0, b_D1, hb, wb, row_offset, col_offset, M_H, M_W`; buffer addresses. CT: `block_pages`, `sources_per_output`, `buffer_depth`, page sizes, `TensorAccessorArgs` (scalar CT args first, accessor args last), defines `G6_ROW_SHIFT`, `G6_COL_SHIFT` (and `G6_DEVICE_BEGIN` in refinement) so each program carries only its regime's code (BB-18, RB-038).

## Circular Buffers

| Semantic Name | Index | Page Size | Num Pages | Sizing rationale | Format | Producer | Consumer | Lifetime |
|---------------|-------|-----------|-----------|------------------|--------|----------|----------|-----------|
| `cb_staged_tiles` | 0 | `x.buffer_page_size()` (2048 bf16) | `buffer_depth · sources_per_output · block_pages` | spans `out_page` (→ `block_pages`) and the stencil (→ `sources_per_output`); streams over blocks | `x.dtype` | reader | writer | whole launch |
| `cb_assembled_tiles` | 1 | `out.buffer_page_size()` | `(1 + needs_below_any) · block_pages` | element_shift only; spans `out_page` and the vertical stencil | `x.dtype` | writer | writer | element_shift launches |
| `cb_begin_index` | 2 | `round_up(4r, get_dram_alignment())` | 1 | device form only (refinement); constant | uint32 | reader | reader | reader prologue |

Push count == wait count: reader pushes `sources_per_output·block_pages` per block, writer waits/pops the same; writer reserves/pushes and waits/pops `(1 + needs_below_any)·block_pages` on `cb_assembled_tiles` per block. Full ledger: `l1_ledger.md`.

## Block Operation Realization

| # | Block operation | Block shape | Helper? | Input CB (semantic name, pages, state) | Output CB (semantic name, pages) | CB state after |
|---|-----------------|-------------|---------|----------------------------------------|----------------------------------|----------------|
| 0 | `load_begin` (refinement) | 1 page | raw dataflow | — (DRAM `starts`) | `cb_begin_index`, 1 | value held in a register for the launch; CB not pushed |
| 1 | `stage_sources_block` | `block_pages × sources_per_output` | raw dataflow | — (DRAM `x`) | `cb_staged_tiles`, `sources_per_output·block_pages` pushed | unread slots hold stale bytes and are never consumed |
| 2 | `assemble_columns_block` (element_shift) | `block_pages × (1 + needs_below_any)` | `copy_tile_column` | `cb_staged_tiles`, `sources_per_output·block_pages`, waited (not popped) | `cb_assembled_tiles`, `(1+needs_below_any)·block_pages` | staged block still held until `store_block` pops it |
| 3 | `store_block` | `block_pages` | raw dataflow | `cb_staged_tiles` (page_copy, row_shift) or `cb_assembled_tiles` (element_shift) | — (DRAM `out`) | both CBs popped after one flush |

## API Mapping

| Block operation | Type | Function | File:Line | Template Params / Args | Input CB | Output CB | Which params are block knobs |
|-----------------|------|----------|-----------|------------------------|----------|-----------|------------------------------|
| host split | host | `ttnn.split_work_to_cores` | `ttnn/cpp/ttnn-nanobind/operations/core.cpp:470-476` | `(grid, total_out_pages, row_wise=True)` | — | — | `num_cores` (core-assignment) |
| host accessor | host | `ttnn.TensorAccessorArgs(t).get_compile_time_args()` | `.claude/references/generic_op_template/template_op_program_descriptor.py` (step 4) | `x`, `out` | — | — | — |
| `stage_sources_block` | raw_api | `cb_reserve_back` / `get_write_ptr` / `cb_push_back` | `dataflow_api.h:406`, `:325`, `:208` | `cb_staged_tiles`, `sources_per_output·block_pages` | — | `cb_staged_tiles` | `block_pages`, `sources_per_output` |
| `stage_sources_block` (whole page) | raw_api | `noc_async_read_page(id, accessor, dst)` | `dataflow_api.h:1243` | page id `A(p)` (+`R`, `B`, below-right) | — | `cb_staged_tiles` | — |
| `stage_sources_block` (row_shift sub-page) | raw_api | `TensorAccessor::get_noc_addr(page, offset)` + `noc_async_read(src, dst, size)` | `tensor_accessor.h:165`, `dataflow_api.h:547` | `offset = lo`, `dst = slot + lo`, `size = hi − lo` | — | `cb_staged_tiles` | — |
| `stage_sources_block` | raw_api | `noc_async_read_barrier` | `dataflow_api.h:1921` | once per block | — | — | `block_pages` sets its frequency |
| `assemble_columns_block` | helper | `dataflow_kernel_lib::copy_tile_column<dfb_id, T>(src_addr, src_col, dst_addr, dst_col)` | `ttnn/cpp/ttnn/kernel_lib/tile_column_dataflow.hpp:91-102` | `dfb_id = cb_staged_tiles` (32×32 tile shape), `T = uint16_t` (bf16) | `cb_staged_tiles` | `cb_assembled_tiles` | — |
| `store_block` | raw_api | `cb_wait_front` / `get_read_ptr` / `cb_pop_front` | `dataflow_api.h:479`, `:346`, `:260` | nominal counts | `cb_staged_tiles` / `cb_assembled_tiles` | — | `block_pages` |
| `store_block` | raw_api | `noc_async_write_page(id, accessor, src, size, offset)` | `dataflow_api.h:1418` | whole page: `size = 0` (page size); segments: `size = run·FR`, `offset = row_byte(y0, c)` | staged/assembled | — | — |
| `store_block` | raw_api | `noc_async_writes_flushed`; `noc_async_write_barrier` (exit) | `dataflow_api.h:1980`, `:1951` | once per block; once per launch | — | — | `block_pages` sets flush frequency |
| `load_begin` (refinement) | raw_api | `noc_async_read_page` + `noc_async_read_barrier` | `dataflow_api.h:1243`, `:1921` | `starts` accessor, page 0 | — | `cb_begin_index` | — |

**Helpers considered and rejected** (every `raw_api` row above):

| Candidate | File:Line | Why it cannot express the shape |
|-----------|-----------|--------------------------------|
| `compute_kernel_lib::copy` (CopyTile→PackTile) | `kernel_lib/eltwise/api/convenience.hpp:109-113` | tile→tile at identical element positions: no row or column relocation; adds an unpack→DEST→pack format path to a pure DRAM copy (int32/uint32 > 2²⁴, NaN payloads); the op has no compute phase |
| `tilize` / `untilize` helpers | `kernel_lib/tilize_helpers.hpp`, `kernel_lib/untilize_helpers.hpp:152-163` | layout conversion is forbidden by the contract and not needed: the shift is a byte relocation |
| `local_copy_helpers_dataflow` (`local_addr`, `set_read_state`, `read_with_state`) | `kernel_lib/local_copy_helpers_dataflow.hpp:57,74-95` | self-aimed L1→L1 reads only (`local_addr` builds a source on this core); the DRAM gather and the DRAM-side segment writes are not self-aimed. Used by the shard_copy refinement and the face-granular lamp |
| `copy_tile_column` for row_shift | `kernel_lib/tile_column_dataflow.hpp:91-102` | copies row `r` to row `r` (`dst[offset] = src[offset]`, one `offset` for both): it has no source-row offset, so it cannot express a row shift. The row shift is done on the NoC write side instead (no RISC-V copy); closing the gap would need a `src_row_offset` parameter and two source tiles |
| `mcast` (`McastArgs`) | `kernel_lib/mcast/` | no operand is shared across cores in Phase 0; the only reuse-shared operand (`starts`, refinement) is 16 B — see `begin_broadcast` (rejected) |

No compute kernel, so `compute_kernel_hw_startup()` does not apply; there is no reduce scaler.

## Structural impossibilities

- bfloat8_b / bfloat4_b with `b_W % 16 != 0`: the selected columns would split a 16-element shared-exponent group; a bit-exact copy is impossible without requantization (computation). No production case has it. Candidate for INVALID or an EXCLUSIONS entry on a dedicated tagger value when block floats enter SUPPORTED — fold in via another `/golden-tests` run; `alignment` itself must stay as is (`h_non_aligned` may never be split off).

## Key Risks and Gotchas

| Risk | Why it bites here | Mitigation in this design |
|------|-------------------|---------------------------|
| Odd `row_offset` on Blackhole | a face row read into its shifted slot is 32 B off mod 64 (DRAM read congruence) | reads always land at the source's own in-page offset; the shift happens on writes (16-B congruence, `FR` = 32 B) |
| Ring wrap with ragged last block | 701 runs 4 blocks on a 2-deep ring; a short final push leaves the pointer off `fifo_limit` | nominal pushes/pops always; only NoC transfers narrow to `last_block_extent` |
| Output padding of row_shift / element_shift pages | only valid rows/columns are written; padding holds stale DRAM/L1 bytes | allowed by contract; do not add zero fills (extra writes, no reader). Graded on the logical shape |
| Input padding reaching a logical position | an unaligned `N_H`/`N_W` image has padded last tiles | every logical output position maps to a logical input position (`b_d + i_d < N_d`); padding only lands in output padding |
| Per-image tile padding | `floor(D0·D1·H/32)` miscounts pages when `H % 32 != 0` (3844, 31, 33, 16 in INPUTS) | page ids use per-image `in_h_tiles = ceil(N_H/32)` and the folded `(D0, D1)` strides |
| `needs_below` / `needs_right` source existence | a `B`/`R` page id past `x` | read only when the page holds a selected element; then it exists |
| Begin values in CT args | a new begin would rebuild the program (cache miss) and under the device form break trace replay | begins and extents are RT; only the regime flags and sizes are CT; acceptance test asserts no new cache entry for a new begin |
| Host rebuild on repeat calls (BB-07) | `generic_op` needs a descriptor per call (new `out` address) | memoize the descriptor per call signature (shape, dtype, layout, memory config, normalized begins/ends, form) and patch only buffer addresses |
| Kernel size (BB-04/BB-18) | three regimes and the device read in one source | `G6_ROW_SHIFT`, `G6_COL_SHIFT`, `G6_DEVICE_BEGIN` compile the unused paths out |
| BB-03 | `noc_async_writes_flushed` does not wait for acks | writer ends with `noc_async_write_barrier`; reader ends after its last barrier |
| Profiling zones (BB-21) | time outside a zone reads as waiting | one `DeviceZoneScopedN` per block operation and per blocking wait (`wait_staged`, `reserve_staged`, `read_barrier`, `write_flush`); a role dump of `(core → reader/writer, pages)` from `g6_slice_plan` |
| `copy_tile_column` template id | it reads the tile shape of `dfb_id` | pass `cb_staged_tiles` (32×32 tiles) |
| Validation order | taggers need well-formed lists | ValueError checks run first inside `validate()`, then taggers, SUPPORTED, EXCLUSIONS; `h_non_aligned` stays in SUPPORTED and in no EXCLUSIONS entry |
| Width-sharded `out` (refinement) | `out` (8704 wide) uses 34 of 36 shards of `x`'s spec | allocate `out` with `x.memory_config()` unchanged; kernels run on the 34 cores that hold output shards |
| L1 `out` (refinement) | `out`'s share counts against 1347776 B | `fixed_bytes` term in `block_pages_cap` |
