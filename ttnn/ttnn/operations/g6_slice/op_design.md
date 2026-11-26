# Operation Design: g6_slice

## Overview

| Field | Value |
|-------|-------|
| Classification | data_movement (no compute kernel, no arithmetic) |
| Goal | Copy the elements of `input_tensor` selected by per-dim begin/end/step into a new tensor of the same dtype, layout and memory placement, bit for bit, in **one** device program that reads from x only the pages that hold selected elements. |
| Math | `out[i_0..i_{r-1}] = x[b_0 + s_0·i_0, …, b_{r-1} + s_{r-1}·i_{r-1}]`, `M_d = ceil((e_d − b_d)/s_d)`, `b_d = clamp(starts[d] (+N_d if <0), 0, N_d)`, `e_d = clamp(ends[d] (+N_d if <0), b_d, N_d)`. Device-tensor form: `r0 = floor(starts[sd]/32)·32`, `out = x[..., r0 : r0 + N_sd/num_devices, ...]`. |
| Tile form (Phase 0) | Aligned (`b_{r-2} % 32 == 0` and `b_{r-1} % 32 == 0`): output tile `(l, R, C)` **is** input tile `(l + b_lead, b_{r-2}/32 + R, b_{r-1}/32 + C)` (math.md F1). Shifted: output tile `(R, C)` takes its rows/columns from at most 2×2 input tiles `q_a..q_b × c_a..c_b` (math.md F4; at most 2 when only the row begin is inside a tile). |
| References | `docs/explorations/g6_slice/{DOSSIER,math,tensix,risks,lessons}.md`; `eval/golden_tests/g6_slice/{feature_spec,helpers,axes}.py`; `.claude/references/blocking-model.md`; `.claude/references/l1-footprint-discipline.md`; catalog `ttnn/ttnn/operations/examples/master.md` entries `double_buffer`, `trid_double_issue`, `tile_reorder`, `noc_placement`, `width_split`, `bank_stagger`, `split_reader`; generic-op cache key `ttnn/cpp/ttnn/operations/generic/device/generic_op_device_operation.cpp:78-147` |

## Parameters

| Name | Type | Required | Valid Range | Default | CT/RT |
|------|------|----------|-------------|---------|-------|
| `input_tensor` | `ttnn.Tensor` | yes | rank 2..4, on device | — | buffer address: RT |
| `starts` | `list[int]` len r, or `ttnn.Tensor` uint32 `(r,)` ROW_MAJOR DRAM | yes | any int (PyTorch clamping); tensor values not checked on host | — | normalized begins: **RT** (common RT args); device form: tensor address RT, value read on device |
| `ends` | same form as `starts` | yes | any int (clamped); tensor form is never read | — | only `M_d` (derived) reaches the kernels: RT |
| `steps` | `list[int]` len r or `None` | no | each ≥ 1; Phase 0 supports all 1 (`step_mode="unit_step"`); >1 → `UnsupportedAxisValue` | `None` (all 1) | not passed (unit steps) |
| `slice_dim` | `int` (kw-only) | only with tensor begins | `[-r, r-1]`, canonicalized to `slice_dim + r` if negative **before** any check | `None` | refinement: CT (which dim the device begin offsets) |
| `num_devices` | `int` (kw-only) | only with tensor begins | divides `N_{slice_dim}` | `None` | refinement: `L = N_sd/num_devices` RT |

Host-side argument errors (`ValueError`): length of starts/ends/steps ≠ r; a step < 1; some `M_d == 0`; `slice_dim` without `num_devices` or the reverse; either with list begins; steps with tensor begins; `slice_dim ∉ [-r, r-1]`; begins tensors not uint32 of shape `(r,)`; `N_sd % num_devices != 0`. These are raised inside `validate()` while it builds the scenario dict (validate is the entry point's first line), before the SUPPORTED/EXCLUSIONS gate.

**Index axis canonicalization:** `slice_dim` is canonicalized to the non-negative convention (`slice_dim % r`) in the entry point before every check; negative begins/ends are canonicalized by the PyTorch clamp rule above. TARGET carries no index axis.

## Tensors

### Input

| Property | Requirement |
|----------|-------------|
| Shape | rank 2..4, any size; any dim may be non-tile-aligned |
| Dtype | Phase 0: `ttnn.bfloat16` (TARGET adds float32, int32, uint32, bfloat8_b, bfloat4_b) |
| Layout | Phase 0: TILE (TARGET adds ROW_MAJOR) |
| Memory | Phase 0: DRAM interleaved (TARGET adds L1 interleaved, L1 width-sharded) |

### Output

| Property | Value |
|----------|-------|
| Shape | `(M_0, …, M_{r-1})`; device form: `N` with `N_sd` replaced by `N_sd/num_devices` |
| Dtype | `input_tensor.dtype` |
| Layout | `input_tensor.layout` |
| Memory | `input_tensor.memory_config()` exactly (interleaved in the same buffer type; width-sharded: same grid, shard shape, orientation). Allocated by the entry point with `ttnn.allocate_tensor_on_device(ttnn.Shape(out_shape), dtype, layout, device, input_tensor.memory_config())`. Positions of an output tile beyond `(M_{r-2}, M_{r-1})` are padding and may hold anything. |

### Registry axes (names exactly as `feature_spec.TARGET`)

| Axis | Tagger reads | Phase 0 SUPPORTED |
|------|--------------|-------------------|
| `dtype` | `scenario["dtype"]` | `[ttnn.bfloat16]` |
| `layout` | `scenario["layout"]` | `[ttnn.TILE_LAYOUT]` |
| `memory_config` | `scenario["memory"]` (`"dram_interleaved"` / `"l1_interleaved"` / `"l1_width_sharded"`) | `["dram_interleaved"]` |
| `alignment` | begins `b` (normalized) and `out_shape` `M`: `"w_non_aligned"` if `b[-1]%32 or M[-1]%32`; elif `b[-2]%32 or M[-2]%32` → `"h_non_aligned"`; else `"tile_aligned"`. Device form: `M` only. | all three — `"h_non_aligned"` never in EXCLUSIONS, never split off by a tagger |
| `rank` | `len(scenario["shape"])` | `[2, 3, 4]` |
| `begins_source` | `scenario["begins_source"]` | `["host_list"]` |
| `step_mode` | `"unit_step"` iff steps is None or all 1, else `"strided"` | `["unit_step"]` |

`EXCLUSIONS = []`. `validate()` builds the scenario dict with the same keys `axes.classify_call` builds (`shape, dtype, layout, memory, shard, begins_source, starts, ends, steps, slice_dim, num_devices, begin_index, out_shape`) and runs the taggers on `(scenario,)`.

## Blocking Model

Everything below is downstream of this section. Semantics: `.claude/references/blocking-model.md`.

### Axes

The host normalizes every call to rank 4 by prepending unit dims (begin 0, extent 1) — a parameter derivation, not a tensor op. The output tile index space is then `(lead_0, lead_1, out_tile_row, out_tile_col)` with extents `(M_0', M_1', out_rows_per_image = ceil(M_{r-2}/32), out_cols = ceil(M_{r-1}/32))`. `total_out_tiles = M_0'·M_1'·out_rows_per_image·out_cols` (alignment-aware: `ceil`, per image — every image is tile-padded independently).

**Fold decision.** All four axes are independent and the op applies the identical per-tile operation along each (one input page per output tile, or ≤ 4 input pages at fixed relative positions in the shifted regime). Interleaved pages have no address contiguity along *any* axis (consecutive page ids land in consecutive banks), so a rectangular block buys no coalescing that a run of the same size lacks. The four axes therefore fold into one **flattened output-tile axis** (row-major over the four), and the block is a run of `block_tiles` consecutive flattened output tiles (blocking-model §1: "leading axes fold into one when the op treats them identically"). The run lets the core split balance to ±1 tile on every production shape — tall-narrow p3 `(16,4096,12)` = 2048×1 tiles and short-wide `(1,3844,5120)` row = 1×160 tiles alike — which no fixed rectangle does. The one place a rectangle *does* buy something (source-tile reuse in the shifted regime) is the deferred regime R3.

| Axis | Character (+ one-clause reason) | Extent knob | Phase 0 value | Knob source | Core-assignment | Later unlock |
|------|--------------------------------|-------------|---------------|-------------|-----------------|--------------|
| `lead_0` (dim r-4 after rank-4 normalization, extent `M_0'`) | independent — each output element reads exactly one input element; no result spans it | folded: covered by `block_tiles` runs (a run wraps from `out_tile_col` into `out_tile_row`, `lead_1`, `lead_0`) | `block_tiles` (see flattened row) | `block_tiles` host constant | flattened contiguous split (row below) | knob-turn |
| `lead_1` (dim r-3, extent `M_1'`) | independent — same reason | folded into `block_tiles` | as above | as above | flattened split | knob-turn |
| `out_tile_row` (dim r-2, `ceil(M_{r-2}/32)` per image) | independent — a begin inside a tile makes an output tile read 2 adjacent input tile rows, but no *result* spans the axis | folded into `block_tiles` | as above | as above | flattened split | knob-turn (rectangular blocks for source reuse = R3, deferred) |
| `out_tile_col` (dim r-1, `ceil(M_{r-1}/32)`) | independent — same reason | folded into `block_tiles` | as above | as above | flattened split | knob-turn |
| **flattened output-tile axis** (`total_out_tiles`) | independent (product of the four above) | `block_tiles` — output tiles per block = pages per CB reserve/push, per read barrier, per writer wait/pop | `min(core_tiles_max, block_tiles_l1_max)` — the coarsest block that fits the L1 budget, capped at the core's whole assignment | host: one constant computed in the descriptor builder, passed as a CT arg to reader and writer; CB capacities derive from it | `ttnn.split_work_to_cores(grid, total_out_tiles, row_wise=True)`: contiguous ranges, ±1 tile; cores used = `min(total_out_tiles, grid cores)` | knob-turn |
| `source_tile` footprint (shifted regime only; ≤ `src_per_out_cap` input tiles per output tile, `src_per_out_cap = (1+[o_h≠0])·(1+[o_w≠0])`) | independent — fixed relative footprint `q_a..q_b × c_a..c_b` of each output tile, overlapping with its neighbours' (affine footprint, blocking-model §6) | `src_per_out_cap` (not a knob: set by the begin offsets) | 1, 2 or 4 | host, derived from `o_h = b_{r-2}%32`, `o_w = b_{r-1}%32` | follows its output tile | scheme-change to reuse = R3 |
| `starts` tensor (device-tensor form, refinement) | reuse-shared — every core needs the same value `starts[sd]` | 1 page | 1 page per core | — | every working core reads it (R7, stepping stone) | scheme-change: read once + multicast (R8, deferred) |

`core_tiles_max = ceil(total_out_tiles / num_cores_used)` (group-1 count of `split_work_to_cores`).
`block_tiles_l1_max = floor((L1_BUDGET_BYTES − out_l1_bytes_per_core) / ((cb_output_depth + src_per_out_staging) · page_size))`, `L1_BUDGET_BYTES = 1347776` (BB-05), `out_l1_bytes_per_core = 0` in Phase 0 (out in DRAM), `src_per_out_staging = 0` in the aligned regime and `src_per_out_cap` in the shifted regime, `page_size = input_tensor.buffer_page_size()` (2048 B bf16 tile). bf16 values: aligned 329, row-shift 164, row+column shift 109 tiles.

Why the per-core assignment is not one block: p1 `(16,4096,8704)→(16,4096,4352)` is 278528 output tiles; on a 64–130-core grid that is 2143–4352 tiles (4.4–8.9 MB) per core, far above L1. So `block_tiles` is L1-bound for every large shape and assignment-bound (whole assignment = one block) for every small one.

Intermediate data of the scheme: the shifted regime's staged source tiles (`cb_source_tiles`) are a per-block footprint of the block's own output tiles — their axes are the same four, already split with their output tiles; no stage of this op runs on a designated core while others wait (there is no collect / coordinate role in any built regime).

### Buffer-depth knobs

| CB | Depth knob | Phase 0 value | What the depth buys |
|----|------------|---------------|---------------------|
| `cb_output_tiles` | `cb_output_depth` (host constant; CB capacity = `cb_output_depth · block_tiles` pages) | 2 | reader fills block k+1 while writer drains block k (read/write overlap on the two NoCs) |
| `cb_source_tiles` (shifted regime only) | `cb_source_depth` | 1 | staging for one block's source tiles; depth 2 would let the reader issue block k+1's reads before assembling block k (perf lamp L7) |

### Mechanism caps

| Mechanism | Cap on which extent | Clamp | What happens unclamped |
|-----------|--------------------|-------|------------------------|
| DRAM read alignment (32 B WH, 64 B BH; `ttnn.get_dram_alignment()`) | sub-tile DRAM reads in the shifted regime (a 32 B bf16 face row at an odd row offset is not 64 B aligned) | only **whole input tiles** are read from DRAM; sub-tile moves happen in L1 from `cb_source_tiles` | misaligned reads return wrong bytes or hang the NoC |
| L1 alignment of NoC transfers (16 B, `ttnn.get_l1_alignment()`) | self-aimed L1→L1 copies in the assemble step | face-row runs with `o_w == 0` are multiples of 32 B at 32 B-aligned offsets (bf16) → legal NoC self-reads; runs with `o_w != 0` are element-granular → RISC-V loads/stores (`l1_helpers.hpp:30` `addr_to_l1_ptr`) | wrong elements at odd column offsets |
| `NOC_MAX_BURST_SIZE` (stateful single-packet self-reads, `local_copy_helpers_dataflow.hpp:66-75`) | size of one face-row run if the stateful path is used | a run is ≤ 16 face rows × 16 elements × datum = 512 B (bf16) per face; below the burst size | static_assert failure at compile time |
| CB ring wrap (pointers reset only on an exact `fifo_limit` hit) | push/pop quantum | every reserve/push/wait/pop is the nominal `block_tiles`; the last block's NoC transfers cover only `last_block_extent = core_tiles − (num_blocks_this_core−1)·block_tiles` | ring pointer desync → corrupt pages or hang |
| Source tiles per output tile | staging capacity | `cb_source_tiles` capacity `src_per_out_cap · block_tiles`, with fixed slots per output tile (slot `2·dr + dc` for `dr, dc ∈ {0,1}`) | staging overrun overwrites the output CB region |
| 32-bit page ids / RT args | flattened indices | largest Phase 0 input has 557056 pages, total < 2^32 | none in TARGET |
| Block formats (bfloat8_b / bfloat4_b, refinement) | sub-tile moves | row moves are exact only as (exponent byte + 16 mantissas) per face row; a column offset must be a multiple of 16 (one exponent group); every production block-format case is aligned (R9 regime row) | regrouped exponents → values change |
| Begin bound of the device form (refinement) | `r0` | `r0 = min(floor(idx/32)·32, N_sd − L)` on device, so reads stay inside x for every value the tensor holds | out-of-bounds DRAM read |

### Regimes

Selection predicate (host, exact; only the begins decide — never the grid, so a regime cannot appear on one device and not another):
`aligned  = layout == TILE and b'[-2] % 32 == 0 and b'[-1] % 32 == 0` (b' = normalized rank-4 begins) → **R1**; otherwise (TILE) → **R2** with CT flags `SHIFT_ROWS = (b'[-2] % 32 != 0)`, `SHIFT_COLS = (b'[-1] % 32 != 0)`. The descriptor builder exposes the chosen regime and flags (`g6_slice.last_plan()` / returned plan dict) for BB-08.

Minimum boundary used in every row: **each input tile (page) that holds a selected element crosses DRAM once, each output tile crosses DRAM once** (`x_sel` = output tile count × page size for TILE).

| Regime | Status | Predicate | Block | Data movement vs. minimum | What a bigger block buys |
|--------|--------|-----------|-------|---------------------------|--------------------------|
| **R1 aligned tile copy** | **built** (Phase 0) | TILE, host begins, `b'[-2]%32==0 and b'[-1]%32==0`. Covers 791 of the 868 Phase 0 cases, incl. p1–p4, all leading-dim slices, all `M`/`N` non-aligned whole-tile cases. Small calls (`total_out_tiles < grid cores`) are this regime with one tile per core. | `block_tiles` consecutive flattened output tiles; reader reads each one's input page straight into `cb_output_tiles` (no input CB) | **minimum**: x_sel read once (p1: 570425344 B), out written once (570425344 B), no cross-core traffic | per block: one `cb_reserve_back`/`cb_push_back` + one `noc_async_read_barrier` (reader), one `cb_wait_front`/`cb_pop_front` + one `noc_async_writes_flushed` (writer), pipeline fill/drain once per core. Intended frequency: once per `block_tiles` pages; TensorAccessor setup and start-coordinate unravel once per kernel. |
| **R2 shifted tile gather** | **built** (Phase 0) | TILE, host begins, `b'[-2]%32 != 0` or `b'[-1]%32 != 0`. Covers the 77 production row-shift cases (all `M_{r-2} = 1`, offsets 1..31) and every other begin inside a tile of the last two dims. | `block_tiles` output tiles; the block's source tiles (≤ `src_per_out_cap` each, only those intersecting the selected input region: `q_b = q_a + [o_h + rows_t − 1 ≥ 32]`, same for columns) staged in `cb_source_tiles`, assembled into `cb_output_tiles` | production cases: **minimum** (an `M_{r-2}=1` output reads exactly one input tile per output tile). General: source tiles shared by adjacent output tiles are re-read — at most `src_per_out_cap ×` x_sel (2× for a multi-row row shift, 4× for row+column), never an input tile outside `[floor(b/32), ceil(e/32))` per dim | R1's terms, plus: one assemble pass per block (face-row runs), staging handshake once per block |
| R3 shifted gather with source reuse | deferred — no INPUTS case re-reads anything under R2 (all 77 row-shift cases have `M_{r-2}=1`; no production TILE case has `b[-1]%32≠0`), so R3 would save **0 bytes** on the golden suite | R2 predicate and (`M_{r-2} > 32 − o_h` or `o_w ≠ 0`) | rectangular blocks `block_rows × block_cols` of output tiles; footprint `(block_rows + [o_h≠0]) × (block_cols + [o_w≠0])` source tiles (affine) | reaches the minimum for every shifted shape (saves up to 1×–3× x_sel vs R2) | Reachable: R2's staging CB and slot addressing stay; only the block shape (run → rectangle) and the source-tile walk change. |
| R4 L1 interleaved | deferred — `memory_config` refinement (24 cases), not Phase 0 by the spec's Phase 0 list | `memory == "l1_interleaved"` | same as R1/R2 | minimum (L1 banks instead of DRAM) | Knob-turn: identical kernels (TensorAccessor resolves L1 banks); host subtracts `out_l1_bytes_per_core = ceil(out_pages / num_l1_banks) · page_size` from the budget (BB-05/rule 10). |
| R5 L1 width-sharded, in place | deferred — `memory_config` refinement (2 cases) | `memory == "l1_width_sharded"` | one block = the core's whole output shard (8 tiles in production) | 0 DRAM; production (b_w = 0, same shard spec): output shard k is on the core of input shard k → local L1 copy only. General b_w: the source shard is read from the core that holds it (its only copy, not a re-read) | Knob-turn on the logical split (the flattened split already *is* a width shard of out); placement change: `cb_output_tiles` backed on the output shard (`ttnn.cb_descriptor_from_sharded_tensor`), source read in place; cores with no output shard (2 of 36) idle. |
| R6 ROW_MAJOR stick window | deferred — `layout` refinement (202 cases) | `layout == ROW_MAJOR` | `block_sticks` output sticks (page = one row) | minimum when `b_w·datum % dram_align == 0` (every bf16/uint32 case); otherwise read the aligned superset of the window (≤ +1 alignment unit per stick; one int32 case at byte 16) and shift in L1 | per block handshake as R1; split over sticks with `split_work_to_cores`; a stick-chunk knob for long sticks. Separate reader/writer pair (CT layout define), no structure of R1 is reused beyond the host split. |
| R7 device-tensor begin, per-core starts read | deferred — `begins_source` refinement (2 cases) | `begins_source == "device_tensor"` | R1 block | x_sel + one `starts` page (r·4 B) per working core | stepping stone: the reader (CT define `DEVICE_BEGIN`) reads `starts` page 0, computes `r0 = min(floor(starts[sd]/32)·32, N_sd − L)` and adds it to the start offset of dim `sd` (tile rows / tile cols / lead stride); writer unchanged; begin never on host, never a CT arg. Successor: R8. |
| R8 device-tensor begin, read once + multicast | deferred — successor of R7 (lesson RB-010); 160 output tiles in production, i.e. ≤ 2 tiles per core, so the per-core 16 B read is the whole duplication | as R7 | R1 block | x_sel + 1 starts page total + one multicast of 4 B to the working cores | `ttnn.Mcast` + `McastArgs` on the R7 structure; reader's begin read becomes a receive. |
| R9 block-format / 32-bit dtypes | deferred — `dtype` refinement | dtype ∈ {float32, int32, uint32, bfloat8_b, bfloat4_b} | as R1/R2 | as R1/R2 (page_size from dtype) | R1: knob-turn (page size, CB format = dtype; bytes copied raw). R2 for 4 B datums: datum size CT arg. R2 for block formats: exponent-aware row moves (face row = 1 exponent byte + 16 mantissas), column offsets multiples of 16. |
| R10 strided gather | deferred — outside TARGET (`step_mode` holds `"unit_step"` only, BB-22); `validate()` refuses steps > 1 | any step > 1 | — | — | The walk is already per-dim; leading-dim strides are a stride multiplier, intra-tile strides need an element gather in R2's assemble step. |
| R11 whole-row read, drop columns in L1 | **rejected** — superseded by R1 everywhere | — | — | 2× DRAM read bytes on p1/p2/p4 (1140850688 vs 570425344 B); forbidden by rule 5 | — |
| R12 layout round trip (untilize → slice → tilize) or any multi-program chain | **rejected** — superseded by R2 (BB-01, rule 1) | — | — | ≥ 3 programs and ≥ 3 full crossings | — |
| R13 compute-kernel copy (unpack → math → pack) | **rejected** — superseded by R1/R2 raw-byte moves | — | — | same bytes, but values pass through the unpacker/packer formats: int32/uint32 above 2^24 change, block-format exponents regroup (Precision, rule 3) | — |

### Traffic ranking

Counted per tier (DRAM crossings per tensor, cross-core transfers); no timing.

| Rank | Candidate split | x DRAM reads | out DRAM writes | Cross-core | Verdict |
|------|-----------------|--------------|-----------------|------------|---------|
| 1 | Flattened contiguous output-tile split (chosen, R1/R2) | x_sel once (R2 general: ≤ `src_per_out_cap`· x_sel; production: x_sel) | once | none | **chosen**: minimum bytes and fills the grid on every shape |
| 1 (tie on bytes) | Split by output tile rows only | x_sel | once | none | same bytes; under-fills the grid on short-wide outputs (`(1,1,5120)` row case: 1 tile row → 1 core). Rejected on occupancy, not traffic (`width_split` catalog entry). |
| 1 (tie on bytes) | Split by output tile columns only | x_sel | once | none | same bytes; under-fills on tall-narrow (p3 12-column cases: 1 tile column → 1 core). Rejected on occupancy. |
| 1 (aligned) / better (shifted) | 2-D rectangular split (block of tile rows × tile cols per core) | aligned: x_sel; shifted: removes per-tile re-reads inside a rectangle | once | none | = R3, deferred (0 bytes saved on INPUTS) |
| — | "Split a dependent axis + combine" | — | — | — | **not applicable**: the op has no dependent axis (nothing reduces, scans or combines; math.md "Axes and dependence"). No combine regime exists. |
| last | Read whole input tile rows, drop columns | up to 2× x_sel | once | none | R11, rejected |

**Operand-reuse check** (every operand × chosen split): x varies along every output axis, so no core re-reads another core's bytes — no broadcast row. The only operand constant along the split is the device-form `starts` tensor → R7 (built later as stepping stone) / R8 (multicast successor).

**Stall-shadow check.** The writer waits on `cb_output_tiles`; the reader waits on its read barrier. Double-buffered `cb_output_tiles` puts the writer's block k drain into the shadow of the reader's block k+1 fetch. In R2 the reader assembles block k only after block k's read barrier; block k+1's source reads are independent of that assembly and could be issued before it (needs `cb_source_depth = 2`) — recorded as perf lamp L7, not built, because every production R2 case has ≤ 10 output tiles per core (640 tiles max over the grid). Nothing in the op changes floating-point order (no arithmetic).

### Block schedule

```cpp
// per core: num_blocks_this_core = ceil(core_tiles / block_tiles); last_block_extent = core_tiles - (num_blocks_this_core-1)*block_tiles
for (uint32_t block_idx = 0; block_idx < num_blocks_this_core; ++block_idx) {
    load_block(block_idx);       // reader
    assemble_block(block_idx);   // reader, R2 only (compiled out in R1)
    store_block(block_idx);      // writer
}
```

| Operation | Block shape | Resident across it | Intended frequency of fixed costs |
|-----------|-------------|--------------------|-----------------------------------|
| `load_block` | `block_tiles` flattened output tiles (`last_block_extent` on the last block). R1: one input page per output tile read into its `cb_output_tiles` slot. R2: the ≤ `src_per_out_cap` source tiles of each output tile read into its fixed slots in `cb_source_tiles`. Page ids come from an incremental 4-D counter `(i0, i1, R, C)` with carry (`in_page = ((i0+b0)·N1 + (i1+b1))·Ht_in·Wt_in + (q0+R)·Wt_in + (c0+C)`); the core's start coordinates are RT args (no div/mod per tile). | the counter state | 1 `cb_reserve_back(block_tiles)` + 1 `noc_async_read_barrier` per block (R1 pushes right after the barrier); TensorAccessor built once per kernel |
| `assemble_block` (R2) | the same `block_tiles` output tiles; per output tile copy, for each valid output row `i < rows_t` and each output face column half, the ≤ 2 contiguous runs from source tile `(i + o_h ≥ 32)`, row `(i + o_h)%32`, columns `o_w + 16h …` (crossing into source column tile `c_b` when `≥ 32`). `o_w == 0`: runs are whole face-row ranges — 8 runs per tile (2 per output face). | `cb_source_tiles` (whole block), `cb_output_tiles` reserved slots | once per block, then 1 `cb_push_back(block_tiles)` |
| `store_block` | `block_tiles` output pages, contiguous output page ids `out_start + block_idx·block_tiles …` | — | 1 `cb_wait_front(block_tiles)` + 1 `noc_async_writes_flushed` + 1 `cb_pop_front(block_tiles)` per block; 1 `noc_async_write_barrier` per kernel at the end (BB-03) |

### Perf lamps

| Lamp | Why the default may be wrong here | Nearby alternative to measure |
|------|-----------------------------------|-------------------------------|
| L1 overlap | `block_tiles` at its L1 bound (329 bf16 tiles, 674 KB per block) means the writer starts only after a whole 329-tile block has landed and the reader issues 329 reads per barrier; catalog `double_buffer` puts the single-core sweet spot at 4–8 tiles per barrier with a `2·block` CB and shows no gain once the grid is DRAM-bound | `block_tiles` ∈ {8, 16, 32, 64} with `cb_output_depth` ∈ {2, 3, 4}, on p1 (perf focus, `LOOSE_CASES[0]`) |
| L2 barrier granularity | one drain-all read barrier per block idles the NoC at every block edge (catalog `trid_double_issue`: +11–63 % at small blocks, ~0 at a large CB) | per-block transaction ids (`set_read_trid` / `async_read_barrier_with_trid`, `local_copy_helpers_dataflow.hpp:32-34`) with 2–4 blocks in flight |
| L3 placement | row-wise `split_work_to_cores` (`row_wise=True`) is the catalog `noc_placement` recommendation for grid-spread interleaved copies | `row_wise=False`; reader/writer NoC swap |
| L4 bank stagger | contiguous flattened ranges start every core on an arbitrary bank, but cores whose ranges start at equal `C` with `Wt_in` a multiple of the bank count line up on one bank (catalog `bank_stagger`) | rotate each core's walk start by `core_idx % num_banks` within its range (CT switch) |
| L5 split reader | a single NCRISC issuing 329 reads + a single BRISC issuing 329 writes per block may be issue-bound on short p3 rows | both RISCs read+write disjoint halves of the core's range (catalog `split_reader`) |
| L6 grid synchronization (tiny calls) | `total_out_tiles < grid cores` puts 1 tile on each of up to 130 cores; launch/teardown per core may dominate a 1–16-tile copy (lesson RB-053: judge on unprofiled trace replay time) | `cores_used = 1` (or ≤ 8) for `total_out_tiles ≤ T_small` |
| L7 R2 stall shadow | assembly of block k serializes after block k's reads on NCRISC | `cb_source_depth = 2`: issue block k+1's source reads, then assemble block k |
| L8 R2 assembly mechanism | RISC-V word copies vs self-aimed NoC reads (`local_copy_helpers_dataflow.hpp:57-133`) for `o_w == 0` face-row runs | both, on case 587 (row 4095) |

## Dataflow Strategy

| Stage | Format | Mechanism | Notes |
|-------|--------|-----------|-------|
| DRAM x → L1 (R1) | input tiles, raw bytes in input dtype | reader (NCRISC, NoC0): `TensorAccessor::get_noc_addr(page_id)` + `noc_async_read_page`, whole pages, into `cb_output_tiles` slot `j` of the reserved block | only selected tiles are read (rule 5); no input CB (pack-into-destination) |
| DRAM x → L1 staging (R2) | input tiles | reader: whole-tile reads of the block's source tiles into `cb_source_tiles` fixed slots | whole tiles only (DRAM alignment cap) |
| staging → output CB (R2) | tile face layout, raw bytes | reader: face-row runs, L1→L1 (RISC-V loads/stores or NoC self-reads) | output rows `≥ rows_t` / cols `≥ cols_t` not written (padding) |
| L1 → DRAM out | output tiles | writer (BRISC, NoC1): `noc_async_write_page` of contiguous output page ids | output memory config = input's |
| Tensix ↔ Tensix | — | none in any built regime | R5 general b_w: a core reads its source shard from the holding core over the NoC (TensorAccessor on the sharded input, its only copy). R8: `ttnn.Mcast` (fixed sender = one working core, receivers = the rest) carries the 4 B begin; receivers block in `receive()` before the first page id. |

**Placement axis as blocking (TARGET `memory_config`).** `l1_interleaved`: knob-turn (same split, L1 banks, output L1 share in the budget). `l1_width_sharded`: the flattened output-tile split cut by tile columns *is* a width shard of out, and the W axis is independent, so the physical shard is a **knob-turn / placement change**, not a scheme change: core-assignment and per-core extent are read off the shard spec (one block = the whole 8-tile shard), `cb_output_tiles` is backed on the output shard (zero-copy), and the source is read in place from L1 (never re-read over the NoC when it is local).

**Structural impossibilities (note for `/golden-tests`).** None missing from INVALID. Note for the dtype refinement: a block-format TILE input with `b[-1] % 16 != 0` cannot be copied bit-exactly (it regroups shared exponents); no production case has it, and `h_non_aligned` (row shifts) is exact for block formats.

## Work Distribution

| Field | Value |
|-------|-------|
| Work unit | one output tile; blocks of `block_tiles` consecutive flattened output tiles |
| Grid | `device.compute_with_storage_grid_size()` (passed into the builder; never a literal) |
| Split | `(num_cores, all_cores, group_1, group_2, tiles_g1, tiles_g2) = ttnn.split_work_to_cores(grid, total_out_tiles, row_wise=True)`; reader and writer kernels both span `all_cores` (≤ 2 rectangular ranges: full rows + one partial row, BB-17); per-core RT args carry `(start_out_tile, core_tiles, start coordinates (i0, i1, R, C))` |
| Per-core work | `core_tiles ∈ {tiles_g1, tiles_g2}` output tiles = `num_blocks_this_core = ceil(core_tiles / block_tiles)` blocks |
| Remainder | ±1 tile between groups; within a core the last block has `last_block_extent` tiles (nominal CB counts, narrowed NoC transfers) |
| Geometry | `Ht_in = padded_shape[-2]/32`, `Wt_in = padded_shape[-1]/32` (per image), `out_rows_per_image = ceil(M_{r-2}/32)`, `out_cols = ceil(M_{r-1}/32)`, `q0 = b'[-2]//32`, `c0 = b'[-1]//32`, `o_h = b'[-2]%32`, `o_w = b'[-1]%32`. All `ceil` + per image. Common RT args: `M0', M1', out_rows_per_image, out_cols, N1', Ht_in, Wt_in, b0', b1', q0, c0, o_h, o_w, M_{r-2}, M_{r-1}`. |
| Program caching | begins/ends/shape quantities are RT values (not hashed by `generic_op_device_operation.cpp:83-102`), so one compiled program per (dtype, regime + shift flags, `block_tiles`, `cb_output_depth`, core ranges, TensorAccessor CT args). The entry point caches the built `ProgramDescriptor` per full call signature (shape, dtype, layout, memory config, normalized begins/ends, device) and on a hit patches only the buffer-address RT args (BB-07). |
| Regime-pinned tests | the acceptance test pins R1 and both R2 flag sets (`SHIFT_ROWS`, `SHIFT_ROWS+SHIFT_COLS`) by their begins; selection does not depend on the grid |

## Circular Buffers

| Semantic Name | Index | Page Size | Num Pages | Sizing rationale | Format | Producer | Consumer | Lifetime |
|---------------|-------|-----------|-----------|------------------|--------|----------|----------|-----------|
| `cb_output_tiles` | 0 | `page_size` (input tile size of dtype; 2048 B bf16) | `cb_output_depth · block_tiles` | spans the flattened axis by `block_tiles`; depth for read/write overlap | input dtype (raw bytes; no unpack/pack touches it) | reader | writer | whole kernel |
| `cb_source_tiles` (R2 only; not allocated in R1) | 1 | `page_size` | `cb_source_depth · src_per_out_cap · block_tiles` | spans the flattened axis by `block_tiles` × the per-tile source footprint | input dtype | reader | reader (scratch: reserved once, reused per block, never pushed) | whole kernel (live from `load_block` to end of `assemble_block` of each block) |

Push count = wait count: reader pushes `block_tiles` per block on `cb_output_tiles`, writer waits/pops `block_tiles` per block, `num_blocks_this_core` times each. `cb_source_tiles` has no cross-thread handshake (one RISC-V both fills and reads it, ordered by the read barrier).

## Block Operation Realization

| # | Block operation | Block shape | Helper? | Input CB (semantic name, pages, state) | Output CB (semantic name, pages) | CB state after |
|---|-----------------|-------------|---------|----------------------------------------|----------------------------------|----------------|
| 1 | `load_block` (R1) | `block_tiles` tiles | raw dataflow API (no helper covers a page gather) | — (DRAM x) | `cb_output_tiles`, `block_tiles` reserved and pushed after the read barrier | `block_tiles` pages visible to writer |
| 1' | `load_block` (R2) | `block_tiles` × ≤ `src_per_out_cap` source tiles | raw dataflow API | — (DRAM x) | `cb_source_tiles` slots (no push) | staged, read barrier done |
| 2 | `assemble_block` (R2) | `block_tiles` tiles | raw L1 copy (optionally `local_copy_helpers_dataflow.hpp` self-reads for `o_w == 0`) | `cb_source_tiles`, staged | `cb_output_tiles`, `block_tiles` reserved, pushed after assembly | `block_tiles` pages visible to writer |
| 3 | `store_block` | `block_tiles` pages (`last_block_extent` transfers on the last) | raw dataflow API | `cb_output_tiles`, wait `block_tiles` | — (DRAM out) | popped after `noc_async_writes_flushed` |

## API Mapping

| Block operation | Type | Function | File:Line | Template Params / Args | Input CB | Output CB | Which params are block knobs |
|-----------------|------|----------|-----------|------------------------|----------|-----------|------------------------------|
| `load_block` / `store_block` | raw_api | `TensorAccessor::get_noc_addr(page_id, offset, noc)` | `tt_metal/hw/inc/api/tensor/tensor_accessor.h:165` | CT `TensorAccessorArgs` of x / out | — | — | none |
| `load_block` | raw_api | `noc_async_read_page` | `tt_metal/hw/inc/api/dataflow/dataflow_api.h:1243` | `(page_id, accessor, l1_dst_addr)` | — | `cb_output_tiles` / `cb_source_tiles` | called `last_block_extent ≤ block_tiles` times per block |
| `load_block` | raw_api | `noc_async_read_barrier` | `dataflow_api.h:1921` | — | — | — | once per block |
| `load_block` / `assemble_block` | raw_api | `cb_reserve_back`, `get_write_ptr`, `cb_push_back` | `dataflow_api.h:406`, `:325`, `:208` | `(cb_output_tiles, block_tiles)` | — | `cb_output_tiles` | `block_tiles` |
| `assemble_block` | raw_api | `get_write_ptr(cb_source_tiles)` + `addr_to_l1_ptr` loads/stores | `dataflow_api.h:325`; `ttnn/cpp/ttnn/kernel_lib/l1_helpers.hpp:30` | datum size (CT) | `cb_source_tiles` | `cb_output_tiles` | — |
| `assemble_block` (`o_w == 0` option) | helper | `dataflow_kernel_lib::local_addr`, `set_read_state`, `read_with_state` | `ttnn/cpp/ttnn/kernel_lib/local_copy_helpers_dataflow.hpp:57`, `:75`, `:95` | run size (≤ 512 B bf16) | `cb_source_tiles` | `cb_output_tiles` | — |
| `store_block` | raw_api | `cb_wait_front`, `get_read_ptr`, `noc_async_write_page`, `noc_async_writes_flushed`, `cb_pop_front` | `dataflow_api.h:479`, `:346`, `:1418`, `:1980`, `:260` | `(cb_output_tiles, block_tiles)` | `cb_output_tiles` | — | `block_tiles` |
| end of writer | raw_api | `noc_async_write_barrier` | `dataflow_api.h:1951` | — | — | — | once per kernel |

**Helpers considered and rejected** (apply to every `raw_api` row; there is no compute phase and no compute kernel):

| Candidate | File:Line of the mismatch | Concrete reason |
|-----------|---------------------------|-----------------|
| `compute_kernel_lib::copy` (eltwise) | `ttnn/cpp/ttnn/kernel_lib/eltwise/api/convenience.hpp:113` | a compute-kernel copy moves tiles through unpack → DEST → pack, i.e. through the unpacker/packer data formats; int32/uint32 above 2^24 and block-format exponents are not preserved (spec Precision, rule 3 = R13). It also cannot select sub-tile rows. |
| `compute_kernel_lib::tilize` / `untilize` | `tilize_helpers.hpp:206`, `untilize_helpers.hpp:163` | layout conversion inside or around the copy is forbidden (rule 1, R12), and both pass through unpack/pack formats |
| `dataflow_kernel_lib::read_sticks_for_tilize` / `write_sticks_after_untilize` | `tilize_helpers_dataflow.hpp:88`, `:130` | stick (ROW_MAJOR) transport feeding a tilize; Phase 0 moves tiles. Candidate for R6 only. |
| `tile_column_dataflow.hpp` column writers | `ttnn/cpp/ttnn/kernel_lib/tile_column_dataflow.hpp:7-24` | writes one column of a constant / moves one column into column 0 (broadcast operands); R2 moves row runs at arbitrary offsets between two source tiles |
| `mcast` helpers | `mcast/kernel/mcast_args.hpp` | no operand is shared in Phase 0; used by R8 |

## Key Risks and Gotchas

| Risk | Why it bites here | Mitigation in this design |
|------|-------------------|---------------------------|
| Row runs from the wrong face (R3 of risks.md) | output row `i` lives in face `(i ≥ 16 ? 2 : 0)` row `i%16` (left half) / `+1` (right half); a source row `o_h + i ≥ 32` comes from the next tile row | assemble by face-row runs computed per output face; acceptance test covers offsets 31 (case 587), 30 (case 1459), 4 at row 196 (case 191), 17 with 3 output tile rows (two-tiles rule case), and a row+column shift |
| Reading an input tile that does not exist | for the last output tile row, `q_a + 1` may lie past the image's tile rows (next image, or past the buffer) | read `q_b = q_a + [o_h + rows_t − 1 ≥ 32]` only (same for columns): never an input tile outside the selected range |
| Ragged last block / ring wrap | `core_tiles` is rarely a multiple of `block_tiles` | nominal `block_tiles` on every reserve/push/wait/pop, `last_block_extent` NoC transfers only |
| L1 budget vs allocator space | a 2·329-page CB is 1.35 MB; the device's free CB space may be smaller than the model's 1347776 B | builder clamps `L1_BUDGET_BYTES` to the device's allocatable CB space; budget expression in `l1_ledger.md`; L1 output share subtracted in R4/R5 |
| Program cache / trace replay with changed begins (risks R9) | 958 distinct begin sets in production | begins are RT values (outside the generic-op hash), descriptor cached per call signature with only addresses patched; no host read of tensor values; out allocated per call outside any kernel build |
| Kernel text size under trace (RB-038, BB-18) | R1, R2 (2 flag sets), R6, R7 paths in one reader | each path behind a CT define (`SHIFT_ROWS`, `SHIFT_COLS`, `DEVICE_BEGIN`, layout) so each compiled role carries only its path |
| In-flight transactions at kernel exit (BB-03) | writer pops after `noc_async_writes_flushed` | final `noc_async_write_barrier` in the writer; reader ends on a read barrier |
| Profiling zones (BB-21) | role per core must be dumpable | one `DeviceZoneScopedN` per block operation and per blocking wait (`cb_reserve_back`, read barrier, `cb_wait_front`); the builder can dump `core → (reader, writer, tiles)` |
| Grid with fewer cores than output tiles in the shifted regime | none — selection is begin-only | regime-pinned tests run on whatever grid the device has |
