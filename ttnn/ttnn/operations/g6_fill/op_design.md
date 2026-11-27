# Operation Design: g6_fill

## Overview

| Field | Value |
|-------|-------|
| Classification | data_movement (write-only generator: no tensor is read, no compute kernel) |
| Goal | Write `e = cast_T(fill_value)` into every logical position of `output_tensor` (rank 1, `[N]`) with one device program of this op's own, and return `output_tensor`. The call is a few pages, so the program's fixed cost is what it costs. |
| Math | `e = cast_T(v)` once on the host; `out[i] = e` for `0 <= i < N`. `input_tensor` supplies shape, dtype, layout and placement only. |
| References | `eval/golden_tests/g6_fill/feature_spec.py` (TARGET, INPUTS, LOOSE_CASES); `eval/golden_tests/g6_fill/helpers.py` (oracle `pytorch_g6_fill`, `TOLERANCES` = exact); `eval/op_template.py` (registry model); `.claude/references/blocking-model.md`; `.claude/references/l1-footprint-discipline.md`; `.claude/references/generic_op_template/`; `ttnn/ttnn/operations/examples/master.md` entries `noc_placement`, `width_split`, `double_buffer`, `tile_reorder`, `split_reader` (see "Catalog entries used" below). The exploration bundle `docs/explorations/g6_fill/` does not exist on this branch, so this design works from the spec. |

### Catalog entries used

| Entry | Knob it informed | What it says that applies here |
|-------|------------------|--------------------------------|
| `noc_placement` | core placement and NoC choice | Spread a line of interleaved-DRAM writers along a **row** (`row_wise=True` order), and keep writes on NoC1 (the `WriterConfigDescriptor` default). |
| `width_split` | core-assignment axis | A one-tile-row tensor (here `H = 1` always) must be split along **W tiles**, otherwise one core does everything. |
| `double_buffer` | buffer depth | Depth buys overlap between a producer and a consumer. This op has no such pair: the L1 source is written once and read by every NoC write, so depth 1 is the decision. |
| `tile_reorder` | transfer shape | Prefer whole contiguous transfers with a batched barrier. Here the largest contiguous logical run in a tile is one face row (64 B), so the transfer unit is the face-row segment. Writes are issued back to back with one barrier per kernel. |
| `split_reader` | RISC assignment | Splitting issue work across BRISC and NCRISC pays only when one RISC is issue-bound. At 1–2 segments per core it is not, so this is recorded as a lamp, not built. |

## Parameters

| Name | Type | Required | Valid Range | Default | CT/RT |
|------|------|----------|-------------|---------|-------|
| `input_tensor` | `ttnn.Tensor` | yes, positional | rank 1 `[N]`, `N >= 1`; dtype `float32` (Phase 0) or `int32` (refinement); on device; DRAM interleaved | — | Supplies shape, dtype, layout and placement only. Never bound to a kernel. |
| `fill_value` | `float` or `int` (not `bool`) | yes, positional | float32: `inf`, `-inf`, `nan`, or finite with `abs(v) <= 3.4028234663852886e38` (int or float). int32: `int` in `[-2**31, 2**31 - 1]`; a `float` is refused. | — | Converted once on the host to the 32-bit pattern `e_bits`, then passed as a **common RT arg** (never a CT arg, never in the cache key). |
| `output_tensor` | `ttnn.Tensor` | yes, keyword-only | same shape, dtype, layout and placement as `input_tensor`; may be `input_tensor` itself | — | Its buffer address is a **common RT arg**. Its `TensorAccessorArgs` are CT args. |

Not parameters: `memory_config` and `compute_kernel_config`. A form without `output_tensor` does not exist.

### Host conversion `e_bits = cast_T(v)` (single source of the element)

| dtype | `v` type | Conversion (exact, one rounding) | Notes |
|-------|----------|----------------------------------|-------|
| float32 | `float` | `struct.unpack("<I", struct.pack("<f", v))[0]` | C `(float)double` cast: RNE, keeps `-0.0`, keeps subnormals (`1e-40` → `0x000116c2`), keeps `±inf`, NaN → `0x7fc00000`. Validation already refused finite `abs(v) > FLT_MAX`, so the pack cannot overflow. |
| float32 | `int` | Exact integer RNE to 24 significant bits **before** any `float()`: `m = abs(v)`; if `m.bit_length() > 24`: `shift = m.bit_length() - 24`, `q, r = divmod(m, 1 << shift)`, round half to even on `(r, 1 << (shift - 1), q & 1)`, `m = q << shift`. Then `float(m)` is exact, so pack `"<f"` and OR in the sign bit. | **`struct.pack("<f", float(v))` is wrong for ints:** int → double → float rounds twice. Measured on this host: `2**60 + 2**36 + 1` gives `0x5d800000` through `float()`, while the correct single rounding (also the torch oracle) is `0x5d800001`. |
| int32 | `int` | `v & 0xFFFFFFFF` | Exact two's complement, after the range check. |

## Tensors

### Input

| Property | Requirement |
|----------|-------------|
| Shape | `[N]`, rank 1, `N >= 1`, no upper limit |
| Dtype | `ttnn.float32` (Phase 0), `ttnn.int32` (refinement) |
| Layout | TILE (float32, Phase 0); ROW_MAJOR (int32, refinement: one page per row) |
| Memory | DRAM interleaved |
| Read? | **Never.** No kernel binds it, and it is not passed to `generic_op` (`io_tensors = [output_tensor]`). |

### Output

| Property | Value |
|----------|-------|
| Shape | `[N]` (= `input_tensor.shape`) |
| Dtype | `input_tensor.dtype` |
| Layout | `input_tensor.layout` |
| Memory | DRAM interleaved, `output_tensor`'s own buffer. No device tensor is allocated. The op returns `output_tensor`. |

### Tile geometry (TILE layout, Phase 0)

| Quantity | Expression (derived, never a literal) | float32 value |
|----------|---------------------------------------|---------------|
| `tile_h, tile_w` | `output_tensor.tile.tile_shape` | 32, 32 |
| `face_h, face_w` | `output_tensor.tile.face_shape` | 16, 16 |
| `elem_bytes` | `output_tensor.element_size()` | 4 |
| `tensor_h_tiles` | `ceil(1 / tile_h)` (rank 1 reads as tile plane `[1, N]`) | 1 |
| `tensor_w_tiles` | `ceil(N / tile_w)` | — |
| `segment_bytes` | `face_w * elem_bytes`: one face row, the longest contiguous logical run in a tile | 64 |
| `segment_stride_bytes` | `face_h * face_w * elem_bytes`: face 0 row 0 → face 1 row 0 | 1024 |
| `segments_per_page` | `tile_w / face_w` | 2 |
| `tensor_segments` | `ceil(N / face_w)`: the logical segments. A tail tile with `<= face_w` logical positions has only its face-0 segment. | — |
| Segment `s` lands at | page `s // segments_per_page`, byte offset `(s % segments_per_page) * segment_stride_bytes` | — |

Logical positions of tile `t` are row 0, columns `0 .. min(tile_w, N - t*tile_w) - 1`: face 0 row 0 holds the first 16 and face 1 row 0 the next 16. Rows 1–31 of every tile, and faces 2–3, are padding. The op never writes them. Their content is unspecified by the contract.

## Blocking Model

### Axes

| Axis | Character (+ one-clause reason) | Extent knob | Phase 0 value | Knob source | Core-assignment | Later unlock |
|------|--------------------------------|-------------|---------------|-------------|-----------------|--------------|
| `w_tiles`: tile columns of the `[1, N]` tile plane | **independent**: every tile holds `e`, and no tile depends on another | `block_w_tiles` | `core_w_tiles`: the core's whole assignment, so one block per core | host `_plan()`. Becomes each core's per-core RT args `seg_start, seg_count`. | Spread across a single `rect_w x rect_h` rectangle anchored at core (0,0), balanced `q / q+1` tiles per core (see Work Distribution). Production: 1 tile per core. | knob-turn (core-count cap `max_active_cores`; see the grid-sync lamp) |
| `h_tiles`: tile rows of the tile plane | **independent** (trivially) | `block_h_tiles` | 1 = whole tensor extent (`tensor_h_tiles = 1` for rank 1) | derived: `ceil(1 / tile_h)` | not assigned across cores: extent 1, nothing to split | none (rank is fixed at 1 by the contract) |
| `segment` (face column within row 0 of a tile) | **independent**: face 0 and face 1 rows are disjoint byte ranges of the page | `segments_per_page` | 2 = whole tile row (the tail tile's count is narrowed by `tensor_segments`) | derived: `tile_w / face_w` (CT arg) | not split across cores. A split below one tile is under the granularity floor: it doubles the cores, each paying the same fixed launch and barrier, for the same bytes. | none |
| `segment_elems`: elements within one segment | **independent** | `segment_bytes` | `face_w * elem_bytes` = 64 B, the whole contiguous run | derived (CT arg). Also the size of `cb_fill_row`. | not split: it is one NoC transfer | knob-turn for the ROW_MAJOR refinement (the run becomes the row page) |
| `face_rows`: rows 1..15 of faces 0/1 and faces 2/3 | **not part of the result**: padding (spec, Shape handling) | — | 0 rows written | contract | — | — |

No axis is **dependent**: no result spans more than one position. No operand is **reuse-shared** through DRAM. The only shared quantity is `e`, 4 bytes, delivered to every core as a common RT arg by the dispatcher (see Operand-reuse check).

### Operand-reuse check (per operand vs. the chosen `w_tiles` split)

| Operand | Varies along `w_tiles`? | Consequence |
|---------|------------------------|-------------|
| `input_tensor` | — | Never read: 0 bytes, 0 crossings. |
| `e` (`e_bits`) | no: reuse-shared by construction | Delivered as one **common RT arg**: one dispatcher write per kernel range, no op-issued NoC traffic. Each core materializes it into its own 64 B `cb_fill_row`. An injector-plus-mcast regime has nothing to move that the RT arg does not already move, so it is `rejected` in the regime table. |

### Stall-shadow check

| Waiting stage | What it waits for | Independent work to schedule into the wait |
|---------------|-------------------|--------------------------------------------|
| `drain_writes` (one `noc_async_write_barrier` per kernel) | the write acks of every segment this core issued | **None.** Every write has been issued before the barrier, and the source is never rewritten. The kernel has no other work. |
| first `store_block` write | `materialize_fill_row` (16 RISC stores) must be in L1 before the NoC reads the source | None: no write can start before its source exists, and all writes use the same source. |

### Buffer-depth knobs

| CB | Depth knob | Phase 0 value | What the depth buys |
|----|------------|---------------|---------------------|
| `cb_fill_row` | `fill_row_depth` | 1 | Nothing more at depth > 1: the page is written once per kernel and then only read (as the NoC source) by every write. There is no producer/consumer pipeline to overlap. |

### Mechanism caps

| Mechanism | Cap on which extent | Clamp | What happens unclamped |
|-----------|--------------------|-------|------------------------|
| TILE face layout | one NoC transfer's length: at most one face row (`segment_bytes = face_w * elem_bytes`) is contiguous in a tile | `segment_bytes` is derived from `tile.face_shape`, and every write is exactly one segment | A 128 B write from offset 0 would put columns 16–31 into face 0 row 1 (padding) and leave face 1 row 0 unwritten. `test_last_tile_second_face` ([185]) catches this. |
| `cb_fill_row` size | `segment_bytes <= cb_fill_row` page bytes | the CB page size **is** `segment_bytes` (same host value) | A write longer than the source would ship L1 bytes beyond the materialized row into the output. |
| NoC write alignment, Blackhole (`noc_parameters.h:376` L1 = 16 B, `:380` DRAM write = 16 B) | source and destination addresses must agree mod 16 | `cb_fill_row` base is L1-aligned. Segment destinations are `page_base + k * segment_stride_bytes` with `segment_stride_bytes = 1024` and an aligned `page_base`. | Misaligned writes corrupt or hang. Clamp holds by construction for TILE. Re-check it for the ROW_MAJOR refinement. |
| `noc_inline_dw_write` (`noc.h:655`: "does not support DRAM addresses") | rules out the inline path to DRAM entirely | not used | — |
| Outstanding non-posted writes per core | none practical at production extents (max 2 per core) | none. If a refinement ever raises `core_w_tiles` into the thousands, `block_w_tiles` becomes the barrier cadence. | — |

### Regimes

| Regime | Status | Predicate | Block | Data movement vs. minimum | What a bigger block buys |
|--------|--------|-----------|-------|---------------------------|--------------------------|
| `segment_write` | **built** | `dtype == float32 and layout == TILE` (all `N >= 1`; both alignment tags) | `block_w_tiles = core_w_tiles` tiles × `segments_per_page` segments (tail tile narrowed by `tensor_segments`). One block per core. | **Minimum** against the DRAM boundary: input crosses 0×. Output: each logical segment crosses once, `tensor_segments * segment_bytes` B in total. Over the logical bytes this adds only the padding of the tail segment (≤ 60 B). For [512]: 2,048 B in 32 writes. | Within a core: nothing further. The fixed costs (kernel launch, `materialize_fill_row`, one barrier round trip) are paid **once per kernel**, whatever `block_w_tiles` is. The extent that does trade fixed cost is **tiles per core**: fewer cores means less launch skew and more serial write issue per core. That is the grid-sync lamp. |
| `segment_write_row_major` | **deferred**: the contract assigns int32 ROW_MAJOR (`[1]`, FLUX_53) to the refinement after Phase 0. Phase 0's SUPPORTED is float32 TILE. Reachable as a knob-turn: the same writer kernel. Segment geometry is already three CT args: `segment_bytes`, `segments_per_page` (1 when one segment covers the page), `segment_stride_bytes` (= `segment_bytes`). The host adds the int32 conversion `v & 0xFFFFFFFF`. | `dtype == int32 and layout == ROW_MAJOR` | the row page's run, cut into segments of `min(page_bytes, segment_cap)` bytes. `[1]` is one 4 B segment on one core (the independent axis has extent 1, so one core is the full spread). | Minimum: one write of the logical bytes (4 B for `[1]`), input 0×. | same as `segment_write` |
| `whole_tile_write` | **rejected**: superseded everywhere by `segment_write`. Padding need not be written (spec, Shape handling). | — | whole tiles | 32× the minimum: 4,096 B per tile vs 128 B. Plus 1,024 RISC stores per core to materialize a tile vs 16. | — |
| `compute_fill` (TRISC `FillBitcast` into DEST → pack → CB → writer) | **rejected**: a dead end superseded by `segment_write`. It adds a compute kernel, a CB handshake and whole-tile writes. It puts the element through DEST and the packer, which needs `fp32_dest_acc_en` to keep 32 bits and can flush subnormals (`1e-40` is a probe value). | — | whole tiles | 32× the minimum (whole tiles) | — |
| `inline_dword_write` (`noc_inline_dw_write`, no L1 source) | **rejected**: the API does not support DRAM addresses (`noc.h:655`). Even where legal it would be 16 transactions per segment. | — | 4 B per transaction | 16× the transaction count | — |
| `injector_mcast_fill_row` (one core materializes the row and multicasts it to the others) | **rejected**: superseded by the common RT arg, which already delivers `e` to every core at zero op-issued NoC cost. The mcast adds a semaphore rendezvous for 64 B. | — | — | adds `num_active_cores - 1` × 64 B of NoC traffic plus a rendezvous | — |
| `host_upload` (build the result on the host, then copy) | **rejected**: forbidden by the contract (rule 1, BB-01). | — | — | — | — |

Single built regime for Phase 0, so there is no selection predicate yet. When `segment_write_row_major` lands, the predicate is the `(dtype, layout)` pair above, checked on the host. Its regime-pinned test is the FLUX_53 `[1]` case.

### Traffic ranking

| Rank | Candidate split | DRAM bytes (input / output) | Cross-core traffic | Verdict |
|------|-----------------|-----------------------------|--------------------|---------|
| 1 | `w_tiles` across a core rectangle, segment writes | 0 / `tensor_segments * 64` (2,048 B at [512]) | none | **chosen** |
| 1 (tie on bytes) | `segment` across cores (one face-row segment per core) | 0 / same | none | not chosen: same bytes, 2× the cores. Each core pays the same launch and barrier. Below the whole-tile granularity floor. |
| 1 (tie on bytes) | no split (one core) | 0 / same | none | not Phase 0 (single-core is disallowed). It is the far end of the grid-sync lamp: all `tensor_segments` writes issued serially on one RISC. |
| — | dependent axis + combine | — | — | does not exist: the op has no dependent axis |
| 4 | `w_tiles` split, whole-tile writes | 0 / 4,096 B per tile (65,536 B at [512]) | none | rejected (`whole_tile_write`) |

Every split moves the same minimum bytes. What separates them is core occupancy and fixed cost, which is the grid-sync lamp, not the traffic ranking.

### Block schedule

```cpp
materialize_fill_row();                                         // once per kernel
for (uint32_t block_idx = 0; block_idx < num_blocks_this_core; ++block_idx) {   // Phase 0: 1
    store_block(block_idx);                                     // one NoC write per segment of the block
}
drain_writes();                                                 // once per kernel
```

| Operation | Block shape | Resident across it | Intended frequency of fixed costs |
|-----------|-------------|--------------------|-----------------------------------|
| `materialize_fill_row` | one segment: `segment_bytes / elem_bytes` copies of `e_bits` (16 for float32) | — | once per kernel: 16 word stores into `cb_fill_row`, no CB handshake |
| `store_block` | `seg_count` segments = this core's `core_w_tiles` tiles, tail-narrowed | `cb_fill_row` (read-only source for every write) | Segment addresses come from `TensorAccessor::get_noc_addr(page, offset)`. Writes are issued back to back with **no barrier inside the block**. One accessor per kernel. |
| `drain_writes` | — | — | **once per kernel**: `noc_async_write_barrier()`. Every write is acked before the kernel returns (BB-03). |

Profiler zones (BB-21; zone macros compile to nothing in the plain build, `kernel_profiler.hpp:1047`): `G6FILL-MATERIALIZE` around `materialize_fill_row`, `G6FILL-STORE` around `store_block`, `G6FILL-DRAIN` around `drain_writes`.

### Perf lamps

| Lamp | Why the default may be wrong here | Nearby alternative to measure |
|------|-----------------------------------|-------------------------------|
| **Grid synchronization** (`max_active_cores`) | One tile per core leaves ~2 writes of work per core. The kernel-duration metric spans first-core start to last-core end, so launch skew across 16 cores may cost more than serial write issue saves. | On the PERF FOCUS case [512] (LOOSE_CASES[0]), compare `max_active_cores` ∈ {16 (default), 8, 4, 2}: 1, 2, 4, 8 tiles per core. |
| Rectangle aspect (BB-17, `noc_placement`) | The default is the fewest rows (`8x2` at [512]). A different aspect changes which NoC1 paths the writes take to the DRAM banks. | `4x4` vs `8x2` at [512], each a single core range. |
| Per-core RT args | `seg_start, seg_count` are unicast per core in the dispatch stream. Everything else is common. | All-common RT args (`q`, `r`, `rect_w`, `tensor_segments`), with each core deriving its range on device from `get_relative_logical_x/y()` (`dataflow_api.h:70,83`). |
| Write-issue RISC (companion of the grid-sync lamp; `split_reader`) | If the grid-sync lamp moves to 4+ tiles per core, one BRISC issues 8+ writes serially. | Split the core's segments between BRISC and NCRISC (two kernels over the same rectangle, each with its own 64 B source). Measure only if the grid-sync lamp lands at >= 4 tiles per core. |

No overlap lamp: there is no data-movement/compute pair to overlap.

## Dataflow Strategy

| Stage | Format | Mechanism | Notes |
|-------|--------|-----------|-------|
| Host: `v` → `e_bits` | 32-bit pattern | Python, exact conversion table above | Once per call. Never routed through a 16-bit type or through float for int32. |
| Host → core | `e_bits`, `out_addr` | **common RT args** of the writer kernel (`KernelDescriptor.common_runtime_args`) | Excluded from the program hash (`generic_op_device_operation.cpp:94,99` hash sizes, not values). Re-applied on a cache hit (`program_descriptors.cpp:188-213` copies per-core and common args). |
| Core: materialize | `segment_bytes / 4` × `e_bits` in `cb_fill_row` | RISC word stores (`fill_l1_range<4>`) | No CB reserve/push: the writer is the only user. |
| Core → DRAM | raw 32-bit words, `segment_bytes` per write | `noc_async_write(cb_fill_row_addr, accessor.get_noc_addr(page, offset), segment_bytes)` on NoC1 (writer default) | One write per logical segment. Destination page = output tile. Offset 0 (face 0 row 0) or `segment_stride_bytes` (face 1 row 0). |
| Completion | — | one `noc_async_write_barrier()` | BB-03 |

**TARGET axes as regime rows.** `alignment` (`w_non_aligned`, `h_non_aligned`): both land in `segment_write`. A non-multiple-of-32 `N` only narrows the tail tile's segment count through `tensor_segments`, which is host-computed. No kernel branch, no EXCLUSIONS entry. `dtype = int32` with `layout = ROW_MAJOR` is the `segment_write_row_major` deferred row, a knob-turn as described there. TARGET has no placement axis (DRAM interleaved only), so no sharded scheme applies.

**Structural impossibilities:** none. INVALID stays `[]`.

## Work Distribution

| Field | Value |
|-------|-------|
| Work unit | one tile column (`w_tiles` = 1), written as its logical face-row segments |
| Grid | `grid_x, grid_y = device.compute_with_storage_grid_size()`. Cap `max_active_cores = grid_x * grid_y` (host knob, single source). |
| Active rectangle | `target = min(tensor_w_tiles, max_active_cores)`, `rect_h = ceil(target / grid_x)`, `rect_w = target // rect_h`, `num_active_cores = rect_w * rect_h`. **One `CoreRange((0,0), (rect_w-1, rect_h-1))`** for the kernel and the CB (BB-17). `rect_w <= grid_x` and `rect_h <= grid_y` hold by construction. |
| Per-core work | Core linear index `i = y * rect_w + x` (row-major). `q, r = divmod(tensor_w_tiles, num_active_cores)`. `core_w_tiles(i) = q + (i < r)`. `core_start_tile(i) = i*q + min(i, r)`. `seg_start(i) = core_start_tile(i) * segments_per_page`. `seg_count(i) = min((core_start_tile(i) + core_w_tiles(i)) * segments_per_page, tensor_segments) - seg_start(i)`. |
| Remainder | Tiles: the `q / q+1` split above. Tail tile: `tensor_segments = ceil(N / face_w)` drops its face-1 segment when it holds `<= face_w` logical positions. All `ceil`-based. Rank 1 has one image, so there is no per-image padding beyond the tail tile. |

Production plans (`grid_x >= 8`, which holds on every Wormhole and Blackhole part):

| N | `tensor_w_tiles` | tail positions | `tensor_segments` | rectangle | tiles / core | output bytes written |
|---|------------------|----------------|-------------------|-----------|--------------|----------------------|
| 512 | 16 | 32 | 32 | 8 × 2 (`grid_x < 16`) | 1 | 2,048 |
| 256 | 8 | 32 | 16 | 8 × 1 | 1 | 1,024 |
| 207 | 7 | 15 | 13 | 7 × 1 | 1 | 832 |
| 196 | 7 | 4 | 13 | 7 × 1 | 1 | 832 |
| 185 | 6 | 25 | 12 | 6 × 1 | 1 | 768 |

**Plan report (BB-08, BB-21).** The op module exposes `g6_fill_plan(input_tensor) -> dict` with keys `regime`, `rect` (`rect_w, rect_h`), `per_core` (`{(x, y): (seg_start, seg_count)}`) and `role` (`"writer"` for every active core). It uses the same `_plan()` the descriptor uses.

## Host-side contract

| Concern | Decision |
|---------|----------|
| Entry point | `g6_fill(input_tensor, fill_value, *, output_tensor)`. The first line is `validate(input_tensor, fill_value, output_tensor=output_tensor)`. |
| `validate()` order | (1) Structural contract → `ValueError`: `output_tensor is None`; rank != 1; dtype not in {float32, int32}; either tensor not on device, or not DRAM interleaved; `output_tensor` shape, dtype, layout or memory config differs from `input_tensor`'s. (2) Value contract → `ValueError`: `bool`; not `int`/`float`; `float` for int32; int32 out of `[-2**31, 2**31-1]`; float32 finite with `abs(v) > 3.4028234663852886e38` (ints compared exactly, as ints). (3) SUPPORTED per axis → `UnsupportedAxisValue`. (4) EXCLUSIONS → `ExcludedCell`. Nothing is dispatched before all four pass, so a refused call leaves `output_tensor` unchanged. |
| Axes | Scenario `({"shape": tuple(input_tensor.shape), "dtype": input_tensor.dtype, "layout": input_tensor.layout, "fill_value": fill_value},)`. Taggers: `tag_dtype` → `inputs[0]["dtype"]`, `tag_layout` → `inputs[0]["layout"]`, `tag_alignment` → `"w_non_aligned"` if `N % 32 != 0` else `"h_non_aligned"`. |
| Phase 0 SUPPORTED (implementer writes it) | `dtype: [float32]`, `layout: [TILE_LAYOUT]`, `alignment: ["w_non_aligned", "h_non_aligned"]`. EXCLUSIONS `[]`. |
| Dispatch | `ttnn.generic_op([output_tensor], descriptor)`: exactly one program, and `output_tensor` is the only io tensor (in-place needs nothing extra). Return `output_tensor`. |
| Descriptor cache (BB-07) | Module-level dict keyed by `(device id, N, dtype, layout, grid_x, grid_y)`. A miss builds the `ProgramDescriptor` once. A hit rewrites only the writer's `common_runtime_args = [out_addr, e_bits]` on the cached descriptor (re-assign the attribute; do not rely on in-place mutation of a nanobind copy) and dispatches. |
| Trace (BB-16) | Nothing is allocated and nothing is built after the first call for a key. A replay rewrites the captured `e_bits`/address. |

## Circular Buffers

| Semantic Name | Index | Page Size | Num Pages | Sizing rationale | Format | Producer | Consumer | Lifetime |
|---------------|-------|-----------|-----------|------------------|--------|----------|----------|-----------|
| `cb_fill_row` | 0 | `segment_bytes` (64 B float32) | `fill_row_depth` (1) | Spans the `segment_elems` axis: one segment of `e`, the NoC source of every write. Streams over `w_tiles` and `segment` (re-read, never rewritten). Independent of `h_tiles` (extent 1). | `output_tensor.dtype` (raw 32-bit words; no unpacker or packer touches it) | writer (RISC stores) | writer (NoC source) | whole kernel |

One kernel is both producer and consumer, with no handshake: it is a scratch source, not a pipe. No CB has two kernels attached. Full ledger: `l1_ledger.md`.

## Block Operation Realization

| # | Block operation | Block shape | Helper? | Input CB (semantic name, pages, state) | Output CB (semantic name, pages) | CB state after |
|---|-----------------|-------------|---------|----------------------------------------|----------------------------------|----------------|
| 1 | `materialize_fill_row` | `segment_bytes / elem_bytes` words | yes: `dataflow_kernel_lib::fill_l1_range<4>` | — (`e_bits` from common RT arg) | `cb_fill_row`, 1 page (via `get_write_ptr`) | holds `e` × 16, read-only from here on |
| 2 | `store_block` | `seg_count` segments (`core_w_tiles` tiles) | raw: `noc_async_write` + `TensorAccessor::get_noc_addr` | `cb_fill_row`, 1 page, read-only | output DRAM pages (no CB) | unchanged |
| 3 | `drain_writes` | — | raw: `noc_async_write_barrier` | — | — | — |

## API Mapping

| Block operation | Type | Function | File:Line | Template Params / Args | Input CB | Output CB | Which params are block knobs |
|-----------------|------|----------|-----------|------------------------|----------|-----------|------------------------------|
| `materialize_fill_row` | helper | `dataflow_kernel_lib::fill_l1_range<val_size>(start_addr, n_bytes, val)` | `ttnn/cpp/ttnn/kernel_lib/l1_helpers.hpp:100` (doc :81-99) | `<4>`, `start_addr = get_write_ptr(cb_fill_row)`, `n_bytes = segment_bytes`, `val = e_bits` | — | `cb_fill_row` | `n_bytes` = `segment_bytes` |
| `store_block` | raw_api | `TensorAccessor::get_noc_addr(page_id, offset)` | `tt_metal/hw/inc/api/tensor/tensor_accessor.h:165` | `page_id = s / segments_per_page`, `offset = (s % segments_per_page) * segment_stride_bytes` | — | — | `seg_start`, `seg_count` (block extent), `segments_per_page`, `segment_stride_bytes` |
| `store_block` | raw_api | `noc_async_write(src_local_l1_addr, dst_noc_addr, size)` | `tt_metal/hw/inc/api/dataflow/dataflow_api.h:876` | non-posted (default), `size = segment_bytes` | `cb_fill_row` | output pages | `segment_bytes` |
| `drain_writes` | raw_api | `noc_async_write_barrier()` | `tt_metal/hw/inc/api/dataflow/dataflow_api.h:1951` | — | — | — | — |
| zones | raw_api | `DeviceZoneScopedN(name)` | `tt_metal/hw/inc/api/debug/kernel_profiler.hpp:1012` | `G6FILL-MATERIALIZE`, `G6FILL-STORE`, `G6FILL-DRAIN` | — | — | — |

**Helpers considered and rejected (`store_block`, `drain_writes`):**

| Candidate | File:Line | Why it cannot express the shape |
|-----------|-----------|---------------------------------|
| `dataflow_kernel_lib::local_noc_addr` / `l1_helpers` | `l1_helpers.hpp:41` | Builds a *local* L1 endpoint. The destination here is an interleaved DRAM page plus an intra-page offset, which only `TensorAccessor` resolves. |
| `noc_inline_dw_write` | `dataflow_api.h:2224`; restriction `noc.h:655` | "does not support DRAM addresses". |
| `compute_kernel_lib::FillBitcast` (eltwise chain fill) | `ttnn/cpp/ttnn/kernel_lib/eltwise/generators/fill.inl:37` | A compute-kernel DEST fill. It needs a TRISC kernel, a CB to the writer, and whole-tile packs. See the `compute_fill` rejected regime. |
| `prepare_reduce_scaler` / `calculate_and_prepare_reduce_scaler` | `ttnn/cpp/ttnn/kernel_lib/reduce_helpers_dataflow.hpp:29-34` | Packs a bf16 scaler tile pattern: 16-bit, and the reduce-scaler layout. Violates the 32-bit rule. |
| `mcast` (`McastArgs`) | `ttnn/cpp/ttnn/kernel_lib/mcast/` | There is nothing to multicast: `e` arrives by common RT arg (`injector_mcast_fill_row` rejected). |

No helper covers "write one L1 source to a list of (page, offset) destinations of an interleaved tensor". The loop over a core's segments lives inside `store_block`, with one accessor and no per-segment barrier. That is the block operation; the gap is recorded here.

## Key Risks and Gotchas

| Risk | Why it bites here | Mitigation in this design |
|------|-------------------|---------------------------|
| Double rounding of an `int` fill for float32 | `float(v)` rounds int → double, and `pack("<f")` rounds again. `2**60 + 2**36 + 1` → `0x5d800000`, while the correct (and oracle) value is `0x5d800001`. | Exact integer RNE path (Host conversion table). The acceptance test pins this value. |
| A cached program replays the old value | Rule 3 / BB-07: a value baked into CT args, defines, or the cache key replays the first `e`, or rebuilds per call. | `e_bits` is a **common RT arg** only. The descriptor cache key excludes it. The repeat test changes the value between calls into the same output. |
| 16-bit route for the element | Rule 2: 1.0 and 0 survive bf16, but `1/3`, `1e-40` and NaN payloads do not. | No TRISC, no packer, no bf16 helper. The element travels as raw 32-bit words from the RT arg to L1 to DRAM. The value probes are in the acceptance test. |
| Face-1 positions of a tile | Columns 16–31 of row 0 live at byte 1024, not at byte 64. | Segment geometry derived from `tile.face_shape`. [185] and [512] exercise face 1. |
| Stale source on repeat calls | `cb_fill_row` sits at the same L1 address on every call. If the NoC read the source before the RISC stores landed, the previous call's value would ship. | `fill_l1_range` stores through a `volatile tt_l1_ptr` (`l1_helpers.hpp:30`), and the NoC command is issued after the stores in program order. The repeat test ([512]: 1.0, 1.0, 0.1, 1.0, -2.5) and the golden `new_value` loose case detect a race. If it ever shows, add a store fence before the first write. |
| Kernel returns with writes in flight | BB-03: the next program starts right after. Posted writes are not acked. | Non-posted writes plus one `noc_async_write_barrier()` before return. Posted writes are not used. |
| Descriptor attribute copies | nanobind may return a copy of `kernels` / `common_runtime_args`, so an in-place edit is silently lost and the cached value is replayed. | Re-assign `common_runtime_args` on the held `KernelDescriptor`, then re-assign `descriptor.kernels`. The repeat-value test verifies it. |
| Refused call writes the output | Rule 7: after a refusal, `output_tensor` must be unchanged. | All checks run in `validate()` before any dispatch. The acceptance test reads the output back after each refusal. |
| ROW_MAJOR refinement: 4-byte DRAM write | `[1]` int32 is one 4 B page. Blackhole NoC write alignment is 16 B (`noc_parameters.h:380`). | Source and destination share the 16 B alignment (page base aligned, `cb_fill_row` L1-aligned). If a sub-16 B DRAM write misbehaves, write the page's aligned size (`buffer_aligned_page_size()`) from a source of that size. The extra bytes are page alignment padding inside `output_tensor`'s own allocation. |
