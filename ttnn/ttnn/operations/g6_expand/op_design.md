# Operation Design: g6_expand

## Overview

| Field | Value |
|-------|-------|
| Classification | data_movement — no compute kernel; every output byte is a byte copy of an input byte |
| Goal | Materialize `x (d_0, d_1, 1, W)` broadcast along dim 2 to `(d_0, d_1, H_out, W)` as a new DRAM-interleaved TILE tensor, bit-identical, in one device program. Production: `x (1, 1, 1, W) -> (1, 1, W, W)`, `W ∈ {512, 256, 196, 185, 207}`, bfloat16, TILE, DRAM interleaved. |
| Math | `out[n, c, h, w] = x[n, c, 0, w]`. In tile terms: `out_tile(img, r, wt) = E(img, wt)` for every out tile row `r`, where `E(img, wt)` is x tile `(img, wt)` with its row 0 copied into all 32 rows. |
| References | Spec (task prompt); `eval/golden_tests/g6_expand/{feature_spec.py, helpers.py}`; `.claude/references/{blocking-model.md, l1-footprint-discipline.md, op-design-template.md}`; `ttnn/cpp/ttnn/kernel_lib/mcast/` (`host/mcast.hpp`, `kernel/mcast_args.hpp`, `kernel/pipes/mcast_pipe.hpp`); `ttnn/cpp/ttnn/kernel_lib/index_tile_dataflow.hpp` (row-replicated tile construction); `ttnn/ttnn/operations/toy_spec_mcast/toy_spec_mcast.py` (host Mcast wiring on a ProgramSpec); catalog `ttnn/ttnn/operations/examples/{mcast_topology, shared_input_reuse, noc_placement, double_buffer, tile_reorder, tensix_all_reduce, split_reader, bank_stagger, trid_double_issue}` |

### Tile facts the design relies on

| Fact | Value | Source |
|------|-------|--------|
| Tile / faces | 32×32 tile, 16×16 faces, face-major: face0 = rows 0–15 × cols 0–15, face1 = rows 0–15 × cols 16–31, face2 = rows 16–31 × cols 0–15, face3 = rows 16–31 × cols 16–31; row-major inside a face | device probe (`x.tile.tile_shape == [32, 32]`, `x.tile.face_shape == [16, 16]`); `kernel_lib/tile_column_dataflow.hpp:9-16` |
| bf16 sizes | page 2048 B, face 512 B, face row 32 B | device probe (`buffer_page_size() == 2048`) |
| Row 0 of a tile | bytes `[0, 32)` (face0 row 0) followed by `[512, 544)` (face1 row 0) | face layout above |
| Expanded tile `E` | face0 rows 0–15 = bytes `[0, 32)`; face1 rows 0–15 = bytes `[512, 544)`; bytes `[1024, 2048)` = bytes `[0, 1024)` | face layout above |
| x storage | padded `(d_0, d_1, 32, ceil(W/32)·32)`; page id = `img · tensor_w_tiles + wt`; rows 1–31 and columns `≥ W` are padding with arbitrary content | device probe (`padded_shape == [1, 1, 32, 512]`) |
| Device (this box) | Blackhole; compute grid 11 × 10 = 110 cores; 8 DRAM banks; DRAM alignment 64 B; L1 alignment 16 B | device probe |

## Parameters

| Name | Type | Required | Valid Range | Default | CT/RT |
|------|------|----------|-------------|---------|-------|
| `input_tensor` | `ttnn.Tensor` | yes, positional | see Input | — | buffer address: per call (bound, never baked into a cached program) |
| `sizes` | `list` or `tuple` of 4 `int` | yes, positional | length 4; each entry `-1` or `≥ 1`; equal to `d` (or `-1`) on every dim where `d ≠ 1`; differs from `d` only on dim 2 | — | host only; the resolved output shape enters the plan and the cache key |

No `memory_config` (out takes x's placement), no output dtype (out takes x's dtype), no `compute_kernel_config` (the op computes nothing).

### validate() — the entry point's first line

| Step | Check | Raises |
|------|-------|--------|
| 1 | `len(input_tensor.shape) != 4` or `len(sizes) != len(input_tensor.shape)` | `ValueError` |
| 2 | any `s` in sizes with `s < -1` or `s == 0` | `ValueError` |
| 3 | any dim `k` with `d_k ≠ 1` and `sizes[k] ∉ {-1, d_k}` | `ValueError` |
| 4 | any dim `k ≠ 2` with `d_k == 1` and `sizes[k] ∉ {-1, 1}` (expansion of a dim other than 2) | `ValueError` |
| 5 | `d_2 ≠ 1` (Phase 0 input requirement: x has one row per image) | `ValueError` |
| 6 | input not on a device | `ValueError` |
| 7 | memory config not `(BufferType.DRAM, TensorMemoryLayout.INTERLEAVED)` | `ValueError` |
| 8 | TILE layout with tile ≠ 32×32 or face ≠ 16×16 (mechanism cap of `expand_rows_block`) | `ValueError` |
| 9 | SUPPORTED per axis: `dtype`, `layout` from `input_tensor`; `alignment` from `INPUT_TAGGERS["alignment"]` on `({"shape": list(input_tensor.shape), "sizes": list(sizes)},)` | `UnsupportedAxisValue` |
| 10 | EXCLUSIONS (empty in Phase 0) | `ExcludedCell` |

`tag_alignment(inputs, axes)`: `scenario = inputs[0]`; `out_shape = [d if s == -1 else s for d, s in zip(scenario["shape"], scenario["sizes"])]`; return `"w_non_aligned"` if `out_shape[-1] % 32`, else `"h_non_aligned"` if `out_shape[-2] % 32`, else `"tile_aligned"`. Axis names and values are TARGET's (`eval/golden_tests/g6_expand/feature_spec.py`). Phase 0 SUPPORTED per the spec: `dtype [bfloat16]`, `layout [TILE_LAYOUT]`, `alignment ["tile_aligned", "w_non_aligned"]`; EXCLUSIONS `[]`. Structural impossibilities to add to INVALID: none.

## Tensors

### Input

| Property | Requirement |
|----------|-------------|
| Shape | `(d_0, d_1, 1, W)`, rank 4; production `(1, 1, 1, W)` |
| Dtype | bfloat16 |
| Layout | TILE (32×32, faces 16×16) |
| Memory | DRAM interleaved |
| Padding | rows 1–31 of every tile and columns `≥ W` hold arbitrary bytes; no logical element of out may depend on them |

### Output

| Property | Value |
|----------|-------|
| Shape | `sizes` with each `-1` replaced by x's size on that dim: `(d_0, d_1, H_out, W)`; production `(1, 1, W, W)` |
| Dtype | x's dtype |
| Layout | x's layout (TILE) |
| Memory | x's memory config (DRAM interleaved); a fresh allocation per call, never a view of x |
| Padding | rows `≥ H_out mod 32` of the last tile row and columns `≥ W mod 32` of the last tile column hold copies of x's row-0 bytes (including x's padding columns); not part of the result |

## Blocking Model

The op is a pure broadcast copy with no dependent axis. Each output tile is one of `tensor_src_col_tiles` distinct *expanded tiles* `E`, written `tensor_out_row_tiles` times. A block is a rectangle of `block_src_col_tiles` x pages × `block_out_row_tiles` out tile rows. The x pages of a column group are read once from DRAM on the group's sender, which expands them and multicasts them to the rest of the group. Every core of the group then writes its own out rows of those columns.

### Axes

| Axis | Character (+ one-clause reason) | Extent knob | Phase 0 value | Knob source | Core-assignment | Later unlock |
|------|--------------------------------|-------------|---------------|-------------|-----------------|--------------|
| `image` (dims 0·1, `num_images = d_0·d_1`) | independent — out image `(n, c)` reads only x image `(n, c)` | folded into `src_col`: `block_src_col_tiles` counts x pages in page order (`page = img·tensor_w_tiles + wt`), so a src block may span an image boundary | 1 image in production (folded; no separate extent) | `select_plan` (host), from `input_tensor.shape` | with the columns: column groups own contiguous x page ranges | knob-turn |
| `tile_col` (dim 3; `tensor_w_tiles = ceil(W/32)` per image) — with `image` forms `src_col`, `tensor_src_col_tiles = num_images · tensor_w_tiles` | independent — out tile column `wt` depends only on x tile `wt` (and x varies only along this axis and `image`) | `block_src_col_tiles` (x pages per block) | `= core_src_col_tiles = ceil(S / min(S, grid_y))` → 2 (W=512), 1 (W=256, 196, 207, 185): the whole core share in one block | `select_plan`, CT arg | `num_col_groups = ceil(S / core_src_col_tiles)` groups along grid **y**; group `y` owns x pages `[y·core_src_col_tiles, min(S, (y+1)·core_src_col_tiles))` → 8, 8, 7, 7, 6 groups | knob-turn |
| `out_row` (dim 2; `tensor_out_row_tiles = ceil(H_out/32)` per image) | independent, and **x is reuse-shared along it** — x does not vary with `h`, so every tile row of out is identical; cutting this axis across cores makes every x page needed by every core of a column group | `block_out_row_tiles` (out tile rows written per block) | `= core_out_row_tiles = ceil(Hc / min(Hc, grid_x))` → 2 (W=512), 1 (others); whole share; costs no L1 (E is re-read from L1 for every row) | `select_plan`, RT args (row start, row count) | `num_row_groups = ceil(Hc / core_out_row_tiles)` cores along grid **x** per column group → 8, 8, 7, 7, 6 | knob-turn for the count; the sharing it needs (group multicast) is built in Phase 0 |
| `row_in_tile` (32 rows of an E tile — data the `expand_rows_block` stage produces) | reuse-shared inside the tile — all 32 rows equal row 0 | not a block axis: transfers move whole 2 KB tiles | 32 rows (whole tile) | tile geometry (CT args from `input_tensor.tile`) | not assigned across cores. The group's sender is its *communication* role. It also builds E because every core of the group needs every E tile of the block, so splitting the build would need an all-gather. Building on every core instead (L2-b) only moves the same build time onto each core's critical path. The sender also writes its own share of out (it is a full worker, not an idle root). | n/a |

`S = tensor_src_col_tiles`, `Hc = tensor_out_row_tiles`; `grid_x × grid_y = device.compute_with_storage_grid_size()`.

### Buffer-depth knobs

| CB | Depth knob | Phase 0 value | What the depth buys |
|----|------------|---------------|---------------------|
| `cb_expanded_tiles` | `src_block_depth` | 1. Every production core holds its whole src share as one block (`num_src_blocks_this_core == 1`), so there is no next block to overlap. | `select_plan` sets 2 when `num_src_blocks_this_core > 1` (the L1 clamp below engaged). The sender's next-block x read, expansion and multicast then overlap the current block's writes on every core of the group. |

### Mechanism caps

| Mechanism | Cap on which extent | Clamp | What happens unclamped |
|-----------|--------------------|-------|------------------------|
| Face geometry used by `expand_rows_block` (face row 32 B, face 512 B, 2 × 2 faces) | `row_in_tile`; byte offsets of every replicated row | validate step 8 refuses tile ≠ 32×32 / face ≠ 16×16; face constants are CT args derived from `input_tensor.tile`, never literals | rows land at wrong offsets: logical rows of out carry x padding bytes; silent wrong values |
| `seed_rows` (fan-out source rows per face) | rows replicated per self-read | `seed_rows ∈ {1, 2, 4, 8}` (must divide the face height 16); host asserts | a non-divisor leaves face rows unwritten: x padding bytes reach logical elements of out |
| Op L1 budget, 1347776 B per core (BB-05) | `block_src_col_tiles` | `block_src_col_tiles = min(core_src_col_tiles, floor(1347776 / (src_block_depth · page_bytes)))`; excess becomes `num_src_blocks_this_core > 1` (streams over `src_col`), never a refusal | launch fails when model tensors occupy the rest of L1 |
| Mcast receiver groups: equal consecutive groups (`kernel_lib/mcast/host/mcast.cpp:41-46`) | `num_row_groups × num_col_groups` | uniform extents (`ceil`), so every group has exactly `num_row_groups` cores; the core set is the full rectangle `x ∈ [0, num_row_groups)`, `y ∈ [0, num_col_groups)` | host `TT_FATAL` |
| Device grid | `num_col_groups ≤ grid_y`, `num_row_groups ≤ grid_x` | `select_plan` reads the grid from the device; the grid size is in the cache key | rectangle off-grid |
| NoC L1 alignment (16 B) for self-reads | replication source/destination offsets | all offsets are multiples of the 32 B face row | corrupted copy |
| DRAM alignment (64 B) | x read granularity | Phase 0 reads whole pages from the page base | n/a in Phase 0; lamp L5's row-only read must use 64 B chunks at page offsets 0 and 512 |
| CB ring cycle and cross-core landing address | `cb_expanded_tiles` push quantum | every push/pop is the nominal `block_src_col_tiles`, also on a ragged last block; capacity = `src_block_depth · block_src_col_tiles`. Sender and receivers advance their write pointers in lockstep, so the sender's write pointer is every receiver's landing address. | sender and receiver slots diverge: the multicast lands in a slot the writer is still reading |

### Regimes

| Regime | Status | Predicate | Block | Data movement vs. minimum | What a bigger block buys |
|--------|--------|-----------|-------|---------------------------|--------------------------|
| `column_group_broadcast` | **built** | every call that passes `validate()` (domain: all of SUPPORTED, every production width). `num_row_groups == 1` is the same regime with the channel absent (column owners). | `block_src_col_tiles × block_out_row_tiles` out tiles per core per block; production: one block per core, 2×2 (W=512), 1×1 (others) | Minimum against DRAM: each x page crosses once (S page reads by the group senders) and each out page crosses once. Above minimum only on NoC: per group per block, one multicast of `block_src_col_tiles × 2 KB` to `num_row_groups − 1` receivers, plus `num_row_groups − 1` ready increments and one flag. Local: ≈1.9 KB of L1→L1 self-reads per E tile, on the sender only. | Per block: one DRAM read barrier, one RISC-V store fence, two self-read barriers, one multicast round (ack wait + payload + flag), one CB reserve/push/wait/pop and one write flush. Intended: once per block, which in production is once per call per core. A larger src block also means fewer groups, so fewer multicast rounds per call. |
| `per_core_dram_read` | **rejected** — superseded by `column_group_broadcast`. It is that regime's stepping stone (same 2-D split, with every core reading its own x pages instead of the group multicast); the successor is built directly because RB-010 applies from Phase 0. | never selected | same | x crosses DRAM `num_row_groups` times. W=512: 128 page reads = 256 KB, half of out's bytes, as 2 KB requests on the 8 banks, queued with out's writes. | same per-block costs without the multicast round, plus `num_row_groups ×` the DRAM read requests |
| `flat_tile_range` | **rejected** — dead end: a 1-D split of out's tile ids where each core reads the x page of every out tile it writes; no good scheme passes through it | never selected | contiguous tile-id range | x crosses DRAM ≈ `Hc` times (≈ one 2 KB read per out tile; W=512 ≈ 512 KB of reads = out's size) | nothing: the per-output-tile read is independent of the range length |
| `global_injector_broadcast` | **rejected** — superseded by `column_group_broadcast`: one core reads all S pages and multicasts every E tile to the whole rectangle | never selected | all S pages per core | x once, but read serially on one core. NoC: every core receives all S E tiles although it writes only its group's columns (W=512: 32 KB × 63 receivers ≈ 2 MB delivered vs 224 KB). | one handshake per call, against one serial injector; column groups split the payload by `num_col_groups` and run `num_col_groups` injectors in parallel |
| `fpu_row_broadcast` | **rejected** — superseded by the byte-exact dataflow expansion. A compute-kernel row broadcast (unpack with row broadcast → DEST → pack) is a value path, not a byte copy: preserving NaN payloads, denormals and signed zeros is not guaranteed, and the spec forbids "a representation that changes it". It also adds a compute kernel and two L1 passes per E tile. | never selected | same | DRAM as `column_group_broadcast`; adds unpack + pack L1 traffic per E tile | adds compute init/reconfig per block |

There is no `split dependent axis + combine` regime: no axis of this op is dependent (see Traffic ranking).

### Traffic ranking

Counted for W=512 (`S = Hc = 16`, 2 KB pages, 11×10 grid). The other widths scale the same way.

| Candidate split | x DRAM crossings | out DRAM crossings | Cross-core traffic | Writer cores | Outcome |
|-----------------|------------------|--------------------|--------------------|--------------|---------|
| A0: cut `src_col` only (`num_row_groups = 1`, column owners) | 1 (16 pages, 32 KB) | 1 (256 pages, 512 KB) | 0 | `num_col_groups ≤ S` (8 here; 6–8 across production) | cheapest traffic; reachable as a knob value of the built regime (lamp L1) |
| A: cut `src_col × out_row`, x shared by group multicast | 1 (32 KB) | 1 | 8 groups × 4 KB payload to 7 receivers (224 KB delivered), 56 ready increments, 8 flags | 64 | **chosen** (Phase 0 default) |
| B: cut `src_col × out_row`, every core reads its own x pages | 8 (256 KB) | 1 | 0 | 64 | rejected (`per_core_dram_read`) |
| C: flat 1-D tile-id split | ≈16 (≈512 KB) | 1 | 0 | 110 | dead end (`flat_tile_range`) |
| D: one injector multicasts all of x to the whole rectangle | 1 | 1 | ≈2 MB delivered | 64 | rejected (`global_injector_broadcast`) |
| E: split a dependent axis + cross-core combine | — | — | — | — | does not exist: no reduction or scan; every out tile is a copy of one x tile's row 0 |

DRAM crossings (the binding term) tie A0 and A at the minimum, and only A0 moves zero cross-core bytes. Phase 0 takes A, at full-grid `num_row_groups`. This trades NoC bytes for occupancy (`blocking-model.md` §4), for two reasons:

1. The spec requires the core count to be the blocking decision's choice, not a limit set by the tensor's shape. A0 is capped at the `S` column owners (6–16 cores).
2. The op is DRAM-write-bound (512 KB written vs 32 KB read). A0 sends all of a W=512 core's writes to one or two banks from ≤ 16 injection points.

The cost of the trade is one ≤ 4 KB multicast round per column group, paid in parallel across groups. Lamp L1 decides the trade by measurement. A0 needs no structural change: set `num_row_groups = 1` and attach the channel as absent.

Orientation: column groups run along grid y (≤ 10) and row groups along grid x (≤ 11). Each group is a horizontal line `y = const`, with its sender at `x = 0` and its receivers at `x ∈ [1, num_row_groups)` — one 1-D multicast per line on NoC0 (catalog `mcast_topology`: broadcast along the axis the operand does not vary with).

Stall shadow: receivers wait for their sender's x read, expansion and multicast. Every out byte is an x byte, so no work is independent of that wait. The wait is minimized by making the x read the sender's first action and the ready-ack the receiver's first action. With `src_block_depth = 2` (multi-block cores only), the next block's read and expansion run inside the current block's write window.

### Block schedule

Sender core (`x = 0` of each group row):

```cpp
for (uint32_t block_idx = 0; block_idx < num_src_blocks_this_core; ++block_idx) {
    load_source_block(block_idx);   // reader, NoC0
    expand_rows_block(block_idx);   // reader: RISC-V seed + NoC0 self-reads, in place
    publish_block(block_idx);       // reader: hand the block to the writer, then broadcast it to the group
    store_block(block_idx);         // writer, NoC1
}
```

Receiver core (`x ∈ [1, num_row_groups)`):

```cpp
for (uint32_t block_idx = 0; block_idx < num_src_blocks_this_core; ++block_idx) {
    receive_block(block_idx);       // reader: ready-ack, wait for the flag, hand to writer
    store_block(block_idx);         // writer, NoC1
}
```

| Operation | Kernel / NoC | Block shape | Resident across it | Intended fixed-cost frequency |
|-----------|--------------|-------------|--------------------|-------------------------------|
| `load_source_block` | reader on senders / NoC0 | `extent = min(block_src_col_tiles, remaining)` x pages, each a whole 2 KB page, into consecutive `cb_expanded_tiles` slots | — | 1 `cb_reserve_back(block_src_col_tiles)`; `extent` async page reads issued back to back; 1 read barrier per block |
| `expand_rows_block` | reader on senders / RISC-V + NoC0 self-reads | the same `extent` pages, transformed in place into E | the block's pages | per block: seed copies for all pages, **1** store→NoC fence (`load_blocking`), fan-out self-reads for all pages, **1** barrier, lower-face self-reads for all pages, **1** barrier — never per tile. Reads only row 0 of faces 0 and 1. |
| `publish_block` (includes the broadcast) | reader on senders / NoC0 | `block_src_col_tiles × page_bytes` contiguous bytes; multicast source address = destination address = this block's slot base | the E block | 1 `cb_push_back(block_src_col_tiles)` **before** the broadcast, so the sender's own writes overlap the multicast (both only read E); then 1 `send` per block: wait for `num_row_groups − 1` acks, 1 payload multicast, 1 flag multicast. Compiled out when the channel is absent (`num_row_groups == 1`). |
| `receive_block` | reader on receivers / NoC0 | same bytes, landing at the same slot address | — | per block: 1 `cb_reserve_back(block_src_col_tiles)`, then `receive(block_idx)` (1 ready increment + 1 flag wait), then 1 `cb_push_back(block_src_col_tiles)` |
| `store_block` | writer on every core / NoC1 | `block_out_row_tiles × extent` out pages, each a whole 2 KB page sourced from E slot `s` | the E block (read-only, re-read for every out row) | 1 `cb_wait_front(block_src_col_tiles)`; all page writes issued back to back, `out_row` outer and `src_col` inner (consecutive out pages → banks round-robin); 1 writes-flushed per block before `cb_pop_front`; 1 write barrier at kernel end |

Kernel exit (BB-03): every reader ends with a write barrier and an atomic barrier (multicast acks; receivers' ready increments); every writer ends with a write barrier.

### Perf lamps

| Lamp | Why the default may be wrong here | Nearby alternative to measure |
|------|-----------------------------------|-------------------------------|
| L1 grid synchronization | At W ≤ 256 each core writes 1 tile; the multicast round (ack + payload + flag) may cost more than the extra writers buy, because the op is DRAM-bound | `num_row_groups ∈ {1, 2, 4, full}` per production width; `1` = column owners, channel absent |
| L2 overlap | Receivers wait for x read + expansion + multicast; the sender's own writes start after its expansion | (a) `publish_block` order: push-then-send (Phase 0) vs send-then-push; (b) broadcast the raw x pages (or only their 64 row-0 bytes) and run `expand_rows_block` on every core |
| L3 replication realization | `seed_rows = 2` is `index_tile_dataflow.hpp:104-113`'s measured optimum for RISC-V-generated seeds; here row 0 already sits in L1, so `seed_rows = 1` needs no stores | `seed_rows ∈ {1, 2, 4}`; fan the seed straight into the lower faces (one barrier round instead of two); all-RISC-V stores |
| L4 write injection | each core writes from one RISC on NoC1 while its reader idles after `receive_block` | split `store_block` across both RISCs (reader takes half the out rows on NoC0); only if the writer is measured issue-bound (catalog `split_reader`) |
| L5 x read size | the sender reads the whole 2 KB page and uses 64 B of it | read two 64 B chunks per page (page offsets 0 and 512, 64-B aligned) |
| L6 orientation | column groups along y, row groups along x | transposed rectangle (column groups along x, multicast down columns) |

### Catalog evidence

| Entry | What it set here |
|-------|------------------|
| `mcast_topology` | broadcast x along the grid axis it does not vary with: one 1-D multicast per column-group line; the device-time win is smaller than the read-count reduction, hence lamp L1 |
| `shared_input_reuse` | read once on an injector and multicast; double-buffer only when there is more than one block (`src_block_depth`) |
| `noc_placement` | reads (and the multicast) on NoC0, writes on NoC1 — the reader/writer config defaults |
| `double_buffer` | issue a block of async reads/writes, then one barrier; no depth gain once DRAM-bound → `src_block_depth = 1` |
| `tile_reorder` | whole-page (2 KB) writes, no sub-tile writes → a full E tile is materialized instead of writing two 1 KB halves |
| `tensix_all_reduce` | the root (sender) also takes a full share of the writes; uniform push quanta on every core |
| `split_reader` | lamp L4 applies only if a data-movement RISC is measured issue-bound |
| `trid_double_issue` | not used: a sender issues ≤ 2 reads per call in production |
| `bank_stagger` | not used: for W=512, `page mod 8 = wt mod 8`, so a core's banks are fixed by its columns regardless of start row |

## Dataflow Strategy

| Stage | Format | Mechanism | Notes |
|-------|--------|-----------|-------|
| x DRAM → sender L1 | bf16 tile page, raw bytes | `Noc::async_read` through a `TensorAccessor` (NoC0) into the `cb_expanded_tiles` slot | senders only; each x page is read exactly once per call |
| row-0 replication | same bytes, in place | RISC-V word copy (seed rows) → `load_blocking` fence → NoC0 self-reads (fan-out) → barrier → 1 KB self-read (lower faces) → barrier | reads only bytes `[0, 32)` and `[512, 544)` of each page; x padding rows are overwritten, never read |
| sender L1 → group receivers' L1 | E pages | `ttnn.Mcast` channel: receivers = the rectangle, `receiver_group_size = num_row_groups`, `McastCoreOrder.RowMajor`, `McastFixedSenderConfig(sender_index=0)`, `McastConfig(noc=NOC_0, handshake=True, data_ready=Flag)` | landing address = the sender's slot address (same `cb_expanded_tiles` slot on every core); channel absent when `num_row_groups == 1` |
| reader → writer (every core) | E pages | `cb_expanded_tiles`, push/wait `block_src_col_tiles` | one producer (reader), one consumer (writer) |
| L1 → out DRAM | E pages | `Noc::async_write` through a `TensorAccessor` (NoC1), whole 2 KB pages | every out page written exactly once, by exactly one core |

### Inter-Tensix contract (built)

| Item | Value |
|------|-------|
| Topology | one 1-D multicast per column group: sender `(0, y)` → receivers `(1..num_row_groups−1, y)` |
| Handshake | pre-handshake: each receiver increments the sender's consumer-ready semaphore after reserving its slot; the sender waits for `num_row_groups − 1` acks, multicasts the payload, then multicasts the data-ready flag; the receiver waits for the flag and clears it (`mcast_pipe.inl`, `receive` / `send`) |
| Rounds | one per block (`round = block_idx`); production: one per call |
| Semaphores | allocated by `Mcast.attach`; initialized by dispatch for every launch (no in-kernel reset needed across calls) |

### TARGET connection

| TARGET axis | Values | How the design covers them |
|-------------|--------|----------------------------|
| `dtype` | bfloat16 | page size and CB format derived from `input_tensor`; the op moves bytes, so the only dtype dependence is `page_bytes` and the face-row byte width |
| `layout` | TILE | face geometry CT args from `input_tensor.tile` |
| `alignment` | `tile_aligned`, `w_non_aligned` | same regime, no algorithmic change: tile counts are `ceil` per image; the last tile column's E carries x's padding columns into out's padding; the last tile row is written whole and its padding rows are don't-care |

TARGET has no placement axis: out takes x's DRAM-interleaved placement, so no sharded variant is planned.

## Work Distribution

| Field | Value |
|-------|-------|
| Work unit | a block: `extent ≤ block_src_col_tiles` x pages × `block_out_row_tiles` out tile rows |
| Grid | rectangle `x ∈ [0, num_row_groups)`, `y ∈ [0, num_col_groups)` on `device.compute_with_storage_grid_size()` |
| Per-core work | core `(x, y)`: x pages `[y·core_src_col_tiles, min(S, (y+1)·core_src_col_tiles))`; out tile rows (per image) `[x·core_out_row_tiles, min(Hc, (x+1)·core_out_row_tiles))` |
| Remainder | `ceil` on both axes; only the last group/row range is ragged; tile counts are per image: `tensor_w_tiles = ceil(W/32)`, `Hc = ceil(H_out/32)` — never `floor` or `//` |
| Roles | senders = `CoreRange((0, 0), (0, num_col_groups−1))`; receivers = `CoreRange((1, 0), (num_row_groups−1, num_col_groups−1))` (absent when `num_row_groups == 1`); writer = the whole rectangle. Three core ranges (BB-17). |
| Out page of `(s, r)` | `(s // tensor_w_tiles) · Hc · tensor_w_tiles + r · tensor_w_tiles + (s % tensor_w_tiles)` |

### `select_plan` — the single source of every knob

```text
page_bytes            = input_tensor.buffer_page_size()                 # 2048 for bf16 32x32
tensor_w_tiles        = ceil(W / 32)                                      # per image
S                     = d_0 * d_1 * tensor_w_tiles                        # tensor_src_col_tiles == x pages
Hc                    = ceil(H_out / 32)                                  # tensor_out_row_tiles, per image
grid_x, grid_y        = device.compute_with_storage_grid_size()
max_col_groups        = grid_y                                            # core-assignment knob
max_row_groups        = grid_x                                            # core-assignment knob (lamp L1)
core_src_col_tiles    = ceil(S / min(S, max_col_groups))
num_col_groups        = ceil(S / core_src_col_tiles)
core_out_row_tiles    = ceil(Hc / min(Hc, max_row_groups))
num_row_groups        = ceil(Hc / core_out_row_tiles)
src_block_depth       = 1 if core_src_col_tiles * page_bytes <= L1_OP_BUDGET else 2
block_src_col_tiles   = min(core_src_col_tiles, floor(L1_OP_BUDGET / (src_block_depth * page_bytes)))
block_out_row_tiles   = core_out_row_tiles
seed_rows             = 2
L1_OP_BUDGET          = 1347776
```

Every CB size, CT arg, RT arg and core range is derived from these values; none is restated as a literal.

### Production plans (11×10 grid)

| W | alignment | S = Hc | core_src_col_tiles | num_col_groups | core_out_row_tiles | num_row_groups | cores | out tiles / core | CB bytes / core |
|---|-----------|--------|--------------------|----------------|--------------------|----------------|-------|------------------|-----------------|
| 512 | tile_aligned | 16 | 2 | 8 | 2 | 8 | 64 | 4 | 4096 |
| 256 | tile_aligned | 8 | 1 | 8 | 1 | 8 | 64 | 1 | 2048 |
| 196 | w_non_aligned | 7 | 1 | 7 | 1 | 7 | 49 | 1 | 2048 |
| 207 | w_non_aligned | 7 | 1 | 7 | 1 | 7 | 49 | 1 | 2048 |
| 185 | w_non_aligned | 6 | 1 | 6 | 1 | 6 | 36 | 1 | 2048 |

### Host contract

| Rule | Requirement in this design |
|------|----------------------------|
| BB-01 | one `ttnn.generic_op` dispatch per call; validation, `select_plan` and output allocation are host work |
| BB-07 | build the program artifacts (descriptor, or ProgramSpec + run args, plus the Mcast attachment) once per key and keep them in a module-level dict. Key: `(tuple(input_tensor.padded_shape), resolved output shape, dtype, layout, memory config, device id, grid size)`, which covers every input of `select_plan`. A hit only binds the input and output buffer addresses. With a `ProgramDescriptor`, carry the two addresses as common runtime args so a hit patches two values; with a `ProgramSpec`, tensors are bound by name and the cached spec/run args are reused unchanged (`toy_spec_mcast.py` shows `Mcast.attach` on a spec). |
| BB-08 | export `g6_expand_plan(input_tensor, sizes) -> dict` from `ttnn.operations.g6_expand`. It runs the same `select_plan` the entry point uses and returns keys `regime` (`"column_group_broadcast"`), `num_col_groups`, `num_row_groups`, `core_src_col_tiles`, `core_out_row_tiles`, `block_src_col_tiles`, `block_out_row_tiles`, `src_block_depth`, `seed_rows`, `num_cores`, `l1_bytes_per_core`, `core_roles` (`{(x, y): "sender" | "receiver"}`) |
| BB-16 | the only device allocation per call is the output; no kernel build after the first call per key |
| BB-21 | `DeviceZoneScopedN` zones: sender reader `load_source_block`, `expand_rows_block`, `publish_block`; receiver reader `receive_block`; writer `wait_block`, `store_block`, `write_drain`. Zones compile to nothing in the plain build. The role dump is `g6_expand_plan(...)["core_roles"]`. |

Only one regime is built, so the regime-pinned test is: `g6_expand_plan` reports `column_group_broadcast` for every production width, and its geometry invariants hold on the running device's grid.

## Circular Buffers

| Semantic Name | Index | Page Size | Num Pages | Sizing rationale | Format | Producer | Consumer | Lifetime |
|---------------|-------|-----------|-----------|------------------|--------|----------|----------|----------|
| `cb_expanded_tiles` | 0 | `page_bytes = input_tensor.buffer_page_size()` (2048) | `src_block_depth · block_src_col_tiles` (production: 2 at W=512, 1 otherwise) | spans `src_col` (`block_src_col_tiles`); streams `out_row` (each page is re-read for every out row; no footprint); `image` folded into `src_col`; `row_in_tile` whole tile | input dtype (bf16 = `Float16_b`); a dataflow-only payload that never passes through DEST | reader | writer | whole call (one block in production) |

The CB is declared on the whole rectangle, so it sits at the same L1 address on every core. That is the multicast landing contract. The sender's x pages land in, and are expanded inside, this CB. No other CB exists: no x staging buffer (Rule 3.1, pack into the destination), no separate E buffer (Rule 3.2, transform in place).

## Block Operation Realization

| # | Block operation | Block shape | Helper? | Input CB (semantic name, pages, state) | Output CB (semantic name, pages) | CB state after |
|---|-----------------|-------------|---------|----------------------------------------|----------------------------------|----------------|
| 1 | `load_source_block` | `extent` x pages | raw Noc API | — (x in DRAM) | `cb_expanded_tiles`, `block_src_col_tiles` reserved, `extent` filled | reserved, not pushed; raw x pages in slots |
| 2 | `expand_rows_block` | `extent` pages, in place | custom block op on the `index_tile_dataflow.hpp` construction | `cb_expanded_tiles`, reserved slots holding raw x pages | same slots | reserved, not pushed; slots hold E |
| 3 | `publish_block` | `block_src_col_tiles · page_bytes` bytes | Mcast helper (`SenderPipe::send`) + CB push | `cb_expanded_tiles` slots (E) | `cb_expanded_tiles`, `block_src_col_tiles` pushed | pushed to writer; same bytes delivered to every receiver's slot at the same address |
| 4 | `receive_block` | same bytes | Mcast helper (`ReceiverPipe::receive`) + CB reserve/push | — | `cb_expanded_tiles`, `block_src_col_tiles` pushed | pushed to writer |
| 5 | `store_block` | `block_out_row_tiles × extent` out pages | raw Noc API | `cb_expanded_tiles`, `block_src_col_tiles` waited (E) | — (out in DRAM) | popped after writes are flushed |

## API Mapping

| Block operation | Type | Function | File:Line | Template Params / Args | Input CB | Output CB | Which params are block knobs |
|-----------------|------|----------|-----------|------------------------|----------|-----------|------------------------------|
| `load_source_block` | raw_api | `Noc::async_read(TensorAccessor, CircularBuffer, size, {.page_id}, {.offset_bytes})`, `Noc::async_read_barrier()` | `tt_metal/hw/inc/api/dataflow/noc.h:252`, `:739`; usage `ttnn/ttnn/operations/examples/mcast_topology/mcast_topology.py:163-170` | `size = page_bytes`; `page_id = src_start + block_idx·block_src_col_tiles + i`; `offset_bytes = i·page_bytes` | — | `cb_expanded_tiles` | `block_src_col_tiles` (read count per barrier) |
| `expand_rows_block` | raw_api | seed: RISC-V `uint32` copy of row 0 into rows `1..seed_rows−1` of faces 0, 1; fence: `ckernel::load_blocking`; fan-out and lower faces: self-aimed `Noc::async_read` + `async_read_barrier` | construction: `ttnn/cpp/ttnn/kernel_lib/index_tile_dataflow.hpp:113-159` (fence `:136`, fan-out `:138-150`, lower faces `:152-159`); self address `ttnn/cpp/ttnn/kernel_lib/l1_helpers.hpp:41` (`local_noc_addr`) | face row bytes, face bytes, faces per tile row: CT from `input_tensor.tile`; `seed_rows` CT | `cb_expanded_tiles` (in place) | `cb_expanded_tiles` | `block_src_col_tiles` (pages per barrier round), `seed_rows` |
| `publish_block` | helper | host `ttnn.Mcast(device, McastConfig, receivers, receiver_group_size, McastFixedSenderConfig, McastCoreOrder.RowMajor)` + `Mcast.attach(...)`; kernel `McastArgs<ct, rt>().sender(noc).send(src_l1, dst_l1, size)` | `ttnn/cpp/ttnn-nanobind/mcast_host.cpp:171-191`, `:59-90`; `ttnn/cpp/ttnn/kernel_lib/mcast/kernel/mcast_args.hpp:309-329`; `ttnn/cpp/ttnn/kernel_lib/mcast/kernel/pipes/mcast_pipe.hpp:91-92` | `receiver_group_size = num_row_groups`; `src_l1 = dst_l1 =` block slot base; `size = block_src_col_tiles·page_bytes`; absent channel (`ttnn.attach_absent_mcast`, `mcast_host.cpp:92-106`) when `num_row_groups == 1` | `cb_expanded_tiles` | `cb_expanded_tiles` (remote) | `block_src_col_tiles` (payload), `num_row_groups` (fan-out) |
| `receive_block` | helper | `McastArgs<ct, rt>().receiver(noc).receive(round)` | `ttnn/cpp/ttnn/kernel_lib/mcast/kernel/pipes/mcast_pipe.hpp:162-163`; `mcast_pipe.inl` (`receive`) | `round = block_idx` | — | `cb_expanded_tiles` | — |
| `store_block` | raw_api | `Noc::async_write(CircularBuffer, TensorAccessor, size, {.offset_bytes}, {.page_id})`, `Noc::async_writes_flushed()`, `Noc::async_write_barrier()` | `tt_metal/hw/inc/api/dataflow/noc.h:411`, `:783`, `:760`; usage `ttnn/ttnn/operations/toy_spec_mcast/kernels/writer.cpp:19` | `size = page_bytes`; `offset_bytes = s_local·page_bytes`; `page_id =` out page of `(s, r)` | `cb_expanded_tiles` | — | `block_out_row_tiles`, `block_src_col_tiles` (writes per flush) |
| kernel exit (all) | raw_api | `Noc::async_write_barrier()`, `Noc::async_atomic_barrier()` | `tt_metal/hw/inc/api/dataflow/noc.h:760`, `:805` | — | — | — | — |

Helpers considered and rejected:

| Raw entry | Candidate | Mismatch (file:line) | Reason |
|-----------|-----------|----------------------|--------|
| `load_source_block` | `read_sticks_for_tilize` | `ttnn/cpp/ttnn/kernel_lib/tilize_helpers_dataflow.hpp:47`, `:87-88` | reads row-major sticks for tilization; x is already tiled, and the op needs whole tile pages into a CB. No kernel_lib helper reads tile pages into a CB. |
| `expand_rows_block` | `generate_index_tile` | `ttnn/cpp/ttnn/kernel_lib/index_tile_dataflow.hpp:74`, seed values `:126-128` | writes generated index values (`w + 16·j + l`) into the seed rows instead of copying an existing row; its seed → fence → fan-out → lower-face construction is reused as the realization |
| `expand_rows_block` | `copy_tile_column` | `ttnn/cpp/ttnn/kernel_lib/tile_column_dataflow.hpp:92` | copies one column of a tile; the op replicates a row |
| `expand_rows_block` | `fill_l1_range` | `ttnn/cpp/ttnn/kernel_lib/l1_helpers.hpp:100-101` | fills with one repeated value; a face row is 16 distinct elements |
| `store_block` | `write_sticks_after_untilize` | `ttnn/cpp/ttnn/kernel_lib/tilize_helpers_dataflow.hpp:96`, `:129-130` | writes row-major sticks; out is tiled and written as whole tile pages |
| all compute phases | `tilize`, `untilize`, `reduce`, `eltwise_*`, `matmul_block` | — | the op has no compute phase: a compute path is not a byte copy (`fpu_row_broadcast`, rejected) |

`compute_kernel_hw_startup()` is not called: there is no compute kernel.

## Key Risks and Gotchas

| Risk | Why it bites here | Mitigation in this design |
|------|-------------------|---------------------------|
| Logical elements read from x's padding | test cases fill rows 1–31 and columns `≥ W` of x with 1000.0 (`poison_padding`); the model's producer does not zero them | `expand_rows_block` reads only bytes `[0, 32)` and `[512, 544)` of each page; seed rows and the fan-out overwrite rows 1–15; the lower-face copy reads only after the fan-out barrier |
| RISC-V store vs NoC read ordering | a seed store can retire before it lands in L1; the NoC self-read could copy stale bytes (x padding) | one `ckernel::load_blocking` on the last stored word of the block before any fan-out read (`index_tile_dataflow.hpp:131-136`) |
| Bit exactness | NaN payloads (`0x7FA1`, `0xFFC5`, signalling `0x7F81`), ±0, denormals must survive | the bytes never pass through unpacker/DEST/packer; every move is a NoC or RISC-V byte copy |
| Multicast landing address | the sender multicasts to its own slot address; receivers must hold the same slot free | single CB over the rectangle (same base on all cores); lockstep nominal push/pop quanta; pre-handshake ack sent only after `cb_reserve_back` |
| Sender's push before its broadcast | the sender's writer reads E while the multicast reads E | both are reads; the sender cannot reserve the next block's slot until `send` returns (`SourceL1Guard::Guard`), so E is not overwritten under the multicast |
| Transactions in flight at kernel exit (BB-03) | `send` with Guard only waits until the source is read (`mcast_pipe.inl` `fence_`); receivers' ready increments are posted atomics; the next program starts at once | every reader ends with `async_write_barrier()` + `async_atomic_barrier()`; every writer ends with `async_write_barrier()` |
| `num_row_groups == 1` | Mcast with one-core groups has no remote receivers | attach the channel as absent; `McastArgs::active == false` compiles the broadcast out |
| Repeat determinism (BB-06) | a receiver that writes before the payload lands would emit stale L1 | writer waits on the CB, which the receiver pushes only after `receive()` saw the flag; the flag follows the payload on the same NoC path (`mcast_pipe.inl` `send_data_`/`signal_ready_`) |
| Program cache key (BB-07) | the plan depends on shape, sizes, dtype, layout, placement and device grid | the module-level artifact cache key contains all of them; the golden `sequence` case (46, 48, 46, 49, 50, 47) exercises it |
| Image folding for `d_0·d_1 > 1` | a src block can cross an image boundary | out page computed per page `s` with `s // tensor_w_tiles` and `s % tensor_w_tiles`, never from a block-start image |
| Kernel text size (BB-04, BB-18) | the Mcast decoder and pipes are template-heavy | one reader source CT-specialized per role, so each binary carries only its role's pipe face; no `-Os` |
