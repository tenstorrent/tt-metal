# L1 Ledger: tilize

Schema and audits: `.claude/references/l1-footprint-discipline.md`. Block axes are defined in `op_design.md` → Blocking Model → Axes: `tile_row` (`block_height`), `tile_col` (`block_width`), `image` (folded into `tile_row`), `stick_in_tile_row` (`tile_h`). All sizes are per Tensix core.

## Built regime (`row_split_interleaved`)

| CB | Capacity (pages) | Live set | Axis accounting | Page format | Producer | Consumer | Lifetime | Shares with / why not |
|----|------------------|----------|-----------------|-------------|----------|----------|----------|-----------------------|
| `cb_input_sticks` | `depth_in * rows_per_quantum * block_width` pages of `in_tile_bytes` (`= tile_h * 32 * in_elem_bytes`) | `2 * rows_per_quantum * block_width` pages: one quantum (`rows_per_quantum` tile-rows) being tilized plus one being filled. Capacity equals the live set; the second quantum slot is the double-buffering mechanism reason | `{tile_row: streams → rows_per_quantum tile-rows per quantum (a fixed window, not the block extent), tile_col: spans → block_width, image: streams (folded into tile_row), stick_in_tile_row: spans → tile_h (inside each page)}` | input dtype. Phase 0 Float16_b under 16-bit DEST (`fp32_dest_acc_en=False`). The fp32 → fp32 / int32 refinement makes it Float32/Int32 with `UnpackToDestFp32` and `fp32_dest_acc_en=True` | reader | compute | whole kernel (every block) | none. It cannot alias `cb_output_tiles`: the lifetimes are concurrent (tile-row `k+1` is loaded while tile-row `k` is packed) and the page formats differ once `dtype=` casts. In-place is forbidden by `static_assert(input_dfb != output_dfb)` (`tilize_helpers.inl:87-88`). Packing into the destination does not apply: this is the source side. Folding into DEST: tilize already goes unpack → DEST → pack with no intermediate buffer |
| `cb_output_tiles` | `depth_out * rows_per_quantum * block_width` pages of `out_tile_bytes` (`= tile.get_tile_size(out_dtype)`) | `2 * rows_per_quantum * block_width` pages: one quantum being written plus one being packed. Capacity equals the live set (double-buffering) | `{tile_row: streams → rows_per_quantum tile-rows per quantum, tile_col: spans → block_width, image: streams (folded), stick_in_tile_row: spans → tile_h (inside each tile page)}` | output dtype. Phase 0 Float16_b (16-bit DEST, so no wide page). Later Float32 only with `fp32_dest_acc_en=True`; Bfp8_b / Bfp4_b for the block-float casts | compute | writer | whole kernel | none. Concurrent lifetime with `cb_input_sticks` (above). Packing into the destination is the `sharded_resident` refinement: this CB is then backed on the output shard (`ttnn.cb_descriptor_from_sharded_tensor`) and costs 0 extra bytes. Interleaved output has no L1-resident destination to pack into |

### Knob-gated CB (implemented, not allocated at the default knob values)

| CB | Capacity (pages) | Live set | Axis accounting | Page format | Producer | Consumer | Lifetime | Shares with / why not |
|----|------------------|----------|-----------------|-------------|----------|----------|----------|-----------------------|
| `cb_input_sticks_odd` (index 2; split reader, `SPLIT_READER_MAX_SEGMENT_BYTES > 0` and stick segment at most that many bytes) | `depth_in * block_width` pages of `in_tile_bytes` (the split reader pins `rows_per_quantum = 1`) | `2 * block_width` pages: the odd walk positions' tile-rows (one being tilized, one being filled by BRISC) | same as `cb_input_sticks` (each CB streams every other tile-row of the walk) | input dtype, identical to `cb_input_sticks` (one tilize init configures both) | writer kernel (BRISC), as the second stick producer | compute | whole kernel (every block) | cannot share `cb_input_sticks`: a CB has exactly one producer and NCRISC already produces into it (CB ownership invariant). When it is allocated, `per_col_tile_bytes` counts it (`2 * depth_in * in_tile_bytes + depth_out * out_tile_bytes`), so `block_width` shrinks and the footprint stays within `CB_BUDGET_BYTES[low_l1]`. Default: not allocated (the knob measured slower on WH; see the program descriptor) |

Implementation notes (deviations from the planner's inventory, all advisory):

- `READ_AHEAD` (CB quanta of stick reads in flight before the oldest one's transaction-id barrier) is a live knob, `1 <= READ_AHEAD <= DEPTH_IN`. It adds no L1: the extra in-flight quantum lands in a slot `depth_in` already provides. Default 1.
- `rows_per_quantum` (verifier, Phase 0 review): the `tile_row` streaming window. Each CB push / pop, read barrier and write flush covers `rows_per_quantum` consecutive walk positions instead of one. Host-derived as `min(ceil(QUANTUM_MIN_TILES / block_width), max_positions // depth_in, CB_BUDGET_BYTES[low_l1] // (block_width * per_col_tile_bytes))`, floored at 1. It never lowers `block_width`, and it engages only when one tile-row is under `QUANTUM_MIN_TILES` = 8 tiles. Only the kernel's final quantum can be partial, and nothing is pushed after it, so the ring-wrap invariant holds. Measured −3 % / −6 % device-kernel ns on [1,1,16384,64] / [1,1,16384,32] (WH B0, 64 Tensix cores). Wider tile-rows are unchanged.
- Per-core traversal rotation: each Tensix core starts its tile-row walk (and its stick order inside a tile-row) at a per-core offset, so concurrent DRAM requests spread over all banks. Reader and writer walk the same rotated order. No L1 or byte-count change.

## Sharded regimes (`sharded_resident` / `sharded_accessor`, Refinement 1)

The same two CB slots, each either **resident** (backed on its own tensor's shard, 0 extra bytes) or **streamed** (the row above). A side is resident iff it is L1-sharded, `resident_ok`, and its per-core shard rectangle equals the core's assigned rectangle (`_core_assignment`).

| CB | Capacity (pages) | Live set | Axis accounting | Page format | Producer | Consumer | Lifetime | Shares with / why not |
|----|------------------|----------|-----------------|-------------|----------|----------|----------|-----------------------|
| `cb_input_sticks`, resident (input shard) | `shard_h / tile_h * shard_w / 32` pages of `in_tile_bytes`: the shard's own allocation (`ttnn.cb_descriptor_from_sharded_tensor`, format descriptor re-set to the tile-sized page + `TileDescriptor(tile_h, 32)`) | the whole shard, published at once (`core_row_tiles * block_width` pages: valid tile-rows at the nominal shard width) | `{tile_row: spans → shard_h / tile_h, tile_col: spans → shard_w / 32 (= block_width), image: spans (within the shard), stick_in_tile_row: spans → tile_h}` | input dtype | reader (publishes only, no NoC read) | compute | whole kernel | shares with the input tensor's shard buffer (zero-copy). A Layout::ROW_MAJOR shard row has stride `shard_w * elem_bytes = block_width * tile_col_bytes`, which is exactly the tilize input layout |
| `cb_output_tiles`, resident (output shard) | `shard_h / tile_h * shard_w / 32` TILE pages of `out_tile_bytes`: the shard's own allocation | the whole shard; compute packs `block_width` pages per tile-row straight into it | `{tile_row: spans → shard_h / tile_h, tile_col: spans → shard_w / 32, image: spans, stick_in_tile_row: spans → tile_h}` | output dtype | compute | writer (waits for completion, no NoC write) | whole kernel | shares with the output tensor's shard buffer (pack into the destination) |

Streamed partner of a resident side: the streamed row above, with `block_width = shard_w / 32` (the whole resident shard width is one block; a wider shard fails `resident_ok` and the side streams) and `per_col_tile_bytes` counting only the streamed CB. `rows_per_quantum` applies to the streamed side only. Footprint: at most `rows_per_quantum * block_width * depth * tile_bytes` for the one streamed CB, which is ≤ `CB_BUDGET_BYTES[low_l1]`; both sides resident = 0 CB bytes beyond the tensors.

Data movement per sub-case (bytes relative to the DRAM-boundary minimum):
- both resident (same spec): 0 NoC bytes. The reader publishes pages and the writer waits.
- input resident, output streamed (interleaved / DRAM-sharded / L1-interleaved out): input 0 bytes, output written once.
- output resident, input streamed (interleaved in, cross-spec, ND in): output 0 bytes, each input stick segment read once (remote L1 shard or DRAM).
- neither resident (DRAM-sharded both sides, ND without a 2-D equivalent): as `row_split_interleaved`, via `TensorAccessor` on both sides.

## Deferred-regime CBs (not allocated in Phase 0; listed so the refinement's footprint is decided now)

| CB | Capacity (pages) | Live set | Axis accounting | Page format | Producer | Consumer | Lifetime | Shares with / why not |
|----|------------------|----------|-----------------|-------------|----------|----------|----------|-----------------------|
| `cb_retile_staging` (`retile_l1_facewalk`) | `block_width` pages of `in_tile_bytes_at_in_tile_h` (one input tile-row) | `block_width` input tiles: the one input tile-row whose faces are being copied into sticks. Depth 1, because the face-walk copy runs synchronously on the reader RISC-V that filled it | `{tile_row: streams → one input tile-row (= in_tile_h / tile_h output tile-rows, or 1/(tile_h / in_tile_h) of one), tile_col: spans → block_width, image: streams, stick_in_tile_row: spans → in_tile_h}` | input dtype (a raw byte copy, no DEST involvement) | reader (NoC read) | reader (face-walk copy). A single RISC-V owns both ends, so there is no cross-thread credit | retile regime only | cannot alias `cb_input_sticks`: the copy reads staging while writing `cb_input_sticks`, so the lifetimes overlap. Cannot pack into the destination: the staging layout (faces) differs from the stick layout the tilize LLK consumes |

## Symbol table

| Symbol | Bound | Predicate / source that establishes it |
|--------|-------|----------------------------------------|
| `depth_in`, `depth_out` | = 2 (constant) | host constants `DEPTH_IN`, `DEPTH_OUT` |
| `rows_per_quantum` | `1 ≤ rows_per_quantum ≤ max(1, ceil(QUANTUM_MIN_TILES / block_width))` ≤ 8 tile-rows, and `rows_per_quantum * block_width * per_col_tile_bytes ≤ CB_BUDGET_BYTES[low_l1]` | host `min(...)` in `create_program_descriptor` (three caps, see implementation notes); `QUANTUM_MIN_TILES` = 8 is a host constant |
| `tile_h` | ∈ {32, 16, 8, 4, 2, 1} | `validate` rule 4 (`tile` height a power-of-two fraction of 32) |
| `in_elem_bytes` | ∈ {1, 2, 4} bytes | input dtype ∈ TARGET `dtype` (uint8, bf16/uint16, fp32/int32/uint32; fp8 = 1) |
| `in_tile_bytes` | `≤ 32 * 32 * 4 = 4096` bytes | `tile_h ≤ 32`, `in_elem_bytes ≤ 4` |
| `out_tile_bytes` | `≤ 4096` bytes (fp32 32×32); bf16 2048, bfp8_b 1088, bfp4_b 576 bytes at 32×32 | output dtype ∈ TARGET `output_dtype` |
| `CB_BUDGET_BYTES[low_l1]` | 524288 bytes (`low_l1=False`), 65536 bytes (`low_l1=True`) | host constant table; a device-independent constant, never a function of the tensor |
| `col_align_tiles` | ∈ {1, 2} tiles | `ceil(align_bytes / (32 * in_elem_bytes))`, with `align_bytes` ≤ 64 bytes (Blackhole DRAM) and `32 * in_elem_bytes` ≥ 32 bytes |
| `block_width_cap` | `col_align_tiles ≤ block_width_cap ≤ 255` tiles | `floor_to(min(255, CB_BUDGET_BYTES[low_l1] // per_col_tile_bytes), col_align_tiles)`, `per_col_tile_bytes = depth_in*in_tile_bytes + depth_out*out_tile_bytes` |
| `block_width` | `≤ block_width_cap` tiles | `balanced_width` (op_design.md). **This is the only knob a CB size scales with, and it is bounded independently of every tensor dim** |
| `block_height`, `core_row_tiles`, `core_col_tiles`, `R`, `C` | unbounded (tensor-derived) | appear in **no** capacity expression: every CB streams over `tile_row`, and `tile_col` enters only through the capped `block_width` |

## Footprint

```
L1_cb_total = rows_per_quantum * block_width * (depth_in * in_tile_bytes + depth_out * out_tile_bytes)
            = rows_per_quantum * block_width * per_col_tile_bytes
            <= CB_BUDGET_BYTES[low_l1]          (block_width_cap, then the budget cap on rows_per_quantum)
```

- Scales with `block_width` (the `tile_col` extent knob), `rows_per_quantum` (the `tile_row` streaming window) and `depth_in` / `depth_out` (the depth knobs). It does not scale with `block_height` or with any tensor dim. `rows_per_quantum * block_width` is at most `max(block_width, QUANTUM_MIN_TILES + block_width - 1)` tiles.
- Phase 0 (bf16 → bf16, 32×32): `per_col_tile_bytes` = 8192 bytes; `block_width = min(C, 64)`-class ⇒ at most 524288 bytes. `[1,1,2048,64]` (C = 2, 1 tile-row per core ⇒ `rows_per_quantum` = 1) uses 16384 bytes; the perf-focus `[1,1,16384,64]` (C = 2, 8 tile-rows per core ⇒ `rows_per_quantum` = 4) uses 65536 bytes.
- `low_l1=True`: at most 65536 bytes for every shape (bf16 cap 8 tiles, fp32 → fp32 cap 4 tiles). This is O(1) in the tensor dims. So is `low_l1=False` (at most 524288 bytes), and neither setting can OOM on `low_l1_forcing_width` [1,1,32,8192] (C = 256): fp32 → fp32 caps `block_width` at 32 tiles, i.e. 524288 bytes, versus the 2 MiB an unbounded row would need.
- The data path is identical at both settings. Only `block_width` differs, and it changes how the NoC reads are grouped, not which bytes land in which output tile, so the A/B readback is bit-identical.

## Data-movement budget (chosen split: `row_split_interleaved`)

Bytes are for a tile-aligned, unpadded input with `n_elems = R * C * tile_h * 32` elements.

| Tensor | DRAM crossings | Why that many | Cross-core traffic added |
|--------|----------------|---------------|--------------------------|
| input (ROW_MAJOR sticks) | 1 (`n_elems * in_elem_bytes` bytes, in `R * tile_h * num_col_blocks` NoC reads) | each output tile-row's stick segments are read exactly once into `cb_input_sticks` and consumed by the tilize. No second pass exists, so there is nothing that residency could save | 0 |
| output (TILE pages) | 1 (`R * C * out_tile_bytes` bytes, in `R * C` tile-page writes) | each tile is packed once and written once | 0 |

Measured (Wormhole B0, 64 Tensix cores, bf16, default knobs): [1,1,16384,64] 26.8 us (4 MiB DRAM traffic ~ 156 GB/s combined), [1,1,16384,32] 19.6 us, [1,1,32768,64] 51.5 us, [1,1,128,64] 3.1 us on 4 Tensix cores. Ablation on [1,1,16384,64]: reads-only ~16 us, writes-only ~16 us, neither ~2.3 us: the two DRAM streams together, not compute, bind the wall.

Totals per tier: DRAM = `n_elems * in_elem_bytes + R * C * out_tile_bytes` bytes (the minimum: one read of the input and one write of the output); cross-core NoC = 0 bytes; core-local L1 = each byte written once into a CB and read once by the unpacker (input) or the NoC (output).

> Cheapest-traffic split considered: `row_split_interleaved` (cut `tile_row` only). Every candidate split (row, column, 2-D) moves the same minimum bytes: 1 DRAM read of the input, 1 DRAM write of the output, 0 cross-core bytes. The row split is also cheapest in transaction count, because it keeps the largest stick segments (`block_width * 32 * in_elem_bytes` bytes). Implemented: `row_split_interleaved`. The implemented split is the cheapest. The `grid_2d_split` successor adds no bytes, only smaller transactions, and is `deferred` for occupancy on `short_wide` / `square_large` (outside the Phase 0 rectangle). The Phase 0 kernels already take a per-core column range, so it stays reachable.
