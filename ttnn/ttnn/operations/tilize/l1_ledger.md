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
- Refinement 3 (perf, all parked at defaults that allocate exactly the rows above): in-flight windows (`READ_WINDOW_MIN_TILES` / `WRITE_WINDOW_MIN_TILES` → `read_ahead` / `write_ahead` CB quanta, `depth_in` / `depth_out` grown to hold them; the writer's write-ahead uses one NoC transaction id per `cb_output_tiles` slot), eager publish, NoC stream split (`DM_DYNAMIC_NOC` kernels), bank-stride addressing. None adds a CB; a window only deepens `cb_input_sticks` / `cb_output_tiles` and stays inside `CB_BUDGET_BYTES[low_l1]`. `rows_per_quantum` gains a fourth cap on the stick reader: `rows_per_quantum * tile_h * (reads per stick segment) ≤ NOC_MAX_TRANSACTION_ID_COUNT` = 255, since one transaction id counts a slot's outstanding reads (inactive at every measured shape).
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

## Retile regime (`retile_l1_facewalk`, Refinement 2)

A Layout::TILE input at `in_tile_h`. `cb_input_sticks` and `cb_output_tiles` are unchanged (streamed rows above, or a resident output); `cb_input_sticks` always streams here, because the face walk fills it. One reader-private CB is added.

| CB | Capacity (pages) | Live set | Axis accounting | Page format | Producer | Consumer | Lifetime | Shares with / why not |
|----|------------------|----------|-----------------|-------------|----------|----------|----------|-----------------------|
| `cb_retile_staging` (index 3), streamed input | `RETILE_STAGE_DEPTH * unit_in_rows * block_width` pages of `in_page_bytes` (one `[in_tile_h, 32]` input tile; `unit_in_rows = max(1, tile_h / in_tile_h)`) | `RETILE_STAGE_DEPTH` units (default 2): one unit being face-walked, the next unit's page-sized tile reads in flight (transaction-id prefetch). Capacity equals the live set | `{tile_row: streams → one retile unit (row_align output tile-rows, fed by unit_in_rows input tile-rows), tile_col: spans → block_width, image: streams, stick_in_tile_row: spans → in_tile_h (inside each input tile)}` | input dtype, `TileDescriptor(in_tile_h, 32)`. Raw bytes only: nothing is unpacked from it | reader (NoC tile reads) | reader (face walk: NoC loopback reads by default, RISC-V copies under `RETILE_FACEWALK_NOC=False`). One RISC-V owns both ends, so it never pushes or pops; ordering is by transaction id | whole kernel, retile only | cannot alias `cb_input_sticks`: the face walk reads staging while it writes `cb_input_sticks`, so the lifetimes overlap. Cannot pack into the destination: face layout ≠ the stick layout the tilize LLK consumes. Counted in `per_col_tile_bytes` (`+ RETILE_STAGE_DEPTH * unit_in_rows * in_page_bytes` per tile-column), so `block_width` shrinks to keep the total ≤ `CB_BUDGET_BYTES[low_l1]` |
| `cb_retile_staging`, resident input shard | the shard's own allocation (`ttnn.cb_descriptor_from_sharded_tensor`; format re-set to the input tile page) | the whole shard, read in place | `{tile_row: spans → shard_h / in_tile_h input tile-rows, tile_col: spans → shard_w / 32 (= block_width), image: spans, stick_in_tile_row: spans → in_tile_h}` | input dtype | (the tensor) | reader (face walk) | whole kernel | shares with the input tensor's shard buffer (zero-copy: 0 NoC bytes into L1, 0 extra CB bytes). Requires `H % in_tile_h == 0` (else the TILE input's physical rows differ from the logical fold and the side streams) |

- `row_align = in_tile_h / tile_h` when `in_tile_h > tile_h` and every image's `H` is a whole number of input tile-rows (and, under a resident output, every core's rectangle is a multiple of it); else 1. The row split cuts `tile_row` in units of `row_align`, and the walk rotation is a multiple of it, so each input tile is read exactly once per Tensix core. With `row_align` forced to 1 and `in_tile_h > tile_h`, each output tile-row reads its whole input tile-row: correct, with `in_tile_h / tile_h` read amplification (H-padded TILE inputs, output shards that cut input tile-rows).
- Knobs: `RETILE_STAGE_DEPTH` (1 = read → barrier → walk serially; ≤ 14 because the face walk takes the next transaction id) and `RETILE_FACEWALK_NOC`. Both are live and covered by `test_tilize_tile_geometry.py::test_retile_knob`.

Data movement (retile): input 1 DRAM crossing in whole-tile NoC reads (`R_in * C` reads of `in_page_bytes`; 2048 bytes at `in_tile_h = 32` bf16, 64 bytes at `in_tile_h = 1`); output 1 crossing as in the row split; plus one core-local L1 → L1 re-lay of every input byte (the face walk), in face-row moves of `16 * in_elem_bytes` bytes. No ROW_MAJOR tensor is materialized.

## Padding (Refinement 4)

`cb_input_sticks` / `cb_output_tiles` are unchanged in size and format; the tile grid they stream is the PADDED shape's (`R = prod(P[:-2]) * P[-2] / tile_h`, `C = P[-1] / 32`). The reader fills every byte of a slot the input does not cover before it pushes the slot. One reader-private CB is added when something is to be filled (`PadSpec.needs_fill`); an unpadded call, or a pad that fills nothing, allocates exactly the rows above.

| CB | Capacity (pages) | Live set | Axis accounting | Page format | Producer | Consumer | Lifetime | Shares with / why not |
|----|------------------|----------|-----------------|-------------|----------|----------|----------|-----------------------|
| `cb_pad_source` (index 4; padded only) | 1 page of `PAD_SOURCE_BYTES` = 1024 bytes | the whole page: filled once with the pad value (`fill_l1_range`), then only read, as the source of NoC loopback fills | none: a constant, independent of every block axis and tensor dim | input dtype (raw fill bytes; never unpacked) | reader (one CPU fill at kernel start) | reader (NoC loopback reads into `cb_input_sticks`). One RISC-V owns both ends, so it never pushes or pops; ordering is by the fill's own transaction id (`depth_in + 1`), drained before every slot push | whole kernel, padded only | cannot be `cb_input_sticks` itself: a slot is overwritten by stick reads every pass, and the source must outlive every slot. Not counted in `per_col_tile_bytes` (it is 1 KiB, dimension-independent), so the streamed CBs keep their `CB_BUDGET_BYTES[low_l1]` bound and the total grows by at most 1024 bytes |

- Fill mover: ranges shorter than `PAD_NOC_MIN_BYTES` = 128 bytes are CPU stores (the W-tail band of a tile-row is split into head / words / tail once per tile-row, then stored per stick); longer ranges (whole pad sticks, the H tail, whole pad tile-rows / images, a wide explicit W growth) are NoC loopback copies in chunks of up to `PAD_SOURCE_BYTES`. Measured on [1,1,8192,64] → [1,1,16384,64] (half the tile-rows whole pad, WH, 64 Tensix cores): 22.0 µs with loopback fills vs 49.5 µs CPU-only; a 2048-byte source 23.6 µs (flat).
- W-tail persistence (`PAD_W_TAIL_PERSIST`, default on): with one column block per core, a CB row's W-tail band is never overwritten after it is filled (stick reads stop at `data_bytes`, whole-stick fills write the same value, compute only reads), so it is filled on the walk's first pass through the ring only. Measured [1,1,65520,50] (32 tile-rows per core): 118.8 vs 129.7 µs.
- The input side never resides under a fill (a padded input's stick layout is not the tilize layout of the padded grid); the split reader and the parked Refinement 3 NoC levers are off on the padded path.

Data movement (padded): input 1 DRAM crossing of the input's own bytes (`X` elements: only existing sticks are read, each for its data bytes only); output 1 crossing of the padded grid (`R * C` tile writes over `P`); plus core-local L1 fills of `(prod(P) - prod(X)) * in_elem_bytes` bytes (CPU stores or NoC loopback), no cross-core traffic.

## 2-D split (`grid_2d_split`, Refinement 5) and `low_l1`

No CB is added, removed or resized by the regime itself: each Tensix core runs the `row_split_interleaved` rows above over its own rectangle `[row_start, +core_row_tiles) x [col_start, +core_col_tiles)`. `block_width = balanced_width(core_col_tiles_max, cap)` is now computed from the busiest core's column-group width rather than from `C`, so it can only shrink. `cb_input_sticks` / `cb_output_tiles` rows are unchanged (`Shares with / why not`: as above: the two CBs are live concurrently by construction, since they are the double-buffered pipeline).

- Assignment (`tilize_program_descriptor.grid_2d_split`): `(g_r, g_c)` minimizes `ceil(R_units / g_r) * row_align * (ceil(col_units / g_c) * col_align_tiles + ROW_COST_TILES)` over `g_r * g_c <= N`. Tie-breaks: wider column groups, then fewer Tensix cores. `ROW_COST_TILES = 1.5` charges each tile-row's fixed cost (`tile_h` stick reads + a CB handshake); 0 is the pinned tile-count rule, which measured +22 % on [4,3,256,96] and +97 % on [1,1,2080,2048]. `g_c == 1` keeps the row split (`split_work_to_cores`, byte-identical to before). Core k of the first `g_r * g_c` Tensix cores (row-wise) owns row group `k // g_c` and column group `k % g_c`.
- `PIPELINE_MIN_POSITIONS = 2` / `PIPELINE_MIN_SEGMENT_BYTES = 2048`: a core with a single walk position cuts its columns into up to 2 blocks, but only while each block keeps at least 2 KiB stick segments. That lets read, tilize and write overlap in the depth-2 CBs. It lowers `block_width` and so the footprint: it never raises it.
- `low_l1=True` → `CB_BUDGET_BYTES[True]` = 65536 bytes. The footprint formula above holds unchanged (`block_width_cap` = 8 tiles at bf16), and the A/B readback is bit-identical (`test_tilize_grid_2d.py::test_low_l1_ab_bit_identical`, golden low_l1 scenarios).

Data movement (`grid_2d_split`): the same minimum bytes as the row split (input 1 DRAM crossing, output 1 DRAM crossing, 0 cross-core bytes). Transactions: `R * tile_h * g_c` stick-segment reads of `core_col_tiles * 32 * in_elem_bytes` bytes (short_wide_canonical [1,1,32,2048] on 64 Tensix cores: 64-byte segments, 32 per core) + `R * C` tile-page writes.

## Symbol table

| Symbol | Bound | Predicate / source that establishes it |
|--------|-------|----------------------------------------|
| `depth_in`, `depth_out` | = 2 at the default knobs; ≤ `MAX_WINDOW_QUANTA` = 8 CB quanta | `depth_in = max(DEPTH_IN, read_ahead)`, `depth_out = max(DEPTH_OUT, write_ahead)` (Refinement 3). The windows open only when `READ_WINDOW_MIN_TILES` / `WRITE_WINDOW_MIN_TILES` > 0 (default 0) and a quantum is narrower than the window, and a budget loop shrinks them until `quantum_tiles * per_col_tile_bytes(depth_in, depth_out) ≤ CB_BUDGET_BYTES[low_l1]` |
| `rows_per_quantum` | `1 ≤ rows_per_quantum ≤ max(1, ceil(QUANTUM_MIN_TILES * (32 / tile_h) / block_width))` tile-rows (the floor counts full 32-row tile equivalents, so ≤ 8 tile-rows at `tile_h = 32`, ≤ 256 at `tile_h = 1` = 16 KiB of bf16 sticks), and `rows_per_quantum * block_width * per_col_tile_bytes ≤ CB_BUDGET_BYTES[low_l1]` | host `min(...)` in `create_program_descriptor` (three caps, see implementation notes); `QUANTUM_MIN_TILES` = 8 is a host constant |
| `tile_h` | ∈ {32, 16, 8, 4, 2, 1} | `validate` rule 4 (`tile` height a power-of-two fraction of 32) |
| `in_tile_h` | ∈ {32, 16, 8, 4, 2, 1} (retile only) | a Layout::TILE input's own tile height (`ttnn.Tile` legal heights) |
| `unit_in_rows` | `≤ 32` input tile-rows | `max(1, tile_h / in_tile_h)`; `unit_in_rows * in_page_bytes = tile_h * 32 * in_elem_bytes` when `in_tile_h < tile_h`, so staging per tile-column is ≤ `RETILE_STAGE_DEPTH * 4096` bytes |
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
