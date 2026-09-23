# Operation Design: tilize

Terminology used throughout (qualified once here, bare afterwards):

| Term | Meaning in this document |
|------|--------------------------|
| **block** | blocking-model block (`.claude/references/blocking-model.md` §1): a rectangle of OUTPUT tiles, `block_height` tile-rows × `block_width` tile-columns, driven through one pipeline pass. Not a bfp8 block, not a matmul block, not `BLOCK_SHARDED`. |
| **core** | Tensix core (`CoreCoord`). A RISC-V processor is always named as such (NCRISC = reader, BRISC = writer, TRISC0/1/2 = compute). |
| **stick** | one ROW_MAJOR page: one row of `W` elements of the input, `W * in_elem_bytes` bytes. |
| **stick segment** | the `block_width * 32` elements of one stick that one block needs (a byte range inside a stick page). |
| **tile-row** | `tile_h` consecutive sticks' worth of output: one row of tiles in the output tile grid. |
| **Layout** | `ttnn.Layout` (`ROW_MAJOR_LAYOUT` / `TILE_LAYOUT`). Placement is always `TensorMemoryLayout`. |
| **shape** | the logical shape, unless "padded shape" is written. |

## Overview

| Field | Value |
|-------|-------|
| Classification | data_movement (a compute-kernel re-lay, no arithmetic) |
| Goal | Re-lay a `Layout::ROW_MAJOR` tensor into `Layout::TILE` (and, later, re-tile a `Layout::TILE` tensor to another tile height) in exactly one `ttnn.generic_op` dispatch, with every Tensix core of the device grid participating on the geometries SUPPORTED declares. |
| Math | `tilize(x)[i, j] == x[i, j]` for every element of the data region; the pad region (later refinement) holds exactly `pad_value`. Values bit-identical unless `dtype=` changes the storage format. |
| Mode | Hybrid: compute = kernel-lib `compute_kernel_lib::tilize` helper; reader / writer = custom block operations (dataflow helper `read_sticks_for_tilize` rejected with file:line reason in API Mapping). |
| References | `.claude/references/blocking-model.md`; `.claude/references/l1-footprint-discipline.md`; `.claude/references/ttnn-cb-memory-fundamentals.md` (Tilize Data Flow Pattern, CB Ring-Wrap Invariant, CB Ownership Invariant); `ttnn/cpp/ttnn/kernel_lib/tilize_helpers.hpp` / `.inl`; `ttnn/cpp/ttnn/kernel_lib/tilize_helpers_dataflow.hpp` / `.inl`; `ttnn/cpp/ttnn/kernel_lib/l1_helpers.hpp`; `tt_metal/hw/inc/api/compute/tilize.h`; `eval/golden_tests/tilize/feature_spec.py`; perf catalog `ttnn/ttnn/operations/examples/{double_buffer,width_split,noc_placement,split_reader,compute_block_size}/README.md` |

### Perf-catalog entries that set knobs

| Entry | Finding used | Knob it set |
|-------|--------------|-------------|
| `examples/double_buffer` | One read then one barrier is the single-core trap (6.5 GB/s bf16); 4–8 reads in flight then one barrier reaches the per-core NoC limit; double buffering adds 11.2 → 17.9 GB/s; at 64 cores DRAM-bound and depth stops mattering. | Barrier once per tile-row (`tile_h` = 32 stick-segment reads in flight); `depth_in = depth_out = 2`. |
| `examples/split_reader` | One RISC-V issues ~8–9 M transactions/s; 32 B transactions ~6× slower than 2 KB; 1–2 KB transactions near the floor. | `block_width` default coarse (stick segment = `block_width * 64` bytes at bf16 ⇒ 4096 bytes at the Phase 0 cap); split-reader perf lamp for narrow sticks. |
| `examples/noc_placement` | `split_work_to_cores(row_wise=False)` produces a column line of cores that is ~2.9× slower than a row line on DRAM traffic; reads on NoC0 / writes on NoC1 are the good default. | `split_work_to_cores(grid, R, row_wise=True)`; reader on NCRISC / NoC0, writer on BRISC / NoC1 (defaults). |
| `examples/width_split` | A tile-row split strands a one-tile-row tensor on one Tensix core; a width split capped by a `WT_CHUNK` constant fills the grid (6.25× at Wt = 64). | The `tile_col` axis carries a bounded extent knob from day 1 and the kernels take a per-core column range; the 2-D assignment is the `grid_2d_split` regime row. |
| `examples/compute_block_size` | Each extra pass costs ~320 ns per phase of init/reconfig/pipeline fill; one pass over the whole work beats per-tile-row passes 1.65×. | Compute is ONE helper call per core (init/uninit once per kernel), not one per block. |

## Parameters

| Name | Type | Required | Valid Range | Default | CT/RT |
|------|------|----------|-------------|---------|-------|
| `input_tensor` | `ttnn.Tensor` | yes | on device; `Layout::ROW_MAJOR_LAYOUT` (or `Layout::TILE_LAYOUT` with `tile=` for retile) | — | buffer address = RT arg |
| `memory_config` | `ttnn.MemoryConfig \| None` | no | any interleaved / sharded config | input's `memory_config()` | host (output allocation); output buffer address = RT arg |
| `dtype` | `ttnn.DataType \| None` | no | same dtype family as input (see Precision) | input dtype (fp8_e4m3 input → `float32`) | host → CB data format of `cb_output_tiles` |
| `low_l1` | `bool` | no | `{False, True}` | `False` | host → selects `cb_budget_bytes`, hence `block_width` (CT) |
| `output_padded_shape` | `list[int] \| ttnn.Shape \| None` | no | ≥ input shape in every dim; last two dims multiples of (`tile_h`, 32) | `None` | host (padded shape) → RT extents |
| `pad_value` | `float \| int \| None` | no | any; integers bit_cast signed → unsigned | `None` | RT arg (packed per input dtype) |
| `tile` | `ttnn.Tile \| None` | no | height ∈ {32,16,8,4,2,1}, width = 32 | `ttnn.Tile([32, 32])` for Layout::ROW_MAJOR input; REQUIRED for Layout::TILE input | host → CB tile descriptor (`tile_h`) |

Index canonicalization: the op has no `dim` argument, so no index axis to canonicalize.

Validation order in the entry point (all before `validate()` support gating):

| # | Condition | Raises |
|---|-----------|--------|
| 1 | `input_tensor.storage_type()` is not device | `ValueError` |
| 2 | `layout` not in {ROW_MAJOR, TILE} | `ValueError` |
| 3 | `layout == TILE` and `tile is None` | `ValueError` |
| 4 | `tile` given and (`tile_shape[1] != 32` or `tile_shape[0]` not in {32,16,8,4,2,1}) | `ValueError` |
| 5 | no padding argument and (rank < 2, or `shape[-2] % tile_h != 0`, or `shape[-1] % 32 != 0`) | `ValueError` |
| 6 | `output_padded_shape` given and any dim smaller than the (left-1-expanded) input shape, or its last two dims not multiples of (`tile_h`, 32) | `ValueError` |
| 7 | `validate()` — SUPPORTED / EXCLUSIONS | `NotImplementedError` (`UnsupportedAxisValue` / `ExcludedCell`) |

The five "none"-sentinel axes (`shard_api`, `orientation`, `pad_mode`, `pad_value`, `in_tile_height`) are never refused on the value `"none"`.

## Tensors

### Input

| Property | Requirement |
|----------|-------------|
| Shape | rank 0–6 (TARGET); Phase 0: rank 4. `R = prod(shape[:-2]) * ceil(shape[-2] / tile_h)`, `C = ceil(shape[-1] / 32)` — per image, `ceil`, never `floor(batch*H/32)` |
| Dtype | TARGET: bfloat16, float32, fp8_e4m3, uint32, int32, uint16, uint8. Phase 0: bfloat16 |
| Layout | `Layout::ROW_MAJOR_LAYOUT` (Phase 0); `Layout::TILE_LAYOUT` at `in_tile_height` = retile (deferred regime) |
| Memory | TARGET: `TensorMemoryLayout::INTERLEAVED` DRAM/L1, HEIGHT/WIDTH/BLOCK sharded (legacy 2D and ND). Phase 0: `TensorMemoryLayout::INTERLEAVED`, `BufferType::DRAM` |

### Output

| Property | Value |
|----------|-------|
| Shape | logical shape = input logical shape (never promoted to the padded shape); padded shape = input shape, or tile-round (`pad_value` only), or `output_padded_shape` |
| Dtype | `dtype` if given, else input dtype (fp8_e4m3 input → float32) |
| Layout | `Layout::TILE_LAYOUT`, tile = `tile` if given else 32×32 |
| Memory | `memory_config` if given, else the input's. Phase 0: `TensorMemoryLayout::INTERLEAVED`, `BufferType::DRAM` |

### Phase 0 SUPPORTED rectangle (the implementer writes the block; this is its content)

| Axis (TARGET name) | Phase 0 values |
|--------------------|----------------|
| `dtype` | `[ttnn.bfloat16]` |
| `output_dtype` | `[ttnn.bfloat16]` |
| `low_l1` | `[False]` |
| `shard_api` | `["none"]` |
| `out_scheme` | `["interleaved"]` |
| `buffer` | `["dram_to_dram"]` |
| `rank` | `[4]` |
| `orientation` | `["none"]` |
| `pad_mode` | `["none"]` |
| `pad_value` | `["none"]` |
| `alignment` | `["tile_aligned"]` |
| `tile_height` | `[32]` |
| `in_tile_height` | `["none"]` |
| `tile_grid` | `["single_tile", "small", "tall_narrow"]` |

`output_dtype` is declared and gated in `validate()` from Phase 0 on, so every later cast goes through the registry. The twelve INPUT_TAGGERS are written from the `Expected INPUT_TAGGERS` block of `feature_spec.py` (each reads `inputs[0]` as the scenario dict), and must agree with `eval/golden_tests/tilize/axes.py:classify_call`; `tag_tile_grid` imports `DOMINANT` from `feature_spec.py` rather than restating 16.

### TARGET − SUPPORTED: refinement map

Each value outside Phase 0 is classified against this op's axis characters (both work axes are **independent** — see Blocking Model). No value requires a cross-core combine, so none is a dependent-axis scheme-change.

| TARGET axis → values | Class | What changes | Regime row |
|---------------------|-------|--------------|------------|
| `tile_grid` → `short_wide`, `square_large` | knob-turn (independent axis spread across Tensix cores) | host core-assignment function only: `num_col_groups` from 1 to the `grid_2d_split` rule; kernels already take a per-core column range | `grid_2d_split` |
| `rank` → 0,1,2,3,5,6 | knob-turn (SUPPORTED widening) | none for 2,3,5,6 (leading dims already fold into `R`); rank 0/1 need the pad refinement (dims synthesized) | `row_split_interleaved` |
| `buffer` → `dram_to_l1`, `l1_to_l1`, `l1_to_dram` | knob-turn (placement) | none in kernels: `TensorAccessor` addresses interleaved L1 and DRAM identically | `row_split_interleaved` |
| `low_l1` → `True` | knob-turn | `cb_budget_bytes` constant 524288 → 65536 bytes, which lowers `block_width`; bit-identical output by construction (same data path) | `row_split_interleaved` |
| `pad_mode` → auto/explicit; `pad_value` → zero/positive/negative; `alignment` → w/h/hw_non_aligned | feature (reader) — no blocking change | `load_block` gains per-image stick mapping and `fill_l1_range` for W tail / H tail / whole pad sticks; `R`, `C` computed from the PADDED shape | `row_split_interleaved` |
| `tile_height` → 16,8,4,2,1 | feature (CB tile descriptor) — no blocking change | both CBs carry `TileDescriptor(tile_h, 32)`; the helper leaves the fast path (`dfb_has_32x32_tiles`, tilize_helpers.inl:67) and uses `tilize_init`/`tilize_block`; reader groups `tile_h` sticks per tile-row | `row_split_interleaved` |
| `in_tile_height` → 32..1 | scheme-change (new reader block operation, same compute) | reader walks input tiles' faces in L1 instead of reading sticks | `retile_l1_facewalk` |
| `dtype` / `output_dtype` casts | feature (CB formats + compute config) | `cb_output_tiles` format = output dtype; pack converts. fp32→fp32 and 32-bit integers need `fp32_dest_acc_en=True`, `UnpackToDestFp32` on `cb_input_sticks`, `Fp32Mode::Lossless` | `row_split_interleaved` |
| `shard_api` → legacy_2d/nd; `out_scheme` → HEIGHT/WIDTH/BLOCK/nd; `orientation` → ROW/COL | knob-turn (placement of the logical shards the split already defines) | core assignment pinned by the shard; resident side's CB backed on the shard buffer | `sharded_resident`, `sharded_accessor` |

### Structural impossibilities (candidate INVALID entries for a later `/golden-tests` run)

| Cell | Why it is impossible |
|------|----------------------|
| `{dtype: fp8_e4m3, in_tile_height: 32 / 16 / 8 / 4 / 2 / 1}` | fp8_e4m3 exists only in `Layout::ROW_MAJOR` (feature_spec.py TARGET comment on `dtype`), so there is no TILE input to re-tile. The harness should skip these rather than let `from_torch` fail. |

## Blocking Model

Semantics: `.claude/references/blocking-model.md`. Everything below is its realization.

### Axes

The op's data is the OUTPUT tile grid `R × C` (task spec, "Work geometry"). Each output tile `(r, c)` is built from the stick segments `[c*32, c*32+32)` elements of sticks `r*tile_h … r*tile_h + tile_h − 1` (Phase 0, where every image's `H` is a multiple of `tile_h`). No output tile reads any other output tile's input.

| Axis | Character (+ one-clause reason) | Extent knob | Phase 0 value | Knob source | Core-assignment | Later unlock |
|------|--------------------------------|-------------|---------------|-------------|-----------------|--------------|
| `tile_row` (R: leading dims × `ceil(H / tile_h)`) | independent — an output tile-row depends only on its own `tile_h` sticks | `block_height` (tile-rows per block) | `core_row_tiles` — the core's whole row assignment (one block along R). The input and output CBs only stream along this axis, so a larger extent costs no L1 | host: per-core RT arg `core_row_tiles` from `split_work_to_cores`; `block_height = core_row_tiles` derived, never restated | `ttnn.split_work_to_cores(grid, R, row_wise=True)` across the full compute grid read from `device.compute_with_storage_grid_size()`; reaches `min(R, N)` Tensix cores | knob-turn |
| `tile_col` (C: `ceil(W / 32)`) | independent — an output tile-column depends only on its own 32-element stick segments | `block_width` (tile-columns per block) | `balanced_width(core_col_tiles, block_width_cap)` (defined below); for bf16 that is `min(C, 64)`-class | host constant `CB_BUDGET_BYTES[low_l1]` → `block_width_cap` → `block_width`; passed once as a compute CT arg (helper template param) and as a reader/writer CT arg | not split across Tensix cores in Phase 0 (`num_col_groups = 1`, a host parameter): every core owns columns `[0, C)`. Reason: the task's Phase 0 rectangle is the row-split corner; the column split is the `grid_2d_split` regime row | knob-turn (`num_col_groups` → `grid_2d_split` rule) |
| `image` (leading dims, folded) | independent — tilize treats every image identically | folded into `block_height` (no separate extent): with `H % tile_h == 0`, consecutive images' tile-rows are contiguous in both stick index (`r*tile_h + s`) and output tile index (`r*C + c`), so the fold is exact | folded | derived: `R = prod(shape[:-2]) * ceil(shape[-2]/tile_h)` on host | follows `tile_row` | knob-turn; the pad refinement keeps the fold but maps `r → (image, h_tile)` to address sticks, because pad breaks the contiguity |
| `stick_in_tile_row` (the `tile_h` sticks of one tile-row) | independent, but never cut — a tile is the tilize LLK's atomic unit | `tile_h` (fixed by the output tile geometry) | 32 | host: `tile.tile_shape[0]`, carried by the CB `TileDescriptor` (the kernels read it from the CB, `unpack_tile_r_dim`) | not assigned: always whole inside one block (mechanism cap: the LLK consumes exactly `tile_h` sticks per tile-row) | not an unlock axis |

`balanced_width` definition (host, single source):

```
per_col_tile_bytes = depth_in * in_tile_bytes + depth_out * out_tile_bytes
block_width_cap    = floor_to(min(255, CB_BUDGET_BYTES[low_l1] // per_col_tile_bytes), col_align_tiles)
num_col_blocks     = ceil(core_col_tiles_max / block_width_cap)
block_width        = round_up(ceil(core_col_tiles_max / num_col_blocks), col_align_tiles)
```

`num_col_blocks_this_core = ceil(core_col_tiles / block_width)`; `last_block_width = core_col_tiles − (num_col_blocks_this_core − 1) * block_width`. Balancing gives the same block count as the coarsest cap-sized split, with less tail waste (C = 100, cap 64 → two 50-wide blocks instead of 64 + 36).

Phase 0 values at bf16 → bf16 (2048-byte tiles): `per_col_tile_bytes` = 8192 bytes, `block_width_cap` = 64 tiles, stick segment = up to 4096 bytes per NoC read.

Every knob is a parameter: the kernels loop `for (col_block_idx = 0; col_block_idx < num_col_blocks_this_core; ++col_block_idx)` and `for (row_idx = 0; row_idx < core_row_tiles; ++row_idx)` on RT args; no loop bound is a literal, no size is baked in, and the Tensix core count comes from the device at runtime.

### Buffer-depth knobs

| CB | Depth knob | Phase 0 value | What the depth buys |
|----|------------|---------------|---------------------|
| `cb_input_sticks` | `depth_in` (tile-rows of `block_width` tiles) | 2 | reader fills tile-row `k+1` while compute tilizes tile-row `k` (`double_buffer`: 11.2 → 17.9 GB/s single-core). Depth is counted in helper quanta (one tile-row = `block_width` pages), so capacity = `2 * block_width` pages, not 2 pages |
| `cb_output_tiles` | `depth_out` (tile-rows of `block_width` tiles) | 2 | compute packs tile-row `k+1` while the writer drains tile-row `k` |

Both are host constants (`DEPTH_IN`, `DEPTH_OUT`) used in exactly two places: the CB `total_size` and `per_col_tile_bytes`.

### Mechanism caps

| Mechanism | Cap on which extent | Clamp | What happens unclamped |
|-----------|--------------------|-------|------------------------|
| Fast tilize path eligibility: `block_width_tiles < 256`, 32×32 output tiles, half-sync DEST, input Float32/Float16_b, output not Float32 (`tilize_helpers.inl:65-69`) | `block_width` | `block_width_cap ≤ 255` | silent fallback to the standard `tilize_block` path: correct but slower (a perf cap, not a correctness cap) |
| Helper capacity asserts: input CB ≥ `block_width_tiles` pages, output CB ≥ `block_width_tiles` pages (`tilize_helpers.inl:199-205`) | CB capacity vs `block_width` | capacity = `depth * block_width` by construction | `ASSERT` under `--dev`; deadlock in release (reader waits on reserve, compute waits on a block that never fits) |
| CB ring-wrap: pointers reset only on an exact `fifo_limit` hit (`ttnn-cb-memory-fundamentals.md` §CB Ring-Wrap Invariant) | the pushed / popped quantum on the ragged last column block | every push/pop is the nominal `block_width` pages; the tail narrows only the NoC transfers (reader reads `last_block_width * 32` elements per stick; writer writes `last_block_width` tiles) and compute tilizes the nominal width over the stale columns | pushing `last_block_width` walks the write pointer past `fifo_limit` into neighbouring L1: wrong data, no hang |
| NoC read alignment: a stick segment's source byte offset `col * 32 * in_elem_bytes` and its L1 destination must share the input buffer's alignment (`ttnn.get_dram_alignment()` for DRAM — 32 bytes on Wormhole, 64 bytes on Blackhole; `ttnn.get_l1_alignment()` = 16 bytes for L1) | `block_width`, column-block starts, and (later) column-group boundaries | `col_align_tiles = ceil(align_bytes / (32 * in_elem_bytes))`; `block_width` and every column start are multiples of `col_align_tiles` (bf16: 64 bytes per tile-column ⇒ 1 on both archs; uint8 on Blackhole ⇒ 2) | NoC returns bytes from the aligned-down address: shifted data, no error |
| Asymmetric (row-page) helper mode hard-codes 32 rows per block (`tilize_helpers.inl:139-143`, `:200`, `:220`) | `tile_h` when `< 32` | the design uses symmetric (tile-sized page) mode only | tiny tiles would wait for 32 row-pages per tile-row: hang or mis-grouped rows |
| Fast tilize truncates fp32 → tf32 in DEST (`tilize_helpers.hpp:104`) | not an extent — the fp32 → fp32 / 32-bit integer datapath | fp32 → fp32 and int32/uint32: `Fp32Mode::Lossless` + `fp32_dest_acc_en=True` + `UnpackToDestFp32` on `cb_input_sticks` (static-asserted at `tilize_helpers.inl:107-115`) | 13 low mantissa bits lost; golden requires `exact` for fp32 → fp32 |
| `compute_kernel_lib::tilize` asserts `num_blocks > 0` (`tilize_helpers.inl:93`) | per-core row assignment | kernels are placed only on the `min(R, N)` cores that `split_work_to_cores` gives work | `ASSERT` / undefined behaviour on an idle core |

### Regimes

Named memory boundary for "minimum": **DRAM (or the input/output buffer's own `BufferType`) — the input crosses it once as a read and the output crosses it once as a write; zero cross-core traffic.** A pure re-lay reads nothing twice and has no operand shared between blocks, so every sensible regime can reach the minimum in bytes. The regimes differ in occupancy, transaction shape and residency.

Selection function (host, exact; evaluated in this order):

```
if input.layout == TILE:                                   -> retile_l1_facewalk        (built, Refinement 2)
elif in_is_L1_sharded or out_is_L1_sharded:
    if resident_ok(side) for every sharded side:          -> sharded_resident          (built, Refinement 1)
    else:                                                  -> sharded_accessor          (built, Refinement 1)
elif out_is_DRAM_sharded or in_is_DRAM_sharded:           -> sharded_accessor          (built, Refinement 1)
elif grid_2d_split(R, C, N).num_col_groups > 1:            -> grid_2d_split             (built, Refinement 5)
else:                                                      -> row_split_interleaved     (built)
```

**`load_block` variant `bank_coalesced` (built, Refinement 6).** Inside `row_split_interleaved` / `grid_2d_split`, a DRAM-interleaved input whose every Tensix core reads whole sticks (`block_width == C`, one column block) of at most `BANK_COALESCE_MAX_STICK_BYTES` = 256 bytes swaps `StickProducer` for the bank-coalesced reader. Stick page `p` sits in bank `p mod NB` at offset `(p div NB) * page`, so one NoC read per bank fetches a run's sticks from that bank into a reader-private staging ring. NoC loopback moves then scatter each stick into the tilize layout of the `cb_input_sticks` slot. Compute, the writer and the CB quanta are unchanged, except that the quantum on this path is `BANK_COALESCE_QUANTUM_ROWS` = 2 tile-rows. Same bytes at the DRAM boundary; per 64-stick unit, about `NB` reads of 5–6 pages instead of 64 one-page reads. See `l1_ledger.md` → Bank-coalesced.

**`load_block` co-read (built, Refinement 8).** When every Tensix core's walk is ONE position (one tile-row of one column block: tiny work such as [1,1,128,64] on 8 cores or [1,1,32,2048] on 64), nothing overlaps across tile-rows and the op is a latency chain: read issue → read landing → tilize → write. Its longest link was the stick-read issue on NCRISC, ~45 cycles per NoC read (~1.46 µs for 32 reads of a ~3.0 µs op), while BRISC idled in `cb_wait_front`. Co-read keeps `StickProducer` on NCRISC for the first `tile_h − co_read` sticks of the tile-row. BRISC (NoC1) reads the last `co_read = CO_READ_SHARE · tile_h` sticks straight into `cb_input_sticks`' first slot (the CB is empty, so that is where NCRISC's reserve lands), then raises a core-local program semaphore after its read barrier. NCRISC waits for that flag (`CoReadLanded`, its `Fill` hook) before the push and re-arms it. NCRISC stays the CB's only producer. Both halves come from one helper, `read_tile_row_sticks`, in the same per-core rotated stick order. It is gated by input `BufferType` on stick-segment bytes: DRAM ≤ 128 bytes (NoC1 reads DRAM poorly once reads stop being issue-bound), and L1 unbounded. Same bytes at every boundary; no CB added.

`resident_ok(side)`: shard spec is legacy-2D or ND with a 2-D equivalent, shard width a multiple of 32 elements and shard height a multiple of `tile_h`. Phase 0's `validate()` refuses everything that does not reach `row_split_interleaved`, and within it refuses `tile_grid ∈ {short_wide, square_large}`.

| Regime | Status | Predicate | Block | Data movement vs. minimum | What a bigger block buys |
|--------|--------|-----------|-------|---------------------------|--------------------------|
| `row_split_interleaved` | **built** | ROW_MAJOR input; both sides `TensorMemoryLayout::INTERLEAVED`; `num_col_groups == 1` | `block_height = core_row_tiles` × `block_width = balanced_width(C, block_width_cap)` | **minimum**: each stick byte read once (in `num_col_blocks` segments per stick), each output tile written once, 0 cross-core bytes. Transactions: `R * tile_h * num_col_blocks` reads of `block_width * 32 * in_elem_bytes` bytes (the stick is its own DRAM page, so a narrow `W` gives small reads whatever the split) + `R * C` tile-page writes | larger `block_width`: fewer, larger stick-segment reads (`R * tile_h * num_col_blocks` transactions; the one-core issue rate of ~8–9 M transactions/s is the bound on narrow sticks) and fewer column blocks. Fixed costs and intended frequency: tilize init/uninit + unpack/pack reconfig **once per kernel** (one helper call covering all blocks); read barrier **once per tile-row** (`tile_h` reads in flight); CB handshake once per tile-row per CB; pipeline fill/drain once per kernel. Larger `block_height` changes nothing: it only sets traversal order |
| `grid_2d_split` | **built (Refinement 5)**; was deferred: the task fixes Phase 0 SUPPORTED `tile_grid` at `{single_tile, small, tall_narrow}`. On those, the row split's makespan `ceil(R/N) * C` tiles per core already equals the 2-D optimum within one tile-row (tall_narrow: `R ≥ 16C`), and it keeps stick segments at full width. It is reachable as a pure knob-turn: kernels already take `(row_start, core_row_tiles, col_start, core_col_tiles)` RT args, so only the host assignment changes | `num_col_groups > 1` under the rule below (fires on `short_wide` and `square_large`, and whenever `R < N` with `C ≥ 2 * col_align_tiles`) | `block_height = core_row_tiles` × `block_width = balanced_width(core_col_tiles_max, cap)` | **minimum bytes** (same as row split, 0 cross-core bytes). Adds transactions: each stick is cut into `num_col_groups` segments of `core_col_tiles * 32 * in_elem_bytes` bytes (short_wide_canonical, C = 64 over 64 Tensix cores: 64-byte reads, 32 per tile) | same fixed costs as row split; additionally a wider `core_col_tiles` per group means fewer, larger stick-segment reads. That is what the transaction-size perf lamp in the row measures |
| `sharded_resident` | **built (Refinement 1)**; was deferred: sharded placement is outside the Phase 0 rectangle. It is placement of logical shards the split already defines, not a new algorithm: HEIGHT cuts `tile_row` and WIDTH cuts `tile_col` (both independent), BLOCK cuts both. Reachable because the kernels already work per core rectangle and the CBs are two plain FIFOs that can be backed on a shard buffer | an L1-sharded side with `resident_ok`; core assignment = that side's shard grid (output shard grid wins when both sides are sharded) | **one block = the whole resident shard**: `block_height = shard_h / tile_h`, `block_width = shard_w / 32` (valid extent of a partial final shard passed as RT extents; nominal CB quanta kept). The streamed side's `block_width` is clamped by `block_width_cap` | resident input shard: 0 NoC bytes on the input side (`cb_input_sticks` backed on the input shard via `ttnn.cb_descriptor_from_sharded_tensor`; the reader only publishes pages). Resident output shard: 0 NoC bytes on the output side (compute packs straight into `cb_output_tiles` backed on the output shard). Both resident with the same spec: **zero data movement**, compute only. Below the DRAM-boundary minimum, because the data never leaves L1 | the whole shard is one helper call: init once, one CB publish, no per-block handshake |
| `sharded_accessor` | **built (Refinement 1)**; was deferred for the same reason as `sharded_resident`. Covers crossovers whose shard is not tile-aligned, cross-spec (height in, width out), ND specs without a resident mapping, and DRAM-sharded output. Reachable because reader and writer already address through `TensorAccessor`, which resolves sharded pages (legacy 2D and ND) with the same `get_noc_addr(page, offset)` call | every sharded case not `sharded_resident` | as `row_split_interleaved`, over the assignment in the selection note (output shard grid if L1-sharded, else input shard grid, else interleaved split) | minimum at the named boundary; remote shards cross the NoC once (cross-spec: each input stick segment crosses from its owner's L1 to the producing Tensix core once) | as `row_split_interleaved` |
| `retile_l1_facewalk` | **built (Refinement 2)**; was deferred: TILE input is outside Phase 0 (`in_tile_height = "none"`). Reachable because only `load_block` changes: compute and writer are identical, and the extra CB is reader-private | `input.layout == TILE` (any `in_tile_height`, `tile=` required) | `block_height` along R rounded to `row_align = max(1, in_tile_h / tile_h)` output tile-rows, so an input tile-row never straddles two Tensix cores or blocks; `block_width` as row split | **minimum**: input tiles read whole (one page-sized NoC read per input tile, e.g. 2048 bytes), each read once; face rows re-laid into sticks by an L1-local copy on the reader RISC-V (no DRAM, no ROW_MAJOR tensor materialized); output written once | as row split, plus the staging copy is amortized over whole input tile-rows |
| `retile_compute_untilize` | **rejected**: superseded by `retile_l1_facewalk`. It needs a second compute phase (untilize → RM staging CB → tilize) with an init switch per block and a third CB on the compute path. It also sends tiny tiles through the untilize LLK path that feature_spec.py records as arch-gated upstream (retile skipped on Wormhole). It would be built *instead of* the facewalk, not on top of it | — | — | minimum bytes, +1 compute phase | — |
| `retile_dram_facerow_reads` | **rejected** (dead end): superseded by `retile_l1_facewalk`. One NoC read per face row = `2 * in_tile_h` reads of 16 elements (32 bytes at bf16) per input tile: transaction-bound (`split_reader`: 32-byte transactions ~6× slower than 2 KB) | — | — | minimum bytes, `~32×` the transaction count | — |
| `stick_granular_row_pages` | **rejected** (dead end): superseded by `row_split_interleaved`'s tile-row quanta. The helper's `TilizeGranularity::ROW` issues a barrier per stick (`tilize_helpers_dataflow.inl:148-157`) — the one-read-one-barrier trap (`double_buffer`: 2.78× slower on one core) — and hard-codes 32 rows (tiny-tile incompatible) | — | — | minimum bytes; one barrier per stick | — |

> **Implementer update (Refinement 2 — tile geometry).**
> - **Tiny tiles** (`tile_h` 16/8/4/2/1) needed no kernel change: both CBs already carried `TileDescriptor(tile_h, 32)`, the helper leaves the fast path on its own (`dfb_has_32x32_tiles`), and the reader groups `tile_h` sticks per tile-row. The host allocates the output through a `TensorSpec` carrying the tile (`tilize._allocate_output`, one overload per placement), and `QUANTUM_MIN_TILES` now counts full 32-row tile equivalents (`quantum_min_tiles = QUANTUM_MIN_TILES * 32 / tile_h`), so a `tile_h = 1` quantum is not 64-byte sticks. On [1,1,16384,64] (64 Tensix cores, WH): `tile_h` 16 = 24.0 µs, 8 = 24.9 µs (the 32×32 path is 25.5 µs), 1 = 62.7 µs (64-byte output pages: writer-bound).
> - **Retile** is `read_retile` in `tilize_reader.cpp` (building blocks in `tilize_stick_reads.hpp`: `FaceWalk`, `RowSlotWriter`, `retile_source_of`). The unit is `row_align` output tile-rows fed by `unit_in_rows = max(1, tile_h / in_tile_h)` input tile-rows of one image. Input tiles are read whole (one page-sized NoC read each) into the reader-private `cb_retile_staging` ring, with `RETILE_STAGE_DEPTH - 1` units prefetched under transaction ids. A resident input TILE shard backs `cb_retile_staging` and is read in place. The row split and the walk rotation are in units of `row_align`. H-padded TILE inputs and output shards that cut input tile-rows fall back to `row_align = 1` (correct, with read amplification). Compute and writer are unchanged.
> - **Face walk mover** (`RETILE_FACEWALK_NOC`, default on). The design's "L1-local copy on the reader RISC-V" measured ~8 cycles per word (blocking L1 loads): 93.5 µs for retile 32→16 on [1,1,16384,64], against 23 µs with the copy stubbed. Each face row (16 elements, 32 bytes at bf16) is now moved by a NoC loopback read on the same Tensix core (`noc_async_read_one_packet_with_state`, own transaction id, drained before every `cb_input_sticks` push and before a staging slot is refilled): 38.3 µs. The RISC-V copy stays a live knob.

> **Implementer update (Refinement 4 — padding).** No new regime: padding is a `load_block` feature on the stick reader, on every placement that streams the input.
> - **Host.** `tilize._resolve_padding` gives the padded shape `P` (auto: the last two dims rounded to (`tile_h`, 32), ranks 0 / 1 left-expanded to rank 2 first; explicit: `output_padded_shape`) and the input left-expanded to `P`'s rank. The output is allocated at `P`, and the program's `R`, `C` come from it. The returned tensor is a zero-copy view at the input's logical shape: `ttnn.reshape(out, logical, padded)` resolves to `tt::tt_metal::view` for a TILE tensor whose padded last dim is unchanged. `_logical_view` checks that the buffer address is unchanged and raises otherwise, so the call can never become a second dispatch. The fill value is packed per input dtype as RT arg 10 (program-cache friendly).
> - **Reader.** `PadMap` is the per-image stick map `r → (image = r / (P[-2] / tile_h), h = (r % (P[-2] / tile_h)) · tile_h + s)`, with a mixed-radix leading-dim decode (only the dims up to the outermost one the pad grows). `StickProducer::issue_row_padded` reads the existing sticks' data bytes only, then fills: the W-tail band of the existing sticks, and the trailing whole pad sticks (the H tail, whole pad tile-rows / images) as one contiguous range. `PadFill` does the stores: `fill_l1_range` or a once-per-tile-row band split for short ranges, and NoC loopback copies from the reader-private 1 KiB `cb_pad_source` (filled once) for long ranges, under their own transaction id and drained before every push. With one column block per core the W-tail band persists across passes through the CB ring (`PAD_W_TAIL_PERSIST`). Compute and writer are unchanged.
> - **Regime interactions.** A padded input never resides: its stick layout is not the tilize layout of the padded grid. A resident output shard over `P` works unchanged (`padded_to_height_sharded`). A legacy-sharded output `MemoryConfig` without a shard spec takes the input's grid and orientation, re-derived over `P` (`_resolve_output_memory_config`). Tiny tiles pad against `tile_h`. The split reader and the parked Refinement 3 NoC levers are off on the padded path.
> - **Not built.** Retile × a pad that fills is an EXCLUSION: the face walk would have to clamp its staging reads and walk to the input's tile-rows / tile-columns, then fill after the walk lands. An auto pad of an aligned TILE input fills nothing and runs the plain retile path. Growing an inner leading dim (e.g. [2, 3, …] → [3, 4, …]) raises `NotImplementedError`. TTNN's logical view maps logical image k to padded image k, so no buffer can satisfy both `to_torch(out) == x` and the `F.pad` padded readback. That was verified on host with a correct `F.pad` buffer.

> **Implementer update (Refinement 5 — `grid_2d_split` + `low_l1`).** Built as a host-only knob-turn: kernels are unchanged (they already took per-core column ranges).
> - **Rule.** `tilize_program_descriptor.grid_2d_split` transcribes the pinned rule with one measured amendment: the busiest core's cost is `rows * (cols + ROW_COST_TILES)`, `ROW_COST_TILES = 1.5` tile-equivalents per tile-row, fitted from row-split timings (`t ≈ rows · (a + b · cols) + c` on WH, `a / b ≈ 1.5`). The pure tile count (`ROW_COST_TILES = 0`, still a live knob) ignores that every tile-row costs `tile_h` stick reads whatever its width. It picked 1-column groups where the row split already filled the grid: [4,3,256,96] 8.9 → 11.0 µs; [1,1,2080,2048] 92 → 181 µs (65 tile-rows × 64-byte reads per core). With 1.5 both keep the row split or a wide 2-D split. `g_c == 1` falls through to the unchanged `split_work_to_cores` row split, so the perf-focus path is byte-identical.
> - **Occupancy (WH, 64 Tensix cores).** short_wide_canonical [1,1,32,2048] 64 cores (was 1), 14.2 → 3.7 µs (ref 3486 ns). short_wide_l1_forcing [1,1,32,8192] 64 cores, 30.8 → 7.2 µs (ref 7142 ns). [1,1,64,4096] 64 cores (2 × 32), 20.2 → 7.3 µs. square_large [1,1,2048,2048] 64 cores; the rule keeps the row split on WH (tie → wider segments) and gives 64 × 2 on a 130-core grid. tall_narrow_grid_scale gives 64 × 2 on a 130-core grid (R < N).
> - **Transaction-size lamp.** `MIN_GROUP_COL_TILES` (a column-group floor) measured 1 / 2 / 4 / 8 / 16 tile-columns: [1,1,32,2048] 3828 / 3969 / 4068 / 4368 / 5395 ns; [1,1,32,8192] 7227 / 7376 / 7376 / 8134 / 8197 ns. Maximum participation wins, so the floor is parked at 1.
> - **Quantum knob.** On short_wide every core owns one walk position, so `rows_per_quantum` is pinned to 1 (`max_positions // DEPTH_IN = 0`) and `QUANTUM_MIN_TILES` cannot engage. The one-position serialization (read → tilize → write) was attacked instead with `PIPELINE_MIN_POSITIONS`: cut a single-position core's columns into more blocks. Ungated it lost on short_wide ([1,1,32,8192] 7.5 → 10.0 µs at 2 blocks) because segments narrow. Gated at ≥ 2 KiB segments (`PIPELINE_MIN_SEGMENT_BYTES`) it wins on square_large: [1,1,2048,2048] 92.3 → 87.0 µs (median of 3). Default on (2 positions, 2 KiB).
> - **low_l1.** `CB_BUDGET_BYTES[True]` = 65536 bytes, admitted as is. The golden A/B and `test_low_l1_ab_bit_identical` are bit-identical.

| `full_width_unbounded` | **rejected**: superseded by the capped `block_width`. `block_width = C` unconditionally makes both CBs O(W): 256 tiles at fp32 = 1 MiB per side on `low_l1_forcing_width` [1,1,32,8192], which cannot fit in 1.5 MB of L1 at any depth | — | — | minimum bytes; OOM | — |
| `single_core` | **rejected**: superseded by `row_split_interleaved` (single-core Phase 0 is disallowed; identical bytes, 1/N occupancy) | — | — | minimum bytes, 1 Tensix core | — |
| dependent-axis split + cross-core combine | **rejected — does not exist for this op**: no axis is dependent (no output tile reads another's input), so there is nothing to combine and no residency to gain; the op already reads each input byte exactly once | — | — | — | — |

> **Implementation note (Refinement 1).** The selection is realized per side, not per call (`tilize_program_descriptor._core_assignment`). The core assignment comes from the first of: a `resident_ok` L1-sharded output, a `resident_ok` L1-sharded input, or the interleaved row split. Each side is then resident iff its own shard rectangles equal the assigned ones, and streamed through `TensorAccessor` otherwise. The mixed cases are therefore half-resident: an L1 input shard with a DRAM / interleaved / DRAM-sharded output keeps the input resident (the reader only publishes pages), and cross-spec height-in / width-out keeps the output resident and reads the input remotely. `resident_ok` also requires the shard width in tiles to fit the streamed partner CB (`shard_w / 32 <= min(255, CB_BUDGET_BYTES[low_l1] // per_col_tile_bytes)`), since the whole shard width is one block. No side re-reads its own local shard through an accessor. The DRAM-sharded and ND-without-2-D-equivalent cases, which have no Tensix owner, use the row split over the full grid rather than "the sharded side's core assignment". Streamed WIDTH / BLOCK / ND Layout::ROW_MAJOR inputs have shard-width pages, so the reader splits each stick-segment read at page boundaries (`pages_per_stick` CT arg; one read per stick when the page is the whole stick). A resident side pins the tile-row walk to shard order (row rotation 0). The in-tile-row stick rotation still spreads reads, and the writer now rotates its tile order inside a tile-row by the same per-core amount (the write twin), which moved HEIGHT-sharded in → DRAM [1,1,2048,512] from 18746 ns to 16502 ns (WH, 64 Tensix cores).

`grid_2d_split` assignment rule (pinned now, so the refinement is a transcription):

```
N         = grid.x * grid.y                       # device.compute_with_storage_grid_size()
col_units = ceil(C / col_align_tiles)
choose (g_r, g_c) with 1 <= g_r <= min(R, N), 1 <= g_c <= min(col_units, N // g_r) minimizing
    makespan = ceil(R / g_r) * ceil(col_units / g_c) * col_align_tiles        # tiles on the busiest core
tie-break 1: larger ceil(col_units / g_c)   (wider stick segments)
tie-break 2: fewer cores (g_r * g_c)
rows: split_work_to_cores-style balanced over g_r groups; columns: balanced col_units over g_c groups
num_col_groups = g_c
```

Row-split Phase 0 is exactly this rule with `g_c` forced to 1. Regime-pinned tests required when it lands: `short_wide_canonical` [1,1,32,2048], `short_wide_l1_forcing` [1,1,32,8192], `square_large` [1,1,2048,2048], `square_large_from_leading_dims` [8,1,249,2048] — the last reaches R = 64 only through the leading-dim fold.

### Traffic ranking

Operand-reuse check: the op has ONE operand. It varies along both `tile_row` and `tile_col`, so no split makes any operand reuse-shared. There is no broadcast row.

| Rank | Candidate split | DRAM crossings (in / out) | Cross-core traffic | Transaction shape | Verdict |
|------|-----------------|---------------------------|--------------------|-------------------|---------|
| 1 | cut `tile_row` only (row split) | 1 / 1 | 0 | full stick segments of `block_width * 32 * in_elem_bytes` bytes, the largest any split can issue | **chosen** (Phase 0) |
| 2 | cut both (2-D) | 1 / 1 | 0 | segments shrink to `core_col_tiles * 32 * in_elem_bytes` bytes | `grid_2d_split`, deferred (needed only where rows under-fill the grid) |
| 3 | cut `tile_col` only | 1 / 1 | 0 | smallest segments, `ceil(C/N) * 32 * in_elem_bytes` bytes | subsumed by rank 2 (`g_r = 1` is one point of its search) |
| — | split a dependent axis + combine | n/a | n/a | n/a | no dependent axis exists |

All candidates tie on bytes at the minimum. The tiebreaker is transaction shape (fewer, larger reads), which ranks the row split first. Occupancy is recorded separately below and does not re-decide the split.

Occupancy reached by the built regime (row split, `N` = usable compute Tensix cores): `min(R, N)` Tensix cores.

| `tile_grid` | Tensix cores reached | Full grid? |
|-------------|----------------------|------------|
| `single_tile` | 1 | the whole work is 1 tile |
| `small` | `min(R, N)` | `R * C < 256` tiles; too few tiles for the axis choice to matter (task spec) |
| `tall_narrow` | `min(R, N)` | yes whenever `R ≥ N`; at `R < N` the busiest core holds `C ≤ R/16` tiles, which is within one tile-row of the 2-D optimum. On Blackhole (110–130 cores), `tall_narrow_grid_scale` (R = 64) reaches 64 Tensix cores at makespan 2 tiles, the same makespan the 2-D rule finds |
| `short_wide` | `R` (1 or 2) | **no** → refused in Phase 0 |
| `square_large` | `min(R, N)` | not balanced 2-D → refused in Phase 0 |

Stall-shadow check. Compute waits on the reader once per tile-row, and the writer waits on compute once per tile-row. With `depth = 2`, the next tile-row's reads are already in flight during that wait, and no other work in the op is independent of the data being waited for. The reader's own read barrier is the remaining shadow: the next tile-row's reads could be issued before it, and the in-flight-depth perf lamp measures that. No algebraic reorder applies: the op does no arithmetic, so floating-point order is not a concern.

### Block schedule

Per Tensix core: the rectangle `[row_start, row_start + core_row_tiles) × [col_start, col_start + core_col_tiles)` of output tiles (Phase 0: `col_start = 0`, `core_col_tiles = C`).

```cpp
for (uint32_t row_block_idx = 0; row_block_idx < num_row_blocks_this_core; ++row_block_idx) {      // Phase 0: 1
    for (uint32_t col_block_idx = 0; col_block_idx < num_col_blocks_this_core; ++col_block_idx) {
        load_block(row_block_idx, col_block_idx);     // reader  (NCRISC)
        tilize_block(row_block_idx, col_block_idx);   // compute (TRISC0/1/2)
        store_block(row_block_idx, col_block_idx);    // writer  (BRISC)
    }
}
```

This is a logical schedule. The three kernels run asynchronously, pipelined at tile-row granularity through the two depth-2 CBs.

| Block operation | Block shape it acts on | Resident across it | Intended fixed-cost frequency |
|-----------------|------------------------|--------------------|-------------------------------|
| `load_block` | `block_height` tile-rows × `block_width` tile-columns. Each tile-row = `tile_h` stick segments of `valid_width * 32` elements, landed at the nominal L1 stride `block_width * 32 * in_elem_bytes` | nothing across blocks; within a block, one quantum of `rows_per_quantum` tile-rows at a time in `cb_input_sticks` | one `cb_reserve_back` + `rows_per_quantum * tile_h` async reads + **one read barrier** + one `cb_push_back(rows_per_quantum * block_width)` **per quantum**. `TensorAccessor` constructed once per kernel |
| `tilize_block` | same rectangle; the helper processes it as `block_height` helper-blocks of `block_width` tiles | the tilize LLK configuration (unpack/pack formats, fast-tilize mode) stays resident across **all** blocks of the core | **tilize init + reconfig + uninit once per kernel**: realized as ONE `compute_kernel_lib::tilize<block_width, …>(core_row_tiles * num_col_blocks_this_core)` call. Compute cannot tell which column block a tile-row belongs to, and every quantum is the nominal `block_width` |
| `store_block` | same rectangle; per tile-row, `valid_width` output tile pages at tile indices `r * C + col` | nothing | one `cb_wait_front(rows_per_quantum * block_width)` + `valid_width` tile writes per tile-row + **one write flush** + one `cb_pop_front` **per quantum** |

`valid_width = block_width` except on the ragged last column block (`last_block_width`). Only NoC transfers narrow; push/pop counts stay nominal (Mechanism caps, ring-wrap row).

> **Verifier update (Phase 0 review).** The `tile_row` streaming window is now a knob. `rows_per_quantum = min(ceil(QUANTUM_MIN_TILES / block_width), max_positions // depth_in, budget cap)`, floored at 1, sets how many consecutive walk positions share one CB push / pop, one read barrier and one write flush. Before, the window was hard-wired to one tile-row, which is only 2 tiles on the perf-focus shape. Compute is unchanged: one `block_width` tile-row per helper block, one helper call per kernel. Only the kernel's final quantum can be partial. Ledger: `l1_ledger.md`.

### Perf lamps

| Lamp | Why the default may be wrong here | Nearby alternative to measure |
|------|-----------------------------------|-------------------------------|
| Reader issue-rate (split-reader) | Narrow sticks (perf-focus [1,1,16384,64]: 128-byte stick segments, R = 512 ⇒ 8 tile-rows = 256 reads per Tensix core on 64 cores) make the reader issue-bound (~8–9 M transactions/s per RISC-V, `split_reader`). Meanwhile the writer RISC-V issues only 16 tile writes | split each tile-row's `tile_h` stick reads across NCRISC and BRISC. Two producers cannot share a CB (ownership invariant), so the realization is two input CBs (`cb_input_sticks_even` / `_odd`, one producer each) that compute alternates between. Measure against the default on the perf-focus entry and report Tensix cores reached (`min(R, N)`) alongside the duration |
| Read in-flight depth (overlap) | One barrier per tile-row serializes the next tile-row's issue behind the current one's completion; on narrow sticks the barrier wait is a large share | reserve two tile-row slots and issue tile-row `k+1`'s reads before tile-row `k`'s barrier (barrier every second tile-row), or `depth_in = 3` |
| Grid synchronization on tiny work | `small` / `single_tile` spread over `min(R, N)` cores with one tile-row each; per-core launch cost may dominate the ~1 µs of work | cap the participating core count (e.g. `≥ 2` tile-rows per core) and compare on [1,1,128,64] (perf-focus entry 4) |
| Column-block width | Default = coarsest fitting (`≤ 64` tiles at bf16, 4 KB reads). Past ~2 KB per transaction the gain flattens (`split_reader`), and a very wide tile-row lengthens pipeline fill before compute starts (overlap) | `block_width_cap` 32 vs 64 on a tall_narrow shape with `C > 32` |
| Read in-flight depth / NoC placement / write batching (Refinement 3, measured) | Measured on [1,1,16384,64], WH, 64 Tensix cores (baseline 25.3–25.6 µs). The shape is bound by aggregate DRAM throughput for its transaction mix: 32 Tensix cores take 30.0 µs (only 17 % slower), and reads-only 16.1 µs + writes-only 19.2 µs − floor 8.1 µs ≈ full. Quantum × depth × read-ahead co-tune, eager publish, write-ahead, write bank-spreading, cheaper read issue (bank-stride addressing) and splitting either stream across both NoCs all measured flat or slower (reads on NoC1 up to +40 %). Every lever is parked at a byte-identical default (`tilize_program_descriptor.py` records the numbers) | fewer, larger DRAM reads on narrow sticks (Refinement 6, bank-coalesced reads on top of the parked bank-stride addressing) |
| Grid synchronization on tiny work → co-read (Refinement 8, measured) | One-position walks are a serial chain whose longest link is one RISC-V issuing every stick read | built: co-read (above). [1,1,128,64] 3122 → 2307 ns (reference 2294), [1,1,2048,32] 4307 → 3413, [1,1,32,2048] 3659 → 3445 (reference 3486). The NoC1 share of DRAM reads of 192 bytes or more loses (+4.6 % to +31 %), hence the gate |
| Transaction size vs occupancy (applies when `grid_2d_split` lands) | Maximum participation on `short_wide` gives 64-byte stick segments (C = 64 on 64 cores) | a floor `core_col_tiles ≥ col_floor` (e.g. 8 tiles = 512 bytes) with fewer participating cores |

## Dataflow Strategy

| Stage | Format | Mechanism | Notes |
|-------|--------|-----------|-------|
| DRAM → L1 (`cb_input_sticks`) | ROW_MAJOR stick segments, input dtype | reader (NCRISC, NoC0): `TensorAccessor(input)` `get_noc_addr(stick_idx, byte_offset)` + `noc_async_read` of `valid_width * 32 * in_elem_bytes` bytes per stick, `tile_h` sticks per tile-row, one barrier per tile-row | stick index `stick_idx = r * tile_h + s` (Phase 0). The pad refinement replaces it with the per-image map `r → (image = r / h_tiles_img, h = (r % h_tiles_img) * tile_h + s)`, reading only `h < H` and `image < num_input_images` and filling the rest with `fill_l1_range<elem_bytes>` (`l1_helpers.hpp:89-90`). The W tail is filled from byte `(W − col*32) * in_elem_bytes` of the segment |
| `cb_input_sticks` → DEST → `cb_output_tiles` | stick-major tile-row (`block_width` tile-sized pages holding `tile_h` sticks of `block_width * 32` elements) → TILE (faces TL, TR, BL, BR) in the output dtype | compute: `compute_kernel_lib::tilize` (unpack-tilize, datacopy, pack). Fast tilize at 32×32 bf16/fp32 input; standard `tilize_block` otherwise | cast happens at pack (output CB format = output dtype) |
| L1 (`cb_output_tiles`) → DRAM | TILE pages, output dtype | writer (BRISC, NoC1): `TensorAccessor(output)` `noc_async_write_page(tile_idx, …)` per valid tile, one barrier per tile-row | tile index `r * C + col` (padded C when padded) |

Placement per scheme (the same regime rows seen as the `out_scheme` / `shard_api` TARGET axes):

| Placement | Axis it cuts | Character | Class | Consumed how |
|-----------|--------------|-----------|-------|--------------|
| `TensorMemoryLayout::INTERLEAVED` (DRAM or L1) | none (pages round-robin over banks) | — | built (DRAM) / knob-turn (L1) | `TensorAccessor` both sides |
| `HEIGHT_SHARDED` | `tile_row` | independent | knob-turn (placement) | core assignment = shard grid; resident side's CB backed on its shard (`ttnn.cb_descriptor_from_sharded_tensor`), never re-read over the NoC through a `TensorAccessor` |
| `WIDTH_SHARDED` | `tile_col` | independent | knob-turn (placement; uses the per-core column range the kernels already take) | same |
| `BLOCK_SHARDED` (ROW / COL orientation) | both | independent | knob-turn (placement) | same; COL_MAJOR only changes the shard → core map on host |
| ND shard spec | any | independent | knob-turn | `TensorAccessor` resolves ND pages; resident when the ND spec has a tile-aligned 2-D equivalent |
| DRAM-sharded output | — (DRAM banks) | — | knob-turn | writer via `TensorAccessor` |

No regime needs Tensix-to-Tensix synchronization (co-read's semaphore is core-local: BRISC → NCRISC on one Tensix core). Cross-spec sharding only reads a remote shard's L1 through the `TensorAccessor` NoC address, with no semaphore, because the input is immutable during the op.

## Work Distribution

| Field | Value |
|-------|-------|
| Work unit | an output tile-row (`block_width`-wide slices of it are the CB quanta); a core's work is its rectangle, cut into `num_col_blocks_this_core` blocks of `core_row_tiles × block_width` |
| Grid | `grid = device.compute_with_storage_grid_size()` (runtime, never hardcoded); `N = grid.x * grid.y` |
| Per-core work | `(num_cores, all_cores, core_group_1, core_group_2, rows_g1, rows_g2) = ttnn.split_work_to_cores(grid, R, row_wise=True)`; cores enumerated group 1 then group 2, row-wise within each (`ttnn.corerange_to_cores(group, row_wise=True)`); `row_start` = running sum of `core_row_tiles`; `col_start = 0`, `core_col_tiles = C` (Phase 0) |
| Remainder | rows: group 1 gets `rows_g1 = ceil`, group 2 `rows_g2 = floor` (from `split_work_to_cores`); columns: `last_block_width` on the final column block, nominal CB quanta |
| Tile geometry | `R = prod(shape[:-2]) * ceil(shape[-2] / tile_h)` (per image, `ceil`; rank < 2 → leading product 1 after the pad synthesizes dims); `C = ceil(shape[-1] / 32)`. Computed from the PADDED shape once padding lands. Never `floor(batch * H / 32)` and never `shape[-2]` alone |
| Kernel placement | reader / compute / writer on `all_cores` only (the `min(R, N)` cores with work) |
| Program cache | buffer addresses are RT args only. CT args depend on (dtype pair, tile, `block_width`, accessor args), so repeated calls with the same config hit the cache (`PROGRAM_CACHE_CASES`) |

Regime selection is stated under Regimes. Regime-pinned tests are required for `grid_2d_split` (four shapes listed there) and for each sharded sub-case when those land.

## Circular Buffers

| Semantic Name | Index | Page Size | Num Pages | Sizing rationale | Format | Producer | Consumer | Lifetime |
|---------------|-------|-----------|-----------|------------------|--------|----------|----------|----------|
| `cb_input_sticks` | 0 | `in_tile_bytes = tile_h * 32 * in_elem_bytes` (2048 bytes, Phase 0); `TileDescriptor(tile_h, 32)` | `depth_in * block_width` | spans `tile_col` (`block_width`), streams `tile_row` (one tile-row per quantum), spans `stick_in_tile_row` (`tile_h`, inside the page) | input dtype (Float16_b in Phase 0); `UnpackToDestFp32` only for the fp32 → fp32 / 32-bit int refinement | reader | compute | whole kernel |
| `cb_output_tiles` | 1 | `out_tile_bytes = tile.get_tile_size(out_dtype)` (2048 bytes, Phase 0); `TileDescriptor(tile_h, 32)` | `depth_out * block_width` | spans `tile_col` (`block_width`), streams `tile_row` | output dtype (Float16_b in Phase 0) | compute | writer | whole kernel |

Compute config (Phase 0): `fp32_dest_acc_en = False` (16-bit DEST matches the Float16_b pages), `dst_full_sync_en = False` (half-sync is a fast-tilize requirement, `tilize_helpers.inl:67`), `unpack_to_dest_mode` default. Refinement 7 (`tilize_program_descriptor.NumericConfig`, the single source): `fp32_dest_acc_en = True` whenever either page is Float32 / Int32 / UInt32 / UInt8, and never for UInt16 (fp32 DEST scrambles it on WH). A 32-bit input's compute input CB(s) are tagged `UnpackToDestFp32` and compute CT arg 5 selects `Fp32Mode::Lossless`. bf16 → bf16 keeps the Phase 0 config byte for byte. An optional `compute_kernel_config` can only add fp32 DEST, and math fidelity / approx are no-ops. Inventory is two CBs, the minimum: tilize cannot run in place (`static_assert(input_dfb != output_dfb)`, `tilize_helpers.inl:87-88`), and the writer needs a source. Full ledger: `l1_ledger.md`.

## Block Operation Realization

| # | Block operation | Block shape | Helper? | Input CB (semantic name, pages, state) | Output CB (semantic name, pages) | CB state after |
|---|-----------------|-------------|---------|----------------------------------------|----------------------------------|----------------|
| 1 | `load_block` | `block_height × block_width` tiles, streamed as `block_height` tile-rows | custom block op (reader); `read_sticks_for_tilize` rejected, see API Mapping | DRAM input via `TensorAccessor` | `cb_input_sticks`: `block_width` pages pushed per tile-row, `block_height` pushes per block | `cb_input_sticks` holds ≤ `depth_in` tile-rows in flight |
| 2 | `tilize_block` | all blocks of the core in one call: `core_row_tiles * num_col_blocks_this_core` helper-blocks of `block_width` tiles | `compute_kernel_lib::tilize` (helper) | `cb_input_sticks`: waits/pops `block_width` pages per tile-row | `cb_output_tiles`: reserves/pushes `block_width` pages per tile-row | both CBs empty at kernel end |
| 3 | `store_block` | `block_height × block_width` tiles; writes `valid_width` tiles per tile-row | custom block op (writer) | `cb_output_tiles`: waits/pops `block_width` pages per tile-row | DRAM output via `TensorAccessor` | — |

## API Mapping

| Block operation | Type | Function | File:Line | Template Params / Args | Input CB | Output CB | Which params are block knobs |
|-----------------|------|----------|-----------|------------------------|----------|-----------|------------------------------|
| (startup) | helper prerequisite | `compute_kernel_hw_startup(icb, ocb)` | `tt_metal/hw/inc/api/compute/compute_kernel_hw_startup.h:106` | `(cb_input_sticks, cb_output_tiles)`, called once before the helper | — | — | none |
| `tilize_block` | helper | `compute_kernel_lib::tilize<block_width_tiles, input_dfb, output_dfb, InitUninitMode::InitAndUninit, WaitMode::WaitBlock, ReconfigureRegisterDatatypeMode::UnpackAndPackReconfigure, Fp32Mode::Fast, RemapMode::Configure>(num_blocks)` | `ttnn/cpp/ttnn/kernel_lib/tilize_helpers.hpp:187-197` (impl `tilize_helpers.inl:84-250`) | `block_width_tiles = block_width` (CT); `num_blocks = core_row_tiles * num_col_blocks_this_core` (RT); `total_input_pages` omitted (symmetric tile-sized pages, required for tiny tiles). Refinements: `Fp32Mode::Lossless` for fp32 → fp32 and 32-bit integers | `cb_input_sticks` (waits/pops `block_width` per helper-block, `inl:223-236`) | `cb_output_tiles` (reserves/pushes `block_width`, `inl:227-235`) | `block_width_tiles` (= `block_width`); `num_blocks` carries `block_height × num_col_blocks` |
| `load_block` | raw_api (dataflow) | `TensorAccessor::get_noc_addr(page, offset)`, `noc_async_read`, `noc_async_read_barrier`, `cb_reserve_back` / `get_write_ptr` / `cb_push_back`; later `dataflow_kernel_lib::fill_l1_range<val_size>` for pad | `tech_reports/tensor_accessor/tensor_accessor.md`; `ttnn/cpp/ttnn/kernel_lib/l1_helpers.hpp:89-90` | per tile-row: reserve `block_width`, `tile_h` reads of `valid_width * 32 * in_elem_bytes` bytes at L1 stride `block_width * 32 * in_elem_bytes`, one barrier, push `block_width` | DRAM | `cb_input_sticks` | `block_width` (stride and push quantum), `block_height` (tile-rows per block) |
| `store_block` | raw_api (dataflow) | `TensorAccessor`, `noc_async_write_page` (or `get_noc_addr` + `noc_async_write`), `noc_async_write_barrier`, `cb_wait_front` / `get_read_ptr` / `cb_pop_front` | `tech_reports/tensor_accessor/tensor_accessor.md` | per tile-row: wait `block_width`, `valid_width` tile-page writes, one barrier, pop `block_width` | `cb_output_tiles` | DRAM | `block_width` (pop quantum), `valid_width` |

Helpers considered and rejected:

| Raw entry | Candidate helper | File:line of the mismatch | Concrete reason |
|-----------|------------------|---------------------------|-----------------|
| `load_block` | `dataflow_kernel_lib::read_sticks_for_tilize<cb, TilizeGranularity::TILE>` | `ttnn/cpp/ttnn/kernel_lib/tilize_helpers_dataflow.inl:92-93, 117, 127` | The push quantum and the L1 stride are both derived from `row_bytes` (`width_in_tiles = round_up(row_bytes, tile_row_bytes) / tile_row_bytes`). Narrowing `row_bytes` for the ragged last column block therefore pushes `last_block_width` pages and breaks the nominal-quantum ring-wrap invariant (wrong data in release). There is no separate "nominal width" argument |
| `load_block` (pad refinement) | same | `tilize_helpers_dataflow.hpp:50-52` ("untouched rows contain stale data"), `tilize_helpers_dataflow.inl:121` (`start_page + block_row + row`) | No fill for the W tail / H tail / whole pad sticks. Stick index is assumed contiguous across tile-rows, which fails at an image boundary when `H % tile_h != 0` |
| `load_block` | `read_sticks_for_tilize<cb, TilizeGranularity::ROW>` | `tilize_helpers_dataflow.inl:148-157`; `tilize_helpers.inl:220` | One barrier per stick (the single-read trap); the paired asymmetric compute mode hard-codes 32 rows (tiny-tile incompatible) |
| `store_block` | none exists: the kernel_lib dataflow side has only `write_sticks_after_untilize` (`tilize_helpers_dataflow.hpp:129-135`), which writes ROW_MAJOR sticks, not tile pages | `tilize_helpers_dataflow.hpp:129-135` | Wrong output Layout (sticks, not tiles) |

What would close the gap: a `read_sticks_for_tilize` overload that takes a nominal `block_width_tiles` (stride + push quantum) separately from the valid `row_bytes`, plus an optional fill value and a stick-index functor. Recorded here; not a blocker.

## Key Risks and Gotchas

| Risk | Why it bites here | Mitigation in this design |
|------|-------------------|---------------------------|
| Ragged last column block overshoots the CB ring | `C` is not in general a multiple of `block_width`; pushing `last_block_width` pages breaks the exact-`fifo_limit` reset | nominal `block_width` push/pop always; only NoC transfers narrow; compute tilizes the stale tail columns and the writer never writes them |
| Idle Tensix cores trip `ASSERT(num_blocks > 0)` | `R < N` on `small` / `single_tile` | kernels placed on `all_cores` from `split_work_to_cores` only |
| Row count derived from `shape[-2]` alone | `square_large_from_leading_dims` [8,1,249,2048] reaches R = 64 only through the fold, and the H tail is per image | `R = prod(shape[:-2]) * ceil(shape[-2]/tile_h)`; the pad refinement maps `r → (image, h_tile)` per image |
| Column split picked by `split_work_to_cores` default | `row_wise=False` gives a column line of cores, ~2.9× slower on DRAM traffic (`noc_placement`) | `row_wise=True` |
| NoC misalignment of stick-segment offsets | 1-byte (uint8) and 2-byte (uint16) dtypes give 32- or 64-byte tile-columns, below Blackhole's 64-byte DRAM alignment | `col_align_tiles` clamp on `block_width` and column starts |
| fp32 silently truncated to tf32 | the helper's default `Fp32Mode::Fast` (`tilize_helpers.hpp:104`); golden requires exact fp32 → fp32 | Lossless + `fp32_dest_acc_en` + `UnpackToDestFp32` in the fp32 / int32 refinement; Page format column in the ledger flips with it |
| Tiny tiles leave the fast path, and the helper's asymmetric mode hard-codes 32 rows | `tile_height < 32` | symmetric tile-sized pages only; `TileDescriptor(tile_h, 32)` on BOTH CBs, so `unpack_tile_r_dim` = `tile_h` for the reader and the LLK |
| Program-cache miss | a buffer address in CT args, or per-call varying CT args | addresses as RT args only; CT args are a function of the config, not of the allocation |
| L1-interleaved / sharded tensors share the Tensix core's L1 with the CBs | `buffer = l1_to_l1` cases allocate tensor pages top-down while CBs grow from the base | `CB_BUDGET_BYTES` = 512 KiB (≈ 1/3 of L1) leaves the rest for tensors; `low_l1=True` = 64 KiB |
| Output allocated with the wrong tile geometry | `ttnn.allocate_tensor_on_device(shape, dtype, layout, device, mem)` defaults to 32×32 | Phase 0 is 32×32. The tiny-tile refinement must allocate through a `TensorSpec` carrying `tile` |
