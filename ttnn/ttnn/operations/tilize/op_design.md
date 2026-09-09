# Operation Design: tilize

## Overview

| Field | Value |
|-------|-------|
| Classification | data_movement (pure layout re-lay; no arithmetic) |
| Goal | Convert a ROW_MAJOR tensor to TILE layout on device, in one native dispatch, with the work spread over the whole core grid on every `tile_grid` geometry. |
| Math | `output_tile[r, c][i, j] = input[fold(r, i), 32*c + j]` — a bijection on addresses; every value and logical position is preserved (value-preserving cast where `dtype=` changes the storage format). |
| Mode | Derivative (LLK `tilize` / `fast_tilize` via `compute_kernel_lib::tilize`) |
| References | `.claude/references/blocking-model.md`, `.claude/references/l1-footprint-discipline.md`, `.claude/references/ttnn-cb-memory-fundamentals.md` (§"Tilize Data Flow Pattern"), `ttnn/cpp/ttnn/kernel_lib/tilize_helpers.hpp`, `ttnn/cpp/ttnn/kernel_lib/tilize_helpers_dataflow.hpp`, `eval/golden_tests/tilize/feature_spec.py`, perf catalog entries `double_buffer`, `width_split`, `compute_block_size`, `split_reader`, `noc_placement` |

### The one claim this design is judged on

The op's `tile_grid` axis exists because a work split across tile-rows alone
collapses to one core on `short_wide` — the decode-phase LLM geometry
(`[1,1,32,16384]`: **R=1**, C=512). This design therefore blocks and assigns on
**both** tile-grid axes from Phase 0. The column extent is not an optimisation
bolted on later: it is *already required* for correctness, because

* `short_wide_l1_forcing` (`[1,1,32,8192]`, C=256) is 1 MB per side at float32 —
  a block spanning the full width does not fit in L1 at all; and
* `low_l1=True` demands a per-core footprint that is O(1) in the tensor dims,
  which is impossible without a constant cap on the column extent.

Once the column extent knob exists, putting the column-chunk index into the
**core assignment** instead of an inner per-core loop is the same enumeration
with one index moved. There is no cheaper day-1 scheme, so there is no reason to
build the row-only one first.

## Parameters

| Name | Type | Required | Valid Range | Default | CT/RT |
|------|------|----------|-------------|---------|-------|
| `input_tensor` | `ttnn.Tensor` | yes | on device; `ROW_MAJOR_LAYOUT`, or `TILE_LAYOUT` **with** `tile=` (retile) | — | host |
| `memory_config` | `ttnn.MemoryConfig \| None` | no | any legal output placement | input's `memory_config()` | host |
| `dtype` | `ttnn.DataType \| None` | no | value-preserving target of the input dtype | input's `dtype` | host → `cb_output_tiles.data_format` (CT via CB) |
| `low_l1` | `bool` | no | `{False, True}` | `False` | host → selects `W_CAP` = `min(W_FIT, LOW_L1_WIDTH_CAP)` |
| `output_padded_shape` | `list[int] \| ttnn.Shape \| None` | no | ≥ input shape in every dim | `None` | host |
| `pad_value` | `float \| int \| None` | no | any; coerced into the dtype domain | `None` | host → RT fill word |
| `tile` | `ttnn.Tile \| None` | no | height ∈ {1,2,4,8,16,32}, width == 32 | `ttnn.Tile([32,32])` for RM input; input's own tile for TILE input | host → `TileDescriptor` on both CBs |
| `tile_h` | derived `uint32` | — | 1..32, power-of-two fraction of 32 | 32 | CT |
| `tensor_row_blocks` (R) | derived `uint32` | — | ≥ 1 | — | CT |
| `tensor_col_tiles` (C) | derived `uint32` | — | ≥ 1 | — | CT |
| `num_row_groups` | derived `uint32` | — | 1..R | — | CT |
| `num_w_chunks` | derived `uint32` | — | 1..C | — | CT |
| `block_width_tiles` | derived `uint32` | — | 1..min(255, `W_FIT`) | coarsest that fits (see Blocking Model) | CT |
| `block_width_tail_tiles` | derived `uint32` | — | 1..`block_width_tiles` | — | CT |
| `write_rows_per_barrier` | derived `uint32` | — | 1..8 | `ceil(8 / block_width_tiles)` | CT |
| `input_depth_rows` | derived `uint32` | — | ≥ 2 | 2 | CT (CB capacity) |
| `output_depth_rows` | derived `uint32` | — | ≥ 2 | `write_rows_per_barrier + 1` | CT (CB capacity) |
| `start_block_id`, `num_blocks_this_core` | derived `uint32` | — | per core | — | RT |

Validation raised **before** the support gate (malformed requests → `ValueError` / `RuntimeError`):
input neither `ROW_MAJOR_LAYOUT` nor `TILE_LAYOUT`; `TILE_LAYOUT` input with no `tile=`;
input not on device; last two dims not multiples of 32 with no padding argument;
`output_padded_shape` smaller than the input shape in any dim; `tile` height not a
power-of-two fraction of 32 or width != 32.

*Note on the `tile` check:* `Tile`'s own constructor already throws
("Tile size is not valid for our hardware", `tt_metal/impl/data_format/tile.cpp:45-47`)
for any shape outside `TILE_FACE_HW_CHOICES`, so the **height** half of this rule is
defensive — no constructible `ttnn.Tile` reaches the op with an illegal height. The
**width** half is genuinely reachable: `{32,16}`, `{16,16}`, `{8,16}`, `{4,16}`,
`{2,16}`, `{1,16}` are all constructible and all must be refused by this op. The
acceptance test uses `ttnn.Tile([16, 16])` as the witness for exactly that reason.

`validate()` (registry support gate,
raising `UnsupportedAxisValue` / `ExcludedCell`) is the entry point's **first** line;
these malformed-request checks live inside it, ahead of the axis loop, so a malformed
call never reaches the support rectangle.

## Tensors

### Input

| Property | Requirement |
|----------|-------------|
| Shape | any rank 0..6. Geometry is `R = prod(shape[:-2]) * ceil(shape[-2]/tile_h)` output tile-rows by `C = ceil(shape[-1]/32)` tile-columns. Rank < 2 only reachable with a pad argument (the pad synthesizes the tile dims). |
| Dtype | Phase 0 `bfloat16`. TARGET adds `float32`, `fp8_e4m3`, `uint32`, `int32`, `uint16`, `uint8`. Never block-float (bfp8/bfp4 have no ROW_MAJOR form — `read_sticks_for_tilize` asserts this, `tilize_helpers_dataflow.inl:85`). |
| Layout | `ROW_MAJOR` (Phase 0). `TILE` at another tile height = the deferred retile regime. |
| Memory | Phase 0 interleaved DRAM. TARGET adds L1 interleaved and legacy-2D / ND sharded. |

### Output

| Property | Value |
|----------|-------|
| Shape | logical shape **identical** to the input's (a padded call grows only the *padded* shape — promoting the logical shape is a bug) |
| Dtype | `dtype` if given, else input's |
| Layout | `TILE_LAYOUT` with the geometry `tile=` names, else 32x32 |
| Memory | `memory_config` if given, else input's |

Host allocation: `ttnn.allocate_tensor_on_device(ttnn.TensorSpec(logical_shape, out_dtype, ttnn.TILE_LAYOUT, memory_layout, shard_spec, buffer_type, tile), device)`
(`ttnn/cpp/ttnn-nanobind/tensor.cpp:286-355` — the `tile=` argument is what carries a tiny tile).
For `pad_mode="auto"` the padded shape TensorSpec derives from `logical_shape` + `tile` **is exactly**
the tile-round the contract asks for, so auto-padding needs no extra host mechanism. `pad_mode="explicit"`
beyond the tile-round has no host binding today — see Key Risks.

---

## Blocking Model

### Axes

`tilize` has **no dependent axis and no reuse-shared operand.** There is exactly one
input operand and each of its bytes feeds exactly one output byte, so no cut creates a
combine and no cut makes any core re-read bytes another core also reads. The
operand-reuse check is therefore run and comes back empty: for every (operand, split)
pair the operand *does* vary along the split axis. No broadcast/mcast regime exists.

| Axis | Character (+ one-clause reason) | Extent knob | Phase 0 value | Knob source | Core-assignment | Later unlock |
|------|--------------------------------|-------------|---------------|-------------|-----------------|--------------|
| `leading` — `prod(shape[:-2])`, the N/C/... fold | **independent** — each image's tiles are computed from that image's sticks alone; nothing spans images | folded into `block_row_extent` via `rows_per_image = ceil(H/tile_h)` | whole fold, not split separately | `num_images` (host, from `shape[:-2]`) | spread across the grid *through* the row axis (it is the outer factor of `R`) | knob-turn (already spread) |
| `tile_row` — output tile-rows, `R = num_images * rows_per_image` | **independent** — one output tile-row consumes `tile_h` sticks and nothing else | `block_row_extent` (runtime `num_blocks` handed to `compute_kernel_lib::tilize`) | the whole row-group the core owns (`row_end - row_start`) — coarsest possible, and **free in L1** (see below) | `num_row_groups` (host) → CT; extent derived in-kernel as `((g+1)*R)/G - (g*R)/G` | `num_row_groups = min(R, max(1, ceil(num_cores/num_w_chunks)))` cores along this axis | knob-turn |
| `tile_col` — output tile-columns, `C = ceil(W/32)` | **independent** — a tile column depends only on its own 32-element slice of each stick | `block_width_tiles` (+ `block_width_tail_tiles` for the ragged last chunk) | coarsest that fits L1, **reduced only to fill the grid**: `ceil(C / num_w_chunks_target)` | `block_width_tiles` (host constant → CT arg on all three kernels; every dependent quantity — CB page counts, `row_bytes`, `byte_offset`, output page base, `write_rows_per_barrier` — is derived from it, never restated) | `num_w_chunks` (host) cores along this axis | knob-turn |
| `within_tile` — the 32x32 element positions / four 16x16 faces of one output tile | **independent, mechanism-owned** — the face permutation is the tilize LLK's unit and is not splittable by this op | the tile geometry itself: `tile_h x 32`, set by `tile=` | `32 x 32` | `tile` kwarg → `TileDescriptor` on both CBs (`ttnn/cpp/ttnn-nanobind/program_descriptors.cpp:351-363`) | **not assigned across cores** — one output tile is produced entirely by one core; splitting a single tile across cores would be a scheme-change with no payoff (a tile is 2 KB) | scheme-change, never taken |

Every axis has a row and every cell is a decision. `within_tile` is deliberately
*not* assigned across cores and `leading` is deliberately *not* a separate assignment
axis (it is the outer factor of `R`, so cutting `R` already cuts it).

**Intermediate stages: the table re-run.** The scheme has exactly one intermediate —
the row-major sub-block in `cb_input_rows`, produced by `load_block` and consumed by
`tilize_block`. Its axes are the same four, with the same characters and the same
extents (`block_row_extent x block_width_tiles`, streamed one tile-row at a time), and
it is assigned to the same core that owns the block. There is **no collect stage, no
designated coordinator core, and no stage whose computation is left on one core** —
so there is no work-assignment idleness for the traffic ranking to miss.

**`block_row_extent` is free in L1, and that is the load-bearing observation.**
`compute_kernel_lib::tilize<block_width_tiles, in, out>(num_blocks)` waits, tilizes and
pushes **one tile-row at a time** inside a single init/uninit pair
(`tilize_helpers.inl:233-259`: `in_dfb.wait_front(input_pages)` /
`out_dfb.reserve_back(block_width_tiles)` / `push_back` / `pop_front` per iteration).
`read_sticks_for_tilize` mirrors it (`tilize_helpers_dataflow.inl:110-127`: reserve
`width_in_tiles`, read `tile_h` sticks, one barrier, push). So the CBs' live set is
**one tile-row** regardless of `block_row_extent`, and raising `block_row_extent`
costs no L1 while amortizing the LLK init/uninit and the data-format reconfig over
more tile-rows. The coarsest value — the core's whole row-group — is therefore taken
unconditionally, and the only L1-constrained extent is `block_width_tiles`.

### Buffer-depth knobs

| CB | Depth knob | Phase 0 value | What the depth buys |
|----|------------|---------------|---------------------|
| `cb_input_rows` | `input_depth_rows` (capacity = `input_depth_rows * block_width_tiles` pages) | `2` | Reader/compute overlap at tile-row granularity. `double_buffer` (`ttnn/ttnn/operations/examples/double_buffer/report.md`) measured depth 1 → 2 as **1.24x–1.99x** on a DRAM→light-compute→DRAM pipeline — this op's exact skeleton. Depth beyond 2 is unmeasured in the catalog; see the overlap lamp. |
| `cb_output_tiles` | `output_depth_rows = write_rows_per_barrier + 1` (capacity = `output_depth_rows * block_width_tiles` pages) | `write_rows_per_barrier + 1` (2 when `block_width_tiles >= 8`) | Lets the writer hold `write_rows_per_barrier * block_width_tiles` tile-page writes **in flight behind one barrier** while compute fills the next tile-row. `double_buffer` measured 1 → 4 transactions in flight as **2.78x** (162383 → 58513 ns) with a plateau at ~4 and no gain past 8; `block=1` single-buffered is the named trap. The `+1` is the compute-side overlap slot. |

`write_rows_per_barrier = max(1, ceil(WRITE_BATCH_MIN_TILES / block_width_tiles))` with
`WRITE_BATCH_MIN_TILES = 8`. It exists because the writer's transaction unit is one
output **tile page**: on a narrow tensor (`C = 1`, e.g. `[1,1,16384,32]`) a per-tile-row
barrier would put exactly **one** 2 KB write in flight — the measured trap. On a wide
block (`block_width_tiles >= 8`) it is already 1 and the knob is inert.

### Mechanism caps

| Mechanism | Cap on which extent | Clamp | What happens unclamped |
|-----------|--------------------|-------|------------------------|
| `can_use_fast_tilize` requires `block_width_tiles < 256` (`tilize_helpers.inl:77`) | `block_width_tiles` | `FAST_TILIZE_WIDTH_CAP = 255` in the `W_FIT` clamp | Silently falls back to the regular `tilize_block` LLK. Correct, but an unexplained perf cliff exactly where the block is largest. |
| `can_use_fast_tilize` requires `!is_fp32_output_format<output_dfb>()` (`tilize_helpers.inl:72-78`) | output dtype, not an extent | fp32 **output** ⇒ regular path, and it must be paired with `Fp32Mode::Lossless` + `fp32_dest_acc_en=true` + `UnpackToDestMode::UnpackToDestFp32` on `cb_input_rows` (`tilize_helpers.inl:115-127`) | fp32 → fp32 comes back truncated through tf32/bf16. The golden oracle for `float32` is `comp_equal` (`helpers.py` `TOLERANCES`), so this is a **wrong-answer** failure, not a precision drift. |
| `can_use_fast_tilize` requires `dfb_has_32x32_tiles<output_dfb>()` and `!get_dst_full_sync_enabled()` (`tilize_helpers.inl:77`) | output tile geometry; `dst_full_sync_en` | tiny tile ⇒ regular path (accepted); keep `dst_full_sync_en=False` (the `ComputeConfigDescriptor` default) | perf-only for the tile geometry; setting full-sync silently disables fast tilize everywhere |
| `read_sticks_for_tilize` block-float guard `ASSERT(tile_size % tile_hw == 0)` (`tilize_helpers_dataflow.inl:85`) | input dtype | input dtype is never block-float — guaranteed by `TARGET["dtype"]`, which omits bfp8/bfp4 because they have no ROW_MAJOR form | `elem_size` is derived wrong → every stick lands at the wrong L1 stride → a strided tile, not a wrong value |
| `compute_kernel_lib::tilize` capacity assert `get_dfb_num_pages(dfb) >= block_width_tiles` (`tilize_helpers.inl:218-222`) | CB capacity | capacity = `depth_rows * block_width_tiles`, `depth_rows >= 2` | deadlock (assert fires under `--dev`) |
| `read_sticks_for_tilize` capacity assert `width_in_tiles <= cb_capacity` (`tilize_helpers_dataflow.inl:105-107`) | `cb_input_rows` capacity | same clamp | reader blocks on `cb_reserve_back` forever — compute never pops a partial block |
| `ASSERT(num_blocks > 0)` (`tilize_helpers.inl:104`) and `ASSERT(total_num_rows > 0)` (`tilize_helpers_dataflow.inl:88`) | per-core work | cores with `num_blocks_this_core == 0` are **excluded from `all_cores`**; no kernel is placed on them | assert / hang on an idle core |
| One `read_sticks_for_tilize` call spans a **contiguous stick run** (`start_page + block_row*tile_h + row`, `tilize_helpers_dataflow.inl:121`) | `block_row_extent` across an image boundary | Legal to span images **iff `H % tile_h == 0`** (then `start_stick = row_start * tile_h` is exact). When `H % tile_h != 0` the block's reader call is **segmented per image**. Phase 0 is `tile_aligned`, so the condition holds by construction; the pad regime must honour the segmentation. | It reads the *next image's* leading sticks into what should be the pad rows — wrong values, no error, and only on shapes whose `R` comes from the leading fold (`square_large_from_leading_dims`, `[8,1,249,2048]`). |
| DRAM/L1 page alignment on a partial-page read (`accessor.get_noc_addr(page, byte_offset)`) | `byte_offset_within_page`, `row_bytes` | both are integer multiples of `32 * element_size(in_dtype)` by construction (a whole tile-column), i.e. >= 32 B even at `uint8` — satisfies `ttnn.get_dram_alignment()` = 32 | misaligned NoC read |
| L1 residency | `block_width_tiles` | `W_FIT` (closed form below) | OOM. `short_wide_l1_forcing` is the shape that proves it. |
| `low_l1=True` contract (footprint O(1) in tensor dims) | `block_width_tiles` | `LOW_L1_WIDTH_CAP` (host constant, independent of every tensor dim) | a footprint proportional to `W`; `low_l1_forcing_width` (C=256) cannot fit in 1.5 MB at any depth |

### Block sizing and the split, in closed form

Everything below has **one source** and is derived from it; no quantity is restated.

```
grid          = device.compute_with_storage_grid_size()      # never a hardcoded core count
num_cores     = grid.x * grid.y
tile_h        = tile.height                                  # 32 unless tile= says otherwise
num_images    = prod(shape[:-2])                             # 1 for rank < 2
rows_per_image= ceil(shape[-2] / tile_h)                     # ceil, per image — never floor(N*H/tile_h)
R             = num_images * rows_per_image                  # tensor_row_blocks
C             = ceil(shape[-1] / 32)                         # tensor_col_tiles

tb_in         = tile_h * 32 * input_tensor.element_size()     # cb_input_rows page bytes
tb_out        = output_tensor.buffer_page_size()              # cb_output_tiles page bytes (block-float safe)
budget        = ttnn.get_max_worker_l1_unreserved_size()

# Solved from the closed-form footprint bound in l1_ledger.md:
#   footprint <= 2*W*tb_in + (2*W + WRITE_BATCH_MIN_TILES)*tb_out
W_FIT   = clamp((budget - WRITE_BATCH_MIN_TILES*tb_out) // (2*tb_in + 2*tb_out), 1, FAST_TILIZE_WIDTH_CAP)
W_CAP   = min(W_FIT, LOW_L1_WIDTH_CAP) if low_l1 else W_FIT

# Named host constants — the single source of each. None is a tensor dimension.
WRITE_BATCH_MIN_TILES  = 8    # output tile-page writes to keep in flight behind one barrier
FAST_TILIZE_WIDTH_CAP  = 255  # can_use_fast_tilize requires block_width_tiles < 256
LOW_L1_WIDTH_CAP       = 4    # low_l1=True cap: 40 KB at bf16, 80 KB at fp32, W-independent

# Rule 2, in its two ordered steps: (1) fill the grid, (2) coarsest block that fits.
w_chunks_for_l1        = ceil(C / W_CAP)                       # step 2's hard floor
w_chunks_for_occupancy = min(C, ceil(num_cores / R))           # step 1
num_w_chunks_target    = max(w_chunks_for_l1, w_chunks_for_occupancy)
block_width_tiles      = ceil(C / num_w_chunks_target)         # coarsest satisfying both
num_w_chunks           = ceil(C / block_width_tiles)           # tighten (<= target); avoids empty chunks
block_width_tail_tiles = C - (num_w_chunks - 1) * block_width_tiles

num_row_groups   = min(R, max(1, ceil(num_cores / num_w_chunks)))
num_blocks_total = num_row_groups * num_w_chunks
```

The `num_w_chunks = ceil(C / block_width_tiles)` re-tighten is not cosmetic: for
`C = 1572, num_w_chunks_target = 64` it gives `block_width_tiles = 25` and
**63** chunks, not 64 — 64 chunks of 25 would over-cover `C` and hand one core an
empty block, tripping `ASSERT(num_blocks > 0)`.

Worked, for every geometry class in `INPUTS` / `LOOSE_CASES` (bf16, 64-core grid):

| Shape | R | C | `num_w_chunks` | `block_width_tiles` | `num_row_groups` | cores | reader transaction |
|-------|---|---|----------------|---------------------|------------------|-------|--------------------|
| `[1,1,32,32]` single_tile | 1 | 1 | 1 | 1 | 1 | **1** | 32 x 64 B |
| `[1,1,32,64]` baseline | 1 | 2 | 2 | 1 | 1 | **2** | 32 x 64 B |
| `[1,1,2048,64]` tall_narrow | 64 | 2 | 1 | 2 | 64 | **64** | 32 x 128 B |
| `[1,1,16384,32]` tall_narrow perf | 512 | 1 | 1 | 1 | 64 | **64** | 256 x 64 B |
| `[1,1,2048,2048]` square_large | 64 | 64 | 1 | 64 | 64 | **64** | 32 x 4096 B |
| `[1,1,1024,1024]` square_large perf | 32 | 32 | 2 | 16 | 32 | **64** | 32 x 1024 B |
| `[1,1,32,2048]` short_wide | 1 | 64 | 64 | 1 | 1 | **64** | 32 x 64 B |
| `[1,1,64,4096]` short_wide | 2 | 128 | 32 | 4 | 2 | **64** | 32 x 256 B |
| `[1,1,32,16384]` **perf focus** | 1 | 512 | 64 | 8 | 1 | **64** | 32 x 512 B |
| `[1,1,32,32768]` | 1 | 1024 | 64 | 16 | 1 | **64** | 32 x 1024 B |
| `[1,1,64,12288]` | 2 | 384 | 32 | 12 | 2 | **64** | 32 x 768 B |
| `[1,1,1,50304]` logits | 1 | 1572 | 63 | 25 | 1 | **63** | 32 x 1600 B |
| `[8,1,249,2048]` leading fold | 64 | 64 | 1 | 64 | 64 | **64** | 32 x 4096 B |
| `[1,1,32,8192]` fp32, `low_l1=False` | 1 | 256 | 64 | 4 | 1 | **64** | 32 x 512 B |
| `[1,1,32,8192]` fp32, `low_l1=True` | 1 | 256 | 64 | 4 | 1 | **64** | 32 x 512 B |

Every geometry reaches the full grid, and no cell exceeds the L1 budget.

### Regimes

| Regime | Status | Predicate | Block | Data movement vs. minimum | What a bigger block buys |
|--------|--------|-----------|-------|---------------------------|--------------------------|
| `grid2d_full_width` | **built** | `layout == ROW_MAJOR` and `pad_mode == "none"` and `alignment == "tile_aligned"` and output interleaved and `num_w_chunks == 1` | `block_row_extent x C` (the block spans the whole tile-row) | **minimum**: input crosses DRAM once, output once, cross-core = 0. The read side is additionally *optimal in transaction shape* — each stick is one contiguous `C*32*elem` read of a whole DRAM page, so the transaction count is exactly `R * tile_h`, the floor. | Raising `block_row_extent` amortizes one `tilize_init`/`tilize_uninit` pair and one unpack+pack data-format reconfig (`tilize_helpers.inl:158-178`) over more tile-rows. Intended frequency: **once per block** ⇒ once per core. `compute_block_size/report.md` measured ~320 ns per phase per pass and 1.65x from 1 → 8 tile-rows per pass, and `report_reconfig_ablation.md` ~110–150 ns per reconfig. Raising `block_width_tiles` is not available here — it is already `C`. |
| `grid2d_width_chunked` | **built** | same, but `num_w_chunks > 1` (i.e. `C > W_CAP`, or `R * 1 < num_cores`) | `block_row_extent x block_width_tiles` | **minimum in bytes** (still one DRAM crossing each way, cross-core = 0) but **above the minimum in transaction count**: each stick is read in `num_w_chunks` partial-page reads of `block_width_tiles*32*elem` bytes instead of one whole-page read. Read transactions = `R * tile_h * num_w_chunks`. Writes are unaffected (the output page is a tile either way). This is the *price of the column split* and it is paid deliberately — see the traffic ranking. | Same init/reconfig amortization from `block_row_extent`; plus, uniquely to this regime, raising `block_width_tiles` **reduces `num_w_chunks` and therefore the read transaction count**, and enlarges each read (`split_reader/report.md`: 64 x 2 KB vs 4096 x 32 B for identical bytes was **6.1x** on the reader RISC-V). That is exactly the tension the grid-synchronization lamp measures. |
| `row_split_only` (assign row-blocks to cores, loop w-chunks inside each core) | **rejected** — superseded by `grid2d_width_chunked`, which is the *same enumeration* with the w-chunk index moved from an inner loop into the core assignment. Zero incremental code, and it is strictly better: `row_split_only` runs `[1,1,32,16384]` (R=1) on **one** core out of 64 and `[1,1,1,50304]` on one out of 63. | — | `block_row_extent x block_width_tiles`, `num_row_groups = min(R, num_cores)`, `num_w_chunks` chunks looped per core | identical bytes and identical transaction count to `grid2d_width_chunked` | nothing it does not already buy — the difference is occupancy, not blocking |
| `single_core_whole_tensor` | **rejected** — it is the `num_w_chunks == num_row_groups == 1` *parameterization* of `grid2d_full_width`, not a distinct algorithm, and it is what that regime already produces on `[1,1,32,32]`. Writing it as its own path would be a dead end: nothing built on top of it survives. | — | `R x C` | minimum bytes; fewest, largest reads | nothing — there is only one block |
| `grid2d_padded` | **deferred** — the fill is an *additive* step on the built block: the reader memsets the pad region of the L1 sub-block (the W tail inside `padded_row_bytes`, and the H tail rows) before/around the stick reads, and `block_row_extent` segments per image when `H % tile_h != 0`. The block grid, the core assignment, the CBs and the compute call are unchanged, so `grid2d_*` is the stepping stone and this layers onto it. Phase 0's `SUPPORTED["pad_mode"] = ["none"]` refuses it. | `pad_mode != "none"` | same | same as the base regime; the pad region adds *no* DRAM reads (it is filled locally) and adds `pad_tiles` output tile writes, which are part of the output's single crossing | same |
| `grid2d_sharded` | **deferred** — the shard fixes the core assignment and the per-core extent, so `num_row_groups`/`num_w_chunks` are read *off the shard spec* instead of solved; the sharded side's CB becomes **zero-copy over the shard buffer** via `ttnn.cb_descriptor_from_sharded_tensor` (`ttnn/cpp/ttnn-nanobind/program_descriptors.cpp:517-540`) rather than a NoC re-read. Everything else — the block operations, the helper calls, the CB page geometry — is unchanged. Phase 0 refuses it via `SUPPORTED["shard_api"] = ["none"]`. | input or output `memory_config().is_sharded()` | the shard, as one block per core (`l1-footprint-discipline.md` Rule 2, sharded case), sub-chunked along `tile_col` only under the same `W_CAP` | **below** the base regime: the sharded side does not cross DRAM at all (0 crossings for an L1-sharded operand), and cross-core stays 0 because a shard is consumed in place | one block per shard is already the coarsest; nothing further |
| `grid2d_tiny_tile` | **deferred** — `tile_h < 32` changes only the `TileDescriptor` on both CBs; `read_sticks_for_tilize` reads `tile_h` from `unpack_tile_r_dim[cb_id]` (`tilize_helpers_dataflow.inl:75`) and the block grid is unchanged. It needs the tiny-tile face-geometry LLK validated on device (and it drops off the fast path, `tilize_helpers.inl:77`). Phase 0 refuses via `SUPPORTED["tile_height"] = [32]`. | `tile.height < 32` | same, with `tile_h` from `tile=` | same as base | same, plus: at `tile_h = 1` a tile-row is one stick, so the reader issues **one** read per barrier — `write_rows_per_barrier` and an equivalent read-batch knob become the dominant lever rather than an inert one |
| `retile` | **deferred** — the input is already TILE, so the reader walks **faces, not sticks**, and `read_sticks_for_tilize` cannot express it (it is stick-indexed: `accessor.get_noc_addr(start_page + block_row + row, ...)`, `tilize_helpers_dataflow.inl:121`). This is a genuinely distinct LLK path, and the golden suite arch-gates it to Blackhole (`helpers.skip_if_retile_unsupported`). It reuses the *same* block grid and the same core assignment — only `load_block` and the compute call change — so the built structure stays reachable. Phase 0 refuses via `SUPPORTED["in_tile_height"] = ["none"]`. | `input_tensor.layout == TILE_LAYOUT` | same block grid, extents in output tiles | same as base (one crossing each way) | same |
| `untilize_and_retilize` (RM round-trip on the retile path) | **rejected** — it is the host-side workaround the Rules forbid, wearing a kernel hat: it doubles the DRAM traffic (2 crossings in, 2 out) and nothing built on it survives when the real face-walking reader lands. | — | — | **2x the minimum** on both directions | nothing |

**`low_l1` is not a regime — it is a value of the `block_width_tiles` knob.**
`W_CAP = min(W_FIT, LOW_L1_WIDTH_CAP)`; the code path, the CBs, the block operations and
the kernels are byte-identical. That is why `helpers.run_tilize`'s bit-identity A/B is
satisfied by construction, and why `low_l1=False` is *also* dimension-independent
(`W_FIT` is a constant of the dtype and the device, not of `W`). Phase 0 refuses
`low_l1=True` by declaration only.

### Traffic ranking

Ranked over **all** candidate splits, in DRAM crossings per tensor, then cross-core
bytes, then DRAM transaction count/size. No nanoseconds.

| Candidate split | Input DRAM crossings | Output DRAM crossings | Cross-core bytes | Read transactions | Bytes per read | Occupancy |
|---|---|---|---|---|---|---|
| **no split** (one core) | 1 | 1 | 0 | `R*tile_h` | `C*32*elem` (whole page) | 1 core |
| **`tile_row` only** | 1 | 1 | 0 | `R*tile_h` | `C*32*elem` (whole page) | `min(R, num_cores)` — **1 core on `short_wide`** |
| **`tile_col` only** | 1 | 1 | 0 | `R*tile_h*num_w_chunks` | `C*32*elem / num_w_chunks` | `min(C, num_cores)` — 1 core on `tall_narrow` |
| **`tile_row` x `tile_col` (chosen)** | 1 | 1 | 0 | `R*tile_h*num_w_chunks`, with `num_w_chunks` **minimized** subject to L1 fit and full occupancy | `C*32*elem / num_w_chunks` | `num_cores` on every geometry in `INPUTS` |
| **split a dependent axis + combine across cores** | — | — | — | — | — | **N/A: `tilize` has no dependent axis.** There is no reduction, scan or cross-block combine anywhere in the op, so this row has no content. It is listed because the ranking must consider it, not omitted because the answer is boring. |
| **split `within_tile` (one tile across cores)** | 1 | 1 | one tile's bytes per split tile — the halves must be gathered before the tile page can be written | `R*tile_h*num_w_chunks` | smaller still | `<= 4x` more cores |

**Bytes are invariant across every split** — this op has no overlapping block
footprints and no shared operand re-fetched per block, so `blocking-model.md` §4's
"the extent is invisible to the split ranking" case applies exactly, and the byte
ranking cannot discriminate. What discriminates is **DRAM transaction count and size**
(Rule 1b: "Many small or scattered page reads are dominated by per-transaction cost"),
and there the ordering is strict: `tile_row` splitting is free — it keeps whole-page
reads — while `tile_col` splitting multiplies the read count by `num_w_chunks` and
divides the size by it. `split_reader/report.md` measures that axis directly: **6.1x**
on the reader RISC-V for the same bytes delivered as 4096 x 32 B instead of 64 x 2 KB.

So the chosen split is: **cut `tile_row` first, and cut `tile_col` only as far as L1 fit
and full occupancy require.** That is what `num_w_chunks = max(w_chunks_for_l1,
w_chunks_for_occupancy)` encodes — it is a *minimization* of the column split, not a
maximization of it. Splitting `within_tile` is rejected outright: it is the only
candidate that introduces cross-core traffic at all, for at most 4x more cores on a
2 KB unit.

**Nothing is deferred on traffic grounds.** The chosen split attains the named-boundary
minimum (each input crosses DRAM once, each output once, zero cross-core), and among the
splits that attain it, it minimizes the read transaction count subject to filling the
grid. The cheapest-traffic split *is* the implemented one.

### Block schedule

```cpp
for (uint32_t block_idx = 0; block_idx < num_blocks_this_core; ++block_idx) {
    resolve_block(block_idx);      // index arithmetic only; each kernel derives its own view
    load_block(block_idx);         // reader:  input sticks -> cb_input_rows
    tilize_block(block_idx);       // compute: cb_input_rows -> cb_output_tiles
    store_block(block_idx);        // writer:  cb_output_tiles -> output tile pages
}
```

A **logical** schedule: `reader`, `compute` and `writer` are three asynchronous
kernels, and within a block the three stages are pipelined at **tile-row** granularity
through the CB depths (that is what `input_depth_rows` and `output_depth_rows` buy).
Adjacent blocks on the same core also overlap at the CB boundary.

| Operation | Block shape | Resident across it | Intended frequency of fixed costs |
|-----------|-------------|--------------------|-----------------------------------|
| `resolve_block` | — (indices) | the CT plan (`R`, `C`, `num_w_chunks`, `num_row_groups`, `block_width_tiles`, `block_width_tail_tiles`, `tile_h`) and the RT `start_block_id` | once per block, in each kernel independently. No cross-kernel handshake: all three derive the identical view from identical CT/RT args, which is why there is no coordinator core. |
| `load_block` | `block_row_extent x block_width_tiles` (in tiles) — `block_row_extent * tile_h` sticks, `block_width_tiles*32*elem` bytes each | nothing | **one `noc_async_read_barrier` per tile-row**, with `tile_h` reads in flight behind it (`tilize_helpers_dataflow.inl:126`). `double_buffer` puts the plateau at ~4 in flight; `tile_h = 32` is comfortably past it. |
| `tilize_block` | `block_row_extent x block_width_tiles` | the LLK tilize init state (`tilize_init(icb, block_width_tiles, ocb)` programs the block width) across all `block_row_extent` tile-rows | **one `tilize_init` + one `tilize_uninit` per block**, and **one unpack+pack data-format reconfig per block**. One `tilize_block` / `fast_tilize_block` LLK call per tile-row inside — that is the helper's own necessary traversal, not a unit-at-a-time main schedule. *Implementer note (not a requirement):* consecutive blocks on one core that share the same width can amortize the init further with `InitUninitMode::InitOnly` / `Neither` / `UninitOnly` (`tilize_helpers.hpp` example 6); the init must be re-run whenever the width changes, because `tilize_init` takes `block_width_tiles`. |
| `store_block` | `block_row_extent x block_width_tiles` | nothing | **one `noc_async_write_barrier` per `write_rows_per_barrier` tile-rows** ⇒ `ceil(block_row_extent / write_rows_per_barrier)` barriers per block, with `write_rows_per_barrier * width` whole-tile-page writes in flight behind each. |

**Stall-shadow check.** The only stages that WAIT are `tilize_block` (on
`cb_input_rows`) and `store_block` (on `cb_output_tiles`) — both waiting on their own
pipeline predecessor, never on a peer core. The work that fills those windows is the
*next* tile-row's read and the *next* tile-row's tilize, and that is exactly what the
two depth knobs schedule into them. There is no long-latency cross-core wait anywhere
in the built regime (zero semaphores, zero multicast), so there is no idle window a
reorder could fill, and no floating-point ordering to gate on the precision baseline —
`tilize` performs no arithmetic.

### Perf lamps

| Lamp | Why the default may be wrong here | Nearby alternative to measure |
|------|-----------------------------------|-------------------------------|
| **Grid-synchronization / transaction-size** (the important one) | On `short_wide`, `num_w_chunks` is raised to `num_cores` *purely for occupancy*, which shrinks every reader transaction to `block_width_tiles*32*elem` bytes — 512 B on the perf-focus shape. `width_split/report.md` measured full occupancy as **7.76x** at Wt=256, but `split_reader/report.md` measured a **1.20x–1.74x** penalty for small transactions and `double_buffer` shows the single-core ceiling scaling ~linearly with bytes per transaction. The two effects pull opposite ways and neither is a rounding error. | On `[1,1,32,16384]` (the mandatory perf target): 64 cores @ `block_width_tiles=8` (512 B reads) vs 32 cores @ 16 (1 KiB) vs 16 cores @ 32 (2 KiB). Report the **core count reached alongside the duration**, and compare against the transposed same-tile-count entry `[1,1,16384,32]` (64 cores, whole-page 64 B reads) — the pair is what makes the occupancy claim checkable. |
| **Overlap (input depth)** | `input_depth_rows = 2` is the catalog-measured default, but depth beyond 2 is **unmeasured** in the catalog, and on the wide-chunk regime one tile-row is `tile_h * block_width_tiles*32*elem` bytes read behind a single barrier while compute is nearly free — the largest legal per-tile-row read may serialize movement against compute. | `input_depth_rows` ∈ {2, 3, 4} on `[1,1,32,32768]` (`block_width_tiles=16`, 1 KiB reads) and on `[1,1,2048,2048]` (`block_width_tiles=64`, 4 KiB reads). |
| **Write batch depth** | `WRITE_BATCH_MIN_TILES = 8` sits above `double_buffer`'s measured plateau (~4 in flight), and that report found `block=32` double-buffered **slightly worse** than `block=4`. 8 may already be past the knee, and it costs `output_depth_rows` pages of L1 on exactly the narrow shapes where it matters. | 4 vs 8 vs 16 on `[1,1,16384,32]` (`block_width_tiles=1`, so `write_rows_per_barrier` = the whole knob) and on `[1,1,2048,64]`. |
| **Reader granularity** | `TilizeGranularity::TILE` is the default because it puts `tile_h` reads behind one barrier; `ROW` puts **one** read behind each barrier (`tilize_helpers_dataflow.inl:148-156`) — normally the trap. But `ROW` lets compute start after 1 stick instead of 32, and shrinks `cb_input_rows` when a block covers fewer than `tile_h` sticks, which is precisely the `low_l1` and sub-tile-`H` corner. | `TILE` vs `ROW` on the `low_l1=True` path (`[1,1,32,8192]`, `low_l1=True`) and on `[1,1,1,50304]` (one stick padded to a tile-row). |
| **Row-group granularity** | `block_row_extent` = the whole row-group means one block per core and one init per core — the coarsest, and free in L1. But it also means the reader's *first* barrier is not overlapped by anything, and a core with many tile-rows fills and drains one long pipeline. | 1 block of `n` tile-rows vs 2 blocks of `n/2` on `[1,1,16384,32]` (`block_row_extent = 8`) and `[1,1,2048,2048]` (`block_row_extent = 1`, where the knob is already inert). |

---

## Dataflow Strategy

| Stage | Format | Mechanism | Notes |
|-------|--------|-----------|-------|
| input tensor → NoC | ROW_MAJOR sticks; one DRAM/L1 page per stick of `shape[-1]*elem` bytes | `TensorAccessor` (`TensorAccessorArgs(input_tensor)`), stick-indexed, `get_noc_addr(page, byte_offset)` | `byte_offset = w_chunk * block_width_tiles * 32 * elem` selects this block's column slice **inside** the stick page. This is `read_sticks_for_tilize`'s documented wide-W chunking parameter (`tilize_helpers_dataflow.hpp`, `byte_offset_within_page`: *"CB sizing (per call) then scales with `row_bytes` (the chunk width), not the full row, bounding L1 footprint regardless of total W"*). |
| NoC → `cb_input_rows` | row-major sub-block: `tile_h` sticks x `block_width_tiles*32` elements, laid out at an L1 stride of `padded_row_bytes = block_width_tiles*32*elem` | `dataflow_kernel_lib::read_sticks_for_tilize<cb_input_rows, TILE>` — `cb_reserve_back(width)`, `tile_h` x `noc_async_read`, **one** barrier, `cb_push_back(width)` | Page size = `tile_h*32*elem` = one tile's worth of row-major bytes, which is why the CB's `TileDescriptor` height must be the **output** tile height. |
| `cb_input_rows` → `cb_output_tiles` | row-major → 4-face tiled | `compute_kernel_lib::tilize<block_width_tiles, cb_input_rows, cb_output_tiles>(block_row_extent)`; `fast_tilize` auto-selected when eligible | The dtype cast happens **at pack time**: `cb_output_tiles.data_format = out_dtype` and `ReconfigureRegisterDatatypeMode::UnpackAndPackReconfigure` (the default) reprograms unpack srcA/srcB and the packer (`tilize_helpers.inl:158-178`). No byte copy, no second pass. |
| `cb_output_tiles` → NoC | whole output tile pages | raw `noc_async_write` per tile page, batched `write_rows_per_barrier` tile-rows per barrier; `TensorAccessor(TensorAccessorArgs(output_tensor))` | Output page id = `row_start*C + w_chunk*block_width_tiles + rr*C + i`. Reads on NoC0 (`ReaderConfigDescriptor`), writes on NoC1 (`WriterConfigDescriptor`) — the defaults, and the ones `noc_placement/report.md` measured at 4.3x–4.8x over the reversed assignment. |

**Cross-core contract: none.** The built regime declares zero semaphores and zero
multicast. If the `grid2d_sharded` deferred regime lands, it still needs none — a shard
is consumed in place through a CB backed on the sharded buffer
(`ttnn.cb_descriptor_from_sharded_tensor`), never re-read over the NoC. The only
deferred regime that would need a Tensix-to-Tensix contract is a hypothetical
`within_tile` split, which the traffic ranking rejects.

**Core placement.** `split_work_to_cores(..., row_wise=True)` — explicitly **not** the
`row_wise=False` default. `noc_placement/report.md` measured a column line of cores at
**2.91x worse** than a row line on an interleaved DRAM→DRAM copy, and names the default
as the trap.

## Work Distribution

| Field | Value |
|-------|-------|
| Work unit | one **block** = `block_row_extent` tile-rows x (`block_width_tiles` or `block_width_tail_tiles`) tile-columns |
| Grid | `device.compute_with_storage_grid_size()`, `num_cores = grid.x * grid.y`. **Never a literal.** A full grid is 64 on Wormhole (56 harvested), 130 on Blackhole, ~32 on Quasar — a hardcoded 64 means "the whole grid" on one arch and "half of it" on another. |
| Per-core work | `ttnn.split_work_to_cores(grid, num_blocks_total, row_wise=True)` over the linearized block id `block_id = row_group * num_w_chunks + w_chunk`. Each core gets `(start_block_id, num_blocks_this_core)` as runtime args and derives everything else. Cores in group 2 get one fewer block; cores with `num_blocks_this_core == 0` are **not** in `all_cores` and get no kernel (`ASSERT(num_blocks > 0)`, `tilize_helpers.inl:104`). |
| Remainder | Three independent raggednesses, all alignment-aware from the start: (1) **block count** — `split_work_to_cores`' two groups; (2) **row group** — `row_start = (g*R)/num_row_groups`, `row_end = ((g+1)*R)/num_row_groups`, a balanced monotone split needing no remainder table; (3) **column tail** — the last w-chunk has `block_width_tail_tiles <= block_width_tiles` tiles, and the compute kernel carries **both** template instantiations (`tilize<block_width_tiles>` and `tilize<block_width_tail_tiles>`) selected by `w_chunk == num_w_chunks - 1`. All tile counts use `ceil` and are **per image**: `R = num_images * ceil(H/tile_h)`, never `floor(num_images*H/tile_h)` — `[8,1,249,2048]` is the case that catches the difference (8*ceil(249/32) = 64, not floor(1992/32) = 62). |

### Regime selection predicate (host, exact)

```
if input_tensor.layout == TILE_LAYOUT:                 -> retile               (deferred)
elif pad_mode != "none":                               -> grid2d_padded        (deferred)
elif input or output memory_config is_sharded():       -> grid2d_sharded       (deferred)
elif tile.height < 32:                                 -> grid2d_tiny_tile     (deferred)
elif num_w_chunks == 1:                                -> grid2d_full_width    (built)
else:                                                   -> grid2d_width_chunked (built)
```

The two built regimes differ only in `num_w_chunks`, so they share every kernel; the
distinction is recorded because their **transaction profiles differ** and the perf
pass needs the boundary named. `num_w_chunks` depends on `num_cores`, i.e. on the
**device** — so the two regimes are not interchangeable across arches and
**regime-pinned tests are required**: the acceptance test pins at least one shape that
lands in `grid2d_full_width` on any grid (`[1,1,2048,64]`, `C=2`, `R>=num_cores` on
every arch) and at least one that lands in `grid2d_width_chunked` on any grid
(`[1,1,32,2048]`, `R=1`, `C=64` — `w_chunks_for_occupancy = min(64, num_cores) > 1`
for every arch with more than one core).

### Phase 0 SUPPORTED and the `tile_grid` declaration

Phase 0 declares (the implementer's claim; `INVALID` is **not** declared here — it is a
test-harness concept living in `feature_spec.py`):

| Axis | Phase 0 SUPPORTED |
|------|-------------------|
| `dtype` | `[bfloat16]` |
| `output_dtype` | `[bfloat16]` |
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
| `tile_grid` | `["single_tile", "small", "tall_narrow", "short_wide", "square_large"]` — **all five** |

The five absent-argument sentinels (`shard_api`, `orientation`, `pad_mode`,
`pad_value`, `in_tile_height`) are `"none"` and `"none"` is always legal — four of
Phase 0's own values *are* sentinels, so omitting them breaks the baseline itself.
`validate()` must never refuse `"none"` on those axes.

**Why `tile_grid` is declared complete, against the narrower baseline the task sketches.**
The governing rule is *"list a value only where the work distribution actually reaches
the core grid on that geometry."* The task's narrower list (`single_tile`, `small`,
`tall_narrow`) is justified in its own text by *"a work split across tile-rows alone
genuinely covers those three values"* — a statement conditional on the row-only split.
This design does not build the row-only split (it is `rejected` above, at zero
incremental cost), and the worked table shows the distribution reaching **the full
grid on all five values**. Declaring only three would refuse geometries the split
demonstrably reaches, and would xfail the mandatory perf-focus target
(`[1,1,32,16384]`) at Phase 0, producing no measurement on the one shape the perf gate
exists for. Declaring all five adds exactly **four** real golden cells at bf16/bf16
(`short_wide_canonical`, `short_wide_two_tile_rows`, `short_wide_l1_forcing`,
`square_large`) — every one of them plain tile-aligned interleaved `dram_to_dram`,
differing from the already-claimed `tall_narrow_grid_scale` in geometry alone.

**The retreat, and its price.** If the 2-D core assignment does *not* land — i.e. if
`num_w_chunks` ends up as a per-core *inner loop* rather than an assignment axis — then
`SUPPORTED["tile_grid"]` **must** narrow to `["single_tile", "small", "tall_narrow"]`
(the task-prescribed baseline), because declaring a geometry the split collapses on is
the one form of SUPPORTED drift the golden suite cannot catch. That retreat is **not
free**, and the design says so deliberately: the acceptance test
(`tests/.../test_tilize.py`) pins `regime_width_chunked__short_wide`
(`[1,1,32,2048]`) and `square_large` (`[1,1,1024,1024]`) as required-passing cases, so
narrowing the declaration fails two acceptance cases and xfails the mandatory
perf-focus target. It is a last resort for a Phase 0 that could not land the 2-D
assignment, not an option to prefer — and since the column *extent* knob is already
mandatory for L1 bounding, the only thing the retreat saves is moving one index from an
inner loop into `split_work_to_cores`.

### Structural impossibilities (informational — the user folds these into `feature_spec.py`; do NOT edit it here)

One candidate `INVALID` cell the authored spec may not prune: `TARGET["dtype"]`'s
comment states `fp8_e4m3` is **ROW_MAJOR-only** ("so it is an INPUT that tilizes to any
float output and never an output itself"), but the `tile_geometry_retile` cases build a
**TILE** input, so the cartesian generates `{dtype: fp8_e4m3, in_tile_height: 32|16|8|4|2|1}`
— an fp8 tensor in TILE layout, which by that same comment has no TILE form.
`helpers.create_ttnn_input_tensor` would have to materialize one. If that is indeed
impossible, `{"dtype": ttnn.fp8_e4m3, "in_tile_height": <each retile height>}` belongs
in `INVALID` (or the `skip_if_fp8_unsupported` gate happens to mask it on non-Blackhole
silicon, which would hide rather than resolve it). Flagged, not acted on.

## Circular Buffers

| Semantic Name | Index | Page Size | Num Pages | Sizing rationale | Format | Producer | Consumer | Lifetime |
|---------------|-------|-----------|-----------|------------------|--------|----------|----------|-----------|
| `cb_input_rows` | 0 | `tile_h * 32 * input_tensor.element_size()` (one tile's worth of row-major bytes), `tile = TileDescriptor(tile_h, 32)` | `input_depth_rows * block_width_tiles` = `2 * block_width_tiles` | Live set **spans** `tile_col` (`block_width_tiles` pages = one tile-row of the block) and **streams over** `tile_row` and `leading` — the helper waits/pops one tile-row at a time (`tilize_helpers.inl:233-259`), so `block_row_extent` does **not** enter the size. Capacity is 2x the live set for reader/compute overlap. | `input_tensor.dtype` (Phase 0 `Float16_b`) | `reader` | `compute` | whole program |
| `cb_output_tiles` | 1 | `output_tensor.buffer_page_size()` (one output tile; block-float and tiny-tile safe), `tile = TileDescriptor(tile_h, 32)` | `output_depth_rows * block_width_tiles` = `(write_rows_per_barrier + 1) * block_width_tiles` | Live set **spans** `tile_col` and a `write_rows_per_barrier`-deep window of `tile_row` (the batch in flight behind one write barrier); **streams over** `leading`. The `+1` tile-row is compute's overlap slot while the batch drains. | `dtype` if given else `input_tensor.dtype` (Phase 0 `Float16_b`) — **this CB's format is where the value-preserving cast happens** | `compute` | `writer` | whole program |

**Two CBs is the floor, not a starting point.** `reader`→`compute` and
`compute`→`writer` cross thread boundaries, so each needs a CB; and
`compute_kernel_lib::tilize` `static_assert`s `input_dfb != output_dfb`
(`tilize_helpers.inl:98-99`) — tilize cannot be done in place, because the row-major and
tiled byte layouts of the same data are different permutations. Rule 3 exhausted in
order: (1) *pack into the destination* — already done, `tilize` packs straight into
`cb_output_tiles` and the writer sends it to DRAM; there is no scratch anywhere.
(2) *transform in place* — forbidden by the `static_assert`. (3) *alias disjoint
lifetimes* — both CBs are live for the whole program and pipelined against each other,
so their lifetimes are concurrent by design. (4) *fold into DEST* — the tilize LLK's
DEST use is internal to the helper and not a caller-visible accumulator.
**No L1 budget predicate, safety fraction or blocking search is introduced to make a
buffer fit** — the `W_FIT` solve is a *closed-form* extent bound over an inventory that
is already minimal, which is Rule 1's permitted last step, not its failure mode.

Each CB has exactly **one** producer kernel and **one** consumer kernel. No CB is read
by two kernels; no in-place transform makes `compute` both producer and consumer.

**CB sync ledger (push count = wait count).**

| CB | Producer pushes | Consumer waits | Per block total |
|----|-----------------|----------------|-----------------|
| `cb_input_rows` | `reader`: `cb_push_back(width)` once per tile-row, `block_row_extent` times | `compute`: `wait_front(width)` / `pop_front(width)` once per tile-row, `block_row_extent` times | `block_row_extent * width` both sides |
| `cb_output_tiles` | `compute`: `reserve_back(width)` / `push_back(width)` once per tile-row, `block_row_extent` times | `writer`: `wait_front(k*width)` / `pop_front(k*width)` with `k = min(write_rows_per_barrier, rows_left)`, summing to `block_row_extent` tile-rows | `block_row_extent * width` both sides |

where `width = (w_chunk == num_w_chunks-1) ? block_width_tail_tiles : block_width_tiles`.
The writer's batching is legal because compute pushes in `width`-page quanta and the
writer waits in integer multiples of `width`, and it cannot deadlock because capacity
`(write_rows_per_barrier+1)*block_width_tiles` strictly exceeds the batch the writer
waits for.

## Block Operation Realization

| # | Block operation | Block shape | Helper? | Input CB (semantic name, pages, state) | Output CB (semantic name, pages) | CB state after |
|---|-----------------|-------------|---------|----------------------------------------|----------------------------------|----------------|
| 1 | `resolve_block` | — | no (host-derived CT plan + integer arithmetic) | — | — | unchanged |
| 2 | `load_block` | `block_row_extent x width` tiles = `block_row_extent*tile_h` sticks x `width*32*elem` bytes | **yes** — `dataflow_kernel_lib::read_sticks_for_tilize<cb_input_rows, TilizeGranularity::TILE>` | source is the input tensor via `TensorAccessor` | `cb_input_rows`, `width` pages pushed per tile-row, `block_row_extent*width` total | `cb_input_rows` drained by compute; ring position advanced |
| 3 | `tilize_block` | `block_row_extent x width` tiles | **yes** — `compute_kernel_lib::tilize<width, cb_input_rows, cb_output_tiles>(block_row_extent)`, two CT instantiations (`block_width_tiles`, `block_width_tail_tiles`) selected by `w_chunk == num_w_chunks-1` | `cb_input_rows`, waits/pops `width` pages per tile-row; requires `compute_kernel_hw_startup(cb_input_rows, cb_output_tiles)` called once at kernel entry, before any helper use | `cb_output_tiles`, `width` pages pushed per tile-row | tilize LLK state torn down by `tilize_uninit` at block end (unless the implementer amortizes across an equal-width run) |
| 4 | `store_block` | `block_row_extent x width` tiles | **no** — raw `noc_async_write` loop (justified in API Mapping) | `cb_output_tiles`, waits/pops `min(write_rows_per_barrier, rows_left)*width` pages per barrier | destination is the output tensor via `TensorAccessor` | `cb_output_tiles` fully popped at block end |

## API Mapping

| Block operation | Type | Function | File:Line | Template Params / Args | Input CB | Output CB | Which params are block knobs |
|-----------------|------|----------|-----------|------------------------|----------|-----------|------------------------------|
| `load_block` | helper | `dataflow_kernel_lib::read_sticks_for_tilize` | `ttnn/cpp/ttnn/kernel_lib/tilize_helpers_dataflow.hpp:87-88` (decl), `tilize_helpers_dataflow.inl:66-157` (impl) | `<cb_input_rows, TilizeGranularity::TILE, Accessor>`; args `(accessor, total_num_rows = block_row_extent*tile_h, row_bytes = width*32*elem, start_page = row_start*tile_h, byte_offset_within_page = w_chunk*block_width_tiles*32*elem)` | — | `cb_input_rows` | `row_bytes` **is** the `tile_col` extent (`width_in_tiles = round_up(row_bytes, tile_row_bytes)/tile_row_bytes`, `tilize_helpers_dataflow.inl:92-93`); `total_num_rows` **is** the `tile_row` extent; `byte_offset_within_page` **is** the column-chunk index — the helper documents this exact use ("wrap this helper in a chunk-outer loop and pass `byte_offset_within_page = chunk_id * row_bytes`"); `granularity` is the reader-granularity lamp |
| `tilize_block` | helper | `compute_kernel_lib::tilize` | `ttnn/cpp/ttnn/kernel_lib/tilize_helpers.hpp:187-197` (decl), `tilize_helpers.inl:84-282` (impl) | `<width, cb_input_rows, cb_output_tiles, InitUninitMode::InitAndUninit, WaitMode::WaitBlock, ReconfigureRegisterDatatypeMode::UnpackAndPackReconfigure, Fp32Mode::Fast, RemapMode::Configure>`; arg `(block_row_extent)`, `total_input_pages` omitted (symmetric tile-sized pages) | `cb_input_rows` | `cb_output_tiles` | `width` (first template param) **is** the `tile_col` block extent; `block_row_extent` (runtime `num_blocks`) **is** the `tile_row` block extent; `init_uninit_mode` is the init-amortization knob; `Fp32Mode` must become `Lossless` on the fp32-output refinement (see Key Risks) |
| `tilize_block` prerequisite | helper | `compute_kernel_hw_startup` | required by `tilize_helpers.hpp:89-93` ("PREREQUISITE: Call `compute_kernel_hw_startup(input_cb, output_cb)` at the start of your kernel before using this function") | `(cb_input_rows, cb_output_tiles)` | — | — | — |
| `store_block` | **raw_api** | `noc_async_write` + `noc_async_write_barrier` + `cb_wait_front` / `cb_pop_front`, over `TensorAccessor::get_noc_addr(tile_page_id)` | `tt_metal/hw/inc/api/dataflow/dataflow_api.h` (`noc_async_write`); accessor pattern per `.claude/references/ttnn-cb-memory-fundamentals.md` §"TensorAccessor Pattern"; reference shape at `ttnn/ttnn/operations/examples/tile_reorder/kernels/tile_reorder_writer_relocate.cpp` | per tile-row batch: `min(write_rows_per_barrier, rows_left)` tile-rows x `width` whole-tile-page writes, one barrier | `cb_output_tiles` | — | `write_rows_per_barrier` is the transactions-in-flight knob; `width` is the `tile_col` extent |

**`store_block` — helpers considered and rejected.**

* `dataflow_kernel_lib::write_sticks_after_untilize` (`ttnn/cpp/ttnn/kernel_lib/tilize_helpers_dataflow.hpp:129-130`, impl `tilize_helpers_dataflow.inl:186-243`). **Concrete mismatch:** it is the *untilize* counterpart and writes ROW_MAJOR **sticks**, not tile pages — `tilize_helpers_dataflow.inl:233-235` does `noc_async_write(l1_addr, noc_addr, row_bytes)` with `l1_addr += padded_row_bytes` per stick, addressing the destination by **stick index** (`start_page + block_row + row`). This op's destination is TILE layout, whose pages are whole tiles addressed by **tile index** (`row*C + col`), so every address it computes would be wrong. Its barrier granularity is also fixed at one tile-row, which is exactly the `write_rows_per_barrier` knob this design needs to expose.
* `dataflow_kernel_lib::local_copy_helpers_dataflow` (`local_copy_helpers_dataflow.hpp:12-30`). **Concrete mismatch:** it is explicitly an **L1 → L1** self-aimed-read family ("A self-aimed READ is the only way to copy L1 -> L1", file header) and its destination must resolve to `AddressType::LOCAL_L1`; the header states outright that `Noc::async_write` "resolves its destination as `AddressType::NOC`" and a CB "passed as a unicast write destination fails that static_assert". The output here is an interleaved DRAM tensor, not local L1.
* `dataflow_kernel_lib::mcast_pipe` — the built regime has no cross-core data movement at all (zero semaphores, zero multicast); there is nothing to broadcast.

**The gap this records:** a `write_tile_pages_for_tilize<cb>(accessor, num_tile_rows, tiles_per_row, tensor_col_tiles, start_tile_row, col_offset, rows_per_barrier)` block helper does not exist and would close it — the symmetric counterpart to `read_sticks_for_tilize` on the tiled side. Per `blocking-model.md` §2 that is a reason to build the missing block operation, not to stop; the implementer may realize `store_block` as a thin kernel function or as such a helper, freely. What must not happen is the batching collapsing to one write per barrier.

**Every compute phase uses a helper.** There is exactly one compute phase and
`compute_kernel_lib::tilize` covers it completely, including the fast/regular path
selection, the dtype reconfig and the CB handshake. No raw LLK is used in the compute
kernel.

## Key Risks and Gotchas

| Risk | Why it bites here | Mitigation in this design |
|------|-------------------|---------------------------|
| **fp32 → fp32 is silently inexact on the default path** | `Fp32Mode::Fast` truncates fp32 → tf32 into DEST (`tilize_helpers.hpp` `Fp32Mode` comment), and `can_use_fast_tilize` already disables the fast path for an fp32 *output* (`tilize_helpers.inl:72-78`) but the **regular** path "still round-trips fp32 through tf32 in Dest" unless DEST is fp32 and the input CB is `UnpackToDestFp32` (`tilize_helpers.inl:115-127`). The golden oracle for `float32` is `comp_equal`, not PCC (`helpers.py` `TOLERANCES`) — so this is a wrong-answer failure, and it will not show up at Phase 0's bf16. | The `float32` refinement **must** pass `Fp32Mode::Lossless` **and** set `ComputeConfigDescriptor(fp32_dest_acc_en=True)` **and** `unpack_to_dest_mode[cb_input_rows] = ttnn.UnpackToDestMode.UnpackToDestFp32`. The helper's own `static_assert`s enforce all three once `Lossless` is requested — so requesting `Lossless` is the *safe* move, and the failure mode is a compile error rather than silent corruption. Conversely for fp32 → **bf16**, keep `Fast` + `UnpackToDestMode::Default`: `tilize_helpers.inl:132-138` `static_assert`s that combining fast tilize with `UnpackToDestFp32` "corrupts output". Two CT instantiations selected on `(in_dtype, out_dtype)`. |
| **A block spanning an image boundary reads the wrong sticks, with no error** | `read_sticks_for_tilize` addresses sticks as `start_page + block_row*tile_h + row` — strictly contiguous. When `H % tile_h != 0` each image is tile-padded *independently*, so tile-row `r` of image `i` starts at stick `i*H + local_r*tile_h`, not `r*tile_h`. A block crossing an image boundary would pull the next image's leading sticks into what should be pad rows. It only manifests on shapes whose `R` comes from the **leading fold** — `[8,1,249,2048]` is exactly that case, and it is in the golden set. | Declared as a mechanism cap. Phase 0 is `tile_aligned`, so `H % tile_h == 0` and the flat formula is exact. The geometry is nonetheless derived per image from the start — `rows_per_image = ceil(H/tile_h)`, `R = num_images * rows_per_image` — never `floor(num_images*H/tile_h)`, so the pad refinement only has to add the per-image reader segmentation, not re-derive the grid. |
| **The ragged column tail cannot use one CT block width** | `block_width_tiles` is the **first template parameter** of `compute_kernel_lib::tilize` and is baked into `tilize_init(icb, block_width_tiles, ocb)`; the reader derives `width_in_tiles` from the runtime `row_bytes`. If the tail chunk is narrower, the reader pushes fewer pages than compute waits for → `cb_wait_front` hangs forever. | The compute kernel carries **both** instantiations (`block_width_tiles` and `block_width_tail_tiles`) as CT args and selects on `w_chunk == num_w_chunks - 1`. Both are compiled once; the branch is a runtime `if` over a CT-parameterized call, not a per-block arg list. |
| **`num_w_chunks` and the two built regimes depend on the device grid** | `w_chunks_for_occupancy = ceil(num_cores/R)` reads the live grid, so the same shape lands in `grid2d_full_width` on a small grid and `grid2d_width_chunked` on a large one. A regime that only triggers on some grids passes on one device and fails on another. | Regime-pinned tests are **required** and specified above: `[1,1,2048,64]` pins `grid2d_full_width` on every arch, `[1,1,32,2048]` pins `grid2d_width_chunked` on every arch with more than one core. Both are in the acceptance test. |
| **`low_l1=True` must be dimension-independent, not merely small** | `low_l1_forcing_width` (`[1,1,32,8192]`, C=256) is sized so that "a buffer extent proportional to W cannot fit in L1 at any depth"; the suite runs every `low_l1` scenario at **both** settings and requires bit-identical output, so `low_l1=False` must *also* not OOM. | The footprint is `2*W*tb_in + (wrpb+1)*W*tb_out` with `W <= W_FIT`, a constant of the dtype and the device — **no tensor dimension appears** at either setting. `low_l1=True` only lowers the cap to `LOW_L1_WIDTH_CAP`. Bit-identity is free: the code path is identical and tilize is exact per tile. Closed form and symbol bounds are in `l1_ledger.md`. |
| **The program cache must not re-derive the work split** | `ttnn.generic_op` hashes the whole `ProgramDescriptor` (`ttnn/cpp/ttnn/operations/generic/device/generic_op_device_operation.cpp:74-138`), so the descriptor is constructed on every call — which makes re-running the entire `W_FIT`/`num_w_chunks`/`split_work_to_cores` derivation per call look free. It is not: that derivation *is* the thing the rule says must be cached. | Memoize the **built descriptor** in a module-level dict keyed on `(logical_shape, in_dtype, out_dtype, in_memory_config, out_memory_config, tile, low_l1, pad_mode, padded_shape, pad_value, grid)`. Buffer addresses live in **runtime args**, whose *values* are not hashed (`hash_kernel` hashes only `kernel.runtime_args.size()`, `generic_op_device_operation.cpp:79-95`) — so refreshing them on a repeat call keeps the hash identical and the device program is reused. `ProgramDescriptor.custom_program_hash` (`ttnn/cpp/ttnn-nanobind/program_descriptors.cpp:1127`) can pin the hash so a repeat call's hashing is O(1). |
| **`pad_mode="explicit"` beyond the tile-round has no host binding** | `TensorSpec` derives the padded shape from the logical shape + tile (`ttnn/cpp/ttnn-nanobind/tensor.cpp:288-309`); there is no exposed way to pin a padded shape *larger* than that round (e.g. `[1,1,32,50] -> [1,1,32,128]`, the `explicit_beyond_round_w` case). `Tensor.padded_shape` is read-only (`pytensor.cpp:780`). | Not needed at Phase 0 (`pad_mode == "none"`). `pad_mode="auto"` is fully expressible today — the tile-round *is* what `TensorSpec` derives. The `explicit` refinement must first establish the host route (a `TensorSpec` whose logical shape is the padded shape, then a metadata-only logical-shape narrowing) and it must not become a second dispatch or a host round-trip. Flagged as the one deferred regime with an unresolved *host* mechanism, distinct from the ones with an unresolved kernel mechanism. |
| **A whole-tensor block looks like the obvious design and OOMs** | `[1,1,32,8192]` at float32 is 1 MB per side; `[1,1,32,32768]` is 2 MB. A design that blocks only tile-rows and takes the full width per block does not merely under-fill the grid — it fails to allocate. And per the Rules, a shape in `INPUTS` that does not fit "is a defect in the block, never a support statement about the shape". | `block_width_tiles <= W_FIT`, solved in closed form from the live L1 budget and the *actual* page sizes (`input_tensor.element_size()`, `output_tensor.buffer_page_size()`), so it adapts to float32 and to block-float outputs without a special case. `short_wide_l1_forcing` and `low_l1_forcing_width` are both covered by the same expression. |
| **Integer dtypes take the regular LLK path and must still be bit-exact** | `has_supported_fast_tilize_format` admits only `Float32` and `Float16_b` (`tilize_helpers.inl:32-37`), so `uint32`/`int32`/`uint16`/`uint8` all fall to `tilize_block`, and their golden tolerance is `comp_equal`. `int32 <-> uint32` is additionally a signedness `bit_cast` at the same width. | Phase 0 does not claim them. The integer refinement must verify per-width exactness on device (the width matters: `TARGET["dtype"]` enumerates integers *by width* precisely because "the per-face vs full-tile dim is width-dependent and getting it wrong yields a strided tile rather than a wrong value"). The design's only integer-specific requirement is that `tb_in = tile_h*32*element_size(in_dtype)` uses the real element size, which it does. |
