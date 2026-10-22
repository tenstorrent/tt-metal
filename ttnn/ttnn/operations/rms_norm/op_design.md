# Operation Design: rms_norm

## Overview

| Field | Value |
|-------|-------|
| Classification | compute (row-wise normalization: one row reduction + two element-wise scalings), single native dispatch |
| Goal | `RMSNorm(x) = x / sqrt(mean(x², dim=-1) + ε) · γ` over the last dimension, for `Layout::TILE` and `Layout::ROW_MAJOR` inputs, tile-aligned and non-tile-aligned shapes, `bfloat16` / `float32`, as **one** `ttnn.generic_op` dispatch spread across the full Tensix-core grid. |
| Math | `out[n, h, w] = x[n, h, w] · rsqrt( (1/W) · Σ_{w'<W} x[n, h, w']² + ε ) · γ[w]` (γ ≡ 1 when absent). `W` is the **logical** last-dimension size; padding never enters the sum or the divisor. |
| Mode | Hybrid — kernel-lib helpers for every compute phase, custom reader/writer dataflow kernels (helpers used where their stride contract matches, see API Mapping). |
| References | `.claude/references/blocking-model.md`; `.claude/references/l1-footprint-discipline.md`; `.claude/references/precision_convention.md`; `.claude/skills/padding-in-ttnn/SKILL.md` §3.2; `ttnn/cpp/ttnn/kernel_lib/{eltwise/api/chain.hpp, reduce_helpers_compute.hpp, reduce_helpers_dataflow.hpp, tilize_helpers.hpp, untilize_helpers.hpp, tilize_helpers_dataflow.hpp, dest_helpers.hpp}`; catalog `ttnn/ttnn/operations/examples/master.md` entries `compute_block_size`, `double_buffer`, `noc_placement`, `width_split`, `row_reduce_accumulate`, `compute_fusion`, `sfpu_tile_scope`, `bank_stagger`, `tensix_all_reduce`, `shared_input_reuse`; `eval/golden_tests/rms_norm/feature_spec.py` (TARGET / INPUTS / INVALID / LOOSE_CASES). |

## Parameters

| Name | Type | Required | Valid Range | Default | CT/RT |
|------|------|----------|-------------|---------|-------|
| `input_tensor` | `ttnn.Tensor` | yes | rank ≥ 2, on device, `dtype ∈ {bfloat16, float32}`, `Layout::TILE` or `Layout::ROW_MAJOR`, `TensorMemoryLayout::INTERLEAVED` (Phase 0) | — | tensor (accessor args CT, base address RT) |
| `gamma` (kw-only) | `Optional[ttnn.Tensor]` | no | logical shape `(1, 1, 1, W)` (volume = W elements, last dim == input last dim), `Layout::ROW_MAJOR` (Phase 0), `dtype ∈ {bfloat16, float32}` (may differ from input dtype) | `None` | presence CT (`has_gamma`), address RT |
| `epsilon` (kw-only) | `float` | no | finite, ≥ 0 | `1e-6` | RT (fp32 bit pattern) |
| `compute_kernel_config` (kw-only) | `ttnn.ComputeConfigDescriptor` | no | Phase 0: `fp32_dest_acc_en=True`; `math_fidelity` / `math_approx_mode` any value (not gated) | `default_compute_kernel_config()` | passed verbatim as the compute kernel's `config=` |
| `memory_config` (kw-only) | `Optional[ttnn.MemoryConfig]` | no | Phase 0: interleaved (`TensorMemoryLayout::INTERLEAVED`) only | input's `memory_config()` | host-only (output allocation) |

`memory_config` is accepted because the golden runner passes `memory_config=input.memory_config()` for sharded cells (`eval/golden_tests/rms_norm/helpers.py`); in Phase 0 those cells are refused earlier by the `memory_layout` axis gate.

### Entry-point contract

| Item | Decision |
|------|----------|
| Default config | One exported factory `default_compute_kernel_config()` returning a **fresh** `ttnn.ComputeConfigDescriptor(math_fidelity=HiFi4, fp32_dest_acc_en=True, math_approx_mode=False)`. Entry point: `cfg = compute_kernel_config if compute_kernel_config is not None else default_compute_kernel_config()`. Nothing else in the op restates the default. |
| Hard validation (ValueError) | `len(input.shape) < 2`; gamma last dim ≠ input last dim; gamma volume ≠ W. Raised before any support check. |
| Support gate | `validate()` builds the axes dict (`dtype`, `fp32_dest_acc_en`, `layout`, `alignment`, `rank`, `gamma_mode`, `gamma_dtype`, `gamma_layout`, `memory_layout`) exactly as `eval/golden_tests/rms_norm/axes.py:classify_call` does, with `gamma_dtype`/`gamma_layout = "none"` when gamma is `None`; raises `UnsupportedAxisValue` / `ExcludedCell` (`ttnn/ttnn/operations/_op_contract.py`). `"none"` is always accepted. |
| EXCLUSIONS | `{dtype: float32, fp32_dest_acc_en: False}` — the natively rejected precision combination (`precision_convention.md`). |
| INPUT_TAGGERS | `tag_alignment` → `"tile_aligned"` (H%32==0 and W%32==0) / `"w_non_aligned"` (W%32≠0) / `"h_non_aligned"` (W%32==0, H%32≠0), with `H = shape[-2]`, `W = shape[-1]`; `tag_rank` → `len(shape)`; `tag_gamma_dtype` / `tag_gamma_layout` → `"none"` when gamma is absent. Axis names are exactly TARGET's. |
| Regime refusal | If the R1 residency predicate (Regimes) fails, raise `NotImplementedError` naming the deferred W-split regime R3. Not an axis — a shape-size refusal. |
| Dispatch | Allocate the output (same logical shape, dtype, `Layout` as input; memory config per `memory_config`), build one `ProgramDescriptor`, call `ttnn.generic_op` once. No `to_layout` / `tilize` / `pad` / `slice` on host. |

### Phase 0 support target (TARGET → Phase 0 / later)

| TARGET axis | Phase 0 values | Later values and their cost class |
|-------------|----------------|-----------------------------------|
| `dtype` | `float32`, `bfloat16` | `bfloat8_b` — knob-turn (CB page format; tile-aligned only per INVALID) |
| `fp32_dest_acc_en` | `True` | `False` (bf16 only) — knob-turn: every `Float32` intermediate page format below is derived from this flag; see perf lamp L8 for accumulation precision |
| `layout` | `TILE`, `ROW_MAJOR` | — |
| `alignment` | all three | — |
| `rank` | 2, 3, 4 | — |
| `gamma_mode` | `gamma`, `no_gamma` | — |
| `gamma_dtype` | `float32`, `bfloat16`, `"none"` | `bfloat8_b` (requires TILE gamma) — knob-turn |
| `gamma_layout` | `ROW_MAJOR`, `TILE` (Refinement 1), `"none"` | — (`TILE`: the writer extracts the two row-0 face segments of each γ tile into `cb_gamma_row` once; everything downstream is unchanged) |
| `memory_layout` | `INTERLEAVED`; `WIDTH_SHARDED`, `BLOCK_SHARDED` (Refinement 4, regime R3); `HEIGHT_SHARDED` (Refinement 5, regime R1-HS — the sharded path with G = 1) | — |

## Tensors

### Input

| Property | Requirement |
|----------|-------------|
| Shape | rank ≥ 2, logical `[..., H, W]`, any H and W ≥ 1 (non-multiples of 32 allowed) |
| Dtype | `bfloat16` or `float32` |
| Layout | `Layout::ROW_MAJOR` or `Layout::TILE` |
| Memory | `TensorMemoryLayout::INTERLEAVED` in DRAM or L1 (`BufferType::DRAM`/`L1`) |

### Gamma (optional)

| Property | Requirement |
|----------|-------------|
| Shape | logical `(1, 1, 1, W)` |
| Dtype | `bfloat16` or `float32`, independent of the input dtype |
| Layout | `Layout::ROW_MAJOR` (one page = one stick of W elements) |
| Memory | `TensorMemoryLayout::INTERLEAVED` |

### Output

| Property | Value |
|----------|-------|
| Shape | identical logical shape to the input |
| Dtype | input dtype |
| Layout | input `Layout` |
| Memory | `memory_config` if given, else the input's (`TensorMemoryLayout::INTERLEAVED`) |

## Blocking Model

The first design section. Everything below it is downstream of these decisions.
Semantics: `.claude/references/blocking-model.md`.

### Axes

Tile-index space of the input: `tile_rows × width` (tiles). Definitions (all host-computed once, all `ceil`, per-image):

| Symbol | `Layout::TILE` | `Layout::ROW_MAJOR` |
|--------|----------------|---------------------|
| `W` (elements) | `shape[-1]` | `shape[-1]` |
| `tensor_width_tiles` (Wt) | `ceil(W / 32)` | `ceil(W / 32)` |
| `num_sticks` (rows) | — | `prod(shape[:-1])` |
| `tensor_tile_rows` | `prod(shape[:-2]) · ceil(shape[-2] / 32)` (each image tile-padded independently) | `ceil(num_sticks / 32)` (rows are independent, so 32-stick groups may straddle images; no per-image padding exists in row-major memory) |
| `w_valid_in_last_tile` | `W − 32·(Wt − 1)` ∈ [1, 32] | same |

| Axis | Character (+ one-clause reason) | Extent knob | Phase 0 value | Knob source | Core-assignment | Later unlock |
|------|--------------------------------|-------------|---------------|-------------|-----------------|--------------|
| `batch` (all dims before H) | independent — the math never mixes leading indices | folded into `tile_rows` (no own knob) | folded | — (folding rule: leading dims are treated identically and are contiguous in both layouts' page order, `blocking-model.md` §1) | via `tile_rows` | — |
| `tile_rows` (32-row groups) | independent — each row's result depends only on its own W elements | `block_rows` (R) | coarsest R ≤ the core's assignment that fits L1 with `x_depth = 2` (closed form, Work Distribution) | host constant → compute/reader/writer CT arg `block_rows`; every `R`-scaled CB capacity and loop trip derived from it | `split_work_to_cores(grid, tensor_tile_rows, row_wise=True)` — contiguous tile-row ranges over the **full** compute grid | knob-turn (R, grid size) |
| `width` (Wt tiles), **reduction stage** (`square_accumulate`, `rms_finalize`) | **dependent** — Σx² spans the whole row | `block_width_tiles` | whole row: `width_padded_tiles` (= `num_w_chunks · w_chunk_tiles` ≥ Wt, see below) — not split | derived: `num_w_chunks = ceil(Wt / w_chunk_tiles)`, `width_padded_tiles = num_w_chunks · w_chunk_tiles` (CT) | **not assigned across cores** — each core holds whole rows resident (R1). Cross-core W split = regime R3 (deferred) | scheme-change (R3) |
| `width`, **streaming stages** (`tilize`, `scale_by_gamma`, `scale_by_inv_rms`, `untilize`, `store`, gamma expansion) | independent — once `inv_rms` exists, every output tile depends only on its own x tile, its γ tile and its row's `inv_rms` | `w_chunk_tiles` (C) | `min(Wt, W_CHUNK_TILES)`, `W_CHUNK_TILES = 8` | one host constant `W_CHUNK_TILES`; CT arg `w_chunk_tiles`; `num_w_chunks`, `width_padded_tiles`, every C-scaled CB capacity derived | within the core (sub-blocks of the resident row) | knob-turn (L3) |
| `gamma` operand (W elements) | **reuse-shared** — γ does not vary along `tile_rows`, the split axis, so every core needs identical bytes | resident compact row (`Wt · 32` elements) + per-chunk expansion of C tiles | whole row resident per core; expansion per (block, chunk) | derived from Wt and C | every active core reads it once from DRAM (stepping stone; multicast = regime row G-MC, deferred) | refinement (G-MC) |
| within-tile (32×32 elements) | — (the tile is the unit; not blocked) | — | — | — | — | partial tiles handled by the padding rules in Dataflow Strategy, not by blocking |

**Why `width` is padded to a whole number of chunks.** Every CB that the streaming stages touch is pushed and popped in quanta of exactly C tiles per tile-row, including the last chunk of a row (nominal accounting, `blocking-model.md` §2 "Ragged tails"). That keeps every C-tile wait/reserve aligned to the ring, so no multi-tile window ever straddles `fifo_limit` (the CB ring-wrap invariant), and lets one CT instantiation `tilize<C>` / `untilize<C>` serve every chunk. The `width_padded_tiles − Wt` (< C) **pad tiles** per row in `cb_x_tiles` are zero (Dataflow Strategy, padding rules), so they add nothing to Σx²; the streaming stages compute on them and the writer never stores them.

**Per-stage re-run of the table (each stage's own data):**

| Stage output | Its axes | Block it is produced in | Assignment | Note |
|--------------|----------|------------------------|------------|------|
| `sumsq_partial` (1 tile per tile-row, raw Σ_t x_t², lanes not yet collapsed) | `tile_rows` independent | R tiles per block | the core that owns the rows | in R3 this tile is exactly the per-core partial to combine — it is pre-collapse on purpose |
| `inv_rms` (1 tile per tile-row, column 0 valid) | `tile_rows` independent | R tiles per block | same core | no coordinating core in R1; every core finalizes its own rows |
| `x·γ`, `out` | `tile_rows` × `width` both independent | R × C per (block, chunk) | same core | no stage computes on one core while others wait — R1 has no cross-core stage at all |

**Scheme committed for Phase 0:** R1 — row-parallel, row-resident. Tile-rows are spread across the full grid; each core walks `num_blocks_this_core` blocks of R tile-rows; each block is read from DRAM once, held resident in `cb_x_tiles` for both the reduction pass and the scaling pass, and streamed out in C-tile chunks. **Every knob (R, C, x_depth, grid / core count, the two streaming depths) is a parameter with one source; none is an inlined literal.**

### Buffer-depth knobs

| CB | Depth knob | Phase 0 value | What the depth buys |
|----|------------|---------------|---------------------|
| `cb_x_tiles` | `x_depth` | 2 when `footprint(R=1, x_depth=2) ≤ L1_budget`, else 1 (closed form, Work Distribution) | reader fetches block k+1 while compute runs the scaling pass of block k |
| `cb_x_rm_sticks` (RM) | `stream_depth` | 2 | reader fills chunk j+1 sticks while compute tilizes chunk j |
| `cb_gamma_tiles` | `stream_depth` | 2 | reader expands γ chunk j+1 while compute scales chunk j |
| `cb_output` | `stream_depth` | 2 | compute packs chunk j+1 while the writer drains chunk j |
| `cb_sumsq_partial`, `cb_inv_rms`, `cb_x_gamma`, `cb_y_tiles` | none (1 block) | 1 | compute→compute hand-offs between sequential helpers on the same TRISCs — nothing to overlap |
| `cb_reduce_scaler`, `cb_gamma_row` | none | 1 | constant / resident for the kernel's life |

`stream_depth` is one host constant (2) shared by the three streaming CBs.

### Mechanism caps

| Mechanism | Cap on which extent | Clamp | What happens unclamped |
|-----------|--------------------|-------|------------------------|
| `tilize<block_width_tiles,…>` / `untilize<block_width_tiles,…>` take the width as a **CT template** argument (`tilize_helpers.hpp:187-197`, `untilize_helpers.hpp:145-154`) | C must be compile-time; one instantiation for every chunk | nominal-C chunks (width padded to `num_w_chunks·C`) so a single `tilize<C>` / `untilize<C>` instantiation covers the tail chunk too | a tail chunk of a different width would need a second instantiation **and** would break the C-quantum ring alignment of `cb_x_rm_sticks` / `cb_y_tiles` / `cb_output` → multi-tile window straddling `fifo_limit` → silent garbage |
| `read_sticks_for_tilize` / `write_sticks_after_untilize` derive the L1 stick stride and the pushed/popped page count from `row_bytes` (`tilize_helpers_dataflow.inl:92-93`, `:208-209`) | the RM tail chunk (valid bytes < `C·32·es`) | tail chunk uses the custom path with stride `C·32·es` and C pages (API Mapping) | stride / page-count mismatch with `tilize<C>` / `untilize<C>` → misplaced rows and a push/pop count imbalance (hang) |
| `read_sticks_for_tilize` asserts `width_in_tiles ≤ cb capacity` (`tilize_helpers_dataflow.inl:105-108`) | `cb_x_rm_sticks` capacity | capacity = `stream_depth · C` ≥ C | deadlock (reserve never satisfied) |
| eltwise chain `DestAccumulation` requires `Dst::D0` and one accumulator (`chain.inl:1091-1093`) | `square_accumulate` holds **one** DEST accumulator per tile-row at a time | walk is `grid(R_b, width_padded_tiles)` with `DestAccumulation::PerRow` (acquire/pack/clear per row) | n/a — R and Wt are not bounded by DEST; the helper serializes rows |
| DEST capacity with `fp32_dest_acc_en=True`: 4 tiles half-sync (`dest_helpers.hpp:89-103`) | per-call DEST slots in chains / reduce / untilize | helpers clamp internally via `DEST_AUTO_LIMIT`; no design extent depends on it | — |
| NoC / DRAM alignment (64 B on Blackhole DRAM, 32 B Wormhole; 16 B L1) | RM chunk byte offset `c·C·32·es` and γ row-0 segment copies (16 elements = 32 B bf16 / 64 B fp32) | C·32·es is a multiple of 64 B for every C; segments are 16 B-aligned | unaligned NoC transactions → corrupted data |
| L1 budget (`ttnn.get_memory_view(device, ttnn.BufferType.L1).largest_contiguous_bytes_free_per_bank`, `ttnn/ttnn/device.py:230`) | R and `x_depth` (the only Wt-proportional buffer is `cb_x_tiles`) | closed-form R, then `x_depth` 2→1, then refuse (R3 domain) | CB allocation clash / OOM at program creation |

### Regimes

| Regime | Status | Predicate | Block | Data movement vs. minimum | What a bigger block buys |
|--------|--------|-----------|-------|---------------------------|--------------------------|
| **R1** row-parallel, row-resident | **built** | `memory_layout == INTERLEAVED` **and** `footprint(R=1, x_depth=1) ≤ L1_budget` (footprint expression in `l1_ledger.md`) | `block_rows × width_padded_tiles` resident (reduction stage); `block_rows × w_chunk_tiles` (streaming stages) | **Minimum** at the DRAM boundary: x crosses once (read), out crosses once (write). Above minimum only by γ: each *active* core reads the W-element γ row once — at most `active_cores · W · es_g` ≤ `(1/32) ·` x bytes, because every active core owns ≥ 1 tile-row (32 rows) and γ is one row. No cross-core traffic. | Per block: `square_accumulate` chain init + data-format reconfig (once per block); `rms_finalize` reduce init (once per block); per (block, chunk): 2 chain inits (`scale_by_gamma`, `scale_by_inv_rms`), 1 `untilize` init (RM), γ chunk expansion on the reader (once per block per chunk, shared by all R rows). Larger R divides all of these by R. Larger C divides the per-chunk inits by C and lengthens RM NoC segments (`C·32·es` B). |
| **R1-HS** height-sharded physical placement of R1 | **built in Refinement 5** (the R3 sharded path with one W-slice per core: G = 1, no combine; work split from `shard_spec()` in buffer shard order) — a placement (knob-turn) refinement: Phase 0's placement is `INTERLEAVED`; the logical split (contiguous tile-row ranges per core) is identical, so reaching it means swapping the reader's DRAM fetch for a CB backed on the shard (`ttnn.cb_descriptor_from_sharded_tensor`, zero-copy) and making the block the whole shard (R = shard tile-rows) | `memory_layout == HEIGHT_SHARDED` and shard width == full W | whole resident shard (one block) unless L1 pressure | Minimum: input already in L1 → 0 DRAM reads, output written once into its own shard | one block per core: one set of inits for the whole shard |
| **R3** W-split + cross-core Σx² combine | **built** (Refinement 4) — WIDTH/BLOCK-sharded inputs (TILE zero-copy; ROW_MAJOR re-laid locally), and interleaved TILE inputs when R1 refuses (row too wide) or on occupancy. Phase 0 note kept for history: R3 was deferred because every Phase-0-SUPPORTED cell satisfied R1's predicate. `square_accumulate` already emitted the pre-collapse raw Σx² partial per tile-row and the finalize/scale phases are unchanged downstream of the combine. | `memory_layout ∈ {WIDTH_SHARDED, BLOCK_SHARDED}`, **or** interleaved TILE and (R1's predicate fails **or** the W-split gives ≥ `R3_MIN_GROUP_CORES` (4) slices per row and ≥ `R3_OCCUPANCY_GAIN` (2) × the active cores of R1 — measured, see Refinement 4 in `changelog.md`) | per core: `block_rows × S` (its W-slice; sharded S = shard width, interleaved S from `R3_ROWS_PER_GROUP` / `R3_MAX_GROUP_CORES` / `R3_MIN_SLICE_TILES`), G_w cores per row-group | x: 1 DRAM crossing interleaved (each slice resident), 0 sharded. **Added** cross-core: per tile-row, G_w − 1 partial tiles (`T_interm` each) unicast to the row-group root + 1 `inv_rms` tile multicast to the group (`l1_ledger.md` → data-movement budget, R3). | per block: one combine round (gather + mcast + semaphore) instead of one per tile-row — the grid-synchronization lamp L2 applies |
| **R2** row-parallel two-pass streaming (re-read x from DRAM for the scaling pass) | **rejected** — dead end, superseded by R3 everywhere it would apply: wherever a row is too wide to stay resident, R3 makes each W-slice resident with 1 DRAM crossing instead of 2, and nothing of R2 survives into R3 | — | `block_rows × w_chunk_tiles` streamed twice | x crosses DRAM **twice** (2× minimum) | — |
| **G-MC** γ read once + multicast | deferred — stepping stone over R1's per-core γ read (structure survives: the multicast only changes where `cb_gamma_row` gets its bytes). Positive reason: γ traffic is bounded at ≤ 1/32 of x traffic in R1 (above) | R1 or R3 with `has_gamma` | `cb_gamma_row` (Wt·32 elements) | removes `(active_cores − 1) · W · es_g` DRAM bytes; adds one multicast of `W · es_g` bytes (`ttnn.Mcast` + `McastArgs`) | — |

**Regime-selection function (host, exact; `select_plan()` in `rms_norm_program_descriptor.py`, Refinement 4):** WIDTH/BLOCK-sharded → R3 sharded (the shard spec is the work split; G_w = shard columns). INTERLEAVED → R3 interleaved if TILE and (R1's `footprint(R=1, x_depth=1, C) > L1_budget`, or `G_w ≥ R3_MIN_GROUP_CORES` and `G_w · G_h ≥ R3_OCCUPANCY_GAIN · R1_active_cores`); else R1; ROW_MAJOR interleaved rows too wide for R1 are refused (`NotImplementedError`: interleaved RM R3 is not built). Each branch is pinned by `test_rms_norm_r3.py`. Phase 0 text: `R1 if memory_layout == INTERLEAVED and footprint(R=1, x_depth=1, C) <= L1_budget else refuse`. The `x_depth ∈ {2, 1}` choice inside R1 is grid-/device-dependent (`L1_budget` differs between Wormhole and Blackhole), so it is **pinned by tests**: the acceptance test includes `(1, 1, 32, 8192)` float32 with float32 γ (forces `x_depth = 1`) and smaller shapes (force `x_depth = 2`).

### Traffic ranking

Bytes per tier for a tensor of `tensor_tile_rows · Wt` tiles (`T_in` = tile bytes of the input dtype).

| Candidate split | DRAM crossings (x / out / γ) | Cross-core traffic | Rank | Decision |
|-----------------|------------------------------|--------------------|------|----------|
| `tile_rows` across cores, rows resident (R1) | 1 / 1 / `active_cores` × one row | none | **1 (cheapest)** | **built** |
| `width` across cores + combine (R3), any row grouping | 1 / 1 / one row per row-group | per tile-row: G_w partial tiles to root + one tile multicast back (fan-out G_w) | 2 | deferred — same DRAM bytes, strictly more NoC bytes; wins only on occupancy (`tensor_tile_rows` < grid) or residency (W too wide), measured as a trade |
| 2-D split (`tile_rows` × `width`) | as R3 | as R3 per row-group | 2 | same row as R3 (G_w = grid columns) |
| `tile_rows` across cores, rows streamed twice (R2) | 2 / 1 / per core | none | 3 | rejected (dead end) |

Occupancy is a separate question (`blocking-model.md` §4): R1 leaves cores idle when `tensor_tile_rows < grid cores` (e.g. one-tile-row decode shapes `(1,1,32,8192)` run on one Tensix core). That is the occupancy argument for R3; it does not change the traffic ranking.

### Block schedule

```cpp
// per Tensix core; num_blocks_this_core = ceil(rows_this_core / block_rows); R_b = rows in this block (ragged last block)
for (uint32_t block_idx = 0; block_idx < num_blocks_this_core; ++block_idx) {
    load_block(block_idx);              // reader: R_b x width_padded_tiles x tiles (TILE) or sticks in C-wide chunks (RM)
    tilize_block(block_idx);            // RM only: R_b x width_padded_tiles, as num_w_chunks C-wide tilize sub-blocks
    square_accumulate_block(block_idx); // R_b x width_padded_tiles -> R_b raw sum-of-squares tiles
    rms_finalize_block(block_idx);      // R_b tiles -> R_b inv_rms tiles: collapse row, x(1/W), +eps, rsqrt
    normalize_block(block_idx);         // for each of num_w_chunks chunks: R_b x C:  [x*gamma] -> *inv_rms -> [untilize] -> store
    release_block(block_idx);           // pop cb_x_tiles (R_b x width_padded_tiles) and cb_inv_rms (R_b)
}
```

| Block operation | Block shape | Resident across it | Intended frequency of fixed costs |
|-----------------|-------------|--------------------|-----------------------------------|
| `load_block` | `R_b × width_padded_tiles` tiles, pushed into `cb_x_tiles` (TILE) / `cb_x_rm_sticks` (RM) in C-tile quanta per tile-row | previous block in the other `x_depth` slot | one NoC read barrier per C-tile quantum (C reads in flight, `double_buffer` catalog: 4–8 per barrier); W-pad zeroing once per tile-row (TILE) / once per tail chunk (RM); pad-tile zeroing once per kernel (TILE) |
| `tilize_block` (RM) | `num_w_chunks` sub-blocks of `1 × C` tiles per tile-row, R_b tile-rows | `cb_x_tiles` filling | tilize init/uninit once per block (`InitUninitMode::InitOnly` / `Neither` / `UninitOnly` across the `R_b · num_w_chunks` calls) |
| `square_accumulate_block` | `grid(R_b, width_padded_tiles)` | `cb_x_tiles` (not popped) | one chain call per block (one init + reconfig); DEST acquire / pack once per tile-row; input waited **cumulatively** so math starts on the first landed tile |
| `rms_finalize_block` | `R_b × 1` tiles | `cb_x_tiles` | one reduce call per block (one `reduce_init`); post-op SFPU inits once per tile-row (inside the helper callback) |
| `normalize_block` | `num_w_chunks` sub-blocks of `R_b × C` | `cb_x_tiles`, `cb_inv_rms` (both waited once per block, popped by `release_block`) | per chunk: `scale_by_gamma` chain (1 init), `scale_by_inv_rms` chain (1 init), `untilize<C>(R_b)` (1 init, RM); writer: one write barrier per C-tile quantum |
| `release_block` | — | — | 2 pops per block |

### Perf lamps

| Lamp | Why the default may be wrong here | Nearby alternative to measure |
|------|-----------------------------------|-------------------------------|
| L1 Overlap (R) | R = coarsest fit makes most INPUTS cores own exactly one block, so `x_depth = 2` buys no cross-block prefetch; read/compute overlap then comes only from the cumulative wait in `square_accumulate` and the C-quantum pushes | `R = ceil(rows_this_core / 2)` (≥ 2 blocks per core) on the 8192-row perf shapes |
| L2 Grid synchronization (R3 only) | does not apply to R1 (no collective). In R3, maximal G_w leaves few tiles per core and the per-block combine dominates | fewer W-slice cores per row-group (G_w ∈ {4, 8, grid_x}) |
| L3 Chunk width C | `W_CHUNK_TILES = 8` keeps the fp32 / W=8192 worst case inside L1; for narrower rows larger C halves the per-chunk chain inits and doubles RM NoC segment size (256–512 B segments at C = 8 are small for DRAM) | C = 16 and C = Wt where `footprint` allows (catalog `compute_block_size`, `bank_stagger`) |
| L4 γ expansion | γ row-0 tiles are rebuilt by the reader every (block, chunk): `2·Wt` small L1 copies per block on the reader RISC-V, which also issues the x reads | keep `Wt` γ tiles resident (`Wt · T_g` bytes) when the budget allows; or move the expansion to the writer RISC-V (catalog `split_reader`) **Refinement 3**: TILE γ on the resident path skips staging + expansion entirely — one DRAM read of each tile's row-0 byte prefix straight into `cb_gamma_tiles` (`GAMMA_DIRECT_READ`). Writer γ supply at 32×7168: 32.5 k → 11 k cycles, now hidden behind `square_accumulate`. |
| L5 Reader-side W-pad zeroing | RISC-V stores of up to 31 columns × 32 rows per tile-row on the reader | compute-side mask: `prepare_reduce_mask<cb, REDUCE_ROW>` (`reduce_helpers_dataflow.hpp:73-74`) + a masked multiply of the last tile — required anyway for sharded inputs (cannot write into the caller's shard) |
| L6 Two FPU scalings via an L1 intermediate | `scale_by_gamma` → `cb_x_gamma` → `scale_by_inv_rms` round-trips through L1 (catalog `compute_fusion`: L1 round-trip beats FPU dest-reuse, so this is the default) | one chain with `DestReuseBinary` against an `inv_rms` tile expanded to all 32 columns **Refinement 3**: not built. The binding cost was the per-tile dst-sync + CB handshake, not the L1 round-trip: blocking both chains (`SCALE_BLOCK_TILES`) cut them 67/69 → 37/47 cycles per tile. `DestReuseBinary` has no broadcast operand, and the catalog measures FPU dest-reuse as a loss. **Perf 2**: built in raw LLK and measured a loss on Blackhole too (8192×1024 BLOCK 15.7 → 20.4 µs: the stage is FPU-math-bound, the x·γ L1 round trip runs free on the unpack/pack threads, MOVD2A adds ~40 cycles per tile to math). What did win: holding the broadcast operand in srcB across the tiles that share it (`kernels/rms_norm_srcb_hold.hpp`, 15.7 → 14.8 µs). |
| L7 SFPU scope of the finalize post-op | `mul_unary_tile` / `add_unary_tile` / `rsqrt_tile` run on all 32×32 lanes; only column 0 is consumed (`BroadcastDim::Col`) | `VectorMode::C` scoping (catalog `sfpu_tile_scope`, ~2× on that op; tiny in absolute terms) **Refinement 3**: not built — the whole finalize phase is 1.8 k of ~40 k cycles at 32×7168, so scoping would save ≤ 0.5 k cycles (~1 %). |
| L8 Accumulation width (`fp32_dest_acc_en=False` refinement) | Σx² in a 16-bit DEST loses precision as Wt grows (catalog `row_reduce_accumulate`) | packer L1-accumulation into an fp32 `cb_sumsq_partial` page, or keep fp32 DEST for this phase only. **Measured in Refinement 2**: not binding up to W = 8192 (precision matrix min PCC 0.99997 at HiFi2 bf16 / 16-bit DEST; the perf-case gate 0.9995 holds at 32×7168), so neither remedy was built. |
| L9 Bank stagger / NoC placement | `row_wise=True` split (catalog `noc_placement`) still has every core start its walk on the same DRAM bank column | per-core start-offset stagger (catalog `bank_stagger`, 1.04–1.32×) **Refinement 3**: built as the `BANK_STAGGER` knob (compile-time `RMS_BANK_STAGGER` define; rotates x reads, output writes and γ reads by `core mod NUM_DRAM_BANKS`). Measured flat on every interleaved perf shape: prefill and 640-row are DRAM-bandwidth-bound, not bank-lockstep-bound. Ships off. **Perf 2**: NoC selection does matter — the three westmost Tensix columns (next to DRAM column 0) swap to reads on NoC1 / writes on NoC0 (`NOC_SWAP_COLUMNS`), 1.03–1.16× on every interleaved shape. |
| L10 Data-format reconfig | chain / reduce default reconfig at every phase boundary; with bf16 input + bf16 γ several boundaries keep the same format | `DataFormatReconfig::Disabled` / `ReduceDataFormatReconfigMode::NONE` where provably constant (catalog `compute_block_size`, up to 1.19×) **Refinement 3**: measured by disabling reconfig on every scaling-chain operand at 32×7168 (all-bf16 config): 30.73 vs 30.8 µs, noise. With C = 32 there are only 2·`num_w_chunks` chain calls per block, so it was not built. |

**Stall-shadow check.** R1's only waits are on the producer of the current block: `square_accumulate` waits on DRAM reads — filled by `WaitPolicy::Cumulative` (math on tile i while tile i+1 lands) and, when `x_depth = 2`, by the reader prefetching block k+1 during `normalize_block(k)`. The writer waits on compute; nothing independent of that exists on the writer. In **R3**, `normalize_block` waits on the cross-core combine. Two things are independent of that wait and are scheduled into it: (1) the next block's `square_accumulate` (needs `x_depth = 2`, already the Phase 0 depth); (2) `scale_by_gamma` of the current block — legal because the two element-wise scalings commute (`x·γ·s = (x·γ)·s`), which is exactly why Phase 0 already orders `scale_by_gamma` **before** `scale_by_inv_rms`: R3 inserts the combine between them without reordering. Floating-point order is identical in R1 and R3 (γ first in both); the γ-first vs. reference order (`x/rms·γ`) difference is one rounding and is covered by the precision baseline.

## Dataflow Strategy

| Stage | Format | Mechanism | Notes |
|-------|--------|-----------|-------|
| DRAM → `cb_x_tiles` (TILE input) | tiles of input dtype | reader: `TensorAccessor` page reads, page id `tile_row · Wt + w`; C reads per barrier; push C per quantum (last quantum of a row pushes nominal C, only `Wt − (num_w_chunks−1)·C` tiles read) | W-pad zeroing (below) before the push of the quantum containing tile `Wt−1` |
| DRAM → `cb_x_rm_sticks` (RM input) | RM sticks, `C·32·es` B stride per stick, C tile-sized pages per (tile-row, chunk) | reader: per tile-row, per chunk: read the stick segments `[c·C·32·es, …)` of the tile-row's ≤ 32 sticks; full chunks via `read_sticks_for_tilize`, tail chunk custom (API Mapping) | sticks of a partial last tile-row: only `num_sticks − 32·t` sticks are read; the rest hold stale finite data (row-independent math, never stored) |
| `cb_x_rm_sticks` → `cb_x_tiles` (RM) | tiles of input dtype | compute `tilize<C>` per (tile-row, chunk) | `fp32_mode = Fast` (FPU consumers downstream truncate to tf32 anyway, `tilize_helpers.hpp:47-71`) |
| `cb_x_tiles` → `cb_sumsq_partial` | Float32 tile per tile-row | compute chain: FPU `Mul(x, x)` with DEST accumulation per row | lanes not collapsed: column j holds Σ over tiles of x² at column j |
| `cb_sumsq_partial` → `cb_inv_rms` | Float32 tile per tile-row, column 0 valid | compute `reduce<SUM, REDUCE_ROW>` with scaler 1.0, post-op `×(1/W)`, `+ε`, `rsqrt` in fp32 DEST | 1/W and ε are fp32 bit patterns (RT args); the bf16 scaler is exactly 1.0, so no bf16 rounding of 1/W |
| γ DRAM → `cb_gamma_row` | RM, γ dtype, `Wt·32` elements | reader, once per kernel: one NoC read of the γ stick (`W·es_g` B) | resident for the kernel's life |
| `cb_gamma_row` → `cb_gamma_tiles` | tiles of γ dtype, only tile row 0 valid | **writer** since Refinement 1 (lamp L4), once per kernel when resident γ fits, else per (block, chunk) one step ahead of the output drain (`l1_ledger.md` → Refinement 1 notes). Phase 0 text: reader, per (block, chunk): for each of the chunk's valid tiles, copy elements `[32t, 32t+16)` to face 0 row 0 and `[32t+16, 32t+32)` to face 1 row 0 (16 B-aligned L1 copies or local NoC reads); push C | `BroadcastDim::Row` reads tile row 0 only (`llk_unpack_AB.h:107-112`); other rows are don't-care |
| `cb_x_tiles` × `cb_gamma_tiles` → `cb_x_gamma` | Float32 | compute chain: FPU `Mul`, `BroadcastDim::Row` | gamma only |
| (`cb_x_gamma` or `cb_x_tiles`) × `cb_inv_rms` → `cb_output` (TILE) / `cb_y_tiles` (RM) | output dtype | compute chain: FPU `Mul`, `BroadcastDim::Col` | final rounding to output dtype happens at this pack |
| `cb_y_tiles` → `cb_output` (RM) | RM rows in tile-sized pages, output dtype | compute `untilize<C>(R_b)` | lossless re-layout of output-dtype values |
| `cb_output` → DRAM | tiles / sticks | writer: TILE: write the chunk's valid tiles (page id `tile_row·Wt + w`), skip pad tiles; RM: write valid stick segments of the tile-row's valid sticks (full chunks via `write_sticks_after_untilize`, tail chunk custom); one barrier per quantum; pop C | |

### Padding rules (op contract: input padding is arbitrary garbage — `padding-in-ttnn` §1.2, §3.2 #1)

| Padding | Where | Treatment |
|---------|-------|-----------|
| W tile padding, TILE input (columns `w_valid_in_last_tile … 31` of tile `Wt−1`) | enters Σx² | **reader zeroes it** in L1 after the read lands, before the push, every tile-row (local L1 stores; face layout: columns < 16 in faces 0/2, ≥ 16 in faces 1/3). Overwrite, not multiply — NaN/Inf-safe. |
| Pad tiles `Wt … width_padded_tiles−1` in `cb_x_tiles` | enter Σx² | ring positions are fixed (capacity = `x_depth·R·width_padded_tiles`, every row exactly `width_padded_tiles` tiles); TILE: **reader zero-fills them once** at kernel start (reserve whole capacity, NoC-read from `MEM_ZEROS_BASE`, no push), never overwritten afterwards. RM: produced by `tilize<C>` from the zeroed tail stick segment (next row). |
| RM tail chunk stick bytes `[W·es − c_tail·C·32·es, C·32·es)` | become pad columns and pad tiles after tilize | **reader zeroes them** in the stick slot for every tail chunk, before the push (slots are reused by full chunks, so zero-once is not enough) |
| H padding (TILE rows `H%32…31` of the last tile-row of each image; RM unread sticks of the last tile-row) | never mixes with valid rows (row reduce, element-wise ops are row-local) | none; results land in output padding / unread rows, never stored to logical positions |
| γ positions ≥ W in `cb_gamma_row`, pad γ tiles | multiply pad columns / pad tiles only | none |

### Tensix-to-Tensix contracts (R3 built in Refinement 4; R1-HS built in Refinement 5; G-MC deferred)

| Regime | Contract |
|--------|----------|
| R3 (**built**) | **Refinement 6 — combine topology knob `R3_SCATTER_ROWS_PER_OWNER`:** with `owners = min(G, rows // knob) > 1` the combine is a reduce-scatter. Each block's tile-rows are split into runs of q = ⌈R_b/owners⌉ over the first `owners` members. Every member writes each owner its partials of that owner's rows in one write, into contributor-major gather slots `j·q + r`. Each owner sums its rows (`reduce<SUM, REDUCE_COL, AccumulateViaAdd, Skip>` over (G × owned), row pitch q) and finalizes them (column-0 fused SFPU pass, `FINALIZE_COL0`). It then unicasts its run into every member's `cb_inv_rms`, counted by `SEM_INV`. With owners = 1 the root + multicast below applies, with the same contributor-major one-write gather. Measured: tall BLOCK shards 2.1–2.5× (`changelog.md` → Refinement 6). Original (Refinement 4) text: row-groups are consecutive (y, x)-ordered runs of G_w cores (the partition `ttnn.Mcast` builds), root = index 0. Contributors unicast `cb_partial_col` into the root's `cb_gather` slot `r·G + index` (same CB address on every core) and bump `SEM_GATHER`; the root waits `(block+1)·(G−1)`, reduces, and sends `cb_rms_root` into every member's `cb_inv_rms` through `McastArgs` (receiver handshake). A contributor only writes block k+1 after receiving block k's inv_rms, so gather slots are never overwritten early. W-pad on a TILE shard: the partial last tile is reduced with `ReducePartialScaler::with_partial()` and accumulated onto the full tiles' collapse. Design text: Row-group = G_w Tensix cores sharing a tile-row range, each owning a contiguous W-slice of `ceil(Wt/G_w)` tiles resident. Per block: each core runs `square_accumulate` on its slice → raw partial tile per tile-row; collapse locally with `reduce<SUM, REDUCE_ROW>` (scaler 1.0, no post-op) → column-0-valid partial; send to the row-group root (`tensix_all_reduce` catalog: for a 1-D line with a tiny payload, root reduce + one multicast back; the root is also a worker); root sums the G_w partials with `reduce<SUM, …, AccumulateViaAdd, …, ReduceWithinTile::Skip>` (`reduce_helpers_compute.hpp:195-236`), applies ×(1/W), +ε, rsqrt, and multicasts the `inv_rms` tiles to the group (`ttnn.Mcast` + `McastArgs`, semaphore handshake). `scale_by_gamma` and the next block's `square_accumulate` run during the wait (stall-shadow). Ragged slices: uniform `max_assigned` push/pop quanta (catalog `tensix_all_reduce`). |
| R1-HS | `cb_x_tiles` = `ttnn.cb_descriptor_from_sharded_tensor(input)` (zero-copy, one block = whole shard); `cb_output` backed on the output shard; reader only pushes the shard pages; W-pad handling moves to the compute-side mask (L5) because the reader must not write into the caller's shard. |
| G-MC | one injector Tensix core reads γ into its `cb_gamma_row` and multicasts the `W·es_g` bytes to every active core's `cb_gamma_row` (same address on every core, `ttnn.Mcast` + `McastArgs`). |

## Work Distribution

| Field | Value |
|-------|-------|
| Work unit | one tile-row (32 rows × `width_padded_tiles` tiles); blocks of `block_rows` tile-rows |
| Grid | `device.compute_with_storage_grid_size()` — full grid; core count is a runtime parameter (per-core RT args), never a constant |
| Split | `(num_cores, all_cores, core_group_1, core_group_2, rows_g1, rows_g2) = ttnn.split_work_to_cores(grid, tensor_tile_rows, row_wise=True)` (catalog `noc_placement`: row-wise spreads across the DRAM-facing axis) |
| Per-core work | RT args: `start_tile_row`, `rows_this_core` (`rows_g1` or `rows_g2`); derived on device: `num_blocks_this_core = ceil(rows_this_core / block_rows)`, `R_b = min(block_rows, rows_this_core − block_idx·block_rows)` |
| Remainder | rows: `split_work_to_cores` two groups (differ by 1 tile-row); blocks: ragged last block per core (`R_b < block_rows`), same block operations at a shorter runtime extent; W: pad tiles + nominal C quanta; tile geometry always `ceil`, per image (TILE) / per stick count (RM) |
| `block_rows` (closed form) | From the ledger total `F(R, d) = R·per_row(d) + fixed(C)` with `per_row(d) = d·Wp·T_in + 2·T32 + has_gamma·C·T32 + is_rm·C·T_out` and `fixed(C) = 2048 + stream_depth·C·T_out + is_rm·stream_depth·C·T_in + has_gamma·(Wt·32·es_g + stream_depth·C·T_g)`: `R2 = floor((L1_budget − fixed(C)) / per_row(2))`; if `R2 ≥ 1`: `x_depth = 2`, `R = min(R2, rows_g1)`; else if `F(1, 1) ≤ L1_budget`: `x_depth = 1`, `R = 1`; else refuse (R3). No search, no safety fraction. |
| Regime pinning | single built regime; the `x_depth` branch is pinned by tests (Regimes) |

## Circular Buffers

Symbols: `R` = `block_rows`, `C` = `w_chunk_tiles`, `Wp` = `width_padded_tiles`, `T_in` / `T_out` / `T_g` = tile bytes (2048 B bf16, 4096 B fp32) of input / output / γ dtype, `T32` = 4096 B, `es`, `es_g` = element bytes.

| Semantic Name | Index | Page Size | Num Pages | Sizing rationale | Format | Producer | Consumer | Lifetime |
|---------------|-------|-----------|-----------|------------------|--------|----------|----------|-----------|
| `cb_x_rm_sticks` (RM only) | 0 | `T_in` | `stream_depth · C` | spans 32 sticks × one C chunk; streams over chunks and rows | input dtype | reader | compute | load/tilize, per chunk |
| `cb_x_tiles` | 1 | `T_in` | `x_depth · R · Wp` | live set spans the whole block (R tile-rows × padded row): read once, used by both passes | input dtype | reader (TILE) / compute `tilize` (RM) | compute | load → release, per block |
| `cb_reduce_scaler` | 2 | 2048 | 1 | constant tile (1.0) | bfloat16 | reader (once) | compute (never popped until kernel end) | kernel |
| `cb_sumsq_partial` | 3 | `T32` | `R` | one raw Σx² tile per tile-row of the block | Float32 | compute | compute | square_accumulate → rms_finalize |
| `cb_inv_rms` | 4 | `T32` | `R` | one column-0-valid tile per tile-row; live through all chunks of the block | Float32 | compute | compute | rms_finalize → release |
| `cb_gamma_row` (γ only) | 5 | `32 · es_g` | `Wt` | compact resident γ row (W elements, padded to Wt·32) | γ dtype | reader | reader (scratch source of the expansion; never pushed) | kernel |
| `cb_gamma_tiles` (γ only) | 6 | `T_g` | `stream_depth · C` | one chunk of row-0 γ tiles, shared by the R rows of the block | γ dtype | reader | compute | per (block, chunk) |
| `cb_x_gamma` (γ only) | 7 | `T32` | `R · C` | x·γ for one (block, chunk) between two sequential FPU chains | Float32 | compute | compute | per (block, chunk) |
| `cb_y_tiles` (RM only) | 8 | `T_out` | `R · C` | final tiles of one (block, chunk) between the last chain and `untilize` | output dtype | compute | compute | per (block, chunk) |
| `cb_output` | 16 | `T_out` | `stream_depth · C` | one tile-row × C chunk in flight to the writer (TILE: tiles; RM: untilized rows in tile-sized pages) | output dtype | compute | writer | per (block, chunk) |

Full ledger (capacity vs live set, axis accounting, sharing): `l1_ledger.md`.

## Block Operation Realization

| # | Block operation | Block shape | Helper? | Input CB (semantic name, pages, state) | Output CB (semantic name, pages) | CB state after |
|---|-----------------|-------------|---------|----------------------------------------|----------------------------------|----------------|
| 0 | kernel prologue | — | yes (`calculate_and_prepare_reduce_scaler`) | — | `cb_reduce_scaler` 1 (reader); `cb_gamma_row` filled (reader); pad tiles of `cb_x_tiles` zeroed (reader, TILE) | scaler and γ row resident for the kernel |
| 1 | `load_block` | `R_b × Wp` | partial (`read_sticks_for_tilize` for RM full chunks) | DRAM | TILE: `cb_x_tiles` C per quantum; RM: `cb_x_rm_sticks` C per (row, chunk) | pushed |
| 2 | `tilize_block` (RM) | `R_b · num_w_chunks` calls of `1 × C` | yes | `cb_x_rm_sticks`, C per call, popped | `cb_x_tiles`, C per call | `cb_x_tiles` holds `R_b·Wp` |
| 3 | `square_accumulate_block` | `grid(R_b, Wp)` | yes (chain) | `cb_x_tiles`, `R_b·Wp`, cumulative wait, **not popped** | `cb_sumsq_partial`, 1 per row (`R_b`) | x resident |
| 4 | `rms_finalize_block` | `R_b × 1` | yes (reduce) | `cb_sumsq_partial` `R_b` (bulk wait/pop per row); `cb_reduce_scaler` 1 (waited, not popped) | `cb_inv_rms`, `R_b` | inv_rms resident |
| 5a | `scale_by_gamma` (per chunk, γ only) | `grid(R_b, C)` | yes (chain) | `cb_x_tiles` (caller-managed, strided `{base = c·C, row_stride = Wp}`); `cb_gamma_tiles` C (Row mapping, wait upfront, pop at end) | `cb_x_gamma` `R_b·C` | γ chunk popped |
| 5b | `scale_by_inv_rms` (per chunk) | `grid(R_b, C)` | yes (chain) | `cb_x_gamma` per tile (γ) or `cb_x_tiles` strided (no γ); `cb_inv_rms` Col mapping caller-managed | TILE: `cb_output`; RM: `cb_y_tiles`; `R_b·C` | `cb_x_gamma` drained |
| 5c | `untilize_chunk` (RM) | `R_b` tile-rows × C | yes | `cb_y_tiles` C per tile-row | `cb_output` C per tile-row | `cb_y_tiles` drained |
| 5d | `store_chunk` | `R_b × C` | partial (`write_sticks_after_untilize` for RM full chunks) | `cb_output` C per tile-row | DRAM | popped |
| 6 | `release_block` | — | raw (`cb_pop_front`) | `cb_x_tiles` `R_b·Wp`; `cb_inv_rms` `R_b` | — | empty for next block |

## API Mapping

| Block operation | Type | Function | File:Line | Template Params / Args | Input CB | Output CB | Which params are block knobs |
|-----------------|------|----------|-----------|------------------------|----------|-----------|------------------------------|
| compute boot | raw_api (mandatory boot) | `compute_kernel_hw_startup(icb0, icb1, ocb)` | `tt_metal/hw/inc/api/compute/compute_kernel_hw_startup.h:60` | `(cb_x_tiles, cb_x_tiles, cb_sumsq_partial)` (TILE) / `(cb_x_rm_sticks, cb_x_rm_sticks, cb_x_tiles)` (RM); once, first statement | — | — | — |
| prologue (scaler) | helper | `dataflow_kernel_lib::calculate_and_prepare_reduce_scaler<cb_reduce_scaler, PoolType::SUM, ReduceDim::REDUCE_ROW>()` | `ttnn/cpp/ttnn/kernel_lib/reduce_helpers_dataflow.hpp:97-99` | pool-type-aware; SUM → 1.0; full tile (32 valid) | — | `cb_reduce_scaler` | — |
| `load_block` RM full chunks | helper | `dataflow_kernel_lib::read_sticks_for_tilize<cb_x_rm_sticks>(accessor, sticks_this_row, C·32·es, first_stick, c·C·32·es)` | `tilize_helpers_dataflow.hpp:87-93` | TILE granularity | DRAM | `cb_x_rm_sticks` | `row_bytes = C·32·es` (C) |
| `load_block` RM tail chunk; TILE reads; W-pad / pad-tile zeroing; γ row read + expansion | raw_api | `TensorAccessor::get_noc_addr`, `noc_async_read`, `noc_async_read_barrier`, `cb_reserve_back` / `cb_push_back`, L1 stores / NoC reads from `MEM_ZEROS_BASE` | `tech_reports/tensor_accessor/tensor_accessor.md`; `.claude/skills/padding-in-ttnn/SKILL.md` §2.2 #1, §3.2 #1, §3.3 | C reads per barrier | DRAM | `cb_x_tiles` / `cb_x_rm_sticks` / `cb_gamma_row` / `cb_gamma_tiles` | C (quantum) |
| `tilize_block` | helper | `compute_kernel_lib::tilize<C, cb_x_rm_sticks, cb_x_tiles, InitUninitMode::{InitOnly,Neither,UninitOnly}>(1)` | `tilize_helpers.hpp:187-197` | `fp32_mode = Fast` | `cb_x_rm_sticks` | `cb_x_tiles` | `block_width_tiles = C` |
| `square_accumulate_block` | helper | `compute_kernel_lib::eltwise_chain(IterationShape::grid(R_b, Wp), BinaryFpu<BinaryFpuOp::Mul, input(cb_x_tiles, WaitPolicy::Cumulative, PopPolicy::None, InputTileMapping::Block), input(cb_x_tiles, WaitPolicy::Cumulative, PopPolicy::None, InputTileMapping::Block), Dst::D0, DestAccumulation::PerRow>{}, PackTile<output(cb_sumsq_partial, ReservePolicy::PerOuter, PushPolicy::PerOuter, DataFormatReconfig::Enabled, TileAddressing::Direct, DestAccumulation::PerRow)>{})` | `eltwise/api/chain.hpp:121-148` (IterationShape), `:208` (Cumulative), `:283-287` (DestAccumulation), `:485-494` (BinaryFpu), `:503-504` (PackTile), `:532-533` (eltwise_chain); legality `eltwise/core/chain.inl:53-63` (Block + Cumulative + None), `:1100-1110` (CbA == CbB requires matching policies), `:969-976` (PerRow ↔ PerOuter/PerOuter); Mul accumulation: `chain.inl:1178-1185` (`mul_init(…, acc_to_dest)`), ELWMUL accumulates into DEST natively (`tt_metal/tt-llk/tt_llk_blackhole/llk_lib/llk_math_eltwise_binary.h:156-186`) | A and B are the same CB (x·x) | `cb_x_tiles` | `cb_sumsq_partial` | shape `(R_b, Wp)` |
| `rms_finalize_block` | helper | `compute_kernel_lib::reduce<PoolType::SUM, ReduceDim::REDUCE_ROW, cb_sumsq_partial, cb_reduce_scaler, cb_inv_rms, ReduceInputPolicy::BulkWaitBulkPop>(ReduceInputBlockShape::of(R_b, 1), ReduceInputMemoryLayout::contiguous(), NoAccumulation{}, post_op)`; `post_op(dst)`: `binop_with_scalar_tile_init(); mul_unary_tile(dst, inv_W_bits); add_unary_tile(dst, eps_bits); rsqrt_tile_init(); rsqrt_tile(dst);` | `reduce_helpers_compute.hpp:620-641` (reduce), `:116` (policy), `:294-305` (shape); `tt_metal/hw/inc/api/compute/eltwise_unary/binop_with_scalar.h:35,60,156`; `.../eltwise_unary/rsqrt.h:21,44` | `ReduceAlgorithm::Auto` (= ReduceTile; catalog `row_reduce_accumulate`: single-tile reduce is fastest at 1–2 tiles) | `cb_sumsq_partial`, `cb_reduce_scaler` | `cb_inv_rms` | shape `(R_b, 1)` |
| `scale_by_gamma` | helper | `eltwise_chain(IterationShape::grid(R_b, C), BinaryFpu<BinaryFpuOp::Mul, input(cb_x_tiles, WaitPolicy::None, PopPolicy::None, InputTileMapping::Block, DataFormatReconfig::Enabled, TileAddressing::Strided), input(cb_gamma_tiles, BroadcastDim::Row, WaitPolicy::Upfront, PopPolicy::AtEnd, InputTileMapping::Row)>{StridedTileRange{c·C, Wp}}, PackTile<output(cb_x_gamma)>{})` | `chain.hpp:232-237` (mapping), `:261-266` (Strided), `:305-310` (BroadcastDim), `:368-375` (bcast input); legality `chain.inl:64-66` (Row + Upfront/AtEnd), `:1118-1124` (Strided ↔ None/None) | | `cb_x_tiles`, `cb_gamma_tiles` | `cb_x_gamma` | shape `(R_b, C)` |
| `scale_by_inv_rms` | helper | `eltwise_chain(IterationShape::grid(R_b, C), BinaryFpu<BinaryFpuOp::Mul, A, input(cb_inv_rms, BroadcastDim::Col, WaitPolicy::None, PopPolicy::None, InputTileMapping::Col)>{…}, PackTile<output(cb_output or cb_y_tiles)>{})`; `A = input(cb_x_gamma, PerTile, PerTile, Block)` with γ, else `input(cb_x_tiles, None, None, Block, Enabled, Strided)` with `StridedTileRange{c·C, Wp}` | `chain.hpp:232-237`, `:305-310`; legality `chain.inl:67-70` (Col + None/None) | caller waits `cb_inv_rms` (`R_b`) once per block | `cb_x_gamma` / `cb_x_tiles`, `cb_inv_rms` | `cb_output` / `cb_y_tiles` | shape `(R_b, C)` |
| `untilize_chunk` | helper | `compute_kernel_lib::untilize<C, cb_y_tiles, cb_output>(R_b)` | `untilize_helpers.hpp:145-154` | default modes | `cb_y_tiles` | `cb_output` | `block_width_tiles = C` |
| `store_chunk` RM full chunks | helper | `dataflow_kernel_lib::write_sticks_after_untilize<cb_output>(accessor, sticks_this_row, C·32·es, first_stick, c·C·32·es)` | `tilize_helpers_dataflow.hpp:129-135` | | `cb_output` | DRAM | `row_bytes = C·32·es` |
| `store_chunk` RM tail; TILE writes | raw_api | `noc_async_write`, `noc_async_write_barrier`, `cb_wait_front` / `cb_pop_front` | `tech_reports/tensor_accessor/tensor_accessor.md` | one barrier per quantum | `cb_output` | DRAM | C |
| `release_block` | raw_api | `cb_pop_front(cb_x_tiles, R_b·Wp)`, `cb_pop_front(cb_inv_rms, R_b)` | `tt_metal/hw/inc/api/compute/cb_api.h` | | | | R_b, Wp |

**Helpers considered and rejected (raw_api rows):**

| Raw use | Candidate helper | Mismatch (file:line) | Why it cannot express the shape |
|---------|------------------|----------------------|---------------------------------|
| RM tail-chunk read | `read_sticks_for_tilize` | `tilize_helpers_dataflow.inl:92-93,117,127` | stride and pushed page count are `round_up(row_bytes, 32·es)`-derived; the tail chunk's valid bytes give fewer than C tiles of stride and pages, but `tilize<C>` and the C-quantum ring need stride `C·32·es` and C pages; it also never zeroes pad bytes (`:120-124` only reads) |
| RM tail-chunk write | `write_sticks_after_untilize` | `tilize_helpers_dataflow.inl:208-209` | same stride / pop-count derivation from `row_bytes`; `untilize<C>` emits rows at stride `C·32·es` and C pages |
| TILE tile reads / writes | `read_sticks_for_tilize` / `write_sticks_after_untilize` | `tilize_helpers_dataflow.hpp:46-93` | stick (row-major) helpers; TILE pages are tiles, no tile-page reader helper exists in `kernel_lib` (`ls ttnn/cpp/ttnn/kernel_lib`) |
| W-pad / pad-tile zeroing | `prepare_zero_tile` (`l1_helpers.hpp:62`) | `l1_helpers.hpp:62` | zeroes a whole tile; pad **columns** of a valid tile need a partial, face-aware zero. Pad **tiles** use the same MEM_ZEROS mechanism (implementer may call `prepare_zero_tile` per pad tile). |
| γ row-0 expansion | `tilize<…>` | `tilize_helpers.hpp:121-125` | tilize consumes 32 rows per tile-row; building row 0 by tilizing a 32-row block with 31 garbage rows costs a `Wt`-tile RM staging CB plus a compute pass per chunk, versus `2·C` 16-element L1 copies on the reader |
| `release_block` pops | chain `PopPolicy::AtEnd` on the last chunk | `chain.hpp:211` | x and inv_rms are read by `num_w_chunks` separate chain calls with strided / Col caller-managed policies (`chain.inl:1118-1124` forces None/None for Strided), so no single chain owns the block's pop |

## Broadcast Verification

| Phase | Op | CB_A (semantic name) Valid Region | CB_B (semantic name) Valid Region | Broadcast Dim |
|-------|-----|-----------------------------------|-----------------------------------|---------------|
| square_accumulate | Mul (DEST-accumulate) | `cb_x_tiles`: All (pad columns / pad tiles zeroed) | `cb_x_tiles` (same tile): All | None |
| rms_finalize | reduce SUM REDUCE_ROW | `cb_sumsq_partial`: All | `cb_reduce_scaler`: Row0 | — (output Col0) |
| scale_by_gamma | Mul | `cb_x_tiles`: All | `cb_gamma_tiles`: Row0 (1D `[W]`) | Row |
| scale_by_inv_rms | Mul | `cb_x_gamma` / `cb_x_tiles`: All | `cb_inv_rms`: Col0 (REDUCE_ROW output) | Col |

## Key Risks and Gotchas

| Risk | Why it bites here | Mitigation in this design |
|------|-------------------|---------------------------|
| Padding leaking into Σx² or the divisor | poisoned-padding loose cases (`extras.poison_padding = 1000`) make a padded-W reduce wrong by up to 27 %; `from_torch` zero-pads, so ordinary tests hide it | reader-side zeroing of W-pad columns, pad tiles and RM tail bytes (Padding rules); divisor is the logical W (`inv_W_bits`), never `Wt·32`; acceptance test includes poisoned TILE inputs |
| Pad-tile zero-once relies on fixed ring positions | holds only if every tile-row pushes exactly `Wp` tiles and capacity is `x_depth·R·Wp` | nominal C quanta per row on `cb_x_tiles` (never a short push); capacity expression fixed in the ledger |
| Multi-tile CB window straddling the ring end | `untilize<C>`, `tilize<C>`, writer waits are C-tile windows | every CB touched by streaming stages has capacity a multiple of C and only C-quantum (or 1-tile) traffic per tile-row |
| Same CB as both FPU operands | `BinaryFpu` with `CbA == CbB` dedups wait/pop | both operands use identical mapping and policies (`chain.inl:1100-1110`) |
| x popped too early | `square_accumulate` and every `scale_*` chain read `cb_x_tiles` | all x reads are `PopPolicy::None` / caller-managed; only `release_block` pops |
| `cb_inv_rms` read by every chunk | chains use Col mapping with caller-managed (None, None) | compute waits `R_b` tiles once before the chunk loop, pops in `release_block` |
| fp32 precision | FPU reads fp32 through tf32 SrcA/SrcB | accepted (PCC 0.999 / RMS 0.02 for fp32); Σx², 1/W, ε, rsqrt all in fp32 DEST; intermediates `Float32` |
| L1 budget is device-dependent | Wormhole vs Blackhole worker-L1 differ; fp32 W=8192 is near the limit at `x_depth = 1` | closed-form R / `x_depth` from the queried budget; explicit R3 refusal instead of an allocation failure; acceptance test pins the `x_depth = 1` case |
| Mixed γ / input dtypes | FPU binary with Float16_b SrcA and Float32 (tf32) SrcB | chains keep `DataFormatReconfig::Enabled` on both operands (default) |
| RM partial last tile-row reads stale / uninitialized sticks | rows ≥ valid stick count are never read | row-local math only; writer stores valid sticks only |
| `compute_kernel_config` must be honored | caller may pass HiFi2 / approx mode | pass the resolved config verbatim as the compute kernel `config=`; `rsqrt_tile` follows `math_approx_mode` via `APPROX` |

### Structural impossibilities

None missing from `feature_spec.py` INVALID.
