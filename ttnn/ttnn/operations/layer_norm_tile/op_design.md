# Operation Design: layer_norm_tile

## Overview

| Field | Value |
|-------|-------|
| Classification | fused (optional residual add + row mean + centered row variance + normalize + affine, one device program) |
| Goal | `y = LayerNorm(x [+ residual]) * gamma + beta` over the last dim, one `ttnn.generic_op` dispatch per call, tile-rows spread across the full compute grid from Phase 0, at the DRAM traffic minimum (each activation tensor crosses its memory tier once) |
| Math | `z[r, w] = x[r, w] (+ res[r, w])`; `mean[r] = (1/W) * sum_w z[r, w]`; `c[r, w] = z[r, w] - mean[r]`; `inv_std[r] = 1 / sqrt((1/W) * sum_w c[r, w]^2 + eps)`; `out[r, w] = c[r, w] * inv_std[r] * gamma[w] + beta[w]` |
| Mode | Derivative (new op on kernel-lib helpers; host model = ProgramDescriptor + `ttnn.generic_op`) |
| References | `eval/golden_tests/layer_norm_tile/feature_spec.py` (TARGET / INPUTS / INVALID / LOOSE_CASES), `eval/golden_tests/layer_norm_tile/helpers.py` (`pytorch_layer_norm`, `make_affine`, TOLERANCES), `eval/golden_tests/layer_norm_tile/axes.py` (`classify_call`, needs `default_compute_kernel_config`, `INPUT_TAGGERS`), `eval/op_template.py`, `.claude/references/blocking-model.md`, `.claude/references/l1-footprint-discipline.md`, `.claude/references/ttnn-cb-memory-fundamentals.md` (alias groups, ring-wrap invariant, ownership), `.claude/references/precision_convention.md`, `.claude/references/generic_op_template/`, `ttnn/ttnn/operations/examples/master.md` (entries cited per knob), `ttnn/cpp/ttnn/kernel_lib/*` (cited in API Mapping), sibling design `ttnn/ttnn/operations/rms_norm/op_design.md` (same row-resident pattern) |

### Entry point contract

| Item | Decision |
|------|----------|
| Module | `ttnn/ttnn/operations/layer_norm_tile/` (`__init__.py`, `layer_norm_tile.py`, `layer_norm_tile_program_descriptor.py`, `kernels/`). `__init__.py` exports `layer_norm`, `SUPPORTED`, `EXCLUSIONS`, `INPUT_TAGGERS`, `default_compute_kernel_config`. |
| Prompt import path | `ttnn/ttnn/operations/layer_norm/__init__.py` is a pure re-export (`from ttnn.operations.layer_norm_tile import layer_norm`), so `from ttnn.operations.layer_norm import layer_norm` resolves to the same function object. No logic there. |
| Signature | `layer_norm(input_tensor, gamma, beta, residual_input_tensor=None, *, epsilon=1e-5, compute_kernel_config=None, memory_config=None) -> ttnn.Tensor` |
| Default config | `default_compute_kernel_config()` (factory, fresh object per call): `ComputeConfigDescriptor(math_fidelity=HiFi4, fp32_dest_acc_en=True, math_approx_mode=False)`. `None` resolves through it; nothing else restates the default (`axes.py` reads the same factory). |
| Config pass-through | The resolved descriptor is passed as-is as `config=` of the compute `KernelDescriptor`. `math_fidelity` / `math_approx_mode` are never gated. `fp32_dest_acc_en` selects the work-CB page format (Circular Buffers). |
| Output allocation | Same logical shape, dtype bf16, TILE layout. Memory config: `memory_config` if given, else the input's. Must be INTERLEAVED (L1 or DRAM). Allocated with `ttnn.allocate_tensor_on_device` BEFORE the L1 budget is queried (an L1 output occupies L1 banks). Output tile padding contents are unspecified. |
| Dispatch | Exactly one `ttnn.generic_op([input, (residual,) gamma, beta, output], program_descriptor)` per call. No `to_layout` / `tilize` / `untilize` / `pad` / `slice` / `add` on host. The residual add is phase `residual_add_block` inside the device program. |
| Validation order | 1. `input.dtype != bfloat16` -> `ValueError`. 2. `input.layout != TILE` -> `ValueError`. 3. rank < 2 -> `ValueError`. 4. gamma or beta missing, `gamma.volume() != W` or `beta.volume() != W`, gamma/beta dtype or layout differ from each other -> `ValueError`. 5. residual present and (shape, dtype, layout) != input's -> `ValueError`. 6. `memory_config` given and not INTERLEAVED -> `ValueError`. 7. SUPPORTED / EXCLUSIONS gate via INPUT_TAGGERS (raises the `_op_contract` `NotImplementedError` subclasses). 8. Mechanism gates: `W % 32 != 0` (outside TARGET), ROW_MAJOR affine last dim `% 32 != 0`, regime predicate (R1 footprint) -> `NotImplementedError` naming the cause. |

### Phase 0 scope (feature_spec TARGET axes, by name)

The implementer writes `SUPPORTED`; this is the rectangle the design commits Phase 0 to. Every TARGET value is a format / placement knob on the one built regime, and the acceptance test exercises all of them (the prompt's `ComputeConfigDescriptor(math_fidelity=LoFi)` call pattern carries `fp32_dest_acc_en=False`; the 125.4 us reference row uses TILE gamma in DRAM), so Phase 0 is the whole TARGET.

| Axis (TARGET name) | Phase 0 values | Refinement candidates (TARGET - Phase 0) | Cost class of each value |
|---|---|---|---|
| `dtype` | `bfloat16` | none | - |
| `fp32_dest_acc_en` | `True`, `False` | none | knob-turn: work/stat CB page format `Float32` vs `Float16_b` (ledger audit 2) |
| `layout` | `TILE_LAYOUT` | none | - |
| `alignment` | `tile_aligned`, `h_non_aligned` | none | `h_non_aligned` is free: tile-rows are independent, pad rows compute finite garbage that the output padding absorbs |
| `residual` | `residual`, `no_residual` | none | CT flag `HAS_RESIDUAL`: adds `cb_residual_tiles` + `residual_add_block` |
| `affine_dtype` | `bfloat16` | none | - |
| `affine_layout` | `ROW_MAJOR_LAYOUT`, `TILE_LAYOUT` | none | knob-turn: only the once-per-kernel `stage_affine` read pattern differs (Dataflow Strategy) |
| `memory_layout` | `INTERLEAVED` | none in TARGET; HEIGHT_SHARDED / WIDTH_SHARDED / BLOCK_SHARDED noted under Dataflow Strategy | - |
| `buffer_type` | `DRAM`, `L1` | none | free: TensorAccessor CT args carry the bank type |
| `affine_buffer_type` | `DRAM`, `L1` | none | free (same) |

`EXCLUSIONS`: empty (bf16-only TARGET has no fp32 + `fp32_dest_acc_en=False` cell).

`INPUT_TAGGERS` required by feature_spec: `tag_alignment(inputs, axes)` -> `"tile_aligned"` (H % 32 == 0 and W % 32 == 0) / `"w_non_aligned"` (W % 32 != 0) / `"h_non_aligned"` (W aligned, H % 32 != 0), shape read from `inputs[0]`.

Structural impossibilities: none beyond `feature_spec.INVALID`.

## Parameters

| Name | Type | Required | Valid Range | Default | CT/RT |
|------|------|----------|-------------|---------|-------|
| `input_tensor` | `ttnn.Tensor` | yes | bf16, TILE, interleaved L1/DRAM, rank >= 2, W % 32 == 0 | - | TensorAccessor CT args (reader) |
| `gamma` | `ttnn.Tensor` | yes | bf16, volume W; TILE `(1,1,1,W)` or ROW_MAJOR with last dim % 32 == 0 (`(1,1,W/32,32)` in production) | - | TensorAccessor CT args + CT `AFFINE_IS_TILE`, CT `affine_page_elems` (RM last dim) |
| `beta` | `ttnn.Tensor` | yes | same layout / dtype / placement class as gamma | - | same |
| `residual_input_tensor` | `ttnn.Tensor` | no | same shape / dtype / layout as input; interleaved L1/DRAM independent of input | `None` | CT flag `HAS_RESIDUAL` + TensorAccessor CT args |
| `epsilon` | float | no | > 0 | `1e-5` | compute CT arg (fp32 bit pattern) |
| `compute_kernel_config` | `ttnn.ComputeConfigDescriptor` | no | any fidelity / approx; `fp32_dest_acc_en` in {True, False} | `default_compute_kernel_config()` | compute kernel config |
| `memory_config` | `ttnn.MemoryConfig` | no | INTERLEAVED, L1 or DRAM | input's | output allocation only |
| derived `inv_w` | float | - | `1.0 / W` (true W) | - | compute CT arg (fp32 bit pattern) |

## Tensors

### Input

| Property | Requirement |
|----------|-------------|
| Shape | `[..., H, W]`; production `(B, 1, S, 768)`, `1024 <= B*S <= 4096`, `S <= 512`; any H (non-tile-aligned allowed), W % 32 == 0 |
| Dtype | bfloat16 |
| Layout | TILE (each `(..., H, W)` image tile-padded independently) |
| Memory | interleaved, L1 or DRAM |
| TILE padding | any finite value or garbage; pad rows never mix with real rows (row-independent math) |

### Residual (optional)

| Property | Requirement |
|----------|-------------|
| Shape / dtype / layout | identical to input |
| Memory | interleaved, L1 or DRAM, independent of input |

### Gamma / Beta

| Property | Requirement |
|----------|-------------|
| Shape | volume W. TILE: `(1, 1, 1, W)` padded to `W/32` tiles (row 0 valid). ROW_MAJOR: last dim `P` with `P % 32 == 0` (production `(1, 1, W/32, 32)`, one 64 B stick per tile-column) |
| Dtype | bfloat16 |
| Memory | interleaved, L1 or DRAM; gamma and beta share layout |

### Output

| Property | Value |
|----------|-------|
| Shape | identical to input |
| Dtype | bfloat16 |
| Layout | TILE |
| Memory | interleaved; `memory_config` or input's |

## Blocking Model

Logical tile index space `(row, w)`: `row` = tile-row index over all leading dims, `tensor_row_tiles = prod(shape[:-2]) * ceil(H/32)` (per-image `ceil`, never `floor(prod(shape[:-1]) / 32)`); `w` = tile column, `tensor_w_tiles = W / 32` (24 in production). Tile-row `t` occupies the contiguous page range `[t * tensor_w_tiles, (t+1) * tensor_w_tiles)` of every activation tensor. Leading dims fold into `row` because the op treats them identically and the tile pages are contiguous in that order.

### Axes

| Axis | Character (+ one-clause reason) | Extent knob | Phase 0 value | Knob source | Core-assignment | Later unlock |
|------|--------------------------------|-------------|---------------|-------------|-----------------|--------------|
| `row` (tile-rows) | independent: every statistic and output of a row depends only on that row | `block_row_tiles` (R) | `min(ceil(core_row_tiles_max / MIN_BLOCKS_PER_CORE), R_fit)`, `MIN_BLOCKS_PER_CORE = 2` (overlap lamp; production headline gives R = 1 with 2 blocks on the 18 two-row cores) | host `select_blocking()` -> CT arg to reader / compute / writer | contiguous tile-row ranges over the whole compute grid: `ttnn.split_work_to_cores(grid, tensor_row_tiles, row_wise=True)`; grid = `device.compute_with_storage_grid_size()` passed in, never a literal | knob-turn (R, grid) |
| `w` (tile columns) | dependent: `mean[r]` and `inv_std[r]` span every column of the row | `core_w_tiles` (block extent along `w`) | `tensor_w_tiles` (whole row resident: `z` is consumed four times, and residency is what keeps every activation at one crossing) | host `select_blocking()` CT arg | not split across cores in R1 (decision: the split ranking puts it at equal DRAM bytes plus a combine; see R2) | scheme-change (R2: cross-core Chan combine) |
| `w` transfer quantum of `load_block` / `store_block` | same axis; DMA granularity only, no compute meaning | `io_chunk_tiles` (C) | largest divisor of `tensor_w_tiles` that is `<= IO_CHUNK_TARGET_TILES = 8` (8 in production) | host `select_blocking()` CT arg | follows the row owner | knob-turn |
| `row` sub-tile band (16-row half of a tile-row) | independent (same reason as `row`) | `band_rows` | 32 (a whole tile-row: the work unit is a tile-row) | host constant `BAND_ROWS = 32` | not used in R1 | knob-turn to 16 = regime R3 (balance + all-pad band skip) |

Why the `w` extent is the whole row: `z` (or `x`) is consumed by the mean, the centering, the variance and the normalize. Any block narrower than the row re-reads the activation from its tier (rejected R5) or needs a cross-core combine (R2). The row axis carries all parallelism in R1.

Intermediate data re-run of the table (every stage's output has axes):

| Stage output | Axes | Assignment | Note |
|---|---|---|---|
| `z = x + res` (`cb_row_work_a`) | `row` x `w` (block `R x core_w_tiles`) | same core as the row | no collector |
| `mean` (`cb_row_mean`, one col-0 tile per tile-row) | `row`; `w` collapsed | same core | no collector |
| `c` (`cb_row_work_b`) | `row` x `w` | same core | - |
| raw `sum_w c^2` (`cb_row_sumsq`, one tile per tile-row) | `row`; `w` accumulated in DEST | same core | - |
| `inv_std` (`cb_inv_std`, col-0) | `row` | same core | - |
| normalized / affine tiles | `row` x `w` | same core -> writer | - |

No stage is left on a designated core; there is no collect / coordinate role in R1.

Operand-reuse check (every input operand vs the chosen split = tile-rows across cores):

| Operand | Varies along `row`? | Consequence |
|---|---|---|
| `x` | yes | no reuse |
| `residual` | yes | no reuse |
| `gamma` | no | reuse-shared by construction -> broadcast regime R4 (deferred; per-core sparse read is bounded below 4% of bytes) |
| `beta` | no | same as gamma (R4) |

Stall-shadow check:

| Waiting stage | Waits on | Work that can run in the window | Decision |
|---|---|---|---|
| compute `residual_add_block(b)` | reader chunks of `x`, `res` for block b | streams chunk by chunk (`C` tiles), so it starts after the first chunk lands; reader issues block b+1 during compute of b (`in_depth = 2`) | built |
| compute `row_mean_block(b)` (no-residual) | whole block `b` of `x` resident | nothing inside compute is independent of it; reader prefetches b+1, writer drains b-1 | built (depth knobs) |
| reader `stage_affine` (gamma/beta) | its own NoC reads | gamma/beta are first used by `affine_scale_block(0)`, after four phases of block 0; the reader issues block 0's activation reads FIRST and the affine reads behind them, one shared barrier | built: reorder in the reader, no floating-point effect |
| writer | compute `affine_shift_block(b)` | the writer RISC is idle while the reader issues both `x` and `res` streams; moving the residual stream to the writer RISC fills this window | perf lamp `split_reader` (below) |
| compute `normalize` phases of block b | its own `inv_std` (self-produced) | nothing external | none needed |

### Buffer-depth knobs

| CB | Depth knob | Phase 0 value | What the depth buys |
|----|------------|---------------|---------------------|
| `cb_input_tiles` | `in_depth` (in blocks) | 2 if `ceil(core_row_tiles_max / R) >= 2` else 1 | reader loads block b+1 while compute works on block b |
| `cb_residual_tiles` | `in_depth` (shared knob, one source) | same | same |
| `cb_output_tiles` | `out_depth` (in blocks) | 1 (= `R * tensor_w_tiles` tiles) | compute finishes `affine_shift_block(b)` and moves to block b+1 while the writer drains b |
| `cb_row_work_a/b`, stat CBs | none (compute-internal, one block) | 1 block | n/a: producer and consumer are the same thread sequence |
| `cb_gamma_tiles`, `cb_beta_tiles` | none (persistent) | 1 row | n/a |

### Mechanism caps

| Mechanism | Cap on which extent | Clamp | What happens unclamped |
|-----------|--------------------|-------|------------------------|
| L1 residency (ledger closed form `F(R)`) | R, `in_depth` | `select_blocking()`: largest `R <= ceil(core_row_tiles_max / MIN_BLOCKS_PER_CORE)` with `F(R) <= budget`, budget = `ttnn.get_memory_view(device, L1).largest_contiguous_bytes_free_per_bank - L1_HEADROOM_BYTES` (64 KiB), queried after the output is allocated; if `F(1)` does not fit -> `NotImplementedError` | CB region overlaps L1 tensors (program creation error) |
| Core assignment | R <= `core_row_tiles_max`; `in_depth` <= blocks per core | `min()` in `select_blocking()` | wasted L1 |
| Ring closure of `cb_input_tiles` / `cb_residual_tiles` (reader pushes `C`) | C | `C` divides `tensor_w_tiles`; every block pushes `R_b * tensor_w_tiles` (multiple of C) | pointer passes `fifo_limit` -> silent L1 corruption (`ttnn-cb-memory-fundamentals.md`, CB Ring-Wrap Invariant) |
| Indexed (upfront / no-pop) windows must not straddle `fifo_limit` | stat CB capacities, work alias pair | every indexed window is the FIRST push into its id within a block, so it starts at the id's block-start pointer; that pointer is at base for every full block and for the final ragged block (all earlier blocks full). Hence separate `cb_row_mean` and `cb_row_sumsq` (not one FIFO), and the a/b alias pair (not one rotating id) | indexed read of a wrapped window reads past the ring end (wrong statistics, no error) |
| Alias pair `cb_row_work_a` / `cb_row_work_b` | none on extent | identical page size and capacity `R * tensor_w_tiles`; every phase moves its whole block through each id it touches | pointer drift -> a phase overwrites unread tiles |
| Chain DEST accumulation (`row_sumsq_block`): `PackTile` with `DestAccumulation::PerRow` requires `(PerOuter, PerOuter)` (`eltwise/core/chain.inl:1139-1148`) | none; one DEST tile per row | as stated | static_assert |
| `reduce` AccumulateViaAdd: scaler CB unread only when aligned and not `CopySeedZeroPair` (`reduce_helpers_compute.inl:320-337`) | W % 32 == 0 | gated in validate (step 8); reduce `scaler` template id bound to `cb_gamma_tiles` (never waited) | partial mask would read gamma as a mask (wrong mean) |
| BH NoC DRAM alignment 64 B (`blackhole/core_config.h:42`), L1 16 B | affine staging read size / offset | every affine read is 64 B at a 64 B-aligned page offset (TILE: offsets 0 and 512; RM: `((32k) % P) * 2`, P % 32 == 0) | misaligned DRAM read returns shifted bytes |
| Final ragged block | R | runtime extent `last_block_extent = core_row_tiles - (num_blocks_this_core - 1) * R`; the same block operations take it as their row count | - |

DEST is not a block cap here: the variance accumulates one DEST tile per row; the chains clamp their own lane batching to `DEST_AUTO_LIMIT` (`dest_helpers.hpp:103`).

### Regimes

Named boundaries: each activation's own tier (DRAM or L1 banks). Minimum = `x` read once, `residual` read once, `out` written once, `gamma` / `beta` read once.

| Regime | Status | Predicate | Block | Data movement vs. minimum | What a bigger block buys |
|--------|--------|-----------|-------|---------------------------|--------------------------|
| R1 `row_resident` | built | `W % 32 == 0 and F(R=1) <= budget` (always true for W = 768: `F(1)` ~ 444 KiB) | `R x tensor_w_tiles`, transfers in `C`-tile chunks | `x`, `res`, `out` each 1x (minimum). `gamma`/`beta`: re-read per active core, only row 0 of each tile: `2 * Wt * 64 B` (RM) or `2 * Wt * 128 B` (TILE) per core, at most 0.66 MiB total = 3.5% of the 18.9 MiB headline | Per block, per phase: one helper init + reconfig (7 phases), one upfront wait / bulk reserve on each indexed CB, one SFPU macro load per reduce; reader one barrier per C-chunk; writer one flush per C-chunk; pipeline fill / drain of 7 sequential phases. A bigger R pays these `ceil(core_row_tiles / R)` times instead of `core_row_tiles` times, and lengthens the window the writer drains in |
| R2 `w_split_combine` | deferred: R1 covers every INPUTS shape at the traffic minimum and R2 moves strictly more bytes (its combine); its only gain is occupancy on `tensor_row_tiles < grid` shapes (T = 32: 32x32, 2x512, 4x256, 8x128) and the 128-on-110 tail, which lamp `occupancy` measures. Reachable: R1's `row_mean_block` / `center_block` / `row_sumsq_block` on a `core_w_tiles` slice ARE R2's per-core statistics (slice mean `m_k`, slice `M2_k`); R2 inserts one Chan combine between `row_sumsq_block` and `row_inv_std_block` and one col-broadcast correction `c -= (mean - m_k)` before `normalize_block`; `core_w_tiles` is already a CT knob | `tensor_row_tiles < num_cores` (when built; group size `k` divides `tensor_w_tiles`) | `R x core_w_tiles`, `core_w_tiles = tensor_w_tiles / k` | activations 1x (same as R1); adds per block an all-gather of `2 * R` col-0 stat tiles among `k` cores (`2 * R * (k-1) * tile_bytes` per core) plus one semaphore rendezvous | same as R1, plus fewer combine rounds (one per block) |
| R3 `half_band_split` | built (Refinement 4), gated by measurement (`_half_band_split`): (a) `tensor_row_tiles > grid` and not a multiple of it (128 on 110: 2 rows -> 3 bands = 1.5 rows of bytes on the heaviest core; 8x512 rows ~-1 us, focus2 47.9 -> 47.25 pooled), (b) every band fits on its own core (32x32 19.9 -> 18.7, 3x500 22.1 -> 18.7, 2x512 14.7 -> 13.3 us). R1 kept where R3 turns a balanced R1 split into an unbalanced one (8x384 36.1 vs 38.4, 64x17 21.7 vs 22.7 us: the balanced split keeps the streamed row-stats path) and on multiples of the grid. A band is the byte range `[0, 1024)` / `[1024, 2048)` of a page; `RowSources` pairs a core's contiguous valid bands into compute tile-rows (a (top, bottom) pair of one tile-row moves whole tiles; other pairs and one lone band per odd-count run move half tiles); all-padding bands (`H % 32 in [1, 16]`) are not counted, read or written. Run order (`_even_count_first`): even-count cores first when the residual is in DRAM, else the 3-band cores first, measured | `BAND_ROWS == 16` and `_half_band_split` | `R` compute tile-rows, each holding up to 2 bands | activations 1x; half-tile transactions only on split pairs / lone bands; skipped all-pad bands save their bytes | same as R1 |
| R4 `affine_broadcast` | deferred: gamma+beta re-read is bounded by 3.5% of headline bytes and is 64-128 B sparse reads issued behind block 0 (stall-shadow), so off the critical path; multicast would add a rendezvous on every core to save that. Reachable: `stage_affine` is one reader function writing `cb_gamma_tiles` / `cb_beta_tiles`; a sender core reads once and `ttnn.Mcast` fans out the 2 x 48 KiB tiles into the same CBs | always (operand transport) | n/a | removes `(active_cores - 1) * 2 * Wt * 64-128 B`; adds one 96 KiB multicast + semaphore | n/a |
| R5 `row_reread_stream` | rejected: superseded by R1 everywhere (the row always fits: W is 768). Dead end: re-reads `x`/`res` for the variance and normalize passes | never | `R x C` | `x`, `res` 2-3x | - |
| R6 `affine_full_tile_read` | rejected: superseded by R1's row-0-only staging. Reading whole gamma/beta tiles is `2 * Wt * 2 KiB` = 96 KiB per core, 10.6 MiB over 110 cores (+56% DRAM bytes on the headline) | never | - | +96 KiB per core | - |

Regime selection function (host, exact): `select_blocking()` returns a `Blocking` when `W % 32 == 0` and `F(1) <= budget`, with `half_band = _half_band_split(shape, grid)` choosing R3 or R1 for the work unit (the CB set, compute kernel and footprint are shared: R3 only changes the reader / writer addressing and the per-core unit runs); otherwise `NotImplementedError("layer_norm: row-resident footprint exceeds L1")`. `test_layer_norm_tile_half_band.py` pins each predicate branch (R3 for `T > grid`, R3 one band per core, R1 for a balanced split R3 would unbalance and for a multiple of the grid).

### Traffic ranking

Units: whole-op bytes per tier. `X` = input bytes (tile-padded), `Rs` = residual bytes, `O` = output bytes, `A` = active cores, `Ga` = per-core affine staging bytes (3 KiB RM, 6 KiB TILE).

| Rank | Candidate split | Activation tier (DRAM / L1 banks) | Cross-core | Notes |
|---|---|---|---|---|
| 1 | tile-rows across cores, row resident (R1) | `X + Rs + O + A*Ga` | 0 | minimum; chosen |
| 1 (tie) | 16-row bands across cores (R3) | `X + Rs + O + A*Ga` (less when all-pad bands are skipped) | 0 | equal bytes, 2x transactions; balance only. Deferred |
| 3 | tile-rows x `w` slices, Chan combine (R2) | `X + Rs + O + A*Ga` | `2*R*(k-1)*tile` per core per block + rendezvous | equal DRAM, extra NoC; occupancy only. Deferred |
| 4 | tile-rows, row streamed (R5) | `2-3 * (X + Rs) + O` | 0 | rejected |

Chosen: R1. It is also the cheapest-traffic split (tied with R3, which adds transactions).

### Block schedule

```cpp
stage_affine();                                   // reader, once per kernel, behind block 0's reads
for (uint32_t block_idx = 0; block_idx < num_blocks_this_core; ++block_idx) {
    load_block(block_idx);                        // reader: R_b x Wt tiles of x (and res) in C-tile chunks
    residual_add_block(block_idx);                // compute (HAS_RESIDUAL): z = x + res -> cb_row_work_a
    row_mean_block(block_idx);                    // compute: mean[r] = sum_w z / W          -> cb_row_mean
    center_block(block_idx);                      // compute: c = z - mean (col bcast)       -> cb_row_work_b
    row_sumsq_block(block_idx);                   // compute: S[r] = sum_w c*c in DEST       -> cb_row_sumsq
    row_inv_std_block(block_idx);                 // compute: rsqrt(collapse(S)/W + eps)     -> cb_inv_std
    normalize_block(block_idx);                   // compute: c * inv_std (col bcast)        -> cb_row_work_a
    affine_scale_block(block_idx);                // compute: * gamma (row bcast)            -> cb_row_work_b
    affine_shift_block(block_idx);                // compute: + beta (row bcast)             -> cb_output_tiles
    store_block(block_idx);                       // writer: R_b x Wt tiles in C-tile chunks
}
```

`R_b` = `R` for every block except the last, which is `last_block_extent`. Without residual, `row_mean_block` and `center_block` read `cb_input_tiles` directly (`center_block` pops it) and `residual_add_block` is absent.

| Operation | Block shape | Resident across it | Intended fixed-cost frequency |
|---|---|---|---|
| `stage_affine` | `2 x Wt` tiles (row 0 only) | - | once per kernel: TILE affine = 2 reads of 64 B per tile (page offsets 0 and 512); RM affine = 1 read of 64 B per tile into slot offset 0 + one 32 B self-aimed local copy offset 32 -> 512; one shared barrier with block 0's reads; one `push_back(Wt)` per CB |
| `load_block` | `R_b x Wt` tiles per tensor | previous block in the other `in_depth` slot | per C-chunk: C async reads per tensor, one `noc_async_read_barrier`, one `push_back(C)` per tensor |
| `residual_add_block` | `R_b x Wt` | `cb_row_work_a` filling | one chain call per block (init + reconfig once); per-tile input pop / output push |
| `row_mean_block` | `R_b x Wt` reduced to `R_b x 1` | source block (no pop) | one reduce call per block: one upfront wait, one SFPU macro load, one bulk reserve of `R_b`; pairwise `add_tiles` per row, one collapse + 1/W per row |
| `center_block` | `R_b x Wt` | `cb_row_mean` (popped at end) | one chain call per block |
| `row_sumsq_block` | `R_b x Wt` accumulated to `R_b x 1` | `cb_row_work_b` (caller waits `R_b * Wt` once, no pop) | one chain call per block; one DEST acquire + pack per row |
| `row_inv_std_block` | `R_b x 1` | `cb_row_work_b` | one reduce call per block; post-op (mul 1/W, add eps, rsqrt) once per row on one DEST tile |
| `normalize_block` | `R_b x Wt` | `cb_inv_std` (popped at end) | one chain call per block |
| `affine_scale_block` | `R_b x Wt` | `cb_gamma_tiles` (never popped until kernel end) | one chain call per block |
| `affine_shift_block` | `R_b x Wt` | `cb_beta_tiles` (same) | one chain call per block |
| `store_block` | `R_b x Wt` | - | per C-chunk: one `wait_front(C)`, C async writes, `noc_async_writes_flushed()`, one `pop_front(C)`; one write barrier per kernel end |

`compute_kernel_hw_startup(cb_input_tiles, cb_gamma_tiles, cb_output_tiles)` once, first statement of the compute kernel. No unit-at-a-time loop in the schedule; per-tile traversal lives inside the helpers.

### Perf lamps

| Lamp | Why the default may be wrong here | Nearby alternative to measure |
|------|-----------------------------------|-------------------------------|
| overlap (block extent) | `MIN_BLOCKS_PER_CORE = 2` trades fewer per-block inits for read/compute/write overlap; with 1-2 tile-rows per core on the headline it decides whether the writer and reader ever overlap compute | `MIN_BLOCKS_PER_CORE` in {1, 2, 4} (R = whole assignment vs R = 1) on the 8 reference rows |
| `io_chunk_tiles` (catalog `double_buffer`, `trid_double_issue`) | 8 tiles per barrier is the catalog sweet spot, but the CB holds whole blocks, so a full-row barrier (C = 24) may tie it, and trids may beat both at small C | C in {4, 8, 24}; trid ring depth 3 at C = 8 |
| `split_reader` (catalog `split_reader`) | the reader RISC issues both activation streams (2 x 24 reads per tile-row) while the writer RISC idles until `affine_shift_block`; on the residual rows the reader may be issue-bound | residual stream issued from the writer RISC (BRISC, NoC1) into `cb_residual_tiles` (ownership unchanged: writer becomes its producer, reader no longer touches it) |
| compute-bound (catalog `compute_fusion`) | 7 FPU passes x 24 tiles per tile-row at HiFi4 with fp32 work pages may exceed the per-core DM time on 2-row cores | `CKL_ELTWISE_CHAIN_SKIP_COMPUTE=1` profile split; then the fused normalize: `BinaryFpu Mul(c, inv_std Col)` + `DestReuseBinary Mul/Add` against gamma/beta tiles pre-expanded to all 32 rows (catalog says FPU dest-reuse usually loses; measure) |
| `bank_stagger` (catalog) | every core starts its tile-row at page `t * 24`; on BH the start banks may align across cores | start each core's chunk walk at chunk `core_idx % (Wt / C)` and wrap |
| `noc_placement` (catalog) | `row_wise=True` spreads cores along the DRAM-facing axis; the 110-core grid is nearly full so the effect may vanish | `row_wise=False` |
| occupancy (R2 / R3 trigger) | T = 32 rows use 32 of 110 cores; T = 128 leaves an 18-core tail | measure 32x32 and 8x512 per-core busy time vs total; R2 / R3 are the regimes that answer it |
| grid synchronization | does not apply: R1 has no collective | - |

## Dataflow Strategy

| Stage | Format | Mechanism | Notes |
|-------|--------|-----------|-------|
| `x` tiles (DRAM or L1 banks) -> `cb_input_tiles` | bf16 tile, 2 KiB page | reader (RISCV_1, NoC0), `TensorAccessor.get_noc_addr(page)`, C reads per barrier | page id `t * Wt + w` |
| `res` tiles -> `cb_residual_tiles` | bf16 tile | same reader, same chunking, independent TensorAccessor (placement independent of `x`) | CT `HAS_RESIDUAL` |
| gamma / beta -> `cb_gamma_tiles` / `cb_beta_tiles` | bf16 tile, only row 0 meaningful | reader, once. TILE: page k, two 64 B reads at offsets 0 and 512 landing at the same offsets of slot k (face 0 row 0, face 1 row 0; the extra row-1 bytes are ignored). ROW_MAJOR (page width P elements): element `32k` lives in page `32k // P` at byte `(32k % P) * 2`; one 64 B read into slot k offset 0, then a 32 B self-aimed local copy (`local_copy_helpers_dataflow.hpp` `local_addr`/`read_with_state`) from slot offset 32 to slot offset 512 | rows 1-31 of the slot are never read by a Row broadcast |
| compute phases | `cb_row_work_a/b`: Float32 when `fp32_dest_acc_en` else Float16_b; stats same rule | CB credit between phases; alias pair gives each phase a real push->wait edge over the same bytes | see Circular Buffers |
| `cb_output_tiles` -> output tiles | bf16 tile | writer (RISCV_0, NoC1), C writes per flush, TensorAccessor of the output (L1 or DRAM per `memory_config`) | output pad rows carry finite garbage |

Placement axis view (TARGET `memory_layout`, for later): HEIGHT_SHARDED cuts `row` (independent) = knob-turn: R1 with `cb_input_tiles` / `cb_residual_tiles` / `cb_output_tiles` backed on the shards (`ttnn.cb_descriptor_from_sharded_tensor`, zero-copy, no TensorAccessor re-read), one block = whole shard. WIDTH_SHARDED / BLOCK_SHARDED cut `w` (dependent) = scheme-change = R2's Chan combine with the slice resident in place. None is in TARGET today.

R2 Tensix-to-Tensix contract (kept reachable): groups of `k` cores along a grid row share tile-rows; after `row_sumsq_block` each core writes its `(m_k, M2_k)` col-0 tiles into a dedicated handoff CB (writer is its consumer), all-gathers the group's `2k` tiles via `ttnn.Mcast` (`McastConfig` row-major groups) into a gather CB consumed by compute, which combines `mean = avg_k m_k`, `M2 = sum_k M2_k + (W/k) * sum_k (m_k - mean)^2`, then `inv_std`, and applies `c -= (mean - m_k)` before `normalize_block`. Stall-shadow for R2: `residual_add_block` + `row_mean_block` of block b+1 run inside the rendezvous of block b (rows independent; no floating-point reorder).

## Work Distribution

| Field | Value |
|-------|-------|
| Work unit | one tile-row (32 rows x W); blocks are `R` work units |
| Grid | `device.compute_with_storage_grid_size()` (110 cores on BH p100a), a parameter, never a literal |
| Split | `(num_cores, all_cores, group_1, group_2, rows_g1, rows_g2) = ttnn.split_work_to_cores(grid, tensor_row_tiles, row_wise=True)`; cores enumerated in the same row-wise order; contiguous ranges |
| Per-core work | RT args `row_tile_start`, `core_row_tiles` (reader, compute, writer); `num_blocks_this_core = ceil(core_row_tiles / R)`, `last_block_extent = core_row_tiles - (num_blocks_this_core - 1) * R`, derived on device from the CT R |
| Remainder | `split_work_to_cores` groups (`rows_g1 = rows_g2 + 1`); tile-row count is `prod(shape[:-2]) * ceil(H/32)` (per-image `ceil`) |
| Production numbers | 8x512: T = 128 -> 18 cores x 2 rows, 92 x 1, R = 1, 2 blocks on the 2-row cores. 8x384: T = 96 -> 96 x 1. 32x32: T = 32 -> 32 x 1. 3x500: T = 48. 64x17: T = 64 |
| Idle cores | cores outside `all_cores` get no kernels |

One regime -> no selection-dependent tests. Every CB descriptor uses `all_cores`.

## Circular Buffers

`Wt` = `tensor_w_tiles`; `tb` = bf16 tile bytes (2048); `wb` = 4096 when `fp32_dest_acc_en` else 2048; `wfmt` = Float32 when `fp32_dest_acc_en` else Float16_b.

| Semantic Name | Index | Page Size | Num Pages | Sizing rationale | Format | Producer | Consumer | Lifetime |
|---------------|-------|-----------|-----------|------------------|--------|----------|----------|-----------|
| `cb_input_tiles` | 0 | tb | `in_depth * R * Wt` | spans `row` (R) and `w` (Wt): no-residual needs the block resident for mean + center; `in_depth` blocks for prefetch | Float16_b | reader | compute | per block (popped by `residual_add_block` or `center_block`) |
| `cb_residual_tiles` | 1 | tb | `in_depth * R * Wt` (only if `HAS_RESIDUAL`) | streamed by `residual_add_block`; block-sized so the reader can run a whole block ahead | Float16_b | reader | compute | per block |
| `cb_gamma_tiles` | 2 | tb | `Wt` | whole row of row-0 tiles, persistent | Float16_b | reader | compute | whole kernel (compute pops `Wt` at kernel end) |
| `cb_beta_tiles` | 3 | tb | `Wt` | same | Float16_b | reader | compute | whole kernel |
| `cb_row_work_a` | 4 | wb | `R * Wt` (alias group with `cb_row_work_b`, one `CBDescriptor`, `total_size = R * Wt * wb`) | spans `row` and `w`: holds `z` (or the normalized block) resident | wfmt | compute | compute | per block |
| `cb_row_work_b` | 5 | wb | `R * Wt` (same allocation) | holds `c` (or the scaled block) resident | wfmt | compute | compute | per block |
| `cb_row_mean` | 6 | wb | `R` | one col-0 tile per tile-row; first indexed window of its id per block | wfmt | compute | compute | `row_mean_block` -> `center_block` |
| `cb_row_sumsq` | 7 | wb | `R` | one raw DEST-accumulated tile per tile-row | wfmt | compute | compute | `row_sumsq_block` -> `row_inv_std_block` |
| `cb_inv_std` | 8 | wb | `R` | one col-0 tile per tile-row | wfmt | compute | compute | `row_inv_std_block` -> `normalize_block` |
| `cb_output_tiles` | 16 | tb | `out_depth * R * Wt` (`out_depth = 1`) | one block so compute never stalls on the writer inside `affine_shift_block` | Float16_b | compute | writer | per block |

No scaler CB: both reduces use `ReduceAlgorithm::AccumulateViaAdd` on tile-aligned W, which never reads the scaler CB (`reduce_helpers_compute.inl:320-337`); the `scaler_dfb_id` template slot is bound to `cb_gamma_tiles` (distinct from every reduce input/output). Reduce scaler format / pool-type-aware overload therefore does not apply.

CB sync (push = wait, per block, `n = R_b * Wt`):

| CB | Pushes | Waits / pops |
|---|---|---|
| `cb_input_tiles` | reader `n / C` pushes of C | residual: `residual_add_block` waits/pops per tile, n. No residual: `row_mean_block` waits n upfront (no pop), `center_block` pops per tile, n |
| `cb_residual_tiles` | reader `n / C` x C | `residual_add_block` per tile, n |
| `cb_row_work_a` | `residual_add_block` n (per tile); `normalize_block` n | `row_mean_block` waits n (no pop); `center_block` pops n; `affine_scale_block` pops n |
| `cb_row_work_b` | `center_block` n; `affine_scale_block` n | caller waits n before `row_sumsq_block` (no pop); `normalize_block` pops n; `affine_shift_block` pops n |
| `cb_row_mean` | `row_mean_block` bulk-reserves and pushes `R_b` | `center_block` waits `R_b` upfront, pops `R_b` at end |
| `cb_row_sumsq` | `row_sumsq_block` `R_b` (one per row, PerOuter) | `row_inv_std_block` BulkWaitBulkPop `R_b` |
| `cb_inv_std` | `row_inv_std_block` `R_b` | `normalize_block` waits `R_b` upfront, pops at end |
| `cb_gamma_tiles` / `cb_beta_tiles` | reader `Wt` once | compute waits `Wt` once (upfront, Row mapping), pops `Wt` at kernel end |
| `cb_output_tiles` | `affine_shift_block` n per tile | writer `n / C` x C |

Alias-pair lockstep (residual path; per block each id is pushed and popped exactly twice by `n`): `a`: push(residual_add) -> read(mean) -> pop(center) -> push(normalize) -> pop(affine_scale). `b`: push(center) -> read(sumsq) -> pop(normalize) -> push(affine_scale) -> pop(affine_shift). No-residual path: `a` is pushed/popped once (normalize / affine_scale), `b` twice. Every read of a page happens through the id that last wrote it; every write to a page happens after the page's last read in program order (unpack is in program order and each tile is unpacked before it is packed).

## Block Operation Realization

| # | Block operation | Block shape | Helper? | Input CB (semantic name, pages, state) | Output CB (semantic name, pages) | CB state after |
|---|-----------------|-------------|---------|----------------------------------------|----------------------------------|----------------|
| 0 | `stage_affine` | `2 x Wt` | dataflow (TensorAccessor + local copy helper) | gamma/beta tensors | `cb_gamma_tiles` Wt, `cb_beta_tiles` Wt | persistent until kernel end |
| 1 | `load_block` | `R_b x Wt` per tensor | dataflow | input / residual tensors | `cb_input_tiles` n, `cb_residual_tiles` n | - |
| 2 | `residual_add_block` | `grid(R_b, Wt)` | yes: `add` | `cb_input_tiles` n (per tile, popped), `cb_residual_tiles` n (per tile, popped) | `cb_row_work_a` n | `a` full |
| 3 | `row_mean_block` | `of(R_b, Wt)` -> `R_b` | yes: `reduce_mean` | `cb_row_work_a` (or `cb_input_tiles`) n, upfront, not popped | `cb_row_mean` R_b | source still resident |
| 4 | `center_block` | `grid(R_b, Wt)` | yes: `sub` | source n (per tile, popped); `cb_row_mean` R_b (Col, upfront, popped at end) | `cb_row_work_b` n | source empty, `b` full |
| 5 | `row_sumsq_block` | `grid(R_b, Wt)` -> `R_b` | yes: `eltwise_chain` (BinaryFpu Mul + PackTile, PerRow DEST accumulation) | `cb_row_work_b` n, caller-waited, both operands, not popped | `cb_row_sumsq` R_b | `b` still resident |
| 6 | `row_inv_std_block` | `of(R_b, 1)` | yes: `reduce` + post op | `cb_row_sumsq` R_b (BulkWaitBulkPop) | `cb_inv_std` R_b | - |
| 7 | `normalize_block` | `grid(R_b, Wt)` | yes: `mul` | `cb_row_work_b` n (per tile, popped); `cb_inv_std` R_b (Col, upfront, popped at end) | `cb_row_work_a` n | - |
| 8 | `affine_scale_block` | `grid(R_b, Wt)` | yes: `mul` | `cb_row_work_a` n (per tile, popped); `cb_gamma_tiles` Wt (Row, upfront, never popped) | `cb_row_work_b` n | - |
| 9 | `affine_shift_block` | `grid(R_b, Wt)` | yes: `add` | `cb_row_work_b` n (per tile, popped); `cb_beta_tiles` Wt (Row, upfront, never popped) | `cb_output_tiles` n | - |
| 10 | `store_block` | `R_b x Wt` | dataflow | `cb_output_tiles` n | output tensor | - |

## API Mapping

Paths relative to `ttnn/cpp/ttnn/kernel_lib/` unless absolute.

| Block operation | Type | Function | File:Line | Template Params / Args | Input CB | Output CB | Which params are block knobs |
|-----------------|------|----------|-----------|------------------------|----------|-----------|------------------------------|
| boot | raw_api (mandatory boot) | `compute_kernel_hw_startup` | `eltwise/api/chain.hpp:24-40` (caller-init contract); `reduce_helpers_compute.hpp:31-35` | `(cb_input_tiles, cb_gamma_tiles, cb_output_tiles)` | - | - | none |
| `residual_add_block` | helper | `add<AInput, BInput, Output>` | `eltwise/api/convenience.hpp:52` | `input(cb_input_tiles)` (PerTile/PerTile, Block), `input(cb_residual_tiles)` (`BroadcastDim::None`, PerTile/PerTile), `output(cb_row_work_a)` (PerTile/PerTile); shape `IterationShape::grid(R_b, Wt)` | `cb_input_tiles`, `cb_residual_tiles` | `cb_row_work_a` | `R_b` (runtime), `Wt` |
| `row_mean_block` | helper | `reduce_mean<REDUCE_ROW, src, cb_gamma_tiles, cb_row_mean, ReduceInputPolicy::WaitUpfrontNoPop, ReduceDataFormatReconfigMode::INPUT_AND_OUTPUT, ReduceFp32Mode::Fast, ReduceAlgorithm::AccumulateViaAdd>` | `reduce_helpers_compute.hpp:691`; no-pop + bulk-reserve behavior `reduce_helpers_compute.inl:183-186, 338-343` | `ReduceInputBlockShape::of(R_b, Wt)`, `n_reduced = W` | `cb_row_work_a` (residual) / `cb_input_tiles` | `cb_row_mean` | `R_b`, `Wt` |
| `center_block` | helper | `sub<AInput, BInput, Output>` | `eltwise/api/convenience.hpp:61`; `BroadcastDim::Col` `eltwise/api/chain.hpp:330`; `InputTileMapping::Col` staged window `eltwise/api/chain.hpp:244-262` | A = `input(src)` PerTile/PerTile Block; B = `input(cb_row_mean, BroadcastDim::Col, WaitPolicy::Upfront, PopPolicy::AtEnd, InputTileMapping::Col)`; `output(cb_row_work_b)` PerTile; `grid(R_b, Wt)` | src, `cb_row_mean` | `cb_row_work_b` | `R_b`, `Wt` |
| `row_sumsq_block` | helper | `eltwise_chain(grid(R_b, Wt), BinaryFpu<BinaryFpuOp::Mul, input(cb_row_work_b, WaitPolicy::None, PopPolicy::None, InputTileMapping::Block), input(cb_row_work_b, BroadcastDim::None, WaitPolicy::None, PopPolicy::None, InputTileMapping::Block), Dst::D0, DestAccumulation::PerRow>{b, b}, PackTile<output(cb_row_sumsq, ReservePolicy::PerOuter, PushPolicy::PerOuter, DataFormatReconfig::Enabled, TileAddressing::Direct, DestAccumulation::PerRow)>{sumsq})` | `eltwise/api/chain.hpp:568` (BinaryFpu), `:588` (PackTile), `:620` (eltwise_chain), `:308-312` (DestAccumulation); PerRow pack contract `eltwise/core/chain.inl:1139-1148` | caller: `b.wait_front(R_b * Wt)` before the call (no pop) | `cb_row_work_b` | `cb_row_sumsq` | `R_b`, `Wt` |
| `row_inv_std_block` | helper | `reduce<PoolType::SUM, REDUCE_ROW, cb_row_sumsq, cb_gamma_tiles, cb_inv_std, ReduceInputPolicy::BulkWaitBulkPop, INPUT_AND_OUTPUT, ReduceFp32Mode::Fast, ReduceAlgorithm::AccumulateViaAdd, ReduceWithinTile::Collapse>` | `reduce_helpers_compute.hpp:637`; post-op contract `reduce_helpers_compute.hpp:519-523` | `ReduceInputBlockShape::of(R_b, 1)`, `contiguous()`, `NoAccumulation{}`, post-op `[](uint32_t dst){ binop_with_scalar_tile_init(); mul_unary_tile(dst, inv_w_bits); add_unary_tile(dst, eps_bits); rsqrt_tile_init(); rsqrt_tile(dst); }` (`tt_metal/hw/inc/api/compute/eltwise_unary/binop_with_scalar.h:35, 156`, `rsqrt.h:21`) | `cb_row_sumsq` | `cb_inv_std` | `R_b` |
| `normalize_block` | helper | `mul<AInput, BInput, Output>` | `eltwise/api/convenience.hpp:70` | A = `input(cb_row_work_b)` PerTile/PerTile; B = `input(cb_inv_std, BroadcastDim::Col, WaitPolicy::Upfront, PopPolicy::AtEnd, InputTileMapping::Col)`; `output(cb_row_work_a)`; `grid(R_b, Wt)` | `cb_row_work_b`, `cb_inv_std` | `cb_row_work_a` | `R_b`, `Wt` |
| `affine_scale_block` | helper | `mul<...>` | `eltwise/api/convenience.hpp:70`; `BroadcastDim::Row` / `InputTileMapping::Row` `eltwise/api/chain.hpp:244-262, 330-335` | A = `input(cb_row_work_a)` PerTile/PerTile; B = `input(cb_gamma_tiles, BroadcastDim::Row, WaitPolicy::Upfront, PopPolicy::None, InputTileMapping::Row)`; `output(cb_row_work_b)`; `grid(R_b, Wt)` | `cb_row_work_a`, `cb_gamma_tiles` | `cb_row_work_b` | `R_b`, `Wt` |
| `affine_shift_block` | helper | `add<...>` | `eltwise/api/convenience.hpp:52` | A = `input(cb_row_work_b)`; B = `input(cb_beta_tiles, BroadcastDim::Row, WaitPolicy::Upfront, PopPolicy::None, InputTileMapping::Row)`; `output(cb_output_tiles)`; `grid(R_b, Wt)` | `cb_row_work_b`, `cb_beta_tiles` | `cb_output_tiles` | `R_b`, `Wt` |
| `load_block`, `store_block` | raw_api (dataflow) | `TensorAccessor::get_noc_addr`, `noc_async_read/write`, barriers / `noc_async_writes_flushed` | `tt_metal/hw/inc/api/tensor/tensor_accessor.h:155` | page id `t * Wt + w` | - / `cb_output_tiles` | `cb_input_tiles`, `cb_residual_tiles` / - | `C`, `R_b` |
| `stage_affine` | raw_api (dataflow) + helper | `TensorAccessor::get_noc_addr(page, offset)`; `dataflow_kernel_lib::local_addr` / `read_with_state` for the 32 B self copy | `tt_metal/hw/inc/api/tensor/tensor_accessor.h:155`; `local_copy_helpers_dataflow.hpp:57, 75, 95` | TILE: offsets {0, 512}; RM: `(page = 32k / P, offset = (32k % P) * 2)` | - | `cb_gamma_tiles`, `cb_beta_tiles` | none (once per kernel) |

Helpers considered and rejected (raw_api rows):

| Raw entry | Helper considered | Mismatch (file:line) |
|---|---|---|
| `load_block` / `store_block` | `tilize_helpers_dataflow.hpp`, `dfb_helpers_dataflow.hpp` | no kernel-lib dataflow helper reads/writes interleaved tile pages through a TensorAccessor with chunked barriers; the tilize dataflow helpers stage row-major sticks (`tilize_helpers_dataflow.hpp`), not TILE pages |
| `stage_affine` | `reduce_helpers_dataflow.hpp` scaler fills (`prepare_reduce_scaler`) | they synthesize a constant tile in L1 (`reduce_helpers_dataflow.hpp`), they cannot gather row 0 of a remote tensor's tiles |
| `row_inv_std_block` post-op lambda (`mul_unary_tile`, `add_unary_tile`, `rsqrt_tile`) | `eltwise_chain` `MulUnary` / `AddUnary` / `Rsqrt` structs (`eltwise/unary/scalar.hpp:26-33`, `eltwise/unary/math.hpp:36-38`) | the reduce helper's only hook on the reduced DEST tile is a `uint32_t dst` callable (`reduce_helpers_compute.hpp:519-523`); a separate chain would pack and re-unpack `R_b` tiles to apply three SFPU ops the callback applies in DEST (catalog `compute_fusion`: SFPU consumers fuse) |

Catalog entries that set knobs: `double_buffer` + `trid_double_issue` -> `io_chunk_tiles` = 8 and `in_depth` = 2; `noc_placement` -> `row_wise=True`, reads NoC0 / writes NoC1; `row_reduce_accumulate` + `reduce_accumulate` -> AccumulateViaAdd for the 24-tile mean (row wins from 4 tiles); `compute_fusion` -> post-reduce SFPU ops fused in the reduce callback, FPU affine steps kept as L1 round trips; `sfpu_tile_scope` -> not applied (post-op runs on `R_b` tiles per block, negligible); `split_reader`, `bank_stagger` -> lamps.

## Broadcast Verification

| Phase | Op | CB_A (semantic name) Valid Region | CB_B (semantic name) Valid Region | Broadcast Dim |
|-------|-----|-----------------------------------|-----------------------------------|---------------|
| `residual_add_block` | add | `cb_input_tiles`: All | `cb_residual_tiles`: All | None |
| `center_block` | sub | `cb_row_work_a` / `cb_input_tiles`: All | `cb_row_mean`: Col0 (REDUCE_ROW out) | Col |
| `row_sumsq_block` | mul | `cb_row_work_b`: All | `cb_row_work_b`: All | None |
| `normalize_block` | mul | `cb_row_work_b`: All | `cb_inv_std`: Col0 | Col |
| `affine_scale_block` | mul | `cb_row_work_a`: All | `cb_gamma_tiles`: Row0 | Row |
| `affine_shift_block` | add | `cb_row_work_b`: All | `cb_beta_tiles`: Row0 | Row |

## Key Risks and Gotchas

| Risk | Why it bites here | Mitigation in this design |
|------|-------------------|---------------------------|
| Indexed window straddling the ring end | `row_mean_block`, `row_sumsq_block`, and the Col / Row operands index into resident windows; a single rotating work CB or one shared stat FIFO would wrap on the ragged final block | alias pair + separate stat CBs; every indexed window is its id's first push in the block (Mechanism caps) |
| Alias pair pointer drift | two ids, one allocation; a phase that moves fewer than `R_b * Wt` tiles desynchronizes them | every phase moves the whole block through each id it touches; the implementer must not add partial pushes |
| Scaler slot bound to `cb_gamma_tiles` | AccumulateViaAdd never waits it on aligned input; switching a reduce to ReduceTile, enabling a partial mask, or `CopySeedZeroPair` would read gamma as a scaler | keep `ReduceAlgorithm::AccumulateViaAdd` explicit on both reduces; validate rejects W % 32 != 0 |
| Work / stat page format vs DEST width | a Float16_b work page under fp32 DEST truncates `z` and `c` between phases (ledger audit 2) | page format derives from `fp32_dest_acc_en` in one place |
| Gamma / beta from DRAM at sub-tile granularity | BH DRAM reads need 64 B alignment; 32 B reads at offset 32 return shifted bytes | always 64 B reads at 64 B-aligned offsets; RM upper half moved by an L1 self copy |
| L1 budget when activations live in L1 | input / residual / output in L1 take bank space; the CB region must end below them | budget = `largest_contiguous_bytes_free_per_bank` queried after output allocation |
| Mean / variance precision | FPU unpacks Float32 pages as TF32; a bf16 scaler `1/768` would be inexact | no scaler tile: 1/W and eps applied in fp32 on the SFPU (`reduce_mean` n_reduced, post-op bit patterns) |
| Prompt import path | golden tests import `ttnn.operations.layer_norm_tile`; the prompt mandates `ttnn.operations.layer_norm` | re-export package; acceptance test checks both resolve to one function |
| Tile-row count on non-aligned H | `floor(B*S/32)` under-counts per-image padding (e.g. 3x500 -> 46 instead of 48) | `prod(shape[:-2]) * ceil(H/32)` everywhere |
