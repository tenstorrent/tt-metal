# Operation Design: rms_norm

## Overview

| Field | Value |
|-------|-------|
| Classification | fused (row reduction + elementwise normalize, one device program) |
| Goal | `y = x * rsqrt(mean(x^2, dim=-1) + eps) * gamma`, one `ttnn.generic_op` dispatch per call, rows spread across the full compute grid from Phase 0 |
| Math | `rinv[row] = 1 / sqrt( (sum_{w < W} x[row, w]^2) / W + epsilon )`; `out[row, w] = x[row, w] * rinv[row] * (gamma[w] if gamma else 1)` |
| Mode | Derivative (new op on kernel-lib helpers; host model = ProgramDescriptor) |
| References | `eval/golden_tests/rms_norm/feature_spec.py` (TARGET/INPUTS/INVALID), `eval/golden_tests/rms_norm/helpers.py` (`pytorch_rms_norm`, TOLERANCES), `.claude/references/blocking-model.md`, `.claude/references/l1-footprint-discipline.md`, `.claude/references/precision_convention.md`, `.claude/references/generic_op_template/`, `ttnn/ttnn/operations/examples/master.md` (entries cited per knob below), `ttnn/cpp/ttnn/kernel_lib/*` (helpers cited in API Mapping) |

### Entry point contract

| Item | Decision |
|------|----------|
| Signature | `rms_norm(input_tensor, *, gamma=None, epsilon=1e-6, compute_kernel_config=None) -> ttnn.Tensor` |
| Default config | `default_compute_kernel_config()` exported from the op module (factory, fresh object per call): `ComputeConfigDescriptor(math_fidelity=HiFi4, fp32_dest_acc_en=True, math_approx_mode=False)`. `None` resolves through it; nothing else restates the default. |
| Config pass-through | After `validate()`, the resolved descriptor is passed as-is: `config=compute_kernel_config` on the compute `KernelDescriptor`. `math_fidelity` / `math_approx_mode` are never gated. |
| Dispatch | Exactly one `ttnn.generic_op([input, (gamma,) output], program_descriptor)` per call. No host-side `to_layout`/`tilize`/`untilize`/`pad`/`slice`. |
| Validation order | 1. rank < 2 -> `ValueError`. 2. gamma present and `gamma.shape[-1] != input.shape[-1]` or `gamma.volume() != input.shape[-1]` -> `ValueError`. 3. SUPPORTED / EXCLUSIONS gate (raises the `_op_contract` `NotImplementedError` subclasses). 4. Regime predicate (R1 footprint, see Regimes) -> `NotImplementedError` naming the deferred regime. |
| Output allocation | Same shape, dtype and layout as input; memory config = input's (interleaved DRAM in the golden suite). Output padding contents are unspecified. |

### Phase 0 scope (feature_spec TARGET axes, by name)

The implementer writes `SUPPORTED`; this is the rectangle the design commits Phase 0 to.

| Axis (TARGET name) | Phase 0 values | Refinement candidates (TARGET - Phase 0) | Cost class |
|---|---|---|---|
| `dtype` | `float32`, `bfloat16` | `bfloat8_b` | knob-turn (CB data formats; TILE + tile_aligned only per INVALID) |
| `fp32_dest_acc_en` | `True` | `False` (bf16 only) | knob-turn (CB page formats follow DEST width, ledger audit 2) |
| `layout` | `TILE_LAYOUT` | `ROW_MAJOR_LAYOUT` | knob-turn: adds `tilize_block` / `untilize` phases and the RM CBs already specified below; no change to the row scheme |
| `alignment` | `tile_aligned`, `h_non_aligned`, `w_non_aligned` | none | built in Phase 0 (H padding is row-independent; W padding is the masked-last-column path) |
| `rank` | 2, 3, 4 | none | leading dims fold into the row axis |
| `gamma_mode` | `gamma`, `no_gamma` | none | |
| `gamma_dtype` | `float32`, `bfloat16`, `"none"` | `bfloat8_b` | knob-turn (gamma staging extraction from a block-float tile) |
| `gamma_layout` | `ROW_MAJOR_LAYOUT`, `"none"` | `TILE_LAYOUT` | knob-turn: only the one-time gamma staging read changes (row 0 of each gamma tile, faces 0/1) |
| `memory_layout` | `INTERLEAVED` | `HEIGHT_SHARDED` (knob-turn), `WIDTH_SHARDED`, `BLOCK_SHARDED` (scheme-change, regime R2) | see Dataflow Strategy, placement mapping |

`EXCLUSIONS` (op-side, xfail-strict): `{dtype: float32, fp32_dest_acc_en: False}` (natively rejected per `precision_convention.md`). `"none"` is always accepted for `gamma_dtype` / `gamma_layout`; `tag_gamma_dtype` / `tag_gamma_layout` return `"none"` when gamma is `None`.

`INPUT_TAGGERS` required by feature_spec: `tag_alignment` (three-valued: `tile_aligned` / `w_non_aligned` (W % 32 != 0) / `h_non_aligned` (W aligned, H % 32 != 0)), `tag_rank`, `tag_gamma_dtype`, `tag_gamma_layout`.

Structural impossibilities: none beyond those already in `feature_spec.INVALID`.

## Parameters

| Name | Type | Required | Valid Range | Default | CT/RT |
|------|------|----------|-------------|---------|-------|
| `input_tensor` | `ttnn.Tensor` | yes | rank >= 2, on device | - | - |
| `gamma` | `ttnn.Tensor` | no | shape `(1,1,1,W)` (volume W), last dim == input last dim | `None` | presence: CT flag `has_gamma` |
| `epsilon` | float | no | > 0 | `1e-6` | CT (fp32 bit pattern) |
| `compute_kernel_config` | `ttnn.ComputeConfigDescriptor` | no | `fp32_dest_acc_en=True` in Phase 0 | `default_compute_kernel_config()` | compute kernel config |
| derived `inv_w` | float | - | `1.0 / W` (true W, never padded) | - | CT (fp32 bit pattern) |

## Tensors

### Input

| Property | Requirement |
|----------|-------------|
| Shape | `[..., H, W]`, rank 2..4; any H, W (non-tile-aligned allowed) |
| Dtype | bfloat16, float32 |
| Layout | TILE (Phase 0); ROW_MAJOR (refinement 1) |
| Memory | interleaved (DRAM or L1) |
| TILE padding | must be finite (masked lanes are multiplied by 0); ttnn tilize fills 0, the poison suite fills 1000.0 |

### Gamma

| Property | Requirement |
|----------|-------------|
| Shape | `(1, 1, 1, W)` |
| Dtype | bfloat16, float32 (independent of input dtype) |
| Layout | ROW_MAJOR (one page = the whole W-element stick) |
| Memory | interleaved |

### Output

| Property | Value |
|----------|-------|
| Shape | identical to input |
| Dtype | input dtype |
| Layout | input layout |
| Memory | input memory config (interleaved) |

## Blocking Model

### Axes

Logical tile index space: `(row, w)`. Leading dims fold into `row` because the op treats them identically and they are contiguous in both layouts' page order.

| Axis | Character (+ one-clause reason) | Extent knob | Phase 0 value | Knob source | Core-assignment | Later unlock |
|------|--------------------------------|-------------|---------------|-------------|-----------------|--------------|
| `row` (32-row groups: TILE `prod(shape[:-2]) * ceil(H/32)`, RM `ceil(prod(shape[:-1]) / 32)`) | independent: each output row depends only on its own input row | `block_row_tiles` (R) | `ceil(core_row_tiles_max / MIN_BLOCKS_PER_CORE)`, `MIN_BLOCKS_PER_CORE = 2`, then shrunk only by the L1 selection function | host `select_blocking()` -> compute/reader/writer CT arg | contiguous tile-row ranges over the full compute grid via `split_work_to_cores(grid, tensor_row_tiles, row_wise=True)` | knob-turn (R, grid size, HEIGHT_SHARDED placement) |
| `w` (tile columns, `ceil(W/32)`) | dependent: `rinv[row]` spans every column of the row | block extent along `w` = `core_w_tiles` (whole row resident; in R1 `core_w_tiles == tensor_w_tiles`) | `tensor_w_tiles` (whole row; not split) | host `select_blocking()` CT arg | not assigned across cores in R1 (each core owns full rows); R2 splits it across a core group | scheme-change (R2: cross-core combine) |
| `w` streaming sub-extent of `normalize_block` (and of RM tilize/untilize) | same axis; the normalize stage is elementwise along `w` once `rinv` exists | `w_chunk_tiles` (C) | largest legal C that fits L1 at the chosen R, floor `min(Wt, W_CHUNK_FLOOR=8)`, ceiling `tensor_w_tiles` | host `select_blocking()` CT arg | follows the row owner | knob-turn |

Why the `w` block extent is the whole row: `x` is consumed twice (sum of squares, then normalize), so a block that does not span the full reduced extent forces a second DRAM read of `x` (rejected regime R3). The row axis carries the parallelism; `w` stays whole per core until R2.

Intermediate data re-run of the table (every stage's output has axes too):

| Stage output | Axes | Assignment | Note |
|---|---|---|---|
| `sumsq` (S[row], one tile per row, raw elementwise sums) | `row` (independent), `w` collapsed to 1 tile (2 when the masked last column is separate) | same core as the row | no collector core in R1 |
| `rinv` (one col-0 tile per row) | `row` (independent) | same core as the row | no collector core in R1; in R2 the combine of `rinv` is spread across the group by row (see R2 row) |
| normalized chunk | `row` x `w`-chunk | same core | streams to writer |

Operand-reuse check (every operand vs the chosen split = rows across cores):

| Operand | Varies along `row`? | Consequence |
|---|---|---|
| `x` | yes | no reuse across cores |
| `gamma` | no | reuse-shared by construction of the row split -> broadcast regime R5 (deferred, bounded re-read) |
| reduce scaler (constant 1.0) | no | generated on-core by the reader, no DRAM traffic |

Stall-shadow check:

| Waiting stage | Waits on | Work that can run in the window | Decision |
|---|---|---|---|
| compute `row_sumsq_block(b)` | reader filling `x` block b | reader loads b+1 while compute does b (`in_depth = 2`) | built (depth knob) |
| compute `normalize_block(b)` | its own `rinv` (self-produced, no external wait) | nothing external | none needed |
| R2 (deferred) `normalize_block(b)` | group combine of `rinv(b)` | `row_sumsq_block(b+1)` (rows independent; no floating-point reorder) | designed into R2: compute order `sumsq(b+1)` before `normalize(b)` |

### Buffer-depth knobs

| CB | Depth knob | Phase 0 value | What the depth buys |
|----|------------|---------------|---------------------|
| `cb_input_tiles` (TILE) | `in_depth` | 2 if `ceil(core_row_tiles_max / R) >= 2` else 1 (dropped to 1 only by the L1 selection function) | reader loads block b+1 while compute works on block b |
| `cb_gamma_tiles` | `gamma_depth` | 2 | writer builds gamma chunk c+1 while compute multiplies chunk c |
| `cb_output_tiles` | `out_depth` | 2 (x C tiles) | compute packs chunk tiles while writer drains |
| `cb_input_rm` (RM refinement) | `rm_in_depth` | 2 (x C tiles) | reader reads the next W-chunk of sticks during tilize |
| `cb_output_rm` (RM refinement) | `rm_out_depth` | 2 (x C tiles) | writer drains sticks while compute untilizes the next chunk |

### Mechanism caps

| Mechanism | Cap on which extent | Clamp | What happens unclamped |
|-----------|--------------------|-------|------------------------|
| L1 residency (R1 footprint, ledger closed form) | R, C, `in_depth` | `select_blocking()` bounded search; predicate failure -> regime R2 (refuse in Phase 0) | CB allocation exceeds L1 / clashes with L1 buffers (program creation error) |
| Core assignment | R <= `core_row_tiles_max`; `in_depth` <= blocks per core | min() in `select_blocking()` | wasted L1 |
| Chain DEST accumulation requires `Dst::D0` (`chain.inl:1273-1275`) and `PerOuter` pack lifecycle (`chain.inl:1141-1147`) | none on extent; one DEST tile per row | n/a | compile error |
| Cross-row `TileAddressing::Strided` requires caller-managed `(None, None)` input policies (`chain.hpp:283-286`) | `cb_input_tiles` wait/pop are caller-owned, nominal `R * Wt` | caller waits once per block, pops once per block | static_assert |
| CB ring closure (`ttnn-cb-memory-fundamentals.md`, CB Ring-Wrap Invariant) | quanta of `cb_input_tiles`, `cb_gamma_tiles` | push/pop the **nominal** quantum (`R*Wt`, `C`) on every block/chunk; ragged tails narrow only the NoC transfer / compute | pointer walks past `fifo_limit` -> silent neighbour-L1 corruption |
| RM only: `tilize` / `untilize` own their per-row quanta (`block_width_tiles` is a template param, `tilize_helpers.hpp:187-197`, `untilize_helpers.hpp:145-154`) | C | RM: C restricted to divisors of `tensor_w_tiles` | ragged helper quanta break ring closure |
| `prepare_partial_reduce_scalers<..., partial_positions>` (`reduce_helpers_dataflow.hpp:131-132`) | none on extent; `partial_positions = W % 32` in [1, 31] | only instantiated when `W % 32 != 0` | static_assert |
| ReduceTile partial scaler applies to the last reduce-dim tile **of the call** (`reduce_helpers_compute.hpp:333-337, :390-393`) | reduce shape along `w` | masked last column reduced in its own call (`ReduceInputBlockShape::of(R, 1)`, `with_partial()`) | valid lanes of earlier tiles erased |

DEST is not a cap here: the sum of squares uses one DEST tile per row (D0); the chains clamp their own lane batching to `DEST_AUTO_LIMIT` (`dest_helpers.hpp:89-103`), which is 4 tiles under `fp32_dest_acc_en=True` half-sync.

### Regimes

Named boundary: DRAM. Minimum = `x` read once, `out` written once, `gamma` read once.

| Regime | Status | Predicate | Block | Data movement vs. minimum | What a bigger block buys |
|--------|--------|-----------|-------|---------------------------|--------------------------|
| R1 `row_resident` | built | `footprint(R=1, C=1, in_depth=1) <= L1_CB_BUDGET` (ledger closed form) | `block_row_tiles x tensor_w_tiles`, normalize walks `w` in `w_chunk_tiles` | `x` 1x, `out` 1x (minimum). `gamma` read once per **active core** (`min(tensor_row_tiles, grid)` times) = at most `1/32 * (gs/xs)` of `x` bytes, because each active core owns >= 1 tile-row of 32 rows | Per block: 1 sumsq chain init (2 if W partial), 1 reduce init (2), 1 caller wait+pop on `cb_input_tiles`, 1 reader barrier, pipeline fill/drain of 3 helper phases. Per chunk: 2 mul-chain inits (1 without gamma), 1 gamma chunk build. Bigger R amortizes the per-block terms across R rows (matters for narrow W, e.g. Wt=2); bigger C amortizes per-chunk inits across `R*C` tiles. |
| R2 `w_split_combine` | deferred: every Phase 0 SUPPORTED shape satisfies R1's predicate and R1 is already at the DRAM minimum, so R2 saves no DRAM bytes on that domain; R2 is required only where R1's predicate fails (W beyond ~19K bf16 / ~9.6K fp32, all such golden cases pin `gamma_layout=TILE`, outside Phase 0) and as the physical realization of WIDTH/BLOCK_SHARDED. Reachable: R1's per-core pipeline is R2's per-core pipeline with a combine stage inserted between `row_sumsq_block` and `row_rinv_block`; the `w` extent is already a named CT knob (`core_w_tiles`) | `not R1_fits or (tensor_row_tiles < grid_cores and tensor_w_tiles >= 2 * W_SPLIT_MIN_SLICE_TILES)`; group size `G = min(floor(grid_cores / tensor_row_tiles), ceil(tensor_w_tiles / W_SPLIT_MIN_SLICE_TILES))` | `block_row_tiles x core_w_tiles`, `core_w_tiles = ceil(tensor_w_tiles / G)` (last slice ragged; the slice holding the partial tile runs the mask path) | DRAM: `x` 1x, `out` 1x, `gamma` 1x per group slice (minimum). Cross-core per block: `G-1` partial tiles (R col-0 tiles each) gathered, `R` rinv tiles multicast to `G-1` receivers. Topology: each core reduces its slice to col-0 partials (`reduce<SUM, ROW>`, scaler 1); combine of row r on core `r mod G` (reduce-scatter style; root-only when R == 1) via `reduce<SUM, ROW, AccumulateViaAdd, ..., ReduceWithinTile::Skip>` + the same finalize post-op; result multicast with `ttnn.Mcast` / `McastArgs`. Added CBs: `cb_partial_handoff` (compute -> sender kernel), `cb_partial_gather` (G x R tiles, written by peers), `cb_rinv_recv` (mcast destination). | Adds per block: one gather rendezvous + one multicast; bigger R amortizes both across rows. Grid-synchronization lamp applies (G vs payload). Stall shadow: `row_sumsq_block(b+1)` scheduled before `normalize_block(b)`. |
| R3 `row_two_pass_stream` | rejected: superseded by R2 everywhere R1 does not fit (R2 keeps each `w` slice resident, so `x` crosses DRAM once instead of twice). Dead end: no good scheme passes through re-reading `x` | `not R1_fits` (never selected) | `block_row_tiles x w_chunk_tiles`, two passes over `w` | `x` 2x | - |
| R4 `row_resident_height_sharded` | deferred: placement refinement of R1 (`memory_layout=HEIGHT_SHARDED` not in Phase 0 SUPPORTED); same loop nest | input HEIGHT_SHARDED | whole shard: `R = shard rows / 32`, `in_depth = 1`, `cb_input_tiles` backed on the shard (`ttnn.cb_descriptor_from_sharded_tensor`), `cb_output_tiles` backed on the output shard with normalize walking rows outer / chunks inner so pack order is shard order | `x` 0x DRAM (resident), `out` 0x DRAM; `gamma` once per core | one block per core: no per-block repetition at all |
| R5 `gamma_broadcast` | deferred: gamma re-read is bounded at `<= 1/32 * (gs/xs)` of input bytes for every shape (each active core owns >= 32 rows), so multicasting it can save at most ~3% (6% for fp32 gamma on bf16 input) of DRAM reads | R1 or R2 active | n/a (operand transport) | removes `(active_cores - 1) * W * gs` DRAM bytes; adds one multicast of `W * gs` | n/a |

Regime selection function (host, exact): compute `select_blocking()` (Work Distribution); if it returns a blocking, R1; else raise `NotImplementedError("rms_norm: row-resident footprint exceeds L1; regime R2 w_split_combine not built")`. Phase 0 has a single built regime, so no regime-pinned test is required yet; when R2 is built, the acceptance suite gains one test pinned to each predicate branch (a wide-W case and a few-rows case) on the grid it runs on.

### Traffic ranking

Units: bytes crossing each tier for the whole op (`X` = input bytes in tile-padded form, `O` = output bytes, `Gm` = `W * gs`, `A` = active cores).

| Rank | Candidate split | DRAM | Cross-core | Notes |
|---|---|---|---|---|
| 1 | rows across cores, whole row resident (R1) | `X + O + A*Gm` (`A*Gm <= X/32 * gs/xs`) | 0 | minimum; chosen |
| 1 (tie on DRAM) | rows x `w` slices, combine (R2) | `X + O + Gm*rows_groups` | `~2 * R * tile` per group per block | equal DRAM; adds a rendezvous. Wins only on occupancy (excluded from this ranking) or when R1 does not fit. Deferred row above. |
| 3 | rows across cores, `w` streamed twice (R3) | `2X + O + A*Gm` | 0 | rejected |

Chosen: R1. It is also the cheapest-traffic split.

### Block schedule

```cpp
for (uint32_t block_idx = 0; block_idx < num_blocks_this_core; ++block_idx) {
    load_block(block_idx);           // reader: R x Wt tiles of x -> cb_input_tiles (RM: + tilize_block)
    row_sumsq_block(block_idx);      // compute: S[r] = sum_t x[r,t]*x[r,t] in DEST, per row
    row_rinv_block(block_idx);       // compute: rinv[r] = rsqrt(rowsum(S[r]) * inv_w + eps)
    normalize_block(block_idx);      // compute: per W-chunk: y = x * rinv (col bcast) [* gamma (row bcast)]
    store_block(block_idx);          // writer: builds gamma chunk tiles ahead, drains output tiles
}
```

| Operation | Block shape | Resident across it | Intended fixed-cost frequency |
|---|---|---|---|
| `load_block` | `R_b x Wt` tiles (`R_b` = runtime rows of this block, `<= R`) | previous block in the other depth slot | reader: `R_b * Wt` async tile reads, one barrier, one `push_back(R * Wt)` (nominal) per block. Scaler tile(s) generated once per kernel. |
| `tilize_block` (RM refinement) | `R_b` tile-rows x `Wt`, in C-wide chunks | `cb_input_tiles` filling | one tilize init per block (`InitOnly`/`Neither`/`UninitOnly` across chunks) |
| `row_sumsq_block` | `R_b x (Wt - P)` (+ `R_b x 1` masked column when `P = W%32 != 0`) | `cb_input_tiles` (caller-waited, not popped) | one chain (two with P) per block; DEST accumulation per row; one pack per row (per column group) |
| `row_rinv_block` | `R_b x 1` reduce input tiles (x2 calls with P) | `cb_input_tiles` | one reduce call (two with P) per block; finalize post-op (scalar mul, scalar add, rsqrt) once per row on one DEST tile |
| `normalize_block` | `R_b x Wt`, walked as `ceil(Wt / C)` chunks of `R_b x C_c` | `cb_input_tiles` and `cb_rinv` (both released at block end) | per chunk: mul-by-rinv chain + mul-by-gamma chain (1 chain without gamma); caller pops `cb_input_tiles` (`R * Wt`) and `cb_rinv` (`R_b`) once per block |
| `store_block` | `R_b x C_c` tiles per chunk | gamma stick (whole kernel) | gamma stick DRAM read once per kernel; gamma chunk build (2 row-0 segment copies per tile) per chunk; one tile write per output tile, `noc_async_writes_flushed` before each pop, one write barrier per block |

`compute_kernel_hw_startup` once per kernel, before the first helper. No unit-at-a-time loop appears in the schedule; the chunk walk inside `normalize_block` is the block operation's internal traversal with inits amortized per chunk (`R_b * C_c` tiles per chain call).

### Perf lamps

| Lamp | Why the default may be wrong here | Nearby alternative to measure |
|------|-----------------------------------|-------------------------------|
| Overlap (block rows) | Default `R = ceil(core_rows / 2)` gives every multi-row core >= 2 blocks for reader/compute overlap; a whole-assignment block removes one block of fixed cost but serializes load and compute; `R = 1` maximizes overlap | `R in {1, ceil(core_rows/2), core_rows}` on the prefill shapes (`8192 x 1024 .. 7168`) |
| Normalize chunk | `C` coarsest that fits; a large C makes `cb_x_normed` (Float32) the second-largest buffer and may starve `in_depth` on wide W | `C in {8, 16, Wt}` on `W in {4096, 7168}` |
| Gamma persistence | Default rebuilds gamma chunk tiles per block (bounded L1); when `Wt * gt` fits, building them once per kernel removes per-block copies | persistent `cb_gamma_tiles` of `Wt` tiles when it fits |
| fp32 alias of `cb_x_normed` with `cb_output_tiles` | for fp32 input both are Float32 pages, so an alias group is legal and frees `R*C*ft` for a larger C | alias group (fp32 only) vs separate buffers |
| Bank stagger (`examples/bank_stagger`) | contiguous row ranges with `Wt` a multiple of the DRAM bank count start every core on the same bank | start each core's block walk at `core % n_banks` tiles and wrap |
| SFPU tile scope (`examples/sfpu_tile_scope`) | the finalize post-op (mul, add, rsqrt) runs on a full tile whose useful data is column 0 | `VectorMode::C` scoping for the post-op |
| Grid synchronization | not applicable to R1 (no collective); applies to R2's G (recorded in its row) | - |

Catalog entries that set knobs:

| Knob / decision | Entry | What it contributed |
|---|---|---|
| `split_work_to_cores(..., row_wise=True)` | `noc_placement` | row/diagonal core lines beat column lines on interleaved DRAM |
| `in_depth = 2`, batched reads with one barrier per block | `double_buffer` | batch + double buffer compound; block of async reads then one barrier |
| `W_CHUNK_FLOOR = 8`, coarse R | `compute_block_size`, `double_buffer` | per-call fixed cost amortization; 4-8 tile sweet spot |
| sum of squares accumulated in DEST, then one reduce per row | `row_reduce_accumulate`, `eltwise_l1_vs_dest_accumulate` | accumulate first, collapse once; DEST accumulation is the upper bound and avoids the x^2 buffer |
| two FPU passes for normalize instead of DEST-reuse fusion | `compute_fusion` | FPU consumer: L1 round-trip is 1.22x faster than dest-reuse |
| R2 design (W split, root + multicast for tiny payload, reduce-scatter for R >= several) | `width_split`, `tensix_all_reduce`, `mcast_topology` | width split for few-row shapes; combine topology choice |

## Dataflow Strategy

| Stage | Format | Mechanism | Notes |
|-------|--------|-----------|-------|
| x DRAM -> L1 | input dtype tiles | reader, `TensorAccessor` over interleaved pages, tile ids `(row_start + r) * Wt + t`, one linear run per block | nominal `push_back(R * Wt)`; ragged last block reads only `R_b * Wt` tiles |
| reduce scaler | bfloat16 tile(s) | reader generates on-core once: `prepare_reduce_scaler<cb_scaler, SUM, REDUCE_ROW>(1.0f)` or `prepare_partial_reduce_scalers<cb_scaler, SUM, REDUCE_ROW, W%32>(1.0f)` | scaler is exactly 1.0; `1/W` applied in fp32 in the finalize post-op |
| gamma DRAM -> L1 | gamma dtype, RM stick | writer reads page 0 of gamma (the whole stick) once into `cb_gamma_stick` (scratch, `align_up(W,32)` elements) | writer, not reader: balances the two data-movement RISCs (reader carries all x traffic) |
| gamma stick -> gamma row-0 tiles | gamma dtype tiles, only row 0 meaningful | writer per chunk: for each tile t, copy elements `[32t, 32t+16)` to face-0 row 0 (byte offset 0) and `[32t+16, 32t+32)` to face-1 row 0 (byte offset `256 * gs`); direct L1 word copies (or `local_copy_helpers_dataflow.hpp` self-aimed reads) | rows 1..31 never read (row broadcast); lanes past W hold stick padding and only reach output padding |
| compute | Float32 intermediates | sumsq chain -> reduce -> normalize chains | page formats per ledger |
| L1 -> DRAM | output dtype tiles | writer, per tile write, tile id `(row_start + r) * Wt + c*C + j` in the chain walk order (chunk-major, row, column) | `noc_async_writes_flushed()` before each `pop_front(1)`; one write barrier per block |
| RM refinement in | RM sticks | reader `read_sticks_for_tilize<cb_input_rm>` per C-wide chunk (`byte_offset_within_page = c*C*32*es`), compute `tilize<C>` into `cb_input_tiles`; reader zero-fills the whole `cb_input_rm` region once at kernel start so partial-W padding lanes are finite | `tensor_row_tiles` over 32-stick groups of the flattened rows |
| RM refinement out | RM sticks | compute `untilize<C, cb_output_tiles, cb_output_rm>` per chunk, writer `write_sticks_after_untilize` with the chunk byte offset; only valid sticks / bytes written | C divides Wt on this path |

Placement mapping (TARGET `memory_layout`, classified by the character of the axis each placement cuts):

| Value | Axis cut | Character | Cost | Realization |
|---|---|---|---|---|
| `INTERLEAVED` | none (op chooses rows) | - | built | R1 |
| `HEIGHT_SHARDED` | `row` | independent | knob-turn | R4: `cb_input_tiles` / `cb_output_tiles` backed on the shards (zero-copy), never re-read over the NoC through a TensorAccessor; block = whole shard |
| `WIDTH_SHARDED` | `w` | dependent | scheme-change | R2 physical: G = shard grid width, slices = shards, combine across the core row |
| `BLOCK_SHARDED` | `row` and `w` | dependent on `w` | scheme-change | R2 physical: one group per grid row |

Tensix-to-Tensix contract reserved for R2: partial tiles are pushed by each group member's sender kernel into the combiner's `cb_partial_gather` slot `member_idx * R` (uniform `R` per member, pad slots unused), signalled by a semaphore increment; the combiner multicasts `R` rinv tiles into `cb_rinv_recv` on all members with one `McastArgs::sender().send()` and a data-ready semaphore. `cb_partial_gather` is allocated identically on every member (push model) and is written only by the NoC, consumed only by compute.

## Work Distribution

| Field | Value |
|-------|-------|
| Work unit | block of `R x Wt` tiles (R tile-rows of 32 rows, full width) |
| Grid | `device.compute_with_storage_grid_size()` (runtime value, never inlined) |
| Per-core work | `split_work_to_cores(grid, tensor_row_tiles, row_wise=True)` -> group 1 cores own `core_row_tiles_max` rows, group 2 cores own one fewer; each core a contiguous range starting at `row_start` |
| Remainder | `num_blocks_this_core = ceil(core_row_tiles / R)`, `last_block_rows = core_row_tiles - (num_blocks_this_core - 1) * R`; TILE `tensor_row_tiles = prod(shape[:-2]) * ceil(H/32)` (per-image padding), RM `ceil(prod(shape[:-1]) / 32)`; `tensor_w_tiles = ceil(W/32)` |

`select_blocking()` (host; the single source of every block knob):

```
W, Wt = shape[-1], ceil(shape[-1] / 32);  P = 1 if W % 32 else 0
core_row_tiles_max  = split_work_to_cores(...).rows_per_core_group_1
budget              = ttnn.get_max_worker_l1_unreserved_size() - L1_HEADROOM_BYTES     # host constants
legal_C(c)          = 1 <= c <= Wt            (TILE)   |  c divides Wt (RM)
depth(R)            = 2 if ceil(core_row_tiles_max / R) >= 2 else 1     (TILE cb_input_tiles); RM: 1
C_floor             = largest legal c <= min(Wt, W_CHUNK_FLOOR)

R = ceil(core_row_tiles_max / MIN_BLOCKS_PER_CORE)
while R > 1 and footprint(R, C_floor, depth(R)) > budget: R -= 1
if footprint(R, C_floor, depth(R)) <= budget:
    C = largest legal c in [C_floor, Wt] with footprint(R, c, depth(R)) <= budget;  in_depth = depth(R)
else:  # R == 1
    C = largest legal c < C_floor with footprint(1, c, depth(1)) <= budget, in_depth = depth(1)
    if none: in_depth = 1; C = largest legal c with footprint(1, c, 1) <= budget
    if none: R1 predicate false -> NotImplementedError (R2 not built)
```

`footprint()` is the ledger's closed-form total. Host constants: `MIN_BLOCKS_PER_CORE = 2`, `W_CHUNK_FLOOR = 8`, `L1_HEADROOM_BYTES = 64 * 1024`. Every CB page count, loop trip count and kernel argument is derived from `(R, C, in_depth)` and the tensor geometry; none is restated as a literal.

## Circular Buffers

Page sizes: `xt` = input tile bytes, `ot` = output tile bytes (= xt), `gt` = gamma tile bytes, `gs` = gamma element bytes, `ft` = 4096 (Float32 tile), `st` = 2048 (bfloat16 tile). `P = (W % 32 != 0)`, `G = has_gamma`.

| Semantic Name | Index | Page Size | Num Pages | Sizing rationale | Format | Producer | Consumer | Lifetime |
|---------------|-------|-----------|-----------|------------------|--------|----------|----------|-----------|
| `cb_input_tiles` | 0 | xt | `in_depth * R * Wt` | spans row (R) and w (Wt): x is read twice (sumsq, normalize) | input dtype | reader (TILE) / compute tilize (RM) | compute | per block: load -> end of normalize |
| `cb_scaler` | 1 | st | `1 + P` | constant: [full, partial] scaler | bfloat16 | reader | compute | whole kernel |
| `cb_gamma_stick` | 2 | `align_up(W, 32) * gs` | 1 (G only) | one RM gamma row; streams nothing | gamma dtype | writer | writer (scratch, no FIFO ops) | whole kernel |
| `cb_gamma_tiles` | 3 | gt | `gamma_depth * C` (G only) | streams over row, spans a w chunk | gamma dtype | writer | compute | per chunk |
| `cb_sumsq` | 4 | ft | `(1 + P) * R` | spans row (one tile per row, two with P) | Float32 | compute | compute | sumsq -> rinv within a block |
| `cb_rinv` | 5 | ft | `R` | spans row | Float32 | compute | compute | rinv -> end of normalize |
| `cb_x_normed` | 6 | ft | `R * C` (G only) | spans row and one w chunk (sequential helpers: N1 must finish before N2) | Float32 | compute | compute | per chunk |
| `cb_output_tiles` | 7 | ot | `out_depth * C` | streaming window over row and w | output dtype | compute | writer (TILE) / compute untilize (RM) | per chunk |
| `cb_input_rm` (RM) | 8 | xt | `rm_in_depth * C` | 32 sticks x one C-wide chunk | input dtype | reader | compute | per chunk |
| `cb_output_rm` (RM) | 9 | ot | `rm_out_depth * C` | 32 sticks x one C-wide chunk | output dtype | compute | writer | per chunk |

CB sync (push count = wait count): `cb_input_tiles` push/wait/pop `R*Wt` per block (nominal); `cb_scaler` push `1+P` once, pop once at kernel end; `cb_gamma_tiles` push/wait/pop `C` per chunk (nominal; ragged last chunk fills `C_c` tiles); `cb_sumsq` push/pop 1 per row per column group; `cb_rinv` push 1 per row per reduce call, pop 1 per row by the accumulate reload (P only) and `R_b` once per block after normalize; `cb_x_normed`, `cb_output_tiles`, `cb_input_rm`, `cb_output_rm` 1 per tile (RM tilize/untilize quanta `C` per tile-row, C | Wt). Ragged last block / chunk is the final lap of its CB on that core, or is pushed at nominal count.

## Block Operation Realization

| # | Block operation | Block shape | Helper? | Input CB (semantic name, pages, state) | Output CB (semantic name, pages) | CB state after |
|---|-----------------|-------------|---------|----------------------------------------|----------------------------------|----------------|
| 1 | `load_block` | `R_b x Wt` | dataflow (TensorAccessor) | DRAM x | `cb_input_tiles`, `R*Wt` pushed | block resident |
| 1r | `tilize_block` (RM) | `R_b` rows x `Wt`, C-wide | `tilize<C>` | `cb_input_rm`, C per tile-row | `cb_input_tiles`, `R_b*Wt` | block resident |
| 2 | `row_sumsq_block` (aligned columns) | grid `(R_b, Wt - P)` | `eltwise_chain` BinaryFpu Mul x*x, DestAccumulation::PerRow | `cb_input_tiles`, caller-waited `R*Wt`, Strided `{0, Wt}`, not popped | `cb_sumsq`, `R_b` | x still resident |
| 2b | `row_sumsq_block` (masked column, P only) | grid `(R_b, 1)` | same chain, Strided `{Wt-1, Wt}` | `cb_input_tiles` (same) | `cb_sumsq`, `R_b` | |
| 3 | `row_rinv_block` | `ReduceInputBlockShape::of(R_b, 1)` per call | `reduce<SUM, REDUCE_ROW>` ReduceTile, BulkWaitBulkPop | `cb_sumsq` (`R_b` per call), `cb_scaler` (not popped) | `cb_rinv`, `R_b` | P: call 1 `Accumulate::at(cb_rinv, 0)`, `none()`; call 2 `Accumulate::at(cb_rinv, 1)`, `with_partial()`, finalize post-op. Aligned: one call, NoAccumulation, finalize. `Wt == 1` with P: only 2b + one call with `with_partial()` + finalize |
| 4 | `normalize_block` N1 (per chunk c) | grid `(R_b, C_c)` | `eltwise_chain` BinaryFpu Mul, B = rinv `BroadcastDim::Col` | `cb_input_tiles` Strided `{c*C, Wt}` (None/None); `cb_rinv` Upfront wait, no pop, `InputTileMapping::Col` | G: `cb_x_normed` `R_b*C_c`; no G: `cb_output_tiles` per tile | |
| 4b | `normalize_block` N2 (per chunk c, G only) | grid `(R_b, C_c)` | `eltwise_chain` BinaryFpu Mul, B = gamma `BroadcastDim::Row` | `cb_x_normed` PerTile/PerTile; `cb_gamma_tiles` caller-waited C, `InputTileMapping::Row`, popped C after the call | `cb_output_tiles` per tile | |
| 4c | `untilize` (RM, per chunk) | `R_b` tile-rows x C | `untilize<C>` | `cb_output_tiles` | `cb_output_rm` | |
| 5 | end of block | - | caller | pop `cb_input_tiles` `R*Wt`, pop `cb_rinv` `R_b` | - | slots free for block b+2 |
| 6 | `store_block` | per chunk `R_b x C_c` | dataflow; RM `write_sticks_after_untilize` | `cb_output_tiles` / `cb_output_rm` | DRAM out; also builds `cb_gamma_tiles` C per chunk ahead of draining | |

Finalize post-op (on the DEST tile of each row, inside row 3's last call): scale by `inv_w` (`mul_unary_tile`), add `epsilon` (`add_unary_tile`), `rsqrt_tile`. Inits inside the lambda (`binop_with_scalar_tile_init`, `rsqrt_tile_init`) as the reduce header's post-op example does.

Writer order per block (deadlock-free with `gamma_depth = 2`): `build_gamma(c0), build_gamma(c1), drain(c0), build_gamma(c2), drain(c1), ...` (gamma builds skipped without G). Reader order: `x(b0), x(b1)` (when `in_depth = 2`), then `x(b+2)` as slots free.

## API Mapping

| Block operation | Type | Function | File:Line | Template Params / Args | Input CB | Output CB | Which params are block knobs |
|-----------------|------|----------|-----------|------------------------|----------|-----------|------------------------------|
| kernel boot | raw_api | `compute_kernel_hw_startup` | (contract) `chain.hpp:24-40`, `reduce_helpers_compute.hpp:31-35` | `(cb_input_tiles, cb_input_tiles, cb_sumsq)`; RM: first-phase CBs | - | - | - |
| `row_sumsq_block` | helper | `eltwise_chain` + `BinaryFpu<BinaryFpuOp::Mul, A, B, Dst::D0, DestAccumulation::PerRow>` + `PackTile<output(cb_sumsq, PerOuter, PerOuter, Enabled, Direct, DestAccumulation::PerRow)>` | `eltwise/api/chain.hpp:620`, `:568`, `:588`, `:308`; Strided `:286-291`; IterationShape `:146-173` | `IterationShape::grid(R_b, Wt - P)`; A = B = `input(cb_input_tiles, WaitPolicy::None, PopPolicy::None, InputTileMapping::Block, DataFormatReconfig::Enabled, TileAddressing::Strided)` with `StridedTileRange{0, Wt}` | `cb_input_tiles` | `cb_sumsq` | `R_b` (grid H), `Wt` |
| `row_rinv_block` | helper | `compute_kernel_lib::reduce<PoolType::SUM, ReduceDim::REDUCE_ROW, cb_sumsq, cb_scaler, cb_rinv, ReduceInputPolicy::BulkWaitBulkPop, INPUT_AND_OUTPUT, ReduceFp32Mode::Fast, ReduceAlgorithm::ReduceTile>` | `reduce_helpers_compute.hpp:621-642`; `ReduceInputBlockShape` `:295-305`; `ReducePartialScaler` `:345-361`; `Accumulate` `:404-451` | `ReduceInputBlockShape::of(R_b, 1)`, `contiguous()`, `NoAccumulation` / `Accumulate::at(cb_rinv, i)`, finalize lambda, `none()` / `with_partial()` | `cb_sumsq`, `cb_scaler` | `cb_rinv` | `R_b` |
| finalize post-op | raw_api (helper's post_reduce_op slot) | `binop_with_scalar_tile_init`, `mul_unary_tile`, `add_unary_tile`, `rsqrt_tile_init`, `rsqrt_tile` | `tt_metal/hw/inc/api/compute/eltwise_unary/binop_with_scalar.h:156`, `:60`, `:35`; `.../rsqrt.h:21`, `:44` | `(dst, inv_w_bits)`, `(dst, eps_bits)`, `(dst)` | DEST | DEST | - |
| `normalize_block` N1 | helper | `eltwise_chain` + `BinaryFpu<Mul, A, input(cb_rinv, BroadcastDim::Col, WaitPolicy::Upfront, PopPolicy::None, InputTileMapping::Col)>` + `PackTile` | `chain.hpp:620`, `:568`, `:588`, `:330-335`, `:257-262` | `IterationShape::grid(R_b, C_c)`; A Strided `{c*C, Wt}` | `cb_input_tiles`, `cb_rinv` | `cb_x_normed` / `cb_output_tiles` | `R_b`, `C` |
| `normalize_block` N2 | helper | `compute_kernel_lib::mul<input(cb_x_normed), input(cb_gamma_tiles, BroadcastDim::Row, WaitPolicy::None, PopPolicy::None, InputTileMapping::Row), output(cb_output_tiles)>` | `eltwise/api/convenience.hpp:63-70` | `IterationShape::grid(R_b, C_c)` | `cb_x_normed`, `cb_gamma_tiles` | `cb_output_tiles` | `R_b`, `C` |
| reduce scaler | helper | `dataflow_kernel_lib::prepare_reduce_scaler<cb_scaler, SUM, REDUCE_ROW>(1.0f)` / `prepare_partial_reduce_scalers<cb_scaler, SUM, REDUCE_ROW, W % 32>(1.0f)` | `reduce_helpers_dataflow.hpp:58-60`, `:131-132` | pool-type-aware overloads | - | `cb_scaler` | - |
| gamma chunk build | raw_api | L1 word copies (or `local_copy_helpers_dataflow.hpp:74-95` self-aimed reads) | `local_copy_helpers_dataflow.hpp:57-95` | 2 segments of `16 * gs` bytes per tile | `cb_gamma_stick` | `cb_gamma_tiles` | `C` |
| `tilize_block` (RM) | helper | `compute_kernel_lib::tilize<C, cb_input_rm, cb_input_tiles>` | `tilize_helpers.hpp:187-197` | `num_blocks = 1` per chunk, InitOnly/Neither/UninitOnly across the block | `cb_input_rm` | `cb_input_tiles` | `C` |
| RM read | helper | `dataflow_kernel_lib::read_sticks_for_tilize<cb_input_rm>` | `tilize_helpers_dataflow.hpp:87-93` | `row_bytes = valid chunk bytes`, `byte_offset_within_page = c*C*32*es` | DRAM | `cb_input_rm` | `C` |
| `untilize` (RM) | helper | `compute_kernel_lib::untilize<C, cb_output_tiles, cb_output_rm>` | `untilize_helpers.hpp:145-154` | `num_blocks = R_b` | `cb_output_tiles` | `cb_output_rm` | `C` |
| RM write | helper | `dataflow_kernel_lib::write_sticks_after_untilize<cb_output_rm>` | `tilize_helpers_dataflow.hpp:129-135` | chunk byte offset | `cb_output_rm` | DRAM | `C` |

Helpers considered and rejected:

| Raw entry | Candidate helper | Mismatch (file:line) | Reason |
|---|---|---|---|
| finalize post-op | `AddUnary` / `MulUnary` / `Rsqrt` chain structs | `eltwise/unary/scalar.inl:54-56`, `eltwise/unary/math.hpp:37-38` | they are chain elements, not callable inside `reduce`'s `post_reduce_op(dst)` slot (`reduce_helpers_compute.hpp:519-523`); the slot takes the raw `*_tile` calls, as its documented `recip_tile` example does (`:604-614`). A separate chain would pack `rinv` out and back for three SFPU ops on one tile. |
| gamma chunk build | `read_sticks_for_tilize` + `tilize` of a 1-row block | `tilize_helpers.hpp:187-197` | puts the gamma conversion on the compute TRISCs once per chunk per block; the copy is 2 x 16-element segments per tile and belongs on the dataflow RISC |
| row sum of squares | `square` + `reduce` with `AccumulateViaAdd` | `convenience.hpp:79-80`, `reduce_helpers_compute.hpp:160` | needs an `x^2` CB spanning the block (Float32 pages: 2x the input bytes for bf16); the DEST-accumulated mul chain folds the square and the cross-tile sum into one DEST tile per row (l1-footprint Rule 3.4) |
| normalize fusion | `DestReuseBinary` | `chain.hpp:575-582`, `chain.inl:1483-1510` | no broadcast operand (gamma needs row broadcast, rinv column broadcast); `compute_fusion` measures dest-reuse 0.82x vs the L1 round trip for an FPU consumer |

## Broadcast Verification

| Phase | Op | CB_A (semantic name) Valid Region | CB_B (semantic name) Valid Region | Broadcast Dim |
|-------|-----|-----------------------------------|-----------------------------------|---------------|
| `row_sumsq_block` | Mul (DEST acc) | `cb_input_tiles`: All | `cb_input_tiles`: All | None |
| `row_rinv_block` | reduce ROW | `cb_sumsq`: All (masked column: lanes `< W%32` via partial scaler) | `cb_scaler`: Row0 | (reduce) -> output Col0 |
| `normalize_block` N1 | Mul | `cb_input_tiles`: All | `cb_rinv`: Col0 (REDUCE_ROW output) | Col |
| `normalize_block` N2 | Mul | `cb_x_normed`: All | `cb_gamma_tiles`: Row0 | Row |

## Key Risks and Gotchas

| Risk | Why it bites here | Mitigation in this design |
|------|-------------------|---------------------------|
| Padding folded into the denominator | poison suite fills TILE padding with 1000.0; folding it in is a 6-27% error at narrow W | last W-tile column summed separately (2b) and reduced with the partial scaler; `inv_w = 1/W` (true W) applied in fp32 |
| Non-finite padding | partial scaler masks by multiplying by 0; `inf * 0 = NaN` | TILE: precondition (finite padding) stated; RM: reader zero-fills `cb_input_rm` once at start, later laps hold previous finite data |
| Partial scaler applied to a combined tile | ReduceTile masks the last tile of the call (`reduce_helpers_compute.hpp:333-337, :390-393`) | masked column is its own `reduce` call (`of(R_b, 1)`), never summed into S before masking |
| Ring overshoot on ragged quanta | `Wt` prime / not a multiple of C; last block ragged | nominal `R*Wt` and `C` quanta on `cb_input_tiles` / `cb_gamma_tiles`; per-tile quanta elsewhere; RM restricts C to divisors of Wt |
| Two consumers on one CB | in-place tricks on `cb_input_tiles` or `cb_output_tiles` | no in-place transforms; `cb_x_normed` is a separate compute-only CB; gamma stick is writer-only scratch |
| Reader/writer deadlock on gamma | gamma chunks and output drains share the writer | writer order `build(c0), build(c1), drain(c0), build(c2), ...` with `gamma_depth = 2` |
| Float32 intermediates under fp32 DEST | a 16-bit `cb_sumsq` / `cb_rinv` / `cb_x_normed` truncates at the phase boundary | all three are Float32 pages (ledger audit 2) |
| `Wt == 1` with partial W | chain over `Wt - 1 = 0` columns | skip 2; run 2b + one finalize reduce with `with_partial()` |
| H padding rows | TILE images pad H independently; padded rows compute garbage | rows are independent; garbage stays in output padding; tile-row count is per image `ceil(H/32)` |
| fp32 through the FPU | srcA/srcB carry TF32 | within float32 PCC 0.999; accumulation stays fp32 in DEST |
| L1-interleaved inputs reduce CB space | allocator places L1 buffers at the top of L1 | `L1_HEADROOM_BYTES` constant; golden suite uses DRAM |
| Rt = 1 shapes run on one core | R1 assigns whole rows | recorded as R2 (deferred) with its predicate; not hidden as a perf lamp |
