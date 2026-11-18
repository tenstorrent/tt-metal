# Operation Design: rms_norm

## Overview

| Field | Value |
|-------|-------|
| Classification | compute (fused: optional tilize → Σx² reduce → rsqrt → scale → optional gamma → optional untilize), single device program |
| Goal | RMS-normalize along the last dim for RM or TILE input, any (non-aligned) H/W, optional RM gamma, bf16/fp32, one native dispatch. |
| Math | `out[..., w] = x[..., w] * rsqrt(sum_w(x²)/W + eps) * gamma[w]`, W = true (unpadded) width |
| References | `eval/golden_tests/rms_norm/feature_spec.py`, `.claude/references/{blocking-model,l1-footprint-discipline,precision_convention}.md`, `ttnn/cpp/ttnn/kernel_lib/{reduce_helpers_compute.hpp, reduce_helpers_dataflow.hpp, tilize_helpers.hpp, untilize_helpers.hpp, eltwise/api/{chain,convenience}.hpp}` |

## Parameters

| Name | Type | Required | Valid Range | Default | CT/RT |
|------|------|----------|-------------|---------|-------|
| input_tensor | ttnn.Tensor | yes | rank ≥ 2, bf16/fp32, RM or TILE, interleaved (Phase 0) | – | buffer addr RT |
| gamma | Optional[Tensor] | no | RM, shape (1,1,1,W), bf16/fp32 (Phase 0) | None | addr RT; presence CT |
| epsilon | float | no | >0 | 1e-6 | RT (fp32 bit-cast) |
| compute_kernel_config | ComputeConfigDescriptor | no | `fp32_dest_acc_en=True` (Phase 0); fidelity/approx not gated | `default_compute_kernel_config()` (HiFi4, fp32_dest_acc_en=True, approx False) | CT (passed as-is to the compute kernel descriptor) |
| Block knobs `Rb`, `Wb`, `Sb`, `x_depth`, `gamma_resident` | derived on host by `pick_blocking()` | – | see Blocking Model | – | CT (shape/grid-dependent ones may be RT where they only set loop trip counts) |
| `num_tile_rows_this_core`, `first_tile_row`, grid | derived | – | – | – | RT |

Entry point: validates (rank<2 → ValueError; gamma last dim ≠ W → ValueError; fp32 input + fp32_dest_acc_en=False → RuntimeError/EXCLUSION), resolves `None` config via the single exported `default_compute_kernel_config()` (the golden axis tagger reads the same factory), builds ONE `ProgramDescriptor`, dispatches once via `ttnn.generic_op`. No host tilize/pad/slice. INPUT_TAGGERS (`tag_alignment` ∈ tile_aligned/w_non_aligned/h_non_aligned, `tag_rank`, `tag_gamma_dtype`/`tag_gamma_layout` returning `"none"` when gamma is None) live in the op file; SUPPORTED includes `"none"` for the gamma axes.

## Tensors

### Input

| Property | Requirement |
|----------|-------------|
| Shape | `(..., H, W)`, rank ≥ 2, any H, W ≥ 1 |
| Dtype | bfloat16, float32 (bf8_b: later refinement; INVALID/xfail) |
| Layout | ROW_MAJOR or TILE (both native) |
| Memory | DRAM interleaved (Phase 0); sharded = deferred regime R4/R5 |

Gamma: RM (1,1,1,W), bf16 or fp32 (any dtype independent of input). TILE gamma = refinement (reader reads tiles directly; row 0 is already the broadcast row).

### Output

| Property | Value |
|----------|-------|
| Shape | identical to input |
| Dtype | same as input |
| Layout | same as input |
| Memory | DRAM interleaved |

## Blocking Model

### Axes

Derived math: per-row reduction over W; rows independent; gamma varies only along W.
Index space: `images = prod(shape[:-2])`, `Ht_img = ceil(H/32)`, `Wt = ceil(W/32)`, `total_tile_rows = images * Ht_img` (per-image padding, ceil — NOT floor of flattened rows; RM uses the same per-image tile-row definition so both layouts share one scheme).

| Axis | Character (+ reason) | Extent knob | Phase 0 value | Knob source | Core-assignment | Later unlock |
|------|----------------------|-------------|---------------|-------------|-----------------|--------------|
| Tile-rows (batch×H, `total_tile_rows`) | independent — each row's mean/scale involves only that row | `Rb` (tile-rows per block). Core share = `split_work_to_cores(total_tile_rows)` | `Rb = clamp(G_CAP // min(Wb,G_CAP), 1, rows_core)` i.e. coarsest with `Rb*Sb ≤ G_CAP=16` tiles; so Rb>1 only when `Wb ≤ 16`. Rb=1 when Wb>16 is justified: a 16-tile segment already amortizes per-block init, and `Rb*Wb` must stay contiguous for window ops | host `pick_blocking()` (single source; CB sizes, trip counts derive from it) | all cores of the grid, contiguous tile-row ranges, `ceil`-balanced (group1 gets +1); grid = RT/host param | knob-turn (more cores / larger Rb) |
| W tiles (`Wt`, the reduced axis) | dependent — Σx² spans it; also reuse-shared for gamma (same gamma for every row) | `Wb` (resident chunk, tiles) and `Sb` (compute/DM granule, tiles) | `Wb = Wt` if resident predicate holds (R1) else largest chunk with 2-deep fit (R2); `Sb = min(Wb, G_CAP)`. Whole axis lives on ONE core in Phase 0 | `pick_blocking()` | NOT split across cores in Phase 0 (within-core sequential accumulate across Sb-segments via `Accumulate`) | scheme-change (R3 W-split + cross-core combine) |
| Elements in last tile (`W % 32`) | tail/mask axis of Wt | `valid_w_last = W - 32*(Wt-1)` | exact value | host RT | n/a – same core as owner of the last segment | – |
| Rows in last tile-row (`H % 32`) | independent (tail) | `valid_h_last = H - 32*(Ht_img-1)` | exact | host RT | with owning row | – |
| Gamma W axis | reuse-shared operand | `gamma_resident` (bool) | true iff `Wt*gamma_tile_bytes ≤ GAMMA_BUDGET(128KB)` | host | loaded once per core (resident) or per-segment each row block (streamed) | knob-turn; mcast-from-injector = deferred R6 |
| Phase-boundary data: `cb_stat` (Σx² → inv_rms, one tile per tile-row) | independent along rows (one per row, stays on the owning core) | block = `Rb` tiles | one block per row-block; computed entirely on the owning core (no designated-core stage; nothing idles) | – | follows rows | – |

### Buffer-depth knobs

| CB | Depth knob | Phase 0 value | What the depth buys |
|----|------------|---------------|---------------------|
| cb_x (tiles) | `x_depth` (blocks of `Rb*Wb`) | 2 if `2*Rb*Wb*tile ≤ X_BUDGET` else 1 (resident-fits-only-once) ; chunked regime always 2 | reader/tilize overlap with compute of previous block/chunk |
| cb_x_rm | `rm_depth` (tile-rows of sticks) | 2 | DRAM read overlaps tilize |
| cb_out_tiles | `out_depth` | 2 granules | compute overlaps writer / untilize |
| cb_out_rm | `out_depth` | 2 tile-rows | untilize overlaps writes |
| cb_gamma | – | resident: 1× Wt; streamed: 2 granules of Sb | – |
| cb_sq, cb_accum, cb_stat, cb_scaler | – | 1 granule / constants (compute-internal, single-owner, serial) | cannot pipeline (helpers own all TRISCs) |

### Mechanism caps

| Mechanism | Cap on which extent | Clamp | What happens unclamped |
|-----------|--------------------|-------|------------------------|
| `G_CAP=16` window granule | `Rb*Sb` | `Rb*Sb ≤ 16`; Rb=1 when Wb>16 | cb_sq/cb_out windows grow with op size → L1 OOM |
| tilize/untilize helper block width | `Sb` as `block_width_tiles` | untilize splits above `DEST_AUTO_LIMIT` internally (`untilize_helpers.inl:81,244`); implementer must check `tilize_helpers.inl` for the equivalent fp32-DEST limit and clamp `Sb_rm=min(Sb, cap)` | wrong tiles / assert under `--dev` |
| untilize output cannot be block-float (`untilize_helpers.inl:164`) | dtype | Phase 0 only bf16/fp32 | assert |
| NoC/DRAM alignment | RM stick-segment read offset `seg*Sb*32*es` | multiple of 64 B for es∈{2,4}, Sb·32·es ≥ 64 ✔; stick base address alignment for RM pages is per-page (`TensorAccessor`) | misaligned NoC read |
| fp32 accumulation | cb_sq, cb_accum, cb_stat, and cb_x for fp32 input need `UnpackToDestMode::UnpackToDestFp32` (host compute config `unpack_to_dest_mode` per CB) | set for all fp32-format CBs | silent bf16-precision truncation |
| Reduce scaler | exact 1.0 packed bf16; 1/W applied in fp32 on the SFPU (`MulUnary`) | – | bf16-rounded 1/W (≈0.4% error) |
| X_BUDGET = 512 KB, GAMMA_BUDGET = 128 KB | `Wb`, `gamma_resident` | worst-case total ≤ ~1.1 MB (see ledger) | OOM |

### Regimes

| Regime | Status | Predicate | Block | Data movement vs. minimum | What a bigger block buys |
|--------|--------|-----------|-------|---------------------------|--------------------------|
| R1 row-resident (row-parallel, whole W on one core) | built | `Wt*in_tile*x_depth*Rb ≤ X_BUDGET` with `x_depth∈{1,2}` (i.e. `Wb==Wt`) | `Rb × Wt` tiles resident across pass 1 (Σx²) and pass 2 (normalize) | minimum 1 DRAM read of x + 1 write of out (+ gamma once per core): **achieved exactly**. No cross-core traffic. | `Rb`: amortizes per-block inits (reduce/eltwise/tilize init, scaler wait) once per block instead of per row; `Wt`: single reduce accumulation chain, no re-read |
| R2 row-streamed two-pass (row-parallel, whole W per core, chunked) | built | not R1 (`Wb < Wt`; `Rb=1`; `Wb` = largest multiple of `Sb` with `2*Wb*in_tile ≤ X_BUDGET`) | pass 1: x segments → Σx² (accumulate); pass 2: re-read x segments → normalize → out | x crosses DRAM **2×** (minimum for a non-resident row; the *stepping stone* — R3 later brings it back to 1×). out 1×. | `Wb`: fewer, bigger transactions per row; amortizes accumulate reload |
| R3 W-split + cross-core combine | deferred | rows < grid OR W too wide for R1 (e.g. `1×32768`, `1×16384`, `64×12288`) | each core owns a W-slice resident (`Wb_slice=ceil(Wt/n_w)`), partial Σx² tile per row | x 1×, out 1×; adds per-row-block gather of `n_w` partial tiles (fp32 tile ≈ 4 KB each) + one reduction. Saves the R2 second x read (≈ `rows*W*es` bytes DRAM) and fills the grid. Positive reason deferred: Phase 0's R1+R2 are correct for every shape; combine needs `ttnn.Mcast`/semaphore plumbing (scheme-change). Reachability: pass-1 already produces a per-row partial via `Accumulate` over segments, and `cb_stat` is the combine slot; adding the gather replaces "local accumulate across segments" with "local + remote partials", loop nest unchanged; pass 2 simply runs on the slice. First refinement for under-filled grids. |  the combine amortized over `Rb` rows per round |
| R4 HEIGHT_SHARDED in place | deferred | input HEIGHT_SHARDED, full W per shard row | shard = whole block, `cb_x` backed by `cb_descriptor_from_sharded_tensor` | 0 DRAM reads of x; knob-turn over R1 (CB/placement change only: reader removed, `Rb` = shard rows) | one block, no per-block sync |
| R5 WIDTH/BLOCK_SHARDED in place | deferred | W cut across cores | shard = block; partial Σx² combine along core row | scheme-change = R3 with the slice pre-placed | – |
| R6 gamma mcast | deferred | many cores re-reading same gamma | injector reads gamma once, `ttnn.Mcast` to group | gamma DRAM reads: n_cores× (tiny, W·es each) → 1× | – |
| Re-read x for 3+ passes (e.g., separate square pass, mean pass, scale pass) | rejected | – | – | superseded by R1/R2 (2 passes max); dead end | – |

Regime selection function (host, `pick_blocking`): compute `Wt`, `rows_core_max = ceil(total_tile_rows / n_cores)`, `Rb`, then `resident = (Wt*in_tile*Rb*(1) ≤ X_BUDGET)`; `x_depth = 2 if 2*Rb*Wt*in_tile ≤ X_BUDGET else 1`. Tests must pin both regimes (force R2 via an env/test hook or a very wide fp32 shape such as `(1,1,32,8192)` fp32 and `(1,1,64,12288)` bf16, vs. `(1,1,32,64)` for R1).

### Traffic ranking

| Split | DRAM crossings (x / out / gamma) | Cross-core | Rank |
|-------|----------------------------------|-----------|------|
| Rows across cores, whole W resident (R1) | 1 / 1 / 1 per core | none | best when it fits and rows ≥ cores |
| W split across cores + combine (R3) | 1 / 1 / 1 per slice | n_w partial tiles per row block (KB-scale) | best when R1 does not fit or rows < cores; strictly cheaper than R2 on DRAM |
| Rows across cores, W streamed (R2) | **2** / 1 / ~1 | none | correct fallback; one extra full x read |
| Rows across cores, re-read x ≥3× | ≥3 / 1 / 1 | none | rejected |

Chosen for Phase 0: R1 where it fits, else R2. R3 is cheaper than R2 on bytes but needs a combine; it is built later because it is the same structure with the combine slot already in place (`cb_stat`).

### Block schedule

Logical schedule per core (reader / compute / writer are separate kernels):

```cpp
// once per core
load_gamma_resident();                  // if gamma_resident (stick → row-0-of-tile scatter, Wt tiles)
prepare_scaler(1.0);                    // once
for (blk = 0; blk < num_row_blocks_this_core; ++blk) {      // Rb tile-rows (tail block smaller)
    // pass 1: sum of squares
    for (seg = 0; seg < num_segments; ++seg) {              // Sb-tile segments over Wt
        load_segment(blk, seg);                // RM: Rb·32 stick-segments; TILE: Rb×Sb tiles; zero the W/H pad
        [tilize_segment]                       // RM only
        square_segment();                      // Rb×Sb → cb_sq
        reduce_row_accumulate_segment(seg);    // SUM, REDUCE_ROW, Accumulate iter=seg → cb_accum / cb_stat on last
    }
    inv_rms_block();                           // cb_stat ← rsqrt(stat*(1/W) + eps), in place, Rb tiles
    // pass 2: normalize (R1: from resident cb_x; R2: re-load segments)
    for (seg = 0; seg < num_segments; ++seg) {
        [reload_segment(blk, seg)]             // R2 only (+ [tilize]); [stream gamma seg] if !gamma_resident
        scale_segment();                       // out = x * inv_rms(col bcast) [* gamma (row bcast)]
        [untilize_segment]                     // RM only
        store_segment(blk, seg);               // writes only valid rows / valid bytes
    }
}
```
Frequencies (intended): scaler/gamma-resident setup once per core; reduce/eltwise/tilize/untilize inits once per **segment of Rb×Sb tiles** (not per tile); `inv_rms_block` once per row block; no multicast/barriers in Phase 0 (R3 adds one gather round per row block). Wide-W R1 (Wb>16, Rb=1) iterates `Sb`-tile segments over the resident row; each segment is one helper call over a contiguous `Sb`-tile span.

### Perf lamps

| Lamp | Why the default may be wrong here | Nearby alternative to measure |
|------|-----------------------------------|-------------------------------|
| overlap | `x_depth=1` for large resident rows prevents prefetch of next row block behind pass 2 | chunk to R2-style 2-deep, or shrink Rb |
| grid-synchronization / occupancy | `total_tile_rows < cores` (decode, 1×W) leaves cores idle | go to R3 (W-split) first |
| `G_CAP` | 16-tile granule may be too small/large for init amortization | 8 or 32 |
| fused scale chain | BinaryFpu(col-bcast)+DestReuse(row-bcast gamma) vs two passes via a `cb_xn` window | measure |
| RM per-image tile-rows | per-image padding wastes work for `(B,17,W)` | flatten RM rows across images |
| fidelity | HiFi4 default vs HiFi2 for the bf16 perf cases | `compute_kernel_config` passthrough |
| subblock/inv-rms | `MulUnary 1/W` + `AddUnary eps` + `Rsqrt` approx mode | accurate vs fast rsqrt |

## Dataflow Strategy

| Stage | Format | Mechanism | Notes |
|-------|--------|-----------|-------|
| Read x (TILE) | tiles, input dtype | reader, `TensorAccessor`, tile pages, per segment; after landing, reader zeroes cols ≥ `valid_w_last` of the last W tile (face-aware scatter) so Σx² ignores pad | pad may be poisoned; zeroing is in the reader (it is the CB producer — legal) |
| Read x (RM) | stick segments (`Sb*32*es` B pages, 32 per tile-row) | reader, `TensorAccessor` page = stick, read segment at byte offset; missing sticks (H tail) and missing bytes (W tail) zero-filled in L1 | non-finite stale L1 × 0 would be NaN, hence explicit zero-fill |
| Tilize (RM) | stick pages → tiles in `cb_x` | compute `tilize<Sb,...>` (Lossless for fp32) | cb_x producer = compute in RM config, reader in TILE config |
| Square → reduce | fp32 tiles | `square` → `cb_sq`; `reduce<SUM,REDUCE_ROW>` with scaler 1.0 and `Accumulate` across segments | reduce output tile has valid col 0 |
| inv_rms | fp32, col0 valid | eltwise chain `MulUnary(1/W)`, `AddUnary(eps)`, `Rsqrt` in place on `cb_stat` | exact 1/W, eps in fp32 |
| Scale | fp32 DEST → out dtype | eltwise chain: `BinaryFpu Mul` (x, cb_stat, `BroadcastDim::Col`) then `DestReuseBinary Mul` (gamma, `BroadcastDim::Row`), pack to `cb_out_tiles` | no gamma → single mul |
| Gamma | RM stick (1,1,1,W) → tile row 0 | reader scatters stick elements into zeroed tiles: tile t, row 0, face0 cols 0–15 at elem offset 0, face1 cols 16–31 at offset 256 | no compute tilize for gamma; zero other rows |
| Untilize (RM) | tiles → sticks | compute `untilize<Sb,...>` → `cb_out_rm` | |
| Write | out dtype | writer, `TensorAccessor`; TILE: tile pages for valid tile-rows; RM: only valid sticks (rows < H) and valid bytes (W) | H/W pad never written |
| R3 contract (deferred) | partial fp32 tile per row | each slice core sends its `cb_accum` tile to a combiner (or all-gather) via `McastArgs` sender/receiver + semaphore; receiver adds then writes `cb_stat` | `cb_stat` is the existing slot |
| Sharded (deferred R4/R5) | in place | `cb_descriptor_from_sharded_tensor` for cb_x / cb_out; no NoC re-read | HEIGHT = knob-turn over R1; WIDTH/BLOCK = scheme-change (R3) |

## Work Distribution

| Field | Value |
|-------|-------|
| Work unit | a tile-row (assignment) grouped into row blocks of `Rb` (compute) |
| Grid | full compute grid (`device.compute_with_storage_grid_size()`), passed as a parameter; `split_work_to_cores(grid, total_tile_rows)` |
| Per-core work | `rows_core ∈ {g1_rows, g2_rows}` contiguous tile-rows from `first_tile_row`; `num_row_blocks = ceil(rows_core/Rb)`, tail block `rows_core % Rb` |
| Remainder | tile-row → (image, row-in-image) via `ceil(H/32)` per image; last tile-row of each image has `valid_h_last` rows; last W tile has `valid_w_last` cols; RM tail sticks zero-filled, never written |

One regime per program (R1 or R2) selected by the host predicate above; regime-pinned tests required (R1: `(1,1,64,128)`, `(4,8,32,256)`; R2: `(1,1,32,8192)` fp32, `(2,1,64,4096)` fp32 and `(1,1,32,16384)` bf16 loose case). Cores with zero rows are not launched.

## Circular Buffers

Notation: `Tx`=input tile bytes, `T32`=4096 (fp32 tile), `Sb=min(Wb,16)`, `G=Rb*Sb`.

| Semantic Name | Index | Page Size | Num Pages | Sizing rationale | Format | Producer | Consumer | Lifetime |
|---------------|-------|-----------|-----------|------------------|--------|----------|----------|-----------|
| cb_x_rm (RM only) | 0 | `Sb*32*es` | `rm_depth*32` | spans 32 sticks×Sb; streams over segments/rows | input dtype | reader | compute (tilize) | per tile-row segment |
| cb_x | 1 | tile | `x_depth*Rb*Wb` | spans Rb, Wb (resident R1) / chunk (R2); R1 live until pass 2 done | input dtype | reader (TILE) / compute (RM) | compute | row block (R1) / segment (R2) |
| cb_scaler | 2 | tile | 1 | constant 1.0 | bfloat16 | reader | compute | whole program |
| cb_gamma | 3 | tile | resident: `Wt`; else `2*Sb` | spans Wt / streams segments | gamma dtype | reader | compute | whole program / per segment |
| cb_sq | 4 | tile | `G` | window of Rb×Sb squares | fp32 | compute | compute | per segment |
| cb_accum | 5 | tile | `Rb` | running Σx² across segments | fp32 | compute | compute | per row block |
| cb_stat | 6 | tile | `Rb` | Σx² result then inv_rms in place; persists through pass 2 | fp32 | compute | compute | per row block |
| cb_out_tiles | 7 | tile | `2*G` | double-buffered window | output dtype | compute | writer (TILE) / compute untilize (RM) | per segment |
| cb_out_rm (RM only) | 8 | `Sb*32*es` | `2*32` | one tile-row of sticks ×2 | output dtype | compute | writer | per segment |

(RM with Rb>1: tilize/untilize act on one tile-row at a time within the block; `cb_out_rm` holds 32 sticks per row.) Producer/consumer = compute→compute for compute-internal CBs; sequential, single owner.

## Block Operation Realization

| # | Block operation | Block shape | Helper? | Input CB | Output CB | CB state after |
|---|-----------------|-------------|---------|----------|-----------|----------------|
| 1 | load_gamma_resident | Wt tiles | raw (reader) | gamma DRAM | cb_gamma (Wt) | resident |
| 2 | prepare_scaler | 1 tile | helper | – | cb_scaler (1) | persistent |
| 3 | load_segment | Rb×Sb tiles / Rb·32 sticks | raw dataflow | DRAM | cb_x / cb_x_rm | – |
| 4 | tilize_segment (RM) | Sb wide × Rb tile-rows | helper | cb_x_rm | cb_x | – |
| 5 | square_segment | Rb×Sb | helper | cb_x (no pop in R1) | cb_sq (G) | cb_x stays (R1) |
| 6 | reduce_row_accumulate_segment | Rb×Sb → Rb | helper | cb_sq | cb_accum / cb_stat | – |
| 7 | inv_rms_block | Rb tiles | helper (chain) | cb_stat | cb_stat | persists to pass 2 |
| 8 | scale_segment | Rb×Sb | helper (chain) | cb_x, cb_stat (no pop), cb_gamma | cb_out_tiles | cb_x popped here (R1) |
| 9 | untilize_segment (RM) | Sb × Rb tile-rows | helper | cb_out_tiles | cb_out_rm | – |
| 10 | store_segment | Rb×Sb / sticks | raw dataflow | cb_out_tiles / cb_out_rm | DRAM | – |

## API Mapping

| Block operation | Type | Function | File:Line | Template Params / Args | Input CB | Output CB | Block knobs |
|-----------------|------|----------|-----------|------------------------|----------|-----------|-------------|
| prepare_scaler | helper | `calculate_and_prepare_reduce_scaler<cb_scaler, SUM, REDUCE_ROW>` (pool-type-aware; or `prepare_reduce_scaler(1.0f)`) | reduce_helpers_dataflow.hpp:109-127 | | – | cb_scaler | – |
| tilize_segment | helper | `compute_kernel_lib::tilize<Sb, cb_x_rm, cb_x, ..., Fp32Mode::Lossless for fp32>(in,out,num_blocks)` | tilize_helpers.hpp:194-208 | num_blocks = Rb | cb_x_rm | cb_x | `Sb`, num_blocks=`Rb` |
| square_segment | helper | `square<Input,Output>(IterationShape, in, out)` | eltwise/api/convenience.hpp:79-80 | `EltwiseShape` = Rb·Sb tiles; input WaitUpfront/NoPop in R1 (resident), pop in R2 | cb_x | cb_sq | `Rb*Sb` |
| reduce_row_accumulate_segment | helper | `reduce<SUM, REDUCE_ROW, cb_sq, cb_scaler, cb_stat_or_accum>(…, ReduceInputBlockShape::of(Rb,Sb), contiguous(), Accumulate(accum, seg))` | reduce_helpers_compute.hpp:654-673; Accumulate 427 | `ReduceAlgorithm::Auto`; reconfig INPUT_AND_OUTPUT | cb_sq | cb_accum/cb_stat | `Rb`, `Sb` |
| inv_rms_block | helper | `eltwise_chain(shape(Rb), CopyTile, MulUnary, AddUnary, Rsqrt, PackTile)` | eltwise/api/chain.hpp (op structs: unary/scalar.hpp:26-32, math.hpp:36-38) | scalars `1/W`, `eps` as fp32 bit-patterns (RT) | cb_stat | cb_stat | `Rb` |
| scale_segment | helper | `eltwise_chain` with `BinaryFpu<Mul, …Col bcast>` + `DestReuseBinary<Mul, input(cb_gamma, Row bcast)>` + `PackTile`; no gamma: `mul<…BroadcastDim::Col…>` (convenience.hpp:63-70) | chain.hpp:358-368 (BroadcastInputSpec), 579 (DestReuseBinary) | cb_stat WaitPolicy upfront / PopPolicy None until block end | cb_x, cb_stat, cb_gamma | cb_out_tiles | `Rb`, `Sb` |
| untilize_segment | helper | `untilize<Sb, cb_out_tiles, cb_out_rm>(in,out,num_blocks=Rb)` | untilize_helpers.hpp:152-163 | | cb_out_tiles | cb_out_rm | `Sb`, `Rb` |
| load/store/gamma scatter/pad zero | raw_api | `noc_async_read/write`, `TensorAccessor`, L1 pointer writes | – | | | | `Sb`,`Rb` |

Helpers considered and rejected for raw dataflow: tilize_helpers_dataflow.hpp / local_copy helpers cannot express (a) per-stick partial-width reads at a byte offset with zero-fill of both H and W tails, or (b) the gamma stick → row-0-of-tile face scatter; the implementer must check them first (file:line) and use them if they cover the shape. `compute_kernel_hw_startup()` is called once at compute-kernel start before any helper. If the fused DestReuse scale chain cannot express the Col-then-Row broadcast, record the gap and add a `cb_xn` (G pages, fp32, compute→compute) window — the only sanctioned extra CB.

## Broadcast Verification

| Phase | Op | CB_A Valid Region | CB_B Valid Region | Broadcast Dim |
|-------|-----|-------------------|-------------------|---------------|
| scale by inv_rms | Mul | cb_x: All | cb_stat: Col0 (REDUCE_ROW output) | Col |
| scale by gamma | Mul (DestReuse) | DEST (x·inv): All | cb_gamma: Row0 (stick scattered into row 0; other rows zero) | Row |

## Reduce Direction Verification

Single direction only (the reduced axis is always the last): omitted beyond: logical dim W → `ReduceDim::REDUCE_ROW`, output Col0, `BroadcastDim::Col`, `ReduceInputBlockShape::of(Rb, Sb)`, `EltwiseShape` Rb·Sb.

## Key Risks and Gotchas

| Risk | Why it bites here | Mitigation in this design |
|------|-------------------|---------------------------|
| Padding leaks into Σx² / wrong divisor | TILE padding may be poisoned (1000.0); RM tails are stale L1 | reader zeroes W-pad cols (and missing sticks) before push; divide by the TRUE W (`1/W` RT scalar), never `Wt*32` |
| 1/W rounded in bf16 scaler | ≈0.4% uniform scale error invisible to PCC | scaler = 1.0; apply 1/W in fp32 SFPU |
| fp32 CBs read through the default unpack path truncate | fp32 input / intermediates | `UnpackToDestFp32` for every fp32-format CB; tilize `Fp32Mode::Lossless` |
| Garbage H-pad rows → NaN/inf | discarded rows only, but must not be written | writer writes only `valid_h_last` rows; zero-fill H tail anyway |
| In-place stat CB | `cb_stat` is transformed in place by compute | compute-only owner, real reserve/push/wait/pop between reduce and chain (no `None/None` credit trick) |
| Accumulate semantics | partial-scaler/iteration index applies per reduce() call | `iteration = seg`, no partial scaler (pad already zero) |
| Regime reachable only by shape | R2 only on wide rows | regime-pinned tests (wide fp32 and 16384/32768 loose cases) |
| Under-filled grid for 1×W | rows < cores; Phase 0 uses only `total_tile_rows` cores | recorded as R3, first refinement; correctness unaffected |
| Gamma dtype ≠ input dtype | reconfig between srcA/srcB formats | chain `DataFormatReconfig` on; gamma CB format = gamma dtype |
| RM stick alignment | segment offset and page size | 64 B multiples (see caps) |
| bf8_b, `fp32_dest_acc_en=False`, sharded, TILE gamma | not Phase 0 | EXCLUSIONS / SUPPORTED omit them; refinement candidates |
| Structural impossibilities | none beyond INVALID in feature_spec.py | – |
