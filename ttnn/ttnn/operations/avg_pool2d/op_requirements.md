# Operation Requirements: avg_pool2d

## Definition
- **Formula**: global window only (`kernel_size == (H, W)`, zero padding): `out[n, c] = (1 / (H*W)) * Σ_{i<H, j<W} in[(n*H + i)*W + j, c]`; the input is `[1, 1, N*H*W, C]`, the output `[1, 1, N, C]` (interleaved, in the input's buffer type, rule R9).
- **PyTorch Reference**: `torch.nn.functional.avg_pool2d(images, kernel_size, stride, count_include_pad=True)` on the images as `[N, C, H, W]`, flattened back to `[1, 1, N*H_out*W_out, C]` (`reference()` in `eval/specs/avg_pool2d/spec.py`); for the global window `x.reshape(N, H*W, C).mean(1)`.
- **Import Path**: `from ttnn.operations.avg_pool2d import avg_pool2d`
- **Function Signature**: `avg_pool2d(input_tensor, batch_size, input_h, input_w, channels, kernel_size, stride, padding, *, compute_kernel_config=None, dtype=ttnn.bfloat16, output_layout=ttnn.ROW_MAJOR_LAYOUT) -> ttnn.Tensor`

## Phases

> **Non-regression rule**: Every refinement must pass all tests from prior phases.
> **Drift signal**: XPASS-strict failures mean the implementer added support but forgot to update SUPPORTED. The implementer fixes by updating SUPPORTED.
> **Checkbox protocol**: Implementer marks `[x]` when the refinement is complete and all tests pass, `[~]` when real work landed but at least one named axis value is deferred (treated as completed by the queue, surfaced as partial), `[ ]` only when nothing usable was produced.
> **Refinement ID + follow-up naming (mandatory — the runner parses this)**: Primary refinements are `Refinement N` (e.g. `Refinement 1`, `Refinement 2`). When you ship `[~]` partial and file the sharper follow-up the partial-tick protocol requires, name it by appending a lowercase letter to the parent's number: `Refinement 1b`, `Refinement 1c`, … (never `Refinement 1.5`, `Refinement 1 (follow-up)`, or a fresh number). Order follow-ups immediately after their parent so the queue runs them before later refinements — a partial's remaining-blocker follow-up must be picked next, not leapfrogged. The runner's parser matches exactly `Refinement \d+[a-z]?`; any other shape is invisible to the queue and silently skipped.
> **Backend**: Quasar on the ttsim simulator, which has no clock. A perf refinement here is gated on a **structural count** (backend page, *Performance measurement*); its gain in time stays unconfirmed until a timed backend or hardware measures it, and no simulator timing is reported as a speedup.

### [x] Phase 0 — Core Implementation

- **SUPPORTED dtype**: [bfloat16]; **output_dtype**: [bfloat16]
- **SUPPORTED layout**: [TILE]; **output_layout**: [TILE]
- **SUPPORTED placement**: memory_config ∈ {DRAM, L1}, sharding ∈ {interleaved, width} (width-sharded only in L1: INVALID)
- **SUPPORTED shape-derived axes**: window = global, batch = single, channel_alignment = tile_aligned
- **SUPPORTED compute config**: fp32_dest_acc_en ∈ {True, False}; any math_fidelity / math_approx_mode (R3); a `ttnn.ComputeConfigDescriptor` or a device config (R4); default `default_compute_kernel_config()` = HiFi4, 32-bit DEST, no approximation (R2)
- **Regime**: `col_split_stream`: channel tile columns over the full grid (`split_work_to_cores`, row-wise), one block = the cluster's whole assignment, positions reduced in DEST by one `reduce<SUM, REDUCE_COL>` per block, `1/(H*W)` in the scaler, R1 padding rows zeroed by the reader. Every placement is read through the input's `TensorAccessor`.
- **Cores**: up to the full grid (`min(ceil(C/32), 32)` clusters on this target); thread plan (R, C, W) = (1, 1, 1) (data-movement bound, op_design.md)
- **Golden baseline**: 53 / 53 supported cells pass (+ loose case, 4 numeric regression cases, 3 rejections); xfail_expected 496, invalid_skipped 272; supported_fail / xpass_drift / xfail_wrong_mode all 0 (`verifier_report.json`)

### [ ] Refinement 1 — Consume width-sharded inputs in place (complete `sharding: width` natively)

**Goal**: `SUPPORTED["sharding"]` already claims `width`, but Phase 0 serves it by gathering every tile through the `TensorAccessor` onto a full-grid channel split, so the shard-holding clusters re-read their own resident shard over the NoC like any interleaved tensor. Build the design's deferred regime `shard_in_place_width` and select it for every width-sharded input. Each cluster that holds a shard reduces that shard where it lies: the input ring is borrowed on the shard (`ttnn.experimental.dfb_spec_from_sharded_tensor` / `DataflowBufferSpec(borrowed_from=...)`), the per-cluster block is the whole resident shard (the sharded block default), and the host split is read off the shard spec, not off `split_work_to_cores`. Interleaved inputs keep `col_split_stream`. No SUPPORTED change.

**Verifier notes**:
- Why this is first: a claimed placement value must be implemented natively before anything is built on top of it. R2 (batched) and R4 (row-major) both have width-sharded cells and should extend this regime, not the gather. The ledger finding folds in here too. Phase 0's data-movement budget defers the zero-copy split on occupancy grounds alone ("2 of 32 clusters"), which is not a positive reason, and the sharded block default (one block = the resident shard) was departed from with no measurement. Update `l1_ledger.md` (borrowed ring row, budget, regime table) and `op_design.md` (regime selection) with the build.
- R1 on a borrowed ring: padding rows `[H*W, 32*Ht)` of the shard's last position tile row hold undefined content. The reader masks them, not the scaler (see op_design.md *As built*). Two options are known. (a) The reader moves the last position tile row of the shard into a small side ring with a local L1→L1 copy, then zero-fills it as Phase 0 does; the input tensor is never written. (b) Zero the padding rows in place in the shard. That writes the caller's tensor (tile padding only), so justify it if chosen. The rows `0 .. Ht-2` are consumed in place either way.
- Order trap: the REDUCE_COL helper streams (DEST chunk, position row, column of the chunk), while a shard sits in L1 row-major (position row, then all of the shard's columns). These agree only when the shard's channel tiles fit in one DEST chunk. Otherwise use a row stride (`ReduceInputMemoryLayout::with_row_stride`) with an indexed input policy, at C = 1 only, or a per-chunk walk. Refinement 3 raises C on this path, so record which one you chose.
- A ragged last shard (fewer real channel tiles than the shard width) must reduce only its real columns.
- The model's call (2 shards) then reduces on 2 clusters instead of 32. That occupancy trade cannot be timed on this backend: say so in the changelog. Refinement 3 restores parallelism inside each shard cluster.

**Done when**: on every width-sharded golden cell and on the loose case `resnet50_global_lofi`, the input crosses the NoC **0 bytes** (structural count, before: `Ht·C_t·2048 B`, e.g. 262,144 B on the model's call). A test reads it off the program, for example: the input ring is borrowed, and the width path's reader has no input `TensorBinding`. `test_padding_rows_do_not_enter_the_mean` gains a width-sharded case with poisoned padding (incl. `last_tile_valid_rows` < 16 and ≥ 16) and passes. The one-session test covers the new regime. The golden suite and every Phase 0 test stay green.

### [ ] Refinement 2 — Several images per call (`batch` = batched)

**Goal**: add `batched` to `SUPPORTED["batch"]`: N images flattened one after another, one output row per image, in order (R7). The output is `[1, 1, N, C]`, so image n's mean is row `n % 32` of output tile row `n // 32`. Images start and end inside tiles whenever `H*W % 32 ≠ 0` (2×7×7: the second image starts at row 49). Neither image's rows may enter the other's mean (R1, R7). Spread `(image block, channel tile)` work units over the grid (both axes are independent) on the interleaved regime, and extend Refinement 1's in-place regime for width-sharded batched inputs. Cells unlocked with today's other axes: the 4 tile-aligned batched entries (2×7×7×64, 4×4×8×32, 2×7×7×4096, 2×2×2×8192) × fp32_dest_acc_en × {DRAM, L1 interleaved, L1 width} = 24 cells.

**Implementation skill**: /interleaved-parallel

**Verifier notes**:
- This is a **scheme-change**, so it stands alone. The design's deferred regime `batched_averaging_matrix` (scaler ring → averaging-matrix ring, a matmul of matrix tiles with input tiles) needs a matmul primitive, which would be raw LLK. A helper-free alternative also exists and is the implementer's call. Reduce each image on its own with the reader zero-filling the rows of a shared tile that belong to the neighbouring image (prefix and suffix; a shared tile is read once per image, ≤ 2 crossings). Then place each image's row-0 result into row `n % 32` of the output tile on the data-movement side (an explicit-sync assembly ring). Masking stays in the reader whichever scheme is chosen.
- `/interleaved-parallel` covers the host split mechanics of the 2-D `(image block, channel tile)` work split only. The image/tile-straddle scheme is the work here and has no skill.
- Depends on Refinement 1 (width-sharded batched cells use its regime). Refinement 4 adds row-major output and must write N sticks.

**Done when**: the 24 cells above pass in the golden suite. A design test with a per-image ramp shows no bleed across an image boundary inside a tile (2×7×7, 3 images of 5×5), with the R1 poisoned-padding case on the last image. Every earlier test stays green. Batched cells whose other axes are still outside SUPPORTED (row-major, height, non-tile-aligned) stay xfail until Refinement 4.

### [ ] Refinement 3 — Put the cluster's four NEOs to work on the in-place width-sharded path

**Type**: perf

**Goal**: after Refinement 1 the model's call (loose case `resnet50_global_lofi`: 7×7×2048, width-sharded on 2 cores in L1, LoFi, 16-bit DEST) reduces each 1024-channel shard (32 channel tiles × 2 position rows) on **one** NEO of each shard cluster. No `LOOSE_CASES` entry is perf-flagged, so this region is free-selected: it is the model's own call, and the shape whose work per engaged cluster is largest. Raise the thread plan on that path so the shard's DEST chunks are dealt to the cluster's NEOs, using the target page's *Thread plan* rules (STRIDED rings, capacity per NEO, whole rounds, plan counts from one host constant), with `toy_thread_plan` / `toy_row_passes` in `.claude/references/quasar_op_examples/` as worked patterns. Relevant catalog levers (`ttnn/ttnn/operations/examples/master.md`): `compute_block_size` (one reduce per NEO block, not per tile-row), `double_buffer` (if the reader still feeds a side ring). No SUPPORTED change.

**Verifier notes**: depends on Refinement 1 (it tunes the path Refinement 1 builds). The reduce helper has not been run under a plan on Quasar. Test at C = 1 and at the plan's C, and keep the front-tile policy or stride the indices yourself. The design argued (1, 1, 1) for the gathered interleaved path, where the op is data-movement bound. In place there are no input bytes to move, so that argument does not carry over. The unit dealt is a DEST chunk of columns (larger than a ring entry): pad the last round as the target page says.

**Done when**: structural count, compute threads at work per engaged cluster on `resnet50_global_lofi`, goes from **1 to 4**. A test reads the plan off the program artifacts, and a per-NEO tag shows that every NEO reduced its own chunks; a correct result alone does not show it. The loose case and the golden suite stay green, and nothing regresses across the config-spanning guard set (one representative per kernel path × layout × placement: DRAM interleaved, L1 interleaved, L1 width-sharded; fp32_dest_acc_en True and False). The changelog states the count before and after and that the gain in time is unconfirmed on this backend.

### [ ] Refinement 4 — Row-major input and output, channels not a multiple of 32, height-sharded input

**Goal**: one reader/writer pass over the op's data-access boundary that adds the four remaining TARGET values:
- `SUPPORTED["layout"]` gains `ROW_MAJOR_LAYOUT`: the input as sticks of C channels, one per position. Tilize in-kernel (a tilize phase before the reduce, or a reader that assembles tile-shaped entries from stick segments); the math stays on tiles.
- `SUPPORTED["output_layout"]` gains `ROW_MAJOR_LAYOUT`: the result as N sticks of C channels. Row 0 of each reduced tile is the stick, either through an untilize phase or by the writer extracting the two face-rows of row 0.
- `SUPPORTED["channel_alignment"]` gains `non_tile_aligned`: C % 32 ≠ 0. In TILE layout the padding columns only ever reach padding output columns. In ROW_MAJOR the reader zero-pads the last channel tile, and stick sizes of `2·C` bytes meet the target page's L1 (16 B) / DRAM (64 B) alignment.
- `SUPPORTED["sharding"]` gains `height`: L1 height-sharded inputs (R10). A height shard cuts the reduced (positions) axis, so it is read through the accessor across the shard cores, as the specification's GUIDANCE says: a genuinely non-local gather, no cross-core combine. Its in-place form is the deferred `positions_split_combine` regime (see the verification report).

Cells: 272 xfail cells carry `layout = ROW_MAJOR`, 272 carry `output_layout = ROW_MAJOR`, 160 carry `non_tile_aligned` and 136 carry `height`. After Refinements 1 and 2 every remaining xfail cell of TARGET is in this refinement.

**Implementation skill**: /memory-layouts

**Verifier notes**:
- Native kernel support is the goal, and in this run it is the only path. The usual partial-tick escape hatch (a `ttnn.to_layout` / `ttnn.tilize` / `ttnn.untilize` wrapper at the entry point) is **not available**. The run requires exactly one native dispatch per call, and a wrapper is a second dispatch. If a layout leg does not fit one pass, ship the values that work natively as `[~]` and file the remaining in-kernel work as `Refinement 4b`.
- Row-major × width-sharded cells must use Refinement 1's in-place regime: tilize from the borrowed row-major shard. Row-major × batched cells inherit Refinement 2's image-boundary handling, in sticks rather than tile rows, which is simpler.
- Precision watch: the 32×32×50 entry has 1024 positions (Ht = 32), and the FPU reduce accumulates through a tf32 round trip. With a 16-bit DEST the error grows with the sum's length. If a cell lands in `numerical-precision`, leave it failing and name it in a follow-up. Do not exclude it.
- Ordering: after Refinements 1 and 2, so that the new layout legs extend the final regimes once (hardest first).

**Done when**: every golden cell of TARGET passes (or a cell that fails only with `numerical-precision` / `OOM` is named in `Refinement 4b`). The R1 poisoned-padding test covers a row-major input and a non-tile-aligned C. The one-session test covers a row-major output followed by `ttnn.to_layout`. Every earlier test stays green.

### [ ] Refinement 5 — Keep reads in flight on the interleaved reader

**Type**: perf

**Goal**: on `col_split_stream` (interleaved inputs) the reader issues one chunk row of tile reads, waits on one barrier, then pushes. So it never has more than `min(core_ct_tiles, dest_chunk_tiles)` reads outstanding (2 on the 7×7×2048 DRAM cells), and the NoC idles at every barrier. Target: the interleaved cells with the most input tiles per cluster once Refinements 2 and 4 have landed: 2×7×7×4096 and 2×2×2×8192 (batched, wide C), 32×32×50 (Ht = 32 on 2 clusters), in DRAM. Free-selected region (no loose case is perf-flagged). Levers from `ttnn/ttnn/operations/examples/master.md`: `double_buffer` (batch several chunk rows per barrier, ring = 2× the batch) and `trid_double_issue` (two transaction ids so one batch stays on the wire across the other's barrier). Both are T2, so one phase. No SUPPORTED change.

**Verifier notes**: the reader is explicit sync because of the R1 zero fill. Only the last position tile row needs the barrier before its fill, so the earlier rows may stay in flight across batches. The landing slot of a batch issued ahead of the write pointer is the catalog's documented trap (`trid_double_issue` README). `INPUT_DEPTH_CHUNK_ROWS` is the depth knob: turn it on the host, do not inline a new literal.

**Done when**: structural count, reads outstanding at each reader barrier on the target cells, goes from `min(core_ct_tiles, dest_chunk_tiles)` to at least `min(tiles left on the cluster, INPUT_DEPTH_CHUNK_ROWS × dest_chunk_tiles / 2)`. A test or probe reads it off the op (for example a per-cluster batch count the reader writes to a debug ring, or the host-side batch plan the reader is driven by). The golden suite stays green, and nothing regresses across the config-spanning guard set (one representative per kernel path × layout × placement). The changelog states that the gain in time is unconfirmed on this backend.
