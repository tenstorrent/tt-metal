# Operation Requirements: max_pool2d

## Definition
- **Formula**: `out[n, i, j, c] = max over (a, b) in the window of in[n, i*sh - ph + a*dh, j*sw - pw + b*dw, c]`; a padded position never wins (R1). `H_out = (H + 2ph - dh(kh-1) - 1)//sh + 1`, `W_out` alike. Images flattened channels-last: input row `(n*H + i)*W + j`, output row `(n*H_out + i)*W_out + j`.
- **PyTorch Reference**: `torch.nn.functional.max_pool2d` on `input.reshape(N, H, W, C).permute(0, 3, 1, 2)`, flattened back to `[1, 1, N*H_out*W_out, C]` (`reference()` in `eval/specs/max_pool2d/spec.py`).
- **Import Path**: `from ttnn.operations.max_pool2d import max_pool2d`
- **Function Signature**: `max_pool2d(input_tensor, batch_size, input_h, input_w, channels, kernel_size, stride, padding, dilation, *, compute_kernel_config: ttnn.ComputeConfigDescriptor = None) -> ttnn.Tensor` (result row-major, the input's dtype, sharded in L1 per R9)

## Phases

> **Non-regression rule**: Every refinement must pass all tests from prior phases.
> **Drift signal**: XPASS-strict failures mean the implementer added support but forgot to update SUPPORTED. The implementer fixes by updating SUPPORTED.
> **Checkbox protocol**: Implementer marks `[x]` when the refinement is complete and all tests pass, `[~]` when real work landed but at least one named axis value is deferred (treated as completed by the queue, surfaced as partial), `[ ]` only when nothing usable was produced.
> **Refinement ID + follow-up naming (mandatory — the runner parses this)**: Primary refinements are `Refinement N` (e.g. `Refinement 1`, `Refinement 2`). When you ship `[~]` partial and file the sharper follow-up the partial-tick protocol requires, name it by appending a lowercase letter to the parent's number: `Refinement 1b`, `Refinement 1c`, … (never `Refinement 1.5`, `Refinement 1 (follow-up)`, or a fresh number). Order follow-ups immediately after their parent so the queue runs them before later refinements — a partial's remaining-blocker follow-up must be picked next, not leapfrogged. The runner's parser matches exactly `Refinement \d+[a-z]?`; any other shape is invisible to the queue and silently skipped.
> **Perf gates on this backend (ttsim)**: the simulator has no clock, so every perf refinement below is gated on a **structural count** read off the op by a test or probe (value before → value that makes it done). The gain in time is **unconfirmed** until a timed backend or hardware measures it; never report a simulator timing as a speedup.

### [x] Phase 0 — Core Implementation

- **SUPPORTED dtype**: [bfloat16]
- **SUPPORTED fp32_dest_acc_en**: [True]
- **SUPPORTED layout**: [TILE, ROW_MAJOR]
- **SUPPORTED placement**: memory_config ∈ {DRAM, L1} × sharding ∈ {interleaved, height, width} (DRAM-sharded is INVALID)
- **SUPPORTED shape-derived axes**: batch=single, channel_alignment=tile_aligned, window=overlapping, pad=padded, dilate=none
- **Cores**: output shards (R9) × whole-round position sub-ranges on spare cores, up to the full 8×4 grid; thread plan (R, C, W) = (4, 4, 2) per cluster
- **Compute config**: `default_compute_kernel_config()` = HiFi4, fp32_dest_acc_en=True, approx off; any fidelity / approx accepted (R4)
- **Golden baseline**: 72 / 72 non-skipped, non-xfail tests passing (68 supported cells incl. 1 loose + 3 numeric regressions, 4 rejections); 896 xfail_expected; 480 INVALID-skipped

### [ ] Refinement 1 — Window geometry and channel generality (batches, disjoint / unpadded / dilated windows, non-tile-aligned C)

**Goal**: add `"batched"` to `SUPPORTED["batch"]`, `"disjoint"` to `SUPPORTED["window"]`, `"none"` to `SUPPORTED["pad"]`, `"dilated"` to `SUPPORTED["dilate"]` and `"non_tile_aligned"` to `SUPPORTED["channel_alignment"]`, in both layouts and every placement. Non-tile-aligned C must be served **in the kernel** for TILE inputs too: the reader stages the tile pages covering the run's `ceil(C_core/32)` channel tiles and the gather reads only the face-row bytes inside the stream pitch (`stick_pitch = align_up(C_core·esize, l1_align)/esize`). The last face-row segment of a position is shortened to the pitch, never past it. Padding columns beyond C are never addressed (R7). This removes the `"a tile-layout channel run must be a multiple of 32"` refusal in `create_program_artifacts`.

**Implementation skill**: /memory-layouts

**Verifier notes**:
- Knob-turn class: every value here is the same staged-gather scheme. The verifier opened `SUPPORTED` to `TARGET` in a scratch run (not committed). Every bfloat16 cell with `batch=batched`, `window=disjoint`, `pad=none`, `dilate=dilated` or `fp32_dest_acc_en=False` then passed (439 / 481 bf16 cells), as did non-tile-aligned C in ROW_MAJOR and in TILE width-sharded. The only bf16 failures were the 42 TILE + non-tile-aligned cells in interleaved and height-sharded placements, all hitting the refusal named above. So the geometry half is a gate change plus its tests, and the TILE non-aligned gather is the kernel work.
- Keep it performant (§ anchor rule): a non-aligned interleaved C currently gets `n_w = 1` output shard (`_interleaved_width_cores`) and relies on position sub-ranges to fill the grid. Check that such a cell still engages the grid (e.g. `1x64x64x47`, `1x32x64x17`) rather than one core, and record the cores engaged in the changelog.
- Batches: `Geometry.receptive_sticks` / the reader's `receptive_field` stage whole image rows from the first image's top row to the last image's bottom row of a sub-range. Confirm a sub-range that crosses an image boundary stages no more than the rows its positions read (R6: no window reaches into the next image).
- No wrapper: `ttnn.to_layout` / `ttnn.untilize` before the kernel is not an acceptable way to land the TILE non-aligned cells (native kernel support policy).

**Done when**: those five axis values are in `SUPPORTED`; every bfloat16 cell carrying them passes the golden suite (exact), the TILE non-tile-aligned interleaved and height-sharded cells included; no new `EXCLUSIONS`; prior tests green.

### [ ] Refinement 2 — Numerical configurability: float32 and 16-bit DEST

**Goal**: add `ttnn.float32` to `SUPPORTED["dtype"]` and `False` to `SUPPORTED["fp32_dest_acc_en"]` (float32 + `fp32_dest_acc_en=False` stays refused by rule R2, natively in `validate()`). Wire the compute kernel's unpack mode for 32-bit `window_taps` pages, and keep every result exact: the max selects an input element, and every cell is graded `exact`.

**Implementation skill**: /numeric-formats-metal

**Verifier notes**:
- bf16 with `fp32_dest_acc_en=False` already passed every cell in the verifier's widened scratch run, so that half is a gate change. float32 failed every cell (240) at program build: `hardware_config.cpp:193` — *"Compute kernel 'compute' consumes FP32 DFB 'window_taps' with enable_32_bit_dest=true, but provides no unpack_modes entry for this DFB"*.
- **This is not a zero-kernel-change refinement on Quasar.** Settle first, with a probe on non-monotonic data, whether `UnpackToSrc` of a float32 page keeps all 32 bits on Quasar (exactness is the test). Only if it does not, restructure the fold: load ≤ section-size taps, fold them, carry the running max.
- Partial-tick escape hatch: if float32 is blocked after a focused attempt, ship `fp32_dest_acc_en=False` (bf16) as `[~]`. In that case file `Refinement 2b` for float32, naming the blocker and the probe that showed it.
- Ordered after Refinement 1 (precision knobs rank below layout / alignment work in difficulty). Refinement 1's non-aligned TILE gather must be written in terms of `esize` / `face_row_bytes` (it already is in Phase 0), so float32's 64 B face rows extend it with no second rewrite.

**Done when**: `float32` and `False` are in the two lists; every float32 cell (fp32_dest_acc_en=True) and every bf16 `fp32_dest_acc_en=False` cell passes exact; the float32 + False cells are refused with `NotImplementedError` (R2 check); the YOLOv8 / ResNet bf16 cells are unchanged.

### [ ] Refinement 3 — Speed up the YOLOv8 SPPF call: separable window

**Type**: perf

**Goal**: no loose case is perf-flagged, so the region is chosen here. It is the heaviest model call in the specification: YOLOv8 SPPF, `image(1, 20, 20, 256, (5, 5), (1, 1), (2, 2))`, bfloat16, ROW_MAJOR, DRAM interleaved, default compute config (in SUPPORTED since Phase 0). Each output stream tile costs K = 25 tap tiles gathered by the reader and 25 unpacks + 24 SFPU max passes on its NEO. Build the op design's deferred **separable-window** regime: pass 1 folds the 1×kw horizontal taps into a staged row-max field covering `kh_eff + rows − 1` rows, and pass 2 folds the kh×1 vertical taps from it. Both passes stay in the one dispatch, with the same gather and fold machinery. Pick and tune the lever with the patterns in `ttnn/ttnn/operations/examples/master.md` (compute block size; keeping bytes in flight for the extra stage). No SUPPORTED change.

**Structural count (gate)**: tap tiles the compute folds per output stream tile on the SPPF cell, counting `window_taps` entries popped per `pooled_tiles` entry pushed plus any row-max entries. **Before: 25.** **Done: ≤ 14** (5 vertical + 5 horizontal × the row-max halo factor of a 4-tile sub-range). Read it with a probe that counts ring entries off the program's runtime and compile-time args, and record before and after in the changelog. The time gain is unconfirmed on ttsim.

**Verifier notes**: a scheme-change, so this refinement stands alone (one T3 is a whole phase). The regime applies only where `sh = sw = 1` and K ≥ 9; every other window keeps the direct fold. The L1 ledger's cheapest-traffic finding is folded in here: the default `position_splits = 4` reads SPPF's input 1.65× from DRAM (channels-only would read it 1.0×), and that rests on the op being compute-bound at 25 folds per tile. Once the per-tile compute drops, re-record that row of `l1_ledger.md` with both counts (input DRAM bytes, NEOs engaged) at `MAX_POSITION_SPLITS ∈ {1, None}`. Keep the default only with a positive reason.

**Done when**: the count is ≤ 14 on the SPPF cell; results stay exact; the golden suite is green; and there is no count regression on the guard set (one representative per kernel path × layout × placement: ResNet TILE height-sharded, SPPF ROW_MAJOR width-sharded in place, a TILE DRAM-interleaved cell). The changelog says the time gain is unconfirmed until a timed backend measures it.

### [ ] Refinement 4 — Speed up the ResNet-50 calls: threads sized to the work, fewer TILE gather transactions

**Type**: perf

**Goal**: the ResNet-50 call `image(1, 32, 32, 64, (3, 3), (2, 2), (1, 1))`, bfloat16, TILE, height-sharded in L1 (the runner deals it over 32 cores), at the default config. Each core owns 8 output rows = half a stream tile (`T_core = 1`). Under the plan (4, 4, 2), three of the four NEOs recompute that tile and three readers gather it again. Each gather is two 32 B face-row reads per 32 channels per position, which a strided window (`sw = 2`) cannot coalesce. Levers: size the thread plan to the work, per the target page's thread-plan rule (*"with fewer units than that per cluster, engage fewer … rather than pad every one"*; the plan stays one host constant, chosen per call from `max T_core`), plus the gather-transaction levers in `ttnn/ttnn/operations/examples/master.md` (`read_coalescing`, `split_reader`; the design's deferred untilized-staging row for TILE). Several T1/T2 levers fit in this one phase. No SUPPORTED change.

**Structural count (gate)**: core-local gather NoC reads per core on that cell. **Before: 1,152** (4 NEOs × 9 taps × 32 face-row reads, from a Python replica of `gather_tap_tile`). Padded-tile recomputes per core, **before: 3**. **Done: ≤ 288 reads per core and 0 recomputes** on that cell. Also record the stage + gather read total on the one-core R10 cell (`test_height_sharded_on_one_core_then_pooled_again[TILE]`). Any TILE-gather lever must not raise that total. Time gain unconfirmed on ttsim.

**Verifier notes**: knob-turn class (thread plan, transaction shape). The test that shows every thread ran (`test_every_neo_owns_a_real_tile`) must keep passing at the default plan. A per-call plan needs its own proof that the reduced plan's threads all ran.

**Done when**: the counts above hold; results stay exact; the golden suite is green; and there is no count regression on the guard set named in Refinement 3.

### [ ] Refinement 5 — Block and depth co-tune: DEST subblocks, writer barriers, ring handshakes

**Type**: perf

**Goal**: the SPPF cell of Refinement 3 (after it lands) and the ResNet cell of Refinement 4. Turn the block / depth knobs the design exposed (perf lamps in `op_design.md`):
- fold two output tiles per DEST section where 32-bit DEST half-sync leaves four slots;
- one writer barrier per `out_ring_depth` tiles instead of per tile (catalog `double_buffer`);
- a `taps_per_batch` divisor of K near 4–8 instead of the whole window;
- compute ring handshakes per tap batch instead of per tap, once indexed `copy_tile` on a STRIDED ring has been verified at C = 4.

Levers: `ttnn/ttnn/operations/examples/master.md` (`double_buffer`, `compute_block_size`, `trid_double_issue`). No SUPPORTED change.

**Structural count (gate)**: per output stream tile on the SPPF cell, before → done:
- DEST acquire/release sections: 1 → ≤ 0.5
- writer barriers: 1 → ≤ 0.5
- compute `wait_front` calls: 25 → ≤ K / taps_per_batch

Read them off the kernels' loop bounds with a probe or test. Time gain unconfirmed on ttsim.

**Verifier notes**: several cheap knob-turns in one phase; it runs after Refinements 3 and 4 because both change the loops these knobs sit in. The equal-entries-per-thread rule on STRIDED rings still holds for every knob value: capacities stay whole multiples of the push and pop units.

**Done when**: the three counts hold; results stay exact; the golden suite is green; and there is no count regression on the guard set named in Refinement 3.
