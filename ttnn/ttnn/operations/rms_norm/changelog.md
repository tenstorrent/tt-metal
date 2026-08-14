# Changelog: rms_norm

## Phase 0 — Core Implementation

- **Date**: 2026-08-13
- **What was done**: Initial implementation via the incremental pipeline (planner → implementer → verifier).
  Native `ProgramDescriptor` op with a 2D core partition — `num_row_groups` rectangles × `num_hidden_slices`
  cores per rectangle — including a real cross-core combine on the dependent (hidden) axis
  (gather-to-root + `Mcast2D` broadcast of the finalized `rsqrt`). TILE and ROW_MAJOR are both handled
  natively in the kernels (tilize/untilize staging, no host-side transform), non-tile-aligned H and W
  natively (W-mask in compute for TILE, reader tail zero-fill for ROW_MAJOR), and the `1/W` factor uses the
  true element count exactly once, after the combine.
- **SUPPORTED at Phase 0**: dtype=[bfloat16, float32], fp32_dest_acc_en=[True],
  layout=[TILE, ROW_MAJOR], alignment=[tile_aligned, w_non_aligned, h_non_aligned], rank=[2, 3, 4],
  gamma_mode=[gamma, no_gamma], gamma_dtype=[bfloat16, float32, "none"],
  gamma_layout=[TILE, ROW_MAJOR, "none"], memory_layout=[INTERLEAVED]. `EXCLUSIONS = []`.
- **Accuracy achieved**: bf16 PCC=0.999997, max_abs_err=0.051, mean_abs_err=1.2e-03, rel_rms_err=2.4e-03;
  fp32 PCC=0.999999, max_abs_err=0.026, mean_abs_err=9.0e-04, rel_rms_err=1.6e-03
  (measured over 40 cells = 5 shapes × 2 dtypes × 2 layouts × gamma/no-gamma via
  `test_rms_norm_precision_baseline.py`). Scale check: `median(got/true)` = 0.9998 (bf16) / 0.9986 (fp32),
  width- and alignment-independent ⇒ no padded-denominator / scale bug.
- **Golden suite at Phase 0**: 737 / 737 supported cells passing; 6174 xfail_expected, 33900 invalid_skipped;
  supported_fail = xpass_drift = xfail_wrong_mode = 0 (per `verifier_report.json`, results dir
  `/tmp/rms_verify2`). Full run: `PASSED=752 FAILED=0 ERRORS=0 SKIPPED=33900 HANGS=0 TOTAL=40828`.
- **Issues encountered** (all fixed in this verification pass, no drift fixes were needed):
  1. ROW_MAJOR reader/writer compared a *stick* counter against the tiles-per-barrier knob `DM_CHUNK_TILES`,
     barriering ~8× more often than the TILE path; both now derive `RM_CHUNK_STICKS` from the same knob.
  2. `cb_w_mask` was allocated on every program; now created only when `mask_enabled`
     (needs a compile-time gate in the reader because `prepare_reduce_mask` `static_assert`s on the CB format).
  3. `IN_CB_DEPTH` looked like a free knob but is load-bearing at 1 (the in-place rewrite of x needs
     `write_ptr == read_ptr`); now asserted with the reason.
  4. Added `static_assert` on the ROW_MAJOR staging stick-pitch alignment.
  5. `l1_ledger.md` was stale in five places against the shipped CB set; brought current.
- **Tests added**: `test_rms_norm_precision_baseline.py` (40 cells, incl. the got/true ratio scale-bug
  detector). Pre-existing: `test_rms_norm.py` (immutable acceptance, 107), `test_rms_norm_debug.py`,
  `test_rms_norm_perf.py` (device-ns harness + the two knob sweeps).

## Refinement 1 — Numerical configurability expansion (unlocks the perf target's config)

- **Date**: 2026-08-14
- **What was done**: grew the precision surface to the whole TARGET rectangle, with **zero kernel
  changes** — the pass condition of `/numeric-formats-metal` held (compute is fully helper-based:
  `tilize` / `eltwise_chain` / `sum_of_squares` / `reduce` / `untilize`; no hardcoded sizes or
  data-format constants in any kernel).
  - `SUPPORTED["fp32_dest_acc_en"] = [True, False]`. The axis only selects the DEST accumulation
    width; the four stat CBs (`cb_sq_partials`, `cb_gathered_partials`, `cb_rms_bcast`,
    `cb_rms_recip`) stay `float32` regardless of input dtype **and** regardless of this axis —
    Phase 0 already pinned them that way (`stat_tile = ttnn.tile_size(ttnn.float32)`), so the
    measured accuracy requirement is met by construction, not by a new branch.
  - `ttnn.bfloat8_b` added to `SUPPORTED["dtype"]` and to `SUPPORTED["gamma_dtype"]`. CB formats
    were already derived from `input_tensor.dtype` / `gamma.dtype` / `output_tensor.dtype`, so the
    tile size (1088 B) and the `Bfp8_b` page format fall out of the existing derivation.
  - `EXCLUSIONS = [{"dtype": ttnn.float32, "fp32_dest_acc_en": False}]` — now expressible for the
    first time, and a permanent refusal, not a refinement candidate.
  - **The one host-side fix**: `Tensor.element_size()` raises `ValueError: datum for bfp2, bfp4,
    bfp8 is invalid` for block-float formats (`tt_backend_api_types.hpp:83`), which broke every
    `bfloat8_b` cell at descriptor-build time. Added `_elem_bytes()` in the program descriptor,
    which substitutes 1 for block-float. The `*_ELEM_BYTES` compile-time args it feeds are consumed
    **only** by the ROW_MAJOR stick paths, and a block-float tensor is necessarily TILE-layout
    (`{bfloat8_b, ROW_MAJOR}` is INVALID in `feature_spec.py`), so the substituted value is never
    dereferenced — it only has to keep the reader/writer `RM_STICK_PITCH % 16 == 0` static_assert
    (which is evaluated on *every* program, not just ROW_MAJOR ones) true.
  - `l1_ledger.md` symbol table brought current: `tile_bytes ∈ {1088, 2048, 4096}`, with the
    block-float datum-size note. No CB was added, resized, merged or deleted; the footprint
    expression and the data-movement budget are unchanged (`bfloat8_b` is the *smallest* tile, so
    it only shrinks the footprint and the `float32`-everywhere fit predicate stays a conservative
    upper bound).
- **Accuracy achieved** (device-measured, `test_rms_norm_precision_matrix.py`, 112 passed /
  36 skipped):
  - **The perf target's exact config** — bf16 / HiFi2 / `fp32_dest_acc_en=False` / TILE / bf16 TILE
    gamma, `(1,1,32,W)` for W ∈ {1024, 2304, 5120, 7168}: PCC **0.99998** at W=7168
    (rel-RMS 0.0138), against the `_perf_case` soft gate of 0.9995. Refinement 3 can now be
    specified at its real config instead of an `fp32_dest_acc_en=True` proxy.
  - `fp32_dest_acc_en=False` at bf16, across all 8 matrix shapes × 2 distributions × 4 fidelities:
    PCC ≥ 0.995 gate held everywhere including LoFi.
  - `bfloat8_b`: PCC ≥ 0.9998, rel-RMS ≤ 0.020 across input × {bf16, fp32, bfloat8_b, no} gamma ×
    both accumulation widths × shapes up to `(1,1,32,8192)` — against the golden bf8b tolerance of
    0.99 PCC / 0.10 rel-RMS, i.e. **~40× margin on rel-RMS**. The verifier's flagged hazard (the
    in-place double rewrite of `cb_input_tiles` re-quantizes `x·r` to block-float before the gamma
    multiply) is real but far below tolerance, so the fused in-place path was **kept** and no cell
    was excluded for it.
  - No regression: `test_rms_norm.py` (107) + `test_rms_norm_precision_baseline.py` +
    `test_rms_norm_debug.py` = 175 passed.
- **Golden test progress** (targeted slices, not the full suite — the harness re-runs everything):
  - `pad_poison` group: **6/6 interleaved passing** (all 6 were xfail at Phase 0). The
    padding-in-the-denominator guard holds at `fp32_dest_acc_en=False`.
  - `perf` group: **8/8 interleaved passing** (all were xfail at Phase 0), including the decisive
    `(1,1,32,7168)` case and all four prefill shapes.
  - `resilience` group: **86/86 interleaved passing** (all were xfail at Phase 0).
  - Cartesian `bfloat8_b` cells: **450/450 passing**. 5-shape full cartesian slice (all
    dtype × fp32_dest_acc_en × layout × gamma combos): **288/288 passing**.
  - `eval/golden_tests/rms_norm/test_regression.py`: 15/15.
  - Every remaining xfail in those groups is a `*_SHARDED` `memory_layout` — Refinement 2's scope.
- **Issues encountered**: one, the `element_size()` block-float raise described above. It presented
  as a clean `ValueError` at descriptor build (a free failure), not as a hang or a numerical
  mismatch. Neither of the two hazards the verifier flagged cost anything: `DEST_AUTO_LIMIT`'s
  4→8 doubling is handled entirely inside the helpers (no number is hardcoded anywhere in the op),
  and the `bfloat8_b` in-place re-quantization stayed ~40× inside tolerance.
- **Tests added**: `tests/ttnn/unit_tests/operations/rms_norm/test_rms_norm_precision_matrix.py` —
  the authoritative precision characterization mandated by `/numeric-formats-metal` §10, in three
  parts: `test_rms_norm_precision_matrix` (the gated axes dtype × `fp32_dest_acc_en`, over 8 shapes
  × 2 input distributions), `test_rms_norm_precision_fidelity` (the ungated `math_fidelity` axis,
  all 4 values, on 2 representative shapes), and `test_rms_norm_perf_target_config` (Refinement 3's
  exact config pinned as a named cell at its tighter 0.9995 gate, so a later perf phase cannot
  silently substitute an `fp32_dest_acc_en=True` proxy). Skips are exactly the op's refusal
  surface: `{float32, False}` (EXCLUSIONS) and `bfloat8_b` on a non-tile-aligned shape (INVALID).

## Refinement 2 — Sharded placement: HEIGHT / WIDTH / BLOCK

- **Date**: 2026-08-14
- **What was done**: `HEIGHT_SHARDED` / `WIDTH_SHARDED` / `BLOCK_SHARDED` added to
  `SUPPORTED["memory_layout"]` and consumed **natively**, plus the `memory_config` kwarg on the entry
  point (the output inherits the input's placement; a mismatched request raises a `ValueError`).

  *Reused, not rebuilt.* All three flavours are the Phase 0 logical scheme with the geometry pinned by
  the caller, exactly as `op_design.md`'s "Physical shard placement" lamp says: HEIGHT is the
  `num_hidden_slices == 1` corner (the reduce stays core-local), WIDTH is the already-built
  gather-to-root + `Mcast2D` combine over one row-group, BLOCK is the Phase 0 2D partition with one
  row-group per grid row. So the kernels' block schedule, combine, W-mask and ragged-tail accounting
  are untouched. The delta is:
  - `_plan_sharded()` **reads** `num_row_groups` / `num_hidden_slices` / `slice_hidden_tiles` /
    `shard_rows` off the shard spec instead of running the rect search (`HIDDEN_TILES_PER_CORE_FLOOR`
    does not apply on this path); `block_rows` is then the largest **divisor** of `shard_rows` that
    fits, a divisor so every block is the same size.
  - **Zero-copy CB placement.** TILE: `cb_input_tiles` / `cb_output_tiles` ARE the caller's resident
    shards (`ttnn.cb_descriptor_from_sharded_tensor`), so `load_block` moves nothing and the writer's
    pop *is* `store_block`. ROW_MAJOR: the shards bind to `cb_shard_in` / `cb_shard_out` and the
    tilize/untilize staging the layout already needs reads and writes them **core-locally** (L1→L1),
    never through a `TensorAccessor`.
  - **A ragged WIDTH shard grid is supported, not refused.** The verifier's constraint (a) proposed
    excluding a non-rectangular grid; on an 11-wide grid almost every `Wt` lands as "N full rows + a
    partial row", so that would have dropped ~90 % of the WIDTH cells. Instead the row-group's
    **bounding box** is the mcast rect, with `Mcast2D(num_active = s−1)` — the helper's documented
    divergent ack count ("the mcast box holds inactive cores that receive but never ack"). The
    non-member cores carry the CBs, so the broadcast lands in reserved L1 and the sender still waits
    for exactly the real receivers.
  - Buffer-depth knobs stay live and are now **turned by the L1 solve** on the ROW_MAJOR-sharded path
    (`rm_in_depth`, `rm_out_depth`: 2 → 1 → and only then the smallest configuration), because a shard
    pins `slice_hidden_tiles` and the depths are the only remaining slack.
- **Accuracy achieved**: PCC ≥ 0.9999, rel-RMS ≈ 0.005 — i.e. at the interleaved path's own level — on
  11 adversarial shapes × 3 placements × 2 layouts × {bf16, fp32, bfloat8_b} × {gamma, no_gamma},
  including `(1,1,256,512)`, `(1,1,3232,96)`, `(1,1,4064,160)`, `(1,1,32,4064)`, `(7136,736)`,
  `(13,777,1023)` and the poisoned-padding shapes. bfloat8_b sharded: PCC 0.99986 / rel-RMS 0.018
  against a 0.99 / 0.10 gate. All five **pinned sharded perf geometries** run green, including the
  block-sharded prefill `(1,1,8192,1024)` `[1024,128]`/(8,8) — the first case ever to exercise
  `block_rows < shard_rows`.
- **Golden test progress**: sharded `pad_poison` 18/18; sharded `perf` 5/5; a 52-cell sharded
  `resilience` slice 48/52. Unit directory 317 passed / 36 skipped, with the 107 interleaved
  regression tests unchanged.
- **Issues encountered**: three real bugs, every one of them invisible to the interleaved suite.
  1. `buffer_num_pages()` counts *shard* pages, so on a width/block-sharded ROW_MAJOR tensor it
     reports (sticks × width-shards) rather than the stick count — the row-group count came out
     N× too large. `total_sticks` now comes from the padded shape.
  2. A ROW_MAJOR shard's **width granule is the L1 alignment (8 elements at bf16), not the tile**, so
     a width/block shard's slice can start off the 64 B DRAM boundary. The per-core gamma stick read
     then silently returned the wrong bytes — PCC 0.23–0.57, and correlating *exactly* with
     `slice_elem_base·elem % 64 != 0` across every shape. Fixed with one DRAM-aligned burst covering
     the whole slice into scratch pages, then hand-placing only the 32 row-0 lanes each tile actually
     uses (`BroadcastDim::Row` reads nothing else).
  3. **The in-place pack indexed from the read window instead of the CB base.** A resident-shard
     `cb_input_tiles` has capacity = the whole shard, so once the L1 solve cut `block_rows` below
     `shard_rows` the read pointer stopped wrapping to base each block — and because only
     reserve/push move a *consumer's* write pointer, and compute never pushes that CB, every block
     after the first rewrote block 0's pages and dropped its own `1/rms` factor. Diagnosed by
     magnitude rather than by instrumentation: the error tracked `1/sqrt(2W)` — precisely the row-rms
     spread, i.e. "the scale is missing" — across W = 96/160/736/1184/3072, and vanished with
     `gamma=None` (which takes the non-in-place branch). Fixed with a runtime
     `pack_base = (block·B·S) % IN_WAIT_TILES`, which is 0 whenever the CB holds exactly one block, so
     the interleaved path is byte-identical. `(1,1,3232,96)` WIDTH: rel-RMS 0.0717 → 0.0050.
- **Left failing on purpose (the next refinement's baseline)**: 4 cells, one class — wide-W
  `HEIGHT_SHARDED`. A HEIGHT shard pins `slice_hidden_tiles = Wt` on *every* core, so x + out + gamma
  alone are ≈ 3·W·2 B and the CBs reach 1.7–3.0 MB against this part's 1.57 MB L1 (W = 6144 and
  11008, and `(3104,4064)`). There is no knob left: the shard spec fixes the hidden extent. The exit
  is the design's lamped **TwoPassStreaming** regime — sub-chunk the hidden axis and re-read x with
  `Accumulate::at` across chunks — which is a scheme change, so per the OOM rule these are left
  failing rather than silenced with an `EXCLUSIONS` entry.
- **Tests added**: `tests/ttnn/unit_tests/operations/rms_norm/test_rms_norm_sharded.py` — 3 placements
  × 2 layouts × 3 shapes for placement correctness; `test_rms_norm_sharded_multi_block_keeps_the_scale`,
  which asserts the plan really is `block_rows < shard_rows` before bounding rel-RMS well under the
  dropped-scale signature (so the pin cannot silently stop testing what it names); and
  `test_rms_norm_sharded_is_zero_copy`, a **structural** check that the input/output CBs carry the
  tensors' buffers — because a `TensorAccessor` re-read of a core's own shard is numerically correct
  and so is invisible to every value-based test.

## Refinement 3 — Speed up the perf-flagged decode profile

- **Date**: 2026-08-14
- **What was done**: three levers on the interleaved decode profile, plus one methodology fix that
  had to come first. **Reused, not rebuilt**: every lever is a knob or a branch on a path that
  already existed — the hidden-split search, `McastConfig.handshake`, the reader's gamma load, and
  the compute kernel's combine reduce. Added: one derived cap, one CT flag, one algorithm dispatch.
  No new CB, no new kernel file, no second program-descriptor branch.

  0. **The premise had to be re-measured first.** The queue said the two knob sweeps were "already
     done and came back flat within 1-2 %, so do not re-spend the phase there". Both were void: the
     fixture did `monkeypatch.setattr(pd, KNOB, v)` on the module imported as
     `ttnn.operations.rms_norm.rms_norm_program_descriptor`, but the op executes in a **second import
     of that same file** with a different `__dict__` — so every "sweep" re-measured the shipped
     configuration. Caught by disbelief at the data, not by inspection: a floor of 32 puts
     `(1,1,32,1024)` on a *single* core with no combine at all, and it cannot cost the same 5.6 us as
     the 8-core plan. Both harnesses now patch `create_program_descriptor.__globals__`, the dict the
     op really reads. Re-measured honestly the knob is worth up to 25 %.
  1. **A fan-in bound on the hidden split** (`FANIN_BALANCE_K`, with `_fanin_slice_cap`). Splitting
     the hidden axis `s` ways cuts the per-core transfer (~c2·Wt/s) but grows the cross-core combine
     (~c1·s: `s` stat tiles incast into one root, `s` gather atomics, a root reduce over `s` tiles, a
     barrier across `s` cores). `HIDDEN_TILES_PER_CORE_FLOOR` bounded only `S`; nothing bounded `s`,
     so the search maximized occupancy and ran the combine off the end (`s = 56` at W=7168).
     Minimizing the sum puts the optimum at `s* ∝ √Wt` — measured k ≈ 2.13, which reproduces the
     measured optimum at Wt = 224/160/72/32. **This is the bulk of the win.**
  2. **TILE-layout gamma is read row-0-only.** gamma is a [W] vector, so a TILE gamma is padded to a
     whole tile-row: only row 0 of each of its `Wt` tiles carries data and `BroadcastDim::Row` is the
     only consumer. The reader now issues two 16-element face-row reads per tile (both offsets are
     multiples of 64, so both start on legal DRAM boundaries) instead of the 2 KB page — **32× fewer
     gamma bytes**, which in the decode regime (Rt = 1) was a full *third* of the op's DRAM traffic.
     Gated off for block-float gamma, whose faces share an exponent header.
  3. **The mcast pre-handshake is dropped when there is only one broadcast** (`num_blocks == 1` — the
     whole decode regime). It costs the root `s-1` inbound remote atomics and buys exactly one thing,
     that broadcast n+1 not overwrite a landing still holding broadcast n. The safety argument that
     survives: every receiver constructs its `ReceiverPipe` (initing its own data-ready flag) at
     kernel boot, and the root cannot send until it has gathered all `s` partials — which is strictly
     after every contributor's reader passed that ctor.
- **Accuracy achieved**: PCC **0.99998** on `(1,1,32,7168)` at the perf group's exact config
  (bf16 / HiFi2 / `fp32_dest_acc_en=False` / TILE / bf16 TILE gamma / INTERLEAVED) against its 0.9995
  soft gate; unchanged elsewhere (the levers move bytes and geometry, not arithmetic, except the
  combine-reduce dispatch, whose accumulate datapath is equal-or-better in fp32).
- **Performance** (Blackhole p150b, 11×10 grid, one fresh-cache run per point, device kernel ns):

  | shape (target config) | before | after | |
  |---|---|---|---|
  | `(1,1,32,7168)` — the decisive case | 11340 | **9657** | **−14.8 %** (goal ≤14894) |
  | `(1,1,32,5120)` | 9428 | 8441 | −10.5 % |
  | `(1,1,32,2304)` | 6615 | 6535 | −1.2 % |
  | `(1,1,32,1024)` | 5609 | 5587 | −0.4 % |

  At the Phase 0 config (`test_rms_norm_perf.py`, incl. prefill) every row is at or below its
  recorded baseline: decode −7.8…−11.6 %, `prefill_2048x1024` −0.4 %, `batch4d` −3.0 %, floor −1.5 %.
- **Golden test progress**: `perf` group **13/13** (8 interleaved + 5 sharded); `pad_poison`
  **24/24**; a 4-shape full-cartesian slice (`2x1x128x100`, `4x1x512x512`, `1x1x224x3072`,
  `1x1x160x11008`) 296 passed / 78 xfailed / **2 failed** — and those 2 are exactly the wide-W
  `HEIGHT_SHARDED` cells Refinement 2 left failing on purpose (`_plan_sharded` does not use the
  fan-in cap, so this refinement cannot have touched them).
- **Issues encountered**: two, both caught as regressions and both fixed rather than accepted.
  (1) A **constant** fan-in cap (32) was the first thing measured and is the wrong shape: tuned to
  the wide single-row decode shapes it costs tall-and-wide ones the occupancy they need
  (`(1,1,64,12288)` +5.5 %). Replaced by the √Wt rule, which the cost model predicts and which fixes
  it (−1.9 %). (2) `ReduceAlgorithm::AccumulateViaAdd` on the root's combine is a null above its
  crossover (9770 vs 9732 ns at s=32, inside noise) and a real **regression below** it (the prefill
  geometry lands at s=2 and paid 2.5 %), so it ships as master.md prescribes — a *dispatch* on reduce
  width, never slower than the library. That null is the informative part: the combine is NoC-bound,
  not math-bound, so the residual cost is the gather **incast**. Separately, the compute kernel's
  recorded claim that `ReduceWithinTile::Skip` is unreachable is now **stale** — the `static_assert`
  has been moved after the `AccumulateViaAdd` early-return upstream.
- **Left on the table (a finding, not a task)**: a **two-level tree combine** (reduce along the
  rect's x axis to row-leaders, then y to the root — master.md `tensix_all_reduce`, 1.45–1.60× over a
  flat root on 2-D groups). On the shipped 8×4 rect it cuts the serial incast from 32 tiles (131 KB)
  to 8 + 4; the model puts it near another −20 %. Not attempted: it is a real topology change to a
  combine shared by the interleaved path *and* all three sharded schemes, and banking a measured,
  regression-free 14.8 % was the better trade with the budget left. Also **retired**: the
  `GammaBroadcast` scheme lamp — lever 2 shrank the DRAM term it exists to remove from 11 % to 0.4 %,
  so it should be struck, not built (`l1_ledger.md` updated, incl. the corrected gamma crossing
  `g·W·elem` in the data-movement budget).
- **Tests added**: `tests/ttnn/unit_tests/operations/rms_norm/test_rms_norm_perf_decode.py` — the
  decode harness at the perf target's *exact* config (the pre-existing harness measures a ROW_MAJOR
  gamma at HiFi4 / fp32-acc-on, a materially cheaper cell), with file-based selectors for the ablation
  and for both hidden-split knobs, and the measured before/after table in its docstring.
  `test_rms_norm_perf.py` keeps its per-shape numbers but its two A/B tables are annotated VOID with
  the reason, and its patch target is fixed.

## Refinement 4 — Speed up the perf-flagged prefill profile

- **Date**: 2026-08-14
- **What was done**: two levers on the interleaved prefill profile, plus one measurement that
  reframed the whole phase. **Reused, not rebuilt**: both levers are knob turns on parameters the
  descriptor already exposes (`DM_CHUNK_TILES`, `IN_CB_DEPTH`, `block_rows`), and the mechanism that
  makes the second one legal is the `pack_base` machinery Refinement 2 already built for resident
  shards — generalized by one word (modulo the CB's *capacity* instead of its *wait count*). No new
  CB, no new kernel file, no second program-descriptor branch. Diff: ~40 lines of descriptor, one new
  CT arg threaded to two kernels, one line changed in each kernel.

  0. **The premise had to be re-measured first — and it was wrong.** The queue put prefill "roughly
     1.9× off" at ≈184 GB/s, extrapolated from `test_rms_norm_perf.py`'s `prefill_2048x1024` row.
     That row is a *different shape at a different config* (HiFi4 / `fp32_dest_acc_en=True` /
     ROW_MAJOR gamma). Measured at the perf group's real config, the Refinement-3 baseline for
     `(1,1,8192,W)` was already **365–405 GB/s** of in+out DRAM traffic, and all four shapes were
     already under their `achievable_ns`.
  1. **`DM_CHUNK_TILES` 8 → 32** — the "keep bytes in flight" lever, and it moves BOTH halves of the
     dataflow because the reader and the writer read the same knob (the ROW_MAJOR kernels convert it
     to sticks through the same expression). At 8 the reader drained the NoC to empty every 16 KB and
     paid the DRAM round-trip serially. Swept 8/16/32/64/128; 32 is the first value at the plateau
     and the only one that wins on every row. The decode regime is untouched — a decode block is 4–7
     tiles, below even the old value.
  2. **The `IN_CB_DEPTH` × `block_rows` L1 ladder** — the design's **Overlap** perf lamp, turned in
     exactly the form the lamp and the queue both specify ("smaller `block_rows` + a second input
     buffer", never "same block, deeper CB"). `IN_CB_DEPTH` had been asserted at 1 since Phase 0
     because the in-place rewrites of x pack at `get_write_ptr(cb) + i*page` and a compute thread that
     never pushes that CB keeps its write pointer at the base forever. The real invariant is weaker:
     the pack index only has to *track the read window*, which `(block*BLOCK_TILES) % IN_CAPACITY_TILES`
     does for any whole-block-multiple capacity. `IN_WAIT_TILES` and `IN_CAPACITY_TILES` are now
     separate CT args (they already differed on the sharded path; conflating them is what pinned the
     knob). The two knobs are then **co-solved by one ladder** against one budget, because they trade
     against each other inside it — and the deeper rung is declined on three predicates:
     - a core owning fewer than `in_depth` blocks (nothing to prefetch — this is the **whole decode
       regime**, which therefore stays byte-identical to Refinement 3),
     - the L1 predicate not affording the second buffer (`(1,1,8192,7168)` lands here and keeps its
       Refinement-3 program exactly),
     - the resulting block being shorter than one in-flight NoC burst (`DM_CHUNK_TILES`). This third
       guard was added *because of a measured regression*: ungated, the rung fired on
       `(1,1,2048,1024)` with a 16-tile block and cost +1.4 % — one hidden read of a few tiles does
       not pay for an extra set of per-block fixed costs (init + reconfig + a whole gather/mcast
       round trip at `s > 1`). The threshold reuses `DM_CHUNK_TILES` rather than adding a second
       literal beside it.
- **Accuracy achieved**: unchanged — neither lever touches arithmetic (one moves barrier granularity,
  the other moves which L1 pages a block lands in). PCC **0.9998–0.99998** on all four prefill shapes
  at the group's exact config against its 0.9995 soft gate; `test_rms_norm_double_buffer.py` bounds
  rel-RMS at < 0.25× the dropped-scale signature `1/sqrt(2W)` on every double-buffered shape.
- **Performance** (Blackhole p150b, 11×10 grid, **median of 3 fresh-cache runs per point**, device
  kernel ns, the perf group's exact config — bf16 / HiFi2 / `fp32_dest_acc_en=False` / TILE / bf16
  TILE gamma / INTERLEAVED):

  | shape | Refinement 3 | Refinement 4 | | `achievable_ns` | `ttnn.neg` (DRAM roof) |
  |---|---|---|---|---|---|
  | `(1,1,8192,1024)` |  92899 |  **88316** | **−4.9 %** |  96744 |  85899 |
  | `(1,1,8192,2304)` | 207404 | **187575** | **−9.6 %** | 211345 | 198515 |
  | `(1,1,8192,5120)` | 414599 | **405003** | **−2.3 %** | 738307 | 436850 |
  | `(1,1,8192,7168)` | 570644 | **558273** | **−2.2 %** | 1032281 | 607069 |

  Decode, at its own exact config (`test_rms_norm_perf_decode.py`): 5594 / 6536 / 8509 / 9487 ns for
  W = 1024 / 2304 / 5120 / 7168 against Refinement 3's 5587 / 6535 / 8441 / 9657 — inside noise, as
  it must be, since the ladder produces a byte-identical program there. Phase-0 harness
  (`test_rms_norm_perf.py`, HiFi4 / fp32-acc / ROW_MAJOR gamma), R3-exact vs shipped, 2 runs each:
  every row at or below — `decode_2rows_w12288` −2.9 %, `batch4d` −2.4 %, `prefill_2048x1024` −0.5 %,
  the rest flat.
- **What actually binds, and how I know**: the profile is **DRAM-bandwidth-saturated**. Two
  independent probes say so.
  - A **reference-op ceiling**: `ttnn.neg` on the identical four shapes — a production, tuned eltwise
    unary that does exactly one read and one write of the same bytes with no reduction, no gamma and
    no cross-core combine — costs 85899 / 198515 / 436850 / 607069 ns. rms_norm is now
    **1.03× / 0.95× / 0.93× / 0.92×** of that, i.e. at or *under* the wall of an op doing strictly
    less work. There is no schedule-side headroom left on this profile.
  - A **fidelity probe** (a zero-code compute-cost classifier): LoFi is indistinguishable from HiFi2
    and HiFi4 costs only +4–6 %, so the FPU math exposed on the wall is a few percent at most.
- **Issues encountered**: two, both caught as regressions and both fixed rather than accepted, and
  both are the same shape of mistake — *a coarser or deeper structure is not free when the wall is a
  serial read*.
  1. **Widening `l1_working_budget` is a measured LOSS.** The queue named it as lever (1) ("a larger,
     measured budget directly coarsens `block_rows`"). Raising the fallback to the part's real
     1.46 MB does coarsen it — `(1,1,8192,5120)` goes B=2 → 5 and `(1,1,8192,7168)` B=1 → 2 — and
     costs **+5.7 %** and **+3.7 %**. A coarser block means a *longer fully-serial DRAM read* before
     compute can start, which is the opposite of what this profile wants. Kept at 928 KB, now for a
     measured reason instead of only a conservative one (`l1_ledger.md` updated).
  2. **The overlap rung, ungated, regressed `(1,1,2048,1024)` by +1.4 %** — see lever 2's third
     predicate. Also measured and rejected: the **writer twin** `OUT_CB_DEPTH` at 3 and 4 (3 is
     inside noise on the two narrow shapes and +1.4 % / +2.7 % on the two wide ones; 4 is worse), so
     the writer's depth was already at its winning value and the writer's *batching* moved with the
     reader's through the shared `DM_CHUNK_TILES`.
- **Left on the table (a finding, not a task)**: nothing on this profile. The interleaved prefill
  regime is within 3–8 % of a bare unary's DRAM wall, so the next real win is not a kernel change
  here — it is Refinement 5's sharded geometries, whose references (25640 ns for the block-sharded
  prefill) sit *below* the measured ~3.5 µs dispatch + boot floor plus any transfer, i.e. a
  fixed-cost problem rather than a bandwidth one.
- **Tests added**: `tests/ttnn/unit_tests/operations/rms_norm/test_rms_norm_perf_prefill.py` — the
  prefill harness at the group's exact config (the pre-existing harness measures a cheaper cell), with
  file-based selectors for `DM_CHUNK_TILES` / `IN_CB_DEPTH` / `OUT_CB_DEPTH` / the L1 budget and a
  fidelity probe, patching `create_program_descriptor.__globals__` (the dict the op really executes
  in). `test_rms_norm_double_buffer.py` — the new path's pin: the immutable acceptance suite's tallest
  shape is 8 tile-rows, so on a 110-core grid **no existing test reaches the depth-2 rung at all**, and
  a wrong pack base is silent (right values, one block missing its `1/rms`). Each case asserts the
  *plan* (`in_depth > 1`, and a block to prefetch) before the values, plus the two negative pins —
  short blocks and the whole decode regime must decline the rung. `test_rms_norm_perf.py` gained the
  same two knob selectors so the R3-vs-R4 A/B is reproducible at the Phase 0 config.

## Refinement 5 — Speed up the sharded perf geometries

- **Date**: 2026-08-14
- **What was done**: three levers on the sharded profile, all of them **reused, not rebuilt** —
  a reordering of work the reader already did, and two knobs on chains that already existed. No
  new CB, no new kernel file, no second program-descriptor branch. New: one bool in the reader,
  two compute CT args, two descriptor knobs.

  0. **A harness first, because none of these cells had ever been timed.**
     `test_rms_norm_perf_sharded.py` reproduces the five pinned geometries exactly as
     `helpers.run_rms_norm` builds them (`eval.sharding.shard_config` off `extras.shard_shape` +
     `extras.core_grid`) at the group's exact config. Baseline vs `achievable_ns`:
     4655/4110, 5535/4617, 6634/5267, 7229/5481, 75727/25640.
  1. **The gamma DRAM read is overlapped with the combine.** *This is the whole decode win.*
     An ablation (`gamma=None`) costed gamma at **1.2-2.0 us of a 4.7-7.2 us wall** — implausible
     for ~500 B of payload, and the reason is ordering, not bytes: once x and out are resident L1,
     gamma is the *only* DRAM tensor, and `load_gamma_once` ran (with its barrier) BEFORE the
     resident-shard publish, while compute waited on `cb_gamma_tiles` at boot. So a full DRAM round
     trip stood in front of `Sum(x^2)` and the entire cross-core combine, for an operand not touched
     until `apply_gamma_block`. Now: the reader publishes x first, issues the gamma read, and defers
     its `noc_async_read_barrier` + push into the block loop (where, on the interleaved path, the
     same barrier already flushes block 0's x read); compute waits gamma at the apply site. Costs
     nothing anywhere and helps the interleaved decode too.
  2. **DEST-window batching** (`DEST_BLOCK_TILES` x `DEST_BLOCK_MIN_ROW_TILES`). The streaming
     eltwise passes walked one tile per `tile_regs` acquire/commit/wait/release — a hard MATH<->PACK
     barrier per tile. Batching them into DEST windows (`IterationShape::block_size`) lets the packer
     drain tile i while math runs tile i+1: BLOCK-shard prefill **74417 -> 72063 (-3.2%)**. It is
     **gated to cores owning >= 2 tile-rows** because the single-tile-row decode regime measured
     **+1-3%** (one group, so the extra pipeline fill/drain never amortizes) — that regime stays
     byte-identical to Refinement 4.
  3. **Gamma fusion** (`GAMMA_FUSE_MIN_ROW_TILES`), **landed, measured, and parked OFF at a
     byte-identical default.** `scale_block` and `apply_gamma_block` in ONE DEST window via
     `DestReuseBinary`, with gamma expanded from its row-0 vector form to full tiles once by a
     `UnaryBcast<Row>` in-place pass (DEST-reuse has no intra-tile broadcast). Halves the L1
     crossings and removes a `sync_pack_to_unpack` per block — and is a **measured loss**
     (74417 -> 78635, +5.7%): `mul_reuse_dest_tiles` copies DEST back into srcA between two dependent
     ops on the math thread, whereas the unfused pair is two independent DEST windows that pipeline
     against the packer. On this op the math thread binds and the L1 crossings do not. The lever is
     correct (the whole sharded suite is green with it on) so it is kept as a live knob at 0.
- **Accuracy achieved**: unchanged — no lever touches arithmetic (one moves a barrier, one moves the
  DEST window boundary, one is off). PCC > 0.9995 on all five geometries against the perf group's
  soft gate; the whole unit directory is green at its Refinement-2 tolerances.
- **Performance** (Blackhole p150b, the group's exact config + pinned geometry, device kernel ns):

  | geometry | R4 base | shipped | | `achievable_ns` | ratio |
  |---|---|---|---|---|---|
  | `(1,1,32,1024)` WIDTH `[32,128]`/(8,1) | 4655 | **3833** | **-17.7 %** | 4110 | **0.93x** |
  | `(1,1,32,2304)` WIDTH `[32,256]`/(9,1) | 5535 | **4507** | **-18.6 %** | 4617 | **0.98x** |
  | `(1,1,32,5120)` WIDTH `[32,160]`/(8,4) | 6634 | **5725** | **-13.7 %** | 5267 | 1.09x |
  | `(1,1,32,7168)` WIDTH `[32,256]`/(7,4) | 7229 | **5732** | **-20.7 %** | 5481 | 1.05x |
  | `(1,1,8192,1024)` BLOCK `[1024,128]`/(8,8) | 75727 | **72041** | **-4.9 %** | 25640 | 2.81x |

  Two of the five are now **inside** their reference and two are within 5-9 %. No interleaved
  regression — every prior perf row improved or held: decode (R3/R4 -> now) 5594/6536/8509/9487 ->
  **5299/6230/8223/9371**, prefill 88316/187575/405003/558273 ->
  **85704/187039/401015/563222** (the last is +0.9 %, inside run-to-run noise on a shape whose
  program is unchanged).
- **What actually binds now, and how I know**: for the four WIDTH decode geometries, the **fixed
  boot + dispatch floor** (the interleaved 1-tile floor case measures 3055 ns) plus the cross-core
  combine. The gather-payload ablation puts the incast at ~1.3 us of the 32-core case. For the BLOCK
  prefill the split is: gather incast **12.4 us (17 %)**, the gamma pass ~6.5 us, and the rest is the
  serial per-block schedule on the row-group ROOT, which does its own 128 tile-ops *plus* a 64-tile
  reduce per block while seven contributors wait on it. A block-factor sweep confirms the schedule
  reading: coarser is better on a resident shard (block_rows 4 / 8 / 16 -> 76188 / 72041 / 70039),
  the exact opposite of Refinement 4's interleaved prefill, because there is no serial DRAM read to
  lengthen — only per-block fixed costs to amortize.
- **Issues encountered**: three, all caught as regressions or hangs and all resolved rather than
  accepted. (1) DEST batching with a group **wider than S** hangs: the grid walk groups row-wise, so
  the row's tail group is short and its per-tile output lifecycle under-pushes — the writer sits in
  `cb_wait_front(cb_output_tiles)` while all three TRISCs have already exited. Fixed by clamping the
  group to the largest **divisor** of S. (2) A grouped walk needs a **PerBlockSize** output
  lifecycle, not PerTile (kernel_lib's own `block_size` example pairs them that way); with PerTile it
  under-pushes the same way. (3) The gamma fusion regression above. Also confirmed **pre-existing,
  not a regression**: `13x777x1023` WIDTH_SHARDED fails a CB-vs-L1-buffer clash — reproduced
  identically at the Refinement 4 commit.
- **Left on the table (a finding, not a task)**: the BLOCK-shard prefill is still 2.8x its reference
  and the two levers that would close it are both topology/L1 changes, not knobs. (a) The **root is
  the critical path**: a two-level tree combine (Refinement 3's deferred lever) would cut both its
  serial incast and its per-block 64-tile reduce; the gather ablation bounds the NoC half at 17 %,
  and the reduce half is on top of that. (b) **block_rows wants to be 16** (-2.8 % measured), which
  needs a wider `l1_working_budget` on the sharded path — deliberately not taken, because one sharded
  resilience cell already clashes CBs against L1 buffers at the *current* conservative budget, and
  trading a 2.8 % win on one geometry for clash risk across the sharded corpus is the wrong bet.
- **Tests added**: `tests/ttnn/unit_tests/operations/rms_norm/test_rms_norm_perf_sharded.py` — the
  sharded harness at the five pinned geometries, with file-based selectors for `DM_CHUNK_TILES`,
  `GAMMA_FUSE_MIN_ROW_TILES`, `DEST_BLOCK_TILES`, the L1 budget (the sharded lever on `block_rows`),
  math fidelity, and a `gamma=None` ablation; the measured before/after table, both ablations and the
  block-factor sweep are recorded in its docstring. Also
  `tests/ttnn/unit_tests/operations/rms_norm/probes/read_perf_csv.py` (prints device kernel ns from
  the newest profiler CSV).

## Perf 1 — Tournament: the cross-core combine was the row-group's critical path

- **Date**: 2026-08-14
- **Round**: 1 of 2. Nothing moved in `SUPPORTED` — a perf tournament changes device-ns only.
- **Focus case (perf-flagged, MANDATORY, measured at its EXACT config, never a proxy)**:
  `(1,1,8192,1024)` **BLOCK_SHARDED**, shard `[1024,128]`, core grid `(8,8)`, bf16 / TILE /
  bf16 TILE gamma in DRAM / `math_fidelity=HiFi2` / `fp32_dest_acc_en=False`, Blackhole p150b
  @1350 MHz. Chosen because it was the only one of the 13 `perf`-group cells materially off its
  reference: **2.87× `achievable_ns`** (72041→73570 measured vs 25640), where the other twelve were
  at 0.93–1.09× or already under. Every axis of that config is in `SUPPORTED` — no generality gap.
  Plan: `s=8` hidden slices, `S=4` hidden tiles/core, `block_rows=8`, 4 blocks, 32 shard tile-rows
  per core, 64 cores = **8 row-groups of 8 cores, each a 1-D line along grid x**.

### Measured breakdown (permanent instrumentation + cumulative ablation)

All three kernels are now instrumented with `MaybeDeviceZoneScope` from the new
`ttnn/cpp/ttnn/kernel_lib/perf_instrumentation.hpp` — **permanent** (the macro compiles out when the
profiler is off) with **wait split from work** and **NoC issue split from barrier**, so the numbers
are payload and not occupancy. Coverage verified before ranking anything: 100 markers max per
`(core, RISC)` against the 250 cap, and the last user zone ends at **1.00** of the `*-KERNEL` span on
every RISC (exhaustion is silent, so this check is mandatory).

Per-stage device-ns **per core, summed over the run** (baseline, 73570 ns wall):

| zone | ns/core | cores | reading |
|---|---|---|---|
| `rd_bcast_recv` | **61257** | 56 | the non-root cores sat in `ReceiverPipe::receive()` for **84 % of the wall** |
| `cp_combine` (math / pack / unpack) | **55847 / 50691 / 32256** | 8 | the root's own reduce over `s*B = 64` fp32 stat tiles per block |
| `rd_gather_wait` | 15761 | 8 | the root's view of the incast |
| `cp_combine_wait` | 14048 | 8 | the same event on the root's unpack thread (hoisted out of the helper's `BulkWaitBulkPop`) |
| `wr_gather_issue` | 7582 | 64 | contributor-side issue (max/p50 = 1.35) |
| `rd_load_total` | 6961 | 64 | resident shard: pure bookkeeping + back-pressure |
| `cp_scale` / `cp_sumsq` / `cp_apply_gamma` | 4734 / 4458 / 4425 | 64 | the local payload — small |

**Cumulative ablation** (payload stubbed, CB/semaphore/barrier scaffolding kept, one fresh-cache
profiled run each; peeled cumulatively because stages overlap and a solo removal under-counts):

| variant | ns | attributed |
|---|---|---|
| full | 73570 | — |
| − the root's combine reduce payload | 31528 | **root reduce = 42042 ns (57 %)** |
| − also the gather incast payload | 19578 | **incast = 11950 ns (16 %)** |
| − also the local compute payload (sumsq + scale + apply_gamma) | 12084 | local = 7494 ns (10 %) |
| floor (boot + dispatch + sync scaffolding + mcast round trips) | 12084 | 16 % |

The verdict this licenses: **floor + local work = 19578 ns is already BELOW the 25640 ns reference**,
so the entire 48 µs of excess was the combine and the target was reachable by fixing it alone.

**Roofline gate** (`/perf-ceiling-dm`), which is what decided *which* lever to spend on:
- the incast moves **1 MB of fp32 stat tiles into ONE core** and measured **87 GB/s** ≈ the
  single-core L1 write-port roofline — so the *transfer* had no headroom. Only the **topology** did.
- the root's reduce chews the same 1 MB, whose **information content is 8192 fp32 (32 KB)** — a 32×
  over-ship. So both halves pointed at "fewer bytes, or more destinations", not at a faster reduce
  (Refinement 3 had already measured `AccumulateViaAdd` vs `ReduceTile` as a null here).
- the local passes move ~768 KB of L1 traffic in 7.5 µs ≈ 100 GB/s ≈ the L1 roofline: at most one
  pass (~2.5 µs) was removable.

**Ranked bottleneck: (1) the root-serialized combine reduce, (2) the gather incast, (3) local
compute, (4) the fixed floor.**

### Portfolio floated (5 ideas, deliberately overlapping)

| # | idea | tier | verdict |
|---|---|---|---|
| I1 | **reduce-scatter the combine** — scatter the block's rows across the group's cores instead of funnelling into one root | T3 | **WIN → GRADUATED** |
| I2 | **two-level tree combine** on the row-group rect (the lever deferred by Refinements 3 and 5) | T3 | WIN, not graduated (superseded where I1 applies) |
| I3 | **compact transposed stat gather** — collapse + `transpose_dest` at the contributor, ship 2×64 B, `REDUCE_COL` at the root (32× payload cut) | T3 | WIN, not graduated (superseded by I1) |
| I4 | **software-pipeline the block loop** + fewer/coarser combine round trips | T3 | WIN, L1-gated, not graduated |
| I5 | **fuse scale+gamma in one DEST window** via a mechanism other than the already-measured `DEST_TO_SRCA` | T2 | NULL for the fusion; a WIN found beside it that did **not** transfer |

I1/I2/I3 all attack the same stage on purpose — that overlap is what let the tournament pick by
measurement instead of by argument.

### Per-idea results (measured in isolation, in the subagents' own benches)

**I1 — reduce-scatter the combine — WIN, 2.25× on the stage.** `s=2` 46216→36142 (1.28×) · `s=4`
56892→30452 (1.87×) · **`s=8` (focus) 64965→28890 (2.25×)** · `s=16` 81454→40709 (2.00×) · `s=28`
53146→26285 (2.02×) · `s=32` 57169→31028 (1.84×) · `B=4` 35074→20783 (1.69×) · `B=32`
L1-infeasible→25040. Domain: every `s>1` with `min(s,B)>1`. One exception, **measured**: `B=1`
12579→13754 (0.91×) and `s=32,B=1` 19586→20699 (0.95×). Arithmetic bit-identical. No helper bypass.
Also returned a **trap worth propagating**: `noc_semaphore_inc_multicast` as the gather signal is
1.05× faster at `s=8` but its path reservation serializes against every other row-group's and
inverts the result on a 2-D group (`s=32`: 149716 vs **31073** ns for per-owner unicast atomics).
Shipped with unicast atomics.

**I2 — two-level tree combine — WIN, not graduated.** `8×4` (s=32,B=1, flagged) 4467→**3473**
(1.286×) · `7×4` (s=28,B=1, flagged) 4195→3461 (1.212×) · `8×8` 6736→3822 (1.762×) · `8×4,B=8`
27045→17593 (1.537×) · the prefill focus geometry (8 lines of 8) 14867→13222 (1.124×). Best fan-in
`K` = the divisor of `s` nearest `√s`; the cost tracks `max(K, s/K)`, so the lever really is fan-in.
Domain: every rect **and both line orientations** (a tree on a 1-D line does *not* degenerate) with
`s*B >= 16`; measured regression at `s*B == 8` (0.93× / 0.96× / 0.88×). **Not graduated** because
where it applies at `B>1` I1 is 2× larger and incompatible, and after I1 each owner gathers only
`s*own_rows = 16` tiles — exactly where I2 measured its weakest 1.045–1.065×. **It remains the right
lever for the one regime I1 cannot reach** (`B == 1` with `s*B >= 16`: the `(1,1,32,5120)` and
`(1,1,32,7168)` WIDTH geometries) and is the **round-2 target**.

**I3 — compact transposed stat gather — WIN, not graduated.** 63534→**51025** (1.245×) at the focus
geometry, and *flat in `s`* where the baseline is linear in it: `s=16` 1.525×, `s=32`/`s=64` run at
all where the baseline exceeds L1. The informative half is the two negative controls: collapsing at
the contributor without the transpose (65449) and shipping 2 KB instead of 4 KB (65879) are both
**nulls**, so the root's cost is the **number of tiles its unpacker chews**, not bytes in flight.
Measured regression at `s<=3` (0.89× / 0.94×). Not graduated: I1 is the larger, simpler win on the
same stage. Two findings that must travel with the idea if round 2 revives it: a partial-landing-tile
scheme **needs a cross-core ordering edge** (the root's boot-zero of un-owned lanes raced the
contributors' writes and wiped landed partials — stat PCC 0.048), and `get_latest_programs_perf_data()`
returns every *unread* program, so summing them double-counts (a 2× inflated number until fixed).

**I4 — pipeline the block loop — WIN (−14.9 % on the stage), L1-gated, not graduated.** `pipe_b8`
68091→**57935**; `s=4` 59642→54058 (−9.4 %); a 2048-row shape 18887→16996 (−10.0 %); byte-identical
and flat at one round trip (the decode regime). The win is **85 % of the 11950 ns the ablation
attributes to the incast**, and it scales with the number of round trips available to overlap. Two
corrections to the idea's own premise, both measured: **decoupling the stat block from the apply
block is a NULL on a resident shard** (66043 vs 66026 — the apply block's L1 cost there is *zero*
because `cb_output_tiles` IS the shard), and **the two levers are substitutes, not additive**
(pipe+coarse 54361 ≈ pipe alone 54058). Blocked at `stat_rows >= 16` by a verbatim CB-vs-L1 clash
(depth-2 landing needs 1410 KB against a measured ~937 KB CB region). Not graduated this round: it
is a second large restructure of the same loop I1 just rewrote, and I1 shrank the exposed combine it
exists to hide. Its L1 term is the `s × SB` landing buffer — which I1 cut 8× — so it is a **round-2
candidate whose gate I1 has partly opened**.

**I5 — scale+gamma DEST fusion — NULL (every mechanism), plus a WIN that did not transfer.**
Baseline two-pass 8836 ns; `DEST_TO_SRCA` (the parked `GAMMA_FUSED` path) 13018 (1.47× — reproducing
this op's recorded +5.7 % exactly, so the harness is calibrated); the untried mirror `DEST_TO_SRCB`
13413 (1.52×); an SFPU DEST binary 75373 (8.5×); `sfpu_mul_bcast_row` 29800 (3.4×). **Mechanism: this
stage is MATH-thread-bound and its L1 crossings are free**, because PACK and UNPACK overlap MATH — so
two FPU ops per tile is the floor and every fusion just adds math-thread work. The fusion question is
now closed. Beside it the subagent found `interm_cb_bulk` — keep two independent DEST windows but
pack pass 1 into a one-block intermediate CB, whose own push/wait replaces `sync_pack_to_unpack`:
8836→8621 (0.976×) at the focus, 1076→**954** (0.887×) at the decode-exact point, no regression on
any geometry measured. **It was integrated, measured end-to-end, and reverted**: whole-op flat
everywhere (sharded focus 34554→34301; interleaved prefill `w1024` **median of 3 87231→87004,
−0.26 %**; decode and the four WIDTH cells flat), because the apply passes are not on the critical
path — and its `B*S`-page CB consumed exactly the L1 that I1 had freed for `block_rows` 16, trading a
measured −2.8 % lever for a +2.4 % one. Recorded, not forced in.

Two bugs the subagent turned up, routed rather than fixed here:
- **`ckl::DestReuseBinary` returns silently WRONG data when `block_size == DEST_AUTO_LIMIT`**
  (PCC 0.9866; correct at window 4). `chain_max_block_v` advertises 8 as legal, so the caller gets no
  signal. This op's parked `GAMMA_FUSED` path would therefore be **wrong**, not merely slow, on any
  `S` making `DEST_BLOCK` 8 — including the perf-flagged `(1,1,32,7168)` at `S=8`. It stays off; the
  knob's comment now carries the warning.
- `api/compute/sfpu_binary_bcast.h` documents an `fp32_dest_acc_en=True` requirement that is **not
  real** (PCC 0.99999 at `False`).

### What graduated, and how widely

**I1, as the op's ONE combine path.** `num_owners = min(s, block_rows)` clamped to a divisor of
`block_rows`; owner `o` owns the block's rows `[o*own_rows, (o+1)*own_rows)`, gathers only those
(`s*own_rows` pages instead of `s*B`), reduces + finalizes them, and funnels its `own_rows` finished
tiles into the root's `cb_rms_bcast` at page `o*own_rows` + one atomic; the root broadcasts once,
unchanged. **The code it replaces is deleted** — there is no flat-root implementation left.

It is **not a dual path**: `num_owners == 1` gives `own_rows == block_rows`,
`cb_combine_out == cb_rms_bcast` and no funnel, which *is* the pre-Perf-1 program, reached through the
same code. The single **carve-out** is `min(s, block_rows) == 1` and it is earned by a **measured
regression** (0.91× at `s=8`, 0.95× at `s=32`: with one owner there is nothing to scatter, so the
scheme would be the flat root *plus* a funnel hop and a semaphore). That is `block_rows == 1`, i.e.
the decode regime, and the predicate is written as the narrow exception (`num_owners == 1` → the
degenerate form), so it shrinks as `block_rows` grows rather than having to be widened by hand.
Everything else — interleaved, ROW_MAJOR, all three sharded schemes, every dtype, masked and
unmasked — takes the new path unqualified, including regimes not benchmarked.

**Second-order win, not designed for:** the gather landing shrinks from `s*B` to `s*own_rows` pages
(1 MB → 32 KB at the focus geometry), so the L1 ladder now affords `block_rows` **8 → 16** — the
coarser block Refinement 5 measured at −2.8 % and had to decline.

Instrumentation extended to the new path (`rd_stat_funnel_wait`, `rd_stat_funnel`), so per-stage
observability did not regress.

### Whole-op before/after (zoned vs zoned — see the note below)

| case (its exact config) | before | after | | `achievable_ns` |
|---|---|---|---|---|
| **`(1,1,8192,1024)` BLOCK `[1024,128]`/(8,8)** | 73570 | **34554** | **2.13×** | 25640 (2.87× → **1.35×**) |
| `(1,1,32,1024)` WIDTH `[32,128]`/(8,1) | 4483 | 4493 | flat | 4110 |
| `(1,1,32,2304)` WIDTH `[32,256]`/(9,1) | 5314 | 5300 | flat | 4617 |
| `(1,1,32,5120)` WIDTH `[32,160]`/(8,4) | 6413 | 6487 | flat | 5267 |
| `(1,1,32,7168)` WIDTH `[32,256]`/(7,4) | 6328 | 6363 | flat | 5481 |
| interleaved prefill 1024 / 2304 / 5120 / 7168 | 89021 / 187246 / 399336 / 559167 | 87953 / 186816 / 399074 / 568245 | flat | 96744 / 211345 / 738307 / 1032281 |
| interleaved decode 1024 / 2304 / 5120 / 7168 (+floor) | 6121 / 7080 / 8936 / 10131 / 3544 | 6164 / 7092 / 9138 / 10215 / 3533 | flat | 9149 / 17003 / 75825 / 104259 |

The four WIDTH cells and the whole interleaved decode profile are `block_rows == 1`, so their
programs are **byte-identical** and flat is the correct result. The interleaved prefill is
DRAM-bandwidth-saturated (Refinement 4 put it at 0.92–1.03× of a bare `ttnn.neg`), so the combine is
not on its critical path; the `w7168` +1.6 % is inside the run-to-run spread measured on this harness
(1.6 % across three runs of `w1024`).

**A measurement caveat that matters for the next round.** The permanent zones cost **nothing** in a
production build, but **while profiling** they cost ~+17 % on a 4 µs kernel and ~+2 % on a 73 µs one.
Every number in this entry is therefore **zoned vs zoned**; comparing any of them against Refinement
5's un-zoned numbers (e.g. `4483` here vs `3833` there) reads as a regression that does not exist.

**What binds now** (re-measured on the focus case after graduation, wall 34554 ns): the combine is no
longer concentrated — `rd_bcast_recv` collapsed **61257 → 2966** ns and `cp_combine` now runs on all
64 cores. The new top items are `wr_store_wait` 20712 (the writer idle behind compute), `cp_combine`
math 16776, `rd_gather_wait` 13255 and `cp_scale` pack 11175. Round 2 should re-rank from there.

### Guard-set no-regression

- Golden `perf` + `pad_poison` + `loose` groups: **43/43 pass**.
- Golden `resilience` (the sharded stress set, one shape × layout × placement sweep):
  **333 passed / 8 failed / 3 skipped** — the **same 8 failures** as the pre-change baseline, verified
  by an A/B run of the identical selector (all pre-existing: the wide-W `HEIGHT_SHARDED` cells
  Refinement 2 left failing, plus the `13x777x1023` `WIDTH_SHARDED` CB-vs-L1 clash Refinement 5
  already reproduced at its own commit).
- 4-shape full-cartesian slice (`2x1x128x100`, `4x1x512x512`, `1x1x224x3072`, `1x1x160x11008`):
  **296 passed / 78 xfailed / 2 failed** — identical to Refinement 3's record, same two cells.
- PCC on the focus case 0.9995+ against its soft gate; arithmetic is bit-identical to before
  (sum-then-collapse == collapse-then-sum), so no precision was traded anywhere. `fp32_dest_acc_en`,
  `math_fidelity`, `math_approx_mode` and every dtype are untouched — none was used as a lever.

**A profiler trap worth writing down**: `generated/profiler/.logs/zone_src_locations.log` persists
*across runs*, and moving an existing zone to a new line registers a new `(name, file, line)` string
that can 16-bit-hash-collide with the stale entry, throwing `"Source location hashes are colliding"`
at device teardown. It looks like a test failure and is not one — delete that log after moving zones.
The live zone set was checked and has 39 distinct hashes with no internal collision.

### Helper bypasses

**none.** The graduated path uses `compute_kernel_lib::reduce` unchanged (only its
`ReduceInputBlockShape` narrows from `(B, s)` to `(own_rows, s)`) and `mcast_pipe`'s
`SenderPipe`/`ReceiverPipe` exactly as the op already shipped them. The owner→root funnel is a raw
`noc_async_write` + `Semaphore::up`, which is the **same gather idiom the op already documents** as
having no helper (`mcast_pipe` is one-source-to-a-rectangle; a gather is the opposite shape and
kernel_lib has no gather/scatter helper) — i.e. it is not a new bypass introduced by this round.

Three helper gaps were nevertheless *found* by ideas that did not graduate, and they are reported
here because this table is the only durable channel for them:

| helper | kind | what was missing / hard | helper ns | raw ns | site |
|---|---|---|---|---|---|
| `compute_kernel_lib::reduce` (`ReduceWithinTile::Skip`) | capability | `Skip` — "sum these tiles, do NOT collapse within the tile" — is **unreachable**: the `static_assert(within_tile == Collapse, "Skip is AccumulateViaAdd-only")` at `reduce_helpers_compute.inl:885-891` sits at function scope **after** the `if constexpr (AccumulateViaAdd) { …; return; }` block, so it is not in a discarded statement and fires for the very instantiation it means to permit. Blocks any hierarchical/partial reduce; hit independently by I2's level-1 pre-reduce and by I3's contributor-side variants (b)/(c). | n/a (won't compile) | I2 level 1: raw pairwise `add_tiles(acc_to_dest)` with explicit DEST seeding, 3473 vs 4467 flat-root | not graduated — `perf_experiments/tree_combine/kernels` |
| `compute_kernel_lib` — no transpose helper at all | capability | There is no `ckl::transpose`, and no way to append an in-DEST op that needs its own MATH re-init to a reduce: `transpose_dest` requires `transpose_dest_init` (which rewrites MATH addrmods+MOP) and needs the reduce's SrcA ALU format undone first, but the helper calls `reduce_init` **once outside** its per-output loop, so a `post_reduce_op` doing this corrupts every later output tile of the same call. Concrete ask: let `post_reduce_op` reconfigure MATH (helper re-inits per output-tile group), or add a `reduce_transposed<…>` that fuses `reduce_tile + transpose_dest` in one window. | n/a (silently corrupts later tiles) | I3: `reduce_init → N×reduce_tile → reduce_uninit → transpose_dest_init → N×transpose_dest → finalize → N×pack`, 51025 vs 63534 | not graduated — `perf_experiments/compact_stat_gather/kernels/csg_compute.cpp` |
| `mcast_pipe` `ReceiverPipe` | capability | A receiver cannot wait for a **count of senders** in one round (`Flag` waits one flag; `Counter` waits `wait_min(++round_)`, one increment per round), so an `s`-sender broadcast into disjoint page ranges of one landing CB is inexpressible. Reported as **known but low priority**: I1 implemented it raw and it measured a NULL (1.04–1.22×), so the gap gates only a variant that loses. | n/a | 57219 (mcast variant) vs 28890 (the graduated single-sender funnel) | not graduated — `perf_experiments/reduce_scatter_combine/kernels` |

Two `ergonomics` frictions were re-confirmed rather than newly found, and are already recorded at
their sites in the op: the eltwise **convenience wrappers default-construct their chain elements**, so
they cannot carry a runtime tile base (which is why `mask_tail_block` and the in-place scale are
spelled as explicit `eltwise_chain`s), and the reduce's **finalize is a `post_reduce_op` lambda of raw
SFPU calls** — the helper's own documented extension point.

### Round-2 queue (measured, characterized, not started)

1. **I2 tree combine, restricted to the regime I1 cannot reach**: `block_rows == 1` with
   `s*block_rows >= 16` — i.e. `(1,1,32,5120)` (s=32) and `(1,1,32,7168)` (s=28), the two sharded
   perf cells still above their reference. Measured 1.286× / 1.212× on the stage there. Needs raw
   level-1 adds (see the table), `K` = divisor of `s` nearest `√s`, and level-1 DEST chunking at
   `DEST_AUTO_LIMIT` before it is safe under a caller's `fp32_dest_acc_en=True`.
2. **I4 pipeline**, whose L1 gate I1 partly opened (landing buffer 8× smaller).
3. Re-rank from the post-graduation zones above (`wr_store_wait` / `cp_combine` math /
   `rd_gather_wait`) rather than from this round's ranking.
4. **`DEST_BLOCK_TILES` 8 → 4** is a free measured 4 % at `S=8/16` and flat at `S=4/5` (I5's sweep),
   and it side-steps the `DestReuseBinary` window bug. Not taken here only because it was outside
   every floated idea's scope.

---

## Perf 2 — Tournament: the combine's SFPU tail, the post-combine gamma pass, and 430 KB of unspent L1

- **Date**: 2026-08-14
- **Round**: 2 of 2. Nothing moved in `SUPPORTED` — a perf tournament changes device-ns only.
- **Focus case (perf-flagged, MANDATORY, measured at its EXACT config, never a proxy)**:
  `(1,1,8192,1024)` **BLOCK_SHARDED**, shard `[1024,128]`, core grid `(8,8)`, bf16 / TILE / bf16 TILE
  gamma in DRAM / `math_fidelity=HiFi2` / `fp32_dest_acc_en=False` / `math_approx_mode=False`,
  Blackhole p150b **@1350 MHz** (== `extras.reference_aiclk_mhz`, so no clock scaling is needed).
  It re-entered the round as the only `perf`-group cell materially off its reference (**1.35×
  `achievable_ns`**, 34616 vs 25640); the four sharded decode cells were at 1.11–1.23× and every
  interleaved cell was already under. Every axis of that config is in `SUPPORTED` — no generality gap.
  Plan on entry: `s=8` hidden slices, `S=4` hidden tiles/core, `block_rows=16` ⇒ **2 blocks**,
  `num_owners=8`, `own_rows=2`, 8 row-groups × 8 cores.

### Round shift — Perf 1's graduation moved the critical path, and its round-2 queue was partly wrong

Perf 1's reduce-scatter left a *different* op, so the breakdown was re-run from scratch. Two of its four
queued items did not survive measurement and are corrected below: `wr_store_wait` (queued as the new #1)
is **occupancy, not a bottleneck** — on a resident output shard the writer moves *zero* bytes and that
20734 ns is the mirror of compute's latency — and `DEST_BLOCK_TILES 8 → 4` (queued as "a free 4 %") is
**refuted**, 8 is the best value at every `S`.

### Measured breakdown (permanent instrumentation + cumulative ablation)

Zone coverage verified before ranking anything: **84 markers max per `(core, RISC)`** against the 250
cap, and the last user zone ends at **1.00** of the `*-KERNEL` span on every RISC (exhaustion is
silent, so this check is mandatory).

**An attribution correction that changes the ranking**, recorded because it is easy to get wrong:
`cb_wait_front` in a compute kernel is **UNPACK-only** (`api/compute/cb_api.h`), so the op's hoisted
wait zones only hoist on `TRISC_0`. `TRISC_1`/`TRISC_2` absorb the wait inside the *following* work
zone — `cp_scale` reads 8327 (math) / 11177 (pack) against 4242 on unpack, and the 4–7 µs difference
is precisely the hoisted `cp_rms_wait` (9284), not payload. **Rank off `TRISC_0` and the ablation,
never off the math/pack zone values.**

Per-stage device-ns **per core, summed over the run** (baseline, 34616 ns wall, 64 cores, 2 blocks):

| zone | ns/core | reading |
|---|---|---|
| `wr_store_total` / `wr_store_wait` | 21859 / 20734 | **not a bottleneck** — a resident output shard moves 0 bytes; this is the writer mirroring compute |
| `rd_gather_wait` | 13246 | the owner's view of the incast |
| `cp_combine_wait` (TRISC_0) | 11434 | the same event on the unpack thread |
| `cp_rms_wait` (TRISC_0) | 9284 | compute blocked on the broadcast — **an idle window with nothing in it** |
| `wr_gather_issue` | 6604 | contributor-side issue (max/p50 = 1.44) |
| `cp_scale` / `cp_apply_gamma` / `cp_sumsq` (TRISC_0) | 4242 / 3564 / 3542 | the local payload |
| `rd_bcast_recv` / `rd_stat_funnel_wait` | 2967 / 5603 | post-Perf-1 broadcast + funnel |

**Cumulative ablation** (payload stubbed via the new `RMS_ABLATE` hook, CB/semaphore/barrier
scaffolding kept, one fresh-cache profiled run each; peeled **cumulatively** because stages overlap
and a solo removal under-counts):

| variant | ns | attributed |
|---|---|---|
| full | 34616 | — |
| − the owner's combine reduce **+ finalize** | 29067 | **combine reduce = 5549 ns (16 %)** |
| − also the gather incast payload | 18773 | **gather = 10294 ns (30 %)** |
| − also the local compute (sumsq + scale + apply_gamma) | 8256 | **local = 10517 ns (30 %)** |
| floor (boot + dispatch + 2 blocks of sync scaffolding + mcast round trips) | 8256 | **24 %** |

Overlap cross-checks, which is what makes the peel trustworthy: solo `−gather` = 10349 ≈ cumulative
10294, so the gather is **not** overlap-hidden (it is a serial latency chain); solo `−local` = 13848 >
cumulative 10517, because removing local work also removes the upstream serialization feeding the gather.

**Roofline gate** (`/perf-ceiling-dm`), which is what decided where *not* to spend:
- The gather moves 8 MB of fp32 stat tiles in 10.3 µs = **815 GB/s aggregate** (~15 % of the 64-core NoC
  roofline) and **12.7 GB/s per receiving core** against the 87 GB/s single-core L1 write-port roofline
  Perf 1 measured. Unlike Perf 1's single-root incast, this transfer is **far** from its roofline, so the
  payload has real headroom — and the tile ships 4096 B whose information content after the within-tile
  collapse is 128 B, a **32× over-ship**.
- The `wr_gather_issue` "206 ns per 4 KB write" looked issue-bound and **is not**: a dedicated
  transaction-count sweep (P2) put only **~13 ns of it per transaction**, on top of a ~3000 ns/block
  floor independent of transaction count. Fewer **bytes** is the lever; fewer transactions is not.

**Ranked bottleneck: (1) the gather incast, (2) local compute, (3) the fixed floor, (4) the owner
combine reduce.**

### Portfolio floated (6 ideas, deliberately overlapping)

| # | idea | tier | verdict |
|---|---|---|---|
| P1 | **compact stat gather** — collapse within-tile at the contributor, `transpose_dest`, ship 2×64 B, sum at the owner (32× payload cut) | T3 | **WIN**, not graduated (conflicts with P6; see aggregation) |
| P2 | **coalesce the gather writes** — re-page the landing so a contributor's `own_rows` tiles are contiguous | T2 | **NULL** |
| P3 | **fold `apply_gamma` into the sum-of-squares pass** (3 passes → 2) | T3 | **WIN → GRADUATED**, but as a *reorder*, not the proposed fold |
| P4 | **scope the finalize's SFPU work to the axis that carries data** | T2/T3 | **WIN → GRADUATED** |
| P5 | **one block per core** — stop paying two combine round trips when L1 can afford one | T2 | **WIN → GRADUATED** |
| P6 | **software-pipeline the block loop** so the combine round trip is hidden | T3 | **WIN**, not graduated (superseded by P5 on the focus geometry) |

P1/P2 both attack the gather and P5/P6 both attack the block count — that overlap is what let the
tournament pick by measurement instead of by argument, and in both pairs the *cheaper* idea lost.

### Per-idea results (measured in isolation, in the subagents' own benches)

**P1 — compact stat gather — WIN, 1.319× on the stage.** `25278 → 19164` at the focus geometry. The
baseline is **linear in `s`** (+~1.2 µs per slice); the candidate is **flat** (22.5 → 24.1 µs from
`s=8` to `s=32`), which is the whole mechanism: `s=16` **1.689×** · `s=28` **2.135×** · `s=32`
**2.466×** · `B=32` 1.380× · decode `s=32,B=1` 1.455×. A two-option menu separated the causes:
shipping half the bytes with the same owner tile count (`collapse_2k`) buys 1.128×, and *also* cutting
the owner from `s*own_rows = 16` unpacked tiles to `own_rows = 2` buys the rest. `out_pcc` identical to
the baseline to 7 digits — the collapse-then-sum reordering is measured PCC-neutral, not assumed.
One exception, **measured**: `s == 2` at 0.876× (two contributors do not repay the contributor-side
collapse); `s == 4` is **flat** (0.98–1.06×) and *in* the domain. It also measured the ceiling on the
transaction-count idea at **~3 %** with a deliberately-wrong 64 B probe — which is what let P2 be
discarded on evidence rather than on P1's say-so.

**P2 — coalesce the gather writes — NULL.** `26156 → 26150` (1.000×), bit-identical output. A
transaction-count sweep at fixed bytes and fixed destinations (8 / 16 / 64 transactions per block per
core) put the cost at **~13 ns per transaction** on top of a **~3000 ns/block floor independent of
transaction count**; coalescing buys 33 ns of a 7108 ns stage and the longer barrier tail gives most of
it back. Isolation confirmed against the real op (`wr_gather_issue` 6577 ns/core, max/p50 1.45, vs the
op's 6604 / 1.44), so the null is about the stage and not the harness.

**P3 — gamma before the combine — WIN, 1.123×, by a mechanism the idea did not propose.** The
**fold** into `sum_of_squares` is a **REGRESSION (0.889×)**: the premise "removes a whole read+pack"
is false because `sum_of_squares` packs *nothing* today, so folding gamma in *adds* that pack to pass 1,
delays the last `Sum(x²)` partial and slips the whole group's gather. What wins is the *reorder the
fold implied* — running `apply_gamma` on the **pre-combine** side as its own pass: `34350 → 30579`,
and 1.077–1.186× across `S ∈ {4,8,16}`, `B ∈ {1,8,16,32}`, `s ∈ {2,8,16}`.

**P4 — scope the finalize to column 0 — WIN, 4.12× on the finalize.** The split it was asked for is
the round's most useful single number: the owner combine is 5535 ns of which the **finalize is 3993
(72 %)** and the `reduce_tile` sum only 1542 (28 %) — *the finalize is the combine*. Menu (isolated
MATH-thread ns per finalize): stock 989.7 → `VectorMode::C` 514.8 → scope-only `c_skip` 283.7 →
**fused + `c_skip` 240.1 (4.12×, 8 vector ops instead of 96)**. **Scope is the lever, not fusion**:
fusion alone is 1.10×. Every option is bit-identical on column 0.

**P5 — one block per core — WIN, 1.063×,** and it found that `device.l1_size_per_core` is **not bound
to Python**, so the 1 MB fallback has been running everywhere against a part that reports 1,461,376 B
per bank. Coarser is monotone on TILE+sharded (B1…B32 = 113250 / 78129 / 51881 / 38549 / 34733 /
**32635**) and **U-shaped with its minimum at the shipped value** on ROW_MAJOR sharded.

**P6 — software-pipeline the block loop — WIN, 1.223×** (`34591 → 28282` at lead (1,1)),
**bit-identical** to the serial baseline, no raw LLK, and **no measured regression anywhere**. Head to
head it beat P5's lever *at the time*: pipelined `B=16/nb=2` 28282 ns @ 476 KB CB vs fat `B=32/nb=1`
32765 ns @ 540 KB — faster *and* 64 KB cheaper. The win grows with round trips (nb=8 1.310×, nb=16
1.349×) and it reaches geometries P5 cannot: on `(1,1,8192,2048)` a fat block is **L1-infeasible** while
pipelined `B=8` lead 2 is 1.130× ahead of the best affordable serial point.

### Aggregation — the fastest NON-CONFLICTING combination

- **P2 discarded** (null).
- **P1 vs P6** both rewrite the gather-landing machinery and could not both graduate this round. P6 won
  the overlap on a **measured whole-op** number (28282) against P1's *estimate*, with no raw LLK and no
  measured regression.
- **P5 vs P6** are not substitutes: P6 needs a budget that can afford its extra landing windows, so P5
  is P6's prerequisite. Both were kept.
- **Then P5 changed the answer.** After graduating P4 + P3 + P5, the focus geometry selects
  `block_rows = 32` ⇒ **`num_blocks == 1`** (verified: the per-zone exec count halved 128 → 64). P6's
  leads clamp to 0 at one block and emit a **byte-identical program**, so on the perf-flagged target
  P6's benefit is now **zero** — it was superseded by the cheaper lever it had beaten. It remains the
  lever for L1-constrained deep shards, where P5 cannot be spent at all. Not graduated; queued below.
- **Graduated: P4 + P3 + P5.**

### What graduated, and how widely

**P4 — the finalize's SFPU scope, as the op's ONE finalize.** `rms_norm_finalize.hpp` is a single
scoped+fused sfpi body (`rsqrt(v*inv_w + eps)` per vector at `VectorMode::C` with an even-parity
`dst_reg` stride). **No predicate at all**: it serves *both* call sites — the owner's cross-core
combine and the `s == 1` collapse — and **the three-wrapper chain it replaces is deleted**. Domain:
**everywhere**, verified correct on 18 cells spanning BLOCK / WIDTH / HEIGHT / INTERLEAVED × TILE /
ROW_MAJOR × bf16 / float32 / bfloat8_b × `fp32_dest_acc_en` both × HiFi2 / HiFi4 × W- and H-non-aligned
× gamma / no-gamma. The two interleaved DRAM-bound cells that measured flat (−0.2 %, +0.1 %) are **flat,
not exceptions**, and stay on the unified path — the same config wins 1.065×/1.388× wherever the
finalize is not diluted.

**P3 — `apply_gamma` before the combine.** ONE implementation invoked at one of two points (a
`GAMMA_FIRST` constant picks the slot; `SCALE_IN_PLACE` follows from it, because whoever runs last must
land the block in its final home), **not** a duplicated body. The single **carve-out** is
`NUM_HIDDEN_SLICES == 1` and it is **structural rather than a benchmark artifact**: the entire benefit
is hiding the pass inside the combine's latency, and at `s == 1` there is no combine — `cp_rms_wait`
does not even compile in. Earned by a measured **−2.2 %** (45608 → 46632, reproduced 3×) that the
subagent then *isolated*: with gamma resident in L1 the same reorder is **dead flat** at `s == 1`
(45612 → 45550) while the focus win is unchanged, so the cause is gamma's DRAM round trip, not the
pattern. Written as the narrow exception, so it disappears by itself if the reader ever lands gamma
earlier on that path. `no_gamma` is flat (nothing to move) and stays on the unified path; ROW_MAJOR is
untested-but-included, and the golden suite covers it.

**P5 — the part's real L1 on the TILE+sharded block ladder.** `_block_ladder_budget` spends
`REAL_L1_PER_CORE` only where a coarser block costs nothing. **Two carve-outs, both earned by measured
regressions**, and both written as "the paths that regress keep the conservative budget":

| carve-out | measured reason |
|---|---|
| ROW_MAJOR sharded | the core tilizes its own sticks into a `B*S`-tile CB before the reduce can start — a serial fill. U-shaped, minimum at the **shipped** `B=8`: B1…B32 = 152885 / 119401 / 99630 / **96401** / 100469 / 101807 ns. Coarsening costs 4.1 % / 5.6 %. |
| the INTERLEAVED partition search | +5.7 % on `(1,1,8192,5120)` (401839 → 424912, median of 5). **Refinement 4's attribution of this was wrong**: `block_rows` is 1 in *both* plans; what moves is the search's admission filter, which a wider budget lets pick `G=110,s=1,S=160,depth1` over `G=55,s=2,S=80,depth2`. The interleaved search is therefore not touched at all and every interleaved plan is byte-identical. |

Instrumentation was extended to every new path (`cp_apply_gamma` now fires in either slot, unchanged in
name so the zone series is continuous across the round), so per-stage observability did not regress. The
`RMS_ABLATE` cumulative-ablation hook that produced the breakdown ships with the op — inert without its
`/tmp` file, and it warns loudly on stderr when active because it produces wrong answers by construction.

### Whole-op before/after (zoned vs zoned)

| case (its exact config) | Perf 1 | **Perf 2** | | `achievable_ns` | ratio before → after |
|---|---|---|---|---|---|
| **`(1,1,8192,1024)` BLOCK `[1024,128]`/(8,8)** | 34616 | **25944** | **1.33×** | 25640 | 1.35× → **1.01×** |
| `(1,1,32,1024)` WIDTH `[32,128]`/(8,1) | 4548 | **3576** | 1.27× | 4110 | 1.11× → **0.87×** |
| `(1,1,32,2304)` WIDTH `[32,256]`/(9,1) | 5329 | **4330** | 1.23× | 4617 | 1.15× → **0.94×** |
| `(1,1,32,5120)` WIDTH `[32,160]`/(8,4) | 6484 | **5239** | 1.24× | 5267 | 1.23× → **0.99×** |
| `(1,1,32,7168)` WIDTH `[32,256]`/(7,4) | 6358 | **5080** | 1.25× | 5481 | 1.16× → **0.93×** |
| interleaved prefill 1024 / 2304 / 5120 / 7168 | 87953 / 186816 / 399074 / 568245 | 84612 / 185528 / 396546 / 560376 | 1.01–1.04× | 96744 / 211345 / 738307 / 1032281 | all under |
| interleaved decode 1024 / 2304 / 5120 / 7168 (+floor) | 6164 / 7092 / 9138 / 10215 / 3533 | **5066 / 5967 / 7847 / 8918 / 2776** | 1.14–1.27× | 9149 / 17003 / 75825 / 104259 | all far under |

**All 13 perf-flagged cells are now at or under their `achievable_ns`.** The focus case's 1.012× is
inside the 1.6 % run-to-run spread measured on this harness, i.e. it is at its reference. The
`(1,1,32,7168)` interleaved decode cell, the only one carrying `minimum_expected_speedup = 7.0`
(goal ≤ 14894 ns at 1350 MHz), measures **8918 ns**.

### Guard-set no-regression

One representative per distinct kernel path × layout × placement, all re-run after the last graduation:

- Golden `perf` + `pad_poison`: **37/37 pass**. Golden `loose`: **6/6 pass**.
- Golden `resilience` (the sharded stress sweep): **333 passed / 8 failed / 3 skipped** — the **same 8
  cells** as the Perf-1 baseline (the wide-W `HEIGHT_SHARDED` cells Refinement 2 left failing, plus the
  `13x777x1023` `WIDTH_SHARDED` CB-vs-L1 clash). P5's wider budget was additionally A/B'd by its
  subagent across 29 sharded resilience cells × 3 budgets: **identical outcome set**, and
  `13x777x1023` is untouched because its per-core shard alone is 1.33 MB, so the ladder returns
  `block_rows == 1` at every budget.
- 4-shape full-cartesian slice (`2x1x128x100`, `4x1x512x512`, `1x1x224x3072`, `1x1x160x11008`):
  **296 passed / 78 xfailed / 2 failed** — identical to the Perf-1 record, same two cells.
- Both interleaved perf harnesses green and **faster**, not merely flat.
- **No precision was traded anywhere.** `fp32_dest_acc_en`, `math_fidelity`, `math_approx_mode` and
  every dtype are untouched; none was used as a lever. P4 never flips `APPROX` and calls the identical
  non-approximate `_calculate_sqrt_body_` the wrapper calls; P3's reorder is one bf16 rounding of a
  symmetric product either way and measured 0.999909 → 0.999908; P5 does not change arithmetic at all.
  A narrower stat-CB format was deliberately **not** floated as a lever.

### Helper bypasses

| helper | kind | what was missing / hard | helper ns | raw ns | site |
|---|---|---|---|---|---|
| `rsqrt_tile` | capability | hardcodes `VectorMode::RC` + `ITERATIONS = 8` at its own call site; there is **no scope parameter at all**, so a caller whose DEST tile is axis-shaped cannot say so. Column parity is the INNER walk axis, so even exposing `ITERATIONS` would not reach it — it needs a `dst_reg` stride. | 748 (whole tile) | 204.5 (col-0) | `rms_norm_finalize.hpp:96` (`ttnn/ttnn/operations/rms_norm/kernels/`) |
| `mul_unary_tile` / `add_unary_tile` | capability | same hardcoded `VectorMode::RC` + `ITERATIONS = 8`; **and** there is no fused scalar `mul_add` unary, so `v*a + b` costs two whole separate tile walks where one vector body would do. | 242 (both, RC) | ~36 (inlined into the fused body) | `rms_norm_finalize.hpp:96` |
| `compute_kernel_lib::reduce` — `post_reduce_op` | ergonomics | the extension point hands the caller only a `dst_idx`. There is **no way to declare "this output is column-0-valid"**, so every caller that knows its reduce result is axis-shaped must drop to raw sfpi to exploit it — the knowledge exists at the call site and the API has nowhere to put it. Concrete ask: a `PostReduceScope{Col0, Row0, Scalar}` hint that maps to the right `VectorMode` + parity stride. It would fix softmax's `recip` epilogue too. | 989.7 | 240.1 | `rms_norm_compute.cpp:287` |
| `compute_kernel_lib` — no "SFPU unary on a reduce result" helper | capability | nothing in kernel_lib productizes axis-scoped SFPU work. `examples/sfpu_tile_scope` documents the mechanism and measures it, but every op that wants it re-derives the sfpi body by hand. | — | — | `ttnn/ttnn/operations/examples/sfpu_tile_scope/` |
| sfpi compiler | ergonomics | holding two live scalar constants across an **unrolled** `_calculate_sqrt_body_` is an **ICE** ("cannot store sfpu register (register spill)"), not a diagnostic. Undiscoverable without hitting it; the fix (`#pragma GCC unroll 1`) costs nothing measurable but nothing says so. | — | — | `rms_norm_finalize.hpp:84` |

P3 and P5 bypassed nothing: P3 is the op's existing chains in a different order, and P5 is host-side
plan selection with byte-identical kernels.

**Gaps found by ideas that did NOT graduate**, reported here because this table is their only durable
channel:

| helper | kind | what was missing / hard | helper ns | raw ns | site |
|---|---|---|---|---|---|
| `ckl::reduce` (`ReduceWithinTile::Skip`) | capability | **Still unreachable**, now confirmed by two *independent* ideas this round (P1's contributor collapse, P2's coalesced landing) on top of Perf 1's two. The `static_assert(within_tile == Collapse, "Skip is AccumulateViaAdd-only")` sits at **function** scope AFTER the `if constexpr (AccumulateViaAdd) { …; return; }` block, so it is not in a discarded statement and fires for the very instantiation it means to permit. Moving it into the `else` is the whole fix. | n/a (won't compile) | P1 19164 · P2 26150 | `reduce_helpers_compute.inl:886` |
| `ckl::ReduceInputMemoryLayout` | capability | carries only a `row_stride` (a row *pitch*, asserted `>= cols`), i.e. `addr(r,k) = r*pitch + k`. There is **no column stride**, so a landing block whose reduce axis is the OUTER index (`addr(r,k) = r + k*own_rows`) is inexpressible — and that transpose is exactly what *any* gather wanting contiguous per-contributor landing produces. | 26156 | 26150 (equal) | `reduce_helpers_compute.hpp:260` |
| `ckl::transpose` / `reduce`'s `post_reduce_op` | capability | there is no `ckl::transpose`, and `post_reduce_op` cannot carry `transpose_dest`: its `transpose_dest_init` rewrites MATH addrmods/MOP while the helper calls `reduce_init` **once outside** its per-output loop, so a re-configuring post-op silently corrupts every later output tile. (Perf 1 found this; P1 re-hit it and worked around it with `reduce_init → N×reduce_tile → reduce_uninit → transpose_dest_init → N×transpose_dest → finalize → N×pack`.) Note P4 tested the *narrower* question directly and found **no live bug** for SFPU inits at `BLOCK_ROWS = 16` — `llk_math_eltwise_unary_sfpu_init` touches only the SFPU config reg, ADDR_MOD_6/7 and the RWC counters, none of which the FPU reduce datapath carries across iterations. | n/a | 19164 | `reduce_helpers_compute.hpp:491-495` |
| `ckl::sum_of_squares` / `ckl::eltwise_chain` | capability | a DEST-accumulating walk admits **exactly one** `PackTile` and packs only at row end, so "accumulate per row AND emit a per-tile transformed copy of the same input" is inexpressible. The chain already has the concept (sticky D0 + `transient_lane_width`); what is missing is letting a transient lane reach a second `PackTile` with its own per-tile lifecycle. | n/a (won't compile) | 38634 (and it lost) | `chain.inl:2916, 2923, 3117` |
| `ckl::eltwise_chain` | ergonomics | nothing in the API surfaces that a DEST accumulator **cannot survive a `tile_regs_release`** at `dst_full_sync_en=False` (`_llk_pack_dest_section_done_` zeroes the packed half and flips the dest-offset id). That is the real constraint forcing everything into one window and capping a fused form at `S+1 ≤ DEST_AUTO_LIMIT`, and it is only discoverable by reading `llk_pack_common.h`. One doc line on `DestAccumulation` would have saved a dead end. | — | — | `chain.hpp:294-303` |
| `ckl::sum_of_squares` and every `convenience.hpp` wrapper | ergonomics | the wrappers **default-construct their chain elements**, so they cannot carry a runtime tile base — P6 needed `TileOffset::Set` with a runtime value and had to re-spell the call as the explicit `eltwise_chain` the wrapper expands to. **Third independent hit** (the op already records the same friction for `mask_tail_block` and the in-place scale). The gap costs **0 ns** — output bit-identical — so this is author/maintenance cost, not device time. Ask: a `base` parameter on the convenience wrappers. | 34591 | 34598 (equal) | `block_pipeline/kernels/bp_compute.cpp` |
| kernel_lib dataflow — no gather/scatter helper | capability | pre-existing and re-confirmed by P1 and P6. `mcast_pipe` is one-source-to-a-rectangle; a gather is the mirror shape and has no helper, so the landing, its paging and its arrival counters are all hand-rolled. P6 sharpens the ask: a `GatherPipe` owning *(windows × pages)* of landing plus a **per-window** arrival barrier. | n/a | the whole gather/funnel path | `rms_norm_writer.cpp`, `rms_norm_reader.cpp` |

### Two correctness findings routed out of this round (neither is in the shipped op)

1. **The cumulative gather counter is not pipeline-safe** (found by P6 while building its fork; the
   shipped op is *not* affected because its contributors advance in lockstep). `gather_progress >=
   (block+1)*s` on ONE counter is sound only while nobody runs ahead; let one contributor ship blocks 0
   *and* 1 first and the counter reaches `s` with half the contributors missing. It does **not hang** —
   PCC sags 0.99994 → 0.99912, a silent wrong answer in the neighbourhood of the 0.9995 gate. Any future
   scheme that lets contributors run ahead inherits this and needs one counter per landing window.
2. **A perf A/B harness must upload a fresh `x` per variant.** The op rewrites `x` in place in the
   caller's resident shard, so a harness that uploads once feeds variant N+1 an already-normalized
   tensor. PCC cannot see it (rms_norm is idempotent) but ~5.6 % of bf16 elements shift by up to 1 ULP,
   and it reads exactly like a race.

### Round-3 queue (measured, characterized, not started)

Re-ranked from the zones **after** graduation (focus wall 25987 ns): `rd_gather_wait` **13028**,
`cp_combine` (math) 9778, `wr_gather_issue` **6979**, then `cp_scale` / `cp_apply_gamma` / `cp_sumsq`
≈ 3.8 / 3.8 / 3.6 k each on the unpack thread.

1. **P1's compact stat gather is now the clear #1 lever, and its blocker is gone.** The gather is again
   the top item, and P6 — the idea it lost the overlap to — is no longer competing for the landing
   machinery, because the focus geometry is now `num_blocks == 1`. Measured 1.319× on the stage at the
   focus and 1.7–2.5× at `s ≥ 16`, with `s == 2` the one carve-out. Its subagent also left a follow-on
   it did not take: the collapse reads tile *i* and writes tile *i*, so it can pack **in place** into
   `cb_sq_partials` and delete `cb_stat_compact` entirely, making it strictly L1-cheaper.
2. **P6's pipeline for the L1-constrained deep shards** — `(1,1,8192,2048)` and `(1,1,16384,1024)` still
   land at 2 and 4 blocks because `B=32` needs 555 KB of stat CBs on top of a 1 MB resident shard.
   Measured 1.130–1.301× there, and it is the only lever that reaches them.
3. **The `B`-scaled stat CBs are full fp32 *tiles* holding a per-row scalar** — shrinking them is what
   would unlock one block on the geometries in (2), and it composes with (1), which already ships the
   compact form across the wire.
4. **Do not re-float**: `DEST_BLOCK_TILES 8 → 4` (refuted, 8 is best at every `S`), gather-write
   coalescing (null, the stage is byte-bound), and folding gamma into `sum_of_squares` (0.889×).
