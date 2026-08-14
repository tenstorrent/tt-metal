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
