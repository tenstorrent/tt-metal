# Changelog: rms_norm

## Phase 0 — Core Implementation

- **Date**: 2026-08-04
- **What was done**: Initial implementation via the incremental pipeline
  (planner → implementer → verifier). Row-parallel, multi-core, coarse-blocked scheme with
  a dual-path fits-in-L1 predicate (RESIDENT / STREAM), native TILE **and** ROW_MAJOR
  layouts, native non-tile-aligned H and/or W, optional gamma at an independent
  dtype/layout. Reader (NCRISC/NoC0) + compute + writer (BRISC/NoC1); every compute phase is
  a `compute_kernel_lib` helper; every block / depth / grid knob is a parameter in
  `rms_norm_program_descriptor.py` solved from `L1_SAFETY_FRACTION` and
  `ttnn.get_max_worker_l1_unreserved_size()`.
- **SUPPORTED at Phase 0**: dtype=[float32, bfloat16], fp32_dest_acc_en=[True],
  layout=[TILE, ROW_MAJOR], alignment=[tile_aligned, w_non_aligned, h_non_aligned],
  rank=[2, 3, 4], gamma_mode=[gamma, no_gamma], gamma_dtype=[float32, bfloat16, "none"],
  gamma_layout=[TILE, ROW_MAJOR, "none"], memory_layout=[INTERLEAVED].
  EXCLUSIONS=[{float32, fp32_dest_acc_en=False}].
- **Accuracy achieved** (4 shapes × 2 dtypes × 2 layouts via
  `test_rms_norm_precision_baseline.py`, HiFi4 + fp32_dest_acc_en=True):
  bfloat16 — PCC=0.999997, max_abs_err=0.0452, mean_abs_err=0.00128, rel_rms_err=0.0024,
  ≤2 ULP (bf16 grid);
  float32 — PCC=0.9999997, max_abs_err=0.0246, mean_abs_err=0.00082, rel_rms_err=0.0015.
  Uniform across shape, regime (RESIDENT/STREAM), layout and alignment. got/true ratio
  median 0.9996 (bf16) / 0.9988 (fp32) with a spread wider than the offset ⇒ precision
  noise, **not** a uniform scale error (the fp32 residue pins at exactly 1 − 2⁻¹⁰, an
  SFPU/FPU datapath effect — see `verification_report.md`).
- **Golden suite at Phase 0**: **737 / 737** supported cells passing; 6172 xfail_expected,
  33900 invalid_skipped, 2 infeasible_skipped, 15 non-registry regression tests passing.
  `supported_fail = 0`, `xpass_drift = 0`, `xfail_wrong_mode = 0` (per `verifier_report.json`).
  Runner line: `PASSED=752 FAILED=0 ERRORS=0 SKIPPED=33902 HANGS=0 TOTAL=40828`.
- **Issues encountered**: no drift fixes were needed — SUPPORTED was already honest.
  Six code-review fixes applied by the verifier, all non-behavioural (golden summary
  identical before/after): deduplicated the block-scoped-CB multiplier into
  `_cb_block_mult()` (was written twice, in the RESIDENT predicate and the STREAM solve);
  deduplicated the scaler CB page count into `scaler_pages`; removed the dead `x_resident`
  variable; made the dead `GRID_W` knob live via an explicit `NotImplementedError` guard;
  removed a bare `except Exception` in `_cores_in()`; added the writer's missing
  CT-arg-offset assert. Known deviations left in place and documented: **D3** (the fp32
  reduce runs `ReduceFp32Mode::Fast` because `accumulate_reduce_block<>` does not expose the
  slot) and the prime-`Wt` STREAM chunk-granularity cliff — both carried into
  `op_requirements.md`. `feature_spec.py`'s INVALID list has three mis-categorised
  author-scoped entries and two missing gamma-bf8b entries; relayed in the report, not
  edited.
- **Tests added**: `test_rms_norm.py` (acceptance, immutable — 205 cases),
  `test_rms_norm_perf.py` (10 device-perf probes),
  `test_rms_norm_precision_baseline.py` (16 cases, new this pass — PCC / abs / rel-RMS /
  ULP / got-true ratio spread with a uniform-scale assertion). Unit suite total: 231 passed.

## Refinement 1 — Numerical configurability expansion (dtype + DEST/fidelity surface)

- **Date**: 2026-08-04
- **What was done** (partial — `[~]`):
  - **Reused**: every CB already declared `data_format` = the dtype of the tensor it carries
    and `page_size` = `ttnn.tile_size()` of that same dtype, so `bfloat8_b` needed **no new
    format machinery and no compute-kernel change** — the numeric-formats pass condition held
    (all phases are `compute_kernel_lib` helpers, no hard-coded formats or sizes).
    `cb_row_stat` was already `float32` and stays so in **both** `fp32_dest_acc_en` modes
    (Refinement 1 lever 1: the cross-chunk `Accumulate::at` reload stays lossless even when
    DEST is 16-bit). The compute config is still passed through unmodified, so `False`
    required only widening `SUPPORTED`.
  - **Added**: `bfloat8_b` to `SUPPORTED["dtype"]` and `SUPPORTED["gamma_dtype"]`; `False` to
    `SUPPORTED["fp32_dest_acc_en"]` (`{float32, False}` EXCLUSION intact and now load-bearing).
    `_stick_elem_bytes()` — `Tensor.element_size()` raises `"datum for bfp2, bfp4, bfp8 is
    invalid"` on a block-float dtype; the number is consumed only by the ROW_MAJOR stick byte
    math, which block-float can never reach (no sticks, only exponent blocks), so it reports 0
    there behind an asserted `layout == TILE` (descriptor **D5**).
  - **`unpack_to_dest_mode` left entirely Default, deliberately**: the only fp32 CB is
    `cb_row_stat`, and although its reduce reload (`AccumulateReloadMode::CopySeedPairs`) and
    the `transform_in_place` finalize are both `copy_tile`-into-DEST and would be compatible,
    **pass B consumes it as operand B of an FPU broadcast multiply** — an `UnpackToDestFp32`
    CB may never be an FPU operand, so tagging it would corrupt silently. Tagging
    `cb_input_sticks` is separately forbidden by `tilize`'s `Fp32Mode::Fast` static_assert
    (design R16). Recorded in the descriptor as part of **D5** so it is not "re-discovered".
  - **Fixed a latent pre-existing bug (descriptor D6)** surfaced by the cells this refinement
    unlocked: `cb_row_stat` is now `CB_ROW_STAT_DEPTH (= 2) * BLOCK_ROWS` pages, derived
    through both L1 solves from one constant. `transform_in_place` *rotates* its CB (pop 1 /
    push 1), so with a ring of exactly `BLOCK_ROWS` a **partial** final row-block left the
    finalized stats straddling the ring wrap and pass B's `OperandKind::Col` bulk-indexed read
    ran off the end — the 2nd..last row of every partial block was garbage. Reproduced
    identically at `fp32_dest_acc_en=True`, so **not** an axis bug: every Phase-0 golden cell
    had `Rt ≤ 64 <` the 110-core grid ⇒ `BLOCK_ROWS == 1` ⇒ no partial block could exist.
    This cleared 11 catastrophic cells (PCC 0.55–0.99 → 0.9999+).
  - **Zero new EXCLUSIONS.** `op_design.md` §9.2 predicted `{gamma_dtype: bfloat8_b,
    alignment: *_non_aligned}` would fail; it is clean (PCC 0.99997 / rel RMS 0.008). A bf8b
    gamma's tile padding is **zero**, and a zero never raises a block's shared exponent, so
    the real weights sharing that block are untouched. Pinned by a new test rather than
    excluded, with the reasoning recorded next to `EXCLUSIONS`.
- **Accuracy achieved** (`test_rms_norm_precision_baseline.py`, 269 unit cases):
  `fp32_dest_acc_en=True` — bfloat16 PCC 0.999997 / rel RMS 0.0024, float32 PCC 0.9999997 /
  rel RMS 0.0015 (**Phase 0 baseline held exactly**), bfloat8_b PCC 0.99987 / rel RMS 0.017.
  `fp32_dest_acc_en=False` — bfloat16 PCC 0.99999 / rel RMS 0.0024–0.006 for `W ≤ 4096`,
  bfloat8_b PCC 0.99986 / rel RMS 0.018, mixed bf8b-gamma × bf16/fp32-activation PCC 0.99997.
  Wide `W` at `False` degrades: rel RMS 0.042 @ 5120, 0.050 @ 7168, 0.127 @ 11008 (PCC stays
  0.99993+) — see *Issues*.
- **Golden test progress**: **1660 / 1670** live cells passing (Phase 0: 737 / 737), 5256
  xfail_expected, 33902 invalid_skipped, `xpass_drift = 0`, zero hangs, **no regression** on
  any previously-passing cell. Newly live and green: 240 `bfloat8_b`-activation cells, 250
  `bfloat8_b`-gamma cells, 420 `bfloat16 × fp32_dest_acc_en=False` cartesian cells, plus the
  resilience / pad-poison loose sweeps at `False`.
- **Issues encountered**: 10 cells still fail, all `severity=precision`, all
  `bfloat16 × fp32_dest_acc_en=False × W ≥ 5120`. PCC is 0.99993–0.99996 — **above** the perf
  cases' soft `pcc_threshold = 0.9995` — so what misses is the `rms ≤ 0.04` component of
  `TOLERANCES[bfloat16]` (0.041–0.127). Diagnosed to the datapath rather than guessed:
  DEVICE_PRINT on `cb_row_stat` shows the **reduce output** wrong (7904 vs a true 7033,
  +12.4 %) while the finalize is exact, and the error is **bit-invariant** across
  `NUM_W_CHUNKS` 4 → 112, across `REDUCE_BULK` ∈ {1,0} and across all four `math_fidelity`
  values. So the cross-chunk `Accumulate::at` reload (the verifier's lever 2) is a **measured
  null**, and the real source is the FPU matmul reduce's *within-tile* 32-column sum
  accumulating all-positive addends into a 16-bit DEST — unreachable by any chunking knob.
  Per protocol these precision-near-miss cells are left **failing, not excluded**; they are
  the baseline for the new `Refinement 1b`, which names
  `ReduceAlgorithm::AccumulateViaAdd` as the exact next lever (and notes that
  `accumulate_reduce_block<>` does not yet forward `reduce<>`'s `algorithm` slot — the same
  class of gap as **D3**).
- **Tests added**: `test_rms_norm_precision_baseline.py` extended in place (no second file)
  from 16 to 96 + 40 cases — matrix 1 is now `shape × dtype × layout × fp32_dest_acc_en` with
  `bfloat8_b` added, matrix 2 is the new `test_rms_norm_precision_mixed_gamma_dtype`
  (independent activation-dtype × gamma-dtype, incl. the §9.2 bf8b-gamma × non-aligned corner
  that was predicted to fail). Both share one `_measure()` body so a further axis cannot fork
  the metrics, and `_skip_unsupported()` mirrors `EXCLUSIONS` + `feature_spec.INVALID` so the
  file never asserts on a cell the op may refuse. Unit suite: **269 passed, 30 skipped**.

## Refinement 1b — wide-`W` reduce precision under `fp32_dest_acc_en=False`

- **Date**: 2026-08-04
- **What was done** (`[x]` full): swapped the reduce datapath to
  `ReduceAlgorithm::AccumulateViaAdd` above a measured crossover, which is the exact lever
  Refinement 1 named. Small, shared-path diff — no new kernel, no second program-descriptor
  branch, no forked compute phase.
  - **Reused**: the same `ckl::accumulate_reduce_block` call in the same pass-A loop, the same
    `cb_x_squared → cb_row_stat` accumulator, the same `transform_in_place` finalize and pass
    B, the same `cb_scaler` slot, the same reader boot block. The whole change is two
    forwarded template args, one `constexpr` datapath selector, one reader `if constexpr`,
    and one host-derived tile count.
  - **Added — helper wrapper (retires deviation D3)**: `accumulate_reduce_block<>` and
    `accumulate_reduce<>` in `streaming_reduce_helpers.hpp/.inl` now forward reduce()'s
    `ReduceFp32Mode` *and* `ReduceAlgorithm` slots (the gap the refinement predicted), and
    route the last block through `Accumulate::at_last` instead of `at` so AccumulateViaAdd's
    within-tile finalize runs exactly once. Byte-identical for ReduceTile, which ignores the
    `last` flag — `toy_variance`, the only other caller, passes template args only up to
    `cb_acc`, so it is unaffected (it is separately, pre-existingly broken against the current
    `eltwise_convenience` API — verified identical before/after this change).
  - **Added — descriptor D7 + the crossover knob**: `REDUCE_ACC_VIA_ADD_MIN_WT = 4` selects
    AccumulateViaAdd once `WT_CHUNK >= 4`, ReduceTile below. Not unconditional, because the
    datapath is a *loss* at 1–2 reduce-dim tiles (0.67× / 0.94×) and narrow rows have no
    precision problem to fix. One source of truth: `reduce_acc_via_add` and `scaler_tiles`
    derive from it and feed the CB sizing, both kernels' CT args and the compute's final pop.
    The knob is also coupled to `REDUCE_BULK == 1` in one place (AccumulateViaAdd + cross-chunk
    Accumulate is `BulkWaitBulkPop`-only), with a compute-side `static_assert` as the backstop.
  - **Added — the coupled partial-`W` swap**: AccumulateViaAdd takes a 0/1 **mask** tile
    (`prepare_reduce_mask` + `ReducePartialScaler::partial_mask`) where ReduceTile takes the
    `[full, partial]` **scaler** pair. Both zero the pad lanes with an exact multiply-by-0, so
    the reader's one-time pad-lane zeroing invariant is unchanged.
  - **WHY it works where chunking could not**: ReduceTile's FPU matmul-with-ones drives
    `WT_CHUNK*32` all-positive addends through ONE DEST word (16-bit at
    `fp32_dest_acc_en=False`) — Refinement 1's bit-invariant +12.4 % error. AccumulateViaAdd
    sums the width tiles *elementwise* with pairwise `add_tiles` and finishes the 32-column sum
    on the SFPU in fp32 LREGs, cutting DEST-resident accumulation depth from `WT_CHUNK*32`
    serial adds to `WT_CHUNK/2` pairwise ones, with the cross-chunk carry still through the
    fp32 `cb_row_stat`.
- **Accuracy achieved**: on the 7 target shapes × 2 gamma layouts at the `_perf_case` config
  (bfloat16 / TILE / HiFi2 / `fp32_dest_acc_en=False`), rel RMS fell from **0.042–0.127 to
  0.0089–0.0109** (gate `rms <= 0.04`), PCC **0.99988–1.0000** (gate 0.995). Narrow-`W`
  control unchanged at rel RMS 0.0065. Pad-poison: rel RMS 0.0056–0.0094, median got/true
  ratio within 0.7 % of 1.0 on both partial mechanisms. Prior baselines held (unit suite green).
- **Golden test progress**: **1670 / 1670** live cells passing (Refinement 1: 1660 / 1670),
  5256 xfail_expected, 33902 invalid_skipped, `xpass_drift = 0`, zero hangs, **no regression**
  on any previously-passing cell. All 10 `severity=precision` failures closed. Unit suite:
  **298 passed, 30 skipped** (also green under `--dev` with LLK asserts on).
- **Issues encountered**: **a latent kernel-library bug the lever exposed.**
  `fold_partial_last` in `reduce_helpers_compute.inl` never reconfigured **SrcB** to the mask
  CB — the code around it leaves SrcB pointing at the *input* CB, and `llk_unpack_AB_init` only
  asserts formats rather than setting them. Latent for any caller whose input format differs
  from its scaler/mask format; it surfaced here as 2 regressed acceptance cells (`float32` +
  non-aligned `W` + `Wt >= 4`), diagnosed exactly by the `--dev` LLK assert
  (`unp_B_src_format mismatch`, actual Float32 vs expected Float16_b) rather than guessed, and
  fixed by bracketing the fold with `reconfig_data_format_srcb` — the same shape as the
  adjacent FoldViaAdd reconfig. The `reduce_block` example's 19 tests (which cover both
  `fold_partial_last` paths) stay green.
  Second finding, recorded for the perf phases: **Refinement 4's item (a) 2.87–2.94× does not
  translate to this op.** Measured whole-op A/B by flipping the knob: `(1,1,32,7168)` 1.06×,
  `(1,1,224,3072)` 1.05×, `(1,1,32,1024)` 1.02×, `(1,1,8192,5120)` 1.00× — a small uniform
  win, no shape slower. rms_norm is dataflow-bound at these widths, so reduce-MATH cycles are
  not the bottleneck; the numbers are in descriptor D7 so a later phase does not re-budget
  against the micro-benchmark.
- **Tests added**: `test_rms_norm_wide_w_precision.py` (25 cases, new — the 7 wide shapes ×
  2 gamma layouts at the perf-case config, a narrow-`W` control, and 8 pad-poison shapes
  spanning both partial mechanisms with a got/true ratio assertion, since a padding leak on a
  wide row is a near-uniform scale error PCC is largely blind to).
  `test_rms_norm_perf.py` extended in place with `test_rms_norm_perf_reduce_datapath`
  (4 shapes at the `_perf_case` config) so the D7 A/B is re-measurable.

## Refinement 2 — Sharded placements: local HEIGHT shard + cross-core WIDTH/BLOCK combine

- Date: 2026-08-04
- What was done:
  - **`SUPPORTED["memory_layout"]` grew all three sharded values.** A placement PLANNER
    (`_plan_placement`) resolves the axis into three internal schemes, each a pure function
    of (layout, placement, shard geometry, alignment, L1 budget):
    `SCHEME_ROWS` (Phase 0's row split + TensorAccessor — also the universal L1 bail-out),
    `SCHEME_SHARD_H` (Lamp L3 knob-turn: rows come from the shard, reduce stays local),
    `SCHEME_SHARD_W` (Lamp L4 scheme-change: width comes from the shard, cross-core combine).
  - **Native, not tolerated.** For TILE activations `cb_input_tiles` / `cb_output_tiles` are
    `ttnn.cb_descriptor_from_sharded_tensor` — zero-copy over the tensor's own resident L1,
    so there is **no NoC read for x at all** and no arena allocation for either CB. The
    reader only PUBLISHES the shard's pages once (`cb_reserve_back` + `cb_push_back`); the
    writer takes a completion barrier and moves nothing.
  - **The §3.4 cross-core combine, built as specified**, and placed in the WRITER (NoC1 is
    idle through pass A, so the handshake overlaps the reader's NoC0 x/gamma traffic, and
    `cb_row_stat` stays compute-private): every core packs its raw partial into a DEDICATED
    `cb_sum_handoff`, `noc_async_write`s it into its slot of the root's
    `cb_partials_gathered` and remote-incs the root's arrival semaphore; the root sums the
    group's partials elementwise (`ckl::copy` + `ckl::add` in place, per-tile streaming, so
    no slot needs to be contiguous), runs the *same* `transform_in_place` finalize, and
    multicasts the stat back with `mcast_pipe.hpp`. `GRID_H`/`GRID_W` and every
    `(w_start, w_count)` are read off the shard spec.
  - **`GRID_W`'s `NotImplementedError` guard deleted.** The knob now drives the same combine
    on an INTERLEAVED input (Lamp L1 — the occupancy lever Refinement 3 needs on `Rt = 1`),
    clamped to the largest divisor of `Wt` that fits the grid so every core owns the same
    width. Parked at its byte-identical 1.
  - **A ragged width tail is handled, not refused**: the last core of a group owns fewer real
    width tiles than the shard is wide, so the reader zeroes its PAD tiles once at boot
    (they then contribute exactly 0 to `sum(x^2)`), the gamma push is topped up to
    `WT_CHUNK`, and the writer skips `wt >= Wt`. Exercised by `(1,1,32,7168)`/75 cores (2-tile
    tail), `(1,1,32,4064)`/64 cores (1-tile tail) and `(1,1,8192,1024)`/110 cores.
  - New knob `L1_CB_ARENA_BASE_RESERVE`; new descriptor deviation **D8** (the reduce-datapath
    crossover is measured against the core's whole reduce dim, not `WT_CHUNK`).
- Accuracy achieved: PCC 0.99997–1.000000, rel RMS 0.0023–0.0126 across
  HEIGHT/WIDTH/BLOCK on `(1,1,256,512)`, `(1,1,224,72)`, `(1,1,32,40)`, `(1,1,17,50)`,
  `(1,1,3232,96)`, `(1,1,32,2048)`, `(1,1,32,4064)`, `(1,1,32,7168)`, `(1,1,8192,1024)`,
  `(1,224,11008)`, `(1,1,96,6144)`. The golden gate is `TOLERANCES[bfloat16]`
  (pcc 0.995 / rms 0.04) and the sharded perf cases' soft `pcc_threshold = 0.9995`.
- Golden test progress: **387 / 387 loose cases** (299 pass, 3 infeasible-skipped,
  85 xfail) and the full **40320-cell cartesian** (4407 pass, 1995 xfail,
  `supported_fail = 0`). Zero hangs. No regression: unit dir 298 passed / 30 skipped.
- Issues encountered (all diagnosed to a root cause, none guessed):
  1. **Inactive cores touched shard-backed CBs.** A WIDTH shard grid is row-major-packed, not
     a rectangle, so the multicast runs over its bounding box and the in-box/out-of-shard
     cores must join the program (their `cb_row_final` is where the stat lands) — but they
     hold no shard. `--dev`'s watcher caught the read to `L1[0x000000]`; all three kernels
     now early-out on `num_rows == 0`.
  2. **`Mcast1D`'s per-row sender rect EXCLUDES the sender** while `Mcast2D`'s contains it, so
     the BLOCK root never got its own finalized stat (PCC 0.91). The root now places its own
     copy and broadcasts IN PLACE (`src == dst` ⇒ EXCLUDE-source) — one behaviour for both
     emitters.
  3. **`Semaphore::up(value)` is a NON-ATOMIC local read-modify-write** (the header says so).
     The root's self-signal raced the members' remote atomic incs and dropped one — a hang in
     one group of eight. Triage was unambiguous: exactly one grid row stuck, root at
     `wait_min`, all seven members already parked in `receive()`. The root writes its own slot
     synchronously, so it waits for `GROUP_SIZE - 1` and never bumps the counter itself.
  4. **`L1_SAFETY_FRACTION` is proportional and cannot cover a fixed offset.** The CB arena
     begins 70656 B above the worker-L1 unreserved base, and metal's check is absolute. Four
     loose cases failed to launch; the reserve is now subtracted whenever a shard is
     L1-resident (and only then, so interleaved builds are byte-identical).
  5. **A resident shard silently switched Refinement 1b's precision fix off** (D8): a 344-tile
     shard squeezed `WT_CHUNK` to 2, below the `AccumulateViaAdd` crossover, and rms 0.127
     came back on exactly the cells 1b closed. Gating on the whole reduce dim fixes it.
  6. **`ROW_MAJOR` + WIDTH/BLOCK shards are a STRUCTURAL gap → `EXCLUSIONS` + Refinement 2b.**
     An RM shard edge rounds to (1 stick × `L1_align/elem_size` elements) and a shard may not
     hold a partial page, so the tensor's PAGE becomes the shard's row SEGMENT (8 or 32
     elements): no core holds a whole width tile, and a stick read keyed on the page index
     lands inside one segment and runs off the end of the shard (PCC 0.005 plus
     out-of-bounds L1 traffic, which is what was cascading later tests into dispatch
     failures). HEIGHT_SHARDED is unaffected — its shard spans the full row, so the page IS
     the stick (PCC 1.000000).
  7. Not an issue, recorded so it is not re-chased:
     `test_translated.py::test_rms_norm_sharded_uneven_multicore_logical_width[
     w200_c3_nonaligned-bfloat8_b]` misses at frobenius 0.112 vs a 0.10 budget, and is
     **bit-identical INTERLEAVED vs WIDTH_SHARDED (0.11224 both)** — the
     `{bfloat8_b, w_non_aligned}` corner `feature_spec.INVALID` already declares out of scope
     (a 1000.0 pad poison raises the shared exponent of the block straddling the logical
     width). The other 11 params of that test went from all-failing to passing.
- Tests added: `tests/ttnn/unit_tests/operations/rms_norm/test_rms_norm_sharded.py` — the
  three schemes × both layouts × the ragged-tail geometries, the pad-poison shapes under
  every placement, a native-dataflow assertion (the plan really is zero-copy, checked
  against the descriptor rather than against output values), and a `GRID_W > 1` case that
  exercises the interleaved width split so the knob is live rather than dead code.

## Refinement 2b — ROW_MAJOR shards that cut the width axis
- Date: 2026-08-04
- What was done: closed the two `EXCLUSIONS` Refinement 2 added
  (`{ROW_MAJOR, WIDTH_SHARDED}` and `{ROW_MAJOR, BLOCK_SHARDED}`) with the **BAND scheme**
  (`_plan_band`, descriptor deviation **D10**). The op's `EXCLUSIONS` list is back to its
  single `{float32, fp32_dest_acc_en: False}` entry — **zero new exclusions** — so the
  layout × placement rectangle is now complete and native throughout.
  - **The insight both listed levers missed: nothing has to reach a whole ROW.** An RM shard
    edge rounds to (1 stick × `L1_align/elem_size` elements) — 8 for bf16, 4 for fp32 — so the
    tensor's page is a row SEGMENT and no core holds a whole width TILE. Refinement 2 read
    that as structural. But the §3.4 combine sums the group's per-row partials
    **elementwise**, so a partial may cover *any* contiguous element range: `Σx²` over a row is
    the sum over the bands however the bands are cut, and pass B scales, gamma-multiplies and
    writes back entirely inside the band. So each core reduces the band it **already holds**,
    staged out of its **own L1** (`x_addr + local_stick * shard_row_bytes`) — no DRAM traffic
    for x or the output, and no accessor on a local shard.
  - Listed lever 2 ("native band tilize when `shard_w % 32 == 0`") survives as the
    **contiguous fast path**: when the band fills its tile columns and the shard stride
    matches, a whole tile-row of 32 sticks moves in ONE local transaction instead of 32. It is
    no longer the precondition for the scheme, which matters because `per_w % 4 == 0` is rare
    (3 of the 44 resilience shapes). Listed lever 1's `ceil(W / shard_w)` reads per stick
    (96 for `(1,1,224,3072)`; 8 × 100 000 for `(99991,64)`) are **never paid**, so its `nw`
    ceiling and its above-the-ceiling refusal are both unnecessary.
  - **Reuse**: the whole cross-core combine (`cb_sum_handoff` → gather → root finalize →
    `mcast_pipe` stat broadcast), the RM `tilize`/`untilize` compute path, and every block /
    depth / datapath knob are UNCHANGED — the compute kernel needed no edit at all. Added:
    `_plan_band`, the reader's `stage_band`, the writer's `write_band`, and a `_Work` record
    that carries the tile-axis and stick/element views of one core's slice together (derived
    from each other, so they cannot drift).
  - The band is staged in the tensor's **GLOBAL tile frame** (first element at lane
    `w_off_elems % 32`), which is what keeps gamma fetchable — and therefore keeps **TILE
    gamma working**, so no exclusion was needed for it.
  - `WT_CHUNK` is the widest global tile span any core's band touches (it is a compile-time
    template on tilize/untilize); a core spanning fewer tiles stages an all-zero pad column.
  - New named CT-arg indices `READER_CT_BAND` / `WRITER_CT_OUT_SHARD_ROW_BYTES` (asserted
    against the built arg lists) so the acceptance test can name an index without a magic
    number.
- Accuracy achieved: PCC **1.000000** / rel RMS **0.0017–0.0025** across
  `(1,1,{64,256},512)`, `(1,1,224,3072)`, `(1,1,224,1000)`, `(1,1,32,50)`, `(4,8,32,47)`,
  `(1,1,3232,96)`, `(7136,736)`, `(1,224,11008)` × {WIDTH, BLOCK}_SHARDED × {RM gamma, TILE
  gamma, no gamma} × {matching, DRAM, L1-interleaved} output placements. Gate is
  `TOLERANCES[bfloat16]` (pcc 0.995 / rms 0.04).
- Golden test progress: **387 / 387 loose cases** (384 pass, 3 infeasible-skipped, **0 xfail**
  — the 85 that were this refinement's cells all pass) and the full **40320-cell cartesian**
  (**5037 pass** vs Refinement 2's 4407, 1365 xfail vs 1995, `supported_fail = 0`,
  `xpass_drift = 0`). Zero hangs, clean under `--dev`'s watcher. No regression: unit dir
  **406 passed / 30 skipped** (was 397 / 30 before this refinement's own cases were added).
  The one remaining golden failure is the pre-existing `test_translated.py::
  test_rms_norm_sharded_uneven_multicore_logical_width[w200_c3_nonaligned-bfloat8_b]`,
  bit-identical at **frobenius 0.11224** to the value Refinement 2 recorded as a known
  non-issue (the `{bfloat8_b, w_non_aligned}` corner `feature_spec.INVALID` declares out of
  scope).
- Issues encountered (all three diagnosed to a root cause, none guessed):
  1. **An unaligned DRAM read offset is SILENTLY TRUNCATED to the 64-byte alignment.** Staging
     each band at its own byte offset meant gamma was fetched at `w_off * elem_bytes`; with an
     8-element shard, bands 1, 2 and 3 all received `gamma[0..8)`. The tell was that a
     positional gamma (`gamma[w] = w+1`) printed `1,2,…,8,1,2,…,8,1,2,…,8,33,34,…` — a period
     of exactly 64 bytes. PCC 0.32 overall while band 0 and every spot-checked ratio read
     1.000, which is why a scalar check would have missed it. Fixed by the global tile frame:
     every gamma fetch lands on a tile column, a multiple of 64 bytes for every dtype.
  2. **A latent deadlock in Refinement 2's writer.** Its combine loop ran *all* row-blocks
     before *any* output write. But compute cannot finish block `blk+1`'s pass A until block
     `blk`'s pass B has been drained, so that ordering deadlocks as soon as `num_blocks`
     exceeds the output CB's depth. The TILE schemes never hit it — their shard fits in ONE
     row-block — and the band scheme hit it immediately, because its per-block gather CB is
     `GROUP_SIZE` fp32 tiles and L1 therefore caps `BLOCK_ROWS` low. Triage was unambiguous:
     TRISC2 stuck in `untilize`'s `reserve_back` on `cb_output_sticks` while BRISC sat in the
     combine's `cb_wait_front(cb_sum_handoff)`. The write-back is a per-block lambda now,
     called from inside both combine branches.
  3. **The reduce mask does not generalize to a band — it is replaced.** A band boundary is
     per-core and cannot be one program-wide `PARTIAL_W`. It does not need to be: the staging
     ring is zeroed once at boot (R3) and only the band's own bytes are ever written into it,
     so every lane outside `[delta, delta + band)` multiplies to an exact 0. `kernel_partial_w`
     is 0 on this path and the reader's boot-zero gate became an explicit `STAGE_ZERO`
     ("some staged stick is narrower than the ring's padded row") that both cases share.
- Perf measured for the record (not a gate; blackhole p150b, 110-core grid, ~1.35 GHz,
  bf16/HiFi2/`fp32_dest_acc_en=False`, one fresh-cache profiled run per variant): against the
  **equivalent TILE shard at the same placement**, the band costs **+2.7 %** on
  `(1,1,224,3072)` WIDTH (133224 → 136856 ns, 96-core group) and **+21 %** on the same shape
  BLOCK (10373 → 12509 ns, 11-core group — and still **2.7× faster than interleaved**'s
  33434 ns). `(1,1,256,512)` WIDTH is 29004 → 96479 ns, and that gap is **not** the band: an
  RM granule of 8 elements makes `auto_shard_config` cut W=512 into 64 slices where the TILE
  granule cuts it into 16, so the same tensor gets a 4× larger combine *group* — a placement
  cost the caller chose, and one the TILE path pays identically at equal group size (the WIDTH
  row, where both are 96 cores, agrees to 3 %). The lever a later perf round would attack is
  the sub-tile band's one-local-read-per-stick staging. No supported shape got slower: the D7
  datapath probes re-measured at 41770 / 22447 / 11160 / 746859 ns vs 42253 / 22544 / 10881 /
  752410 recorded.
- Tests added: `test_rms_norm_sharded.py` gained `test_row_major_width_band` (8 band
  geometries × WIDTH/BLOCK — sub-tile bands whose in-tile offset cycles, a band that is
  exactly one tile column, short last bands, non-aligned W, a 92-core band group, many row
  blocks), `test_row_major_width_band_gamma_layouts` (the 64-byte-alignment regression net,
  which needs a RANDOM gamma to be visible), `test_row_major_width_band_is_local` (asserts on
  the DESCRIPTOR that the reader is on the BAND path and the writer writes into the resident
  output shard — the golden suite cannot see this) and
  `test_row_major_width_band_non_matching_output` (a band input with a DRAM / L1-interleaved
  output, which nothing else covers since the golden harness always requests a matching output
  shard). `test_sharded_row_major` now sweeps all four placements. `test_rms_norm_perf.py`
  gained `test_rms_norm_perf_row_major_band` (both band granularities plus their interleaved
  controls) so the trailing perf rounds can re-measure this dataflow. The refusal test
  `test_row_major_width_shard_refused` is gone — its subject is now supported.

## Refinement 3 — Speed up the wide/decode profiles (post-combine)
- Date: 2026-08-04
- What was done (`[x]` full, perf — no SUPPORTED change): turned **GRID_W**, the interleaved
  cross-core width split (Lamp L1) that Refinement 2 built and parked at its byte-identical
  1, into an **AUTO policy** (descriptor deviation **D11**). The row split can only ever use
  `min(Rt, num_cores)` cores, so an `Rt = 1` decode profile ran an arbitrarily wide tensor
  through ONE core (measured 41803 ns on `(1,1,32,7168)` = 1.34 MB at ~32 GB/s, which is one
  core's NoC and nothing else). Small, shared-path diff — no new kernel, no kernel edit at
  all, no second program-descriptor branch.
  - **Reused**: the whole §3.4 combine (`cb_sum_handoff` → per-sender slot of the root's
    `cb_partials_gathered` + `noc_semaphore_inc` → root sums elementwise + runs the *same*
    `transform_in_place` finalize → `mcast_pipe` stat broadcast), both `mcast_host.hpp`
    emitters, the ragged-free uniform-width contract, every block/depth/datapath knob, and
    all three kernels **unchanged** (`git diff` touches no `kernels/*.cpp`).
  - **Added**: `_auto_width_split()` (the policy), `_resolve_width_split()` (knob → decision),
    `_width_group_cores()` (the divisor clamp), a **PACKED single-group topology** for a group
    wider than one grid row (`Mcast2D` over its bounding box with in-box/out-of-group cores
    joining INACTIVE — the same shape a row-major-packed WIDTH shard grid already had), and
    three policy knobs, each measured: `WIDTH_SPLIT_MIN_WT_PER_CORE = 4`,
    `WIDTH_SPLIT_MAX_GROUP_CORES = 16`, `WIDTH_SPLIT_MIN_GAIN = 4`. `GRID_W` keeps its
    override meaning (0 = AUTO, 1 = off, ≥2 = forced) and is the A/B handle the perf sweep
    measures with. `_plan_rows` gained `allow_width_split` so the L1 bail-out
    (`force_rows=True`) and the two shard-geometry fallbacks stay byte-identical.
- Perf achieved (blackhole p150b, 110-core 11×10 grid, CHIP_FREQ 1350 MHz == the reference
  clock, bf16 / TILE / INTERLEAVED / `fp32_dest_acc_en=False` / HiFi2 — the `_perf_case`
  config; one fresh-cache profiled run per variant, whole sweep reproduced twice within 2 %):
  **`(1,1,32,7168)` 41803 → 12756 ns (3.28×)** = **8.17× the 104259 ns reference**, above the
  required **7.0×** and inside the **≤ 14894 ns** goal; **`(1,1,32,1024)` 11196 → 7199 ns
  (1.56×)**, 1.27× under its 9149 ns reference. Also `(1,1,32,8192)` 3.46×,
  `(1,1,32,5120)` 2.92×, `(1,1,32,4096)` 2.59×, `(1,1,32,2304)` 2.43×,
  `(1,1,128,4096)` 1.72×, `(1,1,224,3072)` 1.29×, `(1,1,224,1000)` 1.29×,
  `(1,1,512,4096)` 1.20×. Byte-identical (no split, 1.00×): `(1024,1024)`,
  `(1,1,2048,256)`, `(1,1,8192,1024)` and every other grid-filling prefill.
- Accuracy achieved: soft `pcc_threshold = 0.9995` holds on both targets —
  `(1,1,32,7168)` PCC 0.999980 / rel RMS 0.0087, `(1,1,32,1024)` PCC 0.999984 / rel RMS
  0.0069 (gate `rms ≤ 0.04`). Precision is safe **by construction** and that is deliberate:
  `gw ≤ Wt // WIDTH_SPLIT_MIN_WT_PER_CORE` ⇒ `wt_per_core ≥ 4 ==
  REDUCE_ACC_VIA_ADD_MIN_WT`, so a split build can never switch Refinement 1b's precision
  fix off the way a resident shard did in Refinement 2 (**D8**'s trap) — and the cross-core
  sum is itself an fp32 elementwise add, so a split row accumulates LESS DEST-resident depth
  than an unsplit one.
- Golden test progress: **5421 pass / 1365 xfail / 33921 invalid-skipped / 0 fail** —
  numerically identical to Refinement 2b (5037 cartesian + 384 loose), as a perf refinement
  should be. `test_regression.py` 15/15. `test_translated.py` 105/106, the one failure
  bit-identical at frobenius **0.112240** to the pre-existing `{bfloat8_b, w_non_aligned}`
  pad-poison cell `feature_spec.INVALID` declares out of scope (recorded as a known
  non-issue in Refinements 2 and 2b). Zero hangs. No regression: unit dir **434 passed /
  30 skipped** (was 406 / 30 before this refinement's 28 new params).
- Issues encountered:
  1. **A gain threshold was REQUIRED, not cosmetic.** At the first `WIDTH_SPLIT_MIN_GAIN = 2`,
     `(1024,1024)` (Rt = 32) split 32 → 80 cores and got **slower**: 21560 → 23315 ns
     (0.92×). 2.5× more cores cannot pay for one combine round when the row split already
     feeds 32 cores. Raised to 4; that shape now stays on the untouched Phase-0 path and
     every shape that does split measured ≥ 1.20×.
  2. **The group size has a measured optimum**, because two terms oppose: per-core bytes fall
     as `1/gw` while the root's gather rises with `gw` (each member ships a full fp32 TILE per
     row-block into ONE root). `(1,1,32,7168)`: gw = 1 → 41803, 8 → 13876, **16 → 12978**,
     32 → 14487, 56 → 19428 ns. Hence the ceiling knob at 16 rather than "fill the grid".
  3. **A ragged width tail is NOT available on the interleaved path**, unlike the sharded
     ones: a resident shard has pad-tile storage the reader zeroes once (NATIVE_X), an
     interleaved core has none, so a ragged core would have to read x tiles it does not own.
     `gw` is therefore clamped to a DIVISOR of `Wt` — the same D1 granularity limit — which
     means a prime `Wt` (e.g. `(1,1,32,4064)`, Wt = 127) does not split. Recorded, not
     worked around.
- Remaining headroom (a FINDING, not a queued task — see D11): a one-core minimal program is
  **3348 ns** of fixed launch/dispatch floor, and at gw = 16 the 7168 case moves only 56 kB
  per core (≈1.8 µs at the measured 32 GB/s single-core NoC), so ~7 µs of its 12756 ns is the
  gather → root sum → stat-multicast round trip, which overlaps nothing because `Rt = 1`
  gives a core a single row-block. The two levers that follow are (a) a hierarchical
  two-stage gather (`examples/tensix_all_reduce` measures 1.45–1.60× over a flat root on 2-D
  groups, and it would raise the useful group ceiling so more cores share the payload) and
  (b) a compact partial handoff — a `REDUCE_ROW` partial is a 32-float column vector shipped
  inside a 4096-byte tile, so the gather moves 128× the bytes it needs. Both change the
  combine's topology / data format rather than turning a knob, so neither was half-built here.
- Tests added: `test_rms_norm_perf.py::test_rms_norm_perf_width_split` (14 params — the
  group-size crossover on both target shapes, the AUTO policy on both plus the two regimes
  that must NOT split, and a one-core minimal program that pins the fixed floor every number
  is read against). `test_rms_norm_sharded.py::test_interleaved_width_split_auto_policy`
  (8 cases — asserts on the DESCRIPTOR that the split engages exactly where the policy says
  it should, since the golden suite cannot see occupancy: a decode shape computes the same
  numbers on 1 core as on 16, just 3× slower) and `test_interleaved_width_split_knob` widened
  to `GRID_W ∈ {0, 1, 2, 8, 16, 56}` so both topologies stay covered (`gw56` is wider than the
  11-core grid row, hence the PACKED `Mcast2D` single group).

## Refinement 4 — Prefill + sharded-geometry perf, and the block/depth knob surface
- Date: 2026-08-04
- What was done: a **perf** refinement (no SUPPORTED change). Four levers landed, each
  measured on device separately and each left in the tree as a live knob:
  1. **Lamp L5 — the op's third compute regime, ROW_RESIDENT** (descriptor **D14**). `X_RESIDENT`
     is now an EXPLICIT CT flag instead of `NUM_W_CHUNKS == 1`, and that decoupling *is* the
     regime: one whole tile-row of x plus the whole row of gamma stay resident while only the
     DERIVED CBs are chunked, so **pass B re-reads nothing**. No second code path — every
     helper call still works on one `WT_CHUNK` and simply indexes the two held CBs at a
     **tile offset** (`TileOffset::Set`, base `c * WT_CHUNK`, folded away to 0 off this path),
     with one explicit `cb_pop_front` per row-block. New knob
     `ROW_RESIDENT_MIN_ROWS_PER_CORE`; `x_hold_wt` is the single source of truth for the held
     CBs' width. **This is the prefill lever.**
  2. **Lamp L6b — `rsqrt` scoped to `VectorMode::C`** (**D15**, knob `RSQRT_COL_SCOPE`). The
     one raw-LLK addition (`rsqrt_tile` hard-codes `VectorMode::RC` and exposes no seam;
     justification at the function, precedent `sdpa/.../compute_common.hpp:251`).
     **This is the sharded-geometry lever.**
  3. **Lamp L6d — DEST-fold pass A's square** (**D12**, knob `DEST_ACC_SQUARE_MAX_WT = 8`).
     `DestAccumulation::PerRow` folds the chunk's width tiles into DEST, so `cb_x_squared`
     holds one tile per tile-row and the reduce's per-call width is 1. A *ceiling*, not a
     floor: the fold accumulates serially where Refinement 1b's `AccumulateViaAdd` accumulates
     pairwise, so bounding `WT_CHUNK` bounds the DEST depth. Gated on `PARTIAL_W == 0`.
  4. **A compact combine gather** (**D13**, knob `GATHER_FACES = 2`). A `REDUCE_ROW` partial
     does not fill its tile; only faces 0 and 2 are ever read back, so the member→root gather
     — the combine's only per-`GROUP_SIZE` term — ships half the bytes.
  Also re-measured the whole **block/depth knob surface** the refinement named
  (`CB_DEPTH_CANDIDATES`, `L1_SAFETY_FRACTION`, `CB_RM_STAGE_DEPTH`): all three are **nulls at
  their shipped values** (below).
- Accuracy achieved: no precision change anywhere. ROW_RESIDENT measured at bf16 /
  `fp32_dest_acc_en=False` (gate `rms ≤ 0.04`): rel RMS **0.0096** on `(1,1,8192,5120)`,
  **0.0105** on `(1,1,8192,7168)`, **0.0093** on `(1,1,64,5120)` ROW_MAJOR, **0.0113** on
  `(1,1,64,8192)` ROW_MAJOR — i.e. unchanged from Refinement 1b's numbers. The pad-poison and
  precision-baseline suites are green, which is what covers the two levers that could have
  moved precision (the DEST fold, and the fold's interaction with the partial-W mask).
- Golden test progress: **5421 pass / 1365 xfail / 0 fail** on the 40320-cell cartesian —
  byte-identical counts to Refinement 3, `supported_fail = 0`, `xpass_drift = 0`.
  `test_regression.py` 15/15, `test_translated.py` 105/106 with the one failure bit-identical
  at frobenius **0.112240** to the pre-existing bf8b pad-poison non-issue. Unit dir
  **463 passed / 30 skipped**, zero hangs.
- Measured perf (blackhole p150b, 110-core 11×10 grid, CHIP_FREQ 1350 MHz == the reference
  clock so no scaling needed; bf16 / TILE / HiFi2 / `fp32_dest_acc_en=False` — the declared
  `_perf_case` config; ONE fresh-cache profiled run per variant, repeated 3× and medianed only
  where a number sat on the noise band):

  | target | `achievable_ns` | before | after | speedup |
  |---|---|---|---|---|
  | `(1,1,8192,1024)` interleaved | 96744 | 105343 | 103693 | 1.02× |
  | `(1,1,8192,2304)` interleaved | 211345 | 215681 | 221472 | 1.00× (noise) |
  | **`(1,1,8192,5120)` interleaved** | 738307 | 753345 | **468093** | **1.61×** |
  | **`(1,1,8192,7168)` interleaved** | 1032281 | 1043918 | **643320** | **1.62×** |
  | `(1,1,32,1024)` WIDTH 8c | 4110 | 6055 | 5464 | 1.11× |
  | `(1,1,32,2304)` WIDTH 9c | 4617 | 6942 | 6456 | 1.08× |
  | `(1,1,32,5120)` WIDTH 32c | 5267 | 11911 | 11094 | 1.07× |
  | `(1,1,32,7168)` WIDTH 28c | 5481 | 11541 | 10502 | 1.10× |
  | **`(1,1,8192,1024)` BLOCK 64c** | 25640 | 102173 | **83316** | **1.23×** |

  Per-lever attribution: L5 is the whole prefill win (5120 753345→468487, 7168
  1043918→655687); L6b is most of the sharded win (BLOCK 94297→82474 = 1.14×, WIDTH
  1.03–1.07×); L6d and the compact gather contribute 1.02–1.03× each. No supported shape is
  slower: the config-spanning guard set is at or better than its recorded baseline everywhere
  — TILE/interleaved RESIDENT 103693 (was 105343), width-split AUTO `(1,1,32,7168)` 12467
  (D11: 12876), `(1024,1024)` 20817 (D11: 20960), one-core floor 2993 (D11: 3456),
  ROW_MAJOR `(1,1,32,4096)` 47016 (**1.11×**, L5 wins there too), RM BAND WIDTH
  `(1,1,224,3072)` 128113 (D10: 136856), RM BAND `(1,1,256,512)` 93965 (D10: 96479), BLOCK
  band 11936 (D10: 12509).
- Issues encountered: three, all diagnosed rather than guessed.
  * **The byte model for the combine over-predicted, and that mis-aimed the first lever.**
    Halving the member→root gather bytes moved the 64-core BLOCK shard only ~5%, so the
    combine is *not* byte-bound at these group sizes. What actually dominated was the ROOT's
    per-round `transform_in_place` — one SFPU rsqrt per tile-row (32 of them on that shard)
    with a `GROUP_SIZE`-wide fan-out of members blocked behind it. A cost invisible in a
    tile-op count; L6b is worth 1.14× there because of it.
  * **Zeroing the whole gather CB at boot to define the unshipped faces is a RACE.** A
    member's partial can land before the root's wipe, and every combine cell failed (pcc
    0.87–0.99, rms 0.08–1.10). Zeroing exactly the UNSHIPPED faces is race-free by
    construction — disjoint from everything any member writes — and is what ships. The same
    experiment is what *established* which faces the datapath reads (leading-3 passes, and
    {0,2} passes, so faces 1 and 3 are dead), which is also what makes L6b's `VectorMode::C`
    provably safe rather than hopeful.
  * **L5 is not free when it costs CB depth.** Holding a whole tile-row can leave no L1 for
    the depth the cross-processor CBs spend on movement↔compute overlap. Measured both ways:
    `(1,1,32,7168)` at `GRID_W=1` (one core, depth 2→1) went 41779 → 50598 ns (**0.83×**),
    while ROW_MAJOR `(1,1,32,4096)` — already depth 1 in *both* regimes, so no sacrifice —
    went 52197 → 47144 (**1.11×**) at the same single row-block. So the gate is on the
    **depth sacrifice**, not on L5, and only then on the row-block count. Guarded, both
    numbers recorded in D14.
- Block/depth co-tune, all measured and all NULL at the shipped values (r4target set, one
  fresh-cache run each): `CB_DEPTH_CANDIDATES = (2,1)` — within noise everywhere, so D4's
  recorded null still holds, and L5 has now taken over the residency band that knob was
  aimed at (a shape there goes ROW_RESIDENT instead of wanting depth 1). `CB_RM_STAGE_DEPTH
  = 3` — within noise (and inert for this TILE-only target set by construction).
  `L1_SAFETY_FRACTION = 0.90` — the one non-null: +2.2% on the BLOCK shard (82487 → 80713,
  `BLOCK_ROWS` 10 → 11) and noise elsewhere. **Declined deliberately**: Refinement 2's L1
  finding is that a proportional margin cannot cover the CB arena's fixed base offset, so
  trading 2% for reduced launch headroom across 5000+ golden cells (a CB-OOM is an op-charged
  hard failure, not a slow path) is the wrong side of that trade.
- Remaining headroom (a FINDING, not a queued task):
  * The **prefill profiles are at the DRAM roofline and the bytes are now near-minimal.**
    `(1,1,8192,7168)` moves 117 MB (x) + 50 (gamma) + 117 (out) = 285 MB in 643 µs = 443
    GB/s, the same aggregate bandwidth the pre-L5 build achieved while moving 470 MB. The
    only byte left to remove is gamma's 50 MB of per-core redundancy, which is **Lamp L2**
    (one injector per grid row multicasts gamma) — a scheme change, and `shared_input_reuse`
    warns it only pays in the wide-W/many-core corner.
  * The **sharded geometries are still 2–3× off their `achievable_ns`**, and after L6b the
    remaining cost is the combine ROUND COUNT, not its bytes or its finalize: the 64-core
    BLOCK shard runs 4 gather→root→mcast rounds because `BLOCK_ROWS` is capped at 10 by
    `cb_partials_gathered` (`GROUP_SIZE × BLOCK_ROWS` fp32 pages). The lever is D11's
    recorded (b) in its *real* form — a partial packed to its 32 useful floats rather than
    merely face-trimmed, which needs a transpose so the root can sum a compact layout — plus
    D11's (a) hierarchical gather. Both are combine data-format/topology changes.
  * The **prime-`Wt` cliff survives**: D1 still forces `WT_CHUNK | Wt`, so `Wt = 127`
    collapses to one tile per chunk. L5 removes that shape's pass-B re-read but not its 127
    one-tile compute phases; a ragged-tail chunk (runtime `wt_c`) is still the lever.
  * **Lamp L6c (eliding dtype reconfig) was costed, not built.** The chain already elides
    compile-time when the previous element on the same side programmed the same CB; the
    remaining candidates are the same-format-different-CB pairs in pass B. At master.md's
    ≈110–150 ns per reconfig and the measured reconfig count (≈2 per chunk per block) the
    ceiling is 1.2% on the BLOCK shard and 0.5% on prefill 7168 — against a silent-corruption
    risk on the mixed-dtype gamma paths the op supports. Not a trade worth taking here.
- Tests added: `test_rms_norm.py::test_rms_norm_row_resident_regime` (20 params) — the third
  compute regime pinned by shape for both layouts and both gamma layouts, with and without
  gamma, including two non-tile-aligned widths. `op_design.md` §4.2 makes regime pinning
  mandatory precisely because the selector depends on a device-dependent L1 budget, and this
  regime is the one with a genuinely different INDEXING scheme (every helper call reads the
  held CBs at a tile offset, popped once per row-block), where an off-by-one is a wrong-tile
  read rather than a hang. `test_rms_norm_perf.py::test_rms_norm_perf_r4target` (9 params) —
  the refinement's own targets: the four interleaved prefill profiles plus the five
  measured-fastest sharded geometries with their `_perf_case` `shard_shape` + `core_grid`
  PINNED rather than auto-derived, since the geometry is what the reference latency was
  measured on. One test function so a single space-free `-k r4target` selects the set (the
  `--profile` Tracy wrapper loses the quoting of a multi-word `-k`).
  `probes/read_perf_csv.py` — prints kernel ns + the per-RISC breakdown from the newest
  profiler CSV, keyed by shape + placement; the BR-vs-TR2 split in it is what identified the
  sharded geometries as finalize-bound rather than gather-bound.
