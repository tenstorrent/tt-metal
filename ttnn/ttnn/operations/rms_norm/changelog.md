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

## Perf 1 — the cross-core combine's ROOT chain (a fan-out tournament)
- Date: 2026-08-05
- A **perf tournament**, not a refinement: `SUPPORTED` is untouched, `EXCLUSIONS` is
  untouched, and `verify_supported` categories are identical before and after. Seven ideas
  were floated at the measured bottleneck, each fanned out to its own
  `blocking-perf-part-optimizer` with its own isolated on-device micro-benchmark under
  `perf_experiments/<slug>/` (all seven artifacts are committed). **All seven measured a WIN;
  four graduated, three are deferred to Perf 2 with their numbers intact.**

### Permanent instrumentation (new, and never to be removed)
- Added `ttnn/cpp/ttnn/kernel_lib/perf_instrumentation.hpp` — `MaybeDeviceZoneScope("<stage>")`
  with its **durability contract** written into the header: it compiles to nothing when the
  profiler is off, so there is never a perf reason to strip it, and stripping it would make the
  next round *guess* where the time goes. 19 stage zones now span the reader, every compute
  phase and the writer. `probes/zone_breakdown.py` reports per-zone ns per RISC per launch.

### Measured breakdown, and the ranked bottleneck
blackhole p150b, 110-core grid, CHIP_FREQ **1350 MHz** (== the reference clock, so no scaling);
bf16 / TILE / HiFi2 / `fp32_dest_acc_en=False` — the `_perf_case` config. One fresh-cache
profiled run per variant.

`feature_spec.py` carries **no `attention:` note**, so the focus shape was free-selected as the
largest *measured* headroom: **`(1,1,8192,1024)` BLOCK_SHARDED, shard `[1024,128]`, grid `(8,8)`
= 64 cores — 84836 ns against a 25640 ns reference, 3.31x off.** Every knob it declares is in
`SUPPORTED`, so no generality gap and no proxy shape.

Per-stage, max ns per core, focus shape:

| stage | ns | note |
|---|---|---|
| `compute_root_sum` | 26773 | root cores only — **rank 1** |
| `compute_root_finalize` | 22501 | root only — **rank 2** |
| `compute_stat_handoff` | 8177 | root only, a **pure fp32 tile copy** — rank 3 |
| `compute_reduce` / `compute_square` / `compute_gamma_mul` | 7970 / 6667 / 7077 | all 64 cores |
| `compute_partial_handoff` | 4089 | all 64 cores, another **pure copy** |
| `writer_gather_wait` / `writer_gather_zero` | 4727 / 7816 | round handshake / one-time boot |
| `reader_read_x` / `writer_write` | **59** / **53** | DM is already ZERO (native zero-copy shard CBs both ends) |

`compute_scale` 62604, `writer_mcast_recv` 52733 and `writer_gather_ship` 35932 are
`cb_wait_front` **waits on the stat** — consequences of the root chain, not independent stages.
The root core's TRISC zones sum to **83.4 us of its 84.5 us kernel wall** (full arithmetic
closure, no gap), so the **root's serial combine chain — root_sum + root_finalize +
stat_handoff = 57.5 us = 68% of the whole-op wall — IS the critical path**, and all seven other
cores of its group block behind it.

**Cumulative payload peel** (payload only; every CB reserve/push/wait/pop and every loop trip
count preserved): full **84836** -> finalize-SFPU-stubbed **68532** -> + root-adds-stubbed
**52096**. So those two stages' payload is 32.7 us and a *further* ~16.6 us of their zone time is
pack/unpack/DEST-sync **scaffolding that only a fusion deletes** — which is what put the fusion
idea in the portfolio.

**Roofline-gated, deliberately not targeted** (`/perf-ceiling-dm`): `reader_read_x` = 59 ns and
`writer_write` = 53 ns on the focus shape (already zero — native sharding is a *precondition* to
this pass, not one of its levers); and `reader_read_x` on interleaved prefill w1024 moves
196 kB/core x 110 cores in 24.4 us ~= **880 GB/s aggregate, at the DRAM roofline**.

**This measurement REFUTED a prior round's recorded guess.** Refinement 4 named the remaining
sharded cost as the combine **round count**; measured, round count is worth ~4.7 us of
`writer_gather_wait` across 4 rounds, while the root's per-tile **compute** is 57.5 us. Ranking
by measurement rather than by the prior note is what aimed this tournament correctly.

A second regime was measured and ranked too — interleaved prefill `(1,1,8192,1024)`, 104632 ns:
`reader_read_gamma` **61099 ns of the 104 us wall**, i.e. 2.5x the x read at a third of the
bytes, because all 110 cores read the *identical* 2 kB of gamma from the same DRAM pages.

### The portfolio (7 ideas; overlap and fusion deliberately allowed)
| idea | verdict | measured | domain |
|---|---|---|---|
| `fuse_root_combine` — sum+finalize+handoff in ONE DEST window | **WIN** | 11933 -> 7409 ns/round (1.61x), 3655 with a scoped finalize (3.26x) | everywhere on COMBINE, no exceptions |
| `root_sum_accumulate` — stop the accumulator's L1 round trip | **WIN** | `dest_acc_any` **2.20x** (GS=8,rows=10) / **2.16x** (GS=32,rows=1) vs the in-tree fold | everywhere, no parity predicate |
| `root_finalize_scope` — (a) column-scope the whole finalize, (b) delete the handoff copy | **WIN** | (a) 762.1 -> 372.8 ns/tile (**2.04x**); (b) **1.21x**; together **3.02x** | everywhere; `cskip_fused` inexpressible off power-of-4 W |
| `reduce_pack_to_handoff` — reduce packs straight into the handoff CB | **WIN** | 11933 -> 9377 ns (**1.27x**), wins all 11 geometries, `torch.equal`-identical | everywhere on COMBINE |
| `compact_partial_transpose` — pack 32 tile-rows' partials into ONE tile | **WIN** | root stage 41944 -> 9535 (4.4x) at BR=10, -> 4341 (9.7x) at BR=32 | everywhere for `BLOCK_ROWS > 1` |
| `hierarchical_gather` — tree / row-split instead of a flat root | **WIN** | `rowsplit` **1.80x** focus, `tree_k4` **2.10x** secondary | `GROUP_SIZE == 4` a measured regression (0.955x) |
| `gamma_mcast_reuse` — one injector broadcasts gamma | **WIN** | stage **3.92x**, reader wall **1.24x**, whole-op **1.19-1.27x**, BIT-EXACT | sharing group >= ~12 cores |

**Nothing was null this round** — unusual, and it is the measured breakdown's doing: every idea
was aimed at a stage that a peel had already proven dominant.

### What graduated, how widely, and what was deleted
Four changes landed. **All four are unqualified single paths with ZERO carve-outs**, and each
deleted the code it replaced.

- **D16 — row-major gather + a per-row fold.** Adopted (it arrived in the tree mid-tournament,
  uncommitted and unattributed, so it is measured and recorded here rather than silently kept).
  Whole-op, standalone: **1.13-1.36x on the width-shard profiles**, 1.05x on the focus shape.
  Its gather-layout half is load-bearing — two winners require a row's partials contiguous.
- **D17 — the WHOLE finalize chain scoped to column 0** (`cskip2`, raw sfpi). Replaces the
  three-call `VectorMode::RC` chain *and* folds `*(1/W)` and `+eps` into one DEST pass.
  **Deleted: the `RSQRT_COL_SCOPE` knob, its whole-tile branch, compute CT arg 16, and the old
  `rsqrt_tile_col`.** This is the widest-domain graduation in the round — the *local*
  (non-combine) finalize runs on every shape the op supports, so every cell gets it. Nothing
  earned a guard: the unscoped form was the slowest cell measured at every geometry.
- **D18 — pass A's reduce packs straight into `cb_sum_handoff`.** Deleted the
  `compute_partial_handoff` copy on every core of every group. **Satisfies** the CB-ownership
  rule rather than bending it (compute is the single producer, the writer the single consumer,
  and `cb_row_stat` becomes strictly compute-private and root-only). Legal because the combine
  path is already `static_assert`ed to one width chunk, so the reduce never re-reads its
  accumulator — an *asserted precondition*, not a runtime guard.
- **D19 — the root finalize reads `cb_row_stat` and writes `cb_stat_handoff` in one chain.**
  Deleted the `compute_stat_handoff` copy and its now-meaningless zone (its cost lives inside
  `compute_root_finalize`; an empty zone reporting ~0 ns would mislead the next round).

**Raw-LLK bypass — one, D17**, carrying its measured justification at the definition so a later
helper-usage pass will not "fix" it back: `mul_unary_tile` / `add_unary_tile` / `rsqrt_tile` all
hard-code `VectorMode::RC` and expose no seam, and column parity is the SFPU's *inner* walk axis
so `ITERATIONS` cannot reach it. **Safety is measured, not assumed**: an isolated bench ran pass
B's real consumer (`mul<BroadcastDim::Col>`) over a stat tile with columns 1..31 seeded five
orders of magnitude wrong and got pcc 0.999992 — the column broadcast reads column 0 only.
Precedent: `sdpa/.../compute_common.hpp` `recip_tile_first_column`.

### Whole-op before/after (one fresh-cache profiled run per variant)
| target | reference | before | after | speedup | vs reference |
|---|---|---|---|---|---|
| **`(1,1,8192,1024)` BLOCK 64c** | 25640 | 84836 | **64641** | **1.31x** | 3.31x -> **2.52x off** |
| `(1,1,32,5120)` WIDTH 32c | 5267 | 11263 | **7612** | **1.48x** | 2.14x -> 1.45x off |
| `(1,1,32,7168)` WIDTH 28c | 5481 | 10871 | **7385** | **1.47x** | 1.98x -> 1.35x off |
| `(1,1,32,1024)` WIDTH 8c | 4110 | 5767 | **4374** | **1.32x** | 1.40x -> 1.06x off |
| `(1,1,32,2304)` WIDTH 9c | 4617 | 6640 | **5270** | **1.26x** | 1.44x -> 1.14x off |
| `(1,1,8192,1024)` interleaved | 96744 | 104632 | 103985 | 1.01x | flat |
| `(1,1,8192,2304)` interleaved | 211345 | 221661 | 223501 | 0.99x | flat (noise) |
| `(1,1,8192,7168)` interleaved | 1032281 | 652323 | 644964 | 1.01x | already 1.60x better |
| `(1,1,8192,5120)` interleaved | 738307 | 463627 | see below | **flat** | already 1.54x better |

The interleaved prefill profiles are flat by construction: they take the *local* finalize (2081 ns
of a 104 us wall) and none of the combine path, so only D17 reaches them and it cannot move a
reader-bound shape.

**Guard-set no-regression: no supported cell got slower.** One apparent exception was chased to
ground rather than guarded: `(1,1,8192,5120)` first read as 0.97x. `compute_finalize` measures
**1672 ns of that shape's 478 us wall (0.35%)**, so the only graduated change touching it cannot
account for a 15 us swing. An A/B probe reverting *only* the finalize spelling measured stock
476476 / 475303 / 477779 against scoped 477994 / 481342 / 478860 — **0.5% apart, i.e. flat**. The
round-1 single-run 463627 was a low outlier on a bandwidth-bound shape (Refinement 4 recorded
468093; the D16 tree 468887). **No predicate was erected** — a suspicion is not a justification.

### Correctness (green throughout; counts numerically identical to Refinement 4)
- Golden cartesian, run as four placement slices: **5037 pass / 1365 xfail / 0 fail**
  (INTERLEAVED 1500+420, HEIGHT 1167+315, WIDTH 1185+315, BLOCK 1185+315), `supported_fail = 0`.
- Golden loose cases: **384 pass / 3 infeasible-skip / 0 xfail**. Golden total **5421 pass**.
- `test_regression.py` **15/15**. `test_translated.py` **105/106** — the one failure bit-identical
  at frobenius **0.1122406** to the pre-existing `{bfloat8_b, w_non_aligned}` pad-poison
  non-issue `feature_spec.INVALID` declares out of scope (recorded the same way in Refinements
  2, 2b, 3 and 4).
- Unit suite **463 passed / 30 skipped** — precision baselines, wide-W precision, pad-poison, all
  three sharded schemes x both layouts, the RM BAND geometries, and the perf probes. Zero hangs.
- **Precision contract untouched**: `fp32_dest_acc_en`, `math_fidelity`, `math_approx_mode` and
  every dtype are exactly as the caller passed them. D17 *removes* a bf16 DEST rounding (it keeps
  `*(1/W)` in an fp32 LREG), so it is at least as accurate as the chain it replaces; D18 is
  `torch.equal`-identical. No option that traded precision for speed was graduated.

### Deferred to Perf 2 — three measured WINs, with their numbers, not re-litigations
Each needs re-measuring against the *new* critical path before it is worth integrating, which is
the designed use of a second round. Deferring a measured win has a real cost and it is recorded
as such, not as a null.
1. **`compact_partial_transpose`** — the single biggest lever (root stage 4.4x at BR=10, 9.7x at
   BR=32; extrapolated 1.94x whole-op). Mechanism is a column permutation by ONE `matmul_tiles`
   against a one-hot tile (26 ns/tile-row to pack, 43-69 to un-pack, against 1211 ns/tile-row
   deleted) — *not* a `transpose_wh`. Deferred because it is extrapolated rather than
   end-to-end; it **conflicts with D17's column scoping above `BLOCK_ROWS = 16`** (a compact tile
   with >16 packed rows has live data in faces 1/3, so the root's rsqrt would have to widen back
   to `RC`); it needs a one-hot L1 bank (fp32, 80-256 kB); it relies on the packer leaving DEST
   zeroed; and it requires every operand column to be FINITE (`inf*0 = NaN`). Also worth noting:
   it makes the gather's boot-zeroing unnecessary, since every compact position is defined by
   construction. Its own baseline is what D16-D19 just replaced, so re-measure first.
2. **`hierarchical_gather`** — `rowsplit` 1.80x on the focus, `tree_k4`/`k8` 2.10x on the
   secondary; single rule "spend gatherers on rows first, then a slot tree with `k` nearest 4".
   Its author's own composition note says the absolute win scales down with the per-fold cost —
   and this round cut per-fold cost — so it *must* be re-measured. Carries a real earned
   carve-out (`GROUP_SIZE == 4` with rows>1 measured 0.955x, expressed as the mechanism
   `GROUP_SIZE - (m + k) >= 2`), needs a 4th semaphore, and it also *lowers* L1.
3. **`gamma_mcast_reuse`** — `mcast_1inj_noc0`, stage 3.92x, whole-op **1.19-1.27x** on every
   interleaved prefill profile, **bit-exact**. It has the largest integration surface (a second
   mcast family in the descriptor, sems 3-4, and the gamma group is a grid *column* while the
   stat group is a *row*). Its author also corrected the breakdown honestly: the 61.1 us
   `reader_read_gamma` zone **overstates** gamma's marginal cost, because the reader is
   DRAM-bound at ~405 GB/s and gamma is 30% of reader *bytes*, not 59% of the wall. Earned
   carve-out: sharing groups of <= 8 cores measured 0.78-0.89x. Keep the reader on NOC_0 — x
   reads cost 1.65x on NOC_1.
4. **`root_sum_accumulate`'s `dest_acc_any`** (2.20x/2.16x on the fold) — deferred only because
   it needs a new 1-page fp32 `cb_zero_tile` (a *constant* 4 kB, so it does not perturb the
   `BLOCK_ROWS` solve the way a per-row term would) and this round had no golden budget left to
   validate a descriptor change. `dest_acc_wide_pad` is ~2x better again but needs the gather CB
   widened to `GROUP_SIZE + GROUP_SIZE % 2` pages per row with the pad boot-zeroed.

### Two adjacent findings, measured in passing, NOT built
- **The reduce is on the wrong datapath at `X_SQUARED_WT == 1`.** At a per-call reduce width of 1
  — which Refinement 4's L6d DEST fold *creates* — `ReduceTile` is **2.91x faster than
  AccumulateViaAdd at equal-or-better precision** (3219 vs 9374 ns for 40 calls; rel-RMS 0.00314
  vs 0.00337). `AccumulateViaAdd` from width 1 to 4 is flat, i.e. pure per-call overhead.
  `REDUCE_ACC_VIA_ADD_MIN_CHUNK_WT` exists to prevent exactly this but is evaluated against
  `wt_chunk` rather than the reduce's *actual* per-call width. A one-line descriptor change —
  but the crossover sits between width 4 and 32 and the precision cost at wide widths is real
  (`ReduceTile` at 32 is both slower *and* pcc 0.9867), so it must stay width-gated and needs its
  own measurement. `compute_reduce` is 7970 ns on all 64 cores of the focus shape.
- **TILE gamma reads 64 kB per core where 2 kB is meaningful.** Gamma is a `(1,1,1,W)` vector
  padded to 32 tile rows, and the consumer is `BroadcastDim::Row`, which reads row 0 only.
  Fetching just the meaningful face-rows is a ~32x byte reduction that *composes* with the gamma
  broadcast above, shrinking gamma's DRAM share from ~30% to ~1%. Unbenched.

### Repo hygiene fixed in passing
`perf_experiments/__init__.py` made `ttnn/ttnn/operations/__init__.py`'s `pkgutil.walk_packages`
import and EXECUTE every experiment scratch file on every `import ttnn` — it broke `import ttnn`
repo-wide twice mid-tournament. Removed, with the reason recorded in
`perf_experiments/README.md` so it is not re-added.

## Perf 2 — the root chain, pass B's DEST lanes, the reduce datapath, and the combine pipeline (a fan-out tournament)
- Date: 2026-08-05
- A **perf tournament**, not a refinement: `SUPPORTED` untouched, `EXCLUSIONS` untouched,
  `verify_supported` categories identical before and after. Eight ideas were floated at the
  measured bottleneck, each fanned out to its own `blocking-perf-part-optimizer` with its own
  isolated on-device micro-benchmark under `perf_experiments/<slug>/` (all eight artifacts
  committed). **All eight measured; six graduated as five deviations (D20–D25), one was a
  measured REGRESSION, one was superseded, and two mutually-exclusive winners are deferred with
  their numbers.**
- Round 2 of 2, so the breakdown was **re-measured on the now-instrumented, partly-optimized op**
  rather than inherited. That mattered: Perf 1's rank-1 and rank-2 stages had moved.

### Measured breakdown, and the ranked bottleneck
blackhole p150b, 110-core grid, CHIP_FREQ **1350 MHz** (== the reference clock, so no scaling);
bf16 / TILE / HiFi2 / `fp32_dest_acc_en=False` — the `_perf_case` config. One fresh-cache
profiled run per variant, **no trial loop** (device kernel time has no warm-up transient).

`feature_spec.py` still carries **no `attention:` note**, so the focus shape was free-selected as
the largest *measured* headroom — the same shape Perf 1 took from 84836 to 64677:
**`(1,1,8192,1024)` BLOCK_SHARDED, shard `[1024,128]`, grid `(8,8)` = 64 cores — 64753 ns against
a 25640 ns reference, 2.52x off.** Every knob it declares is in `SUPPORTED`; no generality gap, no
proxy shape. Derived: Rt=256, 32 tile-rows x Wt=4 per core, GROUP_SIZE=8, BLOCK_ROWS=8 (4 combine
rounds), `X_SQUARED_WT=1`, `GATHER_FACES=2`.

**Arithmetic closure was established on the critical-path core before ranking anything.** A new
probe (`probes/zone_percore.py`) reports one *core's* zone totals rather than max-over-cores,
because max-over-cores cannot close (different zones peak on different cores). The **root** core's
TRISC_0 zones sum to **63214 ns of its 64221 ns kernel wall** (gap 1007 ns), so on that core the
zones *are* the wall:

| stage | ns (root core) | note |
|---|---|---|
| `compute_root_sum` | 26773 -> **25583** | root only; **~13610 of it is the gather WAIT** (payload ~11973) |
| `compute_scale` | **13591** | real on the root; 42800 on a member — almost all `cb_wait_front` on the stat |
| `compute_root_finalize` | 22501 -> **9565** (math) | root only |
| `compute_gamma_mul` / `compute_square` / `compute_reduce` | 7047 / 5901 / 5270 | all 64 cores |
| `writer_gather_zero` | **7633** | one-time boot on the 8 roots |
| `reader_read_x` / `writer_write` | **57 / 56** | DM is ZERO — native zero-copy shard CBs both ends |

**Cumulative peel** (payload stubbed, every CB reserve/wait/push/pop and trip count kept; peeled
cumulatively, not one at a time): full **64753** -> root_sum payload stubbed **54806** -> +
root_finalize stubbed **41523** -> + gather-zero stubbed **40460**.

Two things only the *cumulative* peel could show:
- Removing `compute_root_sum` alone bought 9947 ns against a 25583 ns zone, while removing
  `compute_root_finalize` *after* it bought 13283 ns against a 9565 ns zone — **more than its own
  zone**. Ranking either stage from a solo ablation would have been wrong in both directions.
- `writer_gather_zero` has a 7633 ns zone but was worth only **1063 ns of wall** —
  **overlap-hidden behind an 11.3 us pass A, not cheap.** Correctly ranked *out* of the portfolio
  on that basis. (It came back to bite; see "what this round did to itself" below.)

**Ranked, roofline-gated (`/perf-ceiling-dm`):**

| rank | stage | wall contribution | share |
|---|---|---|---|
| **1** | the root combine chain (`root_sum` payload + `root_finalize`) | **23230 ns** | **35.9%** |
| 2 | the root's residual gather-arrival WAIT, 4 rounds | 13610 ns (~3400/round) | 21% — latency, not payload |
| 3 | pass B: `compute_gamma_mul` 7187 + `compute_scale` 5887, all 64 cores | 13074 ns | real cost, measured with the root chain ablated |
| 4 | pass A: `compute_square` 6198 + `compute_reduce` 4994, all 64 cores | 11192 ns | |
| gated OUT | `writer_gather_zero` | 1063 ns of wall | overlap-hidden |
| gated OUT | `reader_read_x` / `writer_write` | 57 ns | native sharding is a **precondition** of this pass, not a lever |
| gated OUT | interleaved prefill's x read | — | 880 GB/s aggregate = the DRAM roofline (hence the gamma idea, not an x idea) |

Secondary regime ranked too — interleaved prefill `(1,1,8192,1024)`, 104705 ns against a 96744
reference, where Perf 1 measured `reader_read_gamma` at 61099 ns because all 110 cores read the
identical 2 kB of gamma from the same DRAM pages.

### The portfolio (8 ideas; overlap and fusion deliberately allowed)
| idea | verdict | measured | domain |
|---|---|---|---|
| `root_chain_dest_fuse` — fold + finalize in ONE DEST window, DEST-resident accumulator | **WIN** | stage pair **5874 -> 2698 ns/round (2.18x)**; 1.73x–3.86x over GS{4,8,9,16,28,32} x rows{1,8,32} | everywhere on COMBINE, **zero exceptions** |
| `root_sum_dest_accumulate` — DEST-accumulate the fold alone | **WIN, superseded** | fold **3715 -> 776 ns (4.79x)**; 1.48x–6.82x over 18 geometries | everywhere; but it **is** the fusion's fold-only half |
| `compact_partial_transpose_r2` — BLOCK_ROWS partials into ONE tile by a one-hot matmul | **WIN, deferred** | root chain **6398 -> 1732 ns/block (3.69x)**; O(BR·G) -> O(G), so 15.2x at BR=32 | everywhere on COMBINE, **zero exceptions**; BR=1 flat |
| `hierarchical_gather_r2` — split the fold across m gatherers + a k-ary slot tree | **WIN, deferred** | combine bench **46436 -> 18748 ns (2.48x)**; 1.00x–5.96x over 15 cells | everywhere on COMBINE, **three carve-outs** |
| `pass_b_fuse_scale_gamma` — x·stat·gamma in one DEST window | **REGRESSION** (+ a WIN found in the slot) | fusion **0.84x**; the `blk` lever **14050 -> 8860 ns (1.59x)** | fusion: do not ship. `blk`: everywhere |
| `reduce_at_percall_width_1` — gate the reduce datapath on the real per-call width | **WIN** | pass A **11551 -> 6079 ns (1.90x)** at *better* rel-RMS | everywhere the D12 fold is on |
| `combine_pipeline_depth` — overlap block r+1's pass A with block r's combine | **WIN** | whole op **64707 -> 53740 ns (1.204x)**, end-to-end | every combine on a sharded input |
| `gamma_broadcast_and_trim` — mcast gamma and/or read only its meaningful face-rows | **WIN** (trim shipped, mcast declined) | trim **104376 -> 90338 ns (1.155x)**; mcast 1.174x; composed 1.178x | every TILE gamma |

**Seven WINs and one REGRESSION.** The regression is the useful one: `master.md`'s `compute_fusion`
says FPU-consumer DEST-reuse loses (0.82x), `op_design.md` §1.7 already cites that to justify the
current two-pass pass B, and the subagent **reproduced it at this exact config** (0.84x) instead of
taking the catalog's word for it — then spent the rest of its slot on four alternatives and found
the `blk` lever that graduated. A slot aimed at a likely-null idea still paid.

### Aggregation — what conflicted with what
- **`root_sum_dest_accumulate` is superseded, not stacked.** Its `destacc_split` variant *is* the
  fold-only half of `root_chain_dest_fuse` (1.93x vs 2.18x for the pair). Both authors independently
  said so. Counting both would have been double-counting.
- **`compact_partial_transpose_r2` and `hierarchical_gather_r2` are mutually exclusive**: compaction
  collapses a sender's BLOCK_ROWS partials into one tile, which **removes the very row axis** the
  row split parallelizes over. Compact won the head-to-head (3.69x vs 2.69x on the chain) and carries
  **zero** exceptions against hierarchical's three — but both are deferred (below).
- The five that graduated (`reduce_at_percall_width_1`, the `blk` lever, `root_chain_dest_fuse`,
  `gamma` trim, `combine_pipeline_depth`) are **mutually independent**: pass A's datapath, pass B's
  call granularity, the root's fold, the reader's gamma bytes, and the loop's issue order.

### What graduated, how widely, and what was deleted
Six changes landed as five deviations. **Every one is a single unqualified path** for the domain it
is correct on, and each deleted the code it replaced. Two carry carve-outs; both were *measured*.

- **D20 — the reduce datapath's third floor.** `REDUCE_ACC_VIA_ADD_MIN_CHUNK_WT` existed to keep
  `AccumulateViaAdd` off narrow reduces but was evaluated against `wt_chunk`, while D12's
  square-DEST-fold had already collapsed the reduce's *per-call* width to `x_squared_wt = 1` — so the
  guard was blind to exactly the case it existed to exclude. Now there are three floors on three
  quantities. **`compute_reduce` 5270 -> 1660 ns on all 64 cores (3.2x).** Every build the carve-out
  does not name is byte-identical to Refinement 4.
  **The carve-out's polarity was itself established by measurement, in both directions**, and this is
  the most instructive thing in the round:
  * *replacing* the old floor with the new one regressed
    `test_sharded_wide_w_keeps_the_reduce_datapath` — `(1,1,160,11008)` HEIGHT, a 344-tile shard —
    to rel-RMS **0.04774** against its 0.04 bound, because at `num_w_chunks > 1` the row total is
    carried *across* calls and ReduceTile accumulates 344 chunks of all-positive addends in a 16-bit
    DEST word. That cross-chunk depth is **invisible to any per-call measurement**;
  * *vetoing* the new floor on `num_w_chunks > 1` instead regressed
    `test_sharded_row_major[ragged_width_tail_wt127]` — `(1,1,32,4064)` RM — to rel-RMS **0.06093**,
    because it re-admitted `AccumulateViaAdd` at `wt_chunk == 1`, the regime the *old* floor was
    measured to exclude.
  **pcc stayed 0.9999 through both regressions.** Only the rel-RMS bound saw them — which is
  precisely why Refinement 1b's nets assert an rms bound and not just a pcc one.
- **D21 — pass B's DEST-lane block size** (`PASS_B_BLK`, the largest divisor of `WT_CHUNK` capped by
  `DEST_AUTO_LIMIT`, never a literal 8) plus the `PerChunk` pack lifecycle it *requires*.
  **14050 -> 8860 ns (1.59x), bitwise identical output.** `compute_gamma_mul` 7047 -> 4500 ns on all
  64 cores. Decomposed: -3.3 us from one reserve/push per chunk instead of per tile, -1.9 us from 4
  DEST lanes per outer iter. The shape argument and the pack policy are **one change** — at
  `block_size > 1` the lifecycle is emitted once per outer iter, so leaving `PerTile` reserves 1 page
  and packs `blk`, corrupts the ring and **hangs**. No carve-out: `WT_CHUNK == 1` clamps `blk` to 1
  and is **flat, therefore in the domain**.
- **D22 — the FUSED ROOT CHAIN** (rank 1). The group fold accumulates **pairwise in DEST** and the
  finalize runs in that same DEST window, with one pack. **Deleted: D16's `ROOT_FOLD_OUT`, D19's
  separate finalize chain, and every COMBINE-path use of `cb_row_stat`**; the `compute_root_sum` and
  `compute_root_finalize` zones are replaced by one `compute_root_fused`. Stage pair
  **5874 -> 2698 ns (2.18x)**; whole op 55561 -> 43865 ns. Needs `GATHER_SLOTS` (`group_size` rounded
  up to even) so the pairwise walk is universal — the pad slot is boot-zeroed, no member can ever
  write it, and it contributes an exact +0.0, so **odd `GROUP_SIZE` is in the domain with no second
  path**. Zero carve-outs.
  **It is MORE accurate than the chain it replaces** — rel-RMS **2.42e-3 vs 3.38e-3** (3.36e-3 vs
  5.09e-3 at GROUP_SIZE=28) — which **refutes D16's recorded reasoning** that an fp32-L1 accumulator
  is "at least as accurate": the packer fold rounds *every* contributor into a 16-bit DEST word
  before its exact fp32 add, paying GROUP_SIZE roundings in a **linear** chain, while the pairwise
  DEST walk pays the same per-addend rounding but sums as a **tree** (log2(G)+1). A precision hedge
  was measured too and was both slower *and* less accurate — there was nothing to hedge.
- **D23 — the TILE-gamma face-row trim.** A `(1,1,1,W)` TILE gamma is tile-padded to 32 rows and its
  only consumer reads **row 0**, so the reader was moving a whole tile for a couple of meaningful
  face-rows. Interleaved prefill **1.107x–1.182x**, **bit-exact** across 7 geometries x 3 gamma
  dtypes. **Flat on the BLOCK focus shape (0.998x) and deliberately NOT guarded there.**
- **D24 — the root publishes its own stat copy before the broadcast**, so its pass B stops waiting
  out its own multicast (-2643 ns on the root). A two-line reorder; legal because `send()` and pass B
  are both *readers* of those pages.
- **D25 — the COMBINE PIPELINE** (rank 2). Block blk+1's pass A is issued *before* block blk's
  combine, filling the root's measured ~3400 ns/round arrival idle, plus `cb_sum_handoff` at depth 2
  (+40 kB/core). **Whole op 43700 -> 34438 ns (1.269x)**, `torch.equal`-identical to serial.

**The two earned carve-outs, and why each is one:**
1. **D25 is carved out to `native_in`** — a *correctness* exception, and doubly earned. On a
   reader-fed input CB the pipeline is **incorrect** (pcc 0.980150: pass A a block ahead addresses x
   at a tile offset past an un-popped front, and a tile offset cannot cross a CB ring **wrap**; a
   shard-backed CB is the whole assignment, so its front never wraps) **and**, once made correct by
   sizing the ring to `num_blocks+1` at +192 kB/core, still **0.894x** because that regime is
   reader/DRAM-bound. Written as the narrow exception: everything shard-backed gets the pipeline.
2. **D23's bfloat8_b demotion to a half-page read** — `inexpressible`, and a **format fact rather
   than a shape guard**: a 1088-byte bf8b tile has a **272-byte face** which is not 64-B DRAM
   aligned, so a face-offset read is silently truncated. The half page is faces 0+1 in *every* tiled
   format, still bit-exact, still 2x fewer bytes. (A ROW_MAJOR gamma has no tile padding to trim at
   all — also `inexpressible`, and it already moves ~2 kB/core instead of ~64 kB.)
   **D25's gather ring stays at ONE round** for a third measured reason: deepening it was a
   regression twice over (1.089x vs 1.204x at the op's `block_rows`; 59435 vs 54474 ns at *equal*
   `block_rows`) and it overshoots L1 by ~115 kB, forcing the `block_rows` solve down and *costing* a
   round. It is also unnecessary — the writer already carries the happens-before chain.

**Nothing was guarded on a suspicion.** `WT_CHUNK == 1` (D21), `num_blocks == 1` (D25), the BLOCK
focus shape (D23) and `BLOCK_ROWS == 1` are all **flat, and all left on the unified path**.

**Raw-LLK bypass — one, D22**, with its measured authorisation at the definition so a later
helper-usage pass will not "fix" it back: the fusion is **inexpressible** through `eltwise_chain`
because every chain element's apply runs on *every* inner iteration, so a `StatFinalize` element
placed after an accumulating `BinaryFpu` would `rsqrt` a **partial** sum `GROUP_SIZE/2` times —
there is no apply-after-the-accumulation element kind and no per-row tail hook. The
helper-expressible split form is recorded at 1.93x against the fusion's 2.18x so the gap stays
re-checkable. The finalize itself is still D17's sanctioned raw-sfpi `stat_finalize_payload`, called
unchanged, and its lane invariant is preserved verbatim (`<STRIDE=2, ITERS=4>` at `VectorMode::C`,
so `+eps` still covers every lane the `rsqrt` touches).

**An integration bug worth recording, because a raw-LLK path is where it lives.** The fused chain
must emit by hand the **data-format reconfig** the chain helpers emit for free: pass A leaves the
unpacker configured for `cb_x_squared` (bf16) and the packer for `cb_sum_handoff`, while the gather
and the handoff are fp32. Without `reconfig_data_format` + `pack_reconfig_data_format` the fold
unpacked fp32 L1 through a bf16 srcA/srcB, read ~0, and the finalize returned `rsqrt(eps)` — a
**uniform ~1000x scale error that held pcc at 0.9997** and was caught only by the rel-RMS bound
(994 against 0.04). Two independent bugs this round were visible only in rel-RMS.

### Whole-op before/after (one fresh-cache profiled run per variant; guard set = the `_perf_case` table)
| target | reference | Perf 1 after (= Perf 2 before) | Perf 2 after | speedup | vs reference |
|---|---|---|---|---|---|
| **`(1,1,8192,1024)` BLOCK 64c** (focus) | 25640 | 64677 | **34438** | **1.878x** | 2.52x -> **1.34x off** |
| `(1,1,32,5120)` WIDTH 32c | 5267 | 7630 | **6235** | **1.224x** | 1.45x -> 1.18x |
| `(1,1,32,7168)` WIDTH 28c | 5481 | 7464 | **6168** | **1.210x** | 1.36x -> 1.13x |
| `(1,1,32,1024)` WIDTH 8c | 4110 | 4359 | **3711** | **1.175x** | 1.06x -> **0.90x, beats ref** |
| `(1,1,32,2304)` WIDTH 9c | 4617 | 5291 | **4549** | **1.163x** | 1.15x -> **0.99x, beats ref** |
| `(1,1,8192,1024)` interleaved | 96744 | 104705 | **88562** | **1.182x** | 1.08x -> **0.92x, beats ref** |
| `(1,1,8192,2304)` interleaved | 211345 | 222536 | **193284** | **1.151x** | 1.05x -> **0.91x, beats ref** |
| `(1,1,8192,5120)` interleaved | 738307 | 475772 | **427777** | **1.112x** | 0.64x -> **0.58x** |
| `(1,1,8192,7168)` interleaved | 1032281 | 641683 | **569009** | **1.128x** | 0.62x -> **0.55x** |

**Guard-set no-regression: every one of the nine targets got FASTER**, by 1.11x–1.88x — one
representative per distinct kernel path x layout x placement (BLOCK / WIDTH / interleaved, root and
member cores, combine and local finalize, `native_in` and reader-fed). **Six of the nine now beat
their reference outright**, where Perf 1 left two doing so. The interleaved prefill profiles, which
Perf 1 recorded as *flat by construction* because only D17 reached them, move 1.11x–1.18x this round
because D21 and D23 both apply there — a regime Perf 1 could not touch.

### Final per-stage state on the focus shape (34438 ns), and what this round did to itself
| stage | Perf 2 before | after | note |
|---|---|---|---|
| `compute_root_sum` + `compute_root_finalize` | 31404 | **13534** (`compute_root_fused`, TRISC_0) | D22 + D25 |
| `compute_reduce` | 5270 | **1660** | D20 |
| `compute_gamma_mul` | 7047 | **4500** | D21 |
| `writer_gather_zero` | 7633 zone / **1063 wall** | 7811 zone / **2625 wall** | **promoted by this round** |

**`writer_gather_zero` is now the honest next target, and this round created that.** It was correctly
ranked *out* of the portfolio because the cumulative peel measured it at 1063 ns of wall —
overlap-hidden behind an 11.3 us pass A. D20 and D21 cut pass A to ~7.6 us, so **re-ablating it now
measures 34438 -> 31813 = 2625 ns of wall (7.6%)**: it is exposed and is a top-3 stage.
**Not graduated**, because removing a defined-ness guarantee on the gather CB was not one of the
eight measured ideas and the argument for it (that faces 1/3 garbage cannot reach column 0 through
the FPU fold or the column-scoped finalize) is *reasoned, not measured*. The principled routes are
(a) measure it the way D17 and D23 measured theirs — seed the unread lanes catastrophically wrong
and look at pcc — or (b) take `compact_partial_transpose_r2`, which makes the stage unnecessary by
construction.

### Deferred, with their numbers intact — two mutually-exclusive measured WINs
Neither is a re-litigation; both are re-measured against the *current* tree, and they cannot both
land. There is no Perf 3 scheduled, so deferring them has a real cost and it is recorded as such.
1. **`compact_partial_transpose_r2`** — root chain **6398 -> 1732 ns/block (3.69x)** at the focus
   geometry, and **O(GROUP_SIZE) instead of O(BLOCK_ROWS·GROUP_SIZE)**, so 15.2x at BR=32. Zero
   exceptions. Mechanism: a partial is a column vector in column 0, so a sender permutes its
   BLOCK_ROWS partials into BLOCK_ROWS *columns* of one tile with a single `matmul_tiles` against a
   one-hot bank (srcB `transpose` serves both directions from one bank). Perf 1's two recorded
   objections were both **refuted**: the one-hot bank is an L1 **win** (16 kB at BR=8, and
   `cb_partials_gathered` drops from `GROUP_SIZE·BLOCK_ROWS` to `GROUP_SIZE` pages — net **-256
   kB/core**), and the D17 conflict above BR=16 costs a flat **+455 ns/block** against a 23412 ns
   saving. It also makes `writer_gather_zero` unnecessary — the stage that this round just exposed.
   Deferred purely on integration surface: writer + compute + descriptor + a host-generated one-hot
   tensor, landed on top of five other graduations in one round. **Note it must be re-based onto
   D22**: its baseline is the pre-D22 root chain, and the finalize must go `<1,8> VectorMode::C`
   (BR<=16) or `<1,8> RC` (BR>16) — D17's `<2,4>` even-parity scope reaches only columns 0,2..14 and
   is silently wrong on a compact tile from BR=2 (measured pcc 0.9974).
2. **`hierarchical_gather_r2`** — combine bench **46436 -> 18748 ns (2.48x)**, up from Perf 1's
   1.80x. Perf 1's own warning that the win would shrink as per-fold cost fell is **refuted**: the
   absolute saving was unchanged (~27.7 us) while the baseline shrank, so the *ratio* grew. One rule
   (`m = min(BLOCK_ROWS, GROUP_SIZE)`, then a k-ary slot tree with two `K=1` guards) is optimal at 13
   of 15 measured cells and never below 1.00x; `writer_mcast_recv` collapses 42778 -> 4733 ns because
   nobody blocks behind a root; L1 goes 352 kB -> 44 kB/core. Perf 1's `GROUP_SIZE == 4` carve-out
   was **corrected**: it belongs to the *tree*, not the row split (row split at G=4 is 1.88x).
   Deferred because it loses the head-to-head to (1) and cannot coexist with it.

### Correctness (green throughout; counts identical to Perf 1)
- Golden cartesian: **5037 pass / 1365 xfail / 0 fail**, `supported_fail = 0`.
- Golden loose cases: **384 pass / 3 infeasible-skip / 0 xfail**. Golden total **5421 pass**.
- `test_regression.py` **15/15**. `test_translated.py` **105/106** — the one failure bit-identical at
  frobenius **0.1122406** to the pre-existing `{bfloat8_b, w_non_aligned}` pad-poison non-issue
  `feature_spec.INVALID` declares out of scope (recorded the same way in Refinements 2, 2b, 3, 4 and
  Perf 1).
- Unit suite **463 passed / 30 skipped**, including the two rel-RMS nets that caught D20's polarity,
  the pad-poison shapes, all three sharded schemes x both layouts, the RM BAND geometries and the
  `GRID_W` overrides that exercise D25's carved-out reader-fed path. **Zero hangs.**
- **Precision contract untouched**: `fp32_dest_acc_en`, `math_fidelity`, `math_approx_mode` and every
  dtype are exactly as the caller passed them. D20 and D22 both come out *more* accurate than the
  code they replace; D21 and D23 are bitwise/`torch.equal` identical; D24 and D25 change only when
  work is issued. **No option that traded precision for speed was graduated** — every subagent
  returned an option menu with per-option precision, and the fastest option meeting the contract was
  taken in each case.

### Instrumentation
`MaybeDeviceZoneScope` extended to every new path and never removed: `compute_root_sum` +
`compute_root_finalize` are replaced by one honest `compute_root_fused` zone (two zones where one
stage now exists would report a meaningless ~0 and mislead the next round), and the pipelined pass A
keeps its `compute_square` / `compute_reduce` zones so its overlap is still visible per block. The
ablation switches used for the cumulative peel (`RMS_ABLATE_ROOT_SUM`, `RMS_ABLATE_ROOT_FINALIZE`,
`RMS_ABLATE_GATHER_ZERO`) are committed **commented out**, with the peel recipe at their definition,
so the next round re-runs the classification instead of re-deriving it. New probe
`probes/zone_percore.py` — per-core zone closure, which is what made the root core's 63214-of-64221
arithmetic checkable.

## Perf 3 — the cross-core combine's LAYOUT and TOPOLOGY (a fan-out tournament)
- Date: 2026-08-05
- A **perf tournament**, not a refinement: `SUPPORTED` untouched, `EXCLUSIONS` untouched,
  `verify_supported` categories identical before and after. Six ideas were floated at the measured
  bottleneck, each fanned out to its own `blocking-perf-part-optimizer` with its own isolated
  on-device micro-benchmark under `perf_experiments/<slug>/` (all six artifacts committed).
  **All six measured; three graduated as three deviations (D26–D28), two were measured REGRESSIONS,
  one was superseded.**
- Round 3 of 3, so the breakdown was **re-measured on the now-instrumented, twice-optimized op**.
  That mattered twice: Perf 2's rank-3 stage had been promoted to rank 3 *by Perf 2 itself*, and the
  coordinator's own headline premise about the transport turned out to be **wrong** (below).

### Measured breakdown, and the ranked bottleneck
blackhole p150b, 110-core grid, CHIP_FREQ **1350 MHz** (== the reference clock, so no scaling);
bf16 / TILE / HiFi2 / `fp32_dest_acc_en=False` — the `_perf_case` config. One fresh-cache profiled
run per variant, **no trial loop**.

`feature_spec.py` still carries **no `attention:` note**, so the focus shape was free-selected as the
largest *measured* headroom — the same shape Perf 1 and Perf 2 took from 84836 to 34438:
**`(1,1,8192,1024)` BLOCK_SHARDED, shard `[1024,128]`, grid `(8,8)` = 64 cores — 34494 ns against a
25640 ns config-matched reference, 1.345x off**, the worst of the nine `_perf_case` targets. Every
knob it declares is in `SUPPORTED`; no generality gap, no proxy shape.

**Root-core arithmetic closure first** (`probes/zone_percore.py`, extended this round with a
`--has ZONE` filter — *without it the combine's ROOT core is unreachable*, because ranking by KERNEL
wall alone surfaces only MEMBER cores, whose wall is one long `cb_wait_front` on the root's stat):
the root's TRISC_0 zones sum to **32799 ns of its 33856 ns kernel wall** (gap 1057), so on that core
the zones *are* the wall.

| stage | ns (root core) | note |
|---|---|---|
| `compute_root_fused` | **13728** | root only; the D22 fold + its gather-arrival wait |
| `compute_scale` | 7842 | 22087 on a member — almost all `cb_wait_front` on the stat |
| `compute_square` | 5044 | all 64 cores |
| `compute_gamma_mul` | 4370 | all 64 cores |
| `compute_reduce` | 1813 | all 64 cores |
| `writer_mcast_send` / `writer_gather_zero` / `writer_gather_wait` / `writer_gather_ship` | 18280 / **7862 (n=1)** / 4429 / 2076 | root BRISC, sums to 32710 — the writer is busy for the whole wall |
| `reader_read_gamma` / `reader_read_x` | 1453 / **56** | NoC0 idle for ~31 us; native zero-copy shard CBs both ends |

**Cumulative peel** (payload stubbed, every CB reserve/wait/push/pop and trip count kept; peeled
cumulatively, not one at a time; pass A+B peeled with the eltwise family's own
`CKL_ELTWISE_CHAIN_SKIP_COMPUTE` switch rather than a hand-rolled flag):
full **34494** -> ablate `writer_gather_zero` **32032** -> + ablate the fused root chain **20735**
-> + ablate all three eltwise chains **16097**.

**Ranked, roofline-gated (`/perf-ceiling-dm`):**

| rank | stage | wall contribution | share |
|---|---|---|---|
| **1** | the fused root chain payload (fold + finalize) | **11297 ns** | **32.8%** |
| **2** | the combine's TRANSPORT + SYNC residual (gather ship/wait + mcast round trip + CB handshake floor) | **16097 ns** | **46.7%** |
| **3** | `writer_gather_zero`, the one-time boot | **2462 ns** | 7.1% |
| gated OUT | the three eltwise chains (`square` + `scale` + `gamma_mul`) | 4638 ns | **AT the FPU roofline**: 384 tile FPU ops / 4638 ns = **12.1 ns per 32x32 bf16 HiFi2 broadcast multiply**. No idea was spent here, and that gating held up — see "what this round did to itself". |
| gated OUT | `reader_read_x` / `writer_write` | 120 ns | native sharding is a **precondition** of this pass, not a lever |

Two things only the *cumulative* peel could show. Removing the root chain alone would have been read
against a 13728 ns zone, but the zone **contains the gather-arrival wait**; the peel prices the
*payload* at 11297. And `writer_gather_zero`, which Perf 2 correctly ranked *out* at 1063 ns of wall,
now measures 2462 — **Perf 2's own graduations exposed it**, exactly as Perf 2's changelog predicted.

**Three sub-findings that shaped the portfolio:**
1. The ROOT core is **compute-saturated** (32799 of a 33856 wall, no idle) while every MEMBER idles
   ~22 us (`writer_mcast_recv` 25859). **The wall IS the root's serial work** — so *rebalancing off*
   the root was floated as a first-class idea alongside *making the fold cheaper*.
2. A partial is a `REDUCE_ROW` result, so at `GATHER_FACES=2` `ship_partial` issued **2 NoC writes of
   1024 B per tile-row = 16 writes / 16 kB per member per round to carry 1 kB of information** — 16x
   byte amplification.
3. `rms_norm_writer.cpp`'s comment claiming a slot-major gather is "a gapped window no chain walk can
   express" is **STALE**: it described the pre-D22 `eltwise_chain`, and D22's raw fold takes
   **explicit tile indices**. Two subagents independently confirmed this.

### The portfolio (6 ideas; overlap and fusion deliberately allowed)
| idea | verdict | measured | domain |
|---|---|---|---|
| `compact_partial_transpose_r3` — one-hot matmul packs BLOCK_ROWS partials into ONE tile | **WIN** | combine bench **34772 -> 10996 ns (3.16x)**; root chain 3024 -> 770 ns/round; L1 288 -> 88 kB/core; 1.01x–9.66x over BR{1..32} x G{4,8,9,28,32} | everywhere on COMBINE, three measured carve-outs |
| `slot_tree_gather` — a k-ary tree over the SLOT axis at m=1, built to COMPOSE with compaction | **WIN (wide groups), NULL (focus)** | w5120 G=32 **5424 -> 3744 (1.45x)**; w7168 G=28 1.40x; focus G=8 **1.00x** | wide groups only; two measured-regression carve-outs |
| `gather_zero_elim` — is the boot zeroing needed at all? | **WIN** | stage **9900 -> 0 ns**; 2462 ns / 7.1% of wall; 9 catastrophic seeds all bit-identical | everywhere, minus the odd-G pad page (correctness) |
| `gather_dual_noc` — split the gather across the idle NoC0 | **WIN, superseded** | `sf` **30147 -> 27683 (1.089x)**; 1.154x/1.158x at G=28/32 | superseded by D27, which deletes the per-row face writes `sf` splits |
| `gather_slot_major_coalesce` — ONE contiguous NoC write per member | **REGRESSION** (+ a WIN found in the slot) | 0.931x focus, 0.782x–1.011x everywhere; by-product `dboot` 1.082x | do not ship; `dboot` superseded by D26 |
| `root_rotation` — rotate the root per block to spread the fold | **REGRESSION** | **0.912x** focus; 0.854x–0.998x across ALL 14 geometries, no cell >= 1.00x | correct everywhere, slower everywhere |

**Four WINs and two REGRESSIONs, and the two negative results are the most valuable things in the
round**, because each refuted a premise the coordinator had written into the brief:

- **`gather_slot_major_coalesce` refuted the coordinator's own rank-2 mechanism.** The brief asserted
  the gather was small-transfer bound at ~13 GB/s and that coalescing 16 writes into 1 would pay. The
  subagent built the decisive control — `rm_f4` vs `sm_f4`, **identical bytes, BLOCK_ROWS transactions
  vs ONE** — and measured it **flat within ±1% across 11 geometries**. Transaction count is *not* the
  constraint; the root's L1 ingress **bytes** are (D13's bound again). So compaction wins because it
  moves **8x fewer bytes**, not because it issues fewer writes. A slot aimed at a wrong premise still
  paid: the same bake-off found the `dboot` lever at 1.082x, which D26 then superseded by deleting the
  stage outright.
- **`root_rotation` refuted the coordinator's first-order rebalance arithmetic**, and refuted it *by
  succeeding at what it set out to do*. The rebalance worked — the fold moved (13400 ns n=4 on one
  core -> 7476 n=1 on each of four) and per-core KERNEL times **equalized** (33.6/34.1 ->
  37.3/37.4) — but equalized **above** the fixed root's level. Mechanism: **D24** makes the fixed root
  a full broadcast latency *ahead* of its group every round, so its fold runs in slack the group
  spends on gather + broadcast (visible as the root's 4449 ns `writer_gather_wait`). A rotating root
  is by construction a core that *did* wait, and must serially wait out every earlier round's
  `writer_mcast_recv` (18977 ns, n=3) before it can publish. **Rotation cannot shorten a serial chain
  by moving a link.** The penalty grows monotonically with round count (0.968x@nb1 -> 0.854x@nb16),
  which is the signature of exactly that mechanism.

### Aggregation — what conflicted with what
- **`gather_dual_noc`'s `sf` is SUPERSEDED, not stacked.** It splits the *two face-writes per
  tile-row* across NoC0/NoC1; D27 replaces those with ONE whole-tile write per round, so there is
  nothing left to split. Its own author flagged the composition as expressible-but-unmeasured. Not
  double-counted.
- **`gather_slot_major_coalesce`'s `dboot` is SUPERSEDED.** It *distributes* the boot zeroing across
  cores; D26 *deletes* it. Deleting strictly dominates distributing.
- **`compact_partial_transpose` and `slot_tree_gather` are NOT mutually exclusive** — which
  **refutes what Perf 1 and Perf 2 both recorded.** That exclusivity belonged to hierarchical
  gather's ROW-SPLIT half, which parallelises over the row axis compaction removes. The SLOT tree
  parallelises over `GROUP_SIZE`, an axis compaction does not touch. Their measured domains turn out
  to be **disjoint**: compaction wins at `BLOCK_ROWS > 1`, the tree at `BLOCK_ROWS == 1` with
  `GROUP_SIZE >= 30`. Both graduated, sharing one code path.
- The three that graduated are mutually independent: the gather CB's defined-ness, the partial's
  layout, and the fan-in's topology.

### What graduated, how widely, and what was deleted
Three deviations. **Every one is a single unqualified path** for the domain it is correct on, and each
deleted the code it replaced.

- **D26 — the gather CB's FACE boot-zeroing is DELETED** (rank 3). Perf 2's changelog named this stage
  "the honest next target" and recorded that its own argument for removing it was *"reasoned, not
  measured"*. It is now measured. An isolated bench seeds faces 1 and 3 of every gathered page with
  **nine** catastrophic patterns — 1e30, -1e30, NaN, ±Inf, fp32 subnormals, a per-lane mix that makes
  the fold evaluate `Inf + (-Inf)`, and a stale-L1 lookalike — then runs the op's **real** D22 fold +
  D17 finalize + pass B's exact column-broadcast consumer. **Every seed came back BIT-IDENTICAL
  (`torch.equal`)** to the boot-zeroed run, at `GROUP_SIZE` 4/8/9/28/32 x `BLOCK_ROWS` 1/8/32.
  **The CONTROL is what makes it a proof rather than an accident:** the packed stat tile's columns
  16..31 came back **100% non-finite** for the NaN/Inf seeds and |max| 7.96e30 for the 1e30 seeds. The
  garbage *does* enter DEST (the FPU has no lane scope), *does* survive the pairwise `acc_to_dest`
  fold, and *is* multicast to every member — it is **carried and never read**.
  **34494 -> 31803 (1.085x)** on the focus shape and **6266 -> 5406 (1.159x)** at WIDTH 32c, output
  bit-identical.
- **D27 — the COMPACT partial** (ranks 1 and 2 at once). A sender's whole row-block travels as **ONE
  tile** whose column *r* is tile-row *r*'s partial sum, via a one-hot `matmul_tiles` column
  permutation; the root's fold becomes **ONE DEST window independent of `BLOCK_ROWS`**; the multicast
  carries one tile; the receiver un-permutes with the *same* bank read through matmul's srcB
  `transpose` flag. The one-hot bank is synthesized **on device by the reader**
  (`reader_bank_boot`, 1461 ns one-shot, off the critical path) rather than plumbed in as a host
  tensor, and bf16 is exact for a one-hot so it costs no accuracy.
  **Deleted:** `ship_partial`'s per-tile-row loop and its row-major landing arithmetic, the root
  fold's per-tile-row window loop, the `GATHER_SLOTS * BLOCK_ROWS` term from **both** the CB
  allocation and the L1-bound `block_rows` solve, the pad-zeroing walk over that same product, and
  **`cb_row_stat`'s allocation on the combine path** (D22 left it strictly dead there — 256 kB/core at
  `block_rows` 32). Removing the `GROUP_SIZE x BLOCK_ROWS` term from the L1 solve is *itself* a lever:
  it lets the solve take a coarser block (8 -> 20 rows, 4 rounds -> 2), worth **1.066x on top of**
  compaction's own 1.235x. **31803 -> 24179 ns (1.321x)**; `compute_root_fused` 13728 -> **1464 ns**.
- **D28 — the two-level SLOT TREE** for wide groups. `f0 = 4` (measured: `4x8` 1.45x vs `8x4` 1.38x),
  **two levels always** (depth 3+ lost at 6 of 7 bench cells, so no depth knob was added — shipping
  one would re-propose a measurement-refuted shape), interior nodes provably do not finalize
  (a `FINALIZE` template arg). **Deleted:** `GATHER_HALF` and the ~45-line inline root chain, lifted
  into ONE `combine_fold<...>` template with three instantiations (flat root, tree L0, tree L1).
  **WIDTH 32c 5376 -> 5046 (1.066x)**, and interleaved width-split at `gw=56` 9002 -> 8256 (1.090x).

**The five earned carve-outs, and the measured reason for each:**
1. **D26's odd-`GROUP_SIZE` PAD page still gets zeroed** — *correctness*. A pad page is folded
   **whole**, so its faces 0/2 land in column 0 and must be an exact +0.0. Measured catastrophic
   without it: `pad = 1e30` gives **rel-RMS 1.00 at pcc 0.999672** — this op's signature pcc-blind
   uniform-scale failure — and `pad = NaN` gives rel-RMS 1.00. Poisoning only the pad's columns 16..31
   is bit-identical, so only its faces 0/2 are load-bearing, **but** zeroing just those two faces is a
   measured **10–11% regression** (1959 vs 1781 ns at G=9): this stage pays per **API call**, not per
   byte (`async_write_zeros` sets state up once and chunks at `MEM_ZEROS_SIZE = 512`). Whole page.
2. **D27 elides the permute pair at `BLOCK_ROWS == 1`** — *inexpressible* (the map is
   `partial_0 x E_0`, the identity) **and** a measured regression with the identity permutes left in:
   0.76x / 0.80x / 0.76x / 0.91x on the four WIDTH decode targets. Spelled once, as
   `COMPACT = CROSS_CORE && (BLOCK_ROWS > 1)`.
3. **D27 keeps D13's two-face gather at `BLOCK_ROWS == 1`** — with no un-permute matmul, nothing needs
   the unshipped faces defined, and a whole-tile ship doubles the root's ingress bytes. The regression
   is **monotone in `GROUP_SIZE`** (0.859x at 32c, 0.908x at 28c) — the fan-in multiplier's own
   signature. `GATHER_FACES` is therefore **scoped, not deleted**.
4. **D27 caps `BLOCK_ROWS <= 32` on the combine path** — a compact tile has 32 columns. Found by
   **measurement, not reasoning**, once the L1 term left the solve: `(1,1,3232,96)` WIDTH solved to a
   **101-row** block and returned **pcc 0.949109 / rel-RMS 0.31**.
5. **D28 is FLAT below its threshold** — `f1 < 2` (`GROUP_SIZE <= 4`: the only legal tree deletes zero
   fold tiles and pays a pure hop, 0.78x–1.02x) and `deleted < 18` (`GROUP_SIZE <= 29`).

**Nothing was guarded on a suspicion.** The `gw=32` interleaved width split is a measured **0.998x
NULL** (a DRAM-bound pass A absorbs the combine saving) and is left **on the unified path**; so are
every flat regime in D26 and D27.

**D28's threshold did not survive re-derivation, and that is the round's cleanest methodological
result.** The isolated bench's rule said `>= 17`; against the post-D26 tree it is `>= 18`, and the
crossover is bracketed by **adjacent measured values** rather than extrapolated — a dedicated 30-core
width shard, `(1,1,32,4800)`, was built purely to get the 18 point:

| G | f1 | deleted | flat | tree | ratio |
|---|---|---|---|---|---|
| 8 | 2 | 2 | 3729 | 4307 | 0.866x |
| 9 | 3 | 2 | 4481 | 5040 | 0.889x |
| 8 (BLOCK 64c) | 2 | 2 | 24181 | 25244 | 0.958x |
| 28 | 7 | **17** | 5717 | 5881 | **0.972x** |
| 30 | 8 | **18** | 5130 | 4991 | **1.028x** |
| 32 | 8 | 20 | 5376 | 5004 | 1.074x |

And **why the isolated 1.40–1.45x became 1.03–1.08x is arithmetic, not noise**, recorded at the
constants: (i) the bench's flat baseline still paid the per-face boot-zero calls that **D26 has since
deleted**, so most of what the bench credited to the tree had already been banked by a *sibling
graduation*; (ii) the combine is only ~1.5–2 us of a 5–5.4 us op sitting on a ~3.5 us one-core
dispatch floor. **A 1.4x on a stage is not 1.4x of a wall**, and a tournament that reports the
isolated number as the op's number is lying.

**Raw-LLK bypasses — one new family, D27's**, with its measured authorisation at the definition:
`matmul_tiles` / `matmul_init` used as a **column permutation**. The eltwise / bcast / reduce families
all preserve or collapse the column axis and `transpose_wh` transposes the whole tile, so the matmul
is the FPU's only horizontal-mixing primitive; srcB `transpose` lets ONE bank serve both directions
(`C = partial_r x E_r` packing, `C = compact x E_r^T` un-packing), and DEST accumulation is free
(`matmul_tiles` is `DST += A*B`), so `rows` matmuls cost one pack. Recorded alongside: **`matmul_init`
does NOT reconfigure data formats**, so `reconfig_data_format<SrcOrder::Reverse>` is emitted at every
matmul site — the same class of omission that in Perf 2 produced a uniform ~1000x scale error holding
pcc at 0.9997. D28 adds **no** new bypass: `combine_fold` carries D22's raw pairwise `add_tiles` fold
and D17's raw-sfpi finalize verbatim, with their justifications relocated to the template head.
A **safety invariant** any later refactor must preserve: the permutation matmul sums 32 products, so
**every column of both operands must be finite** — an Inf/NaN in an unused column becomes `Inf*0 =
NaN` and poisons column 0. That is exactly why a compact page is shipped **whole**, and it is what
makes D26's face-zeroing unnecessary on the compact path *by construction* rather than by D26's
never-read argument. A related **untested** hazard is documented at the kernel site rather than
guarded: at `epsilon == 0` the compact stat's unused columns would be `rsqrt(0) = +inf` and the
un-permute's `inf*0` would return NaN everywhere, where the flat path degraded only on an all-zero
row. No `epsilon` axis exists in `SUPPORTED`, the default is 1e-6, and the suite runs 1e-12…1e-2.

**THE ROUND'S SHARPEST HAZARD, and it is a correctness change, not a perf one.** D27's finalize must
run at a **wider VectorMode scope**. D17's shipped `<STRIDE=2, ITERS=4> VectorMode::C` reaches only
columns 0,2..14 — right for a stat living in column 0, and **silently wrong on a compact tile from
`BLOCK_ROWS = 2`**, where the odd rows' stats are never scaled and never rsqrt-ed. Measured
**pcc 0.9972987 with rel-RMS 1036** against a 0.04 bound: **the fourth bug in this op that pcc alone
would have waved through.** The compact fold gets `<1,8> C` (`BLOCK_ROWS <= 16`) / `<1,8> RC` (above);
**D17's narrow scope STAYS** for the local non-combine finalize, which really does own only column 0.
Two scopes, two call sites, each with its measured justification at the definition, and the bench
keeps the hazard as a **live assertion** (`test_finalize_scope_hazard`) that fails if the narrow scope
ever starts passing.

**A fifth instance of this op's pcc-blind pattern, found in passing and worth recording as a policy:**
`gather_slot_major_coalesce`'s deliberate falsification (writer slot-major vs compute row-major)
measured **pcc 0.999564 / rel-RMS 2.892e-02 — INSIDE both the 0.9995 pcc gate AND the 0.04 rel-RMS
bound.** Only **bit-exactness against the baseline** caught it. Any future gather-layout change should
be gated on bit-exactness, not on pcc/rms.

### Whole-op before/after (one fresh-cache profiled run per variant; guard set = the `_perf_case` table)
| target | reference | Perf 2 after (= Perf 3 before) | Perf 3 after | speedup | vs reference |
|---|---|---|---|---|---|
| **`(1,1,8192,1024)` BLOCK 64c** (focus) | 25640 | 34494 | **24307** | **1.419x** | 1.345x off -> **0.948x, BEATS ref** |
| `(1,1,32,5120)` WIDTH 32c | 5267 | 6266 | **5046** | **1.242x** | 1.190x -> **0.958x, BEATS ref** |
| `(1,1,32,7168)` WIDTH 28c | 5481 | 6112 | **5703** | **1.072x** | 1.115x -> 1.041x |
| `(1,1,32,2304)` WIDTH 9c | 4617 | 4530 | **4463** | 1.015x | 0.981x -> **0.966x** |
| `(1,1,32,1024)` WIDTH 8c | 4110 | 3719 | **3680** | 1.011x | 0.905x -> **0.895x** |
| `(1,1,8192,1024)` interleaved | 96744 | 89132 | **88795** | 1.004x | **0.918x** |
| `(1,1,8192,2304)` interleaved | 211345 | 192736 | 194635 | 0.990x (noise) | **0.921x** |
| `(1,1,8192,5120)` interleaved | 738307 | 413370 | 413383 | 1.000x | **0.560x** |
| `(1,1,8192,7168)` interleaved | 1032281 | 569533 | 577288 | 0.987x (noise) | **0.559x** |

**Guard-set no-regression: no target regressed** — one representative per distinct kernel path x
layout x placement (BLOCK / WIDTH / interleaved, root, member, tree interior node, combine and local
finalize, `native_in` and reader-fed, even and odd `GROUP_SIZE`, compact and identity paths).
**Eight of the nine now beat their reference outright**, where Perf 2 left six. The two interleaved
cells at 0.990x / 0.987x sit inside the ±2–3% noise band **and run untouched code** — the combine is
OFF there (`group_size == 1`); their measured run-to-run spread across six profiled runs this round
was ±2–4%. The one target still off its reference, WIDTH 28c at 1.041x, is off because D28's rule
**correctly** keeps it flat (`deleted = 17 < 18`, a measured 0.972x for the tree there).

Across three tournaments the focus shape has gone **84836 -> 64677 -> 34438 -> 24307 ns**, a
cumulative **3.49x**, from 3.31x off its reference to beating it.

### Final per-stage state on the focus shape (24307 ns), and what this round did to itself
Root core TRISC_0: KERNEL 23717, stages sum 22898 (gap 819).

| stage | Perf 3 before | after | note |
|---|---|---|---|
| `compute_root_fused` | 13728 | **1464** | D27 — **9.4x**; no longer the bottleneck, no longer even top-4 |
| `writer_gather_zero` | 7862 zone / 2462 wall | **absent** | D26 — a literal no-op at even `GROUP_SIZE` |
| `compute_square` | 5044 | **6935** | **now rank 1** |
| `compute_scale` + `compute_gamma_mul` | 12212 | **8303** | now rank 2 |
| `compute_recv_unpack` + `compute_member_pack` | 0 | **4598** | D27's new cost, paid in PARALLEL on all 64 cores |
| `compute_reduce` | 1813 | 1597 | |

**The critical path has left the combine, and this round put it where it is.** The op's rank-1 stage
is now pass A's `compute_square`, with pass B second — **precisely the stages this round
roofline-gated OUT of its portfolio at 12.1 ns per 32x32 bf16 HiFi2 broadcast multiply.** That gating
was right (no idea was wasted on them) and it is now the binding constraint: the combine, which was
79.5% of the wall in the breakdown above, is down to ~25% including D27's new permute pair. A Perf 4
would have to attack the elementwise FPU work itself — a different kind of problem, at or very near
its roofline, where the levers are the *shape* of the arithmetic (fewer passes over x, or a cheaper
`x^2` datapath) rather than the movement around it.

### Deferred, with their numbers intact
There is no Perf 4 scheduled, so deferring has a real cost and it is recorded as such.
1. **`gather_dual_noc`'s `mc` / `mcs`** — moving the stat multicast to the idle NoC0 was **1.448x /
   1.363x on the transport** but **0.912x / 0.917x on the wall**, because the multicast was hidden
   behind the root chain. Its author's exact note was *"re-measure if a sibling shrinks the root
   chain"* — and D27 shrank it **9.4x**. This is the most likely live lever left in the combine and it
   was **not** re-measured after D27 landed. Also recorded there: NCRISC issues NoC work ~3 us slower
   than BRISC here (`z`, the gather-zero move, was a regression even in the transport-only
   measurement at 0.860x), and the `SPLIT_ROOT=0` knob is load-bearing when composing the two levers
   (0.977x -> 1.426x).
2. **One combine round at the focus geometry** — L1-infeasible by a **measured** 48 kB once pass A's
   and pass B's CBs are counted. Two named, unmeasured levers could close it: `_cb_block_mult` counts
   `cb_x_squared` as `block_rows * wt_chunk` pages while D12's DEST fold allocates only `block_rows`
   (a pre-existing over-count), and `cb_row_final` / `cb_sum_handoff` are `CB_ROW_STAT_DEPTH *
   block_rows`.
3. **D27's 12-row ragged tail** pays the `RC` finalize scope because the cap is decided from
   `BLOCK_ROWS` (20 > 16) rather than the round's actual `rows`. Correct (a superset), measured flat
   (~452 ns/round on the short round), not optimized.

### Correctness (green throughout; counts identical to Perf 2)
- Golden: **5541 passed / 1365 xfailed / 33921 skipped**; cartesian **5037 pass / 1365 xfail**,
  `supported_fail = 0`; loose **384 pass / 3 infeasible-skip / 0 xfail**.
- `test_regression.py` **15/15**. `test_translated.py` **105/106** — the one failure bit-identical at
  relative Frobenius **1.122406e-01** to the pre-existing `{bfloat8_b, w_non_aligned}` pad-poison
  non-issue `feature_spec.INVALID` declares out of scope (recorded the same way in Refinements 2, 2b,
  3, 4 and Perf 1 and 2).
- Unit suite **463 passed / 30 skipped**, including the two rel-RMS nets that caught D20's polarity,
  the pad-poison shapes, all three sharded schemes x both layouts, the RM BAND geometries and the
  `GRID_W` overrides that exercise D25's carved-out reader-fed path. **Zero hangs.**
- **Precision contract untouched**: `fp32_dest_acc_en`, `math_fidelity`, `math_approx_mode`,
  `dst_full_sync_en` and every dtype are exactly as the caller passed them. **No option that traded
  precision for speed was graduated** — every subagent returned an option menu with per-option
  precision, and the fastest option meeting the contract was taken in each case. D26 is bit-identical.
  D28 is **more accurate** (rel-RMS 0.0074483 -> 0.0067687 at WIDTH 32c, never worse across
  bf16/HiFi2/acc=F and fp32/HiFi2/acc=T). D27's *stat-level* rel-RMS is 1.7x worse in isolation (two
  extra roundings through a 16-bit DEST word at `fp32_dest_acc_en=False`, understood not mysterious,
  and 4 orders inside the 0.04 bound) — but **at whole-op scale accuracy IMPROVED**: focus rel-RMS
  0.006648 -> 0.006209, `(1,1,3232,96)` 0.006459 -> 0.006020, and the four WIDTH geometries are
  **bit-for-bit identical** (identity path). Recorded as an option cost, not hidden.

### Instrumentation
`MaybeDeviceZoneScope` extended to every new path and never removed: `compute_member_pack`,
`compute_recv_unpack`, `reader_bank_boot`, `compute_tree_fold_l0`, `writer_tree_forward`,
`writer_tree_wait`. `writer_gather_zero` survives as a zone even though it is a no-op at even
`GROUP_SIZE`, because the odd-`GROUP_SIZE` pad path still runs there and a retired zone would hide it.
The ablation switches (`RMS_ABLATE_ROOT_SUM`, `RMS_ABLATE_GATHER_ZERO`) are committed **commented
out**, still compile against the compact and tree paths, and now cover the tree's rings, with the peel
recipe at their definition. `probes/zone_percore.py` gains a **`--has ZONE`** filter — the single most
load-bearing tooling change of the round, because ranking cores by KERNEL wall surfaces only the
combine's idle MEMBER cores and the ROOT (the core the critical path actually runs on) is otherwise
unreachable.
