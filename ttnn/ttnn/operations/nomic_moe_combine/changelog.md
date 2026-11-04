# Changelog: nomic_moe_combine

## Phase 0 — Core Implementation
- **Date**: 2026-11-04
- **What was done**: Initial implementation via the incremental pipeline (planner → implementer → verifier). Regime
  `owner_computes`: each core of the output's ND-shard grid computes its own shard and packs it in place (zero-copy
  output CB, no writer, no NoC writes). Choice 0 rows are gathered on RISCV_0/NOC_0 and choice 1 on RISCV_1/NOC_1 with
  stateful one-packet reads in 8 x 2 trid batches. Compute realization R2: fast tilize to a CB, then
  `mul_tiles_bcast_cols` accumulated in fp32 DEST on a bias row-broadcast seed, with one bf16 pack.
- **SUPPORTED at Phase 0**: dtype=[bfloat16], rows_buffer=[l1], output_layout=[norm_layout], window=[aligned, ragged],
  batch=[single, multi]
- **Accuracy achieved**: PCC=0.9999986, max_abs_err=8.6e-3, mean_abs_err=6.6e-4, rel_rms_err=0.00167–0.00168, 98.8% of
  values within 1 bf16 ULP of the correctly rounded value (92% exact). Measured on 4 shapes (1x1000, 2x512, 4x512,
  8x512) via test_nomic_moe_combine_precision_baseline.py; this matches the incumbent's one-rounding calibration.
- **Perf at Phase 0** (device kernel, µs): 8x512 14.7–14.9 (accept 20.0, stretch 13.0, incumbent 32.22); 4x512 8.6–8.8
  (12.5 / 8.0 / 19.33); 2x512 6.8–7.3 (8.5 / 5.5 / 12.88).
- **Golden suite at Phase 0**: 28 / 98 cells passing, 70 xfail_expected, 0 supported_fail / xpass_drift /
  xfail_wrong_mode (per `verifier_report.json`)
- **Issues encountered** (verifier fixes): compute kernel's `top_k` literal now a CT arg from host `TOP_K`; CB sizing
  now has one source (`circular_buffer_specs`), shared by the descriptor and `circular_buffer_bytes()` / the budget
  check; validate() now enforces the reader's transaction-id cap on shard height (`BATCH_DEPTH + 3 + r ≤ 15`), which
  was a compile-time-only assert before; l1_ledger.md corrected (8 x 2 reads in flight, trid bound, R2 totals) and the
  `group_shared_index_weights` deferral flagged as falsified by measurement (filed as Refinement 2).
- **Tests added**: test_nomic_moe_combine.py (acceptance, planner), test_nomic_moe_combine_perf_sweep.py (implementer),
  test_nomic_moe_combine_precision_baseline.py (verifier)

## Refinement 2 — Speed up the perf-flagged 8x512 profile: share the index and weight reads across each row group
- Date: 2026-11-04
- What was done: built regime layer `group_shared_index_weights` on `owner_computes`. For each choice k, the row
  group's leader (member 0; the knob `STAGGER_LEADERS` moves choice 1's leader to member G-1) reads the index slice and
  the weight faces 0/2 from DRAM. As each lands, it unicasts the bytes into the peers' existing `cb_index_choice{k}` /
  `cb_weights_raw_choice{k}` at the same L1 offsets. Once those writes are acked, it bumps each peer's semaphore k by
  the items delivered. A peer's reader k does no index or weight DRAM reads: it waits for count ≥ 1 for the index and
  pushes weight tile t once count ≥ 1 + t + 1. The reader polls this forwarding and signalling inside its row-read
  waits and its final drain, not only at batch boundaries. Without that, 2x512 jittered to 9.3 µs because the peers'
  weights arrived only after the leader's drain. Weight id checks are now gated per tile on a row batch behind them
  (`weights_safe`). Host: knobs `SHARE_INDEX` / `SHARE_WEIGHTS` (default True; no-op at G = 1), two
  `SemaphoreDescriptor`s, per-choice reader runtime args carrying the peers' NoC coordinates, CT args 17–21. No CB was
  added and no SUPPORTED change. Reused: the reader's stateful row path, the CBs, the `poll_side_reads` push
  machinery. Added: the forward/signal lambdas, the peer semaphore-driven pushes, the sweep variants.
- Perf (device kernel µs, one fresh-cache run per variant, 3 sessions): 8x512 14.89 → 12.8–13.6 (median ≈ 13.0;
  `share_none` 14.8–15.0 in the same sessions); 4x512 8.80 → 8.3–8.55; 2x512 6.88 → 6.8–7.5 (neutral). Weights-only
  13.5–13.6, index-only 14.4–14.7 at 8x512; `stagger_leaders` neutral.
- Accuracy achieved: unchanged. Golden tolerances (PCC ≥ 0.999995, rel RMS ≤ 0.0019, one-hot exact) pass on all
  acceptance shapes (1x1000, 2x512, 8x256, 4x512, 8x512, 1x4000) and in the precision baseline.
- Golden test progress: 28/98 passing (test_golden.py norm_layout/l1 slice 17/17, test_regression.py 11/11); the 70
  deferred placement cells stay xfail_expected.
- Issues encountered: the first build signalled peers only at the leader's batch boundaries. This was fixed by
  polling inside the read waits (above).
- Tests added: perf sweep variants `share_none`, `share_index_only`, `share_weights_only`, `stagger_leaders` in
  test_nomic_moe_combine_perf_sweep.py.

## Refinement 3 — Speed up the perf-flagged 8x512 profile: per-tile compute cost and compute/read overlap
- Date: 2026-11-04
- What was done: profiled the compute block surface at 8x512 (ablation + per-sub-stage device zones) and turned the
  winning knob. **Math fidelity HiFi4 → HiFi3** (new op knob `MATH_FIDELITY`; validate() accepts
  `PRECISION_FIDELITIES = (HiFi4, HiFi3)`). The multiply is FPU-bound per fidelity phase (~35 cycles a tile), and HiFi3
  drops only the rows-low x weights-low partial product. Levers built as live knobs, measured, and parked at their
  old, byte-identical defaults:
  - L3 `DST_FULL_SYNC` (default config's `dst_full_sync_en`): 8-tile DEST groups, but the pack stops overlapping the
    math. 15.6 µs at HiFi4, 14.8 at HiFi3.
  - L4 `BIAS_SEED_SRCB` (compute CT arg 13): seeds the bias through srcB (raw `llk_unpack_A` + B2D row-broadcast
    datacopy, the calls `unary_bcast` uses for 16-bit CBs; exact, the bias is bf16-valued) instead of `unary_bcast`'s
    unpack-to-dest switch. Neutral.
  - DEST group 2: neutral. R1 at HiFi3: 28.9 µs.
  Probes measured and not kept:
  - `matmul_block` diag(w)·R in place of the two `mul_tiles_bcast_cols`: ≈ equal.
  - The bias as a third K tile of that matmul (no seed, one init a tile row): 12.6 µs, worse; a tile matmul costs more
    than the seed.
  Added sub-stage zones (`NMC-C-TILIZE/BIAS/MUL/PACK`, compiled only under `NOMIC_MOE_COMBINE_ABLATE=ZONES`).
  `ABLATE=ZONES,NOCZONES` keeps the reader zones and drops the compute ones, so compute timing is not perturbed. Reused:
  the R2 compute path, the CBs, the sweep harness. Added: 3 op knobs, 1 compute CT arg, sweep variants.
- Perf (device kernel µs, one fresh-cache run per variant, 2–4 sessions): 8x512 13.0–13.5 → 11.8–12.2; 4x512 8.4 →
  7.7–8.0; 2x512 7.0–7.2 → 6.5–7.2 (noise band). All under the 20.0 / 12.5 / 8.5 accept bars; the 13.0 stretch is met
  at 8x512.
- Bottleneck now (reader-only zones, 8x512): tile row 0 lands ~6.9k cycles after launch, then the compute chain runs 4
  tile rows x ~2.3k cycles. Reads deliver at the same ~2.3k-cycle pace, and the last push is ~2.4k cycles before the
  compute ends. With the reads stubbed, a tile row costs ~1.6k cycles (in-situ L1/NoC contention). At HiFi3, stubbing
  the multiply saves ~1.4 µs, tilize ~0.85 and the bias seed ~0.8 (median of 3). The seed's cost is its existence and
  init churn per DEST group, not its path.
- Accuracy achieved: PCC=0.9999986, rel RMS=0.00168 (both unchanged vs HiFi4), max_abs 8.1e-3–8.6e-3, 90.1–90.3%
  ULP-exact and 98.44–98.47% within 1 ULP (HiFi4: 92% / 98.8%) on shapes [1x1000, 2x512, 4x512, 8x512]. One-hot exact,
  padding = bias and bias multiplicity pass on all acceptance shapes. HiFi2 measured rel RMS 0.0036 (fails 0.0019).
- Golden test progress: 28/98 passing (test_regression.py + test_golden.py norm_layout slice: 28 passed, 14
  xfail_expected); the 70 deferred placement cells stay xfail_expected.
- Issues encountered: in-situ compute is ~40% slower than with the reads stubbed. Single ablations are noisy at
  ±0.5 µs, so I used medians of 3.
- Tests added: perf sweep variants `full_sync`, `hifi3`, `hifi4`, `hifi2` (measurement only, fails precision),
  `full_sync_hifi3`, `bias_srcb`, `hifi3_bias_srcb`, `hifi3_dest2`, `hifi3_r1`, `hifi3_full_sync_bias_srcb`.

## Refinement 4 — Host time per call: cache the descriptor, patch only the addresses
- Date: 2026-11-04
- What was done:
  - **Call-state cache** (`_CALL_CACHE`). Key: (T, B, rows shape, rows dtype, rows buffer type,
    repr(memory_config), compute config fields, device id). It holds the plan, the output shape, the resolved
    compute config, and the whole ProgramDescriptor, which is built on the key's first call. validate() is now
    `_validate()`, still the entry point's first line, split two ways:
    - Cached per key: axes / SUPPORTED / EXCLUSIONS, domain, grid, compute config, plan, CB budget.
    - Run every call: `_check_tensors()`, the layout / dtype / shape / buffer-type checks on the four inputs.
  - **Per-call binding** (`bind_call()`): writes the four input addresses into the readers' common args 0..3 in
    place, and rebinds the output shard's CB with `set_buffer_from_cb(cb_descriptor_from_sharded_tensor(output))`. A
    cached descriptor never dispatches with a previous call's addresses.
  - **No per-core runtime args.** ttnn.generic_op re-copies and re-sends every per-core runtime arg on each call. On
    empty kernels at 110 cores: 21 us with none, 49-57 us with 4-26 a core. Common args cost almost nothing, even at
    224 words.
    - The readers' common args are now [4 addresses, N core records, N NoC xy]. The compute's are [N core records].
    - Record i packs `tr0 | core_row_tiles | g | bias_copy` at `CORE_RECORD_FIELD_BITS` = 8 bits a field
      (`pack_core_record`, host-side single source; the kernels take the field width as a CT arg).
    - A kernel finds its record as core i = y * 11 + x. `_plan` refuses a shard grid that is not the grid's first
      cores in row-major order (the norm layout always is).
    - The leader computes its peers' NoC coordinates from the group's index range.
    - New CT args: reader 22-24 (grid_x, num_cores, field bits; accessors now start at 25), compute 14-15.
  - The perf sweep also resets `_CALL_CACHE` when it patches a knob.
  - Reused: the plan, the descriptor builder, the kernels' schedules (only their arg reads changed). Added:
    `_CALL_CACHE`, `bind_call`, the core tables.
- Host time (50 calls after warm-up, time until the call returns, median):
  - Before: 449 / 446 / 335 us at 8x512 / 4x512 / 2x512. The 212 us in the requirement was measured on nomic-opt;
    this box is slower.
  - Descriptor cached only: 86 / 75 / 58 us.
  - Final: 32.3 / 32.0 / 27.8 us (target <= 60 met).
  - Breakdown at 8x512: generic_op 7.5 us (53.5 with per-core args), allocate_tensor_on_device 11.5, validate (hit)
    8.1 (tensor checks 5.2), bind_call 1.6.
- Device kernel time (`--profile` perf sweep baseline, us): before 11.87 / 12.36 / 11.87, 7.90 / 7.84 / 7.85,
  6.52 / 8.00 / 8.05. After 11.85-12.45, 7.82-7.91, 6.29-8.25. Unchanged within noise; 2x512 is bimodal in both.
- Accuracy achieved: unchanged. Same compute path, and the golden tolerances (PCC >= 0.999995, rel RMS <= 0.0019)
  hold on [1x1000, 2x512, 8x256, 4x512, 8x512, 1x4000] plus every golden norm_layout / L1 shape.
- Golden test progress: test_regression.py + test_op_loose 14/14, test_op norm_layout / rows-in-L1 slice 14/14
  (includes test_new_tensors_between_calls). The 70 deferred placement cells are untouched.
- Issues encountered: a failed WIP commit (pre-commit hooks rewrote files) followed by `git checkout HEAD -- <op>`
  reset the uncommitted work. It was recovered from the staged blobs (`git fsck --unreachable`). Each fixture
  device reopen gets a new device id, so it gets a fresh cache entry (correct: the key includes the device).
- Tests added: `test_cached_descriptor_is_rebound_every_call` in test_nomic_moe_combine.py. It checks two calls
  share one descriptor, each call carries its own input / output addresses, and both outputs are correct.
