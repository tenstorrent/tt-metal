# Verification Report: mhc_post

Phase 0 = regime R1 plain-2D (op_design.md): plain mode, fp32 streams + sublayer, separate coefficients, one
`generic_op` dispatch per call over up to the full compute grid (110 cores on the BH p150 used here).

## Code Review

Fixed in place:
- **DRY: DEST slot layout restated as a literal.** The host's `MIX_FIXED_SLOTS = 2` (coefficient + operand DEST
  slots, used by `choose_block_cols()` for the DEST cap) was re-written as a bare `2` in the compute kernel's
  `static_assert` and in `acc_slot<U>()`. It is now passed as compute CT arg 7 (`mix_fixed_slots`); the
  kernel derives the accumulator base and the DEST-fit assert from it, and asserts the fixed D0/D1 layout it
  hard-codes (`Dc`, `Do`).
- **Reader literal**: `coef_words = 1024` is now `coef_page / sizeof(uint32_t)` (derived from the CB page).
- **L1 ledger currency**: `cb_coef_raw` row said "3, or 1"; the code allocates 2 pages in Phase 0 (post +
  comb, 1 used when they alias). Row and footprint expression corrected.

Tried and reverted (measured / built, not kept):
- **Ragged column pass as one block.** When a core's column range is not a multiple of `block_cols`, the last
  pass runs as 1-unit blocks (one coefficient broadcast per unit instead of per block). At T=160, C=5120 every
  core has a 5- or 6-unit pass plus a 2–3-unit remainder, so this is real cost. A compile-time dispatch to
  `mix_units<B>` for every B < block_cols makes the compute binary overflow the kernel-config buffer
  (`TT_FATAL program.cpp:3646 state.offset <= max_size`). Reverted; filed as a lever in Refinement 4 (balance
  the passes, or bound the instantiations).

Reviewed, no change needed:
- Registry wiring, single dispatch (`ttnn.generic_op` once), TensorAccessor everywhere, `kernel_main()`
  syntax, `api/dataflow/dataflow_api.h` include, CB push/wait counts balanced (coef: n+n² per row; F: bc per
  pass; X: bc·n per pass; X': PackTile reserve/push per tile ↔ writer wait b·n, single-tile pops because a
  ragged block's window may wrap the ring).
- Helper usage: the mix is one `eltwise_chain` per output stream j over the block (UnaryBcast<Col> via
  unpack-to-dest + CopyTile + SFPU `MulBinary` / `Addcmul` with value 1.0 as a single-rounding fp32 MAD);
  no raw compute API besides `compute_kernel_hw_startup`. The coefficient product never touches the FPU (tf32
  bias), as the design requires. No inter-core communication in R1, so no `mcast_pipe` candidate yet.
- Broadcast: coefficients are column-0 pages broadcast along Col (the only correct dim: one value per token
  row); no redundant full-tile fill.
- Program cache: `generic_op` hashes runtime-arg *counts*, not values, so tensor addresses and
  post_col / comb_col (common runtime args) are refreshed per call. Correct.
- Dispatch: kernel / CB core sets are merged rectangles (`_merged_ranges`: full rows + one partial row), not
  one range per core.

Prompt rules: `eval/prompts/mhc_post.txt` has no `## Rules` section; the stock native-kernel policy applies
and is met (no manipulation-op wrappers).

### Design conformance
- **Algorithm / pipeline**: matches R1 (reader expands coefficient columns into column-0 pages, compute mixes
  in fp32 SFPU, writer streams X'). The implementer went beyond the design's per-output-tile chain: one
  coefficient broadcast is shared by all `block_cols` units of a block (accumulators in DEST, full-sync
  DEST → `block_cols ≤ 8 − 2 = 6`). That is a better realization of the same block operation, not a
  deviation (scope boundary: mechanism is the implementer's).
- **Work distribution**: the split chooser is implemented exactly (score `max(total/DRAM_BW,
  max_core/CORE_BW)`, ties → balanced → smaller R); measured 104–110 of 110 cores busy on every perf shape
  (fewer only where the shape has fewer units: 1 core at X 32×128, 16 at 64×1024). Both dataflow halves are
  block-batched: the reader issues `block_cols·(1+n)` reads per barrier (up to 30 in flight), the writer
  `block_cols·n` writes per flush with a single final barrier. Fills the machine.
- **Blocking-model fidelity**: `BLOCK_ROWS`, `block_cols` (`choose_block_cols()`), `IN_BLOCKS`, `OUT_BLOCKS`,
  `COEF_DEPTH`, R/Q are named host values flowing into CT/RT args; every CB size derives from them. No CB
  scales with a whole-op dimension (`cb_coef_cols` scales with n² × COEF_DEPTH only). `block_rows = 1` is the
  design's reasoned decision and stays a CT parameter. Expression: the reader, compute and writer each
  acquire / synchronize / retire per **block** (one barrier, one wait/pop, one flush per pass) — not per unit.
  The ragged pass is the one place where the compute (and writer) step at unit granularity (see above).
- **Inter-core communication**: none in R1, as designed.
- **Port reservation** (design: the worker list excludes the 4 port cores whenever the fabric is active):
  not implemented — `worker_list()` says so. It is irrelevant for plain mode (X' bits never depend on the
  split) and belongs to Refinement 2 (feed), where it is required for feed vs no-feed bit identity.

## Registry Conformance
- `INPUT_TAGGERS` (`alignment` over T, `feed_split` over C), `SUPPORTED` (all 10 axes, incl. both tagger
  axes), `EXCLUSIONS = []`, and `validate()` (malformed-call `ValueError`s → SUPPORTED per axis →
  EXCLUSIONS) are present and correctly wired; tagger signatures are `(inputs, axes)`. `mhc_post()` calls
  `validate()` first. The op file does **not** declare `INVALID`.
- No XPASS drift, so no SUPPORTED auto-fix was needed.
- **INVALID audit** (`eval/golden_tests/mhc_post/feature_spec.py`):
  - `{mode ∈ stats/collapse} × {dtype | sublayer_dtype = float32}`: couples a kwarg axis with tensor dtypes;
    justified by the documented owner contract decision (the projection / collapse read the bf16 X' the op
    writes; with fp32 streams that is a different quantity). Universe change, not "not yet": acceptable.
  - `norm=on` / `feed=on` × `mode ∈ {plain, stats}`: structural (no y exists without collapse). Good.
  - `feed=on × feed_split=not_block_aligned`: structural (no whole-tile C/G column block). Good.
  - No cross-tensor layout × dtype entries. bf8b + ROW_MAJOR canonical entry: not applicable (TARGET has
    neither bf8b nor ROW_MAJOR). Not a norm-like op with optional weight canonicalization cells.
- **Harness note (not an op bug)**: `merge_axes` warns of 202 declared/captured axis mismatches, all on
  `norm=on` / `feed=on` cells (xfail now). `run_mhc_post` / `prepare_mhc_post` first call the op *without*
  norm / feed to build their reference, so the first (and, while the cell is refused, only) observed call is
  that reference. Once Refinements 1–2 make these cells run, the second call will also be observed; if the
  verifier keeps reading the first captured call, the eval owner should make the observe hook record the
  last call (or skip the reference calls).

## L1 Ledger Audit
- **Currency**: rows for the 5 Phase 0 CBs match the descriptor (after the `cb_coef_raw` fix above). Size
  expressions match: F `IN_BLOCKS·bc`, X `n·IN_BLOCKS·bc`, coef raw 2, coef cols `(n+n²)·COEF_DEPTH`, X'
  `n·OUT_BLOCKS·bc`.
- **Capacity vs live set**: `cb_coef_cols` uses 128 B of each 4 KB page (column 0) — justified by the
  unpack-to-dest COL-broadcast mechanism cap (reads column 0 of faces 0/2), recorded in the row. The F / X
  rings' filler slots on a ragged pass are justified (windows start on block boundaries). No unexplained
  over-capacity; no collapsed extent (every row whose live set spans `c` scales with `bc`).
- **Page format vs DEST**: fp32_dest_acc_en is always on; fp32 pages (coef cols, fp32 F/X) are unpacked to
  DEST (`UnpackToDestFp32`); X' is X.dtype by contract. No over/under finding.
- **Disjoint lifetimes**: every pair carries a `Shares with / why not` answer (concurrent streaming CBs; raw
  vs cols live together during the expansion).
- **Bounds**: `bc ≤ min(core_c_tiles, ⌊BLOCK_L1_BUDGET / bytes_per_block_col⌋, 8 − MIX_FIXED_SLOTS)`; all
  symbols bounded. Per-core Phase 0 footprint
  `bc·(IN_BLOCKS·pgF + n·IN_BLOCKS·pgX + n·OUT_BLOCKS·pgX) + (n+n²)·COEF_DEPTH·4096 + 2·4096`, worst case
  fp32 bc=5: 360 K + 160 K + 8 K = 528 KiB. `bc` scales the streaming rings; the coefficient term is fixed.
- **Block-size defaults**: interleaved → full grid, then the coarsest block that fits — held (`block_cols` =
  the whole per-core range up to the L1 (5 fp32 / 10 bf16) and DEST (6) caps). The implementer measured
  balanced passes worse at T=1024 C=1792 (243 vs 215 µs) — a measured departure, recorded in the code.
- **Data-movement budget**: present and consistent with the code for plain (F, X, X' once each; coefficient
  tile read by all Q cores of a row: counted). Cheapest-traffic split recorded (C-columns-only for
  stats/collapse) with positive reasons for the deferred R3 (W mcast) row. The coefficient re-read is a
  reuse-shared broadcast the design defers as a perf lamp → filed as Refinement 5.

## Precision Baseline

`tests/ttnn/unit_tests/operations/mhc_post/test_mhc_post_precision_baseline.py` (fp32 plain, doubly
stochastic comb, float64 reference):

| Shape (F) | PCC | Max Abs Err | Mean Abs Err | Relative RMS Err | ULP p50 / p99 | got/true ratio p5 / median / p95 |
|-----------|-----|-------------|--------------|------------------|---------------|----------------------------------|
| (32, 128) | 1 − 1.4e-15 | 7.1e-7 | 4.1e-8 | 5.3e-8 | 0 / 12 | 0.99999988 / 1.0 / 1.00000012 |
| (1, 100, 1024) (T non-aligned) | 1 − 1.3e-15 | 1.0e-6 | 4.0e-8 | 5.3e-8 | 0 / 12 | 0.99999988 / 1.0 / 1.00000012 |
| (1, 1, 160, 5120) | 1 − 1.3e-15 | 9.6e-7 | 4.2e-8 | 5.3e-8 | 0 / 12 | 0.99999988 / 1.0 / 1.00000012 |
| (1, 1, 640, 1792) | 1 − 1.3e-15 | 1.0e-6 | 4.1e-8 | 5.3e-8 | 0 / 12 | 0.99999988 / 1.0 / 1.00000012 |

**Assessment**: fp32-rounding limited (rel-RMS ≈ 5e-8, about half an fp32 ulp per element: one rounding per fused MAD); ratio tightly
centred on exactly 1.0 — no scale / structural bug. The large max-ULP values (up to ~2.6e6) occur only
where the reference nearly cancels (|X'| ≪ the summands), i.e. absolute error ~1e-6 on a ~0 value — expected
for any fp32 evaluation order. Golden `ulp_p99` is 13–19 on every supported cell.
**Recommended tolerances (fp32)**: PCC ≥ 0.99999, relative RMS < 2e-6, atol 2e-6 (|values| ~ O(1)).

## Verifier CLI Summary
(`verifier_report_summary.json` next to this report — summary, counts and the passing node ids; the full 2 MB `verifier_report.json` stays in the run dir `/tmp/mhc_post_verify_golden2`)
- supported_pass: 92
- xfail_expected: 625
- invalid_skipped: 2578
- no_axes_found: 40 (test_regression / test_feed gates that skip until their mode is supported)
- supported_fail: 0
- xpass_drift: 0
- xfail_wrong_mode: 0
- acceptance: 11 / 11; precision baseline: 4 / 4

`TARGET − SUPPORTED` → queue: dtype bf16, sublayer_dtype bf16, coef_source packed, mode stats / collapse →
Refinement 1; norm on, feed on → Refinement 2. Nothing is left undocumented; every xfail_expected bucket
(plain bf16 / mixed / packed, stats, collapse × norm × feed) maps onto Refinement 1 or 2.

### Device time, Phase 0 (plain fp32, single chip, profiler device-kernel duration)

| X shape | T, C | measured | DRAM roofline | % of peak | reference |
|---------|------|----------|---------------|-----------|-----------|
| (1,1,160,20480) | 160, 5120 | 117.5 µs | 57.6 µs | 49 % | — |
| (1,1,640,28672) | 640, 7168 | 516 µs | 322 µs | 62 % | 3 781 µs (81-op composite) → 7.3× |
| (1,1,1024,7168) | 1024, 1792 | 214 µs | 129 µs | 60 % | — |

## Recommendations
- **Blocker for every perf refinement (eval harness, not the op):** `perf_measure.py` cannot measure any case.
  `helpers.prepare_mhc_post`'s `call()` → `call_mhc_post_mesh` re-creates every input with `from_torch` on each
  call, and `ccl_perf.measure` invokes `call()` inside `begin_trace_capture` → `TT_FATAL: Writes are not
  supported during trace capture` (fd_mesh_command_queue.cpp), then a teardown `TT_FATAL` and, in one run, a
  dispatch-timeout hang (triage: device idle, no user kernel resident). Reproduced on a supported plain case
  (`CCL_PERF_CASES=19`). Fix: upload the inputs once in `prepare_mhc_post` and have `call()` dispatch only.
  Device timings above therefore come from the profiler (`run_safe_pytest.sh --profile`) on one chip.
- Refinement 1 is large by design (anchor rule: R1 + R2 must land the whole perf-flagged contract, and the
  fused modes are INVALID without bf16). If it has to partial-tick, the natural cut is stats first (the
  combine tree), collapse (hc SFPU op + y) as `Refinement 1b`.
- The packer's fp32 → bf16 rounding mode is the first thing to check on bf16 (design Risks).
- L1: the R2 additions (W slice `n·core_c_tiles·4 KB`, slots, y hold) sit on top of a Phase 0 footprint of
  up to 528 KiB fp32 — fine for bf16 (≈ 230 KiB plain part per the ledger), but `BLOCK_L1_BUDGET` must be
  re-derived for the R2 bytes-per-block-col (`l1_ledger.md` symbol table) rather than reused.
- The 4 reserved port cores cost ~3.6 % of the grid on fabric-on, feed-off calls (accepted by design for
  bit-identity); measure in the perf phase.
