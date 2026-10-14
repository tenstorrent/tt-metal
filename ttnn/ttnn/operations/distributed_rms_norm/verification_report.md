# Verification Report: distributed_rms_norm

Box: Blackhole QuietBox 2×2 (4 × p150), worker grid 11×10 (99 workers + port row), FABRIC_2D_TORUS_XY realized
as MESH (no wrap links), router payload 14 400 B. G = 2 along either axis on the golden mesh; G = 4 on the
ring mock (1×4 / 4×1).

## Code Review

Every item below was **fixed in place**. Nothing in this list was deferred.

| # | Finding | Fix |
|---|---------|-----|
| 1 | **L1 capacity failure on a corpus shape.** Per-device `(2, 2048, 4096)` bf16 + bf16 gamma (8 translated cells, `test_distributed_norm_allclose[...-8192-2048-2]`) raised the op's own `P_resident` RuntimeError: 1 388 544 B > 1 MiB. Of that, 704 KB was `cb_xg`, the fp32 x·gamma block that exists only for the stall-shadow prescale. The design claimed no corpus shape reaches the capacity error; the translated suite does. | New per-call regime choice `GAMMA_PRESCALE = HAS_GAMMA ∧ F_worker(prescale) ≤ L1_BUDGET`. When it fails, `cb_xg` is not allocated. Instead, normalize fuses gamma as `BinaryFpu<Mul, x, rstd Col>` → `DestReuseBinary<Mul, gamma (Row mapping)>` → pack (the design's own "gamma prescale" perf-lamp alternative). DEST reuse has no broadcast, so the reader replicates gamma row 0 into all 32 rows on that path. Footprint drops to 684 032 B, and the 8 cells pass. The perf-focus shapes still take the prescale path (unchanged). The fused path was validated by forcing it on for the full acceptance suite (27/27) and the full supported golden subset (579/579). |
| 2 | **DRY: the output batch knob was restated.** `out_batch_tiles_knob()` sized `cb_output_tiles`, but the writer hard-coded `cw` as its wait/pop/barrier quantum. Turning the knob would have desynchronized CB size and consumer. | `out_batch` is now a writer RT arg sourced from the knob. The writer walks each tile row in `out_batch` batches, and the host asserts that the batch divides the chunk (wrap safety). Verified live by running the acceptance suite with the knob at 1 tile (18/18), then restoring it. |
| 3 | **DRY: the record size was restated.** The host had `RECORD_BYTES = 128`, and the reader and writer each had a literal `128`. | Passed as a CT arg to both kernels. |
| 4 | **Missing caller error: gamma layout.** A TILE gamma would have been read as a stick (silent garbage). | `ValueError` when gamma is not ROW_MAJOR. |
| 5 | **An explicit `num_links` was silently clamped** to `grid_x // 2` (the port-column cap). | Explicit `num_links` with `2·L > grid_x` now raises `ValueError`. Only `None` (auto) clamps. |
| 6 | Redundant special-case skips in `validate()` for `num_links=None` and `gamma_dtype="none"`. | Removed. Both values are now listed in SUPPORTED, so the generic loop handles them (see Registry). |

Design-conformance review (no change needed):
- **Algorithm:** matches the design. Flat fp32 record all-gather, Accurate (SFPU) stats sums, `transpose_dest` post-ops, x resident across the exchange (DRAM read once and written once).
- **Topology:** reader on NCRISC/NoC0, writer on BRISC/NoC1, port cores on BRISC/NoC1 in logical row 0.
- **Work distribution:** the grid is filled. `_choose_split` spreads rows × W chunks over all 99 workers (80–99 busy at the flagged shapes; one worker only at the 1-tile-row × 2-tile shapes).
- **Batching:** both dataflow halves batch one tile row per barrier, and the record gather is face-row runs.
- **Blocking-model fidelity:** `block_rows` (knob function, RT arg), `core_w_tiles`/`S_w` (`MIN_CORE_W_TILES`), `X_BLOCKS_IN_FLIGHT`, `OUT_DEPTH`, `out_batch_tiles` and `HEADER_RING` are each defined once on the host and flow to CB sizes and kernel args. No CB scales unconditionally with a whole-op dimension.
- **Expression:** compute is block-scoped (one helper call per block op). The reader pushes per tile row into a resident block CB, which is the planned batch: the block live set is held, and there is no per-tile completion boundary.
- The multi-round loop (`num_blocks`) is present in all three worker kernels. The port kernel runs one round, and the host asserts `num_blocks == 1` (R2 deferred, as designed).
- Raw API usage is justified in-file. These are sub-tile record placement and Fabric transport; no helper covers them.

Prompt `## Rules` check:
- **MUST be one fused native implementation, with no composite or CCL op calls.** Holds. There is exactly one `ttnn.generic_op(MeshProgramDescriptor)` per call, and no forbidden symbol appears in the op package.
- **Advisory (not a MUST):** the prompt's implementation notes ask for "no host synchronization inside the op; semaphores and scratch persistent per program".
  - The first call per (mesh, port-set) creates the global semaphores and runs one `ttnn.synchronize_device`. It is cached after that, as in the design's risk table.
  - The exchange scratch is a per-call `allocate_tensor_on_device`, not cached.
  - Both are trace-capture hazards only if the *first* call happens inside a capture. Reported here, not changed: a cached scratch would need lifetime management across mesh re-opens. This should be revisited when the op is traced in a model.

## Registry Conformance

- `INPUT_TAGGERS` (`alignment`, `rank`; signature `(inputs, axes)`), `SUPPORTED`, `EXCLUSIONS` and `validate()` are present. `validate()` checks SUPPORTED per axis and then EXCLUSIONS, raising `UnsupportedAxisValue` / `ExcludedCell`. The entry point calls `validate()` first. The op file does **not** declare INVALID.
- **Auto-fixes to SUPPORTED, based on evidence:**
  - **`num_links`: added `None`.** The 197 `xpass_drift` cells (translated tests calling with the default `num_links=None`) were all this one gap. The op always accepted `None` (auto = all usable links), but the registry did not list it.
  - **`topology`: added `Ring`.** Ring cells are pruned from `test_golden.py` on this wrap-less box, so `test_ring_mock.py` is the only place Ring runs. With Ring allowed it passes **16/16** (1×4 and 4×1, Ring and Linear, L = 1 and 2, decode and 640-row shapes). A probe (`probes/probe_001.py`) additionally checked:
    - fp32 and bf16;
    - gamma and no gamma;
    - batch 2;
    - 2048 rows;
    - three back-to-back calls, which were **bit-identical** (fence and counters re-arm correctly at G = 4).
  - **`alignment`: added `h_non_aligned`.** With it allowed, all **144** h_non_aligned golden cells on the supported path pass. Rows are independent, so tile-padding rows produce output only in padding rows (the design's own analysis).
- **INVALID audit** (`feature_spec.py`): the 3 entries are the gamma `"none"` canonicalization, both directions. They are single-tensor (both axes describe gamma) and structural. There are no cross-tensor couplings and no "not yet supported" entries. The canonical bf8b + ROW_MAJOR entry is correctly absent: the activation is TILE-only in TARGET, and `gamma_dtype` has no bf8b, as stated in the spec comment. **No change requested.**

## L1 Ledger Audit

- **Ledger currency:** fixed in `l1_ledger.md`.
  - The symbol table said `min_core_w_tiles = 4`; the code uses 8.
  - The worst-case footprint example used a 13-wide grid and a stale split.
  - `cb_xg` is now conditional on `GAMMA_PRESCALE`.
  - `cb_output_tiles` is sized by `OUT_DEPTH · out_batch_tiles`, and the batch is now a single-source knob.
  - `cb_gamma_tiles` is row-replicated on the fused path.
  - A "Verifier deltas" section records each change. Every CB declared in the code has a row, and the size expressions match the allocation (per chunk group).
- **Capacity vs live set:**
  - `cb_output_tiles` is 2× its live set; that is double buffering, justified.
  - No over-sized buffer otherwise. The spanned axes (`cb_x_resident`, `cb_xg`: rows × w_local; `cb_stats_combine`: rows × sources) all scale with their axis.
  - The only above-need term was `cb_xg` on shapes where the prescale does not fit. That is now resolved by the fused fallback (item 1).
- **Page format vs DEST:** `fp32_dest_acc_en = True` is the only supported value, and every DEST-produced intermediate is Float32. Consistent in both directions.
  - When the `fp32_dest_acc_en=False` refinement lands, `cb_xg` must drop to Float16_b.
  - The stats CBs stay Float32 by spec requirement ("fp32 on the wire"); that is a recorded mechanism reason.
- **Disjoint lifetimes:** `cb_sq_acc` / `cb_rstd` / `cb_stats_combine` are disjoint and kept separate. The justification is recorded: the deferred R2 overlap needs round b+1's sumsq live while round b's rstd is held. No blank justification cells.
- **Bounds:** `block_rows`, `core_w_tiles`, `n_src_tiles` (G ≤ 8, `S_w ≤ ceil(Wt/8)`) and `GAMMA_PRESCALE` each carry a bound and a predicate. The total is closed-form:

  ```
  F_worker = br·cw·(T_x + GAMMA_PRESCALE·T_f) + HAS_GAMMA·cw·T_g + 2048 + br·T_f·(3 + n_src_tiles) + OUT_DEPTH·out_batch·T_x
  ```

  `br·cw` scales with the block, `cw` with the W chunk, and `n_src_tiles` with `G·S_w`.
- **Data-movement budget:**
  - Present and consistent with the code: x read once, out written once.
  - gamma is read S_r× (deferred broadcast R8, with a positive reason: ≤ 3 % of a worker's DRAM bytes).
  - Records stay in L1/Fabric only.
  - The cheapest-traffic split (R1) is implemented.
- **Block-size defaults:** interleaved splits fill the grid first (`_choose_split` minimizes the max tiles per worker over all 99 workers), and then the block = the whole per-core assignment (coarsest).
  - The departure `MIN_CORE_W_TILES` 4 → 8 was justified by **measurement**: 640×1792 went 28.7 → 21.7 µs because record pulls grow as S_w². Recorded in the implementer deltas.

## Precision Baseline

`tests/ttnn/unit_tests/operations/distributed_rms_norm/test_distributed_rms_norm_precision_baseline.py`: randn x and gamma (same dtype), ε = 1e-6, cluster_axis 1 (G = 2), HiFi4 + fp32 accumulation. The worst device of 4 is reported. ULP is measured in the output dtype against the fp32 reference rounded to that dtype. Ratio = got/true over |ref| > 1e-3·max.

| Shape | dtype | PCC | Max Abs Err | Mean Abs Err | Rel RMS Err | Max / mean ULP | Ratio p5 / median / p95 |
|-------|-------|-----|-------------|--------------|-------------|----------------|-------------------------|
| (1,1,32,1792) | bf16 | 0.999998 | 0.0406 | 0.00099 | 0.00189 | 1 / 0.13 | 0.9965 / 0.9993 / 1.0020 |
| (1,1,640,1792) | bf16 | 0.999998 | 0.0431 | 0.00099 | 0.00186 | 1 / 0.12 | 0.9965 / 0.9994 / 1.0020 |
| (1,1,2048,1280) | bf16 | 0.999998 | 0.0428 | 0.00098 | 0.00186 | 1 / 0.12 | 0.9965 / 0.9994 / 1.0021 |
| (32,4096) | bf16 | 0.999998 | 0.0352 | 0.00096 | 0.00184 | 1 / 0.11 | 0.9966 / 0.9995 / 1.0021 |
| (1,1,32,1792) | fp32 | 1.000000 | 0.0239 | 0.00075 | 0.00136 | 50 584 / 13 911 | 0.9976 / 0.9989 / 0.99995 |
| (1,1,640,1792) | fp32 | 1.000000 | 0.0311 | 0.00084 | 0.00146 | 60 699 / 15 423 | 0.9975 / 0.9987 / 0.99983 |
| (1,1,2048,1280) | fp32 | 1.000000 | 0.0236 | 0.00080 | 0.00146 | 62 942 / 15 246 | 0.9975 / 0.9987 / 0.99984 |
| (32,4096) | fp32 | 1.000000 | 0.0261 | 0.00079 | 0.00145 | 54 870 / 14 930 | 0.9975 / 0.9988 / 0.99989 |

**Assessment**
- **bf16** is at the output format's limit: at most 1 bf16 ULP, and rel-RMS 0.0019 against a gate of 0.01.
- **fp32** passes its gates with margin (rel-RMS 0.0015 against 0.005; PCC 1.0). However, it carries a **small systematic downward bias**: median ratio 0.9988, p95 < 1, and a mean error of about 15 k fp32 ULP (≈ 2⁻¹⁰ relative).
  - **Scale-vs-precision triage:** this is *not* the scale-bug signature. rel-RMS is low, and the bias is ~0.1 %, not a constant ≠ 1 with large RMS.
  - **Cause:** it is the design's documented tf32 risk. The FPU unpacks fp32 operands as tf32 (truncation toward zero), and three FPU-truncated operands sit on the output path: x·gamma, then xg·rstd. Each costs about −2⁻¹¹ on average.
  - **Concrete lever:** SFPU (full fp32) multiplies, or unpack-to-DEST fp32 for the normalize / prescale multiplies when the input is float32. This is noted on the numeric refinement (Refinement 4). No gate is violated today.

**Recommended tolerances (Phase 0 cells):**
- bf16: PCC ≥ 0.9999, rel-RMS ≤ 0.004, atol 0.06.
- fp32: PCC ≥ 0.99999, rel-RMS ≤ 0.003, atol 0.04, rtol 3e-3.
- The golden gates (0.999/0.01 and 0.9999/0.005) hold with ≥ 3× margin.

## Verifier CLI Summary

The final run (`eval/eval_test_runner.sh eval/golden_tests/distributed_rms_norm/` + `python3 -m eval.verify_supported`, artifact `verifier_report.json` in this directory: a compact copy with the summary, every loud-category entry and xfail counts by missing axis value. The full 11 MB report was produced at `/tmp/drn_verify/golden2/verifier_report.json`):

| Category | Count |
|---|---|
| supported_pass | 931 |
| xfail_expected | 7125 |
| invalid_skipped | 7776 |
| no_axes_found | 79 (non-registry: ring mock 16/16 pass, regression/numerics, translated helpers. All pass except the item below.) |
| **supported_fail** | **0** |
| **xpass_drift** | **0** (was 197: the `num_links=None` registry gap) |
| **xfail_wrong_mode** | **1**. Harness artifact, see below. |

Before the fixes, the first run showed:
- supported_pass 579;
- xpass_drift 197;
- xfail_wrong_mode 9 (8 of which were the capacity RuntimeError on `(2, 2048, 4096)`, fixed by item 1);
- 8 ring-mock failures (Ring not in SUPPORTED, now promoted).

**The remaining `xfail_wrong_mode`** is `test_translated.py::test_pre_all_gather_non_welford_fp32_requires_fp32_dest_acc`, and it is **not an op defect**.
- The test wraps the call in `pytest.raises((NotImplementedError, RuntimeError, ValueError))`, and the op correctly raises `ExcludedCell` (float32 + `fp32_dest_acc_en=False`) before allocating anything.
- The failure comes from `eval/l1_profiling.py`: `L1ProfilingError: L1 profiling returned peak_total=0; expected a positive allocation peak`. The L1-profiling plugin demands an allocation from every graded test, which no refusal test can provide.
- The same refusal passes in the acceptance suite (`test_fp32_without_fp32_acc_is_refused`).
- This needs a harness fix (exempt `pytest.raises`-only tests from the positive-peak check). It cannot be fixed op-side without allocating before `validate()`, which the contract forbids.

Acceptance suite: **27/27** (`scripts/run_safe_pytest.sh tests/ttnn/unit_tests/operations/distributed_rms_norm/test_distributed_rms_norm.py`).

**Test-mode note:** `--dev` cannot run multi-test fabric sessions on this box. Under `--dev`, the first test passes, and every later mesh re-open in the same session fails with `Timed out while waiting for active ethernet core … to become active again` (reproduced at the start of this pass; the same issue is in the implementer's friction log). All multi-test runs here are therefore non-dev. `--dev` stays usable for single-test hang triage.

## Measured device time (baseline for the perf refinements)

`tests/ttnn/unit_tests/operations/distributed_rms_norm/test_distributed_rms_norm_perf_bench.py` under `scripts/run_safe_pytest.sh --profile`. DEVICE KERNEL DURATION (ops CSV column 20), second call, per device (the group waits for the slowest):

| Shape (LOOSE_CASES config) | dev0 | dev1 | dev2 | dev3 | max | Device-side floor (DRAM + exchange) |
|---|---|---|---|---|---|---|
| 640×1792 **PERF FOCUS** | 23.3 | 24.7 | 24.9 | 21.7 | **24.9 µs** | ~9.3 µs |
| 32×1792 decode | 9.8 | 10.6 | 10.9 | 9.8 | **10.9 µs** | ~0.75 µs |
| 2048×1280 | 41.3 | 44.6 | 42.0 | 44.2 | **44.6 µs** | ~20.8 µs |

The feature_spec `roofline_us` / `target_us` are traced wall-clock values that include the 8.8 µs program floor. Subtracting the floor gives device budgets of ~15.3 µs (perf focus), ~4.0 µs (decode) and ~30.7 µs (2048×1280). All three are currently above budget, so the perf refinements have real headroom.

## Recommendations

- **Refinement ordering and cross-cutting concerns:** see `op_requirements.md`.
- **Sharded placement:**
  - `auto_shard_config` may place shards on logical row 0, which is the port row. The sharding refinement must handle that collision, either by moving the record send to the worker NCRISC and the port role onto a shard core, or by placing ports on cores the shard grid leaves free.
  - The stats-ready multicast rectangle must cover only cores of the program.
- **L1 headroom:**
  - With the fused fallback, no corpus or translated shape reaches `P_resident`.
  - The next failure would be a per-worker block above ~1 MiB *without* `cb_xg`, e.g. W_local ≥ ~16 K at M ≥ 4 K on G = 2.
  - The deferred multi-round regime R2 (`block_rows < core_rows`) is the fix, and it is also the perf-1 overlap lever. The loop scaffolding is already in all worker kernels; the port kernel needs per-round thresholds.
- **`WORKER_L1_BUDGET = 1 MiB`** is conservative for Blackhole: about 1.4 MB of CB space is free on workers, since port scratch never lands on workers. Raising it would re-enable the prescale path on larger shapes, but should be measured, not assumed.
- **Device skew:** the same program differs by 1–3 µs between devices, and which device is slowest varies by shape. Because the exchange makes every call wait for the slowest device, the perf refinements should report the per-device max, not the min.
- **Trace hazards (advisory):** first-call semaphore creation and synchronize; per-call scratch allocation. See Code Review.
