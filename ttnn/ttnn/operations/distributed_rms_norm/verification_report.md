# Verification Report: distributed_rms_norm

Box: Blackhole QuietBox 2 (4 chips; system mesh 2×2, ring mocks 1×4 / 4×1 under FABRIC_2D_TORUS_XY, 14 KiB + 64 B
router payload). Phase 0 as handed over: commits `64dc695a925..ef0a17fbf7a`. Verifier commits follow `ef0a17fbf7a`.

## Code Review

Four real kernel / host defects were found and fixed. Two of them gave **wrong numbers on SUPPORTED cells** that the
acceptance and golden matrices did not reach. Each fix has a regression test or probe.

1. **Wave race in the stats exchange (wrong results, `num_waves > 1`).** `SEM_OWN_STATS` on a sending port counts
   root publishes *cumulatively over waves*: the port ships wave w once `own_stats ≥ (w+1)·n_roots`. A root whose next
   block is empty (`R_b = 0`) incremented for wave w+1 immediately after finishing wave w. A fast root's wave-(w+1)
   increment could therefore satisfy the wave-w threshold while a slower root's wave-w rows were still unpublished, and
   the port shipped stale rows. Measured: R1-X with `num_waves = 2` (e.g. per-device 4096×1024 with
   `BLOCK_ROWS_CAP = 1`, or 2×2048×4096 when L1 forces two waves) gave **~40 of 108 wave-0 rows wrong on every call**
   on the p = 1 devices, each with a constant got/true ratio (√2 on the first call over a fresh landing, the previous
   call's partial afterwards). The bug is invisible when the same input is repeated, which is why `test_waves_forced`
   (one call per input) and repeat-determinism tests passed.
   *Fix*: a new program-local semaphore `SEM_OWN_COUNTED`. Each sending port signals its link's roots once it has
   counted wave w, and a root waits `SEM_OWN_COUNTED ≥ w·n_pub` before its wave-w increment. This costs nothing when
   `num_waves = 1` (port → root signals only for `w < num_waves − 1`). Diagnosis ruled out the alternatives on the
   way, by measurement: the arrival counters are clean (DPRINT); payload-size caps (4 / 8 KiB) don't help; a
   payload-carried completion flag doesn't help; delaying root reads does. Recorded in `op_design.md` →
   Tensix-to-Tensix contracts. Regression: `test_waves_alternating_inputs` (R1 many-roots and the R3 perf-focus shape,
   with and without γ).
2. **Output store order (wrong results, `R_b ≥ 2` and `nch ≥ 2`).** Compute emits output quanta chunk-outer /
   tile-row-inner (`scale_by_inv_rms_block` runs `grid(R_b, C)` per chunk). The writer consumed them tile-row-outer, so
   row 1's chunk 0 was stored at row 0's chunk 1. This hit R1-X whenever a core owns ≥ 2 rows and `Wt > 64`
   (e.g. 4096×4096 without γ: 40 bad tile-rows, rel-RMS 0.56). *Fix*: the writer walks compute's order. Regression:
   `test_chunked_multi_row_store_order`.
3. **Persistent landing-scratch cache (golden-suite cascade + unbounded L1).**
   (a) The cache kept one never-freed sharded L1 buffer per *exact* landing size. A sharded L1 buffer reserves its
   range on every bank, so a run over many shapes shrank every later plan's L1 budget: a 2×2048×4096 translated cell
   refused `block_rows = 1` mid-suite but fits on its own. (b) `make_plan` held the cache dict as a frame local, so
   when that refusal raised, the golden conftest's failed-cell cleanup (it deallocates every device tensor reachable
   from the traceback's locals, including dict values) freed the persistent scratches. Every later call then failed
   with `Tensor is not allocated` (159 translated cells). (c) Separately, under a no-dispatch collection pass
   (inline precompile) the cache stored address-0 mock semaphores and scratch that the next real call reused.
   *Fix*: power-of-two buckets (min 4 KiB; smallest live bucket ≥ need reused; total < 2× the largest), an
   `is_allocated()` liveness check that drops externally freed buffers, integer-only locals on the planning path, and
   address-0 allocations never cached. Probes: `probes/scratch_cache_probe.py`, `probes/collect_window_cache_probe.py`.
4. **Reader word moves**: `invalidate_l1_cache()` is a bare `asm("fence")` on Blackhole (a hardware fence, not a
   compiler barrier). The reader's plain-load batching now goes through `l1_acquire()` (fence + `"memory"` clobber).
   This was not the cause of bug 1; it is hardening.

Smaller fixes:
- `cb_output` page size taken from the **output** dtype (was the input's: equal today, wrong once dtypes may differ).
- DRY: `TILE_DIM` single-sourced in the op file. The common CT-arg prefix length is a named host constant
  (`COMMON_CT_ARGS = 18`) with an assertion, mirrored by the kernels' `TensorAccessorArgs<18>`.
- **Drift fix**: `SUPPORTED["num_links"]` now lists `None` (the documented default, "every usable link"). The
  translated suite calls with `num_links=None`, the CLI classified it outside SUPPORTED (46 `xpass_drift`), and the op
  accepts it; `validate()`'s special case for `None` is gone.

Blocking-model fidelity: every knob in the design is a host module constant with a single source, read at call time
(`BLOCK_ROWS_CAP`, `MAX_PIPELINE_DEPTH`, `STREAM_DEPTH`, `W_CHUNK_TILES`, `R3_*`, `MAX_HEADERS_IN_FLIGHT`). CB page
counts, loop trips, landing layout and port counts derive from `R`, `num_waves`, `S`/`WS`, `C`, `G`, `G_w`. No CB scales
unconditionally with a whole-op dimension: `cb_x_tiles` spans `R·WS`, which is the planned resident block, proven to fit
by the closed-form solve, with WAVES as the fallback. Expression: compute runs whole-block helper calls (`grid(R_b, WS)`
square, `reduce` over `(R_b, 1)` / `(G_w, R_b)`, chunk chains over `grid(R_b, C)`). The reader pushes per tile-row and
the writer drains per (chunk, row) quantum, as the design specifies (squaring starts on the first landed row; one
barrier per quantum). Work distribution fills the grid: R1-X rows over all compute cores row-wise; R3-X W-slices when
that buys ≥ 2× the cores (640×1792 runs on 100 cores, decode on 14).

Helper usage: every compute phase is a kernel-lib helper (`eltwise_chain`, `reduce` incl. `AccumulateViaAdd` +
`ReduceWithinTile::Skip`, `calculate_and_prepare_reduce_scaler`). The two custom chain elements (`FinalizeCol0`,
`AddCol0`) are column-0-scoped SFPU bodies wrapped as chain ops (design lamp L7, catalog `sfpu_tile_scope`). Dataflow is
raw NoC / Fabric for the reasons in `op_design.md` → "Helpers considered and rejected" (no tile-page reader helper; the
in-chip gather is many-to-one, not a multicast). TensorAccessor everywhere; `void kernel_main()`;
`api/dataflow/dataflow_api.h` includes; CB push/pop counts balance (nominal quanta on shared rings).

Prompt rules: the single MUST rule holds. The op is one `ttnn.generic_op` per call (one MeshProgramDescriptor) with
its own kernels and Fabric transport; it calls no CCL / norm op, never changes the fabric config, and makes no host
round trip. Deterministic: fixed sum order g = 0..G−1 (`test_regression.py::test_deterministic` passes).

Deferred (architectural, filed in `op_requirements.md`): sharded placements (R1), RM γ + alignment (R2),
bf8b / 16-bit DEST (R4), exchange-transport rework (R5).

## Registry Conformance

- Confirmed in `distributed_rms_norm.py`: `INPUT_TAGGERS` (exactly `alignment`, `rank`; both `(inputs, axes)`),
  `SUPPORTED` (all 12 axes), `EXCLUSIONS` (`{float32, fp32_dest_acc_en: False}`), and `validate()`. `validate()` runs
  caller errors (`ValueError`), then SUPPORTED per axis (`UnsupportedAxisValue`), then EXCLUSIONS (`ExcludedCell`),
  and is the entry point's first statement. `"none"` is always accepted for `gamma_dtype` / `gamma_layout`. The op file
  declares **no** `INVALID`.
- Auto-fix from drift evidence: `num_links: None` added to SUPPORTED (above).
- INVALID audit (`eval/golden_tests/distributed_rms_norm/feature_spec.py`): six entries, all the canonical
  single-tensor gamma "none" sentinel coupling (gamma present ⇒ real dtype / layout; absent ⇒ "none"). There are no
  cross-tensor couplings and no "not yet supported" entries. The bf8b + ROW_MAJOR activation entry is legitimately
  absent, because the input `layout` axis is TILE-only and `gamma_dtype` has no bf8b (the file says so). No
  `INVALID_FOR_ARCH`. No change requested.

## L1 Ledger Audit

- **Currency**: fixed. `cb_x_tiles` and `cb_gamma_tiles` are allocated over `WS = nch·C` (the chunk-padded slice,
  `S ≤ WS < S + nch`), not `S`; the ledger rows, symbol table and footprint now say `WS`. The port-core row now
  describes the bucketed landing scratch. Every declared CB has a row and every row a CB.
- **Capacity vs live set**: no over-sized buffer. `cb_inv_rms` / `cb_x_tiles` / `cb_gather` depth-2 copies exist only
  with `num_waves > 1` (overlap mechanism, stated). `cb_gather` is allocated on non-root compute cores as unused space
  because the root's address must be uniform (stated). No collapsed extent: every buffer that spans `R`, `WS`, `C`,
  `G` or `G_w` scales with it.
- **Page format vs DEST**: `fp32_dest_acc_en = True` only, so every intermediate is Float32. No over / under finding.
  Refinement 4 must derive them from the flag (noted there); `cb_stats_slots` stays Float32 by requirement.
- **Disjoint lifetimes**: every pair carries a justification (e.g. `cb_sumsq_partial` vs `cb_inv_*` overlap across
  waves; `cb_inv_root` vs `cb_inv_rms` would need two producers).
- **Bounds / closed form**: `F(R) = R·per_row + fixed` with every symbol bounded in the table; `R_fit = 0` raises
  `NotImplementedError`. The landing scratch is now bounded (< 2× the largest bucket, ≤ ~log2(max/4 KiB) buffers).
  Before the fix it was unbounded across a process, which is what made the budget drift.
- **Data-movement budget**: present and consistent with the code (x and out cross DRAM once; γ per-core slice
  prefixes; stats in L1 end to end). The cheapest-traffic split (R1-X) is implemented; R3-X is an occupancy trade
  recorded in the regime table. Block-size defaults hold: interleaved spreads the row split over the grid and takes
  the coarsest resident `R` (one wave on every INPUTS shape).
- Per-core footprint (compute core): `F(R) = R·[xd·WS·T_in + T32·(2 + G + id + r3·(2 + gd·G_w)) + γ·C·T32] + 2048 +
  γ·WS·T_g + sd·C·T_out`. `R` scales everything per row; `WS` scales x and γ; `C` scales x·γ and the output stream; `G`
  scales the slots; `num_waves` turns the depths 1 → 2. Perf focus 640×1792 ≈ 0.27 MB.

## Precision Baseline

`test_distributed_rms_norm_precision_baseline.py`: 2×2 mesh, group along axis 1 (G = 2, Linear), bf16 γ, HiFi4 +
fp32 DEST. Worst device of the group per shape. ULP is in the output dtype. The ratio is got/true over elements with
|ref| > 5% of max.

| Shape | dtype | PCC | Max Abs Err | Mean Abs Err | Rel RMS Err | ULP max / mean | got/true p5 / median / p95 |
|-------|-------|-----|-------------|--------------|-------------|----------------|----------------------------|
| 1×1×32×1792 (R3-X) | bf16 | 0.999998 | 0.0625 | 5.0e-4 | 2.13e-3 | 1 / 0.14 | 0.9939 / 1.0000 / 1.0000 |
| 1×1×640×1792 (R3-X) | bf16 | 0.999998 | 0.0625 | 4.4e-4 | 2.03e-3 | 1 / 0.13 | 0.9944 / 1.0000 / 1.0000 |
| 1×1×2048×1280 (R1-X) | bf16 | 0.999998 | 0.0625 | 5.2e-4 | 2.18e-3 | 1 / 0.15 | 0.9941 / 1.0000 / 1.0000 |
| 2×4×128×512 (R3-X) | bf16 | 0.999998 | 0.0625 | 4.5e-4 | 2.04e-3 | 1 / 0.13 | 0.9943 / 1.0000 / 1.0000 |
| 1×1×32×1792 | fp32 | 1.000000 | 0.0184 | 6.1e-4 | 1.14e-3 | 51,741 / 11,191 | 0.9979 / 0.9991 / 1.0002 |
| 1×1×640×1792 | fp32 | 1.000000 | 0.0229 | 6.0e-4 | 1.12e-3 | 56,498 / 11,005 | 0.9980 / 0.9991 / 1.0002 |
| 1×1×2048×1280 | fp32 | 1.000000 | 0.0248 | 6.8e-4 | 1.25e-3 | 57,632 / 12,716 | 0.9978 / 0.9990 / 1.0000 |
| 2×4×128×512 | fp32 | 1.000000 | 0.0218 | 5.8e-4 | 1.12e-3 | 54,013 / 11,189 | 0.9980 / 0.9991 / 1.0002 |

**Assessment**: bf16 is at the format limit (≤ 1 bf16 ULP, median ratio exactly 1). fp32 shows a small, consistent
negative bias (median got/true 0.9991, ~5.5e4 fp32 ULP ≈ 0.3% worst-case). That is the FPU's TF32 operand truncation
in the x·x, x·γ and ·inv multiplies, not a scale bug: the p5–p95 spread is wide and rel-RMS is 1.1e-3 against the
5e-3 gate. The cross-device sum itself is exact fp32 (SFPU adds with UnpackToDestFp32).
**Recommended tolerances**: bf16 PCC ≥ 0.999, rel-RMS ≤ 0.01; fp32 PCC ≥ 0.9999, rel-RMS ≤ 0.005 (the golden
`TOLERANCES`; measured headroom 4–5×).

## Verifier CLI Summary

Final run: `EVAL_CAPTURE_L1=0 eval/eval_test_runner.sh eval/golden_tests/distributed_rms_norm/ /tmp/drms_golden3`
→ PASSED=821 FAILED=0 ERRORS=0 SKIPPED=33747 HANGS=0 (47,025 items). Then `python3 -m eval.verify_supported`
(artifacts: `verifier_artifacts/verifier_report.json`, `verifier_artifacts/golden_results.txt`):

- supported_pass: 784
- xfail_expected: 12457
- invalid_skipped: 33696
- supported_fail: 0
- xpass_drift: 1 — **classifier false positive, not drift.**
  `test_translated.py::test_pre_all_gather_non_welford_fp32_requires_fp32_dest_acc` *asserts that the op raises* for
  {float32, `fp32_dest_acc_en=False`}. The EXCLUSION raises `ExcludedCell`, the `pytest.raises` passes, and the CLI
  reads "outside SUPPORTED and passed". Promoting the cell would contradict the precision convention (fp32 input
  requires fp32 accumulation), so it stays excluded.
- xfail_wrong_mode: 0
- no_axes_found: 87 (ring-mock, regression and some translated tests call the op without the axes-tagging wrapper;
  all of them passed).

Golden breakdown: `test_golden::test_op` 576 passed / 12,384 xfail; `test_op_loose` 3/3; `test_ring_mock` 24/24;
`test_regression` 12/12; `test_translated` 209 passed, every other translated cell a support-refusal xfail.

Acceptance (`--dev --no-precompile`, split by mesh shape): 66 passed, 1 skipped. Verifier regressions: 7/7.
Debug test: 1/1.

**Harness / infrastructure findings** (not op defects; reported, not worked around in the op):
1. **Golden L1 capture cannot see this op's CBs.** With the default `EVAL_CAPTURE_L1=1`, every cached call fails with
   `L1ProfilingError: peak_total=0` (630 cells in the first run). A per-chip (heterogeneous) MeshWorkload compiles its
   programs on the mesh thread pool (`mesh_workload.cpp:compile`), and `GraphTracker`'s processors are
   `thread_local`, so `circular_buffer_allocate` events never reach the capture. Only the first call per key shows a
   peak, and that is the persistent scratch, not the CBs. Any fused CCL op with per-chip RT args will hit this.
   The final run used `EVAL_CAPTURE_L1=0`.
2. **Inline precompile + test-body mesh opens.** The acceptance file opens its meshes inside test bodies (a module
   mesh cache), which the inline NO_DISPATCH collection pass cannot handle: the real pass then reads back all-zero
   inputs. Run it with `--no-precompile`. (The op-side half, caching address-0 mock allocations, is fixed.)
3. **Re-opening a fabric mesh in-process under `--dev`** intermittently fails at `open_mesh_device` ("Timed out while
   waiting for active ethernet core … to become active again"). Reproduced with stock `ttnn.all_gather`
   (`probes/reopen_all_gather_probe.py`), so it is not this op. Run mesh-shape groups in separate processes.

## Performance (Phase 0 measured, for the perf refinements)

Traced, 6 back-to-back calls, `test_distributed_rms_norm_perf.py` under `--profile` (no `--dev`: the watcher distorts
timing). Slowest device of the 1×4 ring per steady-state call, bf16 x / TILE γ, Ring, 2 links:

| Case | Regime | Device kernel ns (median, range) | roofline / target µs | composite today µs |
|------|--------|----------------------------------|----------------------|--------------------|
| 640×1792 (PERF FOCUS) | R3-X, 100 cores | ~26,500 (25,900–27,650) | 10.6 / 14.1 | 92.1 |
| 32×1792 decode | R3-X, 14 cores | ~11,400 (11,250–12,100) | 2.0 / 2.7 | 87.2 |
| 2048×1280 | R1-X, 64 cores | ~43,100 (42,700–43,980) | 22.1 / 29.4 | 93.0 |

The golden runner's `device_kernel_ns` for the loose cases (66–272 µs) comes from one untraced call. It includes the
chips' launch skew absorbed by the ready fence, so it is not a perf number.

## Recommendations

- Queue (see `op_requirements.md`): R1 sharded placements → R2 RM γ + alignment → R3 perf knob co-tune (flagged
  640×1792) → R4 bf8b / 16-bit DEST → R5 exchange-transport rework (flagged 640×1792). The perf-flagged config is
  fully supported at Phase 0, so R1/R2 are ordered by difficulty, not by the anchor rule.
- The decode case (11.4 µs vs 2.7 µs target) is dominated by fixed costs: exchange latency, in-chip gather / inv
  distribution, launch. R5's transport work and lamp L2 (smaller `G_w`) are the levers. It is not flagged as a perf
  target, so it is a guard-set member, not a queue entry.
- fp32 bias (median got/true 0.9991) has no failing cell and no concrete lever in scope. An SFPU fp32 multiply would
  remove TF32 truncation at a measured ~0.58× speed cost (catalog `compute_fusion`); not recommended.
- γ multicast (regime G-MC, deferred) is worth ≤ 0.33 MB DRAM at 640×1792 (100 cores × 6 tiles × 544 B). It is a
  possible lever inside R3 if DRAM time dominates after the waves overlap. Not filed separately.
