# Verification Report: distributed_rms_norm

Box: Blackhole QuietBox 2 (4 chips; system mesh 2×2, snake ring opened as 1×4), `FABRIC_2D_TORUS_XY`, 14 KiB + 64 B
router payload. Every test run through `scripts/run_safe_pytest.sh --dev` / `eval/eval_test_runner.sh`.

## Escalation (infra, not the op) — read first

1. **The golden runner's L1 capture force-fails this op's passing cells.** With the default `EVAL_CAPTURE_L1=1`,
   the first golden run reported 1352 `test_op` + 9 ring-mock failures in the first 20 % of the matrix. Each was
   `eval.l1_profiling.L1ProfilingError: L1 profiling returned peak_total=0`, raised by `metrics_plugin` *after* the
   op had passed. Root cause, verified with a graph-capture probe: a heterogeneous `MeshWorkload` (one program per
   chip, as every fused CCL op with a per-position program needs) compiles and allocates its CBs on the mesh
   device's thread pool (`tt_metal/distributed/mesh_workload.cpp`, `MeshWorkloadImpl::compile`), while
   `GraphTracker` processors are `thread_local` (`tt_metal/impl/graph/graph_tracking.cpp:23`). The capture
   therefore records **no** `circular_buffer_allocate` event. It only sees the 288 B landing scratch on a cold call,
   so every warm call reads 0. The op has no lever on this. The golden numbers below come from
   `EVAL_CAPTURE_L1=0`. Fix belongs in tt-metal graph tracking (propagate processors to the pool threads) or in
   `eval/l1_profiling.py` (tolerate multi-program mesh workloads).
2. **Up-front precompile zeroes module-scoped inputs.** Under `run_safe_pytest.sh`'s default precompile pass, the
   new precision baseline read its `ttnn.from_torch` inputs back as zeros (PCC 0.0, outputs 0). The implementer
   saw the same thing with the perf driver. It is a harness artifact: with `--no-precompile` all 12 cases pass. The
   acceptance suite is unaffected. Both measurement files skip themselves under `UP_FRONT_COLLECT=1` and say so.
3. **Phase 0's reported golden counts were from a sample and are wrong.** The implementer reported "852 passed /
   0 wrong-result failures, 31 L1 refusals" from a 1620-cell sample. The full matrix has **440** L1-refused cells
   plus 15 refused translated tests. Its perf claim ("exchange hidden") used the untraced last-launched chip. On the
   traced slowest chip, 6 µs of exchange is exposed at the focus case (Perf baseline below).

## Code Review

Fixed in place:
- **Deprecated API**: `copy_tile_to_dst_init_short(cb_stats)` → `copy_init(cb_stats)` in the compute kernel's
  cross-device fold (`tile_move_copy.h`: removed after 20-09-2026). The remaining deprecation warnings in the build
  log come from `kernel_lib/reduce_helpers_compute.inl`, not this op.
- **`footprint()` drifted from the allocation** (ledger currency, single source). It counted `cb_partial_col` on
  the D1 / D1-HS regimes, which never allocate it; it counted `cb_gather` with `max_owned` while the CB allocates
  `max(1, max_owned)`; and it hard-coded `4096` for the Float32 stats pages. Now `footprint()` mirrors
  `create_mesh_program_descriptor` term for term, with one `STATS_FORMAT` constant used by both the CB descriptors
  and the footprint.
- **Dead code**: removed the overwritten `G_w, G_h, rows_max = …` tuples in the HEIGHT / WIDTH branches of
  `local_plan()` (`G_h` is derived from the members).
- **Registry drift**: added `None` to `SUPPORTED["num_links"]` (see Registry Conformance).
- **`l1_ledger.md` total** re-derived from the code (below).

Reviewed and found conformant (no change):
- **Blocking-model fidelity.** Every knob the design names reaches the kernels as one host constant or CT arg:
  `S`, `R`, `q_w`, `num_waves`, `C`, `STREAM_DEPTH`, `SCALE_BLOCK_TILES`, `MIN_SLICE_TILES`, `MAX_GROUP_CORES`,
  `WAVES_MAX`, `READ_LOOKAHEAD_WAVES`. CB capacities and trip counts derive from them, and the writer's `WP` and
  the compute's `NCH·C` come from the same `plan`. `cb_x_tiles = R·WS` scales with `R = ceil(T / G_h)` and is
  guarded by the footprint predicate. Its streaming fallback (`x_depth = 2`) is **not built**, so the predicate
  refuses instead of degrading. That is the Refinement 1 gap, not a collapsed knob on the shapes that fit.
- **Expression.** Reads are issued per wave with one transaction-id barrier per wave. Square / collapse / scale
  are `grid(q_w, ·)` chains per wave. The writer's quantum is one `C`-tile chunk per tile-row, which is the
  design's "one write barrier per C-tile quantum" (`cb_output` streams over `tile_rows`). `q_w = 1` on all three
  perf cases, so no per-unit handshake inside a multi-row block is serialized.
- **Performance conformance.** 98 / 106 workers busy at the focus case (D3, `G_w 14 × G_h 7`), 104 / 106 at
  2048×1280. Both dataflow halves are batched. x is read once, the output written once.
- **Design deviations** are recorded in `op_design.md` → Implementation deviations, each with a measured reason:
  port placement, per-wave `own_stats`, wave-ordered reads, the Phase B poll loop, per-wave bit-field `SEM_GATHER`,
  inv all-gather on NCRISC, `x_depth = 2` not built.
- **Helpers.** Square (`eltwise_chain` PerRow), collapse and chip-sum (`reduce`, `AccumulateViaAdd`/`Skip`), and the
  scaling passes (`eltwise_chain` strided / broadcast) use kernel-lib helpers. The raw uses (SFPU fp32 fold,
  `finalize_col0_body`, faces-0/2 combine transport, port kernel) carry the design's helper-mismatch table. The
  combine transport keeps the base op's measured `mcast_pipe` bypass (faces-0/2 subset writes).
- **Correctness hygiene.** `TensorAccessor` everywhere, `void kernel_main()`, `api/dataflow/dataflow_api.h`
  includes, `noc_async_atomic_barrier()` at reader end. CB push / wait / pop counts match per CB (checked
  `cb_x_tiles`, `cb_inv_rms`, `cb_partial_col`, `cb_gather`, `cb_chip_partial`, `cb_stats`, `cb_rms_root`,
  `cb_output`, `cb_gamma_tiles`, `cb_x_gamma`). A ragged `n_own < q` push only happens on a core's last wave
  (owners per member are fixed across full waves), so no window straddles a ring end.
- **Broadcasts.** γ `BroadcastDim::Row` from row-0 prefixes, inv `BroadcastDim::Col` from faces 0/2. No
  full-tile fills of a broadcast value.

Prompt `## Rules` (`eval/prompts/distributed_rms_norm.txt`): the one MUST holds. The op is one fused
`ttnn.generic_op`; it calls no norm op, CCL op or example op, and has no host round-trip on a warm call. The landing
tensor / global semaphores are created and synchronized only on the cold call.

Advisory (soft guidance, not met everywhere): the prompt's *Precision* section asks that the exchanged partials be
summed in fp32 "whatever the input dtype". Under `fp32_dest_acc_en=False` the cross-device fold runs in 16-bit DEST
(`copy_tile` of the Float32 `cb_stats` page into bf16 DEST). Measured: `inv_rms` is +0.27 % high on random rows at
16-bit DEST, against exact on constant rows. Moving the fold onto the reader (fp32 records, RISC-V adds) fixes it
and is also an OVERLAP lever, so it is folded into Refinement 3.

## Registry Conformance

- `INPUT_TAGGERS` has exactly `alignment` and `rank`, both `(inputs, axes)` on the per-device shape. `SUPPORTED`
  covers all 12 axes. `EXCLUSIONS` = `{float32, fp32_dest_acc_en=False}`. `validate()` checks SUPPORTED per axis,
  then EXCLUSIONS, raising `UnsupportedAxisValue` / `ExcludedCell`. The entry point calls `validate()` first. The op
  file declares no `INVALID`. `default_compute_kernel_config()` is the single default, read by `axes.py`.
- **Auto-fix (drift)**: `SUPPORTED["num_links"]` gains `None`. 246 translated calls omit `num_links`. `axes.py`
  tags the raw `None`, while `validate()` canonicalizes it to every discovered link and gates on that, so the op
  supports it. The registry listed only `{1, 2}`, which made those passing calls `xpass_drift` (246) and their
  failing siblings `xfail_wrong_mode` (15). After the fix, the CLI re-run on the same results gives
  `xpass_drift` 1 and `xfail_wrong_mode` 0. TARGET is `{1, 2}`, so `test_op` gains no cells.
- The remaining `xpass_drift` (1) is a false positive: `test_translated.py::test_pre_all_gather_non_welford_fp32_requires_fp32_dest_acc`
  wraps the excluded `{float32, fp32_dest_acc_en=False}` call in `pytest.raises`. The op raises `ExcludedCell`, so
  the test passes, and the classifier reads that as an xpass. The exclusion is behaving as intended.
- **INVALID audit** (`feature_spec.py`): 6 entries, all the γ "none" canonicalization (`gamma_mode` ↔
  `gamma_dtype` / `gamma_layout`, all describing γ). Single-tensor coupling holds, there are no "not built yet"
  entries, and the no-weight canonicalization is present. The canonical bf8b + ROW_MAJOR entry is correctly absent
  because the input is TILE only and `gamma_dtype` has no bf8b (the spec says so). No issue.

## L1 Ledger Audit

- **Currency**: every declared CB has a row and every row a CB. The `Total per-core footprint` formula had drifted
  (`cb_chip_partial` / `cb_partial_col` are two deep, the record scratch is `(G+1)·q`, combine CBs exist only when
  `G_w > 1`). Re-derived, it now matches `footprint()`: focus case **211,584 B** (was stated 203,264), decode
  154,240 B, 2048×1280 281,216 B.
- **Capacity vs live set**:
  - `cb_stats` holds `G·q` Float32 tile pages (16 KiB at the focus case) for a live set of 32 fp32 values per page
    (column 0). The tile pages exist only because the SFPU fold reads through `copy_tile`. Folded into
    Refinement 3 (reader-side fold → one record per row).
  - `cb_gather` holds `G_w·q` full tiles whose consumers read faces 0/2. It is the largest CB (57 KB). Folded into
    Refinement 3 (lamp L5 compact combine wire).
  - No under-capacity found: `cb_x_tiles`, `cb_inv_rms` and `cb_x_gamma` span `R`. The streaming CBs
    (`cb_output`, `cb_sumsq_partial`) hold fixed windows over `tile_rows`.
- **Page format vs DEST**: every intermediate follows `intermediate_format(fp32_dest_acc_en)`. `cb_stats` is Float32
  under 16-bit DEST too, but it is a dataflow-written wire payload (exact fp32 records) that never traversed DEST,
  which is a sanctioned mechanism reason. No 16-bit page under fp32 DEST.
- **Disjoint lifetimes**: every pair carries a reason in `Shares with / why not` (concurrent across waves, or a
  different page format). OK.
- **Bounds**: every non-block symbol has a bound and a predicate (`G ≤ 8`, `G_w ≤ 24` / shard count,
  `num_waves ≤ 8`, footprint ≤ `l1_budget`). The op-sized terms (`R`, the resident γ slice) are footprint-guarded.
  Their missing streaming fallback is Refinement 1.
- **Data-movement budget**: present and consistent with the code (x once, out once, γ once per row-group,
  on-chip combine and inv bytes counted per tile-row). The cheapest-traffic split, G-MC (γ read once per slice
  column + multicast, −183 KB / −3.8 % DRAM at the focus case), is deferred for "helper friction". The base op
  built exactly this mechanism in its Perf 1 (`gamma_mcast_groups`), so that is not a positive reason. Folded into
  Refinement 2 as its first lever.
- **Block-size defaults**: held. Interleaved spreads `T × Wt` across the worker grid and keeps the whole assignment
  resident (coarsest block). Sharded uses the whole shard. The `fused_split()` key (`rows_max·S`, waves, `G_w`) is
  the design's derived rule for the fused grid; the base rule falls off a cliff at 2048×1280.
- **Total**: `F = R·Wp·T_in + T_i·(q_w + NW·q_w + 3q) + [G_w>1]·T_i·(2q_w + max_owned·G_w·q) + 4096·G·q
  + (G+1)·q·128 + γ·(Wp·T_g + R·WS·T_i) + STREAM_DEPTH·C·T_out + 2048`. `R` and `S` drive x and x·γ, `G_w` drives
  the gather, `G` drives stats and records, `C` drives the output stream.

## Precision Baseline

`tests/ttnn/unit_tests/operations/distributed_rms_norm/test_distributed_rms_norm_precision_baseline.py` (run with
`--no-precompile`): 1×4 ring, Ring, G = 4, TILE bf16 γ, eps 1e-5, HiFi4. Worst device per row. ULP is in the output
dtype against the fp64 reference rounded to it. The ratio is got/true over |ref| > 1e-3.

| Shape | Config | PCC | Max Abs Err | Mean Abs Err | Relative RMS Err | ULP max / mean | ratio median | ratio p05 / p95 |
|-------|--------|-----|-------------|--------------|------------------|----------------|--------------|-----------------|
| (1,1,32,1792) | bf16, fp32 acc | 0.999998 | 0.0333 | 0.00098 | 0.00185 | 1 / 0.116 | 0.99944 | 0.9966 / 1.0021 |
| (1,1,32,1792) | bf16, 16-bit acc | 0.999993 | 0.0966 | 0.00248 | 0.00481 | 3 / 0.643 | 1.00291 | 0.9972 / 1.0093 |
| (1,1,32,1792) | fp32, fp32 acc | 1.000000 | 0.0216 | 0.00065 | 0.00121 | 49084 / 12059 | 0.99904 | 0.9978 / 1.0001 |
| (1,1,64,1000) | bf16, fp32 acc | 0.999998 | 0.0374 | 0.00102 | 0.00187 | 1 / 0.124 | 0.99936 | 0.9965 / 1.0021 |
| (1,1,64,1000) | bf16, 16-bit acc | 0.999993 | 0.1486 | 0.00255 | 0.00493 | 3 / 0.662 | 1.00301 | 0.9973 / 1.0094 |
| (1,1,64,1000) | fp32, fp32 acc | 1.000000 | 0.0200 | 0.00059 | 0.00113 | 53473 / 11071 | 0.99913 | 0.9980 / 1.0002 |
| (1,1,640,1792) | bf16, fp32 acc | 0.999998 | 0.0459 | 0.00097 | 0.00186 | 1 / 0.124 | 0.99935 | 0.9965 / 1.0020 |
| (1,1,640,1792) | bf16, 16-bit acc | 0.999993 | 0.1324 | 0.00247 | 0.00476 | 3 / 0.643 | 1.00285 | 0.9971 / 1.0093 |
| (1,1,640,1792) | fp32, fp32 acc | 1.000000 | 0.0260 | 0.00060 | 0.00112 | 57381 / 11003 | 0.99914 | 0.9980 / 1.0002 |
| (1,1,2048,1280) | bf16, fp32 acc | 0.999998 | 0.0484 | 0.00099 | 0.00185 | 1 / 0.119 | 0.99940 | 0.9966 / 1.0021 |
| (1,1,2048,1280) | bf16, 16-bit acc | 0.999993 | 0.1498 | 0.00243 | 0.00475 | 4 / 0.644 | 1.00291 | 0.9971 / 1.0092 |
| (1,1,2048,1280) | fp32, fp32 acc | 1.000000 | 0.0353 | 0.00060 | 0.00110 | 54684 / 10790 | 0.99916 | 0.9980 / 1.0002 |

**Assessment**: bf16 with fp32 accumulation is at the format limit: ULP max 1, mean 0.12, PCC 0.999998, identical
across shapes and devices. All devices agree, as expected since every device folds the same G records in the same
order. The ratio spread is tight around a slightly non-1.0 median, so I triaged it as a possible scale bug with a
constant-input probe. It is not one:
- **fp32 input**: a constant-1 row returns exactly `0.99951171875` = 1 − 2⁻¹¹, while the true value is
  0.999995. That is the fp32 `inv_rms` truncated to tf32 when the FPU unpacks it to SrcB for the multiply.
  Blackhole's FPU operands are tf32 with round-toward-zero, so every FPU multiply of fp32 data biases low by up to
  2⁻¹¹ (median −0.09 % with γ: two truncated multiplies). fp32 inputs therefore get ≈ tf32-accurate outputs
  (rel-RMS 1.1e-3, ~11k fp32 ULP mean), not fp32-accurate ones. This is inherited from the base op's FPU scaling
  chain. A fix needs SFPU (fp32) scaling for fp32 inputs, a perf trade-off with no failing cell. Recommendation
  only.
- **bf16, 16-bit DEST**: constant rows are exact. Random rows give `inv_rms` +0.27 % high: Σx² is accumulated, and
  folded across devices, in 16-bit DEST, so small contributions are lost. This is the expected cost of
  `fp32_dest_acc_en=False`. The cross-device share of it goes away with the reader-side fp32 fold (Refinement 3).

**Recommended tolerances**: bf16 + fp32 acc: PCC ≥ 0.9999, atol 0.06, rtol 0.01. bf16 + 16-bit acc: PCC ≥ 0.9999,
atol 0.2, rtol 0.02. fp32: PCC ≥ 0.99999, atol 0.05, rtol 2e-3 (tf32 operand truncation, not 1e-5). bfloat8_b:
golden PCC targets (0.99) hold across the matrix.

## Verifier CLI Summary

`EVAL_CAPTURE_L1=0 eval/eval_test_runner.sh eval/golden_tests/distributed_rms_norm/ /tmp/drms_verify`, then
`python3 -m eval.verify_supported /tmp/drms_verify ttnn.operations.distributed_rms_norm` (after the `num_links`
drift fix). Trimmed report: `verifier_report.json` in this directory (summary + every loud-category cell).

- supported_pass: 10426 (test_op 10160; translated + loose cases 266)
- xfail_expected: 2040 (exactly the `{float32, fp32_dest_acc_en=False}` EXCLUSION, both link counts)
- invalid_skipped: 33696
- infeasible_skipped: 320 (harness shard geometry does not fit L1; uncharged)
- no_axes_found: 87 (ring-mock 24, regression 12, untagged translated tests; all passed)
- **supported_fail: 455**. All are host-side refusals: 440 `test_op` + 3 translated L1 footprint refusals, and 12
  translated output-reshard refusals. No wrong result, no hang, no numerical failure. They stay failing by rule
  (OOM-class refusals become refinements, not EXCLUSIONS) and are exactly Refinement 1's done-when.
- xpass_drift: 1 (false positive, above)
- xfail_wrong_mode: 0

Also green: acceptance 192 passed / 1 skipped (`--dev`), ring-mock 24/24, regression 12/12, perf cases 3/3,
translated 264/279 (the 15 = the refusals above).

## Perf baseline (new, traced)

`tests/ttnn/unit_tests/operations/distributed_rms_norm/test_distributed_rms_norm_perf_traced.py` (new; in-process
device profiler, as the `fabric_ring_exchange` bench). 1×4 ring, Ring, 2 links, bf16 x / TILE bf16 γ, HiFi4 +
fp32 acc, median of 8 traced calls (untraced: 5).

| case | untraced last-launched | untraced slowest | **traced slowest** | traced, exchange stubbed | exposed exchange | roofline / target |
|---|---|---|---|---|---|---|
| 640×1792 | 20.05 | 68.75 | **25.86** | 19.89 | **5.96** | 9.0 / 11.9 |
| 32×1792 | 9.36 | 23.54 | **13.07** | 9.64 | **3.43** | 0.45 / 0.6 |
| 2048×1280 | 32.17 | 87.13 | **41.83** | 32.40 | **9.43** | 20.5 / 27.3 |

(µs.) Breakdown on the slowest chip at the focus case: local work 19.9 µs (2.2× the DRAM roofline) + exposed
exchange 6.0 µs. At 2048×1280 the exposed part (9.4 µs) exceeds the whole relay exchange measured alone
(≈ 5.4 µs), so the wave pipeline serializes somewhere. Refinement 3 starts by diagnosing that.

## Recommendations

- **Queue order** (`op_requirements.md`): Refinement 1 is generality, the only one left since SUPPORTED == TARGET
  (the base op's L1 fallbacks plus output resharding). Then three perf refinements ranked by the measured
  breakdown: Refinement 2 LOCAL, Refinement 3 OVERLAP, Refinement 4 TRANSPORT. LOCAL goes before the cross-device
  work as the fused-CCL rule allows when local dominates with headroom: 19.9 of 25.9 µs, 2.2× its roofline, with
  the base op's own perf rounds (γ multicast, srcB hold, NoC swap) unported. OVERLAP is then tuned against the
  faster local schedule.
- **Run the golden suite with `EVAL_CAPTURE_L1=0`** until the L1-capture escalation is fixed.
- **fp32-input accuracy** (no failing cell): outputs are tf32-accurate because of FPU operand truncation. If a model
  needs fp32-accurate output from fp32 input, the lever is an SFPU scaling path for fp32 inputs, a measured
  perf / precision trade-off. Not queued.
- **Decode (32×1792)** sits at 13 µs against a 0.6 µs target. The prompt accepts that a case this small misses its
  target. Most of it is the fixed fence (~5 µs) plus the serial per-wave chain on 14 cores. Refinements 2 (lamp L9:
  fewer slice cores at T = 1) and 3 both touch it.
- **L1 pressure without OOM**: the focus case uses 207 KiB per core. `cb_gather` (`G_w`-scaled) and `cb_x_gamma`
  (`R·WS`) are the terms that grow first. Refinement 3 shrinks the former, and Refinement 1 gives the latter a
  per-chunk fallback.
