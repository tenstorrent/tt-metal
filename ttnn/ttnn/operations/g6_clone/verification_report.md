# Verification Report: g6_clone

Phase 0 scope: `dtype=[float32]` and `layout=[TILE]`, DRAM interleaved, regime `flat_page_range`.
The op is a reader (NCRISC, NoC0) → one CB → writer (BRISC, NoC1) byte copy with no compute kernel.
Board: Blackhole p150b, chip 4, 11×10 = 110 compute cores, 8 DRAM banks.

## Code Review

### Fixed in this pass

| # | Finding | Fix |
|---|---------|-----|
| 1 | **Collapsed block-count knob.** `block_pages` was hard-wired to `min(core_pages_max, l1_cap)`. A core could never cut its assignment into more than one block, so the design's overlap lamp (`ceil(core_pages/2)`, depth 2) could not be measured without editing the plan. | Added host constant `BLOCKS_PER_CORE_KNOB`, default 1: `block_pages = min(ceil(core_pages_max / BLOCKS_PER_CORE_KNOB), l1_cap)`. The CB capacity, CT arg and kernel trip counts still derive from `block_pages`, which stays the one source. The default of 1 is justified by measurement (see *Perf baseline*). |
| 2 | **DRY / BB-08.** The entry point and `g6_clone_plan()` each wrote the `compute_plan(device, num_pages, page_size)` call themselves. A future input added to one would silently desync the reported plan from the plan that ran. | Added a single `_plan(input_tensor)` that both call. The comment names the cache-key coupling. |
| 3 | **Untested ring path.** On every INPUTS shape a core gets one block (depth 1), so the depth-2 ring wrap, the nominal-count push/pop on a ragged last block, and the writer's flush-before-pop never ran. | Added `test_g6_clone_multi_block_ring`: p2 forced to blocks of 5/5/4 and 5/5/3 at depth 2. It checks the plan, then checks bit-exactness. |
| 4 | Ledger currency | Added the `BLOCKS_PER_CORE_KNOB` symbol and the capacity expression. Added a deferral-expiry note on `logical_rows_partial_tile` (see *L1 Ledger Audit*). |

### Checked, no change needed

- **Kernels.** Both use `void kernel_main()`, `api/dataflow/dataflow_api.h`, and `TensorAccessor` / `TensorAccessorArgs`, not `InterleavedAddrGen`. The buffer address is a common RT arg, patched in place on a cache hit (BB-07).
- **CB sync.** Push equals wait: `block_pages` per block on both sides, with the nominal count even on a ragged block. Capacity is `cb_depth_blocks · block_pages`, so the ring closes on every cycle.
- **Expression.** `load_block`: 1 reserve, then all page reads issued back to back, then 1 barrier, then 1 push. `store_block`: 1 wait, then all writes, then 1 `noc_async_writes_flushed`, then 1 pop. No per-unit handshake on either side. The writer ends with `noc_async_write_barrier` (BB-03). The reader has nothing in flight at return.
- **Helper usage.** No kernel_lib helper expresses a DRAM page range → CB → DRAM page range behind one barrier per block (`op_design.md`, *Helpers considered and rejected*). The engine copy (`compute_kernel_lib::copy`) is forbidden by spec rule 1. There is no multicast or semaphore code to migrate.
- **Broadcast.** Not applicable.
- **Prompt `## Rules`, MUST items (all hold).**
  - Bit-exact with no engine path: there is no compute kernel.
  - The output is a new buffer: `allocate_tensor_on_device(x.spec)`.
  - Exactly one `generic_op` dispatch per call, with no wrapped ttnn op.
  - DRAM is crossed once each way.
  - No host-side layout or pad op.
  - Partial-tile logical elements are independent of the input's padding: whole pages land at the same page id, and padding goes only to output padding.
  - No size, alignment or rank axis or tagger.
  - The core count is the blocking choice: 110 cores on p4's single tile row.
  - L1 stays ≤ 1347776 B (`block_pages_l1_cap`).
  - The cache key covers everything that shapes the descriptor.
  - Core ranges are ≤ 2 rectangles.
  - Per-block profiler zones are present.
  - `g6_clone_plan` reports the plan and the roles.
- **Prompt rules, soft items.** None unfollowed.

### Design conformance

| Dimension | Verdict |
|-----------|---------|
| Algorithm | Matches: NoC byte moves only, flat page axis. |
| Pipeline topology | Matches: reader NCRISC/NoC0, writer BRISC/NoC1, one CB. |
| Work distribution | Matches. `split_work_to_cores(grid, pages, row_wise=True)` over the full queried grid (110 cores on every shape of ≥ 110 pages; 48 cores on Q3's 48 rows). Both NoC halves are batched. |
| Blocking-model fidelity | `block_pages` (now with `BLOCKS_PER_CORE_KNOB`), `CB_DEPTH_KNOB` and the grid are parameters with one source each. The CB is sized to `cb_depth_blocks · block_pages`, clamped by `block_pages_l1_cap`: a bounded clamp, not an unconditional op-sized CB. |
| Axis accounting | `page` is spanned (capacity scales with `block_pages`). `in_page` is spanned (the whole aligned page). The block walk is streamed (`cb_depth_blocks` slots). |
| Inter-core communication | None, as designed. |

No mismatch. The one design statement measurement contradicts is the reason given for deferring `bank_affine_runs`: "it changes only the transaction count". It is carried into Refinement 2 (see *Recommendations*).

## Registry Conformance

- The op file has `INPUT_TAGGERS = {dtype, layout}`, each `(inputs, axes)`. `SUPPORTED = {dtype: [float32], layout: [TILE]}` and `EXCLUSIONS = []`.
- `validate()` runs the device, rank and placement `ValueError`s, then the taggers, then SUPPORTED (`UnsupportedAxisValue`), then EXCLUSIONS (`ExcludedCell`).
- `g6_clone()` and `g6_clone_plan()` both call `validate()` first.
- The op file declares no `INVALID`.
- No SUPPORTED auto-fix was needed: no `xpass_drift` and no `xfail_wrong_mode`.
- **INVALID audit** (`feature_spec.py`): `INVALID = []`, which is correct.
  - Every TARGET (dtype, layout) pair is storable.
  - There is a single tensor, so no cross-tensor coupling exists.
  - `bfloat8_b` is not in TARGET, so the canonical bf8b+ROW_MAJOR entry does not apply.
  - The op is not norm-like.

## L1 Ledger Audit

- **Currency.** There is one CB, `cb_page_staging`. Its ledger size `cb_depth_blocks · block_pages · aligned_page_bytes` matches the `CBDescriptor.total_size` in the code, and its format `x.dtype` matches. The ledger now includes `BLOCKS_PER_CORE_KNOB`.
- **Capacity vs live set.**
  - *Over:* none. `cb_depth_blocks = min(2, num_blocks_max)`, so a one-block core has no idle second slot.
  - *Under:* none. Capacity scales with `block_pages` (spanned) and with the whole aligned page.
- **Page format vs DEST.** Not applicable: there is no DEST and no `fp32_dest_acc_en`. The payload is dataflow-only.
- **Disjoint lifetimes.** There is only one buffer.
- **Bounds.** `aligned_page_bytes ≤ 673888` is enforced by the host `RuntimeError`. `core_pages_max` is clamped by `block_pages_l1_cap`. The total is closed-form and ≤ `L1_BUDGET_BYTES`.
- **Traffic budget.** `x` is read from DRAM once and `out` is written once, with zero cross-core traffic. This matches the code.
- **Cheapest-traffic split.** `logical_rows_partial_tile` is deferred with a positive reason: its only Phase 0 shape is `(1,1)` f32, whose cost is fixed program overhead. That reason **expires with Refinement 1**: adding bf16 makes Q8 `[1,1,1,62080]` (19.4 µs measured, the same as the full-tile p4) and F13 `[1,5120]` reachable. Refinement 1 therefore carries the regime, and the ledger says so.
- **Block-size defaults.** Held. The split spreads over the full grid, then takes the coarsest block that fits (the whole per-core assignment). Departing from it (2 to 7 blocks per core) measured flat on p1, and 28 blocks regresses by 10%. The departure is not taken.
- **Per-core footprint.** `L1 = cb_depth_blocks · block_pages · aligned_page_bytes`. It is 114688 B on p1 (8.5% of the budget). `block_pages` scales with `tensor_pages / G`, divided by `BLOCKS_PER_CORE_KNOB`, up to the L1 cap. `aligned_page_bytes` is fixed by dtype and layout.

## Precision Baseline

`tests/ttnn/unit_tests/operations/g6_clone/test_g6_clone_precision_baseline.py`: float32 TILE, randn input, gated on bit equality.

| Shape | PCC | Max Abs Err | Mean Abs Err | Relative RMS Err | Max ULP | got/true ratio (median / p5 / p95) |
|-------|-----|-------------|--------------|------------------|---------|------------------------------------|
| (1,1) | 1.0 | 0.0 | 0.0 | 0.0 | 0 | 1.0 / 1.0 / 1.0 |
| (1,12,128,128) | 1.0 | 0.0 | 0.0 | 0.0 | 0 | 1.0 / 1.0 / 1.0 |
| (8,12,128,128) | 1.0 | 0.0 | 0.0 | 0.0 | 0 | 1.0 / 1.0 / 1.0 |
| (16,12,128,128) | 1.0 | 0.0 | 0.0 | 0.0 | 0 | 1.0 / 1.0 / 1.0 |

**Assessment.** Bit-exact on every shape. The golden suite and `test_regression.py` show the same on uniform random bit patterns: NaN payloads, ±inf, ±0, subnormals and max finite. The ratio spread is exactly 1.0, so there is no scale signature.

**Recommended tolerances.** Exact, for every dtype (`helpers.TOLERANCES` is `("exact", None)`). No PCC or RMS limit applies.

## Verifier CLI Summary

Golden run: `eval/eval_test_runner.sh eval/golden_tests/g6_clone/` gave 10 passed and 28 xfailed of 38, with 0 failed, 0 errors and 0 hangs. Report: `golden_results/verifier_report.json`.

- supported_pass: 7 (4 INPUTS f32 cells, plus loose p1, p2 and `poisoned_padding_nan_f32`)
- xfail_expected: 28 (14 bf16 TILE, 3 bf16 ROW_MAJOR, 5 uint32 RM, 3 int32 RM, plus loose p3, p4 and `poisoned_padding_nan_bf16`). All 28 are carried by Refinement 1.
- invalid_skipped: 0
- supported_fail: 0
- xpass_drift: 0
- xfail_wrong_mode: 0
- no_axes_found: 3. These are the non-registry `test_regression.py` cases (special bit patterns, new storage, repeat), and all 3 pass.

**The xfail bucket is all SUPPORTED gating, with no kernel gap.** Probe `probes/probe_006.py` widened SUPPORTED to TARGET in-process and graded every refused INPUTS and LOOSE_CASES cell with `helpers.run_g6_clone`: **28/28 pass bit-exact with zero kernel change**, including `poisoned_padding_nan_bf16`.

Acceptance: `tests/ttnn/unit_tests/operations/g6_clone/` gave 16/16 (12 acceptance + 4 precision).

## Perf baseline (device kernel ns, profiled build, median of 6 calls)

`probes/perf_baseline_probe.py` widens SUPPORTED in-process for the bf16 and RM rows.

| Case | Shape | Pages × bytes | Cores | Median ns | DRAM bytes (r+w) | Effective GB/s |
|------|-------|---------------|-------|-----------|------------------|----------------|
| p1 (Q2) | (16,12,128,128) f32 TILE | 3072 × 4096 | 110 | 63182 | 25165824 | 398 |
| p2 (Q6) | (8,12,128,128) f32 TILE | 1536 × 4096 | 110 | 32492 | 12582912 | 387 |
| Q13 | (1,12,128,128) f32 TILE | 192 × 4096 | 110 | 4700 | 1572864 | — (fixed cost) |
| p3 (F4) | (1,512,5120) bf16 TILE | 2560 × 2048 | 110 | 25502 | 10485760 | 411 |
| p4 (Q15) / Q8 | (1,1,32,62080) / (1,1,1,62080) bf16 TILE | 1940 × 2048 | 110 | 19431 | 7946240 | 409 |
| F13 | (1,5120) bf16 TILE | 160 × 2048 | 110 | 2351 | 655360 | — |
| Q3 | (16,3,2560) bf16 RM | 48 × 5120 | 48 | 2059 | 491520 | — |

**p1 against its goal.** The stock time is 68.32 µs and the bound is 1.35 × 54.61 = 73.72 µs, so p1 is already inside both. The gap to the lead's roofline (54.61 µs, 461 GB/s) is 1.16×. At the p150b DRAM spec of 512 GB/s, p1 runs at **77–78% of DRAM peak**.

**Lamp measurements on p1.** `probes/grid_cap_probe.py` and `probes/block_knob_probe.py`, 6 calls each:

| Variant | Median ns |
|---------|-----------|
| grid 11×10 (110 cores) | 63785 |
| grid 11×8 (88) | 64060 |
| grid 11×6 (66) | 64084 |
| grid 8×8 (64) | 64418 |
| grid 11×4 (44) | 64004 |
| 1 block/core (`block_pages` 28, depth 1) | 63594 |
| 2 blocks (14, depth 2) | 64410 |
| 3 blocks (10) | 63504 |
| 4 blocks (7) | 63996 |
| 7 blocks (4) | 63915 |
| 28 blocks (1) | 70269 |

Reading:
- **Occupancy lamp: closed.** 44 cores do what 110 do, so p1 is not per-core issue-bound. The full grid costs nothing measurable, so it stays.
- **Overlap lamp: closed.** Read/write overlap within a core buys nothing (1 to 7 blocks are flat). The one-page-per-barrier trap costs 10%.
- **The bound is DRAM/NoC contention.** Every core walks all 8 banks: page `p` sits in bank `p mod 8`, and each 28-page range starts on bank 0 or bank 4. That is the maximal-route-overlap regime of `/perf-ceiling-dm` Step 4b and `tech_reports/Saturating_DRAM_bandwidth`. Refinements 2 and 3 address it.

## Recommendations

1. **The design's `bank_affine_runs` deferral reason is falsified by measurement.** The design defers it as "transaction count only; pays only when a DM RISC is issue-bound". The grid-cap sweep shows p1 is not issue-bound, yet it sits at 77% of DRAM peak. What bank affinity would buy is bank and route locality: one bank per core, with cores near their bank's DRAM column. A smaller command count is incidental. This is Refinement 2's lever; the design's Regimes row should be updated when that refinement lands.
2. **Refinement 1 must extend the cache key (BB-07).** `logical_rows_partial_tile` makes the descriptor depend on the logical `H mod 32` and `W mod 32` and on the tile geometry (which page ids are partial). The current key (`tensor_pages`, `aligned_page_bytes`, …) cannot tell `[1,1,1,62080]` from `[1,1,32,62080]`. Both have 1940 pages of 2048 B. An unextended key replays the full-tile program on Q8, or the reverse: the logical-rows program on p4, which is wrong output.
3. **Untested TARGET pairs.** Widening the axis lists in Refinement 1 also claims f32 ROW_MAJOR and int32/uint32 TILE, which INPUTS never builds: the taggers read the scenario, so only production pairs are cells. The kernels are dtype- and layout-agnostic byte movers, so those pairs are covered by construction. A logical-rows transfer for int32/uint32 partial tiles uses the same 4-byte element width as f32.
4. **Environment friction.** `python3 -m eval.verify_supported` fails to import ttnn with `undefined symbol: MPIX_Comm_shrink` unless `.tt_chip_scope` is sourced first: it sets `LD_PRELOAD` to the MPIX stub. `run_safe_pytest.sh` and `eval_test_runner.sh` source it themselves; a bare `python3` does not.
5. **No L1 pressure.** The largest per-core footprint on INPUTS is 114688 B (p1). `block_pages_l1_cap` (164 f32 tiles, 329 bf16 tiles, 131 rows of 5120 B) binds only above about 18000 f32 tiles at G = 110, and no TARGET shape gets near that.
