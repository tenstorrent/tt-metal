# Verification Report: g6_to_memory_config

Verified on one Blackhole device: 110 compute cores (11 × 10), 110 L1 banks (so `bank_cores_cover_grid`
holds and the bank regimes are live), 8 DRAM banks. The design is the owner-move design
(`op_design.md`); the kernels are byte movers with no compute kernel.

## Code Review

Fixes made in this pass:

| # | Finding | Fix |
|---|---------|-----|
| 1 | **DRY: the per-kernel `movers_per_owner` / `mover_index` CT args came from two derivations.** `create_program_descriptor` recomputed them from `len(configs)` plus an `if regime != "bounce"` special case, while `build_plan` already held `movers_per_owner` (forced to 1 for bounce). A knob change had two places to land. | `_mover_configs` now returns `(config, source, mover_index)` per kernel, and the CT args take `plan["movers_per_owner"]` directly. There is one source of truth. Knob liveness was re-checked by `test_g6_to_memory_config_perf_knobs.py` (15/15: `movers_per_owner` 1/2 × `bounce_depth` 2/4 in all five regimes, all bit-exact). |
| 2 | **Dead file.** `kernels/g6_to_memory_config_compute.cpp`, a 9-line stub, was referenced by nothing. The design has no compute phase ("no compute kernel, no arithmetic"). | Deleted. |
| 3 | **Ledger currency.** `l1_ledger.md` gave the bf4b 2183 bounce CB as 429120 B (745 pages). The code allocates `2 · ceil(745/2) · 576 = 429696 B`, which is what `plan_of` reports. | Ledger corrected. |
| 4 | Temporary verifier measurement harnesses (`test_tmp_verifier_{perf,knobs}.py`) were left in the test directory. | Removed. Their numbers are in this report. |

Reviewed and found conformant (no change needed):

- **Prompt `## Rules`.** These hold: one program per call (`ttnn.generic_op` once), no host-side tensor transformation, an exact returned placement, bit-exact pages (no unpack/repack), native H-non-aligned handling (whole pages, `ceil` per image), and every production shard form (ragged last shard, more cores than shards, multi-range and off-origin grids). The BB-05 host refusal is implemented in `validate()`. BB-07 is met: the cache key holds device, padded shape, tile shape, dtype, both placements with range order, and all three knobs, and a cache hit patches only the two base addresses. BB-08 is met by `plan_of`, which reports regime, movers, kernel core box and roles. BB-17 is met: each kernel group is one bounding-box `CoreRange`. BB-21 is met: zones per issue/drain/wait and a role dump. BB-03 is met: every push kernel and the bounce writer end in `noc_async_write_barrier`, and every pull kernel in `noc_async_read_barrier`. No MUST/MUST NOT violation was found.
- **Design conformance (algorithm, pipeline, work distribution).** The selection function in `build_plan` matches `op_design.md` exactly. All 105 golden cells land on the regime the design predicts; the acceptance test pins 16 production cases plus `push_bank` and `bounce`. Grid fill:
  - bank and bounce regimes launch the full 110-core grid (`split_work_to_cores(..., row_wise=True)` for bounce);
  - shard regimes launch every core that owns a shard of the moved side (owner count = shard count; the `fan_out` rejection is reasoned in the design).

  Both dataflow halves are batched:
  - pull and push issue the whole owned set, then one barrier;
  - bounce issues a block of reads, one read barrier and one push, then a block of writes, one `writes_flushed` and one pop.
- **Blocking-model fidelity.**
  - **Knobs are parameters.** `BOUNCE_DEPTH`, `MOVERS_PER_OWNER` and `L1_OP_BUDGET_BYTES` are single host constants. `bounce_block_pages`, the CB capacity, the per-core RT args and every CT arg derive from them. No CB scales with a whole-op dimension: the one CB is `bounce_depth · bounce_block_pages`, and `bounce_block_pages` is clamped by the L1 budget.
  - **The schedule reads as blocks.** Reader, writer and owner kernels each retire a whole block per synchronization. No per-page handshake exists anywhere.
  - **No half-turned split.** Per-core work is the whole owned shard or bank stripe, or `core_pages / bounce_depth`.
- **API usage.** The kernels use `TensorAccessor` throughout: sharded `shard_pages()`, interleaved `get_noc_addr()`, and both arg blocks compile-time. They use `void kernel_main()` and `api/dataflow/dataflow_api.h` includes. CB push/wait/reserve/pop counts are the nominal `bounce_block_pages` on both sides for every block, including the short last block, so the ring closes.
- **Helpers.** The design rejected kernel_lib helpers with file:line reasons (byte moves, no compute phase, no reuse operand for mcast). That stands; there is no raw multicast or semaphore code to migrate.

Advisories (not fixed; no behaviour impact):

- **Cross-language ABI constants.** `WALK_SHARD/BANK/BOUNCE` and `NO_SHARD` are written once in the host module and once in `g6_to_memory_config_common.hpp`. They are protocol enums, not block knobs, and the host comment points at the header. Mention it if either side ever changes.
- **Import-time alias.** `g6_to_memory_config.py` aliases `ttnn.reports = ttnn._ttnn.reports` at import. `ttnn.reports` is not exported by this tree's `ttnn/__init__.py`, and the immutable acceptance test calls `ttnn.reports.get_device_info`. This mutates the global `ttnn` namespace and should go away once `ttnn` exports `reports`.
- **BB-21 zone coverage.** `cb_push_back` / `cb_pop_front`, accessor construction and RT-arg reads fall outside named zones. They are single-digit-cycle operations.

## Registry Conformance

- **Op file.** `INPUT_TAGGERS` has 8 taggers, all with the `(inputs, axes)` signature, and `validate()` runs the same taggers over the same scenario dict the golden suite builds. `SUPPORTED` covers every tagged axis. `EXCLUSIONS` holds exactly the one entry the spec names. `validate()` checks in this order: device tensor (`ValueError`), then SUPPORTED (`UnsupportedAxisValue`), then EXCLUSIONS (`ExcludedCell`), then structural checks (`ValueError`), then the BB-05 budget (`ValueError`). It is the first line of `g6_to_memory_config()` and of `plan_of()`. The op file does **not** declare `INVALID`.
- **Phase 0 declaration.** It matches the spec's mandatory Phase 0 list exactly: dtype bf16, TILE, rank 2–4, both alignments, all four schemes and both buffers on each side, and the one EXCLUSION. Nothing is narrowed.
- **SUPPORTED honesty beyond the golden cells.** SUPPORTED claims combinations no production case exercises: DRAM height/block-sharded targets and DRAM-sharded inputs. These were probed (`probes/probe_014.py`, `probe_015.py`): DRAM interleaved → DRAM height-sharded and → DRAM block-sharded (bounce); L1 width-sharded → DRAM height-sharded (push_shard); L1 interleaved → DRAM height-sharded (push_bank); DRAM width-sharded → L1 interleaved (pull_bank), → DRAM interleaved (bounce); DRAM height-sharded → DRAM height-sharded (bounce), → L1 interleaved (pull_bank). All were bit-exact with exact placement. The claim is honest.
- **No XPASS evidence.** No SUPPORTED auto-fix was needed.
- **INVALID audit (`feature_spec.py`).** `INVALID = []`. This is correct: every axis is projected from a production case, so no impossible combination is formed. The canonical bf8b + ROW_MAJOR entry is unnecessary because TARGET's `layout` is `[TILE]` only, so the combination cannot arise. There are no cross-tensor couplings and no "not yet" entries. The DRAM width-sharded refusal correctly lives in the op's EXCLUSIONS, not in INVALID.

## L1 Ledger Audit

- **Ledger currency.** There is one CB, `cb_bounce_pages`, and one row. Index 0, page size `buffer_aligned_page_size()`, capacity `bounce_depth · bounce_block_pages · page_size` and format = input dtype all match the code. The one stale number (fix 3 above) is corrected. The pull and push regimes allocate no CB, as the ledger states.
- **Capacity vs live set.** Capacity equals live set: both slots are live by design. In axis accounting the CB spans `page` (`bounce_block_pages` per slot × `bounce_depth`), and `tile_row` / `tile_col` / `leading` fold into `page`. No over- or under-sized buffer was found.
- **Page format vs DEST.** Not applicable. The CB carries a byte payload that never enters DEST; no compute kernel exists.
- **Disjoint lifetimes.** Only one CB exists. Not aliasing it to the input or output is justified: in `bounce` both live in DRAM, and a NoC read lands only in the issuer's L1.
- **Bounds and closed form.** `P` enters CB sizing only through `core_pages_max`, which the `min(..., L1_OP_BUDGET_BYTES // (bounce_depth · page_size))` clamp bounds. The total is closed-form: `L1_op = output_share(regime) + [bounce] · bounce_depth · bounce_block_pages · page_size`.
- **Block-size defaults.** Interleaved regimes spread over the full grid and take the coarsest block: the whole bank stripe for bank regimes, `core_pages / bounce_depth` for bounce. Sharded regimes take one block = the whole resident shard. Bounce departs from the coarsest block (2 blocks of half the assignment), justified as "overlap at zero L1 cost", since the CB holds the whole assignment either way. This pass measured the lamp (L6): depth 16 vs 2 is −4% on Qwen 2183 and −7% on Qwen 2179. The knob is live, and depth is a co-tune lever for the perf refinements.
- **Data-movement budget.** Present and consistent with the code: each tensor crosses its tier once, with 1 NoC transfer per page for owner regimes and 2 for DRAM→DRAM bounce (the minimum). The cheapest-traffic split is implemented for every placement pair. The deferred `bounce_dram_shard` row carries a positive reason: it is unreachable in Phase 0 because the only DRAM-sharded targets are bf4b. It becomes reachable with Refinement 1 and is the lever of Refinement 2.
- **Per-core footprint.** The golden maxima are:
  - 192512 B: `pull_bank`, Qwen 2078; scales with `ceil(P / nb)`;
  - 40960 B: L1-sharded targets; scales with the shard;
  - 429696 B: `bounce`, bf4b 2183; scales with `bounce_depth · bounce_block_pages`, clamped.

  All are ≤ 1347776 B.

Nothing was folded into a refinement: the ledger had no finding beyond the stale number.

## Precision Baseline

`tests/ttnn/unit_tests/operations/g6_to_memory_config/test_g6_to_memory_config_precision_baseline.py` covers one production case per built regime, bf16:

| Case (regime) | Shape | PCC | Max Abs Err | Mean Abs Err | Relative RMS Err | Max ULP | got/true ratio median [p5, p95] |
|---|---|---|---|---|---|---|---|
| Qwen 2058 (pull_shard) | (1,16,1,256) | 1.000000 | 0 | 0 | 0 | 0 | 1.0 [1.0, 1.0] |
| Qwen 2105 (pull_bank) | (1,77,4352) | 1.000000 | 0 | 0 | 0 | 0 | 1.0 [1.0, 1.0] |
| Qwen 2071 (push_shard, ragged last shard) | (1,1,16,12928) | 1.000000 | 0 | 0 | 0 | 0 | 1.0 [1.0, 1.0] |
| Qwen 2077 shape (bounce, DRAM→DRAM) | (16,128,1536) | 1.000000 | 0 | 0 | 0 | 0 | 1.0 [1.0, 1.0] |

`push_bank` is covered bit-exact by the acceptance test. With the support gate lifted in-process, all 19 bf8b / bf4b golden cells are bit-exact too (`probes/probe_013.py`).

**Assessment**: a bit-exact byte move in every regime, as the contract requires. There is no rounding to characterize.
**Recommended tolerances**: exact, for every dtype (`helpers.TOLERANCES` is `("exact", None)`). The PCC/RMS tables are not used by this op.

## Verifier CLI Summary

Golden run: `eval/eval_test_runner.sh eval/golden_tests/g6_to_memory_config/ /tmp/g6_verify_golden` gives 179/205 passed, 0 failed, 0 hangs. The report is `verifier_report.json` (a copy is committed in this directory).

- supported_pass: 91 (all 86 bf16 production cells plus the 5 Phase 0 loose cases)
- xfail_expected: 26: 12 bf8b cells, 7 bf4b cells (bf4b plus the DRAM width-sharded EXCLUSION), and 7 loose cases (PERF FOCUS 2183, p2 2184, p3 2182, p4 2179, Panoptic 7, and the two outlier cases). All 26 are queued in Refinement 1.
- invalid_skipped: 0 (INVALID is empty)
- supported_fail: 0
- xpass_drift: 0
- xfail_wrong_mode: 0
- no_axes_found: 88. These are `test_regression.py` (non-registry: BB-03/06/07 regressions plus the placement check of every Phase 0 case), all passed.

Acceptance: `test_g6_to_memory_config.py` 24/24. Knob test: 15/15. Precision baseline: 4/4.

`TARGET − SUPPORTED` = `{dtype: [bfloat8_b, bfloat4_b]}`, plus the one EXCLUSION. Both go into Refinement 1, and no other axis gap exists. **Refinement 1 is therefore the only generality refinement.** This is not under-filing: every other axis of TARGET is already fully SUPPORTED.

## Device Performance Baseline

Device kernel duration was measured under the Tracy device profiler (`run_safe_pytest.sh --profile`), min of 3 calls. Only cells above 32 tiles were measured (`perf_rules.txt`). The roofline is DRAM-bound at the lead's 460.8 GB/s (2183's figure: 2 × 47185920 B in 204.80 µs). "×roof" is measured / roofline (≤ 1.35 is the lead's goal). Rows marked † ran with the support gate lifted in-process (outside Phase 0 SUPPORTED); the disk was untouched.

| Case | Regime | Pages × B | Cores | Device ns (min) | DRAM roofline | ×roof |
|---|---|---|---|---|---|---|
| **Qwen 2183 † (PERF FOCUS)** | bounce | 81920 × 576 | 110 | **306847** (307–318 µs over 6 calls) | 204.8 µs | **1.50** |
| Qwen 2184 † (p2) | bounce | 64640 × 576 | 110 | 250472 | 161.6 µs | 1.55 |
| Qwen 2179 † (p4) | bounce | 46080 × 576 | 110 | 170569 | 115.2 µs | 1.48 |
| Qwen 2182 † (p3) | bounce | 17920 × 576 | 110 | 60613 | 44.8 µs | 1.35 |
| Qwen 2178 † | bounce | 7680 × 576 | 110 | 25786 | 19.2 µs | 1.34 |
| bf16 DRAM→DRAM (16,128,1536) | bounce | 3072 × 2048 | 110 | 30227 | 27.3 µs | 1.11 |
| Qwen 2078 | pull_bank | 10240 × 2048 | 110 | 53516 | 45.5 µs | 1.18 |
| Qwen 2075 | pull_bank | 8704 × 2048 | 110 | 45791 | 38.7 µs | 1.18 |
| Qwen 2177 | pull_bank | 640 × 2048 | 110 | 3884 | 2.8 µs | 1.37 |
| Panoptic 34 † | pull_bank | 8192 × 1088 | 110 | 24213 | 19.3 µs | 1.25 |
| Panoptic 0 † | push_shard → DRAM | 8192 × 1088 | 103 | 30616 | 19.3 µs | 1.58 |
| Panoptic 7 † | push_shard → DRAM | 4096 × 1088 | 80 | 15814 | 9.7 µs | 1.64 |
| Qwen 2072 | push_shard → DRAM | 512 × 2048 | 8 | 3867 | 2.3 µs | 1.70 |
| Qwen 2071 | push_shard → DRAM | 404 × 2048 | 8 | 3136 | 1.8 µs | 1.75 |
| Qwen 2171 | push_shard → DRAM | 512 × 2048 | 8 | 3762 | 2.3 µs | 1.65 |
| Qwen 2067 | pull_shard L1→L1 | 160 × 2048 | 40 | 2952 | — (L1↔L1) | — |
| Qwen 2073 | push_shard L1→L1 | 160 × 2048 | 40 | 3632 | — | — |
| Qwen 2163 | pull_shard L1→L1 | 160 × 2048 | 40 | 2984 | — | — |
| Qwen 2065 | push_shard L1→L1 | 272 × 2048 | 34 (44-core box) | 1556 | — | — |
| Qwen 2064 | pull_shard L1→L1 | 136 × 2048 | 8 | 1339 | — | — |
| Qwen 2074 | pull_shard DRAM→L1 | 160 × 2048 | 40 | 1549 | 0.7 µs | — (launch-bound) |

**PERF FOCUS (2183).** It beats stock ttnn (348.36 µs) by ~12% but misses the lead's ≤ 276.48 µs bound by ~11%. The table points at transaction size: bf16 bounce with 2 KB pages on both sides reaches 1.11× roofline, while bf4b bounce with 576 B pages sits at 1.48–1.55×. Writes into a DRAM width-sharded target are contiguous along each shard row (64 pages = 36 KB on 2183), yet the writer issues them one 576 B page at a time. This is the design's deferred `bounce_dram_shard` row, and it is Refinement 2's lever.

**Lamp measurements** (same profiler, min of 3):

- **L6 `bounce_depth`.** On Qwen 2183, depth 2 → 312.4 µs and depth 16 → 300.4 µs (−4%). On Qwen 2179, 172.3 → 159.5 µs (−7%). The knob is live but cannot reach the 2183 bound alone.
- **L1 `movers_per_owner = 2`.** The effect is regime- and placement-dependent:

  | Case | 1 mover | 2 movers | Change |
  |---|---|---|---|
  | Qwen 2067 (pull_shard L1→L1) | 2.95 µs | 1.75 µs | 1.69× faster |
  | Qwen 2073 (push_shard L1→L1) | 3.63 µs | 2.14 µs | 1.70× faster |
  | Qwen 2078 (pull_bank) | 53.5 µs | 48.9 µs | −9% |
  | Panoptic 0 † (push_shard → DRAM) | 30.6 µs | 27.1 µs | −11% |
  | Panoptic 7 † (push_shard → DRAM) | 15.8 µs | 15.5 µs | noise |
  | Qwen 2072 (8-owner push → DRAM) | 3.87 µs | 6.94 µs | 1.79× slower |
  | Qwen 2071 (8-owner push → DRAM) | 3.14 µs | 5.29 µs | 1.69× slower |

  In the slow cases the second mover writes on NoC0, the `noc_placement` trap. A global flip of the knob is therefore wrong; a per-regime/per-placement choice is right.

## Recommendations

- **Queue.** The queue is Refinement 1 (generality: block formats + DRAM width-sharded target, which lands the PERF FOCUS contract), then Refinements 2 and 3 (perf, both on the flagged bf4b DRAM→DRAM profile). Generality candidates run out after Refinement 1, so every later phase is perf. See `op_requirements.md`.
- **Non-flagged headroom (not queued).** These cases are below the lead's 1.35× goal, but no loose case flags them, and the queue rules restrict perf refinements to flagged shapes while a flag exists. Keep them in the perf guard set and pick them up when the flagged profile is done:
  - **push_shard → DRAM interleaved** sits at 1.58–1.75× roofline (Panoptic 0/7, Qwen 2071/2072/2171). The writes cannot coalesce: consecutive pages sit on different DRAM banks. The measured lever is `movers_per_owner = 2` on the high-owner-count bf8b cases (−11% on Panoptic 0). That is a per-regime knob choice, keyed on owner count / pages per owner, never global, because it regresses the 8-owner bf16 cases 1.7×. `bank_stagger` on the push writers is the other catalog lever.
  - **L1↔L1 shard regimes** (Qwen 2067/2073/2163) are launch/issue-bound at ~3 µs. `movers_per_owner = 2` measures 1.7× on 2067 and 2073. This is a cheap per-regime knob choice (T1).
- **Memory pressure.** No buffer is near the BB-05 bound for any golden shape. The largest is 429696 B, the bounce CB on 2183, and the clamp keeps any future shape bounded.
- **No precision concerns.** The op is a byte move with an exact contract.
