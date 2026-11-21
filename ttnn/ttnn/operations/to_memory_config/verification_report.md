# Verification Report: to_memory_config

Target `quasar` (8×4 clusters, 4 MiB L1), backend ttsim (untimed, slow dispatch). Specification
`eval/specs/to_memory_config/spec.py` (sha256 a1c1bb90…). Reviewed at commit `b68fa885883`.

## Code Review

**Result: no correctness or conformance defect needed a code change.** The op file, the program builder
and both kernels were read against the spec RULES, `op_design.md` and the target page.

| Area | Finding |
|------|---------|
| Spec rules R1–R8 | R1: bytes are moved, never unpacked (no compute kernel, DFB is DM→DM), bit-exact in every cell. R2: output allocated from `TensorSpec(..., memory_config, tile=tensor.tile)`; the golden check reads the placement; the own tests also compare shard shape, grid and orientation. R3: `canonical_to` from the spec reads the config; width / block / DRAM-height configs are refused by name (tested). R4: all three input placements served. R5: `memory.memory_axes(tensor)` derives the input axes; a width-sharded input is refused by name (tested). R6: reshard 2→1 and 1→2 in both layouts tested; the core table does not assume equal grids. R7: always a copy (one dispatch). R8: `validate()` raises `NotImplementedError` subclasses outside SUPPORTED. X1 / X2: `TypeError` from `canonical_to`. |
| One dispatch | exactly one `ttnn.generic_op` per call, no other TTNN op. |
| Kernels | `void kernel_main()`, `api/dataflow/...` includes, `TensorAccessor`, implicit-sync tagged transfers on both ring ends (no reserve/push/wait/pop, so CB push/pop counts trivially balance), sticky tag handed back with `noc_async_{read,write}_set_trid(0, …)` as the last statement (target page, *Ring protocols*). |
| Helpers | none apply (no compute phase; no shared operand, so no mcast). The design's rejection of compute copy helpers is right: they would route bytes through DEST (an R1 hazard for float32). |
| Broadcast | n/a. |

## Registry Conformance

- `INPUT_TAGGERS = AXES` imported from the spec; no op-local axis function. `SUPPORTED` equals `PHASE0`
  (lint: `python3 -m eval.validate_contract to_memory_config --phase0` PASSED, 0 errors, 7 UNRESOLVED
  items that are the spec's open questions, not defects). `EXCLUSIONS = []`. `validate()` checks
  SUPPORTED per axis, then EXCLUSIONS, then two mechanism caps (DRAM-sharded input; a page above one
  64 KiB NoC packet); it is the entry point's first statement. The op file declares no `INVALID`.
- No SUPPORTED auto-fix was needed (`xpass_drift` 0).
- **INVALID audit (spec)**: one entry, `{"memory_config": "DRAM", "sharding": "height"}`. Both axes
  describe the same tensor (the input), and the universe changes (the runner builds no DRAM-sharded
  tensor): well-formed. The canonical bf8b + ROW_MAJOR entry is not needed: `bfloat8_b` is not on any
  dtype axis of TARGET, and Quasar has no bf8b path (target page). Not a norm op: no weight
  canonicalization. No issue to report.

## Design Conformance

| Dimension | Verdict |
|-----------|---------|
| Algorithm / pipeline | as designed: reader → `dfb_pages` → writer per engaged cluster, page `p` → page `p`. |
| Work distribution | R-bal fills the whole grid with `split_work_to_cores(..., row_wise=True)` (32 cores for `[1,1,1600,256]`, 13/12 pages, test-pinned); R-in / R-out use the shard grid in orientation order (COL_MAJOR test-pinned). Both halves issue one transfer per page with no per-page barrier (implicit sync), so neither half dribbles. |
| Thread plan | read off the host: `THREAD_PLAN = (1, 0, 1)` is one named constant, passed to both kernels as compile-time `reader_threads` / `writer_threads`, loops stride from `get_my_thread_id()`. The design states the op as bound by data movement with the reason (one read and one write per page, zero compute), so one hart per side and no NEO is the target page's rule (C = 0 because there is no compute kernel at all). The threads-ran coverage test and the one-session test (ours after ours, `ttnn.to_layout` after ours, ours after `to_layout`) exist and pass; `(2, 0, 2)` is exercised by `test_thread_plan_knob`. No plan comparison was measured: this backend has no timing model (no `model_cycles.jsonl` exists, as expected). |
| Blocking-model fidelity | `block_pages`, depth and the core table are host parameters computed once (`ring_geometry`, `split_work`); the DFB `num_entries` and the kernels' block trip counts derive from the single `block_pages`. No CB scales with an unbounded op dimension (ring ≤ `RING_BYTES_CAP`). Per-core loops are not half-turned: a core's whole assignment is one block when it fits. Expression: the kernels' inner per-page loop has no per-page sync boundary (the DM0 ISR credits per txn-id slice), so it is not an expression failure. |
| **Native sharding (finding)** | In R-in each shard owner reads its **own** input shard back through the input `TensorAccessor` into the staging ring (a NoC loopback), and in R-out it stages pages and copies them again into its own output shard. Values are right and the spec's GUIDANCE favours accessor-only I/O, but the owner side is not zero-copy: one core-local L1 copy per page that the design's deferred R-borrow regime removes. That is architectural rework of the shard-side protocol (borrowed DFB + explicit sync on one side, txn-id cap), beyond this pass: **queued as Refinement 1**, first in the queue, with a structural gate (owner-side staging copies per page 1 → 0). |

## L1 Ledger Audit

- **Currency**: one buffer, `dfb_pages`, matches the code (`entry_size = min(in, out aligned page)`,
  `num_entries = block_pages × depth`). Symbol table bounds and predicates hold; total closed-form
  `block_pages × depth × entry_bytes ≤ min(262,144, unreserved query)`.
- **Capacity vs live set**: no over-sizing (depth 1 only when the whole assignment fits; depth 2
  otherwise). Axis accounting correct: `page` spanned by `block_pages × depth`; RM W inside an entry.
- **Page format vs DEST**: dataflow-only payload, never in DEST: the rule does not apply.
- **Disjoint lifetimes**: single buffer, justified ("DM self-loop forbidden").
- **Traffic budget**: present and consistent with the code (each tensor crosses DRAM at most once each
  way; mesh / core-local bytes per regime). **Cheapest-traffic split**: R-borrow was deferred with a cost
  argument, not a sanctioned reason → folded into Refinement 1. R-bank was deferred as unreachable (no
  L1 bank → core binding); it is reachable (`MemoryView.num_banks`, kernel-side bank discovery from the
  accessor's NoC coordinates) → folded into Refinement 4. The ledger now carries this disposition.
- **Block-size defaults**: interleaved spreads over the full grid, then the coarsest block that fits:
  held. **Finding**: the coarsest single block makes the ring exactly `core_pages_max` entries, and the
  runtime gives a ring with no factor in {2, 3, 4} a single txn id (`dataflow_buffer.cpp:1452`), so
  reads and writes on that core serialize (6 R-bal configurations of INPUTS × layout, e.g.
  `[1,1,1600,256]` TILE, 13 pages per core). Departing from the default needs a measurement, and here
  that is a structural count → folded into Refinement 3.
- Per-core footprint: `block_pages × depth × entry_bytes`; `block_pages` scales with the per-core page
  count up to the ring cap, `entry_bytes` with dtype and RM row width. Largest worked case 262,144 B.

## Precision Baseline

`tests/ttnn/unit_tests/operations/to_memory_config/test_to_memory_config_precision_baseline.py`
(bfloat16, standard normal input, one case per regime):

| Shape | Layout | Route (regime) | PCC | Max Abs Err | Mean Abs Err | Relative RMS Err | Max ULP | got/true ratio median [p5, p95] |
|-------|--------|----------------|-----|-------------|--------------|------------------|---------|-------------------|
| (32, 64) | TILE | DRAM → L1 (R-bal) | 1.0 | 0.0 | 0.0 | 0.0 | 0 | 1.0 [1.0, 1.0] |
| (1, 1, 17, 50) | ROW_MAJOR | L1 → DRAM (R-bal, 112 B vs 128 B slots) | 1.0 | 0.0 | 0.0 | 0.0 | 0 | 1.0 [1.0, 1.0] |
| (1, 1, 400, 512) | TILE | DRAM → L1/height (R-out, 13 cores) | 1.0 | 0.0 | 0.0 | 0.0 | 0 | 1.0 [1.0, 1.0] |
| (1, 1, 1600, 256) | TILE | L1/height → DRAM (R-in, 25 cores) | 1.0 | 0.0 | 0.0 | 0.0 | 0 | 1.0 [1.0, 1.0] |

**Assessment**: bit-exact, as a byte move must be. No scale signature (ratio exactly 1).
float32 was probed in-process with the dtype gate relaxed: 108 / 108 bit-exact (6 shapes × 2 layouts ×
3 × 3 placements); it stays outside SUPPORTED until Refinement 2.
**Recommended tolerances**: exact (the spec's `legacy_default = (0.9999, 0.0)` already demands it);
defaults hold.

## Verifier CLI Summary

Golden run `eval/eval_test_runner.sh` (8 sim workers, forked): 560 passed (558 cells + X1 + X2),
0 failed, 0 errors, 0 hangs, 373 skipped (372 INVALID + the empty loose-case placeholder).
`verifier_report.json`: `/tmp/tmc_golden_v/verifier_report.json`, copied to `pipeline_logs/verifier_report.json`.

- supported_pass: 558
- xfail_expected: 558 (every one float32 → Refinement 2)
- invalid_skipped: 372
- supported_fail: 0
- xpass_drift: 0
- xfail_wrong_mode: 0
- supported_marked_xfail / invalid_unexpected: 0
- queue coverage (`--queue op_requirements.md`): 1 value in TARGET − SUPPORTED, 0 queue gaps

Own tests: acceptance 55 / 55, regime tests 47 / 47, precision baseline 4 / 4.

## Recommendations

- **Queue order** (hardest first, 2:1 perf cadence): R1 native height-shard side → R2 float32 → R3
  (perf) ring slicing → R4 (perf) R-bank → R5 (perf) coalesced shard runs. TARGET has a single
  generality gap (float32), so the queue is short on generality and the trailing slots are perf.
  No `LOOSE_CASES` is perf-flagged, so no anchor contract constrains R1/R2.
- Perf gates are structural counts (untimed simulator). Re-measure R3–R5 and the thread-plan lamp
  (`(2, 0, 2)` vs `(1, 0, 1)`, catalog `split_reader`) on a timed backend; the YOLO detect-head cells move
  1–3 pages per core and are the likely issue-bound case.
- Thread-plan knob caveat: with R or W = 2, a ring whose entry count is odd fails the runtime's
  implicit-sync divisibility (`entry_threshold % num_prods == 0`) and the host rejects the DFB.
  `validate_thread_plan` does not check this; whoever turns the plan must make `num_entries` a multiple
  of the thread count.
- Kernel variants: `block_pages` is a compile-time arg, so each distinct per-core page count builds a
  new reader/writer binary (the float32 probe built 76 kernels for 108 calls; tensor-accessor args
  contribute too). If first-call latency matters, pass `block_pages` as a runtime arg instead: the
  kernels use it only as a loop bound.
- Spec open items (UNRESOLVED: special values, freedoms, perf bar) are the spec author's; NaN / inf
  are moved bit-exactly by construction.
