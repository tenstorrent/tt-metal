# Verification Report: pad

Target: Quasar on the ttsim simulator (`libttsim.so`). Specification: `eval/specs/pad/spec.py` (lint passes with 0
errors and 5 UNRESOLVED open items, none binding here). Date: 2026-11-21.

## Code Review

### Fixed

1. **Correctness bug, now fixed: rows wider than 8 KiB.** It was found by
   running the golden suite with SUPPORTED temporarily set to TARGET.
   - *Cause:* Quasar's NoC overlay packetizes every transfer at `NOC_OVERLAY_MAX_BYTES_IN_PACKET = 8 KiB`
     (`tt_metal/hw/inc/internal/tt-2xx/quasar/noc_cmd_buf_common.h:74`).
   - *Why it is a Phase 0 bug:* it is reachable **inside Phase 0's SUPPORTED**. For example, bf16
     `(1,1,128,8192)` is rank 4, after-padded and in DRAM. No golden cell hit it only because
     the wide INPUTS entries have 1 row per core.
   - *Fix* (the design's documented fallback for this ring, structure unchanged): `pad_program_artifacts.py`
     selects a CT flag `explicit_out_rows = entry_bytes > NOC_PACKET_BYTES`.
     - Above the cap, the writer consumes `dfb_out_rows` with explicit sync: per reader op, `wait_front`, one
       tagged write of `out_row_bytes` per row (trid 0), one barrier, `pop_front`. The ring is then listed in
       the writer's `disable_dfb_implicit_sync_for`.
     - At or below the cap, nothing changes.
     - Thread plans with `R > W` are refused on the explicit path, by a host `ValueError` and a kernel
       `static_assert`: an explicit consumer rotates counters per pop, i.e. per whole reader op.
   - *Regression tests* (`test_pad_plan.py::test_rows_wider_than_a_packet`, 12 cases): both sides of the
     boundary × plans `(1,1)`, `(1,2)`, `(2,2)` × resident / streamed, plus the `R > W` refusal.
2. **Removed `kernels/pad_compute.cpp`.** It was an unbound `kernel_main() {}` placeholder: dead code. The
   "no compute kernel" decision is recorded in op_design.md and in the artifacts docstring.
3. **`max_core_data_rows`** now has `default=0`, so `max()` of an empty split cannot raise. Not reachable today
   (every dimension ≥ 1); this is robustness for `D = 0`.
4. **Tests added:**
   - the precision baseline (below);
   - **R6**: a height-sharded input is tagged `sharding = "height"` by `eval.memory.memory_axes` and refused with
     `NotImplementedError` (`test_sharded_input_is_refused_by_name`). The specification says this test is the
     op's own to carry.
   - **R4**: the output is DRAM and interleaved for a DRAM input (asserted in the precision baseline).

### Deferred
- None architectural. The design's perf lamps go to the queue or to Recommendations below.

### Specification rules

| Rule | Applies now | Status |
|---|---|---|
| R1 byte-exact moves | yes (bf16) | Holds: no compute kernel and no format conversion. The baseline shows 0 ULP. fp32 is exact under the TARGET experiment. |
| R2 added elements = `value` | the kernels are value-general | Value 1.0 is exact through `_run` (plan tests) and in the TARGET experiment. |
| R3 pairs, any sequence form | yes | The tuple and list forms both pass (golden cells). |
| R4 output in the input's buffer, interleaved | yes | `allocate_tensor_on_device(TensorSpec(..., buffer_type))`. Asserted for DRAM in the baseline; L1 is a Refinement 1 test item. |
| R5 DRAM or L1 input | L1 not in SUPPORTED | Passes in the TARGET experiment; Refinement 1. |
| R6 placement axes from `memory_axes` | yes | `_axes()` uses `eval.memory.memory_axes`; new test. |
| R7 `NotImplementedError` outside SUPPORTED | yes | 242 / 242 `xfail_expected`, 0 `xfail_wrong_mode`. |
| X1–X4 | yes | 4 / 4 golden rejection tests, plus `test_rejections`. |

GUIDANCE advisories: every item is followed (no compute; row-major pages; the value's bit pattern in the element
format; L1 read through the accessor).

## Registry Conformance
- **`INPUT_TAGGERS = AXES`**, imported from `eval.specs.pad.spec`. The op file defines no axis function.
- **`SUPPORTED`** equals `PHASE0`; the lint checks this.
- **`EXCLUSIONS = []`**.
- **`validate()`** first parses `padding` (X1–X4: `TypeError` / `ValueError`), then checks SUPPORTED per axis
  (`UnsupportedAxisValue`), then EXCLUSIONS (`ExcludedCell`). `pad()` calls `validate()` as its first statement.
- The op file declares **no `INVALID`**.
- **No auto-fix to SUPPORTED.** There was no XPASS evidence; the TARGET experiment was temporary and reverted,
  because the spec requires Phase 0's SUPPORTED to equal PHASE0. Its results feed Refinement 1.
- **INVALID audit (specification):** `INVALID = []`.
  - The canonical bf8b + ROW_MAJOR entry does not apply. `bfloat8_b` is not in TARGET and has no path on Quasar;
    TILE layout is not in TARGET (see DECISIONS).
  - There are no cross-tensor couplings and no weight axes (the op is not norm-like).
  - No issue.

## Design Conformance
- **Algorithm and dataflow**: as designed.
  - The reader assembles each data row in place: the read lands at `bw·e`, and the margins are filled on the
    first lap only.
  - The writer writes pad rows first from the per-core template (one barrier per core), then data rows.
  - There is one `generic_op` dispatch.
- **Work distribution**: data rows and pad rows are two independent `split_work_to_cores(row_wise=True)` lists over
  the full 8×4 grid. Every cluster is engaged whenever D or P ≥ 32.
- **Thread plan**, read off the host program: `PAD_THREADS = (1, 1)` is one named constant. It is passed as the CT
  args `reader_threads` / `writer_threads`, and `KernelSpec(num_threads=R/W)`. There is no compute kernel. The
  design states the op as bound by data movement, with the reason (zero arithmetic per row); the target page
  prescribes (1,1,1)-class plans for that bound. So the plan is acceptable, not a finding.
  - The threads-ran tests exist: value-1.0 coverage under (1,1), (1,2), (2,1), (2,2).
  - The session test exists: tt-metal's tilize, then pad, then `to_layout` on pad's output, in one session.
  - This backend has no timing model, and `pipeline_logs/model_cycles.jsonl` does not exist. So the design's
    alternatives were not timed.
- **Blocking-model fidelity**:
  - `block_rows`, `ring_blocks` (from `BUFFER_DEPTH`), `pad_rows_per_barrier` and the plan are each computed once,
    in `choose_blocking` / `create_program_artifacts`. They pass to both kernels as RT/CT args, and the ring
    capacity is derived from them (DRY holds).
  - There is no CB scaled by a whole-op dimension. The ring is clamped by `rows_fit`.
  - Expression:
    - The reader does one `reserve_back`, back-to-back reads, one barrier and one `push_back` per block.
    - The writer has no per-unit sync on the implicit path. On the explicit path it does one wait / barrier /
      pop per reader op.
    - Pad rows get one barrier per `pad_rows_per_barrier`.
- **Mechanism caps**: the design's table misses the 8 KiB packet cap on implicit-sync entries (fixed above). It
  should gain a row (recorded in `l1_ledger.md`). The design is the planner's document, and I did not edit it.

## L1 Ledger Audit
- **Currency**: two DFBs, `dfb_out_rows` (`T·ring_blocks·block_rows × entry_bytes`) and `dfb_pad_row`
  (`T × entry_bytes`). These match the code. A verifier note was appended for the consumer protocol now depending
  on `entry_bytes`.
- **Capacity vs live set**: both rings equal their live sets.
  - *Over:* the only gap is the < 16 B per entry of L1 alignment (`entry_bytes − out_row_bytes`), which is
    justified.
  - *Under:* no collapsed extent. `dfb_out_rows` spans data rows (`block_rows · ring_blocks`) and the whole row.
- **Page format vs DEST**: not applicable. There is no compute kernel and no DEST; the payloads are dataflow-only
  raw bytes.
- **Disjoint lifetimes**: none. The two rings overlap for the whole kernel, and that is justified in the ledger.
  `dfb_pad_row` is allocated on cores with no pad rows: one `entry_bytes`, kept for program uniformity, justified.
- **Bounds**: `max_core_data_rows` is unbounded in the shape but enters a capacity only through
  `block_rows ≤ rows_fit`. Every symbol has a bound and a predicate, and the total is closed-form:
  `L1 = T·entry_bytes·(1 + ring_blocks·block_rows) ≤ T·entry_bytes + min(1,048,560, budget − T·entry_bytes)`.
- **Block-size defaults**: held. On interleaved memory the split spreads over the full grid and then takes the
  coarsest block, the whole assignment (`rows_resident`). The worst case on INPUTS is 32 KiB per core (< 1 % of
  budget).
- **Data-movement budget**: present and consistent.
  - The input crosses DRAM once, the output once, 0 B cross-core.
  - On the explicit path a data write is `out_row_bytes` (slightly less than `entry_bytes`).
  - The cheapest-traffic split is the one implemented.

Findings filed: none beyond the in-place note. The overlap lamp touches `dfb_out_rows` sizing and is Refinement 2.

## Precision Baseline

`tests/ttnn/unit_tests/operations/pad/test_pad_precision_baseline.py`: bf16, DRAM, value 0.0, after-padding. It is
compared with the float64 reference on the stored input.

| Shape | PCC | Max Abs Err | Mean Abs Err | Relative RMS Err | Max ULP | got/true ratio (median, p5–p95) |
|-------|-----|-------------|--------------|------------------|---------|-------------------------------|
| (1,3,20,20) + C(0,13) | 1.0 | 0.0 | 0.0 | 0.0 | 0 | 1.0, 1.0–1.0 |
| (4,8,47,256) + all dims | 1.0 | 0.0 | 0.0 | 0.0 | 0 | 1.0, 1.0–1.0 |
| (1,1,2048,256) + H(0,48) | 1.0 | 0.0 | 0.0 | 0.0 | 0 | 1.0, 1.0–1.0 |
| (1,1,32,4096) + W(0,4096) | 1.0 | 0.0 | 0.0 | 0.0 | 0 | 1.0, 1.0–1.0 |

**Assessment**: bit-exact. There is no scale signature: the ratio is identically 1.0.

**Recommended tolerances**: exact (`NUMERIC_RESOLVED`: `exact` for bf16 and fp32).

## Verifier CLI Summary

Final run (after fixes) in `/tmp/pad_golden_final`; the report is also committed as `verifier_report.json` in this
directory.
- supported_pass: 14
- xfail_expected: 242
- invalid_skipped: 0
- supported_fail: 0
- xpass_drift: 0
- xfail_wrong_mode: 0
- (no_axes_found: 5. These are the 4 rejection tests and the empty loose-case parameter, which is not
  registry-driven.)

**TARGET experiment.** The golden suite was run with SUPPORTED temporarily set to TARGET, then reverted.
- Before the fix: 256 / 261 passed, and 4 tests (the fp32 `(2,1,64,4096)` C-pad cells) failed.
- After the fix: **260 / 261 passed** (the 1 skip is the empty loose-case parameter). This is the evidence for
  Refinement 1.

Pad unit tests: 63 pass (acceptance 4, plan / coverage / regime / session / wide-row / R6 / rejections 55,
precision baseline 4).

## Recommendations
- **Queue**: Refinement 1 (generality: all of TARGET − SUPPORTED, measured passing with no kernel change), then
  Refinement 2 (perf: overlap on cores with many rows, gated on blocks per core).
  - Generality is exhausted after Refinement 1, so later phases are perf only.
  - Levers that would need timing to establish headroom are not queued. They are listed below.
- **Thread-plan lamp, (1,0,2) / (2,0,2)**: not queued.
  This backend cannot time it. Worth one measurement on `ttsim-perf` or hardware for the pad-heavy
  YOLOv8 640×640 stem call: 260 pad writes plus 60 data writes per core on one writer hart. The kernels already
  deal by thread id, so it is a one-constant change.
- **Grid-occupancy lamp** (engage fewer clusters on tiny shapes, e.g. the traced 20×20 stem with ~2 data and ~8 pad
  rows per core): not queued. The trade-off is launch cost against parallelism, which only a timed backend can
  settle.
- **Wide rows (> 8 KiB) use the explicit writer**: one barrier per reader op. In `rows_resident` that is one
  barrier per core, so there is no headroom on INPUTS (the wide entries have 1 row per core). If wide-and-tall
  shapes matter later, `row_chunked` entries of ≤ 8 KiB would restore implicit writes. That is the design's
  deferred regime, now with a second motivation.
- **op_design.md** should add the packet-cap row to its Mechanism caps table (see the ledger note).
- **Specification**: no defect found. Five UNRESOLVED items remain (`governs`, `special_values`, `freedoms`, and
  the PERFORMANCE `bar` / `measure`). None blocks this phase.
