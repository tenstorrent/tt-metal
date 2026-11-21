# Verification Report: clone

Target: quasar (8×4 = 32 clusters, 4 MiB L1 each) on the ttsim simulator (untimed). Specification: `eval/specs/clone/spec.py` (sha256 `561da024baa5…`). Reviewed at commit `7b961a50da3` plus the fixes below.

## Code Review

**Fixed:**
- **The page-size refusal now has a typed exception** (`clone.py`, `validate()`). The mechanism cap (`entry_bytes > NOC_MAX_BURST_BYTES`, the unbuilt `oversized_page` regime) raised a bare `NotImplementedError`. It now raises `ttnn.operations._op_contract.SupportRefusal`, which is still a `NotImplementedError`, so R8 holds. The harness recognises a deliberate refusal by this type, as it does for the SUPPORTED and EXCLUSIONS refusals.
- **New test `test_oversized_page_refused`** (`test_clone_plan.py`): a `(2, 32800)` bf16 ROW_MAJOR input (65,600-B rows) is refused with `SupportRefusal` before any device work. The cap had no test before.
- **L1 ledger currency** (`l1_ledger.md`): fixed two lines, details under *L1 Ledger Audit*.

**Reviewed, nothing to fix:**
- *Kernels* (`clone_reader.cpp`, `clone_writer.cpp`):
  - `void kernel_main()`, `api/dataflow/...` includes, and `TensorAccessor` from the tensor binding token.
  - Implicit-sync tagged transfers with no reserve, push, wait, pop or barrier (the target page's protocol for a ring a NoC transfer fills or drains).
  - The trid is handed back as each kernel's last statement.
  - The thread stride comes from a compile-time arg, the page range from runtime args.
  - The ring is a DM→DM STRIDED ring: one producer kernel, one consumer kernel.
- *Ring sync balance.* Producer thread t and consumer thread t both handle `ceil((core_pages − t)/R)` entries (R = W, enforced by `validate_thread_plan`).
- *Helpers.* No kernel-lib helper covers an interleaved page copy. The design's "Helpers considered and rejected" table holds: `local_copy_helpers_dataflow` is self-aimed L1→L1, mcast has no reuse-shared operand, and the compute helpers would need a compute stage the design rejects for R1. There is no multicast or semaphore code to migrate.
- *Broadcast efficiency.* Not applicable: no binary or unary helper, no constant CB.
- *One native dispatch.* `clone()` calls `validate()`, `allocate_tensor_on_device`, `create_program_artifacts`, then exactly one `ttnn.generic_op`. There are no composite ttnn ops (no `to_layout`, no `to_memory_config`).

## Specification rules

| Rule | Applies in Phase 0 | Status |
|------|-------------------|--------|
| R1 copy every element unchanged | yes (bf16). float32 is not yet in SUPPORTED. | ✓ Raw NoC page copy, no DEST. The precision baseline is bit-exact. A verifier probe copied float32 bit for bit through the same program (48/48 cells, see Refinement 1). |
| R2 accept `dtype` None / same, refuse others | yes | ✓ `canonical_dtype` gives a value no cell has, so `UnsupportedAxisValue`. Golden `test_rejection[X2]` passes. |
| R3 output placement (review) | yes (L1) | ✓ `_output_memory_config()`. `test_placement_and_fresh_buffer` reads `output.memory_config()` = L1 INTERLEAVED for DRAM and L1 inputs. The None/DRAM branches are written but refused until Refinement 1, which must extend this test. |
| R4 `memory_config` as a MemoryConfig object, read with `canonical_memory` | yes | ✓ `_axes()` canonicalises first. X1 (the word "L1") raises `TypeError`. Golden `test_rejection[X1]` passes. |
| R5 a new buffer, never the input (review) | yes | ✓ The output is always allocated and the program always dispatched. The test asserts `output.buffer_address() != input.buffer_address()` for an L1 input with an L1 output, and that the input is untouched. |
| R6 input DRAM or L1 interleaved | yes | ✓ golden cells |
| R7 `memory_axes` gating; sharded refused by name (review) | yes | ✓ `memory.memory_axes(input)` and `canonical_memory`. `test_sharded_refused_by_name` builds a HEIGHT-sharded config and a sharded input on the simulator, and both are refused by axis name. It passes and does not skip. |
| R8 Python-side refusal with NotImplementedError | yes | ✓ every refusal is a `SupportRefusal` subclass (including the fixed page-size cap) |

Structural rules: none in this specification (no rule has a `structure` part). GUIDANCE is followed: no compute, the input read through its accessor, the output allocated from a spec that carries the requested config, and the input never returned.

`python3 -m eval.validate_contract clone --phase0 --text`: **PASSED**, 0 errors. Its 6 unresolved items (`governs`, `special_values`, `freedoms`, `PERFORMANCE`, and the numeric-list note) are the specification's open items, not defects.

## Registry Conformance

- `INPUT_TAGGERS = AXES`, imported from `eval.specs.clone.spec`. The op file defines no tagger of its own.
- `SUPPORTED` equals `PHASE0` axis for axis, covering all 8 axes, including `dtype_argument` / `output_memory` (read with the specification's `canonical_dtype` / `canonical_memory`) and `memory_config` / `sharding` (from `eval.memory.memory_axes`).
- `EXCLUSIONS = []`. Nothing inside SUPPORTED fails.
- `validate()` checks SUPPORTED per axis (`UnsupportedAxisValue`), then EXCLUSIONS (`ExcludedCell`), then the page-size mechanism cap (`SupportRefusal`). It is the entry point's first line.
- The op file declares **no** `INVALID`.
- No XPASS evidence, so no auto-fix to SUPPORTED was needed.
- **INVALID audit (specification).** `INVALID = []`, which is consistent: nothing in the TARGET rectangle is structurally impossible, and the design found nothing either.
  - The bf8b+ROW_MAJOR canonical entry is not applicable: `bfloat8_b` is not in TARGET, and Quasar has no path for it anyway (target page, *Family facts*).
  - No weight axes, so no no-weight canonicalisation cells apply.
  - No `INVALID_FOR_ARCH`.

## Design Conformance

- **Algorithm and pipeline.** As designed: regime `interleaved_page_stream`, reader → STRIDED DM→DM ring `staged_pages` → writer. No compute kernel, because `compute_passthrough` is rejected: two extra L1 crossings, RM sticks are not tiles, and fp32 through DEST breaks R1.
- **Work distribution (performance conformance).** Pages are split in contiguous ranges over the **full** queried grid with `split_work_to_cores(..., row_wise=True)`, which avoids the catalog's column-line trap. `min(32, tensor_pages)` clusters are engaged: 32 for every traced YOLOv8 call. Both halves of the dataflow are batched equally: one tagged transfer per page, no per-page barrier on either side, and up to `ring_entries_per_thread · R` transfers in flight.
- **Thread plan.** Read off the host program: `THREAD_PLAN = (1, 1)` is one named constant. `KernelSpec(num_threads=R/W)` and the compile-time args `reader_threads` / `writer_threads` both come from it. There is no compute kernel, so C = 0.
  - The design states the op is **bound by data movement** ("zero compute per page against `entry_bytes` read + written per page"), which is the target page's case for (1, 1, 1). The plan is acceptable.
  - The ring is per thread pair (`num_entries = depth · R`), with the implicit-sync protocol on both ends as the design's buffer table gives.
  - The every-thread-ran test passes at R = W ∈ {1, 2, 3}: a sentinel-prefilled output, 7 pages on one cluster, and full-grid ragged shapes.
  - The device-session tests (a tt-metal untilize before clone; `to_layout` of clone's output after it; then clone again) are present and pass.
  - **Measuring the plan against its alternatives is not possible here.** This run is on the untimed ttsim backend, and no reference figure is on record (`pipeline_logs/model_cycles.jsonl` does not exist). The (2,2) and (3,3) lamps on small RM sticks wait for `ttsim-perf` or hardware.
- **Blocking-model fidelity.**
  - Every knob is single-sourced on the host: `THREAD_PLAN`, `RING_ENTRIES_PER_THREAD` (clamped by `ring_entries_per_thread_for`), the grid (queried; overridable only through the factory's keyword), and `entry_bytes` (derived once in `entry_bytes_of`, used by both `validate()` and the factory). `core_pages` / `start_page` come from the split as runtime args.
  - No CB scales with an op dimension.
  - The per-core block is the whole contiguous range (one block per core).
  - The one-page-per-transfer granularity is not a half-turned knob: it is the design's mechanism cap (an implicit-sync STRIDED entry is exactly one NoC transaction, and consecutive interleaved pages sit in different banks).
  - Expression: no per-unit completion boundary in either kernel (no barrier, push or pop per page).
  - Axis accounting matches the ledger: the ring spans `in_page` and streams over `page`.
- **Inter-core communication.** None by design. No dependent axis exists and no operand is reuse-shared.

## L1 Ledger Audit

- **Ledger currency.** One DFB, `staged_pages`: entry `entry_bytes = align_up(buffer_page_size, l1_alignment)`, `num_entries = ring_entries_per_thread · R`. This matches `DataflowBufferSpec(entry_size=entry_bytes, num_entries=depth * R)`. **Fixed:** the symbol table bounded the depth by `min(8, …)`, a cap the code does not apply. It now reads `min(RING_ENTRIES_PER_THREAD, floor(1,048,560 / (entry_bytes · R)))`.
- **Capacity vs live set.** Capacity is 4 entries; the correctness minimum is 1. The surplus is justified as pipelining: ≥ 2 implicit-sync trid slices, so the ISR credits one while the next is in flight. Nothing is undersized: the ring spans `in_page` with a whole page per entry and only streams over `page`.
- **Page format vs DEST.** Not applicable. The payload is dataflow-only (never unpacked or packed), and its format is the tensor's own, as R1 requires.
- **Disjoint lifetimes.** None: there is only one buffer.
- **Bounds and closed form.** `footprint = ring_entries_per_thread · R · entry_bytes`. That is 8,192 B for any bf16 tile and ≤ 65,536 B for bf16 RM on INPUTS; the worst legal case is clamped to the 1,048,560-B ring cap. Every symbol has a bound and a predicate. Nothing scales with `core_pages` or `tensor_pages`.
- **Block-size defaults.** Interleaved: the split spreads pages over the full grid, and the block is the coarsest possible (the whole per-core range). Held.
- **Data-movement budget.** Present and consistent with the code: each tensor crosses its tier once, and an L1 endpoint costs one cross-core NoC crossing.
  - The cheapest-traffic split (`owner_local`) is a `deferred` row with a delta. **Finding:** one half of its positive reason, "no ttnn Python binding exposes the bank → core map", does not hold as written. `ttnn.reports.get_buffer_pages(device)` returns `core_x`, `core_y` and `bank_id` per page (`ttnn/cpp/ttnn-nanobind/reports.cpp:38-52`), though it has not been run on Quasar. A kernel can also identify local pages from `TensorAccessor` NoC addresses.
  - Disposition: a verifier note was added to the ledger, and the split is queued as perf **Refinement 2**, gated on the cross-core-bytes count. It is not a standalone ledger refinement.
  - The other half of the reason (DRAM→DRAM needs the built regime anyway) stands.

## Precision Baseline

`tests/ttnn/unit_tests/operations/clone/test_clone_precision_baseline.py`: bf16, standard-normal inputs (`NUMERIC["inputs"]`), `memory_config=ttnn.L1_MEMORY_CONFIG`. PCC is from `models.common.metrics.compute_pcc`; abs errors from `comp_allclose` and torch; relative RMS and ULP p99 from `eval.metrics.compute_metrics_torch`. (`models.common.utility_functions` does not exist in this tree; `models.common.metrics` holds the same functions.)

| Shape | Layout / input buffer | PCC | Max Abs Err | Mean Abs Err | Relative RMS Err | ULP (p99 / max) | got/true ratio (median, p5–p95, std) |
|-------|----------------------|-----|-------------|--------------|------------------|-----------------|---------------------------------------|
| (1,1,400,64) | RM / L1 (traced) | 1.0 | 0.0 | 0.0 | 0.0 | 0 / 0 | 1.0, 1.0–1.0, 0.0 |
| (2,1,100,47) | TILE / DRAM (both non-aligned) | 1.0 | 0.0 | 0.0 | 0.0 | 0 / 0 | 1.0, 1.0–1.0, 0.0 |
| (1,1,1600,512) | TILE / DRAM (largest traced) | 1.0 | 0.0 | 0.0 | 0.0 | 0 / 0 | 1.0, 1.0–1.0, 0.0 |
| (1,32,4096) | RM / DRAM (8-KiB sticks) | 1.0 | 0.0 | 0.0 | 0.0 | 0 / 0 | 1.0, 1.0–1.0, 0.0 |

Every output is **bit-equal** to its input (`int16` views compared).

**Assessment**: an exact copy, as R1 and the `movement` class require. There is no scale or structural signature, since the ratio is exactly 1 everywhere.

**Recommended tolerances**: exact. The specification's `legacy_default` (PCC ≥ 0.9999, relative RMS ≤ 0.0) holds, and only an exact copy can meet relative RMS 0. The baseline test additionally gates bitwise equality.

## Verifier CLI Summary

Golden suite (`eval/eval_test_runner.sh`, 8 simulator workers): 1203 tests, 202 passed, 0 failed, 0 errors, 0 hangs, 1 skipped (the empty `test_op_loose[NOTSET]`: `LOOSE_CASES = []`). The final run was after the fixes; the report is also copied to `ttnn/ttnn/operations/clone/verifier_report.json`.

- supported_pass: 200
- xfail_expected: 1000 (float32, and `output_memory` ∈ {None, DRAM}; all queued in Refinement 1)
- invalid_skipped: 0
- supported_fail: 0
- xpass_drift: 0
- xfail_wrong_mode: 0
- no_axes_found: 3 (`test_rejection[X1]` and `[X2]`, both passed; the empty loose placeholder)
- queue_gaps: 0 (`--queue op_requirements.md`: all 3 TARGET−SUPPORTED values are named)

Unit tests: `scripts/run_safe_pytest.sh tests/ttnn/unit_tests/operations/clone/` gives 53 passed: acceptance 25, plan / session / refusal 24, precision baseline 4.

## Recommendations

- **Queue.** Refinement 1 is generality: float32 plus `output_memory` None/DRAM, which closes TARGET. It is a knob-turn, verified 48/48 bit-exact by probe with no kernel change. Refinements 2 and 3 are perf: `owner_local` (T3 scheme-change), then bank-run coalescing (T2, depends on 2).
  - Generality is exhausted after Refinement 1, so the 2:1 cadence continues with perf slots only.
  - Each perf entry is gated on a structural count, because ttsim has no clock.
- **Perf lamps not filed (no structural count would show a gain on this backend):**
  - Small-tensor occupancy: `[32,17]` RM and `[1,1,32,8192]` RM engage 32 clusters with 1 page each, so per-cluster launch and DFB setup dominate. A `min_pages_per_core` knob trades clusters for pages, but fewer clusters is not a structural improvement in bytes or transactions; only a timed backend can rank it.
  - Thread plan (2,2)/(3,3) on small sticks, ring depth 2 vs 8, NoC assignment: the same reason. Run these lamps when a `ttsim-perf` or hardware run is available. Each is one host constant.
- **L1 pressure.** None. The ring is ≤ 64 KiB on every INPUTS shape, against 4 MiB per cluster. The callers' largest L1 output (1.6 MB over 32 banks, about 51 KB per bank) coexists with it; the session test runs that shape.
- **For Refinement 1's implementer.** `test_refusals` asserts today that `memory_config=None` and `memory_config=DRAM` are refused. Those asserts must become placement checks (R3) and fresh-buffer checks (R5) when SUPPORTED grows. Otherwise the plan tests will fail and look like a regression.
- **Specification.** No defect found. Its open items (`special_values`: NaN/Inf moved unchanged; `PERFORMANCE`) are unresolved, not contradictory. A raw page copy moves NaN and Inf bit for bit, so it would meet the likely resolution.
