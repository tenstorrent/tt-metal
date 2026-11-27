# Verification Report: g6_zeros_like

Board: Blackhole p150, chip 0, worker grid 11 × 10 = 110 cores, 8 DRAM banks. Kernel time in this
report is the profiler's `DEVICE KERNEL DURATION` (zone build, `run_safe_pytest.sh --profile`).

## Code Review

### Fixed

1. **Stall shadow of the zero fill (design conformance, lost overlap) + BB-21 zone gap.**
   `op_design.md` (Traffic ranking, stall-shadow check (a)) puts the core's RT-arg reads and the
   output `TensorAccessor` construction in the shadow of the fill's loopback reads. The kernel ran
   them after `zero_tile()` returned (the helper fuses issue and barrier), so they sat serially on
   every core's critical path before its first write. They also fell outside any profiler zone,
   which BB-21 forbids. Measured gap: 41 ns per core.
   Fix: the fill is issued as the two calls `zero_tile` makes
   (`Noc::async_write_zeros(cb, bytes)` / `write_zeros_l1_barrier()`), with the setup between
   them, all inside the one `g6zl-zero-page` zone. An earlier variant with three separate zones
   lost the gain to marker overhead (~18 ns per marker) in the profiled metric, so the single zone
   was kept.
   A/B over 12 samples, same harness: kernel start → first write **253 → 243 ns** (fp32) and
   **184 → 175 ns** (bf8b). End-to-end kernel time is unchanged within noise: fp32 3587 → 3603 ns
   median, bf8b 822 → 819 ns. This is a conformance fix, not a speedup.
2. **Collapsed buffer-depth knob (DRY).** `ZERO_SCRATCH_PAGES` sized the CB on the host, but the
   kernel always filled exactly one page (`cb.get_tile_size()`). The knob never reached the
   kernel, so the one-page assumption was duplicated without being stated. Turning it would have
   grown L1 with no effect. The deferred `bank_run_coalesced` regime (the first perf refinement's
   lever) has to grow exactly this scratch.
   Fix: host `zero_scratch_bytes = ZERO_SCRATCH_PAGES · page_size` is the single source for both the
   CB `total_size` and a new CT arg the kernel fills from (`static_assert(zero_scratch_bytes >= page_size)`;
   `TensorAccessorArgs` moved to offset 3). Probed under `--dev` with `ZERO_SCRATCH_PAGES = 2`:
   exact zeros on fp32 and bf8b.
3. **Design doc and ledger kept current with the two fixes**: `op_design.md` API Mapping /
   Block Operation Realization / zone table / risk row, and `l1_ledger.md` producer column and
   symbol table.
4. **Tests added**:
   - `test_g6_zeros_like_precision_baseline.py`: production cases only (see Precision Baseline).
   - `test_g6_zeros_like_trace.py`: BB-16 trace capture and replay, fp32 production shape, both
     call forms. Each of two replays rewrites a NaN-refilled buffer to +0.0. Nothing exercised
     trace replay before.
   - `test_profile_core_knobs_bf8` in the profiling driver: placement and core-count lamps on the
     bf8b production case.

### Checked, no change needed

- **One dispatch.** One `generic_op([output], descriptor)` per call. The input is never bound: no
  reader, no input accessor, not in `io_tensors`.
- **Program cache and buffer identity.** The output address is a common RT arg, patched on the
  held `KernelDescriptor` every call.
  - `compute_program_descriptor_hash` (`generic_op_device_operation.cpp:78-140`) hashes only RT-arg
    *counts*, so a new address does not miss the cache.
  - `apply_descriptor_runtime_args` (`program_descriptors.cpp:188-212`) copies per-core and common
    RT args into the cached program on a hit.
  - The Python cache key covers every program-changing value, including `ZERO_SCRATCH_PAGES` and
    the writer NoC/RISC.
- **BB-03.** One `write_zeros_dram_barrier()` is the kernel's last act.
- **BB-17.** One kernel on one rectangle per production shape: fp32 `(0,0)-(10,9)`, bf8b `(0,0)-(3,9)`.
- **Kernel hygiene.** `kernel_main()` form, `api/dataflow/...` includes, `TensorAccessor`, no CB
  credit on the private scratch (no sync to mismatch).
- **Helpers.** `store_zero_block` has no kernel_lib helper (design table of rejected candidates
  re-checked). It uses the dedicated `Noc::async_write_zeros` DRAM overload, one call per page
  (mechanism cap: one page per call). No multicast exists to migrate.
- **Expression.** Every page write of a core is issued back to back from one resident page, with
  one barrier per core and no per-page handshake. The block factor (`num_pages_this_core`) drives
  the issue loop. No half-turned knob.

### Deferred (architectural, filed as refinements)

- The DRAM write stream runs at about half the roofline (see Performance Baseline). The lever is
  the design's deferred `bank_run_coalesced` regime plus placement, filed as Refinement 2.
- The remaining per-core fixed cost (launch skew, fill on the critical path) is filed as
  Refinement 3.

## Prompt Rules (`eval/prompts/g6_zeros_like.txt` `## Rules`)

All rules are MUST/MUST NOT. Every rule whose condition applies is met:

| Rule | Status | Evidence |
|------|--------|----------|
| +0.0 everywhere, not computed from x | ✓ | bit-exact on random 32-bit patterns and NaN/inf/-0.0/subnormal inputs (acceptance, precision baseline, golden `test_bit_patterns`) |
| input data never read; output crosses DRAM once | ✓ | input not bound; one write per page (`describe_plan` coverage test); golden `test_input_never_read` |
| new tensor / `optional_tensor` itself; input unchanged | ✓ | acceptance + golden `test_given_buffer`, `test_new_storage`, `test_two_live_outputs` |
| repeat call writes the buffer it names | ✓ | common RT arg + cache-hit copy (above); acceptance, golden `test_second_buffer`, `test_reused_allocation`, new trace test |
| own program, no delegation | ✓ | own descriptor + kernel, one `generic_op` |
| no extra axis / tagger / EXCLUSIONS | ✓ | `INPUT_TAGGERS = {dtype}`, `EXCLUSIONS = []` |
| core count chosen by blocking, not by shape | ✓ | folded page split: 110 cores for 48 fp32 tile rows, 40 for 5 bf8b tile rows |
| BB-01/02/05/06/07/08/09/17 | ✓ | one program; no conversions; L1 4096 B/core; deterministic; no rebuild on repeat; `describe_plan`; one rectangle |
| BB-16 trace | ✓ | `test_g6_zeros_like_trace.py`: allocation at capture (new-output form), replay rewrites the captured buffer |
| BB-21 zones | ✓ (after fix 1) | all kernel work is in `g6zl-zero-page` / `g6zl-issue-writes` / `g6zl-write-barrier`; only the ~20 ns kernel prologue (object construction before the first zone) is unzoned |
| BB-22 tests limited to production cases | ✓ | every new test uses the production shapes only (precision baseline deliberately not shape-swept) |
| BB-04 / BB-18 kernel-config fit | partially verified | builds and runs in the plain and profiler builds; the NoC-tracing build was not run |

No soft guidance (prefer / consider / avoid) applies.

## Registry Conformance

- `INPUT_TAGGERS = {"dtype": tag_dtype}`, signature `(inputs, axes)`.
- `SUPPORTED = {"dtype": [float32]}`, `EXCLUSIONS = []`.
- `validate()` does the structural `ValueError` checks, then SUPPORTED (`UnsupportedAxisValue`),
  then EXCLUSIONS (`ExcludedCell`). The entry point calls it first. The op file declares no
  `INVALID`.
- No auto-fix to SUPPORTED: no XPASS drift exists. bfloat8_b is refused in `validate()` and works
  when dispatched directly (probe: 0 non-zero of 40960 elements, new and optional buffers, 1088 B
  pages). It is deliberately left to Refinement 1, per the prompt's Phase 0 definition.
- **INVALID audit** (`feature_spec.py`): `INVALID = []` is correct.
  - TARGET has one axis (`dtype`). Both values are TILE-only and layout is not an axis.
  - The canonical bf8b + ROW_MAJOR entry does not apply: ROW_MAJOR is refused structurally by
    `validate()` (`ValueError`), not by a cell.
  - No cross-tensor entries. Not norm-like, so no weight canonicalization.

## L1 Ledger Audit

- **Currency.** One CB (`cb_zero_page`, index 0), one ledger row. The capacity expression
  `zero_scratch_pages · page_size` matches the allocation, and the fill size now matches it too.
- **Capacity vs live set.** 1 page capacity = 1 page live set. It spans no block axis and streams
  over `page` (every write reads the same page). Neither over- nor under-sized.
- **Page format vs DEST.** No DEST in this op. `Float32` / `Bfp8_b` pages are dataflow-only
  payloads (NoC loopback in, NoC write out). The format must equal the output dtype so the CB tile
  size equals the DRAM page size (ASSERTed).
- **Disjoint lifetime.** Not applicable (one CB).
- **Bounds.** `page_size ≤ 4096 B` (TILE, dtype ∈ TARGET), `zero_scratch_pages = 1`. Closed form.
  No op dimension appears in a capacity expression.
- **Data-movement budget.** Present and consistent with the code: input 0 crossings, output 1
  write, cross-core 0, core-local `C · page_size`.
- **Cheapest-traffic split.** Implemented: the built split is the minimum at the DRAM and
  cross-core tiers. `bank_run_coalesced` ties on bytes and is recorded as `deferred` with a
  positive reason (equal bytes, fewer commands).
  - Measurement says its value is not the command count the ledger weighs. The write stream
    reaches ~250 GB/s whatever the core count (Performance Baseline), so the open question is
    DRAM-side write efficiency (contiguous in-bank runs, bank affinity).
  - That is not a ledger defect, since bytes really are equal. It is folded into Refinement 2,
    which builds that regime.
- **Block-size defaults.** Held: interleaved → the split's units spread over the full grid, block
  = the core's whole range.
- **Per-core footprint** = `zero_scratch_bytes = ZERO_SCRATCH_PAGES · page_size` = 4096 B (fp32) /
  1088 B (bf8b), 0.3 % of the 1347776 B model budget. Only `ZERO_SCRATCH_PAGES` and the dtype
  scale it; nothing scales with `P`, `num_pages_this_core` or the core count.

## Precision Baseline

`test_g6_zeros_like_precision_baseline.py`. Production cases only (BB-22 forbids shape sweeps),
so the parametrization is call form × input draw instead of 3–4 shapes.

| Case | PCC | Max Abs Err | Mean Abs Err | Relative RMS Err | ULP (max) | Elements ≠ +0.0 |
|------|-----|-------------|--------------|------------------|-----------|-----------------|
| float32 (1,12,128,128) new_output, randn | 1.0 | 0 | 0 | n/a (ref std = 0); abs RMS 0 | 0 | 0 / 196608 |
| float32 (1,12,128,128) new_output, random 32-bit patterns | 1.0 | 0 | 0 | n/a; abs RMS 0 | 0 | 0 / 196608 |
| float32 (1,12,128,128) new_output, NaN/inf/-0.0/subnormal | 1.0 | 0 | 0 | n/a; abs RMS 0 | 0 | 0 / 196608 |
| float32 (1,12,128,128) into_optional (NaN-filled), all three draws | 1.0 | 0 | 0 | n/a; abs RMS 0 | 0 | 0 / 196608 each |
| bfloat8_b (5,1,32,256) new_output | — | — | — | — | — | refused (not in SUPPORTED); probe via direct dispatch: 0 / 40960 |

- **Got/true ratio.** Undefined: the reference has no non-zero element. No scale-bug signature is
  possible on a constant output.
- **Assessment.** Bit-exact. The op has no arithmetic and its only bytes are copies of the
  firmware zero region.
- **Recommended tolerances.** Exact for both dtypes, as `helpers.TOLERANCES` already says.

## Verifier CLI Summary

Results dir: `/tmp/g6zl_golden_final` (report copied to `verifier_report.json` in this directory).

- supported_pass: 4 (fp32 new_output + into_optional, cartesian and loose)
- xfail_expected: 2 (bf8b cartesian + loose) → Refinement 1
- invalid_skipped: 0
- supported_fail: 0
- xpass_drift: 0
- xfail_wrong_mode: 0
- no_axes_found: 8 (`test_regression.py`, all passed)

Golden runner: 12 / 14 passed, 2 xfail, 0 failed, 0 hangs.

## Performance Baseline

| Case | Median kernel ns (n) | Range | First→last core start |
|------|----------------------|-------|-----------------------|
| fp32 (1,12,128,128), 110 cores (perf focus) | 3603 (12) | 3272–3873 | ~455 ns |
| bf8b (5,1,32,256), 40 cores (direct dispatch) | 819 (12) | 780–845 | ~275 ns |

Goal (`LOOSE_CASES[0]`): ≤ 2.31 µs (1.35 × the 1.71 µs DRAM roofline). Stock ttnn: 4.92–4.94 µs.

fp32 zones, median per core:

| Zone | Time |
|------|------|
| `g6zl-zero-page` | ~190 ns |
| `g6zl-issue-writes` | ~100 ns (2 pages) |
| `g6zl-write-barrier` | 1.65 µs median, 3.1 µs max |

The last barrier ends ~3.6 µs after the first core starts. The barrier is the DRAM write tail:
786432 B over ~3.3 µs of write phase is ≈ 240 GB/s, about half the roofline's 460 GB/s.

Lamps (fp32 median of 3, bf8b median of 5; kernel ns, column-major / row-major):

| Cores | fp32 col | fp32 row | | Cores | bf8b col | bf8b row |
|-------|----------|----------|-|-------|----------|----------|
| 110 | 3642 | 3782 | | 40 | 815 | 954 |
| 96 | 3777 | 3690 | | 20 | 737 | 956 |
| 64 | 3688 | 3048 | | 8 | 931 | 910 |
| 48 | 4081 | 4080 | | | | |
| 32 | 4279 | 3500 | | | | |
| 24 | 5019 | 4238 | | | | |
| 16 | 6017 | 3464 | | | | |
| 8 | 9241 | 3176 | | | | |

Implementer's NoC lamp: writer on NoC0 5.71 µs vs NoC1 3.58 µs.

Readings:

- A row of 8 cores matches the full grid, so occupancy is not the limit; the write stream is.
- Column lines are bad, as `noc_placement` predicts.
- No knob-only setting reaches the goal. The best single point (row, 64 cores, 3.05 µs) is still
  1.3× off it and within the run-to-run spread of its neighbours.

## Recommendations

- **Refinement order.**
  - TARGET − SUPPORTED is only `dtype = bfloat8_b`. It needs no kernel change and is
    Refinement 1.
  - Nothing else in TARGET exists to bundle with it. Inventing generality work to reach the
    3–5 target would be padding.
  - The flagged perf contract (fp32, TILE, DRAM interleaved, new output, no compute config) is
    fully in SUPPORTED at Phase 0, so no anchor unlock is needed. The remaining queue is perf.
- **Perf guard set** for every perf refinement:
  - fp32 (1,12,128,128) into_optional (same path, different buffer).
  - bf8b (5,1,32,256) new_output (1088 B pages, P = 40 < grid).
  - The flagged shape itself.
- **`ROW_WISE` / `NUM_CORES_CAP` are global host constants** shared by both dtypes. A perf
  refinement that changes them must re-measure bf8b. If the best setting differs per P, make it a
  function of `P` inside `_plan()`, keep it in the cache key, and keep `describe_plan` honest
  (BB-08).
- **Host program-hash lamp** (design, Perf lamps): `custom_program_hash` from the cache key would
  skip the per-call descriptor hash walk on untraced calls. It is host time only and invisible to
  device-ns, so it is not a queue entry. Worth doing if untraced host latency matters.
- **Precompile pass noise.** The precision test prints `pcc=0.999999` during the precompile
  collect pass. The real pass reports 1.0. This is a harness artifact, not the op.
