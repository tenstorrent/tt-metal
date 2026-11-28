# Verification Report: g6_unsqueeze

Verifier pass on the Phase 0 build (commit `4dece1cd408` + breadcrumbs). Board: Blackhole, compute grid 11×10.

## Code Review

### Fixed in this pass
1. **BB-07 — host rebuilt the program description on every call.** `g6_unsqueeze()` called
   `select_plan()` + `create_program_descriptor()` per call (split_work_to_cores, TensorAccessorArgs,
   CB/kernel descriptors). Added a host-side descriptor cache
   (`g6_unsqueeze_program_descriptor.get_program_descriptor`): keyed by everything that changes the
   built program (shape, dtype, layout, memory layout + buffer type, compute grid, and every module knob
   `L1_BUDGET_BYTES`/`MAX_CORES`/`MAX_BLOCK_W_TILES`/`HALVES_PER_RISC`), LRU-bounded to 32 entries. A hit
   patches only the two buffer addresses in place (`RT_IN_ADDR`/`RT_OUT_ADDR` slots of each core's runtime
   args). The device-side program cache key (generic_op descriptor hash) was already complete: `logical_w`
   and every knob are CT args. New test `test_descriptor_cache_patches_addresses` checks the patch on a hit
   (two different input buffers, bit-exact, one cache entry). The golden `sequence`, `trace` and `repeat`
   cases pass on the cached path.
2. **DRY — 64 B read alignment was a literal in two places.** The host had `READ_ALIGN_SLACK_BYTES = 64`
   (CB slack) and the kernel had its own `read_align = 64` (landing align-up). Now the host passes it as
   CT arg 8 (`read_align_bytes`, with a power-of-two `static_assert`). The TensorAccessorArgs offset moved
   from 8 to 9.
3. **Dead template parameter.** `FaceMover`'s `cb_face_stage` template parameter was never used inside the
   struct (only `kernel_body` uses the CB id for `get_write_ptr`). Removed.

### Reviewed, no change needed
- **Prompt `## Rules`** (all MUST rules apply — every shape is production):
  - Bit-exact: no compute kernel; values move only through the NoC and 32-bit RISC stores. ✓
  - Padding is never read (only row 0 of faces 0/1). ✓
  - One program per call (`ttnn.generic_op`); no staging tensor; no host-side transformation. ✓
  - Core count comes from the blocking decision (`select_plan`), not from the tensor shape. ✓
  - Malformed requests are refused on the host before dispatch (golden `test_validation_errors`, 10
    cases). ✓
  - BB-03: one read barrier and one write barrier before each kernel returns. ✓
  - BB-05: p1/p2 use 2176 B per core; the cap formula keeps any W within 1347776 B. ✓
  - BB-17: p1 runs on **2 core ranges** (11 + 5), p2 on 1. That is "few" ranges, but not one rectangle;
    filed as the Refinement 1 lever.
  - BB-21: five `DeviceZoneScopedN` zones per kernel, and `describe_plan()` reports roles.
- **Registry conformance:**
  - `INPUT_TAGGERS` are `(inputs, axes)` → dtype, layout. `SUPPORTED` = {float32, TILE} = TARGET.
    `EXCLUSIONS = []`.
  - `validate()` runs as the entry point's first line, in the right order: dim → SUPPORTED → EXCLUSIONS →
    shape/placement.
  - The op file declares no `INVALID`.
- **Design conformance:**
  - Algorithm: face-row move, as designed.
  - Topology: the two DM RISCs each own one tile half (reader-config NCRISC half 0, writer-config BRISC
    half 1).
  - Work split: `split_work_to_cores(row_wise=True)` over `min(grid, tensor_w_tiles)` cores. p1 uses 16
    cores × 1 tile, p2 uses 8 × 1, filling the machine the design called for.
  - Both NoC halves are batched: reads issued back-to-back with one barrier per block, writes
    back-to-back with one barrier per kernel.
- **Blocking-model fidelity:**
  - `block_w_tiles`, `buffer_depth`, `halves_per_risc` and core count are CT args or `select_plan()`
    outputs. Each is defined once (`select_plan`) and every CB size and trip count derives from it.
  - The staging CB scales with `buffer_depth·block_w_tiles·halves_per_risc`, not with W. `streaming` is
    the bounded fallback for long assignments (tested with forced knobs, ragged last block).
  - Schedule expression: one read barrier per block and no per-tile handshakes. The in-place scatter is a
    loop internal to a block operation.
- **Helpers:** the raw NoC calls and RISC stores are justified in `op_design.md` ("Helpers considered and
  rejected"). No kernel_lib helper moves a tile row into a tile column, and no multicast/semaphore exists.
- **Correctness hygiene:**
  - TensorAccessor (not InterleavedAddrGen).
  - `void kernel_main()`.
  - `api/dataflow/dataflow_api.h` include.
  - No CB push/pop (private scratch), so push count = wait count = 0.

### Deferred to refinements
Nothing correctness-related. The perf levers are in `op_requirements.md`.

## Registry Conformance
- INPUT_TAGGERS, SUPPORTED, EXCLUSIONS and validate() are present and correctly wired. The op file has no
  `INVALID`.
- No auto-fixes to SUPPORTED: no xpass_drift and no xfail_wrong_mode.
- **INVALID audit** (`feature_spec.py`): `INVALID = []`. That is correct:
  - TARGET has one dtype (float32) and one layout (TILE), and the pair is a legal tensor.
  - The canonical bf8b+ROW_MAJOR entry is not applicable: neither value is in TARGET.
  - No cross-tensor entries. Not a norm-like op.

## L1 Ledger Audit
- **Currency:** two rows (`cb_face_stage_half0/1`), matching the declared CBs. The size expression
  `buffer_depth·block_w_tiles·halves_per_risc·face_bytes + 64` matches `stage_total_bytes`. The 64 B
  `page_size` (needed so total_size is a page multiple) is recorded in the implementer note.
  - Updated: the total-footprint and cap expressions now carry `halves_per_risc`, as the code's
    `_block_cap()` does.
- **Capacity vs live set:** capacity = live set + 64 B alignment slack in both regimes.
  - Over: none. The face (1024 B) is both the landing slot of the 64 B row and the output payload; the
    scatter is in place.
  - Under: the live set spans `w_tile` (block) and the capacity scales with `block_w_tiles`. ✓
- **Page format vs DEST:** Float32 with no DEST. The CB is a dataflow-only payload whose width must equal
  the 4 B element for bit-exactness. Justified.
- **Disjoint lifetimes:** the two CBs live concurrently on different RISCs, and the ledger justifies it.
- **Bounds:** `block_w_tiles ≤ cap(buffer_depth)`, a closed form in `L1_BUDGET_BYTES`; `core_w_tiles`
  beyond the cap goes to `streaming`.
- **Block-size defaults:**
  - Held: interleaved spreads `w_tile` over min(grid, tiles) cores, and the block is the whole per-core
    assignment.
  - The grid-sync lamp was measured by the implementer on p1 (MAX_CORES 8/4/2/1 → 1439/1901/3018/4839 ns
    vs 1335 ns at 16). The default holds by measurement.
- **Data-movement budget:** present and consistent with the code (p1: read 2048 B, write 32768 B, 0 B
  cross-core).
  - The cheapest-traffic variant (16 B column-segment writes, −24576 B write) is deferred on an
    *expectation*, not a measurement. That is not a valid reason, so it is **folded into Refinement 2**
    (output write shape), which must measure it.
- **Per-core footprint:**
  - `L1_op = (2/halves_per_risc)·(buffer_depth·block_w_tiles·halves_per_risc·1024 + 64)`.
  - Production p1/p2: 2176 B, as the golden L1 capture recorded. Scales with `block_w_tiles` and
    `buffer_depth` only.

## Precision Baseline
`tests/ttnn/unit_tests/operations/g6_unsqueeze/test_g6_unsqueeze_precision_baseline.py`:
- Inputs are randn with these values injected: NaN, ±inf, −0.0, ±denormal min, FLT_MIN-ish, ±3.4e38, and
  a NaN with a non-canonical payload (0x7FC01234).
- The gate is bit equality. The metrics below are computed on finite, non-zero reference elements.

| Shape | Bit-exact | PCC | Max Abs Err | Mean Abs Err | Relative RMS Err | ULP (max) | got/true ratio p5 / p50 / p95 |
|-------|-----------|-----|-------------|--------------|------------------|-----------|-------------------------------|
| (1, 32) | yes | 1.0 | 0 | 0 | 0 | 0 | 1 / 1 / 1 |
| (1, 256) | yes | 1.0 | 0 | 0 | 0 | 0 | 1 / 1 / 1 |
| (1, 512) | yes | 1.0 | 0 | 0 | 0 | 0 | 1 / 1 / 1 |
| (1, 1, 4100) (ragged, 129 tiles) | yes | 1.0 | 0 | 0 | 0 | 0 | 1 / 1 / 1 |

**Assessment:** bit-identical on every shape. Special values and NaN payloads survive, as expected from a
path with no FPU/SFPU.

**Recommended tolerances:** exact (the golden `gate = "bit_exact"`). No tolerance relaxation is needed.

## Verifier CLI Summary
Two runs of `eval/eval_test_runner.sh --no-precompile eval/golden_tests/g6_unsqueeze/`. Precompile is off
because the implementer's breadcrumb shows the traced helper hangs the precompile collect pass.

**Run A — `EVAL_CAPTURE_L1=0`** (`verifier_artifacts/verifier_report.json`): **20/20 passed.**
- supported_pass: 9
- xfail_expected: 0
- invalid_skipped: 0
- supported_fail: 0
- xpass_drift: 0
- xfail_wrong_mode: 0
- no_axes_found: 11 (the regression tests: `test_repeat_bit_identical` plus the 10
  `test_validation_errors`; they carry no registry cell and all pass)

**Run B — default runner (L1 capture on)** (`verifier_artifacts/verifier_report_l1capture.json`): 10/20
passed. Every failure is a harness interaction, not an op defect:
- **9 × `test_validation_errors`** → `L1ProfilingError: peak_total=0`. A correct host-side refusal
  allocates nothing, and `eval/l1_profiling.capture_for_item` turns a zero peak into a forced failure. The
  `l1_interleaved` case passes only because its input tensor itself is in L1.
- **1 × `test_op_loose[...-trace]`** (`supported_fail`) → `Cannot load new binaries during trace capture`.
  `l1_profiling.start_for_item` calls `device.disable_and_clear_program_cache()` for the whole test body,
  so the helper's eager warm-up is never cached and capture must build. It passes in Run A, where the
  captured call adds no cache entry and the replay equals eager bit for bit.
- **Neither can be fixed from the op.** Both need a change in the harness (`eval/l1_profiling.py` /
  golden helpers): skip capture for tests that legitimately dispatch nothing, and keep the cache enabled
  (or capture only the graded call) for traced cases. `feature_spec.py` and the harness are read-only to
  this stage, so this is reported for the user / golden-suite owner.

**Device kernel time (Run A, `device_kernel_ns`):**
- p1 `[1,512]`: 1347–1458 ns on 16 cores.
- p2 `[1,256]`: 1353–1396 ns on 8 cores.
- Stock ttnn (lead's figure): 3320 ns. Phase 0 already meets "faster than the incumbent" on both cases.
- The lead's roofline (0.01 µs × 1.35) is far below the fixed cost of one program launch. It is not
  reachable by any kernel change, so report it as the bound and do not use it as a gate.

## TARGET − SUPPORTED
Empty. TARGET = {dtype: [float32], layout: [TILE]} = SUPPORTED, `INVALID = []`, and no `xfail_expected`
cells exist. No generality refinement is possible, so the queue is perf only (§ Perf refinements: once
generality is exhausted, the remaining phases are all perf).

## Recommendations
- **Perf headroom is in fixed cost, not bytes.**
  - p2 (8 tiles) and p1 (16 tiles) take the same ~1.4 µs, so launch, core-range, kernel-image and
    barrier-latency costs dominate.
  - The refinement queue targets those (placement / core-range count / NoC selection, then output
    transfer shape, then per-kernel fixed cost).
- **Harness:** see Run B above. Until it is fixed, verify this op with `EVAL_CAPTURE_L1=0`, or read the
  Run B failures as known harness noise. Also see the implementer's breadcrumb: precompile plus the trace
  helper hangs at teardown.
- **Advisory:** the prompt has no soft (prefer/consider/avoid) rules.
