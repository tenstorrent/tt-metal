# Verification Report: g6_copy

Board: Blackhole, 11 × 10 compute grid (110 cores), 8 DRAM banks. Date: 2026-11-27.
Golden results dir: `/tmp/g6_copy_verify_r0` (copy of the CLI report: `verifier_report.json` beside this file).

## Code Review

The op file and kernels needed no code fix. What was fixed (tests and docs):

- **Stale untracked artifacts deleted.** `tests/.../g6_copy/test_g6_copy_perf.py` and its `ablation_kernels/`
  were left behind by the implementer's first session, from before the rewrite. They monkeypatch
  `_PROGRAM_CACHE` and `DYNAMIC_NOC`, which the current descriptor module does not define, so running the
  test directory (Task 2) failed 24 tests. `test_g6_copy_knob_sweep.py` (untracked) duplicated the committed
  `test_g6_copy_lamps.py` sweep. All three were deleted, along with duplicate or broken probes 004–006.
  The ablation kernels were not read (they predate the current implementation).
- **BB-16 had no test. Added `test_g6_copy_trace.py`.** The model replays g6_copy from captured traces, and
  nothing checked that a replay equals the eager call or picks up new source contents written in place.
  The test runs (1, 12, 128, 128) and (16, 12, 128, 128): eager call, capture, then replay into a zeroed
  destination, then replay after a new source is written in place. Both pass.
- **The local `conftest.py` always overwrote the module's device params.** It added a bare
  `use_module_device` marker to every item, so a module could never pass its own device params, such as the
  trace region. It now leaves an existing marker alone.
- **`op_design.md` brought up to date with the code.** The implementer flipped `BANK_STAGGER` to `True` on
  measurement, but the design still said `False` (Bank-run walk, Appendix A). Updated both, with the
  measured numbers, and listed the two verifier tests in Appendix D.
- **Precision baseline added** (`test_g6_copy_precision_baseline.py`, below).

Reviewed and found correct, no change:
- Kernels: `void kernel_main()`, `api/dataflow/dataflow_api.h`, `TensorAccessor` with the default (aligned)
  page size. That size is the buffer's `aligned_page_size()` CT arg (`tensor_accessor_args.cpp:45-51`), so
  `num_contiguous_pages` coalesces and the L1 stride equals the DRAM bank stride. Checked for ROW_MAJOR rows
  too: a 4 B uint32 row has a 64 B page.
- CB sync: the reader reserves and pushes the nominal `block_pages`; the writer waits on and pops the same
  amount, short last block included. Capacity is `cb_depth_blocks · block_pages`, so the ring never wraps
  inside a block.
- BB-03: `noc_async_writes_flushed()` before every pop, and `noc_async_write_barrier()` at writer exit. The
  reader ends on its last read barrier.
- BB-04 / BB-18: kernel images 2092 B (BRISC) and 1812 B (NCRISC) in the zone build (profiler CSV).
- BB-07: the descriptor cache is keyed on stable identity plus every knob; only the two buffer addresses
  are patched per call. `test_g6_copy_rebinds_buffers` covers this.
- BB-17: `split_work_to_cores(row_wise=True)` gives the full grid or full rows plus one partial row.
- Helper usage: there is no compute phase. The design's helper-rejection table holds: no kernel-lib helper
  reads or writes multi-burst bank runs between DRAM and a CB. There is no inter-core communication, so no
  multicast helper applies.

Prompt `## Rules` (all MUST rules; all hold):
- The result lands in `output_tensor`'s buffer, which is returned (identity and address checked).
- No DRAM allocation, no hand-off to another library op, no value-changing path: raw page bytes, no
  unpacker or packer.
- Non-aligned shapes are handled natively (whole-page copy), with no host transforms.
- Core count is the blocking decision: `min(tensor_pages, grid)`.
- One program per call (BB-01). L1 ≤ 1347776 B for every golden shape, and a page that cannot fit is refused
  on the host (BB-05).
- Repeat calls are bit-identical (BB-06, `test_repeat_equality`). The plan is reported (BB-08,
  `describe_plan`). Zones and role dump are in place (BB-21).

No soft-rule advisories apply.

## Registry Conformance

- `INPUT_TAGGERS = {"alignment": tag_alignment}` with the `(inputs, axes)` signature. Tags match
  `feature_spec` (rank 1 → H = 1).
- `SUPPORTED` covers every gated axis: dtype [float32], layout [TILE], and all three alignment values.
- `EXCLUSIONS` match the contract: {fp32, RM}, {int32, TILE}, {uint32, TILE}.
- `validate()` runs the malformed-request checks (ValueError), then SUPPORTED (`UnsupportedAxisValue`), then
  EXCLUSIONS (`ExcludedCell`). The entry point calls it on its first line.
- The op file does **not** declare INVALID.
- No auto-fix to SUPPORTED was needed: no `xpass_drift`.
- **INVALID audit** (`feature_spec.py`): one entry, `{bfloat8_b, ROW_MAJOR}`.
  - Single-tensor coupling: dtype and layout are shared by both tensors.
  - Structural: a block format has no row form.
  - The canonical bf8b + RM entry is present. The op has no weights, so there is no canonicalization cell.
  - Well-formed.

## Design Conformance

Every binding dimension matches `op_design.md`:
- **Algorithm:** a whole-page byte copy.
- **Pipeline:** reader on NCRISC / NoC0, writer on BRISC / NoC1, one CB.
- **Work distribution:** the `page` axis over the full grid, row-major, balanced to ±1 page.
- **Inter-core communication:** none.

Blocking-model fidelity:
- `block_pages`, `cb_depth_blocks` and the core pool are parameters with a single host source (`_plan()`).
  `block_pages` reaches both kernels as one common RT arg, and the CB capacity is derived from it.
- The `one_block` CB is sized to the whole per-core assignment. That is the sanctioned predicate-guarded
  resident fast path (`core_pages_max · page_bytes ≤ L1_BUDGET_BYTES`), with `streamed` as the fallback.
- Expression: each kernel acquires, issues and retires one block per synchronization (one reserve / push,
  one read barrier, one wait / pop, one flush), not one unit at a time.

Performance conformance: the grid is filled (110 cores on the perf shape), both dataflow halves are batched
per bank run, and depth 1 on `one_block` is the design's stated choice. Read/write overlap is a recorded
lamp: Stall-shadow check, and Refinement 2 below.

## L1 Ledger Audit

- **Ledger currency:** the one CB (`cb_page_staging`) matches the code
  (`total_size = cb_depth_blocks · block_pages · page_bytes`). Its size expressions match `_plan()`, and the
  symbol-table bounds match the probe (largest golden footprint 114,688 B, fp32 (16, 12, 128, 128)).
- **Capacity vs live set:** capacity equals the live set.
  - `one_block`: the whole block is live from the last read landing until the flush.
  - `streamed`: two blocks are live at once.
  - The block axis `page` is spanned and scales with `block_pages × depth`. No over- or under-scaling.
- **Page format:** the tensor's own format. This is a dataflow-only payload that never passes through DEST,
  and the bytes must stay bit-identical, so the mechanism reason is stated. No finding.
- **Disjoint lifetimes:** there is a single CB, so no CB pairs exist.
- **Bounds:**
  - `page_bytes` is bounded by the host refusal.
  - `core_pages_max` enters capacity only under the `one_block` predicate.
  - The total is closed-form: `L1_op = cb_depth_blocks · block_pages · page_bytes ≤ L1_BUDGET_BYTES`, scaling
    with `block_pages` and depth only.
- **Data-movement budget:** present and consistent. Each tensor crosses DRAM once, with 0 B cross-core. The
  cheapest-traffic split is the one implemented.
- **Block-size defaults** held: full grid, then the coarsest block (the whole assignment). Refinement 2 is
  expected to depart from that default on measurement: 4-page blocks measured 60.6 µs against 67.6 µs for
  `one_block`. That departure must update the ledger row's Phase 0 values. This is folded into Refinement 2,
  with no standalone entry.

## Precision Baseline

`tests/ttnn/unit_tests/operations/g6_copy/test_g6_copy_precision_baseline.py`, float32 randn, gated on bit
equality.

| Shape | PCC | Max Abs Err | Mean Abs Err | Relative RMS Err | Max ULP | got/true ratio (median, p5, p95) |
|-------|-----|-------------|--------------|------------------|---------|----------------------------------|
| (1, 1) | 1.0 | 0.0 | 0.0 | 0.0 | 0 | 1.0, 1.0, 1.0 |
| (1, 12, 128, 128) | 1.0 | 0.0 | 0.0 | 0.0 | 0 | 1.0, 1.0, 1.0 |
| (8, 12, 128, 128) | 1.0 | 0.0 | 0.0 | 0.0 | 0 | 1.0, 1.0, 1.0 |
| (16, 12, 128, 128) | 1.0 | 0.0 | 0.0 | 0.0 | 0 | 1.0, 1.0, 1.0 |

**Assessment:** bit-identical everywhere. The adversarial bit patterns also pass: NaN payloads, ±inf, ±0,
denormals and extremes (acceptance and golden loose cases, plus `test_float32_bit_patterns`). There is no
precision axis to tune.

**Recommended tolerances:** float32, bfloat16, int32 and uint32 are exact. bfloat8_b uses the one-step group
gate (`helpers.TOLERANCES`). The defaults hold.

## Verifier CLI Summary

Golden run: 13 / 42 passed. That is 10 registry cells plus 3 `numerics` regression tests, with 0 failed and
0 hangs.

- supported_pass: 10
- xfail_expected: 29 (every bf16 / bf8b / int32 / uint32 / ROW_MAJOR production and loose cell; all covered by Refinement 1)
- invalid_skipped: 0 (no production case is bf8b × RM)
- supported_fail: 0
- xpass_drift: 0
- xfail_wrong_mode: 0
- no_axes_found: 3 (the `test_regression.py` numerics tests, which call the op directly and all pass)

**Out-of-SUPPORTED probe** (`probes/probe_007.py`). SUPPORTED and EXCLUSIONS were widened in memory only.
All 29 xfail cells and all 10 loose cases passed bit-exact (bfp8: within one step), with no kernel change.
So did the three contract-EXCLUDED combinations (int32 TILE, uint32 TILE, fp32 RM).
- Every case selects `one_block`.
- Largest footprints: bf16 TILE 36,864 B (1, 1, 32, 62080); RM 5120 B rows.
- uint32 / int32 rows of 4–32 B become 64 B pages.

## Performance Baseline

Profiled with `--profile --no-py-zones` on the committed lamp sweep (`test_g6_copy_lamps.py`). Figures are
device kernel duration, median of 3.

| (16, 12, 128, 128) fp32, perf-flagged | µs | vs roofline 54.61 |
|----------------------------------------|----|--------------------|
| one_block, no stagger | 74.7 (74.2–75.2) | 1.37× |
| **one_block + stagger (current default)** | **67.6 (67.2–68.7)** | **1.24×** |
| block 4 / depth 2 + stagger | 60.6 (60.5–61.8) | 1.11× |
| block 4 / depth 3 + stagger | 61.0 (60.9–61.2) | 1.12× |
| block 2 / depth 2 + stagger | 64.6 | 1.18× |
| block 3 / depth 2 + stagger | 66.3 | 1.21× |
| block 2 / depth 4 + stagger | 65.4 | 1.20× |
| block 1 / depth 4 + stagger | 70.5 | 1.29× |

- The default meets the < 68.56 µs goal on its median, but one of the three runs (68.65 µs) misses it. The
  margin is inside run-to-run spread.
- Per-RISC times on the default: the reader (NCRISC) finishes at 38.5 µs and the writer (BRISC) at 67.2 µs.
  Writes start only after each core's whole assignment has landed, so the reads and writes of a core never
  overlap. With 4-page blocks, both RISCs end around 60 µs.
- Other production fp32 shapes on the default plan: (8, 12, 128, 128) 32.6 µs, (2, 12, 128, 128) 8.5 µs,
  (1, 12, 128, 128) 4.66 µs, (1, 1) 1.09 µs.
- Bank stagger on the small shapes over 6 samples is neutral within noise: (1, 12, 128, 128) +2.9%,
  (2, 12, 128, 128) −1.8%.

## Recommendations

- **Queue shape:** one generality refinement, then perf refinements.
  - The whole TARGET gap is a SUPPORTED widening with no kernel change (probe above), so splitting it into
    more refinements would be artificial.
  - The perf-flagged contract (fp32, TILE, DRAM interleaved, no compute config) is already fully in
    SUPPORTED. The anchor rule therefore costs nothing; Refinement 1 runs first only so the perf guard set can
    span every kernel path × layout.
- **Documented omissions** (TARGET − SUPPORTED values covered by no refinement):
  - Contract-EXCLUDED combinations: int32 × TILE, uint32 × TILE, float32 × ROW_MAJOR. The contract keeps
    them excluded ("no production call uses these combinations … a later contract revision may add them"),
    and no golden cell exercises them.
  - The probe shows they already work, so a contract revision needs only the EXCLUSIONS entries removed and a
    test of each.
- **`test_g6_copy_plan` over-pins BB-08.** It asserts `plan == "one_block"` and
  `block_pages == max(core_pages)` for every production shape. BB-08 requires that the plan be deterministic
  and reported, not that it carry a particular label. The implementer flagged this as friction: the measured
  overlap lever cannot become the default without changing it. Refinement 2 owns relaxing it.
- **Small fp32 production shapes** ((1, 12, 128, 128) at 1–2 pages per core) are dominated by fixed
  per-core cost. The design's "grid synchronization" lamp (`MAX_CORES` so `core_pages ≥ 8`) is unmeasured.
  It is not a queue entry, because every perf refinement must target the flagged shape. Keep these shapes in
  the perf guard set.
- **Repo hygiene:** `ttnn/ttnn/operations/g6_copy/pipeline_logs/` is untracked pipeline state. It includes a
  diff of the pre-rewrite implementation (`preexisting_uncommitted_descriptor.patch`). It is not committed.
