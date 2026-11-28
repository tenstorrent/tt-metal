# Verification Report: g6_sharded_to_interleaved

Board: Blackhole p150b, chip-scoped slot, worker grid 11 x 10 (`TT_METAL_CORE_GRID_OVERRIDE_TODEPRECATE=10,9`),
110 L1 banks, 8 DRAM banks. `ttnn.open_device` returns a 1 x 1 `MeshDevice`, so every run below goes
through the mesh dispatch path.

## Code Review

State reviewed: the implementer's Phase 0 plus uncommitted edits left by an interrupted earlier verifier
attempt of this run (op file and descriptor, +31/-14 lines; six superseded debug/perf test files deleted;
precision-baseline and perf harness added). I reviewed and re-ran everything in its final form.

**Fixes applied in this pass: none were needed.** Every item below was checked against the code and the
design, and none is a defect:

| Area | Finding |
|------|---------|
| Kernel shape | `void kernel_main()`, `api/dataflow/dataflow_api.h`, `TensorAccessor` (no `InterleavedAddrGen`). One non-posted single-packet `noc_async_write<page_size>` per tile, then one `noc_async_write_barrier` per kernel instance, including idle lanes (BB-03). |
| Zero-copy shard | `cb_input_shard` comes from `ttnn.cb_descriptor_from_sharded_tensor` on the input. The writer reads it in place via `get_read_ptr` and never reads its own shard through an accessor. The sharded placement is implemented natively. |
| Helper usage | No `kernel_lib` helper fits. Eltwise copy goes through DEST and re-derives bfp8 exponents, and the local-copy helpers target local L1. The kernel header and the design's API Mapping both record this. No multicast exists. |
| CB sync | No `wait_front`/`pop_front`/`push_back` anywhere. The one CB is a read-only resident shard, so there is nothing to balance. |
| Registry conformance | See below: correct. |
| Blocking-model fidelity | `Knobs(num_write_lanes, block_tile_rows, block_tile_cols)` is the single source. Lane rectangles, block extents, kernel instances and RT args all derive from it. No literal block value appears in the kernel. The default is one block equal to the whole clipped shard, which is the sharded default. The per-core loop walks one tile per write, and the interleaved bank map forces that (a multi-page write would put the next pages in the wrong bank). This is not a half-turned knob. |
| Expression | `store_block` issues every write of the block with no completion boundary inside it, and `drain_writes` is the single barrier. No reader or compute kernel exists. |
| Cache key (BB-07) | The key covers the device, padded shape, dtype, input memory layout, shard shape, ordered ranges, orientation, output buffer type and knobs. `id(input_tensor.device())` is stable (probe_015: `tensor.device()` is the same Python object as the handle). Per call, the code patches the output address (common RT 0) and rebinds `cb_input_shard`, and the two-live-inputs alternation test covers this. |
| Validation order | Structural checks raise `ValueError` before the support gate. The L1 bound (BB-05) raises `RuntimeError` on a cache miss, before the output is allocated. |
| Plan report (BB-08/21) | `describe_plan` returns the regime, lanes, and each core's role and block. |
| Core ranges (BB-17) | The kernel group is the input's own shard grid: 1 range for p2, 2 ranges for p1. |

**Design defect (perf, not correctness), filed as Refinement 2:** the design's `bank_stagger` row says the
issue order is "already spread" and checks only p2's DRAM banks (`5k mod 8`). On p1 the destination is the
110 L1 banks, core k writes pages `10k .. 10k+9`, and at write step j all 103 cores target banks
`(10k + j) mod 110`. That is only **11 distinct banks per step, about 9.4 concurrent writers each**. The
measured write time (about 1.2 us of 2.11 us, above an 807 ns all-writes-stubbed floor) is consistent with
ten serialized waves on 11 banks. The implementer measured an earlier stagger slower, but it took a modulo
on every row, so the lever still has to be measured in a modulo-free form.

**Advisories (no MUST violated):**
- BB-21: RT-arg loads and accessor construction run outside the two named zones (`S2I-STORE-BLOCK`,
  `S2I-WRITE-BARRIER`). They are inside the firmware's per-RISC kernel zone and cost under 100 cycles, so
  no extra zone was added.
- BB-16 / trace: the signature has no output parameter, so a call made during capture allocates its output
  at capture time. Replay allocates nothing and builds nothing (`test_trace_replay_matches_eager`). The
  contract line "the caller or the unit allocates it before capture" cannot be met with the fixed
  signature. This is a contract tension, not an op defect.
- BB-04: kernel config fit was exercised in the plain and the profiler-zone builds. The NoC-tracing build
  was not run in this pass.
- Mesh (rule 5): the four-chip p2 deployment cannot be tested on this one-chip slot. The 1 x 1 mesh path
  (mesh tensor allocation and `generic_op` SPMD dispatch) runs on every test. A multi-chip check needs
  another board, so it is recorded here and not queued.
- `NOC_MAX_BURST_SIZE = 16384` is the Blackhole value. It is only a host guard, because
  `noc_async_write<page_size>` picks its path from the template argument. Both TARGET page sizes (2048 B
  and 1088 B) are below the Wormhole value of 8192 as well.

## Registry Conformance

- `INPUT_TAGGERS` holds `tag_dtype`, `tag_layout`, `tag_in_scheme` and `tag_buffer`, all with the
  `(inputs, axes)` signature, applied in that order on the scenario built from the tensors.
- `SUPPORTED` declares all four axes: `dtype=[bfloat16]`, `layout=[TILE]`, `in_scheme=["width_sharded"]`,
  `buffer=["dram"]`. This is exactly the contract's Phase 0.
- `EXCLUSIONS = []`. The contract forbids refusing a production case.
- `validate()` checks structure (`ValueError`), then SUPPORTED per axis (`UnsupportedAxisValue`), then
  EXCLUSIONS (`ExcludedCell`). It is the first line of `g6_sharded_to_interleaved`.
- The op file does not declare `INVALID`.
- No XPASS drift, so no SUPPORTED auto-fix. The bf8b, height-sharded and L1-output paths already pass
  bit-exact through `_dispatch` (impl_debug, precision baseline). Claiming them is Refinement 1, because
  the contract fixes Phase 0's SUPPORTED.
- **INVALID audit** (`feature_spec.py`): `INVALID = []` is correct. TARGET holds only `TILE_LAYOUT`, and
  both dtypes are storable in tiles. No cross-tensor coupling entries exist. The canonical bf8b+ROW_MAJOR
  entry would match no cell, because ROW_MAJOR is outside TARGET. The no-weight canonicalization does not
  apply (this is not a norm-like op).

## L1 Ledger Audit

- **Currency:** one CB (`cb_input_shard`, index 0). Its row matches the code: page size =
  `buffer_aligned_page_size()`, pages = `shard_tile_rows * shard_tile_cols`, globally allocated on the
  input's shard buffer. The output is written through `TensorAccessor` and never staged. No stale or
  missing rows.
- **Capacity vs live set:** equal. The CB is the caller's resident shard and costs 0 B of new L1. It spans
  `tile_row` and `tile_col` and is sized on both, so no axis is collapsed or inflated.
- **Page format vs DEST:** `Float16_b` / `Bfp8_b`, a dataflow-only payload that never enters DEST (there
  is no compute kernel). Not a finding.
- **Disjoint lifetime:** only one buffer exists, so this audit is not applicable.
- **Bounds and closed form:** every symbol is bounded in the symbol table. `op_allocated_L1 = 0`, and
  `model_budget_used = [buffer == l1] * ceil(T / n_l1_banks) * page_size`. For p1 that is 10,880 B per
  bank at 110 banks, inside 1,347,776 B. A host `RuntimeError` covers any shape above the bound.
- **Block-size defaults:** held. Sharded: one block equal to the whole clipped shard. No split away from
  it.
- **Data-movement budget:** present and consistent with the code. The input makes 0 DRAM crossings and is
  read in place. The output is written once: p2 sends 327,680 B to DRAM and p1 sends 1,114,112 B core to
  core. The cheapest-traffic split (owner writes, one crossing per tile) is the one implemented.
  `destination_pull` ties only for L1 outputs and is a `rejected` row with a positive reason (it covers
  only half the buffer types and adds a second code path).
- **Filed refinements:** none. No ledger finding.

## Precision Baseline

`tests/ttnn/unit_tests/operations/g6_sharded_to_interleaved/test_g6_sharded_to_interleaved_precision_baseline.py`
covers the two production shapes, each on its production spec and on the contract's other spec. The p1
cells are outside SUPPORTED and run through `_dispatch`, which skips only the support gate. Input is randn
scaled per tile by `2^(t mod 16 - 8)`.

| Case | Shape / dtype | PCC | Max Abs Err | Mean Abs Err | Relative RMS Err | Max ULP | Bit mismatches | got/true ratio p5 / p50 / p95 |
|------|---------------|-----|-------------|--------------|------------------|---------|----------------|-------------------------------|
| p2 production (32 x 1x5 width shards, row-major) | (1, 5120) bf16 | 1.000000 | 0 | 0 | 0 | 0 | 0 | 1.0 / 1.0 / 1.0 |
| p2 other spec (54 shards of 3, last of 1, col-major) | (1, 5120) bf16 | 1.000000 | 0 | 0 | 0 | 0 | 0 | 1.0 / 1.0 / 1.0 |
| p1 production (103 x 10x1 height shards, 2 ranges) | (1, 128, 256, 32) bf8b | 1.000000 | 0 | 0 | 0 | 0 | 0 | 1.0 / 1.0 / 1.0 |
| p1 other spec (94 shards of 11, last of 1, col-major) | (1, 128, 256, 32) bf8b | 1.000000 | 0 | 0 | 0 | 0 | 0 | 1.0 / 1.0 / 1.0 |

**Assessment:** bit-exact on every cell, bf8b shared exponents included (the reference is the host unpack
of the device input as stored). The ratio is exactly 1.0, so there is no scale signature. `comp_allclose`
reports `Max RTOL Delta: nan` on bf8b only because the reference holds exact zeros (0/0). It is not an
error.

**Recommended tolerances:** exact for both dtypes, as `helpers.TOLERANCES` already declares. No PCC or RMS
limit applies.

## Verifier CLI Summary

Golden run `eval/eval_test_runner.sh eval/golden_tests/g6_sharded_to_interleaved/`: 13/17 passed, 0
failed, 0 hangs. `python3 -m eval.verify_supported` (artifact: `verifier_report.json`):

- supported_pass: 13 (1 cartesian cell, 7 loose cases, 5 regression tests, all inside Phase 0)
- xfail_expected: 4 (p1 production, p1 loose, p1 other spec: bf8b/height/l1; p2 to L1: bf16/width/l1).
  All four are covered by Refinement 1.
- invalid_skipped: 0
- supported_fail: 0
- xpass_drift: 0
- xfail_wrong_mode: 0

Other suites, all green:

| Suite | Result |
|-------|--------|
| Acceptance | 14 passed, 3 xfailed (the same out-of-SUPPORTED cells) |
| impl_debug | 29 passed: 7 scenarios x 4 knob sets, bit-exact, plus the plan and memo tests |
| trace_replay | 1 passed |
| precision baseline | 4 passed |
| perf harness | 8 passed |

## Performance Baseline (input to the perf refinements)

Device kernel duration is the median of calls 3–12 of 12 eager calls, measured with
`test_g6_sharded_to_interleaved_perf.py::test_eager_calls` under `--profile`. Trace replay is the host
clock over 50 replays of a 16-call trace, 3 samples, from `test_trace_replay_ns_per_op`.

| Case | lanes | Kernel ns (median / min) | BRISC / NCRISC ns | First-to-last start ns | FW duration ns | Trace replay ns/op |
|------|-------|--------------------------|-------------------|------------------------|----------------|--------------------|
| p1 (PERF FOCUS) bf8b height to L1, 103 cores | 1 | **2110** / 2044 | 2110 / — | 452 | 3042 | **5950–5953** |
| p1 | 2 | 1875 / 1829 | 1462 / 1873.5 | 458 | 2807 | 6224–6233 |
| p2 bf16 width to DRAM, 32 cores | 1 | **1494.5** / 1462 | 1494.5 / — | 434 | 2016 | **2843–2847** |
| p2 | 2 | 1563.5 / 1533 | 1207.5 / 1563.5 | 451 | 2084 | 2888–2893 |

Readings:
- p1 at the default of 1 lane is **slower than the stock incumbent (2.05 us)**. The flagged goal is not
  met at Phase 0.
- On p1, 2 lanes take 235 ns off the kernel but add 280 ns to trace replay: the second kernel's dispatch
  cost outweighs the saving. The NoC0 lane (NCRISC) is also the long pole at 1873 ns against 1462 ns on
  NoC1. On p2, 2 lanes lose on both metrics. The lane default of 1 is right under the current dispatch
  cost.
- p1 replay is 5.95 us against a 2.11 us kernel, so per-call dispatch and firmware cost is about 3.8 us.
  That is about 2.5 us more than p2 carries (1.35 us), across 71 more cores and one more range. Per-core
  unique RT args (6 words x 103 cores, written per core on every replay) are the likely driver (Refinement
  3).

## Recommendations

- Queue (`op_requirements.md`): Refinement 1 claims bf8b, height_sharded and L1 together. Each is a
  knob-turn with no kernel change, and the perf-flagged p1 needs all three. Then two perf phases on p1:
  Refinement 2, the write issue order (L1 bank spread, incremental bank walk); and Refinement 3, the
  per-call dispatch cost (per-core RT args).
- The lane knob is measured, not open. Re-measure it after Refinement 3, when lower dispatch cost may let
  the kernel-side gain of 2 lanes show in replay.
- Deleted test files (`*_debug.py`, `*_debug_dispatch.py`, `*_knobs_debug.py`, `*_lane_lamp.py`,
  `*_perf_cases.py`, `*_trace_perf.py`) were superseded by `test_..._impl_debug.py` and
  `test_..._perf.py`. They are removed in this commit (BB-22).
