# Verification Report: g6_interleaved_to_sharded

Board: Blackhole p150 (chip 4, 11x10 compute-grid override), 2026-11-27. All numbers below were re-measured in this pass.

## Code Review

### Fixed in this pass

Items 1–4 were found by an earlier verifier pass in this run that ended without committing. This pass re-checked each one against the source and on device, kept it, and commits it.

| # | Finding | Fix |
|---|---------|-----|
| 1 | `validate()` refused only a width shard *shorter* than the tensor and a height shard *narrower* than it. TensorSpec requires **equality** on the unsharded dim (`tt_metal/impl/tensor/spec/tensor_spec.cpp:52-68`, re-read). A taller width shard (`(1, 5120)` into `(64, 160)`) or a wider height shard (`(1,1,1,128)` into `(32, 256)`) therefore passed `validate()`, built and cached a program, then hit a `TT_FATAL` in `allocate_tensor_on_device`. | Both checks are now `!=` and raise `ValueError` inside `validate()`, before planning. `test_g6_interleaved_to_sharded_validation.py` pins both cases and asserts that a refusal caches no program. Re-run of the gate-lifted probe: the wide height shard now raises `ValueError` ("height shard of 8 tile columns does not match the tensor's 4"). |
| 2 | The descriptor cache key used `id(input_tensor.device())`. Python can reuse a wrapper's address after a device closes. | The key now uses `input_tensor.device().id()`. |
| 3 | The shard-multiple check used a literal 32. `describe_plan()` returned the cached plan dict by reference, so a caller could mutate the cache. Argument loading and `TensorAccessor` setup sat outside any zone (BB-21). | The check now uses the tile shape from `input_tensor.tile.tile_shape`. `describe_plan()` returns a deep copy. The arg loads moved inside `I2S-FETCH`; zones compile away in the plain build. |
| 4 | `op_design.md` Parameters listed `partial_tile_read ∈ {valid_rows, whole_tile}`, but the code also implements `face_rows` (lamp L2). | The design row is updated. |
| 5 | **This pass.** DRY: `KNOB_VALUES["issuers_per_core"] = (1, 2)` repeated the length of `ISSUERS` as a literal. Adding a value without an `ISSUERS` row would have raised an `IndexError` in `_build`. | `KNOB_VALUES["issuers_per_core"]` is now derived from `ISSUERS` (`tuple(range(1, len(ISSUERS) + 1))`). |

`--dev` (watcher, NoC sanitizer, lightweight asserts) on the acceptance and knob files: **52 passed, 2 xfailed**, no watcher or sanitizer report.

### Checked and conforming (no change)

- **Kernel API.** `void kernel_main()` and `api/dataflow/dataflow_api.h` are used. Addressing goes through `TensorAccessor` (no `InterleavedAddrGen`). Reads use the one-packet `noc_async_read<tile_bytes>(src, dst, size, noc)`, guarded by `static_assert(tile_bytes <= NOC_MAX_BURST_SIZE)`. The issuer's NoC id goes to `get_noc_addr(page, 0, noc)`, `noc_async_read` and `noc_async_read_barrier` alike. Signatures checked at `dataflow_api.h:546`, `tensor_accessor.h:554`.
- **CB sync.** There are no CBs: every read lands at its final shard slot.
- **Helpers.** None applies, and the design's rejection table is accurate. No kernel_lib helper reads interleaved DRAM pages into an arbitrary L1 address. Compute helpers would route the bytes through unpack/DEST/pack and break bit-exactness. `local_copy_helpers` are L1→L1 only. `McastArgs` has no shared operand to carry.
- **Broadcast efficiency.** Not applicable: no operand is reused across cores.
- **Measurement scaffolding.** `ABLATE_READS` / `G6I2S_ABLATE_READS` is kept on purpose. It compiles out, it is in the cache key, and the perf bench uses it to measure the sync-only floor.
- **Host cost per call.** `validate()` recomputes `compute_geometry()` (including `corerange_to_cores`) on every call, hit or miss. This is host-only and absent under trace replay, which is how the model calls the op. Noted, not changed.

### Prompt rules (`eval/prompts/g6_interleaved_to_sharded.txt` `## Rules`)

| Rule (condition applies now) | Status |
|---|---|
| One program, each page read once, no staging tensor, no ttnn data-movement op | ✓ one `ttnn.generic_op`. Golden: `device_num_programs = 1` on graded p1 calls. |
| Phase 0 SUPPORTED holds the listed values, EXCLUSIONS empty | ✓ |
| Result carries exactly `memory_config`; tiles row-major in each shard | ✓ output is allocated from `memory_config`. Golden `placement_problems` and acceptance `_check` pass. |
| bf16 bit-exact, incl. NaN payloads, ±inf, −0.0, denormals | ✓ no engine on the path. `special_values_bit_exact` and `poisoned_padding` pass. |
| Core count is the blocking decision's choice | ✓ one owner per shard (p1: 32 cores; golden `device_num_cores = 32`). |
| Non-aligned H handled natively, independent of tile padding | ✓ the valid-row span lives in the reader; NaN-poisoned padding never reaches the result. |
| OOM on golden INPUTS | n/a. The op adds 0 B; golden `device_l1_peak_bytes = 10,240` for p1. |
| Orientation followed | ✓ `G` comes from `ttnn.corerange_to_cores(..., row_wise=orientation==ROW_MAJOR)`; col-major cases pass. |
| Height-sharded `G[r div sh]` for any shards/images | Re-arms with Refinement 1. Pre-checked by the gate-lifted probe (below). |
| BB-03 nothing in flight at return | ✓ one `noc_async_read_barrier(issuer_noc)` per issuer. `test_repeat_then_other_program` passes in plain and `--dev` builds. |
| BB-05 L1 ≤ 1,347,776 B, refused on the host otherwise | ✓ shard-bytes check in `validate()`. |
| BB-06 bit-identical repeats | ✓ |
| BB-07 no rebuild on repeat; complete cache key | ✓ `test_program_cache_and_addresses`; golden `test_cache_key_shard_spec`, `test_new_input_address`. |
| BB-08 / BB-21 plan report with per-core role | ✓ `describe_plan()`. |
| BB-16 trace replay = eager; no build or alloc on replay | ✓ re-run of probe 011: replay equals eager bit for bit, and new input values written into the same buffer are picked up. |
| BB-17 few rectangular core ranges | ✓ one kernel group over the shard grid's own ranges. |
| BB-04 / BB-18 kernel config fits | ✓ one small reader: 8 CT + accessor args, 6 RT per core, 2 common RT. |
| BB-22 tests limited to production + rule-required cases | ✓ |

No soft (prefer/consider/avoid) rules apply.

## Registry Conformance

- `INPUT_TAGGERS` (5 taggers, all `(inputs, axes)`), `SUPPORTED` (all 5 axes), `EXCLUSIONS = []` and `validate()` are present. `validate()` order:
  1. malformed checks → `ValueError`;
  2. taggers over the live call's scenario dict (one tagging source);
  3. SUPPORTED → `UnsupportedAxisValue`;
  4. EXCLUSIONS → `ExcludedCell`.

  The entry point calls `validate()` first.
- The op file does **not** declare `INVALID`.
- No SUPPORTED auto-fix was needed: there is no XPASS evidence, and `validate()` refuses rank 4 / height as Phase 0 requires.
- **INVALID audit** (`feature_spec.py`): `INVALID = []`, which is correct. Every axis is projected from the two production scenario dicts, so `cartesian()` forms no combination outside them. TARGET is bf16 + TILE only, so the canonical bf8b + ROW_MAJOR entry has no cell to cover. There are no cross-tensor couplings, and this is not a norm-like op.

## Design Conformance / Blocking Model

- **Algorithm and topology.** Owner-fetch, exactly as designed. Each shard owner reads its rectangle from DRAM straight into its L1 slots, one request per tile, then issues one barrier. There is no CB, no compute or writer kernel, and no core-to-core traffic.
- **Work distribution.** One owner per shard on the shard grid; p1 fills all 32 cores of the 8x4 grid. The placement fixes the core count. More issuers come from inside the core (BRISC, lamp L1), as the design says.
- **Block knobs.**
  - Block extents are per-core RT values from one host function (`owner_block` → `issuer_blocks`).
  - Spans are CT values from `partial_spans`.
  - `issuers_per_core` and `partial_tile_read` live in `KNOBS` and in the cache key.
  - `tile_bytes`, face sizes and strides are derived once from the tensor's tile.
  - No count, extent or grid is inlined in the kernel. DRY holds after fix #5.
- **Sharded default.** One block = the whole shard. Nothing consumes the data inside the program, so a smaller block would only add barriers. This is not a half-turned split.
- **Expression.** The reader issues the whole block, then one barrier. No unit is acquired, synchronized and retired on its own. There is no compute or writer side to check.

## L1 Ledger Audit

- **Currency.** The ledger lists zero op CBs, and the code declares `cbs=[]`. The only L1 row is the output shard (caller's placement), `shard_tile_rows * shard_tile_cols * tile_bytes`. The golden harness measured `device_l1_peak_bytes = 10,240` for p1, exactly the ledger's figure.
- **Capacity vs live set.** Over-capacity exists only for a partial last shard, and it comes from the caller's geometry. Under-capacity: the live set spans `tile_row` and `tile_col`, and capacity scales with both. Nothing flagged.
- **Page format vs DEST.** n/a: there is no DEST on the path.
- **Disjoint lifetimes.** n/a: one buffer.
- **Bounds and closed form.** Every symbol is bounded:
  - shard bytes ≤ 1,347,776 (BB-05 check in `validate()`);
  - block ≤ shard (`min` clamps);
  - span ≤ `tile_bytes` (kernel `static_assert`).

  Fix #1 makes the unsharded dim equal by construction.
- **Data-movement budget.** Present and consistent with the code. Each page crosses DRAM once (p1: 160 × 544 B = 87,040 B) with 0 core-to-core bytes. The cheapest-traffic split is the implemented one. `face_rows` (the 10,240 B logical floor) is built as a knob and measures slower (request-count bound); its row carries that positive reason.
- **Block-size defaults.** Sharded → whole shard, held.
- **Per-core footprint.** `L1_op(core) = shard_tile_rows * shard_tile_cols * tile_bytes + 0`. No term scales with a knob.

No ledger finding to fold into a refinement.

## Precision Baseline

`tests/ttnn/unit_tests/operations/g6_interleaved_to_sharded/test_g6_interleaved_to_sharded_precision_baseline.py` (randn bf16, seed 0; PCC via `comp_pcc`, abs errors via `comp_allclose`):

| Case | Shape | PCC | Max Abs Err | Mean Abs Err | Relative RMS Err | Max ULP | got/true ratio median [p5, p95] |
|------|-------|-----|-------------|--------------|------------------|---------|-------------------------------|
| p1 row-major, 32 shards on 8x4 | (1, 5120) | 1.000000 | 0 | 0 | 0 | 0 | 1.0 [1.0, 1.0] |
| p1 col-major, 32 shards on 8x4 | (1, 5120) | 1.000000 | 0 | 0 | 0 | 0 | 1.0 [1.0, 1.0] |
| p1, 8 shards of (32, 640) on 8x1 | (1, 5120) | 1.000000 | 0 | 0 | 0 | 0 | 1.0 [1.0, 1.0] |
| regime pin (full + partial tile rows), (64, 160) on 8x4 | (33, 5120) | 1.000000 | 0 | 0 | 0 | 0 | 1.0 [1.0, 1.0] |

**Assessment**: bit-exact on every case. There is no rounding to characterize and no scale signature.
**Recommended tolerances**: exact. `helpers.TOLERANCES[bfloat16] == ("exact", None)` holds. Any nonzero ULP is a bug (a value-changing path), never precision.

## Verifier CLI Summary

Golden run: `eval/eval_test_runner.sh eval/golden_tests/g6_interleaved_to_sharded/ ttnn/ttnn/operations/g6_interleaved_to_sharded/verify_results` → 9/13 passed, 1 failed, 3 xfailed. CLI report: `verify_results/verifier_report.json`. The CLI needs the chip-scope env (`source .tt_chip_scope`, which preloads the `MPIX_Comm_*` stub); without it `import ttnn` fails on `MPIX_Comm_shrink`.

- supported_pass: 4 (p1 `test_op`, loose `flux58_p1`, `special_values_bit_exact`, `poisoned_padding`)
- xfail_expected: 2 (p2 `test_op` and loose `flux57_p2`: rank 4 / height → Refinement 1)
- invalid_skipped: 0 (INVALID is empty)
- supported_fail: 0
- xpass_drift: 0
- xfail_wrong_mode: 0
- no_axes_found: 7 regression tests:
  - 5 passed;
  - `test_height_multi_shard` is xfail-strict on rank 4 (Refinement 1);
  - `test_refuse_uncovered_shard_spec` failed (below).

Golden metrics for `flux58_p1`: `device_kernel_ns = 1106`, `device_num_cores = 32`, `device_num_programs = 1`, `device_l1_peak_bytes = 10,240`.

**`test_refuse_uncovered_shard_spec` is a harness defect, not an op failure.**
- **Cause.** Under `eval/eval_test_runner.sh` (`EVAL_CAPTURE_L1=1`), `eval/l1_profiling.py:capture_for_item` raises `L1ProfilingError: peak_total=0` (line 84) for any test that passes without allocating L1. This test only places a DRAM input and expects a `ValueError` for a shard spec with fewer cores than shards. The op raises it in `validate()` before allocating anything, which is the correct behaviour.
- **Under the plain runner.** The same test body passes under `scripts/run_safe_pytest.sh`, which does not load the metrics plugin. The acceptance twin `test_malformed_request_is_refused[fewer_cores_than_shards]` passes as well.
- **Fix location.** The op cannot make this pass honestly: allocating L1 before refusing would be wrong. The fix belongs in the harness (accept `peak_total == 0` when the test dispatched no device program) or in `test_regression.py` (exempt refusal tests from L1 grading). Please update it via `/golden-tests` or directly. The eval tree is read-only for this pass.

**Gate-lifted pre-check for Refinement 1** (probe 010, re-run in this pass: in-process `SUPPORTED` override, NaN-poisoned padding, bit-exact `check_output`):

| Case | Result |
|---|---|
| p2 `(1,1,1,128)`, 1 height shard on (0,0) | PASS |
| `(2,1,33,128)`, 4 height shards on 4 cores (multi-image, full + partial rows) | PASS |
| `(2,3,33,128)`, 6 height shards on 4x2 col-major | PASS |
| `(3,1,1,256)`, partial last height shard | PASS |
| rank-4 width `(1,1,1,5120)` and `(2,1,33,256)` | PASS |
| height shard wider than the tensor, `(1,1,1,128)` → `(32,256)` | `ValueError` in `validate()` (fix #1) |

## Performance (perf-focus case, FLUX #58 p1)

Device kernel duration, 5-call medians, profiled (`scripts/run_safe_pytest.sh --profile --no-py-zones test_g6_interleaved_to_sharded_perf.py`), re-measured in this pass:

| Variant | ns |
|---|---|
| default (1 issuer, `valid_rows` 544 B span) | **1101** (golden harness: 1106) |
| 2 issuers (NCRISC/NOC0 3 tiles + BRISC/NOC1 2 tiles), `valid_rows` | 1044 (−5.2%) |
| 1 / 2 issuers, `face_rows` (2 × 32 B per tile) | 1306 / 1133 |
| 1 / 2 issuers, `whole_tile` (2048 B) | 1702 / 1479 |
| reads ablated (launch + args + barrier only), 1 / 2 issuers | 548 / 575 |

**Where the time goes** (per-core zone medians, raw device profile log, same runs):

| Variant | I2S-FETCH (issue) | I2S-RETIRE (wait) | NCRISC kernel | kernel-start skew across the 32 cores |
|---|---|---|---|---|
| default | 233 ns (5 reads → ~44 ns/read) | 326 ns | 637 ns | 444 ns |
| 2 issuers, `valid_rows` | 184 ns (≤3 reads → ~56 ns/read) | 301 ns | 566 ns | 459 ns |
| reads ablated, 1 issuer | 13 ns | 28 ns | 96 ns | 452 ns |

**Reading.** The 1101 ns device duration has two parts:

1. **Kernel-start skew, ~450 ns.** The cores enter their kernels up to ~450 ns apart. The skew appears at `BRISC-FW` zone start, which opens right after go-signal receipt (`tt_metal/hw/firmware/src/tt-1xx/brisc.cc:571`), and it is the same with reads ablated. It is dispatch or firmware side (go-signal arrival, or core timer bases), and no op change removes it.
2. **Per-core kernel time, ~640 ns.** Of this, ~300 ns is the DRAM round trip after the last issue (irreducible) and ~230 ns is NCRISC issuing 5 reads.

At ~44 ns per one-packet read issue, read issue is the op-controllable headroom. Ways to recover it:
- split the issue across two RISCs (measured −57 ns);
- cut the per-read issue cost (stateful one-packet read registers, incremental bank addressing instead of a per-page `get_noc_addr`).

Together that is at most ~150–200 ns of the 1101.

The feature spec's goals: "faster than stock 1.43 µs" is met. "≤ 1.35 × 0.02 µs" is a pure DRAM-bandwidth bound that ignores program launch; the read-ablated floor alone is 20× larger, so it is unreachable for any one-program implementation.

## Recommendations

- **Refinement order.** Refinement 1 (height + rank 4) is a knob-turn on the existing kernel and host split; the gate-lifted probe already shows it bit-exact. Refinement 2 (perf on p1) is independent of it. Once Refinement 1 lands, the height path joins Refinement 2's correctness guard set.
- **`issuers_per_core` default.** The implementer kept 1 issuer despite a 5.2% profiled win, citing RB-053: profiled duration can mislead for RISC-placement changes. That is a reason to *check the change on trace replay*, not to stop. Refinement 2 owns the decision.
- **No third perf phase is filed.** After Refinement 2 the remainder is go-signal skew (~450 ns) + firmware launch + one DRAM round trip (~300 ns) per core, none of which an owner-fetch kernel controls. Regime C (idle cores forward pages to owners) is the only scheme that adds issuers beyond two per owner. It adds a second hop per page and spreads the launch skew over more cores; the design rejected it for that reason, and nothing measured here contradicts that.
- **Bank stagger (lamp L3).** Likely a no-op on p1: `shard_tile_cols = 5` is coprime with the 8 banks, so same-row cores start on distinct banks. It matters on the `(32, 640)` 8-core placement (20 tiles per core, starts alternate banks 0/4). Measure it there as part of Refinement 2.
- **Harness.** Fix the L1-profiling defect on refusal tests (above). Until then the golden summary shows 1 failed test that no op change can clear.
