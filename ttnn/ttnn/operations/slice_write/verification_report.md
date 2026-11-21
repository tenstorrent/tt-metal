# Verification Report: slice_write

Target: Quasar (`quasar`), on the ttsim simulator. Specification: `eval/specs/slice_write/spec.py` (sha256 `c4fe3356226d…`). Specification lint (`python3 -m eval.validate_contract slice_write --phase0`): **PASSED**, 0 errors. Its 5 open items are UNRESOLVED fields of the draft specification (`governs`, `special_values`, `freedoms`, `PERFORMANCE.bar` / `measure`), none of which the op can act on.

## Code Review

Fixed in place:
1. **Writer-hart knob refused valid calls (design conformance).** `create_program_artifacts` raised `ValueError` when `THREAD_PLAN`'s W did not divide the shard's rows. The design says "requires W | shard_rows, else W = 1" (Perf lamps). At W = 1 this never fired, but turning the knob would have broken the traced call (6785 rows) and any odd shard. It now runs at W = 1 in that case, with the same ring and kernel thread count. Regression test: `test_slice_write_plan.py::test_writer_harts_fall_back_when_rows_do_not_divide[2,4]`.
2. **DRY: run-time argument names.** The writer's `common_runtime_arg_names` was a literal list restating the keys of the `common` dict built further down. The schema is now `list(common)`, from the one dict.
3. **Ledger currency.** The `l1_ledger.md` symbol table described the old refusal and now records the W = 1 fallback.

Reviewed and found conforming, so nothing changed:
- **Kernels.** Both use `void kernel_main()`, `api/dataflow/*` includes and `TensorAccessor` (no `InterleavedAddrGen`).
- **Ring balance.** Push and wait/pop are balanced per entry (1:1 at W = 1). A ragged last shard writes only `rows_this_core` rows but keeps the nominal wait/pop.
- **Ring protocol.** The ring is explicit on both kernels (`disable_dfb_implicit_sync_for`). Every writer transfer and barrier is tagged trid 0, the tag is handed back as the last statement, and the reader issues no NoC transfer.
- **Helpers.** None applies. There is no compute kernel, the `dataflow_kernel_lib` local copy is L1→L1 only, and there is nothing to multicast. The raw per-row write is the block operation by construction, since each row is its own DRAM page in its own bank.

Deferred: nothing architectural. The perf headroom is queued (Refinements 3 and 4).

## Specification rules

| Rule | Applies at Phase 0 | Status |
|---|---|---|
| R1: values unchanged, no dtype round trip | yes (bf16) | holds by construction: bytes go L1 → NoC → DRAM with no format. Precision baseline is bit-exact. Note for Refinement 2: uint32's ring label must be a Quasar DFB format (`int32`), which is a label only |
| R2: write the block only | yes | the page map addresses only block pages. Golden destinations are random, and every cell passes the exact reference |
| R3: in place, return the caller's tensor | yes | `io_tensors = [input, output]`, nothing allocated, returns the `output_tensor` object (tests assert `out is dst_dev`) |
| R4: source untouched | yes | the reader only publishes and the writer only reads the borrowed entry. The baseline and plan tests compare the source afterwards |
| R5: lists or tuples | yes | `_as_int_tuple`. The loose case `bounds_as_lists` passes |
| R6: `step` via `canonical_step` | yes | `validate()` reads the raw `step` through the specification's `canonical_step` |
| R7: placement axes via `memory.memory_axes(input)` | yes (height-sharded L1, any core count) | gated like any other axis. 1, 4, 8 and 32 cores, a ragged shard and a COL_MAJOR grid are tested |
| R8: destination DRAM interleaved, same dtype and layout | yes | `validate()` reads `memory_axes(output_tensor)`, its dtype and its layout, and raises `UnsupportedAxisValue` (a `NotImplementedError`) |
| R9: `NotImplementedError` outside SUPPORTED | yes | 167 xfail_expected, 0 xfail_wrong_mode |
| REJECTIONS X1–X3 | yes | `ValueError` before `validate()`'s axis gate. The 3 golden rejection tests pass |

GUIDANCE (advisory):
- The guidance's "one data path for every placement: read through the source's accessor" is deliberately not followed for the sharded source. The design reads the resident shard zero-copy through a borrowed ring instead of re-reading it through an accessor. This is the verifier's native-sharding requirement, and it also matches the guidance's own "each core can move the pages of its own shard".
- The other guidance items hold: no compute, page map, in place, nothing allocated.

## Registry Conformance

- `INPUT_TAGGERS = AXES`, imported from the specification; there is no op-local axis function.
- `SUPPORTED` equals `PHASE0` axis for axis, and covers every axis the kernel path gates on.
- `EXCLUSIONS = []`.
- `validate()` checks REJECTIONS, then SUPPORTED per axis, then EXCLUSIONS, then R8 and the zero-volume refusal. It is the first statement of `slice_write()`. The op file declares no `INVALID`.
- No XPASS drift, so SUPPORTED needed no auto-fix.
- **INVALID audit** (specification): one entry, `{memory_config: DRAM, sharding: height}`.
  - It describes a single tensor (the source's two placement axes), and the universe changes structurally because the runner builds no DRAM-sharded tensor (CONVENTIONS §2.3). Well-formed.
  - The canonical `bfloat8_b + ROW_MAJOR` entry is not needed: `bfloat8_b` is not in TARGET and has no path on Quasar.
  - Not norm-like: no weight canonicalization.

## Design Conformance

- **Algorithm, pipeline, inter-core communication.** These match `op_design.md`: page map, regime `shard_owner_writes`, borrowed one-entry ring, reader publishes, writer walks the page map incrementally with one division per entry. There is no inter-core communication, as designed.
- **Work distribution.** The independent axis `src_row` is cut exactly at the source's shards (caller-pinned). That is also the cheapest-traffic split.
  - On the golden shard configs this engages up to all 32 clusters. Cores with no valid rows are left out of the work unit.
  - The writer batches its barriers (32 writes per barrier). The reader moves no data.
  - The one under-filled shape is the caller's own single-shard config of the traced call: 1 cluster, 6785 rows. That is the deferred regime `spread_beyond_shard_grid`, queued as Refinement 4.
- **Thread plan.** Read off the host program: `THREAD_PLAN = (1, 0, 1)`, one named constant, passed to both kernels as CT args. The writer's entry loop strides by `writer_threads` from `get_my_thread_id()`.
  - The design states the op as bound by data movement, with its reason: zero compute and one NoC write per row. That is the target page's plan for such an op.
  - Present and passing: the test that reads the plan off the built `ProgramSpec`, the writer-hart knob at W = 1, 2, 4 (a missing hart leaves rows stale against a random destination), and the session test (tilize → slice_write ×2 → `to_layout` tile and back, in one device session).
  - ttsim has no timing model, so the plan was not timed against alternatives. No model-cycle reference exists for this phase (`pipeline_logs/` has no `model_cycles.jsonl`).
- **Blocking-model fidelity.** Each knob is a parameter with one source: `THREAD_PLAN`, `WRITE_BATCH_ROWS` → `rows_per_write_batch` (CT), `src_shard_entries = W`, `entry_rows` (RT).
  - No CB scales with an op dimension. The single ring is borrowed onto the caller's shard, so it is the shard itself.
  - The schedule reads as blocks. There is one block per core (the whole resident shard, the sharded default), one credit handshake per entry, and synchronization grouped per batch of 32 rows. The per-row write is the transfer unit (one page), not a per-unit synchronization.

## L1 Ledger Audit

- **Currency.** One ring, `dfb_src_shard`: W entries of `(shard_rows / W) · src_row_stride_bytes`, borrowed. The ledger row matches the code. The symbol table was updated for the W fallback (fix 3 above).
- **Capacity vs live set.** The capacity is the whole shard. The live set is also the whole shard, because every row is a NoC write source until its batch barrier and the shard is the caller's memory. There is no over- or under-sizing: the row spans `src_row` and `row` and streams over `write_batch`.
- **Page format vs DEST.** Not applicable: metadata only, a dataflow-only payload that never passes through DEST, and there is no compute.
- **Disjoint lifetimes.** None: there is one buffer.
- **Bounds.** Every symbol is bounded in the table, and the total is closed-form: 0 B added per core.
- **Data-movement budget.** Present, and consistent with the code: DRAM `B` written once, 0 read, 0 cross-core.
  - The cheapest-traffic split is the implemented one.
  - The deferred regimes carry positive reasons. `interleaved_stream` is outside PHASE0; `spread_beyond_shard_grid` adds bytes and is held back by the traffic ranking; `bank_coalesced_writes` is covered by the built regime and needs the interleave function.
- **Block-size defaults.** The sharded default (one block = the whole resident shard) holds.
- **Per-core footprint.** `0 B` allocated by the op, with no term scaling with any knob. In the deferred `interleaved_stream` regime it becomes `2 · rows_per_block · src_row_stride_bytes`.

## Precision Baseline

`tests/ttnn/unit_tests/operations/slice_write/test_slice_write_precision_baseline.py`: bfloat16 source height-sharded in L1, random destination in DRAM. Metrics are over the whole destination, against the float64 reference.

| Shape (destination, block) | Cores | PCC | Max Abs Err | Mean Abs Err | Relative RMS Err | got/true ratio median [p5, p95] |
|---|---|---|---|---|---|---|
| (1,64,32,32), rows 0–32 | 32 | 1.0 | 0.0 | 0.0 | 0.0 | 1.0 [1.0, 1.0] |
| (1,115,115,32), rows 0–59 (traced) | 32 | 1.0 | 0.0 | 0.0 | 0.0 | 1.0 [1.0, 1.0] |
| (1,4,512,32), rows 1–3 | 8 | 1.0 | 0.0 | 0.0 | 0.0 | 1.0 [1.0, 1.0] |
| (1,2,32,4096), row 1 | 4 | 1.0 | 0.0 | 0.0 | 0.0 | 1.0 [1.0, 1.0] |

**Assessment**: bit-exact (max ULP error 0). Every element of the block equals the source and every element outside it equals the old destination. The op moves bytes and never converts them, so there is no scale or rounding signature to triage.

**Recommended tolerances**: exact. Every cell resolves to `exact` in `NUMERIC_RESOLVED` (bfloat16, float32 and uint32 alike).

## Verifier CLI Summary

Results: `/tmp/sw_golden2` (copied to `verifier_report.json` in this directory). 244 tests.
- supported_pass: 14
- xfail_expected: 167
- invalid_skipped: 60
- supported_fail: 0
- xpass_drift: 0
- xfail_wrong_mode: 0
- no_axes_found: 3 (the rejection tests X1–X3, all passed)
- queue gaps (`--queue op_requirements.md`): 0 of 6 `TARGET − SUPPORTED` values

Unit tests: 19/19 (acceptance 4, plan / session / shard configs 11, precision baseline 4).

## Recommendations

- **Probe findings feed Refinement 2.** All 47 out-of-phase sharded cells were run with SUPPORTED widened temporarily, then reverted.
  - float32, `h` and `w_non_aligned` passed with no code change, which retires the design's sub-16 B-multiple write risk on the sharded path.
  - uint32 fails at program build because `UInt32` is not a Quasar DFB format. It passes with the ring labelled `ttnn.int32`.
- **The interleaved regime is the only structural work left (Refinement 1).** It must ship batched (one read barrier per block, depth-2 ring, full-grid split), because Refinement 3's guard set includes it.
- **Perf headroom is all on the traced call in the caller's own single-shard config.**
  - Refinement 3 covers write issue within one cluster. The writer-hart knob cannot engage today because 6785 has no small divisor.
  - Refinement 4 covers occupancy across clusters. It is a scheme-change that adds cross-core bytes, so its value is occupancy only.
  - On ttsim both are gated on structural counts, and their time gains are unconfirmed until a timed backend measures them.
- **Not queued:** `bank_coalesced_writes`. It needs the page → bank interleave function that `TensorAccessor` does not expose, and it only trades local copies for fewer DRAM transactions. Revisit only if a timed backend shows the per-row DRAM transaction rate bounding the op once Refinement 3 has landed.
- **Unconfirmed risk:** the per-trid outstanding-write counter width is undocumented (`noc.h:422`, TODO #31535). Keep `WRITE_BATCH_ROWS ≤ 32` until it is confirmed; Refinement 3 must not raise it blind.
