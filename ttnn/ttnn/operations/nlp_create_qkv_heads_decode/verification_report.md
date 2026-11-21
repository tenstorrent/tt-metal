# Verification Report: nlp_create_qkv_heads_decode

Target: Quasar on the ttsim simulator (no clock; perf is structural). Specification: `eval/specs/nlp_create_qkv_heads_decode/spec.py` (lint: PASSED, 0 errors, 5 UNRESOLVED open items, none of them the op's).

## Code Review

Fixed in place:

- **DRY: tile geometry defined twice on the host.** `TILE = 32` was a literal in both `nlp_create_qkv_heads_decode.py` and `nlp_create_qkv_heads_decode_program_artifacts.py`, and `FACE = 16` sat only in the op file.
  - The program-artifacts module is now the one host source of `TILE` and `FACE`.
  - `FACE_ROWS_PER_TILE` is derived from them (`(TILE // FACE) ** 2 * FACE` = 64), and the op module imports them.
  - The dead constant `FACES_PER_TILE_ROW` is removed.
- **DRY: `input_width_tiles` passed as a separate compile-time arg.** It restated `sections * head_dim_tiles`; the kernel only cross-checked it with a `static_assert`.
  - The kernel now derives it from `num_q_heads`, `num_kv_heads` and `head_dim_tiles`, and the host asserts the equality.
  - The design's Parameters table now says where it is derived.
- **Dead placeholder kernels removed.** `kernels/nlp_create_qkv_heads_decode_compute.cpp` and `..._writer.cpp` were empty `kernel_main(){}` files and not part of the program. They were kept for a "conventional layout" that nothing requires (checked: no harness or agent rule names them). A later implementer could mistake them for live stages.

Reviewed, no change needed:

- **Kernel API usage.** `void kernel_main()`; `api/...` include paths; `TensorAccessor(tensor::in)` as the source; `LocalTensorAccessor` destinations (zero-copy into the shards).
- **Transaction ids.** Every read is `NocOptions::TXN_ID` with `{.trid = 0}` and the barrier is on trid 0. The last statement is `noc_async_read_set_trid(0, noc.get_noc_id())`, which closes the sticky-tag hazard on the target page.
- **Helpers.** None apply. No compute and no ring exist, and the design's "helpers considered and rejected" table holds: no kernel-library helper gathers sub-page runs of an interleaved tensor into a tensor shard.
- **Expression.** Reader-only program. Each hart issues every run it owns in a block with no per-unit synchronization, then one barrier per block. Nothing serializes per unit.
- **Advisory (not a defect).** Each hart walks the whole flat run list and issues only the runs where `phase == thread_id`, so the loop runs R× per hart. That is a few scalar instructions per run against one NoC issue, so I left it. Refinement 2 may step by R directly if it raises R.

Deferred to refinements: none from code review.

## Specification Rules

| Rule | Applies now | Status |
|------|-------------|--------|
| R1 exact movement, no dtype pass-through | yes (bf16) | OK: no intermediate buffer. Reads land in the output, and the baseline is bit-exact. Re-arms for fp32 (Refinement 1) and is met by construction there too. |
| R2 list of three, shapes, dtype, tile layout | yes | OK (golden + acceptance). |
| R3 users from the logical shape, padding rows never read | yes | OK: only row `u % 32` of tile row `u // 32` is read, and `users = shape[2]`. |
| R4 `num_kv_heads=None` → `num_heads` | yes | OK (INPUTS entry 4608/12/None passes). |
| R5 placement: height-sharded L1, exact shard spec and cores | yes | OK: configs are built as the spec's `output_memory_configs` builds them, and the hook compares with `==`. |
| R6 `memory_config` read via `canonical_memory`; others refused by name | yes | OK: interleaved, DRAM-sharded and own-shard-spec configs become values outside SUPPORTED, refused with `UnsupportedAxisValue`. A non-config raises `TypeError` (X7). |
| R7 `overlap_qk_coregrid` never read or gated | review | OK: it passes through the signature of `validate()` and nothing reads it. |
| R8 DRAM or L1 interleaved input | not yet (DRAM outside PHASE0) | Refinement 1. |
| R9 placement axes via `eval.memory.memory_axes`, gated like any axis | yes | OK. |
| R10 / X8 refuse row-major by name | yes | OK (`ValueError`, rejection test passes). |
| R11 `NotImplementedError` outside the phase | yes | OK: `UnsupportedAxisValue` subclasses `NotImplementedError`, and 72 / 72 out-of-phase cells are `xfail_expected`. |

GUIDANCE followed: one core per user, two runs per tile row dealt over two harts, reads through the input's accessor by page and offset. GUIDANCE pending: the aligned DRAM read through a scratch buffer, which lands in Refinement 1.

## Registry Conformance

- `INPUT_TAGGERS = AXES` is imported from the specification, and there are no op-local taggers.
- `SUPPORTED` covers every axis (`dtype`, `layout`, `memory_config`, `sharding`, `output_memory`, `users`, `head_rows`) and equals `PHASE0`.
- `EXCLUSIONS = []`.
- `validate()` refuses malformed calls first (X1–X8), then derives the axes exactly as the harness does, then checks SUPPORTED, then EXCLUSIONS. It is the entry point's first statement.
- The op file declares no `INVALID`.
- No drift auto-fixes were needed: 0 `xpass_drift`, 0 `xfail_wrong_mode`.

INVALID audit (specification): `INVALID = []`. That is correct for this rectangle, since every TARGET combination is buildable. The canonical bf8b + ROW_MAJOR entry does not apply: TARGET's layout is TILE only, its dtypes are bf16 and fp32, and Quasar has no bf8b path. There are no weight axes and no cross-tensor couplings.

## Design Conformance

- **Algorithm and topology.** Regime A, as designed: a direct gather of face-row runs from the interleaved input into the three local shards. No compute, no ring, no writer.
- **Parallelization.** One core, the R5 shard grid for users = 1. That is the specification's placement, not an under-filled split. The design recorded the occupancy question (perf lamp "Grid synchronization": regime D, heads across cores, was rejected because it doubles NoC crossings).
- **Thread plan, read off the host program.**
  - The `gather` `KernelSpec` has `num_threads=GATHER_THREADS` (= 2) and CT `gather_threads` from the same constant, and the kernel deals runs by `get_my_thread_id()`.
  - No compute kernel (C = 0) and no writer (W = 0), as the design's (2, 0, 0) says. The design states the op as bound by data movement (per-transaction issue) and gives its reason, and R = 2 follows `split_reader` and tt-metal's two-processor face split.
  - The required tests exist and pass: `test_every_hart_runs` (R = 1, 2, 3, 4, 6 on Llama and Qwen, exact, `num_threads == GATHER_THREADS`), and `test_one_device_session` (`quasar.add` before, two calls back to back, then `to_layout` to row-major, `to_memory_config` and another `quasar.add` after).
  - ttsim has no timing model, so the plan could not be measured against the design's alternatives (R = 1 and R = 4). `pipeline_logs/model_cycles.jsonl` does not exist for this run. Refinement 2 carries the measurement for a timed backend.
- **Blocking-model fidelity.**
  - `GATHER_THREADS`, `HEADS_PER_BLOCK` and `USERS_PER_CORE` are each one host constant.
  - `heads_per_block` is live (tested at 1, 5, 13), and every CT arg is derived from these once.
  - No CB exists, so none can be sized to an op dimension.
  - The single source was broken only by the duplicate literals fixed above.

## L1 Ledger Audit

- **Ledger currency.** The program declares no dataflow buffer, scratchpad or semaphore, and the ledger says so. Its three rows are the tensor-owned output shards, listed for accounting, and their size expressions (`(rows/32) * head_dim_tiles * out_page_bytes`) match the allocation.
- **Capacity vs live set.** The shards' capacity exceeds their data rows only by the head-axis tile padding the R5 contract fixes. No over- or under-sized buffer.
- **Page format vs DEST.** Not applicable: dataflow-only payloads, no DEST, no `fp32_dest_acc_en`.
- **Disjoint lifetimes.** None. The three shards are distinct outputs that are live at the same time, and that is justified.
- **Bounds.** Every symbol is bounded by a validate predicate. The total is closed-form: op-allocated 0 B; tensor-owned `(q_rows + 2*kv_rows)/32 * head_dim_tiles * out_page_bytes`, at most 98,304 B over INPUTS × TARGET (fp32, 8192 wide).
- **Data-movement budget.** Present and consistent with the code: 0 DRAM bytes; each used input byte crosses the NoC once in `2 * input_width_tiles` reads; outputs never move. The cheapest-traffic split is implemented, and the alternatives are `rejected` rows with reasons.
- **Block-size defaults.** Held: one block per core, the whole user.
- **Future.** Refinement 1 adds the first op-allocated buffer (the DRAM bounce scratch). Its ledger row is part of that refinement's scope.

## Precision Baseline

`tests/ttnn/unit_tests/operations/nlp_create_qkv_heads_decode/test_nlp_create_qkv_heads_decode_precision_baseline.py`: bf16, L1 input, `L1/height`, standard-normal input, seed 1234. Values are the same for q, k and v.

| Shape (heads / kv) | PCC | Max Abs Err | Mean Abs Err | Relative RMS Err | Max ULP | got/true ratio median [p5, p95] |
|--------------------|-----|-------------|--------------|------------------|---------|----------------------------------|
| (1,1,1,96) 1/1 | 1.0 | 0.0 | 0.0 | 0.0 | 0 | 1.0 [1.0, 1.0] |
| (1,1,1,3072) 32/8 (Llama) | 1.0 | 0.0 | 0.0 | 0.0 | 0 | 1.0 [1.0, 1.0] |
| (1,1,1,6144) 32/8 (Qwen) | 1.0 | 0.0 | 0.0 | 0.0 | 0 | 1.0 [1.0, 1.0] |
| (1,1,1,8192) 16/8 | 1.0 | 0.0 | 0.0 | 0.0 | 0 | 1.0 [1.0, 1.0] |

**Assessment**: bit-exact. Every element is the input element, as the movement class requires, and there is no scale signature.

**Recommended tolerances**: exact. `NUMERIC_RESOLVED` grades every cell `exact` for bf16 and fp32.

A verifier probe (`probes/probe_003.py`) also confirmed the out-of-phase values bit-exact with the kernel unchanged: fp32, DRAM input, `memory_config=None`, 64 and 40 query heads.

## Verifier CLI Summary

Results: `pipeline_logs/verifier_golden/` (`verifier_report.json`), after the code-review fixes.

- supported_pass: 8
- xfail_expected: 72 (float32, DRAM, output_memory None, head_rows 64, and their combinations; all four values are queued in Refinement 1; queue check: 0 gaps)
- invalid_skipped: 0
- supported_fail: 0
- xpass_drift: 0
- xfail_wrong_mode: 0
- no_axes_found: 9 (8 rejection tests X1–X8, all pass; 1 empty loose-case parametrization, skipped)

Other tests: acceptance 4/4, `test_thread_plan.py` 25/25, precision baseline 4/4 (33 / 33 in the op's test directory).

## Recommendations

- **Queue.** Refinement 1 (generality) completes TARGET: the DRAM input regime, which is the only real kernel work, plus fp32, head_rows 64 and `memory_config=None`, which the probe shows already work. Refinement 2 (perf) raises read-issue parallelism on the traced calls, gated on reads issued by the busiest hart.
  - Only four TARGET values are missing and three are pure SUPPORTED changes, so a separate refinement for them would be far smaller than the initial build. They are bundled.
- **DRAM alignment.** The GUIDANCE names the hazard (tt-metal #43270). Refinement 1 gates the aligned path structurally. Whether Quasar silicon actually needs it is a finding for hardware, not a queue item.
- **Spreading heads over the idle cores (regime D) is unmeasured.** Rule R5 pins the op to core (0,0) and leaves 31 cores idle. For an op bound by issuing tiny reads, spreading the heads' reads over idle cores (each writing its runs into core (0,0)'s shard) doubles the NoC crossings but multiplies the issue rate. The design rejected it on traffic, and the trade-off can only be decided on a timed backend (`ttsim-perf` or hardware). It is not queued: no gate meetable here would show whether it helps.
- **No L1 pressure.** The op's worst footprint is about 96 KiB of tensor-owned shards against 4 MiB per core.
- **Spec open items** (UNRESOLVED, not defects): `special_values` (NaN/inf movement) and `PERFORMANCE`. The op moves bytes, so NaN and inf pass through unchanged by construction.
