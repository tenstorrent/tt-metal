# Verification Report: g6_indexed_fill

Scope: `g6_indexed_fill.py`, `g6_indexed_fill_program_descriptor.py`, `kernels/g6_indexed_fill_dm.cpp`,
`l1_ledger.md`, checked against `op_design.md`, `eval/prompts/g6_indexed_fill.txt` (`## Rules`) and
`eval/golden_tests/g6_indexed_fill/feature_spec.py`.

## Code Review

### Fixed

| # | Finding | Fix |
|---|---------|-----|
| 1 | **MUST violation (prompt rule "every B, K, d1, d2 and W ... MUST NOT refuse another shape on the host").** Phase 0 `SUPPORTED["alignment"] = ["h_non_aligned"]` made `validate()` raise `UnsupportedAxisValue` for every input with `W % 32 != 0` (`w_non_aligned`) or `d2 % 32 == 0` (`tile_aligned`). The kernel never branches on alignment: ROW_MAJOR pages carry no padding, reads move `align_up(chunk, A)` inside the page's aligned DRAM slot, writes move logical bytes. | `SUPPORTED["alignment"]` widened to `["h_non_aligned", "w_non_aligned", "tile_aligned"]`. No kernel change. Device evidence: `test_g6_indexed_fill_precision_baseline.py` (`w17` = `[5,1,1,17]`, 68-B pages; `d2_32` = `[3,1,32,32]`) and `probes/probe_007.py` (W ∈ {1, 17, 50, 100}, `d2 = 32` with K = 3, and the `ids_streamed` + multi-block depth-2 + 64-B row-chunk plans on odd widths with 4-B pages and 16-B tail chunks), every case bit-exact with inputs unchanged. |

Nothing else needed fixing. What I checked:

- **Kernel correctness.** Every read is barriered before its slot or window is read. `noc_async_writes_flushed` runs before a slot set is refilled. `noc_async_write_barrier` is the kernel's last act (BB-03). The ids are loaded through `volatile tt_l1_ptr` only after a read barrier (Blackhole L1 cache). `t < B` is tested unsigned before `t*R + r` is formed, and `SKIP = 0xFFFFFFFF` cannot collide with a real page because `units < 2^32` is asserted on the host. Across ids windows, a value unit's state lives in `cb_block_dest`, and later entries in later windows are checked through the `i < j0` path, so the largest `i` wins. Each result page-chunk gets exactly one writer.
- **Alignment assumptions.** The kernel needs every L1 slot to be 64-B aligned. CB bases are allocated at `get_alignment(BufferType::DRAM)` (`tt_metal/impl/program/program.cpp:2505-2509`), and every CB size and slot stride is a multiple of `A`, so this holds by the allocator contract, not by luck. The ids window bound `align_up(4E, A) + A` (when `4E % A != 0`) covers the worst-case window misalignment (checked by hand).
- **Helper usage.** No compute phase and no multicast. The design rejected each dataflow helper with a reason (`noc_async_read_page` has no size argument, so it cannot move a row chunk; the local-copy helpers are self-aimed L1→L1). The reasons hold. TensorAccessor is used throughout, with no `InterleavedAddrGen`. The kernel uses `void kernel_main()` and `api/` include paths. There is no CB credit protocol (one RISC owns every CB), so push and wait counts are both 0.
- **Host.** The descriptor cache key covers device, grid, shapes, dtypes, layouts, memory configs and `PLAN_CAPS`. A cache hit patches the four buffer addresses on every core: `test_index_rewritten_on_device_and_trace_replay` keeps every result alive and checks it, and the program-cache entry count stays flat. Nothing on the host reads a value of `batch_ids`.

### Deferred

None.

## Registry Conformance

- `INPUT_TAGGERS = {"alignment": tag_alignment}`, signature `(inputs, axes)`. W takes precedence, as the prompt specifies.
- `SUPPORTED` covers every axis the op gates on (`dtype`, `layout`, `alignment`). `EXCLUSIONS = []`.
- `validate()` order: `dim`, then ranks (before the tagger, which reads `shape[-2:]`), then SUPPORTED (`UnsupportedAxisValue`), then EXCLUSIONS (`ExcludedCell`), then `ValueError` for companion dtype/layout, shape fit, extents ≥ 1, on-device, same device, DRAM interleaved. The entry point calls it first.
- The op file declares no `INVALID`.
- **SUPPORTED now exceeds TARGET** on `alignment`: TARGET lists only `h_non_aligned`, and SUPPORTED lists all three tags. The prompt's MUST rule forces this. The golden suite has no cell for the two new tags, so they rest on the unit-test and probe evidence above. Feature-spec owner, via `/golden-tests`: add `w_non_aligned` and `tile_aligned` to `TARGET["alignment"]`, with one INPUTS shape each. `op_design.md` "Note for the lead" asks for the same change. Agents may not edit `feature_spec.py`.
- **INVALID audit** (`feature_spec.py`): `INVALID = []`, which is correct. TARGET holds only uint32 and ROW_MAJOR, so the canonical bf8b + ROW_MAJOR entry has no cell to skip (documented in the file). No cross-tensor entries. Not a norm-like op.

## Prompt Rules Audit (`eval/prompts/g6_indexed_fill.txt` `## Rules`)

| Rule (MUST) | Status |
|---|---|
| batch_ids read on device every call; no position in CT/RT args or on host | ✓. CT args hold the plan and shapes. RT args hold 4 addresses plus the unit range. The cache key holds no ids value. Shown by the in-place rewrite + trace replay tests. |
| bit-exact; no float/16-bit path; no copy landing after the value write | ✓. NoC → L1 → NoC only. Exactly one writer per page-chunk (no copy-then-overwrite). |
| entry ≥ B read unsigned, writes nothing, no byte outside the result | ✓. Guard-tensor tests with 7 and 4294967295. |
| new result; inputs unchanged | ✓ |
| ROW_MAJOR natively; no host-side transformation | ✓ |
| own kernels and program | ✓. `generic_op` with the op's own DM kernel. |
| validate raises ValueError/RuntimeError before device work; never reads an ids value | ✓. `UnsupportedAxisValue` is a `NotImplementedError`, which is a `RuntimeError`. |
| every B, K, d1, d2, W; no host refusal of a valid shape | **was violated, fixed** (Code Review #1). The remaining host refusals are `units >= 2^32` (more than 16 GiB of pages) and the L1 bound, which the caps make unreachable (≤ 1179712 B). |
| OOM on a golden INPUTS shape | n/a. Production uses 256 B per core. |
| BB-01 / 03 / 05 / 06 / 07 / 08 / 16 / 17 / 18 / 21 / 22 | ✓. One program. Final write barrier. 256 B L1. Repeat runs bit-identical. Descriptor plus program cache. `describe_plan` with roles. Trace replay. One CoreRange `(0,0)-(7,0)`. Default opt level. One zone per operation and per wait. Tests limited to production plus rule-required shapes; the two extra baseline shapes exist only to back the rule-9 fix. |

No soft (prefer / consider / avoid) rules apply.

## Design Conformance

- **Algorithm and topology.** Matches `ids_resident` / `ids_streamed`: one DM kernel on RISCV_0/NOC_0, no compute kernel. Block 0's unconditional loads are issued in the ids read's shadow under one read barrier. `select_block` then `store_block` follow the decision contract.
- **Work distribution.** `split_work_to_cores(num_cores_to_corerangeset(cores_cap, grid, True), units, True)` over the full grid (110 cores on this board). Production gets 8 units on 8 cores, one rectangle. Shapes with more units than cores spread over all 110 (probe_007: `tile_aligned_r32` on 110 cores, 2 units each).
- **Blocking-model fidelity.** `block_units`, `staging_depth`, `row_chunk_bytes`, `ids_chunk_entries` and `max_work_cores` are each produced once by `derive_plan` from `PLAN_CAPS`. Every CB size, CT arg and RT arg is derived from them, and the kernel contains no shape, block or core literal (DRY holds). No CB scales with a whole-op dimension unconditionally: `cb_batch_ids` is clamped by `ids_chunk_cap_entries`, `cb_staged_pages` by `staging_budget_bytes`.
- **Expression.** `load_block` issues every read of the block with no barrier inside. `store_block` issues every write with no barrier inside. There is one read barrier per block and one `writes_flushed` per slot reuse. The per-unit loops issue transactions only; nothing is retired per unit. This is block-expressed on both NoC halves.
- **Regime table.** Matches the code. `ids_mcast` and `gather_after_index` are deferred with deltas and a positive reason (a latency-bound op, and both add a serial hop for under 600 B).

## L1 Ledger Audit

- **Currency.** Three ledger rows match the three CBs in `_build_program`. Size expressions match: `ids_window_bytes`, `staging_depth * block_units * row_chunk_bytes` (page `row_chunk_bytes`), `align_up(4 * block_units, A)`.
- **Capacity vs live set.** No over-capacity: each CB exceeds its live set only by the alignment to `A`, a mechanism reason stated in the row. No collapsed extent: `cb_staged_pages` spans `u` and `w` and scales with both, `cb_block_dest` spans `u` and scales with it, and `cb_batch_ids` spans `j` and scales with `ids_chunk_entries`.
- **Page format vs DEST width.** n/a. No DEST and no compute kernel. UInt32 is a label on a dataflow-only payload.
- **Disjoint lifetime.** None. All three CBs are live together, and each row's justification is given.
- **Bounds.** Every capacity symbol is bounded through `min()` against a cap. `K, B, d1, d2, W` appear only through those caps. Closed-form total ≤ 1179712 B, under the 1347776 B bound.
- **Block-size defaults.** Held: spread over the full grid first, then the coarsest block that fits (the whole per-core assignment at depth 1, up to `staging_budget_bytes`).
- **Data-movement budget.** Present and consistent with the built split: ids once per work core, x and s once each (named x pages loaded and dropped), y once. The cheapest-traffic split (`gather_after_index` + `ids_mcast`, 960 B vs 1536 B read) is recorded as `deferred` with its delta and a positive reason.
- **Per-core footprint.** `ids_window_bytes + staging_depth·block_units·row_chunk_bytes + align_up(4·block_units, A)`. The first term scales with `ids_chunk_entries`; the second with `block_units`, `row_chunk_bytes` and `staging_depth`; the third with `block_units`. Production: 64 + 128 + 64 = **256 B**.

No ledger findings to fold into a refinement.

## Precision Baseline

`tests/ttnn/unit_tests/operations/g6_indexed_fill/test_g6_indexed_fill_precision_baseline.py`. The gate is bit equality. PCC and rel-RMS come from int64 → float64 views. ULP means integer steps.

| Shape | ids | tag | PCC | Max Abs Err | Mean Abs Err | Relative RMS Err | Max ULP | got/true ratio median, p5–p95 |
|-------|-----|-----|-----|-------------|--------------|------------------|---------|-------------------------------|
| `[7,1,1,32]` (production) | `[3]` | h_non_aligned | 1.0 | 0 | 0 | 0 | 0 | 1.0, 1.0–1.0 |
| `[5,2,1,64]` (rule 9) | `[4, 1]` | h_non_aligned | 1.0 | 0 | 0 | 0 | 0 | 1.0, 1.0–1.0 |
| `[5,1,1,17]` | `[4, 4294967295, 1]` | w_non_aligned | 1.0 | 0 | 0 | 0 | 0 | 1.0, 1.0–1.0 |
| `[3,1,32,32]` | `[2, 0]` | tile_aligned | 1.0 | 0 | 0 | 0 | 0 | 1.0, 1.0–1.0 |

**Assessment.** Bit-exact over the full uint32 range: `UINT32_MAX` and `2^24 + 1` appear in every batch, and the payload never passes through an engine. No scale signature.
**Recommended tolerances.** Exact, matching `helpers.TOLERANCES[uint32] = ("exact", None)`.

## Verifier CLI Summary

The golden suite was run twice. The op is graded on run B. Run A shows a harness defect that the op cannot fix.

| Category | A: `eval_test_runner.sh` default (`EVAL_CAPTURE_L1=1`) | B: `EVAL_CAPTURE_L1=0` |
|---|---|---|
| supported_pass | 7 | **9** |
| supported_fail | 2 | **0** |
| xfail_expected | 0 | 0 |
| invalid_skipped | 0 | 0 |
| xpass_drift | 0 | 0 |
| xfail_wrong_mode | 0 | 0 |
| no_axes_found (`test_regression.py`, raw op, both pass) | 2 | 2 |
| raw pytest | 9 / 11 | **11 / 11** |

Artifacts: `verifier_results/verifier_report.json` (B) and `verifier_results/verifier_report_l1_capture.json` (A).

**Run A's two `supported_fail` cells are harness defects, not op defects.** Escalate to the eval owners; this op cannot fix them.

1. `test_op_loose[...ROW_MAJOR2]` (`new_index`) fails with `TT_FATAL: Cannot load new binaries during trace capture`. `eval/l1_profiling.start_for_item` calls `device.disable_and_clear_program_cache()` around every graded test, so the capture call cannot find a cached program. Control in `probes/probe_006.py`: stock `ttnn.add` fails identically under the same condition, and both `ttnn.add` and this op capture fine once the cache is enabled.
2. `test_op_loose[...ROW_MAJOR6]` (`refusals`) fails with `L1ProfilingError: peak_total=0`. Every call in this case is refused by `validate()` before device work, which the prompt requires, so no CB is ever allocated. probe_006 shows a refused call's graph has `peak_total = 0`. Any correct implementation gets 0.

Fix on the harness side: skip L1 capture for loose cases that capture a trace or only exercise refusals (or let `feature_spec` mark them). Run A was not silenced through EXCLUSIONS or SUPPORTED. Both cells have the same axis cell as the passing ones, and silencing them would be dishonest.

## Recommendations

- **Refinement queue is empty** (see `op_requirements.md`). `TARGET − SUPPORTED = ∅`. Every golden and loose cell is at most 10 tiles (production 7 tiles, rule-9 shape 10 tiles), so `eval/prompts/perf_rules.txt` ("measure only cells of more than 32 tiles ... overrides any other instruction") forbids measuring or targeting any of them. The flagged `perf_focus` case's stated goal (kernel time under the 2.10 µs incumbent) is already met: the implementer reports 1.09 µs on 8 cores in `agent_logs/ttnn-implementer_breadcrumbs.jsonl`. I did not re-measure, per perf_rules. Trailing perf passes get the same perf_rules text and should record a null result instead of tuning a 7-tile cell.
- **Harness.** Fix the two L1-capture interactions above. Until then the default runner reports 9/11 for an op that passes 11/11.
- **TARGET widening.** Add `w_non_aligned` and `tile_aligned` to `TARGET["alignment"]` with one INPUTS shape each, so the golden suite backs the rule-9 SUPPORTED values directly.
- **Large-shape overlap (perf lamp, unmeasurable here).** When a core owns more than one unit and the assignment fits `staging_budget_bytes` (1 MiB), `derive_plan` picks one block at depth 1. The core then issues every read, waits once, and issues every write, with no read/write overlap. Catalog `double_buffer` measured that batching alone saturates a single core at ~13 GB/s because read and write cannot overlap, and that depth 2 with ~4–8 transfers per barrier lifts it to ~17.9 GB/s (~1.4×). If a production shape above 32 tiles ever enters the corpus, co-tune `staging_budget_bytes` with `max_staging_depth` (the knobs already exist; this is a knob-turn) and measure device-ns. No production shape reaches this regime today.
- **`select_block` cost** is `O(entries_in_block × K)` scalar compares on the value path. Negligible at K = 1. At `K ≈ 16384` with a value-heavy block it reaches millions of RISC-V compares. Only a concern if a large-K caller appears; the design leaves the realization to the implementer (e.g. a sorted or hashed scan).
