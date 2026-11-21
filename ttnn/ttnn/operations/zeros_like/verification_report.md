# Verification Report: zeros_like

Target: Quasar (`quasar`, 32 clusters) on the ttsim simulator (no clock). Specification:
`eval/specs/zeros_like/spec.py` (sha256 `102258bfa831…`). Verified at Phase 0 on 2026-11-21.

## Code Review

**Fixed**
- **R5, sharded input not refused by name.** `validate()` raised on the first axis outside SUPPORTED in dict
  order. A sharded L1 input therefore failed on `memory_config=L1` and never on `sharding`, and so was never
  "refused by name" as R5 requires (this would also have stayed true for an L1 shard after a refinement that
  forgot the order). `validate()` now names **every** axis outside SUPPORTED in one `UnsupportedAxisValue`.
  The refusal type is unchanged, so the golden xfail gate (by `isinstance`) is unaffected.
- **R5, missing test.** R5's check says the op carries the sharded-input test itself, because the golden suite
  builds interleaved tensors only. Added `test_refuses_sharded_input_by_name[height|width]`: an L1 shard built
  with `eval.memory.build/place`, then `memory_axes` checked, then refused with `sharding` in the message.
- **Ledger drift.** The `l1_ledger.md` symbol row for `pages_per_barrier` stated `min(num_pages_this_core,
  PAGES_PER_BARRIER)`. The code clamps on the host by the trid-counter cap and per block in the kernel. The row now
  states both.

**Checked, no change needed**
- **Kernel hygiene.** `void kernel_main()`, `api/dataflow/...` includes, `TensorAccessor` (no
  `InterleavedAddrGen`), no circular buffers or rings (so no push/pop to balance), every NoC write tagged
  `trid = 0` with the barrier on the same id, the iDMA zero barrier before the first write, and the tag handed
  back (`noc_async_write_set_trid(0, …)`) as the last statement. This follows the target page's ring-protocol
  rules.
- **Helper usage.** The design's rejections hold. `dataflow_kernel_lib::zero_tile` / `prepare_zero_tile`
  (`l1_helpers.hpp:53-70`) need a DataflowBuffer's tile size, which a scratchpad does not have; the kernel calls
  the same two primitives they wrap. `Noc::async_write_zeros(TensorAccessor, …)` issues untagged writes and
  `static_assert`s DRAM. There is no compute kernel and no multicast, so no compute, eltwise or mcast helper applies.
- **Broadcast efficiency.** Not applicable: there is no compute and the input is never read. The zero page is
  generated locally per core, with zero NoC bytes.
- **Evidence that the iDMA zero actually zeroes** (probe_006, now in `probes/`). With `zero_source` removed
  from the kernel and run after `toy_thread_plan` in one session, 307,200 of 307,200 elements came out non-zero
  (stale L1). The real kernel left 0. So `test_one_device_session`, which runs the same sequence, would catch a
  kernel that skips or mis-sizes the zero source. It does not merely pass on L1 that is zero from the start.

**Deferred**: nothing architectural.

### Specification rules
| Rule | Status |
|---|---|
| R1 (all +0.0, input not read) | met: no kernel binds the input; the precision baseline gives 0 ULP and no −0.0, also with NaN/±inf inputs |
| R2 (new buffer, input kept) | met: `test_input_kept_and_result_is_new`; the baseline checks that the input bits are unchanged (NaN/inf included) |
| R3 (placement = input's) | met: `allocate_tensor_on_device(tensor.spec, …)`; graded by HOOKS["check"] in every cell |
| R4 (shape/dtype/layout) | met (the cells) |
| R5 (`memory_axes`, sharded refused by name) | met **after the fix above** |
| R6 (`NotImplementedError` outside SUPPORTED) | met: 147 `xfail_expected`, 0 `xfail_wrong_mode` |

GUIDANCE (no input data, no compute, whole-tile padding, placement from the spec): all followed. No advisories.

## Registry Conformance
- `INPUT_TAGGERS = AXES`, imported from the specification and not redefined. `SUPPORTED` holds every axis
  (dtype, layout, memory_config, sharding, alignment, rank) and equals `PHASE0`. `EXCLUSIONS = []`.
  `validate()` checks SUPPORTED, then EXCLUSIONS (typed `UnsupportedAxisValue` / `ExcludedCell`), then refuses
  a zero-volume tensor (no program to dispatch). `zeros_like()` calls `validate()` first. The op file declares
  no `INVALID`.
- `python3 -m eval.validate_contract zeros_like --phase0 --text`: PASSED, 0 errors. The 5 "unresolved" entries
  are the specification's own open items (`UNRESOLVED`), not defects.
- No XPASS auto-fixes were needed (0 `xpass_drift`).
- **INVALID audit (specification)**: `INVALID = []`. That is right: TARGET has no block-float dtype (Quasar has
  none), so the canonical bf8b + ROW_MAJOR entry does not apply. There is one tensor, so there is no cross-tensor
  coupling, and no weights to canonicalize.
- **One dispatch**: one `ttnn.generic_op` of one `ProgramSpec`; host work is validation and output allocation only.

## Design Conformance
- **Algorithm and topology**: regime `fill_from_local_zero_page`, as designed. Writer-only work unit; no reader
  and no compute kernel.
- **Work distribution**: output pages are split over the full `compute_with_storage_grid_size()` grid in contiguous
  ranges (`split_work_to_cores(..., row_wise=True)`). The pre-filled tests show every cluster wrote its pages
  (`(70,512)`: ragged 16 × 2 + 16 × 1; `(1024,1024)`: 32 × 32).
- **Batching**: one trid barrier per block; for every INPUTS shape a core's whole range is one block. There is
  no per-page barrier, and the writer is the only dataflow half.
- **Thread plan (from the host program)**: `THREAD_PLAN = {"reader_threads": 0, "compute_threads": 0,
  "writer_threads": 1}`, one named constant. W goes to `KernelSpec(num_threads=W)`, to CT `writer_threads` and to
  the scratchpad size (`W · zero_src_bytes`). The kernel deals pages `t, t+W, …` from `get_my_thread_id()`. The
  design states the op as bound by data movement, with a reason: one written page per unit and no compute. That
  matches the target page's *Choosing the plan*.
  Tests: every page written at W = 1 and W = 2; the op in one
  session after `toy_thread_plan` (implicit) and before `ttnn.to_layout`; the op right after tt-metal's softmax.
  All pass. The plan was not measured against W = 2 here: ttsim has no timing model, and no
  `pipeline_logs/model_cycles.jsonl` exists for this run.
- **Blocking-model fidelity**: `PAGES_PER_BARRIER`, `ZERO_SRC_MAX_BYTES` and `THREAD_PLAN` are each defined once
  on the host. `pages_per_barrier`, `zero_src_bytes`, `page_bytes` and `writer_threads` go in as CT args, and the
  kernel derives `chunks_per_page`, `num_blocks` and the ragged extent from them. No collapsed knob: the
  scratchpad scales with `W · zero_src_bytes`, never with an op dimension, and `test_several_blocks_per_core`
  turns the block knob (1, 3). No half-turned split: the per-core loop issues the whole block before one barrier.
  `FREE_DM_HARTS = 6`, `TRID_COUNTER_CAP` and `NOC_PACKET_BYTES` restate hardware constants that have no Python
  query; each is named once with its source.
- **Expression**: the page loop sits inside one block-scoped phase with no per-page completion boundary. That
  conforms.

## L1 Ledger Audit
- **Currency**: one buffer (scratchpad `zero_page`), one row. Its size `W · zero_src_bytes` matches
  `ScratchpadSpec(size_per_node=W * zero_src_bytes)`. The symbol-table drift for `pages_per_barrier` is fixed
  (above).
- **Capacity vs live set**: capacity equals the live set (each thread's slice is read by its in-flight writes for
  the whole kernel). Axis accounting is correct: it spans `page_chunk` and streams over `output_page`. Nothing
  flagged.
- **Page format vs DEST**: raw bytes, no format metadata, never through DEST. Not applicable.
- **Disjoint lifetimes**: a single buffer, so there are no pairs.
- **Bounds**: `zero_src_bytes ≤ 65536` (clamp), `W ≤ 6`. The total is closed form: `W · zero_src_bytes ≤ 384 KiB`;
  Phase 0 uses 2048 B per core.
- **Data-movement budget**: present and consistent with the code. The input crosses DRAM 0 times, the output
  once, and nothing crosses between cores. The cheapest split is implemented. `fill_bank_runs` is `deferred`
  with a positive reason, which the perf catalog confirms: coalescing is "invalid across interleaved pages"
  (`read_coalescing`), and a whole page per write already is the coalesced unit (`tile_reorder`).
- **Block-size defaults**: held. Interleaved work is spread over the full grid, then the coarsest block is taken
  (a core's whole range is one block).

## Precision Baseline

`tests/ttnn/unit_tests/operations/zeros_like/test_zeros_like_precision_baseline.py`, bf16 TILE DRAM. Each shape
ran with a standard-normal input, and again with an input that is ⅜ NaN / +inf / −inf.

| Shape | Input | PCC | Max Abs Err | Mean Abs Err | RMS Err (absolute; constant reference) | Max ULP | −0.0 count | got/true ratio |
|-------|-------|-----|-------------|--------------|-----|-----|-----|-----|
| (70, 512) | normal / NaN+inf | 1.0 | 0.0 | 0.0 | 0.0 | 0 | 0 | n/a (reference all zero) |
| (2, 1, 100, 47) | normal / NaN+inf | 1.0 | 0.0 | 0.0 | 0.0 | 0 | 0 | n/a |
| (1, 1, 32, 8192) | normal / NaN+inf | 1.0 | 0.0 | 0.0 | 0.0 | 0 | 0 | n/a |
| (1024, 1024) | normal / NaN+inf | 1.0 | 0.0 | 0.0 | 0.0 | 0 | 0 | n/a |

**Assessment**: exact. Every element is +0.0 bitwise, whatever the input holds, and the input keeps its bits. The
scale-bug detector (got/true ratio) has no non-zero reference element to read; that is not needed here, since
the error is identically zero.
**Recommended tolerances**: exact (`NUMERIC_RESOLVED`: `exact` for bfloat16 and float32). The legacy default
`(0.9999, 0.0)` holds.

## Verifier CLI Summary
Results: `/tmp/zl_verifier_final/` (copy of the report: `verifier_report.json` beside this file).
- supported_pass: 21
- xfail_expected: 147
- invalid_skipped: 0
- supported_fail: 0
- xpass_drift: 0
- xfail_wrong_mode: 0
- no_axes_found: 1 (`test_op_loose[NOTSET]`: `LOOSE_CASES` is empty, so its parameter set is empty and it skips)
- queue_gaps: 0 (3 values in TARGET − SUPPORTED, all named in Refinement 1)

## Recommendations
- **TARGET − SUPPORTED is one refinement.** float32, ROW_MAJOR and L1 are knob-turns on the built regime. With
  SUPPORTED widened to TARGET (experiment, reverted) the golden suite passed 168 / 168 with no kernel change. Since
  a fresh simulator buffer may already be zero, Refinement 1's work is the pre-filled coverage proof per value.
  Splitting it into a dtype refinement and a layout refinement would make each ~10× smaller than the Phase 0
  build, so it is bundled. With only one generality refinement, the cadence's perf slot follows it directly.
- **Perf headroom on this backend is structural only.** `/perf-ceiling-dm` cannot give a ceiling: the NoC
  estimator's `noc_latencies.yaml` has no Quasar entries. Levers considered:
  - **Bank stagger**: filed as Refinement 2, with a strong structural signal on L1 outputs (peak same-bank
    fan-in 32 → 1).
  - **Writer threads W = 2**: not filed.
    Tested correct (`test_every_page_written[W2]`)
    and kept as a one-constant lamp for a timed run.
  - **Grid participation (`MIN_PAGES_PER_CORE`) for small outputs**: not filed. Its effect (launch cost against
    fewer writers) has no structural count that improves monotonically; measure it on `ttsim-perf` or hardware.
    Shapes: `(32,17)` (1 page), traced `(70,512)` (48 pages).
  - **Placement (`row_wise`)**: already `True`. Quasar's 2 DRAM banks and the topology differ from the WH
    measurement; also a timed-backend question.
- **Finding for another backend**: the gain of Refinement 2 and every lamp above is unconfirmed until a run on
  `ttsim-perf` or hardware measures it.
- **L1 / memory pressure**: none. The footprint is ≤ 64 KiB per thread whatever the shape.
