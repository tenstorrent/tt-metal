# Verification Report: concat

Target `quasar` on the ttsim simulator (`.claude/references/targets/quasar.md`, `.claude/references/backends/ttsim.md`). Specification `eval/specs/concat/spec.py` (sha256 `7e12ffd47819…`). Phase 0 by the implementer at `7e2c33e7a82`.

## Code Review

The implementation matches the design closely; little needed fixing.

**Fixed**
- **R6 had no test.** The rule's check says a sharded tensor "is a test the op carries itself", and none existed. Added `test_sharded_refused_by_name` (`test_concat_regimes.py`): YOLOv8's untraced call shape, `[1,1,64,64] + [1,1,64,32]` height-sharded in L1 (shard height 32, two cores), in TILE bfloat16 so that `sharding` is the only axis outside SUPPORTED. The test asserts `NotImplementedError` whose message names `sharding`. It passes.
- **Output layout was hard-coded.** `_dispatch` allocated the output with `ttnn.TILE_LAYOUT`. The spec's `OUTPUT["layout"]` is `same_as: tensors`, so it now uses `first.layout`. This changes nothing in Phase 0 (TILE only), and it removes a trap for Refinement 1.
- **L1 ledger.** `L1_RESERVE_BYTES` was used in `budget_tiles` but had no row in the symbol table. Row added (see the L1 Ledger Audit).

**Reviewed, no change needed**
- **Specification rules.**
  - R1: exact byte moves, no DEST, ring format = output dtype.
  - R2: `dim` is canonicalised by the spec's `tag_dim` before the gate, and `_dispatch` uses the canonical value.
  - R3: DRAM and L1 inputs are read through accessors.
  - R4: refusals are `UnsupportedAxisValue`, a `NotImplementedError`.
  - R5: the output is `DRAM_MEMORY_CONFIG`, asserted by `test_output_is_dram_interleaved`.
  - R6: `memory_axes` is taken from every tensor, and mixed placements are refused.
  - REJECTIONS X1–X4 raise `ValueError` / `IndexError` before the support check.
  - No rule has a `structure` part. The spec has no GUIDANCE item that applies and is not followed: no compute kernel, whole-page moves where the seams allow them, a separate path for a join inside a tile, padding not treated as data, L1 inputs read through the accessor, output in DRAM.
- **Kernel API.** `kernel_main()`, `api/...` include paths, `TensorAccessor` (no `InterleavedAddrGen`). Implicit sync on the page-copy reader and the writer; explicit sync, reads tagged trid 0, on the `run_assembly` reader, whose ring is listed in `disable_dfb_implicit_sync_for` from the same `regime` value as the CT arg. Both kernels hand the trid back. Push count equals reserve count, including the ragged last block.
- **Helpers.** There is no compute phase. No multicast or semaphore handshake exists that should use the mcast helper, and no operand is reuse-shared. Raw NoC calls are the right mechanism (design, "Helpers considered and rejected").
- **Broadcast.** Not applicable.

**Deferred** (architectural, filed as refinements): the element-granular `dim=-1` issue rate (Refinement 3) and the explicit-sync reader's thread count and barrier overlap (Refinement 5).

## Registry Conformance

- `INPUT_TAGGERS = AXES` is imported from `eval.specs.concat.spec`. No op-local axis function.
- `SUPPORTED` holds all 8 axes (dtype, layout, memory_config, sharding and the four taggers), equal to `PHASE0`. The lint (`eval.validate_contract concat --phase0`) passes with 0 errors; its 5 unresolved items are the spec's own open items.
- `EXCLUSIONS = []`.
- `validate()` checks the per-axis SUPPORTED test, then EXCLUSIONS. Malformed-call checks (X1–X4) and mixed-list refusals come first, in `classify`. `concat()` calls `validate()` on its first line.
- The op file declares no `INVALID`.
- No SUPPORTED auto-fix: there was no `xpass_drift`.
- **INVALID audit (spec).** One entry, `{dtype: bfloat8_b, layout: ROW_MAJOR}`.
  - It is single-tensor (both axes describe the same list) and structural (block format × no blocks).
  - It is the canonical bf8b+RM entry.
  - concat has no weight axes, so no canonicalisation cells are needed.
  - bfloat8_b on Quasar is the target's own skip (`unbuildable on quasar`), not an INVALID entry.
  - No issue.

## Design Conformance

- **Algorithm and topology** match `op_design.md`: two regimes chosen by `select_regime` from the shapes alone, runs read straight into ring entries, and no staging buffer.
- **Work distribution.** The flat `out_tile` axis is split over the full grid (`split_work_to_cores(row_wise=True)` on `compute_with_storage_grid_size()`), so one-line shapes still fill all 32 clusters. Both dataflow halves are batched: the implicit reader and writer post one tagged transfer per entry with no per-unit barrier; the explicit reader does one reserve, one barrier and one push per block.
- **Thread plan.** Read off the host program: `THREAD_PLAN = {"reader_threads": 1, "writer_threads": 1}`, one named constant, passed as CT args, with kernel loops sized from them. That is (R, C, W) = (1, 0, 1). The design states the op is bound by data movement, with its reason, which the target page accepts for a one-thread plan. Session tests exist and pass: both regimes after a tt-metal program and before `ttnn.to_layout`, and the op twice in a row.
  - One qualification: the Quasar examples measured that **explicit** sync needs two reader harts to reach the DRAM floor. The `run_assembly` reader is explicit with R = 1. That is filed as Refinement 5, not counted as a design bug: the plan's rule for data-movement-bound ops is met, and no timing is available here.
  - There is no timed backend, so no plan comparison could be measured. `pipeline_logs/model_cycles.jsonl` does not exist for this run.
- **Blocking-model fidelity.**
  - `BLOCK_TILES` and `OUTPUT_DEPTH` are each defined once. The ring capacity (`OUTPUT_DEPTH · block`), the CT `block_tiles` and the reader's loop bound all derive from them.
  - No CB scales with an op dimension.
  - The per-core block is the block knob, not one tile. The `run_assembly` loop is reserve → all runs of `block_extent` tiles → one barrier → push, so it is block-expressed.
  - The writer's per-entry tagged write is the implicit protocol's unit: there is no per-unit completion boundary in the kernel.
  - `MAX_INPUTS = 5` appears in both host and kernel. It is a mechanism cap tied to the reader's five hand-written accessor branches, not a block knob, and the kernel's `static_assert` guards the pairing.

## L1 Ledger Audit

1. **Currency.** One ring, `cb_output_tiles`, with `entry_size = out.buffer_page_size()`, `num_entries = OUTPUT_DEPTH · block` and `data_format_metadata = out.dtype`. This matches the ledger row.
2. **Capacity vs live set.** Two blocks (fill + drain) span `out_tile`. Capacity scales with `block_tiles`. Runs from several pieces stream into one entry and none is held. No over- or under-sizing.
3. **Page format vs DEST.** Not applicable: no compute kernel, and the payload never passes through DEST.
4. **Disjoint lifetimes.** Only one buffer.
5. **Bounds.** Every symbol is bounded. **Fixed:** `L1_RESERVE_BYTES` (262,144 B), used in `budget_tiles`, had no row. Row added.

**Data-movement budget.** It is present and consistent with the code: each input byte that lands in the output crosses DRAM once, each output page once, and there is no cross-core traffic. The cheapest-traffic split is the one implemented. The `staged_shift` row is deferred with a positive reason (more bytes; `run_assembly` already serves its cells). Refinement 3 now revisits it on transaction count, not bytes.

**Block-size defaults.** Units are spread over the full grid (held). The block is not the coarsest block that fits: `BLOCK_TILES = 8` versus a core's whole share. That departure is justified by the catalog's `double_buffer` measurement and the Quasar toy's floor measurement (blocks of 4 and 8 at the floor), not by a measurement of this op. A whole-share block would serialise read against write. This cannot be measured on ttsim, and the design records it as a lamp.

**Per-core footprint.** `OUTPUT_DEPTH · block_tiles · tile_bytes` = 32,768 B at Phase 0 (bf16). It scales with `BLOCK_TILES`, `OUTPUT_DEPTH` and dtype width, and it is independent of every op dimension, of `count` and of the regime.

## Precision Baseline

`test_concat_precision_baseline.py`: bfloat16 TILE, DRAM, seed 1234. Metrics come from `comp_pcc`, `comp_allclose` and `comp_ulp(…, 0)`, plus relative RMS (RMS error / reference std) and the got/true ratio.

| Shape | Regime | PCC | Max Abs Err | Mean Abs Err | Relative RMS Err | ULP | got/true ratio (median, p5–p95) |
|-------|--------|-----|-------------|--------------|------------------|-----|------|
| `[1,1,64,64]+[1,1,64,80]` dim 3 (YOLOv8) | page_copy | 1.0 | 0 | 0 | 0 | 0 | 1.0 (1.0–1.0) |
| `[1,64,144]+[1,16,144]+[1,4,144]` dim 1 (YOLOv8) | run_assembly, row-granular | 1.0 | 0 | 0 | 0 | 0 | 1.0 (1.0–1.0) |
| `[1,2,400]+[1,2,400]` dim 1 (YOLOv8) | run_assembly, join inside a tile | 1.0 | 0 | 0 | 0 | 0 | 1.0 (1.0–1.0) |
| `(4,8,32,47)+(…,100)+(…,50)` dim -1 | run_assembly, element-granular | 1.0 | 0 | 0 | 0 | 0 | 1.0 (1.0–1.0) |
| `[1,1,32,4096]×2` dim -1 (wide) | page_copy | 1.0 | 0 | 0 | 0 | 0 | 1.0 (1.0–1.0) |
| `[2,512,1024]×2` dim 1 (tall, 2048 tiles) | page_copy | 1.0 | 0 | 0 | 0 | 0 | 1.0 (1.0–1.0) |

**Assessment.** Bit-exact on every shape (`torch.equal`), as R1 and `NUMERIC_RESOLVED` (`exact` for bfloat16) require. There is no scale signature.

**Recommended tolerances.** exact.

## Verifier CLI Summary

Run `/tmp/concat_golden_final` (copy: `verifier_report.json` in this directory). 1265 tests, 8 workers, 0 hangs.

- supported_pass: 56
- xfail_expected: 1024
- invalid_skipped: 180 (90 INVALID bf8b×RM, plus 90 bf8b×TILE `unbuildable on quasar`)
- supported_fail: 0
- xpass_drift: 0
- xfail_wrong_mode: 0
- no_axes_found: 5 (4 rejection tests X1–X4 + the default call; all pass)
- queue_gaps: 0. The 12 values of TARGET − SUPPORTED are all named in `op_requirements.md`: bfloat8_b is "Omitted" per the target page; the other 11 are in Refinements 1, 2 and 4.

Local tests: `tests/ttnn/unit_tests/operations/concat/` 81/81 (acceptance 7, regimes 68 incl. the new R6 test, precision baseline 6).

## Recommendations

- **Refinement order.** Layout (R1, tier 3), then leading dims / count / rank (R2), then perf (R3), then dtypes (R4, tier 5), then perf (R5). There are no perf-flagged `LOOSE_CASES`, so the perf targets are free-selected and already inside SUPPORTED: the anchor rule needs nothing from R1 and R2.
- **Cheap values.** count ∈ {1, 5} and rank 2 already run bit-exact on the Phase 0 kernels with `validate()` bypassed (`probes/probe_002.py`). SUPPORTED stays equal to PHASE0, as the phase-0 lint requires, so they ride in R2.
- **Structural read counts** (verifier's model of the reader's emit/merge loop; the hardware does not report them):
  - page_copy: ≈1 read per tile.
  - Row-granular: 1.7–7.3 reads per tile. YOLOv8's full-size `[1,4,8400]+[1,80,8400]` is 7.3; `[1,2,8400]×2` is 4.0.
  - Element-granular `dim=-1`: 56–101 reads per tile of 18–28 B.
  - Only the last is far from the minimum (R3).
- **Not filed (no measurable direction on this backend).**
  - The grid-synchronisation lamp: YOLOv8's small-grid joins give each of 10/13/15 clusters one tile. The reads and writes are already at the minimum count; only launch cost per cluster could change, and ttsim cannot show it.
  - `bank_stagger` (2 DRAM channels).
  - Re-examine both on a timed backend (`ttsim-perf`) or on hardware.
- **Not a hardware-checkable item here.** Every simulator count above is structural. No timing claim is made.
- **Memory pressure.** None. The ring is < 1% of L1 at every Phase 0 knob, and the clamp (`ring_cap_tiles`, `budget_tiles`) keeps it bounded for wider dtypes in R4.
