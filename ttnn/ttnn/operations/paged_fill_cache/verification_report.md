# Verification Report: paged_fill_cache

Target: Quasar (`quasar`, 32 clusters, 4 MiB L1 per cluster) on the ttsim simulator (no clock, so no timing
claims anywhere in this report). Specification: `eval/specs/paged_fill_cache/spec.py`
(sha256 b47723604b20…). Golden results: `/tmp/pfc_golden_v2/` (`verifier_report.json` copied beside this file).

## Code Review

Fixed in place:

1. **DRY: page-table ring entry size derived twice.** `validate()`'s ring cap (`_mechanism_caps`) and
   `create_program_artifacts` each computed `align_up(page_table.buffer_page_size(), dram_alignment)`.
   The derivation now lives in one function, `page_table_entry_bytes()` in the program-artifacts module,
   and both callers use it.
2. **DRY: the row-copy count was the literal `2` in both kernels**, separate from the host's
   `PT_RING_ENTRIES` that sizes the ring. It is now a compile-time argument, `page_table_row_copies`
   (= `PT_RING_ENTRIES`). The reader's `reserve_back`, read loop and final `push_back`, and the writer's final
   `wait_front`/`pop_front`, are counted from it, and the reader has a `static_assert(ROW_COPIES >= 2)`.
   Ring size and kernel counts now come from one constant.
3. **DRY: `TILE = 32` was defined in both the op file and the artifacts module.** The op file now imports it.
   It also defined `SKIP = -1`, which nothing used; removed.
4. `op_design.md` Knob summary row for `PT_RING_ENTRIES` updated to name the new CT arg. `l1_ledger.md`
   data-movement budget corrected (see L1 Ledger Audit).

Reviewed and found sound (no change):

- Kernel syntax `void kernel_main()`, includes `api/dataflow/...`, `TensorAccessor` on all three tensors,
  `experimental/kernel_args.h` named args. Every shape quantity is a runtime arg, so one compiled kernel pair
  serves every shape.
- **Ring balance.** `dfb_page_table_row` has ROW_COPIES pushed and ROW_COPIES popped. `dfb_input_tiles` is
  implicit sync: one tagged read per mapped tile, and exactly one tagged write drains each entry. Both
  kernels skip `-1` rows through the one shared header `paged_fill_cache_walk.hpp`, so the two walks cannot
  diverge.
- **Sticky NoC tag.** Both kernels hand it back as their last statement. The page-table reads are explicit
  and tagged `{.trid = 0}`, and the ring is listed in `disable_dfb_implicit_sync_for` on both kernels. The
  one-device-session test (another program → op → `to_layout` → op through the returned tensor) passes.
- **Helpers.** No kernel-lib helper moves an interleaved DRAM page into a dataflow buffer or scatters ring
  entries to accessor pages (design table "Helpers considered and rejected"; checked against
  `ttnn/cpp/ttnn/kernel_lib/`). The op has no compute phase and no inter-core traffic, so no compute or
  mcast helper applies.
- Broadcast efficiency: not applicable (no broadcast operand; the page-table row is looked up once per tile
  row, not per tile).

Left as is, with reason:

- `kernels/paged_fill_cache_compute.cpp` is an empty placeholder that the program never loads. The
  implementer's output list requires the file (implementer breadcrumb, friction entry). It is harmless; the
  harness should make it conditional on C ≥ 1.
- **Thread plan (2, 0, 2) is not admitted** (implementer's probe_001).
  The host admits only `VERIFIED_THREAD_PLANS = {(1, 0, 1)}`, so no shipping path can reach it. Nothing is
  queued for it: the op is data-movement bound.
  How a STRIDED DM→DM ring deals entries between several
  reader and writer threads is a question for a timed backend, if ever.

## Specification Rules

| Rule | Applies at Phase 0 | Verdict |
|---|---|---|
| R1 copy unchanged | yes | ✓ The tile ring's format is the input's dtype, moved as raw NoC payload. Bit-exact in every golden cell and on all 4 baseline shapes. |
| R2 rest of cache untouched | yes | ✓ Only mapped tile rows are written. The tail of a partly filled block is never addressed (`6x3x64x96` cells). |
| R3 in place | yes | ✓ `io_tensors[-1]` is `cache_tensor`, the entry point returns it, nothing is allocated. HOOKS["check"] passes in every cell. |
| R4 `-1` skips a block | yes | ✓ The value is read on the device. Loose `skipped_block` passes; the walk tests cover `-1` in multi-cluster splits. |
| R5 only the reached entries | yes | ✓ The kernels index `row[k]` only for `k < ceil(seq/block_size)`. Loose `unmapped_blocks` passes. |
| R6 `batch_idx` | yes | ✓ Keyword-only, default 0, an int and not a bool. X4, X5 and X11 raise; users 0 to 3 pass. |
| R7 input DRAM or L1 via `memory_axes` | gating yes; L1 is Refinement 1 | ✓ `validate()` gates on `eval.memory.memory_axes(input_tensor)`. L1 is refused for now as `xfail_expected`. |
| R8 cache / page-table placement, input dtype | yes (review rule) | ✓ `_placement_checks`: cache DRAM interleaved and TILE, page table int32 / ROW_MAJOR / DRAM interleaved, input dtype == cache dtype; each refusal is `NotImplementedError`. |
| R9 refusal outside SUPPORTED | yes | ✓ All 36 out-of-SUPPORTED cells are refused with `UnsupportedAxisValue`. |
| REJECTIONS X1 to X11 | yes | ✓ 11 / 11 raise (`ValueError` / `TypeError`). |

GUIDANCE: everything that applies is followed. The page table is read on the device, `batch_idx` is a runtime
arg, whole tiles are copied, and the split is tile-granular.

## Registry Conformance

- `INPUT_TAGGERS = AXES`, imported from the specification. The op file defines no axis function.
- `SUPPORTED` equals `PHASE0` on all five axes (`validate_contract --phase0`: PASSED, 0 errors; the 5
  UNRESOLVED items are the specification's open items, not the op's).
- `EXCLUSIONS = []`.
- `validate()` order: structural rejections (X1 to X11), then R8 placement, then SUPPORTED per axis, then
  EXCLUSIONS, then mechanism caps. It is the entry point's first statement.
- The op file declares no `INVALID`.
- XPASS auto-fixes: none needed (0 `xpass_drift`).
- **INVALID audit (specification):** `INVALID = []`. That is correct: TARGET is
  `{bf16, fp32} × TILE × {DRAM, L1} × interleaved × tile_aligned`, and all of it is buildable (12 / 12 probe
  calls bit-exact, below). With no ROW_MAJOR in TARGET and no bfloat8_b (unbuildable on Quasar, handled by
  the target page), the canonical bf8b+ROW_MAJOR entry does not arise. There is no cross-tensor coupling, and
  no norm-style weight axes.

## Design Conformance

- **Algorithm:** a whole-tile copy with a per-tile-row page-table indirection read on the device, as designed
  (regime R-filter; `-1` rows filtered at the source).
- **Work distribution:** `split_work_to_cores(grid, tiles, row_wise=True)` over the queried full grid, tile
  granular. For example, the Llama call engages 32 clusters × 8 tiles, and the 8192-wide row 32 × 8
  (worked table matches).
- **Both dataflow halves batched:** the reader issues tagged reads back to back and the writer tagged writes,
  with no per-tile barrier; DM0 posts credits per txn-id half-ring (8-entry ring = 2 × 4).
- **Thread plan:** read off the host program. It is one constant, `THREAD_PLAN = (1, 0, 1)`, passed as the CT
  args `reader_threads`/`writer_threads`; the kernels have `num_threads = 1` and there is no compute kernel.
  The design states the op as data-movement bound, with the reason: zero compute per tile.
  So one thread of each kind is acceptable. `test_paged_fill_cache_plan.py`
  reads the plan off the `ProgramSpec`, and its one-device-session test passes. No timed backend here, so the
  plan was not measured against alternatives.
- **Blocking-model fidelity:**
  - Knobs `SPLIT_UNIT_TILES`, `TILE_RING_ENTRIES`, `PT_RING_ENTRIES` and `THREAD_PLAN` are host constants,
    each defined once (now including the row-copy count, fix 2).
  - No CB scales with an op dimension. `dfb_page_table_row` scales with `blocks_per_seq`, bounded by
    `validate()`'s ring cap.
  - The per-core loop walks tiles, but the tile *is* the interleaved transfer unit and synchronization is
    grouped per half-ring, so this is not a half-turned knob and not an expression failure.
  - Axis accounting in the ledger is correct: `input_tile` streams, and the row spans `page_table_entry`.

## L1 Ledger Audit

- **Currency:** two rings (`input_tiles`, `page_table_row`) declared; two rows. Sizes match:
  `TILE_RING_ENTRIES × R × tile_bytes` and `PT_RING_ENTRIES × R × pt_entry_bytes`, with R = 1 at Phase 0.
- **Capacity vs live set:** the tile ring is a fixed 8-entry streaming window (2 × txn-id half); it spans no
  axis, correctly. The row ring spans the whole row, one copy per kernel, so capacity equals live set. Nothing
  flagged.
- **Page format vs DEST:** neither ring goes through DEST (no compute kernel). Tile pages are in the tensors'
  dtype, as R1 requires; row pages are int32. Nothing flagged.
- **Disjoint lifetime:** both rings are live for the whole kernel lifetime and concurrent, with the sharing
  cell justified.
- **Bounds:** `blocks_per_seq ≤ 131,056` (validate), `tile_bytes ∈ {2048, 4096}`, closed-form total.
- **Block-size defaults:** held. Interleaved data is spread over the full grid, and the block is the whole
  per-core range.
- **Per-core footprint:** `TILE_RING_ENTRIES × tile_bytes + PT_RING_ENTRIES × align_up(4·blocks_per_seq, 64)`
  = 16,512 B (bf16) / 32,896 B (fp32). Term 1 scales with ring depth and dtype, term 2 with `blocks_per_seq`
  only.
- **Data-movement budget — finding:** the closing line said the page-table excess (2 DRAM reads of the same
  page per engaged cluster) could not be removed without a cross-cluster handshake. That holds only for
  removing it entirely. The *second* read per cluster can come from the first copy in the cluster's own L1.
  The ledger line is corrected, and the item is **folded into Refinement 2** (perf). It was not fixed in
  place, because the DM cores keep an L1 data cache, so a RISC-side copy handed to the other hart needs a
  verified cache-maintenance sequence.

## Precision Baseline

`tests/ttnn/unit_tests/operations/paged_fill_cache/test_paged_fill_cache_precision_baseline.py`, bfloat16,
DRAM, page tables with permuted blocks; whole cache compared:

| Shape (cache ← input) | PCC | Max Abs Err | Mean Abs Err | Relative RMS Err | got/true ratio median [p5, p95] |
|-------|-----|-------------|--------------|------------------|------|
| [8,2,32,128] ← [1,2,64,128] (Qwen traced) | 1.0 | 0.0 | 0.0 | 0.0 | 1.0 [1.0, 1.0] |
| [6,3,64,96] ← [1,3,96,96] (partial block, user 2 of 3) | 1.0 | 0.0 | 0.0 | 0.0 | 1.0 [1.0, 1.0] |
| [128,8,32,64] ← [1,8,512,64] (Llama captured) | 1.0 | 0.0 | 0.0 | 0.0 | 1.0 [1.0, 1.0] |
| [2,1,32,8192] ← [1,1,32,8192] (8192 wide) | 1.0 | 0.0 | 0.0 | 0.0 | 1.0 [1.0, 1.0] |

**Assessment**: bit-exact. The op moves whole tiles and converts nothing. The ratio is exactly 1 on every
written element, so there is no scale or structural signature.
**Recommended tolerances**: exact (every cell resolves to `exact` in `NUMERIC_RESOLVED`; class `movement`).
float32 was measured bit-exact as well in the validate-bypassed probe below.

**Extra evidence for Refinement 1** (probe, `scripts/tt-probe.sh`, validate() bypassed): float32 and
bfloat16 × input in DRAM and in L1, on `[8,2,64,1024]←[1,2,160,1024]` (a `-1` entry, user 1), the Llama
call and the 8192-wide row. All 12 calls are bit-exact with no kernel change.

## Verifier CLI Summary

- supported_pass: 15 (12 matrix cells + 3 loose cases)
- xfail_expected: 36 (float32 × {DRAM, L1}: 24; bfloat16 × L1: 12), all covered by Refinement 1
- invalid_skipped: 0 (INVALID is empty)
- supported_fail: 0
- xpass_drift: 0
- xfail_wrong_mode: 0
- rejection tests (no axes): 11 / 11 passed
- Queue coverage (`--queue op_requirements.md`): 0 gaps

Tests: acceptance 4 / 4, plan / walk / session 6 / 6, precision baseline 4 / 4 (14 / 14 after the fixes).

## Recommendations

- **Queue:**
  - Refinement 1 (generality) takes SUPPORTED to TARGET. Both values are validate()-only knob-turns,
    bundled.
  - Refinement 2 (perf, structural gate) cuts the page-table DRAM reads on small calls.
  - Generality is then exhausted, so there are no further generality slots.
- **Perf lamps not queued, each with a positive reason:**
  - *Stall shadow (the page-table round trip before the first tile read):* hiding it needs speculative tile
    reads that a `-1` row would leave undrained. That forces the writer's tile ring to explicit sync.
    The trade-off is in time, not in a structural
    count, so it can only be judged on `ttsim-perf` or hardware.
- **Harness:** make `kernels/<op>_compute.cpp` conditional on the design having C ≥ 1.
- **Specification (advisory, not a defect):**
  - The page-table values are unchecked by design (R5 / DECISIONS). An out-of-range entry writes outside the
    cache, and this is documented as caller contract.
  - The uint32 cache-tile cap `blocks·H·bs_t·D_t < 2^32` is enforced in `validate()` as a mechanism cap.
