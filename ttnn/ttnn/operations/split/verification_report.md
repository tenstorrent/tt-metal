# Verification Report: split

Target: Quasar (`quasar`) on the ttsim simulator. Specification: `eval/specs/split/spec.py`. Design:
`op_design.md`. Ledger: `l1_ledger.md`. Date: 2026-11-21.

## Summary

Phase 0 is correct, exact, and its SUPPORTED block is honest: 36/36 in-phase golden cells pass bit-exactly,
228 out-of-phase cells are refused with `NotImplementedError`, and no loud category is non-zero. Review found one
latent geometry bug (fixed), one performance-conformance defect in the `tile_row_shift` reader (fixed: blocks were
clamped to one tile column, so every Phase 0 `tile_row_shift` shape read one input tile per DRAM barrier), a
missing R6 test (added), a DRY duplicate and a wrong ledger bound (both fixed). A widened-SUPPORTED exploration run
shows the built regimes already pass 211 of the 264 TARGET cells; the other 56 are exactly the two `dim = -1`
regimes the design deferred.

## Code Review

Fixed in place:

1. **TILE `dim = -1` at a tile multiple got wrong geometry (latent bug).** `select_regime` routes it to
   `page_remap`, but `derive_geometry` sent every TILE `dim != -2` to the `dim <= -3` branch (`cut_pages = W` in
   elements, `inner_pages` over the batch dims). Unreachable in Phase 0 (`validate()` refuses `dim = -1`), but it
   would have produced wrong chunks the moment `-1` entered SUPPORTED. Added the `dim == -1` branch
   (`outer = prod(shape[:-2]) * ceil(H/32)`, `cut_pages = ceil(W/32)`, `inner_pages = 1`, as the design's
   `rm_substick` row specifies). Confirmed by the exploration run: all 32 TILE `dim = -1` tile-multiple cells pass.
2. **`tile_row_shift` read one input tile per DRAM barrier: the block knob was effectively 1 on every shape it
   serves.** The reader clamped each block to the tiles left in the current tile column
   (`walk.tiles_left_in_column()`), and the design fixed `width` = 1 column per block. Every Phase 0
   `tile_row_shift` entry, the traced YOLO calls included, has 1–2 output tiles per column. So a block was 1–2
   output tiles over one input tile: the read-one/barrier trap (`master.md` `double_buffer`), against the design's
   stated intent of "one barrier per `block_h_tiles` outputs". The design's reason ("holding several columns
   removes no read") misses that extra columns remove *barriers*. I rewrote the load phase so that a block is
   `BLOCK_H_TILES` consecutive output tiles of the walk, crossing column boundaries. Input tiles are numbered in
   read order, and slot = seq % `WINDOW_TILES`. Because the walk's input keys are monotone, the last tile read is
   the only one a later tile can reuse, and a column segment of m output tiles reads ≤ m + 1 inputs. That bounds a
   block at `2 × BLOCK_H_TILES` input tiles, and `WINDOW_TILES` is now derived as `2 × BLOCK_H_TILES` (one source,
   reader `static_assert`). Structural counts (Python port of the reader's window logic, 32 cores):

   | Shape (TILE) | input tile reads (min) | DRAM read barriers before → after | reads / barrier |
   |---|---|---|---|
   | `(1,4,8400) s2` (YOLO 640×640 shape) | 270 (263) | 270 → **71** | 1.0 → 3.8 |
   | `(1,34,4096) s17` | 256 (256) | 128 → **32** | 2.0 → 8.0 |
   | `(2,40,8192) s20` (new test) | 1024 (1024) | — → 128 | 8.0 |
   | `(1,4,400) s2` (YOLO small grid) | 26 (13) | 26 → 26 | 1.0 (1 unit per core: nothing to batch) |

   The gain in time is unconfirmed on ttsim (no clock). Input reads are unchanged: still minimum + halo. The halo
   is Refinement 3's.
3. **R6 had no test.** The spec says "a sharded input is a test the op carries itself", and none existed. I added
   `test_sharded_input_is_refused_by_name` (HEIGHT and WIDTH shards built with `eval.sharding.auto_shard_config`;
   asserts `memory_axes` tags them and `validate()` raises `UnsupportedAxisValue` naming `sharding`).
4. **DRY: the chunk cap was written twice** (host `MAX_CHUNKS = 4`, writer `#if SPLIT_NUM_CHUNKS > 4`). The host now
   passes `SPLIT_MAX_CHUNKS` as a define and the writer checks against it. The literal 4 that remains is the
   number of accessor cases the writer names, which is guarded by an `#error`.
5. **The tile shape was implicit.** The `tile_row_shift` assembly addresses four 16×16 faces of a 32×32 tile
   (kernel `TILE_ROWS = 32`, `page_bytes/4`, `/64`), while the host reads the tensor's tile shape. Added
   `ASSEMBLY_TILE = (32, 32)` and a host `NotImplementedError` for any other tile in that regime, so a tiny-tile
   input cannot silently mis-assemble.
6. **`split()` ran the REJECTIONS checks twice** (once in `validate()`, once again to canonicalize `dim`). It now
   takes the canonical dim from `axes["dim"]` (`tag_dim`, R3).
7. **Wrap coverage.** Added `(2,40,8192) s20` to the window tests: 32 output tiles per core over short columns, so
   the window's read sequence wraps its 16 slots twice. DRAM and L1 both pass.

Checked and conforming (no change):

- **Specification rules.** R1: pages and face-row runs are copied byte-for-byte, the ring and window carry the
  input format, and the result is exact (precision baseline). R2: a list of `ceil(S/split)` tensors, in order
  (golden hook). R3: `tag_dim` canonicalizes before the gate. R4/R5: DRAM and L1 inputs; chunks allocated
  interleaved in the input's buffer (`_chunk_memory_config`, checked in `test_split_plan._check_chunks`). R6:
  `memory_axes` gates `sharding` (new test). R7: `UnsupportedAxisValue` / `ExcludedCell` subclass
  `NotImplementedError`. X1–X3: `TypeError` / `ValueError` / `IndexError` (3/3 rejection tests pass).
- **One dispatch per call.** One `ttnn.generic_op` writes every chunk; no `to_layout` / manipulation op anywhere.
- **Metal 2.0 only.** `ttnn.ProgramSpec`; no `ProgramDescriptor`.
- **Thread plan.** (R, C, W) = (1, 0, 1) from one host constant `THREAD_PLAN`, passed as CT args `reader_threads` /
  `writer_threads`. Kernel loops stride by them from `get_my_thread_id()`. The design states the op as
  data-movement bound with the reason (a copy; implicit sync at one thread of each kind sits on the DRAM floor,
  per the Quasar examples README). The one-thread plan is therefore acceptable, not a finding.
  `tile_row_shift` `static_assert`s R == 1 (its window is per hart). The session tests exist and pass:
  `test_after_another_program_and_before_tt_metal` runs a tt-metal program, then split in all three regime/layout
  paths in L1, then `ttnn.to_layout` on every TILE chunk, in one session; `test_split_back_to_back` alternates
  explicit and implicit programs. No timing model on this backend, so the plan was not measured against
  alternatives. The phase's reference figure on record would be `pipeline_logs/model_cycles.jsonl`, which does not
  exist for this ttsim run.
- **Ring protocols.** `page_remap` is implicit/implicit with one tagged transfer per entry. `tile_row_shift` is an
  explicit producer (opted out on the host, every transfer `{.trid = 0}`, barrier on the same id) with an implicit
  consumer. Both kernels hand the sticky tag back as their last statement.
- **Work distribution.** Full grid, `split_work_to_cores(row_wise=True)`; cores with no units get no kernel. The
  traced RM `(1,4,8400)` has 4 sticks, so it uses 4 cores. The design deferred W-segments with a positive reason
  (an interleaved page lives in one bank, so segments add no DRAM channel).
- **Helpers.** There is no compute phase. The self-aimed copies use the `Scratchpad` as the typed loopback source
  rather than `dataflow_kernel_lib::local_addr`: same transfer, bounds-checked, and the mechanism is the
  implementer's choice. No multicast pattern is warranted (no reuse-shared operand).
- **API hygiene.** `void kernel_main()`, `api/dataflow/...` includes, `TensorAccessor`, no `InterleavedAddrGen`.
  Push/pop balance: the reader posts one entry per unit and the writer drains one per unit, with both walks taken
  from the shared `split_walk.hpp` and the same RT args.
- **Blocking-model expression.** `page_remap` handshakes per page by mechanism: an implicit-sync transfer is one
  ring entry, and the page is the NoC transaction unit, so this is not a collapsed knob. `tile_row_shift` assembly
  still reserves, barriers and pushes per output tile inside a block. The design planned it that way with
  `RING_DEPTH = 2 < BLOCK_H_TILES`, and it is filed as lever 2 of Refinement 3 with a structural gate.

Deferred to refinements (structural rework beyond this pass): block-batched assembly and whole-column core
assignment (Refinement 3), and the stall-shadow prefetch (Refinement 4).

## Registry Conformance

- `INPUT_TAGGERS = AXES`, imported from `eval.specs.split.spec`; no op-local axis function.
- `SUPPORTED` names every axis the op gates on (dtype, layout, memory_config, sharding, and the five spec axes)
  and equals `PHASE0` exactly, which the refinement runner's `--phase0` lint requires.
- `EXCLUSIONS = []`. `validate()` checks the REJECTIONS first, then SUPPORTED per axis (`UnsupportedAxisValue`),
  then EXCLUSIONS (`ExcludedCell`). `split()` calls `validate()` first.
- The op file declares no `INVALID`.
- `python3 -m eval.validate_contract split --phase0 --text`: PASSED, 0 errors (5 UNRESOLVED open items in the spec,
  none blocking).
- No XPASS-based SUPPORTED auto-fix: `validate()` refuses before the kernels run, so no drift could show.
- **INVALID audit (spec).** `INVALID = []`, and the spec states "nothing is structurally impossible in this
  rectangle". That is correct: every TARGET cell is buildable (confirmed by the exploration run plus the two
  deferred regimes). The canonical bf8b + ROW_MAJOR entry is not needed because `bfloat8_b` is not in TARGET (and
  this part has no block-float path). There are no weight axes. No issue to report.

## L1 Ledger Audit

- **Currency.** Two buffers, `cb_output_pages` (ring, `RING_DEPTH × R × page_bytes`) and `scratch_input_window`
  (scratchpad, `WINDOW_TILES × page_bytes`, `tile_row_shift` only), match the code. I updated the window row for
  the new `2 × BLOCK_H_TILES` and corrected the `page_bytes` bound: RM float32 at W = 8192 is 32,768 B, not
  16,384 B.
- **Capacity vs live set.** The ring's 2 entries equal its live set (1 per side; the second is the double buffer).
  The window's 16 slots equal its maximum live set (`2 × block_h_tiles`). The axis accounting now records that the
  window spans `width` (several short columns) as well as `cut`, with capacity scaling from `BLOCK_H_TILES`, the
  one knob both draw on. No over- or under-sizing.
- **Page format vs DEST.** Not applicable: dataflow-only payloads in the input's format, no DEST.
- **Disjoint lifetimes.** None. Both buffers live for the whole program and are concurrent during assembly
  (justified in the ledger).
- **Bounds.** `page_bytes ≤ 64 KiB` (host refuses above one NoC burst); no op dimension appears in any capacity.
  Per-core total: `page_remap` `RING_DEPTH × page_bytes` (≤ 65,536 B in TARGET); `tile_row_shift`
  `(RING_DEPTH + 2 × BLOCK_H_TILES) × page_bytes` (bf16 36,864 B, fp32 73,728 B). That is far below 4 MiB, so no
  budget predicate is needed.
- **Data-movement budget.** Present and consistent with the code. **Finding:** the cheapest-traffic split (whole
  tile columns per core, no halo) is deferred on an occupancy-only reason. On the traced `[1,4,400]` the input
  crosses its tier twice (26 tile reads for 13 tiles), which is the case the occupancy argument cannot settle.
  Folded into Refinement 3, lever 1.
- **Block-size defaults.** Interleaved: the walk is spread over the full grid ✓. `BLOCK_H_TILES = 8` rather than
  the coarsest fitting block is justified by catalog measurements on WH/BH (reads-per-barrier gain saturates at
  ~4–8), not by a Quasar measurement. Refinement 3 lever 3 co-tunes it.

## Precision Baseline

`tests/ttnn/unit_tests/operations/split/test_split_precision_baseline.py`, bfloat16, both layouts × DRAM/L1, 16
cases. Worst over every chunk of each case:

| Shape (split_size, dim) | Layouts × placements | PCC | Max Abs Err | Mean Abs Err | Relative RMS Err | Max ULP | got/true ratio p5 / median / p95 |
|---|---|---|---|---|---|---|---|
| (1,4,400) (2, 1) | TILE, RM × DRAM, L1 | 1.0 | 0.0 | 0.0 | 0.0 | 0 | 1.0 / 1.0 / 1.0 |
| (1,4,8400) (2, 1) | TILE, RM × DRAM, L1 | 1.0 | 0.0 | 0.0 | 0.0 | 0 | 1.0 / 1.0 / 1.0 |
| (1,34,4096) (17, -2) | TILE, RM × DRAM, L1 | 1.0 | 0.0 | 0.0 | 0.0 | 0 | 1.0 / 1.0 / 1.0 |
| (4,128,512) (64, 1) | TILE, RM × DRAM, L1 | 1.0 | 0.0 | 0.0 | 0.0 | 0 | 1.0 / 1.0 / 1.0 |

**Assessment**: bit-exact in both regimes, both layouts and both placements; there is no scale or rounding
signature. The exploration run also showed float32 bit-exact across every regime it reached (R1's check).
**Recommended tolerances**: exact (every cell resolves to `NUMERIC_RESOLVED` `exact`); the defaults hold.

## Verifier CLI Summary

Golden run on the final code (`eval/eval_test_runner.sh eval/golden_tests/split/`, 8 sim workers, `--forked`):
39 / 268 passed (36 cells + 3 rejections), 0 failed, 0 hangs, 1 skipped (the empty `LOOSE_CASES` placeholder).

- supported_pass: 36
- xfail_expected: 228 (all covered by Refinements 1–2; queue check `--queue`: 10 TARGET−SUPPORTED values, 0 gaps)
- invalid_skipped: 0 (INVALID is empty)
- supported_fail: 0
- xpass_drift: 0
- xfail_wrong_mode: 0
- no_axes_found: 4 (the 3 rejection tests + the loose-case placeholder)

Report: `pipeline_logs/verifier_report.json` (a copy of the CLI output).

Exploration (not shipped; SUPPORTED restored to PHASE0 afterwards): SUPPORTED temporarily set to TARGET gave
211 / 264 cells passing. All 56 failures were `NotImplementedError` from the deferred `rm_substick` (44) and
`tile_col_shift` (12) regimes. Run twice, before and after the reader change, with identical results.

Op tests: `tests/ttnn/unit_tests/operations/split/`: 77 passed (acceptance 13, design/plan tests 48, precision 16).

## Recommendations

- **Queue order.** Refinement 1 (the two `dim = -1` regimes, real kernel work) → Refinement 2 (SUPPORTED widening
  to TARGET, evidence-backed, no kernel change expected) → Refinement 3 (perf: halo, batched assembly,
  `BLOCK_H_TILES` on the YOLO TILE calls) → Refinement 4 (perf: stall shadow). With no LOOSE_CASE perf-flagged, the
  perf target is the model's own calls.
- **Spec open items** (not defects, nothing blocks): `PERFORMANCE` bar / measure, `NUMERIC` special values /
  freedoms, and DEFINITION `governs` are UNRESOLVED. NaN/Inf are moved byte-for-byte by construction, so a
  `special_values` entry would pass as written.
- **The traced RM `[1,4,8400]` call** runs on 4 cores (4 sticks). The design's W-segment regime would add
  requesters but no DRAM channel; only a timed backend can say whether it pays. Not queued.
- **Target-specific constant.** `NOC_MAX_BURST_BYTES = 64 KiB` is quasar's `NOC_MAX_BURST_SIZE`, written as a
  constant with its citation because Python has no query. It is unreachable in TARGET (largest page 32 KiB), but
  it would need a query or a per-arch value if the op ran elsewhere.
- **Thread plan for the shift regimes** (design lamp: the reader does fetch *and* assembly). After Refinements
  3–4, a timed backend should check (2, 0, 2) with columns dealt to two reader harts. On ttsim it cannot be shown
  to pay, so it is not queued.
