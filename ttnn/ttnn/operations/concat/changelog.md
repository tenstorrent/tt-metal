# Changelog: concat

## Phase 0 — Core Implementation
- **Date**: 2026-11-21
- **What was done**: Initial implementation via incremental pipeline (planner → implementer → verifier). Metal 2.0 `ttnn.ProgramSpec`, reader + writer only, one ring, two regimes (`page_copy`, `run_assembly`), full-grid split of the flat output-tile axis.
- **SUPPORTED at Phase 0**: dtype=[bfloat16], layout=[TILE], memory_config=[DRAM, L1], sharding=[interleaved], dim=[-1, -2], count=[2, 3], alignment=[tile_aligned, w_non_aligned, h_non_aligned], rank=[3, 4] (= PHASE0)
- **Accuracy achieved**: PCC=1.000, max_abs_err=0, rms_err=0, ULP=0 — bit-exact (measured on 6 shapes covering both regimes via test_concat_precision_baseline.py)
- **Golden suite at Phase 0**: 56 / 56 in-SUPPORTED cells passing; 1024 xfail_expected, 180 skipped (INVALID + bf8b unbuildable on quasar); 0 supported_fail / xpass_drift / xfail_wrong_mode (per `verifier_report.json`)
- **Issues encountered** (verifier fixes): added the missing R6 test (a height-sharded L1 input is refused by name, `test_sharded_refused_by_name`); output layout now follows the inputs (`first.layout`, spec OUTPUT) instead of a hard-coded TILE; `L1_RESERVE_BYTES` row added to the ledger's symbol table. No drift fixes.
- **Tests added**: test_concat.py (generated acceptance), test_concat_regimes.py (implementer; + R6 test by the verifier), test_concat_precision_baseline.py (verifier)
- **Structural counts at Phase 0** (simulator; no timing claim): NoC reads per output tile ≈1 in `page_copy`; 1.7–7.3 in row-granular `run_assembly`; 56–101 (18–28 B each) in element-granular `dim=-1` `run_assembly` — 100.7 on `(4,8,32,47)+(…,100)+(…,50)` (Refinement 3's baseline). Reader harts on `run_assembly`: 1 (Refinement 5's baseline).

### Implementer notes (Phase 0)


Built from op_design.md: Metal 2.0 `ttnn.ProgramSpec`, reader + writer only (thread plan (R, C, W) = (1, 0, 1),
host constant `THREAD_PLAN`), one ring `cb_output_tiles`, flat `out_tile` axis split over the whole grid in contiguous
ranges. Regimes `page_copy` (implicit-sync whole-page reads) and `run_assembly` (explicit sync, sub-page reads tagged
trid 0 straight into ring entries, one barrier per block), selected by `select_regime` from the shapes alone.

Knobs, each defined once in `concat_program_artifacts.py`: `BLOCK_TILES = 8` (CT `block_tiles`, clamped by the ring
cap, the L1 budget and the largest core share), `OUTPUT_DEPTH = 2` (ring = `OUTPUT_DEPTH · block_tiles` entries),
`THREAD_PLAN`. `concat_with_block_tiles` turns the block knob for tests.

Structural counts (simulator: no timing claim): 1 read + 1 write per output tile in `page_copy`; in `run_assembly`
1 reserve / 1 barrier / 1 push per block of `block_tiles` tiles, writes 1 per tile, reads merged to whole faces or
whole tiles wherever source and destination are contiguous (an aligned tile is one 2,048 B read).

Deviations from the design (advisory parts):
- `run_assembly` skips cross-axis tile padding instead of reading it into output padding (fewer bytes and transactions).
- `page_copy` finds a tile's piece by a scan over the ≤ 5 pieces per tile rather than an incremental pointer.

Tests: acceptance 7/7; `test_concat_regimes.py` 67/67; golden suite 60 passed, 0 failed, 0 hangs.

## Refinement 1 — ROW_MAJOR layout (native stick kernels)
- Date: 2026-11-21
- What was done: `ttnn.ROW_MAJOR_LAYOUT` added to `SUPPORTED["layout"]` with a native stick path (page = stick), the
  design's `row_major_runs` regime, built as two host regimes chosen by `select_regime(shapes, dim, layout)`:
  - `row_major_copy` (dim -2): each output stick is one whole input stick — implicit-sync tagged read per entry, exactly
    the `page_copy` loop with a row → (piece, local row) map; round-robin over the plan's reader threads; trid hand-back.
  - `row_major_runs` (dim -1): output stick q is input stick q of every piece side by side — per block of sticks:
    `reserve_back`, `count` sub-page reads per stick (`ext_k · elem_bytes` B into the entry at `off_k · elem_bytes`, any
    byte alignment), all tagged trid 0, one barrier, `push_back`. Ring opted out of implicit sync (`EXPLICIT_REGIMES`).
  - Ring entry = one output page (`page_geometry`): a tile, or the output stick rounded up to `ttnn.get_l1_alignment()`
    (asserted ≤ the output's aligned page; every input's aligned page is at least that too). The flat output-page axis
    (now sticks) is split over the whole grid exactly as before. The block knob counts pages; its ROW_MAJOR default
    (`default_block_pages`) is as many sticks as `BLOCK_TILES` tiles hold bytes (single source `BLOCK_TILES`), clamped
    by the ring cap / L1 budget / core share as before — a 16 KiB stick (`[1,32,4096]×2` along W) gives a block of one.
  - Reader CT arg `tile_bytes` renamed `page_bytes`, plus `elem_bytes` (from `out.element_size()`; the TILE path used
    `tile_bytes / 1024`, same value) — the later dtype refinement needs no further plumbing for element width.
  - Reused: the reader's accessor set, `with_piece`, the explicit-sync block loop shape, the writer (unchanged: one
    implicit write per entry is layout-agnostic), the ring, the split, `resolve_block_tiles`. Added: two `if constexpr`
    regime branches in the reader, `page_geometry` / `default_block_pages` on the host.
- Accuracy achieved: bit-exact (PCC=1.0, relative RMS=0; `torch.equal`) on every shape tested, among them
  `(1,1,32,50)+(1,1,32,14)` d-1, `(4,8,32,47)+(…,100)+(…,50)` d-1, `(1,32,4096)×2` d-1, `(1,64,144)+(1,16,144)+(1,4,144)` d-2,
  `(1,1,17,64)+(1,1,15,64)` d-2, DRAM and L1 inputs.
- Golden test progress: targeted slice — 58/58 in-SUPPORTED ROW_MAJOR cells (bf16, dim -1/-2, count 2/3, rank 3/4, every
  alignment, DRAM and L1) pass, plus 4/4 TILE L1 representatives; the full suite is the harness's run.
- Structural counts (simulator; no timing claim): `row_major_copy` 1 read + 1 write per output stick; `row_major_runs`
  `count` reads + 1 write per output stick, 1 barrier per block (default 128 sticks of 128 B for `(1,1,32,50)+(…,14)`;
  32 sticks per cluster on `(4,8,32,47)+(…,100)+(…,50)` → one block per cluster).
- Issues encountered: None — the first run of the RM path was green.
- Tests added (test_concat_regimes.py): `test_regime_selection_row_major`, `test_pinned_row_major` (DRAM, L1),
  `test_ragged_last_block_row_major` (block of 1, 3, 7 sticks and the default), `test_wide_stick_row_major`,
  `test_concat_layout_matrix` (TILE→TILE and RM→RM × 8 shapes × dim -1/-2), `test_row_major_output_is_dram_interleaved`,
  `test_mixed_layouts_refused`; `test_programs_share_a_device_session` now runs both RM regimes interleaved with TILE
  calls, after a tt-metal program and before `to_layout(out, ROW_MAJOR)` / `to_layout(out, TILE)`.
  `test_row_major_refused` removed (ROW_MAJOR is supported now).

## Refinement 10 — Height-sharded tensors in L1 (the YOLOv8 neck join)
- Date: 2026-11-21
- What was done: `"height"` added to `SUPPORTED["sharding"]` — every tensor of the list height-sharded in L1, each on
  a shard grid of its own, in TILE and ROW_MAJOR, at every supported value of the other axes. The output stays DRAM
  interleaved (R5); DRAM × height is the specification's INVALID (no EXCLUSION needed).
  - Operator's question answered: the sibling-op hint holds for this op. The readers reach every input page through
    its `TensorAccessor` (bound from the tensor's spec), which maps a logical page id to the cluster that holds its
    shard; a probe with `validate()` bypassed (`probes/`, 18 shapes × both layouts, inputs placed by
    `eval.memory.place` exactly as the golden runner places them) was bit-exact in all four regimes before any
    change, including tensors of one call on different shard grids (32/16/4, 8/24/8, 2/1/1 cores), one-core grids
    and 25-cluster grids. So no kernel or host-program change was made; the op does not depend on how many cores a
    shard grid spans (nothing in the host program reads the shard spec).
  - Data path kept, as the entry asks: a sharded input is read over the NoC like an L1-interleaved one, not consumed
    in place; the output split (contiguous output pages over the whole grid) follows no input's shard grid. Native
    per-shard reading remains a perf refinement of its own (ledger: data-movement budget, placement note).
  - Reused: the whole kernel set, the program artifacts, `classify()` (memory_axes per tensor already compared
    `(buffer, sharding)` only, so different shard grids pass as one placement). Added: the SUPPORTED value, tests,
    ledger and docstring notes.
- Accuracy achieved: bit-exact (PCC=1.0, relative RMS=0; `torch.equal`) on every sharded shape run, among them the
  neck join `[1,1,64,64]+[1,1,64,32]` d-1 RM on two cores and one, `(1,64,144)+(1,16,144)+(1,4,144)` d-2,
  `(1,2,400)×2` d-2, `(1,1,32,50)+(1,1,32,14)` d-1, `(4,8,32,47)+(…,100)+(…,50)` d-1, both layouts.
- Golden test progress: targeted slice — 58/58 in-SUPPORTED `sharding=height` cells (bf16, L1, dim -1/-2, count 2/3,
  rank 3/4, every alignment, TILE and ROW_MAJOR) pass, plus 4/4 interleaved L1 representatives; the full suite is the
  harness's run. `verify_supported --queue`: 0 gaps.
- Issues encountered: `ttnn.to_layout` (tt-metal) refuses to tilize a row-major shard narrower than a tile ("Physical
  shard shape (1, 64) must be tile sized") — a limit of the tt-metal program the session test runs before ours, not
  of concat; the session test runs that program on an interleaved copy for its sharded steps.
- Tests added (test_concat_regimes.py): `test_model_neck_join_height_sharded` (the model's call, RM, two cores and
  one: values, shard specs, `[1,1,64,96]` DRAM interleaved), `test_height_sharded` (7 entries × TILE/RM, runner
  placement, every regime); `test_programs_share_a_device_session` gains four sharded calls (page_copy,
  row_major_copy, row_major_runs, run_assembly) between tt-metal programs; `test_sharded_refused_by_name` now refuses
  a width-sharded call (R6 by name; height is supported).

## Refinement 20 — `memory_config`, `sub_core_grids`, four and nine tensors (Llama, Qwen)
- Date: 2026-11-21
- What was done:
  - Signature equals the specification's stub: `concat(tensors, dim=0, *, memory_config=None, sub_core_grids=None)`.
  - `memory_config` → axis `output_memory` through the specification's own `canonical_memory` (CALL["canonical"],
    imported, not copied): None for no argument or the DRAM interleaved config, "L1" for the L1 interleaved config, a
    sharded config reads "L1/height" and is refused by name (UnsupportedAxisValue). SUPPORTED["output_memory"] =
    [None, "L1"]; the output is allocated DRAM or L1 interleaved accordingly (`OUTPUT_MEMORY_CONFIGS`, R5).
  - `sub_core_grids`: None is the one accepted value; anything else is refused by name.
  - SUPPORTED["count"] gains 4 and 9. The reader's accessor list grew from `in_0 … in_4` to `in_0 … in_8`
    (`MAX_INPUTS` 5 → 9, TARGET's largest count); the host passes `MAX_INPUTS` as a CT arg and the reader
    static-asserts it equals its binding list, so the two cannot drift. Only bound inputs are instantiated.
  - Reused: every regime, the ring, the split, the writer (a `TensorAccessor` of an L1-interleaved output needs no
    change). Added: the keywords, the axis, 4 more accessor bindings, tests, ledger bounds for Qwen's join.
- Accuracy achieved: bit-exact (PCC=1.0, relative RMS=0, `torch.equal`) on Llama `8×[1,1,32,512]+[1,1,32,256]` d-1
  (TILE and RM, L1 result), Qwen `4×[1,1,32,37984]` d3 (TILE and RM, DRAM and L1 results), nine element-granular
  pieces d-1 (run_assembly, row_major_runs) and four row-joined pieces d-2 (run_assembly, row_major_copy).
- Golden test progress: targeted slice — 196/198 in-SUPPORTED cells with `count` 4/9 or `output_memory=L1` pass
  (2 skipped by the runner: Qwen TILE height-sharded, four 2,430,976 B shards on one core do not fit L1); 3 cells
  outside SUPPORTED xfail as expected; 8/8 random prior representatives pass. Full suite is the harness's run.
- Issues encountered: None. Large-join check (operator note): Qwen RM uses one 303,872 B output stick per ring
  entry, ring 607,744 B < 1,048,560 B cap; worst cluster ≈ 1.2 MB of L1 with L1 inputs and output (l1_ledger.md).
  Recorded bound: a ROW_MAJOR output stick wider than 524,280 B would overflow the ring cap at block 1 — no INPUTS
  entry nears it.
- Tests added (test_concat_regimes.py): `test_model_call_llama` (L1 interleaved result asserted),
  `test_model_call_qwen` (`sub_core_grids=None`, DRAM interleaved result asserted), `test_many_pieces` (count 4/9 in
  all four regimes × memory_config None/DRAM/L1, regime asserted), `test_sharded_output_config_refused_by_name`,
  `test_sub_core_grids_refused_unless_none`; `test_programs_share_a_device_session` gains three nine/four-piece calls
  with L1 results (run_assembly, row_major_runs) between tt-metal programs.
