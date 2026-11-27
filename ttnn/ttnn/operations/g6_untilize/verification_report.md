# Verification Report: g6_untilize

Regime `row_gather`: one 8×4 rectangle (32 cores, 1 output row each on the production case). On each core
the reader (NCRISC/NoC0) and the writer (BRISC/NoC1) gather the row's 64-B face-row pieces from the TILE
input straight into row-major positions of `cb_row_block`. The writer then stores the row as one ROW_MAJOR
page with `dataflow_kernel_lib::write_sticks_after_untilize`. There is one `generic_op` dispatch and no
compute kernel.

## Code Review

### Fixed in this pass

1. **Collapsed knobs in `derive_block_tile_cols` (blocking-model fidelity).** The function took
   `depth=ROW_BLOCK_DEPTH, budget=L1_OP_BUDGET_BYTES` as default arguments, and `_plan` called it without
   them. Python binds defaults when the module is imported, so a tuned `ROW_BLOCK_DEPTH` or
   `L1_OP_BUDGET_BYTES` never reached the block-size derivation. In a probe, `(1,1,32,1024)` at a budget of
   24576 B kept `block_tile_cols = 32` (it should be 2). The fix makes both parameters required, and `_plan`
   now passes the module knobs at call time. The same probe now gives 32 → 2 → 32 as the budget is changed
   and restored, bit-exact each time.
2. **Incomplete descriptor cache key (BB-07).** `L1_OP_BUDGET_BYTES` shapes the program (CB size and the
   `block_tile_cols` CT arg) but was missing from the key. `_knob_key()` is now the single list of the
   module knobs that shape the program: `CORE_LIMIT`, `ROW_BLOCK_DEPTH`, `BANK_STAGGER`, `GATHER_RISCS`,
   `L1_OP_BUDGET_BYTES`. It is read at call time. In the probe, a budget change builds one new program and
   restoring the budget hits the cached one (0 new programs).
3. **Non-32×32 input tiles refused (kept from an interrupted earlier verifier pass of this run; reviewed).**
   The kernels take tile and face geometry from the CB, which the host formats as the default 32×32 tile.
   A 16×32-tile input has 2048-B pages. Where the host and CB tile counts disagree, the kernel
   `static_assert`s fail at JIT time (a RuntimeError). Where they agree (e.g. H ≤ 16), every piece would be
   read at a 32×32 offset inside a 16×32 page and the output would be silently wrong. `validate()` now
   raises `ValueError` for any TILE input whose tile is not 32×32 (re-probed: `ValueError ... got (16, 32)`).
   This is input validation, not an axis: TARGET and the spec define TILE as 32×32.
4. **Kernel guards (same earlier pass; reviewed).** The DRAM-alignment `static_assert` now uses
   `NOC_DRAM_READ_ALIGNMENT_BYTES` instead of a literal 64, so it is correct on both Wormhole and Blackhole.
   Two `static_assert`s pin the host-derived tile counts (`tensor_tile_cols`, `tile_rows_per_image`) to the
   CB's tile geometry.
5. **BB-21 zone split (same earlier pass; reviewed).** The reader's `cb_reserve_back` used to sit inside
   `G6U-gather-issue`. It now has its own zone, `G6U-gather-reserve`, so the only blocking wait on the
   reader side before the barrier is attributed separately.
6. **`l1_ledger.md` currency.** The producer column now names the writer's gather share. The
   `block_tile_cols` bound now matches the code (the fit is checked at `ROW_BLOCK_DEPTH`, not at the
   clamped `buffer_depth`). The footprint section notes that capacity follows the helper's 32-row
   quantum, not `core_rows` (see L1 Ledger Audit).

### Reviewed, no change needed

- **CB sync.** Each block: the reader reserves and pushes `block_tile_cols` pages, and the writer waits
  and pops `width_in_tiles = ceil(chunk_row_bytes/128) = block_tile_cols` pages. That holds for the last
  col block with a partial tile too, because `block_tile_cols` divides `tensor_tile_cols`. The writer's
  explicit `cb_wait_front` is a zone-only duplicate of the helper's wait, with the same count.
- **Writer-side gather into `get_read_ptr` before `cb_wait_front`.** This is race-free. Slot `k mod depth`
  previously held block `k − depth`, which the writer has already written out (write barrier) and popped.
  The reader cannot reserve block `k + depth` into that slot until the writer pops block `k`. The reader and
  writer fill disjoint 128-B tile-column bands. The writer stores only after its own read barrier and the
  reader's push, which comes after the reader's barrier (BB-06). The probe exercised this at depth 2 with
  3 blocks per core (`(1,1,96,136)`, `CORE_LIMIT=1`).
- **BB-03.** Read barrier per block on both RISCs. Write barrier per block inside the helper.
- **API usage.** `TensorAccessor` (no `InterleavedAddrGen`), `void kernel_main()`, and
  `api/dataflow/dataflow_api.h` includes.
- **Helpers.** The store uses `write_sticks_after_untilize`. No kernel-lib helper gathers face rows of
  TILE pages into sticks; the design records this gap (`gather_sticks_from_tiles`). There is no compute
  phase to fuse, no broadcast, and no multicast.
- **Blocking-model fidelity, DRY.** `block_tile_cols`, `buffer_depth`, `CORE_LIMIT`, `BANK_STAGGER` and
  `GATHER_RISCS` are each set once on the host. They reach the kernels as CT args, and the CB size and
  every loop bound derive from them. The row-block cap comes from the CB tile height. The per-core row
  formula exists twice, as the host `_core_rows` (for `describe_plan`) and the kernel `core_rows()`. They
  use the same CT inputs, and any drift fails every value test. The 11-entry shared CT-arg prefix is the
  only host↔kernel positional contract (`next_ct_arg = 11`).
- **Expression.** One reserve/push, one read barrier, one wait/pop and one write barrier per block. No
  per-row or per-piece handshake on either RISC. The per-core loop runs whole blocks
  (`block_rows = min(core_rows − 32k, 32)`): the size half of the core-split knob is turned, not just the
  count.
- **Zones (BB-21).** Every blocking wait and issue phase is in a named zone. The few-instruction prologue
  (core index, `TensorAccessor` construction) sits outside a zone, inside the implicit per-RISC kernel
  zone. Accepted.

## Registry Conformance

- `INPUT_TAGGERS = {dtype: tag_dtype, layout: tag_layout}`, both with the signature `(inputs, axes)`.
  `SUPPORTED = {dtype: [uint32], layout: [TILE]}`. `EXCLUSIONS = []`. `validate()` runs the spec's
  ValueError checks, then the SUPPORTED per-axis check (`UnsupportedAxisValue`), then EXCLUSIONS
  (`ExcludedCell`). The entry point calls `validate()` on its first line. The op file declares no
  `INVALID`.
- There is no alignment, rank or size axis and no extra tagger, as the spec rule requires.
- No auto-fixes to SUPPORTED: the CLI reports no `xpass_drift`.
- **INVALID audit** (`eval/golden_tests/g6_untilize/feature_spec.py`): `INVALID = []`, which is correct.
  TARGET holds one dtype (uint32) and one input layout (TILE), so there is no bf8b+ROW_MAJOR cell to
  canonicalize. The op is not norm-like and has no weights. No cross-tensor entries.

## Prompt Rules Audit (`eval/prompts/g6_untilize.txt` § Rules)

Every MUST rule applies at Phase 0. All hold:

| Rule | Status |
|------|--------|
| Bit-identical, no path narrower than 32 bits | NoC byte moves DRAM→L1→DRAM only. No compute kernel, no RISC-V load/store of values. Bit-exact on all four precision distributions |
| New storage, input unchanged | Output allocated per call. `test_new_storage_input_unchanged` passes |
| Own program, no delegated layout op | One `generic_op` of this op's descriptor |
| Each tensor crosses DRAM at most once | Each 64-B piece is read once by its owning core; each row page is written once (ledger budget) |
| ROW_MAJOR produced natively | No host-side conversion op |
| Mesh: every device written from its own input | `generic_op` wraps the descriptor as an SPMD mesh program. **Not exercised on this single-chip slot** (finding below) |
| Keyword refusal | `ValueError` for `use_multicore` ≠ True and `sub_core_grids` ≠ None (acceptance test) |
| OOM never refused or excluded | No OOM. CB ≤ 16 KB on the production case |
| No extra axis or tagger, no production EXCLUSIONS | Holds |
| Core count = blocking decision | `choose_core_rectangle(tensor_rows, grid, CORE_LIMIT)` |
| BB-01/02/03/05/06/07/08/16/17/18/22 | Hold. BB-07 needed fix 2 above |
| BB-21 | Holds (see Zones) |

The prompt has no soft (prefer/consider) rules.

## Design Conformance (`op_design.md`)

- **Algorithm and topology.** `row_gather` as designed: gather on read, whole-page writes, no combine.
- **Work distribution.** One 8×4 rectangle at logical (0,0), contiguous balanced row ranges derived
  in-kernel, and common runtime args only. The production case has 32 row units, and the units, not the
  110-core grid (11×10, `TT_METAL_CORE_GRID_OVERRIDE_TODEPRECATE` on this board), bind the core count. That
  is the design's choice; splitting `tile_cols` across cores is the deferred grid-sync lamp.
- **Perf lamps adopted, each measured by the implementer** (`agent_logs/ttnn-implementer_breadcrumbs.jsonl`):
  - read-with-state issue form: 2380 → 2389 ns, a null result inside noise. It is the design's lamp form
    and was kept.
  - `bank_stagger`: 2389 → 1910 ns.
  - split gather across both RISCs and NoCs (`GATHER_RISCS = 2`): 1910 → 1601 ns.
  All three are switches that default to the measured setting, and each was probed bit-exact at its other
  setting.
- **Cosmetic deviations.** The kernel file names differ from the Host Contract
  (`g6_untilize_{reader,writer}.cpp` plus a shared `g6_untilize_gather.hpp`). The writer's role string in
  `describe_plan` is `gather_row_block+store_row_block`. Mechanism is not binding.

## L1 Ledger Audit

- **Ledger currency.** There is one CB, `cb_row_block`. Its size is
  `block_tile_cols · buffer_depth · tile_bytes`, which is exactly what `_build_program_descriptor`
  allocates (production 16384 B). The page is one 4096-B uint32 tile, and the producer/consumer columns
  match the code (after fix 6).
- **Capacity vs live set.** *Over:* 16384 B allocated against 512 B live on the production case. The stated
  mechanism reason holds: the store helper's wait/pop quantum is `width_in_tiles` tile pages per ≤ 32-row
  block, so a slot holds 32 rows whatever `core_rows` is. *Under:* rows are spanned up to the 32-row cap
  and capacity is sized at the cap. Tile columns are spanned and capacity scales with `block_tile_cols`.
  Blocks per core stream through the `buffer_depth` window. No collapsed extent. Disposition: left
  as-is. 16 KB of 1347776 B is not worth replacing the helper on the production case. The large-W note is
  under Recommendations.
- **Page format vs DEST width.** UInt32 dataflow-only payload, never unpacked and never in DEST. The audit
  does not apply.
- **Disjoint lifetimes.** Single buffer; nothing to share.
- **Bounds.** Every symbol has a bound and a predicate. The total `buffer_depth · block_tile_cols · 4096`
  is closed-form, and `ROW_BLOCK_DEPTH · block_tile_cols · 4096 ≤ L1_OP_BUDGET_BYTES` holds by derivation
  for every shape. No refusal path is needed.
- **Data-movement budget.** Present and consistent with the code. x is 1 read (256 × 64 B, 4 per RISC per
  core) and out is 1 write (32 × 512 B). No cross-core traffic and no L1 copies. The cheapest-traffic split
  (`rows` only) is the one implemented. The equal-traffic mirror `tile_scatter` is a `deferred` row with a
  positive reason: identical bytes and transactions, it writes pages in 64-B pieces from 8 cores, and it
  has no store helper.
- **Block-size defaults.** Interleaved: the work units are spread across the grid (32 rows → 32 cores; the
  grid has 110), then the coarsest block is taken (the whole per-core assignment).
- **Per-core footprint.** `L1_op = buffer_depth · block_tile_cols · tile_bytes`. Linear in
  `block_tile_cols` and `buffer_depth`, constant in `block_rows`. Production: 16384 B per core on 32 cores.

## Precision Baseline

`tests/ttnn/unit_tests/operations/g6_untilize/test_g6_untilize_precision_baseline.py`. BB-22 limits tests
to the production case, so the baseline varies the **input distribution**, not the shape. Each
distribution targets a different narrow-path failure. Metrics are computed on the unsigned values in
float64 (exact below 2^53); the gate is bit equality.

| Shape | Distribution | Mismatches | Max ULP | PCC | Max Abs Err | Mean Abs Err | Relative RMS Err | got/true ratio median, p95/p5 |
|-------|--------------|-----------|---------|-----|-------------|--------------|------------------|------------------------------|
| (1,1,32,128) | full range 0 … 2^32−1 | 0/4096 | 0 | 1.000000 | 0 | 0 | 0 | 1.0, 1.0 |
| (1,1,32,128) | odd values above 2^24 | 0/4096 | 0 | 1.000000 | 0 | 0 | 0 | 1.0, 1.0 |
| (1,1,32,128) | sign bit set | 0/4096 | 0 | 1.000000 | 0 | 0 | 0 | 1.0, 1.0 |
| (1,1,32,128) | index-coded | 0/4096 | 0 | 1.000000 | 0 | 0 | 0 | 1.0, 1.0 |

**Assessment.** Bit-exact. The op moves bytes and nothing else. No scale or structural signature.
**Recommended tolerances.** Exact (`helpers.TOLERANCES[uint32] = ("exact", None)`); the defaults are not
used.

**Robustness probe (exploratory, not a committed test; BB-22).** SUPPORTED has no shape axis, so every
rank-4 uint32 TILE DRAM tensor is claimed. One probe ran 15 configurations, all bit-exact:

- ragged shapes `(2,3,40,100)`, `(1,2,17,33)`, `(1,1,1,16)`
- many rows per core `(3,1,96,4096)` (110 cores)
- several col blocks at depth 2 `(1,1,64,5440)` (`block_tile_cols = 85`)
- every knob at its non-default setting: `GATHER_RISCS=1`, `BANK_STAGGER=False`, `CORE_LIMIT ∈ {1,2,3,4}`,
  `ROW_BLOCK_DEPTH=1`, a reduced `L1_OP_BUDGET_BYTES`

## Verifier CLI Summary

`ttnn/ttnn/operations/g6_untilize/verifier_results/verifier_report.json` (golden: 6/6 passed, 0 hangs)

- supported_pass: 2 (`test_op`, `test_op_loose`: the one cell, uint32 × TILE)
- xfail_expected: 0
- invalid_skipped: 0
- supported_fail: 0
- xpass_drift: 0
- xfail_wrong_mode: 0
- no_axes_found: 4 (`test_regression.py`: face order, extremes, new storage, repeat; all passed, untagged
  by design)

Infra note: `python3 -m eval.verify_supported` needs `.tt_chip_scope` sourced. That file sets the
`LD_PRELOAD` MPIX stub; without it `import ttnn` fails with `undefined symbol: MPIX_Comm_shrink`.
`run_safe_pytest.sh` and `eval_test_runner.sh` source it themselves.

## Refinement Queue Status

- **Generality: none.** `TARGET − SUPPORTED = ∅` on both axes (dtype {uint32}, layout {TILE}).
  `xfail_expected` is 0. There is no candidate to queue and no omission to document.
- **Perf: none filed.** The only golden cell, the perf-flagged `LOOSE_CASES[0]` `(1,1,32,128)`, is 4 tiles.
  `eval/prompts/perf_rules.txt`, which the runner injects into every perf refinement, says to measure only
  cells of more than 32 tiles and that this overrides any other instruction. BB-22 forbids adding test
  shapes, and `feature_spec.py` is read-only. A perf refinement targeting the flagged case would therefore
  carry a device-ns gate its implementer is told not to measure. That is an unmeetable Done-when, and it
  would stall the queue. A perf refinement on any other shape would violate the flagged-case rule and
  BB-22. `feature_spec.py` anticipates this: "a perf pass with a tile-count threshold may skip it; it is
  also measured in the model's trace." The perf findings are recorded below instead. For the same reason
  the verifier did not measure device-ns itself; the figures quoted are the implementer's.

## Recommendations and Findings

- **Perf position (implementer-measured, device kernel duration).** 1601 ns, against the incumbent
  2.17 µs (lead's goal 1: met) and the hardware bound of 0.07 µs (goal 2: ≤ 0.0945 µs). Goal 2 is
  not reachable as a kernel duration: launch plus one DRAM read round trip plus one write round trip
  already exceed it. The implementer recorded the remaining spread as ~453 ns from the first core's start
  to the last. If the tile threshold is lifted or the model trace becomes the judge, the unexplored lamps
  from `op_design.md` are:
  - `CORE_LIMIT ∈ {4, 8, 16}`: grid-sync lamp, a knob-turn.
  - `rows × tile_cols` across ≤ 110 cores: a knob-turn, but it trades 512-B page writes for 128-B partial
    writes.
  - A single fused gather+store kernel on one RISC (fewer binaries, BB-18).
  - Rectangle aspect and origin (`master.md` `noc_placement`).

  Judge them on unprofiled trace-replay time, not profiled kernel duration (lessons.md RB-053).
- **Mesh.** The SPMD mesh path (`generic_op` over a `MeshDevice`, one address per mesh buffer) is not
  exercised on this single-chip slot. It is correct by construction of `generic_op`, but only a
  multi-device run confirms it. That is a finding for a multi-chip board, not a queue item.
- **Large-W L1 (outside INPUTS).** Slot capacity is `32 · padded_chunk_row_bytes` whatever `core_rows` is.
  For example, `(3,1,96,4096)` on 110 cores holds ≤ 48 KB live in a 512-KB slot. It stays inside the
  bound by derivation. Shrinking it needs a store loop with a `min(core_rows, 32)`-row slot in place of the
  helper. Not needed for any golden shape.
- **Stale probes.** `probes/probe_001.py` through `probe_005.py` target earlier iterations of the
  descriptor API (`plan`, `create_program_descriptor`, `BLOCK_TILE_COLS_REQ`) and no longer run as written.
  They are kept as history.
