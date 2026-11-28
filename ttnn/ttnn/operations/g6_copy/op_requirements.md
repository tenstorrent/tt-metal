# Operation Requirements: g6_copy

## Definition
- **Formula**: `output_tensor[i] = input_tensor[i]` for every logical index i; returns `output_tensor` (same object, same buffer). Bit-identical for float32 / bfloat16 / int32 / uint32; bfloat8_b within one quantization step of its group's shared exponent.
- **PyTorch Reference**:
  ```python
  def pytorch_g6_copy(input_tensor, output_tensor):
      assert input_tensor.shape == output_tensor.shape and input_tensor.dtype == output_tensor.dtype
      return output_tensor.clone().copy_(input_tensor)
  ```
- **Import Path**: `from ttnn.operations.g6_copy import g6_copy`
- **Function Signature**: `g6_copy(input_tensor: ttnn.Tensor, output_tensor: ttnn.Tensor) -> ttnn.Tensor`. Both tensors share rank (1–4), shape, dtype, layout and memory config (DRAM interleaved). No other parameter; no compute configuration.

## Phases

> **Non-regression rule**: Every refinement must pass all tests from prior phases.
> **Drift signal**: XPASS-strict failures mean the implementer added support but forgot to update SUPPORTED. The implementer fixes by updating SUPPORTED.
> **Checkbox protocol**: Implementer marks `[x]` when the refinement is complete and all tests pass, `[~]` when real work landed but at least one named axis value is deferred (treated as completed by the queue, surfaced as partial), `[ ]` only when nothing usable was produced.
> **Refinement ID + follow-up naming (mandatory — the runner parses this)**: Primary refinements are `Refinement N` (e.g. `Refinement 1`, `Refinement 2`). When you ship `[~]` partial and file the sharper follow-up the partial-tick protocol requires, name it by appending a lowercase letter to the parent's number: `Refinement 1b`, `Refinement 1c`, … (never `Refinement 1.5`, `Refinement 1 (follow-up)`, or a fresh number). Order follow-ups immediately after their parent so the queue runs them before later refinements — a partial's remaining-blocker follow-up must be picked next, not leapfrogged. The runner's parser matches exactly `Refinement \d+[a-z]?`; any other shape is invisible to the queue and silently skipped.

> **Queue shape**: one generality refinement, then perf. The whole `TARGET − SUPPORTED` gap is a SUPPORTED
> widening with no kernel change (verifier probe `probes/probe_007.py`: every one of the 29 xfail cells passes
> bit-exact when validate() lets it through). It is one bundle; splitting it would be artificial. With
> generality exhausted after Refinement 1, every remaining phase is perf. The perf-flagged contract
> (`LOOSE_CASES[0]`: float32, TILE, DRAM interleaved, no compute config) is already fully supported at Phase 0.

### [x] Phase 0 — Core Implementation

- **SUPPORTED dtype**: [float32]
- **SUPPORTED layout**: [TILE]
- **SUPPORTED shape-derived axes**: alignment ∈ {tile_aligned, w_non_aligned, h_non_aligned}
- **SUPPORTED op-specific axes**: none
- **EXCLUSIONS (contract)**: {float32, ROW_MAJOR}, {int32, TILE}, {uint32, TILE}
- **Cores**: full grid. The `page` axis is split by `split_work_to_cores(row_wise=True)`, `min(tensor_pages, 110)` cores
- **Compute config**: none. Dataflow only: a reader on NCRISC/NoC0 and a writer on BRISC/NoC1, one CB. Bank-run coalesced transfers, bank stagger on
- **Golden baseline**: 10 / 39 registry cells supported_pass, 29 xfail_expected, 0 loud; 3 / 3 numerics regression tests pass (13 / 42 total)
- **Perf (flagged shape)**: (16, 12, 128, 128) fp32 measured 67.6 µs median, range 67.2–68.7 µs (goal < 68.56 µs; roofline 54.61 µs)

### [x] Refinement 1 — Full dtype set and ROW_MAJOR layout

**Goal**: Add the full dtype set and the ROW_MAJOR layout:
- Add `ttnn.bfloat16`, `ttnn.bfloat8_b`, `ttnn.int32` and `ttnn.uint32` to `SUPPORTED["dtype"]`.
- Add `ttnn.ROW_MAJOR_LAYOUT` to `SUPPORTED["layout"]`, for every tensor of the op (source and destination
  share it).
- Keep the three contract EXCLUSIONS ({float32, RM}, {int32, TILE}, {uint32, TILE}).
- bfloat8_b × RM stays INVALID (test-side).

This moves all 29 `xfail_expected` cells to `supported_pass`:
- bf16 TILE, 10 production rows.
- bf8b TILE, 1 row.
- bf16 RM, 4 rows.
- uint32 RM, 6 rows.
- int32 RM, 3 rows.
- 5 loose cases: adversarial bf16 / bf8b / int32 / uint32, and poisoned-padding bf16.

Support is native. The existing page walk already is the pure-RM data-movement pattern: a page is a row, and
`page_bytes = buffer_aligned_page_size()`. No host-side layout conversion, tilize, pad or slice.

**Implementation skill**: /memory-layouts, /numeric-formats-metal

**Verifier notes**:
- Probe evidence (`probes/probe_007.py`) says this should be a SUPPORTED and test change with zero kernel
  edits. Every gap cell selects `one_block`. Largest footprint is 36,864 B (bf16 (1, 1, 32, 62080)), and
  4–32 B int / uint rows become 64 B pages.
- Apply only the dtype-surface part of `/numeric-formats-metal`. The contract forbids a
  `compute_kernel_config` parameter, there is no compute kernel and no DEST, and the CB format is the tensor's
  own and never interpreted. Do not add a compute path: it changes values (design Regimes,
  `compute_passthrough`).
- From `/memory-layouts`, only the pure-RM (data-movement) pattern applies, not tilize-wrapped.
- Hold the new paths to the performance bar: they must stay on the same full-grid page split. They do today:
  bf16 RM (16, 3, 2560) uses 48 cores, and bf16 TILE (1, 1, 32, 62080) uses 110.
- Extend the acceptance tests within BB-22 (production cases plus cases the rules require):
  - Values per production (dtype, layout), graded by `helpers.TOLERANCES` (bit-exact, or the bfp8 one-step
    gate).
  - `describe_plan` for every production shape: deterministic, L1 ≤ 1347776 B.
  - Adversarial bits per dtype.
  - bf16 poisoned padding on (1, 1, 1, 256).
  - Shortest RM row: uint32 / int32 `(1,)`.
- Update `op_design.md` Phase 0 rows and the `l1_ledger.md` page-format column.
- This lands first, ahead of the perf phases, so their no-regression guard set can span every kernel path ×
  layout.
- The three contract EXCLUSIONS also pass in the probe. Leave them excluded until the contract changes.

**Done when**:
- The verifier CLI shows 39 / 39 registry cells `supported_pass`, with `xfail_expected`, `xpass_drift`,
  `xfail_wrong_mode` and `supported_fail` all 0.
- The 3 numerics regression tests still pass.
- The acceptance directory is green.

### [x] Refinement 2 — Speed up the perf-flagged profile: overlap reads and writes

**Type**: perf

**Goal**: `feature_spec.LOOSE_CASES[0]` flags float32 TILE (16, 12, 128, 128), DRAM interleaved, as the
mandatory perf target.
- Goal: device kernel time below the incumbent's 68.56 µs and toward the 54.61 µs roofline (bound dram,
  460.8 GB/s).
- Gate: `bit_exact`.
- The default plan meets the goal only on its median (67.6 µs; one of three runs was 68.65 µs).

Why it is slow: on `one_block` each core's writer starts only after its whole 28-page assignment has landed.
The reader (NCRISC) finishes at 38.5 µs and the writer (BRISC) at 67.2 µs, so reads and writes never overlap
on a core.

The lever is the planner's overlap lamp, a knob-turn on the existing `streamed` plan: co-tune block size ×
buffer depth together with bank stagger. Relevant `ttnn/ttnn/operations/examples/master.md` entries are
`double_buffer`, `bank_stagger`, and `trid_double_issue` for its note that batched barriers come first.
Measured hints on this board (median of 3):

| Plan | µs |
|------|----|
| `one_block` + stagger (current default) | 67.6 |
| block 4 / depth 2 + stagger | 60.6 |
| block 4 / depth 3 + stagger | 61.0 |
| block 2 / depth 2 + stagger | 64.6 |
| block 3 / depth 2 + stagger | 66.3 |
| block 1 / depth 4 + stagger | 70.5 |

Express the block knob so the optimum carries across shapes rather than as a per-shape constant: pages per
block, or relative to `num_banks`, which is known only on device. The model-flagged `LOOSE_CASES[1]`
(8, 12, 128, 128) fp32, 32.6 µs today, should move with the same knob. No SUPPORTED change.

**Verifier notes**:
- This refinement owns relaxing `test_g6_copy_plan`. It pins `plan == "one_block"` and
  `block_pages == max(core_pages)`, which is more than BB-08 requires. BB-08 requires the plan to be
  deterministic and reported, every production shape to get the default plan with model placement, and L1 to
  stay ≤ 1347776 B. The implementer logged this pin as the blocker that kept the measured 60.6 µs variant off
  the default.
- Keep the regime predicate single-sourced in `_plan()`.
- Leaving the coarsest-block default is justified by measurement. Record it:
  - `op_design.md`: Knob decisions, Regimes, Buffer-depth knobs.
  - `l1_ledger.md` `cb_page_staging` row: capacity becomes `depth · block_pages`, no longer the whole
    assignment.
- Shapes with few pages per core (1–4 per core at 110 cores) must not pick up extra block hand-offs. Keep them
  on one block where `core_pages ≤ block`.

**Done when**:
- The flagged shape's measured device kernel time is below 68.56 µs on every sample (≥ 5 samples), and its
  median improves on 67.6 µs by more than the 2–3% noise band.
- The `bit_exact` gate holds and the golden suite is green.
- No regression beyond noise on the config-spanning guard set. Measure cells of more than 32 tiles only:
  - fp32 TILE (8, 12, 128, 128), (2, 12, 128, 128), (1, 12, 128, 128).
  - bf16 TILE (1, 1, 32, 62080), (1, 1, 8448, 128).
  - bf8b TILE (5, 1, 32, 256).
  - bf16 RM (16, 3, 2560).

**Outcome**: `BLOCK_BYTES = 16384` (4 fp32 / 8 bf16 tiles per block), depth 2, stagger on; shapes whose per-core
assignment fits one block stay `one_block`. Flagged (16, 12, 128, 128) fp32: 68.0 → 61.6 µs median of 5 (max
61.7, every sample < 68.56); (8, 12, 128, 128): 32.9 → 30.8 µs; bf16 (1, 1, 32, 62080): 21.3 → 19.2 µs; the rest
of the guard set is on `one_block` either way or within noise. Remaining 1.13× vs the 54.61 µs roofline sits in
the per-block drain-everything read barrier and the short (4 KB) bank runs — Refinement 3's trid / bank-affinity
levers, not done here because they are that refinement's scope. Depth 3 (60.7 µs flagged) and 32 KiB (61.4 µs)
are within noise of the default and regress (8, 12, 128, 128), so not taken.

### [ ] Refinement 3 — Speed up the perf-flagged profile: transaction shape toward the roofline

**Type**: perf

**Goal**: same target, `LOOSE_CASES[0]` fp32 TILE (16, 12, 128, 128), toward the 54.61 µs roofline. After
Refinement 2, about 1.11× remains if the 60.6 µs hint lands. That remainder sits in the block hand-off and in
transaction shape:
- Every block ends in a drain-everything `noc_async_read_barrier()`, so the read stream idles at each
  hand-off.
- Each core spreads ≤ 8 short runs over all 8 banks.

Levers from `ttnn/ttnn/operations/examples/master.md` (⭐⭐ T2 each; two in one phase):
- `trid_double_issue`: per-block transaction ids, so block k+1's reads stay on the wire across block k's
  hand-off. Helpers: `dataflow_kernel_lib::set_read_trid` / `async_read_barrier_with_trid`.
- `read_coalescing` bank affinity: the design's traffic-ranking candidate B, which gives each core contiguous
  slots of one bank, with longer runs.

No SUPPORTED change.

**Verifier notes**:
- Headroom gate: if Refinement 2 leaves the flagged shape at ≤ 1.05× roofline (≤ 57.3 µs), close this
  refinement with that measurement as "no headroom" in the changelog rather than forcing a lever.
- Bank affinity measured 0.91–1.28× in the catalog and can regress, so measure it, don't assume it.
- Bank affinity changes the page → core map and the walk's page enumeration. The reader and writer must still
  walk a block identically: the shared header invariant, which the forced `streamed` tests with stagger cover.
- `trid` with the CB: per the catalog README, the CB write pointer lags the in-flight blocks. Land each block
  in its own slot.

**Done when**, either of:
- **Improved**: the flagged shape's measured device-ns improves on Refinement 2's result beyond noise,
  `bit_exact` holds, the golden suite is green, and the Refinement 2 guard set shows no regression.
- **No headroom**: the headroom-gate closure above is recorded with its measurement.
