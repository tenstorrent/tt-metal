# Operation Requirements: g6_unsqueeze

## Definition
- **Formula**: `out[0,…,0, w, 0] = x[0,…,0, w]` for `0 ≤ w < W`. On tiles, input tile `t` (row 0, col
  `w%32`) → output tile `t` (row `w%32`, col 0). Bit-identical.
- **PyTorch Reference**: `def ref(x, dim): return torch.unsqueeze(x, dim)` (only `dim ∈ {r, -1}` is
  accepted)
- **Import Path**: `from ttnn.operations.g6_unsqueeze import g6_unsqueeze`
- **Function Signature**: `g6_unsqueeze(input_tensor: ttnn.Tensor, dim: int) -> ttnn.Tensor`. The input is
  `(1,…,1,W)`, rank 1–3, float32, TILE, DRAM interleaved; the output is `(…, W, 1)`.

## Phases

> **Non-regression rule**: Every refinement must pass all tests from prior phases.
> **Drift signal**: XPASS-strict failures mean the implementer added support but forgot to update SUPPORTED. The implementer fixes by updating SUPPORTED.
> **Checkbox protocol**: Implementer marks `[x]` when the refinement is complete and all tests pass, `[~]` when real work landed but at least one named axis value is deferred (treated as completed by the queue, surfaced as partial), `[ ]` only when nothing usable was produced.
> **Refinement ID + follow-up naming (mandatory — the runner parses this)**: Primary refinements are `Refinement N` (e.g. `Refinement 1`, `Refinement 2`). When you ship `[~]` partial and file the sharper follow-up the partial-tick protocol requires, name it by appending a lowercase letter to the parent's number: `Refinement 1b`, `Refinement 1c`, … (never `Refinement 1.5`, `Refinement 1 (follow-up)`, or a fresh number). Order follow-ups immediately after their parent so the queue runs them before later refinements — a partial's remaining-blocker follow-up must be picked next, not leapfrogged. The runner's parser matches exactly `Refinement \d+[a-z]?`; any other shape is invisible to the queue and silently skipped.

> **Queue shape (verifier):**
> - `TARGET − SUPPORTED` is empty (TARGET = SUPPORTED = {float32} × {TILE}, `INVALID = []`, no
>   `xfail_expected` cells), so there are no generality refinements and the 2:1 cadence collapses to perf
>   only.
> - Every refinement targets the perf-flagged loose case **p1 `[1,512] → [1,512,1]`, float32, TILE, DRAM
>   interleaved, dim 2**, at its full contract. That is the op's whole SUPPORTED surface, so no unlock is
>   needed first.
> - Its goal: below stock 3.32 µs (met at Phase 0, ~1.35–1.45 µs) and toward the lead's bound. The bound
>   (0.0135 µs) is below one program launch, so the practical goal is the launch floor.
> - Soft precision gate: bit-exact.
> - All levers are **knob-turns or transfer-shape changes** on the planner's surface. No scheme-change
>   exists: no dependent axis, and no reuse-shared operand.
>
> **Run the golden suite with `EVAL_CAPTURE_L1=0 eval/eval_test_runner.sh --no-precompile …`.** With L1
> capture on, the harness fails the 9 validation-refusal cases (zero allocation peak) and the traced case
> (it disables the program cache). Precompile hangs on the trace helper. See verification_report.md.
>
> **Measure device time with the golden runner's `device_kernel_ns`, or `--profile` on the knob tests.**
> Take medians of ≥5 runs: per-run noise is ±50 ns at this scale.

### [x] Phase 0 — Core Implementation

- **SUPPORTED dtype**: [float32]
- **SUPPORTED layout**: [TILE]
- **SUPPORTED shape-derived axes**: none (the one-row shape class is the op itself, not an axis)
- **SUPPORTED op-specific axes**: none (`dim ∈ {r, -1}` is a validation contract, not a registry axis)
- **Cores**: `w_tile` split over min(grid, ceil(W/32)) cores (p1 16, p2 8), with tile halves split across
  NCRISC/BRISC. `resident` regime, plus a `streaming` fallback for long assignments.
- **Compute config**: none (no compute kernel; pure NoC + RISC-store bit move)
- **Golden baseline**: 20 / 20 passing with `EVAL_CAPTURE_L1=0` (9 supported_pass, 11 regression tests
  without axes). Device time: p1 ≈ 1.35–1.45 µs, p2 ≈ 1.35–1.40 µs.

### [x] Refinement 1 — Core placement, core-range count and NoC selection on p1

**Type**: perf

**Goal**:
- p1 is fixed-cost-bound: p2 (8 tiles) and p1 (16 tiles) both take ~1.4 µs.
- On the 11×10 Blackhole grid, `num_cores_to_corerangeset(16, row_wise=True)` gives p1 **two core ranges**
  (11×1 + 5×1). Binaries and launch messages are written per range (BB-17).
- Co-tune the cheap T1/T2 placement knobs from `ttnn/ttnn/operations/examples/master.md`:
  - `noc_placement`: one rectangle (8×2, 4×4) vs the 11+5 row line vs a diagonal; per-instance NoC
    selection.
  - `bank_stagger`: issue order vs DRAM bank for the 16 × 64 B reads and 1024 B writes.
  - Re-run the grid-sync lamp: `MAX_CORES` ∈ {16, 8} with the halves on two cores vs one.
- Implement the winner as a `select_plan()` decision: a core-rectangle chooser in place of
  `num_cores_to_corerangeset`, still the single source of the core set. Report it in `describe_plan()`.
- Keep p2's single range.
- No SUPPORTED change.

**Verifier notes**: this comes first because it is the cheapest lever and the one rule-adjacent finding
(BB-17). The data path must not change: roles are chosen by core assignment, not by a different
transfer. The traffic is identical across candidates, so any win is launch or NoC-path latency.

**Done when**:
- Measured device-ns on p1 improves (median of ≥5) with the golden suite green (`EVAL_CAPTURE_L1=0`) and
  bit-exact.
- p1 runs on one rectangular core range, or the report records the measured reason a two-range set is
  faster.
- No regression on the guard set: p2 (`resident`, 8 cores), the forced-`streaming` knob test shape, and
  `HALVES_PER_RISC=2`.

**Outcome**: `select_plan()` now builds the core set with `_core_pool()`. It picks the most-square
rectangle at the origin with exactly `num_cores` cores: p1 4×4 (**one range, not 11+5**), p2 4×2 (one
range), full grid 11×10. Counts that do not factor into the grid (for example a prime 13) fall back to
the row-wise `num_cores_to_corerangeset`. `describe_plan()` reports `placement=` and `core_ranges=`.
The data path is unchanged and bit-exact; unit tests 59/59 and golden 20/20 pass (`EVAL_CAPTURE_L1=0`).
**No device-ns was measured.** p1's largest tensor is 16 tiles, below the >32-tile perf-measurement
floor of the task brief, which overrides this heading's measure-p1 instruction. So the single-range win
(one binary/launch-message write instead of two, per BB-17) is structural and unquantified here. The
`noc_placement` diagonal / per-instance NoC selection and `bank_stagger` variants were not built: with
no measurement allowed on the only target cell, there is no evidence to pick among them. p1 remains
fixed-cost (launch) bound at ~1.4 µs. Next lever if a >32-tile production cell ever appears: A/B the
rectangle against the row line and test bank-staggered issue order on that cell.

### [ ] Refinement 2 — Output write transfer shape on p1

**Type**: perf

**Goal**:
- Each tile half is written as one 1024 B face, 4× the aligned-minimum bytes (column 0 needs only
  16 rows × 16 B = 256 B per half at the 16 B DRAM write alignment). p1 writes 32768 B where 8192 B
  suffice.
- Measure the ledger's deferred cheapest-traffic variant against the current face write and the
  whole-page form. These are `tile_reorder` / `double_buffer` levers in
  `ttnn/ttnn/operations/examples/master.md`:
  - **(a)** 16 × 16 B column-segment writes per half, with the scatter into a 16 B-stride staging (smaller
    pages: 256 B per half).
  - **(b)** the current 2 × 1024 B face writes.
  - **(c)** one 4096 B whole-tile write per tile (both halves on one RISC, `HALVES_PER_RISC = 2`).
- Also measure the stateful one-packet write API for the small-write variant (set state once,
  `noc_async_write_one_packet_with_state` per segment), so (a) is judged on its best issue cost.
- Build the winner.
- No SUPPORTED change.

**Verifier notes**:
- Folds in the L1-ledger finding: the 16 B-segment variant was deferred on an expectation ("transaction
  count dominates"), not a measurement. This refinement must settle it either way.
- If (b) stays, record the measured numbers in `l1_ledger.md`'s cheapest-traffic line.
- The scatter must stay 32-bit RISC stores (bit-exactness).
- Output padding (faces 1/3, rows ≥ W) still need not be written.

**Done when**:
- Variants (a)/(b)/(c) are measured on p1 (median of ≥5 device-ns), and the fastest is built (or (b) kept
  with its numbers recorded).
- p1 device-ns improves or the ledger records why it cannot.
- Golden suite green and bit-exact (special values, poisoned padding, trace).
- No regression on the guard set: p2, forced `streaming`, ragged W (e.g. `(1, 500)`, `(1, 17)`), and
  `HALVES_PER_RISC=2`.

### [ ] Refinement 3 — Per-kernel fixed cost on p1 (issue path, args, image)

**Type**: perf

**Goal**: with placement and transfer shape settled, trim the fixed per-RISC cost on p1 inside
`g6_unsqueeze_face_move.hpp`. Each core does one 64 B read and one 1024 B write per RISC, so setup
dominates. Candidates:
- **Address setup:** `TensorAccessor` construction and `get_noc_addr` vs precomputed bank addressing for
  the single page each RISC touches. Stateful read/write (`noc_async_read_one_packet_set_state`, or the
  write equivalent) where a RISC issues several same-size transfers.
- **Runtime args:** move the two buffer addresses to common runtime args (one patch per kernel in the
  host descriptor cache, fewer RT words per core), and drop now-unused CT args.
- **Profiling zones:** check that the zones compile out of the plain build (BB-21) and that kernel image
  size stays minimal (BB-04/BB-18).
- **Single block:** in `resident` with one block, skip the block-loop scaffolding at compile time.

Use `master.md`'s `trid_double_issue` / `double_buffer` entries only if a RISC issues more than one block.
No SUPPORTED change.

**Verifier notes**:
- Last in the queue because its wins are the smallest and depend on Refinements 1 and 2. Refinement 2 may
  change how many transfers each RISC issues, and that decides whether stateful issue pays off.
- Gate on headroom: if p1 is within noise of an empty-kernel launch on the same core set (measure a
  no-op kernel pair on the chosen rectangle), record that floor in `verification_report.md` and close this
  refinement as no-op rather than forcing changes.

**Done when**:
- Measured device-ns on p1 improves (median of ≥5), or the report records the measured empty-kernel
  launch floor showing no headroom.
- Golden suite green and bit-exact.
- No regression on the guard set: p2, forced `streaming`, ragged W, and `HALVES_PER_RISC=2`.
