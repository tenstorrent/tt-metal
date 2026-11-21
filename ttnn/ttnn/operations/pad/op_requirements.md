# Operation Requirements: pad

## Definition
- **Formula**: `out[i0+b0, i1+b1, i2+b2, i3+b3] = in[i0, i1, i2, i3]`. Every other `out` element is `value`
  converted to the input's dtype. `out.shape[d] = b_d + in.shape[d] + a_d`, with `padding[d] = (b_d, a_d)`, first
  dimension first.
- **PyTorch Reference**:
  ```python
  def reference(input_tensor, padding, value):
      flat = [amount for pair in reversed(list(padding)) for amount in pair]
      return torch.nn.functional.pad(input_tensor.double(), flat, mode="constant", value=float(value))
  ```
- **Import Path**: `from ttnn.operations.pad import pad`
- **Function Signature**: `pad(input_tensor: ttnn.Tensor, padding, value: float) -> ttnn.Tensor`.
  `padding` is any sequence of `(before, after)` int pairs, one per dimension, first dimension first.
- **Specification**: `eval/specs/pad/spec.py` (governs). The numeric requirement is `exact` for every cell.

## Phases

> **Non-regression rule**: Every refinement must pass all tests from prior phases.
> **Drift signal**: XPASS-strict failures mean the implementer added support but forgot to update SUPPORTED. The implementer fixes by updating SUPPORTED.
> **Checkbox protocol**: Implementer marks `[x]` when the refinement is complete and all tests pass, `[~]` when real work landed but at least one named axis value is deferred (treated as completed by the queue, surfaced as partial), `[ ]` only when nothing usable was produced.
> **Refinement ID + follow-up naming (mandatory — the runner parses this)**: Primary refinements are `Refinement N` (e.g. `Refinement 1`, `Refinement 2`). When you ship `[~]` partial and file the sharper follow-up the partial-tick protocol requires, name it by appending a lowercase letter to the parent's number: `Refinement 1b`, `Refinement 1c`, … (never `Refinement 1.5`, `Refinement 1 (follow-up)`, or a fresh number). Order follow-ups immediately after their parent so the queue runs them before later refinements — a partial's remaining-blocker follow-up must be picked next, not leapfrogged. The runner's parser matches exactly `Refinement \d+[a-z]?`; any other shape is invisible to the queue and silently skipped.

### [x] Phase 0 — Core Implementation

- **Target**: Quasar on the ttsim simulator. One `ttnn.ProgramSpec`, dispatched once by `ttnn.generic_op`.
  Two DM kernels (reader, writer) and no compute kernel.
- **SUPPORTED dtype**: [bfloat16]
- **SUPPORTED layout**: [ROW_MAJOR]
- **SUPPORTED memory_config / sharding**: [DRAM] / [interleaved]
- **SUPPORTED shape-derived axes**: alignment ∈ {tile_aligned, w_non_aligned, h_non_aligned}; rank = 4
- **SUPPORTED op-specific axes**: pad_value = 0.0; padding_side = after
- **Cores**: full grid. Data rows and pad rows are split as two lists with `split_work_to_cores(row_wise=True)`.
  Thread plan (R, C, W) = (1, 0, 1); the op is bound by data movement.
- **Compute config**: none. The op moves bytes and computes nothing, so the specification gives it no compute config.
- **Golden baseline**: 14 / 14 PHASE0 cells pass, plus 4 / 4 rejections. 242 cells outside SUPPORTED are
  refused (`xfail_expected`). The loud categories are 0 (per `verifier_report.json`).

### [ ] Refinement 1 — Complete TARGET: float32, L1, value 1.0, padding before / none, ranks 2–3

**Goal**: add every remaining TARGET value to SUPPORTED, one per axis:
- SUPPORTED["dtype"] += ttnn.float32
- SUPPORTED["memory_config"] += L1
- SUPPORTED["pad_value"] += 1.0
- SUPPORTED["padding_side"] += before, none
- SUPPORTED["rank"] += 2, 3

Each one is a knob-turn on the existing program; the Phase 0 design already threads them through.
- `elem_bytes` is a CT arg and `pad_bits` an RT arg.
- The input accessor reads L1 or DRAM; the output is allocated in the input's `buffer_type` (R4).
- `bn, bc, bh, bw` are RT args. The reader's data lands at `head_bytes`, and `pad_row_to_output_row` counts
  `before` segments.
- `none` is the same program with `P = 0`. It is still one dispatch (a copy); never return the input.
- Ranks 2 and 3 are folded to rank 4 on the host (`fold_to_rank4`).

**Implementation skill**: /numeric-formats-metal

**Verifier notes**:
- **The surface is measured, not assumed.** I ran the golden suite with SUPPORTED temporarily set to TARGET, on
  the code after verification. Result: **260 / 261 passed** (all 256 cells and 4 rejections; the 1 skip is the
  empty loose-case parameter).
  - Before the verifier's wide-entry fix (see `verification_report.md`), the same run aborted the simulator on
    the 4 fp32 `(2,1,64,4096)` C-pad cells.
  - So the expected kernel change is **zero**. The work is the SUPPORTED widening plus the tests and reviews
    below, not code.
- **/numeric-formats-metal** applies to the float32 part only. There is no `compute_kernel_config` to expose and no
  intermediate-CB precision to set: the specification says "it therefore has no compute config", and there is no
  DEST. R1 holds because rows move byte-exact over the NoC. Check it on the fp32 cells (`exact`).
- **No sharding work.** `memory_config` L1 is interleaved-L1 placement only; `sharding` stays `["interleaved"]`,
  which equals TARGET.
- **Things the golden suite cannot see.** Cover them in `tests/ttnn/unit_tests/operations/pad/`:
  - R4 for an L1 input: `output.memory_config()` is L1 and interleaved. The precision baseline checks only DRAM.
  - `padding_side = before` under the plans `(1,0,1)`, `(1,0,2)`, `(2,0,2)`, and in `rows_streamed` with
    `block_rows_override`. The head margin is filled on each tile counter's first lap only, and today's plan
    tests use after-padding exclusively.
  - fp32 rows wider than one packet (`W_o > 2048`) with ≥ 4 rows per core. This is the explicit-writer path,
    already covered for bf16 by `test_rows_wider_than_a_packet`.
- **Ordering:** first. Refinement 2 (perf) is gated on shapes and guard paths (L1, before-padding) that this
  refinement opens.

**Done when**: SUPPORTED equals TARGET. The golden suite has 256 / 256 `supported_pass`, 0 `xpass_drift` and
0 `xfail_wrong_mode`; the `exact` numeric requirement holds on every fp32 and bf16 cell. The R4 L1-placement test
and the before-padding plan and regime tests above pass, along with every Phase 0 test.

### [ ] Refinement 2 — Overlap reads and writes on cores with many data rows

**Type**: perf

**Goal**: in the `rows_resident` regime each core's whole data assignment is **one** block (`ring_blocks = 1`).
The writer therefore issues no data write until the core's last read barrier has passed, and only the core's
pad rows fill that window. This is op_design.md's "Overlap" perf lamp.

Target shapes, from INPUTS:
- `(1,1,2048,256)` + H `(0,48)`: 64 rows per core, 48 pad rows over the whole grid.
- `(2,4,128,512)` + N `(0,1)`: 32 rows per core.

YOLOv8's untraced 640×640 stem call has the same profile: `D = 1920`, 60 rows per core, 1280 B rows.

Lever: split each core's data assignment into ≥ 2 blocks with `ring_blocks = BUFFER_DEPTH` at the same or smaller
ring bytes. This is a co-tune of the planner's `block_rows` and buffer-depth knobs (catalog patterns
`double_buffer` and `trid_double_issue`, ⭐⭐ T2, in `ttnn/ttnn/operations/examples/master.md`). Per the catalog's
chunk floor, keep blocks coarse: a few rows minimum, not one row per barrier.

Optional cheap second lever (⭐ T1, op_design.md's "Margin fill" lamp): when `pad_bits == 0`, build the template
and margins with the iDMA zero path, `noc.async_write_zeros` + `write_zeros_l1_barrier` (`noc.h:816-849`; no
other NoC write between the two), instead of CPU stores.

No SUPPORTED change.

**Verifier notes**:
- **Gate unit.** This backend has no clock (backend page, *Performance measurement*), so the gate is a
  **structural count**, read off the host program (`create_program_artifacts` → `block_rows`, `ring_blocks`,
  data count per core) by a test:
  - **Data blocks per core** on the two target shapes: **1 now → ≥ 2 when done**. Equivalently, the data rows
    whose write can issue before the core's last read barrier go from **0 to ≥ half** of the core's rows.
  - If the T1 lever is taken: CPU L1 stores for fills on a `value = 0.0` call, **`W_o` + margins per core now → 0**.
- **Time gain is unconfirmed** until a timed backend (`ttsim-perf`) or hardware measures it. Say so in the
  changelog, and report no simulator timing as a speedup.
- **Guard set** (structural, no regression): one representative per distinct path:
  - implicit writer (`entry_bytes ≤ 8192`) and explicit writer (`entry_bytes > 8192`);
  - pad-only cores;
  - DRAM and L1 placement (L1 opened by Refinement 1);
  - after- and before-padding.

  On every guard: the input still crosses DRAM once (1 read per input row), the output once (1 write per output
  row), 0 B cross-core, and the ring stays within `RING_ENTRY_CAP` / `DFB_RING_MAX_BYTES`.
- **Splitting the block weakens the margin invariant.** Head and tail margins are filled on each tile counter's
  first lap only, so with ≥ 2 blocks the ring wraps in the default regime. `test_rows_streamed_regime` already
  covers wraps through the override; extend it to the new default on the target shapes, with value 1.0.

**Done when**: on both target shapes the default program has ≥ 2 data blocks per core with `ring_blocks ≥ 2`, read
off the op by a test. Results stay exact with value 1.0, the golden suite is green, and every guard-set count above
is unchanged. The changelog records the before / after counts and states that the time gain is unconfirmed on this
backend.
