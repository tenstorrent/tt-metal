# Operation Requirements: g6_zeros_like

## Definition
- **Formula**: `out[i] = +0.0` for every logical index `i` of `input_tensor` (float32: all 32 bits zero; bfloat8_b: decodes to 0). `shape/dtype/layout/placement(out) = those of input_tensor`; `out = optional_tensor` when given, else a new tensor. The input's values are never read.
- **PyTorch Reference**:
  ```python
  def pytorch_g6_zeros_like(input_tensor, *, optional_tensor=None):
      if optional_tensor is None:
          return torch.zeros_like(input_tensor)
      return torch.ops.aten.zeros_like.out(input_tensor, out=optional_tensor)
  ```
- **Import Path**: `from ttnn.operations.g6_zeros_like import g6_zeros_like`
- **Function Signature**: `g6_zeros_like(input_tensor: ttnn.Tensor, *, optional_tensor: ttnn.Tensor = None) -> ttnn.Tensor` (no compute configuration exists, and none may be added: the signature is fixed by the spec).

## Phases

> **Non-regression rule**: Every refinement must pass all tests from prior phases.
> **Drift signal**: XPASS-strict failures mean the implementer added support but forgot to update SUPPORTED. The implementer fixes by updating SUPPORTED.
> **Checkbox protocol**: Implementer marks `[x]` when the refinement is complete and all tests pass, `[~]` when real work landed but at least one named axis value is deferred (treated as completed by the queue, surfaced as partial), `[ ]` only when nothing usable was produced.
> **Refinement ID + follow-up naming (mandatory — the runner parses this)**: Primary refinements are `Refinement N` (e.g. `Refinement 1`, `Refinement 2`). When you ship `[~]` partial and file the sharper follow-up the partial-tick protocol requires, name it by appending a lowercase letter to the parent's number: `Refinement 1b`, `Refinement 1c`, … (never `Refinement 1.5`, `Refinement 1 (follow-up)`, or a fresh number). Order follow-ups immediately after their parent so the queue runs them before later refinements — a partial's remaining-blocker follow-up must be picked next, not leapfrogged. The runner's parser matches exactly `Refinement \d+[a-z]?`; any other shape is invisible to the queue and silently skipped.

> **Queue shape**: TARGET has one axis (`dtype`), and its only missing value (`bfloat8_b`) is Refinement 1. The perf-flagged contract (`LOOSE_CASES[0]`: float32, TILE, DRAM interleaved, new output, no compute config) is fully in SUPPORTED at Phase 0, so no anchor unlock is needed. After Refinement 1 the generality candidates are exhausted, and the rest of the queue is measured perf on the flagged shape.

### [x] Phase 0 — Core Implementation

- **SUPPORTED dtype**: [float32]
- **SUPPORTED layout**: TILE only (structural: `validate()` raises `ValueError` otherwise; not a registry axis)
- **SUPPORTED shape-derived axes**: none (rank 4, last two dims multiples of 32, DRAM interleaved are structural `ValueError` checks)
- **SUPPORTED op-specific axes**: none (`optional_tensor` given or not is a call form, supported in every phase, never an axis)
- **Cores**: folded page split over the full grid, column-major. float32 (1,12,128,128): 192 pages on 110 cores, rectangle `(0,0)-(10,9)`. bfloat8_b (5,1,32,256): 40 pages on 40 cores, `(0,0)-(3,9)`.
- **Compute config**: none (no compute kernel, no arithmetic; one writer kernel on BRISC/NoC1).
- **Golden baseline**: 4 / 4 supported cells pass, 2 `xfail_expected` (bfloat8_b), 8 / 8 regression tests pass, 0 loud categories.
- **Perf baseline (flagged shape)**: 3603 ns median kernel time (12 samples; goal ≤ 2310 ns, stock 4920 ns).

### [ ] Refinement 1 — bfloat8_b

**Goal**: add `ttnn.bfloat8_b` to `SUPPORTED["dtype"]`. This moves the 2 `xfail_expected` golden cells (`test_op[5x1x32x256-new_output-dtype=BFLOAT8_B]` and its loose twin) to passing. Expected kernel and host delta: none.
- The page size already comes from `output.buffer_aligned_page_size()` (1088 B).
- The CB format already follows the output dtype.
- The verifier probed the unchanged program dispatched directly on a bfloat8_b output: 0 non-zero elements of 40960, fresh and pre-filled buffers.

Also flip the test drivers that bypass `validate()` for bf8b to the public entry point:
- `test_g6_zeros_like_perf.py::test_profile`
- `test_g6_zeros_like_perf.py::test_profile_core_knobs_bf8`

`test_bfloat8_b` and `test_precision_bfloat8_b` switch to their exact-grade branch automatically.

**Implementation skill**: /numeric-formats-metal

**Verifier notes**:
- Only the dtype part of the skill applies. The spec fixes the signature with no compute configuration, so do **not** add `compute_kernel_config`. There is no DEST and no intermediate CB.
- Do not add a tagger, an `EXCLUSIONS` entry or any size/rank/call-form axis (a prompt MUST).
- Do not widen this refinement with placement retuning. `ROW_WISE` / `NUM_CORES_CAP` are global and shared with float32, and the flagged-shape perf refinements own them.
- The bf8b path already ships performance-conformant: 40 cores on one rectangle, 1 page each. Measured lamps (verification_report.md): col-40 815 ns, col-20 737 ns, row-40 954 ns, row-8 910 ns, all within ~±100 ns of each other.
- TARGET has nothing else to bundle with this. Its small size is the honest size of the gap.

**Done when**:
- The golden suite shows 6 / 6 axis cells passing, with `xpass_drift = xfail_wrong_mode = supported_fail = 0`.
- Acceptance, precision baseline (bf8b exact: 0 of 40960 elements ≠ 0) and trace tests are green.
- The bf8b production case's kernel time is recorded in `changelog.md` as its perf-guard baseline (median of ≥ 12 samples).

### [ ] Refinement 2 — DRAM write stream of the perf-flagged profile

**Type**: perf

**Goal**: `feature_spec.LOOSE_CASES[0]` flags float32 (1,12,128,128), TILE, DRAM interleaved, new output as the mandatory perf target.
- Goal: kernel time ≤ 2.31 µs (1.35 × the 1.71 µs DRAM roofline). Exact gate, no fidelity.
- Phase 0 measures 3603 ns median.

The region is the **DRAM write stream**:
- Per core, the zero fill takes ~190 ns and issuing 2 page writes ~100 ns. The write barrier then takes 1.65 µs median and 3.1 µs max.
- The aggregate stream reaches only ≈ 240 GB/s, about half the roofline.
- The core-count and placement lamps show it does not scale with occupancy: a row of 8 cores (3.18 µs) ≈ the full grid (3.6 µs). Column lines are 2–3× worse, and the writer on NoC0 is 1.6× worse.

Speed it up using the relevant patterns in `ttnn/ttnn/operations/examples/master.md`:
- **`read_coalescing`**, mirrored for writes, with bank affinity. This is the design's deferred `bank_run_coalesced` regime: each core owns contiguous in-bank runs (pages `b, b+8, b+16, …` of one bank are contiguous bytes), written as one `noc_async_write` per ≤ 16 KiB burst instead of one 4 KiB write per page, with the zero scratch grown to `min(run_bytes, 16 KiB)`.
- **`noc_placement`**: row or diagonal lines, writes on NoC1.
- **`bank_stagger`**.

Co-tune run length, core count (`NUM_CORES_CAP`) and placement (`ROW_WISE`). No SUPPORTED change.

**Verifier notes**:
- **Classification.** Knob-turns plus one transfer-shape change. It is an independent-axis re-split with no cross-core combine, so it stays one perf phase (T2 levers).
- **Scratch growth.** Since the verifier's fix, `ZERO_SCRATCH_PAGES` single-sources both the CB size and the kernel's fill size (CT arg `zero_scratch_bytes`). Growing the scratch is a one-place change.
  - The fill is on every core's critical path: ~160 ns of issue for 4 KiB (8 × 512 B loopback reads), so a 16 KiB scratch costs ~4×. Weigh it against the coalescing gain, or fill only one burst's worth and reuse it.
  - The DRAM-overload mechanism cap still applies: one `async_write_zeros` call stays within one page. Coalesced runs need plain `noc_async_write` from the scratch.
- **Plan constraints.**
  - Keep BB-17: few rectangular core ranges per production shape. Choose which core gets which bank-run, rather than scattering cores.
  - Keep `describe_plan` reporting the selected plan (BB-08). If a regime predicate appears, report the regime too.
  - Add every new knob to `cache_key` (BB-07).
- **Measurement.** Judge on the profiler's device kernel time, median of ≥ 12 samples (the spread is ±300 ns on this shape). Lesson RB-053: cores that finish at different times read slower under the profiler, so confirm the winner with the golden harness too.

**Done when**:
- Measured device-ns on the flagged shape improves on the 3603 ns baseline (same harness, ≥ 12 samples), moving toward ≤ 2.31 µs.
- The exact gate holds and the golden suite, acceptance, precision baseline and trace tests are green.
- No regression across the config-spanning guard set:
  - float32 (1,12,128,128) into_optional.
  - bfloat8_b (5,1,32,256) new_output, against its Refinement 1 baseline.

### [ ] Refinement 3 — Per-core fixed cost of the perf-flagged profile

**Type**: perf

**Goal**: same target, float32 (1,12,128,128) new output (`LOOSE_CASES[0]`, ≤ 2.31 µs). After Refinement 2 reshapes the write stream, the per-call fixed costs that no write-stream change removes become the remaining term. At Phase 0 these are:
- First→last core start: ~455 ns at 110 cores, ~275 ns at 40.
- The zero fill before every core's first write: ~160 ns issue + ~30 ns wait at 4 KiB, more if Refinement 2 grows the scratch.
- The write-barrier tail of the last core.

Speed it up using the relevant patterns in `ttnn/ttnn/operations/examples/master.md`:
- Fewer cores / fewer, rectangular ranges (`noc_placement`, the grid-synchronization lamp), where Refinement 2's stream keeps its bandwidth with them.
- Hide the fill: e.g. source the first write of each core straight from the 512 B `MEM_ZEROS` region while the loopback fill runs. This is the design's rejected `sub_page_writes_from_mem_zeros`, revisited as a hybrid only for the first burst. Alternatively, measure the RISC-store fill (`fill_l1_range`).
- Stateful writes (`noc_async_write_one_packet_set_state` / `_with_state`) if pages per core grow.
- A second writer RISC (`split_reader`, mirrored) only if the profile shows the BRISC issue-bound after Refinement 2.

No SUPPORTED change.

**Verifier notes**:
- Depends on Refinement 2's outcome, which fixes the core count and run length these costs scale with. Re-profile zones first.
- If Refinement 2 already meets ≤ 2.31 µs, this refinement still targets the remaining gap to the 1.71 µs roofline. If the profile shows no fixed-cost term above noise, close it with the measurement rather than a code change.

**Done when**:
- Measured device-ns on the flagged shape improves on Refinement 2's result (same harness, ≥ 12 samples).
- The exact gate holds and the golden suite, acceptance, precision baseline and trace tests are green.
- No regression on the guard set: float32 (1,12,128,128) into_optional; bfloat8_b (5,1,32,256) new_output.
