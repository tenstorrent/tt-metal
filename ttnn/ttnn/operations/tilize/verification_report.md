# Verification Report: tilize

Verifier pass over Phase 0 (prompt A0). Box: **Blackhole, compute grid 11×10 = 110 cores**,
AICLK 1349.98 MHz. All device runs via `scripts/run_safe_pytest.sh` / `eval/eval_test_runner.sh`.

## Code Review

### Fixed in this pass

1. **`_check_structural`: the "padding is never implicit" gate applied to TILE inputs too**
   (`tilize.py`). A retile-path (TILE-layout) input carries its **own** tile geometry and is
   tile-aligned in it by construction, but the check measured `shape[-2] % tile_h` against the
   *requested output* tile height. A legal retile such as `H = 16, in_tile_height = 16 → tile 32`
   therefore raised `ValueError: … padding is never implicit …` instead of the honest support refusal
   (`in_layout=TILE_LAYOUT` is not in SUPPORTED yet). That is a **wrong-refusal-mode** defect: it
   reports a contract violation where the truth is a missing capability, and it contradicts
   `feature_spec.INVALID` rule 4 (a TILE input can be neither padded nor non-aligned).
   *Fix*: gate that check on `input_tensor.layout != ttnn.TILE_LAYOUT`. The retile+pad refusal (which
   must stay a `ValueError` mentioning `pad`) and the ROW_MAJOR unaligned refusal are unchanged.
   *Measured effect*: 5 `test_translated.py::test_tilize_retile` cells moved from **hard failure** to
   **xfail (support refusal)**; the golden directory's hard-failure count went 221 → 216 and the
   number of non-refusal failures anywhere in the suite is now **0**.

2. **DRY: the default tile height was a literal restated in six places.** `32` appeared as the
   fallback in `tag_alignment`, `tag_tile_height`, `tag_in_tile_height`, `SUPPORTED["tile_height"]`,
   `validate()`'s `tile_shape` default, `_dispatch`'s `tile_height` default and
   `create_program_descriptor`'s default arg. Refinement 8 (tiny tiles) turns exactly this knob, and a
   value restated seven times drifts the instant it is turned.
   *Fix*: `DEFAULT_TILE_HEIGHT = 32` defined once in `tilize_program_descriptor.py` next to
   `TILE_WIDTH`, imported by the op file, and used at every site.

3. **`_cb_budget_bytes` never queried the device.** It probed a non-existent
   `ttnn.get_device_info(...).cb_limit` and therefore *always* fell through to the hard-coded
   400 KiB fallback — i.e. the depth-2 "use only if it fits" rule (prompt: *auto-fall-back to depth-1
   rather than OOM*; master.md **C16**) was gated on a magic number rather than on L1.
   *Fix*: query `ttnn.get_max_worker_l1_unreserved_size()` (1 532 160 B on this part) scaled by a named
   `_CB_BUDGET_L1_FRACTION = 0.5` — half, not all, because the sharded paths and an L1-interleaved
   output share the same per-core L1. The constant remains the fallback when the binding is absent, and
   the now-unused `device` parameter was dropped. No behaviour change today (the op spends a constant
   128 KiB/core, under both budgets); it becomes load-bearing at Refinements 2/7.

### Checked and found correct (no change needed)

- **Helper usage.** The one compute phase is `compute_kernel_lib::tilize<WT_BLOCK, …>` (plus a second
  instantiation at `WT_TAIL`), with `compute_kernel_hw_startup` first, `WaitMode::WaitBlock`,
  `InitUninitMode::InitAndUninit` on both calls (correct — the two calls carry different
  `block_width_tiles` and `tilize_init` takes the width), `Fp32Mode::Fast`, and
  `ReconfigureRegisterDatatypeMode` selected by a CT `needs_cast` flag — the prompt-mandated
  "skip the reconfigure when there is nothing to cast" lever is landed, not just noted. The aligned
  reader is `dataflow_kernel_lib::read_sticks_for_tilize<cb, TILE>` — one call per block, `tile_h`
  reads batched behind **one** barrier. Both `ASSERT(num_blocks > 0)` call sites are guarded by
  `if (n_full)` / `if (n_tail)`.
- **The one raw-API site is justified.** `ls ttnn/cpp/ttnn/kernel_lib/` + a grep for `noc_async_write`
  confirm there is **no tile-page writer helper**: the only write helper,
  `write_sticks_after_untilize`, indexes its destination by *row* and would scatter each tile across
  `tile_h` destination stick-pages. `noc_async_write` + `TensorAccessor::get_noc_addr(tile_index)` is
  the only mechanism; `op_design.md` §7.2 records this with file:line.
- **No multicast opportunity exists to miss.** The map is a bijection with no reuse-shared operand, so
  `mcast_pipe.hpp` is correctly absent (master.md **B12** is structurally not-applicable, not deferred)
  — pinned by `test_tilize_debug.py::test_b12_multicast_is_structurally_absent`.
- **CB sync ledger balances.** Per block, with `w ∈ {WT_BLOCK, WT_TAIL}` derived from the *same*
  decode in all three kernels: reader `push_back(w)`; compute `wait_front(w)`/`pop_front(w)` and
  `reserve_back(w)`/`push_back(w)`; writer `wait_front(w)`/`pop_front(w)`. Each CB has exactly one
  producer thread and one consumer thread. `num_pages = CB_DEPTH * WT_BLOCK ≥ WT_BLOCK ≥ WT_TAIL`, so
  the reader cannot deadlock in `cb_reserve_back`.
- **Kernel hygiene.** `void kernel_main()` (not the deprecated namespace form), `TensorAccessor` (not
  `InterleavedAddrGen`), `#include "api/dataflow/dataflow_api.h"` /
  `"api/compute/compute_kernel_hw_startup.h"`, `TensorAccessorArgs` appended **last** in the CT arg
  list (master.md **D18**) and only addresses + the per-core block range as runtime args (**D19**).

### Design conformance (`op_design.md`)

| Binding dimension | Verdict |
|---|---|
| Algorithm | ✅ One block = one output tile-row × `WT_BLOCK` tile-columns; helper-driven `tilize_block`, no materialized intermediate. |
| Pipeline topology / RISC ownership | ✅ Three separate kernels (not folded — `zero_copy_fold` measured folding at 0.74×); reader NCRISC/NoC0, writer BRISC/NoC1 (**B9**). |
| Parallelization / fills the machine | ✅ `b = wchunk*nt_h + r` linearized and split with `split_work_to_cores(…, row_wise=True)` **and** `grid_to_cores(…, row_wise=True)` (same flag on both — the §9 risk-18 trap avoided). Measured occupancy: square 110/110 cores, wide-short 32 cores (= `total_blocks`), tall-narrow 64, smallest 1 — every regime at `min(total_blocks, grid_cores)`. Both dataflow halves are batched (`tile_h` reads / `w` writes per barrier), CB depth 2. |
| Inter-core communication | ✅ None required and none present (bijection). |
| Blocking-model fidelity | ✅ See below. |

**Blocking-model fidelity — no collapsed knobs found.** `TARGET_READ_BYTES` → `WT_BLOCK_MAX` →
`WT_BLOCK` → (`n_wchunks`, `WT_TAIL`, both CBs' page counts, all three kernels' CT args, the reader's
per-block byte count) is a single derivation chain with one source; `CB_DEPTH`, `grid_cores`,
`NEEDS_CAST` and `tile_h` are likewise defined once. **No CB page count references a whole-op
dimension**: `num_pages = CB_DEPTH * WT_BLOCK` with `WT_BLOCK = min(Wt, WT_BLOCK_MAX)`, so per-core CB
L1 is a constant **128 KiB** — and, because the knob is a *byte* target, it is 128 KiB for bf16, fp32,
uint32 and uint8 alike, with no per-dtype literal (pinned by
`test_tilize_debug.py::test_cb_l1_is_constant_in_w` and `::test_wt_block_max_is_a_byte_target`). The
per-core compute loop is **not** half-turned: it iterates `WT_BLOCK = 16` tiles at bf16 (1024 B reads),
not the minimal 1, and the coarse default is backed by a recorded sweep (128 B → 152.6 µs … 1024 B →
44.5 µs on the square). The one pinned-at-1 factor, `NT_ROWS_PER_BLOCK`, is pinned **by the LLK**
(`tilize_helpers.hpp:121-125` calls the LLK once per `1 × block_width` tile-row), not by the
implementer, and the design records it as such. `use_multicore=False` is the trivial value of the
`grid_cores` parameter, not a second kernel — one program factory, confirmed.

### Prompt-rule audit (`eval/prompts/tilize.txt` §Rules)

Only rules whose condition is live at Phase 0 are checkable; the rest re-arm at the refinement that
adds their axis.

| Rule | Applies now? | Verdict |
|---|---|---|
| Padding is never implicit (MUST raise) | yes | ✅ `ValueError` naming `pad_value` / `output_padded_shape`; acceptance test green. **Narrowed this pass** to ROW_MAJOR inputs (see fix 1). |
| Retile + padding mutually exclusive (MUST refuse) | yes | ✅ explicit `ValueError` mentioning `pad`, raised before the SUPPORTED gate. |
| ROW_MAJOR input MUST tilize natively; MUST NOT call `ttnn.to_layout`/`ttnn.tilize` in Python | yes | ✅ no manipulation-op wrapper anywhere; the entry point calls `validate()` then `ttnn.generic_op`. |
| Skip the register-datatype reconfigure when there is no cast | yes | ✅ CT `needs_cast` → `NoReconfigure`. |
| Double-buffering user-controllable + "use only if possible" | yes | ✅ `use_double_buffer` kwarg → `CB_DEPTH`, with an L1-fit fallback (hardened this pass, fix 3). The `double_buffer=False` **axis** is not yet in SUPPORTED — Refinement 1. |
| Work distribution MUST parallelize wide, short tensors | yes | ✅ the 2-D linearization is unconditional; measured `width_split=0` off-arm is **6.03×** slower on wide-short. |
| A tiny tile redefines H-alignment; do NOT hardcode 32 | not yet (tile_height=32 only) | ✅ pre-satisfied — `tag_alignment` and `validate()` both measure against `tile_h`; re-armed at Refinement 8. |
| 8-bit dtype needs the per-face row dim | not yet (bf16 only) | re-arms at Refinement 7. |
| Fill packed in the INPUT format / replicated across the store word | not yet (no pad path) | re-arms at Refinement 5. |
| Arch-gate RETILE only, and SKIP rather than fail | not yet | re-arms at Refinement 8. |
| No host round-trip / no extra full-tensor DRAM pass | yes | ✅ exactly `2 × tensor_bytes` of DRAM traffic; tt-npe pin recorded in `changelog.md`. |

No MUST/MUST-NOT violation found. No unfollowed soft ("prefer/consider/avoid") rule to surface.

## Registry Conformance

- **Confirmed**: `INPUT_TAGGERS` (13 taggers, every one with the `(inputs, axes)` signature),
  `SUPPORTED` (15 axes — one per TARGET axis, `dtype`/`output_dtype` included), `EXCLUSIONS`,
  and `validate()` are all present and correctly wired. `validate()` order is: structural
  `ValueError`s (contract violations — required first, because the acceptance test asserts a
  `ValueError`/`RuntimeError` and `SupportRefusal` subclasses `NotImplementedError`), then
  **SUPPORTED per-axis**, then **EXCLUSIONS cell-level**. The public entry point calls `validate()`
  as its first statement, before any allocation or kernel work.
- **Confirmed**: the op file does **not** declare `INVALID` (it lives in `feature_spec.py`; the module
  docstring says so explicitly).
- **Tagger/registry agreement is structural, not coincidental**: `validate()` synthesizes the same
  scenario-dict shape the golden harness passes (`_scenario_from_call`) and then runs the *same*
  taggers over it, so the op and the registry cannot disagree about which cell a live call lands in.
- **No auto-fixes to SUPPORTED were required** — `xpass_drift = 0`. Note that multi-core *is* wired,
  correctness-tested and measured while `SUPPORTED["use_multicore"] = [False]`; that is the prompt's
  Phase-0 rectangle, not drift (validate() refuses the cell, so the kernel never runs and no XPASS can
  occur), and Refinement 1 flips it.
- **EXCLUSIONS are currently inert but correct.** Both entries pair `use_multicore=False` with a
  `shard_api` value that is not yet in SUPPORTED, so the per-axis gate refuses first. They are the
  design's mandated forward declaration and go live at Refinement 2 — keep them.

### INVALID audit (`eval/golden_tests/tilize/feature_spec.py`)

Well-formed against the three sanity rules; **no change requested**.

- **Single-tensor coupling.** `dtype × output_dtype` (cast family) and `dtype × pad_value` (a negative
  fill in an unsigned format) both look two-tensor at a glance but are not: the cast family is a
  *contract* statement (an int↔float reinterpretation is a different operation, documented in the
  file header), and the pad value is materialized **in the input tensor's element format**, so
  `dtype` and `pad_value` describe the same tensor. `in_layout × in_tile_height`,
  `in_layout × pad_mode` and `in_layout × alignment` are all single-tensor (the input).
- **Universe-must-change.** No entry encodes "not implemented yet" — those live in the op's
  `EXCLUSIONS` / the queue.
- **Canonicalization-only multi-axis.** The `in_layout`/`in_tile_height` pair collapses six redundant
  ROW_MAJOR copies onto the `"none"` sentinel — textbook.
- **bf8b + ROW_MAJOR**: correctly *absent*, and the absence is documented — `bfloat8_b` is not an
  input dtype in this op's TARGET at all (the input is always ROW_MAJOR, and block-float has no
  row-major form), so there is no cell to prune.
- **Norm-like weight canonicalization**: n/a (no weight tensors).
- **One observation, not a defect** (no action needed now, worth knowing at Refinement 8): rules
  `{in_layout: TILE, alignment: h_non_aligned}` etc. rest on "a TILE input is tile-aligned by
  construction" — true w.r.t. the input's **own** tile height, whereas `tag_alignment` measures H
  against the **requested output** tile height. So a legal retile like `H = 16, in_tile_height = 16 →
  tile_height = 32` tags `h_non_aligned` and is skipped as INVALID. It costs no golden coverage today
  (every retile scenario in `INPUTS` uses H ∈ {128, 256}); if Refinement 8 wants that cell, the entry
  — not the op — is what would need revisiting.

## Precision Baseline

`tests/ttnn/unit_tests/operations/tilize/test_tilize_precision_baseline.py`, bf16 → bf16, rank 4,
DRAM→DRAM, single-core (the Phase-0 SUPPORTED cell). `assert_with_pcc` +
`comp_allclose(rtol=0, atol=0)`.

| Shape | PCC | Max Abs Err | Mean Abs Err | Relative RMS Err | got/true median | got/true spread (p95−p5) |
|-------|-----|-------------|--------------|------------------|-----------------|--------------------------|
| (1,1,32,32) | 1.000000 | 0.000e+00 | 0.000e+00 | 0.000e+00 | 1.000000 | 0.000e+00 |
| (1,1,64,128) | 1.000000 | 0.000e+00 | 0.000e+00 | 0.000e+00 | 1.000000 | 0.000e+00 |
| (1,1,32,512) | 1.000000 | 0.000e+00 | 0.000e+00 | 0.000e+00 | 1.000000 | 0.000e+00 |
| (1,1,512,512) | 1.000000 | 0.000e+00 | 0.000e+00 | 0.000e+00 | 1.000000 | 0.000e+00 |

**Assessment**: `tilize` is a bijection on byte positions, so the correct baseline is not "high PCC"
but **bit-identity**, and that is what is measured — `torch.equal` holds on all four shapes, including
the wide-short `(1,1,32,512)` (which exercises the width-block path, i.e. a different block decode) and
the multi-block `(1,1,512,512)`. The got/true ratio is exactly 1.0 with zero spread everywhere, so the
scale-bug signature (a tight cluster at a non-1.0 constant — how a shuffled/strided tile would present)
is ruled out, not merely unobserved. There is no `severity=precision` cell anywhere in the suite and
therefore no precision refinement to file.

**Recommended tolerances**: PCC ≥ 1.0 with `rtol = atol = 0` (exact) for every same-dtype cell — keep
the test's `assert torch.equal`. Loosen only where a *cast* is genuinely lossy: `→ bfloat8_b` at
PCC ≥ 0.99 and `fp32 → bf16` at PCC ≥ 0.995 (Refinement 7); the integer and same-dtype paths must stay
exact, and `uint8` must be compared exactly (an every-other-row-zero strided tile survives a loose
numeric check).

## Verifier CLI Summary

`eval/eval_test_runner.sh eval/golden_tests/tilize/ <results_dir>` →
`python3 -m eval.verify_supported <results_dir> ttnn.operations.tilize --output <results_dir>/verifier_report.json`.
The full per-cell report (with axis dicts) is the run artifact in `<results_dir>`; a compacted archive
— summary + per-category node ids, stripped of the per-cell axis dicts to stay under the repo's
500 KB file limit — is committed next to this report as `verifier_report.summary.json`.

- supported_pass: **1**
- xfail_expected: **379**
- invalid_skipped: **580**
- supported_fail: **0**   ✅
- xpass_drift: **0**      ✅
- xfail_wrong_mode: **0** ✅
- supported_marked_xfail: 0 · invalid_unexpected: 0
- no_axes_found: 946 — the non-registry files in the same directory (`test_golden_main_tests.py`, the
  hidden external grader; `test_regression.py`; `test_translated.py`; `test_golden_main_trace.py`).
  They carry no registry axes, so the CLI cannot categorize them. Their 216 failures were audited
  individually: **all 216 are typed support refusals** (`UnsupportedAxisValue`) for capabilities the
  queue schedules — **zero** non-refusal failures, zero hangs, zero errors after the fix above.

Acceptance suite (`tests/ttnn/unit_tests/operations/tilize/`): **47 passed, 51 failed**, every failure
a support refusal for a not-yet-claimed axis (the acceptance file is deliberately whole-contract, per
its own docstring). The passing set is 4/4 single-core identities + all 4 structural refusals + 34
debug/blocking-model pins + 5 new precision-baseline tests. No regression vs. the Phase-0 record.

## Recommendations

- **Refinement priority.** Refinement 1 (multi-core into SUPPORTED) is the highest-value entry in the
  queue by a wide margin and is nearly free: the split is already the only code path, is identity-tested
  on all four distribution regimes, and is measured at **7.49× / 6.01× / 16.87×** (square / wide-short /
  tall-narrow) versus the single-core arm. Until it lands, the op's own **default** invocation
  (`use_multicore=True`) is refused, and no perf slot can measure a real config.
- **Do not re-litigate the bandwidth-knee cap.** `master.md:358-361` records it implemented, measured
  **~2.4× slower on this very op**, and refuted; the design explains why (the reader is
  transaction-rate bound at small pages, so there is no reachable knee). Use the full grid.
- **L1 / memory pressure (no OOM today).** Per-core CB L1 is a constant 128 KiB, independent of H, W,
  `Wt`, rank and batch, at every dtype. Two future pressure points to watch rather than pre-solve:
  (a) an **L1-interleaved output** (Refinement 1) puts the output tensor in the same per-core L1 the
  CBs spend — the depth-2 fit check now reads the real unreserved-L1 figure, but the *output* buffer is
  not in that accounting; (b) the **zero-copy sharded** path (Refinement 2) makes the CB *be* the shard,
  so `CB_DEPTH` must be forced to 1 there or the alias is meaningless. Neither is a `/memory-budget-metal`
  case yet — there is no OOM to move.
- **Perf headroom is real and localized**, which is why the perf slots target named regions rather than
  "make it faster": the square is at `achieved = 0.91` (84 % of theoretical DRAM peak — the residual is
  bank-adjacency/VC, master.md **A3**/**B10**), while the mandatory wide-short shape is at **0.70** and
  is *not* bandwidth-limited — it idles 78 of 110 cores because `WT_BLOCK = 16` leaves it only 32
  column-blocks. That block-size × grid-fill tension is the single most concrete unspent lever in the op.
- **The lever harness is an asset, keep it.** `DEFAULT_LEVERS` + `_bench_tilize.py`'s forcing arms make
  every landed optimization's counterfactual re-runnable (`levers=dict(knob=0)`) instead of an ad-hoc
  kernel edit, and `test_tilize_debug.py::test_lever_off_arms_are_still_correct` proves every non-stub
  arm still computes the right answer. Refinements should *add* knobs there, not bypass it. The
  `stub_*` arms are compile-time-guarded ablation paths and never emit in production.
- **Nothing was deferred that should have been fixed here.** The three items above are fixed in place;
  everything else in this report is either already correct or scheduled as a queue entry.
