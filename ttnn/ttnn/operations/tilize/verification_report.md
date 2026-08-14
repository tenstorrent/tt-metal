# Verification Report: tilize

Verified against `op_design.md` (incl. its **Blocking Model** §1), `eval/prompts/tilize.txt`
(`## Rules`), `eval/golden_tests/tilize/feature_spec.py`, and the `ttnn/cpp/ttnn/kernel_lib`
tilize helpers. Artifacts: `verifier_report.json` (this directory),
`op_requirements.md`, `changelog.md`, `tests/ttnn/unit_tests/operations/tilize/test_tilize_precision_baseline.py`.
`verifier_report.json` in this directory is the **trimmed** form (summary + per-category counts +
the xfail blocking-axis histogram + 10 sample cells per category) — the untrimmed 1.2 MB report
exceeds the repo's 500 KB file limit and stays in the eval results dir.

## Code Review

### Fixed

1. **Collapsed knob in the L1 budget math (DRY violation) — `tilize_program_descriptor.py`.**
   The CB page formula `cb_depth * NT_BLK * wt_chunk` was written in one place, but the two
   quantities that must agree with it were hand-restated *without the `NT_BLK` factor*:
   `derive_blocking()`'s L1 ceiling (`per_chunk_tile = cb_depth * (in + out)`) and the
   never-OOM depth fallback (`cb_depth * wt_chunk * (in + out)`). `NT_BLK` is a real knob
   (design §1.4, lamp **L3**, and the `NT_BLK > 1` trid-double-issue perf lever in the queue):
   raising it to 2 would have doubled the CB footprint while the ceiling it was checked against
   stayed put — i.e. a silent L1 overflow the moment the knob is turned. Introduced
   `cb_pages(cb_depth, wt_chunk)` and `cb_bytes(...)` as **the** single source, and routed all
   three consumers (ceiling, fallback, both `CBDescriptor.total_size`) through them.
   No behavioural change at `NT_BLK == 1` (verified: identical verifier categories before/after).
2. **Same formula restated a fourth time in a test** — `test_tilize_regimes.py::test_cb_footprint_is_bounded_in_w`
   asserted against its own copy of the byte formula, so the guard could pass while the shipped
   sizing drifted. Now asserts against `cb_bytes()`.
3. **Added `test_cb_geometry_has_a_single_source`** (`test_tilize_levers.py`) — pins that the CB
   geometry formula lives only in `cb_pages()`/`cb_bytes()`, that the L1 cost per chunk-tile-column
   scales with `NT_BLK`, and that `create_program_descriptor` does not restate it. This is the
   regression guard for fix (1).
4. **Added `test_production_switches_ship_in_their_optimal_state`** (`test_tilize_levers.py`) —
   `LEVERS` / `ABLATE` are module-level dicts the perf bench mutates to build counterfactual and
   ablation arms. An `ABLATE` entry left at 1 produces *deliberately wrong output* and an `OFF`
   lever ships a measured-slower kernel; nothing pinned the shipped corner. Now
   `ABLATE == {compute: 0, dm: 0}` and every `LEVERS` value `== 1` is asserted.
5. **Added the precision baseline suite** (see below) — it doubles as a bit-exactness guard: the
   no-cast pairs are asserted `max_abs == 0` and got/true ratio identically `1.0`, which is the
   contract for a byte-permutation op and was previously only checked through a PCC floor.

### Reviewed and found correct (no change needed)

- **Blocking-model fidelity.** Work unit = 1 tile-row × `WT_CHUNK` tile-columns, exactly design §1.3
  candidate 1. `WT_CHUNK` is the **coarsest** exact divisor of `WT` that fits the L1 ceiling and fills
  the grid — not the minimal unit, so the granularity floor (whole tiles minimum, coarser amortizes)
  is respected; `n_chunks` divides `WT` exactly, so there is one compute kernel and no cliff-width
  variant. Neither CB is a function of `WT` / `NT_H` / any tensor dimension. `NUM_CORES`, `CB_DEPTH`,
  `NT_BLK` are parameters. **No half-turned split**: the per-core compute loop is one helper call over
  all of that core's blocks at `WT_CHUNK` width, and the 1-tile-row H granularity is not a collapsed
  knob but the LLK's own unit (`tilize_block` is one tile-row; `read_sticks_for_tilize` barriers per
  tile-row) with the escape route named as lamp L3.
- **Performance conformance / grid fill.** The design says spread over the whole grid on *both* shape
  regimes; the implementation does, including the wide/short `nt_h=1` regime where a pure height split
  collapses to one core (pinned by `test_grid_fill_on_wide_short_shape`, and measured 4.1–4.8× vs the
  single-core arm). Both dataflow halves are batched: the reader coalesces `TILE_H` stick reads under
  **one** barrier per block (via the library helper, and in the pad reader too), and the writer issues
  `WT_CHUNK` whole tile pages under one barrier — no dribbling half. CBs are depth-2 by default with a
  never-OOM fallback to depth-1.
- **Helper usage.** Compute uses `compute_kernel_lib::tilize<...>` once per core (init/uninit amortized
  across every block) with `WaitMode::WaitBlock` and, correctly, `ReconfigureRegisterDatatypeMode::NoReconfigure`
  when `output_dtype == input_dtype` — the ~150 ns reconfigure is a required-by-rule saving and it is
  wired. The aligned reader delegates verbatim to `dataflow_kernel_lib::read_sticks_for_tilize<TILE>`.
  The two raw-dataflow substitutions are justified in-file and I re-verified both claims against the
  library: the reader helper has **no fill parameter** (`.inl:120-123` reads only `row_bytes` while
  advancing L1 by the padded stride, so the W tail keeps stale data), and `kernel_lib` contains exactly
  one `noc_async_write` anywhere (`tilize_helpers_dataflow.inl`, `write_sticks_after_untilize`) which is
  the *inverse* direction — there is no CB-tiles → tiled-pages writer helper to use. No mcast/semaphore
  code exists, correctly: design §1.2 proves no operand is ever read by more than one core, so
  `mcast_pipe.hpp` has nothing to do here.
- **Correctness mechanics.** CB push/wait counts balance in every path (`wt_chunk` per block on both
  CBs, including the ablation arms); `TensorAccessor` everywhere (no `InterleavedAddrGen`);
  `void kernel_main()`; includes are `api/dataflow/dataflow_api.h`. Kernels early-return on
  `num_blocks == 0` so zero-work cores in the second `split_work_to_cores` group are safe.
- **fp32 datapath.** `fp32_dest_acc_en` + `UnpackToDestFp32` are enabled *only* on fp32→fp32, which is
  exactly the case where `can_use_fast_tilize()` is false (fp32 output disables the fast path), so the
  helper's "fast + UnpackToDestFp32 corrupts output" static assert cannot fire and the slow path gives
  bit-exact fp32. Confirmed empirically: fp32→fp32 measures `max_abs = 0`.
- **Prompt `## Rules` (conditional rules whose condition currently applies).** All hold:
  padding never implicit (raises with a message naming "pad"); fill packed in the **input** element
  format (`_pack_pad_word(..., input_tensor.dtype)`); sub-word fills replicated across the 32-bit store
  word (`fill_l1_with_val<elem_bytes>`, incl. the 1-byte case); no extra DRAM pass for padding (fill is
  L1 stores inside the reader); padded shape changes, logical shape restored by a zero-copy reshape;
  RM input tilized natively on device (no `ttnn.to_layout` / `ttnn.tilize` wrapper anywhere);
  `dtype=` performs a real value-preserving cast; reconfigure skipped when there is no cast;
  `use_double_buffer` exposed with the auto-fallback-to-depth-1. Rules for `uint8` per-face row dim,
  tiny-tile H-alignment, and retile↔pad exclusivity do not apply yet — they re-arm with Refinements 4
  and 5 and are quoted in those entries.

### Advisories (not blocking; no code change)

- **fp32 → bf16 casts truncate rather than round-to-nearest.** The measured got/true ratio for that
  pair lies *entirely below 1.0* (median 0.99729, p5 0.99394, p95 0.99974) with `Max RTOL Delta ≈ 0.0078
  ≈ 2^-7`, the signature of dropping the low 16 mantissa bits — consistent with `fast_tilize`'s pack
  stage stepping DEST at bf16 stride. It is inside the value-preserving-cast tolerance the prompt
  allows (golden floor 0.995; PCC measured 0.999998) and is *not* a scale bug (the errors are spread
  across the ulp band, not clustered at a constant), but it is a systematic 1-ulp bias toward zero
  rather than symmetric rounding. Filed here rather than in the queue because no concrete lever is in
  scope: there is no `pack_rounding` knob exposed by the tilize helper, and forcing `Fp32Mode::Lossless`
  changes the DEST path, not the pack rounding. Worth a look while Refinement 4 is in the numeric
  surface anyway.
- **The prompt's Track-A recording rule asks for a tt-npe pin** (cycles + DRAM util + congestion).
  `tt_npe.sh` does not exist in this checkout, so it could not be produced and Phase 0's perf record has
  the `/perf-ceiling-dm` bracket + device measurement but no tt-npe number. Not a code defect; carried
  into Refinement 3's verifier notes so the omission is stated rather than silently repeated.
- **Cross-language CB index duplication.** `CB_INPUT_STICKS = 0` / `CB_OUTPUT_TILES = 16` are restated as
  `constexpr` literals in each kernel. Left as-is: these are buffer *indices*, not block factors, and the
  framework has no shared header for them; the DRY rule that matters (block factors) is now satisfied.
- **`LEVERS` / `ABLATE` in production code.** Design §9.1 mandates the counterfactual and ablation arms,
  and `lever_ledger.json` + `_bench_tilize.py` consume them, so they stay. Now guarded by a test (fix 4)
  so the shipped corner cannot drift.
- **Two setup ERRORs in the hidden grader are harness friction, not op failures.**
  `test_golden_main_tests.py` carries a module-level `pytestmark = use_module_device` *and* a
  `device_params`-parametrized trace test, a combination the root conftest rejects by design; the file's
  own docstring says that test "lives in test_golden_main_trace.py". Nothing in the op is reachable from
  those two ids. Unfixable from here (grader file is read-only).

## Registry Conformance

- **Confirmed present and correctly wired** in `tilize.py`: `INPUT_TAGGERS` (13 taggers, every one with
  the `(inputs, axes)` signature), `SUPPORTED` (15 axes — `dtype`, `output_dtype`, and one per tagger:
  no axis the kernel gates on is missing), `EXCLUSIONS` (2 cell-dicts, both documented), and `validate()`
  which checks **SUPPORTED per-axis first, then EXCLUSIONS**, raising `UnsupportedAxisValue` /
  `ExcludedCell` from `ttnn.operations._op_contract`. `tilize()` calls `validate()` on its **first line**,
  before any allocation or kernel work.
- **`INVALID` is NOT declared in the op file** ✓ (it is sourced from `feature_spec.py`).
- Notable and correct: `validate()` projects the axes from the **raw** call and gates *before*
  `_canonicalize()`'s shape-legality checks, so an out-of-rectangle cell (e.g. rank 0) is refused with
  the typed support refusal instead of an earlier `ValueError`. `PROPERTIES` is the optional block from
  `eval/op_template.py` §3b.
- **No auto-fixes to SUPPORTED were needed**: `xpass_drift = 0`, so there is no under-claim; and
  `supported_fail = 0`, so there is no over-claim. SUPPORTED describes reality exactly.

### INVALID audit (`eval/golden_tests/tilize/feature_spec.py`)

All 24 entries pass the three sanity rules; no change requested.

- **Single-tensor coupling.** The 12 cast-family entries couple `dtype` × `output_dtype`, which are the
  input and output tensors — the one shape of cross-tensor entry that is legitimate here, because it is
  the *contract-level* cast family (int↔float casts are out of contract per the prompt), not a coupling
  between two independent input tensors. Every other entry is single-tensor: `pad_value` sign × unsigned
  input `dtype` (the fill is materialized in the input element format, so a negative fill has no
  encoding); `in_layout` × `in_tile_height` (a ROW_MAJOR tensor has no tile geometry — canonicalization
  only); `in_layout=TILE` × `pad_mode` / `alignment` (a TILE input is tile-aligned by construction).
- **Universe-must-change.** Every entry removes real cells (568 cells were skipped as INVALID in the
  golden run), and none of them shadows a cell the op would otherwise be graded on.
- **Canonicalization-only multi-axis exception.** The `in_layout` × `in_tile_height` pair is exactly that.
- **The canonical bf8b + ROW_MAJOR entry is correctly absent**: `TARGET["dtype"]` (the ROW_MAJOR *input*)
  never contains `bfloat8_b` — block-float has no row-major form and it appears only as an
  `output_dtype`. So the impossibility is excluded by construction rather than needing an INVALID row.
  Not a norm-like op, so the no-weight canonicalization cells do not apply.
- Deliberately absent and correctly so (documented in the file): the `pad_mode ↔ pad_value ↔ alignment`
  couplings and the retile `tile_height ≠ in_tile_height` inequality, because every one of those axes is
  projected off a single scenario dict — `cartesian()` cannot generate an incoherent combination.

## Precision Baseline

`tests/ttnn/unit_tests/operations/tilize/test_tilize_precision_baseline.py`, 4 shapes × 4 dtype pairs,
`assert_with_pcc` + `comp_allclose`. tilize is a byte permutation, so the reference is the identity and
every non-zero error below is purely cast-representation error.

| Shape | dtypes | PCC | Max Abs Err | Mean Abs Err | Relative RMS Err | got/true ratio med (p5..p95) |
|---|---|---|---|---|---|---|
| (1,1,32,32) | bf16 → bf16 | 1.000000 (exact) | 0.000e+00 | 0.000e+00 | 0.000e+00 | 1.000000 (1.000000..1.000000) |
| (1,1,256,256) | bf16 → bf16 | 1.000000 (exact) | 0.000e+00 | 0.000e+00 | 0.000e+00 | 1.000000 (1.000000..1.000000) |
| (2,3,64,128) | bf16 → bf16 | 1.000000 (exact) | 0.000e+00 | 0.000e+00 | 0.000e+00 | 1.000000 (1.000000..1.000000) |
| (1,1,1024,1024) | bf16 → bf16 | 1.000000 (exact) | 0.000e+00 | 0.000e+00 | 0.000e+00 | 1.000000 (1.000000..1.000000) |
| (1,1,32,32) | fp32 → fp32 | 1.000000 (exact) | 0.000e+00 | 0.000e+00 | 0.000e+00 | 1.000000 (1.000000..1.000000) |
| (1,1,256,256) | fp32 → fp32 | 1.000000 (exact) | 0.000e+00 | 0.000e+00 | 0.000e+00 | 1.000000 (1.000000..1.000000) |
| (2,3,64,128) | fp32 → fp32 | 1.000000 (exact) | 0.000e+00 | 0.000e+00 | 0.000e+00 | 1.000000 (1.000000..1.000000) |
| (1,1,1024,1024) | fp32 → fp32 | 1.000000 (exact) | 0.000e+00 | 0.000e+00 | 0.000e+00 | 1.000000 (1.000000..1.000000) |
| (1,1,32,32) | fp32 → bf16 | 0.999998 | 1.524e-02 | 2.237e-03 | 3.329e-03 | 0.997316 (0.993944..0.999739) |
| (1,1,256,256) | fp32 → bf16 | 0.999998 | 2.929e-02 | 2.249e-03 | 3.335e-03 | 0.997288 (0.994085..0.999724) |
| (2,3,64,128) | fp32 → bf16 | 0.999998 | 1.562e-02 | 2.257e-03 | 3.333e-03 | 0.997281 (0.994075..0.999714) |
| (1,1,1024,1024) | fp32 → bf16 | 0.999998 | 3.118e-02 | 2.245e-03 | 3.314e-03 | 0.997296 (0.994085..0.999729) |
| (1,1,32,32) | bf16 → bf8b | 0.999972 | 2.344e-02 | 6.913e-03 | 9.218e-03 | 1.005714 (0.960361..1.056355) |
| (1,1,256,256) | bf16 → bf8b | 0.999971 | 4.688e-02 | 7.020e-03 | 9.288e-03 | 1.005988 (0.958084..1.053498) |
| (2,3,64,128) | bf16 → bf8b | 0.999972 | 4.688e-02 | 7.017e-03 | 9.259e-03 | 1.005988 (0.960000..1.053498) |
| (1,1,1024,1024) | bf16 → bf8b | 0.999971 | 4.688e-02 | 7.057e-03 | 9.306e-03 | 1.005988 (0.960000..1.053498) |

**Assessment**: the no-cast paths (bf16→bf16, fp32→fp32) are **bit-exact at every shape** — max abs error
0, ratio identically 1.0 — which is the correct and strongest possible result for a byte-permutation op,
and it confirms the fp32 lossless configuration (`fp32_dest_acc_en` + `UnpackToDestFp32` on the fp32→fp32
pair) actually works. Error is shape-independent, as it must be for a permutation. `fp32 → bf16` error is
one bf16 ulp, biased downward (see the truncation advisory above) — a *scale bug was ruled out* via the
ratio spread: the ratios are spread across the ulp band rather than clustered on a constant.
`bf16 → bf8b` error is the block-float shared-exponent quantization, spread symmetrically about 1.0.

**Recommended tolerances**: no-cast pairs — assert **exact** (`max_abs == 0`), do not use a PCC floor.
`fp32 → bf16`: PCC ≥ 0.999, `rtol = 8e-3`, `atol = 4e-2`. `bf16 → bf8b`: PCC ≥ 0.99, `rtol = 6e-2`,
`atol = 5e-2`. Integer dtypes (Refinement 4) must be compared **exactly**.

## Verifier CLI Summary

`eval/eval_test_runner.sh eval/golden_tests/tilize/ <results_dir>` +
`python3 -m eval.verify_supported <results_dir> ttnn.operations.tilize` (re-run after the code fixes;
identical to the pre-fix run):

- supported_pass: **102**
- xfail_expected: **246**  (216 cells on axes not yet built + 30 cells matching the two EXCLUSIONS)
- invalid_skipped: **568**
- xfail_other: **24**  (retile cells, `skip`ped by the harness — Blackhole-only, correct on this Wormhole box)
- **supported_fail: 0** ✓
- **xpass_drift: 0** ✓
- **xfail_wrong_mode: 0** ✓
- supported_marked_xfail: 0 · invalid_unexpected: 0 · infeasible_skipped: 0
- no_axes_found: 949  (the non-registry files in the golden dir: the hidden grader, `test_regression.py`,
  `test_translated.py` — they declare no axes, so `verify_supported` cannot categorize them)

Acceptance suite: `scripts/run_safe_pytest.sh --run-all tests/ttnn/unit_tests/operations/tilize/` →
**95 passed** (52 acceptance + 9 regime pins + 17 lever pins + 17 precision-baseline cells, incl. the
4 new guard tests).

Whole golden directory (informational): 344 passed / 155 failed / 2 errors / 884 skipped, and **every one
of the 155 failures is an `UnsupportedAxisValue` from `validate()`** — 106 sharding, 29 integer dtype,
8 rank-0/misc — i.e. axes the queue covers, not defects. The 2 errors are the grader-harness friction
described above.

## Recommendations

1. **Refinement priority is dominated by one axis group: sharding.** 106 of the 155 whole-directory
   failures and roughly 150 of the 246 xfail cells are blocked on `shard_api` / `out_scheme` /
   `orientation`. It is also the hardest tier, which is why it takes both of the first two queue slots
   (same-spec + crossover, then the cross-spec gather).
2. **Do not let the easy axes jump the queue.** `uint32` / `uint8` / tiny tiles look cheap, but landing
   them before the sharded kernels exist means re-extending them over those kernels afterwards. The
   ordering in `op_requirements.md` is deliberate (hardest-first, with a true-dependency override only).
3. **Cross-spec reshard is only partly expressible on the current tagger set** (details in Refinement 2's
   notes): a legacy→legacy *scheme* change projects to the same axis tuple as the same-spec cell, so it
   cannot be fenced off with an `EXCLUSIONS` dict. If a future run wants that gate, the sanctioned fix is
   a new `reshard` tagger — justified because the kernel genuinely has two code paths — added to
   `INPUT_TAGGERS` and `SUPPORTED` together.
4. **L1 / memory-pressure observation (no OOM today).** The per-core CB footprint is bounded by
   `CB_L1_BUDGET = 1 MiB` of the 1.5 MiB L1 by construction, and `derive_blocking()` shrinks `WT_CHUNK`
   to hold it — verified up to `W = 65536`. The one place the bound is *not* yet enforced is the sharded
   path that does not exist yet: a wide HEIGHT shard hands the kernel a full-width per-core block, which
   is why the wide-W chunking is written into Refinement 1's goal rather than left to be discovered by an
   OOM. Also note the depth-2→depth-1 fallback loop is currently unreachable (the ceiling inside
   `derive_blocking()` already guarantees the fit); it is kept as a genuine safety net for the sharded
   path, where `WT_CHUNK` will be pinned by the shard rather than chosen.
5. **Numerical-precision concern without a lever in scope**: the fp32→bf16 truncation bias (advisory
   above). Not queued — there is no exposed pack-rounding knob and no failing cell to move.
6. **Perf state is honest and measured**, not asserted: DM-bound classification via ablation, a ceiling
   bracket per shape, 9 levers with measured on/off deltas, and 7 structurally-closed levers each pinned
   by a passing test. The remaining headroom is real and named (A3+B10, ~35% predicted), which is what
   Refinement 3 spends, and the completeness audit (Refinement 6) closes the ledger.
