# Verification Report: tilize

Verified 2026-07-29 against `ttnn/ttnn/operations/tilize/` at Phase 0 (single implementer pass,
`incremental-implementer`). Box: WH B0, 8×8 compute grid, AICLK ≈ 985 MHz.

Artifacts produced by this pass:

| Artifact | Path |
|---|---|
| verifier CLI report | `ttnn/ttnn/operations/tilize/verifier_report.json` (copy of `/tmp/tilize_verify_final2/verifier_report.json`) |
| precision baseline test | `tests/ttnn/unit_tests/operations/tilize/test_tilize_precision_baseline.py` |
| extended coverage test | `tests/ttnn/unit_tests/operations/tilize/test_tilize_extended.py` |
| refinement queue | `ttnn/ttnn/operations/tilize/op_requirements.md` |

---

## Code Review

### Fixed in this pass

| # | What | Why | Evidence after the fix |
|---|---|---|---|
| 1 | **Writer: per-block `noc_async_write_barrier()` → `noc_async_writes_flushed()` + one trailing barrier** (`kernels/tilize_writer.cpp`) | Recycling a CB page only requires the writes to have *departed* (data read out of L1) — `dataflow_api.h:1802` documents exactly that semantic. A full barrier per block idled BRISC for the destination round-trip of every block's last tile. The single barrier after the loops still guarantees completion before the kernel ends. | Re-measured full bench: `c_single_core` **31 465 → 30 472 ns (−3.2 %)**, `x_wide_short_1core` **61 526 → 57 459 ns (−6.6 %)**; every 64-core regime inside the ±2 % noise floor; **no regression anywhere**. The payoff is concentrated where the barrier is not hidden by other cores' traffic. |
| 2 | **Planner: depth-2→depth-1 L1 fallback moved *before* chunk-width selection** (`tilize_program_descriptor.py`) | The fallback was a post-hoc clamp that could never fire: `chunk_wt` was already bounded by `max_chunk_l1`, so `depth*chunk_wt*bytes ≤ budget` held by construction. The prompt's "auto-fall-back to depth-1 rather than OOM" rule was therefore satisfied by accident, in the wrong place. Now the decision is at planner step 2 (as `op_design.md` specifies) and the old clamp is an explicit invariant `assert`. | Behaviour identical (`x_square_depth1` 65 536 B/core, `a_square` 131 072 B/core unchanged); 93/93 unit tests pass; the invariant is now checked on every build. |
| 3 | **Plan key `shard_row_bytes` → `source_page_bytes`** | On the generic path the value is `input_tensor.buffer_page_size()`, not a shard row; the reader kernel already calls its CT arg `source_page_bytes`. Misleading name on the one code path (sharded RM input, `row_page_stride > 1`) that is hardest to reason about. | Rename only; all tests pass. |
| 4 | **Reader: literal `32` → `constexpr tile_height`** (`kernels/tilize_reader.cpp`) | Three magic 32s in the raw strided fallback, one of which is the tile-row block size and two of which are row counts derived from it. | Cosmetic; identical codegen. |
| 5 | **`SUPPORTED["rank"] = [2,3,4]` → `[2,3,4,5,6]`** | Under-claim: the program folds all leading dims into one row axis (`H % 32 == 0` guarantees no leading dim straddles a tile), so it is rank-agnostic by construction — a rank-5 caller was refused by a contract the kernel already satisfies. | Probed, then locked in by `test_tilize_extended.py::test_tilize_high_rank`: rank 5 and rank 6, single- and multi-core, **bit-exact**. |
| 6 | **`PROPERTIES["bounded_cb"]`: `declared` → `verified`** | The claim now has automated evidence (see #7 and the new plan-invariant test), so the weaker tag was stale. | `test_tilize_extended.py::test_tilize_plan_invariants` asserts `cb_bytes_per_core ≤ L1_CB_BUDGET_BYTES` for `Wt ∈ {5, 256}` and the 64×64 square; the bench asserts it for every regime. |
| 7 | **Bench: A0 and bounded-CB gates are now `assert`ed, not printed** (`_bench_tilize.py::_assert_structural_gates`) | `op_design.md` and `changelog.md` both claim the A0 active-core gate is "machine-checked, not eyeballed" — it was a print column. A height-only-split regression on the wide-short regime would have gone unnoticed by the bench itself. | Bench passes with the asserts on all 14 regimes (interleaved → `min(grid_cores, total_tiles)`, alias → the shard's own cores, forced single-core → 1). |
| 8 | **Bench: 3 new regimes** — `g_dram_to_sharded`, `g_sharded_to_dram`, `e_square_fp32_to_bf16` | The two crossover directions and the narrowing-cast path had **no measured baseline at all**, so the refinements that target them (queue R3, and the `Fp32Mode` question) had nothing to be gated on. | Measured (see changelog "verification re-measure"): 18 923 ns / 19 780 ns / 120 813 ns. `g_sharded_to_dram` immediately exposes `chunk_wt = 2` ⇒ 128 B read transactions (B6 threshold is 512 B). |

### Confirmed correct (no change needed)

- **CB synchronisation ledger balances on both paths.** Generic: reader pushes `chunk_count × row_count × chunk_wt` pages, compute waits `chunk_wt` × `row_count·chunk_count` blocks, writer waits/pops the same — identical totals *and* identical per-wait page counts. Alias: reader pushes `shard_tiles` once, compute waits `chunk_wt` × `shard_h/32` blocks (= `shard_tiles`), writer waits/pops `shard_tiles` once. The output aliased CB's capacity is exactly `shard_tiles`, so the last `reserve_back` has precisely `chunk_wt` free pages — tight but correct.
- **Reader/writer nesting order** (`op_design.md` Risk #2, the silent-corruption one): the reader's chunk loop is outer and `read_sticks_for_tilize`'s tile-row loop is inner; the writer computes `base_page = (row_start + r) * Wt + (chunk_start + c) * chunk_wt` with `c` outer and `r` inner. Matches.
- **Helper usage.** Compute uses `compute_kernel_lib::tilize<chunk_wt, …, InitAndUninit, WaitBlock, RECONFIG, FP32MODE>` (symmetric mode, one call for all blocks — the fused form, not per-block calls); read uses `dataflow_kernel_lib::read_sticks_for_tilize<TILE>` (32 reads per barrier, lever B7). The **writer's raw-API substitution is justified and re-verified**: the only write helper in `kernel_lib`, `write_sticks_after_untilize`, emits ROW_MAJOR *sticks* (`tilize_helpers_dataflow.inl:232-236`) — it is the untilize partner and would scatter tile bytes to stick addresses. No `kernel_lib` helper moves TILE pages from a CB to a `TensorAccessor`-addressed buffer. No multicast/semaphore code exists in this op (every core's read and write set is disjoint), so `mcast_pipe.hpp` does not apply.
- **API hygiene.** `TensorAccessor` + CT `TensorAccessorArgs` everywhere (no `InterleavedAddrGen`); `void kernel_main()`; `#include "api/dataflow/dataflow_api.h"` (not the bare path).
- **Broadcast efficiency**: n/a — zero arithmetic, no broadcast CBs, no repeated-value fills.
- **Program-cache re-binding on the zero-copy path** (`op_design.md` Risk #10) — probed, not assumed: two consecutive same-spec sharded calls hit the cache (`num_program_cache_entries` 1 → 1) with **different** input/output buffer addresses, and both results are bit-exact (and the first result is untouched by the second call). `apply_descriptor_runtime_args` re-patches the aliased CB address from `cb_desc.tensor` on every call (`program_descriptors.cpp:198-209`).
- **Design conformance** on the binding dimensions: algorithm (FPU tilize, candidate C1), pipeline topology (reader NCRISC/NoC0 → compute TRISC → writer BRISC/NoC1; two CBs, no intermediate), work distribution (2D height-first rectangular split, one compile-time `chunk_wt`, all per-core variation in RT args), inter-core communication (none by construction). Two advisory deviations remain (crossover one-sided aliasing; split reader) — both are `op_design.md` R3b/R3c items and are queued as Refinement 3, not silently dropped.

### Prompt-rules audit (`eval/prompts/tilize.txt` § Rules)

| Rule | Verdict |
|---|---|
| MUST parallelize wide-short tensors (width-blocking, 2D split) | **Satisfied.** `[1,1,32,16384]` (`nt_h=1`) runs on **64** cores, 13 383 ns vs 57 459 ns forced-1-core (4.3×). Now `assert`ed by the bench (fix #7) and by `test_tilize_plan_invariants`. |
| MUST tilize natively in-kernel; MUST NOT call `ttnn.to_layout`/`ttnn.tilize` in Python | **Satisfied.** Entry point builds a `ProgramDescriptor` and dispatches `ttnn.generic_op`; no manipulation-op wrapper anywhere. |
| MUST do a real value-preserving cast at pack time; declare `output_dtype` in SUPPORTED and thread it into `validate()` | **Satisfied.** `output_dtype` is a SUPPORTED axis and a `validate()` argument; the cast happens at pack (output CB format = output dtype, `needs_cast` drives the reconfigure). Precision baseline below shows the widening/identity transitions are bit-exact and the narrowing ones are 1-ULP. |
| MUST select `NoReconfigure` when there is no cast | **Satisfied.** `needs_cast` CT arg selects `NoReconfigure` vs `UnpackAndPackReconfigure` with `if constexpr`. |
| MUST expose `use_double_buffer` and auto-fall-back to depth-1 rather than OOM | **Satisfied** (and the fallback is now expressed at the right place — fix #2). `double_buffer` is a SUPPORTED axis with both values; depth-1 halves per-core CB L1 (131 072 → 65 536 B at `chunk_wt=16`). |
| MUST NOT round-trip through host / add extra full-tensor DRAM passes | **Satisfied.** One read leg + one write leg; same-spec sharded has zero NoC traffic (proven by the `no_dm == full` ablation). |
| MUST record target / tt-npe pin / measured + used-optimization ledger when claiming a perf win | **Satisfied** by `changelog.md` Phase 0 §2–§4; this pass adds a re-measure table and keeps the same discipline. |

**Advisory (soft guidance, not blocking):** the compute kernel picks `Fp32Mode::Lossless` purely from the *input* CB format, so an fp32 → bf16/bf8b **cast** also pays for the slow tilize path even though the narrower output cannot hold the extra precision (`tilize_helpers.hpp:47-71` calls Lossless "rarely useful"). Measured at grid-filling size this costs nothing: `e_square_fp32_to_bf16` = 120 813 ns for 25.17 MB = 208.3 GB/s, i.e. **0.72 of the 288 GB/s DRAM floor — the same ratio as every other DM-bound 64-core regime**, so the bound is DRAM, not the LLK. It also currently *helps* accuracy (fp32→bf16 agrees with torch RNE on all but ≤3 of 196 608 elements). Left as-is; re-examine only in the low-core / small-shape cast regime, where compute is a visible fraction (see Refinement 4's ledger).

---

## Registry Conformance

- **`INPUT_TAGGERS`** — 6 taggers (`use_multicore`, `shard_api`, `out_scheme`, `buffer`, `rank`, `double_buffer`), all with the `(inputs, axes)` signature, names and sources exactly as `feature_spec.py`'s docstring specifies. ✓
- **`SUPPORTED`** — every tagger key plus the two free cartesian axes (`dtype`, `output_dtype`) is declared. No axis the kernel gates on is missing: input layout (always ROW_MAJOR) and tile-alignment are *structural* preconditions checked before `validate()` with `RuntimeError`/`ValueError`, exactly as `op_design.md` and the immutable acceptance test require — they are not registry axes. ✓
- **`EXCLUSIONS`** — present and empty, with the rationale recorded in the op file: the design's proposed `{use_multicore: False} × sharded` refusal turned out not to be a kernel boundary (the generic `TensorAccessor` path addresses sharded pages from any core count), and the reference suite exercises that exact cell. ✓
- **`validate()`** — checks SUPPORTED per-axis first, then EXCLUSIONS cell-wise; raises `UnsupportedAxisValue` / `ExcludedCell` from `ttnn.operations._op_contract`. Called by the entry point before any device work; the only things ahead of it are the three structural checks and the L1-shard-grid bounds check, all of which the spec requires to raise `RuntimeError`/`ValueError` and which must not be misreported as support refusals. ✓
- **No `INVALID` in the op file.** ✓ (The only occurrence of the word is the docstring line explaining that it lives in `feature_spec.py`.)

### Fix-in-place applied to SUPPORTED

`rank`: `[2,3,4]` → `[2,3,4,5,6]` (code-review under-claim, not XPASS drift — the golden matrix has no rank-5 cell). Backed by a new test, not by inference. No other axis changed; no EXCLUSIONS added (nothing was silenced).

### TARGET vs SUPPORTED — the gap is empty

Computed per axis (`TARGET − SUPPORTED`, before subtracting INVALID):

| axis | missing from SUPPORTED | SUPPORTED beyond TARGET (and why) |
|---|---|---|
| `dtype` | — | `uint16`, `int32` — exercised by the acceptance test's integer-passthrough cases (bit-exact). |
| `output_dtype` | — | `uint16`, `int32` — same. |
| `use_multicore` | — | — |
| `double_buffer` | — | — |
| `shard_api` | — | — |
| `out_scheme` | — | — |
| `buffer` | — | — |
| `rank` | — | `5`, `6` — rank-agnostic fold, verified bit-exact (fix #5). |

`xfail_expected` is **0 entries** (not just count-0): there is no cell in the golden matrix that the op refuses, so there is no `(axis, missing_value)` pair left to file. **The refinement queue is therefore perf-only** (concrete measured levers + the run-closing Mode-D audit), per the hard rule in the verifier contract. No axis value is omitted without a reason — the table above is exhaustive.

### INVALID audit (`eval/golden_tests/tilize/feature_spec.py`)

5 entries, all `dtype` × `output_dtype` int↔float crosses (`uint32 → bf16/fp32/bf8b` and `bf16/fp32 → uint32`).

| Rule | Verdict |
|---|---|
| 1. Single-tensor coupling | **Pass.** tilize is an identity map on values: `dtype` and `output_dtype` are the *storage format of the same logical tensor* before and after the layout change, and the coupling is the op's own definitional contract (a value-preserving cast family), stated in the feature-spec comment. This is not the conv2d-style mistake of crossing axes belonging to two independent tensors. |
| 2. Universe-must-change | **Pass.** An int↔float "cast" is a value *reinterpretation* — a different operation, not a kernel improvement. No amount of kernel work makes `uint32 → bfloat16` be tilize. |
| 3. Canonicalization-only multi-axis exception | n/a — no canonicalization entries are needed (no weight-like axes, no redundant cells). |
| Canonical `bf8b + ROW_MAJOR` entry | **Correctly absent, and documented.** bf8b is not an input dtype here (the input is always ROW_MAJOR and block-float has no row-major form — `tilize_helpers.inl:156` asserts it), and there is no free `layout` axis. bf8b appears only in `TARGET["output_dtype"]`, which is legal and covered. |

**No change requested to `feature_spec.py`.** One caveat worth carrying forward (already flagged by the implementer, re-confirmed here): `helpers.py:run_tilize` never forwards `use_double_buffer`, so the harness tags `double_buffer=False` while the op runs its default `True`. Consequences: (a) `SUPPORTED["double_buffer"]` must keep **both** values or those two cells fail; (b) the depth-1 CB path's only correctness coverage is `test_tilize.py::test_tilize_double_buffer` plus the new `test_tilize_extended.py::test_tilize_sharded_double_buffer` (which covers depth-1 against an *aliased* CB — previously uncovered anywhere).

---

## Precision Baseline

`tests/ttnn/unit_tests/operations/tilize/test_tilize_precision_baseline.py` — 4 shapes × 5 dtype
transitions, 20/20 pass. PCC via `assert_with_pcc`, allclose via `comp_allclose`, plus exact-match
count (the meaningful metric for a value-preserving op) and relative RMS.

| Shape | Transition | PCC | Max Abs Err | Mean Abs Err | Relative RMS Err | Mismatching elements |
|-------|-----------|-----|-------------|--------------|------------------|---------------------|
| (1,1,32,32) | bf16→bf16 | 1.0 | 0 | 0 | 0 | 0 / 1 024 |
| (1,1,32,32) | fp32→fp32 | 1.0 | 0 | 0 | 0 | 0 / 1 024 |
| (1,1,32,32) | bf16→fp32 | 1.0 | 0 | 0 | 0 | 0 / 1 024 |
| (1,1,32,32) | fp32→bf16 | 1.0 | 0 | 0 | 0 | 0 / 1 024 |
| (1,1,32,32) | bf16→bf8b | ≥0.99 | 2.344e-02 | 7.397e-03 | 9.611e-03 | 829 / 1 024 |
| (1,1,64,128) | bf16→bf16 | 1.0 | 0 | 0 | 0 | 0 / 8 192 |
| (1,1,64,128) | fp32→fp32 | 1.0 | 0 | 0 | 0 | 0 / 8 192 |
| (1,1,64,128) | bf16→fp32 | 1.0 | 0 | 0 | 0 | 0 / 8 192 |
| (1,1,64,128) | fp32→bf16 | 1.0 | 0 | 0 | 0 | 0 / 8 192 |
| (1,1,64,128) | bf16→bf8b | ≥0.99 | 2.344e-02 | 7.058e-03 | 9.302e-03 | 6 520 / 8 192 |
| (1,1,256,512) | bf16→bf16 | 1.0 | 0 | 0 | 0 | 0 / 131 072 |
| (1,1,256,512) | fp32→fp32 | 1.0 | 0 | 0 | 0 | 0 / 131 072 |
| (1,1,256,512) | bf16→fp32 | 1.0 | 0 | 0 | 0 | 0 / 131 072 |
| (1,1,256,512) | fp32→bf16 | 1.0 | 1.562e-02 (1 ULP) | 1.490e-07 | 4.428e-05 | 2 / 131 072 |
| (1,1,256,512) | bf16→bf8b | ≥0.99 | 4.688e-02 | 7.107e-03 | 9.322e-03 | 104 633 / 131 072 |
| (2,3,128,256) | bf16→bf16 | 1.0 | 0 | 0 | 0 | 0 / 196 608 |
| (2,3,128,256) | fp32→fp32 | 1.0 | 0 | 0 | 0 | 0 / 196 608 |
| (2,3,128,256) | bf16→fp32 | 1.0 | 0 | 0 | 0 | 0 / 196 608 |
| (2,3,128,256) | fp32→bf16 | 1.0 | 1.562e-02 (1 ULP) | 1.788e-07 | 5.041e-05 | 3 / 196 608 |
| (2,3,128,256) | bf16→bf8b | ≥0.99 | 4.688e-02 | 7.101e-03 | 9.323e-03 | 156 863 / 196 608 |

**Assessment.** Every transition whose output format can represent the input exactly is **bit-exact**
on every shape — bf16→bf16, fp32→fp32 (this is what `Fp32Mode::Lossless` + `fp32_dest_acc_en` +
`UnpackToDestFp32` buy) and the bf16→fp32 widening. Integer passthrough (uint32/uint16/int32) is
likewise bit-exact (`test_tilize.py::test_tilize_integer_passthrough`, `torch.equal`). The only
non-zero error sources are the two genuinely lossy casts:

- **fp32→bf16**: at most **3 of 196 608 elements** differ, each by exactly **1 bf16 ULP** — the
  packer's tie-rounding differs from torch's round-to-nearest-even on the occasional exact tie.
  Relative RMS ≤ 5.0e-05, PCC 1.0. Not a layout or addressing effect (bf16→bf16 on the same shapes is
  bit-identical); purely rounding.
- **bf16→bf8b**: shared-exponent block-float quantization. Relative RMS ~9.3e-03 ≈ 2⁻⁷, exactly the
  format's mantissa resolution; max abs error scales with the block's exponent (2.3e-02 on small
  tensors, 4.7e-02 once a block contains a larger magnitude). ~20 % of elements land exactly.

**Recommended tolerances** (these are what the golden suite already uses, and the measurements
support them):

| Transition | Gate |
|---|---|
| bf16→bf16, fp32→fp32, bf16→fp32, int passthrough | **exact** (`comp_equal` / `torch.equal`) |
| fp32→bf16 | PCC ≥ 0.999, `rtol = 1e-2`, `atol = 1e-2` (1 ULP of bf16 at magnitude 4 is 1.6e-2) |
| →bf8b | PCC ≥ 0.99, `rtol = 5e-2`, `atol = 5e-2`; do **not** use allclose at 1e-2 — block-float legitimately fails it |

---

## Verifier CLI Summary

`eval/eval_test_runner.sh eval/golden_tests/tilize/ /tmp/tilize_verify_final2` →
`python3 -m eval.verify_supported /tmp/tilize_verify_final2 ttnn.operations.tilize`

- `supported_pass`: **126**
- `xfail_expected`: **0** (SUPPORTED covers the whole TARGET matrix — nothing left to refuse)
- `invalid_skipped`: **90**
- `supported_fail`: **0** ✓
- `xpass_drift`: **0** ✓
- `xfail_wrong_mode`: **0** ✓
- `supported_marked_xfail` / `invalid_unexpected` / `xfail_other`: **0**
- `no_axes_found`: 420 — the non-registry files in the same directory (`test_translated.py` 276,
  `test_golden_main_tests.py` 107 collected + 28 skipped, `test_regression.py` 9). They carry no axes
  dict by design, so the CLI cannot categorize them; they are reported below instead.

Whole-directory pytest result: **515 passed, 118 skipped, 1 failed, 2 errors, 0 hangs**. Registry
matrix (`test_golden.py`): **126 passed / 90 INVALID-skipped / 0 failed / 0 xfail / 0 xpass**.

The 3 non-passing items are **all outside the registry matrix and none is caused by the op**:

| Item | Diagnosis |
|---|---|
| `test_translated.py::test_tilize_width_sharded_dram_input_to_l1_sharded_output_49107` (failed) | The test derives an **L1** output shard grid from `device.dram_grid_size().x` (= 12 on WH B0) while the compute grid is 8×8, i.e. it asks for an L1 shard on core (11,0), which does not exist on this box (it is valid on a ≥12-column grid such as Blackhole). The op refuses it up front with a clean `ValueError` from `_check_l1_shard_grid`. That guard is load-bearing: without it the failure surfaced later inside `TensorAccessorArgs::get_compile_time_args()` *after* the output buffer was allocated, wedging the command queue and aborting the whole file. Keep the guard; the reference test needs a device-portability gate (it is unmodifiable from here). |
| `test_golden_main_tests.py::test_deepseek_v3_mla_tilize_trace_mode[…]` × 2 (setup errors) | Internal to the unmodifiable grader file: it combines `@pytest.mark.use_module_device` with a `device_params` parametrize, which `conftest.py:431` rejects. No op code runs. |

Unit-test suites: `tests/ttnn/unit_tests/operations/tilize/` = **93 passed** (60 acceptance + 13
extended + 20 precision), green in **both** default and `--dev` (watcher/asserts) mode — no race, no
watcher trip, no hang.

---

## Recommendations

1. **The queue is perf-only.** SUPPORTED already equals TARGET on every axis, `xfail_expected` is
   empty, and no cell is failing. Per the verifier contract, that means the only legitimate
   refinements left are (a) concrete `master.md` levers with a *measured* off-ceiling gap and (b) the
   single run-closing Mode-D completeness audit. `op_requirements.md` files exactly that: 4 measured
   levers + 1 audit. Do not add capability entries — there is nothing left to add.
2. **Where the remaining perf actually is** (measured, from the re-measure table in `changelog.md`):
   the grid-filling regimes are **at** the achievable-copy ceiling (a_square 196.4 GB/s vs the in-tree
   64-core DRAM→DRAM copy's 193.8 GB/s), so the headroom is *not* there. It is in the
   **low-work-per-core** regimes (`d_tall_narrow` 0.38 of the bandwidth knee, 33 % sync floor;
   `f_sharded_small` 50 % compute + 50 % sync, zero DM) and on the **crossover** paths
   (`g_dram_to_sharded` 0.39 of its DRAM-side floor; `g_sharded_to_dram` running 128 B read
   transactions, 4× under the B6 one-packet threshold). Refinements 1, 3 and 4 target exactly those.
3. **Depth-2 CBs are a measured no-op on the DRAM-bound regimes** (85 417 ns vs 85 806 ns depth-1,
   inside noise) while costing 65 536 B/core. That is the "lever that doesn't pay" the prompt warns
   about; Refinement 1 bundles the gating decision. Note it is *not* a correctness or OOM risk —
   per-core CB L1 is bounded by a constant in `W` by construction, now asserted.
4. **L1 / memory-pressure observation (no OOM today).** The generic path is safe by construction
   (`chunk_wt ≤ WT_CHUNK_MAX = 16`, so ≤ 131 072 B/core for any `W`). The **alias path is not
   bounded**: its CBs *are* the caller's shards, so per-core CB L1 equals the shard size. That costs
   no extra allocation (verified: aliased addresses, no second buffer), but it does mean the tilize
   LLK block width equals `shard_W/32`, and `_alias_eligible` correctly refuses ≥ 256 (falls back to
   the generic path). If a future refinement widens Path B, keep that guard.
5. **Fragility worth knowing, not fixing now**: `_alias_eligible` compares shard grids by
   `str(CoreRangeSet)`. Two textually different but set-equal grids fall back to the generic path —
   correct, just slower. Cheap to tighten if a real case appears.
6. **For whoever owns the harness**: the two `test_golden_main_tests.py` collection errors and the
   `dram_grid_size()`-derived L1 shard grid in `test_translated.py` are reference-file bugs that
   permanently cost 3 non-green results on an 8-column-grid box. Worth a device-portability skip so
   the directory can go fully green.
