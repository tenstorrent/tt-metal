# Verification Report: rms_norm

Verified 2026-08-19 on **blackhole p150b** (13×10 compute grid). Build: `cmake --build build_Release`
with `TTNN_BUILD_TESTS=OFF` / `TT_METAL_BUILD_TESTS=OFF` — the full `./build_metal.sh` fails on
unrelated *nuked-op* gtest sources (`test_graph_basic.cpp` → `normalization/softmax/softmax.hpp`,
`test_conv2d.cpp` → `conv/conv2d/conv2d.hpp`), which are test targets only. `_ttnn.so` links clean.

---

## Code Review

### Fixed

1. **DRY violation — the CB set was described twice.**
   `_working_set_bytes()` (the L1 budget solver) and `create_program_descriptor()` each carried their
   own independent expression of every CB's page count. They happened to agree today, but a knob turn
   (`BLOCK_HT`, `IN_BUF_DEPTH`, `WT_*_BLOCK`, a new CB) that landed in only one of them would drift
   silently: the solver would believe a plan fits L1 that the descriptors then over-allocate (OOM) or
   under-use. This is precisely the "same block value restated in two places" failure the blocking
   model forbids.
   **Fix:** introduced `_cb_layout(...)` as the single description — it returns
   `[(cb_index, num_pages, page_bytes, kind)]`. `_working_set_bytes()` is now a one-line **sum** over
   that list, `BlockingPlan` carries the final list as a frozen `cb_layout` field, and
   `create_program_descriptor()` **instantiates exactly that list** (`kind` → dtype via one small
   map). No page count is derived in two places any more; adding or resizing a CB is one edit.

2. **`cb_sumsq` was over-allocated in Regime A.**
   It was unconditionally `2 * BLOCK_HT` pages. The double generation exists only because Regime B
   accumulates across `W`-chunks through that CB (`Accumulate::at(cb_sumsq, c)`); Regime A writes it
   once per row-block and pops it in the same iteration. Now `(2 if regime == "B" else 1) * BLOCK_HT`
   — exact, and it slightly widens Regime A's L1 reach (Regime A residency is what keeps wide shapes
   on the single-read path).

3. **Output-placement hole in `validate()`.**
   The `memory_layout` axis was tagged from the *input* tensor only, so an interleaved input with an
   explicit sharded `memory_config=` request passed validation, allocated a sharded output, and would
   have been written through an interleaved `TensorAccessor` — silent corruption rather than an honest
   refusal. `validate()` now also gates `memory_config.memory_layout` against
   `SUPPORTED["memory_layout"]` (raising `UnsupportedAxisValue`), between the per-axis SUPPORTED loop
   and the EXCLUSIONS loop. No golden category moved: sharded golden cells still trip the *input*
   axis check first, exactly as before.

### Attempted and reverted (recorded so it is not re-attempted blind)

4. **Amortizing `tilize` init across the ROW_MAJOR-gamma ingest loop.**
   `op_design.md` prescribes `InitUninitMode` to amortize LLK init across back-to-back `tilize` calls,
   and `ingest_gamma()` issues `N / GAMMA_INGEST_BLOCK` chunks with nothing between them — apparently
   the textbook case. Implemented as `InitOnly` / `Neither` / `UninitOnly` and it is **numerically
   wrong here**: the chunks are separate `cb_reserve_back` / `cb_push_back` groups on
   `cb_gamma_tiles`, and skipping the per-call init corrupts every chunk after the first
   (PCC 0.2356 on `(1,1,32,4096)` TILE, −0.0184 on the same shape ROW_MAJOR, 0.0035 on
   `(1,1,32,16384)`). Reverted to `InitAndUninit` and the reason is now a comment in the kernel so the
   next reader does not repeat it. Cost is bounded and small: 1–16 extra inits per gamma ingest, which
   in Regime A happens **once per kernel**.

### Reviewed and found correct (no change)

- **Registry surface**: `INPUT_TAGGERS` (both taggers carry the `(inputs, axes)` signature),
  `SUPPORTED`, `EXCLUSIONS`, `validate()` in the right order, `validate()` as the entry point's first
  statement. No `INVALID` symbol in the op file.
- **Design conformance**: algorithm (two pinned regimes — resident fused `sum_of_squares` vs streaming
  masked `square`+accumulating `reduce`) matches `op_design.md`; RISC ownership matches (reader NCRISC
  /NoC0, writer BRISC/NoC1, `compute` all-helper); `1/W_true` and `epsilon` are carried as fp32 bit
  patterns through `MulUnary`/`AddUnary` and never through the mandatory-bf16 scaler (R2); the scaler
  is `PoolType::SUM` with value exactly `1.0`.
- **Performance conformance — the machine is actually filled.** `Rt` is split over the whole grid via
  `ttnn.split_work_to_cores(grid, ceil(Rt/BLOCK_HT), row_wise=True)`; `(1,1,8192,1024)` runs on
  **128 of 130 cores**. Both dataflow halves are batched: the reader issues a whole row-block of tile
  reads then **one** `noc_async_read_barrier`, and the writer mirrors it (measured lever B7 = 1.33× on
  `grid_filling`, 3.56× on `grid_starved`). CBs are double-buffered when the budget affords it
  (`IN_BUF_DEPTH` up to 4).
- **Blocking-model fidelity**: every knob the planner named (`BLOCK_HT`, `WT_REDUCE_BLOCK`,
  `WT_SCALE_BLOCK`, `DEST_BLOCK`, `IN_BUF_DEPTH`, `OUT_BUF_DEPTH`, `RM_BUF_DEPTH`,
  `GAMMA_INGEST_BLOCK`, `ACTIVE_CORE_CAP`) is a live parameter emitted by the one `blocking_plan()`
  and threaded to the kernels as CT args; loop trip counts in the kernels are *derived*
  (`NUM_REDUCE_CHUNKS = Wt_core / WT_REDUCE_BLOCK`), never restated.
  **Not a collapsed knob**: the `Wt_core`-sized CBs (`cb_gamma_tiles`, `cb_normed`, Regime-A
  `cb_input_tiles`) exist only under the Regime-A predicate, whose whole content is "this working set
  was *proved* to fit L1", with Regime B's `WT_*_BLOCK`-bounded streaming as the fallback — the
  sanctioned predicate-guarded residency fast-path.
  **Not a half-turned split**: the per-core compute loop runs the *whole* block
  (`IterationShape::grid(BLOCK_HT, Wt_core)`, chunked only when L1 forces it), and the chunk search is
  over **divisors of `Wt_core` taken coarsest-first** — 1 is only ever the search's output. Measured:
  forcing the chunk to one tile (`coarse_chunk=0`) is 3.68× slower on `(1,1,32,7168)`.
- **Helper coverage**: no raw-LLK compute anywhere; `compute_kernel_hw_startup` is the chain's
  documented caller-init contract. The two dataflow helper substitutions (`read_sticks_for_tilize`,
  `write_sticks_after_untilize`) are justified in-file with concrete mismatches — the RM reader must
  zero-fill the pad tail (which is exactly what makes Regime A's `maskless_w` predicate valid) and
  both need this op's `(row-block × W-chunk)` iteration order plus the phantom-row clamp, neither of
  which the helpers' own block loops can express. `mcast_pipe.hpp` is correctly *not* used: Phase 0
  has no inter-core communication at all.
- **Correctness mechanics**: CB push/wait counts balance in both regimes (re-derived by hand per CB);
  `TensorAccessor` everywhere (no `InterleavedAddrGen`); `void kernel_main()`; includes are
  `api/dataflow/dataflow_api.h`.
- **Broadcast dims**: `BroadcastDim::Col` for `x * (1/rms)` (a `REDUCE_ROW` output is Col0-valid) and
  `BroadcastDim::Row` for `* gamma` (a 1-D `[W]` operand is Row0-valid). Confirmed empirically by the
  non-square shapes in the acceptance suite and by the ratio-spread measurement below — a swapped
  Col/Row is the design's named first-run bug and it is not present.

### Deferred (architectural — filed as refinements, not silently dropped)

- **Lamp L4 (fuse the two pass-B multiplies, eliminate `cb_normed`)** rests on
  `DestReuseBinary`'s reuse semantics, which take a plain `InputSpec` with no `BroadcastDim` field, so
  it needs a boot-time gamma broadcast-materialize pass and an extra CB. Real kernel rework →
  Refinement 4.
- **Lamp L1 (cross-core combine over the dependent `Wt` axis)** and **Lamp L3 (`HEIGHT_SHARDED`
  zero-copy CB)** are new topologies / placements → Refinement 2.

---

## Registry Conformance

Confirmed in `ttnn/ttnn/operations/rms_norm/rms_norm.py`:

- `INPUT_TAGGERS = {"alignment": tag_alignment, "rank": tag_rank}` — both `(inputs, axes)`.
- `SUPPORTED` names **every** axis `feature_spec.TARGET` enumerates, including the shape-derived ones
  and the `"none"` sentinel on `gamma_dtype` / `gamma_layout`.
- `EXCLUSIONS = [{"dtype": float32, "fp32_dest_acc_en": False}]` — a cell strictly inside
  `cartesian(SUPPORTED)`; refusing fp32 activations with a bf16 DEST accumulator is a deliberate
  native refusal, not a gap.
- `validate()` checks per-axis `SUPPORTED` → output-placement gate (new, §Code Review 3) →
  `EXCLUSIONS`, raising `UnsupportedAxisValue` / `ExcludedCell` from
  `ttnn.operations._op_contract`. It is the entry point's first statement.
- The op file does **not** declare `INVALID`. ✓

**Auto-fixes applied to SUPPORTED from XPASS evidence: none were needed** — `xpass_drift` is 0.

### INVALID audit (`eval/golden_tests/rms_norm/feature_spec.py`) — 8 entries to revisit

The structural core is well-formed:

- ✓ `{dtype: bfloat8_b, layout: ROW_MAJOR}` — the canonical activation entry, present.
- ✓ `{gamma_dtype: bfloat8_b, gamma_layout: ROW_MAJOR}` — the same impossibility on the weight tensor.
  Single-tensor coupling on both.
- ✓ The no-weight canonicalization is complete in both directions (`gamma present ⇒ real
  dtype/layout`, `gamma absent ⇒ sentinel only`), so exactly one `("none","none")` cell survives.

Two problems, both in the block the file itself labels *"Author-scoped exclusions ('for now', NOT
structural impossibility)"*:

1. **Cross-tensor coupling — the canonical authoring mistake (3 entries).**
   `{layout: ROW_MAJOR, memory_layout: *_SHARDED, gamma_layout: TILE}` couples the **activation's**
   layout and placement to the **weight's** layout. There is no kernel-level coupling that makes a
   tiled gamma impossible under an RM sharded activation — gamma is ingested through its own
   `cb_gamma_rm`/`cb_gamma_tiles` path that is independent of the activation's placement.
   **Recommend removing** all three.
2. **"Not yet supported" wearing INVALID's clothes (5 entries).**
   The same 3 above plus `{dtype: bfloat8_b, alignment: w_non_aligned}` and
   `{... h_non_aligned}` encode *op* limitations, not universe-would-have-to-change facts. Those
   belong in the op file's `EXCLUSIONS` (where a refinement can retire them), or simply on the
   SUPPORTED ladder. Parking them in INVALID makes the harness **skip** ~33.9 k cells that would
   otherwise be honest xfails, which understates the real refinement surface.
   Concretely, it is these two bf8b entries that produce the **2 `invalid_unexpected`** results below.

I did not edit `feature_spec.py` (out of the verifier's remit) — please update via `/golden-tests`
or directly.

---

## Precision Baseline

`tests/ttnn/unit_tests/operations/rms_norm/test_rms_norm_precision_baseline.py`, TILE input, gamma
present (ROW_MAJOR), `epsilon=1e-6`, default (maxed-out) compute config. PCC via `assert_with_pcc`,
`Max ATOL/RTOL` via `comp_allclose`.

| Shape | dtype | PCC gate | Max Abs Err | Mean Abs Err | Relative RMS Err | ratio median | ratio p5..p95 |
|---|---|---|---|---|---|---|---|
| (1,1,32,64) | bfloat16 | ≥0.995 ✓ | 0.0438 | 1.66e-3 | 3.54e-3 | 0.99995 | 0.9944 .. 1.0055 |
| (1,1,32,64) | float32 | ≥0.999 ✓ | 0.0135 | 7.89e-4 | 1.48e-3 | 0.99878 | 0.9975 .. 0.9999 |
| (1,1,128,512) | bfloat16 | ≥0.995 ✓ | 0.0554 | 1.71e-3 | 3.34e-3 | 0.99998 | 0.9945 .. 1.0055 |
| (1,1,128,512) | float32 | ≥0.999 ✓ | 0.0224 | 8.00e-4 | 1.46e-3 | 0.99880 | 0.9975 .. 0.9999 |
| (1,1,32,72) *(W non-aligned → Regime B)* | bfloat16 | ≥0.995 ✓ | 0.0432 | 1.96e-3 | 3.45e-3 | 1.00001 | 0.9944 .. 1.0053 |
| (1,1,32,72) *(Regime B)* | float32 | ≥0.999 ✓ | 0.0223 | 9.43e-4 | 1.50e-3 | 0.99875 | 0.9975 .. 0.9999 |
| (1,1,2048,4096) | bfloat16 | ≥0.995 ✓ | 0.0998 | 1.72e-3 | 3.32e-3 | 1.00010 | 0.9946 .. 1.0056 |
| (1,1,2048,4096) | float32 | ≥0.999 ✓ | 0.0286 | 6.58e-4 | 1.23e-3 | 0.99906 | 0.9977 .. 1.0002 |

**Assessment.** Errors are flat in shape and in `W` — no accumulation-order drift as the reduced axis
widens (the fp32 DEST accumulator plus the `accumulate-then-finalize` reduce shape is doing its job).
bf16 sits at ~0.33 % relative RMS, i.e. **below one bf16 quantization step** (0.39 %), which is the
floor for the format. fp32 sits at ~0.14 %.

**Scale-bug triage (the signature this op is most at risk of).** fp32 shows a *tight* ratio cluster at
a **non-1.0 median (0.9988, spread 7e-4)** — the exact shape the triage table calls out. It is **not**
a kernel scale bug:
- it is **0.12 %**, whereas the padding-fold bug it would otherwise indicate is `sqrt(W_pad/W) − 1`
  = **15.5 %** on the `(1,1,32,72)` shape measured here;
- that non-aligned shape's ratio is **identical** to the aligned shapes' (0.99875 vs 0.99878), so the
  denominator provably counts only real elements (risk R1 clear), and the bf16 rows sit at 1.00000;
- the magnitude and the sign match the FPU datapath: `x`, `1/rms` and `gamma` each round-trip through
  SrcA/SrcB as **tf32** (11-bit mantissa, truncating toward zero), biasing each multiply's magnitude
  down by up to 2⁻¹¹, so three truncations ≈ 0.1 % low. bf16's own round-to-nearest quantization is
  unbiased and larger, which is why bf16's median is 1.0000 with a wide spread.

The baseline test asserts `|ratio_median − 1| < 0.02`, so a genuine scale/padding regression trips
loudly and immediately.

**Recommended tolerances:** `PCC ≥ 0.999` (fp32) / `≥ 0.995` (bf16) — the golden suite's existing
gates are correct. `rtol = 2e-2`, `atol = 2e-2` for bf16; `rtol = 5e-3`, `atol = 5e-3` for fp32.

---

## Verifier CLI Summary

`eval/eval_test_runner.sh eval/golden_tests/rms_norm/ /tmp/rms_verify2` →
`python3 -m eval.verify_supported /tmp/rms_verify2 ttnn.operations.rms_norm`
(752/40828 passed, **0 failed, 0 errors, 0 hangs**). Trimmed report committed as
`verifier_report.json`.

- supported_pass: **737**
- xfail_expected: **6172**
- invalid_skipped: **33900**
- infeasible_skipped: 2 *(uncharged — `(99991,64)` WIDTH_SHARDED shard geometry vs this device's L1)*
- no_axes_found: 15 *(all `test_regression.py`, all passed — that file declares no axes)*
- **supported_fail: 0** ✓
- **xpass_drift: 0** ✓
- **xfail_wrong_mode: 0** ✓
- supported_marked_xfail: 0 ✓
- invalid_unexpected: **2** — see below

**The 2 `invalid_unexpected`** are
`test_translated.py::test_rms_norm_sharded_uneven_multicore_logical_width[{w72_c2,w200_c3}_nonaligned-bfloat8_b]`.
They match `INVALID`'s author-scoped `{dtype: bfloat8_b, alignment: w_non_aligned}` entry, but
`test_translated.py` does not apply the INVALID skip gate, so they *ran* and xfailed. **The op behaved
correctly** — `validate()` refused bf8b with `UnsupportedAxisValue`. This is a harness/feature-spec
authoring issue (see the INVALID audit, item 2), not an op defect, and it clears the moment those two
non-structural entries move out of INVALID.

**Every `xfail_expected` cell is accounted for.** `TARGET − SUPPORTED` decomposes into exactly four
missing axis values, each routed to a refinement:

| Missing (axis, value) | xfail cells | Routed to |
|---|---|---|
| `fp32_dest_acc_en = False` | 3 607 | Refinement 1 |
| `dtype = bfloat8_b` | 962 | Refinement 1 |
| `gamma_dtype = bfloat8_b` | 860 | Refinement 1 |
| `memory_layout = WIDTH_SHARDED` | 1 622 | Refinement 2 |
| `memory_layout = BLOCK_SHARDED` | 1 619 | Refinement 2 |
| `memory_layout = HEIGHT_SHARDED` | 1 593 | Refinement 2 |

(Counts are per missing value and overlap — a cell missing two values is counted under both; the
6 172 `xfail_expected` cells are covered by exactly these six.)

No axis value is omitted and none is undocumented.

---

## Recommendations

1. **`fp32_dest_acc_en=False` is on the critical path, not a nicety.** *Every* perf loose case in
   `feature_spec.LOOSE_CASES` — including the mandatory `(1,1,32,7168)` `minimum_expected_speedup=7.0`
   gate — runs at `bf16 / HiFi2 / fp32_dest_acc_en=False`, which today is **xfail**. Until Refinement 1
   lands, no perf pass can measure the real target config (only a `fp32_dest_acc_en=True` stand-in,
   which is a different datapath and a different DEST capacity). That is why Refinement 1 is first.
   It also doubles DEST (4 → 8 tiles), so `DEST_BLOCK` / `BLOCK_HT` / `IN_BUF_DEPTH` must be
   re-swept afterwards — the existing `levers=dict(block_ht=…, dest_block=…)` arms make that a
   measurement, not a rewrite.
2. **The design's "movement-dominated in every regime" classification is refuted in one regime, and
   it matters.** The implementer's stub ablation measures `(1,1,32,7168)` at ≥58.6 % compute on one
   core. Perf work on the grid-starved decode shapes must therefore parallelize *compute*, not only
   shave bytes — which is what makes Lamp L1 (Refinement 2/3) the only route to the 7× gate, and what
   makes `math_fidelity` (measured 1.45× at HiFi2 on that shape) a real lever there.
3. **L1 pressure that has not yet OOM'd but will move.** `cb_gamma_tiles` + `cb_normed` +
   Regime-A `cb_input_tiles` all scale with `Wt_core`. The regime predicate keeps them safe today, but
   it also means every widening of Regime A's reach (Lamp L4) trades directly against that budget.
   The budget is now summed from `_cb_layout` so re-checking it after a knob change is exact.
4. **`GAMMA_STAGE_MAX_BYTES = 64 KiB` is a hard-coded policy constant**, not a solved budget: the
   RM-gamma staging CB is capped, then `GAMMA_INGEST_BLOCK` is the largest divisor of
   `WT_SCALE_BLOCK` under it. It works and it is bounded, but it is the one blocking-ish number in
   `blocking_plan()` that is not derived from the L1 budget. Worth folding into the solver when
   Refinement 1 re-sweeps the knobs. Not a refinement on its own (no cell moves).
5. **`ACTIVE_CORE_CAP` is dead weight at Phase 0 and should stay `None`.** The measured core sweep on
   `grid_filling` (full grid 93.7 µs, 96c 96.0, 64c 94.5, 32c 110.1, 16c 106.0) shows **no bandwidth
   knee below the full grid**, refuting the design's lever-A0 prediction that capping cores would pay.
   Keep the knob (it is free and re-measurable via `levers=dict(active_cores=N)`), but do not spend a
   refinement on it.
6. **`_levers` on the public signature.** `rms_norm(..., _levers=dict(...))` is an internal bench hook
   with a documented underscore. It is honest and it is what makes every lever counterfactual
   re-runnable without editing a kernel, so I left it — but it should not grow, and it must never be
   read by `validate()`.
7. **Non-regression baseline for later phases.** Re-measured after this pass's edits (Tracy, device
   kernel duration): `grid_filling` 93 656 ns, `wide_prefill` 1 019 959 ns, `grid_starved` 76 161 ns,
   `smallest` 3 221 ns, `row_major` 95 881 ns — all within ±1 % of the Phase 0 numbers, i.e. the
   verifier's edits are perf-neutral. Every later phase must re-measure **all five**.
