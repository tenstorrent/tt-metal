# Verification Report: rms_norm

Date: 2026-08-13 · Device: Blackhole p150b (11×10 compute grid) · Results dir: `/tmp/rms_verify2`

---

## Code Review

Everything below was **fixed in place**; nothing correctness- or conformance-related was deferred.

| # | Finding | Fix |
|---|---|---|
| 1 | **Half-turned data-movement knob on the ROW_MAJOR path.** `DM_CHUNK_TILES` is a *byte-budget* knob ("this many tiles per NoC barrier"), but the ROW_MAJOR reader and writer compared a **stick** counter against it. A stick is `S*32` elements = `S/32` of a tile, so at the common decode geometry (`S = 4`, bf16) the RM path barriered every **2 KB** where the TILE path barriers every **16 KB** — the same knob turned 8× finer on one layout, i.e. a reader/writer that dribbles. | Both kernels now derive `RM_CHUNK_STICKS = clamp(DM_CHUNK_TILES * 32 / S, 1, 32)` from the *same* single knob, so the two layouts batch at one granularity and the knob stays one source of truth. `rms_norm_reader.cpp`, `rms_norm_writer.cpp`. |
| 2 | **`cb_w_mask` allocated on every program**, including the ~⅔ of programs that never touch it (TILE with `W % 32 == 0`, and *all* ROW_MAJOR programs, where the reader's tail zero-fill replaces the mask). `l1_ledger.md` claimed it was conditional, so ledger and code disagreed. | The CB is now created only when `mask_enabled` (`TILE && W % 32 != 0`). This required a new `MASK_ENABLED` compile-time arg in the reader: `prepare_reduce_mask` `static_assert`s on the CB's page *format*, so the call has to sit in a **discarded statement** (`if constexpr`), not merely an untaken runtime branch — the first attempt (runtime-only gate) failed to compile, which is the useful record of why the gate is compile-time. |
| 3 | **`IN_CB_DEPTH` read as a free knob but is load-bearing at 1.** `cb_input_tiles` is rewritten in place twice and that relies on `get_write_ptr() == get_read_ptr()`, which only holds when capacity is exactly one block. Turning the "knob" to 2 would silently write `x·r` into the wrong half — no error, no hang. | Added a module-level `assert IN_CB_DEPTH == 1` with the reason, so the overlap perf lamp is forced into its only correct form ("smaller `block_rows` + a second buffer", never "same block, deeper CB"). `rms_norm_program_descriptor.py`. |
| 4 | ROW_MAJOR staging **stick pitch alignment** was relied on but never asserted. | `static_assert(RM_STICK_PITCH % 16 == 0)` in both dataflow kernels. |
| 5 | `l1_ledger.md` was **out of date against the code** in four places (a row for the `cb_slice_stat` CB that the implementation deliberately does not create; no row for `cb_thread_sync`, which does exist; `cb_gathered_partials` / `cb_rms_bcast` described as "root cores only" when they are declared on every core of the rect; `cb_rm_stage_in` described with stick-sized pages when the code declares tile-sized pages of identical total bytes). | Ledger updated to the implementation, including a real `Shares with / why not` justification for `cb_thread_sync` and an honest symbol-table entry for `l1_working_budget`. |

### Checked and found correct (no change needed)

- **Helper usage.** Every compute phase goes through `compute_kernel_lib` (`tilize`, `untilize`, `sum_of_squares`, `mul`, `reduce`), and the multicast half of the combine uses `dataflow_kernel_lib::McastArgs` → `SenderPipe`/`ReceiverPipe` rather than raw `noc_async_write_multicast` + semaphores. The three raw-API deviations are each justified in-file and each is genuinely helper-less: interleaved DRAM addressing (`TensorAccessor` *is* the sanctioned mechanism), the **gather** (s sources → s different destination pages on one core; `mcast_pipe` is the opposite shape and no gather/scatter helper exists), and the finalize (`×1/W`, `+eps`, `rsqrt`) written into `reduce`'s documented `post_reduce_op` extension point so `mean+eps` is never packed to L1.
- **CB sync.** push count == wait count on every CB, in both layouts and both regimes, including the caller-managed `cb_input_tiles` window (exactly one `cb_wait_front(B*S)` … one pop path per layout).
- **API hygiene.** `TensorAccessor` everywhere (no `InterleavedAddrGen`), `void kernel_main()`, `api/dataflow/...` include paths.
- **Broadcast dims.** `BroadcastDim::Col` for the column-0-valid rsqrt tile, `BroadcastDim::Row` for the 1-D gamma and the W-mask — verified against the design's broadcast table; no CB is filled with repeated full tiles where a broadcast suffices (gamma is row-0-valid, the scaler/mask are 1 page each).
- **Design conformance / does it fill the machine.** The implemented 2D partition matches the design: `_plan` maximizes engaged cores first, then prefers fewer combines, then the widest rectangle (so a hidden line lies along a grid *row* — a column line is 2.91× slower when bandwidth-bound). Measured occupancy on this part: 56 cores for `(1,1,32,7168)`, 110 for `(1,1,64,12288)` and for the prefill shapes. Both dataflow halves batch (reader coalesces a whole block; writer drains a tile-row per `out_cb_depth` window), and the cross-core combine the design specified is genuinely built, not stubbed.
- **Blocking-model fidelity.** `block_rows`, `slice_hidden_tiles`, `num_hidden_slices`, `num_row_groups`, `out_cb_depth`, `rm_in/out_depth`, `hidden_tiles_per_core_floor`, `dm_chunk_tiles` are all host-side parameters defined once and pushed into the kernels as compile-time args; every CB page count, loop bound and grid dimension is computed *from* them. No CB capacity contains a whole-op dimension (`Wt`/`Rt` appear only as **page-stride** arithmetic in the accessors, never as a page count). After fix #1 there is no remaining duplicated block literal.
- **Documented design deviation is sound.** The implementation fuses the per-slice within-tile collapse into the root's combine (contributors ship the raw per-column partial tile) instead of the design's `cb_slice_stat` + `ReduceWithinTile::Skip`. I verified the stated cause upstream: `reduce_helpers_compute.inl:886` places the "Skip is AccumulateViaAdd-only" `static_assert` *after* the `if constexpr (AccumulateViaAdd) { … return; }` block, so it is instantiated for every algorithm and `Skip` cannot compile at all — while `reduce_helpers_compute.hpp:158-180` documents `Skip` as *the* idiom for exactly this cross-core combine. The deviation is arithmetically identical (sum-then-collapse == collapse-then-sum), ships the same NoC payload, and removes a compute phase from every non-root core. **Upstream bug worth filing against `kernel_lib`, not against this op.**
- **Cross-block gather safety.** With `num_blocks > 1` and `s > 1` a contributor could in principle overwrite the root's gather pages before the root consumed them. Traced the ordering: a contributor cannot produce block `b+1`'s partial until it has received block `b`'s multicast, which the root sends only after its `reduce` has unpacked all `s·B` gathered pages. No race.

---

## Prompt-rule check (`eval/prompts/rms_norm.txt`)

The prompt has no `## Rules` section; its MUST-level statements were checked directly and all hold: no host-side layout/shape workaround in the entry point (also enforced by the immutable acceptance test's monkeypatch guard), native non-tile-aligned handling in both layouts, a single exported `default_compute_kernel_config()` factory that `None` resolves through, `config=compute_kernel_config` passed through as-is, `ValueError` on rank < 1 / gamma-width mismatch, and `"none"` accepted for both gamma format axes.

**One advisory, deliberately not "fixed".** The prompt asks for `tag_gamma_dtype` / `tag_gamma_layout` in `INPUT_TAGGERS`. The op instead derives those axes inside `validate()` (and `axes.py` mirrors it). That is the **correct** wiring here and changing it would break the suite: `eval/feature_matrix.cartesian` removes any tagger key from the iterated finite axes ("the tagger wins — the axis isn't iterated"), and gamma's dtype/layout are *not* derivable from the input shape tuple, so declaring them as taggers would collapse both axes to a single value and silently delete ~2/3 of the gamma cartesian. Registry conformance is unaffected: every declared tagger key (`alignment`, `rank`) appears in `SUPPORTED`, and the runtime-captured cell matches the declared cell (`test_axes_consistency.json` is empty).

---

## Registry Conformance

- **Confirmed present and correctly wired** in `ttnn/ttnn/operations/rms_norm/rms_norm.py`: `INPUT_TAGGERS` (both taggers carry the `(inputs, axes)` signature), `SUPPORTED` (all nine axes, including every tagger key and every op-specific axis), `EXCLUSIONS` (empty, with a written reason — the one refusal the design names, `{float32, fp32_dest_acc_en=False}`, lies *outside* SUPPORTED today and so cannot be an EXCLUSIONS entry yet), and `validate()` (structural checks → SUPPORTED per-axis → EXCLUSIONS cell-level, raising `UnsupportedAxisValue` / `ExcludedCell` from `ttnn.operations._op_contract`). `validate(...)` is the entry point's first statement.
- **Confirmed the op file does NOT declare `INVALID`** — it is sourced from `eval/golden_tests/rms_norm/feature_spec.py`.
- **No auto-fixes to SUPPORTED were needed**: `xpass_drift = 0`, so there is no under-claim to promote.
- The op also exports a `PROPERTIES` dict (multi_core / bounded_cb / math_fidelity). Harmless and informative; left as is.

### INVALID audit (`eval/golden_tests/rms_norm/feature_spec.py`) — 2 findings, for the user

Well-formed and correct:

- `{bfloat8_b, ROW_MAJOR}` — the canonical activation entry, present. ✓
- `{gamma_dtype: bfloat8_b, gamma_layout: ROW_MAJOR}` — the same impossibility on the gamma tensor, single-tensor coupling. ✓
- The six no-gamma canonicalization cells couple `gamma_mode` to the `"none"` sentinel in **both** directions, leaving exactly one canonical `("none","none")` cell. ✓ (This is what makes the `no_gamma` cells land inside SUPPORTED rather than xpass-failing.)

Flagged (do **not** edit `feature_spec.py` from here — please fold in via `/golden-tests`):

1. **Cross-tensor coupling in the three "author-scoped" ROW_MAJOR + sharded + TILE-gamma entries** (`feature_spec.py:94-96`). These couple the **activation's** `layout` to the **gamma tensor's** `gamma_layout` — the canonical INVALID authoring mistake — and the file's own comment says they are "NOT structural impossibility" but parked "out of scope for now". By the registry model that is `EXCLUSIONS` (op-side, xfail, visible to the queue), not `INVALID` (harness skip, invisible). Recommendation: drop them from `INVALID`; if the op still refuses those corners after Refinement 2, they become one honest EXCLUSIONS entry there.
2. **Same category, `{bfloat8_b, w_non_aligned}` / `{bfloat8_b, h_non_aligned}`** (`feature_spec.py:100-101`). These *are* single-tensor (both axes describe the activation), so they pass the coupling rule, but they likewise encode "my kernel doesn't do this yet" rather than impossibility. They are consistent with Refinement 1's plan (bf8b lands tile-aligned first), so the practical impact is nil — but if bf8b ever grows non-aligned support these must be promoted out of INVALID or the cells stay invisible.

Neither finding affects today's numbers: with both sets skipped, `invalid_skipped = 33900` and the loud categories are all zero.

Two rows land in an `invalid_unexpected` bucket: `test_translated.py::test_rms_norm_sharded_uneven_multicore_logical_width[...-bfloat8_b]` ×2. Those are the *translated* suite, which routes through the runtime `SupportRefusal` hook (lenient xfail) instead of the harness's collection-time skip, so an INVALID cell reports `xfail` rather than `skipped`. Harness bookkeeping, not an op signal.

---

## L1 Ledger Audit

- **Ledger currency**: five discrepancies found and fixed (see Code Review #5). Post-fix, every declared CB has a row, every row corresponds to a live CB, and every capacity expression matches what the descriptor allocates.
- **Capacity vs live set, both directions**: `cb_input_tiles` (`B*S`), `cb_gamma_tiles` (`S`), `cb_sq_partials`/`cb_rms_recip`/`cb_rms_bcast` (`B`), `cb_gathered_partials` (`s*B`), `cb_scaler`/`cb_w_mask`/`cb_thread_sync` (1) all have capacity == live set. The three that exceed it — `cb_output_tiles` (`2S` vs `S`), `cb_rm_stage_in`, `cb_rm_stage_out` — exceed it by exactly one **stated depth knob**, in each case across a *different-processor* producer/consumer pair (compute↔writer, reader↔compute), which is the legal case for a depth. No CB's live set spans a block axis whose capacity fails to scale with it: `cb_input_tiles` spans both block axes and scales as `B*S`; `cb_gathered_partials` spans the contributor axis and scales as `s*B`; everything else streams over the axes it does not span, with a fixed window.
- **Disjoint lifetime without justification**: none. Every row carries a concrete reason, and the two that could naively be merged are the interesting ones — `cb_sq_partials` vs `cb_rms_recip` cannot alias because `reduce` asserts `input_dfb != output_dfb` (and that assert is compiled out in production builds, so aliasing would corrupt rather than halt), and `cb_rms_bcast` vs `cb_rms_recip` cannot alias because one is compute→reader and the other reader→compute (plus `mcast_pipe`'s loopback mode *requires* `src_l1 != dst_l1`).
- **Block-size defaults**: both held. Interleaved spreads the split's work units across the grid (occupancy is the first term of `_plan`'s score) and then takes the coarsest block that fits (`block_rows = core_row_tiles`, decremented only by the fit predicate) — so the common case is `num_blocks_this_core == 1`, one init set and one fill/drain per core. The only departure from "coarsest" is L1-forced, and the inventory was minimized *first*: three block-sized buffers (`cb_masked_x`, `cb_squared`, `cb_normalized`) plus `cb_output_tiles` on the ROW_MAJOR path were removed by in-place transform / DEST folding before any budget solve.
- **Per-core footprint** (from the ledger, in tiles):
  `B*S` + `[S if gamma]` + `B` + `[s*B + B if s>1]` + `B` + `3` + (`(rm_in_depth+rm_out_depth)*S` for ROW_MAJOR, else `out_cb_depth*S`).
  `B*S` dominates and is the term the fit predicate solves against; the depth terms scale with the hidden extent only; `s*B` scales with row × contributor.
- **Data-movement budget**: present, consistent with the implemented split, and it names its cheapest-traffic alternative with a delta — input 1× (residency across all three normalize phases is what buys it), output 1×, gamma `g×` (reuse-shared along the row axis), plus the combine's `Rt·s·T` NoC payload. The `g = 1` corner *is* the cheapest DRAM split and *is* what the decode regime picks; in the prefill regime the implemented `g = 8` costs +0.44 MB DRAM to save 28 MB of NoC, which is stated with numbers rather than settled by occupancy. The two unimplemented cheaper-traffic options (GammaBroadcast, reduce-scatter) are lamped with their deltas and the measured reason they are unmeasured.
- **Budget honesty (noted, not fixed)**: `l1_working_budget` is `1 MB − 96 KB = 928 KB`, because `device.l1_size_per_core()` is not bound to Python on this build. The part actually reports 1.46 MB unreserved (`ttnn.get_max_worker_l1_unreserved_size()`), but that query is keyed off `KERNEL_CONFIG` and overshoots by the kernel-config ringbuffer, so raising the budget needs its own reserve and a measured re-run. It is **conservative in the safe direction** (every fit decision assumes less L1 than exists) and it costs only block coarseness — folded into Refinement 4 as a named lever, per the "ledger findings never get a refinement of their own" rule.

---

## Precision Baseline

`tests/ttnn/unit_tests/operations/rms_norm/test_rms_norm_precision_baseline.py` — 40 cells
(5 shapes × {bf16, fp32} × {TILE, ROW_MAJOR} × {gamma, no_gamma}), **all passing**.
Representative rows (TILE layout, gamma on):

| Shape | dtype | PCC | Max Abs Err | Mean Abs Err | Relative RMS Err | got/true median (p5 … p95) |
|-------|-------|-----|-------------|--------------|------------------|-----------------------------|
| (32, 64) | bf16 | 0.9999973 | 1.81e-02 | 1.11e-03 | 2.32e-03 | 0.99992 (0.9961 … 1.0036) |
| (2, 64, 256) | bf16 | 0.9999976 | 3.70e-02 | 1.22e-03 | 2.36e-03 | 0.99982 (0.9960 … 1.0037) |
| (1, 1, 512, 1024) | bf16 | 0.9999975 | 5.06e-02 | 1.22e-03 | 2.35e-03 | 0.99984 (0.9960 … 1.0037) |
| (1, 1, 32, 8192) †| bf16 | 0.9999965 | 4.77e-02 | 1.23e-03 | 2.35e-03 | 0.99984 (0.9960 … 1.0037) |
| (47, 100) ‡| bf16 | 0.9999972 | 2.41e-02 | 1.34e-03 | 2.32e-03 | 0.99975 (0.9959 … 1.0038) |
| (32, 64) | fp32 | 0.9999998 | 1.01e-02 | 8.29e-04 | 1.49e-03 | 0.99861 (0.9973 … 0.9997) |
| (2, 64, 256) | fp32 | 0.9999995 | 1.68e-02 | 8.92e-04 | 1.56e-03 | 0.99863 (0.9973 … 0.9998) |
| (1, 1, 512, 1024) | fp32 | 0.9999990 | 2.26e-02 | 8.98e-04 | 1.58e-03 | 0.99864 (0.9973 … 0.9997) |
| (1, 1, 32, 8192) †| fp32 | 1.0000000 | 2.55e-02 | 8.89e-04 | 1.55e-03 | 0.99866 (0.9974 … 0.9997) |
| (47, 100) ‡| fp32 | 0.9999997 | 1.72e-02 | 9.77e-04 | 1.61e-03 | 0.99863 (0.9973 … 0.9997) |

† forces the cross-core combine regime (`num_hidden_slices > 1`). ‡ non-tile-aligned in both H and W.
ROW_MAJOR mirrors TILE for bf16 exactly and is ~20 % worse for fp32 (extra tilize/untilize passes):
rel-RMS 1.83e-03 … 1.99e-03, ratio median ≈ 0.9983.

**Assessment.** PCC ≥ 0.999996 everywhere, three orders of magnitude inside the golden gates, and error
does **not** grow with W (2.32e-03 at W=64 vs 2.35e-03 at W=8192) — the fp32 stat CBs and the fp32 DEST
accumulation are doing their job across a 128× range of reduce length.

The one thing worth naming is the **scale check**, since this op's most dangerous failure mode (folding
tile padding into the RMS denominator) is a near-uniform scale error that PCC is blind to. The got/true
ratio is a *narrow* distribution — bf16 median 0.9998 with a ±0.4 % spread, fp32 median 0.9986 with a
±0.13 % spread — so there is a small systematic magnitude deficit, but it is:

- **width-independent** (identical at W=100 with 28 padding columns and at W=8192 with none — a padded-`W`
  denominator would show `sqrt(128/100) − 1 = 13 %` on the first and 0 % on the second), and
- **alignment-independent** (the non-aligned rows sit on the same median as the aligned ones).

That rules out the mask/scaler class of bug and identifies the residual as ordinary datapath rounding
(round-toward-zero into the FPU source registers on the fp32 path, which is why fp32's bias is one-sided
and bf16's — packed with round-to-nearest — is centred). Not a precision refinement: no failing cell, and
no lever short of moving the final multiply off the FPU.

**Recommended tolerances**: bf16 `PCC ≥ 0.995`, rel-RMS ≤ 0.04; fp32 `PCC ≥ 0.999`, rel-RMS ≤ 0.02
(i.e. keep the golden suite's existing gates — the measured margin is ~1000×). For a scale-regression
guard, assert `|median(got/true) − 1| < 0.02`, which the baseline test now does.

---

## Verifier CLI Summary

`eval/eval_test_runner.sh eval/golden_tests/rms_norm/ /tmp/rms_verify2` →
`python3 -m eval.verify_supported /tmp/rms_verify2 ttnn.operations.rms_norm` (run **after** the code fixes;
identical to the pre-fix run):

- supported_pass: **737**
- xfail_expected: **6174**
- invalid_skipped: **33900**
- supported_fail: **0** ✓
- xpass_drift: **0** ✓
- xfail_wrong_mode: **0** ✓
- supported_marked_xfail / supported_skipped / xfail_other / infeasible_skipped: 0
- invalid_unexpected: 2 (translated-suite bookkeeping, explained above) · no_axes_found: 15 (`test_regression.py`, all passing, not registry-driven)
- Golden run: `PASSED=752 FAILED=0 ERRORS=0 SKIPPED=33900 HANGS=0 TOTAL=40828`

Acceptance suite: `tests/ttnn/unit_tests/operations/rms_norm/` — **142 passed** before the fixes and
**142 + 40 (new precision baseline) passed** after.

### Where the 6174 xfails come from (the refinement queue's source)

| Missing axis value | xfail cells implicated | Covered by |
|---|---|---|
| `fp32_dest_acc_en = False` | 3609 | Refinement 1 |
| `dtype = bfloat8_b` | 962 | Refinement 1 |
| `gamma_dtype = bfloat8_b` | 860 | Refinement 1 |
| `memory_layout = WIDTH_SHARDED` | 1624 | Refinement 2 |
| `memory_layout = BLOCK_SHARDED` | 1619 | Refinement 2 |
| `memory_layout = HEIGHT_SHARDED` | 1593 | Refinement 2 |

(Cells commonly miss on two axes at once, hence the overlap.) `TARGET − SUPPORTED` is exactly these six
values, and Refinements 1–2 cover all six — no documented omissions, no queue gap.

---

## Recommendations

1. **Refinement 1 is the gate for everything measured.** Every hand-authored loose case in
   `feature_spec.py` — the whole `perf`, `resilience` and `pad_poison` corpus — pins
   `fp32_dest_acc_en=False`, so today none of them executes. That includes the `pad_poison` group, which
   is the *only* corpus specifically built to catch a padded-denominator bug at a magnitude PCC can see
   (W=40 ⇒ 26.5 % error). The interleaved cartesian's small-W non-aligned shapes (W=17/47/50) plus the
   precision baseline's width-independence check cover that risk today, but the dedicated guard only
   arms after Refinement 1.
2. **Do not re-spend the perf phases on the two knobs already swept.** `HIDDEN_TILES_PER_CORE_FLOOR ∈
   {2,4,8,16}` and the row-group rectangle search (11 vs 56 cores on the decode shape) both came back flat
   within 1–2 %, recorded in `test_rms_norm_perf.py`'s docstring. The binding costs on this op are the
   ~3.5 µs fixed dispatch + boot floor and the short per-core DRAM transfer, not per-core work.
3. **The prefill regime is where the measured gap is largest** (~184 GB/s achieved vs a reference implying
   ~347 GB/s, i.e. ≈1.9×). That is Refinement 4, and it is also the only regime where the block-size ×
   buffer-depth co-tune has real room, because `core_row_tiles` is large there.
4. **L1 budget is 928 KB against a 1.46 MB part** and cannot be queried honestly from Python today
   (`device.l1_size_per_core()` unbound; `get_max_worker_l1_unreserved_size()` overshoots by the
   kernel-config ringbuffer). Raising it is the cheapest structural lever on block coarseness — folded
   into Refinement 4 rather than filed separately.
5. **Two upstream `kernel_lib` bugs are worth filing** (already captured in the implementer's friction
   log, independently confirmed here): (a) `ReduceWithinTile::Skip` is unreachable through
   `compute_kernel_lib::reduce()` because its guarding `static_assert` sits outside the discarded branch,
   despite being documented as *the* cross-core-combine idiom; (b) a `PackTile` with
   `(ReservePolicy::None, PushPolicy::None)` and `TileOffset::Unset` silently drops its output when a
   previous chain packed to the same CB. Both cost this op a workaround; neither is this op's to fix.
6. **`feature_spec.py`'s two author-scoped INVALID groups should be revisited** (see the INVALID audit).
   The cross-tensor-coupled ROW_MAJOR × sharded × TILE-gamma trio in particular belongs in `EXCLUSIONS`
   after Refinement 2, not in `INVALID`.
