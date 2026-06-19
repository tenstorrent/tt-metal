# Verification Report: rms_norm

Phase 0 verification of the registry-model `rms_norm` op (RMS normalization over
the last dim, performance design with two regimes: A = row-parallel resident,
B = wide-W cross-core all-gather of partial Σx²).

---

## Headline finding (blocker)

**Regime B (the cross-core wide-W path) is numerically broken.** Regime A is
correct; Regime B produces output too large by exactly `sqrt(2·num_chunks)` —
the gathered/combined Σx² underflows by a clean factor of **1/(2·num_chunks)**,
where `num_chunks = ceil(Wt_s / reduce_block)` is the PASS-1 chunk count. This
makes every wide / few-row golden cell (21 of them) fail. It is filed as
**Refinement 1 (blocker)** in `op_requirements.md`. See "Verifier CLI Summary"
and "Regime B defect" below. All other Phase-0 cells (row-parallel Regime A) are
correct.

---

## Code Review

Fixes applied in this pass:

1. **`__init__.py` did not re-export the registry symbols.** The golden harness
   imports `INPUT_TAGGERS`, `SUPPORTED`, `EXCLUSIONS` from
   `ttnn.operations.rms_norm`; only `rms_norm` was exported, so the entire
   golden suite failed at *collection* (ImportError). **Fixed** — now exports
   `rms_norm`, `validate`, `INPUT_TAGGERS`, `SUPPORTED`, `EXCLUSIONS`.

2. **`INPUT_TAGGERS` did not match the feature_spec contract.**
   - `tag_alignment` returned a 2-value split (`tile_aligned` /
     `non_tile_aligned`); `non_tile_aligned` is not even in `TARGET["alignment"]`.
     **Fixed** to the documented 3-value split (`tile_aligned` /
     `w_non_aligned` / `h_non_aligned`) so W-mask vs H-mask refinement signal
     routes correctly.
   - `tag_rank` was missing entirely, so `rank` was iterated as a finite axis
     (3× redundant cases per shape, all running the same actual rank).
     **Added** `tag_rank(inputs, axes) -> int(len(inputs[0]))`.

3. **SUPPORTED was missing axes the harness records/gates → silent drift.**
   `gamma_dtype`, `gamma_layout`, `gamma_mode`, `fp32_dest_acc_en`, and `rank`
   were absent from `SUPPORTED`, so the harness treated *all* their values as
   supported. fp32 / bf8b / ROW_MAJOR gamma cells would have **run and failed**
   (the kernel reads gamma with the input's bf16 tile format) → dishonest
   `supported_fail`. **Fixed** — `SUPPORTED` now declares every gated axis;
   unsupported gamma/precision/layout values now xfail correctly.

4. **`validate()` ignored gamma and `compute_kernel_config`.** It took only
   `input_tensor`, so it could not gate the gamma axes or `fp32_dest_acc_en`.
   **Fixed** — `validate(input_tensor, gamma, compute_kernel_config)`; the entry
   point now forwards both. The axes reconstruction mirrors
   `helpers.classify_call` exactly (incl. the no_gamma canonical pinning of
   gamma_dtype=float32 / gamma_layout=TILE).

5. **Prompt MUST validations were missing.** The op prompt requires Python-side
   `ValueError`/`RuntimeError` for rank < 2 and for a gamma whose last dim does
   not match the input's. **Added** both as `ValueError` guards at the top of
   `validate()` (these are input errors, *not* support refusals — they must not
   be `NotImplementedError`).

Deferred (documented, not fixed — architectural / risk):

- **`cb_normalized` and `cb_gamma` are sized to `Wt` (full row), not the
  constant `REDUCE_BLOCK` the design specifies.** Pass-2 currently does a
  full-row Col multiply into `cb_normalized` then a full-row Row multiply into
  `cb_output`, so `cb_normalized` must hold `Wt` tiles. This makes the per-core
  L1 footprint scale with `Wt` on the gamma path, so `RESIDENT_BUDGET_TILES`
  (560) understates real pressure — a single-core Regime A on a wide gamma row
  (e.g. W=8192, Wt=256) would allocate input(256)+gamma(256)+normalized(256) ≈
  1.5 MB and OOM. This is *why the verifier did not "fix" Regime B by rerouting
  wide rows to Regime A* — it would trade a correctness bug for an OOM. The
  streaming fix (per-`REDUCE_BLOCK` fused Col→Row multiply, eliminating the
  `cb_normalized` round-trip per the design's sanctioned optimization) is a
  pass-2 rewrite; noted here, not in the refinement queue (it changes no
  SUPPORTED axis).

- **Writer barriers per tile** (`noc_async_write_barrier` inside the per-tile
  drain loop). Correct but slow; a batched write + single barrier would be
  faster. Pure perf, no failing cell — advisory only.

---

## Registry Conformance

- **Confirmed present and wired:** `INPUT_TAGGERS` (taggers now take
  `(inputs, axes)`), `SUPPORTED`, `EXCLUSIONS`, `validate()`. The public entry
  point calls `validate()` as its first line.
- **Confirmed the op file does NOT declare `INVALID`** — it is sourced from
  `feature_spec.py` (registry model). Good.
- **EXCLUSIONS:** one entry — `{gamma_mode: gamma, gamma_dtype: float32}`.
  `float32` is listed in `SUPPORTED["gamma_dtype"]` *only* so the no_gamma
  canonical cell (gamma_dtype=float32, gamma_layout=TILE) is supported; a real
  fp32 gamma is refused here (single-tensor coupling — both axes describe gamma).
- **No drift after fixes:** `xpass_drift = 0`, `xfail_wrong_mode = 0`,
  `supported_marked_xfail = 0`.

### INVALID audit (`eval/golden_tests/rms_norm/feature_spec.py`)

All five INVALID entries are well-formed against the three sanity rules:

| Entry | Verdict |
|---|---|
| `{dtype: bf8b, layout: ROW_MAJOR}` | ✓ canonical bf8b+ROW_MAJOR for the activation (block format has no RM); single-tensor. |
| `{gamma_dtype: bf8b, gamma_layout: ROW_MAJOR}` | ✓ same impossibility on the gamma tensor; single-tensor. |
| `{gamma_mode: no_gamma, gamma_dtype: bf16}` | ✓ no_gamma canonicalization. |
| `{gamma_mode: no_gamma, gamma_dtype: bf8b}` | ✓ no_gamma canonicalization. |
| `{gamma_mode: no_gamma, gamma_layout: ROW_MAJOR}` | ✓ no_gamma canonicalization (keeps gamma_dtype=float32, gamma_layout=TILE as the single canonical no_gamma cell). |

No cross-tensor-axis couplings; no "kernel doesn't do it yet" entries
masquerading as INVALID; canonical bf8b+ROW_MAJOR present for both tensors;
no-weight canonicalization present. **No changes recommended.**

---

## Precision Baseline

Measured on the correct Regime A path (bf16 / TILE / tile-aligned,
HiFi4 / fp32_dest_acc_en=True), `tests/.../test_rms_norm_precision_baseline.py`,
seed 0, standard-normal input:

| Shape | gamma | Max Abs Err | Mean Abs Err | Relative RMS Err |
|-------|-------|-------------|--------------|------------------|
| (1,1,32,32)    | no  | 0.01778 | 0.00204 | 0.00319 |
| (1,1,64,128)   | no  | 0.02523 | 0.00203 | 0.00315 |
| (2,4,128,512)  | no  | 0.03136 | 0.00203 | 0.00316 |
| (1,1,2048,256) | no  | 0.03235 | 0.00195 | 0.00303 |
| (1,1,32,32)    | yes | 0.02550 | 0.00216 | 0.00373 |
| (1,1,64,128)   | yes | 0.07095 | 0.00198 | 0.00395 |
| (2,4,128,512)  | yes | 0.07790 | 0.00197 | 0.00391 |
| (1,1,2048,256) | yes | 0.06393 | 0.00183 | 0.00377 |

PCC ≥ 0.999 on every case (the golden run reports pcc ≈ 0.99992 even on the
*broken* Regime B cells — PCC is insensitive to the uniform per-row scale error,
which is why the relative-RMS check is the load-bearing signal here).

**Assessment:** Regime A precision is excellent and well inside the bf16 band.
Gamma roughly doubles max-abs (extra bf16 multiply) but mean/RMS stay flat.

**Recommended tolerances (Regime A, bf16):** PCC ≥ 0.995, relative RMS ≤ 0.04
(matches `helpers.TOLERANCES[bfloat16]`). rtol ≈ 0.02, atol ≈ 0.10 for
`comp_allclose`-style checks.

---

## Regime B defect (detail for Refinement 1)

Diagnosed with all-ones input (expected output ≡ 1.0; deviation reveals the
summed fraction of Σx² exactly):

| Shape | Wt | Ht_total | K | Wt_s | num_chunks | summed fraction |
|-------|----|----|---|------|------------|-----------------|
| (1,1,32,4096)  | 128 | 1 | 64 | 2  | 1 | 0.5001 |
| (1,1,32,1280)  | 40  | 1 | 40 | 1  | 1 | 0.5001 |
| (1,1,64,8192)  | 256 | 2 | 32 | 8  | 2 | 0.2500 |
| (1,1,128,4096) | 128 | 4 | 16 | 8  | 2 | 0.2500 |
| (1,1,256,2048) | 64  | 8 | 8  | 8  | 2 | 0.2500 |
| (1,1,64,12288) | 384 | 2 | 32 | 12 | 3 | 0.1662 |

Fraction = **1/(2·num_chunks)** exactly. Regime A with the *same* PASS-1
parameters (e.g. Wt=8, reduce_block=4, num_chunks=2) returns 1.0000 — so PASS-1
reduce-accumulate is correct; the defect is isolated to the Regime-B-only path
(the mcast all-gather in `rms_norm_reader_mcast.cpp` + the K-partial combine in
`rms_norm_compute.cpp`, plus the fragile *double* producer/consumer handshake on
the 2-page `cb_partial_sumsq`: compute produces it in PASS-1, the reader consumes
it as the mcast source, then compute re-produces it in the combine and consumes
it in finalize). The clean `1/(2·num_chunks)` dependence on a compute-side chunk
count that neither the gather nor the combine can see points squarely at that
cross-thread CB handshake / staleness. Full repro is in `op_requirements.md`
Refinement 1.

---

## Verifier CLI Summary (`verifier_report.json`)

- total: 5142
- supported_pass: **22**
- xfail_expected: **2144**
- invalid_skipped: **2940**
- supported_fail: **21**  ← all Regime B (wide / few-row); single root cause → Refinement 1
- xpass_drift: **0** ✓
- xfail_wrong_mode: **0** ✓
- supported_marked_xfail: **0** ✓
- no_axes_found: 15 — `test_regression.py` float32 numerics tests. These dispatch
  the op directly (not via the observe wrapper) with **float32** input, which
  Phase 0 refuses (float32 ∉ SUPPORTED["dtype"]); they will pass once Refinement
  2 lands fp32. Not registry cells; not a bug.

`supported_fail = 21` is the only non-clean loud category and is entirely the
Regime B defect. Per the agent doc this is a real kernel bug beyond an in-place
verifier patch → filed as **Refinement 1 (blocker)**; the queue must not advance
to Refinements 2/3 until it is clean.

---

## Recommendations

1. **Refinement 1 (Regime B correctness) is a hard blocker** — it breaks the
   op's headline feature. Do it first; the 21 failing cells and the 3 LOOSE
   cross-core cases all clear with it. There is no SUPPORTED axis to gate Regime
   B by (it is shape/L1-fit-selected), so gating-out is not an option and would
   be dishonest — it must be fixed.
2. **`cb_normalized`/`cb_gamma` = Wt sizing** (deferred above) should be folded
   into the Regime B fix or done alongside, since the resident-budget math that
   selects A vs B depends on it. If pass-2 is rewritten to the streaming fused
   Col→Row multiply, the L1 budget tightens and the A/B heuristic can be retuned.
3. **Numerical levers for later:** the bf16 Σx² accumulation goes through the
   `cb_partial_sumsq` bf16 CB; for very wide W an fp32 intermediate would tighten
   RMS. No `numerical-precision` Regime A cell currently fails, so this is a note,
   not a refinement — revisit if wide-W RMS drifts after Refinement 1.
