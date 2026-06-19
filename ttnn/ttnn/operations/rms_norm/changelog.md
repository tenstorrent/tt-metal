# Changelog: rms_norm

## Phase 0 — Core Implementation
- **Date**: 2026-06-19
- **What was done**: Initial implementation via the incremental pipeline
  (planner → implementer → verifier). Two-regime performance design:
  Regime A (row-parallel, full row resident, embarrassingly-parallel multi-core)
  and Regime B (wide-W cross-core W-split with an mcast all-gather of partial
  Σx²). Verifier pass: registry conformance hardening, golden run, precision
  baseline, refinement queue.
- **SUPPORTED at Phase 0**:
  - dtype = [bfloat16]
  - layout = [TILE_LAYOUT]
  - alignment = [tile_aligned]
  - rank = [2, 3, 4]
  - fp32_dest_acc_en = [True]
  - gamma_mode = [gamma, no_gamma]; gamma_dtype = [bfloat16, float32(no_gamma canonical only)]; gamma_layout = [TILE_LAYOUT]
  - EXCLUSIONS = [{gamma_mode: gamma, gamma_dtype: float32}]
- **Accuracy achieved (Regime A, bf16, measured on 8 cases via
  test_rms_norm_precision_baseline.py)**:
  PCC ≥ 0.999; max_abs_err ≤ 0.078 (gamma) / ≤ 0.032 (no gamma);
  mean_abs_err ≈ 0.002; relative RMS ≈ 0.003–0.004.
- **Golden suite at Phase 0** (per `verifier_report.json`):
  total 5142 — supported_pass 22, xfail_expected 2144, invalid_skipped 2940,
  **supported_fail 21** (all Regime B), xpass_drift 0, xfail_wrong_mode 0,
  supported_marked_xfail 0, no_axes_found 15 (float32 test_regression.py).
- **Issues encountered / fixed this pass**:
  - `__init__.py` did not re-export `INPUT_TAGGERS`/`SUPPORTED`/`EXCLUSIONS` →
    whole golden suite failed at collection. Fixed (now re-exported).
  - `tag_alignment` was a 2-value split returning an out-of-universe value;
    replaced with the feature_spec-mandated 3-value split. Added missing
    `tag_rank`. Both taggers now take `(inputs, axes)`.
  - `SUPPORTED` was missing `rank`, `fp32_dest_acc_en`, `gamma_mode`,
    `gamma_dtype`, `gamma_layout` → fp32/bf8b/ROW_MAJOR gamma cells would have
    run-and-failed (silent over-claim). Added all; gating now honest.
  - `validate()` now takes `gamma` + `compute_kernel_config` and mirrors
    `helpers.classify_call`; added prompt-required `ValueError` guards (rank < 2,
    gamma last-dim mismatch). Entry point forwards both args.
  - **Known blocker (NOT fixed — filed as Refinement 1):** Regime B
    (cross-core all-gather) is numerically broken — output too large by
    `sqrt(2·num_chunks)`; gathered Σx² underflows by exactly 1/(2·num_chunks).
    Regime A is correct. This is the op's headline feature and gates the queue.
  - Deferred (noted in verification_report.md, to fold into Refinement 1):
    `cb_normalized`/`cb_gamma` sized to `Wt` (not constant `REDUCE_BLOCK`), which
    understates the resident-L1 budget; writer barrier-per-tile (perf only).
- **Tests added**:
  - `tests/ttnn/unit_tests/operations/rms_norm/test_rms_norm_precision_baseline.py`
    (PCC + max/mean abs + relative RMS over 4 shapes × gamma/no-gamma, Regime A).
  - (`test_rms_norm.py` acceptance suite already present — 20/20 passing.)

## Refinement 1 — Fix Regime B cross-core all-gather correctness (BLOCKER)
- **Date**: 2026-06-19
- **What was done**:
  - **Root-caused two compounding bugs** at the `cb_partial_sumsq` cross-thread
    handshake that made Regime B output too large by `sqrt(2·num_chunks)`:
    1. *(factor 1/num_chunks)* The mcast reader's `cb_wait_front(cb_partial_sumsq, 1)`
       was satisfied after PASS-1's **first** W-chunk (the reduce-accumulate pushes
       once per chunk), so it grabbed an early front L1 address holding only chunk-0's
       partial. The accumulator pops/repushes across chunks, landing the final sum in a
       **different physical page**, so the reader mcast only chunk-0's contribution.
    2. *(factor 1/2)* The K-partial combine read the **stale, un-popped PASS-1 local
       sum** as its in-place accumulator front (the reader pops `cb_partial_sumsq` only
       after pushing the gathered CB, racing compute's reuse).
  - **Fix**: added a dedicated single-push CB `cb_local_sumsq` (CB 28, Regime B only).
    Compute copies the fully-accumulated local Σx² into it exactly once after PASS-1
    completes (`copy<>` also pops `cb_partial_sumsq`, emptying it). The reader now waits
    on `cb_local_sumsq` (observes only the final value); the combine writes into the
    now-empty `cb_partial_sumsq` with no stale-front aliasing. Regime A (num_partials==1)
    skips this path entirely and is unchanged.
  - **Folded in the deferred `cb_normalized = Wt` sizing fix**: the gamma pass-2 now
    streams the Col→Row multiply per `REDUCE_BLOCK` with `cb_normalized` sized to one
    block (was `Wt`/`Wt_s`), so per-core L1 no longer scales with row/shard width and the
    A/B resident-budget heuristic is sound. (The design's single-`eltwise_chain` fusion
    was not used — the helper lib has no broadcast `DestReuseBinary` and `BinaryFpu`
    cannot take DST as an operand; the streaming two-helper chunked form is the bounded
    equivalent.)
- **Accuracy achieved (Regime B, bf16, measured)**: all-ones → exactly 1.0 across
  num_chunks ∈ {1,2,3} and LOOSE W ∈ {16384, 32768}; random standard-normal vs torch
  PCC ≥ 0.99999, relative RMS 0.0035–0.0091 (±gamma) — well inside the bf16 band
  (relRMS ≤ 0.04). Regime A precision baseline unchanged (8/8).
- **Golden test progress**: 43/43 supported cells passing (was 22/43 — the +21 were all
  Regime B); 0 failed, 0 xpass-drift. (The 15 `test_regression.py` failures are float32,
  out of scope until Refinement 2.)
- **Issues encountered**: None blocking. The clean single-chain pass-2 fusion the design
  references is not expressible with the current helper library (noted above); the
  bounded chunked-streaming form was used instead with identical accuracy.
- **Tests added**:
  - `tests/ttnn/unit_tests/operations/rms_norm/test_rms_norm_regime_b.py` — 39 cases:
    all-ones exact (==1.0), per-shard-distinguishable ramp (verifies the all-gather
    delivers all K distinct partials), random vs torch (±gamma), and the LOOSE wide-W
    cases. All pass in production timing.

## Refinement 2 — Numerical configurability expansion
- **Date**: 2026-06-19
- **What was done**:
  - **SUPPORTED grown**: `dtype += [float32, bfloat8_b]`; `fp32_dest_acc_en += [False]`;
    `gamma_dtype += [bfloat8_b]`. EXCLUSIONS: removed the old gamma-float32 entry
    (gamma-present float32 now works) and added `{dtype: float32, fp32_dest_acc_en: False}`
    (fp32 input mandates fp32 accumulation — the prompt's documented EXCLUSION).
  - **Per-CB format derivation in the program descriptor** (both regimes): input/output
    CBs follow the tensor dtype; the gamma CB follows `gamma.dtype` (this is what unblocks
    mixed-precision gamma — bf16 activations + fp32/bf8b gamma); accumulator intermediates
    (Σx², scaler, recip-rms, normalized block, Regime-B gathered partials) follow
    `_intermediate_dtype`: promoted to `Float32` when `fp32_dest_acc_en`, else bf16 — and
    **never bf8b** (a block-float accumulator is wrong). bf16 input keeps bf16 intermediates,
    byte-identical to Phase 0 / Refinement 1 (no regression). Per-CB tile bytes via
    `ttnn.tile_size(format)` (was a single shared `buffer_page_size`).
  - **No compute-kernel changes** — the eltwise/reduce helpers reconfig unpack (BinaryFpu
    `Input`, CopyTile `Input`) and pack (PackTile `Output`) data formats automatically, so
    mixed input/intermediate/gamma/output formats just work (numeric-formats skill pass
    condition held).
  - **Regime-B mcast reader fix (the one real bug uncovered)**: the all-gather strided its
    slots and sized the cross-core transfer with `get_tile_size(cb_input_resident)`, a latent
    assumption that input format == partials format. True for bf16 (both 2048 B), false for
    bf8b input (1088 B) with fp32 partials (4096 B) → the gather copied/strode the wrong byte
    count → `Inf` in a subset of outputs (72 golden cells). Now uses
    `get_tile_size(cb_partials_gathered)` for the gather slot stride / local copy / `sender.send`
    size, and the input tile bytes only for the input read.
- **Accuracy achieved** (precision matrix, fp32_dest_acc_en=True, HiFi4, 128x512):
  float32 PCC ≥ 0.99999, relRMS ≤ 0.0015; bfloat16 PCC ≥ 0.99999, relRMS ≤ 0.004;
  bfloat8_b PCC ≥ 0.9999, relRMS ≤ 0.015. fp32_dest_acc_en=False (bf16/bf8b) and the full
  LoFi→HiFi4 sweep all stay above the asserted floors (bf16/fp32 ≥ 0.99, bf8b ≥ 0.98);
  no Inf/NaN in any cell.
- **Golden test progress**: 418/418 supported passing (was 346/346 before this refinement;
  the +72 are the bf8b Regime-B cells the mcast fix unblocked), 2940 skipped, 1784 xfailed,
  0 failed, 0 xpass-drift. The 15 float32 `test_regression.py` `no_axes_found` cases are
  cleared (float32 is now in SUPPORTED).
- **Issues encountered**: the bf8b Regime-B `Inf` bug (root-caused + fixed, above). No other
  blockers — fp32, bf8b, the False precision corner, and mixed-precision gamma all landed.
- **Tests added**:
  - `tests/ttnn/unit_tests/operations/rms_norm/test_rms_norm_precision_matrix.py` — the
    authoritative precision matrix: 8 shapes (Regime A + B) × {bf16, fp32, bf8b} ×
    {HiFi4..LoFi} × {fp32_acc, bf16_acc} × {uniform, normal} × {gamma, no_gamma} = 641 cases
    (+128 skipped EXCLUSION cells), plus `test_rms_norm_precision_matrix_fp32_no_acc_refused`
    asserting the EXCLUSION raises a support refusal.
  - `tests/ttnn/unit_tests/operations/rms_norm/precision_matrix_results.md` — results table.
