# Verification Report: gated_delta_net

Golden results dir: `/tmp/gdn_golden3` (CLI output copied to `verifier_report.json` in this directory).

## Code Review

Fixed in place:

- **DRY: CB indices and spec constants were restated in all three kernels.** The 30 CB indices
  (`CB_*`), `HISTORY`, `CONV_TAPS` and `TOK_COEF_PAGES` were literals in the descriptor and again
  in the reader, writer and compute kernels. They are now defined once in
  `gated_delta_net_program_descriptor.py` and passed to every kernel as `GDN_*` defines
  (`_kernel_defines()`); `CONV_TAPS` derives from `HISTORY`. `gated_delta_net.py` imports `TILE` and
  `HISTORY` from the descriptor instead of redeclaring them.
- **cb_conv_acc capacity 2 vs live set 1.** It is compute-to-compute; with one compute thread a
  second page buys nothing. `CONV_ACC_DEPTH = 1` (4 KiB per core saved).
- **Unused reader runtime arg** (`batch`) removed.
- **SUPPORTED drift: `batched_prefill`.** The kernels already handle a sequence boundary inside a
  tile row (pair `[lo, hi)` ranges, row-segment I/O, zeroed window rows). Probed on 4 shapes
  (B=2 T=5, B=3 T=32, B=2 T=33, chip B=2 T=32), clean and inf-poisoned padding: all pass. Added to
  `SUPPORTED["sequence_mode"]`; the 3 golden float32 `batched_prefill` cells now pass. Extended test:
  `test_gated_delta_net_batched_prefill.py` (6 cases).
- **L1 ledger currency** (see below).

Deferred (needs kernel rework, filed or recorded):

- **Collapsed knob `TOKEN_TILE_ROWS_PER_BLOCK`.** The host constant exists and CB sizes scale with
  it, but the kernels hard-code one tile row per prep block and the descriptor asserts `== 1`. Only
  prefill would benefit (decode is one tile row), and the only flagged perf target is decode, so it is
  not queued; recorded under Recommendations.
- **Expression: reader window hand-off per column tile.** The design's conv block is one channel
  group (Kt or Vt columns); the reader acquires, zero-fills (barrier), reads (barrier), shifts rows
  (barrier) and pushes one column tile at a time, and compute waits per column. Planned extent > 1,
  each unit synchronized and retired separately: an expression failure on the decode critical path.
  Folded into Refinement 2 (perf).
- `V_BLOCK_TILES` asserts `== Vt`: this is R-VSPLIT, a scheme-change; Refinement 3.

Advisories (prompt `## Rules`): all MUST rules hold for the current SUPPORTED scope: one
`generic_op` dispatch per call; state only via U2D + SFPU decay, increment accumulated onto fp32 DEST;
state crosses DRAM once each way; steps only for `lo <= row < hi`; inf padding passes
(poison_inf_* loose cases); conv_state_out bit-exact for T < 3; no ttnn op on inputs/outputs;
`fp32_dest_acc_en` read from the caller config and `False` refused. Note: the descriptor writes
`unpack_to_dest_mode` onto the caller's `compute_kernel_config` object (design-sanctioned, documented
in op_design.md Parameters); every other field is left as passed.

Known environment issue: the implementer recorded that 4 acceptance cases (K=V=128 and K>V)
overflow the kernel-config buffer under `--dev` (watcher + LLK asserts inflate TRISC text). This
is a debug-build size limit, not a correctness failure; production mode passes.

## Registry Conformance

- `INPUT_TAGGERS` (`alignment`, `sequence_mode`), both `(inputs, axes)`; `SUPPORTED` covers dtype,
  layout, fp32_dest_acc_en and both tagger axes; `EXCLUSIONS = []`; `validate()` order: shape checks
  -> axes -> SUPPORTED (`UnsupportedAxisValue`) -> EXCLUSIONS (`ExcludedCell`) -> L1 predicate. The
  entry point calls `validate()` first. The op file declares no `INVALID`.
- Auto-fix applied: `batched_prefill` added to `SUPPORTED["sequence_mode"]` (probe evidence above).
- INVALID audit (`feature_spec.py`): `INVALID = []`. TARGET has neither bfloat8_b nor ROW_MAJOR, so
  the bf8b+RM rule is vacuous; no cross-tensor entries; not a norm with optional weights. Well formed.

## L1 Ledger Audit

- **Currency**: the main table was stale against the build (cb_sq listed as `max(Kt, Vt)` and shared
  with the RMS phase, cb_rinv shared, cb_sq_o / cb_rinv_o / cb_conv_acc / cb_u_f32 / cb_scratch only
  in an appendix, footprint formula and total wrong: 223 pages claimed vs 217 allocated). Rows merged
  into the main table, the formula rewritten to match `_cb_plan`, totals corrected (now 216 pages +
  2 KiB = 886784 B at Kt = Vt = 4 after the cb_conv_acc fix).
- **Capacity vs live set**: cb_conv_acc 2 vs 1, fixed. cb_window holds `4*max(Kt,Vt)*WINDOW_DEPTH`
  pages while the hand-off is 4 pages: it is 2 groups of reader run-ahead, which is the design's
  depth knob, so it is justified; it changes with the hand-off rework in Refinement 2. cb_state
  (Kt*Vt pages, 64 KiB at the chip geometry) is never touched when T = 1 (decode goes cb_state_in ->
  cb_state_out); harmless at 0.61 of L1, noted for Refinement 3, which resizes the state buffers.
  cb_conv_w spends 3/4 of its pages on tap expansion (design perf lamp); folded into Refinement 2.
- **Page format vs DEST width**: every page is Float32 with `fp32_dest_acc_en = True` in every
  cell. No over/under finding. Refinement 1 must give the token-tensor CBs (window, gate_in, z,
  conv_w, gate_w, norm_w, out_rows, scratch) the input dtype; intermediate CBs stay Float32.
- **Disjoint lifetime**: cb_sq / cb_sq_o and cb_rinv / cb_rinv_o are disjoint and justified (shared
  FIFO left a block wrapped; measured wrong RMS factor). No unjustified pair.
- **Bounds**: Kt, Vt bounded by the host predicate `l1_footprint_bytes(Kt, Vt) <= L1_BUDGET`; no op
  dimension (B, T, N, Hv, C) appears in any capacity.
- **Block-size defaults**: interleaved, work units (pairs) spread over the grid with
  `split_work_to_cores(row_wise=True)`; per-pair block is the whole state (coarsest). Held.
- **Footprint**: `4096 * (Kt*Vt*(STATE_IN_DEPTH + 1 + STATE_OUT_DEPTH) + 14Kt + 11Vt + 2*ROW_DEPTH*Vt
  + 4*max(Kt,Vt)*WINDOW_DEPTH + CONV_ACC_DEPTH + 2*ROW_DEPTH + 3*TOK_COEF_DEPTH + 9) + 2048`. State
  term scales with Kt*Vt and the state depths; the rest linear in Kt, Vt and the depth knobs.
- **Data-movement budget**: present and consistent with the implemented R-STEP split (q/k columns
  read r times: 1.8 crossings). The cheaper R-BCAST split is a `deferred` regime row with a positive
  reason (4 % of decode bytes; prefill bound by ordered steps). Accepted.

## Precision Baseline

`test_gated_delta_net_precision_baseline.py`, float32, HiFi4, `fp32_dest_acc_en=True`, seed 0.

| Shape | Output | PCC | Max Abs Err | Mean Abs Err | Rel RMS | ULP p99 | got/true median (p5, p95) |
|-------|--------|-----|-------------|--------------|---------|---------|---------------------------|
| decode B=1 small (1,3,32,64) | output | 0.99999983 | 2.29e-3 | 2.24e-4 | 5.8e-4 | 395738 | 0.99993 (0.9976, 1.0026) |
| | state | 0.99999993 | 1.70e-3 | 6.90e-5 | 3.8e-4 | 193844 | 0.99997 (0.9981, 1.0012) |
| decode B=1 chip (1,12,128,128) | output | 0.99999977 | 5.33e-3 | 1.88e-4 | 7.1e-4 | 455319 | 0.99971 (0.9967, 1.0027) |
| | state | 0.99999998 | 1.85e-3 | 2.22e-5 | 1.8e-4 | 76310 | 0.99999 (0.9988, 1.0004) |
| prefill T=33 small | output | 0.99999942 | 1.41e-2 | 2.66e-4 | 1.11e-3 | 674754 | 0.99981 (0.9958, 1.0042) |
| | state | 0.99999920 | 3.20e-3 | 1.85e-4 | 1.34e-3 | 673332 | 0.99916 (0.9950, 1.0039) |
| prefill T=512 small | output | 0.99999946 | 1.47e-2 | 2.41e-4 | 1.07e-3 | 707572 | 0.99977 (0.9962, 1.0033) |
| | state | 0.99999949 | 3.43e-3 | 9.93e-5 | 1.47e-3 | 525664 | 0.99895 (0.9950, 1.0027) |
| prefill T=64 chip | output | 0.99999970 | 1.24e-2 | 1.83e-4 | 8.1e-4 | 469230 | 0.99977 (0.9969, 1.0024) |
| | state | 0.99999973 | 4.44e-3 | 3.85e-5 | 1.15e-3 | 316864 | 0.99900 (0.9972, 1.0008) |

**Assessment**: every output is an order of magnitude inside the float32 row (PCC >= 0.999,
rel RMS <= 0.01). Prefill states show a small negative bias (median got/true ~0.999) with a broad
p5..p95 spread: consistent with the 19-bit srcA/srcB truncation of the increment matmul operands
(truncation toward zero), which the spec allows. Not a scale bug (no tight cluster away from 1;
rel RMS 1e-3). state_carry_f32 (graded at 0.002) passes at 6.3e-4. ULP p99 is large in absolute
count because many reference values sit near zero after RMS normalization; it is recorded, not gated.
**Recommended tolerances**: defaults hold.

## Verifier CLI Summary

- supported_pass: 64 (61 before the `batched_prefill` drift fix)
- xfail_expected: 35 (all `dtype=bfloat16`: 29 INPUTS cells + 6 bf16 LOOSE_CASES incl. the flagged
  `perf_decode_b1`) -> Refinement 1
- invalid_skipped: 0
- supported_fail: 0
- xpass_drift: 0
- xfail_wrong_mode: 0
- no_axes_found: 9 (`test_regression.py`). Under `eval_test_runner.sh` 8 of them fail with
  `L1ProfilingError` (peak 0 / no device fixture) because they test refusals and never dispatch:
  a harness artifact. Under `run_safe_pytest.sh` the regression file passes 36/36.

Measured device time (fp32, golden run): decode B=1 chip 106 us (12 cores), decode B=16 chip 288
us (110 cores), prefill T=4096 chip 90.5 ms (12 cores, ~22 us per token step), decode B=1 small
44 us. From decode vs prefill per-token cost, the per-pair prep (gates, conv, L2 norm, RMS norm and
the reader's window assembly) is roughly 80 of the 106 us at the chip geometry; the token step
roughly 20 us.

## Recommendations

- Queue: Refinement 1 (bfloat16) lands the whole contract of the flagged `perf_decode_b1` case;
  after it the TARGET gap is empty, so every later phase is perf on that case (goal 5.3 us,
  currently ~106 us in fp32: about 20x).
- Prefill throughput (90.5 ms at T=4096) is dominated by the ordered token step. R-CHUNK (and
  `TOKEN_TILE_ROWS_PER_BLOCK > 1`, the collapsed knob above) are the levers, but no prefill shape is
  perf-flagged, so they are not queued.
- The bf16 refinement must solve 32 B row-segment writes into tile rows shared by several
  sequences (decode B > 1, batched_prefill): see Refinement 1 notes.
