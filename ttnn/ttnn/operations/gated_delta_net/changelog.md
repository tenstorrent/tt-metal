# Changelog: gated_delta_net

## Phase 0 — Core Implementation
- **Date**: 2026-11-15
- **What was done**: Initial implementation via incremental pipeline (planner -> implementer -> verifier). Regime R-STEP: one (sequence, value head) pair per work unit over the full grid, state resident per pair, one token per recurrence step, one `generic_op` dispatch per call.
- **SUPPORTED at Phase 0**: dtype=[float32], layout=[TILE], fp32_dest_acc_en=[True], alignment=[tile_aligned, h_non_aligned], sequence_mode=[decode, prefill, batched_prefill]
- **Accuracy achieved** (float32, 5 shapes, `test_gated_delta_net_precision_baseline.py`): output PCC >= 0.9999994, max_abs_err <= 1.5e-2, rel RMS <= 1.1e-3; recurrent_state_out PCC >= 0.9999992, max_abs_err <= 4.4e-3, rel RMS <= 1.5e-3
- **Golden suite at Phase 0**: 64 supported_pass, 35 xfail_expected (all bfloat16), 0 supported_fail / xpass_drift / xfail_wrong_mode (`verifier_report.json`)
- **Issues encountered** (verifier fixes): `batched_prefill` worked in the kernels but was refused by SUPPORTED (added); CB indices and spec constants restated in all three kernels (now host `GDN_*` defines); cb_conv_acc depth 2 -> 1; unused reader runtime arg removed; L1 ledger table and footprint brought in line with the build.
- **Tests added**: test_gated_delta_net.py (planner), test_gated_delta_net_precision_baseline.py, test_gated_delta_net_batched_prefill.py
