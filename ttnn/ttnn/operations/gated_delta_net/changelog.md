# Changelog: gated_delta_net

## Phase 0 — Core Implementation
- **Date**: 2026-11-15
- **What was done**: Initial implementation via incremental pipeline (planner -> implementer -> verifier). Regime R-STEP: one (sequence, value head) pair per work unit over the full grid, state resident per pair, one token per recurrence step, one `generic_op` dispatch per call.
- **SUPPORTED at Phase 0**: dtype=[float32], layout=[TILE], fp32_dest_acc_en=[True], alignment=[tile_aligned, h_non_aligned], sequence_mode=[decode, prefill, batched_prefill]
- **Accuracy achieved** (float32, 5 shapes, `test_gated_delta_net_precision_baseline.py`): output PCC >= 0.9999994, max_abs_err <= 1.5e-2, rel RMS <= 1.1e-3; recurrent_state_out PCC >= 0.9999992, max_abs_err <= 4.4e-3, rel RMS <= 1.5e-3
- **Golden suite at Phase 0**: 64 supported_pass, 35 xfail_expected (all bfloat16), 0 supported_fail / xpass_drift / xfail_wrong_mode (`verifier_report.json`)
- **Issues encountered** (verifier fixes): `batched_prefill` worked in the kernels but was refused by SUPPORTED (added); CB indices and spec constants restated in all three kernels (now host `GDN_*` defines); cb_conv_acc depth 2 -> 1; unused reader runtime arg removed; L1 ledger table and footprint brought in line with the build.
- **Tests added**: test_gated_delta_net.py (planner), test_gated_delta_net_precision_baseline.py, test_gated_delta_net_batched_prefill.py

## Refinement 1 — bfloat16 token tensors (lands the perf_decode_b1 contract)
- Date: 2026-11-15
- What was done: `SUPPORTED["dtype"]` = [float32, bfloat16]. Token CBs (`TOKEN_CBS`: conv_w, window, gate_in, gate_w, z, norm_w, out_rows, scratch) take the token dtype and page size; every compute intermediate and the state stay Float32. `UnpackToDestFp32` is set only on Float32 buffers (explicitly reset to Default on bf16 token CBs, so a reused caller config does not carry fp32 tags). Reader/writer tile, face and segment bytes are host compile-time args (token tile bytes, state tile bytes) instead of fp32 literals. bf16 face rows are 32 B, below the Blackhole 64 B DRAM read alignment: the reader's new `RowReader` reads each half-row as its aligned 64 B chunk into `cb_row_stage` (allocated only when needed) and copies it L1-to-L1, batched up to `ROW_STAGE_SLOTS` per barrier; fp32 keeps direct 64 B row reads (compile-time branch). Row writes stay 32 B direct (Tensix-to-DRAM writes need only L1 alignment); output / conv_state_out rows of one 64 B chunk written by different cores were checked bit-identical over 8 repeated calls with the watcher NoC sanitizer on. Compute kernel unchanged apart from a comment. No ttnn ops added; still one generic_op dispatch.
- Reused: the shared reader/writer/compute paths, CB plan, accessors, row-segment logic. Added: dtype-derived CB formats + geometry CT args, `RowReader` staging, `cb_row_stage`.
- Accuracy achieved (bf16 tokens, fp32 state, HiFi4, fp32 DEST): output PCC >= 0.999998, relative RMS <= 1.9e-3; recurrent_state_out PCC >= 0.9999997, relative RMS <= 1.2e-3; conv_state_out bit-exact; shapes [(1,1,2560)/(1,12,128,128), (1,16,2560)/(16,12,128,128), (1,64,2560)/(1,12,128,128), (1,10,256)/(2,3,32,64)].
- Perf: perf_decode_b1 (bf16) 101.9 us device kernel time on 12 cores (Phase 0 fp32 equivalent ~106 us). Baseline for Refinement 2.
- Golden test progress: 35/35 bf16 cells pass (29 test_op + 6 LOOSE_CASES incl. perf_decode_b1, state_carry_bf16); fp32 paths covered by the unit dir (51/51 pass).
- Issues encountered: `--dev` (asserts) overflows the kernel-config text buffer by 704 B on the bf16 build (pre-existing Phase 0 --dev limit); the sanitizer check was run with watcher on and asserts off instead.
- Tests added: test_gated_delta_net_bf16.py (decode B=1/3/40, batched T=5 odd boundary, prefill T=33; clean and inf padding; 8 repeated calls bit-identical).
