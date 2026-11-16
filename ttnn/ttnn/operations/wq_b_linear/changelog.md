# Changelog: wq_b_linear

## Phase 0 — Core Implementation
- **Date**: 2026-11-16
- **What was done**: Initial implementation via incremental pipeline (planner → implementer → verifier). Regime R1 `n_split_x_bcast_resident`: n split over the full grid (110 cores on p100a), x read once and multicast to every core (8 rotating injectors, Flag + handshake), W streamed per core whole-K resident, one `matmul_block` per core with packer L1 accumulation, y written to L1 interleaved. One `ttnn.generic_op` dispatch.
- **SUPPORTED at Phase 0**: dtype=[bfloat16], weight_dtype=[bfloat8_b], layout=[TILE], fp32_dest_acc_en=[False] (== TARGET; one cell)
- **Accuracy achieved**: PCC=0.999968, max_abs_err=0.045, mean_abs_err=0.0051, rel_rms_err=0.0083, ULP p99≈85 bf16 ULP (standard distribution; 8 distribution/seed cases via test_wq_b_linear_precision_baseline.py, all PCC ≥ 0.999968)
- **Performance**: 26.6–27.0 µs device kernel time vs 16.0 µs SOL (1.66–1.69×). Ablation: the x broadcast binds (25.4 µs alone; one-page sync floor 14.7 µs); the W stream alone takes 20.8 µs.
- **Golden suite at Phase 0**: 2 / 2 registry cells passing + 1/1 regression (per `verifier_report.json`); 0 supported_fail / xpass_drift / xfail_wrong_mode
- **Issues encountered**: verifier fixes: removed the unused `x_cb_k_blocks` / `w_cb_k_blocks` host aliases (kernels assume whole-K residency); recorded the as-built knob values in `op_design.md`; updated `l1_ledger.md` currency and the R5 deferral note (measurement falsifies "cross-core tier does not bind").
- **Tests added**: test_wq_b_linear.py (planner), test_wq_b_linear_precision_baseline.py, _op_bench.py (perf ablation bench, not collected by default)
