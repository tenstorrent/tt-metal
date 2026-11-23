# Changelog: wq_b_linear

## Phase 0 — Core Implementation
- **Date**: 2026-11-23
- **What was done**: Initial implementation via the incremental pipeline (planner → implementer → verifier). One `generic_op` program:
  - x is read once from DRAM by a sender core (NCRISC/NoC0) and multicast to 64 workers (BRISC/NoC1, Counter data-ready, no handshake, x resident).
  - Each worker streams its 2 bank-spread W columns, runs one `matmul_block` over K with packer L1 accumulation, and stores 5×2 output tiles to L1-interleaved y.
- **SUPPORTED at Phase 0**: dtype=[bfloat16], weight_dtype=[bfloat8_b], layout=[TILE], fp32_dest_acc_en=[False]. This equals TARGET (one cell).
- **Accuracy achieved** (`test_wq_b_linear_precision_baseline.py`, 4 input profiles on the single shape, golden distribution):
  - PCC=0.999931, max_abs_err=0.090, mean_abs_err=0.0083, rel_rms=0.0139, ULP p99≈105.
  - Global scale 1.0069, from 16-bit DEST accumulation over `block_k_tiles = 8`.
- **Performance**: device span 26.6 µs (p150, 8 banks), 1.9× the 14.0 µs SOL. The op is x-delivery-bound: the single sender reads x in 14.6 µs and finishes multicasting at 22.4 µs.
- **Golden suite at Phase 0**: 2 / 2 registry cells passing, plus regression (bit-identical repeats). supported_fail / xpass_drift / xfail_wrong_mode = 0.
- **Issues encountered** (verifier fixes):
  - The DEST subblock limit and the compute `fp32_dest_acc_en` now derive from the compute config instead of constants.
  - Documented the `block_k_tiles ≤ 16` precision ceiling: PCC falls below 0.9998 at kb ≥ 24, from 16-bit DEST drift.
  - Corrected stale kb-4 figures in `l1_ledger.md` and appended the verifier audit.
- **Tests added**: `test_wq_b_linear_precision_baseline.py`, `_op_bench.py` (device-time bench for perf phases, not collected by default).
