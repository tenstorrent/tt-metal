# Changelog: qkv_proj_heads_rope

## Phase 0 - Core Implementation
- **Date**: 2026-11-02
- **What was done**: Initial implementation via incremental pipeline (planner -> implementer -> verifier). Regime R1: 2D token_rows x head_cols output-stationary grid (108/110 cores on p100a), unicast reads, matmul with the bias folded as the 25th K tile into Float32 L1-accumulated partials, rotary as two DEST-accumulated FPU multiplies per output tile, one `generic_op` dispatch.
- **SUPPORTED at Phase 0**: dtype=[bfloat16], layout=[TILE], alignment=[tile_aligned], fp32_dest_acc_en=[True], memory_layout=[INTERLEAVED], buffer_type=[L1], out_buffer_type=[L1], table_buffer_type=[L1]
- **Accuracy achieved**: PCC >= 0.9999984, max_abs_err <= 0.0212, rel_rms <= 1.91e-3 (gate 4e-3), ULP p99 <= 4.9 (measured on 4 shapes via test_qkv_proj_heads_rope_precision_baseline.py); q/k got/true median 0.99925 (truncation bias of the rotary FPU multiply, inside tolerance), v 1.0000
- **Golden suite at Phase 0**: 25 supported_pass / 175 xfail_expected / 0 loud of 210 (per `verifier_results/verifier_report.json`, run with a local, reverted patch for two golden-harness defects; see verification_report.md)
- **Perf at Phase 0**: 8x512 279-281 us (stock 226.3, target 90); every flagged row slower than stock
- **Issues encountered**: verifier DRY fix: the reader derived the table row size and page index from literals; `TILES_PER_HEAD` is now a reader CT arg. Golden harness defects (`packer_l1_acc` kwarg, tuple tolerance) reported, not edited.
- **Tests added**: test_qkv_proj_heads_rope.py, test_qkv_proj_heads_rope_perf.py, test_qkv_proj_heads_rope_precision_baseline.py
