# Changelog: scaled_dot_product_attention

## Phase 0 — Core Implementation
- **Date**: 2026-06-29
- **What was done**: Initial implementation via incremental pipeline (planner → implementer → verifier). Flash Attention v2 online-softmax recurrence with tiled matmul + eltwise + reduce helpers. Single-core per (B,H) work unit, embarrassingly parallel via split_work_to_cores.
- **SUPPORTED at Phase 0**: dtype=[bfloat16], fp32_dest_acc_en=[True], layout=[TILE], alignment=[tile_aligned], attention_kind=[self, cross], kv_heads_mode=[mha, gqa, mqa], mask_mode=[none, custom], scale_mode=[auto, explicit]
- **Accuracy achieved**: PCC=0.999997, max_abs_err=0.004569, rms_err=0.002663 (measured on 4 shapes via test_scaled_dot_product_attention_precision_baseline.py)
- **Golden suite at Phase 0**: 200 / 2767 cells passing (per `verifier_report.json`); 8 OOM on D=512/D=1024; 2440 xfail_expected
- **Issues encountered**:
  - Fixed: Missing running-max step in online softmax (Phase 4b). The kernel used `m_blk` directly as `m_new` instead of `max(m_old, m_blk)`, causing `alpha = inf` when a KV-block had all -inf scores (fully masked). Added `BinaryMax` eltwise_chain between row-max and alpha computation.
  - Fixed: Partial last Q-block/KV-block out-of-bounds read. `B_q_t` did not divide `S_q_tiles` for shapes like S=192 (6 tiles, B_q_t=4). Added divisor-reduction loop in the program descriptor.
  - 8 OOM cells on D=512/D=1024 head dims (cb_o + cb_o_accum exceed 1.5 MB L1). Filed as Refinement 4 (/memory-budget-metal).
- **Tests added**: test_scaled_dot_product_attention.py (acceptance), test_scaled_dot_product_attention_precision_baseline.py, test_scaled_dot_product_attention_extended.py
