# Changelog: rotary_embedding_llama

## Phase 0 — Core Implementation
- **Date**: 2026-11-21
- **What was done**: Initial implementation via the incremental pipeline (planner → implementer → verifier), for Quasar on the ttsim simulator.
  - One `ttnn.generic_op` dispatch of a Metal 2.0 `ProgramSpec`. Every tensor's L1 shard backs a borrowed, zero-copy ring.
  - Reader and writer post and drain credits only; there is no NoC traffic and no DRAM traffic.
  - Compute is `x @ T` (raw LLK matmul) → `rotated` → `eltwise_chain` (row-broadcast `rotated·sin` + `x·cos`, SFPU add), packed straight into the output shard.
  - Thread plan `(1, 4, 1)`, with C clamped to divide the head_dim tiles.
- **SUPPORTED at Phase 0**: dtype=[bfloat16], fp32_dest_acc_en=[True], layout=[TILE], memory_config=[L1], sharding=[height], is_decode_mode=[True], users=[1], alignment=[tile_aligned, h_non_aligned]. EXCLUSIONS: `{float32, fp32_dest_acc_en=False}` (R1).
- **Accuracy achieved**: PCC=0.999997 (worst), max_abs_err=0.226, rel_rms_err=0.0025 (worst), mean ULP ≤ 0.73. Measured on 5 shapes, (1,1,1,32) to (1,1,32,128), via `test_rotary_embedding_llama_precision_baseline.py`.
- **Golden suite at Phase 0**: 18 / 18 supported tests passing (12 cells + 6 numeric), 36 xfail_expected, 10 / 10 rejections, 0 in every loud category (`verifier_report.json`).
- **Issues encountered**:
  - Implementer: `rotated` is kept in the input dtype. This is folded into Refinement 1.
  - Verifier: added the R1 `EXCLUSIONS` entry ahead of float32 support, and removed the unused `CONTRACTION_TILES` constant.
- **Tests added**: `test_rotary_embedding_llama.py` (harness-generated acceptance), `test_rotary_embedding_llama_plan.py` (thread plan, clamp, session order), `test_rotary_embedding_llama_precision_baseline.py`.
