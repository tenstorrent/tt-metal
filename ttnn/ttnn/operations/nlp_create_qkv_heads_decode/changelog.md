# Changelog: nlp_create_qkv_heads_decode

## Phase 0 — Core Implementation
- **Date**: 2026-11-21
- **What was done**: Initial implementation via incremental pipeline (planner → implementer → verifier). Regime A of `op_design.md` (direct gather, input interleaved in L1), on Quasar / ttsim. One `ttnn.experimental.ProgramSpec`, one `ttnn.generic_op` dispatch, one data-movement kernel `gather` with thread plan (R, C, W) = (2, 0, 0).
- **SUPPORTED at Phase 0**: dtype=[bfloat16], layout=[TILE], memory_config=[L1], sharding=[interleaved], output_memory=["L1/height"], users=[1], head_rows=[32]
- **Accuracy achieved**: PCC=1.000, max_abs_err=0.0, rms_err=0.0, max ULP=0 (bit-exact; measured on 4 shapes via test_nlp_create_qkv_heads_decode_precision_baseline.py)
- **Golden suite at Phase 0**: 8 / 8 supported cells passing, 72 xfail_expected, 8 / 8 rejections passing, 0 loud (per `pipeline_logs/verifier_golden/verifier_report.json`)
- **Issues encountered**: none failing. Verifier code-review fixes, behaviour unchanged:
  - Tile geometry (`TILE`, `FACE`, `FACE_ROWS_PER_TILE`) now has one host source, the program-artifacts module.
  - The kernel derives `input_width_tiles` from `sections * head_dim_tiles`. It was a duplicate CT arg.
  - The empty compute and writer placeholder kernels are removed (they were never part of the program).
- **Tests added**: test_nlp_create_qkv_heads_decode.py (generated acceptance), test_thread_plan.py, test_nlp_create_qkv_heads_decode_precision_baseline.py

### Implementer notes (Phase 0)

- **Host.** `gather` (`kernels/nlp_create_qkv_heads_decode_reader.cpp`) runs on the shard grid of rule R5 with `num_threads = GATHER_THREADS = 2`. There is no dataflow buffer, scratchpad or semaphore (op-allocated L1 = 0 B).
- **Kernel.** Every face-row run of the user's input row is one `NocOptions::TXN_ID` read (trid 0), from the interleaved input page straight into its final place in the local q / k / v shard (`LocalTensorAccessor` destination).
  - Runs are dealt round-robin over the harts. With R = 2, hart 0 moves the left-face runs and hart 1 the right-face runs.
  - One trid-0 barrier per block. `heads_per_block` (`HEADS_PER_BLOCK`; default: the whole user) is a live knob.
  - The last statement resets the hart's tag register.
- **Structural counts (bf16).** Llama [1,1,1,3072]: 192 reads of 32 B, 96 per hart, 1 barrier per hart. Qwen [1,1,1,6144]: 384 reads, 192 per hart. 0 DRAM bytes. No timed measurement on ttsim.
