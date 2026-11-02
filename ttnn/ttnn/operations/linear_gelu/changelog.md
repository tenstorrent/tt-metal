# Changelog: linear_gelu

## Phase 0 - Core Implementation
- **Date**: 2026-11-02
- **What was done**: Initial implementation via incremental pipeline (planner -> implementer -> verifier). Regime R1 (2D grid, W slice resident per core via column multicast, full K + bias tile in fp32 DEST, accurate tanh GELU on the packer) plus R2 (x row-block multicast along grid rows, built early on measurement: 8x512 393 -> 195 us).
- **SUPPORTED at Phase 0**: dtype=[bfloat16], layout=[TILE], alignment=[tile_aligned], fp32_dest_acc_en=[True], memory_layout=[INTERLEAVED], buffer_type=[L1], out_buffer_type=[L1]
- **Accuracy achieved**: PCC=0.99997, max_abs_err=5.5e-2, rel_rms_err=1.17e-2..1.18e-2 (measured on 4 shapes via test_linear_gelu_precision_baseline.py; stock 1.21e-2..1.30e-2)
- **Perf at Phase 0 (device us)**: 8x512 195.6, 16x256 195.7, 32x128 195.7, 8x384 155.0, 4x512 114.1, 8x256 114.2, 2x512 73.1, 4x256 73.0 (beats stock on every row; target 90.8 at 8x512)
- **Golden suite at Phase 0**: 23 / 23 supported cells passing, 73 xfail_expected, 0 loud (per `golden_results/verifier_report.json`)
- **Issues encountered**: verifier code-review fixes: deduplicated config/placement resolution (`_resolve`), moved the output memory_layout refusal ahead of SUPPORTED, hoisted the loop-invariant subblock derivation. No drift fixes.
- **Tests added**: test_linear_gelu.py, test_linear_gelu_perf.py, test_linear_gelu_precision_baseline.py

## Refinement 1 - Non-aligned S + DRAM placement for x and output
- Date: 2026-11-02
- What was done: writer stage W3. For S % 32 != 0, rows `S % 32`..31 of every tile in a tail tile row (`m % seq_tiles == seq_tiles - 1`) are zeroed in the `cb_output_tiles` page via `noc.async_write_zeros` before the NoC write, with one L1 barrier per tail-row subblock row. `seq_tiles` and `last_valid_rows` are writer CT args, so aligned S compiles the path out. DRAM x / output needed no kernel change (TensorAccessor, R2 x multicast reads x once). SUPPORTED widened: alignment += h_non_aligned, buffer_type += DRAM, out_buffer_type += DRAM.
- Accuracy achieved: PCC >= 0.9999 on all golden cells (unchanged path); padded rows exactly 0 with NaN / Inf padding in x.
- Golden test progress: 96/96 (73 newly supported cells pass, 23 prior pass), regression file 6/6.
- Perf (device us, 8x512): L1/L1 195.4 (no regression vs 195.6), DRAM x 199.4, DRAM out 196.2, 8x500 non-aligned 195.5.
- Issues encountered: None.
- Tests added: `test_linear_gelu_perf.py::test_linear_gelu_perf_guard` (8x512 DRAM x, 8x512 DRAM out, 8x500 non-aligned).
