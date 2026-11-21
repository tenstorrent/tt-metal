# Changelog: linear

## Phase 0 — Core Implementation
- **Date**: 2026-11-21
- **What was done**: Initial implementation via incremental pipeline (planner → implementer → verifier). Regime
  R-2D-stream on Quasar (ttsim): `[block_h × block_w]` output blocks accumulated in DEST over the whole of K, the
  bias folded into DEST, rounds of C = 4 blocks dealt to clusters; one `ttnn.generic_op` (ProgramSpec) dispatch.
- **SUPPORTED at Phase 0**: dtype=[bfloat16], fp32_dest_acc_en=[True, False], layout=[TILE],
  memory_config=[DRAM, L1], sharding=[interleaved, width], output_memory=[DRAM, L1/width, L1/width/auto],
  bias=[None, row, vector], dtype_argument=[same], alignment=[tile_aligned], rank=[4],
  weight_form=[matrix, same_rank], batch=[single]
- **Accuracy achieved**: fp32 DEST: PCC ≥ 0.999999, max_abs_err ≤ 0.50, rel RMS ≤ 0.0017, ULP p99 = 1;
  bf16 DEST (worst, K = 2048): PCC = 0.99987, max_abs_err = 10.5 (ref max 236), rel RMS = 0.016, ULP p99 = 82
  (measured on 4 shapes × 2 DEST widths + ResNet LoFi via test_linear_precision_baseline.py)
- **Golden suite at Phase 0**: 331 supported_pass (324 cells + 1 loose + 6 regression), 2916 xfail_expected,
  1080 invalid_skipped, 7 rejections; 0 supported_fail / xpass_drift / xfail_wrong_mode (`verifier_report.json`)
- **Issues encountered** (verifier):
  - R2: `resolve_compute_config` restated the default config in its `getattr` fallbacks; they now read
    `default_compute_kernel_config()`.
  - Removed the no-op `_dm_config` wrapper.
  - Found by probe: K padding is not masked (exact only for zero-padded inputs); queued in Refinement 1.
  - float32 needs `unpack_modes` on FP32 rings; queued in Refinement 2.
- **Tests added**: test_linear.py (harness), test_linear_plan.py (implementer), test_linear_precision_baseline.py
  (verifier)


### Implementer record

Built regime **R-2D-stream** of `op_design.md` as designed, for the whole PHASE0 rectangle.

- Host: `linear.py` (registry declarations, `validate()`, `default_compute_kernel_config()`), and
  `linear_program_artifacts.py` (Metal 2.0 `ttnn.ProgramSpec`, one `ttnn.generic_op` dispatch): `select_blocking`
  exactly as the design states it, whole rounds of C blocks dealt to clusters with `split_work_to_cores`,
  `THREAD_PLAN = (act_readers=1, weight_readers=1, compute=4, writers=2)` as one host constant.
- Kernels: `linear_reader.cpp` (activation → ALL ring), `linear_weight_reader.cpp` (weight → BLOCKED ring, bias →
  BLOCKED ring), `linear_compute.cpp` (bias preload into DEST by `unary_bcast<ROW>`, `matmul_block` over K chunks
  into DEST, one `pack_block` per round), `linear_writer.cpp`. All data-movement kernels use implicit sync and hand
  the transaction tag back at exit. The design's "R" of two is two kernels of one hart each, so the reader
  appears as two files.
- Raw compute API instead of kernel-library helpers, for the reasons in the design's "Helpers considered and
  rejected" table (recorded at the head of `linear_compute.cpp`): `matmul_block_helpers.hpp` and
  `bias_add_helpers.hpp` are not used, and an eltwise chain cannot act on an accumulator a raw matmul
  left in DEST.
- The ALL and BLOCKED rings with a Tensix consumer, which the Quasar examples do not cover, work with
  `matmul_block` at C = 1, 2 and 4 on ttsim (bit-identical results across plans).

### Design deviations / findings

- `op_design.md` Work Distribution, *Tests the plan needs* (1), asks for a `(2, 1, 2)` plan. W = 2 does not divide
  C = 1 (target page: on STRIDED rings W divides C), so the test uses `(2, 1, 1)`, plus `(2, 2, 2)` and `(2, 4, 1)`
  against the design's `(2, 4, 2)`.
- On the 32-cluster grid every Phase 0 shape picks 1 × 1 blocks, so the multi-tile block path is exercised by
  `test_forced_blocks` (blocks up to 4 × 2 and 2 × 4, with and without bias, both DEST widths).

### Results (ttsim, Quasar)

- Acceptance test: 163 passed.
- Golden suite: 324 first-phase cells passed (108 INVALID skipped), the ResNet-50 LoFi loose case, 6 numeric
  regression cases and 7 rejections passed; all 2,916 out-of-phase cells xfail as refused, no XPASS.
- `test_linear_plan.py`: 25 passed (select_blocking pins on 32 and 2 clusters, every NEO's column, plans
  C = 1/2/4, ResNet's two K chunks with a width-sharded activation and an auto-sharded result, the device compute
  config, forced blocks, one device session with `ttnn.to_layout` after the op).

### Performance (structural, ttsim: no timing is a result here)

- Clusters engaged: ResNet 8, Qwen 16, Llama 1, 64x96 4 of 32; NEOs engaged per cluster: 4 (padded slots only in
  the last round of a row block). DRAM crossings as in `l1_ledger.md`'s data-movement budget. Any gain in time is
  unconfirmed until a timed backend measures it.
