# Changelog: matmul_reduce_scatter

## Phase 0 — Core Implementation
- **Date**: 2026-10-06
- **What was done**: Initial implementation via incremental pipeline (planner → implementer → verifier). One
  `ttnn.generic_op` per call: per chip a 2D-multicast matmul on the compute rectangle walks the G scatter blocks in
  transport order (regimes R1 resident invariant operand / R2 streamed), parks each block in L1, and the transport
  row (ports + finals, `4·num_links` cores) runs the line reduce-scatter over Fabric with fp32 adds.
- **SUPPORTED at Phase 0**: dtype=[bfloat16], weight_dtype=[bfloat16, bfloat8_b], layout=[TILE],
  alignment=[tile_aligned], cluster_axis=[0, 1], scatter_dim=[-1, -2], topology=[Linear], num_links=[1, 2];
  EXCLUSIONS=[].
- **Accuracy achieved** (`test_matmul_reduce_scatter_precision_baseline.py`, 6 configs, worst of 8 chips):
  fp32 DEST: PCC ≥ 0.99999, rel-RMS 0.0035–0.0054, max_abs ≤ 0.10. FOCUS at production precision (HiFi2,
  `fp32_dest_acc_en=False`): PCC 0.99986, rel-RMS 0.024 (gate 0.055; unfused production 0.036), max_abs 0.29. It
  carries a K-dependent bf16-accumulation bias (slope 1.016 at K=2048), triaged as rounding and not a structural bug.
- **Perf** (device kernel duration, steady state, max over chips): FOCUS 209.5 us (unfused 394, target 118),
  GLM 241.9 us (409 / 198), MiMo 439.3 us (385 / 215).
- **Golden suite at Phase 0**: 375 supported_pass / 1041 cells, 660 xfail_expected (= TARGET − SUPPORTED),
  0 supported_fail, 0 xpass_drift, 0 xfail_wrong_mode; 6/6 registry-free regression tests pass (per
  `verifier_report.json`).
- **Issues encountered** (verifier fixes):
  - Persistent per-plan L1 `handoff_l1` shard leaked across plan keys and clashed with later plans' CB regions
    (~600 golden failures). Now allocated per call; generic_op re-points the CB on a cache hit.
  - Per-plan global semaphores leaked the same way. Now one set per `(mesh, cluster_axis, num_links)` over the whole
    worker grid.
  - L1 CB budget used `get_max_worker_l1_unreserved_size()` (includes the kernel-config ring) and over-budgeted by
    70.6 KB, so R1 `scatter_dim=-2` plans overflowed into the hand-off shard. Now uses the allocator's L1 bank size.
  - DRY: accumulator dtype / tile size and DEST capacity each defined once.
- **Tests added**: `test_matmul_reduce_scatter.py` (acceptance, planner), `test_matmul_reduce_scatter_debug.py`,
  `test_matmul_reduce_scatter_perf.py` (implementer), `test_matmul_reduce_scatter_precision_baseline.py` (verifier).
