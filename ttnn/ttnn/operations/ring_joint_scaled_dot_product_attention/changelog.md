# Changelog: ring_joint_scaled_dot_product_attention

## Phase 0 — Core Implementation
- **Date**: 2026-11-03
- **What was done**: Initial implementation via incremental pipeline (planner → implementer → verifier). Regime R1
  `qstationary_dram_kv`: one `generic_op` over a `MeshProgramDescriptor`. Port cores (TRANSPORT, adapted from
  `examples/fabric_all_gather`) relay K/V head slices around the group into persistent DRAM landing buffers;
  compute cores (OVERLAP) run Q-stationary online-softmax attention over local → joint → remote key segments,
  gating each remote segment on the landing counters.
- **SUPPORTED at Phase 0**: dtype=[bfloat16], kv_dtype=[bfloat16], fp32_dest_acc_en=[True], layout=[TILE],
  alignment=[tile_aligned], attention_kind=[self], kv_heads_mode=[mha], joint_len=[none, aligned],
  joint_mode=[none, replicated], mask_mode=[none], is_balanced=[False], seq_padding=[exact, padded],
  cluster_axis=[0, 1], topology=[Linear, Ring]
- **Accuracy achieved** (default config; 8 shape × placement cells via
  `test_ring_joint_scaled_dot_product_attention_precision_baseline.py`, worst device): PCC ≥ 0.999997,
  max_abs_err ≤ 0.0037, rel_rms_err ≤ 0.0028, bf16 ULP p99 ≤ 18.3, got/true ratio median 0.9988 (broad ±0.5 %
  band: rounding noise, no scale bug). Large logits (`test_large_logits`): rel-RMS 0.0045.
- **Golden suite at Phase 0**: 110 supported_pass / 0 supported_fail / 1705 xfail_expected / 5984 invalid_skipped,
  0 xpass_drift / 0 xfail_wrong_mode (per `verifier_report.json`, run with `EVAL_CAPTURE_L1=0`, see below).
- **Measured** (perf-flagged Wan2.2 shape at the fp32-DEST proxy, 1×4 ring, 2 links): ≈ 198 ms per call. Transport
  done by 2.3–4.5 ms (hidden); local compute ≈ 7× its roofline. Drives the perf queue order.
- **Issues encountered** (verifier fixes):
  - Race: the compute reader polled port arrival counters into its own `sem_arrival` word. A core re-placed as a
    port in a later call (topology / num_links change) then started with a stale count, and its relay and
    consumers read landing slots too early (golden ring-mock 1×4 joint cells after Linear 2-link cells: PCC
    0.71–0.84). Fixed with a dedicated poll-scratch global semaphore. Regression test added.
  - Precision: `test_large_logits` failed the 0.02 rel-RMS gate at HiFi2 + fp32 DEST (0.0206). The op now runs at
    ≥ HiFi3 when `fp32_dest_acc_en=True` (precision-floor rule), giving 0.0045. bf16 DEST is unchanged.
  - DRY: the L1 fit and the CB allocation now share one `compute_cb_plan()`.
  - Program hash now covers the output `memory_config` (compile-time `TensorAccessorArgs`).
  - `l1_ledger.md` updated to the implemented fixed-role accumulators and the budget constant.
  - Harness limitation recorded, not fixed (outside the op): with `EVAL_CAPTURE_L1=1` every supported cell fails
    `L1ProfilingError: peak_total=0`, because the graph tracker is `thread_local` and multi-program mesh
    workloads compile on the mesh thread pool.
- **Tests added**: `test_ring_joint_scaled_dot_product_attention_precision_baseline.py`,
  `test_ring_joint_scaled_dot_product_attention_placement_switch.py`, `probes/focus_breakdown.py` (perf-breakdown
  probe, not collected). Existing: `test_ring_joint_scaled_dot_product_attention.py` (acceptance, 15 cases),
  `test_ring_joint_scaled_dot_product_attention_debug.py`.
