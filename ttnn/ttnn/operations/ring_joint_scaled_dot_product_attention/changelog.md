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

## Refinement 1 — Numerical configurability: bf16 DEST (perf-focus enabler) + bfloat8_b Q and K/V
- Date: 2026-11-03
- What was done: `SUPPORTED["dtype"]` and `["kv_dtype"]` += `bfloat8_b`, `SUPPORTED["fp32_dest_acc_en"]` += `False`. Reused:
  the existing dtype-derived CB plan (`compute_cb_plan`: Q / K / V CBs follow Q / K/V dtypes, scores / stats / O follow
  `fp32_dest_acc_en` → Float16_b at bf16 DEST, no Float32 page), the K/V-dtype landing buffers, port CB and `run_pages`, the
  bf16 output, the HiFi3 floor for fp32 DEST only (bf16 DEST keeps the requested HiFi2). Added: nothing in the kernels or
  descriptor; EXCLUSIONS stays empty. At bf16 DEST `choose_block_extents()` picks Sq=16, Sk=8 for the Wan2.2 case (8, 8 at fp32).
- Accuracy achieved (Linear G=2, worst device, HiFi2 requested): fp32 DEST all dtype pairs PCC ≥ 0.999997, rel-RMS ≤ 0.0026;
  bf16 DEST PCC ≥ 0.99985, rel-RMS 0.011–0.018 (gate 0.05 / 0.06 / 0.08) on shapes [1x1x64x64, 1x4x256x128 padded,
  1x10x512x128 padded, 1x6x512x128 + joint 512 padded]. Large logits (std ~36): fp32 DEST ≤ 0.0051; bf16 DEST 0.058–0.085
  for every dtype pair, also 0.071 at HiFi4 — Q·Kᵀ accumulates in 16-bit DEST (host model with per-face 16-bit
  accumulation reproduces 0.058–0.066), so it is inherent to the config, not a kernel defect. The golden
  `test_large_logits` runs fp32 DEST only and is unaffected.
- Golden test progress: slice of every bf8b / bf16-DEST cell in test_golden + test_ring_mock + test_fabric_configs:
  280 passed / 0 failed / 0 XPASS (1228 xfail_expected); `test_op_loose` (Wan2.2 FOCUS, bf16 / bf16 / False, 1x4 ring) passes.
- Issues encountered: None in the op. Finding: the bf16-DEST large-logit precision above.
- Tests added: `test_ring_joint_scaled_dot_product_attention_precision_baseline.py` now parametrizes 7 (dtype, kv_dtype,
  DEST width) configs gated on the golden TOLERANCES, plus `test_large_logits_config` (fp32 DEST: golden gate; bf16 DEST:
  0.10 regression ceiling).
