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

## Refinement 1b — Numerical configurability: bf16 DEST (perf-focus enabler) + bfloat8_b Q and K/V (debug: fix gate violations)
- Date: 2026-11-03
- What was done: fixed the harness's bullet-3 regression and a bullet-2 setup error. Neither was in the op's kernels or
  descriptor (both unchanged).
  - Bullet 3: the gate runs the golden suite with `EVAL_CAPTURE_L1=1`. This op dispatches a heterogeneous MeshWorkload
    (one program per chip). `MeshWorkloadImpl::compile` compiled those on the mesh thread pool, but the GraphTracker is
    `thread_local`, so the capture saw no CBs. `peak_total` was 0 (`L1ProfilingError`) unless the call happened to
    allocate the cached global semaphores (32 B). The passing set therefore depended on test order, and one Phase-0-passing
    cell flipped. Runtime fixes, active only while a graph capture is running:
    `tt_metal/distributed/mesh_workload.cpp` compiles on the calling thread while capturing, and each program reports its
    CBs only on the devices in its range (`ProgramImpl::set_graph_tracked_devices`, used in `program.cpp`
    `allocate_circular_buffers`). `ttnn/core/graph/graph_trace_utils.cpp` `extract_resource_usage_per_core` now keys
    per-core usage by (device, core), so chips' CBs are no longer summed together.
  - Bullet 2: every unit-test module imported the golden suite's session-scoped `_fabric_mesh_cache`/`mesh_device`. That
    registers one session cache per module, so the first module's mesh stayed open and the next module's
    `set_fabric_config` failed with `!devices_still_open`. Both fixtures are now registered once in the directory
    `conftest.py`.
- Accuracy achieved: unchanged from Refinement 1. No numerics were touched.
- Golden test progress: full suite through `eval/eval_test_runner.sh` (the gate's runner, L1 capture on): 391 passed /
  0 failed / 0 hangs (1424 xfail_expected, 5984 invalid_skipped). 0 regressions against Phase 0 and Refinement 1.
  Recorded L1 peak per chip: 113 KB to 1.27 MB.
- Issues encountered: the regression was a harness-instrumentation gap, not an op defect.
- Tests added: none. The unit directory runs as one session, 91/91 pass. `test_graph_trace_utils.py` 19/19 pass after the
  extraction change.

## Perf 1
- Date: 2026-11-03. Focus (mandatory, `LOOSE_CASES[0]`): Wan2.2 Q/K/V (1, 40, 8192, 128) per device, bf16 / bf16,
  fp32_dest_acc_en=False, HiFi2, math_approx_mode=False, 1x4 ring (FABRIC_2D_TORUS_XY), cluster_axis 1, 2 links,
  logical_n 32760. Every knob is in SUPPORTED, so the exact config was measured (no proxy). roofline 18076.5 us,
  target 24101.9 us.
- **Instrumentation (permanent):** new `ttnn/cpp/ttnn/kernel_lib/perf_instrumentation.hpp` (`MaybeDeviceZoneScope` =
  `DeviceZoneScopedN`; `MaybeDeviceZoneScopeIf(name, active)`, a sampled zone for per-key-block phases, needed because
  the focus walks 128 key blocks x 6 units per core against a ~125-zone budget per RISC). Zones: compute `compute_unit`,
  `compute_finalize`, and per sampled block (unit 0, kb 4..9) `compute_k_wait / qk / mask / rowmax / sub_exp / rowsum /
  v_wait / pv / rescale / copy_max`; reader `reader_unit / q_reserve / q_read / landing_wait`, sampled
  `reader_kv_reserve / kv_issue / kv_barrier`; writer `writer_out_wait / out_write`; port sender `port_ready_fence /
  port_send / port_wait_consumers / port_wait_arrivals`. Max 138 markers per RISC (under the cap; zones cover the
  kernel span). Probe: `tests/.../probes/perf_breakdown.py` (device run + host-side analysis; `SPANS=1` for per-call spans).

### Measured breakdown (baseline, one fresh-cache call, all 4 chips within 0.01 %)
- Op span 121.31 ms on every chip (6.7x roofline). 106 compute cores, 640 units of Sq = 16 rows: 102 cores ran 6 units
  and ended at 103.9 ms, 4 cores ran 7 units and ended at 121.3 ms. That is a **14 % work-split tail**.
- Per key block (TRISC_1, ns): sub+exp 64 000 | rescale 25 200 | rowmax 15 400 | QK 13 200 | PV 13 300 |
  rowsum 3 200 | copy 900. That sums to ~135 us; x 128 blocks = 17.3 ms per unit. The QK/PV matmuls run at
  ~35 cycles per tile-mult, ≈ the HiFi2 rate.
- Ablation (the exp SFPU payload stubbed, sync kept): sub_exp 64.0 → 5.1 us, op 121.3 → 62.5 ms. So the
  **exp is ~59 us per block = ~47 % of the op** (~620 cycles per tile).
- Data movement and the cross-device part have no headroom:
  - The reader sits in `kv cb_reserve_back` 131 us per 135 us block, so it is compute-bound.
  - TRANSPORT: `port_send` takes 3.3–4.4 ms and is done long before compute.
  - OVERLAP: landing waits are 154 ns mean and launch skew ≤ 0.3 us, so the exposed exchange is ≈ 0.
  - No cross-device idea had headroom on the focus. The local compute dominates.
- Ranked bottleneck: (1) the SFPU exp, (2) the work-split tail, (3) rescale / rowmax (FPU+SFPU chains),
  (4) the matmuls (at their rate).

### Portfolio floated (cap: 2 experiments)
1. **[tested] sfpu_exp_bf16**: the same exp_21f math as an SFPLOADMACRO pipeline (local compute).
2. **[tested] balanced_tail_split**: re-block the leftover units into small row pieces spread over all cores, with
   compute honouring the valid rows (local work split).
3. [not tested] Exp on the pack thread overlapped with the matmuls (production-SDPA style). Overlaps with 1.
4. [not tested] Fuse sub+exp+rowsum, rowsum by packer L1 accumulation. Fuse the rescale chains, rowmax in DEST on the SFPU.
5. [not tested] K/V row multicast (R2). DRAM is not on the critical path at the focus now, so this was deferred.
6. Cross-device (TRANSPORT / OVERLAP): none floated, because the measured exposed exchange is ≈ 0 on the focus.

### Per-idea verdicts
- **sfpu_exp_bf16 — WIN.** Isolated bench, single core, the op's exact chain:
  - 128-tile block 64.25 → 34.66 us (1.85x); 16-tile rescale exp 8.24 → 4.53 us.
  - Output is bit-identical to `exp_tile<false, true>` on N(0,3) scores, large logits (std 36), the [−97, 0] sweep and
    -inf lanes. Max 1 ULP, -inf → 0.
  - Menu: "trim" 58.2 us (bit-identical); "split2c" 45.2 (bit-identical); "fold2c" 42.2 (no-loadmacro fallback);
    split2 / fold2 without the clamp 36–39 us but **incorrect** (-inf → NaN), so rejected.
  - Domain: bf16 DEST. fp32 DEST is inexpressible: it uses a different, more accurate algorithm, so the functor keeps
    stock `exp_tile` there. Needs scale·x < 88.7, which always holds for softmax (≤ 0).
- **balanced_tail_split — WIN.** Op copy on the 4-chip mesh: focus 121.31 → 103.26 ms (−14.9 %).
  - Piece size r = 2 picked by the cost model. r = 1: 103.9 ms (DRAM-bound tail). r = 4: 105.3 ms.
  - About 3 ms of the win is from compiling the rescale/copy chains with a compile-time Sq for whole units.
  - Subagent domain sweep: 14/14 correct. (1,10,512,128) and (1,27,256,64) flat. fp32 (1,8,1024,128) about −10 %.
    (1,6,1024,128) + joint L128 **+7 %**.

### Graduated (both: they don't conflict — the exp changes only `ScaledExp`, the split changes the unit structure)
- `kernels/ring_joint_sdpa_scaled_exp.hpp` (new): `ScaledExp` is now the SFPLOADMACRO exp at bf16 DEST. Both chain
  sites (exp_block, rescale) use it; the old inline `ScaledExp` was deleted.
  - Carve-out (correctness/feasibility): `if constexpr (DST_ACCUM_MODE)` keeps stock `exp_tile`, because the fp32
    DEST exp is a different algorithm.
  - `DISABLE_SFPLOADMACRO` falls back to the fold2c body.
- Tail split, applied as the one path for every shape (no predicate):
  - `plan_work_split()` in the descriptor (W whole units, then ≤ 1 tail piece per core, through RT args).
  - `tail_piece()` in `ring_joint_sdpa_common.hpp`. The dead `total_units` / `units_of_core` were deleted.
  - Reader, writer and compute run units of R rows.
  - `cb_scores` keeps its nominal Sq·Sk credit, with a pad push/pop on R < Sq pieces. Without it the in-place
    phases read stale slots; this was found and fixed in the bench.
  - The bench-only env hooks (`RJSDPA_TAIL_ROWS`, `BTS_KERNEL_DIR`) were not graduated.
- **Whole op, focus:** 121.31 → **76.48 ms on every chip** (76480.7 / 76483.6 / 76480.2 / 76478.3 us), **1.59x**.
  - Per block: sub_exp 64.0 → 34.3 us, rescale 25.2 → 17.1 us.
  - Core-end quantiles [74.54, 74.56, 74.62, 76.46, 76.48] ms. Landing waits are unchanged (166 ns mean).
  - Now 4.2x the roofline (3.2x the target).

### Guard set (one call compiles, then 7 measured calls each; per-call slowest / fastest chip, median, us)
| case (1x4 ring unless noted) | base | graduated |
|---|---|---|
| (1,6,1024,128) + joint L128 replicated, bf16 DEST | 987.8 / ~845 | 1079.1 / ~917 (**+9 %, measured regression**) |
| same, fp32 DEST | 1121.9 / 1072 | 960.0 / 899 (−14 %) |
| (1,10,512,128) bf16 DEST | 705.7 / 630 | 676.4 / 609 (−4 %) |
| (1,8,1024,128) fp32 DEST, Linear | 1469.1 / 1245 | 1508.9 / 1094 (slowest +2.7 %, within its ±40 us noise; fastest −12 %) |
| (1,2,128,64) bf16 DEST, exact | 64.1 / 30.8 | 62.7 / 26.7 |
- **The joint bf16-DEST regression was not carved out.** Zones on it (base → graduated):
  - Every compute phase got faster: sub_exp 32.0 → 17.2 us, rescale 3.8 → 2.5 us per block.
  - The shape is TRANSPORT-bound: `port_send` max went 679 → 849 us and `reader_landing_wait` max 690 → 848 us, with
    the port kernels unchanged. The faster compute cores contend harder with the transport for DRAM/NoC.
  - A compute carve-out would protect the transport by keeping compute slow, and would keep a second kernel path, so
    the unified path stays. Recorded as the first Perf 2 target: OVERLAP/TRANSPORT on transport-bound small joint
    shapes (e.g. port-read priority, or ordering local / joint work around the landing).
- Golden: `scripts/run_safe_pytest.sh eval/golden_tests/ring_joint_scaled_dot_product_attention/`
  **391 passed / 0 failed** (1424 xfailed, 5984 skipped), the same counts as Refinement 1b. `test_op_loose` (the focus)
  passes. Unit dir 91/91 (precision baselines, all 7 dtype/DEST configs).

### Helper bypasses
| helper | kind | what was missing / hard | helper ns | raw ns | site |
|---|---|---|---|---|---|
| `exp_tile<false, true>` (LLK API; `_sfpu_exp_21f_bf16_tti_`), used via the chain because `compute_kernel_lib` `Exp<>` has no scale parameter | capability | The accurate bf16 exp has no SFPLOADMACRO-pipelined form: 17 issued SFPU instructions per row. A cheaper int/frac split (SFP_STOCH_RND + SFPCAST + one MAD with power-of-two-rescaled coefficients) and the scale folded into the 1/ln2 constant are not expressible through the API. `Exp<>` also lacks the `scale_en` / scale argument, which forced a custom functor even before this round. | 64 250 (128-tile block) / 8 240 (16-tile) | 34 660 / 4 530 | `kernels/ring_joint_sdpa_scaled_exp.hpp:221` (used at `ring_joint_scaled_dot_product_attention_compute.cpp:234`, `:314`) |

Summary: 2 ideas measured, 2 graduated, 0 null. The focus is 1.59x faster on every chip (121.3 → 76.5 ms). One measured
regression is recorded, not carved out: the small joint bf16-DEST shape (+9 %), which is transport-bound and taken up
in Perf 2.
