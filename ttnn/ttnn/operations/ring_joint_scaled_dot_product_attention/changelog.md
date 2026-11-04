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

## Perf 2
- Date: 2026-11-04. Focus (mandatory, `LOOSE_CASES[0]`, exact config, no proxy): Wan2.2 Q/K/V (1, 40, 8192, 128) per
  device, bf16 / bf16, fp32_dest_acc_en=False, HiFi2, math_approx_mode=False, 1x4 ring, cluster_axis 1, 2 links,
  logical_n 32760. roofline 18076.5 us, target 24101.9 us. Experiment cap 2.

### Measured breakdown (start of round = Perf 1 result, one fresh-cache call, every chip)
- Op span 76.48 ms on all 4 chips (4.2x roofline). Core end quantiles [74.54, 74.56, 74.62, 76.45, 76.48] ms.
- Per key block (TRISC_1, sampled zones, ns): sub_exp 34 324 | rescale 17 055 | rowmax 15 151 | PV 13 377 |
  QK 13 310 | rowsum 3 195 | copy_max 712 | k/v waits 15. Sum 97.1 us × 128 blocks = 12.42 ms = the `compute_unit`
  median, so the zones cover the whole unit. These phases run serially on the math thread, so the zone times add up.
- The reader waits in `kv cb_reserve_back` for 92 of the 97 us of each block, and the writer is idle, so the op is
  compute-bound. Perf 1's ablation (exp payload stubbed: op −49 %) still holds for the exp share.
- Cross-device part on the focus:
  - TRANSPORT: `port_send` takes 3.5 ms (max 4.65 ms) and is hidden behind compute.
  - OVERLAP: landing waits average 165 ns, so the exposed exchange is ≈ 0.
  - No cross-device idea had measured headroom on the focus. The local compute ranks first even under the
    "cross-device first" rule.
- Roofline gate: QK + PV ≈ 26.7 us per block, at the HiFi2 rate (≈ 20 ms over the op, close to the 18 ms roofline).
  The other ≈ 70 us per block is all headroom.
- Ranked bottlenecks:
  1. sub_exp (35 %). The SFPU exp is already at its SFPLOADMACRO floor in isolation, so only running it concurrently
     with other work can hide it.
  2. rescale (18 %).
  3. rowmax (16 %).

### Portfolio (cap 2: both tested)
1. **[tested] exp_matmul_overlap** (local compute, overlap across TRISCs). Stream the score block through the phases
   one q tile-row at a time and issue the exp from the PACK thread. The exp of row i then runs while the MATH thread
   does the FPU work of other rows: QK of row i+2, the max of row i+1, and the rowsum and P·V of row i−1.
2. **[tested] pv_rescale_fusion** (local compute, fusion). Fold O_acc = a·O_acc + P·V and the l update into the P·V
   matmul's DEST: the matmul accumulates, an ELWMUL of a·O_acc goes on top, and the result packs straight to O_acc.
   This drops `cb_out_blk` and the 64-tile add pass.
3. [not tested, next round] Column-only exp for a (16 full-tile exps where only column 0 is used), and alternating
   m buffers instead of `copy_max`.
4. [not tested] Larger Sk (Sq 8 × Sk 16) to amortise the per-block rescale. With `cb_out_blk` removed, Sq 16 × Sk 16
   would need ≈ 1.57 MB of CBs against the 1.3 MB budget.
5. Cross-device (TRANSPORT / OVERLAP): none on the focus, because the exposed exchange is ≈ 0. The Perf 1
   transport-bound small joint bf16 shape (+9 %) was not re-targeted. Its guard below now measures flat against
   Perf 1, and the focus had far larger measured headroom.

### Per-idea verdicts
- **exp_matmul_overlap: WIN.** Measured in an op copy on the 4-chip mesh, focus: 76.03 → **52.36 ms per chip**;
  per key block 96.53 → 66.05 us.
  - Bit-identical to the op on all 17 cases tried (focus, padded, linear, bf8 K/V, bf8 Q+K/V, joint L128 / L512,
    ragged q, dh 64 / 96 / 256, fp32, minimal). The pack-thread exp is ScaledExp's exact instruction stream.
  - Option menu (op ms / us per block, all bit-identical unless noted):

    | option | op ms | us/block | note |
    |---|---|---|---|
    | streamed rows, exp still on MATH | 68.72 | 87.23 | the reorganisation alone |
    | + exp on PACK, one stall per tile | 55.35 | 69.94 | |
    | + one SFPU start per 8-tile section | 54.90 | 69.35 | |
    | + blocked pack | 53.97 | 68.17 | |
    | **+ each row's exp split across two FPU sections** | **52.36** | **66.05** | graduated |
    | no SFPU stall | — | — | INCORRECT (pcc 0.02) |
    | stock reduce_tile MAX with m_acc preloaded | — | — | INCORRECT (inf) |
    | `TRISC_CFG` group stall | same as grp | | correct on 3 cases, but its ordering is not guaranteed: rejected |

  - No-exp ablation floor: 60.22 us per block, so ≈ 5.8 us of exp is still exposed.
  - Domain: every bf16-DEST regime. Compute-bound shapes 0.69–0.80x per block (bf8 K/V 0.70x, dh 64 0.69x, joint
    L512 0.70x, dh 256 0.80x). Small, transfer-bound shapes are flat.
  - Exception (correctness / feasibility): fp32 DEST, for two reasons. The block max-row reduce takes bf16
    operands only. The fp32 exp is a different (stock) algorithm.
- **pv_rescale_fusion: WIN in isolation, NOT graduated (conflict).**
  - Measured in an op copy on the mesh, focus: 76.48 → 68.82 ms (−10 %). Single core: 97.87 → 88.50 us per block
    (rescale + PV + rowsum 34.4 → 24.8 us).
  - Precision: rel-RMS ×1.000 / ×1.008 (normal / large logits) on the 128-block walk, but ×1.026 on 32-block walks.
    fp32 DEST is a measured precision regression (+17–28 % rel-RMS).
  - Domain swept (bf16 DEST): bf8 K/V −8.8 %, dh 64 −12.9 %, dh 96 −7.7 %, dh 256 −12.4 %, MLA −15.5 %, small
    −2.6 %, ragged −9.2 %, tails −4.7 / −8.4 %, joint −9.1 %.
  - Why it was not graduated: it moves the P·V matmul and the `a` exp (a MATH-thread SFPU op). In the graduated
    streamed block, P·V runs per row inside the S2 section, and a MATH-thread SFPU op inside the streamed region
    races the pack-thread exp's LREG / LOADMACRO state. The two cannot both graduate as measured, and the overlap's
    win is 3x larger.
  - Next round: re-target it as an S2-section fusion, O(i−1) = a(i−1)·O_acc + P(i−1)·V per row, with `a` computed
    on the pack thread. The rescale (17.06 us of the now 66 us block, 26 %) is now the top phase outside the stream.

### Graduated: exp_matmul_overlap (bf16 DEST, every shape; fp32 DEST carve-out)
- `kernels/ring_joint_scaled_dot_product_attention_compute.cpp`: `if constexpr (!DST_ACCUM_MODE) stream_block(...)`.
  - It replaces the bf16-DEST QK, mask, rowmax + BinaryMax, sub_exp, rowsum and P·V phases. The old helper sequence
    is deleted for bf16 DEST.
  - The fp32-DEST body keeps the helper phases. That is the carve-out above, earned by correctness / feasibility,
    not by a measurement.
  - The rescale, `copy_max` and finalize are unchanged, and shared by both paths.
  - The bench's variant switches (`EMO_*`) were not graduated. The winning option is the only path; the rejected
    options are recorded above.
- `kernels/ring_joint_sdpa_scaled_exp.hpp`: the exp namespace now compiles for the pack thread too. Added the
  grouped `tiles<cond>(n)` (one SFPU start and one RWC reset per DEST section) and `PackThreadScaledExp`.
- Descriptor: new credit-only `CB_P_TOKEN = 14` (4 × 32 B), compute compile-time arg 27. `compute_cb_plan` and the
  L1 fit are unchanged. `l1_ledger.md` gained a row.
- Instrumentation: `compute_stream` (one sampled zone per streamed block; its steps interleave across the TRISCs,
  so per-phase zones no longer measure separable work). The fp32 body keeps `compute_k_wait / qk / mask / rowmax /
  sub_exp / rowsum / v_wait / pv`. `compute_rescale / copy_max / unit / finalize` are unchanged. Max 122 markers per
  RISC.
- **Whole op, focus: 76.48 → 52.69 ms on every chip** (52695.1 / 52692.6 / 52694.5 / 52693.3 us), **1.45x**.
  - Per block: compute_stream 48.6 us + rescale 17.06 us + copy ≈ 0.7 us ≈ 66.4 us.
  - Core-end quantiles [51.04, 51.06, 51.12, 52.67, 52.70] ms. Landing waits 175–204 ns mean. `port_send` still
    hidden (max 5.9 ms).
  - Now 2.9x roofline. The 24.1 ms target is 2.2x away.
  - Since Phase 0: 121.3 → 52.7 ms (2.30x).

### Guard set (same session, A/B; 1 compile call + 8 measured calls each; per-call slowest / fastest chip, median, us)
| case (1x4 ring unless noted) | base (Perf 1 op) | graduated |
|---|---|---|
| (1,6,1024,128) + joint L128 replicated, bf16 DEST | 1018.0 / 788 (bimodal 744–947) | 1016.7 / 847 (tight 737–860); means 834 vs 835: flat |
| same, fp32 DEST (unchanged code) | 954.7 / 900 | 972.9 / 896 (within the ±40 us noise) |
| (1,10,512,128) bf16 DEST | 686.1 / 610 | 682.4 / 612 (flat) |
| (1,8,1024,128) fp32 DEST, Linear (unchanged code) | 1542.3 / 1098 | 1518.0 / 1096 (flat) |
| (1,2,128,64) bf16 DEST, exact | 63.7 / 26.8 | 63.7 / 24.8 (fastest −7 %) |
- No material regression. One base run of the fp32 joint case failed once and passed on rerun (base code, an infra
  transient). The graduated runs all passed.
- Golden: `scripts/run_safe_pytest.sh --run-all eval/golden_tests/ring_joint_scaled_dot_product_attention/`
  **391 passed / 0 failed** (1424 xfailed, 5984 skipped), the same counts as Perf 1. The unit dir is 91/91.

### Helper bypasses
| helper | kind | what was missing / hard | helper ns | raw ns | site |
|---|---|---|---|---|---|
| `eltwise_chain(BinaryFpu Sub bcast-col → ScaledExp → PackTile)` | capability | No chain element runs an SFPU op on the PACK thread (between `tile_regs_wait` and the pack of the same DEST half) while MATH fills the other half: every chain op is issued from TRISC1. The exp also needs one `STALLWAIT(SFPU, MATH)` + RWC reset per DEST section, which has no API. | 96 530 per key block (whole helper sequence, bench) | 66 050 | `ring_joint_scaled_dot_product_attention_compute.cpp:213` (`exp_chunk`), `ring_joint_sdpa_scaled_exp.hpp:290` |
| `matmul_block` (Q·K^T and P·V) | capability | The helper owns a whole phase over a whole block with one output CB. The overlap needs per-row sub-blocks interleaved with other ops in one DEST section, out-of-order / blocked packs into a held-`wr_ptr` score block, and P·V co-resident in DEST with the max and rowsum results, packed to three different CBs. | (same row) | (same row) | `…_compute.cpp:185` (`qk_chunk`), `:236` (`s2`) |
| `reduce<MAX, REDUCE_ROW>` + `binary_sfpu<BinaryMax>` | capability | The running max has to fold into the row reduce. BinaryMax is a MATH-thread SFPU op and would race the pack-thread exp's LREG / LOADMACRO state, so m_acc is preloaded (within-face transposed) into the GMPOOL accumulator. The helper has neither a DEST-seeded accumulator nor the one-MOP-per-row `reduce_block_max_row`. | (same row) | (same row) | `…_compute.cpp:236–250` |
| `reduce<SUM, REDUCE_ROW>` | capability | Has to share a DEST section with the max and P·V of other rows (it acquires its own DEST per row and has a single output CB). | (same row) | (same row) | `…_compute.cpp:236` (`s2`) |

The helper ns / raw ns pair is one measured sequence: all four bypasses replace a single helper sequence together.
The bypassed helpers are still used by the fp32-DEST body, the rescale and the finalize.

Summary: 2 ideas measured, 1 graduated (exp_matmul_overlap, bf16 DEST everywhere). The other (pv_rescale_fusion) is a
WIN in isolation, not graduated because it conflicts with the streamed P·V; it is recorded as the next target. 0 null.
The focus is **1.45x faster on every chip (76.48 → 52.69 ms)**, with no guard regression and the golden suite green.

## Refinement 2 — Causal prefill family: causal + zig-zag balanced, GQA / MQA / MLA heads
- Date: 2026-11-04
- What was done:
  - SUPPORTED: `mask_mode` += causal, `is_balanced` += True, `kv_heads_mode` += gqa, mqa, mla. No EXCLUSIONS.
  - Causal (R4), kept in one place so the reader and compute walk the same blocks: `ring_joint_sdpa_common.hpp` (`global_tile`, `causal_visible_tiles`, `causal_mask_class`). Positions grow along a device in both the contiguous and the zig-zag layout. So each key segment is cut at the unit's latest query: those keys are never read or computed. A source entirely in the unit's future is skipped, landing wait included (OVERLAP: fewer waits). Inside the kept blocks each (q tile, key tile) pair is full, diagonal (mask tile 0, a triangle) or dead (mask tile 1, all −inf). The bf16-DEST streamed path adds these with `add_reuse_dest_tiles` on the score DEST, the same mechanism as the key-tail mask. The fp32-DEST helper path adds them with one `eltwise_chain` per run (B input `TileAddressing::Offset` picks the mask tile). The writer builds both mask tiles: NoC zero, 120 + 16 CPU lanes, the rest NoC copies. Causal drops the key-tail padding mask (a key ≥ logical_n is visible only to padded queries). The local segment comes first and its tile 0 is visible to every query, so the online-softmax finite-max cap holds.
  - Head sharing (R3, knob-turn of `q_heads_per_unit`): `Q_HEADS_PER_UNIT` (None = hq / max(hk, hv)) sets g. The g heads' tile rows are contiguous in Q and out, so a unit is a block of the stacked row space (`NQ = g·n_q_t`, `BG = b·hq/g`). GQA/MQA units of hq/hk heads share one key-block stream (K/V DRAM reads ÷ hq/hk). `choose_block_extents` / `plan_work_split` see the stacked space, so Sq can grow past one head's rows. MLA: g = 1, and the K head index (h / (hq/hk)) is separate from the V head index (h / (hq/hv)).
  - Transport (TRANSPORT): an entry is (b, j < max(hk, hv)). It carries K head j iff j < hk and V head j iff j < hv, so MLA's one K head rides with V head 0. The port reader and sender take per-part chunk counts (k_chunks, v_chunks) and fire the arrival increment on each item's last chunk. A host check rejects a link split that would leave an item with no chunk. The compute reader waits for entry b·Hmax + max(kh, vh).
  - Reused: the whole R1 pipeline, the CBs, the streamed compute block, the key-mask CB and the add-reuse-dest mask mechanism, unit decode and tail split (fed stacked counts), and the transport schedule. Added: the causal plan functions, one more mask tile, per-tile mask selection, and the K/V-part split of transport items.
- Accuracy achieved: every targeted cell passes the golden TOLERANCES gates (e.g. bf16/bf16 fp32 DEST PCC ≥ 0.999, rel-RMS ≤ 0.02; bf16/bf8b bf16 DEST PCC ≥ 0.995, rel-RMS ≤ 0.06). Shapes: Qwen3-VL GQA 8:1, MiniMax M3 MQA 16:1, GPT-OSS 8:1, GQA 2:1, DeepSeek MLA (dh 576 / dv 128), Mistral MLA (320 / 128), plus MHA causal.
- Golden test progress, slices only (the harness runs the full suite):
  - Golden, GQA/MQA/MLA × {none, causal, causal+balanced} × fp32/bf16 DEST, axis 1, padded: 36/36.
  - Golden, 1-in-7 sample of every other no-joint GQA/MQA/MLA cell and MHA causal cell: 145 passed, 46 skipped (INVALID), 0 failed.
  - `test_ring_mock.py` causal-balanced MQA bf8b (G=4, every fabric config, Ring/Linear, 1/2 links): 24/24.
  - `test_regression.py`: 10/10.
  - Unit dir: acceptance 15/15, placement_switch + debug 13/13, new causal_heads 12/12.
- Issues encountered: a local `g` shadowed the DRAM-grid variable in the descriptor (renamed to `qh`). Finding, not filed: zig-zag balancing cut the busiest chip from 1.23 ms to 1.04 ms on GQA 8:1, n=2048/chip, 4-chip ring, bf16 DEST (per chip, contiguous 1.23/0.86/0.55/0.35 ms; balanced 0.84/0.89/1.04/0.97 ms). It falls short of the ~0.75 ms mean because at about one unit per core the round-robin split assumes equal-cost units, and the costliest unit (the latest rows) sets the wall. A causal-cost-aware unit assignment (pair early and late units per core) is the next lever for the perf rounds.
- Tests added: `tests/ttnn/unit_tests/operations/ring_joint_scaled_dot_product_attention/test_ring_joint_scaled_dot_product_attention_causal_heads.py` (4-chip ring, GQA 8:1 n=2048, GQA 2:1, MLA dh 320; contiguous / balanced; fp32 / bf16 DEST; bf8b K/V).

## Refinement 3 — Speed up the perf-flagged Wan2.2 case: local compute (attention) — the measured dominant part
- Date: 2026-11-04
- Focus, exact config (`LOOSE_CASES[0]`): Wan2.2 (1, 40, 8192, 128) per device, bf16 / bf16, fp32_dest_acc_en=False,
  HiFi2, 1x4 ring, 2 links, logical_n 32760. Probe: `probes/perf_breakdown.py` (device-profiler KERNEL spans + zones),
  one fresh-cache call per variant.
- **Result: 52.81 → 42.64 ms fastest chip, 52.81 → 42.80 ms slowest chip (1.24x)**. Per key block 66.6 → 53.2 us
  (stream 47.8 + rescale 18.1 + copy 0.7 → stream 53.2). Now 2.36x the 18.08 ms roofline; the 24.1 ms target is 1.77x
  away.
- What was done (bf16-DEST streamed block only; the fp32-DEST helper body is unchanged):
  1. **Single l / O accumulators (pv_rescale_fusion, re-targeted as the S2-section fusion Perf 2 named)**:
     52.81 → 43.53 ms.
     - a_i = exp(s(m_acc_i − m_blk_i)) is computed in a spare DEST slot of row i's first exp chunk, by the same
       pack-thread `PackThreadScaledExp` stream. It goes to its own one-tile section when the chunk fills DEST. No
       MATH-thread SFPU op is added, so it does not race the pack-thread exp.
     - In S2 the row sum and the P·V matmul run into fresh DEST slots. ELWMUL (`mul_tiles_bcast_cols`, which
       accumulates into DEST) then adds a·l_acc and a·O_acc on top, and the result packs back over
       cb_sum_acc / cb_out_acc. The CB rotates per row / per pv_w chunk (pop the front, push back), in place for a
       full accumulator and into the free window for a tail piece.
     - This removes cb_sum_blk, cb_out_blk and the 96-tile rescale pass.
  2. **Running max ping-pongs between cb_max_acc / cb_max_blk** (runtime CB ids swapped per block) instead of the
     per-block m copy: 43.53 → 42.73 ms.
  3. **Ledger fold-in**: `compute_cb_plan` drops cb_sum_blk / cb_out_blk when T_acc is 16-bit, freeing 160 KB per core
     at the focus (1,190 → 1,030 KiB).
  - Reused: the streamed block, its CBs, the pack-thread exp, cb_row_factor (α role), the finalize. Added: the a slot
    in exp_chunk, the ELWMUL corrections in S2, the CB rotation, the m swap.
- Levers weighed (measured on the focus, block = compute_stream zone; FOCUS_HEADS=8 gives the same per-block time):
  - Single-payload ablations of the new block: none moves it much. exp stubbed 53.3 → 48.9 us, rowmax → 53.0,
    rowsum → 49.9, ELWMUL corrections → 50.3, sub (S − m) → 52.2, Q·Kᵀ matmul → 51.5, P·V matmul → 47.3.
  - Combined ablations: both matmuls stubbed 46.0 us; everything except the matmuls stubbed 39.8 us; every payload
    stubbed 16.3 us median (then K/V-wait bound: mean 22.8, max 110). The stages are **balanced**: the MATH, PACK and
    UNPACK threads all carry comparable work, so no single stage holds the wall.
  - Fidelity: LoFi cuts the matmul-only block 39.8 → 29.1 us (≈ 21 us of HiFi2 FPU math, at the 1024 tile-matmul
    estimate) but the full block only 53.3 → 48.4 us.
  - compute_block_size (init churn): doubling ~80 inits per block costs +1.3 us (≈ 16 ns each), so the per-section
    inits are not the bound. No change.
  - Block co-tune Sq × Sk: even with the freed L1, Sq 16 × Sk 16 needs 1.58 MB and Sq 32 × Sk 8 1.85 MB, against the
    1.33 MB budget, so the focus stays at (16, 8). The fold-in still lowers other shapes' footprints.
  - Partial last wave: the tail-piece model still picks r = 2 against the new per-block cost (~6 us fixed + 3.3 us per
    row). r = 1 is DRAM-bound (64 pieces) and r = 4 slower. The tail is 1.7 ms (4 %).
  - matmul_output_subblock (height 2): not implemented. It reorganises the whole row pipeline (QK / max / exp / PV
    per row pair; DEST cannot hold 2 rows of P·V plus the stats). With balanced stages, the unpack it saves
    (1216 → 768 tile unpacks per block) is not shown to be the binding resource. This is the next candidate for the
    perf rounds.
- Accuracy achieved: unchanged within noise. bf16-DEST large logits rel-RMS 0.0667 / 0.0847 (out / joint), as at
  Refinement 1. Precision baseline: all bf16-DEST cases PCC ≥ 0.99984, rel-RMS 0.012–0.018. `test_op_loose` (Wan2.2
  focus, 0.995 / 0.05 gate) passes.
- Golden test progress (slices; the harness runs the full suite):
  - 1-in-8 sample of the bf16-DEST self-attention TILE cells (incl. MLA, GQA, causal, joint), plus `test_regression`
    and `test_op_loose`: 127 passed, 132 skipped (INVALID), 0 failed.
  - Unit dir: 103/103.
- Guard set (same session A/B against 5940d0d, 1 compile + 8 measured calls, median slowest / fastest chip, us):

  | case | base | new |
  |---|---|---|
  | joint L128 bf16 DEST (1,6,1024,128) | 1052.9 / 862 | 1030.3 / 862 |
  | joint L128 fp32 DEST | 958.3 / 899 | 959.5 / 895 |
  | (1,10,512,128) bf16 DEST | 686.1 / 604 | 676.8 / 601 |
  | (1,8,1024,128) fp32 Linear | 1492.6 / 1093 | 1531.2 / 1098 |
  | (1,2,128,64) bf16 exact | 66.8 / 24.8 | 62.5 / 24.2 |

  - The fp32 code path is behaviourally unchanged; its Linear slowest-chip span ranges 1163–1547 us within each run
    (noise).
- Issues encountered:
  - The first build hung: cb_max_acc had been popped by the removed rescale chain, so the stream must pop it itself.
  - `--dev` overflows the 70 656 B kernel config buffer on this kernel (watcher instrumentation); triage ran in
    default mode.
- Tests added: none (existing precision-baseline / causal-heads / acceptance files cover the bf16-DEST path).
  `probes/perf_breakdown.py` gained `FOCUS_FID` (fidelity ablation).

## Refinement 4 — DiT joint / cross variants: sharded joint, non-aligned replicated joint, cross-attention
- Date: 2026-11-04
- What was done:
  - SUPPORTED: `joint_mode += sharded`, `joint_len += non_aligned`, `attention_kind += cross`. No EXCLUSIONS.
  - Sharded joint (R5) — TRANSPORT: the joint K/V slices are a second transport entry *kind* on the existing relay
    (port reader / sender loop over `ee = entry * NK + kind`; kind 1 reads the joint inputs / relays from and lands in
    separate persistent joint landing buffers `_landing_buffers(..., tag="joint")`). The arrival count stays an entry
    index (`ee * S_in + rank + 1`); the host checks a port carries a joint item iff it carries a main one.
    OVERLAP: remote joint segments (kind 3) follow each remote main segment of the same source in the key plan and
    are gated like it (joint slice lands right after the main slice of the same (batch, kv head)). Joint Q stays local;
    `joint_out` is the device's own slice. Wholly padded slices are skipped (no read, no wait).
  - Non-aligned joint (replicated L % 32 or sharded logical_l % 32): second key-tail mask tile (`JOINT_MASK_TILES = 2`
    → `compute_cb_plan` mask_tiles ≤ 2); each segment carries its mask-tile index (+1), the writer builds tile 1 with
    the joint valid columns; both DEST bodies apply it (bf16 stream: dest-reuse add; fp32: the mask chain now indexes
    the tile with `TileAddressing::Offset`). Masked in-kernel; inputs arrive tile-padded, no host pad.
  - Cross-attention: the unit decode / key plan were already split on `n_q_t` vs `n_kv_t`; widened the gate only.
  - Reused: unit decode, key_block_plan segments, entry/slot/arrival protocol, key-tail mask path, CB plan.
    Added: entry kind loop in both port kernels, kind-3 segments in the reader, joint mask tile.
- Accuracy achieved: golden tolerances (bf16/bf16 fp32 DEST: PCC ≥ 0.999, rel-RMS ≤ 0.02; bf16 DEST 0.995 / 0.05;
  bf8b combos per TOLERANCES) on Flux / SD3.5 (L 333) / Motif (L 334) / Mochi (L 118) joint shapes and the LTX-2 cross
  shapes, G = 2 and 4.
- Golden test progress: test_golden slice (`sharded or non_aligned or cross`) 320 passed / 0 failed;
  test_ring_mock + test_fabric_configs + test_regression 118 passed.
- Issues encountered: None.
- Tests added: `tests/ttnn/unit_tests/operations/ring_joint_scaled_dot_product_attention/test_ring_joint_scaled_dot_product_attention_joint_cross.py`
  (sharded L 333 / 512 / 118, replicated L 333 / 118, cross q32 / q64; fp32 and bf16 DEST; 1×4 ring).

## Refinement 5 — Perf-flagged case: K/V reuse multicast across a grid row (design regime R2) — scheme-change
- Date: 2026-11-04
- What was done (OVERLAP is unchanged; this is a local-dataflow topology change, design regime R2 `qstationary_kv_row_mcast`):
  - Host: `plan_kv_mcast_groups` cuts each logical grid row of the compute grid left after the ports into
    x-contiguous row groups of size g | Uq (largest first, capped by the new knob `KV_MCAST_GROUP`, None = whole row
    run, 1 = off ⇒ R1). FOCUS: the ports take (0..2, 0) and (0, 1), giving 10 groups of 8, 9 groups of 2 and 8 single
    cores.
  - Host: `plan_work_split` now returns explicit per-core unit lists. Per pass it hands each group a g-aligned run of
    g consecutive units (one head's) and each single core one unit, greedily in head-major order, so passes stay in
    lockstep with the head-major transport. Units that no piece can take go to the tail; if passes cannot be filled
    the groups are dropped. Without groups the lists equal the old `c + i·C`.
  - Wiring: one `McastFamily` per chip (fixed sender = the group's first core, handshake), attached to the
    compute-core reader. With no group `attach_absent` compiles the path out. The reader's NoC is now explicit
    (`READER_NOC` = NoC0, what `ReaderConfigDescriptor` resolved to) so the family and the kernel agree.
  - Reader: on whole units the injector alone awaits landings and reads each K/V block, then `send`s the K block and
    the V block into the same CB slots of every member. Members reserve, `receive` twice and push: no DRAM read and
    no landing poll. Tail pieces and single cores read their own K/V (R1). Compute and writer only switched to the
    unit lists.
  - Profiler: compute zone `compute_rowmax` kept at a line whose 16-bit source-location hash does not collide with
    the ERISC firmware zones (Refinement 4's line shift had broken `TT_METAL_DEVICE_PROFILER` runs with "Source
    location hashes are colliding"). `probes/perf_breakdown.py` gained `FOCUS_KV_MCAST` and `FOCUS_HK`.
  - Reused: the CB inventory (multicast lands in `cb_k_block` / `cb_v_block`), unit decode, key plan, tail split,
    transport. Added: the row-group planner, the unit lists, the reader's two pipe roles, the mcast family.
- Measured (device profiler, A/B in one session, R1 = `KV_MCAST_GROUP = 1`):
  - Focus (3 calls): slowest chip 42 770 → 42 765 us, fastest 42 611 → 42 585 us. Flat: the focus is compute-bound
    (reader in `kv cb_reserve_back` 46 us of every ~53 us block before R2).
  - What R2 changed at the focus: compute K/V DRAM reads ≈ 10.7 → 3.25 GB; per-block K/V barrier 4.9 → 0.85 us;
    member wait for a multicast block 5.8 us inside the 47 us reserve; `port_send` 5.5–8.5 → 3.0–4.1 ms (transport
    back at the link roofline once compute cores stop contending for DRAM); worst landing wait 330–460 → 12–15 us.
  - Guard set (slowest-chip median of 8 calls, us, R1 → R2):
    - joint L128 bf16 DEST (1,6,1024,128): 1063.9 → 562.2 (1.89×)
    - joint L128 fp32 DEST: 963.7 → 832.6 (1.16×)
    - (1,10,512,128) bf16 DEST: 662.2 → 489.5 (1.35×)
    - (1,8,1024,128) fp32 DEST, Linear: 1531.8 → 1003.3 (1.53×)
    - GQA (1,16/4,2048,64) bf16 DEST: 1506.9 → 1469.4 (−2.5 %)
    - (1,2,128,64): 61.5 → 61.7 (no group forms, so R1 runs)
    - Causal runs R1 by construction.
  - Remaining headroom at the focus is the per-block compute (balanced MATH / PACK / UNPACK, Refinement 3). The next
    candidate is the `matmul_output_subblock` row-pair pipeline. It was not tried here because it is a compute rewrite.
- Accuracy achieved: R2 output is bit-identical to R1 on every pinned case (mha bf16 / fp32-Linear, GQA, replicated
  joint L128, sharded joint L333). PCC vs torch ≥ 0.999 (fp32 DEST) / 0.995 (bf16 DEST). `test_op_loose` (focus) passes.
- Golden test progress (slices): test_regression + test_ring_mock + test_fabric_configs 118 passed / 0 failed; a
  1-in-10 sample of the TILE `test_op` cells: 167 passed, 601 skipped, 0 failed. Unit dir 122/122.
- Issues encountered: the mcast family's NoC must equal the reader kernel's (ReaderConfigDescriptor resolves to
  NoC0 on Blackhole), which is now explicit. The profiler zone-hash collision is described above.
- Tests added: `tests/ttnn/unit_tests/operations/ring_joint_scaled_dot_product_attention/test_ring_joint_scaled_dot_product_attention_kv_mcast.py`
  (R2 vs R1 bit-identical + torch PCC, asserts the row groups were used; 5 cases, ring and line).
