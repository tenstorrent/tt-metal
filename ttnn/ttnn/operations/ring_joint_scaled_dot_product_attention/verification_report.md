# Verification Report: ring_joint_scaled_dot_product_attention

Phase 0 = regime R1 `qstationary_dram_kv`: one `ttnn.generic_op` over a `MeshProgramDescriptor` per call. Port cores
(TRANSPORT) relay K/V head slices around the group into persistent DRAM landing buffers. Compute cores run
Q-stationary flash attention over local → joint → remote key segments, gating each remote segment on the landing
counters (OVERLAP).

## Code Review

### Fixed

1. **Correctness bug (race / stale state): the landing poll used a counter word as scratch.** The compute-core
   reader polled a port's arrival counter by `noc_async_read(port.arrival → arrival_addr)`, i.e. into its own copy
   of the global `sem_arrival` semaphore. Port cores are placed per (topology, num_links, fabric config), so a
   core that polled as a compute core in call N can be a port core in call N+1. It then starts with a stale arrival
   count, and its relay plus every consumer read landing slots before the data has landed. Found by the golden
   ring mock: every 1×4 config failed the first Ring 1-link joint cell after the Linear 2-link cells (PCC 0.71 /
   0.75 / 0.84, and PCC 0.0002 with rel-RMS 1e38 in a rerun). The failure was intermittent in isolation. **Fix:** a
   fourth global semaphore `sem_poll` (`poll_addr`) is the compute reader's private NoC-read target; it is never
   waited on or decremented by anyone. Regression test
   `test_ring_joint_scaled_dot_product_attention_placement_switch.py` replays the golden order (8 steps). It fails
   with the old poll (PCC 0.001) and passes with the fix.
2. **Precision: `test_regression::test_large_logits[0/1]` failed at the default config** (HiFi2 + fp32 DEST,
   logit std ≈ 36): rel-RMS 0.0206 / 0.0203 vs the 0.02 gate, PCC 0.9998. Triage: PCC high with rel-RMS only
   marginally over is not the scale-bug signature (the got/true ratio band is broad around 1.0, see the baseline
   below). A fidelity sweep gave HiFi2 0.0206 → HiFi3 0.0045 → HiFi4 0.0045 rel-RMS, so the cause is Q·Kᵀ operand
   truncation amplified by `exp`. **Fix (prompt rule: `compute_kernel_config` is a precision floor, the op MAY use
   more):** `FP32_DEST_MIN_FIDELITY = HiFi3`. When `fp32_dest_acc_en=True` the compute kernel runs at
   ≥ HiFi3. bf16 DEST (the DiT production / perf-focus config) keeps the requested fidelity. Cost: ≈ 1.5× the
   matmul time on fp32-DEST calls only.
3. **DRY / blocking-model fidelity: the L1 fit restated the CB inventory.** `choose_block_extents()` computed
   the footprint from a hand-written formula (`5·Sq·T_acc + 2·Sq·DVt·T_acc + …`, `T_bf = 2048` literal) while the
   descriptor allocated the CBs from a second list. A depth or CB change would drift between the two. **Fix:**
   `compute_cb_plan(Sq, Sk, …)` is the single source (index, pages, dtype): the fit sums it and the descriptor
   allocates exactly it.
4. **Program-cache key gap.** The writer's `out` / `joint_out` `TensorAccessorArgs` are compile-time and depend on
   `memory_config` (DRAM vs L1 interleaved), but the `custom_program_hash` did not include it. A call with a
   different output memory config could have hit a stale binary. **Fix:** `out_mem` added to the hash.
5. Dead code removed from the reader (`seg_rows`).
6. `l1_ledger.md` brought current. The stats / O rows described ping-pong pairs (`cb_max_a/b`, …) while the
   code has fixed-role accumulators (`*_blk` scratch + `*_acc` running). `L1_CB_BUDGET` was claimed to be
   queried from the device but is a host constant. Footprint numbers were recomputed from `compute_cb_plan`.

### Reviewed, no change

- **Helper usage**: the compute kernel is helper-based throughout (`matmul_block` for Q·Kᵀ / P·V, `reduce<MAX/SUM,
  REDUCE_ROW>`, `eltwise_chain` for mask / sub-bcast + scaled exp / rescale with dest-reuse, `binary_sfpu`
  max, `unary<Recip>`, `mul<>`). The writer uses `calculate_and_prepare_reduce_scaler`. The documented deviations
  (row sum via `reduce<SUM>` instead of an L1-acc second pack, fixed-role accumulators) are design lamps, not
  defects. No raw multicast or semaphore pattern exists that `mcast_pipe` would replace (no Tensix multicast in
  Phase 0).
- **Correctness hygiene**: `void kernel_main()`, `api/dataflow/dataflow_api.h` includes, `TensorAccessor` everywhere.
  CB push = wait counts are nominal per block by construction (one shared unit decode + segment table in reader
  and compute).
- **Blocking model**: `Sq`, `Sk` come from `choose_block_extents()` (fill the 106-core grid first, then the
  coarsest Sk that fits) and flow as CT args. Buffer depths (`Q_BLOCK_DEPTH`, `KV_BLOCK_DEPTH`, `OUT_BLOCK_DEPTH`,
  `PORT_BATCH_CHUNKS`) are single host constants that the CB sizes derive from. No CB scales with a whole-op
  dimension. The schedule is block-expressed in all three kernels: the reader issues a whole Q block / K+V block
  and barriers once, compute runs each phase over the Sq×Sk block, and the writer writes a unit and barriers once.
  No half-turned knob.
- **Work distribution**: units (b, head, Sq rows) round-robin, head-major, over every worker core minus
  `2·num_links` port cores. The machine is filled (106 compute cores at 2 links on a QuietBox 2), matching the
  design.
- **Transport**: adapted from `fabric_all_gather` (port placement by NoC1 hops to the Ethernet core, `run_pages`
  from the router payload, batch of 4 chunks, fused write + increment per item, per-call ready fence, drain /
  re-arm gated on `consumers_done`). The chunk walk is entry-relative rather than absolute. Correct, because source
  and landing bases are whole pages and a stride-B run stays in one bank on both sides. The balanced far-source
  split is not implemented (TRANSPORT lamp, not queued: see Recommendations).
- **Broadcast**: the mask is one tile added to one tile column of the boundary block. Max / α / 1/l broadcast
  along columns, as in the design's verification table.

### Prompt `## Rules` check

| Rule | Status |
|------|--------|
| MUST be one fused native implementation (one `generic_op`, own kernels, no attention / CCL op calls) | ✓ |
| `compute_kernel_config` is a precision floor; MUST NOT go below the requested fidelity / accumulation | ✓ (fidelity is only raised: HiFi3 floor on fp32 DEST; DEST width follows the request). Advisory: the exp scale is rounded to bf16 (`exp_tile` takes a 16-bit scale; dh = 128 → relative error 1.1e-4) |
| causal + balanced: output in the input's zig-zag layout | N/A (not SUPPORTED yet; restated in Refinement 2) |
| repeated calls: no host sync inside the op; persistent semaphores / scratch; RT-arg patching; deterministic | ✓ (semaphores, landing buffers and the Ethernet probe are cached per structural key; `test_deterministic` and the acceptance second-call bitwise check pass). The placement-switch race above was a violation of "correct run to run" and is fixed |
| correct under every fabric config / any G Ring; no ring-mock special case | ✓ (`test_fabric_configs.py` 12/12, `test_ring_mock.py` 48/48) |
| `from ttnn.operations.ccl import Topology` at import time | ✓ |

## Registry Conformance

- `INPUT_TAGGERS` (alignment, attention_kind, kv_heads_mode, joint_len), all `(inputs, axes)`. `SUPPORTED`
  covers every gated axis (14). `EXCLUSIONS = []`. `validate()` checks SUPPORTED per axis then EXCLUSIONS and
  raises `UnsupportedAxisValue` / `ExcludedCell`. The entry point calls `validate()` first. The op file declares no
  `INVALID`. Caller errors raise `ValueError` (num_links, Ring without wrap, `logical_n` range, partial joint,
  shape mismatches).
- No SUPPORTED auto-fixes were needed (0 xpass_drift).
- **INVALID audit** (`feature_spec.py`): 8 entries. All are op-definition impossibilities (balanced needs causal;
  joint and cross are non-causal) or the joint-presence canonicalization (`joint_len` × `joint_mode`, both
  describing the joint tensors: the documented multi-axis canonicalization exception). There are no cross-tensor
  "not built yet" entries. The canonical bf8b + ROW_MAJOR entry does not apply: `layout` TARGET is TILE only.
  Well-formed.

## L1 Ledger Audit

- **Currency**: fixed in place (see Code Review 6). Every declared CB has a row and the size expressions now
  come from `compute_cb_plan`.
- **Capacity vs live set**: every compute CB's capacity equals its live set times its stated depth. `cb_k/v_block`
  is 2× for the reader-ahead double buffer; the rest are 1×. Ragged key / q blocks keep the nominal credit and
  narrow only the transfer and compute extents (stated). No over-sized buffer. Axis accounting is consistent:
  `cb_scores` spans q_rows × keys and scales with Sq·Sk; stats span q_rows; O spans q_rows × v_head_dim. One
  improvable pair: `cb_out_blk` + `cb_out_acc` (2·Sq·DVt·T_acc) could fold into one accumulator with an
  L1-accumulating PV pack (design lamp "single O accumulator"). **Folded into Refinement 3** (frees L1 for a
  larger Sq).
- **Page format vs DEST width**: `cb_scores`, the stats and O are Float32 iff `fp32_dest_acc_en`, both directions
  consistent. `cb_out_tiles` is Float16_b as the output ABI. Mask and scalers are Float16_b, which is exact for
  0 / −inf / 1.
- **Disjoint lifetimes**: `cb_row_factor` is shared by role (α during the walk, 1/l at finalize). Every other
  pair is concurrent, with a stated reason.
- **Bounds / closed form**: Sq, Sk, DHt, DVt, depths, `mask_tiles`, `run_pages`, T_* and `L1_CB_BUDGET` all have
  bounds and the predicates that establish them. `choose_block_extents()` raises `ValueError` when nothing fits.
- **Block-size defaults**: interleaved default held: fill the grid (`Sq_occ = total_q_tiles // C`), then the coarsest
  Sk that fits. FOCUS shape: fp32 DEST (8, 8) at 1,062 KiB, 12.08 waves; bf16 DEST (16, 8) at 1,190 KiB, 6.04
  waves.
- **Per-core footprint**:
  `F = Sq·DHt·T_q + 2·Sk·(DHt+DVt)·T_kv + Sq·Sk·T_acc + 5·Sq·T_acc + 2·Sq·DVt·T_acc + Sq·DVt·2048 + 3·2048`.
  Sq scales Q, scores, stats, O and out; Sk scales K/V and scores. Port core: `2·4·run_pages·page` = 112 KiB.
- **Data-movement budget**: present and consistent with the implemented split (K/V re-read Uq times per head;
  own K/V read once per sending direction; landing written once). The cheapest-traffic split, R2 K/V row
  multicast (≈ 10.7 → ≈ 1.1 GB/call), is a `deferred` regime row with a positive reason (layers on R1 unchanged)
  and is now queued as Refinement 5.

## Precision Baseline

`test_ring_joint_scaled_dot_product_attention_precision_baseline.py`: bf16 Q/K/V, default config (HiFi2 requested →
HiFi3 by the fp32 policy, fp32 DEST), seed 1234, worst device, valid rows only. ULP is in units of the bf16 spacing
at the reference value (p99 reported: max ULP is dominated by near-zero references). The ratio is got/true over
|ref| > 5 % of max.

| Placement | Shape (per device) | Output | PCC | Max Abs Err | Mean Abs Err | Rel RMS Err | ULP p50 / p99 | got/true p5 / med / p95 |
|-----------|--------------------|--------|-----|-------------|--------------|-------------|---------------|-------------------------|
| Linear G=2 (2×2, axis 0) | (1,1,64,64) | out | 0.999998 | 0.00365 | 2.43e-4 | 0.00240 | 0.37 / 11.1 | 0.9951 / 0.9990 / 1.0032 |
| Linear G=2 | (1,4,256,128) padded | out | 0.999998 | 0.00203 | 1.33e-4 | 0.00250 | 0.40 / 13.1 | 0.9947 / 0.9988 / 1.0029 |
| Linear G=2 | (1,10,512,128) padded | out | 0.999998 | 0.00138 | 9.31e-5 | 0.00249 | 0.41 / 14.0 | 0.9948 / 0.9989 / 1.0029 |
| Linear G=2 | (1,6,512,128) + joint L 512, padded | out / joint | 0.999997 / 0.999997 | 0.00152 / 0.00103 | 7.8e-5 / 7.8e-5 | 0.00260 / 0.00260 | 0.43 / 15.6 | 0.9944 / 0.9988 / 1.0031 |
| Ring G=4 (1×4 TORUS_XY) | (1,1,64,64) | out | 0.999997 | 0.00214 | 1.87e-4 | 0.00265 | 0.41 / 13.9 | 0.9943 / 0.9989 / 1.0034 |
| Ring G=4 | (1,4,256,128) padded | out | 0.999997 | 0.00161 | 1.01e-4 | 0.00269 | 0.44 / 15.7 | 0.9942 / 0.9987 / 1.0031 |
| Ring G=4 | (1,10,512,128) padded | out | 0.999997 | 0.00100 | 7.1e-5 | 0.00266 | 0.44 / 16.4 | 0.9942 / 0.9988 / 1.0033 |
| Ring G=4 | (1,6,512,128) + joint L 512, padded | out / joint | 0.999997 / 0.999997 | 0.00086 / 0.00091 | 6.6e-5 / 6.5e-5 | 0.00277 / 0.00277 | 0.46 / 18.3 | 0.9941 / 0.9987 / 1.0037 |

**Assessment**: well inside the golden gate (0.999 / 0.02) on every shape and placement. Error grows only slightly
with sequence length and group size (more blocks in the online-softmax combine). The got/true band is broad
(±0.5 %) around a median of 0.9988. That is rounding noise, not a structural scale bug: a scale bug clusters
tightly around a non-1 constant. The small consistent negative median (−0.12 %) suggests a round-toward-zero bias
in a bf16 conversion (output pack or exp scale); it is not attributed further and is far below the gate. Large
logits (std ≈ 36, `test_large_logits`) at the default config: rel-RMS 0.0045 with the HiFi3 floor (0.0206 at HiFi2).
**Recommended tolerances** (bf16, fp32 DEST): PCC ≥ 0.9995, rel-RMS ≤ 0.01, atol 0.005.

## Verifier CLI Summary

`EVAL_CAPTURE_L1=0 eval/eval_test_runner.sh eval/golden_tests/ring_joint_scaled_dot_product_attention/ <dir>` +
`python3 -m eval.verify_supported` (report: `verifier_report.json` in this directory):

- supported_pass: 110 (test_golden 40, test_ring_mock 48, test_fabric_configs 12, test_regression 10)
- xfail_expected: 1705
- invalid_skipped: 5984
- supported_fail: 0
- xpass_drift: 0
- xfail_wrong_mode: 0

History in this verification: the first run (default L1 capture on) gave 36 pass / 74 fail. 69 failures were
`L1ProfilingError: peak_total=0` (below), 3 were the placement-switch race (fixed) and 2 were `test_large_logits`
(fixed). Second run, L1 capture off, race fixed: 108 / 2. Final: 110 / 0.

**Harness limitation (not an op defect, cannot be fixed in the op): `EVAL_CAPTURE_L1=1` fails every supported cell
of this op.** `eval/l1_profiling.py` reads CB allocations from a graph capture. `GraphTracker` processors are
`thread_local` (`tt_metal/api/tt-metalium/graph_tracking.hpp:225`). `MeshWorkloadImpl::compile`
(`tt_metal/distributed/mesh_workload.cpp:147-158`) compiles and allocates CBs on the mesh thread pool whenever a
workload has more than one program (this op has one program per device: per-device port placement and runtime args),
so `track_allocate_cb` never reaches the capturing thread. Probe result: the graph trace holds 20
`circular_buffer_deallocate_all` nodes and 0 `circular_buffer_allocate` nodes. Cells that "pass" with capture on
report only incidental 32–48 B L1 tensors. Any heterogeneous `MeshProgramDescriptor` op hits the same thing. The
op cannot collapse its per-device programs into one (runtime args are per core, not per device). Needs an
infra-side fix: compile on the capturing thread when graph capture is active, or propagate the processors to the
pool. Until then, run this op's golden suite with `EVAL_CAPTURE_L1=0`.

## Measured per-chip breakdown (perf-flagged shape, SUPPORTED proxy)

Wan2.2 shape per device (1, 40, 8192, 128), 1×4 TORUS_XY Ring, 2 links, fp32 DEST (HiFi3 by policy), Sq = Sk = 8, one
call, device profiler (`probes/focus_breakdown.py`; run without `--dev` because watcher polling distorts timing):
op span ≈ 198.4 ms on every chip; port readers (TRANSPORT: own sends + all relays) finished at 2.3–4.5 ms; compute
cores end at 183.1 ms (median) / 198.4 ms (max); the 15 ms tail is the partial last wave (1280 units / 106 cores =
12.08 waves). Local compute dominates at ≈ 7× its HiFi3 roofline (≈ 27 ms), and the transport is hidden. This
ordering is used in `op_requirements.md`: local compute first (Refinement 3), then the DRAM reuse multicast
(Refinement 5), then OVERLAP (Refinement 6).

## Recommendations

- **Queue** (`op_requirements.md`): R1 numeric bundle (fp32 DEST off: the perf-flagged contract; bf8b Q and K/V) →
  R2 causal + balanced + GQA/MQA/MLA → R3 perf local compute (perf-flagged case) → R4 sharded joint + non-aligned
  joint + cross → R5 perf K/V row multicast (scheme-change) → R6 perf OVERLAP. Every `TARGET − SUPPORTED` value is
  in R1, R2 or R4. Nothing is omitted.
- **TRANSPORT not queued, by measurement**: the port readers finish by 2.3–4.5 ms, in line with the 3.46 ms links
  roofline (`fabric_all_gather` measured 35.9 GB/s busiest link at 2 links) and inside the first ~2 % of the call.
  The design's TRANSPORT lamps (ring balance split by bank halves, `inc_every` within an entry, 1 vs 2 links) cannot
  move device time until local work is under ≈ 5 ms. R6 re-measures the exposure.
- **Landing poll back-off**: the compute reader spins on a 4-byte NoC read of the port counter with no back-off,
  contending with the port's own NoC traffic on the first wave. Advisory, inside R6's scope.
- **`--dev` and mesh switches**: under `--dev`, switching the fabric mesh mid-session stalls at teardown (Ethernet
  cores stuck; reproduced without the op by the implementer's debug test). Run one placement per invocation.
- **Precision without a lever in scope**: the exp scale is rounded to bf16 by the `exp_tile` API (16-bit scale).
  The pre-scale-Q lamp (R3) would remove it. The fp32 HiFi3 floor costs ≈ 1.5× matmul time on fp32 DEST calls. If a
  user needs fast fp32 DEST, raising fidelity only for the Q·Kᵀ matmul (per-phase fidelity) would recover most of
  it. Not queued (no failing cell, no perf case on fp32 DEST).
- **Small-shape occupancy**: golden 1×10×512×128 on a 4-ring gets Sq = 1 (160 units / 108 cores). One-tile-row
  blocks are the poor end of the chunk-granularity floor, but no perf case is there.
