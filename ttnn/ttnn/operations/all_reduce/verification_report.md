# Verification Report: all_reduce

**Op kind:** multi-device compute-CCL — line store-and-forward gather of whole
shards fused, in ONE `ttnn.generic_op` dispatch, with an arrival-ordered
incremental N-way SUM on a dedicated reduce core. Every device ends up holding
the identical element-wise sum (output spec == one input shard). Built as a
self-contained Python `ttnn.generic_op` + `ttnn.MeshProgramDescriptor` op with
five newly-authored kernels (4 relay descriptors + 3 reduce descriptors per
device program).

**Verification date:** 2026-08-25
**Verification platform:** REAL 4-chip Blackhole hardware — topology
`bh_quietbox_1x4_hw` (mesh `(1, 4)`, `FABRIC_1D`), via
`scripts/run_multidevice_sim_pytest.py --op all_reduce` (runtime `hardware`).

---

## TL;DR

- **On-device verification PASSED on real hardware.** 27/27 across acceptance
  (14), extended (5), precision (8); golden suite 11 passed + 1 lenient xfail
  (the beyond-TARGET Ring cell). Verifier CLI: **all loud categories 0**;
  `supported_pass = 6` — the ENTIRE golden cartesian.
- **`TARGET − SUPPORTED = ∅` on every axis.** Phase 0 covers the feature spec's
  whole ambition (dtype {bf16, f32} × TILE × Linear), so `xfail_expected = 0`
  and the refinement queue is **empty by gap accounting** (see
  `op_requirements.md` for the audit trail and the beyond-TARGET candidates).
- **Code review: no correctness bugs found; zero code changes needed.** The
  implementation is a faithful, hardware-green derivative of the
  `reduce_scatter_average` single-dispatch shape; every helper is used per its
  header contract; CB sync balances in every regime; design conformance holds
  on all binding dimensions (single dispatch, arrival-major overlap, 3-core
  roles, two-semaphore double-inc contract).
- **Precision is excellent and shape-stable.** Worst-device bf16 PCC ≥
  0.9999955 (rel-RMS ≈ 0.0035 — the bf16 mantissa budget for an N=4 sum, max
  err ≤ 1 ULP at output scale); f32 PCC ≥ 0.9999994 (rel-RMS ≈ 6.3e-4 — FPU
  srcA/srcB ~10-bit mantissa truncation, a hardware datapath property, in line
  with the reduce_scatter reference's 4.4e-4).
- **Verification-environment footgun found and neutralized** (test-side, not
  op-side): the login shell's `python3` resolves `ttnn` to a DIFFERENT clone
  (`/localdev/wransom/tt-metal-eval`) that ships an older two-dispatch
  all_reduce. An early verification pass silently exercised that op instead of
  this one. All graded runs were redone with this repo's
  `python_env/bin/python3`; the discrepancy was caught because the stale op
  lacks this op's accumulator-budget ValueError (extended test failed) and was
  confirmed via a probe (`probes/probe_budget_gate.py`). **Always drive this
  op's suites as `python_env/bin/python3 scripts/run_multidevice_sim_pytest.py …`.**

---

## Code Review

### Fixed

Nothing required fixing in the op or kernels. (The only artifacts this pass
added are verifier-authored tests and the probe above.)

### Reviewed clean (no change needed)

- **Design conformance (binding dimensions).**
  - *Algorithm*: ONE `generic_op` per invocation (mesh PD with one program per
    coordinate); gather + reduce fused; arrival-major incremental accumulate
    (C1 seed → C2 rearm → C3 per-arrival run → C4 drain), NOT the forbidden
    two-dispatch gather-then-reduce and NOT the position-major `sum_blocks(N,…)`
    the design rejects (R15). ✔
  - *Pipeline topology*: (0,0) fwd relay, (0,1) bwd relay, (0,2) reduce; 7
    kernel descriptors per program from 5 sources; reader NCRISC / writer
    BRISC. ✔
  - *Work distribution*: fixed 3-core roles; single reduce core (P ≤ 8 on
    acceptance/golden shards) — matches the design's deliberate deferral of
    multi-core reduce. ✔
  - *Inter-core/device communication*: two `GlobalSemaphore`s (fwd/bwd), TWO
    fabric atomic-incs per block on the SAME connection as the pages (in-order
    ⇒ `sem ≥ k` implies data-complete, R8); reduce reader two-way-polls both
    counters (`invalidate_l1_cache()` + volatile reads) so whichever direction
    lands first is consumed first — the overlap mechanism (T4/T7). ✔
- **Helper usage.** Relay writer drives the fabric egress exclusively through
  the safety-by-construction helper (`FabricStreamSender<>` ctor advancing
  `conn_arg_idx` by reference → `open(unicast_route(1))` →
  `arm_unicast_write(page_size)` / `arm_inc(1)` → `write_page` / `inc` →
  `close()`). Compute is **all-helper**: `binary_op_init_common` (the
  documented pre-condition) + `BlockAccumulate::arm/rearm/run` +
  `sum_blocks(num_blocks=1)` as the documented seed/drain copy with
  `pop_input=true` (R4). The raw pieces (relay wait/flush/pop around
  `write_page`, `noc_semaphore_wait_min`/`set`, the reduce reader's two-way
  poll) are exactly the halves the helper banner assigns to the op — there is
  no `FabricStreamReceiver`, and a single-counter wait would serialize the two
  directions. Every rejected-helper decision in the design's mandatory table
  checks out against the in-clone headers (e.g. `run_seeded` would deadlock on
  the empty accumulator; `run_chunked` reserves before popping and would
  deadlock the in-place accumulator at capacity P; a second armed
  `BlockAccumulate` is forbidden by the header's singular-hw-state note). ✔
- **CB sync (push == wait/pop) — balanced in every regime.**
  `cb_relay_pages`: reader pushes `num_sends·P` (seed P + (num_sends−1)·P
  read-backs), writer pops `num_sends·P`; 0 = 0 on idle line-end directions
  (reader seed and writer body both gated on `num_sends > 0`).
  `cb_contributions`: reader pushes N·P in g-granules; compute pops P (C1) +
  (N−1)·P (C3). `cb_accumulator` (compute-only, single producer per R2):
  N·P pushed = (N−1)·P + P popped. `cb_summed`: P = P. All waits on a given CB
  use a single count (g on reduce CBs, 1 page on the relay CB); every CB
  capacity is a whole multiple of its quantum, so multi-page reserves never
  straddle the ring wrap. ✔
- **Granularity/DEST contract.** Host clamps `g ∈ {4,2,1}`, `g | P`, so no tail
  chunk exists; `g ≤ DEST_AUTO_LIMIT` is `static_assert`ed in the compute
  kernel AND asserted at `BlockAccumulate::arm`; compute config HiFi4 +
  `fp32_dest_acc_en=True` fixes DEST_AUTO_LIMIT = 4. ✔
- **Cache-reuse re-arm (R1).** Every semaphore consumer resets its OWN core's
  counter after its final wait — relay readers each reset their direction's
  counter (including pure line-end receivers, which wait ALL arrivals first),
  the reduce reader resets BOTH after consuming all N−1 arrivals. Verified by
  the two-call program-cache acceptance test AND the translated
  `test_nd_program_cache_hit`, both green on hardware. ✔
- **Fabric arg contract (R9/R10).** Relay writers are constructed with EMPTY rt
  args; `build_ccl_fabric_rt_args` (which mutates the program) appends the
  connection block FIRST post-construction via the live `runtime_args[x][y]`
  view; idle directions get `[]` and the whole kernel body is inside
  `if constexpr (num_sends > 0)` — no unconditional `get_arg_val` before the
  guard. `route.num_hops == 1` asserted (store-and-forward invariant);
  `is_forward` peeked from the block's leading flag, never hand-derived. ✔
- **`noc_async_writes_flushed()` before every `cb_pop_front`** in the relay
  writer (R7) — the fabric write sources the CB slot. ✔
- **API correctness.** `void kernel_main()` in all five kernels (incl. compute
  — matches the hardware-validated reference convention); includes are
  `api/dataflow/dataflow_api.h` / `api/compute/eltwise_binary.h`; addressing is
  `TensorAccessor` + `TensorAccessorArgs` (no deprecated `InterleavedAddrGen`);
  no broadcast op anywhere (correct — the reference's SCALAR-bcast 1/N pass is
  deleted; SUM needs no scaling operand, hence no scaler CB either). ✔
- **Host assembly.** Per-device DISTINCT programs (CT: `my_chip_id`,
  send/arrival counts) — cache-stable; gather_buffer allocated FRESH per call
  and passed mid-`io_tensors` (output LAST, R14); semaphores module-cached per
  `id(mesh_device)` with ONE miss-branch `synchronize_device`, parked on
  `mesh_pd.semaphores` (the `hasattr` guard matches both hardware-validated
  references verbatim). Block-flow table checks out against T1/T2 for both
  line ends and interiors (invariant `fwd_arrivals + bwd_arrivals = N−1`). ✔

### Benign, well-justified deviations from `op_design.md` (no action)

- **`_MIN_RANK = 2` instead of the design's rank-4 pin.** The immutable golden
  translated suite carries rank-2/3 cases (`test_nd`); the kernels see only P
  dense pages and the gather_buffer stacks shards on dim 0 (block c stays at
  pages `[c·P, (c+1)·P)` for any rank ≥ 2 since dim 0 is outermost). Widening
  costs no kernel change; goldens are ground truth. Documented in the op file;
  rank-2/3/4 all verified green on hardware. ✔

### Minor observations (not fixed — churn risk on hardware-green kernels; recorded here)

- **Per-page `noc_async_read_barrier` in the relay reader** serializes its
  ingress (one page per barrier); batching reads per CB slot pair would overlap
  more. Pure perf, no failing cell — mirrors the all_gather/point_to_point
  precedent of not churning verbatim-from-reference relay loops.
- **Per-page `noc_async_writes_flushed()` in the relay writer** is conservative
  but provably safe; per-block flushing would overlap more fabric egress. Perf
  only.
- **Semaphore cache keyed by `id(mesh_device)`**: if a mesh device is closed
  and a new one allocates at the same CPython id, the cache would hand out
  dead semaphores. Same pattern as both hardware-validated references; the
  eval harness opens one mesh per session. Known-benign; noted for a future
  hardening pass.

---

## Registry Conformance

- **INPUT_TAGGERS** — present, `{}` (correct: no shape-derived axis — every
  golden INPUT is tile-aligned by construction, and the op has no dim
  parameter). ✔
- **SUPPORTED** — present; declares every gated axis: `dtype` [bf16, f32],
  `layout` [TILE], `topology` [Linear]. Every axis the kernels gate on appears. ✔
- **EXCLUSIONS** — present, `[]`. No cell inside SUPPORTED is refused. ✔
- **validate()** — first line of `all_reduce(...)`. Ordering: universal
  structural ValueErrors (MeshDevice, `(1,N)` line, N ≥ 2, rank ≥ 2, not
  sharded, tile-aligned H/W) → per-axis SUPPORTED gate (typed
  `UnsupportedAxisValue`) → EXCLUSIONS (`ExcludedCell`) → axis-value-dependent
  structural checks (16 B page gate, `P·page ≤ 512 KiB` accumulator budget,
  output-spec equality). Typed refusals verified on hardware (extended test +
  golden lenient-xfail Ring cell). ✔
- **Op file does NOT declare INVALID** — confirmed; INVALID lives only in
  `feature_spec.py`. ✔
- **Package `__init__.py`** re-exports `all_reduce`, `SUPPORTED`, `EXCLUSIONS`,
  `INPUT_TAGGERS` (the harness reads them at package level). ✔

### Prompt rules

`eval/prompts/all_reduce.txt` has no `## Rules` section — stock policy applies.
Its hard mandates are all satisfied and hardware-verified: generated from
scratch, no wrapping/import/dispatch of any existing CCL op (imports audited);
exact import path + signature; registry contract with typed refusals; ONE
`generic_op` dispatch with compute overlapping fabric arrival (T4/T7 double-inc
+ two-way poll — the sequential two-dispatch split that fails acceptance is
exactly what the stale sibling-clone op does, underscoring the env footgun
above); output_tensor path returns the supplied handle; program-cache hit with
surviving GlobalSemaphores.

### INVALID audit (`eval/golden_tests/all_reduce/feature_spec.py`)

`INVALID = []`. Correct: the TARGET universe is {bf16, f32} × TILE × Linear —
every cell is constructible, so there is no structural impossibility to
declare. The canonical `bf8b × ROW_MAJOR` entry does not apply (neither value
is in TARGET, so the cell is outside the golden cartesian). No index axes → no
canonicalization cells. Not norm-like → no weight cells. Well-formed; no
change requested.

---

## Precision Baseline

`tests/ttnn/unit_tests/operations/all_reduce/test_all_reduce_precision_baseline.py`
— 8/8 green on the `(1, 4)` Blackhole line (N = 4 contributions). Oracle:
host element-wise sum accumulated in fp32, cast once to the device dtype.
Metrics are **worst-device** values (each device computes its own sum with its
own arrival order). `max ULP@scale` = max |err| in units of the ULP spacing at
the oracle's max magnitude.

| Shard shape | dtype | PCC (worst dev) | Max Abs Err | Mean Abs Err | Rel RMS Err | Max ULP@scale |
|-------------|-------|-----------------|-------------|--------------|-------------|---------------|
| (1,1,32,32)   | bfloat16 | 0.9999955 | 0.03125  | 0.00343  | 0.00342  | 1    |
| (1,1,32,32)   | float32  | 0.9999999 | 0.00689  | 0.000895 | 0.000627 | ~14.5k |
| (1,1,64,128)  | bfloat16 | 0.9999955 | 0.03125  | 0.00352  | 0.00348  | 0.5  |
| (1,1,64,128)  | float32  | 1.0000000 | 0.00693  | 0.000944 | 0.000634 | ~7.3k |
| (2,1,32,64)   | bfloat16 | 0.9999955 | 0.0625   | 0.00357  | 0.00357  | 1    |
| (2,1,32,64)   | float32  | 1.0000000 | 0.00890  | 0.000932 | 0.000642 | ~9.3k |
| (1,1,256,256) | bfloat16 | 0.9999957 | 0.0625   | 0.00353  | 0.00351  | 1    |
| (1,1,256,256) | float32  | 0.9999994 | 0.00899  | 0.000927 | 0.000630 | ~9.4k |

**Assessment**: Excellent and shape-stable up to P = 64 (the largest resident
accumulator any suite exercises; acceptance/golden stop at P = 8). **bf16**
worst error is ≤ 1 ULP of the output at its own scale — pure bf16 storage
quantization of the sum; rel-RMS ≈ 0.0035 is the bf16 mantissa budget for an
N=4 accumulate. **float32** rel-RMS ≈ 6.3e-4 (~2⁻¹⁰·⁶ relative) is NOT
fp32-rounding-limited: fp32 operands are truncated in the FPU srcA/srcB
registers (~10-bit mantissa) on the add path even with `fp32_dest_acc_en` — a
hardware datapath property (cf. reduce_scatter's 4.4e-4 for the same N=4 sum).
No lever in this op changes it (HiFi4 + fp32 DEST are already on).

**Recommended tolerances** (match acceptance/golden): bf16 `PCC ≥ 0.99`, f32
`PCC ≥ 0.999`; `check_output` tolerance `(0.99, 0.05)`. Generous headroom vs.
observed (worst PCC 0.9999955).

---

## Verifier CLI Summary

Artifact: `generated/all_reduce_verify/verifier_report.json` (from
`python_env/bin/python3 -m eval.verify_supported generated/all_reduce_verify
ttnn.operations.all_reduce`), joined from the golden hardware run
(`--junitxml` + `-p eval.axes_plugin` sidecar under the multidevice runner).

- supported_pass: **6** (3 INPUTS × {bf16, f32} × TILE × Linear — the ENTIRE golden cartesian)
- xfail_expected: **0** (`TARGET − SUPPORTED = ∅` on every axis)
- invalid_skipped: 0 (INVALID = [])
- supported_fail: **0**   ✓ ship gate
- xpass_drift: **0**      ✓ ship gate
- xfail_wrong_mode: **0** ✓ ship gate
- supported_marked_xfail: 0
- no_axes_found: 6 — the translated-suite tests (no axes sidecars by design;
  same as the reduce_scatter_average precedent): 5 passed (rank-2/3/4 shapes,
  program-cache-hit loop, every-device-same-sum) + 1 lenient xfail
  (`test_ring_all_reduce_refinement_axis` — typed `UnsupportedAxisValue`, the
  shared conftest's SupportRefusal hook).

### Gap accounting (`TARGET − SUPPORTED`, per task mandate)

Iterated `by_category.xfail_expected`: **empty**. Per-axis audit:

| Axis | TARGET | SUPPORTED | Missing |
|------|--------|-----------|---------|
| dtype | [bf16, f32] | [bf16, f32] | ∅ |
| layout | [TILE] | [TILE] | ∅ |
| topology | [Linear] | [Linear] | ∅ |

Zero `(axis, missing_value)` pairs → zero required refinements. The queue in
`op_requirements.md` is empty by construction, not by omission.

**Test suites run (all on real (1,4) Blackhole hardware via
`python_env/bin/python3 scripts/run_multidevice_sim_pytest.py --op all_reduce`,
aggregate exit 0):**

- Acceptance: `test_all_reduce.py` — 10 passed (8 shape×dtype cells,
  program-cache hit, output_tensor handle).
- Deterministic debug: `test_all_reduce_debug.py` — 4 passed (all-ones exact,
  per-device-constant exact, index-encoded positional-alignment exact).
- Extended (verifier-authored): `test_all_reduce_extended.py` — 5 passed
  (L1-interleaved bf16 + f32, f32 output_tensor path, typed-refusal pair
  Ring/ROW_MAJOR, structural ValueErrors: output-spec mismatch + accumulator
  budget).
- Precision baseline: 8 passed.
- Golden: `eval/golden_tests/all_reduce/` — 11 passed, 1 lenient xfail.

---

## Recommendations

1. **Refinement queue is empty** (see `op_requirements.md` — Phase 0 covers the
   full TARGET). Widening the op requires widening `TARGET` in
   `eval/golden_tests/all_reduce/feature_spec.py` first (via `/golden-tests`);
   without that, none of the candidates below can produce golden cells and they
   stay out of the queue per the registry model.
2. **Prioritized beyond-TARGET candidates** (from the design's own table, in
   suggested adoption order should TARGET widen):
   1. `topology = Ring` — cheapest: kernels are already ring-modular (T3); a
      host block-flow-table change + `ccl_dm_route(.., Ring)` wrap links. A
      live translated cell (`test_ring_all_reduce_refinement_axis`, currently
      lenient-xfail) flips to pass with no test edit. None of the current
      implementation skills cover CCL fabric axis expansion — this would be
      verifier-authored.
   2. Large-P support — lift the `P·page ≤ 512 KiB` resident-accumulator gate
      by chunking/spilling the accumulator. The gate currently rejects loudly
      (hardware-verified); nothing in TARGET's INPUTS approaches it (max P=8,
      precision suite proved P=64). If queued later this is
      `/memory-budget-metal` territory (bounded-CB reduction chunking).
   3. `bfloat8_b` / ROW_MAJOR / sharded / non-tile-aligned / multi-link /
      2-D-mesh `cluster_axis` — each needs its own pipeline change and a
      TARGET axis; the translated suite's Stage-3 reject notes document the
      2-chip-submesh and TG/Galaxy surfaces as structurally out of scope for
      this op shape.
3. **Perf observations (no failing cell — not queued):** the bandwidth-optimal
   RS+AG decomposition (drops the N× gather fabric traffic; the design defers
   it with the same reasoning as both reference reduce collectives), packet
   coalescing via `ccl_packet_dims`, per-block instead of per-page relay
   flush/barrier batching, and multi-core reduce (needs per-core inc fan-out).
4. **Environment guard for future passes:** always invoke the runner as
   `python_env/bin/python3 scripts/run_multidevice_sim_pytest.py …` from this
   repo. The login shell's `python3` belongs to a sibling clone
   (`/localdev/wransom/tt-metal-eval`) whose stale `ttnn` package shadows this
   tree and silently swaps in a different (two-dispatch) all_reduce.
   `probes/probe_budget_gate.py` documents the detection.
