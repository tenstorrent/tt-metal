# Verification Report: reduce_scatter

**Op class**: multi-device compute-CCL (fabric store-and-forward gather fused, in ONE
`ttnn.generic_op` dispatch, with an arrival-ordered incremental N-way SUM and a per-device-DISTINCT
scatter).
**Verified on**: real 4-chip Blackhole QuietBox, mesh `(1, 4)` with `FabricConfig.FABRIC_1D`
(topology `bh_quietbox_1x4_hw`, runtime = hardware), via
`scripts/run_multidevice_sim_pytest.py --op reduce_scatter` — the correct runner for a CCL op
(`run_safe_pytest.sh` is single-device-oriented and forces slow dispatch on sim). Aggregate exit = 0
on every run below.

---

## Code Review

The implementation follows `op_design.md` faithfully on every binding dimension. Review found **no
correctness defects and no required fixes** — the helper usage, CB protocol, semaphore discipline,
and registry wiring are all as designed. Verified in detail:

### Design conformance (binding dimensions — all match)

| Dimension | Design | Implementation |
|---|---|---|
| Algorithm | line store-and-forward gather of whole shards fused with an arrival-ordered incremental reduce, ONE dispatch | matches — single `MeshProgramDescriptor`, one `generic_op`; reduce pass k starts on arrival k's semaphore, overlapping arrival k+1's flight (T4/T7) ✓ |
| Pipeline topology | 3 cores/device: (0,0) fwd relay, (0,1) bwd relay, (0,2) reduce; 7 kernels from 5 sources | matches ✓ |
| Work distribution | fixed roles, Linear block-flow table T1/T2 (`_block_flow` — fwd sends `1+i` iff `i<N-1`, arrivals `i`; mirror bwd) | matches ✓ |
| Inter-core comm | fabric `write_page` into the neighbour's `gather_buffer` (uniform mesh address) + TWO in-order counting incs per block (relay core + reduce core), two `GlobalSemaphore`s cached per mesh, parked on the descriptor, R1 re-arm by every consumer | matches ✓ (program-cache test exercises the re-arm; passes) |

### Helper usage (verified against the helper banners)

- **Fabric egress** is the full typestate path: `FabricStreamSender<>` (rt-arg block FIRST, cursor
  from 0) → `open(unicast_route(1))` → `arm_unicast_write(page_size)` / `arm_inc(1)` →
  `write_page` / `inc` → `close()`. `noc_async_writes_flushed()` before every `cb_pop_front` (R7);
  both incs on the SAME connection after the block's last page (R8). 2 armed channels — well inside
  the 8-header pool.
- **Compute** is helper-native end to end: `binary_op_init_common` boot (helper pre-condition),
  ONE armed `BlockAccumulate` (R10), C1 seed via degenerate-copy `sum_blocks(num_blocks=1,
  pop_input=true)` (R4), C2 `rearm()` after `sum_blocks`'s acc_to_dest post-condition (R3), C3
  in-place `run(g)` (`cb_b == cb_out`, sound per the helper's verified pop-before-reserve ordering),
  C4 degenerate-copy `sum_blocks` to the writer CB. Matches the header contract exactly, including
  the documented reason C4 cannot be a second armed accumulator (singular unpack/math state).
- **Schedule** uses `slice_tile_offset` + `SliceRowWalker` + `is_supported_scatter_dim` from the
  shared header; identical walk per contribution (R11) keeps `add_tiles` positionally aligned and
  makes the writer's dense drain valid with no walker of its own.
- **Receive-side sync** (two-way monotone poll + `noc_semaphore_wait_min` + re-arm) is op-owned raw
  NoC, matching the dataflow helper's documented scope split (banner: receive ingress/wait is the
  op's). No raw multicast/handshake exists — nothing to migrate to `mcast_pipe.hpp`.
- All raw-API fallbacks carry the design's file:line-cited rejections (`sum_blocks` position-major,
  `reduce_helpers`' `reduce()`, `FabricDuplexSender`, the `LineSliceCursor` step machine).

### Correctness checks

- **CB sync balance** (per device): `cb_relay_pages` push = pop = `num_sends_dir * P` per relay
  core (0 = 0 on idle line-end directions); `cb_contributions` push `N*S` = pops `S + (N-1)*S`;
  `cb_accumulator` push `N*S` = pops `(N-1)*S + S` (compute is the single producer, R2);
  `cb_output_tiles` push `S` = pop `S`. Semaphore waits observed == incs issued before every reset
  (line-end pure receivers still wait all arrivals before re-arming, T6).
- **TensorAccessor** everywhere (no `InterleavedAddrGen`), `void kernel_main()`, includes via
  `api/dataflow/dataflow_api.h` — all correct.
- **Granule contract**: host `_granule` picks g ∈ {4,2,1} dividing S; kernels `static_assert(S % g
  == 0)` and `g <= DEST_AUTO_LIMIT`; every CB capacity is a multiple of its quantum (wrap-safe).
- **Broadcast**: N/A — every binary op is full-tile `add_tiles`; no broadcast operand, no redundant
  broadcast fills.
- **Registry/kernel arg wiring**: CT/RT layouts cross-checked host↔kernel for all 7 descriptors
  (relay reader/writer ×2 directions, reduce reader/compute/writer) — indices line up; the relay
  writer's fabric block is appended post-construction via the live `runtime_args` view because
  `build_ccl_fabric_rt_args` mutates the program, exactly as the design mandates.

### Advisory (no fix; no failing cell to point at)

- **Fused write+inc not used**: the relay writer sends each block's last page via `write_page` then
  issues two separate `inc()` packets. `FusedWriteIncChannel` (`arm_fused_write_inc`, flush=true)
  could fold the last page + the relay-core inc into one packet, saving one fabric packet per block.
  The design's API mapping deliberately specifies the `arm_unicast_write` + `arm_inc`×2 shape (the
  two incs target two different cores, so at most one can fuse); the implementation matches the
  design and is silicon-verified. Performance-only; revisit with Refinement 6 (packet coalescing)
  if bandwidth ever matters here.
- **`_SEMAPHORE_CACHE` keyed by `id(mesh_device)`**: id-reuse after a mesh_device is garbage
  collected could theoretically alias a new device at the same address. Same idiom as the adopted
  siblings (all_reduce, reduce_scatter_average); acceptable for the pipeline's usage pattern.
- **Relay reader seeds page-by-page** (read → barrier → push per page). A 2-page lookahead could
  overlap the local NoC read with the previous page's fabric egress. Runs on the smallest data path;
  marginal.

---

## Registry Conformance

Confirmed the four declarations present and correctly wired in `reduce_scatter.py` (re-exported at
package level for the golden harness):

- **`INPUT_TAGGERS = {}`** — empty by design (every golden INPUT is tile-aligned by construction; no
  shape-derived axis). `validate()` still iterates it with the correct `tagger(inputs, axes)`
  signature, so adding a tagger later is a drop-in.
- **`SUPPORTED = {dtype:[bfloat16, float32], layout:[TILE], topology:[Linear], dim:[3]}`** — covers
  every axis the kernels gate on, including the single-valued `dim` and `topology` keys the feature
  spec mandates (the harness derives xfail marks only from declared axes). `dim` is canonicalized to
  the positive convention (`dim % 4`) BEFORE the membership test, so `-1 ≡ 3` (acceptance-verified).
- **`EXCLUSIONS = []`** — present, empty (no in-SUPPORTED cell is refused).
- **`validate()`** — universal structural checks (`ValueError`: MeshDevice, `(1,N)` line N ≥ 2,
  rank 4, dim range, not sharded, H/W tile-aligned) → axis gate (`UnsupportedAxisValue` per axis,
  then `ExcludedCell` cell-level — correct order) → axis-value-dependent structural checks
  (`ValueError`: slice divisibility, 16 B page alignment, S ≤ 256 accumulator budget, output_tensor
  spec). The design-blessed ordering: an out-of-SUPPORTED axis value always yields the typed
  refusal, never a shape ValueError computed under the wrong axis. Public `reduce_scatter()` calls
  `validate()` on its first line, before any allocation.
- **No `INVALID` symbol in the op file** — confirmed absent. INVALID is sourced from the golden
  feature spec.

**No drift. No auto-fixes needed.** `xpass_drift = supported_fail = xfail_wrong_mode = 0`.

### INVALID audit (`eval/golden_tests/reduce_scatter/feature_spec.py`)

`INVALID = []`, and this is **correct** for the current TARGET:

- `TARGET = {dtype:[bf16, f32], layout:[TILE], topology:[Linear, Ring], dim:[3, 2]}` — every
  combination is constructible (the INPUTS keep dims 2 AND 3 multiples of 256 precisely so every
  `dim`/mesh-size combination stays tile-aligned; the spec's own comment documents this design).
- Single-tensor op — no cross-tensor-axis coupling risk.
- The canonical bf8b + ROW_MAJOR entry is correctly omitted: TARGET contains neither ROW_MAJOR nor
  bfloat8_b, so the entry would reference axis values the harness never generates.
- No norm-like weight axes → no no-weight canonicalization cells.

No changes to `feature_spec.py` proposed.

### Prompt rules

`eval/prompts/reduce_scatter.txt` has no `## Rules` section — stock policy applies. Its hard
mandates are all satisfied: generated from scratch (5 newly-authored kernels; no wrap/import/call of
any existing CCL op), exact import path + positional signature, registry contract with `dim` and
`topology` as SUPPORTED keys, ONE `generic_op` dispatch with compute overlapping fabric arrival
(no gather-then-reduce two-dispatch split), loud ValueError shape rejection, typed
`UnsupportedAxisValue` refusals.

---

## Precision Baseline

Measured on the `(1, 4)` Blackhole line (N = 4 summands), worst case across all four devices'
DISTINCT output slices; oracle accumulated in fp32 then cast, so the reference is not itself limited
by bf16 rounding. From `tests/ttnn/unit_tests/operations/reduce_scatter/
test_reduce_scatter_precision_baseline.py` (8/8 pass).

| Shard shape | dtype | PCC (worst dev) | Max Abs Err | Mean Abs Err | Relative RMS Err | Max ULP* |
|-------------|-------|-----------------|-------------|--------------|------------------|----------|
| (1,1,32,256)  | bfloat16 | 0.9999954 | 0.031250 | 0.003532 | 0.003474 | 3.0 |
| (1,1,32,256)  | float32  | 0.9999999 | 0.008422 | 0.000953 | 0.000640 | ~3.5e4 |
| (1,1,64,512)  | bfloat16 | 0.9999955 | 0.062500 | 0.003548 | 0.003513 | 3.0 |
| (1,1,64,512)  | float32  | 0.9999999 | 0.007335 | 0.000934 | 0.000636 | ~5.3e4 |
| (1,1,256,512) | bfloat16 | 0.9999954 | 0.062500 | 0.003562 | 0.003521 | 3.0 |
| (1,1,256,512) | float32  | 0.9999999 | 0.008462 | 0.000933 | 0.000631 | ~5.2e4 |
| (2,1,64,256)  | bfloat16 | 0.9999954 | 0.062500 | 0.003548 | 0.003533 | 3.0 |
| (2,1,64,256)  | float32  | 0.9999999 | 0.007335 | 0.000930 | 0.000626 | ~5.4e4 |

\* ULP of the OUTPUT dtype at the oracle's magnitude (floored at magnitude 1).

**Assessment**: Accuracy is excellent and shape-independent.

- **bf16 error is exactly the accumulation-rounding budget the design predicted (R16)**: max error
  is **3 bf16 ULP = N−1 pack roundings** — the running sum is packed back to the bf16
  `cb_accumulator` once per incremental pass, so each of the 3 remote-contribution passes
  contributes at most one rounding. rel-RMS ≈ 0.0035 ≈ half the bf16 mantissa step. Refinement
  candidate "fp32 accumulator under bf16 inputs" (design Refinement 5) would cut this to a single
  final rounding — no failing cell to point at, so it stays a recommendation.
- **fp32 error (rel-RMS ≈ 6.3e-4)** reflects the FPU's TF32-precision SrcA/SrcB mantissa under
  HiFi4 (the fp32 DEST accumulates exactly, but each `add_tiles` operand load rounds to ~11 mantissa
  bits). Same magnitude as the all_reduce baseline (7e-4); expected, negligible vs the 0.999 gate.
  The large "Max ULP" numbers are near-zero oracle elements measured against the eps-at-magnitude-1
  floor, not large relative errors.

**Recommended tolerances** (match the golden/acceptance gates, generous headroom vs observed):
bf16 `PCC ≥ 0.99`, fp32 `PCC ≥ 0.999`; allclose `atol ≈ 0.15` (bf16) / `0.02` (fp32) for N = 4–8
device sums of unit-normal shards.

---

## Verifier CLI Summary

Artifact: `generated/reduce_scatter_verify/verifier_report.json` (from
`python3 -m eval.verify_supported generated/reduce_scatter_verify ttnn.operations.reduce_scatter`).
Golden suite: 3 INPUTS × {bf16, f32} × TILE × {Linear, Ring} × {3, 2} = 24 registry cells
(+ 5 translated tests, tracked as `no_axes_found` — they carry no registry axes by design).

- supported_pass:     **6**   (3 INPUTS × 2 dtypes × Linear × dim=3)
- xfail_expected:     **18**  (typed `UnsupportedAxisValue` refusals — breakdown below)
- invalid_skipped:    0   (INVALID = [])
- supported_fail:     **0**  ✓ ship gate
- xpass_drift:        **0**  ✓ ship gate
- xfail_wrong_mode:   **0**  ✓ ship gate
- supported_marked_xfail: 0
- no_axes_found:      5   (test_translated.py: 4 passed + 1 xfail Ring refinement cell — expected)

All loud categories are 0: SUPPORTED describes reality exactly.

**`xfail_expected` breakdown (TARGET − SUPPORTED), per-cell iteration of the report:**

| Axis gap | Cells | Disposition |
|---|---|---|
| `topology=Ring`, `dim=3` | 6 | → **Refinement 1** (`op_requirements.md`) |
| `topology=Ring`, `dim=2` | 6 | → Refinements 1 + 2 jointly (flip when BOTH land; whichever lands second collects them) |
| `topology=Linear`, `dim=2` | 6 | → **Refinement 2** |

Every `(axis, missing_value)` pair — `(topology, Ring)` and `(dim, 2)` — is covered by a refinement
entry. No pair is left undocumented; nothing is masked by INVALID.

**Test suites run (all on real silicon, aggregate exit 0):**
- Acceptance: `tests/.../reduce_scatter/test_reduce_scatter.py` — **15 passed** (5 shapes × 2 dtypes
  incl. the S=9/g=1 path + negative-dim alias + program-cache + output_tensor + 2 loud rejections).
- Golden: `eval/golden_tests/reduce_scatter/` — **10 passed, 19 xfailed** (xfail-strict).
- Precision baseline: `test_reduce_scatter_precision_baseline.py` — **8 passed**.
- Extended (verifier-authored): `test_reduce_scatter_extended.py` — **5 passed** (L1 interleaved
  input; the S=256 accumulator-budget boundary AT the cliff — shard (1,1,256,4096), the largest
  per-core CB footprint the Phase-0 op allocates — and one column PAST it rejecting loudly; fp32
  output_tensor path; out-of-range-dim + mismatched-output_tensor ValueErrors).
- Ring fabric precondition: `test_ring_fabric_probe.py` — **4 passed** (wrap-link route math,
  connection formation, 1-hop transfers both directions under FABRIC_1D — Refinement 1's fabric
  contract is confirmed live on this box).

---

## Recommendations

Refinement queue (see `op_requirements.md`): **Refinement 1 = Ring topology, Refinement 2 = dim=2.**
Both are CCL schedule/algorithm work with cross-device data dependencies — no skill in the current
inventory covers them (explicitly out of scope for `/interleaved-parallel`), so both are
verifier-authored entries working from `op_design.md`'s refinement sketches and the adopted-sibling
worked examples.

Directions **beyond the current TARGET** (not refinements — each would need `/golden-tests` to
expand `feature_spec.py` first; no failing cell points at any of them today):

- **fp32 `cb_accumulator` under bf16 inputs** (design Refinement 5, second half): would cut the bf16
  baseline's 3-ULP accumulation error to a single final rounding. Requires a mid-kernel
  data-format reconfig story (breaks the one-boot-init R12 economy) — bundle with large-S if ever
  taken.
- **Large-S support** (S > 256): the resident `cb_accumulator = S pages` is the Phase-0 L1 cliff,
  currently a loud ValueError (boundary verified at S=256 pass / S=264 reject). Chunking or
  spilling the accumulator is `/memory-budget-metal`-shaped work if a shape axis ever enters TARGET.
- **Multi-core reduce** (design Refinement 3): splitting S across reduce cores multiplies the
  per-block inc fan-out (or needs a local mcast of the arrival signal). Golden S ≤ 32 — one core is
  right-sized today.
- **True partial-sum line reduce-scatter** (design Refinement 4, the `LineSliceCursor`/
  `SyncCadence`/`line_rs_*` machine): drops the N× gather traffic; a different algorithm with a
  3-kernel step-flag agreement surface. The bandwidth lever if perf ever gates.
- **Packet coalescing / fused write+inc** (design Refinement 6 + the advisory above): fabric packet
  count reduction.
- **Sharded input** (currently a loud ValueError): a memory-config expansion needing a sharded
  reader/writer.
