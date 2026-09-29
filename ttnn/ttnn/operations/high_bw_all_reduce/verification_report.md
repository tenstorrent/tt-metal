# Verification Report: high_bw_all_reduce

Box: Blackhole P150 ×4 QuietBox (2×2 mesh, FABRIC_2D). Every group has `G = 2` (head + tail).
The `middle` role is **not** exercised on this box by any Phase 0 cell. It is first exercised by the
`cluster_axis=None` snake (`G = 4`) in Refinement 1.

## Code Review

Fixes applied in this pass:

1. **Compute schedule now works in whole blocks (the expression check).** `reduce_block` used to be one
   helper call over the core's entire assignment with the default `PerTile` wait/pop/reserve/push. That
   meant a CB handshake per tile, and `chunk_tiles` never reached the compute kernel, even though the
   reader pushes whole chunks and the writer waits on whole chunks. The kernel now takes `chunk_tiles` as
   a CT arg (single source: the host's `chunk_tiles`) and makes one `add` / `copy` call per block with
   `WaitPolicy::Upfront` + `PopPolicy::AtEnd` + `InputTileMapping::Block` on the inputs and
   `ReservePolicy::Upfront` + `PushPolicy::AtEnd` on `cb_reduced`. `block_size(chunk_tiles)` lets the chain
   batch tiles into DEST, and the chain clamps it to DEST capacity. The RT arg changed from
   `num_tiles` to `num_blocks`.
   - Pitfall hit and fixed during the pass: with an upfront wait, the default `InputTileMapping::Scalar`
     means "wait/pop **1** tile and always read tile 0". That gave PCC ≈ 0.03 on 29/36 acceptance tests.
     `InputTileMapping::Block` is required for a chunk-wide window.
   - Device-ns is unchanged. The op is fabric-bound (see Perf baseline).
2. **DRY: `CONTROL_WORD_STRIDE` and `MAX_DATA_HEADERS`.** Both were restated as `constexpr` literals in
   `port_fwd.cpp` and `port_bwd.cpp`, and the stride also lived on the host. They are now host constants
   in `high_bw_all_reduce_program_descriptor.py`, passed as CT args 10/11 to both port kernels.
3. **Removed a dead CT arg.** The reader's `packet_tiles` was unused after the implementer's
   chunk-granular arrival counter (`[[maybe_unused]]`). It was dropped and the CT indices shifted.
4. **L1 ledger currency.** The ledger's main body still carried the design's 32 KiB / `W = 8` / 192 KiB
   numbers, and only an appended deltas table held the implemented values. The main body is now corrected
   (see L1 Ledger Audit).

Checked and found correct (no change):
- `void kernel_main()`, `api/dataflow/dataflow_api.h` includes, `TensorAccessor` for all DRAM I/O.
- CB sync balances on every path. `cb_local_input`: the reader pushes `chunk_tiles` per block and compute
  pops `chunk_tiles`. `cb_remote_partial` (non-head): the reader pushes `chunk_tiles` per block and compute
  pops `chunk_tiles`. `cb_reduced`: compute pushes `chunk_tiles` and the writer pops `chunk_tiles`. The
  ragged last chunk keeps nominal counts, and only DRAM I/O narrows to the valid pages.
- Every port wait loop forwards the neighbour's credits (the deadlock-freedom rule).
- Cross-invocation counters use an atomic subtract of the invocation total and are never reset.
  `test_back_to_back_invocations` passes.
- Broadcast: none needed (both add operands hold full tiles).
- Helper usage: the compute path uses `compute_kernel_lib::add` / `copy`. The dataflow paths are raw by
  necessity: the kernel library has no Fabric send-ring helper, `mcast_pipe` is an on-chip multicast and
  does not apply to chip-to-chip unicast, and `local_copy_helpers` only covers self-aimed reads.
  `op_design.md` records this.

Deferred to the refinement queue (architectural):
- Perf headroom on the fabric send path (Refinement 3). Details under Recommendations.

## Prompt rules check (`eval/prompts/high_bw_all_reduce.txt ## Rules`)

| Rule | Applies now? | Status |
|---|---|---|
| MUST be native Fabric transport, no CCL ops, no host round-trip | yes | ✓ One `ttnn.generic_op(MeshProgramDescriptor)` per call. The kernels use raw EDM sender connections. |
| MUST accumulate each step in fp32 | yes | ✓ `fp32_dest_acc_en=True` on the reducer compute. The FPU bf16+bf16 add is exact into fp32 DEST, and the partial travels in bf16 (allowed). |
| MUST include every member exactly once | yes | ✓ `test_rank_identity` and `test_single_contributor` are bit-exact on axes 0 and 1. |
| MUST honour `num_links` (no clamping) | yes | ✓ An explicit value is used as `num_lanes` or rejected with `ValueError`. `None` means `usable_links`. |
| MUST use `from ttnn.operations.ccl import Topology` at import time | yes | ✓ |
| prefer spreading the payload over many worker cores per link | yes | **Advisory.** `W = 4` reducers per link (10 cores/device at 2 links). The implementer measured this as link-bound: the port send loop, not reducer compute, is the bottleneck. Refinement 3 re-measures `W`. |
| prefer reduction math overlapped with transport | yes | ✓ The chunk pipeline adds chunk `k` while chunks `k±1` are in transport. |
| prefer both ring directions for Ring; MUST use one-hop links for `None`; MUST NOT leak padding; MUST move fp32 end-to-end | not yet (Ring / None / non-aligned / fp32 are outside SUPPORTED) | These re-arm in Refinements 1 and 2. They are restated in those entries. |

## Registry Conformance

- `INPUT_TAGGERS` has exactly one entry, `alignment`, with signature `(inputs, axes)`. It reads the
  per-device `inputs[0]`.
- `SUPPORTED` covers all six axes (dtype, layout, alignment, cluster_axis, topology, num_links) and
  matches the Phase 0 contract exactly.
- `EXCLUSIONS = []`.
- `validate()` checks SUPPORTED per axis and then EXCLUSIONS, raising
  `UnsupportedAxisValue` / `ExcludedCell`. `num_links=None` is the default and skips the check.
- The entry point calls `validate()` first. Caller errors (`ValueError`) come after it.
- The op file does **not** declare `INVALID`.
- No auto-fixes to SUPPORTED were needed (zero `xpass_drift`).
- **INVALID audit** (`feature_spec.py`): `INVALID = []`, which is correct. TARGET has no `bfloat8_b` and
  no `ROW_MAJOR`, so the canonical bf8b+RM entry does not apply. There are no weight tensors, so the
  norm-style canonicalization does not apply either. Topology feasibility is pruned by device capability
  in `topology.py`, not by INVALID, which is also correct.

## Design Conformance

- **Algorithm**: R1 `chain_line`, as designed: reduce hop by hop toward `p = G−1`, relay finals toward
  `p = 0`, pipelined per chunk. ✓
- **Pipeline topology / RISC ownership**: reader on RISCV_1 (NoC0), writer on RISCV_0 (NoC1), port_fwd
  on RISCV_0, port_bwd on RISCV_1. This matches the design. The implementer moved the reducer NoCs
  explicitly as a perf-lamp choice.
- **Work distribution**: the lane × reducer split of the independent `tile` axis is as designed.
  `W = min(REDUCERS_PER_LANE, max lane blocks)`, and the layout is identical on every device.
- **Implementer deltas (measured, documented in `l1_ledger.md`)**:
  - `CHUNK_BYTES_TARGET` is 64 KiB (design: 32 KiB).
  - `REDUCERS_PER_LANE` is 4 (design: 8).
  - Arrival counters count chunks rather than packets: a fused atomic inc is sent only on the last packet
    of each chunk.
  - Tail reducers write output DRAM directly.

  All are sanctioned perf-lamp turns.
- **Blocking-model fidelity**:
  - Every block knob is a single-sourced host constant: `CHUNK_BYTES_TARGET`, `REDUCERS_PER_LANE`, the
    five depths, and now `MAX_DATA_HEADERS` / `CONTROL_WORD_STRIDE`.
  - Every CB capacity is `depth · chunk_tiles`, and no capacity depends on the tensor size.
  - After fix 1, the compute, reader and writer schedules all sync once per chunk. The port kernels loop
    per packet only inside `forward_partial_block` / `relay_final_block`. That is the transport quantum
    (mechanism cap), with no per-packet completion boundary: headers ring-buffer and flush only on header
    reuse.

## L1 Ledger Audit

- **Currency**: the three reducer CBs (`cb_remote_partial` on the `reducer_landing` shard,
  `cb_local_input`, `cb_reduced`) and the three port raw-L1 regions (control, staging, final landing) all
  have rows. The main-body size expressions and example totals were **stale** (32 KiB / `W = 8`) and are
  now corrected to the implemented values: reducer 384 KiB, port 514 KiB (bf16).
- **Capacity vs live set**:
  - Every CB is `2 · chunk_tiles` with a live set of `2 · chunk_tiles` (double-buffered), and each depth
    is justified in its row.
  - The only over-capacity case is `cb_remote_partial` on the head, which is allocated but unused. It is
    justified: the landing tensor must be lockstep/uniform across devices because senders address it
    symmetrically.
  - Nothing is spanned while under-scaled.
- **Page format vs DEST**:
  - bf16 pages under `fp32_dest_acc_en=True` is the audit's "under" case.
  - Answered by the spec: "partials MAY travel in the input dtype". The output ABI is the input dtype, and
    each step still accumulates in fp32 DEST.
  - Refinement 2 makes every page Float32 for fp32 inputs.
- **Disjoint lifetimes**: none. All buffers are concurrent in the chunk pipeline, and each has a stated
  reason.
- **Bounds / closed form**: every symbol is bounded. `tensor_tiles` appears only in trip counts.
- **Traffic budget**:
  - DRAM `2S` per device, which is the minimum.
  - Fabric `S` per link direction per hop (the line lower bound).
  - On-chip `3S`. This is consistent with the code, including the tail's direct DRAM write.
- **Cheapest-traffic split**: R3 `rotated_chain_ring` at `(G−1)/G · S`. It is a `deferred` regime with
  the positive reason "topology=Ring is outside Phase 0 SUPPORTED". It is now queued as Refinement 1.
- **Block-size defaults**: the design departs from "spread over the full grid". The reason is structural
  (link rate caps throughput, not grid size), and the implementer measured it (64 KiB × `W = 4` beat
  32 KiB × `W = 8` by 7–9%, and `W = 8` at 64 KiB collided with the reducer CB region). That is recorded.
  Refinement 3 re-tunes it.
- **Per-core footprint**:
  - Reducer: `(RECV + INPUT + REDUCED)_DEPTH · chunk_bytes`, which scales with `CHUNK_BYTES_TARGET` and
    the depths.
  - Port: `align_up(3W·16, tile) + W·(STAGING + FINAL)_DEPTH · chunk_bytes`, which scales with `W` and
    `CHUNK_BYTES_TARGET`.
  - No ledger findings needed a refinement.

## Precision Baseline

`test_high_bw_all_reduce_precision_baseline.py`, bf16 randn, `num_links=None`, `G = 2`. The worst device
per shape is shown. Errors are measured against the exact fp32 group sum; ULP is measured against the
once-rounded bf16 reference.

| Shape | axis | PCC | Max Abs Err | Mean Abs Err | Rel RMS Err | ULP max / mean (bf16) | bit-exact vs bf16 ref | got/true ratio p5 / median / p95 |
|-------|------|-----|-------------|--------------|-------------|------------|------------|------------|
| (1,1,32,32) | 0 | 0.9999985 | 0.0156 | 0.00156 | 0.00190 | 1 / 0.107 | 89.3% | 0.9982 / 1.0000 / 1.0033 |
| (1,1,32,32) | 1 | 0.9999988 | 0.0088 | 0.00146 | 0.00181 | 1 / 0.120 | 88.0% | 0.9984 / 1.0000 / 1.0032 |
| (1,1,256,512) | 0 | 0.9999987 | 0.0156 | 0.00144 | 0.00180 | 1 / 0.111 | 88.9% | 0.9983 / 1.0000 / 1.0033 |
| (1,1,256,512) | 1 | 0.9999987 | 0.0156 | 0.00145 | 0.00180 | 1 / 0.109 | 89.1% | 0.9983 / 1.0000 / 1.0033 |
| (4,2048,1024) | 0 | 0.9999987 | 0.0156 | 0.00145 | 0.00180 | 1 / 0.111 | 88.9% | 0.9983 / 1.0000 / 1.0033 |
| (4,2048,1024) | 1 | 0.9999987 | 0.0156 | 0.00145 | 0.00180 | 1 / 0.110 | 89.0% | 0.9983 / 1.0000 / 1.0033 |
| (1,1,4096,4096) | 0 | 0.9999987 | 0.0156 | 0.00145 | 0.00180 | 1 / 0.110 | 89.0% | 0.9983 / 1.0000 / 1.0033 |
| (1,1,4096,4096) | 1 | 0.9999987 | 0.0156 | 0.00145 | 0.00180 | 1 / 0.111 | 88.9% | 0.9983 / 1.0000 / 1.0033 |

**Assessment**:
- Output is within **1 bf16 ULP** of the best-possible bf16 result everywhere. About 89% of outputs are
  bit-exact, and the rest differ by exactly 1 ULP. This is a packer tie-rounding difference versus torch's
  round-to-nearest-even; the fp32 DEST add itself is exact.
- The ratio median is 1.00000 and the spread is symmetric, so there is **no scale bug**.
- Error is shape-independent, as expected for a fixed `G`.
- With `G > 2` the partial is rounded to bf16 once per hop, so the bound is `G−1` ULP. The test asserts
  `≤ G−1`.

**Recommended tolerances** (bf16): PCC ≥ 0.9999, atol = 1 bf16 ULP of |sum| (≈ 0.0156 for |sum| < 4 at
unit-variance inputs, G=2), rtol ≈ 0.008 (2⁻⁷). The golden TOLERANCES (0.995 / 0.04 abs RMS) hold with a
very large margin.

## Perf baseline (device kernel ns, `--profile`, max over the 4 devices)

| Shape (bf16) | num_links=1 | num_links=2 | GB/s per link direction |
|---|---|---|---|
| 1×1×2048×2048 (8 MB) | 392 µs | 203 µs | 21.4 / 20.7 |
| 1×1×4096×4096 (32 MB) | 1.53 ms | 0.77 ms | 21.9 / 21.8 |

Both axes are identical within noise, and the numbers are unchanged by this pass's fixes (they match the
implementer's recorded 0.204 / 0.77 ms). The ceiling reference is
`tests/tt_metal/tt_fabric/test_infra/golden/golden_bandwidth_summary_blackhole_p150_x4.csv`:
`NOC_UNICAST_WRITE` at 4096 B packets is **≈ 38.4 GB/s** per link direction. The op therefore runs at
**~56%** of the plain-write link ceiling, which leaves **≈ 1.8× headroom** (Refinement 3).

## Verifier CLI Summary

Golden run: `eval/eval_test_runner.sh eval/golden_tests/high_bw_all_reduce/`: **52/400 passed, 12
failed, 0 hangs**. The report is committed as `verifier_report.json` next to this file.

- supported_pass: 48
- xfail_expected: 336
- invalid_skipped: 0
- supported_fail: 0
- xpass_drift: 0
- xfail_wrong_mode: 0
- no_axes_found: 16. These are the non-registry `test_regression.py` tests: 4 pass (tile-aligned
  rank-identity and single-contributor on both axes). The other 12 fail with `UnsupportedAxisValue`:
  `alignment=w_non_aligned` for the 4096×2050 cases, and `dtype=FLOAT32` for every `test_magnitude`.
  They are expected refusals at Phase 0 and turn on with Refinement 2.

xfail_expected breakdown by blocking axis (2×2 box):

| Missing axis values | Cells | Refinement |
|---|---|---|
| `cluster_axis=None` (± Ring, ± fp32, ± non-aligned) | 192 | R1 (+R2 for the fp32/non-aligned products) |
| `dtype=FLOAT32` (axis 0/1, Linear) | 48 (+48 × non-aligned) | R2 |
| `alignment=w_non_aligned` / `h_non_aligned` (axis 0/1, Linear) | 48 (+48 × fp32) | R2 |

Per-axis `topology=Ring` cells (`cluster_axis ∈ {0,1}`) are not emitted on a 2×2 (they need `G ≥ 3` and a
wrap link). They appear on a torus Galaxy and are covered by R1.

Acceptance: `tests/ttnn/unit_tests/operations/high_bw_all_reduce/`, **36/36** (`--dev`), plus the
8-test perf harness and the 8-test precision baseline.

## Recommendations

- **Non-aligned shapes already work.** A verifier probe (`probes/probe_023.py`) bypassed `validate()` on
  (1,1,64,70), (1,1,70,64), (3000,3000) and (1,4001,2048), axes 0/1. All returned the exact logical shape
  with max error of 1 bf16 ULP. The op sums physical pages and reuses the input TensorSpec, as the design
  predicted. The alignment half of Refinement 2 is therefore SUPPORTED + tests.
- **Test-infra friction (not an op bug).** Under `--dev` (watcher), a passing FABRIC_2D run can leave the
  ERISC routers unable to return to base FW. `run_safe_pytest.sh` only resets after a hang, so the next run
  dies at mesh open with `Timed out waiting for active ethernet core`. The workaround used throughout this
  pass was `touch /tmp/tt-device.dirty` before each run, so the script resets the board under its own
  device lock. Refinement implementers should do the same. Also run **one fabric test module per session**: two module-scoped mesh fixtures in one `--dev` session fail the second module at mesh open (observed: acceptance 36 pass + precision 8 errors in a combined run, and 8/8 pass when run alone).
- **`_global_semaphores` key hard-codes `"Linear"`** and caches per `id(mesh_device)`, keeping the mesh
  object alive. Refinement 1 must key on the real topology and on the path shape (snake vs axis), or a
  Ring and a Linear config would share counters. This is in R1's notes.
- **Output allocation** uses `input_tensor.spec` and ignores `memory_config`. That is equivalent today,
  because only DRAM interleaved is accepted for both. If a non-DRAM output memory config is ever
  admitted, allocate from `memory_config`.
- **Middle role is untested on this box.** Every Phase 0 cell on a 2×2 has `G = 2`. The `middle` code path
  (receive + add + forward, with both port connections open) first runs in R1's `G = 4` snake. R1 should
  include a `G ≥ 3` exactness test (rank identity) before perf work.
- **Kernel-library gap.** A credit-gated Fabric send-ring helper (header ring + chunk-granular fused inc +
  credit forwarding inside waits) would remove ~200 lines of raw EDM code shared by port_fwd and port_bwd.
  It is noted for the helper-library owners and is not a refinement.
