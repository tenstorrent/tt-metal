# Verification Report: all_reduce

**Op class**: multi-device CCL **with a compute stage** (fabric gather + TRISC element-wise SUM).
**Verified on**: simulated Wormhole T3K line mesh `(1, 8)` with `FabricConfig.FABRIC_1D`
(topology `wh_t3k_allmmio_all_reduce`), via `scripts/run_multidevice_sim_pytest.py --op all_reduce`.
This is the correct runner for a CCL op — `run_safe_pytest.sh` forces slow dispatch on sim and has
no multichip/hang awareness. Aggregate exit = 0 on every run below.

---

## Code Review

The implementation closely follows `op_design.md` and the prompt's framework-owner mandates. It is a
gather-then-reduce algorithm across two ordered `ttnn.generic_op` dispatches on one command queue
(Phase A fabric store-and-forward gather → Phase B local N-way tile sum). Review found no
correctness defects; the fixes below are cleanliness.

### Fixed

1. **Unused `ring_size` compile-time arg (both Phase-A kernels).** `all_reduce_reader.cpp` (slot 5)
   and `all_reduce_writer.cpp` (slot 4) read `ring_size` into a `constexpr` local that is never used
   — every block index is derived from `my_chip_id` + `num_targets_{fwd,bwd}`. The slot is part of
   the deliberate uniform CT superset (keeps the discarded `if constexpr` branch's accessor offset at
   the fixed literal `8`), so the arg must stay in the layout. Marked the locals `[[maybe_unused]]`
   with a comment documenting why the slot is intentionally unread — preserves the CT-layout
   documentation while silencing any `-Wunused` noise. Recompiled fresh under the sim; all cells pass.

2. **`_num_line_devices` defined after use in `all_reduce_program_descriptor.py`.** The helper was
   defined at the *bottom* of the file but first called at line ~269 inside
   `create_gather_mesh_program_descriptor` (worked only via Python's runtime name resolution). Moved
   the definition up next to `_round_up`, before its first use.

### Reviewed and intentionally left as-is

- **Fabric egress uses the CCL kernel helper (`ccl_helpers_dataflow.hpp`).** The writer builds
  `FabricStreamSender<>` → `.open(unicast_route)` → `arm_unicast_write` / `arm_inc` → `write_page` /
  `inc` → `close()`. This is exactly the mandated safety-by-construction typestate path; the receive
  ingress, the counting `noc_semaphore_wait_min`, and the cache-reuse `noc_semaphore_set(sem, 0)`
  re-arm are op-owned raw NoC calls, matching the helper's documented split (the header banner states
  reduction collectives are out of scope for the dataflow helper). No raw
  `noc_async_write_multicast` / hand-rolled handshake exists — nothing to migrate to `mcast_pipe.hpp`.
- **Phase-B compute is the raw seed-then-accumulate idiom** (`copy_tile` seed into `DST[0]`, then
  `binary_dest_reuse_tiles<ELWADD, DEST_TO_SRCA>` for blocks 1..N-1, single-DST). `op_design.md`
  §"Helpers considered and rejected (Phase B)" justifies this: `reduce_helpers`' `reduce()` collapses
  the within-tile 32×32 dims (wrong op), and no element-wise-add chain helper exists in this tree.
  Verified by re-reading `ccl_helpers_dataflow.hpp` (reductions explicitly out of scope) and
  `reduce_helpers_compute.hpp`. Single-`DST[0]` accumulation is DST-capacity-safe for any N and for
  float32 — correct.
- **CB sync balance.** Phase A `cb_relay_pages`: reader pushes `(1 + num_relay_blocks)·P` pages when
  `my_num_targets > 0` (seed + relays), gated to push nothing at a line end; writer pops the same
  count (`k = 0..num_relay_blocks`, `P` pages each). Phase B: reader pushes N per position, compute
  waits/pops N, writer waits/pops 1 per compute push 1. Push == pop on every CB and every direction.
- **TensorAccessor** (not deprecated `InterleavedAddrGen`), `void kernel_main()`, and
  `api/dataflow/dataflow_api.h` includes — all correct.

### Advisory (no fix; no failing cell to point at)

- **Phase-A self-copy is fully serialized** — the forward reader does `noc_async_read_barrier()` then
  `noc_async_write_barrier()` per page over the P-page shard (`all_reduce_reader.cpp:75-80`). Correct,
  but it serializes each page; a batched read-all-then-write-all (or double-buffered pipeline) would
  overlap the local NoC copy. Runs once per device on the smallest of the data paths, so the win is
  marginal. Performance-only, no correctness or capability impact.
- **`_num_line_devices` is duplicated** across `all_reduce.py` and `all_reduce_program_descriptor.py`
  (4 identical lines). Left duplicated to avoid a cross-module import of an underscore-private helper;
  consolidating is low-value.

---

## Registry Conformance

Confirmed the four declarations are present and correctly wired in `all_reduce.py`:

- **`INPUT_TAGGERS = {}`** — empty by design. There is no shape-derived axis: the reduction is always
  the full element-wise sum, and every `INPUT` is tile-aligned by construction (the reduction is a
  tile compute on TILE_LAYOUT). `validate()` still iterates `INPUT_TAGGERS` with the correct
  `tagger(inputs, axes)` signature, so adding a tagger later is a drop-in.
- **`SUPPORTED = {dtype:[bfloat16, float32], layout:[TILE], topology:[Linear]}`** — covers every axis
  the kernels gate on. (`memory_config` DRAM/L1 is accepted but not a gated categorical axis; there is
  no reduce-dim parameter.)
- **`EXCLUSIONS = []`** — present, empty (no in-SUPPORTED cell is refused).
- **`validate()`** — checks structural requirements (MeshDevice, `(1, N)` line, `N ≥ 2`, interleaved,
  16-byte-aligned page, output-spec match) raising `ValueError`, then the axis gate: SUPPORTED per-axis
  (raises `UnsupportedAxisValue`) **then** EXCLUSIONS cell-level (raises `ExcludedCell`). Correct order.
  The public `all_reduce()` calls `validate()` on its first line before any allocation or dispatch.
- **No `INVALID` symbol in the op file** — confirmed absent. INVALID is sourced from the golden
  feature spec.

**No drift.** The verifier CLI reports `xpass_drift = 0`, `supported_fail = 0`, `xfail_wrong_mode = 0`.
No auto-fixes to SUPPORTED were needed — every SUPPORTED cell passes and there is nothing outside
SUPPORTED to promote (SUPPORTED already equals TARGET).

### INVALID audit (`eval/golden_tests/all_reduce/feature_spec.py`)

`INVALID = []`, and this is **correct** for the current TARGET:

- `TARGET = {dtype:[bf16, f32], layout:[TILE], topology:[Linear]}` — every combination is
  constructible (TILE + float dtypes), so there is no structurally-impossible cell to declare.
- **Single-tensor op** — no cross-tensor-axis coupling risk (the canonical authoring mistake).
- **Canonical bf8b + ROW_MAJOR entry is correctly omitted.** That entry only applies when TARGET
  contains ROW_MAJOR / bfloat8_b; this TILE-only float TARGET does not, so an INVALID entry would
  reference axis values the harness never generates. `op_design.md` §"Structural impossibilities"
  documents this explicitly.
- **No norm-like weight axes** → no no-weight canonicalization cells needed.

No changes to `feature_spec.py` are proposed.

### Design conformance

Checked the implementation against `op_design.md` on the binding dimensions — all match:

| Dimension | Design | Implementation |
|---|---|---|
| Algorithm | gather-then-reduce, 2 ordered dispatches, no full-matrix materialize | Phase A fabric gather → Phase B local N-way tile sum ✓ |
| Pipeline topology | Phase A: 2 workers/device (fwd/bwd), NCRISC reader + BRISC writer; Phase B: reader/compute/writer | matches ✓ |
| Parallelization | Phase A 1 worker/direction; Phase B `split_work_to_cores(grid, P)` two-group split | matches ✓ |
| Inter-core comm | ONE op-internal GlobalSemaphore, per-`(device,core)` counting, in-order inc-after-write, `noc_semaphore_set` re-arm | matches ✓ (program-cache test exercises the re-arm) |

### Prompt rules

`eval/prompts/all_reduce.txt` has no `## Rules` (MUST/MUST NOT) section — it carries the
framework-owner guidance. All four numbered mandates are satisfied: (1) per-device programs over the
mesh; (2) op-internal GlobalSemaphore created once + parked + no per-call post-dispatch barrier
(verified in `all_reduce.py` and by the passing program-cache test); (3) host route via
`ccl_dm_route` + `setup_fabric_connection` (no reimplemented packet sizing — the op uses 1:1
page↔packet framing like all_gather, so `ccl_packet_dims` is legitimately unused); (4) kernel-side
fabric egress through `ccl_helpers_dataflow.hpp` with the op-owned ingress/wait/re-arm.

---

## Precision Baseline

Measured on the `(1, 8)` line mesh (N = 8 summands), oracle accumulated in fp32 then cast so the
reference is not itself limited by bf16 rounding. From
`tests/ttnn/unit_tests/operations/all_reduce/test_all_reduce_precision_baseline.py`.

| Shard shape | dtype | PCC | Max Abs Err | Mean Abs Err | Relative RMS Err |
|-------------|-------|-----|-------------|--------------|------------------|
| (1,1,32,32)  | bfloat16 | 0.9999821 | 0.125000 | 0.016816 | 0.008483 |
| (1,1,32,32)  | float32  | 0.9999999 | 0.010442 | 0.001531 | 0.000711 |
| (1,1,64,128) | bfloat16 | 0.9999814 | 0.187500 | 0.017415 | 0.008710 |
| (1,1,64,128) | float32  | 0.9999999 | 0.012442 | 0.001499 | 0.000710 |
| (1,1,256,256)| bfloat16 | 0.9999809 | 0.187500 | 0.017331 | 0.008743 |
| (1,1,256,256)| float32  | 0.9999999 | 0.015086 | 0.001520 | 0.000723 |
| (2,1,32,64)  | bfloat16 | 0.9999806 | 0.187500 | 0.017170 | 0.008715 |
| (2,1,32,64)  | float32  | 0.9999998 | 0.012635 | 0.001531 | 0.000730 |

**Assessment**: Accuracy is excellent and consistent across shapes. bf16 error is dominated by the
bfloat16 storage quantization of the summed output (max-abs ≈ one bf16 ULP at magnitude ~2–4, the
scale of a sum of 8 unit-normal terms; relative RMS ≈ 0.0087 ≈ the bf16 mantissa budget). float32 error
is tiny (rel-RMS ≈ 7e-4), attributable to the on-device HiFi4 + `fp32_dest_acc_en` accumulation not
being bit-identical to torch fp32 — expected and negligible. Every measured PCC is ≥ 0.99998, far
above the golden thresholds.

**Recommended tolerances** (match the golden suite / acceptance test): bf16 `PCC ≥ 0.99`,
float32 `PCC ≥ 0.999`; `atol ≈ 0.2` (bf16) / `0.02` (f32) for allclose on shards summed over 8 devices.
These have generous headroom vs. the observed values.

---

## Verifier CLI Summary

Artifact: `generated/all_reduce_verify/verifier_report.json` (from `python3 -m eval.verify_supported
generated/all_reduce_verify ttnn.operations.all_reduce`). Golden suite: 3 INPUTS × {bf16, f32} × TILE
× Linear = 6 cells.

- supported_pass:     **6**
- xfail_expected:     0   (empty — SUPPORTED already equals TARGET)
- invalid_skipped:    0   (INVALID = [])
- supported_fail:     **0**  ✓ ship gate
- xpass_drift:        **0**  ✓ ship gate
- xfail_wrong_mode:   **0**  ✓ ship gate
- supported_marked_xfail: 0

All loud categories are 0. The report is honest: SUPPORTED describes reality exactly.

**Test suites run (all green on the multi-device sim, aggregate exit 0):**
- Acceptance: `tests/.../all_reduce/test_all_reduce.py` — 10 passed (4 shapes × 2 dtypes +
  program-cache + output_tensor).
- Golden: `eval/golden_tests/all_reduce/test_golden.py` — 6 passed.
- Precision baseline: `test_all_reduce_precision_baseline.py` — 8 passed.

---

## Recommendations

**The refinement queue is empty** — SUPPORTED == TARGET on every axis, so there is no
`(axis, missing_value)` gap to close. Phase 0 already delivers the op's full (deliberately-scoped,
CCL+compute-probe) ambition. `op_requirements.md` records this with no open refinements.

The directions below are **beyond the current TARGET** and are therefore *not* refinements (a
refinement can only move SUPPORTED toward TARGET). They would each require `/golden-tests` to expand
`feature_spec.py`'s TARGET first; listed here so a future TARGET-expansion pass has the map:

- **Ring topology** (`ttnn.Topology.Ring`) — a ring reduce-scatter + all-gather is a fundamentally
  different data pipeline (would be an algorithm-level refinement, not a bundle). The current
  gather-then-reduce is Linear-only by construction.
- **ROW_MAJOR layout** — the reduction is inherently a tile compute; supporting RM input would need a
  tilize-wrapped reader (would map to `/memory-layouts`).
- **bfloat8_b dtype** — block-float on the summands; the gather path is byte-verbatim so it would move,
  but the reduce compute-config and CB formats would need the dtype treatment (would map to
  `/numeric-formats-metal`).
- **Multi-link / worker-mux fabric** — the helper already exposes `MuxConn<N>`; the op currently uses
  a single link (`_LINK_IDX = 0`) with one worker per direction. Higher-bandwidth links are a
  throughput lever with no correctness gap today.
- **Sharded input** (`validate()` currently rejects sharded with a `ValueError`) — a memory-config
  expansion needing a sharded reader/writer.

None of these has a failing golden cell or an under-claimed SUPPORTED axis today, so none is filed as
a refinement.
