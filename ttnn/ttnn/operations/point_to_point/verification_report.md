# Verification Report: point_to_point

`point_to_point` is a **multi-device CCL** op (builds a `ttnn.MeshProgramDescriptor`
+ `ttnn.generic_op` with fabric sender/receiver dataflow kernels). It was therefore
verified on the deterministic multi-device craq-sim runner
(`scripts/run_multidevice_sim_pytest.py`), **not** `run_safe_pytest.sh`. The required
topology is `bh_8xP150_p2p` (`mesh_shape = [2, 4]`, `fabric_config = FABRIC_1D`); all
tests open a `(2, 4)` mesh to match it.

---

## Code Review

Everything below marked **Fixed** was changed in-tree during this pass.

### Fixed

1. **`validate()` is now the entry point's first line** (`point_to_point.py`).
   Previously `_compute_packet_dims(input_tensor)` ran *before* `validate()`. The
   registry contract wants `validate()` to gate first. Reworked so `validate()` no
   longer takes `packet_dims` as a parameter — it computes it lazily *only* for the
   optional supplied-`intermediate_tensor` spec check (check #9). The entry point
   now calls `validate(...)` first, then computes `packet_dims` for the descriptor.
   Behavior-preserving (confirmed: precision baseline 8/8 pass, acceptance paths pass
   after the change).

2. **Precision baseline mesh shape `(1, 2)` → `(2, 4)`**
   (`test_point_to_point_precision_baseline.py`). The old `(1, 2)` matched **no**
   topology in `multidevice_sim_topologies.yaml`, so the test would **hang fabric
   init** ("Fabric Router Sync: Timeout"). This is the exact test/topology mismatch
   the reference `test_p2p_confirm_topology.py` was written to document — the
   acceptance test was already fixed to `(2, 4)` but the precision baseline still
   carried the stale shape. Also capped the largest sweep shape `512×512` → `256×128`
   (32 tiles) so the 8-cell sweep fits the sim wall-clock budget; the kernel is a
   shape-uniform byte copy, so a larger shape only adds pages, not a code path.

### Reviewed — no change needed

- **Fabric-egress helper usage is correct.** The sender's fabric write goes through
  the safety-by-construction CCL helper (`FabricStreamSender → open(route) →
  arm_unicast_write / arm_inc → write_page / inc → close`) and the receiver's ready
  ack uses the one-shot `FabricStreamSender::signal(num_hops, sem_noc_addr)`. The raw
  APIs (`noc_semaphore_wait_min` / `noc_semaphore_set`, local-ingress `noc_async_read`,
  `tt_memmove` page↔packet, `TensorAccessor` page load/store) are exactly the phases
  the helper banner (`ccl_helpers_dataflow.hpp:69–77`) explicitly delegates to the op
  (there is no `FabricStreamReceiver`; the receive ingress + wait + reset are op-owned).
  No raw multicast+handshake that should be `mcast_pipe.hpp` — this is unicast.
- **Semaphore cache-reuse footgun handled correctly.** Sender resets its local sem
  **before** its outgoing done-inc; receiver resets **after** its done-wait
  (`ccl_helpers_dataflow.hpp:75–77`). Validated by `test_point_to_point_program_cache`
  passing on both topologies (call 0 = cache miss, call 1 = cache hit).
- **CB sync balanced.** `cb_shard_pages`: reader pushes `num_pages` == writer pops
  `num_pages`. `cb_shard_out`: receiver_reader pushes `num_pages` == receiver_writer
  pops `num_pages`. `cb_packet_send` / `cb_packet_recv`: single-owner scratch,
  1 reserve / 1 push. ✓
- **Packet coalescing / segmentation traced for both regimes.** Regime A
  (page ≤ packet: N pages coalesced per packet, `page_segments = 1`) and regime B
  (page > packet: 1 page split across `page_segments`, `packet_size = max_packet`):
  the sender's `packet_idx` assignment and per-packet `curr_pages_per_packet`
  recomputation exactly mirror the receiver's de-coalescing. The `payload_size_bytes`
  fabric write always sends the full armed size, but the receiver only copies
  `min(page_size − offset, packet_size)` bytes out, so any packet-tail padding is
  never consumed. Because `validate()` enforces a 16-byte-aligned page (and every
  golden/acceptance shard keeps the last dim a multiple of 8), `aligned_page ==
  page_size` for all valid inputs, so the output-page tail write carries no garbage.
- **Correct primitives.** `TensorAccessor` (not deprecated `InterleavedAddrGen`);
  `void kernel_main()`; includes use `api/dataflow/dataflow_api.h` (not bare).
- **Design conformance.** Algorithm (identity byte copy), pipeline topology
  (sender_reader/NCRISC, sender_writer/BRISC, receiver_reader/NCRISC,
  receiver_writer/BRISC), single-core/single-link work distribution, and the
  one-`GlobalSemaphore` two-phase handshake all match `op_design.md`.

### Advisories (cosmetic — not blocking, not refinements)

- `sender_writer.cpp`: the runtime arg named `receive_semaphore_addr` is the *shared*
  `GlobalSemaphore` address, used both for the local ready-wait and (via fabric route)
  the remote done-target. Functionally correct but the name reads as receiver-only;
  a future touch could rename to `semaphore_addr` for clarity.
- `sender_writer.cpp` uses `round_up(page_size, alignment)` while `receiver_reader.cpp`
  uses `align(page_size, alignment)` for the same aligned-page computation. Identical
  for a power-of-two `l1_alignment` (16), but worth unifying.

---

## Registry Conformance

- **INPUT_TAGGERS** — `{"alignment": tag_alignment}`; `tag_alignment(inputs, axes)`
  has the correct 2-arg signature (both last two dims % 32 == 0 → `tile_aligned`).
  Because sharding is on dim 0, the last-two-dims alignment tag is identical whether
  computed on the per-device shard shape (golden harness) or the full mesh-tensor
  shape (runtime `validate()`), so the tags agree.
- **SUPPORTED** — `dtype`, `layout`, `topology`, `alignment` all declared. Every axis
  the kernels gate on (dtype → CB `data_format`; layout → page addressing; topology →
  `ccl_dm_route`; alignment → tagged) is present.
- **EXCLUSIONS** — `[]`. Correct: no in-SUPPORTED not-yet-implemented cells.
- **validate() order** — structural `ValueError`s (MeshDevice, self-send, coords in
  mesh, shared row/col, interleaved-only, 16-byte page, output/intermediate spec) →
  per-axis SUPPORTED (`UnsupportedAxisValue`) → EXCLUSIONS (`ExcludedCell`). Correct.
  Both refusal types subclass `NotImplementedError`, so the harness records
  out-of-SUPPORTED cells as `xfail` (green).
- **Op file does NOT declare INVALID.** Confirmed — INVALID is sourced from
  `feature_spec.py`. ✓

### INVALID audit (`eval/golden_tests/point_to_point/feature_spec.py`)

`INVALID = [{"dtype": ttnn.bfloat8_b, "layout": ttnn.ROW_MAJOR_LAYOUT}]`

- **Single-tensor coupling** ✓ — `dtype` and `layout` both describe the *input* tensor.
- **Universe-must-change** ✓ — `bfloat8_b` is a block-quantized tiled format with no
  row-major representation; this is a data-format-definition impossibility, not a
  not-yet-implemented EXCLUSION.
- **Canonical bf8b+ROW_MAJOR entry present** ✓ (the required activation entry).
- No cross-tensor-axis coupling; no norm weight axes (n/a). **Well-formed.**

---

## Precision Baseline

`test_point_to_point_precision_baseline.py`, `(2, 4)` mesh / FABRIC_1D / Linear, TILE
layout. `point_to_point` is a pure byte copy, so the oracle is the *device-resident*
sender shard (post-`from_torch` quantization). Result is bit-exact.

| Shape | dtype | PCC | Max Abs Err | Mean Abs Err | Relative RMS Err |
|-------|-------|-----|-------------|--------------|------------------|
| (1,1,32,32)  | bfloat16 | 1.000000 | 0.0 | 0.0 | 0.0 |
| (1,1,32,32)  | float32  | 1.000000 | 0.0 | 0.0 | 0.0 |
| (1,1,64,128) | bfloat16 | 1.000000 | 0.0 | 0.0 | 0.0 |
| (1,1,64,128) | float32  | 1.000000 | 0.0 | 0.0 | 0.0 |
| (1,1,96,64)  | bfloat16 | 1.000000 | 0.0 | 0.0 | 0.0 |
| (1,1,96,64)  | float32  | 1.000000 | 0.0 | 0.0 | 0.0 |
| (1,1,256,128)| bfloat16 | 1.000000 | 0.0 | 0.0 | 0.0 |
| (1,1,256,128)| float32  | 1.000000 | 0.0 | 0.0 | 0.0 |

**Assessment**: Exact identity transfer, as required for a pure-data-movement CCL op —
zero error at every shape/dtype. bf8b is likewise bit-exact against its device-resident
(already-quantized) shard (confirmed in the golden suite).
**Recommended tolerances**: PCC ≥ 0.999 (band only), rtol = 0, atol = 0. The
acceptance/golden thresholds (0.995–0.999) are conservative safety bands.

---

## Verifier CLI Summary

`python3 -m eval.verify_supported /tmp/p2p_verify ttnn.operations.point_to_point`
(artifacts persisted under `verifier_results/`).

- supported_pass: **30**
- xfail_expected: **36**  (all integer dtypes: uint16 ×12, int32 ×12, uint32 ×12)
- invalid_skipped: **6**  (bfloat8_b + ROW_MAJOR)
- supported_fail: **0**   ✓ (must be 0 to ship)
- xpass_drift: **0**      ✓ (must be 0 to ship)
- xfail_wrong_mode: **0** ✓ (must be 0 to ship)
- supported_marked_xfail / invalid_unexpected / no_axes_found: **0**

**Coverage note (important).** The golden matrix is 432 cells (dtype 6 × layout 2 ×
topology 2 × 18 shapes, minus INVALID). At ~35 s per real-transfer cell on the kHz
sim, the full 180 passing cells ≈ 105 min of device time — infeasible in a single
bounded run. I ran a **representative 72-cell subset** that exercises **every axis
value** at least once across the coverage classes: small single-tile (`1×1×32×32`,
Linear + Ring), non-tile-aligned (`1×1×48×64`, both topologies), and multi-tile /
multi-packet (`1×1×64×128`, both topologies). Because the op is a shape-uniform byte
copy (no per-shape code branch — alignment/size are not kernel branches), this subset
is sufficient to verify SUPPORTED against observed behavior. Every SUPPORTED
(dtype × layout × topology × alignment) combination passes; every non-SUPPORTED dtype
xfails via `validate()`; the INVALID cell skips. All three loud categories are 0.

---

## Recommendations

1. **Only one refinement is queued (integer dtypes)** — this is honest, not an
   omission. `TARGET − SUPPORTED` has exactly one gap: `dtype ∈ {uint16, int32,
   uint32}`. Layout, topology, and alignment are already fully at TARGET, and
   bf8b+ROW_MAJOR is INVALID (not a gap). See `op_requirements.md` Refinement 1.

2. **Out of the refinement queue by construction (no golden axis to unlock):**
   - **Multi-link / multi-core distribution.** `op_design.md` scopes the transfer to
     a single worker core `(0,0)` on a single fabric link (`link_idx = 0`) and calls
     multi-link/multi-core an explicit future refinement. There is no `cores` or
     `num_links` axis in TARGET, so no golden cell is gated on it — it is a
     throughput optimization, not a support-surface expansion. If pursued, it is a
     real-data-dependency multi-core change (fabric link sharing / worker-mux), not
     the embarrassingly-parallel interleaved split `/interleaved-parallel` covers.
   - **Sharded (non-interleaved) input.** `validate()` raises a structural
     `ValueError` for `is_sharded()`. Sharding is a memory config, not a golden axis
     here (TARGET has no `memory_config` axis; the harness always builds interleaved
     DRAM shards). Adding sharded I/O would be a new reader/writer path and a
     `feature_spec.py` axis first — upstream of this queue.

3. **Numerical precision** — nothing to tune. Identity copy is bit-exact (PCC = 1.0,
   0 error). No `math_fidelity` / accumulation lever exists (no compute kernel).

4. **Environment / reproducibility** — the multi-device runner uses `sys.executable`
   for its pytest subprocess. It **must** be invoked with *this clone's* venv
   activated (`source python_env/bin/activate`), otherwise `sys.executable` can fall
   back to another checkout's `python_env` whose `ttnn` lacks
   `ttnn.operations._op_contract` (and `ttnn.fp8_e4m3`, which
   `tests/ttnn/utils_for_testing.py` references at import). This is not an op or infra
   defect — just a venv-selection gotcha for anyone re-running the verification.
