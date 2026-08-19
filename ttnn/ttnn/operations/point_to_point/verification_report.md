# Verification Report: point_to_point

**Op class**: multi-device CCL, **dataflow-only** (no compute kernel) — copies one mesh device's
interleaved shard to another over the TT-Fabric.
**Verified on**: real silicon — a 4-chip Blackhole QuietBox, mesh `(1, 4)`, `FabricConfig.FABRIC_1D`
(topology `bh_quietbox_1x4_hw`, `grade_primary` for `runtime: hardware`), driven exclusively through

```
scripts/run_multidevice_sim_pytest.py --op point_to_point --runtime hardware -- <target>
```

`run_safe_pytest.sh` is the wrong runner for a CCL op (no multichip/hang awareness, forces slow
dispatch under sim). `--list` confirms `point_to_point` is in the matrix, so this is the deterministic
gate. **Aggregate exit = 0** on every final run below.

---

## Code Review

### Fixed

**1. `ccl_packet_dims` sized the fabric payload with the *channel-buffer* size — a silent
out-of-bounds write into ethernet-core L1. (The real defect this pass found.)**

`ttnn/cpp/ttnn/operations/ccl/common/host/ccl_helpers_dataflow_host.hpp:75` capped a packet at
`get_tt_fabric_channel_buffer_size_bytes()`. But a channel buffer is
`packet_header_size + max_payload_size` (`tt_metal/fabric/fabric_context.cpp:159`), and the worker
writes the header at the channel slot's base and the **payload at `slot_base +
sizeof(PACKET_HEADER_TYPE)`** (`edm_fabric_worker_adapters.hpp:685,708`). Sizing a payload at the
channel-buffer size therefore overruns the slot by exactly the header size — 48 B on Blackhole
(`max_payload=4352`, `header=48`, `channel_buffer=4400`) — into the next slot's header, or past the
end of the channel region for the last slot. The only in-kernel guard,
`ASSERT(size_bytes <= this->buffer_size_bytes)` (`:713`), omits the header **and compiles out in
Release**, so the overrun is completely silent.

Measured reachability before the fix (from the widened `test_packet_geometry_within_edm_slot` sweep):

| cell | page | framing | packet |
|---|---|---|---|
| **golden** `(1,1,56,88)` `uint16` `ROW_MAJOR` | 176 B | regime A, 25 pages/packet | **4400 B** ✗ |
| `(1,1,8,2048)` `float32`/`int32`/`uint32` `ROW_MAJOR` | 8192 B | regime B, 2 segments | **4400 B** ✗ |
| `(1,1,8,4096)` `uint16` `ROW_MAJOR` | 8192 B | regime B, 2 segments | **4400 B** ✗ |

i.e. **a cell in the graded golden cartesian already triggered it** and "passed" while writing 48 B
out of bounds. `bfloat16` was accidentally safe only because its `std::bit_floor` special case lands
on 4096.

Fixed at the source — `get_tt_fabric_channel_buffer_size_bytes()` →
`get_tt_fabric_max_payload_size_bytes()` — rather than worked around in the op, because:
* the prompt's framework-owner guidance mandates the op *consume* `ccl_packet_dims` and explicitly
  forbids reimplementing the packet sizing, so a clamp in the op would have had to re-derive
  `pages_per_packet` / `page_segments` / `total_packets` (exactly the duplication the mandate bans);
* the same helper backs the pre-existing bound C++ `ttnn.point_to_point` (the body is "moved verbatim
  from point_to_point `detail::compute_aligned_packet_dims`"), so the bug is fixed for both;
* the change strictly *shrinks* packets to a legal size, so it cannot break correctness.

Behaviour delta is minimal: `bfloat16` (4096) and every tile-page case are unchanged; the offending
cells move to 4224 B (regime A) / 4352 B (regime B). Rebuilt (`ninja install`, 11 edges) and re-ran
the full suite — 396/396 golden, 136 passed in the op directory, zero offenders in the geometry sweep.

**2. The op-internal `GlobalSemaphore` cache was keyed by `id(mesh_device)` in a module-level dict.**
(`point_to_point.py`, as prescribed by `op_design.md` § "Semaphore lifecycle" — deviated
deliberately.) That is unsound twice over: the dict **outlives the device**, pinning a closed
device's L1 allocation for the life of the process (an unbounded leak in a suite that opens a mesh
per test — the golden run does 396 of them), and **CPython recycles `id()`s**, so a new `MeshDevice`
allocated at a freed one's address silently inherits the *previous* device's semaphore — a dangling
handle that is then parked in `mpd.semaphores` and baked into both kernels' runtime args, where the
kernels atomic-inc and zero it. The implementer's own probe flagged the risk but concluded it was
safe from 4 observed-unique ids.

Fixed by binding the handle as an attribute **on the mesh-device object** (`MeshDevice` accepts
dynamic attributes — the root `conftest.py` already does this for `cache_entries_counter`), so the
semaphore's lifetime is exactly the device's, with the `id()`-keyed dict retained only as an
unreachable fallback should that ever stop being true. `test_point_to_point_debug.py`'s
`test_semaphore_is_per_live_device` now guards both halves of the contract (address stable within a
device; handle stored on the device, fallback dict unused). Exercised by 396 open/close cycles.

**3. `_same_row_or_column` raised `IndexError` on a 1-D mesh coordinate.** It indexed `[0]` and `[1]`
unconditionally, so a 1-D `MeshCoordinate` would crash the validator instead of validating. Now
returns `True` for rank-1 coords (trivially the single row) before indexing.

**4. Dead `geom["aligned_page_size"]`** in `point_to_point_program_descriptor.py` — computed and
never consumed (the kernels re-derive the intra-packet stride from the `alignment` compile-time arg).
Removed.

**5. `test_point_to_point_precision_baseline.py` opened a `(1, 2)` mesh and ERRORed on every case.**
Fabric init timed out — `Fabric Router Sync: Timeout after 10000 ms ... Ethernet handshake likely
failed` — the classic test/topology mismatch: the graded topology is `(1, 4)`. Repinned to
`MESH_SHAPE = (1, 4)` (same contract comment as the acceptance suite), added `bfloat8_b` to the dtype
sweep, and strengthened the oracle from PCC-only to **bit-exactness** (`max_abs == 0`,
`rel_rms == 0`) — a tolerance-only assertion lets a single corrupted page through on a large shard,
which is precisely what the framing/alignment logic can produce. 12/12 pass.

**6. `test_point_to_point_debug.py`: 267 cases → 28, and one control removed.**
 * The **C++-reference control** (`test_cpp_reference_op_fixture_cycle`, 120 cases) dispatched the
   bound `ttnn.point_to_point` from inside the generated op's own test directory. The generation
   mandate says to treat that op as if it does not exist; its diagnostic conclusion is preserved in
   the file's docstring instead. Deleted.
 * `test_hop_count_stress` 20 reps → 2, fixture-cycle control 16 → 3 (all measured green at full
   count; kept as a standing canary at a fraction of the runtime).
 * The `id(mesh_device)` reuse probe was retargeted as the guard for fix 2.
 * **`test_packet_geometry_within_edm_slot` was widened from 5 (dtype, layout) pairs to all 11 valid
   ones.** It had covered only `bfloat16`/`float32`/`bfloat8_b` — and `bfloat16`'s `bit_floor` makes
   it accidentally safe — which is exactly why defect 1 went unnoticed. This widened sweep is what
   caught it.

**7. `black` (repo config, line-length 120)** applied to the op module and the test files this pass
owns. `point_to_point.py` was already non-conforming before these edits.

### Reviewed and intentionally left as-is

* **Fabric egress goes entirely through the mandated kernel helper.** The sender writer builds
  `FabricStreamSender<>` → `.open(unicast_route(num_hops))` → `arm_unicast_write(packet_size)` /
  `arm_inc(1)` → `write_page` / `inc` → `close()`; the receiver reader uses the one-shot
  `signal(num_hops, sem_noc_addr)` for its ready ack. That is exactly the safety-by-construction
  typestate progression in `ccl_helpers_dataflow.hpp`. There is **no** raw
  `noc_async_write_multicast` + hand-rolled `noc_semaphore_set/wait/inc` anywhere, so nothing to
  migrate to `mcast_pipe.hpp`.
* **The raw NoC that remains is exactly what the helper banner says it does not own**: the local
  DRAM↔L1 page moves (address generation via `TensorAccessor`, "consumed, never re-wrapped"), the
  page↔packet coalescing/segmentation, the *waiting* half of the handshake
  (`noc_semaphore_wait_min`), and the cache-reuse re-arm (`noc_semaphore_set(sem, 0)`) — with the
  correct ordering: **sender resets before its own outgoing inc, receiver after its wait**. The
  handshake is sound across program-cache hits in both directions (verified by reasoning through
  invocations k / k+1, and by `test_point_to_point_program_cache`).
* **CB synchronisation ledger balances.** `cb_shard_pages`: reader pushes `num_pages`, writer
  waits/pops `num_pages`. `cb_output_pages`: reader pushes `num_pages`, writer waits/pops
  `num_pages`. `cb_packet_staging` / `cb_packet_landing` are reserve-once scratch (0 pushes / 0
  waits) whose producer and consumer are the *same* kernel, so there is no handshake to balance —
  depth 1 is justified because `write_page` copies the payload into the fabric channel buffer under
  flow control before returning.
* **`TensorAccessor` with the 2-argument constructor in all four kernels** (never the deprecated
  `InterleavedAddrGen`, never a runtime page-size override that would set the per-bank stride to the
  unaligned logical page). `void kernel_main()`, and `api/dataflow/dataflow_api.h` (not bare
  `dataflow_api.h`) includes. `kernel_lib` includes match the `all_gather` / `all_reduce` convention.
* **`_dram_slot_stride`** — sizing the two DRAM-facing CBs at `round_up(page_size, 64)` instead of the
  design's `round_up(page_size, l1_alignment)` is an advisory deviation the implementer documented and
  it is **correctness-required** on Blackhole (a CB's page size is its slot stride, and the NoC
  requires `(l1_addr & mask) == (dram_addr & mask)` with a 64 B DRAM read alignment). Kept, verified
  by the 96 B / 48 B row-major cases.
* **`ttnn.clone(input_tensor)` as the default output.** The programs write only the receiver device's
  shard, so seeding the output is what makes "every other device's shard is unchanged" a total
  statement instead of an undefined one — the acceptance suite asserts exactly that. Contract-required.
* **The `GlobalSemaphore` spans the whole worker grid though only core `(0, 0)` is used.** Harmless
  (a 4 B per-core allocation) and matches the reference CCL pattern; the address is mesh-wide either
  way, which is what makes `get_noc_addr(sem_addr)` name the same semaphore on the routed-to chip.

### Advisory (no fix — no failing cell to point at)

* **Per-page NoC serialization.** The sender reader and receiver writer each issue one transfer and
  then barrier immediately (`noc_async_read` + `noc_async_read_barrier` per page;
  `noc_async_write` + `noc_async_write_barrier` per page). Correct, and the depth-2 CB still overlaps
  the reader against the framer *across* kernels, but within each kernel there is no DRAM-latency
  overlap. Batching the issues (or double-issuing into the two CB slots before barriering) would
  pipeline it. Performance only.
* **The default output path costs a full-mesh tensor copy per call.** `ttnn.clone` runs on *every*
  mesh device, including the N−2 that do not participate. Contract-required for the default path;
  a caller on a hot path should pass `output_tensor=` (that path is tested and skips the clone).
* **The staging tensor is re-allocated per call** when not supplied, and allocated on every mesh
  device though only two use it. DRAM, cheap, but a caller in a loop should hoist it via
  `intermediate_tensor=` (now covered by `test_point_to_point_extended.py`).
* **Single link, single core.** `_LINK_IDX = 0` and one worker core per participating device — the
  design's explicit choice (multi-link needs the `MuxConn<N>` mux policy). Bandwidth, not correctness.
* **`Ring` never routes the long way on the graded topology.** With `FABRIC_1D` and the adjacent
  coordinate pair the golden driver pins, `Topology.Ring` resolves to the same 1-hop route as
  `Topology.Linear`, so the axis is exercised as "short way == line way". A true wraparound needs
  `FabricConfig.FABRIC_1D_RING`, which no topology in `scripts/multidevice_sim_topologies.yaml`
  provides. Not a `SUPPORTED` gap (the value is accepted, routed and verified) and not a refinement —
  a *harness* limitation, recorded under Recommendations.

### Design conformance (`op_design.md`)

| Binding dimension | Design | Implementation |
|---|---|---|
| Algorithm | pure byte copy, no arithmetic, no compute kernel | ✓ dataflow-only, 4 kernels |
| Data pipeline topology | sender NCRISC read → BRISC frame+fabric; receiver NCRISC ack/wait/read-back/de-frame → BRISC write | ✓ exact, incl. RISC ownership |
| Parallelization | one core `(0,0)` on each of the two participating devices, one link; no program elsewhere | ✓ two `MeshCoordinateRange` entries only |
| Inter-core communication | one op-internal `GlobalSemaphore`, receiver-ready → sender-done, in-order inc-after-write, re-arm ordering | ✓ |
| Packet framing | both regimes (A coalesce / B segment) via `ccl_packet_dims` | ✓ both implemented; **regime B was untested until this pass** (see Recommendations) |
| CBs | 4 CBs, indices/depths/formats as tabled | ✓ (one advisory deviation: DRAM-aligned slot stride, above) |
| Validation | 8 structural `ValueError` checks then the axis gate | ✓ same order |

Two documented deviations, both deliberate and both improvements: the DRAM-aligned CB slot stride
(implementer, correctness) and the semaphore cache key (this pass, correctness).

### Prompt rules (`eval/prompts/point_to_point.txt`)

No `## Rules` (MUST / MUST NOT) section — the prompt carries the framework-owner guidance instead.
All four numbered mandates hold:

1. **Two programs, two devices.** `MeshProgramDescriptor` with exactly two
   `(MeshCoordinateRange, ProgramDescriptor)` entries, one per participating coordinate. ✓
2. **Cross-device `GlobalSemaphore`, op-internal, created once, parked on the descriptor, no per-call
   post-dispatch barrier.** ✓ — and now correctly scoped to the *live* device (fix 2). One
   `synchronize_device` at creation only; `mpd.semaphores = [sem]`; no trailing barrier.
3. **Host route + packet framing via the bound helpers.** `ccl_packet_dims`, `ccl_dm_route`,
   `ttnn.setup_fabric_connection`. ✓ — nothing reimplemented; the packet-size defect was fixed *in*
   the helper precisely so the op would not have to duplicate the framing rule.
4. **Kernel-side egress via `ccl_helpers_dataflow.hpp`**, with the op owning the ingress read, the
   `noc_semaphore_wait_min`, the re-arm, and the framing. ✓ including the reset-ordering footgun.

### Refuted: the changelog's "platform limitation"

Phase 0's changelog reports a *"multi-packet / multi-hop ethernet wedge"* — that after a transfer of
more than one packet or over more than one hop, the next `open_mesh_device` dies in
`RiscFirmwareInitializer::assert_active_ethernet_cores_to_reset` ("Timed out while waiting for active
ethernet core 24-25 to become active again") — and therefore that **every acceptance case needs its
own pytest process plus a board reset**. That drove a 90-process driver script.

**It does not reproduce.** Every run in this report used a single pytest process with the stock
function-scoped `mesh_device` fixture, i.e. one fabric mesh open/close *per case*:

| suite | cases | result |
|---|---|---|
| acceptance `test_point_to_point.py` | 90 | **90 passed, 61 s** |
| golden cartesian `test_golden.py` | 432 | **396 passed + 36 INVALID-skipped, 143 s** |
| `test_hop_count_stress` (1/2/3 hops × 2 payloads × 20 reps) | 120 | **120 passed** |
| whole op directory | 143 | **136 passed, 7 skipped** |

That is ~600 consecutive fabric mesh open/close cycles with real multi-hop, multi-packet traffic and
no wedge. The most likely explanation is that the wedge *was* the NoC-alignment fault diagnosed
alongside it: the watcher's NoC sanitizer **halts** the offending core, which wedges the ethernet, and
the `_dram_slot_stride` fix removed the fault. The trimmed guards in `test_point_to_point_debug.py`
stay as the canary. Practical consequence: the op needs no special per-case process isolation.

---

## Registry Conformance

Confirmed all four declarations present and correctly wired in `ttnn/ttnn/operations/point_to_point/`:

* **`INPUT_TAGGERS = {"alignment": tag_alignment}`** — one shape-derived axis; signature is
  `(inputs, axes)` (verified by `inspect.signature`); `validate()` iterates the dict and feeds it
  `(tuple(input_tensor.shape),)`, matching the golden harness's 1-tuple convention.
* **`SUPPORTED`** — `dtype` (6), `layout` (2), `topology` (2), `alignment` (2). Covers **every** axis
  the op gates on and every `INPUT_TAGGERS` key. Machine-checked: `SUPPORTED.keys() ==
  TARGET.keys()`, and `TARGET[axis] - SUPPORTED[axis]` is empty for all four axes.
* **`EXCLUSIONS = []`** — present and empty; no in-`SUPPORTED` cell is refused, and the golden run
  proves none needs to be.
* **`validate()`** — 8 structural checks raising `ValueError` (MeshDevice; no self-send; both coords
  in-mesh and sharing a row or column; interleaved-only; rank ≥ 2; 16 B page alignment;
  `output_tensor` spec; `intermediate_tensor` spec) **then** the axis gate: per-axis `SUPPORTED`
  (raises `UnsupportedAxisValue`) **then** cell-level `EXCLUSIONS` (raises `ExcludedCell`), both from
  `ttnn.operations._op_contract`. Correct order — structural misuse can never be mistaken for a
  support refusal. The public `point_to_point()` calls `validate()` on its **first line**, before any
  clone/allocation/dispatch.
* **No `INVALID` symbol in the op file** — confirmed absent (grepped). INVALID is sourced from the
  feature spec.

**No drift.** `xpass_drift = 0`, `supported_fail = 0`, `xfail_wrong_mode = 0`,
`supported_marked_xfail = 0`. **No auto-fixes to `SUPPORTED` were needed** — every cell in it passes,
and there is nothing outside it to promote, because `SUPPORTED == TARGET` on every axis.

### INVALID audit (`eval/golden_tests/point_to_point/feature_spec.py`)

```python
INVALID = [{"dtype": ttnn.bfloat8_b, "layout": ttnn.ROW_MAJOR_LAYOUT}]
```

Well-formed against all three sanity rules, no changes proposed:

* **Single-tensor coupling** ✓ — `dtype` and `layout` both describe the one input tensor. This is a
  single-input op, so the canonical cross-tensor-axis authoring mistake is structurally impossible
  here.
* **Universe-must-change** ✓ — `bfloat8_b` is a block-quantized *tiled* format with no row-major
  representation. A data-format definition impossibility, not a not-yet-implemented gap, so it is
  correctly INVALID rather than an `EXCLUSIONS` entry.
* **Canonicalization-only multi-axis exception** — n/a (not a norm-like op; no weight axes, so no
  no-weight canonicalization cells are needed).
* **The canonical `bfloat8_b` + `ROW_MAJOR` entry for a tile-or-RM activation is present** ✓ — and it
  is load-bearing: it skips 36 of the 432 cells, and the harness confirms exactly 36
  `invalid_skipped`.
* `topology` is correctly *not* coupled to `dtype`/`layout` (orthogonal), and the 16 B page-size gate
  is correctly left as a `validate()` shape×dtype check rather than modelled as an axis — every
  `INPUTS` shard keeps it satisfiable (last dim a multiple of 8).

---

## Precision Baseline

From `tests/ttnn/unit_tests/operations/point_to_point/test_point_to_point_precision_baseline.py`
(12/12 pass), mesh `(1, 4)`, `TILE_LAYOUT`, `Topology.Linear`, sender `(0,0)` → receiver `(0,1)`.
The reference is the **device-resident sender shard** (post-`from_torch` quantization), because the
op's oracle is identity — not the original torch tensor.

| Shape | dtype | PCC | Max Abs Err | Mean Abs Err | Relative RMS Err |
|-------|-------|-----|-------------|--------------|------------------|
| (1,1,32,32)   | bfloat16  | 1.0000000 | 0.0 | 0.0 | 0.0 |
| (1,1,32,32)   | float32   | 1.0000000 | 0.0 | 0.0 | 0.0 |
| (1,1,32,32)   | bfloat8_b | 1.0000000 | 0.0 | 0.0 | 0.0 |
| (1,1,64,128)  | bfloat16  | 1.0000000 | 0.0 | 0.0 | 0.0 |
| (1,1,64,128)  | float32   | 1.0000000 | 0.0 | 0.0 | 0.0 |
| (1,1,64,128)  | bfloat8_b | 1.0000000 | 0.0 | 0.0 | 0.0 |
| (1,1,96,64)   | bfloat16  | 1.0000000 | 0.0 | 0.0 | 0.0 |
| (1,1,96,64)   | float32   | 1.0000000 | 0.0 | 0.0 | 0.0 |
| (1,1,96,64)   | bfloat8_b | 1.0000000 | 0.0 | 0.0 | 0.0 |
| (1,1,512,512) | bfloat16  | 1.0000000 | 0.0 | 0.0 | 0.0 |
| (1,1,512,512) | float32   | 1.0000000 | 0.0 | 0.0 | 0.0 |
| (1,1,512,512) | bfloat8_b | 1.0000000 | 0.0 | 0.0 | 0.0 |

**Assessment**: **bit-exact**, not merely accurate. Every error metric is identically zero for every
shape and every dtype, including `bfloat8_b` (whose block-float bytes round-trip verbatim) — which is
the correct result for a pure data-movement op that performs no arithmetic: there is no accumulation
order, no `math_fidelity`, and no dest-accumulate precision to trade off. The same exactness holds in
the wider suites: `test_point_to_point_debug.py`'s deterministic payloads (all-ones, monotonic,
row/column position encoding) and `test_point_to_point_extended.py` all assert
`max|actual − expected| == 0` on the receiver shard **and** on every untouched shard.

**Recommended tolerances**: the honest assertion is **exact equality** (`atol = rtol = 0`), and that
is what the precision baseline, the debug guards and the extended suite now assert. The
tolerance-based thresholds the acceptance suite (`PCC ≥ 0.995`/`0.999`) and the golden helper
(`tolerance = (0.999, 0.02)`) carry are safety bands inherited from the arithmetic-op convention;
they are met with unlimited margin. Anything less than exact for this op is a bug, so **new tests for
`point_to_point` should assert exactness** rather than a PCC floor — a PCC threshold smears a
single corrupted page out of sight on a large shard, which is exactly the failure mode the framing
and page-stride logic produces.

---

## Verifier CLI Summary

Artifact: `generated/p2p_verify/verifier_report.json`
(`python3 -m eval.verify_supported generated/p2p_verify ttnn.operations.point_to_point`).
Golden suite: 18 `INPUTS` × 6 `dtype` × 2 `layout` × 2 `topology` = 432 cells (`alignment` is tagged
from the shape, not enumerated).

* supported_pass:        **396**
* xfail_expected:        0   (empty — `SUPPORTED` already equals `TARGET` on every axis)
* invalid_skipped:       **36**   (the `bfloat8_b` + `ROW_MAJOR` INVALID entry)
* supported_fail:        **0**  ✓ ship gate
* xpass_drift:           **0**  ✓ ship gate
* xfail_wrong_mode:      **0**  ✓ ship gate
* supported_marked_xfail: 0
* no_axes_found:          0

All loud categories are 0 and every cell is accounted for (396 + 36 = 432). The report is honest:
`SUPPORTED` describes reality exactly.

**Every suite, final state (aggregate exit 0):**

| suite | cases | result |
|---|---|---|
| Acceptance — `test_point_to_point.py` | 90 | 90 passed |
| Golden — `eval/golden_tests/point_to_point/test_golden.py` | 432 | 396 passed, 36 skipped (INVALID) |
| Precision baseline — `test_point_to_point_precision_baseline.py` | 12 | 12 passed |
| Extended — `test_point_to_point_extended.py` | 5 | 5 passed |
| Regression guards — `test_point_to_point_debug.py` | 28 | 28 passed |
| Whole op directory (all files) | 143 | 136 passed, 7 skipped |

The 7 skips are pre-existing throwaway sim-topology confirmation tests (`test_moe_p2p.py`,
`test_p2p_confirm_galaxy.py`, `test_p2p_confirm_topology.py`, `test_p2p_ring_confirm.py`) that request
8 or 32 devices and cannot open on a 4-chip box. They are not this op's artifacts and were left alone.

---

## Recommendations

### The refinement queue is empty — and that is the honest result

`TARGET[axis] − SUPPORTED[axis]` is **empty for all four axes** (machine-checked), `EXCLUSIONS` is
empty, and `xfail_expected` in `verifier_report.json` contains **zero** entries. Per the registry
model a refinement must either add a value to `SUPPORTED[axis]` or move named failing cells out of a
non-trivial failure category, and there is neither: no `OOM`, no `numerical-precision`, no
`numerical-bug`, no `hang`. Phase 0 delivers the op's entire declared ambition.
`op_requirements.md` records this with no open refinements.

### Coverage caveats a future pass should keep in mind

* **`Topology.Ring` is exercised but not *distinguished*.** On `FABRIC_1D` with the adjacent
  coordinate pair the golden driver pins, Ring resolves to the identical 1-hop route as Linear, so
  all 216 Ring cells re-verify the Linear path. The acceptance suite deliberately caps Ring at ≤ 2
  hops for the same reason. Genuinely testing the short-way wraparound needs a
  `FabricConfig.FABRIC_1D_RING` topology entry in `scripts/multidevice_sim_topologies.yaml`; this is
  a *harness* gap (no `SUPPORTED` value missing, no failing cell), so it is not a refinement.
* **Regime-B framing had zero coverage before this pass.** Every `feature_spec.INPUTS` shard and
  every acceptance shape has a page ≤ 4096 B, so `page_segments` was always 1 and the segmentation
  branches in both the sender writer and the receiver reader were unexecuted code — in an op whose
  design calls out "handle both packing multiple pages per transfer and splitting one page across
  transfers" as a requirement. `test_point_to_point_extended.py::test_segmented_page_framing` now
  covers it (and asserts it *stays* covered). If `/golden-tests` ever revisits `feature_spec.py`,
  adding one wide row-major shard (e.g. `(1, 1, 8, 2048)`, whose page is 8192 B for float32) would
  fold regime B into the graded cartesian.
* **`memory_config` is not an axis.** The op's contract admits interleaved DRAM *or* L1 and
  `validate()` only rejects sharded input, but `TARGET` has no `memory_config` axis and every golden
  cell is DRAM. `test_point_to_point_extended.py::test_l1_interleaved_input` covers L1 (TILE and a
  96 B row-major page). Promoting it to a real axis is a `feature_spec.py` (TARGET-expansion)
  decision, not a refinement.

### Beyond the current TARGET (not refinements — each needs `/golden-tests` to expand TARGET first)

* **Sharded input/output** — `validate()` rejects it with a `ValueError` today, per the spec. A
  memory-config expansion needing a sharded reader/writer (`/memory-layouts`).
* **Multi-link / worker-mux fabric** — the kernel helper already exposes the `MuxConn<N>` policy; the
  op uses a single link with one worker core. A pure bandwidth lever, no correctness gap.
* **Multi-core work split** — `ttnn.split_work_to_cores(grid, total_packets)` with one mux client per
  core, as `op_design.md` § "Work Distribution" sketches. Requires the multi-link work above (sharing
  one link between several workers is what `MuxConn` exists for), so it is not the
  embarrassingly-parallel `/interleaved-parallel` case.
* **True Ring wraparound** — needs the `FABRIC_1D_RING` topology entry described above.

### Escalation (outside this op)

`ccl_packet_dims`' payload-cap bug (fix 1) also affected the **bound C++ `ttnn.point_to_point`**,
which shares the helper. The one-line fix is committed here, but the fabric layer should probably
tighten its own guard too: `edm_fabric_worker_adapters.hpp:713`'s
`ASSERT(size_bytes <= this->buffer_size_bytes)` should be
`ASSERT(size_bytes + sizeof(PACKET_HEADER_TYPE) <= this->buffer_size_bytes)` — the assert as written
cannot catch the very overrun it exists to catch, and it compiles out in Release regardless. Worth
raising with the fabric owners.
