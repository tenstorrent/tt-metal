# Verification Report: all_reduce

**Op kind:** multi-device CCL **with a compute stage** — cross-chip movement (duplex
line multicast over TT-Fabric) *plus* an arithmetic reduction (element-wise SUM of
N tiles on the TRISCs). Built as a self-contained Python `ttnn.generic_op` +
`ttnn.MeshProgramDescriptor` with newly-authored reader (NCRISC) / compute (TRISC) /
writer (BRISC) kernels.

**Verification date:** 2026-07-25
**Verification vehicle:** deterministic craq-sim WH multi-device runner
`scripts/run_multidevice_sim_pytest.py --op all_reduce` → topology
`wh_t3k_allmmio_all_reduce`, mesh `(1, 8)`, `FABRIC_1D`.

---

## TL;DR

- **On-device verification PASSED, everywhere.** 21/21 in
  `tests/ttnn/unit_tests/operations/all_reduce/` (10 acceptance + 3 extended +
  8 precision) and 11 passed / 1 xfail in `eval/golden_tests/all_reduce/`
  (6 registry cells + 5 translated + the Ring refinement cell). Aggregate exit 0
  on every run — the cross-device multicast, the fused write+atomic-inc, and the
  N-way tile fold all actually executed and PCC asserted.
- **Verifier CLI is clean**: `supported_pass=6`, all five loud categories **0**.
- **`TARGET − SUPPORTED` is empty on every axis.** Phase 0 already covers the whole
  declared universe (`dtype ∈ {bf16, f32}`, `layout = TILE`, `topology = Linear`).
  That is why the refinement queue is short (2 entries) — it is a finding, not an
  omission. See "Refinement queue derivation" below for the audit that proves no
  `(axis, missing_value)` pair is unaccounted for.
- **Code review: no correctness bugs found in the shipped algorithm.** Four fixes
  applied (one latent cross-device lifetime bug, one registry-contract ordering
  hardening, one helper under-use, one dead symbol) — all re-verified green.
- **Three coverage holes closed with new tests** (`test_all_reduce_extended.py`):
  odd-N compute fold (dead code on an 8-device mesh), L1-interleaved memory
  (claimed but untested), and the design's documented cross-call semaphore
  re-arm window (Risk 5 — probed, does **not** reproduce).
- **Precision**: bf16 PCC ≈ 0.999994, fp32 PCC ≈ 0.99999996. The fp32 relative
  RMS (4.4e-4 ≈ 2⁻¹¹) is the Wormhole FPU's 19-bit SrcA/SrcB operand format, not
  a kernel defect — see Precision Baseline.

---

## Code Review

### Fixed

1. **Op-internal `GlobalSemaphore` cache keyed on `id(mesh_device)` — latent
   cross-device reuse (`all_reduce.py`).** The cache was a module-level
   `{id(mesh_device): sem}` dict that is never cleared, so it outlives the device
   it was created for. `MeshDevice` does **not** support weak references
   (`MeshDevice.__weakrefoffset__ == 0`), and CPython freely reuses a freed
   object's address, so the *next* `open_mesh_device` can land on the same `id()`
   and be handed a `GlobalSemaphore` belonging to a **closed** device. With a
   function-scoped `mesh_device` fixture (this op's whole test surface) that path
   is exercised on every test; it currently "works" only because the L1 allocator
   is deterministic and hands a fresh device the same semaphore address.
   **Fix:** bind the semaphore's lifetime to the device object itself
   (`mesh_device._ttnn_all_reduce_recv_semaphore`) — `MeshDevice` carries a Python
   `__dict__` (the root `conftest.py` already attaches `cache_entries_counter` the
   same way), so the cache cannot outlive, or be aliased across, devices. Still
   created exactly once per mesh with exactly one `synchronize_device`, still
   parked on the descriptor, still no per-call barrier — the mandate is unchanged.
2. **`validate()` ordering: dtype-dependent framing gates could mask an axis
   refusal (`all_reduce.py`).** The per-axis `SUPPORTED` gate ran *after* the
   `page_size % l1_alignment` and `ccl_packet_dims(...).page_segments == 1` checks.
   Both of those are functions of the **dtype**, so a future out-of-SUPPORTED dtype
   that trips them would raise `ValueError` instead of `UnsupportedAxisValue`, and
   the golden harness would record `xfail_wrong_mode` instead of a clean xfail.
   **Fix:** the axis + `EXCLUSIONS` gate now runs immediately after the
   placement/shape checks and before the framing gates. `rank >= 2` deliberately
   stays above the gate — it is a precondition of the shape-derived
   `tag_alignment`, and a rank-1 input deserves the structural message.
3. **Helper under-use in the writer's fabric egress
   (`kernels/all_reduce_writer.cpp`).** The plain-payload path hand-rolled the
   destination with `tt::tt_fabric::linear::addrgen_detail::get_noc_address(...)`
   and then called `payload.write(dst, l1)`. The duplex tier ships exactly that
   convenience — `DuplexWriteChannel::write_page(src_l1, page_idx, accessor)`
   (`ccl_helpers_dataflow.hpp:709`, impl `.inl:463-467`), which is what
   `all_gather`'s writer uses. **Fix:** plain pages now use
   `payload.write_page(l1, slot_base + p, gathered)`. The single remaining
   `addrgen_detail` call is on the **fused** page and is unavoidable —
   `DuplexFusedWriteIncChannel` has no page/addrgen overload — and is now
   commented as such.
4. **Dead code + clarity.** Removed the unused `_num_line_devices()` helper;
   `output_tensor` annotated `ttnn.Tensor | None`; the writer's kernel index in
   `ProgramDescriptor.kernels` is now a named `_WRITER_KERNEL_IDX` derived from a
   `_KERNEL_ORDER` tuple instead of a bare `program.kernels[2]`.

### Reviewed clean (no change needed)

- **CB sync balance.** `cb_broadcast_pages`: reader pushes `P`×1, writer waits/pops
  `P`×1. `cb_shard_tiles`: reader pushes `P`×N, compute waits/pops `P`×N — every
  wait uses the same count `N`. `cb_output_tiles`: compute pushes `P`×1, writer
  waits/pops `P`×1. Balanced in every regime, including `P == 1` (the
  `(1,1,32,32)` cell, where the *only* page is the fused one and the armed plain
  channel is legitimately never issued). ✔
- **`cb_shard_tiles` contiguity invariant holds.** `add_tiles(cb, cb, d, d+1, 0)`
  needs the N contributions to one output tile at N *contiguous* page offsets.
  `num_pages = 2N` (a multiple of N) and every push/pop is exactly N ⇒ the write
  pointer is always at page offset 0 or N ⇒ `get_write_ptr() + k*page_size` never
  wraps. Design Risk 2, correctly implemented and correctly commented. ✔
- **The N-way fold is correct for odd *and* even N** — and, unlike the shipped C++
  reference it is modelled on (`all_reduce_async/.../reduction.cpp`, whose odd
  branch is an empty `// TODO` that silently drops slice 0), the odd path is real:
  `copy_tile` seeds DEST then every pair accumulates. `acc_to_dest=false` on the
  first even-N pair means the fold does not depend on `tile_regs_acquire()`
  zeroing DEST. **Now empirically verified** for odd N via the new `(1,3)`
  submesh test — it was dead code under the `(1,8)` verification mesh. ✔
- **Duplex helper usage is idiomatic and the "two armed channels live at once"
  pattern is explicitly sanctioned.** `FabricDuplexSender` declared before the
  stream it lends its connection to (Risk 12); route pair bound once at `open()`;
  `arm_write` + `arm_fused_write_inc` armed once and issued many times. The
  header's `@note` (`ccl_helpers_dataflow.hpp:113-121`) states each `arm_*` draws
  its **own** pooled header and any mix may be live at once — a duplex stream with
  two armed channels draws 4 of the 8 per-RISC headers. No shared-header footgun
  here (unlike `arm_multicast_inc`/`arm_inc`). ✔
- **`flush=true` on the fused channel is present and load-bearing.** The payload
  lands in DRAM, the semaphore in L1; without the flush the inc could overtake the
  write and a peer would reduce stale DRAM (wrong values, no hang). Armed via
  `arm_fused_write_inc(page_size, 1, /*flush=*/true)`; the issue mask only updates
  dst/sem/size, so the armed `flush` survives every packet (verified in
  `.inl:341-368` + `kFusedIssueMask`). ✔
- **Multicast coverage is exactly-once.** `range_hops = N-1-i` through the `i+1`
  neighbour and `i` through the `i-1` neighbour sum to `N-1`; each slot is filled
  by the neighbour whose `ccl_dm_route(...).is_forward` says so (never by index
  sign), and `validate()` asserts the two neighbours land in *different* slots
  (Risk 4). An absent direction's 6 route words are zeros and are never
  programmed, because `arm_*`/issue both gate on `DuplexConn::has(dir)` — which is
  also what keeps a `range_hops == 0` header (a router `ASSERT(false)`, Risk 3)
  off the wire on Linear. ✔
- **`noc_async_writes_flushed()` before every `cb_pop_front`** in writer phase 1 —
  `close()`/`drain()` are write+atomic barriers only and do **not** guarantee the
  fabric sender has read the CB slot. Correct per Risk 16. ✔
- **API correctness.** `void kernel_main()` in all three kernels (no deprecated
  namespace/`MAIN` pattern); includes are `api/dataflow/dataflow_api.h` and
  `api/compute/*` (no bare `dataflow_api.h`); addressing is `TensorAccessor`
  everywhere (no deprecated `InterleavedAddrGen`); `compute_kernel_hw_startup`
  exactly once at the top of the compute kernel with only short inits inside the
  DEST window; `pack_tile(0, cb)` with no index (the index is ignored when
  `out_of_order_output == false`). ✔
- **Broadcast dimensions.** The op's only binary op is a full-tile `add_tiles`
  (`BroadcastType::NONE`) with both operands from the same CB at different tile
  indices — no broadcast dim to get wrong, and no reduce-produced operand (so no
  `Row0`/`Col0` valid-region restriction). No CB is filled with repeated data. ✔
- **`fp32_dest_acc_en` tracks the dtype** (`HiFi3` for fp32 to dodge the WH
  HiFi4+fp32-dest-acc hardware bug, `HiFi4` for bf16), and the kernel's
  `static_assert(1 <= DEST_AUTO_LIMIT)` auto-detects the host setting so kernel and
  host cannot desync (Risk 9). ✔
- **No wrapping / re-export of any existing CCL op.** The op imports only `ttnn`,
  `ttnn._ttnn.operations.ccl.Topology`, and its own descriptor module; the kernels
  include only fabric/compute APIs plus `kernel_lib`. The generation mandate holds. ✔

### Design conformance (`op_design.md`)

Checked on the four binding dimensions — **all match**:

| Dimension | Design | Implementation |
|---|---|---|
| Algorithm | broadcast-all (chip-level multicast) then local N-way sum; local contribution read from `input_tensor`, not mirrored | identical (`*_with_local_copy` deliberately not used) |
| Pipeline topology / RISC ownership | reader phases 1/2/3 (stage → barrier+re-arm → interleave N), writer phases 1/2 (duplex multicast → drain), compute folds | identical |
| Parallelization | one worker core `(0,0)` per device, one `ProgramDescriptor` per mesh coordinate, whole shard per core | identical |
| Inter-core / inter-chip comms | duplex `Cast::Multicast` stream, last packet a fused write+atomic-inc, one cached `GlobalSemaphore` as a receive counter, receiver-side re-arm | identical |

Only cosmetic deviation: the gathered buffer is allocated with the
`allocate_tensor_on_device(shape, dtype, layout, device, memory_config)` overload
rather than the design's `TensorSpec` form — same resulting single mesh allocation
(which is what makes the noc0-encoded destination resolve per chip, Risk 7).

### Prompt-rule audit (`eval/prompts/all_reduce.txt`)

The prompt has no `## Rules` section; its framework-owner guidance is written as
directives, and each applicable one is satisfied: MeshProgramDescriptor per device
✔; GlobalSemaphore created once + `synchronize_device` once + parked on the
descriptor + no per-call barrier ✔; host route/framing via `ccl_dm_route` /
`ccl_packet_dims` / `setup_fabric_connection` (nothing reimplemented) ✔; kernel
egress through the **DUPLEX** tier, not hand-rolled dual headers with per-send
`has_*_connection()` checks ✔; reduction authored as a real TRISC compute kernel ✔.

*Advisory (not followed, justified):* the prompt shows
`write_fused_with_local_copy` in its sketch. The op reads its own contribution
straight from `input_tensor` instead. That is the better call — the fused mirror
deliberately does **not** flush local writes, so using it would require a second
local handshake before the reader could trust slot `my_id`, and the receive
counter's target would become `N` instead of `N-1`. Documented in `op_design.md`.

### Not fixed (recorded, no failing cell)

- **fp32 operand precision is FPU-bound, not kernel-bound.** `add_tiles` feeds
  SrcA/SrcB, which are 19-bit (1s+8e+10m) on Wormhole, so each fp32 addend is
  rounded to ~11 mantissa bits before the add even though DEST is fp32. Measured
  fp32 rel-RMS 4.4e-4 ≈ 2⁻¹¹·¹ matches exactly. The lever would be an SFPU-based
  fold (`copy_tile` into DEST + SFPU binary add, which is fp32-native) at the cost
  of N DEST registers and N copies. **No refinement filed**: every fp32 cell
  passes at PCC 0.9999999 against a 0.999 gate, so there is no failing cell to
  move and no axis to add. Note also that exposing `math_fidelity` would change
  nothing — `add_tiles` pins `MathFidelity::LoFi` internally.
- **Single worker core per device.** All P pages of the broadcast and the whole
  reduction run on one Tensix, so wide shards are latency-bound. Multi-core here
  is *not* the embarrassingly-parallel case `/interleaved-parallel` covers:
  `FabricDuplexSender` is defined as one worker owning both directions of one
  `FabricConnectionManager`, and `MuxConn<N>` cannot back the duplex tier (it
  exposes `sender()` with no direction). It needs one fabric link per core — a
  cross-core-dependency change with no failing cell and no SUPPORTED axis, so it
  stays out of the queue (design Risk 18).
- **Ring reduce-scatter + all-gather** (the bandwidth-optimal algorithm) is a
  perf-only alternative to broadcast-all; `2(N-1)` fabric phases instead of 1.
  Not queued — no failing cell. (Distinct from Refinement 1, which is about the
  `Ring` *topology* value on the existing broadcast-all algorithm.)
- **Reader reads each input page twice** (once to feed the fabric in phase 1, once
  to feed the fold in phase 3). Removing the second read requires either the
  rejected local-mirror handshake or an L1 staging buffer sized to the whole
  shard. Perf, no failing cell.
- **Per-page `noc_async_read_barrier` in reader phase 1** — conservative but
  correct with a 2-deep CB (phase 3 already batches: N reads, one barrier). Perf.

### Memory-pressure observations (no OOM triggered)

- Per-core L1 CB footprint is `page_size × (2 + 2N + 2)`, i.e. **bounded by the
  mesh size, not by the tensor shape**: 80 kB at N=8/fp32, ~272 kB at N=32/fp32,
  against a ~1.5 MB budget. Shape growth costs nothing (the reduction streams tile
  by tile), so `/memory-budget-metal` has nothing to chunk here. A mesh beyond
  N≈100 would be the first L1 concern.
- The **op-internal gathered buffer is `N ×` the shard** and inherits
  `input_tensor.memory_config()`. On DRAM that is invisible; on **L1-interleaved**
  input it puts `N ×` shard bytes in L1 and will fail to allocate for large
  shards. The new `test_all_reduce_l1_interleaved` cell pins the small-shape L1
  path as working; large-shape L1 is a documented sharp edge, not a tracked
  failure. (Slot `my_id` of that buffer is intentionally never written — `1/N` of
  it is unused, which is what removes a writer→reader handshake.)

---

## Registry Conformance

- **`INPUT_TAGGERS`** — present: `{"alignment": tag_alignment}`, signature
  `(inputs, axes)`. Reads the per-device shard's last two dims (`both % 32 == 0 →
  tile_aligned`). Load-bearing, not cosmetic: the landing buffer scales dim 0 by N,
  and "slot `k` == pages `[k·P, (k+1)·P)`" only survives that scaling when each
  shard occupies whole tile-rows. ✔
- **`SUPPORTED`** — present and declares every gated axis: `dtype`, `layout`,
  `topology`, `alignment`. Nothing the kernels branch on is missing (there is no
  index/`dim` parameter and no boolean mode). ✔
- **`EXCLUSIONS`** — present, empty `[]`. No cell inside the SUPPORTED rectangle
  is refused, and none needed to be added during verification (no
  `numerical-bug` / structural-gap cells appeared). ✔
- **`validate()`** — present and is the **first statement** of `all_reduce(...)`.
  Order after this pass: placement/shape structural checks → per-axis `SUPPORTED`
  → `EXCLUSIONS` → dtype/page framing gates → `output_tensor` spec → fabric
  direction slotting. Raises `UnsupportedAxisValue` / `ExcludedCell` from
  `ttnn.operations._op_contract` (both `NotImplementedError` subclasses, so
  `xfail(strict=True, raises=NotImplementedError)` works) and `ValueError` for
  structural input errors. ✔
- **Op file does NOT declare `INVALID`** — confirmed by inspection; `INVALID` is
  sourced only from `feature_spec.py`. ✔
- **Package re-export** — `__init__.py` exports `all_reduce`, `SUPPORTED`,
  `EXCLUSIONS`, `INPUT_TAGGERS`, which is where `test_golden.py` reads them. ✔
- **Auto-fixes applied to SUPPORTED from XPASS evidence: none.** `xpass_drift` is
  0; no cell outside SUPPORTED passed. SUPPORTED was not widened by this pass.

### INVALID audit (`eval/golden_tests/all_reduce/feature_spec.py`)

`INVALID = []`.

- **Correct for the current TARGET, and no entry is missing.** The three sanity
  rules have nothing to bite on: there is exactly one tensor axis-group (the input
  shard), `layout` is pinned to `TILE`, and both TARGET dtypes (`bfloat16`,
  `float32`) are constructible on TILE. No cross-tensor coupling exists to get
  wrong (single-input op, no weights).
- **The canonical `{bf8b, ROW_MAJOR}` entry is legitimately absent**, not
  forgotten: that cell requires *both* `bfloat8_b` and `ROW_MAJOR_LAYOUT` to be in
  TARGET, and neither is. It becomes **required** the moment `/golden-tests`
  widens TARGET on either axis — `op_design.md` §"Structural impossibilities"
  already records it verbatim for that day.
- Not a norm-like op → no weight/no-weight canonicalization cells expected. ✔
- Nothing currently out of reach is mis-filed as INVALID: `ROW_MAJOR` (needs
  tilize/untilize around the fold), `Ring` (needs the alternating target-count
  math + the `range_hops == 0` guard), `non_tile_aligned` (needs a rank-general
  landing-buffer mapping) and `bfloat8_b` on TILE are all *kernel improvements* —
  i.e. EXCLUSIONS/refinement territory, correctly left out of INVALID.

**Verdict:** well-formed. No change recommended, and no `/golden-tests` action is
required for correctness — only the optional TARGET widenings noted below.

---

## Precision Baseline

`tests/ttnn/unit_tests/operations/all_reduce/test_all_reduce_precision_baseline.py`
— 8/8 green on the WH sim. Oracle = element-wise sum of the 8 per-device shards
accumulated in **fp32** then cast to the tensor dtype, so the reference is not
itself limited by bf16 rounding; metrics measured on device 0 (all 8 devices are
asserted against the same oracle). N = 8 addends in every row.

| Shard shape | dtype | PCC | Max Abs Err | Mean Abs Err | Relative RMS Err |
|-------------|-------|-----|-------------|--------------|------------------|
| (1,1,32,32)   | bfloat16 | 0.9999935 | 0.062500 | 0.006571 | 0.003906 |
| (1,1,64,128)  | bfloat16 | 0.9999938 | 0.062500 | 0.006199 | 0.003702 |
| (1,1,256,256) | bfloat16 | 0.9999937 | 0.062500 | 0.006283 | 0.003754 |
| (2,1,32,64)   | bfloat16 | 0.9999939 | 0.062500 | 0.006153 | 0.003691 |
| (1,1,32,32)   | float32  | 0.99999996 | 0.006040 | 0.000979 | 0.000441 |
| (1,1,64,128)  | float32  | 0.99999996 | 0.005925 | 0.000966 | 0.000434 |
| (1,1,256,256) | float32  | 0.99999996 | 0.006564 | 0.000957 | 0.000435 |
| (2,1,32,64)   | float32  | 0.99999996 | 0.004521 | 0.000967 | 0.000440 |

**Assessment:** error is **shape-independent** (identical to 3 significant figures
across 1 → 64 tiles per shard) and depends only on dtype — exactly what a
streaming per-tile fold should look like: no accumulation growth with shard size,
because the reduction depth is N, not P.

- **bfloat16**: rel-RMS ≈ 3.7e-3 ≈ 2⁻⁸·¹, i.e. one bf16 output quantum. Max abs
  0.0625 is 2 ulps at the |sum| ≈ 8–16 tail. This is the fp32-oracle-vs-bf16-device
  gap, not chained-accumulation blow-up (the fold accumulates in DEST).
- **float32**: rel-RMS ≈ 4.4e-4 ≈ 2⁻¹¹·¹ — *not* fp32 epsilon. This is the
  Wormhole FPU operand format: SrcA/SrcB are 19-bit (1s+8e+10m), so each fp32
  addend is rounded to ~11 mantissa bits on its way into `add_tiles`, even though
  `fp32_dest_acc_en` makes the accumulator fp32. Hardware-level, uniform, and
  ~2000× inside the gate. See "Not fixed" for the SFPU lever.

**Recommended tolerances** (what the suites use, and they are right):
`PCC ≥ 0.99` for bfloat16, `PCC ≥ 0.999` for float32; golden `check_output`
tolerance `(0.99, 0.05)`. For an allclose-style gate on a magnitude-~3 sum:
bf16 `atol=0.07, rtol=0.05`; fp32 `atol=0.01, rtol=0.01`.

---

## Verifier CLI Summary

`eval/results/all_reduce/verifier_report.json` (produced by
`python3 -m eval.verify_supported eval/results/all_reduce ttnn.operations.all_reduce`
over the golden-directory JUnit + `eval.axes_plugin` axes sidecar, run on the
`(1,8)` WH sim):

- supported_pass: **6**  (3 `feature_spec.INPUTS` shapes × {bf16, f32} × TILE × Linear × tile_aligned)
- xfail_expected: **0**  (empty *by construction* — `TARGET − SUPPORTED = ∅`; see below)
- invalid_skipped: **0**  (`INVALID = []`)
- supported_fail: **0**   ✓ must be 0 to ship
- xpass_drift: **0**      ✓ must be 0 to ship
- xfail_wrong_mode: **0** ✓ must be 0 to ship
- supported_marked_xfail: **0**, invalid_unexpected: **0**
- no_axes_found: **6** — the `test_translated.py` cases, which are not
  registry-parametrized (no `axes` param ⇒ no sidecar entry). 5 passed; **1 is the
  `topology=Ring` refinement cell**, which the shared golden conftest converted to
  a lenient xfail after `validate()` raised
  `UnsupportedAxisValue: topology=Topology.Ring not in SUPPORTED [Topology.Linear]`
  — i.e. the correct refusal, in the correct mode, in the wrong bucket only because
  translated tests carry no axes sidecar.
- total: 12

### Refinement queue derivation (the `xfail_expected` audit)

Iterating `by_category.xfail_expected` yields **zero entries**, so the
`Counter`-over-axis-combinations is empty. That is not a queue gap; it is the
consequence of:

```
TARGET   = {dtype: [BFLOAT16, FLOAT32], layout: [TILE], topology: [Linear]}
SUPPORTED= {dtype: [BFLOAT16, FLOAT32], layout: [TILE], topology: [Linear],
            alignment: [tile_aligned]}          # alignment is tagger-only, absent from TARGET
TARGET[axis] − SUPPORTED[axis] = ∅   for every axis
```

Every `(axis, missing_value)` pair, and every out-of-SUPPORTED cell observed
anywhere in the suite, is accounted for:

| Source | `(axis, value)` | Disposition |
|---|---|---|
| `TARGET − SUPPORTED` | *(none)* | Phase 0 covers the full declared universe |
| translated xfail cell `test_ring_all_reduce_refinement_axis` | `topology = Ring` | **Refinement 1** (the queue's anchor) |
| `SUPPORTED["alignment"]` missing value (tagger-only axis) | `alignment = non_tile_aligned` | **Refinement 2** |
| out-of-TARGET, documented omission | `layout = ROW_MAJOR_LAYOUT` | Not queued — TARGET pins TILE (the reduction is a tile compute); needs `/golden-tests` to widen TARGET first, and then also the `{bf8b, RM}` INVALID entry |
| out-of-TARGET, documented omission | `dtype = bfloat8_b` | Not queued — absent from TARGET; would need `bfp8_pack_precise` tuning + the INVALID entry above |
| no axis exists | sharded `memory_config` | Not queued — `validate()` rejects sharded; TARGET has no `memory_config` axis. Interleaved **L1** is allowed and is now covered by `test_all_reduce_l1_interleaved` |
| no axis exists | `num_links > 1`, multi-core | Not queued — perf, no axis, no failing cell (see "Not fixed") |

---

## Extended Tests Added

`tests/ttnn/unit_tests/operations/all_reduce/test_all_reduce_extended.py` — 3/3
green. Each closes a hole the acceptance suite structurally cannot reach:

1. **`test_all_reduce_back_to_back_no_sync`** — two `all_reduce` dispatches with
   **no** intervening `synchronize_device` (both inputs staged first). This is the
   direct probe of design **Risk 5** (a peer's next-call increment being wiped by
   this device's post-wait `noc_semaphore_set(sem, 0)`). **Result: PASSES.** The
   window does not reproduce on the deterministic sim — see Recommendations for
   why it is still recorded as a residual risk rather than declared absent.
2. **`test_all_reduce_l1_interleaved`** — L1-interleaved input/output. `validate()`
   only rejects *sharded*, so L1 was an accepted-but-untested configuration; the
   op-internal landing buffer inherits the memory config, so this is also the
   first cell that puts an `N ×` shard buffer in L1. **PASSES.**
3. **`test_all_reduce_odd_line_submesh`** — a `(1,3)` submesh of the same 8-chip
   line. N is always **even** on the `(1,8)` verification mesh, so the compute
   kernel's `if constexpr (num_devices % 2 == 1)` seed-with-`copy_tile` branch (the
   one the shipped C++ reference gets wrong) was dead code in every existing test.
   Also exercises shorter multicast ranges on both directions. **PASSES.**

Deliberately *not* added (belongs to refinements, per the verifier contract): rank
sweeps (translated suite already covers rank 2/3/4), multi-core edges, batch-size
sweeps, `Ring`, `non_tile_aligned`.

---

## Recommendations

1. **Refinement order** (see `op_requirements.md`): **R1 Ring topology** first —
   it is the only entry with a named failing cell, and it is the algorithm-shaped
   one (alternating target counts + the mandatory `range_hops == 0` guard). Then
   **R2 non_tile_aligned**, which is mostly a host-side landing-buffer mapping and
   composes cleanly on top of whatever routing R1 lands. Neither maps onto a
   current implementation skill except R2's nominal `/memory-layouts` pointer —
   the skill inventory covers single-device compute precision, in-kernel layouts,
   interleaved multi-core and L1 budget, none of which is CCL fabric routing.
2. **Re-verify every refinement on the WH sim, not silicon.** This host is
   single-device; the only multi-device path is
   `scripts/run_multidevice_sim_pytest.py --op all_reduce` (topology
   `wh_t3k_allmmio_all_reduce`). Tests **must** open exactly `(1,8)` + `FABRIC_1D`
   or fabric init hangs (`Fabric Router Sync: Timeout`) — that is a test/topology
   mismatch, never a sim or op defect. Note `--timeout` and keep each invocation
   bounded; a full golden-directory pass is ~3 min, the unit dir ~4 min.
3. **Residual cross-call risk (design Risk 5) — keep it in mind, do not gate on
   it.** The probe passes, and the mechanism is genuinely bounded (exactly `N-1`
   incs per call; the sender's `close()` drains before its program ends). But the
   sim schedules chips in near-lockstep, so it is weak evidence about real
   silicon skew, where a fast device entering call *k+1* while a slow peer is
   still parked in call *k* would both (a) risk a wiped increment and (b) let a
   late multicast land in a *freed-and-reused* gathered buffer. If a future
   refinement ever needs a hard fence, the mandate-compliant shape is a **second
   parked `GlobalSemaphore` used as an entry barrier** (the
   `all_gather_async`/`llama_shapes_sharded_writer.cpp:96-118` pattern), not a
   shared-header `arm_multicast_inc` barrier and not a per-call
   `synchronize_device`. Filed here, not in the queue, because there is no failing
   cell to move.
4. **fp32 fold precision** (rel-RMS 2⁻¹¹, FPU-operand-bound) and **single-core
   throughput** are the two known non-blocking limits; both are analysed under
   "Not fixed" with their concrete levers (SFPU fold; one fabric link per core).
   Neither belongs in the queue today.
5. **If more golden breadth is wanted, it needs `/golden-tests`, not the
   implementer.** `feature_spec.INPUTS` is 3 tile-aligned shapes and TARGET is
   3 axes wide, so the registry surface is only 6 cells. Adding a non-tile-aligned
   INPUT (e.g. `((1,1,48,96),)`) would give **Refinement 2** real golden cells to
   flip; widening TARGET with `ROW_MAJOR_LAYOUT` / `bfloat8_b` would open two more
   refinements (and would then *require* the `{bf8b, ROW_MAJOR}` INVALID entry).
   Recorded as a request, not performed here — the verifier does not edit
   `feature_spec.py`.
