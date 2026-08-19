# Verification Report: reduce_scatter

**Op class**: multi-device CCL **with a compute stage** (fabric line gather + TRISC element-wise
N-way SUM + scatter-by-addressing).
**Verified on**: REAL 4-chip Blackhole hardware, mesh `(1, 4)` with `FabricConfig.FABRIC_1D`
(topology `bh_quietbox_1x4_hw`, `runtime = hardware`), via
`scripts/run_multidevice_sim_pytest.py --op reduce_scatter`. This is the correct runner for a CCL
op — `run_safe_pytest.sh` forces slow dispatch on sim and has no multichip/hang awareness.
Aggregate exit = 0 on every run below.

---

## Code Review

The implementation follows `op_design.md` and the prompt's framework-owner mandates closely.
**No correctness defects were found and no code changes were required** — the review pass produced
two new test files (extended + precision baseline) and the verification artifacts, nothing else.
Helper usage is complete: this op composes all three CCL helper families the prompt mandates
(fabric dataflow, compute accumulation, shared schedule), and improves on the all_reduce
predecessor by using the purpose-built `sum_blocks` helper where all_reduce hand-rolled the
seed-then-accumulate idiom.

### Reviewed and confirmed correct

- **Fabric egress via the CCL kernel helper** (`ccl_helpers_dataflow.hpp`). The gather writer is
  the mandated safety-by-construction typestate path: `FabricStreamSender<>(conn_arg_idx,
  is_forward, alignment)` → `.open(unicast_route(num_hops))` → `arm_unicast_write(page_size)` /
  `arm_inc(1)` → `write_page` / `inc` → `close()`. `dst_is_forward` is correctly peeked from the
  conn block's leading `has_forward` flag; the route direction comes from the host's
  `ccl_dm_route(...).is_forward` (sign reversal host-owned). `noc_async_writes_flushed()` runs
  after every `write_page` BEFORE `cb_pop_front` (the CB-slot-reuse guard). Line-end direction
  (`my_num_targets == 0`): the writer opens no connection and every RT read is guarded behind the
  `if constexpr` early return — matching risk #13 in the design.
- **Op-owned receive half is exactly the helper's documented split**: counting
  `noc_semaphore_wait_min` (incremental, `running = 1..num_relay_blocks`), local `noc_async_read`
  as receive ingress (no FabricStreamReceiver exists by design), and the cache-reuse
  `noc_semaphore_set(sem, 0)` re-arm AFTER the last wait in every reader path (RECEIVER resets
  after its wait). The 2-call program-cache test exercises the re-arm.
- **Semaphore accounting balances**: device i's forward core receives exactly `i` incs (blocks
  `i-1..0` from device i-1's forward writer, seed-then-relay order matches the reader's
  `c = my_chip_id - 1 - k` walk); the backward core receives `N-1-i`. Every Phase-A reader's final
  wait equals the number of blocks that must land locally in its direction, so the Phase-A program
  cannot return before device i's `gather_buffer` is complete — the same-queue ordering of the two
  `generic_op` dispatches is therefore sufficient (no missing cross-phase barrier).
- **Compute stage uses the purpose-built helper**: `compute_kernel_lib::sum_blocks(cb_gathered,
  cb_out, N, 1, /*pop_input=*/true)` — the exact "gathered per-device partials land as contiguous
  blocks of one input CB" shape the helper banner names. Verified against
  `accumulate_helpers_compute.inl:106-157`: waits the whole N-tile input, DST-chunks against
  `DEST_AUTO_LIMIT` (fp32 dest-accum aware), odd-N copy_tile-seeds / even-N accumulates from
  DST-zero, pops N before pushing 1. `binary_op_init_common` (the helper's documented `@pre`)
  runs once before the loop. The design's rejection table for `BlockAccumulate` /
  `RingRsSchedule` / `eltwise add` / `reduce()` / `FabricDuplexSender` / `mcast_pipe` was
  re-checked against the headers — every rejection is justified (two-CB shapes, multi-step ring
  schedules, within-tile reduction, and intra-device multicast respectively — none of which exist
  in this algorithm).
- **The scatter lives in the shared schedule header, not hand-rolled arithmetic**:
  `SliceRowWalker(slice_Wt, tensor_Wt)` + `set_base(slice_tile_offset(3, my_chip_id, ...))` +
  `reset_offsets(start_tile % slice_Wt, (start_tile / slice_Wt) * tensor_Wt)`. Verified against
  `ccl_helpers_schedule.hpp:491-540`: `next()` yields exactly
  `(t / slice_Wt)·Wt + i·slice_Wt + (t % slice_Wt)` — the design's src-page formula — including
  mid-row `start_tile` values (covered by the new extended test) and batch > 1 (row index runs
  over all `B·C·Ht` rows; the gather blocks are full row-major shards, so the mapping holds — the
  translated multibatch cell covers it). The kernel `static_assert`s both
  `is_supported_scatter_dim(dim)` and the Phase-0 `dim == 3` pin.
- **CB sync balance** (push == wait == pop on every CB): `cb_relay_pages` — reader pushes
  `(1 + num_relay_blocks)·P` pages iff `my_num_targets > 0`, writer pops the same count;
  `cb_self_copy_scratch` — reserve-only, never pushed/popped (the proven all_reduce idiom, kept
  deliberately per design risk #6); `cb_gathered_slices` — N per position pushed, `sum_blocks`
  waits/pops N; `cb_reduced_slice` — 1 per position each side. `cb_gathered_slices` capacity
  `2·N` tiles ≥ the helper's whole-input wait, and reserve/push are always in N-groups so the
  reserved region never straddles the CB wrap.
- **TensorAccessor** everywhere (no deprecated `InterleavedAddrGen`), `void kernel_main()` in all
  five kernels (including compute — the modern pattern), `api/dataflow/dataflow_api.h` include
  paths, scalar CT args before `TensorAccessorArgs`. No unused CT args (the all_reduce review's
  `ring_size` nit does not recur here — the five single-purpose kernel sources avoid the uniform
  CT superset entirely).
- **Broadcast**: none anywhere in the op (equal-shaped tile blocks summed elementwise); nothing to
  check beyond the design's own table.
- **Program-cache discipline**: all buffer addresses, page counts, `sem_addr`, `num_hops`, NoC
  coords, `start_tile`/`n`, `slice_Wt`/`Wt` are runtime args; CB indices, direction, `my_chip_id`,
  N, dim, alignment, accessor args are compile-time. Confirmed by the passing program-cache test
  (entry count stable across the 2nd call) and the translated `test_program_cache_hit`.

### Justified deviation from op_design.md (design snippet bug, not a code bug)

`op_design.md`'s "Registry contract (Phase-0)" snippet shows `"dim": [3]` inside `SUPPORTED`. The
implementation instead keeps `dim` as an op-level gate (`_SUPPORTED_DIMS = (3,)`, refusing with the
typed `UnsupportedAxisValue` after canonicalizing `-1 → 3`). This deviation is **load-bearing and
correct**: the golden `feature_spec.TARGET` does not sweep a `dim` axis, and
`eval/feature_matrix.py:unsupported_reason` xfails any cell whose values dict is missing a
SUPPORTED axis (`"axis 'dim' missing from values dict"`) — declaring `dim` in SUPPORTED would have
turned all 6 golden cells into strict-xfail XPASS reds. Verified directly against the harness
source. The op-file comment documents the rationale and the promotion path (move into SUPPORTED
when a future TARGET sweeps the axis).

### Advisory (no fix; no failing cell to point at)

- **Phase-A self-copy is fully serialized** — the forward reader does `noc_async_read_barrier()`
  then `noc_async_write_barrier()` per page over the P-page shard. Correct but page-serial; a
  pipelined copy using both scratch slots would overlap read/write. Runs once per device on the
  smallest data path; performance-only. (Same advisory as the all_reduce review — shared idiom.)
- **`_SEMAPHORE_CACHE` keyed on `id(mesh_device)`** — a closed-then-reallocated mesh device could
  in principle reuse the same `id` and pick up a stale GlobalSemaphore handle. Same pre-existing
  idiom as the proven all_reduce op; kept consistent rather than diverging the two ops in a
  verification pass. A shared fix (weakref keying in one place) would be a cleanup across both ops.
- **Gather-level Phase-A traffic** — each shard traverses the line whole instead of shedding 1/N
  per hop, and the `gather_buffer` is N× the shard. This is the design's explicit Phase-0
  algorithm choice (correct-first); the traffic-optimal ring/line reduce-en-route alternative is
  recorded under Recommendations (beyond-TARGET, no failing cell).

---

## Registry Conformance

Confirmed the four declarations are present and correctly wired in `reduce_scatter.py`:

- **`INPUT_TAGGERS = {}`** — empty by design: no shape-derived axis exists (the reduction is
  always the full element-wise sum; every accepted input is tile-aligned by construction via
  TILE_LAYOUT + the `W % (N·32)` structural gate). `validate()` still iterates the dict with the
  correct `tagger(inputs, axes)` signature, so a future tagger is a drop-in.
- **`SUPPORTED = {dtype: [bfloat16, float32], layout: [TILE], topology: [Linear]}`** — covers
  every axis the kernels gate on. (`memory_config` DRAM/L1 is accepted but not a gated categorical
  axis — the new extended L1 test proves both buffer types work; `dim` is the documented op-level
  gate above.)
- **`EXCLUSIONS = []`** — present, empty (no in-SUPPORTED cell is refused).
- **`validate()`** — structural errors (`ValueError`: not MeshDevice, mesh not `(1, N)`, N < 2,
  sharded input, rank ≠ 4, page not 16-B aligned, `W % (N·32) != 0` (loud, never padded),
  output-spec mismatch), then the axis gates in the correct order: SUPPORTED per-axis
  (`UnsupportedAxisValue`) **then** EXCLUSIONS cell-level (`ExcludedCell`), both the typed
  refusals from `ttnn.operations._op_contract`. `dim` is canonicalized to positive BEFORE its
  membership test (the `-1 ≡ 3` alias passes — covered by the acceptance test). The public
  `reduce_scatter()` calls `validate()` on its first line, before any allocation or dispatch.
- **No `INVALID` symbol in the op file** — confirmed absent. INVALID is sourced from the golden
  feature spec only.
- **Package exports** — `__init__.py` re-exports `reduce_scatter`, `SUPPORTED`, `EXCLUSIONS`,
  `INPUT_TAGGERS` at package level, as the harness requires.

**No drift.** The verifier CLI reports `xpass_drift = 0`, `supported_fail = 0`,
`xfail_wrong_mode = 0`. No auto-fixes to SUPPORTED were needed — every SUPPORTED cell passes and
there is nothing outside SUPPORTED to promote (SUPPORTED already equals TARGET on every axis).

### INVALID audit (`eval/golden_tests/reduce_scatter/feature_spec.py`)

`INVALID = []`, and this is **correct** for the current TARGET:

- `TARGET = {dtype: [bf16, f32], layout: [TILE], topology: [Linear]}` — every combination is
  constructible (TILE + float dtypes), so there is no structurally-impossible cell to declare.
- **Single-tensor op** — no cross-tensor-axis coupling risk (the canonical authoring mistake).
- **Canonical bf8b + ROW_MAJOR entry correctly omitted** — it only applies when TARGET contains
  ROW_MAJOR / bfloat8_b; this TILE-only float TARGET does not, so the entry would reference axis
  values the harness never generates (`op_design.md` documents this in the Verification Topology
  row).
- **No norm-like weight axes** → no no-weight canonicalization cells needed.

No changes to `feature_spec.py` are proposed.

### Design conformance

Checked against `op_design.md` on the binding dimensions — all match:

| Dimension | Design | Implementation |
|---|---|---|
| Algorithm | GATHER-THEN-REDUCE-LOCAL-SLICE, 2 ordered dispatches, scatter folded into Phase-B addressing | matches ✓ (no full-schedule ring; `sum_blocks` per position) |
| Pipeline topology | Phase A: 2 direction cores/device, NCRISC reader + BRISC writer, 5 single-purpose kernel sources; Phase B: reader/compute/writer | matches ✓ |
| Parallelization | Phase A: 1 worker/direction, single link; Phase B: `split_work_to_cores(grid, P_out)` two-group split | matches ✓ |
| Inter-core comm | ONE op-internal GlobalSemaphore, per-(device, core) counting words, inc-after-data on one connection, receiver re-arm | matches ✓ |
| Compute config | HiFi4 + `fp32_dest_acc_en=True`, `math_approx_mode=False`, `dst_full_sync_en=False` | matches ✓ |

### Prompt rules

`eval/prompts/reduce_scatter.txt` has no `## Rules` section — the stock policy applies. The
generation mandate (from scratch, no wrapping of any existing CCL op) holds: the op imports only
`ttnn` public APIs, the bound host helpers, and its own kernels. All framework-owner mandates are
satisfied: per-device MeshProgramDescriptor entries (1); op-internal GlobalSemaphore created once,
synchronized once, parked on the descriptor, no per-call post-dispatch barrier (2); host route via
`ccl_dm_route` + `setup_fabric_connection` — `ccl_packet_dims` legitimately unused since the op
uses 1:1 page↔packet framing like the proven all_gather/all_reduce, with the load-bearing
`page_size % 16` guard kept in `validate()` (3); kernel-side fabric egress through the typestate
helper with the op-owned waiting half / ingress / re-arm (4); reduction compute through
`accumulate_helpers_compute.hpp` per the gather-then-reduce algorithm choice (5).

---

## Precision Baseline

Measured on the `(1, 4)` Blackhole line mesh (N = 4 summands), oracle accumulated in fp32 then
cast, so the reference is not itself limited by bf16 rounding. Metrics are the WORST device of the
4 (per-device-distinct outputs — a slice-addressing bug would show as one catastrophic device).
From `tests/ttnn/unit_tests/operations/reduce_scatter/test_reduce_scatter_precision_baseline.py`.
"Max ULP@scale" = worst error in units of the output dtype's representable step at the tensor's
working magnitude (element magnitude clamped below at the oracle RMS).

| Shard shape | dtype | PCC (worst dev) | Max Abs Err | Mean Abs Err | Relative RMS Err | Max ULP@scale |
|-------------|-------|-----------------|-------------|--------------|------------------|---------------|
| (1,1,32,128)  | bfloat16 | 0.9999963 | 0.031250 | 0.002688 | 0.002722 | 2 |
| (1,1,32,128)  | float32  | 1.0000000 | 0.004704 | 0.000682 | 0.000446 | 24292 |
| (1,1,64,256)  | bfloat16 | 0.9999964 | 0.031250 | 0.002677 | 0.002712 | 1 |
| (1,1,64,256)  | float32  | 1.0000000 | 0.004789 | 0.000679 | 0.000440 | 20088 |
| (1,1,256,512) | bfloat16 | 0.9999964 | 0.062500 | 0.002624 | 0.002690 | 3 |
| (1,1,256,512) | float32  | 1.0000000 | 0.005069 | 0.000669 | 0.000436 | 29728 |
| (2,1,32,256)  | bfloat16 | 0.9999964 | 0.031250 | 0.002677 | 0.002712 | 1 |
| (2,1,32,256)  | float32  | 1.0000000 | 0.004789 | 0.000679 | 0.000440 | 20088 |

**Assessment**: Accuracy is excellent and shape-independent.
- **bfloat16** error is 1–3 output-ULP at tensor scale (rel-RMS ≈ 0.0027 ≪ the bf16 mantissa
  budget 2⁻⁷ ≈ 0.0078) — dominated by the bf16 storage quantization of the sum, with the fp32 DST
  accumulation keeping the intermediate exact. Every PCC ≥ 0.999996, far above the 0.99 gate.
- **float32** error (rel-RMS ≈ 4.4e-4 ≈ 2⁻¹¹, ~2·10⁴ fp32 ULP at scale) is the signature of the
  FPU elementwise-add path quantizing SrcA/SrcB operands to the ~10-bit-mantissa (TF32-class)
  register format, even at HiFi4 with `fp32_dest_acc_en` — the same accumulation budget the
  all_reduce baseline measured (rel-RMS 7e-4 over N=8). Expected hardware behaviour, not an op
  defect; PCC is 1.0000000 and the error is far inside the 0.999 gate. Callers needing tighter
  f32 sums would need an SFPU-based add or multi-pass fidelity treatment — noted under
  Recommendations only, since no cell fails.

**Recommended tolerances** (match the golden suite / acceptance tests, generous headroom vs.
observed): bf16 `PCC ≥ 0.99`, float32 `PCC ≥ 0.999`; allclose `atol ≈ 0.13` (bf16, N=4 sums of
unit normals) / `0.02` (f32); golden `check_output` tolerance `(0.99, 0.05)` confirmed appropriate.

---

## Verifier CLI Summary

Artifact: `generated/reduce_scatter_verify/verifier_report.json` (from
`python3 -m eval.verify_supported generated/reduce_scatter_verify ttnn.operations.reduce_scatter`).
Golden registry suite: 3 INPUTS × {bf16, f32} × TILE × Linear = 6 cells.

- supported_pass:     **6**
- xfail_expected:     0   (empty — SUPPORTED already equals TARGET; `by_category.xfail_expected`
  iterated and confirmed empty, not just the count)
- invalid_skipped:    0   (INVALID = [])
- supported_fail:     **0**  ✓ ship gate
- xpass_drift:        **0**  ✓ ship gate
- xfail_wrong_mode:   **0**  ✓ ship gate
- supported_marked_xfail: 0
- no_axes_found:      5 — the five `test_translated.py` tests (translated suites carry no `axes`
  parametrize by design; all 4 in-support cells pass, and the Ring cell is the deliberate
  lenient-xfail refinement axis with reason
  `topology=Topology.Ring not in SUPPORTED [Topology.Linear]`). Not a loud category.

All loud categories are 0. The report is honest: SUPPORTED describes reality exactly.

**Test suites run (all on REAL (1,4) Blackhole hardware via the multi-device runner, aggregate
exit 0):**
- Acceptance: `tests/.../reduce_scatter/test_reduce_scatter.py` — **13 passed** (4 shapes × 2
  dtypes + negative-dim alias + program-cache + output_tensor + 2 rejection tests).
- Extended (new): `test_reduce_scatter_extended.py` — **3 passed** (L1-interleaved end-to-end,
  multi-core mid-row slice walk, typed refusals + output-spec mismatch).
- Precision baseline (new): `test_reduce_scatter_precision_baseline.py` — **8 passed**.
- Golden: `eval/golden_tests/reduce_scatter/` — **10 passed, 1 xfailed** (6 registry cells +
  4 translated passes + the Ring lenient-xfail).
- Final combined directory run: **24 passed**.

---

## Refinement-queue audit (mandated TARGET − SUPPORTED accounting)

Every `(axis, missing_value)` pair from `TARGET − SUPPORTED`:

| Axis | TARGET | SUPPORTED | Missing values |
|---|---|---|---|
| dtype | [bfloat16, float32] | [bfloat16, float32] | ∅ |
| layout | [TILE] | [TILE] | ∅ |
| topology | [Linear] | [Linear] | ∅ |

**The set is empty** — there is no refinement candidate to file, no INVALID entry needed to excuse
one, and therefore no queue gap: the empty `xfail_expected` bucket is the *consequence* of
SUPPORTED == TARGET, verified cell-by-cell above, not an unexamined success signal.
`op_requirements.md` records Phase 0 complete with an explicitly empty queue.

---

## Recommendations

The directions below are **beyond the current TARGET** and are therefore *not* refinements (a
refinement can only move SUPPORTED toward TARGET). Each would require `/golden-tests` to expand
`feature_spec.py`'s TARGET first; listed so a future TARGET-expansion pass has the map, in
suggested adoption order:

1. **Ring topology** (`ttnn.Topology.Ring`) — the only direction with a live (lenient-xfail)
   golden cell already waiting: `test_translated.py::test_ring_reduce_scatter_refinement_axis`
   flips to a pass with no test edit once adopted. A ring reduce-scatter is an
   algorithm-fundamental change (reduce-en-route with `RingRsSchedule`/`LineChannelWalk` +
   `BlockAccumulate` + `SyncCadence`, per the silicon-verified `reduce_scatter_minimal_async`
   model) — it would be a standalone refinement, not a bundle.
2. **`dim ∈ {1, 2}` (and 0)** — `slice_tile_offset` in the shared schedule header already covers
   dims 1–3; the Phase-B reader would plumb `slice_C`/`slice_Ht` and drop its `dim == 3`
   static_assert. Dim 0 is a batch-block scatter (dense pages — simpler than dim 3). The op-level
   `_SUPPORTED_DIMS` gate moves into `SUPPORTED["dim"]` **only together with** a feature_spec that
   sweeps the axis (see the load-bearing harness note under Code Review).
3. **Traffic-optimal Phase A** — relay only slices still needed downstream (each hop sheds 1/N) or
   full reduce-en-route; cuts hop-bytes ~N× and removes the N×-shard `gather_buffer`. Pure
   performance; no failing cell.
4. **Multi-link / multi-worker Phase A** — the helper exposes `MuxConn<N>`; the op uses a single
   link with one worker per direction. Throughput lever only.
5. **Sharded input** (`validate()` rejects with `ValueError`) — a memory-config expansion needing
   sharded readers/writers; currently out of pipeline scope.
6. **float32 sum tightness** — if a future consumer needs better than the TF32-class operand
   quantization measured above, the lever is an SFPU add or higher-fidelity accumulation
   treatment; no concrete lever is in scope today and no cell fails, so this stays a note.
7. **Shared `_SEMAPHORE_CACHE` keying cleanup** (this op + all_reduce): weakref-based keying to
   remove the theoretical `id()` reuse hazard — cross-op cleanup, not an op refinement.
