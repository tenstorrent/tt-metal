# Verification Report: reduce_scatter

**Op class**: multi-device CCL **with a compute stage** (fabric line gather + TRISC N-way tile SUM +
per-device-distinct slice output).
**Verified on**: REAL 4-chip Blackhole hardware, mesh `(1, 4)` with `FabricConfig.FABRIC_1D`
(topology `bh_quietbox_1x4_hw`), via `scripts/run_multidevice_sim_pytest.py --op reduce_scatter`.
This is the correct runner for a CCL op — `run_safe_pytest.sh` forces slow dispatch on sim and has
no multichip/hang awareness. The runner's `--list` gate confirms `reduce_scatter` is registered in
the topology matrix. Aggregate exit = 0 on every run below.

---

## Code Review

The implementation closely follows `op_design.md` and the prompt's framework-owner mandates: a
gather-then-reduce-local-slice algorithm across two ordered `ttnn.generic_op` dispatches on one
command queue (Phase A fabric store-and-forward gather → Phase B SliceRowWalker-addressed local
N-way `sum_blocks` reduce). Review found no correctness defects in the kernels; the fixes below are
host-side hardening plus one capability promotion.

### Fixed

1. **`SUPPORTED["dim"]` under-claimed reality — promoted `[3]` → `[3, 2]`** (drift fix-in-place).
   Reading the code showed the dim=2 machinery was already fully implemented: the host slice rows
   (`_slice_quantities` carries dim 3, 2, AND 1), the Phase-B reader's
   `static_assert(is_supported_scatter_dim(dim))` (accepts 1|2|3), and the `SliceRowWalker`
   run/stride math all generalize over the scatter dim — only the SUPPORTED membership list gated
   dim=2 behind `UnsupportedAxisValue`. This is the hidden-under-claim variant of `xpass_drift`
   (validate() refuses before the kernel could XPASS, so the harness can't see it). Promoted and
   **mechanically verified on hardware**: the 6 golden `dim=2 × Linear` cells now pass
   (`supported_pass` 6 → 12), plus 6 new verifier-authored extended tests (dim=2 bf16/f32,
   multibatch, `-2` alias, dim=2 program-cache hit). The immutable acceptance test's
   `test_reduce_scatter_unsupported_dim_refuses` self-skips as designed once dim=2 is refined in.
2. **`validate()` ordering — axis gate moved before axis-value-dependent structural checks.**
   Previously the scatter-dim divisibility check (`shape[dim] % (N*32)`), the H/W tile-alignment
   check, and the output-spec check ran BEFORE the SUPPORTED membership test. An out-of-SUPPORTED
   axis value whose shape trips a dependent check (e.g. `dim=1` with `C=1`, or a ROW_MAJOR input
   with sub-tile H) would raise `ValueError` instead of the registry contract's typed
   `UnsupportedAxisValue`. Reordered: universal structural checks (MeshDevice, line view, N ≥ 2,
   rank, dim range, not-sharded) → axis gate (SUPPORTED per-axis, then EXCLUSIONS) →
   axis-value-dependent structural checks (tile alignment, slice divisibility, page-size guard,
   output spec). Unobservable in the current golden universe (every INPUT is valid for every TARGET
   dim), but the refusal type is now correct for ALL out-of-SUPPORTED values regardless of shape.
   Re-verified: `xfail_wrong_mode = 0`, and `test_reduce_scatter_rejects_indivisible_shape` (an
   in-SUPPORTED cell with a bad shape) still gets its `ValueError`.
3. **Signature typing**: `output_tensor: ttnn.Tensor = None` → `ttnn.Tensor | None = None`
   (matches the prompt's exact signature).

### Reviewed and intentionally left as-is

- **Fabric egress uses the CCL kernel helper** (`ccl_helpers_dataflow.hpp`): `FabricStreamSender<>`
  (declared before the stream — the one lifetime the types don't check) → `open(unicast_route)` →
  `arm_unicast_write` / `arm_inc` → `write_page` / `inc` → `close()`. The staged open→arm→issue→close
  path is correct for a many-packet stream (`signal()` is terminal — rightly rejected in the design).
  The receive ingress, the counting `noc_semaphore_wait_min`, and the cache-reuse
  `noc_semaphore_set(sem, 0)` re-arm (receiver-after-wait, both relay and line-end paths — §R1) are
  op-owned per the helper's documented split. No raw multicast/hand-rolled handshake exists —
  nothing to migrate to `mcast_pipe.hpp`.
- **Phase-B compute is exactly the mandated helper**: `binary_op_init_common` (kernel-owned hardware
  startup, per the accum header's ownership note) + `compute_kernel_lib::sum_blocks(cb, cb_out, N, 1,
  pop_input=true)`. `pop_input=true` is load-bearing (§R7 — real producer/consumer CB; the default
  `false` would deadlock the reader) and correctly set. The helper owns wait(N)/pop(N)/reserve/push,
  the tile_regs lifecycle, DEST chunking, and the odd-N seed — no wrapper CB ops around it.
- **Slice addressing has ONE definition**: host `_slice_quantities` (the `slice_tile_offset` formula)
  → CT args → kernel `SliceRowWalker` with the design's seed formula
  (`reset_offsets(start % run, (start / run) * stride)`), and ONE `next()` per output position
  hoisted above the N-block read loop (§R5). Verified the dim-3 AND dim-2 inverse-map algebra by
  hand; hardware agrees.
- **CB sync balance** (per the design ledger): `cb_relay_pages` producer `(1+num_relay_blocks)·P`
  == consumer (same `if constexpr` guard both sides); `cb_self_copy` reserve-only scratch (0/0,
  intentional — §R9, do not "fix"); `cb_gathered_slices` `n·N` == `n × sum_blocks(N)`;
  `cb_summed_slice` `n` == `n`. Push == wait/pop on every CB, both phases, all core roles.
- **Line-end discipline**: the entire Phase-A writer body sits inside
  `if constexpr (my_num_targets > 0)` and line-end writers get an empty rt-arg list (§R10); the
  distinct rt-arg COUNT is a deliberately distinct program-cache hash per device role.
- **Block-order agreement**: writer sends seed `i` then relays `i∓k`; reader stages the same order;
  the k-th counting inc on the receiver corresponds exactly to the block the reader reads back —
  traced both directions, both interior and line-end roles.
- **`TensorAccessor`** (not deprecated `InterleavedAddrGen`), `void kernel_main()` in all three
  kernels (including compute — the current style, same as all_reduce), and
  `api/dataflow/dataflow_api.h` includes — all correct. The uniform 7-scalar CT superset with
  zero-padding and the `[[maybe_unused]]` documentation of intentionally-unread slots
  (`ring_size`, the reduce reader's second accessor) is exactly the design's shared-source rule.
- **Phase-B writer per-tile `noc_async_write_barrier()`** — byte-identical to the proven all_reduce
  Phase-B writer (the design pins this shape). A `noc_async_writes_flushed()` in-loop + single final
  barrier would be marginally cheaper; performance-only, no failing cell — left as the proven shape.

### Advisory (no fix; no failing cell to point at)

- **Phase-A self-copy is fully serialized** (read barrier + write barrier per page over P pages,
  forward reader only). Same advisory as all_reduce; runs once per device on the smallest data path.
- **Phase A moves full shards** (all_gather traffic, N·P pages per device) where only the S·N slice
  tiles are eventually consumed — the design's acknowledged cost of gather-then-reduce (§R12).
  The bandwidth-right fix is the ring algorithm (see Refinement 1 in `op_requirements.md`) or a
  slice-only relay; recorded under Recommendations, not filed as a refinement (no failing cell).

---

## Registry Conformance

Confirmed the four declarations are present and correctly wired in `reduce_scatter.py`:

- **`INPUT_TAGGERS = {}`** — empty by design (every golden INPUT is valid for every TARGET dim; no
  shape-derived axis). `validate()` still iterates it with the correct `tagger(inputs, axes)`
  call shape, so adding a tagger later is a drop-in.
- **`SUPPORTED = {dtype:[bfloat16, float32], layout:[TILE], topology:[Linear], dim:[3, 2]}`** —
  covers every axis the kernels gate on, including the op-specific `dim` and `topology` kwargs
  (the prompt's hard requirement: `dim` is a key even when single-valued). `dim` uses the positive
  convention with canonicalization (`-1 ≡ 3`, `-2 ≡ 2`) BEFORE the membership test (§R14) —
  both aliases hardware-tested. (`memory_config` DRAM/L1-interleaved is accepted but not a gated
  categorical axis, matching the all_reduce precedent; both verified working.)
- **`EXCLUSIONS = []`** — present, empty (no in-SUPPORTED cell is refused).
- **`validate()`** — axis gate raises `UnsupportedAxisValue` per-axis then `ExcludedCell` per-cell
  (correct order), from `ttnn.operations._op_contract` with the ImportError fallback subclassing
  `NotImplementedError`. The public `reduce_scatter()` calls `validate()` on its first line, before
  any allocation or dispatch.
- **No `INVALID` symbol in the op file** — confirmed absent (INVALID is a feature-spec concept).

**Drift**: one under-claim found and fixed in place (`dim=2`, see Code Review #1). After the fix the
verifier CLI reports `xpass_drift = 0`, `supported_fail = 0`, `xfail_wrong_mode = 0`.

### INVALID audit (`eval/golden_tests/reduce_scatter/feature_spec.py`)

`INVALID = []`, and this is **correct** for the current TARGET:

- `TARGET = {dtype:[bf16, f32], layout:[TILE], topology:[Linear, Ring], dim:[3, 2]}` — every
  combination is constructible (float dtypes on TILE, either topology, either dim; the INPUTS are
  deliberately multiples of 256 on dims 2 AND 3 so both scatter dims stay tile-aligned on both the
  (1, 8) sim line and the (1, 4) hardware box — the spec's own note says this avoids needing an
  INPUT_TAGGER + INVALID pair).
- **Single-tensor op** — no cross-tensor-axis coupling risk.
- **Canonical bf8b + ROW_MAJOR entry correctly omitted** — TARGET contains neither bfloat8_b nor
  ROW_MAJOR, so the entry would reference axis values the harness never generates.
- **No norm-like weight axes** → no no-weight canonicalization cells needed.

No changes to `feature_spec.py` are proposed.

### Design conformance

Checked against `op_design.md` on the binding dimensions — all match:

| Dimension | Design | Implementation |
|---|---|---|
| Algorithm | gather-then-reduce-local-slice, 2 ordered dispatches, queue order as phase barrier | Phase A fabric gather → Phase B slice-addressed N-way `sum_blocks` ✓ |
| Pipeline topology | Phase A: 2 workers/device (fwd `(0,0)` / bwd `(0,1)`), NCRISC reader + BRISC writer, no compute; Phase B: reader/compute/writer on the work grid | matches ✓ |
| Parallelization | Phase A 1 worker/direction; Phase B `split_work_to_cores(grid, S)` two-group split, `corerange_to_cores(row_wise=True)` accumulation | matches ✓ |
| Inter-core comm | ONE op-internal GlobalSemaphore, counting inc-after-block in-order on the connection, receiver re-arm §R1 | matches ✓ (program-cache tests exercise the re-arm, incl. a dim=2 cache-hit test) |
| Shared schedule | `SliceRowWalker` + host-evaluated `slice_tile_offset`, compile-time `is_supported_scatter_dim` gate | matches ✓ |

One deliberate post-design delta: `SUPPORTED["dim"]` includes 2 (the design's Phase-0 declared
`[3]` with dim=2 as a refinement candidate — the implementer built the general machinery, the
verifier promoted it on hardware evidence). `op_design.md` is left as the historical design record.

### Prompt rules

`eval/prompts/reduce_scatter.txt` has no `## Rules` (MUST/MUST NOT) section — it carries the
generation mandate and framework-owner guidance. All satisfied: (1) self-contained op, no
wrapping/importing of existing CCL ops (verified imports); (2) all THREE helper families composed —
fabric egress (`FabricStreamSender`), compute accumulation (`sum_blocks`), shared schedule header
(`SliceRowWalker`/`slice_tile_offset`/`is_supported_scatter_dim` as the ONE slice-addressing
definition); (3) per-device `MeshProgramDescriptor` entries; (4) GlobalSemaphore created once +
`synchronize_device` once + parked on the descriptor + no per-call post-dispatch barrier;
(5) `ccl_dm_route` + `setup_fabric_connection` for route/connection (no reimplemented packet
sizing — 1:1 page↔packet framing like all_gather/all_reduce, `ccl_packet_dims` legitimately
unused and its rejection documented in the design); (6) hardware startup in the kernel, `arm`-vs-run
granularity conflation avoided (`sum_blocks` free function, no `BlockAccumulate` coexists);
(7) registry contract — `dim` and `topology` exported as SUPPORTED keys; typed refusals verified by
18→12 clean `xfail_expected` cells across both runs. The prompt's "prefer DUPLEX for a bidirectional
ring shape" soft guidance does not apply to this algorithm (the two directions send DIFFERENT block
sequences from DIFFERENT cores — the design's rejection table covers this).

---

## Precision Baseline

Measured on the `(1, 4)` Blackhole line (N = 4 summands), dim=3, oracle accumulated in fp32 then
cast so the reference is not itself limited by bf16 rounding. Metrics are **worst-device** values
(the output is per-device distinct). From
`tests/ttnn/unit_tests/operations/reduce_scatter/test_reduce_scatter_precision_baseline.py`.

| Shard shape | dtype | PCC (worst dev) | Max Abs Err | Mean Abs Err | Relative RMS Err |
|-------------|-------|-----------------|-------------|--------------|------------------|
| (1,1,32,256)  | bfloat16 | 0.9999964 | 0.031250 | 0.002649 | 0.002731 |
| (1,1,32,256)  | float32  | 1.0000000 | 0.004516 | 0.000691 | 0.000445 |
| (1,1,64,512)  | bfloat16 | 0.9999963 | 0.031250 | 0.002692 | 0.002732 |
| (1,1,64,512)  | float32  | 1.0000000 | 0.004954 | 0.000671 | 0.000436 |
| (1,1,256,256) | bfloat16 | 0.9999964 | 0.031250 | 0.002634 | 0.002719 |
| (1,1,256,256) | float32  | 1.0000000 | 0.005202 | 0.000664 | 0.000437 |
| (2,1,64,256)  | bfloat16 | 0.9999964 | 0.031250 | 0.002673 | 0.002732 |
| (2,1,64,256)  | float32  | 1.0000000 | 0.004954 | 0.000671 | 0.000435 |

**Assessment**: Excellent and shape-stable. bf16 error is exactly the bfloat16 storage quantization
of the summed output (max-abs 0.03125 = one bf16 ULP at magnitude ~2–4, the scale of a sum of 4
unit-normal terms; rel-RMS ≈ 0.0027 ≈ the bf16 mantissa budget for N=4 — cf. all_reduce's ≈ 0.0087
at N=8, scaling as expected). float32 error is tiny (rel-RMS ≈ 4.4e-4), attributable to the HiFi4 +
`fp32_dest_acc_en` on-device accumulation order differing from torch — expected and negligible.
Every PCC is ≥ 0.999996, far above the golden thresholds.

**Recommended tolerances** (match the golden suite / acceptance test): bf16 `PCC ≥ 0.99`,
float32 `PCC ≥ 0.999`; `atol ≈ 0.1` (bf16) / `0.02` (f32) for allclose on N=4–8 sums. Generous
headroom vs. observed.

---

## Verifier CLI Summary

Artifact: `generated/reduce_scatter_verify/verifier_report.json` (from
`python3 -m eval.verify_supported generated/reduce_scatter_verify ttnn.operations.reduce_scatter`).
Golden universe: 3 INPUTS × {bf16, f32} × TILE × {Linear, Ring} × dim {3, 2} = 24 cells
(+ 5 translated-suite tests without axes sidecars → `no_axes_found`, all at their expected status).

- supported_pass:     **12** (3 INPUTS × 2 dtypes × Linear × dim {3, 2})
- xfail_expected:     **12** — ALL `topology=Ring` (× dim {3, 2} × 3 INPUTS × 2 dtypes)
- invalid_skipped:    0   (INVALID = [])
- supported_fail:     **0**  ✓ ship gate
- xpass_drift:        **0**  ✓ ship gate
- xfail_wrong_mode:   **0**  ✓ ship gate
- supported_marked_xfail: 0

**Gap accounting** (`TARGET − SUPPORTED`, per the queue contract — every pair accounted for):

| (axis, missing_value) | Cells | Disposition |
|---|---|---|
| (dim, 2) | 6 | **Closed in this pass** — promoted into SUPPORTED on hardware evidence (Code Review #1) |
| (topology, Ring) | 12 | **Refinement 1** in `op_requirements.md` |

**Test suites run (all on real (1,4) Blackhole hardware, aggregate exit 0):**
- Acceptance: `tests/.../reduce_scatter/test_reduce_scatter.py` — 12 passed, 1 skipped (the dim=2
  refusal test self-skips post-promotion, by its own design).
- Extended (verifier-authored): `test_reduce_scatter_extended.py` — 6 passed (dim=2 bf16/f32 +
  multibatch, `-2` alias, L1-interleaved in/out, dim=2 program-cache hit).
- Precision baseline: `test_reduce_scatter_precision_baseline.py` — 8 passed.
- Golden: `eval/golden_tests/reduce_scatter/` — 16 passed, 13 xfailed (12 golden Ring cells + 1
  translated Ring refinement cell), xfail-strict clean.

---

## Recommendations

**Refinement queue** (see `op_requirements.md`): exactly one open refinement — **Ring topology**
(the only remaining `TARGET − SUPPORTED` gap, 12 cells). It is algorithm-fundamental (the ring
communication schedule IS the work) and stands alone per the grouping rules.

**Performance observations** (no failing cell → not refinements; most are subsumed by the Ring
refinement if implemented as a true ring reduce-scatter):

- **Slice-only gather**: Phase A moves N·P pages per device where Phase B consumes only N·S = P of
  them. A slice-only relay (or the ring algorithm) reduces fabric traffic by ~N×.
- **gather_buffer footprint**: N × shard bytes of DRAM per call, allocated fresh per call. The ring
  algorithm needs only O(S) intermediate space.
- **Packet coalescing** via `ttnn._ttnn.fabric.ccl_packet_dims` (multi-page packets) — available,
  deliberately unused in the 1:1 page↔packet Phase-0 framing.
- **`sum_blocks` granularity > 1**: batching multiple output positions per call would amortize the
  per-call init; note §R11 (fp32_dest_acc halves DEST to 4) binds any granularity > 1.
- **Multi-link fabric** (`MuxConn<N>`): single link (`_LINK_IDX = 0`) today.
- Phase-A self-copy serialization and the Phase-B writer's per-tile barrier (see Code Review).

**Beyond-TARGET directions** (each requires `/golden-tests` to expand `feature_spec.py`'s TARGET
first; NOT refinements today):

- **dim=1 scatter**: the host row and the kernel `static_assert` already accept dim=1; but
  `validate()`'s divisibility check over-constrains it (`shape[1] % (N*32)` — C is not a tiled dim,
  the correct requirement is `C % N == 0` with the H/W tile alignment unchanged). Fix that check
  when a dim=1 TARGET lands.
- **ROW_MAJOR layout** (tilize-wrapped reader — `/memory-layouts`), **bfloat8_b** (dtype treatment
  on the reduce path — `/numeric-formats-metal`), **sharded memory** (validate() rejects with
  ValueError today), **2-D mesh lines** (validate() pins the `(1, N)` view).
