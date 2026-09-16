# Verification Report: reduce_scatter_average

**Op class**: multi-device CCL **with a compute stage** (single-dispatch fused fabric line gather +
arrival-ordered TRISC N-way sum + 1/N broadcast-scalar scale + per-device-distinct slice output).
**Verified on**: REAL 4-chip Blackhole hardware, mesh `(1, 4)` with `FabricConfig.FABRIC_1D`
(topology `bh_quietbox_1x4_hw`), via `scripts/run_multidevice_sim_pytest.py --runtime hardware
--op reduce_scatter_average`. This is the correct runner for a CCL op — `run_safe_pytest.sh`
forces slow dispatch on sim and has no multichip/hang awareness. The runner's `--list` gate
confirms `reduce_scatter_average` is registered in the topology matrix (sim entry
`wh_t3k_allmmio_reduce_scatter_average` at `(1, 8)`; the sim `.so` + cluster descriptors are not
staged on this box, so hardware is the active runtime). Aggregate exit = 0 on every run below.

---

## Code Review

The implementation is a faithful realization of `op_design.md` — the single-dispatch fused
gather+reduce with arrival-ordered overlap, exactly as designed. Review found **no correctness
defects in the op code or kernels**. The only fixes were in the **golden test harness**, which the
design itself had flagged as defective (`op_design.md` §"Structural impossibilities", items 1–2).

### Fixed (golden harness — `eval/golden_tests/reduce_scatter_average/helpers.py`)

1. **`NameError` on every golden cell**: the driver called `reduce_scatter(...)` — an undefined
   name in that module (only `reduce_scatter_average` is imported). Fixed to call
   `reduce_scatter_average(ttnn_input, dim=scatter_dim, topology=topology)`.
2. **SUM oracle contradicting the op's MEAN semantics**: the oracle built
   `.sum(dim=0)` with no `/ N`, contradicting the module's own docstring ("MEAN-THEN-SLICE") and
   the op spec. PCC (scale-invariant) would have passed but the `rms` half of
   `tolerance=(0.99, 0.05)` would have failed every in-SUPPORTED cell. Fixed to
   `.mean(dim=0)` (fp32-accumulated, cast once at the end).
3. **Stale docstrings** in the same file (referred to the op as `reduce_scatter`, pointed the
   runner at `--op reduce_scatter`) — updated.

These are harness-side fixes; `feature_spec.py` was NOT touched (verifier contract). Both defects
were pre-existing harness authoring errors, not op regressions — the fixed suite passes 10/10
in-SUPPORTED cells with zero op changes.

### Reviewed and intentionally left as-is (op code)

- **Fabric egress is fully helper-managed** (`ccl_helpers_dataflow.hpp`): `FabricStreamSender<>`
  ctor from the rt-arg cursor (advanced past the conn block) → `open(unicast_route(num_hops))` →
  `arm_unicast_write(page_size)` / `arm_inc(1)` → `write_page` / `inc` ×2 per block → `close()`.
  The two counting incs per block (receiving relay core + receiving reduce core) ride the SAME
  connection in-order behind the block's pages (T4/R8) — this is the design's overlap mechanism,
  implemented exactly. `noc_async_writes_flushed()` before every `cb_pop_front` (R7) present.
- **Receive-side sync is op-owned by design**: relay readers use `noc_semaphore_wait_min` + the
  R1 cache-reuse re-arm (`noc_semaphore_set(sem, 0)` after the final wait, on every role including
  pure line-end receivers); the reduce reader's TWO-WAY poll (volatile reads +
  `invalidate_l1_cache()`, whichever direction lands first is consumed first) is the design's
  documented raw-API case — the helper header explicitly scopes receive-side sync out, and a
  single-counter `noc_semaphore_wait_min` would serialize the two directions and destroy the
  overlap. Both `sem_fwd`/`sem_bwd` re-armed after all arrivals observed (safe: no inc can still
  be in flight once observed).
- **Compute is exactly the designed helper pipeline**: `binary_op_init_common` (kernel-owned hw
  startup) → `BlockAccumulate::arm(cb_contributions, cb_accumulator, cb_accumulator, g)` → C1 seed
  via `sum_blocks(…, num_blocks=1, g, pop_input=true)` (R4 load-bearing, correctly set) →
  `acc.rearm()` (R3 — restores after sum_blocks's acc_to_dest post-condition) → (N−1)×S/g
  `acc.run(g)` (in-place `cb_b == cb_out`, sound per the helper's pop-before-reserve verified
  ordering with capacity exactly S) → C4 raw `mul_tiles_bcast_scalar` pass with `init_short`
  strictly AFTER the last `run()` (R10). The raw C4 is design-justified: the designated eltwise
  broadcast helper (`eltwise_convenience.hpp`) is **absent from this clone at HEAD** (verified —
  `kernel_lib/` has no eltwise headers), and every other helper candidate is rejected with
  citations in `op_design.md` §"Raw-API justifications".
- **Broadcast usage**: `cb_scaler` carries one persistent page with only element (0,0) written
  (bf16 via `generate_bcast_unary_scalar`, fp32 via the mirrored raw one-word store — the helper
  assumes 16-bit elements), waited once with count 1, never popped; SCALAR-bcast multiply reads
  only (0,0). No full-tile fill of repeated data anywhere.
- **CB sync balance** (per the design ledger, re-audited per core role): `cb_relay_pages`
  `num_sends·P` push == pop (0 == 0 on idle line-end directions); `cb_contributions` N·S push ==
  S + (N−1)·S pop; `cb_accumulator` compute-only single producer (R2), N·S push == (N−1)·S + S
  pop/wait; `cb_scaler` 1 push, one count-1 wait, never popped; `cb_averaged` S == S. Every CB
  capacity is a multiple of its quantum (wrap-contiguity, R6); `g` divides `S` by host
  construction so no tail chunk exists (R5), `g ≤ DEST_AUTO_LIMIT = 4` statically asserted.
- **`TensorAccessor` everywhere** (no deprecated `InterleavedAddrGen`), `void kernel_main()` in
  all five kernels (including compute), `api/dataflow/dataflow_api.h` includes — all current
  style. `static_assert(is_supported_scatter_dim(dim))` guarded per R9, plus an explicit
  `static_assert(dim == 3)` documenting the Phase-0 pin (Refinement 1 removes it).
- **Host assembly**: one `ttnn.generic_op` per invocation (the prompt's hard single-dispatch
  mandate); fabric conn blocks appended post-construction via the live `runtime_args` view (the
  builder mutates the program) and placed FIRST; `route.num_hops == 1` asserted per direction;
  `route.is_forward` from `ccl_dm_route` (never hand-derived); GlobalSemaphores created once per
  mesh_device with ONE `synchronize_device` in the miss branch, parked on
  `mesh_pd.semaphores` (attribute confirmed present at runtime — the `hasattr` guard matches the
  hardware-validated reference idiom), no per-call post-dispatch barrier. `gather_buffer`
  allocated fresh per call and passed in `io_tensors` (R14).
- **C4 `pack_tile` loop**: `pack_tile(t, cb_averaged)` ×g in sequential in-order mode is the
  design's pinned API mapping and matches the documented contract. `pack_tile_block(0, cb, g)`
  would be a one-call equivalent — cosmetic only; the hardware-verified shape is kept.

### Deferred to refinements (architectural, not fixable in this pass)

- `dim=2` scatter: genuinely unimplemented (kernel `static_assert(dim == 3)`; the reduce reader's
  walk is dim-3-shaped). → Refinement 1.
- `Ring` topology: relay indices are already ring-modular (T3) but the Linear send/arrival depth
  tables, route selection, and double-delivery split are Linear-only. → Refinement 2.

---

## Registry Conformance

Confirmed all four declarations present and correctly wired in
`ttnn/ttnn/operations/reduce_scatter_average/reduce_scatter_average.py`:

- **`INPUT_TAGGERS = {}`** — empty by design (every golden INPUT is tile-aligned by construction;
  no shape-derived axis). `validate()` still iterates it with the correct `tagger(inputs, axes)`
  call shape, so a later tagger is a drop-in.
- **`SUPPORTED = {dtype: [bfloat16, float32], layout: [TILE], topology: [Linear], dim: [3]}`** —
  covers every axis the kernels gate on, including both op-specific kwargs. `dim` is a SUPPORTED
  key even though single-valued (the feature-spec's hard requirement), positive convention, with
  `dim % 4` canonicalization BEFORE the membership test (`-1 ≡ 3` hardware-tested in the extended
  suite; `-2 ≡ 2` verified to raise the typed refusal). (`memory_config` DRAM/L1-interleaved is
  accepted but not a gated categorical axis — matching the reduce_scatter/all_reduce precedent;
  both verified working on hardware in the extended suite.)
- **`EXCLUSIONS = []`** — present, empty (no in-SUPPORTED cell is refused).
- **`validate()`** — ordering is the verifier-blessed reference pattern out of the box: universal
  structural (ValueError: MeshDevice, (1,N) line view, N ≥ 2, rank 4, dim range, not-sharded, H/W
  tile-aligned) → axis gate (`UnsupportedAxisValue` per axis, then `ExcludedCell`) →
  axis-value-dependent structural (slice divisibility, page-size guard, L1 accumulator budget,
  output-tensor spec). Refusal types from `ttnn.operations._op_contract` (module confirmed
  present; the ImportError fallback subclasses `NotImplementedError`). The public entry point
  calls `validate()` on its first line, before any allocation or dispatch.
- **No `INVALID` symbol in the op file** — confirmed absent (INVALID is a feature-spec concept).

**Drift**: none. `xpass_drift = 0`, `supported_fail = 0`, `xfail_wrong_mode = 0` on the first
clean run after the harness fix. Unlike the reference reduce_scatter, there is no hidden
under-claim here: `dim=2` is structurally unimplemented in the kernel (compile-time pinned), not
merely un-listed.

### INVALID audit (`eval/golden_tests/reduce_scatter_average/feature_spec.py`)

`INVALID = []`, and this is **correct** for the current TARGET:

- `TARGET = {dtype: [bf16, f32], layout: [TILE], topology: [Linear, Ring], dim: [3, 2]}` — every
  combination is constructible. INPUTS keep dims 2 AND 3 at multiples of 256 so both scatter dims
  stay tile-aligned on both the (1, 8) sim line and the (1, 4) hardware box — the spec's own note
  says this deliberately avoids an INPUT_TAGGER + INVALID pair.
- **Single-tensor op** — no cross-tensor-axis coupling risk.
- **Canonical bf8b + ROW_MAJOR entry correctly omitted** — TARGET contains neither bfloat8_b nor
  ROW_MAJOR, so the entry would reference axis values the harness never generates.
- **No norm-like weight axes** → no no-weight canonicalization cells needed.

No changes to `feature_spec.py` proposed.

### Design conformance

Checked against `op_design.md` on the binding dimensions — all match:

| Dimension | Design | Implementation |
|---|---|---|
| Algorithm | SINGLE-dispatch fused gather + ARRIVAL-ORDERED incremental reduce + 1/N scale epilogue; explicitly NOT the reference's two-dispatch split | one `generic_op`, own-contribution pass starts immediately, per-arrival passes overlap fabric flight, scale after last pass ✓ |
| Pipeline topology | (0,0) fwd relay, (0,1) bwd relay, (0,2) reduce reader/compute/writer; 7 kernel descriptors/program | matches ✓ |
| Work distribution | fixed 3-core roles; single reduce core (multi-core reduce deliberately deferred, S ≤ 32 on golden shapes) | matches ✓ |
| Inter-core comm | TWO GlobalSemaphores (fwd/bwd), TWO fabric incs per block (relay + reduce core) in-order behind the pages, R1 re-arm on every consumer | matches ✓ (program-cache tests pass on hardware — the R1 trap) |
| CB protocol | 5 CBs per the design table, capacities exact multiples of quanta, `cb_accumulator` capacity exactly S, single producer | matches ✓ |
| Overlap contract | arrival-major C1–C3 (R15: position-major would serialize) | matches ✓ |

### Prompt rules

`eval/prompts/reduce_scatter_average.txt` has no `## Rules` section — stock policy applies. Its
hard mandates are all satisfied: self-contained op, zero imports/wrapping of existing CCL ops
(verified imports); exact import path + signature; registry contract with typed refusals
(hardware-verified in golden + extended suites); ONE `generic_op` dispatch with compute
overlapping fabric arrival (the design's T4/T7 mechanism, implemented); output_tensor path returns
the supplied handle (hardware-verified); program-cache hit with surviving GlobalSemaphores
(hardware-verified); loud ValueError on unsplittable shapes (hardware-verified).

---

## Precision Baseline

Measured on the `(1, 4)` Blackhole line (N = 4 contributions + 1/N scale), dim=3, oracle = shards
quantized to the device dtype first, mean accumulated in fp32, cast once. Metrics are
**worst-device** values. From
`tests/ttnn/unit_tests/operations/reduce_scatter_average/test_reduce_scatter_average_precision_baseline.py`
(8/8 pass). `max ULP@scale` = max |err| in units of the ULP spacing at the tensor's max magnitude.

| Shard shape | dtype | PCC (worst dev) | Max Abs Err | Mean Abs Err | Relative RMS Err | Max ULP@scale |
|-------------|-------|-----------------|-------------|--------------|------------------|---------------|
| (1,1,32,256)  | bfloat16 | 0.9999952 | 0.007812 | 0.000917 | 0.003590 | 1 |
| (1,1,32,256)  | float32  | 0.9999998 | 0.002775 | 0.000357 | 0.000992 | ~23k |
| (1,1,256,256) | bfloat16 | 0.9999955 | 0.015625 | 0.000879 | 0.003479 | 2 |
| (1,1,256,256) | float32  | 0.9999998 | 0.004001 | 0.000349 | 0.000975 | ~26k |
| (1,1,64,512)  | bfloat16 | 0.9999954 | 0.015625 | 0.000900 | 0.003554 | 1 |
| (1,1,64,512)  | float32  | 0.9999998 | 0.002956 | 0.000353 | 0.000988 | ~12k |
| (2,1,32,256)  | bfloat16 | 0.9999953 | 0.007812 | 0.000907 | 0.003581 | 1 |
| (2,1,32,256)  | float32  | 0.9999998 | 0.003581 | 0.000351 | 0.000996 | ~24k |

**Assessment**: Excellent and shape-stable. **bf16** worst error is 1–2 ULP of the OUTPUT at its
own scale — i.e. exactly bf16 storage quantization of the mean; rel-RMS ≈ 0.0035 is the bf16
mantissa budget for an N=4 accumulate + scale (cf. the reduce_scatter reference's ≈ 0.0027 for the
plain sum; the extra ~30% is the 1/N multiply pass's added rounding at the smaller output
magnitude). **float32** rel-RMS ≈ 1.0e-3 (~2⁻¹⁰ relative, ≈ 12–26k fp32 ULPs at scale) is NOT
fp32-rounding-limited: fp32 operands are truncated in the FPU srcA/srcB registers (~10-bit
mantissa) on the add and multiply paths even with `fp32_dest_acc_en` — consistent with the
reduce_scatter reference's 4.4e-4 for sum-only, roughly doubled here by the extra multiply pass.
This is a hardware datapath property, not an op bug; every PCC is ≥ 0.999995, far above the
golden thresholds.

**Recommended tolerances** (match the golden suite / acceptance test): bf16 `PCC ≥ 0.99`,
float32 `PCC ≥ 0.999`; allclose `rtol = 0.05`, `atol ≈ 0.1` (bf16) / `0.02` (f32). Generous
headroom vs. observed.

---

## Verifier CLI Summary

Artifact: `generated/reduce_scatter_average_verify/verifier_report.json` (from
`python3 -m eval.verify_supported generated/reduce_scatter_average_verify
ttnn.operations.reduce_scatter_average`). Golden universe: 3 INPUTS × {bf16, f32} × TILE ×
{Linear, Ring} × dim {3, 2} = 24 cells (+ 5 translated-suite tests without axes sidecars →
`no_axes_found`, all at their expected status: 4 passed + 1 Ring lenient-xfail).

- supported_pass:     **6** (3 INPUTS × 2 dtypes × Linear × dim=3)
- xfail_expected:     **18** — 6 `dim=2 × Linear`, 6 `dim=3 × Ring`, 6 `dim=2 × Ring`
- invalid_skipped:    0   (INVALID = [])
- supported_fail:     **0**  ✓ ship gate
- xpass_drift:        **0**  ✓ ship gate
- xfail_wrong_mode:   **0**  ✓ ship gate
- supported_marked_xfail: 0

**Gap accounting** (`TARGET − SUPPORTED`, iterated from `by_category.xfail_expected` — every
`(axis, missing_value)` pair accounted for):

| (axis, missing_value) | Cells | Disposition |
|---|---|---|
| (dim, 2) | 12 (6 Linear + 6 Ring-overlap) | **Refinement 1** in `op_requirements.md` |
| (topology, Ring) | 12 (6 dim=3 + 6 dim=2-overlap) | **Refinement 2** in `op_requirements.md` |

The 6 `dim=2 × Ring` cells sit in the intersection: Refinement 1 moves them from
"refused on dim" to "refused on topology"; Refinement 2 then unlocks them.

**Test suites run (all on real (1,4) Blackhole hardware via
`run_multidevice_sim_pytest.py --runtime hardware`, aggregate exit 0):**

- Acceptance dir: 13 passed, 11 skipped — the immutable `(1, 8)`-pinned acceptance file
  self-skips on the 4-chip box by design; the mesh-adaptive debug mirror (same op body,
  `CCL_HW_MESH_SHAPE`-sized) carries the hardware verification: all-ones/chip-index deterministic
  cases, 8 shape×dtype cells, program-cache hit (R1 trap), output_tensor handle, ValueError
  rejection.
- Extended (verifier-authored): `test_reduce_scatter_average_extended.py` — 5 passed
  (dim=-1 alias, L1-interleaved bf16 + f32 in/out, fp32 output_tensor path, typed-refusal triple
  Ring / dim=2 / dim=-2).
- Precision baseline: `test_reduce_scatter_average_precision_baseline.py` — 8 passed.
- Golden: `eval/golden_tests/reduce_scatter_average/` — 10 passed, 19 xfailed (xfail-strict
  clean).

---

## Recommendations

**Refinement queue** (see `op_requirements.md`): two refinements — **dim=2 scatter** then
**Ring topology** — closing the entire `TARGET − SUPPORTED` gap. dim=2 first: it is the local,
infra-risk-free change, and Ring's verification sweep then covers both dims in one pass.

**Performance / resource observations** (no failing cell → not refinements; several are subsumed
by a future true-ring algorithm):

- **Full-shard gather traffic**: the relay moves N·P pages per device while the reduce consumes
  only N·S = P slice tiles. A slice-only relay (or the true ring reduce-scatter) cuts fabric
  traffic ~N×. Design refinement-candidate 4; no failing cell.
- **`gather_buffer` footprint**: N × shard bytes of DRAM, allocated fresh per call (R14).
- **L1 accumulator cliff**: `cb_accumulator` = S pages resident on the reduce core;
  `validate()` rejects S > 256 with a loud ValueError (design's conservative budget: fp32
  S=256 → ~1.1 MB of the 1.5 MB L1). Golden/acceptance max is S=32 — no cell approaches the
  cliff. Lifting it (accumulator chunking or DRAM spill; design candidate 5) has no failing cell
  today; `/memory-budget-metal`'s streaming patterns apply when a larger-S TARGET lands.
- **Single reduce core**: multi-core reduce (design candidate 3) multiplies the per-block inc
  fan-out; golden S ≤ 32 makes one core right-sized. Not `/interleaved-parallel` territory
  (cross-core arrival signaling is a real data dependency).
- **Packet coalescing** (`ccl_packet_dims` multi-page packets + per-chunk incs; design
  candidate 6) — available, deliberately unused in the 1:1 page↔packet framing all four
  reference collectives ship with.
- **C4 micro-cleanup**: `pack_tile_block(0, cb_averaged, g)` could replace the g-iteration
  `pack_tile` loop — cosmetic; the hardware-verified design-pinned shape is kept.

**Beyond-TARGET directions** (each requires `/golden-tests` to expand `feature_spec.py`'s TARGET
first; NOT refinements today): dim ∈ {1, 0} scatter, ROW_MAJOR layout (tilize-wrapped reduce-path
readers — `/memory-layouts`), bfloat8_b (`/numeric-formats-metal`), sharded memory (validate()
rejects with ValueError today), 2-D mesh lines (validate() pins the `(1, N)` view), N not a power
of 2 (1/N inexact in bf16 — R12; fp32 scaler already exact).
