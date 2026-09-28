# Verification Report: all_gather

**Op kind:** multi-device CCL (collective communication) — pure cross-chip byte
movement, no arithmetic (identity gather, PCC ≈ 1.0). Built as a self-contained
Python `ttnn.generic_op` + `ttnn.MeshProgramDescriptor` with newly-authored
fabric ring/line dataflow kernels (reader NCRISC + writer BRISC per worker core).

**Verification date:** 2026-06-25

---

## TL;DR

- **On-device verification PASSED.** Unlike `point_to_point` (whose BH sim was
  fabric-blocked), `all_gather`'s WH sim works. The op ran green on the
  deterministic `wh_t3k_allmmio_all_gather` sim (`(1,8)` line, `FABRIC_1D`):
  **22/22 acceptance** cases + **8/8 precision** cases, aggregate exit 0. The
  cross-device gather actually executed and PCC asserted — this is observed
  behavior, not code review alone.
- **Precision is exact.** all_gather is a bit-for-bit identity copy: every
  measured cell shows `max_abs = mean_abs = rel_rms = 0` (ATOL/RTOL delta 0.0)
  for both bf16 and f32. The only "error" present is the dtype quantization that
  already happened at `from_torch`, before the op ran — the transfer adds none.
- **Program-cache footgun cleared on hardware.** The two-call program-cache test
  passed, proving the cache-reuse semaphore re-arm (reader resets to 0 at the
  end) is correct *without* the design's optional startup barrier (see Design
  Conformance).
- **Code review:** no correctness bugs found. One clarity fix applied (a vacuous
  sub-clause in the page-alignment check). Registry conformance is fully correct
  as shipped (the `alignment` axis + tagger that had to be added to
  `point_to_point` during *its* verification is already present here).
- **Refinement queue is non-empty.** `SUPPORTED ⊊ TARGET` on three axes:
  `dtype` (missing `bfloat8_b`), `gather_dim` (missing `-3,-2,-1`), `topology`
  (missing `Ring`). All three are filed in `op_requirements.md`. The
  `bf8b × ROW_MAJOR` region is correctly INVALID (skipped), not queued.

---

## Code Review

### Fixed

1. **Vacuous sub-clause in the page-alignment gate** (`all_gather.py:validate`).
   The check read `if page % 16 != 0 and page != 16:`. The `and page != 16`
   clause is dead — when `page % 16 != 0` is true, `page` can never equal 16
   (`16 % 16 == 0`). Simplified to `if page % 16 != 0:` and added a comment
   explaining *why* the check is load-bearing: the fabric writer sends
   `align(page_size, l1_alignment)` bytes per page
   (`ccl_helpers_dataflow.inl:35`), while the output `TensorAccessor` spaces
   pages by the raw `page_size`, so a non-16-aligned page would let the
   rounded-up on-wire payload overrun into the next output page. Requiring
   16B-aligned pages makes the round-up a no-op. (No behavior change; clarity +
   documents an invariant the kernels silently depend on.)

### Reviewed clean (no change needed)

- **CB sync (push == pop) — balanced in every regime.** `cb_relay_pages`
  (idx 16): when `my_num_targets > 0`, the reader pushes `P` seed pages +
  `num_relay_blocks × P` relay pages; the writer pops `(num_relay_blocks + 1) × P`
  — equal. When `my_num_targets == 0` (line-end in that direction), the reader
  takes the pure-receiver branch (pushes nothing) and the writer early-returns
  (pops nothing) — also balanced. The seed push and relay read-back are
  *correctly gated* on `num_targets_<dir> > 0`, exactly as the design's "CB
  push/pop balance at line ends" risk requires. ✔
- **`cb_self_copy` (idx 24) is scratch, correctly unbalanced.** The forward
  reader `cb_reserve_back(.., 1)` once and reuses the write pointer across all
  `P` pages (read input → local self-copy write to own output block); there is
  no cross-kernel consumer, so it is deliberately not push/pop balanced. ✔
- **Store-and-forward ordering via the counting semaphore.** Each upstream
  writer forwards blocks in canonical slice order (seed `i`, then `i-1, i-2, …`
  for forward flow) and issues one `AtomicIncChannel::inc` per block; the
  downstream reader `noc_semaphore_wait_min(sem, running)` before each
  read-back, so the `running` count maps 1:1 onto block arrival order (fabric
  in-order delivery lands the payload before the inc). Increment counts match
  exactly: device `i`'s forward writer issues `i+1` incs; device `i+1`'s forward
  reader waits for `i+1`. Verified by hand for both directions and both line
  ends, and confirmed on the sim. ✔
- **Counting-sem isolation per `(device, core)`.** The forward core's sem is
  touched only by the immediate forward-flow upstream; the backward core's sem
  only by the immediate backward-flow upstream (`fwd_noc`/`bwd_noc` target the
  same logical core on the neighbour). No cross-talk. ✔
- **Cache-reuse re-arm.** The reader resets the counting sem to 0 after its last
  wait (`all_gather_reader.cpp:101`), and the `GlobalSemaphore` is created with
  initial value 0 and reused across program-cache hits. The two-call
  program-cache acceptance test passed on the sim — the "green run 1, hang run
  2" footgun is cleared. ✔
- **Helper usage (fabric egress).** The writer drives the safety-by-construction
  helper exactly as intended: `FabricStreamSender<> → open → arm_unicast_write /
  arm_inc → write_page / inc → drain → close`. The line-end writer opens no
  connection (early `return` on `my_num_targets == 0`), mirroring the
  reference's `valid_targets` gating. The raw-API fallbacks (`noc_async_read`
  ingress, local self-copy `noc_async_write`, `noc_semaphore_wait_min/set`) are
  precisely the pieces the helper banner (`ccl_helpers_dataflow.hpp:63-86`) says
  the op must own. No helper is under-used or bypassed. ✔
- **API correctness.** `void kernel_main()` (not the deprecated namespace
  pattern); includes use `api/dataflow/dataflow_api.h`; addressing uses
  `TensorAccessor` (not deprecated `InterleavedAddrGen`). ✔
- **Fabric arg contract.** The writer reads 7 scalar RT args (idx 0–6), then the
  `[has_forward][fwd?][has_backward][bwd?]` block at `conn_arg_idx = 7`, matching
  the host `_append_fabric_rt_args` layout; the leading `has_forward` flag is
  peeked as `dst_is_forward`. Fabric args are appended only on the forwarding
  writer (`num_targets_<dir> > 0`). ✔
- **`validate()` shape.** First line of the public entry point; structural input
  checks (MeshDevice, `(1,N)` line view, ≥2 devices, interleaved-only, 16B page
  alignment, output-spec equality) → per-axis `SUPPORTED` gate (incl. taggers) →
  `EXCLUSIONS`. Raises typed `UnsupportedAxisValue` / `ExcludedCell`. ✔

### Benign, well-justified deviations from `op_design.md` (no action)

- **No startup barrier (design Phase 1 `arm_multicast_inc`).** The writer omits
  the N-party startup barrier the design specifies and relies solely on the
  Phase-2 counting semaphore for cross-device ordering, plus the reader's
  end-of-kernel reset (+ the `GlobalSemaphore`'s initial 0) for cache re-arm.
  **This is a sound simplification, not a bug.** (1) The counting sem already
  provides the only ordering the op needs (data-then-inc, fabric in-order), and
  the persistent pre-allocated output makes "early" fabric writes correct. (2)
  Omitting the barrier *avoids* the helper's documented SHARED-SEM-HEADER footgun
  (`arm_multicast_inc` and `arm_inc` share one pooled header and cannot be live
  at once) entirely — there is no block-scoping hazard to get wrong. (3) It is
  empirically correct: all 30 sim cases pass, **including the two-call
  program-cache test**, which is exactly the scenario the barrier+reset dance
  exists to protect. Adding the barrier now would add risk for no correctness
  benefit. Documented in the changelog. (If a future refinement ever needs a
  hard startup fence, the design's mandate-compliant fallback is a second parked
  `GlobalSemaphore`, not the shared-header barrier.)
- **No compute (TRISC) kernel wired.** `kernels/all_gather_compute.cpp` exists
  only to document the deliberate absence (CCL = pure data movement); it is not
  referenced by the descriptor. Inherent to a no-arithmetic collective. ✔

### Minor observations (not fixed — churn risk on green kernels; recorded here)

- **Forward reader reads each input page twice** — once into `cb_self_copy` for
  the local self-copy, once into `cb_relay_pages` for the fabric seed. A fusion
  (read once, self-copy from the relay slot, then push) would halve the forward
  reader's input reads, but it entangles the line-end gating (the line-end
  forward device self-copies but must *not* push to `cb_relay`), reintroducing
  the exact CB-balance edge case the current clean split avoids. Pure perf, no
  failing cell — left as-is (mirrors the `point_to_point` precedent of not
  churning verbatim-from-reference CCL kernels).
- **`noc_async_writes_flushed()` per page in the writer** serializes the fabric
  egress (flush after every `write_page`). With the double-buffered `cb_relay`,
  flushing per chunk-pair would overlap more, but the per-page flush is
  conservative and provably safe (the CB slot must not be reused until the
  fabric sender has drained it). Perf, not correctness.
- **`ring_size` CT arg is currently unused** in both reader and writer (the
  Linear slice-walk uses `my_chip_id` + `num_relay_blocks`). It is a deliberate
  placeholder for the Ring refinement's modulo slice-walk; removing it now would
  churn the CT-arg indices and host, only to be re-added by Refinement 3. Left
  in place.

---

## Registry Conformance

- **INPUT_TAGGERS** — present: `{"alignment": tag_alignment}`, signature
  `(inputs, axes)`. The tagger reads the per-device shard's last two dims (both
  `% 32 == 0 → tile_aligned`). Correct for a byte-mover that copies padded tiles
  / RM rows verbatim. ✔
- **SUPPORTED** — present; declares every gated axis: `dtype`, `layout`,
  `topology`, `gather_dim` (negative-canonicalized index axis), `alignment`. ✔
- **EXCLUSIONS** — present, empty (`[]`). No cell inside SUPPORTED is refused. ✔
- **validate()** — present, first line of `all_gather(...)`; checks SUPPORTED
  per-axis (running taggers generically) then EXCLUSIONS; raises the
  registry-model refusal types. `gather_dim` is canonicalized to negative
  *before* the membership test (so `gather_dim=0` ≡ `-4` is accepted). ✔
- **Op file does NOT declare INVALID** — confirmed; INVALID lives only in
  `feature_spec.py`. ✔

### INVALID audit (`eval/golden_tests/all_gather/feature_spec.py`)

`INVALID = [{"dtype": ttnn.bfloat8_b, "layout": ttnn.ROW_MAJOR_LAYOUT}]`.

- **Single-tensor coupling:** `dtype` and `layout` both describe the *input*
  tensor — no cross-tensor coupling. ✔
- **Universe-must-change:** `bfloat8_b` is a block-quantized tiled format with no
  row-major representation — ttnn cannot construct `{bf8b, ROW_MAJOR}`. A
  data-format-definition impossibility, not a not-yet-implemented EXCLUSION. ✔
- **Canonical bf8b+ROW_MAJOR entry present** (the mandatory entry for a tile-or-RM
  byte-mover). ✔
- Not a norm-like op → no weight/canonicalization cells expected. ✔

Verdict: INVALID is well-formed; no change recommended. It correctly skips 64 of
the 384 golden cells (all `bf8b × ROW_MAJOR`).

---

## Precision Baseline

`tests/ttnn/unit_tests/operations/all_gather/test_all_gather_precision_baseline.py`
— ran green on the WH sim (8/8). Each device's output is compared against the
host-side concat of all 8 shards along `gather_dim=0`.

| Shard shape | full (gathered) | dtype | PCC | Max Abs Err | Mean Abs Err | Rel RMS Err |
|-------------|-----------------|-------|-----|-------------|--------------|-------------|
| (1,1,32,32)   | (8,1,32,32)   | bf16 | 1.0 | 0 | 0 | 0 |
| (1,1,32,32)   | (8,1,32,32)   | f32  | 1.0 | 0 | 0 | 0 |
| (1,1,64,128)  | (8,1,64,128)  | bf16 | 1.0 | 0 | 0 | 0 |
| (1,1,64,128)  | (8,1,64,128)  | f32  | 1.0 | 0 | 0 | 0 |
| (1,1,96,64)   | (8,1,96,64)   | bf16 | 1.0 | 0 | 0 | 0 |
| (1,1,96,64)   | (8,1,96,64)   | f32  | 1.0 | 0 | 0 | 0 |
| (1,1,256,256) | (8,1,256,256) | bf16 | 1.0 | 0 | 0 | 0 |
| (1,1,256,256) | (8,1,256,256) | f32  | 1.0 | 0 | 0 | 0 |

**Assessment:** all_gather is a *bit-for-bit byte copy*. Every device's output
equals the oracle exactly (zero abs/RMS error, ATOL/RTOL delta 0.0) — the
transfer introduces no error, and all N devices agree (replicated output). PCC is
identically 1.0 across all measured cells.

**Recommended tolerances** (identity oracle): PCC ≥ 0.9999 (f32) / 0.999 (bf16) /
0.99 (bf8b once added); `atol`/`rtol` effectively 0 for float dtypes — the suites
gate on PCC, matching the acceptance thresholds.

---

## Verifier CLI Summary

The standard `eval/eval_test_runner.sh` → `eval.verify_supported` flow cannot run
mechanically for this op: CCL golden dirs ship only `feature_spec.py` (no
`test_golden.py`/`helpers.py`/`conftest.py` — those are authored upstream by
`/golden-tests`), and the harness cannot stand up the `(1,8)` WH `FABRIC_1D` mesh
the golden cells require. So `eval/results/all_gather/verifier_report.json` is
**HYBRID**: the golden-cartesian categories are computed with the harness's own
logic (`eval.feature_matrix`: `cartesian` + `invalid_reason` + `is_supported` +
`unsupported_reason`), and the SUPPORTED rectangle's representative cells are
**observed passing on the WH sim** (the 22 acceptance + 8 precision cells above).

- supported_pass: 32  (`gather_dim=-4 × Linear × {bf16,f32} × {TILE,RM} × 8 shapes`;
  the directly-run subset is `observed_pass_sim`, the rest same kernel path)
- xfail_expected: 288  (`TARGET − SUPPORTED`; see queue mapping below)
- invalid_skipped: 64  (all `bf8b × ROW_MAJOR`)
- supported_fail: 0      (✓ — 0 by construction *and* 0 observed: 30/30 sim cells green)
- xpass_drift: 0         (✓ — no SUPPORTED under-claim found)
- xfail_wrong_mode: 0    (✓)
- total: 384

### `xfail_expected` → refinement mapping (every missing pair is covered)

Missing `(axis, value)` pairs from `TARGET − SUPPORTED` and their refinement:

| Missing pair | xfail cells | Covered by |
|--------------|-------------|------------|
| `dtype = bfloat8_b` | 64 | **Refinement 1** (bf8b TILE only; `bf8b×RM` is INVALID) |
| `gather_dim = -3` | 80 | **Refinement 2** |
| `gather_dim = -2` | 80 | **Refinement 2** |
| `gather_dim = -1` | 80 | **Refinement 2** |
| `topology = Ring` | 160 | **Refinement 3** |

(Cells with multiple out-of-SUPPORTED axes are counted once per axis above; they
clear cumulatively as the refinements land. No `xfail_expected` cell is left
without a queue entry — no queue gap.)

---

## Recommendations

1. **Refinement order (see `op_requirements.md`):** bf8b first (cheapest,
   independent, completes the dtype axis), then `gather_dim != 0` (strided concat
   addressing), then Ring (the most involved routing/algorithm change; sequence
   it after `gather_dim` so the ring slice-walk is validated against an already
   stable strided-addressing path). None of these map onto a current
   implementation-skill (the skill inventory covers single-device compute
   precision / layouts / multi-core / L1 budget — not CCL fabric axis expansions),
   so all three are verifier-authored with full goal + done-when.
2. **Re-verify each refinement on the WH sim, not silicon.** This host is
   single-device; the *only* multi-device path is
   `scripts/run_multidevice_sim_pytest.py --op all_gather` (topology
   `wh_t3k_allmmio_all_gather`, `required=true`, validated working). The op's
   tests MUST open exactly `(1,8)` + `FABRIC_1D` or fabric init hangs.
3. **Out-of-TARGET scope items (NOT refinements — would need `/golden-tests` to
   widen TARGET first):**
   - *Sharded memory config.* `validate()` rejects sharded input
     ("interleaved only"). TARGET has no `memory_config` axis, so this is not a
     refinement candidate today.
   - *Multi-link / multi-core fan-out per direction.* Single worker core per
     direction by design; a perf enhancement with no SUPPORTED axis or failing
     cell to point at.
   Both are deliberately kept out of the refinement queue.
4. **Perf (no failing cell — not queued):** the forward reader's double input
   read and the writer's per-page fabric flush (see Minor observations) are the
   two obvious throughput levers if a future perf pass is commissioned.
