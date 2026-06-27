# Verification Report: matmul (2D dual-multicast)

Phase 0 verification of the fused `C = A @ B` op (registry model). Verdict:
**ships.** The registry-driven golden suite is clean (0 in every loud
category); the precision baseline is excellent; one documented helper-usage
fix was applied to the compute kernel.

---

## Code Review

### Fixed in this pass

- **Compute kernel — use `matmul_block`'s `batch` param instead of an external
  per-block loop** (`kernels/matmul_compute.cpp`). The kernel previously called
  `matmul_block<>(...)` inside `for (i < total_blocks)` with `MatmulBlockShape`
  `batch=1`, which re-issues `mm_block_init_short` once per output block. Since
  every per-core output block is identical work with no per-block compute phase
  in between, the helper's documented matmul-only pattern is to pass the block
  count as `batch`: the helper inits once, then loops the full K-pipeline
  `total_blocks` times internally — strictly fewer init MMIO re-issues, byte-for-byte
  identical CB push/pop counts (so the multicast lock-step is unchanged).
  Changed to a single `matmul_block<>` call with `batch=total_blocks`.
  **Verified:** acceptance 16/16 and golden 20/20 supported cells still pass
  (incl. the large multi-block shapes where `total_blocks > 1` exercises the new
  internal batch loop), precision unchanged.

### Reviewed, correct as-is (no change needed)

- **Registry conformance.** `INPUT_TAGGERS` (both taggers carry the `(inputs, axes)`
  signature), `SUPPORTED`, `EXCLUSIONS`, `validate()` all present and wired.
  `validate()` checks structural shape-contract (ValueError) → SUPPORTED per-axis
  (`UnsupportedAxisValue`) → EXCLUSIONS (`ExcludedCell`), and is the entry point's
  first call. The op file does **not** declare `INVALID` (correct — it lives in
  feature_spec.py).
- **Helper usage.** Multicast is `dataflow_kernel_lib::SenderPipe`/`ReceiverPipe`
  (`mcast_pipe.hpp`), not raw `noc_async_write_multicast`+semaphores. Compute is
  `compute_kernel_lib::matmul_block` (no hand-rolled `matmul_tiles`/spill loop).
  DRAM tile gather + output scatter use `TensorAccessor` (not deprecated
  `InterleavedAddrGen`) — the design's "helpers considered and rejected" section
  correctly justifies raw `TensorAccessor` here (no kernel-lib helper covers
  interleaved per-tile addressing with edge clamping).
- **Kernel syntax / includes.** `void kernel_main()`, `api/dataflow/dataflow_api.h`
  (not the bare header), `api/compute/...` — all correct.
- **CB sync.** push == wait per CB, per output block, over `total_blocks`:
  in0 (`block_M_tiles*in0_block_w` × `num_k_blocks`), in1
  (`in0_block_w*block_N_tiles` × `num_k_blocks`), out (full block ×1), interm
  (helper-owned, distinct index 24). Verified against the design's CB-sync table.
- **DEST sizing.** `_choose_subblock` caps `out_subblock_h*out_subblock_w` at
  `_dest_limit(fp32_acc, full_sync)` (= 4 for fp32+`fp32_dest_acc_en`+half-sync),
  satisfying the prompt MUST rule.
- **Bounded L1 for arbitrary M/N/K.** The descriptor's footprint-shrink loop plus
  the per-core block loop keep CBs constant-sized in `in0_block_w`/`block_*_tiles`
  (never sized to full `Kt`/`Mt`/`Nt`). Empirically confirmed: every large golden
  shape passes with **no OOM** — 8192×8192, 4096×16384, and wide-K
  4096/8192 (see below). The "OOM is a correctness failure" MUST rule holds.

### Advisory / deferred (architectural — not changed)

- **`cb_interm` format is hardcoded `float32` and the L1 footprint estimate uses
  `tileA_bytes` for every CB** (`matmul_program_descriptor.py`). Correct and
  consistent for Phase 0 (fp32, `tile_size=4096` everywhere). The numerical-config
  refinement (bf16/bf8b) must make the interm format dtype-aware and fix the
  footprint estimate (interm stays fp32 under `fp32_dest_acc_en`, so its bytes
  must be counted separately from a bf16 `tileA_bytes`). Flagged in the refinement's
  verifier notes — not a Phase 0 bug.
- **Reader constructs `SenderPipe`/`ReceiverPipe` per-K-block.** Matches the design
  (proven-safe `pipe_rotating.cpp` pattern). Left as-is; correct under `--dev` and
  production timing.

---

## ⚠ Operational hazard found during verification — kernel binary cache staleness

`matmul` dispatches via `ttnn.generic_op` with a `ProgramDescriptor` built from
kernel **source paths**. The on-disk JIT kernel-binary cache
(`$TT_METAL_CACHE/.../kernels/matmul_{compute,reader,writer}/<hash>/`) keys on
path + compile-time args + defines — **NOT on the kernel `.cpp` source content.**
Consequence: **editing a kernel `.cpp` does not bust the cache**; the previously
compiled binary is silently reused, and the JIT-server precompile warm pass makes
this worse (it serves the stale binary to the graded run).

I proved this by injecting a deliberate compile error: with the cache warm the
test still **passed** (stale binary); after `rm -rf` of the three
`matmul_*` kernel cache dirs the cold recompile correctly **failed** to compile.

**Implication for every refinement here (all touch kernels):** an implementer
who edits a kernel and re-runs tests will see them pass on the *old* binary and
believe the change landed. **Before testing any kernel edit, delete the cached
kernels:**

```bash
find "$TT_METAL_CACHE" -type d \( -name matmul_compute -o -name matmul_reader \
  -o -name matmul_writer \) -exec rm -rf {} +
```

and run with `--no-precompile` (and `--no-jit-server` for the golden runner) for
the authoritative pass. This is the single most important thing the next
implementer must know; it is repeated at the top of `op_requirements.md`.

---

## Registry Conformance

- **Confirmed present & wired:** `INPUT_TAGGERS` {`tag_alignment`, `tag_weight_batch`},
  `SUPPORTED` (6 axes), `EXCLUSIONS` (`{float32, fp32_dest_acc_en=False}`),
  `validate()` (called first in `matmul()`). Op file does **not** declare INVALID. ✓
- **No auto-fixes to SUPPORTED were needed** — there was no `xpass_drift` on the
  registry-driven `test_golden.py` (the one `xpass_drift` in the raw verifier
  output is a translated-test artifact, see below).
- **INVALID audit** (`eval/golden_tests/matmul/feature_spec.py`, `INVALID = []`):
  well-formed and correctly empty.
  - *Canonical bf8b+ROW_MAJOR rule is vacuous here:* `TARGET["layout"]` is
    `TILE_LAYOUT`-only (matmul is tiled-FPU by design; no ROW_MAJOR in the
    universe), so no ROW_MAJOR cell exists in the cartesian to forbid.
  - *No cross-tensor-axis coupling:* `dtype` (activation) and `weight_dtype`
    (weight) describe different tensors and are independent — every
    `dtype × weight_dtype` combination is meaningful, so none are INVALID.
  - *`{float32, fp32_dest_acc_en=False}`* is legal-but-lossy → an op-side
    EXCLUSION (refinable), **not** INVALID (universe-must-change test: only a
    precision-policy decision, not an impossibility). Correctly placed.
  - *No norm-like no-weight canonicalization needed:* matmul's weight is a real
    second operand, always present.

---

## Precision Baseline

float32 activation + float32 weight, HiFi4, `fp32_dest_acc_en=True` (the sole
Phase 0 corner). Reference: `torch.matmul` in fp32. Measured by
`tests/ttnn/unit_tests/operations/matmul/test_matmul_precision_baseline.py`.

| Shape (A @ B) | PCC | Max Abs Err | Mean Abs Err | Relative RMS Err |
|---|---|---|---|---|
| (32,64)@(64,32)       | 0.99999981 | 0.0373 | 0.00752 | 0.00122 |
| (256,512)@(512,1024)  | 0.99999972 | 0.2016 | 0.03057 | 0.00171 |
| (1024,1024)@(1024,1024) | 0.99999751 | 63.90 | 0.06279 | 0.00316 |
| (512,4096)@(4096,4096) | 0.99999414 | 16.17 | 0.37006 | 0.00729 |

**Assessment.** PCC ≥ 0.99999 across the ladder; relative RMS error stays well
below the golden (float32, True) band of 0.02 even at K=4096. Max-abs error grows
with K (expected — it tracks the magnitude of deep fp32 dot products), but the
*relative* RMS is flat and tiny, confirming the fp32-DEST accumulation is faithful.
(`comp_allclose` reports large rtol deltas because some reference elements are
near-zero cancellations — pointwise rtol is not a meaningful metric for matmul;
PCC + relative RMS are.)

**Recommended tolerances (match the golden bands):** PCC ≥ 0.999, relative
RMS ≤ 0.02 for the (float32, fp32_dest_acc_en=True) corner.

---

## Verifier CLI Summary

Artifact: `verifier_report.json` (in this dir and in the results dir
`/tmp/matmul_verify2/`). Source run: `eval/eval_test_runner.sh --no-precompile
--no-jit-server eval/golden_tests/matmul/` (cold, current kernels).

Raw totals across the whole golden directory (894 collected — includes
`test_golden.py` + `test_regression.py` + `test_translated.py`):

- supported_pass: 34
- xfail_expected: 826
- invalid_skipped: 0
- **supported_fail: 1**
- **xpass_drift: 1**
- **xfail_wrong_mode: 1**
- no_axes_found: 31 (regression/translated tests not driven by the registry)

**The registry-driven `test_golden.py` is the suite the verifier governs, and it
is fully clean:**

| Category (test_golden.py only) | Count |
|---|---|
| supported_pass | 20 (all tile-aligned + single + fp32/fp32/acc=True cells, incl. all large/wide-K shapes) |
| xfail_expected | 646 |
| **supported_fail** | **0** ✓ |
| **xpass_drift** | **0** ✓ |
| **xfail_wrong_mode** | **0** ✓ |
| supported_marked_xfail | 0 ✓ |

### The 3 loud signals are all `test_translated.py` artifacts (not registry drift, not kernel bugs)

`test_translated.py` holds reference tests translated from production `ttnn.matmul`
tests. They are **not** xfail-decorated and **not** part of the registry
parametrization; they get categorized only because `axes.observed` captures their
runtime axes. Each was inspected:

1. **supported_fail** — `test_bert_linear_batch4_fp32_input_output[1536-4096-1024-LoFi-fp32]`.
   Uses **LoFi** `math_fidelity` on fp32 inputs with magnitude ×1000 and K=4096.
   `math_fidelity` is deliberately **not gated** (prompt rule: any value accepted),
   so all *registry* axes are in SUPPORTED — but LoFi truncates the FPU mantissa,
   so fp32×fp32 over K=4096 drifts past this translated test's tight Frobenius
   threshold. The matmul_block header itself states fp32 needs HiFi4. The golden
   suite always runs HiFi4, so this is a non-axis (fidelity) precision artifact of
   one translated case, **not** a registry over-claim and **not** a kernel bug. No
   action (cannot gate fidelity; golden never exercises LoFi).
2. **xpass_drift** — `test_matmul_same_shape_but_invalid[input_a0-input_b0]`.
   Feeds A=(1,1,1,32), B=(1,1,1,32) → A[-1]=32 ≠ B[-2]=1 (K-mismatch) and asserts
   `pytest.raises((RuntimeError, ValueError))`. The op **correctly rejects** → the
   test **passes**. The verifier sees "outside SUPPORTED + passed" (the shape tags
   as bf16/m_non_aligned) and labels it `xpass_drift`, but the op did **not**
   compute an unsupported cell — it raised, exactly as the test wanted. False drift
   signal; no action.
3. **xfail_wrong_mode** — `test_matmul_with_transpose_and_configs[1-2-4096-32-256]`.
   After the squeeze the weight is (2,32,256) — a **batched** weight
   (`weight_batch=batched`), which `validate()` correctly refuses with
   `NotImplementedError` (a Phase 0 limitation → Refinement 3). The translated test
   isn't xfail-decorated, so the refusal counts as a plain failure → `xfail_wrong_mode`.
   The op behaves correctly; this cell becomes a real pass once Refinement 3 lands.

None require changing SUPPORTED, EXCLUSIONS, or the kernel.

---

## Recommendations

1. **Refinement ordering.** Do Refinement 1 (numerical config) first — it
   establishes dtype-aware CB-format/footprint derivation that Refinements 2
   (alignment) and 3 (batched) build on. See `op_requirements.md`.
2. **bf16 + HiFi4 cross-cutting concern (Refinement 1).** The golden harness
   (`eval/golden_tests/matmul/helpers.py:run_matmul`) hardcodes `MathFidelity.HiFi4`
   for **all** dtypes. The `matmul_block` header warns: *"AVOID HiFi4 +
   fp32_dest_acc_en with bf16 inputs — silent K-accumulator corruption on Wormhole
   B0 (issue #38306); use HiFi2/HiFi3."* So bf16 cells will be tested at
   HiFi4+acc=True, the exact corruption-prone combination. The dtype-refinement
   implementer must account for this (and the golden harness fidelity for bf16 may
   need a `/golden-tests` adjustment — out of the implementer's scope to edit
   `helpers.py`). Captured in Refinement 1's verifier notes.
3. **interm-CB precision (Refinement 1).** Make `cb_interm` format and the L1
   footprint estimate dtype-aware (currently fp32-hardcoded; fine for Phase 0).
4. **Kernel-cache staleness (all refinements).** See the operational-hazard box
   above — clear the `matmul_*` kernel cache before testing any kernel edit, or the
   change silently no-ops on a stale binary.
5. **No memory/parallelism refinement is warranted.** The op is already multi-core
   (2D grid) and already bounds L1 for arbitrary M/N/K (all large golden shapes
   pass). No `/memory-budget-metal` or standalone `/interleaved-parallel` work
   needed.
