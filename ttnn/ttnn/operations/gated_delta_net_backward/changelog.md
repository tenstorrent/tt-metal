# Changelog: gated_delta_net_backward

## Phase 0 — Core Implementation

- **Date**: 2026-09-17
- **Device**: Blackhole, 11 × 10 = 110 worker cores, `max_worker_l1_unreserved = 1 531 904 B`
- **What was done**: Initial implementation via the incremental pipeline (planner → implementer →
  verifier). One `ttnn.generic_op` dispatch computes all six gradients, internally phased into
  P (prep, grid-parallel over `(bh, chunk)`) → per-`(b,h)`-group semaphore fan-in → S (the forward
  and reverse state scans, sequential in the chunk index, one core per `(b,h)`) → per-group fan-out
  → G (gradient assembly, grid-parallel, V-accumulated in L1). The `(I−A)⁻¹` UT inverse is built by
  Neumann doubling; the backward uses the algebraic inverse VJP `dA = Tinvᵀ·d_attn·Tinvᵀ` and never
  re-runs the forward substitution. The four `[C,C]` constant masks are generated in the reader at
  boot with raw L1 stores, so no `ttnn.zeros`/`ttnn.eye` becomes a second dispatch.

- **SUPPORTED at Phase 0**:
  `dtype=[float32, bfloat16]`, `layout=[TILE]`, `state_mode=[do_only, with_h0, with_h0_and_dht]`,
  `chunk_size=[32, 64]`, `seq_alignment=[chunk_aligned, chunk_ragged]`,
  `head_dims=[square, wide_v]`. `EXCLUSIONS = []`.
  This is the **whole** of TARGET except `dtype=bfloat8_b`, which `feature_spec.py` declares
  INVALID — so there is no generality gap left and the refinement queue is one precision entry
  followed by measured perf work.

- **Accuracy achieved** (measured on 4 shapes × 2 dtypes × 6 gradients via
  `test_gated_delta_net_backward_precision_baseline.py`, against the float64 autograd oracle):
  worst `PCC = 0.99999077`, worst `rel-RMS = 8.1e-3`, worst `max_abs_err = 8.0e-1` (on `dk`/`dg`,
  which carry the largest dynamic range), worst `mean_abs_err = 7.1e-2`. Bands are
  fp32 PCC 0.999 / RMS 0.02 and bf16 PCC 0.99 / RMS 0.12, so every gradient sits 2.5–14× inside
  its gate. bfloat16 is not materially worse than float32, which confirms the design's choice to
  keep every internal CB `Float32` regardless of input dtype.
  `got/true` ratio spread is broad and centred near 1.0 (p5/p95 ≈ ±2%), i.e. rounding noise and not
  a scale bug — with a systematic 0.4–0.8% low bias recorded in the report as the signature of
  truncating behaviour at the FPU source-register width.

- **Perf at Phase 0** (warm cache, best-of-3 `min`, `test_..._perf_baseline.py`; the real-time
  device profiler is inactive in this build so this is warm host-to-host latency for a
  single-dispatch op): 0.429 ms `(2,64,4,64,64)` c32 · 0.593 ms `(1,100,2,64,64)` c64 ·
  0.685 ms `(1,128,2,64,128)` c64 · 0.709 ms `(1,512,2,64,64)` c32 · 1.574 ms
  `(1,256,4,128,256)` c64 bf16 · 1.731 ms `(1,256,4,128,256)` c64 fp32.

- **Golden suite at Phase 0** (per `verifier_report.json`): **145 passing** of 147 runnable
  (218 collected − 66 INVALID-skipped − 5 host-only harness failures).
  `xfail_expected = 0`, `xpass_drift = 0`, `xfail_wrong_mode = 0`, `supported_marked_xfail = 0`.
  `supported_fail = 2` — both the same saturated-gate (`g_scale = 8.0`) `numerical-precision` cell,
  deliberately left failing per the registry-model routing rule and owned by Refinement 1.

- **Issues encountered** (all fixed in this verification pass unless noted):
  1. **Env-var override of math fidelity removed** — `_compute_config()` read `GDN_FID` from the
     environment as the default `math_fidelity`. A knob that changes the answer must travel with
     the call, not the shell. Default is now the literal `HiFi4`, overridable only via
     `compute_kernel_config`.
  2. **DRY violation fixed (host ↔ kernel)** — the seven uniform CB block sizes
     (`MAXV, LVB, LITEM, MAXBLK, MAXBLK_G, NCOL, NCONST`) and the three named-slot counts were each
     written out twice, once per language, held together by a `// mirrors _cb_pages()` comment.
     They are dependent quantities of the block extents and a divergence is a fifo wrap (a hang).
     Now defined once on the host (`_cb_blocks()` + `NUM_CONST_MASKS`/`NUM_VECA_SLOTS`/
     `NUM_VECB_SLOTS`) and passed to the kernels as CT args 40..49, with `static_assert`s tying the
     kernel's slot layout to the host's counts.
  3. **Latent overrun fixed** — the reader's gather pipeline sized its per-slot in-flight counter
     as `issued[2]` while indexing it by `w % GATHER_DEPTH`; raising the depth knob would have
     written past it. Now `issued[GATHER_DEPTH]`.
  4. **Block helper adopted** — `tr_blk()` paid a full DEST handshake per tile; it now uses
     `transpose_block()`, the uniform block entry point of the transpose op group, over
     `DEST_LIMIT`-sized groups. Measured 1.813 → 1.726 ms at `(1,256,4,128,256)` fp32,
     1.620 → 1.567 ms bf16, 0.468 → 0.439 ms at `(2,64,4,64,64)`; neutral at `Ct = 1`.
  5. **Dead debug scaffolding removed** — `STAGE_MASK` plus the `DO_P/DO_S/DO_G` stage gates and
     the `ABLATE_GATHER`/`ABLATE_COMPACT` bits. Unreferenced by any test, and these kernels are
     kernel-config-ring-buffer bound, so the branches and the CT arg were real cost.
  6. **Dead code and a stale comment** — the unused NoC address in `zero_l1()`, the unused `ntiles`
     in `gather_block()`, and a comment claiming the program descriptor configures
     `UnpackToDestFp32` on `cb_gatein` (it does not exist as a descriptor field; the datacopy init
     selects unpack-to-DEST from the operand's DEST format, which is the mechanism the `g` path
     actually relies on).
  7. **`l1_ledger.md` currency** — `cb_veca` was documented as `6*Ct` pages with six slots where
     the code allocates `5*Ct` with five. The error propagated into the closed form and all five
     rows of the footprint table. Corrected; the table now matches **device-measured** peak L1
     (`metric.device_l1_peak_bytes`) to 0.1 KB — 1396 KB predicted vs 1396.1 KB measured at
     `(1,128,2,64,128)` c64 fp32. Audit 2's mechanism attribution was also corrected: the ~tf32
     resolution it observes comes from the FPU source registers of a `Float32` page's *consumer*,
     not from the packer — which is what makes Refinement 1's approach the right one and the op
     file's suggested coarse+fine `decay` pair the wrong one.
  8. **Measured and reverted** — restricting the reader's per-block zero-fill to the ragged case
     (removing ~250 KB per item of local NoC traffic at the largest shape) measured
     **neutral to 1.03× worse**: this reader is RISC-issue bound, not bandwidth bound. Reverted,
     with the measurement recorded at the site so a perf refinement does not re-attempt it blind.
  9. **Not fixed, reported** — five host-only oracle tests in the golden suite's
     `test_regression.py` fail in graded runs with `L1ProfilingError` because
     `eval_test_runner.sh` exports `EVAL_CAPTURE_L1=1` and those tests take no `device` fixture.
     They pass standalone, land in the uncharged `no_axes_found` bucket, and are a harness/golden-
     suite defect, not an op defect; not edited, since making graded tests pass is not the
     verifier's call.
  10. **Documented omissions** — `H ≤ 32` is a hard `ValueError` (the source page-index formula
      assumes `ceil(H/32) == 1`); it is not a TARGET axis and no INPUTS entry exceeds `H = 8`, so
      it is neither a queue entry nor drift, but a >32-head model would hard-error.
      `PROPERTIES["math_fidelity"]` claims all four fidelities with only `HiFi4` exercised — folded
      into Refinement 1's test matrix.

- **Tests added**:
  `test_gated_delta_net_backward_precision_baseline.py` (4 shapes × 2 dtypes × 6 gradients: PCC,
  max/mean abs error, relative RMS, and the `got/true` ratio spread that separates a scale bug from
  rounding noise) and `test_gated_delta_net_backward_perf_baseline.py` (the config-spanning
  no-regression guard set every perf refinement is gated against — one representative per distinct
  kernel path × dtype, with the instrument's ~3% run-to-run drift documented in-file).
  Pre-existing: `test_gated_delta_net_backward.py` (50 acceptance cases, all passing) and
  `test_gated_delta_net_backward_debug.py` (the scratch-readback harness).

- **Refinements filed**: 4 — one precision (owns the two failing cells), then three measured perf
  phases ordered hardest-first: the cross-core V-split that addresses 70–96% grid idleness, the
  per-item fixed-cost amortization on the low-occupancy shapes, and the `block_val_tiles` floor at
  `Kt = 4`. See `op_requirements.md`.
