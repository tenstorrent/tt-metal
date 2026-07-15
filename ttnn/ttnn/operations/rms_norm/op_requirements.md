# Operation Requirements: rms_norm

## Definition
- **Formula**: `RMSNorm(x) = x / sqrt(mean(x², dim=-1, keepdim=True) + epsilon) * gamma`
- **PyTorch Reference**:
  ```python
  def rms_norm(x, gamma=None, epsilon=1e-6):
      xf = x.to(torch.float32)
      rms = torch.sqrt(torch.mean(xf**2, dim=-1, keepdim=True) + epsilon)
      out = xf / rms
      if gamma is not None:
          out = out * gamma.to(torch.float32).reshape(-1)
      return out.to(x.dtype)
  ```
- **Import Path**: `from ttnn.operations.rms_norm import rms_norm`
- **Function Signature**:
  ```python
  rms_norm(
      input_tensor: ttnn.Tensor,
      *,
      gamma: Optional[ttnn.Tensor] = None,
      epsilon: float = 1e-6,
      compute_kernel_config: ttnn.ComputeConfigDescriptor = None,
      memory_config: ttnn.MemoryConfig = None,
  ) -> ttnn.Tensor
  ```

## Gap analysis (TARGET − SUPPORTED)

| Axis | TARGET | SUPPORTED (Phase 0) | Gap | Where |
|------|--------|---------------------|-----|-------|
| dtype | f32, bf16, **bf8b** | f32, bf16 | bf8b | R2 |
| fp32_dest_acc_en | True, False | True, False (both `False` cells EXCLUDED) | drop `{bf16, False}` exclusion | R2 |
| layout | TILE, RM | TILE, RM | — | — |
| alignment | tile / w_non / h_non | all three | — | — |
| rank | 2, 3, 4 | 2, 3, 4 | — | — |
| gamma_mode | gamma, no_gamma | both | — | — |
| gamma_dtype | f32, bf16, **bf8b**, none | f32, bf16, none | bf8b | R2 |
| gamma_layout | TILE, RM, none | RM, none | **TILE** | R2 |
| memory_layout | INTERLEAVED, **HEIGHT**, **WIDTH**, **BLOCK** | INTERLEAVED | HEIGHT (R4); WIDTH, BLOCK (R5) | R4, R5 |

`{float32, fp32_dest_acc_en=False}` is a **permanent** EXCLUSION (fp32 requires fp32 accumulation) —
not a refinement. `bf8b + ROW_MAJOR` (input and gamma) is INVALID — not a refinement. Every other
gap maps to a refinement below.

## Phases

> **Non-regression rule**: Every refinement must pass all tests from prior phases.
> **Drift signal**: XPASS-strict failures mean the implementer added support but forgot to update SUPPORTED. Fix by updating SUPPORTED.
> **Checkbox protocol**: `[x]` complete + tests pass; `[~]` real work landed but a named axis value deferred (treated as completed, surfaced as partial); `[ ]` nothing usable produced.
> **Refinement ID + follow-up naming (the runner parses `Refinement \d+[a-z]?`)**: primary = `Refinement N`; a partial's sharper follow-up appends a letter (`Refinement 1b`), ordered immediately after its parent.

### [x] Phase 0 — Core Implementation

- **SUPPORTED dtype**: [float32, bfloat16]
- **SUPPORTED fp32_dest_acc_en**: [True, False] — both `False` cells EXCLUDED (`{f32,False}` permanent; `{bf16,False}` → R2)
- **SUPPORTED layout**: [TILE, ROW_MAJOR] (both native, no host-side conversion)
- **SUPPORTED shape-derived**: alignment ∈ {tile_aligned, w_non_aligned, h_non_aligned}; rank ∈ {2, 3, 4}
- **SUPPORTED gamma**: gamma_mode ∈ {gamma, no_gamma}; gamma_dtype ∈ {f32, bf16, none}; gamma_layout ∈ {ROW_MAJOR, none}
- **SUPPORTED memory_layout**: [INTERLEAVED]
- **Cores**: single (`CoreCoord(0,0)`), row-parallel streaming reduce; L1 bounded in `W`
- **Compute config**: `compute_kernel_config` exposed; `default_compute_kernel_config()` = HiFi4 + fp32_dest_acc_en=True
- **Golden baseline**: **463 / 472 supported cells passing** (9 supported_fail = fp32 W=8192 → Refinement 1). xpass_drift=0, xfail_wrong_mode=0.

### [x] Refinement 1 — Fix the fp32 Σx² reduce scale bug (BLOCKING)

**Goal**: move the 9 `float32 + W=8192` cells (ranks 2/3/4, gamma & no_gamma) out of `supported_fail`
into passing. The fp32 reduce has a **structural scale bias linear in W** (got/true ratio
`≈ 1 + 2.5e-6·W`, tight std ≈ 0.001 — see `verification_report.md` and
`test_rms_norm_precision_baseline.py`); the device mean(x²) is systematically undercounted ∝ W, so
the output is too large. This is NOT rounding noise (bf16 shows broad noise centred on 1.0), so it is
NOT a precision-surface refinement — do **not** reach for `/numeric-formats-metal`/fp32 intermediates
(the path is already fp32). No SUPPORTED axis changes.

**Verifier notes**: **This is the blocking refinement — land it before any other, because it touches
the Σx² reduce datapath that R2 (bf8b CB formats) and R4/R5 (sharded reduce) all build on.** The
runner stamps these `severity=precision` (PCC 0.999999, rel-RMS just over the 0.02 fp32 target); the
got/true-ratio analysis reclassifies them as a scale/structural bug — the failure metric is
interpretable, so do **not** silence via EXCLUSIONS or a `shape_size` tagger (that would delete the
signal and leave float32 quietly wrong for wide hidden dims — exactly the LLM regime that matters).
Two candidate root causes to check with `/debug-ttnn-op`: (a) the cross-call accumulator reload
(`Accumulate::at` → `copy_tile` from `cb_sumsq`) losing fp32 precision per block via a stale SRCA
reconfig; (b) the matmul-with-ones (`ReduceTile`) datapath biasing per tile. The design's anticipated
fix is the **accumulate + SFPU-finalize (`ReduceAlgorithm::AccumulateViaAdd`)** reduce datapath
(`master.md::reduce_block` / `row_reduce_accumulate`; `reduce_helpers_compute.hpp` — note its
`AccumulateViaAdd` restrictions: SUM only, `BulkWaitBulkPop`, and tile-aligned, so the non-tile-aligned-W
partial path needs the masked-last-tile handling the docstring describes). Repro table in
`verification_report.md § Precision Baseline`; `probes/probe_011.py` regenerates it.

**Done when**: every `float32 + W=8192` cell currently in `supported_fail` passes the golden suite,
the `test_rms_norm_precision_baseline.py::...[wide-fp32]` xfail flips to xpass (then un-mark it), and
the got/true ratio at W=16384 fp32 is within precision noise of 1.0 (not `1.04`).

### [~] Refinement 2 — Numerical configurability + gamma format flexibility

**Goal**: grow the precision/weight surface:
- add `ttnn.bfloat8_b` to `SUPPORTED["dtype"]` (TILE input only — `bf8b + ROW_MAJOR` is INVALID);
- add `ttnn.bfloat8_b` to `SUPPORTED["gamma_dtype"]`;
- add `ttnn.TILE_LAYOUT` to `SUPPORTED["gamma_layout"]` (read gamma as tiles directly, no
  RM→tilize) — required for bf8b gamma (block-quantized needs tiles) and independently useful for
  mixed-precision (bf16 activations + fp32/bf8b TILE weights);
- drop the `{bfloat16, fp32_dest_acc_en=False}` EXCLUSION (bf16 without fp32 dest-acc is a supported
  target, per the design); keep `{float32, fp32_dest_acc_en=False}` (permanent);
- wire `compute_kernel_config` precision fully (intermediate-CB formats, `UnpackToDestFp32` tagging).
- Cells that fail out of the box — typically `bfloat8_b + w_non_aligned` / `bfloat8_b + h_non_aligned`
  — go to `EXCLUSIONS`, not their own refinement.

**Implementation skill**: /numeric-formats-metal, /memory-layouts

**Verifier notes**: land **after** Refinement 1 — bf8b's lower-precision reduce path must sit on the
fixed fp32 reduce datapath, and the `{bf16, False}` cells will re-derive intermediate-CB formats from
the datapath R1 settles. `/numeric-formats-metal` is descriptor-only (its pass condition is "zero
kernel changes when helpers are wired") and owns the dtype set + config + intermediate-CB precision;
`/memory-layouts` owns the **TILE-gamma reader path** (a second gamma leg alongside the existing
RM+tilize one — the CB single-producer rule applies). Once this lands, the 3 WIDE loose cases
(`gamma_layout=TILE`) flip from `xfail_expected` to `supported_pass`.

### [x] Refinement 2a — bf16 wide-W reduce-accumulation precision (R1-analog for bf16)

**Goal**: close the two bf16 reduce-accumulation precision misses that Refinement 2
exposed (both left failing, not silenced — they are this refinement's baseline):
- **`bfloat16 + fp32_dest_acc_en=True`, extreme W** — loose case `1x1x32x32768`
  (`test_op_loose`): `pcc=0.999987` but `rms=0.404` ≫ 0.04 (a pure scale drift). `W=16384`
  passes (`rms≈0.037`); `W=32768` falls off the cliff. Root cause is the same one R1 fixed
  for fp32: the ReduceTile matmul-with-ones path parks the running `Σx²` reduced-partial in a
  bf16 `cb_sumsq` between W-blocks and carries a residual `∝W` bias. R1 fixed fp32 by moving
  to `ReduceAlgorithm::AccumulateViaAdd` (raw element-wise fp32 accumulator, reduced once) but
  explicitly kept **bf16 on ReduceTile**. **Measured null result (R2)**: forcing `cb_sumsq`
  to fp32 on the ReduceTile path is a **net regression** — it removes the bf16 cliff but leaves
  the smooth `∝W` bias, so `W=16384` flips 0.037→0.044 (fail) while `W=32768` (0.092) still
  fails; reverted byte-for-byte. The real lever is **extending R1's AccumulateViaAdd fp32-raw-
  accumulator datapath to the bf16 tile-aligned path** (gate widened from `IS_FP32` to
  `(IS_FP32 || bf16) && !HAS_PARTIAL_W`, with an fp32 accumulator CB so the raw sum doesn't
  truncate). This case is also a designated Refinement-5 W-split forcing case; either the R1-
  analog datapath or the R5 cross-core W-split resolves it.
- **`bfloat16 + fp32_dest_acc_en=False`, wide W** — 12 `test_translated.py::test_rms_norm_row_major`
  cells (`w=4096`, uniform `torch.rand`): relative Frobenius `0.05225` vs the reference-derived
  `0.052` (a 0.5% overshoot; PCC + allclose pass). The golden **cartesian** suite (normal
  `torch.randn`, `rms≤0.04`) passes bf16+False at every `W≤8192`, so bf16+False is genuinely
  supported — this is a distribution-sensitive precision floor of running the accumulation in
  **bf16 dest** (there is no fp32 dest to lean on). The R1-analog fp32 accumulator CB helps the
  parked-value precision but the per-block reduce still runs in bf16 dest; investigate whether
  an fp32 `cb_sumsq` + accumulate-in-CB (reload-add pattern) recovers the 0.5%, or whether this
  is an inherent `fp32_dest_acc_en=False` limitation to be documented as a tighter-than-golden
  translated threshold.

**Verifier notes**: standalone datapath refinement, the bf16 sibling of Refinement 1 (blocking
for fp32). Do **not** silence via a `shape_size` tagger / EXCLUSIONS (same reasoning as R1: it
would leave bf16 quietly wrong for wide hidden dims — the LLM regime). No SUPPORTED axis change
(bf16 and `fp32_dest_acc_en=False` are already supported). `/debug-ttnn-op` + the got/true-ratio
triage from R1 (`probes/probe_010.py`); `reduce_helpers_compute.hpp` `AccumulateViaAdd` (SUM /
`BulkWaitBulkPop` / tile-aligned only — matches the tile-aligned wide-W target).

**Done when**: `test_op_loose[1x1x32x32768…]` passes the golden suite (or is resolved by R5),
the 12 `test_rms_norm_row_major[*-False-*-4096-*]` translated cells pass, and no regression on
the bf16/fp32 cartesian set.

### [x] Refinement 3 — Data-movement co-tune (PERF)

**Type**: perf

**Goal**: the reader is latency-bound — it issues one `noc_async_read` + one barrier per tile (TILE
regime) / per stick (RM regime), leaving the NoC mostly idle, and `W_BLOCK_TILES=1` means each reduce
chunk is a single tile. Speed up the already-supported cells (no SUPPORTED change) using the relevant
`ttnn/ttnn/operations/examples/master.md` patterns — `double_buffer` (issue a *block* of ~4–8 async
reads then one barrier, CBs sized `2·block`), `compute_block_size` (raise `W_BLOCK_TILES` so a reduce
chunk is several tiles — amortizes per-helper reconfig/init), and `reader_placement` (`row_wise=True`)
once Refinement 4 makes the reader multi-core. Co-tune the block-size and buffer-depth knobs the
planner already exposed; the chunk-granularity floor is whole tiles (a one-tile chunk is the poor
case, coarser amortizes up to the L1 budget).

**Verifier notes**: cadence perf after Refinements 1–2. Raising `W_BLOCK_TILES` requires handling
`Wt` **not** divisible by `W_BLOCK_TILES` (a partial *last W-block*) — the descriptor currently
asserts `Wt % W_BLOCK_TILES == 0`; generalize it, and keep the partial-scaler routed only to the true
last W-tile of the last block. **Also thread `ROW_BLOCK_TILES` into the compute outer loop here** —
it currently only sizes `cb_sumsq` host-side and is ignored by the row loop (a half-wired knob); if
you raise it, the compute must process `ROW_BLOCK_TILES` tile-rows per pass or `cb_sumsq` mis-sizes.
Wide-W bf16 cells (W=4096/8192, few tile-rows) are the most latency-bound targets on one core.

**Done when**: measured device-ns improves on the most-impacted wide-W shape(s) with the golden suite
still green, and no regression across the config-spanning guard set (one representative per distinct
kernel path × layout × placement: {TILE, RM} × {gamma, no_gamma} × {bf16, f32}, interleaved).

### [x] Refinement 4 — Multi-core row distribution + HEIGHT_SHARDED

**Goal**: add `ttnn.TensorMemoryLayout.HEIGHT_SHARDED` to `SUPPORTED["memory_layout"]` and distribute
the row work across the grid for interleaved tensors. Both are the **Row-axis knob-turn** (the design's
independent axis): each core owns a contiguous span of tile-rows and computes their RMS locally — **no
cross-core communication**. Interleaved multi-core uses `ttnn.split_work_to_cores(grid,
total_tile_rows)` with per-core `(start_tile_row, num_tile_rows)` runtime args (the descriptor already
carries these as RT args — Phase 0 just pins them to the whole range on one core). HEIGHT_SHARDED is
the same split with the row→core assignment pinned by the shard spec instead of `split_work_to_cores`;
the reduction stays local per core.

**Implementation skill**: /interleaved-parallel

**Verifier notes**: land after Refinement 2 (the multi-core/sharded kernels must already handle the
full dtype set). This bundles the embarrassingly-parallel interleaved multi-core split (no SUPPORTED
change on its own) with the HEIGHT_SHARDED axis value — both are the Row knob-turn, same reader/writer
work-distribution change, so they share one refinement. HEIGHT_SHARDED placement has **no dedicated
skill**: it is a reader/writer **CB-placement** change — point the input/output CBs at the sharded L1
buffer via `ttnn.cb_descriptor_from_sharded_tensor` (golden shard specs come from
`eval.sharding.auto_shard_config`; no per-scheme test work). This does **not** shrink per-core CBs, so
it is orthogonal to any L1 concern. The HEIGHT_SHARDED loose case (`1x1x256x512`) flips to
`supported_pass` when this lands.

### [x] Refinement 5 — WIDTH_SHARDED + BLOCK_SHARDED cross-core reduction

**Goal**: add `ttnn.TensorMemoryLayout.WIDTH_SHARDED` and `ttnn.TensorMemoryLayout.BLOCK_SHARDED` to
`SUPPORTED["memory_layout"]`. Here the hidden `W` is split across cores, so the RMS denominator spans
core boundaries: each core computes a **partial Σx²** over its `W`-slice, the partials are combined
across the grid (sum, or Welford), and the finalized `1/rms` is broadcast back. This is the design's
**dependent-axis scheme-change** — the cross-core combine topology *is* the work, so it stands alone.

**Verifier notes**: **scheme-change, standalone** (not a knob-turn — do not try to bundle it with
Refinement 4's local HEIGHT split). No implementation skill covers cross-core reduction yet: work from
`.claude/references/cross_core_reduction_design.md` and the `mcast_pipe.hpp`
(`SenderPipe`/`ReceiverPipe`/`McastRect`) transport; `master.md::tensix_all_reduce*` (T3) is the
measured topology evidence. Phase 0's streaming reduce already materialises a per-core local partial
`Σx²` (the `cb_sumsq` accumulator), so the combine only has to gather + fold + broadcast-back. Land
last — it is the highest-complexity refinement and stresses inter-core sync. The WIDTH/BLOCK loose
cases (`1x1x32x2048` WIDTH, `1x1x256x512` BLOCK) flip to `supported_pass` when this lands.

**Landed (2026-07-15)**: WIDTH_SHARDED + BLOCK_SHARDED added to SUPPORTED via a reduce-root
cross-core combine (each core reduces its LOCAL W-slice → partial `Σx²/W_global`; the group
root gathers via unicast+`progress` sem, folds + rsqrt-finalizes, and broadcasts `1/rms` back
over the group rectangle via `mcast_pipe` `SenderPipe`/`ReceiverPipe` + host `Mcast2D`). Golden
cartesian: **WIDTH/BLOCK 2102 passed / 0 failed / 2100 xfailed**; INTERLEAVED/HEIGHT 3346 passed
(no regression); both loose cases pass. Rectangular groups only (BLOCK grid-row lines; WIDTH
shard-grid bbox); ragged / non-tile-aligned-W route to a correct interleaved-collapse fallback.
`{ROW_MAJOR, WIDTH/BLOCK_SHARDED}` EXCLUDED (RM W-sharding splits row sticks sub-tile across
cores) → Refinement 5a below.

### [ ] Refinement 5a — ROW_MAJOR + WIDTH/BLOCK_SHARDED cross-core reduction

**Goal**: lift the R5 `{ROW_MAJOR, WIDTH_SHARDED}` / `{ROW_MAJOR, BLOCK_SHARDED}` EXCLUSIONS. RM
width-sharding splits each logical row's `W` across cores at sub-tile (stick) granularity, so a
row is not contiguous in any one core's L1 — neither the R5 TILE-only cross-core path (it never
tilizes) nor the interleaved-collapse fallback (it reads a full-W stick as one NoC page, which
returns garbage for a split stick; measured PCC 0.0098) handles it. The fix: extend the sharded
reader/compute to **tilize each core's partial-width stick slice** before the partial reduce, then
reuse the R5 gather + mcast-broadcast combine unchanged. Low priority — RM width-sharding is an
unusual model config; the TILE surface (the norm-op default) is fully covered by R5.

**Verifier notes**: standalone follow-up to R5's structural gap. The cross-core combine topology
(gather + fold + broadcast-back) is already proven; this is a per-core RM→TILE tilize of a
partial-width W-slice added ahead of the existing pass-1 reduce (the interleaved/HEIGHT RM reader
already tilizes full-W stick blocks — the extension is the partial-width offset + count). No new
transport. `test_rms_norm_sharded.py::test_rms_norm_rejects_row_major_width_block` is the current
guard; flip it to an acceptance test when this lands.

**Done when**: `{ROW_MAJOR, WIDTH_SHARDED}` + `{ROW_MAJOR, BLOCK_SHARDED}` move out of EXCLUSIONS
into passing golden cells, no regression on the R5 TILE surface.

> **Trailing perf (once generality is exhausted, after Refinement 5)**: the wide-W / few-tile-row
> interleaved cells (W=16384/32768, 1–2 tile-rows) are latency-bound on one core — the cross-core
> W-split machinery from Refinement 5 can be applied to interleaved tensors for grid occupancy
> (`master.md::reader_placement` + `tensix_all_reduce`). File as a measured perf phase (device-ns
> gated, no-regression on the guard set) when Refinement 5 lands.
