# Operation Requirements: rms_norm

## Definition
- **Formula**: `out[..., h, w] = x[..., h, w] * rsqrt( (1/W)·Σ_w x[..., h, w]² + eps ) * gamma[w]` (gamma optional)
- **PyTorch Reference**:
  ```python
  def rms_norm_ref(x, gamma=None, epsilon=1e-6):
      x = x.to(torch.float32)
      out = x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + epsilon)
      if gamma is not None:
          out = out * gamma.to(torch.float32).reshape(-1)
      return out
  ```
- **Import Path**: `from ttnn.operations.rms_norm import rms_norm`
- **Function Signature**:
  ```python
  def rms_norm(
      input_tensor: ttnn.Tensor,
      *,
      gamma: "ttnn.Tensor | None" = None,
      epsilon: float = 1e-6,
      compute_kernel_config: "ttnn.ComputeConfigDescriptor | None" = None,
      memory_config: "ttnn.MemoryConfig | None" = None,
  ) -> ttnn.Tensor
  ```

## Phases

> **Non-regression rule**: Every refinement must pass all tests from prior phases.
> **Drift signal**: XPASS-strict failures mean the implementer added support but forgot to update SUPPORTED. Fix by updating SUPPORTED (and keep `eval/golden_tests/rms_norm/axes.py:classify_call` in lockstep with `validate()` when a new tensor-derived axis is added).
> **Checkbox protocol**: Implementer marks `[x]` when the refinement is complete and all tests pass, `[~]` when real work landed but at least one named axis value is deferred (treated as completed by the queue, surfaced as partial), `[ ]` only when nothing usable was produced.
> **Refinement ID + follow-up naming (mandatory — the runner parses `Refinement \d+[a-z]?`)**: primary refinements are `Refinement N`; a partial-tick follow-up appends a lowercase letter to the parent (`Refinement 4b`), ordered immediately after its parent.

### [x] Phase 0 — Core Implementation

- **SUPPORTED dtype**: [float32, bfloat16]
- **SUPPORTED fp32_dest_acc_en**: [True] (Phase-0 maxed precision corner)
- **SUPPORTED layout**: [TILE, ROW_MAJOR] — both native (tilize-on-read / untilize-on-write)
- **SUPPORTED alignment**: [tile_aligned, w_non_aligned, h_non_aligned] — native masked/padded reduce
- **SUPPORTED rank**: [2, 3, 4]
- **SUPPORTED gamma_mode**: [gamma, no_gamma]
- **SUPPORTED gamma_dtype**: [float32, bfloat16, "none"]
- **SUPPORTED gamma_layout**: [ROW_MAJOR, "none"]  (RM gamma is the phase-1 contract)
- **SUPPORTED memory_layout**: [INTERLEAVED]
- **Scheme**: row-parallel, bounded two-pass streaming reduce over W; multi-core from day 1 (`split_work_to_cores(R, grid, row_wise=True)`). All block knobs (`BLOCK_SIZE`, `DEPTH`, grid) are live parameters; no CB sized by an op dimension.
- **Compute config**: `default_compute_kernel_config()` = HiFi4 + fp32_dest_acc_en=True; `compute_kernel_config` exposed on the entry point.
- **Golden baseline**: **472 / 40438** cells passing (`supported_pass`); 6051 xfail_expected (the TARGET−SUPPORTED gap below); 33900 invalid_skipped. supported_fail = xpass_drift = xfail_wrong_mode = 0.

---

### [x] Refinement 1 — Numerical configurability expansion

**Goal**: widen the precision surface to the full TARGET:
- add `ttnn.bfloat8_b` to `SUPPORTED["dtype"]` and to `SUPPORTED["gamma_dtype"]`
  (bf8b is valid only at `tile_aligned` + TILE — `bf8b+RM` and `bf8b+non_aligned`
  are already INVALID in `feature_spec.py`, so no EXCLUSION is needed for those);
- add `False` to `SUPPORTED["fp32_dest_acc_en"]` (bf16 with bf16 DEST accumulation);
- keep `compute_kernel_config` (already exposed) driving `math_fidelity` /
  `fp32_dest_acc_en` / `math_approx_mode`; correct intermediate-CB precision and
  add `UnpackToDestFp32` tagging where it applies.
- **Add the one EXCLUSIONS entry** `{"dtype": ttnn.float32, "fp32_dest_acc_en": False}`
  — the design's legal-but-refused precision corner (f32 with non-fp32 accumulation
  is lossy). It is out-of-rectangle today (fp32_dest_acc_en=[True]); once `False`
  enters SUPPORTED it must move to EXCLUSIONS so `{f32,False}` stays xfail-strict.
  Any tile-aligned `bf8b` cell that fails out of the box also goes to EXCLUSIONS,
  not its own refinement.

Pass condition: zero kernel changes when the helpers are wired correctly (DEST
budget already honoured via `DEST_AUTO_LIMIT`).

**Implementation skill**: /numeric-formats-metal

**Verifier notes**: land this **first** — it is half of the perf-1 anchor. The
interleaved LLM perf loose cases (`_perf_case(...)` in `feature_spec.LOOSE_CASES`)
run at `fp32_dest_acc_en=False`, which this refinement adds; without it the R3
perf pass would have no supported config to measure. Refinement 2 (tiled gamma)
completes the anchor. The precision baseline (this dir's
`test_rms_norm_precision_baseline.py`) is the before/after guard — bf8b lands at
PCC ≥ 0.99 / rel-RMS ≤ 0.10 (golden `TOLERANCES`). Moves ~3255 (`fp32_dest_acc_en=False`)
+ 960 (`dtype=bf8b`) + 860 (`gamma_dtype=bf8b`) xfail cells toward passing (minus
the one `{f32,False}` EXCLUSION cell, which stays xfail-strict).

**Done when**: `bfloat8_b`∈dtype/gamma_dtype and `False`∈fp32_dest_acc_en in
SUPPORTED; `{float32, fp32_dest_acc_en=False}` in EXCLUSIONS; golden suite green
with those cells moving from `xfail_expected` to `supported_pass` (except the
excluded/INVALID cells); acceptance + precision-baseline suites still pass.

---

### [ ] Refinement 2 — Tiled-gamma layout support

**Goal**: add `ttnn.TILE_LAYOUT` to `SUPPORTED["gamma_layout"]`. Per `op_design.md`
§5 tiled gamma is a knob-turn: gamma arrives already tiled, so the reader reads it
as tiles straight into `cb_gamma` and the compute-side gamma `tilize` phase is
**skipped** (`cb_gamma_sticks` and the pass-2 `ckl::tilize<…,cb_gamma_sticks,
cb_gamma>` disappear on the TILE-gamma path). The RM-gamma path (the phase-1
contract) stays intact — both selected by a host predicate on `gamma.layout`.

**Implementation skill**: /memory-layouts

**Verifier notes**: second half of the perf-1 anchor — the interleaved perf loose
cases all pin `gamma_layout=TILE`, so R1+R2 together make the exact perf-1 config
(`bf16 / fp32_dest_acc_en=False / TILE / TILE gamma / INTERLEAVED / HiFi2`)
`supported_pass`. Build it to the performance-conformance bar (the tiled-gamma
reader must still batch its reads and fill the grid), because R3 optimizes this
very path. Do **not** wrap the input in `ttnn.to_layout`/`ttnn.tilize` — this is a
native reader-path change (the escape hatch does not apply; there is no hard
blocker here). Moves ~2599 (`gamma_layout=TILE`) xfail cells toward passing. No
dependency on R1's kernel work, but keep R1 first so the anchor config is complete
before R3.

**Done when**: `TILE_LAYOUT`∈`SUPPORTED["gamma_layout"]`; TILE-gamma cells pass via
the native tiled reader (no host-side gamma transform); golden green; RM-gamma
cells unregressed.

---

### [ ] Refinement 3 — Speed up the interleaved prefill perf profile

**Type**: perf

**Goal**: `feature_spec.LOOSE_CASES` carries model-derived interleaved perf targets
via `_perf_case(rows, W, achievable_ns, …)`. Target the **prefill** interleaved
shapes — `(1,1,8192,W)` for `W ∈ {1024, 2304, 5120, 7168}` (rows=8192 → 256
tile-rows fill the 110-core grid) — at their exact config
(`bf16 / fp32_dest_acc_en=False / TILE / TILE gamma / INTERLEAVED / HiFi2`, soft
`pcc_threshold=0.9995`), and drive their median device-ns toward the case's
`achievable_ns`. Pick the lever(s) from `ttnn/ttnn/operations/examples/master.md`
after a roofline check; the applicable ones for this bandwidth-bound row-parallel
regime are the **resident single-read fast-path** (design lamp 1 — load each row
once and do both passes over L1, eliminating the second DRAM read; every perf
width fits one core's L1), `double_buffer` (block / buffer-depth co-tune to keep
bytes in flight), and `compute_block_size` (raise `ROWS_PER_CALL` to amortize the
two-pass per-helper reconfig). No SUPPORTED change.

**Verifier notes**: do NOT target the **decode** perf shapes (`rows=32` → `R=1`
tile-row → 1 core) here — they are latency-bound and need the cross-core W-split
(R4). Optimize the flagged config exactly; R1+R2 must be complete so
`fp32_dest_acc_en=False` + TILE gamma are supported (never substitute a
`fp32_dest_acc_en=True` proxy — different datapath). Co-tune block-size and
buffer-depth; the master.md chunk-granularity floor (whole tiles minimum, coarser
amortizes) bounds the search.

**Done when**: measured median device-ns improves on the prefill perf shapes
(fresh-cache trial loop, cleared against the noise threshold) with the soft PCC
gate (0.9995) still holding and the golden suite green, and no regression across
the config-spanning guard set (one representative per distinct kernel path × layout
× placement — at minimum: TILE interleaved, RM interleaved, no-gamma).

---

### [ ] Refinement 4 — Cross-core W-split: WIDTH/BLOCK sharding + logical wide-interleaved split

**Goal**: split the reduced dim `W` across cores and combine partials across the
grid — the design's lamp-2 scheme-change (`op_design.md` §1 lamp 2, §5 cross-core
contract). This lands **three** things that share one topology:
- add `ttnn.TensorMemoryLayout.WIDTH_SHARDED` and `ttnn.TensorMemoryLayout.BLOCK_SHARDED`
  to `SUPPORTED["memory_layout"]` — the hidden dim is pre-placed across cores;
  each core reduces its local slice to a partial `Σx²`, then one cross-core round
  (gather → add → broadcast the finalized `1/RMS`) precedes per-core normalize;
- the same combine, applied to a **logical** W-split of a wide *interleaved* input
  (each core reads its `W/K` slice from interleaved DRAM), so wide/few-row shapes
  that under-fill the grid stop running on one core: the `LOOSE_CASES` wide
  interleaved shapes (`W=16384/32768/12288`) and the **decode** perf shapes
  (`rows=32`, `R=1` tile-row).

Built on `mcast_pipe` (`SenderPipe`/`ReceiverPipe`) + a partial-stat gather; the
per-core slice is consumed **locally** (WIDTH/BLOCK shards via a zero-copy CB on
the sharded L1 buffer — `ttnn.cb_descriptor_from_sharded_tensor`; logical
interleaved slices via the reader's per-core `W/K` offset). Group geometry must be
rectangular (NoC mcast addresses a rectangle) — the host core assignment
guarantees it. Use a **separate** `cb_stat_handoff` for the mcast handoff, never
`cb_rstd` (two-consumer trap, §7).

**Verifier notes**: **scheme-change — stands alone.** This is the hardest
generality refinement (tier-1/2: complicated sharding + cross-core restructure),
so per the hardest-first rule it precedes R5 (local HEIGHT shard). No dedicated
skill yet — follow `references/cross_core_reduction_design.md` (Pattern A→B) and
the §5 cross-core contract. `memory_layout`/sharding is NOT `/memory-layouts`
(that's RM/TILE layout). Golden shard specs come from `eval.sharding` (auto +
`shard_config` for the exact perf geometry); no test authoring needed. Sharded
loose cases (`_SHARDED`, and the WIDTH/BLOCK `_perf_case` geometries) become
`supported_pass` once this lands. **Verify the native path**: the local slice must
be consumed via the zero-copy sharded CB, never re-read through a `TensorAccessor`
as if interleaved — an accessor read of a core's own shard means the layout was
never implemented; send it back to be completed natively (do not file as a
follow-up perf refinement). Moves ~3007 (`WIDTH/BLOCK_SHARDED`) xfail cells toward
passing and unlocks the decode/wide perf regime for R6.

**Done when**: `WIDTH_SHARDED` and `BLOCK_SHARDED`∈`SUPPORTED["memory_layout"]`;
their golden + sharded-loose cells pass via the native cross-core combine (partials
crossing the NoC, slices consumed locally); wide interleaved / decode shapes fill
more than one core; golden green; all prior phases unregressed.

---

### [ ] Refinement 5 — HEIGHT_SHARDED (local per-core reduction)

**Goal**: add `ttnn.TensorMemoryLayout.HEIGHT_SHARDED` to `SUPPORTED["memory_layout"]`.
Rows are split across cores and each core keeps **full-W** rows, so the reduction
stays **local** per core (`op_design.md` §1 lamp 3, HEIGHT branch — a knob-turn,
not a cross-core scheme). The row-shard is already resident in each core's L1, so
back `cb_x_in` on the sharded buffer via `ttnn.cb_descriptor_from_sharded_tensor`
(zero-copy, **no NoC read**) and consume the whole resident shard as the per-core
block; the writer mirrors it for the sharded output. No new math — a reader/writer
CB-placement change on the existing row-parallel path.

**Verifier notes**: **local sharding = knob-turn**, easier than R4 (tier-4 simple
sharding), so it is ordered after the harder cross-core R4 per hardest-first. No
dependency on R4 (orthogonal mechanism — local vs cross-core). No dedicated skill;
the pattern is the `ttnn.cb_descriptor_from_sharded_tensor` zero-copy CB placement
named in `op_design.md` §5/§8 (NOT `/memory-layouts`, which is RM/TILE). **Verify
the native path**, exactly as R4: the local shard *is* the per-core block and must
be consumed via the zero-copy CB, never re-read through a `TensorAccessor` — an
accessor read of a core's own resident shard means HEIGHT_SHARDED was never
implemented (the op merely tolerates the layout); send it back to be completed
natively, do not defer as perf. Golden shard specs auto-synthesize via
`eval.sharding.auto_shard_config`. Moves ~1501 (`HEIGHT_SHARDED`) xfail cells to
passing; the HEIGHT `_SHARDED` loose case (`(1,1,256,512)`) becomes `supported_pass`.

**Done when**: `HEIGHT_SHARDED`∈`SUPPORTED["memory_layout"]`; its golden + loose
cells pass via a zero-copy sharded CB (no accessor read of a core's own shard);
golden green; all prior phases unregressed.

---

### [ ] Refinement 6 — Speed up the decode + sharded perf profiles

**Type**: perf

**Goal**: with the cross-core W-split (R4) and local shard (R5) in place, target
the perf profiles R3 could not: the **decode** interleaved shapes
(`_perf_case(32, W, …)` for `W ∈ {1024, 2304, 5120, 7168}` — now multi-core via the
logical W-split) and the **WIDTH/BLOCK-sharded** `_perf_case` geometries (their
`achievable_ns` is the sharded latency), at their exact configs. Drive median
device-ns toward each case's `achievable_ns`. Levers from
`ttnn/ttnn/operations/examples/master.md` after a roofline/bottleneck check: the
cross-core collective topology (`tensix_all_reduce` grid-two-stage vs flat root;
`tensix_all_reduce_ring_transport` for direction-sensitive NoC contention),
`noc_placement` (NoC0-read / NoC1-write, row/diagonal line placement for the mcast),
and `shared_input_reuse` (mcast gamma once instead of per-row re-read). No SUPPORTED
change.

**Verifier notes**: hard perf pass (T3 collective restructure) — one lever family
per pass, do not pack the cross-core topology tuning together with cheap knobs.
Optimize each flagged config exactly (decode uses `fp32_dest_acc_en=False`+TILE
gamma from R1/R2; sharded geometries pinned via the `_perf_case` `shard_shape`/
`core_grid` extras). Gate on the soft `pcc_threshold` (0.9995).

**Done when**: measured median device-ns improves on the decode and sharded perf
shapes (fresh-cache trial loop, cleared against noise) with soft PCC holding and
golden green, and no regression across the config-spanning guard set (one
representative per distinct kernel path × layout × placement: TILE interleaved, RM
interleaved, HEIGHT-sharded, WIDTH-sharded, BLOCK-sharded, no-gamma).
