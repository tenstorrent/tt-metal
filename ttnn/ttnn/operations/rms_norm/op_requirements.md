# Operation Requirements: rms_norm

## Definition

- **Formula**: `out[..., w] = x[..., w] / sqrt( (1/W) · Σ_{w'} x[..., w']² + eps ) · gamma[w]`
  — reduction over the last dim only; `W` is the **logical** width (tile padding never
  enters the denominator).
- **PyTorch Reference**:

  ```python
  def torch_rms_norm(x, gamma=None, epsilon=1e-6):
      original_dtype = x.dtype
      xf = x.to(torch.float32)
      rms = torch.sqrt(torch.mean(xf ** 2, dim=-1, keepdim=True) + epsilon)
      out = xf / rms
      if gamma is not None:
          out = out * gamma.to(torch.float32).reshape(-1)
      return out.to(original_dtype)
  ```

- **Import Path**: `from ttnn.operations.rms_norm import rms_norm`
- **Function Signature**:

  ```python
  def rms_norm(
      input_tensor: ttnn.Tensor,
      *,
      gamma: Optional[ttnn.Tensor] = None,
      epsilon: float = 1e-6,
      compute_kernel_config: Optional[ttnn.ComputeConfigDescriptor] = None,
      memory_config: Optional[ttnn.MemoryConfig] = None,
      program_config: Optional[Any] = None,
  ) -> ttnn.Tensor
  ```

  `compute_kernel_config=None` resolves through the exported
  `default_compute_kernel_config()` (HiFi4 / `fp32_dest_acc_en=True` /
  `math_approx_mode=False`).

## Phases

> **Non-regression rule**: Every refinement must pass all tests from prior phases.
> **Drift signal**: XPASS-strict failures mean the implementer added support but forgot to update SUPPORTED. The implementer fixes by updating SUPPORTED.
> **Checkbox protocol**: Implementer marks `[x]` when the refinement is complete and all tests pass, `[~]` when real work landed but at least one named axis value is deferred (treated as completed by the queue, surfaced as partial), `[ ]` only when nothing usable was produced.
> **Refinement ID + follow-up naming (mandatory — the runner parses this)**: Primary refinements are `Refinement N` (e.g. `Refinement 1`, `Refinement 2`). When you ship `[~]` partial and file the sharper follow-up the partial-tick protocol requires, name it by appending a lowercase letter to the parent's number: `Refinement 1b`, `Refinement 1c`, … (never `Refinement 1.5`, `Refinement 1 (follow-up)`, or a fresh number). Order follow-ups immediately after their parent so the queue runs them before later refinements — a partial's remaining-blocker follow-up must be picked next, not leapfrogged. The runner's parser matches exactly `Refinement \d+[a-z]?`; any other shape is invisible to the queue and silently skipped.

### [x] Phase 0 — Core Implementation

- **SUPPORTED dtype**: [float32, bfloat16]
- **SUPPORTED fp32_dest_acc_en**: [True]  (`{float32, False}` is a standing EXCLUSION)
- **SUPPORTED layout**: [TILE, ROW_MAJOR] — both native, no host-side layout ops
- **SUPPORTED shape-derived axes**: alignment ∈ {tile_aligned, w_non_aligned, h_non_aligned} (all three); rank ∈ {2, 3, 4} (all)
- **SUPPORTED op-specific axes**: gamma_mode ∈ {gamma, no_gamma}; gamma_dtype ∈ {float32, bfloat16, "none"}; gamma_layout ∈ {TILE, ROW_MAJOR, "none"}
- **SUPPORTED memory_layout**: [INTERLEAVED]
- **Cores**: multi-core on day 1 — `split_work_to_cores(full_grid, Rt, row_wise=True)` over the independent `row` axis; `GRID_W = 1` (the dependent `width` axis stays inside a core)
- **Compute config**: caller-supplied and passed through unmodified; `math_fidelity` / `math_approx_mode` ungated; `fp32_dest_acc_en=True` gated in Phase 0
- **Golden baseline**: **737 / 737** supported cells passing (`supported_fail = 0`,
  `xpass_drift = 0`, `xfail_wrong_mode = 0`); 6172 xfail, 33900 invalid-skipped,
  2 infeasible-skipped (per `verifier_report.json`)

---

### [~] Refinement 1 — Numerical configurability expansion (dtype + DEST/fidelity surface)

**Goal**: complete the precision surface in one pass.
1. add `ttnn.bfloat8_b` to `SUPPORTED["dtype"]` and to `SUPPORTED["gamma_dtype"]`
   (mixed bf8b-weight / bf16-activation included);
2. add `False` to `SUPPORTED["fp32_dest_acc_en"]`, keeping the existing
   `{float32, fp32_dest_acc_en: False}` EXCLUSION intact — i.e. the newly reachable
   corner is `{bfloat16 | bfloat8_b} × fp32_dest_acc_en=False`;
3. set intermediate-CB formats correctly for the new dtypes (`cb_x_squared`,
   `cb_normalized`, `cb_output_tiles` follow the input dtype; `cb_row_stat` stays fp32;
   `cb_gamma_tiles` follows the *gamma* dtype) and add `UnpackToDestFp32` tagging where it
   applies — note `AccumulateReloadMode`'s contract: a `to-dest` accumulator CB must not be
   folded via SrcA/SrcB.
   Cells that fail out of the box go to `EXCLUSIONS`, not to their own refinement.

**Implementation skill**: /numeric-formats-metal

**Verifier notes**:
* **This refinement is the gate on the whole perf track — it must land first.** Every
  perf loose case in `feature_spec.LOOSE_CASES` (the `_perf_case` table with
  `achievable_ns`) is pinned at `bfloat16 + fp32_dest_acc_en=False + HiFi2`, and so is the
  entire `_RESILIENCE_SHAPES` × placement sweep and every pad-poison case. Until `False` is
  in SUPPORTED, Refinement 3/4 would have to measure a `fp32_dest_acc_en=True` stand-in —
  a different datapath, so a meaningless number. Landing `False` converts **3607** xfail
  cells (100 of them loose cases), and bf8b a further 1662.
* **Precision risk on the one corner that matters most.** With `fp32_dest_acc_en=False` the
  reduce accumulates `Σx²` in bf16 DEST, and a sum of squares is the all-positive
  worst case for bf16 accumulation (`row_reduce_accumulate/report.md`: 5.83 ULP @ 32 tiles,
  error *grows* with reduce width). The perf cases run `W` up to 7168 (`Wt = 224`) with a
  soft `pcc_threshold = 0.9995`. If that misses, **do not exclude the cell** — it is
  Refinement 3's mandatory target. Levers, in order: keep `cb_row_stat` fp32 (already true)
  so the cross-chunk `Accumulate::at` reload is lossless; shrink the per-call reduce block
  so more accumulation happens through that fp32 CB; and wire the `ReduceFp32Mode` slot
  that `accumulate_reduce_block<>` currently hides (descriptor deviation **D3**) if the
  float32 path needs it too.
* Measured Phase 0 baseline to hold against: bf16 PCC 0.999997 / rel RMS 0.0024, fp32
  PCC 0.9999997 / rel RMS 0.0015 (`verification_report.md` § Precision Baseline). Extend
  `test_rms_norm_precision_baseline.py` with the new dtype × `fp32_dest_acc_en` rows
  rather than writing a second file.
* `feature_spec.INVALID` already pre-excludes bf8b's hardest corner
  (`{bfloat8_b, w_non_aligned}` / `{…, h_non_aligned}`), so the activation side should be
  clean. The **gamma** side has no such entry: expect `{gamma_dtype: bfloat8_b,
  alignment: *_non_aligned}` to fail (a bf8b gamma shares one exponent per block, so pad
  lanes perturb real weights) and park it in `EXCLUSIONS` with a pointer to
  `op_design.md` §9.2, which proposes the matching INVALID entries.

**Outcome** (`[~]` partial): both named axis values LANDED and are confirmed live —
`SUPPORTED["dtype"]` and `SUPPORTED["gamma_dtype"]` gained `bfloat8_b`,
`SUPPORTED["fp32_dest_acc_en"]` gained `False`, `{float32, False}` stays the sole
EXCLUSION. Golden **1660 / 1670** (Phase 0: 737 / 737), `xpass_drift = 0`, no regression.
Zero new EXCLUSIONS: the `{gamma_dtype: bfloat8_b, alignment: *_non_aligned}` corner
§9.2 predicted would fail is **clean** (PCC 0.99997) — gamma's tile padding is zero, and a
zero never raises a block-float block's shared exponent, so the straddling block's real
weights are untouched. Two findings:
* **A latent pre-existing bug, now fixed (descriptor D6).** 11 cells failed
  *catastrophically* (PCC 0.55–0.99), identically at `fp32_dest_acc_en=True` — so not this
  axis. `transform_in_place` *rotates* its CB (pop 1 / push 1), so with `cb_row_stat` sized
  exactly `BLOCK_ROWS` a **partial** final row-block leaves the finalized stats straddling
  the ring wrap, and pass B's `OperandKind::Col` bulk-indexed read runs off the end. Every
  Phase-0 golden cell had `Rt ≤ 64 <` the 110-core grid ⇒ `BLOCK_ROWS == 1` ⇒ no partial
  block ever existed, which is why Phase 0 could not see it; the resilience loose cases
  this refinement unlocked reach it. Fixed by `CB_ROW_STAT_DEPTH = 2`, counted through both
  L1 solves.
* **The remaining 10 failures are the wide-`W` bf16-DEST reduce**, and they are
  Refinement 1b's target (below), NOT excluded. All are `severity=precision`,
  `W ≥ 5120`, `fp32_dest_acc_en=False`: PCC 0.99993–0.99996 (so the perf cases' soft
  `pcc_threshold = 0.9995` **holds** — it is the `rms ≤ 0.04` component of
  `TOLERANCES[bfloat16]` that misses, at 0.041–0.127). Diagnosed to the exact datapath, not
  guessed: DEVICE_PRINT on `cb_row_stat` shows the *reduce output* wrong (`7904` vs a true
  `7033`, +12.4 %) while the finalize is exact, and the error is **bit-invariant** across
  `NUM_W_CHUNKS` 4 → 112, across `REDUCE_BULK` ∈ {1,0} and across all four
  `math_fidelity` values. So it is not the cross-chunk `Accumulate::at` reload (verifier
  lever 2 — **measured null**) but the FPU matmul reduce's *within-tile* 32-column sum
  accumulating all-positive addends into a 16-bit DEST, which chunking cannot reach.

---

### [x] Refinement 1b — wide-`W` reduce precision under `fp32_dest_acc_en=False`

**Goal**: close the 10 `severity=precision` cells Refinement 1 left failing — every
`fp32_dest_acc_en=False` loose case with `W ≥ 5120`
(`(1,1,32,{5120,7168})`, `(1,1,8192,{5120,7168})`, `(1,1,96,6144)`,
`(1,1,160,11008)`, `(1,224,11008)`, both gamma layouts). They must reach
`rms ≤ 0.04` at `TOLERANCES[bfloat16]` without regressing the 1660 cells that pass now.
No SUPPORTED change — the axis value is already in; this closes its wide-`W` corner.

**The exact next lever** (do not re-litigate the three already measured null — chunk count,
`REDUCE_BULK`, `math_fidelity`): **`ReduceAlgorithm::AccumulateViaAdd`**. It replaces the
FPU matmul-vs-scaler reduce with pairwise `add_tiles` plus an **SFPU** within-tile finalize
(`sfpu_reduce` SUM), which is precisely the step that currently accumulates 32 all-positive
addends into a 16-bit DEST. `reduce<>` exposes it as the `algorithm` template parameter,
but **`accumulate_reduce_block<>` does not forward that slot** (its template list is
pool / rdim / cb_in / cb_scaler / cb_acc / in_policy / reconfig_mode / PostOp) — the same
class of gap as deviation **D3**, so step one is widening the wrapper in
`streaming_reduce_helpers.hpp`, not rewriting the kernel.

**Verifier notes**:
* This is the *same* lever as Refinement 4's item (a), which independently measures
  **2.87–2.94×** on `REDUCE_ROW` at `Wt ≥ 4` — so it is a precision fix and a perf win in
  one change. Whichever refinement runs first should land it; the other then only verifies.
* It is a **coupled** change, not a one-word swap: `AccumulateViaAdd` restricts
  `Accumulate` to SUM + `BulkWaitBulkPop` and swaps the partial-`W` mechanism from a scaler
  tile to a 0/1 **mask** tile (`prepare_reduce_mask` in place of
  `prepare_partial_reduce_scalers`). Re-run the poisoned-padding cells specifically — the
  acceptance set plus the 24 pad-poison loose cells are the only tests that can catch a
  masked-reduce regression, and a padding leak on a wide row is a near-uniform scale error
  PCC is largely blind to (`test_rms_norm_precision_baseline.py`'s ratio-spread assertion is
  the second net).
* Do **not** reach for `ReduceFp32Mode::Accurate` first: it routes **Float32** SUM through
  the SFPU, and these cells are `bfloat16` activations, so it does not apply.
* Confirmation harness already in the tree: `test_rms_norm_precision_baseline.py` carries
  the `dtype × fp32_dest_acc_en` matrix, and
  `tests/ttnn/unit_tests/operations/rms_norm/probes/` holds the chunk-count / `REDUCE_BULK` /
  fidelity sweeps that established the nulls — re-run them to prove the lever moves what
  those could not.

**Outcome** (`[x]` full): all 10 cells closed — golden **1670 / 1670** (Refinement 1:
1660 / 1670), `supported_fail = 0`, `xpass_drift = 0`, zero hangs, no regression. No
SUPPORTED change (none was needed). `ReduceAlgorithm::AccumulateViaAdd` landed exactly as
named: `accumulate_reduce_block<>` / `accumulate_reduce<>` now forward reduce()'s
`ReduceFp32Mode` **and** `ReduceAlgorithm` slots (this also retires deviation **D3**) and
route the last block through `Accumulate::at_last` so the datapath's within-tile finalize
runs once; the op selects it from a new crossover knob `REDUCE_ACC_VIA_ADD_MIN_WT = 4`
(descriptor **D7**). Measured rel RMS on the target cells fell from **0.042–0.127 to
0.0089–0.0109** (vs the `rms ≤ 0.04` gate), PCC 0.99988+. Three findings:
* **A latent library bug the lever exposed, now fixed.** `fold_partial_last` (the masked
  partial fold, `reduce_helpers_compute.inl`) never reconfigured **SrcB** to the mask CB —
  everything around it leaves SrcB pointing at the input CB, and `llk_unpack_AB_init` only
  *asserts* formats, it does not set them. Latent for any caller whose input format differs
  from its scaler/mask format; here it broke `float32` + non-aligned `W` + `Wt ≥ 4`
  (`unp_B_src_format mismatch` under `--dev`, PCC 0.9990 in production). Two acceptance
  cells caught it, which is why the pad-poison re-run the verifier notes demanded mattered.
* **Both partial-`W` mechanisms stay live and covered.** The threshold means the pad-poison
  shapes span `Wt = 2, 3` (partial-SCALER pair) and `Wt = 5, 7` (0/1 MASK tile); both are
  clean, median got/true ratio within 0.7 % of 1.0.
* **The 2.87–2.94× that Refinement 4 item (a) predicts does NOT translate to this op.**
  Measured whole-op A/B at the `_perf_case` config (bf16 / HiFi2 / `fp32_dest_acc_en=False`,
  one fresh-cache profiled run per variant): `(1,1,32,7168)` 44690 → 42253 ns (**1.06×**),
  `(1,1,224,3072)` 23758 → 22544 ns (1.05×), `(1,1,32,1024)` 11132 → 10881 ns (1.02×),
  `(1,1,8192,5120)` 754579 → 752410 ns (1.00×). A small uniform win, no shape slower.
  The gap is the finding: rms_norm is **dataflow-bound** at these widths, so shaving reduce
  MATH cycles moves the total a few percent — Refinement 4 should score item (a) as already
  landed and spend its budget on the byte-count / occupancy levers instead.

---

### [~] Refinement 2 — Sharded placements: local HEIGHT shard + cross-core WIDTH/BLOCK combine

**Goal**: add all three sharded values to `SUPPORTED["memory_layout"]` —
`HEIGHT_SHARDED`, `WIDTH_SHARDED`, `BLOCK_SHARDED` — natively, for both layouts and for a
sharded *output* (`memory_config=` is already threaded through `validate()` and the
allocator; the golden harness requests an output shard matching the input's). Two pieces
of work, one refinement because they share all of the placement plumbing:

* **HEIGHT_SHARDED — knob-turn** (`op_design.md` Lamp L3, §5.3). The shard cuts the
  *independent* `row` axis, so each core already holds whole rows and the reduction stays
  **local**: no combine, no new math. `cb_input_tiles` / `cb_output_tiles` become
  `ttnn.cb_descriptor_from_sharded_tensor(...)` — zero-copy, **no NoC read for x** — and
  `BLOCK_ROWS` defaults to the shard's full tile-row count (sub-chunk only under L1
  pressure). Do this half first: it establishes the CB-placement path that WIDTH/BLOCK
  also need.
* **WIDTH_SHARDED / BLOCK_SHARDED — scheme-change** (Lamp L4 ⊃ L1). These cut the
  **dependent** `width` axis, so per-core partial `Σx²` must be combined across the grid.
  Required topology, already specified in `op_design.md` §3.4 — build it as written:
  each core packs its `BLOCK_ROWS` raw partials into a **dedicated** `cb_sum_handoff`
  (never `cb_row_stat`, which is a compute accumulator — a dataflow reader on it would be
  a second consumer); non-root cores `noc_async_write` into a per-sender slot of the
  root's `cb_partials_gathered` + `noc_semaphore_inc`; the root sums, runs the *same*
  `transform_in_place` finalize, and multicasts the finalized stat tiles back with
  `mcast_pipe.hpp`'s `SenderPipe::send()` / `ReceiverPipe::receive()` (host side:
  `Mcast1D` + `McastConfig`, `Mcast1DShape::PerRow`). `GRID_W` / `GRID_H` and the per-core
  `(w_start, w_count)` extents are read off the shard spec instead of chosen. No output
  combine is needed — the output is width-partitioned.

**Verifier notes**:
* No skill in the inventory covers placement/`memory_layout`; do **not** reach for
  `/memory-layouts` (that is RM↔TILE layout, not placement). The two patterns you need are
  named above: `ttnn.cb_descriptor_from_sharded_tensor` for the local shard, and the §3.4
  partial-reduce + mcast contract for the cross-core one.
* **Native or nothing.** Re-reading a core's own local shard through a `TensorAccessor`
  does not implement `HEIGHT_SHARDED`; it merely tolerates it (each core happens to hold
  full rows, so the accessor reads the right bytes and the golden cells go green). That
  would be an unimplemented axis value claimed in SUPPORTED. Verify the dataflow, not the
  test colour.
* Ordering: hardest-first, and it is also a hard dependency for the perf track —
  Refinement 3's target shapes are `Rt = 1` decode profiles where the row split can only
  ever fill **one** core, so the cross-core width combine built here is the mechanism the
  perf phases then tune. Build it performantly, not correct-only: it ships already filling
  the grid, with both dataflow halves batched at whole-tile granularity, per the same
  performance-conformance bar as Phase 0.
* `GRID_W` is currently a live knob pinned at 1 with an explicit
  `NotImplementedError` guard in `create_program_descriptor` — turning it up is this
  refinement's entry point; delete the guard as part of the change.
* Shard specs on the golden side come from `eval.sharding.auto_shard_config` (and
  `shard_config` for the perf cases' pinned geometries), so there is **no test work per
  scheme**. Watch `infeasible_skipped`: a shard geometry that doesn't fit the live device's
  L1 is uncharged, not a failure — don't chase those two cells.
* `feature_spec.INVALID` permanently skips
  `{layout: ROW_MAJOR, memory_layout: *_SHARDED, gamma_layout: TILE}` (all three schemes),
  so RM-activation + TILE-gamma sharded cells are out of scope here by construction. See
  the INVALID audit in `verification_report.md` — those entries are mis-categorised
  (author-scoped, not structural) but they are the harness's contract today; do not work
  around them.

**Done when**: the 4834 `memory_layout`-blocked xfail cells (including the 279 sharded
loose cases and the sharded `_perf_case` geometries) pass, with `supported_fail = 0` and
no regression on the interleaved cells.

**Outcome** (`[~]` partial): all three sharded values are LIVE in
`SUPPORTED["memory_layout"]` and NATIVE for TILE activations — `cb_input_tiles` /
`cb_output_tiles` are `ttnn.cb_descriptor_from_sharded_tensor` (zero-copy, **no NoC read
for x at all**), and the §3.4 cross-core width combine is built as specified (dedicated
`cb_sum_handoff` → per-sender slot of the root's `cb_partials_gathered` +
`noc_semaphore_inc` → root sums + runs the *same* `transform_in_place` finalize → stat
multicast back via `mcast_pipe.hpp`). `GRID_W`'s `NotImplementedError` guard is gone; the
knob now drives the *same* combine on an interleaved input (Lamp L1, the lever
Refinement 3 needs), clamped to a divisor of `Wt` and parked at its byte-identical 1.
Golden: **387 / 387 loose cases** (299 pass, 3 infeasible-skipped, 85 xfail) and the full
**40320-cell cartesian** (4407 pass, 1995 xfail, `supported_fail = 0`), zero hangs, no
regression (unit dir 298/298).

Deferred to **Refinement 2b**: `{ROW_MAJOR, WIDTH_SHARDED}` and
`{ROW_MAJOR, BLOCK_SHARDED}` are now op-side `EXCLUSIONS`. Five findings:
* **Two host emitters, one kernel.** A WIDTH shard grid is row-major-*packed*, not a
  rectangle (64 cores on an 11-wide grid = 5 full rows + 9), so its group is ONE
  `Mcast2D` over the bounding box with the few in-box/out-of-shard cores joining as
  INACTIVE (`row_count == 0`) purely so the stat multicast lands in a `cb_row_final` this
  program owns. BLOCK is a true rectangle whose grid ROWS are the groups — `Mcast1D`
  `PerRow`. Both emit the identical 5-word CT / 4-word RT wire, so `McastArgs` decodes
  either and the kernel is one code path.
* **`Mcast1D`'s per-row sender rect EXCLUDES the sender** (`sender_rect_`), while
  `Mcast2D`'s contains it — so the BLOCK root never received its own finalized stat
  (PCC 0.91). The root now places its own copy and broadcasts **in place**
  (`src == dst` ⇒ EXCLUDE-source), which makes the two emitters behave identically.
* **`Semaphore::up(value)` is NON-ATOMIC** (a local read-modify-write; the header says so).
  The root's self-signal raced the members' remote atomic incs and dropped one — a hang in
  whichever group lost, one group in eight. The root writes its own slot synchronously, so
  it waits for `GROUP_SIZE - 1` and never bumps the counter itself.
* **`L1_SAFETY_FRACTION` cannot cover a FIXED offset.** The CB arena starts 70656 B above
  the worker-L1 unreserved base, and metal's check is absolute ("static circular buffer
  region ends at 179072 / L1 buffer allocated at 163840"). A shard pair holding 1.38 MB of
  1.53 MB leaves 52 kB of real headroom while 0.85 of the nominal remainder claims 105 kB.
  New knob `L1_CB_ARENA_BASE_RESERVE`, subtracted only when a shard is L1-resident, so
  every interleaved build stays byte-identical.
* **A resident shard can switch Refinement 1b's precision fix OFF** (descriptor **D8**).
  The `AccumulateViaAdd` crossover was measured against `WT_CHUNK`; a 344-tile shard
  squeezed `WT_CHUNK` to 2, dropping below the threshold and bringing rms 0.127 back on
  exactly the cells 1b closed. The gate is now this core's WHOLE reduce dim
  (`wt_per_core`) — identical in RESIDENT, and the total is what decides once L1 chunks.

Known non-issue, recorded so it is not re-chased: `test_translated.py::
test_rms_norm_sharded_uneven_multicore_logical_width[w200_c3_nonaligned-bfloat8_b]`
fails at frobenius 0.112 vs a 0.10 budget. Measured **bit-identical INTERLEAVED and
WIDTH_SHARDED (0.11224 both)** — it is the `{bfloat8_b, w_non_aligned}` corner
`feature_spec.INVALID` already declares out of scope (a 1000.0 pad poison raises the
shared exponent of the block that straddles the logical width), not a placement bug. The
other 11 params of that test went from *all failing* (the op refused the placement) to
passing.

---

### [x] Refinement 2b — ROW_MAJOR shards that cut the width axis

**Goal**: close the two `EXCLUSIONS` Refinement 2 added —
`{layout: ROW_MAJOR, memory_layout: WIDTH_SHARDED}` and
`{..., BLOCK_SHARDED}`. Everything else about sharding is done and native; this is the
one placement × layout corner left.

**Why it is not a knob-turn.** `eval.sharding` rounds an RM shard edge to
(1 stick × `L1_align / elem_size` elements) — 8 for bf16, 4 for fp32 — and a shard may not
hold a partial page, so the tensor's **page becomes the shard's row SEGMENT, not the row**:
`(1,1,224,3072)` width-shards to `[224, 32]` and `(1,1,256,512)` to `[256, **8**]`. Two
consequences, both measured:
* no core holds a whole width TILE, so the tile-granular combine Refinement 2 built cannot
  be mounted on the placement; and
* the SCHEME_ROWS fallback cannot reach a row either — `read_sticks_for_tilize` keys on the
  page index, so a stick read lands inside one segment and runs off the end of the shard
  (**PCC 0.005**, plus out-of-bounds L1 traffic that cascaded every later test in the same
  process into dispatch failures until the cell was excluded).

**The exact next lever**, in preference order:
1. **Segment-gathered staging, gated on cost.** `read_sticks_for_tilize` already takes
   `start_page` + `byte_offset_within_page`; a row is reachable as
   `nw = ceil(W_gran / shard_w_gran)` reads per stick. Viable only when `nw` is small —
   it is 96 for `(1,1,224,3072)` and 8 × 100 000 sticks for `(99991,64)`, so this needs an
   explicit `nw` ceiling with SCHEME_ROWS-on-a-DRAM-copy or a refusal above it. Verify the
   page ORDER first (`page = stick * nw + segment` is the assumption, unconfirmed).
2. **Native band tilize when `shard_w % 32 == 0`.** When the RM shard's width happens to be
   a whole number of tile columns (`per_w * w_gran % 32 == 0`, true for
   `(1,1,224,3072)`: 32 elements = 1 tile), the band IS resident and the core can stage it
   from its own L1 with `shard_h` local reads, then join the *existing* width combine
   unchanged. That covers the aligned subset natively and leaves only sub-tile shards to
   lever 1 — likely the best value-per-line here.

**Verifier notes**:
* Do **not** reach for `ttnn.to_memory_config` / `to_layout` to sidestep this: the op's
  contract is native placement, and a host-side relayout would also be a new tensor.
* `feature_spec.INVALID` already skips `{ROW_MAJOR, *_SHARDED, gamma_layout: TILE}`, so the
  in-scope cells all pair RM activations with RM gamma. The gamma side already works —
  `stage_gamma_chunk`'s RM branch reads gamma's own (interleaved) sticks and is
  placement-independent.
* The regression net for lever 1 is the `_RESILIENCE_SHAPES` × ROW_MAJOR × {WIDTH, BLOCK}
  cells (2 per shape × 44 shapes) plus `test_rms_norm.py`'s RM regime-pinned cases. Watch
  for the out-of-bounds signature specifically: a wrong page mapping shows up as a
  *cascading* dispatch failure in later tests, not as a local PCC miss.

**Outcome** (`[x]` full): both cells are CLOSED and the op's `EXCLUSIONS` list is back to the
single `{float32, fp32_dest_acc_en: False}` entry — **zero new exclusions**, and the
layout × placement rectangle is now complete and native throughout. Golden: **387 / 387 loose
cases** (384 pass, 3 infeasible-skipped, **0 xfail** — the 85 this refinement's cells used to
occupy all pass) and the full **40320-cell cartesian** (5037 pass vs Refinement 2's 4407,
1365 xfail vs 1995, `supported_fail = 0`, `xpass_drift = 0`), zero hangs, unit dir 406/406.
The only remaining golden failure is the pre-existing `test_translated.py` bf8b pad-poison
cell, bit-identical at **frobenius 0.11224** to the value Refinement 2 recorded as a known
non-issue.

Neither listed lever was the answer, and the reason is the finding: **both assumed a core has
to reach a whole ROW.** It does not. The §3.4 combine sums the group's per-row partials
**elementwise**, so a partial may cover *any* contiguous element range — a band need not start
or end on a tile column. So each core reduces the band it already holds, staged out of its
**own L1** (`_plan_band`, descriptor **D10**): lever 2 generalized off its
`shard_w % 32 == 0` precondition (it survives as the *contiguous fast path* — one local
transaction per tile-row instead of one per stick), and lever 1's `ceil(W / shard_w)` reads per
stick are never paid at all, so no `nw` ceiling and no refusal is needed. Three findings:
* **An unaligned DRAM read offset is silently TRUNCATED to the 64-byte alignment.** Staging at
  the band's own byte offset gave bands 1–3 of an 8-element shard `gamma[0..8)` — PCC 0.32
  *while band 0 and every spot-checked position read exactly 1.000*. Fixed by staging in the
  tensor's **global tile frame** (the band's first element at lane `w_off_elems % 32`), which
  keeps every gamma fetch on a tile column for x's dtype and gamma's independently. That is
  also why **TILE gamma works on this path**, so the corner
  `feature_spec.INVALID` reserves needed no op-side exclusion.
* **The reduce mask is replaced, not generalized.** A band boundary is per-core and cannot be
  one program-wide `PARTIAL_W`; it does not need to be. The staging ring is zeroed once at
  boot and only the band's bytes are ever written into it, so every lane outside the band
  multiplies to an exact 0. `kernel_partial_w` is 0 on this path.
* **A latent deadlock in Refinement 2's writer, now fixed.** Its combine ran *all* row-blocks
  before *any* output write, but compute cannot finish block `blk+1`'s pass A until block
  `blk`'s pass B is drained — so it deadlocks the moment `num_blocks` exceeds the output CB's
  depth. The TILE schemes never hit it (their shard is one row-block); the band scheme hit it
  on the first shape tried, because its per-block gather CB is `GROUP_SIZE` fp32 tiles and L1
  caps `BLOCK_ROWS` low. The two are interleaved per block now.

Perf, measured for the record (not a gate — this is a generality refinement): the band's own
cost over the **equivalent TILE shard at the same placement** is **+2.7 %** on
`(1,1,224,3072)` WIDTH (133224 → 136856 ns, a 96-core group) and **+21 %** on the same shape
BLOCK (10373 → 12509 ns, an 11-core group, and still **2.7× faster than interleaved**). The
one large gap, `(1,1,256,512)` WIDTH at 29004 → 96479 ns, is **not** the band: an RM shard's
8-element granule makes `auto_shard_config` cut W=512 into 64 slices where the TILE granule
cuts it into 16, so the same tensor gets a 4× larger combine *group*. The lever a later perf
round would attack is the sub-tile band's one-local-read-per-stick staging
(`test_rms_norm_perf.py::test_rms_norm_perf_row_major_band` pins both granularities).

---

### [x] Refinement 3 — Speed up the wide/decode profiles (post-combine)

**Type**: perf

**Goal**: no `LOOSE_CASE` carries an `attention:` perf-focus note, so shape selection is
mine: take the **interleaved decode profiles** from `feature_spec.LOOSE_CASES`' `_perf_case`
table — primarily `(1, 1, 32, 7168)` (DeepSeek-V3 decode, `achievable_ns = 104259`,
`minimum_expected_speedup = 7.0` ⇒ a **≤ 14894 ns** goal at 1350 MHz) and
`(1, 1, 32, 1024)` (`achievable_ns = 9149`) as the narrow-`W` control. Their full config is
`bfloat16 / TILE / INTERLEAVED / fp32_dest_acc_en=False / math_fidelity=HiFi2`, soft
`pcc_threshold = 0.9995` — measure and optimize **exactly that config** (Refinement 1
guarantees it is in SUPPORTED; never substitute a `fp32_dest_acc_en=True` stand-in, it is a
different datapath). Remember to clock-scale the reference:
`scaled_ns = achievable_ns × 1350 / actual_aiclk_mhz`, then divide by
`minimum_expected_speedup` where present. Pick levers from
`ttnn/ttnn/operations/examples/master.md`; the relevant situation here is *grid occupancy
plus DRAM bytes*, because `Rt = 1` means the Phase-0 row split fills exactly one core and
wide `W` additionally lands in the STREAM regime (x read twice). No SUPPORTED change.

**Done when**: measured device-ns improves on `(1,1,32,7168)` and `(1,1,32,1024)` at their
declared config and moves toward the clock-scaled goal; their soft `pcc_threshold = 0.9995`
still holds; the golden suite is green; and no regression across the config-spanning guard
set (one representative per distinct kernel path × layout × placement — at minimum:
TILE/interleaved RESIDENT, TILE/interleaved STREAM, ROW_MAJOR/interleaved,
HEIGHT-sharded, WIDTH-sharded, plus one `no_gamma` and one `w_non_aligned` cell).

**Verifier notes**: depends on Refinement 2 — the cross-core width combine is the only
lever that fills the grid on `Rt = 1`, and the ≥7× requirement on the 7168 case is not
reachable by knob-tuning a single-core kernel. Use `/perf-ceiling-dm` first on the narrow
control shape: `(1,1,32,1024)` may already be near its single-shot DRAM roofline, in which
case only the wide case should move and that is the honest result to record.

**Outcome** (`[x]` full): the named lever — `GRID_W`, Refinement 2's interleaved
cross-core width combine, parked at its byte-identical 1 — is now an AUTO policy
(descriptor **D11**, `_auto_width_split`), and **both goals are met**. Measured on device
at the declared `_perf_case` config (bf16 / TILE / INTERLEAVED / `fp32_dest_acc_en=False` /
HiFi2, blackhole p150b, 110-core 11×10 grid, CHIP_FREQ 1350 MHz == the reference clock so
no scaling is needed; one fresh-cache profiled run per variant, reproduced twice within
2 %): **`(1,1,32,7168)` 41803 → 12756 ns (3.28×)**, which is **8.17× the 104259 ns
reference — above the required 7.0× and inside the ≤ 14894 ns goal** — and
**`(1,1,32,1024)` 11196 → 7199 ns (1.56×)**, beating its 9149 ns reference by 1.27×.
`pcc_threshold = 0.9995` holds (0.99998 / rel RMS 0.0087 and 0.99998 / 0.0069). The whole
decode family moved with it: `(1,1,32,8192)` 3.46×, `(1,1,32,5120)` 2.92×,
`(1,1,32,4096)` 2.59×, `(1,1,32,2304)` 2.43×, plus the few-row shapes
`(1,1,128,4096)` 1.72×, `(1,1,224,3072)` 1.29×, `(1,1,224,1000)` 1.29×,
`(1,1,512,4096)` 1.20×. Golden: **5421 pass / 1365 xfail / 0 fail** (identical to
Refinement 2b's counts — no SUPPORTED change), `test_regression.py` 15/15,
`test_translated.py` 105/106 with the one failure bit-identical at frobenius **0.112240**
to the pre-existing bf8b pad-poison non-issue. Unit dir 434 passed / 30 skipped, zero
hangs. Three findings:
* **The group size has a measured optimum, so the ceiling is a knob, not a guess.** Per-core
  bytes fall as `1/gw` but the root's gather RISES with `gw` (every member ships a full
  fp32 tile per row-block into one root): on `(1,1,32,7168)` gw = 1 → 41803, 8 → 13876,
  **16 → 12978**, 32 → 14487, 56 → 19428 ns. `WIDTH_SPLIT_MAX_GROUP_CORES = 16` and
  `WIDTH_SPLIT_MIN_WT_PER_CORE = 4` (the narrow control's own optimum: gw = 8 at 4 tiles
  per core) are both measured, not assumed.
* **A gain threshold was REQUIRED to avoid a regression.** At `WIDTH_SPLIT_MIN_GAIN = 2`,
  `(1024,1024)` (Rt = 32) split 32 → 80 cores and got **slower**: 21560 → 23315 ns (0.92×).
  2.5× more cores cannot pay for a combine round when the row split already feeds 32 cores.
  At the shipped `MIN_GAIN = 4` that shape, `(1,1,2048,256)` and every prefill stay
  byte-identical on the Phase-0 row split, and every shape that does split measured ≥ 1.20×.
* **What is left is the COMBINE, quantified.** A one-core minimal program is 3348 ns of
  fixed launch/dispatch floor, and at gw = 16 the 7168 case moves only 56 kB per core
  (≈1.8 µs at the measured 32 GB/s single-core NoC), so ~7 µs of its 12756 ns is the gather
  → root sum → stat-multicast round trip, which cannot overlap anything because `Rt = 1`
  gives each core a single row-block. The next levers are therefore (a) a hierarchical
  two-stage gather (`examples/tensix_all_reduce`: 1.45–1.60× over a flat root on 2-D groups,
  and it would raise the useful group ceiling, letting more cores share the payload) and
  (b) a compact partial handoff — a `REDUCE_ROW` partial is a 32-float column vector shipped
  inside a 4096-byte tile, so the gather moves 128× the bytes it needs. Both are changes to
  the combine's topology / data format rather than knob turns, so they are recorded in D11
  and left un-built here.

---

### [x] Refinement 4 — Prefill + sharded-geometry perf, and the block/depth knob surface

**Type**: perf

**Goal**: the remaining `_perf_case` rows — the prefill profiles
`(1, 1, 8192, {1024, 2304, 5120, 7168})` (`achievable_ns` 96744 / 211345 / 738307 /
1032281, interleaved) and the measured-fastest sharded geometries
(`(1,1,32,{1024,2304,5120,7168})` WIDTH_SHARDED and `(1,1,8192,1024)` BLOCK_SHARDED with
their pinned `shard_shape` + `core_grid`), all at `bfloat16 / fp32_dest_acc_en=False /
HiFi2`. Prefill is the opposite regime from Refinement 3 (`Rt = 256`+ fills the grid, so it
is bandwidth-bound, not occupancy-bound) — the relevant levers from
`ttnn/ttnn/operations/examples/master.md` are therefore the cheap (⭐/⭐⭐) block-surface
ones, several of which fit in one phase:
* **co-tune the block/depth knobs** already exposed in `rms_norm_program_descriptor.py`
  (`BLOCK_ROWS` via `L1_SAFETY_FRACTION`, `CB_DEPTH_CANDIDATES`, `CB_RM_STAGE_DEPTH`) —
  block size trades L1 for reuse/reconfig, depth trades L1 for movement↔compute overlap;
  the master.md granularity floor (whole tiles minimum, coarser amortizes) bounds the
  search. Note the recorded null result: depth `(2, 1)` was measured at 0.83× and reverted
  (see the descriptor's **D4** table) — re-test it only *after* the width split gives a
  core many blocks again;
* the `op_design.md` **Lamp L6** micro-knobs, each already a parameter: (a) `reduce<>`'s
  `ReduceAlgorithm::AccumulateViaAdd` (+ `prepare_reduce_mask` in place of the partial
  scaler), measured **2.87–2.94×** on `REDUCE_ROW` at `Wt ≥ 4`; (b) `rsqrt` scoped to
  `VectorMode::C`, **1.94×** on that step; (c) eliding dtype reconfig when every CB shares
  one format, up to **1.19×**; (d) `DestAccumulation::PerRow` to delete `cb_x_squared`'s L1
  round-trip;
* the **prime-`Wt` STREAM cliff** (`verification_report.md` § Recommendations 2): D1 forces
  `WT_CHUNK | Wt`, so `Wt = 127` (`W = 4064`, a `_RESILIENCE_SHAPES` entry) collapses to
  one tile per chunk and per NoC barrier. A ragged-tail chunk (runtime `wt_c`) or Lamp L5
  removes it.
No SUPPORTED change.

**Done when**: measured device-ns improves on at least the prefill profiles and one sharded
geometry at their declared configs (or every lever is measured and correctly reverted with
the null result recorded — that is a completed investigation); golden suite green; no
regression across the same config-spanning guard set as Refinement 3.

**Verifier notes**: keep each lever's measurement separate — (a) and (d) both touch the
reduce/square path and will confound each other if applied together. (a) is a *coupled*
change, not a one-word swap: `AccumulateViaAdd` restricts `Accumulate` to SUM +
`BulkWaitBulkPop` and swaps the partial-W mechanism from a scaler tile to a 0/1 mask tile,
so re-run the poisoned-padding cells (acceptance + the 24 pad-poison loose cells) on it
specifically — they are the only tests that can catch a masked-reduce regression, and a
padding leak on a wide row is a near-uniform scale error that PCC is largely blind to (the
precision baseline's ratio-spread check is the second net).

**Outcome** (`[x]` full): four levers landed, each measured separately and each left in the
tree as a live knob; **both halves of the goal are met** and no supported shape is slower.
Measured at the declared `_perf_case` config (bf16 / TILE / HiFi2 / `fp32_dest_acc_en=False`,
blackhole p150b, 110-core 11×10 grid, CHIP_FREQ 1350 MHz == the reference clock; one
fresh-cache profiled run per variant, medianed over 3 only where a number sat on the ~3 %
noise band). **Prefill: `(1,1,8192,5120)` 753345 → 468093 ns (1.61×) and `(1,1,8192,7168)`
1043918 → 643320 ns (1.62×)**, both now FASTER than their `achievable_ns` (738307 /
1032281); the two narrow prefill rows (`W = 1024`, `2304`) were already at the DRAM roofline
and are flat. **Sharded: all five pinned geometries improved — BLOCK `(1,1,8192,1024)` 64c
102173 → 83316 ns (1.23×)**, and the four WIDTH geometries 1.07×–1.11×. Golden **5421 pass /
1365 xfail / 0 fail** (identical to Refinement 3 — no SUPPORTED change), unit dir 463 passed /
30 skipped, zero hangs.
* **What the levers were.** Prefill was won by **Lamp L5**, the op's third compute regime
  (descriptor **D14**): `X_RESIDENT` is now an explicit flag decoupled from
  `NUM_W_CHUNKS == 1`, so one whole tile-row of x *and the whole row of gamma* stay resident
  while only the derived CBs are chunked. Pass B re-reads nothing and gamma is read once per
  core instead of once per pass-B chunk of every row-block — on `(1,1,8192,7168)` that is
  470 MB → 285 MB at the same measured 443 GB/s. No second code path: the held CBs are
  indexed at a `TileOffset::Set` base that folds away off this regime. The sharded
  geometries were won by **Lamp L6b** (`rsqrt` scoped to `VectorMode::C`, **D15**, 1.14× on
  the BLOCK shard alone), with **L6d** (`DestAccumulation::PerRow` on pass A's square,
  **D12**) and a compact combine gather (**D13**) worth 1.02×–1.03× each. Item (a),
  `AccumulateViaAdd`, was already landed by Refinement 1b and only verified.
* **The bottleneck is not where the tile-op count said it was, and that is the main finding.**
  Halving the member→root gather bytes moved the 64-core BLOCK shard only ~5 %, so the
  combine is not byte-bound at these group sizes. What dominates is the ROOT's per-round
  `transform_in_place`: one SFPU rsqrt per tile-row (32 on that shard) with a
  `GROUP_SIZE`-wide fan-out of members blocked behind it — a per-tile SFPU cost invisible in
  an op count. The per-RISC split in the profiler CSV (BR spanning the kernel while its own
  NoC work is microseconds) is what showed it.
* **L5 is not free when it costs CB depth, and that is measured both ways.** Holding a whole
  tile-row can leave no L1 for the depth the cross-processor CBs spend on movement↔compute
  overlap. `(1,1,32,7168)` at `GRID_W=1` (one core, depth 2→1) went 41779 → 50598 ns
  (**0.83×**), while ROW_MAJOR `(1,1,32,4096)` — already depth 1 in *both* regimes, so no
  sacrifice — went 52197 → 47144 ns (**1.11×**) at the same single row-block. So the gate is
  on the **depth sacrifice**, not on L5, and only then on the row-block count.
* **The block/depth knob surface was re-measured and is null at its shipped values.**
  `CB_DEPTH_CANDIDATES = (2,1)` is within noise (D4's null still holds, and L5 has taken over
  the residency band that knob targeted); `CB_RM_STAGE_DEPTH = 3` within noise;
  `L1_SAFETY_FRACTION = 0.90` is +2.2 % on the BLOCK shard but was **declined** — Refinement
  2's L1 finding is that a proportional margin cannot cover the arena's fixed base offset, and
  a CB-OOM is an op-charged hard failure, not a slow path.
* **What is left, quantified, and why it was not built.** Prefill is at the DRAM roofline with
  near-minimal bytes; the only byte left is gamma's 50 MB of per-core redundancy, i.e. **Lamp
  L2** (a scheme change). The sharded geometries are still 2–3× off `achievable_ns`, and after
  L6b the residue is the combine ROUND COUNT — the 64-core BLOCK shard runs 4 rounds because
  `cb_partials_gathered` (`GROUP_SIZE × BLOCK_ROWS` fp32 pages) caps `BLOCK_ROWS` at 10; the
  lever is D11's recorded compact handoff in its real form (a partial packed to its 32 useful
  floats, which needs a transpose) plus D11's hierarchical gather, both combine
  data-format/topology changes. The prime-`Wt` cliff survives: L5 removes `Wt = 127`'s pass-B
  re-read but not its 127 one-tile phases, so a ragged-tail chunk is still the lever. **Lamp
  L6c** was costed rather than built: at master.md's ≈110–150 ns per reconfig and the measured
  count, its ceiling is 1.2 % on the BLOCK shard and 0.5 % on prefill 7168, against a
  silent-corruption risk on the mixed-dtype gamma paths.
