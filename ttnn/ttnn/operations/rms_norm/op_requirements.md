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

### [x] Refinement 2 — Tiled-gamma layout support

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

### [x] Refinement 3 — Speed up the interleaved prefill perf profile

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

### [~] Refinement 4 — Cross-core W-split: WIDTH/BLOCK sharding + logical wide-interleaved split

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

**Landed (partial)**: `WIDTH_SHARDED` + `BLOCK_SHARDED` in SUPPORTED; the native
cross-core combine (zero-copy sharded `cb_x_in`/`cb_out`; Pattern-A all-unicast
gather → master fold → broadcast `1/RMS`; 3 monotone counter semaphores; one
round per tile-row so no CB grows with the tile-row count) for **TILE input +
TILE/no gamma**. All WIDTH/BLOCK golden loose + `_perf_case` geometries pass
(18/18 loose, PCC ≥ 0.99999); 1160 sharded cartesian cells pass, 0 supported_fail.
Deferred to R4a (RM tilize on the cross-core path — see below). The logical
wide-interleaved / decode W-split (shapes that under-fill the grid) is NOT yet
wired — those still run the interleaved row-parallel path (correct, single-/few-core);
this is a perf-parallelism gap folded into R4a, not a correctness gap.

---

### [~] Refinement 4a — Cross-core W-split: RM tilize path + logical interleaved W-split

**Goal**: complete R4's deferred corners, all needing a tilize/untilize on the
cross-core (`rms_norm_xcore_*`) kernels or a new host dispatch:
- **RM gamma + sharded** (TILE input): the reader reads the gamma W-slice as
  row-major sticks and the compute tilizes them (`ckl::tilize`, runtime tile count)
  before the pass-2 `·gamma` — mirror the interleaved RM-gamma knob. Remove the
  two `{gamma_layout: ROW_MAJOR, memory_layout: *_SHARDED}` EXCLUSIONS.
- **RM input + sharded**: consume the resident RM shard via a tilize-on-read into
  `cb_x_in`, and untilize `cb_out` back to the sharded RM output. Remove the two
  `{layout: ROW_MAJOR, memory_layout: *_SHARDED}` EXCLUSIONS.
- **Logical wide-interleaved / decode W-split**: for wide/few-row INTERLEAVED
  shapes that under-fill the grid (`W=16384/32768/12288`, decode `rows=32`),
  add a host heuristic that assigns `K` cores per tile-row, each reading its `W/K`
  slice from DRAM (per-core column offset via `TensorAccessor`), and route them
  through the same cross-core combine. Currently these run correct-but-single/few-core
  on the interleaved path. This is the parallelism R6's decode perf pass builds on.

**Verifier notes**: the cross-core combine itself is done and correct (R4) — R4a
is the RM-layout tilize plumbing + the interleaved-side per-core slice reader, both
reusing the existing xcore compute/writer. The ~1840 deferred cartesian cells are
the RM-input/RM-gamma sharded EXCLUSIONS; they xfail-strict today (not silenced).

**Done when**: the four EXCLUSIONS above removed with their cells passing; wide
interleaved / decode shapes fill more than one core; golden green; prior phases
unregressed.

**Landed (partial `[~]`)**: two of the three corners shipped, reusing the R4 xcore
combine (kernels + topology) via CT-arg flags — no forked kernel files:
- **RM gamma + sharded** (TILE input): `GAMMA_IS_RM` flag on the xcore reader/compute;
  the reader reads the gamma W-slice as row-major sticks (one tile-column per
  `read_sticks_for_tilize`), compute `tilize<1,cb_gamma_sticks,cb_gamma>(vwt)` before
  the pass-2 `·gamma` (held resident). gamma stays interleaved DRAM. The two
  `{gamma_layout: ROW_MAJOR, memory_layout: *_SHARDED}` EXCLUSIONS are **removed** →
  +598 RM-gamma sharded cells xfail→pass (full golden `test_op` 2660→3258 passed,
  0 failed, 0 xpassed; probes PCC ≥ 0.999996 incl non-aligned W).
- **Logical wide-interleaved / decode W-split**: `_create_logical_xcore_descriptor`
  (one group of `K = min(Wt, num_cores)` cores splits W, `HT_LOCAL = R`) + host trigger
  (TILE input, INTERLEAVED, `R < num_cores`, `Wt > R`). Kernel flags `X_FROM_DRAM`
  (reader reads its W/K slice tiles from interleaved DRAM into `cb_x_in`), `X_ZERO_COPY=0`
  (compute waits on the reader's push), `OUT_TO_DRAM` (writer drains `cb_out` to DRAM
  per tile-row). Wide (`W=16384/32768/12288`) + decode (`rows=32`) now fill `K>1` cores
  instead of 1–2 (probes PCC ≥ 0.999996; prefill `R≥cores` stays on the R3 resident path).

**Deferred to R4b** (structural, characterized at depth): **RM input + sharded**. RM
WIDTH/BLOCK `auto_shard_config` uses a width granule of 8 (bf16) / 4 (fp32), so the
per-core resident W-slice is width-`8·k` — **never a multiple of 32 for any golden W**
(W=64→8el, W=1024→16el, W=4096→40el, W=8192→80el). Core boundaries straddle tile-column
boundaries, so the tile-based cross-core reduce (whole `per_w_t` tiles per core + a
single partial-holder) cannot consume the shard. The two
`{layout: ROW_MAJOR, memory_layout: *_SHARDED}` EXCLUSIONS stay xfail-strict (see R4b).

---

### [x] Refinement 4b — RM-input sharded: per-core arbitrary-width tilize sub-scheme

**Goal**: complete R4a's deferred corner — **RM input + WIDTH/BLOCK sharded** — by
handling a resident RM shard whose per-core width is an arbitrary multiple of the RM
granule (8/4 elements), NOT a whole number of 32-wide tiles. Remove the two
`{layout: ROW_MAJOR, memory_layout: *_SHARDED}` EXCLUSIONS with their cells passing.

**Landed (full `[x]`)**: RM-input WIDTH/BLOCK sharded lands via an `IS_RM` CT flag on the
three R4 xcore kernels (no forked files), reusing the cross-core combine + transport
unchanged. The sub-scheme:
- **Phase-align to the global tile grid** — a core's slice starts at element column
  `w_offset` (sub-tile). `g0 = w_offset//32` is the first global tile; `phase = w_offset%32`
  the leading offset. The reader **loopback-repacks** the resident RM shard sticks
  (`cb_descriptor_from_sharded_tensor` alias → local NoC loopback, no remote re-fetch)
  into tile-padded `cb_x_sticks` at column `phase`; compute `tilize`s `ceil((phase+sw)/32)`
  padded tiles.
- **Per-core partial scaler** — leading `[0,phase)` columns stay 0 (contribute 0 to the
  SUM reduce), and EVERY core masks its trailing `valid_end%32` lanes via
  `ReducePartialScaler::last_tile_at(1)` (reader emits full+partial scaler tiles). The
  associative cross-core sum stays correct.
- **Aligned gamma** — the gamma W-slice is read at the tile-ALIGNED global column
  `(g0+wt)*32` (a sub-tile DRAM offset faults), matching x's phase-aligned tiles; gamma
  applies to the `vwt` valid tiles, copy elsewhere.
- **Untilize back** — compute `untilize`s `cb_out`; the writer loopback-writes the valid
  columns `[phase, phase+valid_cols)` into the resident RM output shard.
Both `{layout: ROW_MAJOR, memory_layout: *_SHARDED}` EXCLUSIONS removed. Probes 13/13
PCC ≥ 0.99997 (WIDTH/BLOCK × gamma/no-gamma × non-aligned H/W × wide (per_w_t=2) × fp32);
golden `test_op` sharded slice across ~20 shapes: 0 failed / 0 xpassed, RM-input cells
xfail→pass (only `{f32, acc=False}` stays xfail via the other EXCLUSION); loose 18/1
(HEIGHT=R5); unit dir 345 pass / 32 skip; interleaved + TILE-sharded unregressed.

**Verifier notes**: this is a **sub-scheme change**, not the RM-gamma tilize knob-turn
(R4a shipped that). The exact levers, in one place:
- Tilize the resident RM shard (a `cb_descriptor_from_sharded_tensor` on the RM buffer
  → stick pages, or the `compute_block_size` page-override to tile pages) into
  `ceil(shard_w / 32)` **padded** tiles per tile-row — the shard width is arbitrary
  (e.g. 40 el → 1 full tile + an 8-wide partial tile).
- Per-core partial scaler: EVERY core (not just the globally-last one) zeros the last
  tile's `[shard_w % 32, 32)` padded lanes, since every core's width is sub-tile-aligned.
  The reader prepares a per-core-`valid_cols` partial scaler; the SUM-reduce over the
  padded tiles yields the slice's partial Σx², and the cross-core sum stays correct
  (sum is associative — tile alignment is irrelevant to the total).
- Untilize `cb_out` back to the resident RM shard (arbitrary width, `write_sticks`-style).
- Reuse the existing xcore cross-core combine + writer transport unchanged.

**Done when**: `{layout: ROW_MAJOR, memory_layout: *_SHARDED}` EXCLUSIONS removed with
their cells passing (~1260 xfail cells); golden green; prior phases unregressed.

---

### [~] Refinement 5 — HEIGHT_SHARDED (local per-core reduction)

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

**Landed (partial `[~]`)**: `HEIGHT_SHARDED` in SUPPORTED. **TILE input** (TILE gamma,
RM gamma, no gamma) lands via a native zero-copy resident-shard path — a knob-turn on
the interleaved row-parallel R3-resident indexed two-pass, REUSING `rms_norm_reader.cpp`
+ `rms_norm_compute.cpp` unchanged except two `if constexpr (X_RESIDENT)` branches (the
interleaved perf path stays byte-identical at `X_RESIDENT=0`):
- `_create_height_sharded_descriptor`: core assignment pinned by the shard grid (each
  core's resident shard = its per-core block, `num_rows = per_h_tiles`); `cb_x_in`/`cb_out`
  backed zero-copy via `ttnn.cb_descriptor_from_sharded_tensor` (NO NoC read/write, verified
  native — no accessor read of the own shard); **no writer kernel** (compute packs `cb_out`
  in place).
- Reader `X_RESIDENT`: skips the x read (x resident); streams gamma per block per row
  (TILE tiles, or RM sticks → `cb_gamma_sticks`).
- Compute `X_RESIDENT`: self-arms the resident `cb_x_in` (whole shard), per-row wait/pop
  walks it (block offset indexing identical to R3); gamma STREAMED per block (small
  `cb_gamma` fits any W — a full-W resident gamma would blow L1 on top of the resident
  input+output shards for wide W; resident gamma is the R6 perf lamp); RM gamma tilizes
  per block before the `·gamma` mul.
Golden HEIGHT slice: **852 passed / 630 xfailed / 0 failed / 0 xpassed** (RM-input cells
xfail via the one remaining EXCLUSION). Loose HEIGHT `(1,1,256,512)` → `supported_pass`
(loose 19/19). Probes + `test_rms_norm_height_sharded.py` (22 cases) PCC ≥ 0.99967 across
aligned/W-/H-/both-non-aligned, `per_h>1` (R>grid), wide W=8192 (L1-pressure, single core),
fp32, bf8b, 2D/3D/4D, mixed precision — `--dev` + non-dev. Unit dir 381 passed / 32 skipped;
interleaved + WIDTH/BLOCK unregressed.

**Deferred to R5a** (structural, characterized at depth): **RM INPUT + HEIGHT_SHARDED**.
The resident shard is full-W row-major sticks (not tiles); consuming it needs a
tilize-on-resident-shard (loopback repack of the resident RM sticks into tile-padded
`cb_x_sticks` → `tilize` → `cb_x_in`) and an untilize-back into the resident RM output
shard (loopback write) — a local-reduction analog of the R4b RM sub-scheme, requiring a
writer kernel again. The one `{layout: ROW_MAJOR, memory_layout: HEIGHT_SHARDED}` EXCLUSION
stays xfail-strict (not silenced; it is R5a's baseline). RM+HEIGHT+TILE-gamma is INVALID.

---

### [x] Refinement 5a — RM-input HEIGHT_SHARDED (tilize-on-resident-shard)

**Goal**: complete R5's deferred corner — **RM input + HEIGHT_SHARDED** — where a core's
resident row-shard is full-W row-major sticks, not tiles. Remove the one
`{layout: ROW_MAJOR, memory_layout: HEIGHT_SHARDED}` EXCLUSION with its cells passing
(~630 golden xfail cells; RM+HEIGHT+TILE-gamma stays INVALID).

**Landed (full `[x]`)**: RM input + HEIGHT_SHARDED lands by extending the interleaved
`_create_height_sharded_descriptor` + the shared `rms_norm_{reader,compute,writer}.cpp`
via CT-arg flags (no forked files); the EXCLUSION is removed. Each core holds FULL-W
rows so the reduce stays LOCAL (no cross-core combine, phase=0, only the W%32 mask). The
sub-scheme:
- **Reader** (`IS_RM && X_RESIDENT`): loopback-repacks the resident RM row-shard sticks
  (`cb_shard_in` = `cb_descriptor_from_sharded_tensor` alias → local NoC loopback via
  `my_x/my_y`, **no DRAM/remote read** — native local consumption) into tile-padded
  `cb_x_sticks`, reading only `origin_W` valid columns per stick, up-front-zeroing the
  W%32 pad. Mirrors the interleaved streaming RM reader order EXACTLY (2 passes; gamma
  interleaved per pass-2 block) so the compute is consumed unchanged. gamma = RM sticks
  (TILE gamma INVALID with RM input) or none.
- **Compute**: REUSES the existing streaming RM path UNCHANGED (`use_resident=0`): tilize
  `cb_x_sticks`→`cb_x_in` per block per pass, two-pass reduce/normalize, untilize
  `cb_out`→`cb_out_sticks` per block. Streaming keeps every CB bounded
  (`cb_x_in = DEPTH*BLOCK_SIZE`, never Wt) so it fits any W/dtype.
- **Writer** (NEW branch, `IS_RM && X_RESIDENT`): loopback-writes the valid columns of
  `cb_out_sticks` into the resident RM output shard (`cb_shard_out` alias). H
  non-alignment via per-core `valid_rows_total` (the last core is short); pad rows/cols
  are tensor-padding, not written.
- **Why streaming, not the resident single-tilize the note sketched**: a whole-tile-row
  resident `cb_x_in` (Wt fp32 tiles = 1 MB at W=8192) + intermediates + shards OOMs L1
  (golden CB-clash on fp32 W=8192, a *feasible* cell since RM shards are small/per-row).
  The streaming re-tilize (2× local loopback, no DRAM) fits every cell; the single-tilize
  fast-path is folded into the R6 sharded-perf pass.

Probes 13/13 PCC ≥ 0.99996 (no-gamma/RM-gamma × aligned/W-/H-/both-non-aligned ×
4D-last-core-short/3D/2D × fp32 × wide W=8192, incl. the prior-OOM fp32 W=8192).
Golden HEIGHT slice **1168 passed / 0 failed / 0 xpassed / 315 xfailed** (RM-input cells
xfail→pass; the 315 xfails are the standing `{f32, acc=False}` EXCLUSION). Loose 19/19.
Unit dir 401 passed / 32 skipped (`--dev` + non-dev). Interleaved + WIDTH/BLOCK + TILE
HEIGHT unregressed.

**Verifier notes**: local-reduction analog of the R4b RM sub-scheme, but SIMPLER — each
core holds FULL-W rows, so there is no cross-core combine, no phase-alignment, and no
per-core partial scaler beyond the standard `W % 32` mask (the shard width IS the full W).
The exact levers:
- **Reader**: loopback-repack the resident RM shard sticks (`cb_descriptor_from_sharded_tensor`
  alias → local NoC loopback, no remote re-fetch) into tile-padded `cb_x_sticks`, reading
  only `origin_W` valid columns per stick; keep the up-front `cb_x_sticks` zeroing for the
  `W % 32` pad. gamma unchanged from R5 (streamed per block, TILE or RM sticks).
- **Compute**: `tilize<BLOCK_SIZE, cb_x_sticks, cb_x_in>` per block per tile-row before the
  existing R3-resident indexed two-pass (x now lives in the allocated `cb_x_in`, not a
  zero-copy alias); after pass 2, `untilize<BLOCK_SIZE, cb_out, cb_out_sticks>` per block.
  This means x is NO LONGER zero-copy on the RM path (it is tilized into an allocated
  `cb_x_in`) — the resident shard is still consumed locally (loopback, no DRAM/remote read),
  just re-paged for the tile engine.
- **Writer**: NEW (R5's TILE path has none) — loopback-write the untilized `cb_out_sticks`
  valid columns into the resident RM output shard (`cb_descriptor_from_sharded_tensor` alias).
- H non-alignment: the last tile-row's padding rows are transparent (discarded on read-back),
  same as the TILE path; the RM writer writes exactly the valid rows per the shard height.

**Done when**: `{layout: ROW_MAJOR, memory_layout: HEIGHT_SHARDED}` EXCLUSION removed with
its cells passing; golden green; all prior phases unregressed.

---

### [~] Refinement 6 — Speed up the decode + sharded perf profiles

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

**Landed (partial `[~]`)** — one collective-topology lever (NoC-mcast broadcast) shipped;
the decode half of the goal was ALREADY MET by R4a. Measured on blackhole_p150b (110-core
grid, median of 8 fresh trials, exact perf configs; `test_rms_norm_perf_r6.py`):
- **Decode interleaved already BEATS achievable_ns** — `(1,1,32,W)` for W∈{1024,2304,5120,7168}
  measure 8.7 / 14.8 / 16.8 / 16.9 µs vs achievable 9.1 / 17.0 / 75.8 / 104.3 µs
  (0.95× / 0.87× / **0.22×** / **0.16×**). R4a's logical W-split delivered the decode target;
  no R6 change needed (and none applied — decode groups are ragged, so they keep the
  byte-identical all-unicast path). The "speed up the decode" half is satisfied.
- **mcast broadcast lever (the named collective-topology family)**: the group master now
  broadcasts the finalized `1/RMS` with ONE `noc_async_write_multicast` + ONE
  `noc_semaphore_set_multicast` instead of K-1 serial unicast writes + K-1 sem-incs — a
  K-independent broadcast (the R4 writer comment's flagged R6 lever). Gated host-side on a
  GAP-FREE virtual rectangle (`group_rect`: bounding-box area == group size); ragged groups
  (logical decode; WIDTH auto-shard wrapping a partial grid row) keep the proven all-unicast
  fallback (3-monotone-counter protocol + CB back-pressure UNCHANGED — only the broadcast
  leg's transport mechanism swaps). Correct (--dev clean; golden green; soft PCC ≥ 0.9995).
  Wins **1.15×** on the one gap-free perf geometry (WIDTH 7×4 / K=28: 11.20→9.73 µs).
- **Why the other 4 sharded targets did not move — characterized at depth (ablation)**:
  1. **Blackhole DRAM-column gap.** The virtual-coord map skips x=8,9 (logical x=0..10 →
     virtual x=[1..7,10,11,12,13]). So any group spanning logical x=0..7 (the 8-wide WIDTH
     8×1/9×1/8×4 and BLOCK 8×8 targets) is NOT a gap-free virtual rectangle → the strict
     rect check correctly keeps them on unicast (a naive mcast to the [1..10] box would hit
     the DRAM columns → NoC fault). Only 7×4 (logical x=0..6, gap-free) qualifies.
  2. **The broadcast is only ~14% of a round.** Ablation (fixed per_w_t=1, gap-free 7-wide,
     mcast engaged): the per-tile-row synchronous gather→fold→broadcast round costs a FLAT
     **~3150 ns**, fully serialized × HT_LOCAL (K7·HT4→13.7µs, HT16→51.0µs, HT32→100.8µs,
     perfectly linear). This dominates BLOCK's 5.76× headroom. A residual **~92 ns/core**
     gather fan-in K-cost remains even with mcast (K7→4.3µs, K28→6.2µs).
- **No regression**: unit dir 165 correctness + RM/HEIGHT/RM-HEIGHT sharded pass; golden
  `test_op_loose` 19/19; golden `test_op` WIDTH/BLOCK cartesian slice 78 passed / 0 failed /
  0 xpassed / 18 xfailed (the standing `{f32,acc=False}` EXCLUSION); R3 prefill perf 4/4.
  The mcast lever is byte-identical on every ragged/gap group, so nothing regressed.

**Deferred to R6a** (the real sharded-latency levers, characterized above, in priority order):
the per-tile-row synchronous-round cost (`~3150 ns × HT_LOCAL`) is the dominant sharded
bottleneck and needs a round-granularity restructure, not the broadcast mechanism.

---

### [~] Refinement 6a — Sharded cross-core: batch the per-tile-row round + gap-aware mcast

**Type**: perf

**Goal**: close the sharded `_perf_case` headroom R6 characterized but could not reach with
the broadcast-only lever. Measured baselines (blackhole_p150b): WIDTH 8×1 1.41×, 9×1 1.71×,
8×4 1.95×, 7×4 1.78×, **BLOCK 8×8 5.76×** above achievable. Three levers, in priority order:

1. **Batch C tile-rows' stats per cross-core round (biggest — BLOCK).** The cross-core round
   is FULLY SERIALIZED per tile-row at a flat ~3150 ns (ablation: BLOCK is `HT_LOCAL × 3150`).
   Restructure compute+writer+host so one round exchanges C tile-rows' partials: compute
   produces C local partials (`cb_stat_local` depth C), the writer gathers `K×C` (bounded —
   `C` is a tunable, keep `cb_gather ≤ K×C` under an L1 gate; C=all OOMs the master at K×HT),
   the master folds C → C rstds, broadcasts C, then pass-2 over the C tile-rows. This cuts
   BLOCK's sync rounds from HT_LOCAL to `ceil(HT_LOCAL/C)` — the 5.76× target. This is a
   scheme change to the round granularity (the R4 "one round per tile-row → cb_gather stays
   K" invariant is deliberately relaxed to `K×C` under an explicit L1 budget, same exception
   class as R3's resident dual-path).
2. **Gap-aware mcast (unblocks the 8-wide WIDTH/BLOCK targets).** Extend the R6 broadcast
   mcast to groups whose virtual coords straddle the Blackhole DRAM-column gap (x=8,9): either
   segment the mcast into contiguous virtual runs ([1..7] mcast + [10] unicast) or build the
   mcast grid via the device's mcast-coordinate utility that excludes DRAM columns. Removes
   the `group_rect` strict-rectangle restriction for 8×1/9×1/8×4/8×8.
3. **Two-stage gather (residual K-cost).** The gather fan-in still scales ~92 ns/core even with
   mcast broadcast (`tensix_all_reduce` grid-two-stage: reduce along x → row-leaders, then y →
   root). Smallest lever; only meaningful for the large-K 2D WIDTH groups.

**Verifier notes**: R6 landed the broadcast mcast + full bottleneck ablation; R6a is the
round-batching restructure (lever 1) as the headline, then gap-aware mcast (lever 2). Keep the
R6 `group_rect`/mcast path — lever 2 extends it, does not replace it. Reuse the R4/R6 xcore
kernels + transport; do not fork. Gate on soft `pcc_threshold` 0.9995.

**Done when**: measured median device-ns improves on the WIDTH/BLOCK sharded perf shapes
(fresh-cache trial loop) toward achievable, soft PCC holding, golden green, no regression
across the guard set.

**Landed (partial `[~]`)**: the two headline levers (1 round-batching, 2 gap-aware mcast)
shipped on the shared `_assemble_xcore_kernels` transport via a `C_ROWS` CT arg + segmented
mcast (no forked files); both correct, `--dev`-clean, non-regressing. Measured
(blackhole_p150b, median of 8 fresh trials, exact perf config):
- **Lever 1 (round-batching)** — one round exchanges C tile-rows' partials (compute produces
  C, writer gathers `K*C`, master folds C, broadcasts C; sync rounds `HT_LOCAL`→`ceil(HT/C)`).
  C=`STAT_BATCH_ROWS`=8, L1-gated (`cb_gather`=`K*C`); C=1 byte-identical to R4; C>1 only on the
  pure tiled resident-shard path (RM/logical keep C=1). **BLOCK 8×8: 147729→119030 ns (5.76×→
  4.64×, 1.24×)** — the only multi-tile-row-per-group target.
- **Lever 2 (gap-aware mcast)** — the 1/RMS broadcast mcasts in up to 2 contiguous virtual-x
  runs (`[xlo..7]`+`[10..xhi]`) for groups straddling the Blackhole DRAM columns (x=8,9); ragged
  groups keep unicast. **WIDTH 8×4 (K=32): 10204→8920 ns (1.94×→1.69×, 1.14×; A/B 8884 mcast vs
  10173 unicast)**; WIDTH 7×4 confirms R6 mcast; WIDTH 8×1/9×1 flat (K=8/9 broadcast cheap either way).
- C-sweep + A/B ablations show the **BLOCK residual is per-tile-row stat data-movement** (K
  bloated 4KB fp32 stat tiles/tile-row, ~2.1 µs/tile-row) + the unpipelined compute floor (~47 µs),
  NOT the broadcast — lever 1 plateaus ~4.6× and lever 2's broadcast is only ~14% of a round.
Lever 3 (two-stage gather) not implemented. Deferred to R6b.

---

### [~] Refinement 6b — Sharded cross-core: stat-tile compaction + round/compute pipelining (+ two-stage gather)

**Type**: perf

**Goal**: close the BLOCK-8×8 residual R6a characterized (still 4.64× above achievable after
round-batching + gap-aware mcast) — the dominant cost is the **per-tile-row stat data-movement**,
not the round count or the broadcast. Levers in priority order:

1. **Stat-tile compaction (biggest — cuts gather+broadcast bandwidth ~32×).** The cross-core
   stat is a REDUCE_ROW result: a 32-value fp32 column (128 bytes), but it is gathered/broadcast
   as a full 32×32 fp32 tile (4 KB) — 32× bloat. The gather moves `K` such tiles per tile-row into
   the master (~2.1 µs/tile-row on BLOCK 8×8, the measured floor). Transfer only the meaningful
   column (partial-tile NoC read/write with the right stride, or repack the K partials into one
   tile before the fold) so the gather/broadcast bytes drop ~32×.
2. **Round/compute pipelining (hides the round under compute).** The batched loop is
   pass-1(C) → synchronous round → pass-2(C), fully serial; the master idles during the round's
   semaphore waits. Software-pipeline so batch b+1's pass-1 overlaps batch b's round (deeper
   `cb_stat_local`/`cb_gather` staging). On BLOCK 8×8 the compute floor (~47 µs) and round
   (~68 µs) are additive today (~115 µs) — overlapping would approach `max(...)`.
3. **Two-stage gather (residual K-cost, R6a lever 3 — smallest).** For the large-K 2D WIDTH
   groups (8×4 K=32), reduce the gather fan-in with a hierarchy (reduce along x → row-leaders,
   then y → root) instead of K-1 workers converging on one master core.

**Verifier notes**: R6a landed round-batching + gap-aware mcast and characterized the residual
via C-sweep + mcast A/B ablations. R6b is the stat-tile-compaction / pipelining restructure
(lever 1 headline, lever 2 next); reuse the R4/R6/R6a xcore kernels + segmented-mcast transport,
do not fork. Gate on soft `pcc_threshold` 0.9995.

**Done when**: measured median device-ns improves further on the BLOCK/WIDTH sharded perf shapes
(fresh-cache trial loop) toward achievable, soft PCC holding, golden green, no regression.

**Landed (partial `[~]`)**: the two headline levers (1 stat compaction, 2 round/compute pipelining)
shipped on the shared `_assemble_xcore_kernels` transport via CT flags — no forked files; both
numerically byte-identical (compaction preserves the only consumed data; pipelining reorders the
same ops), `--dev`-clean, non-regressing. Measured (blackhole_p150b, median of 8 fresh trials, exact
perf config bf16 / fp32_dest_acc_en=False / TILE / TILE gamma / HiFi2):
- **Lever 1 — stat-tile compaction (the winner).** The cross-core stat is a REDUCE_ROW result whose
  only consumed data is COLUMN 0, which in an fp32 tile lives entirely in faces 0 (rows 0-15) + 2
  (rows 16-31). The GATHER (K partials converging on the master, ~86% of the round per the R6a
  ablation) now moves ONLY those faces (`G_OFF/G_LEN` writer CT args + `_gather_runs`/`STAT_COMPACT_MODE`);
  the untransferred faces leave stale L1 the fold sums-then-ignores → **numerically byte-identical**
  (PCC 1.001005 == baseline). The literal "32× column-only" transfer is infeasible (col 0 is strided
  across faces at 64 B stride; the NoC rewards contiguous runs and every in-tree precedent —
  `tensix_all_reduce`, `combine_welford` — moves whole tiles), so the achievable compaction is
  the col-0 FACES. **Mode ablation (BLOCK 8x8): full 4 KB = 119036, faces 0-2 3 KB/1 txn = 116219,
  faces 0&2 2 KB/2 txn = 108028** — the core-to-core L1 gather is bandwidth-dominated (unlike the DRAM
  `tile_reorder` regime), so mode 2 (skip the unused middle face) wins despite the extra transaction.
  Gated to the pure tiled sharded path (WIDTH/BLOCK physical shards); RM / logical / decode keep the
  full transfer (mode 0, byte-identical). The broadcast leg (~14%, one mcast of C contiguous tiles)
  stays full — per-tile mcast-splitting would add more overhead than the byte saving (R6a ablation
  shows it is off the critical path).
- **Lever 2 — round/compute pipelining (the COMPLEMENTARY step to lever 1).** Compute issues batch
  r+1's pass-1 one round ahead so the local reduce overlaps the writer's synchronous round
  (`PIPELINE_LOOKAHEAD` CT flag; `cb_stat_local` already 2*C deep; writer/semaphore protocol +
  fixed-base addressing UNCHANGED — only the compute issue order changes). **Flat on its own**
  (mode0 pipe-on 119036 == baseline 119027 — with the full gather the round dwarfs the compute), but
  once lever 1 shrinks the gather it **wins**: BLOCK 8x8 113810 (pipe-off) → 107995 (pipe-on), 1.05x
  on top of compaction. Shipped ON on the multi-round tiled path; single-round WIDTH groups
  (num_rounds==1) degenerate byte-identically.
- **Measured speedups** (R6a → R6b, all sharded WIDTH/BLOCK perf shapes; soft PCC ≥ 0.99998):
  BLOCK 8x8 119027 → **107995 (1.10x)** (4.64x → 4.21x above achievable); WIDTH 8x4 8933 → 8132 (1.10x);
  WIDTH 7x4 9850 → 9182 (1.07x); WIDTH 8x1 5836 → 5619 (1.04x); WIDTH 9x1 7883 → 7691 (1.02x). Decode
  interleaved (logical W-split, gated off) unaffected/byte-identical (8691 ns).
- **No regression**: golden `test_op_loose` 19/19; `test_op` cartesian slice (1x1x2048x256, multi-round
  BLOCK) 165 passed / 0 failed / 0 xpassed / 39 xfailed (the standing `{f32,acc=False}` EXCLUSION);
  unit dir 431 passed / 32 skipped (`--dev` + non-dev); R6a batched-round correctness 14/14 (`--dev`
  + non-dev). Both levers gate to the tiled path, so every non-tiled sharded/interleaved cell is
  byte-identical.

**Deferred to R6c** (lever 3, characterized): **two-stage gather.** Even after compaction+pipelining,
BLOCK 8x8 is still 4.21x above achievable — the residual is the master-serialized fold+gather
(the K-1 workers still converge on one master core, and the master's fold sits ON the round's
critical path, which is exactly why lever 2 (pipelining) only recovers the pass-1 overlap and not
the fold). Lever 3 restructures the gather into a hierarchy (reduce along x → row-leaders, then y →
root) so the fold is distributed off the master's critical path — the smallest R6b lever, most
relevant to the large-K 2D WIDTH groups (8x4 K=32) and the missing complementary step that would let
pipelining also hide the fold. Reuse the R4/R6/R6a xcore kernels + segmented-mcast transport; do not
fork. Gate on soft `pcc_threshold` 0.9995.

---

### [ ] Refinement 6c — Sharded cross-core: two-stage (hierarchical) gather

**Type**: perf

**Goal**: close more of the BLOCK/large-K WIDTH sharded residual R6b left (BLOCK 8x8 still 4.21x above
achievable after stat-tile compaction + round/compute pipelining). R6b's ablation showed the dominant
remaining cost is the **master-serialized gather + fold**: all K-1 workers unicast their partials to one
master core (fan-in latency ~92 ns/core, and the 4 KB→2 KB compaction only halved the bytes, not the
transaction count), and the master's fold sits ON the round's critical path — which is precisely why
lever 2 (pipelining) recovered only the pass-1 overlap, not the fold. One lever:

1. **Two-stage (hierarchical) gather.** Replace the flat K-1→master gather with a 2-stage reduce over
   the group rectangle: stage 1 reduces along x (each grid-row's cores → the row-leader, folding
   locally), stage 2 reduces the row-leaders along y → the root, which finalizes (+eps, rsqrt) and
   broadcasts (the existing segmented mcast). This cuts the fan-in from K-1 to (nx-1)+(ny-1) converging
   transfers and distributes the fold off the single master — the `tensix_all_reduce` grid-two-stage
   pattern. Most relevant to the large-K 2D WIDTH groups (8x4 K=32) and BLOCK's per-row groups; it is
   also the complementary step that would let R6b's pipelining hide the (now-distributed) fold.

**Verifier notes**: R6b landed compaction (winner) + pipelining (its complementary step) and characterized
the residual as the master-serialized fold+gather via mode/pipeline ablations. R6c is the gather-topology
restructure (lever 3). Reuse the R4/R6/R6a/R6b xcore kernels + segmented-mcast transport + the 3-monotone-
counter protocol; do not fork. This is a T3 collective-topology change (two semaphore stages), so verify
`--dev`-clean across HT_LOCAL and ragged groups. Gate on soft `pcc_threshold` 0.9995.

**Done when**: measured median device-ns improves further on the BLOCK/large-K WIDTH sharded perf shapes
(fresh-cache trial loop) toward achievable, soft PCC holding, golden green, no regression.
