# Operation Requirements: rms_norm

## Definition

- **Formula**: `output[..., r, c] = input[..., r, c] * rsqrt( (1/W) * Σ_{c'=0}^{W-1} input[..., r, c']² + epsilon ) * gamma[c]`
  where `W` is the **true, unpadded** last-dimension extent — tile padding never enters the denominator.
- **PyTorch Reference**:

  ```python
  def torch_rms_norm(x, gamma=None, epsilon=1e-6):
      x32 = x.to(torch.float32)
      out = x32 * torch.rsqrt(x32.pow(2).mean(dim=-1, keepdim=True) + epsilon)
      return out if gamma is None else out * gamma.to(torch.float32)
  ```

- **Import Path**: `from ttnn.operations.rms_norm import rms_norm`
- **Function Signature**:

  ```python
  def rms_norm(
      input_tensor: ttnn.Tensor,
      *,
      gamma: Optional[ttnn.Tensor] = None,
      epsilon: float = 1e-6,
      compute_kernel_config: ttnn.ComputeConfigDescriptor = None,
      memory_config: ttnn.MemoryConfig = None,
      _levers: dict = None,          # INTERNAL perf-bench hook; never read by validate()
  ) -> ttnn.Tensor
  ```

## Phases

> **Non-regression rule**: Every refinement must pass all tests from prior phases — and must not regress **performance**. Check the refinement's `changelog.md` perf table against the cumulative bench set recorded by prior phases: no prior bench shape's device kernel duration may regress beyond the measurement noise margin. A refinement that speeds up its own target while slowing a shape a prior phase already measured fast is a regression to resolve, not a trade-off to accept silently. If the refinement changed a shape-dependent code path (work distribution, blocking, CB geometry, dtype branch) but the changelog benched only a single point on that axis, flag the missing coverage — a non-regression gate that never measured the other regime cannot have cleared it. This rule is generic: it applies to every op and every optimization and prescribes no particular technique for avoiding the regression.
> **Drift signal**: XPASS-strict failures mean the implementer added support but forgot to update SUPPORTED. The implementer fixes by updating SUPPORTED.
> **Checkbox protocol**: Implementer marks `[x]` when the refinement is complete and all tests pass, `[~]` when real work landed but at least one named axis value is deferred (treated as completed by the queue, surfaced as partial), `[ ]` only when nothing usable was produced.
> **Refinement ID + follow-up naming (mandatory — the runner parses this)**: Primary refinements are `Refinement N` (e.g. `Refinement 1`, `Refinement 2`). When you ship `[~]` partial and file the sharper follow-up the partial-tick protocol requires, name it by appending a lowercase letter to the parent's number: `Refinement 1b`, `Refinement 1c`, … (never `Refinement 1.5`, `Refinement 1 (follow-up)`, or a fresh number). Order follow-ups immediately after their parent so the queue runs them before later refinements — a partial's remaining-blocker follow-up must be picked next, not leapfrogged. The runner's parser matches exactly `Refinement \d+[a-z]?`; any other shape is invisible to the queue and silently skipped.

### The cumulative bench set (every phase re-measures ALL of these)

`RMS_BENCH_MODE=baseline python -m tracy -r -m pytest tests/ttnn/unit_tests/operations/rms_norm/test_rms_norm_bench.py -s`,
then `_bench_rms_norm.report_from_csv(<csv>)`. Device kernel duration, blackhole p150b:

| name | shape | layout | Phase 0 ns (verifier re-measure) |
|---|---|---|---|
| `grid_filling` | (1,1,8192,1024) | TILE | 93 656 |
| `wide_prefill` | (1,1,8192,7168) | TILE | 1 019 959 |
| `grid_starved` | (1,1,32,7168) | TILE | 76 161 |
| `smallest` | (32,17) | TILE | 3 221 |
| `row_major` | (1,1,8192,1024) | ROW_MAJOR | 95 881 |

This is also the **config-spanning guard set** the perf refinements must not regress: it covers both
regimes (A on `grid_filling`/`row_major`, B on `wide_prefill`/`grid_starved`/`smallest`), both
layouts, the full-grid and the single-core core counts.

### [x] Phase 0 — Core Implementation

- **SUPPORTED dtype**: [float32, bfloat16]
- **SUPPORTED fp32_dest_acc_en**: [True]
- **SUPPORTED layout**: [TILE, ROW_MAJOR] — both native, no host-side `to_layout`/`tilize`/`untilize`
- **SUPPORTED shape-derived axes**: alignment ∈ {tile_aligned, w_non_aligned, h_non_aligned}; rank ∈ {2,3,4}
- **SUPPORTED op-specific axes**: gamma_mode ∈ {gamma, no_gamma}; gamma_dtype ∈ {float32, bfloat16, "none"}; gamma_layout ∈ {TILE, ROW_MAJOR, "none"}
- **SUPPORTED memory_layout**: [INTERLEAVED]
- **EXCLUSIONS**: `{dtype: float32, fp32_dest_acc_en: False}`
- **Cores**: multi-core — `Rt` (the independent axis) split over the whole grid, `row_wise=True`; 128 of 130 cores on `(1,1,8192,1024)`
- **Compute config**: caller's descriptor passed through verbatim; default = HiFi4 + `fp32_dest_acc_en=True` + `math_approx_mode=False`
- **Golden baseline**: **737 / 737** supported cells passing (per `verifier_report.json`); 6 172 xfail_expected, 33 900 invalid_skipped, 0 in every loud category

---

### [ ] Refinement 1 — Numerical configurability expansion

**Goal**: add `ttnn.bfloat8_b` to `SUPPORTED["dtype"]` and to `SUPPORTED["gamma_dtype"]`, and add
`False` to `SUPPORTED["fp32_dest_acc_en"]` for `bfloat16` activations. Expose the full
`ttnn.ComputeConfigDescriptor` surface on the entry point (it is already a pass-through; what changes
is which corners `validate()` accepts) and correct intermediate-CB precision to match — including
`UnpackToDestFp32` tagging where it applies, and the bf8b block-format CBs. The existing
`{dtype: float32, fp32_dest_acc_en: False}` EXCLUSION stays (fp32 activations with a bf16 DEST
accumulator remains a native refusal); any bf8b corner that fails out of the box joins it in
`EXCLUSIONS` rather than becoming its own refinement.

**Implementation skill**: /numeric-formats-metal

**Verifier notes**: **this must land first — it is the gate on every later perf phase.** Every perf
loose case in `feature_spec.LOOSE_CASES`, including the mandatory `(1,1,32,7168)`
`minimum_expected_speedup=7.0` gate, runs at `bf16 / HiFi2 / fp32_dest_acc_en=False`, which is xfail
today; until this lands a perf pass can only measure a `fp32_dest_acc_en=True` stand-in, which is a
different datapath and a different DEST capacity, and says nothing about the target.
Three consequences to carry into the work:
(a) `fp32_dest_acc_en=False` **doubles DEST** (4 → 8 tiles at half-sync), so `_dest_limit()` already
returns 8 on that path and `DEST_BLOCK` / `BLOCK_HT` / `IN_BUF_DEPTH` must be re-swept —
`levers=dict(dest_block=N, block_ht=N)` makes that a measurement, not a rewrite;
(b) the CB set is now emitted by the single `_cb_layout()` in `rms_norm_program_descriptor.py`, so a
new dtype's page size lands in exactly one place — do not re-derive page counts anywhere else;
(c) `feature_spec.INVALID` already parks `{bfloat8_b × w_non_aligned}` and `{bfloat8_b ×
h_non_aligned}`, so bf8b only has to work on tile-aligned shapes here. Build this refinement to the
performance-conformance bar (full grid, batched dataflow both directions) — Refinement 3 optimizes
the very path it creates.

**Done when**: `SUPPORTED["dtype"]` and `SUPPORTED["gamma_dtype"]` contain `bfloat8_b`,
`SUPPORTED["fp32_dest_acc_en"] == [True, False]`, the golden suite is green with 0 in every loud
category, and the `(1,1,32,7168)` interleaved perf loose case reaches the op (i.e. it is a
`supported_pass`/measurable cell, not an xfail) at its exact config: bf16 / TILE / INTERLEAVED /
HiFi2 / `fp32_dest_acc_en=False`.

---

### [ ] Refinement 2 — Sharding-native `memory_layout` (local shard + cross-core combine)

**Goal**: add all three sharded values to `SUPPORTED["memory_layout"]` — `HEIGHT_SHARDED`,
`WIDTH_SHARDED`, `BLOCK_SHARDED` — natively, for both `TILE` and `ROW_MAJOR` inputs, with the output
inheriting the input's shard spec. Two structurally different halves:

- **HEIGHT_SHARDED (knob-turn, design Lamp L3).** The shard cuts the *independent* `Rt` axis, so each
  core already holds whole rows and the reduction stays entirely local. This is a **CB-placement**
  change: back `cb_input_tiles` (and the output CB) on the shard via
  `ttnn.cb_descriptor_from_sharded_tensor(CB_INPUT_TILES, input_tensor)` — zero-copy, **no NoC read
  at all** — and drive `CORE_GRID` plus each core's row count off the shard spec inside
  `blocking_plan()`, with `BLOCK_HT` defaulting to the **whole resident shard height** (sub-chunk only
  if the shard exceeds the working-set budget).
- **WIDTH_SHARDED / BLOCK_SHARDED (scheme-change, design Lamp L1).** The shard cuts the *dependent*
  `Wt` axis, so partial sums must be combined across cores. Required topology, per
  `op_design.md` → "Unlocked scheme L1": each core reduces its `Wt_core` slice into
  `cb_partial_sumsq`; the group's cores `noc_async_write` their partials into distinct slots of the
  sender's `cb_partial_gather` and `noc_semaphore_inc` a gather semaphore; the sender waits for
  `group_size − 1`, sums the partials in DEST, then broadcasts the result back with
  `mcast_pipe.hpp`'s `SenderPipe::send()` / `ReceiverPipe::receive()` (`PRE_HANDSHAKE = true`), host
  side wired by `Mcast1D(device, group_grid, Mcast1DShape::PerRow, 0, McastConfig{...})`. For
  `BLOCK_SHARDED` the combine group is one **grid row** (the `PerRow` family only — a 2-D split needs
  two 1-D mcast families, never one 2-D mcast). The last core's short slice is masked by the same
  `ReducePartialScaler::last_tile_at(1)` mechanism Regime B already uses.

**Verifier notes**: no skill in the inventory covers `memory_layout`/placement — do **not** reach for
`/memory-layouts` (that is RM↔TILE layout, which this op already has natively). The required
mechanisms are named above; the design's L1 contract table is the spec.
Ordering and scope:
- This is the hardest refinement in the queue and it is deliberately second, because Refinement 1 is a
  hard dependency (the sharded perf loose cases and the whole resilience sweep run at
  `fp32_dest_acc_en=False`).
- **Do not fake the local shard.** Re-reading a core's own resident shard through a `TensorAccessor`
  is *not* sharding — it merely tolerates the layout because each core still holds its full rows. If
  golden cells go green that way the axis is not implemented; the zero-copy CB placement is the
  deliverable (design risk R11).
- The combine machinery you build here is what Refinement 3 tunes on the *interleaved* decode shapes —
  keep the group size, the gather topology and the mcast depth as **parameters off `blocking_plan()`**,
  not constants, or Refinement 3 has nothing to turn.
- If the cross-core half proves larger than one focused pass, ship `[~]` with `HEIGHT_SHARDED` landed
  and file `Refinement 2b` for `WIDTH_SHARDED`/`BLOCK_SHARDED` — but do **not** ship the reverse (a
  `TensorAccessor` stand-in for the local shard) as `[x]`.
- Expect the `_RESILIENCE_SHAPES` sweep (prime/awkward tile counts × 4 placements × 2 layouts) to be
  the real test surface; `eval.sharding.auto_shard_config` supplies the specs, so there is no
  golden-side work.

**Done when**: `SUPPORTED["memory_layout"] == [INTERLEAVED, HEIGHT_SHARDED, WIDTH_SHARDED,
BLOCK_SHARDED]`, the ~4 800 sharded `xfail_expected` cells become `supported_pass` (minus anything
that lands, with a written reason, in `EXCLUSIONS`), the local-shard path performs **zero** NoC reads
of its own shard, the golden suite is green with 0 in every loud category, and the cumulative bench
set shows no regression.

---

### [ ] Refinement 3 — Speed up the perf-flagged grid-starved decode profile

**Type**: perf

**Goal**: `feature_spec.LOOSE_CASES` carries the only case in the file with a hard speedup
requirement: `(1, 1, 32, 7168)` **interleaved**, bf16 / TILE / HiFi2 / `fp32_dest_acc_en=False`,
`achievable_ns = 104 259` on blackhole_p150b @1350 MHz with `minimum_expected_speedup = 7.0` — i.e. a
goal of **≤ 14 894 ns** (clock-scaled). Today that shape measures **76 161 ns on one core**: `Rt = 1`,
so the independent-axis split reaches exactly one core, and the implementer's stub ablation puts
**≥ 58.6 % of that wall on compute**, not on bytes. Optimize **this exact config**.

The lever is the `distribution_gate` pattern from `ttnn/ttnn/operations/examples/master.md` Part 1:
keep the `Rt` split as the default and **divert to a cross-core `Wt` split only behind a
utilization predicate** (e.g. "the `Rt` split fills ≤ 1/K of the grid"), reusing the combine machinery
Refinement 2 builds — `width_split` measures up to 7.76× for exactly this wide-and-short geometry, and
`distribution_gate` measured the gated form **byte-identical** on the shapes the default already
saturated. Secondary levers on the same shape, in catalog order: `tensix_all_reduce` (tree /
reduce-scatter gather beats a flat gather 4.64–6.48×, and at 1 tile/core tree reduce is the one to
try first), then the block-size / buffer-depth co-tune (`compute_block_size`; `DEST_BLOCK` doubles
under `fp32_dest_acc_en=False`, so the Phase 0 sweep is stale) and `sfpu_tile_scope` (the rms chain's
`rsqrt` runs a full 32×32 tile of which only Col0 is meaningful — up to 7.26× on the SFPU call in
isolation, and this is the one regime compute-bound enough for it to surface). No SUPPORTED change.

**Done when**: measured device-ns on `(1,1,32,7168)` interleaved **at its exact loose-case config**
(bf16 / TILE / HiFi2 / `fp32_dest_acc_en=False`) improves decisively toward the 14 894 ns goal, its
soft `pcc_threshold = 0.9995` gate still holds, the golden suite is green with 0 in every loud
category, and the cumulative bench set above shows no regression — in particular `grid_filling` and
`wide_prefill` must be untouched, which is exactly what gating the diversion buys.

---

### [ ] Refinement 4 — Speed up wide-`W` prefill by widening the single-read regime

**Type**: perf

**Goal**: `wide_prefill` `(1,1,8192,7168)` measures **1 019 959 ns**, which is *at* the data-movement
bound — but for the **2-pass** algorithm: Regime A does not fit L1 at `Wt = 224`, so the shape falls
back to Regime B and moves **1.5× the DRAM bytes** (2 reads + 1 write instead of 1 + 1). The lever is
the catalog's `compute_fusion` shape applied as the design's Lamp L4: pre-broadcast `gamma` to full
32-row tiles once at boot and fuse the two pass-B multiplies (`x·(1/rms)` then `·gamma`) into a single
`eltwise_chain` via `DestReuseBinary` (`chain.hpp:518-520`), which **eliminates `cb_normed`** — a
`BLOCK_HT × Wt_core` L1 write+read round-trip per row-block — and, because `cb_normed` is one of the
three `Wt_core`-scaled CBs in the budget, widens Regime A's reach to about `Wt = 224` so this shape
lands on the single-read path. The design predicts **672 µs vs 1 032 µs ≈ 1.5×**; the perf-flagged
`_perf_case(8192, 7168, 1032281)` loose case is the reference. Co-tune the freed L1 into
`IN_BUF_DEPTH` / `BLOCK_HT` (`double_buffer` C16 + `compute_block_size`) once the regime flips — C16
measured flat at Phase 0 precisely because the shape was already DRAM-saturated, so re-measure it
here rather than assuming either way. No SUPPORTED change.

**Done when**: `(1,1,8192,7168)` selects Regime A and its measured device-ns improves materially
against the 1 019 959 ns baseline, precision is unchanged (this shape is `w`-aligned so no mask is
involved; PCC must not move), the golden suite is green with 0 in every loud category, and the
cumulative bench set shows no regression — especially `grid_starved` and `smallest`, which share the
Regime-B code path this refinement stops using for wide prefill.

---

### [ ] Refinement 5 — Perf completeness audit

**Type**: perf

**Goal**: rms_norm is a perf-focused op (its feature spec ships eight config-matched `achievable_ns`
references and a hard 7× gate), so close the run with a `/perf-ceiling-dm` **Mode D** completeness
audit. Walk the full lever list — `ttnn/ttnn/operations/examples/master.md` Part 1 **and** Part 2 —
and account for **every** lever this op did not apply, classified not-applicable / deferred /
measured-no-payoff / missed, with an estimated counterfactual delta for the promising unused ones.
Fold it into the existing `lever_ledger.json` (already 12/29 closed with evidence,
`python3 -m eval.verify_levers <ledger> --report`) and record a completeness ledger plus a ranked list
of remaining opportunities in `changelog.md`. Known open items to resolve rather than re-list:
**B8 `split_reader` / trid double-issue** was never measurable at Phase 0 (`grid_filling` gives each
core one row-block — nothing to overlap) and needs `wide_prefill` (2 blocks/core) or `(99991,64)`
(28 blocks/core); **B12 / Lamp L2 gamma mcast** has a measured 5.3 % *upper bound* on `grid_filling`
(against the design's predicted 21 %) and should be closed with a real build-and-measure, not an
ablation inference; and `/perf-ceiling-dm` **Mode A** still owes an NPE bracket to replace the design's
hand-derived 350 GB/s constant — `noc_estimate` is a test target and is not built in this tree, so
either build it or record why the bracket stays estimated.

**Done when**: every Part-1 and Part-2 lever has a status and evidence field in the ledger,
`verify_levers` reports 0 blocking / 0 signal / 0 stale, the counterfactual estimates for the unused
levers are written into `changelog.md` as a ranked opportunity list, and no bench shape regressed.

---

## Not in the queue (and why)

- **`{dtype: float32, fp32_dest_acc_en: False}`** — an op-side `EXCLUSIONS` cell by design (fp32
  activations through a bf16 DEST accumulator is a silent precision downgrade). Only move it out if
  the precision convention changes.
- **`{dtype: bfloat8_b} × {w_non_aligned, h_non_aligned}`** and the three
  `{ROW_MAJOR × *_SHARDED × TILE-gamma}` cells — parked in `feature_spec.INVALID`. They are
  *author-scoped*, not structural (the feature spec says so itself), and the three cross-tensor ones
  are mis-authored; see the verification report's INVALID audit. They stay out of this queue until
  they move out of INVALID.
- **`ACTIVE_CORE_CAP` tuning** — measured: no bandwidth knee below the full grid (93.7 / 96.0 / 94.5 /
  110.1 / 106.0 µs at 130/96/64/32/16 cores). The knob stays, the refinement does not.
- **Folding `GAMMA_STAGE_MAX_BYTES` into the L1 solver**, and the reverted `tilize` init-amortization
  in `ingest_gamma()` — neither moves a cell nor has a measured device-ns gap; both are recorded in
  `verification_report.md`.
