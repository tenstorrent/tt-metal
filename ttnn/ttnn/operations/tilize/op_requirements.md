# Operation Requirements: tilize

## Definition

- **Formula**: `output[n, …, h, w] = input[n, …, h, w]` — pure layout conversion. ROW_MAJOR memory
  order is re-laid into TILE order (32×32 tiles of four 16×16 faces, faces face-row-major, elements
  row-major within a face). Element **values** are unchanged; only byte positions move. An optional
  `dtype=` narrows/widens the storage format (value-preserving cast; int↔float is out of contract).
- **PyTorch Reference** (standalone):

  ```python
  def torch_tilize_reference(x: torch.Tensor, dtype=None) -> torch.Tensor:
      """tilize performs NO arithmetic — layout changes, values do not."""
      return x if dtype is None else x.to(dtype)
  ```

  The oracle is a round-trip identity: `ttnn.to_torch(tilize(ttnn.from_torch(x, ROW_MAJOR))) == x`
  (exact for every format that can represent the input; PCC for `→bfloat8_b` and the `fp32→bf16`
  narrowing).
- **Import Path**: `from ttnn.operations.tilize import tilize`
- **Function Signature**:

  ```python
  tilize(
      input_tensor: ttnn.Tensor,                       # ROW_MAJOR_LAYOUT, on device, rank >= 2, H%32==0, W%32==0
      memory_config: ttnn.MemoryConfig | None = None,  # output mem config (default: the input's)
      *,
      dtype: ttnn.DataType | None = None,              # output dtype (default: the input's)
      use_multicore: bool = True,                      # distribute over the compute grid
      use_double_buffer: bool = True,                  # depth-2 CBs (overlap, +L1); False = depth-1
  ) -> ttnn.Tensor
  ```

## Phases

> **Non-regression rule**: Every refinement must pass all tests from prior phases — and must not regress **performance**. Check the refinement's `changelog.md` perf table against the cumulative bench set recorded by prior phases: no prior bench shape's device kernel duration may regress beyond the measurement noise margin. A refinement that speeds up its own target while slowing a shape a prior phase already measured fast is a regression to resolve, not a trade-off to accept silently. If the refinement changed a shape-dependent code path (work distribution, blocking, CB geometry, dtype branch) but the changelog benched only a single point on that axis, flag the missing coverage — a non-regression gate that never measured the other regime cannot have cleared it. This rule is generic: it applies to every op and every optimization and prescribes no particular technique for avoiding the regression.
> **Drift signal**: XPASS-strict failures mean the implementer added support but forgot to update SUPPORTED. The implementer fixes by updating SUPPORTED.
> **Checkbox protocol**: Implementer marks `[x]` when the refinement is complete and all tests pass, `[~]` when real work landed but at least one named axis value is deferred (treated as completed by the queue, surfaced as partial), `[ ]` only when nothing usable was produced.
> **Refinement ID + follow-up naming (mandatory — the runner parses this)**: Primary refinements are `Refinement N` (e.g. `Refinement 1`, `Refinement 2`). When you ship `[~]` partial and file the sharper follow-up the partial-tick protocol requires, name it by appending a lowercase letter to the parent's number: `Refinement 1b`, `Refinement 1c`, … (never `Refinement 1.5`, `Refinement 1 (follow-up)`, or a fresh number). Order follow-ups immediately after their parent so the queue runs them before later refinements — a partial's remaining-blocker follow-up must be picked next, not leapfrogged. The runner's parser matches exactly `Refinement \d+[a-z]?`; any other shape is invisible to the queue and silently skipped.

> **This queue is perf-only — by measurement, not by choice.** At Phase 0 `SUPPORTED` already equals
> the golden `TARGET` on **every** axis (`TARGET − SUPPORTED = ∅`), the verifier CLI reports
> `xfail_expected = 0` **entries**, and `supported_fail = xpass_drift = xfail_wrong_mode = 0`. There
> is no `(axis, missing_value)` pair left to promote and no failing cell to rescue, so every entry
> below is a concrete `ttnn/ttnn/operations/examples/master.md` lever with a **measured** off-ceiling
> gap, plus the single run-closing completeness audit that the perf prompt requires. See
> `verification_report.md` § "TARGET vs SUPPORTED" for the exhaustive per-axis diff.

### [x] Phase 0 — Core Implementation (+ verification pass)

- **SUPPORTED dtype**: [bfloat16, float32, uint32, uint16, int32]
- **SUPPORTED output_dtype**: [bfloat16, float32, bfloat8_b, uint32, uint16, int32]
- **SUPPORTED layout**: input always ROW_MAJOR (structural precondition, not an axis); output always TILE
- **SUPPORTED shape-derived axes**: `rank` ∈ {2,3,4,5,6} (verifier-extended from {2,3,4}); tile-aligned H/W only (this op does not pad — `tilize_with_val_padding` does)
- **SUPPORTED op-specific axes**: `use_multicore` ∈ {False, True}; `double_buffer` ∈ {False, True}; `shard_api` ∈ {none, legacy_2d, nd}; `out_scheme` ∈ {interleaved, HEIGHT, WIDTH, BLOCK, nd}; `buffer` ∈ {dram_to_dram, dram_to_l1, l1_to_l1, l1_to_dram}
- **EXCLUSIONS**: none
- **Cores**: multi-core, 2D height-first rectangular split (`ncores == min(grid_cores, total_tiles)`; sharded → the shard's own cores) — asserted, not eyeballed
- **Compute config**: `fp32_dest_acc_en` derived from dtypes, `unpack_to_dest_mode[0] = UnpackToDestFp32` when set, `dst_full_sync_en = False`, `NoReconfigure` on every no-cast call
- **Golden baseline**: **126 / 126** registry cells passing (90 INVALID-skipped); whole golden dir 515 passed / 1 failed / 2 errors, the 3 non-green all reference-file issues outside the registry matrix (see `verification_report.md`)
- **Perf baseline** (median, WH B0 8×8, this pass): a_square 85 417 ns @ 196.4 GB/s (= 1.01× the in-tree 64-core DRAM→DRAM copy); b_wide_short 13 383 ns on **64** cores; d_tall_narrow 3 658 ns; f_sharded_small 1 382 ns with **zero** DM

---

### [ ] Refinement 1 — Per-core-overhead gating for the low-work-per-core regimes (A0 knee + B0 + depth-2 default)

**Goal**: move the two named low-work-per-core bench regimes off their measured floor by gating
per-core overhead on work-per-core instead of applying it globally:

- **A0 bandwidth-knee clause** — stop launching the full 64-core grid when each core owns ~1–2 tiles.
  `examples/dram_saturation/report.md` measures the knee at **~16 cores @ 190.9 GB/s** (16 → 64 buys
  **+1.5 %**), while dispatch/sync cost scales with the core count. Target: `d_tall_narrow`
  `[1,1,2048,32]` — **measured 3 658 ns, `sync_only` ablation 1 200 ns (33 % of the total), 71.7 GB/s
  achieved = 0.38 of the knee**. Predicted up to ~2×.
- **C16 depth-2 default gating (B0)** — depth-2 CBs are a **measured no-op** on the DRAM-bound
  regimes (`a_square` 85 417 ns depth-2 vs 85 806 ns depth-1, inside the ±2 % noise floor) while
  costing **65 536 B/core**. Default `use_double_buffer` off (or to depth-1) once the planner sees the
  op is DRAM-saturated with large per-core work, keeping depth-2 for the latency-bound small regimes.
  The public kwarg keeps both values and its documented meaning — only the *default* is gated.

**Implementation skill**: /perf-measure, /perf-ceiling-dm

**Verifier notes**: first in the queue because it changes the **core count** on the small regimes, and
Refinement 2's levers act on the per-core transaction budget that this refinement resizes — doing them
in the other order would force re-measuring R2. Keep `use_multicore=False` meaning exactly 1 core (the
acceptance test and the `c_single_core` bench regime depend on it) — the knee cap is a `G` clamp inside
the multicore path, never a new user-visible mode. Both `d_tall_narrow` and `f_sharded_small` are B0
regimes, so counterfactual every lever **on the smallest regime it runs in**, not on `a_square`.

**Done when**: `d_tall_narrow` ≥ 1.5× faster than 3 658 ns with the A0 assert in
`_bench_tilize.py::_assert_structural_gates` updated to the new (gated) expectation rather than
deleted; no regression beyond noise on `a_square`, `b_wide_short`, `c_single_core`, `e_*`, `f_*`,
`g_*`; per-core CB bytes recorded at the gated default for at least one wide and one narrow shape; a
Mode-C ledger row (`lever → predicted Δ → measured Δ → keep/drop`) for each of the two levers; golden
suite still 126/126.

---

### [ ] Refinement 2 — Close the read-path transaction-overhead gap on the DM-bound interleaved regimes

**Goal**: move `b_wide_short` and `c_single_core` toward their computed bounds with the per-transaction
levers Phase 0 left unapplied (all four are `master.md` Part 2 items explicitly pre-classified
`deferred` in `op_design.md`'s Mode-D table):

- **B8 trid double-issue** — keep ≥ 1 read request in flight across the per-block barrier instead of
  draining to zero 32 times per block.
- **B13 `set_state` / `with_state`** — the reader issues **32 same-shape reads per block**, which is
  precisely this lever's use case; today each read re-programs the full NoC command.
- **B10 per-reader VC assignment** and **A3 reader-adjacent-to-its-DRAM-bank placement**
  (`get_optimal_dram_bank_to_logical_worker_assignment`, which *is* bound on this build) — both stack
  on top of the already-applied A1 `row_wise=True` placement.

Measured gaps: `b_wide_short` `[1,1,32,16384]` **13 383 ns for 2.10 MB = 156.7 GB/s, i.e. 0.54 of its
7 300 ns DRAM floor** — it is launch/transaction-rate bound, not bandwidth bound; `c_single_core`
`[1,1,512,512]` **30 472 ns vs the 25 200 ns single-core tt-npe pin = 0.83**. `a_square` is *not* a
target: at 196.4 GB/s it is already 1.01× the in-tree measured 64-core DRAM→DRAM copy of the same
16.78 MB, so its residual 0.68-of-spec-peak is interleaved round-robin congestion, not op overhead —
use it as the no-regression witness.

**Implementation skill**: /perf-ceiling-dm, /perf-measure

**Verifier notes**: run after Refinement 1 (it resizes per-core work, which is the denominator every
lever here divides into). Each lever must be landed and measured **individually** — the prompt's
"every optimization must pay" rule means a bundle that nets +3 % is not evidence that all four paid.
Watch for the interaction with the reader's `noc_async_read_barrier()`: B8's whole point is to make the
barrier non-draining, so do not also convert the barrier to a per-read one (that would re-break B7,
which the design measured at +52 % composed).

**Done when**: `b_wide_short` ≥ 1.2× faster than 13 383 ns **or** each of the four levers has a
recorded measured-no-payoff verdict with its counterfactual number; tt-npe re-pinned (cycles + DRAM
util + congestion %) for `b_wide_short` and `c_single_core`; no regression beyond noise on any prior
bench regime; golden suite still 126/126.

---

### [ ] Refinement 3 — Crossover paths: one-sided zero-copy + bigger sharded-side read transactions

**Goal**: apply the two levers Phase 0 deferred on the interleaved↔sharded crossover (the design's
R3b/R3c; `changelog.md` records them as advisory deviations):

- **C14 one-sided CB aliasing** — on DRAM-interleaved RM → sharded TILE, alias the **output** CB onto
  the local L1 shard so the tilize LLK packs straight into its final address, and symmetrically alias
  the **input** CB on sharded RM → DRAM-interleaved TILE (Phase 0 routes both through the generic
  `TensorAccessor` on *both* sides, so the sharded side pays a full NoC leg it does not need). Measured
  gap: `g_dram_to_sharded` `[1,1,2048,512]` = **18 923 ns** moving 2.10 MB from DRAM + 2.10 MB into L1;
  the DRAM-side leg alone floors at **7 300 ns** ⇒ **achieved 0.39**. `g_sharded_to_dram` = **19 780
  ns**. Same-spec sharded already proves the mechanism works and pays (`f_*`: `no_dm == full` within
  0.2 %, zero DRAM).
- **B5/B6 read-transaction size on a sharded RM input** — `g_sharded_to_dram` plans `chunk_wt = 2`
  ⇒ **128 B** reads, **4× below the 512 B one-packet threshold**, because `chunk_wt` must divide
  `gcd(Wt, page_wt)` so a chunk never straddles a source page. Coalesce rows that *are* contiguous
  inside a shard into one transaction (or issue several chunk-reads per barrier) to get the transaction
  size back up without breaking that invariant.
- **C7 split reader** on DRAM→sharded, where BRISC is idle once the write side is an aliased CB
  (`examples/split_reader` measures up to **1.7×**).

**Implementation skill**: /perf-ceiling-dm, /perf-measure

**Verifier notes**: the reason Phase 0 used the generic path is real and is the whole difficulty here —
one-sided aliasing needs a per-scheme **shard-index → global-tile-index** mapping (HEIGHT / WIDTH /
BLOCK × ROW/COL orientation × ND), which `TensorAccessor` currently does for free. Get that mapping
right before chasing the transaction size: a wrong mapping keeps every CB count balanced and silently
transposes blocks (`op_design.md` Risk #2, and exactly the class of bug that cost Phase 0 26 reference
cells). Order after Refinements 1–2 so the interleaved baseline the crossover shares is already
stable. Do **not** stage the reshard through DRAM to simplify the mapping — the prompt forbids extra
full-tensor DRAM passes, and tt-npe will show it.

**Done when**: `g_dram_to_sharded` ≥ 1.4× faster than 18 923 ns and `g_sharded_to_dram` ≥ 1.2× faster
than 19 780 ns, with tt-npe showing **zero DRAM traffic on the sharded side** of each; the golden
crossover and cross-spec-reshard cells (`[1,1,128,64]` DRAM→HEIGHT, HEIGHT→DRAM, HEIGHT→HEIGHT with a
different grid) still pass, and `test_translated.py` stays at 275 passed; no hangs under `--dev`; a
Mode-C ledger row per lever.

---

### [ ] Refinement 4 — Path-B (same-spec sharded) compute + sync floor

**Goal**: cut the fixed per-launch cost of the zero-copy path, which is the only thing left in it.
Measured decomposition of `f_sharded_small` `[1,1,512,64]` H-sharded, 4 cores: **full 1 382 ns,
`no_compute` 690 ns, `no_dm` 1 362 ns (= full — there is no DM at all), `sync_only` 690 ns** ⇒ roughly
**50 % compute LLK + 50 % dispatch/CB-sync, 0 % data movement**. Levers:

- **Kernel-count reduction (B0)**: on the alias path the reader is one `cb_reserve_back`/`cb_push_back`
  and the writer one `cb_wait_front`/`cb_pop_front` — two whole kernel launches and two CB handshakes
  that exist only to hand compute data that is *already at the CB address*. Collapse to a compute-only
  program (the `examples/zero_copy_fold` precedent with `fold=1`).
- **`InitUninitMode` amortization**: `InitAndUninit` re-issues `tilize_init`/`tilize_uninit` around a
  loop that is often 4–16 blocks on this path.
- **`Fp32Mode::Fast` where it is legal**: the compute kernel picks `Lossless` off the *input* CB format
  alone, so an fp32 → bf16/bf8b cast pays the slow LLK path although the narrower output cannot hold
  the extra precision. Measured **no** cost at grid-filling size (`e_square_fp32_to_bf16` 120 813 ns =
  0.72 of its DRAM floor, the same ratio as every DM-bound regime), so this lever only makes sense
  **here**, on the compute-bound sharded/small cases — and only if the fp32-in/narrow-out sharded cell
  is measured, not assumed.

**Implementation skill**: /perf-measure

**Verifier notes**: the trap is CB bookkeeping, not perf. `WaitMode::NoWait` suppresses only
`wait_front` — `pop_front`, `reserve_back` and `push_back` still execute (`op_design.md` Risk #13), so
a compute-only program has to be reasoned through against `tilize_helpers.inl:224-255` rather than
assumed to work; and `tilize` cannot be in-place (`static_assert(input_dfb != output_dfb)`), so the two
aliased CBs must stay distinct even when in and out dtypes match. Keep the program-cache re-binding
property: with no `Buffer*` runtime arg nothing forces the aliased CB address to be re-patched — this
pass verified re-binding works **because** each core still emits a base-address runtime arg
(`program_descriptors.cpp:198-209`); if the reader/writer kernels go away, that arg must move onto the
compute kernel and the re-binding probe must be re-run (2 calls, different shard addresses, both
bit-exact, cache entries unchanged). Order after Refinement 3 so both sharded paths are touched once.

**Done when**: `f_sharded_small` ≥ 1.3× faster than 1 382 ns and `f_sharded_large` no slower than
2 071 ns; the re-binding probe passes (cache hit + different addresses + bit-exact on both calls); all
same-spec sharded golden cells (HEIGHT/WIDTH/BLOCK/nd, ROW and COL) still pass; `no_dm == full` still
holds (the zero-DRAM claim must survive); Mode-C ledger row per lever.

---

### [ ] Refinement 5 — Perf retrospective: completeness audit (run-closing, LAST)

**Goal**: the one deliberately non-capability entry in this queue — the `/perf-ceiling-dm` **Mode D**
completeness audit over the **full** lever list (`ttnn/ttnn/operations/examples/master.md` Part 1
examples + all 24 Part 2 propositions). For every lever the finished op does **not** use, record
`lever → status (not-applicable / deferred / measured-no-payoff / missed) → predicted Δ if applied →
reason`, and estimate the counterfactual for anything not clearly not-applicable. Grade **A0–A2 per
regime** (tall-narrow, wide-short, square, tiny, sharded-in, crossover) with the *measured* core count
asserted, never holistically. Close with a **ranked list of the real remaining opportunities**, so
anything `missed` or `deferred` with a large predicted delta is surfaced for the next run instead of
being silently dropped.

**Implementation skill**: /perf-ceiling-dm, /perf-measure

**Verifier notes**: order **last** — it audits the finished op, so running it before Refinements 1–4
would audit a state that no longer exists. It is the only entry here that unlocks no cell and changes
no `SUPPORTED` value; do not file a second one. Start from `op_design.md`'s pre-classification table
(it already covers all 24 Part 2 levers) and **overturn it with measurement where the run's numbers
disagree** — e.g. Phase 0 pre-classified C16 (depth-2) as "applied/pays" and the measurement says
neutral-with-an-L1-cost, and this pass's writer-flush change shows B7's cost was per-block-barrier
latency on low-core-count regimes, not transaction count. Both belong in the ledger with their
measured verdicts.

**Done when**: `changelog.md` carries a completeness ledger covering every `master.md` Part 1 example
and Part 2 lever with the four-way status and a predicted delta for each non-not-applicable entry; A0
graded per regime against measured core counts; a ranked remaining-opportunity list; **no regression
beyond noise on any bench regime in the cumulative set** (a, b, c, d, e×3, f×2, g×2, x×3); golden
suite still 126/126 and `tests/ttnn/unit_tests/operations/tilize/` still 93/93 in both default and
`--dev` mode.
