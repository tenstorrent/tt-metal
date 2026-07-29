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
      use_double_buffer: bool | None = None,           # True = depth-2 CBs (overlap, +L1); False =
                                                       # depth-1; None (default) = the planner gates it
                                                       # (Refinement 1, lever C16)
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

### [~] Refinement 1 — Per-core-overhead gating for the low-work-per-core regimes (A0 knee + B0 + depth-2 default)

> **Outcome (2026-07-29): PARTIAL.** Lever 2 (C16 depth-2 default gating) landed and pays: zero
> regressions across the whole cumulative bench set and **−65 536 B/core of L1** on the two widest
> interleaved regimes. Lever 1 (the A0 bandwidth-knee core cap) was implemented and measured on its
> own target regime and is **2.4× SLOWER** (`d_tall_narrow` 3 623 → 8 580 ns at the 16-core knee), so
> it was dropped — the knee is a *bandwidth* phenomenon and this op is read-transaction-rate bound at
> 64 B/page, unable to reach the knee's bandwidth at any core count. The `d_tall_narrow` ≥ 1.5 % gate
> is therefore **not met** (measured 3 609 ns = 1.014×) and is shown unreachable from this
> refinement's lever set; the residual is decomposed to the ns and handed to **Refinement 1c**. Full
> ledger, sweeps and the corrected C16 predicate: `changelog.md` § "Refinement 1".

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



### [x] Refinement 1b — Per-core-overhead gating for the low-work-per-core regimes (A0 knee + B0 + depth-2 default) (debug: fix gate violations)

**Goal**: fix the hard violation from Refinement 1 so the completion gate's three bullets hold.

**Verifier notes** (mechanical, from the harness completion gate):

```
Bullet 3 FAIL: golden responsible cells 126/216 below majority threshold.
```

**Root cause (diagnosed 2026-07-29, this entry's own pass)**: not a kernel, hang, or correctness
defect — a **registry-declaration** defect. Nothing regressed: `HANGS=0`, and all **240/240**
prior-passing golden nodeids still passed (`golden_phase0` vs `golden_refinement_1` `test_results.json`
set-diff = ∅). The 126/216 ratio is **byte-identical to Phase 0's** (the 90 INVALID-skipped cells sit
in the harness's denominator), and a perf-only refinement cannot move it by construction.

What moved was the *threshold*: Refinement 1 declared a third value `"auto"` in
`SUPPORTED["double_buffer"]` for the new `use_double_buffer=None` default. The registry snapshot diff
between the two phases is exactly `[+"auto"]`, which makes
`eval/run_refinements.py::_supported_grew()` classify the phase as a **cartesian expansion** and raise
the bullet-3 bar from `GOLDEN_MAJORITY_FIX = 0.50` to `GOLDEN_MAJORITY_EXPANSION = 0.75`. 126/216 =
**0.583** clears 0.50 and fails 0.75. The value also bought **zero** coverage: `tag_double_buffer`
projects the scenario dict onto a bool, so the golden axis only ever takes `True` (192 cells) /
`False` (24) — never `"auto"`.

**Fix**: `SUPPORTED["double_buffer"]` back to `[False, True]` (== the golden TARGET). `None` is a
*request shape* — "let the planner choose" — not a capability, and the planner only ever picks
depth-1 or depth-2, so `validate()` gates the delegated request against **both** depths instead of
declaring a sentinel. The C16 lever itself (the shipped perf/L1 win) is untouched.
**Generalisable rule**: only declare an axis value the op can be *asked for* and a golden cell can
*take*; gating a *default* is never a new axis value. Guarded by
`test_tilize_refinement1.py::test_double_buffer_axis_stays_two_valued` and
`::test_validate_gates_every_depth_the_planner_may_pick`.

---

### [x] Refinement 1c — `d_tall_narrow` sub-one-packet read path: B13 `set_state` on the 32 stick reads, then C7 split reader

> **Outcome (2026-07-29): COMPLETE via the second gate clause.** Both levers landed, each swept over
> **five read-transaction sizes × two block counts** and gated to exactly where it was measured to
> pay, each with its counterfactual bench row. `d_tall_narrow` **3 609 → 3 431 ns (−4.9 %)**; zero
> regressions across all 22 cumulative bench regimes (max +0.6 %); golden 126/126, unit suite 153/153
> in both modes. The **1.5× gate (≤ 2 439 ns) is not met and is shown unreachable** from this lever
> set: the 437 ns address-gen prize is real (re-priced **461 ns** by the same subtraction) but
> **≈ 73 % of it is hidden behind DRAM service latency**, so removing 20 of 32 accessor calls buys
> only **78 ns** end-to-end. What is left is the 735 ns launch/CB/handshake floor, ~250 ns of LLK, and
> the DRAM **64 B packet rate**, whose only lever (fewer transactions) is measured *slower* because
> the required row permutation costs 32 local L1 gathers per block. Two findings worth carrying:
> the levers are **mutually exclusive** (B13 on top of C7 is +0.9 %), and
> `noc_async_read_one_packet_with_state` **hangs every core on a watcher build** (metalium bug — the
> general `set_state`/`with_state` API is fine). Full ledger: `changelog.md` § "Refinement 1c".

**Goal**: the remaining half of Refinement 1 — get `d_tall_narrow` `[1,1,2048,32]` below
**2 439 ns** (the parent's 1.5× gate) from its measured **3 609 ns**. Refinement 1 proved the parent's
lever (an A0 core cap) is a 2.4× regression here and that the regime already sits **inside its own
per-core NoC bracket** `[2892 … 4461] ns`, so the win cannot come from work distribution. It must come
from the two terms that dominate a 1-tile-per-core block, both **measured by subtraction** this pass
(temporary reader patch, reverted; `changelog.md` § "Refinement 1" → Perf gate):

| term | ns | share | lever |
|---|---|---|---|
| bare launch + CB handshakes (1 block) | 764 | 21 % | none in scope (per-launch floor) |
| **32 × `accessor.get_noc_addr`** | **437** | **12 %** | **B13 `set_state` / `with_state`** |
| **32 × 64 B `noc_async_read` issue + DRAM service + barrier + 1 × 2048 B write** | **1 504** | **42 %** | **B13 (command programming) + C7 (halve the issue count)** |
| tilize LLK (1 tile) | 931 | 26 % | none in scope |

Budget: −1 197 ns is needed. B13 can plausibly take most of the 437 ns of address-gen plus a share of
the per-read command programming inside the 1 504 ns; C7 halves what is left of the issue cost by
putting the idle BRISC to work. **Neither alone is likely to be enough — this entry is deliberately
two levers.**

- **B13 `noc_async_read_one_packet_set_state` / `..._with_state`** — the reader issues **32
  same-shape 64 B reads per block** to varying addresses (`tilize_helpers_dataflow.inl:117-127`),
  which is exactly this lever's use case: program the command buffer once per block, then per read
  write only the varying address. At 64 B every read is already on the one-packet path (B6), so the
  `one_packet` variant is the right family. This subsumes most of the priced 437 ns because the
  general `TensorAccessor::get_noc_addr` re-runs the rank loop plus a `% / /` by the non-power-of-two
  bank count (12) per read (`tensor_accessor.h:175-193, 303-311`) for a page sequence whose bank id is
  a strength-reducible **increment** of `row_page_stride`.
- **C7 split reader** — with 1 block/core BRISC parks in `cb_wait_front` for the entire read window
  (~1.5–3 µs) and then does ~1.4 µs of work, so NoC1 issue capacity is the only structurally unused
  resource on the core (`ttnn-static-analyzer`, this pass). Split the 32 stick reads across NCRISC and
  BRISC.

**Implementation skill**: /perf-measure, /perf-ceiling-dm

**Verifier notes**: ordered immediately after its parent, and **before Refinement 2** — R2 owns B13
for `b_wide_short` (512 B reads, 8 tiles/core), but the lever's *payoff* is largest where the reads
are smallest, so calibrate it on this regime first (master.md B0: counterfactual a per-core-overhead
lever on the smallest regime it runs in) and let R2 inherit the working code. Two hard constraints
from this pass's static analysis, both of which silently corrupt rather than fail loudly:
(a) **a split reader must not have both NCRISC and BRISC `cb_reserve_back`/`cb_push_back` into
`cb_rm_input`** — that is a single-producer-per-CB violation and corrupts the CB pointers; use two
input CBs (compute alternating blocks) or a semaphore-ordered handoff; (b) **`WaitMode::WaitUpfront`
is now a guaranteed hang** for any core with `num_blocks > 1`, because Refinement 1 made depth-1 the
default (`tilize_helpers.inl:216-219` would wait `chunk_wt * num_blocks` pages on a `chunk_wt`-page
CB) — `WaitBlock` in `tilize_compute.cpp` is load-bearing. Also do **not** reach for a bigger read
transaction here: a `W=32` RM input has 64 B DRAM pages and consecutive pages land on *different*
banks (round-robin), so rows cannot be coalesced without a permutation the tilize LLK cannot consume —
that is why this entry is about transaction *rate*, not transaction *size*.

**Done when**: `d_tall_narrow` ≥ 1.5× faster than 3 658 ns **or** each of B13 and C7 has a recorded
measured-no-payoff verdict with its counterfactual number and the re-priced decomposition; the
address-gen term re-priced after B13 (same subtraction method, so it is comparable to the 437 ns
baseline); no regression beyond noise on any regime in the cumulative bench set (a, b, c, d, e×3,
f×2, g×2, x×11); golden suite still 126/126; `tests/ttnn/unit_tests/operations/tilize/` still green in
both default and `--dev` mode.

---

### [x] Refinement 2 — Close the read-path transaction-overhead gap on the DM-bound interleaved regimes

> **Outcome (2026-07-29): COMPLETE via the second gate clause.** All three levers in the re-scoped set
> were implemented and measured **individually**. **B8 (trid double-issue) landed and pays**: gated to
> `blocks ≥ 2 ∧ (ncores ≤ 4 ∨ read ≤ 128 B)` from two device sweeps (7 core counts at 1024 B; 5 read
> sizes at 64 cores × 2 blocks), it takes `c_single_core` **30 359 → 27 343 ns (−9.9 %)**,
> `x_wide_short_1core` **−11.3 %**, `n_tall_narrow_4blk` **−15.3 %** and `x_tall_narrow_16c` **−7.1 %**,
> with an isolation row (`b8=3`: third CB window, no trid pipeline) measuring **1.000** — so the whole
> win is attributable to the reads staying in flight, not to the deeper CB. **B10 and A3 each carry a
> measured-no-payoff verdict with its counterfactual**: B10 is a **regression** (reads 1.083/1.105,
> writes 1.780/1.893, both 1.991/2.142) and A3 is neutral (1.002–1.017). The `b_wide_short` ≥ 1.2 %
> clause is **not met and is shown unreachable by any per-transaction lever**: tt-npe pins that regime
> at **103.2 % DRAM BW utilisation with a 0.4 % congestion term**, which overturns this entry's own
> "launch/transaction-rate bound, 0.54 of its 7 300 ns DRAM floor" premise (that floor divided by
> 288 GB/s *spec* peak, unattainable for 512 B **partial-page** reads). Zero regressions across all 39
> carried bench regimes (max +1.4 %); golden 126/126; unit suite 193/193 in both modes. Residual
> decomposed and handed to **Refinement 2b**. Full ledger: `changelog.md` § "Refinement 2".

> **Re-scope before running (Refinement 1c finding, measured on this exact regime): B13 and C7 are
> REFUTED on `b_wide_short`, do not re-try them.** Refinement 1c implemented both, and forced each
> onto this regime as a bench row: at `b_wide_short`'s 512 B reads **B13 = 15 981 ns vs 13 423 ns
> (+19.1 %)** and **C7 = 15 265 ns (+13.7 %)**; at 256 B they are +3.0 % / +6.2 %, and only at
> **≤ 128 B** does B13 pay. Mechanism, both measured: `set_state` pins the NoC coordinate, so B13 can
> only issue **bank-major**, and queueing 2-3 consecutive same-bank reads of ≥ 256 B costs more
> DRAM-endpoint serialization than the saved command programming buys; C7 doubles the read issuers per
> core, which helps only while each core reads its **own** rows — `b_wide_short` has `nt_h == 1`, so
> all 64 cores read the same 32 source pages and a second issuer just deepens the hot spot. The
> counterfactual rows are permanent (`x_wide_short_b13_forced`, `x_wide_short_c7_forced`,
> `x_wide_short_8k_b13_forced`, `x_wide_short_8k_c7_forced`), so the verdict is re-measurable. This
> leaves **B8, B10 and A3** as this entry's real lever set.

**Goal**: move `b_wide_short` and `c_single_core` toward their computed bounds with the per-transaction
levers Phase 0 left unapplied (all four are `master.md` Part 2 items explicitly pre-classified
`deferred` in `op_design.md`'s Mode-D table):

- **B8 trid double-issue** — keep ≥ 1 read request in flight across the per-block barrier instead of
  draining to zero 32 times per block.
- ~~**B13 `set_state` / `with_state`**~~ — **refuted on this regime by Refinement 1c** (+19.1 %); see
  the re-scope note above. Landed and gated to ≤ 128 B reads, where it pays 3-5 %.
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

**Done when**: `b_wide_short` ≥ 1.2× faster than 13 383 ns **or** each of the remaining three levers
(B13's verdict is already recorded by Refinement 1c) has a
recorded measured-no-payoff verdict with its counterfactual number; tt-npe re-pinned (cycles + DRAM
util + congestion %) for `b_wide_short` and `c_single_core`; no regression beyond noise on any prior
bench regime; golden suite still 126/126.

---

### [x] Refinement 2b — `b_wide_short`'s 64-way partial-page fan-in: whole-page reads + L1 redistribution

> **Outcome (2026-07-29): COMPLETE via the second gate clause.** The entry's named algorithm was
> implemented **in full** — 3 phases, bit-exact on first light — and is **refuted with its
> counterfactual number and a re-pinned tt-npe DRAM-util figure**: **18 468 ns vs 13 416 = 1.377×
> SLOWER**. The decisive number is its own **read-side ceiling probe** (phase 1 only, no exchange):
> the read leg is **5 966 → 5 985 ns**, i.e. a **32× bigger transaction moves the same bytes in the
> same time**, so this entry's premise — that a partial-page fan-in costs DRAM bandwidth — is **false
> on this hardware**. The 156.9-vs-179.3 GB/s gap it rests on is between two shapes with different
> bytes-per-fixed-overhead ratios, not two page-access patterns. The whole algorithm's ceiling is
> **5.9 %** (the 32 saved read *issues*), and its L1 leg (+4 676 ns) plus 32-core barrier (+1 217 ns)
> spend that three times over.
>
> A **one-sided DM ablation** (new `TILIZE_SKIP_DM=2/3`) built to produce that verdict found the real
> residual on the side nobody had measured: the **WRITE** leg is the slower one (135 vs 176 GB/s), and
> the two legs overlap by only 2 482 of a possible 5 966 ns because `nt_h == 1` gives one chunk-block
> per core. That pointed at the issue **order** rather than its size, and a **per-core
> transaction-order rotation** landed: **`b_wide_short` 13 367 → 12 554 ns (1.065×)**,
> **`m_wide_short_8k` 8 124 → 7 208 (1.127×)**, 32k member 1.082×, at **zero L1 cost** — i.e. it beats
> the refuted algorithm's own theoretical ceiling. Both halves are **superadditive** (0.992/0.985 alone
> vs 0.929 together), which is why they ship as one gate. Re-blocking to create the missing overlap was
> also measured and refuted (1.019 / 1.153 / 1.892 at chunk 4 / 2 / 1).
>
> The ≥ 1.14× clause is **not met and is shown unreachable**: 11 700 ns needs the DM at **222 GB/s =
> 1.15× the measured achievable 64-core DRAM copy**, on a regime tt-npe now pins at **116.5 % DRAM BW
> utilisation with 0.2 % congestion** (up from 102.3 % / 0.7 %). The residual is the launch + tilize-LLK
> floor (`no_dm` = 17.9 % of the runtime on a one-block kernel) — Refinement 4's lever, not a DM one.
> Zero regressions across all **81** carried bench regimes; golden 126/126 (240 passed / 0 failed);
> unit suite 253/253 in both modes. Full ledger: `changelog.md` § "Refinement 2b".

**Goal**: `b_wide_short` `[1,1,32,16384]` from **13 367 ns / 156.9 GB/s** toward the **179.3 GB/s** the
same 512 B transaction already achieves when it reaches *private* source pages, i.e. ≥ 1.14× (≤ 11 700
ns). Refinement 2 measured **every** per-transaction lever on this regime and all of them are refuted,
so this entry is deliberately an **algorithm** change, not a knob:

| lever tried on `b_wide_short` | measured | owner |
|---|---|---|
| B13 stateful bank-major reads | +19.5 % | R1c |
| C7 split reader | +14.2 % | R1c |
| B10 per-core static unicast VC | +99 % (writes +78 %, reads +8.4 %) | R2 |
| A3 bank-adjacent core order | +1.7 % | R2 |
| B8 trid double-issue | structurally inapplicable (1 chunk-block/core) | R2 |
| finer chunking to 2 blocks × 256 B | +1.3 % (`p_2blk_256B` 13 536 vs 13 367) | R2 |

**The residual, priced by two bench rows with identical transaction size and core count:**
`b_wide_short` gets **156.9 GB/s** where all 64 cores read 512 B slices of the **same 32** source pages
(a `W=16384` bf16 RM row *is* one 32 768 B DRAM page, and `nt_h == 1`), while `p_2blk_512B`
`[1,1,4096,256]` gets **179.3 GB/s** at the same 512 B read because each core owns its **own** 64
consecutive pages. `p_2blk_1024B` gets **188.8**. So the binding term is *which* pages the 64 readers
hit — a 64-way partial-page fan-in — and tt-npe agrees it is the DRAM endpoint (**103.2 % DRAM BW util,
0.4 % congestion, max link demand 185.9 %**), not the NoC and not congestion.

**Concrete lever to try**: read **whole source pages** instead of 512 B slices. A subset of cores
(≥ 32, one per source row) each issues **one 32 768 B** `noc_async_read` of its whole row into L1, then
the row is redistributed to the 64 tile-column owners over **L1-to-L1** transfers (or the tile-column
owners read their slices from that core's L1 instead of from DRAM). DRAM then sees **32 whole-page
reads** rather than 2 048 partial-page reads for the same bytes. Cost to weigh: one extra L1 hop for
every byte, and 32 768 B/core of staging L1 (which is inside the existing `L1_CB_BUDGET_PREFETCH_BYTES`
headroom only for a *chunked* variant — stage one 8 KB quarter-row at a time).

**Implementation skill**: /perf-ceiling-dm, /perf-measure

**Verifier notes**: this is the first entry in the queue that changes the *transfer algorithm* rather
than its parameters, so run `/perf-ceiling-dm` **Mode A** first and rank at least two candidates
(whole-page + mcast to the column owners vs. whole-page + each owner pulling from the reader's L1)
before writing a kernel — the L1 hop is real and could easily eat the 14 %. Two hard constraints:
(a) the prompt forbids extra full-tensor **DRAM** passes, so the redistribution must stay in L1;
(b) `d_tall_narrow`/`a_square` must not be touched — this pattern only exists when `nt_h == 1`, so gate
it on that (a `distribution_gate`, not a wholesale switch), and keep the `x_wide_short_*` counterfactual
rows so the R1c/R2 refutations stay re-measurable.

**Done when**: `b_wide_short` ≥ 1.14× faster than 13 367 ns **or** the whole-page+redistribute
algorithm has a recorded measured-no-payoff verdict with its counterfactual number and a re-pinned
tt-npe DRAM-util figure showing why; no regression beyond noise on any regime in the cumulative bench
set; golden suite still 126/126; `tests/ttnn/unit_tests/operations/tilize/` still green in both default
and `--dev` mode.

---

### [~] Refinement 3 — Crossover paths: one-sided zero-copy + bigger sharded-side read transactions

> **Outcome (2026-07-29): PARTIAL.** Both named levers landed on **both** crossover directions, plus
> the coalesced sharded read, and three further levers were implemented and refuted with
> counterfactuals. **One of the two numeric clauses is met**: `g_sharded_to_dram` **19 780 → 15 112 ns
> = 1.309×** (bar 1.2×), pinned by tt-npe at **98.1 % DRAM BW util / 0.39 % congestion** with a −1.6 %
> prediction error, i.e. at its DRAM write bound. **The other is not**: `g_dram_to_sharded`
> **19 158 → 16 006 ns = 1.197×** (bar 1.4×). Everything else in the gate holds — zero DRAM traffic on
> the sharded side of each, proven three ways (kernel CT arg, `no_write == full` / `no_read == full`
> ablation, tt-npe per-NoC demand **0.0**); golden crossover + cross-spec cells pass (126/126, 240/240);
> `test_translated.py` 275; no hangs in either mode (two were *found and fixed*, one reachable from a
> plain public call); a Mode-C ledger row per lever.
>
> Why the 1.4× is not met, priced by ablation: the read payload is **10 055 ns for 2.10 MB = 209 GB/s**,
> within 2.3 % of the best DRAM read rate this op has ever measured — so the DM is done. The residual is
> address generation (3 815 ns, already ~70 % hidden — B13 removes 62 % of the calls and saves only
> 930 ns) plus a ~2 136 ns launch/CB floor. Every issue-side lever was measured on this exact operating
> point and refuted: **C7 +11.6 %, B8 +9.9 %, read-grouping 1.020/1.116/1.245**, because the read is
> **bank**-bound, not issue-bound (the address probe that collapses 12 banks onto 1 costs **2.80×**).
> **2 340 ns of the 16 006 is not attributed by any ablation variant**, and that is where a 1.4× would
> have to come from — handed to **Refinement 3b** with the one measurement this pass did not take.
> Full ledger, sweeps, tt-npe pins and the two hang post-mortems: `changelog.md` § "Refinement 3".

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

### [x] Refinement 3b — `g_dram_to_sharded`'s unattributed 2 340 ns: per-RISC timeline, then the writer-kernel drop

> **Outcome (2026-07-29): COMPLETE on the gate's OR clause.** The 1.4× is **not** met
> (**16 231 ns** vs a 13 517 bar) — and this entry is the proof that it *cannot be* on this
> regime. **The attribution clause is met in full**: a per-RISC Tracy timeline
> (`TILIZE_ZONES=1`) assigns every one of the 16 954 cycles to a named term, and the answer is
> **TRISC0 blocked in `cb_wait_front` 15 592 / 17 316 = 90 %, NCRISC blocked in
> `cb_reserve_back` 350 / 17 017 = 2 %, BRISC blocked in `cb_wait_front` 17 372 / 17 522 =
> 99 %** — the reads are the bound, compute never is, the writer is pure overhead. The
> residual is **dispatch, not kernel**: launch prologue 863 + FW epilogue 580 + **core-to-core
> skew 1 129** = **2 572 ns (15.2 %)**, plus the reader's own ~1 400 ns prologue; the per-block
> CB handshake that Refinement 3 folded into its 2 136 ns estimate is only **~600**.
> **Why 1.4× is unreachable**: the reader kernel *alone* is **14 346 ns**, above the bar,
> before one cycle of dispatch — and its read leg is at 209 of a 214 GB/s DRAM best, with a
> 1024 B transaction worth only 2.4 % over the 128 B the shard-shaped split forces. Floor with
> the reader prologue, both CB stages **and all dispatch at zero**: **12 340 ns = 1.53×**.
>
> Both named levers carry their own measured verdict. **Lever 2 (drop the writer kernel on
> `alias_out`) LANDED** — a fixed **~45 ns/launch**: 0.995 on the small shape (4 sessions, same
> sign, CV ≤ 0.6 %), 0.999 on the 16 µs one, with the timeline giving the mechanism
> (**BRISC-FW end −42 cycles**, NCRISC/TRISC unchanged); it ships **two kernels**, keeps
> program-cache re-binding, and is R4's precedent. **Lever 3 (hoisted interleaved bank table)
> IMPLEMENTED AND REFUTED** — `dataflow_kernel_lib::InterleavedStickBands` removes **84 of 96**
> accessor calls per core and is **1.016** against B13 (reproduced 1.009), which reframes B13:
> what it buys is the **armed command buffer**, not the arithmetic, and the arithmetic is hidden
> behind DRAM service anyway. Zero regressions across **162** bench rows; golden 126/126
> (240/240); `test_translated.py` 275; unit suite **359/359** in both modes; no hangs.
> Full timeline, ledger and both refutations: `changelog.md` § "Refinement 3b".

**Goal**: the remaining half of Refinement 3 — get `g_dram_to_sharded` `[1,1,2048,512]` → BLOCK-sharded
below **13 517 ns** (the parent's 1.4× gate) from its measured **16 006 ns**. Unlike the parent, this
entry starts from a decomposition rather than a lever list, because Refinement 3 measured **every**
DM lever on this operating point (128 B reads × 8 blocks × 64 cores) and refuted all of them:

| term | ns | share | status |
|---|---|---|---|
| DRAM read payload | **10 055** | 63 % | **irreducible** — 209 GB/s of a 214 GB/s measured best |
| launch + per-block CB handshakes | ~2 136 | 13 % | this entry's lever 2 |
| exposed share of the 3 815 ns address generation (~30 %) | ~1 145 | 7 % | ceiling ~400 ns, see below |
| tilize LLK (marginal) | 334 | 2 % | overlapped |
| **unattributed** | **~2 340** | **15 %** | **this entry's lever 1** |

Arithmetic floor with the read irreducible and the launch/CB floor intact: **~12 200 ns = 1.57×**. So
the gate is reachable *if and only if* the unattributed term is recoverable — which is why the first
deliverable is a measurement, not a code change.

- **Lever 1 — a per-RISC Tracy timeline on the aliased plan.** No `TILIZE_SKIP_*` ablation variant can
  attribute the residual: `no_dm` keeps the address-gen sink, `no_compute` keeps the CB dance, and the
  read payload's marginal cost already accounts for 10 055 of the 16 006. What is missing is *which
  RISC is waiting when* — NCRISC blocked in `cb_reserve_back` (compute is the bound), TRISC blocked in
  `cb_wait_front` (the reads are), or neither (a per-block handshake latency that neither side
  attributes). Instrument with `DeviceZoneScopedN` around the reader's per-block reserve / read / push
  and the compute's wait / LLK / push, then read the zones per RISC. **The answer selects the next
  lever; do not guess it.**
- **Lever 2 — drop the writer kernel on `alias_out` (kernel-count reduction, B0).** Refinement 3
  verified the precondition rather than assuming it: the aliased output CB has exactly `shard_tiles`
  pages and compute pushes exactly `shard_tiles`, so **the CB never needs recycling** — the writer's
  single `cb_wait_front`/`cb_pop_front` exists only to close the loop. Worth ~200-400 ns from R1c's
  bare-launch price, on every DRAM→sharded call. Two traps, both already priced by prior phases:
  `WaitMode::NoWait` suppresses only `wait_front` (`op_design.md` Risk #13), and the base-address
  runtime arg must move onto a surviving kernel or program-cache re-binding breaks (Refinement 4's
  verifier note, and `test_alias_program_cache_rebinding` is the probe that would catch it).
- **Lever 3 — row-major incremental interleaved address generation.** 12 accessor calls per **core**
  (not per block): compute the 12 bank base addresses once, then row *r* is
  `base[(p0+r) % 12] + ((p0+r) / 12) * aligned_page`, keeping **row-major** issue order (which is what
  B13 gives up, and why B13's bank-major order costs it its own saving). **Its ceiling is already
  priced at ~400 ns / 2.5 %** by B13's measured delta (B13 removes 62 % of the calls and saves 930 of a
  3 815 ns term ⇒ ~70 % of the term is hidden), so this is a *combination* lever, not a standalone one.

**Do not re-try on this regime** (all measured by Refinement 3, counterfactual rows retained):
C7 split reader (+11.6 %), B8 trid double-issue (+9.9 %), read-grouping B7' (1.020 / 1.116 / 1.245 at
G = 2 / 4 / 8), a bigger read transaction (the shard is 64 columns wide — the transaction size is the
work split's, and the read is bank-bound anyway: collapsing 12 banks onto 1 costs 2.80×).

**Implementation skill**: /perf-measure, /perf-ceiling-dm

**Verifier notes**: ordered immediately after its parent and **before Refinement 4**, because lever 2
*is* R4's kernel-count reduction with its precondition already verified on this path — landing it here
gives R4 a working precedent on the simpler (one-sided) alias before it attempts the same on Path B,
where the reader must also go. Keep `x_g_alias_*` and `p_g_to_sharded_r3_off` as the counterfactual
rows so the parent's refutations stay re-measurable, and re-run `--dev` after any CB-arithmetic change:
Refinement 3's second hang (a `cb_push_back` that straddled the FIFO end) was **silent** in the default
build and only surfaced as an ebreak under the lightweight asserts.

**Done when**: `g_dram_to_sharded` ≥ 1.4× faster than 18 923 ns **or** the per-RISC timeline attributes
the ~2 340 ns residual to a named term with a measured number, and each of levers 2 and 3 carries its
own measured verdict (kept or refuted with a counterfactual row); no regression beyond noise on any
regime in the cumulative bench set (155 rows); golden suite still 126/126;
`tests/ttnn/unit_tests/operations/tilize/` still green in both default and `--dev` mode.

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
  > **Re-scope before running (Refinement 1 finding, `ttnn-static-analyzer`)**: this lever as written
  > has **zero headroom**. `InitAndUninit` already places init and uninit *outside* the `num_blocks`
  > loop (`tilize_helpers.inl:179-200`, `:258-272`), so it is 2 config bursts per *kernel*, never per
  > block; and the `uninit` cannot be dropped — `_llk_unpack_tilize_init_` leaves `tileize_mode=1` +
  > `shift_amount` in `THCON_SEC0` (`llk_unpack_tilize.h:97-108`) and leaking that to the next program
  > on the core is a silent-corruption bug. The only remaining form is chaining
  > `InitOnly/Neither/UninitOnly` across *multiple* `tilize()` calls, which this kernel does not have.
- **`Fp32Mode::Fast` where it is legal**: the compute kernel picks `Lossless` off the *input* CB format
  alone, so an fp32 → bf16/bf8b cast pays the slow LLK path although the narrower output cannot hold
  the extra precision. Measured **no** cost at grid-filling size (`e_square_fp32_to_bf16` 120 813 ns =
  0.72 of its DRAM floor, the same ratio as every DM-bound regime), so this lever only makes sense
  **here**, on the compute-bound sharded/small cases — and only if the fp32-in/narrow-out sharded cell
  is measured, not assumed.

**Implementation skill**: /perf-measure

> **Precedent from Refinement 3b (2026-07-29)**: the kernel-count reduction is **already done on
> the one-sided `alias_out` alias** — the writer is not launched, the output base-address runtime
> arg moved onto the compute kernel, re-binding was re-probed, and the cross-launch CB question is
> settled (the firmware's `setup_local_cb_read_write_interfaces` sets
> `tiles_acked_received_init = 0` every launch, so un-popped pages cannot leak forward). It is
> worth a **fixed ~45 ns**, not the 200-400 ns projected — so on `f_sharded_small` (1 365 ns)
> expect the *reader's* removal to be the larger half. R3b also prices what B0 is competing with:
> **2 572 ns of dispatch/firmware** (launch prologue 863 + FW epilogue 580 + core-to-core skew
> 1 129), of which only the kernel entry/exit is B0's to take.

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
