# Design: Quasar-native `binary_ng` program factory (phase 1)

> **Committed for the record.** This is an engineering record of the Quasar-native `binary_ng` effort, not
> user documentation. Two kinds of reference in here point outside the repository and are expected to
> dangle for a reader who was not on the original branch:
> - `debug/attrib/*` — the diagnostic drivers, sweeps and plotting scripts. `debug/` is deliberately
>   untracked; the numbers they produced are reproduced inline here.
> - `.link_to_claude/plans/*` — the implementation plan, the specialist review findings, and the
>   measurement-discipline notes, which stayed out of the repo.
>
> Corrections and withdrawn claims are kept in place rather than deleted. Several conclusions here were
> wrong for days before being caught, and the record of how is the more useful half of the document.

Status: **design v3 — measured.** Five specialist reviews (2026-08-19); re-scoped against craq-sim's real
measurement capability and reviewed again by four specialists (2026-08-20); then **every lever testable
without the native factory was actually run** (2026-08-20). Not yet implemented.

> **Read §2.4 first.** All three levers measurable on the existing factory are small (~1.17× combined **on
> craq-sim**; two of them are undervalued there by construction and may be far larger on the emulator), so the
> project now rests on DM thread count, which cannot be measured before implementing. §2.4 states the
> expectation and a **kill criterion** to apply at the first milestone. Four magnitude estimates were made from
> attribution during this work and all four were wrong, while every *mechanism* prediction held — hence the
> standing rule: **attribution locates cost; only an experiment reveals what is recoverable.**

> **Review provenance — `.link_to_claude/plans/quasar-native-binary-ng-review-findings.md`: round 1 in §A–§F (five specialists),
> round 2 in §G–§K (five specialists), and §K-MEASURED-1..5 (the measurements).** Four blockers and the bulk of the ~30 other corrections are folded
> in below; what remains in the findings doc is plan-level detail (`NativeTuning`'s home, pruning the dead
> `#if` branches, recording `NOC_API_V2`, the `worker_grid` severity note, the one-time cache invalidation, the
> `log_debug` on gate rejection). Five findings were confirmed independently by two agents each. The blockers
> were:
> 1. **`SubtileBroadcastType::NONE` is not "no broadcast"** — it compares H and W only, so leading-dim
>    broadcast passes the gate and breaks both §4.3's linear collapse and §3.2's tile count. Fixed in §3.2.
> 2. **A byte-identical commit 1 cannot compile or link** — 32 redefinitions (not 4), and
>    `create_program_artifacts` is a duplicate external symbol even with unity off. Fixed in §3.1/§3.4.
> 3. **The kernels are not thread-generic today** — a faithful copy hangs at R>1. Fixed in §3.4.
> 4. **`kernel_launches` is a null test** — it counts RISCs out of reset, so R and W cannot move it and C is
>    already saturated. Removed from §2.3.1.
Working branch `dchen/binary_ng_quasar_native` (cut from `origin/main` @ `0cf20188874`).

- Substrate facts + measured baseline: [`QUASAR_NATIVE_RESEARCH.md`](QUASAR_NATIVE_RESEARCH.md)
- Review findings this revision is derived from, with all evidence:
  [`.link_to_claude/plans/quasar-native-binary-ng-review-findings.md`](.link_to_claude/plans/quasar-native-binary-ng-review-findings.md)

Several v1 decisions were revised and six of its factual claims were measured false; the findings doc
carries the evidence. Read it before questioning a decision here.

---

## 1. Goal and scope

Use the machine the current functional path leaves idle: **6 user DM cores instead of 2, and 4 Tensix engines
instead of 1.** Those two are the goal. The other two levers originally listed alongside them — per-thread ring
depth and hardware-posted credits instead of a per-tile `async_read_barrier` — have since been **measured at
1.02× and ≤1.10× on craq-sim** (§2.4). Treat them as refinements *for the sim bring-up only*: both are
latency-hiding levers and craq-sim has no latency, so those figures are lower bounds and the emulator may rank
them first.

**Phase-1 admitted slice:** no-broadcast tensor-tensor, TILE 32×32, **bf16**, FPU `add`, all three operands
**DRAM-interleaved**, no activations, **even divisibility** (§3.2).

**Not in phase 1:** uneven tile counts, sharded/borrowed operands, mixed layouts, fp32/SFPU, MX formats,
activations, any broadcast, tensor-scalar, row-major, where/quant/int32. Ordering in §8.

**Two prerequisites outside this op** (§9.1): a tt-llk `thread_local` fix before `compute_threads > 1`, and
a decision about `llk_tdma_guard` before trusting `TT_METAL_LLK_ASSERTS` at C>1.

## 2. Success criteria

### 2.1 The metric — one estimator, thread-count invariant

**Headline: median across clusters of the per-cluster kernel span.** For each cluster, span =
`max(ZONE_END) − min(ZONE_START)` over that cluster's RISCs on zones containing `KERNEL`; then take the
median over the 32 clusters. Divide by tiles-per-cluster for a rate. (The scripts and the profiler say
`ncores`/`CoreCoord` — one of those is one Quasar cluster, 8 DM + 4 Neos.)

This is invariant to thread count by construction — adding RISCs adds members to a max/min rather than
changing the meaning. **Never divide a per-RISC duration by tiles-per-cluster**: at R=4 the reader is four rows each covering
`T/R` tiles, so dividing each by `T` fabricates a ~3.4× win from the metric alone. `debug/prof_summary.py` implements this metric (verified: reproduces 8549 / 213.72). Do not revert it to a
global span across all clusters — that folds in the 24-cycle inter-cluster dispatch skew, which grows as more DM
RISCs are released.

Report alongside the headline, never instead of it: `(intercept, slope)` and the per-role spans as the
load-balance instrument.

**The model, with its terms named.**

```
span(T) = prologue + marginal x T          T = tiles per CLUSTER, not total tiles
```

- **prologue** — the T-independent term: init, DFB setup, launch. It **grows with thread count** (more
  producer threads, three DFBs, a `sync_threads` in the drain), which is the whole reason it must be
  separated rather than amortized.
- **marginal** — the slope of **span vs T**. It is a measured steady-state **cost** in cycles per tile,
  so lower is better. It is *not* a gain, and it is *not* the slope of `span/T` (that derivative is
  `-prologue/T^2`, the decay of the span basis).
- **the span basis**, `span/T = marginal + prologue/T`, decreases toward the marginal **from above** as T
  grows and never reaches it. Span itself is linear in T and approaches nothing. ("Span basis" rather
  than "average basis" because the headline span is already a median over clusters — calling the rate an
  average invites confusion about which averaging is meant.)

**Perf gain is a ratio, and it has two bases. Name the basis every time.**

| basis | formula | property |
|---|---|---|
| **marginal** | `marginal_baseline / marginal_config` | prologue-free ⇒ the asymptotic gain; **this is what the gates use** |
| **span** | `(span/T)_baseline / (span/T)_config` at a stated T | always **smaller**, because the prologue is a larger fraction of the faster config's total |

Same run, both defensible, different numbers: `R=4,C=4,W=2` is **4.00x** on the marginal basis and
**2.70x** on the span basis at the 1280-tile benchmark shape. Quoting one against the other silently
shifts the target, which is why `(prologue, marginal)` is reported alongside — either basis can then be
recomputed by the reader.

**Marginal perf gain is more illustrative, but span perf gain is more real.** The marginal isolates
steady-state scaling and is the only basis on which configs with different thread counts can be compared
at all; the span basis is what somebody running the op at a given tensor size actually experiences. Lead
with the span number for a result and give the marginal as the asymptote — never the reverse.

> **Corrected 2026-08-28 — this previously specified a TWO-POINT fit, and that is what produced a week of
> wrong marginals.** Two points fit a line through anything and leave no residual, so they cannot reveal
> that the range is curved. Every Milestone 1 marginal was `(span@40 - span@20)/20`; the native path's
> span is still bending there, which biased each config low by an amount scaling with prologue/slope
> (1.3% at `1,1,1`, 5.8% at `4,4,2`) and manufactured a 4.18x end-to-end speedup **above a hard 4.0
> ceiling**. Re-fitted over 60/120/180 the answer is 4.00x, and the cost model goes from 27/31-within-1.3%
> to **31/31 within 0.04%**.
>
> **The rule:** fit over **at least three** tile counts, and *check the successive differences are equal*
> before trusting the slope — that check is the point of the third point, not precision. **Linearity is
> per-config, not per-op:** a larger prologue pushes the bend further out, so the linear region must be
> re-established for each configuration rather than inherited (research base §3.4's "40/core is within 4%
> of the asymptote" is true of the 2-DM-core baseline and false of `4,4,2`, which is 64% above it).
> Where a theoretical bound exists, check the result against it — exceeding one is proof of method error.
> Note the *range* was the error, not the point count: two points at 60 and 180 give the right answer.

### 2.2 Baseline and the measured floor

**Every number here is craq-sim at T=40 tiles/core.** It is a functional simulator with no transfer latency
and no contention model, so these figures bound *instruction-count* effects only. See §2.4 for which levers
that flatters and which it penalises. Full tables, sweeps and the de-confounding experiments:
implementation Task 3, "Milestone 0 — measured record".

| quantity | fit | span/T at T=40 |
|---|---|---|
| **baseline per-core span** | `1069 + 187.0·T` | **213.7 cyc/tile** |
| **no-DM-loop roofline** (all operands sharded, same knobs) | `772 + 45.25·T` | **64.6 cyc/tile** |

Those two are the gate's basis (§2.3.2); everything below is what the design has to assume about them.

**The headroom is the DM per-tile loop.** 141.75 of the 187.0 marginal (76%) is that loop, 45.25 (24%) is
DFB + Tensix machinery. The 45.25 is not a term added to a DM cost — it is a floor that becomes visible
only once the DM cost drops below it, which is why the design targets DM thread count and not the machinery.

**It is not "the DRAM path".** craq-sim performs the transfer as a host `memcpy` inside the issue
instruction and pre-satisfies barriers, so **bytes moved cost zero cycles** (research §10.4). Every
conclusion in this document that depends on transfer cost is therefore unmeasured here, not measured-small.

**Ring depth is not the explanation and not a lever.** Worth 1.8% across its whole range, asymptotic by
depth 4. It survives as an *enabler* for call batching (`capacity >= 2n`, §2.4), not as a win.

**DFB API calls cost ~120 cyc/tile of instruction time on the reader** — real, and present on hardware too.
Its 56% *share* of the baseline is a craq-sim artifact of the zero-cost transfer, so the defensible claim is
the absolute figure, never "data movement is not the bottleneck". On silicon it very likely is.

### 2.3 Gates

**Two kinds of gate, because craq-sim is trustworthy about exactly one of them.** Phase 1's acceptance gate
is *structural*; the cycle number is a **tripwire**, not the result. Rationale in §2.3.3.

#### 2.3.1 Structural — this is the phase-1 acceptance gate

Pass/fail, no timing model required, and collectively they are what "native" *means*. All but the suite run
are in the §7.1 per-run record.

| gate | pass condition | instrument |
|---|---|---|
| **Occupancy** | `QUASAR_DM*` RiscTypes carrying a **KERNEL** zone == `R+W` exactly, and distinct `QUASAR_NEO<i>` indices carrying one == `C` | device profiler |
| **Work is split, not duplicated** | `Σ over reader threads of RD_BAR == 9.00·T ± 1%` and `max/min over threads ≤ 1.05`; same on `WR_BAR` (13.00·T) | device profiler + `TT_METAL_PROFILER_SUM=1` |
| **Native factory engaged** | `generated/inspector/kernels.yaml` names exactly the 3 `kernels_qsr/` sources and zero `kernels_dfb/` ones | inspector (on by default, no env var) |
| **Not silently serialized** | `sem_stall` falls; `unpack_stall`/`pack_stall` stay 0 (§7.2) | perf trace |
| **Cycle floor** | ≥50% of headroom, §2.3.2 | device profiler |
| **Correctness** | exact oracle, §6.1 | — |
| **No-regression** | full suite matches with the env var **ON**, **unset**, *and* **=0**, §6.4 | — |

Three things that make these gates trip falsely if stated loosely:

- **Do not gate on a `1/R` slope — it is unsatisfiable by a correct implementation at *any* R.** Every role's
  fitted slope **is** the pipeline slope, verified in two independent configurations (baseline
  187.0/187.0/187.0/187.0 for reader/writer/math/span; instrumented 227.95/227.93/227.95/227.93). That is
  structural: at R>1 each thread covers `T/R` tiles but stays *live* for the whole window, blocked between its
  tiles, so its duration ≈ the span. Achievable reader-slope ratios are 0.699 at R=2,W=1 and 0.349 at
  R=4,W=2 — never `1/R`. Choosing R does not fix it: at R=2,W=2 the ratio is exactly 0.500, but only because
  `(R+W)/2 = 187/2`, i.e. it measures *joint* R+W scaling and would pass identically if the reader did nothing
  and the writer did everything.
- **`RD_BAR` is the quantity that works**, which is why the row above uses it: slope **exactly 9.00**,
  intercept **exactly 0**, no prologue, no blocking, emitted per reader thread. Total unchanged ⇒ no
  duplication (a duplicating implementation reports `R×`); equal across threads ⇒ no striding bug. It
  discriminates at R=4, where a slope gate cannot.
- **Count zones, not rows.** `QUASAR_DM0` always appears (its firmware zone — it is the ISR core), and
  `NEO0_TRISC3` emits a **16-cycle KERNEL zone while doing nothing**, so occupancy proves *launch*, not work:
  "N RiscTypes carry a KERNEL zone" is satisfied by N threads launching and one doing all the work. Filter on
  KERNEL zones and count **Neo indices**, not TRISC rows.

#### 2.3.2 Cycle count — a craq-sim gate, because nothing else catches silent serialization

**This is a craq-sim gate and it is valid in one direction only.** Because craq-sim is an *upper* bound for
the thread levers (no contention) it is a sound **stop** signal — failing here means failing on silicon too.
It is not a **go** signal: passing says nothing about the emulator, where §9.2's silicon-only ceilings apply.

Expressed as **fraction of measured headroom captured**, not a bare multiplier:

```
captured = (213.7 − achieved) / (213.7 − 64.6)      # nominal headroom = 149.1 cyc/tile
```

| marker | captured | cyc/tile @ 40/core | multiplier |
|---|---|---|---|
| **Floor — below this, suspect a bug** | **≥ 50%** | **≤ 139.1** | 1.54× |
| Sim ceiling | 100% | 64.6 | 3.31× — needs the DM loop's calls *and* instructions to vanish |

**One basis, one number: 50% of 149.175 ⇒ 139.1 cyc/tile ⇒ 1.54×.** State it once and derive the rest; a
gate quoted three ways drifts by more than the §7.3 sensitivity threshold calls significant.

**Gate on the fitted slope, with the span basis reported alongside.** The span-at-40 basis charges the
prologue at 1/40 weight, and at R=4 the prologue grows (4 producer threads, 3 DFBs, a `sync_threads` in the
drain) — ~1000 extra cycles of init is 25 cyc/tile at T=40 and would eat a third of the margin for a reason
that vanishes at T=80. The T=40/T=80 fit is already run, so this costs nothing.

**State which combination law the prediction assumes.** At depth 2 the stages measurably **add**
(reader-blocked ≈ writer-own and writer-blocked ≈ reader-own, each to 0.35 cycles), so additive is the
default; a `max` prediction must be justified by showing the chain decoupled. The two laws are 44% apart at
the target config, so an unlabelled prediction is not falsifiable.

**Read the floor as a bug detector, not a win detector — but it is a gate, not a tripwire.** Round 2
established why: occupancy proves only launch, and the `1/R` row is degenerate at R=4 (§2.3.1). A run where
all six threads launch, each covers `T/R` tiles, and the output is exact can still be serialized by credit
misattribution with the span never improving. The floor and the stall signature are the only two instruments
that see that. craq-sim has no NoC, DRAM-bank, or L1-port
contention model at all (§10.4), so splitting a pure instruction-cost loop across R cores scales *by
construction*. Clearing 50% therefore says little; **failing to clear it says a lot** — striding bug, threads
serialized behind one DFB, or the extra threads never launched. That is genuinely worth a gate, which is why
the floor stays.

**Neither bound is about DRAM.** In the sim DRAM is free (§2.2), so the floor is not "headroom being
overlapped" and the ceiling is not "DRAM isn't free". 64.6 is bounded because the reader's per-tile
instructions divide by R rather than vanishing, and the residual `45.25` does not divide at C=1 at all.

#### 2.3.3 Why the cycle number cannot be the success criterion here

Two reasons, both now measured rather than argued:

1. **No contention model ⇒ the thread-count magnitude is an upper bound, not an estimate.** Nothing arbitrates
   DRAM banks, NoC VCs (`get_vc_space` returns `0xffffffff`), or the 4-NoC-read-port L1 budget — which on
   silicon are exactly what bounds R. A clean scaling curve here is the model reporting its own assumptions.
   This is the honest reason magnitude belongs elsewhere; it is **not** that "the sim cannot price these
   levers".
2. **The floor is optimistic, so these are percentages of *nominal* headroom.** The 64.6 roofline's reader
   does not loop at all, so the true interleaved floor sits above it and real headroom is under 149.175. That
   direction is safe — a higher true floor shrinks the denominator and makes the same result capture a
   *larger* fraction — but it means "100% captured" is not a state of the world.

**No lever "registers as zero" — but a small craq-sim number can be a floor rather than a ceiling.**
`implicit_sync` moves craq-sim by at most `RD_BAR + WR_BAR = 22` of 228 instrumented cyc/tile (≤9.6%),
and that is a **floor**: barriers are pre-satisfied here so only their instruction cost is visible, while
on hardware a barrier is a real stall. The lever ranked #1 in research §9 therefore scores near-worst on
the sim, for a reason that does not apply on silicon. Never read a small sim delta as a small effect
without asking which direction the simulator errs in.

⇒ **The speedup claim belongs to the §7.5 emulator campaign.** craq-sim's job in phase 1 is to establish the
*shape* — occupancy achieved, work split evenly, instruction count down, output still exact — and to catch
bugs cheaply at 15 s a run. That division of labour is also what keeps emulator use to one campaign.

**~~64.6 is the C=1 floor~~ — SUPERSEDED by Milestone 1.** This section previously argued that compute at
C=1 does not pin the span, on the grounds that the Tensix is ~94% stalled and the roofline was measured at
C=1 knobs. The measured cost model is `max(165.0/R, 176.5/C, 83.5/W)`, so at C=1 the compute term alone
floors the span at **176.5 cyc/tile** no matter how many DM cores are added — the flat `C=1` wall in status
§5. A stalled engine is not a cheap one: it was stalled *waiting for work it then had to do serially*. The
gate's 149.1 headroom is only reachable with `C > 1`.

**Basis note — every "% captured" figure must name its estimator.** The gate above is on the
**span-at-40** basis: irreducible machinery contributes `(772 + 45.25×40)/40 = 64.55` cyc/tile, so the
reducible share of the 213.7 baseline is 149.1. On the **marginal** basis the same physical target is a
different number — 187.0 → 93.5 against an irreducible 45.25 is 66%, not 72%. Both are correct on their own basis; quoting one
against the other silently shifts the target. **Report the pair `(intercept, slope)` alongside the span basis so
either basis can be recomputed, and never mix them in one sentence.**

**How the result must be stated:** *"reduces per-core instruction count and raises engine occupancy,
measured as cycle deltas on a functional simulator with no NoC or DRAM timing model."* Not "removes
serialization" (unmeasurable here, §7.4) and never a bandwidth figure. Report the sharded roofline
alongside the headline so a reader can see which half of the number the simulator is trustworthy about.

Phase-1 end state is **experiment, not merge**: minimal gate, no CI wiring, hardening pass afterwards.

### 2.4 Perf expectation, and the kill criterion

**craq-sim and the emulator are different platforms and their numbers must never be added or compared.**
craq-sim is a functional simulator with no transfer latency and no contention; the emulator behaves like
silicon. Every knob therefore has two figures, and for three of the five they are biased in *opposite*
directions:

| lever | craq-sim | direction | emulator expectation |
|---|---|---|---|
| DFB call batching, reader n=2 | **1.08× measured** | **two-sided** | **Two mechanisms pulling opposite ways.** (a) It removes DM-core instructions — real on silicon, but only pays if the DM core is the bottleneck, so *sim is an upper bound* for that half. (b) It also leaves `n` NoC reads outstanding per barrier instead of 1 (measured: barrier cost falls as `1/n`) — that is latency hiding, invisible on craq-sim, so *sim is a lower bound* for that half. Net direction unknown |
| `entries_per_thread` (depth) | **1.02× measured** | sim is a **LOWER** bound | **potentially large, unmeasured.** Depth exists to hide transfer latency; craq-sim does the transfer as a host `memcpy` inside the issue instruction, so there is nothing to hide |
| `implicit_sync` | **≤1.10× measured** | sim is a **LOWER** bound | **potentially large, unmeasured.** On craq-sim a barrier is pre-satisfied, so only its instruction cost is visible; on silicon it is a real NoC round-trip stall, per tile |
| DM threads `R`, `W` | unmeasured | sim will be an **upper** bound | **≤ sim.** Bounded on silicon by the 4 NoC-read ports/Tensix, DRAM bank conflicts, and §9.2's two silicon-only ceilings (shared txn-id rendezvous, DM0's single ISR core) |
| compute threads `C` | unmeasured, blocked | sim will be an **upper** bound | ≤ sim, same reasons |

**The non-thread levers do not multiply — they compose to ~1.17× on craq-sim, not `1.08 × 1.10 × 1.02`.**
They attack the same DM per-tile cost and overlap: batching at n=2 halves the barrier count, so it has
already taken roughly half of what `implicit_sync` would remove. Composition arithmetic: implementation
Task 7.

**The consequence, and it is the most important thing on this page: the craq-sim ranking and the emulator
ranking are probably close to INVERSE.** On craq-sim, call batching is the best measured lever (1.08×) and
depth/implicit-sync are the worst (1.02×, ≤1.10×). On silicon the ordering plausibly flips — depth and implicit
sync are the *same* latency-hiding mechanism seen from two angles (§7.6: at `entries_per_thread = 2` in-flight
is 1, so implicit sync is *equivalent* to the explicit barrier — you need both to get outstanding
transactions), and that mechanism is the one craq-sim cannot model at all. Meanwhile batching's instruction
saving may simply disappear behind DRAM.

⇒ **Do not carry the craq-sim lever order (§7.4) into the emulator campaign (§7.5).** They are different
orders for different reasons, and the campaign must sweep depth and `implicit_sync` **together** — not because
they scored well on the sim, but precisely because they scored badly there for a reason that does not apply on
silicon.

**Sharper still: those are not three levers, they are one.** Depth, batch `n`, and `implicit_sync` are three
facets of *how many tile transfers are in flight at once* — depth provides the slots, `n` sets how many
transfers are issued before waiting, `implicit_sync` removes the wait. That is why all three measure ~nothing
on craq-sim: **one** shared reason, that in-flight concurrency cannot pay when a transfer costs zero cycles.
It also means they are not independent — `capacity >= 2n` is required for any overlap at all, so raising `n`
without raising depth *destroys* double buffering. **Sweep in-flight concurrency as one axis** (research §2.3).

**So the result rests entirely on thread count**, which was unmeasurable before implementation because
`num_threads > 1` is a host-side path. It has since been measured — **4.00× asymptotic at `R=4,C=4,W=2`,
the theoretical ceiling** — and the per-config evidence is status §5. What survives here is the shape of
the claim, not the number: it is **sim-basis** (the 213.7 baseline and the 64.6 floor both are), and
whatever threads deliver on craq-sim is an **upper bound for silicon**, since there is no contention model
and the 4 NoC-read ports per Tensix plus DRAM bank conflicts are what bound `R` on hardware.

**The one thread-scaling risk that was live, and how it resolved:** at depth 2 the two DM cores measurably
ping-pong (reader-blocked ≈ writer-own and vice versa, each to 0.35 cycles), and threads would have
disappointed had adding them not broken that credit chain. It broke — but only along `C`. The measured law
is `max(165.0/R, 176.5/C, 83.5/W)`, so **DM cores buy nothing until the Neos are there**, which is the
opposite of the DM-led framing this section was written under.

#### Kill criterion — apply it at step 1, before any follow-on work

> **If the R/W sweep delivers under ~1.3× on craq-sim (total under ~1.5×), stop and report that.** Do not
> proceed to Milestones 1.1-3.4 (§8). The criterion was cleared on 2026-08-27. The design's premise — that the idle engines are the win on this op shape — would be
> false, and the correct output of the project is that finding, cheaply obtained.

**The criterion is asymmetric, and only the stop direction is sound.** craq-sim models thread parallelism
faithfully but applies no contention, so it is an *upper* bound for `R`/`W`: **a sim failure is a real
failure** (silicon cannot beat an un-contended model), while **a sim pass proves nothing about silicon** —
§9.2's shared txn-id rendezvous and DM0 single-ISR serialization are invisible here and could erase the gain.
So: use it to stop early, never to declare success. Declaring success is the emulator campaign's job (§7.5).

This is deliberately the *first* milestone after the copy rather than the last, because it either validates the
whole approach or kills it in one experiment. It also reframes what this project is: not "build a 3.3× native
path", but **"determine whether multi-DM threading is worth anything on this op shape"** — a smaller, sharper,
answerable question, with every other lever now known-small rather than assumed-large.

## 3. Architecture

### 3.1 Placement — a third variant alternative

New TU `device/binary_ng_quasar_native_factory.cpp` holding `ProgramFactoryQuasarNative` with a single
`create_program_artifacts()` satisfying `ProgramSpecFactoryConcept`. The variant becomes
`variant<ProgramFactory, ProgramFactoryMetalV2, ProgramFactoryQuasarNative>`; `select_program_factory`
tries the native gate, then the metal_v2 gate, then the descriptor.

Verified additive: `AllFactoriesValid` folds over every alternative requiring exactly one factory concept
(`ttnn/api/ttnn/operation_concepts.hpp:174-189`, used `:208`), and a second `ProgramSpecFactoryConcept`
alternative gets its own adapter with distinct cached types. Nothing outside `select_program_factory`
inspects the variant.

Four mechanical constraints, each of which breaks the build or the op if missed:
1. **Append at index 2.** `program_factory_index` is persisted per cache entry and never serialized, so
   appending is safe; inserting at the front is not.
2. **The factory struct must be a stateless literal type** — the framework default-constructs every
   alternative into `static constexpr` storage (`device_operation.hpp:54`). `NativeTuning` is produced by a
   free function, never stored as a member.
3. **Add the `.cpp` to `ttnn/cpp/ttnn/operations/experimental/quasar/sources.cmake`** — sources are
   hand-listed, not globbed.
4. **Wrap file-local helpers in `CMAKE_UNIQUE_NAMESPACE` — with copy-then-modify (§3.4) this is a
   guaranteed build break, not a hazard.** The target is a unity build; `binary_ng_metal_v2_factory.cpp:85`
   uses a **bare** `namespace {` and defines `extract_nD_dims`, `get_shape_dims`,
   `calculate_compute_kernel_args`, `get_shards_per_width`. Copying that file wholesale yields a second
   bare anonymous namespace defining the same four names, merged into one TU by the unity build →
   redefinition error on the first compile.
   **Measured, by building the merged TU: 32 redefinitions, not 4.** The first 22 are the
   `constexpr const char*` kernel-path literals, then `DfbKernelSources`, `select_dfb_kernel_sources`, the
   four named helpers, `full_shard_tiles`, `make_dfb`, `create_no_bcast_artifacts`. Two consequences:
   **(a) "factor the four into `binary_ng_utils`" is not a fix** — it leaves 27 collisions standing; drop
   that option. **(b) The 32nd is a *link* error, not a compile error, and it does not depend on the unity
   build at all**: `create_program_artifacts` (`factory:1088-1091`) is an out-of-class member definition with
   external linkage, so a wholesale copy is a duplicate symbol even with `TT_UNITY_BUILDS=OFF`. **The class
   name must change.** Verified fix (32 → 0): rename the out-of-class definition's class to
   `ProgramFactoryQuasarNative` **and** wrap the *entire* anonymous-namespace body in
   `namespace CMAKE_UNIQUE_NAMESPACE` — the idiom `binary_ng_program_factory.cpp:19` already uses, and why
   the existing two TUs link today. Note the ODR half is *not* actually guaranteed: the current unity blob
   holds 8 files, so whether the copy lands in it depends on its position in `sources.cmake` — land it
   elsewhere and the ODR bug is latent (the link error is unconditional, so it still fails, just worse).
   Kernels need no `sources.cmake` entry: they are globbed (`quasar/CMakeLists.txt:19`
   `file(GLOB_RECURSE kernels */device/kernels*/*)`), though with no `CONFIGURE_DEPENDS`, so a new
   `kernels_qsr/` needs an explicit re-configure before the install `FILE_SET` sees it.

### 3.2 The gate — `matches_quasar_native_slice`

- `tensor_args.input_tensor_a.device()->arch() == tt::ARCH::QUASAR`.
  **Not `is_gen2_arch()`** — that is a file-local `inline` in `program_spec.cpp:166`, declared in no header.
  Not a host arch-name helper either: it returns `"invalid"` under the simulator.
- **`a.padded_shape() == b.padded_shape() == out.padded_shape()`, at full rank.**
  `SubtileBroadcastType::NONE` is **not** sufficient and is not what it sounds like:
  `get_subtile_broadcast_type` takes four scalars — H and W only
  (`binary_ng_device_operation.cpp:198-200`) — so leading-dim (N/C/D/nD) broadcast is `NONE`. Admitting it
  breaks two things at once: §4.3's collapse to `page_id = start + linear` is invalid because
  `next_c_shift`/`next_n_shift` are nonzero (`reader_no_bcast_dfb.cpp:88-98`), and the tile count below is
  read from the wrong tensor. The `no_bcast` suite cannot catch it — it passes one shape for *both* operands
  everywhere — so add a leading-dim-broadcast case asserting rejection.
- `SubtileBroadcastType::NONE`, tensor-tensor, no scalar
- `Layout::TILE` on both inputs and the output; 32×32 tile
- both operand dtypes `BFLOAT16`; `op_type == BinaryOpType::ADD`
- all three operands interleaved; no activations; not `is_where_op` / `is_quant_op`
- **`total_tiles % (num_cores * lcm(R, C, W)) == 0`** — W included. Omitting it admits hangs: `R=2, C=1,
  W=3` passes `lcm(R,C)=2` but `40/3` is not integral, and its `out` DFB is the documented-broken `2S×3S`.
  (Not `R=2,C=2,W=3` — that fails the `(C,W)` ratio rule first and never reaches divisibility.)
- **"larger divisible by smaller" on both endpoint pairs** `(R,C)` and `(C,W)`. The platform enforces it as
  two *directional* `TT_FATAL`s, not one symmetric rule: `num_consumers % num_producers` when `C ≥ R`
  (`impl/dataflow_buffer/dataflow_buffer.cpp:1267-1271`) and `num_producers % num_consumers` when `R ≥ C`
  (`:1278-1283`) — **and both are STRIDED-only; `AccessPattern::ALL` has no ratio constraint at all**
  (`:1246-1259`). Phase 1 is all-STRIDED so the symmetric summary is safe here, but do not carry it into the
  broadcast phases. The gate should reject rather than trip these. Separate hard ceilings, independent of the
  ratio: DM producers ≤ 6 (`:808-812`), Tensix consumers ∈ [1,4] (`:830-834`), and
  `MAX_NUM_TILE_COUNTERS_TO_RR = 6` (`hw/inc/internal/tt-2xx/dataflow_buffer/dataflow_buffer_config.h:63`).
  Consequence: **with C=4, R and W ∈ {1,2,4}, so R=4/W=2 is the only 4-Tensix config that saturates the
  6-DM budget.**

`num_cores` is computable here — `split_work_to_cores` is closed form:
`num_cores = min(total_tiles, attributes.worker_grid.num_cores())`, with `total_tiles` taken from the
**output** spec, not `input_tensor_a`: the factory uses `c.physical_volume() / tile_hw` (`factory:936`), and
under leading-dim broadcast `a.physical_volume() != c.physical_volume()`, so reading `a` gives wrong per-core
counts — a hang with compile-time counts. Derive from `compute_output_specs(...)` so gate and factory read
the same spec by construction; then the gate does not *depend* on the shape-equality condition holding.
One dependency to assert: this closed form matches `split_work_to_cores` only because the factory calls the
**`CoreRangeSet`** overload (`factory:944`, `work_split.cpp:405-427`) — the `CoreCoord` overload uses
`grid.x*grid.y` and diverges on a non-rectangular sub-device grid, which is a hang.
Verified to reproduce the ladder: 1280 tiles admits, 64 rejects.

### 3.3 Tuning knobs

```cpp
struct NativeTuning {
    bool     implicit_sync      = false;  // false = reserve/read/barrier/push (baseline-equivalent)
    uint32_t entries_per_thread = 2;      // PER-THREAD ring depth (see below)
    uint32_t reader_threads     = 1;      // R
    uint32_t compute_threads    = 1;      // C ∈ {1,2,4}
    uint32_t writer_threads     = 1;      // W, R + W <= 6
};
```

**Vocabulary, because conflating these two is what produced the stride-guard bug (§4.3).** There are two
different sets of roles and they must never be mixed:

- **Kernel roles — `R`, `C`, `W`**: the three thread counts. Reader, Compute, Writer. (Renamed from `P` on
  2026-08-20: `P` borrowed the DFB's "producer" vocabulary, but *compute is also a producer* — of `out` — so
  `P` was actively misleading.)
- **DFB endpoint roles — producers and consumers — which are PER DFB**: for `in0`/`in1` the pair is
  `(producers, consumers) = (R, C)`; for `out` it is `(C, W)`.

So a platform rule quoted generically — `num_entries % max(producers, consumers) == 0`,
`stride_in_entries = max(producers, consumers)`, "larger divisible by smaller" — instantiates to `max(R,C)` on
`in0`/`in1` and `max(C,W)` on `out`. **Always name the DFB when stating one.** Writing them as "max(R,C)" in
general is exactly how §4.3's guard came to say `n > 1 ⇒ max(R,C) == 1` and silently omit `out`.

**`entries_per_thread`, not `ring_depth`.** A DFB is a ring of `num_entries` tile slots, and depth is *how far
a producer may run ahead of its consumer before `reserve_back` blocks* — `capacity = 1` is lock-step,
`capacity = 2` is double buffering. But `capacity = num_entries / max(R,C)` is what reaches the hardware
credit register, not `num_entries`, so a global depth knob measures the wrong thing: depth 2 is *illegal* at R=4 (must divide
by `max(R,C)`), and depth 4 at R=C=4 yields capacity 1 — no per-thread double buffering at all. Derive
`num_entries = entries_per_thread * max(R,C)` for `in0`/`in1` and `* max(C,W)` for `out`. The point of the
per-thread form is that a *global* depth means different things at different thread counts, so a global sweep
varies two things at once and cannot be interpreted; the per-thread form holds each thread's buffering fixed
while thread counts vary. Full substrate detail in research §2.3.

Env vars, one per knob: `TTNN_QSR_NATIVE` (master, default off), `TTNN_QSR_IMPLICIT_SYNC`,
`TTNN_QSR_ENTRIES_PER_THREAD`, `TTNN_QSR_READER_THREADS`, `TTNN_QSR_COMPUTE_THREADS`,
`TTNN_QSR_WRITER_THREADS`. `select_program_factory` runs on **every** dispatch, so check arch and attributes
before `std::getenv` and cache the parse in a trivially-destructible `static const`.

**One process per configuration is a HARD rule, not a convenience.** On a cache hit the framework calls
`select_program_factory` and then resolves the factory from the **cached index**, discarding the answer
(`device_operation.hpp:268-271`, `map_index_to_variant` `:52-56`); factory identity is in neither the hash
nor the canonical key, and the cache is enabled unconditionally at device init. Flipping a knob in-process
silently re-runs the previous program. **Do not attempt an in-process comparison.** The escape hatch v2 offered is broken:
`disable_and_clear_program_cache()` *disables* the cache (`mesh_device.cpp:1075,1891`), so "assert the entry
count grew" can never pass — it would need a following `enable_program_cache()`. Deleting the hatch rather
than fixing it, because it invites exactly the mistake the one-process rule exists to prevent.

**Add `"worker_grid"` to `attribute_names`/`attribute_values`.** It feeds `split_work_to_cores` and hence
per-core tile counts, but is absent from the hash today. That is a latent wrong-answer hole for the existing
paths; with compile-time per-thread counts it becomes a **hang**. `CoreRangeSet` is already
reflection-hashable (`sub_core_grids` is in the list). Consider `sub_device_id` likewise.

Per-thread tile counts as compile-time args are safe — tensor shape **is** hashed for this op. Note the
fallback deliberately uses *runtime* args for the same counts, so this is a deliberate divergence.

### 3.3.1 The legal `(R,C,W)` space — by enforcing layer

Measured and sourced 2026-08-27. Separated by **what enforces each rule**, because only the first two
groups are op-independent; group C is ours and can be lifted, and group D is a defect, not a rule.

**A. Platform (`impl/metal2_host_api/program_spec.cpp`) — any kernel, any op**

| rule | source |
|---|---|
| `num_threads > 0` | `:752` |
| compute `C <= 4` | `QUASAR_TENSIX_ENGINES_PER_NODE = 4` (`:50`, `:757`) |
| **compute `C != 3`** — legal values are 1, 2, 4 | `:763` (explicit, with that wording) |
| DM `R <= 6` and `W <= 6` individually | `QUASAR_USER_DM_CORES_PER_NODE = 8 - 2 reserved` (`:779`) |
| `R + W <= 6` and `sum(C) <= 4` per cluster | work-unit budget check `:1743` |

`C != 3` is a **platform** rule, not ours. **It bans `C=3`, not `R=3`** — a distinction worth keeping,
because `R=3` survives into the legal set. The stride rule below needs `C in {1,3}` for `R=3`, so with 3
illegal only `C=1` remains, giving three legal configs (`3,1,1`, `3,1,2`, `3,1,3`). All three then fail
group D's `C >= max(R,W)`, so they are **legal but corrupt** — which is why the usable ladder jumps
R=2 -> R=4. `R=5` has the same shape (`5,1,1` only, corrupt); `R=6` is excluded outright, since any
`W >= 1` breaks `R+W <= 6`.

**B. Per DFB (`impl/dataflow_buffer/dataflow_buffer.cpp`, `hw/inc/api/kernel_thread_globals.h`)**

| rule | source |
|---|---|
| DM producers/consumers 1..6 | `MAX_PRODUCERS_PER_DFB = 6` (DM2-DM7; DM0 = ISR, DM1 = remapper) |
| Tensix producers/consumers 1..4 | `:803`, `:832` |
| STRIDED ratio: `max(producers, consumers) % min(...) == 0` | mirrors the two directional asserts |
| all KernelSpecs on the **same DFB side** share `num_threads` | `program_spec.cpp:1296` |
| at most **one producer-role and one consumer-role** multi-thread DM rendezvous group per cluster | `NUM_KERNEL_BARRIERS = 2` |

The last row constrains splitting `in0` and `in1` across two reader kernels — but **only when both are
multi-thread.** Two multi-thread producer-role groups collide on barrier slot 0; `wait_threads()` returns
immediately at `participants <= 1` (`kernel_thread_globals.h`), so a multi-thread + **single-thread** pair
never contends for a slot at all. That asymmetric shape is exactly what F14 (M2.5) wants.

**Host validation does not block the topology either** — verified 2026-08-28, and the header comment's
parenthetical ("host validation admits at most one same-role DFB instance per node") is narrower than it
reads. The one-producer-one-consumer-per-node check runs **inside a per-DFB loop**
(`program_spec.cpp:1372-1402`, `for (const auto& dfb : spec.dataflow_buffers)`), so it forbids two
producer *instances on the same DFB*, not two reader kernels each owning their own input DFB. Both
arrangements pass it.

Depth/entry caps are separate — see §4.4.

**What that leaves reachable, if you want per-operand reader kernels (F14 / M2.5).** Enumerated against
all four constraints — DM budget, barrier, STRIDED ratio, and the group-D corruption rule:

| in0 + in1 + W | C | verdict |
|---|---|---|
| **4 + 1 + 1** | 4 | **legal today** — the single-thread `in1` group never touches a barrier slot |
| 2 + 1 + 2 | 2 or 4 | legal today |
| 1 + 2 + 2 | 2 or 4 | legal today |
| **2 + 2 + 2** | 2 or 4 | **blocked by the barrier alone** — passes budget, stride and correctness |
| 3 + 2 + 1 | — | dead twice: barrier, *and* `in0`=3 forces `C=1` by the stride rule, which then corrupts |

Two things worth carrying:

1. **`3` is as dead in a split as it is single-reader.** `max(3,C) % min(3,C) == 0` admits only `C=1`, and
   `C=1` under 3 DM producers is exactly the group-D corruption case. Splitting the operands does not
   rescue an awkward core count, it just relocates it.
2. **The symmetric 2+2 split is blocked by one `constexpr`, not by the architecture.** Failure mode
   differs by shape: equal thread counts give a *spurious release* (`arrived` hits the shared target
   before both groups are in), mismatched counts give a genuine **hang** — the header's own stated
   motivation ("mixed counts never hit a target for some arrival orders").

**The fix is plumbing, not design** (assessed 2026-08-28, device side traced end-to-end):
`KernelBarrier` is two `uint32_t`, allocated as a plain `__attribute__((used))` global in
`hw/firmware/src/tt-2xx/dm.cc` with no fixed address, so widening the array costs nothing;
`sync_threads()` already takes the slot as a parameter; and the per-hart plumbing a group id would need
already exists in the same shape — `kernel_config.num_sw_threads[hartid]` / `kernel_thread_id[hartid]`
(`hostdev/dev_msgs.h`), read into thread-locals in `dmk.cc`. What actually blocks it is **ownership and
absence of a use case**: it is firmware plus a host/device shared-header ABI change touching every arch,
and host validation currently keeps the topology from arising, so the two-slot shortcut has never cost
anything. The host-side group assignment is **not** scoped. Sequence it after the emulator shows
operand-split helps (F14), not before — on craq-sim 2+2 measures identically to a single 4-thread reader,
since per-core read count is `T/2` either way.

**C. Ours, in `matches_quasar_native_slice` — liftable**

| rule | why | lift path |
|---|---|---|
| `total_tiles % (clusters * lcm(R,C,W)) == 0` | kernels use a strided share with **no tail handling** | F1 |
| `num_entries <= 255` | unguarded `uint8_t` threshold (§4.4) | field width; cap the sweep |
| `num_tiles_per_cycle == 1` unless every stride is 1 | chunk loop cannot express a batched strided reserve | with F1 |

**D. `R <= C and W <= C` is a DEFECT, not a rule.** Nothing above rejects `R > C` or `W > C`; those
configs are accepted by every validator and return **wrong data** (`R=2,C=1,W=1`: 589,234 / 1,310,720
elements differ). Consequence worth stating: a **more** DM-bound op is hurt more, since it is the
DM-heavy corner that breaks. For `add` it costs nothing only because compute-per-Neo happens to be
comparable to reader-per-DM-core — an `add`-specific accident, not a general result.

**Resulting legal space: 31 of 108 candidates — and the 108 already has `C in {1,2,4}` applied.** The raw
grid is `R,C,W in 1..6` = **216**; group A's compute rule removes half of it (`C=5,6` by
`QUASAR_TENSIX_ENGINES_PER_NODE`, `C=3` by `:763`), leaving the 108 that the rest of the ladder acts on:

| stage | rule | removed | left |
|---|---|---|---|
| raw grid | `R,C,W in 1..6` | — | 216 |
| A | `C in {1,2,4}` | 108 | **108** |
| A | `R + W <= 6` | 63 | 45 |
| B | in-DFB stride, `max(R,C) % min(R,C) == 0` | 8 | 37 |
| B | out-DFB stride, `max(C,W) % min(C,W) == 0` | 6 | **31** |

**Every count here is derived by `debug/attrib/enumerate_legal_space.py`**, which asserts them, so a
platform rule changing underneath this section fails loudly rather than silently rotting the numbers.

| `R` | legal | usable | the legal set |
|---|---|---|---|
| 1 | 11 | 6 | `1,1,1` `1,1,2` `1,1,3` `1,1,4` `1,1,5` `1,2,1` `1,2,2` `1,2,4` `1,4,1` `1,4,2` `1,4,4` |
| 2 | 10 | 5 | `2,1,1` `2,1,2` `2,1,3` `2,1,4` `2,2,1` `2,2,2` `2,2,4` `2,4,1` `2,4,2` `2,4,4` |
| 3 | **3** | **0** | `3,1,1` `3,1,2` `3,1,3` |
| 4 | 6 | 2 | `4,1,1` `4,1,2` `4,2,1` `4,2,2` `4,4,1` `4,4,2` |
| 5 | **1** | **0** | `5,1,1` |
| 6 | 0 | 0 | — (any `W >= 1` breaks `R+W <= 6`) |
| | **31** | **13** | |

**Why `R=3` and `R=5` are legal but never usable.** The stride rule needs `C` to divide `R` or `R` to
divide `C`, which for `C in {1,2,4}` leaves only `C=1`:

| | `C=1` | `C=2` | `C=4` |
|---|---|---|---|
| `R=3` | `3 % 1 = 0` ✅ | `3 % 2 = 1` ❌ | `4 % 3 = 1` ❌ |
| `R=5` | `5 % 1 = 0` ✅ | `5 % 2 = 1` ❌ | `5 % 4 = 1` ❌ |

`C=1` then fails group D (`C >= max(R,W)`), so all four survivors corrupt. **Keep "legal but unusable"
distinct from "illegal":** group D is a defect, so if it is fixed `3,1,1` and `5,1,1` become usable
immediately, whereas no `C=3` config ever can — one is a bug to be waited out, the other is the hardware.

**What `C != 3` actually costs:** `3,3,3` would be legal *and* clean, predicted at
`max(165/3, 176/3, 83/3) = 58.7` cyc/tile — a real intermediate tier between ~83 and 41.7, using 6 DM and
3 Neos. It is dominated by `4,4,2` (44.12 predicted and measured, on the same 6 DM cores plus one more
Neo), so the ban costs an operating point, not the optimum.

Exhaustive enumeration with measured cost per config is in status 08-27 slide 8.

### 3.4 Kernels — COPY-THEN-MODIFY, in a parallel tree

**Method (decided 2026-08-19, user call): the native factory and kernels begin as a *copy* of the v2
originals, not as new files.** The point is reviewability — with new-from-scratch files there is nothing to
diff against, so "what makes this native" is unanswerable by tooling; with a copy, `diff -r` answers it at
any point. It also makes Milestone 0 near-tautological (a byte-identical copy at baseline knob settings
*must* reproduce 8549, so any deviation is deliberate rather than something to hunt for) and protects the
§6.1 replicated-config contract by construction.

**Commit 1 is a mechanical copy plus exactly three mandated deviations — it cannot be byte-identical.**
Round 2 established why: the class rename and the `CMAKE_UNIQUE_NAMESPACE` wrap are forced by §3.1(4), and a
pure copy is not even *selectable*. So commit 1's true minimum is: copy + class rename + namespace wrap +
header variant alternative + `sources.cmake` entry + a gate hard-pinned to R=C=W=1 + a local
`static_assert(ProgramSpecFactoryConcept<ProgramFactoryQuasarNative>)`. **That set is Milestone 0's
prerequisite** — say it explicitly, or Milestone 0 silently becomes commit 4 and "reproduces 8549" stops
being tautological. The `static_assert` also localizes concept failures, which otherwise surface as
`DeviceOperationConcept` errors pointing at the *op* rather than the factory.

Every later commit is then a readable diff of what "Quasar-native" means. If commit 1 does not reproduce the
baseline, the copy is wrong, not the design. **Record `git rev-parse HEAD:<path>` for the factory and the
four kernels in commit 1's message**, so a later `diff -r` can be split into "ours" (`git diff <sha>`) versus
upstream's — `kernels_dfb/` is not frozen, and one upstream edit otherwise makes `diff -r` report *their*
change as *our* nativeness.

#### 3.4.1 Unmandated divergence: the reader/writer lost outer-dim broadcast (= F13, a regression)

**Outer-dim broadcast is baseline op behaviour, not a feature.** `matches_metal_v2_slice` gates only on
`subtile_broadcast_type` (`binary_ng_device_operation.cpp:505`), so the **shared `kernels_dfb/` path
already handles it** — the "no_bcast" kernels walk the OUTPUT dims while indexing each operand through
its own strides, which the factory zeroes for any unit input dim
(`binary_ng_quasar_native_factory.cpp:996-1003`). `no_bcast` means no *subtile* broadcast.

`kernels_qsr/` collapsed that cascade to `page = start_tile_id + k` (Task 4). That is a **fourth
divergence beyond the three mandated above, and it was not recorded as one.** It narrows the copy rather
than making it native. It is scheduled as **F13, opening the broadcast milestone** — a regression to
close, not a capability ranked by value. Sequencing it there is what makes "Milestone 3 = broadcast-
complete" literally true: outer-dim, subtile and mixed broadcast all land in the same milestone.

**Consequence today.** Correctness is safe — the native gate's full-rank shape-equality check
(`:716`) rejects these shapes, the fallback runs, results are bit-exact (verified). But **every
broadcast `add` gets zero benefit from the native path**, and leading-dim broadcast is common (bias add,
residual with a unit batch dim). So this caps the reachable win on real models, and it must be closed
before the native path can be considered for default-on.

**The fix, and why the obvious one is wrong.** Restoring the 6-deep odometer verbatim does not work: it
is inherently sequential, and each thread must advance by `num_threads`, not 1. Two viable shapes:

| approach | cost | note |
|---|---|---|
| decompose the global output index per tile (divmod by `tiles_per_nd`/`_d`/`_n`/`HtWt`) | 4 div + 4 mod + ~10 mul-add **per tile** | pure function of `g = start_tile_id + k`, so trivially per-thread. But at ~165 cyc/tile reader budget this is not free — measure before adopting |
| per-thread odometer with an advance-by-`N` carry | ~compares only | faster, materially fiddlier |

Either way, put it behind a **factory-set compile-time flag** (`HAS_OUTER_BCAST`, set when any stride is
zeroed) so the dense case keeps today's measured 165 cyc/tile untouched and only broadcast shapes pay.

**Guard until then:** `a_dims_are_output_dims` `TT_FATAL` in the factory, so widening the gate without
doing the kernel work is fatal at dispatch rather than silently wrong pages. Proven to fire.

Scope:
- **Factory: copy whole** (`binary_ng_metal_v2_factory.cpp`, 1092 lines →
  `binary_ng_quasar_native_factory.cpp`). The diff then shows the bcast/scalar/activation branches being
  deleted, which is informative, and deleting from a copy beats re-deriving the ~40% phase 1 needs. The
  dead branches are unreachable under the gate; hardening removes them. The 22 `constexpr const char*`
  kernel-path literals (`:91+`) get a mechanical rename to the new folder.
- **Kernels: copy only what phase 1 binds**, into `kernels_qsr/` with **filenames preserved** so `diff -r`
  lines up file-for-file: `dataflow/reader_no_bcast_dfb.cpp` (149), `dataflow/writer_no_bcast_dfb.cpp` (96),
  `compute/eltwise_binary_no_bcast_dfb.cpp` (105), plus `compute/eltwise_utils_dfb.hpp`.
  **Not** the other 19 of `kernels_dfb/`'s 22 files (3865 lines total) — those are bcast/scalar/SFPU
  variants the gate rejects, and copying them creates dead duplicates that **rot silently** when the
  originals are fixed upstream. Each later phase copies its own kernels *when it starts*, from a current
  original rather than a stale one. `diff -r` reporting "only in kernels_dfb" for the untouched 19 is
  honest noise.

**This deliberately overrides the review's argument against a fourth kernel tree.** That objection was
about long-term structure for a future reader; phase 1's end state is explicitly *experiment, not merge*,
and near-term diffability is worth more now. **Hardening must revisit it** — the collapse-back options are
folding the native kernels into `kernels_dfb/` behind `#define`s (the existing idiom there, 13 sibling
compute variants already coexist) or keeping the tree and documenting the paradigm boundary.

**Not a JIT-default-include question — and the prescribed rename would break it.** `include_paths` is
*already* the mechanism: `factory:874` passes `{kComputeIncludeCommon, kComputeIncludeDfb}`, defined at
`:164-165` and `:166-167`, with the intent stated at `:162-163`. Of the 22 path literals, 20 are kernel
sources and 2 are include directories — and `kComputeIncludeCommon` points at a **third** tree
(`kernels/compute`), home of the `eltwise_utils_common.hpp` that §3.4 deliberately does not copy. So a
blanket rename of "the 22 literals" retargets it at `kernels_qsr/compute`, where that header does not exist,
and the first JIT compile fails. **Rename the 20 sources and `kComputeIncludeDfb`;
`kComputeIncludeCommon` must keep pointing at `kernels/compute`.**

That asymmetry is also the load-bearing hole in the Milestone-0 tautology: leaving `kComputeIncludeDfb`
pointed at the old tree looks correct by analogy, and would make Milestone 0 reproduce 8549 exactly while
compiling against the **original** headers — a reproducibility failure with a green signal. Guard it: assert
on the resolved include set, or temporarily `#error` the old `kernels_dfb/compute/eltwise_utils_dfb.hpp` and
confirm the native build still compiles.

**The kernels are NOT thread-generic today — this is work, not an inherited property.** All three files
contain zero references to `get_my_thread_id()`/`get_num_threads()` and do not include
`api/kernel_thread_globals.h`. The *APIs* are available on both DM and TRISC as claimed
(`kernel_thread_globals.h:55-77`, outside the `#ifndef COMPILE_FOR_TRISC` block at `:79`; `my_thread_id` is
`thread_local` per-Neo via `trisck.cc:32-33,92-94` — which is the same fix pattern tt-llk #1678 asks for, so
they are *not* the same bug). Consequence: **a faithful copy hangs at R>1** — the host declares 4 producers
so `capacity = num_entries/4`, while all four threads run the full T-tile loop, posting 4× the credits. Copy
the loop from `tests/tt_metal/tt_metal/test_kernels/dataflow/dfb_producer_2_0.cpp:27-32,47` verbatim,
including its `break` guard and its `dfb.finish()`; it takes the count from `get_num_threads()` rather than a
CTA, so no host/kernel mismatch is possible and the uneven-count follow-on needs no kernel change. The two DM kernels are additionally **sync-generic** (`#if IMPLICIT_SYNC`). The compute kernel is
**always explicit** — there is no compute-side implicit-sync opt-out, and `producer_is_tensix_only`
short-circuits producer txn allocation so compute's side of `out` is structurally explicit anyway.

Note one API limitation that bites the uneven-work follow-on: `sync_threads()`/`wait_threads()` are
`#ifndef COMPILE_FOR_TRISC` (**DM only**), so compute threads have no software rendezvous — per-thread
counts must be derived in-kernel from `get_my_thread_id()`.

## 4. Dataflow

### 4.1 DFB topology (at target knobs R=4, C=4, W=2)

| DFB | producer | consumer | `num_entries` |
|---|---|---|---|
| `in0` | reader, 4 threads, STRIDED | compute, 4 threads, STRIDED | `entries_per_thread × 4` |
| `in1` | reader, 4 threads, STRIDED | compute, 4 threads, STRIDED | `entries_per_thread × 4` |
| `out` | compute, 4 threads, STRIDED | writer, 2 threads, STRIDED | `entries_per_thread × 4` |

6 DM threads + 4 Tensix engines in one `WorkUnitSpec` — the full budget. `4S×4S` and `4S×2S` are both in
the passing DFB matrix.

**~~Phase 1 reaches R=4, C=1, W=2 — the full DM budget — without the tt-llk fix.~~ SUPERSEDED — that
config is one of the 18 corrupt ones.** The legality argument still holds (`max(R,C) % min(R,C)` is
`4%1`, `(C,W)` is `2%1`) and `DMTensixTest1xDFB4Sx1S` does exist and pass
(`test_dataflow_buffer_base.cpp:53`) — but `4,1,2` measures **FAIL 73.3%** here, and that gtest cannot
contradict it: no DM→Tensix test verifies delivered data (`dfb_test_common.hpp:539`). **A passing
upstream gtest for an endpoint shape is not evidence that shape delivers correct bytes.** The rule is
`R <= C and W <= C`, so the DM budget is only spendable once the Neos are there; the target topology
above is the *only* full-budget topology that works.

#### 4.1.1 Why STRIDED on every endpoint — and where ALL would belong instead

`AccessPattern` is `{STRIDED, ALL, BLOCKED}` (`kernel_spec.hpp:138`), but for an elementwise op only one
of the three is a real option.

- **STRIDED** — thread *t* takes every *N*-th entry. This is the work split, so it is the whole point.
- **ALL** — every consumer thread reads *every* entry, i.e. replication. For elementwise that means all
  `C` Neos computing the same output tile: `C`x redundant work, not a split. Wrong by construction.
- **BLOCKED** — no API exists. `BlockedConsumerOf` is commented out (`kernel_spec.hpp:289-300`,
  "Uncomment when BLOCKED support is added (currently TT_FATALs)"). Unreachable, not merely untested.

Two structural facts narrow it further. **Producers can only ever be STRIDED** — `ProducerOf` hardcodes
it and the header states "All DFB producers are STRIDED" (`:245-252`), so the pattern is a consumer-side
decision only. And **the choice is per binding, not per DFB**, which is why the gtest matrix is indexed
`nS x m{S|A}` with the left letter invariant.

**Where ALL would genuinely fit:** when consumers need *the same* entry rather than different ones. In
this op's family that is **tensor-scalar** (F10, milestone 3.x). Today the writer fills `in1` once into
the buffer so there is no stream to replicate; if `in1` instead became a one-entry DFB read by all `C`
Neos, `AllConsumerOf` is exactly that pattern. **Broadcast is not such a case** — subtile and outer-dim
broadcast both stay STRIDED, because the broadcast operand is re-read per output tile through index math
(repeated page ids). Broadcasting the *tensor* is not broadcasting the *DFB entries*.

**Consequence for the legal space.** STRIDED is what partitions the ring into sub-rings, so it is what
imposes `max(p,c) % min(p,c) == 0` (§3.3.1); ALL partitions nothing and carries no such rule — which is
why `6Sx4A` exists upstream and `6Sx4S` does not. An elementwise op therefore cannot buy its way out of
the divisibility constraint by switching pattern, because the pattern *is* the work split. That rule
costs 8 + 6 of the 45 budget-legal configs.

Style note: the factory binds consumers with `ConsumerOf`, whose doc comment says to prefer
`StridedConsumerOf` for multi-threaded kernels. Identical behaviour (the default is STRIDED); the
explicit spelling states the intent.

### 4.2 The pairing invariant — guaranteed, with one condition to assert

Compute consumes `in0[k]` and `in1[k]` as one output tile. This is **guaranteed by construction**, not
assumed: producer slot bases are assigned in ascending risc-id order, and
`TT_FATAL(std::is_sorted(dm_processors_...))` plus `kernel_thread_id[processor_index] = thread_idx` make
thread *t* ⇒ producer index *t*. With identical `(R,C)` both DFBs get identical `capacity`,
`stride_in_entries` and `num_tcs_to_rr`, so `tc_idx` and `wr_ptr` advance in lockstep.

**The one condition nothing else enforces: `in0` and `in1` must be constructed with the same `entry_size`
AND the same `num_entries`, with no per-DFB `DFBRunOverrides`.** Assert it in the factory.

### 4.3 Loop shapes

The narrow gate collapses the existing five-deep nD stride cascade to `page_id = start_tile_id + linear`:

```
reader thread t of R:   for j in 0 .. (T/R - 1):  page = start + j*R + t
                        implicit: noc.async_read<NocOptions::TXN_ID>(acc, dfb, {.page_id=page}, {})  x2
                        explicit: reserve_back(1) x2; async_read x2; async_read_barrier(); push_back(1) x2
compute thread of C:    for j in 0 .. (T/C - 1):  existing per-tile body, num_tiles_per_cycle = 1
writer thread w of W:   for j in 0 .. (T/W - 1):  page = start + j*W + w
every DM thread, last:  dfb.finish() on each bound DFB
```

**`dfb.finish()` is not optional, and it is why even divisibility is a *safety* property rather than a
simplification.** On the implicit arm credits arrive only from the ISR at `threshold` granularity, so a tail
batch shorter than `per_txn` is never posted → hang; the platform's reference producer calls it
unconditionally (`test_kernels/dataflow/dfb_producer_2_0.cpp:47`). But `finish_impl` →
`handle_final_credits` does an **unconditional** `sync_threads(is_producer ? 0 : 1)`
(`dataflow_buffer.inl:390`), reached under `if (ptiles_read_ > 0)` / `if (ctiles_written_ > 0)`
(`:255,258`) — so **a thread with zero tiles skips the barrier while its siblings block: hard deadlock.**
Even divisibility guarantees every thread gets ≥1 tile. The uneven-count follow-on must therefore clamp
thread count to the work available, not merely handle a remainder.

Two further constraints this pulls in. **Barrier slots are a budget of 2, keyed by role**
(`NUM_KERNEL_BARRIERS = 2`, `kernel_thread_globals.h:40-41`): reader (producer, slot 0) + writer (consumer,
slot 1) fits, but **two producer-role multi-thread DM groups on one node share slot 0 and deadlock** — which
rules out splitting `in0`/`in1` across two reader kernels (DFB-legal, looks like better DM utilisation, and
wrong) and is a live hazard at F10, where the writer becomes an `in1` producer (`factory:797`). Platform
validation is per *DFB* (`program_spec.cpp:1250-1262`) and will not catch it. And whether the **explicit**
multi-thread arm also needs `finish()` is genuinely unresolved — no production Quasar DFB kernel calls it
today, and `finish_impl` still runs an unconditional `all_acked` drain spin (`:262-266`). Settle it at
Milestone 0 by experiment: R=2, with and without, compare cycles and hang.

`page = start + k·W + tid` was verified algebraically to be a real invariant of the STRIDED mapping for
R≠C≠W (both `C ≥ W` and `C < W`), not only at equal thread counts.

**`num_tiles_per_cycle` stays 1, and this is now a hard constraint, not a simplification.**
`stride_in_entries = max(R,C)`, but the kernel's intra-chunk tile indices step by **1** (unpack
`rd_entry_idx + tile_index`, pack `wr_entry_idx + wr_entry_ptr++`) while the slot advance is
`num_tiles × stride_size_tiles`. At `n > 1` with `stride > 1`, tiles 1..n-1 are read from and written into
**other threads' slots** — silent corruption, no hang. It works today only because everything is
single-threaded. `stride_in_entries` is computed **per DFB** from that DFB's own endpoints (`dataflow_buffer.cpp:1140`), so
`out`'s stride is `max(C,W)` — the guard must therefore be **`n > 1 ⇒ max(R,C) == 1 AND max(C,W) == 1`**.
Naming only `max(R,C)` admits `R=1, C=1, W=2, n=8`, which packs tiles 1..7 into writer-thread-1's slots; that
is F3 (sharded, the phase that wants `n=8`) crossed with any `W>1`. Alternatively pass a `STRIDE_TILES` CTA
and index `i * STRIDE_TILES` — but that requires the **`out_of_order_output = true`** pack template
(`pack.h:88-89`); the default `false` overload *ignores* `output_tile_index` (`llk_pack_tile_api.h:66-72`). (DST capacity is not the limit: bf16 `DEST_NUM_TILES_FP16_HALF == 8`.)

### 4.4 Depth limits

A bf16 tile entry is 2 KB; three DFBs at `entries_per_thread`=32, R=C=4 is 32×4×3×2 KB = 768 KB of the
4 MB pool **minus the reserved region** (subtract it before quoting a budget). The real caps:

| cap | value | where |
|---|---|---|
| `capacity = num_entries / max(R,C)` | ≤ 65535 | `dataflow_buffer.cpp:1156` |
| **`threshold = num_entries / num_txn_ids`** | **≤ 255, and `per_txn ≥ 1`** | **unguarded — see below** |
| `ring_bytes` | ≤ unreserved L1 | `:1223-1229` |
| `num_entries % max(R,C)` | == 0 (per DFB; `max(C,W)` for `out`) | `:1133` |


**The one silent cliff, and it is twice as close as v2 thought.** `threshold` and
`num_entries_per_txn_id` are `uint8_t` with **no guard** (`:1047-1071`) — the only `TT_FATAL` there checks
divisibility, which `0 % anything == 0` passes. v2 put the cliff at 512 by assuming `num_txn_ids >= 2`, but
`compute_optimal_txn_id_count` **falls back to 1** when no `n ∈ [2,4]` satisfies
`num_entries % (n · prods_or_cons · tcs_per_risc) == 0` (`:1092-1106`). **So the cliff is at
`num_entries > 255`.** `threshold → 0`, and on device `x % 0` returns `x` on RISC-V, so the txn index never
rotates. Add the two `TT_FATAL`s and cap the sweep at 255 total entries.

Two more corrections to the table above: `stride_in_entries <= 255` (`:1203`) is **dead code** — it is
`max()` of two `uint8_t` fields (`dataflow_buffer.hpp:41,44`) — so it is removed rather than listed. And a
**stricter** divisibility rule is missing from it:
`num_entries % (num_txn_ids · prods_or_cons · tcs_per_risc) == 0` (`:1040-1044`), tighter than the
`% max(R,C)` row and the one that bites at `entries_per_thread == 1`.

## 5. Failure modes and defences

Every serious failure is a **hang, not a wrong answer**, so the defences aim at making credit imbalance
unrepresentable.

1. **One count formula, computed once on the host**, passed as compile-time args — *and* `worker_grid` in the
   hash (§3.3), without which a cache hit can reuse counts computed for a different grid. The formula alone
   is not sufficient; the hash is what makes it hold.
2. **Sync mode is atomic per DFB, chosen at compile time** via `#if`, with
   `DataMovementGen2Config::disable_dfb_implicit_sync_for_all` set consistently. Mixing double-counts the
   16-bit counter → `TILE_COUNTERS` fault, `mtval 0x1`. The existing fallback already passes
   `disable_dfb_implicit_sync_for_all = true`, so the A/B reference has no latent double-count.
3. **Host `TT_FATAL`s — and two of the six v2 called "new" are not.** The platform already enforces the
   divisibility, `capacity`, alignment and `entry_size`/`stride_size` checks in
   `impl/dataflow_buffer/dataflow_buffer.cpp`, **and also `C ∈ {1,2,4}`** (`program_spec.cpp:756-768`,
   whose message literally says "Legal values are 1, 2, and 4") **and `R + W <= 6`**
   (`:1737-1755`, summed across DM kernels). Do not re-implement those two: the platform messages are
   adequate, and `QUASAR_USER_DM_CORES_PER_NODE = 8 - 2` is a file-local `static constexpr` (`:46-48`) with
   a TODO saying it *should* be public — so an op-side check would have to hardcode 6, which the project
   rule on arch constants forbids. Hardcode it only in the sweep clamp. Genuinely new, and worth adding:
   total L1 budget; `threshold ≤ 255` and `per_txn ≥ 1` (§4.4); `in0`/`in1` same `entry_size` and
   `num_entries` (§4.2); `num_tiles_per_cycle > 1 ⇒ max(R,C) == 1 AND max(C,W) == 1` (§4.3 — per DFB); and
   — once DFB call batching is wired — **`capacity >= 2n` per DFB**, i.e. `entries_per_thread >= 2n`. At
   `capacity == n` a producer reserves its whole allocation and the ring degenerates to lock-step at batch
   granularity: no hang, no wrong answer, just a silent loss of all double buffering, which is exactly the
   confound that would make a batch sweep uninterpretable (research §2.3).
   Gate-level *rejection* of the platform-enforced pair is still worth it — a clean `false` beats a throw.
   **Put these in `validate_on_program_cache_miss`, not in `create_program_artifacts`** — the latter runs
   only on a miss, while `validate_*` is called unconditionally even under fast-runtime mode.
4. **Bring-up tooling, with one caveat.** `TT_METAL_WATCHER=10`; `TT_METAL_LLK_ASSERTS=1` **only at C=1** —
   `llk_tdma_guard::armed_mask()` is a function-local `static` shared across the 4 Neos, so at C>1 it
   produces both false negatives and false positives. Also: the real TEN-4746 rule is **same-DFB
   WAIT→retire**, not "no two counter ops back-to-back" — three consecutive counter ops on three *different*
   DFBs are legal. The existing loop satisfies it and the planned loop inherits that; do not add interposed
   dummy copies to satisfy the stricter reading.
5. **Enforce the hang timeout.** A healthy run is 12–15 s, but wall clock tracks *simulated instruction
   count*, so R=4/C=4 plus watcher can legitimately exceed a couple of minutes. Use
   `--timeout=600 --timeout-method=thread` (the `signal` default will not unwedge a device-side spin) or
   `timeout -s KILL`; record a per-configuration budget per ladder row; discriminate hang from slow via the
   sim's global clock (advancing = slow, stalled = hang) and the last `generated/watcher/watcher.log` dump.
6. **Not applicable in phase 1:** the `#51291` release-fence gap — the interleaved no-broadcast path does no
   software fill. It becomes load-bearing at F8 and F10.

## 6. Correctness

**Where these live in phase 1 — reversed in round 2.** The oracle, the routing assertion and the §6.6 cases
are **one real pytest file** from the start, `tests/ttnn/nightly/.../test_binary_ng_quasar_native.py`; only
the sweep/bench harness stays in `debug/`. Three reasons the earlier `debug/`-only call was wrong: the
DM→Tensix DFB tests verify **no data** (`dfb_test_common.hpp:539-540`,
`test_kernels/compute/dfb_t6_consumer_2_0.cpp:21` — the consumer `copy_tile`s and discards), so **nothing in
the tree data-verifies a multi-thread STRIDED producer writing its own slots** and this oracle is the first —
that is a missing regression test, not an experiment; "experiment, not merge" argues against **CI wiring**,
not file location, and an unreferenced file under `tests/` is equally un-merged while getting the `device`
fixture, parametrization and XML nodeids that §6.4 explicitly wants; and the `debug/` pytest trap is a
*deletable stale file* — `debug/conftest.py` is a gitignored 46 KB copy of the root conftest whose
`pytest_addoption` re-registers `--tt-arch` (`:858` vs root `conftest.py:949`), so rename it rather than
designing around it. The §6.4 no-regression run invokes the existing suites in place.

### 6.1 Primary oracle — single-run, exact

bf16 DRAM-interleaved `add` on the current fallback is **bit-exact against a torch bf16 golden** (measured
twice: 0/65536 and 0/1310720 mismatching elements, `max_raw_bit_delta = 0`), because `fp32_dest_acc_en` is
false for bf16 and `HiFi4` is fidelity-inert for FPU add. So:

```
torch.equal(out.view(int16), (a_dev.float() + b_dev.float()).to(bfloat16).view(int16))
```

with `a_dev`/`b_dev` read back from device. One run, no second process, no env flip — and strictly stronger
than comparing against the fallback, since it also catches a bug both factories share. Use the `int16` view
so `-0.0`/NaN cannot alias.

**Replicated-config contract** (the oracle depends on it, so assert it): `MathFidelity::HiFi4`,
`math_approx_mode=false`, `fp32_dest_acc_en=false`, `double_buffer_dest`, per-DFB `unpack_modes`, and the
three DFB `data_format`/`tile_format` values must match the fallback exactly.

**The oracle expires at F5** (`UnpackToDest` changes the precision path) **and F6** (MX). F2
`multiply` is fidelity-dependent under HiFi4 mantissa splitting — copy the config verbatim or the oracle
self-destructs.

### 6.2 Routing assertion — without it, a green run proves nothing

**Primary instrument: `generated/inspector/kernels.yaml`, which costs nothing.** The inspector is on by
default with no env var and writes the bound kernel sources every run; a probe run emitted exactly the three
files phase 1 binds. So `grep -c 'kernels_qsr/' == 3 && grep -c 'kernels_dfb/' == 0` is an exact positive
proof **and** a negative control (`rm` the file before each run). Two advantages over the alternative: it
works at **Milestone 0**, where the RiscType-set discriminator is blind by construction (that only separates
at `R+W>2`), and it pins the *bound kernel identity*, which is what §6.1's config contract otherwise cannot
assert.

Optionally also bind `BinaryNgDeviceOperation::select_program_factory` to nanobind and assert the variant
index is 2, with a negative control (one knob nudged so divisibility fails → index 1). **This is not the
~6-line freebie v2 claimed:** binary_ng has no nanobind file at all (`sources.cmake`: "device backend; no
host op / no nanobind"), so it means adding to `binary/binary_nanobind.cpp` and returning
`program_factory.index()` as an `int` rather than the variant. The quasar matmul precedent is real
(`matmul_nanobind.cpp:1251-1256`) but no test uses it, so that pattern is itself unexercised.

Cache-entry counting cannot substitute, and the tree proves it:
`test_binary_ng_descriptor_cache_hit.py:16-18` still asserts interleaved falls through to the descriptor,
which is **false on this HEAD** — that file's coverage is already silently gone while still passing.

### 6.3 Determinism and a positive control

Establish **native-vs-native** bit-identity *before* any cross-factory comparison, and re-establish it at
every step that raises a thread count — 4 DM × 4 Tensix threads is a new nondeterminism surface, and if
repeat runs differ then every oracle above is meaningless. Add a one-time **positive control** (deliberately
off-by-one on `in1`'s `page_id`) to prove the oracle has teeth.

### 6.4 No-regression — the ON direction is the load-bearing one

With the var **off** the gate returns false by construction, so that run only catches a build break, which
`AllFactoriesValid` catches at compile time. Everything that can actually go wrong — the gate
over-capturing fp32, `subtract`, `post_relu=True`, sharded, or 2-tiles-per-core — is reachable only with the
var **ON**.

Re-baseline first: the quasar binary_ng suites collect **257** on this HEAD — no_bcast 88 + bcast 130 + scalar 25 = 243, **plus
`test_binary_ng_descriptor_cache_hit.py` 4 and `test_binary_ng_resnet_add.py` 10**, with zero skip/xfail
markers anywhere. Include both omitted files: the cache_hit ones are the only coverage of a program-cache hit
with reallocated buffers, which is precisely the risk §3.3 creates. Note what the ON run actually is —
5 of 257 are admitted at default knobs (4 of them in cache_hit), and at **target** knobs `lcm = 4` drops
those, leaving **exactly one** admitted test, the perf benchmark itself. So the ON run is a 256-case
*gate-rejection* test, not a native-factory test. Record per-file counts and diff the `most_recent_tests.xml` nodeid→outcome map, not the summary
line.

### 6.5 PCC is a smoke gate, not the correctness gate

The inherited `test_no_bcast_interleaved[post_relu=False-bf16-add]` genuinely covers the admitted slice, but
its bf16 threshold of 0.997 tolerates **3 to 7 wrong tiles out of 1280 depending on the bug's shape**, and
the `golden_std` guard only catches a near-constant output (measured 10× headroom). The tolerance is not one
number: a skipped tile follows `√(1−f)` → 7 tiles, but the **canonical striding bug** — `out[k] = a[k']+b[k']`
— follows `1−f` → **3 tiles**. And the conclusion v2 drew was inverted: a *systematic* per-thread tail bug
(32 cores × 4 threads = 128 tiles) gives PCC **0.90–0.95** and is caught with enormous margin. **The real
blind spot is 1–7 tiles** — one wrap slot, or a one-tile boundary error on ≤3 cores.

### 6.6 Correctness cases inside the gate

Not perf rungs — run at every knob step:
- **T/thread == 1.** 128 total tiles is *admitted* at 4 tiles/core, so at R=C=4 every thread runs one
  iteration: the ring never wraps and the drain is immediate. The ladder's smallest rung is 8, so this is
  currently untested.
- **`entries_per_thread == 1`** — the tightest credit accounting, most likely to expose an off-by-one.
- **`entries_per_thread` ≫ tiles-per-thread** — a green deep-ring run proves *nothing* about wrap-around.

- **R ≠ W — the phase-1 *target* topology, and the case most likely to be wrong.** R=4/C=1/W=2 has three
  distinct thread counts (`in0`/`in1` stride 4, `out` stride 2), and §4.3's `page = start + k·W + tid`
  invariant is verified **algebraically, not by a run**. None of the cases above varies R against W.
- **Non-divisor wrap** — e.g. `entries_per_thread=3` at 10 tiles/thread. The cases above cover depth 1 (no
  slack) and depth ≫ T (never wraps) but nothing where the ring wraps a *non-integral* number of times,
  which is where `tc_idx`/`wr_ptr` rotation and the `threshold` modulo actually break.
- **A native-factory program-cache hit** — nothing else dispatches the native factory twice, and §3.3 moves
  per-thread counts to compile-time args, which is exactly what a stale hit would get wrong.
- **A second shape**, one where `T/R ≠ 10`. Otherwise nothing distinguishes "correct" from "correct at 1280".

Previously dismissed, now reinstated: **`R=1,C=4` and `R=4,C=1` are NOT covered by the DFB matrix** — the
DM→Tensix tests verify no data at all (`dfb_test_common.hpp:539-540`,
`test_kernels/compute/dfb_t6_consumer_2_0.cpp:21`), so the matrix proves liveness only. Still correctly
dismissed: "tiles-per-core divisible but total not divisible by `num_cores`" cannot occur under the gate
(`work_split.cpp:405-427`), **provided** the factory keeps calling the `CoreRangeSet` overload — the
`CoreCoord` one uses `grid.x*grid.y` and diverges on a non-rectangular sub-device grid, which is a hang.

## 7. Measurement protocol

Fixed shape **32×40 tiles (1280 total, 40/core)**, confirmation at 80/core. **One process per
configuration** (§3.3).

**Pin the simulator's cycle model in every run, or A/B numbers silently cross cycle models.** At least
twelve env vars change craq-sim's cycle counts or its schedule, and they are latched in `static const` — so
they are per-process, and a stale `export` silently applies to a whole sweep. Beyond the three obvious ones
(`..._TENSIX_RTL_AWARE_SCHEDULER`, default 1; `..._TENSIX_PIPE_ISSUE_BUDGET`;
`..._PARALLEL_TENSIX_TILE_CLOCK`) the list includes **`TT_METAL_SIMULATOR_TENSIX_DEFAULT_LINGER`, default
ON**, which adds 1-4 wait cycles per TRISC-sync resource conflict and whose own comment says it *dominates*
for workloads with frequent cross-TRISC synchronisation — i.e. exactly the C=4 regime step 3 targets (verified
cycle-neutral at C=1, so it is a C>1 risk); plus `..._TENSIX_MATH_ISSUE_GAP`, `..._TENSIX_SFPU_ISSUE_GAP`,
`..._TENSIX_FRONTEND_FIFO_THRESHOLD`, `..._CQ_WAIT_CLOCKS`, `..._PARALLEL_CHIP_CLOCK`,
**`..._PARALLEL_CLOCK_THREADS` (default derives from `hardware_concurrency()`, i.e. host-CPU-dependent)**,
`TTSIM_QSR_SUBTILE_AUTOPOST_FIX` (default ON; inert now, live at the broadcast phases) and
`TT_METAL_SIMULATOR_DIRECT_TENSOR_WRITES`. **Dump `env | grep -E 'TT_METAL_SIMULATOR|TTSIM'` into every
per-run record** rather than maintaining a list by hand, and re-verify determinism at C>1.

### 7.1 Per-run record

- headline metric (§2.1) + `(intercept, slope)` from T=40/T=80
- **sorted set of RiscTypes present** — a direct pass/fail on each knob step; this is how the 2-of-6-DM /
  1-of-4-NEO baseline was established
- **the full `env | grep -E 'TT_METAL_SIMULATOR|TTSIM'` dump** (see above)
- **`run host ID` from the profiler CSV, asserted to be a single value** — the CSV has no dispatch key, so two
  dispatches in one process leave a *per-core* blend and the median silently reports whichever dispatch touched
  more cores (measured: a 24/8 core split; it flips at 17/15). Delete the CSV before each run and check its
  mtime afterwards, or a run that dies before the profiler flush reports the previous run's numbers.
- per-role (min, median, max) durations — a R=4 reader with one thread 3× another is a striding bug, not a
  win
- **all 7 `(instr, stall)` pairs**, not `total_stalls`
- the exact oracle result (§6.1)

Instrumentation bias, measured: kernel spans ≲1.4%, Tensix counters ≤0.05%, **global sim clock 11.3%**. The
global clock is *not* a headline metric — 52% of it is not the op (init, program load, 7.5 MB
upload/readback, profiler flush) and that part grows with tensor size, so a genuine 2× kernel win moves it
only 1.31×. Keep it as a free run-identity/determinism fingerprint.

### 7.2 Stall signature

Baseline per-pipe at 40/core: unpack 2560 instr / **0 stall**, pack 1280 / **0**, sfpu 128 / 0,
math 10240 / 246730 (96%), sem 5152 / 224426 (97.8%), other 27392 / 230442 (89.4%).

**A decidable gate, with both endpoints measured** (baseline → sharded roofline). This is a gate row in
§2.3.1, so "falls" is not good enough:

1. `unpack_stall == 0 && pack_stall == 0 && sfpu_stall == 0` — **exact equality, no tolerance.** Non-zero means
   the bottleneck moved to output-DFB backpressure rather than the lever working.
2. `other_stall / (n_active_cores × per_core_span) ≤ 0.43` — this is the one metric that collapses when the DM
   pipeline stops pacing the Tensix: **0.842 → 0.014, a 60× swing.** Primary.
3. `sem_stall / (n_active_cores × per_core_span) ≤ 0.63` (0.820 → 0.447). Secondary.
4. Record the *fraction* forms but **never gate on them** — `sem_stall/(sem_stall+sem_instr)` **rises**
   97.8% → 98.2% from baseline to roofline, so it is not monotone in success.

**Do not gate on `math_stall` collapsing.** Even a fully compute-bound configuration is 85.7% math-stalled,
because `math_instr` is only 8/tile — the signature would not fire even if the prediction it was meant to
confirm were right.

**The validity identity is an inequality, not an equation.** v2.1 stated
`Σinstr + Σstall ≈ (#active TRISCs) × 32 × per-core span` and quoted 1.42%; that is wrong twice over — as
written it is `3 × 32 × 8549 = 820704` against a measured 748350, i.e. **+9.7%**, and the 737888 it quoted is
actually `32 × Σ(per-TRISC own spans)`. It also hardcodes 32 cores, and it holds only while every pipe has an
instruction pending ~100% of the time: an **idle** pipe contributes to neither term, so at the roofline — the
state the design is trying to reach — it fails by **+61.2%** (unpack is live 80 cycles inside a 2099-cycle
span, 96% idle rather than stalled). Note too that `instr` is an event count while `stall` is a cycle count,
which is masked only because stalls are 94% of the sum. Use
`Σ_pipes(instr + stall) ≤ n_active_cores × Σ_active_TRISCs(own span)` and **record the slack as a metric in its
own right** — 1.4% at baseline, 61% at the roofline, is itself a good progress signal.

Restate the v1 summary correctly: on the one active Tensix, the three active TRISC **issue slots** had an
instruction pending ~100% of the kernel window and were blocked 94% of those cycles; NEO1–3 and TRISC3 are
unused and contribute to neither numerator nor denominator.

### 7.3 Milestone 0 — reproduce, do not beat

Native factory at baseline-equivalent knobs (explicit, `entries_per_thread`=2, R=C=W=1).

- **Zero tolerance** on instrumentation-invariant work counters, stated **per active core** (v2.1 stated
  totals, which false-fail the moment the active core count changes): `unpack_instr=80`, `math_instr=320`,
  `pack_instr=40`, `cb_waits=80`, `cb_reserves=40`, `cb_pushes=40`, `cb_pops=80`. (`semaphores`/`other_instr`
  may legitimately differ — the native kernel is a rewrite.) **Excluded as null tests:** `kernel_launches`
  (512 in both a 32-core interleaved and a 4-core sharded run — it counts RISCs taken out of reset, 4 TRISCs ×
  4 Tensix × 32 nodes, so R and W cannot move it and C is already saturated) and `sfpu_instr` (128 in both —
  a global constant invariant to core count *and* layout). Leaving them in a "zero tolerance" list means 2 of
  its checks cannot fail.
- **±0.5%** per stage: reader 7781±39, writer 8019±40, math 8036±40, per-core span 8549±43. ("A few percent"
  would hide ~6 cyc/tile; one extra instruction per tile is +0.51%.)
- **Gate on slope**: fitted marginal **187.55 ± 0.5** from T=40 and T=80 (~30 s). This separates "prologue
  differs" (harmless) from "steady state differs" (must be explained).
- Plus the routing assertion (§6.2) — without it, matching cycles and matching output is exactly what a
  factory that never engaged looks like.
- Worth one `DPRINT` of `num_txn_ids / threshold / per_txn` here: §7.6's constraints are source-derived and
  one run would confirm them.

### 7.3.1 Attribution — how to do it, and the coverage rule that makes it valid

**Any attribution must state what fraction of the span its zones cover. Below ~90% it is a hypothesis, not
a split.** This is the rule the project paid for: an earlier version claimed a measured split while the
reader's two `noc.async_read()` *issue* calls sat in no zone at all, leaving ~44% unattributed. Reading
that remainder as "compute + stall" produced a prediction of ≤1.10× for thread scaling against an actual
**4.00×**. The measured law and per-config evidence are status §5; what belongs here is the method.

**Which stage binds depends on the config**, so "the op is X-bound" is not a property of the op — it is a
property of `(R,C,W)`. An attribution is only ever valid at the knobs it was taken at.

- **Instrument the blocking calls with `DeviceZoneScopedSumN1/N2` and `TT_METAL_PROFILER_SUM=1`**, not a zone
  around the loop. A wall-clock zone includes every cycle the loop is *blocked*, so it returns the pipeline
  rate, which divided by tiles is the fabricated-metric error of §2.1 one level down. Zone mechanics are fine
  either way — nesting inside `KERNEL` works and `PROFILER_L1_OPTIONAL_MARKER_COUNT = 250` at 2 words/marker
  gives 125 zones/RISC, so the cost is 1/125.
- **`SUM_COUNT = 2` (`profiler_common.h:18`) caps accumulators at two per RISC.** Two zones fill the
  *accumulator* budget, not the *cycle* budget — that is exactly the trap above. A third region costs one of
  the two.
- **Subtract the zone overhead**, calibrated as (instrumented − clean) / instances per tile. Absolute terms
  are ±5%; the *ratio* between terms is robust.
- **Every carried-over constant is re-measure-on-use.** Plan literals `RD_BAR 9.00 / WR_BAR 13.00` measured
  8.00/8.00 on the current sim.
- Keep the instrumentation portable (device profiler, not `TTSIM_*` counters) so it survives to the §7.5
  emulator campaign, which otherwise has no way to attribute its own number.

### 7.4 Lever order on craq-sim — threads first, because every other lever measures small *here*

**This order is craq-sim-specific and does not transfer to the emulator — see §2.4 and §7.5.** Per-lever
craq-sim results and their bias directions are the §2.4 table; the sequencing follows from it.

| step | knob | values | note |
|---|---|---|---|
| 1 | `reader_threads` R, then `writer_threads` W | R,W ∈ {1,2,4}, `R+W≤6` | **the whole question.** Requires the native factory — `num_threads > 1` is a host-side path, which is why this cannot be measured before implementing |
| 2 | `compute_threads` C | {1,2,4} | where the win actually is; also where `TENSIX_DEFAULT_LINGER` becomes live |
| 3 | DFB call batching, reader only | n=2 | banked 1.08×, worth keeping; do **not** batch the writer — `wait_front(n)` delays `pop_front` and starves compute of ring slots, monotonically worse to n=8 |
| 4 | `entries_per_thread` | as needed | an enabler here, a genuine latency-hiding lever on silicon |
| 5 | `implicit_sync` | false → true | mechanism study; small here, may matter on silicon |

Threads are first for an empirical reason — everything else has been *tried* and is small — which is also
what a thread-sweep failure would have meant: if R/W/C did not deliver, nothing left on this design would.
**That is why step 1 is the first thing the implementation answers, not the last.**

**The methodological rule this project earned the hard way: a cost being large does not make it recoverable.**
DFB calls are 56% of the per-tile cost and cutting call count 8× really does cut call cost ~8× — yet
end-to-end gain caps at 8%, because the pipeline re-absorbs the saving. Four magnitude estimates were made
from attribution today and all four were wrong, while every mechanism prediction held. **Attribution tells you
where time goes; only an experiment tells you what you can get back.** Structure every milestone as a
measurement with a written-down prediction, and never put an unmeasured multiplier in a gate.

### 7.5 The pre/post software gate — one emulator campaign at the end

**Development runs on craq-sim. Emulator use is deliberately minimized: one campaign after phase 1 is
functionally complete, not per-milestone.** What the design must therefore guarantee is not a schedule but
a *capability*:

> **A software gate that selects the pre-native path or the native path at runtime, in one build, on any
> platform — so a single emulator campaign can measure both arms back to back.**

**What the campaign must sweep — and both halves are roadmap items, so they are scheduled rather than
remembered.** The campaign has exactly two axes:

| axis | roadmap | why craq-sim cannot value it |
|---|---|---|
| **in-flight concurrency** — `implicit_sync`, ring depth, DFB batching | **F15 (M2.6)** | latency-hiding levers, and craq-sim has no latency to hide. Measured <= 1.10x / 1.02x / 1.08x there, but two of the three are **floors, not ceilings** |
| **per-operand reader allocation** — `R_in0` / `R_in1` | **F14 (M2.5)** | changes no instruction count; the case is DRAM/NoC locality, which craq-sim does not model at all |

Sweep them against thread counts, and sweep in-flight concurrency as **one** axis — ring `capacity`,
batch `n` and removing the wait are facets of the same quantity and are coupled by `capacity >= 2n`
(§7.6). Treating them as three independent knobs is how the craq-sim numbers got mistaken for verdicts in
the first place.

F14 carries the sharper edge of the two: it changes no instruction count and cannot move
the roofline, so a craq-sim sweep would return a confident ~0% and retire a live idea. Its entire case is
DRAM/NoC locality — the leading hypothesis being that tile-split pairs `in0[k]` and `in1[k]` on the *same
bank* (same page index under interleaved allocation, different buffers, therefore different rows). Only
the emulator can price that. Sweep it as an axis, not as a yes/no.

This matters because §7.4 establishes that implicit sync and ring depth produce no signal on craq-sim
(zero-cycle memcpy transfers, pre-satisfied barriers, no ISR or credit batching, IPC=1, no contention),
and §9.2 lists two silicon-only ceilings craq-sim cannot show at all. The emulator campaign is the only
place those become real numbers, so the switch must still work when we get there.

What that capability imposes — each of these is easy to lose by accident:

1. **The v2 path stays byte-identical and reachable, permanently.** This is why §3.4 is copy-then-modify
   rather than modify-in-place. `TTNN_QSR_NATIVE=0` must route the *same* input shapes back to
   `ProgramFactoryMetalV2` — which holds because `matches_metal_v2_slice` admits everything the native gate
   admits. Never narrow the v2 gate to "make room" for the native one.
2. **Both arms selectable in one build.** The switch is a host-side env var plus an arch check; nothing
   about it is craq-sim-specific, and nothing may become so.
3. **Nothing platform-specific in the harness.** `debug/bench_binary_ng_shapes.py` and
   `debug/prof_summary.py` must not hardcode `/workspaces/sim/libttsim.so`; platform comes from the
   environment. The runtime already abstracts it — `rtoptions().is_simulator_or_emulated()` covers both,
   `TT_METAL_SIMULATOR` selects craq-sim, `TT_METAL_EMULE_MODE=1` (+ `TT_METAL_MOCK_CLUSTER_DESC_PATH`)
   selects the emulated backend — and the quasar test lists already carry per-platform variants.
4. **The gated metric stays portable.** The per-core kernel span (§2.1) comes from the device profiler,
   which works on both. Do not let a craq-sim-only counter become load-bearing: `TTSIM_PERF_TRACE` is a
   *diagnostic*, and the §7.3 Milestone-0 work-counter checks are craq-sim-only by nature — say so where
   they are used, so their absence on the emulator is not read as a failure.
5. **Per-process isolation applies on the emulator too.** The §3.3 cache finding is platform-independent:
   one process per arm, or the second arm silently re-runs the first program.
6. **Watcher/assert flags differ by platform.** Per the ResNet bring-up notes, watcher asserts and
   NOC-sanitize are disabled on tt-sim because they do not work there, but **should stay enabled on the
   emulator**. Record the flag set with each run.
7. **Emulator numbers compare only to emulator numbers.** The campaign must capture *both* arms itself; a
   craq-sim baseline is not a valid reference for an emulator native run. This is the reason the switch has
   to survive to the end rather than being a temporary development aid.

**What the single campaign should measure.** The emulator behaves like silicon; craq-sim does not. So the
campaign's job is precisely the levers whose craq-sim numbers are least trustworthy — which is **not** the same
set as the levers that scored well on craq-sim.

1. **Sweep in-flight concurrency as ONE axis, and `R`/`W` as the other.** In-flight concurrency is the
   `(entries_per_thread, n, implicit_sync)` triple — they are facets of one quantity (§2.4), constrained by
   `capacity >= 2n`, so sweeping them independently wastes runs and produces uninterpretable results. Round 2
   recommended dropping
   `implicit_sync` from the campaign because craq-sim shows it at ≤1.10×, and I applied that here —
   **it was wrong, and this restores it.** The ≤1.10× is a craq-sim artifact: barriers there are
   pre-satisfied, so only their instruction cost is visible, whereas on silicon a barrier is a real per-tile
   NoC round-trip stall. Depth and `implicit_sync` are the *same* latency-hiding mechanism from two angles and
   must be swept **together** — §7.6: at `entries_per_thread = 2` in-flight is 1, so implicit sync is
   *equivalent* to the explicit barrier, and neither delivers without the other.
   Sweep `R`/`W` as well, for a different reason: §9.2's two silicon-only ceilings — all `R` producer threads
   sharing one txn-id set so they rendezvous every `per_txn`, and DM0's single ISR core servicing every credit
   of every DFB — cannot be pre-measured, and are exactly what `R=4`/`W=2` stresses. With six DM cores' credits
   funnelling through one ISR core, the additive behaviour measured at depth 2 could get *worse* with `R`,
   which would be a phase-1-invalidating result discoverable only there.
2. **Six runs, not four.** Two end-point configs cannot separate a prologue change from a steady-state
   one, and the prologue is exactly what `R=4` inflates (DFB init, the drain's `sync_threads`). This
   previously said "four runs — a T=40/T=80 two-point fit per arm"; **that is the error that cost a week
   of craq-sim marginals** (§2.1). Take **three** tile counts per arm, chosen so the successive
   differences come out equal — on craq-sim the native path is still bending at T=40 and only
   straightens by 60, and the emulator's prologue will be *larger*, so its linear region starts later
   still. **Verify equal increments on the emulator itself; do not inherit craq-sim's 60/120/180.**
   Getting this wrong is far more expensive here — emulator access is scoped to one campaign, so a
   biased slope cannot be re-run.
3. **Port the SUM zones into the native kernels first** (§7.3.1). They are the only instrument that can
   attribute the emulator's number, and they are portable by construction. §7.5 item 4 says "do not let a
   craq-sim-only counter become load-bearing" — this is the positive form of that rule.

**On the craq-sim → silicon calibration model: do not plan around it as a multiplier.** Its GBDT fits eltwise
well (holdout median 1.56%, n=1868 over 12,921 joined nodeids), but **every dataset is `bh-*`** — the target is
a Blackhole part, Quasar is pre-silicon, so there is no target column to fit against; the feature basis is
`TTSIM_PERF_TRACE`, which on our path is Tensix-only with all NoC counters zero; and the model's **weakest
family is `data_movement` at median 45% / p90 143%**, which is 76% of this op. Plan around it as
*methodology*: land the one-call-site `ttsim_perf_trace_noc` patch in the ROCC issue path (§10.4) so Quasar
features become join-compatible the moment emulator or silicon targets exist.

**Tradeoff being accepted deliberately:** deferring all emulator validation to the end means a structural
surprise (say, the DM0-ISR serialization dominating) is discovered late, after the craq-sim-guided design is
settled. That is the chosen cost of minimizing emulator time; the mitigation is that §9.2's silicon-only
ceilings are written down now, so a late surprise is recognized rather than re-derived.

### 7.6 What actually bounds in-flight transactions

`in_flight_per_thread = entries_per_thread / num_txn_ids`, where `num_txn_ids` is auto-derived as the
smallest `n ∈ [2,4]` satisfying `num_entries % (n · prods_or_cons · tcs_per_risc) == 0` — usually 2, and
**1 when none does** (n=3 is reachable; only n=4 is not). The fallback is not a corner case: at
`entries_per_thread = 1, R = 4` — a mandatory §6.6 correctness case — `num_entries = 4`, nothing in [2,4]
divides, so `n = 1` and `threshold = num_entries`, which is the worst case for the unguarded `uint8_t` cliff
(§4.4). Do not assume `n ≥ 2` anywhere.

`NUM_TXN_IDS = 4` bounds *batch granularity*, not concurrency; the NoC per-trid limit is effectively
unbounded. At `entries_per_thread = 2` this evaluates to **1 outstanding read — identical to the explicit
barrier**, which is why steps **4 and 5** of §7.4 are not separable and why the sync lever must be run deep.
And since the measured value of that lever is ≤9.6% (§2.3.3), "run it deep" is a mechanism study, not a
perf experiment.

## 8. Roadmap after phase 1 — milestones

**All of these are gated on §2.4's kill criterion**, which **Milestone 1.0 cleared on 2026-08-27**
(`R=4,C=4,W=2`, 2.70x measured at the 1280-tile benchmark shape / **4.00x asymptotic — exactly the
theoretical ceiling** — against a 1.30x bar).

**Two numbering systems, on purpose.** **`M#.#` is the sequence** — what gets done in what order, and
where the milestone boundaries fall. **`F#` is the stable identity** — labels never get renumbered, so
every cross-reference in this document (§4.3, §5.3, §6.1, §8.1) keeps working when priority changes.
Read the table top-to-bottom for order; cite `F#` in prose. Note `F#` in
`.link_to_claude/plans/quasar-native-binary-ng-review-findings.md` is a **different namespace** (review findings) — do not
conflate them.

| M# | F# | Item | Why here / what it unlocks |
|---|---|---|---|
| **1.0** | — | **Phase-1 slice + thread sweep — DONE 2026-08-27** | no-bcast tensor-tensor, TILE 32x32, bf16, FPU `add`, DRAM-interleaved, no activations, even divisibility. Kill criterion cleared; `R=4,C=4,W=2` is the optimum. |
| 1.1 | F1 | **Uneven tile counts** | First follow-on now that the criterion is cleared — every later milestone inherits the restriction otherwise. **Explicitly out of Milestone 0 and phase 1.** Mechanism settled: `KernelSpec` has **no per-thread runtime args**, so per-thread counts must be computed in-kernel from `get_my_thread_id()` (available on DM *and* TRISC). |
| 1.2 | F2 | **Rest of FPU op set** (subtract, multiply) | Gate widening. `multiply` is fidelity-dependent — copy the compute config verbatim or the §6.1 oracle breaks. |
| 1.3 | F3 | **Sharded / borrowed operands** | Zero NoC ⇒ isolates the compute levers. **Note:** 4-Tensix and `num_tiles_per_cycle > 1` are mutually exclusive (§4.3) — pick one per experiment, or implement `STRIDE_TILES` first. High model relevance (ResNet residual add). |
| 1.4 | F4 | **Mixed layouts** | Falls out of F3; existing kernels already parameterize per operand. |
| 1.5 | F5 | **fp32 + SFPU ops (divide)** | SFPU compute kernel, `enable_32_bit_dest`, `UnpackToDest` (free on Gen2, inert before here). int32 excluded pending the DFB-compute bug. **The §6.1 oracle expires here.** |
| 1.6 | F7 | **Activations (lhs/rhs/post)** | Compute-side self-loop DFBs, credit-balanced by construction. **New cost:** since #52762 (our branch point) `binary_tiles_init` inside `process_tiles` does 2 × `llk_unpack_program_bfd` per tile, burning 2 of 16 unpack partition ids per tile (wraps every 8). Re-measure; do not carry phase-1 cycles/tile over. |
| **2.0** | — | **Milestone 2 — reached once F7 lands** | The op is dtype-, layout-, memory- and activation-complete for whole-tile operands. Everything above is "the same op, wider"; everything below changes how a tile is *addressed*. |
| 2.1 | F13 | **Outer-dim broadcast** (leading dims, `N`/`C`/`D`/`nD`) | **First in this milestone, because it is a regression rather than new capability** — `kernels_dfb/` already does this and `kernels_qsr/` lost it (§3.4.1). `SubtileBroadcastType::NONE` compares H and W only, so outer dims are a separate axis that `no_bcast` still has to carry. Until it lands every broadcast `add` falls back and gets none of the multi-thread win, which caps the reachable model-level gain. Not a revert: the odometer is sequential and each thread must advance by `num_threads`; put it behind a factory-set compile-time flag so dense shapes keep the measured 165 cyc/tile. |
| 2.2 | F8 | **Subtile broadcast ROW/COL/SCALAR** | `ALL` consumer access + remapper fan-out. **Gated on `#51291`** or its stopgap. |
| 2.3 | F9 | **Mixed broadcast** | Preserve the ROW-via-LLK / COL-via-reader-fill hybrid; do not collapse to a third LLK pass. |
| 2.4 | F10 | **Tensor-scalar** | Writer fills `in1` once. Same fence dependency as F8. |
| 2.5 | F14 | **Per-operand reader allocation** (independent `R_in0` / `R_in1`) | **EMULATOR-ONLY EXPERIMENT — do not sweep this on craq-sim.** Today one `R` feeds both input DFBs, so every reader core reads both operands. There is **no roofline gain**: per-core read count is `T/2` either way, so `165/R` cannot move. The whole case is below that level, and craq-sim models a NoC read as a host `memcpy` — no banks, no rows, no contention — so it would return a confident ~0% and retire the idea for the wrong reason. Per §4.2's rule this is a **floor, not a ceiling**, like `implicit_sync` and ring depth. **Leading hypothesis (mechanism named, magnitude unmeasured): tile-split systematically pairs same-bank DRAM accesses.** With interleaved allocation page `p` maps to bank `p mod N`, and the loop reads `in0[k], in1[k]` — same page index, same bank, different buffers, therefore different rows, so a precharge+activate per tile. Operand-split walks `in0[k], in0[k+2], ...` across distinct banks. Caveat: `R` cores issue concurrently, so the controller may already see an interleaved stream and wash the per-core pattern out. Secondary: each core currently alternates between two `TensorAccessor`s, doubling the cached address-translation working set. **Proportional allocation is the more solid motivation, and it does not need broadcast** — F4 (mixed layouts) is the first point operand costs diverge: a sharded `in0` costs zero NoC while an interleaved `in1` does not, and tile-split forces every core to do one of each. F13 and F10 sharpen it further. **Binding design constraint is the STRIDED ratio rule, not the barrier**: each operand needs `max(p,C) % min(p,C) == 0`, so at `C=4` only `p in {1,2,4}` — 2+2, 4+2, 4+1 legal, **3+1 illegal**, i.e. the naive allocation for a 3:1 traffic ratio is the one that fails. **Nothing at LLK or DFB level prevents any of this:** STRIDED mode exists for `producers != consumers` and one DFB already admits multiple producer KernelSpecs (`:1130`); all three blockers are host/firmware software — `NUM_KERNEL_BARRIERS = 2` (whose header comment names the fix: key the barrier per kernel-group) and deriving a DFB's producer count from its kernel's `num_threads`. The per-node producer/consumer census is **not** a blocker: it is scoped per DFB (§3.3.1-B), so two reader kernels owning one input DFB each pass it. `wait_threads()` returns immediately for `participants <= 1`, so an asymmetric multi-thread + single-thread split never contends for a barrier slot at all — **4+1+1 and 2+1+2 are reachable today; only the symmetric 2+2 needs the barrier change**, and §3.3.1-B enumerates the space and assesses that change as plumbing rather than design. |
| 2.6 | F15 | **In-flight concurrency** (`implicit_sync`, ring depth, DFB batching) | **EMULATOR-ONLY — same campaign as F14, and for the same reason.** §7.4 measured all three on craq-sim: `implicit_sync` <= **1.10x**, `entries_per_thread` **1.02x**, reader batching `n=2` **1.08x**, composing to ~1.17x. **Two of those three are floors, not ceilings** — they are latency-hiding levers and craq-sim has no latency to hide (zero-cycle memcpy transfers, pre-satisfied barriers), so the small numbers are the simulator declining to answer, not a verdict on the knobs. Their real size is **unknown**. **They are one lever, not three** (§7.6): ring `capacity`, batch `n` and removing the wait are all facets of *how many tile transfers are in flight at once*, and they are coupled — `capacity >= 2n` is required for any overlap at all. Sweep in-flight concurrency as a single axis against thread counts, never as three independent knobs. **Known negative, do not re-run: batching the WRITER is actively harmful** — `wait_front(n)` delays `pop_front` and starves compute of ring slots, degrading monotonically to 1.02x at `n=8`. The native factory currently runs explicit sync (`disable_dfb_implicit_sync_for_all=true`), so `implicit_sync` is off and turning it on is the experiment. |
| **3.0** | — | **Milestone 3 — reached once F10 lands** | Broadcast-complete. What remains is the long tail: a different physical layout, the op families with their own kernels, and a format that does not exist in TTNN yet. |
| 3.1 | F11 | **Row-major** | Quasar needs explicit 16-byte RM shard-width alignment. |
| 3.2 | F12 | **where / quantization / int32** | Own kernel families; int32 blocked on the DFB-compute bug. |
| 3.3 | F6 | **MX formats** — **LAST** | Quasar replaces all BFP with MX; the ResNet slice that ships as bf8 is MXINT8 here. Bigger than a dtype widening — see §8.1. Deprioritized 2026-08-22 and placed last: it needs a new TTNN `DataType` plus IDMA gasket support, so it is the one item whose cost is dominated by work **outside this op**. |

**F6 last (M3.3)** is a decision, not a derivation — by explicit call 2026-08-22, reaffirmed. Its cost
sits outside this op.

**F13 is a regression, not a feature** — it restores behaviour `kernels_dfb/` already has, so it opens
the broadcast milestone rather than being ranked against the capabilities beside it. Origin and fix: §3.4.1.

**Removed from F5:** "first shot at FPU/SFPU overlap on TRISC3". TRISC3 idle is *correct* for an
FPU-only op, and the lever is **structurally blocked**, not merely unwired: TRISC3 has no DFB producer
interface (the pack-shaped `LocalDFBInterface` is `#if UCK_CHLKC_PACK` only and `dfb_advance_slot` is not
compiled for it), there is no Compute-API surface (the only precedent is hand-written raw LLK in a tt-llk
test, which also violates the `[24,32)` BFD partition), there is a known HW bug on exactly this shape
(no unpacker auto-loops for binary unpacking, issue #1635), and it would invalidate the bit-exact oracle
(SFPU add ≠ FPU add bitwise). Tracked as a separate LLK-blocked investigation.

**Cross-cutting:** validate **fast dispatch** for DFB-bearing specs before calling any of this
production-ready; then the hardening pass (strict gate, CI wiring, env-var default flip, knobs into the hash).

### 8.1 Notes for the MX-format phase (F6, sequenced last; read only when it is actually scheduled)

- **Prerequisite v1 missed entirely: there is no MX `DataType` in TTNN.** `tensor_types.hpp` ends at
  `INT8 = 9`, so a TTNN tensor cannot hold MX. F6 needs a new `DataType`, a
  `datatype_to_dataformat_converter` arm, a tile-size arm and a host pack path — not just format plumbing.
- **Host MX packing does exist** (`mxint.hpp`, `mxfp4/6/8.hpp`, `impl/data_format/mx_tile_pack.hpp`). Only the **IDMA gaskets** lack it, so MX cannot be
  staged or converted through the dispatch engine.
- **Host and device `DataFormat` disagree numerically, and the device reuses the BFP encodings:** host
  `MxInt8=12, MxInt4=16, MxInt2=17, MxFp4_2x_B=29` vs device `MxInt8=2, MxInt4=3, MxInt2=11,
  MxFp4_2x_B=24` — device MX sits in the host's Bfp8/Bfp4/Bfp2 slots. The remap
  (`jit_build/genfiles.cpp:641-663`) applies **only** to arrays passing through `emit_formats_array`; any
  format value pushed to the device by another route must be remapped by hand.
- **An MX tile is a two-region object** `[scales padded to L1 alignment][packed elements]`. That breaks more
  than `entry_size`: `base_entry_idx = (base − tc_slots[0].base_addr) / entry_size` and the BFD x/y/z address
  generator both assume a homogeneous tile, and `llk_unpack_program_bfd` takes a single `TensorShape` + base
  + format. The "multi-TC not handled" TODO there is benign for phase 1 and F3 but **not** for MX.
- **Carve-out to encode:** `enable_2x_src_register` must stay **false** for eltwise — Mxfp4-only, and valid
  only for the matmul family (MVMUL/MVMULDI) and the GAPOOL column reduce; other instructions "produce
  garbage math results". Silent wrong answer, not a hang. It defaults to false.
- `is_block_float()` is a `DataType` predicate over BFLOAT8_B/BFLOAT4_B — **vacuously false** on Quasar and
  therefore harmless today. It needs an arm only once MX `DataType`s exist.
- MX is two's-complement with an E8M0 block scale — a genuinely different encoding, not a rename.

## 9. Prerequisites, deferred, and open

### 9.1 Prerequisites outside this op — gate items, not nice-to-haves

1. **`bfd_state` must become `thread_local` before `compute_threads > 1`.**
   **Filed: https://github.com/tenstorrent/tt-llk/issues/1678** (2026-08-19; draft + full reasoning in
   `tt-llk-issue-bfd-state-shared-across-neos.md`).
   **Status semantics: this blocks CONCLUSIONS at C>1, not progress.** Writing the C=4 path and running it
   on craq-sim is fine; treating a green craq-sim C=4 run as validation, or reporting its cycles, is not.
   craq-sim is blind to mechanism 2 (synchronous stores, no store buffer ⇒ no reordering) and *deterministic*
   about mechanism 1 (instruction-granularity interleaving ⇒ fires every run or never), so a pass there is
   evidence-free. Note the bit-exact oracle would silently absorb this: a race that happens to produce
   correct data on the fixed schedule passes, banking a false green. If mechanism 1 does fire, expect
   garbage or an unpacker hang at C=4 — recognize it as this issue rather than debugging our own kernel.
   `tt_llk_quasar/common/inc/llk_bfd_alloc.h:117` declares it as a plain `inline` global; the shared-globals
   handshake is keyed on **trisc_id, not neo_id**, so NEO0–3's TRISC0 share one allocator and 4 SPMD compute
   threads race in `binary_tiles_init`. Two failure paths: `llk_unpack_AB_api.h:50-57` re-reads
   `bfd_current<E>()` after programming, so a concurrent alloc points this Neo's MOP at a descriptor never
   programmed in its own table; and the unsynchronised lazy-init lets a Neo read `next == 0` and hand out an
   id in another TRISC's partition, where the shared 32-entry BFD table lets the packer clobber an unpack
   descriptor. Fix: `thread_local` (mirroring `dest_register_offset`) plus passing the returned id through
   instead of re-reading.
   **Why this survived:** `DMTensixTest1xDFB4Sx4S` exists but its consumer kernel documents that DM→Tensix
   L1 verification is **omitted** — 4-Tensix DFB coverage with no data check. Worth reporting upstream
   independently of this project.
2. **`llk_tdma_guard::armed_mask()` is a function-local `static` shared across Neos**, so
   `TT_METAL_LLK_ASSERTS=1` yields false negatives *and* false positives at C>1. Either make it
   `thread_local` or restrict the flag to C=1 and say so in the bring-up notes.

### 9.2 Silicon-only ceilings — invisible to craq-sim

A clean multi-DM scaling curve on the simulator is not evidence about silicon:
1. **All R producer threads share one txn-id set**, and the CMDBUF counter aggregates across DMs, so reader
   threads rendezvous every `per_txn` transactions — per-thread jitter does **not** overlap.
2. **DM0's ISR is a single-core serialization point** for every credit post/ack of every DFB on the node —
   one core servicing R=4, W=2 across three DFBs.

### 9.3 Still open

- **Knobs vs program hash** — resolved in the hardening pass, not phase 1.
- **`TT_METAL_DEVICE_PROFILER_NOC_EVENTS` — an emulator instrument, if the buffer allows.** Device-side
  recording is already on the Quasar path (`noc.h` `RECORD_NOC_EVENT_WITH_ADDR`, `enable_noc_tracing`
  defaults true; tt-2xx internals reference the macros too), but the profiler headers have no Quasar arm,
  so host-side decode is untested. It yields issue timestamps and **barrier start/end pairs** — no
  per-transaction completion event — so it measures barrier cost on the *explicit* path and cannot
  recover in-flight depth on the implicit one. Useless on craq-sim (barriers are pre-satisfied); belongs
  to §7.5. Volume is the blocker: one event per transaction against a buffer that already saturates at
  22 RISCs × 125 zones, so it needs a tiny shape or `shouldRecordEvent` filtering.
