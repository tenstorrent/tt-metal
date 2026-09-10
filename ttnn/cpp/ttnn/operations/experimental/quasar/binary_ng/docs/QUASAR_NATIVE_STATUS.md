# Quasar-native `binary_ng` — status

> **Committed for the record.** This is an engineering record of the Quasar-native `binary_ng` effort, not
> user documentation. Two kinds of reference in here point outside the repository and are expected to
> dangle for a reader who was not on the original branch:
> - `debug/attrib/*` — the diagnostic drivers, sweeps and plotting scripts. `debug/` is deliberately
>   untracked; the numbers they produced are reproduced inline here.
> - `.link_to_claude/plans/*` — the implementation plan, the specialist review findings, and the
>   measurement-discipline notes, which stayed out of the repo.
>
>
*Slide-style status deck. Each `---` is a slide. Keep slides to one screen.*
***Chronological: newest week first.***

Two platforms, and the distinction runs through everything below.** *craq-sim* is a fast functional
simulator (~15 s/run, deterministic) with no transfer latency and no contention — it is where all development
happens. The *hardware emulator* is the closest thing to real behaviour we can get; it is available but far
less accessible, so it is spent deliberately and rarely. **Never mix numbers from the two.**

# ► Week of 2026-09-03 — Milestone 1.0 shipped; 1.1 measured 34/34; a main regression found

## 1. TL;DR

- *
- Milestone 1.0 is merged.** PR #55000, squashed as `d66f111add3`. The Quasar-native factory, its
  kernels and the three record docs are on main.
- Fixed metal host 2.0 production binary_ng issue #54138
- **Two defects filed**, one to each owner: craq-sim#338 (implicit sync unimplemented) and
  tt-metal#55276 (chained-DFB corruption — the 18-config data corruption).
- **Milestone 1.1 correctness is answered: 34 of 34 bit-exact**, full coverage matrix, on a
  **one-line** kernel fix. Uneven tile counts work, **including zero-work threads** (slide 3).
- **A third defect found, in main itself:** multi-threaded Quasar DFB **hangs** on `origin/main` while
  passing at our pre-merge anchor. Attributed by bisection, not yet root-caused (slide 4). **This is now
  the critical path** — M1.1 cannot land until it is fixed.

Also: reviewed the 1.0 perf analysis for inaccuracies and corrected the docs in place

Also: attemped to support quasar CI, but currently blocked by craq-sim release procedure.

---

## 2. Two defects filed, routed by owner

| issue | what | why that owner |
|---|---|---|
| **craq-sim#338** | Implicit sync is **not implemented**: `PER_TR_ID_IP_*` reads 0 unconditionally, and DFB credit posting is `posted++` per transaction keyed only on `(tensix_id, counter)` — no txn-ID dimension, no threshold. The host computes the whole txn-ID apparatus and the simulator discards it. | Two named, verified simulator gaps. An existing upstream test is already red on an unmodified tree. |
| **tt-metal#55276** | Two DFBs chained through a Tensix stage silently return wrong data when a DM endpoint outnumbers the Tensix side — the 18-of-31 config corruption, reproduced in a standalone gtest with no TTNN. | Two of the three candidate layers are tt-metal code, and the one mechanism we can name is host-side: `tile_counter_allocator_` is a member of `ProgramImpl`, so it is **program-scoped, not per-DFB** — both DFBs on a cluster draw counters from one allocator. |

Both carry a reproducer that applies to `origin/main`. The tests stayed **unstaged by decision**: these
DFB gtests run in no CI list, so merging them would add tests nothing executes.

---

## 3. Milestone 1.1 / F1 — measured: 34 of 34 bit-exact on a one-line fix

**The whole change is one line.** The compute kernel's `my_tiles = num_tiles / get_num_threads()`
truncated, so `Tc % C != 0` left entries unconsumed and the writer waited forever. It becomes
`num_tiles / N + (get_my_thread_id() < num_tiles % N ? 1 : 0)` — the share the DFB already hands
consumer thread `c`. Reader and writer needed nothing: their strided loops are already uneven-safe, and
per-cluster unequal counts were already plumbed through `split_work_to_cores` (metal's "core" is a
whole Quasar cluster).

**Result: 34/34 PASS, `mismatch = 0` on every case, native routing asserted throughout.** The full
matrix below was run one process per case, on the pre-merge anchor `8e3f13a177b` (main itself cannot run
multi-threaded — slide 4).

**Two predictions from source analysis that the data falsified:**

- **The empty-thread deadlock does not happen — the barrier is unreachable on our path.** The only
  `sync_threads` in the DFB path (`dataflow_buffer.inl:390`) lives inside `handle_final_credits`, whose
  two callers are both guarded by `ptiles_read_ > 0` / `ctiles_written_ > 0`. Those counters are
  incremented *only* by `commit_implicit_read`/`commit_implicit_write` (`:538`, `:572`), reached only
  from the **implicit-sync** overloads. **Our factory hardcodes explicit sync, so both are permanently 0
  and no thread ever calls it.** The deadlock needs an asymmetry — some threads arriving, one not — and
  nobody arrives. Empty threads aren't handled; they never matter. *The hazard is still real for
  implicit sync (M2.6), which craq-sim cannot run anyway (craq-sim#338).*

**Coverage rests on two independent levels**, which is what makes a small matrix sufficient:

1. **Cluster level** — `split_work_to_cores` yields at most two groups differing by 1.
2. **Thread level** — depends *only* on each cluster's count `Tc`, so `T = 32·Tc` isolates it completely.

### Level 2 — thread level, all clusters identical (`T = 32·Tc`), at `(4,4,2)` — **all PASS**

| `Tc` | `T` | R=4 → | C=4 → | W=2 → | what only this case exercises |
|---:|---:|---|---|---|---|
| 1 | 32 | 1,0,0,0 | 1,0,0,0 | 1,0 | empties on **all three** axes |
| 2 | 64 | 1,1,0,0 | 1,1,0,0 | 1,1 | empties on R/C while **W is exactly even** |
| 3 | 96 | 1,1,1,0 | 1,1,1,0 | 2,1 | a **single** empty thread + tail on W |
| **4** | **128** | 1,1,1,1 | 1,1,1,1 | 2,2 | today's minimum legal shape — even everywhere |
| 5 | 160 | 2,1,1,1 | 2,1,1,1 | 3,2 | tail on all three, **no** empties |
| 6 | 192 | 2,2,1,1 | 2,2,1,1 | 3,3 | tail on R/C while W is even, no empties |
| 7 | 224 | 2,2,2,1 | 2,2,2,1 | 4,3 | tail on all three, opposite parity |
| 8 | 256 | 2,2,2,2 | even | 4,4 | even, ring **exactly** full |
| 9 | 288 | 3,2,2,2 | 3,2,2,2 | 5,4 | first tail that **wraps** the ring |
| 40 | 1280 | 10 each | 10 each | 20,20 | even, deep wrap — the perf shape |
| 41 | 1312 | 11,10,10,10 | same | 21,20 | tail at depth, deep wrap |

`Tc=2` and `Tc=6` are the discriminating pair: both leave W exactly even while R and C are ragged, one
with empty threads and one without — which separates the truncating divide from the empty-thread case.
A single ragged shape conflates them.

### Level 1 — cluster level — **all PASS**

| `T` | clusters | per-cluster | exercises |
|---:|---:|---|---|
| 1 | 1 | 1 | single cluster, 3 empty readers |
| 31 | 31 | 1 each | **fewer clusters than the grid** — one cluster gets no kernel at all |
| 33 | 32 | one 2, thirty-one 1 | two groups **and** empties |
| 129 | 32 | one 5, thirty-one 4 | two groups, tails, no empties |
| 1281 | 32 | one 41, thirty-one 40 | two groups at perf scale |

**`(4,4,2)` cannot be the only config.** Its input DFBs are entirely `num_tcs_to_rr = 1`, so it never
reaches the `handle_final_credits` tail branch. `1,4,4` and `4,4,1` drive `N=4` with one thread owning
every counter — first-class cases, not sanity checks. Config coverage, all PASS:

| config | cases | reaches |
|---|---:|---|
| `4,4,2` | 16 | full Level-2 `Tc` sweep + all Level-1 splits |
| `1,4,4` | 6 | `num_tcs_to_rr = 4` on the **producer** side |
| `4,4,1` | 6 | `num_tcs_to_rr = 4` on the **consumer** side |
| `2,4,2` | 4 | `N = 2` on **both** sides at once |
| `1,1,1` | 2 | degenerate control — every `N = 1` |

**Any `T` is constructible** as `[1, 1, 32, T·32]` — legitimate because the reader walks
`page = start_tile_id + k`, so behaviour depends on `T` alone, not on how it factors.

**F1 is now implemented, not just measured.** The divisibility gate is **deleted** — not bypassed —
so `matches_quasar_native_slice` requires only a non-empty output and a non-empty worker grid; the
`lcm(R,C,W)` computation that existed solely to serve it is gone. Eleven ragged shapes and a
cache-interference case are **checked in** to `test_binary_ng_quasar_native.py`, each asserting native
routing from `kernels.yaml` so a silent fallback cannot pass. 14 passed with the knob on, and the
fallback arm is unaffected (1 passed, 14 skipped with it off).

Program-cache behaviour is asserted rather than assumed: the op has no `compute_program_hash` override,
so the framework hashes tensor specs and different shapes take different entries — the new test walks
even → ragged → even → empty-thread → even in **one process** and re-checks each.

**Two caveats stand.** This is craq-sim, which prices data movement at zero, so 34/34 bit-exact is a
correctness result and says nothing about what uneven work *costs*. And none of it can land while main
hangs multi-threaded (slide 4).

Full write-up: `.link_to_claude/plans/quasar-m1-1-uneven-tiles.md`.

---

## 4. BLOCKER — two regressions in main; Quasar multi-thread does not run there

Two independent problems, found after resetting the branch onto main. **Neither is visible to CI**,
because Quasar tests run only on real WH/BH — never craq-sim.

**4.1 No Quasar DFB kernel compiles (confirmed, patched locally).** `831426fef6f` (#51597) appended
`&& !defined(NOC_API_V1)` to both include guards in `dataflow_buffer.h`, and `jit_build/build.cpp:220`
defines `NOC_API_V1` for Quasar-on-`.so`-simulator — so craq-sim builds take the **tt-1xx (Gen1)**
headers. Reproduced on the untouched upstream nightly test. The impl-selection guard is collateral: the
commit only meant to hide the *zeroing* API, and used the correct skip-pattern for `noc_zero_dram.inl`
but not for the other two.

**4.2 Multi-threaded DFB hangs on main — ATTRIBUTED by bisection, not root-caused.**

Same test, same simulator (`ad401613`), same env; the only variable is the commit:

| commit | native `4,4,2`, 1280 tiles | |
|---|---|---|
| **`8e3f13a177b`** — our pre-merge branch head | **PASS 4.83 s** | the anchor; docs' `44.12 cyc/tile` reproduce here |
| **`origin/main`** `d2f4b3afeca` | **HANG** — 101% CPU, indefinite | |

⇒ **main regressed multi-threaded Quasar DFB** somewhere between the branch's base and today.

Ruled out by direct test rather than reasoning: our kernel edits (reverted, still hangs); the probe and
its shape (reproduces on the untouched upstream nightly test); the two simulator scheduling env vars;
and **craq-sim version** — identical hang on `9ed8f797` and `ad401613`. On main, native `1,1,1` passes
(6.91 s) and the non-native path passes (7.15 s), so the surviving difference is *one thread vs many*.

**Note the merge commit is not a safe anchor.** `831426fef6f` landed 2026-09-01, a day before PR #55000
merged, so `d66f111add3` inherits 4.1 as well. The last commit that runs Quasar multi-thread on craq-sim
is the pre-merge branch head. Next step is bisecting main between the branch base and `d2f4b3afeca`.

---

## 5. Quasar has no craq-sim CI, and it cost us twice this week

The Quasar regression lists (`tests/scripts/quasar/quasar_regression_tests.yaml`,
`quasar_sim_regresion_tests.yaml`) are explicit per-test allowlists, and nightly runs Quasar only on real
WH/BH SKUs. So **nothing in CI ever executes a Quasar DFB test on craq-sim.**

Two regressions this week landed in main and sat there unnoticed as a direct result: the `NOC_API_V1`
guard breakage (4.1), and an upstream implicit-sync test that is already red on an unmodified tree
(craq-sim#338). Both were found by hand, locally.

**Local experiment on the version axis:** craq-sim was 13 commits stale (2026-08-26 against a Sep-3
main). We updated to `ad401613`, rebuilt, and re-tested — then rebuilt again at the old `9ed8f797` to
test causality. Result: the skew was real housekeeping but **not** the cause of 4.2. Version skew has now
bitten three times, and this instance presented as a **silent spin**, not the loud
`UnimplementedFunctionality` abort of the earlier two — the more dangerous form, since it is
indistinguishable from the DFB deadlocks we actually hunt.

**Standing rule adopted:** after any rebase onto main, run a known-good sanity case *and* check craq-sim
is current, before taking any new measurement.

---

## 6. Next

1. **Bisect 4.2** between the branch base and `d2f4b3afeca`. It is the critical path: nothing merges
   while main cannot run Quasar multi-thread, and it is a regression against a shipped feature.
2. **Report 4.1 upstream** — small, well-evidenced, unblocks every craq-sim user. Local patch ready.
---

# ► Week of 2026-08-27 — Milestone 1 measured

## 1. TL;DR — kill criterion cleared, premise validated

The founding question was whether Quasar's idle engines are worth exploiting for elementwise ops: the
baseline used **2 of 6** DM cores and **1 of 4** Tensix. Answer: **yes**, by a wide margin.

`R=4, C=4, W=2` — all 6 user DM cores, all 4 Neos — delivers a measured **2.70x latency gain** at the
1280-tile benchmark shape (re-measured 2026-09-09 as 2.69x), on a **4.00x throughput gain** that the
latency gain approaches as tensors grow past the fixed launch cost — **exactly the theoretical ceiling** —
**bit-exact**, against a **1.30x** go/no-go criterion. Baseline is the native factory at `1,1,1`
(Milestone-1 code). **GO.**

Also this week: rebased onto main (302 commits; tt-llk #1678 landed, so `C > 1` is live), and craq-sim blocker issue #319 has been fixed. Tasks 4 and 5
landed (thread-generic kernels + host wiring), and the full legal `(R,C,W)` space was measured
exhaustively for correctness.

---

## 2. The legal space, and a platform defect

31 of 108 `(R,C,W)` candidates are legal — where **the 108 already has `C ∈ {1,2,4}` applied**,  and the 31 measures DM-budget and stride
attrition only. More detailes how 31 is legal is from design doc §3.3.1,
reproducible via `debug/attrib/enumerate_legal_space.py`. All 31 measured for correctness at 60
tiles/cluster, one process each, bit-exact vs a torch golden with a routing assertion:

| | count | outcome |
|---|---|---|
| `R <= C` and `W <= C` | **13** | all bit-exact, `mismatch = 0` |
| `R > C` or `W > C` | **18** | all wrong, 43%–84% of elements |

**31 of 31 agree; 0 disagreements.** A DFB whose **DM cores outnumber its Tensix cores** silently returns
wrong data — no hang, no error. Localised per tile: exactly `C` of the `n` DM sub-streams are serviced,
the other `n - C` receive nothing valid. Upstream DFB gtests never cover `producers != consumers` with a
Tensix on the narrow side. Issue is being filed to craq-sim for fix.

Those 18 were measured for perf as well
as correctness, and they are the only reason the compute term is known at all.

---

## 3. Throughput gain and latency gain — two different quantities

Both arms are the **Quasar-native factory** (Milestone 1): identical kernels and factory, thread counts
set to `1,1,1` for the baseline, so the ratio isolates threading and nothing else. The Milestone-0
`metal_v2` measurement (§3 of the 2026-08-27 entry) is a **history record** and is never a denominator
for perf gains. Basis rule: design §2.1.

`span(T) = prologue + marginal × T`. Measured spans, median over 32 clusters, both arms, one sim session:

| tiles/cluster | total tiles | `1,1,1` span | `4,4,2` span | **latency gain** |
|---:|---:|---:|---:|---:|
| 40 | 1280 | 7813 | 2909 | **2.69x** |
| 60 | 1920 | 11333 | 3751 | 3.02x |
| 120 | 3840 | 21893 | 6395 | 3.42x |
| 180 | 5760 | 32453 | 9031 | 3.59x |
| ∞ | — | — | — | 4.00x |

Fitted over 60/120/180 (the straight region): `1,1,1` = **773 + 176.00·T**, `4,4,2` = **1112 + 44.00·T**.
The `1,1,1` arm is exactly linear (both 60-cycle steps are 10560); `4,4,2` least-squares to 44.00 with a
±0.07 spread. Reproduces the earlier independent fit (176.50/44.12, prologue 767/1106) within 0.3%.

| the gain | value | basis | what it answers |
|---|---:|---|---|
| **throughput gain** | **4.00x** | `176.00 / 44.00`, slope ratio | how much faster tiles retire in steady state |
| **latency gain** | **2.69x** | `7813 / 2909`, span ratio @ 1280 tiles | how much sooner the op finishes at the benchmark shape |

**They are not two readings of one number.** Throughput is a rate: its gain is the slope ratio and carries
no prologue, so it is size-independent. Latency is the time to finish a fixed tensor: its gain is
`(773 + 176T) / (1112 + 44T)`, which climbs monotonically with `T` and approaches 4.00x without reaching
it. **Throughput gain is the ceiling; latency gain is what is delivered at a stated size.** Quote the
latency gain for a result and name the size; quote the throughput gain as the asymptote — never the
reverse, and never either one unlabelled.

**4.00x is the ceiling, not a coincidence.** Going `1,1,1 → 4,4,2` shrinks the three roofline terms by
(reader 4x, compute 4x, **writer only 2x**), so no cost model of the form `f(Rc/R, Cc/C, Wc/W)` can
exceed 4x. Measured twice on independent fits: `176.50 / 44.12 = 4.0005` and `176.00 / 44.00 = 4.0000`.
The constants were pinned independently — `Cc` from the `C=1` plateau, `Rc`/`Wc` from the `R=1` and `W=1`
rows — so landing on the cap is a check the data could have failed, not a construction. Note the *ratio*
reproduced to 4 digits across the two fits even though the absolute constants moved 0.3%.

**The whole gap between the two gains is prologue.** `4,4,2` carries the larger fixed cost (1112 vs 773
cycles — more cores to launch and rendezvous), and at 40 tiles/cluster that is **38% of its span**. So the
faster config pays a bigger entry fee, which is why the latency gain starts well below the ceiling and
climbs as the body grows: 2.69x → 3.02x → 3.42x → 3.59x over 40 → 180 tiles/cluster. **Quote 2.69x as the
measured result at the benchmark shape; 4.00x is the asymptote.**

---

## 4. Full measured table — the 13 correctness-clean configs

Marginal = slope of `span` vs tiles/cluster, fitted over **60/120/180** (the linear region), `span` =
per-cluster KERNEL-zone span median over 32 clusters, `entries_per_thread = 4`. **Units differ**:
marginal and raw@60 are cyc/tile; **prologue is absolute cycles** (the fit's intercept). `bound` names
the roofline term that sets the value — read it as reader-bound / compute-bound / writer-bound.

| R | C | W | DM | Neo | marginal | speedup | bound | raw @60 | prologue |
|---|---|---|---|---|---|---|---|---|---|
| **4** | **4** | **2** | 6 | 4 | **44.12** | **4.00x** | cmp | 62.55 | 1106 |
| 2 | 4 | 2 | 4 | 4 | 82.53 | 2.14x | rdr | 102.62 | 1205 |
| 2 | 4 | 4 | 6 | 4 | 82.53 | 2.14x | rdr | 103.33 | 1248 |
| 2 | 4 | 1 | 3 | 4 | 83.53 | 2.11x | wtr | 105.62 | 1325 |
| 4 | 4 | 1 | 5 | 4 | 83.53 | 2.11x | wtr | 104.48 | 1257 |
| 2 | 2 | 1 | 3 | 2 | 88.25 | 2.00x | cmp | 104.25 | 962 |
| 2 | 2 | 2 | 4 | 2 | 88.25 | 2.00x | cmp | 102.53 | 858 |
| 1 | 2 | 1 | 2 | 2 | 165.00 | 1.07x | rdr | 184.63 | 1179 |
| 1 | 2 | 2 | 3 | 2 | 165.00 | 1.07x | rdr | 184.93 | 1197 |
| 1 | 4 | 1 | 2 | 4 | 165.07 | 1.07x | rdr | 188.92 | 1431 |
| 1 | 4 | 2 | 3 | 4 | 165.07 | 1.07x | rdr | 189.42 | 1461 |
| 1 | 4 | 4 | 5 | 4 | 165.07 | 1.07x | rdr | 189.92 | 1491 |
| 1 | 1 | 1 | 2 | 1 | 176.50 | 1.00x | cmp | 189.28 | 767 |

**Every config sits exactly on its binding term** — `176.5/C`, `165/R` or `83.5/W`, to within 0.04%.
That is sharper than the old two-point table, which showed three fuzzy tiers; the tiers were never
fuzzy, the measurement was. Cheapest per tier: `1,1,1` (2 DM) → `2,2,1` (3 DM) → `4,4,2` (6 DM).

**Along the balanced frontier scaling is exactly linear.** `2,2,1` (3 DM, 2 Neo) 88.25 → `4,4,2`
(6 DM, 4 Neo) 44.12 = **2.0002x on exactly 2x the engines**.


---

## 5. Exhaustive 31-config space — correctness and perf

**Correctness** for all 31 at 48x40 (60 tiles/cluster), one process each, bit-exact vs a torch golden
with a routing assertion. **Perf** = marginal fitted over 60/120/180 tiles/cluster; all 31 admitted at
every point, `occ:OK route:OK` throughout. `pred` is `max(165.0/R, 176.5/C, 83.5/W)` and `bound` names
the term that sets it — reader-bound / compute-bound / writer-bound. Rules and rejection counts:
design §3.3.1.

The **DFB** columns give each config's endpoint shape in the notation the upstream gtest matrix uses:
`<producers>S x <consumers>S`, where `S` = the STRIDED access pattern (thread *t* takes every *N*-th
entry). Every endpoint we bind is STRIDED on both sides, so the letters never vary — the counts do:
`in0`/`in1` are `(R, C)` DM→Tensix, `out` is `(C, W)` Tensix→DM. **† = no `DFB_TEST_2_0` declares that
shape** (8 of 62 cells). Generated by `debug/attrib/add_access_pattern_column.py`, which parses the
declarations out of `test_dataflow_buffer_base.cpp` rather than transcribing them.

**A shape without † is run upstream, not checked upstream.** The DM→Tensix tests assert only that the
program ran — the consumer `copy_tile`s each entry into dest and discards it, and
`dfb_test_common.hpp:539-540` states the L1 verification is omitted. So `†` marks a coverage gap and its
absence marks nothing; see the note under the table.

| pred | bound | R,C,W | DM | Neo | in0/in1 DFB | out DFB | marginal | correctness |
|---|---|---|---|---|---|---|---|---|
| 44.12 | **cmp** | **4,4,2** | 6 | 4 | 4Sx4S | 4Sx2S | **44.12** | **PASS — OPTIMUM** |
| 82.50 | rdr | 2,4,2 | 4 | 4 | 2Sx4S | 4Sx2S | 82.53 | PASS |
| 82.50 | rdr | 2,4,4 | 6 | 4 | 2Sx4S | 4Sx4S | 82.53 | PASS |
| 83.50 | wtr | 2,4,1 | 3 | 4 | 2Sx4S | 4Sx1S | 83.53 | PASS |
| 83.50 | wtr | 4,4,1 | 5 | 4 | 4Sx4S | 4Sx1S | 83.53 | PASS |
| 88.25 | **cmp** | 2,2,1 | 3 | 2 | 2Sx2S † | 2Sx1S | 88.25 | PASS |
| 88.25 | **cmp** | 2,2,2 | 4 | 2 | 2Sx2S † | 2Sx2S † | 88.25 | PASS |
| 88.25 | **cmp** | 4,2,1 | 5 | 2 | 4Sx2S | 2Sx1S | 88.23 | **FAIL** 43.3% |
| 88.25 | **cmp** | 2,2,4 | 6 | 2 | 2Sx2S † | 2Sx4S | 88.25 | **FAIL** 50.0% |
| 88.25 | **cmp** | 4,2,2 | 6 | 2 | 4Sx2S | 2Sx2S † | 88.23 | **FAIL** 43.3% |
| 165.00 | rdr | 1,2,1 | 2 | 2 | 1Sx2S | 2Sx1S | 165.00 | PASS |
| 165.00 | rdr | 1,4,1 | 2 | 4 | 1Sx4S | 4Sx1S | 165.07 | PASS |
| 165.00 | rdr | 1,2,2 | 3 | 2 | 1Sx2S | 2Sx2S † | 165.00 | PASS |
| 165.00 | rdr | 1,4,2 | 3 | 4 | 1Sx4S | 4Sx2S | 165.07 | PASS |
| 165.00 | rdr | 1,2,4 | 5 | 2 | 1Sx2S | 2Sx4S | 165.00 | **FAIL** 50.0% |
| 165.00 | rdr | 1,4,4 | 5 | 4 | 1Sx4S | 4Sx4S | 165.07 | PASS |
| 176.50 | **cmp** | 1,1,1 | 2 | 1 | 1Sx1S | 1Sx1S | 176.50 | PASS |
| 176.50 | **cmp** | 1,1,2 | 3 | 1 | 1Sx1S | 1Sx2S | 176.50 | **FAIL** 50.0% |
| 176.50 | **cmp** | 2,1,1 | 3 | 1 | 2Sx1S | 1Sx1S | 176.50 | **FAIL** 46.6% |
| 176.50 | **cmp** | 1,1,3 | 4 | 1 | 1Sx1S | 1Sx3S | 176.50 | **FAIL** 66.6% |
| 176.50 | **cmp** | 2,1,2 | 4 | 1 | 2Sx1S | 1Sx2S | 176.50 | **FAIL** 50.4% |
| 176.50 | **cmp** | 3,1,1 | 4 | 1 | 3Sx1S | 1Sx1S | 176.50 | **FAIL** 61.6% |
| 176.50 | **cmp** | 1,1,4 | 5 | 1 | 1Sx1S | 1Sx4S | 176.50 | **FAIL** 74.9% |
| 176.50 | **cmp** | 2,1,3 | 5 | 1 | 2Sx1S | 1Sx3S | 176.50 | **FAIL** 84.0% |
| 176.50 | **cmp** | 3,1,2 | 5 | 1 | 3Sx1S | 1Sx2S | 176.50 | **FAIL** 81.6% |
| 176.50 | **cmp** | 4,1,1 | 5 | 1 | 4Sx1S | 1Sx1S | 176.47 | **FAIL** 69.9% |
| 176.50 | **cmp** | 1,1,5 | 6 | 1 | 1Sx1S | 1Sx5S † | 176.50 | **FAIL** 79.9% |
| 176.50 | **cmp** | 2,1,4 | 6 | 1 | 2Sx1S | 1Sx4S | 176.50 | **FAIL** 74.9% |
| 176.50 | **cmp** | 3,1,3 | 6 | 1 | 3Sx1S | 1Sx3S | 176.50 | **FAIL** 66.6% |
| 176.50 | **cmp** | 4,1,2 | 6 | 1 | 4Sx1S | 1Sx2S | 176.47 | **FAIL** 73.3% |
| 176.50 | **cmp** | 5,1,1 | 6 | 1 | 5Sx1S † | 1Sx1S | 176.50 | **FAIL** 73.3% |

**13 usable, 18 corrupt. 31 of 31 agree with `R <= C and W <= C`; 0 disagreements.**

**The DFB columns explain why this defect survived upstream — and it is not a missing shape.**
`DMTensixTest1xDFB4Sx1S`, `2Sx1S`, `3Sx1S`, `6Sx1S`, `6Sx2S` and `4Sx2S` all exist and pass, and those
are precisely the shapes `4,1,1`, `2,1,1`, `3,1,1`, `5,1,1`(≈), `4,2,*` corrupt here. The gap is not
coverage but **verification**: no DM→Tensix test looks at the delivered bytes. Nor would a hang or
timeout catch it — our corrupt runs complete at the same marginal cyc/tile as the clean ones, because
the credit count is right and only the payload is wrong. † correlates with nothing: `2,2,1` and `2,2,2`
carry uncovered `2Sx2S` endpoints and **pass**, while every covered `nSx1S` shape at `C=1` fails.

**The model is now exact, not approximate: 31 of 31 within 0.04%.** The `pred` column and the `marginal`
column agree everywhere, so there is nothing left to explain — the four ~6% "misses" in the previous
version were the two-point artifact, not a real overlap effect.

**Read the `C=1` block as one result.** Fifteen configs, DM cores rising 2 → 6, marginal pinned at
**176.47–176.50 — a spread of 0.017%**. Adding four DM cores at `C=1` is worth **1.000x**, measured.
That block is why the compute term is identified (slide 8) and why the win is compute-led (slide 7).

**Corrupt timings are still timings** — the trip count and the tensor written are unchanged, only the
data is wrong. And they cost nothing measurable: the corrupt `C=1` configs sit at 176.47–176.50 against
the one clean `C=1` config at **176.50**.

---

## 6. The whole space in one figure

![All 31 legal (R,C,W) configs vs measured marginal cyc/tile](rcw_space.png)

**Three things the tables above do not show at a glance.** The `C=1` panel is a solid wall: 15 configs,
DM cores rising 2 → 6 across and up, marginal pinned between 173.8 and 176.7 — spending the entire DM
budget buys nothing without Neos. Correctness improves monotonically with `C` (**1 of 15** bit-exact at
`C=1`, 4 of 8 at `C=2`, **8 of 8** at `C=4`), which is the slide-2 defect seen from the other side. And
the single green cell sits at the *edge* of the legal region — `4,4,2` has no slack in any direction.

**Blank cells are illegal, not untested:** `R+W > 6` cuts the upper-right triangle, and the STRIDED
ratio rule empties the `R=3` and `R=5` columns. Hatching marks corrupt output, not slowness.

Regenerate with `python debug/attrib/plot_rcw_space.py <this-dir>/rcw_space.png`; the data is
asserted against this deck's own table.

---

## 7. What it tells us

- **The win is compute-led and DM-enabled, and it is exactly at the ceiling.** `marginal =
  max(165.0/R, 176.5/C, 83.5/W)`, per-stage cost **compute 176.5 > reader 165.0 > writer 83.5**
  cyc/tile. Two single-axis steps, both measured:

  | step | held at | ratio |
  |---|---|---|
  | DM cores **2 → 6** | `C=1` | **1.000x** — nothing, to three digits |
  | `C 1 → 4` | `R=4, W=2` | **4.000x** — the `1/C` limit, exactly |

  `1.000 × 4.000 = 4.00`. **All six DM cores are worth nothing until the Neos are there** — at `C=1`
  the marginal is 176.47–176.50 across 15 configs spanning 2 → 6 DM cores, a spread of 0.017%.
- **Per-axis attribution is path-dependent.** The same `C 1→4` step is worth **1.07x** taken first (at
  `R=W=1` the reader caps you at 165) and **4.000x** taken last. "What did the Neos buy" has no
  order-independent answer; "is every term below target" does.
- **An axis is worth 2x only while it binds, and exactly 1.000x when it does not** — single-axis steps
  between bit-exact endpoints:

  | step | held at | ratio | why |
  |---|---|---|---|
  | `R 1→2` | `C=4, W=2` | **2.000x** | reader binds throughout |
  | `R 2→4` | `C=4, W=2` | 1.871x | *partial* — compute takes over at 44.12 |
  | `W 1→2` | `R=4, C=4` | 1.893x | *partial* — same reason |
  | `R 2→4` | `C=4, W=1` | **1.000x** | writer binds at 83.5 |
  | `W 1→2` | `R=1, C=4` | **1.000x** | reader binds at 165 |

  The two partial steps are the roofline working correctly: at `4,4,2` compute binds at 44.12, so
  neither DM axis can deliver its full 2x.
- **Along the balanced frontier, scaling is exactly linear in hardware.** `2,2,1` (3 DM, 2 Neo) 88.25 →
  `4,4,2` (6 DM, 4 Neo) 44.12 — **2x the engines, 2.0002x the throughput**.
- **`4,4,2` is not "use everything" — it is the exact match to 4 Neos, with zero slack.** Four Neos put
  the compute floor at `176.5/4 = 44.12`. To stay under it the reader needs `R >= 165/44.12 = 3.74 → 4`
  and the writer `W >= 83.5/44.12 = 1.89 → 2`. That is `R+W = 6` — **precisely the user DM budget, fully
  consumed, nothing spare.** The structure generalises to other binary ops; the constants do not, and a
  compute-heavier op would need more Neos than exist.
- **Hardware validation is now the gating question, not feasibility.**

---

## 8. The cost model — exact, and only the corrupt configs could pin it

`marginal = max(165.0/R, 176.5/C, 83.5/W)` cyc/tile. **31 of 31 configs within 0.04%.**

| term | single-thread cyc/tile | how it is pinned |
|---|---|---|
| reader | **165.0** | the `R=1` rows: `1,2,x` measure 165.00 |
| **compute** | **176.5** | **15 configs at `C=1`, spread 0.017% while DM cores go 2 → 6** |
| writer | **83.5** | `2,4,1` / `4,4,1` measure 83.53 with the writer binding |

**Measuring the corrupt configs is what identified the compute term** — the legal space cannot, because
correctness forces `C >= max(R,W)`, so `176.5/C <= 176.5/R` and the compute term never binds there. The
18 corrupt configs are the **only** region with `C < max(R,W)`. Wrong data does not mean wrong timing:
the compute trip count is `num_tiles / C` regardless of what flows, and the writer writes a full tensor
either way, so all the work still happens.

| `C=1`, 15 configs | |
|---|---|
| marginal | **176.47 – 176.50**, spread **0.017%** |
| DM cores spanned | 2 → 6 (`R` 1→5, `W` 1→5) |
| DM terms spanned | 165.0 down to **41.25** |

`4,1,2` is the decisive row: reader term 41.25, writer term 41.75 — both 4x below — and it still
measures **176.47**. So 176.5 is the compute stage, cleanly separated from data movement, and **compute
is the most expensive stage**, above the reader's 165.0.

**`Cc = 176.5`, fitted over the linear region on three points with a max residual of 3.3 cycles.** It is
identifiable only from the 18 corrupt configs, because they are the only region where the compute term
binds; a fit taken on `1,1,1` alone is unsound, since reader and compute sit within 6% of each other there.

**The roofline is a hard `max()` with no overlap bonus** — every config sits on its binding term to within
0.04%, so there is no balance-point bonus to model. **And corruption costs nothing measurable:** the
corrupt `C=1` configs sit at 176.47-176.50 against the clean `1,1,1` at 176.50.

**Adding Neos that do not bind still costs raw performance** at small shapes: more Neos raise the
prologue (767 at `1,1,1` → 1106 at `4,4,2` → 1491 at `1,4,4`), which is why raw@40 lags the asymptote.

**Open:** no *legal* config isolates `C` at `R=4` — `4,2,1` and `4,2,2` are both corrupt. They measure
88.23, i.e. exactly `176.5/2`, so the model covers them; but that is corrupt-source data, and a working
`4,2,2` remains the only clean discriminator.

---

## 9. Roadmap — restructured into milestones

Supersedes the flat `F1–F12` list on 08-21 slide 10. **`M#.#` is the sequence; `F#` is the stable
identity** — F-labels are cross-referenced throughout the design doc and never get renumbered, so both
columns stay. (`F#` in the review-findings doc is an unrelated namespace.)

| M# | F# | item | note |
|---|---|---|---|
| **1.0** | — | **phase-1 slice + thread sweep — DONE** | bf16 `add`, TILE, interleaved, no bcast, even divisibility. `4,4,2` optimum, criterion cleared |
| **1.1** | F1 | **uneven tile counts — DONE 2026-09-04** | gate deleted; one-line compute-kernel fix; 34/34 bit-exact incl. zero-work threads. Cannot land while main hangs |
| 1.2 | F2 | rest of FPU op set (sub, mul) | `multiply` is fidelity-dependent |
| 1.3 | F3 | sharded / borrowed operands | zero NoC ⇒ isolates the compute levers |
| 1.4 | F4 | mixed layouts | falls out of F3 |
| 1.5 | F5 | fp32 + SFPU (divide) | **bit-exact oracle expires here** |
| 1.6 | F7 | activations (lhs/rhs/post) | re-measure cyc/tile; `binary_tiles_init` added cost since our branch point |
| **2.0** | — | **milestone 2 — once F7 lands** | dtype/layout/memory/activation-complete for whole-tile operands |
| 2.1 | F13 | **outer-dim broadcast** | **a regression, not a feature** — `kernels_dfb/` has it, `kernels_qsr/` lost it; until it lands every broadcast `add` falls back |
| 2.2 | F8 | subtile broadcast ROW/COL/SCALAR | gated on `#51291` |
| 2.3 | F9 | mixed broadcast | keep the ROW-via-LLK / COL-via-reader-fill hybrid |
| 2.4 | F10 | tensor-scalar | writer fills `in1` once |
| 2.5 | F14 | **per-operand reader allocation** | **emulator-only** — no roofline gain (per-core reads are `T/2` either way); the case is DRAM/NoC locality, which craq-sim cannot price. Hypothesis: tile-split pairs `in0[k]`/`in1[k]` on the **same bank**. Proportional allocation matters from F4 (mixed layouts) onward, not just broadcast. STRIDED rule limits splits to `p in {1,2,4}` at `C=4` |
| 2.6 | F15 | **in-flight concurrency** (`implicit_sync`, ring depth, batching) | **emulator-only, same campaign as F14** — craq-sim says <=1.10x / 1.02x / 1.08x but two of three are **floors**: latency-hiding levers, and craq-sim has no latency. One axis, not three (`capacity >= 2n`). Writer batching is a known negative |
| **3.0** | — | **milestone 3 — once F10 lands** | broadcast-complete; the rest is the long tail |
| 3.1 | F11 | row-major | 16-byte RM shard-width alignment |
| 3.2 | F12 | where / quantization / int32 | own kernel families; int32 blocked on the DFB-compute bug |
| 3.3 | F6 | MX formats | **last**; cost is dominated by work outside this op |

**The boundaries are where the op changes kind.** Milestone 1 is "the same op, wider" — more dtypes,
layouts, memory configs, fused activations, but always whole tiles addressed one-to-one. Milestone 2
changes how a tile is *addressed* (broadcast). Milestone 3 is the long tail: a different physical
layout, op families with their own kernels, and a format TTNN cannot represent yet.

**F6 last** is a decision, not a derivation — explicit call 2026-08-22; its cost sits outside this op.

**F13 opens Milestone 2 because it is a regression, not a feature.** `SubtileBroadcastType::NONE` compares H
and W only, so outer dims are a separate axis that a `no_bcast` kernel still has to carry — the shared
`kernels_dfb/` path does, and `kernels_qsr/` lost it when Task 4 collapsed the stride cascade (an
unmandated narrowing of the copy; design §3.4.1). Correctness is safe today — the gate rejects those
shapes, the fallback runs, verified bit-exact — but **every broadcast `add` gets zero benefit from the
native path**, and leading-dim broadcast is common (bias add, residual with a unit batch dim). That
caps the reachable model-level win. It sits with the other broadcast work rather than being ranked
against it — which is also what makes Milestone 3's "broadcast-complete" literally true.

**Cross-cutting, before any of this is production-ready:** validate fast dispatch for DFB-bearing specs,
then the hardening pass — strict gate, CI wiring, env-var default flip, knobs into the program hash.

---

## 10. Caveats, and open

- **craq-sim models no contention** — 4.00x is an upper bound, and this op is DM-bound, precisely what
  contention degrades. Not a silicon forecast.
- Numbers are bf16 `add`. A compute-heavier binary op shifts the optimum toward higher `C`.
- **Task 6's remaining two gates ran 2026-08-28 and both pass.** Work-split: the `RD_BAR` sum per core is
  **320 at both 1 and 4 reader threads** — a duplicating implementation would report 4x — with `max/min`
  across threads **1.000**, so work is genuinely split in equal shares. Stall signature: `unpack`, `pack`
  and `sfpu` stalls are **exactly 0**, so the bottleneck did not move to output-DFB backpressure; per
  active-core-cycle, semaphore stall density *fell* 34% while span fell 2.70x. Raw record:
  `debug/attrib/milestone1_results.md`.
- Both gates' own thresholds turned out to be unusable as written — one keys on a stale constant, the
  other divides by an undefined core count and would reject the baseline. Replaced with equivalents that
  do not depend on either; the plan records the fix.

**Measurement protocol this week established.** Fit the marginal over at least three tile counts in a
verified-linear region and check the successive differences are equal; build every golden from the
operands as the device holds them rather than from intended values; and check any result against a
theoretical bound where one exists. Design §2.1 carries these as requirements.

---

# ► Week of 2026-08-21 — design + measurement phase

Period covered: design + measurement. **Implementation plan not yet written — deliberately.**

---

## 1. TL;DR

- **Designed** a Quasar-native `binary_ng` program factory (multi-DM, multi-Tensix) behind the existing
  `program_factory_t` variant seam, so the current functional path stays live as a reference arm.
- **Measured a baseline** on craq-sim: **213.72 cyc/tile**, using **2 of 6** DM cores and **1 of 4** Tensix
  engines, with the one active Tensix ~96 % stalled. That idle hardware is the entire premise of the project.
  (This is the `metal_v2` arm — a Milestone-0 **history record**; later perf gains all divide by the
  native factory at `1,1,1`, never by this.)
- **Investigated what craq-sim can and cannot measure** — and it changed the plan, twice.
- **Measured every tunable knob reachable without new code.** Each moves craq-sim by **<=1.10x**
  (~1.17x combined) — **but that is a statement about craq-sim, not about the knobs.** Two of the three are
  latency-hiding levers, and craq-sim has no latency to hide, so it cannot value them at all. Their real size
  is **unknown** and only the emulator can settle it.
- => **The project rests on the two levers that cannot be measured without building the factory:**
  - **DM thread count** (`R`, `W`) — 2 of 6 cores used today. Testable as soon as the factory exists.
  - **Compute thread count** (`C`) — 1 of 4 Tensix engines used today. Blocked on an upstream LLK fix
    (tt-llk #1678) expected imminently, so treat it as available for planning purposes.

---

## 2. Deliverables

| artifact | lines | what it is |
|---|---|---|
| `QUASAR_NATIVE_RESEARCH.md` | 911 | **Research base.** The machine, the Metal 2.0 API surface, prior-art file map, measured baseline, craq-sim capability, landmines, lever ranking |
| `QUASAR_NATIVE_DESIGN.md` | 1197 | **Design spec.** Scope, success criteria, architecture, dataflow, failure modes, correctness, measurement protocol, roadmap |
| measurement harness | — | Depth sweep, batch sweep, profiler summarizer — all under `debug/` |
| **tt-llk issue #1678** | — | Filed upstream: `bfd_state` shared across all 4 Neos — blocks `compute_threads > 1` |

Both documents went through **two review rounds, 10 specialist passes**. Four blockers found; five findings
independently confirmed by two reviewers each. All evidence archived with file:line citations.

---

## 3. Measured baseline — the `metal_v2` arm (Milestone 0), craq-sim, 32x40 tiles, bf16 DRAM-interleaved `add`

| quantity | value |
|---|---|
| per-cluster kernel span | 8549 cycles -> **213.72 cyc/tile** |
| marginal cost | **187.0 cyc/tile**, exactly linear across 5 shape rungs |
| DM cores active | **2 of 6** (`DM2` reader, `DM3` writer) |
| Tensix engines active | **1 of 4**; within it TRISC3 runs **16 cycles** — SFPU wholly unused |
| Tensix utilisation | ~**96 % stalled** — compute is starved, not busy |
| all-operands-sharded roofline | 64.6 cyc/tile => **3.31x headroom** (craq-sim basis) |

**History record (Milestone 0): every number in this table is the `metal_v2` factory**, the arm
Milestone 0 reproduces. **Perf gains are never computed against it** — the baseline for every gain in
these docs is the **native factory at `1,1,1`** (176.00 marginal / 7813 span @ 40 t/c). The
187.00 → 176.00 delta is the F13 stride-cascade price (research §5.0.2), a cost record, not part of any
gain.

**Reproducible and deterministic:** bit-identical across runs (sim clock 17934), ~15 s per run. Re-verified
after every experiment.

---

## 4. craq-sim: what it can and cannot measure

Verified against simulator source, not assumed.

**Faithful:** instruction issue on DM cores (1/cycle), thread parallelism, determinism, DM cache coherence.

**Not modelled:** NoC/DRAM transfer cost (a host `memcpy` inside the issue instruction), barrier cost
(pre-satisfied), contention or queueing of any kind, store ordering, cache *timing*.

**The one-sentence rule that predicts every bias:**

> **craq-sim over-reports levers that remove instructions and under-reports levers that hide latency.**

Consequences we hit in practice:

- Three traps that produce *wrong* conclusions rather than missing ones (ring-full instruction replay faking a
  depth knee; deterministic races making a green multi-thread run evidence-free; no-contention linear scaling).
- **And one in our own harness**: the profiler CSV has no dispatch key, so two dispatches in one process
  leave a *per-cluster blend* of two shapes — now guarded against.
- Tensix is **not** 1 instr/cycle (up to 3), so compute-thread sweeps sit on a different scale than DM sweeps.

---

## 5. Knobs: what craq-sim can and cannot value

| lever | craq-sim result | is that a bound? | emulator expectation |
|---|---|---|---|
| DFB call batching (reader, n=2) | **1.08x** | **neither — two-sided** | unknown — removes instructions (sim = upper) *and* raises NoC concurrency (sim = lower) |
| `implicit_sync` | **<=1.10x** | **a floor, not a ceiling** | **potentially large** — a barrier is free on craq-sim, a real stall on the emulator |
| `entries_per_thread` (ring depth) | **1.02x** | **a floor, not a ceiling** | **potentially large** — depth hides transfer latency; craq-sim has none |
| **DM threads `R`, `W`** | **unmeasured** | will be a **ceiling** | <= sim — NoC ports, DRAM bank conflicts, txn-id rendezvous, DM0 ISR |
| **Compute threads `C`** | **unmeasured** | will be a **ceiling** | <= sim; blocked on tt-llk #1678, expected imminently |

*Reading the third column:* a **ceiling** means craq-sim flatters the lever and the emulator will be no
better. A **floor** means craq-sim cannot see the lever's real mechanism, so the emulator could be much
better. **So the two small numbers in rows 2-3 are not verdicts on those knobs — they are the simulator
declining to answer.**

Batching the **writer** is actively negative — `wait_front(n)` delays `pop_front` and starves compute of ring
slots, degrading monotonically to 1.02x at n=8.

---

## 6. The three small levers are actually one lever

| knob | what it controls |
|---|---|
| `entries_per_thread` -> `capacity` | how many slots exist to receive in-flight data |
| batch `n` | how many transfers are issued before waiting |
| `implicit_sync` | removes the wait entirely |

All three are facets of **how many tile transfers are in flight at once**, and they are *not* independent —
`capacity >= 2n` is required for any overlap at all.

**So it is not three coincidences that all three measured ~nothing on craq-sim. It is one cause:** in-flight
concurrency cannot pay when a transfer costs zero cycles. On the emulator they are one lever with three knobs,
and they may be large.

=> Emulator campaign sweeps **in-flight concurrency as one axis**, thread counts as the other.

---

## 7. Expected performance

**On craq-sim**, the three measured knobs compose to **~1.17x** (they overlap, so they do not multiply).
**That is the craq-sim figure only, and it is a floor for two of the three** — do not present it as the
expected gain from those knobs on hardware.

Given that, **thread parallelism** — `R`, `W`, and `C` once unblocked — must supply:

| target | threads must deliver |
|---|---|
| **gate floor, 1.54x** | **1.32x** |
| stretch, 2x | 1.71x |
| craq-sim ceiling, 3.31x | 2.83x |

**Reasonable to expect the gate.** Going 2 DM cores -> 6 is 3x more resource, so 1.32x is **under half of
ideal scaling on the DM side alone** — before counting the 3 idle Tensix engines. Nothing measured argues against threads — the measurements
eliminated the *alternatives*, which concentrates the hypothesis rather than weakening it. The founding premise
is untouched.

**Reasons for caution, both unmeasured:** at depth 2 the two DM cores measurably **ping-pong**, so if threads
do not break that serialization they disappoint too; and any craq-sim result is an upper bound for the
emulator.

---

---

## 8. Kill criterion

> **The criterion is on thread parallelism as a whole.** If `R`/`W` *and* `C` together fail to clear ~1.3x on
> craq-sim (total under ~1.5x), stop and report that rather than proceeding to the 12 roadmap follow-ons.

- `R`/`W` is measurable first and gives the early read. A poor `R`/`W` result alone is a **pause**, not a kill,
  because `C` is the other half of the same premise and unblocks shortly.
- **Asymmetric — only the stop direction is sound.** craq-sim applies no contention, so it is an *upper* bound
  for threads: a craq-sim failure is a real failure, but a craq-sim pass proves nothing about the emulator. Use
  it to stop early, never to declare success.

This reframes the project: not "build a 3.3x native path" but **"determine whether multi-engine threading is
worth anything on this op shape"** — one open question, cheap to answer, with a defined exit.

---

## 9. Status and next step

**Done:** research base, design spec (v3, measured), measurement harness, baseline, craq-sim capability study,
all reachable knobs measured, one upstream LLK issue filed.

**Not done, deliberately:** the implementation plan. Every knob was measured first, because several
early estimates were overturned once run — writing the plan earlier would have baked those in as premises.

**Next:**

1. Implementation plan. Commit 1 is a mechanical copy of the existing factory plus the three deviations that
   make it compile, link and be selectable; Milestone 0 reproduces 8549 to prove the copy is faithful.
2. **Milestone 1 is the thread sweep** — `R`/`W` immediately, `C` as soon as #1678 lands. This is the first
   question the implementation answers, not the last, because it either validates the premise or triggers the
   kill criterion.
3. One emulator campaign afterwards, sweeping in-flight concurrency and thread counts — the only place the
   latency-hiding levers can be valued at all.

---

## 10. Roadmap after phase 1

**Phase-1 admitted slice:** no-broadcast tensor-tensor, TILE 32x32, **bf16**, FPU `add`, all three operands
**DRAM-interleaved**, no activations, **even divisibility**. Everything below widens that.

**All twelve are gated on the kill criterion (slide 8).** If thread parallelism does not pay, none start.

**Label order is not priority order** — labels are stable identifiers, so they do not get renumbered when
priority changes. **F6 (MX formats) is the lowest priority of the twelve; do it last.** And **F1 is not
Milestone-0 or phase-1 work** — it is the first follow-on *after* the criterion is cleared.

| # | follow-on | why there |
|---|---|---|
| F1 | **Uneven tile counts** | First follow-on once the criterion is cleared — every later phase inherits the restriction otherwise. Explicitly out of Milestone 0 / phase 1 |
| F2 | **Rest of FPU op set** (subtract, multiply) | Gate widening; `multiply` is fidelity-dependent |
| F3 | **Sharded / borrowed operands** | Zero NoC, so it isolates the compute levers. High model relevance (ResNet residual add) |
| F4 | **Mixed layouts** | Falls out of F3; kernels already parameterise per operand |
| F5 | **fp32 + SFPU ops** (divide) | New compute path; the bit-exact oracle expires here |
| F6 | **MX formats** — **lowest priority, do last** | Quasar replaces all BFP with MX. Needs a new TTNN `DataType` *and* IDMA gasket support — the one follow-on whose cost is dominated by work outside this op |
| F7 | **Activations** (lhs/rhs/post) | Compute-side self-loop DFBs, credit-balanced by construction |
| F8 | **Subtile broadcast** ROW/COL/SCALAR | `ALL` consumer access + remapper fan-out; gated on a release-fence fix |
| F9 | **Mixed broadcast** | Preserve the ROW-via-LLK / COL-via-reader-fill hybrid |
| F10 | **Tensor-scalar** | Writer fills `in1` once; same fence dependency as F8 |
| F11 | **Row-major** | Quasar needs explicit 16-byte RM shard-width alignment |
| F12 | **where / quantization / int32** | Furthest out; int32 blocked on a DFB-compute bug |

**Cross-cutting, before any of this is production-ready:** validate **fast dispatch** for DFB-bearing specs,
then the hardening pass — strict gate, CI wiring, env-var default flip, knobs into the program hash.

---

## 11. Risks and open items

| item | status |
|---|---|
| `compute_threads > 1` | Blocked on tt-llk #1678 (`bfd_state` shared across Neos), **expected imminently**. Does not block phase 1 — `R=4/C=1/W=2` is legal and already uses the full DM budget |
| `TT_METAL_LLK_ASSERTS` at `C>1` | Unreliable — `llk_tdma_guard` is also shared across Neos, so the recommended bring-up tool degrades exactly when multi-Tensix debugging needs it |
| Ceilings craq-sim cannot show | Shared txn-id rendezvous and DM0's single ISR core serving every credit — invisible on craq-sim and stressed exactly by `R=4/W=2` |
| Data verification gap | **No test in the tree data-verifies a multi-thread STRIDED producer.** Our oracle would be the first, with no independent cross-check |
| Emulator campaign | Entirely unexercised. Access is limited, so it must be scoped tightly and run once |
| Uneven tile counts | Out of phase-1 scope by decision; even divisibility is also what makes the DFB drain safe, so lifting it is real work, not a relaxation |
