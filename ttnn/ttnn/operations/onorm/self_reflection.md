# Self-reflection — onorm (advisory; nothing here is auto-applied)

## Summary

Blind final is **11/11 green, 0 failures, 0 hangs** (`golden_blind_final/golden_results.txt`):
5 golden cells + 6 non-registry numerics cells, **0 translated cells** (onorm has no
`test_translated.py`). Every phase (`golden_phase0` → `golden_refinement_3`) was also 11/11, so
there is **no failure cluster to mine** — the findings below are therefore about the *measurement*,
not about broken code. The important ones are framework-level: the graded universe is a **single
axes cell** (`INPUT_TAGGERS = {}`, `TARGET` = one `(bf16, TILE)` cell) with **no `LOOSE_CASES` at
all**, so (a) the run's most consequential decision — flipping the prompt-mandated
`fp32_dest_acc_en=True` default to `False` — landed with **zero graded coverage on either side**,
and (b) the refinements' headline wins (16.1× from the cross-core re-tile, 1.2× from R1b/R3) are
**not regression-gated by anything the pipeline grades**. Op-level quality looks high; the exposure
is in the test universe and in three helper/reference docs that misdirected the perf work.

---

## 1. Golden coverage → `eval/golden_tests/onorm/feature_spec.py`

**F1.1 — `fp32_dest_acc_en` is an axis-blind facet, and the run flipped it.** (most important)
*What*: `.claude/references/precision_convention.md` declares, universally for any op exposing
`compute_kernel_config`, that precision is gated on **two** axes and that `SUPPORTED` keys on both.
onorm's `TARGET` (feature_spec.py:43) and `SUPPORTED` (onorm.py:49) carry only `dtype`/`layout`;
`helpers.TOLERANCES` (helpers.py:82) is keyed by dtype alone; `axes.py:classify_call` accepts
`compute_kernel_config` and never tags it. Refinement 1b then moved the `None` default to
`fp32_dest_acc_en=False` (onorm.py:143) as a *documented deviation* from `eval/prompts/onorm.txt:124`
("accumulate the sum-of-squares in fp32 in DST (fp32_dest_acc_en=True)"). Both configurations are
legal call patterns and compile different kernels (`_dest_tile_limit()`,
onorm_program_descriptor.py:430, returns 4 vs 8 → different DEST windows and CB asserts), yet **no
graded cell exercises either one explicitly**; the only evidence is the implementer's private guard
set ("0.9999× fp32on-override", breadcrumbs R1b `measurement`).
*Evidence*: `precision_convention.md:9-17` "Precision is gated on **two independent axes** …
`SUPPORTED` keys on both"; `verification_report.md:424` "**`fp32_dest_acc_en=True` is a
precision-rule requirement, not a free knob** … Do not flip it silently."
*Recommendation*: add `fp32_dest_acc_en` to `TARGET` + the op's `SUPPORTED`, key `TOLERANCES` by
`(dtype, fp32_dest_acc_en)`, and tag it in `axes.py` from `default_compute_kernel_config()` per
`precision_convention.md:78-81`. Minimum alternative if the axis is refused: two `LOOSE_CASES`
entries at `((1,32,HV,V),(1,32,FLAT),(1,1,1,V))` pinning `extras={"fp32_dest_acc_en": True}` /
`False` — this also requires `run_onorm` to forward `extras` into a `compute_kernel_config` (today
`extras` is accepted and ignored, helpers.py:98).
*Confidence*: high.

**F1.2 — no graded cell ever leaves the `blocks_per_group == 1` regime.**
*What*: the work unit is a 32-token block; `_retile_group_cores` (onorm_program_descriptor.py:557)
picks the group size `g` from `num_token_blocks` vs the grid, and `_grid_assignment` (…:674-698)
spreads blocks over groups with a `base`/`remainder` split. The five golden shapes are
1 / 2 / 4 / 20 / 4 blocks (op_design.md:57) and land on 32/64/64/80/64 cores (`test_results.json`
`device_num_cores`), i.e. **every graded cell gets exactly one block per group and a zero
remainder**. The serial multi-block loop *and* the uneven remainder split — the path the
descriptor's own model is written for ("B=8/T=640 has 160 blocks on 110 cores, so … 55 groups x 3
blocks", …:606-609) — are exercised only by op-side, non-graded tests
(`tests/ttnn/unit_tests/operations/onorm/test_onorm_retile_group.py:105`). Semaphore reuse across
iterations of that loop is exactly what a green graded suite would not catch. The facet is
**invisible to the axis model** (`INPUT_TAGGERS = {}`, onorm.py:39), so all five golden cells
collapse to one axes tuple and the region reads as "covered".
*Evidence*: `onorm_program_descriptor.py:636` `num_groups = min(num_token_blocks, total_cores // g)`;
`golden_blind_final/test_axes.json` — all five nodeids map to the identical
`{dtype: BFLOAT16, layout: TILE}`.
*Recommendation*: add a `LOOSE_CASES` entry just past grid saturation, e.g.
`{"inputs": ((8,448,HV,V),(8,448,FLAT),(1,1,1,V)), "dtype": ttnn.bfloat16, "layout": ttnn.TILE_LAYOUT}`
(112 blocks > the 110-core grid; `(8,640)`/160 blocks if you prefer the already-measured shape),
**and** consider promoting the facet to an `INPUT_TAGGERS` axis — e.g.
`token_blocks ∈ {"single","sub_grid","over_grid"}` bucketed on `B*T/32` against a documented nominal
grid — so the regime is visible in the registry instead of an accident of `B`,`T`.
*Confidence*: high (that the regime is ungraded); med (on the exact boundary shape — the threshold
is device-dependent).

**F1.3 — the suite has no `LOOSE_CASES`, so nothing perf-related is gated.**
*What*: `feature_spec.py` (79 lines) defines `TARGET`/`INVALID`/`INPUTS` only, and `test_golden.py`
has no `test_op_loose`. Consequence: R2's 16.1×/13.3×/8.5× on the under-filled shapes and R3's
1.205× survive only as changelog prose — a change that reverted the cross-core re-tile would still
be 11/11 green (blind rows do carry `device_kernel_ns`, but nothing asserts on it). It also starves
the framework's own perf machinery: `blocking-verifier.md:303` and `blocking-perf-coordinator.md`
both take their mandatory focus shape from a perf-flagged `LOOSE_CASE`, so with none present the
queue free-selected its shapes.
*Evidence*: `changelog.md:453` "Refinement 2 — Cross-core re-tile: stop leaving 108 of 110 cores
idle at small `T`"; `test_results.json` `device_kernel_ns` 12293/14573/22185/61499/22790, ungated.
*Recommendation*: add 2–3 loose cases in the `_perf_case` shape used by
`eval/golden_tests/rms_norm/feature_spec.py:312`, `extras={"achievable_ns": …, "measured_on":
"blackhole_p150b"}` — one under-filled (`B=1,T=32`), one target (`B=1,T=640`, flagged
`# attention: PERF FOCUS`), one saturated (`B=8,T=640`) — using the measured post-R3 numbers as the
guarantee.
*Confidence*: high.

---

## 2. SUPPORTED honesty → `SUPPORTED` / `EXCLUSIONS` in `onorm.py`

**F2.1 — the declaration is honest; no fix / demote / promote is warranted.**
`golden_blind_final/verifier_report.json`: `supported_pass=5, supported_fail=0, xpass_drift=0,
xfail_expected=0, invalid_unexpected=0`; `registry_snapshot.json` shows
`supported={dtype:[BFLOAT16], layout:[TILE]}, exclusions=[], invalid=[]`, and `TARGET` equals it.
No over-claim to demote, no under-claim to promote. Nothing proposed. *Confidence*: high.

**F2.2 — the honesty report sees only 5 of 11 graded rows, and would absorb a failure.**
*What*: 6/11 rows land in `no_axes_found` because `test_regression.py:22` imports the raw op
(`from ttnn.operations.onorm import onorm`) instead of the observe-only wrapper `helpers.py:24` uses
(`from …axes import observed as onorm`). Sibling suites do the same
(`rms_norm/test_regression.py:24`), so this is a **framework convention**, not an onorm slip. But
`verify_supported.verify()` `continue`s on a missing axes sidecar *before* categorizing
(verify_supported.py:268-274), so a **failed** untagged row shows up only as a trailing
"# Warning: N test(s) had no axes sidecar entry" and contributes 0 to `supported_fail`. Run-level
pass/fail is still caught by `golden_results.txt`; the honesty view is what goes blind.
*Evidence*: `verifier_report.json` `"no_axes_found": 6`; `verify_supported.py:399` "likely
test_regression.py — not driven by the registry".
*Recommendation*: split `no_axes_found` by status in `verify_supported` (a counted, listed
`no_axes_failed`), and/or route `test_regression.py` through `axes.observed` so numerics rows carry
the `(dtype, layout)` cell like every other row.
*Confidence*: high.

**F2.3 — `validate()` cannot refuse a precision cell it does not support.**
*What*: `validate()` (onorm.py:154) takes `compute_kernel_config` and never reads it, so any
`fp32_dest_acc_en` / `math_fidelity` is silently accepted with tolerance data for neither.
`precision_convention.md:53-59` expects the op to *read* the caller's `fp32_dest_acc_en`;
`verification_report.md:433` already records the sibling foot-gun (`dst_full_sync_en=True` → 0.931×,
"The op accepts it silently").
*Recommendation*: pair with F1.1 — once `fp32_dest_acc_en` is a declared axis, gate on it in
`validate()` (both values SUPPORTED here). Declaration change, not a kernel change.
*Confidence*: med (the convention is universal, but this suite was authored single-axis
deliberately — a human should decide whether the convention or the narrow spec wins).

---

## 3. Helper / reference docs

**F3.1 — `perf_instrumentation.hpp` never says a zone includes its `cb_wait_front`.**
*What*: the header sells `MaybeDeviceZoneScope` as measuring "the wall-clock of the enclosing stage"
(perf_instrumentation.hpp:9-13) with no caveat that a zone wrapping a helper also wraps that
helper's `cb_wait_front` — so a *starved* phase reads as a *slow* phase and per-stage percentages
are occupancy, not attribution. The entire refinement queue was ranked off one such number
("P7b sigmoid 152.7 µs (**63.9 %**)", changelog.md:44); the implementer had to build a third
`SIGMOID_ENGINE="ablate"` value to learn what was real (244,495 → 92,212 ns, breadcrumbs R1
`device_print_observation`), after logging a `CORRECTION`: "the per-phase zone wraps each helper
INCLUDING its `cb_wait_front`, so a starved phase reads as a slow phase."
*Recommendation*: one line in the header doc — "A zone enclosing a helper also times its
`cb_wait_front`: per-stage ns is **occupancy**, not payload. Confirm any attribution by ablating the
stage's payload while keeping its sync scaffolding." (`verification_report.md:418` says this
op-locally; it belongs upstream.)
*Confidence*: high.

**F3.2 — `tilize_helpers.hpp` `StreamMode::PerTile` omits the same-kernel deadlock invariant.**
*What*: the doc says use PerTile "when a downstream consumer drains the tiled output incrementally"
(tilize_helpers.hpp:89-93, repeated 151-160) and lists constraints on the input stripe, pages and
arch — but never that the consumer must live on a **different kernel/thread**. When the consumer is
later code in the *same* compute kernel (the fused structure this op's prompt mandates), producer
and consumer share the three TRISCs and a 2-tile output CB deadlocks. The prompt prescribed exactly
that combination and the implementer had to file a soft-rule deviation.
*Evidence*: `changelog.md:85-89` "the rules prescribe `tilize<StreamMode::PerTile>` with a 2-tile
`cb_flat`, which would **deadlock** here because tilize and its consumer share the same compute
kernel and therefore the same three TRISCs"; `eval/prompts/onorm.txt` Rules: "`cb_flat` can be just
2 tiles".
*Recommendation*: add to the PerTile constraints — "the consumer must be a *different* kernel
(dataflow / other Tensix thread). If it is downstream code in the **same** compute kernel, the
output CB must still hold a full tile-row; PerTile with a small CB self-deadlocks."
*Confidence*: high.

**F3.3 — `master.md` T2 `compute_block_size` states both levers without their preconditions.**
*What*: the catalog claims reconfig-off is "up to **1.19×**" (master.md:121) and that coarser blocks
amortize per-phase overhead (…:108-118). Both inverted here: reconfig-off measured a **wash**
("all deltas <=1.4% inside 0.4-5.4% spread", breadcrumbs R3 lever 1 — the three threads run lockstep
behind the exchange, so compute-thread MMIO hides in stalls), and **finer** blocks won ("FINER wins,
opposite of catalog. n8->n2 and g64->g8 … Mechanism = pipeline fill depth (writer waits a whole gate
chunk of sigmoid)", breadcrumbs R3 lever 2; the coarse corner measured 0.785×/0.908×). The design had
already banked the 1.19× as a lamp (op_design.md:78).
*Recommendation*: two clauses in the entry — reconfig-off pays "only when the MATH/PACK thread is
the critical path; if the threads are sync-bound the MMIO hides and the lever is a wash"; the
block-size lever reverses "when a downstream consumer waits for a whole block (pipeline-fill latency
dominates amortization) — sweep both directions."
*Confidence*: high (measured, in-repo counterexamples).

**F3.4 — `numerical_stability_analysis_reference.md` §2.4 over-generalizes SFPU approx mode.**
*What*: "Each operation has an approximate mode (`math_approx_mode=true`) and a precise mode"
(…:101). For sigmoid the LLK reportedly ignores `APPROXIMATION_MODE` on both Blackhole and Wormhole
B0 (same 6-entry LUT), making a queued lead unbuildable by construction.
*Evidence*: `refinements/refinement_1_output.json` "**Lead 2 (`fast_and_approx`) — cannot be
built.** The LLK ignores `APPROXIMATION_MODE` for sigmoid on both Blackhole and Wormhole B0";
same claim in onorm.py:95-99.
*Recommendation*: add "Not every SFPU op honours `math_approx_mode` — some (e.g. LUT-based
activations) compile the same datapath either way. Check the op's `ckernel_sfpu_*.h` before pricing
an approx-mode lever."
*Confidence*: med (the LLK submodule is not checked out in this clone, so the claim rests on the
agent's source inspection, reported twice).

**F3.5 — the `fast_tilize` ↔ `dst_full_sync_en` coupling is a parenthetical.**
*What*: `tilize_helpers.hpp:116` mentions the fast path needs "half-sync dest mode" in passing; the
consequence is a silent perf cliff a *caller* can trigger via a public config field (measured
0.931× / 7 %, no correctness signal — changelog.md:53, verification_report.md:433).
*Recommendation*: promote to a warning — "`dst_full_sync_en=True` disqualifies `fast_tilize`; wide
tilize blocks fall to the slow path with no correctness signal."
*Confidence*: med.

---

## 4. Agent prompts → `.claude/agents/*.md`

**F4.1 — `blocking-planner.md`: an unmeasured bottleneck regime was allowed to dismiss levers.**
*What*: `op_design.md` asserts "the op is DRAM-bandwidth-bound" as fact and uses it to wave off
three levers — the grid-fill gap ("That is acceptable and deliberate", op_design.md:60-66, quoting
`master.md:45` "No gain once DRAM-bandwidth-bound"), `RECONFIG_MODE=off` ("Phase 1 keeps reconfig
**on** … because the op is DRAM-bound", …:78) and `math_fidelity` (…:130). Verification measured the
opposite: "the op is **SFPU-bound, not DRAM-bound** … This contradicts `op_design.md`'s central
premise and voids its 'that lever will not pay because we are DRAM-bound' dismissals"
(changelog.md:47-52). The dismissed grid-fill lever was then worth **16.1×** (Refinement 2). The
prompt rightly asks for qualitative structural reasoning and "**never a nanosecond estimate**"
(blocking-planner.md:56) — but does not forbid turning that argument into a lever veto.
*Recommendation*: one clause — "You may hypothesize a bound regime, but label it a hypothesis and
name the measurement that falsifies it. **Never** dismiss a parallelism / grid-fill lever on an
unmeasured roofline; leave it as a lamp for the perf queue."
*Confidence*: high.

**F4.2 — `blocking-verifier.md`: the perf queue may attribute a phase without an ablation gate.**
*What*: the verifier ranked the queue from per-phase device zones (verification_report.md:323 "Where
the time goes … device zones") and filed R1 as "worth more than everything else combined" off the
63.9 % figure; R1 then spent a refinement establishing that the number came from a zone wrapping
`cb_wait_front` (and that an early 248 µs reading was transient — breadcrumbs `CORRECTION`).
`blocking-perf-coordinator.md:56-60` already mandates ablation before ranking; the verifier's
perf-queue section (…:303-317) does not.
*Recommendation*: mirror the coordinator's rule into blocking-verifier.md — "a perf refinement's
region must be justified by an **ablation, or by two methods agreeing**, not by a per-phase zone
share (zones include `cb_wait_front`; starvation reads as cost)."
*Confidence*: high.

**F4.3 — `blocking-verifier.md`: a "documented deviation" can ship without a graded cell.**
*What*: the R1b entry authorizes "**Explicit documented deviation** with the numbers above"
(op_requirements.md:248), and the deviation was executed to the letter — docstring, changelog,
guard-set measurement. What no prompt required was a **graded** cell pinning the flipped default or
the escape hatch, so neither configuration of the flag is touched by the blind pass (F1.1). The same
prompt already says "never flip it silently" (op_requirements.md:156); the missing half is "and
never flip it untested".
*Recommendation*: add to the deviation protocol — "a refinement that deviates from a prompt Rule by
changing a default must also land a graded cell (a `LOOSE_CASES` entry or a `test_regression` case)
pinning **both** sides of the changed default; if the facet is not a declared axis, file the axis
request in the report."
*Confidence*: high.

**F4.4 — `blocking-verifier.md`: no sanctioned channel for a golden-spec defect.**
*What*: the run's only `supported_fail` was a *spec* bug — `INPUTS[4]` declared `weight` as
`(2,1,1,V)`, contradicting the contract in the same file, and "the case raised a torch broadcast
error inside the harness *before any device tensor was built*, so no op-side change could ever have
cleared it" (changelog.md:68-73). The prompt says "Do not edit `feature_spec.py` yourself; report
the issue" (blocking-verifier.md:79), yet the only route to a green gate was to edit it — which the
verifier did, and disclosed ("*This is the one edit made outside the op directory*",
verification_report.md:79; commit `45ff193087`). The agent under test therefore modified its own
graded universe, and the blind pass measures the edited spec.
*Recommendation*: (a) harness pre-flight — evaluate every `INPUTS` / `LOOSE_CASES` entry through the
suite's own reference (`pytorch_<op>`) on host at collection time, so spec-side shape/broadcast bugs
surface as their own category rather than as `supported_fail`; (b) blocking-verifier.md: define a
spec-defect protocol (report + a quarantine marker the runner honours) so no agent needs to edit the
graded spec to make progress.
*Confidence*: high.

---

*Advisory only. Every item above is a proposal for a human to ratify; nothing in this run was
changed on account of it.*
