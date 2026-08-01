# Self-Reflection: moe_fused_swiglu

_Advisory only. Nothing below is applied; every item is a proposal for a human to ratify._

## Summary

The blind pass is fully green: **45/45 golden cells passed** (36 `test_op` + 9 `test_op_loose`), 0 translated
and 0 regression cells — the universe is golden-only. `verifier_report.json` is clean on every category
(`supported_pass 45`, `supported_fail 0`, `xpass_drift 0`), and `SUPPORTED` equals `TARGET` on all five axes
with empty `EXCLUSIONS`/`INVALID`. There are therefore **no blind failures to cluster**, and no SUPPORTED edit
is warranted.

The interesting findings are framework-level, not op-level. The single most important one: golden's only
correctness gate is a constant PCC against an *unquantized* fp32 oracle, pinned just under a bfp4 format floor
— it failed 44/45 cells at Phase 0 for a reason no correct kernel could avoid, and after being relaxed to
0.975 it grades ~2e-2 of unavoidable quantization against a kernel residual of ~6e-4. Two real defects the run
did hit (a silent multicast race, an unimplemented design section) were found by *perf* reading and by luck,
not by the suite. Separately, `golden_blind_final` was recorded **before** the two trailing perf rounds and
understates the shipped op by 33–42%.

---

## 1. Golden coverage → `eval/golden_tests/moe_fused_swiglu/feature_spec.py`

### G1 · The graded PCC gate is pinned under a format floor, so it cannot detect kernel error
**What.** The suite's only pass/fail is one constant PCC vs an fp32 oracle that carries none of the bfp4/bfp8
quantization the op's own signature mandates. The measured format ceiling on this fixture is 0.97967–0.98019;
the gate started at 0.98 (unreachable on 11 of 12 combinations) and now sits at 0.975. The op's
kernel-attributable residual is 5.2e-4–6.8e-4, i.e. ~40× smaller than the slack the gate leaves.

**Evidence.** `golden_phase0/test_results.json`: 44 cells `failure_category=numerical-precision`, e.g.
`"pcc=0.979057 < target 0.98"`. Commit `2c4b563cb0`: "the golden PCC gate (0.98) sits ABOVE the unbeatable
bfloat4_b format floor". `helpers.py:167-170`: "This gate is therefore pinned just under a FORMAT FLOOR, which
makes it a weak detector of kernel error". The durable fix already exists but lives **outside** the graded
suite, op-locally: `tests/ttnn/unit_tests/operations/moe_fused_swiglu/test_moe_fused_swiglu_precision_baseline.py:196`
`assert_with_pcc(expected, actual, floor_pcc - FLOOR_SLACK)` — ~13× tighter, per-shape.

**Recommendation.** Propose folding the quantization-aware oracle into `helpers.py`: compute `floor_pcc` per
cell (weights → bfp4, `h`/out → bfp8 via `from_torch`/`to_torch` round-trips, chain in torch), gate at
`floor_pcc - 0.0015`, and record kernel-vs-fp32 PCC ungated; retire the constant `_PCC_GATE`.
*(Process note worth preserving: the verifier correctly refused to edit the suite itself and asked the harness
owner — `blocking-verifier.md:79` — which is the behaviour we want.)*
**Confidence:** high.

### G2 · The gate constant is defined in four places, kept in sync by comment
**What.** `_PCC_GATE` is independently declared in the suite twice and copied into the op prompt, plus a
fallback literal in the op-local baseline test. When the operator moved it 0.98 → 0.975, the copies did not
follow and an entire refinement iteration was spent proving no kernel had changed.

**Evidence.** `feature_spec.py:282` and `helpers.py:175` both `_PCC_GATE = 0.975` ("Kept in lockstep with
helpers.py's gate" — a comment, not a mechanism); `eval/prompts/moe_fused_swiglu.txt:188` "**PCC >= 0.975**";
`test_moe_fused_swiglu_precision_baseline.py:51` `GOLDEN_PCC_GATE = 0.975` fallback. Commit `24a4c5f03b`: "the
gate violation was a stale PCC literal, not a kernel defect … ZERO kernel, host-descriptor or op-file lines
changed".

**Recommendation.** Propose `feature_spec.py` import `_PCC_GATE` from `helpers.py` (one definition), and the
prompt cite the file instead of the number.
**Confidence:** high.

### G3 · Axis-blind gap: `count % 32` only ever takes the values {0, 31}
**What.** The tile-padding seam is the spec's named central hazard, but every count in the 45-cell universe is
either tile-aligned or exactly one row short of a tile. `balanced = capacity//32` → 32/64/160 (≡0),
`full = capacity` → ≡0, `partial = capacity//4 - 1` → 255/511/1279 (all ≡31), loose cases 0/128/256/512/5120
(all ≡0). A count with *many* phantom rows in the last tile (residues 1..30) or a sub-tile count (`count < 32`)
is never exercised — and `tag_fill` labels them "partial"/"balanced", so the axes-tuple looks covered.

**Evidence.** Derived from the 45 nodeids in `golden_blind_final/test_results.json`: counts
`{0,32,64,128,160,255,256,511,512,1024,1279,2048,5120}` → `count % 32 ∈ {0, 31}`.
`feature_spec.py:24-26` "rows `[count, ceil_tile(count))` are UNDEFINED"; `feature_spec.py:118,123` define the
two generators that produce this collapse.

**Recommendation.** Propose two `LOOSE_CASES`: `_case_inputs(7168, 1024, 33)` (31 phantom rows, `m_tiles=2`)
and `_case_inputs(7168, 1024, 1)` (sub-tile, `m_tiles=1`), both `bf16_rm`, no extras. Optionally promote the
facet to an axis: `tag_tile_align -> "aligned" if count % 32 == 0 else "ragged"`, so the region is visible in
the registry rather than hidden inside `fill`.
**Confidence:** high on the gap; the recommendation is coverage, not a bug claim.

### G4 · The suite is structurally blind to "op ignores the runtime count" — and that blindness hid a race
**What.** `helpers.py` says so in its own header. The op shipped Phase 0 doing `M_BLOCK=8` tile-rows regardless
of count (an unimplemented design section) with all cells green; that over-computation was simultaneously
*masking* a non-deterministic corruption of the last token tile-row, which only surfaced when refinement 1
shrank `m_eff`.

**Evidence.** `helpers.py:13-15`: "an op that ignores `counts` and grinds through all `capacity` rows still
passes … Only perf distinguishes that". Commit `2c4b563cb0`: "the runtime m_tiles shrink op_design.md §3
specifies was never implemented". Breadcrumb `blocking-implementer_breadcrumbs.jsonl` #2: "the h-send loopback
was a PRE-EXISTING latent bug (present at Phase 0); the m_eff shrink removed the (m_eff-1)*KR_PAD tile-matmuls
of cover that hid it".

**Recommendation.** Propose making count-proportionality *graded* using cells already present: assert a
machine-independent ratio between the two same-allocation loose cases, e.g.
`device_kernel_ns(cap5120,n=5120) / device_kernel_ns(cap5120,n=128) >= 4` (blind data: 3,515,609 / 140,833 =
25×; an op ignoring the count would score ~1). Requires `helpers.py` to act on one perf extra — today
`helpers.py:236` states "Correctness is this runner's only pass/fail".
**Confidence:** med (the constant needs a human's calibration; the discriminative power is not in doubt).

### G5 · No repeat-invariance check; the run's only real race was caught by a hand-built 4-rep suite
**What.** Each cell runs once with a fixed seed, so a run-to-run race is a coin flip against the gate.

**Evidence.** Breadcrumb #1: "nondeterministic low values in the LAST token tile-row, bf16_rm only, m_eff in
{2,4} but not {1,8}"; #2: "all 16 (format x m_eff-regime) cells bit-stable across 4 reps (spread exactly 0.0)".
That repeat suite lives op-locally (`tests/.../test_moe_fused_swiglu_m_tiles.py`), not in golden.
`mcast_pipe.hpp:184`: "Silent: correct output most runs, corrupt some runs".

**Recommendation.** Propose one `LOOSE_CASES` entry with `extras={"repeat": 2}` on a multi-round-collective
shape (7168 / 5120 / 128) and have `helpers.py` assert the two outputs are bit-identical.
**Confidence:** med.

### G6 · `input_m_tiles` — a prompt-mandated public parameter — is never exercised, and the two axis models disagree on it
**What.** The op's signature is fixed by the prompt and includes `input_m_tiles`; `axes.py` gives it explicit
axis semantics; no golden cell passes it. Meanwhile `validate()` derives the `capacity` axis from the tensor
shape, contradicting `axes.py` — a latent inconsistency that is currently unobservable.

**Evidence.** `eval/prompts/moe_fused_swiglu.txt:39,45` "`input_m_tiles` defaults to `capacity / 32`; it lets a
caller whose `x` is wider than one expert's region size the op to that region". `axes.py:75-77`
"`input_m_tiles`, when supplied, is what the op sizes its work to, so it — not x's allocated row count — is the
capacity `fill` is relative to. The golden suite always passes None." Op file
`moe_fused_swiglu.py:186` `"capacity": int(input_tensor.shape[-2])`.

**Recommendation.** Propose one `LOOSE_CASES` entry `extras={"input_m_tiles": 32}` on
`_case_inputs(7168, 5120, 256)` (with a one-line `helpers.py` passthrough) — it exercises the knob and forces
the two axis models to agree.
**Confidence:** med.

### G7 · `fill="empty"` is claimed on 12 combinations and tested on 1
**What.** `SUPPORTED["fill"]` includes `"empty"` across the whole cross, but the only zero-count cell is
(emb 7168, capacity 1024, bf16_rm). The zero-count dispatch is precisely the path that can hang.

**Evidence.** `feature_spec.py` LOOSE_CASES: a single `_case_inputs(7168, 1024, 0)` entry;
`golden_blind_final/test_results.json` shows one `n0` nodeid, `device_kernel_ns = 6036`.

**Recommendation.** Propose one more empty case at the other activation format and the largest capacity —
`_case_inputs(7168, 5120, 0)`, `bfp8_tile`. Cost is ~6 µs.
**Confidence:** high (cheap, low severity).

---

## 2. SUPPORTED honesty → the op file's `SUPPORTED` / `EXCLUSIONS`

**The mechanical check is clean and I recommend no `SUPPORTED`/`EXCLUSIONS` edit.**
`golden_blind_final/verifier_report.json`: `total 45, supported_pass 45, supported_fail 0, xpass_drift 0,
xfail_wrong_mode 0, invalid_unexpected 0, no_axes_found 0`. `registry_snapshot.json` shows `exclusions: []`,
`invalid: []`, and `SUPPORTED == TARGET` on all five axes — there is no over-claim to demote and no under-claim
to promote. The same holds at every refinement phase from `golden_refinement_1` onward (Phase 0 was 1/45, all
44 failures being the unreachable gate of §G1, not a support claim).

One honest caveat, not a dishonesty signal: two declared regions rest on very thin evidence — `fill="empty"`
is claimed on 12 (emb × capacity × input_format) combinations and tested on 1 (§G7), and every
`input_m_tiles ≠ None` call is claimed-by-signature and tested zero times (§G6). The proposal is to widen the
*evidence* via `LOOSE_CASES`, not to narrow the claim.

Also worth recording as correct-and-honest: `fill` is declared in `SUPPORTED` but is unenforceable in
`validate()` because it derives from a device-resident value; the op documents exactly that at
`moe_fused_swiglu.py:178-181` ("observed-but-uncheckable") rather than faking a host-side check. No change
proposed.

---

## 3. Helper / reference docs → helper docstrings + `.claude/references/`

### H1 · `mcast_pipe`'s loopback default is the hazardous mode for the common consumer; the doc was silent
**What.** `SenderPipe::send()` selects INCLUDE-source (loopback) multicast automatically from
`src_l1 != dst_l1` — which is the natural call shape when staging and landing CBs differ. Under
`ROTATING_SENDER` that races the flag reset, which sits behind `async_writes_flushed()` (SENT, not LANDED),
producing silent run-to-run corruption of the last round. The pre-existing `ROTATING_SENDER` doc described the
reset protocol but never mentioned the interaction — an absence, not a wrong statement.

**Evidence.** `ttnn/cpp/ttnn/kernel_lib/mcast_pipe.inl:84` `const bool loopback = in_rect_ && src_l1 != dst_l1;`
(and `:88` selects the fan-out from it). `mcast_pipe.hpp:169-173` (added 2026-07-05) documents the rotating
reset with no loopback caveat. Breadcrumb #1: "rotating-sender Flag reset (set(INVALID) behind a writes_flushed
fence = SENT not LANDED) races the sender's own MCAST_INCL_SRC loopback VALID write". Commit `38b2826d02`:
"Exposed and fixed a PRE-EXISTING race". The run added the warning at `mcast_pipe.hpp:174-193`.

**Recommendation.** The doc is now right but the *default* still is not: propose `SenderPipe` either perform
the local self-copy internally when `ROTATING_SENDER && src != dst`, or `static_assert`/`ASSERT` that
combination out — so the hazard is unreachable rather than commented.
**Confidence:** high.

### H2 · `DataReadySignal::Counter` was offered as a free perf choice while being unusable on device
**What.** The enum doc presented Flag vs Counter as a tuning decision. Counter hung twice for two independent
reasons; the op never uses it, so the entire investigation was tax.

**Evidence.** `mcast_pipe.hpp:69` (pre-run) "Counter: a monotone, reset-free counter. Pick this ONLY for tight
multi-phase streaming"; `mcast_pipe.hpp:72` (added by this run) "!! KNOWN BROKEN … the Counter path HANGS
whichever fan-out it is given"; `mcast_pipe.inl:153-158` "a guaranteed hang for every Counter user whose src cb
!= dst cb". No `DataReadySignal`/`Counter` reference exists anywhere in
`ttnn/ttnn/operations/moe_fused_swiglu/kernels/`.

**Recommendation.** Propose marking helper enum arms that have never run on device as such in the docstring
(or gating them behind an explicit experimental macro), so an implementer does not discover it with a hang.
**Confidence:** high.

### H3 · The device-profiler include rule for compute TUs is documented nowhere
**What.** `DeviceZoneScopedN` in a compute translation unit compiles to nothing without an explicit profiler
include — no error, just an empty CSV — and a compute kernel does not get it transitively the way a dataflow
kernel does via `dataflow_api.h`.

**Evidence.** Breadcrumb `blocking-perf-coordinator_breadcrumbs.jsonl` #0: "root-caused Refinement 2 note (c)
empty-CSV: the compute TU used DeviceZoneScopedN without including the profiler header (a compute kernel does
not get it via dataflow_api.h)". `grep -rn DeviceZoneScopedN .claude/references/` → 0 hits.

**Recommendation.** Propose one line in the perf/measurement reference: "compute kernels must `#include` the
kernel profiler header explicitly — `DeviceZoneScopedN` silently compiles out otherwise and you get an empty
CSV, not a build error."
**Confidence:** high.

### H4 · `noc_async_read_barrier` is documented accurately but its prefetch consequence is not, and the TRID form is 500 lines away
**Evidence.** `.claude/references/data_transfer_analysis_reference.md:151` "`noc_async_read_barrier()` — waits
for all pending reads to complete"; the selective form appears only at `:657`
(`noc_async_read_barrier_with_trid(trid)`). Breadcrumb `blocking-implementer` #14: "noc_async_read_barrier
drains ALL reads, so the per-round W_down prefetch paid full DRAM latency on the spot - this is why WD_AHEAD
measured neutral at Phase 0."

**Recommendation.** Propose appending to `:151`: "— this defeats any prefetch in flight; to wait for one
transfer use `noc_async_read_barrier_with_trid()`."
**Confidence:** med.

---

## 4. Agent prompts / process → `.claude/agents/*.md`

### P1 · The blind pass ran *before* the two trailing perf rounds, so `golden_blind_final` is not the final state
**What.** `golden_blind_final` was recorded immediately after Refinement 3 and before Perf 1 and Perf 2. Its
device timings match `golden_refinement_3` to within noise and understate the shipped op by 33–42%. Anything
downstream that reads it as "final perf" is reading a 4.5-hour-old build.

**Evidence.** `test_results.json` mtimes: `golden_blind_final` 2026-07-31 21:14, `golden_refinement_3` 21:10,
`golden_perf_1` 23:11, `golden_perf_2` 2026-08-01 01:41. Graded focus cell
`cap5120-n256-emb7168` `device_kernel_ns`: r3 223,018 → **blind 223,079** → perf_1 203,706 → perf_2 149,369
(DRAM-read util 0.249 → 0.372); `cap5120-n5120`: blind 3,515,609 → perf_2 2,037,161. Commit `92703b6a6a`
("every cell 20-37% faster") post-dates the blind pass.

**Recommendation.** Orchestrator-level, not a prompt edit: propose re-running the blind pass after the last
graded phase, or renaming the artifact to the phase it actually measures.
**Confidence:** high.

### P2 · No prompt scopes edits to the shared kernel helper library
**What.** The verifier changed the *semantics* of `mcast_pipe.inl` for an arm this op never executes, gated by
this op's suite alone. The change looks correct, but its blast radius is every `mcast_pipe` consumer and
nothing in the run could have detected a regression in another op.

**Evidence.** Commit `2c4b563cb0` stat: `ttnn/cpp/ttnn/kernel_lib/mcast_pipe.inl | 10 +-`,
`mcast_pipe.hpp | 11 +`; the changed arm is `DataReadySignal::Counter`, absent from
`ttnn/ttnn/operations/moe_fused_swiglu/kernels/`. `.claude/agents/blocking-verifier.md:79` explicitly protects
the suite ("Do not edit `feature_spec.py` yourself; report the issue and ask the user to update") but nothing
protects `kernel_lib/`, while `:100` actively pushes ops *into* it ("**Replace it.**").

**Recommendation.** Propose one line in `blocking-verifier.md` (and mirrored in `blocking-implementer.md`):
"Edits under `ttnn/cpp/ttnn/kernel_lib/` are cross-op — change only the arm your op executes and can test;
anything else, document it in the header and file it." (The doc-only `.hpp` additions this run made are
exactly the wanted behaviour.)
**Confidence:** med-high.

### P3 · The gate-drift class: op-local tests copying golden tolerances
**What.** The acceptance test and the precision baseline each copied `PCC_GATE = 0.98` from the suite; when the
suite moved to 0.975, a full refinement cycle was consumed proving the kernel was innocent.

**Evidence.** Commit `24a4c5f03b`: "the acceptance test copied `PCC_GATE = 0.98` while its own docstring names
eval/golden_tests/…/feature_spec.py as the source of that number … The copy never followed"; breadcrumbs #5-#8.
Fixed by importing (`test_moe_fused_swiglu_precision_baseline.py:49`), but the standalone fallback literal at
`:51` keeps the class alive.

**Recommendation.** Propose one line in `blocking-implementer.md`'s test-authoring section: "never copy a
tolerance from the golden suite — import it; a standalone fallback must `pytest.skip`, not restate a number."
**Confidence:** high.

### P4 · Design conformance classified a work-proportionality deviation as perf-only; it was masking a race
**What.** The verifier correctly found that the design's runtime `m_tiles` shrink was never implemented, but
filed it as a perf refinement and declared the queue all-perf. The same deviation was hiding a silent
correctness bug that only appeared once the deviation was closed.

**Evidence.** `.claude/agents/blocking-verifier.md:87` frames work distribution as "a
**performance-conformance** check". Commit `2c4b563cb0`: "Found, filed not fixed (needs a device-ns
measurement -> Refinement 1) … Queue is all-perf". Breadcrumb #2 shows the deviation was covering a
pre-existing race that then required its own debug cycle.

**Recommendation.** Propose extending the Design Conformance bullet: "a deviation that makes the op do *more*
work than the design specifies is also a correctness risk — surplus work masks races and over-runs; require it
closed (or a repeat-invariance test added) before declaring the queue all-perf."
**Confidence:** med.
