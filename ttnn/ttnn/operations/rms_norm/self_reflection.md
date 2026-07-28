# Self-Reflection: rms_norm (advisory — nothing here is auto-applied)

**Summary.** Final blind pass is clean: **4525 passed / 0 failed / 0 hangs**, split 4407 golden
cartesian + 19 loose + 84 translated + 15 regression, with 1995 xfail and 33918 skipped
(33900 `INVALID` + 18 `INFEASIBLE_L1`). `verifier_report.json` shows `supported_fail=0`,
`xpass_drift=0`, and final `SUPPORTED` == `TARGET` on every axis (3 EXCLUSIONS). So there is no
failure cluster to mine — the run's residual risk is **framework-level, not op-level**: 84 % of the
declared universe never executes, **1980 of those skips are "author-scoped" `INVALID` entries**
(1035 of them would count as `supported_pass` today), and they cover exactly the region where the
run's most dangerous bug lived — a reduce that silently dropped elements, which the eval metric pair
(PCC, rel-RMS) **passed** and only a hand-written all-ones probe caught. Secondary theme: two
helper/reference rules that were wrong-by-default or silent cost the run a 82-cell failure wave, a
device hang, and a PCC-0.0 regression.

---

## 1. Golden coverage → `eval/golden_tests/rms_norm/feature_spec.py`

### 1.1 "Author-scoped" `INVALID` hides 1980 cells, 1035 of which pass today
**What.** Five `INVALID` entries are self-labelled as *not* structural impossibilities and are parked
there to keep them out of the refinement backlog. They uniquely account for **1980 skipped cells**;
re-projecting them through `INPUT_TAGGERS` + final `SUPPORTED`/`EXCLUSIONS` gives **1035 that would
be `supported_pass`** (480 `bf8b ∧ w_non_aligned`, 240 `bf8b ∧ h_non_aligned`, 315
`ROW_MAJOR ∧ HEIGHT_SHARDED ∧ gamma TILE`) and 945 that would be honest xfails. All are in-TARGET.
**Evidence.** `feature_spec.py:93` — `# --- Author-scoped exclusions ("for now", NOT structural
impossibility) ---`; `verification_report.md:73` — "**1 260 cells are skipped that should be
xfailing**"; `:74` — "They uniquely account for **720 skipped cells**" (my re-count reproduces both
numbers exactly). The bf8b buckets are known-good on device:
`tests/.../test_rms_norm_precision_matrix.py:70-73` runs bf8b at `32x48`, `48x64`, `17x50`, `100x47`
and passes; `changelog.md:165` — "No `EXCLUSIONS` entry was needed for bf8 non-aligned … they remain
`feature_spec.INVALID`-skipped".
**Recommendation.** Delete the three cross-tensor `{layout, memory_layout, gamma_layout}` entries and
the two `{bfloat8_b, *_non_aligned}` entries from `INVALID`; if any region must stay refused, home it
in the op's `EXCLUSIONS` (visible xfail, stays a refinement candidate). Net effect: +1035 measured
`supported_pass` cells for zero new device time beyond the cells themselves.
**Confidence.** high.

### 1.2 The eval universe has no absolute (non-correlational) check — the run's worst bug class is invisible to it
**What.** Two of the run's three most severe bugs were *element-count* errors in the reduce: the
padded lane mask decoding to zeros, and the cross-core combine double-counting. Both only rescale
each row, so PCC is blind and rel-RMS shrinks as `dropped/W` → undetectable on wide shapes. Both were
caught by op-private all-ones probes, **not** by the eval suite; `test_regression.py` (the eval-side
numerics file) only varies magnitude/sign at fp32 on three tile-aligned shapes.
**Evidence.** `changelog.md:158` — "A silently-wrong bfloat8_b cell that PCC could not see … Random-data
PCC was **0.9998** and the golden gate (0.99 / 0.10) passed anyway"; `changelog.md:252` — "all-ones
`W=64` produced `mean(x²) = 8.75` instead of `1.0` — and **PCC scored 0.9999**";
`agent_logs/blocking-implementer_breadcrumbs.jsonl` (`device_print_observation`) — "Random-data
PCC=0.9998 HID it".
**Recommendation.** (a) Cheapest: add one all-ones / hand-calculable case at a partial-W boundary to
`eval/golden_tests/rms_norm/test_regression.py` (e.g. `W ∈ {49, 4097}` × `{bf16, bf8b}`) asserting the
*absolute* recovered element count, mirroring `test_partial_w_reduce_counts_every_element`.
(b) General: implement the `extras` key `feature_spec.py:178-179` already promises ("precision
thresholds, **input distribution**, etc.") but `helpers.py:144-150` does not recognise, so any op's
`LOOSE_CASES` can pin a deterministic distribution + exact gate.
**Confidence.** high (mechanism), med (best home for the check).

### 1.3 Axis-blind gap: W-magnitude selects the code path, and W > 8192 runs at exactly one corner
**What.** The op's blocking model branches on per-core `Wt` — `NW>1` chunking, `WT_CHUNK`,
`x_resident` vs streaming, `colpack`, and the `L1_GATHER_BUDGET` halve-and-re-derive loop whose
terminal state is a host `assert blk.fits`. No tagger captures magnitude (`alignment`, `rank` only),
and the spec itself notes `INPUTS` tops out where a row still fits one core: `feature_spec.py:190` —
"W=8192 (256 tiles), which still fits one core's ~1.2 MB L1 resident". Above that, coverage is 3 loose
cases at **n=1 each**, all bf16 / TILE / `fp32_dest_acc_en=True` / INTERLEAVED. The L1 model is
tightest along `dtype × W`: all 18 `INFEASIBLE_L1` skips are `FLOAT32 ∧ W=8192 ∧ HEIGHT_SHARDED`, and
R4 hit "fp32 W=4096 refused by 10KB with 361KB of the bank free"
(`agent_logs/blocking-implementer_breadcrumbs.jsonl`, refinement 4).
**Recommendation.** Add one boundary `LOOSE_CASES` entry at the fp32 corner —
`{"inputs": ((1, 1, 32, 16384),), dtype=float32, fp32_dest_acc_en=True, layout=TILE,
gamma_mode="gamma", gamma_dtype=float32, gamma_layout=TILE, memory_layout=INTERLEAVED}` (1 tile-row
forces the W-split; fp32 maximises bytes/tile). Separately consider promoting a derived
`w_regime` tagger (e.g. `wt_per_core` bucketed at the chunking threshold) — a shape-derived axis costs
**no extra cells** and would let the dashboard/refinements address the chunked-reduce path by name.
**Confidence.** med (risk-based; no failure observed here).

### 1.4 Regression cells bypass the axis tagger and land uncategorised
**What.** `test_regression.py` imports the raw op, so its 15 cells carry no axes and
`verify_supported` files them under `no_axes_found: 15` — they pass but contribute nothing to any
coverage statement.
**Evidence.** `eval/golden_tests/rms_norm/test_regression.py:24` — `from ttnn.operations.rms_norm import
rms_norm`; `golden_blind_final/verifier_report.json` — `"no_axes_found": 15`.
**Recommendation.** One-line change: `from eval.golden_tests.rms_norm.axes import observed as rms_norm`
(what `test_translated.py:15` already does). Likely a template-wide fix for every op's regression file.
**Confidence.** high.

---

## 2. SUPPORTED honesty → `ttnn/ttnn/operations/rms_norm/rms_norm.py`

**The declaration is honest on everything the harness measured.** `supported_fail=0`,
`xpass_drift=0`, `xfail_wrong_mode=0`, `invalid_unexpected=0`; the accounting reconciles exactly
(4425 cartesian-supported − 18 infeasible + 19 loose + 84 translated = 4510 `supported_pass`, + 15
`no_axes_found` = 4525 passed). No fix/demote/promote is warranted. Two caveats worth a human's eye:

**2.1 An `EXCLUSIONS` entry is justified against the test harness, not the device.** The
`{ROW_MAJOR × WIDTH/BLOCK_SHARDED}` refusal is argued from the shard geometry the eval harness
synthesises — `rms_norm.py:126` — "*eval.sharding's RM granule is (1 row, L1_align/elem_bytes
columns)*". A production RM width-sharded tensor whose shard *is* a tile block would be refused
anyway. Recommend re-stating the refusal in device/kernel terms (or narrowing it to the shard shapes
that genuinely can't be tilized in place); note the harness also cannot synthesise the counter-example,
so neither side of the claim is currently testable. Confidence: med.

**2.2 A declared PROPERTY with no eval-side coverage.** `rms_norm.py:156` declares
`math_fidelity: ["LoFi", "HiFi2", "HiFi3", "HiFi4"]`, but golden exercises HiFi4 (default) and HiFi2
(perf loose cases only) — LoFi/HiFi3 rest entirely on the op-private 320-cell precision matrix.
Recommend either pinning one LoFi loose case via `extras["math_fidelity"]` or marking the property
`source: "unit-tested"` so the claim's evidence base is explicit. Confidence: med.

---

## 3. Helper / reference docs

### 3.1 The partial-reduce mask's format binding is nowhere stated — and the generic checklist states the opposite
**What.** On the `AccumulateViaAdd` partial path, `fold_partial_last` unpacks the mask as srcB with
**no** reconfig, so the mask CB's format must equal the *reduce input CB's* format. `prepare_reduce_mask`'s
docstring documents layout and the two legal formats but never that binding; the design checklist
mandates the opposite unconditionally, and the numerics reference tells the reader to trust the helper's
reconfig. Cost: 82 golden failures (all `float32 ∧ w_non_aligned`), then a silently-wrong bf8b class.
**Evidence.** `.claude/references/op-design-template.md:147` — "- [ ] Reduce scaler CB is bfloat16";
`.claude/references/numerical_stability_analysis_reference.md:410` — "handles … **data-format reconfig
automatically** — trust its defaults"; `ttnn/cpp/ttnn/kernel_lib/reduce_helpers_dataflow.hpp:62-74`
(no format-binding rule; `.inl:255` only asserts `Float16_b || Float32`);
`op_design.md:439` R4 — "`cb_scaler` format is `Float16_b`"; fix commit `62d656826e` "fix fp32 partial-W
mask format; golden 703/0"; breadcrumb — "op_design.md R4 mandates Float16_b; that is correct for the
ReduceTile datapath, **not** for the AccumulateViaAdd mask".
**Recommendation.** Add one line to `prepare_reduce_mask`'s docstring and to the PARTIAL paragraph at
`reduce_helpers_compute.hpp:141-143`: *"the mask DFB's data_format MUST equal the reduce **input** DFB's
format (fold_partial_last unpacks it as srcB with no reconfig); a block-float input must therefore
reduce out of a bf16 intermediate"*; make `op-design-template.md:147` conditional on the datapath.
**Confidence.** high.

### 3.2 `OutputLifecycle::Streaming` + `block_size > 1` is silently wrong; only the input side is clamped
**What.** The chain documents that streaming **inputs** clamp `block_size` to 1, but `PackTile` is not a
CB-reader element, so a blocked chain paired with `OutputLifecycle::Streaming` reserves/pushes one page
per iteration while the pack loop writes `block_size` — PCC 0.0.
**Evidence.** `eltwise_chain.hpp:462-465` — "The chain clamps `block_size` … Streaming CB-reader chains
clamp block_size to 1 for them"; `:289` — "Default: reserve and push 1 output tile each step";
`changelog.md:682` — "**`block_size > 1` with `OutputLifecycle::Streaming` gives PCC 0.0** … `Chunked`
is the matching policy" — and "invisible had the block happened to be 1 on the shapes under test".
**Recommendation.** State the pairing rule at `OutputLifecycle::Streaming` (use `Chunked` when
`block_size > 1`), and ideally add the missing static_assert so the mismatch cannot compile.
**Confidence.** high.

### 3.3 A multicast rectangle is a *virtual* rectangle — the reference conflates logical with virtual
**What.** The cross-core reference discusses non-rectangular groups only in *logical* space and elsewhere
writes "logical (virtual)" as if they were one space. On this part they are not: harvesting leaves gaps in
virtual-x, so a logically dense group can multicast into non-worker endpoints — a hard device hang on the
WIDTH_SHARDED cells.
**Evidence.** `.claude/references/cross_core_reduction_design.md:467-469` — "Feed the pipe / `get_noc_addr`
**logical (virtual)** coords"; `:269-270` — "always feed the pipe **virtual/logical** coords"; §8
`:420-461` (logical-only rectangularity); `changelog.md:259-261` — "logical x 0..6 → virtual 1..7 but
logical x 7..10 → virtual 10..13 … a hard device hang".
**Recommendation.** Add a §9 bullet: *"A multicast rectangle is a VIRTUAL rectangle. Logical
contiguity ≠ virtual contiguity (harvesting) — split each group's broadcast into one family per
virtually-contiguous run, and never write 'logical (virtual)'."*
**Confidence.** high.

### 3.4 Two measured helper facts that no header states
**What.** (a) `ckl::UnaryBcast::exec` hardcodes `in_tile_index=0`, so a multi-tile broadcast operand is
silently wrong — the invariant is a comment in the `.inl` body, not on the public declaration, and
"an all-ones gamma scores perfectly on this bug". (b) `DestReuseBinary` at `block_size ==
DEST_AUTO_LIMIT` corrupted one face of the highest DEST lane (PCC 0.988), which the clamp doc's
reassurance ("an oversized value can't overflow DEST") does not lead you to expect.
**Evidence.** `agent_logs/blocking-perf-coordinator_breadcrumbs.jsonl` (`helper_bugs_found`) — both
quotes verbatim; `eltwise_chain.inl:1239-1247` (invariant in the body); `eltwise_chain.hpp:662`
(declaration, no note); `:462-464` (clamp reassurance).
**Recommendation.** Hoist the "always reads tile 0" line onto the `UnaryBcast` declaration; add a caveat
to the `block_size` clamp paragraph naming `DestReuseBinary` at the limit (or fix/assert it).
**Confidence.** med for (a), low for (b) — one measurement inside a perf experiment.

---

## 4. Agent prompts → `.claude/agents/*.md`, `eval/prompts/*`

### 4.1 The refinement prompt lets ablation alone close a "*-bound*" verdict; zones then contradicted it for five rounds
**What.** The perf paragraph of the refinement prompt prescribes ablation as *the* bottleneck classifier
into three coarse buckets. R5 closed on "MATH-bound" and that verdict was written into the op as a
standing rationale. Perf 1 — whose prompt *mandates* per-stage zones — immediately found 38 % of the
kernel in an unrelated stage (SFPU computing 1024 lanes where 32 are consumed) and won 3.53× on it.
The same round shows the mirror error: the coordinator's first zone read said 74 % combine, ablation
corrected it to 26 % — a zone can be WAIT, not work.
**Evidence.** `eval/prompts/blocking_refinement_prompt.txt:44` — "ablation profiling to classify the
bottleneck as compute / dataflow / memory bound"; `eval/prompts/perf_refinement_prompt.txt:11` (same);
vs `eval/prompts/blocking_perf_tournament_prompt.txt:9` — "instrument every stage boundary … **Never
guess where the time goes**"; `rms_norm_program_descriptor.py:193` — "it is MATH-bound, so removing FPU
ops is the only lever that reaches it"; `agent_logs/blocking-perf-coordinator_breadcrumbs.jsonl`
(`breakdown`) — "Prior refinement-5 ablation … called the case MATH-bound; the per-stage zones
contradict that", and (`breakdown_corrected`) — "Cumulative ablation … corrected my first zone read".
**Recommendation.** Backport the tournament rule into the refinement prompts: a "*X-bound*" claim needs
per-stage zone attribution that sums to the measured kernel time **and** an ablation cross-check, with
an explicit warning that a stage zone may be a wait; neither instrument alone may close a perf phase.
**Confidence.** high.

### 4.2 The verifier's `INVALID` audit was correct at phase 0 and had nowhere to go
**What.** The verifier diagnosed both mis-homed `INVALID` clusters of §1.1 in the phase-0 report,
correctly declined to edit the golden author's file, and nothing consumed the finding for the rest of
the run — this reflection's top item is a re-derivation of it 14 hours later.
**Evidence.** `verification_report.md:69` — "Two clusters to raise with the golden-test author (**I did
not edit `feature_spec.py`**)"; `changelog.md:88` — "*Reported, not edited* … 1 260 cells skipped that
should be xfailing"; the finding appears in no `verifier_report.json` category (`INVALID`-skipped cells
are simply absent from the counts).
**Recommendation.** Have `.claude/agents/blocking-verifier.md` emit `feature_spec` findings as a
machine-readable block in `verifier_report.json` (e.g. `feature_spec_findings: [{entry, cells,
proposed_home}]`) that the pipeline surfaces in the Docs tab immediately, instead of only in prose.
**Confidence.** high.

### 4.3 The design step copies datapath-independent CB rules from the template checklist
**What.** `op_design.md` R4 restated the generic checklist's "scaler CB is bfloat16" as a mandate for a
design that had already chosen the non-default `AccumulateViaAdd` datapath, where the rule is wrong
(§3.1). The verifier saw the resulting drift but scored it non-blocking.
**Evidence.** `op_design.md:439` R4; `changelog.md:91` — "*Design-doc drift noted, not blocking*:
`op_design.md` R4 (`cb_scaler` format) … no longer match the shipped, helper-mandated behaviour".
**Recommendation.** In `.claude/agents/blocking-planner.md`: when the design selects a non-default helper
datapath, every CB format/lifecycle rule it states must be re-derived from *that* datapath's helper
contract (cite the helper line), never copied from `op-design-template.md`'s generic checklist; and the
verifier should treat a design rule contradicted by shipped code as a doc fix, not a note.
**Confidence.** med.
