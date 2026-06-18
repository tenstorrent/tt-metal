---
name: tune-helper
description: Design a kernel-side helper when you ALREADY know the block to wrap and have a vague idea of the helper in mind, and you want to resolve the remaining style choices EMPIRICALLY rather than by argument. API-first variant of design-helper - intent carries just two things (the low-level primitives to find + a vague sketch of the block to wrap), block discovery is skipped, and pattern evaluation is replaced by an on-device "style bake-off" that runs matched raw-primitive kernel pairs across a parameter matrix to pick winners by data (coverage -> perf -> L1), promoting coverage/perf tradeoffs to internal dual-path dispatch on a recognizable predicate. Kernel annotation is a FULL census of every primitive call site, fanned out to parallel per-group subagents, producing a clean/refactor/defer migration-blocker audit BEFORE any op is touched. INTERACTIVE - stops at a checkpoint after every step. Two tracks - DATAFLOW (coverage = correct across the topology matrix; perf = NoC ns; no-hang) and COMPUTE (coverage = passes the dtype/config matrix; perf = kernel ns; precision = PCC). Use when you've already decided WHAT to wrap and roughly HOW, and the open question is WHICH style variant is actually better on the hardware. Args = two things - the low-level primitives to find, and a vague one-line sketch of the block to wrap (plus optionally a few known instances of the block, used only as a census recall-check).
---

# Tune a Kernel Helper by Empirical Style Bake-off

Use this skill when you already know **which call-site block you want to wrap** and you already have **a rough API in mind**, and the thing you're actually unsure about is **which style variant of the implementation is better on the hardware** — flush vs barrier, counter-sem vs flag-sem, linked vs unlinked, bf16-fast vs fp32-accurate, etc.

It is the **API-first, empirically-decided sibling of `design-helper`**. Where `design-helper` *discovers* the block and *argues* the style choices on paper, `tune-helper` *takes the block and API as input* and *measures* the style choices on device.

## Relationship to design-helper and build-helper

```
design-helper   block EMERGES (Step C) · style picked by QUALITATIVE argument · paper-only · exits at proposal
tune-helper     block is GIVEN (Step 0) · style picked by ON-DEVICE BAKE-OFF   · device-aware · exits at proposal
build-helper    takes either proposal through test plan -> validation -> migration -> report
```

`tune-helper` exits at the same place `design-helper` does — a signed-off `proposed_helpers.md`, which **is Gate 1 of `build-helper`**. The difference is that this proposal's style choices arrive **already backed by data**, and the throwaway bake-off kernels carry over to `build-helper` as a ready-made raw baseline. (This skill knows only that handoff seam — it does not reference `build-helper`'s internal phases.)

## The two core principles (inherited from design-helper)

1. **The unit of abstraction is the recurring CODE BLOCK at call sites, not the individual primitives.** Outcome = 1–2 fat helpers, not N thin wrappers.
2. **A style choice is decided by evidence, not eloquence.** Every style-driven difference between observed patterns gets resolved by running both options on the hardware and reading the numbers — not by scoring bullet points.

## When to use

- You can already name the block ("the mcast + handshake sequence in the matmul in0 reader") and sketch the API (`mcast_block_with_handshake<Mode>(...)`).
- The genuine open question is **which implementation style wins**, and you suspect the honest answer is "it depends on the topology / dtype / config" — i.e. you might need **two internal paths**, not one.
- You want the perf/correctness frontier mapped by measurement, then dispatched automatically so callers in the cheap region don't pay the safe region's tax.

## When NOT to use

- You don't yet know the block or the API → use `design-helper` (it discovers both).
- There is exactly one obvious implementation and no style fork → just write the helper, or run `design-helper` for the audit trail.
- You're past design and want full validation + migration + report → that's `build-helper`.

---

## The pipeline

```
Step 0 — Intent (primitives + vague block sketch)  the ONLY two inputs: primitives to find + a vague block sketch
Step A — Primitive contracts               built from the prompt's low-level API list; REUSABLE across runs
Step B — Hazards / invariants catalog      unchanged from design-helper
[Step C — Block discovery]                 SKIPPED — the block is given in Step 0
Step D — Kernel annotations (FULL census)  annotate every primitive call site; parallel subagent per group; migration audit
Step ★ — API feasibility checkpoint        draft the concrete API; can it be made over the annotated reality?
Step E — Style bake-off (EXPANDED)         on-device A/B of style variants; coverage -> perf -> L1; dual-path frontier
Step F — Helper proposal                   opinionated; only use-case knobs + empirically-justified dual-paths
```

Each step has a written artifact. Don't skip artifacts.

## Interactive checkpoints — read this first

**This skill is pedagogical, not autonomous** (same contract as design-helper). After every step — including Step 0 and the new Step ★ — STOP and run an explicit checkpoint. The pattern:

```
1. Hand the user the artifact path(s) just produced.
2. Summarize in ~5–10 lines what's in the artifact and the key concepts.
3. Call out the 2–3 load-bearing ideas the user must understand to move on.
4. End with: "Does this land? Anything to clarify before I move to Step <next>?"
5. WAIT for the user's response. Only proceed on an explicit OK.
```

The bake-off (Step E) has an extra obligation: **it runs code on the device.** Treat its checkpoint as a Gate-2-equivalent — present the bake-off *plan* (matched kernels + matrix + metrics) and get an OK **before** running anything on device, then present the *results* and get an OK on the chosen winners before Step F.

---

## Step 0 — Intent (primitives + vague block sketch)

`intent.md` carries **exactly two required things** — anything more is premature here:

1. **The primitives to find** — the list of low-level APIs the helper is built out of. This is the prompt's authoritative input and it is *all* Step A needs. (It also makes Step A reusable: a later run with the same list reuses the contracts.)
2. **A vague sketch of the block to wrap** — plain language, *not* a firm signature. "The mcast-data → sem-handshake → flush sequence in producer readers." Just enough to name what the helper is *about* and roughly where the block starts and ends. The actual API shape is **not** decided here — it firms up at Step ★ (feasibility) and Step F (proposal).

Everything else in the template is light bookkeeping, not a design commitment.

```markdown
# Intent: <helper name>

## Primitives to find (drives Step A — the authoritative input)
- noc_async_write_multicast, noc_async_writes_flushed, noc_async_write_barrier,
  noc_semaphore_set_multicast, noc_semaphore_wait, ...

## Vague sketch of the block to wrap (NOT a signature)
- One or two sentences: what sequence, roughly where it starts / ends.
- e.g., "wrap mcast-data + sem-handshake + flush so a reader broadcasts a block
  to a receiver rectangle and re-syncs before the next iteration."
- The rough idea of the helper, in words — no `template<...> void foo(...)` yet.

## Bookkeeping (not design decisions)
- Track: dataflow (data-landed + no-hang)  OR  compute (PCC).
- Known instances (optional): call sites you already know contain the block.
  <path> — role: <reader/writer/compute>
  Used ONLY as a **census recall-check** — the Step D census must find at least these,
  or its discovery is broken. NOT a migration order or an annotation boundary (Step D
  annotates the full census regardless; migration sequencing is build-helper's job).
```

### Checkpoint — after Step 0
Quote the **primitive list** and the **vague block sketch** back. Confirm the primitive list is complete (it gates Step A) and that the block sketch names the right sequence. Do **not** push for an API signature here — it's deliberately vague until the annotations are in. Wait for explicit OK.

---

## Step A — Primitive contracts (prompt-driven, reusable)

Same artifact and rigor as design-helper's Step A (`primitive_contracts.md`): one precise contract card per primitive, built from `grep` + `Read`, pinning each primitive's wait condition to an exact lifecycle stage (enqueued → sent → arrived → ACKed for dataflow; init/reconfig state for compute).

**Two differences from design-helper:**

1. **The primitive list comes from the prompt** (`## Low-level APIs to contract` in `intent.md`), not from inference. If a hazard in Step B reveals a missing primitive, add it and re-checkpoint — but the prompt is the starting authority.
2. **The artifact is reusable across runs.** Prior-work detection: if a `primitive_contracts.md` already exists covering the **same low-level API set**, reuse it and skip Step A. Regenerate only when the prompt's API list changes. Record the API list it was built from at the top of the file (`PRIMITIVES: a, b, c`) so a later run can detect a match or a drift.

### Checkpoint — after Step A
One line per primitive: "what it really waits for." Highlight non-obvious arg semantics and hardware ordering assumptions (VC, in-order arrival, degenerate inputs). Note whether contracts were freshly built or reused. Wait for OK.

---

## Step B — Hazards / invariants catalog

**Unchanged from design-helper.** Produce `hazards_catalog.md` (dataflow) or `invariants_catalog.md` (compute). Each hazard is a race with one or more mitigations; each invariant is required state with one establisher. Same table format, same architecture-conditional sub-catalog for compute (WH vs BH).

**Step B reads NO kernels.** Hazards are derived entirely from the Step A **primitive contracts** — they are a property of the primitives, not of any call site. "flush vs barrier" is a fork because the *primitives* permit two mitigations of the same race, whether or not any current kernel uses both. This is why the bake-off can later test a mitigation **no existing kernel uses**: a call site using these same primitives can never present a style fork the primitive-derived catalog doesn't already contain — at most a use-case knob. (Step D still annotates *every* call site — not to complete the fork set, which is already complete, but to surface migration blockers before you touch the ops.)

This is still **the most important checkpoint** — and it now does double duty: the hazards here are exactly what the Step E bake-off will probe. A style choice like "flush vs barrier" only exists *because* a specific hazard ("source L1 reused with different value") can be mitigated two ways. **This is where the bake-off backlog is born:** flag every hazard whose mitigation has ≥2 viable options — each such hazard is a style fork to bake off, and its mitigation column tells the bake-off what to vary and which matrix region to stress.

### Checkpoint — after Step B
Walk each hazard/invariant with a concrete firing example. Call out the hazards with ≥2 viable mitigations — those *are* the bake-off backlog. Wait for OK.

---

## Step C — Block discovery — SKIPPED

The block is given in Step 0, so there is no discovery, no 4-line gate, and no candidate selection. **The job the 4-line gate used to do — "is this block real and fat enough to deserve a helper?" — moves to Step ★**, which validates the *API over the given block* instead of the block itself. If Step ★ finds the premise is wrong, it loops back to Step 0 (the analog of design-helper looping C back to 0).

---

## Step D — Kernel annotations

### Scope — annotate the FULL census (this is a pre-migration audit)

Step B's catalog is kernel-agnostic, so the **forks are already complete** regardless of which kernels you read — a sample would suffice *for the style decisions*. But Step D has a second, bigger job here: **discover migration blockers before you commit to touching the ops.** For that you do want the **full census** — *every* kernel that uses the Step 0 primitives — because the surprises that derail a migration (a call site whose block won't fit the API, an outlier that needs a refactor, a kernel that should stay on the raw API) only show up when you actually look at all of them.

So: annotate **all call sites of the primitives**, not a sample. Because that's a lot of kernels, **fan it out — one annotation subagent per group, in parallel.**

### D.1 — Census + grouping

The census is of **the block, not of a primitive name** — and **recall matters more than precision here.** Missing a differently-spelled mcast means it never enters the pattern set and silently breaks at migration. So the grep is a *starting net*, not the boundary; the boundary is "every kernel that does the block, however it's spelled."

**Bias for recall, in two stages then a sweep:**

1. **Anchor grep on the diagnostic *family*** — every spelling of the block's invariant operation, not one name. From the Step A contracts (which enumerate raw API **and** the wrapper-layer methods that delegate to it), assemble the family:
   - all multicast-send spellings: `noc_async_write_multicast`, `*_multicast_loopback_src`, `*_one_packet`, …
   - the `kernel_lib` **wrapper** forms (`Noc` / `Semaphore<>` methods) as well as the raw calls;
   - the handshake in any form (`noc_semaphore_set_multicast`, or `noc_semaphore_inc` + `noc_semaphore_wait` / `_wait_min`);
   - the tells of an **open-coded** mcast (`get_noc_multicast_addr`, multicast NOC-address construction) for kernels that never call a named multicast helper at all.
   Discover the family **bidirectionally** (like the old llk catalog): grep the names bottom-up **and** scan the wrapper/header layer top-down to *learn* names you didn't know to grep, then cross-reference. **Never anchor on a fork primitive** (flush/barrier/counter-vs-flag) — a kernel using one variant still has the block.
2. **Block-presence filter.** Keep a candidate only if the characteristic cluster **co-occurs as a recognizable sequence** (the block), not an incidental lone use (e.g. a bare `noc_async_write_barrier` on an ordinary write). Drop incidental users; exclude the helper itself and the test / `bakeoff_*` kernels.

Then **group by op-family *directory*, and make the subagent sweep the whole directory — not just the grep hits.** A subagent owning all of matmul's kernels reads them and recognizes a multicast *by what it does*, catching one that grep missed. The grep hits decide *which* op-families are in scope; the subagent's read decides *which kernels in them are actually the block*. Groups: e.g. matmul, conv, layernorm, sdpa, groupnorm/group-attn, data-movement, deepseek-moe — the unit of parallelism and of the migration audit.

**Write the result to a durable, reusable manifest — `census.txt`** — in the design-helper kernels-of-interest format (one path per line, grouped by op-family with `#` headers, optional `— role:` suffix). This is the list of the block's call sites, persisted so it can be **reused later**: a re-run can re-seed from it instead of re-grepping, `build-helper` can consume it as the migration list, and a future audit can diff against it when new kernels land. D.1 writes the initial list; D.3 finalizes it after recall converges (adding any swept-in kernels + each kernel's clean/refactor/defer tag). If `Known instances` were given in Step 0, **confirm the census contains them** — a known instance missing from the census means discovery is broken; fix the anchor family and re-run before proceeding.

(If you can't name a diagnostic invariant family to anchor on, the Step 0 block sketch is under-specified — tighten it before censusing.)

### D.2 — Dispatch one annotation subagent per group (parallel)

Launch **one subagent per group, all at once** (general-purpose). Each subagent, for its group:

0. **Sweeps the whole op-family directory for recall** — reads the kernels, not just the grep hits, and identifies every block-like sequence *by what it does*, including ones written with primitives outside the anchor family (a wrapper form, a different handshake, an open-coded multicast). **Any block found that the anchor grep missed is reported as a recall miss** with the new spelling it used.
1. Annotates every kernel line-by-line into `kernel_annotations/<kernel>.md` (which hazard it mitigates / invariant it establishes / protocol step it is). A line that won't classify is a hole — flag it.
2. Clusters its kernels into **patterns** — and a differently-spelled block is a pattern too; it must appear in the pattern set, never get dropped for not matching the grep.
3. Reports **catalog amendments** and **recall misses** back to the orchestrator — it does **not** edit the shared catalog itself (avoid write races).
4. Writes a **per-group migration audit** `migration_audit/<group>.md` with, per kernel: `clean` (fits the drafted block/API), `refactor` (fits but the current style disagrees with a likely winner — flag the cost), or `defer/raw` (outlier that shouldn't migrate this round — say why). This is the blocker inventory.

### D.3 — Consolidate

The orchestrator merges all subagent reports: applies catalog amendments **once, centrally** (single source of truth — no agent writes it directly), assembles the full pattern inventory, and rolls the per-group audits into `migration_audit/_SUMMARY.md` (counts of clean / refactor / defer, and the headline blockers). A line no subagent could classify, or a kernel that fits *no* pattern, is a signal the Step 0 block boundary may be wrong — loop back.

**Recall cross-check + re-census (don't skip this — it's the whole point of the census).** Collect every **recall miss** the subagents reported. For each new spelling found:

- add it to the anchor family in D.1;
- **re-grep globally** for that spelling — a wrapper form or open-coded mcast found in one op almost certainly appears in others, and those groups must pick it up too;
- record it: a new spelling that mitigates a hazard differently is a **catalog amendment** (possibly a new fork); a new block shape is a **new pattern**.

Iterate until a census pass surfaces no new spelling. Only a converged census — where every block-like sequence, however spelled, is in the pattern set — is allowed to reach Step ★.

**Step D does not own a separate fork list — it reconciles the one Step B already owns.** The hazards catalog from Step B is the single source of truth for the bake-off backlog. Annotation amends it (centrally, in D.3 — subagents *report*, the orchestrator *applies*) when real code demands:

- a kernel reveals a **hazard Step B missed** → add the row (and its mitigation options);
- a kernel uses a **mitigation Step B didn't list** → add it to that hazard's options (this can turn a single-option hazard into a new fork);
- a theoretical fork from Step B **never actually appears** and isn't plausibly needed → mark it `speculative` (still bakeable, but lower priority) rather than deleting it.

So annotation's job here is threefold: (1) keep the hazards catalog honest against real code, (2) cluster kernels into **patterns** (input to Step ★'s feasibility check — *not* a second fork list), and (3) produce the **migration audit** — the clean/refactor/defer inventory that surfaces blockers *before* migration rather than during it.

### Checkpoint — after Step D
Present the **migration audit** first — `migration_audit/_SUMMARY.md`: how many call sites are clean / refactor / defer, and the headline blockers (the whole reason we did a census). Report any **recall misses** found by the whole-directory sweep (differently-spelled blocks grep didn't catch) and confirm the census **converged** (the last re-census pass surfaced no new spelling). Then report **amendments to the hazards catalog** (added hazards / mitigation options / forks demoted to `speculative`), and summarize the **patterns** + which kernels match each — that's what ★ checks the API against. Wait for OK.

---

## Step ★ — API feasibility checkpoint (NEW)

**Goal:** this is where the API actually takes shape. Intent gave only a *vague* block sketch; now — with the annotated patterns from Step D in hand — draft the concrete 1–2 helper signature and decide whether it can actually be built over those patterns, *before* spending device time on a bake-off. This is the gate that inherits Step C's rejection power.

Artifact: `api_feasibility.md` — contains the **first concrete API draft** (promoted from the vague Step-0 sketch) plus the verdict below.

Map every annotated pattern onto the drafted signature and classify:

```
Does the drafted API cover this pattern?
├── YES, as sketched              → covered by the signature / an existing sketched knob.
├── YES, with revision            → needs something the sketch lacks:
│     ├── a USE-CASE difference    → add a knob (run merge-bias in Step E first).
│     └── a STYLE difference       → fine; Step E's bake-off resolves it (winner or dual-path).
└── NO                            → patterns diverge past a 1–2 helper API without
                                     becoming a config-blob. Premise is wrong.
```

**Outliers don't count against feasibility.** A call site the Step D audit marked `defer/raw` is *explicitly out of scope* — it stays on the raw API by choice. Judge feasibility over the `clean` + `refactor` set only; otherwise the census would let one weird kernel veto an API the rest of the codebase wants. (A large `defer` pile is still a yellow flag worth discussing — maybe the block boundary is off — but it isn't an automatic "infeasible.")

**API-shape smells to catch while drafting the signature (cheaper to fix here than after migration):**

- **Asymmetric faces want separate types.** If the block has two roles (a broadcaster and a
  listener, a producer and a consumer) and the two roles need *different* constructor inputs, that is
  a signal to draft **two types**, not one type whose listener-side ctor args are dead ("pass 1 by
  convention"). Per-call context one role needs and the other doesn't (e.g. the sender's coords, known
  only at receive time) belongs in a **method argument**, not a shared ctor field. A single object
  with role-irrelevant args is a config-blob in disguise — split it in the draft.
- **A caller-facing count must be statable from the caller's own topology alone.** If the only way a
  caller can fill a count arg correctly is by first reasoning about which internal mode the helper will
  pick (and pre-subtracting for it), the arg is wrong. Draft the count as the thing the caller *knows*
  (e.g. "all cores that receive the broadcast, including me if I'm one") and let the helper derive
  every mode-specific hardware count internally. The caller states intent; the helper does dispatch.
- **The helper should own the synchronization-primitive lifecycle.** Prefer a signature that takes
  primitive **IDs** and constructs + inits them inside, over one that demands pre-built, pre-initialized
  objects from the caller — the same encapsulation argument that puts the H11 reset inside the helper.
  If the draft requires the caller to init a semaphore to a magic value before first use, fold that init
  into the ctor. If the census shows the call sites uniformly pre-seed some value (even one the helper
  *writes* rather than waits on), give the ctor an arg defaulting to the most-frequent observed value so
  the common call site drops the line and the rare one overrides it. **Caveat — only race-free inits:**
  a cell can be ctor-initialized only if this core establishes a happens-before edge to every other
  writer before they write it. A cell a *remote* core writes (a counter others increment, a flag another
  core mcasts in) generally must keep its initial value from the host allocation, NOT a kernel ctor —
  initializing it in the ctor races (e.g. a receiver acks before the sender's ctor runs → the ctor wipes
  the ack → hang). Drafting it as a ctor init anyway is a hang waiting to happen at migration.
- **Classify every arg compile-time-template vs runtime — don't default everything to runtime.** For
  each value the helper takes, check the census for *how the kernels source it*: a value that is
  `get_compile_time_arg_val` **and** identical across every core running the kernel binary should be a
  **template / constexpr** param (type honesty + constant folding; the host can't pass a per-core value
  by mistake). A value is **hard-runtime** only if it is genuinely (a) per-core-varying under one
  compiled binary (e.g. each row-sender targets a different mcast rect — set per-core via runtime args),
  (b) a device-resolved address / pointer (CB read/write ptr), or (c) data-dependent / rotating
  per-iteration. Don't assume an arg is runtime just because the old open-coded kernel read it with
  `get_arg_val` — confirm it actually *varies*. (Watch the interaction with other decisions: a count
  that *was* per-core can become core-uniform once the helper infers the per-core part internally, which
  then makes it templatable.)

**Verdict, one of three:**

- **Feasible as drafted** → proceed to Step E. The bake-off backlog is just the hazards-catalog forks (hazards with ≥2 viable mitigations, as amended by Step D) that the drafted API leaves open.
- **Feasible with revision** → adjust the API draft in `api_feasibility.md`, re-checkpoint, then proceed. New style forks discovered here get added to the bake-off backlog.
- **Infeasible** → fails the "name it in one sentence without listing primitives" test, or needs >2 non-mergeable knobs. **Loop back to Step 0** — re-scope the block or rethink the API. Do not paper over it with knobs.

### Checkpoint — after Step ★
State the verdict per pattern and the overall verdict. If "feasible with revision," show the revised signature and the updated bake-off backlog. If "infeasible," explain which patterns broke it and propose how to re-scope. **This is a human-decision checkpoint** — the user must accept the verdict before any device work. Wait for explicit OK.

---

## Step E — Style bake-off (EXPANDED — the heart of this skill)

**Goal:** for each style-driven difference on the bake-off backlog, **decide the winner by measurement**, and — crucially — recognize when the honest answer is "it depends," promoting that difference to **two internal paths dispatched on a recognizable predicate** rather than forcing one global choice.

Artifact: `style_bakeoff.md` (supersedes design-helper's `pattern_comparison.md`).

### E.1 — Triage the one fork list (use-case vs style)

The bake-off backlog is **the hazards-catalog forks** — hazards with ≥2 viable mitigations (Step B, as amended by Step D). There is no second list to derive; do not re-scan the annotations for "observed differences." Take each fork and classify its mitigation options:

- **use-case-driven** (which mitigation is correct is *forced by what the caller's program does*) → not a bake-off, it's a **knob**; settle it on paper with merge-bias (design-helper Step E.2).
- **style-driven** (the caller doesn't force it; more than one mitigation is correct for the same situation) → **goes to the bake-off**.

Note the consequence of sourcing forks from the catalog rather than from observed kernels: a fork is bakeable **even if no existing kernel uses one of its mitigations.** That's the whole point — Step B knows `flush` and `barrier` both mitigate the hazard, so `flush` gets baked off even if every kernel today uses `barrier`. (Step D's `speculative` tag just means "bake it last.")

### E.2 — Build matched kernel pairs

For each style fork, generate a **matched pair** of raw-primitive micro-kernels (raw APIs, not the helper) that are **identical except the one style axis**. Example: `bakeoff_flush.cpp` vs `bakeoff_barrier.cpp` differ only in the post-set() sync. If two axes interact (e.g. flush×counter-sem), bake off the **cross product**, not each axis alone.

These kernels live under `tests/ttnn/unit_tests/kernel_lib/kernels/bakeoff_*` and are throwaway-but-kept — they carry over as a ready-made raw baseline when the helper is later validated.

### E.3 — Run the matrix, STAGED (this is what makes it empirical)

Device runs are **sequential** (no parallelism) and the matrix can be large — especially the cross-product of interacting forks. So run it in two passes, **coverage before perf, never perf-first**:

**Pass 1 — coverage screen (cheap, correctness only).** Run every variant across the coverage axes under `scripts/run_safe_pytest.sh --dev` so a hang is a first-class captured result (triage JSON), not a crash. Record pass / hang / mismatch per cell. **This pass alone resolves any fork where one variant covers cells the other fails** — that's a correctness verdict, no perf needed. Forks decided here never reach the perf pass.

**Pass 2 — perf pass (only on coverage-survivors).** A fork "survives" when both variants pass the *same* cells, so coverage can't pick a winner. Only then measure perf + L1 — and only on the cells both pass, only for the surviving variants. This is where the budget threshold from E.4 gets its numbers.

| | DATAFLOW track | COMPUTE track |
|---|---|---|
| **Coverage axes** (Pass 1) | rectangle shape · `num_dests` vs `num_cores` · sender placement (out-of-rect / skip-self / loopback / self-only) · multi-iteration | dtype (bf16/fp32/bf8b/mixed) · `fp32_dest_acc` · math_fidelity · feature-bool combos · num_tiles |
| **Coverage = pass means** | data landed bit-exact AND no hang | PCC ≥ threshold |
| **Perf** (Pass 2) | NoC latency in ns (tracy / `DeviceZoneScoped`) | kernel ns (tracy) |
| **Footprint** (Pass 2) | L1 bytes (sem slots, CB depth) | L1 bytes / DEST pressure |

The output of E.3 is a **coverage map** (pass/hang/mismatch per variant per cell) plus ns/L1 numbers **only where the screen left the decision open**. Most forks should be settled by Pass 1.

### E.4 — Decide, with the dual-path frontier

For each style fork, read the coverage map:

```
1. DOMINANT  — one variant passes ALL cells AND is best (or tied) on perf/L1.
   → Bake that single path in. Done. (design-helper's classic outcome.)

2. TRADEOFF + RECOGNIZABLE PREDICATE
   — the cheap variant passes only in region R; the safe variant passes everywhere
     but is slower by MORE than the dual-path threshold (see budget below); AND region R
     is characterized by a predicate the CALL SITE CAN KNOW
     (ideally constexpr: num_dests==1, sender-in-rect, source-const-across-iters, arch==WH).
   → INTERNAL DUAL-PATH: the helper branches on the predicate — cheap path in R, safe path
     elsewhere. Callers in the cheap region pay no penalty; callers in the safe region stay correct.
     This is the whole point of the skill: don't pick one globally, dispatch on what's recognizable.

3. TRADEOFF + NON-RECOGNIZABLE PREDICATE
   — cheap variant is conditionally correct but the call site CANNOT tell which region it's in.
   → Either (a) fall back to the safe variant globally (correctness wins), or
     (b) expose a caller knob with a DOCUMENTED PRECONDITION the caller must assert
         (e.g. "set fast=true only if your source is constant across iters").
     Default to (a) unless the perf gap is large and the precondition is easy to state.

4. NOISE TIE — both pass full coverage and the perf delta is below the noise threshold.
   → Don't let noise decide. Fall back to qualitative (simplicity, migration cost) + human pick.
```

**The recognizability test is the load-bearing gate.** A dual-path is only safe if the predicate selecting the cheap path is knowable at the call site — preferably `constexpr` from the caller's topology/config, or at worst a documented precondition the caller explicitly asserts. If you can't state the predicate crisply, it's case 3, not case 2. Never dispatch the cheap path on a guess.

**Dual-path budget — single-path is the default; a dual-path must earn its place.** A helper full of predicate-dispatched branches is just the config-blob design-helper bans, moved inside the function. So:

1. **Prefer single-path.** A fork only becomes a dual-path if it clears BOTH gates: the predicate is `constexpr`-clean (not a runtime precondition — that's case 3b, a knob) **and** the perf gap exceeds the dual-path threshold (default **≥10% / ≥ a meaningful ns delta**; below that, the cheap path isn't worth a branch → take safe-global).
2. **Cap at ~2 dual-paths per helper.** If more than ~2 forks want to dual-path, keep only the highest-value ones (largest perf gap × widest cheap region); the rest fall back to safe-global. More than 2 internal branches is a signal the helper is doing too much — reconsider the block boundary or split the helper.
3. **Independence check.** If two dual-path predicates interact (the cheap path of one is only safe given the other's path), you don't have 2 dual-paths — you have one 4-way branch. Count it against the cap as such, and usually that's the cue to collapse to safe-global for one of them.
4. **A fork whose only consumer is DEFERRED doesn't ship.** If every `clean`/`refactor` call site takes the same arm and the *other* arm exists only for a call site the Step D audit tagged `defer/raw` (out of scope this round), do not materialize the knob — bake in the single arm the in-scope set uses, and record that the fork is re-introduced as a **refinement** if/when that deferred call site is actually migrated. Carrying a knob for a kernel you aren't migrating is the speculative-config-blob failure wearing a "for safety" hat. (This is distinct from a Step-B `speculative` fork: that one had no observed user *yet*; this one has a user, but it's explicitly out of scope.)

**Perf here is a micro-benchmark — a screen, not a verdict.** The bake-off times the block in isolation; in a real op the slow path's wait may be hidden behind other work, so the isolated ns gap can *overstate* the in-context cost. A dual-path justified purely on micro-bench perf is therefore **provisional**: in `style_bakeoff.md` record the measured gap and the threshold it cleared, and mark the dual-path "confirm against the real op; collapse to safe-global if the in-context gap falls below threshold." A coverage-decided fork (case 2 where the safe path was *required* by failing cells, or case 1) is not provisional — correctness verdicts stand.

### E.5 — Surface, don't bury

Record per fork: the coverage map (a small table), the perf/L1 numbers, the outcome (dominant / dual-path / knob / global-safe), and — for dual-paths — the exact predicate and where it's evaluated. Any refactor cost (a chosen winner that disagrees with an existing kernel) is flagged as a human-review item, never silent.

### Checkpoint — after Step E
**Gate-2-equivalent, in two beats.** Beat 1 (before device work): present the bake-off *plan* — the matched kernels, the matrix, the metrics — and get an OK to run on device. Beat 2 (after runs): present the coverage maps and the per-fork outcomes, especially each proposed **dual-path with its predicate**, and get explicit approval of each winner/dual-path before Step F. Wait for OK at both beats.

---

## Step F — Helper proposal

**Same as design-helper's Step F, with one addition.** Write `proposed_helpers.md`: 1–2 fat helpers, use-case differences as the only caller-facing knobs, all *dominant* style choices baked in. Run the **fat-helper gate** (≥4 lines replaced · absorbs ≥1 hazard · nameable without primitives · ≤2 helpers).

The addition: **internal dual-paths from Step E.4 case 2 are first-class in the proposal**, documented as *implementation dispatch*, not caller knobs:

```cpp
// USE-CASE knob: sender in receiver grid?  (caller sets this)
template <McastMode Mode>
void mcast_block_with_handshake(...) {
    // INTERNAL dual-path (empirically justified, style_bakeoff.md §flush-vs-barrier):
    //   predicate = (Mode == Loopback || num_dests == 1)  [constexpr, recognizable]
    //   cheap path (flush)  used when predicate is false  -> proven over cells {…}, N ns
    //   safe path (barrier) used when predicate is true    -> required by cells {…} (cheap path hung)
    if constexpr (needs_ack<Mode>()) { /* barrier path */ } else { /* flush path */ }
}
```

For each helper, document: use-case sentence (no primitive names), baked-in style choices + one-line justification *citing the bake-off*, use-case knobs, **internal dual-paths + their predicates**, preconditions, postconditions, and the migration list with refactor costs flagged.

Also commit, in the proposal, to the **implementation contract** the materializer (`apply-helper` Phase 1) will enforce — so it isn't rediscovered at build time:
- **Built on the object/2.0 API only** — name the object layer it sits on (`Noc`, `Semaphore<>`, the endpoint types) and state that NO raw free function survives in the body, including the mcast send and any address math. If the object API is missing a needed overload, say so here as a known gap.
- **Object split** — if Step ★ split the block into separate types per role, list them and what each face's ctor takes vs. what its methods take.
- **Caller-facing counts** — state each count as the caller states it (topology-only), and note that the helper derives the mode-specific hardware counts internally.
- **Synchronization-primitive lifecycle** — state that the helper takes IDs and constructs + inits the primitive it waits on inside the ctor.

### Checkpoint — after Step F
Confirm each helper passed the fat-helper gate (cite line count + hazard absorbed). Summarize each helper in one sentence + its knobs + its internal dual-paths and predicates. List refactor costs. Ask: *"Is this the helper you want to build?"* On sign-off the skill exits — implementation is out of scope here.

**Hand off:** the sign-off on `proposed_helpers.md` **is Gate 1 of `build-helper`**. The bake-off kernels under `tests/ttnn/unit_tests/kernel_lib/kernels/bakeoff_*` carry over as a ready-made raw baseline, and `style_bakeoff.md` carries the coverage/perf evidence already gathered. Continue with `build-helper` to take the proposal through test plan → validation → migration → report — including the real-op confirmation of any provisional dual-path (see E.4).

---

## Output directory contract

```
helper_design/<helper-name>/
├── intent.md                  (Step 0 — just two things: primitives to find + vague block sketch)
├── primitive_contracts.md     (Step A — prompt-driven, reusable; records PRIMITIVES: list)
├── hazards_catalog.md         (Step B; or invariants_catalog.md for compute)
│                              (Step C skipped — no block_discovery from scratch)
├── census.txt                 (Step D — durable, reusable manifest of every block call site, grouped + tagged; kernels-of-interest format)
├── kernel_annotations/
│   └── <kernel>.md            (Step D — full census, one per call site; written by per-group subagents)
├── migration_audit/
│   ├── <group>.md             (Step D.2 — per-group clean/refactor/defer inventory)
│   └── _SUMMARY.md            (Step D.3 — rolled-up blocker counts + headline blockers)
├── api_feasibility.md         (Step ★ — the feasibility verdict, checked against the full census)
├── style_bakeoff.md           (Step E — coverage maps, perf/L1, dual-path decisions)
└── proposed_helpers.md        (Step F — the deliverable)

tests/ttnn/unit_tests/kernel_lib/kernels/bakeoff_*   (Step E matched kernels — carry over as the raw baseline for later validation)
```

---

## Anti-patterns to avoid

```
❌ Deciding a style fork by argument when you could just run both on device.
   → The entire reason this skill exists is to replace eloquence with a coverage map.

❌ Picking one global style across a coverage/perf tradeoff.
   → If the cheap path is correct in a recognizable region, dual-path it. Don't tax the
     whole codebase for a hazard that only fires in some topologies.

❌ Dual-pathing every fork "to be safe / fast."
   → The opposite failure: a helper that's a nest of `if constexpr` is the config-blob
     moved inside the function. Single-path is the default; a dual-path must clear the
     threshold AND the constexpr-predicate gate, and there's a ~2-per-helper cap. Below
     threshold or non-constexpr predicate → safe-global.

❌ Dispatching the cheap path on a predicate the call site can't actually evaluate.
   → That's a footgun, not an optimization. If you can't state the predicate as constexpr
     or a documented precondition, fall back to safe-global (case 3a).

❌ Turning every style fork into a caller knob.
   → Use-case differences are knobs. Style differences are bake-off winners or internal
     dual-paths. A style fork should almost never leak to the caller.

❌ Letting perf noise decide a tie.
   → Below the noise threshold with equal coverage, decide on simplicity/migration, not ns.

❌ Trusting the anchor grep as the census boundary.
   → Grep is recall-limited: it misses wrapper forms, alternate handshakes, and
     open-coded multicasts. The boundary is "does the block, however spelled" — so
     subagents sweep whole op directories and recognize blocks semantically, recall
     misses feed back into a re-grep, and the census must converge before ★. A block
     dropped for not matching the grep is exactly the kind of bug this census exists
     to prevent.

❌ Skipping Step ★ because "we already have the API."
   → The API is a sketch until the annotations prove it covers every pattern. ★ is where
     the premise gets to be wrong cheaply, before any device time is spent.
```
