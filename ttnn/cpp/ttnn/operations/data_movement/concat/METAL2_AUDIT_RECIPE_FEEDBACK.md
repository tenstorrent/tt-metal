# Feedback on `audit/metal2_audit.md` — from an audit of `data_movement/concat`

**For:** the recipe maintainer / recipe-author Claude
**From:** an auditing Claude, cold (no prior project context loaded), running the recipe end-to-end for the first time
**Recipe version:** `0846547f407 2026-08-27 docs(metal_2.0): a run in flight freezes the kernel sources`
**Op audited:** `ttnn/cpp/ttnn/operations/data_movement/concat` — 1 DeviceOperation, 6 program factories, 9 own kernels + 2 borrowed donors
**Outcome reached:** RED at op level, with a clean multi-factory subset; both output documents produced

> This is the same feedback as the `Recipe notes` section of `METAL2_PREPORT_AUDIT.md`, plus the parts
> that don't belong in a per-op report: what the recipe got *right*, priority ordering, and notes on
> the run itself. Kept as a standalone file so it can travel without the audit.

---

## How the run went, overall

The recipe worked. I never felt lost, the subject order made sense, and the deliverable shape was clear
by the time I reached it. Most of what follows is friction at the margins, not structural doubt.

The single most important thing to know: **almost everything below is about ambiguity, not about
missing content.** In each case the recipe says enough for a careful agent to reach *an* answer; the
risk is that two careful agents reach *different* answers and both believe they followed the recipe.
That's the failure mode worth designing against for users who can't intervene.

---

## Friction, ordered by how badly it could hurt an unsupervised run

### 1. Whether a config-scoped RED issues a brief is stated both ways — HIGH

This is the one that could change the deliverable.

- `:74` (Red outcome scoping rule): when a clean factory subset survives, run the informational
  subjects for that subset — "**its brief needs them**."
- `:111` (the four roles): "A *config-scoped* GATE — e.g. GlobalCircularBuffer confined to one factory —
  **still issues a brief for the clean subset**."
- `:507` (Output: the two documents): the brief is "**emitted only on a fully GREEN audit** (every gate
  cleared) … On any **RED** there is no brief — there is no port yet."
- `:639` (brief template header): "Issued only on a fully GREEN audit — every gate cleared. **Never on
  RED.**"

Concat lands exactly on the seam: op-level RED, several factories clear. I emitted a subset-scoped brief
with a prominent scope box, reasoning that `:74`'s rationale is void if no brief can exist. But a Claude
that weighted `:507`/`:639` instead would have produced one document rather than two — a visibly
different result from the same inputs, with no way for either to know it had diverged.

**Suggested fix:** amend `:507` and `:639` to carry the `:111` carve-out explicitly, e.g. *"…or, on a
config-scoped GATE, a brief scoped to the clean subset — see Code-path scope."* Also worth stating
whether the subset brief needs a scope banner; I invented one, and it seemed too important to leave to
taste.

### 2. `set_globally_allocated_address` is named as *the* borrowed-memory signal, but descriptor-API ops don't use it — HIGH

This is the most likely silent wrong answer in the whole recipe, because the failure is a *grep that
returns nothing*.

The causal-link gate (`:300`) and the false-positive guards (`:331`, `:332`) both name host-side
`set_globally_allocated_address(buffer)` as the recognition signal for borrowed memory. But the
`ProgramDescriptor` API — which the audit repeatedly calls "the in-scope case" — expresses borrowed
memory as `.buffer = tensor.buffer()` on a `CBDescriptor` instead.

Concat has **zero** occurrences of `set_globally_allocated_address` and **thirteen** `CBDescriptor`
literals with `.buffer` set, across five of its six factories. An auditor who greps the documented
signal concludes "no borrowed memory here" and pushes every one of those bindings into Case 1 or Case 2
— inventing port work that doesn't exist, and losing the `borrowed_from` translation that is the
actual answer. Nothing downstream catches it: the port would just be wrong in a plausible-looking way.

**Suggested fix:** name both spellings wherever the signal appears — the imperative
`set_globally_allocated_address(buffer)` **and** the descriptor-form `CBDescriptor::buffer` field —
and add the descriptor form to the false-positive guard list. Given the guide's stated audience is
descriptor-API ops, the descriptor spelling arguably belongs first.

### 3. The status-summary template has a `Variadic-CTA` row with no Appendix A entry behind it — MEDIUM

`:557` prescribes `| *Feature Support* — Variadic-CTA | Ok / Unsupported |`. Appendix A holds exactly
three entries (GlobalCircularBuffer, `address_offset`, GlobalSemaphore), and the Gate-detail feature
table at `:590-594` lists only those three. Since the maintenance rule (`:710`) says a feature that
gains support has its entry **removed**, this row reads as a survivor of an older Appendix A.

It bit specifically here. Concat *does* use a genuine CTA vararg — `reader_concat_stick_layout_
interleaved_start_id.cpp:57,70` reads `kernel_compile_time_args[base + curr_tensor]` at a runtime index
— and the RTA-varargs subject (`:496`) explicitly says CTA varargs "don't gate either" and port onto
`compile_time_varargs`. So for the one op where the row has real subject matter, neither `Ok` nor
`Unsupported` is an honest cell. I filled it with a pointer to a note.

**Suggested fix:** delete the row, or replace it with an explicitly non-gating cross-reference to the
RTA-varargs subject.

### 4. The endpoint census doesn't say whether *constructing* a DFB object without accessing it is a touch — MEDIUM

`:378` defines an endpoint as a kernel that FIFO-produces, FIFO-consumes, "**or** accesses the memory by
**raw pointer**." `:378` then reasons separately that "in Metal 2.0 a kernel cannot touch a DFB it
hasn't bound, so every access is a binding, hence an endpoint."

Concat's tiled compute kernel constructs `DataflowBuffer output_dfb(output_dfb_id)` and never uses it
(`height_sharded_width_concat_two_tensors.cpp:57`). Two defensible readings:

- **Access test** (the rule as written): not a toucher → the output CB's census is 1 → **self-loop**.
- **Needs-a-binding-to-compile** reading: a toucher → census 2 → **1P+1C**.

I took the access test and flagged it in the brief, but a real binding decision hangs on one unstated
sentence, and the two answers produce different `dfb_bindings`.

**Suggested fix:** one sentence in the census — construction alone is (or is not) a touch. If the answer
is "not a touch, and the dead local should be deleted," saying so also tells the porter what to do with
it.

### 5. "Which side does the blocker clear on?" has no anchor for free-text `Known op issues` cells — MEDIUM

The `:76` exception ("a RED that clears without touching the op's code: run them anyway") is crisp for
the enumerated blockers — it names Device 2.0 migration, an offset split, a PD migration as op-code-side,
and an unattributed verdict or an Appendix A feature landing as elsewhere. But **all four** of concat's
blocks arrived through `Known op issues`, which is free text by design.

`DFB misuse; will need semi-manual port` — is that an ops-team kernel rewrite (op-code side → skip the
seven subjects) or a semi-manual port (elsewhere → run them)? I ran them, because the phrase points at
the *port* absorbing the work and because running was cheap with the code already in hand. But I was
guessing, and I said so in the report.

**Suggested fix:** add a line to `:76` for the free-text case — e.g. *"a `Known op issues` cell is
free text; decide the side from what the cell says will change, and if it doesn't say, run them and
record the judgment."* The default-to-running bias seems right: over-running costs tokens, under-running
costs a full second pass.

### 6. No guidance for a `kernel_source` that resolves to no file — LOW, but high value

The Scope section (`:85`) covers *unreferenced* kernel files in the op directory. Concat has the mirror
case: `ConcatS2IProgramFactory` binds `.../kernels/dataflow/reader_s2i_width.cpp`, and **no file of that
name exists anywhere in the repository.**

That single check was the strongest evidence in the whole audit — it upgraded the sheet's "this factory
is dead code" from a claim I was reading to a claim I could prove independently, and it's what let me
recommend deletion with confidence. I filed it as an incidental anomaly, which felt right but was a
judgment call.

**Suggested fix:** a line in Scope — *if a `kernel_source` path resolves to no file, record it as an
anomaly and treat that factory as non-auditable* — plus a mention that it's worth checking, since it's
one `find` and it decided a whole factory's disposition here.

### 7. The recipe offers subset ports as first-class but never says the mixed-concept variant is legal — MEDIUM

`Code-path scope` (`:525`) presents a scoped-subset port as a normal, encouraged outcome, and the
finding-roles section has a config-scoped GATE issue a brief for the clean subset. But a subset port
necessarily leaves the device-op's `program_factory_t` `std::variant` holding a **mix** of converted
(`MetalV2`) and unconverted (`descriptor`) factories — and nothing in the recipe says whether that is
supported.

Concat hits it head-on: three of six factories convert, so `ConcatDeviceOperation`
(`concat_device_operation.hpp:29-35`) carries a 3/3 split. I could not settle it from the audit's own
materials, so I raised it as a question and had the brief open with a *confirm before you start* — which
is the wrong shape for a precondition of an outcome the recipe actively recommends. (It **is** supported;
the answer came back from the framework owner and both documents now state it as settled.)

**Suggested fix:** one sentence in `Code-path scope` confirming that a device-op may hold a mixed set of
factory concepts across a partial port, so the auditor asserts it rather than asking. If there are
caveats, that's the place for them.

### 8. The sheet's "read the cell, don't vet it" rule has no branch for *"a human tells you the blocker cleared"* — MEDIUM

`Is able to port?` is derived, the CSV carries values not formulas, and the recipe is emphatic: read it,
don't vet it, and a verdict you can't attribute is still a verdict. That's clear and I followed it.

What isn't covered is the case where the invoker says a blocker has since been resolved. The blocking
column empties, but `Is able to port?` — the cell that actually gates — still reads `no` in the fetched
CSV, and the auditor cannot recompute it. Strictly following "read the cell" gates a factory the invoker
has just said is unblocked; ignoring it re-derives a formula the recipe forbids touching.

I resolved it from evidence inside the fetched data — two sibling rows with an empty `Known op issues`
and *identical* values in every other column both read `yes`, so an emptied cell on an
otherwise-identical row implies `yes` — and documented the override in a callout box with an action for
the sheet owner. That inference turned out to be correct, but it was mine to make and it could as easily
have gone the other way.

**Suggested fix:** a short branch in the readiness doc or the factory-concept subject: when a blocker is
reported cleared out-of-band, treat the derived cell as stale rather than authoritative, state the
override and its basis explicitly in the report, and route a refresh request to the sheet owner. Naming
the sibling-row comparison as the sanctioned basis would make it reproducible instead of improvised.

### 9. Minor: the "shapes to census" hints for concat were half-right

`:423` predicts concat `S2SRM` / `S2SMulti` / `BlockSharded` "expect 1P+1C" — which held exactly, on all
three, and was genuinely orienting. It also predicts concat `S2S`-tiled "has compute → a co-touched
intermediate may hit ≥3 touchers." The tiled factory's maximum census is **2**: reader, writer and
compute form a clean 3-stage chain in which each intermediate is touched by exactly two of the three.

The hints are framed as prompts rather than verdicts, so this isn't a defect — recording it only in case
the example list is meant to stay calibrated against the code.

---

## What the recipe got right, and shouldn't lose

Worth saying explicitly, because these are the parts that most changed my output for the better:

- **The sanctioned-free-function paragraph (`:128-130`) saved the Device 2.0 gate.** Concat contains
  exactly two CB-index free functions — `get_tile_size(cb_id_in)` and
  `get_local_cb_interface(cb_id_out)` — and in *both* cases a `DataflowBuffer` is in scope and the DFB
  exposes its own equivalent. That is precisely the "holdover" shape from the Red bullet. Without the
  emphatic *"sanctioned still means sanctioned … Kernels already on `DataflowBuffer` are where that cue
  misfires hardest"*, I would have filed two false violations and RED'd a gate that is actually GREEN.
  The paragraph reads as over-written until you hit the exact case it was written for; keep it.

- **"Never let 'not in the tables' stand in for 'scanned and clean'" (`:250`, `:342`) mattered.** Concat
  is in neither dated triage doc. Both subjects still needed real work: the offset scan resolved to a
  grep-verified *zero* `->address()` sites op-wide, and the 3rd-arg scan needed me to chase
  `make_tensor_accessor_tuple` into `tensor_accessor.h:626` to confirm it constructs two-argument
  accessors. Two clean verdicts, both earned rather than inherited. The doc-vs-sheet contract
  distinction (authoritative vs. dated prior) was clear and I never had to guess which I was holding.

- **The `Buffer*`-binding-form guidance (`:315`) was calibrated exactly right.** Concat's interleaved
  factory pushes `Buffer*` objects into RTA lists. The instruction to *enumerate it but not overstate
  the urgency* is what kept a routine port-work item from being written up as a correctness hazard —
  and the sheet's `Op Classification` of `PD Op (pointer-patching)` then made sense rather than alarming.

- **The "distrust a `(0,0)`" framing (`:401`) and the asymmetry argument behind it** changed how I
  hunted. I found no dead CBs, but I went looking for indirect index paths rather than trusting a first
  pass, specifically because the recipe explained that the validator catches a missed-dead CB loudly
  while a wrongly-dropped live one fails silently. Stating the asymmetry, not just the rule, is what made
  it stick.

- **The `:76` "run them anyway" exception paid off immediately and concretely.** I judged
  `ConcatProgramFactory`'s blocker to clear elsewhere (a framework feature) and ran the seven
  informational subjects for it. That feature has since landed — and because the detail already existed
  against unchanged code, promoting that factory into the clean subset was an edit, not a re-audit. The
  rule did exactly what it was designed to do, on the first op I tried it on. (See #5 above for the one
  gap: the test needs an anchor for free-text cells.)

- **The `Questions for the user` section earned its place.** Both #7 and #8 above went into it, both came
  back answered, and both are now settled facts in the body of the audit and brief rather than caveats
  the porter has to resolve. That is the section working exactly as intended — worth noting because it
  is easy to treat as a dumping ground for uncertainty rather than as a channel that actually closes.
  The thing that made it work was writing each question with the `file:line` context that prompted it,
  as the template asks: both were answerable in one line because of that.

- **The readiness-sheet fetch doc pre-warns that the oversized result returns with an `Error:` prefix.**
  That warning is load-bearing — the tool result genuinely reads as a failure, and without the
  heads-up I'd have retried or reported the fetch as broken. Keep it prominent.

---

## Notes on the run itself

- **Recipe length:** 822 lines, which the Read tool paginated into three calls (the first capped at
  ~319 lines with a truncation notice). Not a problem — the notice was clear and the section
  headers made it easy to resume — but worth knowing that an agent with a smaller read budget meets the
  document in pieces, and the cross-references between distant sections (the Red-outcome rule at `:74`
  governing subjects defined 300+ lines later) are the parts most likely to be missed on a partial read.

- **The Drive fetch worked first try** in a session flagged non-interactive, with no OAuth wall. The
  four-step procedure in `ttnn_op_porting_readiness.md` was followed literally with no improvisation.
  One small thing: the header row contains embedded newlines inside quoted cells, so `head -1` shows a
  truncated header. A one-line note to parse with a real CSV parser rather than `head`/`cut` would save
  the next agent a confused minute.

- **The "reference by header name, never position" rule earned its emphasis.** The sheet I fetched has
  28 columns, and several audit-relevant ones (`Is able to port?` at index 15, `TensorParameter
  relaxation` at 19) sit nowhere near where a naive reading would put them. Matching on the distinctive
  stem also mattered: the real headers carry parentheticals and embedded newlines
  (`Custom hash \n (compute_program_hash)`, `Override runtime args method? \n (PD only)`), so an
  exact-string match on a remembered full header would have missed them.

- **`experimental/quasar/` stayed out of bounds** and cost nothing to avoid — but the warning was
  well-placed: broad greps for concat kernels and for `_metal2` forks both surfaced quasar hits, and
  the *locational* fork test (`ls` the original's directory, don't tree-grep the filename) is what kept
  them out. That distinction is easy to skip past on a first read and is exactly what makes the
  difference; both real forks concat needs are genuine siblings, and a filename grep would have mixed
  them with quasar copies.

- **One thing I'd have liked and didn't need:** a worked example of a *mixed* factory verdict. Concat's
  sheet rows split `yes`/`no` within one DeviceOperation, which drives Code-path scope, the subset
  brief, the per-factory informational-subject decision, and the Result line format all at once. The
  recipe covers each of those individually and the composition worked, but the composition is where
  most of my judgment went, and it's the case most likely to recur on multi-factory ops.
