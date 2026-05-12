# Proposed Edits — Eltwise Helper + LLK Pipeline Docs

Source: 10 lessons from eltwise helper iteration (`eltwise_run6` branch).

Three target docs, divided by scope. **No duplication across files** — each lesson lives in exactly one home; other docs reference it.

- `llk_helpers_hq.md` — cross-helper rules (general principles + migration rules)
- `eltwise_helper_lessons.md` — eltwise-specific surfaces (op structs, chain element tags, eltwise policies)
- `llk_helpers_pipeline.md` — concrete phase steps that point at the hq rules

Status: **proposal only**. Confirm per section before applying.

---

## Scope Division

| # | Lesson | Home | Other docs |
|---|---|---|---|
| 1 | CB id is `uint32_t` at the API boundary; `experimental::CircularBuffer` supported internally | hq (general principle) | — |
| 2 | No `timeout` wrapper on safe pytest / tt-probe | hq (Validation Tests) | — |
| 3 | `with_dt_tree` reconfig is first-class | hq (general principle) | — |
| 4 | `compute_kernel_hw_startup` necessity investigation | pipeline Phase 1 sub-step → result documented in helper `.hpp` doc-comment | — |
| 5 | Struct: runtime via ctor, compile-time via template | eltwise §1.2 (already exists, strengthen) | — |
| 6 | Unary broadcast as eltwise chain element (NOT a separate helper) | eltwise §2.8 (chain element) | — |
| 7 | Ternary + quaternary CRTP bases | eltwise §1.1 (specific surface) | — |
| 8a | Fill / rand tile tag distinct from CopyTile | eltwise §1.10 (specific tag) | — |
| 8b | Slim context before migration | hq (Kernel Migration Steps, pre-Step-1) | pipeline Phase 5 (Implementation) → ref hq |
| 9 | One representative pytest per kernel during migration | hq (Step 4 rule) | pipeline (Phase 4 sub-loop step → ref hq) |
| 10 | Catalog-vs-coverage audit (remediate, not just log) | hq (self-maintenance rule) | pipeline (Phase 6 step → ref hq) |
| 11 | Find tests covering the migration path before audit | hq (Kernel Migration Steps, new Step 1; renumber rest) | — |
| 12 | Commit proposal / test plan artifact after gate sign-off | hq (Approval Gates) | — |
| 13 | One eltwise helper with specialised convenience entry points (not standalone helpers per pattern) | eltwise §3.8 (architecture rule) | — |

---

## File 1 — `llk_helpers_hq.md`

### 1.1 [NEW under "Helper Design Principles (general)"] CB id abstraction — Lesson 1

Insert after "DEST capacity is compile-time, never literal":

```markdown
### Helper API takes `uint32_t` cb id; `experimental::CircularBuffer` supported internally

Public helper signatures take **only** the raw `uint32_t` cb id —
never an `experimental::CircularBuffer` reference, never a templated
CB-type wrapper. The single `uint32_t` parameter is the helper's
contract with the caller.

Internally, the helper is free to use `experimental::CircularBuffer`
machinery — e.g. constructing an `experimental::CircularBuffer` from
the id to query metadata, drive iterator-based access, or invoke
features only the experimental type exposes. That is an implementation
detail, not part of the API.

Reason: kernels mid-transition to `experimental::CircularBuffer` and
legacy kernels on the bare uint id share one API surface. The id is
what every kernel already has; constructing the experimental wrapper
inside the helper is cheap. Lifting the type into the signature
forces every caller to construct one (or maintain two helper variants),
and the surface drifts as some helpers take the wrapper and others
don't.

Where the helper queries CB metadata (page size, num pages, dtype),
prefer compile-time `static_assert` paths when the metadata is reachable
through the `experimental::CircularBuffer` constructed from the id,
falling back to runtime reads when it is not. `if constexpr` selects
between paths.
```

---

### 1.2 [NEW under "Helper Design Principles (general)"] Reconfig is first-class — Lesson 3

```markdown
### Reconfig is a first-class helper capability

Every helper that bridges init / dtype / format state must expose
reconfiguring as an in-helper option (policy enum, not caller-side
fixup). Use `with_dt_tree`-style decomposition to map each reconfig
variant to its underlying LLK calls — the mapping is the canonical
reference for migration and pattern matching.
```

---

### 1.3 [EDIT "Validation Tests"] No `timeout` wrapper — Lesson 2

Append to the "Run via:" block:

```markdown
Do NOT prefix with the shell `timeout` command — `run_safe_pytest.sh`
and `tt-probe.sh` have built-in dispatch-timeout detection that runs
`tt-triage` and emits the JSON triage report. An outer `timeout` kills
the wrapper before triage runs and the hang artifact is lost.
```

---

### 1.4 [EDIT Step 4] Incremental testing during migration — Lesson 9

Add after the dtype-matrix bullet:

```markdown
- During a multi-kernel migration cycle, run ONE representative pytest
  per kernel (the row from the Pytest Manifest). Defer the full
  kernel_lib suite + cross-helper regression set to the end of the
  cycle. Per-kernel full-suite runs multiply wall time without adding
  coverage.
```

---

### 1.5 [NEW pre-Step-1 under "Kernel Migration Steps"] Slim context before migration — Lesson 8b

Insert as a pre-step before "### Step 1 — Audit the target kernel":

```markdown
### Step 0 — Slim context before migration

Before auditing the first kernel of a migration cycle, drop stale
context that would mislead pattern matching:

- Archive prior `agent_logs/` entries from helper-creation runs that
  preceded this migration cycle.
- Strip closed gap-map entries; keep only open `GAP-N` rows that the
  migration may hit.
- Drop superseded proposal artifacts and old investigation outputs
  that no longer reflect the helper's current API.

Reason: bloated migration context surfaces dead op names, rejected
enums, and pre-redesign claims as phantom requirements during audit.
Pipeline-wide cycles (catalog → implementation) do NOT need this
reset between phases — only the migration phase does.
```

---

### 1.5b [NEW Step 1, elevate from buried sub-task] Find tests before audit — new lesson

Reorder migration steps so test discovery is **explicit Step 1**,
before kernel audit. Currently the Pytest Manifest section says "Step 1
(audit) | If the target kernel is missing from the manifest, find its
pytest..." — that buries test discovery as a side-task of audit. Elevate.

```markdown
### Step 1 — Find tests that exercise the migration path

Before reading a single LLK call in the kernel under migration:

1. Look up the kernel's pytest in the [Pytest Manifest](#pytest-manifest).
2. If the manifest row is missing, run the discovery procedure
   ([Pytest Manifest → Discovery procedure](#pytest-manifest)) and
   append the row before continuing.
3. Verify the test JIT-compiles THIS exact kernel file (per
   [Verifying the Test Exercises the Changed Kernel](#verifying-the-test-exercises-the-changed-kernel)).
   Same-named files in other directories silently shadow the one being
   migrated; finding out post-migration that the test never touched
   the kernel is the worst failure mode.
4. Run the test once UNCHANGED to establish a green baseline. A test
   that was already failing pre-migration is not a regression detector.

Only after a green-baseline test exists does the migration proceed to
audit (Step 2). Migration without a known-exercising test is
unsupported — the result cannot be verified.
```

Then renumber:
- existing "Step 1 — Audit the target kernel" → **Step 2 — Audit**
- existing "Step 2 — Gate-check" → **Step 3 — Gate-check**
- existing "Step 3 — Write the migration" → **Step 4 — Write the migration**
- existing "Step 4 — Verify on device" → **Step 5 — Verify on device**
- existing "Step 5 — Record the migration" → **Step 6 — Record the migration**

Update the Pytest Manifest "When each step writes to it" table
accordingly so "Step 1 (audit)" → "Step 1 (test discovery)" with
write-on-create-row responsibility, and "Step 4 (verify)" → "Step 5
(verify)".

---

### 1.6 [NEW under "Approval Gates"] Commit on acceptance — new lesson

Insert as a new subsection after "### What counts as 'explicit approval'":

```markdown
### Commit the artifact on acceptance

The moment a proposal artifact (Gate 1: `{category}_helper_proposal.md`)
or a test plan artifact (Gate 2: `{category}_test_plan.md`) clears its
gate, **commit the approved file before starting the next phase**. The
commit message names the gate and the user's approval message.

Reason: the artifact must be git-permanent at the version that was
approved. If implementation reveals a needed delta, the new version is
a *new* commit that re-enters the gate — not an in-place edit of the
approved file. In-place edits dissolve the audit trail of "what was
actually signed off."

Concretely:

1. After Gate 1 sign-off, commit `{category}_helper_proposal.md`
   verbatim, then begin Phase 3.5 (Test Plan).
2. After Gate 2 sign-off, commit `{category}_test_plan.md` verbatim,
   then begin Phase 4 (Validation).
3. If the user lists deltas instead of a clean approval, revise the
   artifact, re-post for second sign-off, and commit only after that
   explicit approval — never before.

Compression modes do NOT skip the commit step.
```

---

### 1.7 [NEW under "Pipeline Self-Maintenance"] Catalog-coverage audit and remediation — Lesson 10

```markdown
### Catalog-coverage audit and remediation before close-out

Every pipeline run ends with a diff of the Phase 0 catalog against the
final helper export list. The audit is **remediation-first, not
log-first** — every catalogued LLK missing from the final API gets
**added** (op struct via the appropriate CRTP base, ~4 lines) before
close-out. Drop is an exception, not a co-equal option: a `dropped:
<reason>` row in the gap map is only acceptable when the LLK is
genuinely out of scope (e.g. macro-injection-only, hardware-not-yet-
landed) and a justification is recorded. If the audit produces drops,
the pipeline does NOT close out — it loops back into implementation
to add the missing op structs first.

Silent omission has shipped helpers without coverage the catalog
promised. Logging the gap without fixing it is the same failure with
extra paperwork.
```

---

## File 2 — `eltwise_helper_lessons.md`

### 2.1 [EDIT §1.1] Extend CRTP base list — Lesson 7

Replace the three-base block with four:

```cpp
template <typename Derived, Dst Slot>                                          struct UnaryOp;
template <typename Derived, Dst In0, Dst In1, Dst Out>                         struct BinaryOp;
template <typename Derived, Dst In0, Dst In1, Dst In2, Dst Out>                struct TernaryOp;
template <typename Derived, Dst In0, Dst In1, Dst In2, Dst In3, Dst Out>       struct QuaternaryOp;
```

Add note:

> `QuaternaryOp` covers 4-input SFPU ops (e.g. fused mask + scale + bias).
> DEST `static_assert` checks `Out + 1 < DEST_AUTO_LIMIT` and that all
> input slots are distinct.

---

### 2.2 [EDIT §1.2] Strengthen ctor / template split — Lesson 5

Append after the existing examples block:

> A field that is "usually compile-time but occasionally runtime" gets
> two op structs (or a template specialization), not a runtime field
> with a constexpr default. Mixing axes in one struct breaks chain-trait
> deduction.

---

### 2.3 [NEW §1.10] Fill / rand tile tag — Lesson 8a

Insert after §1.7:

```markdown
### 1.10 Fill / rand tiles tagged distinctly from CB-load ops

`FillTileTag {}` and `RandTileTag {}` are 0-byte markers separate from
`CopyTileTag`. They denote chain elements that *write* a DEST slot from
constants / RNG state — they do NOT consume from a CB.

Why a separate tag: chain traits like `chain_loads_share_cb_v`,
`chain_has_duplicate_upfront_cbs_v`, and `is_copy_tile_op_v` would
mis-classify fill/rand under `CopyTileTag` (no CB to wait/pop, no
indexing, no fan-out semantics). Conflating the two has shipped
double-wait bugs in past chain combinator logic.

`FillScalar`, `RandTile` derive from the new tag. They expose
`is_upfront = false`, `clashes_with_fpu = false` by default.
```

---

### 2.4 [NEW §3.8] One eltwise helper with specialised entry points — Lesson 13

Insert at end of §3 (Composition):

```markdown
### 3.8 One eltwise helper, specialised convenience entry points — not parallel helpers

Eltwise has **one** helper surface (`eltwise_chain` + the chain element
type system). Frequently-used patterns ship as **specialised
convenience entry points** that wrap the chain — not as parallel
standalone helpers (`binary_op`, `unary_bcast_op`, `dest_reuse_op`,
etc. living as peer top-level APIs).

Pattern:

```cpp
// Underlying primitive (always available, fully expressive):
eltwise_chain(CopyTile<cb_a>{}, CopyTile<cb_b>{}, Add{}, ...);

// Convenience wrapper for the common case — thin, expands to the chain:
binary_add(cb_a, cb_b, cb_out);              // == eltwise_chain(...)
unary_bcast_mul(cb_in, BroadcastDim::ROW);   // == eltwise_chain(...)
dest_reuse_mul(cb_in, dst_slot);             // == eltwise_chain(...)
```

Why:

- Shared trait machinery (FPU clash, hoist safety, fan-out, CB
  lifecycle, reconfig deduction) lives once on the chain. Parallel
  helpers each grow their own copy and drift.
- Fast call sites stay one-liners (the convenience wrapper is the same
  ergonomics as a standalone helper would be).
- The "drop down to chain when convenience doesn't fit" path is
  trivial — convenience and chain are the same machinery.
- Helper-design rules (Gate 1 / Gate 2, validation suite, gap map)
  apply to one surface, not N.

Convenience entry points:

- Are picked by frequency-of-use, not "every binary op gets one." A
  one-line wrapper for a pattern used in three kernels is noise; one
  for the pattern used in fifty kernels saves real call-site code.
- Are pure inline forwarders to `eltwise_chain` — no logic of their
  own, no parallel policy enums, no separate validation kernels (the
  chain validation already covers them).
- Live in the same `eltwise_chain.{hpp,inl}` (or a sibling
  `eltwise_convenience.hpp` aggregator), not in standalone files
  named after the pattern.

Anti-pattern: shipping `unary_bcast_op` as a peer of `binary_op` as a
peer of `eltwise_chain`. Three surfaces, three trait surfaces, three
test surfaces. One helper, specialised entry points.
```

---

### 2.5 [NEW §2.8] Unary broadcast as eltwise chain element — Lesson 6

Insert after §2.7:

```markdown
### 2.8 Unary broadcast is a chain element, not a separate helper

Unary bcast (one-tile-into-many, scalar-into-tile, row/col bcast on a
single input) is a **chain element type** that participates in
`eltwise_chain` the same way FPU `binary_op` does. It is NOT a
separate `unary_bcast_op` helper, and NOT a flag on `CopyTile`.

Why a chain element and not a separate helper: the chain combinator
already owns DEST allocation, init/exec ordering, FPU-clash
reinitialization, and CB lifecycle. Lifting unary bcast into a
sibling helper duplicates all of that and forces callers to choose
between mixing helpers (loses chain composition) or replicating bcast
inside `eltwise_chain` (the actual ask).

Why a chain element and not a `CopyTile` flag: bcast changes the
unpack MOP and the per-tile DEST write pattern. Folding it into the
existing CopyTile element pollutes the trait surface
(`is_copy_tile_op_v`, FPU-clash deduction, hoist-safety analysis) for
the common non-bcast case. New element type, separate trait surface.

Surface: a chain element parameterized on `BroadcastDim::{NONE, ROW,
COL, SCALAR}` (the same enum already in use by `binary_op`). Caller
passes `BroadcastDim` explicitly — per §9, no inference. The element
plugs into `eltwise_chain(...)` like any other chain participant; chain
traits (`chain_has_any_copy_tile_v`, fan-out, reuse) extend to cover
it via the existing trait machinery rather than special-case branches.
```

---

## File 3 — `llk_helpers_pipeline.md`

Pipeline doc gets concrete phase steps; the *rule* lives in hq. Each step references the hq section.

### 3.1 [NEW Phase 1 sub-step] `compute_kernel_hw_startup` necessity investigation — Lesson 4

Add a new focus row to Phase 1's investigation table (or append to "Existing Helpers"):

```markdown
| **Init Surface** | For each op group, determine whether the chain
  prologue genuinely needs `compute_kernel_hw_startup(...)`, or whether
  a minimal subset of `*_init` / `*_init_short` / `hw_configure_*` /
  `reconfig_data_format_*` reproduces the same hw state. Output: per-op
  recommendation + reasoning, to be written into the helper's `.hpp`
  doc-comment when the helper is implemented. |
```

The recommendation is documented in the helper header (not in a
separate substitution table) so it lives next to the code that consumes
it.

---

### 3.2 [EDIT Phase 5 (Implementation)] Slim context before migration — Lesson 8b

Add a step at the start of Phase 5:

```markdown
0. Slim migration context — archive prior `agent_logs/`, strip closed
   gap-map entries, drop superseded proposal artifacts before auditing
   the first kernel. Rule at
   [llk_helpers_hq.md → Step 0 — Slim context before migration](llk_helpers_hq.md#step-0--slim-context-before-migration).
```

---

### 3.3 [EDIT Phase 4] Migration sub-loop step — Lesson 9

Insert after "### 4c: Helper Integration":

```markdown
### Migration sub-loop: one test per kernel, full suite at end

When Phase 5 (Implementation) migrates multiple kernels in sequence,
each per-kernel migration runs ONE representative pytest from the
Pytest Manifest. Run

```
scripts/run_safe_pytest.sh --run-all tests/ttnn/unit_tests/kernel_lib/*.py
```

once at the end across all kernels touched in the cycle.

Rule lives at [llk_helpers_hq.md → Step 4](llk_helpers_hq.md#step-4--verify-on-device).
```

---

### 3.4 [EDIT Phase 6] Catalog-coverage diff in report — Lesson 10

Add to Phase 6 report contents (after item 5):

```markdown
6. **Catalog-vs-coverage diff**: every op enumerated in
   `{category}_catalog.md` either appears in the final helper export
   list OR has an explicit `dropped: <reason>` row in the gap map. No
   silent omissions. Rule lives at
   [llk_helpers_hq.md → Catalog-coverage audit before close-out](llk_helpers_hq.md#catalog-coverage-audit-before-close-out).
```

Also add to Phase 4 description one line:

> Before sub-stage 4d, validation agent re-reads the catalog and
> verifies every catalogued LLK is reachable through the helper API.

---

## Apply Order

1. **hq.md** (1.1–1.7) — cross-helper rules land first; everything else references them.
2. **eltwise_helper_lessons.md** (2.1–2.5) — eltwise-specific surfaces.
3. **pipeline.md** (3.1–3.4) — concrete steps point at the now-existing hq rules.

Outstanding question: keep the prior memory-file entries
(`project_eltwise_helper_design.md`, `feedback_pipeline_migration.md`,
`feedback_safe_pytest_no_timeout.md`) or remove them once these doc
edits land?
