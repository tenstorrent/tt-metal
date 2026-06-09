# Kernel Helper Pipeline

End-to-end workflow for creating or updating `compute_kernel_lib` helpers. Orchestrates investigation, design, validation, and implementation through a linear phase sequence.

## Overview

```
Phase 0: Catalog ─> Phase 1: Investigation ─> Phase 2: Verification
                    (parallel per group)

─> Phase 3: Proposal ─> [STOP — Gate 1: explicit user sign-off on API]
                         │
                         v
   Phase 3.5: Test Plan ─> [STOP — Gate 2: explicit user sign-off on test plan]
                         │
                         v
   Phase 4: Validation ─> Phase 5: Implementation
                         (TDD sub-stages 4a..4d)

─> Phase 6: Report
```

Gates 1 and 2 are BLOCKING REQUIREMENTS. See
[llk_helpers_hq.md → Approval Gates](llk_helpers_hq.md#approval-gates) for
what counts as approval (and what does not — scope answers don't).
Compression / terse modes don't override the gates.

---

## Review Cadence

Gate 1 (API) and Gate 2 (test plan) are the **only** mid-pipeline blocking reviews — one before any code, one before any test runs. After Gate 2 clears, the pipeline runs Phases 4 → 5 → 6 to completion without further review prompts. The single end-of-cycle review is the Phase 6 report.

Do NOT ask the user to approve continuing between:

- Validation sub-stages (4a → 4b → 4c → 4d)
- Individual test results inside a sub-stage
- Per-kernel migrations during Phase 5
- Phase 4 → Phase 5, Phase 5 → Phase 6

Failures inside Phase 4 follow the Feedback Loops table — fix and re-run, no user prompt. Per-kernel migration failures follow the same rule: fix the kernel or skip with a structured `blocker:` row (only if heavily blocked skip), then continue.

The only exception: if a test result forces a design change that re-opens Gate 1 or Gate 2 scope (e.g. a dtype combo not in the approved test plan turns out unsupported, or the API needs a new policy enum value), re-post the relevant artifact and re-enter the corresponding gate. Anything narrower stays inside the automatic feedback loop.

---

## Prior Work Detection

Before starting, check if outputs from previous runs exist:

1. `agent_logs/{category_slug}/catalog_*.md` → skip Phase 0
2. `{category}_investigation.md` → skip Phase 1
3. `{category}_helper_proposal.md` → skip Phase 3 (still requires Gate 1
   sign-off in conversation history before resuming downstream phases)
4. `{category}_test_plan.md` → skip Phase 3.5 (still requires Gate 2 sign-off)
5. Existing `.hpp`/`.inl` → start at Phase 4 (validation only)

If prior outputs exist, resume from the earliest phase with missing outputs. Never ask the human to choose a path — the pipeline decides based on what exists.

If a `--patterns <tsv>` was supplied on resume, the TSV header is recorded in
each artifact (`PATTERNS_HEADER:` line). Mismatch → discard catalog +
investigation outputs and re-run from Phase 0. Pattern drift poisons every
downstream phase.

---

## Patterns File (optional)

If the orchestrator received `--patterns <path>`, the file is the **source
of truth** for whatever data it carries. Replace mode, not augment. The
schema is **not fixed** — different runs may pass different column sets,
different formats, or even free-form summaries. Agents must read the
top-of-file at runtime to learn the schema, then map the fields it
actually contains onto the relevant phase outputs.

**Consumers (schema-agnostic, best-effort)**:

| Phase | What it pulls (if present in the file) |
|-------|----------------------------------------|
| 0 Catalog | Call-site locations for the Locator Results table. LLK signature enumeration grep still runs — the file is not assumed to carry signatures. |
| 1 Investigation (USAGE) | Call-site list, chaining / sync / pairing / loop / parameter data — whatever fields exist. Whatever is NOT present, fall back to grep for that slice only. |
| 1 Investigation (Encapsulation) | Loop / cross-iteration / parameter-independence data, when present. |
| 3 Proposal | Consumed indirectly via investigation output. |

Each consuming agent must record the file's header / first lines verbatim
as `PATTERNS_HEADER:` in its output artifact, so resume runs can detect
drift between the artifact and a re-supplied patterns file.

If an agent cannot make sense of the file (unrecognized format, missing
the field it needs for its phase), it MUST stop and emit
`STAGE_INCOMPLETE: PATTERNS_FILE not interpretable for <phase>` rather
than silently fall back. The user decides whether to drop the flag or
regenerate the file.

Pass the absolute path through to agents as the `{{PATTERNS_FILE}}`
placeholder. When the placeholder resolves to an empty string, agents
fall back to grep-driven discovery as before.

---

## Logging

All agents log breadcrumbs to `agent_logs/{category_slug}/`. See `tt_metal/third_party/tt-agents/scripts/logging/` for format. Breadcrumbs are always enabled.

---

## Phase 0: Catalog

**Goal**: Enumerate all ops in the category, group them, locate source files.

**Agent**: `llk_catalog_agent.md` (subagent_type: Explore)

**What it does**:
1. Bidirectional grep: bottom-up (LLK prefix functions) + top-down (compute API headers)
2. Cross-reference gaps between directions
3. Assign ops to functional groups
4. Locate all source files per op (wrapper header, LLK file, codegen entry, program factory, custom kernels)

**Output**: `{category}_catalog.md` containing:
- Full op list with gap analysis
- Group-to-ops assignment table
- Locator results table (op -> file paths)
- `PATTERNS_HEADER:` line + `Pattern Drift` table (only when `{{PATTERNS_FILE}}` is set)

**Skip if**: Op list, groups, and file locations are already known.

**Patterns hook**: when `{{PATTERNS_FILE}}` is provided, the catalog agent
inspects the file's schema at runtime. If the file carries call-site
locations, those rows replace the wide call-site grep for the Locator
Results table. LLK signature enumeration (Phase 1A/1B) still runs — the
file is not assumed to carry signatures.

---

## Phase 1: Investigation

**Goal**: Deep analysis of each op group's device behavior, host parameter flow, usage patterns, **encapsulation requirements**, and **parameter independence**.

**Agent**: `llk_investigation_agent.md` (subagent_type: Explore)

Launch ONE investigation agent per functional group, all in parallel. Each agent receives a **role-based focus directive** scoping its analysis:

| Focus | What to produce |
|-------|----------------|
| **Device** | Wrapper signatures (init + exec), init state compatibility, DEST batching limits, FP32 accumulation requirements, disruptive inits list |
| **Host** | Code generation table, program factory layout, parameter encoding reference (user API value -> host transform -> kernel receives) |
| **Usage** | All kernel call sites, init/exec pairing rules, init mutual exclusion, chaining patterns, parameter usage matrix per LLK |
| **Encapsulation** | Compile-time feature matrix, cross-iteration state analysis, side-effect operations, parameter independence analysis |
| **CB Management** | Every `cb_reserve_back`/`cb_push_back`/`cb_wait_front`/`cb_pop_front` in the production kernel with purpose annotation. Flag reserves that don't pair with an obvious push (shared memory protection), reserves used for flow control, and any CB overlap assumptions (e.g., out_cb and interm_cb sharing L1). Document which CBs share memory and what ordering constraints that creates. |
| **Existing Helpers** | Grep all `.inl` files in `ttnn/cpp/ttnn/kernel_lib/` for `ASSERT`, `static_assert`, CB validation, DEST limit checks, and policy enum patterns. Produce a table of mandatory validation patterns that the new helper must include. Also note any reusable infrastructure (e.g., `get_cb_num_pages` from `cb_helpers.hpp`, `DEST_AUTO_LIMIT` from `dest_helpers.hpp`). |
| **Init Surface** | For each op group, determine whether the chain prologue genuinely needs `compute_kernel_hw_startup(...)`, or whether a minimal subset of `*_init` / `*_init_short` / `hw_configure_*` / `reconfig_data_format_*` reproduces the same hw state. Output: per-op recommendation + reasoning, to be written into the helper's `.hpp` doc-comment when the helper is implemented. |

All seven focus areas are covered by a single agent per group. The agent's prompt specifies which focus areas to prioritize based on the category.

**Patterns hook**: when `{{PATTERNS_FILE}}` is provided, the agent inspects
the file's schema at runtime and uses whatever fields are present as the
authoritative input for the matching investigation tables (call sites,
chaining, sync/pairing, loop / cross-iteration state, parameter usage).
For slices the file does NOT cover, fall back to grep on that slice only.
The agent records `PATTERNS_HEADER:` in its output.

**Output**: `{category}_investigation.md` (orchestrator consolidates per-group outputs)

---

## Phase 2: Verification

**Goal**: Check investigation claims against actual code.

**Agent**: `llk_verification_agent.md` (subagent_type: Explore)

**Input**: `{category}_investigation.md`

**What it does**: Takes each claim from investigation, checks against actual code. Returns CONFIRMED / INCORRECT / UNVERIFIABLE per claim.

**Output**: `{category}_verification.md`

INCORRECT verdicts are high-value — they directly change the helper design.

---

## Phase 3: Proposal

**Goal**: Design the helper API, op structs, and migration plan.

**Agent**: `llk_helper_proposal_agent.md` (subagent_type: general-purpose)

**Input**: Investigation + verification outputs, locator results

**What it does**: Proposes helper API (signatures, enums, dispatch), designs CRTP-based op structs, creates before/after examples, assigns migration tiers. Uses upstream data directly:
- DEST batching limits -> chunking logic
- Parameter encoding reference -> op struct field design
- Init mutual exclusion -> validates grouping decisions
- Chaining patterns -> multi-op helper design
- Compile-time feature matrix -> template bool parameters
- Cross-iteration state analysis -> loop ownership decisions
- Parameter independence analysis -> minimal API surface
- Side-effect operations -> correctness requirements to preserve
- CB compile-time analysis -> template vs runtime param decisions

**Output**: `{category}_helper_proposal.md`

### STOP — Gate 1: API proposal sign-off

> BLOCKING REQUIREMENT. Do NOT proceed to Phase 4 (or write any `.inl`,
> `.cpp`, kernel source, or test file) until the user explicitly approves
> the proposal.

After writing the proposal:

1. Output one line: `Proposal at <path>. Awaiting sign-off.`
2. End the turn. Do not continue.
3. Wait for the user reply.

What counts as approval is defined in
[llk_helpers_hq.md → Approval Gates](llk_helpers_hq.md#what-counts-as-explicit-approval).
Notably: scope answers ("yes, full surface migration") do NOT approve the
API. Clarifying-question answers do NOT approve the API. The user must
explicitly accept the proposal sections (or list deltas) for the gate to
clear.

If the user lists deltas, revise the proposal artifact and re-post for a
second sign-off. Don't smuggle deltas into Phase 4.

---

## Phase 3.5: Test Plan

**Goal**: Propose the validation test list as a separate artifact before any
test kernel or pytest is written.

**Input**: Approved proposal.

**Output**: `{category}_test_plan.md` containing per-test:
- Kernel name + source path it will live at
- What it covers (which proposal section)
- `num_tiles` parameterization
- Dtype matrix (and **explicit list of dtypes / fp32_dest_acc / mixed-dtype
  combos that will be SKIPPED, with rationale**)
- PCC threshold
- Skip-on-arch list (e.g. Blackhole)

### STOP — Gate 2: Test plan sign-off

> BLOCKING REQUIREMENT. Do NOT write any test kernel, pytest, or run any
> validation until the user explicitly approves the test plan.

After writing the test plan:

1. Output one line: `Test plan at <path>. Awaiting sign-off.`
2. End the turn. Wait for the user reply.

The list of test changes that need approval is enumerated in
[llk_helpers_hq.md → Gate 2](llk_helpers_hq.md#gate-2--test-plan-requires-explicit-user-approval):
adding, removing, skipping, retitling, changing tolerance, changing
parameterization, changing dtype matrix, marking XFAIL↔PASS.

If the test plan needs revision mid-validation (e.g. a dtype combo turns
out to be unsupported), STOP and re-post the revised plan. Don't disable
tests inline.

---

## Phase 4: Validation

**Goal**: Prove the proposal is correct on device, then prove the helper implementation works.

**Agent**: `llk_validation_agent.md` (subagent_type: general-purpose)

Runs 4 sub-stages sequentially. Each gates the next.

> Before sub-stage 4d, validation agent re-reads the Phase 0 catalog and verifies every catalogued LLK is reachable through the helper API. Missing ops loop back into implementation per [llk_helpers_hq.md → Catalog-coverage audit and remediation before close-out](llk_helpers_hq.md#catalog-coverage-audit-and-remediation-before-close-out).

### 4a: Raw LLK Validation

Generate test kernels using raw LLK calls (not the helper) that exercise the EXACT proposed LLK sequences. Run on device against golden references.

- Pass -> proceed to 4b
- Fail/hang -> BLOCKER, fix proposal, re-run

### 4b: Parameter Coverage

Use the parameter usage matrix from Phase 1 to test each LLK across its full parameter space: data formats, template arguments, runtime arguments.

Three mandatory dimensions:
1. **Data format**: Float16_b, BFloat16, Float32, mixed I/O
2. **Template args**: Every value of Approx, Legacy, RoundingMode, etc.
3. **Runtime args**: Typical + edge + negative values

- Observed combo fails -> BLOCKER, fix proposal
- Unobserved combo fails -> record as UNSUPPORTED

### 4c: Helper Integration

Write test kernels using the ACTUAL helper API (.hpp). Test:
1. Default path (default dtype, default args)
2. Dtype variation (at least 2 formats)
3. Template arg variation (non-default values)
4. Runtime arg variation (at least 2 values)
5. Policy variation (at least 2 input policies)
6. Chain composition (combine new op with another in a chain)
7. Feature flag matrix — for every template bool from the compile-time feature matrix,
   test both true and false. Combinatorial testing is required when flags interact
   (e.g., PACKER_L1_ACC × PACK_RELU × FUSE_BIAS). Use the investigation's
   compile-time feature matrix to enumerate the combinations.

- Helper fails but raw passed -> bug in .hpp/.inl, fix and re-run 4c only

### Migration sub-loop: one test per kernel, full suite at end

When Phase 5 (Implementation) migrates multiple kernels in sequence, each per-kernel migration runs ONE representative pytest from the Pytest Manifest. Run

```
scripts/run_safe_pytest.sh --run-all tests/ttnn/unit_tests/kernel_lib/*.py
```

once at the end across all kernels touched in the cycle.

Rule lives at [llk_helpers_hq.md → Step 5](llk_helpers_hq.md#step-5--verify-on-device).

### 4d: Performance

Benchmark helper vs raw LLK. Reuse raw kernels from 4a as baseline.

- Test across tile count range (powers of 2, 8 to 32K)
- Test full complexity spectrum (single op, chains, multi-slot loads)
- Use min of trimmed runs
- Report results table

Thresholds: <2% OK, 2-5% REVIEW, >5% BLOCKER

**Output**: `{category}_validation.md` with per-sub-stage results, parameter support matrix, performance table, generated test files.

---

## Phase 5: Implementation

**Goal**: Write the helper files and migrate easy call sites.

**Agent**: general-purpose

**Input**: Validated proposal + validation outputs

Steps:
0. Slim migration context — archive prior `agent_logs/`, strip closed gap-map entries, drop superseded proposal artifacts before auditing the first kernel. Rule at [llk_helpers_hq.md → Step 0 — Slim context before migration](llk_helpers_hq.md#step-0--slim-context-before-migration).
1. Read validated proposal (signatures, op structs, LLK sequences, parameter support)
2. Create `{name}_helpers.hpp` + `{name}_helpers.inl`
3. Migrate Tier 1 call sites
4. Run Phase 4c/4d tests against final implementation
5. Commit

**Output**: `.hpp`, `.inl`, migrated kernel files

---

## Phase 6: Report

**Goal**: Summarize the pipeline run.

Generate `{category}_report.md` containing:
1. Summary: what was created, overall result
2. Pipeline phases: table with agents, outputs, status
3. Validation results: per-op pass/fail, parameter support matrix, performance table
4. Migration status: which call sites were updated
5. Open items: Tier 2/3 sites, unsupported parameter combos
6. **Catalog-vs-coverage diff**: every op enumerated in `{category}_catalog.md` either appears in the final helper export list OR has an explicit `dropped: <reason>` row in the gap map. No silent omissions. Audit is remediation-first — drops force a loop back into implementation, not a sign-off. Rule lives at [llk_helpers_hq.md → Catalog-coverage audit and remediation before close-out](llk_helpers_hq.md#catalog-coverage-audit-and-remediation-before-close-out).

Commit the report.

---

## Agent Reference

| Agent | File | Phase | Purpose |
|-------|------|-------|---------|
| Catalog | `llk_catalog_agent.md` | 0 | Enumerate ops, group, locate files |
| Investigation | `llk_investigation_agent.md` | 1 | Device + host + usage analysis (per group, parallel) |
| Verification | `llk_verification_agent.md` | 2 | Confirm/deny claims against code |
| Proposal | `llk_helper_proposal_agent.md` | 3 | Design helper API + op structs |
| Validation | `llk_validation_agent.md` | 4 | Raw LLK -> params -> integration -> perf (TDD loop) |
| Implementation | (general-purpose) | 5 | Write .hpp/.inl, migrate sites |

### Dependency Graph

```
catalog ──> investigation (parallel per group)
                │
                v
         verification ──> proposal ──> [STOP: Gate 1 sign-off]
                                            │
                                            v
                                       test_plan ──> [STOP: Gate 2 sign-off]
                                                          │
                                                          v
                                            validation (4a->4b->4c->4d)
                                                          │
                                                          v
                                                   implementation ──> report
```

---

## Feedback Loops

Validation sub-stages can loop back:

- **4a fails** (raw LLK invalid) -> fix proposal, re-enter Phase 3
- **4b fails** (observed param combo) -> fix proposal, re-enter Phase 3
- **4c fails** (helper bug, raw passed) -> fix .hpp/.inl, re-run 4c
- **4d fails** (>5% overhead) -> fix .hpp/.inl, re-run 4c + 4d
- **Phase 1 incomplete** (discovered during Phase 3) -> re-run Phase 1 for missing groups

---

## Discovery Strategy

Discovery MUST search bidirectionally:

1. **Bottom-up**: Grep for LLK prefix functions (`llk_math_eltwise_unary_sfpu_*`, etc.)
2. **Top-down**: List all headers in compute API dir, grep for `*_tile_init`/`*_tile(`
3. **Cross-reference**: Ops found in only one direction are gaps to investigate
