# Kernel Helper Pipeline

End-to-end workflow for creating or updating `compute_kernel_lib` helpers. Orchestrates investigation, design, validation, and implementation through four phases with explicit feedback loops.

## Overview

```
Phase 0: Understand ──> Phase 1: Design [checkpoint] ──> Phase 2: Validate ──> Phase 3: Implement
                               ^                                  ^                      |
                               └──────────── L1: validate fail ──┘                      |
                               ^                                  ^                      |
                               └──────── L3: scope gap ──────────┘                      |
                                                                  ^                     |
                                                                  └──── L2: regression ─┘
```

Two modes, selected automatically based on whether `.hpp`/`.inl` exist:

| Mode | Trigger | Phase 0 behavior | Phase 3 behavior |
|------|---------|-----------------|-----------------|
| **New helper** | No `.hpp`/`.inl` found | Full catalog + investigation | Create files |
| **Update** | `.hpp`/`.inl` exist | Read existing files, scope the change | Edit files, diff-verify scope |

---

## Prior Work Detection

Before starting, the orchestrator checks state in this order:

1. **Mode selection** (first — determines everything else):
   - Existing `.hpp`/`.inl` for this category → **Update mode**
   - Otherwise → **New helper mode**

2. **Phase skipping** (within the selected mode):
   - Update mode: `{category}_delta_scope.md` exists → skip Phase 0
   - New mode: `{category}_investigation.md` exists → skip Phase 0
   - Either mode: `{category}_helper_proposal.md` exists and no L1 trigger pending → skip Phase 1, enter Phase 2

Never ask the human to choose a path — the pipeline decides based on what exists.

**L1/L3 re-entry overrides Phase skipping**: if a validation failure (L1) or scope gap (L3) is pending, always run Phase 1 with the failure/gap context, regardless of whether a proposal already exists.

---

## Logging

All agents log to a per-category subdirectory `agent_logs/{category_slug}/`. Conventions:

| File | Purpose |
|------|---------|
| `agent_logs/{category_slug}/{phase}_breadcrumbs.jsonl` | Per-phase breadcrumb stream (JSONL) |
| `agent_logs/{category_slug}/{phase}_execution_log.md` | Per-phase execution log (Markdown summary) |
| `agent_logs/{category_slug}/{group_slug}_investigation.md` | Per-group investigation output (Phase 0 step 2) |

Top-level consolidated outputs live at the repo root (or workspace cwd), not under `agent_logs/`:
`{category}_investigation.md`, `{category}_delta_scope.md`, `{category}_helper_proposal.md`, `{category}_validation.md`, `{category}_report.md`.

See `tt_metal/third_party/tt-agents/scripts/logging/` for the JSONL event schema.

---

## Phase 0: Understand

**Goal**: Build a complete picture of what exists and what needs to change.

**Agents** (see per-step breakdown below):
- Catalog step: `llk_catalog_agent.md` (subagent_type: Explore) — new mode only
- Investigation step: `llk_investigation_agent.md` (subagent_type: Explore) — one per group, parallel — new mode only
- Update scope step: orchestrator handles directly (no subagent) — update mode only

### New helper mode

**Step 1 — Catalog** (one `llk_catalog_agent.md` invocation):
1. Bidirectional grep: bottom-up (LLK prefix functions) + top-down (compute API headers)
2. Cross-reference gaps between directions
3. Assign ops to functional groups
4. Locate all source files per op (wrapper header, LLK file, codegen entry, program factory, custom kernels)

Output: group→ops table + locator results (consumed by Step 2).

**Step 2 — Investigation** (one `llk_investigation_agent.md` per group, in parallel):
5. Deep analysis per group covering all focus areas (device behavior, host parameter flow, usage patterns, encapsulation, CB management, existing helper patterns) — mark each claim **CONFIRMED** (file:line cited) or **UNCERTAIN** (what would confirm) inline
6. Identify parameter independence, init mutual exclusion, disruptive inits, compile-time feature matrix, cross-iteration state

The orchestrator consolidates per-group outputs into `{category}_investigation.md`.

**Focus areas** (all covered in a single agent per group):

| Focus | What to produce |
|-------|----------------|
| **Device** | Wrapper signatures (init + exec), init state compatibility, DEST batching limits, FP32 accumulation requirements, disruptive inits list |
| **Host** | Code generation table, program factory layout, parameter encoding reference (user API value -> host transform -> kernel receives) |
| **Usage** | All kernel call sites, init/exec pairing rules, init mutual exclusion, chaining patterns, parameter usage matrix per LLK |
| **Encapsulation** | Compile-time feature matrix, cross-iteration state analysis, side-effect operations, parameter independence analysis |
| **CB Management** | Every `cb_reserve_back`/`cb_push_back`/`cb_wait_front`/`cb_pop_front` in production kernels with purpose annotation. Flag reserves without obvious paired push, reserves used for flow control, CB overlap assumptions. Document which CBs share memory and ordering constraints. |
| **Existing Helpers** | Grep all `.inl` files in `ttnn/cpp/ttnn/kernel_lib/` for `ASSERT`, `static_assert`, CB validation, DEST limit checks, policy enum patterns. Produce mandatory validation pattern table. Note reusable infrastructure. |

**Output**: `{category}_investigation.md` — full op list, group-to-ops table, file locator results, per-group analysis with inline CONFIRMED/UNCERTAIN flags

### Update mode

**Triggered when**: existing `.hpp`/`.inl` found for this category (or user specifies a change to an existing helper).

**Handled by the orchestrator directly** — no subagent is invoked because this work is narrow and file-local. The catalog and investigation agents are optimized for unfamiliar territory; for update mode the target files are already known.

1. Read existing `.hpp` and `.inl` fully
2. Identify the requested change (new op, API change, bug fix, perf improvement)
3. Trace which LLK sequences, parameter paths, and call sites are affected by the change
4. Identify any files shared with other helpers or ops outside the change scope
5. Flag any dependencies that could cause scope bleed (shared inits, shared CBs, shared LLK functions)

**Output**: `{category}_delta_scope.md` — change description, affected LLK sequences, affected call sites, dependency flags

---

## Phase 1: Design

**Goal**: Produce a concrete, implementable design. Human reviews before proceeding.

**Agent**: `llk_helper_proposal_agent.md` (subagent_type: general-purpose)

**Orchestrator-supplied placeholders**:
- `{{INVESTIGATION_FILE}}` → `{category}_investigation.md` (new mode) or `{category}_delta_scope.md` (update mode)
- `{{MODE}}` → `new` or `update`
- `{{L1_FAILURE_CONTEXT}}` → contents of the `L1_TRIGGER_START`…`L1_TRIGGER_END` block from the last Phase 2 run, or empty string if no pending L1
- `{{LOCATOR_RESULTS}}` → locator table from Phase 0 catalog step (new mode) or empty (update mode — delta scope already lists affected files)

**Input**: `{category}_investigation.md` (new mode) or `{category}_delta_scope.md` (update mode)

### New helper mode

Full proposal: helper API (signatures, enums, dispatch), CRTP-based op structs, before/after examples, migration tiers. Uses upstream data directly:

- DEST batching limits → chunking logic
- Parameter encoding reference → op struct field design
- Init mutual exclusion → validates grouping decisions
- Chaining patterns → multi-op helper design
- Compile-time feature matrix → template bool parameters
- Cross-iteration state analysis → loop ownership decisions
- Parameter independence analysis → minimal API surface
- Side-effect operations → correctness requirements to preserve
- CB compile-time analysis → template vs runtime param decisions

### Update mode

Delta proposal: what changes, which LLK sequences are affected, what the new/modified API looks like, which call sites need updating. Scope is strictly limited to the change — do not redesign unrelated parts of the helper.

### L1 re-entry (validation failure)

Receive failure context from Phase 2: which op, which sub-stage, which LLK sequence or param combo failed. Amend the proposal to fix the specific failure. Do not redesign unrelated ops. Output amended proposal with a changelog section noting what changed and why.

**Checkpoint**: human reviews proposal (or delta proposal) before Phase 2 starts. Check LLK sequence validation table, before/after examples (new mode) or delta diff (update mode), tier assignments, parameter independence analysis.

**Output**: `{category}_helper_proposal.md` or amendment to existing proposal

---

## Phase 2: Validate

**Goal**: Prove the design is correct on device before writing final files.

**Agent**: `llk_validation_agent.md` (subagent_type: general-purpose)

Runs 4 sub-stages sequentially. Each gates the next. Internal review/fix loop handles `.hpp`/`.inl` bugs without escalating.

### 2a: Raw LLK Validation

Generate test kernels using raw LLK calls (not the helper) that exercise the EXACT proposed LLK sequences. Run on device against golden references.

- Pass → proceed to 2b
- **Fail/hang → L1 trigger**: pass failure details (op, sequence, error) to Phase 1

### 2b: Parameter Coverage

Use the parameter usage matrix from Phase 0 to test each LLK across its full parameter space.

Three mandatory dimensions:
1. **Data format**: Float16_b, BFloat16, Float32, mixed I/O
2. **Template args**: Every value of Approx, Legacy, RoundingMode, etc.
3. **Runtime args**: Typical + edge + negative values

- Observed combo fails → **L1 trigger**: pass failing combo to Phase 1
- Unobserved combo fails → record as UNSUPPORTED, continue

### 2c: Helper Integration

Write test kernels using the ACTUAL helper API (`.hpp`). Test:
1. Default path (default dtype, default args)
2. Dtype variation (at least 2 formats)
3. Template arg variation (non-default values)
4. Runtime arg variation (at least 2 values)
5. Policy variation (at least 2 input policies)
6. Chain composition (combine new op with another in a chain)
7. Feature flag matrix — for every template bool from the compile-time feature matrix, test both true and false; combinatorial testing required when flags interact

- Helper fails but raw passed → **internal fix**: fix `.hpp`/`.inl`, re-run 2c only. If the fix requires an API change → L1 trigger.

### 2d: Performance

Benchmark helper vs raw LLK. Reuse raw kernels from 2a as baseline.

- Test across tile count range (powers of 2, 8 to 32K)
- Test full complexity spectrum (single op, chains, multi-slot loads)
- Use min of trimmed runs
- Report results table

Thresholds: <2% OK, 2-5% REVIEW, >5% → **internal fix**: optimize `.hpp`/`.inl`, re-run 2c + 2d. If fix requires API change → L1 trigger.

**L1 trigger conditions summary**:
- 2a fails (raw LLK sequence invalid)
- 2b observed combo fails (proposed param combo is unsupported)
- 2c fix requires API change
- 2d fix requires API change

**L1 trigger payload**: sub-stage, op name, LLK sequence or param combo, error details, what specifically needs to change in the proposal.

**Output**: `{category}_validation.md` — per-sub-stage results, parameter support matrix, performance table, generated test files

---

## Phase 3: Implement

**Goal**: Write or update the helper files, verify scope, then confirm the implementation works on device.

**Agent**: orchestrator directly, with `llk_review_fix_agent.md` invoked for the L2 loop (subagent_type: general-purpose)

Why no dedicated Phase 3 agent: writing files is a mechanical translation of the validated proposal. L2 (post-write validation) re-uses the Phase 2 validation agent. Scope-gap detection (L3) is a judgement call the orchestrator makes while editing — not delegable to a subagent that lacks the full context.

**Input**: validated proposal + validation outputs

### New helper mode

1. Read validated proposal (signatures, op structs, LLK sequences, parameter support)
2. Create `{name}_helpers.hpp` + `{name}_helpers.inl`
3. Migrate Tier 1 call sites
4. → L2 (always)

### Update mode

1. Read delta proposal + delta scope
2. Edit existing `{name}_helpers.hpp` and `{name}_helpers.inl`
3. Diff the changes against the original — confirm only intended files and symbols were touched
4. If diff contains unexpected changes (unrelated ops, unrelated files, removed defines) → stop, investigate, revert unintended changes
5. Update affected call sites
6. → L2 (always)

### L2: Post-implementation validation (always runs)

After writing or editing files, re-run Phase 2 sub-stages 2c + 2d against the actual implementation (not the test-only kernels from 2a/2b).

- 2c pass + 2d pass → proceed to report
- 2c fail → fix inline, re-run 2c. If fix is non-trivial → loop L2 from 2c start
- 2d fail → fix inline, re-run 2c + 2d. If fix is non-trivial → loop L2 from 2c start
- Max 3 inline fix attempts; if still failing → L1 trigger with full context

### L3: Scope gap trigger

Trigger when, during implementation (file edits or call site migration), you discover:
- A shared file also controls ops outside the current change scope
- A CB layout or init assumption affects another helper
- A LLK function change alters behavior for ops not in the proposal
- Existing call sites depend on behavior the change would break, and those sites are not in Tier 1

**L3 payload**: newly discovered file paths, op names, or symbols outside the current proposal scope; description of why they're affected.

**L3 action**: suspend implementation (do not commit partial changes), send payload to Phase 1. Phase 1 amends the design to cover the expanded scope. Phase 2 then validates the new/changed parts before implementation resumes. Do not skip Phase 2 for the expanded scope — that is how the gap was missed in the first place.

### Report

After L2 passes, generate `{category}_report.md` inline:
1. Summary: what was created or changed, overall result
2. Pipeline phases: table with agents, outputs, status, loop iterations
3. Validation results: per-op pass/fail, parameter support matrix, performance table
4. Migration status: which call sites were updated
5. Open items: Tier 2/3 sites, unsupported parameter combos

---

## Feedback Loop Reference

| Loop | From | To | Trigger | Payload |
|------|------|----|---------|---------|
| **L1** | Phase 2 (2a/2b fail, or 2c/2d fix needs API change) | Phase 1 | Raw LLK sequence invalid; observed param combo unsupported; API change required | Sub-stage, op, sequence/combo, error details |
| **L2** | Phase 3 (always, post-write) | Phase 2 (2c + 2d only) | Files written or edited | Actual `.hpp`/`.inl` paths |
| **L3** | Phase 3 (scope gap discovered) | Phase 1 → Phase 2 → Phase 3 | Shared file / CB / LLK dependency outside proposal scope | Newly discovered file paths, op names, symbols, why they're affected |

---

## Agent Reference

| Agent | File | Phase | Mode |
|-------|------|-------|------|
| Catalog | `llk_catalog_agent.md` | 0 step 1 | New only |
| Investigation | `llk_investigation_agent.md` | 0 step 2 | New only (one per group, parallel) |
| Update scope | (orchestrator) | 0 | Update only |
| Design | `llk_helper_proposal_agent.md` | 1 | Both; supports L1 re-entry via `{{L1_FAILURE_CONTEXT}}` |
| Validate | `llk_validation_agent.md` | 2 | Both; emits `L1_TRIGGER_START`…`L1_TRIGGER_END` on failure |
| Implement | (orchestrator) + `llk_review_fix_agent.md` for L2 | 3 | Both |

Deprecated (do not invoke): `llk_verification_agent.md` — inline CONFIRMED/UNCERTAIN flags in investigation output replace it. `llk_device_validation_agent.md` is reference material for sub-stage 2a, not a standalone agent.

### Dependency Graph

```
Understand ──> Design [checkpoint] ──> Validate ──> Implement
                    ^                      ^              |
                    └──── L1 (fail) ───────┘              |
                    ^                      ^              |
                    └── L3 (scope gap) ────┘              |
                                           ^              |
                                           └── L2 (always)┘
```

---

## Discovery Strategy

Discovery MUST search bidirectionally:

1. **Bottom-up**: Grep for LLK prefix functions (`llk_math_eltwise_unary_sfpu_*`, etc.)
2. **Top-down**: List all headers in compute API dir, grep for `*_tile_init`/`*_tile(`
3. **Cross-reference**: Ops found in only one direction are gaps to investigate
