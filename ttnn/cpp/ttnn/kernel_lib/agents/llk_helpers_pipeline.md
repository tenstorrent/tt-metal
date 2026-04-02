# Kernel Helper Pipeline

End-to-end workflow for creating or updating `compute_kernel_lib` helpers. Orchestrates investigation, design, validation, and implementation through a linear phase sequence.

## Overview

```
Phase 0: Catalog ─> Phase 1: Investigation ─> Phase 2: Verification
                    (parallel per group)

─> Phase 3: Proposal ─> Phase 4: Validation ─> Phase 5: Implementation
                         (TDD sub-stages)

─> Phase 6: Report
```

---

## Prior Work Detection

Before starting, check if outputs from previous runs exist:

1. `agent_logs/{category_slug}/catalog_*.md` → skip Phase 0
2. `{category}_investigation.md` → skip Phase 1
3. `{category}_helper_proposal.md` → skip Phase 3
4. Existing `.hpp`/`.inl` → start at Phase 4 (validation only)

If prior outputs exist, resume from the earliest phase with missing outputs. Never ask the human to choose a path — the pipeline decides based on what exists.

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

**Skip if**: Op list, groups, and file locations are already known.

---

## Phase 1: Investigation

**Goal**: Deep analysis of each op group's device behavior, host parameter flow, and usage patterns.

**Agent**: `llk_investigation_agent.md` (subagent_type: Explore)

Launch ONE investigation agent per functional group, all in parallel. Each agent receives a **role-based focus directive** scoping its analysis:

| Focus | What to produce |
|-------|----------------|
| **Device** | Wrapper signatures (init + exec), init state compatibility, DEST batching limits, FP32 accumulation requirements, disruptive inits list |
| **Host** | Code generation table, program factory layout, parameter encoding reference (user API value -> host transform -> kernel receives) |
| **Usage** | All kernel call sites, init/exec pairing rules, init mutual exclusion, chaining patterns, parameter usage matrix per LLK |

All three focus areas are covered by a single agent per group. The agent's prompt specifies which focus areas to prioritize based on the category.

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

**Output**: `{category}_helper_proposal.md`

**Checkpoint**: Review proposal before proceeding. Check LLK sequence validation table, before/after examples, tier assignments.

---

## Phase 4: Validation

**Goal**: Prove the proposal is correct on device, then prove the helper implementation works.

**Agent**: `llk_validation_agent.md` (subagent_type: general-purpose)

Runs 4 sub-stages sequentially. Each gates the next.

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

- Helper fails but raw passed -> bug in .hpp/.inl, fix and re-run 4c only

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
         verification ──> proposal ──> validation (4a->4b->4c->4d)
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
