# Kernel Helper Library HQ

Entry point for creating and maintaining `compute_kernel_lib` helpers — unified APIs that hide LLK init/compute/pack complexity in tt-metal compute kernels.

## Quick Start

| What you need | Document |
|---|---|
| **Writing helper code** (file structure, op structs, policies, CRTP bases, perf testing) | [llk_helpers_conventions.md](llk_helpers_conventions.md) |
| **Running the agent pipeline** (catalog, investigate, propose, validate, implement) | [llk_helpers_pipeline.md](llk_helpers_pipeline.md) |

## When to Use What

| Situation | Action |
|---|---|
| Adding ops to an existing helper | Read [conventions](llk_helpers_conventions.md) section 5 (op structs). Add CRTP struct + init/call. |
| Creating a new helper (known ops, known LLK calls) | Read [conventions](llk_helpers_conventions.md), write .hpp/.inl directly, add perf tests. |
| Creating a new helper (unknown territory) | Run the [pipeline](llk_helpers_pipeline.md) — new helper mode, starts at Phase 0. |
| Updating/improving an existing helper | Run the [pipeline](llk_helpers_pipeline.md) — update mode, starts at Phase 0 (reads existing files). |

## Agent Files

| Agent | Pipeline Phase | Purpose |
|-------|---------------|---------|
| `llk_catalog_agent.md` | 0: Understand (new mode, step 1) | Catalog ops via bidirectional grep; produces group→ops + locator |
| `llk_investigation_agent.md` | 0: Understand (new mode, step 2) | Deep analysis per group (parallel); inline CONFIRMED/UNCERTAIN flags |
| (orchestrator) | 0: Understand (update mode) | Read existing `.hpp`/`.inl`, scope the change |
| `llk_helper_proposal_agent.md` | 1: Design | Full proposal (new) or delta proposal (update); handles L1 re-entry |
| `llk_validation_agent.md` | 2: Validate | Raw LLK → params → integration → perf; emits L1 trigger on failure |
| (orchestrator) + `llk_review_fix_agent.md` | 3: Implement | Write/edit files, L2 post-write validation, L3 scope gap detection, report |

**Deprecated**: `llk_verification_agent.md` (inline flags in investigation output replace it). `llk_device_validation_agent.md` is reference material for sub-stage 2a, not an agent.

## Feedback Loops

| Loop | Trigger | Path |
|------|---------|------|
| **L1** | Validation sub-stage 2a/2b fails, or 2c/2d fix needs API change | Phase 2 → Phase 1 (amend proposal) |
| **L2** | Files written or edited (always) | Phase 3 → Phase 2 (re-run 2c + 2d only) |
| **L3** | Scope gap found during implementation | Phase 3 → Phase 1 (amend design) → Phase 2 (validate new scope) → Phase 3 (resume) |

## Helpers Location

```
ttnn/cpp/ttnn/kernel_lib/
  {name}_helpers.hpp      <- declarations, enums, structs, examples
  {name}_helpers.inl      <- implementation
  agents/                 <- this directory (pipeline docs + agent prompts)
```
