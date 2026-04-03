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
| Creating a new helper (unknown territory) | Run the [pipeline](llk_helpers_pipeline.md) from Phase 0. |
| Updating/improving an existing helper | Run [pipeline](llk_helpers_pipeline.md) from Phase 4 (validation only). |

## Agent Files

| Agent | Pipeline Phase | Purpose |
|-------|---------------|---------|
| `llk_catalog_agent.md` | 0 | Enumerate ops, group, locate source files |
| `llk_investigation_agent.md` | 1 | Device + host + usage analysis (per group) |
| `llk_verification_agent.md` | 2 | Confirm/deny investigation claims |
| `llk_helper_proposal_agent.md` | 3 | Design helper API + op structs |
| `llk_validation_agent.md` | 4 | Raw LLK -> params -> integration -> perf |
| `llk_review_fix_agent.md` | 4 | Document review and fix loop (within validation) |
| `llk_device_validation_agent.md` | 4 | Device-side test generation (within validation) |

## Helpers Location

```
ttnn/cpp/ttnn/kernel_lib/
  {name}_helpers.hpp      <- declarations, enums, structs, examples
  {name}_helpers.inl      <- implementation
  agents/                 <- this directory (pipeline docs + agent prompts)
```
