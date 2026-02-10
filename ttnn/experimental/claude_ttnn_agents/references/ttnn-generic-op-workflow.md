# TTNN Generic Operation Creation Workflow

This reference contains the mandatory routing and workflow for creating new TTNN operations using the Python-based generic_op infrastructure. It is intended for the orchestrator agent only.

## Overview

This workflow uses the **generic_op** infrastructure which bypasses C++ TTNN scaffolding entirely, producing Python orchestration code, ProgramDescriptor APIs, and stub kernel files. This is ideal for:
- Rapid prototyping without CMake/nanobind overhead
- Custom operations that don't require C++ TTNN registration
- Operations using `ttnn.generic_op()` and ProgramDescriptor

## Canonical Operation Path (SOURCE OF TRUTH)

All generic_op operations MUST be created at:

```
ttnn/ttnn/operations/{operation_name}/
```

This is the **single source of truth** for operation location. All agents in this pipeline read this path from here.

This places operations within the `ttnn` package, enabling direct imports:
```python
from ttnn.operations.<op_name> import <op_name>
```

**Directory structure** (flat, with tests colocated):
```
ttnn/ttnn/operations/{operation_name}/
├── __init__.py                             # Re-export main function
├── {operation_name}.py                     # Entry point with output allocation
├── {operation_name}_program_descriptor.py  # CB config, work distribution, kernel setup
├── test_{operation_name}.py                # PyTorch reference comparison (colocated)
├── kernels/
│   ├── {operation_name}_reader.cpp         # Data movement: DRAM → L1
│   ├── {operation_name}_compute.cpp        # FPU/SFPU operations
│   └── {operation_name}_writer.cpp         # Data movement: L1 → DRAM
├── agent_logs/                             # Execution logs (if logging enabled)
│   ├── {agent_name}_breadcrumbs.jsonl
│   └── {agent_name}_execution_log.md
├── {operation_name}_spec.md                # Functional spec (from planner)
└── kernel_design.md                        # Kernel design doc (from designer)
```

**Example**: For an operation named `row_centralize`:
```
ttnn/ttnn/operations/row_centralize/
├── __init__.py
├── row_centralize.py
├── row_centralize_program_descriptor.py
├── test_row_centralize.py
├── kernels/
│   ├── row_centralize_reader.cpp
│   ├── row_centralize_compute.cpp
│   └── row_centralize_writer.cpp
├── row_centralize_spec.md
└── kernel_design.md
```

**Running tests**:
```bash
pytest ttnn/ttnn/operations/row_centralize/test_row_centralize.py -v
```

## Pipeline Structure

```
analyzer → planner → ┬─ generic_op_builder ─┬─→ kernel_writer
                     └─ kernel_designer ────┘

                     (parallel)              (waits for both)
```

**Key difference from standard workflow**: The `generic_op_builder` and `kernel_designer` agents run **in parallel** after the planner completes. The `kernel_writer` agent only starts after **both** have finished.

---

## Mandatory Routing

When user requests a new TTNN operation via generic_op, STOP and answer these questions:

### Step 1: Are reference operations specified?
- YES with paths to reference_operation_analysis.md → Skip to Phase 1 (Analyzer)
- YES but vague ("like softmax") → Search for that operation's program_factory.cpp
- NO → Continue to discovery

### Step 2: Discovery Checklist (if references not specified)

**⚠️ CRITICAL: COMPUTE REQUIRES TILES**
All compute operations (FPU/SFPU) require tilized data. Even if BOTH input AND output are row-major:
- Row-major input → MUST tilize before compute
- Compute operates on 32×32 tiles ONLY
- Row-major output → MUST untilize after compute

Pattern: `RM input → read sticks → tilize → compute (tiles) → untilize → write sticks → RM output`

□ **First: Determine if operation has compute**:
  - ANY math operation (reduction, eltwise, matrix ops) → REQUIRES tilized data
  - Row-major input + compute → MUST include tilize reference (Hybrid Mode)
  - Compute + row-major output → MUST include untilize reference (Hybrid Mode)
  - Row-major input + compute + row-major output → Hybrid Mode with 3 references:
    1. tilize (input_stage)
    2. compute operation (compute_core)
    3. untilize (output_stage)

□ Parse for format keywords:
  - "row-major input" + ANY compute → need tilize reference
  - ANY compute + "row-major output" → need untilize reference
  - "sharded" → need sharded-input reference (layernorm, etc.)

□ Select appropriate variant:
  - Match memory layout: interleaved → *_interleaved_*, sharded → *_sharded_*
  - Prefer simpler variant (single_core) for templates

□ Query DeepWiki for unknowns:
  - "Which TTNN operations perform [X]?"
  - "Which operations convert ROW_MAJOR to TILE_LAYOUT?"

### Step 3: Mode Determination
- Single reference → Derivative mode
- Multiple references with different roles → Hybrid mode

### Step 4: Reference Confirmation (USER CHECKPOINT)

Before running analyzers, present discovered references:

"I identified these references:
| Role | Operation | Path | Reason |
|------|-----------|------|--------|
| input_stage | tilize | .../tilize_multi_core_interleaved_program_factory.cpp | row-major + tilize keywords |
| output_stage | untilize | .../untilize_multi_core_program_factory.cpp | untilize + row-major keywords |

Planning Mode: Hybrid

Proceed with analysis, or suggest different references?"

- User confirms → proceed to Phase 1
- User suggests alternatives → update references and re-confirm

### Step 5: Enable Logging (If Requested)

If the user requests breadcrumbs or logging, create the signal file **before launching any agents**:

```bash
mkdir -p .claude && echo '{"operation_path": "{operation_path}"}' > .claude/active_logging.json
```

A `SubagentStart` hook automatically injects breadcrumb instructions into every agent's context. No need to mention logging in agent prompts. See `.claude/references/logging-mechanism.md` for details.

After the pipeline completes, clean up:
```bash
rm -f .claude/active_logging.json
```

### Step 6: Execute Workflow

1. **Phase 1**: Run `ttnn-operation-analyzer` on EACH confirmed reference
2. **Phase 2**: Run `ttnn-operation-planner` with all analyzer outputs
3. **USER REVIEW** (MANDATORY): Present the generated `{new_op}_spec.md` to the user
   - User approves → proceed to Phase 3
   - User requests changes → refine spec, re-present for approval
   - Do NOT proceed without explicit user approval
4. **Phase 3** (PARALLEL EXECUTION):
   - Launch `ttnn-generic-op-builder` with the spec
   - Launch `ttnn-kernel-designer` with the spec
   - **Both agents run concurrently** - do NOT wait for one to start the other
5. **Phase 3 Sync Point**: Wait for BOTH agents to complete before proceeding
   - `ttnn-generic-op-builder` produces: Python orchestration, ProgramDescriptor, stub kernels
   - `ttnn-kernel-designer` produces: `kernel_design.md` (helper vs raw call decisions)
6. **USER REVIEW**: Present outputs from both agents:
   - Generic op structure (CBs, runtime args, kernel paths)
   - Kernel design summary (helper vs raw decisions, CB flow)
   - User approves → proceed to Phase 4
   - User requests changes → refine, re-present
7. **Phase 4**: Run `ttnn-kernel-writer` with:
   - Kernel Design Document from `ttnn-kernel-designer`
   - Stub kernels from `ttnn-generic-op-builder`
   - Functional spec from planner

---

## Workflow Diagram

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  User provides: New operation requirements                                   │
│  (may or may not specify reference operations)                              │
└──────────────────────────────────┬──────────────────────────────────────────┘
                                   │
          ┌────────────────────────┴────────────────────────┐
          │                                                 │
          ▼ (refs not specified)                            ▼ (refs specified)
┌─────────────────────────┐                                 │
│ Phase 0: Discovery      │                                 │
│ (Orchestrator)          │                                 │
│ - Query DeepWiki        │                                 │
│ - Search codebase       │                                 │
│ - Select candidates     │                                 │
└───────────┬─────────────┘                                 │
            │                                               │
            ▼                                               │
┌─────────────────────────────────────────────────────────────────────────────┐
│ Phase 1: Analyzer(s)                                                         │
│ (Opus)                                                                       │
│                                                                              │
│  Run ttnn-operation-analyzer on each reference operation                     │
│  Output: {operation}_analysis.md for each reference                          │
└──────────────────────────────────────────────────┬──────────────────────────┘
                                                   │
                                                   ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ Phase 2: Planner                                                             │
│ (Opus)                                                                       │
│                                                                              │
│  Run ttnn-operation-planner with all analyzer outputs                        │
│  Output: {new_op}_spec.md                                                    │
└──────────────────────────────────────────────────┬──────────────────────────┘
                                                   │
                                         ▼ [USER REVIEW SPEC]
                                                   │
┌─────────────────────────────────────────────────────────────────────────────┐
│ Phase 3: Parallel Execution                                                  │
│                                                                              │
│  ┌───────────────────────────┐     ┌───────────────────────────┐            │
│  │ ttnn-generic-op-builder   │     │ ttnn-kernel-designer      │            │
│  │ (Opus)                    │     │ (Opus)                    │            │
│  │                           │     │                           │            │
│  │ Input: spec.md            │     │ Input: spec.md            │            │
│  │                           │     │        helper headers     │            │
│  │ Output:                   │     │                           │            │
│  │ - Python orchestration    │     │ Output:                   │            │
│  │ - ProgramDescriptor       │     │ - kernel_design.md        │            │
│  │ - Stub kernel files       │     │   (USE HELPER / NO HELPER │            │
│  │ - CB configuration        │     │    for each phase)        │            │
│  └─────────────┬─────────────┘     └─────────────┬─────────────┘            │
│                │                                 │                           │
│                └────────────┬────────────────────┘                           │
│                             │                                                │
│                             ▼ SYNC POINT: Wait for BOTH                      │
└─────────────────────────────┬───────────────────────────────────────────────┘
                              │
                    ▼ [USER REVIEW BOTH OUTPUTS]
                              │
┌─────────────────────────────────────────────────────────────────────────────┐
│ Phase 4: Kernel Writer                                                       │
│ (Opus)                                                                       │
│                                                                              │
│  Input:                                                                      │
│  - kernel_design.md (from designer)                                          │
│  - Stub kernels (from generic_op_builder)                                    │
│  - spec.md (from planner)                                                    │
│                                                                              │
│  Output:                                                                     │
│  - Working reader, compute, writer kernels                                   │
│  - Following design's USE HELPER / NO HELPER guidance                        │
└─────────────────────────────┬───────────────────────────────────────────────┘
                              │
                              ▼ [TEST: E2E passes]
                              │
┌─────────────────────────────────────────────────────────────────────────────┐
│ Done!                                                                        │
└─────────────────────────────────────────────────────────────────────────────┘

     ┌──────────────────────────┐
     │ ttnn-riscv-debugger      │ ◄── Can be invoked at any
     │ (Sonnet, hypothesis-     │     phase when kernel issues
     │  driven, reverts code)   │     arise (hangs, wrong output)
     └──────────────────────────┘
```

---

## Agent Responsibilities

### ttnn-operation-analyzer
**Purpose**: Deep architectural analysis of existing TTNN operations as references.

**Input**: Path to reference operation's program factory

**Output**: `{operation_name}_analysis.md` containing:
- Work unit definition
- Data flow pattern
- Circular buffer configuration
- Memory access patterns
- Core distribution strategy

### ttnn-operation-planner
**Purpose**: Design the new operation, producing a functional specification.

**Input**: Analyzer outputs + requirements

**Output**: `{new_operation}_spec.md` containing:
- Mathematical definition and API specification
- Input/output tensor requirements
- CB layout decisions
- Core distribution strategy
- Test criteria

### ttnn-generic-op-builder
**Purpose**: Create Python-based operation infrastructure using generic_op.

**Input**: Functional spec (`{operation}_spec.md`)

**Output**:
- Python orchestration code using `ttnn.generic_op()`
- ProgramDescriptor configuration
- Stub kernel files (reader, compute, writer)
- CB configuration in Python

**Key Feature**: Bypasses C++ scaffolding for rapid iteration.

### ttnn-kernel-designer
**Purpose**: Design kernel implementation strategy by mapping computation phases to helpers or raw calls.

**Input**:
- Functional spec (`{operation}_spec.md`)
- Kernel helper library headers (`ttnn/cpp/ttnn/kernel_lib/*.hpp`)

**Output**: `kernel_design.md` containing:
- Per-kernel breakdown (reader, compute, writer)
- For each phase: "USE HELPER: `function()`" or "NO HELPER: use raw calls"
- CB synchronization summary

### ttnn-kernel-writer
**Purpose**: Implement kernels following the Kernel Design Document.

**Input**:
- Kernel Design Document (`kernel_design.md`) - MANDATORY
- Stub kernels from generic_op_builder
- Functional spec

**Output**: Working kernels that:
- Call helpers for phases marked "USE HELPER"
- Use raw calls ONLY for phases marked "NO HELPER"
- **MUST NOT** add raw CB operations around helper calls

---

## Parallel Execution Requirements

### Phase 3 Orchestration

When launching Phase 3, the orchestrator MUST:

1. **Launch both agents in a single message** with multiple Task tool calls:
   ```
   [Task: ttnn-generic-op-builder with spec path]
   [Task: ttnn-kernel-designer with spec path]
   ```

2. **Do NOT** launch sequentially - both agents are independent and can run concurrently.

3. **Wait for both** to complete before presenting results to user or proceeding to Phase 4.

### Dependency Graph

```
analyzer_1 ─┐
analyzer_2 ─┼──► planner ──► spec.md ──┬──► generic_op_builder ──► stubs + Python
analyzer_n ─┘                          │
                                       └──► kernel_designer ──────► kernel_design.md
                                                                          │
                    ┌─────────────────────────────────────────────────────┘
                    │
                    ▼
              kernel_writer (requires BOTH outputs)
```

---

## User Checkpoints

| Checkpoint | Location | What to Review |
|------------|----------|----------------|
| Reference Selection | After Step 4 | Confirm reference operations are appropriate |
| Spec Approval | After Phase 2 | Review API, tensor requirements, CB layout |
| Parallel Output Review | After Phase 3 | Review generic_op structure + kernel design |
| Final Validation | After Phase 4 | E2E test passes with correct values |

---

## Comparison with Standard Workflow

| Aspect | Standard Workflow | Generic Op Workflow |
|--------|-------------------|---------------------|
| C++ Scaffolding | Yes (scaffolder + factory-builder) | No (generic_op_builder) |
| Build Required | Yes (CMake/nanobind) | No (Python-only) |
| Stages 1-6 | Sequential (scaffolder → factory-builder) | Parallel (generic_op_builder \|\| designer) |
| Kernel Design | After factory-builder | Parallel with generic_op_builder |
| Best For | Production operations | Prototyping, custom ops |

---

## Troubleshooting

### Phase 3 Sync Issues

If one agent completes much faster than the other:
- Do NOT start kernel_writer with partial inputs
- Wait for both agents to finish
- If one agent fails, debug that agent before proceeding

### Kernel Writer Dependency Issues

If kernel_writer cannot find required inputs:
- Verify `kernel_design.md` exists from designer
- Verify stub kernels exist from generic_op_builder
- Check paths match expected locations

### Runtime Errors After Kernel Writing

Invoke `ttnn-riscv-debugger` with:
- Symptom description
- Test command
- Paths to kernels and generic_op code

---

## Additional Resources

- `.claude/subagent_breakdown.md` - Detailed workflow breakdown
- `.claude/references/ttnn-operation-workflow.md` - Standard C++ workflow
- https://docs.tenstorrent.com/tt-metal/latest/ttnn/ttnn/adding_new_ttnn_operation.html - Official docs
