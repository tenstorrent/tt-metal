# TTNN Operation Creation - Subagent Breakdown

This document outlines the strategy for splitting TTNN operation creation into multiple subagents to preserve context and enable efficient development.

## Problem Statement

Creating a new TTNN operation is a complex, multi-stage task that includes:
- Analyzing an existing reference operation
- Planning the new operation's design
- Scaffolding the operation registration
- Building the program factory
- Implementing reader, compute, and writer kernels
- Debugging and integration testing

When done in a single session, context fills up during build-test-debug loops, requiring manual intervention.

## Design Principles

### 1. Explicit Artifact Boundaries
Each agent produces well-defined outputs that serve as inputs to subsequent agents:
```
analyzer_output.md → planner → spec.md + tdd_plan.md
spec.md → scaffolder → registered_operation/
spec.md + scaffolded_op → factory_builder → program_factory.cpp
spec.md + factory → kernel_reader → reader_kernel.cpp
...
```

### 2. Fresh Context for Debugging
The debugger agent is stateless and can be invoked repeatedly. It reads the current state from files, not from conversation history. This prevents context exhaustion.

### 3. Validation Gates Between Phases
Each phase has explicit pass/fail criteria. Don't proceed until the gate passes.

### 4. Model Selection by Task Type
- **Opus**: Planning, design, complex reasoning (Phases 1-2, 4)
- **Sonnet**: Implementation, debugging (Phases 3, 5-6)
- **Haiku**: Mechanical scaffolding if faster/cheaper execution is preferred (Phase 3)

---

## Phase 0: Reference Discovery (Orchestrator)

**Agent**: Main orchestrator (not a subagent)

**Purpose**: Determine which existing operations to analyze as references before invoking the analyzer and planner agents.

This phase is executed by the main orchestrator when the user's request doesn't specify exact reference operations, or when creating a hybrid operation that requires components from multiple sources.

### When to Execute Phase 0

- User describes desired behavior without naming specific references
- User requests a composite/hybrid operation (e.g., "sharded input → reduction → interleaved output")
- User asks for an operation that combines patterns from different categories

### Discovery Workflow

```
┌─────────────────────────────────────────────────────────────────────────────┐
│ PHASE 0: DISCOVERY (Orchestrator)                                           │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  Step 1: IDENTIFY COMPONENT NEEDS                                          │
│  ─────────────────────────────────                                         │
│  Parse user request to identify needed components:                         │
│  • Input stage: What memory layout? Sharded/interleaved? What format?      │
│  • Compute core: What operation? Reduction/eltwise/transform?              │
│  • Output stage: What memory layout? Format conversion needed?             │
│                                                                             │
│  Step 2: QUERY FOR CANDIDATES                                              │
│  ────────────────────────────                                              │
│  Use DeepWiki and/or codebase search to find operations with needed        │
│  patterns. See "Discovery Query Templates" below.                          │
│                                                                             │
│  Step 3: SELECT BEST CANDIDATES                                            │
│  ─────────────────────────────                                             │
│  Choose references based on:                                               │
│  • Code quality and documentation                                          │
│  • Similarity to desired behavior                                          │
│  • Complexity (prefer simpler when adequate)                               │
│                                                                             │
│  Step 4: INVOKE ANALYZER(S)                                                │
│  ─────────────────────────                                                 │
│  Run ttnn-operation-analyzer on each selected reference.                   │
│  For hybrid mode, analyze all references needed for different roles.       │
│                                                                             │
│  Step 5: EVALUATE COMPONENTS                                               │
│  ────────────────────────────                                              │
│  Review each analysis, determine which components to use:                  │
│  ✓ = Use this component from this reference                                │
│  ✗ = Don't need this component from this reference                         │
│                                                                             │
│  Step 6: INVOKE PLANNER                                                    │
│  ─────────────────────────                                                 │
│  Call ttnn-operation-planner with:                                         │
│  • Derivative mode: Single reference path + requirements                   │
│  • Hybrid mode: Multiple references with roles + composition instructions  │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Discovery Query Templates

Use these DeepWiki query patterns to find candidate operations:

| Component Need | DeepWiki Query |
|---------------|----------------|
| Sharded input | "Which TTNN operations support HEIGHT_SHARDED or BLOCK_SHARDED input tensors?" |
| Interleaved input | "Which TTNN operations read from DRAM interleaved tensors?" |
| Reduction compute | "Which TTNN operations use reduce_tile or accumulation in compute kernels?" |
| Eltwise compute | "Which TTNN operations perform element-wise math on tiled data?" |
| Sharded output | "Which TTNN operations write to sharded output tensors?" |
| Interleaved output | "Which TTNN operations write row-major sticks to DRAM interleaved?" |
| Tilize pattern | "Which TTNN operations convert ROW_MAJOR to TILE_LAYOUT?" |
| Untilize pattern | "Which TTNN operations convert TILE_LAYOUT to ROW_MAJOR?" |
| Binary operation | "Which TTNN operations take two input tensors and produce one output?" |
| Broadcast pattern | "Which TTNN operations broadcast a smaller tensor across a larger one?" |

### Codebase Search Patterns

When DeepWiki results are insufficient, search the codebase:

```bash
# Find operations with sharded input handling
grep -r "is_sharded" ttnn/cpp/ttnn/operations/*/device/*_program_factory.cpp

# Find reduction patterns
grep -r "reduce_tile" ttnn/cpp/ttnn/operations/*/device/kernels/compute/

# Find interleaved output patterns
grep -r "INTERLEAVED" ttnn/cpp/ttnn/operations/*/device/*_program_factory.cpp

# Find specific CB patterns
grep -r "CBIndex::c_" ttnn/cpp/ttnn/operations/*/device/*_program_factory.cpp
```

### Example: Hybrid Discovery Workflow

```
User: "Create operation that reads sharded, does reduction, writes interleaved"

Orchestrator:

Step 1 - IDENTIFY NEEDS:
  • Input stage: HEIGHT_SHARDED reader
  • Compute core: Reduction (sum/max/etc)
  • Output stage: INTERLEAVED writer

Step 2 - QUERY CANDIDATES:
  DeepWiki: "Which TTNN operations support HEIGHT_SHARDED input?"
    → layernorm, softmax, matmul, eltwise_binary...

  DeepWiki: "Which TTNN operations perform reduction?"
    → reduce_sum, reduce_max, global_avg_pool, moreh_sum...

  DeepWiki: "Which TTNN operations write INTERLEAVED output?"
    → untilize, concat, most eltwise ops...

Step 3 - SELECT:
  • Sharded input: layernorm (well-documented sharding)
  • Reduction: reduce_sum (cleanest pattern)
  • Interleaved output: untilize (canonical pattern)

Step 4 - ANALYZE:
  [Invoke ttnn-operation-analyzer on layernorm] → layernorm_analysis.md
  [Invoke ttnn-operation-analyzer on reduce_sum] → reduce_analysis.md
  [Invoke ttnn-operation-analyzer on untilize] → untilize_analysis.md

Step 5 - EVALUATE:
  From layernorm_analysis.md:
    ✓ Reader kernel (shard handling)
    ✓ CB_in config (shard-backed)
    ✗ Compute kernel (layernorm-specific)
    ✗ Writer kernel (also sharded)

  From reduce_analysis.md:
    ✗ Reader kernel (expects interleaved)
    ✓ Compute kernel (reduce_tile loop)
    ✓ CB sizing (accumulator pattern)
    ✗ Writer kernel (writes reduced shape)

  From untilize_analysis.md:
    ✗ Reader kernel (not needed)
    ✗ Compute kernel (untilize, not reduction)
    ✓ Writer kernel (interleaved sticks)
    ✓ CB_out config (row-major staging)

Step 6 - INVOKE PLANNER (Hybrid Mode):
  references:
    - path: layernorm_analysis.md
      role: input_stage
      components: [reader_kernel, cb_in_config]

    - path: reduce_analysis.md
      role: compute_core
      components: [compute_kernel, cb_accumulator]

    - path: untilize_analysis.md
      role: output_stage
      components: [writer_kernel, cb_out_config]

  requirements: "Reduction on sharded input producing interleaved output"

  composition_instructions: |
    - Sharded reader feeds tiles to CB_in
    - Compute accumulates tiles, writes to CB_out
    - Interleaved writer writes reduced rows to DRAM
```

### Skipping Phase 0

Phase 0 can be skipped when:
- User provides specific reference operation(s)
- User says "use X as reference"
- Creating a simple derivative (variant of single op)

In these cases, proceed directly to Phase 1 (Analysis).

---

## Phase 1: Analysis

**Agent**: `ttnn-operation-analyzer` (Opus)

**Status**: ✅ Implemented

**Purpose**: Deep architectural analysis of an existing TTNN operation to serve as a reference.

**Input**: Path to existing reference operation's program factory

**Output**: `{operation_name}_analysis.md` containing:
- Work unit definition
- Data flow pattern
- Circular buffer configuration
- Index calculations
- Memory access patterns
- Core distribution strategy
- Compile-time and runtime arguments
- Kernel implementations detail

**Location**: `.claude/agents/ttnn-operation-analyzer.md`

---

## Phase 2: Design & Planning

**Agent**: `ttnn-operation-planner` (Opus)

**Status**: ✅ Implemented

**Purpose**: Design the new operation, producing a functional specification.

### Planning Modes

The planner supports two modes:

#### Derivative Mode (Single Reference)
Design a new operation as a variant of one existing operation.
- **Input**: One reference analysis + requirements
- **Output**: Spec comparing new op to single reference
- **Use case**: Creating masked_softmax from softmax, stack from concat, etc.

#### Hybrid Mode (Multiple References)
Design a new operation by combining components from multiple existing operations.
- **Input**: Multiple reference analyses with roles + composition instructions
- **Output**: Spec showing component sources and interface compatibility
- **Use case**: Combining tilize reader + reduce compute + untilize writer

### Input

**Derivative Mode**:
- Analyzer output (`{reference_operation}_analysis.md`)
- New operation requirements (user-provided description)

**Hybrid Mode**:
- Multiple analyzer outputs with roles:
  - `{ref1}_analysis.md` (role: `input_stage`)
  - `{ref2}_analysis.md` (role: `compute_core`)
  - `{ref3}_analysis.md` (role: `output_stage`)
- New operation requirements
- Composition instructions (how components connect)

**Role Definitions**:
- `input_stage`: Reader kernel, input CBs, compute input phase (e.g., tilize)
- `compute_core`: Main compute logic, intermediate CBs, math operations
- `output_stage`: Compute output phase (e.g., untilize), output CBs, writer kernel

### Output

`{new_operation}_spec.md` - Functional specification including:
- Mathematical definition and API specification
- Input tensor requirements (with error message hints)
- Output tensor specification (shape formula)
- CB layout decisions
- Core distribution strategy
- **Hybrid Mode**: Component sources table, interface compatibility analysis, CB ID resolution
- Test criteria (what to test, not how)
- Implementation phases (which agents handle what)

**Key Principle**: Planner defines WHAT to build. Implementation agents define HOW.

**User Checkpoint**: Review and approve spec before proceeding to implementation.

**Location**: `.claude/agents/ttnn-operation-planner.md`

---

## Phase 3: Incremental Operation Building (Stages 1-6)

The TDD plan produced by the planner defines 6 incremental stages to build up the operation infrastructure before implementing actual kernel logic. Each stage has a dedicated test file in `test_dev/`.

### Stage 1: API Existence
**Agent**: `ttnn-operation-scaffolder` (Sonnet)

**Input**: Reads spec's "API Specification > Parameters" section

**Purpose**: Create minimal Python binding that is callable.

**Output**:
- `{operation_name}_pybind.hpp/cpp` with stub that throws `NotImplementedError`
- Category pybind updated

**Test**: `test_dev/test_stage1_api_exists.py`

### Stage 2: Parameter Validation
**Agent**: `ttnn-operation-scaffolder` (continues)

**Input**: Reads spec's "Input Tensor Requirements" table (Property, Requirement, Error Message Hint)

**Purpose**: Add host-side validation with meaningful error messages.

**Output**:
- `device/{operation_name}_device_operation.hpp/cpp`
- `invoke()` with `TT_FATAL` validations matching spec's requirements

**Test**: `test_dev/test_stage2_validation.py` - one test per row in spec's table

### Stage 3: TTNN Registration
**Agent**: `ttnn-operation-scaffolder` (continues)

**Input**: Reads spec's "Output Tensor Specification" table for shape formula

**Purpose**: Properly register operation with TTNN infrastructure.

**Output**:
- `{operation_name}.hpp` with `register_operation`
- `bind_registered_operation` in pybind
- `compute_output_specs()` implementing shape formula from spec

**Test**: `test_dev/test_stage3_registration.py`

**Key Principle**: Scaffolder knows HOW (official TTNN patterns). Spec defines WHAT (requirements).

### Stage 4: Device Operation
**Agent**: `ttnn-factory-builder` (Opus) ✅

**Input**: Spec already processed by scaffolder; device op structure exists

**Purpose**: Complete device operation validation methods.

**Output**:
- Complete `validate_on_program_cache_miss()` and `validate_on_program_cache_hit()`
- `select_program_factory()` implementation

**Test**: `test_dev/test_stage4_device_op.py`
- Error mentions "program" or "kernel", not validation

### Stage 5: Program Factory Structure
**Agent**: `ttnn-factory-builder` (continues)

**Input**: Reads spec's "Circular Buffer Requirements" and "Work Distribution" sections

**Purpose**: Create program factory with CBs and work distribution.

**Output**:
- `{operation_name}_program_factory.hpp/cpp`
- Core grid setup from spec's "Parallelization Strategy"
- CB allocation from spec's "Circular Buffer Requirements"
- Throws before `CreateKernel` calls

**Test**: `test_dev/test_stage5_program_factory.py`
- Error mentions "kernel" not "circular buffer"

### Stage 6: Kernel Compilation
**Agent**: `ttnn-factory-builder` (continues)

**Input**: Reads spec's "Kernel Data Movement" and "Memory Access Patterns" sections

**Purpose**: Create stub kernels that compile and pass data through.

**Output**:
- Stub kernel files (RISCV_1, RISCV_0, compute)
- `CreateKernel` calls in factory
- Runtime argument setup

Note: Kernel naming (reader/writer) reflects RISC-V core assignment, not function.
"Writer" kernels may read data (split reader pattern, auxiliary inputs).

**Test**: `test_dev/test_stage6_kernel_compilation.py`
- Kernels compile successfully
- Program executes without hanging
- Output has correct shape/dtype

**User Checkpoint**: Review factory and stub kernels before implementing real kernel logic.

**Key Principle**: Factory-builder knows HOW (CB patterns, work split). Spec defines WHAT (sizes, grid).

---

## Phase 4: Kernel Implementation

Split into three separate agents to contain scope and prevent context exhaustion.

**Important**: Kernel naming (reader/writer) reflects RISC-V core assignment convention, not actual function:
- "Reader" runs on RISCV_0 (BRISC), typically uses NOC0
- "Writer" runs on RISCV_1 (NCRISC), typically uses NOC1
- **Both can read AND write data** - see spec's "Kernel Data Movement" table for actual functions

### Phase 4a: RISCV_0 Data Movement Kernel (Reader)

**Agent**: `ttnn-kernel-dataflow` (Sonnet)

**Status**: 🔲 To be implemented

**Purpose**: Implement the RISCV_0 (BRISC) kernel based on spec's "RISCV_0 Access" pattern.
Typically the "reader" - fetches data into L1 CBs via NOC0.

**Input**:
- Operation spec ("Kernel Data Movement" and "RISCV_0 Access" sections)
- Factory code
- Reference operation's corresponding kernel

**Output**: Working RISCV_0 kernel

**Validation Gate**: Data reaches compute CBs correctly (verified via DPRINT or test)

### Phase 4b: Compute Kernel

**Agent**: `ttnn-kernel-compute` (Sonnet)

**Status**: 🔲 To be implemented

**Purpose**: Implement the compute kernel that performs the actual operation.

**Input**:
- Operation spec ("Compute Access" section)
- Factory code
- Working RISCV_0 kernel
- Reference operation's compute kernel

**Output**: Working compute kernel

**Validation Gate**: Compute output CBs have correct values for simple test cases

### Phase 4c: RISCV_1 Data Movement Kernel (Writer)

**Agent**: `ttnn-kernel-dataflow` (Sonnet)

**Status**: 🔲 To be implemented

**Purpose**: Implement the RISCV_1 (NCRISC) kernel based on spec's "RISCV_1 Access" pattern.
Typically the "writer" - sends data from L1 CBs to DRAM via NOC1.
Note: May also READ auxiliary data (masks, indices) in addition to writing output.

**Input**:
- Operation spec ("Kernel Data Movement" and "RISCV_1 Access" sections)
- Factory code
- Working compute kernel
- Reference operation's corresponding kernel

**Output**: Working RISCV_1 kernel

**Validation Gate**: E2E test passes

---

## Phase 5: Debug/Integration

**Agent**: `ttnn-riscv-debugger` (Sonnet)

**Status**: ✅ Implemented

**Purpose**: Systematic debugging of TTNN kernel issues using hypothesis-driven methodology.

**Input**:
- **Bootstrap mode**: Symptoms, repro command, file paths, experiment budget
- **Continuation mode**: Current debug journal, experiment budget

**Output**:
- Debug journal with hypotheses, observations, experiments
- Proposed next steps with cost estimates
- Code changes shown in diffs (all changes reverted before returning)

**Key Features**:
- Hypothesis → Falsifier → Experiment → Update methodology
- Watcher log interpretation for hang diagnosis
- CB deadlock debugging playbook
- Kernel correctness debugging (wrong outputs)
- Strategic DPRINT placement
- Always reverts code changes before returning

**When to Use**:
- Kernel hangs (CB deadlocks, semaphore issues)
- Wrong output values
- Need systematic debugging with hypothesis tracking
- Build/test failures involving kernel behavior
- Runtime errors in kernel execution

**Location**: `.claude/agents/ttnn-riscv-debugger.md`

---

## Alternative: Checkpoint-Based Single Kernel Agent

If splitting kernels feels too granular, use a single agent with explicit checkpoints:

```
ttnn-kernel-builder (single agent)
├── Checkpoint 1: Reader kernel compiles
├── Checkpoint 2: Reader kernel passes unit test
├── Checkpoint 3: Compute kernel compiles
├── Checkpoint 4: Compute kernel passes unit test
├── Checkpoint 5: Writer kernel compiles
├── Checkpoint 6: E2E test passes
```

If context fills, restart from the last checkpoint with a fresh agent that reads current file state.

---

## Workflow Summary

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
│  DERIVATIVE MODE:                    HYBRID MODE:                            │
│  ┌──────────────┐                    ┌──────────────┐  ┌──────────────┐     │
│  │ Analyze      │                    │ Analyze      │  │ Analyze      │ ... │
│  │ single ref   │ ──► analysis.md    │ ref1         │  │ ref2         │     │
│  └──────────────┘                    │ (input_stage)│  │ (compute)    │     │
│                                      └──────┬───────┘  └──────┬───────┘     │
│                                             │                 │              │
│                                             ▼                 ▼              │
│                                      ref1_analysis.md  ref2_analysis.md ...  │
└──────────────────────────────────────────────────┬──────────────────────────┘
                                                   │
                                                   ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ Phase 2: Planner                                                             │
│ (Opus)                                                                       │
│                                                                              │
│  DERIVATIVE MODE:                    HYBRID MODE:                            │
│  - Single reference                  - Multiple references with roles        │
│  - Compare differences               - Component sources table               │
│  - {new_op}_spec.md                  - Interface compatibility               │
│                                      - CB ID resolution                      │
│                                      - {new_op}_spec.md                      │
└──────────────────────────────────────────────────┬──────────────────────────┘
                                                   │
                                         ▼ [USER REVIEW SPEC]
                                                   │
┌─────────────────────────────────────────────────────────────────────────────┐
│ Phase 3: Incremental Building (6 Stages)                                     │
│                                                                              │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐                   │
│  │ Stage 1:     │    │ Stage 2:     │    │ Stage 3:     │                   │
│  │ API Exists   │───►│ Validation   │───►│ Registration │                   │
│  │ (pybind)     │    │ (TT_FATAL)   │    │ (register_op)│                   │
│  └──────────────┘    └──────────────┘    └──────────────┘                   │
│         │                                       │                            │
│         ▼ test_stage1                           ▼                            │
│                                                                              │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐                   │
│  │ Stage 4:     │    │ Stage 5:     │    │ Stage 6:     │                   │
│  │ Device Op    │───►│ Program      │───►│ Stub Kernels │                   │
│  │ (shape comp) │    │ Factory (CBs)│    │ (passthrough)│                   │
│  └──────────────┘    └──────────────┘    └──────────────┘                   │
│                                                  │                           │
│                                    test_stage6 ──┘                           │
└──────────────────────────────────────────────────┬──────────────────────────┘
                                                   │
                                     ▼ [USER REVIEW FACTORY]
                                                   │
┌─────────────────────┐                            │
│ Phase 4a: RISCV_0   │◄───────────────────────────┘
│ (reader/BRISC)      │ ──► dataflow kernel
└──────────┬──────────┘
           │
           ▼ [TEST: data in compute CBs]
           │
┌─────────────────────┐
│ Phase 4b: Compute   │ ──► compute_kernel.cpp
│ (Sonnet)            │
└──────────┬──────────┘
           │
           ▼ [TEST: correct compute output]
           │
┌─────────────────────┐
│ Phase 4c: RISCV_1   │ ──► dataflow kernel
│ (writer/NCRISC)     │     (may read too!)
└──────────┬──────────┘
           │
           ▼ [TEST: E2E passes]
           │
┌─────────────────────┐
│ Done!               │
└─────────────────────┘

     ┌──────────────────────────┐
     │ Phase 5: RISCV Debugger  │ ◄── Can be invoked at any
     │ (Sonnet, hypothesis-     │     phase when kernel issues
     │  driven, reverts code)   │     arise (hangs, wrong output)
     └──────────────────────────┘
```

---

## Implementation Priority

1. **`ttnn-operation-planner`** ✅ - Produces the spec (WHAT to build)
2. **`ttnn-operation-scaffolder`** ✅ - Stages 1-3, knows HOW (official TTNN patterns)
3. **`ttnn-factory-builder`** ✅ - Stages 4-6, knows HOW (CB patterns, work split)
4. **`ttnn-riscv-debugger`** ✅ - The escape hatch for kernel issues (hangs, wrong output)
5. **`ttnn-kernel-dataflow`** 🔲 - RISCV_1 and RISCV_0 kernels (may read AND write)
6. **`ttnn-kernel-compute`** 🔲 - Compute kernel implementation
