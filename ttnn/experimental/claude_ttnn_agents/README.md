# Claude TTNN Agents

This package contains Claude Code agents for creating new TTNN operations.

## Quick Start

1. Run the activation script:
   ```bash
   ./ttnn/experimental/claude_ttnn_agents/activate_agents.sh
   ```

2. Start Claude Code in the repository root.

3. The agents will be available via the Task tool.

## Agents

| Agent | Purpose |
|-------|---------|
| ttnn-operation-analyzer | Deep architectural analysis of reference operations |
| ttnn-operation-planner | Design new operation spec (derivative or hybrid mode) |
| ttnn-operation-scaffolder | Build Stages 1-3 (API, validation, TTNN registration) |
| ttnn-factory-builder | Build Stages 4-6 (device op, program factory, stub kernels) |
| ttnn-kernel-designer | Design kernel implementation strategy (helper vs raw calls) |
| ttnn-kernel-writer | Implement kernels following the design document |
| ttnn-riscv-debugger | Debug kernel issues (hangs, CB deadlocks, wrong output) |
| ttnn-pipeline-analyzer | Analyze pipeline execution and blocking for performance optimization |

## Workflow

This is a **highly experimental** system for generating TTNN operations using AI agents. Feedback and contributions are welcome!

### Overview

Creating a new TTNN operation from existing reference(s) involves eight phases:

```
Reference Op(s) -> Analyze -> Plan -> Scaffold -> Build Factory -> Design Kernels -> Write Kernels -> Test
                  (Phase 1)  (Phase 2) (Phase 3)   (Phase 4)       (Phase 5)        (Phase 6)     (Phase 7)
                                  |                    |                |
                            USER REVIEW          USER REVIEW       USER REVIEW
                                                                       |
                                                                ttnn-riscv-debugger
                                                                  (if issues)
```

### Planning Modes

The system supports two planning modes:

- **Derivative Mode**: Create a new operation as a variant of ONE existing operation (e.g., masked_softmax from softmax)
- **Hybrid Mode**: Combine components from MULTIPLE reference operations (e.g., tilize reader + reduce compute + untilize writer)

### Phase 1: Analyze Reference Operation(s)

Ask Claude to analyze existing operation(s) similar to what you want to build:

```
"Please use the ttnn-operation-analyzer agent to analyze the tilize operation
at ttnn/cpp/ttnn/operations/data_movement/tilize/device/tilize_multi_core_interleaved_program_factory.cpp"
```

**Output**: `{reference_op}_analysis.md` containing:
- Work unit definition and data flow pattern
- Circular buffer configuration
- Kernel implementations and memory access patterns
- Core distribution strategy

### Phase 2: Plan New Operation

Pass the analysis path(s) to the planner along with your requirements:

**Derivative Mode** (single reference):
```
"Please use the ttnn-operation-planner agent with the analysis at
ttnn/cpp/.../softmax_analysis.md to plan a new 'masked_softmax' operation."
```

**Hybrid Mode** (multiple references):
```
"Please use the ttnn-operation-planner agent with:
- tilize_analysis.md (role: input_stage)
- reduce_analysis.md (role: compute_core)
- untilize_analysis.md (role: output_stage)
to plan a new operation that tilizes input, reduces along width, and untilizes output."
```

**Output**: `{new_op}_spec.md` containing:
- API signature and parameters
- Input/output tensor requirements
- Circular buffer sizing and work distribution
- Kernel modifications needed

**IMPORTANT**: Review this spec carefully! Iterate with Claude if needed. The spec is the contract for all downstream agents.

### Phase 3: Scaffold the Operation

Once the spec is finalized, build the scaffolding (Stages 1-3):

```
"Please use the ttnn-operation-scaffolder agent with the spec at
ttnn/cpp/ttnn/operations/reduction/my_op/my_op_spec.md"
```

**Output**:
- Python API and C++ pybind
- Device operation with validation
- TTNN operation registration
- Stage 1-3 tests in `test_dev/`

### Phase 4: Build Program Factory

Build the program factory and stub kernels (Stages 4-6):

```
"Please use the ttnn-factory-builder agent with the spec at
ttnn/cpp/ttnn/operations/reduction/my_op/my_op_spec.md"
```

**Output**:
- Complete device operation
- Program factory with circular buffers
- **Stub kernels** (compile and pass data through, but produce garbage output)
- Stage 4-6 tests

**Note**: Stub kernels verify infrastructure works (no hangs). Correctness comes in Phase 6.

### Phase 5: Design Kernel Implementation

Design how kernels should be implemented:

```
"Please use the ttnn-kernel-designer agent with the spec at
ttnn/cpp/ttnn/operations/reduction/my_op/my_op_spec.md"
```

**Output**: `kernel_design.md` containing:
- Per-kernel phase breakdown
- For each phase: "USE HELPER" (with exact function) or "NO HELPER" (with raw call guidance)
- CB synchronization summary
- Helper encapsulation acknowledgment

**IMPORTANT**: Review the kernel design before proceeding. The design determines which helper functions vs raw calls are used.

### Phase 6: Write Kernels

Implement kernels following the design document:

```
"Please use the ttnn-kernel-writer agent with the design at
ttnn/cpp/ttnn/operations/reduction/my_op/kernel_design.md"
```

**Output**:
- Working reader, compute, and writer kernels
- Stage 7 correctness tests
- Verification against PyTorch reference

### Phase 7: Test & Debug

Run tests to verify correctness:

```bash
pytest ttnn/cpp/ttnn/operations/reduction/my_op/test_dev/ -v
```

If tests hang or produce incorrect output, use the debugger:

```
"Please use the ttnn-riscv-debugger agent to debug this issue:
Symptom: test hangs after 10 seconds
Test: pytest .../test_dev/test_stage7_kernel_correctness.py
Operation analysis: .../my_op_analysis.md"
```

### Phase 8: Performance Analysis (Optional)

For performance optimization of existing operations:

```
"Please use the ttnn-pipeline-analyzer agent to analyze the pipeline behavior of
ttnn/cpp/ttnn/operations/reduction/my_op/device/my_op_program_factory.cpp"
```

**Output**: `{operation_name}_pipeline_analysis.md` containing:
- CB configuration with capacity/block ratios
- Blocking point analysis and execution timeline
- Performance metrics and optimization recommendations

## Example: One-Shot Automated Operation Creation

With these agents, Claude can create a simple TTNN operation in a fully automated mode. Here's an example prompt that successfully generated a working reduce operation:

```
Build me a simple TTNN reduction operation which takes row-major interleaved input,
computes average along width and outputs a row-major interleaved output. See tilize
and untilize operations and kernel helper library for details. Output's last dim -
width - should be 1 (even though tile-aligned physical width of 32 is produced,
tensor's logical width dimension should be 1).

This time I want you to run in a FULLY AUTOMATED mode. Introduce reasonable assumptions
and DO NOT ask for confirmation or clarifications. In the end, summarize the decisions
you made (eg. if you had any open questions).
```

This prompt resulted in:
- Analysis of tilize and untilize reference operations
- A functional spec combining tilize (input) + reduce (compute) + untilize (output)
- Complete scaffolding with validation
- Program factory with circular buffers
- Working kernels using the kernel helper library
- Passing tests verified against PyTorch

**Note**: The "FULLY AUTOMATED" directive can be omitted if you prefer to review and provide feedback at each stage. The default workflow includes user review checkpoints at:
- Phase 2 (spec review)
- Phase 5 (kernel design review)

## Tips

- **Keep DeepWiki handy**: Ask about kernel APIs, hardware concepts, or patterns
- **Read the analysis**: Understanding the reference operation is crucial
- **Iterate on the spec**: Don't rush past Phase 2 - a good spec saves debugging time
- **Start simple**: Test the simplest case first in Stage 7
- **Use Debug builds**: Always build with `./build_metal.sh -b Debug`
- **Use the debugger agent**: For kernel hangs or CB issues, invoke `ttnn-riscv-debugger` with the symptom and operation analysis
- **Use the pipeline analyzer**: For performance issues, use `ttnn-pipeline-analyzer` to understand blocking and overlap behavior

See `subagent_breakdown.md` for additional technical details.

## Reference Documents

The agents use several reference documents in `.claude/references/`:

| Document | Purpose |
|----------|---------|
| ttnn-cb-memory-fundamentals.md | CB page concepts, sync rules, tilize/untilize patterns |
| factory-builder-stages.md | TDD cycles, test templates, implementation code |
| ttnn-pipeline-analysis-methodology.md | Pipeline analysis methodology and case studies |
| ttnn-riscv-debugger-reference.md | Watcher/DPRINT usage for debugging |
| cb-debugging-strategy.md | CB deadlock debugging playbook |
| semaphore-debugging-strategy.md | Semaphore coordination debugging |
| table-templates.md | Standardized tables for analysis documents |

## DeepWiki Integration

The activation script configures the DeepWiki MCP server for accessing tt-metal documentation. It modifies:

- `~/.claude.json` - Adds MCP server and project context
- `.claude/settings.local.json` - Adds permission for the tool

DeepWiki provides context about:
- Hardware architecture (Tensix cores, NoC, etc.)
- Kernel development patterns
- API documentation
- Existing operation implementations

### Manual Setup (Alternative)
```bash
claude mcp add -s user -t http deepwiki https://mcp.deepwiki.com/mcp
```
Then add `"deepWikiContext": "tenstorrent/tt-metal"` to your project in `~/.claude.json`.

## Kernel Helper Library

The agents leverage the kernel helper library at `ttnn/cpp/ttnn/kernel_lib/`:

- **tilize_helpers.hpp**: Unified `tilize()` - handles simple/activation/fast/DT patterns
- **untilize_helpers.hpp**: Unified `untilize()` - auto-dispatches based on width/datatype
- **reduce_helpers.hpp**: Unified `reduce()` - handles ROW/COL/SCALAR with streaming or preloaded input
- **dest_helpers.hpp**: Auto-detects DEST register limits based on sync/accum mode

These helpers encapsulate CB management, DST register handling, and init/uninit sequences, allowing kernel writers to focus on the computation logic rather than low-level synchronization.

## Implementation Status

| Agent | Status | Model |
|-------|--------|-------|
| ttnn-operation-analyzer | Implemented | Opus |
| ttnn-operation-planner | Implemented | Opus |
| ttnn-operation-scaffolder | Implemented | Sonnet |
| ttnn-factory-builder | Implemented | Sonnet |
| ttnn-kernel-designer | Implemented | Opus |
| ttnn-kernel-writer | Implemented | Opus |
| ttnn-riscv-debugger | Implemented | Opus |
| ttnn-pipeline-analyzer | Implemented | Opus |

All 8 agents are fully implemented and operational.
