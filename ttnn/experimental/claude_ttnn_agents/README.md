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
| ttnn-operation-analyzer | Analyze reference operation |
| ttnn-operation-planner | Design new operation spec |
| ttnn-operation-scaffolder | Build Stages 1-3 (API, validation, registration) |
| ttnn-factory-builder | Build Stages 4-6 (device op, factory, kernels) |
| ttnn-riscv-debugger | Debug kernel issues (hangs, CB deadlocks, incorrect output) |

## Workflow

This is a **highly experimental** system for generating TTNN operations using AI agents. Feedback and contributions are welcome!

### Overview

Creating a new TTNN operation from an existing reference involves five main stages:

```
Reference Op → Analyze → Plan → Scaffold → Build Factory → Test & Debug
             (Stage 1)  (Stage 2) (Stage 3)  (Stage 4)      (Stage 5)
                                                               ↓
                                                        ttnn-riscv-debugger
```

### Stage 1: Analyze Reference Operation

Ask Claude to analyze an existing operation similar to what you want to build:

```
"Please use the ttnn-operation-analyzer agent to analyze the grid_sample operation
at ttnn/cpp/ttnn/operations/pool/grid_sample/device/grid_sample_program_factory.cpp"
```

**Output**: `{reference_op}_analysis.md` containing:
- Compile-time and runtime arguments
- Circular buffer configuration
- Kernel implementations
- Data flow patterns

### Stage 2: Plan New Operation

Pass the analysis path to the planner along with your requirements:

```
"Please use the ttnn-operation-planner agent with the analysis at
ttnn/cpp/ttnn/operations/pool/grid_sample/device/grid_sample_analysis.md
to plan a new 'image_rotate' operation. The new operation should:
- Take an input tensor and rotation angle in degrees
- Rotate the image around its center
- Use bilinear interpolation for pixel values
See torch.nn.functional.rotate for reference behavior"
```

**Output**: `{new_op}_spec.md` containing:
- API signature
- Parameter differences from reference
- Circular buffer sizing
- Kernel modifications needed

**IMPORTANT**: Review this spec carefully! Iterate with Claude if needed:
- Check if the API matches your requirements
- Verify circular buffer calculations make sense
- Ensure kernel modifications are appropriate

Ask Claude to revise the spec until you're satisfied, then confirm it's ready.

### Stage 3: Scaffold the Operation

Once the spec is finalized, build the scaffolding:

```
"Please use the ttnn-operation-scaffolder agent with the spec at
ttnn/cpp/ttnn/operations/pool/image_rotate/image_rotate_spec.md
to build the scaffolding (Stages 1-3)"
```

**Output**:
- Python API in `ttnn/ttnn/operations/{category}/{op_name}.py`
- C++ operation registration
- Parameter validation
- Device operation structure

This creates the API layer but not the actual device implementation.

### Stage 4: Build Program Factory

Now build the program factory and stub kernels:

```
"Please use the ttnn-factory-builder agent with the spec at
ttnn/cpp/ttnn/operations/pool/image_rotate/image_rotate_spec.md.
The scaffolding from Stage 3 is already complete."
```

**Output**:
- Complete device operation in `{op_name}_device_operation.cpp`
- Program factory with circular buffers in `device/{op_name}_program_factory.cpp`
- Stub kernels (reader, compute, writer) in `device/kernels/`
- CMakeLists.txt updated

The stub kernels will compile and pass data through but won't perform the actual operation yet.

### Stage 5: Test-Driven Debugging

At this stage, you have a compilable operation with stub kernels. The next phase involves:

1. **Gradual complexity testing**: Start simple and increase complexity
   - Single pixel input, simple transformation
   - Verify output coordinates are correct
   - Check if any output exists at all
   - Verify pixel values are correct

2. **Incremental kernel implementation**: Implement actual logic in compute kernels

3. **Debug and iterate**: Use the `ttnn-riscv-debugger` agent for kernel issues

**Example workflow** (from `image_rotate` built from `grid_sample`):
- Test 1: Single pixel, 5-degree rotation → Check output coordinates
- Test 2: 2x2 image, 0-degree rotation → Should match input exactly
- Test 3: Simple pattern, 90-degree rotation → Verify geometry
- Test 4: Real image, arbitrary angle → Check interpolation quality

This stage discovered only 2 bugs through TDD, which were quickly fixed.

**See the complete implementation**: The `image_rotate` operation built using these agents can be found in branch [`dev/dnijemcevic/image_rotate`](https://github.com/tenstorrent/tt-metal/tree/dev/dnijemcevic/image_rotate).

#### Using the Debug Agent

When tests hang or produce incorrect output, invoke the `ttnn-riscv-debugger` agent:

```
"Please use the ttnn-riscv-debugger agent to debug this issue:
Symptom: test_avgpool2d hangs after 30 seconds
Test: pytest tests/ttnn/.../test_avgpool2d.py::test_run_avg_pool2d
Operation analysis: ttnn/cpp/ttnn/operations/pool/generic/device/pool_analysis.md"
```

The debugger agent:
- Uses a structured journal to track observations, hypotheses, and experiments
- Enables watcher automatically and interprets core states
- Forms ONE hypothesis per invocation and tests it with a falsifier experiment
- Proposes verified fixes with diffs

**Journal-based debugging**: The agent is stateless—all history is in a JSON journal. The orchestrator maintains the journal and invokes the coprocessor repeatedly until a fix is found or debugging budget is exhausted.

**Note**: Future automation planned:
- **Kernel writer agents**: Specialized agents for implementing reader, compute, and writer kernels
- **Operation-specific debug agents**: Tailored debugging strategies for different operation types

### Tips

- **Keep DeepWiki handy**: Ask about kernel APIs, hardware concepts, or patterns
- **Read the analysis**: Understanding the reference operation is crucial
- **Iterate on the spec**: Don't rush past Stage 2 - a good spec saves debugging time
- **Start simple**: In Stage 5, test the simplest case first
- **Use Debug builds**: Always build with `./build_metal.sh -b Debug`
- **Use the debugger agent**: For kernel hangs or CB issues, invoke `ttnn-riscv-debugger` with the symptom and operation analysis

See `subagent_breakdown.md` for additional technical details.

## DeepWiki Integration

The activation script configures the DeepWiki MCP server for accessing
tt-metal documentation. It modifies:

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
