# TTNN Operation Creation Workflow

This reference contains the mandatory routing and workflow for creating new TTNN operations. It is intended for the orchestrator agent only.

## Mandatory Routing

When user requests a new TTNN operation, STOP and answer these questions:

### Step 1: Are reference operations specified?
- YES with paths to reference_operation_analysis.md → Skip to Phase 1 (Analyzer)
- YES but vague ("like softmax") → Search for that operation's program_factory.cpp
- NO → Continue to discovery

### Step 2: Discovery Checklist (if references not specified)

□ Parse for format keywords:
  - "row-major input" + "tilize" → need tilize reference
  - "untilize" + "row-major output" → need untilize reference
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

### Step 5: Execute Workflow
1. Phase 1: Run `ttnn-operation-analyzer` on EACH confirmed reference
2. Phase 2: Run `ttnn-operation-planner` with all analyzer outputs
3. **USER REVIEW** (MANDATORY): Present the generated `{new_op}_spec.md` to the user
   - User approves → proceed to Phases 3-6
   - User requests changes → refine spec, re-present for approval
   - Do NOT proceed without explicit user approval
4. Phases 3-6: Run `ttnn-operation-scaffolder` then `ttnn-factory-builder`
5. Phase 7a: Run `ttnn-kernel-designer` to produce Kernel Design Document
   - Maps computation phases to kernel helper functions (priority) or raw calls
   - Creates `{operation_dir}/kernel_design.md`
   - **USER REVIEW**: Present kernel design summary (helper vs raw decisions, CB flow)
   - User approves → proceed to Phase 7b
   - User requests changes → refine design, re-present
   - Do NOT proceed to kernel writing without approval
6. Phase 7b: Run `ttnn-kernel-writer` with the design document
   - Implements kernels following the design's USE HELPER / NO HELPER guidance
   - MUST NOT add raw CB operations for phases where design says USE HELPER

### Step 6: Post-Agent Log Review (If Logging Enabled)

Agent logging is **OPTIONAL**. To enable logging for a pipeline run, the orchestrator MUST create the signal file **before launching any agents**:

```bash
mkdir -p .claude && echo '{"operation_path": "{operation_path}"}' > .claude/active_logging.json
```

Example:
```bash
mkdir -p .claude && echo '{"operation_path": "ttnn/ttnn/operations/layer_norm_rm"}' > .claude/active_logging.json
```

A `SubagentStart` hook (`inject-logging-context.sh`) automatically detects this file and injects breadcrumb instructions into every agent's context. No prompt text needed — agents receive logging instructions via the hook infrastructure.

After the pipeline completes, clean up the signal file:
```bash
rm -f .claude/active_logging.json
```

See `.claude/references/logging-mechanism.md` for full documentation.

**If logging was enabled**, after each agent completes:

1. **Read the execution log**:
   ```
   Read: {operation_path}/agent_logs/{agent_name}_execution_log.md
   ```

2. **Summarize for the user**:
   - Final status (SUCCESS/PARTIAL/FAILED)
   - Number of recovery attempts needed
   - Key issues encountered

3. **Extract and act on upstream feedback** (Section 1 of log):
   - If agent reports issues with its input (spec, design doc, etc.), note these for future runs
   - Consider whether upstream agent instructions need improvement

4. **Collect instruction recommendations** (Section 7 of log):
   - Track patterns across multiple operations
   - If same recommendation appears 3+ times, propose updating the agent instructions

5. **Check for unresolved issues** (Section 3 of log):
   - If agent couldn't resolve an issue, you may need to invoke `ttnn-riscv-debugger`

**If logging was NOT enabled**: Skip this step.

**Log location** (when enabled): `{operation_path}/agent_logs/`

See `.claude/references/logging/` for per-agent logging instructions.

## Additional Resources

- `.claude/subagent_breakdown.md` - Detailed workflow breakdown
- https://docs.tenstorrent.com/tt-metal/latest/ttnn/ttnn/adding_new_ttnn_operation.html - Official docs
