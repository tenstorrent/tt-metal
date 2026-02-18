---
name: tdd-kernels
description: Stage-gated TDD pipeline for TTNN kernel implementation. Use after generic-op-builder and kernel-designer have produced stubs and a design document. Invoked by /create-op at Phase 4, or standalone. Args = operation path.
---

# TDD Kernel Pipeline

Stage-gated test-driven development for TTNN kernels. Each stage is tested independently before advancing.

## Prerequisites

Before using this pipeline, the operation directory MUST contain:
1. A functional spec (`*_spec.md`)
2. A kernel design document (`kernel_design.md`)
3. Stub kernel files from the generic-op-builder (reader, compute, writer)
4. A working `__init__.py` that exports the operation function
5. A program descriptor that configures CBs and kernel args

## CLI Reference

All commands go through:
```
python3 .claude/scripts/tdd-pipeline/tdd_orchestrator.py <command> [args]
```

| Command | Purpose |
|---------|---------|
| `init <spec_path> --op-path <path>` | Initialize pipeline, create state file |
| `add-stage '<json>' --op-path <path>` | Register a stage (must be done before any testing) |
| `status --op-path <path>` | Show pipeline state: stages, pass/fail, current stage |
| `test --op-path <path>` | Run current stage's test (uses dev-test.sh with watcher + hang detection) |
| `advance --op-path <path>` | Mark current stage as passed, move to next |
| `parse-failure --op-path <path>` | Parse test failure into structured JSON |
| `rollback --op-path <path>` | Restore kernels to last passing commit (on max retries) |

---

## Stage Determination Protocol

Read `kernel_design.md` and the spec, then apply two complementary heuristics to break the operation into ordered, testable stages.

### Heuristic 1: Kernel Complexity Ordering (H1)

**Purpose**: Decides which kernel to finalize first.

1. Assess each kernel's complexity:
   - Count distinct operations, phases, or data movement patterns
   - Note dependencies: does it generate special tiles? use complex addressing? coordinate with other cores?
   - Rank: simplest → most complex

2. Plan kernel finalization order:
   - **Meta-stage 1**: Bring the simplest kernel(s) to their FINAL implementation. Keep the others at their MINIMUM VIABLE state — just enough to enable end-to-end testing of the simple kernel.
   - **Meta-stage 2**: Bring the next kernel to final. Others remain at current state.
   - **Meta-stage 3**: Build up the most complex kernel incrementally.

3. Adapt per operation:
   - Each meta-stage is NOT necessarily a single TDD stage. A complex kernel may require multiple stages within its meta-stage (guided by H2).
   - If two kernels are similar in complexity, finalize them together in one meta-stage.

### Heuristic 2: Semantic Goal Progression (H2)

**Purpose**: Decides how to break a kernel's development into stages.

1. Identify testable intermediate results — functional milestones that produce a meaningful output verifiable against a PyTorch reference.
2. Group operations that form a logical unit. Operations that are only meaningful together (e.g., mean reduction + subtraction = "centralize") should be in the same stage.
3. Each stage's output must be independently verifiable: the test computes the expected result from the original input tensor, not from the previous stage's output.

### Combining H1 + H2

H1 determines the **order** (which kernel to work on). H2 determines the **granularity** (how many stages for that kernel).

The first stage almost always establishes the data pipeline: the simplest kernels at full implementation, bookend operations (tilize/untilize for RM I/O) if applicable, and the complex kernel at minimum viable. This verifies end-to-end data movement before adding compute complexity.

### Illustrative Patterns

**Pattern A — Single-core, compute-dominant** (most common for generic_op):
The reader and writer are typically simple (read/write sticks, generate a few constant tiles). The compute kernel contains many phases. H1 finalizes reader+writer together in stage 1 with a minimal compute (identity or passthrough). H2 then slices the compute kernel into incremental stages based on semantic milestones. Later stages only modify the compute kernel.

**Pattern B — Multi-core with multicast coordination**:
Some operations have cores that produce partial results and multicast them to peer cores (e.g., reader → compute → reader → compute → writer pipelines, or weight download patterns where one core streams from DRAM and multicasts to others). Here the reader and writer themselves may be complex, involving semaphore-based synchronization and multi-phase data movement. H1 might identify the writer as simplest, the compute as medium, and the reader (with mcast) as most complex. Stages would first finalize the writer, then the compute, then incrementally build up the reader's mcast coordination. Some stages will modify MULTIPLE kernels simultaneously (e.g., adding semaphore handshakes requires changes in both reader and compute).

**Pattern C — Operations with complex output patterns**:
When the writer has complex addressing (e.g., scatter writes, transposed output, sharded output with non-trivial stick extraction), H1 may identify the writer as the complex kernel. Reader and compute are finalized first with a simple output pattern, then the writer is built up incrementally.

**Key insight**: For single-core operations, the compute kernel is almost always the most complex, and reader/writer can usually be finalized in the first stage. For multi-core operations with inter-core communication, the complexity often shifts to the dataflow kernels.

---

## Stage Registration

Register ALL stages before implementing any. Each stage generates a test file.

### Command
```bash
python3 .claude/scripts/tdd-pipeline/tdd_orchestrator.py add-stage '<json>' --op-path <path>
```

### JSON Schema

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `name` | string | Yes | snake_case — becomes test file name (`test_stage_{name}.py`) |
| `description` | string | Yes | Human-readable — what this stage adds. Goes into kernel-writer prompt. |
| `reference_body` | string | Yes | Python expression: receives `input_tensor`, returns expected output |
| `tolerance` | object | Yes | `{"rtol": float, "atol": float}` |
| `shapes` | list | Yes | Shape strings like `"(1, 1, 32, 64)"` |
| `kernel_files` | list | No | Kernel files this stage modifies (informational) |
| `extra_imports` | string | No | Additional import lines |
| `extra_args` | string | No | Appended to op call (e.g., `", gamma, beta"`) |
| `extra_setup` | string | No | Python code for extra tensor setup |
| `extra_ttnn_setup` | string | No | Python code for extra TTNN setup after input creation |
| `output_shape_expr` | string | No | Python expression for output shape if different from input |

### Writing `reference_body`

The body becomes the return statement of `pytorch_reference(input_tensor)`:
```python
# Identity/passthrough:
"return input_tensor.clone()"

# Intermediate computation:
"return input_tensor - input_tensor.mean(dim=-1, keepdim=True)"

# Multi-line (use \n + 4-space indent):
"mean = input_tensor.mean(dim=-1, keepdim=True)\n    centered = input_tensor - mean\n    var = (centered ** 2).mean(dim=-1, keepdim=True)\n    return centered * (var + 1e-5).rsqrt()"
```

### Setting Tolerances

| Stage Type | rtol | atol | Rationale |
|------------|------|------|-----------|
| Identity/passthrough | 0.01 | 0.01 | Data should be bit-exact through tilize/untilize |
| Simple compute (add, sub, mul) | 0.01 | 0.05 | Single-step bf16 operations |
| Reductions (mean, sum, max) | 0.02 | 0.1 | Accumulation error scales with reduction width |
| Multi-step compute (normalize) | 0.05 | 0.2 | Error compounds across chained operations |

---

## Stage Loop Protocol

For each registered stage, in order:

### 1. Check Status
```bash
python3 .claude/scripts/tdd-pipeline/tdd_orchestrator.py status --op-path <path>
```

### 2. Implement — Invoke Kernel Writer

Launch `ttnn-kernel-writer` with a **stage-scoped** prompt.

**Agent reuse rule**: Track the kernel-writer's `total_tokens` from the task notification after each invocation.
- **< 160k tokens**: Resume the same agent (pass `resume: {agentId}`). Faster — agent already has design doc, kernel files, and prior work in context.
- **>= 160k tokens**: Spawn a fresh agent. Context is getting heavy and risks compaction.

This applies across both retries and new stages. The agent only knows what the current prompt tells it — scope is controlled by the prompt, not by spawning fresh.

**Prompt template** (used for both fresh and resumed):

```
Implement TDD stage '{stage_name}' for {op_name}.
Design: {op_path}/kernel_design.md

Stage description: {stage_description}
Kernel files to modify: {kernel_files}
Previous failure: {failure_json or 'None — first attempt'}

IMPORTANT: Only implement what this stage requires. Do not implement future stages.
Previous stages already pass — do not modify their behavior.
```

For Stage 1 (data pipeline): Specify which kernels need full implementation vs. minimum viable, and what the compute passthrough should do (e.g., "tilize then immediately untilize" for RM I/O operations).

### 3. Test
```bash
python3 .claude/scripts/tdd-pipeline/tdd_orchestrator.py test --op-path <path>
```

This runs the stage test with `dev-test.sh` (watcher enabled, hang detection at 5s timeout, device reset on failure).

### 4. Branch on Result

**PASS (exit 0)**:
```bash
python3 .claude/scripts/tdd-pipeline/tdd_orchestrator.py advance --op-path <path>
git add {op_path}/kernels/ && git commit -m "[kw-tdd] {op_name}: stage {stage_name} passed"
```
Continue to next stage.

**FAIL (exit 1 or 2)**:
```bash
python3 .claude/scripts/tdd-pipeline/tdd_orchestrator.py parse-failure --op-path <path>
```
Read the structured JSON. It includes:
- `classification`: failure type (see retry policy below)
- `summary`: one-line description
- `suggested_action`: what to try next
- `cost`: `FREE` or `HARD`
- `remaining_attempts`: hard retries left
- `budget_exhausted`: whether to rollback

**Retry Policy — Classification Escalation**:

| Classification | Cost | Rationale |
|---|---|---|
| `compilation_error` | FREE | Typos, missing includes — trivially fixable |
| `shape_mismatch` | FREE | Wrong CB count, bad allocation — trivially fixable |
| `runtime_error` | HARD (1) | Needs algorithmic understanding |
| `numerical_mismatch` | HARD (1) | Needs parameter/logic debugging |
| `watcher_assert` | HARD (1) | Hardware-level issue |
| `hang_cb_deadlock` | HARD (1) | CB synchronization bug |
| `hang_noc_error` | HARD (1) | DMA address issue |
| `hang_unknown` | HARD (1) | Unclear root cause |

- **Hard budget**: 6 attempts per stage (only HARD failures consume this)
- **Free safety cap**: 10 retries (prevents infinite loops on repeated easy errors)
- A stage is exhausted when `budget_exhausted == true`

**If NOT exhausted**: go to step 2, include failure JSON in the kernel-writer prompt:
```
RETRY — Previous attempt failed:
  Classification: {classification} (cost: {cost})
  Summary: {summary}
  Action: {suggested_action}

Fix the issue described above. Do not change parts that were working.
```

**If exhausted**:
```bash
python3 .claude/scripts/tdd-pipeline/tdd_orchestrator.py rollback --op-path <path>
```
STOP. Report: `HUMAN REVIEW REQUIRED — see {op_path}/tdd_failure_report.md`

---

## Anti-Patterns

| Anti-Pattern | Why It's Wrong | What To Do Instead |
|--------------|----------------|-------------------|
| Skip `advance` after a pass | Gate marker not cleared, next stage breaks | Always advance before moving on |
| Register stages after pipeline started | Index confusion in state file | Register ALL stages before first `test` |
| Test before kernel-writer returns | Wastes an attempt on unchanged stubs | Wait for kernel-writer to complete |
| Implement future stages early | Untested code that may break subtly | Each stage implements ONLY its scope |
| Commit without gate marker | Pre-commit hook will reject | Re-run `test` to regenerate marker |
| Change multiple kernels without considering impact | May break earlier stages | Only modify kernels listed for the current stage |
