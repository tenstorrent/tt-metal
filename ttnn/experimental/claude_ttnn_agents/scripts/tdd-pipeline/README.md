# Self-Healing TDD Pipeline for TTNN Kernel Writer

A stage-gated test-driven development pipeline for implementing TTNN kernels incrementally. Each stage is independently testable with a PyTorch reference, with automatic failure classification, retry limits, and rollback to last passing commit.

## Quick Start

```bash
# 1. Initialize pipeline from an operation spec
python3 .claude/scripts/tdd-pipeline/tdd_orchestrator.py init path/to/my_op_spec.md --op-path ttnn/ttnn/operations/my_op

# 2. Register stages (order matters — they execute sequentially)
python3 .claude/scripts/tdd-pipeline/tdd_orchestrator.py add-stage '{
  "name": "data_pipeline",
  "description": "Reader + writer, passthrough compute",
  "reference_body": "return input_tensor.clone()",
  "tolerance": {"rtol": 0.01, "atol": 0.01},
  "shapes": ["(1, 1, 32, 64)", "(1, 1, 64, 128)"]
}' --op-path ttnn/ttnn/operations/my_op

# 3. Implement kernels for the current stage, then test
python3 .claude/scripts/tdd-pipeline/tdd_orchestrator.py test --op-path ttnn/ttnn/operations/my_op

# 4. On pass, advance to next stage
python3 .claude/scripts/tdd-pipeline/tdd_orchestrator.py advance --op-path ttnn/ttnn/operations/my_op

# 5. Check progress
python3 .claude/scripts/tdd-pipeline/tdd_orchestrator.py status --op-path ttnn/ttnn/operations/my_op
```

## CLI Reference

### `init <spec_path> [--op-path PATH]`

Initialize a TDD pipeline from an operation spec file.

- Creates `.tdd_state.json` in the operation directory
- Parses spec for layout hints (TILE_LAYOUT vs ROW_MAJOR_LAYOUT)
- Does NOT create stages — use `add-stage` for that

**Exit codes:** 0 success, 1 error

**Example:**
```bash
python3 .claude/scripts/tdd-pipeline/tdd_orchestrator.py init \
  ttnn/ttnn/operations/my_op/my_op_spec.md \
  --op-path ttnn/ttnn/operations/my_op
```

### `add-stage <stage_json> [--from-file PATH] [--op-path PATH]`

Register a new test stage. Appends to the stage list and renders a test file.

- Stage JSON can be inline or from a file (`--from-file`)
- Test file is rendered from `test_stage_template.py.j2` into the op directory
- Stage names must be snake_case and unique

**Exit codes:** 0 success, 1 error

**Example:**
```bash
python3 .claude/scripts/tdd-pipeline/tdd_orchestrator.py add-stage '{
  "name": "full_compute",
  "description": "Add row-mean subtraction",
  "reference_body": "return input_tensor - input_tensor.mean(dim=-1, keepdim=True)",
  "tolerance": {"rtol": 0.02, "atol": 0.1},
  "shapes": ["(1, 1, 32, 64)", "(2, 1, 32, 256)"],
  "output_shape_expr": "list(shape)"
}' --op-path ttnn/ttnn/operations/my_op
```

### `test [stage_name] [--op-path PATH]`

Run the test for the current stage (or a named stage) via `dev-test.sh`.

- Captures output and classifies failures automatically
- Creates/removes `.tdd_gate_passed` marker based on result
- Increments attempt counter on failure

**Exit codes:** 0 pass, 1 test failure, 2 hang detected

**Example:**
```bash
python3 .claude/scripts/tdd-pipeline/tdd_orchestrator.py test --op-path ttnn/ttnn/operations/my_op
```

### `advance [--op-path PATH]`

Advance to the next stage after the current one passes.

- Requires `.tdd_gate_passed` marker (created by `test` on pass)
- Records current git commit as the stage's passing commit
- Updates `last_passing_commit` for rollback
- Sets next stage to `in_progress`

**Exit codes:** 0 success, 1 error

### `rollback [--op-path PATH]`

Rollback kernel files to the last passing commit.

- Runs `git checkout <commit> -- <op_path>/kernels/`
- Marks current stage as `failed_permanent`
- Generates `tdd_failure_report.md` with all failure attempts
- Prints "HUMAN REVIEW REQUIRED"

**Exit codes:** 1 always (signals human intervention needed)

### `status [--op-path PATH] [--json]`

Show pipeline progress summary.

- Without `--json`: human-readable table
- With `--json`: raw state file for programmatic use

**Exit codes:** 0 always

### `parse-failure [--op-path PATH]`

Output the last failure as structured JSON to stdout.

**Output format:**
```json
{
  "stage": "tilize_untilize",
  "attempt": 2,
  "classification": "hang_cb_deadlock",
  "summary": "BRISC stuck at cb_wait_front(cb_tilized) on 8 cores",
  "details": {"stuck_riscv": "BRISC", "stuck_cb": "cb_tilized"},
  "suggested_action": "Check CB sync: total pushes must equal total pops.",
  "remaining_attempts": 1
}
```

## Stage JSON Schema

| Field | Required | Type | Description |
|-------|----------|------|-------------|
| `name` | Yes | string | Snake_case identifier, used in test filename |
| `description` | Yes | string | Human-readable, used in test docstring |
| `reference_body` | Yes | string | Python expression for pytorch_reference body |
| `tolerance` | Yes | `{"rtol": float, "atol": float}` | Numerical comparison tolerances |
| `shapes` | Yes | list of strings | Tensor shapes as tuple strings |
| `kernel_files` | No | list of strings | Informational: which kernel files this stage touches |
| `extra_imports` | No | string | Additional imports for the test file |
| `extra_args` | No | string | Additional args to operation call (e.g., `, gamma, beta`) |
| `extra_setup` | No | string | Extra tensor setup code before op call |
| `extra_ttnn_setup` | No | string | Extra TTNN tensor setup code after input creation |
| `output_shape_expr` | No | string | Output shape expression if different from input |
| `max_attempts` | No | int | Override default max attempts (default: 3) |

## Architecture

```
Orchestrator Agent                   tdd_orchestrator.py              dev-test.sh
    |                                      |                              |
    |-- init <spec> ---------------------->| Create .tdd_state.json       |
    |                                      |                              |
    |-- add-stage '{json}' -------------->| Register stage + render      |
    |                                      | test file from template      |
    |                                      |                              |
    |-- [kernel-writer implements] --------|------------------------------|
    |                                      |                              |
    |-- test <stage_name> --------------->|---- dev-test.sh ------------>|
    |                                      |<--- exit code + output -----|
    |                                      |                              |
    |<-- PASS / structured failure --------|                              |
    |                                      |                              |
    |-- advance ------------------------->| Record commit, next stage    |
    |     OR                               |                              |
    |-- rollback ------------------------>| git checkout kernels/        |
```

## State File (`.tdd_state.json`)

Lives in the operation directory. Example:

```json
{
  "op_name": "my_op",
  "op_path": "ttnn/ttnn/operations/my_op",
  "spec_path": "ttnn/ttnn/operations/my_op/my_op_spec.md",
  "layout": "ROW_MAJOR_LAYOUT",
  "current_stage_index": 1,
  "last_passing_commit": "abc1234...",
  "stages": [
    {
      "name": "data_pipeline",
      "status": "passed",
      "commit": "abc1234...",
      "attempts": 1,
      "max_attempts": 3,
      "failure_history": []
    },
    {
      "name": "full_compute",
      "status": "in_progress",
      "commit": null,
      "attempts": 1,
      "max_attempts": 3,
      "failure_history": [{"classification": "numerical_mismatch", "...": "..."}]
    }
  ]
}
```

**Status values:** `pending` | `in_progress` | `passed` | `failed_permanent`

## Pre-Commit Hook

The hook blocks commits that touch operation files when the TDD gate hasn't passed.

**Install:**
```bash
bash .claude/scripts/tdd-pipeline/install_hooks.sh
```

**How it works:**
1. Scans staged files for operation directories (`ttnn/ttnn/operations/*`)
2. For each operation with a `.tdd_state.json`, checks for `.tdd_gate_passed`
3. Blocks the commit if the marker is missing

**Bypass (emergency):** `git commit --no-verify`

## Failure Classifications

| Classification | Trigger | Suggested Action |
|---------------|---------|-----------------|
| `numerical_mismatch` | `Max diff` or `allclose` assertion | Check helper parameters, CB data format, tile counts |
| `compilation_error` | `CompileError`, `undefined reference` | Check includes, template parameters, API signatures |
| `shape_mismatch` | Shape assertion failure | Verify output shape in ProgramDescriptor |
| `runtime_error` | `RuntimeError`, `TypeError`, etc. | Check runtime args, tensor addresses, buffer sizes |
| `watcher_assert` | `tripped assert on line N` | Check the kernel at that line (NoC alignment, CB bounds) |
| `hang_cb_deadlock` | `cb_wait_front` in triage | CB sync mismatch: pushes != pops |
| `hang_noc_error` | NoC error in triage | Address alignment (32-byte), invalid DRAM addresses |
| `hang_unknown` | Hang without clear pattern | Check all three kernels for progress |

## Troubleshooting

### Stale state file
If `.tdd_state.json` gets out of sync, edit it directly or delete and re-init.

### Gate marker confusion
The `.tdd_gate_passed` marker is created by `test` on pass and removed by `advance` or `rollback`. If manually deleted, just re-run `test`.

### Rollback with no passing commit
If stage 0 fails, there's no commit to rollback to. The orchestrator prints a warning and only marks the stage as `failed_permanent`.

### Test timeout
The `test` subcommand has a 5-minute overall timeout (on top of dev-test.sh's per-operation timeout). If hit, the process is killed and classified as a hang.

### Template rendering without Jinja2
If `jinja2` is not installed, the orchestrator falls back to simple string replacement. The output is functionally identical.

## Design Decisions

- **Dynamic stages (not fixed phases):** Kernel implementation is cross-kernel. A single stage may touch reader + compute + writer simultaneously. The agent defines stages based on the operation's complexity.
- **Kernel-only rollback:** Only kernel files are rolled back on failure. The test files, state, and program factory are preserved so the agent (or human) can retry with a different approach.
- **Marker files for gate:** A simple `.tdd_gate_passed` file is used instead of state-file flags because git hooks need a fast, filesystem-level check without parsing JSON.
- **Failure history accumulation:** All failure attempts are preserved in the state file for debugging. The failure report generated on rollback includes the full history.
