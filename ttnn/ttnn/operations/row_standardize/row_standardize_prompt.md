# Row Standardize Operation — Generic Op Pipeline Prompt

Build a new TTNN operation called `row_standardize` using the **generic op pipeline** (Python-based, no C++ scaffolding).

## Mathematical Definition

For each row (last dimension) of the input tensor:

```
output[..., i] = (x[..., i] - mean_row) * rsqrt(var_row + epsilon)
```

Where:

```
mean_row = (1/W) * sum(x[..., j] for j in 0..W-1)
var_row  = (1/W) * sum((x[..., j] - mean_row)^2 for j in 0..W-1)
```

This is equivalent to `(x - mean) / sqrt(var + eps)` per row — i.e., layer norm without learnable affine parameters (no gamma/beta), applied along dim=-1.

## Operation Parameters

- **Input tensor**: bfloat16 OR float32, ROW_MAJOR layout, interleaved memory, at least 2D
- **Output tensor**: Same shape, dtype, layout, and memory config as input
- **epsilon**: float, default = 1e-5 (runtime parameter, NOT compile-time)
- **Supported dtypes**: bfloat16, float32
  - CB page sizes MUST be dtype-aware: tile_size = 32×32×datum_size (2048 for bf16, 4096 for f32)
  - The program descriptor / factory must compute page sizes from the input tensor's dtype at runtime
  - When input dtype is float32, **fp32 dest accumulation MUST be enabled** in the compute kernel (to preserve full precision through the reduction and normalization pipeline)
- Operation name: `row_standardize`
- Operation path: `ttnn/ttnn/operations/row_standardize/`

## Tensor Shape Constraints

- Last dimension (W) must be a multiple of 32 (tile width)
- Second-to-last dimension (H) must be a multiple of 32 (tile height)
- All other dimensions are treated as batch dimensions (flattened into rows)

## Compute Pattern (CRITICAL)

Since input/output are ROW_MAJOR but all compute requires TILES:

```
RM input → read sticks → tilize → compute (tiles) → untilize → write sticks → RM output
```

This makes it a **Hybrid Mode** operation with 3 reference roles:

1. **input_stage**: tilize operation (RM sticks → tiles)
2. **compute_core**: layernorm or similar row-reduction operation (mean, variance, normalize)
3. **output_stage**: untilize operation (tiles → RM sticks)

## Reference Operations to Analyze

The orchestrator should discover appropriate reference program factories for:

- Tilize (interleaved variant, for input_stage)
- A row-reduction + normalization op like layernorm (for compute_core)
- Untilize (interleaved variant, for output_stage)

## Execution Mode

Run in **FULLY AUTOMATED** mode:

- Introduce reasonable assumptions; do NOT ask for user confirmation at any checkpoint
- Skip ALL user review steps (spec approval, Phase 3 output review, etc.)
- Proceed through the entire pipeline: analyzer(s) → planner → (generic_op_builder || kernel_designer) → kernel_writer
- If ambiguity arises, choose the simpler/more conservative option and LOG the decision

## Breadcrumbs Logging (MANDATORY)

Before launching ANY agents, the orchestrator MUST:

1. Create the operation directory: `mkdir -p ttnn/ttnn/operations/row_standardize/agent_logs`
2. Create the logging config file by running:
   ```
   ttnn/experimental/claude_ttnn_agents/scripts/logging/set_logging_config.sh ttnn/ttnn/operations/row_standardize --enable --verbosity=detailed
   ```
3. Instruct EVERY agent with "enable detailed logging" / "with breadcrumbs" in their prompts
4. Each agent should use `init_breadcrumbs.sh` and `append_breadcrumb.sh` to log events

## Agent Pain Point & Decision Reporting

Each agent MUST, in its final output:

- Report any **pain points** encountered (ambiguous specs, missing info, unexpected behavior)
- Report every **decision** made that wasn't explicitly specified
- Report every **deviation** from the spec with justification
- Log these as `deviation` breadcrumb events

## Final Summary Report

After ALL agents complete, produce a well-structured markdown report at:

```
ttnn/ttnn/operations/row_standardize/pipeline_report.md
```

The report must include:

1. **Executive Summary**: What was built, final test status
2. **Pipeline Overview**: Which agents ran, in what order, total duration
3. **Per-Agent Summaries** (one section each for: analyzer(s), planner, generic_op_builder, kernel_designer, kernel_writer):
   - What the agent produced
   - Key decisions made
   - Pain points encountered
   - Deviations from spec
   - Reference to breadcrumbs file: `agent_logs/{agent_name}_breadcrumbs.jsonl`
   - Reference to execution log: `agent_logs/{agent_name}_execution_log.md`
4. **Architecture Summary**: CB layout, kernel structure, data flow
5. **Test Results**: Shapes tested, PCC values, pass/fail
6. **Lessons Learned**: Cross-agent issues, what could be improved

## Test Criteria

- Use PCC (Pearson Correlation Coefficient) > 0.99 for correctness (NOT torch.allclose)
- PyTorch reference:
  ```python
  (x - x.mean(dim=-1, keepdim=True)) / torch.sqrt(x.var(dim=-1, unbiased=False, keepdim=True) + epsilon)
  ```
- Test shapes AND dtypes (parameterized via pytest, cartesian product):
  - **Shapes**:
    - (32, 32) — minimal tile-aligned 2D
    - (32, 64) — single tile row, two tile cols
    - (64, 128) — small square-ish
    - (128, 128) — medium square
    - (32, 1024) — single tile row, wide
    - (128, 1024) — medium batch, wide
    - (1024, 32) — tall, single tile col
    - (1024, 1024) — large square
    - (2, 32, 64) — 3D, small batch
    - (4, 64, 128) — 3D, medium batch
    - (2, 4, 32, 64) — 4D
  - **Dtypes**:
    - ttnn.bfloat16 (PCC > 0.99)
    - ttnn.float32 (PCC > 0.999, tighter since no precision loss)
- Use the `device` fixture, do NOT open device manually
- Import torch inside test functions, not globally
