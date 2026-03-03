# layer_norm_rm — Operation Creation Report

## Summary

**Operation**: `layer_norm_rm` — Layer normalization on row-major interleaved tensors
**Result**: ALL 3 TDD STAGES PASSED
**Total pipeline time**: ~35 minutes across 5 phases
**Mode**: Fully automated (no manual confirmations)
**Breadcrumb logging**: Enabled

### What it does

Computes layer normalization across the W (last) dimension of a row-major interleaved tensor:
```
mean = sum(x) / W
centered = x - mean
var = sum(centered^2) / W
inv_std = rsqrt(var + epsilon)
output = gamma * centered * inv_std + beta
```

Inputs: bfloat16 RM interleaved tensors (input, gamma, beta). Output: same shape as input.

---

## Pipeline Execution

| Phase | Agent | Duration | Output | Status |
|-------|-------|----------|--------|--------|
| 0 - Discovery | Orchestrator | ~1 min | Reference selection: tilize, untilize, softmax | Done |
| 1 - Analysis | 3x ttnn-operation-analyzer (parallel) | ~7 min | tilize_analysis.md, untilize_analysis.md, softmax_analysis.md | Done |
| 2 - Design | ttnn-operation-architect | ~8 min | op_design.md, .tdd_state.json, 3 test stage files | Done |
| 3 - Build | ttnn-generic-op-builder | ~10 min | Python infra + stub kernels + integration test | Done |
| 4 - TDD Kernels | ttnn-kernel-writer-tdd | ~9 min | 3 kernel files implemented, all 3 stages passed | Done |
| 5 - Report | Orchestrator | ~1 min | This file | Done |

---

## Agent Summaries

### Phase 1: Analyzers (3 parallel agents)

**Tilize Analyzer (input_stage)**
- Analyzed `tilize_multi_core_interleaved_program_factory.cpp`
- Key findings: stick-to-tile batching pattern (32 RM sticks per block), TensorAccessor setup, 1D block-based core distribution
- Commit: `ef6add2dab`

**Untilize Analyzer (output_stage)**
- Analyzed `untilize_multi_core_program_factory.cpp`
- Key findings: untilize helper usage, writer kernel RM stick extraction pattern, output CB sizing
- Commit: `bc124a87aa`

**Softmax Analyzer (compute_core)**
- Analyzed `softmax_program_factory_general_w_small.cpp`
- Key findings: multi-phase row-wise computation with CB persistence via `WaitUpfrontNoPop`, reduce helper with post-reduce lambda, scalar broadcast pattern for row statistics, constant CBs
- Commit: `7da33267b8`

### Phase 2: Architect

- Designed 14 circular buffers (3 RM input, 2 scaler, 1 RM output, 8 tiled intermediates)
- 8 compute phases: tilize → mean reduce → subtract mean → square → variance reduce+rsqrt → multiply inv_std → multiply gamma → add beta → untilize
- Single-core implementation for initial version
- Registered 3 TDD stages: data_pipeline, subtract_mean, full_layer_norm
- Commit: `1a0128e61c`

### Phase 3: Builder

- Created `layer_norm_rm.py` (entry point with validation)
- Created `layer_norm_rm_program_descriptor.py` (14 CB descriptors, 3 kernel descriptors)
- Created 3 stub kernels (reader, compute, writer)
- Created integration test and conftest
- Commit: `a5c0e554ed`

### Phase 4: TDD Kernel Writer

- Implemented all 3 kernel files through 3 TDD stages
- Stage 1 (data_pipeline): identity pass-through with tilize/untilize — passed first try
- Stage 2 (subtract_mean): mean reduce + broadcast subtract — passed first try
- Stage 3 (full_layer_norm): all 8 compute phases — passed on second try (first attempt had a compilation error)
- Commits: `3ddf343bc3`, `2a09bd1132`, `e48117fedc`

---

## TDD Pipeline Results

| Stage | Description | Status | Hard Attempts | Free Retries | Failure Classifications |
|-------|-------------|--------|---------------|--------------|------------------------|
| data_pipeline | Tilize input → identity → untilize output | PASSED | 0/6 | 0 | None |
| subtract_mean | Mean reduce + subtract broadcast | PASSED | 0/6 | 0 | None |
| full_layer_norm | All phases: mean, center, var, rsqrt, scale, gamma, beta | PASSED | 0/6 | 1 | `compilation_error` (FREE) |

### Stage 3 Failure Detail

The single failure was a compilation error: `'WaitUpfrontPopAtEnd' is not a member of 'compute_kernel_lib::ReduceInputPolicy'`. The kernel writer used the wrong enum member name for the reduce variance phase. Fixed by switching to `BulkWaitBulkPop` policy. Classification: `compilation_error` (FREE retry — did not consume a hard attempt).

---

## Files Produced

### Operation (ttnn/ttnn/operations/layer_norm_rm/)
```
├── __init__.py                          # Re-exports layer_norm_rm function
├── layer_norm_rm.py                     # Entry point with validation
├── layer_norm_rm_program_descriptor.py  # CB config, kernel setup, ProgramDescriptor
├── kernels/
│   ├── layer_norm_rm_reader.cpp         # RM stick reader + scaler/eps prep
│   ├── layer_norm_rm_compute.cpp        # 8-phase tile computation
│   └── layer_norm_rm_writer.cpp         # RM stick writer
├── op_design.md                         # Architecture + kernel design document
├── .tdd_state.json                      # TDD pipeline state (all passed)
├── REPORT.md                            # This report
└── agent_logs/
    ├── tilize_analysis.md               # Input stage reference analysis
    ├── untilize_analysis.md             # Output stage reference analysis
    └── softmax_analysis.md              # Compute core reference analysis
```

### Tests (tests/ttnn/unit_tests/operations/layer_norm_rm/)
```
├── __init__.py
├── conftest.py                          # Device fixture
├── layer_norm_rm.py                     # Test entry point wrapper
├── test_layer_norm_rm.py                # Integration test
├── test_stage_data_pipeline.py          # TDD stage 1 test
├── test_stage_subtract_mean.py          # TDD stage 2 test
└── test_stage_full_layer_norm.py        # TDD stage 3 test
```

---

## Git History

```
e48117fedc [ttnn-kernel-writer-tdd] stage full_layer_norm: passed
2a09bd1132 [ttnn-kernel-writer-tdd] stage subtract_mean: passed
3ddf343bc3 [ttnn-kernel-writer-tdd] stage data_pipeline: passed
a5c0e554ed [ttnn-generic-op-builder] stubs: layer_norm_rm
1a0128e61c [ttnn-operation-architect] design: layer_norm_rm
7da33267b8 [ttnn-operation-analyzer] analysis: softmax (compute_core focus)
ef6add2dab [ttnn-operation-analyzer] analysis: tilize (multi-core interleaved)
bc124a87aa [ttnn-operation-analyzer] analysis: untilize (output_stage focus)
```

---

## Decisions and Deviations

### Assumptions Made (Automated Mode)
1. **Single-core implementation**: Chose single-core for simplicity. Multi-core extension would add block distribution logic.
2. **Epsilon default**: Set to 1e-5 (standard PyTorch default for LayerNorm).
3. **Tolerance values**: data_pipeline rtol=0.01, subtract_mean rtol=0.02/atol=0.1, full_layer_norm rtol=0.05/atol=0.2 — progressively looser to account for cumulative bfloat16 precision loss.
4. **Gamma/beta reading**: Reader reads single RM stick (the real data row) and relies on zero-initialized CB memory for the remaining 31 padding rows. This avoids reading phantom sticks.

### Deviations from Design
1. **Phase 4b (eps + rsqrt)**: The architect designed this as a reduce post-op lambda, but the kernel writer implemented it as a separate `add<SCALAR>` followed by rsqrt in a lambda. This is functionally equivalent but uses an intermediate CB differently.
2. **CB reuse**: The kernel writer reused `cb_mean` (c_25) as temporary storage for inv_std after adding epsilon, and reused `cb_input_tiled` (c_24) and `cb_centered` (c_26) as intermediate output buffers in later phases. This is valid since the data was consumed before reuse.

### Pain Points
1. **Reduce input policy naming**: The kernel writer initially used `WaitUpfrontPopAtEnd` for reduce (which doesn't exist in the ReduceInputPolicy enum). The correct name was `BulkWaitBulkPop`. This was a FREE compilation error, caught and fixed automatically.
2. **Analysis size**: Each analyzer produced 23-32KB of analysis. The role-based focus directives helped but the output is still substantial.

---

## Infrastructure Issues

- **No device hangs encountered**: All kernel runs completed without hangs.
- **No build failures**: The C++ build was already complete before the pipeline started.
- **No venv issues**: Python environment was pre-configured.
- **One compilation error**: Incorrect enum member name for reduce input policy (auto-fixed).
- **Environment artifacts**: Files like `.build_complete`, `.venv_complete`, `.claude/active_logging` existed from the evaluation infrastructure but did not interfere.

---

## Suggestions for Improving the Agent Pipeline

1. **Analyzer output trimming**: Even with role-based focus directives, analyses are 23-32KB. A structured summary section at the top (< 2KB) would help the architect extract key patterns faster.

2. **Enum/API validation**: The kernel writer's compilation error (wrong policy enum name) could be prevented by having the architect or design document include a verified list of enum members for each helper API.

3. **CB reuse documentation**: The design document should explicitly specify which CBs can be reused after which phases. The kernel writer had to infer this, which worked but adds cognitive load.

4. **Multi-core extension path**: The single-core design is complete but the pipeline doesn't provide a follow-up path for multi-core extension. A "Phase 5b: Scale" stage could be useful.

5. **Test tolerance auto-calibration**: The tolerances were manually set by the architect. An auto-calibration step (run PyTorch reference in bfloat16, measure actual variance, set tolerance) would be more robust.

6. **Parallel Phase 2+3**: The architect and builder are sequential but largely independent (builder only needs the CB layout and kernel args, not the full implementation strategy). Partial parallelism could save ~5 minutes.
