# Layer Norm RM — Operation Creation Report

## Executive Summary

Successfully created the `layer_norm_rm` TTNN operation using the Generic Op Workflow (Python-based, `ttnn.generic_op()`). The operation implements row-major layer normalization on Tenstorrent hardware with single-core execution.

**Final test results: 61 passed, 4 skipped, 0 failed**

- 8 shapes x 4 gamma/beta combinations x 2 dtypes = 64 correctness tests + 1 validation test
- 4 skips: float32 with shape `[1,1,32,1024]` exceeds L1 capacity (1.7MB > 1.5MB for single-core)
- bfloat16 PCC > 0.999 (threshold: 0.99)
- float32 PCC = 1.000000 (threshold: 0.999)

---

## Pipeline Execution Summary

### Phase 0: Discovery (Orchestrator)

**Decision**: Hybrid Mode with 3 references — this operation has row-major input, compute, and row-major output, requiring tilize + compute + untilize patterns.

| Role | Reference | Factory Path |
|------|-----------|-------------|
| input_stage | tilize (single-core) | `tilize_single_core_program_factory.cpp` |
| compute_core | softmax (general W) | `softmax_program_factory_general.cpp` |
| output_stage | untilize (single-core) | `untilize_single_core_program_factory.cpp` |

**Rationale**: Tilize/untilize single-core matched the single-core requirement. Softmax general provides the closest row-wise reduction + broadcast + element-wise pattern to layer normalization.

---

### Phase 1: Analyzers (3 in parallel)

| Analyzer | Duration | Output |
|----------|----------|--------|
| Tilize single-core | ~5.5 min | `agent_logs/tilize_single_core_analysis.md` |
| Softmax general | ~6.3 min | `agent_logs/softmax_general_analysis.md` |
| Untilize single-core | ~5.6 min | `agent_logs/untilize_single_core_analysis.md` |

**Key findings**:
- Tilize: Reader reads 32 RM sticks per tile-row, compute tilizes via `tilize_init_short()` + `tilize_block()`
- Softmax: Row-wise reduction using reduce scaler (1/W), broadcast subtract (COL), multiple intermediate CBs for pipeline stages
- Untilize: Compute untilizes tiles back to RM sticks, writer writes sticks directly to DRAM

**Breadcrumbs**: `agent_logs/analyzer_*_breadcrumbs.jsonl`

---

### Phase 2: Planner

**Duration**: ~5.5 min
**Output**: `layer_norm_rm_spec.md` (36KB, comprehensive)

**Key decisions**:
1. **WSmall pattern**: All Wt tiles fit in L1 simultaneously, enabling two-pass computation without DRAM re-reads
2. **16 circular buffers**: Carefully assigned across input (0-7), output (8, 16), and intermediate (24-30) ranges
3. **Reduce scaler = 1/W**: Hardware auto-multiplies SUM result by 1/W, computing mean directly
4. **Reduce scaler CB always bfloat16**: Hardware requirement regardless of operation dtype
5. **Gamma/beta tilized once**: Read once at program start, reused across all tile-rows

**Breadcrumbs**: `agent_logs/planner_breadcrumbs.jsonl`

---

### Phase 3a: Generic Op Builder (parallel)

**Duration**: ~6.1 min
**Output**: Python orchestration + stub kernels

**Files produced**:
- `__init__.py` — Package initialization
- `layer_norm_rm.py` — Entry point with output allocation (140 lines)
- `layer_norm_rm_program_descriptor.py` — CB config, runtime args (440 lines)
- `test_layer_norm_rm.py` — Parametrized test suite (204 lines)
- `kernels/layer_norm_rm_reader.cpp` — Stub reader
- `kernels/layer_norm_rm_compute.cpp` — Stub compute
- `kernels/layer_norm_rm_writer.cpp` — Stub writer

**Key decisions**:
- Output tensor LAST in `io_tensors` list (generic_op requirement)
- `allocate_tensor_on_device` with positional args
- torch imports inside functions (pre-commit hook compliance)
- PCC-based correctness checks (not torch.allclose)

**Breadcrumbs**: `agent_logs/generic_op_builder_breadcrumbs.jsonl`

---

### Phase 3b: Kernel Designer (parallel)

**Duration**: ~3.1 min
**Output**: `kernel_design.md` (12.5KB)

**Helper vs raw call mapping**:

| Phase | Operation | Decision | Function |
|-------|-----------|----------|----------|
| Tilize input | RM → tiles | USE HELPER | `compute_kernel_lib::tilize()` |
| Mean reduction | SUM across W | USE HELPER | `compute_kernel_lib::reduce<SUM, REDUCE_ROW, WaitUpfrontNoPop>()` |
| Centering | X - mean | USE HELPER | `compute_kernel_lib::sub<COL>()` |
| Square | (X-mean)^2 | USE HELPER | `compute_kernel_lib::square<WaitUpfrontNoPop>()` |
| Variance | SUM of squares | USE HELPER | `compute_kernel_lib::reduce<SUM, REDUCE_ROW>()` |
| Add epsilon | var + eps | USE HELPER | `compute_kernel_lib::add<SCALAR>()` |
| Rsqrt | 1/sqrt(var+eps) | NO HELPER | `copy_tile` + `rsqrt_tile` + `pack_tile` |
| Normalize | centered * rstd | USE HELPER | `compute_kernel_lib::mul<COL>()` |
| Gamma multiply | normed * gamma | USE HELPER | `compute_kernel_lib::mul<NONE>()` |
| Beta add | result + beta | USE HELPER | `compute_kernel_lib::add<NONE>()` |
| Untilize output | tiles → RM | USE HELPER | `compute_kernel_lib::untilize()` |

---

### Phase 4: Kernel Writer (incremental)

**Duration**: ~96 min (longest phase — 4 incremental steps with testing)
**Output**: Working reader, compute, writer kernels

#### Incremental Steps

**Step 1 — Passthrough (tilize → untilize)**: Validated full data path. Reader reads RM sticks, compute tilizes then immediately untilizes, writer writes RM sticks.

**Step 2 — Mean + subtraction**: Added row-wise reduction with 1/W scaler and broadcast column subtraction. Verified against `X - mean(X, dim=-1, keepdim=True)`.

**Step 3 — Variance + rstd**: Added square, variance reduction, epsilon addition, and rsqrt. Verified standardized output against PyTorch.

**Step 4 — Gamma/beta affine**: Added gamma/beta reading, tilization, and affine transform. Verified against `torch.nn.functional.layer_norm`.

#### Bug Fixes During Implementation

1. **Intermediate CB format mismatch**: Changed from float32 to match input dtype. Float32 intermediates caused `reconfig_data_format()` failures when switching unpacker between bf16 and f32 modes. Precision maintained via `fp32_dest_acc_en=True`.

2. **Epsilon scalar format**: Changed to use `generate_bcast_scalar_bfloat16()` with packed bf16 pair format to match the (now-bf16) intermediate CB format.

3. **Binary op NONE broadcast bug**: The `binary_op` helper with `BroadcastDim::NONE`, `WaitAndPopPerTile` for A, and `NoWaitNoPop` for B produced wrong results when Wt > 1. Worked around with manual `mul_tiles`/`add_tiles` loops for gamma/beta application.

4. **Import paths**: Fixed `ttnn.ttnn.operations` → `ttnn.operations`.

---

## Deviations from Spec

| Deviation | Rationale |
|-----------|-----------|
| Gamma/beta phases use manual `mul_tiles`/`add_tiles` instead of `binary_op` helper | Confirmed bug in helper with `NONE` broadcast + `WaitAndPopPerTile`/`NoWaitNoPop` policy combination when Wt > 1 |
| Intermediate CB format matches input dtype instead of always float32 | Avoids unpacker reconfiguration issues; precision preserved by `fp32_dest_acc_en=True` |
| 4 float32 tests skipped for `[1,1,32,1024]` | L1 capacity exceeded (1.7MB > 1.5MB) — expected for single-core with large W and 4-byte elements |

---

## File Inventory

### Implementation Files
| File | Lines | Purpose |
|------|-------|---------|
| `__init__.py` | ~10 | Package re-export |
| `layer_norm_rm.py` | ~140 | Entry point, output allocation, generic_op call |
| `layer_norm_rm_program_descriptor.py` | ~440 | CB config, runtime args, kernel setup |
| `test_layer_norm_rm.py` | ~200 | Parametrized PyTorch comparison tests |
| `kernels/layer_norm_rm_reader.cpp` | ~130 | Reduce scaler, epsilon, gamma/beta, input reading |
| `kernels/layer_norm_rm_compute.cpp` | ~250 | 10-phase compute pipeline |
| `kernels/layer_norm_rm_writer.cpp` | ~45 | RM stick writer |

### Documentation Files
| File | Purpose |
|------|---------|
| `layer_norm_rm_spec.md` | Functional specification (36KB) |
| `kernel_design.md` | Kernel helper/raw call mapping (12.5KB) |
| `IMPLEMENTATION_SUMMARY.md` | Generic op builder summary |
| `REPORT.md` | This report |

### Analysis & Logs
| File | Purpose |
|------|---------|
| `agent_logs/tilize_single_core_analysis.md` | Tilize reference analysis |
| `agent_logs/softmax_general_analysis.md` | Softmax reference analysis |
| `agent_logs/untilize_single_core_analysis.md` | Untilize reference analysis |
| `agent_logs/analyzer_*_breadcrumbs.jsonl` | Analyzer execution traces |
| `agent_logs/planner_breadcrumbs.jsonl` | Planner execution trace |
| `agent_logs/generic_op_builder_breadcrumbs.jsonl` | Builder execution trace |
| `agent_logs/critical_patterns.md` | Critical patterns reference |

---

## Git History

```
0bda25b [ttnn-kernel-writer] stage 7: implement layer_norm_rm kernels with full correctness
ef03fb0 [ttnn-generic-op-builder] Implement layer_norm_rm generic_op infrastructure
78755fe [ttnn-kernel-designer] chore: commit parallel agent test file to unblock stop hook
da6130c [ttnn-kernel-designer] chore: commit parallel agent files to unblock stop hook
ace7173 [ttnn-kernel-designer] design: layer_norm_rm
c633a78 [ttnn-operation-planner] spec: layer_norm_rm
3e56fd6 [ttnn-operation-analyzer] Softmax general W-dimension analysis
22a0c07 [ttnn-operation-analyzer] Add tilize single-core analysis
bf84be8 [ttnn-operation-analyzer] Add untilize single-core analysis
```

---

## Timing Summary

| Phase | Agent(s) | Duration | Mode |
|-------|----------|----------|------|
| Phase 1 | 3 analyzers | ~6 min | Parallel |
| Phase 2 | Planner | ~5.5 min | Sequential |
| Phase 3 | Builder + Designer | ~6 min | Parallel |
| Phase 4 | Kernel Writer | ~96 min | Sequential (4 incremental steps) |
| **Total** | | **~114 min** | |
