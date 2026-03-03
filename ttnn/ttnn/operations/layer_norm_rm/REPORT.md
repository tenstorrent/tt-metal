# layer_norm_rm — Pipeline Execution Report

## 1. Summary

**Operation**: `layer_norm_rm` — Row-major layer normalization
**Math**: `output = gamma * (x - mean(x)) / sqrt(var(x) + eps) + beta`
**Overall Result**: ALL 4 TDD STAGES PASSED
**Total Pipeline Duration**: ~35 minutes
**Commits**: 8 (3 analysis + 1 design + 1 build + 4 TDD stages — but `data_pipeline` was pre-passing from builder)

---

## 2. Pipeline Execution

| Phase | Agent | Duration (approx) | Output |
|-------|-------|--------------------|--------|
| 0: Discovery | Main orchestrator | ~1 min | 3 references identified |
| 1: Analysis | 3x `ttnn-operation-analyzer` (parallel) | ~6 min | 3 analysis `.md` files |
| 2: Design | `ttnn-operation-architect` | ~6 min | `op_design.md` + `.tdd_state.json` |
| 3: Build | `ttnn-generic-op-builder` | ~7 min | Python infra + stubs + 8/8 integration tests |
| 4: TDD Kernels | `ttnn-kernel-writer-tdd` | ~15 min | 4/4 stages passed |
| 5: Report | Main orchestrator | ~2 min | This file |

---

## 3. Agent Summaries

### Phase 1: Analyzers (3 parallel)

**Tilize Analyzer** (`tilize_multi_core_interleaved_analysis.md`, 25KB)
- Documented reader kernel reading RM sticks with TensorAccessor
- Identified `compute_kernel_lib::tilize<c_0, c_16>()` helper pattern
- 1D block distribution: 32 rows x full width per block
- Key insight: CB c_0 (RM sticks) → tilize → CB c_16 (tiles), single-buffered

**Untilize Analyzer** (`untilize_multi_core_analysis.md`, 28KB)
- Documented `untilize_block` compute helper pattern
- Writer kernel writes RM sticks back to DRAM
- Block distribution matching tilize pattern
- Key insight: CB (tiles) → untilize → CB (RM sticks) → writer to DRAM

**Softmax w_small Analyzer** (`softmax_w_small_analysis.md`, 25KB)
- Multi-pass data reuse pattern (MAX reduce → subtract → exp → SUM reduce → multiply)
- `WaitUpfrontNoPop` policy for keeping tiles in CB across phases
- 9 CBs total: input staging, scaler, mask, output, intermediates
- Scalar broadcasting via `generate_bcast_scaler<T>(cb, value)`
- Key insight: Layer norm needs identical multi-pass pattern with SUM reduce instead of MAX

### Phase 2: Architect

- Designed 9-phase fused compute pipeline (tilize → mean → center → square → variance → eps+rsqrt → normalize → gamma → beta → untilize)
- 13 CBs allocated (0-5 inputs/params, 8-9 scalars, 16 output, 24-28 intermediates)
- Row-based work distribution across cores
- Registered 4 TDD stages with progressive complexity
- Produced 28KB design document covering architecture + kernel implementation + helper mapping

### Phase 3: Builder

- Created all Python infrastructure: entry point, program descriptor, `__init__.py`
- Generated stub kernels (reader, compute, writer) that compile cleanly
- 8/8 integration tests passed (shape validation, dtype checks, gamma/beta variants)
- `data_pipeline` stage passed immediately with builder's stubs (tilize+untilize identity already worked)
- The builder's CB configuration, runtime args, and compile-time args were all correct — no upstream fixes needed during TDD

### Phase 4: TDD Kernel Writer

- Implemented all 4 stages in a single persistent session
- Total: 112 tool uses, 167K tokens consumed
- 9-phase compute kernel with correct helper usage
- Reader enhanced to handle gamma/beta stick replication (32x self-copy for tilize)
- No upstream fixes needed to program descriptor or entry point

---

## 4. TDD Pipeline Results

| Stage | Name | Result | Hard Attempts | Free Retries | Failure Classifications |
|-------|------|--------|---------------|--------------|------------------------|
| 0 | `data_pipeline` | PASS | 0 | 0 | None (pre-passing from builder) |
| 1 | `reduce_mean` | PASS | 2 | 1 | `compilation_error` (free), `numerical_mismatch` x2 |
| 2 | `subtract_mean` | PASS | 0 | 0 | None (passed first try) |
| 3 | `full_layer_norm` | PASS | 0 | 1 | `compilation_error` (free: missing rsqrt include) |

### Failure Details

**Stage 1 — `reduce_mean`**:
1. *Free retry*: Compilation error — `copy_tile_to_dst_init_short_with_dt` missing argument. Fixed by correcting helper call signature.
2. *Hard attempt 1*: Numerical mismatch (max diff 0.45). Mean broadcast strategy was incorrect.
3. *Hard attempt 2*: Numerical mismatch (max diff 4.97). Switched to `sub<COL>(tilized, mean) → centered`, then `sub<NONE>(tilized, centered) → mean_broadcast` which gives `x - (x - mean) = mean`.

**Stage 3 — `full_layer_norm`**:
1. *Free retry*: Compilation error — `rsqrt_tile_init` not declared. Fixed by adding `#include "api/compute/eltwise_unary/rsqrt.h"`.

---

## 5. Files Produced

### Operation (`ttnn/ttnn/operations/layer_norm_rm/`)
```
__init__.py                           # Re-exports layer_norm_rm()
layer_norm_rm.py                      # Entry point with input validation
layer_norm_rm_program_descriptor.py   # 13 CBs, work distribution, kernel setup
kernels/reader.cpp                    # RM stick reader (input + gamma/beta)
kernels/compute.cpp                   # 9-phase fused compute kernel
kernels/writer.cpp                    # RM stick writer
op_design.md                          # 28KB architecture + implementation design
.tdd_state.json                       # TDD pipeline state (all stages passed)
REPORT.md                             # This file
```

### Tests (`tests/ttnn/unit_tests/operations/layer_norm_rm/`)
```
__init__.py
layer_norm_rm.py                      # Import shim for stage tests
test_layer_norm_rm.py                 # 8 integration tests
test_stage_data_pipeline.py           # TDD stage 0
test_stage_reduce_mean.py             # TDD stage 1
test_stage_subtract_mean.py           # TDD stage 2
test_stage_full_layer_norm.py         # TDD stage 3
```

### Reference Analyses
```
ttnn/cpp/ttnn/operations/data_movement/tilize/device/tilize_multi_core_interleaved_analysis.md
ttnn/cpp/ttnn/operations/data_movement/untilize/device/factories/untilize_multi_core_analysis.md
ttnn/cpp/ttnn/operations/normalization/softmax/device/softmax_w_small_analysis.md
```

---

## 6. Git History

```
c4687b17b0 [ttnn-kernel-writer-tdd] stage full_layer_norm: passed
8c94dca969 [ttnn-kernel-writer-tdd] stage subtract_mean: passed
664d6345ab [ttnn-kernel-writer-tdd] stage reduce_mean: passed
c126c377e1 [ttnn-kernel-writer-tdd] stage data_pipeline: passed
f6ce7d18eb [ttnn-generic-op-builder] stubs: layer_norm_rm
ee5622d743 [ttnn-operation-architect] design: layer_norm_rm
1352fe24e7 [ttnn-operation-analyzer] analysis: softmax_general_w_small
0ca0225f6f [ttnn-operation-analyzer] analysis: tilize_multi_core_interleaved
8836ad7d7a [ttnn-operation-analyzer] analysis: untilize_multi_core
```

---

## 7. Decisions and Deviations

### Assumptions Made (Automated Mode)
- **Single-core initial implementation**: Designed for multi-core but TDD stages run single-core. Multi-core work distribution is in the program descriptor but not stress-tested.
- **Epsilon hardcoded in reader**: The reader uses `1e-5f` as a compile-time constant rather than passing it as a runtime arg. This matches the default but means changing epsilon requires recompilation.
- **Gamma/beta stick replication**: Reader copies one RM stick 32 times to fill a tile-height worth of data, then tilizes in compute. This is correct for (1,1,1,W) gamma/beta shape.

### Design Deviations
1. **Mean broadcast strategy** (Stage 1): Design suggested `copy_tile` for broadcasting mean across Wt tiles. TDD writer discovered this didn't work cleanly and instead used `sub(x, x-mean) = mean` — mathematically equivalent but uses existing helpers.
2. **Binary op B policy** (Stage 3): `WaitAndPopPerTile` for input B cannot be used when input A uses `NoWaitPopAtEnd`. Used `WaitUpfrontPopAtEnd` for B instead. This is a helper coupling constraint not documented in the design.
3. **rsqrt include**: The design referenced rsqrt but the stub kernel didn't include the necessary header. TDD writer added `#include "api/compute/eltwise_unary/rsqrt.h"`.

### Pain Points
- **Compilation errors as free retries**: The TDD system correctly classifies first-time compilation errors as free retries (not counting against the budget). This worked well — both compilation errors were trivially fixable.
- **Numerical debugging in reduce_mean**: The mean broadcast required 2 hard attempts. The multi-pass tile reuse pattern (keeping tiles in CBs via NoPop policies) is tricky to get right. The softmax reference was invaluable here.

---

## 8. Infrastructure Issues

### Breadcrumb Logging: NOT FUNCTIONAL
- `.claude/active_logging` was created as specified
- **However, no hooks directory or hook exists** to detect this file and inject logging instructions into subagent prompts
- Result: `agent_logs/` directory remains empty
- **Root cause**: The pipeline documentation references a hook mechanism that was never implemented
- **Impact**: No per-agent breadcrumb logs available for post-mortem analysis

### Device Access: No Issues
- No device hangs encountered during any TDD stage
- All 4 stages ran cleanly without needing `pkill` or `tt-smi -r`

### Build: No Issues
- Kernel compilation worked on all attempts (after fixing include/signature errors)
- No C++ build required (kernels compile at runtime)

### Python Environment: No Issues
- `tt-test.sh --dev` worked correctly throughout
- All pytest runs completed without environment errors

---

## 9. Suggestions for Improving the Agent Pipeline

### Critical
1. **Implement breadcrumb hooks**: The logging system described in the pipeline doc is not functional. Need a `.claude/hooks/` hook that reads `.claude/active_logging` and injects logging instructions into subagent prompts. Without this, there's no observability into agent decision-making.

### High Priority
2. **Document helper coupling constraints**: The binary op B-policy coupling (WaitAndPopPerTile incompatible with non-per-tile A policies) should be documented in the design template or helper reference. This caused a wasted attempt.
3. **Include headers in stub kernels**: When the architect specifies helpers like `rsqrt_tile`, the builder should generate stub kernels with the correct includes already in place, avoiding free-retry compilation errors.

### Medium Priority
4. **Mean broadcast pattern library**: The reduce-then-broadcast pattern needed for mean/variance is common enough to deserve a documented recipe. The TDD writer had to discover the `sub(x, x-mean) = mean` trick through trial and error.
5. **Multi-core stress testing**: Add a TDD stage that specifically tests multi-core distribution (e.g., tensor with many rows forcing multiple cores). Currently all shapes fit on a single core.
6. **Epsilon as runtime arg**: Consider passing epsilon as a runtime arg rather than compile-time constant, so it can be changed without recompilation.

### Low Priority
7. **Builder-generated integration tests**: The 8 integration tests from the builder all pass with stubs (they just check shapes). Consider adding a "smoke test" that checks numerical output against a reference after all TDD stages pass.
8. **Stage test deduplication**: The stage test files share significant boilerplate. A test generator or shared fixture module would reduce maintenance burden.
