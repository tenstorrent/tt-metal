# Softmax Operation - Pipeline Report

## Summary

**Operation**: softmax
**Result**: SUCCESS - All 6 TDD stages passed
**Formula**: `softmax(x, dim)_i = exp(x_i - max(x, dim)) / sum(exp(x_j - max(x, dim)), dim)`
**Import**: `from ttnn.operations.softmax import softmax`
**Signature**: `softmax(input_tensor, dim=-1, *, numeric_stable=True) -> ttnn.Tensor`

Supports both `dim=-1` (width reduction) and `dim=-2` (height reduction), with optional numerically stable mode (max subtraction before exp).

---

## Pipeline Execution

| Phase | Agent | Duration | Output |
|-------|-------|----------|--------|
| 0 - Discovery | Orchestrator | ~1 min | Identified moreh_sum_w and moreh_sum_h as references |
| 1 - Analysis | ttnn-operation-analyzer (x2) | ~5 min (parallel) | moreh_sum_w_analysis.md, moreh_sum_h_analysis.md |
| 2 - Design | ttnn-operation-architect | ~6 min | op_design.md, .tdd_state.json |
| 3 - Build | ttnn-generic-op-builder | ~10 min | Python infra, stub kernels, 6 test files |
| 4 - TDD Kernels | ttnn-kernel-writer-tdd | ~21 min | All 3 kernels implemented, 6 stages passed |
| 5 - Report | Orchestrator | ~1 min | REPORT.md |

---

## Agent Summaries

### Phase 1: Analyzers (ttnn-operation-analyzer)
- **moreh_sum_w**: Analyzed W-dimension reduction pattern. Key findings: matmul-based reduction via scaler tile, DEST register management, CB layout for intermediates. Used `binary_op_init_common` initialization pattern.
- **moreh_sum_h**: Analyzed H-dimension reduction pattern. Key findings: tile reordering for column reduction, REDUCE_COL dispatch, chunk-based DEST processing.
- Both analyses saved to `agent_logs/`.

### Phase 2: Architect (ttnn-operation-architect)
- **Key design decision**: Unified compute pattern using REDUCE_ROW for dim=-1 and REDUCE_COL for dim=-2, parametrized by compile-time `is_dim_h` flag.
- **CB layout**: 6 circular buffers (input, scaler, output, max, exp, recip_sum).
- **Helper library mapping**: reduce_helpers for max/sum, binary_op_helpers for sub/mul with broadcast, copy_tile_helpers for passthrough/exp.
- Registered 6 TDD stages with progressive complexity.

### Phase 3: Builder (ttnn-generic-op-builder)
- Created Python entry point with full validation (dtype, layout, rank, dim, alignment).
- Created program descriptor with single-core execution, proper CB sizing, and kernel argument layout.
- Generated 6 test files with PyTorch reference implementations.
- Created stub kernels with correct includes.

### Phase 4: TDD Kernel Writer (ttnn-kernel-writer-tdd)
- Implemented all 3 kernels through 6 progressive stages.
- Total hard attempts across all stages: 7 (6 first-pass + 1 retry for softmax_h_stable).
- No device hangs encountered.
- Two upstream fixes applied during implementation (see below).

---

## TDD Pipeline Results

| Stage | Name | Result | Hard Attempts | Free Retries | Failure Classifications |
|-------|------|--------|---------------|--------------|------------------------|
| 0 | passthrough | PASS | 1 | 0 | None |
| 1 | exp_only | PASS | 1 | 0 | None |
| 2 | softmax_w_stable | PASS | 1 | 0 | None |
| 3 | softmax_w_unstable | PASS | 1 | 0 | None |
| 4 | softmax_h_stable | PASS | 2 | 0 | 1st attempt: missing is_dim_h CT arg |
| 5 | softmax_h_unstable | PASS | 1 | 0 | None |

**Total attempts**: 7/36 budget (19% utilization)

---

## Upstream Fixes During TDD

1. **Writer CB sync (Stage 1)**: Writer initially tried to wait for R tiles at once (`cb_wait_front(cb_out, R)`), but `cb_out` only has 2 pages (double-buffered). Fixed to per-tile wait/pop pattern matching the compute's `BinaryOutputPolicy::PerTile`.

2. **Missing is_dim_h CT arg for compute (Stage 5)**: Compute kernel needed awareness of reduction dimension to select between `REDUCE_ROW`/`REDUCE_COL` and `BroadcastDim::COL`/`BroadcastDim::ROW`. Added `is_dim_h` as compile-time arg index 9 and parametrized all dimension-dependent operations.

---

## Files Produced

```
ttnn/ttnn/operations/softmax/
├── __init__.py                           # Re-export
├── softmax.py                            # Entry point with validation
├── softmax_program_descriptor.py         # CB config, work distribution, kernel setup
├── kernels/
│   ├── softmax_reader.cpp                # Tile reader (contiguous + strided)
│   ├── softmax_compute.cpp               # 4-phase softmax compute
│   └── softmax_writer.cpp                # Tile writer (contiguous + strided)
├── op_design.md                          # Architecture + kernel design
├── .tdd_state.json                       # TDD pipeline state (all passed)
├── REPORT.md                             # This report
└── agent_logs/
    ├── moreh_sum_w_analysis.md
    ├── moreh_sum_h_analysis.md
    ├── ttnn-operation-analyzer_breadcrumbs.jsonl
    ├── ttnn-operation-architect_breadcrumbs.jsonl
    ├── ttnn-generic-op-builder_breadcrumbs.jsonl
    └── ttnn-generic-op-builder_execution_log.md

tests/ttnn/unit_tests/operations/softmax/
├── __init__.py
├── softmax.py                            # Re-export for test imports
├── test_softmax.py                       # Integration test
├── test_stage_passthrough.py
├── test_stage_exp_only.py
├── test_stage_softmax_w_stable.py
├── test_stage_softmax_w_unstable.py
├── test_stage_softmax_h_stable.py
└── test_stage_softmax_h_unstable.py
```

---

## Git History

```
085578db8c [ttnn-kernel-writer-tdd] stage softmax_h_unstable: passed
0656621e4c [ttnn-kernel-writer-tdd] stage softmax_h_stable: passed
46b9df85e9 [ttnn-kernel-writer-tdd] stage softmax_w_unstable: passed
4b84d4c7c8 [ttnn-kernel-writer-tdd] stage softmax_w_stable: passed
aaed5c5911 [ttnn-kernel-writer-tdd] stage exp_only: passed
5b51408bac [ttnn-kernel-writer-tdd] stage passthrough: passed
a72f5b56ca [ttnn-generic-op-builder] logs: softmax execution log and breadcrumbs
c5d7e22582 [ttnn-generic-op-builder] stubs: softmax
1ed398f2d4 [ttnn-operation-architect] finalize: softmax breadcrumbs
9d883c1d4f [ttnn-operation-architect] design: softmax
8ff6c282f4 [ttnn-operation-analyzer] update breadcrumbs for moreh_sum_h analysis
8757a610a2 [ttnn-operation-analyzer] analysis: moreh_sum_w
```

---

## Decisions and Deviations

### Key Design Decisions
1. **Unified compute kernel**: Both dim=-1 and dim=-2 share the same compute kernel, parametrized by `is_dim_h` compile-time flag. This avoids code duplication.
2. **Single-core execution**: Chose single-core for initial implementation simplicity. Multi-core can be added later via `split_work_to_cores`.
3. **Helper library usage**: Used `reduce_helpers_compute.hpp`, `binary_op_helpers.hpp`, and `copy_tile_helpers.hpp` extensively. This provides correct DST management, CB synchronization, and data format reconfiguration without manual tile register loops.
4. **Work unit = one virtual row**: Each work unit processes R tiles (Wt for dim=-1, Ht for dim=-2), treated uniformly as a "row" for reduction.

### Deviations from Original Spec
1. **Compute kernel syntax**: The kernel writer used `namespace NAMESPACE { void MAIN {} }` instead of `void kernel_main() {}`. Both work; the namespace form is the older convention but remains valid.
2. **Reader bulk reserve**: Reader reserves R tiles at once (`cb_reserve_back(cb_input, R)`) and pushes them in bulk, rather than per-tile reserve/push. This is more efficient for the WaitUpfrontNoPop pattern.

### Pain Points
1. **TDD orchestrator symlink bug**: `.claude` is a symlink to `tt_metal/third_party/tt-agents`, causing `Path(__file__).resolve()` to resolve through the symlink. REPO_ROOT was computed incorrectly, writing test files to the wrong location. Worked around by writing test files directly.
2. **Architect wrote state directly**: The architect wrote `.tdd_state.json` directly instead of using the orchestrator `add-stage` command, so test files weren't auto-generated. The builder handled this.

---

## Infrastructure Issues

- **No device hangs**: All 6 stages completed without device hangs.
- **No build failures**: Kernels compiled on first try for most stages (1 compilation issue on stage 5 due to missing CT arg, fixed immediately).
- **No venv problems**: Python environment worked correctly throughout.
- **Device access**: No contention issues with device access.

---

## Suggestions for Improving the Agent Pipeline

1. **Fix orchestrator REPO_ROOT**: The `.claude` symlink breaks `Path(__file__).resolve()`. Use `git rev-parse --show-toplevel` or an explicit `REPO_ROOT` env var.
2. **Architect should use orchestrator**: The architect should register stages via the orchestrator CLI rather than writing state JSON directly, to ensure test files are generated atomically.
3. **Multi-core from the start**: Future operations should consider multi-core execution in the initial design, as adding it later requires touching the program descriptor, reader, and writer.
4. **Compute kernel syntax**: Standardize on `void kernel_main() {}` across all generated kernels. The namespace form works but is inconsistent with the documentation.
5. **Builder should validate imports**: The builder should verify that test relative imports work (e.g., `from .softmax import softmax`) by checking for the re-export module.
