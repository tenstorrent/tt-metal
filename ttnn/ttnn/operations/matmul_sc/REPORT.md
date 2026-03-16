# REPORT: matmul_sc

## Summary

`matmul_sc` is a single-core tiled matrix multiplication (`C = A × B`) implemented using the new
`matmul_1d` LLK helper library. This is the Phase 2 validation operation from the
`matmul_1d_helper_task.md` — its primary purpose is confirming that the helpers are AI-usable
end-to-end with no prior context.

**Result: ALL STAGES PASSED**

---

## Pipeline Execution

| Phase | Agent | Output | Result |
|-------|-------|--------|--------|
| 0 | Orchestrator | Reference discovery | matmul_multicore (compute_core) |
| 1 | ttnn-operation-analyzer | `matmul_multicore_analysis.md` | Complete |
| 2 | ttnn-operation-architect | `op_design.md`, `.tdd_state.json` (2 stages) | Complete |
| 3 | ttnn-generic-op-builder | Python infra + stubs + tests (5 integration tests pass) | Complete |
| 4 | ttnn-kernel-writer-tdd | Both TDD stages passing | Complete |

---

## TDD Pipeline Results

| Stage | Name | Result | Hard Retries | Free Retries | Notes |
|-------|------|--------|-------------|-------------|-------|
| 1 | data_pipeline | PASS | 0 | 3 | Identity copy: reader A→cb_in0, compute copy, writer via helper |
| 2 | matmul_compute | PASS | 0 | 2 | Full `matmul_1d` + `read_matmul_tiles` + manual writer |

---

## Files Produced

```
ttnn/ttnn/operations/matmul_sc/
├── __init__.py
├── matmul_sc.py                        # Entry point, validation, ttnn.generic_op call
├── matmul_sc_program_descriptor.py     # CBs, kernel setup, TensorAccessorArgs
├── kernels/
│   ├── matmul_sc_reader.cpp            # Uses read_matmul_tiles helper
│   ├── matmul_sc_compute.cpp           # Uses matmul_1d helper
│   └── matmul_sc_writer.cpp            # Manual writer (equivalent to write_matmul_tiles)
├── op_design.md
├── .tdd_state.json
├── matmul_multicore_analysis.md
└── agent_logs/

tests/ttnn/unit_tests/operations/matmul_sc/
├── __init__.py
├── test_matmul_sc.py                   # Integration test (shape/validation)
├── test_stage_data_pipeline.py         # Stage 1 test
└── test_stage_matmul_compute.py        # Stage 2 test
```

---

## Git History

```
f74803d  [ttnn-kernel-writer-tdd] stage matmul_compute: passed
3d30b0f  [ttnn-kernel-writer-tdd] stage data_pipeline: passed
67bfe5c  [ttnn-generic-op-builder] stubs: matmul_sc
c947a81  [ttnn-operation-architect] design: matmul_sc
b1063b3  [ttnn-operation-analyzer] analysis: matmul_multicore
```

---

## Decisions and Deviations

### CB indices removed from positional compile-time args
The architect specified CB indices (cb_in0=0, cb_in1=1, cb_out=16) in named compile-time args
AND positional compile-time args. This conflicted with `TensorAccessorArgs<0>()` which must start
at positional index 0. Fix: CB indices are named-only; TensorAccessorArgs start at positional 0.

### Writer uses manual code instead of `write_matmul_tiles` helper
Including `matmul_1d_dataflow_helpers.hpp` in the writer kernel caused eager constexpr evaluation
of `TensorAccessorArgs<2>` from the reader's chained args pattern, which fails because the writer
only has 2 compile-time args. The writer was implemented manually with identical logic.

**This is a bug in `matmul_1d_dataflow_helpers.hpp`** — the reader and writer helpers should be
in separate headers to avoid this cross-contamination. Recommended fix: split into
`matmul_1d_reader_helpers.hpp` and `matmul_1d_writer_helpers.hpp`.

### `cb_helpers.hpp` missing from `matmul_1d_helpers.inl`
The inl file uses `get_cb_num_pages()` but doesn't include `cb_helpers.hpp`. The compute kernel
works around this by adding the include explicitly. Recommended fix: add the include to the .inl.

### `fp32_dest_acc_en=True` added
Not in op_design.md, but required for K=256 shapes to meet rtol=0.05/atol=0.2 tolerances.
bf16 accumulation over 8 tiles degrades precision below the test threshold.

---

## Issues Found in Helper Library (Actionable Fixes)

| Issue | File | Recommended Fix |
|-------|------|-----------------|
| Reader + writer helpers in same header cause TensorAccessorArgs eval conflict | `matmul_1d_dataflow_helpers.hpp` | Split into `matmul_1d_reader_helpers.hpp` and `matmul_1d_writer_helpers.hpp` |
| Missing `cb_helpers.hpp` include | `matmul_1d_helpers.inl` | Add `#include "api/compute/cb_helpers.hpp"` |
| `fp32_dest_acc_en` not mentioned in `matmul_1d_reference.md` | `docs/matmul_1d_reference.md` | Add note: enable for K > 4 tiles to maintain accuracy |
