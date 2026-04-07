# Layernorm Pipeline Report

## Scope

- Operation path: `ttnn/ttnn/operations/layernorm`
- Evidence used: current workspace files only
- Excluded by request: git history, `gh` history, web search
- Timeline note: `agent_logs/` is absent, so phase timing below is reconstructed from file modification times and should be treated as approximate.

## Phase Timeline

| Phase | Approx. UTC window | Primary evidence | Outcome |
| --- | --- | --- | --- |
| Discovery | 15:39 | `references/discovery.md` | Completed. Discovery locked the target to a real host/device `layer_norm` op and selected the batch-norm and moreh-norm reference set. |
| Analysis | 15:46-15:55 | `references/*_analysis.md` | Completed. Host API, device launch, program-factory, binding, module-registration, and build-integration reference analyses were emitted. |
| Architecture | 16:06 | `architecture.md`, `design_journal.jsonl` | Completed. The architecture established a row-local, last-dimension, same-shape normalization design with optional residual/affine inputs. |
| Engineering | 16:33-16:35 | `op_design.md`, `engineer_journal.jsonl` | Completed. Engineering resolved architecture/design conflicts, fixed the kernel/CB contract, and defined the staged TDD plan. |
| Build | 16:52-17:27 | generated tests, `layer_norm.py`, `layer_norm_program_descriptor.py`, device kernels | Partially completed relative to the original op goal. The run produced a Python `generic_op` scaffold, stage tests, and device kernels, but not the planned C++ host/device wrapper and nanobind integration. |
| TDD | final state recorded at 17:34 | `.tdd_state.json` | Historically completed in the saved state: all six stages are marked `passed`. Current rerun validation exposes a post-TDD handoff bug in the scaffold wrapper. |

## Phase Outputs

### Discovery

- `references/discovery.md` captured the requirement that this must be a real normalization device op rather than a composite wrapper and selected `batch_norm` plus `moreh_norm` as the primary references. See `design_journal.jsonl:1-12`.

### Analysis

- The analysis phase emitted:
  - `references/host_api_surface_analysis.md`
  - `references/device_validation_and_launch_analysis.md`
  - `references/device_program_factory_analysis.md`
  - `references/last_dim_reduction_pattern_analysis.md`
  - `references/operation_binding_analysis.md`
  - `references/module_registration_analysis.md`
  - `references/build_integration_analysis.md`
- The resulting decisions included row-unit work splitting, tail masking, `epsilon` keyword compatibility, and normalization-module binding/build integration requirements. See `design_journal.jsonl:5-12`.

### Architecture

- `architecture.md` defined the intended deliverable as a real C++ host/device op under `ttnn/cpp/ttnn/operations/normalization/layernorm`, with dedicated program factory, kernels, normalization CMake integration, and nanobind registration. See `op_design.md:7-11` and `op_design.md:26-40`.
- Key architectural choices recorded in `design_journal.jsonl`:
  - real fused normalization/reduction op, not a composite fallback (`design_journal.jsonl:1`)
  - public `ttnn::layer_norm` plus `ttnn::prim::layer_norm` split (`design_journal.jsonl:2`)
  - row-local last-dimension scheduling with `split_work_to_cores` (`design_journal.jsonl:5`)
  - reader-owned dense residual/affine streams and compute-owned normalization math (`design_journal.jsonl:8`)
  - Python-facing `epsilon`, `residual_input_tensor`, and `program_config` compatibility surface (`design_journal.jsonl:9`)

### Engineering

- `engineer_journal.jsonl` records the main design corrections:
  - nanobind build wiring lives in both `ttnn/cpp/ttnn/operations/normalization/CMakeLists.txt` and `ttnn/CMakeLists.txt`, not just the normalization runtime CMake (`engineer_journal.jsonl:11`)
  - the workspace requires both `device/layernorm_types.hpp` and `device/layernorm_common.hpp` (`engineer_journal.jsonl:12`)
  - the baseline should use row-local `mean` and `mean(x^2)` accumulation instead of cross-core Welford combine (`engineer_journal.jsonl:13`)
  - `.tdd_state.json["op_name"]` must be `layer_norm`, not the directory name `layernorm` (`engineer_journal.jsonl:14`)
  - program-cache identity must include padded-shape-sensitive fields (`engineer_journal.jsonl:15`)
- `op_design.md` turned those corrections into a concrete CB map, runtime-arg map, compute-phase plan, and six-stage TDD plan. See `op_design.md:75-154`.

### Build

- Actual generated/implemented artifacts in the workspace:
  - Python scaffold: `layer_norm.py`
  - program descriptor builder: `layer_norm_program_descriptor.py`
  - device kernels: `ttnn/cpp/ttnn/operations/normalization/layernorm/device/kernels/dataflow/reader_layernorm.cpp`, `writer_layernorm.cpp`, `compute/layernorm_kernel.cpp`, `compute/layernorm_sfpu_kernel.cpp`, `compute/layernorm_compute_common.hpp`
  - generated tests under `tests/ttnn/unit_tests/operations/layer_norm/`
- Deviation from the planned source layout:
  - `op_design.md` planned C++ host/device runtime files and nanobind files such as `layernorm.cpp`, `layernorm_device_operation.cpp`, `layernorm_program_factory.cpp`, and `layernorm_nanobind.cpp` (`op_design.md:26-40`).
  - The current normalization runtime CMake still only packages and compiles `batch_norm` sources (`ttnn/cpp/ttnn/operations/normalization/CMakeLists.txt:14-35`).
  - The top-level nanobind source list still includes only `batch_norm_nanobind.cpp` and `normalization_nanobind.cpp` for normalization (`ttnn/CMakeLists.txt:367-368`).
  - The normalization registrar still binds only batch norm (`ttnn/cpp/ttnn/operations/normalization/normalization_nanobind.cpp:9-13`).
- Result: the build phase produced a working `generic_op` scaffold path for staged testing, but it did not finish the originally planned C++ normalization runtime integration.

### TDD

- `.tdd_state.json` shows six stages, all marked `passed`, with `current_stage_index` advanced to `6` to indicate completion (`.tdd_state.json:6-8`).

| Stage | Saved status | Notable blocker(s) recorded in `.tdd_state.json` |
| --- | --- | --- |
| `data_pipeline` | passed | numerical mismatch, then hang (`.tdd_state.json:46-80`) |
| `mean_reduce` | passed | repeated numerical mismatch while tightening row-mean behavior (`.tdd_state.json:118-163`) |
| `invstd_reduce` | passed | numerical mismatch in variance/rsqrt path (`.tdd_state.json:201-216`) |
| `normalize` | passed | numerical mismatch after restoring full-shape output (`.tdd_state.json:254-269`) |
| `residual_affine` | passed | reader-kernel compilation error around `noc_async_read_tile_helper(...)` (`.tdd_state.json:308-319`) |
| `acceptance` | passed | no recorded failure history in the final saved state |

## Build And Test Outcomes

### Historical pipeline outcome from workspace state

- The saved TDD state indicates the staged run completed successfully: all registered stages are `passed` and no stage is left pending (`.tdd_state.json:41`, `.tdd_state.json:113`, `.tdd_state.json:196`, `.tdd_state.json:249`, `.tdd_state.json:303`).

### Current validation run on 2026-04-07 UTC

| Command | Outcome | Notes |
| --- | --- | --- |
| `./build_metal.sh --release` | passed | Exit code `0`. This confirms the current workspace still builds in release mode, but it does not prove that the intended normalization C++ runtime integration landed because the normalization build surfaces still only reference `batch_norm`. |
| `scripts/tt-test.sh --dev tests/ttnn/unit_tests/operations/layer_norm` | failed | Exit code `1`. The first test fails before kernel execution because `layer_norm.py` reads `current_stage_index = 6` and then indexes `state["stages"][6]`, which is out of range for the six-entry stage list. See `layer_norm.py:89-94`, `.tdd_state.json:6-8`, and `tests/ttnn/unit_tests/operations/layer_norm/test_layer_norm.py:18-31`. |

## Key Decisions, Deviations, And Resolved Blockers

### Key decisions

- Keep the op as a real device normalization op and not a composite fallback (`design_journal.jsonl:1-3`).
- Use row-local last-dimension scheduling so no logical row is split across cores (`design_journal.jsonl:5`; `op_design.md:84-93`).
- Preserve caller compatibility with `epsilon`, `residual_input_tensor`, and `program_config` on the public surface (`design_journal.jsonl:9`).
- Engineer the baseline around two-pass reread plus row-local `mean` / `mean(x^2)` accumulation rather than preloading whole rows or merging Welford partials across cores (`engineer_journal.jsonl:13`; `op_design.md:128-154`).

### Deviations from the original target

- The original target was full normalization-module integration in C++ (`op_design.md:7-11`, `op_design.md:26-40`).
- The delivered workspace artifacts are a staged Python scaffold plus kernels and tests, without the planned C++ host/device runtime files or nanobind integration.
- The release build passes despite that gap because the build surfaces still compile/package only `batch_norm` in the normalization module (`ttnn/cpp/ttnn/operations/normalization/CMakeLists.txt:14-35`) and only bind `batch_norm` in normalization nanobind (`ttnn/CMakeLists.txt:367-368`, `ttnn/cpp/ttnn/operations/normalization/normalization_nanobind.cpp:9-13`).

### Blockers resolved during the run

- Reader/compute/writer pipeline correctness issues were resolved after an initial catastrophic numerical mismatch and a hang in `data_pipeline`.
- Temporary reduced-output stages were stabilized after repeated numerical mismatches in `mean_reduce` and `invstd_reduce`.
- The residual/affine stage resolved a reader-kernel compile error involving `noc_async_read_tile_helper(...)`.
- The `op_name` naming mismatch between directory name and public symbol was explicitly corrected in the TDD state design.

## Remaining Risks And Assumptions

### Assumptions

- Phase timing is inferred from file timestamps because `ttnn/ttnn/operations/layernorm/agent_logs/` does not exist in the workspace.
- Historical stage pass/fail conclusions come from `.tdd_state.json`, not from replayable per-attempt logs.

### Remaining risks

- The current workspace is not in a clean post-TDD handoff state for reruns because the scaffold wrapper cannot handle `current_stage_index == len(stages)` (`layer_norm.py:89-94`, `.tdd_state.json:6-8`).
- The implementation still falls short of the original normalization-module C++ integration target. That gap is architectural, not just a reporting artifact.
