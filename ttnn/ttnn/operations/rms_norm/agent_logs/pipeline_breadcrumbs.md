# RMS Norm Pipeline Breadcrumbs

## Pipeline Configuration
- **Operation**: rms_norm
- **Mode**: FULLY AUTOMATED
- **Start time**: 2026-03-16
- **Op path**: ttnn/ttnn/operations/rms_norm/
- **Test path**: tests/ttnn/unit_tests/operations/rms_norm/

## Phase 0: Discovery
- **Status**: COMPLETE
- **References**:
  - input_stage: tilize_single_core_program_factory.cpp
  - output_stage: untilize_single_core_program_factory.cpp
  - compute_core: batch_norm_program_factory.cpp
- **Mode**: Hybrid (3 references with different roles)

## Phase 1: Analysis
- **Status**: IN PROGRESS
- Three analyzers launched in parallel
  - tilize-analyzer (input_stage)
  - untilize-analyzer (output_stage)
  - batchnorm-analyzer (compute_core)
