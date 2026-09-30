# mhc_pre_ttnn (ttnn.bringup.mhc_pre)

Source: ai-generated (tt_ops_code_gen eval run 1045, branch `2026_09_30_0115_run1_mhc_pre` @ `963102ebe2f`, started from
`mstaletovic/mhc-codegen` @ `5066bfadf95`): the op folder `ttnn/ttnn/operations/mhc_pre` and its unit suite
`tests/ttnn/unit_tests/operations/mhc_pre`. Left out: the run's `perf_experiments/` scripts, `agent_logs/` and the
agents' scratch probes (all on the source branch). The op's own history is in `changelog.md` (design, refinements,
two perf-tournament rounds); its golden suite is `eval/golden_tests/mhc_pre` on tt_ops_code_gen `mstaletovic/mhc-goldens`.

1. ai-generated perf optimized version (Python ProgramDescriptor host side), moved to ttnn/ttnn/bringup/mhc_pre_ttnn (the folder is not
   named mhc_pre: a package of the op's own name would shadow ttnn.bringup.mhc_pre): imports `ttnn.operations.mhc_pre` ->
   `ttnn.bringup.mhc_pre_ttnn`; `kernel_lib/perf_instrumentation.hpp` (on the source branch, not
   in this tree) ships as `kernels/perf_instrumentation.hpp`, as rms_norm_ttnn's does. Kernels otherwise unchanged.
   Unit suite in `tests/unit/`: passes on this tree's kernel helper library.
   - Needed by: glm53_flash_d_p (mHC: attn / ffn hc + collapse, 4 streams, C 4096).
