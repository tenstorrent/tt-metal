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
2. C++ host side: `ttnn.bringup.mhc_pre` is now a C++ TTNN op (the Python op stays importable as
   `ttnn.bringup.mhc_pre_ttnn.mhc_pre`, the parity reference). Kernels untouched.
   - What: `device/mhc_pre_ttnn_program_factory.cpp` is a line-for-line port of `make_plan()` (group-width cost
     model, L1 fit, depth fallback, group growth) and `create_program_descriptor()` (CB table incl. aliases, W column
     all-gather Mcast1D, per-group Mcast2D, NoC-flip placement sets, CT / RT args, fp32 unpack modes,
     `MHC_PRE_KERNEL_DEFINES`); a ProgramDescriptor factory (build on a cache miss, `override_runtime_arguments`
     patches the address slots of every reader / writer set on a hit); `MhcPreDeviceOperation` (outputs y, post,
     comb) + `ttnn::prim::bringup::mhc_pre_ttnn` (program hash = the input specs, n, the scalars, the compute config
     fields and the defines env); `mhc_pre_ttnn.cpp` keeps `validate()`'s checks, order and exception types;
     `_mhc_pre_ttnn_program_descriptor` for the test. The scalars are Python-style doubles rounded to fp32 once.
   - Why: at GLM-5.3's shape (T 5120, n*C 16384, bf16 X and W) the Python op was host-bound: host 2.037 -> 0.043 ms
     per call, wall 2.067 -> 0.597 ms per call.
   - Tests: `tests/unit/test_mhc_pre_cpp_parity.py` (17 program-descriptor parity cases over every dtype pair, narrow
     and tall groups, decode, ragged T, n 1..4, ranks 2..4, GLM's shape; 6 bit-identical device outputs incl. a
     program-cache hit); the unit suite and the eval golden suite (`tests/golden/`, from tt_ops_code_gen
     `mstaletovic/mhc-goldens`, run on the C++ op) pass.
   - Needed by: glm53_flash_d_p.
