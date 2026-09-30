# mhc_post_ttnn (ttnn.bringup.mhc_post)

Source: ai-generated (tt_ops_code_gen eval run 1046, branch `2026_09_30_0115_run1_mhc_post` @ `84be38c18b7`, started from
`mstaletovic/mhc-codegen` @ `5066bfadf95`): the op folder `ttnn/ttnn/operations/mhc_post` and its unit suite
`tests/ttnn/unit_tests/operations/mhc_post`. Left out: the run's `perf_experiments/` scripts, `agent_logs/` and the
agents' scratch probes (all on the source branch). The op's own history is in `changelog.md` (design, refinements,
two perf-tournament rounds); its golden suite is `eval/golden_tests/mhc_post` on tt_ops_code_gen `mstaletovic/mhc-goldens`.

1. ai-generated perf optimized version (Python ProgramDescriptor host side), moved to ttnn/ttnn/bringup/mhc_post_ttnn (the folder is not
   named mhc_post: a package of the op's own name would shadow ttnn.bringup.mhc_post): imports `ttnn.operations.mhc_post` ->
   `ttnn.bringup.mhc_post_ttnn`; `kernel_lib/perf_instrumentation.hpp` (on the source branch, not
   in this tree) ships as `kernels/perf_instrumentation.hpp`, as rms_norm_ttnn's does. Kernels otherwise unchanged.
   Unit suite in `tests/unit/`: passes on this tree's kernel helper library.
   - Needed by: glm53_flash_d_p (mHC: attn / ffn residual, 4 streams, C 4096).
2. C++ host side: `ttnn.bringup.mhc_post` is now a C++ TTNN op (the Python op stays importable as
   `ttnn.bringup.mhc_post_ttnn.mhc_post`, the parity reference). Kernels untouched.
   - What: `device/mhc_post_ttnn_program_factory.cpp` is a line-for-line port of `create_program_descriptor()`
     (row-weighted work split, L1 block solve, read help, CBs, CT / RT args, NoC modes, semaphores); a
     ProgramDescriptor factory (build on a cache miss, `override_runtime_arguments` patches the five buffer-address
     slots on a hit); `MhcPostDeviceOperation` + `ttnn::prim::bringup::mhc_post_ttnn` (program hash = the four
     tensor specs + the compute-config fields the builder reads); `mhc_post_ttnn.cpp` keeps `validate()`'s checks,
     order and exception types (UnsupportedAxisValue, ValueError); `_mhc_post_ttnn_program_descriptor` for the test.
   - Why: a Python ProgramDescriptor op pays its host build on every call. At GLM-5.3's shape (T 5120, C 4096,
     n 4, bf16) the host time per call went 0.676 -> 0.019 ms; device-bound wall 1.195 -> 1.161 ms per call.
   - Tests: `tests/unit/test_mhc_post_cpp_parity.py` (15 program-descriptor parity cases incl. GLM's shape, 5
     bit-identical device outputs incl. a program-cache hit on moved buffers); the unit suite runs on the C++ op
     (the knob tests stay on the Python builder, the only one with the knobs): 110 passed.
   - Needed by: glm53_flash_d_p.
3. The eval golden suite ships in `tests/golden/` (from tt_ops_code_gen `mstaletovic/mhc-goldens`; imports pointed at
   this folder, dispatching the C++ op): passes.
4. Model cases (bringup-fork-tests, task O.1): `tests/cases.py` (1 glm53_flash_d_p call(s), captured with
   GLM_MHC_IMPL=fused on the s56320 rung), `tests/reference.py` (float64 torch semantics), `tests/test_mhc_post_ttnn.py`
   (random per-chip inputs on the 2x2 mesh, PCC + rel L2 per output; limits from the measured error with margin).
5. Opt-in `comb_transposed=False` (default True keeps the op's comb^T math, bit-identical program: no define added).
   - What: `X'_j = post_j F + sum_i comb[j*n + i] X_i` (comb applied as stored, row-major comb[j][i] for output j).
     Only the coefficient expansion changes: `kernels/mhc_post_coef_expand.hpp` reads raw column `j*n + (t-1)`
     under the kernel define `MHC_POST_COMB_DIRECT` (added to both DM kernels by the program factory when the option
     is off). `MhcPostParams::comb_transposed` is in the program hash; `mhc_post` / the prim / the program descriptor
     / the nanobind (`comb_transposed=True` kw-only, also on `_mhc_post_ttnn_program_descriptor`) carry it. The
     Python builder (parity reference) has no such option; the parity suite only builds the default.
   - Why: Xing4.0's mHC residual is `post * y + comb @ x` (`xing40_a4b_d_p/reference/xing_ref.py:hc_residual`), not
     GLM's comb^T. Replaces the 20-op addcmul chain (90 ms per 5120-token chunk each for attn / ffn residual).
   - Files: kernels/mhc_post_coef_expand.hpp, kernels/mhc_post_common.hpp (comment), device/*_types.hpp,
     device/mhc_post_ttnn_device_operation.{hpp,cpp}, device/mhc_post_ttnn_program_factory.{hpp,cpp},
     mhc_post_ttnn.{hpp,cpp}, mhc_post_ttnn_nanobind.cpp; tests: cases.py (xing40 case, comb_transposed False),
     reference.py (comb_transposed), test_mhc_post_ttnn.py (passes the option; checks the comb and comb^T
     references differ; the mesh fixture opens the case mesh matching the box's device count, since a 2x2 submesh of
     a 4x2 box fails the 2D fabric handshake).
   - Needed by: xing40_a4b_d_p P.2 (tt/residual.py, XING_RESIDUAL_MIX=fused, the default).
