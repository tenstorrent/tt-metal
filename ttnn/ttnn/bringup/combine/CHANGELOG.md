# combine (fork)

- Source: `ttnn/cpp/ttnn/operations/experimental/deepseek_prefill/combine`
- Source SHA: `67ca5f3af4860c04e7ce31e345107f51cb22aa9b`
- Python: `ttnn.bringup.*` (was `ttnn.experimental.deepseek_prefill.*`)
- Forked for: ernie45_d_p P3.1
- Used by: ernie45_d_p, gemma4_a4b_d_p, mimo_v2_6_d_p

Mechanical fork changes (fork_op.py): namespace `ttnn::operations::bringup`, CMake target `ttnn_op_bringup_combine`, kernel paths and includes pointing at this folder, Python prefix `ttnn.bringup.`.

## Changes

### Support a 1-device dispatch axis (no fabric)
- What: all fabric setup is gated on `use_fabric = num_links > 0 && dispatch-axis devices > 1`; otherwise the
  kernels' existing no-fabric build (`#ifdef DEST_CHIP_ID`) is used. No change for dispatch axes with more than one
  device.
- Why: with one chip per dispatch group every routed expert is local or absent, so no token crosses the fabric, but
  the host factory still looked up fabric neighbours (TT_FATAL "No neighbors found").
- Needed by: ernie45_d_p P3.1 (originally commit e30d505c4ae)
- Files: `device/combine_program_factory.cpp`

### Tests: fork test suite with a first model case
- What: `tests/` (cases.py, reference.py, test file) with a random-input case for the call mimo_v2_6_d_p makes
  (bringup-fork-tests skill). No op change.
- Why: task O.1, every call a model makes to a fork gets a case.
- Needed by: mimo_v2_6_d_p O.1
- Files: `tests/__init__.py`, `tests/cases.py`, `tests/reference.py`, `tests/test_*.py`

### Tests: gemma4_a4b_d_p case (BFLOAT8_B buffer)
- What: appended the random-input case for the call gemma4_a4b_d_p makes (1x4 mesh, S 5120, H 2816, 128 experts
  top-8, 32 per chip, BFLOAT8_B TILE expert-output buffer). The test now compares a non-bf16 buffer against its own
  device-held rounded values (read back from the input tensor); combine unpacks bfp8 to bf16 losslessly, so the check
  stays exact. bf16 cases are unchanged. No op change.
- Why: task O.1, every call a model makes to a fork gets a case.
- Needed by: gemma4_a4b_d_p O.1
- Files: `tests/cases.py`, `tests/test_combine.py`

### Tests: ernie45_d_p case (BFLOAT8_B buffer)
- What: appended the random-input case for the call ernie45_d_p makes (1x4 mesh, FABRIC_1D_RING, S 5120, H 2560,
  64 experts top-6, 16 per chip, BFLOAT8_B TILE expert-output buffer). No op change, no test-file change.
- Why: task O.1, every call a model makes to a fork gets a case.
- Needed by: ernie45_d_p O.1
- Files: `tests/cases.py`

### Tests: hy4_preview_d_p case
- What: appended the random-input case for the call hy4_preview_d_p makes (2x2 mesh, dispatch groups of 2 chips
  along axis 0, S 2560 per chip, H 6144, bf16 expert-output buffer, 256 experts top-8, 64 per chip). No op change.
- Why: task O.1, every call a model makes to a fork gets a case.
- Needed by: hy4_preview_d_p O.1
- Files: `tests/cases.py`

### Opt-in combine group along mesh axis 1 (`allow_cluster_axis_1`)
- What: new keyword `allow_cluster_axis_1` (default False). When True, the host check accepts `cluster_axis=1` as
  well as 0. The program factory and kernels already handle both axes, so only `combine.cpp`'s TT_FATAL changes.
  Option off: the same refusal, and the same program for every existing call.
- Why: the combine half of the dispatch fork's change of the same name (CP=4 on 1x4: one group of 4 chips on axis 1).
- Needed by: mimo_v2_6_d_p_cp4 C.sliding_moe.experts
- Files: `combine.cpp`, `combine.hpp`, `combine_nanobind.cpp`, `tests/unit/test_combine_cluster_axis_1.py`
  (1x4 FABRIC_2D, every (token, slot) of every chip exact, TILE bf16 buffer, at S 256 / H 1024 and S 1280 / H 4096;
  also checks that the default still refuses axis 1)
- Regression with the option off: the model cases in `tests/test_combine.py` 6/6 pass; `fork_source`: 117 tests, 0 regressions vs the baseline.

### Tests: mimo_v2_6_d_p_cp4 case (combine group along axis 1)
- What: appended the random-input case for the call mimo_v2_6_d_p_cp4 makes (1x4 mesh, one combine group of 4 chips
  on cluster_axis 1 with `allow_cluster_axis_1`, 2 links, S 1280 per chip, H 4096, bf16 buffer, 256 experts top-8),
  and `_combine_axis1` in `tests/test_combine.py` for cases with `cluster_axis == 1` (whole output exact).
  Other cases run unchanged. No op change.
- Why: task O.1, every call a model makes to a fork gets a case.
- Needed by: mimo_v2_6_d_p_cp4 O.1
- Files: `tests/cases.py`, `tests/test_combine.py`
