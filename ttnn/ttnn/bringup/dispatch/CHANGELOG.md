# dispatch (fork)

- Source: `ttnn/cpp/ttnn/operations/experimental/deepseek_prefill/dispatch`
- Source SHA: `67ca5f3af4860c04e7ce31e345107f51cb22aa9b`
- Python: `ttnn.bringup.*` (was `ttnn.experimental.deepseek_prefill.*`)
- Forked for: ernie45_d_p P3.1
- Used by: ernie45_d_p, gemma4_a4b_d_p, mimo_v2_6_d_p

Mechanical fork changes (fork_op.py): namespace `ttnn::operations::bringup`, CMake target `ttnn_op_bringup_dispatch`, kernel paths and includes pointing at this folder, Python prefix `ttnn.bringup.`.

## Changes

### Support a 1-device dispatch axis (no fabric)
- What: all fabric setup is gated on `use_fabric = num_links > 0 && dispatch-axis devices > 1`; otherwise the
  kernels' existing no-fabric build (`#ifdef DEST_CHIP_ID`) is used. No change for dispatch axes with more than one
  device.
- Why: with one chip per dispatch group every routed expert is local or absent, so no token crosses the fabric, but
  the host factory still looked up fabric neighbours (TT_FATAL "No neighbors found").
- Needed by: ernie45_d_p P3.1 (originally commit e30d505c4ae)
- Files: `device/dispatch_program_factory.cpp`

### Tests: fork test suite with a first model case
- What: `tests/` (cases.py, reference.py, test file) with a random-input case for the call mimo_v2_6_d_p makes
  (bringup-fork-tests skill). No op change.
- Why: task O.1, every call a model makes to a fork gets a case.
- Needed by: mimo_v2_6_d_p O.1
- Files: `tests/__init__.py`, `tests/cases.py`, `tests/reference.py`, `tests/test_*.py`

### Tests: gemma4_a4b_d_p case
- What: appended the random-input case for the call gemma4_a4b_d_p makes (1x4 mesh, S 5120, H 2816, 128 experts
  top-8, 32 per chip). No op change.
- Why: task O.1, every call a model makes to a fork gets a case.
- Needed by: gemma4_a4b_d_p O.1
- Files: `tests/cases.py`

### Tests: ernie45_d_p case
- What: appended the random-input case for the call ernie45_d_p makes (1x4 mesh, FABRIC_1D_RING, S 5120, H 2560,
  64 experts top-6, 16 per chip). No op change, no test-file change.
- Why: task O.1, every call a model makes to a fork gets a case.
- Needed by: ernie45_d_p O.1
- Files: `tests/cases.py`

### Tests: hy4_preview_d_p case
- What: appended the random-input case for the call hy4_preview_d_p makes (2x2 mesh, dispatch groups of 2 chips
  along axis 0, S 2560 per chip, H 6144, 256 experts top-8, 64 per chip). No op change.
- Why: task O.1, every call a model makes to a fork gets a case.
- Needed by: hy4_preview_d_p O.1
- Files: `tests/cases.py`

### Opt-in dispatch group along mesh axis 1 (`allow_cluster_axis_1`)
- What: new keyword `allow_cluster_axis_1` (default False). When True, the host check accepts `cluster_axis=1` as
  well as 0. The program factory and kernels already handle both axes (`AXIS` define, `ReplicateGroup::ROWS`, the
  FABRIC_2D same-group peer scan), so only `dispatch.cpp`'s TT_FATAL changes. Option off: the same refusal, and the
  same program for every existing call.
- Why: under CP=4 on a 1x4 mesh each chip dispatches its own S/4 rows over all 4 chips: one dispatch group along
  axis 1. Axis 0 has one device there, and the source's check rejected axis 1 (TT_FATAL "cluster_axis must be 0").
  With the option on, MiMo layer-1 experts score PCC 0.99998 and rel L2 0.0063 (component gate).
- Needed by: mimo_v2_6_d_p_cp4 C.sliding_moe.experts
- Files: `dispatch.cpp`, `dispatch.hpp`, `dispatch_nanobind.cpp`, `tests/unit/test_dispatch_cluster_axis_1.py`
  (1x4 FABRIC_2D, a group of 4 on axis 1, every received buffer / metadata row exact, at S 256 / H 1024 and
  S 1280 / H 4096; also checks that the default still refuses axis 1)
- Regression with the option off: the model cases in `tests/test_dispatch.py` 6/6 pass; `fork_source`: 82 tests,
  0 regressions vs the baseline.
