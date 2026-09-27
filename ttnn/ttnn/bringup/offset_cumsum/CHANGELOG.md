# offset_cumsum (fork)

- Source: `ttnn/cpp/ttnn/operations/experimental/deepseek_prefill/offset_cumsum`
- Source SHA: `67ca5f3af4860c04e7ce31e345107f51cb22aa9b`
- Python: `ttnn.bringup.*` (was `ttnn.experimental.deepseek_prefill.*`)
- Forked for: ernie45_d_p P3.1
- Used by: ernie45_d_p, gemma4_a4b_d_p, mimo_v2_6_d_p

Mechanical fork changes (fork_op.py): namespace `ttnn::operations::bringup`, CMake target `ttnn_op_bringup_offset_cumsum`, kernel paths and includes pointing at this folder, Python prefix `ttnn.bringup.`.

## Changes

### Skip the internal all_gather on a 1-device dispatch axis
- What: with one device on the dispatch axis, the local histogram goes straight to the prefix-sum prim. No change
  for axes with more than one device.
- Why: `ttnn::all_gather` rejects a single-device cluster axis, so a MoE with one chip per dispatch group (a 1xN
  mesh dispatching along its size-1 rows) could not use offset_cumsum.
- Needed by: ernie45_d_p P3.1 (originally commit 2727352de41)
- Files: `offset_cumsum.cpp`

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
