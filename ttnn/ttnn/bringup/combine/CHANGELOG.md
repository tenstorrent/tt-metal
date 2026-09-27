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
