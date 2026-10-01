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

### Tests: ernie45_d_p case
- What: appended the random-input case for the call ernie45_d_p makes (1x4 mesh, FABRIC_1D_RING, S 5120, H 2560,
  64 experts top-6, 16 per chip). No op change, no test-file change.
- Why: task O.1, every call a model makes to a fork gets a case.
- Needed by: ernie45_d_p O.1
- Files: `tests/cases.py`

### Tests: hy4_preview_d_p case
- What: appended the random-input case for the call hy4_preview_d_p makes (2x2 mesh, axis 0, 256 experts, 64 per
  chip; the same op arguments as the mimo_v2_6_d_p_2x2 case, recorded under this model). No op change.
- Why: task O.1, every call a model makes to a fork gets a case.
- Needed by: hy4_preview_d_p O.1
- Files: `tests/cases.py`

### Upstream sync: a fourth output, `all_global_dispatch_offsets` (#57859, source @ `73027b6e6ff`, 2026-10-01)
- What: 3-way merge (base = fork_op.py's copy of the source @ `67ca5f3af48`, theirs = the same copy @ `73027b6e6ff`,
  ours = this fork; no conflicts) of #57859: the op returns four tensors, the new last one
  `[devices along cluster_axis, n_routed_experts]`, every device's `global_dispatch_offsets` row, replicated along
  the axis (store-and-forward dispatch sizes other devices' runs from it); the kernel reads all rows once and
  writes in 8-column blocks (source: 2-3x faster). The first three outputs are unchanged by position and value.
  The fork's own change (no all_gather on a 1-device dispatch axis) is kept: the table then has one row.
- Why: the source's tests and wrappers now unpack four outputs, so the fork's carried source tests could not run
  (21 failed with "expected 4, got 3"); keeping the fork on the source's contract as main moves.
- Default behaviour: changes the return arity (bug-fix-style exception to "behind an option"): every caller now
  unpacks four (ernie45_d_p, gemma4_a4b_d_p, mimo_v2_6_d_p, mimo_v2_6_d_p_2x2, hy4_preview_d_p, glm53_flash_d_p,
  xing40_a4b_d_p, `tt/experts.py` / `tt/moe_unified.py`, the fourth ignored). `tests/test_offset_cumsum.py` checks the
  fourth output against the torch reference too.
- Needed by: rebase onto origin/malimpic/llk_helper_library_rebased_0110_2
- Files: `device/*`, `offset_cumsum.{hpp,cpp}`, `offset_cumsum_nanobind.cpp`, `tests/test_offset_cumsum.py`
