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

### Upstream sync to source @ `bdc576fa652` (2026-10-12): cache-hit dispatch overhead, include cleanup
- What: a 3-way merge (base = fork_op.py's mechanical copy of the source @ `73027b6e6ff`, theirs = the same copy of
  the source @ `bdc576fa652`, ours = this fork), no conflicts, of #58517 (less host work on a program-cache hit) and
  #59418 (include cleaner).
- Default behaviour: unchanged results; a cache hit does less host work.
- Needed by: rebase onto origin/llk_helper_library @ `bdc576fa652` (2026-10-12)
- Files: `combine.cpp`, `device/combine_program_factory.cpp`

### Upstream sync to source @ 6b80daf43e3 (rebase onto llk_helper_library, owner: full sync, 2026-10-27)
- Taken: #58910 (writer_untilize no longer pops the expert-count CB at exit: a reader / compute wait_front on it could
  hang), #60555 (reader_combine ends with noc_async_atomic_barrier: an unacknowledged credit return could land in the
  next kernel's counter), #60445's local-only path for a single-device dispatch group, and the kernel include moves
  (#60172, #61991).
- Conflict: #60445 forces num_links=0 on any singleton dispatch group and then requires every routed expert on the
  chip, which rejects this fork's EP calls (experts split over the other mesh axis, gemma4 / ernie45 / mimo / hy4).
  Owner: keep the fork's gating; upstream's local-only path applies only when the chip holds every routed expert
  (combine.cpp, `expert_token_counts.logical_shape()[-1] == experts_per_chip`); the factory keeps the fork's use_fabric
  neighbour gating (local-only implies no links, so no fabric).
- Behaviour: the two hang fixes apply to every caller. To report upstream: #60445's condition is too wide for EP.
