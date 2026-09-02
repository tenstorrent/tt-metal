# QUASAR_UPLIFT_REPORT — ttnn.linear (matmul family)

**Date:** 2026-09-01
**Recipe:** `docs/source/ttnn/ttnn/ai/quasar_porting.md` + `docs/source/tt-metalium/tt_metal/apis/host_apis/metal_2.0/ai/audit/quasar_audit.md`
**Driving test:** `models/experimental/llama32_1b_quasar/tests/graph_ops/test_linear.py`
**Leave uncommitted for review; delete before merge.**

---

## Status: RED — Not Metal 2.0 on Gen1 yet

This is the first RED-stop condition in `quasar_porting.md` ("factory still
`create_descriptor`/`ProgramDescriptor`. Do the Metal 2.0 port first"). Per the recipe, a RED
result is a *success* of the audit — it stops a bad port. **No uplift was performed and no source
file was changed.**

## Program-factory paths in scope (what the test actually exercises)

The test's 8 captured cases carry exactly two `program_config` kinds. Routing per
`MatmulDeviceOperation::select_program_factory`
(`ttnn/cpp/ttnn/operations/matmul/device/matmul_device_operation.cpp:2187`):

| Test cases | Program config | Program factory |
|---|---|---|
| 00, 01, 02, 03, 04, 07 (6 cases, 1872 captured calls) | `MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig` | `MatmulMultiCoreReuseMultiCastDRAMShardedProgramFactory` (`device/factory/matmul_multicore_reuse_mcast_dram_sharded_program_factory.{hpp,cpp}`) |
| 05, 06 (2 cases, 96 captured calls) | `MatmulMultiCoreReuseMultiCastProgramConfig` | `MatmulMultiCoreReuseMcast2DProgramFactory` (`device/factory/matmul_multicore_reuse_mcast_2d_program_factory.{hpp,cpp}`) — descriptor path (not gather_in0, no global CB) |

**Factories deliberately NOT audited** (not reached by this test's configs):
`MatmulMultiCoreProgramFactory`, `MatmulMultiCoreReuseOptimizedProgramFactory`,
`MatmulMultiCoreReuseMcast1DProgramFactory`,
`MatmulMeshWorkloadMultiCoreReuseMcast1DProgramFactory` (gather_in0 / global-CB),
`MatmulMultiCoreReuseBatchedHSDRAMShardedProgramFactory`, and the sparse matmul op
(`device/sparse/`). Nothing in this report speaks for them.

## Gate evidence (§1 of quasar_porting.md)

The uplift precondition is a factory on `create_program_artifacts` → `ProgramArtifacts` with
`dfb::`/`args::`/`tensor::`/`scratch::` bindings, and kernels on the device-2.0 named-binding
model. Neither in-scope path meets it:

- **Host factories are ProgramDescriptor, not ProgramArtifacts.**
  - `matmul_multicore_reuse_mcast_dram_sharded_program_factory.hpp:14` — `static tt::tt_metal::ProgramDescriptor create_descriptor(...)`.
  - `matmul_multicore_reuse_mcast_2d_program_factory.hpp:41` — `create_descriptor(...)` (plus a legacy `override_runtime_arguments` at :34 and the `matmul_multi_core_reuse_mcast_2d_optimized_helper` legacy builder used by the CCL-fused path).
  - `grep -rn "create_program_artifacts\|ProgramArtifacts" ttnn/cpp/ttnn/operations/matmul/` → **zero hits** anywhere in the op.
  - Factory bodies build `KernelDescriptor`/`CBDescriptor`/`SemaphoreDescriptor` with positional `compile_time_args` and CB indices (e.g. dram_sharded factory :412–:448, 2D factory throughout).
- **Kernels are Device 2.0 includes but NOT the Metal 2.0 binding model.** The in-scope kernels
  (`reader_bmm_tile_layout_in0_sender_dram_sharded.cpp`,
  `reader_bmm_tile_layout_in1_sender_dram_sharded.cpp`,
  `reader_bmm_tile_layout_in0_sender_padding.cpp`,
  `reader_bmm_tile_layout_in1_sender_writer_padding.cpp`,
  `reader_bmm_tile_layout_in0_receiver.cpp`,
  `reader_bmm_tile_layout_in1_receiver_writer_padding.cpp`,
  compute `bmm_large_block_zm_fused_bias_activation.cpp`) include `api/dataflow/*` /
  `api/compute/*` headers, but use positional `get_compile_time_arg_val(N)` /
  `get_arg_val<uint32_t>(N)` and CB indices. `grep -rn "experimental/kernel_args.h\|dfb::"
  ttnn/cpp/ttnn/operations/matmul/device/kernels/` → **zero hits**.

Conclusion: the op is on the Host-2.0 *descriptor* API — the state `metal2_audit.md` treats as the
*starting point* for a Metal 2.0 port, not the finished port. The correct next step is the
canonical flow: `ai/audit/metal2_audit.md` → `METAL2_PREPORT_AUDIT.md`/`METAL2_PORT_BRIEF.md` →
`ai/port/metal2_port.md`, scoped to these two factories; only then re-run this Quasar-uplift audit.

## Files changed

**None.** (The recipe forbids performing the Metal 2.0 port under this uplift pass, and forbids
manufacturing changes on a RED.) `QUASAR_UPLIFT_REPORT.md` (this file) is the only addition.

## §7–§8 gotchas: applied vs considered

**Applied: none** — the RED gate stops before the uplift, and §7–§8 fixes are reactive (no device
run was performed in this session).

**Considered and verified against the code (forward-looking notes for the eventual uplift):**

- **§8.1 `blank.cpp` noop kernels missing:** *not currently applicable.* `grep -rn
  "blank.cpp\|NOOP_.*_KERNEL_PATH" ttnn/cpp/ttnn/operations/matmul/` → zero hits; neither in-scope
  factory references noop kernels today. This pitfall arises during/after the Metal 2.0 port —
  re-verify then.
- **§8.2 K-spill matmul `0x10000` tile-counter race:** *potentially applicable at Quasar run
  time, nothing statically fixable.* Both in-scope paths iterate K in blocks (DRAM-sharded cases:
  K = 2048/8192 with `in0_block_w` = 1–8, i.e. many K blocks; 2D cases: `in0_block_w` = 1/8).
  It is a HW-team issue (interim mitigation: DPRINT on); record only.
- **§8.3 in-place partials DFB rewind (`evil_set_*` is Gen1-only):** *not applicable.* That row is
  the conv in-place-matmul-partials pattern (packer DFB aliasing the output tensor with per-K-block
  ring rewind). Neither in-scope factory rewinds DFB ring pointers or aliases partials in place;
  no `evil_set_*` usage in the op.
- **§7 / quasar_audit.md check 2 — non-zero-init semaphore:** *genuine future uplift blocker
  found.* `matmul_multicore_reuse_mcast_dram_sharded_program_factory.cpp:657–658` creates
  `in0_mcast_sender_valid_semaphore_id` with `.initial_value = VALID` (VALID = 1,
  `hostdevcommon/common_values.hpp:14`). Non-zero-init semaphores do not port to Quasar
  (op-owner change required). The 2D factory's descriptor path uses only `INVALID` (= 0) inits
  (`matmul_multicore_reuse_mcast_2d_program_factory.cpp:1128–1135`) — clean.
- **§5 `fifo_page_size` staleness / `get_entry_size()`:** premature to assess meaningfully before
  the M2 port rewrites kernel buffer access; noted for the port brief.

## Deferred / follow-up items

1. **Metal 2.0 port of the two in-scope factories** (the actual blocker). Run
   `ai/audit/metal2_audit.md` scoped to `MatmulMultiCoreReuseMultiCastDRAMShardedProgramFactory`
   and `MatmulMultiCoreReuseMcast2DProgramFactory`, then `ai/port/metal2_port.md`. Note for that
   audit: the 2D factory file also hosts the legacy `matmul_multi_core_reuse_mcast_2d_optimized_helper`
   / `override_runtime_arguments` path used by CCL-fused matmul — a code-path-scope split will be
   needed, and the compute kernel `bmm_large_block_zm_fused_bias_activation.cpp` is shared across
   several matmul factories (shared-kernel rules of `metal2_port.md` apply).
2. **Non-zero-init semaphore** in the DRAM-sharded factory (`VALID`-initialized
   `in0_mcast_sender_valid_semaphore_id`) — op-owner change required before any Quasar uplift of
   that path (quasar_audit.md check 2).
3. Re-run this Quasar-uplift audit (`quasar_audit.md` + `quasar_porting.md` §7–§12, including the
   `cb_dfb_quasar_audit_helper.md` per-buffer classification) once the M2 port lands.

## Parity claim (WH/BH)

**Trivially preserved: the diff to op source is zero.** No file under
`ttnn/cpp/ttnn/operations/matmul/` was modified — this report is the only new file, and it is not
compiled or imported. Structurally, WH/BH behavior cannot have changed. (Per §9: no builds or
tests were run in this session; the commands below are for the user.)

## Test commands (user-run; per recipe §9 the agent runs nothing)

- **BH / WH parity (should pass unchanged — zero source diff):**
  ```
  pytest models/experimental/llama32_1b_quasar/tests/graph_ops/test_linear.py -v
  ```
- **Quasar (expected to be blocked until the M2 port lands):** same pytest under the Quasar
  emulator environment (craq-sim, `quasar` branch — see the craqsim runbook), with
  `TT_METAL_FORCE_JIT_COMPILE=1` after any kernel change, and run both with
  `TT_METAL_LLK_ASSERTS` on and off:
  ```
  TT_METAL_FORCE_JIT_COMPILE=1 pytest models/experimental/llama32_1b_quasar/tests/graph_ops/test_linear.py -v
  ```
