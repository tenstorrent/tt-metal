# Metal 2.0 Audit Findings — `ttnn/cpp/ttnn/operations/experimental/test/hang_device`

- **`ExecuteTestHangDeviceOperation`** (`hang_device_operation.hpp:12`) — the only DeviceOperation in the directory
  - **Direct-descriptor factory**: `create_descriptor` is a static member of the device-op itself, with no `program_factory_t` (`hang_device_operation.hpp:23`, body at `hang_device_program_factory.cpp:13-38`). The framework wraps it in the `MeshDeviceOperationAdapter::DirectDescriptorFactory` shim (`ttnn/api/ttnn/mesh_device_operation_adapter.hpp:171`, selected by `HasDirectDescriptor`, `ttnn/api/ttnn/operation_concepts.hpp:158`).
  - One kernel: `device/kernels/compute/hang_device_kernel.cpp`, a compute kernel on core `{0,0}` that runs `while (true);`.

The op exists only to hang the device for graph-capture and debug testing (nanobind doc, `hang_device_operation_nanobind.cpp:17-18`). Its Python entry point is `ttnn.test_hang_device_operation`.

**Layout note:** the device-op and factory `.cpp`/`.hpp` files sit at the op root. `device/` holds only `kernels/`. The op was resolved by its full path, so this didn't matter here; see Recipe notes.

**Scope:** TTNN op, Gen1 (WH/BH) target. This is within the scope of `audit/metal2_audit.md`.

**Recipe docs:** `4bd4bf42bfe 2026-09-03 docs(metal_2.0): state the offset-base wall as a category, not as slice's current state`
*(The recipe tree isn't in this checkout (`Metal_Ports`, branch `edwinlee/PD_Metal_Ports`). The hash comes from the `Port_Recipe` checkout (branch `akertesz/op-porting-recipe`). This audit followed `/localdev/edwinlee/metal2_audit.md`, which is a symlink to that checkout's `ai/audit/metal2_audit.md`.)*

**Readiness sheet:** fetched live on 2026-10-01 via the Google Drive connector (`download_file_content`, CSV). One row for this op.

## Status summary

| Field | Value |
|---|---|
| **Op directory** | `ttnn/cpp/ttnn/operations/experimental/test/hang_device` |
| **Overall** | **GREEN (user waiver).** Every code-side gate is clean. The only RED, the stale readiness-sheet row, was waived by the user on 2026-10-01 (see Result). |
| **DOps / Factories** | `ExecuteTestHangDeviceOperation` → direct `create_descriptor` (no factory struct). The sheet calls the factory `SingleCore`, which no longer exists. |
| *Prereqs* — Device 2.0 (every kernel used) | Yes. One compute kernel, with no CBs, no NoC and no semaphores. |
| *Prereqs* — Cross-op escapes | Ok. Includes are `tt_metal/*` only, and no kernel is borrowed. |
| *Feature Support* — overall | GREEN (all N/A) |
| *Feature Support* — Variadic-CTA | Ok (no CTAs) |
| *TTNN Readiness* — `Is able to port?` (the gate) | Sheet says `yes (with PD step)`, but the row is **stale**. `Concept` conflicts with the code, and the `SingleCore` factory row is a phantom. Treated as **spreadsheet-broken**, which makes this a GATE routed to the readiness-sheet owner. |
| *TTNN Readiness* — Concept (current) | Code: **`descriptor`** (direct-descriptor shape, since PR #57409 on 2026-09-25). Sheet: `legacy device-op`. |
| *TTNN Readiness* — Secretly SPMD (WorkloadDescriptor only) | N/A |
| *TTNN Readiness* — Custom hash | No. Sheet agrees (`no` / backdoor `no`). |
| *TTNN Readiness* — `get_dynamic_runtime_args` | No. Sheet agrees. |
| *TTNN Readiness* — `override_runtime_arguments` | No: the code has none. The sheet says `n/a`, which fits its stale `legacy` concept. |
| *TTNN Readiness* — Pybind `create_descriptor` | No. The only binding is the user entry point (`hang_device_operation_nanobind.cpp:22`). |
| *TTNN Readiness* — Op-owned tensors | No. Sheet agrees. |
| *TTNN Readiness* — Target concept | `ProgramSpecFactoryConcept`. The port must first introduce a factory struct (see Heads-ups). |
| *Port work* — Offset base pointer | none (no address args at all) |
| *Port work* — Tensor bindings (per binding) | none. The kernel touches no tensor memory. |
| *TTNN Readiness* — TensorParameter relaxation | `none` (clears) |
| *Port work* — TensorAccessor 3rd arg | none. No `TensorAccessor` exists anywhere in the op. |
| *Port work* — CB endpoints | legal (no CBs) |

CB endpoints are dispositions, not gates. This op declares no CBs, so there is nothing to classify.

## Result

**GREEN by user waiver → brief issued.** As audited, this was RED on the TTNN factory concept gate (spreadsheet-broken). On 2026-10-01 the user waived that gate: the blocker is only the sheet being out of date, not anything in the op. The sheet refresh is still owed to the readiness-sheet owner (Diego, `dgomez@tenstorrent.com`) as housekeeping; it is no longer a port blocker. The original finding is kept below.

**Original finding (pre-waiver):** RED → blocked on the TTNN factory concept gate (spreadsheet-broken), routed to the readiness-sheet owner.

The live sheet still describes the op as it was before PR #57409, *[Cleanup] Port More Ops to PD* (`f5093e705ae`, 2026-09-25). That PR moved the op from a legacy `SingleCore` factory (`create()` + `override_runtime_arguments()`) to a direct `create_descriptor`. The sheet still shows `Concept` = `legacy device-op` and the `SingleCore` factory row. The op code is already in a portable shape.

**This RED is cleared outside the op's code.** The sheet owner updates the row; nothing in the op changes. Per the recipe's exception, I ran all the informational subjects. The detail below should survive re-audit unchanged, so once the row is refreshed (`Concept` = `descriptor`, factory renamed or removed, `Is able to port?` re-derived), the re-audit should be a quick re-read and issue the brief.

The op has a single code path. Whole-op RED, with no subset distinction (there is nothing to split).

## Gate detail

- **TTNN factory concept (`Is able to port?`): RED (spreadsheet-broken)**, routed to the readiness-sheet owner to reconcile. The sheet row is `Op` = `experimental/test/hang_device`, `Device operation` = `ExecuteTestHangDeviceOperation`, `Factory (variant)` = `SingleCore`.
  - **Primary-column conflict: `Concept`.** The sheet says `legacy device-op` (and `Op Classification` = `Legacy Op`). The code has a `create_descriptor` returning `tt::tt_metal::ProgramDescriptor` (`hang_device_operation.hpp:23`, `hang_device_program_factory.cpp:13`). It has no `create()` and no `override_runtime_arguments()`, so the concept is `descriptor` (direct-descriptor form, which satisfies `ProgramDescriptorFactoryConcept` via the adapter shim).
  - **Phantom factory row: `SingleCore`.** `struct SingleCore` (with `create` / `override_runtime_arguments`) existed at `f5093e705ae^:…/hang_device_operation.hpp` and was deleted by #57409. The current code has no named factory. The real factory, the direct descriptor, has no row of its own.
  - **Cells that agree with the code:** `Custom hash` = `no` (no `compute_program_hash` in `hang_device_operation.hpp`); `Runtime-args update (get_dynamic_runtime_args)` = `no`; `Pybind descriptor` = `no`; `Op-owned tensors?` = `no`; `Smuggled pointer` = `no`.
  - **Cells consistent with the stale concept:** `Override runtime args method?` = `n/a` (that is the legacy-concept value; on a `descriptor` row it would be `no`, matching the code). `Is able to port?` = `yes (with PD step)` reads as "yes, once the PD migration lands", and that migration has now landed.
  - **Cross-column invariants:** none violated. `TensorParameter relaxation` = `none`, and `Known op issues` is empty.
  - **Path forward:** the sheet owner refreshes the row for #57409's change. If the re-derived `Is able to port?` comes back `yes`, the op clears every gate on its current code (see the other subjects below).
- **Device 2.0 (every kernel used): GREEN.** The op uses one kernel, `device/kernels/compute/hang_device_kernel.cpp` (bound only at `hang_device_program_factory.cpp:24-25`). It is a compute kernel whose whole body is `DPRINT_MATH(...)` plus `while (true);` (lines 8-11). It has no CB, NoC, semaphore, or address-generator use, so there is nothing to migrate. No donor kernels.
- **Feature compatibility: GREEN (no gate fired).**

  | Feature | Status | Notes |
  |---|---|---|
  | GlobalCircularBuffer | N/A | No `GlobalCircularBuffer` / `global_circular_buffer` / `remote_index`; the descriptor has no `cbs` |
  | CBDescriptor `address_offset` (non-zero) | N/A | No `CBDescriptor` at all |
  | GlobalSemaphore | N/A | No semaphores of any kind |

- **CB endpoints (GATE-free):** legal. The `ProgramDescriptor` declares no CBs (`hang_device_program_factory.cpp:21-37`, only `desc.kernels` is populated).
- **Offset base pointers: GREEN.** The kernel gets no runtime args, CRTAs, or CTAs, and no `->address()` expression appears anywhere in the op. The op isn't in `2026-07-19_offset_base_pointers.md` (grep confirmed), which fits the clean scan.
- **TensorAccessor 3rd argument: N/A.** No accessor exists in the op, so this subject never fires. The op isn't in `2026-07-06_tensor_accessor_3rd_arg_triage.md`.

## Port-work summary  *(mirrors the brief)*

- **Tensor bindings:** none. The kernel never touches the input or output tensor (no dataflow kernel, no address args), so the port declares no `TensorParameter` / `TensorBinding`. *(`create_output_tensors` still allocates an output, `hang_device_operation.cpp:27-31`; that is device-op plumbing and needs no binding.)*
- **TensorParameter relaxation:** `none`.
- **TensorAccessor 3rd arg:** none.
- **CB endpoints:** no CBs, so there are no DFB specs to declare.

## Heads-ups  *(mirrors the brief)*

- **Direct-descriptor shape: the port must introduce a factory struct.** The op declares `create_descriptor` on the device-op with no `program_factory_t` (`hang_device_operation.hpp:23-26`). The `DirectDescriptorFactory` shim is keyed on the name `create_descriptor` and has no `create_program_artifacts` counterpart. Swapping the method in place would leave the op failing `DeviceOperationConcept`. Follow `ttnn_factory.md` §"3. Give a direct-descriptor op a conventional program factory":
  - nest `struct ExecuteTestHangDeviceOperationProgramFactory { static ProgramArtifacts create_program_artifacts(...); };`
  - add `using program_factory_t = std::variant<...>;`
  - keep the body in `hang_device_program_factory.cpp`

  Record it under Handoff points. *(This op went legacy `SingleCore` → direct-descriptor in #57409, so it arrives without a `program_factory_t`; check again at port time in case TTNN has since added one.)*
- **Target concept: `ProgramSpecFactoryConcept`.** The code has no `override_runtime_arguments`. There is nothing to carry across cache hits: no RTAs, no bindings. The existing comment at `hang_device_program_factory.cpp:10-12` says the same.
- **KernelSpec carry-over:** a compute kernel on `CoreRangeSet(CoreRange({0,0},{0,0}))` with `HiFi4`, `fp32_dest_acc_en = false`, `math_approx_mode = false`, `opt_level = O3` (`hang_device_program_factory.cpp:19-33`). There are no CTAs, RTAs, CRTAs, defines, DFBs or semaphores. The validator needs at least one `KernelSpec` (`tt_metal/impl/metal2_host_api/program_spec.cpp:1250`) and at least one `WorkUnitSpec` (`:2303`), so the spec is one `KernelSpec` plus one `WorkUnitSpec` covering core `{0,0}`.
- **Kernel source likely needs no edit.** It uses no CB, tensor, semaphore or arg idioms, so there are no binding tokens to adopt. It includes only `api/debug/dprint.h` and `api/compute/compute_kernel_api.h`.
- **Cross-op / shared kernels:** none. `hang_device_kernel.cpp` lives in the op's own directory, and the only binder in `ttnn/cpp/ttnn/operations/` is this factory (`grep -rl hang_device_kernel`). No fork is needed; it is private to the op.
- **RTA varargs:** none (no args of any kind).
- **Build registration is split.** `hang_device_operation.cpp` and the nanobind file are listed in `experimental/test/sources.cmake:5,16`, but `hang_device_program_factory.cpp` is listed in the top-level `ttnn/sources.cmake:86`. That matters only if the port adds or renames a `.cpp`; keeping the factory body in the existing file needs no build change.
- **Testing caveat.** The op hangs the device by design, so a functional post-port test can't run it to completion. The golden function is deliberately `None` (`ttnn/ttnn/experimental_loader/golden_functions.py:217-218`). Verification will likely be build + program-spec validation, or whatever graph-capture test uses it. I found no direct test reference under `tests/`.

## Team-only

- **Out-of-directory coupling & donor shape:** ✓ clean.
  - Kernel `#include`s: `api/debug/dprint.h`, `api/compute/compute_kernel_api.h`. Both are in the `tt_metal/*` class, so no concern.
  - No kernel-lib, in-family, or cross-family donor includes, and no donor function calls.

  | Op kernel | Donor file | Class | Status |
  |---|---|---|---|
  | `hang_device_kernel.cpp` | `api/debug/dprint.h` | `tt_metal/*` | ✓ |
  | `hang_device_kernel.cpp` | `api/compute/compute_kernel_api.h` | `tt_metal/*` | ✓ |

  - Borrowed kernel files: none.
- **Relaxation candidates:** none (no custom hash).
- **TTNN factory analysis:**
  - Op-owned tensors: none.
  - MeshWorkload: not needed (single program, single core).
  - Pybind `create_descriptor`: none. `bind_test_hang_device_operation` binds only the user function, `hang_device_operation_nanobind.cpp:22-23`.
  - Other risky pybind: none.
  - Custom hash: none (default hash over the empty `operation_attributes_t` plus tensor args).
  - `get_dynamic_runtime_args`: none.
  - `override_runtime_arguments`: none.
  - Target concept: `ProgramSpecFactoryConcept`, via the factory-struct conversion above.

## Misc anomalies  *(team-only, non-gating)*

- **Input tensor and output allocation are unused, by design.** `tensor_args.tensor` is used only to shape the output (`hang_device_operation.cpp:19-25`). The output is allocated (`:27-31`) but never written, because the kernel never returns. This is expected for a hang op; recorded only so a reader doesn't mistake the unbound tensors for a missed binding.
- **Unused includes** in `hang_device_operation.cpp:5-9` (`ttnn/operation.hpp`, `ttnn/operations/core/core.hpp`, `ttnn/tensor/tensor_ops.hpp`, `tt-metalium/hal.hpp`), apparently left over from the legacy form. Cosmetic.

## Recipe notes

- **Op-resolution check assumes factories under `device/`.** "Read this first → Resolving the op" says to confirm "it has a `device/` subdirectory containing a `*_device_operation.*` and one or more program factories". This op keeps `hang_device_operation.*` and `hang_device_program_factory.cpp` at the op root, and `device/` holds only `kernels/`. It is clearly an op, but the literal test fails. Suggest wording like "a `*_device_operation.*` and factory in the op directory or its `device/` subdir".
- **Stale-sheet case for an op that recently landed its PD migration.** The sheet's `Is able to port?` = `yes (with PD step)` isn't in the gate's `yes`/`no` vocabulary. The routing rules cover `yes`, `no`, `MetalV2` and spreadsheet-broken, but not this tag. I treated it as moot, because a `Concept` conflict plus a phantom factory row already make the row spreadsheet-broken. It would still help to say what a `yes (with …)` tag means. That especially matters for the likely-common case where the PD step has *landed* since the sheet was derived, which is exactly how this RED arose (PD-migration batch #57409, 2026-09-25).
- **The "spreadsheet-broken → GATE" rule produces a RED with no code-side work.** *(In this audit the user resolved it by waiving the gate and issuing the brief.)* Every gate is clean on the code, so this RED rests entirely on sheet staleness. That follows the recipe as written, and I didn't override it. A note on whether the auditor may issue a *provisional* brief in this situation (staleness from a recently landed PD migration) would save a re-audit round-trip.
- **Provenance in a separate checkout** (repeat of an earlier report's note): the provenance `git log` prints nothing in `Metal_Ports`, because the recipe tree lives in the sibling `Port_Recipe` checkout. The hash above comes from there.
