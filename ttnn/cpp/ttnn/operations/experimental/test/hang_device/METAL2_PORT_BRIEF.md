# Metal 2.0 Port Brief — `ttnn/cpp/ttnn/operations/experimental/test/hang_device`

> Audit cleared all gates. One gate was cleared by user waiver; see below. This is your actionable input; the full record is in `METAL2_PREPORT_AUDIT.md`.

**Gates cleared:** Device 2.0 ✓ · Features ✓ · TTNN factory concept ✓ *(user waiver: readiness-sheet row is stale, see below)* · Offset base pointers ✓ · TensorAccessor 3rd arg ✓

**Recipe docs:** `4bd4bf42bfe 2026-09-03 docs(metal_2.0): state the offset-base wall as a category, not as slice's current state` *(carry this line into the port report's Provenance section; hash from the `Port_Recipe` checkout)*

**TTNN gate waiver (record in the port report).** The live readiness sheet (fetched 2026-10-01) still describes this op as it was before PR #57409: `Concept` = `legacy device-op`, factory `SingleCore`, `Is able to port?` = `yes (with PD step)`. The code is already on a direct `create_descriptor`, so the audit flagged the sheet as broken. On 2026-10-01 the user waived that gate, since the problem is only the out-of-date sheet. The sheet refresh is still owed to the readiness-sheet owner.

## TTNN factory analysis

These facts feed the port's TTNN ProgramFactory wiring (→ `ttnn_factory.md`). The op ports to `ProgramSpecFactoryConcept`. Carry them forward:

- **Current concept:** `descriptor`, in the **direct-descriptor** shape. `create_descriptor` is a static member of `ExecuteTestHangDeviceOperation` itself (`hang_device_operation.hpp:23-26`, body `hang_device_program_factory.cpp:13-38`), with no `program_factory_t`.
- **Op-owned tensors:** none.
- **Target concept:** `ProgramSpecFactoryConcept`. There is no `override_runtime_arguments`.
- **Gate-cleared, confirmed absent:** a non-clearing `TensorParameter relaxation` (cell is `none`) and `get_dynamic_runtime_args` (absent). The op also has no custom hash, no `override_runtime_arguments`, and no pybound `create_descriptor`. The only nanobind binding is the user function, `hang_device_operation_nanobind.cpp:22-23`, so no pybind line needs deleting.

## Construct — to do

**Introduce a factory struct (forced; `ttnn_factory.md` §3 "Give a direct-descriptor op a conventional program factory").** The `DirectDescriptorFactory` shim (`ttnn/api/ttnn/mesh_device_operation_adapter.hpp:171`, gated by `HasDirectDescriptor`, `ttnn/api/ttnn/operation_concepts.hpp:158`) only recognizes `create_descriptor`, and has no `create_program_artifacts` counterpart. In `hang_device_operation.hpp`, nest `struct ExecuteTestHangDeviceOperationProgramFactory` with `static ttnn::device_operation::ProgramArtifacts create_program_artifacts(const operation_attributes_t&, const tensor_args_t&, tensor_return_value_t&);`. Then add `using program_factory_t = std::variant<ExecuteTestHangDeviceOperationProgramFactory>;` and remove the device-op-level `create_descriptor`. Keep the body in `hang_device_program_factory.cpp`. Record this under Handoff points: the op arrived in the direct-descriptor shape. *(Check first that no `program_factory_t` has appeared since the audit.)*

**ProgramSpec contents:** one compute `KernelSpec` plus one `WorkUnitSpec` on core `{0,0}`. Nothing else.
- Source: `ttnn/cpp/ttnn/operations/experimental/test/hang_device/device/kernels/compute/hang_device_kernel.cpp`
- Compute config: `MathFidelity::HiFi4`, `fp32_dest_acc_en = false`, `math_approx_mode = false`. Opt level `O3` (`hang_device_program_factory.cpp:19-33`).
- No CTAs, RTAs, CRTAs, defines, DFBs, or semaphores.
- The validator requires at least one `KernelSpec` (`tt_metal/impl/metal2_host_api/program_spec.cpp:1250`) and at least one `WorkUnitSpec` (`:2303`). Both are met.

**Tensor bindings:** none. The kernel touches no tensor memory, so declare no `TensorParameter` / `TensorBinding`. The input tensor only shapes the output, and the output is allocated by `create_output_tensors` but never written (the kernel never returns). That is by design, not a missed binding.

**TensorParameter relaxation:** `none`.

**TensorAccessor 3rd arg:** none (no accessors).

**CB endpoints:** no CBs → no DFB specs.

## Watch for

- **CB endpoints (multi-binding):** none.
- **Cross-op / shared kernels:** none. `hang_device_kernel.cpp` is private to this op, and this factory is its only binder, so no `_metal2` fork is needed. Do not borrow anything from `experimental/quasar/`.
- **RTA varargs:** none.
- **Kernel source likely needs no edit.** Its body is `DPRINT_MATH(...); while (true);`, and it includes only `api/debug/dprint.h` and `api/compute/compute_kernel_api.h`. There are no CB, tensor, semaphore or arg idioms to rewrite. If you change it at all, keep within the kernel-side whitelist.
- **Build registration is split.** The factory `.cpp` is registered in top-level `ttnn/sources.cmake:86`, not in `experimental/test/sources.cmake`. Keep the body in the existing file to avoid build edits.
- **Verification.** The op hangs the device by design, and its golden function is deliberately `None` (`ttnn/ttnn/experimental_loader/golden_functions.py:217-218`). Don't try to run it to completion. Verify by build plus program-spec construction and validation (cache-miss path), and say in the port report which checks were possible.
