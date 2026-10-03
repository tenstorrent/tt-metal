# Port Plan — experimental/test/hang_device

Port plan for `ttnn/cpp/ttnn/operations/experimental/test/hang_device` (`ExecuteTestHangDeviceOperation`), ported from the direct-descriptor `ProgramDescriptor` API to Metal 2.0.
Written during the inventory and planning steps; committed alongside the port for review.

## Legacy Inventory

### Legacy factory shape
- Concept: `ProgramDescriptorFactoryConcept`, reached through the **direct-descriptor shim**. `create_descriptor` is a static member of `ExecuteTestHangDeviceOperation` itself (`hang_device_operation.hpp:23-26`, body `hang_device_program_factory.cpp:13-38`). There is no `program_factory_t` and no factory struct. The framework wraps it in `MeshDeviceOperationAdapter::DirectDescriptorFactory` (selected by `HasDirectDescriptor`). Re-checked at port time: no `program_factory_t` has appeared since the audit.
- Variants: single.
- Custom `compute_program_hash`: none. The default reflection-based hash covers the empty `operation_attributes_t` plus `tensor_args`. No `attribute_values` / `to_hash` backdoor.
- `override_runtime_arguments`: none.

*(The Metal 2.0 factory concept the port targets was chosen during the audit; see the brief's TTNN factory analysis. Carried forward in [TTNN ProgramFactory](#ttnn-programfactory) below.)*

### Kernels
| unique_id | source | core_ranges | CTAs (positional) | CTAs (named) | RTAs | CRTAs | defines | opt_level | config |
|---|---|---|---|---|---|---|---|---|---|
| (unnamed) → `compute` | `ttnn/cpp/ttnn/operations/experimental/test/hang_device/device/kernels/compute/hang_device_kernel.cpp` | `CoreRangeSet(CoreRange({0,0},{0,0}))` | none | none | none | none | none | `O3` (explicit, `hang_device_program_factory.cpp:28`) | `ComputeConfigDescriptor{.math_fidelity = HiFi4, .fp32_dest_acc_en = false, .math_approx_mode = false}`. Every other field is at its default: `dst_full_sync_en = false`, `unpack_to_dest_mode = {}`, `bfp8_pack_precise = false`, `enable_trisc2_rvv = false` |

Kernel body: `DPRINT_MATH("Hanging the device, use this only for testing!!!\n"); while (true);`. It includes only `api/debug/dprint.h` and `api/compute/compute_kernel_api.h`.

### CBs
none. The descriptor populates only `desc.kernels`.

### Semaphores
none

### Tensor accessors
none. No host- or device-side `TensorAccessor`, and no buffer address in any arg.

### Work split
n/a. Single core `{0,0}`.

### Shared kernels
none. `grep -rl hang_device_kernel ttnn/cpp/ttnn/operations/` hits only `hang_device_program_factory.cpp`. The kernel is private to this op, and no `_metal2` fork exists or is needed.

### Flags
- Build registration is split. The factory `.cpp` is in top-level `ttnn/sources.cmake:86`; the device-op and nanobind `.cpp`s are in `experimental/test/sources.cmake:5,16`. The port keeps the factory body in the same file, so no build edit.
- No test anywhere in `tests/` (or `models/`) exercises the op. Its golden function is `None` by design (`ttnn/ttnn/experimental_loader/golden_functions.py:217-218`).

## TTNN ProgramFactory

- **Concept (inherited from audit)**: `ProgramSpecFactoryConcept`.
- **Custom `compute_program_hash`**: none.
- **Implementation notes**: forced device-op-class edit, `ttnn_factory.md` §3 ("Give a direct-descriptor op a conventional program factory"). Nest `struct ExecuteTestHangDeviceOperationProgramFactory { static ProgramArtifacts create_program_artifacts(...); };` in `ExecuteTestHangDeviceOperation`, add `using program_factory_t = std::variant<ExecuteTestHangDeviceOperationProgramFactory>;`, and remove the device-op-level `create_descriptor`. The body stays in `hang_device_program_factory.cpp`. No pybind references `create_descriptor` (the only binding is the user function, `hang_device_operation_nanobind.cpp:22-23`), so exception 1 does not apply.

## Planned Spec Shape

- KernelSpecs: one, `compute`, from the same source, with `hw_config = ComputeHardwareConfig{.fpu_math_fidelity = HiFi4, .sfpu_precision_mode = Precise, .enable_32_bit_dest = false}`. Style B: the op sets a Metal `ComputeConfigDescriptor` directly, so the struct is built by hand, not through the TTNN helper. `double_buffer_dest` stays at its default `true` (= `!dst_full_sync_en`, with `dst_full_sync_en` defaulting to `false`). `unpack_modes` is empty (legacy vector empty, no DFBs). `config_1xx` is unset, since `bfp8_pack_precise` was the default `false`, which equals the `Approximate` default. `compiler_options.opt_level = O3` (carried verbatim). No bindings, no CTAs, no RTA schema.
- DataflowBufferSpecs: none.
- SemaphoreSpecs: none.
- TensorParameters: none. The kernel touches no tensor memory. The input only shapes the output, and the output is allocated by `create_output_tensors` but never written. That is by design.
- WorkUnitSpecs: one, `hang_device`, `kernels = {compute}`, `target_nodes = CoreCoord{0,0}`. The validator requires at least one `KernelSpec` and at least one `WorkUnitSpec`.
- Op-owned tensors: none.
- ProgramRunArgs: empty. There are no kernel run-args (no RTAs/CRTAs) and no tensor args.

## Preserved Multiplicity

none. There is no work-split multiplicity in legacy (one `KernelDescriptor`, one core).

## Dropped Plumbing

none. The legacy kernel takes no RTAs, CRTAs or CTAs, uses no CB indices, has no `TensorAccessorArgs`, and has no semaphore IDs. Nothing needs dropping or renaming.

| legacy location (file:line) | legacy form | Metal 2.0 replacement |
|---|---|---|
| `hang_device_program_factory.cpp:26` | `source_type = FILE_PATH` | implied by `KernelSpec::source` holding a `std::filesystem::path` |

## Applied Patterns

- `ttnn_factory.md` §3: give a direct-descriptor op a conventional program factory (forced).

No catalog DFB, binding or vararg patterns apply.

## Deferred / Flagged

- **Verification shape.** The op hangs the device by design and has no tests, so a functional run is impossible. Verification uses a temporary, uncommitted gtest that calls `ttnn::device_operation::prepare<ExecuteTestHangDeviceOperation>` twice. That runs the real adapter's cache-miss path (factory, then spec validation, JIT compile and finalize) and then a cache-hit path, without dispatch. It runs pre-port (baseline) and post-port, with the Metal 2.0 legality checks forced on.
