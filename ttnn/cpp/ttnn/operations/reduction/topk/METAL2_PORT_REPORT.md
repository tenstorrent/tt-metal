# Port Report — topk (`TopKSingleCoreProgramFactory`)

Metal 2.0 port of the **single-core** factory only. The multi-core factory is RED
(cross-node gather DFB, unsupported) and was not touched — it stays on the legacy
`ProgramDescriptorFactoryConcept`. The op keeps building with mixed concepts in the
`program_factory_t` variant.

## TTNN ProgramFactory

### Concept realized
`MetalV2FactoryConcept` — `TopKSingleCoreProgramFactory::create_program_artifacts`
returns `ttnn::device_operation::ProgramArtifacts{.spec, .run_params}`. No op-owned
tensors, no semaphores, single-program, single WorkUnitSpec.

### Device-op-class edits
- Custom `compute_program_hash` deleted: **none** — the op never had one (default reflection hash).
- Pybind entry points removed: **none** — `topk_nanobind.cpp` binds only the op function
  (`ttnn::bind_function<"topk">`); it never referenced `create_descriptor`. The framework
  dispatches the factory by concept detection, so renaming `create_descriptor` →
  `create_program_artifacts` on the single-core factory required no pybind change.
- Header edit (forced, in scope): `device/topk_device_operation.hpp` — the single-core
  factory's method signature changed from
  `static tt::tt_metal::ProgramDescriptor create_descriptor(...)` to
  `static ttnn::device_operation::ProgramArtifacts create_program_artifacts(...)`, and
  `#include <ttnn/metal_v2_artifacts.hpp>` was added. The multi-core factory's
  `create_descriptor` declaration is unchanged.

### Open items
- **`unpack_to_dest_mode` not set (potential validator interaction — WATCH AT BUILD/TEST).**
  The legacy `ComputeConfigDescriptor` left `unpack_to_dest_mode` empty (default for all CBs)
  while setting `fp32_dest_acc_en = !uint16_output`. The port faithfully mirrors that: the
  `ComputeHardwareConfig` sets `fp32_dest_acc_en` only, `unpack_to_dest_mode` left empty.
  Per `compute_hardware_config.hpp`, the Metal 2.0 validator REQUIRES an
  `unpack_to_dest_mode` entry for any DFB when ALL of: (1) the kernel is the consumer,
  (2) the DFB data format is Float32, (3) `fp32_dest_acc_en == true`. For TopK this triple
  can hold for the `input`/`transposed_val`/`result_prep_val` DFBs when the input dtype is
  Float32 and `dim` is large enough that `uint16_output == false`. If the validator fires
  ("missing unpack_to_dest_mode for DFB ..."), the fix is to add
  `{<float32 compute-consumer DFB>, UnpackToDestMode::UnpackToDestFp32}` entries for those
  DFBs under `!uint16_output`. I did NOT pre-emptively add them because (a) it changes the
  spec vs. the legacy default, and (b) the test sweep will tell us whether it's actually
  required (most TopK tests use bf16/fp16 inputs, where the triple never holds). Flagging
  rather than guessing. The accumulation reference port DID set `UnpackToDestFp32` for its
  Float32 acc DFB, so the validator very likely enforces this for fp32-input TopK.
  **RESOLVED (test sweep):** the full `test_topk.py` sweep (117 passed / 80 xfailed / 0 failed)
  did NOT trigger the validator — fp32 inputs are rejected upstream by TopK's dtype guard
  (`test_topk_input_dtypes_raise` confirms fp32/uint32/int32 raise), so the Float32-consumer
  triple never holds in practice. No `unpack_to_dest_mode` entry is needed for this op as it
  stands. Left empty, matching legacy. (Would need revisiting only if TopK ever accepts fp32 input.)
- **Relaxation candidates:** none identified; strict `TensorSpec` matching kept on all 4 params.

## Handoff points

- **GH #36329 (op owner) — `GENERATE_INDICES` hardcoded.** The single-core factory hardcodes
  `GENERATE_INDICES=1` (legacy comment cites #36329), so the precomputed-indices read path
  in `reader_create_index_tensor.cpp` (`#if not GENERATE_INDICES`) is dead, and the optional
  `indices` input is effectively unused in single-core. The port preserves this behavior:
  `generate_indices` is a `constexpr bool = true`, the `GENERATE_INDICES=1` define is emitted
  to the reader, and the optional `indices` TensorParameter / binding / tensor-arg are gated by
  `bind_precomputed_indices = tensor_args.indices.has_value() && !generate_indices` (always
  false today). When #36329 is fixed (set the define from `indices.has_value()`), flipping
  `generate_indices` will light up the typed `ta::indices` binding with no other change. This
  is an op-owner decision, not a port change. NOT in port scope.

## Successes

- **Self-loop DFB binding pattern** ([catalog](metal2_port_patterns.md#pattern-self-loop-dfb-binding))
  fit the four compute staging/result-prep buffers (c_2..c_5) exactly. Each is both reserved/
  pushed and waited/popped within `topk.cpp` itself — genuine producer AND consumer — so the
  PRODUCER+CONSUMER pair on one `KernelSpec` (shared `accessor_name`) is the honest model,
  not a fake-CB white lie. The pattern doc steered this correctly.
- **`dfb::name` → `uint32_t` implicit conversion** ([catalog](metal2_port_patterns.md#pattern-pass-dfb-handles-directly-to-llks-and-kernel-lib-helpers))
  carried the bulk of the compute-kernel port: `transpose_wh_*`, `pack_tile`, `copy_tile`,
  `reconfig_data_format_*`, and the file-local helpers (`transpose_and_pack`, `pack_results`,
  etc., which take `uint32_t cb` params) all accept `dfb::name` unchanged. Crucially it also
  let the reader pass `dfb::index` into the SHARED `generate_index_tile(uint32_t cb_id, ...)`
  helper in `topk_dataflow_common.hpp` WITHOUT editing that header (which is shared with the
  RED multi-core kernels). The `constexpr uint32_t cb = dfb::name;` aliases in `topk.cpp`
  (needed because runtime vars `cb0..cb3` are reassigned among them) also rely on the
  conversion being `constexpr`.

## Friction

### Gaps
- **`unpack_to_dest_mode` legacy-default behavior is undocumented for the port path.** The recipe
  ([Construct → Hardware-config shortcuts](port_op_to_metal2_recipe.md#construct-paired-spec--run-args))
  says "configure it separately when needed (e.g., for FP32 DFB consumers under fp32_dest_acc_en)"
  but the legacy `ComputeConfigDescriptor` for this op leaves it EMPTY while enabling
  `fp32_dest_acc_en`. It's unclear whether the legacy ProgramDescriptor path silently defaulted
  the per-CB modes or whether the kernel simply never hit the fp32 triple in practice. The
  recipe could state explicitly: "if the legacy compute config left `unpack_to_dest_mode` empty
  AND set `fp32_dest_acc_en`, you must determine whether any compute-consumer DFB is Float32 and
  add `UnpackToDestFp32` for it — the Metal 2.0 validator enforces what the legacy path tolerated."
  See Open items. (Resolved-pending-test rather than blocking.)

### Confusion
- **`dataflow_buffer.h` include needed beyond the documented `kernel_args.h`.** The recipe
  ([kernel-side whitelist](port_op_to_metal2_recipe.md#kernel-side-whitelist)) states "the single
  `#include` a porter adds to a kernel is `experimental/kernel_args.h`; the generated headers are
  auto-included." But the `DataflowBuffer` wrapper class lives in `api/dataflow/dataflow_buffer.h`
  (NOT in `api/dataflow/circular_buffer.h`, which only defines `CircularBuffer`), and neither
  `kernel_args.h` nor `circular_buffer.h` pulls it in. All three ported kernels needed an explicit
  `#include "api/dataflow/dataflow_buffer.h"` for the `CircularBuffer`→`DataflowBuffer` swap to
  compile. This matches what `ttnn/cpp/ttnn/kernel_lib/*.inl` do. The whitelist's "only
  kernel_args.h" claim is too strong — `dataflow_buffer.h` is a required second include whenever
  the kernel constructs a `DataflowBuffer`. (Resolved.)
- **⚠ Tensor-binding namespace is `tensor::`, not `ta::` (docs are STALE — caused 94 test failures).**
  The migration guide / patterns docs and this recipe refer to the kernel-side tensor-binding
  tokens as `ta::name` (e.g. `TensorAccessor(ta::inout)`). That namespace was **renamed to
  `tensor::`** (the framework now injects `namespace tensor {…}` — `tt_metal/jit_build/genfiles.cpp:181`).
  Following the docs, this port first emitted `ta::inout` / `ta::values` / `ta::indices` /
  `ta::out_indices`, which JIT-failed to compile (`'ta' was not declared in this scope`) on every
  single-core case — 94/94 single-core test failures, all the identical error. Fix was mechanical:
  `ta::` → `tensor::` in `reader_create_index_tensor.cpp` and `writer_binary_interleaved.cpp`
  (4 sites). After the fix: 117 passed / 80 xfailed / 0 failed. **Action for doc owner:** sweep the
  Metal 2.0 docs (`metal2_migration_guide.md`, `metal2_port_patterns.md`, this recipe, the audit's
  Case-1/Case-2 wording) and replace `ta::name` with `tensor::name`. (The legacy host-side
  TensorAccessor *args* type is still `TensorAccessorArgs`; only the injected kernel-token
  namespace changed.) (Resolved.)
- **Reference branch method name is stale.** The recommended reference port
  (`akertesz/porting-experiment-accumulation-jun10`) names the factory method
  `create_program_spec` and the concept `ProgramSpecFactoryConcept`; the current tree renamed
  these to `create_program_artifacts` / `MetalV2FactoryConcept`. The recipe/ttnn_factory doc
  use the current names, so following the docs over the reference code was correct, but a porter
  copying the reference verbatim would pick the wrong method name and silently fail the
  `MetalV2FactoryConcept` detection (the factory would match neither concept). Worth a note in
  the recipe's "Optional reference port" line.
- **`TensorArgument` construction.** The reference wraps tensors in `std::cref` / `TensorArgument{std::cref(t)}`;
  the recipe says pass the `MeshTensor` directly (`{INPUT, input}`). I followed the recipe
  (direct `{INPUT_TENSOR, input_tensor}` brace-init into the `Table`). The two should be
  equivalent via the `reference_wrapper` implicit conversion. (Resolved by following the recipe.)

## Open items for downstream
- **Fake-CB self-loop bindings:** none. The c_2..c_5 self-loops are GENUINE (real producer and
  consumer in the same compute kernel), not validator-satisfying fake-CB stand-ins.
- **Cross-op kernel touches:** none. The shared in-dir headers (`topk_dataflow_common.hpp`,
  `topk_common_funcs.hpp`) were NOT modified — they are shared with the RED multi-core kernels,
  and the `dfb::name`→`uint32_t` conversion made `generate_index_tile` usable as-is.
- **Dead `Ht` CTAs preserved.** The reader/writer `Ht` named CTA and the compute `Ht` CTA are
  unused by the kernels (carried from legacy). Preserved verbatim rather than removed — a cleanup
  candidate for the op owner, out of port scope.
