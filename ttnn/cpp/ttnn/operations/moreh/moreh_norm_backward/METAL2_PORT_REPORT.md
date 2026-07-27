# Metal 2.0 Port Report — moreh_norm_backward

## Outcome

**PORTED** — the single factory of `MorehNormBackwardOperation` converted to `MetalV2FactoryConcept`
(`create_program_artifacts`). All three op-owned kernels (reader, writer, compute) converted.
**Build and test verification is the orchestrator's** (this porter did not build or run tests, per
orchestration constraints). Exact commands below.

## Provenance

- **Recipe docs (this port):** `de19c9df758 2026-07-22 docs(metal_2.0): route Gen1 porters away from the Quasar-uplift audit helper`
- **Audit docs (inherited):** `de19c9df758 2026-07-22 docs(metal_2.0): route Gen1 porters away from the Quasar-uplift audit helper`

## TTNN ProgramFactory

### Concept realized
`MetalV2FactoryConcept`, as inherited from the audit. The legacy op was **HasDirectDescriptor**
(`create_descriptor` directly on `MorehNormBackwardOperation`, no `program_factory_t`). The framework
only auto-synthesizes a `DirectDescriptorFactory` for a direct `create_descriptor` — there is no
equivalent auto-wrap for a direct `create_program_artifacts` (`mesh_device_operation_adapter.hpp`
`resolve_program_factory` / `DirectDescriptorFactory`). So the port introduces a nested factory struct
`MorehNormBackwardProgramFactory` (holding `create_program_artifacts`) and
`using program_factory_t = std::variant<MorehNormBackwardProgramFactory>` on the DeviceOperation. Single
variant → no `select_program_factory` needed. This mirrors `experimental/quasar/binary_ng`'s
`ProgramFactoryMetalV2`.

### Device-op-class edits
- Custom `compute_program_hash` deleted: **none** (op never had one).
- Pybind entry points removed: **none** — nanobind binds only the `moreh_norm_backward` free function
  (`moreh_norm_backward_nanobind.cpp`); no `create_descriptor` pybind existed.
- Header change: replaced the `static ProgramDescriptor create_descriptor(...)` declaration with the
  factory struct + `program_factory_t` typedef (`device/moreh_norm_backward_device_operation.hpp`), and
  swapped `#include <tt-metalium/program_descriptors.hpp>` for `#include "ttnn/metal_v2_artifacts.hpp"`
  (+`<variant>`). The three namespace-scope helper declarations (`get_floored_p_and_decimal_and_p_is_negative`,
  `get_tensor_dim`, `get_output_grad_shape`) are unchanged and still defined in the factory `.cpp`.

### Open items
- **Compute `dst_full_sync_en` / `packer_l1_acc` never reach the config (pre-existing).** The legacy factory
  resolves a full `ComputeKernelConfig` via `get_compute_kernel_config_args` but wires only
  `math_fidelity` / `fp32_dest_acc_en` / `math_approx_mode` into its Metal `ComputeConfigDescriptor`;
  `dst_full_sync_en` and `packer_l1_acc` are dead locals. The port reproduces the *effective* legacy config
  (see Friction → Compute-config style). If the op owner intended `dst_full_sync_en` / `packer_l1_acc` to
  take effect, that is a functional fix for a separate PR — not this port.
- **Relaxation candidates:** none. No `ArgConfig::Runtime*` in the kernels; strict `TensorSpec` match kept.

## Handoff points

None. No capitulation; no out-of-op-directory edits; no `sem::`/`tensor::` boundary violations; the shared
`moreh_common.hpp` headers (dataflow + compute) were **not** modified — every consumed helper
(`fill_cb_with_value`, `sign_tile_to_cb`, `power_tile_with_abs_x_to_cb`, `power_and_recip_tile_to_cb`,
`mul_tiles_to_cb`, the `*_with_dt` init/pack family) already takes `DataflowBuffer` objects, so the
`dfb::name → uint32_t` implicit conversion covers the raw-cb-id LLK call sites with no callee change.

## Successes

- **Patterns catalog — self-loop DFB / two-toucher distinction.** The brief flagged the 8 intermediates
  (`c_24`–`c_31`) as self-loop and the compute work-split as "two KernelDescriptors over disjoint node
  sets — ordinary 1:1 per node, not a two-toucher." Both held: I bound each intermediate PRODUCER+CONSUMER
  on each compute KernelSpec (self-loop) and used two compute KernelSpecs over disjoint node sets with **no**
  `allow_instance_multi_binding`. Verified against the per-node census in
  `tt_metal/impl/metal2_host_api/program_spec.cpp` (`ValidateProgramSpec`, ~L1337-1412: exactly one producer
  + one consumer *per node*), which confirmed the shared read/write DFBs (reader→both computes; both
  computes→writer) are legal because each node hosts exactly one compute instance.
- **Migration guide — "identical WorkUnitSpec membership" is not the real rule.** The troubleshooting note
  ("a local DFB's producer and consumer kernels must share identical WorkUnitSpec membership") reads as a
  hard membership check, which would be impossible to satisfy for a reader shared across two per-group
  compute work units. The actual validator enforces per-*node* 1P+1C coverage, which the two-per-group work
  units (`{READER,WRITER,COMPUTE_1}`@group1, `{READER,WRITER,COMPUTE_2}`@group2) satisfy. Reading the code
  rather than the prose prevented an unnecessary restructure. See Friction.

## Friction

### Confusion
- **`docs migration_guide.md` local-DFB invariant wording (Gap/Confusion).** The troubleshooting-table phrasing
  "producer and consumer kernels must share **identical WorkUnitSpec membership**" contradicts the actual
  per-node-census implementation and the DFB-spec header's own "you MAY bind more than one KernelSpec to a
  producer/consumer endpoint … non-overlapping node coverage" allowance. For any work-split op with a shared
  reader/writer and per-group compute, membership is *not* identical but coverage *is* equal. Suggest the doc
  say "equal per-node coverage (each node runs exactly one producer and one consumer instance)" instead of
  "identical WorkUnitSpec membership."
- **Compute-config style (A vs B) is ambiguous when an op does both.** This op *resolves* a TTNN
  `ComputeKernelConfig` (Style A signal: `get_compute_kernel_config_args`) but then *constructs a Metal
  `ComputeConfigDescriptor` directly* with only three of the knobs (Style B signal), silently dropping
  `dst_full_sync_en`. Routing this through `to_compute_hardware_config` (the Style-A path) would have wired
  the resolved `dst_full_sync_en` the legacy op never applied — a silent perf/precision change. I treated it
  as **Style B** (build `ComputeGen1Config` directly; `double_buffer_dest` left at its default `true`, which
  equals the legacy `!dst_full_sync_en` with the descriptor's default `dst_full_sync_en=false`) to preserve
  the *effective* legacy config. Recipe §Hardware configuration could add: "when an op resolves a
  ComputeKernelConfig but only partially wires it into a Metal descriptor, mirror the descriptor
  (Style B) — the resolved-but-unwired knobs did not take effect."

### Gaps
- **`unpack_modes` under FP32.** The legacy op set no `unpack_to_dest_mode` (defaulted everywhere). The Metal
  2.0 validator requires an explicit entry for every Float32 DFB a compute kernel *consumes* when
  `enable_32_bit_dest = true`. I add `UnpackMode::UnpackToSrc` (== legacy `Default`) for the 8 intermediates
  (Float32 whenever `fp32_dest_acc_en`) and, additionally, for the 4 consumed I/O DFBs when
  `cb_data_format == Float32`. `DX` (produced, not consumed) gets none. This is derived from the census, not
  guessed; the recipe's guidance on this was clear and sufficient.

## Open items for downstream

- **Cross-op kernel touches:** none. All three kernels are op-owned; shared `moreh_common.hpp` headers
  untouched (already `DataflowBuffer`-based).
- **Compute CTA `num_output_tiles` is a dead read.** `moreh_norm_backward_kernel.cpp` reads compile-time
  arg 0 into a `constexpr` that the kernel body never uses (the loop bound is the RTA
  `num_input_tiles_per_core`). Preserved faithfully as the per-group named CTA `num_output_tiles` (it *is*
  the value distinguishing COMPUTE_1 from COMPUTE_2, so it carries the work-split multiplicity; not demoted).
  The op owner may want to remove the dead read in a separate cleanup.
- **Reader dim blocks could be CRTAs / common varargs.** `output_grad_dim` / `input_grad_dim` /
  `need_bcast_dim` are identical on every node (computed once, not per-core), so they could be
  common-runtime varargs (`num_common_runtime_varargs` / `get_common_vararg`) rather than per-node runtime
  varargs. Kept as per-node runtime varargs to preserve legacy dispatch semantics (RTA→CRTA is a separate
  pass). Same for `decimal` (a scalar identical on every node — candidate CRTA).
- **`decimal_minus_one` unused local** in the factory (from
  `get_floored_p_and_decimal_and_p_is_negative(p - 1.0f)`) — pre-existing dead local, left untouched
  (audit "Misc anomalies").
- **Audit open item carried forward:** the readiness sheet `Is safe to port?` axis could not be fetched in
  the audit subagent session. The port proceeded on the user's explicit GREEN go-ahead; if the sheet
  disagrees, route to the readiness-sheet owner.

## Files created / modified

- `device/moreh_norm_backward_device_operation.hpp` — replaced the direct `create_descriptor` declaration
  with a nested `MorehNormBackwardProgramFactory { create_program_artifacts }` and
  `using program_factory_t = std::variant<...>`; swapped the `program_descriptors.hpp` include for
  `ttnn/metal_v2_artifacts.hpp` (+`<variant>`).
- `device/moreh_norm_backward_program_factory.cpp` — rewrote `create_descriptor`→`create_program_artifacts`:
  builds `ProgramSpec` (4 KernelSpecs, 13 DataflowBufferSpecs, 4 TensorParameters, 1–2 WorkUnitSpecs) +
  `ProgramRunArgs`; TensorBindings replace buffer-address RTAs; named CTAs/RTAs + reader runtime varargs;
  Style-B `ComputeGen1Config` with FP32 `unpack_modes`; DM configs via `create_reader/writer_datamovement_config`.
- `device/kernels/reader_moreh_norm_backward.cpp` — `+experimental/kernel_args.h`; `TensorAccessorArgs`/
  buffer-address RTAs → `TensorAccessor(tensor::input|output|output_grad)`; `ArgFetcher`/`get_compile_time_arg_val`
  → `get_arg(args::…)`; the three dim blocks → `get_vararg`; cb-id `DataflowBuffer`/`get_tile_size(cb)` →
  `dfb::…` + `dfb.get_tile_size()`.
- `device/kernels/writer_moreh_norm_backward.cpp` — `+experimental/kernel_args.h`; `TensorAccessorArgs`/
  buffer-address RTA → `TensorAccessor(tensor::input_grad)`; `get_arg_val` → `get_arg(args::…)`;
  cb-id → `dfb::input_grad` + `dfb.get_tile_size()`.
- `device/kernels/moreh_norm_backward_kernel.cpp` — `+experimental/kernel_args.h`; CTAs/RTAs →
  `get_arg(args::…)`; `DataflowBuffer` objects constructed from `dfb::…`; raw cb-id LLK call args
  (`binary_op_init_common`, `mul_tiles*`) → `dfb::…`.
- `METAL2_PORT_PLAN.md`, `METAL2_PORT_REPORT.md` — written by this port.
- (`METAL2_PREPORT_AUDIT.md`, `METAL2_PORT_BRIEF.md` — audit inputs, committed alongside.)

## Test commands (verification is the orchestrator's)

No C++ gtest coverage exists for this op. Python coverage (the no-regression baseline) is a single file
that also covers the forward op; run only the backward tests:

```bash
# From the checkout root, with the venv active (source python_env/bin/activate):
pytest tests/ttnn/nightly/unit_tests/operations/moreh/test_moreh_norm.py -v -k "norm_backward"
```

This selects `test_moreh_norm_backward` (dtype/shape/p sweeps),
`test_moreh_norm_backward_compute_kernel_options` (exercises `fp32_dest_acc_en` → the `unpack_modes` path),
and `test_moreh_norm_backward_callback` (program-cache hit → the MetalV2 tensor-arg refresh path). All three
must pass unchanged from pre-port.

Build (orchestrator): `./build_metal.sh --build-tests`.
