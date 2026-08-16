# Port Plan — `uniform`

Port plan for `ttnn/cpp/ttnn/operations/uniform`, ported from `ProgramDescriptor` to Metal 2.0.
Written during the inventory and planning steps; committed alongside the port for review.

## Legacy Inventory

### Legacy factory shape

- Concept: `ProgramDescriptorFactoryConcept` — reached via the **direct-descriptor** form:
  `create_descriptor` sits on `UniformDeviceOperation` itself, with **no** `program_factory_t`
  (`device/uniform_device_operation.hpp:39-42`, defined `device/uniform_program_factory.cpp:107`).
  The framework wraps it in its own `DirectDescriptorFactory`
  (`ttnn/api/ttnn/mesh_device_operation_adapter.hpp:170-196`).
- Variants: single. One factory, one `ProgramDescriptor`.
- Custom `compute_program_hash`: **none**. **Backdoor** (`attribute_names` / `attribute_values`) present at
  `device/uniform_device_operation.hpp:28-29` — **left intact.** It lists only `memory_config` and
  `compute_kernel_config`, deliberately excluding `from` / `to` / `seed`; that exclusion is safe *only because*
  `override_runtime_arguments` re-applies all three on every cache hit. The two are a matched pair.
- `override_runtime_arguments`: present at `device/uniform_program_factory.cpp:213-247`
  (declared `device/uniform_device_operation.hpp:51-56`), returning `void` and mutating the cached `Program`.

*(The Metal 2.0 factory concept the port targets was chosen during the audit — `CustomProgramSpecFactoryConcept`.
Carried forward in the [TTNN ProgramFactory](#ttnn-programfactory) section below.)*

### Kernels

| unique_id | source | core_ranges | CTAs (positional) | CTAs (named) | RTAs | CRTAs | defines | opt_level | config |
|---|---|---|---|---|---|---|---|---|---|
| writer | `device/kernels/writer_uniform.cpp` (**lent** — also bound by `rand`) | `all_cores` (from `split_work_to_cores`) | `[0] intermed_cb_id = CBIndex::c_24`, `[1] dst_cb_id = CBIndex::c_0`, `[2..] TensorAccessorArgs(output.buffer())` @ `:163` | none | per core: `{output.buffer() (annotated Buffer*), tile_offset, units_per_core}` @ `:204` | none | `OUTPUT_DTYPE_BFLOAT16=1` **or** `OUTPUT_DTYPE_FLOAT32=1` (switch on `output.dtype()`, `default: break;`) @ `:156-160` | absent → resolved **O2** (DM) | `WriterConfigDescriptor{}` @ `:171` |
| compute | `device/kernels/compute_uniform.cpp` (**lent** — also bound by `rand`) | `all_cores` | `[0] intermed_cb_id = CBIndex::c_24` @ `:179` | none | per core: `{seed, f2u_from, f2u_to, tile_offset, units_per_core}` @ `:201-202` | none | none | absent → resolved **O3** (compute) | `ComputeConfigDescriptor{.math_fidelity, .fp32_dest_acc_en = true (hardcoded), .dst_full_sync_en, .math_approx_mode}` @ `:180-186` |

Notes:
- `grep -n opt_level` over the op directory returns **nothing** — neither `KernelDescriptor` sets the field, so
  both resolve to their legacy per-kernel-type defaults (O2 DM / O3 compute).
- The compute config is a **hybrid**: `get_compute_kernel_config_args(...)` @ `:127-128` resolves a TTNN
  `ComputeKernelConfig`, but the descriptor then **overrides** `fp32_dest_acc_en` to a hardcoded `true` and never
  uses `packer_l1_acc`. So it is neither a clean Style A nor a clean Style B (see [Planned Spec Shape](#planned-spec-shape)).
- The writer's `default: break;` arm emits **no** dtype define for an out-of-range dtype, compiling a writer whose
  loop body does no NOC write. Unreachable today (`validate_inputs` constrains the dtype). Carried across verbatim.

### CBs

| index | total_size | core_ranges | data_format | page_size | tile (if set) |
|---|---|---|---|---|---|
| `c_24` (intermed) @ `:133-141` | `2 * tile_size(Float32)` | `all_cores` | `tt::DataFormat::Float32` | `tile_size(Float32)` | not set |
| `c_0` (dst) @ `:144-152` | `1 * tile_size(out_data_format)` | `all_cores` | `datatype_to_dataformat_converter(output.dtype())` | `tile_size(out_data_format)` | not set |

No `GlobalCircularBuffer`, no `address_offset`, single-element `format_descriptors` on both (no aliasing).

### Semaphores

none — `grep -i semaphore` over the op directory returns nothing.

### Tensor accessors

| host site (file:line) | originating Tensor | RTA slot (host) |
|---|---|---|
| `device/uniform_program_factory.cpp:163` (`TensorAccessorArgs(output.buffer()).append_to(writer_ct_args)`) | `output` (= the input tensor; `uniform` is in-place, `device/uniform_device_operation.cpp:31-35`) | writer RTA 0 — as an annotated `Buffer*` on cache miss @ `:204`, as `output.buffer()->address()` on cache hit @ `:228,:243` |

Device side: `constexpr auto dst_args = TensorAccessorArgs<2>()` @ `device/kernels/writer_uniform.cpp:17`,
`TensorAccessor(dst_args, dst_addr)` @ `:24` (**two**-arg — no page-size third argument).

### Work split

- Driver: `split_work_to_cores(grid, units_to_divide)` @ `device/uniform_program_factory.cpp:44-45`, wrapped by
  `uniform_work_split()` @ `:41-55`; per-core assignment in `uniform_core_layout()` @ `:65-82`.
- `units_to_divide` = `output.physical_volume() / constants::TILE_HW`
- num_cores / all_cores / core_group_1 / core_group_2 / units_per_core_group_1 / units_per_core_group_2 — all
  structured-bound from `split_work_to_cores`; the core list is `grid_to_cores(num_cores, grid.x, grid.y)`.
- **Both kernels are declared over `all_cores` with a single `KernelDescriptor` each** — the per-group unit count
  travels as an **RTA** (`units_per_core`), not a per-group CTA. There is therefore **no** multi-`KernelDescriptor`
  work-split multiplicity to preserve.

### Shared kernels

Both kernels are **lent**: they live in `uniform`'s own directory (inside the writeable surface) but `rand` binds
them by file path. Census run per the caution's procedure and each hit checked to be a real kernel-source binding:

```
grep -rl writer_uniform.cpp  ttnn/cpp/ttnn/operations/   → uniform factory, rand factory (+ the two METAL2_*.md)
grep -rl compute_uniform.cpp ttnn/cpp/ttnn/operations/   → uniform factory, rand factory (+ the two METAL2_*.md)
```

| Kernel | Kind | Other binder | `_metal2` fork beside it? | Rung |
|---|---|---|---|---|
| `device/kernels/writer_uniform.cpp` | lent | `rand` — `ttnn/cpp/ttnn/operations/rand/device/rand_program_factory.cpp:27` (path const), bound at `:165` | **no** — this port creates the first | **2 — create the fork** |
| `device/kernels/compute_uniform.cpp` | lent | `rand` — `.../rand_program_factory.cpp:28` (path const), bound at `:181` | **no** — this port creates the first | **2 — create the fork** |

`{rand}` is a **sunset list, not authorization** to convert in place; `rand` carries the identical readiness-sheet
profile and sits under the same family-wide hold, so it cannot co-migrate today. Since `rand` inherits this fork's
binding vocabulary at sunset, bindings are named for the **kernel's** role vocabulary, not `uniform`'s locals.

### Flags

- No unreferenced kernel files in the op directory.
- Every descriptor type the legacy factory uses (`CBDescriptor`, `KernelDescriptor`, `WriterConfigDescriptor`,
  `ComputeConfigDescriptor`, `ProgramDescriptor`) maps onto an audit Appendix A entry. No stop signal.
- `uniform` is **in-place**: `create_output_tensors` returns the input tensor, so the op has exactly **one**
  tensor and one `TensorParameter`.

## TTNN ProgramFactory

- **Concept (inherited from audit)**: `CustomProgramSpecFactoryConcept` — selected by the presence of
  `override_runtime_arguments`, which the port **translates** (into a `ProgramRunArgs`-returning method), never
  deletes.
- **Custom `compute_program_hash`**: none; **backdoor** hash present at `device/uniform_device_operation.hpp:28-29`
  — **left intact, byte-identical.**
- **Implementation notes** — one structural consequence the audit did not have to name:

  The legacy op reaches `ProgramDescriptorFactoryConcept` through the framework's **direct-descriptor** shortcut
  (`HasDirectDescriptor` = `create_descriptor` present *and* `program_factory_t` absent). There is **no spec-path
  analog** of that shortcut: `resolve_program_factory` synthesises a `DirectDescriptorFactory` only for
  `create_descriptor`, and `DeviceOperationConcept` requires `HasDirectDescriptor || HasProgramFactoryType`. So the
  moment `create_descriptor` goes away, the op *must* grow a `program_factory_t`.

  The port therefore introduces a nested `UniformDeviceOperation::ProgramFactory` struct carrying both
  `create_program_artifacts` and `override_runtime_arguments`, plus
  `using program_factory_t = std::variant<ProgramFactory>;`. This is a device-op-**header** edit that the port
  forces and that the recipe's two documented exceptions do not cover — flagged in the port report. Nothing else
  in the device-op class changes: `validate_inputs`, `validate_on_program_cache_miss`, `compute_output_specs`,
  `create_output_tensors`, `operation_attributes_t`, `tensor_args_t` and the backdoor hash are untouched.

  No pybind change is forced: `uniform_nanobind.cpp` binds only the user-facing `ttnn::uniform` — it never
  referenced `create_descriptor`.

## Planned Spec Shape

Default 1:1 with legacy.

- **KernelSpecs** (2):
  - `WRITER` (`"writer"`) → `device/kernels/writer_uniform_metal2.cpp` (**new fork**)
  - `COMPUTE` (`"compute"`) → `device/kernels/compute_uniform_metal2.cpp` (**new fork**)
- **DataflowBufferSpecs** (2 — one per legacy `CBDescriptor`, no aliasing, no borrowed memory):
  - `INTERMED` (`"intermed"`): `entry_size = tile_size(Float32)`, `num_entries = 2`,
    `data_format_metadata = Float32`
  - `DST` (`"dst"`): `entry_size = tile_size(out_data_format)`, `num_entries = 1`,
    `data_format_metadata = out_data_format`
- **SemaphoreSpecs**: none — legacy declares no semaphores.
- **TensorParameters** (1): `OUTPUT` (`"dst"`), spec from the output `MeshTensor`'s `tensor_spec()`.
  `uniform` is in-place, so this is the op's only tensor.
- **WorkUnitSpecs** (1): `{"main", kernels = {WRITER, COMPUTE}, target_nodes = all_cores}` — both legacy
  `KernelDescriptor`s carry the identical `core_ranges`, so one work unit covers them.
- **Op-owned tensors**: none — the legacy factory allocates no device tensors beyond the op's io.

### DFB endpoint assignment (re-derived from the kernel-touch census, not transcribed)

| DFB | Distinct touchers on a node | Tags | Assignment |
|---|---|---|---|
| `INTERMED` | 2 — compute, writer | compute **locked producer** (`reserve_back` `compute_uniform.cpp:31`, `push_back` `:41`); writer **locked consumer** (`wait_front` `writer_uniform.cpp:36`, `pop_front` `:49`/`:64`) | **1P + 1C** — plain 1:1, no flag |
| `DST` | 1 — writer only (`reserve_back` `:32`, `get_write_ptr` `:33`, `push_back` `:78`) | writer locked producer, no consumer anywhere | **self-loop** — writer bound PRODUCER *and* CONSUMER (one shared `accessor_name`) |

Census agrees with the brief in both rows. No `allow_instance_multi_binding` anywhere; no dead CB (`DST` is
reserved / peeked / pushed in **both** dtype configs, and its entry size is read every iteration).

### Hardware configuration

- **`WRITER`** — legacy `WriterConfigDescriptor{}` resolves to the writer default triple
  (`RISCV_0` / `NOC_1` / `DM_DEDICATED_NOC`) → `ttnn::create_writer_datamovement_config(device->arch())`.
- **`COMPUTE`** — the legacy config is a hybrid (TTNN-resolved values, then a hardcoded override), so it is
  ported as **Style B**: a `ComputeGen1Config` built field-by-field. Routing it through
  `to_compute_hardware_config` would silently restore the user's `fp32_dest_acc_en` in place of the op's
  deliberate `true`, and flip any field the op left at a Metal default onto the helper's high-performance one.

  | legacy `ComputeConfigDescriptor` field | value | Metal 2.0 `ComputeGen1Config` |
  |---|---|---|
  | `math_fidelity` | resolved from `compute_kernel_config` | `fpu_math_fidelity` — 1:1 |
  | `math_approx_mode` (bool) | resolved from `compute_kernel_config` | `sfpu_precision_mode` — `true → Approximate`, `false → Precise` |
  | `fp32_dest_acc_en` | **hardcoded `true`** @ `:182` | `enable_32_bit_dest = true` — 1:1 |
  | `dst_full_sync_en` | resolved from `compute_kernel_config` | `double_buffer_dest = !dst_full_sync_en` — **inverted** |
  | `unpack_to_dest_mode` | **unset** (legacy default: all `Default`) | `unpack_modes` — **left empty** |
  | `bfp8_pack_precise` | **unset** (`false`) | `bfp_pack_precision_mode` — left at its `Approximate` default (defaults coincide) |

  **`unpack_modes` stays empty, and the FP32 required-entry rule does not fire.** The validator's rule is scoped
  to `binding.endpoint_type == CONSUMER` (`tt_metal/impl/metal2_host_api/program_spec.cpp:1056-1078`), and the
  compute kernel binds `INTERMED` as **PRODUCER** only. Legacy set no `unpack_to_dest_mode`, so an empty table
  reproduces the legacy all-`Default` vector byte-for-byte.

  `packer_l1_acc` is destructured from the TTNN config and never used — the same in the port as in legacy
  (routed to the report as a pre-existing finding, not changed here).

### Compiler options

| KernelSpec | legacy resolved `opt_level` | Metal 2.0 |
|---|---|---|
| `WRITER` (DM) | `O2` (absent field on a DM descriptor) | left at `CompilerOptions`' `O2` default — no action |
| `COMPUTE` | **`O3`** (absent field on a `ComputeConfigDescriptor`) | **explicit** `.opt_level = KernelBuildOptLevel::O3` |

`WRITER::compiler_options.defines` carries the `OUTPUT_DTYPE_*` define, built with the same `switch` (including
its `default: break;`) as legacy.

## Preserved Multiplicity

none — no work-split multiplicity in legacy. Both kernels are a single `KernelDescriptor` over `all_cores`; the
per-core unit count already travels as an RTA in legacy, so nothing is demoted and nothing is duplicated.

## Dropped Plumbing

| legacy location (file:line) | legacy form | Metal 2.0 replacement |
|---|---|---|
| `uniform_program_factory.cpp:162` (writer CTA slot 0) | `intermed_cb_id` (magic CB index `CBIndex::c_24`) | `DFBBinding{INTERMED, "intermed", CONSUMER}` on `WRITER` |
| `uniform_program_factory.cpp:162` (writer CTA slot 1) | `dst_cb_id` (magic CB index `CBIndex::c_0`) | `DFBBinding{DST, "dst", PRODUCER}` + `{DST, "dst", CONSUMER}` on `WRITER` (self-loop) |
| `uniform_program_factory.cpp:163` (writer CTA slots 2..) | `TensorAccessorArgs(output.buffer()).append_to(writer_ct_args)` | `TensorBinding{OUTPUT, "dst"}` on `WRITER`; kernel builds `TensorAccessor(tensor::dst)` |
| `uniform_program_factory.cpp:179` (compute CTA slot 0) | `intermed_cb_id` (magic CB index) | `DFBBinding{INTERMED, "intermed", PRODUCER}` on `COMPUTE` |
| `uniform_program_factory.cpp:204` (writer RTA slot 0, cache miss) | `output.buffer()` — an annotated `Buffer*` (the framework's interim `BufferBinding`) | `TensorParameter OUTPUT` + `TensorArgument` in `ProgramRunArgs::tensor_args` |
| `uniform_program_factory.cpp:228,243` (writer RTA slot 0, cache hit) | `output.buffer()->address()` written into `writer_args[0]` | the same `TensorArgument`, now returned from the translated `override_runtime_arguments` |
| `writer_uniform.cpp:15,16` | `get_compile_time_arg_val(0)` / `(1)` → `intermed_cb_id` / `dst_cb_id` | `dfb::intermed` / `dfb::dst` |
| `writer_uniform.cpp:17` | `constexpr auto dst_args = TensorAccessorArgs<2>()` | dropped — binding token carries the layout metadata |
| `writer_uniform.cpp:19` | `uint32_t dst_addr = get_arg_val<uint32_t>(0)` | dropped — binding token carries the base address |
| `writer_uniform.cpp:20,21` | `get_arg_val<uint32_t>(1)` / `(2)` | `get_arg(args::start_id)` / `get_arg(args::num_tiles)` (re-indexed away entirely) |
| `writer_uniform.cpp:26` | `get_local_cb_interface(dst_cb_id).fifo_page_size` | `dfb_dst.get_entry_size()` — CB→DFB whitelist §B |
| `compute_uniform.cpp:11` | `get_compile_time_arg_val(0)` → `intermed_cb_id` | `dfb::intermed` |
| `compute_uniform.cpp:13,18,19,21,22` | `get_arg_val<uint32_t>(0..4)` | `get_arg(args::seed)`, `args::f2u_from`, `args::f2u_to`, `args::start_id`, `args::num_tiles` |

**Page-size 3rd-argument CTAs/RTAs:** none — the op's single accessor is already 2-arg.
**Semaphore-ID RTAs:** none — the op declares no semaphores.
**Positional CTAs:** both kernels' entire positional CTA lists are consumed by the rows above; post-port both
`KernelSpec::compile_time_args` tables are **empty**.

## Applied Patterns

- [Sync-free and single-ended CBs → self-loop DFB](../../../../../docs/source/tt-metalium/tt_metal/apis/host_apis/metal_2.0/ai/shared/port_patterns.md#pattern-sync-free-and-single-ended-cbs--self-loop-dfb):
  `DST` on the `WRITER` KernelSpec, bound both PRODUCER and CONSUMER under one `accessor_name`. One toucher, a
  DM kernel — legal on Gen1, Quasar-uplift debt.
- [Caution: Porting a shared kernel](../../../../../docs/source/tt-metalium/tt_metal/apis/host_apis/metal_2.0/ai/shared/port_patterns.md#caution-porting-a-shared-kernel):
  **rung 2** on both kernels — create `writer_uniform_metal2.cpp` / `compute_uniform_metal2.cpp` beside the
  originals, convert the copies, add the pointer comment to each original, leave `rand`'s binding untouched.
- [Pass DFB handles directly to LLKs](../../../../../docs/source/tt-metalium/tt_metal/apis/host_apis/metal_2.0/ai/shared/port_patterns.md#pattern-pass-dfb-handles-directly-to-llks-and-kernel-lib-helpers):
  `init_sfpu(dfb::intermed, dfb::intermed)` and `pack_tile(0, dfb::intermed, 0)` in the compute fork.

Explicitly **not** applied: conditional/optional DFB bindings (both DFBs are bound on every path — the
`OUTPUT_DTYPE_*` define selects *code*, not *bindings*), aliased DFBs, same-FIFO aliasing, multi-variant
factories, multi-binding, varargs, op-owned tensors.

## Deferred / Flagged

- **New finding (structural, planning step):** the direct-descriptor → `program_factory_t` restructure described
  under [TTNN ProgramFactory](#ttnn-programfactory). Not a blocker; the recipe's device-op-class exception list
  simply does not name it. Routed to the port report.
- **New finding (doc, planning step):** the brief / audit record that `DataflowBuffer` has "no direct
  `fifo_page_size` analog" and infer `get_tile_size()`. The CB→DFB whitelist §B maps it directly to
  `get_entry_size()`, which on a DM build is an exact identity (`cb_addr_shift == 0` off-TRISC, so
  `address_units_to_bytes` is the identity function). The whitelist answer is used. Routed to the port report.
- No feature gate fired that the audit missed. No host-computed base-pointer offset. No GlobalCircularBuffer.
  No `get_cb_tiles_*_ptr`. No out-of-op call site demanding `sem::` or `tensor::`.
