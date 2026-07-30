# Port Plan — `ttnn/cpp/ttnn/operations/reduction/manual_seed`

Port plan for `manual_seed`, ported from the legacy `ProgramDescriptor` API to Metal 2.0.
Written during the inventory and planning steps; committed alongside the port for review.

Scope: **all four factories in one change.** The audit cleared every factory, nothing is blocked,
and the one intra-op shared kernel (`manual_seed_set_seed.cpp`, bound by factories 1 and 2)
converts in place only because both of its binders convert together — see
[Shared kernels](#shared-kernels).

## Legacy Inventory

### Legacy factory shape

- Concept: `ProgramDescriptorFactoryConcept` — all four factories define
  `static ProgramDescriptor create_descriptor(const ManualSeedParams&, const ManualSeedInputs&, Tensor&)`
  ([manual_seed_program_factory.hpp:19,24,29,34](device/manual_seed_program_factory.hpp#L19-L34)).
- Variants: four factories in one `program_factory_t` variant
  ([manual_seed_operation.hpp:21-25](device/manual_seed_operation.hpp#L21-L25)), selected by
  `select_program_factory` ([manual_seed_operation.cpp:22-49](device/manual_seed_operation.cpp#L22-L49)):

  | # | Factory | Selected when | Defined at |
  |---|---|---|---|
  | 1 | `ManualSeedSingleSeedToAllCoresProgramFactory` | scalar seed, no user ids | [manual_seed_program_factory.cpp:58](device/manual_seed_program_factory.cpp#L58-L80) |
  | 2 | `ManualSeedSingleSeedSingleCoreProgramFactory` | scalar seed, scalar user id | [manual_seed_program_factory.cpp:82](device/manual_seed_program_factory.cpp#L82-L106) |
  | 3 | `ManualSeedSingleSeedSetCoresProgramFactory` | scalar seed, `user_ids` tensor | [manual_seed_program_factory.cpp:108](device/manual_seed_program_factory.cpp#L108-L174) |
  | 4 | `ManualSeedSetSeedsSetCoresProgramFactory` | `seeds` tensor, `user_ids` tensor | [manual_seed_program_factory.cpp:176](device/manual_seed_program_factory.cpp#L176-L249) |

  These are four separate factory *classes*, not four branches of one factory, so each gets its own
  `create_program_artifacts`. The "multi-variant factory" catalog pattern (one factory branching on an
  attribute) does not apply.

- Custom `compute_program_hash`: **none** — the device-operation
  ([manual_seed_operation.hpp:16-34](device/manual_seed_operation.hpp#L16-L34)) declares no such method,
  so the port keeps the default reflection-based hash. Nothing to delete.

Two host helpers are shared by the factories and live in the file's anonymous namespace:

- `compute_core_grid` @ [manual_seed_program_factory.cpp:20-36](device/manual_seed_program_factory.cpp#L20-L36)
  — full device grid, overridden by `operation_attributes.sub_core_grids` when present. Untouched by the port
  (it produces a `CoreRangeSet`, which *is* a `NodeRangeSet`).
- `push_tensor_circular_buffer` @ [manual_seed_program_factory.cpp:39-54](device/manual_seed_program_factory.cpp#L39-L54)
  — pushes a one-entry `CBDescriptor` sized to a tensor's tile. This is legacy CB API and is replaced by a
  `DataflowBufferSpec`-returning helper.

---

### Variant: factory 1 — `SingleSeedToAllCores`

#### Kernels

| unique_id | source | core_ranges | CTAs (positional) | CTAs (named) | RTAs | CRTAs | defines | opt_level | config |
|---|---|---|---|---|---|---|---|---|---|
| compute | `device/kernels/compute/manual_seed_set_seed.cpp` | `core_grid` (full device grid or `sub_core_grids`) | `{seeds.value_or(0)}` | none | none | none | none | **O3** (resolved; field absent on a `ComputeConfigDescriptor`) | `ComputeConfigDescriptor{}` — all fields default |

#### CBs

none — the factory pushes no `CBDescriptor`.

#### Semaphores

none.

#### Tensor accessors

none — `tensor_args` is unused ([manual_seed_program_factory.cpp:59](device/manual_seed_program_factory.cpp#L59)).

#### Work split

n/a — no `split_work_to_cores`; one kernel over the whole grid.

---

### Variant: factory 2 — `SingleSeedSingleCore`

#### Kernels

| unique_id | source | core_ranges | CTAs (positional) | CTAs (named) | RTAs | CRTAs | defines | opt_level | config |
|---|---|---|---|---|---|---|---|---|---|
| compute | `device/kernels/compute/manual_seed_set_seed.cpp` | `chosen_core_ranges` — the single core `cores.at(user_ids.value_or(0))` ([:88-90](device/manual_seed_program_factory.cpp#L88-L90)) | `{seeds.value_or(0)}` | none | none | none | none | **O3** (resolved) | `ComputeConfigDescriptor{}` — all fields default |

#### CBs / Semaphores / Tensor accessors

none of each. `tensor_args` unused ([manual_seed_program_factory.cpp:83](device/manual_seed_program_factory.cpp#L83)).

#### Work split

n/a — single core, chosen by index into the grid's core list.

---

### Variant: factory 3 — `SingleSeedSetCores`

#### Kernels

| unique_id | source | core_ranges | CTAs (positional) | CTAs (named) | RTAs | CRTAs | defines | opt_level | config |
|---|---|---|---|---|---|---|---|---|---|
| reader | `device/kernels/dataflow/reader_manual_seed_read_user_id.cpp` | `core_grid` | `{user_ids_cb_index(=c_0), kernel_communication_cb_index(=c_1), number_of_ids}` then `TensorAccessorArgs(user_ids_mesh).append_to(...)` @ [:143](device/manual_seed_program_factory.cpp#L143) | none | per core: `{user_ids_mesh, core_id}` @ [:157](device/manual_seed_program_factory.cpp#L157) | none | none | **O2** (resolved; DM default) | `ReaderConfigDescriptor{}` → `(RISCV_1, NOC_0, DM_DEDICATED_NOC)` |
| compute | `device/kernels/compute/manual_seed_single_seed_receive_user_id.cpp` | `core_grid` | `{kernel_communication_cb_index(=c_1), seeds.value_or(0)}` | none | none | none | none | **O3** (resolved) | `ComputeConfigDescriptor{}` — all fields default |

The reader's first RTA slot is a `MeshTensor` reference, not a raw address — `emplace_runtime_args` registers
it as a framework-patched buffer binding. It still reaches the kernel as a `uint32_t` base address at
`get_arg_val<uint32_t>(0)` ([reader_manual_seed_read_user_id.cpp:16](device/kernels/dataflow/reader_manual_seed_read_user_id.cpp#L16)).

#### CBs

| index | total_size | core_ranges | data_format | page_size | tile (if set) |
|---|---|---|---|---|---|
| `c_0` (`user_ids`) | `user_ids` tile size (UINT32 → 4096 B) | `core_grid` | from `user_ids.dtype()` | same as total_size | **not set** |
| `c_1` (`kernel_communication`) | `user_ids` tile size | `core_grid` | from `user_ids.dtype()` | same as total_size | **not set** |

Both are built by `push_tensor_circular_buffer` from the *`user_ids`* tensor
([:128,131](device/manual_seed_program_factory.cpp#L128-L131)); `total_size == page_size` in both, so each CB is
exactly one entry. No `.tile` is ever set, so there is no `tile_format_metadata` to carry over.

#### Semaphores

none.

#### Tensor accessors

| host site (file:line) | originating Tensor | RTA slot (host) |
|---|---|---|
| [manual_seed_program_factory.cpp:143](device/manual_seed_program_factory.cpp#L143) (`TensorAccessorArgs(user_ids_mesh).append_to`) | `tensor_args.user_ids` | reader RTA slot 0 (`user_ids_mesh` @ [:157](device/manual_seed_program_factory.cpp#L157)) |

Device side: `TensorAccessorArgs<3>()` @ [reader_manual_seed_read_user_id.cpp:23](device/kernels/dataflow/reader_manual_seed_read_user_id.cpp#L23),
consumed by `TensorAccessor(args, addr)` @ [:30](device/kernels/dataflow/reader_manual_seed_read_user_id.cpp#L30). Two-argument
construction — no page-size third argument.

#### Work split

n/a — both kernels cover the whole grid; the per-core RTA `core_id` is the core's index in
`corerange_to_cores(core_grid, num_cores, true)`, not a work-split count.

---

### Variant: factory 4 — `SetSeedsSetCores`

#### Kernels

| unique_id | source | core_ranges | CTAs (positional) | CTAs (named) | RTAs | CRTAs | defines | opt_level | config |
|---|---|---|---|---|---|---|---|---|---|
| reader | `device/kernels/dataflow/reader_manual_seed_read_all_data.cpp` | `core_grid` | `{user_ids_cb_index(=c_0), seeds_cb_index(=c_1), kernel_communication_cb_index(=c_2), number_of_ids}` then two `TensorAccessorArgs(...).append_to(...)` @ [:220-221](device/manual_seed_program_factory.cpp#L220-L221) | none | per core: `{user_ids_mesh, seeds_mesh, core_id}` @ [:235](device/manual_seed_program_factory.cpp#L235) | none | none | **O2** (resolved; DM default) | `ReaderConfigDescriptor{}` → `(RISCV_1, NOC_0, DM_DEDICATED_NOC)` |
| compute | `device/kernels/compute/manual_seed_receive_all_data.cpp` | `core_grid` | `{kernel_communication_cb_index(=c_2)}` | none | none | none | none | **O3** (resolved) | `ComputeConfigDescriptor{}` — all fields default |

This factory's compute kernel takes **no** `seed` CTA — the seed travels through the
`kernel_communication` buffer as the reader's second word.

#### CBs

| index | total_size | core_ranges | data_format | page_size | tile (if set) |
|---|---|---|---|---|---|
| `c_0` (`user_ids`) | `user_ids` tile size | `core_grid` | from `user_ids.dtype()` | same as total_size | **not set** |
| `c_1` (`seeds`) | `seeds` tile size | `core_grid` | from `seeds.dtype()` | same as total_size | **not set** |
| `c_2` (`kernel_communication`) | `seeds` tile size | `core_grid` | from `seeds.dtype()` | same as total_size | **not set** |

`c_1` and `c_2` are both built from the *`seeds`* tensor ([:204,207](device/manual_seed_program_factory.cpp#L204-L207)).

#### Semaphores

none.

#### Tensor accessors

| host site (file:line) | originating Tensor | RTA slot (host) |
|---|---|---|
| [manual_seed_program_factory.cpp:220](device/manual_seed_program_factory.cpp#L220) | `tensor_args.user_ids` | reader RTA slot 0 |
| [manual_seed_program_factory.cpp:221](device/manual_seed_program_factory.cpp#L221) | `tensor_args.seeds` | reader RTA slot 1 |

Device side: `TensorAccessorArgs<4>()` chained into
`TensorAccessorArgs<...next_compile_time_args_offset()>()` @
[reader_manual_seed_read_all_data.cpp:25-27](device/kernels/dataflow/reader_manual_seed_read_all_data.cpp#L25-L27);
both accessors are constructed with two arguments @ [:34,37](device/kernels/dataflow/reader_manual_seed_read_all_data.cpp#L34-L37).

#### Work split

n/a — as factory 3.

---

### Shared kernels

| kernel | sharing kind | `_metal2` fork beside it? | rung |
|---|---|---|---|
| `device/kernels/compute/manual_seed_set_seed.cpp` | **intra-op** — bound by factory 1 @ [:68-69](device/manual_seed_program_factory.cpp#L68-L69) and factory 2 @ [:94-95](device/manual_seed_program_factory.cpp#L94-L95) | no | **3 — convert in place** |

Census re-run for all five kernel filenames across `ttnn/cpp/ttnn/operations/`: the only binding hit for each is
this op's own `manual_seed_program_factory.cpp`. No borrowed kernels, no lent kernels, no cross-op consumers,
so there is no sunset list.

Rung 3 (in-place conversion) is legal here because the invoker assigned the whole op and **both** binders of
`manual_seed_set_seed.cpp` convert in this same change — the condition the rung requires. The two `KernelSpec`s
differ only in target nodes and both pass the same single `seed` CTA, so one converted source serves both.

### Flags

- No unreferenced kernel files — all five sources in `device/kernels/` are bound by a factory.
- No `GlobalCircularBuffer` / "remote CB" anywhere; no `set_globally_allocated_address`; no borrowed-memory CBs.
- No semaphores, no varargs, no `override_runtime_arguments`, no `get_dynamic_runtime_args`, no op-owned tensors.
- The op returns a dummy 1-element output tensor purely because the device-operation framework has no void
  return ([manual_seed_operation.cpp:104-117](device/manual_seed_operation.cpp#L104-L117)). No kernel touches it, so it
  becomes **no** `TensorParameter` in any factory.

## TTNN ProgramFactory

- **Concept (inherited from audit)**: `ProgramSpecFactoryConcept` — uniform across all four factories.
- **Custom `compute_program_hash`**: none — already the default reflection-based hash.
- **Implementation notes**:
  - Each factory's `static ProgramDescriptor create_descriptor(...)` becomes
    `static ttnn::device_operation::ProgramArtifacts create_program_artifacts(const ManualSeedParams&, const ManualSeedInputs&, Tensor&)`.
    All four flip together, so the `program_factory_t` variant lands entirely on the new concept.
  - No pybind entry point to remove: `manual_seed_nanobind.cpp` exposes only the top-level `manual_seed`
    function ([manual_seed_nanobind.cpp:73-81](manual_seed_nanobind.cpp#L73-L81)), never `create_descriptor`.
  - `operation_attributes.device` supplies `arch()` for the generation-agnostic DM config helper.

## Planned Spec Shape

### Variant: factory 1 — `SingleSeedToAllCores`

- **KernelSpecs**: 1 — `COMPUTE` (`manual_seed_set_seed.cpp`), `compile_time_args = {{"seed", …}}`,
  `hw_config = ComputeGen1Config{}` (all defaults), `compiler_options.opt_level = O3`.
- **DataflowBufferSpecs**: none.
- **SemaphoreSpecs**: none.
- **TensorParameters**: none.
- **WorkUnitSpecs**: 1 — `{COMPUTE}` over `core_grid`.
- **Op-owned tensors**: none.

A `ProgramSpec` with no DFBs and no tensor parameters is the legitimate minimal shape here, not an omission.

### Variant: factory 2 — `SingleSeedSingleCore`

Identical to factory 1, except the single `WorkUnitSpec` targets `chosen_core_ranges` (one node).

### Variant: factory 3 — `SingleSeedSetCores`

- **KernelSpecs**: 2
  - `READER` (`reader_manual_seed_read_user_id.cpp`): `compile_time_args = {{"number_of_ids", …}}`,
    `runtime_arg_schema.runtime_arg_names = {"core_id"}`,
    `dfb_bindings` = `USER_IDS` PRODUCER + `USER_IDS` CONSUMER (self-loop, accessor `"user_ids"`) and
    `KERNEL_COMMUNICATION` PRODUCER (accessor `"kernel_communication"`),
    `tensor_bindings` = `USER_IDS_TENSOR` (accessor `"user_ids"`),
    `hw_config = create_reader_datamovement_config(arch)`, `opt_level` left at the `O2` default.
  - `COMPUTE` (`manual_seed_single_seed_receive_user_id.cpp`): `compile_time_args = {{"seed", …}}`,
    `dfb_bindings` = `KERNEL_COMMUNICATION` CONSUMER (accessor `"kernel_communication"`),
    `hw_config = ComputeGen1Config{}`, `opt_level = O3`.
- **DataflowBufferSpecs**: 2 — `USER_IDS` and `KERNEL_COMMUNICATION`, each
  `entry_size = <user_ids tile size>`, `num_entries = 1`,
  `data_format_metadata = <user_ids data format>`, no `tile_format_metadata` (legacy never set `.tile`).
- **SemaphoreSpecs**: none.
- **TensorParameters**: 1 — `USER_IDS_TENSOR`, spec = `user_ids.tensor_spec()`, no relaxations.
- **WorkUnitSpecs**: 1 — `{READER, COMPUTE}` over `core_grid`.

### Variant: factory 4 — `SetSeedsSetCores`

- **KernelSpecs**: 2
  - `READER` (`reader_manual_seed_read_all_data.cpp`): `compile_time_args = {{"number_of_ids", …}}`,
    `runtime_arg_schema.runtime_arg_names = {"core_id"}`,
    `dfb_bindings` = `USER_IDS` PRODUCER + CONSUMER (self-loop), `SEEDS` PRODUCER + CONSUMER (self-loop),
    `KERNEL_COMMUNICATION` PRODUCER,
    `tensor_bindings` = `USER_IDS_TENSOR` (accessor `"user_ids"`) and `SEEDS_TENSOR` (accessor `"seeds"`),
    `hw_config = create_reader_datamovement_config(arch)`, `opt_level` default `O2`.
  - `COMPUTE` (`manual_seed_receive_all_data.cpp`): **no** compile-time args,
    `dfb_bindings` = `KERNEL_COMMUNICATION` CONSUMER, `hw_config = ComputeGen1Config{}`, `opt_level = O3`.
- **DataflowBufferSpecs**: 3 — `USER_IDS` (sized/formatted from `user_ids`), `SEEDS` and
  `KERNEL_COMMUNICATION` (both sized/formatted from `seeds`, matching legacy), each one entry.
- **SemaphoreSpecs**: none.
- **TensorParameters**: 2 — `USER_IDS_TENSOR` and `SEEDS_TENSOR`, specs from their tensors, no relaxations.
- **WorkUnitSpecs**: 1 — `{READER, COMPUTE}` over `core_grid`.

### DFB endpoint census (re-derived, not transcribed)

Per node, counting the **distinct kernels** that touch each buffer, and how:

| factory | DFB | reader touches | compute touches | distinct touchers | disposition |
|---|---|---|---|---|---|
| 3 | `USER_IDS` | `reserve_back` + `get_write_ptr` + NoC read destination | — (the compute kernel receives only the `kernel_communication` index, so it structurally cannot reach this buffer) | 1 | **self-loop** (PRODUCER + CONSUMER on the reader) |
| 3 | `KERNEL_COMMUNICATION` | `reserve_back` + `get_write_ptr` + `push_back` → locked producer | `wait_front` + `read_tile_value` + `pop_front` → locked consumer | 2, one per role | **1P + 1C** |
| 4 | `USER_IDS` | as above | — | 1 | **self-loop** |
| 4 | `SEEDS` | as above | — | 1 | **self-loop** |
| 4 | `KERNEL_COMMUNICATION` | locked producer | locked consumer | 2, one per role | **1P + 1C** |

No DFB has three or more touchers and none has two kernels locked to the same FIFO role, so
**`allow_instance_multi_binding` is set nowhere**. No DFB has zero endpoints, so nothing is dropped as dead.
This census agrees with the brief on every row.

The three self-looped buffers are NoC-read landing areas: the reader reserves an entry, hands the write
pointer to `noc.async_read`, then reads the landed data straight back through `CoreLocalMem`. Nothing else
ever sees them, and nothing pushes or pops them. A DM self-loop is legal on Gen1 (the DFB lowers to a plain
circular buffer that one DM RISC both fills and drains) and is a Quasar-uplift concern, not a Gen1 blocker.

## Preserved Multiplicity

none — no work-split multiplicity in legacy. No factory calls `split_work_to_cores` or pushes two
`KernelDescriptor`s of the same source; every kernel is instantiated once over one node set.

## Dropped Plumbing

### Factory 1 and factory 2

| legacy location (file:line) | legacy form | Metal 2.0 replacement |
|---|---|---|
| `manual_seed_program_factory.cpp:67,93` CTA slot 0 | positional `{seeds.value_or(0)}` | named CTA `{{"seed", seeds.value_or(0)}}` |
| `manual_seed_set_seed.cpp:10` | `get_compile_time_arg_val(0)` | `get_arg(args::seed)` |

### Factory 3

| legacy location (file:line) | legacy form | Metal 2.0 replacement |
|---|---|---|
| factory `:142` reader CTA slot 0 | `user_ids_cb_index` (magic `CBIndex::c_0`) | `DFBBinding{USER_IDS, "user_ids", PRODUCER}` + `{…, CONSUMER}` |
| factory `:142` reader CTA slot 1 | `kernel_communication_cb_index` (magic `CBIndex::c_1`) | `DFBBinding{KERNEL_COMMUNICATION, "kernel_communication", PRODUCER}` |
| factory `:142` reader CTA slot 2 | positional `number_of_ids` | named CTA `{{"number_of_ids", …}}` |
| factory `:143` reader CTA tail | `TensorAccessorArgs(user_ids_mesh).append_to(reader_compile_time_args)` | `TensorParameter{USER_IDS_TENSOR}` + `TensorBinding{…, "user_ids"}` |
| factory `:157` reader RTA slot 0 | `user_ids_mesh` (framework-patched buffer binding) | `TensorArgument{USER_IDS_TENSOR, user_ids_mesh}` |
| factory `:157` reader RTA slot 1 | positional `core_id` | named RTA `"core_id"` |
| factory `:162-163` compute CTA slot 0 | `kernel_communication_cb_index` (magic index) | `DFBBinding{KERNEL_COMMUNICATION, "kernel_communication", CONSUMER}` |
| factory `:162-163` compute CTA slot 1 | positional `seeds.value_or(0)` | named CTA `{{"seed", …}}` |
| `reader_manual_seed_read_user_id.cpp:16` | `get_arg_val<uint32_t>(0)` (buffer address) | gone — the `TensorBinding` injects the base address |
| `reader_manual_seed_read_user_id.cpp:17` | `get_arg_val<uint32_t>(1)` | `get_arg(args::core_id)` |
| `reader_manual_seed_read_user_id.cpp:20-22` | `get_compile_time_arg_val(0..2)` | two DFB handles + `get_arg(args::number_of_ids)` |
| `reader_manual_seed_read_user_id.cpp:23` | `TensorAccessorArgs<3>()` | gone |
| `reader_manual_seed_read_user_id.cpp:30` | `TensorAccessor(args, addr)` | `TensorAccessor(tensor::user_ids)` |
| `manual_seed_single_seed_receive_user_id.cpp:16-17` | `get_compile_time_arg_val(0..1)` | `dfb::kernel_communication` + `get_arg(args::seed)` |

### Factory 4

| legacy location (file:line) | legacy form | Metal 2.0 replacement |
|---|---|---|
| factory `:218-219` reader CTA slots 0-2 | three magic CB indices (`c_0`, `c_1`, `c_2`) | `DFBBinding`s: `USER_IDS` P+C, `SEEDS` P+C, `KERNEL_COMMUNICATION` P |
| factory `:218-219` reader CTA slot 3 | positional `number_of_ids` | named CTA `{{"number_of_ids", …}}` |
| factory `:220-221` reader CTA tail | two `TensorAccessorArgs(...).append_to(...)` | `TensorParameter`s `USER_IDS_TENSOR` / `SEEDS_TENSOR` + their `TensorBinding`s |
| factory `:235` reader RTA slots 0-1 | `user_ids_mesh`, `seeds_mesh` | two `TensorArgument`s |
| factory `:235` reader RTA slot 2 | positional `core_id` | named RTA `"core_id"` |
| factory `:244` compute CTA slot 0 | `kernel_communication_cb_index` (magic index) | `DFBBinding{KERNEL_COMMUNICATION, "kernel_communication", CONSUMER}` — leaves the compute kernel with no CTAs at all |
| `reader_manual_seed_read_all_data.cpp:16-17` | `get_arg_val<uint32_t>(0..1)` (two buffer addresses) | gone — both `TensorBinding`s inject their base addresses |
| `reader_manual_seed_read_all_data.cpp:18` | `get_arg_val<uint32_t>(2)` | `get_arg(args::core_id)` |
| `reader_manual_seed_read_all_data.cpp:21-24` | `get_compile_time_arg_val(0..3)` | three DFB handles + `get_arg(args::number_of_ids)` |
| `reader_manual_seed_read_all_data.cpp:25-27` | `TensorAccessorArgs<4>()` + `next_compile_time_args_offset()` chain | gone |
| `reader_manual_seed_read_all_data.cpp:34,37` | `TensorAccessor(args, addr)` ×2 | `TensorAccessor(tensor::user_ids)` / `TensorAccessor(tensor::seeds)` |
| `manual_seed_receive_all_data.cpp:16` | `get_compile_time_arg_val(0)` | `dfb::kernel_communication` |

### Host-side CB API removed everywhere

`push_tensor_circular_buffer` @ `:39-54`, `CBDescriptor` / `CBFormatDescriptor` / `desc.cbs`, and the
`<tt-metalium/program_descriptors.hpp>` and `<tt-metalium/tensor_accessor_args.hpp>` includes all go; a
`make_tensor_dataflow_buffer` helper returning a `DataflowBufferSpec` takes the CB helper's place.

### Not dropped

- **Page-size third accessor argument**: none exists — every `TensorAccessor` is constructed with two
  arguments, so there is nothing to drop here.
- **Semaphore-ID RTAs**: none — the op declares no semaphores.
- **Varargs**: none introduced. Every legacy RTA is a distinct field read once at a literal index, so all
  become named.

## Applied Patterns

- [Sync-free and single-ended CBs → self-loop DFB](../../../../../../docs/source/tt-metalium/tt_metal/apis/host_apis/metal_2.0/ai/shared/port_patterns.md#pattern-sync-free-and-single-ended-cbs--self-loop-dfb):
  `USER_IDS` in factories 3 and 4, and `SEEDS` in factory 4 — each a one-toucher NoC landing area on the
  reader, bound PRODUCER **and** CONSUMER on that one kernel.
- [Self-loop DFB binding](../../../../../../docs/source/tt-metalium/tt_metal/apis/host_apis/metal_2.0/ai/shared/port_patterns.md#pattern-self-loop-dfb-binding):
  the mechanism the three self-loops above borrow — both endpoints share one `accessor_name`, so the kernel
  keeps a single `DataflowBuffer` object.
- [Two-toucher DFB → assign 1P+1C](../../../../../../docs/source/tt-metalium/tt_metal/apis/host_apis/metal_2.0/ai/shared/port_patterns.md#pattern-two-toucher-dfb--assign-1p1c-dual-instance-work-split):
  used as the endpoint-assignment procedure for the whole census above, and directly for
  `KERNEL_COMMUNICATION` in both factories (a genuine locked-producer / locked-consumer pair).
- [Porting a shared kernel](../../../../../../docs/source/tt-metalium/tt_metal/apis/host_apis/metal_2.0/ai/shared/port_patterns.md#caution-porting-a-shared-kernel),
  intra-op shape, **rung 3**: `manual_seed_set_seed.cpp` converts in place because both of its binding
  factories convert in this change.
- [Pass DFB handles directly to LLKs and kernel-lib helpers](../../../../../../docs/source/tt-metalium/tt_metal/apis/host_apis/metal_2.0/ai/shared/port_patterns.md#pattern-pass-dfb-handles-directly-to-llks-and-kernel-lib-helpers):
  **not** applied — see Deferred / Flagged item 1. The kernels' only cb-id-keyed calls are metadata
  lookups, which whitelist rule 7 routes to the `DataflowBuffer` object instead.

Not applied, and why: no conditional/optional bindings (nothing is gated on a host-time flag), no aliased
DFBs (no multi-element `format_descriptors`), no same-FIFO aliasing (no kernel-side CB-index alias), no
multi-variant factory (the four variants are four factory classes, not branches inside one), no
CTA-to-RTA demotion risk (no work split), no pybind entry point to remove.

## Deferred / Flagged

1. **The brief's "leave the three `get_dataformat` lines as they are" is not achievable as written, so the
   port follows whitelist rule 7 instead.** The brief asks to leave
   `constexpr DataFormat … = get_dataformat(<dfb_index>);` untouched at
   `reader_manual_seed_read_user_id.cpp:29` and `reader_manual_seed_read_all_data.cpp:33,36`, on the grounds
   that the results are dead and moving the call onto the `DataflowBuffer` object would cost a reorder. But
   the argument those calls take — the CB-index CTA — is exactly what the port removes, so the lines cannot
   survive unchanged in any form. The two remaining options are passing `dfb::<name>` into the legacy
   free function (which whitelist rule 2's note explicitly steers away from when a native mechanism exists)
   or the rule-7 object getter. The port takes the rule-7 getter and pays the small reorder: the
   `Noc` / `DataflowBuffer` declarations move above the tensor-config block in both readers. The variables
   become `const` rather than `constexpr`, because a `DataflowBuffer` is not constexpr-constructible; they
   are dead either way, and the JIT compiles kernels with `-Wno-unused-variable`. Deleting them outright
   stays out of scope, as the audit directed. Recorded in the port report as a brief-vs-recipe disagreement.

2. **No other new findings.** Planning turned up nothing the audit missed: no feature outside the audit's
   Appendix A, no descriptor type without a Metal 2.0 counterpart, no kernel construct that resists a
   binding-token replacement.
