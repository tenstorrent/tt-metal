# Port Plan — slice_reshard_async

Port plan for `ttnn/cpp/ttnn/operations/experimental/ccl/slice_reshard_async`, ported from the `WorkloadDescriptor`
(ProgramDescriptor) API to Metal 2.0. Written during the inventory and planning steps; committed alongside the port for review.

## Legacy Inventory

### Legacy factory shape
- Concept: `ProgramDescriptorFactoryConcept` through `create_workload_descriptor` (`WorkloadDescriptor`), plus
  `override_runtime_arguments(Program&, …, optional<MeshCoordinate>)`. Methods live on
  `SliceReshardAsyncProgramFactory`, the one alternative of `program_factory_t`.
- Variants: single.
- Custom `compute_program_hash`: present at `device/slice_reshard_async_device_operation.cpp:50` — left intact. It leaves out
  `final_semaphore` and `barrier_semaphore`.

*(The Metal 2.0 factory concept was chosen during the audit; carried forward in [TTNN ProgramFactory](#ttnn-programfactory).)*

### Kernels

One program per mesh coordinate. In each program, every worker core `{core_idx, 0}`, `core_idx = link * 2 + direction`,
gets its own reader and its own writer `KernelDescriptor` (`slice_reshard_async_program_factory.cpp:137-280`).

| unique_id | source | core_ranges | CTAs (positional) | CTAs (named) | RTAs | CRTAs | defines | opt_level | config |
|---|---|---|---|---|---|---|---|---|---|
| reader (per core) | `device/kernels/minimal_default_reader.cpp` | `{core_idx, 0}` | 0 `is_first_chip` = `direction ? is_first_device : is_last_device`; 1 `is_last_chip` = `direction ? is_last_device : is_first_device`; 2 `cb_output_id` = `c_in0`; 3 `direction`; 4 `stick_size` = page size; 5+ `TensorAccessorArgs(input)` | `cb_forward_id` = `c_in0` (unused by the kernel) | 0 input `Buffer*`; 1 `stick_start_id`; 2 `num_sticks_to_read`; 3 `input_outer_dim_size`; 4 `outer_dims_to_forward` (`direction ? to_forward : to_backward`); 5 `outer_dims_from_forward`; 6 `outer_dims_to_keep_start`; 7 `outer_dims_to_keep_end`; 8 `num_sticks_per_outer_dim`; 9 final semaphore address | none | none | O2 (unset, DM) | `ReaderConfigDescriptor{}` |
| writer (per core) | `device/kernels/minimal_default_writer.cpp` | `{core_idx, 0}` | 0 `is_first_chip`; 1 `is_last_chip` (same as reader); 2 `cb_output_id` = `c_in0`; 3 `direction`; 4+ `TensorAccessorArgs(output)` | `cb_forward_id` = `c_in0` (unused) | 0 input `Buffer*` (dead); 1 output `Buffer*`; 2 `stick_size`; 3 `stick_start_id`; 4 `num_sticks_to_read`; 5 `output_outer_dim_size`; 6 `outer_dims_to_forward`; 7 `outer_dims_to_keep_start`; 8 `outer_dims_to_keep_end`; 9 `outer_dims_to_receive` (`direction ? from_backward : from_forward`); 10 `outer_dims_from_forward`; 11 `num_sticks_per_outer_dim`; 12/13 `out_ready_sem_noc0_x/y` (own core); 14 final semaphore address; 15 `use_barrier_sem` = 1; 16/17 `barrier_sem_noc0_x/y` (opposite core); 18 barrier semaphore address; 19+ fabric tail (forward flag, conn args, backward flag, conn args) | none | none | O2 (unset, DM) | `WriterConfigDescriptor{}` |

### CBs
| index | total_size | core_ranges | data_format | page_size | tile (if set) |
|---|---|---|---|---|---|
| `c_in0` (`sender_cb_index`) | `3 * page_size` | `worker_core_ranges` (from the work split) | `datatype_to_dataformat_converter(input dtype)` | `page_size` = `input_buffer->page_size()` | not set |

### Semaphores
none (no `SemaphoreDescriptor`). Two caller-supplied GlobalSemaphores ride as raw address RTAs (reader 9, writer 14, 18).

### Tensor accessors
| host site (file:line) | originating Tensor | RTA slot (host) |
|---|---|---|
| `slice_reshard_async_program_factory.cpp:160` (reader CTAs) | input | reader RTA0 (`:172`) |
| `slice_reshard_async_program_factory.cpp:202` (writer CTAs) | output | writer RTA1 (`:213`); kernel passes `stick_size` as 3rd arg |

### Work split
- Driver: `split_work_to_cores(CoreCoord(num_links * 2, 1), num_sticks_per_outer_dim * 2)`
- num_cores: up to `num_links * 2`
- core_group_1: `num_sticks_per_core_group_1` sticks per core
- core_group_2: `num_sticks_per_core_group_2` sticks per core
- Kernels are created on every core of the grid (not only on `worker_core_ranges`); a core outside group 1 gets the group-2
  count. `stick_start_id` advances once per link, by the direction-1 core's count.

### Shared kernels
none. Both kernels are op-owned and only this factory binds them (`grep -rn slice_reshard_async/device/kernels`). Converted in place.

### Flags
- Fabric: `append_fabric_connection_rt_args` (`:240`, `:256`) on the writer's core, link index = `link`. Direction 1 →
  forward neighbor, direction 0 → backward neighbor; at most one connection per writer.
- `override_runtime_arguments` (`:302`) rewrites reader RTA 9 and writer RTAs 14, 18 with the semaphore addresses on every hit.

## TTNN ProgramFactory

- **Concept (inherited from audit)**: `MeshWorkloadSpecFactoryConcept`, with the optional
  `override_runtime_arguments(attrs, tensor_args, tensor_return_value, const MeshCoordinateRange&)`.
- **Custom `compute_program_hash`**: present at `device/slice_reshard_async_device_operation.cpp:50` — leave intact.
- **Implementation notes**:
  - `create_mesh_workload_artifacts`: one `PerCoordProgram` per coordinate of `tensor_coords`, range = that coordinate.
  - `MeshWorkloadArtifacts::global_semaphores` stays empty: both semaphores are caller-owned.
  - The override returns the run args that every dispatch supplies: `tensor_args` for input and output (WorkloadDescriptor
    source → every io tensor) and `global_semaphore_args` for both semaphores (the ported-from RTA writes). The cache-miss
    run args use the same helper, plus the kernel RTAs.
  - The `slice_reshard_async_dynamic` namespace (kernel push order and RTA slot indices for the legacy override) has no use
    after the port and goes away with the descriptor builder.

## Planned Spec Shape

- KernelSpecs: per worker core, `reader_<core_idx>` and `writer_<core_idx>` (1:1 with the legacy descriptors).
- DataflowBufferSpecs: `sender` — `entry_size = page_size`, `num_entries = 3`, `data_format_metadata = df`.
- SemaphoreSpecs: none.
- GlobalSemaphoreParameters: `final_semaphore`, `barrier_semaphore` (L1).
- FabricConnectionSpecs: `connection`, `src_fabric_node_id = get_fabric_node_id(coord)`, a target on each writer core that
  has a neighbor in its direction, `link_idx = link`.
- TensorParameters: `input` (input `TensorSpec`), `output` (output `TensorSpec`).
- WorkUnitSpecs: `core_<core_idx>` = `{reader_<core_idx>, writer_<core_idx>}` on `{core_idx, 0}`.
- Per-coordinate programs: one per coordinate (`slice_reshard_async_program_factory.cpp:291-296`). Per coordinate: ring
  index, neighbors, the outer-dim counts, `is_first_chip` / `is_last_chip` CTAs, fabric targets.

## Preserved Multiplicity

| legacy KernelDescriptors | same-source KernelSpecs | WorkUnitSpecs | shared DFBs (endpoint role each binds) |
|---|---|---|---|
| reader on each core `{i, 0}` (`minimal_default_reader.cpp`) | `reader_<i>` | `core_<i>` | `sender`: PRODUCER (disjoint node sets, one per core) |
| writer on each core `{i, 0}` (`minimal_default_writer.cpp`) | `writer_<i>` | `core_<i>` | `sender`: CONSUMER |

The per-core CTAs (`direction`, `is_first_chip`, `is_last_chip`) stay CTAs.

## Dropped Plumbing

| legacy location (file:line) | legacy form | Metal 2.0 replacement |
|---|---|---|
| reader RTA0 (`factory.cpp:172`; `reader.cpp:43`) | input `Buffer*` | `TensorBinding(input, "src")` |
| writer RTA1 (`factory.cpp:213`; `writer.cpp:41`) | output `Buffer*` | `TensorBinding(output, "dst")` |
| writer RTA0 (`factory.cpp:212`; `writer.cpp:40`) | input `Buffer*`, never used by the kernel | dropped, no binding |
| reader CTA 2, writer CTA 2 (`factory.cpp:159`, `:201`) | `cb_output_id = c_in0` | `DFBBinding(sender, "output", PRODUCER / CONSUMER)` |
| reader/writer named CTA `cb_forward_id` (`factory.cpp:161`, `:203`) | CB index, unused by the kernels | dropped (the DFB binding carries the buffer) |
| reader CTAs 5+, writer CTAs 4+ (`factory.cpp:160`, `:202`) | `TensorAccessorArgs(*buffer).append_to` / `TensorAccessorArgs<5>()`, `<4>()` | tensor bindings |
| writer accessor 3rd arg (`writer.cpp:66`) | `stick_size` | dropped (Class 2); `stick_size` stays a named RTA as transfer size |
| reader RTA9, writer RTA14 (`factory.cpp:182`, `:224`) | final semaphore address | `GlobalSemaphoreBinding(final_semaphore, "out_ready_sem")` (override 4a) |
| writer RTA18 (`factory.cpp:233`) | barrier semaphore address | `GlobalSemaphoreBinding(barrier_semaphore, "barrier_sem")` (override 4a) |
| writer RTA 19+ (`factory.cpp:236-262`) | fabric flags + `append_fabric_connection_rt_args` tail | `FabricConnectionBinding(connection, "connection")` (override 4b) |
| all positional CTAs / RTAs | positional | named, as in the Kernels table |

## Applied Patterns

- `MeshWorkloadSpecFactoryConcept` per-coordinate programs (`ttnn_factory.md`), with the translated
  `override_runtime_arguments`.
- Branch override 4a (GlobalSemaphore binding) and 4b (fabric connection binding).

## Deferred / Flagged

- Edge case `num_sticks_per_outer_dim < num_links`: legacy kernels run on cores with no CB. In Metal 2.0 the DFB is placed on
  every node of its binding kernels, so those cores get the DFB too. Behavior on that path was undefined in legacy; record it.
