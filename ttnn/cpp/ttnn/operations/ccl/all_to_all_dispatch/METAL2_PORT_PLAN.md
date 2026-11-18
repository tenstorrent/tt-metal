# Port Plan — all_to_all_dispatch

Port plan for `ttnn/cpp/ttnn/operations/ccl/all_to_all_dispatch`, ported from the `WorkloadDescriptor` (ProgramDescriptor)
API to Metal 2.0. Written during the inventory and planning steps; committed alongside the port for review. Line numbers
refer to the pre-port tree `94bf191ce8f`.

## Legacy Inventory

### Legacy factory shape
- Concept: `ProgramDescriptorFactoryConcept` through `create_workload_descriptor` (`WorkloadDescriptor`). No
  `override_runtime_arguments`. The method lives on `AllToAllDispatchSparse`, the one alternative of `program_factory_t`
  (`all_to_all_dispatch_device_operation.hpp:67-84`).
- Variants: single factory; two modes inside it, `impl == FullPacket` (CB `c_5` and its kernel code) and
  `impl == PageByPage` (no `c_5`).
- Custom `compute_program_hash`: none — default reflection-based hash.

*(The Metal 2.0 factory concept was chosen during the audit; carried forward in [TTNN ProgramFactory](#ttnn-programfactory).)*

### Kernels

One program per mesh coordinate (`all_to_all_dispatch_program_factory.cpp:529-538`). In each program one reader and one
writer `KernelDescriptor` over `sender_core_grid` (`:396-421`), with per-core RTAs. Both receive the same CTA vector
(`:322-375`).

| unique_id | source | core_ranges | CTAs (positional, shared) | CTAs (named) | RTAs | CRTAs | defines | opt_level | config |
|---|---|---|---|---|---|---|---|---|---|
| reader | `device/kernels/dataflow/reader_all_to_all_dispatch.cpp` (op-owned) | `sender_core_grid` | 0-4 CB ids `c_0`-`c_4`; 5-9 `*_pages` (input, indices, mapping, output, metadata); 10-14 page sizes; 15 `num_devices`; 16 `hidden_size`; 17 `batch_size`; 18 `selected_experts_k`; 19 `experts`; 20 `tokens_per_device`; 21 `num_links`; 22 `topology`; 23 `src_mesh_id`; 24 `src_chip_id`; 25 `mesh_rows`; 26 `mesh_cols`; 27-31 aligned page sizes; 32 `fabric_max_packet_size`; 33 `alignment` (L1); 34 metadata buffer CB id `c_5`; 35 `write_page_by_page`; 36 `linearized_mesh_coord`; 37+ `TensorAccessorArgs(input, indices, mapping, output, metadata)`. Reader reads 0-2, 5-7, 14, 15, 20, 22-26, 28, 29, 31, 34-36 | none | 0-4 tensor `Buffer*` (input, indices, mapping, output, metadata); 5 final GlobalSemaphore address; 6 `token_start_idx`; 7 `token_end_idx` | none | `AXIS` (axis or `-1`) | O2 (unset, DM) | `DataMovementConfigDescriptor{RISCV_1, NOC_1}` (custom; noc_mode default `DM_DEDICATED_NOC`) |
| writer | `device/kernels/dataflow/writer_all_to_all_dispatch.cpp` (op-owned) | `sender_core_grid` | same vector; writer reads 0-36 | none | 0-4 tensor `Buffer*`; 5 final GlobalSemaphore address; 6 init GlobalSemaphore address; 7 `token_start_idx`; 8 `token_end_idx`; 9+ per neighbor direction in `{E, W, N, S}` order: `append_fabric_connection_rt_args(..., link_id = core index, ...)` | none | `DEST_CHIP_ID`, `DEST_MESH_ID` (whole mesh), `DIRECTIONS` (per coord), `AXIS` (when `axis`) | O2 (unset, DM) | `DataMovementConfigDescriptor{RISCV_0, NOC_0}` (custom) |

### CBs
Sizes from `detail::get_cb_sizes` (`:43-88`; also called by `all_to_all_dispatch.cpp:39` for the L1 budget that chooses
the mode).

| index | total_size | core_ranges | data_format | page_size | tile (if set) |
|---|---|---|---|---|---|
| `c_0` input (`:250`) | `2 * aligned_input_page` | `sender_core_grid` | input dtype | `aligned_input_page` | not set |
| `c_1` indices (`:259`) | `tokens_per_core * aligned_indices_page` | same | indices dtype | `aligned_indices_page` | not set |
| `c_2` mapping (`:268`) | `mapping_pages * aligned_mapping_page` | same | mapping dtype | `aligned_mapping_page` | not set |
| `c_3` packet header (`:277`) | `3 * packet_header_size` | same | RawUInt32 | `packet_header_size` | not set |
| `c_4` send preparation (`:286`) | `num_devices * tokens_per_core` | same | UInt8 | `tokens_per_core` | not set |
| `c_5` metadata buffer (`:295-305`, FullPacket only) | `tokens_per_device * dispatch_devices * aligned_indices_page` | same | mapping dtype | `aligned_indices_page` | not set |

### Semaphores
none (no `SemaphoreDescriptor`). Two op-owned GlobalSemaphores (`:514-517`) ride as raw RTAs (reader 5; writer 5, 6).

### Tensor accessors
| host site (file:line) | originating Tensor | RTA slot (host) | used by |
|---|---|---|---|
| `:369` | input | reader/writer RTA0 (`:434`, `:484`) | reader |
| `:370` | indices (`expert_indices_tensor`) | RTA1 | reader |
| `:371` | mapping (`expert_mapping_tensor`) | RTA2 | reader |
| `:372` | output (`tensor_return_value[0]`) | RTA3 | writer |
| `:373` | metadata (`tensor_return_value[1]`) | RTA4 | reader (FullPacket copy), writer |

### Work split
- Driver: `num_cores_to_corerangeset_in_subcoregrids(subdevice_cores[0], num_cores, worker_core_range_set, true)` with
  `tokens_per_core = div_up(tokens_per_device, num_links)`, `num_cores = min(num_links, div_up(tokens, tokens_per_core))`.
- Each sender core takes a contiguous token range; core `i` uses fabric link `i`.

### Shared kernels
none — `grep -rn` for both kernel file names over `ttnn/`, `tests/`, `models/` hits only this factory.

### Flags
- Fabric: `append_fabric_connection_rt_args` (`:472`) once per neighbor of `get_neighbors` (`:137`), link index = core
  index.
- `DIRECTIONS` define (`:388`) from `get_neighbors`.
- `AXIS` is always defined for the reader (`-1` without an axis), only with an axis for the writer.

## TTNN ProgramFactory

- **Concept (inherited from audit)**: `MeshWorkloadSpecFactoryConcept`, no `override_runtime_arguments`.
- **Custom `compute_program_hash`**: none.
- **Implementation notes**:
  - `create_mesh_workload_artifacts`: create the two GlobalSemaphores and run `Synchronize` as today, park them in
    `MeshWorkloadArtifacts::global_semaphores` (reserve two first), then one `PerCoordProgram` per coordinate of
    `tensor_coords`, range = that coordinate.
  - No override: on a cache hit the framework refreshes the five tensor bindings; the GlobalSemaphore args stay as at cache
    miss, which is right because the semaphores are op-owned and live with the cached workload.

## Planned Spec Shape

- KernelSpecs: `reader`, `writer` over `sender_core_grid` (1:1 with legacy).
- DataflowBufferSpecs: `input` (2 entries), `indices` (`tokens_per_core`), `mapping` (`mapping_pages`), `packet_header`
  (3), `send_preparation_buffer` (`num_devices` entries of `tokens_per_core` bytes), and **only under FullPacket**
  `metadata_buffer` (`tokens_per_device * dispatch_devices`). Entry sizes, counts and formats from `get_cb_sizes` and the
  legacy formats.
- SemaphoreSpecs: none.
- GlobalSemaphoreParameters: `init_barrier_semaphore`, `final_barrier_semaphore` (L1). Bindings: reader
  `global_semaphore` (final); writer `global_semaphore` (final) and `init_semaphore` (init).
- FabricConnectionSpecs: `east`, `west`, `north`, `south`; `src_fabric_node_id = get_fabric_node_id(coord)`; on sender
  core `i`, a target `{neighbor in that direction, link i}` when the direction has a neighbor. All four bound by the writer.
- TensorParameters: `input`, `indices`, `mapping`, `output`, `metadata`. Bindings: reader `input`, `indices`, `mapping`,
  `metadata`; writer `output`, `metadata`.
- WorkUnitSpecs: `sender` = `{reader, writer}` on `sender_core_grid`.
- Per-coordinate programs: one per coordinate (`:529-538`). Per coordinate: `src_mesh_id`, `src_chip_id`, linearized index,
  fabric targets.

## Preserved Multiplicity

none — no work-split multiplicity in legacy (one reader and one writer descriptor per program).

## Dropped Plumbing

| legacy location (file:line) | legacy form | Metal 2.0 replacement |
|---|---|---|
| reader RTA0-4 (`:434-438`; `reader.cpp:61-65`) | five tensor `Buffer*` | `TensorBinding`s `input`, `indices`, `mapping`, `metadata`; output not bound (unused) |
| writer RTA0-4 (`:484-488`; `writer.cpp:150-154`) | five tensor `Buffer*` | `TensorBinding`s `output`, `metadata`; input / indices / mapping not bound (unused) |
| CTAs 0-4, 34 (`:323-327`, `:365`) | CB indices | `DFBBinding`s (see Applied Patterns for roles) |
| CTA 35 (`:366`) | `write_page_by_page` | host define `FULL_PACKET` on both kernels, emitted with the `metadata_buffer` DFB and binding |
| CTAs 37+ (`:369-373`) | `TensorAccessorArgs(...).append_to` / `TensorAccessorArgs<37>()` chain | tensor bindings |
| reader RTA5, writer RTA5-6 (`:439`, `:454-455`) | final / init GlobalSemaphore addresses | `GlobalSemaphoreBinding`s `global_semaphore`, `init_semaphore` (override 4a) |
| writer RTA9+ (`:459-480`) and `DIRECTIONS` define (`:388`) | per-direction `append_fabric_connection_rt_args` tail, direction flags | four `FabricConnectionBinding`s `east` / `west` / `north` / `south`; direction flag = `has_fabric_connection` (override 4b, 7) |
| shared positional CTA vector | positional, same for both kernels | named, per kernel: each kernel gets exactly the CTAs it declares (reader 15, writer 30) |
| all positional RTAs | positional | named `token_start_idx`, `token_end_idx` |

## Applied Patterns

- `MeshWorkloadSpecFactoryConcept` per-coordinate programs (`ttnn_factory.md`), no override; op-owned GlobalSemaphores in
  `MeshWorkloadArtifacts::global_semaphores` (override 2).
- 1P+1C (reader PRODUCER, writer CONSUMER) on `input`, `indices`, `mapping`.
- Sync-free / single-ended → self-loop DFB (`port_patterns.md`): `packet_header` and `send_preparation_buffer` on the
  writer (shared accessor name).
- Conditional / optional resource bindings (`port_patterns.md`), "Promote a CTA gate to a define": `metadata_buffer` DFB,
  binding on both kernels and `FULL_PACKET` define on both kernels, all under `impl == FullPacket`. Two role-free
  touchers → 1P+1C (writer PRODUCER, reader CONSUMER).
- Branch overrides 4a, 4b, 6, 7.

## Deferred / Flagged

- 2-device Ring along the dispatch axis: East and West are the same neighbor, so two specs carry `(src, dst, link i)` and the
  validator rejects the program. Legacy connected two senders of one core to one router (suspect). Recorded, not changed.
- `detail::get_cb_sizes` keeps its name: it is declared in the device-op header and also called by the op's host
  `all_to_all_dispatch.cpp` (outside the factory), so a rename is out of scope.
