# Port Plan — all_to_all_combine

Port plan for `ttnn/cpp/ttnn/operations/ccl/all_to_all_combine`, ported from the `WorkloadDescriptor` (ProgramDescriptor)
API to Metal 2.0. Written during the inventory and planning steps; committed alongside the port for review. Line numbers
refer to the pre-port tree `66b645994d5`.

## Legacy Inventory

### Legacy factory shape
- Concept: `ProgramDescriptorFactoryConcept` through `create_workload_descriptor` (`WorkloadDescriptor`). No
  `override_runtime_arguments`. The method lives on `AllToAllCombineFromSparse`, the one alternative of `program_factory_t`
  (`all_to_all_combine_device_operation.hpp:56-71`).
- Variants: single.
- Custom `compute_program_hash`: none — default reflection-based hash.

*(The Metal 2.0 factory concept was chosen during the audit; carried forward in [TTNN ProgramFactory](#ttnn-programfactory).)*

### Kernels

One program per mesh coordinate (`all_to_all_combine_program_factory.cpp:362-371`). In each program one reader and one
writer `KernelDescriptor` over `sender_core_grid` (`:248-272`), with per-core RTAs.

| unique_id | source | core_ranges | CTAs (positional) | CTAs (named) | RTAs | CRTAs | defines | opt_level | config |
|---|---|---|---|---|---|---|---|---|---|
| reader | `device/kernels/dataflow/reader_all_to_all_combine.cpp` (lent; see Shared kernels) | `sender_core_grid` | 0 `mapping_cb_id` = `c_1`; 1 `local_experts_cb_id` = `c_2`; 2 `metadata_cb_id` = `c_3`; 3 `data_cb_id` = `c_0`; 4 `num_local_experts` = `experts_per_device`; 5 `batch_size`; 6 `seq_size`; 7 `num_mapping_pages` = `experts`; 8 `linearized_mesh_coord` = `flat_mesh_idx`; 9 `data_size_bytes` = input page size; 10 `selected_experts_k`; 11 `mapping_page_size_bytes`; 12 `metadata_page_size_bytes`; 13 `locally_reduced`; 14+ `TensorAccessorArgs(input, mapping, metadata)` | none | 0 mapping `Buffer*`; 1 metadata `Buffer*`; 2 input `Buffer*`; 3 `token_start_idx`; 4 `token_end_idx` | none | none | O2 (unset, DM) | `DataMovementConfigDescriptor{RISCV_1, NOC_1}` (custom; noc_mode default `DM_DEDICATED_NOC`) |
| writer | `device/kernels/dataflow/writer_all_to_all_combine.cpp` (op-owned) | `sender_core_grid` | 0 `metadata_cb_id` = `c_3`; 1 `local_experts_cb_id` = `c_2`; 2 `packet_header_cb_id` = `c_4`; 3 `data_cb_id` = `c_0`; 4 `batch_size`; 5 `seq_size`; 6 `selected_experts_k`; 7 `num_local_experts`; 8 `num_devices`; 9 `src_chip_id`; 10 `data_size_bytes`; 11 `alignment` = L1 alignment; 12 `mesh_rows`; 13 `mesh_cols`; 14 `fabric_max_packet_size_bytes`; 15 `linearized_mesh_coord`; 16 `topology`; 17 `locally_reduced`; 18+ `TensorAccessorArgs(output)` | none | 0 output `Buffer*`; 1 final (cross-device) GlobalSemaphore address; 2 init GlobalSemaphore address; 3 `token_start_idx`; 4 `token_end_idx`; 5+ per neighbor direction in `{E, W, N, S}` order: `append_fabric_connection_rt_args(..., link_id = core index, ...)` | none | `DEST_CHIP_ID`, `DEST_MESH_ID` (whole mesh), `DIRECTIONS` (per coord), `REPLICATE_GROUP_AXIS` (when `axis`) | O2 (unset, DM) | `DataMovementConfigDescriptor{RISCV_0, NOC_0}` (custom) |

### CBs
| index | total_size | core_ranges | data_format | page_size | tile (if set) |
|---|---|---|---|---|---|
| `c_0` data (`:110`) | `experts_per_device * aligned_input_page` | `sender_core_grid` | input dtype | `aligned_input_page` (DRAM or L1 alignment) | not set |
| `c_1` mapping (`:122`) | `aligned_mapping_page` | same | mapping dtype | same | not set |
| `c_2` local experts (`:138`) | `aligned_local_expert_page` | same | UInt16 | same | not set |
| `c_3` metadata (`:150`) | `aligned_metadata_page` | same | metadata dtype | same | not set |
| `c_4` packet header (`:164`) | `3 * CLIENT_INTERFACE_SIZE` | same | UInt32 | `CLIENT_INTERFACE_SIZE` | not set |

### Semaphores
none (no `SemaphoreDescriptor`). Two op-owned GlobalSemaphores (`:347-350`) ride as raw writer RTAs 1 and 2.

### Tensor accessors
| host site (file:line) | originating Tensor | RTA slot (host) |
|---|---|---|
| `:192` (reader CTAs) | input | reader RTA2 (`:288`) |
| `:193` | mapping | reader RTA0 (`:286`) |
| `:194` | metadata | reader RTA1 (`:287`) |
| `:222` (writer CTAs) | output | writer RTA0 (`:321`) |

### Work split
- Driver: `num_cores_to_corerangeset_in_subcoregrids(subdevice_cores[0], num_cores, worker_core_range_set, true)` with
  `tokens_per_core = div_up(batch * seq, num_links)`, `num_cores = min(num_links, div_up(tokens, tokens_per_core))`.
- Each sender core takes a contiguous token range; core `i` uses fabric link `i`.

### Shared kernels
- `reader_all_to_all_combine.cpp` — *lent*: also bound by `data_movement/moe_expert_token_remap`
  (`moe_expert_token_remap_program_factory.cpp:176`). No `_metal2` fork exists beside it → rung 2: create
  `reader_all_to_all_combine_metal2.cpp` beside it, pointer comment in the original. Binding vocabulary (new): `dfb::mapping`,
  `dfb::local_experts`, `dfb::metadata`, `dfb::data`, `tensor::data`, `tensor::mapping`, `tensor::metadata`; named CTAs
  `num_local_experts`, `batch_size`, `seq_size`, `num_mapping_pages`, `linearized_mesh_coord`, `data_size_bytes`,
  `selected_experts_k`, `mapping_page_size_bytes`, `metadata_page_size_bytes`, `locally_reduced`; RTAs `token_start_idx`,
  `token_end_idx`.
- `writer_all_to_all_combine.cpp` — op-owned (also named in `scripts/check_kernel_cb_ptrs.py:52`, a lint allow-list, not a
  binder). Converted in place.

### Flags
- Fabric: `append_fabric_connection_rt_args` (`:315`) once per neighbor of `get_neighbors` (`:233`), link index = core index.
- `DIRECTIONS` define (`:238`) from `get_neighbors`.

## TTNN ProgramFactory

- **Concept (inherited from audit)**: `MeshWorkloadSpecFactoryConcept`, no `override_runtime_arguments`.
- **Custom `compute_program_hash`**: none.
- **Implementation notes**:
  - `create_mesh_workload_artifacts`: create the two GlobalSemaphores and run `Synchronize` as today, park them in
    `MeshWorkloadArtifacts::global_semaphores` (reserve two first), then one `PerCoordProgram` per coordinate of
    `tensor_coords`, range = that coordinate.
  - No override: on a cache hit the framework refreshes the tensor bindings; the GlobalSemaphore args stay as at cache miss,
    which is right because the semaphores are op-owned and live with the cached workload (legacy kept their addresses in the
    cached RTAs the same way).

## Planned Spec Shape

- KernelSpecs: `reader` (the `_metal2` fork) and `writer` over `sender_core_grid` (1:1 with legacy).
- DataflowBufferSpecs: `data` (`experts_per_device` entries), `mapping`, `local_experts`, `metadata` (1 entry each),
  `packet_header` (3 entries of `CLIENT_INTERFACE_SIZE`, UInt32). Sizes and formats as the legacy CBs.
- SemaphoreSpecs: none.
- GlobalSemaphoreParameters: `init_barrier_semaphore`, `final_barrier_semaphore` (L1), bound by the writer as
  `init_semaphore` / `global_semaphore`.
- FabricConnectionSpecs: `east`, `west`, `north`, `south`; `src_fabric_node_id = get_fabric_node_id(coord)`; on sender
  core `i`, a target `{neighbor in that direction, link i}` when the direction has a neighbor. All four bound by the writer.
- TensorParameters: `input`, `mapping`, `metadata`, `output`.
- WorkUnitSpecs: `sender` = `{reader, writer}` on `sender_core_grid`.
- Per-coordinate programs: one per coordinate (`:362-371`). Per coordinate: `src_chip_id`, linearized index, fabric targets.

## Preserved Multiplicity

none — no work-split multiplicity in legacy (one reader and one writer descriptor per program).

## Dropped Plumbing

| legacy location (file:line) | legacy form | Metal 2.0 replacement |
|---|---|---|
| reader RTA0-2 (`:286-288`; `reader.cpp:75-77`) | mapping / metadata / input `Buffer*` | `TensorBinding`s `mapping`, `metadata`, `data` |
| writer RTA0 (`:321`; `writer.cpp:159`) | output `Buffer*` | `TensorBinding(output, "output")` |
| reader CTAs 0-3, writer CTAs 0-3 (`:177-180`, `:203-206`) | CB indices | `DFBBinding`s (see Applied Patterns for roles) |
| reader CTAs 14+, writer CTAs 18+ (`:192-194`, `:222`) | `TensorAccessorArgs(...).append_to` / `TensorAccessorArgs<14>()`, `<18>()` | tensor bindings |
| writer RTA1, RTA2 (`:300-301`) | final / init GlobalSemaphore addresses | `GlobalSemaphoreBinding`s `global_semaphore`, `init_semaphore` (override 4a) |
| writer RTA5+ (`:313-317`) and `DIRECTIONS` define (`:238`) | per-direction `append_fabric_connection_rt_args` tail, direction flags | four `FabricConnectionBinding`s `east` / `west` / `north` / `south`; direction flag = `has_fabric_connection` (override 4b, 7) |
| all positional CTAs / RTAs | positional | named, as in the Kernels table |

## Applied Patterns

- `MeshWorkloadSpecFactoryConcept` per-coordinate programs (`ttnn_factory.md`), no override; op-owned GlobalSemaphores in
  `MeshWorkloadArtifacts::global_semaphores` (override 2).
- 1P+1C (reader PRODUCER, writer CONSUMER) on `data`, `local_experts`, `metadata`.
- Sync-free / single-ended → self-loop DFB (`port_patterns.md`): `mapping` on the reader, `packet_header` on the writer
  (shared accessor name).
- Caution: Porting a shared kernel, rung 2 (create the fork) for the reader.
- Branch overrides 4a, 4b, 6, 7.

## Deferred / Flagged

- 2-device Ring along the dispatch axis: East and West are the same neighbor, so two specs carry `(src, dst, link i)` and the
  validator rejects the program. Legacy connected two senders of one core to one router (suspect). Recorded, not changed.
