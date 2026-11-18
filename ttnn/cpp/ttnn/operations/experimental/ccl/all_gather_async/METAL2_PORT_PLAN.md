# Port Plan — all_gather_async (llama-sharded factory)

Port plan for `ttnn/cpp/ttnn/operations/experimental/ccl/all_gather_async`, factory `LlamaShardedMeshWorkloadFactory`,
ported from the `WorkloadDescriptor` API to Metal 2.0. `DefaultMeshWorkloadFactory` and `AllGatherViaBroadcastFactory`
are not ported (see `METAL2_PREPORT_AUDIT.md`). Written during the inventory and planning steps; committed alongside the
port for review.

## Legacy Inventory

### Legacy factory shape
- Concept: `ProgramDescriptorFactoryConcept` through `create_workload_descriptor(attrs, tensor_args, out, tensor_coords)`
  (`WorkloadDescriptor`), plus `override_runtime_arguments(Program&, …, optional<MeshCoordinate>)`. One alternative of the
  three-way `program_factory_t`.
- Variants: single.
- Custom `compute_program_hash`: present at `device/all_gather_async_device_operation.cpp:243` — left intact.

*(The Metal 2.0 factory concept was chosen during the audit; carried forward in [TTNN ProgramFactory](#ttnn-programfactory).)*

### Kernels

One program per mesh coordinate. One worker core per link.

| unique_id | source | core_ranges | CTAs (positional) | CTAs (named) | RTAs | CRTAs | defines | opt_level | config |
|---|---|---|---|---|---|---|---|---|---|
| reader (idx 0) | `device/kernels/llama_shapes_sharded_reader.cpp` | `sender_worker_core_range` | 0 `my_chip_id` = ring index (unused); 1 `cb0_id` = `c_in0`; 2 `tensor0_page_size` | none | 0 `num_tiles_per_core`; 1 `num_tiles_to_read`; 2 `first_core_tile_start_offset`; 3 `num_cores`; then `num_cores` x coords, `num_cores` y coords | 0 input `Buffer*` | none | O2 (unset, DM) | `ReaderConfigDescriptor{}` |
| writer (idx 1) | `device/kernels/llama_shapes_sharded_writer.cpp` | `sender_worker_core_range` | 0 `my_chip_id` (unused); 1 `reserved_packet_header_cb_id` = `c_in1`; 2 `num_packet_headers_storable` = 8; 3 `cb0_id` = `c_in0`; 4 `packet_size_in_pages`; 5 `tensor0_page_size`; 6/7 `num_targets_forward/backward_direction`; 8 `ring_size`; 9 `use_barrier_sem`; 10-15 forward line-mcast route; 16-21 backward route | none | 0 `num_tiles_per_core`; 1 `num_tiles_to_read`; 2 `first_core_tile_start_offset`; 3 `num_cores`; 4 `wait_output_semaphore`; 5 `reset_global_semaphore`; 6/7 `out_ready_sem_noc0_x/y`; 8 `out_ready_sem_wait_value`; 9/10 `barrier_sem_noc0_x/y`; then x coords, y coords; then forward flag + conn args, backward flag + conn args | 0 output `Buffer*`; 1 semaphore address; 2 barrier address (0 when absent) | none | O2 (unset, DM) | `WriterConfigDescriptor{}` |

### CBs
| index | total_size | core_ranges | data_format | page_size | tile (if set) |
|---|---|---|---|---|---|
| `c_in0` | `(input_pages / num_links + 1) * page_size` | `sender_worker_core_range` | input dtype | `page_size` | not set |
| `c_in1` (packet header) | `8 * header_size * 2` | `sender_worker_core_range` | `RawUInt32` | `header_size` | not set |

### Semaphores
none (no `SemaphoreDescriptor`). Two caller-supplied GlobalSemaphores ride as raw address CRTAs on the writer.

### Tensor accessors
none. Both tensors are read by raw base address (Case 2): input base at reader CRTA 0, output base at writer CRTA 0.

### Work split
Input pages split over links: `base = pages / num_links`, the first `pages % num_links` links get one more page
(`all_gather_async_llama_sharded_program_factory.cpp:229-232`). Each link core reads its range from the input shard
cores and writes it to the output shard cores of this device's slice.

### Shared kernels
none. `grep -rn llama_shapes_sharded ttnn/ tests/ models/ tt_metal/` hits only this factory. Converted in place.

### Flags
- Fabric: `append_fabric_connection_rt_args` (`:317`, `:324`) on each link core, link index = `link`: forward neighbor,
  then backward neighbor, each only when the neighbor exists, each after a flag RTA.
- `override_runtime_arguments` (`:361`) rewrites the writer's semaphore and barrier CRTAs on every hit.

## TTNN ProgramFactory

- **Concept (inherited from audit)**: `MeshWorkloadSpecFactoryConcept`, with the optional
  `override_runtime_arguments(attrs, tensor_args, tensor_return_value, const MeshCoordinateRange&)`.
- **Custom `compute_program_hash`**: present at `device/all_gather_async_device_operation.cpp:243` — leave intact.
- **Implementation notes**:
  - `create_mesh_workload_artifacts`: one `PerCoordProgram` per coordinate of `tensor_coords`, range = that coordinate
    (the legacy `WorkloadDescriptor` had one program per coordinate).
  - `MeshWorkloadArtifacts::global_semaphores` stays empty: both semaphores are caller-owned.
  - The override returns `tensor_args` for `input` and `output` and `global_semaphore_args` for `semaphore` and, when
    bound, `barrier_semaphore`. The miss run args use the same helper (`make_dispatch_run_args`), plus the kernel RTAs.
  - `kLlamaShardedReaderKernelIdx` / `kLlamaShardedWriterKernelIdx` and the `LlamaGather*CommonArgs` indices exist only
    for the index-based legacy override and the CRTA layout; the factory and kernels stop using them (the shared header
    keeps the structs).
  - `DefaultMeshWorkloadFactory` and `AllGatherViaBroadcastFactory` are untouched.

## Planned Spec Shape

- KernelSpecs: `reader`, `writer` (1:1 with the legacy descriptors).
- DataflowBufferSpecs: `data` (`entry_size = page_size`, `num_entries = input_pages / num_links + 1`, input format),
  `packet_header` (`entry_size = header_size`, `num_entries = 16`, `RawUInt32`).
- SemaphoreSpecs: none.
- GlobalSemaphoreParameters: `semaphore` (L1); `barrier_semaphore` (L1) when `barrier_semaphore.has_value() &&
  !using_persistent_buffers`.
- FabricConnectionSpecs: `forward`, `backward`; `src_fabric_node_id = get_fabric_node_id(coord)`; on each link core a
  target per existing neighbor, `link_idx = link`.
- TensorParameters: `input` (input `TensorSpec`), `output` (output `TensorSpec`).
- WorkUnitSpecs: `worker` = `{reader, writer}` on `sender_worker_core_range`.
- Per-coordinate programs: one per coordinate. Per coordinate: ring index, neighbors, target counts, route CTAs, per-link
  tile ranges and core lists, fabric targets.

## Preserved Multiplicity

none — one descriptor per kernel role.

## Dropped Plumbing

| legacy location (file:line) | legacy form | Metal 2.0 replacement |
|---|---|---|
| reader CRTA 0 (`factory.cpp:333`; `reader.cpp:37`) | input `Buffer*` | `TensorBinding(input, "input")`, Case 2 via `LocalTensorAccessor` |
| writer CRTA 0 (`:337-338`; `writer.cpp:52`) | output `Buffer*` | `TensorBinding(output, "output")`, Case 2 via `LocalTensorAccessor` |
| writer CRTA 1 (`:339`; `writer.cpp:53-54`) | semaphore address | `GlobalSemaphoreBinding(semaphore, "out_ready_semaphore")` (override 4a) |
| writer CRTA 2 (`:340`; `writer.cpp:64`) | barrier address | `GlobalSemaphoreBinding(barrier_semaphore, "barrier_semaphore")`, conditional (override 4a) |
| writer CTA 9 (`:177-178`; `writer.cpp:34, 107`) | `use_barrier_sem` | define `USE_BARRIER_SEM` (the binding's condition) |
| reader CTA 1, writer CTA 3 (`:150, 171`) | `cb0_id = c_in0` | `DFBBinding(data, "data", PRODUCER / CONSUMER)` |
| writer CTA 1 (`:169`) | `reserved_packet_header_cb_id = c_in1` | `DFBBinding(packet_header, "packet_header", PRODUCER + CONSUMER)` (self-loop) |
| writer RTA tail (`:313-326`) | flags + `append_fabric_connection_rt_args` | `FabricConnectionBinding(forward, "forward")`, `FabricConnectionBinding(backward, "backward")` (override 4b) |
| writer CTAs 10-21 (`:180-181`; `writer.cpp:36-39`) | positional route args | named CTAs `forward_mcast_*`, `backward_mcast_*` |
| core lists (reader RTA 4+, writer RTA 11+) | positional, variable count | runtime varargs, padded with zeros to the longest list in the program |
| all other positional CTAs / RTAs | positional | named, as in the Kernels table |

## Applied Patterns

- `MeshWorkloadSpecFactoryConcept` per-coordinate programs (`ttnn_factory.md`), with the translated
  `override_runtime_arguments`.
- Self-loop DFB binding (`port_patterns.md`): `packet_header` on `writer` (DM).
- Conditional / optional resource binding (`port_patterns.md`): `barrier_semaphore` + `USE_BARRIER_SEM`.
- Branch overrides 4a, 4b, 5, 6, 7.

## Deferred / Flagged

- The vararg lists differ in length per worker; padded to the longest (the deprecated per-node vararg count is not used).
- The legacy `log_trace` loops over the positional arg vectors go with the vectors.
