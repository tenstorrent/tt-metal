# Port Plan — all_to_all_async

Port plan for `ttnn/cpp/ttnn/operations/experimental/ccl/all_to_all_async`, ported from the per-coordinate
`ProgramDescriptor` API to Metal 2.0. Written during the inventory and planning steps; committed alongside the port for
review.

## Legacy Inventory

### Legacy factory shape
- Concept: `ProgramDescriptorFactoryConcept` through `create_descriptor(attrs, tensor_args, out, const
  std::optional<MeshCoordinate>&)` (`descriptor (per-coord)`), plus `override_runtime_arguments(Program&, …,
  optional<MeshCoordinate>)`. Methods live on `AllToAllAsyncProgram`, the one alternative of `program_factory_t`.
- Variants: single.
- Custom `compute_program_hash`: present at `device/all_to_all_async_device_operation.cpp:246` — left intact. It leaves
  out `semaphore`.

*(The Metal 2.0 factory concept was chosen during the audit; carried forward in [TTNN ProgramFactory](#ttnn-programfactory).)*

### Kernels

One program per mesh coordinate. `num_links` is forced to 1 by validation, so there is one sender core and `ring_size`
receiver cores.

| unique_id | source | core_ranges | CTAs (positional) | CTAs (named) | RTAs | CRTAs | defines | opt_level | config |
|---|---|---|---|---|---|---|---|---|---|
| sender reader (idx 0) | `device/kernels/interleaved_all_to_all_reader.cpp` | `sender_worker_core_range` | 0 `my_chip_id` = ring index; 1 `num_chips` = ring size; 2 `cb0_id` = `c_in0`; 3 `packet_size_in_pages`; 4 `tensor0_page_size`; 5 `num_targets_forward_direction`; 6 `num_targets_backward_direction`; 7+ `TensorAccessorArgs(input)` | none | 0 input `Buffer*`; 1 `in_row_tiles`; 2 `in_col_tiles`; 3 `input_row_device_stride`; 4 `input_col_device_stride`; 5 `input_shard_row_tiles`; 6 `input_shard_col_tiles`; 7 `out_row_offset` (= `out_row_start`); 8 `out_col_offset` (= `out_col_start`) | none | none | O2 (unset, DM) | `ReaderConfigDescriptor{}` |
| sender writer (idx 1) | `device/kernels/interleaved_all_to_all_writer.cpp` | `sender_worker_core_range` | 0 ring index; 1 ring size; 2 `reserved_packet_header_cb_id` = `c_in1`; 3 `num_packet_headers_storable` = 8; 4 `cb0_id` = `c_in0`; 5 `packet_size_in_pages`; 6 `tensor0_page_size`; 7/8 `num_targets_forward/backward_direction`; 9 `dynamic_alternate`; 10 `chunk_granularity`; 11 `contig_pages_advanced`; 12 `num_dram_banks`; 13+ `TensorAccessorArgs(intermediate)`, `TensorAccessorArgs(output)` | none | 0 intermediate `Buffer*`; 1 output `Buffer*` (dead); 2 semaphore address; 3 `out_row_tiles`; 4 `out_col_tiles`; 5 `out_row_start`; 6 `out_col_start`; 7 `input_shard_row_tiles`; 8 `input_shard_col_tiles`; 9 `wait_output_semaphore`; 10 `reset_global_semaphore`; 11 `receiver_core_x`; 12 `receiver_core_y`; 13+ fabric tail (forward flag, conn args, backward flag, conn args) | none | none | O2 (unset, DM) | `WriterConfigDescriptor{}` |
| receiver writer (idx 2) | `device/kernels/interleaved_all_to_all_receiver_writer.cpp` | `receiver_worker_core_range` | 0 ring index; 1 ring size; 2 `num_pages_per_packet`; 3 `chunk_granularity`; 4 `chunk_num_tiles`; 5 `num_chunks_per_shard`; 6 `page_size`; 7 `cb_id` = `c_in0`; 8+ `TensorAccessorArgs(output)` | none | 0 output `Buffer*`; 1 `in_row_tiles`; 2 `in_col_tiles`; 3/4 `input_row/col_device_stride`; 5/6 `input_shard_row/col_tiles`; 7/8 `out_row/col_start`; 9/10 `out_row/col_tiles`; 11 `num_pages_per_packet`; 12 `remote_device_ring_id` (all per receiver core `i`) | none | none | O2 (unset, DM) | `WriterConfigDescriptor{}` |
| receiver reader (idx 3) | `device/kernels/interleaved_all_to_all_receiver_reader.cpp` | `receiver_worker_core_range` | 0–7 as the receiver writer; 8 `contig_pages_advanced` = `pages_per_packet`; 9 `num_dram_banks`; 10+ `TensorAccessorArgs(input)`, `TensorAccessorArgs(intermediate)` | none | 0 intermediate `Buffer*`; 1 input `Buffer*`; 2 semaphore address; 3–14 as receiver writer RTAs 1–12 | none | none | O2 (unset, DM) | `ReaderConfigDescriptor{}` |

### CBs
| index | total_size | core_ranges | data_format | page_size | tile (if set) |
|---|---|---|---|---|---|
| `c_in0` (sender) | `3 * pages_per_packet * page_size` | `sender_worker_core_range` | input dtype | `page_size` | not set |
| `c_in1` (packet header) | `8 * header_size * 2` | `sender_worker_core_range` | `RawUInt32` | `header_size` | not set |
| `c_in0` (receiver) | `3 * pages_per_packet * page_size` | `receiver_worker_core_range` | input dtype | `page_size` | not set |

### Semaphores
none (no `SemaphoreDescriptor`). One caller-supplied GlobalSemaphore rides as a raw address RTA (sender writer 2,
receiver reader 2).

### Tensor accessors
| host site (file:line) | originating Tensor | RTA slot (host) |
|---|---|---|
| `all_to_all_async_program_factory.cpp:327` (sender reader CTAs) | input | sender reader RTA0 (`:478`) |
| `:357-358` (sender writer CTAs) | intermediate | sender writer RTA0 (`:540`) |
| `:359` (sender writer CTAs) | output | sender writer RTA1 (`:541`) — accessor never used |
| `:382-383` (receiver writer CTAs) | output | receiver writer RTA0 (`:588`) |
| `:404` (receiver reader CTAs) | input | receiver reader RTA1 (`:567`) |
| `:405-406` (receiver reader CTAs) | intermediate | receiver reader RTA0 (`:566`) |

### Work split
n/a — `choose_worker_cores` picks 1 sender core per link and `ring_size` receiver cores per link; no work split.

### Shared kernels
none. All four kernels are op-owned; `grep -rn interleaved_all_to_all_ ttnn/ tests/ models/ tt_metal/` hits only this
factory. Converted in place.

### Flags
- Fabric: `append_fabric_connection_rt_args` (`:529`, `:535`) on the sender core, link index = `link`: forward neighbor,
  then backward neighbor, each only when the neighbor exists, each after a flag RTA.
- `override_runtime_arguments` (`:607`) rewrites the input, intermediate, output and semaphore address slots of all four
  kernels on every hit.

## TTNN ProgramFactory

- **Concept (inherited from audit)**: `MeshWorkloadSpecFactoryConcept`, with the optional
  `override_runtime_arguments(attrs, tensor_args, tensor_return_value, const MeshCoordinateRange&)`.
- **Custom `compute_program_hash`**: present at `device/all_to_all_async_device_operation.cpp:246` — leave intact.
- **Implementation notes**:
  - `create_mesh_workload_artifacts`: one `PerCoordProgram` per coordinate of `tensor_coords`, range = that coordinate
    (the legacy adapter called `create_descriptor` once per coordinate).
  - `MeshWorkloadArtifacts::global_semaphores` stays empty: the semaphore is caller-owned.
  - The override returns `tensor_args` for `input`, `intermediate`, `output` and `global_semaphore_args` for
    `semaphore` — the four things the ported-from override wrote. The cache-miss run args use the same helper
    (`make_dispatch_run_args`), plus the kernel RTAs.
  - The kernel push-order and RTA-slot constants (`kWorkerSenderReaderKernelIdx`, …, `kReceiverReaderSemaphoreAddr`)
    exist only for the index-based legacy override; they go away.

## Planned Spec Shape

- KernelSpecs: `sender_reader`, `sender_writer`, `receiver_writer`, `receiver_reader` (1:1 with the legacy descriptors).
- DataflowBufferSpecs: `sender_data` (`entry_size = page_size`, `num_entries = 3 * pages_per_packet`), `packet_header`
  (`entry_size = header_size`, `num_entries = 16`, `RawUInt32`), `receiver_data` (`entry_size = page_size`,
  `num_entries = 3 * pages_per_packet`).
- SemaphoreSpecs: none.
- GlobalSemaphoreParameters: `semaphore` (L1).
- FabricConnectionSpecs: `forward`, `backward`; `src_fabric_node_id = get_fabric_node_id(coord)`; on each sender core a
  target per existing neighbor, `link_idx = link`.
- TensorParameters: `input` (input `TensorSpec`), `intermediate` (intermediate `TensorSpec`), `output` (output
  `TensorSpec`).
- WorkUnitSpecs: `sender` = `{sender_reader, sender_writer}` on `sender_worker_core_range`; `receiver` =
  `{receiver_writer, receiver_reader}` on `receiver_worker_core_range`.
- Per-coordinate programs: one per coordinate. Per coordinate: ring index, neighbors, `num_targets_forward/backward`,
  the `my_ring_id` CTA of all four kernels, the sender RTAs and receiver core coordinates, fabric targets
  (`all_to_all_async_program_factory.cpp:159-246, 423-546`).

## Preserved Multiplicity

none — no work-split multiplicity in legacy (one descriptor per kernel role).

## Dropped Plumbing

| legacy location (file:line) | legacy form | Metal 2.0 replacement |
|---|---|---|
| sender reader RTA0 (`factory.cpp:478`; `reader.cpp:231`) | input `Buffer*` | `TensorBinding(input, "input")` |
| sender writer RTA0 (`:540`; `writer.cpp:50`) | intermediate `Buffer*` | `TensorBinding(intermediate, "intermediate")` |
| sender writer RTA1 (`:541`; `writer.cpp:51, 94-95`) | output `Buffer*`, accessor never used | dropped, no binding |
| receiver reader RTA0/1 (`:566-567`; `receiver_reader.cpp:32-33`) | intermediate / input `Buffer*` | `TensorBinding(intermediate)`, `TensorBinding(input)` |
| receiver writer RTA0 (`:588`; `receiver_writer.cpp:159`) | output `Buffer*` | `TensorBinding(output, "output")` |
| all `TensorAccessorArgs(...).append_to` (`:327, 357-359, 382-383, 404-406`) | accessor CTAs + `TensorAccessorArgs<N>()` chains | tensor bindings |
| sender reader CTA 2, writer CTA 4 (`:321, 347`) | `cb0_id = c_in0` | `DFBBinding(sender_data, "sender_data", PRODUCER / CONSUMER)` |
| sender writer CTA 2 (`:345`) | `reserved_packet_header_cb_id = c_in1` | `DFBBinding(packet_header, "packet_header", PRODUCER + CONSUMER)` (self-loop) |
| receiver CTA 7 (`:381, 401`) | `cb_id = c_in0` | `DFBBinding(receiver_data, "receiver_data", PRODUCER / CONSUMER)` |
| sender writer RTA2, receiver reader RTA2 (`:506, 570`) | semaphore address | `GlobalSemaphoreBinding(semaphore, "semaphore")` (override 4a) |
| sender writer RTA 13+ (`:526-537`) | flags + `append_fabric_connection_rt_args` tails | `FabricConnectionBinding(forward, "forward")`, `FabricConnectionBinding(backward, "backward")` (override 4b) |
| all positional CTAs / RTAs | positional | named, as in the Kernels table (CTA 0 `my_chip_id` → `my_ring_id`, CTA 1 → `ring_size`, `N_DRAM_BANKS` → `num_dram_banks`; receiver CTA 2 → `packet_size_in_pages`, because the RTA `num_pages_per_packet` takes that name) |

## Applied Patterns

- `MeshWorkloadSpecFactoryConcept` per-coordinate programs (`ttnn_factory.md`), with the translated
  `override_runtime_arguments`.
- Self-loop DFB binding (`port_patterns.md`): `packet_header` on `sender_writer` (DM).
- Branch overrides 4a (GlobalSemaphore binding), 4b/6/7 (two fabric connection specs bound by one writer, each sender
  built only under its flag).

## Deferred / Flagged

- The receiver kernels shadow the CTA `num_pages_per_packet` with an RTA of the same name. The CTA gets the arg name
  `packet_size_in_pages`; both kernel variables stay.
