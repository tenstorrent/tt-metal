# Port Plan — all_to_all_async

Port plan for `experimental/ccl/all_to_all_async`, ported from the `ProgramDescriptor` API (per-coordinate `create_descriptor`) to Metal 2.0.
Written during the inventory and planning steps; committed alongside the port for review.

## Legacy Inventory

### Legacy factory shape
- Concept: `ProgramDescriptorFactoryConcept`, per-coordinate: `AllToAllAsyncProgram::create_descriptor(…, const std::optional<MeshCoordinate>&)` (`device/all_to_all_async_program_factory.cpp:149`) plus `override_runtime_arguments` (`:607`). The factory is a struct in `program_factory_t` (`device/all_to_all_async_device_operation.hpp:26`), not the direct-descriptor shape.
- Variants: single.
- Custom `compute_program_hash`: present at `device/all_to_all_async_device_operation.cpp:246`. Left intact.

### Kernels
All `KernelDescriptor`s, push order = legacy kernel index. `ring_index` is the coordinate's position in `get_ring_coordinates()`.

| unique_id | source | core_ranges | CTAs (positional) | RTAs | CRTAs | defines | opt_level | config |
|---|---|---|---|---|---|---|---|---|
| sender_reader | `kernels/interleaved_all_to_all_reader.cpp` | `sender_worker_core_range` (1 core per link) | 0 `ring_index`, 1 `ring_size`, 2 `c_in0`, 3 `pages_per_packet`, 4 `page_size`, 5 `num_targets_forward`, 6 `num_targets_backward`, 7.. `TensorAccessorArgs(input)` | 0 `input` (`Buffer*`), 1 `in_row_tiles`, 2 `in_col_tiles`, 3 `input_row_device_stride`, 4 `input_col_device_stride`, 5 `input_shard_row_tiles`, 6 `input_shard_col_tiles`, 7 `out_row_start`, 8 `out_col_start` | none | none | O2 (unset, DM) | `ReaderConfigDescriptor` (reader default) |
| sender_writer | `kernels/interleaved_all_to_all_writer.cpp` | `sender_worker_core_range` | 0 `ring_index`, 1 `ring_size`, 2 `c_in1`, 3 `PACKET_HEADER_BUFFER_SIZE`, 4 `c_in0`, 5 `pages_per_packet`, 6 `page_size`, 7 `num_targets_forward`, 8 `num_targets_backward`, 9 `dynamic_alternate`, 10 `chunk_granularity`, 11 `contig_pages_advanced`, 12 `N_DRAM_BANKS`, 13.. `TensorAccessorArgs(intermediate)`, then `TensorAccessorArgs(output)` | 0 `intermediate` (`Buffer*`), 1 `output` (`Buffer*`), 2 `semaphore.address()`, 3 `out_row_tiles`, 4 `out_col_tiles`, 5 `out_row_start`, 6 `out_col_start`, 7 `input_shard_row_tiles`, 8 `input_shard_col_tiles`, 9 `wait_output_semaphore`, 10 `reset_global_semaphore`, 11 `receiver_core_x`, 12 `receiver_core_y`, 13 forward present-flag, [forward connection args], backward present-flag, [backward connection args] | none | none | O2 | `WriterConfigDescriptor` (writer default) |
| receiver_writer | `kernels/interleaved_all_to_all_receiver_writer.cpp` | `receiver_worker_core_range` (`ring_size` cores per link) | 0 `ring_index`, 1 `ring_size`, 2 `pages_per_packet`, 3 `chunk_granularity`, 4 `chunk_num_tiles`, 5 `num_chunks_per_shard`, 6 `page_size`, 7 `c_in0`, 8.. `TensorAccessorArgs(output)` | 0 `output` (`Buffer*`), 1 `in_row_tiles`, 2 `in_col_tiles`, 3 `input_row_device_stride`, 4 `input_col_device_stride`, 5 `input_shard_row_tiles`, 6 `input_shard_col_tiles`, 7 `out_row_start`, 8 `out_col_start`, 9 `out_row_tiles`, 10 `out_col_tiles`, 11 `pages_per_packet`, 12 `remote_device_ring_id` (= receiver index `i`) | none | none | O2 | `WriterConfigDescriptor` (writer default) |
| receiver_reader | `kernels/interleaved_all_to_all_receiver_reader.cpp` | `receiver_worker_core_range` | 0 `ring_index`, 1 `ring_size`, 2 `pages_per_packet`, 3 `chunk_granularity`, 4 `chunk_num_tiles`, 5 `num_chunks_per_shard`, 6 `page_size`, 7 `c_in0`, 8 `pages_per_packet` (as `contig_pages_advanced`), 9 `N_DRAM_BANKS`, 10.. `TensorAccessorArgs(input)`, then `TensorAccessorArgs(intermediate)` | 0 `intermediate` (`Buffer*`), 1 `input` (`Buffer*`), 2 `semaphore.address()`, 3–14 as receiver_writer 1–12 | none | none | O2 | `ReaderConfigDescriptor` (reader default) |

### CBs
| index | total_size | core_ranges | data_format | page_size | tile |
|---|---|---|---|---|---|
| `c_in0` (sender data) | `3 * pages_per_packet * page_size` | sender cores | input dtype | `page_size` | unset |
| `c_in1` (packet headers) | `8 * packet_header_size * 2` | sender cores | `RawUInt32` | `packet_header_size` | unset |
| `c_in0` (receiver data) | `3 * pages_per_packet * page_size` | receiver cores | input dtype | `page_size` | unset |

### Semaphores
None declared by the op. `append_fabric_connection_rt_args` adds two program semaphores per connection (teardown, buffer index) on each sender core (`tt_metal/fabric/fabric.cpp:218-235`); in Metal 2.0 the runtime allocates them for the fabric binding.

GlobalSemaphores: one, `operation_attributes.semaphore`, caller-supplied, excluded from the hash. The sender writer reads it at RTA 2 (`interleaved_all_to_all_writer.cpp:52`, `global_semaphore_addr`) to build the remote receiver core's NoC address (`safe_get_noc_addr`, `:102-103`). The receiver reader reads it at RTA 2 (`interleaved_all_to_all_receiver_reader.cpp:34`, `global_semaphore_addr`): `noc_semaphore_wait_min` (`:108`) and `noc_semaphore_set` (`:130`). The override re-applies it on both (`:619, 640, 664`).

Fabric connections: sender writer only, `append_fabric_connection_rt_args` per sender core `sender_worker_cores[link]` on link `link`. Forward side (peer `fabric_node_ids[ring_index + 1]`, wrapping on `Ring`) and backward side (peer `[ring_index - 1]`, wrapping), each behind a present-flag. Kernel: `FabricConnectionManager::build_from_args<BUILD_AND_OPEN_CONNECTION_START_ONLY>` (`interleaved_all_to_all_writer.cpp:65-66`), then `open_finish` and `close`.

### Tensor accessors
| host site (file:line) | originating Tensor | RTA slot (host) |
|---|---|---|
| `:327` (sender reader CTA 7..) | `input_tensor` | sender reader RTA 0 (`:478`) |
| `:357-358` (sender writer CTA 13..) | `persistent_intermediate_buffer` | sender writer RTA 0 (`:540`) |
| `:359` (sender writer, chained) | `persistent_output_buffer` | sender writer RTA 1 (`:541`) (accessor unused in the kernel) |
| `:382-383` (receiver writer CTA 8..) | `persistent_output_buffer` | receiver writer RTA 0 (`:588`) |
| `:404` (receiver reader CTA 10..) | `input_tensor` | receiver reader RTA 1 (`:567`) |
| `:405-406` (receiver reader, chained) | `persistent_intermediate_buffer` | receiver reader RTA 0 (`:566`) |

### Work split
n/a. `ttnn::ccl::choose_worker_cores` picks `num_links` sender cores and `num_links * (1 + ring_size)` total cores; receivers are total minus senders (`:256-266`). Sender RTAs are per link; receiver RTAs are per receiver core `i`, as if from device `i`.

### Shared kernels
None. Each kernel file is bound only by this factory (grep of `ttnn/`, `tests/`, `models/`; the only other hits are `experimental/ccl/CMakeLists.txt`). Convert in place.

### Flags
- The op aborts on 2 devices (audit, Misc anomalies), so N300 verification is limited.
- Dead kernel args (audit, Misc anomalies): kept as named args. The recipe gives no rule for a dead RTA/CTA, and a 1:1 translation preserves the legacy kernel's reads.
- `num_pages_per_packet` is both a CTA (file scope) and an RTA (local, shadowing it) in both receiver kernels. Metal 2.0 arg names share one namespace per kernel (`kernel_spec.hpp:225-226`), so the two can't both be named `num_pages_per_packet`.

## TTNN ProgramFactory

- **Concept (inherited from audit)**: `MeshWorkloadSpecFactoryConcept`, with the translated `override_runtime_arguments(attrs, args, ret, range)`.
- **Custom `compute_program_hash`**: present at `device/all_to_all_async_device_operation.cpp:246`. Leave intact.
- **Implementation notes**:
  - Keep the struct name `AllToAllAsyncProgram`, so `program_factory_t` is unchanged.
  - `create_mesh_workload_artifacts` iterates `tensor_coords.coords()` and builds one `PerCoordProgram` per coordinate (`range = MeshCoordinateRange(coord)`) through a per-coordinate helper holding the legacy body.

## Planned Spec Shape

Per coordinate:

- **KernelSpecs**: `sender_reader`, `sender_writer`, `receiver_writer`, `receiver_reader`. 1:1 with the legacy descriptors, same sources, edited in place.
- **DataflowBufferSpecs**:
  - `sender_data` (`c_in0`): `entry_size = page_size`, `num_entries = 3 * pages_per_packet`, `data_format_metadata = input dtype`.
  - `packet_headers` (`c_in1`): `entry_size = packet_header_size`, `num_entries = 16`, `RawUInt32`.
  - `receiver_data` (receiver `c_in0`): `entry_size = page_size`, `num_entries = 3 * pages_per_packet`, input dtype.
  - No `tile` set in legacy, so no `tile_format_metadata`.
- **SemaphoreSpecs**: none.
- **TensorParameters**: `input`, `intermediate`, `output`, from each tensor's `tensor_spec()`. Strict, relaxation `none`.
- **GlobalSemaphoreParameters**: `chunk_ready` (the receiver's per-chunk counter), bound by the sender writer and the receiver reader. `global_semaphore_args = {chunk_ready → attrs.semaphore}`.
- **Fabric connections** (`advanced_options.fabric_connections`): `forward` and `backward`, each a one-target spec.
  - Targets come from `fabric_target(self, peer)` (`ccl_port.md`'s helper).
  - `links = {sender_worker_cores[link] → link}` for each link.
  - Each is declared and bound (`sender_writer.advanced_options.fabric_connection_bindings`) only when that side exists.
- **WorkUnitSpecs**: `sender` {sender_reader, sender_writer} on `sender_worker_core_range`; `receiver` {receiver_writer, receiver_reader} on `receiver_worker_core_range`.
- **Per-coordinate programs**: one entry per coordinate in `tensor_coords`.
  - Varies per coordinate: `ring_index` (CTA of all four kernels), forward/backward peers (`:173-193`), `receiver_core_x/y` (`:462-463`), sender strides/offsets (`:437-438`).
- **Op-owned tensors**: none.

## Preserved Multiplicity

none. No work-split multiplicity in legacy: each source is bound once, and sender and receiver kernels run on disjoint core sets.

## Dropped Plumbing

| legacy location (file:line) | legacy form | Metal 2.0 replacement |
|---|---|---|
| factory `:478`; reader RTA 0 | `input_tensor.buffer()` | `TensorBinding(input → "input")` |
| factory `:540`; writer RTA 0 | `persistent_intermediate_buffer.buffer()` | `TensorBinding(intermediate → "intermediate")` |
| factory `:541`; writer RTA 1 | `persistent_output_buffer.buffer()` | `TensorBinding(output → "output")` |
| factory `:506`; writer RTA 2 | `semaphore.address()` (`smuggled-rta-ok`) | `GlobalSemaphoreBinding(chunk_ready)`; kernel `GlobalSemaphore(global_sem::chunk_ready).address()` |
| factory `:526-537`; writer RTA 13.. | present-flags + `append_fabric_connection_rt_args` | `FabricConnectionsSpec` `forward`/`backward` + bindings; kernel `build_from_optional_bindings` |
| factory `:588`; receiver writer RTA 0 | `persistent_output_buffer.buffer()` | `TensorBinding(output → "output")` |
| factory `:566`; receiver reader RTA 0 | `persistent_intermediate_buffer.buffer()` | `TensorBinding(intermediate → "intermediate")` |
| factory `:567`; receiver reader RTA 1 | `input_tensor.buffer()` | `TensorBinding(input → "input")` |
| factory `:570`; receiver reader RTA 2 | `semaphore.address()` | `GlobalSemaphoreBinding(chunk_ready)` |
| reader CTA 2 | `c_in0` | `DFBBinding(sender_data → "data", PRODUCER)` |
| writer CTA 2 | `c_in1` | `DFBBinding(packet_headers → "packet_headers", PRODUCER + CONSUMER)` (self-loop) |
| writer CTA 4 | `c_in0` | `DFBBinding(sender_data → "data", CONSUMER)` |
| receiver writer CTA 7 | `c_in0` | `DFBBinding(receiver_data → "data", CONSUMER)` |
| receiver reader CTA 7 | `c_in0` | `DFBBinding(receiver_data → "data", PRODUCER)` |
| all `TensorAccessorArgs(...).append_to` (`:327, 357-359, 382-383, 404-406`) and kernel `TensorAccessorArgs<N>()` chains | accessor CTA plumbing | the bindings above |
| factory constants `kSender*Addr` / `kReceiver*Addr` (`:43-50`) and `k*KernelIdx` (`:37-40`) | positional slot indices | gone (named args, named kernels) |
| writer `size_t arg_for_fab = arg_idx` (`:63`) | positional cursor snapshot, unused | gone with the positional cursor |
| positional CTAs/RTAs (all others) | positional | named, after the kernel's variable names. **Exception:** the receivers' dead file-scope CTA `num_pages_per_packet` is named `num_pages_per_packet_ct`, because the RTA read by the same name inside `kernel_main` (which shadows it) keeps `num_pages_per_packet`. The kernel variables are unchanged. |

## Applied Patterns

- **Self-loop DFB binding** (`port_patterns.md`): `packet_headers` on `sender_writer`, PRODUCER + CONSUMER under one accessor name.
- **`ccl_port.md` — Caller-supplied: rebind it in the override**: `chunk_ready`.
- **`ccl_port.md` — One sender, or a FabricConnectionManager**: `forward` / `backward`.
- **`ttnn_factory.md` — Per-coordinate programs**, with an override.

## Deferred / Flagged

- **Recipe conflict on the semaphore address crossing the op boundary.** The writer passes `output_semaphore_noc_addr_in_pkt`, computed from the `GlobalSemaphore`'s address, into `perform_atomic_fabric_write` in `ccl/common/kernels/minimal_ccl_common.hpp`. `metal2_port.md` rule 3's boundary note calls that an assumption violation (stop); `ccl_port.md` § Kernel side says helpers in or outside the op directory stay unchanged. Proceeding per `ccl_port.md`: the address is a plain integer that the kernel holds, so nothing is structurally unavailable (§ Read this first's definition of the violation).
