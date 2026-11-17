# Port Plan — point_to_point

Port plan for `ttnn/cpp/ttnn/operations/point_to_point`, ported from the `WorkloadDescriptor` API to Metal 2.0.
Written during the inventory and planning steps; committed alongside the port for review.

Branch-specific overrides from `METAL2_PORT_BRIEF.md` apply: GlobalSemaphore parameters/bindings (1),
op-owned GlobalSemaphores in `MeshWorkloadArtifacts::global_semaphores` (2), fabric connections (3), and the
kernel-side exception for the `global_sem::` and `fabric::` bindings (4).

## Legacy Inventory

### Legacy factory shape
- Concept: `WorkloadDescriptor` — `PointToPointOp::SendReceive::create_workload_descriptor`
  (`device/host/point_to_point_device_op.cpp:199`), methods on a `program_factory_t` variant member (`SendReceive`).
- Variants: one factory. It emits one of two program sets:
  - `send_coord != receive_coord`: a send program on `send_coord` (`send_program_factory`) and a receive program on
    `receive_coord` (`receive_program_factory`), plus one op-owned GlobalSemaphore in `WorkloadDescriptor::semaphores`.
  - `send_coord == receive_coord`: one local-copy program on that coordinate (`local_copy_program_factory`).
- Custom `compute_program_hash`: none — default reflection-based hash (`attribute_values()` at
  `point_to_point_device_op.hpp:30`).
- No `override_runtime_arguments`, no `get_dynamic_runtime_args`, no pybound `create_descriptor`.

### Variant: send program (`device/host/send_program_factory.cpp`)

#### Kernels
| unique_id | source | core_ranges | CTAs (positional) | CTAs (named) | RTAs | CRTAs | defines | opt_level | config |
|---|---|---|---|---|---|---|---|---|---|
| reader | `reader_unary_interleaved_start_id_gen.cpp` | `all_cores` (1 node, `{1,1}` split) | `TensorAccessorArgs(input)` | — | 0 `Buffer*` input, 1 num_tiles (`increment`), 2 start_id, 3 page_bytes | — | — | O2 (unset, DM) | `ReaderConfigDescriptor{}` |
| writer | `writer_send.cpp` | `all_cores` | 0 sender CB id, 1 packet-header CB id, 2 packet CB id, 3 `l1_alignment`, 4.. `TensorAccessorArgs(intermediate)` | — | 0 `Buffer*` intermediate, 1 page_idx_start, 2 page_idx_end, 3 num_hops, 4 page_size_bytes, 5 payload (packet) size, 6 pages per packet, 7 page segments, 8 GlobalSemaphore address, 9 `dst_is_forward`, then the `FabricConnectionManager` tail (fwd flag, [conn args], bwd flag, [conn args]) | — | — | O2 (unset, DM) | `WriterConfigDescriptor{}` |

#### CBs
| index | total_size | core_ranges | data_format | page_size | tile |
|---|---|---|---|---|---|
| c_0 sender | 2 × aligned input page | all_cores | input dtype | aligned input page | — |
| c_1 packet header | 2 × 2 × header size | all_cores | RawUInt32 | header size | — |
| c_2 packet | packet_size_bytes | all_cores | input dtype | packet_size_bytes | — |

#### Semaphores
none (program semaphores). One op-owned GlobalSemaphore address in writer RTA 8.

#### Tensor accessors
| host site | originating Tensor | RTA slot |
|---|---|---|
| `send_program_factory.cpp:102`, RTA `:143` | `input_tensor` | reader RTA 0 |
| `send_program_factory.cpp:115`, RTA `:179` | `tensor_return_value[0]` (intermediate) | writer RTA 0 |

#### Work split
- Driver: `split_work_to_cores(CoreCoord{1,1}, total_packets)` — one node, group 1 only.

#### Fabric
- `append_fabric_connection_rt_args(*mesh_device, this_fabric_id = get_fabric_node_id(send_coord), next_fabric_id, link_idx = 0, desc, c, …)` on the writer, appended in the forward or backward slot by `dst_is_forward` (`:168,:173`). `next_fabric_id` from `detail::fabric_1d_routing` (`:103`).

### Variant: receive program (`device/host/receive_program_factory.cpp`)

#### Kernels
| unique_id | source | core_ranges | CTAs (positional) | CTAs (named) | RTAs | CRTAs | defines | opt_level | config |
|---|---|---|---|---|---|---|---|---|---|
| reader | `reader_receive.cpp` | `all_cores` (1 node) | 0 packet-header CB id, 1 packet CB id, 2 receiver CB id, 3 `l1_alignment`, 4.. `TensorAccessorArgs(intermediate)` | — | 0 page_idx_start, 1 page_idx_end, 2 pages per packet, 3 `Buffer*` intermediate, 4 packet size, 5 output page size, 6 page segments, 7 GlobalSemaphore address, 8 num_hops, 9 `sender_is_forward`, then the `FabricConnectionManager` tail | — | — | O2 | `ReaderConfigDescriptor{}` |
| writer | `writer_unary_interleaved_start_id_gen.cpp` | `all_cores` | 0 receiver CB id, 1.. `TensorAccessorArgs(output)` | — | 0 `Buffer*` output, 1 num_tiles, 2 start_id, 3 page_bytes | — | — | O2 | `WriterConfigDescriptor{}` |

#### CBs
| index | total_size | core_ranges | data_format | page_size | tile |
|---|---|---|---|---|---|
| c_0 packet header | 2 × 2 × header size | all_cores | RawUInt32 | header size | — |
| c_1 packet | packet_size_bytes | all_cores | intermediate dtype (UInt8) | packet_size_bytes | — |
| c_2 receiver | 3 × pages_per_packet × output page | all_cores | intermediate dtype (UInt8) | output page size | — |

#### Tensor accessors
| host site | originating Tensor | RTA slot |
|---|---|---|
| `receive_program_factory.cpp:90`, RTA `:167` | `tensor_return_value[0]` (intermediate) | reader RTA 3 |
| `receive_program_factory.cpp:103`, RTA `:177` | `tensor_return_value[1]` (output) | writer RTA 0 |

#### Work split / Fabric
- `split_work_to_cores(CoreCoord{1,1}, total_packets)`; fabric args on the reader, same shape as send, with `this_fabric_id = get_fabric_node_id(receive_coord)` and the route from `fabric_1d_routing(mesh_device, receive_coord, send_coord, topology)` (`:90`).

### Variant: local-copy program (`device/host/local_copy_program_factory.cpp`)

#### Kernels
| unique_id | source | core_ranges | CTAs | RTAs | opt_level | config |
|---|---|---|---|---|---|---|
| reader | `reader_unary_interleaved_start_id_gen.cpp` | `all_cores` (worker grid split) | `TensorAccessorArgs(input)` | 0 `Buffer*` input, 1 num_tiles, 2 start_id, 3 page_bytes | O2 | `ReaderConfigDescriptor{}` |
| writer | `writer_unary_interleaved_start_id_gen.cpp` | `all_cores` | 0 CB id, 1.. `TensorAccessorArgs(output)` | 0 `Buffer*` output, 1 num_tiles, 2 start_id, 3 page_bytes | O2 | `WriterConfigDescriptor{}` |

#### CBs
| index | total_size | data_format | page_size |
|---|---|---|---|
| c_0 | 2 × aligned page | input dtype | aligned page |

#### Work split
- `split_work_to_cores(compute_with_storage_grid_size(), num_pages)`; groups 1 and 2 run the same kernels with per-node RTAs, so no multiplicity.

### Shared kernels
All four kernels live in this op's directory and no other op factory binds them (`grep -rl` over `ttnn/cpp/ttnn/operations/`
finds only this op's factories). `tests/ttnn/unit_tests/operations/debug/test_generic_op.py::test_point_to_point` drives all four
through the V1 `ProgramDescriptor` path with positional args, so they are *lent* kernels. No `_metal2` sibling exists.
Rung 2: create `<stem>_metal2.cpp` beside each original, add the pointer comment to each original.

### Flags
- `device/kernels/common.hpp` is also included by four `ccl/reduce_to_root` kernels; the forks do not include it. Left unchanged.
- `extract_tensor_buffers_t<tensor_args_t>` specialization (`point_to_point_device_op.hpp:149`) is read only by the legacy descriptor adapter; the Metal 2.0 adapter enumerates tensors with `visit_object_of_type<ttnn::Tensor>`. Left unchanged (outside the factory); see report.

## TTNN ProgramFactory

- **Concept (inherited from audit)**: `MeshWorkloadSpecFactoryConcept` — `SendReceive::create_mesh_workload_artifacts` replaces `create_workload_descriptor`.
- **Custom `compute_program_hash`**: none.
- **Implementation notes**: the three program builders return a `ProgramArtifacts` (spec + run args, no op-owned tensors); `create_mesh_workload_artifacts` places each on the same single-coordinate range as today. The GlobalSemaphore is created (and `Synchronize` runs) exactly as today, then moved into `MeshWorkloadArtifacts::global_semaphores`; the run args reference that vector element.

## Planned Spec Shape

### Variant: send program (range `send_coord`)
- KernelSpecs: `reader` → `reader_unary_interleaved_start_id_gen_metal2.cpp`; `writer` → `writer_send_metal2.cpp`.
- DataflowBufferSpecs: `sender` (entry aligned input page × 2, input format); `packet_header` (entry header size × 4, RawUInt32); `packet` (entry packet_size × 1, input format).
- SemaphoreSpecs: none.
- TensorParameters: `input`, `intermediate`.
- GlobalSemaphoreParameters: `receive_semaphore` (L1), bound by `writer` as `receive_semaphore`.
- FabricConnectionSpecs: `connection` — `src = get_fabric_node_id(send_coord)`, per node `{next_fabric_id, link_idx 0}`; bound by `writer` as `connection`.
- WorkUnitSpecs: one, `{reader, writer}` on `all_cores`.

### Variant: receive program (range `receive_coord`)
- KernelSpecs: `reader` → `reader_receive_metal2.cpp`; `writer` → `writer_unary_interleaved_start_id_gen_metal2.cpp`.
- DataflowBufferSpecs: `packet_header` (header × 4, RawUInt32); `packet` (packet_size × 1, intermediate format); `receiver` (output page × 3·pages_per_packet, intermediate format).
- TensorParameters: `intermediate`, `output`.
- GlobalSemaphoreParameters: `sender_semaphore` (L1), bound by `reader` as `sender_semaphore`.
- FabricConnectionSpecs: `connection` — `src = get_fabric_node_id(receive_coord)`, per node `{next_fabric_id, link_idx 0}`; bound by `reader`.
- WorkUnitSpecs: one, `{reader, writer}` on `all_cores`.

### Variant: local-copy program (range `send_coord`)
- KernelSpecs: `reader` → `reader_unary…_metal2.cpp`; `writer` → `writer_unary…_metal2.cpp`.
- DataflowBufferSpecs: `pages` (aligned page × 2, input format).
- TensorParameters: `input`, `output`.
- WorkUnitSpecs: one, `{reader, writer}` on `all_cores`.

### Per-coordinate programs
| Ported-from program | range | per-coordinate values |
|---|---|---|
| send | `MeshCoordinateRange(send_coord)` | `this_fabric_id`, route (`send_program_factory.cpp:100-104`) |
| receive | `MeshCoordinateRange(receive_coord)` | `this_fabric_id`, route (`receive_program_factory.cpp:89-91`) |
| local copy | `MeshCoordinateRange(send_coord)` | — |

### Op-owned tensors
none. Op-owned GlobalSemaphore: one (send/receive path), in `MeshWorkloadArtifacts::global_semaphores`.

## Preserved Multiplicity

none — no work-split multiplicity in legacy (groups 1 and 2 share one kernel descriptor each; counts are RTAs in legacy).

## Dropped Plumbing

| legacy location | legacy form | Metal 2.0 replacement |
|---|---|---|
| reader_unary RTA 0 (send `:143`, local `:96`) | `input.buffer()` | `TensorBinding(input → src)` |
| reader_unary CTA 0.. | `TensorAccessorArgs(input)` | binding |
| reader_unary `:26` 3rd arg | `page_bytes` | dropped from accessor; `page_bytes` stays a named RTA (transfer size) |
| reader_unary `:18` | hardcoded `cb_id_in0 = 0` | `dfb::in0` |
| writer_unary RTA 0 (recv `:177`, local `:103`) | `output.buffer()` | `TensorBinding(output → dst)` |
| writer_unary CTA 0 | CB id | `dfb::out` |
| writer_unary CTA 1.. | `TensorAccessorArgs(output)` | binding |
| writer_unary `:24` 3rd arg | `page_bytes` | dropped from accessor; named RTA stays |
| writer_send CTA 0/1/2 | CB ids | `dfb::sender`, `dfb::packet_header`, `dfb::packet` |
| writer_send CTA 3 | `l1_alignment` | named CTA `alignment` |
| writer_send CTA 4.. | `TensorAccessorArgs(intermediate)` | binding `tensor::dst` |
| writer_send RTA 0 | intermediate `Buffer*` | `TensorBinding(intermediate → dst)` |
| writer_send `:59` 3rd arg | `payload_size_bytes` | dropped from accessor; named RTA stays |
| writer_send RTA 8 | `semaphore.address()` | GlobalSemaphore binding `global_sem::receive_semaphore` (override 4a) |
| writer_send RTA 9 + tail | `dst_is_forward`, fwd/bwd flags, connection args | fabric binding `fabric::connection` (override 4b) |
| reader_receive CTA 0/1/2 | CB ids | `dfb::packet_header`, `dfb::packet`, `dfb::receiver` |
| reader_receive CTA 3 | `l1_alignment` | named CTA `alignment` |
| reader_receive CTA 4.. | `TensorAccessorArgs(intermediate)` | binding `tensor::packet_buffer` |
| reader_receive RTA 3 | intermediate `Buffer*` | `TensorBinding(intermediate → packet_buffer)` |
| reader_receive `:69` 3rd arg | `packet_size_bytes` | dropped from accessor; named RTA stays |
| reader_receive RTA 7 | `semaphore.address()` | `global_sem::sender_semaphore` (override 4a) |
| reader_receive RTA 9 + tail | `sender_is_forward`, fwd/bwd flags, connection args | `fabric::connection` (override 4b) |
| all other positional RTAs | positional | named RTAs (names = the kernel's local variable names) |

## Applied Patterns

- [Sync-free and single-ended CBs → self-loop DFB] (patterns catalog): send `packet_header`, send `packet`, receive `packet_header`, receive `packet` (DM self-loops; legal on Gen1, Quasar debt).
- 1P+1C plain FIFO: send `sender` (reader P → writer C), receive `receiver` (reader P → writer C), local copy `in0/out` (reader P → writer C).
- [Caution: Porting a shared kernel] (patterns catalog), rung 2: four `_metal2` forks in this op's directory.
- Override 2/3/4: op-owned GlobalSemaphore in `MeshWorkloadArtifacts::global_semaphores`; `GlobalSemaphoreParameter`/binding; `FabricConnectionSpec`/binding.

## Deferred / Flagged

- Fabric connection kernel side: `build_from_binding` + `open_start()` where `build_from_args<…START_ONLY>` ran, `open_finish()` and `close()` at the same places. `has_fabric_connection` is not checked, because every node of the kernel has a target (as the legacy kernel always built a connection).
- Local-copy DFB is one DFB with the reader as PRODUCER and writer as CONSUMER; the two unary forks name it `dfb::in0` (reader) and `dfb::out` (writer), so both bindings use different accessor names on one spec.
