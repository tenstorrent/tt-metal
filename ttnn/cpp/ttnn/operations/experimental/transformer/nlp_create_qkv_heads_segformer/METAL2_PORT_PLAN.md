# Port Plan — nlp_create_qkv_heads_segformer

Port plan for `experimental/transformer/nlp_create_qkv_heads_segformer`, ported from the direct `ProgramDescriptor` API to Metal 2.0.
Written during the inventory and planning steps; committed alongside the port for review.

## Legacy Inventory

### Legacy factory shape
- Concept: `ProgramDescriptorFactoryConcept`, in the **direct-descriptor** shape. `create_descriptor` is a static member of `NlpCreateHeadsSegformerDeviceOperation` (`device/nlp_create_qkv_heads_segformer_device_operation.hpp:21-27`; body `device/nlp_create_qkv_heads_segformer_program_factory.cpp:19-155`). There is no `program_factory_t`, which forces `ttnn_factory.md` exception 3 (nest a factory struct).
- Variants: single.
- Custom `compute_program_hash`: none. Default reflection-based hash, with no `attribute_values` / `to_hash`.

### Kernels
| unique_id | source | core_ranges | CTAs (positional) | CTAs (named) | RTAs | CRTAs | defines | opt_level | config |
|---|---|---|---|---|---|---|---|---|---|
| reader | `device/kernels/dataflow/reader_tm_tile_layout_nlp_create_qkv_heads.cpp` | `all_cores` | `[0] q_num_tiles`, then `TensorAccessorArgs(in0_buffer)` | — | per core: `[0] in0_buffer` (Buffer*), `[1] in1_buffer_addr` (=0), `[2] num_blocks_per_core`, `[3] num_blocks_written*per_tensor_tiles`, `[4] 0u` | — | — | O2 (unset, DM) | `ReaderConfigDescriptor{}` |
| writer | `device/kernels/dataflow/writer_tm_tile_layout_nlp_create_qkv_heads.cpp` | `all_cores` | `[0] q_out_h_tiles`, `[1] q_out_w_tiles`, `[2] q_out_HtWt`, `[3] num_q_heads` (q_out_c), then `TensorAccessorArgs(q_buffer)` | — | per core: `[0] q_buffer` (Buffer*), `[1] num_blocks_per_core`, `[2] q_out_h_dim`, `[3] q_out_tensor_tile_id` | — | — | O2 (unset, DM) | `WriterConfigDescriptor{}` |

`grep -n opt_level` on the factory finds no hits.

### CBs
| index | total_size | core_ranges | data_format | page_size | tile (if set) |
|---|---|---|---|---|---|
| 1 | `per_tensor_tiles * 2 * single_tile_size` | `all_cores` | `datatype_to_dataformat_converter(a.dtype())` | `single_tile_size` | not set |

Endpoint census for CB 1: the reader is the only toucher on the producer side (`reserve_back`/`push_back`, reader `:40,44`). The writer is the only toucher on the consumer side (`wait_front`/`pop_front`, writer `:49,59`). Plain 1:1.

### Semaphores
none

### Tensor accessors
| host site (file:line) | originating Tensor | RTA slot (host) |
|---|---|---|
| `program_factory.cpp:74` (`TensorAccessorArgs(in0_buffer)`) / reader `:26,32` | `tensor_args.input_tensor` | reader RTA 0 (`program_factory.cpp:129`) |
| `program_factory.cpp:82` (`TensorAccessorArgs(q_buffer)`) / writer `:27,31` | `std::get<0>(output)` (q) | writer RTA 0 (`program_factory.cpp:143`) |

K and V outputs (`std::get<1>/<2>(output)`) are never touched by the factory or a kernel.

### Work split
- Driver: `split_work_to_cores(compute_with_storage_grid_size, num_blocks)`, where `num_blocks = B*1*S/32`.
- `num_cores`, `all_cores`, `core_group_1` / `num_blocks_per_core_group_1`, `core_group_2` / `num_blocks_per_core_group_2`.
- Per-core RTAs are emitted in column-major node order `{i / num_cores_y, i % num_cores_y}`. The block count is an RTA, not a CTA, so there is no per-group kernel multiplicity.

### Shared kernels
none. Both kernels are this op's private copies. Same-named files under `nlp_create_qkv_heads/` and `nlp_create_qkv_heads_vit/` are different files, and no other op binds these paths (`grep -rl` over `ttnn/cpp/ttnn/operations/` for the segformer kernel paths finds only this factory).

### Flags
- Reader RTA 1 `in1_tensor_addr` (host constant `0`) and RTA 4 `in1_tensor_tile_id` (host constant `0u`) are read by the kernel but never used (reader `:17,20`).
- The device-op returns three outputs, but only Q is written (audit misc anomaly).

## TTNN ProgramFactory

- **Concept (inherited from audit)**: `ProgramSpecFactoryConcept`.
- **Custom `compute_program_hash`**: none.
- **Implementation notes**: this is the exception-3 restructure. Nest `struct NlpCreateQkvHeadsSegformerProgramFactory { static ProgramArtifacts create_program_artifacts(...); }`, add `using program_factory_t = std::variant<NlpCreateQkvHeadsSegformerProgramFactory>;`, and remove the device-op-level `create_descriptor`. The cache-hit comment moves onto the struct. No `program_factory_t` has appeared since the audit (checked).

## Planned Spec Shape

- KernelSpecs: `reader`, `writer` (1:1 with legacy). The reader uses `create_reader_datamovement_config()` and the writer uses `create_writer_datamovement_config()`, the same values as `ReaderConfigDescriptor{}` / `WriterConfigDescriptor{}`. Default compiler options (O2), matching legacy.
- DataflowBufferSpecs: `qv` (`entry_size = single_tile_size`, `num_entries = per_tensor_tiles * 2`, `data_format_metadata = data_format`, no tile metadata). Bindings: reader PRODUCER, writer CONSUMER.
- SemaphoreSpecs: none.
- TensorParameters: `input` (reader binding `input`) and `q` (writer binding `q`). K and V get no parameter, since no kernel touches them.
- WorkUnitSpecs: one, `{reader, writer}` on `all_cores`. Legacy has both kernels on `all_cores` and no per-group kernel, so no split by core group is needed.

## Preserved Multiplicity

none. Legacy has no work-split multiplicity: one reader and one writer descriptor over `all_cores`, with per-group counts carried as RTAs.

## Dropped Plumbing

| legacy location (file:line) | legacy form | Metal 2.0 replacement |
|---|---|---|
| reader RTA 0 (`program_factory.cpp:129`; reader `:16`) | `in0_buffer` (Buffer*) | `TensorBinding(input)`; kernel `TensorAccessor(tensor::input)` |
| reader CTA chain (`program_factory.cpp:74`; reader `:26`) | `TensorAccessorArgs(in0_buffer).append_to` / `TensorAccessorArgs<1>()` | binding mechanism |
| reader RTA 1 (`program_factory.cpp:32,130`; reader `:17`) | dummy `in1_buffer_addr = 0` (address slot for a nonexistent in1 tensor, never used) | dropped (see Deferred / Flagged) |
| writer RTA 0 (`program_factory.cpp:143`; writer `:17`) | `q_buffer` (Buffer*) | `TensorBinding(q)`; kernel `TensorAccessor(tensor::q)` |
| writer CTA chain (`program_factory.cpp:82`; writer `:27`) | `TensorAccessorArgs(q_buffer).append_to` / `TensorAccessorArgs<4>()` | binding mechanism |
| reader/writer hardcoded `cb_id_qv = 1` (reader `:28`, writer `:29`; host `src1_cb_index = 1` `:103`) | magic CB index | `DFBBinding(qv)`; kernel `DataflowBuffer dfb_qv(dfb::qv)` |
| reader `:35`, writer `:34` `get_tile_size(cb_id_qv)` (non-constexpr) | cb-id free function | `dfb_qv.get_tile_size()` (rule 7) |
| reader CTA 0 | positional `q_num_tiles` | named CTA `q_num_tiles` |
| reader RTAs 2–4 | positional | named RTAs `num_blocks`, `in0_tensor_tile_id`, `in1_tensor_tile_id` (value `0`, carried) |
| writer CTAs 0–3 | positional | named CTAs `q_out_h_tiles`, `q_out_w_tiles`, `q_out_HtWt`, `q_out_c` |
| writer RTAs 1–3 | positional | named RTAs `num_blocks`, `q_out_h_dim`, `q_out_tensor_tile_id` |

## Applied Patterns

none beyond the base translation. Plain 1:1 DFB, Case 1 tensor bindings, and the `AddRuntimeArgsForNode` bridge for the node-first RTA loop.

## Deferred / Flagged

- **`in1_tensor_addr` drop vs. the brief.** The brief says to carry both unused reader RTAs as named args with value `0`. I carry `in1_tensor_tile_id`, a tile index. I drop `in1_tensor_addr`, because it is an address slot (rule 5) and has no consumer at all in this kernel. This is the same disposition as the sibling `nlp_create_qkv_heads_vit` port (a5a563abb3a). There is zero functional change, since no code reads the value.
