# Port Plan — `experimental/padded_slice`

Port plan for `ttnn/cpp/ttnn/operations/experimental/padded_slice`, ported from the `ProgramDescriptor`
host API (`ProgramDescriptorFactoryConcept`, PR #57550) to Metal 2.0.
Written during the inventory and planning steps; committed alongside the port for review.

Both factories are ported in this pass. Each is a complete sub-port: the RM factory with its two
selectable writer sources, the Tile factory with its three kernels.

## Legacy Inventory

*Observed on the port branch (= `origin/main` @ `65b565e8f8b`, which carries pre-port fix #62499 as
`0226df45c4a`); line numbers are pre-port.*

### Legacy factory shape
- Concept: `ProgramDescriptorFactoryConcept` on both factories (`create_descriptor`), in the
  `program_factory_t` variant at `device/padded_slice_device_operation.hpp:25` — not the direct-descriptor
  shape, so ttnn_factory.md exception 3 does not apply.
- Variants: two factories, one each: `PaddedSliceRMProgramFactory` (row-major input) and
  `PaddedSliceTileProgramFactory` (tiled input). Selected by input layout in
  `select_program_factory` (`device/padded_slice_device_operation.cpp:15-23`).
- `override_runtime_arguments`: RM only (`padded_slice_rm_program_factory.cpp:411-436`, returns `void`).
- Custom `compute_program_hash`: none — default reflection-based hash (no `attribute_values` / `to_hash`).

### Variant: `PaddedSliceRMProgramFactory` (`device/padded_slice_rm_program_factory.cpp`)

#### Kernels
| unique_id | source | core_ranges | CTAs (positional) | CTAs (named) | RTAs | CRTAs | defines | opt_level | config |
|---|---|---|---|---|---|---|---|---|---|
| reader | `device/kernels/dataflow/padded_slice_reader_rm_interleaved_start_id.cpp` | `total_cores` (output shard grid) | `[0] is_non_aligned`, `[1] non_aligned_temp_cb_index`, `[2] num_trids` (=2), then `TensorAccessorArgs(src0_buffer)` tail (`:359-361`) | none | per core (`:168-237`): `[0] input base address`, `[1] input_page_size` (3rd accessor arg), `[2] unpadded_stick_size` (`this_input_row_size_bytes`), `[3] stick_size_offset`, `[4] num_dims`, `[5] start_id`, `[6] num_sticks_per_core`, `[7] num_sticks_per_core_read` (= num_sticks), `[8] num_read_per_barrier` (= num_sticks), `[9] src_offset_bytes`, `[10] misalignment`, then three `num_dims`-long blocks: `num_output_sticks_per_dim`, `num_input_sticks_per_dim`, `id_per_dim` | none | none | O2 (resolved; unset) | `ReaderConfigDescriptor{}` |
| writer (`pad_output_row`) | `device/kernels/dataflow/writer_unary_sharded_padded_rm.cpp` | `total_cores` | `[0] output_cb_index`, `[1] temp_pad_cb_index`, `[2] output.element_size()` (`:357,377-378`) | none | per core (`:251-252`): `[0] num_units` (`this_core_num_sticks`), `[1] num_elements_per_row`, `[2] unpadded_row_size_bytes`, `[3] padded_row_size_bytes` | none | none | O2 (resolved; unset) | `WriterConfigDescriptor{}` |
| writer (`!pad_output_row`) | `data_movement/sharded/device/kernels/dataflow/writer_unary_sharded.cpp` (**borrowed**; `_metal2` fork exists beside it) | `total_cores` | `[0] output_cb_index` | none | same four values as above; the kernel reads only `[0] num_units` | none | none | O2 (resolved; unset) | `WriterConfigDescriptor{}` |

`grep -n opt_level` on the factory: zero hits.

#### CBs
| index | total_size | core_ranges | data_format | page_size | tile (if set) | built at |
|---|---|---|---|---|---|---|
| 0 `output` | `num_output_sticks_per_core * output_cb_page_size` | `total_cores` | `cb_data_format` (input dtype) | `output_cb_page_size` (row size, aligned up when `is_non_aligned`) | — | `:339`, **borrowed memory** (`.buffer = dst_buffer`) |
| 1 `temp_pad` (only `pad_output_row`) | `output_row_size_bytes` | `total_cores` | `cb_data_format` | `output_row_size_bytes` | — | `:343-344` |
| 2 `non_aligned` (index 1 when there is no `temp_pad`; only `is_non_aligned`) | `num_trids * scratch_page_size` | `total_cores` | `cb_data_format` | `scratch_page_size` (`:351-352`) | — | `:353-354` |

No GlobalCircularBuffer, no `address_offset`.

#### Semaphores
none

#### Tensor accessors
| host site (file:line) | originating Tensor | RTA slot (host) |
|---|---|---|
| `TensorAccessorArgs(src0_buffer).append_to(reader_compile_time_args_vec)` (`:361`); kernel `TensorAccessorArgs<3>()` + `TensorAccessor(src_args, src_addr, padded_stick_size)` (reader `:33,45`) | input | reader RTA `[0]` (base, `:226`; refreshed by the override `:434`); 3rd argument from reader RTA `[1]` (`:170`) |

The output tensor is reached only through the borrowed-memory CB 0 (refreshed by the override, `:86-92`, `:424`).

#### Work split
n/a — one kernel instance per core of the output shard grid (`corerange_to_cores(total_cores, …, rm_orientation)`, `:287`); per-core counts come from the shard spec, not `split_work_to_cores`.

#### Override (ported-from `override_runtime_arguments`, `:411-436`)
Writes, in order: (1) guards `TT_FATAL(dst_tensor.is_sharded())`, `TT_FATAL(src_buffer != nullptr)`, `TT_FATAL(dst_buffer != nullptr)` (`:419-423`); (2) `UpdateDynamicCircularBufferAddress` on CB 0 with the output buffer (`:424` → `:86-92`); (3) reader RTA `[0]` = input buffer address on every core (`:432-435`). Nothing else.

### Variant: `PaddedSliceTileProgramFactory` (`device/padded_slice_tile_program_factory.cpp`)

#### Kernels
| unique_id | source | core_ranges | CTAs (positional) | CTAs (named) | RTAs | CRTAs | defines | opt_level | config |
|---|---|---|---|---|---|---|---|---|---|
| compute | `sliding_window/halo/device/kernels/compute/pack_untilize.cpp` (**borrowed**; no `_metal2` fork) | `total_cores` | `[0] cb_input_index`, `[1] cb_untilized_index`, `[2] cb_untilized_index`, `[3] max_num_tiles_per_row`, `[4] 1` (`:490-496`) | none | per core (`:337-339`): `[0] total_blocks` (`num_tiles_this_core / max_num_tiles_per_row`) | none | none | **O3** (resolved; unset on a `ComputeConfigDescriptor`) | `ComputeConfigDescriptor{.fp32_dest_acc_en = false, .dst_full_sync_en = false}` (`:506`) — fidelity HiFi4, `math_approx_mode = false`, `bfp8_pack_precise = false`, `unpack_to_dest_mode` empty (all defaults) |
| reader | `device/kernels/dataflow/padded_slice_reader_tiled_interleaved_start_id.cpp` | `total_cores` | `[0] max_num_tiles_per_row`, then `TensorAccessorArgs(src0_buffer)` tail (`:517-518`) | none | per core (`:165-177`, `:224-226`, `:315-335`): `[0] Buffer*` (input), `[1] num_dims`, `[2] start_id` (`input_start_id + width_offset_start_tile`), `[3] num_tiles_per_core`, `[4] num_tiles_per_barrier` (= `max_num_tiles_per_row`), `[5] num_tiles_per_row_this_core`, then three `num_dims`-long blocks: `num_output_tiles_per_dim` (element 0 overwritten per core with `max_num_tiles_per_row`, `:225`), `num_input_tiles_per_dim` (element 0 overwritten per core, `:226`), `reversed_tile_start_index` (the `id_per_dim` seed) | none | none | O2 (resolved; unset) | `ReaderConfigDescriptor{}` |
| writer | `device/kernels/dataflow/writer_unary_sharded_padded_tiled.cpp` | `total_cores` | `[0] cb_untilized_index`, `[1] cb_output_index`, `[2] cb_padding_index`, `[3] is_non_aligned`, `[4] num_dims` (= rank, 4), `[5] output.element_size()`, `[6] output_row_size_bytes` (`:508-515`) | none | per core (`:353-362`): `[0] total_num_tiles`, `[1] num_tiles_per_read` (= `max_num_tiles_per_row`), `[2] num_sticks_this_core`, `[3] padded_channels_elems`, `[4] misalignment`, then three `num_dims`-long blocks: `reversed_start_index` (`output_coord` seed), `reversed_output_start_in_input`, `reversed_output_end` | none | none | O2 (resolved; unset) | `WriterConfigDescriptor{}` |

`grep -n opt_level` on the factory: zero hits.

#### CBs
| index | total_size | core_ranges | data_format | page_size | tile (if set) | built at |
|---|---|---|---|---|---|---|
| 0 `input` | `cb_buffer_size(4) * max_num_tiles_per_row * input_single_tile_size` | `total_cores` | input format | `input_single_tile_size` | — | `:450-455` |
| 1 `untilized` | `4 * max_num_tiles_per_row * output_single_tile_size` | `total_cores` | output format | `output_single_tile_size` | — | `:457-462` |
| 2 `output` | `num_output_sticks_per_core * output_row_size_bytes` | `total_cores` | output format | `output_row_size_bytes` | — | `:470-476`, **borrowed memory** (`.buffer = dst_buffer`) |
| 3 `padding` | `output_row_size_bytes` | `total_cores` | output format | `output_row_size_bytes` | — | `:478-483` |

No GlobalCircularBuffer, no `address_offset`.

#### Semaphores
none

#### Tensor accessors
| host site (file:line) | originating Tensor | RTA slot (host) |
|---|---|---|
| `TensorAccessorArgs(src0_buffer).append_to(reader_compile_time_args_vec)` (`:518`); kernel `TensorAccessorArgs<1>()` + `TensorAccessor(src_args, src_addr)` (tiled reader `:32-33`) | input | reader RTA `[0]`, delivered as a `Buffer*` (`emplace_padded_slice_tile_reader_args`, `:38-50`) and patched by the descriptor framework on cache hits |

The output tensor is reached only through the borrowed-memory CB 2 (framework-patched).

#### Work split
n/a — one kernel instance per core of the output shard grid (`:408`), per-core tile counts derived from the shard spec and the slice bounds (`:187-365`).

### Shared kernels
| kernel | kind | consumers on `main` (factory bindings; quasar copies and comments excluded) | `_metal2` fork beside it | rung |
|---|---|---|---|---|
| `data_movement/sharded/device/kernels/dataflow/writer_unary_sharded.cpp` | borrowed (RM, `!pad_output_row`) | `data_movement/sharded/interleaved_to_sharded`, `sharded_partial/interleaved_to_sharded_partial`, `tilize` (2 factories), `tilize_with_val_padding`, `transpose` (wh sharded), `untilize` (2 factories), `experimental/transformer/nlp_kv_cache_load_slice`, `reduction/generic` (2 factories), this op | **yes** — `writer_unary_sharded_metal2.cpp`. Vocabulary: `dfb::out` bound CONSUMER; `args::num_units` (RTA); no CTAs, no defines | **1 — reuse** |
| `sliding_window/halo/device/kernels/compute/pack_untilize.cpp` | borrowed (Tile compute) | `sliding_window/halo/device/untilize_with_halo_program_factory.cpp` (passes **two distinct** output CBs, `:185-197`), this op (passes the same CB twice) | **no** | **2 — create** `pack_untilize_metal2.cpp` beside it + pointer comment in the original |
| `device/slice_cb_descriptor.hpp`, `device/padded_slice_utils.hpp` | lent host headers (not kernels) | the three `experimental/slice_write` factories | n/a | leave in place; this port stops including `slice_cb_descriptor.hpp` |

The op's own four kernels are bound by nothing else on `main` (the quasar tree keeps private copies).

### Flags
- Mutable argument memory: the RM reader (`:140`, `:212`), the Tile reader (`:106-114`) and the Tile writer
  (`:316-323`) write back into their vararg blocks (`id_per_dim[j]++`, `output_coord[...] +=`); Metal 2.0
  varargs are read-only.
- `experimental::CB` (= `CircularBuffer`) comes from `pool/device/kernels/experimental_device_api.hpp`,
  which also supplies `experimental::local_addr` used by three kernels; the include stays.
- The RM writer source is selected at runtime (`pad_output_row`, `:376-385`); both sources convert with the
  factory.
- The legacy RM factory sends the pad writer's four RTAs to the plain writer too (`:251-252`, `:394`); the
  plain writer reads only `[0]`.
- `padded_slice_nanobind.cpp` binds only the `ttnn.experimental.padded_slice` wrapper — nothing to delete.
- `padded_slice_rm_program_factory.cpp:403-407` asserts the reader's positional kernel index for the
  override's `GetRuntimeArgs`; the subject disappears with the override's translation.

## TTNN ProgramFactory

- **Concept (inherited from audit)**: `PaddedSliceRMProgramFactory` → **`CustomProgramSpecFactoryConcept`**
  (the ported-from factory has an `override_runtime_arguments`); `PaddedSliceTileProgramFactory` →
  **`ProgramSpecFactoryConcept`**.
- **Custom `compute_program_hash`**: none.
- **Implementation notes**: the RM override is translated, not deleted — its three guards move with it,
  and its two address refreshes (output CB backing buffer, reader base RTA) become the `OUTPUT` and `INPUT`
  `TensorArgument`s of the returned `ProgramRunArgs`; nothing else is refreshed, as before. Both factories
  live in one CMake target with a unity build, so spec-name constants go in per-file op-scoped namespaces
  (`padded_slice_rm_metal2`, `padded_slice_tile_metal2`) rather than anonymous ones.

## Planned Spec Shape

Default 1:1 with legacy.

### Variant: RM
- KernelSpecs: `reader` (op reader) and `writer` — the writer's source, bindings and RTA schema selected by
  `pad_output_row`: the op's pad writer (`dfb::out` CONSUMER, `dfb::temp_pad` self-loop, CTA
  `output_elem_size`, four RTAs) or the existing `writer_unary_sharded_metal2.cpp` fork (`dfb::out`
  CONSUMER, RTA `num_units` only).
- DataflowBufferSpecs: `out` (`borrowed_from = OUTPUT`); `temp_pad` only when `pad_output_row`;
  `non_aligned` only when `is_non_aligned`.
- ScratchpadSpecs: `id_per_dim` (`num_dims * 4` bytes) on the reader — writable home for the per-dimension index counter.
- SemaphoreSpecs: none.
- TensorParameters: `input` (bound by the reader as `tensor::src`), `output` (backs `out`).
- WorkUnitSpecs: one, `{reader, writer}` on `total_cores`.
- Op-owned tensors: none.

### Variant: Tile
- KernelSpecs: `compute` (new fork `pack_untilize_metal2.cpp`), `reader`, `writer`.
- DataflowBufferSpecs: `in`, `untilized`, `out` (`borrowed_from = OUTPUT`), `padding`.
- ScratchpadSpecs: `id_per_dim` on the reader, `output_coord` on the writer (`num_dims * 4` bytes each).
- SemaphoreSpecs: none.
- TensorParameters: `input` (reader, `tensor::src`), `output` (backs `out`).
- WorkUnitSpecs: one, `{compute, reader, writer}` on `total_cores`.
- Op-owned tensors: none.

### DFB endpoint census (re-derived from the kernel bodies)
| DFB | touchers | assignment |
|---|---|---|
| RM `out` | reader `reserve_back`/`get_write_ptr`/`push_back` (locked PRODUCER); writer: fork `wait_front`/`pop_front` (locked CONSUMER) or pad writer `get_read_ptr` + NoC self-reads (role-free) | 1P+1C: reader PRODUCER, writer CONSUMER (both sources). Pad-writer config is fill-only; no pop added |
| RM `temp_pad` | pad writer only (`get_read_ptr`, zero-fill, NoC source) | self-loop on the pad writer |
| RM `non_aligned` | reader only (`reserve_back`/`get_write_ptr`/NoC dst/`push_back`) | self-loop on the reader |
| Tile `in` | reader PRODUCER; compute CONSUMER (`untilize`) | 1:1 |
| Tile `untilized` | compute PRODUCER (under two names `out_cb_id0`/`out_cb_id1`); writer CONSUMER | 1:1; one spec, the compute fork aliases the second name to the first |
| Tile `out` | writer only (NoC self-reads with `.offset_bytes`, no FIFO ops) | self-loop on the writer |
| Tile `padding` | writer only (`get_read_ptr`, zero-fill, NoC source) | self-loop on the writer |

Agrees with the brief. No multi-binding flag anywhere.

## Preserved Multiplicity

none — no work-split multiplicity in legacy (one `KernelDescriptor` per kernel per factory).

## Dropped Plumbing

### RM
| legacy location (file:line) | legacy form | Metal 2.0 replacement |
|---|---|---|
| reader RTA `[0]` (`padded_slice_rm_program_factory.cpp:169,226`; refreshed `:434`; kernel `:12`) | `input_tensor.buffer()->address()` | `TensorBinding{INPUT, "src"}`; `TensorAccessor(tensor::src)`; the override returns `{INPUT, input}` |
| reader RTA `[1]` (`:170`; kernel `:13,45`) | `input_page_size` as the accessor's 3rd argument (Class 2) | dropped — the binding supplies the aligned page size |
| reader CTA tail (`:361`; kernel `:33`) | `TensorAccessorArgs(src0_buffer)` / `TensorAccessorArgs<3>()` | the binding mechanism |
| reader CTA `[0]` (`:360`; kernel `:30`) | `is_non_aligned` (gates the scratch CB) | `compiler_options.defines = {{"NON_ALIGNED", "1"}}` + conditional `dfb::non_aligned` binding; kernel `#ifdef NON_ALIGNED` |
| reader CTA `[1]` (`:360`; kernel `:31`) | `non_aligned_temp_cb_index` | `DFBBinding{NON_ALIGNED, "non_aligned", PRODUCER}` + `CONSUMER` (self-loop), conditional |
| reader CTA `[2]` (`:360`; kernel `:32`) | `num_trids` (positional) | named CTA `num_trids` |
| reader RTAs `[2..10]` (kernel `:14-23`) | positional scalars | named RTAs `unpadded_stick_size`, `stick_size_offset`, `num_dims`, `start_id`, `num_sticks_per_core`, `num_sticks_per_core_read`, `num_read_per_barrier`, `src_offset_bytes`, `misalignment` |
| reader RTAs `[11..11+3*num_dims)` (kernel `:24-28`) | three `num_dims` blocks via `get_arg_addr(11)` | `num_runtime_varargs = 3 * num_dims`, read with `get_vararg`; the `id_per_dim` block seeds `Scratchpad<uint32_t>(scratch::id_per_dim)` |
| kernel `:37` | `constexpr uint32_t cb_id_in0 = 0` (magic index) | `DFBBinding{OUT, "out", PRODUCER}`; `DataflowBuffer dfb_out(dfb::out)` |
| kernel `:100`, `:88` | `get_local_cb_interface(cb).fifo_page_size` | `dfb.get_entry_size()` |
| pad writer CTAs `[0..1]` (`:357,377`; kernel `:12-13`) | `output_cb_index`, `temp_pad_cb_index` | `DFBBinding{OUT, "out", CONSUMER}`, `DFBBinding{TEMP_PAD, "temp_pad", PRODUCER}` + `CONSUMER` |
| pad writer CTA `[2]` (`:378`; kernel `:14`) | `output.element_size()` positional | named CTA `output_elem_size` |
| pad writer RTAs `[0..3]` (`:251-252`; kernel `:7-10`) | positional | named RTAs `num_units`, `num_elements_per_row`, `unpadded_row_size_bytes`, `padded_row_size_bytes` |
| plain writer CTA `[0]` / RTAs `[1..3]` (`:357`, `:394`) | `output_cb_index`; three values the kernel never reads | fork vocabulary: `dfb::out` CONSUMER, RTA `num_units` only |
| override (`:424` → `:86-92`) | `UpdateDynamicCircularBufferAddress(CB 0, output buffer)` | `TensorArgument {OUTPUT, output}` |
| `:400-407` | positional kernel-index assertion for `GetRuntimeArgs` | subject deleted (kernels are addressed by `KernelSpecName`) — reported |

### Tile
| legacy location (file:line) | legacy form | Metal 2.0 replacement |
|---|---|---|
| reader RTA `[0]` (`:36-50`, `:166`, `:542`; kernel `:15,33`) | `Buffer*` run-arg | `TensorBinding{INPUT, "src"}`; `TensorAccessor(tensor::src)` |
| reader CTA tail (`:518`; kernel `:32`) | `TensorAccessorArgs(src0_buffer)` / `TensorAccessorArgs<1>()` | the binding mechanism |
| reader CTA `[0]` (`:517`; kernel `:13`) | `max_num_tiles_per_row` positional | named CTA `num_tiles_per_row` |
| reader RTAs `[1..5]` (kernel `:16-20`) | positional scalars | named RTAs `num_dims`, `start_id`, `num_tiles_per_core`, `num_tiles_per_barrier`, `num_tiles_per_row_this_core` |
| reader RTAs `[6..6+3*num_dims)` (kernel `:21-25`) | three `num_dims` blocks via `get_arg_addr(6)` | `num_runtime_varargs = 3 * num_dims`; `id_per_dim` block seeds `Scratchpad<uint32_t>(scratch::id_per_dim)` |
| kernel `:27` | `constexpr uint32_t cb_id_in0 = 0` | `DFBBinding{IN, "in", PRODUCER}`; `DataflowBuffer dfb_in(dfb::in)` |
| kernel `:31` | `get_tile_size(cb_id_in0)` (runtime `const`) | `dfb_in.get_tile_size()` |
| compute CTAs `[0..2]` (`:491-493`; kernel `:14-16`) | `cb_input_index`, `cb_untilized_index` ×2 | fork vocabulary: `DFBBinding{IN, "src", CONSUMER}`, `DFBBinding{UNTILIZED, "out0", PRODUCER}`; the kernel's `out_cb_id1` aliases `dfb::out0` unless `SEPARATE_OUT1` is defined (not defined here) |
| compute CTAs `[3..4]` (`:494-495`; kernel `:17-18`) | `max_num_tiles_per_row`, `1` positional | named CTAs `tiles_per_row`, `block_size` |
| compute RTA `[0]` (`:337-339`; kernel `:20`) | positional | named RTA `total_blocks` |
| writer CTAs `[0..2]` (`:509-511`; kernel `:14-16`) | `cb_untilized_index`, `cb_output_index`, `cb_padding_index` | `DFBBinding{UNTILIZED, "untilized", CONSUMER}`, `DFBBinding{OUT, "out", PRODUCER}` + `CONSUMER`, `DFBBinding{PADDING, "padding", PRODUCER}` + `CONSUMER` |
| writer CTAs `[3..6]` (`:512-515`; kernel `:17-20`) | positional | named CTAs `is_non_aligned`, `num_dims`, `output_elem_size`, `output_row_size_bytes` |
| writer RTAs `[0..4]` (`:353-358`; kernel `:22-26`) | positional | named RTAs `total_num_tiles`, `num_tiles_per_read`, `num_sticks_this_core`, `padded_channels_elems`, `misalignment` |
| writer RTAs `[5..5+3*num_dims)` (`:359-362`; kernel `:27-29`) | three blocks via `get_arg_addr(5 + k*num_dims)` | `num_runtime_varargs = 3 * num_dims`; the `output_coord` block seeds `Scratchpad<uint32_t>(scratch::output_coord)`; the other two are read with `get_vararg` |
| kernel `:31` | `constexpr uint32_t tile_size = get_tile_size(cb_out_id)` | `get_tile_size(dfb::out)` (token form; legacy `constexpr`) |

### Both factories
| legacy location | legacy form | Metal 2.0 replacement |
|---|---|---|
| `desc.cbs.push_back(make_slice_cb_descriptor(...))` (RM `:339-354`, Tile `:450-483`) | `CBDescriptor` via the lent `slice_cb_descriptor.hpp` | `DataflowBufferSpec`s; the header is no longer included here but stays for `slice_write` |
| `ReaderConfigDescriptor{}` / `WriterConfigDescriptor{}` | | `ttnn::create_reader_datamovement_config()` / `create_writer_datamovement_config()` |
| `ComputeConfigDescriptor{.fp32_dest_acc_en = false, .dst_full_sync_en = false}` (Tile `:506`) | Style B | `ComputeHardwareConfig{.enable_32_bit_dest = false, .double_buffer_dest = true}` (fidelity, SFPU precision, BFP pack mode at their coinciding defaults; no `unpack_modes` entry needed — 32-bit Dest is off) + `compiler_options.opt_level = O3` |

## Applied Patterns

- [Sync-free and single-ended CBs → self-loop DFB](../../../../../../docs/source/tt-metalium/tt_metal/apis/host_apis/metal_2.0/ai/shared/port_patterns.md#pattern-sync-free-and-single-ended-cbs--self-loop-dfb):
  RM `non_aligned` on the reader, RM `temp_pad` on the pad writer, Tile `out` and `padding` on the writer.
- [Two-toucher DFB → assign 1P+1C](../../../../../../docs/source/tt-metalium/tt_metal/apis/host_apis/metal_2.0/ai/shared/port_patterns.md#pattern-two-toucher-dfb--assign-1p1c-dual-instance-work-split):
  RM `out` — reader PRODUCER, writer CONSUMER (cosmetic on the pad-writer path).
- [Conditional / optional resource bindings](../../../../../../docs/source/tt-metalium/tt_metal/apis/host_apis/metal_2.0/ai/shared/port_patterns.md#pattern-conditional--optional-resource-bindings):
  RM `non_aligned` (`NON_ALIGNED` define; the kernel's `if constexpr (is_non_aligned)` gate is promoted to
  `#ifdef`). RM `temp_pad` needs no gate — it is bound only on the pad-writer `KernelSpec`, the only source
  that references it.
- [Same-FIFO aliasing](../../../../../../docs/source/tt-metalium/tt_metal/apis/host_apis/metal_2.0/ai/shared/port_patterns.md#pattern-same-fifo-aliasing-one-dfb-multiple-kernel-side-names),
  path-dependent variant: the compute fork's `out_dfb1` aliases `dfb::out0` unless `SEPARATE_OUT1` is
  defined, in which case it is its own binding `dfb::out1` — the shape halo (two distinct output CBs) will
  need when it ports.
- [Pass DFB handles directly to LLKs and kernel-lib helpers](../../../../../../docs/source/tt-metalium/tt_metal/apis/host_apis/metal_2.0/ai/shared/port_patterns.md#pattern-pass-dfb-handles-directly-to-llks-and-kernel-lib-helpers):
  compute fork (`compute_kernel_hw_startup`, `compute_kernel_lib::untilize*` template arguments).
- [Caution: Porting a shared kernel](../../../../../../docs/source/tt-metalium/tt_metal/apis/host_apis/metal_2.0/ai/shared/port_patterns.md#caution-porting-a-shared-kernel):
  rung 1 for `writer_unary_sharded_metal2.cpp`, rung 2 for `pack_untilize_metal2.cpp`.
- [Caution: Avoid varargs](../../../../../../docs/source/tt-metalium/tt_metal/apis/host_apis/metal_2.0/ai/shared/port_patterns.md#caution-avoid-varargs-unless-absolutely-necessary):
  only the three `num_dims`-long blocks per DM kernel stay varargs (`j < num_dims` loops / data-indexed
  reads); every scalar is named.
- Mutable vararg memory → `Scratchpad` seeded from the vararg block (the `data_movement/slice` RM port's
  shape; not a catalog entry yet — see the report).
- [Unity-build hygiene](../../../../../../docs/source/tt-metalium/tt_metal/apis/host_apis/metal_2.0/ai/shared/port_patterns.md#pattern-unity-build-hygiene-for-anonymous-namespace-symbols):
  per-file op-scoped name namespaces.

## Deferred / Flagged

- The compute fork gets one `#ifdef`-gated optional binding (`SEPARATE_OUT1`) so that halo — whose
  factory feeds the kernel two distinct output CBs — can bind it later without editing the fork. This port
  does not define it.
- The `#ifdef DEBUG` DPRINT blocks print CB indices and the dropped `src_addr` / `padded_stick_size` RTAs;
  those fields are removed from the prints (the values no longer exist kernel-side). Everything else in the
  DPRINT blocks is kept.
- No structural finding beyond the audit's; nothing stops the port.
