# Port Plan — `ttnn/cpp/ttnn/operations/data_movement/pad`

Port plan for `data_movement/pad`, ported from `ProgramDescriptor` / `WorkloadDescriptor` to Metal 2.0.
Written during the inventory and planning steps; committed alongside the port for review.

**Scope: all seven factories**, converted in one change. Every kernel source in the op directory is
bound only by pad's own factories (census below), so there is no *lent* kernel; the single *borrowed*
kernel already has a `_metal2` fork beside it (rung 1 — reuse).

**Invoker decisions taken before construction** (both raised by the audit / brief):

1. **`PadRmReaderWriterMultiCoreProgramFactory` is ported, not deleted.** It is unreachable from
   `select_program_factory` (`pad_device_operation.cpp:99-101`), so **its port is build-verified
   only** — no test can exercise it. Recorded again in the report.
2. **`PadRmShardedHeightOnlyProgramFactory` lands on the base `ProgramSpecFactoryConcept`**, dropping
   its `override_runtime_arguments`. Rationale in [TTNN ProgramFactory](#ttnn-programfactory).

---

## Legacy Inventory

### Legacy factory shape

| Factory | Concept | Entry point | Op-owned tensors |
|---|---|---|---|
| `PadRmReaderWriterProgramFactory` | `MeshWorkloadFactoryConcept` (`WorkloadDescriptor`) | `create_workload_descriptor` @ `pad_rm_reader_writer_program_factory.cpp:185` | yes — pad-value const tensor @ `:200` |
| `PadRmReaderWriterMultiCoreProgramFactory` | `MeshWorkloadFactoryConcept` (`WorkloadDescriptor`) | `create_workload_descriptor` @ `pad_rm_reader_writer_multi_core_program_factory.cpp:404` | yes — pad-value const tensor @ `:419` |
| `PadRmReaderWriterMultiCoreDefaultProgramFactory` | `ProgramDescriptorFactoryConcept` | `create_descriptor` @ `pad_rm_reader_writer_multi_core_default_program_factory.cpp:32` | none |
| `PadRmShardedHeightOnlyProgramFactory` | `ProgramDescriptorFactoryConcept` + `override_runtime_arguments` | `create_descriptor` @ `pad_rm_sharded_height_only_program_factory.cpp:195`; override @ `:412` | none |
| `PadRmShardedWidthOnlyProgramFactory` | `ProgramDescriptorFactoryConcept` | `create_descriptor` @ `pad_rm_sharded_width_only_program_factory.cpp:20` | none |
| `PadTileCoreProgramFactory` | `ProgramDescriptorFactoryConcept` | `create_descriptor` @ `pad_tile_program_factory.cpp:18` | none |
| `PadTileMulticoreProgramFactory` | `ProgramDescriptorFactoryConcept` | `create_descriptor` @ `pad_tile_multicore_program_factory.cpp:31` | none |

- **Variants**: seven factories in one `program_factory_t` variant (`pad_device_operation.hpp:32-39`).
  All factory methods live in factory structs — **no direct-descriptor shape**, so
  `ttnn_factory.md` exception 3 does not apply.
- **Custom `compute_program_hash`**: **none** — no `compute_program_hash`, no backdoor
  `attribute_values` / `to_hash` anywhere in the op. Default reflection hash applies. Nothing to
  leave alone.
- **`opt_level`**: `grep -n opt_level` over the op directory returns **zero hits in code** — no
  `KernelDescriptor::opt_level` is set anywhere. Every kernel is a reader/writer DM kernel
  (`ReaderConfigDescriptor` / `WriterConfigDescriptor`), so the resolved legacy level is `O2` on all
  22 kernel descriptors, which is exactly Metal 2.0's `CompilerOptions` default. **No `opt_level`
  line is required on any `KernelSpec`.** (The op owns no compute kernels, so the `O3` rule never
  fires.)
- **Semaphores**: **none** — the op declares no semaphores of any kind.

### Kernels

#### `PadRmReaderWriterProgramFactory` (single core) and `PadRmReaderWriterMultiCoreProgramFactory`

Identical kernel pair and identical 27-slot RTA layout; the multicore version differs only in the
per-core work split. `writer_rt_args = reader_rt_args` (`:172` / `:385`), so each kernel is handed
roughly half the slots it never reads.

| unique_id | source | core_ranges | CTAs (positional) | CTAs (named) | RTAs | CRTAs | defines | opt_level | config |
|---|---|---|---|---|---|---|---|---|---|
| reader | `device/kernels/dataflow/reader_pad_dims_rm_interleaved.cpp` | sc: `CoreRange({0,0},{0,0})`; mc: `all_cores` from `split_across_cores` | `{unpadded_row_size_nbytes, padded_row_size_nbytes}` + `TensorAccessorArgs(src)` + `(dst)` + `(pad_value_const)` | none | 27 slots (see below) | none | none | `O2` (unset) | `ReaderConfigDescriptor{}` |
| writer | `device/kernels/dataflow/writer_pad_dims_rm_interleaved.cpp` | same | `= reader_ct_args` (a verbatim copy) | none | same 27 slots | none | none | `O2` (unset) | `WriterConfigDescriptor{}` |

RTA slot map (host emission order, `:141-169` / `:346-374`) and per-kernel liveness:

| slot | value | reader | writer |
|---|---|---|---|
| 0 | `src0_buffer` (`Buffer*`) | **addr** | — |
| 1 | `dst_buffer` (`Buffer*`) | — | **addr** |
| 2 | `a.padded_shape()[0]` → `num_unpadded_W` | live | — |
| 3 | `output_shape[0]` → `num_total_W` | **dead** (declared, unused) | **dead** |
| 4 | `a.padded_shape()[1]` → `num_unpadded_Z` | live | — |
| 5 | `output_shape[1]` → `num_total_Z` | live | live |
| 6 | `a.padded_shape()[2]` → `num_unpadded_Y` | **dead** (declared, unused) | — |
| 7 | `output_shape[2]` → `num_total_Y` | **dead** | **dead** |
| 8 | `a.padded_shape()[3]` | — | — |
| 9 | `output_shape[3]` → `num_total_X` | — | **dead** |
| 10 | `unpadded_row_size_nbytes` → `unpadded_X_nbytes` | live | — |
| 11 | `padded_row_size_nbytes` → `padded_X_nbytes` | live | live |
| 12 | `padded_row_diff_size_nbytes` | live | — |
| 13 | `pad_value_const_buffer` (`Buffer*`) | **addr** | — |
| 14 | `pad_value_const_buffer_nbytes` | **never read** (kernel hardcodes 64, `reader…:52`) | — |
| 15 | `packed_pad_value` | live | — |
| 16 | `start_src_stick_id` | live | — |
| 17 | `start_dst_stick_id` | — | live |
| 18 | `start_src_stick_wi` | **dead** | — |
| 19 | `start_dst_stick_wi` | — | **dead** (assigned to `dst_stick_wi`, never advanced) |
| 20 | `start_src_stick_offset` | live | — |
| 21 | `num_local_Y` | live | live |
| 22 | `num_local_unpadded_Y` | live | **dead** |
| 23 | `full_unpadded_X_nbytes` | **dead** | — |
| 24 | `full_padded_X_nbytes` | — | **dead** |
| 25 | `dst_stick_offset` | — | live |
| 26 | `num_local_W` | live | live |

> The brief lists the reader's dead set as slots 3, 7, 18, 23. The census adds **slot 6
> (`num_unpadded_Y`)** — declared at `reader_pad_dims_rm_interleaved.cpp:46`, never referenced
> (the loop at `:87` tests `num_local_unpadded_Y`, slot 22). Recorded in the port report.

Both kernels' **positional CTAs 0 and 1 are never read** — the kernels contain no
`get_compile_time_arg_val` call at all. Slots 0/1 exist only to offset the `TensorAccessorArgs<2>()`
chain, and the writer's copy of the pad-value tensor's accessor args (`:85`) is never instantiated.

#### `PadRmReaderWriterMultiCoreDefaultProgramFactory`

| unique_id | source | core_ranges | CTAs (positional) | RTAs | opt_level | config |
|---|---|---|---|---|---|---|
| reader | `…/reader_pad_dims_rm_interleaved_v2.cpp` | `all_cores` (`split_work_to_cores` over `NCH_padded`) | 22 values @ `:141-163` + `TensorAccessorArgs(src)` @ `:164` | `src_buffer`, `num_sticks_per_core`, `num_sticks_per_barrier`, `curr_sticks_read*num_input_pages_in_row`, `front_pad_n/c/h`, then `num_dims` × `start_dim_offset` | `O2` (unset) | `ReaderConfigDescriptor{}` |
| writer | `…/writer_pad_dims_rm_interleaved_v2.cpp` | `all_cores` | 5 values @ `:166-171` + `TensorAccessorArgs(dst)` @ `:172` | `dst_buffer`, `num_sticks_per_core`, `num_sticks_per_barrier`, `curr_sticks_write*num_output_pages_in_row` | `O2` (unset) | `WriterConfigDescriptor{}` |

Reader CTA slot map, with kernel liveness:

| slot | host value | kernel name | live |
|---|---|---|---|
| 0-2 | `N+front_pad[-4]`, `H+front_pad[-2]`, `C+front_pad[-3]` | `N`,`H`,`C` | yes |
| 3 | `stick_size` | `stick_size_bytes` | yes |
| 4-6 | `N_padded`,`H_padded`,`C_padded` | same | yes |
| 7 | `stick_size_padded` | same | yes |
| 8 | `stick_size_padded_front` | `stick_size_padded_front` **and** `front_padding` (two names, one slot) | yes |
| 9 | `stick_size_padded_end` | declared `:70` | **dead** |
| 10 | `div_up(stick_size_padded,512)` | `num_zero_pad_sticks_read` `:71` | **dead** |
| 11 | `stick_size_padded % 512 …` | `last_zero_stick_size` `:72` | **dead** |
| 12 | `not_pad_by_zero` | same | yes |
| 13 | `packed_pad_value` | read via `kernel_compile_time_args[13]` `:85` | yes |
| 14 | `row_major_min_bytes` | *not declared in kernel* | **dead** |
| 15 | `stick_size_padded_front / row_major_min_bytes` | *not declared* | **dead** |
| 16 | `stick_size_padded_end / row_major_min_bytes` | *not declared* | **dead** |
| 17 | `stick_size_padded / row_major_min_bytes` | *not declared* | **dead** |
| 18 | `stick_size_padded_aligned` | same | yes |
| 19 | `unaligned` | same | yes |
| 20 | `num_input_pages_in_row` | same | yes |
| 21 | `input_accessor_page_size` | `accessor_page_size` | **TensorAccessor 3rd arg — dropped** |
| 22.. | `TensorAccessorArgs(src)` | `TensorAccessorArgs<22>()` | dropped |

Writer CTA slot map: 0 `src0_cb_index` (→ DFB binding), 1 `stick_size_padded` (kernel
`stick_size_bytes`), 2 `stick_size_padded_aligned`, 3 `num_output_pages_in_row`,
4 `output_accessor_page_size` (**3rd arg — dropped**), 5.. `TensorAccessorArgs(dst)` (dropped).

#### `PadRmShardedHeightOnlyProgramFactory`

| unique_id | source | core_ranges | CTAs (positional) | RTAs | opt_level | config |
|---|---|---|---|---|---|---|
| reader | `…/reader_pad_dims_rm_sharded.cpp` | `all_cores_padded` (output shard grid) | `{stick_size_padded, shard_height_padded}` @ `:335` | variable-length block, per core (`:158-186`) | `O2` (unset) | `ReaderConfigDescriptor{}` |
| writer | `…/writer_pad_dims_rm_sharded.cpp` | `all_cores_padded` | 13 values @ `:337-350` | `{num_sticks_per_core_padded, curr_sticks_read, front_pad[-4], front_pad[-3], front_pad[-2]}` + `num_dims` × `start_dim_offset` (`:77-84`) | `O2` (unset) | `WriterConfigDescriptor{}` |

Reader RTA block layout (`get_pad_runtime_args_rm_sharded`, `:158-186`): `num_cores`, then
`2·num_cores` interleaved NoC (x,y), then `num_cores` chunk counts, then `2·Σchunks`
`(chunk_start_id, chunk_length)` pairs. The kernel reaches all but the first through
`get_arg_addr()` at **runtime-computed** offsets (`:20-22`).

Writer CTA slot map: 0-2 `N+front_pad[-4]`/`H+front_pad[-2]`/`C+front_pad[-3]`, 3 `stick_size_padded`,
4-6 `N_padded`/`H_padded`/`C_padded`, 7 `num_zero_pad_sticks_read`, 8 `zero_pad_stick_size`,
9 `not_pad_by_zero`, 10 `packed_pad_value`, 11 `row_major_min_bytes`,
12 `stick_size_padded/row_major_min_bytes`. Slots 10-12 are read via `kernel_compile_time_args[10..12]`
inside `if constexpr (not_pad_by_zero)` at **constant** indices; the host emits them
unconditionally — a fixed named set, **not** a CTA vararg block. All 13 are live.

#### `PadRmShardedWidthOnlyProgramFactory`

| unique_id | source | core_ranges | CTAs (positional) | RTAs | opt_level | config |
|---|---|---|---|---|---|---|
| reader | `…/reader_pad_dims_rm_sharded_stickwise.cpp` | `all_cores_padded` | 9 values @ `:134-143` | **empty list** per core (`:175`) | `O2` (unset) | `ReaderConfigDescriptor{}` |
| writer | `…/writer_pad_dims_rm_sharded_stickwise.cpp` | `all_cores_padded` | 7 values @ `:145-152` | **empty list** per core (`:176`) | `O2` (unset) | `WriterConfigDescriptor{}` |

Reader CTAs: 0 `unpadded_stick_bytes`, 1 `padded_stick_bytes` (**dead**), 2 `unpadded_shard_height`,
3 `padded_shard_height` (**dead**), 4 `W_padding_front_bytes`, 5 `input_shard_cb_index` (→ DFB
binding), 6 `output_shard_cb_index` (→ DFB binding), 7 `unpadded_stick_step`, 8 `padded_stick_step`.
Writer CTAs: 0 `padded_stick_bytes`, 1 `shard_height_padded`, 2 `padding_value_as_u32`,
3 `output.element_size()`, 4 `output_shard_cb_index` (→ binding), 5 `pad_val_cb_index` (→ binding),
6 `padded_stick_step` (**never declared in the kernel — dead**).

#### `PadTileCoreProgramFactory`

| unique_id | source | core_ranges | CTAs (positional) | RTAs | opt_level | config |
|---|---|---|---|---|---|---|
| reader | **borrowed** `ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/reader_unary_interleaved_start_id.cpp` (`:104-105`) | `{{0,0},{0,0}}` | `TensorAccessorArgs(src)` only | `{src0_buffer, num_unpadded_tiles, 0}` | `O2` (unset) | `ReaderConfigDescriptor{}` |
| writer | `…/writer_unary_pad_dims_interleaved.cpp` | `{{0,0},{0,0}}` | `{src0_cb_index, src1_cb_index}` + `TensorAccessorArgs(dst)` | `{dst_buffer, num_unpadded_W, num_padded_Wt, num_unpadded_Z, num_padded_Zt, num_unpadded_Yt, num_padded_Yt, num_unpadded_Xt, num_padded_Xt, packed_pad_value}` | `O2` (unset) | `WriterConfigDescriptor{}` |

Writer reads `get_tile_size(cb_id_out0)` at `:28` — a **non-`constexpr`** `const uint32_t`, so it
becomes the DFB **member getter** `dfb_out0.get_tile_size()` (whitelist rule 7; the `constexpr`
carve-out does not apply).

#### `PadTileMulticoreProgramFactory`

| unique_id | source | core_ranges | CTAs (positional) | RTAs | opt_level | config |
|---|---|---|---|---|---|---|
| reader | `…/reader_pad_tiled.cpp` | `all_cores` (`split_work_to_cores` over `num_pages`) | `{input_cb_index, page_size, output_padded_shape.rank()}` + `TensorAccessorArgs(input)` | `input_buffer`, `num_pages_per_core`, `input_page_offset`, then 4 × `num_dims` blocks | `O2` (unset) | `ReaderConfigDescriptor{}` |
| writer | `…/writer_pad_tiled.cpp` | `all_cores` | `{input_cb_index, output_cb_index, pad_val_cb_index, page_size, rank, packed_pad_value, element_size}` + `TensorAccessorArgs(output)` | `output_buffer`, `num_pages_per_core`, `output_page_offset`, then the same 4 × `num_dims` blocks | `O2` (unset) | `WriterConfigDescriptor{}` |

Writer CTA 1 (`output_cb_index`) is unpacked at `writer_pad_tiled.cpp:23` and **never referenced**
— the dead CTA that accompanies the dead `c_1` CB.

The 4 × `num_dims` RTA block is `input_page_shape`, `output_page_shape`, `input_id_per_dim`,
`output_id_per_dim`, reached via `get_arg_addr(rt_ind)` plus `+ num_dims` pointer strides
(`reader:22-25`, `writer:35-38`) and consumed in `for (d < num_dims)` loops. **`input_id_per_dim` and
`output_id_per_dim` are *written back* in place** by `advance_tensor_index`
(`device/kernels/dataflow/common.hpp:12-19`) — the kernel uses the RTA buffer as mutable scratch.

### CBs

| factory | index | total_size | core_ranges | data_format | page_size | tile |
|---|---|---|---|---|---|---|
| RM sc | `c_0` | `16 * cb_pagesize` | `{{0,0},{0,0}}` | input dtype | `round_up(padded_row_size_nbytes, max(src_align, TILE_WIDTH))` | unset |
| RM mc | `c_0` | `16 * cb_pagesize` | `all_cores` | input dtype | `ceil(dst_nbytes_per_core_w / align) * align` | unset |
| RM default | `c_0` | `16 * cb_npages * stick_size_padded_aligned` | `all_cores` | input dtype | `stick_size_padded_aligned` | unset |
| RM default | `c_1` | `stick_size_padded_DRAM_aligned` | `all_cores` | input dtype | same | unset |
| RM default | `c_2` | `stick_size_padded_DRAM_aligned` | `all_cores` | input dtype | same | unset — **allocated only if `stick_size_padded_front != 0 \|\| unaligned`** (`:127-139`) |
| sharded H | `c_0` | `shard_height_unpadded * stick_size_unpadded` | `total_cores` (full grid) | input dtype | `stick_size_unpadded` | unset — **`buffer = src_buffer`** (`:294`) |
| sharded H | `c_16` | `shard_height_padded * stick_size_padded` | `total_cores` | output dtype | `stick_size_padded` | unset — **`buffer = dst_buffer`** (`:309`) |
| sharded H | `c_1` | `stick_size_padded` | `total_cores` | input dtype | `stick_size_padded` | unset |
| sharded W | `c_0` | `shard_height_unpadded * unpadded_stick_bytes` | `total_cores` | input dtype | `unpadded_stick_bytes` | unset — **`buffer = input_buffer`** (`:75`) |
| sharded W | `c_16` | `shard_height_padded * padded_stick_bytes` | `total_cores` | output dtype | `padded_stick_bytes` | unset — **`buffer = output_buffer`** (`:91`) |
| sharded W | `c_1` | `padded_stick_bytes` | `total_cores` | input dtype | `padded_stick_bytes` | unset |
| tile sc | `c_0` | `2 * single_tile_size` | `{{0,0},{0,0}}` | input dtype | `single_tile_size` | unset |
| tile sc | `c_1` | `1 * single_tile_size` | `{{0,0},{0,0}}` | input dtype | `single_tile_size` | unset |
| tile mc | `c_0` | `2 * page_size` | `all_cores` | input dtype | `page_size` | unset |
| tile mc | `c_1` | `2 * page_size` | `all_cores` | input dtype | `page_size` | unset — **DEAD, zero touchers** |
| tile mc | `c_2` | `page_size` | `all_cores` | input dtype | `page_size` | unset |

- **No `GlobalCircularBuffer` anywhere** — the audit's feature scan is confirmed by
  `grep -rn 'global_circular_buffer\|remote_cb\|GlobalCircularBuffer'` returning nothing.
- **No aliased CBs** — every `CBDescriptor` has exactly one `format_descriptors` element.
- **No `tile` field is set on any format descriptor**, so `tile_format_metadata` stays `nullopt`
  everywhere (JIT fallback is observably identical for standard 32×32).
- **Note — the two sharded factories give their CBs `core_ranges = total_cores` (the whole compute
  grid) while their kernels run on `all_cores_padded` only.** Metal 2.0 *derives* DFB placement from
  kernel bindings, so the ported DFBs land on `all_cores_padded`. Flagged below.

### Semaphores

none — the op declares no semaphores. `grep -rn '[Ss]emaphore'` over the op directory: zero hits.

### Tensor accessors

| host site (file:line) | originating Tensor | RTA slot (host) | kernel construction |
|---|---|---|---|
| `pad_rm_reader_writer_program_factory.cpp:82` / `:143` | input | reader/writer slot 0 | `reader_pad_dims_rm_interleaved.cpp:75` |
| `…:83` / `:144` | output | slot 1 | `writer_pad_dims_rm_interleaved.cpp:32` |
| `…:84` / `:156` | pad-value const (**op-owned**) | slot 13 | `reader_pad_dims_rm_interleaved.cpp:77` |
| `…_multi_core_program_factory.cpp:246` / `:348` | input | slot 0 | same reader |
| `…:247` / `:349` | output | slot 1 | same writer |
| `…:248` / `:361` | pad-value const (**op-owned**) | slot 13 | same reader |
| `…_default_program_factory.cpp:164` / `:221` | input | reader slot 0 | `reader_pad_dims_rm_interleaved_v2.cpp:95` (**3-arg**) |
| `…:172` / `:222` | output | writer slot 0 | `writer_pad_dims_rm_interleaved_v2.cpp:25` (**3-arg**) |
| `pad_tile_program_factory.cpp:98` / `:122` | input | reader slot 0 | donor `reader_unary_interleaved_start_id.cpp:30` |
| `…:100` / `:126` | output | writer slot 0 | `writer_unary_pad_dims_interleaved.cpp:30` |
| `pad_tile_multicore_program_factory.cpp:121` / `:222` | input | reader slot 0 | `reader_pad_tiled.cpp:29` |
| `…:132` / `:223` | output | writer slot 0 | `writer_pad_tiled.cpp:42` |

The two sharded factories construct **no `TensorAccessor` at all** — their kernels reach input and
output purely through the borrowed CBs. Their two tensor parameters exist solely as
`borrowed_from` sources.

### Work split

| factory | driver | groups |
|---|---|---|
| RM sc | n/a — single core `{0,0}` | — |
| RM mc | `split_across_cores(grid, nbatch, ntiles_h, ntiles_w)` @ `:217` — hardcoded resnet table | one `all_cores` range; per-core args vary but there is a **single** `KernelDescriptor` per role |
| RM default | `split_work_to_cores(grid_or_sub_core_grids, NCH_padded)` @ `:90-91` | `num_cores`, `all_cores`, `core_group_1`/`core_group_2`, `num_sticks_padded_per_core_group_1/2` — **groups differ only in a per-core RTA**, not in CTAs |
| sharded H | output shard grid `all_cores_padded`, `num_cores_padded` | single group |
| sharded W | `get_optimal_worker_cores_for_sharded_tensor(output)` | single group |
| tile sc | n/a — single core `{0,0}` | — |
| tile mc | `split_work_to_cores(grid_or_sub_core_grids, num_pages)` @ `:45-47` | `core_group_1`/`core_group_2` — again only a per-core **RTA** differs |

**No factory creates more than one `KernelDescriptor` per role.** The per-group counts travel as
runtime args in legacy already, so there is no work-split CTA multiplicity to preserve.

### Shared kernels

Census run as `grep -rl <filename> ttnn/cpp/ttnn/operations/ tests/`, quasar copies discarded.

| kernel | class | consumers | fork? | rung |
|---|---|---|---|---|
| `eltwise/unary/device/kernels/dataflow/reader_unary_interleaved_start_id.cpp` | **borrowed** | `data_movement/pad` (this factory), `data_movement/untilize_with_unpadding` ×2, `examples/example` ×2, `examples/example_multiple_return`, `experimental/transformer/nlp_create_qkv_heads_falcon7b`, `reduction/topk`, plus `tests/.../test_generic_op.{cpp,py}` | **yes** — `reader_unary_interleaved_start_id_metal2.cpp`, same directory | **rung 1 — reuse.** Bind the fork; create nothing; touch neither the original nor the fork. |
| `reader_pad_dims_rm_interleaved.cpp`, `writer_pad_dims_rm_interleaved.cpp` | **intra-op**, 2 consumers | `PadRmReaderWriterProgramFactory`, `PadRmReaderWriterMultiCoreProgramFactory` | n/a | **both consumers convert in this change** → convert in place, no fork. |
| every other kernel under `device/kernels/dataflow/` | private | exactly one pad factory each | n/a | convert in place |
| `device/kernels/dataflow/common.hpp` | private | `reader_pad_tiled.cpp`, `writer_pad_tiled.cpp` (relative include, same directory) | n/a | in the port unit; both consumers convert together |

**Fork vocabulary (rung-1 constraint).** `reader_unary_interleaved_start_id_metal2.cpp` declares its
interface in its header comment: `dfb::in`, `tensor::src`, `args::num_pages`, `args::start_id`, and
it already replaces `get_local_cb_interface(cb).fifo_page_size` with `dfb.get_entry_size()`. It gates
nothing but `BACKWARDS`, which pad does not define. **The factory conforms to those names — note
`tensor::src`, not `tensor::input`.** The differently-named fork at
`copy/typecast/device/kernels/dataflow/reader_unary_interleaved_start_id_metal2.cpp` (vocabulary
`tensor::input`) is an op-local copy, **not** the sibling of the original — do not bind it.

### Flags

- **No unreferenced kernel files.** All 12 kernel sources in the op directory are bound.
- `PadRmReaderWriterMultiCoreProgramFactory` is **unreachable** from `select_program_factory`
  (`pad_device_operation.cpp:99-101`). Ported by invoker decision; build-verified only.
- `split_across_cores`'s `default:` branch throws at
  `pad_rm_reader_writer_multi_core_program_factory.cpp:97` and is followed by ~30 lines of
  unreachable code (`:99-131`). **Left untouched** — out of scope.
- `reader_pad_dims_rm_interleaved.cpp:52` hardcodes `pad_value_const_buffer_nbytes = 64` (issue
  #21978) while the host computes and passes the real value at slot 14. **The hardcode stays**; the
  arg simply gets no name.
- `pad_tile_multicore_program_factory.cpp:194-197` divides *both* trailing dims by `TILE_HEIGHT`
  (never `TILE_WIDTH`). Benign while both are 32. **Not touched**; reported.

---

## TTNN ProgramFactory

- **Concept (inherited from audit)**: `ProgramSpecFactoryConcept` for **all seven** factories.
  - Six of them carry it directly from the brief.
  - The seventh, `PadRmShardedHeightOnlyProgramFactory`, was routed to
    `CustomProgramSpecFactoryConcept` by the audit, which simultaneously raised an open question
    about whether that was necessary. **The invoker was asked and chose the base concept**, so the
    `override_runtime_arguments` at `pad_rm_sharded_height_only_program_factory.cpp:412-424` (and its
    declaration at `.hpp:22-27`) is removed rather than translated.
  - **Justification, verified against the code.** The override's entire body re-points two CB base
    addresses — the input CB and the output CB — and nothing else (`:420-423`). Both of those CBs are
    `borrowed_from` DFBs in the port, and Metal 2.0 refreshes a borrowed DFB's backing L1 address from
    its `TensorArgument` on every dispatch; on the base concept the framework refreshes every
    `TensorArgument` on a cache hit. So the refreshed set is *identical*, statement for statement,
    and the override has no non-tensor work to lose. `PadRmShardedWidthOnlyProgramFactory` already
    carries the same two borrowed CBs with no override at all, which is independent evidence.
  - Two `MeshWorkloadFactoryConcept` factories collapse to the single-program concept: each builds
    **one** `ProgramDescriptor` above the loop and pushes the *same* object once per range in
    `tensor_coords` (`:202-211` / `:421-429`). Nothing is per-mesh-coordinate; the
    `WorkloadDescriptor` wrapper existed only to carry op-owned tensors, which
    `ProgramArtifacts::op_owned_tensors` carries natively.
- **Custom `compute_program_hash`**: **none.** Nothing to leave intact.
- **Pybind**: `pad_nanobind.cpp` binds only the two public `ttnn::pad` overloads — **no**
  `create_descriptor` / `create_workload_descriptor` exposure, so no pybind line disappears and
  the port removes no user-visible API.
- **Implementation notes**:
  - The op directory builds under a **unity build** (`ttnn_op_data_movement` uses
    `Unity/unity_*_cxx.cxx`), and seven factory `.cpp`s land in one translation unit. Every
    anonymous-namespace spec-name constant is therefore **prefixed per factory** (`RM_SC_`, `RM_MC_`,
    `RM_DEF_`, `SH_H_`, `SH_W_`, `TILE_SC_`, `TILE_MC_`) per
    [Unity-build hygiene](../../../../../../docs/source/tt-metalium/tt_metal/apis/host_apis/metal_2.0/ai/shared/port_patterns.md).
  - Each factory's `.hpp` swaps `create_descriptor` / `create_workload_descriptor` for
    `create_program_artifacts`; nothing else in the headers changes except the include of
    `ttnn/metal_v2_artifacts.hpp` and, for the two workload factories, dropping
    `workload_descriptor.hpp`.

---

## Planned Spec Shape

Naming convention: DFB spec names describe the buffer's role; accessor names are the kernel-side
`dfb::` handles. **No name anywhere contains `cb`.**

### `PadRmReaderWriterProgramFactory` (`RM_SC_`) and `PadRmReaderWriterMultiCoreProgramFactory` (`RM_MC_`)

Identical shape; the two differ only in node set and per-node RTA values.

- **KernelSpecs**: `reader` (`reader_pad_dims_rm_interleaved.cpp`), `writer`
  (`writer_pad_dims_rm_interleaved.cpp`).
- **DataflowBufferSpecs**: `in0` ← legacy `c_0`. `entry_size = cb_pagesize`, `num_entries = 16`,
  `data_format_metadata = in_df`. reader PRODUCER (`dfb::in0`), writer CONSUMER (`dfb::in0`).
- **SemaphoreSpecs**: none.
- **TensorParameters**: `input` (reader binds as `tensor::src`), `output` (writer binds as
  `tensor::dst`), `pad_value_const` (**op-owned**; reader binds as `tensor::pad_value`).
- **WorkUnitSpecs**: one — `{reader, writer}` over `{0,0}` (sc) / `all_cores` (mc).
- **Op-owned tensors**: one — the 32-element bfloat16 pad-value const tensor. Construction
  (`build_pad_value_const_tensor_sc` / `_mc`) is kept **verbatim**; only the tail changes to
  `release_mesh_tensor()` into `ProgramArtifacts::op_owned_tensors`.

### `PadRmReaderWriterMultiCoreDefaultProgramFactory` (`RM_DEF_`)

- **KernelSpecs**: `reader`, `writer` (the `_v2` pair).
- **DataflowBufferSpecs**:
  - `in0` ← `c_0`. reader PRODUCER, writer CONSUMER (plain 1:1).
  - `pad` ← `c_1`. reader-only → **self-loop** (reader PRODUCER *and* CONSUMER, one accessor name).
  - `pad_align` ← `c_2`. reader-only → **self-loop**, and **conditional** on
    `stick_size_padded_front != 0 || unaligned`.
- **TensorParameters**: `input` (reader, `tensor::src`), `output` (writer, `tensor::dst`).
- **WorkUnitSpecs**: one — `{reader, writer}` over `all_cores`.
- Both core groups share one `KernelSpec` per role, exactly as legacy: the group difference is a
  per-node RTA (`num_sticks_per_core`), never a CTA.

### `PadRmShardedHeightOnlyProgramFactory` (`SH_H_`)

- **KernelSpecs**: `reader` (`reader_pad_dims_rm_sharded.cpp`), `writer`
  (`writer_pad_dims_rm_sharded.cpp`).
- **DataflowBufferSpecs**:
  - `in_shard` ← `c_0`, `borrowed_from = input`. Reader raw-peeks only → **self-loop**.
  - `out_shard` ← `c_16`, `borrowed_from = output`. **Two touchers** — reader is a locked producer
    (`reserve_back`/`push_back` @ `:31,69`), writer raw-peeks `get_write_ptr()` @ `:90` with no FIFO
    ops → **1P+1C**: reader PRODUCER, writer CONSUMER. **Not** multi-binding.
  - `pad` ← `c_1`. Writer-only → **self-loop**.
- **TensorParameters**: `input`, `output` — both **borrow-only**: no kernel binds either, and no
  `TensorAccessor` exists in either kernel. The validator's "≥1 TensorBinding" rule carves out
  exactly this case (a parameter named by `borrowed_from` counts as used).
- **WorkUnitSpecs**: one — `{reader, writer}` over `all_cores_padded`.

### `PadRmShardedWidthOnlyProgramFactory` (`SH_W_`)

- **KernelSpecs**: `reader` (`…_stickwise.cpp`), `writer` (`…_stickwise.cpp`).
- **DataflowBufferSpecs**:
  - `in_shard` ← `c_0`, `borrowed_from = input`. Reader raw-peek only → **self-loop**.
  - `out_shard` ← `c_16`, `borrowed_from = output`. Writer locked-producer (`reserve_back`/`push_back`
    @ `:56,75`), reader locked-consumer (`wait_front`/`pop_front` @ `:37,55`) → **plain 1:1**
    (writer PRODUCER, reader CONSUMER).
  - `pad` ← `c_1`. Writer-only → **self-loop**.
- **TensorParameters**: `input`, `output` — both borrow-only, as above.
- **WorkUnitSpecs**: one — `{reader, writer}` over `all_cores_padded`.

### `PadTileCoreProgramFactory` (`TILE_SC_`)

- **KernelSpecs**: `reader` → **the existing fork**
  `eltwise/unary/device/kernels/dataflow/reader_unary_interleaved_start_id_metal2.cpp`;
  `writer` → `writer_unary_pad_dims_interleaved.cpp`.
- **DataflowBufferSpecs**:
  - `in0` ← `c_0`. reader PRODUCER (accessor name **`in`**, fixed by the fork), writer CONSUMER
    (accessor name `out0`, this op's choice).
  - `pad` ← `c_1`. Writer-only — `reserve_back(1)` + `get_write_ptr()`, never pushed →
    **self-loop**.
- **TensorParameters**: `input` — reader binds it as **`tensor::src`** (fork's vocabulary);
  `output` — writer binds as `tensor::dst`.
- **WorkUnitSpecs**: one — `{reader, writer}` over `{0,0}`.

### `PadTileMulticoreProgramFactory` (`TILE_MC_`)

- **KernelSpecs**: `reader` (`reader_pad_tiled.cpp`), `writer` (`writer_pad_tiled.cpp`).
- **DataflowBufferSpecs**:
  - `in0` ← `c_0`. reader PRODUCER, writer CONSUMER.
  - `pad` ← `c_2`. Writer-only → **self-loop**.
  - legacy `c_1` → **no spec.** Dead in every config; the allocation
    (`pad_tile_multicore_program_factory.cpp:70-78`) and the dead CTA it feeds
    (`:125` → `writer_pad_tiled.cpp:23`) are both dropped.
- **TensorParameters**: `input` (reader, `tensor::src`), `output` (writer, `tensor::dst`).
- **WorkUnitSpecs**: one — `{reader, writer}` over `all_cores`.

---

## Preserved Multiplicity

**none — no work-split multiplicity in legacy.** No factory emits more than one `KernelDescriptor`
per role. `PadRmReaderWriterMultiCoreDefaultProgramFactory` and `PadTileMulticoreProgramFactory` both
call `split_work_to_cores` and get two core groups, but the only value that differs between the
groups is a **runtime** arg (`num_sticks_per_core` / `num_pages_per_core`) that legacy already varied
per core. There is no per-group CTA to preserve and therefore nothing that could be demoted.

---

## Dropped Plumbing

### Buffer-address RTAs → `TensorBinding`

Every one arrives today as a `Buffer*` pushed into `KernelDescriptor::RTArgList` (the framework's
`BufferBinding` form), never as a bare `->address()`.

| legacy location (file:line) | legacy form | Metal 2.0 replacement |
|---|---|---|
| `pad_rm_reader_writer_program_factory.cpp:143` (reader RTA 0) | `src0_buffer` | `TensorBinding{input, "src"}` |
| `…:144` (writer RTA 1) | `dst_buffer` | `TensorBinding{output, "dst"}` |
| `…:156` (reader RTA 13) | `pad_value_const_buffer` | `TensorBinding{pad_value_const, "pad_value"}` |
| `…_multi_core_program_factory.cpp:348 / 349 / 361` | same three | same three |
| `…_default_program_factory.cpp:221` (reader RTA 0) | `src0_buffer` | `TensorBinding{input, "src"}` |
| `…_default_program_factory.cpp:222` (writer RTA 0) | `dst_buffer` | `TensorBinding{output, "dst"}` |
| `pad_tile_program_factory.cpp:122` (reader RTA 0) | `src0_buffer` | `TensorBinding{input, "src"}` — fork's name |
| `pad_tile_program_factory.cpp:126` (writer RTA 0) | `dst_buffer` | `TensorBinding{output, "dst"}` |
| `pad_tile_multicore_program_factory.cpp:222` (reader RTA 0) | `input_buffer` | `TensorBinding{input, "src"}` |
| `pad_tile_multicore_program_factory.cpp:223` (writer RTA 0) | `output_buffer` | `TensorBinding{output, "dst"}` |
| `…_height_only_program_factory.cpp:294 / 309` | `cb.buffer = src/dst_buffer` | `DataflowBufferSpec::borrowed_from` |
| `…_width_only_program_factory.cpp:75 / 91` | `cb.buffer = input/output_buffer` | `DataflowBufferSpec::borrowed_from` |

**Idle-core `0u` sentinel disappears.** `…_default_program_factory.cpp:224-225` and
`pad_tile_multicore_program_factory.cpp:225-226` push a literal `0u` instead of the `Buffer*` on
cores with no work, to skip `BufferBinding` registration. Metal 2.0 delivers the base through the
binding channel for every node the kernel runs on, so there is nothing to skip and the branch has no
translation — it goes away. Both kernels already short-circuit on a zero work count, and in practice
`split_work_to_cores` returns `all_cores == group_1 ∪ group_2`, so no idle core exists today.

### Magic CB indices in CTAs → `DFBBinding`

| legacy location | legacy form | Metal 2.0 replacement |
|---|---|---|
| `…_default_program_factory.cpp:167` (writer CTA 0) | `src0_cb_index` | `DFBBinding{IN0, "out0", CONSUMER}` |
| `…_width_only_program_factory.cpp:140` (reader CTA 5) | `input_shard_cb_index` | `DFBBinding{IN_SHARD, "in_shard", PRODUCER+CONSUMER}` |
| `…_width_only_program_factory.cpp:141` (reader CTA 6) | `output_shard_cb_index` | `DFBBinding{OUT_SHARD, "out_shard", CONSUMER}` |
| `…_width_only_program_factory.cpp:150` (writer CTA 4) | `output_shard_cb_index` | `DFBBinding{OUT_SHARD, "out_shard", PRODUCER}` |
| `…_width_only_program_factory.cpp:151` (writer CTA 5) | `pad_val_cb_index` | `DFBBinding{PAD, "pad", PRODUCER+CONSUMER}` |
| `pad_tile_program_factory.cpp:99` (writer CTA 0, 1) | `src0_cb_index`, `src1_cb_index` | `DFBBinding{IN0,"out0",CONSUMER}`, `DFBBinding{PAD,"pad",PRODUCER+CONSUMER}` |
| `pad_tile_multicore_program_factory.cpp:117` (reader CTA 0) | `input_cb_index` | `DFBBinding{IN0,"in0",PRODUCER}` |
| `pad_tile_multicore_program_factory.cpp:124,126` (writer CTA 0, 2) | `input_cb_index`, `pad_val_cb_index` | `DFBBinding{IN0,"in0",CONSUMER}`, `DFBBinding{PAD,"pad",PRODUCER+CONSUMER}` |
| `reader_pad_dims_rm_interleaved.cpp:66`, `writer…:29` (hardcoded `tt::CBIndex::c_0`) | kernel-side magic constant | `dfb::in0` |
| `reader_pad_dims_rm_interleaved_v2.cpp:88-90` (three hardcoded `tt::CBIndex::c_*`) | kernel-side magic constants | `dfb::in0`, `dfb::pad`, `dfb::pad_align` |
| `reader_pad_dims_rm_sharded.cpp:24-25`, `writer…:75-76` (hardcoded) | kernel-side magic constants | `dfb::in_shard` / `dfb::out_shard` / `dfb::pad` |

### `TensorAccessorArgs` plumbing → binding mechanism

| host `append_to` site | kernel-side chain |
|---|---|
| `pad_rm_reader_writer_program_factory.cpp:82,83,84` (and `:85`, the writer's verbatim copy) | `reader…:62-64` `TensorAccessorArgs<2>()` → `next_compile_time_args_offset()` ×2; `writer…:26-27` (`src_args` declared **only** to offset `dst_args`) |
| `…_multi_core_program_factory.cpp:246,247,248,249` | same |
| `…_default_program_factory.cpp:164` / `:172` | `reader…_v2.cpp:81` `TensorAccessorArgs<22>()`; `writer…_v2.cpp:23` `TensorAccessorArgs<5>()` |
| `pad_tile_program_factory.cpp:98` / `:100` | donor `:20` `TensorAccessorArgs<0>()`; `writer_unary_pad_dims_interleaved.cpp:26` `TensorAccessorArgs<2>()` |
| `pad_tile_multicore_program_factory.cpp:121` / `:132` | `reader_pad_tiled.cpp:27` `TensorAccessorArgs<3>()`; `writer_pad_tiled.cpp:40` `TensorAccessorArgs<7>()` |

### Page-size 3rd-argument CTAs

| legacy CTA | host computation | kernel site |
|---|---|---|
| reader CTA 21 `input_accessor_page_size` (`…_default_program_factory.cpp:163`) | `:57` / `:63` | `reader_pad_dims_rm_interleaved_v2.cpp:80,95` |
| writer CTA 4 `output_accessor_page_size` (`…_default_program_factory.cpp:171`) | `:68` / `:73` | `writer_pad_dims_rm_interleaved_v2.cpp:22,25` |

Both **Class 2 (redundant)** per the audit: the sharded branch passes `buffer->aligned_page_size()`
verbatim, the interleaved branch passes the true logical page which the interleaved accessor
realigns to the same value. The `:57-73` computation of both goes with them. **`dynamic_tensor_shape`
is NOT set** — these are compile-time args, so the page size cannot vary across shapes sharing one
compiled program.

> `num_input_pages_in_row` / `num_output_pages_in_row` are computed in the *same* `if is_sharded()`
> blocks but are **live kernel logic** (`reader…_v2.cpp:36-48`, `writer…_v2.cpp:36-55`). They stay,
> as named CTAs. Only the two `*_accessor_page_size` values drop.

### Semaphore-ID RTAs

none — the op has no semaphores.

### Positional CTAs → named CTAs

Every surviving CTA is named. The full renaming is the per-kernel CTA slot map in the Legacy
Inventory; names are the kernel's own local names (`stick_size_bytes`, `H_padded`,
`num_sticks_padded`, …).

### Dead args dropped outright

Neither named nor emitted, because no kernel reads them. Each is zero-functional-change: a CTA is a
compile-time constant nothing reads, and a dead RTA slot is a word nothing loads.

| where | slots |
|---|---|
| v1 RM reader RTAs | 3 `num_total_W`, 6 `num_unpadded_Y`, 7 `num_total_Y`, 14 `pad_value_const_buffer_nbytes`, 18 `start_src_stick_wi`, 23 `full_unpadded_X_nbytes` |
| v1 RM writer RTAs | 3, 7, 9 `num_total_X`, 19 `start_dst_stick_wi`, 22 `num_local_unpadded_Y`, 24 `full_padded_X_nbytes` — plus every reader-only slot, since `writer_rt_args = reader_rt_args` no longer holds |
| v1 RM reader/writer CTAs | 0, 1 (`unpadded_row_size_nbytes`, `padded_row_size_nbytes`) — no `get_compile_time_arg_val` exists in either kernel |
| RM default reader CTAs | 9, 10, 11 (declared-unused), 14, 15, 16, 17 (never declared) |
| sharded W reader CTAs | 1 `padded_stick_bytes`, 3 `padded_shard_height` |
| sharded W writer CTAs | 6 `padded_stick_step` |
| tile mc writer CTA | 1 `output_cb_index` — the dead-CB CTA the brief calls out |

---

## Applied Patterns

- [**Sync-free and single-ended CBs → self-loop DFB**](../../../../../../docs/source/tt-metalium/tt_metal/apis/host_apis/metal_2.0/ai/shared/port_patterns.md#pattern-sync-free-and-single-ended-cbs--self-loop-dfb)
  — seven DFBs, one toucher each: `pad` and `pad_align` (RM default, reader-only), `in_shard` and
  `pad` (sharded height-only), `in_shard` and `pad` (sharded width-only), `pad` (tile single-core),
  `pad` (tile multicore). All are DM self-loops, legal on Gen1, Quasar-uplift debt.
- [**Two-toucher DFB → assign 1P+1C**](../../../../../../docs/source/tt-metalium/tt_metal/apis/host_apis/metal_2.0/ai/shared/port_patterns.md#pattern-two-toucher-dfb--assign-1p1c-dual-instance-work-split)
  — `out_shard` in `PadRmShardedHeightOnlyProgramFactory`: reader locked-producer + writer role-free
  raw peek (`writer_pad_dims_rm_sharded.cpp:90`) → reader PRODUCER, writer CONSUMER. Census re-derived
  independently; agrees with the brief. **No multi-binding flag anywhere in this port.**
- [**Conditional / optional DFB bindings**](../../../../../../docs/source/tt-metalium/tt_metal/apis/host_apis/metal_2.0/ai/shared/port_patterns.md#pattern-conditional--optional-dfb-bindings)
  — `pad_align` (`c_2`) in the RM default factory. The host binds it only under
  `stick_size_padded_front != 0 || unaligned` and emits a matching
  `compiler_options.defines` entry `PAD_ALIGN_DFB`. The kernel currently constructs the buffer and
  calls `get_read_ptr()` **unconditionally** (`reader…_v2.cpp:93,99`) while gating only the *uses*
  behind `if constexpr`; since `if constexpr` does not suppress `dfb::` name lookup, the
  construction, the `get_read_ptr()`, **and the two `if constexpr` branches that reference the
  handle** all move behind a real `#ifdef`.
- [**Pass DFB handles directly to LLKs and kernel-lib helpers**](../../../../../../docs/source/tt-metalium/tt_metal/apis/host_apis/metal_2.0/ai/shared/port_patterns.md#pattern-pass-dfb-handles-directly-to-llks-and-kernel-lib-helpers)
  — not needed for LLKs (no compute kernels), but the file-local `fill_*` helpers in three kernels
  take a `uint32_t cb_id` and build a **second** `DataflowBuffer` for a buffer the caller already
  holds. Each helper's parameter becomes `DataflowBuffer&` so exactly one object exists per DFB, per
  the catalog's "alias the handle, keep one object" rule.
- [**Caution: Porting a shared kernel**](../../../../../../docs/source/tt-metalium/tt_metal/apis/host_apis/metal_2.0/ai/shared/port_patterns.md#caution-porting-a-shared-kernel)
  — rung 1 (reuse the existing `_metal2` fork) for the borrowed
  `reader_unary_interleaved_start_id.cpp`. No new fork; the original is not touched.
- [**Unity-build hygiene for anonymous-namespace symbols**](../../../../../../docs/source/tt-metalium/tt_metal/apis/host_apis/metal_2.0/ai/shared/port_patterns.md#pattern-unity-build-hygiene-for-anonymous-namespace-symbols)
  — seven factories in one unity TU; every spec-name constant carries a per-factory prefix.
- [**Caution: Avoid varargs**](../../../../../../docs/source/tt-metalium/tt_metal/apis/host_apis/metal_2.0/ai/shared/port_patterns.md#caution-avoid-varargs-unless-absolutely-necessary)
  — two genuine vararg sites, and two look-alikes that are **named**:
  - **Vararg** `reader_pad_dims_rm_sharded.cpp` — one named RTA (`num_cores_read`, a distinct field
    read once) followed by a runtime-count collection: `2·num_cores` NoC (x,y), `num_cores` chunk
    counts, `2·Σchunks` (start,length) pairs. Counts come from the data, so nothing past the first
    word is nameable.
  - **Vararg** `reader_pad_tiled.cpp` / `writer_pad_tiled.cpp` — four `num_dims`-long blocks, read in
    `for (d < num_dims)` loops with `num_dims` a CTA.
  - **Named** `reader_pad_dims_rm_interleaved_v2.cpp:59,112` and
    `writer_pad_dims_rm_sharded.cpp:53,93` — both index a `get_arg_addr()` pointer at **fixed**
    positions `[1]`, `[2]`, `[3]` only. Three distinct nameable fields
    (`start_dim_offset_h/_c/_n`), not a data-directed pick.

---

## Deferred / Flagged

New findings from planning (none blocks the port; all are reported):

1. **`get_vararg()` is read-only, and `reader_pad_tiled` / `writer_pad_tiled` *write back* into their
   vararg block.** `advance_tensor_index` (`device/kernels/dataflow/common.hpp:12-19`) mutates
   `input_id_per_dim` / `output_id_per_dim` in place in the RTA buffer. Metal 2.0 exposes
   `get_vararg(i)` (a value getter) and no address form, so the two mutable blocks are **copied into
   local `uint32_t[num_dims]` arrays** at kernel entry — `num_dims` is a CTA, so these are
   fixed-size arrays. The two read-only blocks (`input_page_shape`, `output_page_shape`) stay as
   direct `get_vararg` reads. This is the only place the port changes *where* a value lives rather
   than *how* it is named; behavior is identical. Friction entry in the report.
2. **DFB placement narrows on the two sharded factories.** Their `CBDescriptor`s carry
   `core_ranges = total_cores` (the full compute grid) while their kernels run on
   `all_cores_padded`. Metal 2.0 derives DFB placement from kernel bindings, so the ported `pad` DFB
   is allocated only where a kernel binds it. The borrowed DFBs allocate nothing either way; the
   change is a strictly smaller L1 footprint on cores that ran no kernel. Not expressible otherwise
   — `DataflowBufferSpec` has no `target_nodes` by design.
3. **`start_dim_offset` is emitted from the live counters, not the vector.** In
   `…_default_program_factory.cpp` and `…_height_only_program_factory.cpp` the vector is seeded to
   `num_dims` zeros and thereafter reassigned to a literal 4-element `{0, curr_h, curr_c, curr_n}`.
   At the point of the push, `start_dim_offset[1..3]` is **exactly** `curr_h`, `curr_c`, `curr_n`
   (both are zero on the first core, and the reassignment happens after the push). The three named
   args are therefore emitted straight from the counters, which is value-identical and sidesteps the
   pre-existing rank-mismatch seeding (audit anomaly #7) without changing it.
4. **`log_rt_args(CoreCoord{0,0}, reader_desc.compile_time_args)`
   (`…_multi_core_program_factory.cpp:276`) loses its subject.** It logs the positional CTA vector,
   which ceases to exist; the helper at `:20-24` then has no caller. Both are removed as part of the
   descriptor code they belong to.
5. **No new stop signals.** No `GlobalCircularBuffer`, no `get_cb_tiles_acked_ptr` /
   `get_cb_tiles_received_ptr`, no compute-kernel Case 2, no host-computed `base + offset`, no
   descriptor type outside the audit's Appendix A scope.
