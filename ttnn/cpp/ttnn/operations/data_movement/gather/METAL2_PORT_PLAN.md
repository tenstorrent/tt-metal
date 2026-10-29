# Port Plan: gather

Port plan for `ttnn/cpp/ttnn/operations/data_movement/gather`, ported from the `ProgramDescriptor` API to Metal 2.0.
Written during the inventory and planning steps; committed alongside the port for review.

The op directory holds two independent device operations with seven factories in total. All seven are ported in this change (the brief's scope). Each factory is its own atomic unit: one factory plus the two kernels it binds.

**About the references in this file.** The legacy inventory describes code that the port rewrites, so a link into today's files would land on the ported code at the wrong lines. Legacy locations are therefore given as plain `path:line` text, relative to the op directory, at commit `e9385a86c5c` (the pre-port tree: the audit commit plus the Device 2.0 commit). Read them with `git show e9385a86c5c:ttnn/cpp/ttnn/operations/data_movement/gather/<path>`.

## Legacy Inventory

### Legacy factory shape

- Concept: `ProgramDescriptorFactoryConcept` for all seven factories. Each one defines `create_descriptor`, which returns a `ProgramDescriptor` and takes no `MeshCoordinate`.
- Where the factory methods live:
  - `GatherDeviceOperation`: four structs nested inside the device-operation struct, in its `program_factory_t` variant (`device/gather_device_operation.hpp:24-46`). Bodies in `device/gather_program_factory.cpp`.
  - `GatherCodegenDeviceOperation`: three free structs in `codegen/gather_codegen_program_factory.hpp:70-88`, in the variant at `codegen/gather_codegen_device_operation.hpp:46-49`. Bodies in `codegen/gather_codegen_program_factory.cpp`.
  - Neither device operation declares `create_descriptor` on itself (the direct-descriptor case), so exception 3 of the TTNN integration doc does not apply.
- Variants: seven factories (four native, three codegen), described one by one below.
- Custom `compute_program_hash`:
  - `GatherCodegenDeviceOperation::compute_program_hash` at `codegen/gather_codegen_device_operation.cpp:55-69`: left intact. It hashes the same `attributes` and `tensor_args` as the default hash (each `Tensor` hashes its full `TensorSpec`), plus the selected factory index and the streaming depth. So two calls share a cache entry only when every tensor's `TensorSpec` is equal, which is what the strict `TensorParameter` match (relaxation `none`) needs.
  - `GatherDeviceOperation`: none (default reflection-based hash). No `attribute_values` / `to_hash` backdoor.
- No `override_runtime_arguments`, no `get_dynamic_runtime_args`, no pybound `create_descriptor`, no op-owned tensors, no semaphores, no `defines`, no named CTAs, no CRTAs, no `opt_level` (`grep -rn opt_level` over the op directory: 0 hits) in any factory.

Common to all seven factories:

- Reader `config` is `ReaderConfigDescriptor{}`, which resolves to `RISCV_1`, `NOC_0` (`preferred_noc_for_dram_read`, WH and BH), `DM_DEDICATED_NOC`, `opt_level` O2 (`tt_metal/impl/kernels/kernel_types.cpp:13-27`, `tt_metal/impl/program/program.cpp:523-529`).
- Writer `config` is `WriterConfigDescriptor{}`, which resolves to `RISCV_0`, `NOC_1`, `DM_DEDICATED_NOC`, O2.
- Each CB is single-format (`format_descriptors` has one element), has no `tile` set, borrows no buffer, and sits on the same `core_ranges` as both kernels.
- Every tensor address reaches a kernel as a `Buffer*` runtime arg; each kernel builds its accessor from `TensorAccessorArgs<N>()` plus that address.

### Variant: `SingleRowSingleCore` (native, TILE; `device/gather_program_factory.cpp:45-199`)

#### Kernels

| unique_id | source | core_ranges | CTAs (positional) | CTAs (named) | RTAs | CRTAs | defines | opt_level | config |
|---|---|---|---|---|---|---|---|---|---|
| reader | `device/kernels/dataflow/gather_reader_single_row_single_core.cpp` | `core_range` | 0 `c_0`, 1 `c_1`, 2 `c_2`, 3 `input_index_tensor_is_dram`, 4 `Ht`, 5 `Wt_input`, 6 `Wt_index`, 7 `total_number_of_cores`, 8 grid x, 9 grid y, 10+ `TensorAccessorArgs(index)` | none | 0 index `Buffer*`, 1 `work_per_core`, 2 `tile_width`, 3 `tile_height`, 4 `id` (idle core: `0u` for slots 0-1) | none | none | O2 (unset) | Reader |
| writer | `device/kernels/dataflow/gather_writer_single_row_single_core.cpp` | `core_range` | 0 `c_0`, 1 `c_2`, 2 `input_tensor_is_dram`, 3 `output_tensor_is_dram`, 4 `Ht`, 5 `Wt_input`, 6 `Wt_index`, 7 `total_number_of_cores`, 8 grid x, 9 grid y, 10+ `TensorAccessorArgs(input)`, then `TensorAccessorArgs(output)` | none | 0 input `Buffer*`, 1 output `Buffer*`, 2 `work_per_core`, 3 `id` (idle core: `0u` for slots 0-2) | none | none | O2 (unset) | Writer |

Host emission order: reader CTAs at `:135-146`, writer CTAs at `:155-167`, RTAs at `:176-194`.

#### CBs

| index | total_size | core_ranges | data_format | page_size | tile |
|---|---|---|---|---|---|
| `c_0` (input) | `Wt_input * tile_size(input fmt)` | `core_range` | input dtype | `tile_size(input fmt)` | unset |
| `c_1` (index) | `tile_size(index fmt)` | `core_range` | index dtype | `tile_size(index fmt)` | unset |
| `c_2` (output) | `tile_size(output fmt)` | `core_range` | output dtype | `tile_size(output fmt)` | unset |

#### Semaphores

none

#### Tensor accessors

| host site | originating Tensor | RTA slot (host) |
|---|---|---|
| `:146` (reader CTAs), `:183` (RTA) | `input_index_tensor` | reader RTA 0 |
| `:166` (writer CTAs), `:185` (RTA) | `input_tensor` | writer RTA 0 |
| `:167` (writer CTAs), `:185` (RTA) | `output_tensor` | writer RTA 1 |

#### Work split

- Driver: `split_work_to_cores(core_grid, Ht, /*row_wise=*/true)`, where `core_grid` is `num_cores_to_corerangeset(max cores, grid, true)` or `attributes.sub_core_grids`.
- `core_group_1` / `core_group_2` with `num_tiles_per_core_group_1/2` tile-rows each; `id` is a running ordinal over group 1 then group 2.

### Variant: `SingleRowMultiCore` (native, TILE; `device/gather_program_factory.cpp:201-356`)

Same kernel layout, CTA list, RTA list, accessors and configs as `SingleRowSingleCore`, with these differences:

- Sources: `gather_reader_single_row_multi_core.cpp`, `gather_writer_single_row_multi_core.cpp`.
- Work split over `Wt_index` instead of `Ht`: `split_work_to_cores(core_grid, Wt_index, true)`.
- `c_0` depth is `buffer_scale_factor` (2) tiles instead of `Wt_input`.
- Host emission: reader CTAs `:292-303`, writer CTAs `:312-324`, RTAs `:333-351`.

### Variant: `RmSingleRowSingleCore` (native, ROW_MAJOR; `device/gather_program_factory.cpp:361-515`)

#### Kernels

| unique_id | source | core_ranges | CTAs (positional) | CTAs (named) | RTAs | CRTAs | defines | opt_level | config |
|---|---|---|---|---|---|---|---|---|---|
| reader | `device/kernels/dataflow/gather_reader_rm_single_row_single_core.cpp` | `core_range` | 0 `c_0`, 1 `c_1`, 2 `c_2`, 3 `H`, 4 `W_input`, 5 `W_index`, 6 `total_number_of_cores`, 7 `input_elem_size`, 8 `input_index_elem_size`, 9 `output_elem_size`, 10+ `TensorAccessorArgs(index)` | none | 0 index `Buffer*`, 1 `work_per_core`, 2 `id`, 3 index per-shard page size (idle core: `0u` for slots 0-1) | none | none | O2 (unset) | Reader |
| writer | `device/kernels/dataflow/gather_writer_rm_single_row_single_core.cpp` | `core_range` | 0 `c_0`, 1 `c_2`, 2 `H`, 3 `W_input`, 4 `W_index`, 5 `total_number_of_cores`, 6 `input_elem_size`, 7 `output_elem_size`, 8+ `TensorAccessorArgs(input)`, then `TensorAccessorArgs(output)` | none | 0 input `Buffer*`, 1 output `Buffer*`, 2 `work_per_core`, 3 `id`, 4 input per-shard page size, 5 output per-shard page size (idle core: `0u` for slots 0-2) | none | none | O2 (unset) | Writer |

Host emission: reader CTAs `:440-451`, writer CTAs `:460-470`, page sizes `:480-484`, RTAs `:486-510`.

#### CBs

| index | total_size | core_ranges | data_format | page_size | tile |
|---|---|---|---|---|---|
| `c_0` (input) | `input_stick_size_bytes` | `core_range` | input dtype | `input_stick_size_bytes` | unset |
| `c_1` (index) | `input_index_stick_size_bytes` | `core_range` | index dtype | `input_index_stick_size_bytes` | unset |
| `c_2` (output) | `output_stick_size_bytes` | `core_range` | output dtype | `output_stick_size_bytes` | unset |

#### Semaphores

none

#### Tensor accessors

| host site | originating Tensor | RTA slot (host) |
|---|---|---|
| `:451`, `:492` | `input_index_tensor` | reader RTA 0 (3rd accessor arg: reader RTA 3) |
| `:469`, `:495` | `input_tensor` | writer RTA 0 (3rd accessor arg: writer RTA 4) |
| `:470`, `:496` | `output_tensor` | writer RTA 1 (3rd accessor arg: writer RTA 5) |

#### Work split

- Driver: `split_work_to_cores(core_grid, H, true)`, `H` = rows of the index tensor.

### Variant: `RmSingleRowMultiCore` (native, ROW_MAJOR; `device/gather_program_factory.cpp:518-683`)

#### Kernels

| unique_id | source | core_ranges | CTAs (positional) | CTAs (named) | RTAs | CRTAs | defines | opt_level | config |
|---|---|---|---|---|---|---|---|---|---|
| reader | `device/kernels/dataflow/gather_reader_rm_single_row_multi_core.cpp` | `core_range` | 0 `c_0`, 1 `c_1`, 2 `c_2`, 3 `H`, 4 `W_input`, 5 `W_index`, 6 `input_elem_size`, 7 `input_index_elem_size`, 8 `output_elem_size`, 9+ `TensorAccessorArgs(index)` | none | 0 index `Buffer*`, 1 `w_per_core`, 2 `w_start`, 3 `id`, 4 index per-shard page size (idle core: `0u` for slots 0-1) | none | none | O2 (unset) | Reader |
| writer | `device/kernels/dataflow/gather_writer_rm_single_row_multi_core.cpp` | `core_range` | 0 `c_0`, 1 `c_2`, 2 `H`, 3 `W_input`, 4 `W_index`, 5 `input_elem_size`, 6 `output_elem_size`, 7+ `TensorAccessorArgs(input)`, then `TensorAccessorArgs(output)` | none | 0 input `Buffer*`, 1 output `Buffer*`, 2 `w_per_core`, 3 `w_start`, 4 `id`, 5 input per-shard page size, 6 output per-shard page size (idle core: `0u` for slots 0-2) | none | none | O2 (unset) | Writer |

Host emission: reader CTAs `:613-623`, writer CTAs `:632-635`, page sizes `:644-648`, RTAs `:650-678`.

#### CBs

| index | total_size | core_ranges | data_format | page_size | tile |
|---|---|---|---|---|---|
| `c_0` (input) | `2 * align(input_stick_size_bytes, input alignment)` | `core_range` | input dtype | `align(input_stick_size_bytes, input alignment)` | unset |
| `c_1` (index) | `max_w_per_core * input_index_elem_size` | `core_range` | index dtype | same as total | unset |
| `c_2` (output) | `max_w_per_core * output_elem_size` | `core_range` | output dtype | same as total | unset |

#### Semaphores

none

#### Tensor accessors

| host site | originating Tensor | RTA slot (host) |
|---|---|---|
| `:623`, `:659` | `input_index_tensor` | reader RTA 0 (3rd accessor arg: reader RTA 4) |
| `:634`, `:662` | `input_tensor` | writer RTA 0 (3rd accessor arg: writer RTA 5) |
| `:635`, `:663` | `output_tensor` | writer RTA 1 (3rd accessor arg: writer RTA 6) |

#### Work split

- Driver: `split_work_to_cores(core_grid, div_up(W_index, w_unit), true)`; each group's unit count is scaled by `w_unit` columns, and the last cores take `min(w_per_group_core, W_index - w_cursor)` columns, which can be 0.

### Variant: `GatherCodegenProgramFactoryInterleaved` (`codegen/gather_codegen_program_factory.cpp:263-338`)

#### Kernels

| unique_id | source | core_ranges | CTAs (positional) | CTAs (named) | RTAs | CRTAs | defines | opt_level | config |
|---|---|---|---|---|---|---|---|---|---|
| reader | `codegen/kernels/gather_reader.cpp` | `split.core_range` | 0 `c_0`, 1 `c_1`, 2 `c_2`, 3 `Wt_input`, 4 `Wt_index`, 5 `num_cores`, 6 `index_valid_h_last`, 7 `index_valid_w_last`, 8 `index_ht_per_batch`, 9+ `TensorAccessorArgs(index)` | none | 0 index `Buffer*`, 1 `n`, 2 `tile_width`, 3 `tile_height`, 4 `id` | none | none | O2 (unset) | Reader |
| writer | `codegen/kernels/gather_writer.cpp` | `split.core_range` | 0 `c_0`, 1 `c_2`, 2 `Wt_input`, 3 `Wt_index`, 4 `num_cores`, 5 `gather_output_cb_tiles(Wt_index)`, 6 `kGatherWriteBatchTiles`, 7+ `TensorAccessorArgs(input)`, then `TensorAccessorArgs(output)` | none | 0 input `Buffer*`, 1 output `Buffer*`, 2 `n`, 3 `id` | none | none | O2 (unset) | Writer |

#### CBs (all built by `make_tile_cb`, `:161-173`: page = the tensor's `aligned_page_size()`, format = the tensor's dtype, no tile)

| index | depth (pages) | tensor |
|---|---|---|
| `c_0` (input) | `Wt_input` | `input_tensor` |
| `c_1` (index) | 1 | `input_index_tensor` |
| `c_2` (output) | `gather_output_cb_tiles(Wt_index)` | `output_tensor` |

#### Tensor accessors: index (reader RTA 0, `:298`, `:330`), input (writer RTA 0, `:315`, `:331`), output (writer RTA 1, `:316`, `:331`).

#### Work split: `split_gather_work(device, sub_core_grids, Ht)` (`row_wise=false`); `gather_assigned_cores` yields every core of `split.core_range` with its count `n`; `id` is a running ordinal.

### Variant: `GatherCodegenProgramFactoryTiled` (`codegen/gather_codegen_program_factory.cpp:340-416`)

| unique_id | source | CTAs (positional) | RTAs |
|---|---|---|---|
| reader | `codegen/kernels/gather_reader_tiled.cpp` | 0 `c_0`, 1 `c_1`, 2 `c_2`, 3 `Wt_input`, 4 `Wt_index`, 5 `index_valid_h_last`, 6 `index_valid_w_last`, 7 `index_ht_per_batch`, 8+ `TensorAccessorArgs(index)` | 0 index `Buffer*`, 1 `start`, 2 `n`, 3 `tile_width`, 4 `tile_height` |
| writer | `codegen/kernels/gather_writer_tiled.cpp` | 0 `c_0`, 1 `c_2`, 2 `Wt_input`, 3 `Wt_index`, 4 `gather_output_cb_tiles(Wt_index)`, 5 `kGatherWriteBatchTiles`, 6+ `TensorAccessorArgs(input)`, then `TensorAccessorArgs(output)` | 0 input `Buffer*`, 1 output `Buffer*`, 2 `start`, 3 `n` |

Core ranges, configs, CTA/RTA kinds and opt levels as in Interleaved. CBs: identical to Interleaved. Work split over `Ht * Wt_index` output tiles; per-core contiguous `[start, start + n)`.

### Variant: `GatherCodegenProgramFactoryStreaming` (`codegen/gather_codegen_program_factory.cpp:418-492`)

| unique_id | source | CTAs (positional) | RTAs |
|---|---|---|---|
| reader | `codegen/kernels/gather_reader_streaming.cpp` | 0 `c_0`, 1 `c_1`, 2 `c_2`, 3 `Ht`, 4 `Wt_input`, 5 `Wt_index`, 6 `num_cores`, 7 `index_valid_h_last`, 8 `index_valid_w_last`, 9 `index_ht_per_batch`, 10 `chunk_tiles`, 11+ `TensorAccessorArgs(index)` | 0 index `Buffer*`, 1 `n`, 2 `tile_width`, 3 `tile_height`, 4 `id` |
| writer | `codegen/kernels/gather_writer_streaming.cpp` | 0 `c_0`, 1 `c_2`, 2 `Ht`, 3 `Wt_input`, 4 `Wt_index`, 5 `num_cores`, 6 `chunk_tiles`, 7+ `TensorAccessorArgs(input)`, then `TensorAccessorArgs(output)` | 0 input `Buffer*`, 1 output `Buffer*`, 2 `n`, 3 `id` |

CBs: `c_0` depth `chunk_tiles` (`gather_streaming_chunk_tiles`, which reads live SRAM occupancy), `c_1` depth 1, `c_2` depth 1. Work split over `Wt_index` (ordinal convention, as Interleaved).

### Shared kernels

none. Check: `grep -rlw <kernel-stem> ttnn/cpp/ttnn/operations/` (`experimental/quasar/` excluded) for each of the 14 kernel files and the two helper headers finds only files in this op's directory. The only binders are the two factory `.cpp` files, one factory per kernel. The other hits are mentions in comments (the codegen factory header, and each codegen kernel naming its partner), which are not consumers. The helper headers are included only by this op's own kernels. No `_metal2` sibling exists beside any kernel. All 14 convert in place.

### Flags

- Kernel helper headers `device/kernels/dataflow/gather_common.hpp` and `codegen/kernels/codegen_gather_common.hpp` hold only raw SRAM element helpers on `uint32_t` addresses. No change needed.
- `docs/Gather.md` describes a `CircularBufferConfig` setup that is already out of date before the port; it is outside the port's writeable surface (routed to the report).
- The row-major kernels call the in-family donor `data_movement/common/kernels/common.hpp` (`noc_async_read_sharded` / `noc_async_write_sharded`). After the Device 2.0 commit `e9385a86c5c` they call the `Noc` overloads. The donor needs no change.

## TTNN ProgramFactory

- **Concept (inherited from audit)**: `ProgramSpecFactoryConcept` for all seven factories (codegen: recorded under user override 1).
- **Custom `compute_program_hash`**: present at `codegen/gather_codegen_device_operation.cpp:55-69`: leave intact. `GatherDeviceOperation`: none.
- **Implementation notes**:
  - The four native factory structs are declared inside `GatherDeviceOperation` (`device/gather_device_operation.hpp`). Their method declarations change from `create_descriptor` to `create_program_artifacts`, and the header swaps `<tt-metalium/program_descriptors.hpp>` for `"ttnn/metal_v2_artifacts.hpp"`. These are factory declarations, so this is factory-surface work; nothing else in that header changes.
  - The codegen factory header `codegen/gather_codegen_program_factory.hpp` changes the same way. The helper functions it also declares (`compute_gather_geometry`, `gather_output_cb_tiles`, `gather_interleaved_fits_l1`, `gather_min_plan_fits_l1`, `gather_streaming_chunk_tiles`) keep their names and behavior, because the device operation and the routing gate call them.
  - Each factory extracts its three `MeshTensor`s at the top (`tensor_args.input_tensor.mesh_tensor()`, `tensor_args.input_index_tensor.mesh_tensor()`, `output_tensor.mesh_tensor()`), declares each `TensorParameter` from that `MeshTensor`'s `tensor_spec()`, and passes the same `MeshTensor` to `tensor_args`. The other host computations keep using the `ttnn::Tensor` accessors they use today.
  - No device-operation-class edits are forced: no pybind entry point disappears, no factory parameter exists only for a hook, and the factories are not declared on the device operation itself.

## Planned Spec Shape

Every factory has the same structure:

- **KernelSpecs** (2): `reader` and `writer`, one per legacy `KernelDescriptor`, same sources.
  - `hw_config`: `ttnn::create_reader_datamovement_config()` / `ttnn::create_writer_datamovement_config()`. On Gen1 these give exactly the resolved legacy triples: reader `RISCV_1`/`NOC_0`/dedicated, writer `RISCV_0`/`NOC_1`/dedicated.
  - `compiler_options.opt_level`: left at the default O2, which matches the resolved legacy O2 of an unset DM `opt_level`. No compute kernel exists, so the O3 rule does not apply.
  - `compile_time_args`: every legacy positional CTA that is not a CB index or a `TensorAccessorArgs` block becomes a named CTA, under the name of the kernel variable it is assigned to (Dropped Plumbing below). This keeps the CTAs that the kernels declare but never read (the `*_is_dram` flags, the grid sizes, `Ht` in two kernels, and so on), since dropping them would change how many kernel binaries the JIT builds.
  - `runtime_arg_schema.runtime_arg_names`: the legacy RTAs minus the address slots and the per-shard page-size slots, named after the kernel variables.
- **DataflowBufferSpecs** (3), one per legacy `CBDescriptor`, same order, `entry_size` = legacy `page_size`, `num_entries` = `total_size / page_size`, `data_format_metadata` = legacy `data_format`, no `tile_format_metadata` (legacy `tile` unset), no `borrowed_from`, no `advanced_options`.
  - Native names: `input_tensor`, `input_index_tensor`, `output_tensor` (from the legacy CTA names `input_tensor_cb_index`, ...; the same names `sort` uses).
  - Codegen names: `input`, `index`, `output` (from the legacy CTA names `cb_input`, `cb_index`, `cb_output`).
  - Accessor names equal the spec names, so the kernels read `dfb::input_tensor`, `dfb::index`, and so on.
- **SemaphoreSpecs**: none.
- **TensorParameters** (3): native `input_tensor`, `input_index_tensor`, `output_tensor`; codegen `input`, `index`, `output`. Each `spec` = the `MeshTensor`'s `tensor_spec()`, no relaxations (brief: `none`). Bindings: index → reader; input and output → writer. Accessor names equal the parameter names.
- **WorkUnitSpecs** (1): `main`, kernels `{reader, writer}`, `target_nodes` = the legacy `core_ranges` (`core_range` native, `split.core_range` codegen).
- **Op-owned tensors**: none.

### DFB endpoints (re-derived from the kernel bodies; agrees with the brief)

| DFB | reader | writer | resolution |
|---|---|---|---|
| input | `wait_front` / `pop_front` → locked CONSUMER | `reserve_back` / `push_back` → locked PRODUCER | plain 1:1 |
| index | `reserve_back` / `push_back` and `pop_front` (plus `wait_front` in the TILE and codegen readers) | not touched | only the reader touches it → self-loop: reader bound PRODUCER and CONSUMER under one accessor name |
| output | `reserve_back` / `push_back` → locked PRODUCER | `wait_front` / `pop_front` → locked CONSUMER | plain 1:1 |

The same endpoint count holds in all seven factories, on every node, in every config. No multi-binding flag, no conditional binding, no dead DFB.

## Preserved Multiplicity

none: no work-split multiplicity in legacy. Each factory has one reader and one writer `KernelDescriptor` over one `core_ranges`; the per-group counts travel as RTAs in legacy and stay RTAs.

## Dropped Plumbing

### Buffer-address RTAs → `TensorBinding` (all `Buffer*` runtime args)

| factory | kernel | legacy RTA slot (form) | Metal 2.0 replacement |
|---|---|---|---|
| SingleRowSingleCore, SingleRowMultiCore | reader | 0 (`input_index_tensor_buffer`) | `TensorBinding(input_index_tensor → "input_index_tensor")` |
| SingleRowSingleCore, SingleRowMultiCore | writer | 0, 1 (`input_tensor_buffer`, `output_tensor_buffer`) | `TensorBinding(input_tensor)`, `TensorBinding(output_tensor)` |
| RmSingleRowSingleCore, RmSingleRowMultiCore | reader | 0 | `TensorBinding(input_index_tensor)` |
| RmSingleRowSingleCore, RmSingleRowMultiCore | writer | 0, 1 | `TensorBinding(input_tensor)`, `TensorBinding(output_tensor)` |
| Interleaved, Tiled, Streaming | reader | 0 (`index_t.buffer()`) | `TensorBinding(index → "index")` |
| Interleaved, Tiled, Streaming | writer | 0, 1 (`in_t.buffer()`, `output_tensor.buffer()`) | `TensorBinding(input → "input")`, `TensorBinding(output → "output")` |

All bindings are Case 1 (`TensorAccessor(tensor::name)`), data movement only.

### Magic CB indices in CTAs → `DFBBinding`

| factory | kernel | legacy CTA slots | replacement |
|---|---|---|---|
| all four native | reader | 0 `c_0`, 1 `c_1`, 2 `c_2` | `dfb::input_tensor` (CONSUMER), `dfb::input_index_tensor` (PRODUCER + CONSUMER), `dfb::output_tensor` (PRODUCER) |
| all four native | writer | 0 `c_0`, 1 `c_2` | `dfb::input_tensor` (PRODUCER), `dfb::output_tensor` (CONSUMER) |
| all three codegen | reader | 0 `c_0`, 1 `c_1`, 2 `c_2` | `dfb::input` (CONSUMER), `dfb::index` (PRODUCER + CONSUMER), `dfb::output` (PRODUCER) |
| all three codegen | writer | 0 `c_0`, 1 `c_2` | `dfb::input` (PRODUCER), `dfb::output` (CONSUMER) |

### `TensorAccessorArgs` plumbing → binding

Every `TensorAccessorArgs(*buffer).append_to(cta)` site (14 host sites: native `:146`, `:166-167`, `:303`, `:323-324`, `:451`, `:469-470`, `:623`, `:634-635`; codegen `:298`, `:315-316`, `:377`, `:393-394`, `:459`, `:470-471`) and every kernel-side `TensorAccessorArgs<N>()` / `next_compile_time_args_offset()` chain (all 14 kernels) goes away.

### Page-size 3rd-argument RTAs / constants

| factory | kernel | legacy | replacement |
|---|---|---|---|
| RmSingleRowSingleCore | reader | RTA 3 `input_index_per_shard_page_size_bytes` (host `:482-483`) | dropped; the token supplies the aligned page |
| RmSingleRowSingleCore | writer | RTA 4 `input_per_shard_page_size_bytes`, RTA 5 `output_per_shard_page_size_bytes` (host `:480-481`, `:484`) | dropped |
| RmSingleRowMultiCore | reader | RTA 4 (host `:646-647`) | dropped |
| RmSingleRowMultiCore | writer | RTA 5, RTA 6 (host `:644-645`, `:648`) | dropped |
| all three codegen | all six kernels | 3rd argument `index_tile_bytes` / `input_tile_bytes` / `output_tile_bytes` (9 accessors) | argument dropped; the `constexpr ..._tile_bytes` locals stay, because each is also a transfer size or SRAM offset step |

With these gone, `using ttnn::operations::data_movement::per_shard_page_size_bytes;` (`device/gather_program_factory.cpp:18`) and its include of `ttnn/operations/data_movement/common/common.hpp` have no remaining use; both go. `<tt-metalium/tensor_accessor_args.hpp>` goes from both factory files.

### Semaphore-ID RTAs

none.

### Positional CTAs → named CTAs (names = the kernel variables they feed)

| kernel(s) | named CTAs |
|---|---|
| native TILE readers (single, multi) | `input_index_tensor_is_dram`, `Ht`, `Wt_input`, `Wt_index`, `total_number_of_cores`, `compute_with_storage_grid_size_x`, `compute_with_storage_grid_size_y` |
| native TILE writers (single, multi) | `input_tensor_is_dram`, `output_tensor_is_dram`, `Ht`, `Wt_input`, `Wt_index`, `total_number_of_cores`, `compute_with_storage_grid_size_x`, `compute_with_storage_grid_size_y` |
| RM single reader | `H`, `W_input`, `W_index`, `total_number_of_cores`, `input_tensor_data_format_size`, `input_index_tensor_data_format_size`, `output_tensor_data_format_size` |
| RM single writer | `H`, `W_input`, `W_index`, `total_number_of_cores`, `input_tensor_data_format_size`, `output_tensor_data_format_size` |
| RM multi reader | `H`, `W_input`, `W_index`, `input_tensor_data_format_size`, `input_index_tensor_data_format_size`, `output_tensor_data_format_size` |
| RM multi writer | `H`, `W_input`, `W_index`, `input_tensor_data_format_size`, `output_tensor_data_format_size` |
| codegen interleaved reader | `Wt_input`, `Wt_index`, `num_cores`, `index_valid_h_last`, `index_valid_w_last`, `index_ht_per_batch` |
| codegen interleaved writer | `Wt_input`, `Wt_index`, `num_cores`, `OUT_DFB_DEPTH` (kernel variable `OUT_CB_DEPTH`, renamed with the API), `WRITE_BATCH` |
| codegen tiled reader | `Wt_input`, `Wt_index`, `index_valid_h_last`, `index_valid_w_last`, `index_ht_per_batch` |
| codegen tiled writer | `Wt_input`, `Wt_index`, `OUT_DFB_DEPTH`, `WRITE_BATCH` |
| codegen streaming reader | `Ht`, `Wt_input`, `Wt_index`, `num_cores`, `index_valid_h_last`, `index_valid_w_last`, `index_ht_per_batch`, `chunk_tiles` |
| codegen streaming writer | `Ht`, `Wt_input`, `Wt_index`, `num_cores`, `chunk_tiles` |

### Positional RTAs → named RTAs

| kernel(s) | named RTAs |
|---|---|
| native TILE readers | `core_loop_count`, `tile_width`, `tile_height`, `core_id` |
| native TILE writers | `core_loop_count`, `core_id` |
| RM single reader and writer | `core_loop_count`, `core_id` |
| RM multi reader and writer | `w_per_core`, `w_start`, `core_id` |
| codegen interleaved and streaming readers | `core_loop_count`, `tile_width`, `tile_height`, `core_id` |
| codegen interleaved and streaming writers | `core_loop_count`, `core_id` |
| codegen tiled reader | `start`, `n`, `tile_width`, `tile_height` |
| codegen tiled writer | `start`, `n` |

No varargs: every argument is a distinct field read once at the top of its kernel.

## Applied Patterns

- [Sync-free and single-ended CBs → self-loop DFB](../../../../../../docs/source/tt-metalium/tt_metal/apis/host_apis/metal_2.0/ai/shared/port_patterns.md#pattern-sync-free-and-single-ended-cbs--self-loop-dfb): the index DFB on the reader in all seven factories (only the reader touches it; a DM self-loop is legal on Gen1).
- [Pass DFB handles directly to LLKs and kernel-lib helpers](../../../../../../docs/source/tt-metalium/tt_metal/apis/host_apis/metal_2.0/ai/shared/port_patterns.md#pattern-pass-dfb-handles-directly-to-llks-and-kernel-lib-helpers): the `constexpr` metadata lines keep the free functions with the token (`get_tile_size(dfb::...)`, `get_tile_hw(dfb::...)`, `get_dataformat(dfb::...)`), per CB→DFB whitelist §A.
- [Unity-build hygiene for anonymous-namespace symbols](../../../../../../docs/source/tt-metalium/tt_metal/apis/host_apis/metal_2.0/ai/shared/port_patterns.md#pattern-unity-build-hygiene-for-anonymous-namespace-symbols): spec names are declared inside each factory function, not at file scope. The two factory files are in the same unity library (`ttnn_op_data_movement`), and this workspace's build has unity builds off (`-e` exports compile commands), so a collision would show only in CI.

## Deferred / Flagged

- **Idle-core branches collapse.** The four native factories emit RTAs through an `if (work_per_core > 0) { ... } else { ... }` whose only difference was `Buffer*` versus `0u` in the address slots (the comment says this kept the descriptor framework from patching idle cores on a cache hit). With the address slots and page-size slots gone, the two branches emit identical values in every factory, since the idle branch runs only when the work count is 0. The port emits the arguments once, without the branch, and drops the branch comments, which describe `Buffer*` bindings that no longer exist. The brief's wording ("the address slots go away in both branches ... keep the comments") assumed the branches would still differ; this is the one place the plan departs from it, and the report records it.
- **`gather_output_cb_tiles` keeps its name.** It is declared in the factory header, but `codegen/gather_codegen_supported.cpp:110` (device-operation side, off-limits) and `tests/ttnn/unit_tests/operations/data_movement/test_gather.py:378-409` name it in comments, so a rename would leave dangling references. The kernel variable `OUT_CB_DEPTH` and the local `out_cb_pos` are renamed with the API (`OUT_DFB_DEPTH`, `out_dfb_pos`).
- **CRTA candidates, not converted.** `tile_width` / `tile_height` (TILE and codegen readers) hold the same value on every node. Noted for a later pass; RTA→CRTA changes dispatch.
- **Unused CTAs kept** (audit misc anomalies): kept as named CTAs, since removing them changes the JIT binary count. Routed to the report.
