# Port Plan — `ttnn/cpp/ttnn/operations/data_movement/sort`

Port plan for `ttnn::prim::SortDeviceOperation`, ported from `ProgramDescriptor` / `WorkloadDescriptor`
to Metal 2.0. Written during the inventory and planning steps; committed alongside the port for review.

Scope planned: all **three** program factories, in one pass.

- `SortProgramFactorySingleRowSingleCore` — `create_descriptor()`
- `SortProgramFactorySingleRowMultiCore` — `create_descriptor()`
- `SortProgramFactoryCrossCoreDataExchange` — `create_workload_descriptor()` (+ op-owned tensor)

> **Outcome: two of the three shipped.** The plan below was executed in full for
> `SingleRowSingleCore` and `CrossCoreDataExchange`, which are on `MetalV2FactoryConcept` and pass
> their tests. `SingleRowMultiCore` was converted per this plan, built cleanly, and then had to be
> reverted to the legacy concept: its programs cannot be enqueued because of a framework
> out-of-bounds write that this factory's node-dependent dataflow-buffer sets provoke. The root cause
> and the reasoning for the revert are in `METAL2_PORT_REPORT.md` under Handoff points. **Its plan
> sections are kept below** because they are the recipe for re-porting it once that lands.

## Legacy Inventory

### Legacy factory shape

- Concept: `ProgramDescriptorFactoryConcept` for all three.
  - `SingleRowSingleCore` — `create_descriptor()` → `ProgramDescriptor` ([sort_program_factory.cpp:21](device/sort_program_factory.cpp#L21-L22))
  - `SingleRowMultiCore` — `create_descriptor()` → `ProgramDescriptor` ([sort_program_factory.cpp:969](device/sort_program_factory.cpp#L969-L970))
  - `CrossCoreDataExchange` — `create_workload_descriptor()` → `WorkloadDescriptor` ([sort_program_factory.cpp:902](device/sort_program_factory.cpp#L902-L906)), secretly SPMD: one `ProgramDescriptor` replicated over every `tensor_coords` range ([sort_program_factory.cpp:924-930](device/sort_program_factory.cpp#L924-L930)).
- Variants: three factories in one `program_factory_t` variant, selected at runtime by `Wt`
  ([sort_device_operation.cpp:16-51](device/sort_device_operation.cpp#L16-L51)). Each factory additionally has two
  *configurations* selected at build time by the input layout (`is_row_major`), which change CB allocation and
  kernel control flow. The configurations are **not** separate `KernelDescriptor`s — one source per role, gated
  internally by an `is_row_major` compile-time arg.
- Custom `compute_program_hash`: **none** — the device-operation class declares only `select_program_factory`,
  `validate_on_program_cache_miss`, `compute_output_specs`, `create_output_tensors`
  ([sort_device_operation.hpp:16-32](device/sort_device_operation.hpp#L16-L32)). Default reflection-based hash retained.

*(The Metal 2.0 factory concept the port targets was chosen during the audit — see the brief's TTNN factory analysis
section. Carried forward in the [TTNN ProgramFactory](#ttnn-programfactory) section below.)*

---

### Variant: `SortProgramFactorySingleRowSingleCore`

Work distribution: one tile-row (`Wt` tiles) per core, `Ht` rows total; each core loops `core_loop_count` times.
`core_range` is a rectangle sized to `min(Ht, total_cores)` cores, with a residual partial row merged in
([sort_program_factory.cpp:61-83](device/sort_program_factory.cpp#L61-L83)).

#### Kernels

| unique_id | source | core_ranges | CTAs (positional) | CTAs (named) | RTAs | CRTAs | defines | config |
|---|---|---|---|---|---|---|---|---|
| reader | `kernels/dataflow/reader_single_row_single_core.cpp` | `core_range` | 0 `input_tensor_cb_index`, 1 `index_tensor_output_cb_index`, 2 `Wt`, 3 `Ht`, 4 `total_number_of_cores`, 5 `grid_x`, 6 `grid_y`, 7 `is_row_major`, 8 `rm_input_cb_index`, 9 `rm_index_output_cb_index`, 10 `W_value_bytes`, 11 `W_index_bytes`, then `TensorAccessorArgs(input)`, `TensorAccessorArgs(index)` | none | `{input_buffer, index_buffer, core_loop_count}` | none | none | `ReaderConfigDescriptor{}` |
| writer | `kernels/dataflow/writer_single_row_single_core.cpp` | `core_range` | 0 `value_tensor_cb_index`, 1 `index_tensor_cb_index`, 2 `Wt`, 3 `Ht`, 4 `total_number_of_cores`, 5 `grid_x`, 6 `grid_y`, 7 `is_32_bit_data`, 8 `is_row_major`, 9 `rm_value_output_cb_index`, 10 `W_value_bytes`, then `TensorAccessorArgs(value)` | none | `{value_buffer, core_loop_count}` | none | none | `WriterConfigDescriptor{}` |
| compute | `kernels/compute/sort_single_row_single_core.cpp` | `core_range` | 0 `input_tensor_cb_index`, 1 `index_tensor_cb_index`, 2 `input_tensor_transposed_cb_index`, 3 `index_tensor_transposed_cb_index`, 4 `value_tensor_cb_index`, 5 `index_tensor_output_cb_index`, 6 `Wt`, 7 `descending`, 8 `stable`, 9 `synchronization_cb_index`, 10 `is_row_major`, 11 `rm_input_cb_index`, 12 `rm_value_output_cb_index`, 13 `rm_index_output_cb_index`, 14 `rm_post_sort_index_cb_index` | none | `{core_loop_count}` | none | none | `ComputeConfigDescriptor{.fp32_dest_acc_en = is_32_bit_data, .unpack_to_dest_mode = …}` |

#### CBs

| index | total_size | core_ranges | data_format | page_size | tile (if set) |
|---|---|---|---|---|---|
| c_0 `input_tensor` | `(is_row_major ? Wt : 4) * input_tile` | `core_range` | input | `input_tile` | (unset) |
| c_1 `index_tensor` | `(is_row_major ? Wt : 4) * index_tile` | `core_range` | index | `index_tile` | (unset) |
| c_2 `input_tensor_transposed` | `Wt * input_tile` | `core_range` | input | `input_tile` | (unset) |
| c_3 `index_tensor_transposed` | `Wt * index_tile` | `core_range` | index | `index_tile` | (unset) |
| c_4 `value_tensor` | `2 * value_tile` | `core_range` | value | `value_tile` | (unset) |
| c_5 `index_tensor_output` | `2 * index_tile` | `core_range` | index | `index_tile` | (unset) |
| c_6 `synchronization` | `TILE_HW * 1` (1024 B) | `core_range` | `UInt8` | 1024 | (unset) |
| c_7 `rm_input` (RM only) | `32 * W_value_bytes` | `core_range` | input | `W_value_bytes` | (unset) |
| c_8 `rm_value_output` (RM only) | `32 * W_value_bytes` | `core_range` | value | `W_value_bytes` | (unset) |
| c_9 `rm_index_output` (RM only) | `32 * W_index_bytes` | `core_range` | index | `W_index_bytes` | (unset) |
| c_10 `rm_post_sort_index` (RM only) | `Wt * index_tile` | `core_range` | index | `index_tile` | (unset) |

No `CBFormatDescriptor::tile` is set anywhere in this op, so no `tile_format_metadata` to carry.

#### Semaphores

none.

#### Tensor accessors

| host site (file:line) | originating Tensor | RTA slot (host) |
|---|---|---|
| reader `TensorAccessorArgs(*input_buffer)` ([:242](device/sort_program_factory.cpp#L242)) | input | reader RTA 0 |
| reader `TensorAccessorArgs(*index_buffer)` ([:243](device/sort_program_factory.cpp#L243)) | index output | reader RTA 1 |
| writer `TensorAccessorArgs(*value_buffer)` ([:266](device/sort_program_factory.cpp#L266)) | value output | writer RTA 0 |

#### Work split

- Driver: hand-rolled, not `split_work_to_cores` ([sort_program_factory.cpp:318-336](device/sort_program_factory.cpp#L318-L336)).
- `num_cores`: `core_range.num_cores()` = `min(Ht, total_cores)`, rounded up into a rectangle + residual row.
- Single kernel instance per role; per-core variation is carried **entirely by the `core_loop_count` RTA**
  (`Ht / total_cores`, plus 1 on the first `Ht % total_cores` cores). **No per-group CTA specialization.**

---

### Variant: `SortProgramFactoryCrossCoreDataExchange`

Work distribution: `Wt` tiles split into `number_of_tiles_per_core`-wide strips, one strip per core; cores exchange
tiles over NoC during the bitonic merge. Core 0 is the barrier leader (running the same reader kernel).

#### Kernels

| unique_id | source | core_ranges | CTAs (positional) | CTAs (named) | RTAs | CRTAs | defines | config |
|---|---|---|---|---|---|---|---|---|
| reader | `kernels/dataflow/reader_cross_core_data_exchange.cpp` | `core_range` | 0 `grid_x`, 1 `grid_y`, 2 `input_tensor_cb_index`, 3 `index_tensor_output_cb_index`, 4 `value_tensor_intermediate_cb_index`, 5 `index_tensor_intermediate_cb_index`, 6 `value_tensor_peer_cb_index`, 7 `index_tensor_peer_cb_index`, 8 `physical_core_lookup_table_cb_index`, 9 `Ht`, 10 `Wt`, 11 `number_of_tiles_per_core`, 12 `all_core_utilization_count`, 13 `!descending`, 14 `semaphore_exchange_readers`, 15 `semaphore_barrier`, 16 `is_row_major`, 17 `rm_input_cb_index`, 18 `rm_index_output_cb_index`, 19 `W_value_slice_bytes`, 20 `W_index_slice_bytes`, then `TensorAccessorArgs` ×3 (input, index, lookup) | none | `{input_buffer, index_buffer, lookup_buffer}` | none | none | `ReaderConfigDescriptor{}` |
| writer | `kernels/dataflow/writer_cross_core_data_exchange.cpp` | `core_range` | 0 `grid_x`, 1 `grid_y`, 2 `index_tensor_cb_index`, 3 `value_tensor_cb_index`, 4 `value_tensor_peer_cb_index` **(dead)**, 5 `physical_core_lookup_table_cb_index`, 6 `Wt`, 7 `Ht`, 8 `number_of_tiles_per_core`, 9 `total_number_of_cores_virtual` **(dead)**, 10 `semaphore_exchange_readers` **(dead)**, 11 `is_32_bit_data`, 12 `is_row_major`, 13 `rm_value_output_cb_index`, 14 `W_value_slice_bytes`, then `TensorAccessorArgs(value)` | none | `{value_buffer}` | none | none | `WriterConfigDescriptor{}` |
| compute | `kernels/compute/sort_cross_core_data_exchange.cpp` | `core_range` | 0 `grid_x`, 1 `grid_y`, 2 `Ht`, 3 `Wt`, 4 `number_of_tiles_per_core`, 5 `number_of_cores_used`, 6 `!descending`, 7 `input_tensor_cb_index`, 8 `index_tensor_cb_index`, 9 `input_tensor_transposed_cb_index`, 10 `index_tensor_transposed_cb_index`, 11 `value_tensor_cb_index`, 12 `index_tensor_output_cb_index`, 13 `value_tensor_intermediate_cb_index`, 14 `index_tensor_intermediate_cb_index`, 15 `value_tensor_peer_cb_index`, 16 `index_tensor_peer_cb_index`, 17 `packer_unpacker_sync_cb_index`, 18 `is_row_major`, 19 `rm_input_cb_index`, 20 `rm_value_output_cb_index`, 21 `rm_index_output_cb_index`, 22 `rm_post_sort_index_cb_index` | none | none | none | none | `ComputeConfigDescriptor{.fp32_dest_acc_en = is_32_bit_data, .unpack_to_dest_mode = …}` |

#### CBs

All on the single `core_range`. `S` = `cb_scale_factor` = 2, `N` = `number_of_tiles_per_core`.

| index | total_size | data_format | page_size |
|---|---|---|---|
| c_0 `input_tensor` | `(is_row_major ? N : S) * input_tile` | input | `input_tile` |
| c_1 `index_tensor` | `S * index_tile` | index | `index_tile` |
| c_2 `input_tensor_transposed` | `N * input_tile` | input | `input_tile` |
| c_3 `index_tensor_transposed` | `N * index_tile` | index | `index_tile` |
| c_4 `value_tensor` | `S * value_tile` | value | `value_tile` |
| c_5 `index_tensor_output` | `S * index_tile` | index | `index_tile` |
| c_6 `value_tensor_intermediate` | `S * value_tile` | value | **`index_tile`** (legacy mismatch, preserved verbatim) |
| c_7 `index_tensor_intermediate` | `S * index_tile` | index | `index_tile` |
| c_8 `value_tensor_peer` | `S * value_tile` | value | **`index_tile`** (legacy mismatch, preserved verbatim) |
| c_9 `index_tensor_peer` | `S * index_tile` | index | `index_tile` |
| c_10 `physical_core_lookup_table` | `lookup_tile` | `UInt32` | `lookup_tile` |
| c_11 `packer_unpacker_sync` | `sync_tile` | `Float16_b` | `sync_tile` |
| c_12 `rm_input` (RM only) | `32 * W_value_slice_bytes` | input | `W_value_slice_bytes` |
| c_13 `rm_value_output` (RM only) | `32 * W_value_slice_bytes` | value | `W_value_slice_bytes` |
| c_14 `rm_index_output` (RM only) | `32 * W_index_slice_bytes` | index | `W_index_slice_bytes` |
| c_15 `rm_post_sort_index` (RM only) | `N * index_tile` | index | `index_tile` |

#### Semaphores

| id | core_type | core_ranges | initial_value |
|---|---|---|---|
| 0 `semaphore_exchange_readers` | `WORKER` | `core_range` | 0 |
| 1 `semaphore_unused` | `WORKER` | `core_range` | 0 |
| 2 `semaphore_barrier` | `WORKER` | `core_range` | 0 |

Semaphore 1 is a placeholder: no kernel uses it. It exists only to keep the positional id numbering stable
([sort_program_factory.cpp:743-748](device/sort_program_factory.cpp#L743-L748)).

#### Tensor accessors

| host site (file:line) | originating Tensor | RTA slot (host) |
|---|---|---|
| reader `TensorAccessorArgs(*input_buffer)` ([:794](device/sort_program_factory.cpp#L794)) | input | reader RTA 0 |
| reader `TensorAccessorArgs(*index_buffer)` ([:795](device/sort_program_factory.cpp#L795)) | index output | reader RTA 1 |
| reader `TensorAccessorArgs(*physical_core_lookup_table_tensor_buffer)` ([:796](device/sort_program_factory.cpp#L796)) | **op-owned** lookup table | reader RTA 2 |
| writer `TensorAccessorArgs(*value_buffer)` ([:823](device/sort_program_factory.cpp#L823)) | value output | writer RTA 0 |

#### Op-owned tensors

One: the physical-core lookup table, built on cache miss by `build_physical_core_lookup_table_tensor()`
([sort_program_factory.cpp:475-497](device/sort_program_factory.cpp#L475-L497)) and parked on `wd.buffers` behind a
`shared_ptr<Tensor>` owner ([sort_program_factory.cpp:915-919](device/sort_program_factory.cpp#L915-L919)).

#### Work split

- Driver: `get_number_of_tiles_per_core()` + `compute_cross_core_range()`
  ([sort_program_factory.cpp:432-465](device/sort_program_factory.cpp#L432-L465)).
- `all_core_utilization_count` = `ceil(Wt / number_of_tiles_per_core)` cores.
- One kernel instance per role over the whole `core_range`; per-core variation is derived on-device from
  `get_absolute_logical_x/y()`. **No per-core RTAs beyond the buffer addresses, and no per-group CTAs.**

---

### Variant: `SortProgramFactorySingleRowMultiCore`

Work distribution: a dedicated **coordinator** node (`{grid_x-1, grid_y-1}`) stages input→output in DRAM and drives the
bitonic schedule by semaphore; the remaining `core_range` nodes are **workers** running reader/writer/compute and
sorting tile pairs in the DRAM output buffers.

#### Kernels

| unique_id | source | core_ranges | CTAs (positional) | CTAs (named) | RTAs | CRTAs | defines | config |
|---|---|---|---|---|---|---|---|---|
| coordinator | `kernels/dataflow/coordinator_single_row_multi_core.cpp` | `{coordinator_core}` | 0 `total_work_units`, 1 `Wt`, 2 `Ht`, 3 `total_number_of_cores`, 4 `number_of_available_cores`, 5 `input_tensor_cb_index`, 6 `index_tensor_cb_index`, 7 `is_32_bit_data`, then `TensorAccessorArgs` ×3 (input, value, index), then `is_row_major`, `rm_coord_value_row_cb_index`, `rm_coord_index_row_cb_index`, `W_tile_bytes`, `W_index_bytes`, `tile_width` | none | `{start_x, start_y, end_x, end_y, sem0, sem1, sem2, number_of_dest, input_buffer, value_buffer, index_buffer}` | none | none | `ReaderConfigDescriptor{}` |
| reader | `kernels/dataflow/reader_single_row_multi_core.cpp` | `core_range` (workers) | 0 `input_tensor_cb_index`, 1 `index_tensor_cb_index`, 2 `Wt`, 3 `Ht`, 4 `total_number_of_cores`, 5 `grid_x`, 6 `grid_y`, 7 `number_of_available_cores`, then `TensorAccessorArgs` ×2 (value, index), then `is_row_major`, `rm_worker_input_value_cb_index`, `rm_worker_input_index_cb_index`, `W_tile_bytes`, `W_index_bytes` | none | `{value_buffer, index_buffer, coord_x, coord_y, sem0, sem1}` | none | none | `ReaderConfigDescriptor{}` |
| writer | `kernels/dataflow/writer_single_row_multi_core.cpp` | `core_range` (workers) | 0 `input_tensor_output_cb_index`, 1 `index_tensor_output_cb_index`, 2 `Wt`, 3 `Ht`, 4 `total_number_of_cores`, 5 `grid_x`, 6 `grid_y`, 7 `number_of_available_cores`, then `TensorAccessorArgs` ×2 (value, index), then `is_row_major`, `rm_worker_output_value_cb_index`, `rm_worker_output_index_cb_index`, `W_tile_bytes`, `W_index_bytes` | none | `{value_buffer, index_buffer, coord_x, coord_y, sem0 (dead), sem2}` | none | none | `WriterConfigDescriptor{}` |
| compute | `kernels/compute/sort_single_row_multi_core.cpp` | `core_range` (workers) | 0 `input_tensor_cb_index`, 1 `index_tensor_cb_index`, 2 `input_tensor_transposed_cb_index`, 3 `index_tensor_transposed_cb_index`, 4 `input_tensor_output_cb_index`, 5 `index_tensor_output_cb_index`, 6 `Wt`, 7 `Ht`, 8 `number_of_available_cores`, 9 `grid_x`, 10 `grid_y`, 11 `descending`, 12 `stable`, 13 `log2Wt`, 14 `is_row_major`, 15 `rm_worker_input_value_cb_index`, 16 `rm_worker_input_index_cb_index`, 17 `rm_worker_output_value_cb_index`, 18 `rm_worker_output_index_cb_index` | none | none | none | none | `ComputeConfigDescriptor{.fp32_dest_acc_en = is_32_bit_data, .unpack_to_dest_mode = …}` |

**The workers read the value-output buffer as their input.** The coordinator first copies input→value-output in DRAM
and generates the index tensor there; workers then sort in place in those two output buffers. So the reader's
kernel-local `input_tensor_*` names are fed the *value output* buffer
([sort_program_factory.cpp:1316-1323](device/sort_program_factory.cpp#L1316-L1323)).

#### CBs

`B` = `buffer_scale_factor` = 2, `TILE_H` = 32.

| index | total_size | core_ranges | data_format | page_size |
|---|---|---|---|---|
| c_0 `input_tensor` | `B * input_tile` | `all_core_set` | input | `input_tile` |
| c_1 `index_tensor` | `B * index_tile` | `all_core_set` | index | `index_tile` |
| c_2 `input_tensor_transposed` | `B * input_tile` | `all_core_set` | input | `input_tile` |
| c_3 `index_tensor_transposed` | `B * index_tile` | `all_core_set` | index | `index_tile` |
| c_4 `input_tensor_output` | `B * value_tile` | `all_core_set` | value | `value_tile` |
| c_5 `index_tensor_output` | `B * index_tile` | `all_core_set` | index | `index_tile` |
| c_6 `rm_coord_value_row` (RM only) | `W_tile_bytes` | `{coordinator_core}` | input | `W_tile_bytes` |
| c_7 `rm_coord_index_row` (RM only) | `W_index_bytes` | `{coordinator_core}` | index | `W_index_bytes` |
| c_6 `rm_worker_input_value` (RM only) | `2 * TILE_H * W_tile_bytes` | `core_range` | input | `W_tile_bytes` |
| c_7 `rm_worker_input_index` (RM only) | `2 * TILE_H * W_index_bytes` | `core_range` | index | `W_index_bytes` |
| c_8 `rm_worker_output_value` (RM only) | `2 * TILE_H * W_tile_bytes` | `core_range` | input | `W_tile_bytes` |
| c_9 `rm_worker_output_index` (RM only) | `2 * TILE_H * W_index_bytes` | `core_range` | index | `W_index_bytes` |

c_6 and c_7 carry **two** `CBDescriptor`s each on **disjoint** node sets (coordinator vs workers).

#### Semaphores

| id | core_type | core_ranges | initial_value |
|---|---|---|---|
| 0 `coordinator_to_cores` | `WORKER` | `all_core_set` | 0 |
| 1 `cores_to_coordinator_ready` | `WORKER` | `all_core_set` | 0 |
| 2 `cores_to_coordinator_done` | `WORKER` | `all_core_set` | 0 |

#### Tensor accessors

| host site (file:line) | originating Tensor | RTA slot (host) |
|---|---|---|
| coordinator `TensorAccessorArgs(*input_buffer)` ([:1226](device/sort_program_factory.cpp#L1226)) | input | coordinator RTA 8 |
| coordinator `TensorAccessorArgs(*value_buffer)` ([:1227](device/sort_program_factory.cpp#L1227)) | value output | coordinator RTA 9 |
| coordinator `TensorAccessorArgs(*index_buffer)` ([:1228](device/sort_program_factory.cpp#L1228)) | index output | coordinator RTA 10 |
| reader `TensorAccessorArgs(*value_buffer)` ([:1267](device/sort_program_factory.cpp#L1267)) | value output | reader RTA 0 |
| reader `TensorAccessorArgs(*index_buffer)` ([:1268](device/sort_program_factory.cpp#L1268)) | index output | reader RTA 1 |
| writer `TensorAccessorArgs(*value_buffer)` ([:1294](device/sort_program_factory.cpp#L1294)) | value output | writer RTA 0 |
| writer `TensorAccessorArgs(*index_buffer)` ([:1295](device/sort_program_factory.cpp#L1295)) | index output | writer RTA 1 |

#### Work split

- Driver: hand-rolled ([sort_program_factory.cpp:1001-1039](device/sort_program_factory.cpp#L1001-L1039)).
- `total_work_units = Wt / 2`; `number_of_available_cores = total_cores - 1` (coordinator excluded).
- One kernel instance per role; per-core variation is derived on-device from `get_absolute_logical_x/y()`.
  **No per-group CTAs.**

---

### Cross-op kernels

none — all 10 kernel sources and all 3 shared headers live under `sort/device/kernels/`. Every `#include` resolves to
a framework `api/*` header or one of the op's own in-directory headers.

### Flags

- **No unreferenced kernel files** in the op directory.
- **Vestigial CTAs in the cross-core writer**: arg 4 `value_tensor_peer_cb_index` (declared, never used to build a
  `DataflowBuffer`), arg 9 `number_of_cores_used`, arg 10 (a semaphore id, now only a comment). Pre-existing.
- **Stale comment**: the cross-core writer labels arg 5 `physical_core_lookup_table_dfb_index` "unused - for future
  improvements", but it *is* used — [writer_cross_core_data_exchange.cpp:50](device/kernels/dataflow/writer_cross_core_data_exchange.cpp#L50)
  builds a `DataflowBuffer` from it and [:102](device/kernels/dataflow/writer_cross_core_data_exchange.cpp#L102)
  `push_back`s it. This stray push is what forces c_10's multi-binding.
- **Dead RTA**: the multi-core writer reads RTA 4 `coordinator_to_cores_semaphore_arg`
  ([writer_single_row_multi_core.cpp:17](device/kernels/dataflow/writer_single_row_multi_core.cpp#L17)) and never uses it.
- **`CircularBuffer` (not `DataflowBuffer`) in one kernel**: `coordinator_single_row_multi_core.cpp` is the only kernel
  still on the `CircularBuffer` wrapper, and the only user of the `use<AddrSelector::WRITE_PTR>` pointer-selection
  wrapper ([:96](device/kernels/dataflow/coordinator_single_row_multi_core.cpp#L96),
  [:133](device/kernels/dataflow/coordinator_single_row_multi_core.cpp#L133)).

## TTNN ProgramFactory

- **Concept (inherited from audit)**: `MetalV2FactoryConcept` for all three factories
  (`ProgramSpecFactoryConcept` in code — [operation_concepts.hpp:119](../../../../../../ttnn/api/ttnn/operation_concepts.hpp#L119-L121)).
- **Custom `compute_program_hash`**: none — already the default reflection-based hash. Nothing to delete.
- **Pybind**: no `create_descriptor` / `create_workload_descriptor` exposure. `sort_nanobind.cpp` binds only
  `ttnn::sort`, so no pybind line disappears.
- **Implementation notes**:
  - `CrossCoreDataExchange` drops `create_workload_descriptor` entirely. Its SPMD replication over
    `tensor_coords.ranges()` is exactly what the adapter now does for free, and its `WorkloadDescriptor::buffers`
    lifetime hack is replaced by `ProgramArtifacts::op_owned_tensors`. The `tensor_coords` parameter disappears from
    the signature.
  - The lookup-table tensor construction is kept verbatim; only the tail changes — instead of
    `make_shared<Tensor>` + `wd.buffers.push_back`, the owning `MeshTensor` is moved out with
    `release_mesh_tensor()` into `op_owned_tensors` and bound as an ordinary `TensorParameter`.

## Planned Spec Shape

Common to all three factories:

- **WorkUnitSpecs**: one per distinct (kernel set, node set) pairing.
- **TensorParameters**: one per distinct originating tensor; multiple kernels binding the same tensor collapse to one
  parameter with several `TensorBinding`s.
- **hw_config**: DM kernels use the arch-agnostic TTNN helpers (`create_reader_datamovement_config(arch)` /
  `create_writer_datamovement_config(arch)`) because every legacy DM config is a bare `ReaderConfigDescriptor{}` /
  `WriterConfigDescriptor{}`. Compute kernels are **Style B** (the op sets a Metal `ComputeConfigDescriptor` directly,
  with no TTNN `ComputeKernelConfig` anywhere), so a `ComputeGen1Config` is built by hand:
  `enable_32_bit_dest = is_32_bit_data` (from `fp32_dest_acc_en`) plus a translated `unpack_modes` table. Every other
  field stays at its Metal 2.0 default, which matches the legacy `ComputeConfigDescriptor` default exactly
  (`math_fidelity` HiFi4, `math_approx_mode` false → `Precision::Precise`, `dst_full_sync_en` false →
  `double_buffer_dest` true, `bfp8_pack_precise` false → `Precision::Approximate`).
- **`unpack_modes`**: the legacy `std::vector<UnpackToDestMode>` (indexed by CB id, filled only when the input format
  is `Float32`) is re-keyed to `Table<DFBSpecName, UnpackMode>`, `UnpackToDestFp32` → `UnpackMode::UnpackToDest`;
  `Default` entries are expressed by omission. **Entries are emitted only for DFBs bound in the active configuration**
  — see [Deferred / Flagged](#deferred--flagged).

### Variant: `SingleRowSingleCore`

- **KernelSpecs**: 3 — `READER`, `WRITER`, `COMPUTE` (1:1 with legacy).
- **DataflowBufferSpecs**: 7 in TILE (`INPUT_TENSOR`, `INDEX_TENSOR`, `INPUT_TRANSPOSED`, `INDEX_TRANSPOSED`,
  `VALUE_TENSOR`, `INDEX_OUTPUT`, `SYNCHRONIZATION`); 9 in ROW_MAJOR (drop `VALUE_TENSOR` + `INDEX_OUTPUT`, add
  `RM_INPUT`, `RM_VALUE_OUTPUT`, `RM_INDEX_OUTPUT`, `RM_POST_SORT_INDEX`).
- **SemaphoreSpecs**: none.
- **TensorParameters**: 3 — `INPUT`, `VALUE_OUTPUT`, `INDEX_OUTPUT`.
- **WorkUnitSpecs**: 1 — `{READER, WRITER, COMPUTE}` on `core_range`.
- **Op-owned tensors**: none.

DFB endpoint dispositions (re-derived from the kernel-touch census, not transcribed):

| DFB | TILE | ROW_MAJOR |
|---|---|---|
| `INPUT_TENSOR` (c_0) | reader P → compute C | **self-loop** on compute (tilize fills, sort drains, untilize refills/drains) |
| `INDEX_TENSOR` (c_1) | writer P → compute C | writer P → compute C |
| `INPUT_TRANSPOSED` (c_2) | **self-loop** compute | **self-loop** compute |
| `INDEX_TRANSPOSED` (c_3) | **self-loop** compute | **self-loop** compute |
| `VALUE_TENSOR` (c_4) | compute P → writer C | **dead — no spec** |
| `INDEX_OUTPUT` (c_5) | compute P → reader C | **dead — no spec** |
| `SYNCHRONIZATION` (c_6) | **self-loop** compute | **self-loop** compute |
| `RM_INPUT` (c_7) | n/a | reader P → compute C |
| `RM_VALUE_OUTPUT` (c_8) | n/a | compute P → writer C |
| `RM_INDEX_OUTPUT` (c_9) | n/a | compute P → reader C |
| `RM_POST_SORT_INDEX` (c_10) | n/a | **self-loop** compute |

### Variant: `CrossCoreDataExchange`

- **KernelSpecs**: 3 — `READER`, `WRITER`, `COMPUTE`.
- **DataflowBufferSpecs**: 12 in TILE (c_0–c_11); 14 in ROW_MAJOR (drop `VALUE_TENSOR` + `INDEX_OUTPUT`, add the four
  RM buffers).
- **SemaphoreSpecs**: 3 — `SEM_EXCHANGE`, `SEM_UNUSED`, `SEM_BARRIER`, all with `target_nodes = core_range`.
  `SEM_UNUSED` is kept as an inert spec (no kernel binds it; the validator does not require a semaphore binding), so
  the program's semaphore footprint is unchanged.
- **TensorParameters**: 4 — `INPUT`, `VALUE_OUTPUT`, `INDEX_OUTPUT`, `LOOKUP_TABLE` (op-owned).
- **WorkUnitSpecs**: 1 — `{READER, WRITER, COMPUTE}` on `core_range`.
- **Op-owned tensors**: 1 — the physical-core lookup table.

DFB endpoint dispositions:

| DFB | TILE | ROW_MAJOR |
|---|---|---|
| `INPUT_TENSOR` (c_0) | reader P → compute C | **self-loop** compute |
| `INDEX_TENSOR` (c_1) | writer P → compute C | writer P → compute C |
| `INPUT_TRANSPOSED` (c_2) / `INDEX_TRANSPOSED` (c_3) | **self-loop** compute | **self-loop** compute |
| `VALUE_TENSOR` (c_4) | compute P → writer C | **dead — no spec** |
| `INDEX_OUTPUT` (c_5) | compute P → reader C | **dead — no spec** |
| `VALUE_INTERMEDIATE` (c_6) / `INDEX_INTERMEDIATE` (c_7) | compute P → reader C | compute P → reader C |
| `VALUE_PEER` (c_8) / `INDEX_PEER` (c_9) | reader P → compute C | reader P → compute C |
| `LOOKUP_TABLE` (c_10) | **1P+1C**: reader P, writer C (see below) | same |
| `PACKER_UNPACKER_SYNC` (c_11) | **self-loop** compute | **self-loop** compute |
| `RM_INPUT` (c_12) | n/a | reader P → compute C |
| `RM_VALUE_OUTPUT` (c_13) | n/a | compute P → writer C |
| `RM_INDEX_OUTPUT` (c_14) | n/a | compute P → reader C |
| `RM_POST_SORT_INDEX` (c_15) | n/a | **self-loop** compute |

**c_10 census (re-derived, and revised during construction).** Two distinct touchers on every node:
- reader — `reserve_back` ([reader_cross_core_data_exchange.cpp:75](device/kernels/dataflow/reader_cross_core_data_exchange.cpp#L75)) and
  `push_back` ([:216](device/kernels/dataflow/reader_cross_core_data_exchange.cpp#L216)) ⇒ **locked producer**; plus
  role-free `get_read_ptr` peeks inside `get_core_physical_coordinates`.
- writer — a lone `push_back` with no matching `reserve_back`
  ([writer_cross_core_data_exchange.cpp:102](device/kernels/dataflow/writer_cross_core_data_exchange.cpp#L102)) ⇒ a
  second **locked producer**.

Two kernels locked to the *same* FIFO role, and **no** locked consumer at all (nothing ever `wait_front`s or
`pop_front`s c_10). Reading the catalog's census table literally, that says multi-binding, which is also what the
brief instructed.

**That assignment is not expressible, and the port does not use it.** The only way to also satisfy "≥1 producer and
≥1 consumer" with zero consumers is to give one toucher both roles, and the validator then rejects the result:
"When a DFB is self-looped, every same-side binding must come from a self-loop participant". The port therefore
follows the catalog's stacking guard instead: recount, assign **1P+1C** (reader PRODUCER, writer CONSUMER), no flag.
Behaviour is identical on Gen1 — the buffer is a reader-private lookup window and the role labels drive no machinery
either kernel invokes. The Gen2 debt is recorded in `METAL2_PORT_REPORT.md` instead of in the flag.

### Variant: `SingleRowMultiCore`

- **KernelSpecs**: 4 — `COORDINATOR`, `READER`, `WRITER`, `COMPUTE`.
- **DataflowBufferSpecs**: 8 in TILE, 12 in ROW_MAJOR. **The legacy shared indices c_0/c_1 (TILE) and c_6/c_7 (RM)
  become separate coordinator-scoped and worker-scoped specs** — see below.
- **SemaphoreSpecs**: 3 — `SEM_COORD_TO_CORES`, `SEM_CORES_TO_COORD_READY`, `SEM_CORES_TO_COORD_DONE`, all with
  `target_nodes = all_core_set`.
- **TensorParameters**: 3 — `INPUT`, `VALUE_OUTPUT`, `INDEX_OUTPUT`.
- **WorkUnitSpecs**: 2 — `WU_COORDINATOR` = `{COORDINATOR}` on `{coordinator_core}`, `WU_WORKERS` =
  `{READER, WRITER, COMPUTE}` on `core_range`.
- **Op-owned tensors**: none.

**Why the coordinator's buffers become their own DFB specs.** Legacy scopes c_0–c_5 to `all_core_set` but the
coordinator kernel only ever touches c_0/c_1 (TILE) or c_6/c_7 (RM). In Metal 2.0 a DFB's placement is *derived from
its bindings*, so the "narrow the core range" action the brief asks for is expressed by simply not binding the
coordinator to the worker buffers. For c_0/c_1 in TILE that is not merely a tidy-up but **required**: a single DFB
spec bound by the coordinator (a DM kernel, self-loop) and by the workers (reader DM producer, *compute* consumer)
would put a DM kernel and a compute kernel on the same consumer endpoint, which the validator rejects — "All
KernelSpecs bound to the same DFB role must be of the same kind"
([program_spec.cpp:1289-1295](../../../../../../tt_metal/impl/metal2_host_api/program_spec.cpp#L1289-L1295)).
Splitting is also exactly how the brief already handles c_6/c_7, whose two legacy `CBDescriptor`s sit on disjoint node
sets. Behaviour is unchanged: each node allocates precisely the buffers its kernels touch.

DFB endpoint dispositions:

*Coordinator node:*

| DFB | TILE | ROW_MAJOR |
|---|---|---|
| `COORD_INPUT` (legacy c_0) | **self-loop** coordinator | not allocated |
| `COORD_INDEX` (legacy c_1) | **self-loop** coordinator | not allocated |
| `COORD_VALUE_ROW` (legacy c_6) | not allocated | **self-loop** coordinator |
| `COORD_INDEX_ROW` (legacy c_7) | not allocated | **self-loop** coordinator |

*Worker nodes:*

| DFB | TILE | ROW_MAJOR |
|---|---|---|
| `WORKER_INPUT` (c_0) | reader P → compute C | **self-loop** compute (tilize from `WORKER_RM_IN_VALUE`) |
| `WORKER_INDEX` (c_1) | reader P → compute C | **self-loop** compute (tilize from `WORKER_RM_IN_INDEX`) |
| `WORKER_INPUT_TRANSPOSED` (c_2) | **self-loop** compute | **self-loop** compute |
| `WORKER_INDEX_TRANSPOSED` (c_3) | **self-loop** compute | **self-loop** compute |
| `WORKER_VALUE_OUTPUT` (c_4) | compute P → writer C | **self-loop** compute (packs, then untilizes out) |
| `WORKER_INDEX_OUTPUT` (c_5) | compute P → writer C | **self-loop** compute |
| `WORKER_RM_IN_VALUE` (c_6) | n/a | reader P → compute C |
| `WORKER_RM_IN_INDEX` (c_7) | n/a | reader P → compute C |
| `WORKER_RM_OUT_VALUE` (c_8) | n/a | compute P → writer C |
| `WORKER_RM_OUT_INDEX` (c_9) | n/a | compute P → writer C |

## Preserved Multiplicity

**none — no work-split multiplicity in legacy.** Every kernel role has exactly one `KernelDescriptor` in every
factory; per-core variation is carried by runtime args (`core_loop_count`) or derived on-device from
`get_absolute_logical_x/y()`. No per-group compile-time-arg specialization exists, so there is nothing to preserve and
no risk of the [CTA→RTA demotion anti-pattern](../../../../../../docs/source/tt-metalium/tt_metal/apis/host_apis/metal_2.0/ai/shared/port_patterns.md).

## Dropped Plumbing

### `SingleRowSingleCore`

| legacy location (file:line) | legacy form | Metal 2.0 replacement |
|---|---|---|
| reader RTA 0 / writer RTA — | `input_buffer`, `index_buffer`, `value_buffer` as `Buffer*` runtime args ([:332-333](device/sort_program_factory.cpp#L332-L333)) | `TensorBinding` → `TensorAccessor(tensor::…)` |
| reader CTA 0, 1, 8, 9 | `input_tensor_cb_index`, `index_tensor_output_cb_index`, `rm_input_cb_index`, `rm_index_output_cb_index` | `DFBBinding` |
| writer CTA 0, 1, 9 | `value_tensor_cb_index`, `index_tensor_cb_index`, `rm_value_output_cb_index` | `DFBBinding` |
| compute CTA 0–5, 9, 11–14 | all CB-index CTAs | `DFBBinding` |
| reader CTA 12+, writer CTA 11+ | `TensorAccessorArgs(buffer).append_to(cta)` + kernel-side `TensorAccessorArgs<N>()` chain | binding mechanism end-to-end |
| reader CTA 7, writer CTA 8, compute CTA 10 | `is_row_major` | `KernelSpec::compiler_options.defines` → `#ifdef IS_ROW_MAJOR` |
| all remaining positional CTAs | positional `compile_time_args` | named CTAs (`Wt`, `Ht`, `total_number_of_cores`, `compute_with_storage_grid_size_x/y`, `is_32_bit_data`, `W_value_bytes`, `W_index_bytes`, `descending`, `stable`) |

### `CrossCoreDataExchange`

| legacy location (file:line) | legacy form | Metal 2.0 replacement |
|---|---|---|
| reader RTAs 0–2, writer RTA 0 | `Buffer*` runtime args ([:837-839](device/sort_program_factory.cpp#L837-L839)) | `TensorBinding` (incl. the op-owned lookup table) |
| reader CTA 2–8, 17, 18 | CB-index CTAs | `DFBBinding` |
| writer CTA 2, 3, 5, 13 | CB-index CTAs | `DFBBinding` |
| writer CTA 4 `value_tensor_peer_cb_index` | CB-index CTA for a CB the writer never touches | **dropped** — no binding, no named arg (a CB index cannot survive as a CTA, and binding it would turn c_8 into a spurious three-toucher) |
| compute CTA 7–17, 19–22 | CB-index CTAs | `DFBBinding` |
| reader CTA 14, 15 | `semaphore_exchange_readers`, `semaphore_barrier` ids | `SemaphoreBinding` |
| writer CTA 10 | `semaphore_exchange_readers` id, read by nothing | **dropped** (positional slot alignment is meaningless once args are named) |
| reader CTA 21+, writer CTA 15+ | `TensorAccessorArgs` chains | binding mechanism |
| reader CTA 16, writer CTA 12, compute CTA 18 | `is_row_major` | `#ifdef IS_ROW_MAJOR` |
| all remaining positional CTAs | positional | named CTAs |

Writer CTA 9 `number_of_cores_used` is unread but carries no CB index or semaphore id, so it survives as a named CTA
with its "unused - for future improvements" comment intact.

### `SingleRowMultiCore`

| legacy location (file:line) | legacy form | Metal 2.0 replacement |
|---|---|---|
| coordinator RTAs 8–10, reader RTAs 0–1, writer RTAs 0–1 | `Buffer*` runtime args | `TensorBinding` |
| coordinator RTAs 4–6, reader RTAs 4–5, writer RTA 5 | semaphore ids as runtime args | `SemaphoreBinding` |
| writer RTA 4 | `coordinator_to_cores_semaphore_arg`, read into an unused local | **dropped** — the writer binds only the done semaphore |
| coordinator CTA 5, 6, and the two `rm_coord_*` CTAs | CB-index CTAs | `DFBBinding` |
| reader CTA 0, 1, and the two `rm_worker_input_*` CTAs | CB-index CTAs | `DFBBinding` |
| writer CTA 0, 1, and the two `rm_worker_output_*` CTAs | CB-index CTAs | `DFBBinding` |
| compute CTA 0–5, 15–18 | CB-index CTAs | `DFBBinding` |
| all three `TensorAccessorArgs` chains | host `append_to` + kernel `TensorAccessorArgs<N>()` | binding mechanism |
| coordinator / reader / writer / compute `is_row_major` CTA | compile-time gate | `#ifdef IS_ROW_MAJOR` |
| all remaining positional CTAs | positional | named CTAs |

## Applied Patterns

- [**Self-loop DFB binding**](../../../../../../docs/source/tt-metalium/tt_metal/apis/host_apis/metal_2.0/ai/shared/port_patterns.md#pattern-self-loop-dfb-binding)
  — the transposed / synchronization / post-sort-index buffers, and (in ROW_MAJOR) the tilize-and-untilize staging
  buffers, are produced and consumed by the one compute kernel. Also the multi-core **coordinator's** own staging
  buffers, which it both fills from DRAM and drains back — a **DM self-loop**, legal on Gen1.
- [**Sync-free and single-ended CBs → self-loop DFB**](../../../../../../docs/source/tt-metalium/tt_metal/apis/host_apis/metal_2.0/ai/shared/port_patterns.md#pattern-sync-free-and-single-ended-cbs--self-loop-dfb)
  — `SYNCHRONIZATION` (c_6, single-core) and `PACKER_UNPACKER_SYNC` (c_11, cross-core) are pure packer/unpacker
  handshake buffers with one toucher.
- [**Conditional / optional DFB bindings**](../../../../../../docs/source/tt-metalium/tt_metal/apis/host_apis/metal_2.0/ai/shared/port_patterns.md#pattern-conditional--optional-dfb-bindings)
  — the whole ROW_MAJOR / TILE split. Every RM-only DFB is bound only in the ROW_MAJOR configuration, `VALUE_TENSOR`
  and `INDEX_OUTPUT` only in TILE, and the legacy `is_row_major` **CTA gate is promoted to a preprocessor define**
  (`IS_ROW_MAJOR`) so `if constexpr` never name-looks-up an unbound `dfb::` token. This applies to all 10 kernels.
- [**Multi-variant factories**](../../../../../../docs/source/tt-metalium/tt_metal/apis/host_apis/metal_2.0/ai/shared/port_patterns.md#pattern-multi-variant-factories)
  — each `create_program_artifacts` branches internally on `is_row_major`.
- [**Pass DFB handles directly to LLKs and kernel-lib helpers**](../../../../../../docs/source/tt-metalium/tt_metal/apis/host_apis/metal_2.0/ai/shared/port_patterns.md#pattern-pass-dfb-handles-directly-to-llks-and-kernel-lib-helpers)
  — `dfb::name` flows straight into `compute_kernel_hw_startup`, `tilize_init`, `pack_untilize_init<>`,
  `binary_op_init_common`, `transpose_init`, `copy_tile`, `pack_tile`, and the op's own `generate_index_tile` helper.
- **Multi-binding advanced option** — **not used anywhere in the op.** The single candidate the brief named,
  `CrossCoreDataExchange`'s `LOOKUP_TABLE` (c_10), resolves to a plain 1P+1C assignment instead; see the c_10 census
  above.

## Deferred / Flagged

- **`unpack_modes` entries must be gated on the active configuration.** The legacy `unpack_to_dest_mode` vector is
  sized `NUM_CIRCULAR_BUFFERS` and indexed by CB id, so it happily carries an entry for a CB that the current
  configuration never allocates — `SingleRowMultiCore` sets index 6 (`rm_worker_input_value`) unconditionally
  ([sort_program_factory.cpp:1361](device/sort_program_factory.cpp#L1361)) even though c_6 exists only in ROW_MAJOR,
  and both `SingleRowSingleCore` and `CrossCoreDataExchange` set the entry for `value_tensor` (c_4), which is dead in
  ROW_MAJOR. Metal 2.0's validator rejects an `unpack_modes` key naming a DFB the kernel does not bind, so those
  entries are emitted only where the DFB is live. Behaviour-neutral: a legacy entry for an unallocated CB was inert.
- **The `use<AddrSelector::WRITE_PTR>` wrapper has no bare-DFB equivalent.** The recipe says the pointer-selection
  wrapper "drops, because a bare `DataflowBuffer` used as a NoC source/destination is already pointer-sourced" — but
  that holds only for `READ_PTR`. As a NoC **source** a bare DFB resolves to `get_read_ptr()`
  ([dataflow_buffer.h:387](../../../../../../tt_metal/hw/inc/api/dataflow/dataflow_buffer.h#L387)), so dropping the
  wrapper at the coordinator's two `WRITE_PTR` sites would silently change which slot is transmitted. The port keeps
  the exact address by feeding the whitelist-sanctioned public peek into a `CoreLocalMem<uint32_t>` source, the same
  idiom the cross-core exchange already uses. Recorded for the port report.
- **`LOOKUP_TABLE` (c_10) endpoint assignment, resolved during construction.** Planning read the census as
  multi-binding (matching the brief); the validator rejects the only binding shape that would express it, and the
  catalog's stacking guard prescribes the assignment the port actually uses. Written up in
  `METAL2_PORT_REPORT.md` as a catalog gap.
- **New findings after planning**, both discovered during construction and verification:
  - The `use<AddrSelector::WRITE_PTR>` wrapper does not simply drop; a bare `DataflowBuffer` is read-cursor-sourced.
    (Moot for the shipped factories, since the only two sites are in the reverted multi-core coordinator, but it is
    a live trap for the next porter.)
  - A kernel cannot query tile metadata off a buffer it does not bind in the active configuration; the cross-core
    reader needs two such sizes and receives them as named compile-time args.
  - `unpack_modes` entries must be gated on the same condition as their binding.
  - **The blocker**: node-dependent dataflow-buffer sets are serialized out of bounds by the dispatch layer. This is
    what stopped `SingleRowMultiCore`. Details in `METAL2_PORT_REPORT.md`.
