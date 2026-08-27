# Port Plan — `experimental/paged_cache`

Port plan for `ttnn/cpp/ttnn/operations/experimental/paged_cache`, ported from the
`ProgramDescriptorFactoryConcept` (`create_descriptor`) API to Metal 2.0.
Written during the inventory and planning steps; committed alongside the port for review.

---

## Scope of this pass

The audit brief scopes the op as **three `DeviceOperation`s, eight factories, four program bodies**.
Planning found a structural blocker the audit did not catch, which splits those eight factories in two
(full detail in [Deferred / Flagged](#deferred--flagged) and in `METAL2_PORT_REPORT.md`):

| Factory | Selected when | This pass |
|---|---|---|
| `PagedUpdateCacheProgramFactory` | `mesh_coords == nullopt` | **PORTED** |
| `PagedFillCacheProgramFactory` | `mesh_coords == nullopt` | **PORTED** |
| `PagedTiledFusedUpdateCacheProgramFactory` | `mesh_coords == nullopt` | deferred — audit Question #1 unresolved |
| `PagedRowMajorFusedUpdateCacheProgramFactory` | `mesh_coords == nullopt` | deferred — audit Question #1 unresolved |
| `PagedUpdateCacheMeshWorkloadFactory` | `mesh_coords.has_value()` | **BLOCKED** — per-coord variation |
| `PagedFillCacheMeshWorkloadFactory` | `mesh_coords.has_value()` | **BLOCKED** — per-coord variation |
| `PagedTiledFusedUpdateCacheMeshWorkloadFactory` | `mesh_coords.has_value()` | **BLOCKED** — per-coord variation |
| `PagedRowMajorFusedUpdateCacheMeshWorkloadFactory` | `mesh_coords.has_value()` | **BLOCKED** — per-coord variation |

Because each `*MeshWorkloadFactory` binds the same kernel sources as its single-device sibling and
cannot convert with it, the two ported factories take the **intra-op fork** rung of
*Caution: Porting a shared kernel*: five `_metal2` kernel forks land beside the originals, and the
originals stay untouched (apart from the mandated pointer comment) serving the four blocked
mesh factories.

---

## Legacy Inventory

### Legacy factory shape

- Concept: `ProgramDescriptorFactoryConcept` — all eight factories declare
  `create_descriptor(...) -> ProgramDescriptor`. The four `*MeshWorkloadFactory` types take an
  extra `mesh_dispatch_coordinate` parameter but still return a `ProgramDescriptor`; there is no
  `create_workload_descriptor` in this directory.
- Factory methods live in dedicated factory structs (`program_factory_t` variants), **not** directly
  on the device-operation struct — so [exception 3](#ttnn-programfactory) (direct-descriptor shape)
  does not apply.
- Variants: `PagedUpdateCacheDeviceOperation` →
  `std::variant<PagedUpdateCacheProgramFactory, PagedUpdateCacheMeshWorkloadFactory>`;
  `PagedFillCacheDeviceOperation` → the analogous pair;
  `PagedFusedUpdateCacheDeviceOperation` → four alternatives (tiled/RM × single/mesh).
- Custom `compute_program_hash`: **present on all three DeviceOperations** — left intact.
  - `device/update_cache/paged_update_cache_device_operation.cpp:313`
  - `device/fill_cache/paged_fill_cache_device_operation.cpp:207`
  - `device/fused_update_cache/paged_fused_update_cache_device_operation.cpp:371`

*(The Metal 2.0 factory concept the port targets was chosen during the audit — see the brief's
TTNN factory analysis section. Carried forward in [TTNN ProgramFactory](#ttnn-programfactory) below.)*

---

### Variant: `PagedUpdateCacheProgramFactory` (`paged_update_cache_program_factory.cpp`)

`all_cores` = the input tensor's shard grid; one "user" (batch row) per core.

#### Kernels

| unique_id | source | core_ranges | CTAs (positional) | CTAs (named) | RTAs | CRTAs | defines | opt_level | config |
|---|---|---|---|---|---|---|---|---|---|
| reader | `device/kernels/dataflow/reader_update_cache_interleaved_start_id.cpp` | `all_cores` | 0 `src0_cb_index`, 1 `src1_cb_index`, 2 `use_index_tensor`, 3 `cb_index_id`, 4 `cache_batch_num_tiles`, 5 `Wt`, 6 `log2_page_size`(=0), 7 `index_stick_size`, 8 `is_paged_cache`, 9 `num_heads`, 10 `block_size`, 11 `block_size_t`, 12 `max_blocks_per_seq`, 13 `log2_page_table_stick_size`(=0), 14 `page_table_stick_size`, 15 `cb_pagetable_id`, 16 `St`, 17 `in0_sequential_mode_semaphore_id`, 18 `cache_position_modulo`, then `TensorAccessorArgs` ×3 (cache, update_idxs, page_table) | none | per core: `[0]`=`Buffer*` cache, `[1]`=`cache_start_id`, `[2]`=`Buffer*` update_idxs \| `0`, `[3]`=`i` (batch idx), `[4]`=`Buffer*` page_table \| `0`, `[5]`=`wait_to_start` | none | none | O2 (absent field, DM) | `ReaderConfigDescriptor{}` |
| writer | `device/kernels/dataflow/writer_update_cache_interleaved_start_id.cpp` | `all_cores` | 0 `output_cb_index`, 1 `intermed0_cb_index`, 2 `intermed1_cb_index`, 3 `intermed2_cb_index`, 4 `use_index_tensor`, 5 `cb_index_id`, 6 `cache_batch_num_tiles`, 7 `Wt`, 8 `Wbytes`, 9 `is_paged_cache`, 10 `num_heads`, 11 `block_size`, 12 `block_size_t`, 13 `max_blocks_per_seq`, 14 `cb_pagetable_id`, 15 `St`, 16 `in0_sequential_mode_semaphore_id`, 17 `cache_position_modulo`, then `TensorAccessorArgs` ×1 (cache) | none | per core: `[0]`=`Buffer*` cache, `[1]`=`cache_start_id`, `[2]`=`tile_update_offset_B`, `[3]`=`i`, `[4]`=`send_signal`, `[5]`=`send_core_x`, `[6]`=`send_core_y` | none | none | O2 (absent field, DM) | `WriterConfigDescriptor{}` |
| compute | `device/kernels/compute/update_cache.cpp` | `all_cores` | 0 `src0_cb_index`, 1 `src1_cb_index`, 2 `intermed0_cb_index`, 3 `intermed1_cb_index`, 4 `intermed2_cb_index`, 5 `output_cb_index`, 6 `Wt`, 7 `num_heads` | none | none | none | none | **O3** (absent field on a `ComputeConfigDescriptor` resolves to O3) | `ComputeConfigDescriptor{.fp32_dest_acc_en = fp32_dest_acc_en}` |

`grep -n opt_level` over the factory returns **nothing** — no kernel sets an explicit level, so the
resolved levels are the per-kernel-type legacy defaults recorded above.

The counter-intuitive CTA names the brief warns about are present: the writer's CTA 0 named
`cache_cb_id` kernel-side is the **output** CB `c_16`, not the cache CB `c_0`. Every index below was
resolved through the factory's argument list, not the kernel's local name.

#### CBs

| index | total_size | core_ranges | data_format | page_size | tile (if set) | notes |
|---|---|---|---|---|---|---|
| `c_0` `src0` | `num_cache_tiles * cache_single_tile_size` (`num_cache_tiles = 2*Wt`) | `all_cores` | `cache_cb_data_format` | `cache_single_tile_size` | — | |
| `c_1` `src1` | `num_input_tiles * input_single_tile_size` | `all_cores` | `input_cb_data_format` | `input_single_tile_size` | — | **borrowed**: `.buffer = in1_buffer` (`:208`); re-pointed on cache hit at `:518` |
| `c_24` `intermed0` | `num_interm_tiles * interm_single_tile_size` (`num_interm_tiles = 2*Wt`) | `all_cores` | `interm_cb_data_format` | `interm_single_tile_size` | — | **aliased** — same `CBDescriptor` as `c_25` (`:210-225`) |
| `c_25` `intermed1` | *(shares the `c_24` descriptor)* | `all_cores` | `interm_cb_data_format` | `interm_single_tile_size` | — | **aliased** with `c_24` |
| `c_26` `intermed2` | `num_interm_tiles * interm_single_tile_size` | `all_cores` | `interm_cb_data_format` | `interm_single_tile_size` | — | |
| `c_16` `output` | `num_output_tiles * cache_single_tile_size` (`num_output_tiles = B*Wt`) | `all_cores` | `cache_cb_data_format` | `cache_single_tile_size` | — | |
| `c_2` `cb_index` | `index_tensor_tile_size` | `all_cores` | `index_data_format` | `index_tensor_tile_size` | — | allocated only `if (use_index_tensor)` (`:254-264`) |
| `c_3` `cb_pagetable` | `page_table_stick_size` | `all_cores` | `page_table_data_format` | `page_table_stick_size` | — | allocated only `if (is_paged_cache)` (`:266-276`) |

No `GlobalCircularBuffer` anywhere in this factory (no `.global_circular_buffer`, no `global_cb`
parameter, no `remote_cb_config`).

Endpoint census (re-derived from the kernels, agrees with the audit): **all eight are 1P+1C.**
No dead CB, no self-loop, no multi-binding.

| CB | producer | consumer |
|---|---|---|
| `c_0` cache | reader (`reader:132,143`) | compute (`untilize<Wt, cache_cb, untilized_cache_cb>`, `compute:48`) |
| `c_1` input | reader (`reader:60-61`) | compute (`untilize<Wt, in_cb, untilized_in_cb>`, `compute:39-45`) |
| `c_24` intermed0 | compute (`compute:48` output) | writer (`writer:122,134`) |
| `c_25` intermed1 | writer (`writer:123,133`) | compute (`tilize<Wt, untilized_cache2_cb, out_cb>`, `compute:51`) |
| `c_26` intermed2 | compute (`compute:39-45` output) | writer (`writer:113,160`) |
| `c_16` output | compute (`compute:51` output) | writer (`writer:137,148`) |
| `c_2` index | reader (`reader:76,81`) | writer (`writer:72,110`) |
| `c_3` pagetable | reader (`reader:96,105`) | writer (`writer:87,100`) |

#### Semaphores

| id | core_type | core_ranges | initial_value |
|---|---|---|---|
| `in0_sequential_mode_semaphore_id` (= 0) | `WORKER` | `all_cores` | 0 |

Used for the `share_cache` chain: writer *i* signals reader *i+1*
(`writer:164` `Semaphore<>(id).up(noc, send_core_x, send_core_y, 1)` + `noc.async_atomic_barrier()`
at `:165`; awaited at `reader:126-128`). `send_core_x/y` are **physical** coordinates baked host-side
via `worker_core_from_logical_core` (`:394`).

#### Tensor accessors

| host site (file:line) | originating Tensor | RTA slot (host) |
|---|---|---|
| `:308` `TensorAccessorArgs(dst_buffer)` → reader CTAs | `cache_tensor` | reader RTA[0] (`:406`) |
| `:309` `TensorAccessorArgs(update_idxs->buffer())` → reader CTAs | `update_idxs_tensor` (optional) | reader RTA[2] (`:409`) |
| `:311` `TensorAccessorArgs(page_table->buffer())` → reader CTAs | `page_table` (optional) | reader RTA[4] (`:415`) |
| `:335` `TensorAccessorArgs(dst_buffer)` → writer CTAs | `cache_tensor` | writer RTA[0] (`:426`) |
| (no accessor) | `input_tensor` | none — reaches the kernel as the **borrowed** CB `c_1` |

All four accessor constructions are the **2-arg** form; no page-size third argument to drop.
All are **Case 1** (consumed through `TensorAccessor`), so no `get_bank_base_address` bridge is needed.

#### Work split

- Driver: **not** `split_work_to_cores`. `all_cores = input_tensor.shard_spec()->grid`; the per-core
  list is `corerange_to_cores(grid, grid.num_cores(), orientation == ROW_MAJOR)`
  (`update_cache_cores`, `:37-41`). One core per batch row.
- num_cores: `all_cores.num_cores()`
- Single group — every core gets the same CTAs and a per-core RTA set. No multi-`KernelDescriptor`
  work split.

---

### Variant: `PagedFillCacheProgramFactory` (`paged_fill_cache_program_factory.cpp`)

No compute kernel. `noop` is the only thing that differs from the mesh sibling.

#### Kernels

| unique_id | source | core_ranges | CTAs (positional) | CTAs (named) | RTAs | CRTAs | defines | opt_level | config |
|---|---|---|---|---|---|---|---|---|---|
| reader | `device/kernels/dataflow/reader_fill_cache_interleaved.cpp` | `all_cores` | 0 `src0_cb_index`, 1 `Wt`, then `TensorAccessorArgs` (input) | none | per core: `[0]`=`Buffer*` input, `[1]`=`start_tile_id`, `[2]`=`num_rows`, `[3]`=`noop` | none | none | O2 | `ReaderConfigDescriptor{}` |
| writer | `device/kernels/dataflow/writer_fill_cache_interleaved.cpp` | `all_cores` | 0 `src0_cb_index`, 1 `page_table_cb_index`, 2 `num_heads`, 3 `num_blocks_of_work_per_head`, 4 `block_size_t`, 5 `Wt`, 6 `log2_page_table_stick_size_B`, 7 `page_table_stick_size_B`, 8 `use_batch_idx_tensor`, 9 `cb_batch_idx_id`, 10 `batch_idx_stick_size_B`, 11 `batch_idx_num_elements`, 12 `num_blocks_of_work_per_batch`, 13 `capacity_t`, 14 `use_valid_seq_len`, 15 `cb_valid_seq_len_id`, 16 `valid_seq_len_stick_size_B`, then `TensorAccessorArgs` ×4 (cache, page_table, batch_idx, valid_seq_len) | none | per core: `[0]`=`Buffer*` cache, `[1]`=`Buffer*` page_table, `[2]`=`start_row_num`, `[3]`=`num_rows`, `[4]`=`Buffer*` batch_idx **\| scalar `batch_idx_fallback`**, `[5]`=`noop`, `[6]`=`Buffer*` valid_seq_len \| `0` | none | none | O2 | `WriterConfigDescriptor{}` |

`grep -n opt_level` returns nothing; both kernels resolve to the DM default `O2`.

#### CBs

| index | total_size | core_ranges | data_format | page_size | tile (if set) |
|---|---|---|---|---|---|
| `c_0` `src0` | `num_input_tiles * single_tile_size` (`num_input_tiles = 2*Wt`) | `all_cores` | `cb_data_format` | `single_tile_size` | — |
| `c_1` `page_table` | `page_table_stick_size_B` | `all_cores` | `page_table_data_format` | `page_table_stick_size_B` | — |
| `c_2` `batch_idx` | `batch_idx_stick_size_B * batch_idx_num_elements` | `all_cores` | `batch_idx_data_format` | `batch_idx_stick_size_B` | — |
| `c_3` `valid_seq_len` | `valid_seq_len_stick_size_B` | `all_cores` | `UInt32` | `valid_seq_len_stick_size_B` | — |

`c_2` allocated only `if (use_batch_idx_tensor)` (`:199-211`); `c_3` only `if (use_valid_seq_len)`
(`:212-222`). No borrowed CB, no `GlobalCircularBuffer`.

Endpoint census (re-derived, agrees with the audit):

| CB | verdict | evidence |
|---|---|---|
| `c_0` | **1P+1C** | reader P (`reader_fill:38,46`) · writer C (`writer_fill:196-197,231-236,244`) |
| `c_1` | **self-loop** | writer only — `reserve_back(1)` @`writer_fill:148`, raw `get_write_ptr()` @`:149`, `noc.async_read` into it @`:210-216`; never pushed, never popped |
| `c_2` | **self-loop** | writer only — `reserve_back(1)` @`:102` + raw @`:103-113` |
| `c_3` | **self-loop** | writer only — `reserve_back(1)` @`:123` + raw @`:124-128` |

#### Semaphores

none.

#### Tensor accessors

| host site (file:line) | originating Tensor | RTA slot (host) |
|---|---|---|
| `:229` (reader CTAs) | `input_tensor` | reader RTA[0] (`:302`) |
| `:259` (writer CTAs) | `cache_tensor` | writer RTA[0] (`:311`) |
| `:260` (writer CTAs) | `page_table` | writer RTA[1] (`:312`) |
| `:261` (writer CTAs) | `batch_idx_tensor` (optional) | writer RTA[4] (`:315-319`) — **overloaded slot** |
| `:263` (writer CTAs) | `valid_seq_len_tensor` (optional) | writer RTA[6] (`:323-327`) — overloaded slot |

All five are 2-arg constructions, all **Case 1**.

#### Work split

- Driver: `split_work_to_cores(compute_with_storage_grid_size, num_blocks_of_work, row_major=true)`
  where `num_blocks_of_work = input_batch * num_heads * (input_seq_len / TILE_HEIGHT)`.
- `num_cores`, `all_cores`, `core_group_1`, `core_group_2`, `num_blocks_per_core_group_1`,
  `num_blocks_per_core_group_2` — but the two groups differ **only in a runtime arg**
  (`num_blocks_per_core`), never in a CTA, so legacy emits **one** `KernelDescriptor` per role, not
  one per group. There is no per-group CTA to preserve.
- The per-core list is `grid_to_cores(num_cores, num_cores_x, num_cores_y, row_major)` — note this
  can be **longer** than `all_cores` (`num_cores` counts only working cores, and cores past
  `g1+g2` get `num_blocks_per_core = 0`).

---

### Shared kernels

All 11 kernel sources live in `device/kernels/` in this op directory and no other op binds them.
`grep -rl <basename> ttnn/cpp/ttnn/operations/` produces hits in
`ttnn/cpp/ttnn/operations/kv_cache/device/kernels/`, but those are **separate private copies** bound
through `kv_cache`'s own paths — not consumers of these files. Confirmed by path, per the brief.

They are nonetheless shared in the sense that matters here: the **intra-op** shape. Each of the five
kernels below is bound by *both* members of a factory pair, and only the single-device member converts
in this change.

| kernel (all under `device/kernels/`) | bound by | `_metal2` fork already beside it? | rung taken |
|---|---|---|---|
| `dataflow/reader_update_cache_interleaved_start_id.cpp` | `PagedUpdateCacheProgramFactory`, `PagedUpdateCacheMeshWorkloadFactory` | no | **2 — create the fork** |
| `dataflow/writer_update_cache_interleaved_start_id.cpp` | same pair | no | **2 — create the fork** |
| `compute/update_cache.cpp` | same pair | no | **2 — create the fork** |
| `dataflow/reader_fill_cache_interleaved.cpp` | `PagedFillCacheProgramFactory`, `PagedFillCacheMeshWorkloadFactory` | no | **2 — create the fork** |
| `dataflow/writer_fill_cache_interleaved.cpp` | same pair | no | **2 — create the fork** |

Remaining consumers after this pass: the four `*MeshWorkloadFactory` factories (blocked — see
[Deferred / Flagged](#deferred--flagged)). Recorded in `METAL2_PORT_REPORT.md` →
*Open items for downstream*.

The six fused kernels are untouched by this pass.

### Flags

- **No unreferenced kernel files.** All 11 sources under `device/kernels/` are bound by a factory.
- The dead CTAs the audit catalogued (`log_base_2_of_page_size`, `log2_page_table_stick_size`,
  `max_blocks_per_seq`) are **carried through as named args**, not removed — dropping one would
  change the arg schema, which is a functional change the port is not entitled to make.
- `paged_fill_cache_program_factory.cpp:116` carries a `TT_FATAL` inside the factory body; it is
  preserved verbatim in the ported body (TT_FATAL census below).

---

## TTNN ProgramFactory

- **Concept (inherited from audit)**: `CustomProgramSpecFactoryConcept` — every ported-from factory
  declares an `override_runtime_arguments`, so the port *translates* it into a `ProgramRunArgs`
  producer rather than deleting it.
- **Custom `compute_program_hash`**: present on all three DeviceOperations (file:line above) —
  **left intact**. Not touched by this port.
- **Implementation notes**:
  - The two ported factories keep their existing structs and their existing `program_factory_t`
    membership. Their `create_descriptor` is replaced by `create_program_artifacts`, and their
    `void override_runtime_arguments(Program&, …)` by
    `ProgramRunArgs override_runtime_arguments(attrs, tensor_args, tensor_return_value, coord)`.
  - The legacy descriptor body each ported factory used to expose is **not deleted** — it moves into
    an anonymous-namespace helper in the same `.cpp`, because the blocked `*MeshWorkloadFactory`
    sibling still needs it. `fill_cache` already had exactly this shape
    (`build_paged_fill_cache_descriptor`); `update_cache` acquires it.
  - Likewise the legacy `Program&`-mutating patch moves into an anonymous-namespace helper the mesh
    factory's `override_runtime_arguments` calls.
  - The `program_factory_t` variants end up **mixed-concept**: one alternative on
    `CustomProgramSpecFactoryConcept`, one on `ProgramDescriptorFactoryConcept`. `AllFactoriesValid`
    permits this (each alternative satisfies exactly one concept) and the framework dispatches
    per-factory at runtime.
  - **No pybind change.** `paged_cache_nanobind.cpp` binds only the three public entry points via
    `ttnn::bind_function`; no `create_descriptor` is pybound, so nothing is removed and the port makes
    no user-visible API change.

---

## Planned Spec Shape

### Variant: `PagedUpdateCacheProgramFactory`

- **KernelSpecs** (3, 1:1 with legacy — no work-split multiplicity):
  - `READER` — `.../reader_update_cache_interleaved_start_id_metal2.cpp`,
    `hw_config = create_reader_datamovement_config(arch)` (legacy `ReaderConfigDescriptor{}` resolves
    to the reader default triple `RISCV_1 / NOC_0 / DM_DEDICATED_NOC`), `compiler_options` left at
    the default `O2`.
  - `WRITER` — `.../writer_update_cache_interleaved_start_id_metal2.cpp`,
    `hw_config = create_writer_datamovement_config(arch)`, `opt_level` default `O2`.
  - `COMPUTE` — `.../compute/update_cache_metal2.cpp`,
    `hw_config = ComputeGen1Config{.enable_32_bit_dest = fp32_dest_acc_en, .unpack_modes = …}`,
    **`compiler_options.opt_level = O3` set explicitly** (legacy `ComputeConfigDescriptor` defaults to
    O3; Metal 2.0 `CompilerOptions` defaults to O2).
- **DataflowBufferSpecs** (8, one per legacy `buffer_index`):

  | name | legacy | entry_size | num_entries | data_format | notes |
  |---|---|---|---|---|---|
  | `CACHE` | `c_0` | `cache_single_tile_size` | `num_cache_tiles` | `cache_cb_data_format` | |
  | `INPUT` | `c_1` | `input_single_tile_size` | `num_input_tiles` | `input_cb_data_format` | `borrowed_from = INPUT_TENSOR` |
  | `UNTILIZED_CACHE` | `c_24` | `interm_single_tile_size` | `num_interm_tiles` | `interm_cb_data_format` | `alias_with = {UNTILIZED_CACHE2}` |
  | `UNTILIZED_CACHE2` | `c_25` | `interm_single_tile_size` | `num_interm_tiles` | `interm_cb_data_format` | `alias_with = {UNTILIZED_CACHE}` |
  | `UNTILIZED_INPUT` | `c_26` | `interm_single_tile_size` | `num_interm_tiles` | `interm_cb_data_format` | |
  | `OUTPUT` | `c_16` | `cache_single_tile_size` | `num_output_tiles` | `cache_cb_data_format` | |
  | `INDEX` | `c_2` | `index_tensor_tile_size` | 1 | `index_data_format` | conditional on `use_index_tensor` |
  | `PAGE_TABLE` | `c_3` | `page_table_stick_size` | 1 | `page_table_data_format` | conditional on `is_paged_cache` |

  `tile_format_metadata` is left unset on all eight — no legacy `CBFormatDescriptor` set `.tile`.
- **SemaphoreSpecs** (1): `IN0_SEQUENTIAL_MODE`, `target_nodes = all_cores`. (Initial value 0 is the
  `SemaphoreSpec` default; the deprecated `advanced_options.initial_value` is not used.)
- **TensorParameters** (4, one per distinct originating tensor):
  `CACHE_TENSOR` (bound by READER + WRITER — two `TensorBinding`s, one `TensorParameter`),
  `INPUT_TENSOR` (no kernel binding — it exists solely as the `borrowed_from` target of `INPUT`),
  `UPDATE_IDXS` (conditional, READER only), `PAGE_TABLE_TENSOR` (conditional, READER only).
- **WorkUnitSpecs** (1): `{READER, WRITER, COMPUTE}` over `all_cores`.
- **Op-owned tensors**: none.

DFB endpoint roles:

| DFB | PRODUCER | CONSUMER |
|---|---|---|
| `CACHE` | READER | COMPUTE |
| `INPUT` | READER | COMPUTE |
| `UNTILIZED_CACHE` | COMPUTE | WRITER |
| `UNTILIZED_CACHE2` | WRITER | COMPUTE |
| `UNTILIZED_INPUT` | COMPUTE | WRITER |
| `OUTPUT` | COMPUTE | WRITER |
| `INDEX` | READER | WRITER |
| `PAGE_TABLE` | READER | WRITER |

`unpack_modes`: `enable_32_bit_dest` is `fp32_dest_acc_en`. When it is set, every DFB the compute
kernel **consumes** whose `data_format_metadata` is `Float32` needs an explicit entry, and legacy's
`unpack_to_dest_mode` was empty (all `Default`) — so each such entry is
`UnpackMode::UnpackToSrc`. The candidate consumed DFBs are `CACHE`, `INPUT` and `UNTILIZED_CACHE2`;
the spec computes the set from the resolved data formats rather than hardcoding it.

### Variant: `PagedFillCacheProgramFactory`

- **KernelSpecs** (2, 1:1 with legacy):
  - `READER` — `.../reader_fill_cache_interleaved_metal2.cpp`,
    `create_reader_datamovement_config(arch)`, `opt_level` default `O2`.
  - `WRITER` — `.../writer_fill_cache_interleaved_metal2.cpp`,
    `create_writer_datamovement_config(arch)`, `opt_level` default `O2`.
  - No compute kernel, so no `ComputeHardwareConfig` and no `unpack_modes`.
- **DataflowBufferSpecs** (4):

  | name | legacy | entry_size | num_entries | data_format | notes |
  |---|---|---|---|---|---|
  | `INPUT` | `c_0` | `single_tile_size` | `num_input_tiles` | `cb_data_format` | |
  | `PAGE_TABLE` | `c_1` | `page_table_stick_size_B` | 1 | `page_table_data_format` | **self-loop** |
  | `BATCH_IDX` | `c_2` | `batch_idx_stick_size_B` | `batch_idx_num_elements` | `batch_idx_data_format` | **self-loop**, conditional |
  | `VALID_SEQ_LEN` | `c_3` | `valid_seq_len_stick_size_B` | 1 | `UInt32` | **self-loop**, conditional |

- **SemaphoreSpecs**: none.
- **TensorParameters** (5): `INPUT_TENSOR` (READER), `CACHE_TENSOR` (WRITER), `PAGE_TABLE_TENSOR`
  (WRITER), `BATCH_IDX_TENSOR` (conditional, WRITER), `VALID_SEQ_LEN_TENSOR` (conditional, WRITER).
- **WorkUnitSpecs** (1): `{READER, WRITER}` over `all_cores`.
- **Op-owned tensors**: none.

DFB endpoint roles:

| DFB | PRODUCER | CONSUMER |
|---|---|---|
| `INPUT` | READER | WRITER |
| `PAGE_TABLE` | WRITER | WRITER (**self-loop**) |
| `BATCH_IDX` | WRITER | WRITER (**self-loop**) |
| `VALID_SEQ_LEN` | WRITER | WRITER (**self-loop**) |

---

## Preserved Multiplicity

**none — no work-split multiplicity in legacy.** Neither ported factory pushes the same
`kernel_source` into two `KernelDescriptor`s. `fill_cache` *does* split work across two core groups,
but the split reaches the kernel only through a per-core **runtime** arg (`num_blocks_per_core`), never
through a per-group CTA, so one `KernelSpec` per role reproduces it exactly. Demoting nothing and
promoting nothing: the legacy CTA/RTA split is preserved as-is.

---

## Dropped Plumbing

### `PagedUpdateCacheProgramFactory`

| legacy location (file:line) | legacy form | Metal 2.0 replacement |
|---|---|---|
| reader RTA[0] (`:406`), writer RTA[0] (`:426`); patched `:504`, `:509` | `Buffer*` cache (`dst_buffer`) | `TensorBinding{CACHE_TENSOR, "cache"}` on both kernels |
| reader RTA[2] (`:409`); patched `:505` | `Buffer*` update_idxs, or literal `0` | conditional `TensorBinding{UPDATE_IDXS, "index"}`; the literal-`0` alternative has **no** scalar counterpart (the kernel never reads it — access is behind the gate) |
| reader RTA[4] (`:415`); patched `:506` | `Buffer*` page_table, or literal `0` | conditional `TensorBinding{PAGE_TABLE_TENSOR, "page_table"}`; same — no scalar counterpart |
| CB `c_1` `.buffer = in1_buffer` (`:208`); re-pointed `:518` via `UpdateDynamicCircularBufferAddress` | globally-allocated CB backed by the input shard | `DataflowBufferSpec::borrowed_from = INPUT_TENSOR`, refreshed by the `TensorArgument` for `INPUT_TENSOR` |
| reader CTA 0/1/3/15 (`:286,287,289,303`) | `src0_cb_index`, `src1_cb_index`, `cb_index_id`, `cb_pagetable_id` | `DFBBinding`s → `dfb::cache`, `dfb::input`, `dfb::index`, `dfb::page_table` |
| writer CTA 0/1/2/3/5/14 (`:314-317,319,330`) | `output_cb_index`, `intermed0/1/2_cb_index`, `cb_index_id`, `cb_pagetable_id` | `DFBBinding`s → `dfb::output`, `dfb::untilized_cache`, `dfb::untilized_cache2`, `dfb::untilized_input`, `dfb::index`, `dfb::page_table` |
| compute CTA 0..5 (`:338-343`) | six CB indices | `DFBBinding`s → `dfb::cache`, `dfb::input`, `dfb::untilized_cache`, `dfb::untilized_cache2`, `dfb::untilized_input`, `dfb::output` |
| reader CTA 17 (`:305`), writer CTA 16 (`:332`) | `in0_sequential_mode_semaphore_id` | `SemaphoreBinding{IN0_SEQUENTIAL_MODE, "receiver"}` → `sem::receiver` |
| reader CTAs from `:308-311`, writer CTA from `:335` | `TensorAccessorArgs(buffer).append_to(cta)` ×4, with kernel-side `TensorAccessorArgs<19>()` / `next_compile_time_args_offset()` chain (`reader:48-50`, `writer:49`) | the binding mechanism end-to-end; kernel writes `TensorAccessor(tensor::cache)` etc. |
| reader CTA 2 (`:289`), writer CTA 4 (`:319`) | `use_index_tensor` gating `if constexpr` blocks that name a conditionally-bound DFB / tensor | promoted to `compiler_options.defines["USE_INDEX_TENSOR"]` + kernel-side `#ifdef` |
| reader CTA 8 (`:296`), writer CTA 9 (`:325`) | `is_paged_cache`, same shape | promoted to `compiler_options.defines["IS_PAGED_CACHE"]` + kernel-side `#ifdef` |
| every remaining positional CTA | positional `compile_time_args` vector | named `compile_time_args = {{name, value}, …}` |

No page-size third-argument CTA/RTA exists in this factory (all accessors are 2-arg), and no
semaphore-ID RTA (the semaphore id travelled as a CTA, replaced by the binding above).

### `PagedFillCacheProgramFactory`

| legacy location (file:line) | legacy form | Metal 2.0 replacement |
|---|---|---|
| reader RTA[0] (`:302`); patched `:407` | `Buffer*` input | `TensorBinding{INPUT_TENSOR, "input"}` |
| writer RTA[0] (`:311`); patched `:411` | `Buffer*` cache | `TensorBinding{CACHE_TENSOR, "cache"}` |
| writer RTA[1] (`:312`); patched `:412` | `Buffer*` page_table | `TensorBinding{PAGE_TABLE_TENSOR, "page_table"}` |
| writer RTA[4] (`:315-319`); patched `:413` | `Buffer*` batch_idx **or** the *meaningful* scalar `batch_idx_fallback` | **splits into two channels**: conditional `TensorBinding{BATCH_IDX_TENSOR, "batch_idx"}` **and** a named RTA `batch_idx_fallback` declared only on the `!use_batch_idx_tensor` path, so exactly one of the two channels exists in any given config — matching the single legacy slot. `batch_idx_fallback` is hash-excluded, so the override re-patches it on every hit, exactly as legacy did |
| writer RTA[6] (`:323-327`); patched `:415` | `Buffer*` valid_seq_len, or literal `0` | conditional `TensorBinding{VALID_SEQ_LEN_TENSOR, "valid_seq_len"}`; the literal-`0` alternative is never read |
| reader CTA 0 (`:228`), writer CTA 0/1/9/15 (`:237,238,248,256`) | `src0_cb_index`, `page_table_cb_index`, `cb_batch_idx_id`, `cb_valid_seq_len_id` | `DFBBinding`s → `dfb::input`, `dfb::page_table`, `dfb::batch_idx`, `dfb::valid_seq_len` |
| reader CTA from `:229`, writer CTAs from `:259-264` | `TensorAccessorArgs(...).append_to(cta)` ×5 with the kernel-side chain (`writer_fill:84-88`) | `TensorAccessor(tensor::name)` |
| writer CTA 8 (`:247`) | `use_batch_idx_tensor` gating blocks that name a conditional DFB + tensor | `compiler_options.defines["USE_BATCH_IDX_TENSOR"]` + `#ifdef` |
| writer CTA 14 (`:255`) | `use_valid_seq_len`, same shape | `compiler_options.defines["USE_VALID_SEQ_LEN"]` + `#ifdef` |
| every remaining positional CTA | positional vector | named `compile_time_args` |

`noop` stays a **named runtime arg** on both kernels — it is a per-dispatch value the override
re-patches, not plumbing the binding model replaces.

---

## Applied Patterns

- **Aliased DFBs** — `UNTILIZED_CACHE` (`c_24`) + `UNTILIZED_CACHE2` (`c_25`) in
  `update_cache`: one legacy `CBDescriptor` with two `CBFormatDescriptor`s becomes two
  `DataflowBufferSpec`s with mutual `advanced_options.alias_with`. The aliasing **is** the algorithm
  (compute publishes an untilized block through index 0; the writer NoC-writes the new row into that
  same L1 region in place and republishes it through index 1 for re-tilization), so the two must not
  be split into independent DFBs.
- **Conditional / optional DFB bindings** — `INDEX` + `PAGE_TABLE` in `update_cache`,
  `BATCH_IDX` + `VALID_SEQ_LEN` in `fill_cache`. Each was already conditionally *allocated* host-side;
  the port additionally makes the **binding** conditional and promotes the guarding CTA to a
  `compiler_options.defines` flag, because a Metal 2.0 binding is not the no-op an unused
  `CircularBuffer(id)` was. This also covers the conditional **tensor** bindings
  (`tensor::index`, `tensor::page_table`, `tensor::batch_idx`, `tensor::valid_seq_len`), where the
  `#ifdef` gate is mandatory rather than merely preferred.
- **Sync-free and single-ended CBs → self-loop DFB** — `fill_cache`'s `PAGE_TABLE`, `BATCH_IDX` and
  `VALID_SEQ_LEN`: each is touched by the writer alone (`reserve_back(1)` + raw pointer writes, never
  pushed or popped), so the writer is bound as both PRODUCER and CONSUMER. Legal on Gen1; a DM
  self-loop is Quasar-uplift's concern, not a Gen1 blocker.
- **Pass DFB handles directly to LLKs and kernel-lib helpers** — `compute_kernel_hw_startup(...)` and
  `compute_kernel_lib::untilize<Wt, in, out, …>` / `tilize<…>` take `uint32_t` CB ids (as NTTPs in the
  helper case). `dfb::name` is passed directly; `DFBBindingToken`'s `constexpr operator uint32_t()`
  bridges it in both value and template-parameter position. No `.id` extraction, no temp wrappers.
- **Caution: Porting a shared kernel — rung 2 (create the fork), intra-op shape** — five `_metal2`
  forks beside their originals, pointer comment added to each original, originals otherwise untouched.

---

## Deferred / Flagged

### New finding the audit missed — per-coord variation blocks the four `*MeshWorkloadFactory` factories

`ttnn_factory.md` → *Feasibility gate* lists **"Multi-program / per-coord variation"** as a hard
BLOCKED case for both Metal 2.0 factory concepts: *"The single-program adapter stamps one spec
everywhere."* All four `*MeshWorkloadFactory` factories are exactly that case, and the audit cleared
them anyway (its *Watch for* entry treats the two mesh-filtering idioms as behaviour to preserve
rather than as a gate).

Concretely, `create_program_artifacts(attrs, tensor_args, tensor_return_value)` takes **no**
`mesh_dispatch_coordinate`, and `ProgramSpecMeshWorkloadFactoryAdapter::create_mesh_workload`
(`ttnn/api/ttnn/mesh_device_operation_adapter.hpp:912-921`) emplaces *the same* `artifacts.spec` for
every range in `tensor_coords` and applies *the same* `artifacts.run_params` to every resulting
program via `SetProgramRunArgs`. There is no per-coordinate hook on the cache-miss path.

Both legacy mesh idioms need one:

- **Empty-descriptor idiom** (`update_cache` `:448-453`, tiled fused `:544-549`, RM fused `:547-552`):
  a coordinate outside `operation_attributes.mesh_coords` gets an **empty `ProgramDescriptor`**, and
  the descriptor adapter *skips adding a program for that coordinate entirely*
  (`mesh_device_operation_adapter.hpp:588-592`). A Metal 2.0 spec factory cannot express "no program
  here."
- **`noop`-RTA idiom** (`fill_cache` `:33-40`, `:348-359`): the spec is identical across coordinates,
  but the initial value of the `noop` runtime arg is **not** — and the cache-miss dispatch executes
  with whatever `SetProgramRunArgs` wrote, which is one value for the whole mesh. The cache-*hit* path
  is fine (`override_runtime_arguments` receives the coordinate), so the gap here is narrower —
  per-coord run args on the **miss** — but the first dispatch would still fill the cache on a
  coordinate the caller excluded.

Neither is porter-resolvable from inside the op directory, and neither may be normalised away (the
brief is explicit: *"neither is the port's to normalise … Preserve both behaviours as they are"*).
`ttnn/api/ttnn/metal_v2_artifacts.hpp:20-22` names the intended vehicle — *"A future
`MeshWorkloadSpecFactoryConcept` will return a different (multi-program) artifact type for ops whose
programs vary across the mesh."*

**Consequence for this pass:** the four mesh factories stay on `ProgramDescriptorFactoryConcept`, and
because each shares its kernel sources with the single-device sibling that *is* converting, the five
kernels those two pairs bind are forked per *Caution: Porting a shared kernel* rung 2. Recorded as a
Handoff point in `METAL2_PORT_REPORT.md`.

### Carried forward — audit Question #1 blocks the two fused single-device factories

The audit's own open design question ("How should the fused factories' runtime-selected input DFB be
expressed in a `ProgramSpec`?") is unresolved, and the brief instructs: *"Get an answer before you
write the fused specs."* Reading `dataflow_buffer_spec.hpp` sharpens the question rather than
answering it — **PLACEMENT is derived**: *"the DFB's effective node set is the union of its bound
kernels' `WorkUnitSpec` `target_nodes`"*. So binding both `src1` and `src2` on a `KernelSpec` that
spans `all_cores_bb` places **both** DFBs across the whole bounding box, whereas legacy allocated
`c_1` only over `input1_cores` and `c_2` only over `input2_cores`. Since both are
**`borrowed_from`** an L1-sharded input tensor that has no shard on the other core set, "bind both and
rely on per-node existence" is not obviously sound, and the alternative (splitting into per-core-set
`KernelSpec`s) promotes the host-computed `is_input1` **runtime** arg into a compile-time define —
a schema change the port is not entitled to make unilaterally.

The two fused factories are therefore left for a later pass. They are additionally subject to the
mesh blocker above for their own `*MeshWorkloadFactory` siblings.

### Other flags

- **No `TensorParameter` relaxation is declared anywhere.** The audit reports `none` on all rows; the
  port keeps strict matching.
- **No varargs.** Every ported kernel reaches each argument as a distinct field a fixed number of
  times, so every argument is named. Neither `num_runtime_varargs` nor `compile_time_varargs` is used.
- **No Case 2 tensor binding**, so no `get_bank_base_address` bridge and no compute-kernel raw-pointer
  block.
- **No `GlobalCircularBuffer`** anywhere in the op.
